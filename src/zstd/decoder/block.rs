//! Compressed blocks, and what one leaves for the next.

use crate::array::{get_or_allocate, grow};
use crate::destination::Destination;
use crate::dictionary::{Dictionary, Entropy};
use crate::error::{Error, Result};
use crate::format::{CodeKind, Encoding, INITIAL_REPEAT_OFFSETS, LONG_SEQUENCE_COUNT};
use crate::huffman::HuffmanTable;
use crate::literals::{LITERALS_PADDING, LiteralsHeader};
use crate::sequences::{PREDEFINED_TABLES, SequenceTables, run_sequences};

/// Where the table is that a block can repeat.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum Source {
    #[default]
    None,
    Predefined,
    Dictionary,
    Own,
}

/// With the padding. They are not worth an allocation.
const FEW_LITERALS: usize = 256;

/// Asks for memory when a block takes it.
#[derive(Default)]
pub(crate) struct BlockDecoder {
    literals: Vec<u8>,
    huffman: HuffmanTable,
    huffman_source: Source,
    sequence_tables: Option<Box<SequenceTables>>,
    /// By [`CodeKind`].
    sequence_sources: [Source; 3],
    repeat_offsets: [u32; 3],
}

/// The number of sequences, and how many bytes it takes.
fn read_sequence_count(section: &[u8]) -> Result<(usize, usize)> {
    match *section {
        [count @ ..0x80, ..] => Ok((count.into(), 1)),
        [high @ 0x80..=0xFE, low, ..] => Ok((usize::from(high - 0x80) << 8 | usize::from(low), 2)),
        [0xFF, low, high, ..] => Ok((
            usize::from(u16::from_le_bytes([low, high])) + LONG_SEQUENCE_COUNT,
            3,
        )),
        _ => Err(Error::SrcSizeWrong),
    }
}

impl BlockDecoder {
    /// All blocks of the frame go with the same `dictionary`.
    pub(crate) fn start_frame(&mut self, dictionary: Option<&Dictionary<'_>>) {
        let entropy = dictionary.and_then(Dictionary::entropy);
        let source = entropy.map_or(Source::None, |_| Source::Dictionary);
        self.huffman_source = source;
        self.sequence_sources = [source; 3];
        self.repeat_offsets = entropy.map_or(INITIAL_REPEAT_OFFSETS, |it| it.repeat_offsets);
    }

    /// Returns how many bytes it wrote.
    pub(crate) fn decode(
        &mut self,
        block: &[u8],
        dictionary: Option<&Dictionary<'_>>,
        mut destination: Destination<'_, '_>,
    ) -> Result<usize> {
        let entropy = dictionary.and_then(Dictionary::entropy);
        let has_table = self.huffman_source != Source::None;
        let (len_max, room) = (destination.block_len_max(), destination.room());
        let header = LiteralsHeader::read(block, has_table, len_max, room)?;
        let section = block.get(header.section_len..).ok_or(Error::Internal)?;
        let has_only_literals = section == [0];

        let mut few_literals;
        let literals = match header.in_place(block) {
            Some(literals) => literals,
            None => {
                let start = header.read_tree(block, &mut self.huffman)?;
                if header.has_tree() {
                    self.huffman_source = Source::Own;
                }
                let table = match entropy {
                    Some(entropy) if self.huffman_source == Source::Dictionary => &entropy.huffman,
                    _ => &self.huffman,
                };
                let len = header.room_worth_clearing(table);
                if has_only_literals {
                    header.decode(
                        block,
                        start,
                        table,
                        destination.lend_room(header.count, len - header.count)?,
                    )?;
                    return destination.finish(header.count, &[]);
                }
                let out: &mut [u8] = if len <= FEW_LITERALS {
                    few_literals = [0; FEW_LITERALS];
                    &mut few_literals
                } else {
                    grow(&mut self.literals, 0, len)?;
                    &mut self.literals
                };
                header.decode(block, start, table, out)?;
                (out.get(..header.count + LITERALS_PADDING)).ok_or(Error::Internal)?
            }
        };

        let (count, count_len) = read_sequence_count(section)?;
        let descriptions = section.get(count_len..).ok_or(Error::Internal)?;
        let (mut len, mut used_literals) = (0, 0);
        if count != 0 {
            let (own, sources) = (&mut self.sequence_tables, &mut self.sequence_sources);
            let (tables, stream) = select_tables(descriptions, count, own, sources, entropy)?;
            if room == 0 {
                return Err(Error::DstSizeTooSmall);
            }
            let start = block.len() - stream.len();
            let offsets = &mut self.repeat_offsets;
            (len, used_literals) = destination.with_overshoot(|destination| {
                run_sequences(tables, block, start, count, literals, offsets, destination)
            })?;
        } else if !descriptions.is_empty() {
            return Err(Error::CorruptionDetected);
        }
        let last_literals = literals.get(used_literals..header.count);
        destination.finish(len, last_literals.ok_or(Error::Internal)?)
    }
}

/// For no more sequences, it is not worth copying tables to have them in one place.
const SCATTERED_TABLES_COUNT_MAX: usize = 10;

/// Returns where the table of each [`CodeKind`] is, and what follows their descriptions in `src`.
fn select_tables<'t, 's>(
    src: &'s [u8],
    count: usize,
    own: &'t mut Option<Box<SequenceTables>>,
    sources: &mut [Source; 3],
    entropy: Option<&'t Entropy>,
) -> Result<([&'t SequenceTables; 3], &'s [u8])> {
    let (&modes, mut src) = src.split_first().ok_or(Error::SrcSizeWrong)?;
    if modes & 3 != 0 {
        return Err(Error::CorruptionDetected);
    }
    let encodings = [6, 4, 2].map(|shift| Encoding::from_low_bits(modes >> shift));
    for (source, encoding) in sources.iter_mut().zip(encodings) {
        *source = match encoding {
            Encoding::Basic => Source::Predefined,
            Encoding::Repeat if *source == Source::None => return Err(Error::CorruptionDetected),
            Encoding::Repeat => *source,
            Encoding::Rle | Encoding::Compressed => Source::Own,
        };
    }
    let shared = |source| match source {
        Source::Predefined => Some(&PREDEFINED_TABLES),
        Source::Dictionary => entropy.map(|entropy| &*entropy.sequences),
        _ => None,
    };
    let [first, second, third] = *sources;
    if let (Some(a), Some(b), Some(c)) = (shared(first), shared(second), shared(third))
        && ((first == second && first == third) || count <= SCATTERED_TABLES_COUNT_MAX)
    {
        return Ok(([a, b, c], src));
    }

    let own = get_or_allocate(own, || SequenceTables::EMPTY)?;
    for ((kind, encoding), source) in CodeKind::ALL.into_iter().zip(encodings).zip(sources) {
        let len = match encoding {
            Encoding::Compressed => own.read(kind, src)?,
            Encoding::Rle => {
                own.set_rle(kind, *src.first().ok_or(Error::CorruptionDetected)?)?;
                1
            }
            _ => 0,
        };
        src = src.get(len..).ok_or(Error::Internal)?;
        if let Some(tables) = shared(*source) {
            own.copy_from(tables, kind);
        }
        *source = Source::Own;
    }
    Ok(([own; 3], src))
}
