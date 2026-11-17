//! The sequences section.

use bun_zstd_decoder::format::{
    DEFAULT_LITERAL_LENGTHS, DEFAULT_MATCH_LENGTHS, DEFAULT_OFFSETS, Distribution, Encoding,
    LITERAL_LENGTH_BITS, LITERAL_LENGTH_TABLE_LOG_MAX, LONG_SEQUENCE_COUNT, MATCH_LENGTH_BITS,
    MATCH_LENGTH_TABLE_LOG_MAX, MAX_OFFSET_CODE, OFFSET_TABLE_LOG_MAX,
};
use bun_zstd_decoder::{Error, Result, with_bmi2};

use super::bit_writer::BitWriter;
use super::fse::{self, Normalized, SYMBOLS, Table};
use super::{Reuse, Slots};
use crate::array::At;
use crate::bytes::put;
use crate::parameters::Strategy;
use crate::sequences::{CodeCounts, Sequence};

type Histogram = super::histogram::Histogram<SYMBOLS>;

/// `-log2(index / 256) * 256`
#[rustfmt::skip]
static INVERSE_PROBABILITY_LOG: [u16; 256] = [
    0, 2048, 1792, 1642, 1536, 1453, 1386, 1329, 1280, 1236, 1197, 1162, 1130, 1100, 1073, 1047,
    1024, 1001, 980, 960, 941, 923, 906, 889, 874, 859, 844, 830, 817, 804, 791, 779,
    768, 756, 745, 734, 724, 714, 704, 694, 685, 676, 667, 658, 650, 642, 633, 626,
    618, 610, 603, 595, 588, 581, 574, 567, 561, 554, 548, 542, 535, 529, 523, 517,
    512, 506, 500, 495, 489, 484, 478, 473, 468, 463, 458, 453, 448, 443, 438, 434,
    429, 424, 420, 415, 411, 407, 402, 398, 394, 390, 386, 382, 377, 373, 370, 366,
    362, 358, 354, 350, 347, 343, 339, 336, 332, 329, 325, 322, 318, 315, 311, 308,
    305, 302, 298, 295, 292, 289, 286, 282, 279, 276, 273, 270, 267, 264, 261, 258,
    256, 253, 250, 247, 244, 241, 239, 236, 233, 230, 228, 225, 222, 220, 217, 215,
    212, 209, 207, 204, 202, 199, 197, 194, 192, 190, 187, 185, 182, 180, 178, 175,
    173, 171, 168, 166, 164, 162, 159, 157, 155, 153, 151, 149, 146, 144, 142, 140,
    138, 136, 134, 132, 130, 128, 126, 123, 121, 119, 117, 115, 114, 112, 110, 108,
    106, 104, 102, 100, 98, 96, 94, 93, 91, 89, 87, 85, 83, 82, 80, 78,
    76, 74, 73, 71, 69, 67, 66, 64, 62, 61, 59, 57, 55, 54, 52, 50,
    49, 47, 46, 44, 42, 41, 39, 37, 36, 34, 33, 31, 30, 28, 26, 25,
    23, 22, 20, 19, 17, 16, 14, 13, 11, 10, 8, 7, 5, 4, 2, 1,
];

#[rustfmt::skip]
const OFFSET_BITS: [u8; MAX_OFFSET_CODE + 1] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
];

const fn padded(bits: &[u8]) -> [u8; SYMBOLS] {
    let mut padded = [0; SYMBOLS];
    padded.split_at_mut(bits.len()).0.copy_from_slice(bits);
    padded
}

/// In the order of their tables in a block.
#[derive(Clone, Copy)]
pub(crate) enum CodeKind {
    LiteralLength,
    Offset,
    MatchLength,
}

struct Alphabet {
    table_log_max: u32,
    predefined: Normalized,
    predefined_table: Table,
    extra_bits: [u8; SYMBOLS],
}

impl Alphabet {
    const fn new(table_log_max: u32, predefined: &Distribution<'_>, extra_bits: &[u8]) -> Self {
        let predefined = Normalized::predefined(predefined);
        Self {
            table_log_max,
            predefined,
            predefined_table: Table::new(&predefined),
            extra_bits: padded(extra_bits),
        }
    }

    /// In bits.
    fn predefined_cost(&self, counts: &[u32]) -> Option<usize> {
        if counts.len() > self.predefined.max_symbol + 1 {
            return None;
        }
        let shift = 8 - self.predefined.table_log;
        let pairs = counts.iter().zip(self.predefined.counts);
        let cost = pairs.map(|(&count, share)| {
            let share = usize::from(share.unsigned_abs()) << shift;
            count * u32::from(*INVERSE_PROBABILITY_LOG.at(share))
        });
        Some((cost.sum::<u32>() >> 8) as usize)
    }

    /// In bits, with the description of the table.
    fn compressed_cost(&self, counts: &[u32], total: usize) -> Result<usize> {
        let log = fse::table_log_for(self.table_log_max, total, counts.len().saturating_sub(1), 2);
        let normalized = Normalized::new(counts, total, log, allows_less_than_1(total))?;
        let description_len = normalized.write(&mut [0; 128])?;
        let cost = counts.iter().map(|&count| {
            let share = (256 * count as usize / total).max(1);
            count * u32::from(*INVERSE_PROBABILITY_LOG.at(share))
        });
        Ok(description_len * 8 + (cost.sum::<u32>() >> 8) as usize)
    }
}

static ALPHABETS: [Alphabet; 3] = [
    Alphabet::new(
        LITERAL_LENGTH_TABLE_LOG_MAX,
        &DEFAULT_LITERAL_LENGTHS,
        &LITERAL_LENGTH_BITS,
    ),
    Alphabet::new(OFFSET_TABLE_LOG_MAX, &DEFAULT_OFFSETS, &OFFSET_BITS),
    Alphabet::new(
        MATCH_LENGTH_TABLE_LOG_MAX,
        &DEFAULT_MATCH_LENGTHS,
        &MATCH_LENGTH_BITS,
    ),
];

/// Tables without such counts are faster to build, which small blocks notice.
fn allows_less_than_1(sequence_count: usize) -> bool {
    sequence_count >= 2048
}

/// Also checks that there is room for what has to follow.
pub(super) fn write_count(count: usize, out: &mut [u8]) -> Result<usize> {
    let [first, second, third, _] = out.first_chunk_mut().ok_or(Error::DstSizeTooSmall)?;
    if count < 128 {
        *first = count as u8;
        Ok(1)
    } else if count < LONG_SEQUENCE_COUNT {
        [*first, *second] = (count as u16 + 0x8000).to_be_bytes();
        Ok(2)
    } else {
        *first = 0xFF;
        [*second, *third] = ((count - LONG_SEQUENCE_COUNT) as u16).to_le_bytes();
        Ok(3)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct CodesChoice([Encoding; 3]);

impl CodesChoice {
    pub(crate) const PREDEFINED: Self = Self([Encoding::Basic; 3]);

    pub(super) fn modes(self) -> u8 {
        let [literal_lengths, offsets, match_lengths] = self.0.map(|encoding| encoding as u8);
        literal_lengths << 6 | offsets << 4 | match_lengths << 2
    }
}

pub(super) struct Chosen {
    pub(super) choice: CodesChoice,
    /// Of the descriptions of the tables.
    pub(super) len: usize,
    /// Of the description of the last compressed table.
    last_compressed_len: usize,
}

impl Chosen {
    /// https://github.com/facebook/zstd/pull/1146
    pub(super) fn trips_old_decoders(&self, stream_len: usize) -> bool {
        self.last_compressed_len != 0 && self.last_compressed_len + stream_len < 4
    }
}

/// How sequences are going to be encoded.
pub(crate) struct CodesPlan {
    chosen: Chosen,
    descriptions: [u8; 133],
}

impl CodesPlan {
    pub(crate) fn choice(&self) -> CodesChoice {
        self.chosen.choice
    }

    pub(crate) fn has_descriptions(&self) -> bool {
        self.chosen.len != 0
    }
}

fn select(
    alphabet: &Alphabet,
    slots: &Slots<Table>,
    histogram: &Histogram,
    total: usize,
    strategy: Strategy,
) -> Result<Encoding> {
    let counts = histogram.used();
    let allows_predefined = histogram.max_symbol <= alphabet.predefined.max_symbol;
    if histogram.largest_count == total {
        return Ok(if allows_predefined && total <= 2 {
            Encoding::Basic
        } else {
            Encoding::Rle
        });
    }
    if strategy < Strategy::Lazy {
        if allows_predefined {
            let log = alphabet.predefined.table_log;
            if slots.reuse == Reuse::Always && total < 1000 {
                return Ok(Encoding::Repeat);
            }
            let min_total = ((10 - strategy as usize) << log) >> 3;
            if total < min_total || histogram.largest_count < total >> (log - 1) {
                return Ok(Encoding::Basic);
            }
        }
        return Ok(Encoding::Compressed);
    }
    let predefined = alphabet.predefined_cost(counts).unwrap_or(usize::MAX);
    let repeated = match slots.reuse {
        Reuse::Never => None,
        _ => slots.current().cost(counts),
    };
    let repeated = repeated.unwrap_or(usize::MAX);
    let compressed = alphabet.compressed_cost(counts, total)?;
    Ok(if predefined <= repeated && predefined <= compressed {
        Encoding::Basic
    } else if repeated <= compressed {
        Encoding::Repeat
    } else {
        Encoding::Compressed
    })
}

pub(crate) struct CodeTables([Slots<Table>; 3]);

impl CodeTables {
    pub(super) const EMPTY: Self = Self([Slots::new(Table::EMPTY); 3]);

    pub(crate) fn get(&self, kind: CodeKind) -> &Slots<Table> {
        self.0.at(kind as usize)
    }

    pub(crate) fn get_mut(&mut self, kind: CodeKind) -> &mut Slots<Table> {
        self.0.at_mut(kind as usize)
    }

    pub(super) fn copy_from(&mut self, other: &Self) {
        for (to, from) in self.0.iter_mut().zip(&other.0) {
            to.copy_from(from);
        }
    }

    pub(super) fn accept(&mut self, choice: CodesChoice) {
        for (slots, encoding) in self.0.iter_mut().zip(choice.0) {
            match encoding {
                Encoding::Basic | Encoding::Rle => slots.reuse = Reuse::Never,
                Encoding::Compressed => slots.accept_candidate(),
                Encoding::Repeat => {}
            }
        }
    }

    fn table(&self, kind: CodeKind, choice: CodesChoice) -> &Table {
        match choice.0.at(kind as usize) {
            Encoding::Basic => &ALPHABETS.at(kind as usize).predefined_table,
            Encoding::Rle | Encoding::Compressed => self.get(kind).candidate(),
            Encoding::Repeat => self.get(kind).current(),
        }
    }

    /// Builds the tables for `sequences`, of which there are some, and writes their descriptions.
    pub(super) fn choose(
        &mut self,
        sequences: &[Sequence],
        counts: &CodeCounts,
        strategy: Strategy,
        out: &mut [u8],
    ) -> Result<Chosen> {
        let mut chosen = Chosen {
            choice: CodesChoice::PREDEFINED,
            len: 0,
            last_compressed_len: 0,
        };
        let last_codes = sequences.last().ok_or(Error::Internal)?.codes();
        for (kind, last) in last_codes.into_iter().enumerate() {
            let (alphabet, slots) = (ALPHABETS.at(kind), self.0.at_mut(kind));
            let mut histogram = Histogram::new(*counts.at(kind));
            let encoding = select(alphabet, slots, &histogram, sequences.len(), strategy)?;
            *chosen.choice.0.at_mut(kind) = encoding;

            let out = out.get_mut(chosen.len..).ok_or(Error::Internal)?;
            let (_, candidate) = slots.current_and_candidate();
            match encoding {
                Encoding::Basic | Encoding::Repeat => {}
                Encoding::Rle => {
                    candidate.build_for_one(last);
                    *out.first_mut().ok_or(Error::DstSizeTooSmall)? = last;
                    chosen.len += 1;
                }
                Encoding::Compressed => {
                    let mut total = sequences.len();
                    let log =
                        fse::table_log_for(alphabet.table_log_max, total, histogram.max_symbol, 2);
                    // The first state stands for it.
                    let count = histogram.counts.at_mut(usize::from(last));
                    if *count > 1 {
                        *count -= 1;
                        total -= 1;
                    }
                    let normalized =
                        Normalized::new(histogram.used(), total, log, allows_less_than_1(total))?;
                    chosen.last_compressed_len = normalized.write(out)?;
                    chosen.len += chosen.last_compressed_len;
                    candidate.build(&normalized);
                }
            }
        }
        Ok(chosen)
    }

    pub(crate) fn plan(
        &mut self,
        sequences: &[Sequence],
        counts: &CodeCounts,
        strategy: Strategy,
    ) -> Result<CodesPlan> {
        let mut descriptions = [0; _];
        let chosen = if sequences.is_empty() {
            Chosen {
                choice: CodesChoice::PREDEFINED,
                len: 0,
                last_compressed_len: 0,
            }
        } else {
            self.choose(sequences, counts, strategy, &mut descriptions)?
        };
        Ok(CodesPlan {
            chosen,
            descriptions,
        })
    }

    /// Of the descriptions and the stream.
    pub(crate) fn estimated_len(&self, plan: &CodesPlan, counts: &CodeCounts) -> usize {
        use CodeKind::*;
        let kinds = [LiteralLength, Offset, MatchLength].into_iter().zip(counts);
        let lens = kinds.map(|(kind, counts)| {
            let alphabet = ALPHABETS.at(kind as usize);
            let histogram = Histogram::new(*counts);
            let bits = match plan.choice().0.at(kind as usize) {
                Encoding::Basic => alphabet.predefined_cost(histogram.used()),
                Encoding::Rle => Some(0),
                _ => self.table(kind, plan.choice()).cost(histogram.used()),
            };
            let Some(bits) = bits else {
                return counts.iter().sum::<u32>() as usize * 10;
            };
            let pairs = histogram.counts.iter().zip(alphabet.extra_bits);
            let extra_bits: u32 = pairs.map(|(&count, bits)| count * u32::from(bits)).sum();
            (bits + extra_bits as usize) / 8
        });
        plan.chosen.len + lens.sum::<usize>()
    }

    /// The section for a part of the sequences that `plan` is for, and whether later parts can repeat its tables. `None`
    /// if it cannot be compressed.
    pub(crate) fn write_part(
        &self,
        plan: &CodesPlan,
        sequences: &[Sequence],
        describes_tables: bool,
        out: &mut [u8],
    ) -> Result<Option<(usize, bool)>> {
        let count_len = write_count(sequences.len(), out)?;
        if sequences.is_empty() {
            return Ok(Some((count_len, false)));
        }
        let out = out.get_mut(count_len..).unwrap_or_default();
        let (modes, out) = out.split_first_mut().ok_or(Error::Internal)?;
        let (choice, descriptions) = if describes_tables {
            let descriptions = plan.descriptions.get(..plan.chosen.len);
            (plan.choice(), descriptions.unwrap_or_default())
        } else {
            (CodesChoice([Encoding::Repeat; 3]), &[][..])
        };
        *modes = choice.modes();
        put(out, descriptions).ok_or(Error::DstSizeTooSmall)?;
        let out = out.get_mut(descriptions.len()..).unwrap_or_default();
        let stream_len = self.encode(plan.choice(), sequences, out)?;

        let len = 1 + descriptions.len() + stream_len;
        // https://github.com/facebook/zstd/pull/1664
        if len < 4 || (describes_tables && plan.chosen.trips_old_decoders(stream_len)) {
            return Ok(None);
        }
        Ok(Some((count_len + len, true)))
    }

    /// Returns the length of the stream.
    pub(super) fn encode(
        &self,
        choice: CodesChoice,
        sequences: &[Sequence],
        out: &mut [u8],
    ) -> Result<usize> {
        use CodeKind::*;
        let tables = [LiteralLength, Offset, MatchLength].map(|kind| self.table(kind, choice));
        let len = with_bmi2(
            #[inline(always)]
            move || encode(tables, sequences, out),
        );
        len.ok_or(Error::DstSizeTooSmall)
    }
}

#[inline(always)]
fn encode(
    [literal_table, offset_table, match_table]: [&Table; 3],
    sequences: &[Sequence],
    out: &mut [u8],
) -> Option<usize> {
    const ROOM_AFTER_FLUSH: u32 = 64 - 8;
    const ROOM_AFTER_STATES: u32 = ROOM_AFTER_FLUSH
        - LITERAL_LENGTH_TABLE_LOG_MAX
        - OFFSET_TABLE_LOG_MAX
        - MATCH_LENGTH_TABLE_LOG_MAX;
    let mut writer = BitWriter::new(out)?;
    let add_extra_bits = |writer: &mut BitWriter<'_>, sequence: &Sequence| {
        let (mut bits, mut count) = sequence.extra_bits();
        if count > ROOM_AFTER_STATES {
            writer.flush();
            if count > ROOM_AFTER_FLUSH {
                core::hint::cold_path();
                writer.add(bits as u32, 32);
                writer.flush();
                (bits, count) = (bits >> 32, count - 32);
            }
        }
        writer.add_all(bits, count);
        writer.flush();
    };

    let (last, rest) = sequences.split_last()?;
    let [literal_code, offset_code, match_code] = last.codes();
    let mut match_state = match_table.first_state(match_code);
    let mut offset_state = offset_table.first_state(offset_code);
    let mut literal_state = literal_table.first_state(literal_code);
    add_extra_bits(&mut writer, last);

    for sequence in rest.iter().rev() {
        let [literal_code, offset_code, match_code] = sequence.codes();
        offset_table.encode(&mut writer, &mut offset_state, offset_code);
        match_table.encode(&mut writer, &mut match_state, match_code);
        literal_table.encode(&mut writer, &mut literal_state, literal_code);
        add_extra_bits(&mut writer, sequence);
    }

    match_table.finish(&mut writer, match_state);
    offset_table.finish(&mut writer, offset_state);
    literal_table.finish(&mut writer, literal_state);
    writer.finish()
}
