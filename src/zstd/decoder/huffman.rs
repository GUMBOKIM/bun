//! Huffman coding: tree descriptions, decoding tables and the streams of literals.

use crate::array::{fill_up, wrapped, wrapped_mut};
use crate::bits::{BackwardBits, bits_below_mark, load_from};
use crate::error::{Error, Result};
use crate::format::HUFFMAN_TABLE_LOG_MAX;
use crate::fse::{States, read_normalized_counts};

const WEIGHT_TABLE_LOG_MAX: u32 = 6;
const WEIGHT_STATES_MAX: usize = 1 << WEIGHT_TABLE_LOG_MAX;
/// Stands for all that are too high.
const INVALID_WEIGHT: u8 = 15;

/// What a tree description says.
pub struct HuffmanWeights {
    /// By symbol. The first `symbol_count` count.
    pub weights: [u8; 256],
    /// By weight: how many symbols have it.
    pub rank_counts: [u32; 16],
    pub symbol_count: usize,
    /// The length of the longest code.
    pub table_log: u32,
    /// Of the description, in bytes.
    pub len: usize,
}

/// Reads the description at the start of `src`.
#[inline(always)]
pub fn read_huffman_weights(src: &[u8]) -> Result<HuffmanWeights> {
    let (&header, rest) = src.split_first().ok_or(Error::CorruptionDetected)?;
    let mut weights = [0; 256];
    let (listed, len) = match header.checked_sub(127) {
        Some(listed @ 1..) => {
            let len = usize::from(listed).div_ceil(2);
            let pairs = rest.get(..len).ok_or(Error::CorruptionDetected)?;
            for (to, pair) in weights.as_chunks_mut().0.iter_mut().zip(pairs) {
                *to = [pair >> 4, pair & 15];
            }
            (listed.into(), len)
        }
        _ => {
            let len = header.into();
            let compressed = rest.get(..len).ok_or(Error::CorruptionDetected)?;
            (decompress_weights(compressed, &mut weights)?, len)
        }
    };

    let mut rank_counts = [0; 16];
    for &weight in weights.iter().take(listed) {
        *wrapped_mut(&mut rank_counts, weight.into()) += 1;
    }
    let ranks = rank_counts.iter().skip(1).zip(0..);
    let total: u32 = ranks.map(|(count, shorter_by)| count << shorter_by).sum();
    let table_log = total.bit_width();
    if total == 0 || table_log > HUFFMAN_TABLE_LOG_MAX {
        return Err(Error::CorruptionDetected);
    }
    let missing = (1 << table_log) - total;
    if !missing.is_power_of_two() {
        return Err(Error::CorruptionDetected);
    }
    let last_weight = missing.bit_width();
    *wrapped_mut(&mut weights, listed) = last_weight as u8;
    *wrapped_mut(&mut rank_counts, last_weight as usize) += 1;
    if rank_counts[1] < 2 || rank_counts[1] % 2 != 0 {
        return Err(Error::CorruptionDetected);
    }
    Ok(HuffmanWeights {
        weights,
        rank_counts,
        symbol_count: listed + 1,
        table_log,
        len: len + 1,
    })
}

#[derive(Clone, Copy)]
struct WeightState {
    weight: u8,
    bits: u8,
    baseline: u8,
}

/// Returns how many weights there are: at most 255.
#[inline(always)]
fn decompress_weights(src: &[u8], weights: &mut [u8; 256]) -> Result<usize> {
    // These limits are what fits in the workspace of zstd's `HUF_readStats()`.
    let mut counts = [0; 92];
    let description = read_normalized_counts(src, &mut counts)?;
    let (log, max_symbol) = (description.table_log, description.max_symbol);
    if log > WEIGHT_TABLE_LOG_MAX || (log == WEIGHT_TABLE_LOG_MAX && max_symbol > 11) {
        return Err(Error::CorruptionDetected);
    }
    let mut states = States::new(counts.get(..=max_symbol).ok_or(Error::Internal)?, log);
    let mut table = [WeightState {
        weight: 0,
        bits: 0,
        baseline: 0,
    }; WEIGHT_STATES_MAX];
    for row in table.iter_mut().take(1 << log) {
        let state = states.next_state();
        *row = WeightState {
            weight: state.symbol.min(INVALID_WEIGHT),
            bits: state.bits as u8,
            baseline: state.baseline as u8,
        };
    }

    let mut bits = BackwardBits::new(src, description.len)?;
    let mut current = *wrapped(&table, bits.read(log) as usize);
    let mut other = *wrapped(&table, bits.read(log) as usize);
    if bits.left() < 0 {
        return Err(Error::CorruptionDetected);
    }
    let advance = |state: WeightState, bits: &mut BackwardBits<'_>| {
        let next = u64::from(state.baseline) + bits.read(state.bits.into());
        *wrapped(&table, next as usize)
    };

    const BATCH: usize = 8;
    const { assert!(BATCH as u32 * WEIGHT_TABLE_LOG_MAX <= BackwardBits::IN_REFILL) };
    let mut count = 0;
    for batch in weights
        .as_chunks_mut::<BATCH>()
        .0
        .iter_mut()
        .take(255 / BATCH)
    {
        bits.refill();
        if bits.left() < (BATCH as u32 * WEIGHT_TABLE_LOG_MAX) as isize {
            break;
        }
        for pair in batch.as_chunks_mut().0 {
            *pair = [current.weight, other.weight];
            current = advance(current, &mut bits);
            other = advance(other, &mut bits);
        }
        count += BATCH;
    }
    while count < 254 {
        bits.refill();
        *wrapped_mut(weights, count) = current.weight;
        (current, other) = (other, advance(current, &mut bits));
        count += 1;
        if bits.left() < 0 {
            *wrapped_mut(weights, count) = current.weight;
            return Ok(count + 1);
        }
    }
    Err(Error::CorruptionDetected)
}

/// Longer codes are not worth more than one symbol at a time.
const QUICK_LOG: u32 = 11;
const QUICK_LEN: usize = 1 << QUICK_LOG;
/// What filling a table writes at once.
const FILL: usize = 16;

#[derive(Clone, Copy)]
#[repr(u8)]
enum SymbolCount {
    One = 1,
    Two = 2,
}

/// By the next [`QUICK_LOG`] bits: the one or two codes that they start with.
#[repr(C)]
struct PairTable {
    lengths: [u8; QUICK_LEN],
    counts: [SymbolCount; QUICK_LEN],
    symbols: [[u8; 2]; QUICK_LEN],
}

/// By the next `64 - shift` bits: the code that they start with.
#[derive(Clone, Copy)]
struct SingleTable<'a> {
    rows: &'a [[u8; QUICK_LEN]; 2],
    shift: u32,
}

#[derive(Default)]
pub(crate) struct HuffmanTable {
    log: u32,
    /// By the next `log` bits: the length of the code in the first half, and its symbol in the second.
    singles: Vec<u8>,
    /// One at most.
    pairs: Vec<PairTable>,
    has_pairs: bool,
}

impl HuffmanTable {
    fn half_len(&self) -> usize {
        QUICK_LEN.max(1 << self.log)
    }

    #[inline(always)]
    pub(crate) fn fill(&mut self, tree: &HuffmanWeights, with_pairs: bool) -> Result<()> {
        self.log = tree.table_log;
        let half_len = self.half_len();
        fill_up(&mut self.singles, 2 * half_len + FILL, 0)?;

        let (mut starts, mut end) = ([0u8; 16], 0);
        for (start, &count) in starts.iter_mut().zip(&tree.rank_counts) {
            *start = end as u8;
            end += count;
        }
        let mut sorted = [0; 256];
        for (symbol, &weight) in tree.weights.iter().enumerate().take(tree.symbol_count) {
            let start = wrapped_mut(&mut starts, weight.into());
            *wrapped_mut(&mut sorted, (*start).into()) = symbol as u8;
            *start = start.wrapping_add(1);
        }

        let halves = self.singles.split_at_mut_checked(half_len);
        let (mut lengths, mut symbols) = halves.ok_or(Error::Internal)?;
        let mut sorted = (sorted.get(tree.rank_counts[0] as usize..)).ok_or(Error::Internal)?;
        let ranks = (0..self.log).zip(tree.rank_counts.iter().skip(1));
        for (shorter_by, &count) in ranks.filter(|&(_, &count)| count != 0) {
            let (rank, rest) = (sorted.split_at_checked(count as usize)).ok_or(Error::Internal)?;
            sorted = rest;
            let rows = lengths.split_off_mut(..rank.len() << shorter_by);
            (rows.ok_or(Error::Internal)?).fill((self.log - shorter_by) as u8);
            let repeats = 1 << shorter_by;
            for &symbol in rank {
                let rows = symbols.get_mut(..repeats.max(FILL));
                for fill in rows.ok_or(Error::Internal)?.as_chunks_mut().0 {
                    *fill = [symbol; FILL];
                }
                symbols = symbols.get_mut(repeats..).ok_or(Error::Internal)?;
            }
        }

        self.has_pairs = with_pairs && self.log <= QUICK_LOG;
        if self.has_pairs {
            self.fill_pairs()?;
        }
        Ok(())
    }

    #[inline(always)]
    fn fill_pairs(&mut self) -> Result<()> {
        if self.pairs.is_empty() {
            self.pairs.try_reserve_exact(1)?;
            self.pairs.push(PairTable {
                lengths: [0; _],
                counts: [SymbolCount::One; _],
                symbols: [[0; _]; _],
            });
        }
        let pairs = self.pairs.first_mut().ok_or(Error::Internal)?;
        let singles = self.singles.as_chunks().0.first_chunk();
        let [lengths, symbols]: &[[u8; QUICK_LEN]; 2] = singles.ok_or(Error::Internal)?;
        let scale = QUICK_LOG - self.log;
        let mut row = 0;
        while row < QUICK_LEN {
            let (length, first) = (
                *wrapped(lengths, row >> scale),
                *wrapped(symbols, row >> scale),
            );
            let room = QUICK_LOG
                .checked_sub(length.into())
                .ok_or(Error::Internal)?;
            for after in 0..1usize << room {
                let next = after << length >> scale;
                let (both, second) = (length + wrapped(lengths, next), *wrapped(symbols, next));
                let fits = u32::from(both) <= QUICK_LOG;
                *wrapped_mut(&mut pairs.lengths, row) = if fits { both } else { length };
                *wrapped_mut(&mut pairs.counts, row) = if fits {
                    SymbolCount::Two
                } else {
                    SymbolCount::One
                };
                *wrapped_mut(&mut pairs.symbols, row) = [first, second];
                row += 1;
            }
        }
        Ok(())
    }

    #[inline(always)]
    fn quick_singles(&self) -> Option<SingleTable<'_>> {
        let rows = self.singles.as_chunks().0.first_chunk()?;
        (self.log <= QUICK_LOG).then_some(SingleTable {
            rows,
            shift: (64 - self.log).clamp(64 - QUICK_LOG, 63),
        })
    }

    #[inline(always)]
    fn pairs(&self) -> Option<&PairTable> {
        self.pairs.first().filter(|_| self.has_pairs)
    }

    /// Fills `out` from the stream in `src[start..]`.
    #[inline(always)]
    pub(crate) fn decode_stream(&self, src: &[u8], start: usize, out: &mut [u8]) -> Result<()> {
        let stream = Stream::new(src, bits_below_mark(src, start)?);
        self.finish(src, start, stream, out)
    }

    /// Fills `out` from the jump table and the four streams in `src[start..]`.
    #[inline(always)]
    pub(crate) fn decode_4_streams(&self, src: &[u8], start: usize, out: &mut [u8]) -> Result<()> {
        let jumps = src.get(start..).and_then(<[u8]>::first_chunk::<6>);
        let &[a0, a1, b0, b1, c0, c1] = jumps.ok_or(Error::CorruptionDetected)?;
        let start_a = start + 6;
        let start_b = start_a + usize::from(u16::from_le_bytes([a0, a1]));
        let start_c = start_b + usize::from(u16::from_le_bytes([b0, b1]));
        let start_d = start_c + usize::from(u16::from_le_bytes([c0, c1]));
        let stream = |start, end| -> Result<Stream> {
            let up_to_end = src.get(..end).ok_or(Error::CorruptionDetected)?;
            Ok(Stream::new(src, bits_below_mark(up_to_end, start)?))
        };
        let mut streams = [
            stream(start_a, start_b)?,
            stream(start_b, start_c)?,
            stream(start_c, start_d)?,
            stream(start_d, src.len())?,
        ];
        let segment = out.len().div_ceil(4);
        let (a, out) = out.split_at_mut_checked(segment).ok_or(Error::Internal)?;
        let (b, out) = out.split_at_mut_checked(segment).ok_or(Error::Internal)?;
        let (c, d) = out.split_at_mut_checked(segment).ok_or(Error::Internal)?;
        let mut outs = [a, b, c, d];
        if let Some(pairs) = self.pairs() {
            outs = pairs.decode(src, &mut streams, outs);
        } else if let Some(singles) = self.quick_singles() {
            outs = singles.decode(src, &mut streams, outs);
        }
        let starts = [start_a, start_b, start_c, start_d];
        for ((start, stream), out) in starts.into_iter().zip(streams).zip(outs) {
            self.finish(src, start, stream, out)?;
        }
        Ok(())
    }

    /// Fills `out` from the rest of `stream`, whose first byte is at `start`.
    #[inline(always)]
    fn finish(&self, src: &[u8], start: usize, stream: Stream, mut out: &mut [u8]) -> Result<()> {
        let mut streams = [stream];
        if let Some(pairs) = self.pairs() {
            [out] = pairs.decode(src, &mut streams, [out]);
        }
        if let Some(singles) = self.quick_singles() {
            [out] = singles.decode(src, &mut streams, [out]);
        }
        let [Stream { mut position, .. }] = streams;
        let halves = self.singles.split_at_checked(self.half_len());
        let (lengths, symbols) = halves.ok_or(Error::Internal)?;
        for symbol in out {
            let row = Stream::new(src, position).bits.wrapping_shr(64 - self.log) as usize;
            *symbol = *symbols.get(row).ok_or(Error::Internal)?;
            let length = *lengths.get(row).ok_or(Error::Internal)?;
            position = (position.checked_sub(length.into())).ok_or(Error::CorruptionDetected)?;
        }
        if position != 8 * start {
            return Err(Error::CorruptionDetected);
        }
        Ok(())
    }
}

/// The lookups from one load to the next. What a load gives is enough for one more, so that it does not have to wait.
const LOOKUPS: usize = 4;
const _: () = assert!((LOOKUPS as u32 + 1) * QUICK_LOG <= 57);

#[derive(Clone, Copy)]
struct Stream {
    /// The next bit is the highest.
    bits: u64,
    /// How many bits of the source are below those that were read.
    position: usize,
}

impl Stream {
    /// Zeros are below the start of `src`.
    #[inline(always)]
    fn new(src: &[u8], position: usize) -> Self {
        let bits = Self::load(src, position).unwrap_or_else(|| {
            load_from(src, 0).unbounded_shl(64u32.wrapping_sub(position as u32))
        });
        Self { bits, position }
    }

    /// At least 57 bits below `position`, if `src` has them.
    #[inline(always)]
    fn load(src: &[u8], position: usize) -> Option<u64> {
        let start = (position.wrapping_add(7) / 8).wrapping_sub(8);
        let bytes = src.get(start..)?.first_chunk()?;
        Some(u64::from_le_bytes(*bytes) << (position.wrapping_neg() % 8))
    }

    #[inline(always)]
    fn skip(&mut self, length: u8) {
        self.bits = self.bits.wrapping_shl(length.into());
        self.position = self.position.wrapping_sub(length.into());
    }
}

#[inline(always)]
fn load_all<const N: usize>(src: &[u8], streams: &[Stream; N]) -> Option<[u64; N]> {
    let mut loaded = [0; N];
    for (loaded, stream) in loaded.iter_mut().zip(streams) {
        *loaded = Stream::load(src, stream.position)?;
    }
    Some(loaded)
}

impl SingleTable<'_> {
    /// Returns what is left to fill of `outs`, the last of which is the shortest.
    #[inline(always)]
    fn decode<'o, const N: usize>(
        self,
        src: &[u8],
        streams: &mut [Stream; N],
        mut outs: [&'o mut [u8]; N],
    ) -> [&'o mut [u8]; N] {
        let [lengths, symbols] = self.rows;
        let rounds = outs
            .iter()
            .map(|out| out.len() / LOOKUPS)
            .min()
            .unwrap_or(0);
        let mut chunks = outs.each_mut().map(|out| {
            let chunks = out.as_chunks_mut::<LOOKUPS>().0;
            chunks.get_mut(..rounds).unwrap_or_default()
        });
        let mut done = 0;
        let mut state = *streams;
        while done < rounds {
            let Some(loaded) = load_all(src, &state) else {
                break;
            };
            for lookup in 0..LOOKUPS {
                for ((stream, chunks), loaded) in state.iter_mut().zip(&mut chunks).zip(loaded) {
                    let row = (stream.bits >> self.shift) as usize;
                    if lookup == 0 {
                        stream.bits = loaded;
                    }
                    stream.skip(*wrapped(lengths, row));
                    if let Some(chunk) = chunks.get_mut(done) {
                        *wrapped_mut(chunk, lookup) = *wrapped(symbols, row);
                    }
                }
            }
            done += 1;
        }
        *streams = state;
        outs.map(|out| out.get_mut(done * LOOKUPS..).unwrap_or_default())
    }
}

impl PairTable {
    /// Returns what is left to fill of `outs`.
    #[inline(always)]
    fn decode<'o, const N: usize>(
        &self,
        src: &[u8],
        streams: &mut [Stream; N],
        mut outs: [&'o mut [u8]; N],
    ) -> [&'o mut [u8]; N] {
        let mut state = *streams;
        while outs.iter().all(|out| out.len() >= 2 * LOOKUPS) {
            let Some(loaded) = load_all(src, &state) else {
                break;
            };
            let mut written = [0; N];
            for lookup in 0..LOOKUPS {
                let each = state
                    .iter_mut()
                    .zip(&mut outs)
                    .zip(&mut written)
                    .zip(loaded);
                for (((stream, out), written), loaded) in each {
                    let row = (stream.bits >> (64 - QUICK_LOG)) as usize;
                    if lookup == 0 {
                        stream.bits = loaded;
                    }
                    stream.skip(*wrapped(&self.lengths, row));
                    if let Some(pair) = out.get_mut(*written..).and_then(<[u8]>::first_chunk_mut) {
                        *pair = *wrapped(&self.symbols, row);
                    }
                    *written += *wrapped(&self.counts, row) as usize;
                }
            }
            for (out, written) in outs.iter_mut().zip(written) {
                *out = core::mem::take(out).get_mut(written..).unwrap_or_default();
            }
        }
        *streams = state;
        outs
    }
}
