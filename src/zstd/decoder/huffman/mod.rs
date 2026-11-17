//! Huffman coding: tree descriptions, decoding tables and the streams of literals.

mod weights;

use core::hint::black_box;

use crate::array::{fill_up, only_item, wrapped, wrapped_mut};
use crate::bits::{bits_below_mark, load_from};
use crate::cpu::with_bmi2;
use crate::error::{Error, Result};
use crate::format::BLOCK_LEN_MAX;

pub use weights::{HuffmanWeights, read_huffman_weights};

/// Longer codes are not worth more than one symbol at a time.
const QUICK_LOG: u32 = 11;
const QUICK_LEN: usize = 1 << QUICK_LOG;
/// What filling a table writes at once, in bytes. It goes up to that far beyond what it fills.
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
    lengths: [u8; QUICK_LEN + FILL],
    counts: [SymbolCount; QUICK_LEN + FILL],
    /// The first is in the low byte.
    symbols: [u16; QUICK_LEN + FILL],
}

/// By the next `log` bits: the code that they start with.
#[derive(Clone, Copy)]
struct SingleTable<'a> {
    /// The lengths, then the symbols.
    rows: &'a [[u8; QUICK_LEN]; 2],
    log: u32,
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

    /// From the tree description at the start of `src`, whose length it returns, for about `literals` of them.
    #[inline(never)]
    pub(crate) fn read(&mut self, src: &[u8], literals: usize) -> Result<usize> {
        with_bmi2(
            #[inline(always)]
            || {
                let mut tree = HuffmanWeights::EMPTY;
                tree.read(src)?;
                self.fill(&tree, literals)?;
                Ok(tree.len)
            },
        )
    }

    #[inline(always)]
    fn fill(&mut self, tree: &HuffmanWeights, literals: usize) -> Result<()> {
        self.log = tree.table_log;
        let half_len = self.half_len();
        fill_up(&mut self.singles, 2 * half_len + FILL, 0)?;
        let mut ranks = Ranks::EMPTY;
        ranks.sort(tree);
        let mut row = 0;
        for (length, symbols) in ranks.up_to(self.log) {
            let rows = symbols.len() << (self.log - length);
            fill_rows::<_, FILL>(&mut self.singles, row, rows, length as u8)?;
            row += rows;
        }
        row = half_len;
        for (length, symbols) in ranks.up_to(self.log) {
            let rows = 1 << (self.log - length);
            repeat_each::<_, FILL>(&mut self.singles, row, symbols, rows, |symbol| symbol)?;
            row += symbols.len() * rows;
        }

        self.has_pairs = self.log <= QUICK_LOG && ranks.are_pairs_quicker(literals);
        if self.has_pairs {
            only_item(&mut self.pairs, || PairTable::EMPTY)?.fill(&ranks)?;
        }
        Ok(())
    }

    #[inline(always)]
    fn quick_singles(&self) -> Option<SingleTable<'_>> {
        let rows = self.singles.as_chunks().0.first_chunk()?;
        (self.log <= QUICK_LOG).then_some(SingleTable {
            rows,
            log: self.log,
        })
    }

    #[inline(always)]
    fn pairs(&self) -> Option<&PairTable> {
        self.pairs.first().filter(|_| self.has_pairs)
    }

    /// Fills `out` from the stream in `src[start..]`.
    pub(crate) fn decode_stream(&self, src: &[u8], start: usize, out: &mut [u8]) -> Result<()> {
        self.finish(src, start, bits_below_mark(src, start)?, out)
    }

    /// Fills `out[..len]` from the jump table and the four streams in `src[start..]`. It is quickest if `out` has
    /// [`AMPLE_ROOM`].
    pub(crate) fn decode_4_streams(
        &self,
        src: &[u8],
        start: usize,
        out: &mut [u8],
        len: usize,
    ) -> Result<()> {
        let jumps = src.get(start..).and_then(<[u8]>::first_chunk::<6>);
        let &[a0, a1, b0, b1, c0, c1] = jumps.ok_or(Error::CorruptionDetected)?;
        let start_a = start + 6;
        let start_b = start_a + usize::from(u16::from_le_bytes([a0, a1]));
        let start_c = start_b + usize::from(u16::from_le_bytes([b0, b1]));
        let start_d = start_c + usize::from(u16::from_le_bytes([c0, c1]));
        let windows = Windows::new(src);
        let lane = |start, end| {
            let position =
                bits_below_mark(src.get(..end).ok_or(Error::CorruptionDetected)?, start)?;
            Lane::new(&windows, position).ok_or(Error::Internal)
        };
        let mut lanes = [
            lane(start_a, start_b)?,
            lane(start_b, start_c)?,
            lane(start_c, start_d)?,
            lane(start_d, src.len())?,
        ];
        let segment = len.div_ceil(4);
        let mut done = [0, segment, 2 * segment, 3 * segment];
        if let (Some(pairs), Some(out)) = (self.pairs(), out.first_chunk_mut()) {
            let ends = [segment, 2 * segment, 3 * segment, len];
            pairs.decode_4_in_place(&windows, &mut lanes, out, &mut done, &ends)?;
        }
        let out = out.get_mut(..len).ok_or(Error::Internal)?;
        let (out, d) = out.split_at_mut_checked(done[3]).ok_or(Error::Internal)?;
        let c = out
            .get_mut(..3 * segment)
            .and_then(|out| out.split_at_mut_checked(done[2]));
        let (out, c) = c.ok_or(Error::Internal)?;
        let b = out
            .get_mut(..2 * segment)
            .and_then(|out| out.split_at_mut_checked(done[1]));
        let (out, b) = b.ok_or(Error::Internal)?;
        let mut outs = [
            out.get_mut(done[0]..segment).ok_or(Error::Internal)?,
            b,
            c,
            d,
        ];

        if let Some(pairs) = self.pairs() {
            pairs.decode_4(&windows, &mut lanes, &mut outs)?;
        } else if let Some(singles) = self.quick_singles() {
            singles.decode_4(&windows, &mut lanes, &mut outs)?;
        }
        let starts = [start_a, start_b, start_c, start_d];
        for ((&start, lane), out) in starts.iter().zip(&lanes).zip(&mut outs) {
            let position = lane.position().ok_or(Error::CorruptionDetected)?;
            self.finish(src, start, position, out)?;
        }
        Ok(())
    }

    /// Fills `out` from the bits below `position` of the stream whose first byte is at `start`.
    #[inline(never)]
    fn finish(&self, src: &[u8], start: usize, position: usize, out: &mut [u8]) -> Result<()> {
        with_bmi2(
            #[inline(always)]
            || self.finish_inline(src, start, position, out),
        )
    }

    #[inline(always)]
    fn finish_inline(
        &self,
        src: &[u8],
        start: usize,
        position: usize,
        mut out: &mut [u8],
    ) -> Result<()> {
        let mut stream = Stream::new(src, position);
        if let Some(pairs) = self.pairs() {
            out = pairs.decode(src, &mut stream, out);
        }
        if let Some(singles) = self.quick_singles() {
            out = singles.decode(src, &mut stream, out);
        }
        let halves = self.singles.split_at_checked(self.half_len());
        let (lengths, symbols) = halves.ok_or(Error::Internal)?;
        for round in out.chunks_mut(LOOKUPS) {
            stream = Stream::new(src, stream.position);
            for symbol in round {
                let row = stream.bits.wrapping_shr(64 - self.log) as usize;
                *symbol = *symbols.get(row).ok_or(Error::Internal)?;
                stream.skip(*lengths.get(row).ok_or(Error::Internal)?);
            }
        }
        if stream.position != 8 * start {
            return Err(Error::CorruptionDetected);
        }
        Ok(())
    }
}

/// Sets `count` rows from `start` on, `N` at a time.
#[inline(always)]
fn fill_rows<T: Copy, const N: usize>(
    rows: &mut [T],
    start: usize,
    count: usize,
    value: T,
) -> Result<()> {
    let rows = (rows.get_mut(start..)).and_then(|rows| rows.get_mut(..count + N - 1));
    for chunk in rows.ok_or(Error::Internal)?.as_chunks_mut().0 {
        *chunk = [value; N];
        // Most are short: neither unrolling nor a call of `memset()` pays.
        black_box(());
    }
    Ok(())
}

/// Sets `repeats` rows for each of `symbols` to what `map` makes of it, from `start` on.
#[inline(always)]
fn repeat_each<T: Copy, const N: usize>(
    rows: &mut [T],
    start: usize,
    symbols: &[u8],
    repeats: usize,
    map: impl Fn(u8) -> T,
) -> Result<()> {
    let mut rows = rows.get_mut(start..).ok_or(Error::Internal)?;
    if repeats <= N {
        for &symbol in symbols {
            *rows.first_chunk_mut().ok_or(Error::Internal)? = [map(symbol); N];
            rows = rows.get_mut(repeats..).ok_or(Error::Internal)?;
        }
    } else {
        for &symbol in symbols {
            fill_rows::<_, N>(rows, 0, repeats, map(symbol))?;
            rows = rows.get_mut(repeats..).ok_or(Error::Internal)?;
        }
    }
    Ok(())
}

/// Sets `count` rows from `to` on to what `map` makes of those from `from` on, `N` at a time.
#[inline(always)]
fn copy_rows<T: Copy, const N: usize>(
    rows: &mut [T],
    [from, to]: [usize; 2],
    count: usize,
    map: impl Fn(T) -> T,
) -> Result<()> {
    for done in (0..count).step_by(N) {
        let from = rows.get(from + done..).and_then(<[T]>::first_chunk::<N>);
        let mut chunk = *from.ok_or(Error::Internal)?;
        chunk.iter_mut().for_each(|it| *it = map(*it));
        let to = rows.get_mut(to + done..).and_then(<[T]>::first_chunk_mut);
        *to.ok_or(Error::Internal)? = chunk;
    }
    Ok(())
}

/// The symbols in the order of their codes: from the longest to the shortest.
struct Ranks {
    symbols: [u8; 256],
    /// By length: how many symbols come before those with shorter codes.
    ends: [u16; 16],
    /// By length: how many of the values of [`QUICK_LOG`] bits start with a code that is no longer.
    rows_up_to: [u32; 16],
    shortest: u32,
    longest: u32,
}

impl Ranks {
    const EMPTY: Self = Self {
        symbols: [0; _],
        ends: [0; _],
        rows_up_to: [0; _],
        shortest: 0,
        longest: 0,
    };

    /// To be called on [`Self::EMPTY`].
    #[inline(always)]
    fn sort(&mut self, tree: &HuffmanWeights) {
        let longest = tree.table_log;
        let count_of = |length: u32| *wrapped(&tree.rank_counts, (longest + 1 - length) as usize);
        let unused = 4 * tree.quarter_len() - tree.symbol_count + tree.rank_counts[0] as usize;
        let (mut starts, mut end) = ([[0u8; 16]; 4], unused as u16);
        *wrapped_mut(&mut self.ends, longest as usize + 1) = end;
        for weight in 1..longest + 1 {
            let length = longest + 1 - weight;
            *wrapped_mut(&mut starts[0], weight as usize) = end as u8;
            end += count_of(length) as u16;
            *wrapped_mut(&mut self.ends, length as usize) = end;
            if count_of(length) != 0 {
                self.shortest = length;
            }
        }
        self.longest = longest;
        let mut before = starts[0];
        for (starts, counts) in starts.iter_mut().zip(&tree.counts_by_quarter) {
            *starts = before;
            for (before, &count) in before.iter_mut().zip(counts) {
                *before = before.wrapping_add(count);
            }
        }
        tree.for_each_in_quarters(
            #[inline(always)]
            |quarter, symbol, weight| {
                let start = wrapped_mut(wrapped_mut(&mut starts, quarter), weight);
                *wrapped_mut(&mut self.symbols, (*start).into()) = symbol;
                *start = start.wrapping_add(1);
            },
        );
        let mut rows = 0;
        for length in self.shortest..QUICK_LOG + 1 {
            if length <= longest {
                rows += count_of(length) << (QUICK_LOG - length);
            }
            *wrapped_mut(&mut self.rows_up_to, length as usize) = rows;
        }
    }

    /// Each length from `longest` down, with its symbols.
    #[inline(always)]
    fn up_to(&self, longest: u32) -> impl Iterator<Item = (u32, &[u8])> {
        (self.shortest..longest.min(self.longest) + 1)
            .rev()
            .map(|length| {
                let [start, end] =
                    [length + 1, length].map(|it| usize::from(*wrapped(&self.ends, it as usize)));
                (length, self.symbols.get(start..end).unwrap_or_default())
            })
    }

    /// Whether looking up two codes at a time saves more than filling the table for it takes.
    #[inline(always)]
    fn are_pairs_quicker(&self, literals: usize) -> bool {
        // In units of the time that a literal takes on its own.
        const FILL_TIME: usize = 1200;
        const LOOKUP_TIME_IN_EIGHTHS: usize = 10;

        if literals <= FILL_TIME {
            return false;
        }
        let up_to = |length: u32| *wrapped(&self.rows_up_to, length as usize);
        let paired = (1..QUICK_LOG)
            .map(|length| (up_to(length) - up_to(length - 1)) * up_to(QUICK_LOG - length));
        let codes_in_table = QUICK_LEN + (paired.sum::<u32>() >> QUICK_LOG) as usize;
        FILL_TIME + literals * QUICK_LEN / codes_in_table * LOOKUP_TIME_IN_EIGHTHS / 8 < literals
    }
}

/// The lookups from one load to the next. What a load gives is enough for one more, so that it does not have to wait.
const LOOKUPS: usize = 4;
const _: () = assert!((LOOKUPS as u32 + 1) * QUICK_LOG <= 56);

/// A stream on its own.
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

/// An index of 8 bits, and the 8 bytes that start there.
const WINDOW_LEN: usize = 255 + 8;
type Window = [u8; WINDOW_LEN];

/// In which loads need no check. Zeros are below the start of the source.
struct Windows<'a> {
    src: &'a [u8],
    padded_start: [Window; 2],
}

impl<'a> Windows<'a> {
    #[inline(always)]
    fn new(src: &'a [u8]) -> Self {
        let mut padded_start = [[0; _]; _];
        let len = src.len().min(WINDOW_LEN);
        if let (Some(to), Some(from)) = (padded_start[1].get_mut(..len), src.get(..len)) {
            to.copy_from_slice(from);
        }
        Self { src, padded_start }
    }

    #[inline(always)]
    fn before(&self, end: usize) -> Option<&Window> {
        match end.checked_sub(WINDOW_LEN) {
            Some(start) => self.src.get(start..)?.first_chunk(),
            None => self.padded_start.as_flattened().get(end..)?.first_chunk(),
        }
    }
}

/// One of four streams, whose position takes no register between loads.
#[derive(Clone, Copy)]
struct Lane {
    /// The next bit is the highest. The lowest that is set was the lowest of the load.
    bits: u64,
    /// Of the last load, in the source.
    end: usize,
}

impl Lane {
    #[inline(always)]
    fn new(windows: &Windows<'_>, position: usize) -> Option<Self> {
        let end = position.div_ceil(8);
        let bytes = windows.before(end)?.last_chunk()?;
        let bits = (u64::from_le_bytes(*bytes) | 1) << (8 * end - position);
        Some(Self { bits, end })
    }

    #[inline(always)]
    fn position(self) -> Option<usize> {
        (8 * self.end).checked_sub(self.bits.trailing_zeros() as usize)
    }
}

/// In which a round can start at any position that has as many bits as those in a block.
pub(crate) const AMPLE_ROOM: usize = BLOCK_LEN_MAX + 2 * LOOKUPS;

/// The most rounds of [`LOOKUPS`] in a window.
const ROUNDS_MAX: usize = 32;
const _: () = assert!(7 + ROUNDS_MAX * LOOKUPS * QUICK_LOG as usize <= 8 * 255);

/// Does `rounds` rounds of lookups in all lanes. `find` returns the row for the next bits and how many of them it takes.
/// `write` gets the lane, the round, the lookup in it, and that row.
#[inline(always)]
fn interleave(
    windows: &Windows<'_>,
    lanes: &mut [Lane; 4],
    rounds: usize,
    find: impl Fn(u64) -> (usize, u8),
    mut write: impl FnMut(usize, usize, usize, usize),
) -> Result<()> {
    let window = |lane: &Lane| windows.before(lane.end).ok_or(Error::Internal);
    let [a, b, c, d] = &*lanes;
    let windows = [window(a)?, window(b)?, window(c)?, window(d)?];
    let mut bits = [a.bits, b.bits, c.bits, d.bits];
    let mut starts = [255usize; 4];
    for round in 0..rounds.min(ROUNDS_MAX) {
        macro_rules! first_lookup {
            ($($lane:literal)*) => {$(
                let used = bits[$lane].trailing_zeros();
                starts[$lane] = starts[$lane].wrapping_sub(used as usize / 8);
                let bytes = windows[$lane].get(starts[$lane] % 256..).and_then(<[u8]>::first_chunk);
                let loaded = u64::from_le_bytes(*bytes.ok_or(Error::Internal)?) | 1;
                let (row, length) = find(bits[$lane]);
                bits[$lane] = loaded.wrapping_shl(used % 8 + u32::from(length));
                write($lane, round, 0, row);
            )*};
        }
        macro_rules! lookup {
            ($lookup:literal: $($lane:literal)*) => {$(
                let (row, length) = find(bits[$lane]);
                bits[$lane] = bits[$lane].wrapping_shl(length.into());
                write($lane, round, $lookup, row);
            )*};
        }
        first_lookup!(0 1 2 3);
        lookup!(1: 0 1 2 3);
        lookup!(2: 0 1 2 3);
        lookup!(3: 0 1 2 3);
    }
    for ((lane, &bits), &start) in lanes.iter_mut().zip(&bits).zip(&starts) {
        let end = lane.end.checked_sub(255usize.wrapping_sub(start));
        (lane.bits, lane.end) = (bits, end.ok_or(Error::CorruptionDetected)?);
    }
    Ok(())
}

/// Out of line, or the compiler makes it one with copies whose length it knows.
#[inline(never)]
fn copy_some(from: &[u8], to: &mut [u8], len: usize) {
    if let (Some(from), Some(to)) = (from.get(..len), to.get_mut(..len)) {
        to.copy_from_slice(from);
    }
}

/// Moves the first `len` bytes of `stage` to `out`, and `out` on.
#[inline(always)]
fn unload<const N: usize>(stage: &[[u8; 128]; N], len: usize, out: &mut &mut [u8]) -> Result<()> {
    let room = core::mem::take(out);
    if let Some(chunks) = room.as_chunks_mut().0.first_chunk_mut::<N>() {
        chunks
            .iter_mut()
            .zip(stage)
            .for_each(|(to, from)| *to = *from);
    } else {
        copy_some(stage.as_flattened(), room, len);
    }
    *out = room.get_mut(len..).ok_or(Error::Internal)?;
    Ok(())
}

impl SingleTable<'_> {
    /// From 64 bits to a row. The compiler knows that there is one.
    #[inline(always)]
    fn shift(self) -> u32 {
        64u32.wrapping_sub(self.log).clamp(64 - QUICK_LOG, 63)
    }

    /// Returns what is left to fill of `out`.
    #[inline(always)]
    fn decode<'o>(self, src: &[u8], stream: &mut Stream, mut out: &'o mut [u8]) -> &'o mut [u8] {
        let ([lengths, symbols], shift) = (self.rows, self.shift());
        while let Some(loaded) = Stream::load(src, stream.position)
            && let Some(round) = out.first_chunk_mut::<LOOKUPS>()
        {
            for (lookup, symbol) in round.iter_mut().enumerate() {
                let row = (stream.bits >> shift) as usize;
                if lookup == 0 {
                    stream.bits = loaded;
                }
                stream.skip(*wrapped(lengths, row));
                *symbol = *wrapped(symbols, row);
            }
            out = out.get_mut(LOOKUPS..).unwrap_or_default();
        }
        out
    }

    /// Leaves in `outs`, the last of which is the shortest, what is left to fill.
    #[inline(never)]
    fn decode_4(
        self,
        windows: &Windows<'_>,
        lanes: &mut [Lane; 4],
        outs: &mut [&mut [u8]; 4],
    ) -> Result<()> {
        with_bmi2(
            #[inline(always)]
            move || self.decode_4_inline(windows, lanes, outs),
        )
    }

    #[inline(always)]
    fn decode_4_inline(
        self,
        windows: &Windows<'_>,
        lanes: &mut [Lane; 4],
        outs: &mut [&mut [u8]; 4],
    ) -> Result<()> {
        let ([lengths, symbols], shift) = (self.rows, self.shift());
        // Four places to write to would take too many registers.
        let mut stages = [[[0; ROUNDS_MAX * LOOKUPS]]; 4];
        let mut left = outs[3].len() / LOOKUPS;
        while left != 0 {
            let rounds = (left - 1) % ROUNDS_MAX + 1;
            interleave(
                windows,
                lanes,
                rounds,
                #[inline(always)]
                |bits| {
                    (
                        (bits >> shift) as usize,
                        *wrapped(lengths, (bits >> shift) as usize),
                    )
                },
                #[inline(always)]
                |lane, round, lookup, row| {
                    let [stage] = wrapped_mut(&mut stages, lane);
                    *wrapped_mut(stage, round * LOOKUPS + lookup) = *wrapped(symbols, row);
                },
            )?;
            for (stage, out) in stages.iter().zip(&mut *outs) {
                unload(stage, rounds * LOOKUPS, out)?;
            }
            left -= rounds;
        }
        Ok(())
    }
}

impl PairTable {
    const EMPTY: Self = Self {
        lengths: [0; _],
        counts: [SymbolCount::One; _],
        symbols: [0; _],
    };

    #[inline(always)]
    fn fill(&mut self, ranks: &Ranks) -> Result<()> {
        const SYMBOLS: usize = FILL / 2;

        let mut row = 0;
        for (length, firsts) in ranks.up_to(QUICK_LOG) {
            let Some((&first, others)) = firsts.split_first() else {
                continue;
            };
            let (start, room) = (row, QUICK_LOG - length);
            let alone = (QUICK_LEN - *wrapped(&ranks.rows_up_to, room as usize) as usize) >> length;
            fill_rows::<_, FILL>(&mut self.counts, row, alone, SymbolCount::One)?;
            fill_rows::<_, FILL>(
                &mut self.counts,
                row + alone,
                (1 << room) - alone,
                SymbolCount::Two,
            )?;
            fill_rows::<_, FILL>(&mut self.lengths, row, alone, length as u8)?;
            fill_rows::<_, SYMBOLS>(&mut self.symbols, row, alone, first.into())?;
            row += alone;
            for (second_length, seconds) in ranks.up_to(room) {
                let (both, rows) = ((length + second_length) as u8, 1 << (room - second_length));
                fill_rows::<_, FILL>(&mut self.lengths, row, seconds.len() * rows, both)?;
                let pair = |second| u16::from_le_bytes([first, second]);
                repeat_each::<_, SYMBOLS>(&mut self.symbols, row, seconds, rows, pair)?;
                row += seconds.len() * rows;
            }
            // What can follow a code depends on nothing but its length.
            for &other in others {
                copy_rows::<_, FILL>(&mut self.counts, [start, row], 1 << room, |it| it)?;
                copy_rows::<_, FILL>(&mut self.lengths, [start, row], 1 << room, |it| it)?;
                let with_other = |pair| pair & 0xFF00 | u16::from(other);
                copy_rows::<_, SYMBOLS>(&mut self.symbols, [start, row], 1 << room, with_other)?;
                row += 1 << room;
            }
        }
        Ok(())
    }

    /// The row for what `bits` start with, and how many of them it takes.
    #[inline(always)]
    fn find(&self, bits: u64) -> (usize, u8) {
        let row = (bits >> (64 - QUICK_LOG)) as usize;
        (row, self.lengths.get(row).copied().unwrap_or_default())
    }

    /// At `position` of `out`, which it moves on.
    #[inline(always)]
    fn write<const N: usize>(&self, row: usize, out: &mut [u8; N], position: &mut usize) {
        if let Some(pair) = out.get_mut(*position..).and_then(<[u8]>::first_chunk_mut) {
            *pair = self
                .symbols
                .get(row)
                .copied()
                .unwrap_or_default()
                .to_le_bytes();
        }
        *position += self.counts.get(row).map_or(1, |&it| it as usize);
    }

    /// Returns what is left to fill of `out`.
    #[inline(always)]
    fn decode<'o>(&self, src: &[u8], stream: &mut Stream, mut out: &'o mut [u8]) -> &'o mut [u8] {
        while let Some(loaded) = Stream::load(src, stream.position)
            && let Some(round) = out.first_chunk_mut::<{ 2 * LOOKUPS }>()
        {
            let mut len = 0;
            for lookup in 0..LOOKUPS {
                let (row, length) = self.find(stream.bits);
                if lookup == 0 {
                    stream.bits = loaded;
                }
                stream.skip(length);
                self.write(row, round, &mut len);
            }
            out = out.get_mut(len..).unwrap_or_default();
        }
        out
    }

    /// Leaves in `outs` what is left to fill.
    #[inline(never)]
    fn decode_4(
        &self,
        windows: &Windows<'_>,
        lanes: &mut [Lane; 4],
        outs: &mut [&mut [u8]; 4],
    ) -> Result<()> {
        with_bmi2(
            #[inline(always)]
            move || self.decode_4_inline(windows, lanes, outs),
        )
    }

    #[inline(always)]
    fn decode_4_inline(
        &self,
        windows: &Windows<'_>,
        lanes: &mut [Lane; 4],
        outs: &mut [&mut [u8]; 4],
    ) -> Result<()> {
        const ROOM_MIN: usize = 4 * LOOKUPS;

        let mut stages = [[0; 256 + 2 * LOOKUPS]; 4];
        loop {
            // They do not go at the same pace. Those that are through repeat what another does.
            let mut busy = outs
                .iter()
                .zip(&*lanes)
                .filter(|(out, _)| out.len() >= ROOM_MIN);
            let (Some((first, &leader)), Some((second, _))) = (busy.next(), busy.next()) else {
                return Ok(());
            };
            let room = busy.fold(first.len().min(second.len()), |room, (out, _)| {
                room.min(out.len())
            });
            let rounds = room.min(255) / (2 * LOOKUPS);
            let mut working = [leader; 4];
            for ((working, lane), out) in working.iter_mut().zip(&*lanes).zip(&*outs) {
                *working = if out.len() >= ROOM_MIN { *lane } else { leader };
            }
            let mut lens = [0; 4];
            interleave(
                windows,
                &mut working,
                rounds,
                #[inline(always)]
                |bits| self.find(bits),
                #[inline(always)]
                |lane, _, lookup, row| {
                    let len = wrapped_mut(&mut lens, lane);
                    if lookup == 0 {
                        *len %= 256;
                    }
                    self.write(row, wrapped_mut(&mut stages, lane), len);
                },
            )?;
            let results = stages.iter().zip(&lens).zip(&working);
            for (((stage, &len), working), (out, lane)) in
                results.zip(outs.iter_mut().zip(&mut *lanes))
            {
                if out.len() >= ROOM_MIN {
                    let stage = stage.as_chunks().0.first_chunk::<2>();
                    unload(stage.ok_or(Error::Internal)?, len, out)?;
                    *lane = *working;
                }
            }
        }
    }

    /// From `positions`, which it moves on, until one is close to its end.
    #[inline(never)]
    fn decode_4_in_place(
        &self,
        windows: &Windows<'_>,
        lanes: &mut [Lane; 4],
        out: &mut [u8; AMPLE_ROOM],
        positions: &mut [usize; 4],
        ends: &[usize; 4],
    ) -> Result<()> {
        with_bmi2(
            #[inline(always)]
            move || {
                let mut at = *positions;
                loop {
                    let room = |lane| wrapped(ends, lane).saturating_sub(*wrapped(&at, lane));
                    let rounds = room(0).min(room(1)).min(room(2)).min(room(3)) / (2 * LOOKUPS);
                    if rounds < ROUNDS_MAX / 4 {
                        *positions = at;
                        return Ok(());
                    }
                    interleave(
                        windows,
                        lanes,
                        rounds,
                        #[inline(always)]
                        |bits| self.find(bits),
                        #[inline(always)]
                        |lane, _, lookup, row| {
                            let position = wrapped_mut(&mut at, lane);
                            if lookup == 0 {
                                *position %= BLOCK_LEN_MAX;
                            }
                            self.write(row, out, position);
                        },
                    )?;
                }
            },
        )
    }
}
