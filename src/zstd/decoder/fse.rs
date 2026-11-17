//! Finite State Entropy: table descriptions, and the decoding tables that follow from them.

use crate::array::{wrapped, wrapped_mut};
use crate::bits::load_from;
use crate::error::{Error, Result};
use crate::format::{FSE_TABLE_LOG_MIN, fse_table_step};

const DESCRIBED_TABLE_LOG_MAX: u32 = 15;

/// What a table description says besides the counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TableDescription {
    /// In bytes.
    pub len: usize,
    pub max_symbol: usize,
    pub table_log: u32,
}

/// Reads the description at the start of `src` into `counts`, which has room for all symbols that are allowed.
#[inline(always)]
pub fn read_normalized_counts(src: &[u8], counts: &mut [i16]) -> Result<TableDescription> {
    const FLAGS_IN_LOAD: u32 = 28;

    let bits_from = |position: usize| load_from(src, position / 8) >> (position % 8);
    counts.fill(0);
    let table_log = (bits_from(0) & 15) as u32 + FSE_TABLE_LOG_MIN;
    if table_log > DESCRIBED_TABLE_LOG_MAX {
        return Err(Error::CorruptionDetected);
    }
    let (mut position, mut symbol) = (4, 0);
    let mut remaining = (1u32 << table_log) + 1;
    while remaining > 1 {
        let short_width = remaining.ilog2();
        let threshold = 1 << short_width;
        let short_values = 2 * threshold - 1 - remaining;
        let bits = bits_from(position) as u32;
        let (short, long) = (bits & (threshold - 1), bits & (2 * threshold - 1));
        let is_short = short < short_values;
        let value = if is_short || long < threshold {
            short
        } else {
            long - short_values
        };
        position += short_width as usize + usize::from(!is_short);
        let count = value as i32 - 1;
        *counts.get_mut(symbol).ok_or(Error::CorruptionDetected)? = count as i16;
        symbol += 1;
        remaining -= count.unsigned_abs();
        if count != 0 {
            continue;
        }
        loop {
            let flags = bits_from(position);
            let threes = (!flags | 1 << (2 * FLAGS_IN_LOAD)).trailing_zeros() / 2;
            symbol += 3 * threes as usize;
            position += 2 * threes as usize;
            if threes < FLAGS_IN_LOAD {
                symbol += (flags >> (2 * threes) & 3) as usize;
                position += 2;
                break;
            }
        }
    }
    let len = position.div_ceil(8);
    if len > src.len() {
        return Err(Error::CorruptionDetected);
    }
    Ok(TableDescription {
        len,
        max_symbol: symbol - 1,
        table_log,
    })
}

/// A row of a decoding table.
pub(crate) struct State {
    pub(crate) symbol: u8,
    /// How many bits to add to `baseline` for the next state.
    pub(crate) bits: u32,
    pub(crate) baseline: u16,
}

const STATES_MAX: usize = 512;
const SYMBOLS_MAX: usize = 128;

/// The rows of a decoding table, from the first.
pub(crate) struct States {
    symbols: [u8; STATES_MAX + 8],
    /// Where in `symbols` that of the next row is, modulo the size of the table.
    index: usize,
    stride: usize,
    table_log: u32,
    /// By symbol: its count, plus how many of its rows are behind.
    ranks: [u16; SYMBOLS_MAX],
}

impl States {
    /// `counts` add up to `1 << table_log`, which is at most 512.
    #[inline(always)]
    #[expect(clippy::indexing_slicing, reason = "below the length, or STATES_MAX")]
    pub(crate) const fn new(counts: &[i16], table_log: u32) -> Self {
        let size = 1 << table_log;
        let step = fse_table_step(size);
        let (mut symbols, mut ranks) = ([0; STATES_MAX + 8], [0; SYMBOLS_MAX]);
        let (mut symbol, mut end, mut rare) = (0, 0, 0);
        while symbol < counts.len() {
            let count = counts[symbol];
            *wrapped_mut(&mut ranks, symbol) = count.unsigned_abs();
            rare += (count < 0) as usize;
            let mut start = end;
            end += if count < 0 { 0 } else { count as usize };
            loop {
                if let Some(eight) = symbols.split_at_mut(start % STATES_MAX).1.first_chunk_mut() {
                    *eight = [symbol as u8; 8];
                }
                start += 8;
                if start >= end {
                    break;
                }
            }
            symbol += 1;
        }
        let stride = if rare == 0 {
            // Row `i * step` has symbol `i`, so row `i` has symbol `i / step`, modulo the size.
            let inverse = step.wrapping_mul(2usize.wrapping_sub(step * step));
            inverse.wrapping_mul(2usize.wrapping_sub(step.wrapping_mul(inverse)))
        } else {
            let sorted = symbols;
            let (mut row, mut index) = (0, 0);
            loop {
                symbols[row % STATES_MAX] = sorted[index % STATES_MAX];
                index += (row < end) as usize;
                row = (row + step) & (size - 1);
                if row == 0 {
                    break;
                }
            }
            let (mut symbol, mut row) = (0, size);
            while symbol < counts.len() {
                if counts[symbol] < 0 {
                    row -= 1;
                    symbols[row % STATES_MAX] = symbol as u8;
                }
                symbol += 1;
            }
            1
        };
        Self {
            symbols,
            index: 0,
            stride,
            table_log,
            ranks,
        }
    }

    #[inline(always)]
    #[expect(clippy::indexing_slicing, reason = "below STATES_MAX")]
    pub(crate) const fn next_state(&mut self) -> State {
        let size = 1 << self.table_log;
        let symbol = self.symbols[(self.index & (size - 1)) % STATES_MAX];
        self.index = self.index.wrapping_add(self.stride);
        let rank = *wrapped(&self.ranks, symbol as usize) as u32;
        *wrapped_mut(&mut self.ranks, symbol as usize) = rank as u16 + 1;
        let bits = (rank.leading_zeros() + self.table_log).wrapping_sub(31);
        State {
            symbol,
            bits,
            baseline: rank.wrapping_shl(bits).wrapping_sub(size as u32) as u16,
        }
    }
}
