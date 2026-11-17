//! Finite State Entropy.

use bun_zstd_decoder::format::{Distribution, FSE_TABLE_LOG_MIN, fse_table_step};
use bun_zstd_decoder::{Error, Result};

use super::bit_writer::BitWriter;
use crate::array::At;
use crate::bytes::{chunk_at_mut, highest_bit};

/// More than any alphabet has.
pub(crate) const SYMBOLS: usize = 64;
const STATES: usize = 512;
const TABLE_LOG_MAX: u32 = 12;

fn min_table_log(len: usize, max_symbol: usize) -> u32 {
    (highest_bit(len as u32) + 1).min(highest_bit(max_symbol as u32) + 2)
}

/// For `len` symbols, of which there are at least 2.
pub(super) fn table_log_for(max_log: u32, len: usize, max_symbol: usize, minus: u32) -> u32 {
    let useful = highest_bit(len.wrapping_sub(1) as u32).wrapping_sub(minus);
    let log = max_log.min(useful).max(min_table_log(len, max_symbol));
    log.clamp(FSE_TABLE_LOG_MIN, TABLE_LOG_MAX)
}

/// Counts scaled to add up to a power of 2. -1 is a count of less than 1.
#[derive(Clone, Copy)]
pub(crate) struct Normalized {
    pub(crate) counts: [i16; SYMBOLS],
    pub(crate) max_symbol: usize,
    pub(crate) table_log: u32,
}

impl Normalized {
    pub(super) const fn predefined(distribution: &Distribution<'_>) -> Self {
        let mut counts = [0; SYMBOLS];
        let (used, _) = counts.split_at_mut(distribution.counts.len());
        used.copy_from_slice(distribution.counts);
        Self {
            counts,
            max_symbol: distribution.counts.len() - 1,
            table_log: distribution.table_log,
        }
    }

    /// `counts` end with a symbol that occurs, have another one, and add up to `total`.
    pub(super) fn new(
        counts: &[u32],
        total: usize,
        table_log: u32,
        allows_less_than_1: bool,
    ) -> Result<Self> {
        const ROUND_UP_FROM: [u64; 8] = [
            0, 473_195, 504_333, 520_860, 550_000, 700_000, 750_000, 830_000,
        ];
        let max_symbol = counts.len().checked_sub(1).ok_or(Error::Internal)?;
        if table_log < min_table_log(total, max_symbol) {
            return Err(Error::Internal);
        }
        let mut normalized = Self {
            counts: [0; SYMBOLS],
            max_symbol,
            table_log,
        };
        let rare = if allows_less_than_1 { -1 } else { 1 };
        let scale = 62 - table_log;
        let step = (1u64 << 62)
            .checked_div(total as u64)
            .ok_or(Error::Internal)?;
        let rare_up_to = (total >> table_log) as u32;
        let mut left = 1i32 << table_log;
        let (mut largest, mut largest_share) = (0, 0);
        for (symbol, (&count, out)) in counts.iter().zip(&mut normalized.counts).enumerate() {
            if count == 0 {
                continue;
            }
            if count <= rare_up_to {
                *out = rare;
                left -= 1;
                continue;
            }
            let scaled = u64::from(count) * step;
            let whole = scaled >> scale;
            let rounds_up = ROUND_UP_FROM
                .get(whole as usize)
                .is_some_and(|&from| scaled - (whole << scale) > from << (scale - 20));
            let share = whole as i16 + i16::from(rounds_up);
            if share > largest_share {
                (largest, largest_share) = (symbol, share);
            }
            *out = share;
            left -= i32::from(share);
        }
        let largest = normalized.counts.at_mut(largest);
        if -left >= i32::from(*largest >> 1) {
            normalized.distribute_evenly(counts, total, rare)?;
        } else {
            *largest += left as i16;
        }
        Ok(normalized)
    }

    /// For when the most frequent symbol cannot make up for the rounding of the others.
    #[cold]
    fn distribute_evenly(&mut self, counts: &[u32], mut total: usize, rare: i16) -> Result<()> {
        const UNASSIGNED: i16 = -2;
        let table_size = 1u32 << self.table_log;
        let rare_up_to = (total >> self.table_log) as u32;
        let mut one_up_to = ((total * 3) >> (self.table_log + 1)) as u32;
        let mut assigned = 0;
        for (&count, out) in counts.iter().zip(&mut self.counts) {
            *out = if count == 0 {
                0
            } else if count <= one_up_to.max(rare_up_to) {
                assigned += 1;
                total -= count as usize;
                if count <= rare_up_to { rare } else { 1 }
            } else {
                UNASSIGNED
            };
        }
        let mut left = table_size.checked_sub(assigned).ok_or(Error::Internal)?;
        if left == 0 {
            return Ok(());
        }

        if total / left as usize > one_up_to as usize {
            one_up_to = ((total * 3) / (left as usize * 2)) as u32;
            for (&count, out) in counts.iter().zip(&mut self.counts) {
                if *out == UNASSIGNED && count <= one_up_to {
                    *out = 1;
                    assigned += 1;
                    total -= count as usize;
                }
            }
            left = table_size.checked_sub(assigned).ok_or(Error::Internal)?;
        }

        if assigned as usize == counts.len() {
            let most = counts.iter().max();
            let first_with_most = counts.iter().position(|count| Some(count) == most);
            *self.counts.at_mut(first_with_most.unwrap_or(0)) += left as i16;
            return Ok(());
        }

        if total == 0 {
            if !self.counts.iter().any(|&count| count > 0) {
                return Err(Error::Internal);
            }
            while left > 0 {
                let positive = self.counts.iter_mut().filter(|count| **count > 0);
                for count in positive.take(left as usize) {
                    *count += 1;
                    left -= 1;
                }
            }
            return Ok(());
        }

        let scale = 62 - self.table_log;
        let half = (1u64 << (scale - 1)) - 1;
        let step = ((u64::from(left) << scale) + half) / total as u64;
        let mut start = half;
        for (&count, out) in counts.iter().zip(&mut self.counts) {
            if *out == UNASSIGNED {
                let end = start + u64::from(count) * step;
                *out = ((end >> scale) - (start >> scale)) as i16;
                if *out < 1 {
                    return Err(Error::Internal);
                }
                start = end;
            }
        }
        Ok(())
    }

    fn used(&self) -> &[i16] {
        self.counts.get(..=self.max_symbol).unwrap_or_default()
    }

    /// Returns the length of the description.
    pub(super) fn write(&self, out: &mut [u8]) -> Result<usize> {
        let mut position = 0;
        let mut bits = self.table_log - FSE_TABLE_LOG_MIN;
        let mut bit_count = 4;
        let mut put_16_bits = |bits: u32, position: u32| {
            let pair = chunk_at_mut(out, position).ok_or(Error::DstSizeTooSmall)?;
            *pair = (bits as u16).to_le_bytes();
            Ok::<_, Error>(())
        };

        let mut remaining = (1i32 << self.table_log) + 1;
        let mut threshold = 1i32 << self.table_log;
        let mut width = self.table_log + 1;
        let mut rest = self.used();
        let mut follows_zero = false;
        while remaining > 1 {
            if follows_zero {
                let mut zeros = rest.iter().take_while(|&&count| count == 0).count();
                rest = rest.get(zeros..).unwrap_or_default();
                while zeros >= 24 {
                    zeros -= 24;
                    bits += 0xFFFF << bit_count;
                    put_16_bits(bits, position)?;
                    position += 2;
                    bits >>= 16;
                }
                while zeros >= 3 {
                    zeros -= 3;
                    bits += 3 << bit_count;
                    bit_count += 2;
                }
                bits += (zeros as u32) << bit_count;
                bit_count += 2;
                if bit_count > 16 {
                    put_16_bits(bits, position)?;
                    position += 2;
                    bits >>= 16;
                    bit_count -= 16;
                }
            }
            let (&count, after) = rest.split_first().ok_or(Error::Internal)?;
            rest = after;
            let short_below = 2 * threshold - 1 - remaining;
            remaining -= i32::from(count).abs();
            let mut value = i32::from(count) + 1;
            if value >= threshold {
                value += short_below;
            }
            bits += (value as u32) << bit_count;
            bit_count += width - u32::from(value < short_below);
            follows_zero = value == 1;
            if remaining < 1 {
                return Err(Error::Internal);
            }
            while remaining < threshold {
                width -= 1;
                threshold >>= 1;
            }
            if bit_count > 16 {
                put_16_bits(bits, position)?;
                position += 2;
                bits >>= 16;
                bit_count -= 16;
            }
        }
        put_16_bits(bits, position)?;
        Ok((position + bit_count.div_ceil(8)) as usize)
    }
}

/// So far up that the compiler knows the number of bits to be a valid shift.
const BITS_SHIFT: u32 = 28;

#[derive(Clone, Copy)]
struct Transform {
    /// From the state, once its bits are out, to where the next state is.
    state_delta: i32,
    /// Added to the state, the highest bits are the number of bits to write.
    bits_delta: u32,
}

/// Which part of [`Tables`] a table takes.
#[derive(Clone, Copy)]
pub(crate) struct Shape {
    first_state: usize,
    first_symbol: usize,
    log: u32,
    max_symbol: usize,
}

impl Shape {
    pub(super) const EMPTY: Self = Self {
        first_state: 0,
        first_symbol: 0,
        log: 0,
        max_symbol: 0,
    };

    #[inline(always)]
    pub(super) fn finish(self, writer: &mut BitWriter<'_>, state: u32) {
        writer.add(state, self.log);
        writer.flush();
    }
}

/// Room for tables side by side. Where no [`Shape`] is given, symbols count from the first table on.
pub(crate) struct Tables<const ALL_STATES: usize, const ALL_SYMBOLS: usize> {
    next_states: [u16; ALL_STATES],
    transforms: [Transform; ALL_SYMBOLS],
}

impl<const ALL_STATES: usize, const ALL_SYMBOLS: usize> Tables<ALL_STATES, ALL_SYMBOLS> {
    pub(super) const EMPTY: Self = Self {
        next_states: [0; ALL_STATES],
        transforms: [Transform {
            state_delta: 0,
            bits_delta: 0,
        }; ALL_SYMBOLS],
    };

    pub(super) fn copy_from(&mut self, other: &Self, shape: Shape) {
        let states = shape.first_state..shape.first_state + (1 << shape.log);
        let symbols = shape.first_symbol..=shape.first_symbol + shape.max_symbol;
        if let Some(to) = self.next_states.get_mut(states.clone())
            && let Some(from) = other.next_states.get(states)
        {
            to.copy_from_slice(from);
        }
        if let Some(to) = self.transforms.get_mut(symbols.clone())
            && let Some(from) = other.transforms.get(symbols)
        {
            to.copy_from_slice(from);
        }
    }

    /// The table number `index`, for a single symbol, which takes no bits.
    pub(super) fn build_for_one(&mut self, index: usize, symbol: usize) -> Shape {
        let (first_state, first_symbol) = (index * STATES, index * SYMBOLS);
        *self.next_states.at_mut(first_state) = 0;
        *self.transforms.at_mut(first_symbol + symbol) = Transform {
            state_delta: first_state as i32,
            bits_delta: 0,
        };
        Shape {
            first_state,
            first_symbol,
            log: 0,
            max_symbol: symbol,
        }
    }

    /// The table number `index`.
    #[expect(clippy::indexing_slicing, reason = "each index is wrapped")]
    pub(super) const fn build(&mut self, index: usize, normalized: &Normalized) -> Shape {
        let (first_state, first_symbol) = (index * STATES, index * SYMBOLS);
        let log = normalized.table_log;
        let size = 1usize << log;
        let step = fse_table_step(size);
        let counts = &normalized.counts;
        let symbol_count = normalized.max_symbol % SYMBOLS + 1;

        let mut symbol_of_state = [0u8; STATES];
        let mut starts = [0u16; SYMBOLS];
        let mut last_spread = size - 1;
        let mut total = first_state as i32;
        let mut symbol = 0;
        while symbol < symbol_count {
            let count = counts[symbol] as i32;
            starts[symbol] = total as u16;
            self.transforms[(first_symbol + symbol) % ALL_SYMBOLS] = match count {
                0 => Transform {
                    state_delta: 0,
                    bits_delta: ((log + 1) << BITS_SHIFT) - size as u32,
                },
                -1 | 1 => Transform {
                    state_delta: total - 1,
                    bits_delta: (log << BITS_SHIFT) - size as u32,
                },
                _ => {
                    let max_bits = log - highest_bit(count as u32 - 1);
                    Transform {
                        state_delta: total - count,
                        bits_delta: (max_bits << BITS_SHIFT) - ((count as u32) << max_bits),
                    }
                }
            };
            if count == -1 {
                symbol_of_state[last_spread % STATES] = symbol as u8;
                last_spread = last_spread.wrapping_sub(1);
            }
            total += count.abs();
            symbol += 1;
        }

        if last_spread == size - 1 {
            // In two passes, no branch depends on the counts as long as they are up to 8.
            let mut sorted = [0u8; STATES + 8];
            let (mut symbol, mut position) = (0, 0);
            while symbol < symbol_count {
                let mut filled = 0;
                loop {
                    let at = position + filled;
                    let (_, rest) = sorted.split_at_mut(if at < STATES { at } else { STATES });
                    if let Some(eight) = rest.first_chunk_mut() {
                        *eight = [symbol as u8; 8];
                    }
                    filled += 8;
                    if filled >= counts[symbol] as usize {
                        break;
                    }
                }
                position += counts[symbol] as usize;
                symbol += 1;
            }
            let (mut index, mut state) = (0, 0);
            while index < size {
                symbol_of_state[state % STATES] = sorted[index % STATES];
                state = (state + step) & (size - 1);
                index += 1;
            }
        } else {
            let (mut symbol, mut state) = (0, 0);
            while symbol < symbol_count {
                let mut left = counts[symbol];
                while left > 0 {
                    symbol_of_state[state % STATES] = symbol as u8;
                    state = (state + step) & (size - 1);
                    while state > last_spread {
                        state = (state + step) & (size - 1);
                    }
                    left -= 1;
                }
                symbol += 1;
            }
        }

        let mut state = 0;
        while state < size {
            let start = &mut starts[symbol_of_state[state % STATES] as usize % SYMBOLS];
            self.next_states[*start as usize % ALL_STATES] = (size + state) as u16;
            *start += 1;
            state += 1;
        }
        Shape {
            first_state,
            first_symbol,
            log,
            max_symbol: normalized.max_symbol,
        }
    }

    /// The most bits that `symbol` takes. One more than the log of the table if it cannot be encoded.
    pub(super) fn max_bits(&self, shape: Shape, symbol: usize) -> u32 {
        if symbol > shape.max_symbol {
            return shape.log + 1;
        }
        let transform = self.transforms.at(shape.first_symbol + symbol);
        transform.bits_delta.div_ceil(1 << BITS_SHIFT)
    }

    /// About how many bits symbols with these counts take, if they can be encoded.
    pub(super) fn cost(&self, shape: Shape, counts: &[u32]) -> Option<usize> {
        const ACCURACY_LOG: u32 = 8;
        if counts.len() > shape.max_symbol + 1 {
            return None;
        }
        let mut cost = 0;
        for (&count, transform) in counts
            .iter()
            .zip(self.transforms.get(shape.first_symbol..)?)
        {
            if count == 0 {
                continue;
            }
            let max_bits = (transform.bits_delta >> BITS_SHIFT) + 1;
            let below = (max_bits << BITS_SHIFT) - (transform.bits_delta + (1 << shape.log));
            let bits = (max_bits << ACCURACY_LOG) - ((below << ACCURACY_LOG) >> shape.log);
            if bits >= (shape.log + 1) << ACCURACY_LOG {
                return None;
            }
            cost += count as usize * bits as usize;
        }
        Some(cost >> ACCURACY_LOG)
    }

    /// The state that stands for `symbol` without any bits written.
    #[inline(always)]
    pub(super) fn first_state(&self, symbol: u8) -> u32 {
        let transform = self.transforms.at(usize::from(symbol));
        let bits = transform.bits_delta.div_ceil(1 << BITS_SHIFT);
        let state = (bits << BITS_SHIFT) - transform.bits_delta;
        self.next_state(state >> bits, *transform)
    }

    #[inline(always)]
    fn next_state(&self, shifted: u32, transform: Transform) -> u32 {
        let index = shifted.wrapping_add_signed(transform.state_delta);
        u32::from(*self.next_states.at(index as usize))
    }

    #[inline(always)]
    pub(super) fn encode(&self, writer: &mut BitWriter<'_>, state: &mut u32, symbol: u8) {
        let transform = self.transforms.at(usize::from(symbol));
        let bits = (*state + transform.bits_delta) >> BITS_SHIFT;
        writer.add(*state, bits);
        *state = self.next_state(*state >> bits, *transform);
    }

    /// With two states that take turns. `None` if there is nothing to gain, or no room.
    pub(super) fn compress(&self, shape: Shape, symbols: &[u8], out: &mut [u8]) -> Option<usize> {
        let &[ref rest @ .., before_last, last] = symbols else {
            return None;
        };
        if rest.is_empty() {
            return None;
        }
        let mut writer = BitWriter::new(out)?;
        let (mut due, mut other) = (self.first_state(last), self.first_state(before_last));
        let (pairs, single) = rest.as_chunks();
        if let &[symbol] = single {
            self.encode(&mut writer, &mut due, symbol);
            (due, other) = (other, due);
        }
        for &[first, second] in pairs.iter().rev() {
            self.encode(&mut writer, &mut due, second);
            self.encode(&mut writer, &mut other, first);
            writer.flush();
        }
        shape.finish(&mut writer, due);
        shape.finish(&mut writer, other);
        writer.finish()
    }
}
