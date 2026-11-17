//! Up to two symbols at a time.

use super::QUICK_LOG;
use super::ranks::Ranks;
use super::streams::{LOOKUPS, Lane, ROUNDS_MAX, Stream, Windows, interleave, unload};
use bun_highway::with_bmi2;

use crate::array::At;
use crate::error::{Error, Result};
use crate::format::BLOCK_LEN_MAX;

pub(super) use layout::PairTable;

/// Loads are cheaper than shifts.
#[cfg(target_arch = "x86_64")]
mod layout {
    use super::super::rows::{FILL, copy_rows, fill_rows, repeat_each};
    use super::super::{QUICK_LEN, QUICK_LOG};
    use crate::error::Result;

    pub(super) const COUNT_MAX: usize = 2;
    const PAIRS_IN_FILL: usize = FILL / 2;

    #[derive(Clone, Copy)]
    #[repr(u8)]
    enum Count {
        None,
        One,
        Two,
    }

    /// By the next [`QUICK_LOG`] bits: the one or two codes that they start with.
    #[repr(C)]
    pub(crate) struct PairTable {
        lengths: [u8; QUICK_LEN + FILL],
        counts: [Count; QUICK_LEN + FILL],
        symbols: [u16; QUICK_LEN + FILL],
    }

    impl PairTable {
        pub(crate) const EMPTY: Self = Self {
            lengths: [0; _],
            counts: [Count::None; _],
            symbols: [0; _],
        };

        #[inline(always)]
        pub(super) fn set_alone(
            &mut self,
            row: usize,
            rows: usize,
            symbol: u8,
            length: u32,
        ) -> Result<()> {
            fill_rows::<_, FILL>(&mut self.lengths, row, rows, length as u8)?;
            fill_rows::<_, FILL>(&mut self.counts, row, rows, Count::One)?;
            fill_rows::<_, PAIRS_IN_FILL>(&mut self.symbols, row, rows, symbol.into())
        }

        #[inline(always)]
        pub(super) fn set_pairs(
            &mut self,
            row: usize,
            repeats: usize,
            first: u8,
            seconds: &[u8],
            length: u32,
        ) -> Result<()> {
            fill_rows::<_, FILL>(
                &mut self.lengths,
                row,
                seconds.len() * repeats,
                length as u8,
            )?;
            fill_rows::<_, FILL>(&mut self.counts, row, seconds.len() * repeats, Count::Two)?;
            let pair = |second| u16::from_le_bytes([first, second]);
            repeat_each::<_, PAIRS_IN_FILL>(&mut self.symbols, row, seconds, repeats, pair)
        }

        #[inline(always)]
        pub(super) fn copy_for(
            &mut self,
            first: u8,
            from_to: [usize; 2],
            rows: usize,
        ) -> Result<()> {
            copy_rows::<_, FILL>(&mut self.lengths, from_to, rows, |it| it)?;
            copy_rows::<_, FILL>(&mut self.counts, from_to, rows, |it| it)?;
            let with_first = |pair| pair & 0xFF00 | u16::from(first);
            copy_rows::<_, PAIRS_IN_FILL>(&mut self.symbols, from_to, rows, with_first)
        }

        /// The row for what `bits` start with, and how many of them it takes.
        #[inline(always)]
        pub(super) fn find(&self, bits: u64) -> (usize, u8) {
            let row = (bits >> (64 - QUICK_LOG)) as usize;
            (row, self.lengths.get(row).copied().unwrap_or_default())
        }

        #[inline(always)]
        pub(super) fn symbols(&self, row: usize) -> u16 {
            self.symbols.get(row).copied().unwrap_or_default()
        }

        #[inline(always)]
        pub(super) fn count(&self, row: usize) -> usize {
            self.counts.get(row).map_or(0, |&it| it as usize)
        }
    }
}

/// Shifts are cheaper than loads.
#[cfg(not(target_arch = "x86_64"))]
mod layout {
    use super::super::rows::{FILL, copy_rows, fill_rows, repeat_each};
    use super::super::{QUICK_LEN, QUICK_LOG};
    use crate::error::Result;

    /// As far as the compiler can tell.
    pub(super) const COUNT_MAX: usize = 3;
    const ROWS_IN_FILL: usize = FILL / 4;
    const SYMBOLS_SHIFT: u32 = 14;
    const COUNT_SHIFT: u32 = 30;

    /// By the next [`QUICK_LOG`] bits: the length, the symbols and the count of the one or two codes that they start with.
    pub(crate) struct PairTable([u32; QUICK_LEN + ROWS_IN_FILL]);

    impl PairTable {
        pub(crate) const EMPTY: Self = Self([0; _]);

        #[inline(always)]
        pub(super) fn set_alone(
            &mut self,
            row: usize,
            rows: usize,
            symbol: u8,
            length: u32,
        ) -> Result<()> {
            let value = length | u32::from(symbol) << SYMBOLS_SHIFT | 1 << COUNT_SHIFT;
            fill_rows::<_, ROWS_IN_FILL>(&mut self.0, row, rows, value)
        }

        #[inline(always)]
        pub(super) fn set_pairs(
            &mut self,
            row: usize,
            repeats: usize,
            first: u8,
            seconds: &[u8],
            length: u32,
        ) -> Result<()> {
            let value = length | u32::from(first) << SYMBOLS_SHIFT | 2 << COUNT_SHIFT;
            let pair = |second| value | u32::from(second) << (SYMBOLS_SHIFT + 8);
            repeat_each::<_, ROWS_IN_FILL>(&mut self.0, row, seconds, repeats, pair)
        }

        #[inline(always)]
        pub(super) fn copy_for(
            &mut self,
            first: u8,
            from_to: [usize; 2],
            rows: usize,
        ) -> Result<()> {
            let with_first =
                |value| value & !(0xFF << SYMBOLS_SHIFT) | u32::from(first) << SYMBOLS_SHIFT;
            copy_rows::<_, ROWS_IN_FILL>(&mut self.0, from_to, rows, with_first)
        }

        /// The row for what `bits` start with, and how many of them it takes.
        #[inline(always)]
        pub(super) fn find(&self, bits: u64) -> (usize, u8) {
            let row = self.0.get((bits >> (64 - QUICK_LOG)) as usize);
            let row = row.copied().unwrap_or_default();
            (row as usize, row as u8)
        }

        #[inline(always)]
        pub(super) fn symbols(&self, row: usize) -> u16 {
            (row >> SYMBOLS_SHIFT) as u16
        }

        #[inline(always)]
        pub(super) fn count(&self, row: usize) -> usize {
            row >> COUNT_SHIFT
        }
    }
}

/// How far a round writes at most.
const ROUND_ROOM: usize = (LOOKUPS - 1) * layout::COUNT_MAX + 2;

/// In which a round can start at any position that has as many bits as those in a block.
pub(crate) const AMPLE_ROOM: usize = BLOCK_LEN_MAX + ROUND_ROOM;

impl PairTable {
    #[inline(always)]
    pub(super) fn fill(&mut self, ranks: &Ranks) -> Result<()> {
        let mut row = 0;
        for (length, firsts) in ranks.up_to(QUICK_LOG) {
            let Some((&first, others)) = firsts.split_first() else {
                continue;
            };
            let (start, room) = (row, QUICK_LOG - length);
            let alone = ranks.rows_beyond(room) >> length;
            self.set_alone(row, alone, first, length)?;
            row += alone;
            for (second_length, seconds) in ranks.up_to(room) {
                let repeats = 1 << (room - second_length);
                self.set_pairs(row, repeats, first, seconds, length + second_length)?;
                row += seconds.len() * repeats;
            }
            // What can follow a code depends on nothing but its length.
            for &other in others {
                self.copy_for(other, [start, row], 1 << room)?;
                row += 1 << room;
            }
        }
        Ok(())
    }

    /// At `position` of `out`, which it moves on.
    #[inline(always)]
    fn write<const N: usize>(&self, row: usize, out: &mut [u8; N], position: &mut usize) {
        if let Some(pair) = out.get_mut(*position..).and_then(<[u8]>::first_chunk_mut) {
            *pair = self.symbols(row).to_le_bytes();
        }
        *position += self.count(row);
    }

    /// Returns what is left to fill of `out`.
    #[inline(always)]
    pub(super) fn decode<'o>(
        &self,
        src: &[u8],
        stream: &mut Stream,
        mut out: &'o mut [u8],
    ) -> &'o mut [u8] {
        while let Some(loaded) = Stream::load(src, stream.position)
            && let Some(round) = out.first_chunk_mut::<ROUND_ROOM>()
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
    pub(super) fn decode_4(
        &self,
        windows: &Windows<'_>,
        lanes: &mut [Lane; 4],
        outs: &mut [&mut [u8]; 4],
    ) -> Result<()> {
        const ROOM_MIN: usize = 4 * LOOKUPS;

        with_bmi2(
            #[inline(always)]
            move || {
                let mut stages = [[0; 256 + ROUND_ROOM]; 4];
                loop {
                    // They do not go at the same pace. Those that are through repeat what another does.
                    let mut busy = outs
                        .iter()
                        .zip(&*lanes)
                        .filter(|(out, _)| out.len() >= ROOM_MIN);
                    let (Some((first, &leader)), Some((second, _))) = (busy.next(), busy.next())
                    else {
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
                            let len = lens.at_mut(lane);
                            if lookup == 0 {
                                *len %= 256;
                            }
                            self.write(row, stages.at_mut(lane), len);
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
            },
        )
    }

    /// From `positions`, which it moves on, until one is close to its end.
    pub(super) fn decode_4_in_place(
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
                    let room = |lane| ends.at(lane).saturating_sub(*at.at(lane));
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
                            let position = at.at_mut(lane);
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
