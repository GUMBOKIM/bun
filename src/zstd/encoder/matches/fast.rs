//! One hash table, and the first match that comes up.

use core::cell::Cell;
use core::convert::Infallible;

use bun_zstd_decoder::with_bmi2;

use super::hash::Hasher;
use super::{Attached, MatchState, Owner, Progress, Reach, SKIP_LOG, cells, prefetch};
use crate::bytes::{AtLeast, chunk_at, u32_at, u64_at, word_in};
use crate::parameters::TAG_BITS;
use crate::sequences::{OffsetValue, Repeats, Sequences};
use crate::window::{FIRST_INDEX, History, WORD_LEN};

pub(super) const FILL_STEP: usize = 3;

/// The entry for `index` in the table of a dictionary.
#[inline(always)]
pub(super) fn tagged(index: u32, hash: usize) -> u32 {
    index << TAG_BITS | u32::from(hash as u8)
}

/// The index that a dictionary has for `hash`, if its tag is the same.
#[inline(always)]
pub(super) fn look_up_tagged(table: &[u32], hash: usize) -> Option<u32> {
    let &entry = table.get(hash >> TAG_BITS)?;
    (entry as u8 == hash as u8).then_some(entry >> TAG_BITS)
}

/// Indexes the history up to its end.
pub(super) fn fill(state: &mut MatchState, history: &History<'_>, owner: Owner) {
    let tag_bits = if owner == Owner::Dictionary {
        TAG_BITS
    } else {
        0
    };
    let hasher = Hasher::new(state.tuning.min_match, state.tuning.hash_log + tag_bits);
    let mut position = history.position(state.next_to_index);
    let table = state.tables.split_mut().hash;
    while let Some(chunk) = chunk_at::<{ WORD_LEN + FILL_STEP - 1 }>(history.prefix, position) {
        let index = history.index(position);
        for (step, word) in chunk.array_windows().enumerate() {
            let hash = hasher.hash(u64::from_le_bytes(*word));
            let Some(entry) = table.get_mut(hash >> tag_bits) else {
                return;
            };
            if owner == Owner::Frame {
                *entry = index;
                break;
            }
            if step == 0 || *entry == 0 {
                *entry = tagged(index + step as u32, hash);
            }
        }
        position += FILL_STEP as u32;
    }
}

pub(super) fn search<const REACH: Reach>(
    state: &mut MatchState,
    history: &History<'_>,
    attached: &Attached<'_>,
    sequences: &mut Sequences,
    repeats: &mut Repeats,
) -> u32 {
    let (tuning, bounds) = (state.tuning, state.bounds());
    let end = history.index(history.end());
    let searcher = Searcher {
        history: *history,
        table: cells(state.tables.split_mut().hash),
        hasher: Hasher::new(tuning.min_match, tuning.hash_log),
        step: tuning.target_length.max(1),
    };

    if REACH == Reach::Attached {
        let mut progress = Progress::new(history.start, *repeats, u32::MAX);
        searcher.search_with_attached(attached, &mut progress, sequences);
        return progress.finish(repeats, u32::MAX);
    }

    let lowest = bounds.lowest(end);
    if REACH == Reach::Older && lowest < history.prefix_index {
        let max_offset = history.index(history.start) - lowest - 1;
        let progress = Progress::new(history.start, *repeats, max_offset);
        let progress = with_bmi2(
            #[inline(always)]
            move || {
                let mut progress = progress;
                searcher.search_with_older(lowest, &mut progress, sequences);
                progress
            },
        );
        return progress.finish(repeats, max_offset);
    }

    let lowest = history.position(bounds.lowest_in_prefix(end));
    let start = history.start + u32::from(history.start == lowest);
    let max_offset = history.index(start) - bounds.lowest_in_prefix(history.index(start));
    let progress = Progress::new(history.start, *repeats, max_offset);
    // In a small window it is hard to predict whether a candidate is still in it.
    let progress = if tuning.window_log < 19 {
        searcher.search_prefix::<true>(lowest, start, progress, sequences)
    } else {
        searcher.search_prefix::<false>(lowest, start, progress, sequences)
    };
    progress.finish(repeats, max_offset)
}

/// Two adjacent positions, each with a word to hash, and the margin that the search keeps.
const PAIR_LEN: usize = WORD_LEN + 2;
type Pair = [u8; PAIR_LEN];

#[derive(Clone, Copy)]
struct Searcher<'a> {
    history: History<'a>,
    table: &'a [Cell<u32>],
    hasher: Hasher,
    step: u32,
}

impl<'a> Searcher<'a> {
    #[inline(always)]
    fn entry(&self, word: u64) -> Option<&'a Cell<u32>> {
        self.table.get(self.hasher.hash(word))
    }

    #[inline(always)]
    fn insert(&self, position: u32) -> Option<()> {
        self.entry(u64_at(self.history.prefix, position)?)?
            .set(self.history.index(position));
        Some(())
    }

    #[inline(never)]
    fn search_prefix<const IS_BRANCHLESS: bool>(
        self,
        lowest: u32,
        start: u32,
        progress: Progress,
        sequences: &mut Sequences,
    ) -> Progress {
        with_bmi2(
            #[inline(always)]
            move || {
                let mut progress = progress;
                progress.anchor -= lowest;
                self.scan_prefix::<IS_BRANCHLESS>(lowest, start, &mut progress, sequences);
                progress.anchor += lowest;
                progress
            },
        )
    }

    /// The steps for a position, which each wait for memory, overlap with those for the next ones.
    #[inline(always)]
    fn scan_prefix<const IS_BRANCHLESS: bool>(
        mut self,
        lowest: u32,
        start: u32,
        progress: &mut Progress,
        sequences: &mut Sequences,
    ) -> Option<Infallible> {
        const STEP_GROWS_AFTER: u32 = 1 << (SKIP_LOG - 1);
        let near = Near::<PAIR_LEN>::new(self.history, lowest)?;
        self.history = near.history;
        let (history, bytes) = (&self.history, self.history.prefix);
        let has_at = |at, value| {
            near.bytes
                .matches_at::<IS_BRANCHLESS>(at, |it| it as u32 == value)
        };
        let mut at = start - lowest;

        loop {
            let mut step = self.step + 1;
            let mut next_step = at + STEP_GROWS_AFTER;
            let mut ahead = at + step;
            let pair: &Pair = chunk_at(bytes, at)?;
            let mut word = word_in::<0, _>(pair);
            let mut entry = self.entry(word)?;
            let mut next_entry = self.entry(word_in::<1, _>(pair))?;
            let mut candidate = history.position(entry.get());

            let (searched, repeats) = loop {
                let ahead_pair: &Pair = chunk_at(bytes, ahead)?;
                let ahead_word = word_in::<0, _>(ahead_pair);
                let repeats = near.repeats(ahead, progress.recent, ahead_word as u32);

                entry.set(history.index(at));
                if repeats {
                    next_entry.set(history.index(at + 1));
                    break (at, true);
                }
                if has_at(candidate, word as u32) {
                    next_entry.set(history.index(at + 1));
                    break (at, false);
                }

                candidate = history.position(next_entry.get());
                entry = self.entry(ahead_word)?;
                next_entry.set(history.index(at + 1));
                if has_at(candidate, (word >> 8) as u32) {
                    // Only if the search is certain to go on after it.
                    if step <= 4 {
                        entry.set(history.index(ahead));
                    }
                    break (at + 1, false);
                }

                candidate = history.position(entry.get());
                next_entry = self.entry(word_in::<1, _>(ahead_pair))?;
                (at, word) = (ahead, ahead_word);
                ahead += step;
                if ahead >= next_step {
                    step += 1;
                    prefetch(bytes, at + 1 + 64);
                    prefetch(bytes, at + 1 + 128);
                    next_step += STEP_GROWS_AFTER;
                }
            };

            if repeats {
                let before = near.repeat_len_before(ahead, progress.recent).min(1);
                let len = before + near.repeat_len(ahead, progress.recent);
                progress.store(sequences, ahead - before, OffsetValue::FIRST_REPEAT, len);
            } else {
                let max = candidate.min(searched - progress.anchor);
                let before = history.match_len_before(searched, candidate, max);
                let len = before + 4 + history.match_len(searched + 4, candidate + 4);
                progress.note_offset(searched.wrapping_sub(candidate));
                let offset = OffsetValue::new(progress.recent);
                progress.store(sequences, searched - before, offset, len);
            }

            at = progress.anchor;
            let mut word = u64_at(bytes, at)?;
            self.insert(searched + 2)?;
            self.insert(at - 2)?;
            while near.repeats(at, progress.second, word as u32) {
                let len = near.repeat_len(at, progress.second);
                progress.swap_offsets();
                self.entry(word)?.set(history.index(at));
                progress.store(sequences, at, OffsetValue::FIRST_REPEAT, len);
                at = progress.anchor;
                word = u64_at(bytes, at)?;
            }
        }
    }

    /// Takes matches at the second most recent offset for as long as there are any.
    #[inline(always)]
    fn take_repeats(&self, progress: &mut Progress, sequences: &mut Sequences) -> Option<()> {
        let history = &self.history;
        loop {
            let at = progress.anchor;
            let word = u64_at(history.prefix, at)?;
            let earlier = history.index(at).wrapping_sub(progress.second);
            if progress.second == 0 || history.u32_at_index(earlier) != Some(word as u32) {
                return Some(());
            }
            let len = history.match_len_at_index(at, earlier, 4);
            progress.swap_offsets();
            self.entry(word)?.set(history.index(at));
            progress.store(sequences, at, OffsetValue::FIRST_REPEAT, len);
        }
    }

    #[inline(always)]
    fn search_with_older(
        mut self,
        lowest: u32,
        progress: &mut Progress,
        sequences: &mut Sequences,
    ) -> Option<Infallible> {
        const STEP_GROWS_AFTER: u32 = 1 << (SKIP_LOG - 1);
        self.history = self.history.with_prefix_of::<PAIR_LEN>()?;
        let history = &self.history;
        let bytes = history.prefix;
        let is_at = |index: u32, word: u64| {
            index >= lowest && history.u32_at_index(index) == Some(word as u32)
        };

        loop {
            let mut at = progress.anchor;
            let mut step = self.step + 1;
            let mut next_step = at + STEP_GROWS_AFTER;
            let mut ahead = at + step;
            let pair: &Pair = chunk_at(bytes, at)?;
            let mut word = word_in::<0, _>(pair);
            let mut entry = self.entry(word)?;
            let mut next_entry = self.entry(word_in::<1, _>(pair))?;
            let mut candidate = entry.get();

            // Also the position that is hashed but not yet in the table.
            let (searched, repeats, pending, pending_entry) = loop {
                let ahead_pair: &Pair = chunk_at(bytes, ahead)?;
                let ahead_word = word_in::<0, _>(ahead_pair);
                let back = history.index(ahead).wrapping_sub(progress.recent);
                let repeated = history.u32_at_index(back);

                entry.set(history.index(at));
                if (repeated == Some(ahead_word as u32))
                    & (progress.recent > 0)
                    & (back != history.prefix_index)
                {
                    break (at, true, at + 1, next_entry);
                }
                if is_at(candidate, word) {
                    break (at, false, at + 1, next_entry);
                }

                candidate = next_entry.get();
                entry = self.entry(ahead_word)?;
                next_entry.set(history.index(at + 1));
                if is_at(candidate, word >> 8) {
                    break (at + 1, false, ahead, entry);
                }

                candidate = entry.get();
                next_entry = self.entry(word_in::<1, _>(ahead_pair))?;
                (at, word) = (ahead, ahead_word);
                ahead += step;
                if ahead >= next_step {
                    step += 1;
                    prefetch(bytes, at + 1 + 64);
                    prefetch(bytes, at + 1 + 128);
                    next_step += STEP_GROWS_AFTER;
                }
            };

            let (found, earlier, offset, max_before);
            if repeats {
                found = ahead;
                earlier = history.index(found).wrapping_sub(progress.recent);
                (offset, max_before) = (OffsetValue::FIRST_REPEAT, 1);
            } else {
                (found, earlier) = (searched, candidate);
                progress.note_offset(history.index(found).wrapping_sub(earlier));
                offset = OffsetValue::new(progress.recent);
                max_before = (found - progress.anchor).min(earlier - lowest);
            }
            let before = history.match_len_before_index(found, earlier, max_before);
            let len = before + history.match_len_at_index(found, earlier, 4);
            progress.store(sequences, found - before, offset, len);

            if pending < progress.anchor {
                pending_entry.set(history.index(pending));
            }
            u64_at(bytes, progress.anchor)?;
            self.insert(searched + 2)?;
            self.insert(progress.anchor - 2)?;
            self.take_repeats(progress, sequences)?;
        }
    }

    fn search_with_attached(
        mut self,
        attached: &Attached<'_>,
        progress: &mut Progress,
        sequences: &mut Sequences,
    ) -> Option<Infallible> {
        const STEP_GROWS_AFTER: u32 = 1 << SKIP_LOG;
        let words = AtLeast::<WORD_LEN>::new(self.history.prefix)?;
        self.history.prefix = words.bytes;
        let (history, bytes) = (&self.history, words.bytes);
        let tuning = &attached.tuning;
        let attached_hasher = Hasher::new(tuning.min_match, tuning.hash_log + TAG_BITS);

        loop {
            let mut at = progress.anchor;
            let mut step = self.step;
            let mut next_step = at + STEP_GROWS_AFTER;
            let mut ahead = at + step;
            let mut ahead_word = u64_at(bytes, ahead)?;
            let mut word = u64_at(bytes, at)?;
            let mut entry = self.entry(word)?;
            let mut attached_candidate = look_up_tagged(attached.hash, attached_hasher.hash(word));
            let mut candidate = entry.get();

            let (found, earlier, repeats) = loop {
                let current = history.index(at);
                let back = (current + 1).wrapping_sub(progress.recent);
                let ahead_entry = self.entry(ahead_word)?;
                let ahead_hash = attached_hasher.hash(ahead_word);
                entry.set(current);

                if history.u32_at_index(back) == Some((word >> 8) as u32) {
                    break (at + 1, back, true);
                }
                if let Some(index) = attached_candidate
                    && index > FIRST_INDEX
                    && u32_at(history.older, Attached::position(index)) == Some(word as u32)
                    && candidate <= history.prefix_index
                {
                    break (at, index.wrapping_add(attached.delta), false);
                }
                if words
                    .matches_at::<true>(history.position(candidate), |it| it as u32 == word as u32)
                {
                    break (at, candidate, false);
                }

                attached_candidate = look_up_tagged(attached.hash, ahead_hash);
                candidate = ahead_entry.get();
                if ahead >= next_step {
                    step += 1;
                    next_step += STEP_GROWS_AFTER;
                }
                (at, word, entry) = (ahead, ahead_word, ahead_entry);
                ahead += step;
                ahead_word = u64_at(bytes, ahead)?;
            };

            let (offset, before) = if repeats {
                (OffsetValue::FIRST_REPEAT, 0)
            } else {
                progress.note_offset(history.index(found).wrapping_sub(earlier));
                let max = found - progress.anchor;
                let before = history.match_len_before_index(found, earlier, max);
                (OffsetValue::new(progress.recent), before)
            };
            let len = before + history.match_len_at_index(found, earlier, 4);
            progress.store(sequences, found - before, offset, len);

            u64_at(bytes, progress.anchor)?;
            self.insert(at + 2)?;
            self.insert(progress.anchor - 2)?;
            self.take_repeats(progress, sequences)?;
        }
    }
}

/// The end of the prefix that is within reach of the whole block. One comparison tells whether a candidate is in it.
#[derive(Clone, Copy)]
pub(super) struct Near<'a, const N: usize> {
    pub(super) bytes: AtLeast<'a, N>,
    pub(super) history: History<'a>,
    /// Offsets from earlier blocks can reach further.
    all: History<'a>,
    start: u32,
}

impl<'a, const N: usize> Near<'a, N> {
    #[inline(always)]
    pub(super) fn new(all: History<'a>, start: u32) -> Option<Self> {
        let bytes = AtLeast::new(all.prefix.get(start as usize..)?)?;
        let history = History {
            prefix: bytes.bytes,
            prefix_index: all.index(start),
            ..all
        };
        Some(Self {
            bytes,
            history,
            all,
            start,
        })
    }

    /// Whether `value`, which is at `at`, is also `offset` bytes before.
    #[inline(always)]
    pub(super) fn repeats(&self, at: u32, offset: u32, value: u32) -> bool {
        let back = at.wrapping_sub(offset);
        if back < at {
            u32_at(self.history.prefix, back) == Some(value)
        } else {
            core::hint::cold_path();
            offset != 0 && u32_at(self.all.prefix, back.wrapping_add(self.start)) == Some(value)
        }
    }

    #[inline(always)]
    pub(super) fn repeat_len(&self, at: u32, offset: u32) -> u32 {
        let after = at + self.start + 4;
        4 + self.all.match_len(after, after.wrapping_sub(offset))
    }

    #[inline(always)]
    fn repeat_len_before(&self, at: u32, offset: u32) -> u32 {
        let at = at + self.start;
        self.all.match_len_before(at, at.wrapping_sub(offset), 1)
    }
}
