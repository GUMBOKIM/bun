//! One hash table, and the first match that comes up.

use core::cell::Cell;
use core::convert::Infallible;
use core::hint::select_unpredictable;

use bun_zstd_decoder::with_bmi2;

use super::hash::Hasher;
use super::{Attached, MatchState, Owner, Progress, Reach, SKIP_LOG, cells, prefetch};
use crate::bytes::{AtLeast, chunk_at, likely, u32_at, u64_at, word_in};
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
        let mut progress = Progress::new(sequences, history.start, *repeats, u32::MAX);
        searcher.search_with_attached(attached, &mut progress);
        return progress.finish(repeats, u32::MAX);
    }

    let lowest = bounds.lowest(end);
    if REACH == Reach::Older && lowest < history.prefix_index {
        let max_offset = history.index(history.start) - lowest - 1;
        let mut progress = Progress::new(sequences, history.start, *repeats, max_offset);
        let progress_mut = &mut progress;
        with_bmi2(
            #[inline(always)]
            move || searcher.search_with_older(lowest, progress_mut),
        );
        return progress.finish(repeats, max_offset);
    }

    let near = Near::new(history, bounds.lowest_in_prefix(end));
    let first = near.history.index(near.history.start);
    let max_offset = first - bounds.lowest_in_prefix(first);
    let anchor = history.start - near.start;
    let mut progress = Progress::new(sequences, anchor, *repeats, max_offset);
    // In a small window it is hard to predict whether a candidate is still in it.
    if tuning.window_log < 19 {
        searcher.search_prefix::<true>(&near, &mut progress);
    } else {
        searcher.search_prefix::<false>(&near, &mut progress);
    }
    near.start + progress.finish(repeats, max_offset)
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
        likely(self.table.get(self.hasher.hash(word)))
    }

    #[inline(always)]
    fn insert(&self, position: u32) -> Option<()> {
        self.entry(likely(u64_at(self.history.prefix, position))?)?
            .set(self.history.index(position));
        Some(())
    }

    #[inline(never)]
    fn search_prefix<const IS_BRANCHLESS: bool>(
        self,
        near: &Near<'a>,
        progress: &mut Progress<'_>,
    ) -> Option<Infallible> {
        with_bmi2(
            #[inline(always)]
            move || self.scan_prefix::<IS_BRANCHLESS>(near, progress),
        )
    }

    /// The steps for a position, which each wait for memory, overlap with those for the next ones.
    #[inline(always)]
    fn scan_prefix<const IS_BRANCHLESS: bool>(
        mut self,
        near: &Near<'a>,
        progress: &mut Progress<'_>,
    ) -> Option<Infallible> {
        const STEP_GROWS_AFTER: usize = 1 << (SKIP_LOG - 1);
        let chunks = AtLeast::<PAIR_LEN>::new(near.history.prefix)?;
        self.history = near.history;
        self.history.prefix = chunks.bytes;
        let history = &self.history;
        let index = |at: usize| history.index(at as u32);
        let position = |entry: &Cell<u32>| history.position(entry.get()) as usize;
        let has_at = |at, value| chunks.matches_at::<IS_BRANCHLESS>(at, |it| it as u32 == value);
        let mut at = history.start as usize;
        let mut recent = near.distance(progress.recent);

        loop {
            let mut step = self.step as usize + 1;
            let mut next_step = at + STEP_GROWS_AFTER;
            let mut ahead = at + step;
            let pair = likely(chunks.chunk(at))?;
            let mut word = word_in::<0, _>(pair);
            let mut entry = self.entry(word)?;
            let mut next_entry = self.entry(word_in::<1, _>(pair))?;
            let mut candidate = position(entry);

            let (searched, repeats) = loop {
                let ahead_pair = likely(chunks.chunk(ahead))?;
                let ahead_word = word_in::<0, _>(ahead_pair);
                let repeats = near.has_before(chunks, ahead, recent, ahead_word as u32);

                entry.set(index(at));
                if repeats {
                    next_entry.set(index(at + 1));
                    break (at, true);
                }
                if has_at(candidate, word as u32) {
                    next_entry.set(index(at + 1));
                    break (at, false);
                }

                candidate = position(next_entry);
                entry = self.entry(ahead_word)?;
                next_entry.set(index(at + 1));
                if has_at(candidate, (word >> 8) as u32) {
                    // Only if the search is certain to go on after it.
                    if step <= 4 {
                        entry.set(index(ahead));
                    }
                    break (at + 1, false);
                }

                candidate = position(entry);
                next_entry = self.entry(word_in::<1, _>(ahead_pair))?;
                (at, word) = (ahead, ahead_word);
                ahead += step;
                if ahead >= next_step {
                    step += 1;
                    prefetch(chunks.bytes, at + 1 + 64);
                    prefetch(chunks.bytes, at + 1 + 128);
                    next_step += STEP_GROWS_AFTER;
                }
            };

            if repeats {
                let (found, offset) = (ahead as u32, recent as u32);
                let before = near.repeat_len_before(found, offset);
                let len = before + near.repeat_len(found, offset);
                progress.store(found - before, OffsetValue::FIRST_REPEAT, len);
            } else {
                let (found, earlier) = (searched as u32, candidate as u32);
                let max = earlier.min(found - progress.anchor);
                let before = history.match_len_before(found, earlier, max);
                let len = before + 4 + history.match_len(found + 4, earlier + 4);
                let offset = found.wrapping_sub(earlier);
                progress.note_offset(offset);
                progress.store(found - before, OffsetValue::new(offset), len);
                recent = offset as usize;
            }

            at = progress.anchor as usize;
            let mut word = likely(chunks.word(at))?;
            for inside in [searched + 2, at - 2] {
                self.entry(likely(chunks.word(inside))?)?.set(index(inside));
            }
            while near.has_before(chunks, at, near.distance(progress.second), word as u32) {
                let len = near.repeat_len(at as u32, progress.second);
                progress.swap_offsets();
                self.entry(word)?.set(index(at));
                progress.store(at as u32, OffsetValue::FIRST_REPEAT, len);
                (at, recent) = (progress.anchor as usize, progress.recent as usize);
                word = likely(chunks.word(at))?;
            }
        }
    }

    /// Takes matches at the second most recent offset for as long as there are any.
    #[inline(always)]
    fn take_repeats(&self, progress: &mut Progress<'_>) -> Option<()> {
        let history = &self.history;
        loop {
            let at = progress.anchor;
            let word = likely(u64_at(history.prefix, at))?;
            let earlier = history.index(at).wrapping_sub(progress.second);
            if progress.second == 0 || history.u32_at_index(earlier) != Some(word as u32) {
                return Some(());
            }
            let len = history.match_len_at_index(at, earlier, 4);
            progress.swap_offsets();
            self.entry(word)?.set(history.index(at));
            progress.store(at, OffsetValue::FIRST_REPEAT, len);
        }
    }

    #[inline(always)]
    fn search_with_older(mut self, lowest: u32, progress: &mut Progress<'_>) -> Option<Infallible> {
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
            let pair: &Pair = likely(chunk_at(bytes, at))?;
            let mut word = word_in::<0, _>(pair);
            let mut entry = self.entry(word)?;
            let mut next_entry = self.entry(word_in::<1, _>(pair))?;
            let mut candidate = entry.get();

            // Also the position that is hashed but not yet in the table.
            let (searched, repeats, pending, pending_entry) = loop {
                let ahead_pair: &Pair = likely(chunk_at(bytes, ahead))?;
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
                    prefetch(bytes, at as usize + 1 + 64);
                    prefetch(bytes, at as usize + 1 + 128);
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
            progress.store(found - before, offset, len);

            if pending < progress.anchor {
                pending_entry.set(history.index(pending));
            }
            likely(u64_at(bytes, progress.anchor))?;
            self.insert(searched + 2)?;
            self.insert(progress.anchor - 2)?;
            self.take_repeats(progress)?;
        }
    }

    fn search_with_attached(
        mut self,
        attached: &Attached<'_>,
        progress: &mut Progress<'_>,
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
            let mut ahead_word = likely(u64_at(bytes, ahead))?;
            let mut word = likely(u64_at(bytes, at))?;
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
                if words.matches_at::<true>(history.position(candidate) as usize, |it| {
                    it as u32 == word as u32
                }) {
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
                ahead_word = likely(u64_at(bytes, ahead))?;
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
            progress.store(found - before, offset, len);

            likely(u64_at(bytes, progress.anchor))?;
            self.insert(at + 2)?;
            self.insert(progress.anchor - 2)?;
            self.take_repeats(progress)?;
        }
    }
}

/// The end of the prefix that is within reach of the whole block. One comparison tells whether a candidate is in it.
pub(super) struct Near<'a> {
    /// Starts at the first position that has something before it.
    pub(super) history: History<'a>,
    /// Offsets from earlier blocks can reach further.
    all: History<'a>,
    pub(super) start: u32,
}

impl<'a> Near<'a> {
    pub(super) fn new(all: &History<'a>, lowest: u32) -> Self {
        let start = all.position(lowest);
        let history = History {
            prefix: all.prefix.get(start as usize..).unwrap_or_default(),
            prefix_index: lowest,
            start: (all.start - start).max(1),
            ..*all
        };
        Self {
            history,
            all: *all,
            start,
        }
    }

    /// For [`has_before`](Self::has_before). Leads out of range if the offset is 0.
    #[inline(always)]
    pub(super) fn distance(&self, offset: u32) -> usize {
        select_unpredictable(offset != 0, offset as usize, 1 << 32)
    }

    /// Whether `value` is `distance` bytes before `at` of the near `chunks`.
    #[inline(always)]
    pub(super) fn has_before<const N: usize>(
        &self,
        chunks: AtLeast<'_, N>,
        at: usize,
        distance: usize,
        value: u32,
    ) -> bool {
        let back = at.wrapping_sub(distance);
        match likely(chunks.word(back)) {
            Some(word) => word as u32 == value,
            None => {
                let far = u32::try_from(back.wrapping_add(self.start as usize));
                far.is_ok_and(|far| u32_at(self.all.prefix, far) == Some(value))
            }
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
