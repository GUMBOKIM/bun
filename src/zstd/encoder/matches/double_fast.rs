//! A hash table for matches of 8 bytes, which are preferred, and one for shorter ones.

use core::cell::Cell;
use core::convert::Infallible;

use bun_zstd_decoder::with_bmi2;

use super::fast::{FILL_STEP, Near, look_up_tagged, tagged};
use super::hash::Hasher;
use super::{Attached, MatchState, Owner, Progress, Reach, SKIP_LOG, cells, prefetch};
use crate::bytes::{chunk_at, u32_at, u64_at, word_in};
use crate::parameters::TAG_BITS;
use crate::sequences::{OffsetValue, Repeats, Sequences};
use crate::window::{FIRST_INDEX, History, WORD_LEN};

/// Indexes the history up to its end.
pub(super) fn fill(state: &mut MatchState, history: &History<'_>, owner: Owner) {
    let tag_bits = if owner == Owner::Dictionary {
        TAG_BITS
    } else {
        0
    };
    let long_hasher = Hasher::new(8, state.tuning.hash_log + tag_bits);
    let short_hasher = Hasher::new(state.tuning.min_match, state.tuning.chain_log + tag_bits);
    let mut position = history.position(state.next_to_index);
    let tables = state.tables.split_mut();
    while let Some(chunk) = chunk_at::<{ WORD_LEN + FILL_STEP - 1 }>(history.prefix, position) {
        let index = history.index(position);
        for (step, word) in chunk.array_windows().enumerate() {
            let word = u64::from_le_bytes(*word);
            let (long_hash, short_hash) = (long_hasher.hash(word), short_hasher.hash(word));
            let (Some(long), Some(short)) = (
                tables.hash.get_mut(long_hash >> tag_bits),
                tables.chain.get_mut(short_hash >> tag_bits),
            ) else {
                return;
            };
            if owner == Owner::Frame {
                (*long, *short) = (index, index);
                break;
            }
            if step == 0 {
                *short = tagged(index, short_hash);
            }
            if step == 0 || *long == 0 {
                *long = tagged(index + step as u32, long_hash);
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
    let tables = state.tables.split_mut();
    let searcher = Searcher {
        history: *history,
        long_table: cells(tables.hash),
        short_table: cells(tables.chain),
        long_hasher: Hasher::new(8, tuning.hash_log),
        short_hasher: Hasher::new(tuning.min_match, tuning.chain_log),
    };

    if REACH == Reach::Attached {
        let mut progress = Progress::new(history.start, *repeats, u32::MAX);
        searcher.search_with_attached(attached, &mut progress, sequences);
        return progress.finish(repeats, u32::MAX);
    }

    let lowest = bounds.lowest(end);
    if REACH == Reach::Older && lowest < history.prefix_index {
        let progress = Progress::new(history.start, *repeats, u32::MAX);
        let progress = with_bmi2(
            #[inline(always)]
            move || {
                let mut progress = progress;
                searcher.search_with_older(lowest, &mut progress, sequences);
                progress
            },
        );
        return progress.finish(repeats, u32::MAX);
    }

    let lowest = history.position(bounds.lowest_in_prefix(end));
    let start = history.start + u32::from(history.start == lowest);
    let max_offset = history.index(start) - bounds.lowest_in_prefix(history.index(start));
    let progress = Progress::new(history.start, *repeats, max_offset);
    // In a small window it is hard to predict whether a candidate is still in it.
    let progress = if tuning.window_log < tuning.hash_log + 4 {
        searcher.search_prefix::<true>(lowest, start, progress, sequences)
    } else {
        searcher.search_prefix::<false>(lowest, start, progress, sequences)
    };
    progress.finish(repeats, max_offset)
}

#[derive(Clone, Copy)]
struct Searcher<'a> {
    history: History<'a>,
    long_table: &'a [Cell<u32>],
    short_table: &'a [Cell<u32>],
    long_hasher: Hasher,
    short_hasher: Hasher,
}

impl<'a> Searcher<'a> {
    #[inline(always)]
    fn long_entry(&self, word: u64) -> Option<&'a Cell<u32>> {
        self.long_table.get(self.long_hasher.hash(word))
    }

    #[inline(always)]
    fn short_entry(&self, word: u64) -> Option<&'a Cell<u32>> {
        self.short_table.get(self.short_hasher.hash(word))
    }

    /// Some positions of a match that was found at `searched` and ends at `end`.
    #[inline(always)]
    fn insert_match(&self, searched: u32, end: u32) -> Option<()> {
        let history = &self.history;
        let before_end = chunk_at::<{ WORD_LEN + 2 }>(history.prefix, end.wrapping_sub(2))?;
        let inside = u64_at(history.prefix, searched + 2)?;
        let (inside_index, end_index) = (history.index(searched + 2), history.index(end));
        self.long_entry(inside)?.set(inside_index);
        self.long_entry(word_in::<0, _>(before_end))?
            .set(end_index - 2);
        self.short_entry(inside)?.set(inside_index);
        self.short_entry(word_in::<1, _>(before_end))?
            .set(end_index - 1);
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

    #[inline(always)]
    fn scan_prefix<const IS_BRANCHLESS: bool>(
        mut self,
        lowest: u32,
        start: u32,
        progress: &mut Progress,
        sequences: &mut Sequences,
    ) -> Option<Infallible> {
        const STEP_GROWS_AFTER: u32 = 1 << SKIP_LOG;
        let near = Near::<WORD_LEN>::new(self.history, lowest)?;
        self.history = near.history;
        let (history, bytes) = (&self.history, self.history.prefix);
        let mut at = start - lowest;

        loop {
            let mut step = 1;
            let mut next_step = at + STEP_GROWS_AFTER;
            let mut ahead = at + step;
            let mut ahead_word = u64_at(bytes, ahead)?;
            let mut word = u64_at(bytes, at)?;
            let mut long_entry = self.long_entry(word)?;
            let mut long = history.position(long_entry.get());

            'stored: {
                let (found, earlier, len, ahead_entry) = loop {
                    let short_entry = self.short_entry(word)?;
                    let short = history.position(short_entry.get());
                    long_entry.set(history.index(at));
                    short_entry.set(history.index(at));

                    if near.repeats(at + 1, progress.recent, (word >> 8) as u32) {
                        let len = near.repeat_len(at + 1, progress.recent);
                        progress.store(sequences, at + 1, OffsetValue::FIRST_REPEAT, len);
                        break 'stored;
                    }

                    let ahead_entry = self.long_entry(ahead_word)?;
                    if near
                        .bytes
                        .matches_at::<IS_BRANCHLESS>(long, |it| it == word)
                    {
                        let len = 8 + history.match_len(at + 8, long + 8);
                        break (at, long, len, ahead_entry);
                    }

                    let ahead_long = history.position(ahead_entry.get());
                    if near
                        .bytes
                        .matches_at::<IS_BRANCHLESS>(short, |it| it as u32 == word as u32)
                    {
                        let len = 4 + history.match_len(at + 4, short + 4);
                        if ahead_long != 0 && u64_at(bytes, ahead_long) == Some(ahead_word) {
                            let ahead_len = 8 + history.match_len(ahead + 8, ahead_long + 8);
                            if ahead_len > len {
                                break (ahead, ahead_long, ahead_len, ahead_entry);
                            }
                        }
                        break (at, short, len, ahead_entry);
                    }

                    if ahead >= next_step {
                        prefetch(bytes, ahead + 64);
                        prefetch(bytes, ahead + 128);
                        step += 1;
                        next_step += STEP_GROWS_AFTER;
                    }
                    (at, word, long_entry, long) = (ahead, ahead_word, ahead_entry, ahead_long);
                    ahead += step;
                    #[cfg(target_arch = "aarch64")]
                    prefetch(bytes, at + 256);
                    ahead_word = u64_at(bytes, ahead)?;
                };

                let max = earlier.min(found - progress.anchor);
                let before = history.match_len_before(found, earlier, max);
                progress.note_offset(found.wrapping_sub(earlier));
                // Only if the search is certain to go on after it.
                if step < 4 {
                    ahead_entry.set(history.index(ahead));
                }
                let offset = OffsetValue::new(progress.recent);
                progress.store(sequences, found - before, offset, before + len);
            }

            self.insert_match(at, progress.anchor)?;
            loop {
                at = progress.anchor;
                let word = u64_at(bytes, at)?;
                if !near.repeats(at, progress.second, word as u32) {
                    break;
                }
                let len = near.repeat_len(at, progress.second);
                progress.swap_offsets();
                self.short_entry(word)?.set(history.index(at));
                self.long_entry(word)?.set(history.index(at));
                progress.store(sequences, at, OffsetValue::FIRST_REPEAT, len);
            }
        }
    }

    /// Stores a match that the search at `searched` found, and what repeats after it.
    #[inline(always)]
    fn take_match(
        &self,
        searched: u32,
        found: Match,
        lowest: u32,
        progress: &mut Progress,
        sequences: &mut Sequences,
    ) -> Option<()> {
        let history = &self.history;
        let (offset, before) = if found.repeats {
            (OffsetValue::FIRST_REPEAT, 0)
        } else {
            progress.note_offset(history.index(found.at).wrapping_sub(found.earlier));
            let max = (found.at - progress.anchor).min(found.earlier - lowest);
            let before = history.match_len_before_index(found.at, found.earlier, max);
            (OffsetValue::new(progress.recent), before)
        };
        let len = before + history.match_len_at_index(found.at, found.earlier, found.known);
        progress.store(sequences, found.at - before, offset, len);

        self.insert_match(searched, progress.anchor)?;
        loop {
            let at = progress.anchor;
            let word = u64_at(history.prefix, at)?;
            let current = history.index(at);
            let earlier = current.wrapping_sub(progress.second);
            if progress.second.wrapping_sub(1) >= current - lowest
                || history.u32_at_index(earlier) != Some(word as u32)
            {
                return Some(());
            }
            let len = history.match_len_at_index(at, earlier, 4);
            progress.swap_offsets();
            self.short_entry(word)?.set(current);
            self.long_entry(word)?.set(current);
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
        self.history = self.history.with_prefix_of::<WORD_LEN>()?;
        let history = &self.history;
        let mut at = history.start;
        loop {
            let (word, next_word) = adjacent_words(history.prefix, at)?;
            let (short_entry, long_entry) = (self.short_entry(word)?, self.long_entry(word)?);
            let current = history.index(at);
            let (short, long) = (short_entry.replace(current), long_entry.replace(current));
            let back = (current + 1).wrapping_sub(progress.recent);

            let found = if progress.recent.wrapping_sub(1) <= current - lowest
                && history.u32_at_index(back) == Some(next_word as u32)
            {
                Match::repeat(at + 1, back)
            } else if long > lowest && history.u64_at_index(long) == Some(word) {
                Match::new(at, long, 8)
            } else if short > lowest && history.u32_at_index(short) == Some(word as u32) {
                let long = self.long_entry(next_word)?.replace(current + 1);
                if long > lowest && history.u64_at_index(long) == Some(next_word) {
                    Match::new(at + 1, long, 8)
                } else {
                    Match::new(at, short, 4)
                }
            } else {
                at += ((at - progress.anchor) >> SKIP_LOG) + 1;
                continue;
            };
            self.take_match(at, found, lowest, progress, sequences)?;
            at = progress.anchor;
        }
    }

    fn search_with_attached(
        mut self,
        attached: &Attached<'_>,
        progress: &mut Progress,
        sequences: &mut Sequences,
    ) -> Option<Infallible> {
        self.history = self.history.with_prefix_of::<WORD_LEN>()?;
        let history = &self.history;
        let tuning = &attached.tuning;
        let long_hasher = Hasher::new(8, tuning.hash_log + TAG_BITS);
        let short_hasher = Hasher::new(tuning.min_match, tuning.chain_log + TAG_BITS);
        let find_long = |candidate: u32, word: u64| {
            if u64_at(history.prefix, history.position(candidate)) == Some(word) {
                return Some(candidate);
            }
            let index = look_up_tagged(attached.hash, long_hasher.hash(word))?;
            (index > FIRST_INDEX && u64_at(history.older, Attached::position(index)) == Some(word))
                .then(|| index.wrapping_add(attached.delta))
        };
        let find_short = |candidate: u32, word: u64| {
            let (run, at, index) = if candidate > history.prefix_index {
                (history.prefix, history.position(candidate), candidate)
            } else {
                let index = look_up_tagged(attached.chain, short_hasher.hash(word))
                    .filter(|&index| index > FIRST_INDEX)?;
                let at = Attached::position(index);
                (history.older, at, index.wrapping_add(attached.delta))
            };
            (u32_at(run, at) == Some(word as u32)).then_some(index)
        };

        let mut at = history.start;
        loop {
            let (word, next_word) = adjacent_words(history.prefix, at)?;
            let (short_entry, long_entry) = (self.short_entry(word)?, self.long_entry(word)?);
            let current = history.index(at);
            let (short, long) = (short_entry.replace(current), long_entry.replace(current));
            let back = (current + 1).wrapping_sub(progress.recent);

            let found = if history.u32_at_index(back) == Some(next_word as u32) {
                Match::repeat(at + 1, back)
            } else if let Some(long) = find_long(long, word) {
                Match::new(at, long, 8)
            } else if let Some(short) = find_short(short, word) {
                let long = self.long_entry(next_word)?.replace(current + 1);
                match find_long(long, next_word) {
                    Some(long) => Match::new(at + 1, long, 8),
                    None => Match::new(at, short, 4),
                }
            } else {
                at += ((at - progress.anchor) >> SKIP_LOG) + 1;
                #[cfg(target_arch = "aarch64")]
                prefetch(history.prefix, at + 256);
                continue;
            };
            self.take_match(at, found, history.older_index(), progress, sequences)?;
            at = progress.anchor;
        }
    }
}

#[inline(always)]
fn adjacent_words(bytes: &[u8], at: u32) -> Option<(u64, u64)> {
    let chunk = chunk_at::<{ WORD_LEN + 1 }>(bytes, at)?;
    Some((word_in::<0, _>(chunk), word_in::<1, _>(chunk)))
}

#[derive(Clone, Copy)]
struct Match {
    at: u32,
    earlier: u32,
    /// How many bytes were compared.
    known: u32,
    repeats: bool,
}

impl Match {
    #[inline(always)]
    fn new(at: u32, earlier: u32, known: u32) -> Self {
        Self {
            at,
            earlier,
            known,
            repeats: false,
        }
    }

    #[inline(always)]
    fn repeat(at: u32, earlier: u32) -> Self {
        Self {
            repeats: true,
            ..Self::new(at, earlier, 4)
        }
    }
}
