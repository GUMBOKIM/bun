//! A hash table for matches of 8 bytes, which are preferred, and one for shorter ones.

use core::cell::Cell;
use core::convert::Infallible;

use bun_zstd_decoder::with_bmi2;

use super::fast::{FILL_STEP, Near, look_up_tagged, tagged};
use super::hash::Hasher;
use super::{Attached, MatchState, Owner, Progress, Reach, SKIP_LOG, cells, prefetch};
use crate::bytes::{AtLeast, chunk_at, likely, u32_at, u64_at, word_in};
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
    let mut searcher = Searcher {
        history: *history,
        long_table: cells(tables.hash),
        short_table: cells(tables.chain),
        long_hasher: Hasher::new(8, tuning.hash_log),
        short_hasher: Hasher::new(tuning.min_match, tuning.chain_log),
    };

    if REACH == Reach::Attached {
        let mut progress = Progress::new(sequences, history.start, *repeats, u32::MAX);
        searcher.search_with_attached(attached, &mut progress);
        return progress.finish(repeats, u32::MAX);
    }

    let lowest = bounds.lowest(end);
    if REACH == Reach::Older && lowest < history.prefix_index {
        searcher.history = history.from_older(lowest);
        let mut progress = Progress::new(sequences, history.start, *repeats, u32::MAX);
        let progress_mut = &mut progress;
        with_bmi2(
            #[inline(always)]
            move || searcher.search_with_older(progress_mut),
        );
        return progress.finish(repeats, u32::MAX);
    }

    let near = Near::new(history, bounds.lowest_in_prefix(end));
    let first = near.history.index(near.history.start);
    let max_offset = first - bounds.lowest_in_prefix(first);
    let anchor = history.start - near.start;
    let mut progress = Progress::new(sequences, anchor, *repeats, max_offset);
    searcher.search_prefix(&near, &mut progress);
    near.start + progress.finish(repeats, max_offset)
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
        likely(self.long_table.get(self.long_hasher.hash(word)))
    }

    #[inline(always)]
    fn short_entry(&self, word: u64) -> Option<&'a Cell<u32>> {
        likely(self.short_table.get(self.short_hasher.hash(word)))
    }

    /// Some positions of a match that was found at `searched` and ends at `end`.
    #[inline(always)]
    fn insert_match(&self, searched: u32, end: u32) -> Option<()> {
        let history = &self.history;
        let before_end = likely(chunk_at::<{ WORD_LEN + 2 }>(
            history.prefix,
            end.wrapping_sub(2),
        ))?;
        let inside = likely(u64_at(history.prefix, searched + 2))?;
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
    fn search_prefix(self, near: &Near<'a>, progress: &mut Progress<'_>) -> Option<Infallible> {
        with_bmi2(
            #[inline(always)]
            move || self.scan_prefix(near, progress),
        )
    }

    #[inline(always)]
    fn scan_prefix(mut self, near: &Near<'a>, progress: &mut Progress<'_>) -> Option<Infallible> {
        const STEP_GROWS_AFTER: usize = 1 << SKIP_LOG;
        let chunks = AtLeast::<WORD_LEN>::new(near.history.prefix)?;
        self.history = near.history;
        self.history.prefix = chunks.bytes;
        let history = &self.history;
        let index = |at: usize| history.index(at as u32);
        let position = |entry: &Cell<u32>| history.position(entry.get()) as usize;
        let match_len = |at: usize, earlier: usize, known: u32| {
            known + history.match_len(at as u32 + known, earlier as u32 + known)
        };
        let mut at = history.start as usize;
        let mut recent = near.distance(progress.recent);

        loop {
            let mut step = 1;
            let mut next_step = at + STEP_GROWS_AFTER;
            let mut ahead = at + step;
            let mut word = likely(chunks.word(at))?;
            let mut long_entry = self.long_entry(word)?;
            let mut long = position(long_entry);

            'stored: {
                let (found, earlier, len, ahead_entry) = loop {
                    let ahead_word = likely(chunks.word(ahead))?;
                    let short_entry = self.short_entry(word)?;
                    let short = position(short_entry);
                    long_entry.set(index(at));
                    short_entry.set(index(at));

                    if near.has_before(chunks, at + 1, recent, (word >> 8) as u32) {
                        let len = near.repeat_len(at as u32 + 1, recent as u32);
                        progress.store(at as u32 + 1, OffsetValue::FIRST_REPEAT, len);
                        break 'stored;
                    }

                    let ahead_entry = self.long_entry(ahead_word)?;
                    if chunks.matches_at::<true>(long, |it| it == word) {
                        break (at, long, match_len(at, long, 8), ahead_entry);
                    }

                    let ahead_long = position(ahead_entry);
                    if chunks.matches_at::<true>(short, |it| it as u32 == word as u32) {
                        let len = match_len(at, short, 4);
                        if ahead_long != 0 && chunks.word(ahead_long) == Some(ahead_word) {
                            let ahead_len = match_len(ahead, ahead_long, 8);
                            if ahead_len > len {
                                break (ahead, ahead_long, ahead_len, ahead_entry);
                            }
                        }
                        break (at, short, len, ahead_entry);
                    }

                    if ahead >= next_step {
                        prefetch(chunks.bytes, ahead + 64);
                        prefetch(chunks.bytes, ahead + 128);
                        step += 1;
                        next_step += STEP_GROWS_AFTER;
                    }
                    (at, word, long_entry, long) = (ahead, ahead_word, ahead_entry, ahead_long);
                    ahead += step;
                    #[cfg(target_arch = "aarch64")]
                    prefetch(chunks.bytes, at + 256);
                };

                // Only if the search is certain to go on after it.
                if step < 4 {
                    ahead_entry.set(index(ahead));
                }
                let (found, earlier) = (found as u32, earlier as u32);
                let max = earlier.min(found - progress.anchor);
                let before = history.match_len_before(found, earlier, max);
                let offset = found.wrapping_sub(earlier);
                progress.note_offset(offset);
                progress.store(found - before, OffsetValue::new(offset), before + len);
                recent = offset as usize;
            }

            self.insert_match(at as u32, progress.anchor)?;
            loop {
                at = progress.anchor as usize;
                let word = likely(chunks.word(at))?;
                if !near.has_before(chunks, at, near.distance(progress.second), word as u32) {
                    break;
                }
                let len = near.repeat_len(at as u32, progress.second);
                progress.swap_offsets();
                recent = progress.recent as usize;
                self.short_entry(word)?.set(index(at));
                self.long_entry(word)?.set(index(at));
                progress.store(at as u32, OffsetValue::FIRST_REPEAT, len);
            }
        }
    }

    /// Stores a match that the search at `searched` found, and what repeats after it.
    #[inline(always)]
    fn take_match(&self, searched: u32, found: Match, progress: &mut Progress<'_>) -> Option<()> {
        let history = &self.history;
        let (offset, before) = if found.repeats {
            (OffsetValue::FIRST_REPEAT, 0)
        } else {
            progress.note_offset(history.index(found.at).wrapping_sub(found.earlier));
            let max = found.at - progress.anchor;
            let before = history.match_len_before_index(found.at, found.earlier, max);
            (OffsetValue::new(progress.recent), before)
        };
        let len = before + history.match_len_at_index(found.at, found.earlier, found.known);
        progress.store(found.at - before, offset, len);

        self.insert_match(searched, progress.anchor)?;
        loop {
            let at = progress.anchor;
            let word = likely(u64_at(history.prefix, at))?;
            let current = history.index(at);
            let earlier = current.wrapping_sub(progress.second);
            if history.u32_at_index(earlier) != Some(word as u32) || progress.second == 0 {
                return Some(());
            }
            let len = history.match_len_at_index(at, earlier, 4);
            progress.swap_offsets();
            self.short_entry(word)?.set(current);
            self.long_entry(word)?.set(current);
            progress.store(at, OffsetValue::FIRST_REPEAT, len);
        }
    }

    #[inline(always)]
    fn search_with_older(mut self, progress: &mut Progress<'_>) -> Option<Infallible> {
        self.history = self.history.with_prefix_of::<WORD_LEN>()?;
        let history = &self.history;
        let (lowest, mut at) = (history.older_index(), history.start);
        loop {
            let (word, next_word) = adjacent_words(history.prefix, at)?;
            let (short_entry, long_entry) = (self.short_entry(word)?, self.long_entry(word)?);
            let current = history.index(at);
            let (short, long) = (short_entry.replace(current), long_entry.replace(current));
            let back = (current + 1).wrapping_sub(progress.recent);

            let found =
                if history.u32_at_index(back) == Some(next_word as u32) && progress.recent != 0 {
                    Match::repeat(at + 1, back)
                } else if history.u64_at_index(long) == Some(word) && long != lowest {
                    Match::new(at, long, 8)
                } else if history.u32_at_index(short) == Some(word as u32) && short != lowest {
                    let long = self.long_entry(next_word)?.replace(current + 1);
                    if history.u64_at_index(long) == Some(next_word) && long != lowest {
                        Match::new(at + 1, long, 8)
                    } else {
                        Match::new(at, short, 4)
                    }
                } else {
                    at += ((at - progress.anchor) >> SKIP_LOG) + 1;
                    continue;
                };
            self.take_match(at, found, progress)?;
            at = progress.anchor;
        }
    }

    fn search_with_attached(
        mut self,
        attached: &Attached<'_>,
        progress: &mut Progress<'_>,
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

            let found =
                if history.u32_at_index(back) == Some(next_word as u32) && progress.recent != 0 {
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
                    prefetch(history.prefix, at as usize + 256);
                    continue;
                };
            self.take_match(at, found, progress)?;
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
