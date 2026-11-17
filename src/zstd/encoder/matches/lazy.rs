//! Takes a match only if none of the next positions has a better one.

use core::convert::Infallible;

use bun_zstd_decoder::with_bmi2;

use super::chain::Chains;
use super::lazy_tree::LazyTree;
use super::rows::Rows;
use super::{Attached, Bounds, MatchState, Progress, Reach, SKIP_LOG};
use crate::bytes::u32_at;
use crate::parameters::Strategy;
use crate::sequences::{OffsetValue, Repeats, Sequences};
use crate::window::{History, WORD_LEN};

const MIN_LEN: u32 = 4;

#[derive(Clone, Copy)]
pub(super) struct Match {
    pub(super) len: u32,
    pub(super) offset: OffsetValue,
}

impl Match {
    /// What to beat. The offset is about as costly as any.
    pub(super) const NONE: Self = Self {
        len: MIN_LEN - 1,
        offset: OffsetValue(999_999_999),
    };

    #[inline(always)]
    fn gain(self, per_byte: u32) -> i32 {
        (self.len * per_byte) as i32 - self.offset.code() as i32
    }
}

/// The longest of the matches for a position.
pub(super) struct Longest {
    at: u32,
    pub(super) best: Match,
    /// A longer match has these bytes, the last of which follows the best one.
    tail: u32,
}

impl Longest {
    #[inline(always)]
    pub(super) fn new(at: u32, word: u64) -> Self {
        Self {
            at,
            best: Match::NONE,
            tail: word as u32,
        }
    }

    /// `None` if the best match reaches the end, so that none can be longer.
    #[inline(always)]
    pub(super) fn consider<const REACH: Reach>(
        &mut self,
        history: &History<'_>,
        index: u32,
    ) -> Option<()> {
        let len = if REACH == Reach::Prefix || index >= history.prefix_index {
            let earlier = history.position(index);
            let tail = earlier.wrapping_add(self.best.len - 3);
            if u32_at(history.prefix, tail) != Some(self.tail) {
                return Some(());
            }
            history.match_len(self.at, earlier)
        } else {
            let earlier = history.older_position(index);
            if u32_at(history.older, earlier) != u32_at(history.prefix, self.at) {
                return Some(());
            }
            4 + history.older_match_len(self.at + 4, earlier + 4)
        };
        if len > self.best.len {
            let offset = OffsetValue::new(history.index(self.at).wrapping_sub(index));
            self.best = Match { len, offset };
            self.tail = u32_at(history.prefix, self.at + len - 3)?;
        }
        Some(())
    }
}

pub(super) trait Finder {
    /// How far from the end the search stops.
    const MARGIN: u32;

    /// Arrays would keep the compiler from holding the other fields in registers.
    type Scratch: Default;

    /// Also indexes the positions up to `at`, unless `is_skipping`. `None` ends the search.
    fn find<const REACH: Reach>(
        &mut self,
        history: &History<'_>,
        scratch: &mut Self::Scratch,
        at: u32,
        is_skipping: bool,
    ) -> Option<Match>;

    /// At the start, and after skipping.
    fn resume(&mut self, _history: &History<'_>, _scratch: &mut Self::Scratch) {}

    fn next_to_index(&self) -> u32;
}

pub(super) fn search<const REACH: Reach>(
    state: &mut MatchState,
    history: &History<'_>,
    attached: &Attached<'_>,
    sequences: &mut Sequences,
    repeats: &mut Repeats,
) -> u32 {
    let bounds = state.bounds();
    let is_first = REACH != Reach::Attached && history.start == 0;
    let start = history.start + u32::from(is_first);
    let max_offset = match REACH {
        Reach::Prefix => history.index(start) - bounds.lowest_in_prefix(history.index(start)),
        _ => u32::MAX,
    };
    let parser = Parser {
        history: *history,
        bounds,
        start,
        depth: (state.tuning.strategy as u32 - Strategy::Greedy as u32).min(2),
    };
    let mut progress = Progress::new(sequences, history.start, *repeats, max_offset);

    let next_to_index = if state.tuning.strategy == Strategy::TreeLazy2 {
        let finder = LazyTree::new(state, attached);
        parser.run::<REACH>(finder, &mut progress)
    } else if !state.uses_rows {
        let finder = Chains::new(state, attached);
        parser.run::<REACH>(finder, &mut progress)
    } else {
        match state.tuning.row_log() {
            4 => parser.run::<REACH>(Rows::<16>::new(state, attached), &mut progress),
            5 => parser.run::<REACH>(Rows::<32>::new(state, attached), &mut progress),
            _ => parser.run::<REACH>(Rows::<64>::new(state, attached), &mut progress),
        }
    };
    state.next_to_index = next_to_index;
    progress.finish(repeats, max_offset)
}

struct Parser<'a> {
    history: History<'a>,
    bounds: Bounds,
    start: u32,
    /// How many positions in a row have to have nothing better.
    depth: u32,
}

impl Parser<'_> {
    /// Returns from which index on the tables are not up to date.
    #[inline(never)]
    fn run<const REACH: Reach>(self, finder: impl Finder, progress: &mut Progress<'_>) -> u32 {
        with_bmi2(
            #[inline(always)]
            move || {
                let mut finder = finder;
                self.parse::<REACH, _>(&mut finder, progress);
                finder.next_to_index()
            },
        )
    }

    /// The length of the match that `offset` gives at `at`, if there is one.
    #[inline(always)]
    fn repeat_len<const REACH: Reach>(&self, at: u32, offset: u32) -> Option<u32> {
        let history = &self.history;
        let value = u32_at(history.prefix, at);
        if REACH == Reach::Prefix {
            let earlier = at.wrapping_sub(offset);
            return ((offset > 0) & (u32_at(history.prefix, earlier) == value))
                .then(|| 4 + history.match_len(at + 4, earlier + 4));
        }
        let current = history.index(at);
        let earlier = current.wrapping_sub(offset);
        let lowest = match REACH {
            Reach::Older => self.bounds.lowest(current),
            _ => 0,
        };
        (offset.wrapping_sub(1) < current - lowest && history.u32_at_index(earlier) == value)
            .then(|| history.match_len_at_index(at, earlier, 4))
    }

    #[inline(always)]
    fn parse<const REACH: Reach, F: Finder>(
        mut self,
        finder: &mut F,
        progress: &mut Progress<'_>,
    ) -> Option<Infallible> {
        const SKIPS_INDEXING_FROM: u32 = 8;
        self.history = self.history.with_prefix_of::<WORD_LEN>()?;
        let history = &self.history;
        let limit = history.end().checked_sub(F::MARGIN)?;
        let mut at = self.start;
        let mut is_skipping = false;
        let mut scratch = F::Scratch::default();
        finder.resume(history, &mut scratch);

        while at < limit {
            let mut start = at + 1;
            let repeat_len = self.repeat_len::<REACH>(start, progress.recent);
            let mut best = Match {
                len: repeat_len.unwrap_or(0),
                offset: OffsetValue::FIRST_REPEAT,
            };

            if self.depth > 0 || repeat_len.is_none() {
                let found = finder.find::<REACH>(history, &mut scratch, at, is_skipping)?;
                if found.len > best.len {
                    (best, start) = (found, at);
                }
                if best.len < MIN_LEN {
                    let skipped = (at - progress.anchor) >> SKIP_LOG;
                    at += skipped + 1;
                    is_skipping = skipped + u32::from(REACH != Reach::Older) > SKIPS_INDEXING_FROM;
                    continue;
                }

                let mut ahead = 0;
                while ahead < self.depth && at < limit {
                    at += 1;
                    if let Some(len) = self.repeat_len::<REACH>(at, progress.recent) {
                        let per_byte = 3 + ahead;
                        let offset = OffsetValue::FIRST_REPEAT;
                        if (len * per_byte) as i32 > best.gain(per_byte) + 1 {
                            (best, start) = (Match { len, offset }, at);
                        }
                    }
                    let found = finder.find::<REACH>(history, &mut scratch, at, is_skipping)?;
                    if found.len >= MIN_LEN && found.gain(4) > best.gain(4) + 4 + 3 * ahead as i32 {
                        (best, start) = (found, at);
                        ahead = 0;
                    } else {
                        ahead += 1;
                    }
                }

                if let Some(offset) = best.offset.offset() {
                    let max = start - progress.anchor;
                    let before = if REACH == Reach::Prefix {
                        history.match_len_before(start, start.wrapping_sub(offset), max)
                    } else {
                        let earlier = history.index(start).wrapping_sub(offset);
                        history.match_len_before_index(start, earlier, max)
                    };
                    start -= before;
                    best.len += before;
                    progress.note_offset(offset);
                }
            }

            progress.store(start, best.offset, best.len);
            at = progress.anchor;
            if is_skipping {
                finder.resume(history, &mut scratch);
                is_skipping = false;
            }

            while at <= limit
                && let Some(len) = self.repeat_len::<REACH>(at, progress.second)
            {
                progress.swap_offsets();
                progress.store(at, OffsetValue::FIRST_REPEAT, len);
                at = progress.anchor;
            }
        }
        None
    }
}
