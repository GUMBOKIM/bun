//! Rows of the most recent positions with about the same hash. Tags have more bits of it.

use core::ops::Range;

use super::hash::Hasher;
use super::lazy::{Finder, Longest, Match};
use super::{Attached, Bounds, MatchState, Reach, WORD, prefetch};
use crate::array::At;
use crate::bytes::{equal_bytes, likely, u64_at};
use crate::parameters::TAG_BITS;
use crate::window::{FIRST_INDEX, History, WORD_LEN};

const CACHED: usize = 8;

/// Indexes the history up to `end`.
pub(super) fn fill(state: &mut MatchState, history: &History<'_>, end: u32) {
    let none = Attached::empty(state.tuning);
    match state.tuning.row_log() {
        4 => Rows::<16>::new(state, &none).fill(history, end),
        5 => Rows::<32>::new(state, &none).fill(history, end),
        _ => Rows::<64>::new(state, &none).fill(history, end),
    };
}

/// The head of a row, and from there on one bit for each of its `tags` that equals `tag`.
#[inline(always)]
fn matching<const N: usize>(tags: &[u8; N], tag: u8) -> (u32, u64) {
    let mut all = 0;
    for (index, chunk) in tags.as_chunks().0.iter().enumerate() {
        all |= u64::from(equal_bytes(chunk, &[tag; 16])) << (16 * index);
    }
    // Or else the compiler would put the first chunk together from the head and the rest.
    core::hint::black_box(());
    let head = u32::from(*tags.at(0));
    let rotated = match N {
        16 => u64::from((all as u16).rotate_right(head)),
        32 => u64::from((all as u32).rotate_right(head)),
        _ => all.rotate_right(head),
    };
    (head, rotated)
}

/// Where the row for `hash` starts.
#[inline(always)]
fn first_entry<const N: usize>(hash: u32) -> usize {
    (hash >> TAG_BITS) as usize * N
}

#[inline(always)]
fn prefetch_row<const N: usize>(indices: &[u32], tags: &[u8], hash: u32) {
    let first = first_entry::<N>(hash);
    prefetch(indices, first);
    if N >= 32 {
        prefetch(indices, first + 16);
    }
    prefetch(tags, first);
}

/// Fills `candidates` with what `note` makes of the indices from `lowest` on, and returns those that it filled.
#[inline(always)]
fn collect<'a, const N: usize>(
    indices: &[u32],
    tags: &[u8],
    hash: u32,
    lowest: u32,
    candidates: &'a mut [u32],
    note: impl Fn(u32) -> u32,
) -> Option<&'a [u32]> {
    let row = first_entry::<N>(hash)..first_entry::<N>(hash) + N;
    let indices = likely(indices.get(row.clone()))?.as_array::<N>()?;
    let tags = likely(tags.get(row))?.as_array::<N>()?;
    let (head, mut matching) = matching(tags, hash as u8);
    let mut count = 0;
    while matching != 0
        && let Some(candidate) = candidates.get_mut(count)
    {
        let slot = (head + matching.trailing_zeros()) as usize % N;
        matching &= matching - 1;
        if slot == 0 {
            continue;
        }
        let index = *indices.at(slot);
        if index < lowest {
            break;
        }
        *candidate = note(index);
        count += 1;
    }
    candidates.get(..count)
}

pub(super) struct Scratch<const N: usize> {
    /// The hashes of the next positions, whose rows are being fetched.
    hashes: [u32; CACHED],
    candidates: [u32; N],
}

impl<const N: usize> Default for Scratch<N> {
    #[inline(always)]
    fn default() -> Self {
        Self {
            hashes: [0; CACHED],
            candidates: [0; N],
        }
    }
}

pub(super) struct Rows<'a, const N: usize> {
    bounds: Bounds,
    indices: &'a mut [u32],
    /// In place of the first one of a row is its head: where the most recent entry is. They are added backwards.
    tags: &'a mut [u8],
    hasher: Hasher,
    attached: Attached<'a>,
    attached_hasher: Hasher,
    attempts: usize,
    next_to_index: u32,
}

impl<'a, const N: usize> Rows<'a, N> {
    /// A comparison in its place becomes a branch, which is hard to predict.
    #[expect(clippy::indexing_slicing, reason = "evaluated at compile time")]
    const NEXT_HEADS: &'static [u8; 256] = &{
        let (mut next, mut head) = ([0; 256], 0);
        while head < next.len() {
            next[head] = if head % N > 1 { head % N - 1 } else { N - 1 } as u8;
            head += 1;
        }
        next
    };

    pub(super) fn new(state: &'a mut MatchState, attached: &Attached<'a>) -> Self {
        let (tuning, bounds) = (state.tuning, state.bounds());
        let hasher =
            |hash_log: u32| Hasher::new(tuning.min_match.min(6), hash_log - N.ilog2() + TAG_BITS);
        let tables = state.tables.split_mut();
        Self {
            bounds,
            indices: tables.hash,
            tags: tables.tags,
            hasher: hasher(tuning.hash_log),
            attached: *attached,
            attached_hasher: hasher(attached.tuning.hash_log),
            attempts: 1 << tuning.search_log,
            next_to_index: state.next_to_index,
        }
    }

    #[inline(always)]
    fn add(&mut self, hash: u32, index: u32) -> Option<()> {
        let row = first_entry::<N>(hash)..first_entry::<N>(hash) + N;
        let tags = likely(self.tags.get_mut(row.clone()))?.as_mut_array::<N>()?;
        let indices = likely(self.indices.get_mut(row))?.as_mut_array::<N>()?;
        let head = *Self::NEXT_HEADS.at(usize::from(*tags.at(0)));
        let slot = usize::from(head) % N;
        *tags.at_mut(0) = head;
        *tags.at_mut(slot) = hash as u8;
        *indices.at_mut(slot) = index;
        Some(())
    }

    fn fill(&mut self, history: &History<'_>, end: u32) -> Option<()> {
        for position in history.position(self.next_to_index)..end {
            let hash = self.hasher.hash(u64_at(history.prefix, position)?) as u32;
            self.add(hash, history.index(position))?;
        }
        Some(())
    }

    /// Of the `word` at `position`. Returns what was cached in its place.
    #[inline(always)]
    fn cache_hash(&self, hashes: &mut [u32; CACHED], position: usize, word: u64) -> u32 {
        let hash = self.hasher.hash(word) as u32;
        prefetch_row::<N>(self.indices, self.tags, hash);
        core::mem::replace(hashes.at_mut(position % CACHED), hash)
    }

    #[inline(always)]
    fn fill_cache(&self, history: &History<'_>, hashes: &mut [u32; CACHED], from: u32) {
        for position in from..from + CACHED as u32 {
            if let Some(word) = u64_at(history.prefix, position) {
                self.cache_hash(hashes, position as usize, word);
            }
        }
    }

    #[inline(always)]
    fn insert(
        &mut self,
        history: &History<'_>,
        hashes: &mut [u32; CACHED],
        positions: Range<u32>,
    ) -> Option<()> {
        let (start, end) = (positions.start as usize, positions.end as usize);
        let ahead = likely(history.prefix.get(CACHED..end + CACHED + WORD_LEN))?;
        for position in start..end {
            let word = ahead.get(position..position + WORD_LEN)?.as_array()?;
            let hash = self.cache_hash(hashes, position, u64::from_le_bytes(*word));
            self.add(hash, history.index(position as u32))?;
        }
        Some(())
    }

    #[inline(always)]
    fn insert_up_to(
        &mut self,
        history: &History<'_>,
        hashes: &mut [u32; CACHED],
        end: u32,
    ) -> Option<()> {
        const SKIPS_FROM: u32 = 384;
        const KEPT_AT_START: u32 = 96;
        const KEPT_AT_END: u32 = 32;
        let mut start = history.position(self.next_to_index);
        if end.wrapping_sub(start) > SKIPS_FROM {
            core::hint::cold_path();
            self.insert(history, hashes, start..start + KEPT_AT_START)?;
            start = end - KEPT_AT_END;
            self.fill_cache(history, hashes, start);
        }
        self.insert(history, hashes, start..end)
    }
}

impl<const N: usize> Finder for Rows<'_, N> {
    const MARGIN: u32 = WORD + CACHED as u32;
    type Scratch = Scratch<N>;

    #[inline(always)]
    fn prepare(&mut self) -> Option<()> {
        // The compiler sees that both have the same length, and compares the start of a row with `last`.
        let last = self.tags.len().checked_sub(N)? as u32 as usize;
        self.tags = core::mem::take(&mut self.tags).get_mut(..last + N)?;
        self.indices = core::mem::take(&mut self.indices).get_mut(..last + N)?;
        Some(())
    }

    #[inline(always)]
    fn find<const REACH: Reach>(
        &mut self,
        history: &History<'_>,
        Scratch { hashes, candidates }: &mut Scratch<N>,
        at: u32,
        is_skipping: bool,
    ) -> Option<Match> {
        let current = history.index(at);
        let word = likely(u64_at(history.prefix, at))?;
        let (attached, attached_hash) = (self.attached, self.attached_hasher.hash(word) as u32);
        if REACH == Reach::Attached {
            prefetch_row::<N>(attached.hash, attached.tags, attached_hash);
        }

        let hash = if is_skipping {
            self.hasher.hash(word) as u32
        } else {
            self.insert_up_to(history, hashes, at)?;
            let ahead = likely(u64_at(history.prefix, at + CACHED as u32))?;
            self.cache_hash(hashes, at as usize, ahead)
        };
        let attempts = self.attempts.min(N);
        let found = collect::<N>(
            self.indices,
            self.tags,
            hash,
            self.bounds.lowest(current),
            candidates.get_mut(..attempts)?,
            |index| {
                let (run, at) = match REACH {
                    Reach::Older => history.locate(index),
                    _ => (history.prefix, history.position(index)),
                };
                prefetch(run, at as usize);
                if REACH == Reach::Older { index } else { at }
            },
        )?;
        self.add(hash, current)?;
        self.next_to_index = current + 1;

        let mut longest = Longest::new(at, word);
        for &found in found {
            let goes_on = match REACH {
                Reach::Older => longest.consider::<REACH>(history, found),
                _ => longest.consider_position(history, found),
            };
            if goes_on.is_none() {
                break;
            }
        }

        if REACH == Reach::Attached {
            let attempts = attempts - found.len();
            let found = collect::<N>(
                attached.hash,
                attached.tags,
                attached_hash,
                FIRST_INDEX,
                candidates.get_mut(..attempts)?,
                |index| {
                    prefetch(history.older, Attached::position(index) as usize);
                    index.wrapping_add(attached.delta)
                },
            )?;
            for &index in found {
                if longest.consider::<REACH>(history, index).is_none() {
                    break;
                }
            }
        }
        Some(longest.best)
    }

    #[inline(always)]
    fn resume(&mut self, history: &History<'_>, scratch: &mut Scratch<N>) {
        let from = history.position(self.next_to_index);
        self.fill_cache(history, &mut scratch.hashes, from);
    }

    fn next_to_index(&self) -> u32 {
        self.next_to_index
    }
}
