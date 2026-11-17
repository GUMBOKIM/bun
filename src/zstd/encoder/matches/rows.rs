//! Rows of the most recent positions with about the same hash. Tags have more bits of it.

use core::hint::select_unpredictable;
use core::ops::Range;

use super::hash::Hasher;
use super::lazy::{Finder, Longest, Match};
use super::{Attached, Bounds, MatchState, Reach, WORD, prefetch};
use crate::array::At;
use crate::bytes::{equal_bytes, u64_at};
use crate::parameters::TAG_BITS;
use crate::window::{FIRST_INDEX, History};

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

/// One bit for each of `tags` that equals `tag`, from the most recent entry on.
#[inline(always)]
fn matching<const N: usize>(tags: &[u8; N], tag: u8, head: u32) -> u64 {
    let mut all = 0;
    for (index, chunk) in tags.as_chunks().0.iter().enumerate() {
        all |= u64::from(equal_bytes(chunk, &[tag; 16])) << (16 * index);
    }
    match N {
        16 => u64::from((all as u16).rotate_right(head)),
        32 => u64::from((all as u32).rotate_right(head)),
        _ => all.rotate_right(head),
    }
}

/// In place of the first tag is the head: where the most recent entry is. They are added backwards.
#[inline(always)]
fn add<const N: usize>((indices, tags): (&mut [u32; N], &mut [u8; N]), index: u32, tag: u8) {
    let slot = u32::from(*tags.at(0)).wrapping_sub(1) as usize % N;
    let slot = select_unpredictable(slot == 0, N - 1, slot) % N;
    *tags.at_mut(0) = slot as u8;
    *tags.at_mut(slot) = tag;
    *indices.at_mut(slot) = index;
}

/// Fills `candidates` with entries from `lowest` on, and returns those that it filled.
#[inline(always)]
fn collect<'a, const N: usize>(
    (indices, tags): (&[u32; N], &[u8; N]),
    tag: u8,
    lowest: u32,
    candidates: &'a mut [u32],
    prefetch: impl Fn(u32),
) -> &'a [u32] {
    let head = u32::from(*tags.at(0));
    let mut matching = matching(tags, tag, head);
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
        prefetch(index);
        *candidate = index;
        count += 1;
    }
    candidates.get(..count).unwrap_or_default()
}

#[inline(always)]
fn prefetch_row<const N: usize>(indices: &[[u32; N]], tags: &[[u8; N]], hash: u32) {
    let first = (hash >> TAG_BITS) * N as u32;
    prefetch(indices.as_flattened(), first);
    if N >= 32 {
        prefetch(indices.as_flattened(), first + 16);
    }
    prefetch(tags.as_flattened(), first);
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
    indices: &'a mut [[u32; N]],
    tags: &'a mut [[u8; N]],
    hasher: Hasher,
    attached_indices: &'a [[u32; N]],
    attached_tags: &'a [[u8; N]],
    attached_hasher: Hasher,
    attached_delta: u32,
    attempts: usize,
    next_to_index: u32,
}

impl<'a, const N: usize> Rows<'a, N> {
    pub(super) fn new(state: &'a mut MatchState, attached: &Attached<'a>) -> Self {
        let (tuning, bounds) = (state.tuning, state.bounds());
        let hasher =
            |hash_log: u32| Hasher::new(tuning.min_match.min(6), hash_log - N.ilog2() + TAG_BITS);
        let tables = state.tables.split_mut();
        Self {
            bounds,
            indices: tables.hash.as_chunks_mut().0,
            tags: tables.tags.as_chunks_mut().0,
            hasher: hasher(tuning.hash_log),
            attached_indices: attached.hash.as_chunks().0,
            attached_tags: attached.tags.as_chunks().0,
            attached_hasher: hasher(attached.tuning.hash_log),
            attached_delta: attached.delta,
            attempts: 1 << tuning.search_log,
            next_to_index: state.next_to_index,
        }
    }

    #[inline(always)]
    fn row(&mut self, hash: u32) -> Option<(&mut [u32; N], &mut [u8; N])> {
        let row = (hash >> TAG_BITS) as usize;
        Some((self.indices.get_mut(row)?, self.tags.get_mut(row)?))
    }

    fn fill(&mut self, history: &History<'_>, end: u32) -> Option<()> {
        for position in history.position(self.next_to_index)..end {
            let hash = self.hasher.hash(u64_at(history.prefix, position)?) as u32;
            add(self.row(hash)?, history.index(position), hash as u8);
        }
        Some(())
    }

    /// Of the `word` at `position`. Returns what was cached in its place.
    #[inline(always)]
    fn cache_hash(&self, hashes: &mut [u32; CACHED], position: u32, word: u64) -> u32 {
        let hash = self.hasher.hash(word) as u32;
        prefetch_row(self.indices, self.tags, hash);
        core::mem::replace(hashes.at_mut(position as usize % CACHED), hash)
    }

    #[inline(always)]
    fn fill_cache(&self, history: &History<'_>, hashes: &mut [u32; CACHED], from: u32) {
        for position in from..from + CACHED as u32 {
            if let Some(word) = u64_at(history.prefix, position) {
                self.cache_hash(hashes, position, word);
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
        for position in positions {
            let ahead = u64_at(history.prefix, position + CACHED as u32)?;
            let hash = self.cache_hash(hashes, position, ahead);
            add(self.row(hash)?, history.index(position), hash as u8);
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
    fn find<const REACH: Reach>(
        &mut self,
        history: &History<'_>,
        Scratch { hashes, candidates }: &mut Scratch<N>,
        at: u32,
        is_skipping: bool,
    ) -> Option<Match> {
        let current = history.index(at);
        let word = u64_at(history.prefix, at)?;
        let attached_hash = self.attached_hasher.hash(word) as u32;
        if REACH == Reach::Attached {
            prefetch_row(self.attached_indices, self.attached_tags, attached_hash);
        }

        let hash = if is_skipping {
            self.hasher.hash(word) as u32
        } else {
            self.insert_up_to(history, hashes, at)?;
            self.cache_hash(hashes, at, u64_at(history.prefix, at + CACHED as u32)?)
        };
        let (lowest, attempts) = (self.bounds.lowest(current), self.attempts.min(N));
        let (indices, tags) = self.row(hash)?;
        let found = collect(
            (indices, tags),
            hash as u8,
            lowest,
            candidates.get_mut(..attempts)?,
            |index| match REACH {
                Reach::Older => {
                    let (run, at) = history.locate(index);
                    prefetch(run, at);
                }
                _ => prefetch(history.prefix, history.position(index)),
            },
        );
        add((indices, tags), current, hash as u8);
        self.next_to_index = current + 1;

        let mut longest = Longest::new(at, word);
        for &index in found {
            if longest.consider::<REACH>(history, index).is_none() {
                break;
            }
        }

        if REACH == Reach::Attached {
            let row = (attached_hash >> TAG_BITS) as usize;
            let attempts = attempts - found.len();
            let found = collect(
                (
                    self.attached_indices.get(row)?,
                    self.attached_tags.get(row)?,
                ),
                attached_hash as u8,
                FIRST_INDEX,
                candidates.get_mut(..attempts)?,
                |index| prefetch(history.older, Attached::position(index)),
            );
            for &index in found {
                let index = index.wrapping_add(self.attached_delta);
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
