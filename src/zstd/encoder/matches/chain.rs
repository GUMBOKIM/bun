//! Each position links to the last one with the same hash.

use super::hash::Hasher;
use super::lazy::{Finder, Longest, Match};
use super::{Attached, Bounds, MatchState, Reach, WORD};
use crate::bytes::{likely, u64_at};
use crate::window::{FIRST_INDEX, History};

/// Indexes the history up to `end`.
pub(super) fn fill(state: &mut MatchState, history: &History<'_>, end: u32) {
    let hasher = Hasher::new(state.tuning.min_match, state.tuning.hash_log);
    let none = Attached::empty(state.tuning);
    let mut chains = Chains::new(state, &none);
    chains.hasher = hasher;
    chains.insert_up_to(history, end, false);
}

pub(super) struct Chains<'a> {
    attached: Attached<'a>,
    bounds: Bounds,
    heads: &'a mut [u32],
    /// The compiler sees its length as `mask + 1`, and that a masked index is below it.
    chain: &'a mut [u32],
    mask: usize,
    hasher: Hasher,
    attached_hasher: Hasher,
    attempts: u32,
    next_to_index: u32,
}

impl<'a> Chains<'a> {
    pub(super) fn new(state: &'a mut MatchState, attached: &Attached<'a>) -> Self {
        let (tuning, bounds) = (state.tuning, state.bounds());
        let min_match = tuning.min_match.clamp(4, 6);
        let tables = state.tables.split_mut();
        Self {
            attached: *attached,
            bounds,
            heads: tables.hash,
            mask: tables.chain.len().wrapping_sub(1),
            chain: tables.chain,
            hasher: Hasher::new(min_match, tuning.hash_log),
            attached_hasher: Hasher::new(min_match, attached.tuning.hash_log),
            attempts: 1 << tuning.search_log,
            next_to_index: state.next_to_index,
        }
    }

    #[inline(always)]
    fn insert_up_to(&mut self, history: &History<'_>, end: u32, is_skipping: bool) -> Option<()> {
        let end = history.index(end);
        for index in self.next_to_index..end {
            let word = likely(u64_at(history.prefix, history.position(index)))?;
            let head = likely(self.heads.get_mut(self.hasher.hash(word)))?;
            *self.chain.get_mut(index as usize & self.mask)? = *head;
            *head = index;
            if is_skipping {
                break;
            }
        }
        self.next_to_index = end;
        Some(())
    }
}

impl Finder for Chains<'_> {
    const MARGIN: u32 = WORD;
    type Scratch = ();

    #[inline(always)]
    fn prepare(&mut self) -> Option<()> {
        self.mask = self.mask as u32 as usize;
        self.chain = core::mem::take(&mut self.chain).get_mut(..self.mask + 1)?;
        Some(())
    }

    #[inline(always)]
    fn find<const REACH: Reach>(
        &mut self,
        history: &History<'_>,
        (): &mut (),
        at: u32,
        is_skipping: bool,
    ) -> Option<Match> {
        self.insert_up_to(history, at, is_skipping)?;
        let word = likely(u64_at(history.prefix, at))?;
        let current = history.index(at);
        let lowest = self.bounds.lowest(current);
        let overwritten = current.saturating_sub(self.chain.len() as u32);
        let mut attempts = self.attempts;
        let mut longest = Longest::new(at, word);

        let mut index = *likely(self.heads.get(self.hasher.hash(word)))?;
        while index >= lowest && attempts > 0 {
            if longest.consider::<REACH>(history, index).is_none() || index <= overwritten {
                break;
            }
            index = *self.chain.get(index as usize & self.mask)?;
            attempts -= 1;
        }

        if REACH == Reach::Attached {
            let attached = &self.attached;
            let overwritten = attached.end.saturating_sub(attached.chain.len() as u32);
            let mut index = *attached.hash.get(self.attached_hasher.hash(word))?;
            while index >= FIRST_INDEX && attempts > 0 {
                let in_history = index.wrapping_add(attached.delta);
                if longest.consider::<REACH>(history, in_history).is_none() || index <= overwritten
                {
                    break;
                }
                index = *likely(
                    attached
                        .chain
                        .get(index as usize & (attached.chain.len() - 1)),
                )?;
                attempts -= 1;
            }
        }
        Some(longest.best)
    }

    fn next_to_index(&self) -> u32 {
        self.next_to_index
    }
}
