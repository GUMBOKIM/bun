//! Each position links to the last one with the same hash.

use super::hash::Hasher;
use super::lazy::{Finder, Longest, Match};
use super::{Attached, Bounds, MatchState, Reach, WORD};
use crate::bytes::{likely, u64_at};
use crate::window::{FIRST_INDEX, History, WORD_LEN};

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
    mask: u32,
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
            mask: (1 << tuning.chain_log) - 1,
            chain: tables.chain,
            hasher: Hasher::new(min_match, tuning.hash_log),
            attached_hasher: Hasher::new(min_match, attached.tuning.hash_log),
            attempts: 1 << tuning.search_log,
            next_to_index: state.next_to_index,
        }
    }

    #[inline(always)]
    fn insert_up_to(&mut self, history: &History<'_>, end: u32, is_skipping: bool) -> Option<()> {
        let start = history.position(self.next_to_index);
        let inserted_end = if is_skipping { end.min(start + 1) } else { end } as usize;
        let bytes = likely(history.prefix.get(..inserted_end + WORD_LEN))?;
        for position in start as usize..inserted_end {
            let word = bytes.get(position..position + WORD_LEN)?.as_array()?;
            let hash = self.hasher.hash(u64::from_le_bytes(*word));
            let (head, index) = (
                likely(self.heads.get_mut(hash))?,
                history.index(position as u32),
            );
            *self.chain.get_mut((index & self.mask) as usize)? = core::mem::replace(head, index);
        }
        self.next_to_index = history.index(end);
        Some(())
    }
}

impl Finder for Chains<'_> {
    const MARGIN: u32 = WORD;
    type Scratch = ();

    #[inline(always)]
    fn prepare(&mut self) -> Option<()> {
        self.chain = core::mem::take(&mut self.chain).get_mut(..self.mask as usize + 1)?;
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
            index = *self.chain.get((index & self.mask) as usize)?;
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
