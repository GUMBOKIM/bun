//! For each hash a binary tree, sorted by the bytes that follow a position. The most recent position is the root.

use core::cell::Cell;

use super::hash::Hasher;
use super::{Attached, Bounds, MatchState, Reach, WORD, cells};
use crate::bytes::u64_at;
use crate::window::{FIRST_INDEX, History};

/// The roots of the subtrees with what is smaller, and with what is larger.
pub(super) type Node = [Cell<u32>; 2];

#[derive(Clone, Copy)]
pub(super) struct Nodes<'a>(&'a [Node]);

impl<'a> Nodes<'a> {
    pub(super) fn new(chain: &'a mut [u32]) -> Self {
        Self(cells(chain).as_chunks().0)
    }

    #[inline(always)]
    fn mask(self) -> usize {
        self.0.len().wrapping_sub(1)
    }

    #[inline(always)]
    pub(super) fn get(self, index: u32) -> Option<&'a Node> {
        self.0.get(index as usize & self.mask())
    }

    /// The last index whose node may have been taken by another one.
    #[inline(always)]
    pub(super) fn overwritten(self, current: u32) -> u32 {
        current.saturating_sub(self.mask() as u32)
    }
}

/// Makes a node the root, while going down the tree that it takes over.
pub(super) struct Descent<'a> {
    /// Where the next node goes that is smaller, and larger.
    slots: [&'a Cell<u32>; 2],
    /// How many bytes the last node that was smaller, and larger, had in common with the root.
    common: [u32; 2],
}

impl<'a> Descent<'a> {
    #[inline(always)]
    pub(super) fn new([smaller, larger]: &'a Node) -> Self {
        Self {
            slots: [smaller, larger],
            common: [0; 2],
        }
    }

    /// How many bytes the next node has in common with the root, at least.
    #[inline(always)]
    pub(super) fn known(&self) -> u32 {
        let [smaller, larger] = self.common;
        smaller.min(larger)
    }

    /// Returns the next index. `node` is that of `index`, or a spare one that leads nowhere.
    #[inline(always)]
    pub(super) fn step(&mut self, index: u32, node: &'a Node, len: u32, is_smaller: bool) -> u32 {
        let ([smaller_slot, larger_slot], [smaller, larger]) = (&mut self.slots, node);
        let [smaller_common, larger_common] = &mut self.common;
        if is_smaller {
            smaller_slot.set(index);
            (*smaller_slot, *smaller_common) = (larger, len);
            larger.get()
        } else {
            larger_slot.set(index);
            (*larger_slot, *larger_common) = (smaller, len);
            smaller.get()
        }
    }

    #[inline(always)]
    pub(super) fn finish(self) {
        for slot in self.slots {
            slot.set(0);
        }
    }
}

/// The length of the match at `index`, and whether what follows it is smaller there.
#[inline(always)]
pub(super) fn compare<const REACH: Reach>(
    history: &History<'_>,
    at: u32,
    index: u32,
    known: u32,
) -> (u32, bool) {
    let index = index.wrapping_add(known);
    let (len, is_smaller) = match REACH {
        Reach::Older => history.compare_index(at + known, index),
        _ => history.compare(at + known, history.position(index)),
    };
    (known + len, is_smaller)
}

/// Goes down the tree of a dictionary until `visit`, which gets an index of the history and a length, returns true.
#[inline(always)]
pub(super) fn descend_attached(
    attached: &Attached<'_>,
    history: &History<'_>,
    at: u32,
    hash: usize,
    attempts: u32,
    mut visit: impl FnMut(u32, u32) -> bool,
) -> Option<()> {
    let (nodes, _) = attached.chain.as_chunks::<2>();
    let mask = nodes.len().wrapping_sub(1);
    let overwritten = attached.end.saturating_sub(mask as u32).max(FIRST_INDEX);
    let (mut smaller_common, mut larger_common) = (0, 0);
    let mut index = *attached.hash.get(hash)?;
    for _ in 0..attempts {
        if index <= FIRST_INDEX {
            break;
        }
        let in_history = index.wrapping_add(attached.delta);
        let known = smaller_common.min(larger_common);
        let (len, is_smaller) = compare::<{ Reach::Older }>(history, at, in_history, known);
        if visit(in_history, len) || index <= overwritten {
            break;
        }
        let &[smaller, larger] = nodes.get(index as usize & mask)?;
        if is_smaller {
            (smaller_common, index) = (len, larger);
        } else {
            (larger_common, index) = (len, smaller);
        }
    }
    Some(())
}

/// Indexes the history up to `end`.
pub(super) fn fill(state: &mut MatchState, history: &History<'_>, end: u32) {
    let min_match = state.tuning.min_match;
    Tree::new(state, min_match).insert_up_to::<{ Reach::Prefix }>(history, end);
}

pub(super) struct Tree<'a> {
    pub(super) bounds: Bounds,
    pub(super) roots: &'a mut [u32],
    pub(super) nodes: Nodes<'a>,
    pub(super) hash3: &'a mut [u32],
    pub(super) hasher: Hasher,
    pub(super) attempts: u32,
    pub(super) next_to_index: u32,
}

impl<'a> Tree<'a> {
    pub(super) fn new(state: &'a mut MatchState, min_match: u32) -> Self {
        let (tuning, bounds) = (state.tuning, state.bounds());
        let tables = state.tables.split_mut();
        Self {
            bounds,
            roots: tables.hash,
            nodes: Nodes::new(tables.chain),
            hash3: tables.hash3,
            hasher: Hasher::new(min_match, tuning.hash_log),
            attempts: 1 << tuning.search_log,
            next_to_index: state.next_to_index,
        }
    }

    /// Returns how many positions that takes care of.
    #[inline(always)]
    fn insert<const REACH: Reach>(
        &mut self,
        history: &History<'_>,
        current: u32,
        lowest: u32,
    ) -> Option<u32> {
        let at = history.position(current);
        let root = self
            .roots
            .get_mut(self.hasher.hash(u64_at(history.prefix, at)?))?;
        let mut index = core::mem::replace(root, current);
        let overwritten = self.nodes.overwritten(current);
        let spare = Node::default();
        let mut descent = Descent::new(self.nodes.get(current)?);
        let mut matched_end = current + WORD + 1;
        let mut longest = WORD;

        for _ in 0..self.attempts {
            if index < lowest {
                break;
            }
            let node = self.nodes.get(index)?;
            let (len, is_smaller) = compare::<REACH>(history, at, index, descent.known());
            if len > longest {
                longest = len;
                matched_end = matched_end.max(index + len);
            }
            // There is no telling which is smaller.
            if at + len == history.end() {
                break;
            }
            let node = if index <= overwritten { &spare } else { node };
            index = descent.step(index, node, len, is_smaller);
        }
        descent.finish();

        let skipped = longest.saturating_sub(384).min(192);
        Some(skipped.max(matched_end - (current + WORD)))
    }

    #[inline(always)]
    pub(super) fn insert_up_to<const REACH: Reach>(
        &mut self,
        history: &History<'_>,
        end: u32,
    ) -> Option<()> {
        let end = history.index(end);
        let lowest = self.bounds.lowest(end);
        let mut index = self.next_to_index;
        while index < end {
            index += self.insert::<REACH>(history, index, lowest)?;
        }
        self.next_to_index = end;
        Some(())
    }
}
