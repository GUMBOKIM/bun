//! A [tree](super::tree) into which positions are only sorted once a search comes across them.

use super::hash::Hasher;
use super::lazy::{Finder, Match};
use super::tree::{self, Descent, Node, Nodes, Tree};
use super::{Attached, MatchState, Reach, WORD};
use crate::bytes::{self, highest_bit, u64_at};
use crate::sequences::OffsetValue;
use crate::window::History;

/// In the second half of a node. The first half links to the last position with the same hash.
pub(super) const UNSORTED: u32 = 1;

pub(super) struct LazyTree<'a> {
    tree: Tree<'a>,
    attached: Attached<'a>,
    attached_hasher: Hasher,
    window_len: u32,
}

impl<'a> LazyTree<'a> {
    pub(super) fn new(state: &'a mut MatchState, attached: &Attached<'a>) -> Self {
        let min_match = state.tuning.min_match.clamp(4, 6);
        Self {
            attached: *attached,
            attached_hasher: Hasher::new(min_match, attached.tuning.hash_log),
            window_len: 1 << state.tuning.window_log,
            tree: Tree::new(state, min_match),
        }
    }

    #[inline(always)]
    fn link_up_to(&mut self, history: &History<'_>, end: u32) -> Option<()> {
        let tree = &mut self.tree;
        let nodes = Nodes::new(tree.nodes)?;
        for index in tree.next_to_index..end {
            let word = u64_at(history.prefix, history.position(index))?;
            let root = tree.roots.get_mut(tree.hasher.hash(word))?;
            let [link, mark] = nodes.get(index)?;
            link.set(core::mem::replace(root, index));
            mark.set(UNSORTED);
        }
        tree.next_to_index = end;
        Some(())
    }

    #[inline(always)]
    fn sort<const REACH: Reach>(
        &self,
        history: &History<'_>,
        current: u32,
        attempts: u32,
        last: u32,
    ) -> Option<()> {
        let nodes = Nodes::new(self.tree.nodes)?;
        let (run, at) = match REACH {
            Reach::Older => history.locate(current),
            _ => (history.prefix, history.position(current)),
        };
        let lowest = self.tree.bounds.older_start;
        let lowest = lowest.max(current.saturating_sub(self.window_len));
        let spare = Node::default();
        let node @ [link, _] = nodes.get(current)?;
        let mut index = link.get();
        let mut descent = Descent::new(node);

        for _ in 0..attempts {
            if index <= lowest {
                break;
            }
            let node = nodes.get(index)?;
            let known = descent.known();
            let (len, is_smaller) = if REACH == Reach::Older && current < history.prefix_index {
                let from = |at: u32| run.get(at as usize + known as usize..);
                let earlier = history.older_position(index);
                let (len, is_smaller) = bytes::compare(from(at)?, from(earlier)?);
                (known + len as u32, is_smaller)
            } else {
                tree::compare::<REACH>(history, at, index, known)
            };
            if (at + len) as usize == run.len() {
                break;
            }
            let node = if index <= last { &spare } else { node };
            index = descent.step(index, node, len, is_smaller);
        }
        descent.finish();
        Some(())
    }

    /// Those that `index` links to, starting with the oldest one.
    #[inline(always)]
    fn sort_chain<const REACH: Reach>(
        &self,
        history: &History<'_>,
        mut index: u32,
        last: u32,
    ) -> Option<()> {
        let nodes = Nodes::new(self.tree.nodes)?;
        let mut attempts = self.tree.attempts;
        let mut newer = 0;
        while index > last {
            let [link, mark] = nodes.get(index)?;
            if mark.get() != UNSORTED {
                break;
            }
            if attempts <= 1 {
                link.set(0);
                mark.set(0);
                break;
            }
            // The chain is reversed, to find the way back.
            mark.set(newer);
            (newer, index) = (index, link.get());
            attempts -= 1;
        }
        while newer != 0 {
            let [_, mark] = nodes.get(newer)?;
            let next = mark.get();
            self.sort::<REACH>(history, newer, attempts, last)?;
            newer = next;
            attempts += 1;
        }
        Some(())
    }
}

impl Finder for LazyTree<'_> {
    const MARGIN: u32 = WORD;
    const GAINS_FROM_BMI2: bool = false;
    type Scratch = ();

    #[inline(always)]
    fn find<const REACH: Reach>(
        &mut self,
        history: &History<'_>,
        (): &mut (),
        at: u32,
        _: bool,
    ) -> Option<Match> {
        let current = history.index(at);
        let mut best = Match {
            len: 0,
            ..Match::NONE
        };
        if current < self.tree.next_to_index {
            return Some(best);
        }
        self.link_up_to(history, current)?;

        let word = u64_at(history.prefix, at)?;
        let lowest = self.tree.bounds.lowest(current);
        let nodes = Nodes::new(self.tree.nodes)?;
        let overwritten = nodes.overwritten(current);
        let root = self.tree.roots.get_mut(self.tree.hasher.hash(word))?;
        let mut index = core::mem::replace(root, current);
        self.sort_chain::<REACH>(history, index, overwritten.max(lowest))?;

        let spare = Node::default();
        let mut descent = Descent::new(nodes.get(current)?);
        let mut matched_end = current + WORD + 1;
        let mut attempts = self.tree.attempts;
        while attempts > 0 && index > lowest {
            let node = nodes.get(index)?;
            let (len, is_smaller) = tree::compare::<REACH>(history, at, index, descent.known());
            if len > best.len {
                matched_end = matched_end.max(index + len);
                let extra_bits =
                    highest_bit(current - index + 1) as i32 - best.offset.code() as i32;
                if 4 * (len - best.len) as i32 > extra_bits {
                    let offset = OffsetValue::new(current - index);
                    best = Match { len, offset };
                }
                if at + len == history.end() {
                    attempts = 0;
                    break;
                }
            }
            if index <= overwritten {
                descent.step(index, &spare, len, is_smaller);
                break;
            }
            index = descent.step(index, node, len, is_smaller);
            attempts -= 1;
        }
        descent.finish();

        if REACH == Reach::Attached {
            let hash = self.attached_hasher.hash(word);
            tree::descend_attached(&self.attached, history, at, hash, attempts, |index, len| {
                if len <= best.len {
                    return false;
                }
                let extra_bits =
                    highest_bit(current - index + 1) as i32 - highest_bit(best.offset.0 + 1) as i32;
                if 4 * (len - best.len) as i32 > extra_bits {
                    let offset = OffsetValue::new(current - index);
                    best = Match { len, offset };
                }
                at + len == history.end()
            })?;
        }
        self.tree.next_to_index = matched_end - WORD;
        Some(best)
    }

    fn next_to_index(&self) -> u32 {
        self.tree.next_to_index
    }
}
