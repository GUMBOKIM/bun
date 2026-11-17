//! Looks for the cheapest way through all matches that there are.

use core::convert::Infallible;

use bun_zstd_decoder::{Result, with_bmi2};

use super::hash::Hasher;
use super::long_distance::Cursor;
use super::prices::{BIT, FIXED_UP_TO, Prices};
use super::tree::{self, Descent, Node, Tree};
use super::{Attached, MatchState, Reach, Search, WORD};
use crate::bytes::{u32_at, u64_at};
use crate::parameters::Strategy;
use crate::sequences::{OffsetValue, Repeats, Sequences};
use crate::vec;
use crate::window::{History, WORD_LEN};

/// How far ahead it looks.
const HORIZON: u32 = 1 << 12;
const MAX_PRICE: i32 = 1 << 30;

#[derive(Clone, Copy, Default)]
struct Match {
    offset: OffsetValue,
    len: u32,
}

/// The cheapest way to a position ends with this match and these literals.
#[derive(Clone, Copy, Default)]
struct Stretch {
    price: i32,
    offset: OffsetValue,
    match_len: u32,
    literal_len: u32,
    repeats: Repeats,
}

struct Stretches<'a>(&'a mut [Stretch]);

impl Stretches<'_> {
    #[inline(always)]
    fn get(&self, position: u32) -> Option<&Stretch> {
        self.0.get(position as usize)
    }

    #[inline(always)]
    fn get_mut(&mut self, position: u32) -> Option<&mut Stretch> {
        self.0.get_mut(position as usize)
    }
}

pub(super) struct State {
    prices: Prices,
    /// By length.
    matches: Vec<Match>,
    /// By position.
    stretches: Vec<Stretch>,
}

impl State {
    pub(super) fn new() -> Result<Box<Self>> {
        vec::boxed(Self {
            prices: Prices::EMPTY,
            matches: Vec::new(),
            stretches: Vec::new(),
        })
    }

    pub(super) fn reset(&mut self, block_len: usize) -> Result<()> {
        let len = block_len.min(HORIZON as usize) + 3;
        self.prices.forget();
        vec::clear_for(&mut self.matches, len)?;
        vec::refill(&mut self.stretches, Stretch::default(), len)
    }
}

pub(super) fn search<const REACH: Reach>(
    state: &mut MatchState,
    search: &Search<'_>,
    attached: &Attached<'_>,
    sequences: &mut Sequences,
    repeats: &mut Repeats,
) -> u32 {
    let mut history = search.history;
    let Some(mut optimal) = state.optimal.take() else {
        return history.start;
    };
    let is_first = optimal.prices.is_empty() && sequences.list.is_empty();
    let mut run = |state: &mut MatchState,
                   history,
                   sequences: &mut Sequences,
                   repeats: &mut Repeats| {
        let search = Search { history, ..*search };
        match state.tuning.strategy {
            Strategy::Optimal => {
                parse::<REACH, false>(state, &mut optimal, &search, attached, sequences, repeats)
            }
            _ => parse::<REACH, true>(state, &mut optimal, &search, attached, sequences, repeats),
        }
    };

    let len = history.block().len();
    if REACH == Reach::Prefix
        && state.tuning.strategy == Strategy::Ultra2
        && is_first
        && history.start == 0
        && len > FIXED_UP_TO
    {
        // Only for the prices. What it put into the tables gets out of reach.
        run(state, history, sequences, &mut { *repeats });
        sequences.list.clear();
        state.window.next += len as u32;
        state.window.raise_start(state.window.next);
        state.next_to_index = state.window.next;
        history.prefix_index = state.window.next;
    }
    let anchor = run(state, history, sequences, repeats);
    state.optimal = Some(optimal);
    anchor
}

#[inline(never)]
fn parse<const REACH: Reach, const IS_ACCURATE: bool>(
    state: &mut MatchState,
    optimal: &mut State,
    search: &Search<'_>,
    attached: &Attached<'_>,
    sequences: &mut Sequences,
    repeats: &mut Repeats,
) -> u32 {
    let (tuning, history) = (state.tuning, &search.history);
    optimal
        .prices
        .start_block::<IS_ACCURATE>(history.block(), search.entropy);
    let hashed = tuning.min_match.min(6);
    let tree = Tree::new(state, hashed);
    let finder = Finder {
        history: *history,
        attached: *attached,
        attached_hasher: Hasher::new(hashed, attached.tuning.hash_log),
        hasher3: Hasher::of_3_bytes(tuning.window_log.min(17)),
        next_to_index3: tree.next_to_index,
        tree,
        long_matches: Cursor::new(search.long_matches, history.block().len()),
        min_match: tuning.min_match.min(4),
        sufficient_len: tuning.target_length.min(HORIZON - 1),
    };
    let (anchor, next_to_index) = with_bmi2(
        #[inline(always)]
        move || {
            let (mut anchor, mut finder) = (finder.history.start, finder);
            finder.parse::<REACH, IS_ACCURATE>(optimal, &mut anchor, sequences, repeats);
            (anchor, finder.tree.next_to_index)
        },
    );
    state.next_to_index = next_to_index;
    anchor
}

struct Finder<'a> {
    history: History<'a>,
    tree: Tree<'a>,
    attached: Attached<'a>,
    attached_hasher: Hasher,
    hasher3: Hasher,
    next_to_index3: u32,
    long_matches: Cursor<'a>,
    /// 3 or 4
    min_match: u32,
    /// A longer match is taken right away.
    sufficient_len: u32,
}

#[inline(always)]
fn push(matches: &mut Vec<Match>, offset: OffsetValue, len: u32) {
    let pushed = matches.push_within_capacity(Match { offset, len });
    debug_assert!(pushed.is_ok());
}

impl Finder<'_> {
    #[inline(always)]
    fn repeat_len<const REACH: Reach>(&self, at: u32, value: u32, offset: u32, lowest: u32) -> u32 {
        let history = &self.history;
        let (current, min_match) = (history.index(at), self.min_match);
        let earlier = current.wrapping_sub(offset);
        let starts_alike = |other: Option<u32>| {
            other.is_some_and(|other| (other ^ value) << (32 - 8 * self.min_match) == 0)
        };
        if offset.wrapping_sub(1) < at {
            let back = at - offset;
            if earlier >= lowest && starts_alike(u32_at(history.prefix, back)) {
                return min_match + history.match_len(at + min_match, back + min_match);
            }
        } else if REACH != Reach::Prefix
            && (REACH == Reach::Attached || offset.wrapping_sub(1) < current - lowest)
            && earlier < history.prefix_index
            && starts_alike(history.u32_at_index(earlier))
        {
            return history.match_len_at_index(at, earlier, min_match);
        }
        0
    }

    /// The last position before `at` with about the same 3 bytes.
    #[inline(always)]
    fn find3(&mut self, at: u32, word: u64) -> Option<u32> {
        let history = &self.history;
        let current = history.index(at);
        for index in self.next_to_index3..current {
            let word = u64_at(history.prefix, history.position(index))?;
            *self.tree.hash3.get_mut(self.hasher3.hash(word))? = index;
        }
        self.next_to_index3 = current;
        self.tree.hash3.get(self.hasher3.hash(word)).copied()
    }

    /// Each match is longer than the one before.
    #[inline(always)]
    fn find_near<const REACH: Reach>(
        &mut self,
        at: u32,
        repeats: Repeats,
        has_literals: bool,
        matches: &mut Vec<Match>,
    ) -> Option<()> {
        let history = self.history;
        let current = history.index(at);
        if current < self.tree.next_to_index {
            return Some(());
        }
        self.tree.insert_up_to::<REACH>(&history, at)?;

        let word = u64_at(history.prefix, at)?;
        let left = history.end() - at;
        let lowest = self.tree.bounds.lowest(current);
        let sufficient_len = self.sufficient_len;
        let mut longest = self.min_match - 1;

        for value in 1..=3 {
            let offset = repeats.get(value - u32::from(has_literals));
            let len = self.repeat_len::<REACH>(at, word as u32, offset, lowest);
            if len > longest {
                longest = len;
                push(matches, OffsetValue(value), len);
                if len > sufficient_len || len == left {
                    return Some(());
                }
            }
        }

        if self.min_match == 3 && longest < 3 {
            let index = self.find3(at, word)?;
            // From further away it would cost too much.
            if index >= lowest && current - index < 1 << 18 {
                let len = match REACH {
                    Reach::Older => history.match_len_at_index(at, index, 0),
                    _ => history.match_len(at, history.position(index)),
                };
                if len >= 3 {
                    longest = len;
                    push(matches, OffsetValue::new(current - index), len);
                    if len > sufficient_len || len == left {
                        self.tree.next_to_index = current + 1;
                        return Some(());
                    }
                }
            }
        }

        let root = self.tree.roots.get_mut(self.tree.hasher.hash(word))?;
        let mut index = core::mem::replace(root, current);
        let nodes = self.tree.nodes;
        let overwritten = nodes.overwritten(current);
        let spare = Node::default();
        let mut descent = Descent::new(nodes.get(current)?);
        let mut matched_end = current + WORD + 1;
        let mut attempts = self.tree.attempts;
        let mut note = |index: u32, len: u32| {
            if len <= longest {
                return false;
            }
            matched_end = matched_end.max(index + len);
            longest = len;
            push(matches, OffsetValue::new(current - index), len);
            len > HORIZON || len == left
        };

        while attempts > 0 && index >= lowest {
            let node = nodes.get(index)?;
            let (len, is_smaller) = tree::compare::<REACH>(&history, at, index, descent.known());
            if note(index, len) {
                attempts = 0;
                break;
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
            tree::descend_attached(&self.attached, &history, at, hash, attempts, note)?;
        }
        self.tree.next_to_index = matched_end - WORD;
        Some(())
    }

    #[inline(always)]
    fn find<const REACH: Reach>(
        &mut self,
        at: u32,
        repeats: Repeats,
        has_literals: bool,
        matches: &mut Vec<Match>,
    ) -> Option<()> {
        matches.clear();
        self.find_near::<REACH>(at, repeats, has_literals, matches)?;
        let long = self
            .long_matches
            .at(at - self.history.start, self.history.end() - at);
        if let Some((offset, len)) = long
            && len >= self.min_match
            && matches.last().is_none_or(|last| len > last.len)
            && matches.len() < HORIZON as usize
        {
            push(matches, OffsetValue::new(offset), len);
        }
        Some(())
    }

    #[inline(always)]
    fn parse<const REACH: Reach, const IS_ACCURATE: bool>(
        &mut self,
        State {
            prices,
            matches,
            stretches,
        }: &mut State,
        anchor: &mut u32,
        sequences: &mut Sequences,
        repeats: &mut Repeats,
    ) -> Option<Infallible> {
        self.history = self.history.with_prefix_of::<WORD_LEN>()?;
        let History { prefix, start, .. } = self.history;
        let limit = self.history.end() - WORD;
        let min_match = self.min_match;
        let mut stretches = Stretches(stretches);
        let mut at = start + u32::from(start == 0);

        'block: while at < limit {
            let literal_len = at - *anchor;
            self.find::<REACH>(at, *repeats, literal_len != 0, matches)?;
            let Some(&longest) = matches.last() else {
                at += 1;
                continue;
            };
            let first = stretches.get_mut(0)?;
            *first = Stretch {
                // What the literals themselves cost is the same for all ways.
                price: prices.literal_len::<IS_ACCURATE>(literal_len),
                match_len: 0,
                literal_len,
                repeats: *repeats,
                ..*first
            };
            let ending_with = |last: Match| Stretch {
                offset: last.offset,
                match_len: last.len,
                ..Stretch::default()
            };

            // The last stretch, and where it would start if it had no literals.
            let (last, mut before_last) = 'forward: {
                if longest.len > self.sufficient_len {
                    break 'forward (ending_with(longest), 0);
                }

                let no_literals = prices.literal_len::<IS_ACCURATE>(0);
                let base_price = first.price + no_literals;
                for len in 1..min_match {
                    let stretch = stretches.get_mut(len)?;
                    stretch.price = MAX_PRICE;
                    stretch.match_len = 0;
                    stretch.literal_len = literal_len + len;
                }
                let mut len = min_match;
                for found in matches.iter() {
                    let price = base_price + prices.offset::<IS_ACCURATE>(found.offset);
                    while len <= found.len {
                        let stretch = stretches.get_mut(len)?;
                        stretch.price = price + prices.match_len::<IS_ACCURATE>(len);
                        stretch.offset = found.offset;
                        stretch.match_len = len;
                        stretch.literal_len = 0;
                        len += 1;
                    }
                }
                stretches.get_mut(len)?.price = MAX_PRICE;
                let mut end = len - 1;

                let one_more_literal = |len: u32| {
                    prices.literal_len::<IS_ACCURATE>(len + 1)
                        - prices.literal_len::<IS_ACCURATE>(len)
                };
                let mut current = 0;
                while current < end {
                    let previous = *stretches.get(current)?;
                    let literal = *prefix.get((at + current) as usize)?;
                    current += 1;

                    let price = previous.price
                        + prices.literal::<IS_ACCURATE>(literal)
                        + one_more_literal(previous.literal_len);
                    if price <= stretches.get(current)?.price {
                        let with_literal = Stretch {
                            price,
                            literal_len: previous.literal_len + 1,
                            ..previous
                        };
                        let replaced =
                            core::mem::replace(stretches.get_mut(current)?, with_literal);
                        // The match may still be the better start for the literal after it.
                        if IS_ACCURATE
                            && replaced.literal_len == 0
                            && one_more_literal(0) < 0
                            && let Some(&next) = prefix.get((at + current) as usize)
                        {
                            let next = prices.literal::<IS_ACCURATE>(next);
                            let after_match = replaced.price + next + one_more_literal(0);
                            let after_literals =
                                price + next + one_more_literal(with_literal.literal_len);
                            if after_match < after_literals
                                && after_match < stretches.get(current + 1)?.price
                            {
                                let from = stretches.get(current - replaced.match_len)?;
                                *stretches.get_mut(current + 1)? = Stretch {
                                    price: after_match,
                                    literal_len: 1,
                                    repeats: from
                                        .repeats
                                        .after(replaced.offset, from.literal_len != 0),
                                    ..replaced
                                };
                                end = end.max(current + 1);
                            }
                        }
                    }

                    let mut here = *stretches.get(current)?;
                    if here.literal_len == 0 {
                        let from = stretches.get(current - here.match_len)?;
                        here.repeats = from.repeats.after(here.offset, from.literal_len != 0);
                        stretches.get_mut(current)?.repeats = here.repeats;
                    }

                    if at + current > limit {
                        continue;
                    }
                    if current == end {
                        break;
                    }
                    if !IS_ACCURATE && stretches.get(current + 1)?.price <= here.price + BIT / 2 {
                        continue;
                    }

                    self.find::<REACH>(at + current, here.repeats, here.literal_len != 0, matches)?;
                    let Some(&longest) = matches.last() else {
                        continue;
                    };
                    let longest_end = current + longest.len;
                    if longest.len > self.sufficient_len
                        || longest_end >= HORIZON
                        || at + longest_end >= self.history.end()
                    {
                        break 'forward (ending_with(longest), current);
                    }

                    let base_price = here.price + no_literals;
                    let mut shortest = min_match;
                    for found in matches.iter() {
                        let price = base_price + prices.offset::<IS_ACCURATE>(found.offset);
                        for len in (shortest..=found.len).rev() {
                            let position = current + len;
                            let price = price + prices.match_len::<IS_ACCURATE>(len);
                            if position > end || price < stretches.get(position)?.price {
                                while end < position {
                                    end += 1;
                                    let skipped = stretches.get_mut(end)?;
                                    skipped.price = MAX_PRICE;
                                    skipped.literal_len = 1;
                                }
                                let stretch = stretches.get_mut(position)?;
                                stretch.price = price;
                                stretch.offset = found.offset;
                                stretch.match_len = len;
                                stretch.literal_len = 0;
                            } else if !IS_ACCURATE {
                                break;
                            }
                        }
                        shortest = found.len + 1;
                    }
                    stretches.get_mut(end + 1)?.price = MAX_PRICE;
                }

                let last = *stretches.get(end)?;
                if last.match_len == 0 {
                    at += end;
                    continue 'block;
                }
                (last, end - last.match_len)
            };

            if last.literal_len == 0 {
                let from = stretches.get(before_last)?;
                *repeats = from.repeats.after(last.offset, from.literal_len != 0);
            } else {
                *repeats = last.repeats;
                before_last -= last.literal_len;
            }

            // Going back, each match gets the literals in front of it. They are stored in what was passed.
            let stored_end = before_last + 2;
            let mut stored_start = stored_end;
            let mut position = before_last;
            *stretches.get_mut(stored_end)? = last;
            loop {
                let stretch = *stretches.get(position)?;
                stretches.get_mut(stored_start)?.literal_len = stretch.literal_len;
                if stretch.match_len == 0 {
                    break;
                }
                stored_start -= 1;
                *stretches.get_mut(stored_start)? = stretch;
                position -= stretch.literal_len + stretch.match_len;
            }

            for stored in stretches
                .0
                .get(stored_start as usize..=stored_end as usize)?
            {
                let start = *anchor + stored.literal_len;
                let literals = prefix.get(*anchor as usize..start as usize)?;
                prices.add(literals, stored.offset, stored.match_len);
                sequences.push(stored.literal_len, stored.offset, stored.match_len);
                *anchor = start + stored.match_len;
            }
            at = *anchor;
            prices.update::<IS_ACCURATE>();
        }
        None
    }
}
