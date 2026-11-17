//! The search for matches.

mod chain;
mod double_fast;
mod fast;
mod gear_table;
mod hash;
mod lazy;
mod lazy_tree;
pub(crate) mod long_distance;
mod optimal;
mod prices;
mod rows;
mod tree;

use core::cell::Cell;
use core::marker::ConstParamTy;

use bun_zstd_decoder::Result;

use crate::entropy::Entropy;
use crate::parameters::{Strategy, TAG_BITS, Tuning};
use crate::sequences::{OffsetValue, Repeats, Sequences};
use crate::vec;
use crate::window::{FIRST_INDEX, History, INDEX_MAX, Input, WORD_LEN, Window};
use long_distance::Candidates;

/// The more bytes in a row do not match, the more are skipped.
const SKIP_LOG: u32 = 8;
const WORD: u32 = WORD_LEN as u32;
const CACHE_LINE_LEN: usize = 64;

/// What there is to search besides the prefix.
#[derive(ConstParamTy, Clone, Copy, PartialEq, Eq)]
enum Reach {
    Prefix,
    /// An older run of bytes, which the same tables index.
    Older,
    /// A dictionary with tables of its own.
    Attached,
}

/// `len` elements of `memory`, from the start of a cache line.
fn aligned<T>(memory: &[T], len: usize) -> core::ops::Range<usize> {
    let start = memory.as_ptr().align_offset(CACHE_LINE_LEN);
    start..start.saturating_add(len)
}

#[derive(Default)]
struct Tables {
    indices: Vec<u32>,
    hash_len: usize,
    chain_len: usize,
    hash3_len: usize,
    tags: Vec<u8>,
    tags_len: usize,
}

struct TablesMut<'a> {
    hash: &'a mut [u32],
    chain: &'a mut [u32],
    hash3: &'a mut [u32],
    tags: &'a mut [u8],
}

impl Tables {
    fn indices_len(&self) -> usize {
        self.hash_len + self.chain_len + self.hash3_len
    }

    fn split(&self) -> (&[u32], &[u32], &[u8]) {
        let indices = self.indices.get(aligned(&self.indices, self.indices_len()));
        let indices = indices.and_then(|it| it.split_at_checked(self.hash_len));
        let (hash, rest) = indices.unwrap_or_default();
        let tags = self.tags.get(aligned(&self.tags, self.tags_len));
        let chain = rest.get(..self.chain_len);
        (hash, chain.unwrap_or_default(), tags.unwrap_or_default())
    }

    fn split_mut(&mut self) -> TablesMut<'_> {
        let indices = aligned(&self.indices, self.indices_len());
        let indices = self.indices.get_mut(indices).unwrap_or_default();
        let (hash, rest) = indices
            .split_at_mut_checked(self.hash_len)
            .unwrap_or_default();
        let (chain, hash3) = rest
            .split_at_mut_checked(self.chain_len)
            .unwrap_or_default();
        let tags = aligned(&self.tags, self.tags_len);
        TablesMut {
            hash,
            chain,
            hash3,
            tags: self.tags.get_mut(tags).unwrap_or_default(),
        }
    }
}

/// Lets a search hold on to entries that may be the same.
#[inline(always)]
fn cells(table: &mut [u32]) -> &[Cell<u32>] {
    Cell::from_mut(table).as_slice_of_cells()
}

#[inline(always)]
fn prefetch<T>(items: &[T], at: usize) {
    let item = items.as_ptr().wrapping_add(at);
    core::hint::prefetch_read(item, core::hint::Locality::L1);
}

/// How far back a match can start.
#[derive(Clone, Copy)]
struct Bounds {
    prefix_start: u32,
    older_start: u32,
    /// Unlimited as long as a dictionary is within reach.
    max_distance: u32,
}

impl Bounds {
    #[inline(always)]
    fn lowest(&self, current: u32) -> u32 {
        self.older_start
            .max(current.saturating_sub(self.max_distance))
    }

    #[inline(always)]
    fn lowest_in_prefix(&self, current: u32) -> u32 {
        self.prefix_start
            .max(current.saturating_sub(self.max_distance))
    }
}

/// The tables of a dictionary, whose content is the older run of a [`History`].
#[derive(Clone, Copy)]
struct Attached<'a> {
    hash: &'a [u32],
    chain: &'a [u32],
    tags: &'a [u8],
    tuning: Tuning,
    /// Of the byte after the last one, in terms of these tables.
    end: u32,
    /// Added to an index of these tables, makes it one of the history.
    delta: u32,
}

impl Attached<'_> {
    /// Stands in where there is none.
    fn empty(tuning: Tuning) -> Self {
        Self {
            hash: &[],
            chain: &[],
            tags: &[],
            tuning,
            end: FIRST_INDEX,
            delta: 0,
        }
    }

    /// Where an index of these tables is in the older run.
    #[inline(always)]
    fn position(index: u32) -> u32 {
        index.wrapping_sub(FIRST_INDEX)
    }
}

/// Where a search that knows two recent offsets has got to.
struct Progress<'a> {
    sequences: &'a mut Sequences,
    /// The first byte that no sequence covers.
    anchor: u32,
    /// 0 stands for one that is out of reach.
    recent: u32,
    second: u32,
}

impl<'a> Progress<'a> {
    fn new(sequences: &'a mut Sequences, anchor: u32, repeats: Repeats, max_offset: u32) -> Self {
        let [recent, second, _] = repeats
            .0
            .map(|offset| if offset > max_offset { 0 } else { offset });
        Self {
            sequences,
            anchor,
            recent,
            second,
        }
    }

    /// For a match that starts at `start`.
    #[inline(always)]
    fn store(&mut self, start: u32, offset: OffsetValue, len: u32) {
        self.sequences
            .push(start.wrapping_sub(self.anchor), offset, len);
        self.anchor = start.wrapping_add(len);
    }

    #[inline(always)]
    fn note_offset(&mut self, offset: u32) {
        (self.recent, self.second) = (offset, self.recent);
    }

    #[inline(always)]
    fn swap_offsets(&mut self) {
        (self.recent, self.second) = (self.second, self.recent);
    }

    /// `repeats` and `max_offset` are those that it started with. Returns the anchor.
    fn finish(self, repeats: &mut Repeats, max_offset: u32) -> u32 {
        let [recent, second, _] = &mut repeats.0;
        if self.second != 0 {
            *second = self.second;
        } else if self.recent != 0 && *recent > max_offset {
            *second = *recent;
        } else if *second <= max_offset {
            *second = 0;
        }
        if self.recent != 0 || *recent <= max_offset {
            *recent = self.recent;
        }
        self.anchor
    }
}

/// What a search is given besides the state.
pub(crate) struct Search<'a> {
    pub(crate) history: History<'a>,
    /// Its content has to be the older run of the history.
    pub(crate) attached: Option<&'a MatchState>,
    /// Of the block before.
    pub(crate) entropy: &'a Entropy,
    pub(crate) long_matches: Option<&'a Candidates>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Owner {
    Frame,
    Dictionary,
}

pub(crate) struct MatchState {
    pub(crate) tuning: Tuning,
    pub(crate) uses_rows: bool,
    pub(crate) window: Window,
    /// Of the byte after a dictionary that is within reach, or 0.
    pub(crate) dictionary_end: u32,
    /// From which on the tables are not up to date.
    pub(crate) next_to_index: u32,
    tables: Tables,
    optimal: Option<Box<optimal::State>>,
}

impl MatchState {
    /// Without tables, until it is [`reset`](Self::reset).
    pub(crate) fn new(tuning: Tuning) -> Self {
        Self {
            tuning,
            uses_rows: false,
            window: Window::starting_at(FIRST_INDEX),
            dictionary_end: 0,
            next_to_index: FIRST_INDEX,
            tables: Tables::default(),
            optimal: None,
        }
    }

    pub(crate) fn reset(
        &mut self,
        tuning: Tuning,
        uses_rows: bool,
        owner: Owner,
        block_len: usize,
    ) -> Result<()> {
        (self.tuning, self.uses_rows) = (tuning, uses_rows);
        self.window = Window::starting_at(FIRST_INDEX);
        self.dictionary_end = 0;
        self.next_to_index = FIRST_INDEX;

        let is_optimal = owner == Owner::Frame && tuning.strategy >= Strategy::Optimal;
        let has_chain = tuning.strategy != Strategy::Fast && !uses_rows;
        let has_hash3 = is_optimal && tuning.min_match == 3;
        let tables = &mut self.tables;
        tables.hash_len = 1 << tuning.hash_log;
        tables.chain_len = usize::from(has_chain) << tuning.chain_log;
        tables.hash3_len = usize::from(has_hash3) << tuning.window_log.min(17);
        tables.tags_len = usize::from(uses_rows) << tuning.hash_log;
        let indices_len = tables.indices_len() + CACHE_LINE_LEN / size_of::<u32>();
        vec::refill(&mut tables.indices, 0, indices_len)?;
        if uses_rows {
            vec::refill(&mut tables.tags, 0, tables.tags_len + CACHE_LINE_LEN)?;
        }

        match (&mut self.optimal, owner) {
            (Some(optimal), Owner::Frame) => optimal.reset(block_len),
            (None, Owner::Frame) if is_optimal => {
                self.optimal.insert(optimal::State::new()?).reset(block_len)
            }
            _ => Ok(()),
        }
    }

    /// Goes on from where the tables of a dictionary, which have the same sizes, are.
    pub(crate) fn copy_from(&mut self, dictionary: &Self) {
        let (hash, chain, tags) = dictionary.tables.split();
        let to = self.tables.split_mut();
        let has_tags = self.tuning.strategy <= Strategy::DoubleFast;
        let shift = if has_tags { TAG_BITS } else { 0 };
        for (to, from) in [(to.hash, hash), (to.chain, chain)] {
            for (to, from) in to.iter_mut().zip(from) {
                *to = from >> shift;
            }
        }
        for (to, from) in to.tags.iter_mut().zip(tags) {
            *to = *from;
        }
        self.window = dictionary.window;
        self.next_to_index = dictionary.next_to_index;
        self.dictionary_end = dictionary.dictionary_end;
    }

    /// Leaves room below the input for the indices of a dictionary.
    pub(crate) fn attach(&mut self, dictionary: &Self) {
        self.window = Window::starting_at(dictionary.window.next);
        self.dictionary_end = self.window.next;
    }

    /// The end of `content` that the tables of `owner` can tell apart.
    pub(crate) fn loadable<'a>(&self, content: &'a [u8], owner: Owner) -> &'a [u8] {
        let has_tags = owner == Owner::Dictionary && self.tuning.strategy <= Strategy::DoubleFast;
        let end_max = if has_tags {
            1 << (32 - TAG_BITS)
        } else {
            INDEX_MAX
        };
        let excess = content
            .len()
            .saturating_sub((end_max - FIRST_INDEX) as usize);
        content.get(excess..).unwrap_or_default()
    }

    /// Indexes [`loadable`](Self::loadable) content, which nothing precedes.
    pub(crate) fn load(&mut self, content: &[u8], owner: Owner) {
        let tuning = self.tuning;
        let worth_log = (tuning.hash_log + 3).max(tuning.chain_log + 1).min(31);
        let start = content.len().saturating_sub(1 << worth_log);
        let history = self
            .window
            .history(&Input::alone(content, &[]), 0, content.len());
        let end = history.index(history.end());
        self.window.next = end;
        self.next_to_index = history.index(start as u32);
        if owner == Owner::Dictionary {
            self.dictionary_end = end;
        }
        if content.len() - start <= WORD_LEN {
            return;
        }

        let last = history.end() - WORD;
        match tuning.strategy {
            Strategy::Fast => fast::fill(self, &history, owner),
            Strategy::DoubleFast => double_fast::fill(self, &history, owner),
            _ if self.uses_rows => rows::fill(self, &history, last),
            Strategy::Greedy | Strategy::Lazy | Strategy::Lazy2 => {
                chain::fill(self, &history, last);
            }
            _ => tree::fill(self, &history, last),
        }
        self.next_to_index = end;
    }

    pub(crate) fn admit(&mut self, input: &Input<'_>) {
        self.window.admit(input);
        if !input.follows_last {
            self.next_to_index = self.window.prefix_start;
        }
    }

    /// Drops what is out of reach for the next `len` bytes.
    pub(crate) fn prepare_for(&mut self, len: usize) {
        if self.window.would_overflow(len) {
            self.lower_indices();
        }
        let max_distance = 1u64 << self.tuning.window_log;
        let reach_end = u64::from(self.dictionary_end) + max_distance;
        let block_end = u64::from(self.window.next) + len as u64;
        if block_end > reach_end || self.dictionary_end != self.window.prefix_start {
            self.dictionary_end = 0;
        }
        if self.dictionary_end == 0 {
            let lowest = self.window.next.saturating_sub(max_distance as u32);
            self.window.raise_start(lowest);
        }
        self.next_to_index = self.next_to_index.max(self.window.older_start);
    }

    #[cold]
    fn lower_indices(&mut self) {
        let by = self
            .window
            .lower_indices(self.tuning.cycle_log(), self.tuning.window_log);
        let lower = |index: u32| {
            if index < by + FIRST_INDEX {
                0
            } else {
                index - by
            }
        };
        let has_marks = self.tuning.strategy == Strategy::TreeLazy2;
        let tables = self.tables.split_mut();
        for index in tables.hash.iter_mut().chain(tables.hash3) {
            *index = lower(*index);
        }
        for index in tables.chain {
            if !(has_marks && *index == lazy_tree::UNSORTED) {
                *index = lower(*index);
            }
        }
        self.next_to_index = self.next_to_index.saturating_sub(by);
        self.dictionary_end = 0;
    }

    /// After a long match, leaves most of it out of the tables.
    pub(crate) fn skip_indexing(&mut self, current: u32, tolerated: u32) {
        if let Some(excess) = current.checked_sub(self.next_to_index + tolerated)
            && excess > 0
        {
            self.next_to_index = current - excess.min(tolerated / 2);
        }
    }

    fn bounds(&self) -> Bounds {
        Bounds {
            prefix_start: self.window.prefix_start,
            older_start: self.window.older_start,
            max_distance: if self.dictionary_end != 0 {
                u32::MAX
            } else {
                1 << self.tuning.window_log
            },
        }
    }

    /// Where a long match was taken without a search, for the strategies that do not catch up by themselves.
    fn index_up_to(&mut self, history: &History<'_>, end: u32) {
        self.skip_indexing(history.index(end), 1024);
        let history = history.part(end, end);
        match self.tuning.strategy {
            Strategy::Fast => fast::fill(self, &history, Owner::Frame),
            Strategy::DoubleFast => double_fast::fill(self, &history, Owner::Frame),
            _ => {}
        }
    }

    /// Returns where the literals start that follow the last match.
    pub(crate) fn search(
        &mut self,
        search: &Search<'_>,
        sequences: &mut Sequences,
        repeats: &mut Repeats,
    ) -> u32 {
        match search.attached {
            _ if self.window.has_older() => {
                let none = Attached::empty(self.tuning);
                self.search_in::<{ Reach::Older }>(search, &none, sequences, repeats)
            }
            Some(dictionary) => {
                let (hash, chain, tags) = dictionary.tables.split();
                let end = dictionary.window.next;
                let attached = Attached {
                    hash,
                    chain,
                    tags,
                    tuning: dictionary.tuning,
                    end,
                    delta: search.history.prefix_index.wrapping_sub(end),
                };
                self.search_in::<{ Reach::Attached }>(search, &attached, sequences, repeats)
            }
            None => {
                let none = Attached::empty(self.tuning);
                self.search_in::<{ Reach::Prefix }>(search, &none, sequences, repeats)
            }
        }
    }

    fn search_in<const REACH: Reach>(
        &mut self,
        search: &Search<'_>,
        attached: &Attached<'_>,
        sequences: &mut Sequences,
        repeats: &mut Repeats,
    ) -> u32 {
        let history = &search.history;
        match self.tuning.strategy {
            Strategy::Fast => fast::search::<REACH>(self, history, attached, sequences, repeats),
            Strategy::DoubleFast => {
                double_fast::search::<REACH>(self, history, attached, sequences, repeats)
            }
            Strategy::Greedy | Strategy::Lazy | Strategy::Lazy2 | Strategy::TreeLazy2 => {
                lazy::search::<REACH>(self, history, attached, sequences, repeats)
            }
            Strategy::Optimal | Strategy::Ultra | Strategy::Ultra2 => {
                optimal::search::<REACH>(self, search, attached, sequences, repeats)
            }
        }
    }
}
