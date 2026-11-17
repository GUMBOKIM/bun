//! Compresses input that is at hand into a frame, or a part of one.

use bun_zstd_decoder::format::{
    BLOCK_HEADER_LEN, BLOCK_LEN_MAX, BlockType, FRAME_HEADER_LEN_MAX, FRAME_MAGIC, WINDOW_LOG_MIN,
};
use bun_zstd_decoder::{Error, Output, Result, XxHash64};

use crate::block::{self, BlockWriter, Content, append};
use crate::dictionary::Dictionary;
use crate::matches::long_distance::{self, Candidates};
use crate::matches::{MatchState, Owner, Search};
use crate::parameters::{Plan, Purpose};
use crate::sequences::{Repeats, Sequences};
use crate::window::{Input, WORD_LEN};
use crate::{post_split, pre_split, sub_blocks};

pub(crate) fn last_empty_block() -> [u8; BLOCK_HEADER_LEN] {
    block::header(BlockType::Raw, 0, true)
}

pub(crate) fn checksum(hash: &XxHash64) -> [u8; 4] {
    (hash.digest() as u32).to_le_bytes()
}

/// What precedes the input.
#[derive(Clone, Copy)]
pub(crate) enum Start<'a> {
    Frame(Option<&'a Dictionary>),
    /// The end of the input of an earlier part of the frame, which the first [`Input`] follows.
    Part(&'a [u8]),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    BeforeHeader,
    InBlocks,
    AfterLastBlock,
}

pub(crate) struct FrameWriter {
    plan: Plan,
    input_len: Option<u64>,
    dictionary_id: u32,
    is_dictionary_attached: bool,
    block_len_max: usize,
    stage: Stage,
    consumed: u64,
    produced: u64,
    hash: XxHash64,
    state: MatchState,
    sequences: Sequences,
    blocks: BlockWriter,
    long_finder: Option<long_distance::Finder>,
    /// Without a finder, they can come from elsewhere.
    pub(crate) long_matches: Candidates,
}

impl FrameWriter {
    pub(crate) fn new(plan: Plan, input_len: Option<u64>, start: Start<'_>) -> Result<Self> {
        let mut writer = Self {
            plan,
            input_len,
            dictionary_id: 0,
            is_dictionary_attached: false,
            block_len_max: 0,
            stage: Stage::BeforeHeader,
            consumed: 0,
            produced: 0,
            hash: XxHash64::new(),
            state: MatchState::new(plan.tuning),
            sequences: Sequences::new(),
            blocks: BlockWriter::new()?,
            long_finder: None,
            long_matches: Candidates::default(),
        };
        writer.restart(plan, input_len, start)?;
        Ok(writer)
    }

    pub(crate) fn restart(
        &mut self,
        mut plan: Plan,
        input_len: Option<u64>,
        start: Start<'_>,
    ) -> Result<()> {
        let dictionary = match start {
            Start::Frame(dictionary) => dictionary,
            Start::Part(_) => None,
        };
        let is_attached = dictionary.is_some_and(|it| it.is_attached_for(input_len));
        if let Some(Dictionary { state, .. }) = dictionary {
            let window_log = plan.tuning.window_log;
            plan.tuning = state.tuning;
            if is_attached {
                let purpose = Purpose::Frame { dictionary_len: 0 };
                plan.tuning = plan.tuning.fitted(input_len, purpose);
            }
            plan.tuning.window_log = window_log;
            plan.uses_rows = state.uses_rows;
        }
        plan.header.has_content_size &= input_len.is_some();
        let tuning = plan.tuning;

        (self.plan, self.input_len) = (plan, input_len);
        self.block_len_max = self.window_len().min(BLOCK_LEN_MAX);
        (self.consumed, self.produced) = (0, 0);
        self.hash = XxHash64::new();
        self.stage = Stage::BeforeHeader;
        self.dictionary_id = dictionary.map_or(0, |it| it.id);
        self.is_dictionary_attached = false;
        self.state
            .reset(tuning, plan.uses_rows, Owner::Frame, self.block_len_max)?;
        self.sequences
            .clear_for(self.block_len_max, tuning.min_match)?;
        self.blocks.is_at_start = true;
        self.long_finder = None;
        self.long_matches.clear();
        if let Some(parameters) = plan.long_distance {
            let parameters = parameters.completed(&tuning);
            self.long_finder = Some(long_distance::Finder::new(parameters, tuning.window_log)?);
            self.long_matches
                .clear_for(self.block_len_max, &parameters)?;
        }

        match start {
            Start::Frame(None) => {
                self.blocks.entropy.forget();
                self.blocks.repeats = Repeats::INITIAL;
            }
            Start::Frame(Some(dictionary)) => {
                self.blocks.entropy.copy_from(&dictionary.entropy);
                self.blocks.repeats = dictionary.repeats;
                if !is_attached {
                    self.state.copy_from(&dictionary.state);
                } else if !dictionary.content().is_empty() {
                    self.state.attach(&dictionary.state);
                    self.is_dictionary_attached = true;
                }
            }
            Start::Part(earlier) => {
                self.blocks.entropy.forget();
                self.blocks.repeats = Repeats::default();
                self.stage = Stage::InBlocks;
                if earlier.len() >= WORD_LEN {
                    self.state.load(earlier, Owner::Frame);
                }
            }
        }
        Ok(())
    }

    pub(crate) fn window_len(&self) -> usize {
        let len = 1u64 << self.plan.tuning.window_log;
        len.min(self.input_len.unwrap_or(u64::MAX)).max(1) as usize
    }

    pub(crate) fn block_len_max(&self) -> usize {
        self.block_len_max
    }

    fn write_header(&self, out: &mut Output<'_>) -> Result<()> {
        let (flags, window_log) = (self.plan.header, self.plan.tuning.window_log);
        if out.room() < FRAME_HEADER_LEN_MAX {
            return Err(Error::DstSizeTooSmall);
        }
        let (id_code, id_len) = match self.dictionary_id {
            _ if !flags.has_dictionary_id => (0, 0),
            0 => (0, 0),
            1..0x100 => (1, 1),
            0x100..0x1_0000 => (2, 2),
            _ => (3, 4),
        };
        let size = self.input_len.filter(|_| flags.has_content_size);
        let is_single_segment = size.is_some_and(|size| size <= 1 << window_log);
        let (size_code, size_len, size) = match size {
            None => (0, 0, 0),
            Some(size @ ..0x100) => (0, usize::from(is_single_segment), size),
            Some(size @ ..0x1_0100) => (1, 2, size - 0x100),
            Some(size @ ..0xFFFF_FFFF) => (2, 4, size),
            Some(size) => (3, 8, size),
        };
        let descriptor = id_code
            | u8::from(flags.has_checksum) << 2
            | u8::from(is_single_segment) << 5
            | size_code << 6;
        let window = ((window_log - WINDOW_LOG_MIN) << 3) as u8;

        let fields: [(&[u8], usize); 5] = [
            (&FRAME_MAGIC.to_le_bytes(), 4 * usize::from(flags.has_magic)),
            (&[descriptor], 1),
            (&[window], usize::from(!is_single_segment)),
            (&self.dictionary_id.to_le_bytes(), id_len),
            (&size.to_le_bytes(), size_len),
        ];
        for (bytes, len) in fields {
            append(out, bytes.get(..len).unwrap_or(bytes))?;
        }
        Ok(())
    }

    /// `dictionary` is the one that the frame started with.
    pub(crate) fn write(
        &mut self,
        input: &Input<'_>,
        dictionary: Option<&Dictionary>,
        is_last: bool,
        out: &mut Output<'_>,
    ) -> Result<()> {
        let consumed = self.consumed + input.len as u64;
        if self
            .input_len
            .is_some_and(|len| consumed > len || (is_last && consumed < len))
        {
            return Err(Error::SrcSizeWrong);
        }
        let written_before = out.written();
        if self.stage == Stage::BeforeHeader {
            self.write_header(out)?;
            self.stage = Stage::InBlocks;
        }
        if input.len != 0 {
            self.write_blocks(input, dictionary, is_last, out)?;
            self.consumed = consumed;
            self.produced += (out.written() - written_before) as u64;
        }
        if is_last {
            if self.stage != Stage::AfterLastBlock {
                append(out, &last_empty_block())?;
            }
            if self.plan.header.has_checksum {
                append(out, &checksum(&self.hash))?;
            }
        }
        Ok(())
    }

    fn write_blocks(
        &mut self,
        input: &Input<'_>,
        dictionary: Option<&Dictionary>,
        is_last: bool,
        out: &mut Output<'_>,
    ) -> Result<()> {
        self.state.admit(input);
        if let Some(finder) = &mut self.long_finder {
            finder.admit(input);
        }
        if self.plan.header.has_checksum {
            self.hash.update(input.new_bytes());
        }

        let mut saved = self.consumed as i64 - self.produced as i64;
        let mut start = input.start();
        while let Some(left @ [_, ..]) = input.segment.get(start..) {
            let len = match left.first_chunk() {
                // More block headers must not make the output larger than it could be without.
                Some(block) if self.block_len_max == BLOCK_LEN_MAX && saved >= 3 => {
                    pre_split::block_len(block, self.plan.tuning.strategy)
                }
                _ => left.len().min(self.block_len_max),
            };
            let written_before = out.written();
            let is_last = is_last && len == left.len();
            self.write_block(input, dictionary, start, len, is_last, out)?;
            saved += len as i64 - (out.written() - written_before) as i64;
            start += len;
            self.blocks.is_at_start = false;
        }
        if is_last {
            self.stage = Stage::AfterLastBlock;
        }
        Ok(())
    }

    fn write_block(
        &mut self,
        input: &Input<'_>,
        dictionary: Option<&Dictionary>,
        start: usize,
        len: usize,
        is_last: bool,
        out: &mut Output<'_>,
    ) -> Result<()> {
        const WORTH_TRYING_FROM: usize = 7;
        let bytes = input.segment.get(start..start + len);
        let bytes = bytes.ok_or(Error::Internal)?;
        self.state.prepare_for(len);
        if len < WORTH_TRYING_FROM {
            self.long_matches.skip(len, &self.state);
            if let Some(finder) = &mut self.long_finder {
                finder.skip(len);
            }
            self.state.window.next += len as u32;
            self.blocks.distrust_offset_table();
            return block::write_raw(bytes, is_last, out);
        }

        let repeats = self.find_sequences(input, dictionary, start, len)?;
        let (blocks, tuning) = (&mut self.blocks, &self.plan.tuning);
        let content = Content {
            bytes,
            sequences: &self.sequences.list,
            literals: self.sequences.literals(bytes),
            counts: &self.sequences.counts,
            repeats,
        };
        if self.plan.target_block_len != 0 {
            let target_len = self.plan.target_block_len;
            sub_blocks::write(blocks, &content, tuning, target_len, is_last, out)
        } else if self.plan.splits_blocks {
            let all = &mut self.sequences;
            post_split::write(blocks, bytes, all, repeats, tuning, is_last, out)
        } else {
            blocks.write(&content, tuning, is_last, out)
        }
    }

    /// Returns the recent offsets after them.
    fn find_sequences(
        &mut self,
        input: &Input<'_>,
        dictionary: Option<&Dictionary>,
        start: usize,
        len: usize,
    ) -> Result<Repeats> {
        let state = &mut self.state;
        state.skip_indexing(state.window.next, 384);
        let mut search = Search {
            history: state.window.history(input, start, len),
            attached: None,
            entropy: &self.blocks.entropy,
            long_matches: None,
        };
        if let Some(dictionary) = dictionary
            && self.is_dictionary_attached
            && state.dictionary_end != 0
        {
            search.history.older = dictionary.content();
            search.attached = Some(&dictionary.state);
        }

        let (sequences, mut repeats) = (&mut self.sequences, self.blocks.repeats);
        sequences.list.clear();
        if let Some(finder) = &mut self.long_finder {
            self.long_matches.clear();
            finder.find(input, start, len, &mut self.long_matches)?;
        }
        if self.long_finder.is_some() || !self.long_matches.are_used_up() {
            self.long_matches
                .search(state, &search, sequences, &mut repeats);
        } else {
            state.search(&search, sequences, &mut repeats);
        }
        sequences.finish(search.history.block())?;
        state.window.next += len as u32;
        Ok(repeats)
    }
}
