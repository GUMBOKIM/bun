//! Writes the blocks for a part of the input whose sequences are known.

use bun_zstd_decoder::format::{BLOCK_HEADER_LEN, BlockType, CodeKind};
use bun_zstd_decoder::{Error, Output, Result};

use crate::bytes::is_one_byte_repeated;
use crate::entropy::{Changes, Entropy, Reuse};
use crate::parameters::Tuning;
use crate::sequences::{CodeCounts, Repeats, Sequence};
use crate::worst_case_len;

pub(crate) fn header(kind: BlockType, len: usize, is_last: bool) -> [u8; BLOCK_HEADER_LEN] {
    let value = u32::from(is_last) | (kind as u32) << 1 | (len as u32) << 3;
    let [low, middle, high, _] = value.to_le_bytes();
    [low, middle, high]
}

pub(crate) fn append(out: &mut Output<'_>, bytes: &[u8]) -> Result<()> {
    if out.append(bytes) < bytes.len() {
        return Err(Error::DstSizeTooSmall);
    }
    Ok(())
}

pub(crate) fn write_raw(bytes: &[u8], is_last: bool, out: &mut Output<'_>) -> Result<()> {
    append(out, &header(BlockType::Raw, bytes.len(), is_last))?;
    append(out, bytes)
}

pub(crate) fn write_repeated(bytes: &[u8], is_last: bool, out: &mut Output<'_>) -> Result<()> {
    append(out, &header(BlockType::Rle, bytes.len(), is_last))?;
    append(out, bytes.get(..1).ok_or(Error::Internal)?)
}

/// Up to `len` bytes, split into a header and the rest.
pub(crate) fn room_for_block<'a>(
    out: &'a mut Output<'_>,
    len: usize,
) -> Result<(&'a mut [u8; BLOCK_HEADER_LEN], &'a mut [u8])> {
    let written = out.written();
    let room = out.lend(len).get_mut(written..).unwrap_or_default();
    room.split_first_chunk_mut().ok_or(Error::DstSizeTooSmall)
}

/// Input, and what it is made of.
#[derive(Clone, Copy)]
pub(crate) struct Content<'a> {
    pub(crate) bytes: &'a [u8],
    pub(crate) sequences: &'a [Sequence],
    pub(crate) literals: &'a [u8],
    pub(crate) counts: &'a CodeCounts,
    /// The recent offsets that a decoder has after the sequences.
    pub(crate) repeats: Repeats,
}

impl Content<'_> {
    /// An attempt at compressing it that takes more is not worth it.
    fn len_max(&self) -> usize {
        let for_parts = self.literals.len() + 12 * self.sequences.len() + 512;
        for_parts.min(worst_case_len(self.bytes.len()))
    }
}

/// What blocks take over from one another.
pub(crate) struct BlockWriter {
    pub(crate) entropy: Entropy,
    pub(crate) repeats: Repeats,
    /// Up to 1.4.3, zstd's command fails on a frame that starts with a block of one byte.
    pub(crate) is_at_start: bool,
}

impl BlockWriter {
    pub(crate) const fn new() -> Self {
        Self {
            entropy: Entropy::EMPTY,
            repeats: Repeats::INITIAL,
            is_at_start: true,
        }
    }

    pub(crate) fn accept(&mut self, changes: Changes, repeats: Repeats) {
        self.entropy.accept(changes);
        self.repeats = repeats;
    }

    /// The table of a dictionary has codes for the offsets of the first block. They can be larger in later ones.
    pub(crate) fn distrust_offset_table(&mut self) {
        let reuse = self.entropy.codes.reuse_mut(CodeKind::Offset);
        if *reuse == Reuse::Always {
            *reuse = Reuse::IfComplete;
        }
    }

    fn can_repeat_byte(&self, bytes: &[u8]) -> bool {
        !self.is_at_start && is_one_byte_repeated(bytes)
    }

    pub(crate) fn write_uncompressed(
        &self,
        bytes: &[u8],
        is_last: bool,
        out: &mut Output<'_>,
    ) -> Result<()> {
        if self.can_repeat_byte(bytes) {
            write_repeated(bytes, is_last, out)
        } else {
            write_raw(bytes, is_last, out)
        }
    }

    /// As one block.
    pub(crate) fn write(
        &mut self,
        content: &Content<'_>,
        tuning: &Tuning,
        is_last: bool,
        out: &mut Output<'_>,
    ) -> Result<()> {
        const REPEATED_LEN_MAX: usize = 25;
        let (start, rest) = room_for_block(out, content.len_max())?;
        match self.entropy.compress(content, tuning, rest)? {
            Some((..REPEATED_LEN_MAX, _)) if self.can_repeat_byte(content.bytes) => {
                write_repeated(content.bytes, is_last, out)?;
            }
            Some((len, changes)) => {
                *start = header(BlockType::Compressed, len, is_last);
                out.commit(BLOCK_HEADER_LEN + len);
                self.accept(changes, content.repeats);
            }
            None => self.write_uncompressed(content.bytes, is_last, out)?,
        }
        self.distrust_offset_table();
        Ok(())
    }
}
