//! Writes the blocks for a part of the input whose sequences are known.

use bun_zstd_decoder::format::{BLOCK_HEADER_LEN, BlockType};
use bun_zstd_decoder::{Error, Output, Result};

use crate::bytes::is_one_byte_repeated;
use crate::entropy::{Changes, CodeKind, Entropy, Reuse, looks_random};
use crate::parameters::Tuning;
use crate::sequences::{CodeCounts, Repeats, Sequence};
use crate::{vec, worst_case_len};

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

/// What blocks take over from one another.
pub(crate) struct BlockWriter {
    pub(crate) entropy: Box<Entropy>,
    pub(crate) repeats: Repeats,
    /// Up to 1.4.3, zstd's command fails on a frame that starts with a block of one byte.
    pub(crate) is_at_start: bool,
}

impl BlockWriter {
    pub(crate) fn new() -> Result<Self> {
        Ok(Self {
            entropy: vec::boxed(Entropy::EMPTY)?,
            repeats: Repeats::INITIAL,
            is_at_start: true,
        })
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

    /// As one block.
    pub(crate) fn write(
        &mut self,
        content: &Content<'_>,
        tuning: &Tuning,
        is_last: bool,
        out: &mut Output<'_>,
    ) -> Result<()> {
        const REPEATED_LEN_MAX: usize = 25;
        let compressed = if content.sequences.is_empty() && looks_random(content.literals) {
            None
        } else {
            let (start, rest) = room_for_block(out, worst_case_len(content.bytes.len()))?;
            let compressed = self.entropy.compress(content, tuning, rest)?;
            compressed.inspect(|&(len, _)| *start = header(BlockType::Compressed, len, is_last))
        };
        match compressed {
            _ if !self.is_at_start
                && compressed.is_none_or(|(len, _)| len < REPEATED_LEN_MAX)
                && is_one_byte_repeated(content.bytes) =>
            {
                write_repeated(content.bytes, is_last, out)?;
            }
            Some((len, changes)) => {
                out.commit(BLOCK_HEADER_LEN + len);
                self.accept(changes, content.repeats);
            }
            None => write_raw(content.bytes, is_last, out)?,
        }
        self.distrust_offset_table();
        Ok(())
    }
}
