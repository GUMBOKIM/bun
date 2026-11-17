//! Writes the blocks for a part of the input whose sequences are known.

use bun_zstd_decoder::format::{BLOCK_HEADER_LEN, BlockType};
use bun_zstd_decoder::{Error, Result};

use crate::bytes::{is_one_byte_repeated, put};
use crate::entropy::{Changes, CodeKind, Entropy, Reuse};
use crate::parameters::Tuning;
use crate::sequences::{CodeCounts, Repeats, Sequence};
use crate::vec;

pub(crate) fn header(kind: BlockType, len: usize, is_last: bool) -> [u8; BLOCK_HEADER_LEN] {
    let value = u32::from(is_last) | (kind as u32) << 1 | (len as u32) << 3;
    let [low, middle, high, _] = value.to_le_bytes();
    [low, middle, high]
}

fn write_plain(header: [u8; BLOCK_HEADER_LEN], content: &[u8], out: &mut [u8]) -> Result<usize> {
    let (start, rest) = out.split_first_chunk_mut().ok_or(Error::DstSizeTooSmall)?;
    put(rest, content).ok_or(Error::DstSizeTooSmall)?;
    *start = header;
    Ok(header.len() + content.len())
}

pub(crate) fn write_raw(bytes: &[u8], is_last: bool, out: &mut [u8]) -> Result<usize> {
    write_plain(header(BlockType::Raw, bytes.len(), is_last), bytes, out)
}

pub(crate) fn write_repeated(bytes: &[u8], is_last: bool, out: &mut [u8]) -> Result<usize> {
    let first = bytes.get(..1).ok_or(Error::Internal)?;
    write_plain(header(BlockType::Rle, bytes.len(), is_last), first, out)
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
        let reuse = &mut self.entropy.codes.get_mut(CodeKind::Offset).reuse;
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
        out: &mut [u8],
    ) -> Result<usize> {
        const REPEATED_LEN_MAX: usize = 25;
        let (start, rest) = out.split_first_chunk_mut().ok_or(Error::DstSizeTooSmall)?;
        let compressed = self.entropy.compress(content, tuning, rest)?;
        let written = match compressed {
            _ if !self.is_at_start
                && compressed.is_none_or(|(len, _)| len < REPEATED_LEN_MAX)
                && is_one_byte_repeated(content.bytes) =>
            {
                write_repeated(content.bytes, is_last, out)?
            }
            Some((len, changes)) => {
                *start = header(BlockType::Compressed, len, is_last);
                self.accept(changes, content.repeats);
                BLOCK_HEADER_LEN + len
            }
            None => write_raw(content.bytes, is_last, out)?,
        };
        self.distrust_offset_table();
        Ok(written)
    }
}
