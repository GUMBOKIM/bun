//! Frame and block headers.

use crate::bits::load_from;
use crate::error::{Error, Result};
use crate::format::{
    BLOCK_HEADER_LEN, BLOCK_LEN_MAX, BlockType, CHECKSUM_LEN, FRAME_MAGIC, SKIPPABLE_MAGIC,
    WINDOW_LOG_MAX, WINDOW_LOG_MIN,
};

const MAGIC_LEN: usize = 4;
/// The magic number and the descriptor, which tells the length of the rest.
pub(crate) const HEADER_PREFIX_LEN: usize = MAGIC_LEN + 1;
const HEADER_LEN_MIN: usize = HEADER_PREFIX_LEN + 1;
const SKIPPABLE_HEADER_LEN: usize = MAGIC_LEN + 4;
/// `ZSTD_CONTENTSIZE_UNKNOWN` and `ZSTD_CONTENTSIZE_ERROR`, which zstd does not tell from what a header says.
const UNKNOWN_SIZE: u64 = u64::MAX;
const INVALID_SIZE: u64 = u64::MAX - 1;

#[derive(Clone, Copy)]
pub(crate) struct FrameHeader {
    pub(crate) len: usize,
    pub(crate) content_size: Option<u64>,
    pub(crate) window_size: u64,
    /// 0 if there is none.
    pub(crate) dictionary_id: u32,
    pub(crate) has_checksum: bool,
}

pub(crate) enum Header {
    Frame(FrameHeader),
    Skippable {
        content_len: u32,
    },
    /// It takes at least this many bytes.
    Incomplete(usize),
}

fn is_skippable(magic: u32) -> bool {
    magic & !15 == SKIPPABLE_MAGIC
}

/// The lengths of the window descriptor, the dictionary ID and the content size.
fn field_lens(descriptor: u8) -> [usize; 3] {
    let is_single_segment = descriptor & 0x20 != 0;
    let content_size = match descriptor >> 6 {
        0 => is_single_segment.into(),
        flag => 1 << flag,
    };
    [
        (!is_single_segment).into(),
        1 << (descriptor & 3) >> 1,
        content_size,
    ]
}

fn header_len(descriptor: u8) -> usize {
    HEADER_PREFIX_LEN + field_lens(descriptor).into_iter().sum::<usize>()
}

/// Whether `bytes` start with a magic number, or with as much of one as they hold. There has to be at least one.
pub fn may_start_frame(bytes: &[u8]) -> bool {
    let missing = u32::MAX.unbounded_shl(8 * bytes.len().min(MAGIC_LEN) as u32);
    let or = |magic: u32| load_from(bytes, 0) as u32 | magic & missing;
    !bytes.is_empty() && (or(FRAME_MAGIC) == FRAME_MAGIC || is_skippable(or(SKIPPABLE_MAGIC)))
}

impl Header {
    pub(crate) fn read(src: &[u8]) -> Result<Self> {
        let Some((&[m0, m1, m2, m3, descriptor], fields)) = src.split_first_chunk() else {
            if !src.is_empty() && !may_start_frame(src) {
                return Err(Error::PrefixUnknown);
            }
            return Ok(Self::Incomplete(HEADER_PREFIX_LEN));
        };
        let magic = u32::from_le_bytes([m0, m1, m2, m3]);
        if is_skippable(magic) {
            return Ok(match fields {
                &[s1, s2, s3, ..] => Self::Skippable {
                    content_len: u32::from_le_bytes([descriptor, s1, s2, s3]),
                },
                _ => Self::Incomplete(SKIPPABLE_HEADER_LEN),
            });
        }
        if magic != FRAME_MAGIC {
            return Err(Error::PrefixUnknown);
        }
        let len = header_len(descriptor);
        if src.len() < len {
            return Ok(Self::Incomplete(len));
        }
        if descriptor & 0x08 != 0 {
            return Err(Error::FrameParameterUnsupported);
        }

        let [window_len, id_len, size_len] = field_lens(descriptor);
        let field = |start, len| load_from(fields, start) & !u64::MAX.unbounded_shl(8 * len as u32);
        let content_size = match size_len {
            0 => None,
            2 => Some(field(window_len + id_len, size_len) + 256),
            _ => Some(field(window_len + id_len, size_len)),
        };
        let window_size = match content_size {
            Some(size) if window_len == 0 => size,
            _ => {
                let descriptor = field(0, window_len);
                let log = (descriptor >> 3) as u32 + WINDOW_LOG_MIN;
                if log > WINDOW_LOG_MAX {
                    return Err(Error::WindowTooLarge);
                }
                (8 + (descriptor & 7)) << (log - 3)
            }
        };
        Ok(Self::Frame(FrameHeader {
            len,
            content_size: content_size.filter(|&size| size != UNKNOWN_SIZE),
            window_size,
            dictionary_id: field(window_len, id_len) as u32,
            has_checksum: descriptor & 0x04 != 0,
        }))
    }
}

impl FrameHeader {
    /// For a frame that has to be all there.
    pub(crate) fn read_whole(src: &[u8]) -> Result<Self> {
        let len = src
            .get(MAGIC_LEN)
            .map_or(HEADER_LEN_MIN, |&it| header_len(it));
        if src.len() < len + BLOCK_HEADER_LEN {
            return Err(Error::SrcSizeWrong);
        }
        match Header::read(src)? {
            Header::Frame(header) => Ok(header),
            _ => Err(Error::SrcSizeWrong),
        }
    }

    /// For what a block holds, and for what it decodes to.
    pub(crate) fn block_len_max(&self) -> usize {
        self.window_size.min(BLOCK_LEN_MAX as u64) as usize
    }
}

#[derive(Clone, Copy)]
pub(crate) struct BlockHeader {
    pub(crate) block_type: BlockType,
    pub(crate) is_last: bool,
    size: u32,
}

impl BlockHeader {
    pub(crate) fn new([b0, b1, b2]: [u8; BLOCK_HEADER_LEN]) -> Result<Self> {
        let bits = u32::from_le_bytes([b0, b1, b2, 0]);
        Ok(Self {
            block_type: match bits >> 1 & 3 {
                0 => BlockType::Raw,
                1 => BlockType::Rle,
                2 => BlockType::Compressed,
                _ => return Err(Error::CorruptionDetected),
            },
            is_last: bits & 1 != 0,
            size: bits >> 3,
        })
    }

    /// How many bytes follow the header.
    pub(crate) fn content_len(self) -> usize {
        match self.block_type {
            BlockType::Rle => 1,
            _ => self.size as usize,
        }
    }

    /// How often the byte of an RLE block is repeated.
    pub(crate) fn repeats(self) -> usize {
        self.size as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameLen {
    Complete(usize),
    /// More than there is. The frame may turn out to be longer still.
    AtLeast(usize),
}

/// With the header.
pub(crate) fn skippable_len(content_len: u32) -> Result<usize> {
    let len = content_len.checked_add(SKIPPABLE_HEADER_LEN as u32);
    len.map(|len| len as usize)
        .ok_or(Error::FrameParameterUnsupported)
}

/// The length of the frame that `prefix` starts with.
pub fn frame_len(prefix: &[u8]) -> Result<FrameLen> {
    let len = match Header::read(prefix)? {
        Header::Incomplete(len) => len,
        Header::Skippable { content_len } => skippable_len(content_len)?,
        Header::Frame(header) => {
            let mut len = header.len;
            loop {
                let Some(block) = prefix.get(len..).and_then(<[u8]>::first_chunk) else {
                    return Ok(FrameLen::AtLeast(len + BLOCK_HEADER_LEN));
                };
                let block = BlockHeader::new(*block)?;
                len += BLOCK_HEADER_LEN + block.content_len();
                if block.is_last {
                    break len + if header.has_checksum { CHECKSUM_LEN } else { 0 };
                }
            }
        }
    };
    Ok(if len <= prefix.len() {
        FrameLen::Complete(len)
    } else {
        FrameLen::AtLeast(len)
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentSize {
    /// What the input claims.
    Known(u64),
    Unknown,
    Invalid,
}

impl ContentSize {
    fn new(size: u64) -> Self {
        match size {
            UNKNOWN_SIZE => Self::Unknown,
            INVALID_SIZE => Self::Invalid,
            _ => Self::Known(size),
        }
    }
}

/// Of the frame that `src` starts with. Only its header has to be there.
pub fn frame_content_size(src: &[u8]) -> ContentSize {
    match Header::read(src) {
        Ok(Header::Frame(header)) => ContentSize::new(header.content_size.unwrap_or(UNKNOWN_SIZE)),
        Ok(Header::Skippable { .. }) => ContentSize::Known(0),
        Ok(Header::Incomplete(_)) | Err(_) => ContentSize::Invalid,
    }
}

/// Of all the frames that `src` consists of.
pub fn decompressed_size(mut src: &[u8]) -> ContentSize {
    let mut total = 0u64;
    while !src.is_empty() {
        let size = match frame_content_size(src) {
            ContentSize::Known(size) => size,
            other => return other,
        };
        let (Some(sum), Ok(FrameLen::Complete(len))) = (total.checked_add(size), frame_len(src))
        else {
            return ContentSize::Invalid;
        };
        total = sum;
        src = src.get(len..).unwrap_or_default();
    }
    ContentSize::new(total)
}
