//! Decompression of input that comes in pieces, into output that is taken in pieces.

use crate::block::BlockDecoder;
use crate::decompress::{check_content_size, check_dictionary, decompress_frame};
use crate::dictionary::Dictionary;
use crate::error::{Error, Result};
use crate::format::{
    BLOCK_HEADER_LEN, BlockType, CHECKSUM_LEN, FRAME_HEADER_LEN_MAX, WINDOW_LOG_LIMIT_DEFAULT,
    WINDOW_LOG_MAX, WINDOW_LOG_MIN,
};
use crate::frame::{BlockHeader, FrameHeader, FrameLen, Header, frame_len};
use crate::output::Output;
use crate::sequences::{Destination, OVERSHOOT};
use crate::xxhash::XxHash64;

/// `ZSTD_d_windowLogMax`
const WINDOW_LOG_MAX_KEY: u32 = 100;
const WINDOW_SIZE_MAX_DEFAULT: u64 = (1 << WINDOW_LOG_LIMIT_DEFAULT) + 1;
const IDLE_STEPS_MAX: u32 = 16;
/// The ring shrinks once it has been this many times too large...
const OVERSIZE_FACTOR: usize = 3;
/// ...for this many frames in a row.
const OVERSIZED_FRAMES_MAX: u32 = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// A frame, or a flush, is over and all of it was written.
    Complete,
    Pending,
}

/// What comes next in the input.
#[derive(Clone, Copy)]
enum Part {
    BlockHeader,
    /// Raw blocks are taken in pieces.
    BlockContent {
        header: BlockHeader,
        left: usize,
    },
    Checksum,
    Nothing,
}

enum Stage {
    Idle,
    Header {
        bytes: [u8; FRAME_HEADER_LEN_MAX],
        len: usize,
    },
    Body(Frame),
    Skipped {
        left: usize,
    },
}

struct Frame {
    header: FrameHeader,
    next: Part,
    decoded_size: u64,
    hash: Option<XxHash64>,
    /// The last byte of the frame counts as consumed once all output is out.
    withholds_last_byte: bool,
}

/// The window, and what is yet to be written to the output.
#[derive(Default)]
struct Ring {
    bytes: Vec<u8>,
    /// What the frame needs. `bytes` may have more from an earlier one.
    capacity: usize,
    oversized_frames: u32,
    /// Of what is yet to be written.
    start: usize,
    end: usize,
    /// Where `end` was when it went back to 0.
    previous_end: Option<usize>,
}

pub struct Decoder {
    stage: Stage,
    blocks: BlockDecoder,
    dictionary: Option<Dictionary<'static>>,
    window_size_max: u64,
    /// The start of a [`Part`] that has to be in one piece.
    gathered: Vec<u8>,
    ring: Ring,
    idle_steps: u32,
    error: Option<Error>,
}

fn take_up_to<'a>(input: &mut &'a [u8], len: usize) -> &'a [u8] {
    let (taken, rest) = input.split_at_checked(len).unwrap_or((input, &[]));
    *input = rest;
    taken
}

impl Decoder {
    pub fn new() -> Self {
        Self {
            stage: Stage::Idle,
            blocks: BlockDecoder::default(),
            dictionary: None,
            window_size_max: WINDOW_SIZE_MAX_DEFAULT,
            gathered: Vec::new(),
            ring: Ring::default(),
            idle_steps: 0,
            error: None,
        }
    }

    fn check_idle(&self) -> Result<()> {
        match (self.error, &self.stage) {
            (Some(error), _) => Err(error),
            (None, Stage::Idle) => Ok(()),
            _ => Err(Error::StageWrong),
        }
    }

    /// For all frames from the next on. See [`Dictionary::new`].
    pub fn load_dictionary(&mut self, bytes: &[u8]) -> Result<()> {
        self.check_idle()?;
        self.dictionary = None;
        self.dictionary = Some(Dictionary::copied(bytes)?);
        Ok(())
    }

    /// `key` is a `ZSTD_dParameter`.
    pub fn set_parameter(&mut self, key: u32, value: i32) -> Result<()> {
        self.check_idle()?;
        if key != WINDOW_LOG_MAX_KEY {
            return Err(Error::ParameterUnsupported);
        }
        self.window_size_max = match value as u32 {
            0 => 1 << WINDOW_LOG_LIMIT_DEFAULT,
            log @ WINDOW_LOG_MIN..=WINDOW_LOG_MAX => 1 << log,
            _ => return Err(Error::ParameterOutOfBound),
        };
        Ok(())
    }

    /// Forgets the frame in progress, the dictionary, the parameters and the error.
    pub fn reset(&mut self) {
        self.reset_session();
        self.dictionary = None;
        self.window_size_max = WINDOW_SIZE_MAX_DEFAULT;
    }

    /// Forgets the frame in progress and the error.
    pub fn reset_session(&mut self) {
        self.stage = Stage::Idle;
        self.idle_steps = 0;
        self.error = None;
    }

    /// Goes as far as `input` and the room in `out` allow, up to the end of a frame. Returns how much of `input` it
    /// consumed. An error takes back what the step wrote, and stays until [`reset`](Self::reset).
    pub fn step(&mut self, input: &[u8], out: &mut Output<'_>) -> (usize, Result<Status>) {
        let written = out.written();
        match self.error.map_or_else(|| self.try_step(input, out), Err) {
            Ok((consumed, status)) => (consumed, Ok(status)),
            Err(error) => {
                out.rewind(written);
                self.error = Some(error);
                (0, Err(error))
            }
        }
    }

    fn try_step(&mut self, input: &[u8], out: &mut Output<'_>) -> Result<(usize, Status)> {
        let written = out.written();
        let mut rest = input;
        if let Stage::Idle | Stage::Header { .. } = self.stage {
            let had_nothing = matches!(self.stage, Stage::Idle | Stage::Header { len: 0, .. });
            let Some(header) = self.gather_header(&mut rest)? else {
                return Ok((input.len(), Status::Pending));
            };
            if let Header::Frame(FrameHeader {
                content_size: Some(size),
                ..
            }) = header
                && had_nothing
                && size <= out.room() as u64
                && let Ok(FrameLen::Complete(len)) = frame_len(input)
            {
                let frame = input.get(..len).ok_or(Error::Internal)?;
                decompress_frame(&mut self.blocks, frame, out, self.dictionary.as_ref())?;
                self.stage = Stage::Idle;
                self.idle_steps = 0;
                return Ok((len, Status::Complete));
            }
            self.stage = match header {
                Header::Frame(header) => Stage::Body(self.start_frame(header)?),
                Header::Skippable { content_len } => Stage::Skipped {
                    left: content_len as usize,
                },
                Header::Incomplete(_) => return Err(Error::Internal),
            };
        }
        if let Stage::Skipped { left } = &mut self.stage {
            *left -= take_up_to(&mut rest, *left).len();
        } else {
            self.decode_body(&mut rest, out)?;
        }

        let consumed = input.len() - rest.len();
        if consumed != 0 || out.written() != written {
            self.idle_steps = 0;
        } else {
            self.idle_steps = self.idle_steps.saturating_add(1);
            if self.idle_steps >= IDLE_STEPS_MAX && out.room() == 0 {
                return Err(Error::NoForwardProgressDestFull);
            }
            if self.idle_steps >= IDLE_STEPS_MAX && rest.is_empty() {
                return Err(Error::NoForwardProgressInputEmpty);
            }
        }

        let frame = match &mut self.stage {
            Stage::Body(frame) if matches!(frame.next, Part::Nothing) => frame,
            Stage::Skipped { left: 0 } => {
                self.stage = Stage::Idle;
                return Ok((consumed, Status::Complete));
            }
            _ => return Ok((consumed, Status::Pending)),
        };
        if self.ring.start != self.ring.end {
            let newly_withheld = usize::from(!frame.withholds_last_byte);
            frame.withholds_last_byte = true;
            let consumed = consumed.checked_sub(newly_withheld);
            return Ok((consumed.ok_or(Error::Internal)?, Status::Pending));
        }
        if frame.withholds_last_byte && rest.is_empty() {
            return Ok((consumed, Status::Pending));
        }
        let released = usize::from(frame.withholds_last_byte);
        self.stage = Stage::Idle;
        Ok((consumed + released, Status::Complete))
    }

    /// `None` if `input` is used up before the header is complete.
    fn gather_header(&mut self, input: &mut &[u8]) -> Result<Option<Header>> {
        if let Stage::Idle = self.stage {
            self.stage = Stage::Header {
                bytes: [0; _],
                len: 0,
            };
        }
        let Stage::Header { bytes, len } = &mut self.stage else {
            return Err(Error::Internal);
        };
        loop {
            let header = Header::read(bytes.get(..*len).ok_or(Error::Internal)?)?;
            let Header::Incomplete(needed) = header else {
                return Ok(Some(header));
            };
            if input.is_empty() {
                return Ok(None);
            }
            for to in bytes.get_mut(*len..needed).ok_or(Error::Internal)? {
                let Some(&byte) = input.split_off_first() else {
                    break;
                };
                *to = byte;
                *len += 1;
            }
        }
    }

    fn start_frame(&mut self, header: FrameHeader) -> Result<Frame> {
        check_dictionary(&header, self.dictionary.as_ref())?;
        let window_size = header.window_size.max(1 << WINDOW_LOG_MIN);
        if window_size > self.window_size_max {
            return Err(Error::WindowTooLarge);
        }
        let capacity = window_size + 2 * (header.block_len_max() + OVERSHOOT) as u64;
        let capacity = capacity.min(header.content_size.unwrap_or(u64::MAX));
        self.ring.start_frame(capacity as usize)?;
        self.gathered.clear();
        self.blocks.start_frame(self.dictionary.as_ref());
        Ok(Frame {
            header,
            next: Part::BlockHeader,
            decoded_size: 0,
            hash: header.has_checksum.then(XxHash64::new),
            withholds_last_byte: false,
        })
    }

    fn decode_body(&mut self, input: &mut &[u8], out: &mut Output<'_>) -> Result<()> {
        let Stage::Body(frame) = &mut self.stage else {
            return Err(Error::Internal);
        };
        while self.ring.flush(out, &frame.header) {
            let needed = match frame.next {
                Part::Nothing => return Ok(()),
                Part::BlockHeader => BLOCK_HEADER_LEN,
                Part::Checksum => CHECKSUM_LEN,
                Part::BlockContent { header, left } if header.block_type == BlockType::Raw => {
                    input.len().max(1).min(left)
                }
                Part::BlockContent { left, .. } => left,
            };
            let missing = needed - self.gathered.len();
            let piece = take_up_to(input, missing);
            let part = if piece.len() == needed {
                piece
            } else if piece.is_empty() {
                return Ok(());
            } else {
                self.gathered.try_reserve(missing)?;
                self.gathered.extend_from_slice(piece);
                if piece.len() < missing {
                    return Ok(());
                }
                &self.gathered
            };
            frame.decode(
                part,
                &mut self.blocks,
                self.dictionary.as_ref(),
                &mut self.ring,
            )?;
            self.gathered.clear();
        }
        Ok(())
    }
}

impl Frame {
    /// `part` is what `next` says.
    fn decode(
        &mut self,
        part: &[u8],
        blocks: &mut BlockDecoder,
        dictionary: Option<&Dictionary<'_>>,
        ring: &mut Ring,
    ) -> Result<()> {
        let block_len_max = self.header.block_len_max();
        let (header, left) = match self.next {
            Part::BlockHeader => {
                let header = BlockHeader::new(*part.first_chunk().ok_or(Error::Internal)?)?;
                let left = header.content_len();
                if left > block_len_max || (left == 0 && header.block_type != BlockType::Raw) {
                    return Err(Error::CorruptionDetected);
                }
                if left != 0 {
                    self.next = Part::BlockContent { header, left };
                    return Ok(());
                }
                (header, 0)
            }
            Part::BlockContent { header, left } => {
                let decoded = ring.decode(header, part, block_len_max, blocks, dictionary)?;
                self.decoded_size += decoded.len() as u64;
                if self.decoded_size > self.header.content_size.unwrap_or(u64::MAX) {
                    return Err(Error::CorruptionDetected);
                }
                if let Some(hash) = &mut self.hash {
                    hash.update(decoded);
                }
                (header, left - part.len())
            }
            Part::Checksum => {
                let checksum = u32::from_le_bytes(*part.first_chunk().ok_or(Error::Internal)?);
                if self.hash.as_ref().map(|hash| hash.digest() as u32) != Some(checksum) {
                    return Err(Error::ChecksumWrong);
                }
                self.next = Part::Nothing;
                return Ok(());
            }
            Part::Nothing => return Err(Error::Internal),
        };
        self.next = if left != 0 {
            Part::BlockContent { header, left }
        } else if !header.is_last {
            Part::BlockHeader
        } else {
            check_content_size(&self.header, self.decoded_size)?;
            if self.header.has_checksum {
                Part::Checksum
            } else {
                Part::Nothing
            }
        };
        Ok(())
    }
}

impl Ring {
    fn start_frame(&mut self, capacity: usize) -> Result<()> {
        let needed = capacity + OVERSHOOT;
        let is_oversized = self.bytes.capacity() / OVERSIZE_FACTOR >= needed;
        self.oversized_frames = if is_oversized {
            self.oversized_frames + 1
        } else {
            0
        };
        if self.bytes.capacity() < needed || self.oversized_frames >= OVERSIZED_FRAMES_MAX {
            *self = Self::default();
            self.bytes.try_reserve_exact(needed)?;
        }
        self.capacity = capacity;
        (self.start, self.end, self.previous_end) = (0, 0, None);
        Ok(())
    }

    /// Returns whether all is out.
    fn flush(&mut self, out: &mut Output<'_>, header: &FrameHeader) -> bool {
        if self.start == self.end {
            return true;
        }
        self.start += out.append(self.bytes.get(self.start..self.end).unwrap_or_default());
        if self.start != self.end {
            return false;
        }
        let holds_all = header
            .content_size
            .is_some_and(|size| size <= self.capacity as u64);
        if !holds_all && self.end + header.block_len_max() > self.capacity {
            self.previous_end = Some(self.end);
            (self.start, self.end) = (0, 0);
        }
        true
    }

    /// Returns what `content` decoded to.
    fn decode(
        &mut self,
        header: BlockHeader,
        content: &[u8],
        block_len_max: usize,
        blocks: &mut BlockDecoder,
        dictionary: Option<&Dictionary<'_>>,
    ) -> Result<&[u8]> {
        let reach = self.end + block_len_max + OVERSHOOT;
        let (mut out, external) = match self.previous_end {
            Some(previous_end) if previous_end > reach => {
                let parts = self.bytes.split_at_mut_checked(reach);
                let (current, previous) = parts.ok_or(Error::Internal)?;
                let previous = previous
                    .get(..previous_end - reach)
                    .ok_or(Error::Internal)?;
                (
                    Output::resume_slice(current, self.end, self.capacity),
                    previous,
                )
            }
            previous_end => {
                let dictionary = dictionary.filter(|_| previous_end.is_none());
                let out = Output::resume_ring(&mut self.bytes, self.end, self.capacity);
                (out, dictionary.map(Dictionary::history).unwrap_or_default())
            }
        };
        let destination = Destination {
            out: &mut out,
            history_start: 0,
            external,
            block_len_max,
        };
        let len = match header.block_type {
            BlockType::Compressed => blocks.decode(content, dictionary, destination),
            BlockType::Raw => destination.append(content),
            BlockType::Rle => destination.append_repeated(content, header.repeats()),
        }?;
        drop(out);
        self.end += len;
        self.bytes.get(self.start..self.end).ok_or(Error::Internal)
    }
}
