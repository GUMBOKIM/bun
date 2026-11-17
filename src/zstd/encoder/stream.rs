use core::ops::Range;

use bun_zstd_decoder::{Error, Output, Result, Status};

use crate::dictionary::Dictionary;
use crate::frame::{FrameWriter, Start};
use crate::parameters::{DEFAULT_LEVEL, JOB_LEN_MIN, Plan, Purpose, Request};
use crate::window::Input;
use crate::workers::{Shared, Workers};
use crate::{vec, worst_case_len};

/// The values are those of `ZSTD_EndDirective`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EndDirective {
    Continue = 0,
    /// Makes all input so far decodable.
    Flush = 1,
    /// Ends the frame.
    End = 2,
}

/// The output that there was no room for.
#[derive(Default)]
pub(crate) struct Backlog {
    bytes: Vec<u8>,
    left: Range<usize>,
}

impl Backlog {
    pub(crate) fn of(bytes: Vec<u8>) -> Self {
        let left = 0..bytes.len();
        Self { bytes, left }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.left.is_empty()
    }

    fn fill(
        &mut self,
        room: usize,
        write: impl FnOnce(&mut Output<'_>) -> Result<()>,
    ) -> Result<()> {
        if self.bytes.is_empty() {
            self.bytes = vec::filled(0, room)?;
        }
        let mut out = Output::slice(&mut self.bytes);
        write(&mut out)?;
        self.left = 0..out.written();
        Ok(())
    }

    pub(crate) fn flush(&mut self, out: &mut Output<'_>) {
        self.left.start += out.append(self.bytes.get(self.left.clone()).unwrap_or_default());
    }
}

/// Where the segment before the current one is.
#[derive(Clone, Copy)]
enum Older {
    InDictionary,
    InBuffer { end: usize },
}

/// Collects input into blocks. At the end of its buffer, it starts over.
struct Buffered {
    writer: FrameWriter,
    buffer: Vec<u8>,
    capacity: usize,
    older: Older,
    /// Of the block that is being filled.
    start: usize,
    filled: usize,
    target: usize,
    backlog: Backlog,
    has_ended: bool,
}

impl Buffered {
    fn new(plan: Plan, input_len: Option<u64>, dictionary: Option<&Dictionary>) -> Result<Self> {
        let writer = FrameWriter::new(plan, input_len, Start::Frame(dictionary))?;
        let block_len = writer.block_len_max();
        Ok(Self {
            buffer: Vec::new(),
            capacity: writer.window_len() + block_len,
            older: Older::InDictionary,
            start: 0,
            filled: 0,
            // If that is all, the end of the frame does not need a block of its own.
            target: block_len + usize::from(input_len == Some(block_len as u64)),
            backlog: Backlog::default(),
            has_ended: false,
            writer,
        })
    }

    /// What is left of the older segment, if the buffer is overwritten up to `overwritten`.
    fn older<'a>(
        older: Older,
        buffer: &'a [u8],
        overwritten: usize,
        dictionary: Option<&'a Dictionary>,
    ) -> &'a [u8] {
        match older {
            Older::InDictionary => dictionary.map(Dictionary::content),
            Older::InBuffer { end } => buffer.get(overwritten..end),
        }
        .unwrap_or_default()
    }

    fn fill(&mut self, input: &mut &[u8]) -> Result<()> {
        if self.buffer.capacity() == 0 {
            vec::clear_for(&mut self.buffer, self.capacity)?;
        }
        let len = input.len().min(self.target - self.filled);
        let mut taken = input.split_off(..len).unwrap_or_default();
        let used = self.buffer.get_mut(self.filled..).unwrap_or_default();
        let overwritten = taken.split_off(..len.min(used.len())).unwrap_or_default();
        for (to, from) in used.iter_mut().zip(overwritten) {
            *to = *from;
        }
        self.buffer.extend_from_slice(taken);
        self.filled += len;
        Ok(())
    }

    fn write_block(
        &mut self,
        dictionary: Option<&Dictionary>,
        is_last: bool,
        out: &mut Output<'_>,
    ) -> Result<()> {
        let input = Input {
            segment: self.buffer.get(..self.filled).ok_or(Error::Internal)?,
            len: self.filled - self.start,
            follows_last: self.start != 0,
            older: Self::older(self.older, &self.buffer, self.filled, dictionary),
        };
        let block_len = self.writer.block_len_max();
        if out.room() >= worst_case_len(input.len) {
            self.writer.write(&input, dictionary, is_last, out)?;
        } else {
            self.backlog.fill(worst_case_len(block_len) + 1, |out| {
                self.writer.write(&input, dictionary, is_last, out)
            })?;
        }
        self.has_ended = is_last;

        if self.filled + block_len > self.capacity {
            self.older = Older::InBuffer { end: self.filled };
            self.filled = 0;
        }
        self.start = self.filled;
        self.target = self.filled + block_len;
        Ok(())
    }

    fn step(
        &mut self,
        input: &mut &[u8],
        dictionary: Option<&Dictionary>,
        out: &mut Output<'_>,
        directive: EndDirective,
    ) -> Result<Status> {
        loop {
            self.backlog.flush(out);
            if !self.backlog.is_empty() {
                return Ok(Status::Pending);
            }
            if self.has_ended {
                return Ok(Status::Complete);
            }
            if directive == EndDirective::End
                && self.filled == 0
                && out.room() >= worst_case_len(input.len())
            {
                let older = Self::older(self.older, &self.buffer, 0, dictionary);
                let input = Input::alone(core::mem::take(input), older);
                self.writer.write(&input, dictionary, true, out)?;
                self.has_ended = true;
                return Ok(Status::Complete);
            }

            self.fill(input)?;
            let is_done = match directive {
                EndDirective::Continue => self.filled < self.target,
                EndDirective::Flush => self.filled == self.start,
                EndDirective::End => false,
            };
            if is_done {
                return Ok(Status::Complete);
            }
            let is_last = directive == EndDirective::End && input.is_empty();
            self.write_block(dictionary, is_last, out)?;
        }
    }
}

enum Frame {
    Buffered(Box<Buffered>),
    Parallel(Box<Workers>),
}

/// Writes one frame after the other.
pub struct Encoder {
    request: Request,
    /// Of the next frame.
    pledged_len: Option<u64>,
    /// Until the first frame starts, when the parameters are known.
    dictionary_bytes: Vec<u8>,
    dictionary: Option<Shared<Dictionary>>,
    frame: Option<Frame>,
    error: Option<Error>,
}

impl Encoder {
    pub fn new() -> Self {
        Self {
            request: Request::new(DEFAULT_LEVEL),
            pledged_len: None,
            dictionary_bytes: Vec::new(),
            dictionary: None,
            frame: None,
            error: None,
        }
    }

    fn ensure_idle(&self) -> Result<()> {
        match (self.error, &self.frame) {
            (Some(error), _) => Err(error),
            (None, Some(_)) => Err(Error::StageWrong),
            (None, None) => Ok(()),
        }
    }

    /// For all frames from the next one on.
    pub fn load_dictionary(&mut self, dictionary: &[u8]) -> Result<()> {
        self.ensure_idle()?;
        self.dictionary = None;
        self.dictionary_bytes = vec::copied(dictionary)?;
        Ok(())
    }

    /// How long the input of the next frame is going to be.
    pub fn set_pledged_len(&mut self, len: Option<u64>) -> Result<()> {
        self.ensure_idle()?;
        self.pledged_len = len;
        Ok(())
    }

    /// `key` is a `ZSTD_cParameter`.
    pub fn set_parameter(&mut self, key: u32, value: i32) -> Result<()> {
        self.ensure_idle()?;
        self.request.set(key, value)
    }

    /// Forgets the frame in progress and an error, but not the parameters and the dictionary.
    pub fn reset_session(&mut self) {
        (self.frame, self.error, self.pledged_len) = (None, None, None);
    }

    fn start_frame(&mut self, input_len: Option<u64>) -> Result<Frame> {
        if !self.dictionary_bytes.is_empty() {
            let dictionary = Dictionary::new(&mut self.dictionary_bytes, &self.request)?;
            let dictionary = Shared::try_new(dictionary).map_err(|_| Error::MemoryAllocation)?;
            self.dictionary = Some(dictionary);
        }
        let dictionary = self.dictionary.as_deref();
        let copied = dictionary.filter(|it| !it.is_attached_for(input_len));
        let dictionary_len = copied.map_or(0, Dictionary::len);
        let tuning = self
            .request
            .tuning(input_len, Purpose::Frame { dictionary_len });
        let plan = self.request.plan(tuning);

        if self.request.workers > 0 && input_len.is_none_or(|len| len > JOB_LEN_MIN as u64) {
            let workers = Workers::new(plan, &self.request, input_len, self.dictionary.as_ref())?;
            return Ok(Frame::Parallel(vec::boxed(workers)?));
        }
        let buffered = Buffered::new(plan, input_len, dictionary)?;
        Ok(Frame::Buffered(vec::boxed(buffered)?))
    }

    fn try_step(
        &mut self,
        input: &mut &[u8],
        out: &mut Output<'_>,
        directive: EndDirective,
    ) -> Result<Status> {
        let frame = match &mut self.frame {
            Some(frame) => frame,
            None => {
                let pledged_len = self.pledged_len.take();
                let frame = self.start_frame(match directive {
                    EndDirective::End => Some(input.len() as u64),
                    _ => pledged_len,
                })?;
                self.frame.insert(frame)
            }
        };
        let (status, has_ended) = match frame {
            Frame::Buffered(frame) => {
                let status = frame.step(input, self.dictionary.as_deref(), out, directive)?;
                (status, frame.has_ended)
            }
            Frame::Parallel(frame) => (
                frame.step(input, out, directive)?,
                directive == EndDirective::End,
            ),
        };
        if has_ended && status == Status::Complete {
            self.frame = None;
        }
        Ok(status)
    }

    /// Takes all of `input` unless `out` fills up. Returns how much it took, and whether there is output left that
    /// `directive` asks for. After an error, every call returns that error until [`reset_session`](Self::reset_session).
    pub fn step(
        &mut self,
        input: &[u8],
        out: &mut Output<'_>,
        directive: EndDirective,
    ) -> (usize, Result<Status>) {
        if let Some(error) = self.error {
            return (0, Err(error));
        }
        let (mut left, written_before) = (input, out.written());
        match self.try_step(&mut left, out, directive) {
            Ok(status) => (input.len() - left.len(), Ok(status)),
            Err(error) => {
                out.rewind(written_before);
                (self.error, self.frame) = (Some(error), None);
                (0, Err(error))
            }
        }
    }
}
