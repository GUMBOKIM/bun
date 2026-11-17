//! Cuts the input into sections, which threads compress. Each job owns what it works on.

use std::collections::VecDeque;
use std::sync::mpmc;
use std::sync::mpsc::{self, TryRecvError};

use bun_zstd_decoder::format::{BLOCK_LEN_MAX, CHECKSUM_LEN, FRAME_HEADER_LEN_MAX};
use bun_zstd_decoder::{Error, Output, Result, Status, XxHash64};

use crate::dictionary::Dictionary;
use crate::frame::{self, FrameWriter, Start};
use crate::matches::long_distance::{Candidates, Finder};
use crate::parameters::{LongDistance, Plan, Request, Strategy};
use crate::stream::{Backlog, EndDirective};
use crate::window::Input;
use crate::{vec, worst_case_len};

pub(crate) type Shared<T> = std::sync::Arc<T>;

/// After how much input a job reports output.
const CHUNK_LEN: usize = 4 * BLOCK_LEN_MAX;
const STACK_LEN: usize = 1 << 20;

/// Sees all sections, and so finds matches between them.
struct LongMatcher {
    parameters: LongDistance,
    finder: Finder,
    /// Starts over at its end.
    buffer: Vec<u8>,
    filled: usize,
    older_end: usize,
    section_len: usize,
}

impl LongMatcher {
    fn find(&mut self, section: &[u8], candidates: &mut Candidates) -> Result<()> {
        if self.filled + section.len() > self.buffer.capacity() {
            self.older_end = core::mem::take(&mut self.filled);
        }
        let start = self.filled;
        let used = self.buffer.get_mut(start..).unwrap_or_default();
        let (overwriting, appended) = section.split_at(section.len().min(used.len()));
        for (to, from) in used.iter_mut().zip(overwriting) {
            *to = *from;
        }
        self.buffer.extend_from_slice(appended);
        self.filled += section.len();

        let older = self.buffer.get(self.filled..self.older_end);
        let input = Input {
            segment: self.buffer.get(..self.filled).ok_or(Error::Internal)?,
            len: section.len(),
            follows_last: start != 0,
            older: older.unwrap_or_default(),
        };
        candidates.clear_for(self.section_len, &self.parameters)?;
        self.finder.admit(&input);
        self.finder.find(&input, start, section.len(), candidates)
    }
}

/// Goes from one job to the next.
struct Relay {
    hash: Option<XxHash64>,
    long_matcher: Option<LongMatcher>,
}

enum Report {
    Output(Vec<u8>),
    /// With the buffer that the input was in.
    Done(Vec<u8>),
    Failed(Error),
}

struct Job {
    plan: Plan,
    /// The end of the section before, and this one.
    input: Vec<u8>,
    start: usize,
    /// For the first one: the length of all input, and the dictionary.
    frame: Option<(Option<u64>, Option<Shared<Dictionary>>)>,
    is_last: bool,
    relay: (mpsc::Receiver<Box<Relay>>, mpsc::SyncSender<Box<Relay>>),
    reports: mpsc::Sender<Report>,
}

#[derive(Default)]
struct Worker {
    writer: Option<FrameWriter>,
    long_matches: Candidates,
    out: Vec<u8>,
}

impl Worker {
    /// An error can also mean that nobody waits for the output any more.
    fn compress(&mut self, job: &Job) -> Result<()> {
        let (earlier, section) = job
            .input
            .split_at_checked(job.start)
            .ok_or(Error::Internal)?;
        let mut relay = job.relay.0.recv().map_err(|_| Error::Internal)?;
        self.long_matches.clear();
        if let Some(matcher) = &mut relay.long_matcher {
            matcher.find(section, &mut self.long_matches)?;
        }
        if let Some(hash) = &mut relay.hash {
            hash.update(section);
        }
        job.relay.1.send(relay).map_err(|_| Error::Internal)?;

        let (len, dictionary, start) = match &job.frame {
            Some((len, dictionary)) => {
                let dictionary = dictionary.as_deref();
                (*len, dictionary, Start::Frame(dictionary))
            }
            None => (Some(section.len() as u64), None, Start::Part(earlier)),
        };
        let writer = match &mut self.writer {
            Some(writer) => {
                writer.restart(job.plan, len, start)?;
                writer
            }
            None => self.writer.insert(FrameWriter::new(job.plan, len, start)?),
        };
        core::mem::swap(&mut writer.long_matches, &mut self.long_matches);
        if self.out.is_empty() {
            let len = worst_case_len(CHUNK_LEN) + FRAME_HEADER_LEN_MAX + CHECKSUM_LEN;
            self.out = vec::filled(0, len + 2 * worst_case_len(BLOCK_LEN_MAX))?;
        }

        let mut start = job.start;
        loop {
            let end = job.input.len().min(start + CHUNK_LEN);
            let input = Input {
                segment: job.input.get(..end).ok_or(Error::Internal)?,
                len: end - start,
                follows_last: start != 0,
                older: dictionary.map(Dictionary::content).unwrap_or_default(),
            };
            let mut out = Output::slice(&mut self.out);
            let is_last = job.is_last && end == job.input.len();
            writer.write(&input, dictionary, is_last, &mut out)?;
            let written = out.written();
            drop(out);
            let output = vec::copied(self.out.get(..written).unwrap_or_default())?;
            let sent = job.reports.send(Report::Output(output));
            sent.map_err(|_| Error::Internal)?;
            if end == job.input.len() {
                return Ok(());
            }
            start = end;
        }
    }

    fn run(mut self, jobs: &mpmc::Receiver<Job>) {
        for job in jobs {
            let report = match self.compress(&job) {
                Ok(()) => Report::Done(job.input),
                Err(error) => Report::Failed(error),
            };
            let _ = job.reports.send(report);
        }
    }
}

/// The lengths of a section, and of what a job indexes of the one before.
fn section_and_overlap_len(plan: &Plan, request: &Request) -> (usize, usize) {
    let tuning = &plan.tuning;
    // For long distances the window is too large to go by.
    let section_log = match plan.long_distance {
        Some(_) => (tuning.cycle_log() + 3).max(21),
        None => (tuning.window_log + 2).max(20),
    };
    let section_log = section_log.min(30);
    let overlap_log = match (request.overlap_log, tuning.strategy) {
        (0, Strategy::Ultra2) => 9,
        (0, Strategy::Optimal | Strategy::Ultra) => 8,
        (0, Strategy::Lazy2 | Strategy::TreeLazy2) => 7,
        (0, _) => 6,
        (log, _) => log,
    };
    let overlap_len = match plan.long_distance {
        Some(_) => 1 << (tuning.window_log.min(section_log - 2) - (9 - overlap_log)),
        None if overlap_log > 1 => 1 << (tuning.window_log - (9 - overlap_log)),
        None => 0,
    };
    let section_len = match request.job_len {
        0 => 1 << section_log,
        len => len,
    };
    (section_len.max(overlap_len), overlap_len)
}

/// Dropping it stops the threads once they have output to report.
pub(crate) struct Workers {
    plan: Plan,
    /// Until the first job takes it.
    frame: Option<(Option<u64>, Option<Shared<Dictionary>>)>,
    input_left: Option<u64>,
    section_len: usize,
    overlap_len: usize,
    /// For the next job.
    input: Vec<u8>,
    start: usize,
    relay: mpsc::Receiver<Box<Relay>>,
    spare_inputs: Vec<Vec<u8>>,
    jobs: (mpmc::Sender<Job>, mpmc::Receiver<Job>),
    threads: usize,
    threads_max: usize,
    /// Of the jobs whose output is still to come, in order.
    reports: VecDeque<mpsc::Receiver<Report>>,
    backlog: Backlog,
    has_posted_last: bool,
    /// What follows the output of the jobs.
    needs_last_block: bool,
    needs_checksum: bool,
}

impl Workers {
    pub(crate) fn new(
        mut plan: Plan,
        request: &Request,
        input_len: Option<u64>,
        dictionary: Option<&Shared<Dictionary>>,
    ) -> Result<Self> {
        let (section_len, overlap_len) = section_and_overlap_len(&plan, request);
        let window_log = plan.tuning.window_log;
        let long_matcher = match plan.long_distance.take() {
            Some(parameters) => {
                let parameters = parameters.completed(&plan.tuning);
                Some(LongMatcher {
                    parameters,
                    finder: Finder::new(parameters, window_log)?,
                    buffer: vec::with_room((1 << window_log) + 2 * section_len)?,
                    filled: 0,
                    older_end: 0,
                    section_len,
                })
            }
            None => None,
        };
        let (sender, relay) = mpsc::sync_channel(1);
        let first = vec::boxed(Relay {
            hash: plan.header.has_checksum.then(XxHash64::new),
            long_matcher,
        })?;
        sender.send(first).map_err(|_| Error::Internal)?;

        let jobs_max = request.workers + 2;
        let mut reports = VecDeque::new();
        reports.try_reserve_exact(jobs_max)?;
        Ok(Self {
            plan,
            frame: Some((input_len, dictionary.map(Shared::clone))),
            input_left: input_len,
            section_len,
            overlap_len,
            input: vec::with_room(overlap_len + section_len)?,
            start: 0,
            relay,
            spare_inputs: vec::with_room(jobs_max)?,
            jobs: mpmc::channel(),
            threads: 0,
            threads_max: request.workers,
            reports,
            backlog: Backlog::default(),
            has_posted_last: false,
            needs_last_block: false,
            needs_checksum: false,
        })
    }

    fn filled(&self) -> usize {
        self.input.len() - self.start
    }

    fn fill(&mut self, input: &mut &[u8]) -> Result<()> {
        let len = input.len().min(self.section_len - self.filled());
        if let Some(left) = &mut self.input_left {
            *left = left.checked_sub(len as u64).ok_or(Error::SrcSizeWrong)?;
        }
        self.input
            .extend_from_slice(input.split_off(..len).unwrap_or_default());
        Ok(())
    }

    fn add_thread(&mut self) -> Result<()> {
        let jobs = mpmc::Receiver::clone(&self.jobs.1);
        let spawned = std::thread::Builder::new()
            .stack_size(STACK_LEN)
            .spawn(move || Worker::default().run(&jobs));
        match spawned {
            Ok(_) => self.threads += 1,
            Err(_) if self.threads == 0 => return Err(Error::MemoryAllocation),
            Err(_) => self.threads_max = self.threads,
        }
        Ok(())
    }

    fn post(&mut self, is_last: bool) -> Result<()> {
        if is_last && self.input_left.is_some_and(|left| left != 0) {
            return Err(Error::SrcSizeWrong);
        }
        let frame = self.frame.take();
        let mut plan = self.plan;
        self.has_posted_last = is_last;
        if frame.is_none() {
            self.needs_checksum = is_last && plan.header.has_checksum;
            plan.header.has_checksum = false;
            if self.filled() == 0 {
                self.needs_last_block = true;
                return Ok(());
            }
        }

        let mut next_input = Vec::new();
        if !is_last {
            next_input = self.spare_inputs.pop().unwrap_or_default();
            vec::clear_for(&mut next_input, self.overlap_len + self.section_len)?;
            let overlap = self.input.len() - self.overlap_len.min(self.filled());
            next_input.extend_from_slice(self.input.get(overlap..).unwrap_or_default());
        }
        if self.threads < self.threads_max.min(self.reports.len() + 1) {
            self.add_thread()?;
        }
        let (relay_sender, next_relay) = mpsc::sync_channel(1);
        let (reports, receiver) = mpsc::channel();
        let job = Job {
            plan,
            start: core::mem::replace(&mut self.start, next_input.len()),
            input: core::mem::replace(&mut self.input, next_input),
            frame,
            is_last,
            relay: (
                core::mem::replace(&mut self.relay, next_relay),
                relay_sender,
            ),
            reports,
        };
        self.jobs.0.send(job).map_err(|_| Error::Internal)?;
        self.reports.push_back(receiver);
        Ok(())
    }

    /// Fills the backlog, which is empty. Returns false if there is nothing to expect or, unless it `waits`, not yet.
    fn receive(&mut self, waits: bool) -> Result<bool> {
        let Some(oldest) = self.reports.front() else {
            return self.finish();
        };
        let report = match oldest.try_recv() {
            Err(TryRecvError::Empty) if waits => oldest.recv().ok(),
            Err(TryRecvError::Empty) => return Ok(false),
            report => report.ok(),
        };
        match report.ok_or(Error::Internal)? {
            Report::Output(bytes) => self.backlog.bytes = bytes,
            Report::Done(input) => {
                self.reports.pop_front();
                let _ = self.spare_inputs.push_within_capacity(input);
            }
            Report::Failed(error) => return Err(error),
        }
        Ok(true)
    }

    fn finish(&mut self) -> Result<bool> {
        if core::mem::take(&mut self.needs_last_block) {
            self.backlog.bytes = vec::copied(&frame::last_empty_block())?;
        } else if core::mem::take(&mut self.needs_checksum) {
            let relay = self.relay.recv().map_err(|_| Error::Internal)?;
            let hash = relay.hash.as_ref().ok_or(Error::Internal)?;
            self.backlog.bytes = vec::copied(&frame::checksum(hash))?;
        }
        Ok(!self.backlog.is_empty())
    }

    pub(crate) fn step(
        &mut self,
        input: &mut &[u8],
        out: &mut Output<'_>,
        directive: EndDirective,
    ) -> Result<Status> {
        if self.has_posted_last && (directive == EndDirective::Continue || !input.is_empty()) {
            return Err(Error::StageWrong);
        }
        loop {
            self.fill(input)?;
            let is_last = directive == EndDirective::End && input.is_empty();
            let is_due = self.filled() == self.section_len
                || (directive != EndDirective::Continue && self.filled() != 0)
                || (is_last && !self.has_posted_last);
            if is_due && self.reports.len() < self.threads_max + 2 {
                self.post(is_last)?;
                continue;
            }

            self.backlog.flush(out);
            if !self.backlog.is_empty() {
                return Ok(Status::Pending);
            }
            if !self.receive(is_due || directive != EndDirective::Continue)? {
                let is_complete = self.reports.is_empty() && self.filled() == 0;
                return Ok(if is_complete {
                    Status::Complete
                } else {
                    Status::Pending
                });
            }
        }
    }
}
