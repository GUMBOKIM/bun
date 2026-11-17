//! Decoding and executing the sequences of a compressed block.

use core::hint::cold_path;

use crate::bits::BackwardBits;
use crate::cpu::with_bmi2;
use crate::error::{Error, Result};
use crate::literals::LITERALS_PADDING;
use crate::output::Output;
use crate::sequence_tables::{Kind, Row, SequenceTables, Width};

/// How many bytes behind the end of a copy it may read and write.
pub(crate) const OVERSHOOT: usize = LITERALS_PADDING;
const CHUNK: usize = 16;

/// Where a block goes, and what it can refer to.
pub(crate) struct Destination<'o, 'a> {
    pub(crate) out: &'o mut Output<'a>,
    /// How much `out` had written when the part of the history that is in it started.
    pub(crate) history_start: usize,
    /// The end of the history before that.
    pub(crate) external: &'o [u8],
    pub(crate) block_len_max: usize,
}

impl Destination<'_, '_> {
    pub(crate) fn check_block_len(&self, len: usize) -> Result<()> {
        if len > self.out.room() {
            Err(Error::DstSizeTooSmall)
        } else if len > self.block_len_max {
            Err(Error::CorruptionDetected)
        } else {
            Ok(())
        }
    }

    pub(crate) fn append(self, bytes: &[u8]) -> Result<usize> {
        self.check_block_len(bytes.len())?;
        Ok(self.out.append(bytes))
    }

    /// `content` is that of an RLE block.
    pub(crate) fn append_repeated(self, content: &[u8], count: usize) -> Result<usize> {
        self.check_block_len(count)?;
        (self.out).append_repeated(*content.first().ok_or(Error::Internal)?, count);
        Ok(count)
    }

    /// The history in `out`, then `len` bytes for the block or more, as far as it may go, and the overshoot if possible.
    fn lend(&mut self, len: usize) -> &mut [u8] {
        let most = self.block_len_max.min(self.out.room()) + OVERSHOOT;
        (self.out).lend_with_slack(self.history_start, (len + OVERSHOOT).min(most), most)
    }
}

#[derive(Clone, Copy)]
struct Sequence {
    literal_len: u32,
    match_len: u32,
    /// As in [`Row::base`], until [`Execution::resolve_offset`].
    offset: i64,
}

#[derive(Clone, Copy)]
struct Reader<'a> {
    bits: BackwardBits<'a>,
    /// By [`Kind`].
    states: [usize; 3],
    left: usize,
}

impl<'a> Reader<'a> {
    fn new(tables: &SequenceTables, block: &'a [u8], start: usize, count: usize) -> Result<Self> {
        let mut bits = BackwardBits::new(block, start)?;
        let states = Kind::ALL.map(|kind| {
            bits.refill();
            bits.read(tables.log(kind)) as usize
        });
        bits.refill();
        Ok(Self {
            bits,
            states,
            left: count,
        })
    }

    #[inline(always)]
    fn rows<'t>(&self, tables: &'t SequenceTables) -> [&'t Row; 3] {
        let [literal_len, offset, match_len] = self.states;
        [
            tables.row(Kind::LiteralLength, literal_len),
            tables.row(Kind::Offset, offset),
            tables.row(Kind::MatchLength, match_len),
        ]
    }

    #[inline(never)]
    fn next_carefully(&mut self, tables: &SequenceTables) -> Sequence {
        let [literal_len, offset, match_len] = self.rows(tables);
        let bits = &mut self.bits;
        let mut value = |row: &Row| {
            let value = i64::from(row.base) + bits.read(row.extra_bits as u32) as i64;
            bits.refill();
            value
        };
        let sequence = Sequence {
            offset: value(offset),
            match_len: value(match_len) as u32,
            literal_len: value(literal_len) as u32,
        };
        self.left -= 1;
        if self.left != 0 {
            let [literal_len, match_len, offset] = [literal_len, match_len, offset]
                .map(|row| usize::from(row.next) + bits.read(row.state_bits as u32) as usize);
            self.states = [literal_len, offset, match_len];
            bits.refill();
        }
        sequence
    }

    /// Not for the last sequence.
    #[inline(always)]
    fn next_quickly(&mut self, tables: &SequenceTables) -> Option<Sequence> {
        let [literal_len, offset, match_len] = self.rows(tables);
        let total = |row: &Row| row.state_bits as u32 + row.extra_bits as u32;
        let total = total(offset) + total(match_len) + total(literal_len);
        if !self.bits.can_take(total) {
            return None;
        }
        let mut taken = self.bits.take(total);
        let mut last = |width: Width| {
            let value = taken & !(u64::MAX << width as u32);
            taken >>= width as u32;
            value
        };
        let next_offset = usize::from(offset.next) + last(offset.state_bits) as usize;
        let next_match_len = usize::from(match_len.next) + last(match_len.state_bits) as usize;
        let next_literal_len =
            usize::from(literal_len.next) + last(literal_len.state_bits) as usize;
        self.states = [next_literal_len, next_offset, next_match_len];
        self.left -= 1;
        Some(Sequence {
            literal_len: (literal_len.base as u32)
                .wrapping_add(last(literal_len.extra_bits) as u32),
            match_len: (match_len.base as u32).wrapping_add(last(match_len.extra_bits) as u32),
            offset: i64::from(offset.base) + last(offset.extra_bits) as i64,
        })
    }
}

#[derive(Clone, Copy)]
struct Execution<'a> {
    /// With [`LITERALS_PADDING`].
    literals: &'a [u8],
    /// In what [`Destination::lend`] returns.
    position: usize,
    repeat_offsets: [usize; 3],
}

/// `len` bytes, rounded up to whole chunks, of which there is at least one.
#[inline(always)]
fn copy_chunks(from: &[u8], to: &mut [u8], len: usize) {
    let (from, to) = (from.as_chunks::<CHUNK>().0, to.as_chunks_mut::<CHUNK>().0);
    let mut chunks = to.iter_mut().zip(from);
    if let Some((to, from)) = chunks.next() {
        *to = *from;
    }
    if len > CHUNK {
        cold_path();
        let mut copied = CHUNK;
        for (to, from) in chunks {
            *to = *from;
            copied += CHUNK;
            if copied >= len {
                break;
            }
        }
    }
}

#[inline(always)]
fn copy_chunk_within<const N: usize>(out: &mut [u8], from: usize, to: usize) -> Option<()> {
    let chunk: [u8; N] = *out.get(from..)?.first_chunk()?;
    *out.get_mut(to..)?.first_chunk_mut()? = chunk;
    Some(())
}

/// The first `period` bytes of `bytes`, over and over.
#[inline(always)]
fn repeated(bytes: [u8; CHUNK], period: usize) -> [u8; CHUNK] {
    let mut bits = 8 * period as u32;
    let mut pattern = u128::from_le_bytes(bytes) & !u128::MAX.wrapping_shl(bits);
    while bits < 128 {
        pattern |= pattern << bits;
        bits *= 2;
    }
    pattern.to_le_bytes()
}

impl Execution<'_> {
    #[inline(always)]
    fn resolve_offset(&mut self, sequence: &mut Sequence) {
        let [first, second, third] = self.repeat_offsets;
        if sequence.offset > 0 {
            self.repeat_offsets = [sequence.offset as usize, first, second];
            return;
        }
        // No history is as long as what stands for an offset of 0.
        self.repeat_offsets = match sequence.offset + i64::from(sequence.literal_len == 0) {
            -2 => [first, second, third],
            -1 => [second, first, third],
            0 => [third, first, second],
            _ if first == 1 => [usize::MAX, first, second],
            _ => [first.wrapping_sub(1), first, second],
        };
        sequence.offset = self.repeat_offsets[0] as i64;
    }

    /// Does not move on if it returns `None`.
    #[inline(always)]
    fn run_quickly(&mut self, sequence: Sequence, out: &mut [u8], external: &[u8]) -> Option<()> {
        let (literal_len, match_len) = (sequence.literal_len as usize, sequence.match_len as usize);
        let offset = sequence.offset as usize;
        let match_start = self.position.wrapping_add(literal_len);
        let end = match_start.wrapping_add(match_len);
        let literals = self.literals.get(..literal_len + LITERALS_PADDING)?;
        let room = out.get_mut(self.position..end.wrapping_add(OVERSHOOT))?;
        copy_chunks(literals, room, literal_len);

        if let Some(from) = match_start.checked_sub(offset) {
            if offset >= CHUNK {
                copy_chunk_within::<CHUNK>(out, from, match_start)?;
                let mut copied = CHUNK;
                while copied < match_len {
                    copy_chunk_within::<CHUNK>(out, from + copied, match_start + copied)?;
                    copied += CHUNK;
                }
            } else {
                let pattern = repeated(*out.get(from..)?.first_chunk()?, offset.max(1));
                let stride = CHUNK - CHUNK % offset.max(1);
                let mut to = match_start;
                while to < end {
                    *out.get_mut(to..)?.first_chunk_mut()? = pattern;
                    to += stride;
                }
            }
        } else {
            let from = external.len().checked_sub(offset - match_start)?;
            let from = external.get(from..)?.get(..match_len + OVERSHOOT)?;
            copy_chunks(from, room.get_mut(literal_len..)?, match_len);
        }
        self.literals = self.literals.get(literal_len..)?;
        self.position = end;
        Some(())
    }

    #[inline(never)]
    fn run_carefully(
        &mut self,
        sequence: Sequence,
        destination: &mut Destination<'_, '_>,
        block_start: usize,
    ) -> Result<()> {
        let (literal_len, match_len) = (sequence.literal_len as usize, sequence.match_len as usize);
        let offset = sequence.offset as usize;
        let match_start = self.position + literal_len;
        let end = match_start + match_len;
        destination.check_block_len(end - block_start)?;
        if literal_len + LITERALS_PADDING > self.literals.len() {
            return Err(Error::CorruptionDetected);
        }
        let (literals, rest) =
            (self.literals.split_at_checked(literal_len)).ok_or(Error::Internal)?;
        let external = destination.external;
        let out = destination.lend(end - block_start);
        copy(literals, out.get_mut(self.position..))?;

        let mut to = match_start;
        if let Some(before_out @ 1..) = offset.checked_sub(match_start) {
            let from = external.len().checked_sub(before_out);
            let from = from.and_then(|from| external.get(from..));
            let from = from.ok_or(Error::CorruptionDetected)?;
            let from = from.get(..match_len).unwrap_or(from);
            copy(from, out.get_mut(to..))?;
            to += from.len();
        }
        let from = to.saturating_sub(offset);
        while to < end {
            let (written, room) = out.split_at_mut_checked(to).ok_or(Error::Internal)?;
            let from = written.get(from..).filter(|from| !from.is_empty());
            let from = from.ok_or(Error::Internal)?;
            let from = from.get(..end - to).unwrap_or(from);
            copy(from, Some(room))?;
            to += from.len();
        }
        self.literals = rest;
        self.position = end;
        Ok(())
    }
}

#[inline(always)]
fn copy(from: &[u8], to: Option<&mut [u8]>) -> Result<()> {
    let to = to.and_then(|to| to.get_mut(..from.len()));
    to.ok_or(Error::Internal)?.copy_from_slice(from);
    Ok(())
}

/// Reads one sequence ahead of `pending`, so that a wrong guess of a branch in the copies does not hold up the reader.
/// Returns the one that it read if it resolved the offset of `pending` and got no further.
#[inline(always)]
fn run_while_quick(
    tables: &SequenceTables,
    reader: &mut Reader<'_>,
    execution: &mut Execution<'_>,
    pending: &mut Sequence,
    out: &mut [u8],
    external: &[u8],
) -> Option<Sequence> {
    let (mut local_reader, mut local_execution, mut current) = (*reader, *execution, *pending);
    let mut ahead = None;
    while local_reader.left > 1 {
        let Some(next) = local_reader.next_quickly(tables) else {
            cold_path();
            break;
        };
        local_execution.resolve_offset(&mut current);
        if (local_execution.run_quickly(current, out, external)).is_none() {
            cold_path();
            ahead = Some(next);
            break;
        }
        current = next;
    }
    (*reader, *execution, *pending) = (local_reader, local_execution, current);
    ahead
}

/// Runs the `count` sequences in `block[start..]`, of which there is at least one. Returns the literals that are left.
pub(crate) fn run_sequences<'l>(
    tables: &SequenceTables,
    block: &[u8],
    start: usize,
    count: usize,
    literals: &'l [u8],
    repeat_offsets: &mut [u32; 3],
    destination: &mut Destination<'_, '_>,
) -> Result<(usize, &'l [u8])> {
    const LOAN_MAX: usize = 32 << 10;

    let block_start = destination.out.written() - destination.history_start;
    let mut reader = Reader::new(tables, block, start, count)?;
    let mut execution = Execution {
        literals,
        position: block_start,
        repeat_offsets: repeat_offsets.map(|offset| offset as usize),
    };
    let mut pending = reader.next_carefully(tables);
    let external = destination.external;
    loop {
        let len = execution.position - block_start;
        let out = destination.lend(len + LOAN_MAX.min(len + 4 * block.len()));
        let ahead = with_bmi2(
            #[inline(always)]
            || {
                run_while_quick(
                    tables,
                    &mut reader,
                    &mut execution,
                    &mut pending,
                    out,
                    external,
                )
            },
        );
        if ahead.is_none() {
            execution.resolve_offset(&mut pending);
        }
        if let Err(error) = execution.run_carefully(pending, destination, block_start) {
            let is_garbage = reader.bits.left() < 0;
            return Err(if is_garbage {
                Error::CorruptionDetected
            } else {
                error
            });
        }
        pending = match ahead {
            Some(next) => next,
            None if reader.left == 0 => break,
            None => reader.next_carefully(tables),
        };
    }
    if reader.bits.left() != 0 {
        return Err(Error::CorruptionDetected);
    }
    *repeat_offsets = execution.repeat_offsets.map(|offset| offset as u32);
    Ok((execution.position - block_start, execution.literals))
}
