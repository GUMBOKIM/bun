//! Reading and running the sequences of a compressed block.

use core::cell::Cell;
use core::hint::{Locality, black_box, cold_path, prefetch_read, select_unpredictable};

use crate::array::wrapped;
use crate::bits::{BackwardBits, Width};
use crate::cpu::with_bmi2;
use crate::destination::{Destination, OVERSHOOT};
use crate::error::{Error, Result};
use crate::literals::LITERALS_PADDING;
use crate::sequence_tables::{Kind, LengthRow, OffsetRow, SequenceTables};

const CHUNK: usize = 16;

struct Sequence {
    literal_len: u32,
    match_len: u32,
    /// -2, -1 and 0 stand for repeated offsets.
    offset: u32,
}

#[derive(Clone, Copy)]
struct States {
    literal_len: usize,
    offset: usize,
    match_len: usize,
}

impl States {
    #[inline(always)]
    fn rows(self, tables: &SequenceTables) -> (&LengthRow, &OffsetRow, &LengthRow) {
        (
            wrapped(&tables.literal_lengths, self.literal_len),
            wrapped(&tables.offsets, self.offset),
            wrapped(&tables.match_lengths, self.match_len),
        )
    }

    /// What the sequence takes, and the way to the next states.
    #[inline(always)]
    fn total_bits(self, tables: &SequenceTables) -> isize {
        let (literal_len, offset, match_len) = self.rows(tables);
        isize::from(offset.total_bits)
            + isize::from(match_len.total_bits)
            + isize::from(literal_len.total_bits)
    }
}

struct Reader<'a> {
    tables: &'a SequenceTables,
    bits: BackwardBits<'a>,
    states: States,
    left: usize,
}

impl<'a> Reader<'a> {
    fn new(
        tables: &'a SequenceTables,
        block: &'a [u8],
        start: usize,
        count: usize,
    ) -> Result<Self> {
        let mut bits = BackwardBits::new(block, start)?;
        bits.refill();
        let [literal_len, offset, match_len] =
            Kind::ALL.map(|kind| bits.read(tables.log(kind)) as usize);
        bits.refill();
        Ok(Self {
            tables,
            bits,
            states: States {
                literal_len,
                offset,
                match_len,
            },
            left: count,
        })
    }

    fn next(&mut self) -> Sequence {
        let (literal_row, offset_row, match_row) = self.states.rows(self.tables);
        let mut bits = self.bits;
        let offset =
            (offset_row.base as u32).wrapping_add(bits.read(offset_row.extra_bits as u32) as u32);
        let match_len = match_row.full_base() + bits.read(match_row.extra_bits as u32) as u32;
        bits.refill();
        let literal_len = literal_row.full_base() + bits.read(literal_row.extra_bits as u32) as u32;
        self.left -= 1;
        if self.left != 0 {
            let mut next = |base: u16, width| usize::from(base) + bits.read(width as u32) as usize;
            self.states = States {
                literal_len: next(literal_row.next, literal_row.state_bits),
                match_len: next(match_row.next, match_row.state_bits),
                offset: next(offset_row.next.into(), offset_row.state_bits),
            };
            bits.refill();
        }
        self.bits = bits;
        Sequence {
            literal_len,
            match_len,
            offset,
        }
    }
}

/// A sequence on its way from the reader to the copies. What is in memory takes no registers from them.
#[derive(Clone, Copy, Default)]
#[repr(C)]
struct Slot {
    /// As in [`Sequence`]. Those of the three slots before the next to run are the offsets to repeat.
    offset: u32,
    literal_len: u16,
    match_len: u16,
    /// In the output and in the literals. The sequence before sets them.
    position: u32,
    literal_position: u32,
}

const REPEATS: usize = 3;
const AHEAD_MAX: usize = 8;
const BATCH: usize = 64;

struct Queue {
    slots: [Slot; REPEATS + AHEAD_MAX + BATCH + 1],
    /// Where the [`REPEATS`] slots before the next to run start.
    start: usize,
    /// Read and yet to run.
    pending: usize,
}

/// The offset that `code` stands for, unless that is 0. It makes it the first to repeat once it is in the next slot.
#[inline(always)]
fn repeat_offset(
    code: i32,
    literal_len: usize,
    [third, second, first]: [&mut Slot; REPEATS],
) -> Option<u32> {
    let code = code.wrapping_add((literal_len == 0).into());
    let (old_first, old_second, old_third) = (first.offset, second.offset, third.offset);
    if code > 0 && old_first == 1 {
        cold_path();
        return None;
    }
    first.offset = select_unpredictable(code < -1, old_second, old_first);
    second.offset = select_unpredictable(code < 0, old_third, old_second);
    Some(select_unpredictable(
        code < 0,
        select_unpredictable(code < -1, old_first, old_second),
        select_unpredictable(code > 0, old_first.wrapping_sub(1), old_third),
    ))
}

#[inline(always)]
fn chunk_at<const N: usize>(bytes: &[u8], position: usize) -> Option<[u8; N]> {
    bytes.get(position..)?.first_chunk().copied()
}

#[inline(always)]
fn set_chunk<const N: usize>(bytes: &mut [u8], position: usize, chunk: [u8; N]) -> Option<()> {
    *bytes.get_mut(position..)?.first_chunk_mut()? = chunk;
    Some(())
}

/// All but the first chunk of `len` bytes.
#[inline(always)]
fn copy_rest(
    from: &[u8],
    from_start: usize,
    to: &mut [u8],
    to_start: usize,
    len: usize,
) -> Option<()> {
    let from = from.get(from_start + CHUNK..)?.get(..len + CHUNK - 1)?;
    let to = to.get_mut(to_start + CHUNK..)?.get_mut(..len + CHUNK - 1)?;
    for (to, from) in (to.as_chunks_mut::<{ 2 * CHUNK }>().0.iter_mut()).zip(from.as_chunks().0) {
        *to = *from;
        // Most are short: neither unrolling nor a call of `memcpy()` pays.
        black_box(());
    }
    Some(())
}

/// For an `offset` of at least a chunk.
#[inline(always)]
fn copy_distant_match(
    out: &mut [u8],
    from: usize,
    offset: usize,
    start: usize,
    len: usize,
) -> Option<()> {
    #[inline(always)]
    fn copy(from: &[Cell<u8>; CHUNK], to: &[Cell<u8>; CHUNK]) {
        let mut bytes = [0; CHUNK];
        bytes
            .iter_mut()
            .zip(from)
            .for_each(|(byte, from)| *byte = from.get());
        bytes.iter().zip(to).for_each(|(byte, to)| to.set(*byte));
    }

    set_chunk(out, start, chunk_at::<CHUNK>(out, from)?)?;
    set_chunk(out, start + CHUNK, chunk_at::<CHUNK>(out, from + CHUNK)?)?;
    if len > 2 * CHUNK {
        let both = out.get_mut(from + 2 * CHUNK..start + len + 2 * CHUNK - 1)?;
        let both = Cell::from_mut(both).as_slice_of_cells();
        let to = both.get(offset..)?;
        let (from, to) = (both.get(..to.len())?.as_chunks().0, to.as_chunks().0);
        for ([to_0, to_1], [from_0, from_1]) in to.as_chunks().0.iter().zip(from.as_chunks().0) {
            copy(from_0, to_0);
            copy(from_1, to_1);
            black_box(());
        }
    }
    Some(())
}

/// How to repeat the bytes of a period of less than a chunk.
struct Period {
    /// For those among the first 8.
    mask: u64,
    /// Has a bit where each repetition starts.
    multiplier: u64,
    /// Where in the first 8 bytes the second 8 start, in bits.
    phase: u8,
    /// As many whole periods as a chunk holds.
    stride: u8,
}

#[expect(clippy::indexing_slicing, reason = "evaluated by the compiler")]
static PERIODS: [Period; CHUNK] = {
    let mut periods = [const {
        Period {
            mask: u64::MAX,
            multiplier: 1,
            phase: 0,
            stride: 1,
        }
    }; _];
    let mut len = 1;
    while len < CHUNK {
        periods[len].stride = (CHUNK - CHUNK % len) as u8;
        if len < 8 {
            periods[len].mask = (1 << (8 * len)) - 1;
            let mut start = len;
            while start < 8 {
                periods[len].multiplier |= 1 << (8 * start);
                start += len;
            }
            periods[len].phase = 8 * (8 % len) as u8;
        }
        len += 1;
    }
    periods
};

/// For an `offset` of less than a chunk.
#[inline(always)]
fn copy_close_match(
    out: &mut [u8],
    from: usize,
    offset: usize,
    start: usize,
    len: usize,
) -> Option<()> {
    let bytes = u128::from_le_bytes(chunk_at(out, from)?);
    let period = wrapped(&PERIODS, offset);
    let repeated = |bytes: u64| (bytes & period.mask).wrapping_mul(period.multiplier);
    let low = repeated(bytes as u64);
    set_chunk(out, start, low.to_le_bytes())?;
    if len > 8 {
        let high = if offset <= 8 {
            repeated(low.wrapping_shr(period.phase.into()))
        } else {
            let bits = 8 * offset as u32 - 64;
            (bytes >> 64) as u64 & !u64::MAX.wrapping_shl(bits) | low.wrapping_shl(bits)
        };
        let chunk = (u128::from(high) << 64 | u128::from(low)).to_le_bytes();
        let mut position = start as u32;
        while (position as usize) < start + len {
            set_chunk(out, position as usize, chunk)?;
            position = position.wrapping_add(period.stride.into());
        }
    }
    Some(())
}

/// Reads a sequence, and runs the one `AHEAD` before it, until either takes care. That way the reader goes on while the
/// processor recovers from a wrong guess about the copies, and with more of a lead far matches arrive in time.
#[inline(never)]
fn run_while_quick<const AHEAD: usize>(
    reader: &mut Reader<'_>,
    queue: &mut Queue,
    literals: &[u8],
    external: &[u8],
    out: &mut [u8],
) {
    with_bmi2(
        #[inline(always)]
        move || run_while_quick_inline::<AHEAD>(reader, queue, literals, external, out),
    );
}

#[inline(always)]
fn run_while_quick_inline<const AHEAD: usize>(
    reader: &mut Reader<'_>,
    queue: &mut Queue,
    literals: &[u8],
    external: &[u8],
    out: &mut [u8],
) -> Option<()> {
    let literals_limit = literals.len().checked_sub(LITERALS_PADDING)?;
    let out_limit = out.len().checked_sub(OVERSHOOT)?;
    let (tables, src) = (reader.tables, reader.bits.src);
    let mut states = reader.states;
    let mut position = reader.bits.position as u32;
    let mut container = reader.bits.container;
    let mut spare_bits = reader.bits.available - states.total_bits(tables);
    let mut read_position = 0u32;
    if AHEAD > 1 {
        let pending = queue
            .slots
            .get(queue.start + REPEATS..)?
            .first_chunk::<AHEAD>()?;
        let lens = pending
            .iter()
            .map(|it| u32::from(it.literal_len) + u32::from(it.match_len));
        read_position = pending.first()?.position.wrapping_add(lens.sum());
    }

    while spare_bits >= 0 && queue.pending == AHEAD {
        if queue.start >= BATCH {
            queue.rewind()?;
        }
        let mut count = reader.left.saturating_sub(1).min(BATCH - queue.start);
        // With 8 bytes before it, a stream is over before a refill leaves `src`.
        if reader.bits.start < 8 {
            count = count.min(position as usize / 7);
        }
        if count == 0 {
            break;
        }
        let mut slots = (queue.slots.get_mut(queue.start..)?).get_mut(..count + REPEATS + AHEAD)?;
        let first_address = slots.as_ptr().addr();

        while let [third, second, first, current, later @ ..] = slots
            && let Some(later) = later.first_chunk_mut::<AHEAD>()
        {
            let (literal_row, offset_row, match_row) = states.rows(tables);
            let mut bits = container.wrapping_shr(spare_bits as u32);
            let mut last = |width: Width| {
                let value = bits & !(u64::MAX << width as u32);
                bits >>= width as u32;
                value
            };
            states.offset = (offset_row
                .next
                .wrapping_add(last(offset_row.state_bits) as u8))
            .into();
            states.match_len =
                (usize::from(match_row.next) + last(match_row.state_bits) as usize) % 512;
            states.literal_len =
                (usize::from(literal_row.next) + last(literal_row.state_bits) as usize) % 512;
            let used_bytes = (spare_bits as u32 ^ 63) / 8;
            spare_bits = (spare_bits | 56) - states.total_bits(tables);
            let Some(read) = later.last_mut() else { break };
            read.literal_len = (literal_row.base).wrapping_add(last(literal_row.extra_bits) as u16);
            read.match_len = (match_row.base).wrapping_add(last(match_row.extra_bits) as u16);
            read.offset = (offset_row.base as u32).wrapping_add(last(offset_row.extra_bits) as u32);
            if AHEAD > 1 {
                read_position = read_position.wrapping_add(read.literal_len.into());
                let from = out
                    .as_ptr()
                    .wrapping_add(read_position.wrapping_sub(read.offset) as usize);
                prefetch_read(from, Locality::L1);
                prefetch_read(from.wrapping_add(64), Locality::L1);
                read_position = read_position.wrapping_add(read.match_len.into());
            }
            position = position.wrapping_sub(used_bytes);
            let refill = src.get(position as usize..position as usize + 8);
            let Some(refill) = refill.and_then(<[u8]>::as_array) else {
                cold_path();
                break;
            };
            container = u64::from_le_bytes(*refill);
            if spare_bits < 0 {
                cold_path();
                break;
            }

            let Some(following) = later.first_mut() else {
                break;
            };
            let (literal_len, literal_start) = (
                current.literal_len.into(),
                current.literal_position as usize,
            );
            following.literal_position = (literal_start + literal_len) as u32;
            if literal_start + literal_len > literals_limit {
                cold_path();
                break;
            }
            let (start, match_len) = (current.position as usize, usize::from(current.match_len));
            let match_start = start + literal_len;
            following.position = (match_start + match_len) as u32;
            if match_start + match_len > out_limit {
                cold_path();
                break;
            }
            let Some(chunk) = chunk_at::<CHUNK>(literals, literal_start) else {
                break;
            };
            let Some(()) = set_chunk(out, start, chunk) else {
                break;
            };
            if literal_len > CHUNK {
                cold_path();
                let Some(()) = copy_rest(literals, literal_start, out, start, literal_len) else {
                    break;
                };
            }

            if current.offset as i32 <= 0 {
                let repeats = [third, second, first];
                let Some(offset) = repeat_offset(current.offset as i32, literal_len, repeats)
                else {
                    break;
                };
                current.offset = offset;
            }
            let offset = current.offset as usize;
            let copied = match match_start.checked_sub(offset) {
                Some(from) if offset >= CHUNK => {
                    copy_distant_match(out, from, offset, match_start, match_len)
                }
                Some(from) => {
                    cold_path();
                    copy_close_match(out, from, offset, match_start, match_len)
                }
                None => {
                    cold_path();
                    copy_external_match(external, offset - match_start, out, match_start, match_len)
                }
            };
            let Some(()) = copied else { break };
            slots = slots.get_mut(1..).unwrap_or_default();
        }

        let done = (slots.as_ptr().addr() - first_address) / size_of::<Slot>();
        let is_ahead = usize::from(done < count);
        queue.start += done;
        queue.pending += is_ahead;
        reader.left -= done + is_ahead;
    }

    reader.states = states;
    reader.bits.position = position as usize;
    reader.bits.container = container;
    reader.bits.available = spare_bits + states.total_bits(tables);
    Some(())
}

/// If all of it, and the overshoot, is in the last `distance` bytes of `external`.
#[inline(always)]
fn copy_external_match(
    external: &[u8],
    distance: usize,
    out: &mut [u8],
    start: usize,
    len: usize,
) -> Option<()> {
    let from = external.get(external.len().checked_sub(distance)?..)?;
    let from = from.get(..len + OVERSHOOT)?;
    set_chunk(out, start, chunk_at::<CHUNK>(from, 0)?)?;
    if len > CHUNK {
        copy_rest(from, 0, out, start, len)?;
    }
    Some(())
}

#[inline(always)]
fn copy(from: &[u8], to: Option<&mut [u8]>) -> Result<()> {
    let to = to.and_then(|to| to.get_mut(..from.len()));
    to.ok_or(Error::Internal)?.copy_from_slice(from);
    Ok(())
}

impl Queue {
    #[inline(always)]
    fn rewind(&mut self) -> Option<()> {
        let kept: [Slot; REPEATS + AHEAD_MAX] = *self.slots.get(self.start..)?.first_chunk()?;
        *self.slots.first_chunk_mut()? = kept;
        self.start = 0;
        Some(())
    }

    /// Returns `sequence` if it does not fit in a slot.
    fn push(&mut self, sequence: Sequence) -> Result<Option<Sequence>> {
        let lens = (
            sequence.literal_len.try_into(),
            sequence.match_len.try_into(),
        );
        let ((Ok(literal_len), Ok(match_len)), -2..) = (lens, sequence.offset as i32) else {
            return Ok(Some(sequence));
        };
        let slot = self.slots.get_mut(self.start + REPEATS + self.pending);
        let slot = slot.ok_or(Error::Internal)?;
        (slot.offset, slot.literal_len, slot.match_len) = (sequence.offset, literal_len, match_len);
        self.pending += 1;
        Ok(None)
    }

    fn run_pending(
        &mut self,
        literals: &[u8],
        destination: &mut Destination<'_, '_>,
    ) -> Result<()> {
        let slot = self
            .slots
            .get(self.start + REPEATS)
            .ok_or(Error::Internal)?;
        let sequence = Sequence {
            literal_len: slot.literal_len.into(),
            match_len: slot.match_len.into(),
            offset: slot.offset,
        };
        self.pending -= 1;
        self.run(&sequence, literals, destination)
    }

    /// In the place of the next slot.
    fn run(
        &mut self,
        sequence: &Sequence,
        literals: &[u8],
        destination: &mut Destination<'_, '_>,
    ) -> Result<()> {
        let slots = self
            .slots
            .get_mut(self.start..)
            .and_then(<[Slot]>::first_chunk_mut);
        let [third, second, first, current, following] = slots.ok_or(Error::Internal)?;
        let (literal_len, match_len) = (sequence.literal_len as usize, sequence.match_len as usize);
        let (start, literal_start) = (current.position as usize, current.literal_position as usize);
        let match_start = start + literal_len;
        let end = match_start + match_len;
        let block_len = end - destination.position();
        destination.check_block_len(block_len)?;
        if literal_start + literal_len + LITERALS_PADDING > literals.len() {
            return Err(Error::CorruptionDetected);
        }
        current.offset = match sequence.offset as i32 {
            code @ -2..=0 => repeat_offset(code, literal_len, [third, second, first])
                .ok_or(Error::CorruptionDetected)?,
            _ => sequence.offset,
        };
        let offset = current.offset as usize;
        let (external, out) = destination.lend(block_len);
        let literals = literals.get(literal_start..literal_start + literal_len);
        copy(literals.ok_or(Error::Internal)?, out.get_mut(start..))?;

        let mut to = match_start;
        if let Some(distance @ 1..) = offset.checked_sub(match_start) {
            let from = external.len().checked_sub(distance);
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
        following.position = end as u32;
        following.literal_position = (literal_start + literal_len) as u32;
        self.start += 1;
        if self.start >= BATCH {
            self.rewind().ok_or(Error::Internal)?;
        }
        Ok(())
    }

    fn run_all<const AHEAD: usize>(
        &mut self,
        reader: &mut Reader<'_>,
        literals: &[u8],
        destination: &mut Destination<'_, '_>,
    ) -> Result<()> {
        const LOAN_MAX: usize = 32 << 10;

        loop {
            while self.pending < AHEAD && reader.left != 0 {
                if let Some(large) = self.push(reader.next())? {
                    while self.pending != 0 {
                        self.run_pending(literals, destination)?;
                    }
                    self.run(&large, literals, destination)?;
                }
            }
            if self.pending == 0 {
                return Ok(());
            }
            if reader.left > 1 {
                let next = self
                    .slots
                    .get(self.start + REPEATS)
                    .ok_or(Error::Internal)?;
                let len = next.position as usize - destination.position();
                let (external, out) =
                    destination.lend(len + LOAN_MAX.min(len + 4 * reader.bits.src.len()));
                run_while_quick::<AHEAD>(reader, self, literals, external, out);
                if reader.bits.position > reader.bits.src.len() {
                    return Err(Error::CorruptionDetected);
                }
            }
            self.run_pending(literals, destination)?;
        }
    }
}

/// Runs the `count` sequences in `block[start..]`. Returns how many bytes they wrote, and how many literals they took.
pub(crate) fn run_sequences(
    tables: &SequenceTables,
    block: &[u8],
    start: usize,
    count: usize,
    literals: &[u8],
    repeat_offsets: &mut [u32; REPEATS],
    destination: &mut Destination<'_, '_>,
) -> Result<(usize, usize)> {
    let mut reader = Reader::new(tables, block, start, count)?;
    let mut queue = Queue {
        slots: [Slot::default(); _],
        start: 0,
        pending: 0,
    };
    let [third, second, first, next, ..] = &mut queue.slots;
    [first.offset, second.offset, third.offset] = *repeat_offsets;
    next.position = destination.position() as u32;

    let result = if destination.history_len() > 1 << 24
        && count > AHEAD_MAX
        && tables.has_many_far_offsets()
    {
        queue.run_all::<AHEAD_MAX>(&mut reader, literals, destination)
    } else {
        queue.run_all::<1>(&mut reader, literals, destination)
    };
    // What fails is made of garbage then.
    if reader.bits.left() < 0 || (result.is_ok() && reader.bits.left() != 0) {
        return Err(Error::CorruptionDetected);
    }
    result?;
    let slots = queue
        .slots
        .get(queue.start..)
        .and_then(<[Slot]>::first_chunk);
    let [third, second, first, next] = slots.ok_or(Error::Internal)?;
    *repeat_offsets = [first.offset, second.offset, third.offset];
    Ok((
        next.position as usize - destination.position(),
        next.literal_position as usize,
    ))
}
