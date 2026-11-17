//! The decoding tables for sequences.

use crate::array::{wrapped, wrapped_mut};
use crate::bits::Width;
use crate::cpu::with_bmi2;
use crate::error::{Error, Result};
use crate::format::{
    DEFAULT_LITERAL_LENGTHS, DEFAULT_MATCH_LENGTHS, DEFAULT_OFFSETS, Distribution,
    LITERAL_LENGTH_BITS, LITERAL_LENGTH_TABLE_LOG_MAX, MATCH_LENGTH_BITS,
    MATCH_LENGTH_TABLE_LOG_MAX, MAX_LITERAL_LENGTH_CODE, MAX_MATCH_LENGTH_CODE, MAX_OFFSET_CODE,
    MIN_MATCH, OFFSET_TABLE_LOG_MAX,
};
use crate::fse::{States, read_counts};

/// More bits than a reader ever has, so that it takes care.
const NEEDS_CARE: u8 = 128;

#[derive(Clone, Copy)]
#[repr(C, align(8))]
pub(crate) struct LengthRow {
    /// Of both kinds, plus [`NEEDS_CARE`] if the length may not fit in 16 bits.
    pub(crate) total_bits: u8,
    pub(crate) state_bits: Width,
    pub(crate) extra_bits: Width,
    /// To which `state_bits` are added.
    pub(crate) next: u16,
    /// To which `extra_bits` are added: the low bits of [`Self::full_base`].
    pub(crate) base: u16,
}

#[derive(Clone, Copy)]
#[repr(C, align(8))]
pub(crate) struct OffsetRow {
    /// Of both kinds, plus [`NEEDS_CARE`] if the offset may not fit in 31 bits.
    pub(crate) total_bits: u8,
    pub(crate) state_bits: Width,
    pub(crate) extra_bits: Width,
    pub(crate) next: u8,
    /// The sum is -2, -1 or 0 for a repeated offset.
    pub(crate) base: i32,
}

impl LengthRow {
    const EMPTY: Self = Self::of_code(0, 0);

    const fn of_code(base: u32, extra_bits: u8) -> Self {
        Self {
            total_bits: extra_bits + if extra_bits >= 15 { NEEDS_CARE } else { 0 },
            state_bits: Width::W0,
            extra_bits: Width::new(extra_bits as u32),
            next: 0,
            base: base as u16,
        }
    }

    pub(crate) fn full_base(self) -> u32 {
        u32::from(self.base) | (self.extra_bits as u32 & 16) << 12
    }
}

impl OffsetRow {
    const EMPTY: Self = Self {
        base: 0,
        ..Self::of_code(0)
    };

    const fn of_code(code: u8) -> Self {
        Self {
            total_bits: code + if code == 31 { NEEDS_CARE } else { 0 },
            state_bits: Width::W0,
            extra_bits: Width::new(code as u32),
            next: 0,
            base: ((1i64 << code) - 3) as i32,
        }
    }
}

const CODES_MAX: usize = 64;

#[expect(clippy::indexing_slicing, reason = "evaluated by the compiler")]
const fn length_codes(mut base: u32, extra_bits: &[u8]) -> [LengthRow; CODES_MAX] {
    let (mut codes, mut code) = ([LengthRow::EMPTY; CODES_MAX], 0);
    while code < extra_bits.len() {
        codes[code] = LengthRow::of_code(base, extra_bits[code]);
        base += 1 << extra_bits[code];
        code += 1;
    }
    codes
}

const LITERAL_LENGTH_CODES: [LengthRow; CODES_MAX] = length_codes(0, &LITERAL_LENGTH_BITS);
const MATCH_LENGTH_CODES: [LengthRow; CODES_MAX] = length_codes(MIN_MATCH, &MATCH_LENGTH_BITS);
#[expect(clippy::indexing_slicing, reason = "evaluated by the compiler")]
const OFFSET_CODES: [OffsetRow; CODES_MAX] = {
    let (mut codes, mut code) = ([OffsetRow::EMPTY; _], 0);
    while code <= MAX_OFFSET_CODE {
        codes[code] = OffsetRow::of_code(code as u8);
        code += 1;
    }
    codes
};

/// In the order of their tables in a block.
#[derive(Clone, Copy)]
pub(crate) enum Kind {
    LiteralLength,
    Offset,
    MatchLength,
}

impl Kind {
    pub(crate) const ALL: [Self; 3] = [Self::LiteralLength, Self::Offset, Self::MatchLength];

    fn max_code(self) -> usize {
        match self {
            Self::LiteralLength => MAX_LITERAL_LENGTH_CODE,
            Self::Offset => MAX_OFFSET_CODE,
            Self::MatchLength => MAX_MATCH_LENGTH_CODE,
        }
    }

    fn table_log_max(self) -> u32 {
        match self {
            Self::LiteralLength => LITERAL_LENGTH_TABLE_LOG_MAX,
            Self::Offset => OFFSET_TABLE_LOG_MAX,
            Self::MatchLength => MATCH_LENGTH_TABLE_LOG_MAX,
        }
    }
}

/// One of each [`Kind`].
#[repr(C)]
pub(crate) struct SequenceTables {
    pub(crate) literal_lengths: [LengthRow; 1 << LITERAL_LENGTH_TABLE_LOG_MAX],
    pub(crate) match_lengths: [LengthRow; 1 << MATCH_LENGTH_TABLE_LOG_MAX],
    pub(crate) offsets: [OffsetRow; 1 << OFFSET_TABLE_LOG_MAX],
    logs: [u32; 4],
}

pub(crate) static PREDEFINED_TABLES: SequenceTables = {
    let mut tables = SequenceTables::EMPTY;
    tables.fill(Kind::LiteralLength, &DEFAULT_LITERAL_LENGTHS);
    tables.fill(Kind::Offset, &DEFAULT_OFFSETS);
    tables.fill(Kind::MatchLength, &DEFAULT_MATCH_LENGTHS);
    tables
};

#[inline(always)]
const fn first_rows<T>(rows: &mut [T], log: u32) -> &mut [T] {
    match rows.split_at_mut_checked(1 << log) {
        Some((rows, _)) => rows,
        None => &mut [],
    }
}

impl SequenceTables {
    pub(crate) const EMPTY: Self = Self {
        literal_lengths: [LengthRow::EMPTY; _],
        match_lengths: [LengthRow::EMPTY; _],
        offsets: [OffsetRow::EMPTY; _],
        logs: [0; _],
    };

    #[inline(always)]
    pub(crate) fn log(&self, kind: Kind) -> u32 {
        *wrapped(&self.logs, kind as usize)
    }

    #[inline(always)]
    const fn fill(&mut self, kind: Kind, distribution: &Distribution<'_>) {
        let log = distribution.table_log;
        *wrapped_mut(&mut self.logs, kind as usize) = log;
        let mut states = States::<512, 520, CODES_MAX>::EMPTY;
        states.spread(distribution.counts, log);
        let (rows, codes) = match kind {
            Kind::LiteralLength => (&mut self.literal_lengths, &LITERAL_LENGTH_CODES),
            Kind::MatchLength => (&mut self.match_lengths, &MATCH_LENGTH_CODES),
            Kind::Offset => {
                let mut rows = first_rows(&mut self.offsets, log);
                while let [row, rest @ ..] = rows {
                    let state = states.next_state();
                    *row = *wrapped(&OFFSET_CODES, state.symbol as usize);
                    row.total_bits += state.bits as u8;
                    row.state_bits = Width::new(state.bits);
                    row.next = state.baseline as u8;
                    rows = rest;
                }
                return;
            }
        };
        let mut rows = first_rows(rows, log);
        while let [row, rest @ ..] = rows {
            let state = states.next_state();
            *row = *wrapped(codes, state.symbol as usize);
            row.total_bits += state.bits as u8;
            row.state_bits = Width::new(state.bits);
            row.next = state.baseline;
            rows = rest;
        }
    }

    /// From the description at the start of `src`, whose length it returns.
    #[inline(never)]
    pub(crate) fn read(&mut self, kind: Kind, src: &[u8]) -> Result<usize> {
        with_bmi2(
            #[inline(always)]
            || self.read_inline(kind, src),
        )
    }

    #[inline(always)]
    fn read_inline(&mut self, kind: Kind, src: &[u8]) -> Result<usize> {
        let mut counts = [0; CODES_MAX];
        let allowed = counts.get_mut(..=kind.max_code());
        let description = read_counts(src, allowed.ok_or(Error::Internal)?)?;
        if description.table_log > kind.table_log_max() {
            return Err(Error::CorruptionDetected);
        }
        let distribution = Distribution {
            counts: (counts.get(..=description.max_symbol)).ok_or(Error::Internal)?,
            table_log: description.table_log,
        };
        self.fill(kind, &distribution);
        Ok(description.len)
    }

    /// For sequences that all have `code`.
    pub(crate) fn set_rle(&mut self, kind: Kind, code: u8) -> Result<()> {
        let code = usize::from(code);
        if code > kind.max_code() {
            return Err(Error::CorruptionDetected);
        }
        *wrapped_mut(&mut self.logs, kind as usize) = 0;
        match kind {
            Kind::LiteralLength => self.literal_lengths[0] = *wrapped(&LITERAL_LENGTH_CODES, code),
            Kind::Offset => self.offsets[0] = *wrapped(&OFFSET_CODES, code),
            Kind::MatchLength => self.match_lengths[0] = *wrapped(&MATCH_LENGTH_CODES, code),
        }
        Ok(())
    }

    pub(crate) fn copy_from(&mut self, other: &Self, kind: Kind) {
        fn copy<T: Copy>(from: &[T], to: &mut [T], len: usize) {
            to.iter_mut()
                .zip(from)
                .take(len)
                .for_each(|(to, from)| *to = *from);
        }

        let log = other.log(kind);
        *wrapped_mut(&mut self.logs, kind as usize) = log;
        match kind {
            Kind::LiteralLength => {
                copy(&other.literal_lengths, &mut self.literal_lengths, 1 << log)
            }
            Kind::Offset => copy(&other.offsets, &mut self.offsets, 1 << log),
            Kind::MatchLength => copy(&other.match_lengths, &mut self.match_lengths, 1 << log),
        }
    }

    /// Whether so many matches are out of the cache that it pays to ask for them ahead of time.
    pub(crate) fn has_many_far_offsets(&self) -> bool {
        let log = self.log(Kind::Offset);
        let rows = self.offsets.iter().take(1 << log);
        let far = rows.filter(|row| row.extra_bits as u32 > 22).count();
        far << (OFFSET_TABLE_LOG_MAX - log) >= 7
    }
}
