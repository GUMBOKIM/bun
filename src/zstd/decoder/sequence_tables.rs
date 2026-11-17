//! The decoding tables for sequences.

use crate::array::wrapped;
use crate::error::{Error, Result};
use crate::format::{
    DEFAULT_LITERAL_LENGTHS, DEFAULT_MATCH_LENGTHS, DEFAULT_OFFSETS, Distribution,
    LITERAL_LENGTH_BITS, LITERAL_LENGTH_TABLE_LOG_MAX, MATCH_LENGTH_BITS,
    MATCH_LENGTH_TABLE_LOG_MAX, MAX_LITERAL_LENGTH_CODE, MAX_MATCH_LENGTH_CODE, MAX_OFFSET_CODE,
    MIN_MATCH, OFFSET_TABLE_LOG_MAX,
};
use crate::fse::{States, read_normalized_counts};

/// A number of bits. The compiler knows that it can shift by it.
#[derive(Clone, Copy)]
#[repr(u8)]
#[rustfmt::skip]
pub(crate) enum Width {
    W0, W1, W2, W3, W4, W5, W6, W7, W8, W9, W10, W11, W12, W13, W14, W15,
    W16, W17, W18, W19, W20, W21, W22, W23, W24, W25, W26, W27, W28, W29, W30, W31,
}

impl Width {
    #[inline(always)]
    #[rustfmt::skip]
    const fn new(bits: u32) -> Self {
        use Width::*;
        *wrapped(&[
            W0, W1, W2, W3, W4, W5, W6, W7, W8, W9, W10, W11, W12, W13, W14, W15,
            W16, W17, W18, W19, W20, W21, W22, W23, W24, W25, W26, W27, W28, W29, W30, W31,
        ], bits as usize)
    }
}

#[derive(Clone, Copy)]
#[repr(C, align(8))]
pub(crate) struct Row {
    /// To which `state_bits` are added.
    pub(crate) next: u16,
    pub(crate) state_bits: Width,
    pub(crate) extra_bits: Width,
    /// To which `extra_bits` are added. An offset of less than 1 is a repeated one.
    pub(crate) base: i32,
}

impl Row {
    const EMPTY: Self = Self::of_code(0, 0);

    const fn of_code(base: i32, extra_bits: u8) -> Self {
        Self {
            next: 0,
            state_bits: Width::W0,
            extra_bits: Width::new(extra_bits as u32),
            base,
        }
    }
}

const CODES_MAX: usize = 64;
const ROWS_MAX: usize = 512;

/// In the order of their tables in a block.
#[derive(Clone, Copy)]
pub(crate) enum Kind {
    LiteralLength,
    Offset,
    MatchLength,
}

struct Alphabet {
    max_code: usize,
    table_log_max: u32,
    codes: [Row; CODES_MAX],
}

#[expect(clippy::indexing_slicing, reason = "evaluated by the compiler")]
const fn length_codes(mut base: i32, extra_bits: &[u8]) -> [Row; CODES_MAX] {
    let (mut codes, mut code) = ([Row::EMPTY; CODES_MAX], 0);
    while code < extra_bits.len() {
        codes[code] = Row::of_code(base, extra_bits[code]);
        base += 1 << extra_bits[code];
        code += 1;
    }
    codes
}

#[expect(clippy::indexing_slicing, reason = "evaluated by the compiler")]
const fn offset_codes() -> [Row; CODES_MAX] {
    let (mut codes, mut code) = ([Row::EMPTY; CODES_MAX], 0);
    while code <= MAX_OFFSET_CODE {
        codes[code] = Row::of_code(((1i64 << code) - 3) as i32, code as u8);
        code += 1;
    }
    codes
}

impl Kind {
    pub(crate) const ALL: [Self; 3] = [Self::LiteralLength, Self::Offset, Self::MatchLength];

    const fn alphabet(self) -> &'static Alphabet {
        match self {
            Self::LiteralLength => {
                &const {
                    Alphabet {
                        max_code: MAX_LITERAL_LENGTH_CODE,
                        table_log_max: LITERAL_LENGTH_TABLE_LOG_MAX,
                        codes: length_codes(0, &LITERAL_LENGTH_BITS),
                    }
                }
            }
            Self::Offset => {
                &const {
                    Alphabet {
                        max_code: MAX_OFFSET_CODE,
                        table_log_max: OFFSET_TABLE_LOG_MAX,
                        codes: offset_codes(),
                    }
                }
            }
            Self::MatchLength => {
                &const {
                    Alphabet {
                        max_code: MAX_MATCH_LENGTH_CODE,
                        table_log_max: MATCH_LENGTH_TABLE_LOG_MAX,
                        codes: length_codes(MIN_MATCH as i32, &MATCH_LENGTH_BITS),
                    }
                }
            }
        }
    }
}

/// One of each [`Kind`].
pub(crate) struct SequenceTables {
    literal_lengths: [Row; ROWS_MAX],
    match_lengths: [Row; ROWS_MAX],
    offsets: [Row; ROWS_MAX / 2],
    logs: [u32; 4],
}

pub(crate) static PREDEFINED_TABLES: SequenceTables = {
    let mut tables = SequenceTables::EMPTY;
    tables.fill(Kind::LiteralLength, &DEFAULT_LITERAL_LENGTHS);
    tables.fill(Kind::Offset, &DEFAULT_OFFSETS);
    tables.fill(Kind::MatchLength, &DEFAULT_MATCH_LENGTHS);
    tables
};

impl SequenceTables {
    pub(crate) const EMPTY: Self = Self {
        literal_lengths: [Row::EMPTY; _],
        match_lengths: [Row::EMPTY; _],
        offsets: [Row::EMPTY; _],
        logs: [0; _],
    };

    #[inline(always)]
    pub(crate) fn log(&self, kind: Kind) -> u32 {
        *wrapped(&self.logs, kind as usize)
    }

    #[inline(always)]
    pub(crate) fn row(&self, kind: Kind, state: usize) -> &Row {
        match kind {
            Kind::LiteralLength => wrapped(&self.literal_lengths, state),
            Kind::Offset => wrapped(&self.offsets, state),
            Kind::MatchLength => wrapped(&self.match_lengths, state),
        }
    }

    pub(crate) fn rows(&self, kind: Kind) -> &[Row] {
        let rows: &[Row] = match kind {
            Kind::LiteralLength => &self.literal_lengths,
            Kind::Offset => &self.offsets,
            Kind::MatchLength => &self.match_lengths,
        };
        rows.get(..1 << self.log(kind)).unwrap_or_default()
    }

    /// Room for a table of `1 << log` rows.
    #[inline(always)]
    const fn rows_mut(&mut self, kind: Kind, log: u32) -> &mut [Row] {
        *crate::array::wrapped_mut(&mut self.logs, kind as usize) = log;
        let rows: &mut [Row] = match kind {
            Kind::LiteralLength => &mut self.literal_lengths,
            Kind::Offset => &mut self.offsets,
            Kind::MatchLength => &mut self.match_lengths,
        };
        match rows.split_at_mut_checked(1 << log) {
            Some((rows, _)) => rows,
            None => &mut [],
        }
    }

    #[inline(always)]
    const fn fill(&mut self, kind: Kind, distribution: &Distribution<'_>) {
        let codes = &kind.alphabet().codes;
        let mut states = States::new(distribution.counts, distribution.table_log);
        let mut rows = self.rows_mut(kind, distribution.table_log);
        while let [row, rest @ ..] = rows {
            let state = states.next_state();
            *row = Row {
                next: state.baseline,
                state_bits: Width::new(state.bits),
                ..*wrapped(codes, state.symbol as usize)
            };
            rows = rest;
        }
    }

    /// From the description at the start of `src`, whose length it returns.
    #[inline(always)]
    pub(crate) fn read(&mut self, kind: Kind, src: &[u8]) -> Result<usize> {
        let alphabet = kind.alphabet();
        let mut counts = [0; CODES_MAX];
        let allowed = counts.get_mut(..=alphabet.max_code);
        let description = read_normalized_counts(src, allowed.ok_or(Error::Internal)?)?;
        if description.table_log > alphabet.table_log_max {
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
        let alphabet = kind.alphabet();
        if usize::from(code) > alphabet.max_code {
            return Err(Error::CorruptionDetected);
        }
        self.rows_mut(kind, 0)
            .fill(*wrapped(&alphabet.codes, code.into()));
        Ok(())
    }

    pub(crate) fn copy_from(&mut self, other: &Self, kind: Kind) {
        for (to, from) in (self.rows_mut(kind, other.log(kind)).iter_mut()).zip(other.rows(kind)) {
            *to = *from;
        }
    }
}
