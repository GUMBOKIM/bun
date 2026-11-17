//! What the search for matches makes of a block.

use bun_zstd_decoder::format::{
    INITIAL_REPEAT_OFFSETS, LITERAL_LENGTH_BITS, MATCH_LENGTH_BITS, MAX_LITERAL_LENGTH_CODE,
    MAX_MATCH_LENGTH_CODE, MIN_MATCH,
};
use bun_zstd_decoder::{Error, Result, with_bmi2};

use crate::array::At;
use crate::bytes::{chunk_at, chunk_at_mut, highest_bit, put};
use crate::entropy::fse::SYMBOLS;
use crate::vec;

/// An offset as the format has it: 1 to 3 repeat a recent one, the others are raised by 3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct OffsetValue(pub(crate) u32);

impl OffsetValue {
    pub(crate) const FIRST_REPEAT: Self = Self(1);
    const REPEATS: u32 = 3;

    #[inline(always)]
    pub(crate) fn new(offset: u32) -> Self {
        Self(offset.wrapping_add(Self::REPEATS))
    }

    /// `None` for a repeat.
    #[inline(always)]
    pub(crate) fn offset(self) -> Option<u32> {
        (self.0 > Self::REPEATS).then(|| self.0 - Self::REPEATS)
    }

    #[inline(always)]
    pub(crate) fn code(self) -> u32 {
        highest_bit(self.0)
    }
}

/// The three most recent offsets. 0 stands for one that cannot be used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Repeats(pub(crate) [u32; 3]);

impl Repeats {
    pub(crate) const INITIAL: Self = Self(INITIAL_REPEAT_OFFSETS);

    /// Without literals in front of a match, the numbers of the repeats are shifted by one.
    #[inline(always)]
    pub(crate) fn get(self, number: u32) -> u32 {
        let [first, second, third] = self.0;
        *[first, second, third, first.wrapping_sub(1)].at(number as usize)
    }

    #[inline(always)]
    pub(crate) fn resolve(self, value: OffsetValue, has_literals: bool) -> u32 {
        match value.offset() {
            Some(offset) => offset,
            None => self.get(value.0.wrapping_sub(u32::from(has_literals))),
        }
    }

    #[inline(always)]
    #[must_use]
    pub(crate) fn after(self, value: OffsetValue, has_literals: bool) -> Self {
        let [first, second, third] = self.0;
        if let Some(offset) = value.offset() {
            return Self([offset, first, second]);
        }
        match value.0.wrapping_sub(u32::from(has_literals)) {
            0 => self,
            1 => Self([second, first, third]),
            number => Self([self.get(number), first, second]),
        }
    }
}

/// A length as the format has it: a code, and how many bits follow.
#[derive(Clone, Copy)]
pub(crate) struct LengthCode(u16);

impl LengthCode {
    const BITS_SHIFT: u32 = 11;

    const fn new(code: u32, bits: u32) -> Self {
        Self((code | bits << Self::BITS_SHIFT) as u16)
    }

    /// For the lengths from 0 on, given the bits of each code.
    #[expect(clippy::indexing_slicing, reason = "evaluated at compile time")]
    const fn table<const N: usize>(bits: &[u8]) -> [Self; N] {
        let mut table = [Self(0); N];
        let (mut len, mut code, mut end) = (0, 0, 1);
        while len < N {
            if len == end {
                code += 1;
                end += 1 << bits[code];
            }
            table[len] = Self::new(code as u32, bits[code] as u32);
            len += 1;
        }
        table
    }

    #[inline(always)]
    pub(crate) fn of_literals(len: u32) -> Self {
        static SHORT: [LengthCode; 64] = LengthCode::table(&LITERAL_LENGTH_BITS);
        match SHORT.get(len as usize) {
            Some(&code) => code,
            None => Self::new(highest_bit(len) + 19, highest_bit(len)),
        }
    }

    /// `base` is the length less [`MIN_MATCH`].
    #[inline(always)]
    pub(crate) fn of_match(base: u32) -> Self {
        static SHORT: [LengthCode; 128] = LengthCode::table(&MATCH_LENGTH_BITS);
        match SHORT.get(base as usize) {
            Some(&code) => code,
            None => Self::new(highest_bit(base) + 36, highest_bit(base)),
        }
    }

    #[inline(always)]
    pub(crate) fn code(self) -> usize {
        usize::from(self.0) % SYMBOLS
    }

    #[inline(always)]
    pub(crate) fn bits(self) -> u32 {
        u32::from(self.0) >> Self::BITS_SHIFT
    }
}

#[expect(clippy::indexing_slicing, reason = "evaluated at compile time")]
const fn first_lengths<const N: usize>(bits: &[u8; N]) -> [u32; N] {
    let mut first = [0; N];
    let mut code = 1;
    while code < N {
        first[code] = first[code - 1] + (1 << bits[code - 1]);
        code += 1;
    }
    first
}

#[inline(always)]
fn low_bits(value: u64, count: u32) -> u64 {
    value & (1u64.wrapping_shl(count) - 1)
}

/// Literals and the match after them.
#[derive(Clone, Copy)]
#[repr(C, packed(4))]
pub(crate) struct Sequence {
    /// What follows the codes, in the order in which it is written.
    extra: u64,
    /// By [`CodeKind`](crate::entropy::CodeKind), and the number of extra bits.
    codes: [u8; 4],
}

impl Sequence {
    #[inline(always)]
    fn new(literal_len: u32, offset: OffsetValue, match_base: u32) -> Self {
        let literals = LengthCode::of_literals(literal_len);
        let matched = LengthCode::of_match(match_base);
        let length_bits = literals.bits() + matched.bits();
        let extra = low_bits(u64::from(literal_len), literals.bits())
            | low_bits(u64::from(match_base), matched.bits()) << literals.bits()
            | low_bits(u64::from(offset.0), offset.code()) << length_bits;
        Self {
            extra,
            codes: [
                literals.code() as u8,
                offset.code() as u8,
                matched.code() as u8,
                (length_bits + offset.code()) as u8,
            ],
        }
    }

    #[inline(always)]
    pub(crate) fn codes(self) -> [u8; 3] {
        let [codes @ .., _] = self.codes;
        codes
    }

    /// And how many there are.
    #[inline(always)]
    pub(crate) fn extra_bits(self) -> (u64, u32) {
        let [.., count] = self.codes;
        (self.extra, u32::from(count))
    }

    pub(crate) fn literal_len(self) -> u32 {
        static FIRST: [u32; MAX_LITERAL_LENGTH_CODE + 1] = first_lengths(&LITERAL_LENGTH_BITS);
        let [code, ..] = self.codes.map(usize::from);
        let bits = u32::from(*LITERAL_LENGTH_BITS.at(code));
        FIRST.at(code) + low_bits(self.extra_bits().0, bits) as u32
    }

    /// The length less [`MIN_MATCH`].
    pub(crate) fn match_base(self) -> u32 {
        static FIRST: [u32; MAX_MATCH_LENGTH_CODE + 1] = first_lengths(&MATCH_LENGTH_BITS);
        let [literals, _, code, _] = self.codes.map(usize::from);
        let extra = self.extra_bits().0 >> LITERAL_LENGTH_BITS.at(literals);
        FIRST.at(code) + low_bits(extra, u32::from(*MATCH_LENGTH_BITS.at(code))) as u32
    }

    pub(crate) fn offset(self) -> OffsetValue {
        let ([_, code, _, count], (extra, _)) = (self.codes, self.extra_bits());
        OffsetValue(1 << code | (extra >> (count - code)) as u32)
    }

    pub(crate) fn set_offset(&mut self, offset: OffsetValue) {
        *self = Self::new(self.literal_len(), offset, self.match_base());
    }
}

/// How often each code occurs, by [`CodeKind`](crate::entropy::CodeKind).
pub(crate) type CodeCounts = [[u32; SYMBOLS]; 3];

#[inline(always)]
fn count(counts: &mut CodeCounts, sequence: Sequence) {
    for (counts, code) in counts.iter_mut().zip(sequence.codes()) {
        *counts.at_mut(usize::from(code)) += 1;
    }
}

pub(crate) fn count_codes(sequences: &[Sequence]) -> CodeCounts {
    let mut counts = [[0; SYMBOLS]; 3];
    for &sequence in sequences {
        count(&mut counts, sequence);
    }
    counts
}

/// How far the copy of literals may go past their end.
const OVERCOPY: usize = 16;

/// Out of line, or the compiler would call `memcpy()` for the short ones as well.
#[inline(never)]
fn copy_literals(block: &[u8], from: u32, literals: &mut [u8], to: u32, len: u32) -> Option<()> {
    let (block, literals) = (
        block.get(from as usize..)?,
        literals.get_mut(to as usize..)?,
    );
    if len <= 64
        && let Some(chunk) = block.first_chunk::<64>()
        && let Some(room) = literals.first_chunk_mut()
    {
        *room = *chunk;
        return Some(());
    }
    put(literals, block.get(..len as usize)?)
}

pub(crate) struct Sequences {
    /// Until they are [`finish`](Self::finish)ed, the fields hold the two lengths and the offset.
    pub(crate) list: Vec<Sequence>,
    literals: Vec<u8>,
    literals_len: usize,
    pub(crate) counts: CodeCounts,
}

impl Sequences {
    pub(crate) const fn new() -> Self {
        Self {
            list: Vec::new(),
            literals: Vec::new(),
            literals_len: 0,
            counts: [[0; SYMBOLS]; 3],
        }
    }

    pub(crate) fn clear_for(&mut self, block_len: usize, min_match: u32) -> Result<()> {
        let shortest = if min_match == 3 { 3 } else { 4 };
        vec::clear_for(&mut self.list, block_len / shortest)?;
        vec::refill(&mut self.literals, 0, block_len + OVERCOPY)
    }

    pub(crate) fn literals(&self) -> &[u8] {
        self.literals.get(..self.literals_len).unwrap_or_default()
    }

    pub(crate) fn split_mut(&mut self) -> (&mut [Sequence], &[u8]) {
        let literals = self.literals.get(..self.literals_len);
        (&mut self.list, literals.unwrap_or_default())
    }

    #[inline(always)]
    pub(crate) fn push(&mut self, literal_len: u32, offset: OffsetValue, match_len: u32) {
        let pushed = self.list.push_within_capacity(Sequence {
            extra: u64::from(literal_len) | u64::from(match_len) << 32,
            codes: offset.0.to_le_bytes(),
        });
        debug_assert!(pushed.is_ok());
    }

    /// `block` is what was searched.
    pub(crate) fn finish(&mut self, block: &[u8]) -> Result<()> {
        let (list, literals) = (self.list.as_mut_slice(), self.literals.as_mut_slice());
        let counts = &mut self.counts;
        *counts = [[0; SYMBOLS]; 3];
        let literals_len = with_bmi2(
            #[inline(always)]
            move || {
                let (mut from, mut to) = (0u32, 0u32);
                for sequence in list {
                    let (literal_len, match_len) =
                        (sequence.extra as u32, (sequence.extra >> 32) as u32);
                    match chunk_at::<OVERCOPY>(block, from) {
                        Some(chunk) if literal_len as usize <= OVERCOPY => {
                            *chunk_at_mut(literals, to)? = *chunk;
                        }
                        _ => copy_literals(block, from, literals, to, literal_len)?,
                    }
                    from = from.wrapping_add(literal_len).wrapping_add(match_len);
                    to = to.wrapping_add(literal_len);

                    let offset = OffsetValue(u32::from_le_bytes(sequence.codes));
                    let coded =
                        Sequence::new(literal_len, offset, match_len.wrapping_sub(MIN_MATCH));
                    count(counts, coded);
                    *sequence = coded;
                }
                let last = block.get(from as usize..)?;
                put(literals.get_mut(to as usize..)?, last)?;
                Some(to as usize + last.len())
            },
        );
        self.literals_len = literals_len.ok_or(Error::Internal)?;
        Ok(())
    }
}
