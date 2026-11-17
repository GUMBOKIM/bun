//! What literals and matches are expected to cost, in 1/256 of a bit.

use bun_zstd_decoder::format::{
    BLOCK_LEN_MAX, MAX_LITERAL_LENGTH_CODE, MAX_MATCH_LENGTH_CODE, MAX_OFFSET_CODE, MIN_MATCH,
};

use crate::array::At;
use crate::bytes::highest_bit;
use crate::entropy::{CodeKind, Entropy, Reuse, histogram};
use crate::sequences::{LengthCode, OffsetValue};

const ACCURACY_LOG: u32 = 8;
pub(super) const BIT: i32 = 1 << ACCURACY_LOG;
/// Blocks up to this length say too little about their literals.
pub(super) const FIXED_UP_TO: usize = 8;

#[inline(always)]
fn log2<const IS_ACCURATE: bool>(count: u32) -> i32 {
    let value = count + 1;
    let whole = highest_bit(value);
    let fraction = if IS_ACCURATE {
        (value << ACCURACY_LOG) >> whole
    } else {
        0
    };
    ((whole << ACCURACY_LOG) + fraction) as i32
}

struct Frequencies<const N: usize> {
    counts: [u32; N],
    sum: u32,
    sum_log: i32,
}

impl<const N: usize> Frequencies<N> {
    const EMPTY: Self = Self {
        counts: [0; N],
        sum: 0,
        sum_log: 0,
    };

    fn set(&mut self, counts: [u32; N]) {
        self.counts = counts;
        self.sum = counts.iter().sum();
    }

    /// From how many bits each symbol takes in a dictionary.
    fn set_from_bits(&mut self, scale_log: u32, bits: impl Fn(usize) -> u32) {
        self.set(core::array::from_fn(|symbol| match bits(symbol) {
            0 => 1,
            bits => 1 << scale_log.saturating_sub(bits),
        }));
    }

    fn scale_down(&mut self, target_log: u32) {
        let factor = self.sum >> target_log;
        if factor > 1 {
            self.set(self.counts.map(|count| 1 + (count >> highest_bit(factor))));
        }
    }

    #[inline(always)]
    fn add(&mut self, symbol: usize) {
        *self.counts.at_mut(symbol) += 1;
        self.sum += 1;
    }

    #[inline(always)]
    fn price<const IS_ACCURATE: bool>(&self, symbol: usize) -> i32 {
        self.sum_log - log2::<IS_ACCURATE>(*self.counts.at(symbol))
    }
}

pub(super) struct Prices {
    literals: Frequencies<256>,
    literal_lengths: Frequencies<{ MAX_LITERAL_LENGTH_CODE + 1 }>,
    match_lengths: Frequencies<{ MAX_MATCH_LENGTH_CODE + 1 }>,
    offsets: Frequencies<{ MAX_OFFSET_CODE + 1 }>,
    is_fixed: bool,
}

impl Prices {
    pub(super) const EMPTY: Self = Self {
        literals: Frequencies::EMPTY,
        literal_lengths: Frequencies::EMPTY,
        match_lengths: Frequencies::EMPTY,
        offsets: Frequencies::EMPTY,
        is_fixed: false,
    };

    pub(super) fn is_empty(&self) -> bool {
        self.literal_lengths.sum == 0
    }

    pub(super) fn forget(&mut self) {
        self.literal_lengths.sum = 0;
    }

    pub(super) fn start_block<const IS_ACCURATE: bool>(&mut self, block: &[u8], entropy: &Entropy) {
        self.is_fixed = false;
        if !self.is_empty() {
            self.literals.scale_down(12);
            self.literal_lengths.scale_down(11);
            self.match_lengths.scale_down(11);
            self.offsets.scale_down(11);
        } else if entropy.literals.reuse == Reuse::Always {
            let literals = entropy.literals.current();
            self.literals.set_from_bits(11, |it| literals.code_len(it));
            self.literal_lengths
                .set_from_bits(10, |it| entropy.codes.max_bits(CodeKind::LiteralLength, it));
            self.match_lengths
                .set_from_bits(10, |it| entropy.codes.max_bits(CodeKind::MatchLength, it));
            self.offsets
                .set_from_bits(10, |it| entropy.codes.max_bits(CodeKind::Offset, it));
        } else {
            self.is_fixed = block.len() <= FIXED_UP_TO;
            let mut counts = [0; 256];
            histogram::add(&mut counts, block);
            self.literals
                .set(counts.map(|count| u32::from(count > 0) + (count >> 8)));
            let starting_with = |first: &[u32], code: usize| first.get(code).copied().unwrap_or(1);
            self.literal_lengths
                .set(core::array::from_fn(|code| starting_with(&[4, 2], code)));
            self.match_lengths.set([1; _]);
            self.offsets.set(core::array::from_fn(|code| {
                starting_with(&[6, 2, 1, 1, 2, 3, 4, 4, 4, 3, 2], code)
            }));
        }
        self.update::<IS_ACCURATE>();
    }

    /// Takes the sequences into account that were added.
    #[inline(always)]
    pub(super) fn update<const IS_ACCURATE: bool>(&mut self) {
        self.literals.sum_log = log2::<IS_ACCURATE>(self.literals.sum);
        self.literal_lengths.sum_log = log2::<IS_ACCURATE>(self.literal_lengths.sum);
        self.match_lengths.sum_log = log2::<IS_ACCURATE>(self.match_lengths.sum);
        self.offsets.sum_log = log2::<IS_ACCURATE>(self.offsets.sum);
    }

    #[inline(always)]
    pub(super) fn add(&mut self, literals: &[u8], offset: OffsetValue, match_len: u32) {
        // Twice, to adapt faster.
        for &literal in literals {
            *self.literals.counts.at_mut(usize::from(literal)) += 2;
        }
        self.literals.sum += 2 * literals.len() as u32;
        self.literal_lengths
            .add(LengthCode::of_literals(literals.len() as u32).code());
        self.offsets.add(offset.code() as usize);
        self.match_lengths
            .add(LengthCode::of_match(match_len - MIN_MATCH).code());
    }

    #[inline(always)]
    pub(super) fn literal<const IS_ACCURATE: bool>(&self, literal: u8) -> i32 {
        if self.is_fixed {
            return 6 * BIT;
        }
        let price = self.literals.price::<IS_ACCURATE>(usize::from(literal));
        price.max(BIT)
    }

    #[inline(always)]
    pub(super) fn literal_len<const IS_ACCURATE: bool>(&self, len: u32) -> i32 {
        if self.is_fixed {
            return log2::<IS_ACCURATE>(len);
        }
        // There is no code for a block of nothing but literals.
        let coded = len.min(BLOCK_LEN_MAX as u32 - 1);
        let code = LengthCode::of_literals(coded);
        let bits = (code.bits() + len - coded) as i32;
        bits * BIT + self.literal_lengths.price::<IS_ACCURATE>(code.code())
    }

    /// Of the offset of a match.
    #[inline(always)]
    pub(super) fn offset<const IS_ACCURATE: bool>(&self, offset: OffsetValue) -> i32 {
        let code = offset.code() as i32;
        if self.is_fixed {
            return (16 + code) * BIT;
        }
        // Both make for faster decoding.
        let for_distance = if IS_ACCURATE {
            0
        } else {
            (code - 19).max(0) * 2 * BIT
        };
        let for_sequence = BIT / 5;
        code * BIT + self.offsets.price::<IS_ACCURATE>(code as usize) + for_distance + for_sequence
    }

    #[inline(always)]
    pub(super) fn match_len<const IS_ACCURATE: bool>(&self, len: u32) -> i32 {
        let base = len - MIN_MATCH;
        if self.is_fixed {
            return log2::<IS_ACCURATE>(base);
        }
        let code = LengthCode::of_match(base);
        code.bits() as i32 * BIT + self.match_lengths.price::<IS_ACCURATE>(code.code())
    }
}
