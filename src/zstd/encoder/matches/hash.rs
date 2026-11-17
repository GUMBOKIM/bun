//! From bytes to entries of tables.

use core::hint::select_unpredictable;

use super::Owner;
use crate::bytes::likely;
use crate::parameters::TAG_BITS;
use crate::window::FIRST_INDEX;

/// Hashes the lowest bytes of a word.
#[derive(Clone, Copy)]
pub(super) struct Hasher {
    /// Shifted up, so that the other bytes of the word overflow.
    multiplier: u64,
    shift: u32,
}

impl Hasher {
    /// For `min_match` bytes, but at least 4.
    #[inline(always)]
    pub(super) fn new(min_match: u32, bits: u32) -> Self {
        let multiplier = match min_match {
            ..5 => 2_654_435_761 << 32,
            5 => 889_523_592_379 << 24,
            6 => 227_718_039_650_203 << 16,
            7 => 58_295_818_150_454_627 << 8,
            _ => 0xCF1B_BCDC_B7A5_6463,
        };
        Self {
            multiplier,
            shift: 64 - bits,
        }
    }

    #[inline(always)]
    pub(super) fn of_3_bytes(bits: u32) -> Self {
        Self {
            multiplier: 506_832_829 << 40,
            shift: 64 - bits,
        }
    }

    #[inline(always)]
    pub(super) fn with_bits(self, bits: u32) -> Self {
        Self {
            shift: 64 - bits,
            ..self
        }
    }

    #[inline(always)]
    pub(super) fn hash(self, word: u64) -> usize {
        word.wrapping_mul(self.multiplier).wrapping_shr(self.shift) as usize
    }
}

/// An entry in the table of a dictionary: an index, and some more bits of the hash.
#[derive(Clone, Copy)]
pub(super) struct Tagged(pub(super) u32);

impl Tagged {
    #[inline(always)]
    pub(super) fn new(index: u32, hash: usize) -> Self {
        Self(index << TAG_BITS | u32::from(hash as u8))
    }

    /// With the tag replaced by how it differs from that of `hash`.
    #[inline(always)]
    pub(super) fn look_up(table: &[u32], hash: usize) -> Option<Self> {
        Some(Self(
            likely(table.get(hash >> TAG_BITS))? ^ u32::from(hash as u8),
        ))
    }

    /// Of what was looked up, if the tags were the same. The first byte of a dictionary is not searched.
    #[inline(always)]
    pub(super) fn index(self) -> Option<u32> {
        if self.0 as u8 != 0 {
            return None;
        }
        // Keeps the compiler from testing this first, which is hard to predict.
        core::hint::black_box(());
        Some(self.0 >> TAG_BITS).filter(|&index| index > FIRST_INDEX)
    }
}

/// Puts `index` into the table of `OWNER`, unless it only `fills_gaps` and there is none.
#[inline(always)]
pub(super) fn insert<const OWNER: Owner>(
    table: &mut [u32],
    hash: usize,
    index: u32,
    fills_gaps: bool,
) -> Option<()> {
    let entry = likely(table.get_mut(hash >> OWNER.tag_bits()))?;
    let new = match OWNER {
        Owner::Frame => index,
        Owner::Dictionary => Tagged::new(index, hash).0,
    };
    *entry = select_unpredictable(fills_gaps && *entry != 0, *entry, new);
    Some(())
}
