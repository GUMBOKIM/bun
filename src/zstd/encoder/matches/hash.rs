//! From bytes to entries of tables.

use crate::parameters::TAG_BITS;

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

/// The entry for `index` in the table of a dictionary.
#[inline(always)]
pub(super) fn tagged(index: u32, hash: usize) -> u32 {
    index << TAG_BITS | u32::from(hash as u8)
}

/// The index that a dictionary has for `hash`, if its tag is the same.
#[inline(always)]
pub(super) fn look_up_tagged(table: &[u32], hash: usize) -> Option<u32> {
    let &entry = table.get(hash >> TAG_BITS)?;
    (entry as u8 == hash as u8).then_some(entry >> TAG_BITS)
}
