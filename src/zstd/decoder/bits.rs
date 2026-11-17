//! Readers of bitstreams.

use crate::error::{Error, Result};

/// The 8 bytes of `src` from `start` on. Zeros follow its end.
#[inline(always)]
pub(crate) fn load_from(src: &[u8], start: usize) -> u64 {
    #[cold]
    #[inline(never)]
    fn padded(src: &[u8], start: usize) -> u64 {
        let bytes = src.get(start..).unwrap_or_default();
        (bytes.iter().rev()).fold(0, |word, &byte| word << 8 | u64::from(byte))
    }

    match src.get(start..).and_then(<[u8]>::first_chunk) {
        Some(bytes) => u64::from_le_bytes(*bytes),
        None => padded(src, start),
    }
}

/// How many bits of `src` are below the mark that ends the stream at its end, which starts at byte `start`.
#[inline(always)]
pub(crate) fn bits_below_mark(src: &[u8], start: usize) -> Result<usize> {
    match src.get(start..) {
        Some(&[.., last @ 1..=u8::MAX]) => Ok(8 * (src.len() - 1) + last.ilog2() as usize),
        _ => Err(Error::CorruptionDetected),
    }
}

/// Reads a stream from its last bit to its first. What it reads beyond the first is garbage.
#[derive(Clone, Copy)]
pub(crate) struct BackwardBits<'a> {
    /// Ends with the bytes in `container`.
    src: &'a [u8],
    container: u64,
    /// The low bits of `container` that are yet to be read.
    available: isize,
    /// How many bits of `src` are below the stream.
    floor: usize,
}

impl<'a> BackwardBits<'a> {
    /// What [`Self::refill`] makes available, if there are as many.
    pub(crate) const IN_REFILL: u32 = 56;

    /// For the stream in `src[start..]`.
    #[inline(always)]
    pub(crate) fn new(src: &'a [u8], start: usize) -> Result<Self> {
        let below = src.len().saturating_sub(8);
        Ok(Self {
            src,
            container: load_from(src, below),
            available: (bits_below_mark(src, start)? - 8 * below) as isize,
            floor: 8 * start,
        })
    }

    #[inline(always)]
    pub(crate) fn read(&mut self, count: u32) -> u64 {
        self.available -= count as isize;
        self.container.wrapping_shr(self.available as u32) & !u64::MAX.wrapping_shl(count)
    }

    #[inline(always)]
    pub(crate) fn refill(&mut self) {
        let below = self.src.len().saturating_sub(8);
        let bytes = below.min((63 - self.available) as usize / 8);
        self.src = self.src.get(..self.src.len() - bytes).unwrap_or_default();
        self.available += 8 * bytes as isize;
        if let Some(last) = self.src.last_chunk() {
            self.container = u64::from_le_bytes(*last);
        }
    }

    /// How many bits of the stream are yet to be read. Negative once more were read than it has.
    #[inline(always)]
    pub(crate) fn left(&self) -> isize {
        8 * self.src.len().saturating_sub(8) as isize + self.available - self.floor as isize
    }

    /// Whether `count` bits, at most [`Self::IN_REFILL`], and a refill after them need no care.
    #[inline(always)]
    pub(crate) fn can_take(&self, count: u32) -> bool {
        self.available >= count as isize && self.src.len() >= 15
    }

    /// If [`Self::can_take`]: the low `count` bits of the result are the next, and it refills.
    #[inline(always)]
    pub(crate) fn take(&mut self, count: u32) -> u64 {
        let available = (self.available - count as isize) as usize % 64;
        let taken = self.container >> available;
        let bytes = (63 - available) / 8;
        self.src = (self.src.get(..self.src.len().wrapping_sub(bytes))).unwrap_or_default();
        self.available = (available | 56) as isize;
        self.container = u64::from_le_bytes(self.src.last_chunk().copied().unwrap_or_default());
        taken
    }
}
