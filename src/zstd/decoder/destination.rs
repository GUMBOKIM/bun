//! Where a block goes, and what it can refer to.

use crate::error::{Error, Result};
use crate::frame::FrameHeader;
use crate::output::Output;

/// How many bytes behind the end of a copy it may read and write.
pub(crate) const OVERSHOOT: usize = 32;
/// Positions in it and in the block behind it fit in 32 bits.
const ADJACENT_HISTORY_MAX: usize = (1 << 32) - (1 << 19);

/// The size in a header is only a claim. It gets memory up to this much...
const CREDIBLE_SIZE: u64 = 16 << 20;
/// ...or this many times what has turned out to be there.
const CREDIBLE_FACTOR: u64 = 8;
/// More output than this is not moved to make room for the overshoot.
const CHEAP_TO_MOVE: usize = 1 << 20;

pub(crate) struct Destination<'o, 'a> {
    out: &'o mut Output<'a>,
    frame: &'o FrameHeader,
    /// How many of the last bytes that `out` has written are history.
    history_len: usize,
    /// The end of the history before those.
    external: &'o [u8],
    /// That in `out`, and in what `external` keeps from being there.
    room: usize,
    /// How long the block can get.
    limit: usize,
}

pub(crate) struct Loan<'a> {
    /// The older and the newer part of the history before `out`. If one is empty, it is the older.
    pub(crate) detached: [&'a [u8]; 2],
    /// The rest of the history, and room for the block.
    pub(crate) out: &'a mut [u8],
}

impl<'o, 'a> Destination<'o, 'a> {
    pub(crate) fn new(
        out: &'o mut Output<'a>,
        frame: &'o FrameHeader,
        history_len: usize,
        external: &'o [u8],
        room: usize,
    ) -> Self {
        Self {
            room,
            limit: frame.block_len_max().min(room),
            out,
            frame,
            history_len,
            external,
        }
    }

    pub(crate) fn block_len_max(&self) -> usize {
        self.frame.block_len_max()
    }

    pub(crate) fn room(&self) -> usize {
        self.room
    }

    #[inline]
    pub(crate) fn check_block_len(&self, len: usize) -> Result<()> {
        if len <= self.limit {
            Ok(())
        } else if len > self.room() {
            Err(Error::DstSizeTooSmall)
        } else {
            Err(Error::CorruptionDetected)
        }
    }

    /// Ends the block with `bytes`, behind `len` bytes of what was lent. Returns its length.
    pub(crate) fn finish(mut self, len: usize, bytes: &[u8]) -> Result<usize> {
        self.check_block_len(len + bytes.len())?;
        self.reserve(len + bytes.len())?;
        self.out.commit(len);
        Ok(len + self.out.append(bytes))
    }

    /// `content` is that of an RLE block.
    pub(crate) fn finish_repeated(mut self, content: &[u8], count: usize) -> Result<usize> {
        self.check_block_len(count)?;
        self.reserve(count)?;
        (self.out).append_repeated(*content.first().ok_or(Error::Internal)?, count);
        Ok(count)
    }

    /// For `len` bytes from the start of the block on.
    #[inline]
    fn reserve(&mut self, len: usize) -> Result<()> {
        if !self.out.grows() || len <= self.out.reserved_room() {
            return Ok(());
        }
        let written = self.history_len as u64;
        let credible = CREDIBLE_SIZE.max(written.saturating_mul(CREDIBLE_FACTOR));
        let size = self.frame.content_size.filter(|&size| size <= credible);
        let left = size.map_or(0, |size| size.saturating_sub(written) as usize + OVERSHOOT);
        self.out.reserve(len.max(left))
    }

    /// About how far back matches can be.
    pub(crate) fn reach(&self) -> u64 {
        ((self.history_len + self.external.len()) as u64).min(self.frame.window_size)
    }

    /// Where the block starts in [`Loan::out`].
    #[inline]
    pub(crate) fn position(&self) -> usize {
        self.history_len.min(ADJACENT_HISTORY_MAX)
    }

    /// What [`Self::lend`] has behind the history.
    pub(crate) fn lend_room(&mut self, len: usize, more: usize) -> Result<&mut [u8]> {
        let position = self.position();
        (self.lend(len, more)?.out.get_mut(position..)).ok_or(Error::Internal)
    }

    /// With `len` bytes for the block, as far as it may go. If possible also `more`, and the overshoot.
    pub(crate) fn lend(&mut self, len: usize, more: usize) -> Result<Loan<'_>> {
        let wanted = (len + more).min(self.limit) + OVERSHOOT;
        if self.out.grows() {
            let overshoot = if self.out.len_to_move() <= CHEAP_TO_MOVE {
                OVERSHOOT
            } else {
                0
            };
            if len.min(self.limit) + overshoot > self.out.reserved_room() {
                self.reserve(wanted)?;
            }
        }
        let detached_len = self.history_len - self.position();
        let lent = (self.out).lend_with_slack(self.history_len, wanted, self.limit + OVERSHOOT);
        let (detached, out) = lent
            .split_at_mut_checked(detached_len)
            .ok_or(Error::Internal)?;
        Ok(Loan {
            detached: if detached.is_empty() {
                [&[], self.external]
            } else {
                [self.external, detached]
            },
            out,
        })
    }
}
