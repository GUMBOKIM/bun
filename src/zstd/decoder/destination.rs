use crate::error::{Error, Result};
use crate::output::Output;

/// How many bytes behind the end of a copy it may read and write.
pub(crate) const OVERSHOOT: usize = 32;
/// Positions in it and in the block behind it fit in 32 bits.
const ADJACENT_HISTORY_MAX: usize = (1 << 32) - (1 << 19);

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

    pub(crate) fn history_len(&self) -> usize {
        self.external.len() + self.out.written() - self.history_start
    }

    /// Where the block starts in the second of what [`Self::lend`] returns.
    pub(crate) fn position(&self) -> usize {
        ADJACENT_HISTORY_MAX.min(self.out.written() - self.history_start)
    }

    /// What [`Self::lend`] has behind the history.
    pub(crate) fn lend_room(&mut self, len: usize) -> &mut [u8] {
        let position = self.position();
        self.lend(len).1.get_mut(position..).unwrap_or_default()
    }

    /// The history in two parts. Behind the second are `len` bytes for the block or more, as far as it may go, and the
    /// overshoot if possible.
    pub(crate) fn lend(&mut self, len: usize) -> (&[u8], &mut [u8]) {
        let detached = self.out.written() - self.history_start - self.position();
        let most = self.block_len_max.min(self.out.room()) + OVERSHOOT;
        let lent =
            (self.out).lend_with_slack(self.history_start, (len + OVERSHOOT).min(most), most);
        match lent.split_at_mut_checked(detached) {
            Some(([], lent)) => (self.external, lent),
            Some((detached, lent)) => (detached, lent),
            None => (&[], &mut []),
        }
    }
}
