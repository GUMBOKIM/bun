use crate::bytes::chunk_at_mut;

const WORD_LEN: usize = 8;

/// Takes a word at a time, of which some bytes count. Running out of room shows at the end.
pub(super) struct WordWriter<'a> {
    /// The compiler sees its length as `last + WORD_LEN`, and that it need not check it.
    out: &'a mut [u8],
    last: u32,
    position: u32,
}

impl<'a> WordWriter<'a> {
    #[inline(always)]
    pub(super) fn new(out: &'a mut [u8]) -> Option<Self> {
        let last = out.len().checked_sub(WORD_LEN)? as u32;
        Some(Self {
            out: out.get_mut(..last as usize + WORD_LEN)?,
            last,
            position: 0,
        })
    }

    #[inline(always)]
    pub(super) fn put(&mut self, word: u64, len: u32) {
        let position = self.position.min(self.last);
        if let Some(bytes) = chunk_at_mut(self.out, position) {
            *bytes = word.to_le_bytes();
        }
        self.position = position + len;
    }

    /// The length of what was written, if there was room for it.
    #[inline(always)]
    pub(super) fn finish(self, has_partial_byte: bool) -> Option<usize> {
        (self.position < self.last).then(|| self.position as usize + usize::from(has_partial_byte))
    }
}

/// Writes bits from the lowest up.
pub(super) struct BitWriter<'a> {
    words: WordWriter<'a>,
    bits: u64,
    count: u32,
}

impl<'a> BitWriter<'a> {
    #[inline(always)]
    pub(super) fn new(out: &'a mut [u8]) -> Option<Self> {
        Some(Self {
            words: WordWriter::new(out)?,
            bits: 0,
            count: 0,
        })
    }

    /// The lowest `count` bits of `value`. At most 63 bits can be pending.
    #[inline(always)]
    pub(super) fn add(&mut self, value: u32, count: u32) {
        self.add_all(u64::from(value) & (1u64.wrapping_shl(count) - 1), count);
    }

    /// For a `value` without higher bits.
    #[inline(always)]
    pub(super) fn add_all(&mut self, value: u64, count: u32) {
        self.bits |= value.wrapping_shl(self.count);
        self.count += count;
    }

    /// Leaves at most 7 bits pending.
    #[inline(always)]
    pub(super) fn flush(&mut self) {
        self.words.put(self.bits, self.count / 8);
        self.bits = self.bits.wrapping_shr(self.count & !7);
        self.count %= 8;
    }

    /// The length of the stream, if there was room for it.
    #[inline(always)]
    pub(super) fn finish(mut self) -> Option<usize> {
        self.add(1, 1);
        self.flush();
        self.words.finish(self.count > 0)
    }
}
