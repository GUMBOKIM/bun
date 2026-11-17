//! The literals section of a compressed block.

use crate::array::fill_up;
use crate::bits::load_from;
use crate::error::{Error, Result};
use crate::format::{Encoding, MIN_LITERALS_FOR_4_STREAMS};
use crate::huffman::{HuffmanTable, read_huffman_weights};

/// How many bytes behind the literals can be read too.
pub(crate) const LITERALS_PADDING: usize = 32;

#[derive(Clone, Copy)]
pub(crate) struct LiteralsHeader {
    encoding: Encoding,
    has_4_streams: bool,
    header_len: usize,
    /// How many literals there are.
    pub(crate) count: usize,
    /// With the header.
    pub(crate) section_len: usize,
}

impl LiteralsHeader {
    /// `block` decodes to at most `block_len_max` bytes, and there is `room` for it.
    pub(crate) fn read(
        block: &[u8],
        has_table: bool,
        block_len_max: usize,
        room: usize,
    ) -> Result<Self> {
        let bits = load_from(block, 0);
        let encoding = Encoding::from_low_bits(bits as u8);
        let size_format = (bits >> 2 & 3) as usize;
        let is_coded = matches!(encoding, Encoding::Compressed | Encoding::Repeat);
        let (header_len, count, content_len) = if is_coded {
            let header_len = size_format.max(1) + 2;
            let size_bits = 4 * header_len - 2;
            let size = |index| (bits >> (4 + index * size_bits)) as usize & ((1 << size_bits) - 1);
            (header_len, size(0), size(1))
        } else {
            let (header_len, count) = match size_format {
                1 => (2, bits as usize >> 4 & 0xFFF),
                3 => (3, bits as usize >> 4 & 0xF_FFFF),
                _ => (1, bits as usize >> 3 & 0x1F),
            };
            (
                header_len,
                count,
                if encoding == Encoding::Rle { 1 } else { count },
            )
        };
        let has_4_streams = is_coded && size_format != 0;
        let section_len = header_len + content_len;

        let len_min = match encoding {
            Encoding::Basic => header_len.max(2),
            Encoding::Rle => section_len,
            _ => 5,
        };
        if encoding == Encoding::Repeat && !has_table && block.len() >= 2 {
            return Err(Error::DictionaryCorrupted);
        }
        if block.len() < len_min || count > block_len_max {
            return Err(Error::CorruptionDetected);
        }
        if has_4_streams && count < MIN_LITERALS_FOR_4_STREAMS {
            return Err(Error::LiteralsHeaderWrong);
        }
        if is_coded && section_len > block.len() {
            return Err(Error::CorruptionDetected);
        }
        if count > room {
            return Err(Error::DstSizeTooSmall);
        }
        if section_len > block.len() {
            return Err(Error::CorruptionDetected);
        }
        Ok(Self {
            encoding,
            has_4_streams,
            header_len,
            count,
            section_len,
        })
    }

    pub(crate) fn has_tree(&self) -> bool {
        self.encoding == Encoding::Compressed
    }

    /// Fills `table` if [`Self::has_tree`]. Returns where the streams start.
    #[inline(always)]
    pub(crate) fn read_tree(&self, block: &[u8], table: &mut HuffmanTable) -> Result<usize> {
        if !self.has_tree() {
            return Ok(self.header_len);
        }
        let content = block.get(self.header_len..self.section_len);
        let content = content.ok_or(Error::Internal)?;
        let tree = read_huffman_weights(content)?;
        let with_pairs = self.has_4_streams && are_pairs_quicker(self.count, content.len());
        table.fill(&tree, with_pairs)?;
        Ok(self.header_len + tree.len)
    }

    /// The literals and the padding, if they are in `block` as they are.
    pub(crate) fn in_place<'a>(&self, block: &'a [u8]) -> Option<&'a [u8]> {
        let padded = block.get(self.header_len..self.section_len + LITERALS_PADDING);
        padded.filter(|_| self.encoding == Encoding::Basic)
    }

    /// `out` has room for `count`. `start` is from [`Self::read_tree`].
    #[inline(always)]
    pub(crate) fn decode(
        &self,
        block: &[u8],
        start: usize,
        table: &HuffmanTable,
        out: &mut [u8],
    ) -> Result<()> {
        let section = block.get(..self.section_len).ok_or(Error::Internal)?;
        match self.encoding {
            Encoding::Basic => {
                let literals = section.get(start..).filter(|it| it.len() == out.len());
                out.copy_from_slice(literals.ok_or(Error::Internal)?);
            }
            Encoding::Rle => out.fill(*section.last().ok_or(Error::Internal)?),
            _ if self.has_4_streams => table.decode_4_streams(section, start, out)?,
            _ => table.decode_stream(section, start, out)?,
        }
        Ok(())
    }

    /// Room in `buffer` for the literals and the padding.
    pub(crate) fn room_in<'a>(&self, buffer: &'a mut Vec<u8>) -> Result<&'a mut [u8]> {
        let len = self.count + LITERALS_PADDING;
        fill_up(buffer, len, 0)?;
        buffer.get_mut(..len).ok_or(Error::Internal)
    }
}

/// Whether looking up two literals at a time saves more than filling the table for it takes.
fn are_pairs_quicker(count: usize, compressed_len: usize) -> bool {
    #[rustfmt::skip]
    const TIMES: [[[u32; 2]; 2]; 16] = [
        [[0, 0], [1, 1]], [[0, 0], [1, 1]],
        [[150, 216], [381, 119]], [[170, 205], [514, 112]], [[177, 199], [539, 110]], [[197, 194], [644, 107]],
        [[221, 192], [735, 107]], [[256, 189], [881, 106]], [[359, 188], [1167, 109]], [[582, 187], [1570, 114]],
        [[688, 187], [1712, 122]], [[825, 186], [1965, 136]], [[976, 185], [2131, 150]], [[1180, 186], [2070, 175]],
        [[1377, 185], [1731, 202]], [[1412, 185], [1695, 202]],
    ];
    let ratio = (compressed_len * 16 / count.max(1)).min(15);
    let [single, pair] = crate::array::wrapped(&TIMES, ratio)
        .map(|[fill, per_256]| fill + per_256 * (count >> 8) as u32);
    pair + (pair >> 5) < single
}
