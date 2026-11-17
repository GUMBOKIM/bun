//! What the encoder and the decoder agree on.

pub const FRAME_MAGIC: u32 = 0xFD2F_B528;
pub const DICTIONARY_MAGIC: u32 = 0xEC30_A437;
/// The low 4 bits are free.
pub const SKIPPABLE_MAGIC: u32 = 0x184D_2A50;

pub const WINDOW_LOG_MIN: u32 = 10;
pub const WINDOW_LOG_MAX: u32 = 31;
/// The largest window that a streaming decoder accepts unless it is told otherwise.
pub const WINDOW_LOG_LIMIT_DEFAULT: u32 = 27;

pub const FRAME_HEADER_LEN_MAX: usize = 18;
pub const BLOCK_HEADER_LEN: usize = 3;
pub const BLOCK_LEN_MAX: usize = 128 << 10;
pub const CHECKSUM_LEN: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BlockType {
    Raw = 0,
    Rle = 1,
    Compressed = 2,
}

/// How literals, or one kind of sequence codes, are encoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Encoding {
    /// Raw literals, or the predefined distribution.
    Basic = 0,
    Rle = 1,
    Compressed = 2,
    Repeat = 3,
}

pub const MIN_MATCH: u32 = 3;
pub const INITIAL_REPEAT_OFFSETS: [u32; 3] = [1, 4, 8];
/// Sequence counts from here on take 3 bytes.
pub const LONG_SEQUENCE_COUNT: usize = 0x7F00;
pub const MIN_LITERALS_FOR_4_STREAMS: usize = 6;

pub const HUFFMAN_TABLE_LOG_MAX: u32 = 12;
pub const FSE_TABLE_LOG_MIN: u32 = 5;

pub const MAX_LITERAL_LENGTH_CODE: usize = 35;
pub const MAX_MATCH_LENGTH_CODE: usize = 52;
pub const MAX_OFFSET_CODE: usize = 31;

pub const LITERAL_LENGTH_TABLE_LOG_MAX: u32 = 9;
pub const MATCH_LENGTH_TABLE_LOG_MAX: u32 = 9;
pub const OFFSET_TABLE_LOG_MAX: u32 = 8;

#[rustfmt::skip]
pub const LITERAL_LENGTH_BITS: [u8; MAX_LITERAL_LENGTH_CODE + 1] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11, 12,
    13, 14, 15, 16,
];

#[rustfmt::skip]
pub const MATCH_LENGTH_BITS: [u8; MAX_MATCH_LENGTH_CODE + 1] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11,
    12, 13, 14, 15, 16,
];

/// A predefined distribution. -1 is a probability of less than 1.
pub struct Distribution {
    pub counts: &'static [i16],
    pub table_log: u32,
}

#[rustfmt::skip]
pub const DEFAULT_LITERAL_LENGTHS: Distribution = Distribution {
    counts: &[
        4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1,
        -1, -1, -1, -1,
    ],
    table_log: 6,
};

#[rustfmt::skip]
pub const DEFAULT_MATCH_LENGTHS: Distribution = Distribution {
    counts: &[
        1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1,
        -1, -1, -1, -1, -1,
    ],
    table_log: 6,
};

#[rustfmt::skip]
pub const DEFAULT_OFFSETS: Distribution = Distribution {
    counts: &[
        1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1,
    ],
    table_log: 5,
};

/// The step with which FSE spreads symbols over a table.
pub const fn fse_table_step(table_size: usize) -> usize {
    (table_size >> 1) + (table_size >> 3) + 3
}
