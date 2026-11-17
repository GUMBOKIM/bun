use crate::array::only_item;
use crate::error::{Error, Result};
use crate::format::{BLOCK_LEN_MAX, DICTIONARY_MAGIC};
use crate::huffman::HuffmanTable;
use crate::sequence_tables::{Kind, SequenceTables};

/// What frames can refer to besides their own content.
pub struct Dictionary<'a> {
    bytes: Bytes<'a>,
    id: u32,
    /// One at most.
    entropy: Vec<Entropy>,
}

enum Bytes<'a> {
    Borrowed(&'a [u8]),
    Owned(Vec<u8>),
}

/// What the first block of a frame can repeat.
pub(crate) struct Entropy {
    pub(crate) huffman: HuffmanTable,
    pub(crate) sequences: SequenceTables,
    pub(crate) repeat_offsets: [u32; 3],
}

impl<'a> Dictionary<'a> {
    /// In the dictionary format if `bytes` start with its magic number, and [`raw`](Self::raw) otherwise.
    pub fn new(bytes: &'a [u8]) -> Result<Self> {
        Self::raw(bytes).with_entropy()
    }

    /// Nothing but content.
    pub fn raw(bytes: &'a [u8]) -> Self {
        Self {
            bytes: Bytes::Borrowed(bytes),
            id: 0,
            entropy: Vec::new(),
        }
    }

    /// [`new`](Self::new) for a copy of `bytes`.
    pub(crate) fn copied(bytes: &[u8]) -> Result<Dictionary<'static>> {
        let mut copy = Vec::new();
        copy.try_reserve_exact(bytes.len())?;
        copy.extend_from_slice(bytes);
        let dictionary = Dictionary {
            bytes: Bytes::Owned(copy),
            ..Dictionary::raw(&[])
        };
        dictionary.with_entropy()
    }

    fn with_entropy(mut self) -> Result<Self> {
        let Some((&[m0, m1, m2, m3, i0, i1, i2, i3], rest)) = self.history().split_first_chunk()
        else {
            return Ok(self);
        };
        if u32::from_le_bytes([m0, m1, m2, m3]) != DICTIONARY_MAGIC {
            return Ok(self);
        }
        let mut entropy = Vec::new();
        let empty = only_item(&mut entropy, || Entropy {
            huffman: HuffmanTable::default(),
            sequences: SequenceTables::EMPTY,
            repeat_offsets: [0; _],
        })?;
        empty.read(rest).map_err(|error| match error {
            Error::CorruptionDetected => Error::DictionaryCorrupted,
            other => other,
        })?;
        self.id = u32::from_le_bytes([i0, i1, i2, i3]);
        self.entropy = entropy;
        Ok(self)
    }

    /// 0 if it has none.
    pub(crate) fn id(&self) -> u32 {
        self.id
    }

    /// What is in front of a frame's content. Offsets reach into the header too, as in zstd.
    pub(crate) fn history(&self) -> &[u8] {
        match &self.bytes {
            Bytes::Borrowed(bytes) => bytes,
            Bytes::Owned(bytes) => bytes,
        }
    }

    pub(crate) fn entropy(&self) -> Option<&Entropy> {
        self.entropy.first()
    }
}

impl Entropy {
    fn read(&mut self, src: &[u8]) -> Result<()> {
        let len = self.huffman.read(src, BLOCK_LEN_MAX)?;
        let mut src = src.get(len..).ok_or(Error::Internal)?;
        for kind in [Kind::Offset, Kind::MatchLength, Kind::LiteralLength] {
            let len = self.sequences.read(kind, src)?;
            src = src.get(len..).ok_or(Error::Internal)?;
        }
        let (offsets, content) = src
            .split_first_chunk::<12>()
            .ok_or(Error::CorruptionDetected)?;
        for (to, from) in self.repeat_offsets.iter_mut().zip(offsets.as_chunks().0) {
            *to = u32::from_le_bytes(*from);
            if *to == 0 || *to as usize > content.len() {
                return Err(Error::CorruptionDetected);
            }
        }
        Ok(())
    }
}
