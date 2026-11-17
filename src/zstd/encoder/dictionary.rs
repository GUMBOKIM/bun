use core::ops::Range;

use bun_zstd_decoder::format::{
    BLOCK_LEN_MAX, DICTIONARY_MAGIC, LITERAL_LENGTH_TABLE_LOG_MAX, MATCH_LENGTH_TABLE_LOG_MAX,
    MAX_LITERAL_LENGTH_CODE, MAX_MATCH_LENGTH_CODE, MAX_OFFSET_CODE, OFFSET_TABLE_LOG_MAX,
};
use bun_zstd_decoder::{Error, Result, read_huffman_weights, read_normalized_counts};

use crate::array::At;
use crate::bytes::{highest_bit, u32_at};
use crate::entropy::fse::{Normalized, SYMBOLS};
use crate::entropy::{CodeKind, Entropy, Reuse};
use crate::matches::{MatchState, Owner, Usage};
use crate::parameters::{Purpose, Request};
use crate::sequences::Repeats;
use crate::vec::{self, Shared};
use crate::window::WORD_LEN;

/// Ready for frames to start from.
pub(crate) struct Dictionary {
    bytes: Shared<Vec<u8>>,
    content: Range<usize>,
    pub(crate) id: u32,
    pub(crate) state: MatchState,
    pub(crate) entropy: Box<Entropy>,
    pub(crate) repeats: Repeats,
}

fn read_counts(bytes: &mut &[u8], max_symbol: usize, log_max: u32) -> Option<Normalized> {
    let mut counts = [0; SYMBOLS];
    let description = read_normalized_counts(bytes, counts.get_mut(..=max_symbol)?).ok()?;
    *bytes = bytes.get(description.len..)?;
    (description.table_log <= log_max).then_some(Normalized {
        counts,
        max_symbol: description.max_symbol,
        table_log: description.table_log,
    })
}

fn reuse(has_all_symbols: bool) -> Reuse {
    if has_all_symbols {
        Reuse::Always
    } else {
        Reuse::IfComplete
    }
}

/// Reads what is between the ID and the content, and returns the content.
fn read_tables<'a>(mut bytes: &'a [u8], entropy: &mut Entropy) -> Option<(Repeats, &'a [u8])> {
    let description = read_huffman_weights(bytes).ok()?;
    bytes = bytes.get(description.len..)?;
    let weights = description.weights.get(..description.symbol_count)?;
    let has_all = weights.len() == 256 && *description.rank_counts.at(0) == 0;
    entropy
        .literals
        .load(weights, description.table_log, reuse(has_all));

    let offsets = read_counts(&mut bytes, MAX_OFFSET_CODE, OFFSET_TABLE_LOG_MAX)?;
    let match_lengths = read_counts(
        &mut bytes,
        MAX_MATCH_LENGTH_CODE,
        MATCH_LENGTH_TABLE_LOG_MAX,
    )?;
    let literal_lengths = read_counts(
        &mut bytes,
        MAX_LITERAL_LENGTH_CODE,
        LITERAL_LENGTH_TABLE_LOG_MAX,
    )?;
    let (repeats, content) = bytes.split_first_chunk::<12>()?;
    let repeats = [0, 4, 8].map(|at| u32_at(repeats, at).unwrap_or(0));
    let max_offset = u32::try_from(content.len() + BLOCK_LEN_MAX).unwrap_or(u32::MAX);
    for (kind, counts, max_needed) in [
        (CodeKind::Offset, offsets, highest_bit(max_offset) as usize),
        (CodeKind::MatchLength, match_lengths, MAX_MATCH_LENGTH_CODE),
        (
            CodeKind::LiteralLength,
            literal_lengths,
            MAX_LITERAL_LENGTH_CODE,
        ),
    ] {
        let has_all = (0..=max_needed).all(|symbol| *counts.counts.at(symbol) != 0);
        entropy.codes.load(kind, &counts, reuse(has_all));
    }
    Some((Repeats(repeats), content))
}

impl Dictionary {
    /// Without the magic number, it is all content.
    pub(crate) fn new(bytes: &Shared<Vec<u8>>, request: &Request) -> Result<Self> {
        let tuning = request.tuning(None, Purpose::Dictionary { len: bytes.len() });
        let mut state = MatchState::new(tuning);
        let usage = Usage {
            owner: Owner::Dictionary,
            preceding_len: 0,
            may_be_short: false,
        };
        state.reset(tuning, tuning.prefers_rows(), &usage)?;
        let mut entropy = vec::boxed(Entropy::EMPTY)?;

        let has_tables = bytes.len() >= WORD_LEN && u32_at(bytes, 0) == Some(DICTIONARY_MAGIC);
        let (id, repeats, content) = match bytes.get(WORD_LEN..) {
            None => (0, Repeats::INITIAL, &[][..]),
            Some(rest) if has_tables => {
                let (repeats, content) =
                    read_tables(rest, &mut entropy).ok_or(Error::DictionaryCorrupted)?;
                (u32_at(bytes, 4).unwrap_or(0), repeats, content)
            }
            Some(_) => (0, Repeats::INITIAL, bytes.as_slice()),
        };
        let content = state.loadable(content, Owner::Dictionary);
        let is_in_content = |offset: u32| (1..=content.len()).contains(&(offset as usize));
        if has_tables && !repeats.0.into_iter().all(is_in_content) {
            return Err(Error::DictionaryCorrupted);
        }
        if !content.is_empty() {
            state.load(content, Owner::Dictionary);
        }
        Ok(Self {
            content: bytes.len() - content.len()..bytes.len(),
            bytes: Shared::clone(bytes),
            id,
            state,
            entropy,
            repeats,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.bytes.len()
    }

    /// What the tables index.
    pub(crate) fn content(&self) -> &[u8] {
        self.bytes.get(self.content.clone()).unwrap_or_default()
    }

    /// Whether a frame searches these tables. Copying them only pays off for large inputs.
    pub(crate) fn is_attached_for(&self, input_len: Option<u64>) -> bool {
        const KB_BY_STRATEGY: [u64; 10] = [8, 8, 16, 32, 32, 32, 32, 32, 8, 8];
        let limit = KB_BY_STRATEGY.at(self.state.tuning.strategy as usize) << 10;
        input_len.is_none_or(|len| len <= limit)
    }
}
