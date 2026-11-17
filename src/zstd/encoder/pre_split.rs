//! Ends a block early where the input seems to change, before anything is known about matches.

use bun_zstd_decoder::format::BLOCK_LEN_MAX;

use crate::array::At;
use crate::entropy::histogram;
use crate::parameters::Strategy;

type Block = [u8; BLOCK_LEN_MAX];

struct Fingerprint<const N: usize> {
    events: [u32; N],
    count: u64,
}

impl<const N: usize> Fingerprint<N> {
    fn distance(&self, other: &Self) -> u64 {
        let pairs = self.events.iter().zip(&other.events);
        let scaled = pairs.map(|(&a, &b)| (u64::from(a) * other.count, u64::from(b) * self.count));
        scaled.map(|(a, b)| a.abs_diff(b)).sum()
    }

    fn differs_from(&self, other: &Self, penalty: u64) -> bool {
        self.distance(other) >= self.count * other.count * (14 + penalty) / 16
    }
}

fn split_by_borders(block: &Block) -> usize {
    const SAMPLE_LEN: usize = 512;
    let sample = |start: usize| {
        let mut events = [0; 256];
        let bytes = block.get(start..start + SAMPLE_LEN).unwrap_or_default();
        histogram::add(&mut events, bytes);
        Fingerprint {
            events,
            count: SAMPLE_LEN as u64,
        }
    };
    let (first, last) = (sample(0), sample(block.len() - SAMPLE_LEN));
    if !first.differs_from(&last, 0) {
        return block.len();
    }
    let middle = sample((block.len() - SAMPLE_LEN) / 2);
    let (from_first, from_last) = (first.distance(&middle), last.distance(&middle));
    if from_first.abs_diff(from_last) < (SAMPLE_LEN * SAMPLE_LEN / 3) as u64 {
        block.len() / 2
    } else if from_first > from_last {
        block.len() / 4
    } else {
        block.len() / 4 * 3
    }
}

/// Looks at one in `RATE` positions, and sorts them into `N` classes.
fn split_by_chunks<const RATE: usize, const N: usize>(block: &Block) -> usize {
    const CHUNK_LEN: usize = 8 << 10;
    let fingerprint = |chunk: &[u8; CHUNK_LEN]| {
        let mut events = [0; N];
        for &pair @ [first, _] in chunk.array_windows().step_by(RATE) {
            let class = if N == 256 {
                u32::from(first)
            } else {
                u32::from(u16::from_le_bytes(pair)).wrapping_mul(0x9E37_79B9) >> (32 - N.ilog2())
            };
            *events.at_mut(class as usize) += 1;
        }
        Fingerprint {
            events,
            count: ((CHUNK_LEN - 1) / RATE) as u64,
        }
    };

    let mut chunks = block.as_chunks().0.iter().map(fingerprint);
    let Some(mut past) = chunks.next() else {
        return block.len();
    };
    for (index, chunk) in chunks.enumerate() {
        if past.differs_from(&chunk, 3u64.saturating_sub(index as u64)) {
            return (index + 1) * CHUNK_LEN;
        }
        for (past, new) in past.events.iter_mut().zip(chunk.events) {
            *past += new;
        }
        past.count += chunk.count;
    }
    block.len()
}

/// How much of `block` to compress as one.
#[inline(never)]
pub(crate) fn block_len(block: &Block, strategy: Strategy) -> usize {
    match strategy {
        Strategy::Fast => split_by_borders(block),
        Strategy::DoubleFast => split_by_chunks::<43, 256>(block),
        Strategy::Greedy | Strategy::Lazy => split_by_chunks::<11, 512>(block),
        Strategy::Lazy2 | Strategy::TreeLazy2 => split_by_chunks::<5, 1024>(block),
        Strategy::Optimal | Strategy::Ultra | Strategy::Ultra2 => split_by_chunks::<1, 1024>(block),
    }
}
