//! The zstd encoder. Derived from zstd 1.5.7 (LICENSE.md), whose output it reproduces.
#![forbid(unsafe_code)]
// Input, dictionaries and parameters come from anywhere and a panic aborts the process.
#![deny(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable
)]
#![warn(unused_must_use)]
#![feature(
    adt_const_params,
    allocator_ext,
    hint_prefetch,
    mpmc_channel,
    vec_push_within_capacity
)]

mod array;
mod block;
mod bytes;
mod dictionary;
mod entropy;
mod frame;
mod levels;
mod matches;
mod parameters;
mod post_split;
mod pre_split;
mod sequences;
mod stream;
mod sub_blocks;
mod vec;
mod window;
mod workers;

use bun_zstd_decoder::format::BLOCK_LEN_MAX;
use bun_zstd_decoder::{Output, Result};

pub use parameters::DEFAULT_LEVEL;
pub use stream::{Encoder, EndDirective};

use frame::{FrameWriter, Start};
use parameters::{Purpose, Request};
use window::Input;

fn worst_case_len(len: usize) -> usize {
    let margin = BLOCK_LEN_MAX.saturating_sub(len) >> 11;
    len.saturating_add(len >> 8).saturating_add(margin)
}

/// The room that [`compress`] can need for `len` bytes. `None` if that is too many.
pub fn compress_bound(len: usize) -> Option<usize> {
    (len < 0xFF00_FF00_FF00_FF00).then(|| worst_case_len(len))
}

/// Writes `bytes` as one frame.
pub fn compress(bytes: &[u8], out: &mut Output<'_>, level: i32) -> Result<()> {
    let (request, len) = (Request::new(level), Some(bytes.len() as u64));
    let tuning = request.tuning(len, Purpose::Frame { dictionary_len: 0 });
    FrameWriter::new(request.plan(tuning), len, Start::Frame(None))?.write(
        &Input::alone(bytes, &[]),
        None,
        true,
        out,
    )
}
