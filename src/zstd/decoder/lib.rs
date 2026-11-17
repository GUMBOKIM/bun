//! The zstd format (RFC 8878) and its decoder. Derived from zstd 1.5.7 (LICENSE.md).
#![deny(unsafe_code)]
// Frames come from anywhere and a panic aborts the process.
#![deny(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable
)]
#![warn(unused_must_use)]

mod cpu;
mod error;
pub mod format;
mod output;
mod xxhash;

pub use cpu::with_bmi2;
pub use error::{Error, Result};
pub use output::Output;
pub use xxhash::XxHash64;
