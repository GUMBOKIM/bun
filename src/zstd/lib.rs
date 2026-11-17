//! zstd as Bun uses it. The codec is in `decoder/` and `encoder/`.
#![warn(unused_must_use)]

mod embed;
mod streaming_decoder;

use bun_removable_features::removable_definition;
use bun_zstd_decoder::{ContentSize, Dictionary};

pub use bun_zstd_decoder::{Decoder, Error, Output, Status, may_start_frame};
pub use bun_zstd_encoder::{DEFAULT_LEVEL, EndDirective, compress_bound};
pub use embed::{inflate_embedded, inflate_embedded_nul};
pub use streaming_decoder::StreamingDecoder;

/// Writes `bytes` as one frame. [`compress_bound`] says how much room is always enough.
#[inline(never)]
pub fn compress(bytes: &[u8], out: &mut Output<'_>, level: i32) -> Result<(), Error> {
    removable_definition!(ZstdCompressor, compress);
    bun_zstd_encoder::compress(bytes, out, level)
}

/// [`compress`] to the end of `out`.
pub fn compress_append(out: &mut Vec<u8>, bytes: &[u8], level: i32) -> Result<(), Error> {
    let bound = compress_bound(bytes.len()).ok_or(Error::SrcSizeWrong)?;
    out.try_reserve_exact(bound)?;
    compress(bytes, &mut Output::spare_capacity(out, bound), level)
}

pub struct Encoder(bun_zstd_encoder::Encoder);

impl Encoder {
    pub fn new(level: i32) -> Self {
        Self(bun_zstd_encoder::Encoder::new(level))
    }

    pub fn load_dictionary(&mut self, dictionary: &[u8]) -> Result<(), Error> {
        self.0.load_dictionary(dictionary)
    }

    pub fn set_pledged_len(&mut self, len: Option<u64>) -> Result<(), Error> {
        self.0.set_pledged_len(len)
    }

    /// `key` is a `ZSTD_cParameter`.
    #[inline(never)]
    pub fn set_parameter(&mut self, key: u32, value: i32) -> Result<(), Error> {
        removable_definition!(ZstdCompressor, Encoder::set_parameter);
        self.0.set_parameter(key, value)
    }

    /// Forgets the frame in progress and an error, but not the parameters and the dictionary.
    pub fn reset_session(&mut self) {
        self.0.reset_session();
    }

    /// See [`bun_zstd_encoder::Encoder::step`].
    #[inline(never)]
    pub fn step(
        &mut self,
        input: &[u8],
        out: &mut Output<'_>,
        directive: EndDirective,
    ) -> (usize, Result<Status, Error>) {
        removable_definition!(ZstdCompressor, Encoder::step);
        self.0.step(input, out, directive)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, strum::IntoStaticStr)]
pub enum ZstdError {
    /// The output, or the decoder state the frame's window size dictates, could not be allocated.
    OutOfMemory,
    InvalidZstdData,
    DecompressionFailed,
    ZstdDecompressionError,
    ShortRead,
}

bun_core::impl_tag_error!(ZstdError);

impl ZstdError {
    /// The error for a failed decompression call; `other` is the non-allocation failure.
    fn for_decompression(error: Error, other: ZstdError) -> ZstdError {
        match error {
            Error::MemoryAllocation => ZstdError::OutOfMemory,
            _ => other,
        }
    }
}

/// The size in a frame header is only a claim.
const MAX_PREALLOCATE_SIZE: usize = 16 * 1024 * 1024;

fn with_capacity(capacity: usize) -> Result<Vec<u8>, ZstdError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(capacity)
        .map_err(|_| ZstdError::OutOfMemory)?;
    Ok(output)
}

fn decompress_sized(
    src: &[u8],
    size: usize,
    dictionary: Option<&Dictionary<'_>>,
) -> Result<Vec<u8>, ZstdError> {
    let mut output = with_capacity(size)?;
    bun_zstd_decoder::decompress(
        src,
        &mut Output::spare_capacity(&mut output, size),
        dictionary,
    )
    .map_err(|error| ZstdError::for_decompression(error, ZstdError::DecompressionFailed))?;
    Ok(output)
}

/// Decompresses one or more frames.
pub fn decompress_alloc(src: &[u8]) -> Result<Vec<u8>, ZstdError> {
    let mut output = match bun_zstd_decoder::decompressed_size(src) {
        ContentSize::Invalid => return Err(ZstdError::InvalidZstdData),
        ContentSize::Known(size) if size <= MAX_PREALLOCATE_SIZE as u64 => {
            return decompress_sized(src, size as usize, None);
        }
        ContentSize::Known(_) => with_capacity(MAX_PREALLOCATE_SIZE)?,
        // A frame's output is rarely smaller than its input.
        ContentSize::Unknown => with_capacity(src.len().clamp(4096, MAX_PREALLOCATE_SIZE))?,
    };
    StreamingDecoder::new().decompress(src, &mut output, usize::MAX, true)?;
    Ok(output)
}

/// Decompresses one frame that was compressed with `prefix` as its raw content dictionary. The size has to be in the
/// header of the frame, and at most 16 MB.
pub fn decompress_alloc_with_prefix(src: &[u8], prefix: &[u8]) -> Result<Vec<u8>, ZstdError> {
    match bun_zstd_decoder::decompressed_size(src) {
        ContentSize::Known(size) if size <= MAX_PREALLOCATE_SIZE as u64 => {
            decompress_sized(src, size as usize, Some(&Dictionary::raw(prefix)))
        }
        _ => Err(ZstdError::InvalidZstdData),
    }
}
