use bun_core::compress::State;

use crate::{Decoder, Output, Status, ZstdError};

/// Minimum spare output capacity offered to the decoder per step.
pub(crate) const OUTPUT_STEP: usize = 4096;

/// Decodes the chunks of a body, one or more frames, into a growing `Vec`.
pub struct StreamingDecoder {
    decoder: Decoder,
    state: State,
    /// The decoder filled the room it was given and may hold more. `max_output` can end a call there.
    output_full: bool,
}

impl StreamingDecoder {
    pub fn new() -> Self {
        Self {
            decoder: Decoder::new(),
            state: State::Uninitialized,
            output_full: false,
        }
    }

    #[inline]
    pub fn is_inflating(&self) -> bool {
        matches!(self.state, State::Inflating)
    }

    /// Append decompressed bytes to `out` (growing in 4096-byte steps) until `input` is
    /// consumed or `out.len()` reaches `max_output`. Returns the input bytes consumed.
    /// Returns `ShortRead` when more input is required and `is_done` is false.
    pub fn decompress(
        &mut self,
        input: &[u8],
        out: &mut Vec<u8>,
        max_output: usize,
        is_done: bool,
    ) -> Result<usize, ZstdError> {
        if matches!(self.state, State::End | State::Error) {
            return Ok(input.len());
        }

        let mut total_in = 0usize;
        loop {
            let next_in = &input[total_in..];

            // Step again with no input until the decoder leaves room unused.
            if next_in.is_empty() && !self.output_full {
                if is_done {
                    if self.state == State::Inflating {
                        self.state = State::Error;
                        return Err(ZstdError::ZstdDecompressionError);
                    }
                    self.state = State::End;
                }
                return Ok(total_in);
            }

            if out.len() >= max_output {
                return Ok(total_in);
            }

            if out.try_reserve(OUTPUT_STEP).is_err() {
                self.state = State::Error;
                return Err(ZstdError::OutOfMemory);
            }
            let budget = max_output - out.len();
            let mut output = Output::spare_capacity(out, budget);
            let (bytes_read, status) = self.decoder.step(next_in, &mut output);
            self.output_full = output.room() == 0;
            drop(output);
            let status = status.map_err(|error| {
                self.state = State::Error;
                ZstdError::for_decompression(error, ZstdError::ZstdDecompressionError)
            })?;
            total_in += bytes_read;

            if status == Status::Complete {
                self.state = State::Uninitialized;
                self.output_full = false;
                if total_in >= input.len() {
                    if is_done {
                        self.state = State::End;
                    }
                    return Ok(total_in);
                }
                continue;
            }

            self.state = State::Inflating;

            if bytes_read == next_in.len() && !self.output_full {
                if is_done {
                    self.state = State::Error;
                    return Err(ZstdError::ZstdDecompressionError);
                }
                return Err(ZstdError::ShortRead);
            }
        }
    }
}
