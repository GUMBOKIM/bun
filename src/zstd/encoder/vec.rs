//! Allocations that may be refused.

use bun_zstd_decoder::{Error, Result};

pub(crate) fn with_room<T>(capacity: usize) -> Result<Vec<T>> {
    let mut vec = Vec::new();
    vec.try_reserve_exact(capacity)?;
    Ok(vec)
}

pub(crate) fn filled<T: Clone>(value: T, len: usize) -> Result<Vec<T>> {
    let mut vec = with_room(len)?;
    vec.resize(len, value);
    Ok(vec)
}

pub(crate) fn copied<T: Copy>(from: &[T]) -> Result<Vec<T>> {
    let mut vec = with_room(from.len())?;
    vec.extend_from_slice(from);
    Ok(vec)
}

/// Empties `vec`, and makes room for `capacity` elements.
pub(crate) fn clear_for<T>(vec: &mut Vec<T>, capacity: usize) -> Result<()> {
    vec.clear();
    Ok(vec.try_reserve_exact(capacity)?)
}

/// Makes `vec` hold `len` times `value`.
pub(crate) fn refill<T: Clone>(vec: &mut Vec<T>, value: T, len: usize) -> Result<()> {
    clear_for(vec, len)?;
    vec.resize(len, value);
    Ok(())
}

pub(crate) fn boxed<T>(value: T) -> Result<Box<T>> {
    Box::try_new(value).map_err(|_| Error::MemoryAllocation)
}
