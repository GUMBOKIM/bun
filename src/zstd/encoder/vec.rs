//! Allocations that may be refused.

use bun_pod::Zeroable;
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

/// For what may stay unused.
pub(crate) fn zeroed<T: Zeroable>(len: usize) -> Result<Vec<T>> {
    bun_pod::try_zeroed_vec(len).ok_or(Error::MemoryAllocation)
}

/// Makes `vec` at least `len` long.
pub(crate) fn grow<T: Clone>(vec: &mut Vec<T>, value: T, len: usize) -> Result<()> {
    if let Some(more) = len.checked_sub(vec.len()) {
        vec.try_reserve_exact(more)?;
        vec.resize(len, value);
    }
    Ok(())
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

pub(crate) fn boxed<T>(value: T) -> Result<Box<T>> {
    Box::try_new(value).map_err(|_| Error::MemoryAllocation)
}

pub(crate) type Shared<T> = std::sync::Arc<T>;

pub(crate) fn shared<T>(value: T) -> Result<Shared<T>> {
    Shared::try_new(value).map_err(|_| Error::MemoryAllocation)
}
