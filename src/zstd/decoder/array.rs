//! Indexing that cannot panic, and allocations that may be refused.

use crate::error::{Error, Result};

/// `table[index % N]`, which the compiler knows to be in bounds.
#[inline(always)]
#[expect(clippy::indexing_slicing, reason = "the index is less than N")]
pub(crate) const fn wrapped<T, const N: usize>(table: &[T; N], index: usize) -> &T {
    const { assert!(N.is_power_of_two()) };
    &table[index % N]
}

#[inline(always)]
#[expect(clippy::indexing_slicing, reason = "the index is less than N")]
pub(crate) const fn wrapped_mut<T, const N: usize>(table: &mut [T; N], index: usize) -> &mut T {
    const { assert!(N.is_power_of_two()) };
    &mut table[index % N]
}

/// Grows `vec` to `len` if it is shorter.
#[inline(always)]
pub(crate) fn fill_up<T: Copy>(vec: &mut Vec<T>, len: usize, value: T) -> Result<()> {
    if let Some(missing @ 1..) = len.checked_sub(vec.len()) {
        vec.try_reserve_exact(missing)?;
        vec.resize(len, value);
    }
    Ok(())
}

/// The only item of `vec`, which is `empty()` at first.
#[inline(always)]
pub(crate) fn only_item<T>(vec: &mut Vec<T>, empty: impl FnOnce() -> T) -> Result<&mut T> {
    /// Out of line, so that no other function has room for a `T` on its stack.
    #[cold]
    #[inline(never)]
    fn allocate<T>(vec: &mut Vec<T>, empty: impl FnOnce() -> T) -> Result<()> {
        vec.try_reserve_exact(1)?;
        vec.push(empty());
        Ok(())
    }

    if vec.is_empty() {
        allocate(vec, empty)?;
    }
    vec.first_mut().ok_or(Error::Internal)
}
