//! Filling the rows of tables, many at a time. That goes up to `N - 1` rows beyond those that are asked for.

use core::hint::black_box;

use crate::error::{Error, Result};

/// In bytes.
pub(super) const FILL: usize = 16;

/// Sets `count` rows from `start` on, `N` at a time.
#[inline(always)]
pub(super) fn fill_rows<T: Copy, const N: usize>(
    rows: &mut [T],
    start: usize,
    count: usize,
    value: T,
) -> Result<()> {
    let rows = (rows.get_mut(start..)).and_then(|rows| rows.get_mut(..count + N - 1));
    for chunk in rows.ok_or(Error::Internal)?.as_chunks_mut().0 {
        *chunk = [value; N];
        // Most are short: neither unrolling nor a call of `memset()` pays.
        black_box(());
    }
    Ok(())
}

/// Sets `repeats` rows for each of `symbols` to what `map` makes of it, from `start` on.
#[inline(always)]
pub(super) fn repeat_each<T: Copy, const N: usize>(
    rows: &mut [T],
    start: usize,
    symbols: &[u8],
    repeats: usize,
    map: impl Fn(u8) -> T,
) -> Result<()> {
    let mut rows = rows.get_mut(start..).ok_or(Error::Internal)?;
    if repeats <= N {
        for &symbol in symbols {
            *rows.first_chunk_mut().ok_or(Error::Internal)? = [map(symbol); N];
            rows = rows.get_mut(repeats..).ok_or(Error::Internal)?;
        }
    } else {
        for &symbol in symbols {
            fill_rows::<_, N>(rows, 0, repeats, map(symbol))?;
            rows = rows.get_mut(repeats..).ok_or(Error::Internal)?;
        }
    }
    Ok(())
}

/// Sets `count` rows from `to` on to what `map` makes of those from `from` on, `N` at a time.
#[inline(always)]
pub(super) fn copy_rows<T: Copy, const N: usize>(
    rows: &mut [T],
    [from, to]: [usize; 2],
    count: usize,
    map: impl Fn(T) -> T,
) -> Result<()> {
    for done in (0..count).step_by(N) {
        let from = rows.get(from + done..).and_then(<[T]>::first_chunk::<N>);
        let mut chunk = *from.ok_or(Error::Internal)?;
        chunk.iter_mut().for_each(|it| *it = map(*it));
        let to = rows.get_mut(to + done..).and_then(<[T]>::first_chunk_mut);
        *to.ok_or(Error::Internal)? = chunk;
    }
    Ok(())
}
