use crate::array::At;

/// How often each of `N` symbols occurs.
#[derive(Clone, Copy)]
pub(crate) struct Histogram<const N: usize> {
    pub(crate) counts: [u32; N],
    /// The last symbol that occurs.
    pub(crate) max_symbol: usize,
    pub(crate) largest_count: usize,
}

impl<const N: usize> Histogram<N> {
    #[inline]
    pub(crate) fn of(symbols: &[u8]) -> Self {
        let counts = if symbols.len() < 1500 {
            let mut counts = [0; N];
            add(&mut counts, symbols);
            counts
        } else {
            count_interleaved(symbols)
        };
        Self::new(counts)
    }

    #[inline]
    pub(crate) fn new(counts: [u32; N]) -> Self {
        Self {
            max_symbol: counts.iter().rposition(|&count| count != 0).unwrap_or(0),
            largest_count: counts.iter().fold(0, |max, &count| count.max(max)) as usize,
            counts,
        }
    }

    /// Up to the last symbol that occurs.
    pub(crate) fn used(&self) -> &[u32] {
        self.counts.get(..=self.max_symbol).unwrap_or_default()
    }
}

#[inline(always)]
pub(crate) fn add<const N: usize>(counts: &mut [u32; N], symbols: &[u8]) {
    for &symbol in symbols {
        *counts.at_mut(usize::from(symbol)) += 1;
    }
}

/// With four tables, a symbol that repeats does not wait for its last count to be stored.
fn count_interleaved<const N: usize>(symbols: &[u8]) -> [u32; N] {
    let mut tables = [[0u32; N]; 4];
    let (words, rest) = symbols.as_chunks();
    for word in words {
        let word = u64::from_le_bytes(*word);
        for byte in 0..8 {
            let symbol = usize::from((word >> (8 * byte)) as u8);
            *tables.at_mut(byte % 4).at_mut(symbol) += 1;
        }
    }
    let [mut a, b, c, d] = tables;
    add(&mut a, rest);
    for (((a, b), c), d) in a.iter_mut().zip(b).zip(c).zip(d) {
        *a += b + c + d;
    }
    a
}
