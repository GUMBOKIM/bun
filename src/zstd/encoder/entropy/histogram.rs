use crate::array::At;

/// How often each of `N` symbols occurs.
#[derive(Clone, Copy)]
pub(crate) struct Histogram<const N: usize> {
    pub(crate) counts: [u32; N],
    /// The last symbol that occurs.
    pub(crate) max_symbol: usize,
    pub(crate) largest_count: usize,
}

impl Histogram<256> {
    #[inline]
    pub(crate) fn of(symbols: &[u8]) -> Self {
        Self::new(match symbols.len() {
            ..1500 => {
                let mut counts = [0; 256];
                add(&mut counts, symbols);
                counts
            }
            1500..16_384 => count_interleaved::<1>(symbols),
            _ => count_interleaved::<2>(symbols),
        })
    }
}

impl<const N: usize> Histogram<N> {
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
pub(crate) fn add(counts: &mut [u32; 256], symbols: &[u8]) {
    for &symbol in symbols {
        *counts.at_mut(usize::from(symbol)) += 1;
    }
}

/// With several tables, a symbol that repeats does not wait for its last count to be stored.
#[inline(never)]
fn count_interleaved<const SETS: usize>(symbols: &[u8]) -> [u32; 256] {
    let mut sets = [[[0u32; 256]; 4]; SETS];
    let (stripes, rest) = symbols.as_chunks::<16>();
    for stripe in stripes {
        for (index, word) in stripe.as_chunks().0.iter().enumerate() {
            let word = u32::from_le_bytes(*word);
            for (byte, table) in sets.at_mut(index % SETS).iter_mut().enumerate() {
                *table.at_mut(usize::from((word >> (8 * byte)) as u8)) += 1;
            }
        }
    }
    let mut counts = [0; 256];
    add(&mut counts, rest);
    for table in sets.as_flattened() {
        for (count, more) in counts.iter_mut().zip(table) {
            *count += more;
        }
    }
    counts
}
