use bun_zstd_decoder::format::HUFFMAN_TABLE_LOG_MAX;
use bun_zstd_decoder::{Error, Result, with_bmi2};

use super::bit_writer::WordWriter;
use super::fse::{self, Normalized};
use super::histogram::Histogram;
use crate::array::At;
use crate::bytes::{highest_bit, put};

pub(super) const DESCRIPTION_LEN_MAX: usize = 128;
const CODE_LEN_MAX: u32 = 11;
const RANKS: usize = 16;

#[derive(Clone, Copy, Default)]
struct Node {
    count: u32,
    parent: u16,
    symbol: u8,
    bits: u8,
}

/// Node 0 is a barrier. The leaves follow, from the most frequent on, and the parents start at [`FIRST_PARENT`].
type Nodes = [Node; 512];
const FIRST_LEAF: usize = 1;
const FIRST_PARENT: usize = 257;

/// The code is in the highest bits, and its length in the lowest byte.
#[derive(Clone, Copy)]
struct Code(u64);

impl Code {
    const END_MARK: Self = Self::new(1, 1);

    const fn new(value: u16, bits: u8) -> Self {
        Self((value as u64).unbounded_shl(64 - bits as u32) | bits as u64)
    }

    #[inline(always)]
    fn bits(self) -> u32 {
        u32::from(self.0 as u8)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Table {
    codes: [Code; 256],
    log: u32,
    max_symbol: usize,
}

/// The order that this leaves equal counts in decides which of them get the shorter codes.
fn quicksort(mut nodes: &mut [Node]) {
    if nodes.len() <= 8 {
        for end in 1..nodes.len() {
            if let Some([sorted @ .., next]) = nodes.get(..=end) {
                let smaller = sorted.iter().rev().take_while(|it| it.count < next.count);
                let start = end - smaller.count();
                nodes
                    .get_mut(start..=end)
                    .unwrap_or_default()
                    .rotate_right(1);
            }
        }
        return;
    }
    while let [.., pivot] = *nodes
        && nodes.len() > 1
    {
        let last = nodes.len() - 1;
        let mut larger = 0;
        for index in 0..last {
            if nodes.get(index).is_some_and(|it| it.count > pivot.count) {
                nodes.swap(larger, index);
                larger += 1;
            }
        }
        nodes.swap(larger, last);
        let (before, rest) = nodes.split_at_mut(larger);
        let after = rest.get_mut(1..).unwrap_or_default();
        nodes = if before.len() < after.len() {
            quicksort(before);
            after
        } else {
            quicksort(after);
            before
        };
    }
}

fn sort_leaves(nodes: &mut Nodes, counts: &[u32]) {
    const LOG_BUCKETS_FROM: u32 = 158;
    const DISTINCT_BELOW: u32 = LOG_BUCKETS_FROM + LOG_BUCKETS_FROM.ilog2();
    const BUCKETS_END: usize = 191;
    let bucket = |count: u32| {
        if count < DISTINCT_BELOW {
            count as usize
        } else {
            (highest_bit(count) + LOG_BUCKETS_FROM) as usize
        }
    };

    // How many symbols are in this bucket or a later one.
    let mut from_here_on = [0u16; 256];
    for &count in counts {
        *from_here_on.at_mut(bucket(count)) += 1;
    }
    let mut sum = 0;
    for entry in from_here_on.iter_mut().take(BUCKETS_END).rev() {
        sum += *entry;
        *entry = sum;
    }

    let mut next = from_here_on;
    for (symbol, &count) in counts.iter().enumerate() {
        let position = next.at_mut(bucket(count) + 1);
        let node = nodes.at_mut(FIRST_LEAF + usize::from(*position));
        (node.count, node.symbol) = (count, symbol as u8);
        *position += 1;
    }

    let buckets = from_here_on.iter().zip(next).take(BUCKETS_END);
    for (&start, end) in buckets.skip(DISTINCT_BELOW as usize) {
        if end - start > 1 {
            let range = FIRST_LEAF + usize::from(start)..FIRST_LEAF + usize::from(end);
            quicksort(nodes.get_mut(range).unwrap_or_default());
        }
    }
}

/// Sets the bits of the leaves up to `last_leaf`, without a limit.
fn build_tree(nodes: &mut Nodes, last_leaf: usize) {
    let root = FIRST_PARENT + last_leaf - FIRST_LEAF - 1;
    for parent in FIRST_PARENT..=root {
        nodes.at_mut(parent).count = 1 << 30;
    }
    nodes.at_mut(0).count = 1 << 31;

    let (mut leaf, mut child) = (last_leaf, FIRST_PARENT);
    for parent in FIRST_PARENT..=root {
        let mut sum = 0u32;
        for _ in 0..2 {
            let smallest = if nodes.at(leaf).count < nodes.at(child).count {
                leaf -= 1;
                nodes.at_mut(leaf + 1)
            } else {
                child += 1;
                nodes.at_mut(child - 1)
            };
            smallest.parent = parent as u16;
            sum = sum.wrapping_add(smallest.count);
        }
        nodes.at_mut(parent).count = sum;
    }

    for node in (FIRST_PARENT..root).rev().chain(FIRST_LEAF..=last_leaf) {
        let parent = usize::from(nodes.at(node).parent);
        nodes.at_mut(node).bits = nodes.at(parent).bits + 1;
    }
}

/// Returns the length of the longest code.
fn limit_code_len(nodes: &mut Nodes, last_leaf: usize, limit: u32) -> u32 {
    const NONE: usize = 0;
    let bits = |nodes: &Nodes, node: usize| u32::from(nodes.at(node).bits);
    let longest = bits(nodes, last_leaf);
    if longest <= limit {
        return longest;
    }

    let mut debt = 0i32;
    let mut node = last_leaf;
    while bits(nodes, node) > limit {
        debt += (1 << (longest - limit)) - (1 << (longest - bits(nodes, node)));
        nodes.at_mut(node).bits = limit as u8;
        node -= 1;
    }
    while bits(nodes, node) == limit {
        node -= 1;
    }
    debt >>= longest - limit;

    // The rarest leaf whose code is that many bits shorter than the limit.
    let mut rarest = [NONE; RANKS];
    let mut current = limit;
    for leaf in (FIRST_LEAF..=node).rev() {
        if bits(nodes, leaf) < current {
            current = bits(nodes, leaf);
            *rarest.at_mut((limit - current) as usize) = leaf;
        }
    }

    while debt > 0 {
        let mut rank = highest_bit(debt as u32) as usize + 1;
        while rank > 1 {
            let (high, low) = (*rarest.at(rank), *rarest.at(rank - 1));
            if high != NONE && (low == NONE || nodes.at(high).count <= 2 * nodes.at(low).count) {
                break;
            }
            rank -= 1;
        }
        while rank <= HUFFMAN_TABLE_LOG_MAX as usize && *rarest.at(rank) == NONE {
            rank += 1;
        }
        debt -= 1 << (rank - 1);
        let leaf = *rarest.at(rank);
        nodes.at_mut(leaf).bits += 1;
        if *rarest.at(rank - 1) == NONE {
            *rarest.at_mut(rank - 1) = leaf;
        }
        let before = leaf.saturating_sub(1);
        let is_in_rank = bits(nodes, before) + rank as u32 == limit;
        *rarest.at_mut(rank) = if is_in_rank { before } else { NONE };
    }

    while debt < 0 {
        if *rarest.at(1) == NONE {
            while bits(nodes, node) == limit {
                node -= 1;
            }
            *rarest.at_mut(1) = node;
        }
        *rarest.at_mut(1) += 1;
        nodes.at_mut(*rarest.at(1)).bits -= 1;
        debt += 1;
    }
    limit
}

/// `None` if there is nothing to gain, or no room.
fn compress_weights(weights: &[u8], out: &mut [u8]) -> Result<Option<usize>> {
    let histogram = Histogram::<{ fse::SYMBOLS }>::of(weights);
    if histogram.largest_count == weights.len() || histogram.largest_count == 1 {
        return Ok(None);
    }
    let log = fse::table_log_for(6, weights.len(), histogram.max_symbol, 2);
    let normalized = Normalized::new(histogram.used(), weights.len(), log, false)?;
    let header_len = normalized.write(out)?;
    let out = out.get_mut(header_len..).unwrap_or_default();
    let mut table = fse::Tables::<64, 16>::EMPTY;
    let shape = table.build(0, &normalized);
    let len = table.compress(shape, weights, out);
    Ok(len.map(|len| header_len + len))
}

/// Bits enter at the top. Only the lowest byte of `count` counts.
#[derive(Clone, Copy, Default)]
struct Pending {
    bits: u64,
    count: u64,
}

impl Pending {
    /// Leaves the length of the code in the lowest 4 bits, which have to be free.
    #[inline(always)]
    fn add_if_roomy(&mut self, code: Code) {
        self.bits = self.bits.wrapping_shr(code.0 as u32) | code.0;
        self.count = self.count.wrapping_add(code.0);
    }

    #[inline(always)]
    fn add(&mut self, code: Code) {
        self.bits = self.bits.wrapping_shr(code.0 as u32) | (code.0 & !0xFF);
        self.count = self.count.wrapping_add(code.0);
    }

    #[inline(always)]
    fn append(&mut self, other: Self) {
        self.bits = self.bits.wrapping_shr(other.count as u32) | other.bits;
        self.count = self.count.wrapping_add(other.count);
    }

    #[inline(always)]
    fn flush(&mut self, writer: &mut WordWriter<'_>) {
        let count = u32::from(self.count as u8);
        writer.put(self.bits.wrapping_shr(count.wrapping_neg()), count / 8);
        self.count &= 7;
    }
}

impl Table {
    pub(super) const EMPTY: Self = Self {
        codes: [Code(0); 256],
        log: 0,
        max_symbol: 0,
    };

    /// Gives symbols the codes that follow from their lengths. `per_len` counts the codes of each length.
    fn assign_codes(&mut self, lens: &[u8], per_len: &[u16; RANKS], log: u32) {
        let mut next_value = [0u16; RANKS];
        let mut first = 0;
        for len in (1..=log as usize).rev() {
            *next_value.at_mut(len) = first;
            first = (first + per_len.at(len)) >> 1;
        }
        for (code, &len) in self.codes.iter_mut().zip(lens) {
            let value = next_value.at_mut(usize::from(len));
            *code = Code::new(*value, len);
            *value += 1;
        }
        (self.log, self.max_symbol) = (log, lens.len().saturating_sub(1));
    }

    /// `weights` are those of a valid description.
    pub(super) fn build_from_weights(&mut self, weights: &[u8], log: u32) {
        let (mut lens, mut per_len) = ([0; 256], [0; RANKS]);
        for (len, &weight) in lens.iter_mut().zip(weights) {
            if weight != 0 {
                *len = log as u8 + 1 - weight;
                *per_len.at_mut(usize::from(*len)) += 1;
            }
        }
        self.assign_codes(lens.get(..weights.len()).unwrap_or(&lens), &per_len, log);
    }

    /// `counts` end with a symbol that occurs, and have another one.
    fn build_limited(&mut self, counts: &[u32], limit: u32) {
        let mut nodes = [Node::default(); 512];
        sort_leaves(&mut nodes, counts);
        let last_leaf = counts.iter().filter(|&&count| count != 0).count();
        build_tree(&mut nodes, last_leaf);
        let log = limit_code_len(&mut nodes, last_leaf, limit);

        let (mut lens, mut per_len) = ([0; 256], [0; RANKS]);
        for leaf in nodes.iter().skip(FIRST_LEAF).take(last_leaf) {
            *lens.at_mut(usize::from(leaf.symbol)) = leaf.bits;
            *per_len.at_mut(usize::from(leaf.bits)) += 1;
        }
        self.assign_codes(lens.get(..counts.len()).unwrap_or(&lens), &per_len, log);
    }

    /// For `len` symbols, of which at least two differ.
    pub(super) fn build(&mut self, counts: &[u32], len: usize, tries_limits: bool) {
        let limit = if tries_limits {
            self.best_limit(counts)
        } else {
            fse::table_log_for(CODE_LEN_MAX, len, counts.len().saturating_sub(1), 1)
        };
        self.build_limited(counts, limit);
    }

    fn best_limit(&mut self, counts: &[u32]) -> u32 {
        let distinct = counts.iter().filter(|&&count| count != 0).count();
        let shortest = highest_bit(distinct as u32) + 1;
        let (mut best, mut best_len) = (CODE_LEN_MAX, usize::MAX - 1);
        for limit in shortest..=CODE_LEN_MAX {
            self.build_limited(counts, limit);
            if self.log < limit && limit > shortest {
                break;
            }
            let Ok(description_len) = self.describe(&mut [0; 2 * DESCRIPTION_LEN_MAX]) else {
                continue;
            };
            let len = self.compressed_len(counts) + description_len;
            if len > best_len + 1 {
                break;
            }
            if len < best_len {
                (best, best_len) = (limit, len);
            }
        }
        best
    }

    /// Returns the length of the description.
    pub(super) fn describe(&self, out: &mut [u8]) -> Result<usize> {
        let mut weights = [0; 256];
        // The last one follows from the others.
        let described = self.max_symbol;
        for (weight, code) in weights.iter_mut().zip(&self.codes).take(described) {
            if code.bits() != 0 {
                *weight = (self.log + 1 - code.bits()) as u8;
            }
        }

        let (header, out) = out.split_first_mut().ok_or(Error::DstSizeTooSmall)?;
        let used = weights.get(..described).unwrap_or_default();
        if let Some(len) = compress_weights(used, out)?
            && len < described / 2
        {
            *header = len as u8;
            return Ok(1 + len);
        }

        if described > 128 {
            return Err(Error::Internal);
        }
        let len = described.div_ceil(2);
        let out = out.get_mut(..len).ok_or(Error::DstSizeTooSmall)?;
        *header = (127 + described) as u8;
        for (byte, [high, low]) in out.iter_mut().zip(weights.as_chunks().0) {
            *byte = (high << 4) + low;
        }
        Ok(1 + len)
    }

    /// 0 if there is no code for it.
    pub(crate) fn code_len(&self, symbol: usize) -> u32 {
        if symbol > self.max_symbol {
            return 0;
        }
        self.codes.at(symbol).bits()
    }

    pub(super) fn compressed_len(&self, counts: &[u32]) -> usize {
        let pairs = counts.iter().zip(&self.codes);
        let bits: u32 = pairs.map(|(&count, code)| count * code.bits()).sum();
        (bits / 8) as usize
    }

    pub(super) fn has_codes_for(&self, counts: &[u32]) -> bool {
        let pairs = counts.iter().zip(&self.codes);
        let misses = pairs.fold(false, |misses, (&count, code)| {
            misses | ((count != 0) & (code.bits() == 0))
        });
        counts.len() <= self.max_symbol + 1 && !misses
    }

    /// `N` codes, 7 pending bits and, if `IS_ROOMY`, 4 more bits fit in 64.
    #[inline(always)]
    fn encode<const N: usize, const IS_ROOMY: bool>(
        &self,
        symbols: &[u8],
        writer: &mut WordWriter<'_>,
    ) -> Pending {
        let code = |symbol: u8| *self.codes.at(usize::from(symbol));
        let add_group = |pending: &mut Pending, group: &[u8; N]| {
            for (index, &symbol) in group.iter().enumerate().rev() {
                if index > 0 || IS_ROOMY {
                    pending.add_if_roomy(code(symbol));
                } else {
                    pending.add(code(symbol));
                }
            }
        };

        let mut pending = Pending::default();
        let (groups, rest) = symbols.as_chunks::<N>();
        if !rest.is_empty() {
            for &symbol in rest.iter().rev() {
                pending.add(code(symbol));
            }
            pending.flush(writer);
        }
        let (pairs, single) = groups.as_chunks::<2>();
        if let [group] = single {
            add_group(&mut pending, group);
            pending.flush(writer);
        }
        for [first, second] in pairs.iter().rev() {
            add_group(&mut pending, second);
            pending.flush(writer);
            // Does not wait for the bits above.
            let mut more = Pending::default();
            add_group(&mut more, first);
            pending.append(more);
            pending.flush(writer);
        }
        pending
    }

    #[inline(always)]
    fn compress_stream(&self, symbols: &[u8], out: &mut [u8]) -> Option<usize> {
        let mut writer = WordWriter::new(out)?;
        let mut pending = match self.log {
            12.. => self.encode::<4, false>(symbols, &mut writer),
            11 => self.encode::<5, false>(symbols, &mut writer),
            10 => self.encode::<5, true>(symbols, &mut writer),
            9 => self.encode::<6, false>(symbols, &mut writer),
            8 => self.encode::<7, false>(symbols, &mut writer),
            7 => self.encode::<8, false>(symbols, &mut writer),
            _ => self.encode::<9, true>(symbols, &mut writer),
        };
        pending.add(Code::END_MARK);
        pending.flush(&mut writer);
        writer.finish(pending.count > 0)
    }

    #[inline(never)]
    fn compress_1_stream(&self, symbols: &[u8], out: &mut [u8]) -> Option<usize> {
        with_bmi2(
            #[inline(always)]
            move || self.compress_stream(symbols, out),
        )
    }

    fn compress_4_streams(&self, symbols: &[u8], out: &mut [u8]) -> Option<usize> {
        const JUMP_TABLE_LEN: usize = 6;
        if out.len() < JUMP_TABLE_LEN + 3 + 8 || symbols.len() < 12 {
            return None;
        }
        let (jump_table, streams) = out.split_first_chunk_mut::<JUMP_TABLE_LEN>()?;
        let mut jumps = jump_table.as_chunks_mut().0.iter_mut();
        let mut position = 0;
        for segment in symbols.chunks(symbols.len().div_ceil(4)) {
            let len = self.compress_1_stream(segment, streams.get_mut(position..)?)?;
            let short_len = u16::try_from(len).ok()?;
            if let Some(jump) = jumps.next() {
                *jump = short_len.to_le_bytes();
            }
            position += len;
        }
        Some(JUMP_TABLE_LEN + position)
    }

    /// `None` if there is no room, or too little to compress.
    pub(super) fn compress(
        &self,
        symbols: &[u8],
        out: &mut [u8],
        is_single_stream: bool,
    ) -> Option<usize> {
        if is_single_stream {
            self.compress_1_stream(symbols, out)
        } else {
            self.compress_4_streams(symbols, out)
        }
    }
}

/// The description of a table that is yet to be written.
pub(crate) struct Description {
    bytes: [u8; DESCRIPTION_LEN_MAX],
    pub(super) len: usize,
}

impl Description {
    pub(super) const EMPTY: Self = Self {
        bytes: [0; DESCRIPTION_LEN_MAX],
        len: 0,
    };

    pub(super) fn of(table: &Table) -> Result<Self> {
        let mut description = Self::EMPTY;
        description.len = table.describe(&mut description.bytes)?;
        Ok(description)
    }

    pub(super) fn write(&self, out: &mut [u8]) -> Option<()> {
        put(out, self.bytes.get(..self.len)?)
    }
}
