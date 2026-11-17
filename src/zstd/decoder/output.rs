/// Where a call writes: a slice, or the spare capacity of a `Vec`. It never grows.
pub struct Output<'a> {
    target: Target<'a>,
    start: usize,
    position: usize,
    end: usize,
}

enum Target<'a> {
    Slice(&'a mut [u8]),
    /// Longer than `position` by what was lent, until the `Output` drops.
    Vec(&'a mut Vec<u8>),
}

impl<'a> Output<'a> {
    pub fn slice(slice: &'a mut [u8]) -> Self {
        Self {
            start: 0,
            position: 0,
            end: slice.len(),
            target: Target::Slice(slice),
        }
    }

    /// Appends at most `max` bytes.
    pub fn spare_capacity(vec: &'a mut Vec<u8>, max: usize) -> Self {
        Self {
            start: vec.len(),
            position: vec.len(),
            end: vec.len() + max.min(vec.capacity() - vec.len()),
            target: Target::Vec(vec),
        }
    }

    #[inline]
    pub fn written(&self) -> usize {
        self.position - self.start
    }

    #[inline]
    pub fn room(&self) -> usize {
        self.end - self.position
    }

    /// Writes as many as there is room for, and returns how many.
    pub fn append(&mut self, bytes: &[u8]) -> usize {
        let bytes = bytes.get(..self.room()).unwrap_or(bytes);
        let end = self.position + bytes.len();
        match &mut self.target {
            Target::Vec(vec) if vec.len() < end => {
                vec.truncate(self.position);
                vec.extend_from_slice(bytes);
            }
            Target::Vec(vec) => copy_to(vec, self.position, bytes),
            Target::Slice(slice) => copy_to(slice, self.position, bytes),
        }
        self.position = end;
        bytes.len()
    }

    /// What was written, and up to `len` bytes of the room behind it. These count once they are [`commit`](Self::commit)ted.
    pub fn lend(&mut self, len: usize) -> &mut [u8] {
        let end = self.position + len.min(self.room());
        let bytes = match &mut self.target {
            Target::Slice(slice) => &mut **slice,
            Target::Vec(vec) => {
                if vec.len() < end {
                    vec.resize(end, 0);
                }
                vec.as_mut_slice()
            }
        };
        bytes.get_mut(self.start..end).unwrap_or_default()
    }

    #[inline]
    pub fn commit(&mut self, len: usize) {
        debug_assert!(len <= self.room());
        self.position += len.min(self.room());
    }

    /// Forgets all but the first `written` bytes.
    pub fn rewind(&mut self, written: usize) {
        self.position = self.position.min(self.start.saturating_add(written));
    }
}

fn copy_to(target: &mut [u8], position: usize, bytes: &[u8]) {
    if let Some(target) = target.get_mut(position..position + bytes.len()) {
        target.copy_from_slice(bytes);
    }
}

impl Drop for Output<'_> {
    fn drop(&mut self) {
        if let Target::Vec(vec) = &mut self.target {
            vec.truncate(self.position);
        }
    }
}
