//! Decompresses the items of ICU's data, for the hooks that include `icu_data.h`.

use core::slice;
use std::sync::OnceLock;

use bun_zstd_decoder::{
    ContentSize, Dictionary, FrameLen, Output, decompress, frame_content_size, frame_len,
};

/// As much of the frame at `item` as it takes for `is_enough`, or all of it.
///
/// # Safety
/// `item` has `length` bytes if that is positive, or else starts with a whole zstd frame.
unsafe fn frame<'a>(
    item: *const u8,
    length: i32,
    is_enough: impl Fn(&[u8]) -> bool,
) -> Option<&'a [u8]> {
    let mut len = length.max(0) as usize;
    loop {
        // SAFETY: that is `length`, or the frame is at least as long.
        let prefix = unsafe { slice::from_raw_parts(item, len) };
        if is_enough(prefix) {
            return Some(prefix);
        }
        len = match frame_len(prefix).ok()? {
            FrameLen::Complete(len) => return prefix.get(..len),
            FrameLen::AtLeast(len) if length <= 0 => len,
            FrameLen::AtLeast(_) => return None,
        };
    }
}

/// -1 if it is not known.
///
/// # Safety
/// As for [`frame`].
#[unsafe(no_mangle)]
unsafe extern "C" fn Bun__ICUItem__size(item: *const u8, length: i32) -> i32 {
    let has_header = |prefix: &[u8]| frame_content_size(prefix) != ContentSize::Invalid;
    // SAFETY: passed on.
    match unsafe { frame(item, length, has_header) }.map(frame_content_size) {
        Some(ContentSize::Known(size)) => i32::try_from(size).unwrap_or(-1),
        _ => -1,
    }
}

/// # Safety
/// As for [`frame`]. `dictionary` has `dictionary_size` bytes for good, or is null. `out` has room for `size` bytes.
#[unsafe(no_mangle)]
unsafe extern "C" fn Bun__ICUItem__decompress(
    item: *const u8,
    length: i32,
    dictionary: *const u8,
    dictionary_size: usize,
    out: *mut u8,
    size: usize,
) -> bool {
    static DICTIONARY: OnceLock<Option<Dictionary<'static>>> = OnceLock::new();
    let dictionary = DICTIONARY.get_or_init(|| {
        if dictionary.is_null() {
            return None;
        }
        // SAFETY: passed on.
        Dictionary::new(unsafe { slice::from_raw_parts(dictionary, dictionary_size) }).ok()
    });
    // SAFETY: passed on. The bytes of `out` are initialized before they are a slice.
    let (frame, out) = unsafe {
        out.write_bytes(0, size);
        (
            frame(item, length, |_| false),
            slice::from_raw_parts_mut(out, size),
        )
    };
    frame.is_some_and(|frame| {
        decompress(frame, &mut Output::slice(out), dictionary.as_ref()).is_ok()
    })
}
