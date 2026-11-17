//! In a process of its own, as the first call decides on the dictionary.

use bun_icu_data as _;

unsafe extern "C" {
    fn Bun__ICUItem__decompress(
        item: *const u8,
        length: i32,
        dictionary: *const u8,
        dictionary_size: usize,
        out: *mut u8,
        size: usize,
    ) -> bool;
}

#[test]
fn the_dictionary_is_optional() {
    let item = b"\x28\xb5\x2f\xfd\x20\x03\x19\x00\x00abc";
    let mut out = [0; 3];
    // SAFETY: the frame is whole, and `out` is as long as it is said to be.
    let is_done = unsafe {
        Bun__ICUItem__decompress(item.as_ptr(), -1, std::ptr::null(), 0, out.as_mut_ptr(), 3)
    };
    assert_eq!((is_done, &out), (true, b"abc"));
}
