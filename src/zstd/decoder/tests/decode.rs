//! The fixtures are from zstd 1.5.7: its CLI, and `decodecorpus` for generated.zst. All their frames have a checksum.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use bun_zstd_decoder::{
    ContentSize, Decoder, Dictionary, Error, FrameLen, Output, Result, Status, TableDescription,
    XxHash64, decompress, decompressed_size, frame_content_size, frame_len, may_start_frame,
    read_huffman_weights, read_normalized_counts,
};

thread_local! {
    /// How many allocations are granted before one is refused.
    static GRANTS: Cell<Option<usize>> = const { Cell::new(None) };
}

struct Refusing;

// SAFETY: what is not refused is left to `System`.
unsafe impl GlobalAlloc for Refusing {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let grants = GRANTS.get();
        GRANTS.set(grants.and_then(|it| it.checked_sub(1)));
        if grants == Some(0) {
            return std::ptr::null_mut();
        }
        // SAFETY: the caller's promises are passed on.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: as above.
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Refusing = Refusing;

const DICTIONARY: &[u8] = include_bytes!("fixtures/dictionary");

struct Fixture {
    frames: &'static [u8],
    dictionary: &'static [u8],
    len: usize,
    hash: u64,
}

const FIXTURES: [Fixture; 4] = [
    Fixture {
        frames: include_bytes!("fixtures/generated.zst"),
        dictionary: &[],
        len: 122576,
        hash: 0x718B_275C_41EA_4A2A,
    },
    Fixture {
        frames: include_bytes!("fixtures/small-window.zst"),
        dictionary: &[],
        len: 40000,
        hash: 0x4328_9FE3_4E28_8052,
    },
    Fixture {
        frames: include_bytes!("fixtures/literals.zst"),
        dictionary: &[],
        len: 24000,
        hash: 0x1677_E591_A46B_0D8F,
    },
    Fixture {
        frames: include_bytes!("fixtures/with-dictionary.zst"),
        dictionary: DICTIONARY,
        len: 700,
        hash: 0xB6E5_C177_FEF5_5DA5,
    },
];

fn hex(digits: &str) -> Vec<u8> {
    let digits: Vec<_> = digits.chars().filter_map(|it| it.to_digit(16)).collect();
    let (pairs, _) = digits.as_chunks();
    pairs
        .iter()
        .map(|[high, low]| (high << 4 | low) as u8)
        .collect()
}

fn at_once(src: &[u8], room: usize) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(room);
    decompress(src, &mut Output::spare_capacity(&mut out, room), None)?;
    Ok(out)
}

/// `out` has room for all.
fn stream(
    decoder: &mut Decoder,
    mut src: &[u8],
    [input_step, output_step]: [usize; 2],
    out: &mut Vec<u8>,
) -> Result<()> {
    loop {
        let input = &src[..input_step.min(src.len())];
        let (consumed, status) = decoder.step(input, &mut Output::spare_capacity(out, output_step));
        src = &src[consumed..];
        if status? == Status::Complete && src.is_empty() {
            return Ok(());
        }
    }
}

fn byte_by_byte(src: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(1 << 18);
    stream(&mut Decoder::new(), src, [1, usize::MAX], &mut out)?;
    Ok(out)
}

#[test]
fn fixtures_decode_at_once() -> Result<()> {
    for fixture in FIXTURES {
        let dictionary = Dictionary::new(fixture.dictionary)?;
        for slack in [0, 1, 31, 32, 33] {
            let mut slice = vec![0xAA; fixture.len + slack];
            let mut out = Output::slice(&mut slice);
            decompress(fixture.frames, &mut out, Some(&dictionary))?;
            assert_eq!(out.written(), fixture.len);
            drop(out);
            assert_eq!(XxHash64::hash(&slice[..fixture.len]), fixture.hash);

            let mut vec = Vec::with_capacity(1 + fixture.len + slack);
            vec.push(7);
            let mut out = Output::spare_capacity(&mut vec, fixture.len);
            decompress(fixture.frames, &mut out, Some(&dictionary))?;
            drop(out);
            assert_eq!((vec[0], &vec[1..]), (7, &slice[..fixture.len]));
        }
        let mut short = vec![0; fixture.len - 1];
        let result = decompress(
            fixture.frames,
            &mut Output::slice(&mut short),
            Some(&dictionary),
        );
        assert_eq!(result, Err(Error::DstSizeTooSmall));
    }
    Ok(())
}

#[test]
fn fixtures_decode_in_pieces() -> Result<()> {
    let mut decoder = Decoder::new();
    for fixture in FIXTURES {
        decoder.load_dictionary(fixture.dictionary)?;
        for steps in [
            [1, 1 << 20],
            [1 << 20, 3],
            [7, 13],
            [1000, 999],
            [1 << 20, 1 << 20],
        ] {
            let mut out = Vec::with_capacity(fixture.len);
            stream(&mut decoder, fixture.frames, steps, &mut out)?;
            assert_eq!(
                (out.len(), XxHash64::hash(&out)),
                (fixture.len, fixture.hash)
            );
        }
    }
    Ok(())
}

#[test]
fn refused_memory_is_an_error() {
    for fixture in FIXTURES {
        for in_pieces in [false, true] {
            for grants in 0.. {
                let mut out = Vec::with_capacity(fixture.len);
                GRANTS.set(Some(grants));
                let result = if in_pieces {
                    let mut decoder = Decoder::new();
                    (decoder.load_dictionary(fixture.dictionary))
                        .and_then(|()| stream(&mut decoder, fixture.frames, [100, 100], &mut out))
                } else {
                    Dictionary::new(fixture.dictionary).and_then(|dictionary| {
                        let mut out = Output::spare_capacity(&mut out, usize::MAX);
                        decompress(fixture.frames, &mut out, Some(&dictionary))
                    })
                };
                if GRANTS.take().is_some() {
                    assert_eq!((result, out.len()), (Ok(()), fixture.len));
                    break;
                }
                assert_eq!(result, Err(Error::MemoryAllocation));
            }
        }
    }
}

#[test]
fn no_changed_byte_goes_unnoticed() -> Result<()> {
    let [.., fixture] = FIXTURES;
    let dictionary = Dictionary::new(fixture.dictionary)?;
    for index in 0..fixture.frames.len() {
        for bit in 0..8 {
            let mut frame = fixture.frames.to_vec();
            frame[index] ^= 1 << bit;
            let mut out = [0; 2000];
            let result = decompress(&frame, &mut Output::slice(&mut out), Some(&dictionary));
            assert!(result.is_err() || XxHash64::hash(&out[..fixture.len]) == fixture.hash);
        }
    }
    Ok(())
}

#[test]
fn unusual_frames_decode() {
    let frames = [
        ("28b52ffd 2000 010000", 0),
        // No literals and no sequences, and two bytes to say that there are no sequences.
        ("28b52ffd 0000 150000 0000", 0),
        ("28b52ffd 0000 250000 0861 8000", 1),
        ("28b52ffd a001000200 020010 00 0b0000 00", 131073),
        (
            "28b52ffd 2027 c50000 80f3f1f0ecebc6c5c7f09d43000000e0e0 06580100603e52",
            39,
        ),
        // A stream of literals with none in it.
        ("28b52ffd 0000 3d0000 02c000 8010 01 00", 0),
        ("28b52ffd c000 ffffffffffffffff 190000 616263", 3),
        ("502a4d18 00000000", 0),
        ("5f2a4d18 02000000 ffff 28b52ffd 2001 090000 61", 1),
    ];
    for (frame, len) in frames {
        let frame = hex(frame);
        assert_eq!(at_once(&frame, len).map(|out| out.len()), Ok(len));
        assert_eq!(byte_by_byte(&frame), at_once(&frame, len));
    }
}

#[test]
fn invalid_frames_fail_however_they_arrive() {
    let frames = [
        ("28b52ffe 2000 010000", Error::PrefixUnknown),
        ("28b52ffd 2800 010000", Error::FrameParameterUnsupported),
        ("28b52ffd 00b0 010000", Error::WindowTooLarge),
        ("28b52ffd 210700 010000", Error::DictionaryWrong),
        (
            "28b52ffd 2328b52ffd 2002 110000 6869",
            Error::DictionaryWrong,
        ),
        ("28b52ffd 2401 090000 61 00000000", Error::ChecksumWrong),
        ("28b52ffd 0000 070000", Error::CorruptionDetected),
        ("28b52ffd 0000 050000", Error::CorruptionDetected),
        (
            "28b52ffd 2005 100000 6869 010000",
            Error::CorruptionDetected,
        ),
        ("28b52ffd 0000 0b2000 78", Error::CorruptionDetected),
        ("28b52ffd 0000 250000 0861 00 00", Error::CorruptionDetected),
        (
            "28b52ffd 0000 2d0000 130000 01 00",
            Error::DictionaryCorrupted,
        ),
        (
            "28b52ffd 0000 2d0000 560000 01 00",
            Error::LiteralsHeaderWrong,
        ),
        (
            "28b52ffd 0000 450000 12c000 8010 07 00 00",
            Error::CorruptionDetected,
        ),
        (
            "28b52ffd 0000 3d0000 12c000 8010 07 00",
            Error::CorruptionDetected,
        ),
        (
            "28b52ffd 0000 550000 728001 04207e1f02aa 00",
            Error::CorruptionDetected,
        ),
        (
            "28b52ffd 0000 450000 0800 02 00 2f430bae",
            Error::CorruptionDetected,
        ),
        (
            "28b52ffd 0000 250000 00 01 fc 01",
            Error::CorruptionDetected,
        ),
        (
            "28b52ffd 0000 250000 00 01 01 01",
            Error::CorruptionDetected,
        ),
        ("28b52ffd 0000 150000 00 01", Error::SrcSizeWrong),
    ];
    for (frame, error) in frames {
        assert_eq!(at_once(&hex(frame), 4096), Err(error), "{frame}");
        assert_eq!(byte_by_byte(&hex(frame)), Err(error), "{frame}");
    }
}

#[test]
fn input_is_a_whole_number_of_frames() {
    let frame = hex("28b52ffd 2001 090000 61");
    assert_eq!(at_once(&[], 0), Ok(Vec::new()));
    assert_eq!(at_once(&frame.repeat(2), 2), Ok(b"aa".to_vec()));
    for len in 1..frame.len() {
        assert_eq!(at_once(&frame[..len], 9), Err(Error::SrcSizeWrong));
    }
    for extra in 1..12 {
        let input = [&frame[..], &[0; 12][..extra]].concat();
        assert_eq!(at_once(&input, 9), Err(Error::SrcSizeWrong));
    }
    assert_eq!(
        at_once(&hex("502a4d18 01000000"), 0),
        Err(Error::SrcSizeWrong)
    );
    let too_long = hex("502a4d18 f8ffffff");
    assert_eq!(at_once(&too_long, 0), Err(Error::FrameParameterUnsupported));
}

#[test]
fn sizes_are_read_from_headers() {
    let known = hex("28b52ffd 6400 00 030800 61 00000000");
    let unknown = hex("28b52ffd 0000 010000");
    let skippable = hex("502a4d18 01000000 00");
    let huge = hex("28b52ffd e0 0000000000000080 010000");
    assert_eq!(frame_content_size(&known[..7]), ContentSize::Known(256));
    assert_eq!(frame_content_size(&known[..6]), ContentSize::Invalid);
    assert_eq!(frame_content_size(&unknown), ContentSize::Unknown);
    assert_eq!(frame_content_size(&skippable), ContentSize::Known(0));

    let all = |frames: &[&[u8]]| decompressed_size(&frames.concat());
    assert_eq!(all(&[]), ContentSize::Known(0));
    assert_eq!(all(&[&known, &skippable, &known]), ContentSize::Known(512));
    assert_eq!(all(&[&known, &unknown]), ContentSize::Unknown);
    assert_eq!(all(&[&known, &[0]]), ContentSize::Invalid);
    assert_eq!(all(&[&known[..14]]), ContentSize::Invalid);
    assert_eq!(all(&[&huge, &huge]), ContentSize::Invalid);

    let lens = [0, 5, 7, 10, 14].map(|len| frame_len(&known[..len]));
    assert_eq!(
        lens,
        [5, 7, 10, 15, 15].map(|len| Ok(FrameLen::AtLeast(len)))
    );
    assert_eq!(frame_len(&known.repeat(2)), Ok(FrameLen::Complete(15)));
    assert_eq!(frame_len(&skippable[..8]), Ok(FrameLen::AtLeast(9)));
    assert_eq!(frame_len(&skippable), Ok(FrameLen::Complete(9)));
    assert_eq!(frame_len(b"\x28\xb5\x00"), Err(Error::PrefixUnknown));
}

fn step(decoder: &mut Decoder, input: &[u8], room: usize) -> (usize, usize, Result<Status>) {
    let mut out = vec![0; room];
    let mut out = Output::slice(&mut out);
    let (consumed, status) = decoder.step(input, &mut out);
    (consumed, out.written(), status)
}

#[test]
fn the_last_byte_is_consumed_when_all_output_is_out() {
    let frame = hex("28b52ffd 2003 190000 616263");
    let mut decoder = Decoder::new();
    assert_eq!(step(&mut decoder, &frame, 1), (11, 1, Ok(Status::Pending)));
    assert_eq!(
        step(&mut decoder, &frame[11..], 1),
        (0, 1, Ok(Status::Pending))
    );
    assert_eq!(step(&mut decoder, &[], 1), (0, 1, Ok(Status::Pending)));
    assert_eq!(
        step(&mut decoder, &frame[11..], 1),
        (1, 0, Ok(Status::Complete))
    );
}

#[test]
fn steps_that_get_nowhere_fail_in_the_end() {
    let frame = hex("28b52ffd 0000 180000 616263");
    let cases = [
        (0, Error::NoForwardProgressDestFull),
        (9, Error::NoForwardProgressInputEmpty),
    ];
    for (room, error) in cases {
        let mut decoder = Decoder::new();
        assert_eq!(step(&mut decoder, &frame, room).0, frame.len());
        for _ in 0..15 {
            assert_eq!(step(&mut decoder, &[], room), (0, 0, Ok(Status::Pending)));
        }
        assert_eq!(step(&mut decoder, &[], room), (0, 0, Err(error)));
    }
}

#[test]
fn an_error_undoes_the_step_and_stays() {
    let mut decoder = Decoder::new();
    let failed = (0, 0, Err(Error::CorruptionDetected));
    assert_eq!(
        step(&mut decoder, &hex("28b52ffd 0000 080000 61 070000"), 9),
        failed
    );
    let frame = hex("28b52ffd 2001 090000 61");
    assert_eq!(step(&mut decoder, &frame, 9), failed);
    assert_eq!(
        decoder.set_parameter(100, 10),
        Err(Error::CorruptionDetected)
    );
    decoder.reset();
    assert_eq!(step(&mut decoder, &frame, 9), (10, 1, Ok(Status::Complete)));
}

#[test]
fn an_error_undoes_a_call() {
    let mut vec = Vec::with_capacity(16);
    vec.extend(b"keep");
    let frame = hex("28b52ffd 2403 190000 616263 00000000");
    let result = decompress(&frame, &mut Output::spare_capacity(&mut vec, 9), None);
    assert_eq!(
        (result, &vec[..]),
        (Err(Error::ChecksumWrong), &b"keep"[..])
    );
}

#[test]
fn a_new_session_keeps_the_settings() -> Result<()> {
    let [.., fixture] = FIXTURES;
    let mut decoder = Decoder::new();
    decoder.load_dictionary(fixture.dictionary)?;
    decoder.set_parameter(100, 10)?;
    let larger = hex("28b52ffd 0001 010000");
    for _ in 0..2 {
        assert_eq!(step(&mut decoder, &larger, 0).2, Err(Error::WindowTooLarge));
        decoder.reset_session();
        assert_eq!(
            step(&mut decoder, fixture.frames, 700).2,
            Ok(Status::Complete)
        );
    }
    Ok(())
}

#[test]
fn earlier_frames_make_no_difference() {
    let mut decoder = Decoder::new();
    assert_eq!(
        step(&mut decoder, &hex("28b52ffd 0050 010000"), 0).2,
        Ok(Status::Complete)
    );
    let frame = hex("28b52ffd 8000 02000000 180000 616263 010000");
    for decoder in [&mut decoder, &mut Decoder::new()] {
        assert_eq!(
            step(decoder, &frame, 1),
            (0, 0, Err(Error::DstSizeTooSmall))
        );
    }
}

#[test]
fn magic_numbers_are_told_from_their_start() {
    for bytes in ["28", "28b52f", "28b52ffd00", "5f", "502a4d18"] {
        assert!(may_start_frame(&hex(bytes)));
    }
    for bytes in ["", "29", "28b52e", "28b52ffe", "60", "502a4d19"] {
        assert!(!may_start_frame(&hex(bytes)));
    }
}

#[test]
fn the_window_is_limited() {
    let mut decoder = Decoder::new();
    for (value, result) in [
        (9, Err(Error::ParameterOutOfBound)),
        (31, Ok(())),
        (0, Ok(())),
    ] {
        assert_eq!(decoder.set_parameter(100, value), result);
    }
    for value in [32, -1] {
        assert_eq!(
            decoder.set_parameter(100, value),
            Err(Error::ParameterOutOfBound)
        );
    }
    assert_eq!(
        decoder.set_parameter(1000, 0),
        Err(Error::ParameterUnsupported)
    );

    assert_eq!(decoder.set_parameter(100, 10), Ok(()));
    let complete = (9, 0, Ok(Status::Complete));
    assert_eq!(
        step(&mut decoder, &hex("28b52ffd 0000 010000"), 0),
        complete
    );
    let larger = hex("28b52ffd 0001 010000");
    assert_eq!(
        step(&mut decoder, &larger, 0),
        (0, 0, Err(Error::WindowTooLarge))
    );
    decoder.reset();
    assert_eq!(step(&mut decoder, &larger, 0), complete);
    let too_large = (0, 0, Err(Error::WindowTooLarge));
    assert_eq!(
        step(&mut decoder, &hex("28b52ffd 0089 010000"), 0),
        too_large
    );
}

#[test]
fn settings_change_between_frames() {
    let mut decoder = Decoder::new();
    assert_eq!(step(&mut decoder, &[0x28], 0), (1, 0, Ok(Status::Pending)));
    assert_eq!(decoder.set_parameter(100, 10), Err(Error::StageWrong));
    assert_eq!(decoder.load_dictionary(b"abc"), Err(Error::StageWrong));
}

#[test]
fn raw_dictionaries_are_history() -> Result<()> {
    // No literals, and 5 bytes from 2 back.
    let frame = hex("28b52ffd 0000 3d0000 00 01 54 000202 05");
    let mut out = [0; 5];
    decompress(
        &frame,
        &mut Output::slice(&mut out),
        Some(&Dictionary::raw(b"xyz")),
    )?;
    assert_eq!(&out, b"yzyzy");
    let short = Dictionary::new(b"z")?;
    let result = decompress(&frame, &mut Output::slice(&mut out), Some(&short));
    assert_eq!(result, Err(Error::CorruptionDetected));
    Ok(())
}

#[test]
fn damaged_dictionaries_are_refused() {
    for len in 8..120 {
        let result = Dictionary::new(&DICTIONARY[..len]).err();
        assert_eq!(result, Some(Error::DictionaryCorrupted));
        let result = Decoder::new().load_dictionary(&DICTIONARY[..len]);
        assert_eq!(result, Err(Error::DictionaryCorrupted));
    }
}

#[test]
fn descriptions_end_within_their_input() -> Result<()> {
    let mut counts = [9; 4];
    let description = read_normalized_counts(&hex("10 a3 0f"), &mut counts)?;
    let expected = TableDescription {
        len: 3,
        max_symbol: 3,
        table_log: 5,
    };
    assert_eq!((description, counts), (expected, [16, 0, 0, 16]));
    for (src, symbols) in [("10 a3", 4), ("10 a3 0f", 3), ("0b", 4), ("", 4)] {
        let result = read_normalized_counts(&hex(src), &mut counts[..symbols]);
        assert_eq!(result, Err(Error::CorruptionDetected));
    }

    let tree = read_huffman_weights(&hex("84 43 20 10 ff"))?;
    assert_eq!((tree.symbol_count, tree.table_log, tree.len), (6, 4, 4));
    assert_eq!(tree.weights[..6], [4, 3, 2, 0, 1, 1]);
    assert_eq!(tree.rank_counts[..6], [1, 2, 1, 1, 1, 0]);
    for src in ["", "84 43 20", "80 00", "80 20", "81 d1", "82 13 30", "01"] {
        assert_eq!(
            read_huffman_weights(&hex(src)).err(),
            Some(Error::CorruptionDetected)
        );
    }
    Ok(())
}
