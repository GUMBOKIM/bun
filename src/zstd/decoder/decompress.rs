//! Frames that are all there, into an output that has room for all of their content.

use crate::block::BlockDecoder;
use crate::dictionary::Dictionary;
use crate::error::{Error, Result};
use crate::format::BlockType;
use crate::frame::{BlockHeader, FrameHeader, HEADER_PREFIX_LEN, Header, skippable_len};
use crate::output::Output;
use crate::sequences::Destination;
use crate::xxhash::XxHash64;

/// `src` consists of any number of frames. An error takes back what it wrote.
pub fn decompress(
    src: &[u8],
    out: &mut Output<'_>,
    dictionary: Option<&Dictionary<'_>>,
) -> Result<()> {
    let written = out.written();
    decompress_frames(src, out, dictionary).inspect_err(|_| out.rewind(written))
}

fn decompress_frames(
    mut src: &[u8],
    out: &mut Output<'_>,
    dictionary: Option<&Dictionary<'_>>,
) -> Result<()> {
    let mut blocks = BlockDecoder::default();
    let mut is_first = true;
    while src.len() >= HEADER_PREFIX_LEN {
        if let Ok(Header::Skippable { content_len }) = Header::read(src) {
            src = (src.get(skippable_len(content_len)?..)).ok_or(Error::SrcSizeWrong)?;
            continue;
        }
        src = match decompress_frame(&mut blocks, src, out, dictionary) {
            Err(Error::PrefixUnknown) if !is_first => Err(Error::SrcSizeWrong),
            result => result,
        }?;
        is_first = false;
    }
    if !src.is_empty() {
        return Err(Error::SrcSizeWrong);
    }
    Ok(())
}

pub(crate) fn check_dictionary(
    header: &FrameHeader,
    dictionary: Option<&Dictionary<'_>>,
) -> Result<()> {
    if header.dictionary_id != 0 && header.dictionary_id != dictionary.map_or(0, Dictionary::id) {
        return Err(Error::DictionaryWrong);
    }
    Ok(())
}

pub(crate) fn check_content_size(header: &FrameHeader, size: u64) -> Result<()> {
    if header.content_size.is_some_and(|expected| expected != size) {
        return Err(Error::CorruptionDetected);
    }
    Ok(())
}

/// Returns what follows the frame.
pub(crate) fn decompress_frame<'s>(
    blocks: &mut BlockDecoder,
    src: &'s [u8],
    out: &mut Output<'_>,
    dictionary: Option<&Dictionary<'_>>,
) -> Result<&'s [u8]> {
    let header = FrameHeader::read_whole(src)?;
    check_dictionary(&header, dictionary)?;
    blocks.start_frame(dictionary);
    let mut src = src.get(header.len..).ok_or(Error::Internal)?;
    let mut hash = header.has_checksum.then(XxHash64::new);
    let history_start = out.written();
    loop {
        let (block, rest) = src.split_first_chunk().ok_or(Error::SrcSizeWrong)?;
        let block = BlockHeader::new(*block)?;
        let (content, rest) =
            (rest.split_at_checked(block.content_len())).ok_or(Error::SrcSizeWrong)?;
        src = rest;
        let destination = Destination {
            out,
            history_start,
            external: dictionary.map(Dictionary::history).unwrap_or_default(),
            block_len_max: header.block_len_max(),
        };
        let len = match block.block_type {
            BlockType::Compressed if content.len() > destination.block_len_max => {
                Err(Error::SrcSizeWrong)
            }
            BlockType::Compressed => blocks.decode(content, dictionary, destination),
            BlockType::Raw => destination.append(content),
            BlockType::Rle => destination.append_repeated(content, block.repeats()),
        }?;
        if let Some(hash) = &mut hash {
            hash.update(out.last_written(len));
        }
        if block.is_last {
            break;
        }
    }
    check_content_size(&header, (out.written() - history_start) as u64)?;
    if let Some(hash) = hash {
        let (checksum, rest) = src.split_first_chunk().ok_or(Error::ChecksumWrong)?;
        if u32::from_le_bytes(*checksum) != hash.digest() as u32 {
            return Err(Error::ChecksumWrong);
        }
        src = rest;
    }
    Ok(src)
}
