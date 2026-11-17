//! Frames that are all there, into an output that has room for all of their content.

use crate::block::BlockDecoder;
use crate::destination::Destination;
use crate::dictionary::Dictionary;
use crate::error::{Error, Result};
use crate::format::BlockType;
use crate::frame::{BlockHeader, FrameHeader, Header, skippable_len};
use crate::output::Output;
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
    while !src.is_empty() {
        src = match Header::read_whole(src) {
            Ok(Header::Frame(header)) => {
                is_first = false;
                decompress_frame(&mut blocks, &header, src, out, dictionary)?
            }
            Ok(Header::Skippable { content_len }) => {
                (src.get(skippable_len(content_len)?..)).ok_or(Error::SrcSizeWrong)?
            }
            Ok(Header::Incomplete(_)) => return Err(Error::SrcSizeWrong),
            Err(Error::PrefixUnknown) if !is_first => return Err(Error::SrcSizeWrong),
            Err(error) => return Err(error),
        };
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

/// `src` starts with `header`. Returns what follows the frame.
pub(crate) fn decompress_frame<'s>(
    blocks: &mut BlockDecoder,
    header: &FrameHeader,
    src: &'s [u8],
    out: &mut Output<'_>,
    dictionary: Option<&Dictionary<'_>>,
) -> Result<&'s [u8]> {
    check_dictionary(header, dictionary)?;
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
        let history_len = out.written() - history_start;
        let external = dictionary.map(Dictionary::history).unwrap_or_default();
        let room = out.room();
        let destination = Destination::new(out, header, history_len, external, room);
        let len = match block.block_type() {
            BlockType::Compressed if content.len() > header.block_len_max() => {
                Err(Error::SrcSizeWrong)
            }
            BlockType::Compressed => blocks.decode(content, dictionary, destination),
            BlockType::Raw => destination.finish(0, content),
            BlockType::Rle => destination.finish_repeated(content, block.size()),
        }?;
        if let Some(hash) = &mut hash {
            hash.update(out.last_written(len));
        }
        header.check_content_size((out.written() - history_start) as u64, block.is_last())?;
        if block.is_last() {
            break;
        }
    }
    if let Some(hash) = hash {
        let (checksum, rest) = src.split_first_chunk().ok_or(Error::ChecksumWrong)?;
        if u32::from_le_bytes(*checksum) != hash.digest() as u32 {
            return Err(Error::ChecksumWrong);
        }
        src = rest;
    }
    Ok(src)
}
