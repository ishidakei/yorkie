//! The evaluation file's header: what it records, and what the engine does
//! with a file whose header says it was made for another build.
//!
//! The header is the whole of the check — the parameters behind it are hundreds
//! of mebibytes and are never re-hashed — so what it records, and what a
//! difference in it produces, is worth pinning on its own.

use std::io::Write as _;
use std::path::PathBuf;

use yorkie_eval::network_file::{self, Header, NetDims, Source};
use yorkie_eval::{NetworkParams, Region};

/// A header describing exactly the file this build reads.
fn matching_header() -> Header {
    Header {
        layout_version: network_file::LAYOUT_VERSION,
        source: network_file::SOURCE,
        target_features: network_file::TARGET_FEATURES.to_string(),
        dims: NetDims::STANDARD,
        data_bytes: network_file::DATA_BYTES as u64,
        net: yorkie_eval::NetHeader {
            version: 0x7AF3_2F16,
            hash: 0x3C20_3B32,
            arch_id: "SFNNwoP1536".to_string(),
        },
        warnings: vec!["Warning : nn.bin hash mismatch.".to_string()],
    }
}

/// Put the file at `path` into region 0 the way the engine does, and report the
/// complaints it carried.
///
/// One test executable is one process, and nothing else in it is reading the
/// region, so filling it here asks nothing of the caller.
fn place(path: &std::path::Path) -> Result<Vec<String>, yorkie_eval::NnueError> {
    // SAFETY: no search exists in a test executable that has not started one,
    // so nothing is reading the region being filled.
    unsafe {
        if network_file::SHARED_MAPPING {
            network_file::map_shared(path)
        } else {
            network_file::load_into_region(0, path)
        }
    }
}

/// A source identity that is not this build's, whichever this build has.
fn other_source() -> Source {
    match network_file::SOURCE {
        Source::Absent => Source::Sha256([7u8; 32]),
        Source::Sha256(_) => Source::Absent,
    }
}

/// Write `header` and a parameter region of zeros at `path`. The region is a
/// hole, so nothing of its size is actually written.
fn write_file(path: &std::path::Path, header: &Header) {
    let encoded = header.encode();
    let mut file = std::fs::File::create(path).expect("create the file");
    file.write_all(&encoded).expect("write the header");
    file.set_len((network_file::DATA_OFFSET + network_file::DATA_BYTES) as u64)
        .expect("size the parameter region");
}

fn temp_path(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("yorkie-eval-file-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create the temp directory");
    dir.join(format!("{tag}-{}", network_file::FILE_NAME))
}

#[cfg_attr(miri, ignore)]
#[test]
fn a_header_round_trips_through_its_bytes() {
    let header = matching_header();
    let decoded = Header::decode(&header.encode()).expect("the header decodes");
    assert_eq!(decoded, header);
}

#[cfg_attr(miri, ignore)]
#[test]
fn a_header_decodes_out_of_a_longer_prefix() {
    // What both the build and the engine actually read: the whole span before
    // the parameters, with the header at the front of it.
    let header = matching_header();
    let mut bytes = header.encode();
    bytes.resize(network_file::DATA_OFFSET, 0);
    assert_eq!(Header::decode(&bytes).expect("the header decodes"), header);
}

#[cfg_attr(miri, ignore)]
#[test]
fn a_truncated_header_is_refused_rather_than_read() {
    let bytes = matching_header().encode();
    let err = Header::decode(&bytes[..bytes.len() - 3]).expect_err("must refuse");
    assert!(err.contains("truncated"), "got: {err}");
}

#[cfg_attr(miri, ignore)]
#[test]
fn something_that_is_not_an_evaluation_file_is_refused() {
    let err = Header::decode(&[0u8; 512]).expect_err("must refuse");
    assert!(err.contains("magic"), "got: {err}");
}

#[cfg_attr(miri, ignore)]
#[test]
fn the_file_this_build_would_write_is_the_one_it_reads() {
    // The same comparison the build makes when it decides whether the file
    // already there has to be written again: matching source, layout and target
    // features means the bytes cannot differ, so it is left alone.
    let header = matching_header();
    assert_eq!(
        header.refusal(network_file::SOURCE, network_file::TARGET_FEATURES),
        None
    );
}

#[cfg_attr(miri, ignore)]
#[test]
fn a_file_made_from_another_network_is_refused_and_says_so() {
    let mut header = matching_header();
    header.source = other_source();
    let reason = header
        .refusal(network_file::SOURCE, network_file::TARGET_FEATURES)
        .expect("must refuse");
    assert!(reason.contains("made from"), "got: {reason}");
}

#[cfg_attr(miri, ignore)]
#[test]
fn a_file_made_for_other_target_features_is_refused() {
    let mut header = matching_header();
    header.target_features = format!("{},not-a-feature", network_file::TARGET_FEATURES);
    let reason = header
        .refusal(network_file::SOURCE, network_file::TARGET_FEATURES)
        .expect("must refuse");
    assert!(reason.contains("target features"), "got: {reason}");
}

#[cfg_attr(miri, ignore)]
#[test]
fn a_file_of_an_older_layout_is_refused() {
    let mut header = matching_header();
    header.layout_version = network_file::LAYOUT_VERSION.wrapping_sub(1);
    let reason = header
        .refusal(network_file::SOURCE, network_file::TARGET_FEATURES)
        .expect("must refuse");
    assert!(reason.contains("layout version"), "got: {reason}");
}

#[cfg_attr(miri, ignore)]
#[test]
fn a_file_of_other_dimensions_is_refused() {
    let mut header = matching_header();
    header.dims = NetDims {
        layer_stacks: 3,
        ..NetDims::STANDARD
    };
    let reason = header
        .refusal(network_file::SOURCE, network_file::TARGET_FEATURES)
        .expect("must refuse");
    assert!(reason.contains("dimensions"), "got: {reason}");
}

#[cfg_attr(miri, ignore)]
#[test]
fn a_matching_file_opens_and_carries_its_warnings_forward() {
    let path = temp_path("matching");
    write_file(&path, &matching_header());

    let header = network_file::read_header(&path).expect("the header is this build's");
    assert_eq!(header.warnings, matching_header().warnings);

    let warnings = place(&path).expect("the file opens");
    assert_eq!(warnings, matching_header().warnings);
    let net = Region::<0>::new();
    assert_eq!(
        net.ft_weights().len(),
        yorkie_eval::HIDDEN_SIZE * yorkie_eval::NUM_FEATURES
    );
    assert!(
        net.ft_weights().iter().all(|&w| w == 0),
        "the parameter region of this file is a hole, which reads as zeros"
    );
    let (addr, len) = net.parameter_region();
    assert_eq!(
        addr % (2 * 1024 * 1024),
        0,
        "the parameters start on a huge-page boundary"
    );
    assert_eq!(len, network_file::DATA_BYTES);

    let _ = std::fs::remove_file(&path);
}

#[cfg_attr(miri, ignore)]
#[test]
fn a_file_from_another_build_is_refused_when_it_is_opened() {
    let path = temp_path("foreign");
    let mut header = matching_header();
    header.source = other_source();
    write_file(&path, &header);

    let err = place(&path).expect_err("must refuse");
    assert!(
        format!("{err}").contains("not the one this build reads"),
        "got: {err}"
    );

    let _ = std::fs::remove_file(&path);
}

#[cfg_attr(miri, ignore)]
#[test]
fn an_absent_file_is_reported_as_the_missing_network_it_is() {
    let path = temp_path("absent");
    let _ = std::fs::remove_file(&path);
    let err = place(&path).expect_err("must fail");
    let message = format!("{err}");
    assert!(
        message.contains("failed to open NNUE file"),
        "got: {message}"
    );
}
