//! The parameters in the evaluation file are the ones in the source network.
//!
//! The build reads the source network once and lays its parameters out for the
//! kernels; the engine then reads those bytes and never sees the source again.
//! This holds the two against each other: the source file is read a second time,
//! by the reader this crate keeps for exactly that purpose, and the bytes it
//! produces must be the bytes the file holds — otherwise the engine would be
//! playing with a network nobody checked.
//!
//! Compiled only where the source reader is (`source-network`), and skipped
//! with a notice when either file is absent.

#![cfg(feature = "source-network")]

use std::io::{Read as _, Seek as _, SeekFrom};
use std::path::{Path, PathBuf};

use yorkie_eval::network_file;
use yorkie_eval::{NetDims, parameter_bytes};

mod common;

/// The source network this build converts, staged out-of-band.
fn source_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../eval/nn.bin")
}

/// The parameter region of the evaluation file at `path`.
fn parameter_region(path: &Path) -> std::io::Result<Vec<u8>> {
    let mut file = std::fs::File::open(path)?;
    file.seek(SeekFrom::Start(network_file::DATA_OFFSET as u64))?;
    let mut bytes = vec![0u8; network_file::DATA_BYTES];
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}

#[cfg_attr(miri, ignore)]
#[test]
fn the_evaluation_file_holds_the_source_networks_parameters() {
    let source = source_path();
    let engine_file = common::engine_network_path();
    if !source.is_file() || !engine_file.is_file() {
        eprintln!(
            "skipping the_evaluation_file_holds_the_source_networks_parameters: \
             the network is obtained out-of-band and is not staged here"
        );
        return;
    }

    let bytes = std::fs::read(&source).expect("read the source network");
    let converted =
        parameter_bytes(&bytes, &NetDims::STANDARD).expect("the source network is readable");
    let written = parameter_region(&engine_file).expect("read the evaluation file");

    assert_eq!(
        converted.len(),
        written.len(),
        "the parameter region's size is the layout's"
    );
    assert!(
        converted == written,
        "the evaluation file's parameters differ from the source network's"
    );
}
