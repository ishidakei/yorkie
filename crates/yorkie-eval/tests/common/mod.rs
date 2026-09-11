//! The network the engine itself plays with, for the tests that need a real
//! one.
//!
//! The parameters are laid out for the kernels when the engine is built, into
//! the evaluation directory beside the binaries — which is where a test
//! executable finds them too, its own path being one directory below. A
//! checkout with no network staged has no such file, and a test that needs one
//! reports that and passes.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use yorkie_eval::network_file;
use yorkie_eval::{NnueError, Region};

/// Where this build wrote the evaluation file: the directory holding the
/// binaries it built, which is the test executable's parent's parent
/// (`<target>/<profile>/deps/<name>-<hash>`).
pub fn engine_network_path() -> PathBuf {
    let exe = std::env::current_exe().expect("test executable path");
    let binaries = exe
        .parent()
        .and_then(Path::parent)
        .expect("a test executable lives under <target>/<profile>/deps");
    network_file::network_path(binaries)
}

/// The engine's network, placed in region 0 the way the engine places it, or
/// `None` when this build had none to convert. The type is the region-backed
/// one the engine itself plays with.
///
/// One test executable is one process, and nothing else in it is reading the
/// region, so filling it here is the whole of what `isready` does with it.
pub fn engine_network() -> Option<Region<0>> {
    let path = engine_network_path();
    // SAFETY: no search exists in a test executable that has not started one,
    // so nothing is reading the region being filled.
    let placed = unsafe {
        if network_file::SHARED_MAPPING {
            network_file::map_shared(&path)
        } else {
            network_file::load_into_region(0, &path)
        }
    };
    match placed {
        Ok(_warnings) => Some(Region::new()),
        Err(NnueError::Io { .. }) => {
            eprintln!(
                "skipping: {} is absent — this build had no network to convert \
                 (it is obtained out-of-band)",
                path.display()
            );
            None
        }
        Err(e) => panic!("the evaluation file this build wrote must open: {e}"),
    }
}
