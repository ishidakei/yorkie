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
use yorkie_eval::{NnueError, NnueNetwork};

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

/// The engine's network, or `None` when this build had none to convert.
pub fn engine_network() -> Option<NnueNetwork> {
    let path = engine_network_path();
    match network_file::open_shared(&path) {
        Ok((net, _warnings)) => Some(net),
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
