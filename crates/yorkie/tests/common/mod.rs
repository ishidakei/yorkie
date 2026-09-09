//! Shared helpers for the tests that spawn the built `yorkie` binary.
//!
//! A spawned engine reads the evaluation file the build laid out for it, beside
//! the binary, and no build has a runtime option surface to point it anywhere
//! else. So there is nothing for a test to stage: what it can ask is whether
//! that file is there at all, which it is exactly when a network was staged
//! when this build ran.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

/// The message a test that is pinned to the test config's values fails with when
/// the binary under test was built from another config.
const WRONG_CONFIG: &str = "this test requires the test config \
     — build with `YORKIE_CONFIG=configs/test.toml`";

/// Assert that this build compiled in the three values the suite's pinned
/// assertions were captured under (`configs/test.toml`: `usi_hash = 16`,
/// `threads = 1`, `pv_interval = 0`).
///
/// A test whose fixture bytes only hold under those values calls this first, so
/// a run that forgot `YORKIE_CONFIG` fails naming the fix rather than as an
/// unexplained node-count or transcript mismatch. It never skips: a suite that
/// quietly passed by running nothing would be worse than a red one.
pub fn require_test_config() {
    use yorkie_protocol::config;
    assert_eq!(config::USI_HASH, 16, "{WRONG_CONFIG}");
    assert_eq!(config::THREADS, 1, "{WRONG_CONFIG}");
    assert_eq!(config::PV_INTERVAL, 0, "{WRONG_CONFIG}");
}

/// The evaluation file a spawned engine reads: the compiled-in `eval_dir`,
/// resolved against the binary's own directory.
pub fn engine_network_path() -> PathBuf {
    let exe = PathBuf::from(env!("CARGO_BIN_EXE_yorkie"));
    let dir = exe
        .parent()
        .expect("the engine binary lives under <target>/<profile>");
    yorkie_eval::network_file::network_path(dir)
}

/// Whether this build had a network to convert, and so whether a spawned engine
/// can answer `readyok`.
///
/// The network is staged out-of-band and never committed, so a fresh checkout
/// has none; a test that needs one reports that and passes.
pub fn engine_has_network() -> bool {
    engine_network_path().is_file()
}

/// A scratch working directory for a spawned engine, under the workspace
/// `target/` directory. Derived from the test executable's own path
/// (`<target>/<profile>/deps/<name>-<hash>`) so it follows `CARGO_TARGET_DIR`
/// wherever it points.
///
/// Where the engine reads its network from no longer depends on this — it reads
/// what sits beside the binary — so this is only for the tests that watch what
/// the engine does, or does not do, with the files in its working directory.
pub fn engine_cwd() -> PathBuf {
    let exe = std::env::current_exe().expect("test executable path");
    let root = exe
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .expect("test executable lives under <target>/<profile>/deps")
        .join("yorkie-engine-cwd");
    std::fs::create_dir_all(&root).expect("create the engine working directory");
    root
}
