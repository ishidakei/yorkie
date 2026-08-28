//! Shared helpers for the tests that spawn the built `yorkie` binary.
//!
//! `EvalDir` is a compile-time constant — no build has a runtime option surface,
//! so a test cannot tell a spawned engine where its network is. It chooses the
//! engine's *working directory* instead: a fixture root under the workspace
//! `target/` directory whose `<EvalDir>` entry links to the real network
//! directory, which is exactly where the engine's relative `EvalDir` resolves.

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

/// Where the real (never-committed) SFNN-1536 network is staged.
pub fn eval_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../eval")
}

/// A working directory for a spawned engine whose compiled-in `EvalDir` resolves
/// to `src`. Idempotent and safe to call concurrently: every caller names the
/// same root and wants the same link target.
pub fn engine_cwd_with_eval_dir(src: &Path) -> PathBuf {
    let root = fixture_root();
    std::fs::create_dir_all(&root).expect("create fixture root");
    let link = root.join(yorkie_protocol::config::EVAL_DIR);
    let src = src.canonicalize().expect("real eval dir resolves");
    if std::fs::read_link(&link).ok().as_deref() != Some(src.as_path()) {
        // Losing the race against another test binary is fine: the winner made
        // the same link.
        let _ = std::os::unix::fs::symlink(&src, &link);
    }
    assert_eq!(
        std::fs::read_link(&link).ok().as_deref(),
        Some(src.as_path()),
        "the fixture root's EvalDir must link to the network directory"
    );
    root
}

/// `<workspace target>/yorkie-engine-cwd`, derived from the test executable's own
/// path (`<target>/<profile>/deps/<name>-<hash>`) so it follows
/// `CARGO_TARGET_DIR` wherever it points.
fn fixture_root() -> PathBuf {
    let exe = std::env::current_exe().expect("test executable path");
    exe.parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .expect("test executable lives under <target>/<profile>/deps")
        .join("yorkie-engine-cwd")
}
