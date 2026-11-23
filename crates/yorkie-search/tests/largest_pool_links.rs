//! A build with the largest worker pool and the largest transposition table the
//! config schema accepts still links.
//!
//! Every worker's tables live in one static region whose size is the worker
//! count times one block, and the transposition table is a static array of the
//! configured size. Code reaches the table's start with a 32-bit PC-relative
//! `lea`, so a link that placed the worker region between the code and the
//! table would stop linking once the region passed 2 GiB — a failure no build
//! with a small pool shows. This test builds the crate's own unit-test binary
//! under a config with `threads` and `usi_hash` at the schema's upper bounds
//! and asserts that the link succeeds.
//!
//! The binary is linked, never run, so the test asks nothing of the machine's
//! CPU count or memory. The build goes into its own directory under cargo's
//! per-target scratch directory, so after the first run only a change to the
//! sources rebuilds it.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The schema's upper bound on `threads`.
const LARGEST_THREADS: u64 = 4096;

/// The schema's upper bound on `usi_hash`, in MiB: 32 TiB.
const LARGEST_USI_HASH: u64 = 33_554_432;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("the crate sits two levels below the workspace root")
        .to_path_buf()
}

/// The play config with the pool and the table at their largest, no worker
/// pinned and no ledger consulted, and the network directory made absolute so
/// the config can live outside the repository.
fn largest_config(root: &Path) -> String {
    let play = std::fs::read_to_string(root.join("configs/default.toml"))
        .expect("the play config is readable");
    let eval_dir = root.join("eval");
    let mut rewritten = 0;
    let lines: Vec<String> = play
        .lines()
        .map(|line| {
            let key = line.split(" = ").next().unwrap_or_default();
            let value = match key {
                "threads" => LARGEST_THREADS.to_string(),
                "usi_hash" => LARGEST_USI_HASH.to_string(),
                "cpu_assignment" => "\"none\"".to_string(),
                "cpu_ledger" => "\"\"".to_string(),
                "original_eval_dir" => format!("\"{}\"", eval_dir.display()),
                _ => return line.to_string(),
            };
            rewritten += 1;
            format!("{key} = {value}")
        })
        .collect();
    assert_eq!(
        rewritten, 5,
        "the play config names each rewritten key once"
    );
    lines.join("\n") + "\n"
}

#[cfg_attr(miri, ignore)]
#[test]
fn the_largest_pool_and_table_link() {
    let root = workspace_root();
    let scratch = Path::new(env!("CARGO_TARGET_TMPDIR")).join("largest-pool");
    std::fs::create_dir_all(&scratch).expect("the scratch directory can be made");
    let config = scratch.join("largest.toml");
    let contents = largest_config(&root);
    // Rewritten only when it changed, so an unchanged config does not make the
    // build scripts that read it run again.
    if std::fs::read_to_string(&config).ok().as_deref() != Some(contents.as_str()) {
        std::fs::write(&config, contents).expect("the scratch config can be written");
    }

    let output = Command::new(env!("CARGO"))
        .current_dir(&root)
        .args([
            "test",
            "-p",
            "yorkie-search",
            "--lib",
            "--no-run",
            "--offline",
        ])
        .arg("--target-dir")
        .arg(scratch.join("target"))
        .env("YORKIE_CONFIG", &config)
        .env("CARGO_TERM_COLOR", "never")
        .env_remove("CARGO_TARGET_DIR")
        .env_remove("CARGO_BUILD_TARGET_DIR")
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .output()
        .expect("cargo can be started");
    assert!(
        output.status.success(),
        "the unit-test binary with threads = {LARGEST_THREADS} and usi_hash = {LARGEST_USI_HASH} \
         did not build:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
