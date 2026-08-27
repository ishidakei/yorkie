//! Driver-level session tests for the `NumaPolicy` option and the NUMA
//! information output (the driver-side half of the NumaPolicy work).
//!
//! These exercise the reference's two info lines (`engine.cpp`,
//! `usi.cpp`): `setoption name NumaPolicy` replies with BOTH the
//! `Available processors: ...` line and the `Using N thread[s]...` allocation
//! line; `setoption name Threads` replies with the allocation line only.
//!
//! `NumaPolicy none` is used so the allocation line never carries a binding
//! suffix — the assertions stay deterministic on any machine, single- or
//! multi-node.

mod common;

use std::sync::{Arc, Mutex};

use yorkie_protocol::UsiDriver;

fn drive(input: &str) -> String {
    let output = Arc::new(Mutex::new(Vec::<u8>::new()));
    let driver = UsiDriver::new(input.as_bytes(), Arc::clone(&output));
    driver.run().expect("driver run");
    let bytes = output.lock().expect("output lock").clone();
    String::from_utf8(bytes).expect("utf-8")
}

#[cfg_attr(miri, ignore)]
#[test]
fn numa_policy_none_emits_both_lines_threads_emits_allocation_only() {
    // `NumaPolicy none` first (both lines), then `Threads value 2` (allocation
    // line only). At the NumaPolicy step the thread count is still the default 4.
    let out = drive(
        "usi\n\
         setoption name NumaPolicy value none\n\
         setoption name Threads value 2\n\
         quit\n",
    );

    // NumaPolicy replies with the config line (exactly once — Threads does not
    // repeat it).
    let processor_lines = out
        .lines()
        .filter(|l| l.starts_with("info string Available processors:"))
        .count();
    assert_eq!(
        processor_lines, 1,
        "NumaPolicy emits the config line once; Threads does not: {out:?}"
    );

    // NumaPolicy's allocation line reflects the still-default 4 threads.
    assert!(
        out.contains("info string Using 4 threads\n"),
        "NumaPolicy emits the allocation line (4 threads): {out:?}"
    );
    // Threads value 2 emits its own allocation line.
    assert!(
        out.contains("info string Using 2 threads\n"),
        "Threads emits the allocation line (2 threads): {out:?}"
    );

    // `none` never binds → no allocation line carries the binding suffix.
    assert!(
        !out.contains("with NUMA node thread binding"),
        "none policy must not bind: {out:?}"
    );
}

#[cfg_attr(miri, ignore)]
#[test]
fn numa_policy_config_line_matches_available_processors_format() {
    // The config line begins with the exact reference prefix and a non-empty
    // processor list (its exact CPU set is machine-specific under `none`).
    let out = drive("setoption name NumaPolicy value none\nquit\n");
    let line = out
        .lines()
        .find(|l| l.starts_with("info string Available processors:"))
        .unwrap_or_else(|| panic!("no Available processors line: {out:?}"));
    let list = line
        .strip_prefix("info string Available processors: ")
        .expect("prefix");
    assert!(
        !list.is_empty(),
        "processor list must be non-empty: {line:?}"
    );
}

/// Turning the binding on must not move a single node.
///
/// A custom `NumaPolicy` node string always suggests binding, so this forces the
/// whole pin-and-place path on — thread pinning, the per-worker
/// `set_mempolicy(MPOL_PREFERRED)`, and the `mbind` migration of the
/// coordinator's history tables — on a machine where `auto` would never bind.
/// All of it is memory *placement*: the search reads the same tables, in the
/// same order, and must reproduce the `none` transcript byte for byte.
///
/// Single-threaded so the comparison is deterministic by construction (the
/// helper-vote path is not, across runs).
#[cfg_attr(miri, ignore)]
#[test]
fn forced_binding_reproduces_the_unbound_search_output() {
    let dir = common::TempDir::new("numa-parity");
    common::write_synthetic_nn_bin(dir.path());

    // A one-node custom config over a CPU the process is definitely allowed on:
    // `custom_affinity` short-circuits `suggests_binding_threads` to true, so
    // binding turns on even on a single-node runner, and the fail-loud
    // `sched_setaffinity` behind it can never see a forbidden CPU.
    let bound_policy = yorkie_numa::startup_affinity()
        .iter()
        .next()
        .copied()
        .unwrap_or(0)
        .to_string();

    let session = |policy: &str| {
        drive(&format!(
            "setoption name EvalDir value {}\n\
             setoption name Threads value 1\n\
             setoption name NumaPolicy value {policy}\n\
             isready\n\
             position startpos\n\
             go depth 4\n\
             quit\n",
            dir.path().display()
        ))
    };

    let bound = session(&bound_policy);
    let unbound = session("none");

    assert!(
        bound.contains("with NUMA node thread binding"),
        "the custom policy must actually bind: {bound:?}"
    );
    assert!(
        !unbound.contains("with NUMA node thread binding"),
        "`none` must not bind: {unbound:?}"
    );

    // The NUMA information lines differ (different configs); the search does not.
    let search_lines = |out: &str| {
        out.lines()
            .filter(|l| l.starts_with("info depth") || l.starts_with("bestmove"))
            .map(str::to_owned)
            .collect::<Vec<_>>()
    };
    let bound_search = search_lines(&bound);
    assert!(
        !bound_search.is_empty(),
        "the bound session must have searched: {bound:?}"
    );
    assert_eq!(
        bound_search,
        search_lines(&unbound),
        "placement-only change: the search transcript must be identical"
    );
}
