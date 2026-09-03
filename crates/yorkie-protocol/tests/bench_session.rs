//! Driver-level session tests for the `bench` command.
//!
//! The syntax tests run *without* a network: each position resigns instantly, so
//! they exercise the argument parse and summary plumbing without a real search.
//! The determinism and multi-thread tests stage a synthetic all-zero network and
//! run a small fixed-depth bench.
//!
//! Gated on `usi-extras`, so with the feature off this file compiles to nothing.
//! The complementary "feature off" assertions live in `src/parser.rs` and
//! `src/driver.rs`, compiled only when it is off.
//!
//! The `bench:` summary line is the command's result, not a diagnostic, so it is
//! asserted unconditionally; the argument-rejection lines are diagnostics and
//! are asserted only under `info-diag`.

#![cfg(feature = "usi-extras")]

mod common;

use common::{drive, stage_configured_eval_dir};

/// Extract the `nodes=` field from the single `bench:` summary line in `out`.
fn bench_summary_nodes(out: &str) -> u64 {
    let line = out
        .lines()
        .find(|l| l.contains("bench: positions="))
        .unwrap_or_else(|| panic!("no bench summary line in:\n{out}"));
    let field = line
        .split_whitespace()
        .find_map(|t| t.strip_prefix("nodes="))
        .unwrap_or_else(|| panic!("no nodes= field in bench summary: {line:?}"));
    field.parse().expect("nodes= is an integer")
}

/// The `positions=` field from the `bench:` summary line.
fn bench_summary_positions(out: &str) -> u64 {
    let line = out
        .lines()
        .find(|l| l.contains("bench: positions="))
        .unwrap_or_else(|| panic!("no bench summary line in:\n{out}"));
    line.split_whitespace()
        .find_map(|t| t.strip_prefix("positions="))
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| panic!("no positions= field in: {line:?}"))
}

// Syntax variants (hermetic — no network, so each position resigns).

#[cfg_attr(miri, ignore)]
#[test]
fn bare_bench_runs_the_four_default_positions() {
    // No network loaded → each of the four default positions resigns instantly,
    // and the summary reports positions=4, nodes=0. This proves the default
    // position list and the summary plumbing without a 60-second real search.
    let out = drive("bench\nquit\n");
    assert_eq!(
        bench_summary_positions(&out),
        4,
        "bare bench runs the four default positions:\n{out}"
    );
    assert_eq!(
        bench_summary_nodes(&out),
        0,
        "no network → zero nodes:\n{out}"
    );
    let bestmoves = common::bestmove_lines(&out);
    assert_eq!(
        bestmoves,
        vec!["resign"; 4],
        "one resign per position:\n{out}"
    );
}

/// An argument rejection is a diagnostic, so the line that names it is
/// `info-diag`. What "loudly" protects — that a bad argument runs no positions
/// rather than benching something else — is asserted in every build.
#[cfg_attr(miri, ignore)]
#[test]
fn garbage_argument_fails_loudly_without_panicking() {
    // A non-integer TT size is a loud parse error, not a panic and not a search.
    let out = drive("bench notanumber\nquit\n");
    if cfg!(feature = "info-diag") {
        assert!(
            out.contains("info string bench: invalid ttSizeMB"),
            "expected a loud parse error:\n{out}"
        );
    }
    assert!(
        !out.contains("bench: positions="),
        "a parse error runs no positions:\n{out}"
    );
}

#[cfg(feature = "info-diag")]
#[cfg_attr(miri, ignore)]
#[test]
fn unsupported_limit_type_fails_loudly() {
    // `perft` / `eval` are out of NPS-bench scope; they are reported, not run.
    let out = drive("bench 16 1 5 default perft\nquit\n");
    assert!(
        out.contains("info string bench: unsupported limit type `perft`"),
        "expected the scope-divergence notice:\n{out}"
    );
}

#[cfg_attr(miri, ignore)]
#[test]
fn current_source_benches_the_set_position() {
    // `current` benches exactly one position (the session position). With no
    // network it resigns, but the summary must report positions=1.
    let session = "position startpos moves 7g7f\n\
                   bench 16 1 4 current depth\n\
                   quit\n";
    let out = drive(session);
    assert_eq!(
        bench_summary_positions(&out),
        1,
        "current source = one position:\n{out}"
    );
}

// Determinism — identical total nodes across runs (threads=1).

/// The standard bench session against a staged synthetic network: a small
/// fixed-depth default bench so the test stays fast. Returns the full transcript.
fn bench_session(bench_line: &str) -> String {
    stage_configured_eval_dir();
    let input = format!(
        "usi\n\
         isready\n\
         {bench_line}\n\
         quit\n"
    );
    drive(&input)
}

#[cfg_attr(miri, ignore)]
#[test]
fn two_runs_in_one_process_report_identical_nodes() {
    stage_configured_eval_dir();

    // Two bench runs in ONE process (one network load). Each resets the TT and
    // histories at its start, so run 2 sees the same clean state as run 1.
    let input = "usi\n\
                 isready\n\
                 bench 16 1 3 default depth\n\
                 bench 16 1 3 default depth\n\
                 quit\n";
    let out = drive(input);
    let summaries: Vec<&str> = out
        .lines()
        .filter(|l| l.contains("bench: positions="))
        .collect();
    assert_eq!(summaries.len(), 2, "two bench summaries:\n{out}");

    let nodes0 = bench_summary_nodes(summaries[0]);
    let nodes1 = bench_summary_nodes(summaries[1]);
    assert!(
        nodes0 > 0,
        "the synthetic-network bench searched real nodes:\n{out}"
    );
    assert_eq!(
        nodes0, nodes1,
        "two in-process bench runs must report identical total nodes:\n{out}"
    );
}

#[cfg_attr(miri, ignore)]
#[test]
fn two_process_launches_report_identical_nodes() {
    // Two independent driver runs (separate "process launches") with identical
    // input must report the same total nodes — determinism does not depend on
    // in-process carry-over.
    let a = bench_session("bench 16 1 3 default depth");
    let b = bench_session("bench 16 1 3 default depth");
    let na = bench_summary_nodes(&a);
    let nb = bench_summary_nodes(&b);
    assert!(na > 0, "real search:\n{a}");
    assert_eq!(
        na, nb,
        "two process launches must agree on total nodes\nA:\n{a}\nB:\n{b}"
    );
}

// A threads=2 bench completes and reports a summary (no determinism).

#[cfg_attr(miri, ignore)]
#[test]
fn threads_two_bench_completes_and_reports() {
    let out = bench_session("bench 16 2 3 default depth");
    assert_eq!(
        bench_summary_positions(&out),
        4,
        "a threads=2 bench still runs all four default positions:\n{out}"
    );
    // No node-count assertion: multi-thread node totals vary run to run.
}
