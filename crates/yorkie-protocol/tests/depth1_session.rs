//! Session-level depth-1 parity test against the real network.
//!
//! Drives a full USI session in-process — `usi` → `isready` →
//! `position startpos` → `go depth 1` — and asserts the emitted `bestmove`,
//! `score` and `nodes` match the reference-captured fixture as one inseparable
//! set: the `(nodes & 14)` root tie-break means a single-node drift can cascade
//! into a different score and a flipped bestmove.
//!
//! `EvalDir` is a compile-time constant, so the real network is reached by
//! entering a fixture root whose `<EvalDir>` links to it.
//!
//! The network is staged locally and never committed; when absent the test
//! prints a notice and passes. The fixture is a single-PV capture, so the test
//! also skips itself in a build whose compiled-in `MultiPV` is not 1.
//!
//! Gated on `usi-extras`: these sessions drive analysis-only `go` clauses, which
//! a default build refuses rather than reinterprets.
//!
//! Gated on `info-output` too, since the assertions read a search `info` line.

#![cfg(all(feature = "usi-extras", feature = "info-output"))]

mod common;

use std::path::PathBuf;

use common::stage_eval_dir_link;
use serde::Deserialize;
use yorkie_protocol::UsiDriver;

fn eval_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../eval")
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/search-depth1/startpos.json")
}

/// The gated subset of a depth-1 fixture (see `depth1_parity.rs`).
#[derive(Debug, Deserialize)]
struct Fixture {
    bestmove: String,
    score: ScoreJson,
    nodes: u64,
}

/// Fixture score: exactly one of `cp` or `mate` is present.
#[derive(Debug, Deserialize)]
struct ScoreJson {
    #[serde(default)]
    cp: Option<i32>,
    #[serde(default)]
    mate: Option<i32>,
}

fn drive(input: &str) -> String {
    let output = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
    let driver = UsiDriver::new(input.as_bytes(), std::sync::Arc::clone(&output));
    driver.run().expect("driver run");
    let bytes = output.lock().expect("output lock").clone();
    String::from_utf8(bytes).expect("utf-8")
}

/// Extract the value following `key` in a whitespace-tokenised `info` line, e.g.
/// `field_after("... nodes 30 pv ...", "nodes") == Some("30")`.
fn field_after<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let mut it = line.split_whitespace();
    while let Some(tok) = it.next() {
        if tok == key {
            return it.next();
        }
    }
    None
}

#[cfg_attr(miri, ignore)]
#[test]
fn depth1_session_matches_reference_startpos_fixture() {
    // The fixture's node count was captured on one worker; helpers sharing the
    // TT move it, so the comparison only means anything under the test config.
    common::require_test_config();

    // The fixture was captured from a single-PV root search. A `MultiPV N` build
    // searches N ranked root moves per iteration and reports the sum — a
    // different node count, and so a different tie-break for the bestmove that
    // count feeds — and there is no second fixture to compare that against.
    if yorkie_protocol::config::MULTI_PV != 1 {
        eprintln!(
            "skipped: this build compiled in MultiPV {}, and the fixture is a \
             single-PV capture",
            yorkie_protocol::config::MULTI_PV
        );
        return;
    }

    let dir = eval_dir();
    if !dir.join("nn.bin").exists() {
        eprintln!(
            "skipping depth1_session_matches_reference_startpos_fixture: {} is not present (obtained out-of-band)",
            dir.join("nn.bin").display()
        );
        return;
    }

    let raw = std::fs::read_to_string(fixture_path()).expect("read startpos fixture");
    let fixture: Fixture = serde_json::from_str(&raw).expect("parse startpos fixture");

    stage_eval_dir_link(&dir);
    let out = drive(
        "usi\n\
         isready\n\
         position startpos\n\
         go depth 1\n\
         quit\n",
    );

    assert!(
        out.contains("readyok\n"),
        "real network must load (readyok), got:\n{out}"
    );

    // bestmove (no ponder expected for the single-move startpos PV, but tolerate
    // one by comparing only the move token).
    let bestmove_line = out
        .lines()
        .find(|l| l.starts_with("bestmove "))
        .unwrap_or_else(|| panic!("missing bestmove in:\n{out}"));
    let got_best = bestmove_line
        .strip_prefix("bestmove ")
        .unwrap()
        .split_whitespace()
        .next()
        .expect("bestmove token");
    assert_eq!(got_best, fixture.bestmove, "bestmove mismatch in:\n{out}");

    // The single depth-1 info line carries the score and node count.
    let info_line = out
        .lines()
        .find(|l| l.starts_with("info depth 1 "))
        .unwrap_or_else(|| panic!("missing depth-1 info line in:\n{out}"));

    let got_nodes: u64 = field_after(info_line, "nodes")
        .expect("nodes field")
        .parse()
        .expect("nodes integer");
    assert_eq!(got_nodes, fixture.nodes, "node count mismatch in:\n{out}");

    // score is `cp <X>` or `mate <Y>`; compare against whichever the fixture has.
    let score_kind = field_after(info_line, "score").expect("score kind");
    let score_val: i32 = field_after(info_line, score_kind)
        .expect("score value")
        .parse()
        .expect("score integer");
    match (fixture.score.cp, fixture.score.mate) {
        (Some(cp), None) => {
            assert_eq!(score_kind, "cp", "expected a cp score in:\n{out}");
            assert_eq!(score_val, cp, "score cp mismatch in:\n{out}");
        }
        (None, Some(mate)) => {
            assert_eq!(score_kind, "mate", "expected a mate score in:\n{out}");
            assert_eq!(score_val, mate, "score mate mismatch in:\n{out}");
        }
        other => panic!("fixture score must be exactly one of cp/mate, got {other:?}"),
    }
}
