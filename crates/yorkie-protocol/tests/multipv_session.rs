//! Driver-level session tests for the MultiPV + PV-output group: the real
//! MultiPV loop, the PV-output path, and `ConsiderationMode`.
//!
//! Each test drives a full `usi → isready → position → go` session in-process
//! against a synthetic (all-zero) network staged at the compiled-in `EvalDir`
//! (see [`common::stage_configured_eval_dir`]), so they are hermetic. They use
//! [`StreamHarness`] and wait for the `bestmove` before quitting: a MultiPV
//! search runs many root searches per iteration, so quitting early would abort
//! it mid-iteration (`quit` sets the stop flag).
//!
//! `MultiPV`, `PvInterval` and `ConsiderationMode` are compile-time constants, so
//! each test asserts what the value it was BUILT with implies and skips itself
//! when that value has nothing to show. `configs/test.toml` builds a
//! single-PV engine; `configs/test-limits.toml` builds `MultiPV 3` with
//! `ConsiderationMode` on. See `tests/limit_session.rs` for how to run the
//! second.
//!
//! **`usi-extras` gate.** These sessions drive the analysis-only `go` clauses
//! (`depth` / `nodes` / `movetime` / `infinite`), which a default build refuses
//! rather than reinterprets, so the whole file is gated on the feature and runs
//! under the `--all-features` gate. See the `usi-extras` reference
//! documentation.

//!
//! **`info-output` gate.** MultiPV is observed through the `info … multipv <i>`
//! lines, which only an `info-output` build emits, so the file is gated on that
//! feature too. `--all-features` carries both.

#![cfg(all(feature = "usi-extras", feature = "info-output"))]

mod common;

use common::{StreamHarness, bestmove_lines, legal, parse, stage_configured_eval_dir};
use yorkie_protocol::config;
use yorkie_state::parse_usi_move;

const STARTPOS: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
/// White to move with exactly one legal move: the king on 5a must capture the
/// checking gold on 5b (every other escape square is covered by that gold).
const ONE_LEGAL_MOVE: &str = "4k4/4G4/9/9/9/9/9/9/4K4 w - 1";

/// Stage the synthetic network, drive `position` + `go` on a stream harness, and
/// wait for the `bestmove` so the search fully completes before `quit`. Returns
/// the transcript.
fn run_session(position: &str, go: &str) -> String {
    stage_configured_eval_dir();
    let h = StreamHarness::start();
    h.send("usi");
    h.send("isready");
    assert!(
        h.wait_until(30_000, |o| o.contains("readyok")),
        "network must load and ack readyok"
    );
    h.send(position);
    h.send(go);
    assert!(
        h.wait_until(30_000, |o| bestmove_lines(o).len() == 1),
        "search `{go}` must finish (one bestmove):\n{}",
        h.output()
    );
    h.quit_join()
}

/// The `multipv` index of an `info` line, if present.
fn multipv_of(line: &str) -> Option<usize> {
    field_after(line, "multipv").and_then(|t| t.parse().ok())
}

/// The value following `key` in a whitespace-tokenised `info` line.
fn field_after<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let mut it = line.split_whitespace();
    while let Some(tok) = it.next() {
        if tok == key {
            return it.next();
        }
    }
    None
}

/// A sortable score key for a `score cp X` / `score mate Y` line: a mate for the
/// side to move outranks any cp, a mate against ranks below any cp.
fn score_key(line: &str) -> i64 {
    match field_after(line, "score") {
        Some("cp") => field_after(line, "cp")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0),
        Some("mate") => {
            let m: i64 = field_after(line, "mate")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            if m >= 0 {
                1_000_000 - m
            } else {
                -1_000_000 - m
            }
        }
        _ => 0,
    }
}

/// The first PV move token of an `info … pv m1 m2 …` line, if any.
fn first_pv_move(line: &str) -> Option<&str> {
    field_after(line, "pv")
}

/// Every `info … multipv <i>` line emitted for a single completed iteration —
/// the last contiguous run of `multipv 1..N` lines before the `bestmove`. This is
/// robust to the pin's `d = max(1, depth - 1)` relabel of an un-searched line: it
/// groups by the emitted block, not by the `depth` field.
fn last_multipv_block(out: &str) -> Vec<&str> {
    let lines: Vec<&str> = out.lines().collect();
    // Find the last `multipv 1` line, then take the contiguous multipv run from it.
    let start = lines
        .iter()
        .rposition(|l| multipv_of(l) == Some(1))
        .expect("at least one multipv 1 line");
    let mut block = Vec::new();
    for l in &lines[start..] {
        if multipv_of(l).is_some() {
            block.push(*l);
        } else {
            break;
        }
    }
    block
}

#[cfg_attr(miri, ignore)]
#[test]
fn the_configured_multipv_emits_that_many_ranked_lines_per_iteration() {
    let want = config::MULTI_PV as usize;
    if want < 2 {
        eprintln!("skipped: this build compiled in MultiPV 1, which emits no multipv index");
        return;
    }
    assert!(
        want <= legal(&parse(STARTPOS)).len(),
        "the fixture must have at least MultiPV legal moves"
    );

    let out = run_session("position startpos", "go depth 3");

    // The last completed iteration emits exactly `want` ranked lines.
    let block = last_multipv_block(&out);
    assert_eq!(
        block.len(),
        want,
        "a completed MultiPV={want} iteration emits exactly {want} lines in:\n{out}"
    );
    let idxs: Vec<usize> = block.iter().filter_map(|l| multipv_of(l)).collect();
    assert_eq!(
        idxs,
        (1..=want).collect::<Vec<_>>(),
        "multipv indices must be 1..={want} in:\n{out}"
    );

    // Scores are non-increasing by index.
    let scores: Vec<i64> = block.iter().map(|l| score_key(l)).collect();
    assert!(
        scores.windows(2).all(|w| w[0] >= w[1]),
        "scores must be non-increasing by multipv index, got {scores:?} in:\n{out}"
    );

    // Distinct first moves, one per line.
    let firsts: Vec<&str> = block.iter().filter_map(|l| first_pv_move(l)).collect();
    assert_eq!(firsts.len(), want, "each line has a pv in:\n{out}");
    let distinct: std::collections::HashSet<&str> = firsts.iter().copied().collect();
    assert_eq!(
        distinct.len(),
        want,
        "first moves must be distinct in:\n{out}"
    );

    // bestmove equals the multipv-1 line's first move.
    let bms = bestmove_lines(&out);
    assert_eq!(bms.len(), 1, "one bestmove in:\n{out}");
    let best = bms[0].split_whitespace().next().unwrap();
    assert_eq!(
        best, firsts[0],
        "bestmove must equal the multipv 1 first move in:\n{out}"
    );
}

#[cfg_attr(miri, ignore)]
#[test]
fn multipv_clamps_to_the_legal_move_count() {
    if config::MULTI_PV < 2 {
        eprintln!("skipped: this build compiled in MultiPV 1, which cannot exceed a move count");
        return;
    }
    let pos = parse(ONE_LEGAL_MOVE);
    let moves = legal(&pos);
    assert_eq!(moves.len(), 1, "fixture must have exactly one legal move");

    // MultiPV above the legal-move count clamps to the legal-move count.
    let out = run_session(&format!("position sfen {ONE_LEGAL_MOVE}"), "go depth 2");
    let block = last_multipv_block(&out);
    assert_eq!(
        block.len(),
        1,
        "a single-move position emits one line in:\n{out}"
    );
    assert_eq!(multipv_of(block[0]), Some(1));
    let bms = bestmove_lines(&out);
    assert_eq!(bms.len(), 1, "one bestmove in:\n{out}");
    let tok = bms[0].split_whitespace().next().unwrap();
    let mv = parse_usi_move(tok, &pos).expect("well-formed USI move");
    assert!(moves.contains(&mv), "{tok} is not the forced legal move");
}

#[cfg_attr(miri, ignore)]
#[test]
fn an_unthrottled_pv_interval_prints_every_iteration() {
    if config::PV_INTERVAL != 0 {
        eprintln!("skipped: this build throttles the PV, so which iterations print is timing");
        return;
    }
    let out = run_session("position startpos", "go depth 3");
    for d in 1..=3 {
        assert!(
            out.lines()
                .any(|l| l.starts_with(&format!("info depth {d} "))),
            "PvInterval 0 must emit a depth-{d} info line in:\n{out}"
        );
    }
}

#[cfg_attr(miri, ignore)]
#[test]
fn a_final_pv_always_precedes_bestmove() {
    // Whatever the throttle does to the intermediate lines, the final PV is
    // always emitted before `bestmove`, and its first move is the bestmove.
    //
    // Which line carries that PV depends on the compiled-in `MultiPV`: a
    // single-PV build emits one line per iteration, so the last one before
    // `bestmove` is it, but a `MultiPV N` iteration ends on its `multipv N`
    // line — the RANKED-FIRST line is `multipv 1`, and that is the one the
    // bestmove comes from. Take the last `multipv 1` line there, and the last
    // PV line outright where no index is emitted.
    let ranked = config::MULTI_PV >= 2;
    let out = run_session("position startpos", "go depth 3");

    let bm_pos = out
        .find("\nbestmove ")
        .map(|p| p + 1)
        .or_else(|| out.starts_with("bestmove ").then_some(0))
        .unwrap_or_else(|| panic!("missing bestmove in:\n{out}"));

    let pv_line = out[..bm_pos]
        .lines()
        .rev()
        .find(|l| {
            l.starts_with("info depth")
                && l.contains(" pv ")
                && (!ranked || multipv_of(l) == Some(1))
        })
        .unwrap_or_else(|| panic!("no info pv line before bestmove in:\n{out}"));

    let bms = bestmove_lines(&out);
    assert_eq!(bms.len(), 1, "one bestmove in:\n{out}");
    let best = bms[0].split_whitespace().next().unwrap();
    assert_eq!(
        first_pv_move(pv_line),
        Some(best),
        "final PV's first move must agree with bestmove in:\n{out}"
    );
}

#[cfg_attr(miri, ignore)]
#[test]
fn a_consideration_mode_pv_replays_as_a_legal_sequence() {
    if !config::CONSIDERATION_MODE {
        eprintln!("skipped: this build did not compile ConsiderationMode in");
        return;
    }

    // ConsiderationMode forces the interval to 0 internally, so per-iteration PVs
    // are emitted; the PV is collected from the transposition table.
    let out = run_session("position startpos", "go depth 4");

    let bms = bestmove_lines(&out);
    assert_eq!(
        bms.len(),
        1,
        "engine must exit cleanly with one bestmove in:\n{out}"
    );

    // The last info line's PV replays legally from the root.
    let pv_line = out
        .lines()
        .rfind(|l| l.starts_with("info depth") && l.contains(" pv "))
        .unwrap_or_else(|| panic!("no info pv line in:\n{out}"));
    let pv_str = pv_line.split(" pv ").nth(1).unwrap();

    let mut pos = parse(STARTPOS);
    let mut count = 0;
    for tok in pv_str.split_whitespace() {
        let mv = match parse_usi_move(tok, &pos) {
            Ok(m) => m,
            Err(_) => break, // a terminal marker or unparseable tail token ends the PV
        };
        assert!(
            legal(&pos).contains(&mv),
            "PV move {tok} (#{count}) is illegal from its position in:\n{out}"
        );
        pos.do_move(mv);
        count += 1;
    }
    assert!(
        count >= 1,
        "the consideration PV must have at least one move in:\n{out}"
    );
}
