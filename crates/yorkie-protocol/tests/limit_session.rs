//! Driver-level session tests for the drive / limit group: the compiled-in
//! `DepthLimit` / `NodesLimit` / `MaxMovesToDraw` seeds, and the `gameover`
//! command.
//!
//! Each test drives a full `usi → isready → position → go` session in-process
//! against a synthetic (all-zero) network staged at the compiled-in `EvalDir`
//! (see [`common::stage_configured_eval_dir`]), so they are hermetic.
//! Multi-depth / node-capped / infinite searches run on a worker thread, so they
//! use [`StreamHarness`] and wait for the `bestmove` before quitting — a fixed
//! result never races the `quit`-driven join.
//!
//! # Reading a setting the build fixed
//!
//! These settings are compile-time constants; a session cannot change one. Each
//! test therefore asserts what the value it was BUILT with implies, branching on
//! the constant, so the same test body is meaningful under either checked-in
//! config: `configs/test.toml` leaves the three limits off, and
//! `configs/test-limits.toml` turns them on. Run the second with
//! `YORKIE_CONFIG=configs/test-limits.toml cargo nextest run -p yorkie-protocol
//! --all-features`.
//!
//! There is no session-level `USI_Hash` test here: the table is sized once from
//! the compiled-in constant and no session can resize it, so there are no two
//! legs to compare. `TranspositionTable`'s own tests cover the resize itself.
//!
//! **`usi-extras` gate.** These sessions drive the analysis-only `go` clauses
//! (`depth` / `nodes` / `movetime` / `infinite`), which a default build refuses
//! rather than reinterprets, so the whole file is gated on the feature and runs
//! under the `--all-features` gate. See the `usi-extras` reference
//! documentation.

#![cfg(feature = "usi-extras")]

mod common;

use common::{StreamHarness, bestmove_lines, legal, parse, stage_configured_eval_dir};
use yorkie_protocol::config;
use yorkie_state::parse_usi_move;

const STARTPOS: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

/// A gold-drop head mate for Black at a high game ply (`G*8a` mates the White
/// king on 9a, which is not itself in check at the root). Used to show the
/// `MaxMovesToDraw` horizon suppressing a mate.
const MATE_AT_PLY_100: &str = "k8/9/G1N6/9/9/9/9/9/8K b G 100";

/// Stage the synthetic network, start a session, and block until `readyok`.
fn start_ready() -> StreamHarness {
    stage_configured_eval_dir();
    let h = StreamHarness::start();
    h.send("usi");
    h.send("isready");
    assert!(
        h.wait_until(30_000, |o| o.contains("readyok")),
        "network must load and ack readyok"
    );
    h
}

/// One summary per completed search in `out`: each search's last `info depth …`
/// line joined with its `bestmove …` line, in order. Loading the 107 MiB
/// synthetic network dominates a session's cost, so tests that compare two
/// searches run both in ONE session (one load) and split the transcript here.
///
/// Which `info` line ends up last is only meaningful when the PV is not
/// throttled — under a non-zero `PvInterval` the per-iteration PV is gated on
/// the wall clock (`yorkie-search/src/qsearch.rs`, `pv_interval_elapsed`) and
/// the coordinator's end-of-search fallback reports a different depth, so two
/// searches over an identical node sequence can print different final lines.
/// Both checked-in test configs set `pv_interval = 0`; a comparison test skips
/// itself under a config that does not.
fn go_summaries(out: &str) -> Vec<String> {
    let mut res = Vec::new();
    let mut cur_info = "";
    for line in out.lines() {
        if line.starts_with("info depth") {
            cur_info = line;
        } else if line.starts_with("bestmove") {
            res.push(format!("{cur_info}\n{line}"));
            cur_info = "";
        }
    }
    res
}

/// Whether a transcript comparison is meaningful in this build.
fn pv_is_deterministic() -> bool {
    config::PV_INTERVAL == 0
}

/// Send `position` then `go`, and block until the transcript holds
/// `expect_bestmoves` total `bestmove` lines (i.e. this search has finished).
fn go_and_wait(h: &StreamHarness, position: &str, go: &str, expect_bestmoves: usize) {
    h.send(position);
    h.send(go);
    assert!(
        h.wait_until(30_000, |o| bestmove_lines(o).len() == expect_bestmoves),
        "search `{go}` must finish (expected {expect_bestmoves} bestmoves):\n{}",
        h.output()
    );
}

// -------------------------------------------------------------------------
// DepthLimit / NodesLimit.
// -------------------------------------------------------------------------

#[cfg_attr(miri, ignore)]
#[test]
fn depth_limit_caps_a_time_bounded_search_and_yields_to_an_explicit_depth() {
    let limit = config::DEPTH_LIMIT;
    if limit == 0 {
        eprintln!("skipped: this build compiled in no DepthLimit");
        return;
    }
    if !pv_is_deterministic() {
        eprintln!("skipped: PvInterval is non-zero, so transcripts are wall-clock dependent");
        return;
    }

    // A generous movetime, seeded by the compiled-in DepthLimit, must stop at the
    // limit — and produce exactly what a plain `go depth <limit>` produces. An
    // explicit `go depth <limit + 2>` in the same session then goes deeper,
    // proving an explicit token overwrites the seed.
    let h = start_ready();
    go_and_wait(&h, "position startpos", &format!("go depth {limit}"), 1);
    let plain_out = h.output();

    // `usinewgame` isolates each search (fresh histories + cleared table) so the
    // seeded run is comparable to the explicit one.
    h.send("usinewgame");
    go_and_wait(&h, "position startpos", "go movetime 5000", 2);
    let capped_out = h.output();
    let capped_block = capped_out[plain_out.len()..].to_string();
    let past = limit + 1;
    assert!(
        !capped_block
            .lines()
            .any(|l| l.starts_with(&format!("info depth {past}"))),
        "DepthLimit {limit} must stop there (no info depth {past}) in:\n{capped_block}"
    );

    h.send("usinewgame");
    let deeper = limit + 2;
    go_and_wait(&h, "position startpos", &format!("go depth {deeper}"), 3);
    let out = h.quit_join();
    let override_block = out[capped_out.len()..].to_string();
    assert!(
        override_block
            .lines()
            .any(|l| l.starts_with(&format!("info depth {deeper}"))),
        "an explicit go depth {deeper} must reach it despite DepthLimit {limit} in:\n{override_block}"
    );

    let s = go_summaries(&out);
    assert_eq!(s.len(), 3, "three searches in:\n{out}");
    assert_eq!(
        s[0], s[1],
        "DepthLimit {limit} must match plain go depth {limit}:\n{out}"
    );
}

#[cfg_attr(miri, ignore)]
#[test]
fn nodes_limit_matches_the_same_go_nodes() {
    let limit = config::NODES_LIMIT;
    if limit == 0 {
        eprintln!("skipped: this build compiled in no NodesLimit");
        return;
    }
    if !pv_is_deterministic() {
        eprintln!("skipped: PvInterval is non-zero, so transcripts are wall-clock dependent");
        return;
    }

    // The compiled-in NodesLimit caps a bare `go` exactly as an explicit
    // `go nodes <limit>` does: same final info line (nodes) and same bestmove.
    let h = start_ready();
    go_and_wait(&h, "position startpos", &format!("go nodes {limit}"), 1);
    // `usinewgame` isolates the two node-capped searches so they are comparable.
    h.send("usinewgame");
    go_and_wait(&h, "position startpos", "go", 2);
    let out = h.quit_join();

    let s = go_summaries(&out);
    assert_eq!(s.len(), 2, "two searches in:\n{out}");
    assert_eq!(
        s[0], s[1],
        "NodesLimit {limit} must behave exactly as go nodes {limit}:\n{out}"
    );
}

// -------------------------------------------------------------------------
// MaxMovesToDraw.
// -------------------------------------------------------------------------

#[cfg_attr(miri, ignore)]
#[test]
fn max_moves_to_draw_adjudicates_a_draw_past_the_horizon() {
    // A mate-in-1 position at game ply 100. Unlimited (`MaxMovesToDraw 0`) the
    // search finds the mate; a horizon below the game ply makes every interior /
    // qsearch node adjudicate a draw before the mate is seen, so the reported
    // score collapses from `mate` to a draw-band `cp` value and the search still
    // terminates with a bestmove. Which of the two this build shows is decided by
    // the constant.
    let horizon = config::MAX_MOVES_TO_DRAW;
    let h = start_ready();
    go_and_wait(
        &h,
        &format!("position sfen {MATE_AT_PLY_100}"),
        "go depth 2",
        1,
    );
    let out = h.quit_join();

    if horizon == 0 || horizon > 100 {
        assert!(
            out.lines().any(|l| l.contains("score mate")),
            "an unreached MaxMovesToDraw horizon must let the mate stand in:\n{out}"
        );
        assert_eq!(
            bestmove_lines(&out)[0].split_whitespace().next().unwrap(),
            "G*8a",
            "the unbounded search plays the mating drop in:\n{out}"
        );
    } else {
        assert!(
            !out.lines().any(|l| l.contains("score mate")),
            "MaxMovesToDraw {horizon} must suppress the mate in:\n{out}"
        );
        assert!(
            out.lines().any(|l| l.contains("score cp")),
            "the capped search must report a draw-adjudicated cp score in:\n{out}"
        );
        assert_eq!(
            bestmove_lines(&out).len(),
            1,
            "the capped search still terminates with one bestmove in:\n{out}"
        );
    }
}

// -------------------------------------------------------------------------
// Gameover (handled exactly like stop).
// -------------------------------------------------------------------------

#[cfg_attr(miri, ignore)]
#[test]
fn gameover_releases_infinite_search_and_a_fresh_go_works() {
    // `go infinite` then `gameover lose` releases the bestmove exactly as `stop`
    // would; afterwards `usinewgame` + a fresh `go` works normally.
    let h = start_ready();
    h.send("position startpos");
    h.send("go infinite");
    // An infinite search emits no bestmove until stopped (this engine reports a
    // single final `info`/`bestmove` on abort, no per-iteration progress lines),
    // so let the worker run briefly and confirm nothing has been emitted yet.
    std::thread::sleep(std::time::Duration::from_millis(80));
    assert!(
        !h.output().contains("bestmove"),
        "no bestmove until gameover:\n{}",
        h.output()
    );

    h.send("gameover lose");
    assert!(
        h.wait_until(30_000, |o| bestmove_lines(o).len() == 1),
        "gameover must release the bestmove:\n{}",
        h.output()
    );

    // A fresh game after gameover works.
    h.send("usinewgame");
    h.send("position startpos");
    h.send("go depth 2");
    assert!(
        h.wait_until(30_000, |o| bestmove_lines(o).len() == 2),
        "a fresh go after gameover must complete:\n{}",
        h.output()
    );
    let out = h.quit_join();
    let bms = bestmove_lines(&out);
    assert_eq!(bms.len(), 2, "two bestmoves total in:\n{out}");
    let start = parse(STARTPOS);
    let tok = bms[1].split_whitespace().next().unwrap();
    let mv = parse_usi_move(tok, &start).expect("well-formed USI move");
    assert!(
        legal(&start).contains(&mv),
        "{tok} is not a legal startpos move"
    );
}

#[cfg_attr(miri, ignore)]
#[test]
fn gameover_result_token_is_optional_and_ignored() {
    // A bare `gameover` (no win/lose/draw token) is accepted and releases the
    // held reply just like `gameover lose` / `stop`.
    let h = start_ready();
    h.send("position startpos");
    h.send("go infinite");
    std::thread::sleep(std::time::Duration::from_millis(80));
    assert!(
        !h.output().contains("bestmove"),
        "no bestmove until gameover:\n{}",
        h.output()
    );
    h.send("gameover");
    assert!(
        h.wait_until(30_000, |o| bestmove_lines(o).len() == 1),
        "bare gameover must release the bestmove:\n{}",
        h.output()
    );
    let out = h.quit_join();
    assert_eq!(bestmove_lines(&out).len(), 1, "one bestmove in:\n{out}");
    // The token itself is never echoed as an unknown command.
    assert!(
        !out.contains("unknown command"),
        "gameover must not be an unknown command in:\n{out}"
    );
}
