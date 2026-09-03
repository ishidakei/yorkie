//! Driver-level session tests for the behaviour group: the compiled-in
//! `DrawValueBlack` / `DrawValueWhite` and `ResignValue`, and the `go mate`
//! limit.
//!
//! Each test drives a full `usi → isready → position → go` session in-process
//! against a synthetic all-zero network staged at the compiled-in `EvalDir`, so
//! they are hermetic. Searches run on a worker thread, so the harness waits for
//! the `bestmove` before comparing or quitting.
//!
//! The behaviour settings are compile-time constants, so each test asserts what
//! the value it was *built* with implies, branching on the constant, and stays
//! meaningful under either checked-in config.
//!
//! Gated on `usi-extras`: these sessions drive analysis-only `go` clauses, which
//! a default build refuses rather than reinterprets.
//!
//! Gated on `info-output` too, since the assertions read a search `info` line.

#![cfg(all(feature = "usi-extras", feature = "info-output"))]

mod common;

use common::{StreamHarness, bestmove_lines, stage_configured_eval_dir};
use yorkie_protocol::config;

/// Standard startpos, at a high game ply so a `MaxMovesToDraw` below it
/// adjudicates an immediate draw at every child node — the reported root score
/// then collapses to the draw contempt for the root side.
const STARTPOS_PLY_100_BLACK: &str =
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 100";
const STARTPOS_PLY_100_WHITE: &str =
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 100";

/// Black to move with a mate-in-1: `G*8a` mates the White king on 9a
/// (`limit_session.rs`'s `MATE_AT_PLY_100`, at ply 1). Used for `go mate`.
const MATE_IN_1_BLACK: &str = "k8/9/G1N6/9/9/9/9/9/8K b G 1";

/// Black to move, forced mated-in-2 by White: the Black king on 9i is boxed in
/// (every escape covered by the White gold on 9g / knight on 7g), and Black's
/// only legal move is the 5e pawn push; White then drops `g*8i` for mate. A real
/// search returns a decisive mated score, so `ResignValue` can fire.
const MATED_IN_2_BLACK: &str = "8k/9/9/9/4P4/9/g1n6/9/K8 b g 1";

/// The `cp` value of the last `info ... score cp N ...` line in `text`, if any.
fn last_score_cp(text: &str) -> Option<i64> {
    let mut found = None;
    for line in text.lines() {
        if let Some(rest) = line.split(" score cp ").nth(1)
            && let Some(tok) = rest.split_whitespace().next()
            && let Ok(v) = tok.parse::<i64>()
        {
            found = Some(v);
        }
    }
    found
}

/// Whether every child of a ply-100 root adjudicates a draw in this build, which
/// is what collapses the reported root score to the draw contempt.
fn draw_horizon_reached_at_ply_100() -> bool {
    let h = config::MAX_MOVES_TO_DRAW;
    h != 0 && h < 100
}

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

// DrawValueBlack / DrawValueWhite.

#[cfg_attr(miri, ignore)]
#[test]
fn the_configured_draw_contempt_shows_up_in_the_root_side_score() {
    // At ply 100 with the draw horizon behind us, every child adjudicates a
    // draw, so the root reports the root side's draw contempt: `value * Pawn /
    // 100`. A quiet default (`-2` ⇒ `-1 cp`) reports a small non-positive cp; a
    // loud one (`500` ⇒ `450`) reports around `+500 cp`. The other colour's value
    // never enters, which is what pins the per-side wiring.
    if !draw_horizon_reached_at_ply_100() {
        eprintln!("skipped: this build's MaxMovesToDraw is not below the fixture's ply");
        return;
    }
    let h = start_ready();

    for (position, own) in [
        (STARTPOS_PLY_100_BLACK, config::DRAW_VALUE_BLACK),
        (STARTPOS_PLY_100_WHITE, config::DRAW_VALUE_WHITE),
    ] {
        let before = h.output();
        h.send("usinewgame");
        go_and_wait(
            &h,
            &format!("position sfen {position}"),
            "go depth 2",
            bestmove_lines(&before).len() + 1,
        );
        let leg = h.output()[before.len()..].to_string();
        let cp = last_score_cp(&leg).expect("a cp score for an adjudicated-draw root");
        let want = own * 90 / 100;
        assert!(
            (cp - want).abs() <= 100,
            "root score {cp} must track the configured contempt {want} in:\n{leg}"
        );
    }
    h.quit_join();
}

// ResignValue.

#[cfg_attr(miri, ignore)]
#[test]
fn a_lost_position_is_resigned_exactly_when_the_configured_threshold_is_reachable() {
    // A forced mated-in-2 for the side to move. With a reachable `ResignValue`
    // the search's decisive negative score drops below `-value cp`, so the reply
    // is `bestmove resign`; with the unreachable default it plays its one legal
    // move instead.
    let h = start_ready();
    go_and_wait(
        &h,
        &format!("position sfen {MATED_IN_2_BLACK}"),
        "go depth 3",
        1,
    );
    let out = h.quit_join();
    let bm = bestmove_lines(&out)[0]
        .split_whitespace()
        .next()
        .unwrap()
        .to_string();

    if config::RESIGN_VALUE >= 99_999 {
        assert_ne!(
            bm, "resign",
            "an unreachable ResignValue must not resign a searchable position:\n{out}"
        );
        assert!(
            bm.starts_with("5e5d"),
            "the only legal move is 5e5d, got {bm} in:\n{out}"
        );
    } else {
        assert_eq!(
            bm,
            "resign",
            "ResignValue {} must resign a lost position:\n{out}",
            config::RESIGN_VALUE
        );
        // The final PV must precede `bestmove resign` (`yaneuraou-search.cpp`
        // in the reference: the PV-output condition includes `|| resign_by_value`), so
        // the GUI can see the score the resignation was decided on.
        let before_bestmove = out
            .split_once("bestmove")
            .map(|(head, _)| head)
            .unwrap_or(&out);
        assert!(
            before_bestmove
                .lines()
                .any(|l| l.starts_with("info ") && l.contains(" score ")),
            "resigning by value must print the deciding PV first:\n{out}"
        );
    }
}

// Go mate.

#[cfg_attr(miri, ignore)]
#[test]
fn go_mate_finds_the_mate_and_terminates_on_quiet() {
    // `go mate 5000` on a mate-in-1 replies the mating move within the budget
    // (the mate-found stop fires at depth 1); `go mate 2000` on a quiet position
    // terminates by the budget with a legal bestmove (no hang).
    let h = start_ready();

    // Mate-in-1: replies the mating drop, and quickly (well under the budget).
    go_and_wait(
        &h,
        &format!("position sfen {MATE_IN_1_BLACK}"),
        "go mate 5000",
        1,
    );
    let mate_out = h.output();
    assert_eq!(
        bestmove_lines(&mate_out)[0].split_whitespace().next(),
        Some("G*8a"),
        "go mate must play the mating drop:\n{mate_out}"
    );

    // Quiet position: no mate exists, so the search runs to whichever bound this
    // build gives it — the 2000 ms budget, or a compiled-in `DepthLimit` /
    // `NodesLimit` ceiling that is reached first — and still returns a legal
    // bestmove (not resign, not a hang). Which bound ends it does not matter
    // here; that it ends, with a playable move, does.
    h.send("usinewgame");
    go_and_wait(&h, "position startpos", "go mate 2000", 2);
    let end = h.quit_join();
    let quiet_leg = end[mate_out.len()..].to_string();
    let quiet_bm = bestmove_lines(&quiet_leg)[0]
        .split_whitespace()
        .next()
        .unwrap()
        .to_string();
    assert_ne!(quiet_bm, "resign", "quiet go mate must not resign");
    assert_ne!(quiet_bm, "win", "quiet go mate must not declare a win");
}

#[cfg_attr(miri, ignore)]
#[test]
fn go_mate_infinite_releases_on_stop() {
    // `go mate infinite` carries no time bound, but whether it is unbounded at
    // all is decided by the build: `go mate` names no depth / nodes token, so
    // `DepthLimit` / `NodesLimit` seed it like any other search, and unlike
    // `go infinite` it sets no `limits.infinite`, so it has no reply-holding
    // loop to sit in once a seeded ceiling is reached.
    let seeded = config::DEPTH_LIMIT != 0 || config::NODES_LIMIT != 0;
    let h = start_ready();
    h.send("position startpos");
    h.send("go mate infinite");

    if seeded {
        // The seeded ceiling terminates the search on its own, so the reply
        // arrives without a `stop`.
        assert!(
            h.wait_until(30_000, |o| bestmove_lines(o).len() == 1),
            "a go mate seeded with DepthLimit {} / NodesLimit {} must terminate by \
             itself:\n{}",
            config::DEPTH_LIMIT,
            config::NODES_LIMIT,
            h.output()
        );
        // A `stop` for a search that already finished is inert: no second reply,
        // no error line, and the session still quits.
        h.send("stop");
        std::thread::sleep(std::time::Duration::from_millis(80));
        let out = h.quit_join();
        assert_eq!(
            bestmove_lines(&out).len(),
            1,
            "the self-terminated go mate must reply exactly once in:\n{out}"
        );
        assert!(
            !out.contains("unknown command"),
            "a late stop must not be an unknown command in:\n{out}"
        );
        return;
    }

    // No ceiling: nothing bounds this search, so it emits no bestmove until
    // `stop`, which then releases a single reply promptly.
    std::thread::sleep(std::time::Duration::from_millis(80));
    assert!(
        !h.output().contains("bestmove"),
        "an unbounded go mate must not reply before stop:\n{}",
        h.output()
    );
    h.send("stop");
    assert!(
        h.wait_until(30_000, |o| bestmove_lines(o).len() == 1),
        "stop must release the go mate reply:\n{}",
        h.output()
    );
    h.quit_join();
}
