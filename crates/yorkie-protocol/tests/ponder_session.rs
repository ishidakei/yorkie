//! Driver-level session tests for thinking ahead: once the engine has answered
//! a clocked `go`, it searches the position its own move reached until the next
//! command arrives, and that search never produces a second `bestmove`.
//!
//! Each test drives a full session in-process against a synthetic all-zero
//! network staged at the compiled-in `EvalDir`, so they are hermetic. The
//! search is watched through the two markers a `verbose1` build prints when one
//! starts and ends.
//!
//! Whether the engine thinks ahead at all is a compile-time constant, so each
//! test asserts what the value this build carries implies: with it off the
//! assertion is that nothing starts, which is the guarantee a measurement run
//! relies on.
//!
//! Gated on `verbose2`, which contains `verbose1`: the markers come from the
//! first and the fixed-depth `go` clauses from the second.
#![cfg(feature = "verbose2")]

mod common;

use std::time::Duration;

use common::{StreamHarness, bestmove_lines, legal, parse, stage_configured_eval_dir};
use yorkie_protocol::config;
mod text_str;

use text_str::parse_usi_move;

const STARTPOS: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

/// Black to move with a mate-in-1: `G*8a` mates the White king on 9a. After
/// that move the position has no legal reply at all.
const MATE_IN_1_BLACK: &str = "k8/9/G1N6/9/9/9/9/9/8K b G 1";

/// Black to move, forced mated-in-2 by White: Black's only legal move is the 5e
/// pawn push, and White then drops `g*8i` for mate. The position the engine's
/// move reaches is therefore a mate the search proves and then stops on.
const MATED_IN_2_BLACK: &str = "8k/9/9/9/4P4/9/g1n6/9/K8 b g 1";

/// Black to move and checkmated: no legal move, so the reply is `resign`.
const MATED_BLACK: &str = "4K4/3ggg3/4k4/9/9/9/9/9/9 b - 1";

/// A CSA-27-point-declarable position for the side to move, whose reply is the
/// bare `win` token.
const DECLARABLE_SFEN: &str = "+R+R+B+B5/3GKG3/2SGGGS2/9/9/9/9/9/4k4 b R 1";

/// A game clock small enough that the reply arrives promptly: the budget is
/// clamped to what is left after the network delay, which is the floor.
const GAME_CLOCK: &str = "go btime 1000 wtime 1000";

const PONDER_START: &str = "info string ponder start";
const PONDER_STOP: &str = "info string ponder stop";

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

/// Block until the engine has answered `n` times.
fn wait_for_replies(h: &StreamHarness, n: usize) {
    assert!(
        h.wait_until(30_000, |o| bestmove_lines(o).len() >= n),
        "expected {n} bestmove(s) in:\n{}",
        h.output()
    );
}

/// The USI move token of the `n`th `bestmove` line, which is the whole payload.
fn bestmove(h: &StreamHarness, n: usize) -> String {
    let out = h.output();
    bestmove_lines(&out)[n].to_string()
}

/// Whether a search announced itself, waiting long enough for one that was
/// going to start to have done so.
fn started_thinking_ahead(h: &StreamHarness) -> bool {
    h.wait_until(2_000, |o| o.contains(PONDER_START))
}

/// Assert the engine is thinking ahead, in a build that does that at all.
fn assert_thinking_ahead(h: &StreamHarness) {
    assert_eq!(
        started_thinking_ahead(h),
        config::PONDER,
        "a build with ponder = {} must {}start a search after its reply:\n{}",
        config::PONDER,
        if config::PONDER { "" } else { "not " },
        h.output()
    );
}

/// Assert no search was started after the reply, whatever this build's setting.
fn assert_no_search_after_the_reply(h: &StreamHarness, why: &str) {
    assert!(
        !started_thinking_ahead(h),
        "nothing may be searched after {why}:\n{}",
        h.output()
    );
}

// 1. A clocked `go`: the engine answers, keeps searching, and the next
//    `position` + `go` ends that search and produces exactly one more reply.

#[cfg_attr(miri, ignore)]
#[test]
fn a_clocked_go_is_followed_by_a_search_the_next_command_ends() {
    let _tt = common::serial_tt();
    let h = start_ready();
    h.send("position startpos");
    h.send(GAME_CLOCK);
    wait_for_replies(&h, 1);
    let first = bestmove(&h, 0);
    assert_thinking_ahead(&h);
    assert_eq!(
        bestmove_lines(&h.output()).len(),
        1,
        "the search after a reply never answers:\n{}",
        h.output()
    );

    h.send(&format!("position startpos moves {first}"));
    h.send(GAME_CLOCK);
    wait_for_replies(&h, 2);
    let out = h.quit_join();
    if config::PONDER {
        assert!(
            out.contains(PONDER_STOP),
            "the next command must end the search:\n{out}"
        );
    }
    assert_eq!(
        bestmove_lines(&out).len(),
        2,
        "one reply per `go`, and none from anything else:\n{out}"
    );
    // Each reply is legal in the position its own `go` was given.
    for (n, moves) in [(0usize, &[][..]), (1, &[first.as_str()])] {
        let mut pos = parse(STARTPOS);
        for m in moves {
            let mv = parse_usi_move(m, &pos).expect("legal setup move");
            pos.do_move(mv);
        }
        let token = bestmove_lines(&out)[n];
        let mv = parse_usi_move(token, &pos)
            .unwrap_or_else(|_| panic!("bestmove {token:?} is not a well-formed USI move"));
        assert!(legal(&pos).contains(&mv), "bestmove {token:?} is not legal");
    }
}

// 2. A search bounded by something other than the clock is an analysis, and
//    nothing follows it — which is what keeps a measurement run's output the
//    same whatever this setting says.

#[cfg_attr(miri, ignore)]
#[test]
fn a_fixed_depth_or_node_go_is_not_followed_by_a_search() {
    let _tt = common::serial_tt();
    let h = start_ready();
    h.send("position startpos");
    h.send("go depth 2");
    wait_for_replies(&h, 1);
    assert_no_search_after_the_reply(&h, "a `go depth`");

    h.send("position startpos");
    h.send("go nodes 500");
    wait_for_replies(&h, 2);
    assert_no_search_after_the_reply(&h, "a `go nodes`");
    assert_eq!(bestmove_lines(&h.quit_join()).len(), 2);
}

// 3. A search that ends by itself — here by proving the mate its own move walked
//    into — says so and emits nothing.
//
//    A build whose `ResignValue` is reachable resigns this position instead of
//    playing into the mate, and then there is nothing to search past at all.

#[cfg_attr(miri, ignore)]
#[test]
fn a_search_that_proves_a_mate_ends_by_itself_and_says_nothing() {
    let _tt = common::serial_tt();
    let h = start_ready();
    h.send(&format!("position sfen {MATED_IN_2_BLACK}"));
    h.send(GAME_CLOCK);
    wait_for_replies(&h, 1);
    if config::RESIGN_VALUE < 99_999 {
        assert_eq!(
            bestmove(&h, 0),
            "resign",
            "a reachable ResignValue resigns a lost position"
        );
        assert_no_search_after_the_reply(&h, "a resignation");
        h.quit_join();
        return;
    }
    assert_eq!(bestmove(&h, 0), "5e5d", "the only legal move is 5e5d");
    assert_thinking_ahead(&h);
    if config::PONDER {
        assert!(
            h.wait_until(30_000, |o| o.contains(PONDER_STOP)),
            "a proved mate ends the search without a command:\n{}",
            h.output()
        );
    }
    assert_eq!(
        bestmove_lines(&h.quit_join()).len(),
        1,
        "a search that ends by itself still answers nothing"
    );
}

// 4. `stop` ends it with nothing to say; `gameover` and `quit` end it too.

#[cfg_attr(miri, ignore)]
#[test]
fn stop_ends_the_search_after_a_reply_and_emits_nothing() {
    let _tt = common::serial_tt();
    let h = start_ready();
    h.send("position startpos");
    h.send(GAME_CLOCK);
    wait_for_replies(&h, 1);
    assert_thinking_ahead(&h);

    h.send("stop");
    if config::PONDER {
        assert!(
            h.wait_until(30_000, |o| o.contains(PONDER_STOP)),
            "`stop` must end the search:\n{}",
            h.output()
        );
    }
    // Nothing follows the stop: the same single reply the `go` produced.
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(
        bestmove_lines(&h.quit_join()).len(),
        1,
        "`stop` during a search that has already answered says nothing"
    );
}

#[cfg_attr(miri, ignore)]
#[test]
fn gameover_ends_the_search_after_a_reply() {
    let _tt = common::serial_tt();
    let h = start_ready();
    h.send("position startpos");
    h.send(GAME_CLOCK);
    wait_for_replies(&h, 1);
    assert_thinking_ahead(&h);

    h.send("gameover lose");
    if config::PONDER {
        assert!(
            h.wait_until(30_000, |o| o.contains(PONDER_STOP)),
            "`gameover` must end the search:\n{}",
            h.output()
        );
    }
    assert_eq!(bestmove_lines(&h.quit_join()).len(), 1);
}

/// `quit` ends it too: the session thread joins, which is what `quit_join`
/// waits for — a search left running would hang here rather than fail.
#[cfg_attr(miri, ignore)]
#[test]
fn quit_ends_the_search_after_a_reply() {
    let _tt = common::serial_tt();
    let h = start_ready();
    h.send("position startpos");
    h.send(GAME_CLOCK);
    wait_for_replies(&h, 1);
    assert_thinking_ahead(&h);
    assert_eq!(bestmove_lines(&h.quit_join()).len(), 1);
}

// 5. `isready` answers once the engine is idle again.

#[cfg_attr(miri, ignore)]
#[test]
fn isready_answers_after_the_search_has_ended() {
    let _tt = common::serial_tt();
    let h = start_ready();
    h.send("position startpos");
    h.send(GAME_CLOCK);
    wait_for_replies(&h, 1);
    assert_thinking_ahead(&h);

    h.send("isready");
    assert!(
        h.wait_until(30_000, |o| o.matches("readyok").count() == 2),
        "`isready` must be answered:\n{}",
        h.output()
    );
    let out = h.quit_join();
    if config::PONDER {
        let stop = out.find(PONDER_STOP).expect("the search ended");
        let second_readyok = out.rfind("readyok").expect("two readyok lines");
        assert!(
            stop < second_readyok,
            "the search must end before `readyok`:\n{out}"
        );
    }
    assert_eq!(bestmove_lines(&out).len(), 1);
}

// 6. Nothing is searched past a reply that ends the game, nor past a move that
//    leaves the opponent without one.

#[cfg_attr(miri, ignore)]
#[test]
fn a_resigned_position_is_not_searched_past() {
    let _tt = common::serial_tt();
    let h = start_ready();
    h.send(&format!("position sfen {MATED_BLACK}"));
    h.send(GAME_CLOCK);
    wait_for_replies(&h, 1);
    assert_eq!(bestmove(&h, 0), "resign");
    assert_no_search_after_the_reply(&h, "a resignation");
    h.quit_join();
}

#[cfg_attr(miri, ignore)]
#[test]
fn a_declared_win_is_not_searched_past() {
    let _tt = common::serial_tt();
    assert_eq!(
        config::ENTERING_KING_RULE,
        "CSARule27",
        "this fixture is 27-point declarable; another configured rule needs \
         another fixture"
    );
    let h = start_ready();
    h.send(&format!("position sfen {DECLARABLE_SFEN}"));
    h.send(GAME_CLOCK);
    wait_for_replies(&h, 1);
    assert_eq!(bestmove(&h, 0), "win");
    assert_no_search_after_the_reply(&h, "a declared win");
    h.quit_join();
}

#[cfg_attr(miri, ignore)]
#[test]
fn a_mating_move_is_not_searched_past() {
    let _tt = common::serial_tt();
    let h = start_ready();
    h.send(&format!("position sfen {MATE_IN_1_BLACK}"));
    h.send(GAME_CLOCK);
    wait_for_replies(&h, 1);
    let played = bestmove(&h, 0);

    let mut pos = parse(MATE_IN_1_BLACK);
    let mv = parse_usi_move(&played, &pos).expect("the reply is a USI move");
    pos.do_move(mv);
    assert!(
        legal(&pos).is_empty(),
        "this fixture is mate in 1, and {played} must deliver it"
    );
    assert_no_search_after_the_reply(&h, "a move that leaves no legal reply");
    h.quit_join();
}

// 7. The two commands a pondering GUI would send: a `go` carrying the `ponder`
//    token is the plain `go`, and `ponderhit` selects nothing.

#[cfg_attr(miri, ignore)]
#[test]
fn a_go_ponder_is_answered_like_a_plain_go_and_ponderhit_changes_nothing() {
    let _tt = common::serial_tt();
    let h = start_ready();
    h.send("position startpos");
    h.send("go ponder btime 1000 wtime 1000");
    // Not held: the token selects nothing, so the reply comes as it would for
    // the same `go` without it.
    wait_for_replies(&h, 1);
    assert_thinking_ahead(&h);

    h.send("ponderhit");
    std::thread::sleep(Duration::from_millis(100));
    let out = h.output();
    assert_eq!(
        bestmove_lines(&out).len(),
        1,
        "`ponderhit` answers nothing:\n{out}"
    );
    if config::PONDER {
        assert!(
            !out.contains(PONDER_STOP),
            "`ponderhit` does not end the search either:\n{out}"
        );
    }
    assert_eq!(bestmove_lines(&h.quit_join()).len(), 1);
}
