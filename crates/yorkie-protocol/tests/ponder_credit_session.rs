//! Driver-level session tests for what thinking ahead does to the clock: the
//! time the search after a reply spent on the reply the opponent then played is
//! counted, in `ponder_credit` percent, as already elapsed by the next clocked
//! `go`, which then answers sooner.
//!
//! The fixture forces both moves around that search. Black, the engine, has one
//! legal move, and after it White has one move in the search's root list — a
//! rook capture that promotes — so that move takes all of the search's effort,
//! while its non-promoting twin is legal but never searched at all.
//!
//! The clock is chosen so the two outcomes cannot overlap. Under
//! `go btime 3000 wtime 3000 binc 1000 winc 1000` an uncredited search spends
//! its minimum think time, 1880 ms (2000 ms less the 120 ms network delay), and
//! its whole budget is the same 1880 ms. A credit larger than the 1000 ms
//! between that budget and the first wall second's 880 ms ends the search at
//! its first check after one iteration, and the reply goes out at 880 ms. The
//! credited reply is asserted to come within three quarters of the uncredited
//! one's floor, 1410 ms: 530 ms above where it lands, and 470 ms below the
//! earliest an uncredited reply can come.
//!
//! Whether the engine thinks ahead, and how much of that time it credits, are
//! compile-time constants, so each test asserts what the values this build
//! carries imply. With either off, the assertion is that every reply comes no
//! sooner than the floor.
//!
//! Gated on `verbose1` for the two markers a search after a reply prints when
//! it starts and stops.
#![cfg(feature = "verbose1")]

mod common;

use std::time::{Duration, Instant};

use common::{StreamHarness, bestmove_lines, legal, parse, stage_configured_eval_dir};
use yorkie_protocol::config;
mod text_str;

use text_str::parse_usi_move;

/// Black to move and in check from the silver on 2d, whose only legal move is
/// to take it with the king: every other square is covered or blocked, no other
/// black piece reaches 2d, and a check from an adjacent piece cannot be blocked.
/// The king stepping off file 1 opens the lance on 1d onto the White king on
/// 1a, and White's only answer is the rook on 1h taking the lance, promoting or
/// not. Both sides keep a camp of pieces that touch none of this, so the search
/// after the reply has a real game to read.
const FORCED: &str = "lnsg2gpk/1b4s2/pppp3PK/7sL/9/9/PPPP5/1B6r/LNSG5 b - 1";
const ENGINE_MOVE: &str = "1c2d";
/// The reply the search after the engine's move searches: its only root move.
const SEARCHED_REPLY: &str = "1h1d+";
/// The same capture without promoting: legal, but not a move the root list
/// carries, so nothing is ever searched below it.
const UNSEARCHED_REPLY: &str = "1h1d";

const CLOCK: &str = "go btime 3000 wtime 3000 binc 1000 winc 1000";
/// The earliest an uncredited reply under [`CLOCK`] can come, in ms.
const FLOOR_MS: u128 = 1880;
/// The latest a credited reply under [`CLOCK`] may come, in ms: three quarters
/// of [`FLOOR_MS`].
const CREDITED_BOUND_MS: u128 = FLOOR_MS * 3 / 4;
/// How long the search after the engine's move is left to run, in real time.
/// At the effort a single root move takes, that is well over the 1000 ms
/// credit the early reply needs.
const PONDER_MS: u64 = 3000;

const PONDER_START: &str = "info string ponder start";
const PONDER_STOP: &str = "info string ponder stop";

/// Whether this build credits a searched reply with enough to answer at the
/// first wall second: it thinks ahead, and credits at least half of that time.
fn credits() -> bool {
    config::PONDER && config::PONDER_CREDIT >= 50
}

/// Whether this build's settings are the ones the bounds are computed from: the
/// time settings above, no depth or node ceiling that would end a clocked `go`
/// before the clock does, and a root list that leaves the unsearched reply out.
fn bounds_apply() -> bool {
    // Only a `verbose2` build has either ceiling.
    #[cfg(feature = "verbose2")]
    let (depth_limit, nodes_limit) = (config::DEPTH_LIMIT, config::NODES_LIMIT);
    #[cfg(not(feature = "verbose2"))]
    let (depth_limit, nodes_limit) = (0, 0);
    let settings = (
        config::NETWORK_DELAY,
        config::NETWORK_DELAY2,
        config::MINIMUM_THINKING_TIME,
        config::SLOW_MOVER,
        config::ROUND_UP_TO_FULL_SECOND,
        depth_limit,
        nodes_limit,
        config::GENERATE_ALL_LEGAL_MOVES,
    );
    if settings == (120, 1120, 2000, 100, true, 0, 0, false) {
        return true;
    }
    eprintln!("skipped: the bounds here assume other time, limit or move-generation settings");
    false
}

/// The fixture is what the module comment says it is.
fn check_fixture() {
    let mut pos = parse(FORCED);
    let mv = parse_usi_move(ENGINE_MOVE, &pos).expect("well-formed");
    assert_eq!(legal(&pos), vec![mv], "the engine has one legal move");
    pos.do_move(mv);
    let legal_replies = legal(&pos);
    assert_eq!(legal_replies.len(), 2, "the opponent has two answers");
    for reply in [SEARCHED_REPLY, UNSEARCHED_REPLY] {
        let mv = parse_usi_move(reply, &pos).expect("well-formed");
        assert!(legal_replies.contains(&mv), "{reply} is one of them");
    }
}

/// Start a session on the fixture, play the engine's forced move, and let the
/// search after it run for [`PONDER_MS`] if this build starts one. `None` when
/// the bounds do not apply to this build.
fn play_the_forced_move_and_think_ahead() -> Option<StreamHarness> {
    check_fixture();
    if !bounds_apply() {
        return None;
    }
    stage_configured_eval_dir();
    let h = StreamHarness::start();
    h.send("usi");
    h.send("isready");
    assert!(
        h.wait_until(30_000, |o| o.contains("readyok")),
        "network must load and ack readyok"
    );
    h.send("usinewgame");
    h.send(&format!("position sfen {FORCED}"));
    h.send(CLOCK);
    assert!(
        h.wait_until(30_000, |o| bestmove_lines(o).len() == 1),
        "the engine must answer:\n{}",
        h.output()
    );
    assert_eq!(bestmove_lines(&h.output())[0], ENGINE_MOVE);
    if config::PONDER {
        assert!(
            h.wait_until(2_000, |o| o.contains(PONDER_START)),
            "a build with ponder = true searches after its reply:\n{}",
            h.output()
        );
        std::thread::sleep(Duration::from_millis(PONDER_MS));
        assert!(
            !h.output().contains(PONDER_STOP),
            "the search after the reply must still be running when the reply \
             comes, or it ran for less than the time the bounds assume:\n{}",
            h.output()
        );
    }
    Some(h)
}

/// Send `commands`, the last of them the clocked `go`, and return how long the
/// second reply took to arrive from the first of them.
fn time_the_next_reply(h: &StreamHarness, commands: &[&str]) -> u128 {
    let sent = Instant::now();
    for command in commands {
        h.send(command);
    }
    assert!(
        h.wait_until(30_000, |o| bestmove_lines(o).len() == 2),
        "the engine must answer again:\n{}",
        h.output()
    );
    sent.elapsed().as_millis()
}

fn reply_position(reply: &str) -> String {
    format!("position sfen {FORCED} moves {ENGINE_MOVE} {reply}")
}

fn assert_uncredited(elapsed: u128, why: &str, h: StreamHarness) {
    let out = h.quit_join();
    assert!(
        elapsed >= FLOOR_MS,
        "{why}: the reply took {elapsed} ms, below the {FLOOR_MS} ms an \
         uncredited search spends:\n{out}"
    );
}

// 1. The reply the search after the engine's move searched: credited, so the
//    engine answers well before the uncredited floor.

#[cfg_attr(miri, ignore)]
#[test]
fn the_searched_reply_is_answered_sooner() {
    let _session = common::serial_session();
    let Some(h) = play_the_forced_move_and_think_ahead() else {
        return;
    };
    let elapsed = time_the_next_reply(&h, &[&reply_position(SEARCHED_REPLY), CLOCK]);
    if !credits() {
        if config::PONDER && config::PONDER_CREDIT > 0 {
            eprintln!(
                "skipped: ponder_credit = {} is too small a share for the bound here",
                config::PONDER_CREDIT
            );
            h.quit_join();
            return;
        }
        assert_uncredited(elapsed, "nothing is credited in this build", h);
        return;
    }
    let out = h.quit_join();
    assert!(
        out.contains(PONDER_STOP),
        "the position ends the search:\n{out}"
    );
    assert!(
        elapsed < CREDITED_BOUND_MS,
        "a credited reply must come within {CREDITED_BOUND_MS} ms, and took \
         {elapsed} ms:\n{out}"
    );
}

// 2. A reply the search never reached earns no credit: the engine answers as
//    though it had not thought ahead at all.

#[cfg_attr(miri, ignore)]
#[test]
fn a_reply_the_search_never_reached_is_not_credited() {
    let _session = common::serial_session();
    let Some(h) = play_the_forced_move_and_think_ahead() else {
        return;
    };
    let elapsed = time_the_next_reply(&h, &[&reply_position(UNSEARCHED_REPLY), CLOCK]);
    assert_uncredited(elapsed, "a reply with no effort below it", h);
}

// 3. A new game in between: the same position, reached again, is the new
//    game's and owes nothing to the last game's search.

#[cfg_attr(miri, ignore)]
#[test]
fn a_new_game_in_between_gets_no_credit() {
    let _session = common::serial_session();
    let Some(h) = play_the_forced_move_and_think_ahead() else {
        return;
    };
    let elapsed = time_the_next_reply(&h, &["usinewgame", &reply_position(SEARCHED_REPLY), CLOCK]);
    assert_uncredited(elapsed, "the position after `usinewgame`", h);
}
