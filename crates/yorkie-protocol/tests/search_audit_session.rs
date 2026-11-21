//! USI sessions that probe inputs known to break other engines of the same
//! lineage, kept as tests that pin what this engine does with them today.
//!
//! Each test drives a full session in-process against the synthetic all-zero
//! network staged at the compiled-in `EvalDir`, so the sessions are hermetic.
//! Each test name says whether the pinned behaviour is the right one.
//!
//! - `go … searchmoves …`: an engine that matches the listed moves against a
//!   generator that leaves some legal non-promotions out can end up with an
//!   empty restriction, read that as "no restriction", and play a move outside
//!   the listed set. This engine has no `searchmoves` clause at all, so the line
//!   is refused whole and no search starts.
//! - `go depth 1` on a fresh session, and a depth-1 `bench`: an engine that
//!   reads the move to think about next from a table entry that holds no move
//!   can hand that missing move to its legality check. This engine names no
//!   such move in its reply.
//! - `setoption` with a non-numeric or out-of-range value: an engine that
//!   converts the value with a throwing parser ends the process. Every option
//!   of this engine is fixed when it is built, so the value is never read.

mod common;

use common::{bestmove_lines, diag_line, drive, stage_configured_eval_dir};

/// Black to move after `7g7f 3c3d`: the bishop exchange `8h2b` is legal
/// without promoting, a non-promotion the selective generator leaves out.
#[cfg(feature = "verbose2")]
const AFTER_7G7F_3C3D: &str = "position startpos moves 7g7f 3c3d\n";

/// A `go` that restricts the root to a legal non-promotion gets no reply at
/// all, rather than a reply outside the set it named: the clause is unknown,
/// so the whole line is dropped and nothing is searched. Right in that no move
/// outside the set is played; a host that relies on `searchmoves` is still
/// left waiting for a `bestmove` that never comes.
#[cfg(feature = "verbose2")]
#[cfg_attr(miri, ignore)]
#[test]
fn searchmoves_with_a_legal_non_promotion_is_refused_whole_today() {
    let _session = common::serial_session();
    stage_configured_eval_dir();
    let go = "go depth 3 searchmoves 8h2b";
    let out = drive(&format!(
        "usi\nisready\n{AFTER_7G7F_3C3D}{go}\nisready\nquit\n"
    ));
    assert_eq!(out.matches("readyok\n").count(), 2, "{out}");
    assert!(
        bestmove_lines(&out).is_empty(),
        "no search starts, so no move is played:\n{out}"
    );
    assert!(
        out.contains(&diag_line(&format!("unknown command: {go}"))),
        "{out}"
    );

    // The same position and depth without the clause is searched as usual.
    let out = drive(&format!(
        "usi\nisready\n{AFTER_7G7F_3C3D}go depth 3\nquit\n"
    ));
    assert_eq!(bestmove_lines(&out).len(), 1, "{out}");
}

/// The same refusal on a clock-bounded `go`, the only kind the tournament
/// build accepts: the line is unknown in every build.
#[cfg_attr(miri, ignore)]
#[test]
fn searchmoves_on_a_clock_go_is_refused_whole_in_every_build_today() {
    let _session = common::serial_session();
    stage_configured_eval_dir();
    let go = "go btime 1000 wtime 1000 searchmoves 8h2b";
    let out = drive(&format!(
        "usi\nisready\nposition startpos moves 7g7f 3c3d\n{go}\nisready\nquit\n"
    ));
    assert_eq!(out.matches("readyok\n").count(), 2, "{out}");
    assert!(bestmove_lines(&out).is_empty(), "{out}");
    assert!(
        out.contains(&diag_line(&format!("unknown command: {go}"))),
        "{out}"
    );
}

/// A fresh session's first depth-1 search answers with a plain `bestmove`: the
/// reply never carries a move to think about next, so nothing is read from a
/// table entry that may hold none. Right.
#[cfg(feature = "verbose2")]
#[cfg_attr(miri, ignore)]
#[test]
fn a_fresh_depth_one_search_names_no_ponder_move() {
    let _session = common::serial_session();
    stage_configured_eval_dir();
    let out = drive("usi\nisready\nposition startpos\ngo depth 1\nquit\n");
    let bestmoves = bestmove_lines(&out);
    assert_eq!(bestmoves.len(), 1, "{out}");
    assert_eq!(
        bestmoves[0].split_whitespace().count(),
        1,
        "a bare move, no `ponder` field: {out}"
    );
}

/// A depth-1 bench over the default positions on one thread runs to its
/// summary. Right.
#[cfg(feature = "verbose3")]
#[cfg_attr(miri, ignore)]
#[test]
fn a_depth_one_default_bench_completes() {
    let _session = common::serial_session();
    stage_configured_eval_dir();
    let out = drive("bench 64 1 1 default depth\nquit\n");
    assert!(out.contains("bench: positions=4"), "{out}");
}

/// A `setoption` value that is not a number, or one too large for any integer
/// type, leaves the session running: the next `isready` is answered, and a
/// search after it plays a move. Right.
#[cfg_attr(miri, ignore)]
#[test]
fn an_invalid_or_oversized_option_value_leaves_the_session_running() {
    let _session = common::serial_session();
    stage_configured_eval_dir();
    let out = drive(
        "usi\n\
         setoption name Threads value abc\n\
         setoption name Threads value 999999999999999999999999999999\n\
         setoption name USI_Hash value -1\n\
         isready\n\
         position startpos\n\
         go btime 1000 wtime 1000\n\
         stop\n\
         quit\n",
    );
    assert!(out.contains("readyok\n"), "{out}");
    assert_eq!(bestmove_lines(&out).len(), 1, "{out}");
}
