//! Driver-level session tests against a **synthetic** SFNN-1536 network.
//!
//! These are hermetic: they write an evaluation file this build accepts into a
//! directory of their own and drive a full session in-process against it, so
//! they run everywhere, not just where the real network is staged. `eval_dir`
//! cannot be pointed anywhere at run time, so what a test chooses is the
//! directory it resolves against — see [`common::stage_configured_eval_dir`].
//!
//! The synthetic network is all zeros, so every position evaluates to the same
//! constant and the chosen move is fully deterministic. That lets the tests
//! assert the driver's `bestmove` equals a direct [`QSearch::run_root`] call.
//! The comparison holds because the config compiles in a single worker; Lazy-SMP
//! with more is deliberately not reproducible move-for-move.
//!
//! Gated on `verbose2`: these sessions drive analysis-only `go` clauses, which
//! a build without that feature refuses rather than reinterprets, and the
//! search-report assertions read the `info depth …` lines, which only a
//! `verbose2` build emits.

#![cfg(feature = "verbose2")]

mod common;

use common::{NOISY_EVALUATION, drive, evaluation_is_noise_free, stage_configured_eval_dir};
use yorkie_protocol::{UsiDriver, config};
use yorkie_search::{QSearch, RootKind, RootOutcome, Search};
use yorkie_state::{Move, Position, format_usi_move, parse_sfen, parse_usi_move};
use yorkie_storage::TranspositionTable;

/// The one transposition table, emptied — what a driver session and a direct
/// `run_root` each need to start from if their results are to be compared.
///
/// The table is a `static`, so the direct search below and the session that
/// follows it use the very same one; only clearing between them makes the two
/// searches comparable.
fn cleared_tt() -> &'static TranspositionTable {
    let tt = TranspositionTable::shared();
    tt.clear();
    tt
}

/// The USI-string form of a `run_root` outcome's bestmove (the synthetic
/// positions never hit the declaration-win exit, but the mapping is exhaustive).
fn bestmove_usi(outcome: &RootOutcome) -> String {
    match outcome.kind {
        RootKind::Resign => "resign".to_string(),
        RootKind::DeclarationWin => "win".to_string(),
        RootKind::Normal => format_usi_move(outcome.best_move),
    }
}

fn legal_moves(p: &Position) -> Vec<Move> {
    let mut moves = Vec::new();
    p.generate_legal_all(&mut moves);
    moves
}

fn bestmove_lines(out: &str) -> Vec<&str> {
    out.lines()
        .filter_map(|l| l.strip_prefix("bestmove "))
        .collect()
}

#[cfg_attr(miri, ignore)]
#[test]
fn synthetic_network_session_matches_direct_search_choice() {
    // The direct `run_root` below installs no game seed, so it is the plain
    // evaluation's choice; only a build whose sessions make the same one can be
    // compared against it.
    if !evaluation_is_noise_free() {
        eprintln!("{NOISY_EVALUATION}");
        return;
    }
    let path = stage_configured_eval_dir();

    // Independent, direct depth-1 root-search choice for the same network +
    // startpos, on the same table the session will use. Scoped so the network
    // frees before the driver session loads its own.
    let startpos = parse_sfen(yorkie_state::STARTPOS_SFEN).expect("startpos SFEN");
    let expected_usi = {
        let (search, _warnings) =
            Search::map_evaluation_file(&path).expect("the synthetic network opens");
        let outcome = QSearch::new(search.network(), cleared_tt()).run_root(&startpos, 1);
        bestmove_usi(&outcome)
    };
    // The session below starts from an empty table, as this search just did.
    cleared_tt();

    // Full session, with a repeat `isready` to exercise idempotent reload.
    let out = drive(
        "usi\n\
         isready\n\
         isready\n\
         position startpos\n\
         go depth 1\n\
         quit\n",
    );

    // Both `isready`s acknowledge; no load failure.
    assert_eq!(
        out.matches("readyok\n").count(),
        2,
        "both isready must emit readyok in:\n{out}"
    );
    assert!(
        !out.contains("eval load failed"),
        "unexpected load failure in:\n{out}"
    );

    // The search emitted its depth-1 progress report.
    assert!(
        out.lines().any(|l| l.starts_with("info depth 1 ")),
        "missing search info report in:\n{out}"
    );

    // Exactly one bestmove, equal to the direct search choice, and legal.
    let bestmoves = bestmove_lines(&out);
    assert_eq!(bestmoves.len(), 1, "expected one bestmove in:\n{out}");
    assert_eq!(
        bestmoves[0], expected_usi,
        "driver bestmove must equal yorkie-search's direct choice"
    );
    let parsed = parse_usi_move(bestmoves[0], &startpos).expect("well-formed USI");
    assert!(
        legal_moves(&startpos).contains(&parsed),
        "{} is not a legal startpos move",
        bestmoves[0]
    );
}

#[cfg_attr(miri, ignore)]
#[test]
fn isready_keep_alive_emits_bare_newline_during_heavy_load() {
    // With a very short injected poll interval, the real heavy work here — the
    // evaluation file's opening and the book load — spans many keep-alive ticks
    // and reliably
    // emits at least one bare newline before `readyok`.
    let staged = stage_configured_eval_dir();
    let eval_root = staged
        .parent()
        .and_then(std::path::Path::parent)
        .expect("the staged network sits under <root>/<eval_dir>")
        .to_path_buf();

    let input = "usi\nisready\nquit\n".to_string();
    let output = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
    let driver = UsiDriver::new(input.as_bytes(), std::sync::Arc::clone(&output))
        .with_eval_root(eval_root)
        .with_keep_alive_poll(std::time::Duration::from_micros(100));
    driver.run().expect("driver run");
    let out = String::from_utf8(output.lock().expect("output lock").clone()).expect("utf-8");

    // The load succeeded: exactly one readyok, no failure notice.
    assert!(
        !out.contains("eval load failed"),
        "unexpected load failure in:\n{out:?}"
    );
    let readyok_pos = out.find("readyok\n").expect("readyok emitted");

    // At least one bare keep-alive newline (an empty transcript line) appeared
    // before readyok. No line the driver emits is otherwise empty, so any empty
    // line is a keep-alive newline.
    let before = &out[..readyok_pos];
    let bare_before = before.split('\n').filter(|s| s.is_empty()).count();
    assert!(
        bare_before >= 1,
        "expected a bare keep-alive newline before readyok in:\n{out:?}"
    );

    // No interleaving: the keep-alive newline goes through the shared writer as a
    // whole line, so `usiok` and `readyok` survive intact and every non-empty
    // line is a complete USI line (never split by a stray newline).
    assert!(out.contains("usiok\n"), "usiok must be intact in:\n{out:?}");
    assert!(
        out.lines().any(|l| l == "readyok"),
        "readyok must be intact in:\n{out:?}"
    );
}

#[cfg_attr(miri, ignore)]
#[test]
fn synthetic_network_reuse_reset_and_mate_resign() {
    // Same reason as above, doubled: the session's second `go` is a new game,
    // which is exactly where the noise changes.
    if !evaluation_is_noise_free() {
        eprintln!("{NOISY_EVALUATION}");
        return;
    }
    let file = stage_configured_eval_dir();

    // A single load (one `isready`) serves all three `go`s below: this pins that
    // the loaded network is reused across positions without reloading.
    let mut post_7g7f = parse_sfen(yorkie_state::STARTPOS_SFEN).expect("startpos");
    let m = parse_usi_move("7g7f", &post_7g7f).expect("legal 7g7f");
    post_7g7f.do_move(m);
    let startpos = parse_sfen(yorkie_state::STARTPOS_SFEN).expect("startpos");

    // Independent depth-1 root-search choices, reproducing the session's TT
    // lifecycle: an empty table for `go` #1 (post-7g7f), then `usinewgame`
    // (tt.clear) before `go` #2 (startpos). Scoped so the network frees before
    // the driver loads its own.
    let (expected_after_7g7f, expected_startpos) = {
        let (search, _warnings) =
            Search::map_evaluation_file(&file).expect("the synthetic network opens");
        let tt = cleared_tt();
        let e1 = bestmove_usi(&QSearch::new(search.network(), tt).run_root(&post_7g7f, 1));
        tt.clear(); // usinewgame equivalent.
        let e2 = bestmove_usi(&QSearch::new(search.network(), tt).run_root(&startpos, 1));
        (e1, e2)
    };
    // The session below starts from an empty table, as the first search did.
    cleared_tt();

    // A mate for the side to move (White): no legal move → search resigns.
    let mate = "4k4/4G4/3S5/9/9/9/9/9/4K4 w - 1";

    // `go depth 1` pins each search to the depth-1 root path so the choice is the
    // deterministic `run_root(pos, 1)` above. (A bare `go` is now an infinite,
    // clock-driven search — so it is not used where a fixed,
    // reproducible depth-1 result is asserted.)
    let session = format!(
        "usi\n\
         isready\n\
         position startpos moves 7g7f\n\
         go depth 1\n\
         usinewgame\n\
         go depth 1\n\
         position sfen {mate}\n\
         go depth 1\n\
         quit\n"
    );
    let out = drive(&session);

    assert_eq!(
        out.matches("readyok\n").count(),
        1,
        "one isready, one readyok in:\n{out}"
    );

    let bestmoves = bestmove_lines(&out);
    assert_eq!(bestmoves.len(), 3, "expected three bestmoves in:\n{out}");
    // 1) post-7g7f position (White to move).
    assert_eq!(
        bestmoves[0], expected_after_7g7f,
        "first go must reflect the post-7g7f position"
    );
    // 2) after usinewgame the position is reset to startpos (Black to move).
    assert_eq!(
        bestmoves[1], expected_startpos,
        "usinewgame must reset the position to startpos"
    );
    // 3) mate → the search finds no legal move → resign.
    assert_eq!(bestmoves[2], "resign", "mate position must resign");
}

/// A CSA-27-point-declarable position for the side to move (Black king walled
/// into the enemy field with 12 own pieces there — the same shape as
/// yorkie-search's `NYUGYOKU` fixture), reached via `position sfen`.
const DECLARABLE_SFEN: &str = "+R+R+B+B5/3GKG3/2SGGGS2/9/9/9/9/9/4k4 b R 1";

/// The rule this build declares under. The other rules (`NoEnteringKing`,
/// `TryRule`) are not reachable from a session: `EnteringKingRule` is a
/// compile-time constant, and the rules themselves are covered by
/// `yorkie-search`'s own `declaration_win` tests, which take the rule as a
/// parameter.
#[cfg_attr(miri, ignore)]
#[test]
fn entering_king_configured_rule_declares_win_without_searching() {
    // A 27-point-declarable position under the configured rule yields
    // `bestmove win` and emits no search `info` line (the pre-search
    // declaration shortcut fires before any worker runs).
    assert_eq!(
        config::ENTERING_KING_RULE,
        "CSARule27",
        "this fixture is 27-point declarable; another configured rule needs \
         another fixture"
    );
    stage_configured_eval_dir();

    let session = format!(
        "usi\n\
         isready\n\
         position sfen {DECLARABLE_SFEN}\n\
         go depth 1\n\
         quit\n"
    );
    let out = drive(&session);

    let bestmoves = bestmove_lines(&out);
    assert_eq!(bestmoves.len(), 1, "expected one bestmove in:\n{out}");
    assert_eq!(bestmoves[0], "win", "default rule must declare in:\n{out}");
    assert!(
        !out.lines().any(|l| l.starts_with("info depth ")),
        "declaration shortcut must not run a search in:\n{out}"
    );
}
