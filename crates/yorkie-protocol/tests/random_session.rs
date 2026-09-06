//! Driver-level session tests for the per-game evaluation noise.
//!
//! Each test drives a full `usi → isready → position → go` session in-process
//! against a synthetic all-zero network staged at the compiled-in `EvalDir`.
//! That network evaluates every position at 0, so a score reported here *is* the
//! noise: its size measures the amplitude, and its change between games measures
//! the seed.
//!
//! What a session can and cannot see. The evaluation of a position is fixed for
//! a whole game — that is the feature, and it is pinned where it is a property,
//! on the noise function and on a search handed a seed
//! (`yorkie-search/src/qsearch.rs::tests::evaluation_noise`). A reported *search*
//! score is not the same thing: it also depends on the transposition table the
//! previous search of the game left warm, so two identical `go` commands in one
//! game can report different scores in any build, with noise or without. The
//! session-level claims are therefore the two a session really settles: the
//! noise stays inside the configured amplitude, and a new game moves it.
//!
//! The amplitude is a compile-time constant, so these tests assert what the
//! value they were *built* with implies and skip themselves when it is 0.
//!
//! Gated on `verbose2` as well as on the feature: the assertions read the search
//! `info` lines' scores, and drive a `go depth`, both of which that feature
//! brings.

#![cfg(all(feature = "random", feature = "verbose2"))]

mod common;

use common::{StreamHarness, bestmove_lines, stage_configured_eval_dir};
use yorkie_protocol::config;

/// How many fresh games a differing score is looked for over. The reported score
/// is a coarse function of the seed, so two games can land on the same one; what
/// the feature promises is that they do not *keep* landing there.
const GAMES: usize = 8;

/// Every `cp` score in `text`, in the order the `info` lines carried them — the
/// whole shape of what the search reported, not just its last line.
fn scores_cp(text: &str) -> Vec<i64> {
    text.lines()
        .filter_map(|line| line.split(" score cp ").nth(1))
        .filter_map(|rest| rest.split_whitespace().next())
        .filter_map(|tok| tok.parse::<i64>().ok())
        .collect()
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

/// Search the start position and return the scores that search reported.
fn go_and_scores(h: &StreamHarness) -> Vec<i64> {
    let before = h.output();
    let want = bestmove_lines(&before).len() + 1;
    h.send("position startpos");
    h.send("go depth 2");
    assert!(
        h.wait_until(30_000, |o| bestmove_lines(o).len() == want),
        "the search must finish:\n{}",
        h.output()
    );
    let leg = h.output()[before.len()..].to_string();
    let scores = scores_cp(&leg);
    assert!(!scores.is_empty(), "a search must report a score:\n{leg}");
    scores
}

/// Whether this build carries an amplitude to see anything with.
fn noisy_build() -> bool {
    if config::RANDOM == 0 {
        eprintln!("skipped: this build's `random` is 0, so there is no noise to observe");
        return false;
    }
    true
}

/// The noise is there, and it is the size the setting asked for.
///
/// Every leaf of this search evaluates to its offset alone, so the backed-up
/// score cannot leave the offsets' own span: `random` centipawns wide, centred
/// on zero, plus one centipawn for the truncation the centipawn rendering does.
#[cfg_attr(miri, ignore)]
#[test]
fn the_reported_score_is_noise_within_the_configured_amplitude() {
    if !noisy_build() {
        return;
    }
    let bound = config::RANDOM / 2 + 1;
    let h = start_ready();
    let scores = go_and_scores(&h);
    let out = h.quit_join();
    assert!(
        scores.iter().any(|cp| *cp != 0),
        "an amplitude of {} must move some score off the zero network's 0: {scores:?}\n{out}",
        config::RANDOM
    );
    for cp in &scores {
        assert!(
            cp.abs() <= bound,
            "score {cp} is outside the configured amplitude (±{bound} cp): {scores:?}\n{out}"
        );
    }
}

/// `usinewgame` redraws the seed, which is the whole point: the next game does
/// not evaluate the position the way this one did.
#[cfg_attr(miri, ignore)]
#[test]
fn a_new_game_evaluates_the_position_differently() {
    if !noisy_build() {
        return;
    }
    let h = start_ready();
    let first = go_and_scores(&h);
    let mut differed = false;
    for _ in 0..GAMES {
        h.send("usinewgame");
        if go_and_scores(&h) != first {
            differed = true;
            break;
        }
    }
    let out = h.quit_join();
    assert!(
        differed,
        "{GAMES} fresh games all reported {first:?} — the seed is not being redrawn:\n{out}"
    );
}
