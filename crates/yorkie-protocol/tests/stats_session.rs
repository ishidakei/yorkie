//! The per-reply statistics line, from the outside: where it sits in a session's
//! output and what one reply's number covers.
//!
//! Gated on `verbose1`, which is the feature that installs the counting
//! allocator and writes the line at all; a build without it is covered by
//! `info_gating.rs`, which pins that no such line appears.
//!
//! The line is written by taking the counter, so what the previous reply
//! reported can never be reported again — the two numbers below being drawn from
//! two separate searches is the observable form of that.

#![cfg(feature = "verbose1")]

mod common;

use common::{drive, stage_configured_eval_dir};

const STATS_PREFIX: &str = "info string stats alloc=";

/// The `(index, count)` of every `info string stats` line in `out`.
fn stats_lines(out: &str) -> Vec<(usize, u64)> {
    out.lines()
        .enumerate()
        .filter_map(|(i, l)| {
            let n = l.strip_prefix(STATS_PREFIX)?;
            Some((
                i,
                n.parse()
                    .unwrap_or_else(|e| panic!("{l:?} must carry a count: {e}")),
            ))
        })
        .collect()
}

fn bestmove_indices(out: &str) -> Vec<usize> {
    out.lines()
        .enumerate()
        .filter(|(_, l)| l.starts_with("bestmove "))
        .map(|(i, _)| i)
        .collect()
}

#[cfg_attr(miri, ignore)]
#[test]
fn every_reply_reports_the_allocations_of_its_own_interval() {
    stage_configured_eval_dir();
    let out = drive(
        "usi\n\
         isready\n\
         usinewgame\n\
         position startpos moves 7g7f 3c3d\n\
         go byoyomi 1000\n\
         go byoyomi 1000\n\
         quit\n",
    );
    assert!(
        out.contains("readyok\n"),
        "the network must load in:\n{out}"
    );

    let bestmoves = bestmove_indices(&out);
    assert_eq!(bestmoves.len(), 2, "expected two replies in:\n{out}");

    let stats = stats_lines(&out);
    assert_eq!(
        stats.len(),
        2,
        "expected one statistics line per reply in:\n{out}"
    );

    for (&(stats_at, count), &bestmove_at) in stats.iter().zip(&bestmoves) {
        assert_eq!(
            stats_at + 1,
            bestmove_at,
            "the statistics line must be the line immediately before its \
             bestmove in:\n{out}"
        );
        assert!(
            count > 0,
            "a reply that searched allocated something, got {count} in:\n{out}"
        );
    }
}

#[cfg_attr(miri, ignore)]
#[test]
fn a_reply_that_starts_no_search_still_reports_its_interval() {
    // No network staged for this session — the working directory is wherever the
    // previous test left it, so this asserts only the shape a `bestmove` line
    // comes in, not which move it is.
    let out = drive(
        "position startpos\n\
         go byoyomi 1000\n\
         quit\n",
    );
    let bestmoves = bestmove_indices(&out);
    assert_eq!(bestmoves.len(), 1, "expected one reply in:\n{out}");
    let stats = stats_lines(&out);
    assert_eq!(stats.len(), 1, "expected one statistics line in:\n{out}");
    assert_eq!(
        stats[0].0 + 1,
        bestmoves[0],
        "the statistics line must directly precede the bestmove in:\n{out}"
    );
}
