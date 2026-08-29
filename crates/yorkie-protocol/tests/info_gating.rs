//! What the `info` features gate, from the outside: a real search in the
//! default (tournament) build says `bestmove` and nothing else, while the
//! `isready` / initialisation-phase `info string`s survive in every build.
//!
//! The three build shapes this file distinguishes:
//!
//! | build | search `info` | diagnostic `info string` | init-phase `info string` |
//! |---|---|---|---|
//! | default | — | — | yes |
//! | `info-diag` | — | yes | yes |
//! | `info-output` | yes | yes | yes |
//!
//! The positive side (what an `info-output` build emits, to the byte) is pinned
//! by `tests/depth1_session.rs`, `tests/limit_session.rs` and
//! `tests/multipv_session.rs`; this file pins the NEGATIVE side, which is the
//! claim the tournament build rests on and which no other test can make.
//!
//! Deliberately NOT `usi-extras`-gated: the search here is driven by a
//! clock-bounded `go`, the only `go` a rated game issues, so the file runs in
//! every build shape.
//!
//! **One test on purpose.** The first half needs the package-root working
//! directory (no `eval/nn.bin`, so the load fails), the second half stages a
//! synthetic network and enters a fixture root. Working directory is
//! process-global, so the two halves are one sequential test rather than two
//! that could interleave under a threaded `cargo test`.

mod common;

use common::{drive, stage_configured_eval_dir};

/// A line that is part of the `info` output surface (`info …`, including
/// `info string …`).
fn info_lines(out: &str) -> Vec<&str> {
    out.lines().filter(|l| l.starts_with("info")).collect()
}

#[cfg_attr(miri, ignore)]
#[test]
fn search_output_is_bestmove_only_unless_info_output_is_built_in() {
    // --- Part 1: the initialisation phase, which no feature gates. ---
    //
    // No network at the package-root working directory, so `isready` fails the
    // load. That notice (and the withheld `readyok`) is how a bad deployment is
    // diagnosed at all, so it must survive in the default build too.
    let out = drive("isready\nquit\n");
    assert!(
        out.contains("info string eval load failed:"),
        "the initialisation-phase notice must survive in every build, got: {out:?}"
    );
    assert!(!out.contains("readyok"), "unexpected readyok in: {out:?}");

    // --- Part 2: a real, clock-bounded search. ---
    //
    // `stop` rides in the same input, so the search aborts at its first
    // checkpoint and the session stays fast; the reply is still a full one
    // (final PV under `info-output`, then `bestmove`).
    stage_configured_eval_dir();
    let out = drive(
        "usi\n\
         isready\n\
         position startpos moves 7g7f\n\
         go btime 60000 wtime 60000 binc 1000 winc 1000\n\
         stop\n\
         quit\n",
    );
    assert!(out.contains("readyok\n"), "the network must load in: {out}");
    let bestmoves = common::bestmove_lines(&out);
    assert_eq!(bestmoves.len(), 1, "expected one bestmove in:\n{out}");

    let infos = info_lines(&out);
    if cfg!(feature = "info-output") {
        assert!(
            infos.iter().any(|l| l.starts_with("info depth ")),
            "an info-output build must report the search in:\n{out}"
        );
    } else {
        // The whole claim of the default build, in one assertion: a search that
        // ran, produced a move, and printed not one `info` line — search info
        // and diagnostics alike. A clean load emits no initialisation-phase
        // line either, so the expected count really is zero.
        assert!(
            infos.is_empty(),
            "a build without `info-output` must emit no info line during a \
             search, got {infos:?} in:\n{out}"
        );
    }
}
