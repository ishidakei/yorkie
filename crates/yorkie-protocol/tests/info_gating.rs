//! What the verbosity features gate on the output side, from the outside: a real
//! search in the default (tournament) build says `bestmove` and nothing else,
//! while the `isready` / initialisation-phase `info string`s survive in every
//! build.
//!
//! The three build shapes this file distinguishes:
//!
//! | build | search `info` | diagnostic `info string` | per-reply `info string stats` | init-phase `info string` |
//! |---|---|---|---|---|
//! | no feature | — | — | — | yes |
//! | `verbose1` | — | yes | yes | yes |
//! | `verbose2` | yes | yes | yes | yes |
//!
//! The positive side is pinned by the session tests; this file pins the negative
//! side, which no other test can make.
//!
//! Deliberately ungated: the search here is driven by a clock-bounded `go`, so
//! the file runs in every build shape.
//!
//! **One test on purpose.** The first half needs the package-root working
//! directory so the network load fails, and the second stages a synthetic
//! network and enters a fixture root. The working directory is process-global,
//! so the two halves must not interleave under a threaded `cargo test`.

mod common;

use common::{drive, stage_configured_eval_dir};

/// A line that is part of the `info` output surface (`info …`, including
/// `info string …`).
fn info_lines(out: &str) -> Vec<&str> {
    out.lines().filter(|l| l.starts_with("info")).collect()
}

#[cfg_attr(miri, ignore)]
#[test]
fn search_output_is_bestmove_only_below_verbose2() {
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
    // (final PV under `verbose2`, then `bestmove`).
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
    if cfg!(feature = "verbose2") {
        assert!(
            infos.iter().any(|l| l.starts_with("info depth ")),
            "a verbose2 build must report the search in:\n{out}"
        );
    } else {
        // The whole claim of the default build, in one assertion: a search that
        // ran, produced a move, and printed not one `info` line about itself.
        // Two lines in this session are not about the search, and each is
        // counted on its own:
        //
        //   - `isready` reports where it placed the transposition table, and
        //     where it put the evaluation network. Initialisation-phase lines,
        //     so both are here in every build.
        //   - `verbose1` adds the per-reply statistics line before the
        //     `bestmove`, which reports what the process allocated.
        let placement: Vec<&&str> = infos
            .iter()
            .filter(|l| {
                l.starts_with("info string transposition table: ")
                    || l.starts_with("info string evaluation network: ")
            })
            .collect();
        assert_eq!(
            placement.len(),
            2,
            "`isready` reports the table's and the network's placement in every \
             build, got {infos:?} in:\n{out}"
        );
        // The line says what the table's pages were given: the node set the
        // compiled thread plan's workers run on, as one node the table prefers
        // or the several it is interleaved over, or the process's own policy
        // where that plan pins no worker and the engine sets none. Which of the
        // three this build gets depends on its thread count and the layout it
        // was built for, so the assertion accepts any of them and insists that
        // one is named.
        assert!(
            placement[0].contains("; preferred on node ")
                || placement[0].contains("; interleave on nodes ")
                || placement[0].contains("; process default policy;"),
            "the placement line says what the pages were given, got {:?} in:\n{out}",
            placement[0]
        );
        // The network's line says the same about its own memory: the one
        // mapping every reader on a single-node machine shares, or the copy per
        // node the workers read it from.
        assert!(
            placement[1].contains("; one shared mapping;")
                || placement[1].contains("; one copy on "),
            "the network line says where the parameters went, got {:?} in:\n{out}",
            placement[1]
        );
        let stats: Vec<&&str> = infos
            .iter()
            .filter(|l| l.starts_with("info string stats "))
            .collect();
        assert_eq!(
            stats.len(),
            usize::from(cfg!(feature = "verbose1")),
            "the statistics line arrives with `verbose1` and with nothing \
             below it, got {infos:?} in:\n{out}"
        );
        assert_eq!(
            infos.len(),
            placement.len() + stats.len(),
            "a build without `verbose2` must emit no info line about the \
             search itself, got {infos:?} in:\n{out}"
        );
    }
}
