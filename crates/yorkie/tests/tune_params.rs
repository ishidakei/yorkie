//! A `tune` build's search parameters, set through `YORKIE_TUNE`: a list the
//! engine cannot honour stops it before the session starts, the defaults search
//! exactly as no list does, and a moved parameter reaches the search.
//!
//! The search comparisons drive `go nodes`, so they need the `verbose2` search
//! surface as well.

#![cfg(feature = "tune")]

use std::io::Write;
use std::process::{Command, Output, Stdio};

mod common;

/// Spawn the engine with `tune` as its tuning list (unset when `None`), feed it
/// `input`, and collect what it wrote.
fn run(tune: Option<&str>, input: &[u8]) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_yorkie"));
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    match tune {
        Some(list) => cmd.env("YORKIE_TUNE", list),
        None => cmd.env_remove("YORKIE_TUNE"),
    };
    let mut child = cmd.spawn().expect("spawn engine");
    // The engine may exit before it reads a byte, so a closed pipe is no error.
    let _ = child.stdin.take().expect("stdin piped").write_all(input);
    child.wait_with_output().expect("wait")
}

#[cfg_attr(miri, ignore)]
#[test]
fn a_list_that_cannot_be_honoured_stops_the_engine_before_usiok() {
    for (list, notice) in [
        (
            "razor_base=480,razor_bias=3",
            "info string YORKIE_TUNE refused: unknown parameter `razor_bias`\n",
        ),
        (
            "razor_base=4.5",
            "info string YORKIE_TUNE refused: bad value `4.5` for `razor_base` \
             (an integer in 0..=2000)\n",
        ),
        (
            "tm_max_ratio=999",
            "info string YORKIE_TUNE refused: bad value `999` for `tm_max_ratio` \
             (an integer in 1000..=20000)\n",
        ),
    ] {
        let out = run(Some(list), b"usi\nisready\nquit\n");
        assert_eq!(out.status.code(), Some(2), "{list}");
        assert_eq!(String::from_utf8_lossy(&out.stdout), notice, "{list}");
    }
}

#[cfg(feature = "verbose2")]
mod search {
    use super::*;

    /// A fixed-node search of a few opening positions: every `info depth` line
    /// with the fields that measure time removed, and each `bestmove`.
    fn search_lines(tune: Option<&str>) -> Vec<String> {
        let input = b"usi\nisready\nusinewgame\n\
            position startpos\ngo nodes 20000\n\
            position startpos moves 7g7f 3c3d 2g2f 8c8d 2f2e 8d8e\ngo nodes 20000\n\
            quit\n";
        let out = run(tune, input);
        assert!(out.status.success(), "engine exited {:?}", out.status);
        let stdout = String::from_utf8(out.stdout).expect("utf-8 stdout");
        assert!(stdout.contains("readyok"), "no readyok in:\n{stdout}");
        stdout
            .lines()
            .filter(|l| l.starts_with("info depth") || l.starts_with("bestmove"))
            .map(|l| {
                let mut kept = Vec::new();
                let mut tokens = l.split(' ');
                while let Some(t) = tokens.next() {
                    if matches!(t, "time" | "nps" | "hashfull") {
                        tokens.next();
                    } else {
                        kept.push(t);
                    }
                }
                kept.join(" ")
            })
            .collect()
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn the_defaults_search_as_no_list_does_and_a_moved_parameter_does_not() {
        common::require_test_config();
        if !common::engine_has_network() {
            eprintln!("no network beside the binary; nothing to search with");
            return;
        }
        let unset = search_lines(None);
        assert!(unset.iter().filter(|l| l.starts_with("bestmove")).count() == 2);
        assert_eq!(search_lines(Some("")), unset, "an empty list");
        assert_eq!(
            search_lines(Some("lmr_base=1133,qs_see_margin=-73,aspiration_delta=5")),
            unset,
            "the defaults spelled out"
        );
        assert_ne!(
            search_lines(Some("lmr_base=3000")),
            unset,
            "a moved parameter must change the search"
        );
    }
}
