//! End-to-end checks of what a search writes to stdout per build flavor: drives the engine
//! binary (`CARGO_BIN_EXE_yorkie`) over USI pipes since `println!` can't be captured in-process. Gated on `material`.

#![cfg(feature = "material")]

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

/// Runs one fixed-depth search on the engine binary; returns the stdout lines between `readyok` and `bestmove`.
fn search_output_lines(depth: u32) -> Vec<String> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_yorkie"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("the engine binary must start");

    let mut stdin = child.stdin.take().expect("engine stdin is piped");
    write!(stdin, "usi\nisready\nposition startpos\ngo depth {depth}\n").expect("the engine must accept USI commands");
    stdin.flush().expect("the engine must accept USI commands");

    let stdout = BufReader::new(child.stdout.take().expect("engine stdout is piped"));
    let mut past_readyok = false;
    let mut lines = Vec::new();
    let mut saw_bestmove = false;
    for line in stdout.lines() {
        let line = line.expect("engine stdout must be valid UTF-8 lines");
        if line.starts_with("bestmove ") {
            saw_bestmove = true;
            break;
        }
        if past_readyok {
            lines.push(line);
        } else if line == "readyok" {
            past_readyok = true;
        }
    }
    assert!(
        saw_bestmove,
        "the search must end with a bestmove line; got so far: {lines:?}"
    );

    stdin.write_all(b"quit\n").expect("the engine must accept quit");
    drop(stdin);
    child.wait().expect("the engine must exit cleanly");
    lines
}

/// Normal (non-tournament) build: search info output is unchanged.
#[cfg(not(feature = "tournament"))]
#[test]
fn normal_build_search_emits_info_depth_and_pv() {
    let lines = search_output_lines(4);
    assert!(
        lines.iter().any(|l| l.starts_with("info depth ") && l.contains(" pv ")),
        "the normal build must report `info depth ... pv ...` during a search; got: {lines:?}"
    );
}

/// Tournament build: a search prints nothing before `bestmove`.
#[cfg(all(feature = "tournament", not(feature = "emit-nps")))]
#[test]
fn tournament_build_search_is_silent_until_bestmove() {
    let lines = search_output_lines(4);
    assert!(
        lines.is_empty(),
        "the tournament build must print nothing between `go` and `bestmove`; got: {lines:?}"
    );
}

/// Benchmark (tournament,emit-nps) build: exactly one `nps <integer>` line before `bestmove`.
#[cfg(all(feature = "tournament", feature = "emit-nps"))]
#[test]
fn emit_nps_build_search_emits_exactly_one_nps_line() {
    let lines = search_output_lines(4);
    assert_eq!(
        lines.len(),
        1,
        "the emit-nps build must print exactly one info line before `bestmove`; got: {lines:?}"
    );
    let line = &lines[0];
    assert!(line.starts_with("info nodes "), "unexpected bench line shape: {line}");
    let re = regex::Regex::new(r"\bnps\s+(\d+)\b").unwrap();
    assert!(
        re.is_match(line),
        "the bench line must carry an `nps <integer>` token; got: {line}"
    );
    assert!(
        !line.contains("depth") && !line.contains(" pv "),
        "the bench line must not carry PV/depth payload; got: {line}"
    );
}
