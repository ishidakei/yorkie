//! Smoke test: spawn the built `yorkie` binary, drive a USI handshake, assert
//! on stdout. Confirms the no-arg entry point routes to `UsiDriver` correctly.
//!
//! Both build shapes reply identically here: neither advertises an option, and
//! neither replies to a `setoption`, having had every setting compiled into it
//! from the TOML config.

use std::io::Write;
use std::process::{Command, Stdio};

/// Spawn the engine binary, feed it `input`, and return its stdout. Fails the
/// test if it exits non-zero.
fn spawn_and_drive(input: &[u8]) -> String {
    let exe = env!("CARGO_BIN_EXE_yorkie");
    let mut child = Command::new(exe)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn engine");

    {
        let stdin = child.stdin.as_mut().expect("stdin piped");
        stdin.write_all(input).expect("write stdin");
    }

    let out = child.wait_with_output().expect("wait");
    assert!(
        out.status.success(),
        "engine exited non-zero: {:?}",
        out.status
    );
    String::from_utf8(out.stdout).expect("utf-8 stdout")
}

/// The shared part of the contract: identity, `usiok`, and an
/// `isready` that reports a load failure (no `eval/nn.bin` in the spawned
/// binary's CWD) rather than answering `readyok`. The positive isready path is
/// covered by `tests/real_network_selfplay` against the real network.
fn assert_common_handshake(stdout: &str) {
    assert!(stdout.contains("id name "), "missing id name in:\n{stdout}");
    assert!(
        stdout.contains("id author "),
        "missing id author in:\n{stdout}"
    );
    assert!(stdout.contains("usiok\n"), "missing usiok in:\n{stdout}");
    assert!(
        stdout.contains("info string eval load failed:"),
        "missing eval-load-failure notice in:\n{stdout}"
    );
    assert!(
        !stdout.contains("readyok"),
        "readyok must not appear on a failed load in:\n{stdout}"
    );
}

/// The binary as it is actually shipped: no options advertised, and a
/// `setoption` consumed in silence — the USI minimum, since there is no option
/// to set and the protocol asks for no reply.
#[cfg_attr(miri, ignore)]
#[test]
fn handshake_round_trip_via_spawned_binary() {
    let stdout = spawn_and_drive(b"usi\nsetoption name USI_Hash value 256\nisready\nquit\n");
    assert_common_handshake(&stdout);

    assert!(
        !stdout.contains("option name "),
        "no build may advertise an option:\n{stdout}"
    );
    assert_eq!(
        stdout,
        spawn_and_drive(b"usi\nisready\nquit\n"),
        "a consumed `setoption` must not add or change a single byte"
    );
}
