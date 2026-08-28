//! The `usi` handshake, pinned to the byte.
//!
//! Both builds reply with identity and `usiok` and nothing between them: no
//! build has a runtime option surface, so there is no `option name ...` line to
//! advertise — every setting was compiled in from the TOML config. The golden is
//! exact, so a stray option line fails.

use std::sync::{Arc, Mutex};

use yorkie_protocol::UsiDriver;

fn drive(input: &str) -> String {
    let output = Arc::new(Mutex::new(Vec::<u8>::new()));
    let driver = UsiDriver::new(input.as_bytes(), Arc::clone(&output));
    driver.run().expect("driver run");
    let bytes = output.lock().expect("output lock").clone();
    String::from_utf8(bytes).expect("utf-8")
}

/// Identity and `usiok`, nothing between them — in every build.
#[cfg_attr(miri, ignore)]
#[test]
fn full_usi_to_usiok_golden() {
    let out = drive("usi\nquit\n");
    let expected = "\
id name Yorkie 3.1.0\n\
id author Kei Ishida <ishida.kei@gmail.com>\n\
usiok\n";
    assert_eq!(out, expected);
}

#[cfg_attr(miri, ignore)]
#[test]
fn isready_without_network_reports_load_failure() {
    // Default EvalDir (`eval`) has no `nn.bin` in the test CWD, so the load
    // fails: an `info string eval load failed:` notice and no `readyok`, per
    // the isready contract — a failed network load must never be answered with
    // `readyok`. The positive path is covered by
    // tests/eval_session.rs (synthetic network) and tests/real_network_selfplay.
    let out = drive("isready\nquit\n");
    assert!(
        out.contains("info string eval load failed:"),
        "expected eval-load-failure notice, got: {out:?}"
    );
    assert!(
        !out.contains("readyok"),
        "readyok must not appear on a failed load: {out:?}"
    );
}

#[cfg_attr(miri, ignore)]
#[test]
fn unknown_command_emits_info_string() {
    let out = drive("frobnicate\nquit\n");
    assert_eq!(out, "info string unknown command: frobnicate\n");
}

#[cfg_attr(miri, ignore)]
#[test]
fn oversized_line_emits_command_too_long() {
    // 64 KB + 1 byte → TooLong. Followed by a real line so the driver loops past it.
    let mut input = "x".repeat(64 * 1024 + 1);
    input.push('\n');
    input.push_str("quit\n");
    let out = drive(&input);
    assert_eq!(out, "info string command too long\n");
}

/// `setoption` is the USI minimum: the line is consumed, nothing is emitted, and
/// nothing changes. USI requires no reply to `setoption`, so the whole
/// transcript is byte-identical to the one where the line was never sent.
#[cfg_attr(miri, ignore)]
#[test]
fn full_handshake_then_consumed_setoption_then_quit() {
    let out = drive("usi\nsetoption name USI_Hash value 256\nisready\nquit\n");
    assert!(out.starts_with("id name Yorkie 3.1.0\n"));
    assert!(out.contains("usiok\n"));
    // The session is still usable: `isready` behaves exactly as it does with no
    // `setoption` at all (default `eval/nn.bin` absent → load fails, no readyok).
    assert!(out.contains("info string eval load failed:"));
    assert!(!out.contains("readyok"));
    assert_eq!(
        out,
        drive("usi\nisready\nquit\n"),
        "a consumed `setoption` must not add or change a single byte"
    );
}

#[cfg_attr(miri, ignore)]
#[test]
fn usinewgame_emits_nothing() {
    let out = drive("usinewgame\nquit\n");
    assert_eq!(out, "");
}
