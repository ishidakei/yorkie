//! The `usi` handshake, pinned to the byte.
//!
//! Both builds reply with identity and `usiok` and nothing between them: no
//! build has a runtime option surface, so there is no `option name …` line to
//! advertise. The golden is exact, so a stray option line fails.
//!
//! The `isready` load-failure notice is asserted unconditionally because it
//! belongs to the initialisation phase, which no `info` feature gates; the
//! unknown-command and too-long lines are diagnostics and go through
//! [`diag_line`].

use std::sync::{Arc, Mutex};

use yorkie_protocol::UsiDriver;

fn drive(input: &str) -> String {
    let output = Arc::new(Mutex::new(Vec::<u8>::new()));
    let driver = UsiDriver::new(input.as_bytes(), Arc::clone(&output));
    driver.run().expect("driver run");
    let bytes = output.lock().expect("output lock").clone();
    String::from_utf8(bytes).expect("utf-8")
}

/// The transcript a diagnostic `info string <body>` contributes in this build —
/// nothing without `verbose1`. (This file predates `tests/common`, and keeps
/// its own two-line harness so the handshake golden depends on nothing else.)
fn diag_line(body: &str) -> String {
    if cfg!(feature = "verbose1") {
        format!("info string {body}\n")
    } else {
        String::new()
    }
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

/// A binary plans its worker binding from the layout of the machine it was built
/// on, so a machine laid out differently is one it must not play on: `isready`
/// names the difference and withholds `readyok`, before it has allocated
/// anything for a machine that is not there.
///
/// The machine is presented through a sysfs tree describing CPUs no host has,
/// so the difference is certain whatever the host running the test looks like.
#[cfg_attr(miri, ignore)]
#[test]
fn isready_refuses_a_machine_that_is_not_the_one_the_binary_was_built_for() {
    let root = std::env::temp_dir().join(format!("yorkie-numa-check-{}", std::process::id()));
    let cpu = root.join("devices/system/cpu");
    let node = root.join("devices/system/node");
    std::fs::create_dir_all(cpu.join("cpu900")).expect("mkdir cpu900");
    std::fs::create_dir_all(node.join("node0")).expect("mkdir node0");
    std::fs::create_dir_all(node.join("node1")).expect("mkdir node1");
    std::fs::write(cpu.join("online"), "900-901\n").expect("write cpu online");
    std::fs::write(node.join("online"), "0-1\n").expect("write node online");
    std::fs::write(node.join("node0/cpulist"), "900\n").expect("write node0");
    std::fs::write(node.join("node1/cpulist"), "901\n").expect("write node1");

    let output = Arc::new(Mutex::new(Vec::<u8>::new()));
    let driver =
        UsiDriver::new(&b"isready\nquit\n"[..], Arc::clone(&output)).with_sysfs_root(root.clone());
    driver.run().expect("driver run");
    let out = String::from_utf8(output.lock().expect("output lock").clone()).expect("utf-8");

    let _ = std::fs::remove_dir_all(&root);

    assert!(
        out.contains("info string NUMA layout mismatch:"),
        "expected the layout notice, got: {out:?}"
    );
    assert!(
        !out.contains("readyok"),
        "readyok must not appear on a layout the binary was not built for: {out:?}"
    );
    assert!(
        !out.contains("eval load failed"),
        "the check runs before anything is loaded: {out:?}"
    );
}

#[cfg_attr(miri, ignore)]
#[test]
fn unknown_command_emits_info_string() {
    let out = drive("frobnicate\nquit\n");
    assert_eq!(out, diag_line("unknown command: frobnicate"));
}

#[cfg_attr(miri, ignore)]
#[test]
fn oversized_line_emits_command_too_long() {
    // 64 KB + 1 byte → TooLong. Followed by a real line so the driver loops past it.
    let mut input = "x".repeat(64 * 1024 + 1);
    input.push('\n');
    input.push_str("quit\n");
    let out = drive(&input);
    assert_eq!(out, diag_line("command too long"));
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
