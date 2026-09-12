//! Driver-level session tests for input this engine's own text is not spelled
//! in: a command line carrying bytes that are not valid UTF-8.
//!
//! USI is ASCII by specification, but a host is free to put a path into a
//! `setoption` value in whatever code page it runs under — a Windows GUI sends
//! `\x82\xa0` for `あ` — and a `position` line can arrive garbled for the same
//! reason. Such a line is read like any other: the tokens in it match no
//! keyword and parse as no number, so the command they belong to takes the path
//! a malformed ASCII token takes, and the session goes on.

mod common;

use common::{bestmove_lines, drive, drive_bytes, stage_configured_eval_dir};

/// The two bytes a Windows code page spells `あ` with. Not valid UTF-8, which
/// is what makes reading the line the interesting part.
const CP932_A: &[u8] = b"\x82\xa0";

/// `setoption` with such a value, then `isready`: the engine has no option to
/// set, so what matters is that the line is read at all and the readiness
/// handshake still answers.
#[cfg_attr(miri, ignore)]
#[test]
fn a_setoption_value_that_is_not_utf8_is_read_and_readiness_still_answers() {
    stage_configured_eval_dir();
    let mut session = b"setoption name EvalDir value C:\\".to_vec();
    session.extend_from_slice(CP932_A);
    session.extend_from_slice(b"\\eval\nisready\nquit\n");

    let out = drive_bytes(&session);
    let text = String::from_utf8_lossy(&out);
    assert!(
        text.contains("readyok"),
        "the session must ready after a non-UTF-8 setoption value, got: {text:?}"
    );
}

/// A `position` line whose move token carries such bytes is refused with the
/// diagnostic a malformed ASCII token gets, and the `go` after it still answers.
#[cfg_attr(miri, ignore)]
#[test]
fn a_non_ascii_position_token_is_refused_like_a_malformed_one_and_the_next_go_answers() {
    stage_configured_eval_dir();
    let mut session = b"usi\nisready\nposition startpos moves ".to_vec();
    session.extend_from_slice(CP932_A);
    session.extend_from_slice(b"\ngo\n");

    let out = drive_bytes(&session);
    // The refusal quotes the token as it arrived, so the transcript is not text.
    if cfg!(feature = "verbose1") {
        let mut expected = b"info string illegal move: ".to_vec();
        expected.extend_from_slice(CP932_A);
        expected.push(b'\n');
        assert!(
            out.windows(expected.len()).any(|w| w == expected),
            "expected the illegal-move refusal, got: {:?}",
            String::from_utf8_lossy(&out)
        );
    }
    let text = String::from_utf8_lossy(&out);
    assert!(
        text.contains("bestmove"),
        "the `go` after a refused position must still answer, got: {text:?}"
    );
}

/// The same line spelled with an ASCII token no move notation accepts takes the
/// same path, which is what "like a malformed one" means.
#[cfg_attr(miri, ignore)]
#[test]
fn a_malformed_ascii_position_token_is_refused_the_same_way() {
    stage_configured_eval_dir();
    let out = drive("usi\nisready\nposition startpos moves zzzz\ngo\n");
    if cfg!(feature = "verbose1") {
        assert!(
            out.contains("info string illegal move: zzzz\n"),
            "expected the illegal-move refusal, got: {out:?}"
        );
    }
    assert!(
        !bestmove_lines(&out).is_empty(),
        "the `go` after a refused position must still answer, got: {out:?}"
    );
}
