//! The command surface a rated game actually uses, and what the default
//! (tournament) build does with the commands it does not.
//!
//! A bridge in a rated game sends only `usi`, `isready`, `setoption`,
//! `usinewgame`, `position`, `go` with clock clauses (`btime` / `wtime` /
//! `binc` / `winc` / `byoyomi`) or `go ponder`, `stop`, `ponderhit`, `gameover`
//! and `quit`. Everything else — `bench` and the `go depth` / `nodes` / `mate` /
//! `movetime` / `infinite` / `rtime` clauses — lives behind `usi-extras`.
//!
//! [`match_shaped_session_is_byte_identical`] is deliberately NOT feature-gated:
//! it runs in both configurations and pins the same bytes, which is the
//! "turning the feature off changes nothing a game can see" claim. The gated
//! module below compiles only with the feature OFF and pins the refusals; run
//! `cargo nextest run -p yorkie-protocol` (default features, i.e. without
//! `--all-features`) to execute it.
//!
//! The diagnostic lines here are composed through `common::diag_line`, so the
//! transcripts stay byte-exact under `info-diag` and without it — a default
//! build refuses the same commands and starts the same searches, it just says
//! nothing about it. What a game can see either way is the `bestmove` lines.
//!
//! No network is loaded, so every `go` resolves through the no-eval path — the
//! one search outcome that is deterministic to the byte. The handshake itself is
//! pinned separately in `tests/handshake.rs`.

mod common;

use common::{diag_line, drive};

/// The play part of a game-shaped session, byte-for-byte. Same expectation with
/// `usi-extras` on and off.
///
/// This is the transcript that must never move: from `usinewgame` onward, the
/// two builds emit the same bytes, and the compile-time configuration changed
/// none of them. The handshake in front of it — where the builds legitimately
/// differ — is pinned separately by [`whole_session_including_the_handshake`]
/// below and by `tests/handshake.rs`.
///
/// The no-network notice is a diagnostic, so it is present exactly when
/// `info-diag` is: without it the two `go`s answer `bestmove resign` and say
/// nothing else. The `bestmove` lines themselves are unconditional in every
/// build — that is the point of the gate.
fn play_output() -> String {
    let notice = diag_line("no eval network loaded; run isready");
    format!("{notice}bestmove resign\n{notice}bestmove resign\n")
}

/// The commands behind [`PLAY_OUTPUT`], minus the terminating `quit`.
const PLAY_SESSION: &str = "\
usinewgame\n\
position startpos moves 7g7f 3c3d\n\
go btime 60000 wtime 60000 binc 1000 winc 1000\n\
stop\n\
position startpos moves 7g7f 3c3d 2g2f 8c8d\n\
go btime 58000 wtime 58000 byoyomi 5000\n\
gameover lose\n";

#[cfg_attr(miri, ignore)]
#[test]
fn match_shaped_session_is_byte_identical() {
    assert_eq!(drive(&format!("{PLAY_SESSION}quit\n")), play_output());
}

/// The same session with the handshake a bridge actually sends in front of it:
/// `usi`, then a `setoption` (bridges send them whether or not the engine asked
/// for any).
///
/// Both builds reply identically: the `usi` reply advertises nothing, because
/// neither build has anything to advertise, and neither replies to the
/// `setoption` — USI asks for no reply, and there is no option to set. From
/// `usinewgame` on, both emit [`PLAY_OUTPUT`] — the same bytes, in the same
/// order, as before the settings were compiled in.
#[cfg_attr(miri, ignore)]
#[test]
fn whole_session_including_the_handshake() {
    let out = drive(&format!(
        "usi\nsetoption name USI_Hash value 256\n{PLAY_SESSION}quit\n"
    ));
    assert_eq!(
        out,
        format!(
            "id name Yorkie 3.1.0\n\
             id author Kei Ishida <ishida.kei@gmail.com>\n\
             usiok\n\
             {}",
            play_output()
        )
    );
}

/// `go ponder` is a match command too: the reply is held until the search is
/// released, and `ponderhit` releases it.
#[cfg_attr(miri, ignore)]
#[test]
fn go_ponder_and_ponderhit_are_match_commands() {
    let session = "\
        usinewgame\n\
        position startpos moves 7g7f\n\
        go ponder btime 60000 wtime 60000\n\
        ponderhit\n\
        quit\n";
    assert_eq!(
        drive(session),
        format!(
            "{}bestmove resign\n",
            diag_line("no eval network loaded; run isready")
        )
    );
}

/// The default build: the non-match commands are gone, and they go out loudly.
#[cfg(not(feature = "usi-extras"))]
mod without_usi_extras {
    use super::{diag_line, drive};

    /// `bench` is not a command token at all — it lands in the ordinary
    /// unknown-command path, exactly as `tt` does.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn bench_is_an_unknown_command() {
        assert_eq!(drive("bench\nquit\n"), diag_line("unknown command: bench"));
        assert_eq!(
            drive("bench 16 1 6 default depth\nquit\n"),
            diag_line("unknown command: bench 16 1 6 default depth")
        );
    }

    /// Every gated `go` clause is refused by name, and no search starts: no
    /// `bestmove`, no `info` beyond the one error line. Silently dropping the
    /// clause would turn `go depth 4` into an unbounded, clock-less search in
    /// the middle of a game.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn gated_go_clauses_are_refused_and_start_no_search() {
        for clause in [
            "depth 4",
            "nodes 1000",
            "mate 5000",
            "movetime 250",
            "rtime 100",
        ] {
            let name = clause.split_whitespace().next().expect("clause name");
            assert_eq!(
                drive(&format!("position startpos\ngo {clause}\nquit\n")),
                diag_line(&format!(
                    "go error: `{name}` requires a usi-extras build; no search started"
                )),
            );
        }
        assert_eq!(
            drive("position startpos\ngo infinite\nquit\n"),
            diag_line("go error: `infinite` requires a usi-extras build; no search started")
        );
    }

    /// A gated clause riding along with legitimate clock clauses is refused too:
    /// the `go` is not quietly downgraded to the subset this build understands.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn a_gated_clause_poisons_the_whole_go_line() {
        assert_eq!(
            drive("go btime 60000 wtime 60000 depth 4\nquit\n"),
            diag_line("go error: `depth` requires a usi-extras build; no search started")
        );
    }

    /// A refusal is not a wedge: the very next match-shaped `go` searches
    /// normally.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn a_refused_go_leaves_the_session_usable() {
        let session = "\
            usinewgame\n\
            position startpos\n\
            go depth 4\n\
            go btime 60000 wtime 60000 byoyomi 5000\n\
            quit\n";
        assert_eq!(
            drive(session),
            format!(
                "{}{}bestmove resign\n",
                diag_line("go error: `depth` requires a usi-extras build; no search started"),
                diag_line("no eval network loaded; run isready"),
            )
        );
    }
}
