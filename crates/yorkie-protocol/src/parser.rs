//! The USI command parser: one line of bytes in, one typed request out.
//!
//! What a `position` or a `go` line *means* — the start position and the moves
//! that reached it, the bounds on a search — is the engine's vocabulary, so the
//! types this fills are [`crate::engine`]'s. This module owns only the reading.
//!
//! A line is bytes. USI is ASCII by specification, so nothing here decodes a
//! character, and a line carrying bytes that are not valid UTF-8 — a `setoption`
//! value spelling a path in a Windows code page, say — is read like any other.
//! Such a token matches no keyword and parses as no number, so it fails exactly
//! where a malformed ASCII token fails and the session goes on.

#[cfg(feature = "verbose2")]
use crate::engine::MATE_UNLIMITED_MS;
use crate::engine::{GoParams, PositionSfen};
#[cfg(feature = "verbose2")]
use yorkie_state::text::atoi_u32;
use yorkie_state::text::{atoi_u64, split_token, trim_ascii_whitespace};

/// Input-validation limit: lines longer than this become
/// `Command::TooLong` and are not parsed further.
pub const MAX_LINE_BYTES: usize = 64 * 1024;

/// The `go` clauses that arrive with `verbose2`: everything here is analysis
/// or tooling, not the clock clauses and `ponder` a game bridge sends.
///
/// Without that feature these tokens are **rejected**, not ignored: silently
/// dropping the clause would turn `go depth 4` into an unbounded, clock-less
/// search in the middle of a game.
#[cfg(not(feature = "verbose2"))]
pub const EXTRA_GO_CLAUSES: [&[u8]; 6] = [
    b"depth",
    b"nodes",
    b"mate",
    b"movetime",
    b"infinite",
    b"rtime",
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Command<'a> {
    Usi,
    IsReady,
    SetOption {
        /// The option's name, borrowed from the command line.
        name: &'a [u8],
        /// Everything after the `value` keyword, still the one run of line it
        /// arrived as. Empty when the command named no value.
        value: &'a [u8],
    },
    UsiNewGame,
    Position {
        sfen: PositionSfen<'a>,
        /// The move tokens as they arrived, still one whitespace-separated run
        /// of the command line: the game path splits them where it replays
        /// them, so no move is ever copied out of the line. Empty when the
        /// command carried none.
        moves: &'a [u8],
    },
    Go(GoParams),
    /// A `go` line carrying one of the [`EXTRA_GO_CLAUSES`], parsed by a build
    /// without `verbose2`. Holds the offending clause token so the session can
    /// name it; **no search is started**. The variant exists only without that
    /// feature — with `verbose2`, every one of those clauses parses into
    /// [`Command::Go`].
    #[cfg(not(feature = "verbose2"))]
    GoExtraClause(&'a [u8]),
    Stop,
    /// `gameover [win|lose|draw]` — the game ended. The optional result token
    /// is ignored; the command is treated exactly like `stop`: over a shogi
    /// GUI, an opponent resign during `go ponder` arrives as `gameover` without
    /// a preceding `stop`, so it must release a held reply.
    GameOver,
    /// `ponderhit` — the opponent played the pondered move; commit the search.
    PonderHit,
    /// `bench [ttSizeMB] [threads] [limit] [default|current|<fenFile>]
    /// [limitType]` — the reproducible NPS benchmark. The raw trailing tokens
    /// are carried verbatim; [`crate::bench::parse_bench`] gives them meaning.
    ///
    /// `verbose3` only, so the default build cannot even name the command.
    #[cfg(feature = "verbose3")]
    Bench(Vec<&'a [u8]>),
    /// `tt <store|probe|children> …` — the verbosity-gated transposition-table
    /// read/write commands (`verbose3`). Like [`Command::Bench`] the trailing
    /// tokens are carried verbatim; [`crate::tt_command::parse_tt`] gives them
    /// meaning. The variant exists only with that feature, so a build without it
    /// cannot even name the command.
    #[cfg(feature = "verbose3")]
    Tt(Vec<&'a [u8]>),
    Quit,
    /// A line no arm recognised. The line is retained only so the `verbose1`
    /// diagnostic can echo it back; without that feature nothing can print it,
    /// so the variant carries nothing. Every construction goes through
    /// `unknown`, which is where the two shapes live.
    #[cfg(feature = "verbose1")]
    Unknown(&'a [u8]),
    #[cfg(not(feature = "verbose1"))]
    Unknown,
    TooLong,
}

/// [`Command::Unknown`] for `line`, carrying the line only where a build can
/// print it.
#[cfg(feature = "verbose1")]
fn unknown(line: &[u8]) -> Command<'_> {
    Command::Unknown(line)
}

#[cfg(not(feature = "verbose1"))]
fn unknown(_line: &[u8]) -> Command<'static> {
    Command::Unknown
}

/// The tokens of `args`, for the two `verbose3` commands that carry their
/// arguments to a parser of their own.
#[cfg(feature = "verbose3")]
fn tokens(args: &[u8]) -> Vec<&[u8]> {
    yorkie_state::text::tokens(args).collect()
}

pub fn parse_line(input: &[u8]) -> Command<'_> {
    if input.len() > MAX_LINE_BYTES {
        return Command::TooLong;
    }
    let trimmed = trim_ascii_whitespace(input);
    if trimmed.is_empty() {
        return unknown(b"");
    }
    let (head, rest) = split_token(trimmed);
    match head {
        b"usi" => Command::Usi,
        b"isready" => Command::IsReady,
        b"usinewgame" => Command::UsiNewGame,
        b"quit" => Command::Quit,
        b"setoption" => parse_setoption(trimmed, rest),
        b"position" => parse_position(trimmed, rest),
        b"go" => parse_go(trimmed, rest),
        b"stop" => Command::Stop,
        // `gameover [result]`: the trailing win/lose/draw token is optional and
        // ignored — the command is handled identically to `stop`.
        b"gameover" => Command::GameOver,
        b"ponderhit" => Command::PonderHit,
        // The trailing `bench` tokens are preserved verbatim for the semantic
        // parse in `crate::bench` (which fills defaults and validates them).
        // `verbose3` only: without that feature this arm does not exist and
        // `bench …` falls through to `Command::Unknown`, exactly like `tt`.
        #[cfg(feature = "verbose3")]
        b"bench" => Command::Bench(tokens(rest)),
        // `verbose3` only. Without that feature this arm does not exist, so
        // `tt …` falls through to `Command::Unknown` like any other unrecognised
        // line — the default build's behaviour is byte-identical to before the
        // command existed.
        #[cfg(feature = "verbose3")]
        b"tt" => Command::Tt(tokens(rest)),
        _ => unknown(trimmed),
    }
}

fn parse_position<'a>(line: &'a [u8], args: &'a [u8]) -> Command<'a> {
    let (kind, rest) = split_token(args);
    let (sfen, after_sfen) = match kind {
        b"startpos" => (PositionSfen::StartPos, rest),
        b"sfen" => {
            // The four SFEN fields are: board, side-to-move, hands, ply. The
            // engine hands them to `yorkie_state::parse_sfen_fields_into` and
            // surfaces any per-field error from there.
            let (board, rest) = split_token(rest);
            let (side_to_move, rest) = split_token(rest);
            let (hands, rest) = split_token(rest);
            let (ply, rest) = split_token(rest);
            if ply.is_empty() {
                return unknown(line);
            }
            (PositionSfen::Sfen([board, side_to_move, hands, ply]), rest)
        }
        _ => return unknown(line),
    };
    if after_sfen.is_empty() {
        return Command::Position { sfen, moves: b"" };
    }
    let (keyword, moves) = split_token(after_sfen);
    if keyword != b"moves" {
        return unknown(line);
    }
    Command::Position { sfen, moves }
}

fn parse_go<'a>(line: &'a [u8], args: &'a [u8]) -> Command<'a> {
    let mut limits = GoParams::default();
    let mut rest = args;
    while !rest.is_empty() {
        let (key, after_key) = split_token(rest);
        // The clause's value, for the arms that take one: empty where the line
        // ends after the keyword, which every one of them rejects.
        let (value, after_value) = split_token(after_key);
        // `verbose2` gate. Checked before the clause is interpreted, so a
        // gated clause is reported by name whatever follows it (including a
        // missing or malformed value, which would otherwise be `Unknown`). The
        // match clauses ahead of it in the line have already been consumed into
        // `limits`, which is then dropped on the floor — the caller starts no
        // search.
        #[cfg(not(feature = "verbose2"))]
        if EXTRA_GO_CLAUSES.contains(&key) {
            return Command::GoExtraClause(key);
        }
        match key {
            #[cfg(feature = "verbose2")]
            b"infinite" => {
                limits.infinite = true;
                rest = after_key;
            }
            b"ponder" => {
                limits.ponder = true;
                rest = after_key;
            }
            // `go mate [ms|infinite]`: the token after `mate` is a millisecond
            // time budget; `infinite`, or nothing following, means unlimited.
            // Anything else that is not a valid `u64` is an error (the
            // reference's `stoi` would throw).
            #[cfg(feature = "verbose2")]
            b"mate" => match value {
                b"" => {
                    limits.mate = Some(MATE_UNLIMITED_MS);
                    rest = after_key;
                }
                b"infinite" => {
                    limits.mate = Some(MATE_UNLIMITED_MS);
                    rest = after_value;
                }
                _ => {
                    let Some(v) = atoi_u64(value) else {
                        return unknown(line);
                    };
                    limits.mate = Some(v);
                    rest = after_value;
                }
            },
            #[cfg(feature = "verbose2")]
            b"depth" => {
                let Some(v) = atoi_u32(value) else {
                    return unknown(line);
                };
                limits.depth = Some(v);
                rest = after_value;
            }
            #[cfg(feature = "verbose2")]
            b"nodes" | b"movetime" | b"rtime" => {
                let Some(v) = atoi_u64(value) else {
                    return unknown(line);
                };
                match key {
                    b"nodes" => limits.nodes = Some(v),
                    b"movetime" => limits.movetime = Some(v),
                    _ => limits.rtime = Some(v),
                }
                rest = after_value;
            }
            b"wtime" | b"btime" | b"winc" | b"binc" | b"byoyomi" => {
                let Some(v) = atoi_u64(value) else {
                    return unknown(line);
                };
                match key {
                    b"wtime" => limits.wtime = Some(v),
                    b"btime" => limits.btime = Some(v),
                    b"winc" => limits.winc = Some(v),
                    b"binc" => limits.binc = Some(v),
                    b"byoyomi" => limits.byoyomi = Some(v),
                    _ => unreachable!("matched a clock clause but no branch"),
                }
                rest = after_value;
            }
            _ => return unknown(line),
        }
    }
    Command::Go(limits)
}

/// `setoption name <NAME> [value <VALUE…>]`.
///
/// Per the protocol, NAME is a single token (option names contain no spaces),
/// and everything after `value` is the value — handed on as the one run of line
/// it arrived as, spacing and all, rather than re-joined.
fn parse_setoption<'a>(line: &'a [u8], args: &'a [u8]) -> Command<'a> {
    let (keyword, rest) = split_token(args);
    if keyword != b"name" {
        return unknown(line);
    }
    let (name, rest) = split_token(rest);
    if name.is_empty() {
        return unknown(line);
    }
    if rest.is_empty() {
        return Command::SetOption { name, value: b"" };
    }
    let (keyword, value) = split_token(rest);
    if keyword != b"value" {
        return unknown(line);
    }
    Command::SetOption { name, value }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_usi() {
        assert_eq!(parse_line(b"usi"), Command::Usi);
        assert_eq!(parse_line(b"usi\n"), Command::Usi);
        assert_eq!(parse_line(b"usi\r\n"), Command::Usi);
        assert_eq!(parse_line(b"  usi  "), Command::Usi);
    }

    #[test]
    fn parses_isready() {
        assert_eq!(parse_line(b"isready"), Command::IsReady);
    }

    #[test]
    fn parses_usinewgame() {
        assert_eq!(parse_line(b"usinewgame"), Command::UsiNewGame);
    }

    #[test]
    fn parses_quit() {
        assert_eq!(parse_line(b"quit"), Command::Quit);
    }

    #[test]
    fn parses_setoption_with_value() {
        assert_eq!(
            parse_line(b"setoption name USI_Hash value 1024"),
            Command::SetOption {
                name: b"USI_Hash",
                value: b"1024",
            }
        );
    }

    #[test]
    fn parses_setoption_with_empty_value() {
        assert_eq!(
            parse_line(b"setoption name EvalDir value"),
            Command::SetOption {
                name: b"EvalDir",
                value: b"",
            }
        );
    }

    #[test]
    fn parses_setoption_with_no_value_keyword() {
        assert_eq!(
            parse_line(b"setoption name UsiNewGameThing"),
            Command::SetOption {
                name: b"UsiNewGameThing",
                value: b"",
            }
        );
    }

    #[test]
    fn parses_setoption_with_multi_word_value() {
        assert_eq!(
            parse_line(b"setoption name EvalDir value /srv/eval dir/sub"),
            Command::SetOption {
                name: b"EvalDir",
                value: b"/srv/eval dir/sub",
            }
        );
    }

    /// A `setoption` value carrying bytes no USI command is spelled in — a path
    /// in a Windows code page, whose `\x82\xa0` is not valid UTF-8 — parses like
    /// any other value. Reading the line is what used to fail here, and a
    /// failed read ended the session.
    #[test]
    fn parses_setoption_whose_value_is_not_utf8() {
        assert_eq!(
            parse_line(b"setoption name EvalDir value C:\\\x82\xa0\\eval"),
            Command::SetOption {
                name: b"EvalDir",
                value: b"C:\\\x82\xa0\\eval",
            }
        );
    }

    #[test]
    fn a_malformed_setoption_is_unknown() {
        assert_eq!(parse_line(b"setoption"), unknown(b"setoption"));
        assert_eq!(
            parse_line(b"setoption nombre X"),
            unknown(b"setoption nombre X")
        );
        assert_eq!(parse_line(b"setoption name"), unknown(b"setoption name"));
        assert_eq!(
            parse_line(b"setoption name X worth 1"),
            unknown(b"setoption name X worth 1")
        );
    }

    /// The retained line is the `verbose1` half of the variant, so this pins it
    /// only where it exists; the rest of the unknown-line assertions compare
    /// against [`unknown`] and hold in every build.
    #[cfg(feature = "verbose1")]
    #[test]
    fn unknown_command_preserves_trimmed_line() {
        assert_eq!(
            parse_line(b"frobnicate the gizmo"),
            Command::Unknown(b"frobnicate the gizmo")
        );
    }

    /// The board / side-to-move / hands / ply fields of the initial position,
    /// as the four the `sfen` form is split into.
    const STARTPOS_FIELDS: [&[u8]; 4] = [
        b"lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL",
        b"b",
        b"-",
        b"1",
    ];

    #[test]
    fn parses_position_startpos() {
        assert_eq!(
            parse_line(b"position startpos"),
            Command::Position {
                sfen: PositionSfen::StartPos,
                moves: b"",
            }
        );
    }

    #[test]
    fn parses_position_startpos_with_moves() {
        assert_eq!(
            parse_line(b"position startpos moves 7g7f 8c8d"),
            Command::Position {
                sfen: PositionSfen::StartPos,
                moves: b"7g7f 8c8d",
            }
        );
    }

    /// The move tokens are handed on as the one run of line they arrived as,
    /// whatever spacing separated them, and a trailing `moves` with nothing
    /// after it is the same command as none at all.
    #[test]
    fn position_moves_are_the_line_run_whatever_its_spacing() {
        assert_eq!(
            parse_line(b"position   startpos   moves   7g7f\t8c8d  "),
            Command::Position {
                sfen: PositionSfen::StartPos,
                moves: b"7g7f\t8c8d",
            }
        );
        assert_eq!(
            parse_line(b"position startpos moves"),
            Command::Position {
                sfen: PositionSfen::StartPos,
                moves: b"",
            }
        );
    }

    #[test]
    fn parses_position_sfen_no_moves() {
        assert_eq!(
            parse_line(
                b"position sfen lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
            ),
            Command::Position {
                sfen: PositionSfen::Sfen(STARTPOS_FIELDS),
                moves: b"",
            }
        );
    }

    #[test]
    fn parses_position_sfen_with_moves() {
        assert_eq!(
            parse_line(
                b"position sfen lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1 \
                  moves 7g7f"
            ),
            Command::Position {
                sfen: PositionSfen::Sfen(STARTPOS_FIELDS),
                moves: b"7g7f",
            }
        );
    }

    #[test]
    fn position_without_kind_token_is_unknown() {
        assert_eq!(parse_line(b"position"), unknown(b"position"));
        assert_eq!(
            parse_line(b"position something"),
            unknown(b"position something")
        );
    }

    #[test]
    fn position_sfen_short_field_count_is_unknown() {
        // Only three tokens (missing ply) → cannot form a valid SFEN.
        assert_eq!(
            parse_line(b"position sfen a b c"),
            unknown(b"position sfen a b c")
        );
    }

    /// A `position` token carrying a byte no USI line is spelled in takes the
    /// path a malformed ASCII token takes: the keyword matches nothing, so the
    /// line is unknown, and the SFEN fields reach the one parser that judges
    /// them.
    #[test]
    fn a_non_ascii_position_token_takes_the_malformed_path() {
        assert_eq!(
            parse_line(b"position \x82\xa0"),
            unknown(b"position \x82\xa0")
        );
        assert_eq!(
            parse_line(b"position startpos moves \x82\xa0"),
            Command::Position {
                sfen: PositionSfen::StartPos,
                moves: b"\x82\xa0",
            }
        );
    }

    #[test]
    fn parses_bare_go() {
        assert_eq!(parse_line(b"go"), Command::Go(GoParams::default()));
    }

    #[cfg(feature = "verbose2")]
    #[test]
    fn parses_go_depth() {
        let expected = GoParams {
            depth: Some(8),
            ..Default::default()
        };
        assert_eq!(parse_line(b"go depth 8"), Command::Go(expected));
    }

    #[cfg(feature = "verbose2")]
    #[test]
    fn parses_go_nodes_movetime_combined() {
        let expected = GoParams {
            nodes: Some(1000),
            movetime: Some(250),
            ..Default::default()
        };
        assert_eq!(
            parse_line(b"go nodes 1000 movetime 250"),
            Command::Go(expected)
        );
    }

    #[cfg(feature = "verbose2")]
    #[test]
    fn parses_go_infinite() {
        let expected = GoParams {
            infinite: true,
            ..Default::default()
        };
        assert_eq!(parse_line(b"go infinite"), Command::Go(expected));
    }

    #[test]
    fn parses_go_time_controls() {
        let expected = GoParams {
            wtime: Some(60000),
            btime: Some(60000),
            byoyomi: Some(5000),
            ..Default::default()
        };
        assert_eq!(
            parse_line(b"go wtime 60000 btime 60000 byoyomi 5000"),
            Command::Go(expected)
        );
    }

    #[test]
    fn parses_go_winc_binc() {
        let expected = GoParams {
            winc: Some(1000),
            binc: Some(2000),
            ..Default::default()
        };
        assert_eq!(parse_line(b"go winc 1000 binc 2000"), Command::Go(expected));
    }

    #[test]
    fn go_with_unknown_subtoken_is_unknown() {
        assert_eq!(
            parse_line(b"go searchmoves 7g7f"),
            unknown(b"go searchmoves 7g7f")
        );
    }

    /// A `go` clause spelled in bytes no USI line carries matches no keyword,
    /// which is the unknown-clause path.
    #[test]
    fn a_non_ascii_go_clause_is_unknown() {
        assert_eq!(
            parse_line(b"go \x82\xa0 1000"),
            unknown(b"go \x82\xa0 1000")
        );
    }

    /// A clock clause whose value carries such a byte parses as no number,
    /// which is the malformed-value path.
    #[test]
    fn a_non_ascii_clock_value_is_unknown() {
        assert_eq!(
            parse_line(b"go btime \x82\xa0"),
            unknown(b"go btime \x82\xa0")
        );
    }

    #[cfg(feature = "verbose2")]
    #[test]
    fn go_with_missing_value_is_unknown() {
        assert_eq!(parse_line(b"go depth"), unknown(b"go depth"));
    }

    #[cfg(feature = "verbose2")]
    #[test]
    fn go_with_non_integer_value_is_unknown() {
        assert_eq!(
            parse_line(b"go nodes not-a-number"),
            unknown(b"go nodes not-a-number")
        );
    }

    #[test]
    fn parses_stop() {
        assert_eq!(parse_line(b"stop"), Command::Stop);
        assert_eq!(parse_line(b"stop\n"), Command::Stop);
    }

    #[test]
    fn parses_gameover_with_and_without_result() {
        assert_eq!(parse_line(b"gameover"), Command::GameOver);
        assert_eq!(parse_line(b"gameover win"), Command::GameOver);
        assert_eq!(parse_line(b"gameover lose"), Command::GameOver);
        assert_eq!(parse_line(b"gameover draw"), Command::GameOver);
        assert_eq!(parse_line(b"gameover\n"), Command::GameOver);
    }

    #[cfg(feature = "verbose3")]
    #[test]
    fn parses_bare_bench() {
        assert_eq!(parse_line(b"bench"), Command::Bench(Vec::new()));
    }

    #[cfg(feature = "verbose3")]
    #[test]
    fn parses_bench_with_all_tokens() {
        assert_eq!(
            parse_line(b"bench 16 1 6 default depth"),
            Command::Bench(vec![
                &b"16"[..],
                &b"1"[..],
                &b"6"[..],
                &b"default"[..],
                &b"depth"[..]
            ])
        );
    }

    #[test]
    fn parses_ponderhit() {
        assert_eq!(parse_line(b"ponderhit"), Command::PonderHit);
        assert_eq!(parse_line(b"ponderhit\n"), Command::PonderHit);
    }

    #[test]
    fn parses_go_ponder() {
        let expected = GoParams {
            ponder: true,
            ..Default::default()
        };
        assert_eq!(parse_line(b"go ponder"), Command::Go(expected));
    }

    #[test]
    fn parses_go_ponder_with_time() {
        let expected = GoParams {
            ponder: true,
            btime: Some(1000),
            wtime: Some(1000),
            ..Default::default()
        };
        assert_eq!(
            parse_line(b"go ponder btime 1000 wtime 1000"),
            Command::Go(expected)
        );
    }

    #[cfg(feature = "verbose2")]
    #[test]
    fn parses_go_mate_with_budget() {
        let expected = GoParams {
            mate: Some(5000),
            ..Default::default()
        };
        assert_eq!(parse_line(b"go mate 5000"), Command::Go(expected));
    }

    #[cfg(feature = "verbose2")]
    #[test]
    fn parses_go_mate_bare_is_unlimited() {
        let expected = GoParams {
            mate: Some(MATE_UNLIMITED_MS),
            ..Default::default()
        };
        assert_eq!(parse_line(b"go mate"), Command::Go(expected));
    }

    #[cfg(feature = "verbose2")]
    #[test]
    fn parses_go_mate_infinite_is_unlimited() {
        let expected = GoParams {
            mate: Some(MATE_UNLIMITED_MS),
            ..Default::default()
        };
        assert_eq!(parse_line(b"go mate infinite"), Command::Go(expected));
    }

    #[cfg(feature = "verbose2")]
    #[test]
    fn go_mate_non_integer_budget_is_unknown() {
        assert_eq!(parse_line(b"go mate soon"), unknown(b"go mate soon"));
    }

    /// Below `verbose3` — the default build: `bench` is not a command token at
    /// all, so it reaches the `Unknown` catch-all like any other stray line,
    /// exactly as `tt` does.
    #[cfg(not(feature = "verbose3"))]
    #[test]
    fn bench_is_not_a_command_below_verbose3() {
        assert_eq!(parse_line(b"bench"), unknown(b"bench"));
        assert_eq!(
            parse_line(b"bench 16 1 6 default depth"),
            unknown(b"bench 16 1 6 default depth")
        );
    }

    /// Below `verbose2`: every gated `go` clause is rejected by name — loudly,
    /// so a misconfigured harness cannot turn `go depth 4` into a clock-less
    /// search.
    #[cfg(not(feature = "verbose2"))]
    #[test]
    fn gated_go_clauses_are_rejected_below_verbose2() {
        for clause in EXTRA_GO_CLAUSES {
            let mut line = b"go ".to_vec();
            line.extend_from_slice(clause);
            line.extend_from_slice(b" 4");
            assert_eq!(
                parse_line(&line),
                Command::GoExtraClause(clause),
                "`go {clause:?} …` must be rejected by name"
            );
        }
        // Bare forms (no value) and clauses trailing a legitimate clock clause
        // are rejected the same way — the gate is checked before the clause is
        // interpreted, so a missing or malformed value cannot mask it.
        assert_eq!(
            parse_line(b"go infinite"),
            Command::GoExtraClause(b"infinite")
        );
        assert_eq!(parse_line(b"go mate"), Command::GoExtraClause(b"mate"));
        assert_eq!(parse_line(b"go depth"), Command::GoExtraClause(b"depth"));
        assert_eq!(
            parse_line(b"go nodes not-a-number"),
            Command::GoExtraClause(b"nodes")
        );
        assert_eq!(
            parse_line(b"go btime 1000 wtime 1000 depth 4"),
            Command::GoExtraClause(b"depth")
        );
    }

    /// Without `verbose2`: the match clauses are untouched — the tournament
    /// surface parses byte-identically to a build that has the feature.
    #[cfg(not(feature = "verbose2"))]
    #[test]
    fn match_go_clauses_still_parse_below_verbose2() {
        assert_eq!(parse_line(b"go"), Command::Go(GoParams::default()));
        assert_eq!(
            parse_line(b"go btime 60000 wtime 60000 binc 1000 winc 1000 byoyomi 5000"),
            Command::Go(GoParams {
                btime: Some(60000),
                wtime: Some(60000),
                binc: Some(1000),
                winc: Some(1000),
                byoyomi: Some(5000),
                ..Default::default()
            })
        );
        assert_eq!(
            parse_line(b"go ponder btime 1000 wtime 1000"),
            Command::Go(GoParams {
                ponder: true,
                btime: Some(1000),
                wtime: Some(1000),
                ..Default::default()
            })
        );
        // A genuinely unknown sub-token is still `Unknown`, not a gate report.
        assert_eq!(
            parse_line(b"go searchmoves 7g7f"),
            unknown(b"go searchmoves 7g7f")
        );
    }

    /// Below `verbose3` — the default build: `tt` is not a command token at all,
    /// so it reaches the `Unknown` catch-all exactly like any other stray line.
    #[cfg(not(feature = "verbose3"))]
    #[test]
    fn tt_is_not_a_command_below_verbose3() {
        assert_eq!(
            parse_line(b"tt probe startpos"),
            unknown(b"tt probe startpos")
        );
        assert_eq!(parse_line(b"tt"), unknown(b"tt"));
    }

    /// At `verbose3`: `tt` splits into verbatim tokens for
    /// [`crate::tt_command::parse_tt`], mirroring how `bench` is handled.
    #[cfg(feature = "verbose3")]
    #[test]
    fn parses_tt_tokens_verbatim_at_verbose3() {
        assert_eq!(
            parse_line(b"tt probe startpos"),
            Command::Tt(vec![&b"probe"[..], &b"startpos"[..]])
        );
        assert_eq!(parse_line(b"tt"), Command::Tt(Vec::new()));
    }

    #[test]
    fn empty_line_is_unknown_empty() {
        assert_eq!(parse_line(b""), unknown(b""));
        assert_eq!(parse_line(b"   \n"), unknown(b""));
    }

    #[test]
    fn oversized_line_returns_too_long() {
        let line = vec![b'x'; MAX_LINE_BYTES + 1];
        assert_eq!(parse_line(&line), Command::TooLong);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn line_at_max_size_is_parsed_normally() {
        // 64 KB exactly — still parses (becomes Unknown since it's not a command).
        let line = vec![b'x'; MAX_LINE_BYTES];
        assert_eq!(parse_line(&line), unknown(&line));
    }
}
