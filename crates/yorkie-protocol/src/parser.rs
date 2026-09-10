/// Input-validation limit: lines longer than this become
/// `Command::TooLong` and are not parsed further.
pub const MAX_LINE_BYTES: usize = 64 * 1024;

/// `position` command's first argument: either the implicit start position or
/// an explicit SFEN, whose four fields — board, side to move, hands, ply — are
/// borrowed from the command line rather than copied out of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PositionSfen<'a> {
    StartPos,
    Sfen([&'a str; 4]),
}

/// All USI `go` sub-tokens captured verbatim, including the ones the driver does
/// not act on, so the parse is lossless.
///
/// Six clauses are `verbose2`, together with the parser arms that fill them:
/// `depth`, `nodes`, `movetime`, `infinite`, `mate` and `rtime`. A `go` line is
/// their only source — the `DepthLimit` / `NodesLimit` config keys that also
/// seed the first two need the same feature — so without it no input could make
/// them anything but their default, and the fields carry the feature rather than
/// standing unfillable.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GoLimits {
    #[cfg(feature = "verbose2")]
    pub depth: Option<u32>,
    #[cfg(feature = "verbose2")]
    pub nodes: Option<u64>,
    #[cfg(feature = "verbose2")]
    pub movetime: Option<u64>,
    pub wtime: Option<u64>,
    pub btime: Option<u64>,
    pub winc: Option<u64>,
    pub binc: Option<u64>,
    pub byoyomi: Option<u64>,
    #[cfg(feature = "verbose2")]
    pub infinite: bool,
    /// `go ponder` — think on the predicted position; hold the reply until
    /// `ponderhit` or `stop`.
    pub ponder: bool,
    /// `go mate [ms|infinite]` — mate-search mode. In USI, unlike UCI, the
    /// token after `mate` is a time budget in milliseconds, not a move count.
    /// `Some(ms)` carries the budget, with [`MATE_UNLIMITED_MS`] standing for
    /// unlimited.
    #[cfg(feature = "verbose2")]
    pub mate: Option<u64>,
    /// `go rtime <ms>` — a randomised minimum-thinking-time budget used for
    /// self-play variety. `init_` seeds all three time bounds to `rtime` (plus
    /// a decaying random bump) and returns early. `None` means no `rtime`.
    #[cfg(feature = "verbose2")]
    pub rtime: Option<u64>,
}

/// The `go mate` unlimited-budget sentinel (`limits.mate = INT32_MAX`):
/// `go mate infinite` and a bare `go mate` both map here.
#[cfg(feature = "verbose2")]
pub const MATE_UNLIMITED_MS: u64 = i32::MAX as u64;

/// The `go` clauses that arrive with `verbose2`: everything here is analysis
/// or tooling, not the clock clauses and `ponder` a game bridge sends.
///
/// Without that feature these tokens are **rejected**, not ignored: silently
/// dropping the clause would turn `go depth 4` into an unbounded, clock-less
/// search in the middle of a game.
#[cfg(not(feature = "verbose2"))]
pub const EXTRA_GO_CLAUSES: [&str; 6] = ["depth", "nodes", "mate", "movetime", "infinite", "rtime"];

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Command<'a> {
    Usi,
    IsReady,
    SetOption {
        name: String,
        value: String,
    },
    UsiNewGame,
    Position {
        sfen: PositionSfen<'a>,
        /// The move tokens as they arrived, still one whitespace-separated run
        /// of the command line: the game path splits them where it replays
        /// them, so no move is ever copied out of the line. Empty when the
        /// command carried none.
        moves: &'a str,
    },
    Go(GoLimits),
    /// A `go` line carrying one of the [`EXTRA_GO_CLAUSES`], parsed by a build
    /// without `verbose2`. Holds the offending clause token so the driver can
    /// name it; **no search is started**. The variant exists only without that
    /// feature — with `verbose2`, every one of those clauses parses into
    /// [`Command::Go`].
    #[cfg(not(feature = "verbose2"))]
    GoExtraClause(String),
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
    Bench(Vec<String>),
    /// `tt <store|probe|children> …` — the verbosity-gated transposition-table
    /// read/write commands (`verbose3`). Like [`Command::Bench`] the trailing
    /// tokens are carried verbatim; [`crate::tt_command::parse_tt`] gives them
    /// meaning. The variant exists only with that feature, so a build without it
    /// cannot even name the command.
    #[cfg(feature = "verbose3")]
    Tt(Vec<String>),
    Quit,
    /// A line no arm recognised. The line text is retained only so the
    /// `verbose1` diagnostic can echo it back; without that feature nothing can
    /// print it, so the variant carries nothing and the text is never copied.
    /// Every construction goes through `unknown`, which is where the two shapes
    /// live.
    #[cfg(feature = "verbose1")]
    Unknown(String),
    #[cfg(not(feature = "verbose1"))]
    Unknown,
    TooLong,
}

/// [`Command::Unknown`] for `line`, carrying the text only where a build can
/// print it.
#[cfg(feature = "verbose1")]
fn unknown(line: &str) -> Command<'static> {
    Command::Unknown(line.to_string())
}

#[cfg(not(feature = "verbose1"))]
fn unknown(_line: &str) -> Command<'static> {
    Command::Unknown
}

/// [`Command::Unknown`] for a malformed `setoption`, whose reported text is the
/// re-joined line. The join exists only for the report, so a build that cannot
/// print one does not perform it.
#[cfg(feature = "verbose1")]
fn unknown_setoption(tokens: &[&str]) -> Command<'static> {
    Command::Unknown(format!("setoption {}", tokens.join(" ")))
}

#[cfg(not(feature = "verbose1"))]
fn unknown_setoption(_tokens: &[&str]) -> Command<'static> {
    Command::Unknown
}

/// Split off the leading token: what precedes the first run of whitespace, and
/// what follows that run. Both halves are empty once nothing is left, and the
/// tail keeps its own inner spacing, so a caller can hand it on whole.
fn split_token(s: &str) -> (&str, &str) {
    match s.find(char::is_whitespace) {
        Some(i) => (&s[..i], s[i..].trim_start()),
        None => (s, ""),
    }
}

pub fn parse_line(input: &str) -> Command<'_> {
    if input.len() > MAX_LINE_BYTES {
        return Command::TooLong;
    }
    let trimmed = input.trim_matches(|c: char| c == '\r' || c == '\n' || c.is_whitespace());
    if trimmed.is_empty() {
        return unknown("");
    }
    let (head, rest) = split_token(trimmed);
    let parts = rest.split_whitespace();
    match head {
        "usi" => Command::Usi,
        "isready" => Command::IsReady,
        "usinewgame" => Command::UsiNewGame,
        "quit" => Command::Quit,
        "setoption" => parse_setoption(parts),
        "position" => parse_position(trimmed, rest),
        "go" => parse_go(trimmed, rest),
        "stop" => Command::Stop,
        // `gameover [result]`: the trailing win/lose/draw token is optional and
        // ignored — the command is handled identically to `stop`.
        "gameover" => Command::GameOver,
        "ponderhit" => Command::PonderHit,
        // The trailing `bench` tokens are preserved verbatim for the semantic
        // parse in `crate::bench` (which fills defaults and validates them).
        // `verbose3` only: without that feature this arm does not exist and
        // `bench …` falls through to `Command::Unknown`, exactly like `tt`.
        #[cfg(feature = "verbose3")]
        "bench" => Command::Bench(parts.map(str::to_string).collect()),
        // `verbose3` only. Without that feature this arm does not exist, so
        // `tt …` falls through to `Command::Unknown` like any other unrecognised
        // line — the default build's behaviour is byte-identical to before the
        // command existed.
        #[cfg(feature = "verbose3")]
        "tt" => Command::Tt(parts.map(str::to_string).collect()),
        _ => unknown(trimmed),
    }
}

fn parse_position<'a>(line: &'a str, args: &'a str) -> Command<'a> {
    let (kind, rest) = split_token(args);
    let (sfen, after_sfen) = match kind {
        "startpos" => (PositionSfen::StartPos, rest),
        "sfen" => {
            // The four SFEN fields are: board, side-to-move, hands, ply. The
            // driver hands them to `yorkie_state::parse_sfen` and surfaces any
            // per-field error from there.
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
        return Command::Position { sfen, moves: "" };
    }
    let (keyword, moves) = split_token(after_sfen);
    if keyword != "moves" {
        return unknown(line);
    }
    Command::Position { sfen, moves }
}

/// The `u64` the clause's value token spells, or `None` when it is missing or
/// malformed.
fn u64_arg(value: &str) -> Option<u64> {
    value.parse::<u64>().ok()
}

fn parse_go<'a>(line: &'a str, args: &'a str) -> Command<'a> {
    let mut limits = GoLimits::default();
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
            return Command::GoExtraClause(key.to_string());
        }
        match key {
            #[cfg(feature = "verbose2")]
            "infinite" => {
                limits.infinite = true;
                rest = after_key;
            }
            "ponder" => {
                limits.ponder = true;
                rest = after_key;
            }
            // `go mate [ms|infinite]`: the token after `mate` is a millisecond
            // time budget; `infinite`, or nothing following, means unlimited.
            // Anything else that is not a valid `u64` is an error (the
            // reference's `stoi` would throw).
            #[cfg(feature = "verbose2")]
            "mate" => match value {
                "" => {
                    limits.mate = Some(MATE_UNLIMITED_MS);
                    rest = after_key;
                }
                "infinite" => {
                    limits.mate = Some(MATE_UNLIMITED_MS);
                    rest = after_value;
                }
                _ => {
                    let Ok(v) = value.parse::<u64>() else {
                        return unknown(line);
                    };
                    limits.mate = Some(v);
                    rest = after_value;
                }
            },
            #[cfg(feature = "verbose2")]
            "depth" => {
                let Ok(v) = value.parse::<u32>() else {
                    return unknown(line);
                };
                limits.depth = Some(v);
                rest = after_value;
            }
            #[cfg(feature = "verbose2")]
            "nodes" | "movetime" | "rtime" => {
                let Some(v) = u64_arg(value) else {
                    return unknown(line);
                };
                match key {
                    "nodes" => limits.nodes = Some(v),
                    "movetime" => limits.movetime = Some(v),
                    _ => limits.rtime = Some(v),
                }
                rest = after_value;
            }
            "wtime" | "btime" | "winc" | "binc" | "byoyomi" => {
                let Some(v) = u64_arg(value) else {
                    return unknown(line);
                };
                match key {
                    "wtime" => limits.wtime = Some(v),
                    "btime" => limits.btime = Some(v),
                    "winc" => limits.winc = Some(v),
                    "binc" => limits.binc = Some(v),
                    "byoyomi" => limits.byoyomi = Some(v),
                    _ => unreachable!("matched key {key} but no branch"),
                }
                rest = after_value;
            }
            _ => return unknown(line),
        }
    }
    Command::Go(limits)
}

fn parse_setoption<'a>(parts: impl Iterator<Item = &'a str>) -> Command<'static> {
    // USI: setoption name <NAME> [value <VALUE...>]
    // Per the protocol, NAME is a single token (option names contain no spaces),
    // and everything after `value` is the value (joined back with single spaces).
    let tokens: Vec<&str> = parts.collect();
    let mut iter = tokens.iter();
    let Some(&kw) = iter.next() else {
        return unknown_setoption(&tokens);
    };
    if kw != "name" {
        return unknown_setoption(&tokens);
    }
    let Some(&name) = iter.next() else {
        return unknown_setoption(&tokens);
    };
    let rest: Vec<&str> = iter.copied().collect();
    let value = match rest.as_slice() {
        [] => String::new(),
        ["value"] => String::new(),
        ["value", rest @ ..] => rest.join(" "),
        _ => return unknown_setoption(&tokens),
    };
    Command::SetOption {
        name: name.to_string(),
        value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_usi() {
        assert_eq!(parse_line("usi"), Command::Usi);
        assert_eq!(parse_line("usi\n"), Command::Usi);
        assert_eq!(parse_line("usi\r\n"), Command::Usi);
        assert_eq!(parse_line("  usi  "), Command::Usi);
    }

    #[test]
    fn parses_isready() {
        assert_eq!(parse_line("isready"), Command::IsReady);
    }

    #[test]
    fn parses_usinewgame() {
        assert_eq!(parse_line("usinewgame"), Command::UsiNewGame);
    }

    #[test]
    fn parses_quit() {
        assert_eq!(parse_line("quit"), Command::Quit);
    }

    #[test]
    fn parses_setoption_with_value() {
        assert_eq!(
            parse_line("setoption name USI_Hash value 1024"),
            Command::SetOption {
                name: "USI_Hash".to_string(),
                value: "1024".to_string(),
            }
        );
    }

    #[test]
    fn parses_setoption_with_empty_value() {
        assert_eq!(
            parse_line("setoption name EvalDir value"),
            Command::SetOption {
                name: "EvalDir".to_string(),
                value: String::new(),
            }
        );
    }

    #[test]
    fn parses_setoption_with_no_value_keyword() {
        assert_eq!(
            parse_line("setoption name UsiNewGameThing"),
            Command::SetOption {
                name: "UsiNewGameThing".to_string(),
                value: String::new(),
            }
        );
    }

    #[test]
    fn parses_setoption_with_multi_word_value() {
        assert_eq!(
            parse_line("setoption name EvalDir value /srv/eval dir/sub"),
            Command::SetOption {
                name: "EvalDir".to_string(),
                value: "/srv/eval dir/sub".to_string(),
            }
        );
    }

    /// The retained text is the `verbose1` half of the variant, so this pins it
    /// only where it exists; the rest of the unknown-line assertions compare
    /// against [`unknown`] and hold in every build.
    #[cfg(feature = "verbose1")]
    #[test]
    fn unknown_command_preserves_trimmed_line() {
        assert_eq!(
            parse_line("frobnicate the gizmo"),
            Command::Unknown("frobnicate the gizmo".to_string())
        );
    }

    /// The board / side-to-move / hands / ply fields of the initial position,
    /// as the four the `sfen` form is split into.
    const STARTPOS_FIELDS: [&str; 4] = [
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL",
        "b",
        "-",
        "1",
    ];

    #[test]
    fn parses_position_startpos() {
        assert_eq!(
            parse_line("position startpos"),
            Command::Position {
                sfen: PositionSfen::StartPos,
                moves: "",
            }
        );
    }

    #[test]
    fn parses_position_startpos_with_moves() {
        assert_eq!(
            parse_line("position startpos moves 7g7f 8c8d"),
            Command::Position {
                sfen: PositionSfen::StartPos,
                moves: "7g7f 8c8d",
            }
        );
    }

    /// The move tokens are handed on as the one run of line they arrived as,
    /// whatever spacing separated them, and a trailing `moves` with nothing
    /// after it is the same command as none at all.
    #[test]
    fn position_moves_are_the_line_run_whatever_its_spacing() {
        assert_eq!(
            parse_line("position   startpos   moves   7g7f\t8c8d  "),
            Command::Position {
                sfen: PositionSfen::StartPos,
                moves: "7g7f\t8c8d",
            }
        );
        assert_eq!(
            parse_line("position startpos moves"),
            Command::Position {
                sfen: PositionSfen::StartPos,
                moves: "",
            }
        );
    }

    #[test]
    fn parses_position_sfen_no_moves() {
        let sfen = STARTPOS_FIELDS.join(" ");
        assert_eq!(
            parse_line(&format!("position sfen {sfen}")),
            Command::Position {
                sfen: PositionSfen::Sfen(STARTPOS_FIELDS),
                moves: "",
            }
        );
    }

    #[test]
    fn parses_position_sfen_with_moves() {
        let sfen = STARTPOS_FIELDS.join(" ");
        assert_eq!(
            parse_line(&format!("position sfen {sfen} moves 7g7f")),
            Command::Position {
                sfen: PositionSfen::Sfen(STARTPOS_FIELDS),
                moves: "7g7f",
            }
        );
    }

    #[test]
    fn position_without_kind_token_is_unknown() {
        assert_eq!(parse_line("position"), unknown("position"));
        assert_eq!(
            parse_line("position something"),
            unknown("position something")
        );
    }

    #[test]
    fn position_sfen_short_field_count_is_unknown() {
        // Only three tokens (missing ply) → cannot form a valid SFEN.
        assert_eq!(
            parse_line("position sfen a b c"),
            unknown("position sfen a b c")
        );
    }

    #[test]
    fn parses_bare_go() {
        assert_eq!(parse_line("go"), Command::Go(GoLimits::default()));
    }

    #[cfg(feature = "verbose2")]
    #[test]
    fn parses_go_depth() {
        let expected = GoLimits {
            depth: Some(8),
            ..Default::default()
        };
        assert_eq!(parse_line("go depth 8"), Command::Go(expected));
    }

    #[cfg(feature = "verbose2")]
    #[test]
    fn parses_go_nodes_movetime_combined() {
        let expected = GoLimits {
            nodes: Some(1000),
            movetime: Some(250),
            ..Default::default()
        };
        assert_eq!(
            parse_line("go nodes 1000 movetime 250"),
            Command::Go(expected)
        );
    }

    #[cfg(feature = "verbose2")]
    #[test]
    fn parses_go_infinite() {
        let expected = GoLimits {
            infinite: true,
            ..Default::default()
        };
        assert_eq!(parse_line("go infinite"), Command::Go(expected));
    }

    #[test]
    fn parses_go_time_controls() {
        let expected = GoLimits {
            wtime: Some(60000),
            btime: Some(60000),
            byoyomi: Some(5000),
            ..Default::default()
        };
        assert_eq!(
            parse_line("go wtime 60000 btime 60000 byoyomi 5000"),
            Command::Go(expected)
        );
    }

    #[test]
    fn parses_go_winc_binc() {
        let expected = GoLimits {
            winc: Some(1000),
            binc: Some(2000),
            ..Default::default()
        };
        assert_eq!(parse_line("go winc 1000 binc 2000"), Command::Go(expected));
    }

    #[test]
    fn go_with_unknown_subtoken_is_unknown() {
        assert_eq!(
            parse_line("go searchmoves 7g7f"),
            unknown("go searchmoves 7g7f")
        );
    }

    #[cfg(feature = "verbose2")]
    #[test]
    fn go_with_missing_value_is_unknown() {
        assert_eq!(parse_line("go depth"), unknown("go depth"));
    }

    #[cfg(feature = "verbose2")]
    #[test]
    fn go_with_non_integer_value_is_unknown() {
        assert_eq!(
            parse_line("go nodes not-a-number"),
            unknown("go nodes not-a-number")
        );
    }

    #[test]
    fn parses_stop() {
        assert_eq!(parse_line("stop"), Command::Stop);
        assert_eq!(parse_line("stop\n"), Command::Stop);
    }

    #[test]
    fn parses_gameover_with_and_without_result() {
        assert_eq!(parse_line("gameover"), Command::GameOver);
        assert_eq!(parse_line("gameover win"), Command::GameOver);
        assert_eq!(parse_line("gameover lose"), Command::GameOver);
        assert_eq!(parse_line("gameover draw"), Command::GameOver);
        assert_eq!(parse_line("gameover\n"), Command::GameOver);
    }

    #[cfg(feature = "verbose3")]
    #[test]
    fn parses_bare_bench() {
        assert_eq!(parse_line("bench"), Command::Bench(Vec::new()));
    }

    #[cfg(feature = "verbose3")]
    #[test]
    fn parses_bench_with_all_tokens() {
        assert_eq!(
            parse_line("bench 16 1 6 default depth"),
            Command::Bench(
                ["16", "1", "6", "default", "depth"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect()
            )
        );
    }

    #[test]
    fn parses_ponderhit() {
        assert_eq!(parse_line("ponderhit"), Command::PonderHit);
        assert_eq!(parse_line("ponderhit\n"), Command::PonderHit);
    }

    #[test]
    fn parses_go_ponder() {
        let expected = GoLimits {
            ponder: true,
            ..Default::default()
        };
        assert_eq!(parse_line("go ponder"), Command::Go(expected));
    }

    #[test]
    fn parses_go_ponder_with_time() {
        let expected = GoLimits {
            ponder: true,
            btime: Some(1000),
            wtime: Some(1000),
            ..Default::default()
        };
        assert_eq!(
            parse_line("go ponder btime 1000 wtime 1000"),
            Command::Go(expected)
        );
    }

    #[cfg(feature = "verbose2")]
    #[test]
    fn parses_go_mate_with_budget() {
        let expected = GoLimits {
            mate: Some(5000),
            ..Default::default()
        };
        assert_eq!(parse_line("go mate 5000"), Command::Go(expected));
    }

    #[cfg(feature = "verbose2")]
    #[test]
    fn parses_go_mate_bare_is_unlimited() {
        let expected = GoLimits {
            mate: Some(MATE_UNLIMITED_MS),
            ..Default::default()
        };
        assert_eq!(parse_line("go mate"), Command::Go(expected));
    }

    #[cfg(feature = "verbose2")]
    #[test]
    fn parses_go_mate_infinite_is_unlimited() {
        let expected = GoLimits {
            mate: Some(MATE_UNLIMITED_MS),
            ..Default::default()
        };
        assert_eq!(parse_line("go mate infinite"), Command::Go(expected));
    }

    #[cfg(feature = "verbose2")]
    #[test]
    fn go_mate_non_integer_budget_is_unknown() {
        assert_eq!(parse_line("go mate soon"), unknown("go mate soon"));
    }

    /// Below `verbose3` — the default build: `bench` is not a command token at
    /// all, so it reaches the `Unknown` catch-all like any other stray line,
    /// exactly as `tt` does.
    #[cfg(not(feature = "verbose3"))]
    #[test]
    fn bench_is_not_a_command_below_verbose3() {
        assert_eq!(parse_line("bench"), unknown("bench"));
        assert_eq!(
            parse_line("bench 16 1 6 default depth"),
            unknown("bench 16 1 6 default depth")
        );
    }

    /// Below `verbose2`: every gated `go` clause is rejected by name — loudly,
    /// so a misconfigured harness cannot turn `go depth 4` into a clock-less
    /// search.
    #[cfg(not(feature = "verbose2"))]
    #[test]
    fn gated_go_clauses_are_rejected_below_verbose2() {
        for clause in EXTRA_GO_CLAUSES {
            assert_eq!(
                parse_line(&format!("go {clause} 4")),
                Command::GoExtraClause(clause.to_string()),
                "`go {clause} …` must be rejected by name"
            );
        }
        // Bare forms (no value) and clauses trailing a legitimate clock clause
        // are rejected the same way — the gate is checked before the clause is
        // interpreted, so a missing or malformed value cannot mask it.
        assert_eq!(
            parse_line("go infinite"),
            Command::GoExtraClause("infinite".to_string())
        );
        assert_eq!(
            parse_line("go mate"),
            Command::GoExtraClause("mate".to_string())
        );
        assert_eq!(
            parse_line("go depth"),
            Command::GoExtraClause("depth".to_string())
        );
        assert_eq!(
            parse_line("go nodes not-a-number"),
            Command::GoExtraClause("nodes".to_string())
        );
        assert_eq!(
            parse_line("go btime 1000 wtime 1000 depth 4"),
            Command::GoExtraClause("depth".to_string())
        );
    }

    /// Without `verbose2`: the match clauses are untouched — the tournament
    /// surface parses byte-identically to a build that has the feature.
    #[cfg(not(feature = "verbose2"))]
    #[test]
    fn match_go_clauses_still_parse_below_verbose2() {
        assert_eq!(parse_line("go"), Command::Go(GoLimits::default()));
        assert_eq!(
            parse_line("go btime 60000 wtime 60000 binc 1000 winc 1000 byoyomi 5000"),
            Command::Go(GoLimits {
                btime: Some(60000),
                wtime: Some(60000),
                binc: Some(1000),
                winc: Some(1000),
                byoyomi: Some(5000),
                ..Default::default()
            })
        );
        assert_eq!(
            parse_line("go ponder btime 1000 wtime 1000"),
            Command::Go(GoLimits {
                ponder: true,
                btime: Some(1000),
                wtime: Some(1000),
                ..Default::default()
            })
        );
        // A genuinely unknown sub-token is still `Unknown`, not a gate report.
        assert_eq!(
            parse_line("go searchmoves 7g7f"),
            unknown("go searchmoves 7g7f")
        );
    }

    /// Below `verbose3` — the default build: `tt` is not a command token at all,
    /// so it reaches the `Unknown` catch-all exactly like any other stray line.
    #[cfg(not(feature = "verbose3"))]
    #[test]
    fn tt_is_not_a_command_below_verbose3() {
        assert_eq!(
            parse_line("tt probe startpos"),
            unknown("tt probe startpos")
        );
        assert_eq!(parse_line("tt"), unknown("tt"));
    }

    /// At `verbose3`: `tt` splits into verbatim tokens for
    /// [`crate::tt_command::parse_tt`], mirroring how `bench` is handled.
    #[cfg(feature = "verbose3")]
    #[test]
    fn parses_tt_tokens_verbatim_at_verbose3() {
        assert_eq!(
            parse_line("tt probe startpos"),
            Command::Tt(vec!["probe".to_string(), "startpos".to_string()])
        );
        assert_eq!(parse_line("tt"), Command::Tt(Vec::new()));
    }

    #[test]
    fn empty_line_is_unknown_empty() {
        assert_eq!(parse_line(""), unknown(""));
        assert_eq!(parse_line("   \n"), unknown(""));
    }

    #[test]
    fn oversized_line_returns_too_long() {
        let line = "x".repeat(MAX_LINE_BYTES + 1);
        assert_eq!(parse_line(&line), Command::TooLong);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn line_at_max_size_is_parsed_normally() {
        // 64 KB exactly — still parses (becomes Unknown since it's not a command).
        let line = "x".repeat(MAX_LINE_BYTES);
        assert_eq!(parse_line(&line), unknown(&line));
    }
}
