//! The `verbose3` `tt` command family: the *semantic parse* of the argument
//! tokens behind `tt store` / `tt probe` / `tt children`.
//!
//! The whole module exists only under the `verbose3` cargo feature. With it
//! off, [`crate::parser::parse_line`] does not know the `tt` token at all and
//! nothing here is compiled.
//!
//! There is no upstream YaneuraOu precedent for these commands — the reference
//! exposes no TT read/write USI command — so the syntax below is this project's
//! own design.
//!
//! This module owns only what can be decided from the tokens alone: the
//! subcommand, the position clause kept as the four fields it arrived as, the
//! scalar fields, and their range validation. Anything needing a `Position` is
//! the command's.
//!
//! Every value on the command surface is expressed relative to the named
//! position as root, which is also how the transposition table stores it: the
//! reference's `value_to_tt(v, ply)` shifts a mate score by the node's distance
//! from the root, so a *stored* mate value is position-absolute. The named
//! position therefore sits at `ply == 0`, and `tt children` reports its children
//! at `ply == 1`, one ply further out than a `tt probe` of the same child SFEN
//! would.
//!
//! Centipawn arguments and output use the reference USI scale, so `tt` speaks
//! the same numbers as an `info … score cp N` line. That mapping is lossy in
//! both directions, so a `cp` round trip quantises; `mate` arguments are exact.
//!
//! The entry's path-dependence mark is on the surface too: `tt store` takes an
//! optional `pathdep <0|1>` (`0` when omitted) and `tt probe` / `tt children`
//! report it, so an entry's mark can be written and read from outside without
//! a search having to produce it.

use yorkie_state::TextWriter;
use yorkie_state::text::atoi_i64;
use yorkie_storage::{Bound, DEPTH_NONE, Depth, Value};

use crate::engine::PAWN_VALUE;
use crate::usi::{VALUE_MATE, VALUE_TB_WIN_IN_MAX_PLY};

/// Largest mate distance the value encoding can carry: `VALUE_MATE - n` must
/// stay decisive (`|v| >= VALUE_TB_WIN_IN_MAX_PLY`), and that threshold is
/// `VALUE_MATE - MAX_PLY` with the reference's `MAX_PLY == 246`.
pub const MAX_MATE_DISTANCE: i64 = (VALUE_MATE - VALUE_TB_WIN_IN_MAX_PLY) as i64;

/// Smallest `depth` an entry can carry. [`yorkie_storage::tt`] stores
/// `depth8 = depth - DEPTH_NONE` in a `u8`, and `depth8 == 0` marks an
/// unoccupied entry, so the representable range is `DEPTH_NONE + 1 ..=
/// DEPTH_NONE + 255`. Out-of-range depths are rejected here rather than
/// reaching the `debug_assert!`s in `TTEntry::save`.
pub const MIN_STORE_DEPTH: Depth = DEPTH_NONE + 1;
/// Largest `depth` an entry can carry (see [`MIN_STORE_DEPTH`]).
pub const MAX_STORE_DEPTH: Depth = DEPTH_NONE + 255;

/// Which clause of the grammar a failure is about, so a message can name it
/// without carrying a sentence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtClause {
    Position,
    Move,
    Value,
    ValueMate,
    ValueCp,
    Depth,
    Bound,
    Eval,
    EvalCp,
    PathDep,
}

impl TtClause {
    /// This clause's spelling, as the command line writes it.
    fn name(self) -> &'static [u8] {
        match self {
            Self::Position => b"position",
            Self::Move => b"move",
            Self::Value => b"value",
            Self::ValueMate => b"value mate",
            Self::ValueCp => b"value cp",
            Self::Depth => b"depth",
            Self::Bound => b"bound",
            Self::Eval => b"eval",
            Self::EvalCp => b"eval cp",
            Self::PathDep => b"pathdep",
        }
    }

    /// The clause as a `tt store …` line names it when it is missing, which
    /// spells the operand it wants too.
    fn required_form(self) -> &'static [u8] {
        match self {
            Self::Position => b"position clause (`sfen \xE2\x80\xA6` or `startpos`)",
            Self::Move => b"move <usi-move|none>",
            Self::Value | Self::ValueMate | Self::ValueCp => b"value <cp|mate <n>>",
            Self::Depth => b"depth <d>",
            Self::Bound => b"bound <exact|lower|upper>",
            Self::Eval | Self::EvalCp => b"eval <cp>",
            Self::PathDep => b"pathdep <0|1>",
        }
    }
}

/// A `tt` argument-parse failure, surfaced by the command as one
/// `info string tt error: <msg>` line so a garbage argument fails loudly
/// without panicking.
///
/// Each variant carries what it refused — borrowed from the command line, so
/// nothing is copied out of it — and [`Self::write_message`] spells it where
/// the notice is written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtParseError<'a> {
    MissingSubcommand,
    UnknownSubcommand(&'a [u8]),
    /// `probe` / `children` took a position clause and something followed it.
    TrailingTokens {
        subcommand: &'a [u8],
        first: &'a [u8],
    },
    ShortSfenClause,
    UnknownPositionClause(&'a [u8]),
    MissingPositionClause,
    MissingOperand(TtClause),
    NotAnInteger {
        clause: TtClause,
        token: &'a [u8],
    },
    DepthOutOfRange(i64),
    UnknownBound(&'a [u8]),
    BadPathDep(&'a [u8]),
    UnexpectedToken(&'a [u8]),
    DuplicateClause(TtClause),
    MissingClause(TtClause),
    /// A centipawn argument that maps into the decisive band, where it would
    /// read back as a mate score instead of a centipawn one.
    DecisiveCp {
        cp: i64,
        value: i64,
    },
    MateOutOfRange(i64),
}

impl TtParseError<'_> {
    /// Append this failure's message to `out`.
    pub fn write_message(&self, out: &mut TextWriter<'_>) {
        match self {
            Self::MissingSubcommand => {
                out.bytes(b"missing subcommand; expected `store`, `probe` or `children`");
            }
            Self::UnknownSubcommand(token) => {
                out.bytes(b"unknown subcommand `")
                    .bytes(token)
                    .bytes(b"`; expected `store`, `probe` or `children`");
            }
            Self::TrailingTokens { subcommand, first } => {
                out.bytes(b"`")
                    .bytes(subcommand)
                    .bytes(b"` takes only a position clause; unexpected trailing `")
                    .bytes(first)
                    .bytes(b"`");
            }
            Self::ShortSfenClause => {
                out.bytes(b"`sfen` needs its four fields (board, side-to-move, hands, ply)");
            }
            Self::UnknownPositionClause(token) => {
                out.bytes(b"expected `sfen <board> <side> <hands> <ply>` or `startpos`, found `")
                    .bytes(token)
                    .bytes(b"`");
            }
            Self::MissingPositionClause => {
                out.bytes(b"missing position clause (`sfen \xE2\x80\xA6` or `startpos`)");
            }
            Self::MissingOperand(clause) => {
                out.bytes(b"`")
                    .bytes(clause.name())
                    .bytes(b"` needs an argument");
            }
            Self::NotAnInteger { clause, token } => {
                out.bytes(b"`")
                    .bytes(clause.name())
                    .bytes(b"` argument `")
                    .bytes(token)
                    .bytes(b"` is not an integer");
            }
            Self::DepthOutOfRange(depth) => {
                out.bytes(b"depth ")
                    .i64(*depth)
                    .bytes(b" out of range; an entry stores ")
                    .i64(i64::from(MIN_STORE_DEPTH))
                    .bytes(b"..=")
                    .i64(i64::from(MAX_STORE_DEPTH));
            }
            Self::UnknownBound(token) => {
                out.bytes(b"unknown bound `")
                    .bytes(token)
                    .bytes(b"`; expected `exact`, `lower` or `upper`");
            }
            Self::BadPathDep(token) => {
                out.bytes(b"`pathdep` takes `0` or `1`, found `")
                    .bytes(token)
                    .bytes(b"`");
            }
            Self::UnexpectedToken(token) => {
                out.bytes(b"unexpected token `")
                    .bytes(token)
                    .bytes(b"` in `tt store`");
            }
            Self::DuplicateClause(clause) => {
                out.bytes(b"duplicate `")
                    .bytes(clause.name())
                    .bytes(b"` clause");
            }
            Self::MissingClause(clause) => {
                out.bytes(b"`tt store` is missing the `")
                    .bytes(clause.required_form())
                    .bytes(b"` clause");
            }
            Self::DecisiveCp { cp, value } => {
                out.bytes(b"cp ")
                    .i64(*cp)
                    .bytes(b" maps to internal value ")
                    .i64(*value)
                    .bytes(b", which is a decisive score; use `mate <n>` for mate values");
            }
            Self::MateOutOfRange(n) => {
                out.bytes(b"mate ")
                    .i64(*n)
                    .bytes(b" out of range; |n| must be <= ")
                    .i64(MAX_MATE_DISTANCE);
            }
        }
    }
}

/// The position clause, kept as the four fields it arrived as: the command
/// turns them into a `Position` so SFEN diagnostics come from the one parser
/// the `position` command uses.
///
/// `startpos` is a shorthand for the four-field `sfen` clause, spelled out here
/// rather than expanded so the command can use `Position::startpos()` directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtPosition<'a> {
    StartPos,
    /// The four SFEN fields (board, side-to-move, hands, ply), exactly as
    /// [`crate::engine::PositionSfen::Sfen`] carries them.
    Sfen([&'a [u8]; 4]),
}

/// A fully validated `tt store` invocation. Every numeric field is already in
/// the engine's internal units and in range for the entry encoding; the only
/// thing left unresolved is [`Self::mv`], which needs the position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TtStoreArgs<'a> {
    pub position: TtPosition<'a>,
    /// The raw `move` token — a USI move, or the literal `none` for "no move"
    /// (fragment `0`, which `TTEntry::save` treats as "keep whatever move the
    /// entry already had for this position").
    pub mv: &'a [u8],
    /// Search value in internal units, relative to [`Self::position`] as root.
    pub value: Value,
    pub depth: Depth,
    pub bound: Bound,
    /// Static eval in internal units (the entry's `eval16` field).
    pub eval: Value,
    pub pv: bool,
    /// The entry's path-dependence mark (`false` when the clause is omitted).
    pub path_dep: bool,
}

/// One parsed `tt` invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtCommand<'a> {
    Store(TtStoreArgs<'a>),
    Probe(TtPosition<'a>),
    Children(TtPosition<'a>),
}

/// Parse the tokens following the `tt` keyword.
///
/// The clauses are keyword-driven and accepted in any order. A missing mandatory
/// clause, a duplicate clause, an unknown token, or an out-of-range number is an
/// error; this function never panics.
pub fn parse_tt<'a>(tokens: &[&'a [u8]]) -> Result<TtCommand<'a>, TtParseError<'a>> {
    let Some((&sub, rest)) = tokens.split_first() else {
        return Err(TtParseError::MissingSubcommand);
    };
    match sub {
        b"store" => parse_store(rest).map(TtCommand::Store),
        b"probe" => parse_lone_position(rest, b"probe").map(TtCommand::Probe),
        b"children" => parse_lone_position(rest, b"children").map(TtCommand::Children),
        other => Err(TtParseError::UnknownSubcommand(other)),
    }
}

/// The whole argument list of `tt probe` / `tt children`: one position clause
/// and nothing else.
fn parse_lone_position<'a>(
    tokens: &[&'a [u8]],
    sub: &'a [u8],
) -> Result<TtPosition<'a>, TtParseError<'a>> {
    let (position, rest) = parse_position_clause(tokens)?;
    if let Some(&first) = rest.first() {
        return Err(TtParseError::TrailingTokens {
            subcommand: sub,
            first,
        });
    }
    Ok(position)
}

/// Consume a leading `sfen <board> <side> <hands> <ply>` or `startpos`,
/// returning the clause and the tokens after it.
fn parse_position_clause<'a, 't>(
    tokens: &'t [&'a [u8]],
) -> Result<(TtPosition<'a>, &'t [&'a [u8]]), TtParseError<'a>> {
    match tokens.split_first() {
        Some((&b"startpos", rest)) => Ok((TtPosition::StartPos, rest)),
        Some((&b"sfen", rest)) => {
            if rest.len() < 4 {
                return Err(TtParseError::ShortSfenClause);
            }
            Ok((
                TtPosition::Sfen([rest[0], rest[1], rest[2], rest[3]]),
                &rest[4..],
            ))
        }
        Some((other, _)) => Err(TtParseError::UnknownPositionClause(other)),
        None => Err(TtParseError::MissingPositionClause),
    }
}

fn parse_store<'a>(tokens: &[&'a [u8]]) -> Result<TtStoreArgs<'a>, TtParseError<'a>> {
    let mut position: Option<TtPosition<'a>> = None;
    let mut mv: Option<&'a [u8]> = None;
    let mut value: Option<Value> = None;
    let mut depth: Option<Depth> = None;
    let mut bound: Option<Bound> = None;
    let mut eval: Option<Value> = None;
    let mut pv = false;
    let mut path_dep: Option<bool> = None;

    let mut i = 0;
    while i < tokens.len() {
        match tokens[i] {
            kind @ (b"sfen" | b"startpos") => {
                reject_duplicate(&position, TtClause::Position)?;
                let (clause, _) = parse_position_clause(&tokens[i..])?;
                position = Some(clause);
                i += if kind == b"sfen" { 5 } else { 1 };
            }
            b"move" => {
                reject_duplicate(&mv, TtClause::Move)?;
                mv = Some(operand(tokens, i, TtClause::Move)?);
                i += 2;
            }
            b"value" => {
                reject_duplicate(&value, TtClause::Value)?;
                // `value mate <n>` | `value cp <n>` | `value <n>`. The bare form
                // is the documented one; `cp` is accepted as an explicit synonym
                // so a caller can mirror USI's `score cp N` spelling.
                let (v, consumed) = match tokens.get(i + 1) {
                    Some(&b"mate") => (
                        mate_to_value(integer(tokens, i + 2, TtClause::ValueMate)?)?,
                        3,
                    ),
                    Some(&b"cp") => (cp_to_value(integer(tokens, i + 2, TtClause::ValueCp)?)?, 3),
                    Some(_) => (cp_to_value(integer(tokens, i + 1, TtClause::Value)?)?, 2),
                    None => return Err(TtParseError::MissingOperand(TtClause::Value)),
                };
                value = Some(v);
                i += consumed;
            }
            b"depth" => {
                reject_duplicate(&depth, TtClause::Depth)?;
                let d = integer(tokens, i + 1, TtClause::Depth)?;
                if !(MIN_STORE_DEPTH as i64..=MAX_STORE_DEPTH as i64).contains(&d) {
                    return Err(TtParseError::DepthOutOfRange(d));
                }
                depth = Some(d as Depth);
                i += 2;
            }
            b"bound" => {
                reject_duplicate(&bound, TtClause::Bound)?;
                bound = Some(parse_bound(operand(tokens, i, TtClause::Bound)?)?);
                i += 2;
            }
            b"eval" => {
                reject_duplicate(&eval, TtClause::Eval)?;
                // `eval <cp>`, with the same optional explicit `cp` synonym.
                let (v, consumed) = match tokens.get(i + 1) {
                    Some(&b"cp") => (cp_to_value(integer(tokens, i + 2, TtClause::EvalCp)?)?, 3),
                    Some(_) => (cp_to_value(integer(tokens, i + 1, TtClause::Eval)?)?, 2),
                    None => return Err(TtParseError::MissingOperand(TtClause::Eval)),
                };
                eval = Some(v);
                i += consumed;
            }
            b"pv" => {
                pv = true;
                i += 1;
            }
            b"pathdep" => {
                reject_duplicate(&path_dep, TtClause::PathDep)?;
                path_dep = Some(parse_path_dep(operand(tokens, i, TtClause::PathDep)?)?);
                i += 2;
            }
            other => return Err(TtParseError::UnexpectedToken(other)),
        }
    }

    Ok(TtStoreArgs {
        position: position.ok_or(TtParseError::MissingClause(TtClause::Position))?,
        mv: mv.ok_or(TtParseError::MissingClause(TtClause::Move))?,
        value: value.ok_or(TtParseError::MissingClause(TtClause::Value))?,
        depth: depth.ok_or(TtParseError::MissingClause(TtClause::Depth))?,
        bound: bound.ok_or(TtParseError::MissingClause(TtClause::Bound))?,
        eval: eval.ok_or(TtParseError::MissingClause(TtClause::Eval))?,
        pv,
        path_dep: path_dep.unwrap_or(false),
    })
}

/// The `pathdep` operand: the same `0` / `1` spelling the field is reported in.
fn parse_path_dep<'a>(tok: &'a [u8]) -> Result<bool, TtParseError<'a>> {
    match tok {
        b"0" => Ok(false),
        b"1" => Ok(true),
        other => Err(TtParseError::BadPathDep(other)),
    }
}

fn reject_duplicate<'a, T>(slot: &Option<T>, clause: TtClause) -> Result<(), TtParseError<'a>> {
    if slot.is_some() {
        return Err(TtParseError::DuplicateClause(clause));
    }
    Ok(())
}

/// The token after `tokens[i]`, or a "needs an argument" error naming `clause`.
fn operand<'a>(
    tokens: &[&'a [u8]],
    i: usize,
    clause: TtClause,
) -> Result<&'a [u8], TtParseError<'a>> {
    tokens
        .get(i + 1)
        .copied()
        .ok_or(TtParseError::MissingOperand(clause))
}

/// `tokens[i]` parsed as a decimal integer. Parsed as `i64` so an absurd literal
/// is a range error below rather than a parse error, and never an overflow.
fn integer<'a>(tokens: &[&'a [u8]], i: usize, clause: TtClause) -> Result<i64, TtParseError<'a>> {
    let Some(&tok) = tokens.get(i) else {
        return Err(TtParseError::MissingOperand(clause));
    };
    atoi_i64(tok).ok_or(TtParseError::NotAnInteger { clause, token: tok })
}

fn parse_bound<'a>(tok: &'a [u8]) -> Result<Bound, TtParseError<'a>> {
    match tok {
        b"exact" => Ok(Bound::Exact),
        b"lower" => Ok(Bound::Lower),
        b"upper" => Ok(Bound::Upper),
        other => Err(TtParseError::UnknownBound(other)),
    }
}

/// USI centipawns → an internal search value: the inverse of the reference
/// `to_cp` (`100 * v / PawnValue`), with C++-style truncating division.
/// Rejected when the result would land in the decisive band, where it would
/// read back as a mate score instead of a centipawn one.
pub fn cp_to_value(cp: i64) -> Result<Value, TtParseError<'static>> {
    let v = cp * PAWN_VALUE as i64 / 100;
    if v.abs() >= VALUE_TB_WIN_IN_MAX_PLY as i64 {
        return Err(TtParseError::DecisiveCp { cp, value: v });
    }
    Ok(v as Value)
}

/// A USI mate distance → an internal search value: `mate_in` / `mated_in`
/// measured from the named position. Positive is a win for the side to move,
/// negative a loss.
pub fn mate_to_value(n: i64) -> Result<Value, TtParseError<'static>> {
    if n.abs() > MAX_MATE_DISTANCE {
        return Err(TtParseError::MateOutOfRange(n));
    }
    // Mirrors the decode in `push_score`: `distance = VALUE_MATE - |v|`,
    // signed by `v`. Solving for `v` gives these two branches.
    Ok(if n >= 0 {
        VALUE_MATE - n as Value
    } else {
        -VALUE_MATE - n as Value
    })
}

/// `value_to_tt(v, ply)` — shift a mate score away from the root before
/// storing, making the stored value position-absolute. Identical to the
/// search's private copy in `yorkie-search`.
pub fn value_to_tt(v: Value, ply: i32) -> Value {
    if v >= VALUE_TB_WIN_IN_MAX_PLY {
        v + ply
    } else if v <= -VALUE_TB_WIN_IN_MAX_PLY {
        v - ply
    } else {
        v
    }
}

/// `value_from_tt(v, ply)` — shift a stored mate score back toward the root.
/// `VALUE_NONE` passes through unchanged.
pub fn value_from_tt(v: Value, ply: i32) -> Value {
    if v == yorkie_storage::VALUE_NONE {
        yorkie_storage::VALUE_NONE
    } else if v >= VALUE_TB_WIN_IN_MAX_PLY {
        v - ply
    } else if v <= -VALUE_TB_WIN_IN_MAX_PLY {
        v + ply
    } else {
        v
    }
}

/// The output spelling of a [`Bound`], and the inverse of `parse_bound` for
/// the three storable values. `Bound::None` never comes from a `tt store`, but
/// an entry the search wrote can carry it, so it has a name too.
pub fn bound_name(b: Bound) -> &'static [u8] {
    match b {
        Bound::None => b"none",
        Bound::Upper => b"upper",
        Bound::Lower => b"lower",
        Bound::Exact => b"exact",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tokens of `s`, as the command's own tokeniser produces them.
    fn toks(s: &str) -> Vec<&[u8]> {
        yorkie_state::text::tokens(s.as_bytes()).collect()
    }

    fn parse(s: &str) -> Result<TtCommand<'_>, TtParseError<'_>> {
        // The tokens borrow `s`, so the vector they sit in may go away.
        let tokens = toks(s);
        parse_tt(&tokens)
    }

    /// A failure's message, so an assertion can read it as text.
    fn message(err: &TtParseError<'_>) -> String {
        let mut bytes = [0u8; 512];
        let mut out = TextWriter::new(&mut bytes);
        err.write_message(&mut out);
        String::from_utf8(out.as_bytes().to_vec()).expect("a message is ASCII")
    }

    const SFEN: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
    const SFEN_FIELDS: [&[u8]; 4] = [
        b"lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL",
        b"b",
        b"-",
        b"1",
    ];

    #[test]
    fn parses_probe_with_sfen_and_startpos() {
        let line = format!("probe sfen {SFEN}");
        let tokens = toks(&line);
        assert_eq!(
            parse_tt(&tokens),
            Ok(TtCommand::Probe(TtPosition::Sfen(SFEN_FIELDS)))
        );
        assert_eq!(
            parse("probe startpos"),
            Ok(TtCommand::Probe(TtPosition::StartPos))
        );
    }

    #[test]
    fn parses_children_with_sfen() {
        let line = format!("children sfen {SFEN}");
        let tokens = toks(&line);
        assert_eq!(
            parse_tt(&tokens),
            Ok(TtCommand::Children(TtPosition::Sfen(SFEN_FIELDS)))
        );
    }

    #[test]
    fn parses_the_documented_store_line() {
        let line = format!("store sfen {SFEN} move 7g7f value 100 depth 12 bound exact eval 50 pv");
        let tokens = toks(&line);
        let cmd = parse_tt(&tokens).expect("documented syntax parses");
        assert_eq!(
            cmd,
            TtCommand::Store(TtStoreArgs {
                position: TtPosition::Sfen(SFEN_FIELDS),
                mv: b"7g7f",
                // 100 cp * 90 / 100 == 90 internal.
                value: 90,
                depth: 12,
                bound: Bound::Exact,
                eval: 45,
                pv: true,
                path_dep: false,
            })
        );
    }

    #[test]
    fn pv_and_pathdep_default_to_false_and_cp_synonym_is_accepted() {
        let cmd = parse("store startpos move none value cp 0 depth 0 bound lower eval cp 0")
            .expect("parses");
        assert_eq!(
            cmd,
            TtCommand::Store(TtStoreArgs {
                position: TtPosition::StartPos,
                mv: b"none",
                value: 0,
                depth: 0,
                bound: Bound::Lower,
                eval: 0,
                pv: false,
                path_dep: false,
            })
        );
    }

    #[test]
    fn pathdep_takes_zero_or_one_and_nothing_else() {
        let stored = |arg: &str| -> Result<bool, ()> {
            let line = format!(
                "store startpos move none value 0 depth 1 bound exact eval 0 pathdep {arg}"
            );
            let tokens = toks(&line);
            match parse_tt(&tokens) {
                Ok(TtCommand::Store(args)) => Ok(args.path_dep),
                Ok(_) => panic!("`store` parses as a store"),
                Err(_) => Err(()),
            }
        };
        assert_eq!(stored("1"), Ok(true));
        assert_eq!(stored("0"), Ok(false));

        for arg in ["true", "2", "-1", "yes"] {
            assert!(stored(arg).is_err(), "`pathdep {arg}` must be rejected");
        }
        // Operand-less and duplicated, as every other clause is checked.
        assert!(
            parse("store startpos move none value 0 depth 1 bound exact eval 0 pathdep").is_err()
        );
        assert!(
            parse(
                "store startpos move none value 0 depth 1 bound exact eval 0 pathdep 1 pathdep 0"
            )
            .is_err()
        );
        // `probe` / `children` take a position clause and nothing else.
        assert!(parse("probe startpos pathdep 1").is_err());
    }

    #[test]
    fn store_clauses_may_come_in_any_order() {
        let a = parse("store startpos move 7g7f value 10 depth 3 bound upper eval 0");
        let b = parse("store bound upper eval 0 depth 3 value 10 move 7g7f startpos");
        assert_eq!(a, b);
        assert!(a.is_ok());
    }

    #[test]
    fn mate_value_maps_to_mate_in_and_mated_in() {
        // `mate 5` is `mate_in(5)`; `mate -5` is `mated_in(5)`.
        assert_eq!(mate_to_value(5), Ok(VALUE_MATE - 5));
        assert_eq!(mate_to_value(-5), Ok(-VALUE_MATE + 5));
        assert_eq!(mate_to_value(0), Ok(VALUE_MATE));
        assert!(mate_to_value(MAX_MATE_DISTANCE).is_ok());
        assert!(mate_to_value(MAX_MATE_DISTANCE + 1).is_err());
        assert!(mate_to_value(-MAX_MATE_DISTANCE - 1).is_err());
    }

    #[test]
    fn cp_maps_through_the_usi_pawn_scale_and_rejects_decisive_values() {
        assert_eq!(cp_to_value(100), Ok(90));
        assert_eq!(cp_to_value(-100), Ok(-90));
        // Truncating division toward zero, matching the reference `to_cp`.
        assert_eq!(cp_to_value(1), Ok(0));
        assert_eq!(cp_to_value(-1), Ok(0));
        assert!(cp_to_value(1_000_000).is_err());
        assert!(cp_to_value(-1_000_000).is_err());
    }

    #[test]
    fn tt_value_conversion_is_the_identity_at_the_root() {
        for v in [0, 123, -123, VALUE_MATE - 5, -VALUE_MATE + 5] {
            assert_eq!(value_to_tt(v, 0), v);
            assert_eq!(value_from_tt(v, 0), v);
        }
    }

    #[test]
    fn tt_value_conversion_shifts_only_decisive_scores() {
        assert_eq!(value_to_tt(123, 1), 123);
        assert_eq!(value_from_tt(123, 1), 123);
        // A child's stored `mate in 5` reads back as `mate in 6` from the parent.
        assert_eq!(value_from_tt(VALUE_MATE - 5, 1), VALUE_MATE - 6);
        assert_eq!(value_from_tt(-VALUE_MATE + 5, 1), -VALUE_MATE + 6);
        assert_eq!(
            value_from_tt(yorkie_storage::VALUE_NONE, 1),
            yorkie_storage::VALUE_NONE
        );
    }

    #[test]
    fn depth_is_range_checked_against_the_entry_encoding() {
        for d in [MIN_STORE_DEPTH, 0, MAX_STORE_DEPTH] {
            let line = format!("store startpos move none value 0 depth {d} bound exact eval 0");
            let tokens = toks(&line);
            assert!(parse_tt(&tokens).is_ok(), "depth {d} must be storable");
        }
        for d in [MIN_STORE_DEPTH - 1, MAX_STORE_DEPTH + 1] {
            let line = format!("store startpos move none value 0 depth {d} bound exact eval 0");
            let tokens = toks(&line);
            assert!(parse_tt(&tokens).is_err(), "depth {d} must be rejected");
        }
    }

    #[test]
    fn missing_and_unknown_pieces_are_errors() {
        assert!(parse("").is_err());
        assert!(parse("frobnicate startpos").is_err());
        assert!(parse("probe").is_err());
        assert!(parse("probe sfen a b c").is_err());
        assert!(parse("probe nonsense").is_err());
        let line = format!("probe sfen {SFEN} extra");
        let tokens = toks(&line);
        assert!(parse_tt(&tokens).is_err());
        // Every mandatory `store` clause, dropped one at a time.
        assert!(parse("store move none value 0 depth 1 bound exact eval 0").is_err());
        assert!(parse("store startpos value 0 depth 1 bound exact eval 0").is_err());
        assert!(parse("store startpos move none depth 1 bound exact eval 0").is_err());
        assert!(parse("store startpos move none value 0 bound exact eval 0").is_err());
        assert!(parse("store startpos move none value 0 depth 1 eval 0").is_err());
        assert!(parse("store startpos move none value 0 depth 1 bound exact").is_err());
        // Bad operands.
        assert!(parse("store startpos move none value x depth 1 bound exact eval 0").is_err());
        assert!(parse("store startpos move none value 0 depth x bound exact eval 0").is_err());
        assert!(parse("store startpos move none value 0 depth 1 bound sideways eval 0").is_err());
        assert!(parse("store startpos move none value 0 depth 1 bound exact eval x").is_err());
        assert!(parse("store startpos move none value 0 depth 1 bound exact eval 0 wat").is_err());
        // Duplicates.
        assert!(
            parse("store startpos startpos move none value 0 depth 1 bound exact eval 0").is_err()
        );
        assert!(
            parse("store startpos move none move none value 0 depth 1 bound exact eval 0").is_err()
        );
        // Trailing operand-less clause.
        assert!(parse("store startpos move none value 0 depth 1 bound exact eval").is_err());
        assert!(parse("store startpos move").is_err());
    }

    /// A token carrying bytes no `tt` line is spelled in matches no keyword and
    /// parses as no number, so it takes the unknown-token path — and the
    /// message quoting it is written out as it arrived.
    #[test]
    fn a_non_ascii_token_is_refused_by_name() {
        let tokens: [&[u8]; 1] = [b"\x82\xa0"];
        let err = parse_tt(&tokens).expect_err("no such subcommand");
        let mut bytes = [0u8; 128];
        let mut out = TextWriter::new(&mut bytes);
        err.write_message(&mut out);
        assert_eq!(
            out.as_bytes(),
            b"unknown subcommand `\x82\xa0`; expected `store`, `probe` or `children`"
        );
    }

    #[test]
    fn a_message_names_the_clause_it_refused() {
        assert_eq!(
            message(&TtParseError::MissingOperand(TtClause::Depth)),
            "`depth` needs an argument"
        );
        assert_eq!(
            message(&TtParseError::DuplicateClause(TtClause::Move)),
            "duplicate `move` clause"
        );
        assert_eq!(
            message(&TtParseError::NotAnInteger {
                clause: TtClause::ValueMate,
                token: b"soon",
            }),
            "`value mate` argument `soon` is not an integer"
        );
    }

    #[test]
    fn bound_names_round_trip() {
        for (tok, b) in [
            (&b"exact"[..], Bound::Exact),
            (&b"lower"[..], Bound::Lower),
            (&b"upper"[..], Bound::Upper),
        ] {
            assert_eq!(parse_bound(tok), Ok(b));
            assert_eq!(bound_name(b), tok);
        }
        assert_eq!(bound_name(Bound::None), b"none");
    }
}
