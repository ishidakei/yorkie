//! The `usi-extras` `tt` command family: the *semantic parse* of the argument
//! tokens behind `tt store` / `tt probe` / `tt children`.
//!
//! The whole module exists only under the `usi-extras` cargo feature. With it
//! off, [`crate::parser::parse_line`] does not know the `tt` token at all and
//! nothing here is compiled.
//!
//! There is no upstream YaneuraOu precedent for these commands — the reference
//! exposes no TT read/write USI command — so the syntax below is this project's
//! own design.
//!
//! This module owns only what can be decided from the tokens alone: the
//! subcommand, the position clause kept as a string, the scalar fields, and
//! their range validation. Anything needing a `Position` is the driver's.
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

use yorkie_storage::{Bound, DEPTH_NONE, Depth, Value};

use crate::driver::{PAWN_VALUE, VALUE_MATE, VALUE_TB_WIN_IN_MAX_PLY};

/// Largest mate distance the value encoding can carry: `VALUE_MATE - n` must
/// stay decisive (`|v| >= VALUE_TB_WIN_IN_MAX_PLY`), and that threshold is
/// `VALUE_MATE - MAX_PLY` with the reference's `MAX_PLY == 246`
/// (`source/types.h`).
pub const MAX_MATE_DISTANCE: i64 = (VALUE_MATE - VALUE_TB_WIN_IN_MAX_PLY) as i64;

/// Smallest `depth` an entry can carry. [`yorkie_storage::tt`] stores
/// `depth8 = depth - DEPTH_NONE` in a `u8`, and `depth8 == 0` marks an
/// unoccupied entry, so the representable range is `DEPTH_NONE + 1 ..=
/// DEPTH_NONE + 255`. Out-of-range depths are rejected here rather than
/// reaching the `debug_assert!`s in `TTEntry::save`.
pub const MIN_STORE_DEPTH: Depth = DEPTH_NONE + 1;
/// Largest `depth` an entry can carry (see [`MIN_STORE_DEPTH`]).
pub const MAX_STORE_DEPTH: Depth = DEPTH_NONE + 255;

/// A `tt` argument-parse failure, surfaced by the driver as one
/// `info string tt error: <msg>` line so a garbage argument fails loudly
/// without panicking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TtParseError(pub String);

impl std::fmt::Display for TtParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

fn err(msg: impl Into<String>) -> TtParseError {
    TtParseError(msg.into())
}

/// The position clause, kept verbatim: the driver turns it into a `Position` so
/// SFEN diagnostics come from the one parser the `position` command uses.
///
/// `startpos` is a shorthand for the four-field `sfen` clause, spelled out here
/// rather than expanded so the driver can use `Position::startpos()` directly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TtPosition {
    StartPos,
    /// The four SFEN fields (board, side-to-move, hands, ply) joined by single
    /// spaces, exactly as [`crate::parser::PositionSfen::Sfen`] carries them.
    Sfen(String),
}

/// A fully validated `tt store` invocation. Every numeric field is already in
/// the engine's internal units and in range for the entry encoding; the only
/// thing left unresolved is [`Self::mv`], which needs the position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TtStoreArgs {
    pub position: TtPosition,
    /// The raw `move` token — a USI move, or the literal `none` for "no move"
    /// (fragment `0`, which `TTEntry::save` treats as "keep whatever move the
    /// entry already had for this position").
    pub mv: String,
    /// Search value in internal units, relative to [`Self::position`] as root.
    pub value: Value,
    pub depth: Depth,
    pub bound: Bound,
    /// Static eval in internal units (the entry's `eval16` field).
    pub eval: Value,
    pub pv: bool,
}

/// One parsed `tt` invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TtCommand {
    Store(TtStoreArgs),
    Probe(TtPosition),
    Children(TtPosition),
}

/// Parse the tokens following the `tt` keyword.
///
/// The clauses are keyword-driven and accepted in any order. A missing mandatory
/// clause, a duplicate clause, an unknown token, or an out-of-range number is an
/// error; this function never panics.
pub fn parse_tt(tokens: &[String]) -> Result<TtCommand, TtParseError> {
    let tokens: Vec<&str> = tokens.iter().map(String::as_str).collect();
    let Some((&sub, rest)) = tokens.split_first() else {
        return Err(err(
            "missing subcommand; expected `store`, `probe` or `children`",
        ));
    };
    match sub {
        "store" => parse_store(rest).map(TtCommand::Store),
        "probe" => parse_lone_position(rest, "probe").map(TtCommand::Probe),
        "children" => parse_lone_position(rest, "children").map(TtCommand::Children),
        other => Err(err(format!(
            "unknown subcommand `{other}`; expected `store`, `probe` or `children`"
        ))),
    }
}

/// The whole argument list of `tt probe` / `tt children`: one position clause
/// and nothing else.
fn parse_lone_position(tokens: &[&str], sub: &str) -> Result<TtPosition, TtParseError> {
    let (position, rest) = parse_position_clause(tokens)?;
    if !rest.is_empty() {
        return Err(err(format!(
            "`{sub}` takes only a position clause; unexpected trailing `{}`",
            rest.join(" ")
        )));
    }
    Ok(position)
}

/// Consume a leading `sfen <board> <side> <hands> <ply>` or `startpos`,
/// returning the clause and the tokens after it.
fn parse_position_clause<'a>(
    tokens: &[&'a str],
) -> Result<(TtPosition, Vec<&'a str>), TtParseError> {
    match tokens.split_first() {
        Some((&"startpos", rest)) => Ok((TtPosition::StartPos, rest.to_vec())),
        Some((&"sfen", rest)) => {
            if rest.len() < 4 {
                return Err(err(
                    "`sfen` needs its four fields (board, side-to-move, hands, ply)",
                ));
            }
            Ok((TtPosition::Sfen(rest[..4].join(" ")), rest[4..].to_vec()))
        }
        Some((other, _)) => Err(err(format!(
            "expected `sfen <board> <side> <hands> <ply>` or `startpos`, found `{other}`"
        ))),
        None => Err(err("missing position clause (`sfen …` or `startpos`)")),
    }
}

fn parse_store(tokens: &[&str]) -> Result<TtStoreArgs, TtParseError> {
    let mut position: Option<TtPosition> = None;
    let mut mv: Option<String> = None;
    let mut value: Option<Value> = None;
    let mut depth: Option<Depth> = None;
    let mut bound: Option<Bound> = None;
    let mut eval: Option<Value> = None;
    let mut pv = false;

    let mut i = 0;
    while i < tokens.len() {
        match tokens[i] {
            kind @ ("sfen" | "startpos") => {
                reject_duplicate(&position, "position")?;
                let (clause, _) = parse_position_clause(&tokens[i..])?;
                position = Some(clause);
                i += if kind == "sfen" { 5 } else { 1 };
            }
            "move" => {
                reject_duplicate(&mv, "move")?;
                mv = Some(operand(tokens, i, "move")?.to_string());
                i += 2;
            }
            "value" => {
                reject_duplicate(&value, "value")?;
                // `value mate <n>` | `value cp <n>` | `value <n>`. The bare form
                // is the documented one; `cp` is accepted as an explicit synonym
                // so a caller can mirror USI's `score cp N` spelling.
                let (v, consumed) = match tokens.get(i + 1) {
                    Some(&"mate") => (mate_to_value(integer(tokens, i + 2, "value mate")?)?, 3),
                    Some(&"cp") => (cp_to_value(integer(tokens, i + 2, "value cp")?)?, 3),
                    Some(_) => (cp_to_value(integer(tokens, i + 1, "value")?)?, 2),
                    None => return Err(err("`value` needs an argument")),
                };
                value = Some(v);
                i += consumed;
            }
            "depth" => {
                reject_duplicate(&depth, "depth")?;
                let d = integer(tokens, i + 1, "depth")?;
                if !(MIN_STORE_DEPTH as i64..=MAX_STORE_DEPTH as i64).contains(&d) {
                    return Err(err(format!(
                        "depth {d} out of range; an entry stores {MIN_STORE_DEPTH}..={MAX_STORE_DEPTH}"
                    )));
                }
                depth = Some(d as Depth);
                i += 2;
            }
            "bound" => {
                reject_duplicate(&bound, "bound")?;
                bound = Some(parse_bound(operand(tokens, i, "bound")?)?);
                i += 2;
            }
            "eval" => {
                reject_duplicate(&eval, "eval")?;
                // `eval <cp>`, with the same optional explicit `cp` synonym.
                let (v, consumed) = match tokens.get(i + 1) {
                    Some(&"cp") => (cp_to_value(integer(tokens, i + 2, "eval cp")?)?, 3),
                    Some(_) => (cp_to_value(integer(tokens, i + 1, "eval")?)?, 2),
                    None => return Err(err("`eval` needs an argument")),
                };
                eval = Some(v);
                i += consumed;
            }
            "pv" => {
                pv = true;
                i += 1;
            }
            other => {
                return Err(err(format!("unexpected token `{other}` in `tt store`")));
            }
        }
    }

    Ok(TtStoreArgs {
        position: position.ok_or_else(|| missing("position clause (`sfen …` or `startpos`)"))?,
        mv: mv.ok_or_else(|| missing("move <usi-move|none>"))?,
        value: value.ok_or_else(|| missing("value <cp|mate <n>>"))?,
        depth: depth.ok_or_else(|| missing("depth <d>"))?,
        bound: bound.ok_or_else(|| missing("bound <exact|lower|upper>"))?,
        eval: eval.ok_or_else(|| missing("eval <cp>"))?,
        pv,
    })
}

fn missing(clause: &str) -> TtParseError {
    err(format!("`tt store` is missing the `{clause}` clause"))
}

fn reject_duplicate<T>(slot: &Option<T>, clause: &str) -> Result<(), TtParseError> {
    if slot.is_some() {
        return Err(err(format!("duplicate `{clause}` clause")));
    }
    Ok(())
}

/// The token after `tokens[i]`, or a "needs an argument" error naming `clause`.
fn operand<'a>(tokens: &[&'a str], i: usize, clause: &str) -> Result<&'a str, TtParseError> {
    tokens
        .get(i + 1)
        .copied()
        .ok_or_else(|| err(format!("`{clause}` needs an argument")))
}

/// `tokens[i]` parsed as a decimal integer. Parsed as `i64` so an absurd literal
/// is a range error below rather than a parse error, and never an overflow.
fn integer(tokens: &[&str], i: usize, clause: &str) -> Result<i64, TtParseError> {
    let Some(tok) = tokens.get(i) else {
        return Err(err(format!("`{clause}` needs an argument")));
    };
    tok.parse::<i64>()
        .map_err(|_| err(format!("`{clause}` argument `{tok}` is not an integer")))
}

fn parse_bound(tok: &str) -> Result<Bound, TtParseError> {
    match tok {
        "exact" => Ok(Bound::Exact),
        "lower" => Ok(Bound::Lower),
        "upper" => Ok(Bound::Upper),
        other => Err(err(format!(
            "unknown bound `{other}`; expected `exact`, `lower` or `upper`"
        ))),
    }
}

/// USI centipawns → an internal search value: the inverse of the reference
/// `to_cp` (`100 * v / PawnValue`, `usi.cpp`), with C++-style truncating
/// division. Rejected when the result would land in the decisive band, where
/// it would read back as a mate score instead of a centipawn one.
pub fn cp_to_value(cp: i64) -> Result<Value, TtParseError> {
    let v = cp * PAWN_VALUE as i64 / 100;
    if v.abs() >= VALUE_TB_WIN_IN_MAX_PLY as i64 {
        return Err(err(format!(
            "cp {cp} maps to internal value {v}, which is a decisive score; \
             use `mate <n>` for mate values"
        )));
    }
    Ok(v as Value)
}

/// A USI mate distance → an internal search value: `mate_in` / `mated_in`
/// (`source/types.h`) measured from the named position.
/// Positive is a win for the side to move, negative a loss.
pub fn mate_to_value(n: i64) -> Result<Value, TtParseError> {
    if n.abs() > MAX_MATE_DISTANCE {
        return Err(err(format!(
            "mate {n} out of range; |n| must be <= {MAX_MATE_DISTANCE}"
        )));
    }
    // Mirrors the decode in `push_score`: `distance = VALUE_MATE - |v|`,
    // signed by `v`. Solving for `v` gives these two branches.
    Ok(if n >= 0 {
        VALUE_MATE - n as Value
    } else {
        -VALUE_MATE - n as Value
    })
}

/// `value_to_tt(v, ply)` (`yaneuraou-search.cpp`) — shift a mate score away
/// from the root before storing, making the stored value position-absolute.
/// Identical to the search's private copy in `yorkie-search`.
pub fn value_to_tt(v: Value, ply: i32) -> Value {
    if v >= VALUE_TB_WIN_IN_MAX_PLY {
        v + ply
    } else if v <= -VALUE_TB_WIN_IN_MAX_PLY {
        v - ply
    } else {
        v
    }
}

/// `value_from_tt(v, ply)` (`yaneuraou-search.cpp`) — shift a stored mate
/// score back toward the root. `VALUE_NONE` passes through unchanged.
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

/// The output spelling of a [`Bound`], and the inverse of [`parse_bound`] for
/// the three storable values. `Bound::None` never comes from a `tt store`, but
/// an entry the search wrote can carry it, so it has a name too.
pub fn bound_name(b: Bound) -> &'static str {
    match b {
        Bound::None => "none",
        Bound::Upper => "upper",
        Bound::Lower => "lower",
        Bound::Exact => "exact",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(s: &str) -> Vec<String> {
        s.split_whitespace().map(str::to_string).collect()
    }

    fn parse(s: &str) -> Result<TtCommand, TtParseError> {
        parse_tt(&toks(s))
    }

    const SFEN: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

    #[test]
    fn parses_probe_with_sfen_and_startpos() {
        assert_eq!(
            parse(&format!("probe sfen {SFEN}")),
            Ok(TtCommand::Probe(TtPosition::Sfen(SFEN.to_string())))
        );
        assert_eq!(
            parse("probe startpos"),
            Ok(TtCommand::Probe(TtPosition::StartPos))
        );
    }

    #[test]
    fn parses_children_with_sfen() {
        assert_eq!(
            parse(&format!("children sfen {SFEN}")),
            Ok(TtCommand::Children(TtPosition::Sfen(SFEN.to_string())))
        );
    }

    #[test]
    fn parses_the_documented_store_line() {
        let cmd = parse(&format!(
            "store sfen {SFEN} move 7g7f value 100 depth 12 bound exact eval 50 pv"
        ))
        .expect("documented syntax parses");
        assert_eq!(
            cmd,
            TtCommand::Store(TtStoreArgs {
                position: TtPosition::Sfen(SFEN.to_string()),
                mv: "7g7f".to_string(),
                // 100 cp * 90 / 100 == 90 internal.
                value: 90,
                depth: 12,
                bound: Bound::Exact,
                eval: 45,
                pv: true,
            })
        );
    }

    #[test]
    fn pv_flag_defaults_to_false_and_cp_synonym_is_accepted() {
        let cmd = parse("store startpos move none value cp 0 depth 0 bound lower eval cp 0")
            .expect("parses");
        assert_eq!(
            cmd,
            TtCommand::Store(TtStoreArgs {
                position: TtPosition::StartPos,
                mv: "none".to_string(),
                value: 0,
                depth: 0,
                bound: Bound::Lower,
                eval: 0,
                pv: false,
            })
        );
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
            assert!(
                parse(&format!(
                    "store startpos move none value 0 depth {d} bound exact eval 0"
                ))
                .is_ok(),
                "depth {d} must be storable"
            );
        }
        for d in [MIN_STORE_DEPTH - 1, MAX_STORE_DEPTH + 1] {
            assert!(
                parse(&format!(
                    "store startpos move none value 0 depth {d} bound exact eval 0"
                ))
                .is_err(),
                "depth {d} must be rejected"
            );
        }
    }

    #[test]
    fn missing_and_unknown_pieces_are_errors() {
        assert!(parse("").is_err());
        assert!(parse("frobnicate startpos").is_err());
        assert!(parse("probe").is_err());
        assert!(parse("probe sfen a b c").is_err());
        assert!(parse("probe nonsense").is_err());
        assert!(parse(&format!("probe sfen {SFEN} extra")).is_err());
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

    #[test]
    fn bound_names_round_trip() {
        for (tok, b) in [
            ("exact", Bound::Exact),
            ("lower", Bound::Lower),
            ("upper", Bound::Upper),
        ] {
            assert_eq!(parse_bound(tok), Ok(b));
            assert_eq!(bound_name(b), tok);
        }
        assert_eq!(bound_name(Bound::None), "none");
    }
}
