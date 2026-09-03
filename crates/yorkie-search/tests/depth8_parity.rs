//! Depth-8 search parity test.
//!
//! Runs the `go depth 8` root search ([`QSearch::run_root`]) against the
//! reference-captured fixtures under `tests/fixtures/search-depth8/` and
//! asserts **bestmove, score, and nodes** as one inseparable set. `nodes` is
//! cumulative over the whole `go`, so the test transitively pins the depth-1..7
//! iterations too.
//!
//! Depth 8 is the first depth at which the singular guard fires directly at
//! interior nodes rather than only through LMR re-search deepening, so it is
//! the first to exercise the singular family end to end. It also newly reaches
//! internal iterative reduction and the `depth > 5` disjunct of the non-PV
//! early TT cutoff.
//!
//! The fixtures were captured with Threads=1, no book, `usinewgame` before each
//! position, USI_Hash 1024 MiB and FV_SCALE 16, reproduced here. Skipped with a
//! notice when `nn.bin` is absent.
//!
//! ## What `startpos` at depth 8 is sensitive to
//!
//! Two whole-engine invariants show up here and nowhere shallower, so this
//! fixture is the test that catches either of them regressing:
//!
//! 1. **Zobrist aliasing (observable from depth 2).** The hash-indexed pawn
//!    history (`pawnHistory`, 8192 planes) and the correction histories alias by
//!    concrete key value, so a privately seeded Zobrist table flips a quiet's
//!    ordering on the first colliding pawn structure. That is a single node at
//!    depth 2, which the `(nodes & 14)` root tie-break amplifies into a flipped
//!    bestmove here (`2g2f` / 11940 nodes instead of `7g7f` / 12636).
//!    `crates/yorkie-state/src/key.rs` reproduces the reference's Zobrist
//!    bit-for-bit; the minimal repro is gated by `tests/depth2_parity.rs`.
//! 2. **Continuation planes in qsearch (observable from depth 6).** The
//!    reference sets `ss->continuationHistory` /
//!    `continuationCorrectionHistory` inside `do_move` for **every** move,
//!    qsearch moves included. A deeper node's `correction_value` reads the
//!    continuation-correction plane at `(ss-2)` / `(ss-4)`, so leaving those
//!    unset at a qsearch ply feeds a wrong `cntcv` to any descendant, shifting
//!    the corrected eval by ~1 and flipping a shallow prune once the correction
//!    tables warm up — a handful of nodes, invisible below depth 6.

use std::path::PathBuf;

use serde::Deserialize;
use yorkie_search::{QSearch, RootKind};
use yorkie_state::{Move, Position, format_usi_move, parse_sfen, parse_usi_move};
use yorkie_storage::TranspositionTable;

/// `VALUE_MATE` (`types.h`).
const VALUE_MATE: i32 = 32000;
/// `VALUE_TB_WIN_IN_MAX_PLY` (`types.h`): the `is_decisive` threshold.
const VALUE_TB_WIN_IN_MAX_PLY: i32 = VALUE_MATE - 246;
/// `Eval::PawnValue` (`NormalizeToPawnValue`, `usi.cpp`).
const PAWN_VALUE: i32 = 90;
/// Engine default `USI_Hash` in MiB (`tests/fixtures/search-depth8/README.md`).
const HASH_MB: usize = 1024;

/// All six fixtures the ported search matches exactly at depth 8. Every one
/// pins `bestmove` / `score` / `nodes` exactly.
const FIXTURES: &[&str] = &[
    "startpos.json",
    "drop-heavy.json",
    "mid-game-tactical.json",
    "check-evasion.json",
    "promotion-zone-edges.json",
    "sennichite.json",
];

#[derive(Debug, Deserialize)]
struct FixtureJson {
    sfen: String,
    /// Optional USI moves applied after the SFEN (USI `position ... moves ...`).
    #[serde(default)]
    moves: Vec<String>,
    depth: i32,
    bestmove: String,
    score: ScoreJson,
    nodes: u64,
    /// The principal variation (desirable but not gated).
    #[serde(default)]
    pv: Vec<String>,
}

/// Fixture score: exactly one of `cp` or `mate` is present.
#[derive(Debug, Deserialize, PartialEq)]
struct ScoreJson {
    #[serde(default)]
    cp: Option<i32>,
    #[serde(default)]
    mate: Option<i32>,
}

fn nn_bin_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../eval/nn.bin")
}

fn load_fixture(name: &str) -> FixtureJson {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/search-depth8")
        .join(name);
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read fixture {}: {e}", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse fixture {name}: {e}"))
}

/// Parse the SFEN and apply the optional `moves` prefix, mirroring USI
/// `position sfen <SFEN> moves <m1> <m2> ...`.
fn setup(fixture: &FixtureJson) -> Position {
    let mut pos = parse_sfen(&fixture.sfen).expect("valid fixture SFEN");
    for usi in &fixture.moves {
        let m = parse_usi_move(usi, &pos).unwrap_or_else(|e| panic!("bad move {usi}: {e:?}"));
        pos.do_move(m);
    }
    pos
}

/// The USI-string form of the outcome's bestmove (fixtures use ordinary moves;
/// the resign / win sentinels never occur for the six fixtures).
fn bestmove_usi(best_move: Move, kind: RootKind) -> String {
    match kind {
        RootKind::Resign => "resign".to_string(),
        RootKind::DeclarationWin => "win".to_string(),
        RootKind::Normal => format_usi_move(best_move),
    }
}

/// `is_decisive` (`types.h`).
fn is_decisive(v: i32) -> bool {
    v.abs() >= VALUE_TB_WIN_IN_MAX_PLY
}

/// Format a search value the way the reference USI layer does (`score.cpp` /
/// `usi.cpp` `format_score`): a mate distance for decisive scores, else `100 *
/// v / PawnValue` centipawns (C++ truncating division).
fn format_score(v: i32) -> ScoreJson {
    if is_decisive(v) {
        let distance = VALUE_MATE - v.abs();
        ScoreJson {
            cp: None,
            mate: Some(if v > 0 { distance } else { -distance }),
        }
    } else {
        ScoreJson {
            cp: Some(100 * v / PAWN_VALUE),
            mate: None,
        }
    }
}

fn assert_fixture(name: &str, net: &yorkie_eval::NnueNetwork, tt: &mut TranspositionTable) {
    let json = load_fixture(name);
    assert_eq!(json.depth, 8, "{name}: depth-8 fixtures only");

    // usinewgame: clear the table (also resets the generation to 0).
    tt.clear();
    let pos = setup(&json);

    let outcome = {
        let mut search = QSearch::new(net, tt);
        search.run_root(&pos, json.depth)
    };

    // bestmove — gated for every fixture.
    let got_best = bestmove_usi(outcome.best_move, outcome.kind);
    assert_eq!(
        got_best, json.bestmove,
        "{name}: bestmove mismatch (got {got_best}, want {})",
        json.bestmove
    );

    // score (cp or mate) — gated for every fixture.
    let got_score = format_score(outcome.score);
    assert_eq!(
        got_score, json.score,
        "{name}: score mismatch (raw value {})",
        outcome.score
    );

    // nodes — gated hard for all six (cumulative over the whole `go`).
    assert_eq!(
        outcome.nodes, json.nodes,
        "{name}: node count mismatch (got {}, want {})",
        outcome.nodes, json.nodes
    );

    // pv is desirable but not gated; surface a divergence as a notice only.
    let got_pv: Vec<String> = outcome.pv.iter().map(|&m| format_usi_move(m)).collect();
    if got_pv != json.pv {
        eprintln!(
            "{name}: pv differs (got {got_pv:?}, want {:?}) — not gated",
            json.pv
        );
    }
}

/// All six fixtures pin `bestmove` / `score` / `nodes` exactly, each an
/// inseparable triple. Skipped with a notice when `nn.bin` is absent.
#[cfg_attr(miri, ignore)]
#[test]
fn depth8_search_matches_reference_fixtures() {
    let path = nn_bin_path();
    if !path.exists() {
        eprintln!(
            "skipping depth8_search_matches_reference_fixtures: {} is not present (obtained out-of-band)",
            path.display()
        );
        return;
    }

    let net = yorkie_eval::load_network(&path).expect("real nn.bin should load and validate");

    // One 1024 MiB table, cleared per fixture (the usinewgame equivalent).
    let mut tt = TranspositionTable::new();
    tt.resize(HASH_MB);

    for name in FIXTURES {
        assert_fixture(name, &net, &mut tt);
    }
}
