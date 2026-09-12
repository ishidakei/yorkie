//! Depth-1 search parity test.
//!
//! Runs the `go depth 1` root search ([`QSearch::run_root`]) against the six
//! reference-captured fixtures under `tests/fixtures/search-depth1/` and
//! asserts **bestmove, score, and nodes** as one inseparable set: the
//! `(nodes & 14)` root tie-break means a single-node drift can cascade into a
//! different score and a flipped bestmove.
//!
//! The fixtures were captured with Threads=1, no book, `usinewgame` before each
//! position, USI_Hash 1024 MiB and FV_SCALE 16. Everything but the hash size is
//! reproduced here: the table is the `static` this build compiled in — 16 MiB
//! under the test config — and these fixtures come out the same on it. Skipped
//! with a notice when `nn.bin` is absent.
//!
//! Not compiled under `tt-entry16`. What that layout promises is exact position
//! identity, not the reference's search: it fits two entries in a cluster
//! instead of three, which changes which position a full cluster keeps, and the
//! fixtures are the reference's numbers.

#![cfg(not(feature = "tt-entry16"))]

use std::path::PathBuf;

use serde::Deserialize;
use yorkie_search::{QSearch, RootKind};
mod text_str;

use text_str::{format_usi_move, parse_sfen, parse_usi_move};
use yorkie_state::{Move, Position};
use yorkie_storage::TranspositionTable;

/// `VALUE_MATE`.
const VALUE_MATE: i32 = 32000;
/// `VALUE_TB_WIN_IN_MAX_PLY`: the `is_decisive` threshold.
const VALUE_TB_WIN_IN_MAX_PLY: i32 = VALUE_MATE - 246;
/// `Eval::PawnValue` (`NormalizeToPawnValue`).
const PAWN_VALUE: i32 = 90;

const FIXTURES: &[&str] = &[
    "startpos.json",
    "drop-heavy.json",
    "mid-game-tactical.json",
    "check-evasion.json",
    "promotion-zone-edges.json",
    "sennichite.json",
];

#[derive(Debug, Deserialize)]
struct Fixture {
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
#[derive(Debug, Deserialize)]
struct ScoreJson {
    #[serde(default)]
    cp: Option<i32>,
    #[serde(default)]
    mate: Option<i32>,
}

fn nn_bin_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../eval/nn.bin")
}

fn load_fixture(name: &str) -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/search-depth1")
        .join(name);
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read fixture {}: {e}", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse fixture {name}: {e}"))
}

/// Parse the SFEN and apply the optional `moves` prefix, mirroring USI
/// `position sfen <SFEN> moves <m1> <m2> ...`.
fn setup(fixture: &Fixture) -> Position {
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

/// `is_decisive`.
fn is_decisive(v: i32) -> bool {
    v.abs() >= VALUE_TB_WIN_IN_MAX_PLY
}

/// Format a search value the way the reference USI layer does (`format_score`):
/// a mate distance for decisive scores, else `100 * v / PawnValue` centipawns
/// (C++ truncating division).
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

fn assert_fixture<N: yorkie_eval::NetworkParams>(name: &str, net: N) {
    let fixture = load_fixture(name);
    assert_eq!(fixture.depth, 1, "{name}: depth-1 fixtures only");

    // usinewgame: clear the table (also resets the generation to 0).
    TranspositionTable::shared().clear();
    let pos = setup(&fixture);

    let outcome = {
        let mut search = QSearch::new(net);
        search.run_root(&pos, fixture.depth)
    };

    // bestmove.
    let got_best = bestmove_usi(outcome.best_move, outcome.kind);
    assert_eq!(
        got_best, fixture.bestmove,
        "{name}: bestmove mismatch (got {got_best}, want {})",
        fixture.bestmove
    );

    // score (cp or mate).
    let got_score = format_score(outcome.score);
    assert_eq!(
        got_score.cp, fixture.score.cp,
        "{name}: score cp mismatch (raw value {})",
        outcome.score
    );
    assert_eq!(
        got_score.mate, fixture.score.mate,
        "{name}: score mate mismatch (raw value {})",
        outcome.score
    );

    // nodes.
    assert_eq!(
        outcome.nodes, fixture.nodes,
        "{name}: node count mismatch (got {}, want {})",
        outcome.nodes, fixture.nodes
    );

    // pv is desirable but not gated; surface a divergence as a notice only.
    let got_pv: Vec<String> = outcome.pv.iter().map(|&m| format_usi_move(m)).collect();
    if got_pv != fixture.pv {
        eprintln!(
            "{name}: pv differs (got {got_pv:?}, want {:?}) — not gated",
            fixture.pv
        );
    }
}

#[cfg_attr(miri, ignore)]
#[test]
fn depth1_search_matches_reference_fixtures() {
    let path = nn_bin_path();
    if !path.exists() {
        eprintln!(
            "skipping depth1_search_matches_reference_fixtures: {} is not present (obtained out-of-band)",
            path.display()
        );
        return;
    }

    let owned = yorkie_eval::load_network(&path).expect("real nn.bin should load and validate");
    let net = owned.network();

    for name in FIXTURES {
        assert_fixture(name, net);
    }
}
