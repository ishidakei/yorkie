//! Depth-2 search parity test — the pawn-history-aliasing regression.
//!
//! Runs the `go depth 2` root search ([`QSearch::run_root`]) against a single
//! reference-captured fixture — `position startpos moves 7g7f` — and asserts
//! **bestmove, score, and nodes** exactly.
//!
//! This is the minimal position at which Zobrist-table aliasing is observable.
//! The pawn and correction histories are hash tables indexed by
//! `key & (size - 1)`, so their collision structure — and the quiet move
//! ordering it drives — depends on the concrete key *values*, not just the key
//! structure. A privately seeded table aliases differently from the reference
//! and flips a quiet's ordering on the first colliding pawn structure, which
//! cascades through PVS re-search bounds into the node count.
//!
//! Captured with Threads=1, no book, `usinewgame`, USI_Hash 1024 MiB and
//! FV_SCALE 16. Everything but the hash size is reproduced here: the table is
//! the `static` this build compiled in — 16 MiB under the test config — and the
//! fixture comes out the same on it. Skipped with a notice when `nn.bin` is
//! absent.
//!
//! Not compiled under `tt-entry16`. What that layout promises is exact position
//! identity, not the reference's search: it fits two entries in a cluster
//! instead of three, which changes which position a full cluster keeps, and the
//! fixture is the reference's numbers.

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

#[derive(Debug, Deserialize)]
struct FixtureJson {
    sfen: String,
    #[serde(default)]
    moves: Vec<String>,
    depth: i32,
    bestmove: String,
    score: ScoreJson,
    nodes: u64,
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
        .join("../../tests/fixtures/search-depth2")
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

/// Format a search value the way the reference USI layer does.
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

/// `position startpos moves 7g7f`, `go depth 2`: bestmove / score / nodes exact.
#[cfg_attr(miri, ignore)]
#[test]
fn depth2_startpos_7g7f_matches_reference() {
    let path = nn_bin_path();
    if !path.exists() {
        eprintln!(
            "skipping depth2_startpos_7g7f_matches_reference: {} is not present (obtained out-of-band)",
            path.display()
        );
        return;
    }

    let owned = yorkie_eval::load_network(&path).expect("real nn.bin should load and validate");
    let net = owned.network();
    // usinewgame: the shared table, emptied.
    TranspositionTable::shared().clear();

    let json = load_fixture("startpos-7g7f.json");
    assert_eq!(json.depth, 2, "depth-2 fixture only");
    let pos = setup(&json);

    let outcome = {
        let mut search = QSearch::new(net);
        search.run_root(&pos, json.depth)
    };

    let got_best = bestmove_usi(outcome.best_move, outcome.kind);
    assert_eq!(
        got_best, json.bestmove,
        "bestmove mismatch (got {got_best}, want {})",
        json.bestmove
    );

    let got_score = format_score(outcome.score);
    assert_eq!(
        got_score, json.score,
        "score mismatch (raw value {})",
        outcome.score
    );

    assert_eq!(
        outcome.nodes, json.nodes,
        "node count mismatch (got {}, want {})",
        outcome.nodes, json.nodes
    );

    let got_pv: Vec<String> = outcome.pv.iter().map(|&m| format_usi_move(m)).collect();
    if got_pv != json.pv {
        eprintln!(
            "pv differs (got {got_pv:?}, want {:?}) — not gated",
            json.pv
        );
    }
}
