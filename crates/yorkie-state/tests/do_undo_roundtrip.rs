//! Property gate: `do_move` followed by `undo_move` restores the position
//! *exactly*.
//!
//! Positions are drawn as random **reachable** positions — a proptest-generated
//! root index plus a vector of selection indices walks a legal game, each index
//! picking one of the position's legal moves (`i % legal.len()`). That keeps
//! every sampled position a genuine game position (kings present, no nifu, hand
//! counts consistent) rather than the free-form board soup
//! `sfen_roundtrip.rs`'s `arb_position` builds for the parser round trip. Roots
//! other than `startpos` seed the walk in already-sharp positions so checks,
//! drops and promotions are sampled from ply 0 rather than only where a random
//! line happens to wander.
//!
//! At **every** position the walk visits, **every** legal move is played and
//! immediately undone, and the restored position is compared against a pre-move
//! clone on:
//!
//! * the structural [`PartialEq`] (board, hands, side to move, ply, history),
//! * the SFEN rendering (a second, independent view of board + hands + side),
//! * every incrementally-maintained Zobrist key the crate exposes — `key`,
//!   `board_key`, `hand_key`, `pawn_key`, `minor_piece_key` and both
//!   `non_pawn_key` halves. These are excluded from `PartialEq` (they are a
//!   derived cache), so the structural comparison alone would not catch an
//!   asymmetric XOR in the undo path.
//!
//! The whole generated line is then unwound move by move and checked against
//! the snapshot taken before each ply, ending back at the root, so a defect that
//! only shows up several plies deep still lands.
//!
//! `#[cfg_attr(miri, ignore)]` matches the existing `sfen_roundtrip.rs` suite:
//! proptest's case volume is far too slow under Miri, and the miri gate has no
//! interest in it. Failure persistence is left at proptest's default, also as in
//! `sfen_roundtrip.rs` — a `proptest-regressions/` file is written only when a
//! case actually fails, and none is committed today.

use proptest::prelude::*;
use proptest::test_runner::TestCaseResult;
use yorkie_state::{Color, Move, Position, Undo, format_sfen, format_usi_move, parse_sfen};

/// Roots the random lines start from: `startpos` plus a side in check, a
/// drop-rich board, a tactical middlegame and a promotion-zone tangle.
const ROOT_SFENS: &[&str] = &[
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1", // startpos
    "4k4/9/4r4/9/9/9/4K3B/9/9 b RG2gs2n3p 1",                          // check-evasion
    "k8/1P7/G8/1N2P4/9/9/9/9/8K b 2PG2pg 1",                           // drop-heavy
    "l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1", // mid-game-tactical
    "4k4/3P3+PL/2N2PR2/1L2BNS2/4N4/9/9/9/4K4 b - 1", // promotion-zone-edges
];

/// Plies walked from the root. Bounded to keep the standard `cargo test` gate
/// fast — each ply already costs one do/undo pair per legal move (30-100+).
const MAX_PLIES: usize = 40;

/// A root index plus a line of legal-move selectors. Each selector picks
/// `legal[i % legal.len()]` at the position it is applied to; the walk stops
/// early at a terminal position.
fn arb_line() -> impl Strategy<Value = (usize, Vec<u16>)> {
    (
        0..ROOT_SFENS.len(),
        proptest::collection::vec(any::<u16>(), 0..=MAX_PLIES),
    )
}

/// Legal moves of `pos`, in generator order.
fn legal_moves(pos: &Position) -> Vec<Move> {
    let mut buf = Vec::with_capacity(64);
    pos.generate_legal_all(&mut buf);
    buf
}

fn root_position(root: usize) -> Position {
    parse_sfen(ROOT_SFENS[root]).expect("root sfen parses")
}

/// Compare every observable field a correct `undo_move` must restore. Returns
/// `Err` with a human-readable reason on the first mismatch, so callers can feed
/// it straight to `prop_assert!`.
fn diff(a: &Position, b: &Position) -> Result<(), String> {
    if a != b {
        return Err(format!(
            "structural mismatch: {} != {}",
            format_sfen(a),
            format_sfen(b)
        ));
    }
    // Independent of `PartialEq`: the SFEN rendering re-derives board, hands and
    // side to move from scratch.
    if format_sfen(a) != format_sfen(b) {
        return Err(format!(
            "sfen mismatch: {} != {}",
            format_sfen(a),
            format_sfen(b)
        ));
    }
    if a.ply() != b.ply() {
        return Err(format!("ply mismatch: {} != {}", a.ply(), b.ply()));
    }
    if a.side_to_move() != b.side_to_move() {
        return Err(format!(
            "side-to-move mismatch: {:?} != {:?}",
            a.side_to_move(),
            b.side_to_move()
        ));
    }
    // The Zobrist cache is excluded from `PartialEq`, so it needs its own pass.
    let keys: [(&str, u64, u64); 7] = [
        ("key", a.key(), b.key()),
        ("board_key", a.board_key(), b.board_key()),
        ("hand_key", a.hand_key(), b.hand_key()),
        ("pawn_key", a.pawn_key(), b.pawn_key()),
        ("minor_piece_key", a.minor_piece_key(), b.minor_piece_key()),
        (
            "non_pawn_key(Black)",
            a.non_pawn_key(Color::Black),
            b.non_pawn_key(Color::Black),
        ),
        (
            "non_pawn_key(White)",
            a.non_pawn_key(Color::White),
            b.non_pawn_key(Color::White),
        ),
    ];
    for (name, lhs, rhs) in keys {
        if lhs != rhs {
            return Err(format!("{name} mismatch: {lhs:#018x} != {rhs:#018x}"));
        }
    }
    Ok(())
}

/// Play and immediately undo **every** legal move of `pos`, checking each time
/// that the position is bit-for-bit what it was.
fn check_every_legal_move_round_trips(pos: &mut Position) -> TestCaseResult {
    let before = pos.clone();
    for mv in legal_moves(pos) {
        let undo = pos.do_move(mv);
        pos.undo_move(mv, undo);
        let outcome = diff(&before, pos);
        prop_assert!(
            outcome.is_ok(),
            "`{}` from {}: {}",
            format_usi_move(mv),
            format_sfen(&before),
            outcome.unwrap_err(),
        );
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 64,
        ..ProptestConfig::default()
    })]

    /// Playing any legal move and undoing it is the identity on the position —
    /// checked at every position the random line visits.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn do_then_undo_restores_position((root, line) in arb_line()) {
        let mut pos = root_position(root);
        for sel in &line {
            check_every_legal_move_round_trips(&mut pos)?;
            let legal = legal_moves(&pos);
            if legal.is_empty() {
                break;
            }
            pos.do_move(legal[*sel as usize % legal.len()]);
        }
        check_every_legal_move_round_trips(&mut pos)?;
    }

    /// Unwinding a whole generated line returns to the root, checking the
    /// intermediate position after every single undo as well.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn unwinding_a_whole_line_returns_to_the_root((root, line) in arb_line()) {
        // Replay the line, snapshotting each position *before* its move so the
        // unwind can be checked ply by ply.
        let mut pos = root_position(root);
        let mut stack: Vec<(Move, Undo, Position)> = Vec::new();
        for sel in &line {
            let legal = legal_moves(&pos);
            if legal.is_empty() {
                break;
            }
            let mv = legal[*sel as usize % legal.len()];
            let snapshot = pos.clone();
            let undo = pos.do_move(mv);
            stack.push((mv, undo, snapshot));
        }

        while let Some((mv, undo, snapshot)) = stack.pop() {
            pos.undo_move(mv, undo);
            let outcome = diff(&snapshot, &pos);
            prop_assert!(
                outcome.is_ok(),
                "unwinding `{}`: {}",
                format_usi_move(mv),
                outcome.unwrap_err(),
            );
        }

        let outcome = diff(&root_position(root), &pos);
        prop_assert!(outcome.is_ok(), "{}", outcome.unwrap_err());
    }
}

/// A non-randomized anchor: one move, done and undone, checked without proptest
/// so the miri gate (which skips the proptest cases) still exercises the
/// `do_move`/`undo_move` symmetry once.
#[test]
fn single_move_round_trip_from_startpos() {
    let mut pos = Position::startpos();
    let before = pos.clone();
    let mv = legal_moves(&pos)[0];
    let undo = pos.do_move(mv);
    pos.undo_move(mv, undo);
    assert_eq!(diff(&before, &pos), Ok(()));
}
