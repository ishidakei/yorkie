//! Invariants of the move generators over random reachable positions.
//!
//! Positions are sampled by walking a legal game, as in `do_undo_roundtrip.rs`.
//! Free-form random boards would not do: the generators assume a well-formed
//! position, with both kings present and the side not to move not already in
//! check, and `Position::is_legal` carries an explicit entry contract.
//!
//! Every position the walk visits is checked, not only the one it ends on,
//! which is what samples the in-check `generate_evasions` path often enough to
//! matter.

use std::collections::HashSet;

use proptest::prelude::*;
use proptest::test_runner::TestCaseResult;
use yorkie_state::{ExtMove, Move, Position, format_sfen, format_usi_move, parse_sfen};

/// Roots the random lines start from: `startpos`, plus already-sharp positions
/// so the evasion, drop and promotion branches are sampled from ply 0.
const ROOT_SFENS: &[&str] = &[
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1", // startpos
    "4k4/9/4r4/9/9/9/4K3B/9/9 b RG2gs2n3p 1",                          // check-evasion
    "k8/1P7/G8/1N2P4/9/9/9/9/8K b 2PG2pg 1",                           // drop-heavy
    "l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1", // mid-game-tactical
    "4k4/3P3+PL/2N2PR2/1L2BNS2/4N4/9/9/9/4K4 b - 1", // promotion-zone-edges
];

/// Upper bound on the plies walked from the root.
const MAX_PLIES: usize = 40;

fn arb_line() -> impl Strategy<Value = (usize, Vec<u16>)> {
    (
        0..ROOT_SFENS.len(),
        proptest::collection::vec(any::<u16>(), 0..=MAX_PLIES),
    )
}

fn legal_moves(pos: &Position) -> Vec<Move> {
    let mut buf = Vec::with_capacity(64);
    pos.generate_legal_all(&mut buf);
    buf
}

/// Every position `line` visits from `ROOT_SFENS[root]`, root included, the
/// walk stopping early at a terminal position.
fn walk(root: usize, line: &[u16]) -> Vec<Position> {
    let mut pos = parse_sfen(ROOT_SFENS[root]).expect("root sfen parses");
    let mut visited = Vec::with_capacity(line.len() + 1);
    for sel in line {
        let legal = legal_moves(&pos);
        if legal.is_empty() {
            break;
        }
        let mv = legal[*sel as usize % legal.len()];
        visited.push(pos.clone());
        pos.do_move(mv);
    }
    visited.push(pos);
    visited
}

/// Apply `mv` to a scratch copy of the board and report whether the mover's
/// king is left unattacked.
///
/// Deliberately not `do_move`, which maintains incremental state that assumes a
/// legal move: the point here is to judge moves that may be illegal.
fn king_safe_after(pos: &Position, mv: Move) -> bool {
    let us = pos.side_to_move();
    let mut scratch = pos.clone();
    {
        let mut board = scratch.board_mut();
        if !mv.is_drop() {
            board.set(mv.from_sq(), None);
        }
        board.set(mv.to_sq(), Some(mv.moved_piece_after()));
    }
    match scratch.king_square(us) {
        // `discount == sq` is harmless: occupancy on the tested square never
        // blocks a ray that terminates there.
        Some(king) => !scratch.is_attacked_discounting(king, us.flip(), king),
        None => false,
    }
}

/// The raw pseudo-legal candidates for the position's check state, with `all`
/// set — the generator output `generate_legal_all` filters.
fn pseudo_legal_candidates(pos: &Position) -> Vec<Move> {
    let mut buf: Vec<ExtMove> = Vec::with_capacity(64);
    if pos.in_check() {
        pos.generate_evasions(true, &mut buf);
    } else {
        pos.generate_non_evasions(true, &mut buf);
    }
    buf.into_iter().map(|em| em.mv).collect()
}

fn as_set(moves: &[Move]) -> HashSet<Move> {
    moves.iter().copied().collect()
}

/// Render a move set as sorted USI strings, for a legible assertion message.
fn describe(moves: &HashSet<Move>) -> Vec<String> {
    let mut out: Vec<String> = moves.iter().map(|m| format_usi_move(*m)).collect();
    out.sort();
    out
}

/// Every generated legal move is distinct, well-formed, and survives a
/// `move16` round trip.
fn check_uniqueness_and_shape(pos: &Position) -> TestCaseResult {
    let legal = legal_moves(pos);
    let unique = as_set(&legal);
    prop_assert_eq!(
        unique.len(),
        legal.len(),
        "duplicate legal move at {}: {:?}",
        format_sfen(pos),
        describe(&unique),
    );

    for mv in legal {
        prop_assert!(
            mv.is_ok(),
            "`{}` at {} is not a well-formed move",
            format_usi_move(mv),
            format_sfen(pos),
        );
        prop_assert_eq!(
            pos.to_move(mv.move16()),
            Some(mv),
            "`{}` at {} did not survive a move16 round trip",
            format_usi_move(mv),
            format_sfen(pos),
        );
    }
    Ok(())
}

/// Every generated legal move passes the crate's legality predicates, and
/// `is_legal` agrees with [`leaves_own_king_safe`] across the whole candidate
/// set — in both directions, so it also rejects nothing legal.
fn check_legality_predicates(pos: &Position) -> TestCaseResult {
    for mv in legal_moves(pos) {
        prop_assert!(
            pos.is_legal(mv),
            "generated `{}` at {} fails is_legal",
            format_usi_move(mv),
            format_sfen(pos),
        );
        prop_assert!(
            pos.pseudo_legal(mv, true),
            "generated `{}` at {} fails pseudo_legal",
            format_usi_move(mv),
            format_sfen(pos),
        );
    }

    for mv in pseudo_legal_candidates(pos) {
        prop_assert_eq!(
            pos.is_legal(mv),
            king_safe_after(pos, mv),
            "is_legal disagrees with the king-safety oracle on `{}` at {}",
            format_usi_move(mv),
            format_sfen(pos),
        );
    }
    Ok(())
}

/// The pseudo-legal-plus-filter paths and the direct legal-generation path
/// agree on the resulting move set.
fn check_generator_paths_agree(pos: &Position) -> TestCaseResult {
    let direct = as_set(&legal_moves(pos));

    // The check-state generator, filtered by `is_legal`.
    let filtered: HashSet<Move> = pseudo_legal_candidates(pos)
        .into_iter()
        .filter(|m| pos.is_legal(*m))
        .collect();
    prop_assert_eq!(
        describe(&filtered),
        describe(&direct),
        "check-state generator + is_legal disagrees with generate_legal_all at {}",
        format_sfen(pos),
    );

    // The captures/quiets split, which must partition the same candidate
    // space. `generate_evasions` has no such split, so an in-check position
    // stops above.
    if pos.in_check() {
        return Ok(());
    }

    let mut captures: Vec<ExtMove> = Vec::with_capacity(32);
    pos.generate_captures(true, &mut captures);
    let mut quiets: Vec<ExtMove> = Vec::with_capacity(64);
    pos.generate_quiets(true, &mut quiets);

    let capture_set: HashSet<Move> = captures.iter().map(|em| em.mv).collect();
    let quiet_set: HashSet<Move> = quiets.iter().map(|em| em.mv).collect();
    let overlap: HashSet<Move> = capture_set.intersection(&quiet_set).copied().collect();
    prop_assert!(
        overlap.is_empty(),
        "captures and quiets overlap at {}: {:?}",
        format_sfen(pos),
        describe(&overlap),
    );

    let split: HashSet<Move> = capture_set
        .union(&quiet_set)
        .copied()
        .filter(|m| pos.is_legal(*m))
        .collect();
    prop_assert_eq!(
        describe(&split),
        describe(&direct),
        "captures+quiets + is_legal disagrees with generate_legal_all at {}",
        format_sfen(pos),
    );
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 128,
        ..ProptestConfig::default()
    })]

    /// All three invariant families, at every position the random line visits.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn movegen_invariants_hold_along_a_random_line((root, line) in arb_line()) {
        for pos in walk(root, &line) {
            check_uniqueness_and_shape(&pos)?;
            check_legality_predicates(&pos)?;
            check_generator_paths_agree(&pos)?;
        }
    }
}

/// A non-randomized anchor, so a Miri run — which skips the proptest cases
/// as far too slow — still exercises the invariants once.
#[test]
fn startpos_legal_moves_are_unique_and_legal() {
    let pos = Position::startpos();
    let legal = legal_moves(&pos);
    assert_eq!(as_set(&legal).len(), legal.len());
    for mv in &legal {
        assert!(pos.is_legal(*mv));
        assert!(pos.pseudo_legal(*mv, true));
        assert!(king_safe_after(&pos, *mv));
    }
}
