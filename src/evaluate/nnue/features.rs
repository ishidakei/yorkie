use arrayvec::ArrayVec;

use super::types::{FeatureIndex, FeatureList, NUM_FEATURES};
use crate::movetypes::*;
use crate::position::*;
use crate::types::*;

const F_HAND_PAWN: usize = 0;
const E_HAND_PAWN: usize = F_HAND_PAWN + 19;
const F_HAND_LANCE: usize = E_HAND_PAWN + 19;
const E_HAND_LANCE: usize = F_HAND_LANCE + 5;
const F_HAND_KNIGHT: usize = E_HAND_LANCE + 5;
const E_HAND_KNIGHT: usize = F_HAND_KNIGHT + 5;
const F_HAND_SILVER: usize = E_HAND_KNIGHT + 5;
const E_HAND_SILVER: usize = F_HAND_SILVER + 5;
const F_HAND_GOLD: usize = E_HAND_SILVER + 5;
const E_HAND_GOLD: usize = F_HAND_GOLD + 5;
const F_HAND_BISHOP: usize = E_HAND_GOLD + 5;
const E_HAND_BISHOP: usize = F_HAND_BISHOP + 3;
const F_HAND_ROOK: usize = E_HAND_BISHOP + 3;
const E_HAND_ROOK: usize = F_HAND_ROOK + 3;
const FE_HAND_END: usize = E_HAND_ROOK + 3;

const F_PAWN: usize = FE_HAND_END;
const E_PAWN: usize = F_PAWN + 81;
const F_LANCE: usize = E_PAWN + 81;
const E_LANCE: usize = F_LANCE + 81;
const F_KNIGHT: usize = E_LANCE + 81;
const E_KNIGHT: usize = F_KNIGHT + 81;
const F_SILVER: usize = E_KNIGHT + 81;
const E_SILVER: usize = F_SILVER + 81;
const F_GOLD: usize = E_SILVER + 81;
const E_GOLD: usize = F_GOLD + 81;
const F_BISHOP: usize = E_GOLD + 81;
const E_BISHOP: usize = F_BISHOP + 81;
const F_HORSE: usize = E_BISHOP + 81;
const E_HORSE: usize = F_HORSE + 81;
const F_ROOK: usize = E_HORSE + 81;
const E_ROOK: usize = F_ROOK + 81;
const F_DRAGON: usize = E_ROOK + 81;
const E_DRAGON: usize = F_DRAGON + 81;
const FE_END: usize = E_DRAGON + 81;

const SQ_NB: usize = Square::NUM;
const E_KING: usize = FE_END + SQ_NB;

const SQ_K_COUNT: usize = 5 * 9;

const PIECE_NUMBER_NB: usize = 40;

#[inline]
fn from_persp(sq: Square, persp: Color) -> Square {
    if persp == Color::BLACK { sq } else { sq.inverse() }
}

#[inline]
fn needs_mirror(king_sq_persp: Square) -> bool {
    File::new(king_sq_persp).0 >= 5
}

#[inline]
fn mirror_if_needed(sq: Square, mirror: bool) -> Square {
    if mirror { sq.inverse_file() } else { sq }
}

#[inline]
fn effective_board_piece_type(pt: PieceType) -> PieceType {
    match pt {
        PieceType::PRO_PAWN | PieceType::PRO_LANCE | PieceType::PRO_KNIGHT | PieceType::PRO_SILVER => PieceType::GOLD,
        _ => pt,
    }
}

#[inline]
fn board_plane(pt: PieceType, is_friend: bool) -> usize {
    match effective_board_piece_type(pt) {
        PieceType::PAWN => {
            if is_friend {
                F_PAWN
            } else {
                E_PAWN
            }
        }
        PieceType::LANCE => {
            if is_friend {
                F_LANCE
            } else {
                E_LANCE
            }
        }
        PieceType::KNIGHT => {
            if is_friend {
                F_KNIGHT
            } else {
                E_KNIGHT
            }
        }
        PieceType::SILVER => {
            if is_friend {
                F_SILVER
            } else {
                E_SILVER
            }
        }
        PieceType::BISHOP => {
            if is_friend {
                F_BISHOP
            } else {
                E_BISHOP
            }
        }
        PieceType::ROOK => {
            if is_friend {
                F_ROOK
            } else {
                E_ROOK
            }
        }
        PieceType::GOLD => {
            if is_friend {
                F_GOLD
            } else {
                E_GOLD
            }
        }
        PieceType::HORSE => {
            if is_friend {
                F_HORSE
            } else {
                E_HORSE
            }
        }
        PieceType::DRAGON => {
            if is_friend {
                F_DRAGON
            } else {
                E_DRAGON
            }
        }
        _ => unreachable!("king / empty / occupied have no board plane"),
    }
}

#[inline]
fn hand_plane(pt: PieceType, is_friend: bool) -> usize {
    match pt {
        PieceType::PAWN => {
            if is_friend {
                F_HAND_PAWN
            } else {
                E_HAND_PAWN
            }
        }
        PieceType::LANCE => {
            if is_friend {
                F_HAND_LANCE
            } else {
                E_HAND_LANCE
            }
        }
        PieceType::KNIGHT => {
            if is_friend {
                F_HAND_KNIGHT
            } else {
                E_HAND_KNIGHT
            }
        }
        PieceType::SILVER => {
            if is_friend {
                F_HAND_SILVER
            } else {
                E_HAND_SILVER
            }
        }
        PieceType::GOLD => {
            if is_friend {
                F_HAND_GOLD
            } else {
                E_HAND_GOLD
            }
        }
        PieceType::BISHOP => {
            if is_friend {
                F_HAND_BISHOP
            } else {
                E_HAND_BISHOP
            }
        }
        PieceType::ROOK => {
            if is_friend {
                F_HAND_ROOK
            } else {
                E_HAND_ROOK
            }
        }
        _ => unreachable!("only the 7 hand piece-types are valid"),
    }
}

#[inline]
fn encode_feature(sq_k_code: usize, p_adj: usize) -> FeatureIndex {
    debug_assert!(sq_k_code < SQ_K_COUNT);
    debug_assert!(p_adj < E_KING);
    let idx = E_KING * sq_k_code + p_adj;
    debug_assert!(idx < NUM_FEATURES);
    idx as FeatureIndex
}

pub fn active_features(pos: &Position, perspective: Color) -> FeatureList {
    let mut list = FeatureList::new();

    let own_king_persp = from_persp(pos.king_square(perspective), perspective);
    let mirror = needs_mirror(own_king_persp);
    let sq_k_code = mirror_if_needed(own_king_persp, mirror).0 as usize;

    for sq in pos.occupied_bb() {
        let pc = pos.piece_on(sq);
        let piece_color = Color::new(pc);
        let piece_type = PieceType::new(pc);
        let is_friend = piece_color == perspective;
        let sq_persp = mirror_if_needed(from_persp(sq, perspective), mirror);
        let sq_code = sq_persp.0 as usize;

        let p_adj = if piece_type == PieceType::KING {
            // Both kings collapse to the shared `[FE_END, E_KING)` plane.
            FE_END + sq_code
        } else {
            board_plane(piece_type, is_friend) + sq_code
        };
        list.push(encode_feature(sq_k_code, p_adj));
    }

    for &hand_color in Color::ALL.iter() {
        let is_friend = hand_color == perspective;
        let hand = pos.hand(hand_color);
        for &pt in PieceType::ALL_HAND.iter() {
            let count = hand.num(pt);
            let base = hand_plane(pt, is_friend);
            for i in 1..=count as usize {
                let p_adj = base + i;
                list.push(encode_feature(sq_k_code, p_adj));
            }
        }
    }

    let pad_index = encode_feature(sq_k_code, 0);
    while list.len() < PIECE_NUMBER_NB {
        list.push(pad_index);
    }

    debug_assert_eq!(list.len(), PIECE_NUMBER_NB);
    list
}

pub fn requires_full_refresh(_pos: &Position, mv: Move, perspective: Color) -> bool {
    if mv.is_drop() {
        return false;
    }
    let pc = mv.piece_moved_before_move();
    PieceType::new(pc) == PieceType::KING && Color::new(pc) == perspective
}

pub fn feature_diff(pos: &mut Position, mv: Move, perspective: Color) -> (FeatureList, FeatureList) {
    debug_assert!(!requires_full_refresh(pos, mv, perspective));

    let before = active_features(pos, perspective);
    let gives_check = pos.gives_check(mv);
    pos.do_move(mv, gives_check);
    let after = active_features(pos, perspective);
    pos.undo_move(mv);

    let mut sorted_before: ArrayVec<FeatureIndex, { super::types::FEATURE_LIST_CAPACITY }> = before.clone();
    let mut sorted_after: ArrayVec<FeatureIndex, { super::types::FEATURE_LIST_CAPACITY }> = after.clone();
    sorted_before.sort_unstable();
    sorted_after.sort_unstable();

    let mut removed = FeatureList::new();
    let mut added = FeatureList::new();
    let (mut i, mut j) = (0usize, 0usize);
    while i < sorted_before.len() && j < sorted_after.len() {
        match sorted_before[i].cmp(&sorted_after[j]) {
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => {
                removed.push(sorted_before[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                added.push(sorted_after[j]);
                j += 1;
            }
        }
    }
    while i < sorted_before.len() {
        removed.push(sorted_before[i]);
        i += 1;
    }
    while j < sorted_after.len() {
        added.push(sorted_after[j]);
        j += 1;
    }

    (removed, added)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn start_pos() -> Position {
        Position::new()
    }

    fn list_as_sorted_vec(list: &FeatureList) -> Vec<FeatureIndex> {
        let mut v: Vec<_> = list.iter().copied().collect();
        v.sort_unstable();
        v
    }

    fn multiset_apply(before: &[FeatureIndex], removed: &[FeatureIndex], added: &[FeatureIndex]) -> Vec<FeatureIndex> {
        let mut counts: std::collections::BTreeMap<FeatureIndex, i32> = std::collections::BTreeMap::new();
        for &x in before {
            *counts.entry(x).or_insert(0) += 1;
        }
        for &x in removed {
            *counts.entry(x).or_insert(0) -= 1;
        }
        for &x in added {
            *counts.entry(x).or_insert(0) += 1;
        }
        let mut out = Vec::new();
        for (&k, &c) in &counts {
            assert!(c >= 0, "multiset_apply: negative count for {k} after applying removed list");
            for _ in 0..c {
                out.push(k);
            }
        }
        out
    }

    #[test]
    fn plane_count_sanity() {
        assert_eq!(FE_HAND_END, 90);
        assert_eq!(FE_END, 1_548);
        assert_eq!(E_KING, 1_629);
        assert_eq!(SQ_K_COUNT * E_KING, NUM_FEATURES);
    }

    #[test]
    fn active_features_start_position_has_40_per_side() {
        let pos = start_pos();
        let black = active_features(&pos, Color::BLACK);
        let white = active_features(&pos, Color::WHITE);
        assert_eq!(black.len(), 40);
        assert_eq!(white.len(), 40);
    }

    #[test]
    fn active_features_emits_exactly_40_for_every_position() {
        let sfens = [
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
            "9/9/4K4/9/9/9/4k4/9/9 b - 1",
            "4G4/4K4/9/9/9/9/9/4k4/4g4 b - 1",
            "4k4/9/9/9/9/9/9/9/4K4 b P 1",
        ];
        for sfen in sfens.iter() {
            let pos = Position::new_from_sfen(sfen).expect("valid sfen");
            for &persp in Color::ALL.iter() {
                let list = active_features(&pos, persp);
                assert_eq!(
                    list.len(),
                    40,
                    "perspective={persp:?} sfen={sfen}: expected 40 features, got {}",
                    list.len()
                );
            }
        }
    }

    #[test]
    fn active_features_is_deterministic() {
        let pos = start_pos();
        let a = list_as_sorted_vec(&active_features(&pos, Color::BLACK));
        let b = list_as_sorted_vec(&active_features(&pos, Color::BLACK));
        assert_eq!(a, b);
    }

    #[test]
    fn active_features_respects_bounds() {
        let pos = start_pos();
        for &persp in Color::ALL.iter() {
            let list = active_features(&pos, persp);
            assert!(list.len() <= list.capacity());
            for idx in list.iter() {
                assert!((*idx as usize) < NUM_FEATURES);
            }
        }
    }

    #[test]
    fn active_features_kings_share_one_plane() {
        let pos = start_pos();
        for &persp in Color::ALL.iter() {
            let own_king_persp = from_persp(pos.king_square(persp), persp);
            let mirror = needs_mirror(own_king_persp);
            let sq_k_code = mirror_if_needed(own_king_persp, mirror).0 as usize;
            let plane_lo = (E_KING * sq_k_code + FE_END) as FeatureIndex;
            let plane_hi = (E_KING * sq_k_code + E_KING) as FeatureIndex;

            let list = active_features(&pos, persp);
            let kings_in_plane: Vec<_> = list.iter().copied().filter(|&f| f >= plane_lo && f < plane_hi).collect();
            assert_eq!(kings_in_plane.len(), 2, "both kings should land in [{plane_lo}, {plane_hi})");
        }
    }

    #[test]
    fn requires_full_refresh_flags_king_moves() {
        let pos = Position::new_from_sfen("4k4/9/9/9/9/9/9/9/4K4 b - 1").expect("valid sfen");
        let king_move = Move::new_from_usi_str("5i5h", &pos).expect("legal king move");

        assert!(requires_full_refresh(&pos, king_move, Color::BLACK));
        assert!(!requires_full_refresh(&pos, king_move, Color::WHITE));

        let pos2 = start_pos();
        let pawn_push = Move::new_from_usi_str("7g7f", &pos2).expect("legal pawn push");
        assert!(!requires_full_refresh(&pos2, pawn_push, Color::BLACK));
        assert!(!requires_full_refresh(&pos2, pawn_push, Color::WHITE));
    }

    #[test]
    fn requires_full_refresh_drops_are_not_refresh() {
        let pos = Position::new_from_sfen("4k4/9/9/9/9/9/9/9/4K4 b P 1").expect("valid sfen with pawn in hand");
        let drop_mv = Move::new_from_usi_str("P*5f", &pos).expect("legal pawn drop");
        assert!(!requires_full_refresh(&pos, drop_mv, Color::BLACK));
        assert!(!requires_full_refresh(&pos, drop_mv, Color::WHITE));
    }

    fn check_diff_invariant(mut pos: Position, usi: &str) {
        let mv = Move::new_from_usi_str(usi, &pos).expect("legal move");
        for &persp in Color::ALL.iter() {
            assert!(!requires_full_refresh(&pos, mv, persp), "test move must not be a king move");
            let before = list_as_sorted_vec(&active_features(&pos, persp));

            let (removed, added) = feature_diff(&mut pos, mv, persp);
            let removed_sorted = list_as_sorted_vec(&removed);
            let added_sorted = list_as_sorted_vec(&added);

            let expected_after = multiset_apply(&before, &removed_sorted, &added_sorted);

            let gives_check = pos.gives_check(mv);
            pos.do_move(mv, gives_check);
            let actual_after = list_as_sorted_vec(&active_features(&pos, persp));
            pos.undo_move(mv);

            assert_eq!(
                actual_after, expected_after,
                "perspective={persp:?}: diff invariant violated for move {usi}"
            );
        }
    }

    #[test]
    fn feature_diff_non_capture_move() {
        check_diff_invariant(start_pos(), "7g7f");
    }

    #[test]
    fn feature_diff_capture_move() {
        let pos = Position::new_from_sfen("4k4/1p7/9/9/9/9/9/1R7/4K4 b - 1").expect("valid sfen");
        check_diff_invariant(pos, "8h8b");
    }

    #[test]
    fn feature_diff_drop_move() {
        let pos = Position::new_from_sfen("4k4/9/9/9/9/9/9/9/4K4 b P 1").expect("valid sfen with pawn in hand");
        check_diff_invariant(pos, "P*5e");
    }

    #[test]
    fn feature_diff_promotion_move() {
        let pos = Position::new_from_sfen("4k4/9/9/1P7/9/9/9/9/4K4 b - 1").expect("valid sfen");
        check_diff_invariant(pos, "8d8c+");
    }
}
