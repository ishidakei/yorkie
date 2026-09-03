//! Attack detection, king lookup, and the drop-legality (`uchifuzume`)
//! predicate — the board-query surface the search-side move generators and the
//! mate and SEE oracles consume.
//!
//! Move generation itself lives in [`crate::search_movegen`]. Uchifuzume
//! (打ち歩詰め) and nifu (二歩) are enforced there at drop-generation time, as
//! the reference's `GenerateDropMoves` does, so no legal-move list is filtered
//! for them after the fact.

use crate::board::Board;
use crate::color::Color;
use crate::move_::Move;
#[cfg(test)]
use crate::piece::Piece;
use crate::piece::PieceKind;
use crate::position::Position;
use crate::square::Square;

pub(crate) const PAWN_STEPS: &[(i8, i8)] = &[(0, -1)];
pub(crate) const KNIGHT_STEPS: &[(i8, i8)] = &[(1, -2), (-1, -2)];
pub(crate) const SILVER_STEPS: &[(i8, i8)] = &[(0, -1), (1, -1), (-1, -1), (1, 1), (-1, 1)];
pub(crate) const GOLD_STEPS: &[(i8, i8)] = &[(0, -1), (1, -1), (-1, -1), (1, 0), (-1, 0), (0, 1)];
pub(crate) const KING_STEPS: &[(i8, i8)] = &[
    (0, -1),
    (1, -1),
    (-1, -1),
    (1, 0),
    (-1, 0),
    (0, 1),
    (1, 1),
    (-1, 1),
];
// The direction tables feed only [`movement`], which only the scanning oracles
// use. The step tables above stay ungated because `crate::bitboard` builds its
// production attack tables from them.
#[cfg(test)]
const KING_ORTH_STEPS: &[(i8, i8)] = &[(0, -1), (1, 0), (-1, 0), (0, 1)];
#[cfg(test)]
const KING_DIAG_STEPS: &[(i8, i8)] = &[(1, -1), (-1, -1), (1, 1), (-1, 1)];
#[cfg(test)]
const LANCE_DIRS: &[(i8, i8)] = &[(0, -1)];
#[cfg(test)]
const BISHOP_DIRS: &[(i8, i8)] = &[(1, -1), (-1, -1), (1, 1), (-1, 1)];
#[cfg(test)]
const ROOK_DIRS: &[(i8, i8)] = &[(0, -1), (0, 1), (1, 0), (-1, 0)];
#[cfg(test)]
const NO_DIRS: &[(i8, i8)] = &[];

pub(crate) fn step_signed(sq: Square, df: i8, dr: i8) -> Option<Square> {
    let f = sq.file() as i8 + df;
    let r = sq.rank() as i8 + dr;
    if (0..Square::FILES as i8).contains(&f) && (0..Square::RANKS as i8).contains(&r) {
        Square::new(f as u8, r as u8)
    } else {
        None
    }
}

pub(crate) fn is_in_promotion_zone(sq: Square, color: Color) -> bool {
    match color {
        Color::Black => sq.rank() < 3,
        Color::White => sq.rank() >= 6,
    }
}

#[cfg(test)]
type StepTable = &'static [(i8, i8)];

#[cfg(test)]
pub(crate) fn movement(piece: Piece) -> (StepTable, StepTable) {
    match (piece.kind, piece.promoted) {
        (PieceKind::Pawn, false) => (PAWN_STEPS, NO_DIRS),
        (PieceKind::Lance, false) => (NO_DIRS, LANCE_DIRS),
        (PieceKind::Knight, false) => (KNIGHT_STEPS, NO_DIRS),
        (PieceKind::Silver, false) => (SILVER_STEPS, NO_DIRS),
        (PieceKind::Gold, _)
        | (PieceKind::Pawn | PieceKind::Lance | PieceKind::Knight | PieceKind::Silver, true) => {
            (GOLD_STEPS, NO_DIRS)
        }
        (PieceKind::Bishop, false) => (NO_DIRS, BISHOP_DIRS),
        (PieceKind::Bishop, true) => (KING_ORTH_STEPS, BISHOP_DIRS),
        (PieceKind::Rook, false) => (NO_DIRS, ROOK_DIRS),
        (PieceKind::Rook, true) => (KING_DIAG_STEPS, ROOK_DIRS),
        (PieceKind::King, _) => (KING_STEPS, NO_DIRS),
    }
}

#[cfg(test)]
pub(crate) fn dr_sign_for(color: Color) -> i8 {
    match color {
        Color::Black => 1,
        Color::White => -1,
    }
}

pub(crate) fn find_king(board: &Board, color: Color) -> Square {
    try_find_king(board, color).unwrap_or_else(|| panic!("position has no {color:?} king"))
}

/// Like `find_king`, but returns `None` instead of panicking when no king of
/// `color` is on the board. A pseudo-legal probe move that captures a king
/// leaves exactly such a transient scratch state.
pub(crate) fn try_find_king(board: &Board, color: Color) -> Option<Square> {
    board
        .pieces_pattern(color, crate::board::pat::KING)
        .squares()
        .next()
}

/// An 81-square scan form of [`try_find_king`].
#[cfg(test)]
pub(crate) fn try_find_king_scan(board: &Board, color: Color) -> Option<Square> {
    for index in 0..Square::COUNT as u8 {
        let sq = Square::from_index(index).unwrap();
        if let Some(p) = board.get(sq)
            && p.color == color
            && p.kind == PieceKind::King
        {
            return Some(sq);
        }
    }
    None
}

#[cfg(test)]
fn is_gold_like(p: Piece) -> bool {
    if p.kind == PieceKind::Gold {
        return true;
    }
    if !p.promoted {
        return false;
    }
    matches!(
        p.kind,
        PieceKind::Pawn | PieceKind::Lance | PieceKind::Knight | PieceKind::Silver
    )
}

#[cfg(test)]
fn matches_unpromoted_kind(board: &Board, sq: Square, color: Color, kind: PieceKind) -> bool {
    match board.get(sq) {
        Some(p) => p.color == color && p.kind == kind && !p.promoted,
        None => false,
    }
}

#[cfg(test)]
fn scan_slider(
    board: &Board,
    sq: Square,
    df: i8,
    dr: i8,
    color: Color,
    kind: PieceKind,
    require_unpromoted: bool,
) -> bool {
    let mut cur = sq;
    loop {
        cur = match step_signed(cur, df, dr) {
            Some(s) => s,
            None => return false,
        };
        match board.get(cur) {
            None => continue,
            Some(p) => {
                if p.color != color || p.kind != kind {
                    return false;
                }
                if require_unpromoted && p.promoted {
                    return false;
                }
                return true;
            }
        }
    }
}

/// True iff `sq` is attacked by any piece of `attacker` on `board` — the
/// reference `attackers_to` reverse-symmetry: an attacker on `s` reaches `sq`
/// iff a piece of the same pattern imagined on `sq`, with the perspective
/// flipped for the asymmetric steppers and the lance, reaches `s`.
pub(crate) fn is_attacked_by(board: &Board, sq: Square, attacker: Color) -> bool {
    use crate::bitboard::{
        bishop_attacks, gold_attacks, king_attacks, knight_attacks, lance_attacks, pawn_attacks,
        rook_attacks, silver_attacks,
    };
    use crate::board::pat;

    let occ = board.occupied();
    let opp = attacker.flip();

    // A Black pawn attacks `sq` from where a White pawn on `sq` would attack.
    if !(pawn_attacks(opp, sq) & board.pieces_pattern(attacker, pat::PAWN)).is_empty() {
        return true;
    }
    // Lance: a forward-only slider; the reverse ray flips colour, occ-limited.
    if !(lance_attacks(opp, sq, occ) & board.pieces_pattern(attacker, pat::LANCE)).is_empty() {
        return true;
    }
    if !(knight_attacks(opp, sq) & board.pieces_pattern(attacker, pat::KNIGHT)).is_empty() {
        return true;
    }
    if !(silver_attacks(opp, sq) & board.pieces_pattern(attacker, pat::SILVER)).is_empty() {
        return true;
    }
    // Gold-like (gold + the four promoted minors, one pattern slot).
    if !(gold_attacks(opp, sq) & board.pieces_pattern(attacker, pat::GOLD)).is_empty() {
        return true;
    }

    // Sliders (symmetric, so no colour flip). Horse / dragon add the king ring.
    let king_ring = king_attacks(attacker, sq);
    let bishop = bishop_attacks(sq, occ);
    if !(bishop & board.pieces_pattern(attacker, pat::BISHOP)).is_empty() {
        return true;
    }
    if !((bishop | king_ring) & board.pieces_pattern(attacker, pat::HORSE)).is_empty() {
        return true;
    }
    let rook = rook_attacks(sq, occ);
    if !(rook & board.pieces_pattern(attacker, pat::ROOK)).is_empty() {
        return true;
    }
    if !((rook | king_ring) & board.pieces_pattern(attacker, pat::DRAGON)).is_empty() {
        return true;
    }
    // King (ring is colour-symmetric).
    if !(king_ring & board.pieces_pattern(attacker, pat::KING)).is_empty() {
        return true;
    }
    false
}

/// The set of `attacker` pieces on `board` that attack `sq` — the reverse
/// lookup [`is_attacked_by`] performs, without the short-circuit. Uses the
/// board's current occupancy; see [`attackers_bb_occ`] to supply another.
pub(crate) fn attackers_bb(
    board: &Board,
    sq: Square,
    attacker: Color,
) -> crate::bitboard::Bitboard {
    attackers_bb_occ(board, sq, attacker, board.occupied())
}

/// The set of `attacker` pieces on `board` that attack `sq`, evaluating slider
/// rays against the supplied `occ` rather than `board.occupied()` — the
/// reference `attackers_to(sq, occ)`, per colour.
///
/// The returned squares come from the true board pieces and are **not** masked
/// by `occ`. A caller passing an `occ` with pieces removed must intersect the
/// result itself to get the still-present set.
pub(crate) fn attackers_bb_occ(
    board: &Board,
    sq: Square,
    attacker: Color,
    occ: crate::bitboard::Bitboard,
) -> crate::bitboard::Bitboard {
    use crate::bitboard::{
        bishop_attacks, gold_attacks, king_attacks, knight_attacks, lance_attacks, pawn_attacks,
        rook_attacks, silver_attacks,
    };
    use crate::board::pat;

    let opp = attacker.flip();
    let king_ring = king_attacks(attacker, sq);
    let bishop = bishop_attacks(sq, occ);
    let rook = rook_attacks(sq, occ);
    (pawn_attacks(opp, sq) & board.pieces_pattern(attacker, pat::PAWN))
        | (lance_attacks(opp, sq, occ) & board.pieces_pattern(attacker, pat::LANCE))
        | (knight_attacks(opp, sq) & board.pieces_pattern(attacker, pat::KNIGHT))
        | (silver_attacks(opp, sq) & board.pieces_pattern(attacker, pat::SILVER))
        | (gold_attacks(opp, sq) & board.pieces_pattern(attacker, pat::GOLD))
        | (bishop & board.pieces_pattern(attacker, pat::BISHOP))
        | ((bishop | king_ring) & board.pieces_pattern(attacker, pat::HORSE))
        | (rook & board.pieces_pattern(attacker, pat::ROOK))
        | ((rook | king_ring) & board.pieces_pattern(attacker, pat::DRAGON))
        | (king_ring & board.pieces_pattern(attacker, pat::KING))
}

/// Both colours' attackers of `sq` under occupancy `occ` in a single pass — the
/// reference `Position::attackers_to(sq, occ)` (`position.cpp`), bit-identical
/// to OR-ing the two [`attackers_bb_occ`] calls.
///
/// The two slider walks are shared across colours, and each side's lances are
/// folded into the rook ray: a `color` lance attacks `sq` iff it lies on the
/// reverse ray, so `rook & that ray` recovers the occupancy-cut file segment
/// without a separate lance walk.
pub(crate) fn attackers_to_both(
    board: &Board,
    sq: Square,
    occ: crate::bitboard::Bitboard,
) -> crate::bitboard::Bitboard {
    use crate::bitboard::{
        Bitboard, bishop_attacks, gold_attacks, king_attacks, knight_attacks, lance_step_effect,
        pawn_attacks, rook_attacks, silver_attacks,
    };
    use crate::board::pat;

    // Sliders computed once for both colours; the king ring is colour-symmetric.
    let bishop = bishop_attacks(sq, occ);
    let rook = rook_attacks(sq, occ);
    let king_ring = king_attacks(Color::Black, sq);

    let mut out = Bitboard::EMPTY;
    for color in [Color::Black, Color::White] {
        let opp = color.flip();
        let lance_ray = lance_step_effect(opp, sq);
        out = out
            | (pawn_attacks(opp, sq) & board.pieces_pattern(color, pat::PAWN))
            | ((rook & lance_ray) & board.pieces_pattern(color, pat::LANCE))
            | (knight_attacks(opp, sq) & board.pieces_pattern(color, pat::KNIGHT))
            | (silver_attacks(opp, sq) & board.pieces_pattern(color, pat::SILVER))
            | (gold_attacks(opp, sq) & board.pieces_pattern(color, pat::GOLD))
            | (bishop & board.pieces_pattern(color, pat::BISHOP))
            | ((bishop | king_ring) & board.pieces_pattern(color, pat::HORSE))
            | (rook & board.pieces_pattern(color, pat::ROOK))
            | ((rook | king_ring) & board.pieces_pattern(color, pat::DRAGON))
            | (king_ring & board.pieces_pattern(color, pat::KING));
    }
    out
}

/// An 81-square scanning form of [`is_attacked_by`], derived independently of
/// the piece sets.
#[cfg(test)]
pub(crate) fn is_attacked_by_scan(board: &Board, sq: Square, attacker: Color) -> bool {
    let dr_sign = dr_sign_for(attacker);

    // Pawn (non-promoted only — promoted pawn moves like Gold and is handled below).
    if let Some(src) = step_signed(sq, 0, dr_sign)
        && matches_unpromoted_kind(board, src, attacker, PieceKind::Pawn)
    {
        return true;
    }

    // Knight (non-promoted only).
    for &df in &[-1i8, 1] {
        if let Some(src) = step_signed(sq, df, 2 * dr_sign)
            && matches_unpromoted_kind(board, src, attacker, PieceKind::Knight)
        {
            return true;
        }
    }

    // Silver (non-promoted only).
    for &(df, dr) in SILVER_STEPS {
        if let Some(src) = step_signed(sq, -df, -dr * dr_sign)
            && matches_unpromoted_kind(board, src, attacker, PieceKind::Silver)
        {
            return true;
        }
    }

    // Gold-like: Gold OR promoted {Pawn, Lance, Knight, Silver}.
    for &(df, dr) in GOLD_STEPS {
        if let Some(src) = step_signed(sq, -df, -dr * dr_sign)
            && let Some(p) = board.get(src)
            && p.color == attacker
            && is_gold_like(p)
        {
            return true;
        }
    }

    // 8-neighbour: King; +Bishop (Horse) for orthogonal step; +Rook (Dragon) for diagonal step.
    for &(df, dr) in KING_STEPS {
        let Some(src) = step_signed(sq, df, dr) else {
            continue;
        };
        let Some(p) = board.get(src) else { continue };
        if p.color != attacker {
            continue;
        }
        if p.kind == PieceKind::King {
            return true;
        }
        let is_orth = df == 0 || dr == 0;
        let is_diag = df != 0 && dr != 0;
        if is_orth && p.kind == PieceKind::Bishop && p.promoted {
            return true;
        }
        if is_diag && p.kind == PieceKind::Rook && p.promoted {
            return true;
        }
    }

    // Lance (slider, forward-only; non-promoted — promoted lance moves like Gold).
    if scan_slider(board, sq, 0, dr_sign, attacker, PieceKind::Lance, true) {
        return true;
    }

    // Bishop / Horse (diagonal slider; both promoted states share the slider component).
    for &(df, dr) in BISHOP_DIRS {
        if scan_slider(board, sq, df, dr, attacker, PieceKind::Bishop, false) {
            return true;
        }
    }

    // Rook / Dragon (orthogonal slider).
    for &(df, dr) in ROOK_DIRS {
        if scan_slider(board, sq, df, dr, attacker, PieceKind::Rook, false) {
            return true;
        }
    }

    false
}

/// True iff dropping the pawn `m` describes into `pre` — the **pre-drop**
/// position — would be uchifuzume (打ち歩詰め), an unanswerable pawn-drop mate,
/// and is therefore illegal.
///
/// [`Position::legal_drop`] does the work but requires that the drop checks the
/// enemy king; this wrapper establishes that precondition and returns `false`
/// for every other drop.
pub(crate) fn drop_is_uchifuzume(pre: &Position, m: Move) -> bool {
    if !m.is_drop() || m.dropped_piece_kind() != PieceKind::Pawn {
        return false;
    }
    let us = pre.side_to_move();
    let to = m.to_sq();
    // A dropped pawn checks exactly the single square directly ahead of it, so
    // it can be uchifuzume only when the enemy king sits on that square.
    let Some(king) = try_find_king(pre.board(), us.flip()) else {
        return false;
    };
    if !crate::bitboard::pawn_attacks(us, to).test(king) {
        return false;
    }
    !pre.legal_drop(to)
}

impl Position {
    /// Port of `Position::legal_drop(to)` (`position.cpp`): `true` iff dropping
    /// a side-to-move pawn on `to` is **legal**, that is, not uchifuzume.
    ///
    /// **Precondition:** the dropped pawn on `to` checks the enemy king.
    pub(crate) fn legal_drop(&self, to: Square) -> bool {
        use crate::bitboard::{file_mask, king_attacks, pawn_attacks};

        let board = self.board();
        let us = self.side_to_move();
        let them = us.flip();
        let king = find_king(board, them);

        debug_assert!(
            pawn_attacks(us, to).test(king),
            "legal_drop precondition: the dropped pawn on `to` must check the enemy king",
        );

        // Undefended, so the king simply captures the pawn.
        if !is_attacked_by(board, to, us) {
            return true;
        }

        let b = attackers_to_pawn(board, them, to);
        let pinned = self.check_info().blockers(them);

        // A capture up the king's own file never leaves the pin ray, so an
        // attacker sharing `to`'s file counts as unpinned.
        if !(b & (!pinned | file_mask(to.file()))).is_empty() {
            return true;
        }

        // The dropped pawn joins the occupancy, blocking rays that would
        // otherwise cover the king's escape ring.
        let occ = board.occupied() ^ crate::bitboard::Bitboard::from_square(to);
        let mut escape = king_attacks(them, king) & !board.pieces_color(them);
        escape ^= crate::bitboard::Bitboard::from_square(to);
        for king_to in escape.squares() {
            if attackers_bb_occ(board, king_to, us, occ).is_empty() {
                return true;
            }
        }

        false
    }
}

/// `c`'s pieces attacking the pawn-drop square `pawn_sq`
/// (`Position::attackers_to_pawn`, `position.cpp`). The uchifuzume test has
/// already handled the enemy king, and a lance can never attack `pawn_sq`
/// because the king stands directly between, so neither is checked here.
fn attackers_to_pawn(board: &Board, c: Color, pawn_sq: Square) -> crate::bitboard::Bitboard {
    use crate::bitboard::{
        bishop_attacks, gold_attacks, knight_attacks, rook_attacks, silver_attacks,
    };
    use crate::board::pat;

    let them = c.flip();
    let occ = board.occupied();

    // A horse or dragon contributes to both the silver and the gold effect.
    let bb_hd = board.pieces_pattern(c, pat::HORSE) | board.pieces_pattern(c, pat::DRAGON);

    let knight = board.pieces_pattern(c, pat::KNIGHT);
    let silver = board.pieces_pattern(c, pat::SILVER);
    let golds = board.pieces_pattern(c, pat::GOLD);
    let bishop_horse = board.pieces_pattern(c, pat::BISHOP) | board.pieces_pattern(c, pat::HORSE);
    let rook_dragon = board.pieces_pattern(c, pat::ROOK) | board.pieces_pattern(c, pat::DRAGON);

    (knight_attacks(them, pawn_sq) & knight)
        | (silver_attacks(them, pawn_sq) & (silver | bb_hd))
        | (gold_attacks(them, pawn_sq) & (golds | bb_hd))
        | (bishop_attacks(pawn_sq, occ) & bishop_horse)
        | (rook_attacks(pawn_sq, occ) & rook_dragon)
}

/// True iff `m` is a pawn drop by `mover` and the position it reaches, already
/// applied to `post`, is checkmate against the opponent. The apply-and-probe
/// form of [`drop_is_uchifuzume`]; `probe_buf` is reused across calls.
#[cfg(test)]
pub(crate) fn is_uchifuzume_after_drop(
    post: &Position,
    m: Move,
    mover: Color,
    probe_buf: &mut Vec<Move>,
) -> bool {
    use crate::search_movegen::ExtMove;
    if !m.is_drop() || m.dropped_piece_kind() != PieceKind::Pawn {
        return false;
    }
    let opp_king_sq = find_king(post.board(), mover.flip());
    if !is_attacked_by(post.board(), opp_king_sq, mover) {
        return false;
    }
    probe_buf.clear();
    let mut pseudo: Vec<ExtMove> = Vec::new();
    post.generate_evasions(true, &mut pseudo);
    for em in pseudo {
        if post.is_legal(em.mv) {
            probe_buf.push(em.mv);
        }
    }
    probe_buf.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sfen::parse_sfen;

    fn legal_count(sfen: &str) -> usize {
        let pos = parse_sfen(sfen).unwrap();
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        moves.len()
    }

    fn legal_moves(sfen: &str) -> Vec<Move> {
        let pos = parse_sfen(sfen).unwrap();
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        moves
    }

    #[test]
    fn startpos_legal_move_count_is_30() {
        assert_eq!(legal_count(crate::sfen::STARTPOS_SFEN), 30);
    }

    #[test]
    fn startpos_contains_7g7f_pawn_push() {
        let pos = Position::startpos();
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        let from = Square::new(6, 6).unwrap();
        let to = Square::new(6, 5).unwrap();
        let pawn = Piece::new(PieceKind::Pawn, Color::Black);
        let expected = Move::make(from, to, pawn);
        assert!(
            moves.contains(&expected),
            "7g7f not in legal-move list: {moves:?}"
        );
    }

    #[test]
    fn pawn_one_rank_from_last_generates_only_promote() {
        // A Black pawn at (4,1) pushing to the last rank.
        let sfen = "4k4/4P4/9/9/9/9/9/9/4K4 b - 1";
        let moves = legal_moves(sfen);
        let pawn_pushes: Vec<&Move> = moves
            .iter()
            .filter(|m| !m.is_drop() && m.from_sq() == Square::new(4, 1).unwrap())
            .collect();
        assert_eq!(
            pawn_pushes.len(),
            1,
            "expected exactly one pawn move (forced promote), got {pawn_pushes:?}"
        );
        assert!(
            pawn_pushes[0].is_promote(),
            "expected forced-promote variant"
        );
    }

    #[test]
    fn pawn_into_zone_but_not_last_rank_generates_both_variants() {
        // A Black pawn pushing to rank 2 — in the zone, not the last rank.
        let sfen = "4k4/9/9/4P4/9/9/9/9/4K4 b - 1";
        let moves = legal_moves(sfen);
        let from = Square::new(4, 3).unwrap();
        let to = Square::new(4, 2).unwrap();
        let pawn = Piece::new(PieceKind::Pawn, Color::Black);
        assert!(moves.contains(&Move::make(from, to, pawn)));
        assert!(moves.contains(&Move::make_promote(from, to, pawn)));
    }

    #[test]
    fn pinned_silver_cannot_move_off_pin_ray() {
        // A Black silver pinned to its king down the 5-file by a White rook.
        let sfen = "4k4/4S4/9/9/9/9/9/9/4r2K1 b - 1";
        let moves = legal_moves(sfen);
        let silver_from = Square::new(4, 1).unwrap();
        for m in &moves {
            if m.is_drop() {
                continue;
            }
            if m.from_sq() == silver_from {
                let to = m.to_sq();
                assert_eq!(to.file(), 4, "pinned silver moved off the 5-file: {to:?}",);
            }
        }
    }

    #[test]
    fn king_cannot_step_into_attacked_square() {
        // A White rook checks the Black king from the adjacent square, so the
        // king's step away along the rook's rank is still attacked once it
        // vacates, while capturing the rook is legal.
        let sfen = "9/9/9/9/3rK4/9/9/9/4k4 b - 1";
        let pos = parse_sfen(sfen).unwrap();
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        let king_from = Square::new(4, 4).unwrap();
        let king = Piece::new(PieceKind::King, Color::Black);
        let illegal_into_ray = Move::make(king_from, Square::new(3, 4).unwrap(), king);
        let legal_capture = Move::make(king_from, Square::new(5, 4).unwrap(), king);
        assert!(
            !moves.contains(&illegal_into_ray),
            "king moved into rook ray: legal moves were {moves:?}",
        );
        assert!(
            moves.contains(&legal_capture),
            "king should be able to capture the checking rook: {moves:?}",
        );
    }

    #[test]
    fn is_attacked_by_pawn_in_front() {
        // Black pawn at (4,4); attacks (4,3).
        let mut board = Board::empty();
        board.set(
            Square::new(4, 4).unwrap(),
            Some(Piece::new(PieceKind::Pawn, Color::Black)),
        );
        assert!(is_attacked_by(
            &board,
            Square::new(4, 3).unwrap(),
            Color::Black
        ));
        assert!(!is_attacked_by(
            &board,
            Square::new(4, 5).unwrap(),
            Color::Black
        ));
        assert!(!is_attacked_by(
            &board,
            Square::new(3, 3).unwrap(),
            Color::Black
        ));
    }

    #[test]
    fn is_attacked_by_lance_through_empty_squares() {
        // Black lance at (4,8); attacks (4,0) through empty path.
        let mut board = Board::empty();
        board.set(
            Square::new(4, 8).unwrap(),
            Some(Piece::new(PieceKind::Lance, Color::Black)),
        );
        assert!(is_attacked_by(
            &board,
            Square::new(4, 0).unwrap(),
            Color::Black
        ));
    }

    #[test]
    fn is_attacked_by_lance_blocked_returns_false() {
        // Black lance at (4,8) with a blocker at (4,4).
        let mut board = Board::empty();
        board.set(
            Square::new(4, 8).unwrap(),
            Some(Piece::new(PieceKind::Lance, Color::Black)),
        );
        board.set(
            Square::new(4, 4).unwrap(),
            Some(Piece::new(PieceKind::Pawn, Color::White)),
        );
        assert!(is_attacked_by(
            &board,
            Square::new(4, 4).unwrap(),
            Color::Black
        )); // first blocker IS attacked
        assert!(!is_attacked_by(
            &board,
            Square::new(4, 0).unwrap(),
            Color::Black
        ));
    }

    #[test]
    fn is_attacked_by_knight_at_jump_offsets() {
        // Black knight at (4,4) attacks (3,2) and (5,2). Nothing else.
        let mut board = Board::empty();
        board.set(
            Square::new(4, 4).unwrap(),
            Some(Piece::new(PieceKind::Knight, Color::Black)),
        );
        assert!(is_attacked_by(
            &board,
            Square::new(3, 2).unwrap(),
            Color::Black
        ));
        assert!(is_attacked_by(
            &board,
            Square::new(5, 2).unwrap(),
            Color::Black
        ));
        assert!(!is_attacked_by(
            &board,
            Square::new(4, 3).unwrap(),
            Color::Black
        ));
        assert!(!is_attacked_by(
            &board,
            Square::new(3, 6).unwrap(),
            Color::Black
        ));
    }

    #[test]
    fn is_attacked_by_bishop_along_diagonal() {
        let mut board = Board::empty();
        board.set(
            Square::new(0, 0).unwrap(),
            Some(Piece::new(PieceKind::Bishop, Color::Black)),
        );
        assert!(is_attacked_by(
            &board,
            Square::new(8, 8).unwrap(),
            Color::Black
        ));
        // Block it.
        board.set(
            Square::new(4, 4).unwrap(),
            Some(Piece::new(PieceKind::Pawn, Color::White)),
        );
        assert!(is_attacked_by(
            &board,
            Square::new(4, 4).unwrap(),
            Color::Black
        ));
        assert!(!is_attacked_by(
            &board,
            Square::new(8, 8).unwrap(),
            Color::Black
        ));
    }

    #[test]
    fn is_attacked_by_horse_picks_up_orthogonal_step() {
        // A horse's orthogonal step, which a plain bishop does not cover.
        let mut board = Board::empty();
        let horse = Piece {
            kind: PieceKind::Bishop,
            color: Color::Black,
            promoted: true,
        };
        board.set(Square::new(4, 4).unwrap(), Some(horse));
        assert!(is_attacked_by(
            &board,
            Square::new(4, 5).unwrap(),
            Color::Black
        ));
        assert!(is_attacked_by(
            &board,
            Square::new(4, 3).unwrap(),
            Color::Black
        ));
        assert!(is_attacked_by(
            &board,
            Square::new(3, 4).unwrap(),
            Color::Black
        ));
        // Plain bishop does NOT attack (4,5).
        let mut board2 = Board::empty();
        board2.set(
            Square::new(4, 4).unwrap(),
            Some(Piece::new(PieceKind::Bishop, Color::Black)),
        );
        assert!(!is_attacked_by(
            &board2,
            Square::new(4, 5).unwrap(),
            Color::Black
        ));
    }

    #[test]
    fn is_attacked_by_dragon_picks_up_diagonal_step() {
        // A dragon's diagonal step.
        let mut board = Board::empty();
        let dragon = Piece {
            kind: PieceKind::Rook,
            color: Color::Black,
            promoted: true,
        };
        board.set(Square::new(4, 4).unwrap(), Some(dragon));
        assert!(is_attacked_by(
            &board,
            Square::new(5, 5).unwrap(),
            Color::Black
        ));
        // Plain rook does NOT.
        let mut board2 = Board::empty();
        board2.set(
            Square::new(4, 4).unwrap(),
            Some(Piece::new(PieceKind::Rook, Color::Black)),
        );
        assert!(!is_attacked_by(
            &board2,
            Square::new(5, 5).unwrap(),
            Color::Black
        ));
    }

    #[test]
    fn is_attacked_by_promoted_pawn_uses_gold_pattern() {
        // A tokin attacks like a gold: forward and forward-diagonal, but not
        // backward-diagonal.
        let mut board = Board::empty();
        let tokin = Piece {
            kind: PieceKind::Pawn,
            color: Color::Black,
            promoted: true,
        };
        board.set(Square::new(4, 4).unwrap(), Some(tokin));
        assert!(is_attacked_by(
            &board,
            Square::new(4, 3).unwrap(),
            Color::Black
        ));
        assert!(is_attacked_by(
            &board,
            Square::new(3, 3).unwrap(),
            Color::Black
        ));
        assert!(!is_attacked_by(
            &board,
            Square::new(3, 5).unwrap(),
            Color::Black
        ));
    }

    #[test]
    fn is_attacked_by_white_attacker_flips_direction() {
        // White's forward is rank-increasing.
        let mut board = Board::empty();
        board.set(
            Square::new(4, 4).unwrap(),
            Some(Piece::new(PieceKind::Pawn, Color::White)),
        );
        assert!(is_attacked_by(
            &board,
            Square::new(4, 5).unwrap(),
            Color::White
        ));
        assert!(!is_attacked_by(
            &board,
            Square::new(4, 3).unwrap(),
            Color::White
        ));
    }

    #[test]
    fn empty_hand_emits_no_drops() {
        let pos = parse_sfen("4k4/9/9/9/9/9/9/9/4K4 b - 1").unwrap();
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        assert!(moves.iter().all(|m| !m.is_drop()), "no drops expected");
    }

    #[test]
    fn bishop_in_hand_can_drop_on_every_empty_square() {
        // Two kings and a bishop in hand, so 79 empty squares.
        let pos = parse_sfen("4k4/9/9/9/9/9/9/9/4K4 b B 1").unwrap();
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        let drops: Vec<&Move> = moves.iter().filter(|m| m.is_drop()).collect();
        assert_eq!(drops.len(), 79, "expected 79 bishop drops, got {drops:?}");
        assert!(
            drops
                .iter()
                .all(|m| m.dropped_piece_kind() == PieceKind::Bishop),
        );
    }

    #[test]
    fn pawn_drop_is_blocked_by_own_pawn_on_same_file() {
        // A pawn in hand with an own pawn on file 4, so nifu forbids that file
        // and the last-rank rule forbids rank 0.
        let pos = parse_sfen("4k4/9/9/9/4P4/9/9/9/4K4 b P 1").unwrap();
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        let pawn_drops: Vec<&Move> = moves
            .iter()
            .filter(|m| m.is_drop() && m.dropped_piece_kind() == PieceKind::Pawn)
            .collect();
        // 78 empty squares, less 6 on file 4 and 8 on rank 0. The two sets meet
        // only at (4,0), which the white king occupies.
        assert_eq!(pawn_drops.len(), 64, "got {pawn_drops:?}");
        assert!(
            pawn_drops.iter().all(|m| m.to_sq().file() != 4),
            "nifu violated"
        );
        assert!(
            pawn_drops.iter().all(|m| m.to_sq().rank() != 0),
            "pawn dropped on Black's last rank",
        );
    }

    #[test]
    fn nifu_does_not_trigger_on_tokin() {
        // A pawn in hand and a Black tokin on file 4. A tokin is not an
        // unpromoted pawn, so nifu must not fire on that file.
        let pos = parse_sfen("4k4/9/9/9/4+P4/9/9/9/4K4 b P 1").unwrap();
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        let pawn_drops_on_file_4: Vec<&Move> = moves
            .iter()
            .filter(|m| {
                m.is_drop() && m.dropped_piece_kind() == PieceKind::Pawn && m.to_sq().file() == 4
            })
            .collect();
        // Six empty squares on file 4, none of them on Black's rank 0.
        assert_eq!(
            pawn_drops_on_file_4.len(),
            6,
            "expected 6 pawn drops on file 4 (Tokin does not trigger nifu), got {pawn_drops_on_file_4:?}",
        );
    }

    #[test]
    fn nifu_on_file_zero_does_not_leak_to_file_one() {
        // An own pawn on file 0: file 0 must be filtered and file 1 must not.
        let pos = parse_sfen("4k4/9/9/9/8P/9/9/9/4K4 b P 1").unwrap();
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        let pawn_drops: Vec<&Move> = moves
            .iter()
            .filter(|m| m.is_drop() && m.dropped_piece_kind() == PieceKind::Pawn)
            .collect();
        assert!(
            pawn_drops.iter().all(|m| m.to_sq().file() != 0),
            "nifu failed to filter file 0: {pawn_drops:?}",
        );
        assert!(
            pawn_drops.iter().any(|m| m.to_sq().file() == 1),
            "file 1 was filtered too — off-by-one in nifu file index: {pawn_drops:?}",
        );
    }

    #[test]
    fn nifu_applies_to_white_pawns_too() {
        // White to move with an own pawn on file 4. White's last rank is 8, so
        // the rank exclusion does not collide with the nifu test.
        let pos = parse_sfen("4k4/9/9/9/4p4/9/9/9/4K4 w p 1").unwrap();
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        let pawn_drops: Vec<&Move> = moves
            .iter()
            .filter(|m| m.is_drop() && m.dropped_piece_kind() == PieceKind::Pawn)
            .collect();
        assert!(
            !pawn_drops.is_empty(),
            "white had no legal pawn drops at all — setup wrong",
        );
        assert!(
            pawn_drops.iter().all(|m| m.to_sq().file() != 4),
            "nifu (white side) failed to filter file 4: {pawn_drops:?}",
        );
    }

    #[test]
    fn nifu_does_not_filter_lance_drops() {
        // The pawn-on-file-4 nifu shape, but with a lance in hand instead. A
        // lance is not subject to nifu, only to the last-rank rule.
        let pos = parse_sfen("4k4/9/9/9/4P4/9/9/9/4K4 b L 1").unwrap();
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        let lance_drops_on_file_4: Vec<&Move> = moves
            .iter()
            .filter(|m| {
                m.is_drop() && m.dropped_piece_kind() == PieceKind::Lance && m.to_sq().file() == 4
            })
            .collect();
        // Six empty squares on file 4, none of them on Black's rank 0.
        assert_eq!(
            lance_drops_on_file_4.len(),
            6,
            "lance drop on shared-file with own pawn was wrongly filtered: {lance_drops_on_file_4:?}",
        );
    }

    #[test]
    fn knight_drop_excluded_from_last_two_ranks() {
        // 79 empty squares, less the 17 empty ones on ranks 0 and 1, where a
        // Black knight would be stuck.
        let pos = parse_sfen("4k4/9/9/9/9/9/9/9/4K4 b N 1").unwrap();
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        let knight_drops: Vec<&Move> = moves
            .iter()
            .filter(|m| m.is_drop() && m.dropped_piece_kind() == PieceKind::Knight)
            .collect();
        assert_eq!(knight_drops.len(), 62);
        assert!(
            knight_drops.iter().all(|m| m.to_sq().rank() >= 2),
            "knight dropped on rank 0 or 1"
        );
    }

    #[test]
    fn drop_blocking_check_is_legal_capture_is_not_required() {
        // A White rook checks the Black king down the open 5-file, with a gold
        // in hand to interpose.
        let pos = parse_sfen("4r4/9/9/9/9/9/9/9/4K4 b G 1").unwrap();
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        let gold_drops: Vec<&Move> = moves
            .iter()
            .filter(|m| m.is_drop() && m.dropped_piece_kind() == PieceKind::Gold)
            .collect();
        for m in &gold_drops {
            let to = m.to_sq();
            assert_eq!(to.file(), 4, "non-blocking gold drop survived: {to:?}");
            assert!((1..=7).contains(&to.rank()), "gold drop off interpose ray");
        }
        assert_eq!(gold_drops.len(), 7);
    }

    // The mating net the next four tests vary: a White king cornered at 9a,
    // with a Black gold and knight covering its escapes and defending (8,1),
    // the square a Black pawn drop would check from.

    #[test]
    fn uchifuzume_filters_pawn_drop_mate() {
        let pos = parse_sfen("k8/9/G1N6/9/9/9/9/9/8K b P 1").unwrap();
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        let mating_drop =
            Move::make_drop(PieceKind::Pawn, Color::Black, Square::new(8, 1).unwrap());
        assert!(
            !moves.contains(&mating_drop),
            "uchifuzume failed to filter pawn-drop mate: {moves:?}",
        );
    }

    #[test]
    fn pawn_drop_check_is_legal_when_attacker_can_be_captured() {
        // The mating shape plus a White silver that can capture the pawn.
        let pos = parse_sfen("k8/9/GsN6/9/9/9/9/9/8K b P 1").unwrap();
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        let checking_drop =
            Move::make_drop(PieceKind::Pawn, Color::Black, Square::new(8, 1).unwrap());
        assert!(
            moves.contains(&checking_drop),
            "pawn drop wrongly filtered when capture is available: {moves:?}",
        );
    }

    #[test]
    fn gold_drop_mate_is_legal_uchifuzume_is_pawn_only() {
        // The mating shape, but with a gold dropped instead of a pawn.
        let pos = parse_sfen("k8/9/G1N6/9/9/9/9/9/8K b G 1").unwrap();
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        let mating_gold_drop =
            Move::make_drop(PieceKind::Gold, Color::Black, Square::new(8, 1).unwrap());
        assert!(
            moves.contains(&mating_gold_drop),
            "gold-drop mate was filtered (uchifuzume must be pawn-only): {moves:?}",
        );
    }

    #[test]
    fn pawn_drop_check_is_legal_when_king_can_step_out() {
        // Two kings only, so the checking pawn drop leaves escape squares.
        let pos = parse_sfen("k8/9/9/9/9/9/9/9/8K b P 1").unwrap();
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        let checking_drop =
            Move::make_drop(PieceKind::Pawn, Color::Black, Square::new(8, 1).unwrap());
        assert!(
            moves.contains(&checking_drop),
            "pawn drop wrongly filtered when king has an escape: {moves:?}",
        );
    }

    #[test]
    fn nifu_and_uchifuzume_compose_without_panic() {
        // The mating shape plus an own Black pawn on the same file, so nifu
        // rejects the drop before the uchifuzume probe can see it.
        let pos = parse_sfen("k8/9/G1N6/9/P8/9/9/9/8K b P 1").unwrap();
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        let pawn_drops_on_file_8: Vec<&Move> = moves
            .iter()
            .filter(|m| {
                m.is_drop() && m.dropped_piece_kind() == PieceKind::Pawn && m.to_sq().file() == 8
            })
            .collect();
        assert!(
            pawn_drops_on_file_8.is_empty(),
            "nifu should reject every pawn drop on file 8: {pawn_drops_on_file_8:?}",
        );
    }

    /// A deterministic xorshift PRNG, so a failing playout replays.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn pick(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    /// Assert [`drop_is_uchifuzume`] and [`is_uchifuzume_after_drop`] agree on
    /// the one pawn-drop-mate candidate at `p`, and return how many candidates
    /// were compared — 0 or 1 — so callers can gauge coverage. The oracle plays
    /// the drop, so this needs a pawn in the mover's hand.
    fn assert_drop_uchifuzume_agrees(p: &Position, ctx: &str) -> usize {
        let us = p.side_to_move();
        if p.hand(us).count(PieceKind::Pawn) == 0 {
            return 0;
        }
        let Some(king) = try_find_king(p.board(), us.flip()) else {
            return 0;
        };
        let mut compared = 0;
        for idx in 0..Square::COUNT as u8 {
            let to = Square::from_index(idx).unwrap();
            if p.board().get(to).is_some() {
                continue;
            }
            // Only the square from which our pawn would check the enemy king
            // can ever be uchifuzume.
            if !crate::bitboard::pawn_attacks(us, to).test(king) {
                continue;
            }
            let m = Move::make_drop(PieceKind::Pawn, us, to);
            let new = drop_is_uchifuzume(p, m);
            let mut post = p.clone();
            post.do_move(m);
            let mut buf = Vec::new();
            let oracle = is_uchifuzume_after_drop(&post, m, us, &mut buf);
            assert_eq!(
                new, oracle,
                "{ctx}: drop_is_uchifuzume={new} disagrees with clone-probe oracle={oracle} \
                 for pawn drop to {to:?}",
            );
            compared += 1;
        }
        compared
    }

    /// Drop-mate-rich seeds: pawn in hand, exposed kings, and (for the pinned
    /// shapes) defenders around the enemy king, plus a few parity-fixture SFENs.
    const UCHI_ORACLE_SFENS: &[&str] = &[
        // Parity fixtures with hands / drops in play.
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        "4k4/9/4r4/9/9/9/4K3B/9/9 b RG2gs2n3p 1",
        "k8/1P7/G8/1N2P4/9/9/9/9/8K b 2PG2pg 1",
        // Drop-mate seeds.
        "k8/9/G1N6/9/9/9/9/9/8K b P 1",  // uchifuzume mate
        "k8/9/GsN6/9/9/9/9/9/8K b P 1",  // capturable attacker → legal
        "k8/9/9/9/9/9/9/9/8K b P 1",     // king can step out → legal
        "k8/9/G1N6/9/P8/9/9/9/8K b P 1", // nifu + mate shape
    ];

    #[cfg_attr(miri, ignore)]
    #[test]
    fn legal_drop_matches_clone_probe_oracle_over_playouts() {
        const MIN_PLIES: usize = 40;
        let mut total_compared = 0usize;
        for (fi, sfen) in UCHI_ORACLE_SFENS.iter().enumerate() {
            let mut p = parse_sfen(sfen).expect("valid SFEN");
            let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ (fi as u64).wrapping_add(1));
            let mut stack: Vec<(Move, crate::position::Undo)> = Vec::new();
            let mut plies = 0usize;
            while plies < MIN_PLIES {
                total_compared +=
                    assert_drop_uchifuzume_agrees(&p, &format!("fixture {fi} ply {plies}"));
                let legal = {
                    let mut v = Vec::new();
                    p.generate_legal_all(&mut v);
                    v
                };
                if legal.is_empty() {
                    // Terminal — unwind and continue from the root.
                    while let Some((m, u)) = stack.pop() {
                        p.undo_move(m, u);
                    }
                    if stack.is_empty() {
                        break;
                    }
                    continue;
                }
                let m = legal[rng.pick(legal.len())];
                let u = p.do_move(m);
                stack.push((m, u));
                plies += 1;
            }
        }
        assert!(
            total_compared > 0,
            "no pawn-drop-mate candidate was ever probed — coverage is vacuous",
        );
    }

    /// The three pin shapes the reference works through, including the
    /// same-file exception (例3) both firing and not. `to` is always the square
    /// in front of a White king cornered at `1a`.
    #[test]
    fn legal_drop_pin_example_shapes() {
        let to = Square::new(0, 1).unwrap();

        // 例3 firing: the sole attacker is a file-pinned White rook that can
        // recapture along its pin line. Without the same-file exception this
        // would be misjudged uchifuzume.
        let p = parse_sfen("8k/6G2/7Gr/8L/9/9/9/9/K8 b P 1").unwrap();
        assert!(
            p.legal_drop(to),
            "例3 firing: same-file pinned rook can recapture — must be legal",
        );
        assert_eq!(assert_drop_uchifuzume_agrees(&p, "例3 firing"), 1);

        // 例1: a rank-pinned White bishop attacks `to` diagonally but cannot
        // recapture without exposing its king, and is not on `to`'s file.
        let p = parse_sfen("5R1bk/9/7G1/9/9/9/9/9/K8 b P 1").unwrap();
        assert!(
            !p.legal_drop(to),
            "例1: rank-pinned bishop cannot recapture — must be uchifuzume",
        );
        assert_eq!(assert_drop_uchifuzume_agrees(&p, "例1"), 1);

        // 例2: a diagonally-pinned White rook attacks `to` along the rank but
        // cannot recapture. The same-file exception must not fire here.
        let p = parse_sfen("8k/6Gr1/6BG1/9/9/9/9/9/K8 b P 1").unwrap();
        assert!(
            !p.legal_drop(to),
            "例2: diagonally-pinned rook cannot recapture — must be uchifuzume",
        );
        assert_eq!(assert_drop_uchifuzume_agrees(&p, "例2"), 1);
    }

    // The reference's perft and its `LEGAL_ALL` movegen are both
    // repetition-blind, so plain 4-fold is not filtered here either. The tests
    // below pin that, so a regression toward filter-the-move is caught.

    /// Two-king board with no other pieces.
    fn setup_king_shuffle_pos() -> Position {
        let mut p = Position::empty();
        p.board_mut().set(
            Square::new(0, 8).unwrap(),
            Some(Piece::new(PieceKind::King, Color::Black)),
        );
        p.board_mut().set(
            Square::new(0, 0).unwrap(),
            Some(Piece::new(PieceKind::King, Color::White)),
        );
        p
    }

    /// Four moves that, applied in order from `setup_king_shuffle_pos`,
    /// return the position to its starting state.
    fn shuffle_cycle() -> [Move; 4] {
        let bk = Piece::new(PieceKind::King, Color::Black);
        let wk = Piece::new(PieceKind::King, Color::White);
        [
            Move::make(Square::new(0, 8).unwrap(), Square::new(1, 8).unwrap(), bk),
            Move::make(Square::new(0, 0).unwrap(), Square::new(1, 0).unwrap(), wk),
            Move::make(Square::new(1, 8).unwrap(), Square::new(0, 8).unwrap(), bk),
            Move::make(Square::new(1, 0).unwrap(), Square::new(0, 0).unwrap(), wk),
        ]
    }

    #[test]
    fn move_completing_three_fold_is_allowed() {
        // Two full cycles, so repeating the side-step reaches a threefold, not
        // a fourfold.
        let mut pos = setup_king_shuffle_pos();
        let cycle = shuffle_cycle();
        for _ in 0..2 {
            for m in cycle {
                pos.do_move(m);
            }
        }
        assert_eq!(
            pos.position_occurrences(),
            2,
            "two cycles should leave the start state matched twice in history",
        );
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        assert!(
            moves.contains(&cycle[0]),
            "BK side-step should be legal at the post-2-cycle state \
             (its post-state would be the 3rd occurrence — below the 4-fold threshold)",
        );
    }

    #[test]
    fn move_completing_four_fold_is_allowed() {
        // Three full cycles, so repeating the side-step reaches a fourfold.
        let mut pos = setup_king_shuffle_pos();
        let cycle = shuffle_cycle();
        for _ in 0..3 {
            for m in cycle {
                pos.do_move(m);
            }
        }
        assert_eq!(
            pos.position_occurrences(),
            3,
            "three cycles should leave the start state matched 3× in history",
        );
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        assert!(
            moves.contains(&cycle[0]),
            "BK side-step should remain legal even when its post-state would be the \
             4th occurrence — no-perft-change semantics, mirrors upstream's \
             repetition-blind LEGAL_ALL/perft",
        );
    }

    #[test]
    fn fresh_history_no_false_positive() {
        // An empty history, where the pushed entry counts as one occurrence of
        // itself.
        let pos = setup_king_shuffle_pos();
        let cycle = shuffle_cycle();
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        assert!(
            moves.contains(&cycle[0]),
            "BK side-step should be legal from the fresh start state (empty history)",
        );

        let mut after = pos.clone();
        after.do_move(cycle[0]);
        assert_eq!(
            after.position_occurrences(),
            1,
            "after one move from empty history, the just-pushed entry matches itself once",
        );
    }

    // A move completing a perpetual-check 連続王手の千日手 fourfold is still
    // generated: the consequence is scored by the search and, in real games,
    // adjudicated by the server.

    /// A White king in check from a Black rook down file 4, with the Black king
    /// far away. Black is the perpetual checker.
    fn setup_perpetual_check_pos() -> Position {
        let mut p = Position::empty();
        p.set_side_to_move(Color::White);
        p.board_mut().set(
            Square::new(4, 0).unwrap(),
            Some(Piece::new(PieceKind::King, Color::White)),
        );
        p.board_mut().set(
            Square::new(4, 1).unwrap(),
            Some(Piece::new(PieceKind::Rook, Color::Black)),
        );
        p.board_mut().set(
            Square::new(0, 8).unwrap(),
            Some(Piece::new(PieceKind::King, Color::Black)),
        );
        p
    }

    /// The four-move cycle back to `setup_perpetual_check_pos`, checking on
    /// every Black move.
    fn perpetual_check_cycle() -> [Move; 4] {
        let wk = Piece::new(PieceKind::King, Color::White);
        let br = Piece::new(PieceKind::Rook, Color::Black);
        [
            Move::make(Square::new(4, 0).unwrap(), Square::new(3, 0).unwrap(), wk),
            Move::make(Square::new(4, 1).unwrap(), Square::new(3, 1).unwrap(), br),
            Move::make(Square::new(3, 0).unwrap(), Square::new(4, 0).unwrap(), wk),
            Move::make(Square::new(3, 1).unwrap(), Square::new(4, 1).unwrap(), br),
        ]
    }

    #[test]
    fn perpetual_check_completing_move_is_generated() {
        // Three full cycles plus three moves, so `cycle[3]` would reach the
        // start state a fourth time, having checked on every Black move.
        let mut pos = setup_perpetual_check_pos();
        let cycle = perpetual_check_cycle();
        for _ in 0..3 {
            for m in cycle {
                pos.do_move(m);
            }
        }
        pos.do_move(cycle[0]);
        pos.do_move(cycle[1]);
        pos.do_move(cycle[2]);
        assert_eq!(
            pos.side_to_move(),
            Color::Black,
            "after 3 cycles + m1+m2+m3 of cycle 4, Black is to move",
        );
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        assert!(
            moves.contains(&cycle[3]),
            "BR 4b→5b is generated (reference is repetition-blind): {moves:?}",
        );
    }

    /// The perpetual-check layout, paired below with a cycle that detours the
    /// rook off-file so only one of Black's two moves checks.
    fn setup_mixed_check_pos() -> Position {
        setup_perpetual_check_pos()
    }

    /// The four-move cycle back to `setup_mixed_check_pos`, in which only the
    /// second of Black's two moves gives check.
    fn mixed_check_cycle() -> [Move; 4] {
        let wk = Piece::new(PieceKind::King, Color::White);
        let br = Piece::new(PieceKind::Rook, Color::Black);
        [
            // The White king escapes the start-state check.
            Move::make(Square::new(4, 0).unwrap(), Square::new(3, 0).unwrap(), wk),
            // The rook detours off file 4, giving no check.
            Move::make(Square::new(4, 1).unwrap(), Square::new(8, 1).unwrap(), br),
            // The king returns, safe while the rook is off file 4.
            Move::make(Square::new(3, 0).unwrap(), Square::new(4, 0).unwrap(), wk),
            // The rook returns to the king's file, giving check.
            Move::make(Square::new(8, 1).unwrap(), Square::new(4, 1).unwrap(), br),
        ]
    }

    #[test]
    fn four_fold_with_check_on_most_but_not_all_cycle_steps_is_generated() {
        // The same fourfold-completion shape, but on the mixed-check cycle.
        let mut pos = setup_mixed_check_pos();
        let cycle = mixed_check_cycle();
        for _ in 0..3 {
            for m in cycle {
                pos.do_move(m);
            }
        }
        pos.do_move(cycle[0]);
        pos.do_move(cycle[1]);
        pos.do_move(cycle[2]);
        assert_eq!(pos.side_to_move(), Color::Black);

        let mut scratch = pos.clone();
        scratch.do_move(cycle[3]);
        assert_eq!(
            scratch.position_occurrences(),
            4,
            "candidate cycle[3] should complete the 4-fold: history = {:?}",
            scratch,
        );

        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        assert!(
            moves.contains(&cycle[3]),
            "BR 9b→5b is generated (reference is repetition-blind): {moves:?}",
        );
    }

    /// `setup_perpetual_check_pos` with the colours flipped, so White is the
    /// perpetual checker and Black is the side to move.
    fn setup_white_perpetual_check_pos() -> Position {
        let mut p = Position::empty();
        // Explicit, though `Position::empty()` already defaults to Black.
        p.set_side_to_move(Color::Black);
        p.board_mut().set(
            Square::new(4, 8).unwrap(),
            Some(Piece::new(PieceKind::King, Color::Black)),
        );
        p.board_mut().set(
            Square::new(4, 7).unwrap(),
            Some(Piece::new(PieceKind::Rook, Color::White)),
        );
        p.board_mut().set(
            Square::new(0, 0).unwrap(),
            Some(Piece::new(PieceKind::King, Color::White)),
        );
        p
    }

    /// [`perpetual_check_cycle`] with the colours flipped.
    fn white_perpetual_check_cycle() -> [Move; 4] {
        let bk = Piece::new(PieceKind::King, Color::Black);
        let wr = Piece::new(PieceKind::Rook, Color::White);
        [
            Move::make(Square::new(4, 8).unwrap(), Square::new(3, 8).unwrap(), bk),
            Move::make(Square::new(4, 7).unwrap(), Square::new(3, 7).unwrap(), wr),
            Move::make(Square::new(3, 8).unwrap(), Square::new(4, 8).unwrap(), bk),
            Move::make(Square::new(3, 7).unwrap(), Square::new(4, 7).unwrap(), wr),
        ]
    }

    #[test]
    fn four_fold_where_opponent_was_the_checker_is_legal_for_mover() {
        // Three full cycles, so `cycle[0]` completes a fourfold. Here the
        // candidate belongs to the *chased* side, which the perpetual-check
        // rule never rejects — only the checker's moves.
        let mut pos = setup_white_perpetual_check_pos();
        let cycle = white_perpetual_check_cycle();
        for _ in 0..3 {
            for m in cycle {
                pos.do_move(m);
            }
        }
        assert_eq!(pos.side_to_move(), Color::Black);
        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        assert!(
            moves.contains(&cycle[0]),
            "BK escape should remain legal — White is the perpetual checker, not Black: {moves:?}",
        );
    }

    #[test]
    fn non_repeating_check_move_is_unaffected() {
        // A checking rook move on an empty history, which is no fourfold
        // completion. Pins against a "gives check ⇒ filtered" reading.
        let mut p = Position::empty();
        p.board_mut().set(
            Square::new(8, 8).unwrap(),
            Some(Piece::new(PieceKind::King, Color::Black)),
        );
        p.board_mut().set(
            Square::new(4, 0).unwrap(),
            Some(Piece::new(PieceKind::King, Color::White)),
        );
        p.board_mut().set(
            Square::new(0, 8).unwrap(),
            Some(Piece::new(PieceKind::Rook, Color::Black)),
        );
        let mut moves = Vec::new();
        p.generate_legal_all(&mut moves);
        let rook = Piece::new(PieceKind::Rook, Color::Black);
        let checking_move =
            Move::make(Square::new(0, 8).unwrap(), Square::new(4, 8).unwrap(), rook);
        assert!(
            moves.contains(&checking_move),
            "rook check (no prior history) should remain legal: {moves:?}",
        );
    }

    #[test]
    fn four_fold_without_any_check_is_generated() {
        // The king-shuffle cycle checks nowhere. A plain fourfold makes the
        // game drawn rather than the move illegal.
        let mut pos = setup_king_shuffle_pos();
        let cycle = shuffle_cycle();
        for _ in 0..3 {
            for m in cycle {
                pos.do_move(m);
            }
        }
        let mut scratch = pos.clone();
        scratch.do_move(cycle[0]);
        assert_eq!(scratch.position_occurrences(), 4);

        let mut moves = Vec::new();
        pos.generate_legal_all(&mut moves);
        assert!(
            moves.contains(&cycle[0]),
            "BK side-step is generated — plain 4-fold is drawn, not illegal: {moves:?}",
        );
    }
}

// `try_find_king` against its 81-square scan form, including on a king-less
// board.
#[cfg(test)]
mod king_lookup_equivalence {
    use super::{try_find_king, try_find_king_scan};
    use crate::color::Color;
    use crate::move_::Move;
    use crate::piece::{Piece, PieceKind};
    use crate::sfen::parse_sfen;
    use crate::square::Square;

    const SFENS: &[&str] = &[
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        "4k4/9/4r4/9/9/9/4K3B/9/9 b RG2gs2n3p 1",
        "k8/1P7/G8/1N2P4/9/9/9/9/8K b 2PG2pg 1",
        "l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1",
        "4k4/3P3+PL/2N2PR2/1L2BNS2/4N4/9/9/9/4K4 b - 1",
        "9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1",
    ];

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn pick(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn piece_set_matches_scan_over_playouts() {
        const MIN_PLIES: usize = 60;
        for (fi, sfen) in SFENS.iter().enumerate() {
            let mut p = parse_sfen(sfen).expect("valid SFEN");
            let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ (fi as u64).wrapping_add(1));
            let mut plies = 0usize;
            loop {
                for c in [Color::Black, Color::White] {
                    assert_eq!(
                        try_find_king(p.board(), c),
                        try_find_king_scan(p.board(), c),
                        "fixture {fi} ply {plies}: {c:?} king lookup diverges",
                    );
                }
                if plies >= MIN_PLIES {
                    break;
                }
                let mut legal = Vec::new();
                p.generate_legal_all(&mut legal);
                if legal.is_empty() {
                    break;
                }
                let m: Move = legal[rng.pick(legal.len())];
                p.do_move(m);
                plies += 1;
            }
        }
    }

    #[test]
    fn king_less_board_returns_none() {
        let p = parse_sfen("4k4/9/9/9/9/9/9/9/4K4 b - 1").expect("valid SFEN");
        let mut board = *p.board();
        let bk = try_find_king(&board, Color::Black).expect("black king present");
        let wk = try_find_king(&board, Color::White).expect("white king present");

        board.set(bk, None);
        assert_eq!(try_find_king(&board, Color::Black), None);
        assert_eq!(try_find_king_scan(&board, Color::Black), None);
        assert_eq!(try_find_king(&board, Color::White), Some(wk));
        assert_eq!(try_find_king_scan(&board, Color::White), Some(wk));

        board.set(wk, None);
        assert_eq!(try_find_king(&board, Color::Black), None);
        assert_eq!(try_find_king(&board, Color::White), None);

        let sq = Square::new(3, 5).unwrap();
        board.set(sq, Some(Piece::new(PieceKind::King, Color::Black)));
        assert_eq!(try_find_king(&board, Color::Black), Some(sq));
        assert_eq!(try_find_king_scan(&board, Color::Black), Some(sq));
    }
}
