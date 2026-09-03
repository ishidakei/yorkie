//! Zobrist position hashing for [`crate::position::Position`], mirroring the
//! reference's `board_key` / `hand_key` split (`position.h`).
//!
//! `board_key` accumulates by XOR, which is its own inverse, so one operation
//! both places and removes a term. `hand_key` accumulates by **addition** of a
//! per-`(color, kind)` step, so `n` held copies contribute `n` steps and a hand
//! change is a single add or subtract.
//!
//! **The 64-bit constants must equal the reference's**, not merely share its
//! structure. The pawn and correction histories are hash tables indexed by these
//! keys masked to a few low bits, so a different table aliases differently and
//! diverges the search at the first collision that flips a quiet's ordering.
//! They are therefore reproduced from `Position::init` (`position.cpp`): the
//! same `xorshift64*` PRNG and seed, the same four-draws-keep-the-first
//! `set_rand`, and the same draw order.

use crate::color::Color;
use crate::piece::{Piece, PieceKind};
use crate::square::Square;

/// Distinct piece codes: `promoted × color × kind`. A promoted king or gold is
/// never realised, but reserving a slot keeps [`piece_code`] branch-free.
const PIECE_CODES: usize = 2 * Color::COUNT * PieceKind::COUNT;

/// The reference Zobrist PRNG seed (`position.cpp`).
const REF_SEED: u64 = 20151225;

/// The `xorshift64*` output multiplier (`misc.h`).
const REF_MULT: u64 = 2685821657736338717;

/// One `PRNG::rand64()` step (`misc.h`), returning `(next_state, value)`. The
/// next draw consumes `next_state`, **not** `value`.
const fn rand64(s: u64) -> (u64, u64) {
    let mut x = s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    (x, x.wrapping_mul(REF_MULT))
}

/// The reference `set_rand` (`position.cpp`): draw four words and keep the
/// first. The other three are discarded but still advance the stream.
const fn set_rand(s: u64) -> (u64, u64) {
    let (s, v) = rand64(s);
    let (s, _) = rand64(s);
    let (s, _) = rand64(s);
    let (s, _) = rand64(s);
    (s, v)
}

/// Map a reference `Piece` code (`types.h`) to this port's [`piece_code`] slot,
/// or `None` for a code never realised on a board. A `None` code is still
/// *drawn*, to keep the PRNG stream aligned, but not stored.
///
/// The reference's `PieceType` order puts `BISHOP` before `GOLD` where
/// [`PieceKind`] does the reverse, so the kind is remapped explicitly.
const fn ref_code_to_slot(pc: usize) -> Option<usize> {
    if pc == 16 {
        return None;
    }
    let (color, local) = if pc <= 15 { (0, pc) } else { (1, pc - 16) };
    if local == 15 {
        return None; // B_GOLDS / W_GOLDS.
    }
    let (promo, ref_pt) = if local <= 8 {
        (0, local)
    } else {
        (1, local - 8)
    };
    let kind = match ref_kind_to_index(ref_pt) {
        Some(k) => k,
        None => return None,
    };
    Some((promo * Color::COUNT + color) * PieceKind::COUNT + kind)
}

/// Reference `PieceType` (`1..=8`) → this port's [`PieceKind::index`].
const fn ref_kind_to_index(ref_pt: usize) -> Option<usize> {
    match ref_pt {
        1 => Some(0), // PAWN
        2 => Some(1), // LANCE
        3 => Some(2), // KNIGHT
        4 => Some(3), // SILVER
        5 => Some(5), // BISHOP
        6 => Some(6), // ROOK
        7 => Some(4), // GOLD
        8 => Some(7), // KING
        _ => None,
    }
}

/// All Zobrist tables, generated once at compile time.
struct Zobrist {
    /// Per-`(piece code, square)` term, XORed into `board_key`. The partial keys
    /// (`pawn_key` / `minor_piece_key` / `non_pawn_key`) reuse this same table —
    /// the reference XORs `Zobrist::psq[pc][s]` into every one of them.
    psq: [[u64; Square::COUNT]; PIECE_CODES],
    /// Per-`(color, hand-kind)` step, added to `hand_key` once per held copy.
    /// Indexed by [`PieceKind::index`]; the King slot is present but unused.
    hand: [[u64; PieceKind::COUNT]; Color::COUNT],
    /// Side-to-move term, XORed into `board_key` while White is to move.
    side: u64,
    /// The empty-board value of `pawn_key`: a dedicated non-zero constant
    /// (the reference's `Zobrist::noPawns`, `position.cpp`). `pawn_key`
    /// starts here — *not* at zero — so that a position with no board pawns
    /// still carries a distinct pawn-structure key.
    no_pawns: u64,
}

const fn build() -> Zobrist {
    let mut psq = [[0u64; Square::COUNT]; PIECE_CODES];
    let mut hand = [[0u64; PieceKind::COUNT]; Color::COUNT];

    // The reference's `Zobrist::zero` takes no draw, so the first two are
    // `side` then `noPawns`.
    let mut s = REF_SEED;
    let r = set_rand(s);
    s = r.0;
    let side = r.1;
    let r = set_rand(s);
    s = r.0;
    let no_pawns = r.1;

    // Every non-zero code consumes four words per square even when it never
    // lands on a board, keeping the stream aligned with the reference.
    let mut pc = 1;
    while pc <= 31 {
        let mut sq = 0;
        while sq < Square::COUNT {
            let r = set_rand(s);
            s = r.0;
            if let Some(slot) = ref_code_to_slot(pc) {
                psq[slot][sq] = r.1;
            }
            sq += 1;
        }
        pc += 1;
    }

    // `pr` runs over the seven hand kinds: there is no king in hand.
    let mut c = 0;
    while c < Color::COUNT {
        let mut pr = 1;
        while pr <= 7 {
            let r = set_rand(s);
            s = r.0;
            if let Some(kind) = ref_kind_to_index(pr) {
                hand[c][kind] = r.1;
            }
            pr += 1;
        }
        c += 1;
    }

    Zobrist {
        psq,
        hand,
        side,
        no_pawns,
    }
}

static ZOBRIST: Zobrist = build();

/// The `noPawns` seed for the `const` contexts that cannot read the `static`
/// [`ZOBRIST`]. Recomputing `build()` happens at compile time only.
pub(crate) const NO_PAWNS_SEED: u64 = build().no_pawns;

/// Encode a piece into its Zobrist table index (`0..PIECE_CODES`).
const fn piece_code(piece: Piece) -> usize {
    let promo = if piece.promoted { 1 } else { 0 };
    (promo * Color::COUNT + piece.color.index()) * PieceKind::COUNT + piece.kind.index()
}

/// `board_key` term for `piece` sitting on `sq`. XOR to place, XOR again to
/// remove.
pub(crate) fn psq(piece: Piece, sq: Square) -> u64 {
    ZOBRIST.psq[piece_code(piece)][sq.index() as usize]
}

/// `hand_key` step for one held `kind` of color `color`. Add once per copy
/// gained, subtract once per copy lost.
pub(crate) fn hand_step(color: Color, kind: PieceKind) -> u64 {
    ZOBRIST.hand[color.index()][kind.index()]
}

/// Side-to-move term. XORed into `board_key` while White is to move; toggled
/// on every move.
pub(crate) fn side() -> u64 {
    ZOBRIST.side
}

/// Whether `piece` is a *minor piece* for the `minor_piece_key`
/// (`minor_piece_table`, `position.cpp`). Bishop, rook, horse, dragon, king and
/// pawn are **not** minor.
pub(crate) fn is_minor_piece(piece: Piece) -> bool {
    if piece.promoted {
        matches!(
            piece.kind,
            PieceKind::Pawn | PieceKind::Lance | PieceKind::Knight | PieceKind::Silver
        )
    } else {
        matches!(
            piece.kind,
            PieceKind::Lance | PieceKind::Knight | PieceKind::Silver | PieceKind::Gold
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The four never-realised promoted-gold and promoted-king codes stay zero:
    /// the reference never draws them, and this port never reads them.
    #[test]
    fn all_reachable_table_entries_are_nonzero() {
        const KINDS: [PieceKind; 8] = [
            PieceKind::Pawn,
            PieceKind::Lance,
            PieceKind::Knight,
            PieceKind::Silver,
            PieceKind::Gold,
            PieceKind::Bishop,
            PieceKind::Rook,
            PieceKind::King,
        ];
        for color in [Color::Black, Color::White] {
            for kind in KINDS {
                let mut pieces = vec![Piece::new(kind, color)];
                if let Some(p) = Piece::promoted(kind, color) {
                    pieces.push(p);
                }
                for piece in pieces {
                    let code = piece_code(piece);
                    for s in 0..Square::COUNT {
                        assert_ne!(
                            ZOBRIST.psq[code][s], 0,
                            "psq[{code}][{s}] ({piece:?}) is zero"
                        );
                    }
                }
                if kind != PieceKind::King {
                    assert_ne!(
                        ZOBRIST.hand[color.index()][kind.index()],
                        0,
                        "hand[{}][{}] ({kind:?}) is zero",
                        color.index(),
                        kind.index()
                    );
                }
            }
        }
        assert_ne!(ZOBRIST.side, 0);
        assert_ne!(ZOBRIST.no_pawns, 0, "noPawns must be a non-zero constant");
    }

    #[test]
    fn is_minor_piece_matches_reference_table() {
        use Color::Black;
        assert!(!is_minor_piece(Piece::new(PieceKind::Pawn, Black)));
        assert!(is_minor_piece(Piece::new(PieceKind::Lance, Black)));
        assert!(is_minor_piece(Piece::new(PieceKind::Knight, Black)));
        assert!(is_minor_piece(Piece::new(PieceKind::Silver, Black)));
        assert!(is_minor_piece(Piece::new(PieceKind::Gold, Black)));
        assert!(!is_minor_piece(Piece::new(PieceKind::Bishop, Black)));
        assert!(!is_minor_piece(Piece::new(PieceKind::Rook, Black)));
        assert!(!is_minor_piece(Piece::new(PieceKind::King, Black)));
        for kind in [
            PieceKind::Pawn,
            PieceKind::Lance,
            PieceKind::Knight,
            PieceKind::Silver,
        ] {
            assert!(is_minor_piece(Piece::promoted(kind, Black).unwrap()));
        }
        assert!(!is_minor_piece(
            Piece::promoted(PieceKind::Bishop, Black).unwrap()
        ));
        assert!(!is_minor_piece(
            Piece::promoted(PieceKind::Rook, Black).unwrap()
        ));
    }

    #[test]
    fn piece_code_is_a_bijection_over_realised_pieces() {
        let mut seen = std::collections::HashSet::new();
        for promoted in [false, true] {
            for color in [Color::Black, Color::White] {
                for kind in [
                    PieceKind::Pawn,
                    PieceKind::Lance,
                    PieceKind::Knight,
                    PieceKind::Silver,
                    PieceKind::Gold,
                    PieceKind::Bishop,
                    PieceKind::Rook,
                    PieceKind::King,
                ] {
                    let piece = Piece {
                        kind,
                        color,
                        promoted,
                    };
                    let code = piece_code(piece);
                    assert!(code < PIECE_CODES);
                    assert!(seen.insert(code), "duplicate code {code} for {piece:?}");
                }
            }
        }
        assert_eq!(seen.len(), PIECE_CODES);
    }

    #[test]
    fn distinct_pieces_have_distinct_psq_on_same_square() {
        let sq = Square::new(4, 4).unwrap();
        let a = psq(Piece::new(PieceKind::Pawn, Color::Black), sq);
        let b = psq(Piece::new(PieceKind::Pawn, Color::White), sq);
        let c = psq(Piece::new(PieceKind::Rook, Color::Black), sq);
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_ne!(b, c);
    }
}
