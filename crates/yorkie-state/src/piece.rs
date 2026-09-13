use crate::color::Color;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PieceKind {
    Pawn = 0,
    Lance = 1,
    Knight = 2,
    Silver = 3,
    Gold = 4,
    Bishop = 5,
    Rook = 6,
    King = 7,
}

impl PieceKind {
    pub const COUNT: usize = 8;

    pub const fn index(self) -> usize {
        let i = self as usize;
        debug_assert!(i < Self::COUNT);
        // SAFETY: the enum defines a variant for exactly the discriminants
        // `0..COUNT`, so this holds for every `PieceKind` that exists. See
        // `Square::index` for why the bound is stated rather than inferred.
        unsafe { core::hint::assert_unchecked(i < Self::COUNT) };
        i
    }

    pub const fn can_promote(self) -> bool {
        matches!(
            self,
            PieceKind::Pawn
                | PieceKind::Lance
                | PieceKind::Knight
                | PieceKind::Silver
                | PieceKind::Bishop
                | PieceKind::Rook
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Piece {
    pub kind: PieceKind,
    pub color: Color,
    pub promoted: bool,
}

// Three bytes, one per field, and an empty square costs no fourth: `PieceKind`
// leaves 248 of its 256 discriminants free, so `Option<Piece>` takes one of them
// for the absent case and the 81-square board is 243 bytes.
const _: () = assert!(size_of::<Piece>() == 3);
const _: () = assert!(size_of::<Option<Piece>>() == size_of::<Piece>());
const _: () = assert!(align_of::<Piece>() == 1);

impl Piece {
    pub const fn new(kind: PieceKind, color: Color) -> Self {
        Self {
            kind,
            color,
            promoted: false,
        }
    }

    pub const fn promoted(kind: PieceKind, color: Color) -> Option<Self> {
        if kind.can_promote() {
            Some(Self {
                kind,
                color,
                promoted: true,
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_unpromoted() {
        let p = Piece::new(PieceKind::Pawn, Color::Black);
        assert_eq!(p.kind, PieceKind::Pawn);
        assert_eq!(p.color, Color::Black);
        assert!(!p.promoted);
    }

    #[test]
    fn promoted_rejects_gold_and_king() {
        assert!(Piece::promoted(PieceKind::Gold, Color::Black).is_none());
        assert!(Piece::promoted(PieceKind::King, Color::White).is_none());
    }

    #[test]
    fn promoted_accepts_promotable_kinds() {
        for kind in [
            PieceKind::Pawn,
            PieceKind::Lance,
            PieceKind::Knight,
            PieceKind::Silver,
            PieceKind::Bishop,
            PieceKind::Rook,
        ] {
            let p = Piece::promoted(kind, Color::Black).unwrap();
            assert!(p.promoted);
            assert_eq!(p.kind, kind);
        }
    }
}
