#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Color {
    Black = 0,
    White = 1,
}

impl Color {
    pub const COUNT: usize = 2;

    pub const fn flip(self) -> Self {
        match self {
            Color::Black => Color::White,
            Color::White => Color::Black,
        }
    }

    pub const fn index(self) -> usize {
        let i = self as usize;
        debug_assert!(i < Self::COUNT);
        // SAFETY: the enum defines a variant for exactly the discriminants
        // `0..COUNT`, so this holds for every `Color` that exists. See
        // `Square::index` for why the bound is stated rather than inferred.
        unsafe { core::hint::assert_unchecked(i < Self::COUNT) };
        i
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flip_is_involution() {
        assert_eq!(Color::Black.flip(), Color::White);
        assert_eq!(Color::White.flip(), Color::Black);
        assert_eq!(Color::Black.flip().flip(), Color::Black);
        assert_eq!(Color::White.flip().flip(), Color::White);
    }

    #[test]
    fn index_matches_repr() {
        assert_eq!(Color::Black.index(), 0);
        assert_eq!(Color::White.index(), 1);
    }
}
