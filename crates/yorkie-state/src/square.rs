use core::fmt;
use core::mem::transmute;

macro_rules! board_squares {
    ($($name:ident),+ $(,)?) => {
        /// One of the 81 board squares.
        ///
        /// A variant's discriminant *is* the square's index, so
        /// [`index`](Square::index) is a cast and construction is nothing but
        /// the range check. The variant names are never spelled by the code —
        /// `S<i>` names the discriminant it carries — and the 175 discriminants
        /// the enum leaves undefined are a niche, so `Option<Square>` needs no
        /// tag byte and is as wide as a `Square`. Holding an index-plus-one in
        /// a `NonZeroU8` would buy the same niche without `unsafe`, but it pays
        /// an add on every construction and a subtract on every read, and the
        /// board is walked a few hundred million times a second.
        ///
        /// The size assertions below hold the niche to its promise: a
        /// toolchain that stopped folding it fails the build instead of
        /// silently widening every `Option<Square>` in the tree.
        #[derive(Clone, Copy, PartialEq, Eq, Hash)]
        #[repr(u8)]
        pub enum Square {
            $($name),+
        }
    };
}

board_squares! {
    S0, S1, S2, S3, S4, S5, S6, S7, S8,
    S9, S10, S11, S12, S13, S14, S15, S16, S17,
    S18, S19, S20, S21, S22, S23, S24, S25, S26,
    S27, S28, S29, S30, S31, S32, S33, S34, S35,
    S36, S37, S38, S39, S40, S41, S42, S43, S44,
    S45, S46, S47, S48, S49, S50, S51, S52, S53,
    S54, S55, S56, S57, S58, S59, S60, S61, S62,
    S63, S64, S65, S66, S67, S68, S69, S70, S71,
    S72, S73, S74, S75, S76, S77, S78, S79, S80,
}

const _: () = assert!(size_of::<Square>() == 1);
const _: () = assert!(size_of::<Option<Square>>() == 1);
/// The last variant's discriminant pins the whole run to `0..COUNT`, which is
/// what the `transmute` in [`Square::from_index`] rests on.
const _: () = assert!(Square::S80 as u8 as usize == Square::COUNT - 1);

impl Square {
    pub const FILES: u8 = 9;
    pub const RANKS: u8 = 9;
    pub const COUNT: usize = 81;

    pub const fn new(file: u8, rank: u8) -> Option<Self> {
        if file < Self::FILES && rank < Self::RANKS {
            Self::from_index(file * Self::RANKS + rank)
        } else {
            None
        }
    }

    pub const fn from_index(index: u8) -> Option<Self> {
        if (index as usize) >= Self::COUNT {
            return None;
        }
        // SAFETY: the enum defines a variant for every discriminant in
        // `0..COUNT` — the assertion above pins the last of them — and the
        // check just made bounds `index` to that range.
        Some(unsafe { transmute::<u8, Self>(index) })
    }

    pub const fn index(self) -> u8 {
        self as u8
    }

    pub const fn file(self) -> u8 {
        self.index() / Self::RANKS
    }

    pub const fn rank(self) -> u8 {
        self.index() % Self::RANKS
    }
}

/// Shows the square's index, which is what the rest of the engine addresses a
/// square by; the variant name carries nothing the index does not.
impl fmt::Debug for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Square").field(&self.index()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corner_round_trip() {
        let sq = Square::new(8, 8).unwrap();
        assert_eq!(sq.file(), 8);
        assert_eq!(sq.rank(), 8);
        assert_eq!(sq.index(), 80);
    }

    #[test]
    fn origin_round_trip() {
        let sq = Square::new(0, 0).unwrap();
        assert_eq!(sq.file(), 0);
        assert_eq!(sq.rank(), 0);
        assert_eq!(sq.index(), 0);
    }

    #[test]
    fn out_of_range_is_none() {
        assert!(Square::new(9, 0).is_none());
        assert!(Square::new(0, 9).is_none());
        assert!(Square::new(9, 9).is_none());
    }

    #[test]
    fn debug_shows_the_index() {
        assert_eq!(format!("{:?}", Square::from_index(0).unwrap()), "Square(0)");
        assert_eq!(
            format!("{:?}", Square::from_index(80).unwrap()),
            "Square(80)"
        );
    }

    #[test]
    fn from_index_out_of_range_is_none() {
        assert!(Square::from_index(81).is_none());
        assert!(Square::from_index(255).is_none());
    }

    #[test]
    fn every_index_is_its_own_discriminant() {
        for i in 0..(Square::COUNT as u8) {
            assert_eq!(Square::from_index(i).unwrap().index(), i);
        }
    }

    #[test]
    fn all_indices_round_trip() {
        for file in 0..Square::FILES {
            for rank in 0..Square::RANKS {
                let sq = Square::new(file, rank).unwrap();
                assert_eq!(sq.file(), file);
                assert_eq!(sq.rank(), rank);
                assert_eq!(Square::from_index(sq.index()).unwrap(), sq);
            }
        }
    }
}
