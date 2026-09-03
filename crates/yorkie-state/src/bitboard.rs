//! An 81-square bitset (`Bitboard`) and the precomputed effect and geometry
//! tables the attack-scan and slider substrate consume.
//!
//! # Bit convention
//!
//! A [`Bitboard`] uses the reference's two-lane layout (`bitboard.h`): a
//! 16-byte-aligned pair of `u64` lanes. Contiguous square index `k` maps to
//! `p[0]` bit `k` for `k = 0..=62` and `p[1]` bit `k - 63` for `k = 63..=80`.
//! **Bit 63 of `p[0]` is deliberately unused** — the reference reserves it so
//! lance and pawn-drop borrow tricks cannot carry across the lane boundary — as
//! are `p[1]` bits `18..`. Every value a public constructor or operator returns
//! keeps those spare bits clear.
//!
//! [`Bitboard::raw`] / [`Bitboard::from_raw`] present a *logical* contiguous
//! 81-bit `u128` view that closes the gap. They cross the lane boundary on
//! every call, so hot paths must not use them.
//!
//! The tables are `const`, so they cost no runtime initialization. Their
//! contents are defined by [`crate::movegen`]'s step data and `step_signed`
//! walk, against which the tests below pin them exhaustively.

use crate::color::Color;
use crate::movegen::{GOLD_STEPS, KING_STEPS, KNIGHT_STEPS, PAWN_STEPS, SILVER_STEPS};
use crate::square::Square;

/// The number of board squares (and meaningful low bits).
const N: usize = Square::COUNT; // 81
/// Files per board (= ranks per board); an index is `file * RANKS + rank`.
const RANKS: usize = Square::RANKS as usize; // 9
const FILES: usize = Square::FILES as usize; // 9

/// All 81 valid square bits set in the *contiguous* `u128` domain.
const BOARD_MASK: u128 = (1u128 << N) - 1;

/// The highest contiguous square index in lane `p[0]` (`part`, `bitboard.h`).
const LANE_SPLIT: usize = 62;
/// Width of lane 0 in the contiguous domain (`p[0]` bit 63 is the unused gap).
const LANE0_SPAN: u32 = 63;
/// Contiguous-`u128` mask of `p[0]`'s live bits.
const LOW63: u128 = (1u128 << LANE0_SPAN) - 1;
/// Canonical live-bit mask of lane 0, `p[0]` bits 0..=62.
const P0_MASK: u64 = 0x7FFF_FFFF_FFFF_FFFF;
/// Canonical live-bit mask of lane 1, `p[1]` bits 0..=17.
const P1_MASK: u64 = 0x0000_0000_0003_FFFF;

/// An 81-square bit set in the reference's two-lane layout — see the module bit
/// convention. The 16-byte alignment lets the set operators load a value as one
/// `__m128i` while the layout stays `const`-constructible for the compile-time
/// tables.
#[derive(Clone, Copy, Eq, Default)]
#[repr(C, align(16))]
pub struct Bitboard {
    p: [u64; 2],
}

/// Per-square single-bit table (`SquareBB`, `bitboard.cpp`), so
/// [`Bitboard::from_square`] is a table load rather than a lane branch.
const SQUARE_BB: [Bitboard; N] = {
    let mut t = [Bitboard::EMPTY; N];
    let mut idx = 0;
    while idx < N {
        t[idx] = if idx > LANE_SPLIT {
            Bitboard {
                p: [0, 1u64 << (idx - LANE0_SPAN as usize)],
            }
        } else {
            Bitboard {
                p: [1u64 << idx, 0],
            }
        };
        idx += 1;
    }
    t
};

impl Bitboard {
    /// The empty set.
    pub const EMPTY: Bitboard = Bitboard { p: [0, 0] };

    /// Every valid board square set (all 81 live lane bits).
    pub const FULL: Bitboard = Bitboard {
        p: [P0_MASK, P1_MASK],
    };

    /// The empty set.
    pub const fn empty() -> Bitboard {
        Bitboard { p: [0, 0] }
    }

    /// The single-square set `{sq}`.
    pub const fn from_square(sq: Square) -> Bitboard {
        SQUARE_BB[sq.index() as usize]
    }

    /// Split a contiguous 81-bit `u128` into the two lanes, closing the `p[0]`
    /// bit-63 gap. Masks off out-of-board bits, so the spare-bit invariant
    /// always holds.
    const fn from_contiguous(bits: u128) -> Bitboard {
        let bits = bits & BOARD_MASK;
        Bitboard {
            p: [(bits & LOW63) as u64, (bits >> LANE0_SPAN) as u64],
        }
    }

    /// Pack the two lanes back into a contiguous 81-bit `u128`, the inverse of
    /// [`Self::from_contiguous`].
    const fn to_contiguous(self) -> u128 {
        (self.p[0] as u128) | ((self.p[1] as u128) << LANE0_SPAN)
    }

    /// Wrap a *logical* contiguous 81-bit pattern, masking off out-of-board
    /// bits. Crosses the lane gap on every call, so hot paths must not use it.
    pub const fn from_raw(bits: u128) -> Bitboard {
        Bitboard::from_contiguous(bits)
    }

    /// The *logical* contiguous 81-bit pattern, bit `i` being square `i`.
    /// Crosses the lane gap on every call, so hot paths must not use it.
    pub const fn raw(self) -> u128 {
        self.to_contiguous()
    }

    /// Is the set empty?
    pub const fn is_empty(self) -> bool {
        self.p[0] == 0 && self.p[1] == 0
    }

    /// Does the set contain `sq`?
    pub fn test(self, sq: Square) -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            !lane_testz(self.p, SQUARE_BB[sq.index() as usize].p)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.contains_index(sq.index() as usize)
        }
    }

    /// Insert `sq`.
    pub fn set(&mut self, sq: Square) {
        *self = self.or(Bitboard::from_square(sq));
    }

    /// Remove `sq`.
    pub fn clear(&mut self, sq: Square) {
        *self = self.without_index(sq.index() as usize);
    }

    /// The number of squares in the set.
    pub const fn popcount(self) -> u32 {
        self.p[0].count_ones() + self.p[1].count_ones()
    }

    /// Iterate the member squares in **ascending index order** — the order the
    /// reference's `bb.pop()` yields, which the generation-order fixtures
    /// depend on.
    pub const fn squares(self) -> BitboardIter {
        BitboardIter {
            p0: self.p[0],
            p1: self.p[1],
        }
    }

    // The bitwise operator traits are not `const`-callable, so the const table
    // builders reach for these index-addressed primitives instead.

    /// The single-square set for a raw board index. An index `>= 81` yields the
    /// empty set, as an `& BOARD_MASK` would.
    pub(crate) const fn single(index: usize) -> Bitboard {
        if index > N - 1 {
            Bitboard::EMPTY
        } else if index > LANE_SPLIT {
            Bitboard {
                p: [0, 1u64 << (index - LANE0_SPAN as usize)],
            }
        } else {
            Bitboard {
                p: [1u64 << index, 0],
            }
        }
    }

    /// `const` bitwise union.
    pub(crate) const fn or(self, other: Bitboard) -> Bitboard {
        Bitboard {
            p: [self.p[0] | other.p[0], self.p[1] | other.p[1]],
        }
    }

    /// `self` with the raw index `index` removed.
    pub(crate) const fn without_index(self, index: usize) -> Bitboard {
        if index > LANE_SPLIT {
            Bitboard {
                p: [
                    self.p[0],
                    self.p[1] & !(1u64 << (index - LANE0_SPAN as usize)),
                ],
            }
        } else {
            Bitboard {
                p: [self.p[0] & !(1u64 << index), self.p[1]],
            }
        }
    }

    /// Does the set contain the raw board index `index`?
    pub(crate) const fn contains_index(self, index: usize) -> bool {
        if index > LANE_SPLIT {
            self.p[1] & (1u64 << (index - LANE0_SPAN as usize)) != 0
        } else {
            self.p[0] & (1u64 << index) != 0
        }
    }

    /// The least set square's raw index, or `128` when empty — callers guard
    /// with [`Self::is_empty`] first.
    pub(crate) const fn lowest_index(self) -> u32 {
        if let Some(i) = self.p[0].lowest_one() {
            i
        } else if let Some(i) = self.p[1].lowest_one() {
            i + LANE0_SPAN
        } else {
            128
        }
    }
}

/// Debug-only guard on the spare-bit invariant.
fn debug_assert_canonical(bb: Bitboard) {
    debug_assert!(
        bb.p[0] & !P0_MASK == 0 && bb.p[1] & !P1_MASK == 0,
        "Bitboard spare bits set: {:#018x} {:#018x}",
        bb.p[0],
        bb.p[1],
    );
}

/// SSE2 lane `OR`.
///
/// SAFETY: SSE2 is in the x86_64 base ABI, so `_mm_or_si128` is always callable
/// here; the `transmute`s only bit-cast between `[u64; 2]` and `__m128i`, which
/// share size (16 bytes) and accept any bit pattern.
#[cfg(target_arch = "x86_64")]
fn lane_or(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
    use core::arch::x86_64::{__m128i, _mm_or_si128};
    unsafe {
        let va: __m128i = core::mem::transmute(a);
        let vb: __m128i = core::mem::transmute(b);
        let r: [u64; 2] = core::mem::transmute(_mm_or_si128(va, vb));
        r
    }
}

/// SSE2 lane `AND`. SAFETY: as [`lane_or`].
#[cfg(target_arch = "x86_64")]
fn lane_and(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
    use core::arch::x86_64::{__m128i, _mm_and_si128};
    unsafe {
        let va: __m128i = core::mem::transmute(a);
        let vb: __m128i = core::mem::transmute(b);
        let r: [u64; 2] = core::mem::transmute(_mm_and_si128(va, vb));
        r
    }
}

/// SSE2 lane `XOR`. SAFETY: as [`lane_or`].
#[cfg(target_arch = "x86_64")]
fn lane_xor(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
    use core::arch::x86_64::{__m128i, _mm_xor_si128};
    unsafe {
        let va: __m128i = core::mem::transmute(a);
        let vb: __m128i = core::mem::transmute(b);
        let r: [u64; 2] = core::mem::transmute(_mm_xor_si128(va, vb));
        r
    }
}

/// SSE4.1 zero-overlap test: `(a & b) == 0`, the reference's
/// `_mm_testz_si128` (`bitboard.cpp`).
///
/// SAFETY: SSE4.1 is part of the target's assumed base (this repo builds with
/// `-C target-cpu`/`target-feature` covering it, as the rest of the SSE helpers
/// already rely on); the `transmute`s only bit-cast `[u64; 2]` ↔ `__m128i`.
#[cfg(target_arch = "x86_64")]
fn lane_testz(a: [u64; 2], b: [u64; 2]) -> bool {
    use core::arch::x86_64::{__m128i, _mm_testz_si128};
    unsafe {
        let va: __m128i = core::mem::transmute(a);
        let vb: __m128i = core::mem::transmute(b);
        _mm_testz_si128(va, vb) != 0
    }
}

/// SSE equality: `pxor` then `ptest`. SAFETY: as [`lane_testz`].
#[cfg(target_arch = "x86_64")]
fn lane_eq(a: [u64; 2], b: [u64; 2]) -> bool {
    use core::arch::x86_64::{__m128i, _mm_test_all_zeros, _mm_xor_si128};
    unsafe {
        let va: __m128i = core::mem::transmute(a);
        let vb: __m128i = core::mem::transmute(b);
        let neq = _mm_xor_si128(va, vb);
        _mm_test_all_zeros(neq, neq) != 0
    }
}

/// Byte-reverse the 128-bit value: one `_mm_shuffle_epi8`.
/// SAFETY: as [`lane_testz`]; the shuffle mask reverses all 16 bytes.
#[cfg(target_arch = "x86_64")]
fn lane_byte_reverse(a: [u64; 2]) -> [u64; 2] {
    use core::arch::x86_64::{__m128i, _mm_set_epi8, _mm_shuffle_epi8};
    unsafe {
        let va: __m128i = core::mem::transmute(a);
        let shuffle = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        core::mem::transmute(_mm_shuffle_epi8(va, shuffle))
    }
}

/// SSE unpack: `(unpackhi(lo_in, hi_in), unpacklo(lo_in, hi_in))`.
/// SAFETY: as [`lane_testz`].
#[cfg(target_arch = "x86_64")]
fn lane_unpack(hi_in: [u64; 2], lo_in: [u64; 2]) -> ([u64; 2], [u64; 2]) {
    use core::arch::x86_64::{__m128i, _mm_unpackhi_epi64, _mm_unpacklo_epi64};
    unsafe {
        let vh: __m128i = core::mem::transmute(hi_in);
        let vl: __m128i = core::mem::transmute(lo_in);
        let hi_out: [u64; 2] = core::mem::transmute(_mm_unpackhi_epi64(vl, vh));
        let lo_out: [u64; 2] = core::mem::transmute(_mm_unpacklo_epi64(vl, vh));
        (hi_out, lo_out)
    }
}

/// SSE 128-bit decrement of the whole register (`Bitboard::decrement`).
/// SAFETY: as [`lane_testz`].
#[cfg(all(test, target_arch = "x86_64"))]
fn lane_decrement(a: [u64; 2]) -> [u64; 2] {
    use core::arch::x86_64::{
        __m128i, _mm_add_epi64, _mm_alignr_epi8, _mm_cmpeq_epi64, _mm_set1_epi64x,
        _mm_setzero_si128,
    };
    unsafe {
        let m: __m128i = core::mem::transmute(a);
        let t2 = _mm_cmpeq_epi64(m, _mm_setzero_si128());
        let t2 = _mm_alignr_epi8(t2, _mm_set1_epi64x(-1i64), 8);
        core::mem::transmute(_mm_add_epi64(m, t2))
    }
}

/// SSE pairwise 128-bit decrement (`Bitboard::decrement(hi, lo, …)`): each lane
/// index `i` decrements the pair `[lo_in[i], hi_in[i]]`.
/// SAFETY: as [`lane_testz`].
#[cfg(target_arch = "x86_64")]
fn lane_pair_decrement(hi_in: [u64; 2], lo_in: [u64; 2]) -> ([u64; 2], [u64; 2]) {
    use core::arch::x86_64::{
        __m128i, _mm_add_epi64, _mm_cmpeq_epi64, _mm_set1_epi64x, _mm_setzero_si128,
    };
    unsafe {
        let vh: __m128i = core::mem::transmute(hi_in);
        let vl: __m128i = core::mem::transmute(lo_in);
        let hi_out: [u64; 2] =
            core::mem::transmute(_mm_add_epi64(vh, _mm_cmpeq_epi64(vl, _mm_setzero_si128())));
        let lo_out: [u64; 2] = core::mem::transmute(_mm_add_epi64(vl, _mm_set1_epi64x(-1i64)));
        (hi_out, lo_out)
    }
}

// Scalar fallbacks for the non-x86_64 path.

#[cfg(not(target_arch = "x86_64"))]
fn lane_eq(a: [u64; 2], b: [u64; 2]) -> bool {
    a[0] == b[0] && a[1] == b[1]
}

#[cfg(not(target_arch = "x86_64"))]
fn lane_byte_reverse(a: [u64; 2]) -> [u64; 2] {
    [a[1].swap_bytes(), a[0].swap_bytes()]
}

#[cfg(not(target_arch = "x86_64"))]
fn lane_unpack(hi_in: [u64; 2], lo_in: [u64; 2]) -> ([u64; 2], [u64; 2]) {
    ([lo_in[1], hi_in[1]], [lo_in[0], hi_in[0]])
}

#[cfg(all(test, not(target_arch = "x86_64")))]
fn lane_decrement(a: [u64; 2]) -> [u64; 2] {
    [
        a[0].wrapping_sub(1),
        if a[0] == 0 {
            a[1].wrapping_sub(1)
        } else {
            a[1]
        },
    ]
}

#[cfg(not(target_arch = "x86_64"))]
fn lane_pair_decrement(hi_in: [u64; 2], lo_in: [u64; 2]) -> ([u64; 2], [u64; 2]) {
    (
        [
            hi_in[0].wrapping_sub((lo_in[0] == 0) as u64),
            hi_in[1].wrapping_sub((lo_in[1] == 0) as u64),
        ],
        [lo_in[0].wrapping_sub(1), lo_in[1].wrapping_sub(1)],
    )
}

impl PartialEq for Bitboard {
    /// Equivalent to the derived field compare, because every canonical value
    /// keeps its spare lane bits clear.
    fn eq(&self, other: &Bitboard) -> bool {
        lane_eq(self.p, other.p)
    }
}

impl core::hash::Hash for Bitboard {
    /// Hashes the two lanes directly, which agrees with the manual
    /// [`PartialEq`] because canonical values keep their spare bits clear.
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.p.hash(state);
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn lane_or(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
    [a[0] | b[0], a[1] | b[1]]
}

#[cfg(not(target_arch = "x86_64"))]
fn lane_and(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
    [a[0] & b[0], a[1] & b[1]]
}

#[cfg(not(target_arch = "x86_64"))]
fn lane_xor(a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
    [a[0] ^ b[0], a[1] ^ b[1]]
}

/// Ascending-index iterator over a [`Bitboard`]'s member squares.
#[derive(Clone, Copy)]
pub struct BitboardIter {
    p0: u64,
    p1: u64,
}

impl Iterator for BitboardIter {
    type Item = Square;

    fn next(&mut self) -> Option<Square> {
        if let Some(i) = self.p0.lowest_one() {
            self.p0 &= self.p0 - 1;
            return Some(Square::from_index(i as u8).expect("lane-0 bit index < 63"));
        }
        if let Some(i) = self.p1.lowest_one() {
            self.p1 &= self.p1 - 1;
            return Some(
                Square::from_index((i + LANE0_SPAN) as u8).expect("lane-1 bit index < 81"),
            );
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = (self.p0.count_ones() + self.p1.count_ones()) as usize;
        (n, Some(n))
    }
}

impl ExactSizeIterator for BitboardIter {}

impl IntoIterator for Bitboard {
    type Item = Square;
    type IntoIter = BitboardIter;

    fn into_iter(self) -> BitboardIter {
        self.squares()
    }
}

impl core::ops::BitOr for Bitboard {
    type Output = Bitboard;
    fn bitor(self, rhs: Bitboard) -> Bitboard {
        let bb = Bitboard {
            p: lane_or(self.p, rhs.p),
        };
        debug_assert_canonical(bb);
        bb
    }
}

impl core::ops::BitAnd for Bitboard {
    type Output = Bitboard;
    fn bitand(self, rhs: Bitboard) -> Bitboard {
        let bb = Bitboard {
            p: lane_and(self.p, rhs.p),
        };
        debug_assert_canonical(bb);
        bb
    }
}

impl core::ops::BitXor for Bitboard {
    type Output = Bitboard;
    fn bitxor(self, rhs: Bitboard) -> Bitboard {
        let bb = Bitboard {
            p: lane_xor(self.p, rhs.p),
        };
        debug_assert_canonical(bb);
        bb
    }
}

impl core::ops::Not for Bitboard {
    type Output = Bitboard;
    /// Complement within the 81-square board. `XOR` with [`Bitboard::FULL`]
    /// rather than a bitwise not, so the spare lane bits stay clear.
    fn not(self) -> Bitboard {
        self ^ Bitboard::FULL
    }
}

impl core::ops::BitOrAssign for Bitboard {
    fn bitor_assign(&mut self, rhs: Bitboard) {
        *self = *self | rhs;
    }
}

impl core::ops::BitAndAssign for Bitboard {
    fn bitand_assign(&mut self, rhs: Bitboard) {
        *self = *self & rhs;
    }
}

impl core::ops::BitXorAssign for Bitboard {
    fn bitxor_assign(&mut self, rhs: Bitboard) {
        *self = *self ^ rhs;
    }
}

impl core::fmt::Debug for Bitboard {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Rank-major 9x9 grid, rank 0 at the top, `*` for a set square.
        writeln!(f, "Bitboard(0x{:021x}):", self.to_contiguous())?;
        for rank in 0..RANKS {
            for file in (0..FILES).rev() {
                let sq = Square::new(file as u8, rank as u8).expect("in range");
                f.write_str(if self.test(sq) { "*" } else { "." })?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

const fn file_of(idx: usize) -> i8 {
    (idx / RANKS) as i8
}

const fn rank_of(idx: usize) -> i8 {
    (idx % RANKS) as i8
}

const fn on_board(file: i8, rank: i8) -> bool {
    file >= 0 && file < FILES as i8 && rank >= 0 && rank < RANKS as i8
}

const fn to_index(file: i8, rank: i8) -> usize {
    (file as usize) * RANKS + (rank as usize)
}

const fn isign(x: i8) -> i8 {
    if x > 0 {
        1
    } else if x < 0 {
        -1
    } else {
        0
    }
}

const fn iabs(x: i8) -> i8 {
    if x < 0 { -x } else { x }
}

/// Build a step-attack table for one color from a step-delta list, applying
/// each `(df, dr)` as `(df, dr * dr_sign)` exactly as
/// [`crate::movegen::step_signed`] does.
const fn build_step_table(deltas: &[(i8, i8)], dr_sign: i8) -> [Bitboard; N] {
    let mut table = [Bitboard::EMPTY; N];
    let mut idx = 0;
    while idx < N {
        let file = file_of(idx);
        let rank = rank_of(idx);
        let mut set = 0u128;
        let mut d = 0;
        while d < deltas.len() {
            let (df, dr) = deltas[d];
            let f = file + df;
            let r = rank + dr * dr_sign;
            if on_board(f, r) {
                set |= 1u128 << to_index(f, r);
            }
            d += 1;
        }
        table[idx] = Bitboard::from_contiguous(set);
        idx += 1;
    }
    table
}

/// `[color][sq]` step-attack tables, `Color::index()`-ordered (Black, White).
const fn build_colored(deltas: &[(i8, i8)]) -> [[Bitboard; N]; Color::COUNT] {
    [build_step_table(deltas, 1), build_step_table(deltas, -1)]
}

const PAWN_ATTACKS: [[Bitboard; N]; Color::COUNT] = build_colored(PAWN_STEPS);
const KNIGHT_ATTACKS: [[Bitboard; N]; Color::COUNT] = build_colored(KNIGHT_STEPS);
const SILVER_ATTACKS: [[Bitboard; N]; Color::COUNT] = build_colored(SILVER_STEPS);
const GOLD_ATTACKS: [[Bitboard; N]; Color::COUNT] = build_colored(GOLD_STEPS);
const KING_ATTACKS: [[Bitboard; N]; Color::COUNT] = build_colored(KING_STEPS);

/// Squares a pawn of `color` on `sq` attacks (step effect).
pub fn pawn_attacks(color: Color, sq: Square) -> Bitboard {
    PAWN_ATTACKS[color.index()][sq.index() as usize]
}

/// Squares a knight of `color` on `sq` attacks (step effect).
pub fn knight_attacks(color: Color, sq: Square) -> Bitboard {
    KNIGHT_ATTACKS[color.index()][sq.index() as usize]
}

/// Squares a silver of `color` on `sq` attacks (step effect).
pub fn silver_attacks(color: Color, sq: Square) -> Bitboard {
    SILVER_ATTACKS[color.index()][sq.index() as usize]
}

/// Squares a gold of `color` on `sq` attacks (step effect). Also the effect of
/// the four promoted minors (`+P`/`+L`/`+N`/`+S`), which move as gold.
pub fn gold_attacks(color: Color, sq: Square) -> Bitboard {
    GOLD_ATTACKS[color.index()][sq.index() as usize]
}

/// Squares a king of `color` on `sq` attacks (step effect). Color-independent
/// in value (the king ring is symmetric), but kept per-color for a uniform API.
pub fn king_attacks(color: Color, sq: Square) -> Bitboard {
    KING_ATTACKS[color.index()][sq.index() as usize]
}

/// The eight queen-line directions as `(df, dr)` unit vectors. `RAY[dir]` and
/// [`ray_dir`] both index this array; the order is arbitrary but fixed.
pub const DIRECTIONS: [(i8, i8); 8] = [
    (0, -1),  // toward rank 0
    (0, 1),   // toward rank 8
    (1, 0),   // toward file 8
    (-1, 0),  // toward file 0
    (1, -1),  // diagonal
    (-1, -1), // diagonal
    (1, 1),   // diagonal
    (-1, 1),  // diagonal
];

/// Number of ray directions.
pub const DIR_COUNT: usize = DIRECTIONS.len();

/// Build one direction's ray table in the contiguous-`u128` domain: entry
/// `[sq]` is every square from `sq` toward the board edge along `(df, dr)`,
/// excluding `sq`.
const fn build_ray_bits_table(df: i8, dr: i8) -> [u128; N] {
    let mut table = [0u128; N];
    let mut idx = 0;
    while idx < N {
        let mut file = file_of(idx);
        let mut rank = rank_of(idx);
        let mut set = 0u128;
        loop {
            file += df;
            rank += dr;
            if !on_board(file, rank) {
                break;
            }
            set |= 1u128 << to_index(file, rank);
        }
        table[idx] = set;
        idx += 1;
    }
    table
}

const fn build_ray_bits() -> [[u128; N]; DIR_COUNT] {
    let mut rays = [[0u128; N]; DIR_COUNT];
    let mut d = 0;
    while d < DIR_COUNT {
        let (df, dr) = DIRECTIONS[d];
        rays[d] = build_ray_bits_table(df, dr);
        d += 1;
    }
    rays
}

/// Contiguous-`u128` ray masks. A `const` rather than a `static` so the
/// `const fn` builders below can read it.
const RAY_BITS: [[u128; N]; DIR_COUNT] = build_ray_bits();

const fn build_rays() -> [[Bitboard; N]; DIR_COUNT] {
    let mut rays = [[Bitboard::EMPTY; N]; DIR_COUNT];
    let mut d = 0;
    while d < DIR_COUNT {
        let mut s = 0;
        while s < N {
            rays[d][s] = Bitboard::from_contiguous(RAY_BITS[d][s]);
            s += 1;
        }
        d += 1;
    }
    rays
}

static RAY: [[Bitboard; N]; DIR_COUNT] = build_rays();

/// The ray from `sq` to the board edge in direction `dir`, an index into
/// [`DIRECTIONS`], excluding `sq`.
pub fn ray(dir: usize, sq: Square) -> Bitboard {
    RAY[dir][sq.index() as usize]
}

/// The occupancy-free forward-file ray of a `color` lance on `sq` — the
/// reference `lanceStepEffect<C>`.
pub fn lance_step_effect(color: Color, sq: Square) -> Bitboard {
    match color {
        Color::Black => RAY[0][sq.index() as usize], // toward rank 0
        Color::White => RAY[1][sq.index() as usize], // toward rank 8
    }
}

/// Diagonal step neighbours — the four 45° adjacent squares. The pattern is its
/// own mirror, so a single colour-independent table suffices.
const CROSS45_STEPS: &[(i8, i8)] = &[(-1, -1), (1, -1), (-1, 1), (1, 1)];
static CROSS45: [Bitboard; N] = build_step_table(CROSS45_STEPS, 1);

/// Build one composite ray table: entry `[sq]` is the union of the four
/// [`RAY_BITS`] rays named in `dirs`, excluding `sq`.
const fn build_multi_ray(dirs: [usize; 4]) -> [Bitboard; N] {
    let mut table = [Bitboard::EMPTY; N];
    let mut s = 0;
    while s < N {
        let mut set = 0u128;
        let mut i = 0;
        while i < 4 {
            set |= RAY_BITS[dirs[i]][s];
            i += 1;
        }
        table[s] = Bitboard::from_contiguous(set);
        s += 1;
    }
    table
}

// [`DIRECTIONS`] indices: 0-3 are the orthogonals, 4-7 the diagonals.
static ROOK_STEP: [Bitboard; N] = build_multi_ray([0, 1, 2, 3]);
static BISHOP_STEP: [Bitboard; N] = build_multi_ray([4, 5, 6, 7]);

/// The four diagonal step neighbours of `sq` (occupancy-free) — the reference
/// `cross45StepEffect`.
pub fn cross45_step_effect(sq: Square) -> Bitboard {
    CROSS45[sq.index() as usize]
}

/// The full orthogonal cross through `sq` to the board edges (occupancy-free) —
/// the reference `rookStepEffect`.
pub fn rook_step_effect(sq: Square) -> Bitboard {
    ROOK_STEP[sq.index() as usize]
}

/// The full diagonal cross through `sq` to the board edges (occupancy-free) —
/// the reference `bishopStepEffect`.
pub fn bishop_step_effect(sq: Square) -> Bitboard {
    BISHOP_STEP[sq.index() as usize]
}

/// The ray from `from` toward `to`, extended to the board edge and excluding
/// `from`; [`Bitboard::EMPTY`] when the two do not share a queen-line. The
/// result includes `to` itself and every square beyond it.
pub fn ray_toward(from: Square, to: Square) -> Bitboard {
    let d = DIR_OF[from.index() as usize][to.index() as usize];
    if (d as usize) < DIR_COUNT {
        RAY[d as usize][from.index() as usize]
    } else {
        Bitboard::EMPTY
    }
}

/// The [`DIRECTIONS`] index of the unit direction from `a` to `b` when the two
/// lie on a queen-line, else `DIR_COUNT` as a "none" sentinel.
const fn dir_index(a: usize, b: usize) -> u8 {
    let df = file_of(b) - file_of(a);
    let dr = rank_of(b) - rank_of(a);
    if df == 0 && dr == 0 {
        return DIR_COUNT as u8;
    }
    let (udf, udr) = if df == 0 {
        (0, isign(dr))
    } else if dr == 0 {
        (isign(df), 0)
    } else if iabs(df) == iabs(dr) {
        (isign(df), isign(dr))
    } else {
        return DIR_COUNT as u8;
    };
    let mut i = 0;
    while i < DIR_COUNT {
        let (cdf, cdr) = DIRECTIONS[i];
        if cdf == udf && cdr == udr {
            return i as u8;
        }
        i += 1;
    }
    DIR_COUNT as u8
}

const fn build_dir_of() -> [[u8; N]; N] {
    let mut table = [[DIR_COUNT as u8; N]; N];
    let mut a = 0;
    while a < N {
        let mut b = 0;
        while b < N {
            table[a][b] = dir_index(a, b);
            b += 1;
        }
        a += 1;
    }
    table
}

const DIR_OF: [[u8; N]; N] = build_dir_of();

/// The unit ray direction from `from` to `to` when the two lie on a queen-line,
/// else `None` — the reference `Effect8::directions_of`.
pub fn ray_dir(from: Square, to: Square) -> Option<(i8, i8)> {
    let d = DIR_OF[from.index() as usize][to.index() as usize];
    if (d as usize) < DIR_COUNT {
        Some(DIRECTIONS[d as usize])
    } else {
        None
    }
}

/// The squares strictly between `a` and `b` along their shared queen-line,
/// excluding both endpoints; empty if they are not aligned.
const fn between_pair(a: usize, b: usize) -> u128 {
    let d = dir_index(a, b);
    if d as usize >= DIR_COUNT {
        return 0;
    }
    let (df, dr) = DIRECTIONS[d as usize];
    let mut file = file_of(a);
    let mut rank = rank_of(a);
    let mut set = 0u128;
    loop {
        file += df;
        rank += dr;
        if !on_board(file, rank) {
            // Unreachable for an aligned pair — the walk hits `b` first — but
            // keeps the loop total.
            break;
        }
        let idx = to_index(file, rank);
        if idx == b {
            break;
        }
        set |= 1u128 << idx;
    }
    set
}

const fn build_between() -> [[Bitboard; N]; N] {
    let mut table = [[Bitboard::EMPTY; N]; N];
    let mut a = 0;
    while a < N {
        let mut b = 0;
        while b < N {
            table[a][b] = Bitboard::from_contiguous(between_pair(a, b));
            b += 1;
        }
        a += 1;
    }
    table
}

// A `static` rather than a `const`: the 81×81 table is ~105 KB, so it must live
// once in `.rodata` instead of being materialized at each use site.
static BETWEEN: [[Bitboard; N]; N] = build_between();

/// The squares strictly between `a` and `b` on a queen-line, endpoints
/// excluded, else the empty set — the reference `between_bb(a, b)`.
pub fn between(a: Square, b: Square) -> Bitboard {
    BETWEEN[a.index() as usize][b.index() as usize]
}

const fn build_file_masks() -> [Bitboard; FILES] {
    let mut masks = [Bitboard::EMPTY; FILES];
    let mut file = 0;
    while file < FILES {
        let mut set = 0u128;
        let mut rank = 0;
        while rank < RANKS {
            set |= 1u128 << to_index(file as i8, rank as i8);
            rank += 1;
        }
        masks[file] = Bitboard::from_contiguous(set);
        file += 1;
    }
    masks
}

const fn build_rank_masks() -> [Bitboard; RANKS] {
    let mut masks = [Bitboard::EMPTY; RANKS];
    let mut rank = 0;
    while rank < RANKS {
        let mut set = 0u128;
        let mut file = 0;
        while file < FILES {
            set |= 1u128 << to_index(file as i8, rank as i8);
            file += 1;
        }
        masks[rank] = Bitboard::from_contiguous(set);
        rank += 1;
    }
    masks
}

const FILE_MASKS: [Bitboard; FILES] = build_file_masks();
const RANK_MASKS: [Bitboard; RANKS] = build_rank_masks();

/// Every square on `file` (0..=8).
pub fn file_mask(file: u8) -> Bitboard {
    FILE_MASKS[file as usize]
}

/// Every square on `rank` (0..=8).
pub fn rank_mask(rank: u8) -> Bitboard {
    RANK_MASKS[rank as usize]
}

/// Per-color promotion zone, matching [`crate::movegen::is_in_promotion_zone`].
const fn build_promo_zones() -> [Bitboard; Color::COUNT] {
    let black = RANK_MASKS[0].or(RANK_MASKS[1]).or(RANK_MASKS[2]);
    let white = RANK_MASKS[6].or(RANK_MASKS[7]).or(RANK_MASKS[8]);
    [black, white]
}

const PROMO_ZONES: [Bitboard; Color::COUNT] = build_promo_zones();

/// The promotion zone of `color`: Black ranks 0..=2, White ranks 6..=8.
pub fn promotion_zone(color: Color) -> Bitboard {
    PROMO_ZONES[color.index()]
}

// Occupancy-limited slider attack queries — the Qugiy (2021) scheme, ported
// from `bitboard.h` / `bitboard.cpp`. Each query cuts every ray at and
// including the first occupied square, branchlessly.
//
// A file ray never straddles the lane split, so lance and rook-file rays are
// computed per lane with plain `u64` borrow arithmetic. Rank and diagonal rays
// do cross it, and use `byte_reverse` + `unpack` + 128-bit `decrement` instead:
// reversing the byte order puts a decreasing direction's nearest blocker in the
// lowest bit, so subtracting 1 propagates the borrow to it. The masks for
// decreasing-index directions are therefore stored *already* byte-reversed.

/// Which lane a square belongs to (`part`, `bitboard.h`).
const fn part(idx: usize) -> usize {
    (idx > LANE_SPLIT) as usize
}

/// Index of the highest set bit (`MSB64`). Callers pass `x | 1`, so `x != 0`;
/// this underflows on `x == 0`.
const fn msb64(x: u64) -> u32 {
    x.bit_width() - 1
}

// `const` scalar twins of the lane byte-reverse and unpack, for the
// compile-time mask tables: SSE intrinsics are not `const`-callable.

const fn cbyte_reverse(p: [u64; 2]) -> [u64; 2] {
    [p[1].swap_bytes(), p[0].swap_bytes()]
}

const fn cunpack(hi_in: [u64; 2], lo_in: [u64; 2]) -> ([u64; 2], [u64; 2]) {
    ([lo_in[1], hi_in[1]], [lo_in[0], hi_in[0]])
}

/// `[sq][0]` is the rank ray toward file 0 packed as the `lo` unpack lane,
/// `[sq][1]` the byte-reversed ray toward file 8 as the `hi` lane
/// (`bitboard.cpp`).
const fn build_qugiy_rook_mask() -> [[Bitboard; 2]; N] {
    let mut t = [[Bitboard::EMPTY; 2]; N];
    let mut s = 0;
    while s < N {
        // Toward file 8 is increasing index, toward file 0 decreasing.
        let left = Bitboard::from_contiguous(RAY_BITS[2][s]).p;
        let right = Bitboard::from_contiguous(RAY_BITS[3][s]).p;
        let right_rev = cbyte_reverse(right);
        let (hi, lo) = cunpack(right_rev, left);
        t[s][0] = Bitboard { p: lo };
        t[s][1] = Bitboard { p: hi };
        s += 1;
    }
    t
}

static QUGIY_ROOK_MASK: [[Bitboard; 2]; N] = build_qugiy_rook_mask();

/// The four bishop diagonals in [`DIRECTIONS`] order, and whether each runs
/// toward *decreasing* square index — so that its mask is byte-reversed and the
/// query byte-reverses the occupancy. The reference `bishopEffect`
/// (`bitboard.cpp`) calls the increasing pair LU/LD and the decreasing one
/// RU/RD, and packs the mask as `[LU, RU, LD, RD]` per 64-bit lane so the
/// unpacked halves feed `occ` to the non-reversed diagonals and `rocc` to the
/// reversed ones.
const BISHOP_DIAG_DIRS: [usize; 4] = [4, 5, 6, 7];
const BISHOP_DIAG_REV: [bool; 4] = [false, true, false, true];

/// Two [`Bitboard`]s packed into one 256-bit value (`Bitboard256`,
/// `bitboard.h`), so `bishop_attacks` covers all four diagonals in one pass.
/// Only the operations that query consumes are ported.
#[derive(Clone, Copy)]
#[repr(C, align(32))]
struct Bitboard256 {
    p: [u64; 4],
}

impl Bitboard256 {
    /// Low half `b1`, high half `b2`.
    const fn from_pair(b1: Bitboard, b2: Bitboard) -> Bitboard256 {
        Bitboard256 {
            p: [b1.p[0], b1.p[1], b2.p[0], b2.p[1]],
        }
    }
}

/// Broadcast one [`Bitboard`]'s lanes into both halves.
///
/// SAFETY: gated on `target_feature = "avx2"` (statically enabled by the
/// release `target-cpu=native`); the `transmute`s only bit-cast between
/// `[u64; N]` and the equally-sized `__m128i` / `__m256i`.
#[cfg(target_feature = "avx2")]
fn bb256_broadcast(a: [u64; 2]) -> [u64; 4] {
    use core::arch::x86_64::{__m128i, __m256i, _mm256_broadcastsi128_si256};
    unsafe {
        let va: __m128i = core::mem::transmute(a);
        core::mem::transmute::<__m256i, [u64; 4]>(_mm256_broadcastsi128_si256(va))
    }
}

/// Bitboard256 `AND` (pin `_mm256_and_si256`). SAFETY: as [`bb256_broadcast`].
#[cfg(target_feature = "avx2")]
fn bb256_and(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
    use core::arch::x86_64::{__m256i, _mm256_and_si256};
    unsafe {
        let va: __m256i = core::mem::transmute(a);
        let vb: __m256i = core::mem::transmute(b);
        core::mem::transmute(_mm256_and_si256(va, vb))
    }
}

/// Bitboard256 `OR` (pin `_mm256_or_si256`). SAFETY: as [`bb256_broadcast`].
#[cfg(target_feature = "avx2")]
fn bb256_or(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
    use core::arch::x86_64::{__m256i, _mm256_or_si256};
    unsafe {
        let va: __m256i = core::mem::transmute(a);
        let vb: __m256i = core::mem::transmute(b);
        core::mem::transmute(_mm256_or_si256(va, vb))
    }
}

/// Bitboard256 `XOR` (pin `_mm256_xor_si256`). SAFETY: as [`bb256_broadcast`].
#[cfg(target_feature = "avx2")]
fn bb256_xor(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
    use core::arch::x86_64::{__m256i, _mm256_xor_si256};
    unsafe {
        let va: __m256i = core::mem::transmute(a);
        let vb: __m256i = core::mem::transmute(b);
        core::mem::transmute(_mm256_xor_si256(va, vb))
    }
}

/// Byte-reverse each 128-bit half. SAFETY: as [`bb256_broadcast`]; the shuffle
/// mask reverses all 16 bytes within each half.
#[cfg(target_feature = "avx2")]
fn bb256_byte_reverse(a: [u64; 4]) -> [u64; 4] {
    use core::arch::x86_64::{__m256i, _mm256_set_epi8, _mm256_shuffle_epi8};
    unsafe {
        let va: __m256i = core::mem::transmute(a);
        let shuffle = _mm256_set_epi8(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15,
        );
        core::mem::transmute(_mm256_shuffle_epi8(va, shuffle))
    }
}

/// Bitboard256 `unpack`, per 128-bit lane. SAFETY: as [`bb256_broadcast`].
#[cfg(target_feature = "avx2")]
fn bb256_unpack(hi_in: [u64; 4], lo_in: [u64; 4]) -> ([u64; 4], [u64; 4]) {
    use core::arch::x86_64::{__m256i, _mm256_unpackhi_epi64, _mm256_unpacklo_epi64};
    unsafe {
        let vh: __m256i = core::mem::transmute(hi_in);
        let vl: __m256i = core::mem::transmute(lo_in);
        let hi_out: [u64; 4] = core::mem::transmute(_mm256_unpackhi_epi64(vl, vh));
        let lo_out: [u64; 4] = core::mem::transmute(_mm256_unpacklo_epi64(vl, vh));
        (hi_out, lo_out)
    }
}

/// Bitboard256 pairwise 128-bit decrement: each lane index `i` decrements the
/// pair `[lo_in[i], hi_in[i]]`. SAFETY: as [`bb256_broadcast`].
#[cfg(target_feature = "avx2")]
fn bb256_pair_decrement(hi_in: [u64; 4], lo_in: [u64; 4]) -> ([u64; 4], [u64; 4]) {
    use core::arch::x86_64::{
        __m256i, _mm256_add_epi64, _mm256_cmpeq_epi64, _mm256_set1_epi64x, _mm256_setzero_si256,
    };
    unsafe {
        let vh: __m256i = core::mem::transmute(hi_in);
        let vl: __m256i = core::mem::transmute(lo_in);
        let hi_out: [u64; 4] = core::mem::transmute(_mm256_add_epi64(
            vh,
            _mm256_cmpeq_epi64(vl, _mm256_setzero_si256()),
        ));
        let lo_out: [u64; 4] =
            core::mem::transmute(_mm256_add_epi64(vl, _mm256_set1_epi64x(-1i64)));
        (hi_out, lo_out)
    }
}

/// Merge the two halves into one [`Bitboard`] by OR.
/// SAFETY: as [`bb256_broadcast`].
#[cfg(target_feature = "avx2")]
fn bb256_merge(a: [u64; 4]) -> [u64; 2] {
    use core::arch::x86_64::{
        __m128i, __m256i, _mm_or_si128, _mm256_castsi256_si128, _mm256_extracti128_si256,
    };
    unsafe {
        let va: __m256i = core::mem::transmute(a);
        let lo: __m128i = _mm256_castsi256_si128(va);
        let hi: __m128i = _mm256_extracti128_si256::<1>(va);
        core::mem::transmute(_mm_or_si128(lo, hi))
    }
}

// Scalar `u64[4]` twins. On the non-AVX2 build these *are* the `bb256_*` ops;
// a test build keeps them so the AVX2 path can be gated against them.

#[cfg(any(test, not(target_feature = "avx2")))]
fn bb256_broadcast_scalar(a: [u64; 2]) -> [u64; 4] {
    [a[0], a[1], a[0], a[1]]
}

#[cfg(any(test, not(target_feature = "avx2")))]
fn bb256_and_scalar(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
    [a[0] & b[0], a[1] & b[1], a[2] & b[2], a[3] & b[3]]
}

#[cfg(any(test, not(target_feature = "avx2")))]
fn bb256_or_scalar(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
    [a[0] | b[0], a[1] | b[1], a[2] | b[2], a[3] | b[3]]
}

#[cfg(any(test, not(target_feature = "avx2")))]
fn bb256_xor_scalar(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
    [a[0] ^ b[0], a[1] ^ b[1], a[2] ^ b[2], a[3] ^ b[3]]
}

#[cfg(any(test, not(target_feature = "avx2")))]
fn bb256_byte_reverse_scalar(a: [u64; 4]) -> [u64; 4] {
    [
        a[1].swap_bytes(),
        a[0].swap_bytes(),
        a[3].swap_bytes(),
        a[2].swap_bytes(),
    ]
}

#[cfg(any(test, not(target_feature = "avx2")))]
fn bb256_unpack_scalar(hi_in: [u64; 4], lo_in: [u64; 4]) -> ([u64; 4], [u64; 4]) {
    (
        [lo_in[1], hi_in[1], lo_in[3], hi_in[3]],
        [lo_in[0], hi_in[0], lo_in[2], hi_in[2]],
    )
}

#[cfg(any(test, not(target_feature = "avx2")))]
fn bb256_pair_decrement_scalar(hi_in: [u64; 4], lo_in: [u64; 4]) -> ([u64; 4], [u64; 4]) {
    let mut hi_out = [0u64; 4];
    let mut lo_out = [0u64; 4];
    let mut i = 0;
    while i < 4 {
        hi_out[i] = hi_in[i].wrapping_sub((lo_in[i] == 0) as u64);
        lo_out[i] = lo_in[i].wrapping_sub(1);
        i += 1;
    }
    (hi_out, lo_out)
}

#[cfg(any(test, not(target_feature = "avx2")))]
fn bb256_merge_scalar(a: [u64; 4]) -> [u64; 2] {
    [a[0] | a[2], a[1] | a[3]]
}

#[cfg(not(target_feature = "avx2"))]
use bb256_and_scalar as bb256_and;
#[cfg(not(target_feature = "avx2"))]
use bb256_broadcast_scalar as bb256_broadcast;
#[cfg(not(target_feature = "avx2"))]
use bb256_byte_reverse_scalar as bb256_byte_reverse;
#[cfg(not(target_feature = "avx2"))]
use bb256_merge_scalar as bb256_merge;
#[cfg(not(target_feature = "avx2"))]
use bb256_or_scalar as bb256_or;
#[cfg(not(target_feature = "avx2"))]
use bb256_pair_decrement_scalar as bb256_pair_decrement;
#[cfg(not(target_feature = "avx2"))]
use bb256_unpack_scalar as bb256_unpack;
#[cfg(not(target_feature = "avx2"))]
use bb256_xor_scalar as bb256_xor;

/// `[sq][i]` is the four diagonal step effects packed as the `Bitboard256` pair
/// `[LU, RU, LD, RD]` in 64-bit lane `i`, with RU and RD stored byte-reversed
/// (`bitboard.cpp`).
const fn build_qugiy_bishop_mask() -> [[Bitboard256; 2]; N] {
    let mut t = [[Bitboard256 { p: [0; 4] }; 2]; N];
    let mut s = 0;
    while s < N {
        // Byte-reversed for the decreasing directions, per `BISHOP_DIAG_REV`.
        let mut diag = [[0u64; 2]; 4];
        let mut d = 0;
        while d < 4 {
            let step = Bitboard::from_contiguous(RAY_BITS[BISHOP_DIAG_DIRS[d]][s]).p;
            diag[d] = if BISHOP_DIAG_REV[d] {
                cbyte_reverse(step)
            } else {
                step
            };
            d += 1;
        }
        // LU=diag[0], RU=diag[1], LD=diag[2], RD=diag[3].
        let mut i = 0;
        while i < 2 {
            t[s][i] = Bitboard256::from_pair(
                Bitboard {
                    p: [diag[0][i], diag[1][i]],
                },
                Bitboard {
                    p: [diag[2][i], diag[3][i]],
                },
            );
            i += 1;
        }
        s += 1;
    }
    t
}

static QUGIY_BISHOP_MASK: [[Bitboard256; 2]; N] = build_qugiy_bishop_mask();

/// The increasing-index (White) file ray cut at the first blocker, per lane.
fn file_up(occ_lane: u64, mask: u64) -> u64 {
    let em = occ_lane & mask;
    let t = em.wrapping_sub(1);
    (em ^ t) & mask
}

/// The decreasing-index (Black) file ray cut at the first blocker, per lane.
fn file_down(occ_lane: u64, se: u64) -> u64 {
    let mocc = se & occ_lane;
    let filled = (!0u64) << msb64(mocc | 1);
    filled & se
}

/// Squares a lance of `color` on `sq` attacks under occupancy `occ` — the
/// single forward ray, cut at the first blocker.
pub fn lance_attacks(color: Color, sq: Square, occ: Bitboard) -> Bitboard {
    let s = sq.index() as usize;
    let pt = part(s);
    let occ_lane = occ.p[pt];
    let mut out = [0u64; 2];
    out[pt] = match color {
        // White: dir 1 (0,1), increasing index.
        Color::White => file_up(occ_lane, RAY[1][s].p[pt]),
        // Black: dir 0 (0,-1), decreasing index.
        Color::Black => file_down(occ_lane, RAY[0][s].p[pt]),
    };
    Bitboard { p: out }
}

/// The rook's file effect — the two lance forms fused on the shared lane.
fn rook_file(s: usize, occ: Bitboard) -> Bitboard {
    let pt = part(s);
    let occ_lane = occ.p[pt];
    let up = file_up(occ_lane, RAY[1][s].p[pt]);
    let down = file_down(occ_lane, RAY[0][s].p[pt]);
    let mut out = [0u64; 2];
    out[pt] = up | down;
    Bitboard { p: out }
}

/// The rook's rank effect, which crosses the lane boundary and so takes the
/// `byte_reverse` + `unpack` + `decrement` route.
fn rook_rank(s: usize, occ: Bitboard) -> Bitboard {
    let mask_lo = QUGIY_ROOK_MASK[s][0].p;
    let mask_hi = QUGIY_ROOK_MASK[s][1].p;

    let rocc = lane_byte_reverse(occ.p);
    let (hi, lo) = lane_unpack(rocc, occ.p);
    let hi = lane_and(hi, mask_hi);
    let lo = lane_and(lo, mask_lo);

    let (t1, t0) = lane_pair_decrement(hi, lo);
    let t1 = lane_and(lane_xor(t1, hi), mask_hi);
    let t0 = lane_and(lane_xor(t0, lo), mask_lo);

    let (hi2, lo2) = lane_unpack(t1, t0);
    Bitboard {
        p: lane_or(lane_byte_reverse(hi2), lo2),
    }
}

/// Squares a bishop on `sq` attacks under occupancy `occ` — all four diagonals
/// in one `Bitboard256` pass (`bishopEffect`, `bitboard.cpp`).
pub fn bishop_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    let s = sq.index() as usize;
    let mask_lo = QUGIY_BISHOP_MASK[s][0].p;
    let mask_hi = QUGIY_BISHOP_MASK[s][1].p;

    let occ2 = bb256_broadcast(occ.p);
    let rocc2 = bb256_broadcast(lane_byte_reverse(occ.p));

    let (hi, lo) = bb256_unpack(rocc2, occ2);
    let hi = bb256_and(hi, mask_hi);
    let lo = bb256_and(lo, mask_lo);

    let (t1, t0) = bb256_pair_decrement(hi, lo);
    let t1 = bb256_and(bb256_xor(t1, hi), mask_hi);
    let t0 = bb256_and(bb256_xor(t0, lo), mask_lo);

    let (hi2, lo2) = bb256_unpack(t1, t0);
    Bitboard {
        p: bb256_merge(bb256_or(bb256_byte_reverse(hi2), lo2)),
    }
}

/// Squares a rook on `sq` attacks under occupancy `occ`.
pub fn rook_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    let s = sq.index() as usize;
    rook_rank(s, occ) | rook_file(s, occ)
}

/// Squares a horse on `sq` attacks under occupancy `occ`: bishop rays plus the
/// king ring, whose diagonal neighbours already lie on those rays.
pub fn horse_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    bishop_attacks(sq, occ) | KING_ATTACKS[0][sq.index() as usize]
}

/// Squares a dragon on `sq` attacks under occupancy `occ`: rook rays plus the
/// king ring.
pub fn dragon_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    rook_attacks(sq, occ) | KING_ATTACKS[0][sq.index() as usize]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::movegen::{dr_sign_for, movement, step_signed};
    use crate::piece::{Piece, PieceKind};

    fn all_squares() -> impl Iterator<Item = Square> {
        (0..Square::COUNT as u8).map(|i| Square::from_index(i).unwrap())
    }

    // Table-driven ray-walk slides, the oracle the branchless production forms
    // are gated against.

    const ORACLE_DIR_POSITIVE: [bool; DIR_COUNT] = {
        let mut t = [false; DIR_COUNT];
        let mut d = 0;
        while d < DIR_COUNT {
            let (df, dr) = DIRECTIONS[d];
            t[d] = (df as isize) * (RANKS as isize) + (dr as isize) > 0;
            d += 1;
        }
        t
    };

    const ORACLE_BISHOP_DIRS: [usize; 4] = [4, 5, 6, 7];
    const ORACLE_ROOK_DIRS: [usize; 4] = [0, 1, 2, 3];

    /// The occupancy-limited attack ray of `sq` in direction `dir`: every square
    /// from `sq` toward the edge up to **and including** the first occupied one.
    fn ray_attacks(dir: usize, sq: Square, occ: Bitboard) -> Bitboard {
        let full = RAY_BITS[dir][sq.index() as usize];
        let blockers = full & occ.raw();
        if blockers == 0 {
            return Bitboard::from_raw(full);
        }
        // The `blockers == 0` guard above keeps this in range.
        let first = if ORACLE_DIR_POSITIVE[dir] {
            blockers.trailing_zeros()
        } else {
            blockers.bit_width() - 1
        };
        Bitboard::from_raw(full ^ RAY_BITS[dir][first as usize])
    }

    fn lance_attacks_oracle(color: Color, sq: Square, occ: Bitboard) -> Bitboard {
        let dir = match color {
            Color::Black => 0,
            Color::White => 1,
        };
        ray_attacks(dir, sq, occ)
    }

    fn bishop_attacks_oracle(sq: Square, occ: Bitboard) -> Bitboard {
        let mut bb = Bitboard::EMPTY;
        let mut i = 0;
        while i < ORACLE_BISHOP_DIRS.len() {
            bb |= ray_attacks(ORACLE_BISHOP_DIRS[i], sq, occ);
            i += 1;
        }
        bb
    }

    fn rook_attacks_oracle(sq: Square, occ: Bitboard) -> Bitboard {
        let mut bb = Bitboard::EMPTY;
        let mut i = 0;
        while i < ORACLE_ROOK_DIRS.len() {
            bb |= ray_attacks(ORACLE_ROOK_DIRS[i], sq, occ);
            i += 1;
        }
        bb
    }

    fn horse_attacks_oracle(sq: Square, occ: Bitboard) -> Bitboard {
        bishop_attacks_oracle(sq, occ) | KING_ATTACKS[0][sq.index() as usize]
    }

    fn dragon_attacks_oracle(sq: Square, occ: Bitboard) -> Bitboard {
        rook_attacks_oracle(sq, occ) | KING_ATTACKS[0][sq.index() as usize]
    }

    /// Every origin square and slider form, over empty, all-ones, and 100_000
    /// deterministic random occupancies.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn qugiy_slides_equal_ray_walk_oracle() {
        // Deterministic xorshift-128; no `rand` dependency in tests.
        let mut state: u128 = 0xDEAD_BEEF_CAFE_F00D_0123_4567_89AB_CDEF;
        let mut rng = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state & BOARD_MASK
        };
        let mut occs = vec![0u128, BOARD_MASK];
        for _ in 0..100_000 {
            occs.push(rng());
        }
        for occ in occs {
            let occ_bb = Bitboard::from_raw(occ);
            for from in all_squares() {
                for color in [Color::Black, Color::White] {
                    assert_eq!(
                        lance_attacks(color, from, occ_bb),
                        lance_attacks_oracle(color, from, occ_bb),
                        "lance {color:?} from {from:?} occ {occ:#x}",
                    );
                }
                assert_eq!(
                    rook_attacks(from, occ_bb),
                    rook_attacks_oracle(from, occ_bb),
                    "rook from {from:?} occ {occ:#x}",
                );
                assert_eq!(
                    bishop_attacks(from, occ_bb),
                    bishop_attacks_oracle(from, occ_bb),
                    "bishop from {from:?} occ {occ:#x}",
                );
                assert_eq!(
                    horse_attacks(from, occ_bb),
                    horse_attacks_oracle(from, occ_bb),
                    "horse from {from:?} occ {occ:#x}",
                );
                assert_eq!(
                    dragon_attacks(from, occ_bb),
                    dragon_attacks_oracle(from, occ_bb),
                    "dragon from {from:?} occ {occ:#x}",
                );
            }
        }
    }

    // A second, independent oracle: the same slides in the contiguous-`u128`
    // domain. `neg_ray_rev` works in the `reverse_bits` domain rather than the
    // production form's `byte_reverse`, because the contiguous 81-bit layout is
    // not byte-aligned per file and so a byte swap does not invert bit order
    // there.

    fn oracle_rev_ray(dir: usize, s: usize) -> u128 {
        RAY_BITS[dir][s].reverse_bits()
    }

    fn oracle_pos_ray(ray: u128, occ: u128) -> u128 {
        let b = occ & ray;
        ray & (b ^ b.wrapping_sub(1))
    }

    fn oracle_neg_ray_rev(rray: u128, rocc: u128) -> u128 {
        let b = rocc & rray;
        rray & (b ^ b.wrapping_sub(1))
    }

    fn lance_attacks_packed(color: Color, sq: Square, occ: Bitboard) -> Bitboard {
        let s = sq.index() as usize;
        let occ = occ.raw();
        match color {
            Color::Black => Bitboard::from_raw(
                oracle_neg_ray_rev(oracle_rev_ray(0, s), occ.reverse_bits()).reverse_bits(),
            ),
            Color::White => Bitboard::from_raw(oracle_pos_ray(RAY_BITS[1][s], occ)),
        }
    }

    fn bishop_attacks_packed(sq: Square, occ: Bitboard) -> Bitboard {
        let s = sq.index() as usize;
        let occ = occ.raw();
        let rocc = occ.reverse_bits();
        let pos = oracle_pos_ray(RAY_BITS[4][s], occ) | oracle_pos_ray(RAY_BITS[6][s], occ);
        let neg = (oracle_neg_ray_rev(oracle_rev_ray(5, s), rocc)
            | oracle_neg_ray_rev(oracle_rev_ray(7, s), rocc))
        .reverse_bits();
        Bitboard::from_raw(pos | neg)
    }

    fn rook_attacks_packed(sq: Square, occ: Bitboard) -> Bitboard {
        let s = sq.index() as usize;
        let occ = occ.raw();
        let rocc = occ.reverse_bits();
        let pos = oracle_pos_ray(RAY_BITS[1][s], occ) | oracle_pos_ray(RAY_BITS[2][s], occ);
        let neg = (oracle_neg_ray_rev(oracle_rev_ray(0, s), rocc)
            | oracle_neg_ray_rev(oracle_rev_ray(3, s), rocc))
        .reverse_bits();
        Bitboard::from_raw(pos | neg)
    }

    /// Independent of the ray-walk oracle above.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn lane_native_equals_packed_qugiy_oracle() {
        for occ in sample_occupancies() {
            let occ_bb = Bitboard::from_raw(occ);
            for from in all_squares() {
                for color in [Color::Black, Color::White] {
                    assert_eq!(
                        lance_attacks(color, from, occ_bb),
                        lance_attacks_packed(color, from, occ_bb),
                        "lance {color:?} from {from:?} occ {occ:#x}",
                    );
                }
                assert_eq!(
                    rook_attacks(from, occ_bb),
                    rook_attacks_packed(from, occ_bb),
                    "rook from {from:?} occ {occ:#x}",
                );
                assert_eq!(
                    bishop_attacks(from, occ_bb),
                    bishop_attacks_packed(from, occ_bb),
                    "bishop from {from:?} occ {occ:#x}",
                );
                assert_eq!(
                    horse_attacks(from, occ_bb),
                    bishop_attacks_packed(from, occ_bb) | KING_ATTACKS[0][from.index() as usize],
                    "horse from {from:?} occ {occ:#x}",
                );
                assert_eq!(
                    dragon_attacks(from, occ_bb),
                    rook_attacks_packed(from, occ_bb) | KING_ATTACKS[0][from.index() as usize],
                    "dragon from {from:?} occ {occ:#x}",
                );
            }
        }
    }

    /// Recomputes the mask tables the way `Bitboards::init` does and compares
    /// lane for lane.
    #[test]
    fn qugiy_mask_tables_match_recomputation() {
        for s in 0..N {
            // Rook rank masks.
            let left = Bitboard::from_raw(RAY_BITS[2][s]).p;
            let right = Bitboard::from_raw(RAY_BITS[3][s]).p;
            let right_rev = cbyte_reverse(right);
            let (hi, lo) = cunpack(right_rev, left);
            assert_eq!(QUGIY_ROOK_MASK[s][0].p, lo, "rook mask lo at {s}");
            assert_eq!(QUGIY_ROOK_MASK[s][1].p, hi, "rook mask hi at {s}");

            // Bishop diagonal masks in the `Bitboard256` pair format.
            let mut diag = [[0u64; 2]; 4];
            for d in 0..4 {
                let step = Bitboard::from_raw(RAY_BITS[BISHOP_DIAG_DIRS[d]][s]).p;
                diag[d] = if BISHOP_DIAG_REV[d] {
                    cbyte_reverse(step)
                } else {
                    step
                };
            }
            for i in 0..2 {
                let want = [diag[0][i], diag[1][i], diag[2][i], diag[3][i]];
                assert_eq!(
                    QUGIY_BISHOP_MASK[s][i].p, want,
                    "bishop mask256 pair {i} at {s}"
                );
            }
        }
    }

    // A third bishop oracle: one `rayEffect` per diagonal, at `Bitboard` rather
    // than `Bitboard256` width, so its masks are rebuilt here.

    /// One diagonal's occupancy-limited ray via `rayEffect` (`bitboard.h`).
    fn diag_ray_oracle(mask: [u64; 2], occ: [u64; 2], reverse: bool) -> [u64; 2] {
        let mut bb = if reverse { lane_byte_reverse(occ) } else { occ };
        bb = lane_and(bb, mask);
        let bb_minus = lane_decrement(bb);
        bb = lane_and(lane_xor(bb, bb_minus), mask);
        if reverse { lane_byte_reverse(bb) } else { bb }
    }

    fn bishop_attacks_four_ray_oracle(sq: Square, occ: Bitboard) -> Bitboard {
        let s = sq.index() as usize;
        let mut r = [0u64; 2];
        for k in 0..4 {
            let step = Bitboard::from_raw(RAY_BITS[BISHOP_DIAG_DIRS[k]][s]).p;
            let mask = if BISHOP_DIAG_REV[k] {
                cbyte_reverse(step)
            } else {
                step
            };
            r = lane_or(r, diag_ray_oracle(mask, occ.p, BISHOP_DIAG_REV[k]));
        }
        Bitboard { p: r }
    }

    /// `bishopEffect` evaluated purely through the scalar `u64[4]` twins. On an
    /// AVX2 host this is the only exercise the fallback ops get.
    fn bishop_attacks_scalar256_oracle(sq: Square, occ: Bitboard) -> Bitboard {
        let s = sq.index() as usize;
        let mask_lo = QUGIY_BISHOP_MASK[s][0].p;
        let mask_hi = QUGIY_BISHOP_MASK[s][1].p;

        let occ2 = bb256_broadcast_scalar(occ.p);
        let rocc2 = bb256_broadcast_scalar(cbyte_reverse(occ.p));

        let (hi, lo) = bb256_unpack_scalar(rocc2, occ2);
        let hi = bb256_and_scalar(hi, mask_hi);
        let lo = bb256_and_scalar(lo, mask_lo);

        let (t1, t0) = bb256_pair_decrement_scalar(hi, lo);
        let t1 = bb256_and_scalar(bb256_xor_scalar(t1, hi), mask_hi);
        let t0 = bb256_and_scalar(bb256_xor_scalar(t0, lo), mask_lo);

        let (hi2, lo2) = bb256_unpack_scalar(t1, t0);
        Bitboard {
            p: bb256_merge_scalar(bb256_or_scalar(bb256_byte_reverse_scalar(hi2), lo2)),
        }
    }

    /// Against both bishop oracles, over the structural corpus plus 100k
    /// deterministic pseudo-random occupancies.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn bishop_matches_four_ray_and_scalar256_oracles() {
        let mut state: u128 = 0xF00D_CAFE_1234_5678_9ABC_DEF0_0FED_CBA9;
        let mut rng = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state & BOARD_MASK
        };
        let mut occs = sample_occupancies();
        for _ in 0..100_000 {
            occs.push(rng());
        }
        for occ in occs {
            let occ_bb = Bitboard::from_raw(occ);
            for from in all_squares() {
                let prod = bishop_attacks(from, occ_bb);
                assert_eq!(
                    prod,
                    bishop_attacks_four_ray_oracle(from, occ_bb),
                    "bishop vs four-ray oracle from {from:?} occ {occ:#x}",
                );
                assert_eq!(
                    prod,
                    bishop_attacks_scalar256_oracle(from, occ_bb),
                    "bishop vs scalar256 oracle from {from:?} occ {occ:#x}",
                );
                // The king ring is independent of `occ`.
                assert_eq!(
                    horse_attacks(from, occ_bb),
                    prod | KING_ATTACKS[0][from.index() as usize],
                    "horse from {from:?} occ {occ:#x}",
                );
            }
        }
    }

    #[test]
    fn empty_and_single_square() {
        assert!(Bitboard::empty().is_empty());
        assert_eq!(Bitboard::empty().popcount(), 0);
        for sq in all_squares() {
            let bb = Bitboard::from_square(sq);
            assert!(!bb.is_empty());
            assert_eq!(bb.popcount(), 1);
            assert!(bb.test(sq));
            assert_eq!(bb.squares().next(), Some(sq));
        }
    }

    #[test]
    fn set_and_clear_round_trip() {
        let mut bb = Bitboard::empty();
        let a = Square::new(3, 5).unwrap();
        let b = Square::new(8, 0).unwrap();
        bb.set(a);
        bb.set(b);
        assert!(bb.test(a) && bb.test(b));
        assert_eq!(bb.popcount(), 2);
        bb.clear(a);
        assert!(!bb.test(a) && bb.test(b));
        assert_eq!(bb.popcount(), 1);
    }

    #[test]
    fn from_raw_masks_high_bits() {
        let bb = Bitboard::from_raw(u128::MAX);
        assert_eq!(bb.raw(), BOARD_MASK);
        assert_eq!(bb.popcount(), Square::COUNT as u32);
    }

    #[test]
    fn bitwise_ops() {
        let a = Bitboard::from_square(Square::new(0, 0).unwrap());
        let b = Bitboard::from_square(Square::new(1, 1).unwrap());
        assert_eq!((a | b).popcount(), 2);
        assert_eq!((a & b).popcount(), 0);
        assert_eq!((a ^ b).popcount(), 2);
        assert_eq!(((a | b) & a), a);
        let full = !Bitboard::empty();
        assert_eq!(full.raw(), BOARD_MASK);
        assert_eq!((!a).popcount(), Square::COUNT as u32 - 1);
        assert!(!(!a).test(Square::new(0, 0).unwrap()));

        let mut c = a;
        c |= b;
        c &= b;
        c ^= b;
        assert!(c.is_empty());
    }

    #[test]
    fn iterator_is_ascending_index_order() {
        let squares = [
            Square::new(8, 8).unwrap(),
            Square::new(0, 0).unwrap(),
            Square::new(4, 3).unwrap(),
            Square::new(1, 7).unwrap(),
        ];
        let mut bb = Bitboard::empty();
        for &sq in &squares {
            bb.set(sq);
        }
        let got: Vec<u8> = bb.squares().map(|s| s.index()).collect();
        let mut want: Vec<u8> = squares.iter().map(|s| s.index()).collect();
        want.sort_unstable();
        assert_eq!(got, want);
        assert_eq!(bb.squares().len(), squares.len());
    }

    /// The step-only attack set of `piece` on `from`, via the movement data and
    /// the `step_signed` walk.
    fn step_walk(piece: Piece, from: Square) -> u128 {
        let dr_sign = dr_sign_for(piece.color);
        let (steps, _slides) = movement(piece);
        let mut set = 0u128;
        for &(df, dr) in steps {
            if let Some(to) = step_signed(from, df, dr * dr_sign) {
                set |= 1u128 << to.index();
            }
        }
        set
    }

    #[test]
    fn step_tables_equal_movement_walk() {
        type StepFn = fn(Color, Square) -> Bitboard;
        let cases: [(PieceKind, StepFn); 5] = [
            (PieceKind::Pawn, pawn_attacks),
            (PieceKind::Knight, knight_attacks),
            (PieceKind::Silver, silver_attacks),
            (PieceKind::Gold, gold_attacks),
            (PieceKind::King, king_attacks),
        ];
        for (kind, table) in cases {
            for color in [Color::Black, Color::White] {
                let piece = Piece::new(kind, color);
                for from in all_squares() {
                    assert_eq!(
                        table(color, from).raw(),
                        step_walk(piece, from),
                        "{kind:?} {color:?} from {from:?}",
                    );
                }
            }
        }
    }

    #[test]
    fn gold_table_serves_promoted_minors() {
        // The four promoted minors move as gold.
        for color in [Color::Black, Color::White] {
            for kind in [
                PieceKind::Pawn,
                PieceKind::Lance,
                PieceKind::Knight,
                PieceKind::Silver,
            ] {
                let promoted = Piece::promoted(kind, color).unwrap();
                for from in all_squares() {
                    assert_eq!(
                        gold_attacks(color, from).raw(),
                        step_walk(promoted, from),
                        "+{kind:?} {color:?} from {from:?}",
                    );
                }
            }
        }
    }

    #[test]
    fn ray_tables_equal_step_walk_to_edge() {
        for (dir, &(df, dr)) in DIRECTIONS.iter().enumerate() {
            for from in all_squares() {
                let mut want = 0u128;
                let mut cur = from;
                while let Some(next) = step_signed(cur, df, dr) {
                    want |= 1u128 << next.index();
                    cur = next;
                }
                assert_eq!(
                    ray(dir, from).raw(),
                    want,
                    "dir {dir} ({df},{dr}) from {from:?}",
                );
            }
        }
    }

    /// The arithmetic form of `ray_dir`, against which the table-backed
    /// [`ray_dir`] and [`between`] are pinned.
    fn ray_dir_reference(king: Square, sq: Square) -> Option<(i8, i8)> {
        let df = sq.file() as i8 - king.file() as i8;
        let dr = sq.rank() as i8 - king.rank() as i8;
        if df == 0 && dr == 0 {
            None
        } else if df == 0 {
            Some((0, dr.signum()))
        } else if dr == 0 {
            Some((df.signum(), 0))
        } else if df.abs() == dr.abs() {
            Some((df.signum(), dr.signum()))
        } else {
            None
        }
    }

    /// The scalar `between_set`, a `step_signed` walk.
    fn between_reference(a: Square, b: Square) -> u128 {
        let Some((df, dr)) = ray_dir_reference(a, b) else {
            return 0;
        };
        let mut set = 0u128;
        let mut cur = a;
        loop {
            cur = match step_signed(cur, df, dr) {
                Some(s) => s,
                None => break,
            };
            if cur == b {
                break;
            }
            set |= 1u128 << cur.index();
        }
        set
    }

    fn aligned_reference(king: Square, s1: Square, s2: Square) -> bool {
        match (ray_dir_reference(king, s1), ray_dir_reference(king, s2)) {
            (Some(a), Some(b)) => a == b,
            _ => false,
        }
    }

    fn aligned_table(king: Square, s1: Square, s2: Square) -> bool {
        match (ray_dir(king, s1), ray_dir(king, s2)) {
            (Some(a), Some(b)) => a == b,
            _ => false,
        }
    }

    #[test]
    fn ray_dir_equals_reference_all_pairs() {
        for a in all_squares() {
            for b in all_squares() {
                assert_eq!(
                    ray_dir(a, b),
                    ray_dir_reference(a, b),
                    "ray_dir({a:?}, {b:?})",
                );
            }
        }
    }

    #[test]
    fn between_equals_reference_all_pairs() {
        for a in all_squares() {
            for b in all_squares() {
                assert_eq!(
                    between(a, b).raw(),
                    between_reference(a, b),
                    "between({a:?}, {b:?})",
                );
            }
        }
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn aligned_equals_reference_all_triples() {
        // 81^3 ≈ 531k combinations, so every king/anchor triple is covered.
        for king in all_squares() {
            for s1 in all_squares() {
                for s2 in all_squares() {
                    assert_eq!(
                        aligned_table(king, s1, s2),
                        aligned_reference(king, s1, s2),
                        "aligned({king:?}, {s1:?}, {s2:?})",
                    );
                }
            }
        }
    }

    #[test]
    fn ray_and_dir_of_are_consistent() {
        for a in all_squares() {
            for (dir, &delta) in DIRECTIONS.iter().enumerate() {
                for b in ray(dir, a).squares() {
                    assert_eq!(
                        ray_dir(a, b),
                        Some(delta),
                        "ray dir {dir} from {a:?} contains {b:?}",
                    );
                }
            }
        }
    }

    #[test]
    fn file_and_rank_masks() {
        for file in 0..Square::FILES {
            let m = file_mask(file);
            assert_eq!(m.popcount(), Square::RANKS as u32);
            for sq in m.squares() {
                assert_eq!(sq.file(), file);
            }
        }
        for rank in 0..Square::RANKS {
            let m = rank_mask(rank);
            assert_eq!(m.popcount(), Square::FILES as u32);
            for sq in m.squares() {
                assert_eq!(sq.rank(), rank);
            }
        }
        let mut u = Bitboard::empty();
        for file in 0..Square::FILES {
            u |= file_mask(file);
        }
        assert_eq!(u.popcount(), Square::COUNT as u32);
    }

    /// The scalar form of an occupancy-limited slide: walk each direction from
    /// `from`, including the first occupied square, then stop.
    fn slider_walk(dirs: &[(i8, i8)], from: Square, occ: u128) -> u128 {
        let mut set = 0u128;
        for &(df, dr) in dirs {
            let mut cur = from;
            while let Some(next) = step_signed(cur, df, dr) {
                set |= 1u128 << next.index();
                if occ & (1u128 << next.index()) != 0 {
                    break;
                }
                cur = next;
            }
        }
        set
    }

    fn king_ring_bits(from: Square) -> u128 {
        let mut set = 0u128;
        for &(df, dr) in KING_STEPS {
            if let Some(to) = step_signed(from, df, dr) {
                set |= 1u128 << to.index();
            }
        }
        set
    }

    /// A deterministic stream of board-masked occupancy patterns: structural
    /// cases plus xorshift noise.
    fn sample_occupancies() -> Vec<u128> {
        let mut out = vec![0u128, BOARD_MASK];
        // Each single-square occupancy, for the nearest-blocker edge cases.
        for i in 0..N {
            out.push(1u128 << i);
        }
        // Xorshift noise, masked to the board.
        let mut state: u128 = 0x9E37_79B9_7F4A_7C15_1234_5678_9ABC_DEF1;
        for _ in 0..64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            out.push(state & BOARD_MASK);
        }
        out
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn slider_queries_equal_scalar_walk() {
        let bishop_dirs = [(1, -1), (-1, -1), (1, 1), (-1, 1)];
        let rook_dirs = [(0, -1), (0, 1), (1, 0), (-1, 0)];
        for occ in sample_occupancies() {
            let occ_bb = Bitboard::from_raw(occ);
            for from in all_squares() {
                assert_eq!(
                    lance_attacks(Color::Black, from, occ_bb).raw(),
                    slider_walk(&[(0, -1)], from, occ),
                    "black lance from {from:?} occ {occ:#x}",
                );
                assert_eq!(
                    lance_attacks(Color::White, from, occ_bb).raw(),
                    slider_walk(&[(0, 1)], from, occ),
                    "white lance from {from:?} occ {occ:#x}",
                );
                let bishop = slider_walk(&bishop_dirs, from, occ);
                let rook = slider_walk(&rook_dirs, from, occ);
                assert_eq!(
                    bishop_attacks(from, occ_bb).raw(),
                    bishop,
                    "bishop from {from:?} occ {occ:#x}",
                );
                assert_eq!(
                    rook_attacks(from, occ_bb).raw(),
                    rook,
                    "rook from {from:?} occ {occ:#x}",
                );
                assert_eq!(
                    horse_attacks(from, occ_bb).raw(),
                    bishop | king_ring_bits(from),
                    "horse from {from:?} occ {occ:#x}",
                );
                assert_eq!(
                    dragon_attacks(from, occ_bb).raw(),
                    rook | king_ring_bits(from),
                    "dragon from {from:?} occ {occ:#x}",
                );
            }
        }
    }

    #[test]
    fn promotion_zone_matches_movegen_predicate() {
        for color in [Color::Black, Color::White] {
            let zone = promotion_zone(color);
            for sq in all_squares() {
                assert_eq!(
                    zone.test(sq),
                    crate::movegen::is_in_promotion_zone(sq, color),
                    "promotion_zone {color:?} at {sq:?}",
                );
            }
        }
    }
}

/// A single-`u128` twin of [`Bitboard`], carrying the whole op surface, against
/// which the two-lane production representation is pinned.
#[cfg(test)]
mod twin {
    use super::*;

    const MASK: u128 = (1u128 << N) - 1;

    #[derive(Clone, Copy, PartialEq, Eq, Default, Hash, Debug)]
    pub(super) struct Twin(pub u128);

    #[allow(dead_code)]
    impl Twin {
        pub const EMPTY: Twin = Twin(0);
        pub const FULL: Twin = Twin(MASK);
        pub fn empty() -> Twin {
            Twin(0)
        }
        pub fn from_square(sq: Square) -> Twin {
            Twin(1u128 << sq.index())
        }
        pub fn from_raw(bits: u128) -> Twin {
            Twin(bits & MASK)
        }
        pub fn raw(self) -> u128 {
            self.0
        }
        pub fn is_empty(self) -> bool {
            self.0 == 0
        }
        pub fn test(self, sq: Square) -> bool {
            self.0 & (1u128 << sq.index()) != 0
        }
        pub fn set(&mut self, sq: Square) {
            self.0 |= 1u128 << sq.index();
        }
        pub fn clear(&mut self, sq: Square) {
            self.0 &= !(1u128 << sq.index());
        }
        pub fn popcount(self) -> u32 {
            self.0.count_ones()
        }
        pub fn squares(self) -> TwinIter {
            TwinIter(self.0)
        }
        pub fn single(index: usize) -> Twin {
            Twin((1u128 << index) & MASK)
        }
        pub fn or(self, other: Twin) -> Twin {
            Twin(self.0 | other.0)
        }
        pub fn without_index(self, index: usize) -> Twin {
            Twin(self.0 & !(1u128 << index))
        }
        pub fn contains_index(self, index: usize) -> bool {
            self.0 & (1u128 << index) != 0
        }
        pub fn lowest_index(self) -> u32 {
            self.0.trailing_zeros()
        }
    }

    impl core::ops::BitOr for Twin {
        type Output = Twin;
        fn bitor(self, rhs: Twin) -> Twin {
            Twin(self.0 | rhs.0)
        }
    }
    impl core::ops::BitAnd for Twin {
        type Output = Twin;
        fn bitand(self, rhs: Twin) -> Twin {
            Twin(self.0 & rhs.0)
        }
    }
    impl core::ops::BitXor for Twin {
        type Output = Twin;
        fn bitxor(self, rhs: Twin) -> Twin {
            Twin(self.0 ^ rhs.0)
        }
    }
    impl core::ops::Not for Twin {
        type Output = Twin;
        fn not(self) -> Twin {
            Twin(!self.0 & MASK)
        }
    }
    impl core::ops::BitOrAssign for Twin {
        fn bitor_assign(&mut self, rhs: Twin) {
            self.0 |= rhs.0;
        }
    }
    impl core::ops::BitAndAssign for Twin {
        fn bitand_assign(&mut self, rhs: Twin) {
            self.0 &= rhs.0;
        }
    }
    impl core::ops::BitXorAssign for Twin {
        fn bitxor_assign(&mut self, rhs: Twin) {
            self.0 ^= rhs.0;
        }
    }

    pub(super) struct TwinIter(u128);
    impl Iterator for TwinIter {
        type Item = Square;
        fn next(&mut self) -> Option<Square> {
            let i = self.0.lowest_one()?;
            self.0 &= self.0 - 1;
            Some(Square::from_index(i as u8).unwrap())
        }
        fn size_hint(&self) -> (usize, Option<usize>) {
            let n = self.0.count_ones() as usize;
            (n, Some(n))
        }
    }
    impl ExactSizeIterator for TwinIter {}

    fn sq(i: usize) -> Square {
        Square::from_index(i as u8).unwrap()
    }

    fn assert_unary_same(bb: Bitboard, tw: Twin) {
        assert_eq!(bb.raw(), tw.raw(), "raw");
        assert_eq!(bb.is_empty(), tw.is_empty(), "is_empty");
        assert_eq!(bb.popcount(), tw.popcount(), "popcount");
        assert_eq!(bb.lowest_index(), tw.lowest_index(), "lowest_index");
        for i in 0..N {
            assert_eq!(bb.test(sq(i)), tw.test(sq(i)), "test {i}");
            assert_eq!(bb.contains_index(i), tw.contains_index(i), "contains {i}");
        }
        // Identical values in identical ascending order.
        let got: Vec<u8> = bb.squares().map(|s| s.index()).collect();
        let want: Vec<u8> = tw.squares().map(|s| s.index()).collect();
        assert_eq!(got, want, "squares order/values for {:#x}", tw.raw());
        assert_eq!(bb.squares().len(), tw.squares().len(), "ExactSize len");
        assert_eq!((!bb).raw(), (!tw).raw(), "not");
    }

    /// A deterministic corpus of contiguous 81-bit patterns: EMPTY, FULL, every
    /// single square, the lane-boundary squares, file and rank masks, and
    /// fixed-seed xorshift noise.
    fn corpus() -> Vec<u128> {
        let mut out = vec![0u128, MASK];
        for i in 0..N {
            out.push(1u128 << i);
        }
        // 62 is the last lane-0 square, 63 the first lane-1 one, 80 the last.
        for &i in &[62usize, 63, 80] {
            out.push(1u128 << i);
            out.push(MASK & !(1u128 << i));
        }
        for f in 0..FILES {
            out.push(file_mask(f as u8).raw());
        }
        for r in 0..RANKS {
            out.push(rank_mask(r as u8).raw());
        }
        let mut state: u128 = 0x1234_5678_9ABC_DEF0_0FED_CBA9_8765_4321;
        for _ in 0..512 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            out.push(state & MASK);
        }
        out
    }

    #[test]
    fn constructors_and_helpers_match_twin() {
        assert_unary_same(Bitboard::EMPTY, Twin::EMPTY);
        assert_unary_same(Bitboard::empty(), Twin::empty());
        assert_unary_same(Bitboard::FULL, Twin::FULL);
        for i in 0..N {
            assert_unary_same(Bitboard::from_square(sq(i)), Twin::from_square(sq(i)));
            assert_unary_same(Bitboard::single(i), Twin::single(i));
        }
        // `single` past the board degenerates to EMPTY.
        for i in N..(N + 4) {
            assert_unary_same(Bitboard::single(i), Twin::single(i));
        }
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn every_value_matches_twin() {
        for bits in corpus() {
            assert_unary_same(Bitboard::from_raw(bits), Twin::from_raw(bits));
        }
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn set_clear_and_without_index_match_twin() {
        for bits in corpus() {
            for i in 0..N {
                let mut bb = Bitboard::from_raw(bits);
                let mut tw = Twin::from_raw(bits);
                bb.set(sq(i));
                tw.set(sq(i));
                assert_unary_same(bb, tw);
                bb.clear(sq(i));
                tw.clear(sq(i));
                assert_unary_same(bb, tw);
                assert_unary_same(
                    Bitboard::from_raw(bits).without_index(i),
                    Twin::from_raw(bits).without_index(i),
                );
            }
        }
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn binary_ops_match_twin() {
        let c = corpus();
        // A fixed subset paired against the whole corpus keeps the cross
        // product bounded while still covering every structural pattern.
        let anchors = [0usize, 1, 2, 65, 83, c.len() - 1];
        for &ai in &anchors {
            let a_bits = c[ai];
            for &b_bits in &c {
                let (abb, bbb) = (Bitboard::from_raw(a_bits), Bitboard::from_raw(b_bits));
                let (atw, btw) = (Twin::from_raw(a_bits), Twin::from_raw(b_bits));
                assert_unary_same(abb | bbb, atw | btw);
                assert_unary_same(abb & bbb, atw & btw);
                assert_unary_same(abb ^ bbb, atw ^ btw);
                assert_unary_same(abb.or(bbb), atw.or(btw));
                let (mut abo, mut ato) = (abb, atw);
                abo |= bbb;
                ato |= btw;
                assert_unary_same(abo, ato);
                let (mut aba, mut ata) = (abb, atw);
                aba &= bbb;
                ata &= btw;
                assert_unary_same(aba, ata);
                let (mut abx, mut atx) = (abb, atw);
                abx ^= bbb;
                atx ^= btw;
                assert_unary_same(abx, atx);
            }
        }
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn chained_compositions_match_twin() {
        let c = corpus();
        for w in c.windows(3) {
            let (x, y, z) = (w[0], w[1], w[2]);
            let (xb, yb, zb) = (
                Bitboard::from_raw(x),
                Bitboard::from_raw(y),
                Bitboard::from_raw(z),
            );
            let (xt, yt, zt) = (Twin::from_raw(x), Twin::from_raw(y), Twin::from_raw(z));
            assert_unary_same((xb | yb) & !zb, (xt | yt) & !zt);
            assert_unary_same(!(xb & yb) ^ zb, !(xt & yt) ^ zt);
            assert_unary_same((xb ^ yb) | (yb & zb), (xt ^ yt) | (yt & zt));
        }
    }
}
