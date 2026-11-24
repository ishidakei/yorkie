//! AVX-512 forms of the bishop attack and the rook's rank attack, compiled when
//! the build enables `avx512f`, `avx512bw` and `avx512vl`. They return the same
//! sets as the 64-bit bishop pass and the 128-bit rank pass they stand in for.
//!
//! Both rest on one identity. A ray that runs toward higher bits through a
//! two-word value `[lo, hi]`, masked to the ray, is cut at its first blocker by
//! `(o ^ (o - 1)) & mask` with a 128-bit `o - 1`. When `lo` is not zero the
//! borrow stops inside `lo` and the high word of the result is zero; when `lo`
//! is zero, the low word of the result is `mask`'s and the high word is the
//! same expression on `hi` alone. So the borrow never has to cross words: each
//! word takes `o - 1` on its own, and a high word is kept only where the low
//! word beside it is zero. A ray that runs toward lower bits takes the same
//! route on the byte-reversed occupancy, whose mask is stored byte-reversed.

use core::arch::x86_64::{
    __m128i, __m512i, __mmask8, _mm_add_epi64, _mm_and_si128, _mm_cmpeq_epi64_mask, _mm_or_si128,
    _mm_set1_epi64x, _mm_setzero_si128, _mm_shuffle_epi8, _mm_xor_si128, _mm256_castsi256_si128,
    _mm256_extracti128_si256, _mm256_or_si256, _mm512_add_epi64, _mm512_and_si512,
    _mm512_broadcast_i32x4, _mm512_castsi512_si256, _mm512_cmpeq_epi64_mask,
    _mm512_extracti64x4_epi64, _mm512_set1_epi64, _mm512_setzero_si512, _mm512_shuffle_epi8,
};

use super::{BISHOP_RAYS, Bitboard, QUGIY_ROOK_MASK};

// `(a ^ b) & c` in the 64-bit words `k` enables, zero in the others. The
// compiler does not fuse a zero-masked `xor` and an `and` into one
// `vpternlogq`, so the build states the ternary-logic instruction directly.
// Miri cannot interpret that intrinsic and runs the two-step form, which
// computes the same words.
//
// SAFETY (every function here): compiled only when the build enables
// `avx512f` and `avx512vl`, which these intrinsics require.
std::cfg_select! {
    miri => {
        fn masked_xor_and_512(k: __mmask8, a: __m512i, b: __m512i, c: __m512i) -> __m512i {
            use core::arch::x86_64::_mm512_maskz_xor_epi64;
            unsafe { _mm512_and_si512(_mm512_maskz_xor_epi64(k, a, b), c) }
        }

        fn masked_xor_and_128(k: __mmask8, a: __m128i, b: __m128i, c: __m128i) -> __m128i {
            use core::arch::x86_64::_mm_maskz_xor_epi64;
            unsafe { _mm_and_si128(_mm_maskz_xor_epi64(k, a, b), c) }
        }
    }
    _ => {
        /// `vpternlogq` truth table for `(a ^ b) & c`: set at the inputs
        /// `(a, b, c) = (1, 0, 1)` and `(0, 1, 1)`, bits 5 and 3.
        const XOR_AND: i32 = 0x28;

        fn masked_xor_and_512(k: __mmask8, a: __m512i, b: __m512i, c: __m512i) -> __m512i {
            use core::arch::x86_64::_mm512_maskz_ternarylogic_epi64;
            unsafe { _mm512_maskz_ternarylogic_epi64::<XOR_AND>(k, a, b, c) }
        }

        fn masked_xor_and_128(k: __mmask8, a: __m128i, b: __m128i, c: __m128i) -> __m128i {
            use core::arch::x86_64::_mm_maskz_ternarylogic_epi64;
            unsafe { _mm_maskz_ternarylogic_epi64::<XOR_AND>(k, a, b, c) }
        }
    }
}

/// Every low word of a 512-bit register of two-word lanes.
const LOW_WORDS: __mmask8 = 0x55;

/// `vpshufb` control that leaves 128-bit lanes 0 and 1 as they are and
/// byte-reverses lanes 2 and 3. It is its own inverse.
const REVERSE_UPPER_LANES: [u8; 64] = {
    let mut t = [0u8; 64];
    let mut i = 0;
    while i < 64 {
        let j = (i % 16) as u8;
        t[i] = if i < 32 { j } else { 15 - j };
        i += 1;
    }
    t
};

/// Squares a bishop on square index `s` attacks under `occ`.
///
/// SAFETY: compiled only when the build enables `avx512f`, `avx512bw` and
/// `avx512vl`, which every intrinsic here requires; the `transmute`s only
/// bit-cast between `[u64; N]` / `[u8; N]` and the equally-sized vector types.
pub(super) fn bishop_attacks(s: usize, occ: Bitboard) -> Bitboard {
    let rays = BISHOP_RAYS[s].p;
    unsafe {
        let ctl: __m512i = core::mem::transmute(REVERSE_UPPER_LANES);
        let mask: __m512i = core::mem::transmute(rays);
        let o: __m128i = core::mem::transmute(occ.p);
        let o = _mm512_shuffle_epi8(_mm512_broadcast_i32x4(o), ctl);
        let o = _mm512_and_si512(o, mask);
        let o_minus = _mm512_add_epi64(o, _mm512_set1_epi64(-1));
        let zero = _mm512_cmpeq_epi64_mask(o, _mm512_setzero_si512());
        let keep = (zero << 1) | LOW_WORDS;
        let r = masked_xor_and_512(keep, o, o_minus, mask);
        let r = _mm512_shuffle_epi8(r, ctl);
        let r = _mm256_or_si256(_mm512_castsi512_si256(r), _mm512_extracti64x4_epi64::<1>(r));
        let r = _mm_or_si128(_mm256_castsi256_si128(r), _mm256_extracti128_si256::<1>(r));
        Bitboard {
            p: core::mem::transmute::<__m128i, [u64; 2]>(r),
        }
    }
}

/// Builds a 16-byte `vpshufb` control: output word 0 takes input word `w0`,
/// output word 1 input word `w1`, each byte-reversed where its flag is set.
const fn word_shuffle(w0: usize, rev0: bool, w1: usize, rev1: bool) -> [u8; 16] {
    let mut t = [0u8; 16];
    let mut k = 0;
    while k < 8 {
        t[k] = (8 * w0 + if rev0 { 7 - k } else { k }) as u8;
        t[8 + k] = (8 * w1 + if rev1 { 7 - k } else { k }) as u8;
        k += 1;
    }
    t
}

/// `[occ.p[0], rev(occ).p[0]]`, the low words of the toward-file-8 ray and of
/// the byte-reversed toward-file-0 ray — the layout of the rook rank mask's
/// `[0]` entry. Its own inverse.
const RANK_LO: [u8; 16] = word_shuffle(0, false, 1, true);
/// `[occ.p[1], rev(occ).p[1]]`, the matching high words (the mask's `[1]`).
const RANK_HI: [u8; 16] = word_shuffle(1, false, 0, true);
/// The inverse of [`RANK_HI`].
const RANK_HI_BACK: [u8; 16] = word_shuffle(1, true, 0, false);

/// The rook's rank effect from square index `s` under `occ`.
///
/// SAFETY: as [`bishop_attacks`].
pub(super) fn rook_rank(s: usize, occ: Bitboard) -> Bitboard {
    let mask_lo = QUGIY_ROOK_MASK[s][0].p;
    let mask_hi = QUGIY_ROOK_MASK[s][1].p;
    unsafe {
        let to_lo: __m128i = core::mem::transmute(RANK_LO);
        let to_hi: __m128i = core::mem::transmute(RANK_HI);
        let hi_back: __m128i = core::mem::transmute(RANK_HI_BACK);
        let mask_lo: __m128i = core::mem::transmute(mask_lo);
        let mask_hi: __m128i = core::mem::transmute(mask_hi);
        let o: __m128i = core::mem::transmute(occ.p);
        let lo = _mm_and_si128(_mm_shuffle_epi8(o, to_lo), mask_lo);
        let hi = _mm_and_si128(_mm_shuffle_epi8(o, to_hi), mask_hi);
        let minus_one = _mm_set1_epi64x(-1);
        let lo_zero = _mm_cmpeq_epi64_mask(lo, _mm_setzero_si128());
        let lo = _mm_and_si128(_mm_xor_si128(lo, _mm_add_epi64(lo, minus_one)), mask_lo);
        let hi = masked_xor_and_128(lo_zero, hi, _mm_add_epi64(hi, minus_one), mask_hi);
        let r = _mm_or_si128(_mm_shuffle_epi8(lo, to_lo), _mm_shuffle_epi8(hi, hi_back));
        Bitboard {
            p: core::mem::transmute::<__m128i, [u64; 2]>(r),
        }
    }
}
