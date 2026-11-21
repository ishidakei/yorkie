//! AVX-512 (F + BW) feature-transformer kernels over packed weight columns.
//!
//! A packed column is half the bytes of a 16-bit one: each 32-lane tile is
//! 32 bytes, sign-extended to 16 bits and doubled in registers, and the few
//! lanes that are not exactly twice their byte are put right afterwards, one
//! scalar correction each. Every kernel is bit-identical to its
//! [`crate::simd::scalar_packed`] counterpart and, through it, to the 16-bit
//! kernels over the unpacked weights.
//!
//! Compiled on every `x86_64` target for the same reason as
//! [`crate::simd::avx512`]: the parity tests below hold it against the scalar
//! baseline on any AVX-512-capable host.

use std::arch::x86_64::{
    __m256i, __m512i, _mm256_loadu_si256, _mm512_add_epi16, _mm512_cvtepi8_epi16,
    _mm512_loadu_si512, _mm512_slli_epi16, _mm512_storeu_si512, _mm512_sub_epi16,
};

use crate::features::FeatureIndex;
use crate::types::{HIDDEN_SIZE, PackedFt};

const NUM_CHUNKS: usize = HIDDEN_SIZE / 32;
const LANES: usize = 32;

/// One pass over the half: `dst = src + sum(added) - sum(removed)`, the columns
/// read as their bytes doubled. The corrections are the caller's to apply
/// afterwards.
///
/// # Safety
/// `src` and `dst` must each address `HIDDEN_SIZE` `i16`s, readable for `src`
/// and writable for `dst`. `dst` may be `src`: every chunk is read before it is
/// written.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
unsafe fn pass<const A: usize, const S: usize>(
    src: *const i16,
    dst: *mut i16,
    packed: PackedFt<'_>,
    added: [FeatureIndex; A],
    removed: [FeatureIndex; S],
) {
    let add_ptrs = added.map(|idx| packed.bases(idx).as_ptr());
    let sub_ptrs = removed.map(|idx| packed.bases(idx).as_ptr());
    for chunk in 0..NUM_CHUNKS {
        let offset = chunk * LANES;
        // SAFETY: `chunk * LANES + LANES <= HIDDEN_SIZE`, the length of every
        // column's bytes and, by the caller's contract, of `src` and `dst`;
        // the unaligned loads and stores impose no alignment.
        unsafe {
            let mut acc = _mm512_loadu_si512(src.add(offset).cast::<__m512i>());
            for col in add_ptrs {
                let w = _mm256_loadu_si256(col.add(offset).cast::<__m256i>());
                acc = _mm512_add_epi16(acc, _mm512_slli_epi16::<1>(_mm512_cvtepi8_epi16(w)));
            }
            for col in sub_ptrs {
                let w = _mm256_loadu_si256(col.add(offset).cast::<__m256i>());
                acc = _mm512_sub_epi16(acc, _mm512_slli_epi16::<1>(_mm512_cvtepi8_epi16(w)));
            }
            _mm512_storeu_si512(dst.add(offset).cast::<__m512i>(), acc);
        }
    }
}

/// [`pass`] over `out` in place, corrections included.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn in_place<const A: usize, const S: usize>(
    out: &mut [i16; HIDDEN_SIZE],
    packed: PackedFt<'_>,
    added: [FeatureIndex; A],
    removed: [FeatureIndex; S],
) {
    let ptr = out.as_mut_ptr();
    // SAFETY: `out` is `HIDDEN_SIZE` lanes, read and written through the one
    // pointer.
    unsafe { pass::<A, S>(ptr, ptr, packed, added, removed) };
    for idx in added {
        packed.correct::<false>(out, idx);
    }
    for idx in removed {
        packed.correct::<true>(out, idx);
    }
}

/// [`pass`] reading `src` and writing `dst`, corrections included.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn from<const A: usize, const S: usize>(
    src: &[i16; HIDDEN_SIZE],
    dst: &mut [i16; HIDDEN_SIZE],
    packed: PackedFt<'_>,
    added: [FeatureIndex; A],
    removed: [FeatureIndex; S],
) {
    // SAFETY: `src` and `dst` are `HIDDEN_SIZE` lanes each and distinct.
    unsafe { pass::<A, S>(src.as_ptr(), dst.as_mut_ptr(), packed, added, removed) };
    for idx in added {
        packed.correct::<false>(dst, idx);
    }
    for idx in removed {
        packed.correct::<true>(dst, idx);
    }
}

/// Chunks of the half held in registers at once by the list kernels. The half
/// is [`NUM_CHUNKS`] chunks, which this divides; AVX-512 has 32 registers, and
/// a tile of 24 leaves room for the column operand.
const TILE: usize = 24;
const _: () = assert!(NUM_CHUNKS.is_multiple_of(TILE));

/// [`pass`] for lists of any length, with `TEE` every result chunk stored to
/// `tee` as well: the half goes through in tiles of [`TILE`] chunks, each
/// loaded once, every column applied to it in registers, and stored once. The
/// corrections are the caller's to apply.
///
/// # Safety
/// `src`, `dst` and (with `TEE`) `tee` must each address `HIDDEN_SIZE` `i16`s,
/// readable for `src` and writable for the other two. `dst` may be `src`:
/// every tile is read before it is written.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
unsafe fn tiled<const TEE: bool>(
    src: *const i16,
    dst: *mut i16,
    tee: *mut i16,
    packed: PackedFt<'_>,
    added: &[FeatureIndex],
    removed: &[FeatureIndex],
) {
    // Every column is checked against the bytes once here, so the tile loop
    // below can address each one by plain pointer arithmetic.
    for &idx in added.iter().chain(removed) {
        packed.bases(idx);
    }
    let base = packed.bases_ptr();
    for tile in 0..NUM_CHUNKS / TILE {
        let first = tile * TILE * LANES;
        // SAFETY: `first + TILE * LANES <= HIDDEN_SIZE`, the length of every
        // column's bytes, each of which was checked to lie inside the packed
        // weights above, and, by the caller's contract, of `src`, `dst` and
        // `tee`; the unaligned loads and stores impose no alignment.
        unsafe {
            let mut acc: [__m512i; TILE] =
                std::array::from_fn(|i| _mm512_loadu_si512(src.add(first + i * LANES).cast()));
            for &idx in removed {
                let col = base.add(idx as usize * HIDDEN_SIZE + first);
                for (i, a) in acc.iter_mut().enumerate() {
                    let w = _mm256_loadu_si256(col.add(i * LANES).cast::<__m256i>());
                    *a = _mm512_sub_epi16(*a, _mm512_slli_epi16::<1>(_mm512_cvtepi8_epi16(w)));
                }
            }
            for &idx in added {
                let col = base.add(idx as usize * HIDDEN_SIZE + first);
                for (i, a) in acc.iter_mut().enumerate() {
                    let w = _mm256_loadu_si256(col.add(i * LANES).cast::<__m256i>());
                    *a = _mm512_add_epi16(*a, _mm512_slli_epi16::<1>(_mm512_cvtepi8_epi16(w)));
                }
            }
            for (i, a) in acc.into_iter().enumerate() {
                _mm512_storeu_si512(dst.add(first + i * LANES).cast(), a);
                if TEE {
                    _mm512_storeu_si512(tee.add(first + i * LANES).cast(), a);
                }
            }
        }
    }
}

/// [`tiled`] over `out` in place, corrections included.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn tiled_in_place(
    out: &mut [i16; HIDDEN_SIZE],
    packed: PackedFt<'_>,
    added: &[FeatureIndex],
    removed: &[FeatureIndex],
) {
    let ptr = out.as_mut_ptr();
    // SAFETY: `out` is `HIDDEN_SIZE` lanes, read and written through the one
    // pointer; `tee` is never touched without `TEE`.
    unsafe { tiled::<false>(ptr, ptr, std::ptr::null_mut(), packed, added, removed) };
    correct_all(out, packed, added, removed);
}

/// Every correction of `added` and `removed`, applied to `out`.
#[inline]
fn correct_all(
    out: &mut [i16; HIDDEN_SIZE],
    packed: PackedFt<'_>,
    added: &[FeatureIndex],
    removed: &[FeatureIndex],
) {
    for &idx in added {
        packed.correct::<false>(out, idx);
    }
    for &idx in removed {
        packed.correct::<true>(out, idx);
    }
}

/// Add each active feature's column into `out`.
#[target_feature(enable = "avx512f,avx512bw")]
pub fn add_features(out: &mut [i16; HIDDEN_SIZE], packed: PackedFt<'_>, indices: &[FeatureIndex]) {
    tiled_in_place(out, packed, indices, &[]);
}

/// Subtract each feature's column from `out`.
#[target_feature(enable = "avx512f,avx512bw")]
pub fn sub_features(out: &mut [i16; HIDDEN_SIZE], packed: PackedFt<'_>, indices: &[FeatureIndex]) {
    tiled_in_place(out, packed, &[], indices);
}

/// Fused single-add / single-sub delta.
#[target_feature(enable = "avx512f,avx512bw")]
pub fn add_sub_features(
    out: &mut [i16; HIDDEN_SIZE],
    packed: PackedFt<'_>,
    added: FeatureIndex,
    removed: FeatureIndex,
) {
    in_place(out, packed, [added], [removed]);
}

/// Fused single-add / double-sub delta.
#[target_feature(enable = "avx512f,avx512bw")]
pub fn add_sub_sub_features(
    out: &mut [i16; HIDDEN_SIZE],
    packed: PackedFt<'_>,
    added: FeatureIndex,
    removed_a: FeatureIndex,
    removed_b: FeatureIndex,
) {
    in_place(out, packed, [added], [removed_a, removed_b]);
}

/// `dst = src + add - sub`, in one pass over the half.
#[target_feature(enable = "avx512f,avx512bw")]
pub fn add_sub_features_from(
    src: &[i16; HIDDEN_SIZE],
    dst: &mut [i16; HIDDEN_SIZE],
    packed: PackedFt<'_>,
    added: FeatureIndex,
    removed: FeatureIndex,
) {
    from(src, dst, packed, [added], [removed]);
}

/// `dst = src + add - sub_a - sub_b`, in one pass over the half.
#[target_feature(enable = "avx512f,avx512bw")]
pub fn add_sub_sub_features_from(
    src: &[i16; HIDDEN_SIZE],
    dst: &mut [i16; HIDDEN_SIZE],
    packed: PackedFt<'_>,
    added: FeatureIndex,
    removed_a: FeatureIndex,
    removed_b: FeatureIndex,
) {
    from(src, dst, packed, [added], [removed_a, removed_b]);
}

/// `dst = src - sum(removed) + sum(added)` for lists of any length, in one pass
/// over the half. Both lists empty is a plain copy.
#[target_feature(enable = "avx512f,avx512bw")]
pub fn sub_add_features_from(
    src: &[i16; HIDDEN_SIZE],
    dst: &mut [i16; HIDDEN_SIZE],
    packed: PackedFt<'_>,
    added: &[FeatureIndex],
    removed: &[FeatureIndex],
) {
    // SAFETY: `src` and `dst` are `HIDDEN_SIZE` lanes each and distinct; `tee`
    // is never touched without `TEE`.
    unsafe {
        tiled::<false>(
            src.as_ptr(),
            dst.as_mut_ptr(),
            std::ptr::null_mut(),
            packed,
            added,
            removed,
        )
    };
    correct_all(dst, packed, added, removed);
}

/// Apply `-sum(removed) + sum(added)` to `out` in place and leave the result in
/// `copy` as well, in one pass over the half that stores each tile to both.
/// The corrections go to `out` before that pass, so the copy takes them with
/// it: under wrapping arithmetic the order of the terms does not matter. Both
/// lists empty is a plain copy.
#[target_feature(enable = "avx512f,avx512bw")]
pub fn sub_add_features_tee(
    out: &mut [i16; HIDDEN_SIZE],
    copy: &mut [i16; HIDDEN_SIZE],
    packed: PackedFt<'_>,
    added: &[FeatureIndex],
    removed: &[FeatureIndex],
) {
    correct_all(out, packed, added, removed);
    let ptr = out.as_mut_ptr();
    // SAFETY: `out` and `copy` are `HIDDEN_SIZE` lanes each and distinct, `out`
    // read and written through the one pointer.
    unsafe { tiled::<true>(ptr, ptr, copy.as_mut_ptr(), packed, added, removed) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::avx512::tests::{COLUMNS, list_lengths};
    use crate::simd::scalar::tests::{DIFFS, Lanes};
    use crate::simd::scalar_packed;
    use crate::types::tests::PackedColumns;

    macro_rules! require_avx512bw {
        () => {
            if !(std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw"))
            {
                eprintln!("skipping AVX-512 parity test: avx512f/avx512bw unavailable");
                return;
            }
        };
    }

    #[test]
    fn every_kernel_matches_the_scalar_packed_baseline() {
        require_avx512bw!();
        let mut rng = Lanes(0x5851_f42d_4c95_7f2d);
        let columns = PackedColumns::from_wide(&rng.weights(8));
        let packed = columns.view();
        for &(added, removed) in DIFFS {
            let src = rng.half();

            let mut expected = src;
            scalar_packed::sub_features(&mut expected, packed, removed);
            scalar_packed::add_features(&mut expected, packed, added);

            let mut in_place = src;
            // SAFETY: guarded by `require_avx512bw!`.
            unsafe {
                sub_features(&mut in_place, packed, removed);
                add_features(&mut in_place, packed, added);
            }
            assert_eq!(in_place, expected, "in place {added:?} {removed:?}");

            let mut dst = rng.half();
            // SAFETY: guarded by `require_avx512bw!`.
            unsafe { sub_add_features_from(&src, &mut dst, packed, added, removed) };
            assert_eq!(dst, expected, "sub_add_features_from {added:?} {removed:?}");

            if let ([a], [r]) = (added, removed) {
                let mut out = src;
                // SAFETY: guarded by `require_avx512bw!`.
                unsafe { add_sub_features(&mut out, packed, *a, *r) };
                assert_eq!(out, expected, "add_sub_features {added:?} {removed:?}");
                let mut dst = rng.half();
                // SAFETY: guarded by `require_avx512bw!`.
                unsafe { add_sub_features_from(&src, &mut dst, packed, *a, *r) };
                assert_eq!(dst, expected, "add_sub_features_from {added:?} {removed:?}");
            }
            if let ([a], [r0, r1]) = (added, removed) {
                let mut out = src;
                // SAFETY: guarded by `require_avx512bw!`.
                unsafe { add_sub_sub_features(&mut out, packed, *a, *r0, *r1) };
                assert_eq!(out, expected, "add_sub_sub_features {added:?} {removed:?}");
                let mut dst = rng.half();
                // SAFETY: guarded by `require_avx512bw!`.
                unsafe { add_sub_sub_features_from(&src, &mut dst, packed, *a, *r0, *r1) };
                assert_eq!(
                    dst, expected,
                    "add_sub_sub_features_from {added:?} {removed:?}"
                );
            }

            let mut out = src;
            let mut copy = rng.half();
            // SAFETY: guarded by `require_avx512bw!`.
            unsafe { sub_add_features_tee(&mut out, &mut copy, packed, added, removed) };
            assert_eq!(out, expected, "tee (in place) {added:?} {removed:?}");
            assert_eq!(copy, expected, "tee (copy) {added:?} {removed:?}");
        }
    }

    #[test]
    fn list_kernels_match_the_scalar_packed_baseline_for_every_list_length() {
        require_avx512bw!();
        let mut rng = Lanes(0x9e37_79b9_7f4a_7c15);
        let wide = rng.weights(COLUMNS);
        let columns = PackedColumns::from_wide(&wide);
        let packed = columns.view();
        for (added, removed) in list_lengths(&mut rng) {
            let src = rng.half();
            let mut expected = src;
            scalar_packed::sub_features(&mut expected, packed, &removed);
            scalar_packed::add_features(&mut expected, packed, &added);
            let mut wide_expected = src;
            crate::simd::scalar::sub_features(&mut wide_expected, &wide, &removed);
            crate::simd::scalar::add_features(&mut wide_expected, &wide, &added);
            assert_eq!(
                expected, wide_expected,
                "packed baseline {added:?} {removed:?}"
            );

            let mut in_place = src;
            // SAFETY: guarded by `require_avx512bw!`.
            unsafe {
                sub_features(&mut in_place, packed, &removed);
                add_features(&mut in_place, packed, &added);
            }
            assert_eq!(in_place, expected, "in place {added:?} {removed:?}");

            let mut dst = rng.half();
            // SAFETY: guarded by `require_avx512bw!`.
            unsafe { sub_add_features_from(&src, &mut dst, packed, &added, &removed) };
            assert_eq!(dst, expected, "from {added:?} {removed:?}");

            let mut out = src;
            let mut copy = rng.half();
            // SAFETY: guarded by `require_avx512bw!`.
            unsafe { sub_add_features_tee(&mut out, &mut copy, packed, &added, &removed) };
            assert_eq!(out, expected, "tee (in place) {added:?} {removed:?}");
            assert_eq!(copy, expected, "tee (copy) {added:?} {removed:?}");
        }
    }

    #[test]
    fn a_packed_column_applies_as_the_16_bit_one_it_came_from() {
        require_avx512bw!();
        let mut rng = Lanes(0x2545_f491_4f6c_dd1d);
        let wide = rng.weights(8);
        let columns = PackedColumns::from_wide(&wide);
        assert!(
            columns.record_count() > 0,
            "uniform 16-bit lanes are mostly odd, so the corpus exercises corrections"
        );
        for &(added, removed) in DIFFS {
            let src = rng.half();
            let mut expected = src;
            crate::simd::scalar::sub_features(&mut expected, &wide, removed);
            crate::simd::scalar::add_features(&mut expected, &wide, added);

            let mut dst = rng.half();
            // SAFETY: guarded by `require_avx512bw!`.
            unsafe { sub_add_features_from(&src, &mut dst, columns.view(), added, removed) };
            assert_eq!(dst, expected, "{added:?} {removed:?}");
        }
    }
}
