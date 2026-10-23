//! AVX-512 (F + BW) feature-transformer accumulate/update kernels.
//!
//! Every kernel here is bit-identical to its [`crate::simd::scalar`]
//! counterpart.
//!
//! Unlike the reference, which `cfg`-gates the whole module on the target
//! features, this module is *compiled* on every `x86_64` target so the parity
//! tests below can hold it against the scalar baseline on any AVX-512-capable
//! host. `#[target_feature]` imposes no whole-binary requirement; it only makes
//! the compiler emit AVX-512 code for that one function. On a build that
//! selects the scalar path these kernels have no caller and the linker drops
//! them.

use std::arch::x86_64::{
    __m512i, _mm512_add_epi16, _mm512_loadu_si512, _mm512_storeu_si512, _mm512_sub_epi16,
};

use crate::features::FeatureIndex;
use crate::simd::ft_column;
use crate::types::HIDDEN_SIZE;

const NUM_CHUNKS: usize = HIDDEN_SIZE / 32;
const LANES: usize = 32;

/// Add each active feature's FT weight column into `out`.
#[target_feature(enable = "avx512f,avx512bw")]
pub fn add_features(out: &mut [i16; HIDDEN_SIZE], weights: &[i16], indices: &[FeatureIndex]) {
    let out_ptr = out.as_mut_ptr();
    for &idx in indices {
        let col_ptr = ft_column(weights, idx).as_ptr();
        for chunk in 0..NUM_CHUNKS {
            let offset = chunk * LANES;
            // SAFETY: `chunk * LANES + LANES <= HIDDEN_SIZE`, which is both
            // the length of both `out` and `col`; the unaligned 512-bit ops impose no
            // alignment requirement.
            unsafe {
                let o = _mm512_loadu_si512(out_ptr.add(offset).cast::<__m512i>());
                let w = _mm512_loadu_si512(col_ptr.add(offset).cast::<__m512i>());
                _mm512_storeu_si512(
                    out_ptr.add(offset).cast::<__m512i>(),
                    _mm512_add_epi16(o, w),
                );
            }
        }
    }
}

/// Subtract each feature's FT weight column from `out`.
#[target_feature(enable = "avx512f,avx512bw")]
pub fn sub_features(out: &mut [i16; HIDDEN_SIZE], weights: &[i16], indices: &[FeatureIndex]) {
    let out_ptr = out.as_mut_ptr();
    for &idx in indices {
        let col_ptr = ft_column(weights, idx).as_ptr();
        for chunk in 0..NUM_CHUNKS {
            let offset = chunk * LANES;
            // SAFETY: see `add_features`.
            unsafe {
                let o = _mm512_loadu_si512(out_ptr.add(offset).cast::<__m512i>());
                let w = _mm512_loadu_si512(col_ptr.add(offset).cast::<__m512i>());
                _mm512_storeu_si512(
                    out_ptr.add(offset).cast::<__m512i>(),
                    _mm512_sub_epi16(o, w),
                );
            }
        }
    }
}

/// Fused single-add / single-sub delta.
#[target_feature(enable = "avx512f,avx512bw")]
pub fn add_sub_features(
    out: &mut [i16; HIDDEN_SIZE],
    weights: &[i16],
    added: FeatureIndex,
    removed: FeatureIndex,
) {
    let out_ptr = out.as_mut_ptr();
    let add_ptr = ft_column(weights, added).as_ptr();
    let sub_ptr = ft_column(weights, removed).as_ptr();
    for chunk in 0..NUM_CHUNKS {
        let offset = chunk * LANES;
        // SAFETY: `chunk * LANES + LANES <= HIDDEN_SIZE`, the length of `out`
        // and of each column.
        unsafe {
            let acc = _mm512_loadu_si512(out_ptr.add(offset).cast::<__m512i>());
            let a = _mm512_loadu_si512(add_ptr.add(offset).cast::<__m512i>());
            let s = _mm512_loadu_si512(sub_ptr.add(offset).cast::<__m512i>());
            let acc = _mm512_sub_epi16(_mm512_add_epi16(acc, a), s);
            _mm512_storeu_si512(out_ptr.add(offset).cast::<__m512i>(), acc);
        }
    }
}

/// Fused single-add / double-sub delta.
#[target_feature(enable = "avx512f,avx512bw")]
pub fn add_sub_sub_features(
    out: &mut [i16; HIDDEN_SIZE],
    weights: &[i16],
    added: FeatureIndex,
    removed_a: FeatureIndex,
    removed_b: FeatureIndex,
) {
    let out_ptr = out.as_mut_ptr();
    let add_ptr = ft_column(weights, added).as_ptr();
    let sub_a_ptr = ft_column(weights, removed_a).as_ptr();
    let sub_b_ptr = ft_column(weights, removed_b).as_ptr();
    for chunk in 0..NUM_CHUNKS {
        let offset = chunk * LANES;
        // SAFETY: `chunk * LANES + LANES <= HIDDEN_SIZE`, the length of `out`
        // and of each column.
        unsafe {
            let acc = _mm512_loadu_si512(out_ptr.add(offset).cast::<__m512i>());
            let a = _mm512_loadu_si512(add_ptr.add(offset).cast::<__m512i>());
            let sa = _mm512_loadu_si512(sub_a_ptr.add(offset).cast::<__m512i>());
            let sb = _mm512_loadu_si512(sub_b_ptr.add(offset).cast::<__m512i>());
            let acc = _mm512_sub_epi16(_mm512_sub_epi16(_mm512_add_epi16(acc, a), sa), sb);
            _mm512_storeu_si512(out_ptr.add(offset).cast::<__m512i>(), acc);
        }
    }
}

/// `dst = src + add - sub`, in one pass over the half.
#[target_feature(enable = "avx512f,avx512bw")]
pub fn add_sub_features_from(
    src: &[i16; HIDDEN_SIZE],
    dst: &mut [i16; HIDDEN_SIZE],
    weights: &[i16],
    added: FeatureIndex,
    removed: FeatureIndex,
) {
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let add_ptr = ft_column(weights, added).as_ptr();
    let sub_ptr = ft_column(weights, removed).as_ptr();
    for chunk in 0..NUM_CHUNKS {
        let offset = chunk * LANES;
        // SAFETY: `chunk * LANES + LANES <= HIDDEN_SIZE`, the length of `src`,
        // `dst` and each column.
        unsafe {
            let acc = _mm512_loadu_si512(src_ptr.add(offset).cast::<__m512i>());
            let a = _mm512_loadu_si512(add_ptr.add(offset).cast::<__m512i>());
            let s = _mm512_loadu_si512(sub_ptr.add(offset).cast::<__m512i>());
            let acc = _mm512_sub_epi16(_mm512_add_epi16(acc, a), s);
            _mm512_storeu_si512(dst_ptr.add(offset).cast::<__m512i>(), acc);
        }
    }
}

/// `dst = src + add - sub_a - sub_b`, in one pass over the half.
#[target_feature(enable = "avx512f,avx512bw")]
pub fn add_sub_sub_features_from(
    src: &[i16; HIDDEN_SIZE],
    dst: &mut [i16; HIDDEN_SIZE],
    weights: &[i16],
    added: FeatureIndex,
    removed_a: FeatureIndex,
    removed_b: FeatureIndex,
) {
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let add_ptr = ft_column(weights, added).as_ptr();
    let sub_a_ptr = ft_column(weights, removed_a).as_ptr();
    let sub_b_ptr = ft_column(weights, removed_b).as_ptr();
    for chunk in 0..NUM_CHUNKS {
        let offset = chunk * LANES;
        // SAFETY: `chunk * LANES + LANES <= HIDDEN_SIZE`, the length of `src`,
        // `dst` and each column.
        unsafe {
            let acc = _mm512_loadu_si512(src_ptr.add(offset).cast::<__m512i>());
            let a = _mm512_loadu_si512(add_ptr.add(offset).cast::<__m512i>());
            let sa = _mm512_loadu_si512(sub_a_ptr.add(offset).cast::<__m512i>());
            let sb = _mm512_loadu_si512(sub_b_ptr.add(offset).cast::<__m512i>());
            let acc = _mm512_sub_epi16(_mm512_sub_epi16(_mm512_add_epi16(acc, a), sa), sb);
            _mm512_storeu_si512(dst_ptr.add(offset).cast::<__m512i>(), acc);
        }
    }
}

/// `dst = src - col` (`SUB`) or `dst = src + col`, one pass over the half.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn column_from<const SUB: bool>(
    src: &[i16; HIDDEN_SIZE],
    dst: &mut [i16; HIDDEN_SIZE],
    col: &[i16; HIDDEN_SIZE],
) {
    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();
    let col_ptr = col.as_ptr();
    for chunk in 0..NUM_CHUNKS {
        let offset = chunk * LANES;
        // SAFETY: `chunk * LANES + LANES <= HIDDEN_SIZE`, the length of `src`,
        // `dst` and `col`.
        unsafe {
            let acc = _mm512_loadu_si512(src_ptr.add(offset).cast::<__m512i>());
            let w = _mm512_loadu_si512(col_ptr.add(offset).cast::<__m512i>());
            let acc = if SUB {
                _mm512_sub_epi16(acc, w)
            } else {
                _mm512_add_epi16(acc, w)
            };
            _mm512_storeu_si512(dst_ptr.add(offset).cast::<__m512i>(), acc);
        }
    }
}

/// `out -= col` (`SUB`) or `out += col` in place, storing each result lane to
/// `copy` as well.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn column_tee<const SUB: bool>(
    out: &mut [i16; HIDDEN_SIZE],
    copy: &mut [i16; HIDDEN_SIZE],
    col: &[i16; HIDDEN_SIZE],
) {
    let out_ptr = out.as_mut_ptr();
    let copy_ptr = copy.as_mut_ptr();
    let col_ptr = col.as_ptr();
    for chunk in 0..NUM_CHUNKS {
        let offset = chunk * LANES;
        // SAFETY: `chunk * LANES + LANES <= HIDDEN_SIZE`, the length of `out`,
        // `copy` and `col`.
        unsafe {
            let acc = _mm512_loadu_si512(out_ptr.add(offset).cast::<__m512i>());
            let w = _mm512_loadu_si512(col_ptr.add(offset).cast::<__m512i>());
            let acc = if SUB {
                _mm512_sub_epi16(acc, w)
            } else {
                _mm512_add_epi16(acc, w)
            };
            _mm512_storeu_si512(out_ptr.add(offset).cast::<__m512i>(), acc);
            _mm512_storeu_si512(copy_ptr.add(offset).cast::<__m512i>(), acc);
        }
    }
}

/// `dst = src - sum(removed) + sum(added)` for lists of any length: the first
/// column's pass reads `src` and writes `dst`, every later one runs in place on
/// `dst`. Both lists empty is a plain copy.
#[target_feature(enable = "avx512f,avx512bw")]
pub fn sub_add_features_from(
    src: &[i16; HIDDEN_SIZE],
    dst: &mut [i16; HIDDEN_SIZE],
    weights: &[i16],
    added: &[FeatureIndex],
    removed: &[FeatureIndex],
) {
    match (removed, added) {
        ([first, rest @ ..], _) => {
            column_from::<true>(src, dst, ft_column(weights, *first));
            sub_features(dst, weights, rest);
            add_features(dst, weights, added);
        }
        ([], [first, rest @ ..]) => {
            column_from::<false>(src, dst, ft_column(weights, *first));
            add_features(dst, weights, rest);
        }
        ([], []) => *dst = *src,
    }
}

/// Apply `-sum(removed) + sum(added)` to `out` in place and leave the result in
/// `copy` as well: the last column's pass stores each lane to both, so the copy
/// costs no pass of its own. Both lists empty is a plain copy.
#[target_feature(enable = "avx512f,avx512bw")]
pub fn sub_add_features_tee(
    out: &mut [i16; HIDDEN_SIZE],
    copy: &mut [i16; HIDDEN_SIZE],
    weights: &[i16],
    added: &[FeatureIndex],
    removed: &[FeatureIndex],
) {
    match (removed, added) {
        (_, [rest @ .., last]) => {
            sub_features(out, weights, removed);
            add_features(out, weights, rest);
            column_tee::<false>(out, copy, ft_column(weights, *last));
        }
        ([rest @ .., last], []) => {
            sub_features(out, weights, rest);
            column_tee::<true>(out, copy, ft_column(weights, *last));
        }
        ([], []) => *copy = *out,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::scalar;
    use crate::simd::scalar::tests::{DIFFS, Lanes};

    /// Skip the body unless the CPU has the features these kernels need, so a
    /// host without them skips rather than fails.
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

    fn fill_weights(weights: &mut [i16], seed: i16) {
        for (i, slot) in weights.iter_mut().enumerate() {
            *slot = ((i as i32 * 7 + seed as i32) % 29 - 14) as i16;
        }
    }

    fn seeded_initial(seed: i16) -> [i16; HIDDEN_SIZE] {
        let mut out = [0i16; HIDDEN_SIZE];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = ((i as i32 * 13 + seed as i32) % 41 - 20) as i16;
        }
        out
    }

    #[test]
    fn add_features_matches_scalar_on_random_corpus() {
        require_avx512bw!();
        let mut weights = vec![0i16; HIDDEN_SIZE * 8].into_boxed_slice();
        fill_weights(&mut weights, 17);
        let indices: [FeatureIndex; _] = [0, 2, 5, 7];
        let initial = seeded_initial(3);

        let mut avx = initial;
        // SAFETY: guarded by `require_avx512bw!`.
        unsafe { add_features(&mut avx, &weights, &indices) };

        let mut sca = initial;
        scalar::add_features(&mut sca, &weights, &indices);

        assert_eq!(avx, sca);
    }

    #[test]
    fn sub_features_matches_scalar_on_random_corpus() {
        require_avx512bw!();
        let mut weights = vec![0i16; HIDDEN_SIZE * 8].into_boxed_slice();
        fill_weights(&mut weights, 29);
        let indices: [FeatureIndex; _] = [1, 3, 6];
        let initial = seeded_initial(13);

        let mut avx = initial;
        // SAFETY: guarded by `require_avx512bw!`.
        unsafe { sub_features(&mut avx, &weights, &indices) };

        let mut sca = initial;
        scalar::sub_features(&mut sca, &weights, &indices);

        assert_eq!(avx, sca);
    }

    #[test]
    fn add_sub_features_matches_scalar_on_random_corpus() {
        require_avx512bw!();
        let mut weights = vec![0i16; HIDDEN_SIZE * 8].into_boxed_slice();
        fill_weights(&mut weights, 23);
        let added: [FeatureIndex; _] = [3];
        let removed: [FeatureIndex; _] = [6];
        let initial = seeded_initial(7);

        let mut avx = initial;
        // SAFETY: guarded by `require_avx512bw!`.
        unsafe { add_sub_features(&mut avx, &weights, added[0], removed[0]) };

        let mut sca = initial;
        scalar::add_sub_features(&mut sca, &weights, added[0], removed[0]);

        assert_eq!(avx, sca);
    }

    #[test]
    fn add_sub_sub_features_matches_scalar_on_random_corpus() {
        require_avx512bw!();
        let mut weights = vec![0i16; HIDDEN_SIZE * 8].into_boxed_slice();
        fill_weights(&mut weights, 31);
        let added: [FeatureIndex; _] = [2];
        let removed_a: [FeatureIndex; _] = [5];
        let removed_b: [FeatureIndex; _] = [7];
        let initial = seeded_initial(19);

        let mut avx = initial;
        // SAFETY: guarded by `require_avx512bw!`.
        unsafe { add_sub_sub_features(&mut avx, &weights, added[0], removed_a[0], removed_b[0]) };

        let mut sca = initial;
        scalar::add_sub_sub_features(&mut sca, &weights, added[0], removed_a[0], removed_b[0]);

        assert_eq!(avx, sca);
    }

    #[test]
    fn from_kernels_match_copy_then_update_in_place() {
        require_avx512bw!();
        let mut rng = Lanes(0x2545_f491_4f6c_dd1d);
        let weights = rng.weights(8);
        for &(added, removed) in DIFFS {
            let src = rng.half();
            let mut expected = src;
            scalar::sub_features(&mut expected, &weights, removed);
            scalar::add_features(&mut expected, &weights, added);

            let mut in_place = src;
            // SAFETY: guarded by `require_avx512bw!`.
            unsafe {
                sub_features(&mut in_place, &weights, removed);
                add_features(&mut in_place, &weights, added);
            }
            assert_eq!(in_place, expected, "in-place {added:?} {removed:?}");

            let mut dst = rng.half();
            // SAFETY: guarded by `require_avx512bw!`.
            unsafe { sub_add_features_from(&src, &mut dst, &weights, added, removed) };
            assert_eq!(dst, expected, "sub_add_features_from {added:?} {removed:?}");

            if let ([a], [r]) = (added, removed) {
                let mut dst = rng.half();
                // SAFETY: guarded by `require_avx512bw!`.
                unsafe { add_sub_features_from(&src, &mut dst, &weights, *a, *r) };
                assert_eq!(dst, expected, "add_sub_features_from {added:?} {removed:?}");
            }
            if let ([a], [r0, r1]) = (added, removed) {
                let mut dst = rng.half();
                // SAFETY: guarded by `require_avx512bw!`.
                unsafe { add_sub_sub_features_from(&src, &mut dst, &weights, *a, *r0, *r1) };
                assert_eq!(
                    dst, expected,
                    "add_sub_sub_features_from {added:?} {removed:?}"
                );
            }

            let mut out = src;
            let mut copy = rng.half();
            // SAFETY: guarded by `require_avx512bw!`.
            unsafe { sub_add_features_tee(&mut out, &mut copy, &weights, added, removed) };
            assert_eq!(
                out, expected,
                "sub_add_features_tee (in place) {added:?} {removed:?}"
            );
            assert_eq!(
                copy, expected,
                "sub_add_features_tee (copy) {added:?} {removed:?}"
            );
        }
    }
}
