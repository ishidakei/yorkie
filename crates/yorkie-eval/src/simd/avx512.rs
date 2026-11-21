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

/// Chunks of the half held in registers at once by the list kernels. The half
/// is [`NUM_CHUNKS`] chunks, which this divides; AVX-512 has 32 registers, and
/// a tile of 24 leaves room for the column operand.
const TILE: usize = 24;
const _: () = assert!(NUM_CHUNKS.is_multiple_of(TILE));

/// `dst = src - sum(removed) + sum(added)` for lists of any length, with `TEE`
/// every result chunk stored to `tee` as well. The half goes through in tiles
/// of [`TILE`] chunks: each tile is loaded once, every column is applied to it
/// in registers, and it is stored once.
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
    weights: &[i16],
    added: &[FeatureIndex],
    removed: &[FeatureIndex],
) {
    // Every column is checked against the block once here, so the tile loop
    // below can address each one by plain pointer arithmetic.
    for &idx in added.iter().chain(removed) {
        ft_column(weights, idx);
    }
    let base = weights.as_ptr();
    for tile in 0..NUM_CHUNKS / TILE {
        let first = tile * TILE * LANES;
        // SAFETY: `first + TILE * LANES <= HIDDEN_SIZE`, the length of `src`,
        // `dst` and `tee` by the caller's contract and of every column, each of
        // which was checked to lie inside `weights` above; the unaligned
        // 512-bit ops impose no alignment requirement.
        unsafe {
            let mut acc: [__m512i; TILE] =
                std::array::from_fn(|i| _mm512_loadu_si512(src.add(first + i * LANES).cast()));
            for &idx in removed {
                let col = base.add(idx as usize * HIDDEN_SIZE + first);
                for (i, a) in acc.iter_mut().enumerate() {
                    *a = _mm512_sub_epi16(*a, _mm512_loadu_si512(col.add(i * LANES).cast()));
                }
            }
            for &idx in added {
                let col = base.add(idx as usize * HIDDEN_SIZE + first);
                for (i, a) in acc.iter_mut().enumerate() {
                    *a = _mm512_add_epi16(*a, _mm512_loadu_si512(col.add(i * LANES).cast()));
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

/// Add each active feature's FT weight column into `out`.
#[target_feature(enable = "avx512f,avx512bw")]
pub fn add_features(out: &mut [i16; HIDDEN_SIZE], weights: &[i16], indices: &[FeatureIndex]) {
    let ptr = out.as_mut_ptr();
    // SAFETY: `out` is `HIDDEN_SIZE` lanes, read and written through the one
    // pointer; `tee` is never touched without `TEE`.
    unsafe { tiled::<false>(ptr, ptr, std::ptr::null_mut(), weights, indices, &[]) };
}

/// Subtract each feature's FT weight column from `out`.
#[target_feature(enable = "avx512f,avx512bw")]
pub fn sub_features(out: &mut [i16; HIDDEN_SIZE], weights: &[i16], indices: &[FeatureIndex]) {
    let ptr = out.as_mut_ptr();
    // SAFETY: as `add_features`.
    unsafe { tiled::<false>(ptr, ptr, std::ptr::null_mut(), weights, &[], indices) };
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

/// `dst = src - sum(removed) + sum(added)` for lists of any length, in one pass
/// over the half. Both lists empty is a plain copy.
#[target_feature(enable = "avx512f,avx512bw")]
pub fn sub_add_features_from(
    src: &[i16; HIDDEN_SIZE],
    dst: &mut [i16; HIDDEN_SIZE],
    weights: &[i16],
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
            weights,
            added,
            removed,
        )
    };
}

/// Apply `-sum(removed) + sum(added)` to `out` in place and leave the result in
/// `copy` as well, in one pass over the half that stores each tile to both.
/// Both lists empty is a plain copy.
#[target_feature(enable = "avx512f,avx512bw")]
pub fn sub_add_features_tee(
    out: &mut [i16; HIDDEN_SIZE],
    copy: &mut [i16; HIDDEN_SIZE],
    weights: &[i16],
    added: &[FeatureIndex],
    removed: &[FeatureIndex],
) {
    let ptr = out.as_mut_ptr();
    // SAFETY: `out` and `copy` are `HIDDEN_SIZE` lanes each and distinct, `out`
    // read and written through the one pointer.
    unsafe { tiled::<true>(ptr, ptr, copy.as_mut_ptr(), weights, added, removed) };
}

#[cfg(test)]
pub(crate) mod tests {
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

    #[test]
    fn list_kernels_match_scalar_for_every_list_length() {
        require_avx512bw!();
        let mut rng = Lanes(0x9e37_79b9_7f4a_7c15);
        let weights = rng.weights(COLUMNS);
        for (added, removed) in list_lengths(&mut rng) {
            let src = rng.half();
            let mut expected = src;
            scalar::sub_features(&mut expected, &weights, &removed);
            scalar::add_features(&mut expected, &weights, &added);

            let mut in_place = src;
            // SAFETY: guarded by `require_avx512bw!`.
            unsafe {
                sub_features(&mut in_place, &weights, &removed);
                add_features(&mut in_place, &weights, &added);
            }
            assert_eq!(in_place, expected, "in place {added:?} {removed:?}");

            let mut dst = rng.half();
            // SAFETY: guarded by `require_avx512bw!`.
            unsafe { sub_add_features_from(&src, &mut dst, &weights, &added, &removed) };
            assert_eq!(dst, expected, "from {added:?} {removed:?}");

            let mut out = src;
            let mut copy = rng.half();
            // SAFETY: guarded by `require_avx512bw!`.
            unsafe { sub_add_features_tee(&mut out, &mut copy, &weights, &added, &removed) };
            assert_eq!(out, expected, "tee (in place) {added:?} {removed:?}");
            assert_eq!(copy, expected, "tee (copy) {added:?} {removed:?}");
        }
    }

    /// Columns in the weight block of the list-length tests.
    pub(crate) const COLUMNS: usize = if cfg!(miri) { 8 } else { 64 };

    /// `(added, removed)` lists of every pair of lengths up to
    /// [`MAX_ACTIVE_FEATURES`](crate::features::MAX_ACTIVE_FEATURES), each
    /// drawn from [`COLUMNS`] columns with repeats allowed: every length on its
    /// own against both an empty list and a full one, and the two lists equal
    /// in length.
    ///
    /// Under Miri, which interprets every lane operation, only the short
    /// lengths 0, 1 and 3, each alone and paired with itself: the tile loop
    /// addresses a column the same way whatever the list's length, so these
    /// cover what Miri checks, and the native run covers every length.
    pub(crate) fn list_lengths(rng: &mut Lanes) -> Vec<(Vec<FeatureIndex>, Vec<FeatureIndex>)> {
        use crate::features::MAX_ACTIVE_FEATURES as MAX;
        let mut list = |len: usize| -> Vec<FeatureIndex> {
            (0..len)
                .map(|_| (rng.next() as u16 as usize % COLUMNS) as FeatureIndex)
                .collect()
        };
        let mut cases = Vec::new();
        if cfg!(miri) {
            for len in [0, 1, 3] {
                for (a, r) in [(len, 0), (0, len), (len, len)] {
                    cases.push((list(a), list(r)));
                }
            }
            return cases;
        }
        for len in 0..=MAX {
            for (a, r) in [(len, 0), (0, len), (len, len), (len, MAX), (MAX, len)] {
                cases.push((list(a), list(r)));
            }
        }
        cases
    }
}
