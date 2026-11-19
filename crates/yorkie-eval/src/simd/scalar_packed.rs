//! Portable feature-transformer kernels over packed weight columns.
//!
//! Each kernel has the shape of its [`crate::simd::scalar`] counterpart and
//! reads a column as its bytes doubled plus that feature's corrections, which
//! is the 16-bit column exactly. A build only packs its weights where it also
//! compiles the AVX-512 kernels, so these are the baseline those are held
//! against, and what a packed network made in memory runs on anywhere else.

use crate::features::FeatureIndex;
use crate::types::{HIDDEN_SIZE, PackedFt};

/// `dst -= col` (`SUB`) or `dst += col`.
fn column<const SUB: bool>(dst: &mut [i16; HIDDEN_SIZE], packed: PackedFt<'_>, idx: FeatureIndex) {
    for (lane, &base) in dst.iter_mut().zip(packed.bases(idx)) {
        let w = i16::from(base) << 1;
        *lane = if SUB {
            lane.wrapping_sub(w)
        } else {
            lane.wrapping_add(w)
        };
    }
    packed.correct::<SUB>(dst, idx);
}

pub fn add_features(out: &mut [i16; HIDDEN_SIZE], packed: PackedFt<'_>, indices: &[FeatureIndex]) {
    for &idx in indices {
        column::<false>(out, packed, idx);
    }
}

pub fn sub_features(out: &mut [i16; HIDDEN_SIZE], packed: PackedFt<'_>, indices: &[FeatureIndex]) {
    for &idx in indices {
        column::<true>(out, packed, idx);
    }
}

pub fn add_sub_features(
    out: &mut [i16; HIDDEN_SIZE],
    packed: PackedFt<'_>,
    added: FeatureIndex,
    removed: FeatureIndex,
) {
    column::<false>(out, packed, added);
    column::<true>(out, packed, removed);
}

pub fn add_sub_sub_features(
    out: &mut [i16; HIDDEN_SIZE],
    packed: PackedFt<'_>,
    added: FeatureIndex,
    removed_a: FeatureIndex,
    removed_b: FeatureIndex,
) {
    column::<false>(out, packed, added);
    column::<true>(out, packed, removed_a);
    column::<true>(out, packed, removed_b);
}

pub fn add_sub_features_from(
    src: &[i16; HIDDEN_SIZE],
    dst: &mut [i16; HIDDEN_SIZE],
    packed: PackedFt<'_>,
    added: FeatureIndex,
    removed: FeatureIndex,
) {
    *dst = *src;
    add_sub_features(dst, packed, added, removed);
}

pub fn add_sub_sub_features_from(
    src: &[i16; HIDDEN_SIZE],
    dst: &mut [i16; HIDDEN_SIZE],
    packed: PackedFt<'_>,
    added: FeatureIndex,
    removed_a: FeatureIndex,
    removed_b: FeatureIndex,
) {
    *dst = *src;
    add_sub_sub_features(dst, packed, added, removed_a, removed_b);
}

pub fn sub_add_features_from(
    src: &[i16; HIDDEN_SIZE],
    dst: &mut [i16; HIDDEN_SIZE],
    packed: PackedFt<'_>,
    added: &[FeatureIndex],
    removed: &[FeatureIndex],
) {
    *dst = *src;
    sub_features(dst, packed, removed);
    add_features(dst, packed, added);
}

pub fn sub_add_features_tee(
    out: &mut [i16; HIDDEN_SIZE],
    copy: &mut [i16; HIDDEN_SIZE],
    packed: PackedFt<'_>,
    added: &[FeatureIndex],
    removed: &[FeatureIndex],
) {
    sub_features(out, packed, removed);
    add_features(out, packed, added);
    *copy = *out;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::scalar;
    use crate::simd::scalar::tests::{DIFFS, Lanes};
    use crate::types::tests::PackedColumns;

    #[test]
    fn every_kernel_matches_the_16_bit_one_over_the_unpacked_weights() {
        let mut rng = Lanes(0x6a09_e667_f3bc_c908);
        let wide = rng.weights(8);
        let columns = PackedColumns::from_wide(&wide);
        let packed = columns.view();
        for &(added, removed) in DIFFS {
            let src = rng.half();
            let mut expected = src;
            scalar::sub_features(&mut expected, &wide, removed);
            scalar::add_features(&mut expected, &wide, added);

            let mut dst = rng.half();
            sub_add_features_from(&src, &mut dst, packed, added, removed);
            assert_eq!(dst, expected, "sub_add_features_from {added:?} {removed:?}");

            if let ([a], [r]) = (added, removed) {
                let mut dst = rng.half();
                add_sub_features_from(&src, &mut dst, packed, *a, *r);
                assert_eq!(dst, expected, "add_sub_features_from {added:?} {removed:?}");
            }
            if let ([a], [r0, r1]) = (added, removed) {
                let mut dst = rng.half();
                add_sub_sub_features_from(&src, &mut dst, packed, *a, *r0, *r1);
                assert_eq!(
                    dst, expected,
                    "add_sub_sub_features_from {added:?} {removed:?}"
                );
            }

            let mut out = src;
            let mut copy = rng.half();
            sub_add_features_tee(&mut out, &mut copy, packed, added, removed);
            assert_eq!(out, expected, "tee (in place) {added:?} {removed:?}");
            assert_eq!(copy, expected, "tee (copy) {added:?} {removed:?}");
        }
    }
}
