//! Scalar feature-transformer accumulate/update kernels.
//!
//! The scalar backend, matching the reference engine's NNUE arithmetic. These
//! are the always-available baseline: the AVX-512 kernels in
//! [`crate::simd::avx512`] must match them bit-for-bit (see that module's
//! parity tests).
//!
//! `wrapping_add`/`wrapping_sub` match upstream NNUE semantics: `i16` overflow
//! is allowed here because the downstream clipped output transform saturates.

use crate::features::FeatureIndex;
use crate::simd::ft_column;
use crate::types::HIDDEN_SIZE;

pub fn add_features(out: &mut [i16; HIDDEN_SIZE], weights: &[i16], indices: &[FeatureIndex]) {
    for &idx in indices {
        let col = ft_column(weights, idx);
        for (o, &w) in out.iter_mut().zip(col.iter()) {
            *o = o.wrapping_add(w);
        }
    }
}

pub fn sub_features(out: &mut [i16; HIDDEN_SIZE], weights: &[i16], indices: &[FeatureIndex]) {
    for &idx in indices {
        let col = ft_column(weights, idx);
        for (o, &w) in out.iter_mut().zip(col.iter()) {
            *o = o.wrapping_sub(w);
        }
    }
}

pub fn add_sub_features(
    out: &mut [i16; HIDDEN_SIZE],
    weights: &[i16],
    added: FeatureIndex,
    removed: FeatureIndex,
) {
    let add = ft_column(weights, added);
    let sub = ft_column(weights, removed);
    for ((slot, &a), &s) in out.iter_mut().zip(add.iter()).zip(sub.iter()) {
        *slot = slot.wrapping_add(a).wrapping_sub(s);
    }
}

pub fn add_sub_sub_features(
    out: &mut [i16; HIDDEN_SIZE],
    weights: &[i16],
    added: FeatureIndex,
    removed_a: FeatureIndex,
    removed_b: FeatureIndex,
) {
    let add = ft_column(weights, added);
    let sub_a = ft_column(weights, removed_a);
    let sub_b = ft_column(weights, removed_b);
    for (((slot, &a), &sa), &sb) in out
        .iter_mut()
        .zip(add.iter())
        .zip(sub_a.iter())
        .zip(sub_b.iter())
    {
        *slot = slot.wrapping_add(a).wrapping_sub(sa).wrapping_sub(sb);
    }
}

/// `dst = src - col` (`SUB`) or `dst = src + col`, one pass over the half.
#[inline]
fn column_from<const SUB: bool>(
    src: &[i16; HIDDEN_SIZE],
    dst: &mut [i16; HIDDEN_SIZE],
    col: &[i16; HIDDEN_SIZE],
) {
    for ((d, &s), &w) in dst.iter_mut().zip(src.iter()).zip(col.iter()) {
        *d = if SUB {
            s.wrapping_sub(w)
        } else {
            s.wrapping_add(w)
        };
    }
}

/// `out -= col` (`SUB`) or `out += col` in place, storing each result lane to
/// `copy` as well.
#[inline]
fn column_tee<const SUB: bool>(
    out: &mut [i16; HIDDEN_SIZE],
    copy: &mut [i16; HIDDEN_SIZE],
    col: &[i16; HIDDEN_SIZE],
) {
    for ((o, c), &w) in out.iter_mut().zip(copy.iter_mut()).zip(col.iter()) {
        *o = if SUB {
            o.wrapping_sub(w)
        } else {
            o.wrapping_add(w)
        };
        *c = *o;
    }
}

/// `dst = src + add - sub`, in one pass over the half.
pub fn add_sub_features_from(
    src: &[i16; HIDDEN_SIZE],
    dst: &mut [i16; HIDDEN_SIZE],
    weights: &[i16],
    added: FeatureIndex,
    removed: FeatureIndex,
) {
    let add = ft_column(weights, added);
    let sub = ft_column(weights, removed);
    for (((d, &s), &a), &r) in dst
        .iter_mut()
        .zip(src.iter())
        .zip(add.iter())
        .zip(sub.iter())
    {
        *d = s.wrapping_add(a).wrapping_sub(r);
    }
}

/// `dst = src + add - sub_a - sub_b`, in one pass over the half.
pub fn add_sub_sub_features_from(
    src: &[i16; HIDDEN_SIZE],
    dst: &mut [i16; HIDDEN_SIZE],
    weights: &[i16],
    added: FeatureIndex,
    removed_a: FeatureIndex,
    removed_b: FeatureIndex,
) {
    let add = ft_column(weights, added);
    let sub_a = ft_column(weights, removed_a);
    let sub_b = ft_column(weights, removed_b);
    for ((((d, &s), &a), &sa), &sb) in dst
        .iter_mut()
        .zip(src.iter())
        .zip(add.iter())
        .zip(sub_a.iter())
        .zip(sub_b.iter())
    {
        *d = s.wrapping_add(a).wrapping_sub(sa).wrapping_sub(sb);
    }
}

/// `dst = src - sum(removed) + sum(added)` for lists of any length: the first
/// column's pass reads `src` and writes `dst`, every later one runs in place on
/// `dst`. Both lists empty is a plain copy.
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
pub(crate) mod tests {
    use super::*;

    fn fill_weights(weights: &mut [i16], seed: i16) {
        for (i, slot) in weights.iter_mut().enumerate() {
            *slot = ((i as i32 * 7 + seed as i32) % 29 - 14) as i16;
        }
    }

    #[test]
    fn add_then_sub_same_indices_is_identity() {
        let mut out = [3i16; HIDDEN_SIZE];
        let initial = out;
        let mut weights = vec![0i16; HIDDEN_SIZE * 4].into_boxed_slice();
        fill_weights(&mut weights, 11);

        let indices: [FeatureIndex; _] = [0, 2, 3];
        add_features(&mut out, &weights, &indices);
        sub_features(&mut out, &weights, &indices);
        assert_eq!(out, initial);
    }

    #[test]
    fn add_accumulates_column_values() {
        let mut out = [0i16; HIDDEN_SIZE];
        let mut weights = vec![0i16; HIDDEN_SIZE * 2].into_boxed_slice();
        for i in 0..HIDDEN_SIZE {
            weights[i] = if i % 2 == 0 { 1 } else { -1 };
            weights[HIDDEN_SIZE + i] = 2;
        }
        let indices: [FeatureIndex; _] = [0, 1];
        add_features(&mut out, &weights, &indices);
        for (i, &o) in out.iter().enumerate() {
            let expected = (if i % 2 == 0 { 1 } else { -1 }) + 2;
            assert_eq!(o, expected);
        }
    }

    #[test]
    fn empty_indices_do_not_mutate() {
        let mut out = [5i16; HIDDEN_SIZE];
        let weights = vec![99i16; HIDDEN_SIZE].into_boxed_slice();
        let indices: [FeatureIndex; _] = [];
        add_features(&mut out, &weights, &indices);
        sub_features(&mut out, &weights, &indices);
        assert!(out.iter().all(|&x| x == 5));
    }

    fn seeded_initial(seed: i16) -> [i16; HIDDEN_SIZE] {
        let mut out = [0i16; HIDDEN_SIZE];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = ((i as i32 * 13 + seed as i32) % 41 - 20) as i16;
        }
        out
    }

    #[test]
    fn add_sub_features_matches_unfused_composition() {
        let mut weights = vec![0i16; HIDDEN_SIZE * 8].into_boxed_slice();
        fill_weights(&mut weights, 23);

        let added: [FeatureIndex; _] = [3];
        let removed: [FeatureIndex; _] = [6];

        let initial = seeded_initial(7);

        let mut fused = initial;
        add_sub_features(&mut fused, &weights, added[0], removed[0]);

        let mut unfused = initial;
        add_features(&mut unfused, &weights, &added);
        sub_features(&mut unfused, &weights, &removed);

        assert_eq!(fused, unfused);
    }

    #[test]
    fn add_sub_sub_features_matches_unfused_composition() {
        let mut weights = vec![0i16; HIDDEN_SIZE * 8].into_boxed_slice();
        fill_weights(&mut weights, 31);

        let added: [FeatureIndex; _] = [2];
        let removed_a: [FeatureIndex; _] = [5];
        let removed_b: [FeatureIndex; _] = [7];

        let initial = seeded_initial(19);

        let mut fused = initial;
        add_sub_sub_features(&mut fused, &weights, added[0], removed_a[0], removed_b[0]);

        let mut unfused = initial;
        add_features(&mut unfused, &weights, &added);
        sub_features(&mut unfused, &weights, &removed_a);
        sub_features(&mut unfused, &weights, &removed_b);

        assert_eq!(fused, unfused);
    }

    /// A deterministic xorshift stream, so the "random" inputs below are the
    /// same on every run.
    pub(crate) struct Lanes(pub(crate) u64);

    impl Lanes {
        pub(crate) fn next(&mut self) -> i16 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 48) as i16
        }

        pub(crate) fn half(&mut self) -> [i16; HIDDEN_SIZE] {
            std::array::from_fn(|_| self.next())
        }

        pub(crate) fn weights(&mut self, columns: usize) -> Box<[i16]> {
            Vec::from_fn(columns * HIDDEN_SIZE, |_| self.next()).into_boxed_slice()
        }
    }

    /// The arities the update path produces, plus empty lists and a repeated
    /// column. Full-range lanes make the wrapping arithmetic overflow.
    pub(crate) const DIFFS: &[(&[FeatureIndex], &[FeatureIndex])] = &[
        (&[], &[]),
        (&[3], &[]),
        (&[], &[5]),
        (&[1], &[6]),
        (&[2], &[5, 7]),
        (&[0, 4], &[1, 7]),
        (&[6, 6, 2], &[3]),
    ];

    #[test]
    fn from_kernels_match_copy_then_update_in_place() {
        let mut rng = Lanes(0x9e37_79b9_7f4a_7c15);
        let weights = rng.weights(8);
        for &(added, removed) in DIFFS {
            let src = rng.half();
            let mut expected = src;
            sub_features(&mut expected, &weights, removed);
            add_features(&mut expected, &weights, added);

            let mut dst = rng.half();
            sub_add_features_from(&src, &mut dst, &weights, added, removed);
            assert_eq!(dst, expected, "sub_add_features_from {added:?} {removed:?}");

            if let ([a], [r]) = (added, removed) {
                let mut dst = rng.half();
                add_sub_features_from(&src, &mut dst, &weights, *a, *r);
                assert_eq!(dst, expected, "add_sub_features_from {added:?} {removed:?}");
            }
            if let ([a], [r0, r1]) = (added, removed) {
                let mut dst = rng.half();
                add_sub_sub_features_from(&src, &mut dst, &weights, *a, *r0, *r1);
                assert_eq!(
                    dst, expected,
                    "add_sub_sub_features_from {added:?} {removed:?}"
                );
            }

            let mut out = src;
            let mut copy = rng.half();
            sub_add_features_tee(&mut out, &mut copy, &weights, added, removed);
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
    fn add_sub_features_with_the_same_column_is_identity() {
        let mut weights = vec![0i16; HIDDEN_SIZE * 4].into_boxed_slice();
        fill_weights(&mut weights, 41);
        let mut out = [5i16; HIDDEN_SIZE];
        add_sub_features(&mut out, &weights, 2, 2);
        assert!(out.iter().all(|&x| x == 5));
    }
}
