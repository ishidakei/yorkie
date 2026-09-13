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

#[cfg(test)]
mod tests {
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

        let indices: [FeatureIndex; 3] = [0, 2, 3];
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
        let indices: [FeatureIndex; 2] = [0, 1];
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
        let indices: [FeatureIndex; 0] = [];
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

        let added: [FeatureIndex; 1] = [3];
        let removed: [FeatureIndex; 1] = [6];

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

        let added: [FeatureIndex; 1] = [2];
        let removed_a: [FeatureIndex; 1] = [5];
        let removed_b: [FeatureIndex; 1] = [7];

        let initial = seeded_initial(19);

        let mut fused = initial;
        add_sub_sub_features(&mut fused, &weights, added[0], removed_a[0], removed_b[0]);

        let mut unfused = initial;
        add_features(&mut unfused, &weights, &added);
        sub_features(&mut unfused, &weights, &removed_a);
        sub_features(&mut unfused, &weights, &removed_b);

        assert_eq!(fused, unfused);
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
