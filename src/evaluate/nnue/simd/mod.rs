use super::types::{Accumulator, FeatureIndex, FC_0_INPUT_DIMS};
use crate::types::Color;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw"))]
mod avx512;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vnni"
))]
mod avx512_post_ft;
mod scalar;
mod scalar_post_ft;

pub mod transformer_kernel {
    use super::FeatureIndex;

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw"))]
    #[inline]
    pub fn add_features(out: &mut [i16], weights: &[i16], indices: &[FeatureIndex]) {
        // SAFETY: only compiled when avx512f + avx512bw are enabled at binary level.
        unsafe { super::avx512::add_features(out, weights, indices) }
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw")))]
    #[inline]
    pub fn add_features(out: &mut [i16], weights: &[i16], indices: &[FeatureIndex]) {
        super::scalar::add_features(out, weights, indices)
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw"))]
    #[inline]
    pub fn sub_features(out: &mut [i16], weights: &[i16], indices: &[FeatureIndex]) {
        // SAFETY: see `add_features`.
        unsafe { super::avx512::sub_features(out, weights, indices) }
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw")))]
    #[inline]
    pub fn sub_features(out: &mut [i16], weights: &[i16], indices: &[FeatureIndex]) {
        super::scalar::sub_features(out, weights, indices)
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw"))]
    #[inline]
    pub fn add_sub_features(out: &mut [i16], weights: &[i16], added: &[FeatureIndex], removed: &[FeatureIndex]) {
        // SAFETY: see `add_features`.
        unsafe { super::avx512::add_sub_features(out, weights, added, removed) }
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw")))]
    #[inline]
    pub fn add_sub_features(out: &mut [i16], weights: &[i16], added: &[FeatureIndex], removed: &[FeatureIndex]) {
        super::scalar::add_sub_features(out, weights, added, removed)
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw"))]
    #[inline]
    pub fn add_sub_sub_features(
        out: &mut [i16],
        weights: &[i16],
        added: &[FeatureIndex],
        removed_a: &[FeatureIndex],
        removed_b: &[FeatureIndex],
    ) {
        // SAFETY: see `add_features`.
        unsafe { super::avx512::add_sub_sub_features(out, weights, added, removed_a, removed_b) }
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f", target_feature = "avx512bw")))]
    #[inline]
    pub fn add_sub_sub_features(
        out: &mut [i16],
        weights: &[i16],
        added: &[FeatureIndex],
        removed_a: &[FeatureIndex],
        removed_b: &[FeatureIndex],
    ) {
        super::scalar::add_sub_sub_features(out, weights, added, removed_a, removed_b)
    }
}

pub mod post_ft_kernel {
    use super::{Accumulator, Color, FC_0_INPUT_DIMS};

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vnni"
    ))]
    #[inline]
    pub fn transformer_ewm(acc: &Accumulator, stm: Color, out: &mut [u8; FC_0_INPUT_DIMS]) {
        // SAFETY: only compiled when avx512f + avx512bw + avx512vnni are enabled.
        unsafe { super::avx512_post_ft::transformer_ewm(acc, stm, out) }
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vnni"
    )))]
    #[inline]
    pub fn transformer_ewm(acc: &Accumulator, stm: Color, out: &mut [u8; FC_0_INPUT_DIMS]) {
        super::scalar_post_ft::transformer_ewm(acc, stm, out)
    }

    // Per-layer kernels are only re-exported on the scalar arm; on the VNNI
    // arm the whole chain runs through `fused_fc_chain` below.
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vnni"
    )))]
    pub use super::scalar_post_ft::{affine, clipped_relu, sqr_clipped_relu};

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vnni"
    ))]
    #[inline]
    pub fn fused_fc_chain(
        transformed: &[u8; FC_0_INPUT_DIMS],
        fc_0_biases: &[i32],
        fc_0_weights: &[i8],
        fc_1_biases: &[i32],
        fc_1_weights: &[i8],
        fc_2_biases: &[i32],
        fc_2_weights: &[i8],
    ) -> i32 {
        // SAFETY: only compiled when avx512f + avx512bw + avx512vnni are enabled.
        unsafe {
            super::avx512_post_ft::fused_fc_chain(
                transformed,
                fc_0_biases,
                fc_0_weights,
                fc_1_biases,
                fc_1_weights,
                fc_2_biases,
                fc_2_weights,
            )
        }
    }
}
