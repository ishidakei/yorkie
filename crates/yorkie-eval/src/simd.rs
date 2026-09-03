//! Scalar/SIMD kernel selection for the NNUE forward pass.
//!
//! The backend is chosen **at compile time**, from the CPU features the build
//! itself enables. This repo builds with `-C target-cpu=native` and runs each
//! binary on the machine that produced it, so a run-time probe could never pick
//! anything the build had not already fixed. Two feature sets matter:
//! `avx512f` + `avx512bw` for the kernels in [`avx512`] and
//! [`avx512_post_ft`], and those plus `avx512vnni` for the fused chain.
//!
//! Both backends stay **compiled unconditionally**, only the call sites being
//! `cfg`-gated, so each backend module's SIMD-equals-scalar tests keep
//! exercising both. Those tests, and only those, probe the CPU at run time, so
//! they skip on a host without the features. The kernels are exact, not
//! approximate: whichever the build selects, the output is bit-identical.
//!
//! Every AVX-512 entry point is an `unsafe fn` with a `#[target_feature]`
//! attribute, sound to call only when those features are present. The wrappers
//! here are the sole non-test callers, and each `unsafe` call sits behind a
//! `cfg(target_feature = ...)` naming exactly what its callee enables, so no
//! `unsafe` escapes the SIMD modules.

// Both backends are compiled unconditionally so the equivalence tests can
// compare them, which leaves the one this build did not select without a caller
// outside `cfg(test)`.
#[allow(dead_code)]
pub mod scalar;
#[allow(dead_code)]
pub mod scalar_post_ft;

#[allow(dead_code)]
#[cfg(target_arch = "x86_64")]
pub mod avx512;
#[allow(dead_code)]
#[cfg(target_arch = "x86_64")]
pub mod avx512_post_ft;

/// Which kernel backend this build compiled into the forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// The portable scalar baseline.
    Scalar,
    /// AVX-512 F + BW + VNNI kernels (the full SIMD forward pass).
    Avx512Vnni,
}

/// The kernel backend baked into this build.
///
/// [`Backend::Scalar`] covers the partial case of an `avx512f` + `avx512bw`
/// build without VNNI too, where the feature transformer is SIMD but the layer
/// stack is not.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vnni"
))]
pub const fn active_backend() -> Backend {
    Backend::Avx512Vnni
}

/// The kernel backend baked into this build — non-VNNI arm (the VNNI arm above
/// carries the documentation).
#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vnni"
)))]
pub const fn active_backend() -> Backend {
    Backend::Scalar
}

/// Feature-transformer accumulate and update kernels. The AVX-512 arm is
/// compiled only when the build enables exactly the features
/// [`super::avx512`]'s kernels declare; every other build gets the scalar
/// baseline.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
pub mod transformer_kernel {
    use super::avx512;
    use crate::features::FeatureIndex;

    /// Add each active feature's FT weight column into `out`.
    #[inline]
    pub fn add_features(out: &mut [i16], weights: &[i16], indices: &[FeatureIndex]) {
        // SAFETY: this module is compiled only into a build enabling exactly
        // the features the callee's `#[target_feature]` names, and such a build
        // is `-C target-cpu=native`, so it only ever runs on a host with them.
        unsafe { avx512::add_features(out, weights, indices) }
    }

    /// Subtract each feature's FT weight column from `out`.
    #[inline]
    pub fn sub_features(out: &mut [i16], weights: &[i16], indices: &[FeatureIndex]) {
        // SAFETY: see `add_features`.
        unsafe { avx512::sub_features(out, weights, indices) }
    }

    /// Fused single-add / single-sub delta.
    #[inline]
    pub fn add_sub_features(
        out: &mut [i16],
        weights: &[i16],
        added: &[FeatureIndex],
        removed: &[FeatureIndex],
    ) {
        // SAFETY: see `add_features`.
        unsafe { avx512::add_sub_features(out, weights, added, removed) }
    }

    /// Fused single-add / double-sub delta (capture-style updates).
    #[inline]
    pub fn add_sub_sub_features(
        out: &mut [i16],
        weights: &[i16],
        added: &[FeatureIndex],
        removed_a: &[FeatureIndex],
        removed_b: &[FeatureIndex],
    ) {
        // SAFETY: see `add_features`.
        unsafe { avx512::add_sub_sub_features(out, weights, added, removed_a, removed_b) }
    }
}

/// Feature-transformer kernels, scalar arm; the AVX-512 arm above carries the
/// selection rule.
#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw"
)))]
pub mod transformer_kernel {
    pub use super::scalar::{add_features, add_sub_features, add_sub_sub_features, sub_features};
}

/// Output-transform and layer element-wise kernels, selected on the same
/// condition as [`transformer_kernel`].
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
pub mod post_ft_kernel {
    use super::{avx512_post_ft, scalar_post_ft};

    /// Pairwise element-wise multiply for one perspective half.
    #[inline]
    pub fn ewm_one_perspective(half: &[i16], out: &mut [u8]) {
        // SAFETY: this module is compiled only into a build enabling exactly
        // the features the callee's `#[target_feature]` names, and such a build
        // is `-C target-cpu=native`, so it only ever runs on a host with them.
        unsafe { avx512_post_ft::ewm_one_perspective(half, out) }
    }

    /// Clipped ReLU.
    #[inline]
    pub fn clipped_relu(input: &[i32], output: &mut [u8]) {
        // SAFETY: see `ewm_one_perspective`.
        unsafe { avx512_post_ft::clipped_relu(input, output) }
    }

    /// Squared clipped ReLU.
    #[inline]
    pub fn sqr_clipped_relu(input: &[i32], output: &mut [u8]) {
        // SAFETY: see `ewm_one_perspective`.
        unsafe { avx512_post_ft::sqr_clipped_relu(input, output) }
    }

    /// Integer affine transform, always scalar: the AVX-512 form needs VNNI and
    /// only repays its setup at the wide `fc_0` shape, where the whole chain
    /// goes through the fused kernel anyway.
    #[inline]
    pub fn affine(
        output: &mut [i32],
        biases: &[i32],
        weights: &[i8],
        input: &[u8],
        in_dims: usize,
        padded_in: usize,
    ) {
        scalar_post_ft::affine(output, biases, weights, input, in_dims, padded_in);
    }
}

/// Output-transform and layer element-wise kernels, scalar arm; the AVX-512 arm
/// above carries the selection rule.
#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw"
)))]
pub mod post_ft_kernel {
    pub use super::scalar_post_ft::{affine, clipped_relu, ewm_one_perspective, sqr_clipped_relu};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn active_backend_is_avx512_when_cpu_supports_vnni() {
        // Backend selection is compile-time, so this checks the *build* was
        // configured for its host: under `-C target-cpu=native` a VNNI-capable
        // CPU must yield a VNNI build. A scalar backend here means the build
        // silently fell back.
        #[cfg(target_arch = "x86_64")]
        {
            let has_vnni = std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw")
                && std::arch::is_x86_feature_detected!("avx512vnni");
            if has_vnni {
                assert_eq!(
                    active_backend(),
                    Backend::Avx512Vnni,
                    "CPU reports AVX-512 VNNI but this build compiled the scalar \
                     backend — check that `-C target-cpu=native` is in effect",
                );
            } else {
                assert_eq!(active_backend(), Backend::Scalar);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        assert_eq!(active_backend(), Backend::Scalar);
    }
}
