use std::sync::atomic::{AtomicI32, Ordering};

use super::simd::post_ft_kernel;
use super::types::{Accumulator, FC_0_INPUT_DIMS, NnueNetwork};
#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vnni"
)))]
use super::types::{
    FC_0_OUTPUT_DIMS, FC_0_PADDED_INPUT_DIMS, FC_1_INPUT_DIMS, FC_1_OUTPUT_DIMS, FC_1_PADDED_INPUT_DIMS, FC_2_INPUT_DIMS,
    FC_2_OUTPUT_DIMS, FC_2_PADDED_INPUT_DIMS, HIDDEN1_DIMS, NetworkStack,
};
use crate::types::{Color, Value};

static FV_SCALE: AtomicI32 = AtomicI32::new(16);

pub fn set_fv_scale(value: i32) {
    FV_SCALE.store(value, Ordering::Relaxed);
}

/// FV_SCALE applied by [`forward`]: under `tournament` the compile-time const (constant-folds), else the runtime global.
#[cfg(feature = "tournament")]
#[inline]
fn current_fv_scale() -> i32 {
    crate::tournament::FV_SCALE
}

#[cfg(not(feature = "tournament"))]
#[inline]
fn current_fv_scale() -> i32 {
    FV_SCALE.load(Ordering::Relaxed)
}

pub fn forward(acc: &Accumulator, net: &NnueNetwork, stm: Color, bucket: usize) -> Value {
    debug_assert!(
        bucket < net.stacks.len(),
        "bucket {} out of range for {}-stack network",
        bucket,
        net.stacks.len()
    );
    let stack = &net.stacks[bucket];

    let fv_scale = current_fv_scale();

    let mut transformed = [0u8; FC_0_INPUT_DIMS];
    post_ft_kernel::transformer_ewm(acc, stm, &mut transformed);

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vnni"
    ))]
    let score = post_ft_kernel::fused_fc_chain(
        &transformed,
        &stack.fc_0_biases,
        &stack.fc_0_weights,
        &stack.fc_1_biases,
        &stack.fc_1_weights,
        &stack.fc_2_biases,
        &stack.fc_2_weights,
    );

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vnni"
    )))]
    let score = per_layer_flow(&transformed, stack);

    Value(score / fv_scale)
}

#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vnni"
)))]
fn per_layer_flow(transformed: &[u8; FC_0_INPUT_DIMS], stack: &NetworkStack) -> i32 {
    let mut fc_0_out = [0i32; FC_0_OUTPUT_DIMS];
    post_ft_kernel::affine(
        &mut fc_0_out,
        &stack.fc_0_biases,
        &stack.fc_0_weights,
        transformed,
        FC_0_INPUT_DIMS,
        FC_0_PADDED_INPUT_DIMS,
    );

    // Both activations use the first HIDDEN1_DIMS outputs; fc_0_out[HIDDEN1_DIMS] feeds only the shortcut.
    let mut ac_0 = [0u8; HIDDEN1_DIMS];
    post_ft_kernel::clipped_relu(&fc_0_out[..HIDDEN1_DIMS], &mut ac_0);
    let mut ac_sqr_0 = [0u8; HIDDEN1_DIMS];
    post_ft_kernel::sqr_clipped_relu(&fc_0_out[..HIDDEN1_DIMS], &mut ac_sqr_0);

    // fc_1 input layout: [SqrClippedReLU(15) | ClippedReLU(15) | 0_pad(2)].
    let mut fc_1_in = [0u8; FC_1_PADDED_INPUT_DIMS];
    fc_1_in[..HIDDEN1_DIMS].copy_from_slice(&ac_sqr_0);
    fc_1_in[HIDDEN1_DIMS..2 * HIDDEN1_DIMS].copy_from_slice(&ac_0);

    let mut fc_1_out = [0i32; FC_1_OUTPUT_DIMS];
    post_ft_kernel::affine(
        &mut fc_1_out,
        &stack.fc_1_biases,
        &stack.fc_1_weights,
        &fc_1_in,
        FC_1_INPUT_DIMS,
        FC_1_PADDED_INPUT_DIMS,
    );

    let mut ac_1 = [0u8; FC_1_OUTPUT_DIMS];
    post_ft_kernel::clipped_relu(&fc_1_out, &mut ac_1);

    let mut fc_2_out = [0i32; FC_2_OUTPUT_DIMS];
    post_ft_kernel::affine(
        &mut fc_2_out,
        &stack.fc_2_biases,
        &stack.fc_2_weights,
        &ac_1,
        FC_2_INPUT_DIMS,
        FC_2_PADDED_INPUT_DIMS,
    );

    // post-fc_2 shortcut: fc_0_out[HIDDEN1_DIMS] is raw (pre-ReLU) and can be negative.
    fc_2_out[0].wrapping_add(fc_0_out[HIDDEN1_DIMS])
}

#[cfg(test)]
mod tests {
    use super::super::aligned::Aligned64;
    use super::super::types::{
        FC_0_OUTPUT_DIMS, FC_0_PADDED_INPUT_DIMS, FC_1_OUTPUT_DIMS, FC_1_PADDED_INPUT_DIMS, FC_2_OUTPUT_DIMS,
        FC_2_PADDED_INPUT_DIMS, HIDDEN_SIZE, HIDDEN1_DIMS, NetHeader, NetworkStack,
    };
    use super::*;

    fn empty_network_with_stack(stack: NetworkStack) -> NnueNetwork {
        NnueNetwork {
            header: NetHeader {
                version: 0,
                hash: 0,
                arch_id: "synthetic".to_string(),
            },
            ft_biases: Aligned64::zeroed(0),
            ft_weights: Aligned64::zeroed(0),
            stacks: vec![stack],
            sha256: [0u8; 32],
        }
    }

    fn zero_stack() -> NetworkStack {
        NetworkStack {
            fc_0_biases: Aligned64::zeroed(FC_0_OUTPUT_DIMS),
            fc_0_weights: Aligned64::zeroed(FC_0_OUTPUT_DIMS * FC_0_PADDED_INPUT_DIMS),
            fc_1_biases: Aligned64::zeroed(FC_1_OUTPUT_DIMS),
            fc_1_weights: Aligned64::zeroed(FC_1_OUTPUT_DIMS * FC_1_PADDED_INPUT_DIMS),
            fc_2_biases: Aligned64::zeroed(FC_2_OUTPUT_DIMS),
            fc_2_weights: Aligned64::zeroed(FC_2_OUTPUT_DIMS * FC_2_PADDED_INPUT_DIMS),
        }
    }

    #[test]
    fn zero_input_zero_weights_returns_zero() {
        let net = empty_network_with_stack(zero_stack());
        let acc = Accumulator::zeroed();
        assert_eq!(forward(&acc, &net, Color::BLACK, 0), Value(0));
        assert_eq!(forward(&acc, &net, Color::WHITE, 0), Value(0));
    }

    #[test]
    fn bucket_indexing_picks_the_right_stack() {
        let stacks: Vec<NetworkStack> = (0..super::super::types::LAYER_STACKS).map(|_| zero_stack()).collect();
        let net = NnueNetwork {
            header: NetHeader {
                version: 0,
                hash: 0,
                arch_id: "synthetic".to_string(),
            },
            ft_biases: Aligned64::zeroed(0),
            ft_weights: Aligned64::zeroed(0),
            stacks,
            sha256: [0u8; 32],
        };
        let acc = Accumulator::zeroed();
        for bucket in 0..super::super::types::LAYER_STACKS {
            assert_eq!(
                forward(&acc, &net, Color::BLACK, bucket),
                Value(0),
                "bucket {bucket} on a zero network must return Value(0)"
            );
        }
    }

    #[test]
    fn classic_chain_with_ewm_and_shortcut_propagates_into_value() {
        // FV_SCALE is process-wide; serialise with set_fv_scale_changes_forward_divisor.
        let _guard = super::super::TEST_MUTEX.lock().expect("TEST_MUTEX poisoned");
        set_fv_scale(16);

        let mut stack = zero_stack();

        // EWM lane 0 → transformed[0]=39; fc_0_out[0]=167, fc_0_out[15]=256 (shortcut).
        stack.fc_0_biases[0] = 50;
        stack.fc_0_weights[0] = 3;
        stack.fc_0_biases[HIDDEN1_DIMS] = 100;
        stack.fc_0_weights[HIDDEN1_DIMS * FC_0_PADDED_INPUT_DIMS] = 4;

        // ac_0[0] = 2; ac_sqr_0 all 0. fc_1_in[15] = 2 → fc_1_out[0] = 30+14 = 44.
        stack.fc_1_biases[0] = 30;
        stack.fc_1_weights[15] = 7;

        // ac_1 all 0 → fc_2_out[0] = 1_000 + shortcut 256 = 1_256; /16 = 78.
        stack.fc_2_biases[0] = 1_000;

        let net = empty_network_with_stack(stack);

        let mut acc = Accumulator::zeroed();
        acc.us[0] = 200;
        acc.us[768] = 100;
        acc.computed = true;

        assert_eq!(forward(&acc, &net, Color::BLACK, 0), Value(78));
    }

    #[test]
    fn forward_is_deterministic() {
        let _guard = super::super::TEST_MUTEX.lock().expect("TEST_MUTEX poisoned");
        set_fv_scale(16);

        let mut stack = zero_stack();
        for (i, b) in stack.fc_0_biases.iter_mut().enumerate() {
            *b = (i as i32) * 10 - 50;
        }
        for (i, w) in stack.fc_0_weights.iter_mut().enumerate() {
            *w = ((i as i32) % 7 - 3) as i8;
        }
        for (i, b) in stack.fc_1_biases.iter_mut().enumerate() {
            *b = (i as i32) * 3 - 20;
        }
        for (i, w) in stack.fc_1_weights.iter_mut().enumerate() {
            *w = ((i as i32) % 5 - 2) as i8;
        }
        stack.fc_2_biases[0] = 42;
        for (i, w) in stack.fc_2_weights.iter_mut().enumerate() {
            *w = ((i as i32) % 11 - 5) as i8;
        }
        let net = empty_network_with_stack(stack);

        let mut acc = Accumulator::zeroed();
        for i in 0..HIDDEN_SIZE {
            acc.us[i] = ((i as i32) % 301 - 150) as i16;
            acc.them[i] = ((i as i32) % 257 - 128) as i16;
        }
        acc.computed = true;

        let a = forward(&acc, &net, Color::BLACK, 0);
        let b = forward(&acc, &net, Color::BLACK, 0);
        assert_eq!(a, b);

        let c = forward(&acc, &net, Color::WHITE, 0);
        let d = forward(&acc, &net, Color::WHITE, 0);
        assert_eq!(c, d);
    }

    // The runtime FV_SCALE global steers `forward` only when the tournament feature is off.
    #[cfg(not(feature = "tournament"))]
    #[test]
    fn set_fv_scale_changes_forward_divisor() {
        let _guard = super::super::TEST_MUTEX.lock().expect("TEST_MUTEX poisoned");

        let mut stack = zero_stack();
        // Same wiring as classic_chain_…: post-shortcut score is 1_256.
        stack.fc_0_biases[0] = 50;
        stack.fc_0_weights[0] = 3;
        stack.fc_0_biases[HIDDEN1_DIMS] = 100;
        stack.fc_0_weights[HIDDEN1_DIMS * FC_0_PADDED_INPUT_DIMS] = 4;
        stack.fc_1_biases[0] = 30;
        stack.fc_1_weights[15] = 7;
        stack.fc_2_biases[0] = 1_000;

        let net = empty_network_with_stack(stack);

        let mut acc = Accumulator::zeroed();
        acc.us[0] = 200;
        acc.us[768] = 100;
        acc.computed = true;

        for (divisor, expected) in [(16i32, 78i32), (8, 157), (32, 39), (1, 1_256), (128, 9)] {
            set_fv_scale(divisor);
            assert_eq!(
                forward(&acc, &net, Color::BLACK, 0),
                Value(expected),
                "FV_SCALE = {divisor} should produce 1_256 / {divisor} = {expected}",
            );
        }

        set_fv_scale(16);
    }

    // Under `tournament`, `forward` divides by the compile-time const, not the runtime global (unaffected by set_fv_scale).
    #[cfg(feature = "tournament")]
    #[test]
    fn forward_uses_compile_time_fv_scale_const() {
        // `const` context: this only compiles if FV_SCALE is genuinely a compile-time const.
        const _: i32 = crate::tournament::FV_SCALE;
        assert_eq!(crate::tournament::FV_SCALE, 16, "v1 tournament config bakes FV_SCALE = 16");

        let _guard = super::super::TEST_MUTEX.lock().expect("TEST_MUTEX poisoned");

        let mut stack = zero_stack();
        // Same wiring as classic_chain_…: post-shortcut score is 1_256.
        stack.fc_0_biases[0] = 50;
        stack.fc_0_weights[0] = 3;
        stack.fc_0_biases[HIDDEN1_DIMS] = 100;
        stack.fc_0_weights[HIDDEN1_DIMS * FC_0_PADDED_INPUT_DIMS] = 4;
        stack.fc_1_biases[0] = 30;
        stack.fc_1_weights[15] = 7;
        stack.fc_2_biases[0] = 1_000;

        let net = empty_network_with_stack(stack);

        let mut acc = Accumulator::zeroed();
        acc.us[0] = 200;
        acc.us[768] = 100;
        acc.computed = true;

        // 1_256 / 16 = 78 — even after set_fv_scale tries (and fails) to change the divisor.
        set_fv_scale(1);
        assert_eq!(
            forward(&acc, &net, Color::BLACK, 0),
            Value(78),
            "tournament build must ignore the runtime global and divide by the const FV_SCALE",
        );
    }
}
