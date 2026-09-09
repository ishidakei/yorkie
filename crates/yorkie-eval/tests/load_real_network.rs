//! Integration test: the evaluation file this build wrote holds a whole
//! SFNN-1536 network, laid out the way the kernels read it.
//!
//! A checkout with no network staged has no such file, so the test prints a
//! notice and passes.

use yorkie_eval::{HIDDEN_SIZE, LAYER_STACKS, NUM_FEATURES};

mod common;

#[cfg_attr(miri, ignore)]
#[test]
fn the_evaluation_file_this_build_wrote_holds_the_whole_network() {
    let Some(net) = common::engine_network() else {
        return;
    };

    assert_eq!(net.stacks.len(), LAYER_STACKS, "layer-stack count");
    assert_eq!(net.ft_biases.len(), HIDDEN_SIZE, "ft bias count");
    assert_eq!(
        net.ft_weights.len(),
        HIDDEN_SIZE * NUM_FEATURES,
        "ft weight count"
    );

    // The SIMD kernels need every buffer on a 64-byte boundary.
    let is_aligned = |ptr: *const u8| (ptr as usize).is_multiple_of(64);
    assert!(is_aligned(net.ft_biases.as_ptr() as *const u8));
    assert!(is_aligned(net.ft_weights.as_ptr() as *const u8));
    for (i, stack) in net.stacks.iter().enumerate() {
        assert!(
            is_aligned(stack.fc_0_biases.as_ptr() as *const u8),
            "stack[{i}].fc_0_biases not 64-byte aligned"
        );
        assert!(
            is_aligned(stack.fc_0_weights.as_ptr() as *const u8),
            "stack[{i}].fc_0_weights not 64-byte aligned"
        );
        assert!(
            is_aligned(stack.fc_1_biases.as_ptr() as *const u8),
            "stack[{i}].fc_1_biases not 64-byte aligned"
        );
        assert!(
            is_aligned(stack.fc_1_weights.as_ptr() as *const u8),
            "stack[{i}].fc_1_weights not 64-byte aligned"
        );
        assert!(
            is_aligned(stack.fc_2_biases.as_ptr() as *const u8),
            "stack[{i}].fc_2_biases not 64-byte aligned"
        );
        assert!(
            is_aligned(stack.fc_2_weights.as_ptr() as *const u8),
            "stack[{i}].fc_2_weights not 64-byte aligned"
        );
    }
}
