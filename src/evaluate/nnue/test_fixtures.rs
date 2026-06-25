use std::sync::{Arc, OnceLock};

use super::aligned::Aligned64;
use super::types::{
    FC_0_OUTPUT_DIMS, FC_0_PADDED_INPUT_DIMS, FC_1_OUTPUT_DIMS, FC_1_PADDED_INPUT_DIMS, FC_2_OUTPUT_DIMS, FC_2_PADDED_INPUT_DIMS,
    HIDDEN_SIZE, LAYER_STACKS, NUM_FEATURES, NetHeader, NetworkStack, NnueNetwork,
};

pub(crate) fn run_with_large_stack<F>(f: F)
where
    F: FnOnce() + Send + 'static,
{
    std::thread::Builder::new()
        .stack_size(crate::stack_size::STACK_SIZE)
        .spawn(f)
        .expect("spawn test thread with enlarged stack")
        .join()
        .expect("test thread panicked");
}

static SYNTHETIC_NET: OnceLock<Arc<NnueNetwork>> = OnceLock::new();

fn synthetic_net_shared() -> &'static Arc<NnueNetwork> {
    SYNTHETIC_NET.get_or_init(|| Arc::new(build_synthetic_net()))
}

pub(crate) fn synthetic_net() -> &'static NnueNetwork {
    synthetic_net_shared().as_ref()
}

pub(crate) fn synthetic_net_arc() -> Arc<NnueNetwork> {
    Arc::clone(synthetic_net_shared())
}

fn build_synthetic_net() -> NnueNetwork {
    let mut ft_biases = Aligned64::<i16>::zeroed(HIDDEN_SIZE);
    for (i, slot) in ft_biases.iter_mut().enumerate() {
        *slot = ((i as i32) % 17 - 8) as i16;
    }

    let mut ft_weights = Aligned64::<i16>::zeroed(HIDDEN_SIZE * NUM_FEATURES);
    for idx in 0..NUM_FEATURES {
        let base = idx * HIDDEN_SIZE;
        for i in 0..HIDDEN_SIZE {
            let mix = ((idx as i32).wrapping_mul(31).wrapping_add(i as i32 * 7)) % 23 - 11;
            ft_weights[base + i] = mix as i16;
        }
    }

    let stacks = (0..LAYER_STACKS)
        .map(|_| NetworkStack {
            fc_0_biases: Aligned64::zeroed(FC_0_OUTPUT_DIMS),
            fc_0_weights: Aligned64::zeroed(FC_0_OUTPUT_DIMS * FC_0_PADDED_INPUT_DIMS),
            fc_1_biases: Aligned64::zeroed(FC_1_OUTPUT_DIMS),
            fc_1_weights: Aligned64::zeroed(FC_1_OUTPUT_DIMS * FC_1_PADDED_INPUT_DIMS),
            fc_2_biases: Aligned64::zeroed(FC_2_OUTPUT_DIMS),
            fc_2_weights: Aligned64::zeroed(FC_2_OUTPUT_DIMS * FC_2_PADDED_INPUT_DIMS),
        })
        .collect();

    NnueNetwork {
        header: NetHeader {
            version: 0,
            hash: 0,
            arch_id: "synthetic".to_string(),
        },
        ft_biases,
        ft_weights,
        stacks,
        sha256: [0u8; 32],
    }
}
