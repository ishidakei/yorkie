//! Layer-stack forward pass, bucket selection, and the public `evaluate` entry
//! point.
//!
//! Ported from `Network::Propagate` (`sfnn-1536.h`) and `ComputeScore` /
//! `stack_index_for_nnue` (`evaluate_nnue.cpp`). The accumulator lives in
//! [`crate::transformer`], the kernels in [`crate::simd`].
//!
//! The prose here follows the reference's naming: the **FT layer** is the
//! feature transformer, and **L1 / L2 / L3** are the dense layers after it.
//! The FT layer is never called "L1". In the identifiers below, L1 is `fc_0`,
//! L2 is `fc_1` and L3 is `fc_2`.
//!
//! L1 is a sparse-input affine with 16 outputs. The first 15 feed two
//! activations whose results concatenate into L2's input; the 16th is raw,
//! pre-ReLU, and is added to L3's single output as a shortcut term. The network
//! output is then divided by the compiled-in [`FV_SCALE`].

use yorkie_state::{Color, Position, Square};

use crate::features::king_square;
use crate::simd::post_ft_kernel;
use crate::transformer::{Accumulator, FT_OUTPUT_DIMS};
use crate::types::{
    FC_0_INPUT_DIMS, FC_0_OUTPUT_DIMS, FC_0_PADDED_INPUT_DIMS, FC_1_INPUT_DIMS, FC_1_OUTPUT_DIMS,
    FC_1_PADDED_INPUT_DIMS, FC_2_INPUT_DIMS, FC_2_OUTPUT_DIMS, FC_2_PADDED_INPUT_DIMS,
    HIDDEN1_DIMS, LAYER_STACKS, NetworkStack, NnueNetwork,
};

/// The fixed-point scale applied to the network output to produce the final
/// score — the reference's mutable global `NNUE::FV_SCALE`, compiled in from
/// the `fv_scale` config key. There is no setter: the engine carries no runtime
/// configuration.
pub const FV_SCALE: i32 = crate::config::FV_SCALE as i32;

// The build script range-checks `fv_scale`; this restates the one bound the
// division depends on, in a `const` context.
const _: () = assert!(FV_SCALE >= 1, "FV_SCALE must be at least 1");

/// `k*ToIndex` tables from `stack_index_for_nnue`: the own-king rank contributes
/// the coarse third `{0,3,6}`, the enemy-king rank the fine third `{0,1,2}`.
const F_RANK_TO_INDEX: [usize; 9] = [0, 0, 0, 3, 3, 3, 6, 6, 6];
const E_RANK_TO_INDEX: [usize; 9] = [0, 0, 0, 1, 1, 1, 2, 2, 2];

/// Select the layer-stack index for `pos` (`stack_index_for_nnue`). The
/// own-king rank is taken in the side-to-move's forward frame and the
/// enemy-king rank in the opposite one.
///
/// # Panics
/// Panics if either king is missing.
pub fn layer_stack_index(pos: &Position) -> usize {
    let stm = pos.side_to_move();
    let f_king = king_square(pos, stm);
    let e_king = king_square(pos, stm.flip());

    // White flips its ranks, so both colours reason in an own-side-forward
    // frame.
    let flip = |sq: Square| (Square::RANKS - 1 - sq.rank()) as usize;
    let f_rank = match stm {
        Color::Black => f_king.rank() as usize,
        Color::White => flip(f_king),
    };
    let e_rank = match stm {
        Color::Black => flip(e_king),
        Color::White => e_king.rank() as usize,
    };

    (F_RANK_TO_INDEX[f_rank] + E_RANK_TO_INDEX[e_rank]).min(LAYER_STACKS - 1)
}

/// Static NNUE evaluation of `pos` from the side-to-move perspective, positive
/// meaning that side is better. Rebuilds both accumulator halves from scratch
/// (`Eval::evaluate`).
///
/// The network is passed explicitly: there is no global registry, and the
/// Evaluation layer never reaches up to Protocol for the loaded parameters.
///
/// # Panics
/// Panics if `pos` is missing either king.
pub fn evaluate(net: &NnueNetwork, pos: &Position) -> i32 {
    let mut acc = Accumulator::new();
    acc.refresh(net, pos);
    evaluate_with(net, &acc, pos)
}

/// [`evaluate`] consuming an already-computed accumulator instead of
/// refreshing. `acc` must be the accumulator for `pos`, both perspectives
/// current; `pos` still supplies the side to move and the king ranks.
///
/// # Panics
/// Panics if `pos` is missing either king.
pub fn evaluate_with(net: &NnueNetwork, acc: &Accumulator, pos: &Position) -> i32 {
    let bucket = layer_stack_index(pos);
    debug_assert!(bucket < net.stacks.len());

    let mut transformed = [0u8; FT_OUTPUT_DIMS];
    acc.output_transform(pos.side_to_move(), &mut transformed);

    let score = per_layer_flow(&transformed, &net.stacks[bucket]);
    // The one site that consumes `FV_SCALE`.
    score / FV_SCALE
}

/// Layer-stack forward pass over the transformed byte buffer, returning the raw
/// network output before [`FV_SCALE`].
///
/// A build with AVX-512 VNNI runs the whole chain through one fused kernel;
/// otherwise the per-layer flow below runs. The two are byte-for-byte
/// equivalent.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vnni"
))]
fn per_layer_flow(transformed: &[u8; FC_0_INPUT_DIMS], stack: &NetworkStack) -> i32 {
    // Imported here rather than at module scope, where the `use` would be
    // unused on a non-VNNI build.
    use crate::simd;

    // SAFETY: this arm is compiled only into a build enabling exactly the
    // features `fused_fc_chain`'s `#[target_feature]` names, and such a build
    // is `-C target-cpu=native`, so it only ever runs on a host providing them.
    // The stack slices carry the layer-stack shapes the kernel expects.
    unsafe {
        simd::avx512_post_ft::fused_fc_chain(
            transformed,
            &stack.fc_0_biases,
            &stack.fc_0_weights,
            &stack.fc_1_biases,
            &stack.fc_1_weights,
            &stack.fc_2_biases,
            &stack.fc_2_weights,
        )
    }
}

/// Layer-stack forward pass — non-VNNI arm (the VNNI arm above carries the
/// documentation).
#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vnni"
)))]
fn per_layer_flow(transformed: &[u8; FC_0_INPUT_DIMS], stack: &NetworkStack) -> i32 {
    per_layer_flow_unfused(transformed, stack)
}

/// The unfused layer-stack forward pass. Compiled unconditionally so the tests
/// can hold it against the fused chain, which is why a VNNI build needs the
/// `allow(dead_code)`.
#[allow(dead_code)]
fn per_layer_flow_unfused(transformed: &[u8; FC_0_INPUT_DIMS], stack: &NetworkStack) -> i32 {
    let mut fc_0_out = [0i32; FC_0_OUTPUT_DIMS];
    post_ft_kernel::affine(
        &mut fc_0_out,
        &stack.fc_0_biases,
        &stack.fc_0_weights,
        transformed,
        FC_0_INPUT_DIMS,
        FC_0_PADDED_INPUT_DIMS,
    );

    // `fc_0_out[HIDDEN1_DIMS]` is reserved for the post-`fc_2` shortcut.
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

    // The shortcut term is raw, pre-ReLU, and can be negative.
    fc_2_out[0].wrapping_add(fc_0_out[HIDDEN1_DIMS])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{NetDims, NetHeader, NnueNetworkBuilder};
    use yorkie_state::parse_sfen;

    fn synthetic_header() -> NetHeader {
        NetHeader {
            version: 0,
            hash: 0,
            arch_id: "synthetic".to_string(),
        }
    }

    /// A single-stack builder with the standard FC dims but a one-feature
    /// transformer, so the FC forward pass runs without a 215 MiB allocation.
    fn builder_1stack_std() -> NnueNetworkBuilder {
        let dims = NetDims {
            layer_stacks: 1,
            num_features: 1,
            ..NetDims::STANDARD
        };
        NnueNetworkBuilder::with_dims(synthetic_header(), [0u8; 32], &dims)
    }

    /// A single-stack, all-zero network with the standard FC dims (tiny FT).
    fn zero_net_1stack() -> NnueNetwork {
        builder_1stack_std().build()
    }

    #[test]
    fn replicate_is_byte_identical_deep_copy() {
        // A per-NUMA-node replica must be byte-for-byte identical to its source
        // in a freshly allocated arena, so the two evaluate alike while living
        // on different nodes.
        let dims = NetDims {
            layer_stacks: 1,
            num_features: 1,
            hidden_size: 5,
            ..NetDims::STANDARD
        };
        let mut builder = NnueNetworkBuilder::with_dims(synthetic_header(), [0xAB; 32], &dims);
        builder.fc_0_biases_mut(0)[0] = 123;
        builder.fc_0_weights_mut(0)[1] = -7;
        builder.fc_1_biases_mut(0)[2] = 55;
        builder.fc_2_weights_mut(0)[0] = 9;
        builder.ft_biases_mut()[0] = 4242;
        builder
            .ft_weights_mut()
            .copy_from_slice(&[1i16, 2, 3, -4, 5]);
        let net = builder.build();

        let copy = net.replicate();

        assert_eq!(&*copy.ft_biases, &*net.ft_biases);
        assert_eq!(&*copy.ft_weights, &*net.ft_weights);
        assert_eq!(copy.sha256, net.sha256);
        assert_eq!(copy.header.arch_id, net.header.arch_id);
        assert_eq!(copy.stacks.len(), net.stacks.len());
        assert_eq!(&*copy.stacks[0].fc_0_biases, &*net.stacks[0].fc_0_biases);
        assert_eq!(&*copy.stacks[0].fc_0_weights, &*net.stacks[0].fc_0_weights);
        assert_eq!(&*copy.stacks[0].fc_2_weights, &*net.stacks[0].fc_2_weights);
        assert_ne!(copy.ft_weights.as_ptr(), net.ft_weights.as_ptr());
        assert_eq!(net.allocation_disclosure().0, 1);
        assert_eq!(copy.allocation_disclosure().0, 1);
    }

    #[test]
    fn zero_network_evaluates_to_zero() {
        // `per_layer_flow` ignores the feature transformer.
        let net = zero_net_1stack();
        let transformed = [0u8; FC_0_INPUT_DIMS];
        assert_eq!(per_layer_flow(&transformed, &net.stacks[0]), 0);
    }

    #[test]
    fn forward_wires_shortcut_and_chain() {
        // The reference's `classic_chain_with_ewm_and_shortcut` wiring:
        // transformed[0]=39 drives fc_0_out[0]=50+3*39=167 and the shortcut
        // fc_0_out[15]=100+4*39=256. ac_0[0]=167>>6=2, ac_sqr_0 all 0.
        // fc_1_in[15]=2 -> fc_1_out[0]=30+7*2=44; ac_1[0]=44>>6=0.
        // fc_2_out[0]=1000+256=1256.
        let mut b = builder_1stack_std();
        b.fc_0_biases_mut(0)[0] = 50;
        b.fc_0_weights_mut(0)[0] = 3;
        b.fc_0_biases_mut(0)[HIDDEN1_DIMS] = 100;
        b.fc_0_weights_mut(0)[HIDDEN1_DIMS * FC_0_PADDED_INPUT_DIMS] = 4;
        b.fc_1_biases_mut(0)[0] = 30;
        b.fc_1_weights_mut(0)[15] = 7;
        b.fc_2_biases_mut(0)[0] = 1_000;
        let net = b.build();
        let stack = &net.stacks[0];

        let mut transformed = [0u8; FC_0_INPUT_DIMS];
        transformed[0] = 39;

        assert_eq!(per_layer_flow(&transformed, stack), 1_256);
        // The whole-pipeline division by the configured `FV_SCALE`.
        assert_eq!(per_layer_flow(&transformed, stack) / FV_SCALE, 78);
    }

    #[test]
    fn selected_forward_path_matches_the_unfused_reference() {
        // On a VNNI build this pits the fused chain against the unfused one; on
        // any other build the two are the same code, and the check merely keeps
        // the unfused form exercised.
        let mut b = builder_1stack_std();
        for (i, w) in b.fc_0_weights_mut(0).iter_mut().enumerate() {
            *w = ((i as i32 * 7) % 61 - 30) as i8;
        }
        for (i, bias) in b.fc_0_biases_mut(0).iter_mut().enumerate() {
            *bias = (i as i32 * 37) % 4_001 - 2_000;
        }
        for (i, w) in b.fc_1_weights_mut(0).iter_mut().enumerate() {
            *w = ((i as i32 * 11) % 53 - 26) as i8;
        }
        for (i, bias) in b.fc_1_biases_mut(0).iter_mut().enumerate() {
            *bias = (i as i32 * 101) % 3_001 - 1_500;
        }
        for (i, w) in b.fc_2_weights_mut(0).iter_mut().enumerate() {
            *w = ((i as i32 * 13) % 47 - 23) as i8;
        }
        for (i, bias) in b.fc_2_biases_mut(0).iter_mut().enumerate() {
            *bias = (i as i32 * 211) % 2_001 - 1_000;
        }
        let net = b.build();
        let stack = &net.stacks[0];

        for seed in [0u32, 1, 97] {
            let mut transformed = [0u8; FC_0_INPUT_DIMS];
            for (i, slot) in transformed.iter_mut().enumerate() {
                *slot = ((i as u32).wrapping_mul(11).wrapping_add(seed) % 256) as u8;
            }
            assert_eq!(
                per_layer_flow(&transformed, stack),
                per_layer_flow_unfused(&transformed, stack),
                "selected path disagrees with the unfused reference (seed {seed})",
            );
        }
    }

    #[test]
    fn evaluate_zero_network_is_zero_for_both_sides() {
        // Full-size zeroed FT weights, so the refresh's feature-column indexing
        // stays in bounds; the sparse positions below touch few pages.
        let net = NnueNetworkBuilder::new(synthetic_header(), [0u8; 32]).build();
        for sfen in ["8K/9/9/9/9/9/9/9/8k b - 1", "8K/9/9/9/9/9/9/9/8k w - 1"] {
            let pos = parse_sfen(sfen).unwrap();
            assert_eq!(evaluate(&net, &pos), 0, "sfen `{sfen}`");
        }
    }

    /// An SFEN-parsing form of the bucket index, sharing no code with
    /// [`layer_stack_index`].
    fn oracle_bucket(sfen: &str) -> usize {
        let mut parts = sfen.split_whitespace();
        let board = parts.next().unwrap();
        let stm = parts.next().unwrap();

        let mut bk = None;
        let mut wk = None;
        for (rank_idx, rank_str) in board.split('/').enumerate() {
            let mut chars = rank_str.chars().peekable();
            while let Some(c) = chars.next() {
                match c {
                    '1'..='9' => {}
                    '+' => {
                        let _ = chars.next();
                    }
                    'K' => bk = Some(rank_idx),
                    'k' => wk = Some(rank_idx),
                    _ => {}
                }
            }
        }
        let bk = bk.unwrap();
        let wk = wk.unwrap();
        let (f_rank, e_rank) = match stm {
            "b" => (bk, 8 - wk),
            "w" => (8 - wk, bk),
            _ => unreachable!(),
        };
        (F_RANK_TO_INDEX[f_rank] + E_RANK_TO_INDEX[e_rank]).min(LAYER_STACKS - 1)
    }

    const BUCKET_FIXTURES: &[(usize, &str)] = &[
        (0, "8K/9/9/9/9/9/9/9/8k b - 1"),
        (1, "8K/9/9/9/9/8k/9/9/9 b - 1"),
        (2, "8K/9/k8/9/9/9/9/9/9 b - 1"),
        (3, "9/9/9/8K/9/9/9/9/8k b - 1"),
        (4, "9/9/9/8K/9/8k/9/9/9 b - 1"),
        (5, "9/9/k8/8K/9/9/9/9/9 b - 1"),
        (6, "9/9/9/9/9/9/8K/9/8k b - 1"),
        (7, "9/9/9/9/9/k8/8K/9/9 b - 1"),
        (8, "9/9/8k/9/9/9/8K/9/9 b - 1"),
    ];

    #[test]
    fn layer_stack_index_matches_oracle_on_each_bucket() {
        for (expected, sfen) in BUCKET_FIXTURES {
            let oracle = oracle_bucket(sfen);
            assert_eq!(oracle, *expected, "oracle disagrees on `{sfen}`");
            let pos = parse_sfen(sfen).unwrap();
            assert_eq!(
                layer_stack_index(&pos),
                *expected,
                "layer_stack_index mismatch on `{sfen}`"
            );
        }
    }

    #[test]
    fn layer_stack_index_white_to_move_mirrors_ranks() {
        let sfen = "8k/9/9/9/9/9/9/9/8K w - 1";
        assert_eq!(oracle_bucket(sfen), 8);
        assert_eq!(layer_stack_index(&parse_sfen(sfen).unwrap()), 8);
    }

    #[test]
    fn layer_stack_index_startpos_is_eight() {
        let startpos = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
        assert_eq!(oracle_bucket(startpos), 8);
        assert_eq!(layer_stack_index(&parse_sfen(startpos).unwrap()), 8);
    }

    #[test]
    fn layer_stack_index_is_always_in_range() {
        for (_, sfen) in BUCKET_FIXTURES {
            let got = layer_stack_index(&parse_sfen(sfen).unwrap());
            assert!(got < LAYER_STACKS);
        }
    }
}
