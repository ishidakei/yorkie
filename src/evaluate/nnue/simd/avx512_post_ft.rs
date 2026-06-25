use std::arch::x86_64::{
    __m128i, __m256i, __m512i, _mm_storeu_si128, _mm_unpacklo_epi64, _mm256_loadu_si256, _mm256_storeu_si256,
    _mm512_castsi512_si256, _mm512_cvtepi16_epi8, _mm512_cvtepi32_epi8, _mm512_cvtepi32_epi64, _mm512_cvtepi64_epi8,
    _mm512_dpbusd_epi32, _mm512_extracti64x4_epi64, _mm512_loadu_si512, _mm512_max_epi16, _mm512_max_epi32, _mm512_min_epi16,
    _mm512_min_epi32, _mm512_min_epi64, _mm512_mul_epi32, _mm512_mulhi_epi16, _mm512_reduce_add_epi32, _mm512_set1_epi16,
    _mm512_set1_epi32, _mm512_set1_epi64, _mm512_setzero_si512, _mm512_slli_epi16, _mm512_srai_epi32, _mm512_srai_epi64,
    _mm512_zextsi256_si512,
};

use super::super::types::{
    Accumulator, FC_0_INPUT_DIMS, FC_0_OUTPUT_DIMS, FC_0_PADDED_INPUT_DIMS, FC_1_OUTPUT_DIMS, FC_1_PADDED_INPUT_DIMS,
    FC_2_OUTPUT_DIMS, FC_2_PADDED_INPUT_DIMS, HIDDEN_SIZE, HIDDEN1_DIMS,
};
use super::scalar_post_ft;
use crate::types::Color;

const VNNI_LANES: usize = 64;
const I32_LANES_PER_M512: usize = 16;
const I16_LANES_PER_M512: usize = 32;
const EWM_HALF: usize = HIDDEN_SIZE / 2;

// Mirror scalar_post_ft constants as u32 (AVX-512 shift intrinsics take IMM8: u32).
const WEIGHT_SCALE_BITS: u32 = 6;
const SQR_SHIFT: u32 = 19;
const EWM_CLAMP_I16: i16 = 127 * 2;

// EWM mulhi trick: mulhi(sum0 << 7, sum1) == (sum0 * sum1) >> 9 on the i16 domain.
const EWM_PRESHIFT: u32 = 7;

#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn transformer_ewm(acc: &Accumulator, stm: Color, out: &mut [u8; FC_0_INPUT_DIMS]) {
    debug_assert_eq!(FC_0_INPUT_DIMS, HIDDEN_SIZE);
    let (stm_half, other_half): (&[i16; HIDDEN_SIZE], &[i16; HIDDEN_SIZE]) = if stm == Color::BLACK {
        (&acc.us, &acc.them)
    } else {
        (&acc.them, &acc.us)
    };
    // SAFETY: each helper reads 2*EWM_HALF i16 and writes EWM_HALF u8 to disjoint out ranges.
    unsafe {
        transformer_ewm_one_perspective(stm_half.as_ptr(), out.as_mut_ptr());
        transformer_ewm_one_perspective(other_half.as_ptr(), out.as_mut_ptr().add(EWM_HALF));
    }
}

// VNNI fast path needs in_dims >= 64 and %64 == 0; fc_1/fc_2 (30/32) fall back to scalar.
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub unsafe fn affine(output: &mut [i32], biases: &[i32], weights: &[i8], input: &[u8], in_dims: usize, padded_in: usize) {
    debug_assert_eq!(output.len(), biases.len());
    debug_assert_eq!(weights.len(), output.len() * padded_in);
    debug_assert!(in_dims <= padded_in);
    debug_assert!(input.len() >= in_dims);

    if in_dims < VNNI_LANES || !in_dims.is_multiple_of(VNNI_LANES) {
        scalar_post_ft::affine(output, biases, weights, input, in_dims, padded_in);
        return;
    }

    // SAFETY: module-level cfg ensures F+BW+VNNI.
    unsafe { affine_avx512_vnni(output, biases, weights, input, in_dims, padded_in) }
}

#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn clipped_relu(input: &[i32], output: &mut [u8]) {
    debug_assert_eq!(input.len(), output.len());
    if input.len() < I32_LANES_PER_M512 || !input.len().is_multiple_of(I32_LANES_PER_M512) {
        scalar_post_ft::clipped_relu(input, output);
        return;
    }
    // SAFETY: module-level cfg ensures F+BW.
    unsafe { clipped_relu_kernel(input, output) }
}

#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn sqr_clipped_relu(input: &[i32], output: &mut [u8]) {
    debug_assert_eq!(input.len(), output.len());
    if input.len() < I32_LANES_PER_M512 || !input.len().is_multiple_of(I32_LANES_PER_M512) {
        scalar_post_ft::sqr_clipped_relu(input, output);
        return;
    }
    // SAFETY: module-level cfg ensures F+BW.
    unsafe { sqr_clipped_relu_kernel(input, output) }
}

// Full fc_0→ReLU/SqrReLU→fc_1→ReLU→fc_2→shortcut in one target_feature body.
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub unsafe fn fused_fc_chain(
    transformed: &[u8; FC_0_INPUT_DIMS],
    fc_0_biases: &[i32],
    fc_0_weights: &[i8],
    fc_1_biases: &[i32],
    fc_1_weights: &[i8],
    fc_2_biases: &[i32],
    fc_2_weights: &[i8],
) -> i32 {
    debug_assert_eq!(fc_0_biases.len(), FC_0_OUTPUT_DIMS);
    debug_assert_eq!(fc_0_weights.len(), FC_0_OUTPUT_DIMS * FC_0_PADDED_INPUT_DIMS);
    debug_assert_eq!(fc_1_biases.len(), FC_1_OUTPUT_DIMS);
    debug_assert_eq!(fc_1_weights.len(), FC_1_OUTPUT_DIMS * FC_1_PADDED_INPUT_DIMS);
    debug_assert_eq!(fc_2_biases.len(), FC_2_OUTPUT_DIMS);
    debug_assert_eq!(fc_2_weights.len(), FC_2_OUTPUT_DIMS * FC_2_PADDED_INPUT_DIMS);

    let mut fc_0_out = [0i32; FC_0_OUTPUT_DIMS];
    // SAFETY: module-level cfg ensures F+BW+VNNI.
    unsafe {
        affine_avx512_vnni(
            &mut fc_0_out,
            fc_0_biases,
            fc_0_weights,
            transformed,
            FC_0_INPUT_DIMS,
            FC_0_PADDED_INPUT_DIMS,
        );
    }

    // fc_0_out[HIDDEN1_DIMS] is the shortcut; only the first HIDDEN1_DIMS lanes feed activations (scalar).
    let mut ac_0 = [0u8; HIDDEN1_DIMS];
    let mut ac_sqr_0 = [0u8; HIDDEN1_DIMS];
    scalar_post_ft::clipped_relu(&fc_0_out[..HIDDEN1_DIMS], &mut ac_0);
    scalar_post_ft::sqr_clipped_relu(&fc_0_out[..HIDDEN1_DIMS], &mut ac_sqr_0);

    // fc_1_in = [ac_sqr_0(15) | ac_0(15) | 0_pad(2)]; the pad keeps VNNI padding lanes safe.
    let mut fc_1_in = [0u8; FC_1_PADDED_INPUT_DIMS];
    fc_1_in[..HIDDEN1_DIMS].copy_from_slice(&ac_sqr_0);
    fc_1_in[HIDDEN1_DIMS..2 * HIDDEN1_DIMS].copy_from_slice(&ac_0);

    let mut fc_1_out = [0i32; FC_1_OUTPUT_DIMS];
    // SAFETY: module-level cfg ensures F+BW+VNNI.
    unsafe { affine_padded32_avx512_vnni(&mut fc_1_out, fc_1_biases, fc_1_weights, &fc_1_in) };

    let mut ac_1 = [0u8; FC_1_OUTPUT_DIMS];
    // SAFETY: module-level cfg ensures F+BW; lengths match.
    unsafe { clipped_relu_kernel(&fc_1_out, &mut ac_1) };

    let mut fc_2_out = [0i32; FC_2_OUTPUT_DIMS];
    // SAFETY: module-level cfg ensures F+BW+VNNI; ac_1.len() == 32.
    unsafe { affine_padded32_avx512_vnni(&mut fc_2_out, fc_2_biases, fc_2_weights, &ac_1) };

    // Shortcut: matches the per-layer flow's wrapping_add byte-for-byte.
    fc_2_out[0].wrapping_add(fc_0_out[HIDDEN1_DIMS])
}

// Reads 2*EWM_HALF i16 from half, writes EWM_HALF u8 to out. See EWM_PRESHIFT for the mulhi trick.
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn transformer_ewm_one_perspective(half: *const i16, out: *mut u8) {
    let zero = _mm512_setzero_si512();
    let cap = _mm512_set1_epi16(EWM_CLAMP_I16);
    let chunks = EWM_HALF / I16_LANES_PER_M512;
    for chunk in 0..chunks {
        let off = chunk * I16_LANES_PER_M512;
        // SAFETY: chunk*32+32 <= EWM_HALF, so both 512-bit unaligned loads stay in range.
        let (in0, in1) = unsafe {
            (
                _mm512_loadu_si512(half.add(off).cast::<__m512i>()),
                _mm512_loadu_si512(half.add(off + EWM_HALF).cast::<__m512i>()),
            )
        };
        let sum0 = _mm512_max_epi16(_mm512_min_epi16(in0, cap), zero);
        let sum1 = _mm512_max_epi16(_mm512_min_epi16(in1, cap), zero);
        let shifted = _mm512_slli_epi16::<EWM_PRESHIFT>(sum0);
        let product = _mm512_mulhi_epi16(shifted, sum1);
        let bytes = _mm512_cvtepi16_epi8(product);
        // SAFETY: chunk*32+32 <= EWM_HALF keeps the 32-byte store in range.
        unsafe { _mm256_storeu_si256(out.add(off).cast::<__m256i>(), bytes) };
    }
}

// in_dims % 64 == 0 only; partial sum << i32::MAX so the reduce wrapping_add is defensive.
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn affine_avx512_vnni(output: &mut [i32], biases: &[i32], weights: &[i8], input: &[u8], in_dims: usize, padded_in: usize) {
    let chunks = in_dims / VNNI_LANES;
    let input_ptr = input.as_ptr();
    for (j, out_slot) in output.iter_mut().enumerate() {
        let row_ptr = weights[j * padded_in..j * padded_in + in_dims].as_ptr();
        let mut acc = _mm512_setzero_si512();
        for k in 0..chunks {
            let offset = k * VNNI_LANES;
            // SAFETY: chunks*VNNI_LANES == in_dims; both loads stay within input and weight row.
            let (a, b) = unsafe {
                (
                    _mm512_loadu_si512(input_ptr.add(offset).cast::<__m512i>()),
                    _mm512_loadu_si512(row_ptr.add(offset).cast::<__m512i>()),
                )
            };
            acc = _mm512_dpbusd_epi32(acc, a, b);
        }
        *out_slot = biases[j].wrapping_add(_mm512_reduce_add_epi32(acc));
    }
}

// input.len() % 16 == 0 only; cvtepi32_epi8 is loss-free because lanes are clamped to [0,127] first.
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn clipped_relu_kernel(input: &[i32], output: &mut [u8]) {
    let chunks = input.len() / I32_LANES_PER_M512;
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let zero = _mm512_setzero_si512();
    let cap = _mm512_set1_epi32(127);
    for k in 0..chunks {
        let off = k * I32_LANES_PER_M512;
        // SAFETY: chunks*16 == input.len() == output.len(); load + store stay in range.
        let x = unsafe { _mm512_loadu_si512(in_ptr.add(off).cast::<__m512i>()) };
        let shifted = _mm512_srai_epi32::<WEIGHT_SCALE_BITS>(x);
        let clamped_lo = _mm512_max_epi32(shifted, zero);
        let clamped = _mm512_min_epi32(clamped_lo, cap);
        let bytes = _mm512_cvtepi32_epi8(clamped);
        // SAFETY: see the load — same length invariant.
        unsafe { _mm_storeu_si128(out_ptr.add(off).cast::<__m128i>(), bytes) };
    }
}

// input.len() % 16 == 0 only; i64 squaring stays bit-identical to scalar across i32 incl. i32::MIN.
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn sqr_clipped_relu_kernel(input: &[i32], output: &mut [u8]) {
    let chunks = input.len() / I32_LANES_PER_M512;
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let cap = _mm512_set1_epi64(127);
    for k in 0..chunks {
        let off = k * I32_LANES_PER_M512;
        // SAFETY: chunks*16 == input.len() == output.len(); load + store in range.
        let x = unsafe { _mm512_loadu_si512(in_ptr.add(off).cast::<__m512i>()) };
        let lo_i32: __m256i = _mm512_castsi512_si256(x);
        let hi_i32: __m256i = _mm512_extracti64x4_epi64::<1>(x);
        let lo_i64 = _mm512_cvtepi32_epi64(lo_i32);
        let hi_i64 = _mm512_cvtepi32_epi64(hi_i32);
        let sq_lo = _mm512_mul_epi32(lo_i64, lo_i64);
        let sq_hi = _mm512_mul_epi32(hi_i64, hi_i64);
        let shr_lo = _mm512_srai_epi64::<SQR_SHIFT>(sq_lo);
        let shr_hi = _mm512_srai_epi64::<SQR_SHIFT>(sq_hi);
        let cl_lo = _mm512_min_epi64(shr_lo, cap);
        let cl_hi = _mm512_min_epi64(shr_hi, cap);
        let bytes_lo = _mm512_cvtepi64_epi8(cl_lo);
        let bytes_hi = _mm512_cvtepi64_epi8(cl_hi);
        let combined = _mm_unpacklo_epi64(bytes_lo, bytes_hi);
        // SAFETY: see the load — same length invariant.
        unsafe { _mm_storeu_si128(out_ptr.add(off).cast::<__m128i>(), combined) };
    }
}

// VNNI affine for fc_1/fc_2 (in_dims=32); zextsi256_si512 zero-extends so one dpbusd suffices.
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn affine_padded32_avx512_vnni(output: &mut [i32], biases: &[i32], weights: &[i8], input: &[u8; 32]) {
    debug_assert_eq!(output.len(), biases.len());
    debug_assert_eq!(weights.len(), output.len() * 32);

    // SAFETY: input is &[u8; 32], so the 256-bit load reads exactly 32 bytes.
    let in_full = unsafe { _mm512_zextsi256_si512(_mm256_loadu_si256(input.as_ptr().cast::<__m256i>())) };

    for (j, out_slot) in output.iter_mut().enumerate() {
        let row_ptr = weights[j * 32..j * 32 + 32].as_ptr();
        // SAFETY: weight row spans 32 bytes; same zero-extension argument.
        let row_full = unsafe { _mm512_zextsi256_si512(_mm256_loadu_si256(row_ptr.cast::<__m256i>())) };
        let acc = _mm512_dpbusd_epi32(_mm512_setzero_si512(), in_full, row_full);
        *out_slot = biases[j].wrapping_add(_mm512_reduce_add_epi32(acc));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seeded_i32_inputs(len: usize, seed: u32) -> Box<[i32]> {
        let mut v = vec![0i32; len].into_boxed_slice();
        for (i, slot) in v.iter_mut().enumerate() {
            *slot = match (i + seed as usize) % 11 {
                0 => 0,
                1 => i32::MAX,
                2 => i32::MIN,
                3 => 8_127,
                4 => 8_128,
                5 => -1,
                6 => 100_000,
                7 => -100_000,
                8 => 725,
                9 => -725,
                _ => (i as i32).wrapping_mul(257).wrapping_sub(seed as i32 * 13),
            };
        }
        v
    }

    fn seeded_weights(len: usize, seed: u32) -> Box<[i8]> {
        let mut v = vec![0i8; len].into_boxed_slice();
        for (i, slot) in v.iter_mut().enumerate() {
            let raw = (i as u32).wrapping_mul(7).wrapping_add(seed) % 251;
            *slot = (raw as i32 - 125) as i8;
        }
        v
    }

    fn seeded_biases(len: usize, seed: u32) -> Box<[i32]> {
        let mut v = vec![0i32; len].into_boxed_slice();
        for (i, slot) in v.iter_mut().enumerate() {
            *slot = (i as i32) * 17 - 9 * (seed as i32);
        }
        v
    }

    fn seeded_input(len: usize, seed: u32) -> Box<[u8]> {
        let mut v = vec![0u8; len].into_boxed_slice();
        for (i, slot) in v.iter_mut().enumerate() {
            *slot = ((i as u32).wrapping_mul(11).wrapping_add(seed) % 256) as u8;
        }
        v
    }

    fn run_parity_check(in_dims: usize, padded_in: usize, out_dims: usize, seed: u32) {
        let weights = seeded_weights(out_dims * padded_in, seed);
        let biases = seeded_biases(out_dims, seed);
        let input = seeded_input(padded_in, seed);

        let mut avx_out = vec![0i32; out_dims].into_boxed_slice();
        // SAFETY: this module is cfg-gated on F+BW+VNNI.
        unsafe { affine(&mut avx_out, &biases, &weights, &input, in_dims, padded_in) };

        let mut sca_out = vec![0i32; out_dims].into_boxed_slice();
        scalar_post_ft::affine(&mut sca_out, &biases, &weights, &input, in_dims, padded_in);

        assert_eq!(
            avx_out, sca_out,
            "AVX-512 / scalar mismatch at shape ({in_dims}, {padded_in}, {out_dims})"
        );
    }

    #[test]
    fn affine_matches_scalar_for_fc_0_shape() {
        run_parity_check(1536, 1536, 16, 41);
    }

    #[test]
    fn affine_matches_scalar_for_fc_1_shape() {
        run_parity_check(30, 32, 32, 53);
    }

    #[test]
    fn affine_matches_scalar_for_fc_2_shape() {
        run_parity_check(32, 32, 1, 67);
    }

    #[test]
    fn affine_padding_lanes_are_ignored_for_fc_1_shape() {
        // fc_1 must respect in_dims=30: garbage in padding lanes must not affect output.
        let in_dims = 30;
        let padded_in = 32;
        let out_dims = 32;
        let seed = 91;

        let mut weights = seeded_weights(out_dims * padded_in, seed);
        let biases = seeded_biases(out_dims, seed);
        let mut input = seeded_input(padded_in, seed);

        for j in 0..out_dims {
            for c in in_dims..padded_in {
                weights[j * padded_in + c] = 0;
            }
        }
        for c in in_dims..padded_in {
            input[c] = 0;
        }
        let mut sca_out = vec![0i32; out_dims].into_boxed_slice();
        scalar_post_ft::affine(&mut sca_out, &biases, &weights, &input, in_dims, padded_in);

        // Scribble garbage into the padding lanes; AVX-512 must still match.
        for j in 0..out_dims {
            for c in in_dims..padded_in {
                weights[j * padded_in + c] = 99;
            }
        }
        for c in in_dims..padded_in {
            input[c] = 0xAA;
        }
        let mut avx_out = vec![0i32; out_dims].into_boxed_slice();
        // SAFETY: module-level cfg ensures F+BW+VNNI.
        unsafe { affine(&mut avx_out, &biases, &weights, &input, in_dims, padded_in) };

        assert_eq!(avx_out, sca_out);
    }

    fn clipped_relu_parity(input: &[i32]) {
        let mut avx_out = vec![0u8; input.len()];
        let mut sca_out = vec![0u8; input.len()];
        // SAFETY: module-level cfg ensures F+BW (subset of F+BW+VNNI).
        unsafe { clipped_relu(input, &mut avx_out) };
        scalar_post_ft::clipped_relu(input, &mut sca_out);
        assert_eq!(avx_out, sca_out, "clipped_relu mismatch at len={}", input.len());
    }

    fn sqr_clipped_relu_parity(input: &[i32]) {
        let mut avx_out = vec![0u8; input.len()];
        let mut sca_out = vec![0u8; input.len()];
        // SAFETY: module-level cfg ensures F+BW.
        unsafe { sqr_clipped_relu(input, &mut avx_out) };
        scalar_post_ft::sqr_clipped_relu(input, &mut sca_out);
        assert_eq!(avx_out, sca_out, "sqr_clipped_relu mismatch at len={}", input.len());
    }

    #[test]
    fn clipped_relu_matches_scalar_on_boundary_values() {
        let inputs: [i32; 16] = [
            -1_000_000,
            -64,
            -1,
            0,
            1,
            63,
            64,
            8_000,
            8_127,
            8_128,
            i32::MAX,
            i32::MIN,
            2,
            -50,
            1_000,
            -1_000,
        ];
        clipped_relu_parity(&inputs);
    }

    #[test]
    fn clipped_relu_matches_scalar_for_fc_1_shape() {
        let inputs = seeded_i32_inputs(32, 17);
        clipped_relu_parity(&inputs);
    }

    #[test]
    fn clipped_relu_falls_back_to_scalar_for_short_input() {
        let inputs = seeded_i32_inputs(15, 23);
        clipped_relu_parity(&inputs);
    }

    // i32::MIN / MAX witness the i64-squaring path: mullo_epi32 would wrap.
    #[test]
    fn sqr_clipped_relu_matches_scalar_on_boundary_values() {
        let inputs: [i32; 16] = [
            0,
            1,
            724,
            725,
            8_191,
            8_192,
            100_000,
            -724,
            -725,
            -100_000,
            i32::MAX,
            i32::MIN,
            -1,
            5_000_000,
            -5_000_000,
            12_345,
        ];
        sqr_clipped_relu_parity(&inputs);
    }

    #[test]
    fn sqr_clipped_relu_matches_scalar_on_full_chunk() {
        let inputs = seeded_i32_inputs(16, 41);
        sqr_clipped_relu_parity(&inputs);
    }

    #[test]
    fn sqr_clipped_relu_falls_back_to_scalar_for_short_input() {
        let inputs = seeded_i32_inputs(15, 59);
        sqr_clipped_relu_parity(&inputs);
    }

    fn seeded_accumulator(seed: u32) -> Accumulator {
        let mut acc = Accumulator::zeroed();
        let fill = |i: usize, salt: u32| -> i16 {
            match (i + seed as usize + salt as usize) % 13 {
                0 => i16::MIN,
                1 => i16::MAX,
                2 => 0,
                3 => -1,
                4 => 254,
                5 => 255,
                6 => 253,
                7 => -10_000,
                8 => 30_000,
                9 => 100,
                10 => 200,
                11 => 1,
                _ => (i as i32).wrapping_mul(31).wrapping_sub(seed as i32) as i16,
            }
        };
        for (i, slot) in acc.us.iter_mut().enumerate() {
            *slot = fill(i, 0);
        }
        for (i, slot) in acc.them.iter_mut().enumerate() {
            *slot = fill(i, 7);
        }
        acc
    }

    fn transformer_ewm_parity(acc: &Accumulator, stm: Color) {
        let mut avx_out = [0u8; FC_0_INPUT_DIMS];
        let mut sca_out = [0u8; FC_0_INPUT_DIMS];
        // SAFETY: module-level cfg ensures F+BW.
        unsafe { transformer_ewm(acc, stm, &mut avx_out) };
        scalar_post_ft::transformer_ewm(acc, stm, &mut sca_out);
        // Manual diff: assert_eq! on [u8; 1536] would dump too much on fail.
        if avx_out != sca_out {
            for (i, (a, s)) in avx_out.iter().zip(sca_out.iter()).enumerate() {
                assert_eq!(a, s, "transformer_ewm mismatch at j={i} for stm={stm:?}: avx={a}, scalar={s}");
            }
        }
    }

    #[test]
    fn transformer_ewm_matches_scalar_for_black_stm() {
        let acc = seeded_accumulator(13);
        transformer_ewm_parity(&acc, Color::BLACK);
    }

    #[test]
    fn transformer_ewm_matches_scalar_for_white_stm() {
        let acc = seeded_accumulator(13);
        transformer_ewm_parity(&acc, Color::WHITE);
    }

    #[test]
    fn transformer_ewm_matches_scalar_on_clamp_boundaries() {
        let mut acc = Accumulator::zeroed();
        let boundaries: [i16; 8] = [i16::MIN, -1, 0, 1, 253, 254, 255, i16::MAX];
        const HALF: usize = HIDDEN_SIZE / 2;
        for (i, &v) in boundaries.iter().enumerate() {
            acc.us[i] = v;
            acc.us[i + HALF] = boundaries[(i + 3) % boundaries.len()];
            acc.them[i] = boundaries[(i + 5) % boundaries.len()];
            acc.them[i + HALF] = v;
        }
        transformer_ewm_parity(&acc, Color::BLACK);
        transformer_ewm_parity(&acc, Color::WHITE);
    }

    use crate::evaluate::nnue::aligned::Aligned64;
    use crate::evaluate::nnue::types::{FC_1_INPUT_DIMS, FC_2_INPUT_DIMS, LAYER_STACKS, NetworkStack};

    fn seeded_stack(seed: u32) -> NetworkStack {
        let fc_0_biases: Aligned64<i32> = (0..FC_0_OUTPUT_DIMS)
            .map(|i| (i as i32).wrapping_mul(11).wrapping_sub(seed as i32 * 5))
            .collect();
        let fc_0_weights: Aligned64<i8> = (0..FC_0_OUTPUT_DIMS * FC_0_PADDED_INPUT_DIMS)
            .map(|i| {
                let raw = (i as u32).wrapping_mul(7).wrapping_add(seed) % 251;
                (raw as i32 - 125) as i8
            })
            .collect();
        let fc_1_biases: Aligned64<i32> = (0..FC_1_OUTPUT_DIMS)
            .map(|i| (i as i32).wrapping_mul(17).wrapping_sub(seed as i32 * 3))
            .collect();
        let fc_1_weights: Aligned64<i8> = (0..FC_1_OUTPUT_DIMS * FC_1_PADDED_INPUT_DIMS)
            .map(|i| {
                let raw = (i as u32).wrapping_mul(13).wrapping_add(seed.wrapping_mul(31)) % 251;
                (raw as i32 - 125) as i8
            })
            .collect();
        let fc_2_biases: Aligned64<i32> = (0..FC_2_OUTPUT_DIMS)
            .map(|i| (i as i32).wrapping_mul(23).wrapping_add(seed as i32 * 7))
            .collect();
        let fc_2_weights: Aligned64<i8> = (0..FC_2_OUTPUT_DIMS * FC_2_PADDED_INPUT_DIMS)
            .map(|i| {
                let raw = (i as u32).wrapping_mul(19).wrapping_add(seed.wrapping_mul(53)) % 251;
                (raw as i32 - 125) as i8
            })
            .collect();
        NetworkStack {
            fc_0_biases,
            fc_0_weights,
            fc_1_biases,
            fc_1_weights,
            fc_2_biases,
            fc_2_weights,
        }
    }

    fn per_layer_reference_score(transformed: &[u8; FC_0_INPUT_DIMS], stack: &NetworkStack) -> i32 {
        let mut fc_0_out = [0i32; FC_0_OUTPUT_DIMS];
        scalar_post_ft::affine(
            &mut fc_0_out,
            &stack.fc_0_biases,
            &stack.fc_0_weights,
            transformed,
            FC_0_INPUT_DIMS,
            FC_0_PADDED_INPUT_DIMS,
        );
        let mut ac_0 = [0u8; HIDDEN1_DIMS];
        scalar_post_ft::clipped_relu(&fc_0_out[..HIDDEN1_DIMS], &mut ac_0);
        let mut ac_sqr_0 = [0u8; HIDDEN1_DIMS];
        scalar_post_ft::sqr_clipped_relu(&fc_0_out[..HIDDEN1_DIMS], &mut ac_sqr_0);

        let mut fc_1_in = [0u8; FC_1_PADDED_INPUT_DIMS];
        fc_1_in[..HIDDEN1_DIMS].copy_from_slice(&ac_sqr_0);
        fc_1_in[HIDDEN1_DIMS..2 * HIDDEN1_DIMS].copy_from_slice(&ac_0);

        let mut fc_1_out = [0i32; FC_1_OUTPUT_DIMS];
        scalar_post_ft::affine(
            &mut fc_1_out,
            &stack.fc_1_biases,
            &stack.fc_1_weights,
            &fc_1_in,
            FC_1_INPUT_DIMS,
            FC_1_PADDED_INPUT_DIMS,
        );
        let mut ac_1 = [0u8; FC_1_OUTPUT_DIMS];
        scalar_post_ft::clipped_relu(&fc_1_out, &mut ac_1);

        let mut fc_2_out = [0i32; FC_2_OUTPUT_DIMS];
        scalar_post_ft::affine(
            &mut fc_2_out,
            &stack.fc_2_biases,
            &stack.fc_2_weights,
            &ac_1,
            FC_2_INPUT_DIMS,
            FC_2_PADDED_INPUT_DIMS,
        );
        fc_2_out[0].wrapping_add(fc_0_out[HIDDEN1_DIMS])
    }

    #[test]
    fn fused_fc_chain_matches_per_layer_flow_for_all_buckets() {
        for bucket in 0..LAYER_STACKS {
            let seed = 100 + bucket as u32;
            let stack = seeded_stack(seed);
            let mut transformed = [0u8; FC_0_INPUT_DIMS];
            for (i, slot) in transformed.iter_mut().enumerate() {
                *slot = ((i as u32).wrapping_mul(11).wrapping_add(seed) % 256) as u8;
            }

            // SAFETY: module-level cfg ensures F+BW+VNNI.
            let fused = unsafe {
                fused_fc_chain(
                    &transformed,
                    &stack.fc_0_biases,
                    &stack.fc_0_weights,
                    &stack.fc_1_biases,
                    &stack.fc_1_weights,
                    &stack.fc_2_biases,
                    &stack.fc_2_weights,
                )
            };

            let per_layer = per_layer_reference_score(&transformed, &stack);

            assert_eq!(
                fused, per_layer,
                "fused vs. per-layer mismatch at bucket {bucket} (seed {seed}): fused={fused}, per_layer={per_layer}"
            );
        }
    }

    #[test]
    fn fused_fc_chain_preserves_shortcut() {
        // Only fc_0_biases[HIDDEN1_DIMS]=K is nonzero, so the shortcut alone must yield exactly K.
        const SHORTCUT_K: i32 = 12_345;
        let mut fc_0_biases = Aligned64::<i32>::zeroed(FC_0_OUTPUT_DIMS);
        fc_0_biases[HIDDEN1_DIMS] = SHORTCUT_K;
        let stack = NetworkStack {
            fc_0_biases,
            fc_0_weights: Aligned64::zeroed(FC_0_OUTPUT_DIMS * FC_0_PADDED_INPUT_DIMS),
            fc_1_biases: Aligned64::zeroed(FC_1_OUTPUT_DIMS),
            fc_1_weights: Aligned64::zeroed(FC_1_OUTPUT_DIMS * FC_1_PADDED_INPUT_DIMS),
            fc_2_biases: Aligned64::zeroed(FC_2_OUTPUT_DIMS),
            fc_2_weights: Aligned64::zeroed(FC_2_OUTPUT_DIMS * FC_2_PADDED_INPUT_DIMS),
        };
        let transformed = [42u8; FC_0_INPUT_DIMS];

        // SAFETY: module-level cfg ensures F+BW+VNNI.
        let fused = unsafe {
            fused_fc_chain(
                &transformed,
                &stack.fc_0_biases,
                &stack.fc_0_weights,
                &stack.fc_1_biases,
                &stack.fc_1_weights,
                &stack.fc_2_biases,
                &stack.fc_2_weights,
            )
        };
        assert_eq!(fused, SHORTCUT_K);
    }
}
