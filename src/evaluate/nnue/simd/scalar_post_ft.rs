use super::super::types::{Accumulator, FC_0_INPUT_DIMS, HIDDEN_SIZE};
use crate::types::Color;

const WEIGHT_SCALE_BITS: i32 = 6;
const SQR_SHIFT: i32 = 2 * WEIGHT_SCALE_BITS + 7;

// FT biases and weights are scaled ×2 at load time, so the post-scale
// accumulator bound is [0, 254], not [0, 127].
const EWM_CLAMP: i32 = 127 * 2;
const EWM_SHIFT: i32 = 9;

// Output layout: `[EWM(stm-half) | EWM(~stm-half)]`. The accumulator stores
// BLACK in `acc.us` and WHITE in `acc.them`; this function reorders per stm.
pub fn transformer_ewm(acc: &Accumulator, stm: Color, out: &mut [u8; FC_0_INPUT_DIMS]) {
    debug_assert_eq!(FC_0_INPUT_DIMS, HIDDEN_SIZE);
    debug_assert_eq!(HIDDEN_SIZE % 2, 0);
    const HALF: usize = HIDDEN_SIZE / 2;

    let (stm_half, other_half): (&[i16; HIDDEN_SIZE], &[i16; HIDDEN_SIZE]) = if stm == Color::BLACK {
        (&acc.us, &acc.them)
    } else {
        (&acc.them, &acc.us)
    };

    ewm_one_perspective(stm_half, &mut out[..HALF]);
    ewm_one_perspective(other_half, &mut out[HALF..]);
}

pub fn affine(output: &mut [i32], biases: &[i32], weights: &[i8], input: &[u8], in_dims: usize, padded_in: usize) {
    debug_assert_eq!(output.len(), biases.len());
    debug_assert_eq!(weights.len(), output.len() * padded_in);
    debug_assert!(in_dims <= padded_in);
    debug_assert!(input.len() >= in_dims);

    for (j, out_slot) in output.iter_mut().enumerate() {
        let row = j * padded_in;
        let mut acc = biases[j];
        for i in 0..in_dims {
            acc = acc.wrapping_add(weights[row + i] as i32 * input[i] as i32);
        }
        *out_slot = acc;
    }
}

pub fn clipped_relu(input: &[i32], output: &mut [u8]) {
    debug_assert_eq!(input.len(), output.len());
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        *out = (inp >> WEIGHT_SCALE_BITS).clamp(0, 127) as u8;
    }
}

// i64 intermediate avoids overflow on the i32 square.
pub fn sqr_clipped_relu(input: &[i32], output: &mut [u8]) {
    debug_assert_eq!(input.len(), output.len());
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        let sq = (*inp as i64) * (*inp as i64);
        *out = (sq >> SQR_SHIFT).clamp(0, 127) as u8;
    }
}

fn ewm_one_perspective(half: &[i16; HIDDEN_SIZE], out: &mut [u8]) {
    const HALF: usize = HIDDEN_SIZE / 2;
    debug_assert_eq!(out.len(), HALF);
    for j in 0..HALF {
        let s0 = (half[j] as i32).clamp(0, EWM_CLAMP);
        let s1 = (half[j + HALF] as i32).clamp(0, EWM_CLAMP);
        out[j] = ((s0 * s1) >> EWM_SHIFT) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clipped_relu_matches_reference_on_boundary_values() {
        let inputs = [-1_000_000i32, -64, -1, 0, 1, 63, 64, 8_000, 8_127, 8_128, i32::MAX];
        let mut out = [0u8; 11];
        clipped_relu(&inputs, &mut out);
        assert_eq!(out, [0u8, 0, 0, 0, 0, 0, 1, 125, 126, 127, 127]);
    }

    #[test]
    fn sqr_clipped_relu_matches_reference_formula() {
        let inputs = [
            0i32,
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
        ];
        let mut out = [0u8; 12];
        sqr_clipped_relu(&inputs, &mut out);
        assert_eq!(out, [0u8, 0, 0, 1, 127, 127, 127, 0, 1, 127, 127, 127]);
    }

    #[test]
    fn transformer_ewm_lays_out_stm_then_other_stm() {
        // us-half: (200*100) >> 9 = 39. them-half: (250*50) >> 9 = 24.
        let mut acc = Accumulator::zeroed();
        acc.us[0] = 200;
        acc.us[768] = 100;
        acc.them[0] = 250;
        acc.them[768] = 50;

        const HALF: usize = HIDDEN_SIZE / 2;
        let mut out = [0u8; FC_0_INPUT_DIMS];

        transformer_ewm(&acc, Color::BLACK, &mut out);
        assert_eq!(out[0], 39);
        assert_eq!(out[HALF], 24);
        for slot in &out[1..HALF] {
            assert_eq!(*slot, 0);
        }
        for slot in &out[HALF + 1..] {
            assert_eq!(*slot, 0);
        }

        let mut out_w = [0u8; FC_0_INPUT_DIMS];
        transformer_ewm(&acc, Color::WHITE, &mut out_w);
        assert_eq!(out_w[0], 24);
        assert_eq!(out_w[HALF], 39);
    }

    #[test]
    fn transformer_ewm_clamps_negative_accumulator_values_to_zero() {
        let mut acc = Accumulator::zeroed();
        acc.us[0] = -1_000;
        acc.us[768] = 100;

        let mut out = [0u8; FC_0_INPUT_DIMS];
        transformer_ewm(&acc, Color::BLACK, &mut out);
        assert_eq!(out[0], 0);
    }

    #[test]
    fn transformer_ewm_saturates_above_254() {
        // saturates at EWM_CLAMP=254: (254*254) >> 9 = 126.
        let mut acc = Accumulator::zeroed();
        acc.us[0] = 30_000;
        acc.us[768] = 30_000;

        let mut out = [0u8; FC_0_INPUT_DIMS];
        transformer_ewm(&acc, Color::BLACK, &mut out);
        assert_eq!(out[0], 126);
    }
}
