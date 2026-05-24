use static_assertions::const_assert;
use thiserror::Error;

pub const HIDDEN_SIZE: usize = 1_536;
pub const NUM_FEATURES: usize = 73_305;
pub const LAYER_STACKS: usize = 9;

// fc_0 produces HIDDEN1_DIMS + 1 = 16 outputs; the 16th feeds only the
// post-fc_2 shortcut. The first 15 feed both ClippedReLU and SqrClippedReLU,
// whose 30-wide concat is fc_1's input.
pub const HIDDEN1_DIMS: usize = 15;
pub const HIDDEN2_DIMS: usize = 32;

pub const FC_0_OUTPUT_DIMS: usize = HIDDEN1_DIMS + 1;
pub const FC_0_INPUT_DIMS: usize = HIDDEN_SIZE;
pub const FC_0_PADDED_INPUT_DIMS: usize = HIDDEN_SIZE;

pub const FC_1_OUTPUT_DIMS: usize = HIDDEN2_DIMS;
pub const FC_1_INPUT_DIMS: usize = HIDDEN1_DIMS * 2;
pub const FC_1_PADDED_INPUT_DIMS: usize = 32;

pub const FC_2_OUTPUT_DIMS: usize = 1;
pub const FC_2_INPUT_DIMS: usize = HIDDEN2_DIMS;
pub const FC_2_PADDED_INPUT_DIMS: usize = 32;

// NUM_FEATURES = 73_305 > u16::MAX, so u16 cannot index the full space.
pub type FeatureIndex = u32;

// A legal Shogi position always has 40 piece slots (board ∪ hand); 48 gives
// slack for diagnostic over-approximations.
pub const FEATURE_LIST_CAPACITY: usize = 48;

const_assert!(FEATURE_LIST_CAPACITY >= 40);

pub type FeatureList = arrayvec::ArrayVec<FeatureIndex, FEATURE_LIST_CAPACITY>;

#[repr(C, align(64))]
#[derive(Clone)]
pub struct Accumulator {
    pub us: [i16; HIDDEN_SIZE],
    pub them: [i16; HIDDEN_SIZE],
    pub computed: bool,
}

impl Accumulator {
    pub fn zeroed() -> Self {
        Accumulator {
            us: [0; HIDDEN_SIZE],
            them: [0; HIDDEN_SIZE],
            computed: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct NetHeader {
    pub version: u32,
    pub hash: u32,
    pub arch_id: String,
}

#[derive(Debug)]
pub struct NetworkStack {
    pub fc_0_biases: Box<[i32]>,
    pub fc_0_weights: Box<[i8]>,
    pub fc_1_biases: Box<[i32]>,
    pub fc_1_weights: Box<[i8]>,
    pub fc_2_biases: Box<[i32]>,
    pub fc_2_weights: Box<[i8]>,
}

#[derive(Debug)]
pub struct NnueNetwork {
    pub header: NetHeader,
    pub ft_biases: Box<[i16]>,
    pub ft_weights: Box<[i16]>,
    pub stacks: Vec<NetworkStack>,
    pub sha256: [u8; 32],
}

#[derive(Error, Debug)]
pub enum NnueError {
    #[error("failed to open NNUE file {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("NNUE file is not SFNNwoP1536 (got {got})")]
    ArchMismatch { got: String },

    #[error("NNUE file has unexpected size: expected {expected} bytes, got {got}")]
    SizeMismatch { expected: usize, got: usize },

    #[error("NNUE file is malformed: {reason}")]
    InvalidFormat { reason: String },

    #[error("no NNUE network loaded; point the Eval_Dir USI option at a directory containing nn.bin")]
    NotLoaded,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    #[test]
    fn hidden_size_matches_sfnnwop1536() {
        assert_eq!(HIDDEN_SIZE, 1536);
    }

    #[test]
    fn num_features_matches_half_ka_hm() {
        assert_eq!(NUM_FEATURES, 5 * 9 * 1629);
    }

    #[test]
    fn layer_stacks_is_nine() {
        assert_eq!(LAYER_STACKS, 9);
    }

    #[test]
    fn fc_0_shapes_match_spec() {
        assert_eq!(FC_0_OUTPUT_DIMS, 16);
        assert_eq!(FC_0_INPUT_DIMS, 1536);
        assert_eq!(FC_0_PADDED_INPUT_DIMS, 1536);
    }

    #[test]
    fn fc_1_shapes_match_spec() {
        assert_eq!(FC_1_OUTPUT_DIMS, 32);
        assert_eq!(FC_1_INPUT_DIMS, 30);
        assert_eq!(FC_1_PADDED_INPUT_DIMS, 32);
    }

    #[test]
    fn fc_2_shapes_match_spec() {
        assert_eq!(FC_2_OUTPUT_DIMS, 1);
        assert_eq!(FC_2_INPUT_DIMS, 32);
        assert_eq!(FC_2_PADDED_INPUT_DIMS, 32);
    }

    #[test]
    fn accumulator_is_cache_line_aligned() {
        assert_eq!(align_of::<Accumulator>(), 64);
    }

    #[test]
    fn accumulator_covers_both_perspectives() {
        assert!(size_of::<Accumulator>() >= 2 * HIDDEN_SIZE * size_of::<i16>());
    }

    #[test]
    fn accumulator_zeroed_has_computed_false() {
        let acc = Accumulator::zeroed();
        assert!(!acc.computed);
        assert!(acc.us.iter().all(|&x| x == 0));
        assert!(acc.them.iter().all(|&x| x == 0));
    }

    #[test]
    fn not_loaded_has_human_readable_message() {
        let msg = format!("{}", NnueError::NotLoaded);
        assert!(!msg.is_empty());
        assert!(msg.contains("Eval_Dir"));
        assert!(msg.contains("nn.bin"));
    }

    #[test]
    fn invalid_format_carries_reason() {
        let msg = format!(
            "{}",
            NnueError::InvalidFormat {
                reason: "bad magic".to_string()
            }
        );
        assert!(msg.contains("bad magic"));
    }

    #[test]
    fn feature_list_is_empty_by_default() {
        let list: FeatureList = FeatureList::new();
        assert_eq!(list.len(), 0);
        assert_eq!(list.capacity(), FEATURE_LIST_CAPACITY);
    }
}

const_assert!(NUM_FEATURES <= FeatureIndex::MAX as usize + 1);
