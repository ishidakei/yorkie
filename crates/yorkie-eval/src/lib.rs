//! Evaluation-layer NNUE support for the SFNN-1536 architecture.
//!
//! This crate is the Evaluation layer: it may import State and Storage, never
//! Protocol or Search. It covers the evaluation file the engine reads its
//! parameters from, feature extraction, the accumulator, the forward pass, and
//! the SIMD kernels.
//!
//! The parameters are laid out for the kernels when the engine is built, not
//! when it starts: [`network_file`] opens that file, holds it against what this
//! binary was built for, and gives the kernels their views of it.

mod aligned;
mod config;
mod features;
mod finny;
#[cfg(any(test, feature = "source-network"))]
mod loader;
mod network;
pub mod network_file;
#[path = "../nnue_layout.rs"]
mod nnue_layout;
#[cfg(any(test, feature = "source-network"))]
#[path = "../nnue_source.rs"]
mod nnue_source;
mod simd;
mod transformer;
mod types;

pub use aligned::Aligned64;
pub use features::{
    FEATURE_DIMENSION, FeatureIndex, MAX_ACTIVE_FEATURES, MoveDelta, PerspectiveDelta,
    active_features, active_features_both, requires_full_refresh,
};
pub use finny::FinnyCache;
#[cfg(any(test, feature = "source-network"))]
pub use loader::{load_network, load_network_with_warnings, parameter_bytes, source_header};
pub use network::{FV_SCALE, evaluate, evaluate_with, layer_stack_index};
pub use simd::{Backend, active_backend};
pub use transformer::{Accumulator, FT_OUTPUT_DIMS};
#[cfg(any(test, feature = "source-network"))]
pub use types::NnueNetworkBuilder;
pub use types::{
    FC_0_INPUT_DIMS, FC_0_OUTPUT_DIMS, FC_0_PADDED_INPUT_DIMS, FC_1_INPUT_DIMS, FC_1_OUTPUT_DIMS,
    FC_1_PADDED_INPUT_DIMS, FC_2_INPUT_DIMS, FC_2_OUTPUT_DIMS, FC_2_PADDED_INPUT_DIMS, HIDDEN_SIZE,
    HIDDEN1_DIMS, HIDDEN2_DIMS, LAYER_STACKS, NUM_FEATURES, NetDims, NetHeader, NetworkStack,
    NnueError, NnueNetwork,
};
