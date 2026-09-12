//! Evaluation-layer NNUE support for the SFNN-1536 architecture.
//!
//! This crate is the Evaluation layer: it may import State and Storage, never
//! Protocol or Search. It covers the evaluation file the engine reads its
//! parameters from, feature extraction, the accumulator, the forward pass, and
//! the SIMD kernels.
//!
//! The parameters are laid out for the kernels when the engine is built, not
//! when it starts: [`network_file`] opens that file, holds it against what this
//! binary was built for, and puts it in the region the kernels read it from —
//! which is at a fixed address, so a parameter array is that address plus a
//! constant. Which network a piece of code reads is a type parameter
//! ([`NetworkParams`]), so that address stays a constant all the way into the
//! kernels.

// The engine's own parameters are at an address the linker fixed and its
// accumulators carry their rows inline, so nothing outside the kernel parity
// tests asks for an aligned buffer of its own.
#[cfg(test)]
mod aligned;
mod config;
mod features;
mod finny;
#[cfg(any(test, feature = "source-network"))]
mod loader;
mod network;
pub mod network_file;
// Shared verbatim with the build script, which renders the whole layout; the
// engine addresses the constant form of it, so the walked one is used only by
// the conversion tooling and by that script.
#[path = "../nnue_layout.rs"]
#[allow(dead_code)]
mod nnue_layout;
// Shared verbatim with the build script, as the layout above is, and used by it
// in full.
#[cfg(any(test, feature = "source-network"))]
#[path = "../nnue_source.rs"]
#[allow(dead_code)]
mod nnue_source;
mod simd;
mod transformer;
mod types;

pub use features::{
    FEATURE_DIMENSION, FeatureIndex, MAX_ACTIVE_FEATURES, MoveDelta, PerspectiveDelta,
    active_features, active_features_both, requires_full_refresh,
};
pub use finny::FinnyCache;
#[cfg(any(test, feature = "source-network"))]
pub use loader::{load_network, load_network_with_warnings, parameter_bytes, source_header};
pub use network::{FV_SCALE, evaluate, evaluate_with, layer_stack_index};
pub use network_file::Region;
pub use simd::{Backend, active_backend};
pub use transformer::{Accumulator, FT_OUTPUT_DIMS};
#[cfg(any(test, feature = "source-network"))]
pub use types::OwnedNetwork;
pub use types::{
    FC_0_INPUT_DIMS, FC_0_OUTPUT_DIMS, FC_0_PADDED_INPUT_DIMS, FC_1_INPUT_DIMS, FC_1_OUTPUT_DIMS,
    FC_1_PADDED_INPUT_DIMS, FC_2_INPUT_DIMS, FC_2_OUTPUT_DIMS, FC_2_PADDED_INPUT_DIMS, HIDDEN_SIZE,
    HIDDEN1_DIMS, HIDDEN2_DIMS, LAYER_STACKS, NUM_FEATURES, NetDims, NetHeader, NetStack,
    NetworkParams, NnueError, PtrNetwork,
};
