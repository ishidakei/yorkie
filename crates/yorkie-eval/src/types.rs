//! The loaded network and the loader error type for SFNN-1536.
//!
//! The dimensions and the byte layout the parameters sit in are the shared
//! definition the build script writes the evaluation file from; what this
//! module adds is the typed views the kernels read through, and the memory
//! those views point into.
//!
//! The prose here follows the reference's naming: the **FT layer** is the
//! feature transformer, and **L1 / L2 / L3** are the dense layers after it. The
//! FT layer is never called "L1". In the identifiers below, L1 is `fc_0`, L2 is
//! `fc_1` and L3 is `fc_2`.

use std::fmt;

use yorkie_storage::{ArenaSlice, MappedRegion};

#[cfg(any(test, feature = "source-network"))]
use yorkie_storage::LargePageArray;

pub use crate::nnue_layout::{
    FC_0_INPUT_DIMS, FC_0_OUTPUT_DIMS, FC_0_PADDED_INPUT_DIMS, FC_1_INPUT_DIMS, FC_1_OUTPUT_DIMS,
    FC_1_PADDED_INPUT_DIMS, FC_2_INPUT_DIMS, FC_2_OUTPUT_DIMS, FC_2_PADDED_INPUT_DIMS, HIDDEN_SIZE,
    HIDDEN1_DIMS, HIDDEN2_DIMS, LAYER_STACKS, NUM_FEATURES, NetDims, NetHeader,
};
use crate::nnue_layout::{Span, StackSpans, net_spans};

/// One layer stack's parameter arrays, each a 64-byte-aligned view into the
/// network's one region of memory.
#[derive(Debug)]
pub struct NetworkStack {
    pub fc_0_biases: ArenaSlice<i32>,
    pub fc_0_weights: ArenaSlice<i8>,
    pub fc_1_biases: ArenaSlice<i32>,
    pub fc_1_weights: ArenaSlice<i8>,
    pub fc_2_biases: ArenaSlice<i32>,
    pub fc_2_weights: ArenaSlice<i8>,
}

/// The memory a network's parameters live in, kept alongside the views so it
/// outlives them.
pub(crate) enum Backing {
    /// The evaluation file's own pages, mapped read-only and shared with every
    /// process that maps them.
    Mapped(MappedRegion),
    /// A region this binary declares, one per NUMA node whose workers read it.
    /// It is `static`, so there is nothing to keep alive here.
    Region,
    /// A block on the heap, for the tooling that builds a network in memory
    /// rather than reading one the build laid out.
    #[cfg(any(test, feature = "source-network"))]
    Owned(LargePageArray<u8>),
}

impl fmt::Debug for Backing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Backing::Mapped(region) => write!(f, "Mapped({region:?})"),
            Backing::Region => f.write_str("Region"),
            #[cfg(any(test, feature = "source-network"))]
            Backing::Owned(bytes) => write!(f, "Owned({} bytes)", bytes.len()),
        }
    }
}

/// A loaded SFNN network.
///
/// Every parameter array is a 64-byte-aligned [`ArenaSlice`] view into **one**
/// contiguous region, which is what the AVX-512 kernels require and what lets
/// the whole network be a single mapping or a single copy.
///
/// The views are read-only by contract: the region behind a mapped network is
/// mapped without write permission, so taking a `&mut` to one would fault.
/// Nothing does — a network is reached through a shared reference from the
/// moment it exists.
pub struct NnueNetwork {
    pub header: NetHeader,
    /// The memory the views point into, dropped after them (views carry no drop
    /// glue, so field order cannot produce a use-after-free).
    backing: Backing,
    /// The address and byte length of the parameters, for a caller placing
    /// them.
    region: (usize, usize),
    pub ft_biases: ArenaSlice<i16>,
    pub ft_weights: ArenaSlice<i16>,
    /// The layer stacks, inline: an evaluation reaches its bucket's stack at a
    /// fixed offset from the network rather than through a pointer of its own.
    pub stacks: [NetworkStack; LAYER_STACKS],
    /// The SHA-256 of the network file this one's parameters came from; all
    /// zero when they were not read from one.
    pub sha256: [u8; 32],
}

/// The parameters themselves are a few hundred mebibytes, so what a network
/// shows of itself is what it is, not what it holds.
impl fmt::Debug for NnueNetwork {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NnueNetwork")
            .field("header", &self.header)
            .field("backing", &self.backing)
            .field("stacks", &self.stacks.len())
            .finish()
    }
}

impl NnueNetwork {
    /// Build the views of a network whose parameters are already laid out at
    /// `base`.
    ///
    /// The stack count is part of the type, so `dims` has to name
    /// [`LAYER_STACKS`] of them.
    ///
    /// # Safety
    /// `base` must address at least `data_bytes(dims)` initialised bytes,
    /// aligned to a 64-byte boundary, laid out as the shared layout describes,
    /// and they must live as long as `backing` keeps them — which is for the
    /// life of the returned network.
    pub(crate) unsafe fn over(
        header: NetHeader,
        sha256: [u8; 32],
        dims: &NetDims,
        base: *mut u8,
        backing: Backing,
    ) -> Self {
        assert_eq!(
            dims.layer_stacks, LAYER_STACKS,
            "a network in memory carries exactly {LAYER_STACKS} layer stacks",
        );
        let spans = net_spans(dims);
        // SAFETY: every span lies inside the region the caller vouched for, and
        // each starts on a 64-byte boundary, so each view is aligned and
        // in-bounds; the spans are disjoint, so no two views alias.
        let stacks = std::array::from_fn(|i| unsafe { stack_views(base, &spans.stacks[i]) });
        // SAFETY: as the stacks above, for the two feature-transformer spans.
        unsafe {
            Self {
                header,
                backing,
                region: (base as usize, spans.total_bytes),
                ft_biases: view(base, spans.ft_biases),
                ft_weights: view(base, spans.ft_weights),
                stacks,
                sha256,
            }
        }
    }

    /// The `(address, byte length)` of the parameters, for a caller placing
    /// them: a huge-page hint over the region, or a memory policy over its
    /// pages.
    ///
    /// The address is a `usize` because the consumer hands it to the kernel as
    /// a range descriptor and never dereferences it. It is on a huge-page
    /// boundary however the parameters were read.
    pub fn parameter_region(&self) -> (usize, usize) {
        self.region
    }
}

/// One typed view of `span` inside the region at `base`.
///
/// # Safety
/// As [`NnueNetwork::over`], for this one span.
unsafe fn view<T>(base: *mut u8, span: Span) -> ArenaSlice<T> {
    // SAFETY: the caller vouches for the region; `span.offset` is 64-byte
    // aligned, hence aligned for `T`.
    unsafe { ArenaSlice::from_raw(base.add(span.offset) as *mut T, span.count) }
}

/// The six views of one layer stack.
///
/// # Safety
/// As [`NnueNetwork::over`], for this stack's spans.
unsafe fn stack_views(base: *mut u8, spans: &StackSpans) -> NetworkStack {
    // SAFETY: as `view`, for each of the stack's disjoint spans.
    unsafe {
        NetworkStack {
            fc_0_biases: view(base, spans.fc_0_biases),
            fc_0_weights: view(base, spans.fc_0_weights),
            fc_1_biases: view(base, spans.fc_1_biases),
            fc_1_weights: view(base, spans.fc_1_weights),
            fc_2_biases: view(base, spans.fc_2_biases),
            fc_2_weights: view(base, spans.fc_2_weights),
        }
    }
}

/// In-place builder for a [`NnueNetwork`] in this process's own memory:
/// allocate the region up front, fill each parameter array through a mutable
/// view of it, then [`build`](Self::build).
///
/// The engine never builds a network this way — it reads one the build already
/// laid out — so this is compiled only into the tooling that needs a network in
/// memory: a synthetic one a test writes parameter by parameter, and the source
/// file's own reader.
#[cfg(any(test, feature = "source-network"))]
pub struct NnueNetworkBuilder {
    header: NetHeader,
    sha256: [u8; 32],
    dims: NetDims,
    spans: crate::nnue_layout::NetSpans,
    bytes: LargePageArray<u8>,
}

#[cfg(any(test, feature = "source-network"))]
impl NnueNetworkBuilder {
    /// A zeroed builder for the shipped SFNN-1536 dimensions.
    pub fn new(header: NetHeader, sha256: [u8; 32]) -> Self {
        Self::with_dims(header, sha256, &NetDims::STANDARD)
    }

    /// A zeroed builder for arbitrary `dims`.
    pub fn with_dims(header: NetHeader, sha256: [u8; 32], dims: &NetDims) -> Self {
        let spans = net_spans(dims);
        Self {
            header,
            sha256,
            dims: *dims,
            bytes: LargePageArray::<u8>::zeroed(spans.total_bytes.max(1)),
            spans,
        }
    }

    /// Mutable view of `ft_biases` (`hidden_size` i16).
    pub fn ft_biases_mut(&mut self) -> &mut [i16] {
        let span = self.spans.ft_biases;
        self.slice_mut(span)
    }

    /// Mutable view of `ft_weights` (`hidden_size * num_features` i16).
    pub fn ft_weights_mut(&mut self) -> &mut [i16] {
        let span = self.spans.ft_weights;
        self.slice_mut(span)
    }

    /// Mutable view of stack `i`'s `fc_0_biases`.
    pub fn fc_0_biases_mut(&mut self, i: usize) -> &mut [i32] {
        let span = self.spans.stacks[i].fc_0_biases;
        self.slice_mut(span)
    }
    /// Mutable view of stack `i`'s `fc_0_weights`.
    pub fn fc_0_weights_mut(&mut self, i: usize) -> &mut [i8] {
        let span = self.spans.stacks[i].fc_0_weights;
        self.slice_mut(span)
    }
    /// Mutable view of stack `i`'s `fc_1_biases`.
    pub fn fc_1_biases_mut(&mut self, i: usize) -> &mut [i32] {
        let span = self.spans.stacks[i].fc_1_biases;
        self.slice_mut(span)
    }
    /// Mutable view of stack `i`'s `fc_1_weights`.
    pub fn fc_1_weights_mut(&mut self, i: usize) -> &mut [i8] {
        let span = self.spans.stacks[i].fc_1_weights;
        self.slice_mut(span)
    }
    /// Mutable view of stack `i`'s `fc_2_biases`.
    pub fn fc_2_biases_mut(&mut self, i: usize) -> &mut [i32] {
        let span = self.spans.stacks[i].fc_2_biases;
        self.slice_mut(span)
    }
    /// Mutable view of stack `i`'s `fc_2_weights`.
    pub fn fc_2_weights_mut(&mut self, i: usize) -> &mut [i8] {
        let span = self.spans.stacks[i].fc_2_weights;
        self.slice_mut(span)
    }

    /// Number of layer stacks the layout carries.
    pub fn layer_stacks(&self) -> usize {
        self.spans.stacks.len()
    }

    /// The raw parameter bytes being filled, in the layout the kernels read
    /// them — what the source file's reader hands out so the same bytes can be
    /// held against the file the build wrote.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.spans.total_bytes]
    }

    /// Overwrite the whole parameter region with `bytes`, which must be the
    /// region's exact size.
    pub fn fill(&mut self, bytes: &[u8]) {
        assert_eq!(
            bytes.len(),
            self.spans.total_bytes,
            "the parameter region's size is fixed by the dimensions",
        );
        self.bytes[..bytes.len()].copy_from_slice(bytes);
    }

    /// Finish: consume the filled region and produce the network (no copy).
    pub fn build(self) -> NnueNetwork {
        let base = self.bytes.as_ptr() as *mut u8;
        // SAFETY: the block holds `total_bytes` zeroed — hence initialised —
        // bytes on a large-page boundary, so every span is aligned and
        // in-bounds, and it moves into the network as the backing, which keeps
        // it alive for exactly as long as the views.
        unsafe {
            NnueNetwork::over(
                self.header,
                self.sha256,
                &self.dims,
                base,
                Backing::Owned(self.bytes),
            )
        }
    }

    /// A typed mutable view of `span` in the block being filled.
    fn slice_mut<T>(&mut self, span: Span) -> &mut [T] {
        let bytes = &mut self.bytes[span.offset..span.offset + span.count * size_of::<T>()];
        // SAFETY: the span starts on a 64-byte boundary inside the block, so
        // the pointer is aligned for `T`, and it covers exactly `span.count`
        // elements of zeroed — hence valid — bytes. The borrow is tied to
        // `&mut self`, so no other view of the block is live.
        unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut T, span.count) }
    }
}

/// Errors returned when the engine's network cannot be read. Every variant
/// carries a human-readable reason; nothing here panics on a malformed file.
#[derive(Debug)]
pub enum NnueError {
    Io {
        path: String,
        source: std::io::Error,
    },
    SizeMismatch {
        expected: usize,
        got: usize,
    },
    InvalidFormat {
        reason: String,
    },
    /// The evaluation file is not the one this binary was built to read.
    Mismatch {
        reason: String,
    },
    NotLoaded,
}

impl fmt::Display for NnueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NnueError::Io { path, source } => {
                write!(f, "failed to open NNUE file {path}: {source}")
            }
            NnueError::SizeMismatch { expected, got } => write!(
                f,
                "NNUE file has unexpected size: expected {expected} bytes, got {got}"
            ),
            NnueError::InvalidFormat { reason } => {
                write!(f, "NNUE file is malformed: {reason}")
            }
            NnueError::Mismatch { reason } => write!(
                f,
                "the evaluation file is not the one this build reads: {reason}"
            ),
            NnueError::NotLoaded => write!(
                f,
                "no NNUE network loaded; the engine reads one from the evaluation directory beside it"
            ),
        }
    }
}

impl std::error::Error for NnueError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            NnueError::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nnue_layout::{DATA_BYTES, SECTION_ALIGN, data_bytes};

    /// A small synthetic net for the layout tests: a tiny feature transformer,
    /// standard FC shapes and stack count.
    fn small_net() -> NnueNetwork {
        let dims = NetDims {
            num_features: 3,
            ..NetDims::STANDARD
        };
        let header = NetHeader {
            version: 0,
            hash: 0,
            arch_id: "layout-test".to_string(),
        };
        NnueNetworkBuilder::with_dims(header, [0u8; 32], &dims).build()
    }

    /// Every parameter sub-array's `(start_addr, byte_len)`.
    fn sub_arrays(net: &NnueNetwork) -> Vec<(usize, usize)> {
        let mut v = vec![
            (net.ft_biases.as_ptr() as usize, net.ft_biases.len() * 2),
            (net.ft_weights.as_ptr() as usize, net.ft_weights.len() * 2),
        ];
        for s in &net.stacks {
            v.push((s.fc_0_biases.as_ptr() as usize, s.fc_0_biases.len() * 4));
            v.push((s.fc_0_weights.as_ptr() as usize, s.fc_0_weights.len()));
            v.push((s.fc_1_biases.as_ptr() as usize, s.fc_1_biases.len() * 4));
            v.push((s.fc_1_weights.as_ptr() as usize, s.fc_1_weights.len()));
            v.push((s.fc_2_biases.as_ptr() as usize, s.fc_2_biases.len() * 4));
            v.push((s.fc_2_weights.as_ptr() as usize, s.fc_2_weights.len()));
        }
        v
    }

    #[test]
    fn every_sub_array_is_64_byte_aligned() {
        let net = small_net();
        for (addr, _) in sub_arrays(&net) {
            assert_eq!(
                addr % SECTION_ALIGN,
                0,
                "sub-array at {addr:#x} is not 64-aligned"
            );
        }
    }

    #[test]
    fn sub_arrays_are_disjoint_and_inside_one_region() {
        let net = small_net();
        let mut spans = sub_arrays(&net);
        spans.sort_by_key(|&(addr, _)| addr);
        // Pairwise non-overlap: each start is at or after the previous end.
        for w in spans.windows(2) {
            let (start_a, len_a) = w[0];
            let (start_b, _) = w[1];
            assert!(
                start_b >= start_a + len_a,
                "sub-arrays overlap: [{start_a:#x}, +{len_a}) then {start_b:#x}",
            );
        }
        // The whole spread fits inside the one region the dimensions size.
        let dims = NetDims {
            num_features: 3,
            ..NetDims::STANDARD
        };
        let (first, _) = *spans.first().unwrap();
        let (last, last_len) = *spans.last().unwrap();
        assert!((last + last_len) - first <= data_bytes(&dims));
    }

    #[test]
    fn builder_fills_are_visible_through_the_views() {
        let dims = NetDims {
            num_features: 3,
            ..NetDims::STANDARD
        };
        let mut b = NnueNetworkBuilder::with_dims(
            NetHeader {
                version: 1,
                hash: 2,
                arch_id: "fill".to_string(),
            },
            [7u8; 32],
            &dims,
        );
        b.ft_biases_mut()[0] = -321;
        b.ft_weights_mut()[dims.hidden_size] = 99; // second feature column, lane 0
        b.fc_0_biases_mut(1)[3] = 77;
        b.fc_2_weights_mut(0)[1] = -5;
        let net = b.build();
        assert_eq!(net.ft_biases[0], -321);
        assert_eq!(net.ft_weights[dims.hidden_size], 99);
        assert_eq!(net.stacks[1].fc_0_biases[3], 77);
        assert_eq!(net.stacks[0].fc_2_weights[1], -5);
        assert_eq!(net.header.version, 1);
        assert_eq!(net.sha256, [7u8; 32]);
    }

    #[test]
    fn every_stack_sits_inside_the_network_itself() {
        let net = small_net();
        let base = &net as *const NnueNetwork as usize;
        for stack in &net.stacks {
            let at = stack as *const NetworkStack as usize;
            assert!(
                at >= base && at + size_of::<NetworkStack>() <= base + size_of::<NnueNetwork>(),
                "a stack must be reachable at an offset from the network, not through a pointer",
            );
        }
    }

    #[test]
    fn the_regions_size_matches_the_layout_walk() {
        assert_eq!(DATA_BYTES, net_spans(&NetDims::STANDARD).total_bytes);
    }

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
    fn not_loaded_has_human_readable_message() {
        let msg = format!("{}", NnueError::NotLoaded);
        assert!(!msg.is_empty());
        assert!(msg.contains("no NNUE network loaded"));
        assert!(msg.contains("evaluation directory"));
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
    fn a_mismatch_names_what_differs() {
        let msg = format!(
            "{}",
            NnueError::Mismatch {
                reason: "made from no network file".to_string()
            }
        );
        assert!(msg.contains("made from no network file"), "got: {msg}");
    }
}
