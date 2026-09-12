//! Where the parameters of SFNN-1536 are, and the loader error type.
//!
//! The dimensions and the byte layout the parameters sit in are the shared
//! definition the build script writes the evaluation file from. What this
//! module adds is the addressing: every array's offset inside the parameter
//! region is a compile-time constant, so a network is nothing but the address
//! that region starts at, and its accessors are `base + literal` — no view to
//! build, no table to walk, nothing to keep alive.
//!
//! *Which* address is a type, not a value: [`NetworkParams`] is the contract,
//! [`crate::network_file::Region`] answers it with the linker's address and
//! therefore has no fields, and [`PtrNetwork`] answers it with an address
//! chosen while the process runs.
//!
//! The prose here follows the reference's naming: the **FT layer** is the
//! feature transformer, and **L1 / L2 / L3** are the dense layers after it. The
//! FT layer is never called "L1". In the identifiers below, L1 is `fc_0`, L2 is
//! `fc_1` and L3 is `fc_2`.

use std::path::PathBuf;

use yorkie_state::TextWriter;
#[cfg(any(test, feature = "source-network"))]
use yorkie_storage::LargePageArray;

use crate::nnue_layout::{DATA_BYTES, SECTION_ALIGN, SPANS, Span, StackSpans};
pub use crate::nnue_layout::{
    FC_0_INPUT_DIMS, FC_0_OUTPUT_DIMS, FC_0_PADDED_INPUT_DIMS, FC_1_INPUT_DIMS, FC_1_OUTPUT_DIMS,
    FC_1_PADDED_INPUT_DIMS, FC_2_INPUT_DIMS, FC_2_OUTPUT_DIMS, FC_2_PADDED_INPUT_DIMS, HIDDEN_SIZE,
    HIDDEN1_DIMS, HIDDEN2_DIMS, LAYER_STACKS, NUM_FEATURES, NetDims, NetHeader,
};

/// The feature transformer's two arrays, at the offsets the layout fixed.
const FT_BIASES: Span = SPANS.ft_biases;
const FT_WEIGHTS: Span = SPANS.ft_weights;

/// The first layer stack's six arrays, and the distance from one stack to the
/// next. Every stack has the same shape, so the stacks are evenly spaced and a
/// bucket's arrays are reached by multiplying rather than by looking an offset
/// up.
const STACK_0: StackSpans = SPANS.stacks[0];
const STACK_STRIDE: usize = SPANS.stacks[1].fc_0_biases.offset - STACK_0.fc_0_biases.offset;

// The even spacing the stride assumes, proved over every stack rather than
// asserted about the first two.
const _: () = {
    let mut i = 0;
    while i < LAYER_STACKS {
        let stack = SPANS.stacks[i];
        let step = i * STACK_STRIDE;
        assert!(stack.fc_0_biases.offset == STACK_0.fc_0_biases.offset + step);
        assert!(stack.fc_0_weights.offset == STACK_0.fc_0_weights.offset + step);
        assert!(stack.fc_1_biases.offset == STACK_0.fc_1_biases.offset + step);
        assert!(stack.fc_1_weights.offset == STACK_0.fc_1_weights.offset + step);
        assert!(stack.fc_2_biases.offset == STACK_0.fc_2_biases.offset + step);
        assert!(stack.fc_2_weights.offset == STACK_0.fc_2_weights.offset + step);
        i += 1;
    }
};

// Every array starts on the cache line the AVX-512 loads assume, and the whole
// walk fits the region the file's data section is sized to.
const _: () = assert!(FT_BIASES.offset.is_multiple_of(SECTION_ALIGN));
const _: () = assert!(FT_WEIGHTS.offset.is_multiple_of(SECTION_ALIGN));
const _: () = assert!(STACK_STRIDE.is_multiple_of(SECTION_ALIGN));
const _: () = assert!(SPANS.total_bytes == DATA_BYTES);

/// One layer stack's parameter arrays: the address its `fc_0_biases` starts at,
/// with the other five a literal away.
#[derive(Clone, Copy, Debug)]
pub struct NetStack {
    /// The stack's own base — the region's base advanced by the stack's stride
    /// — so the six offsets below are the same literals for every bucket.
    base: *const u8,
}

impl NetStack {
    pub fn fc_0_biases(&self) -> &[i32] {
        // SAFETY: as `NetworkParams::parameters`, for this array of this stack.
        unsafe { array(self.base, STACK_0.fc_0_biases) }
    }

    pub fn fc_0_weights(&self) -> &[i8] {
        // SAFETY: as above.
        unsafe { array(self.base, STACK_0.fc_0_weights) }
    }

    pub fn fc_1_biases(&self) -> &[i32] {
        // SAFETY: as above.
        unsafe { array(self.base, STACK_0.fc_1_biases) }
    }

    pub fn fc_1_weights(&self) -> &[i8] {
        // SAFETY: as above.
        unsafe { array(self.base, STACK_0.fc_1_weights) }
    }

    pub fn fc_2_biases(&self) -> &[i32] {
        // SAFETY: as above.
        unsafe { array(self.base, STACK_0.fc_2_biases) }
    }

    pub fn fc_2_weights(&self) -> &[i8] {
        // SAFETY: as above.
        unsafe { array(self.base, STACK_0.fc_2_weights) }
    }
}

/// The module [`NetworkParams`] is sealed against: implementing it outside this
/// crate would mean naming a trait that is not reachable from outside it.
pub(crate) mod sealed {
    pub trait Sealed {}
}

/// A network: the address its parameters start at, and the arrays a constant
/// away from it.
///
/// Everything but the address is a constant — which array sits where, how long
/// each one is, how far apart the layer stacks are — so an evaluation reaches a
/// parameter at the address plus a literal. There is no view to load, nothing
/// to reference-count, and nothing that has to be kept alive alongside it.
///
/// Which address is a *type*: [`crate::network_file::Region`] carries none at
/// all, because its parameters are where the linker put them, and
/// [`PtrNetwork`] carries one because its are wherever the process put them.
/// Everything that reads parameters — the kernels, the accumulator, the finny
/// cache, the search — is generic over this trait and monomorphised for the one
/// it is given. `Sized` is a supertrait for that reason: it makes
/// `dyn NetworkParams` a compile error, so nothing can turn the choice back
/// into a run-time one.
pub trait NetworkParams: sealed::Sealed + Copy + Send + Sync + Sized {
    /// The address the parameters start at.
    ///
    /// It must address at least `DATA_BYTES` initialised bytes, aligned to a
    /// 64-byte boundary and laid out as the shared layout describes, and they
    /// must stay there, unwritten, for as long as this network is read. Every
    /// accessor below relies on that, which is why the trait is sealed.
    fn parameters(&self) -> *const u8;

    /// The address and length of the parameters, for a caller placing them: a
    /// NUMA policy over the pages, or a huge-page hint.
    fn parameter_region(&self) -> (usize, usize) {
        (self.parameters() as usize, DATA_BYTES)
    }

    fn ft_biases(&self) -> &[i16] {
        // SAFETY: as `NetworkParams::parameters`, for the feature transformer's
        // biases.
        unsafe { array(self.parameters(), FT_BIASES) }
    }

    fn ft_weights(&self) -> &[i16] {
        // SAFETY: as above, for its weights.
        unsafe { array(self.parameters(), FT_WEIGHTS) }
    }

    /// Layer stack `bucket`'s six arrays.
    ///
    /// The stride carries the bucket, so what the stack's own accessors add is
    /// the first stack's offsets — the same literals whichever bucket this is.
    fn stack(&self, bucket: usize) -> NetStack {
        debug_assert!(bucket < LAYER_STACKS);
        NetStack {
            // SAFETY: `bucket` is below `LAYER_STACKS`, so the stack it names
            // lies inside the region the implementor vouched for.
            base: unsafe { self.parameters().add(bucket * STACK_STRIDE) },
        }
    }
}

/// A network at an address the process chose, rather than one the linker fixed.
///
/// The engine never plays with one: what it plays with is
/// [`crate::network_file::Region`], whose address is a constant. This is for
/// everything that reads parameters from somewhere else — a network a test made
/// up, and the source file's own reader — and it is what lets two different
/// networks exist in one process, which is the whole of why it is a second
/// type.
#[derive(Clone, Copy, Debug)]
pub struct PtrNetwork {
    base: *const u8,
}

// SAFETY: a network is read-only from the moment it exists — the parameters are
// written once, before any thread reads them — so handing the address to
// another thread exposes nothing a `&[u8]` would not.
unsafe impl Send for PtrNetwork {}
unsafe impl Sync for PtrNetwork {}

impl sealed::Sealed for PtrNetwork {}

impl PtrNetwork {
    /// The network whose parameters start at `base`.
    ///
    /// # Safety
    /// `base` must meet what [`NetworkParams::parameters`] requires of the
    /// address it returns.
    pub const unsafe fn over(base: *const u8) -> Self {
        Self { base }
    }
}

impl NetworkParams for PtrNetwork {
    fn parameters(&self) -> *const u8 {
        self.base
    }
}

/// The typed array `span` describes, measured from `base`.
///
/// # Safety
/// As [`NetworkParams::parameters`], for this one array.
unsafe fn array<'a, T>(base: *const u8, span: Span) -> &'a [T] {
    // SAFETY: the caller vouches for the region; every span starts on a
    // 64-byte boundary, hence aligned for `T`.
    unsafe { std::slice::from_raw_parts(base.add(span.offset) as *const T, span.count) }
}

/// A network in this process's own memory, filled array by array.
///
/// The engine never builds one — it reads the region the evaluation file was
/// placed in — so this is compiled only into the tooling that needs a network
/// it made up: a synthetic one a test writes parameter by parameter, and the
/// source file's own reader. The parameters go in the layout the kernels read,
/// which is the layout the constants above address, so the dimensions are the
/// shipped ones and nothing else.
#[cfg(any(test, feature = "source-network"))]
pub struct OwnedNetwork {
    bytes: LargePageArray<u8>,
}

#[cfg(any(test, feature = "source-network"))]
impl OwnedNetwork {
    /// A network of all-zero parameters.
    pub fn zeroed() -> Self {
        Self {
            bytes: LargePageArray::<u8>::zeroed(DATA_BYTES),
        }
    }

    /// The network these parameters make up.
    ///
    /// Borrowed from the owner, so nothing reads the parameters after the block
    /// holding them is gone.
    pub fn network(&self) -> PtrNetwork {
        // SAFETY: the block holds `DATA_BYTES` zeroed — hence initialised —
        // bytes on a large-page boundary, so every span is aligned and
        // in-bounds, and the borrow keeps it alive for as long as the network.
        unsafe { PtrNetwork::over(self.bytes.as_ptr()) }
    }

    /// Mutable view of `ft_biases` (`HIDDEN_SIZE` i16).
    pub fn ft_biases_mut(&mut self) -> &mut [i16] {
        self.slice_mut(FT_BIASES)
    }

    /// Mutable view of `ft_weights` (`HIDDEN_SIZE * NUM_FEATURES` i16).
    pub fn ft_weights_mut(&mut self) -> &mut [i16] {
        self.slice_mut(FT_WEIGHTS)
    }

    /// Mutable view of stack `i`'s `fc_0_biases`.
    pub fn fc_0_biases_mut(&mut self, i: usize) -> &mut [i32] {
        self.slice_mut(SPANS.stacks[i].fc_0_biases)
    }
    /// Mutable view of stack `i`'s `fc_0_weights`.
    pub fn fc_0_weights_mut(&mut self, i: usize) -> &mut [i8] {
        self.slice_mut(SPANS.stacks[i].fc_0_weights)
    }
    /// Mutable view of stack `i`'s `fc_1_biases`.
    pub fn fc_1_biases_mut(&mut self, i: usize) -> &mut [i32] {
        self.slice_mut(SPANS.stacks[i].fc_1_biases)
    }
    /// Mutable view of stack `i`'s `fc_1_weights`.
    pub fn fc_1_weights_mut(&mut self, i: usize) -> &mut [i8] {
        self.slice_mut(SPANS.stacks[i].fc_1_weights)
    }
    /// Mutable view of stack `i`'s `fc_2_biases`.
    pub fn fc_2_biases_mut(&mut self, i: usize) -> &mut [i32] {
        self.slice_mut(SPANS.stacks[i].fc_2_biases)
    }
    /// Mutable view of stack `i`'s `fc_2_weights`.
    pub fn fc_2_weights_mut(&mut self, i: usize) -> &mut [i8] {
        self.slice_mut(SPANS.stacks[i].fc_2_weights)
    }

    /// The raw parameter bytes, in the layout the kernels read them — what the
    /// source file's reader hands out so the same bytes can be held against the
    /// file the build wrote.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..DATA_BYTES]
    }

    /// Overwrite the whole parameter region with `bytes`, which must be the
    /// region's exact size.
    pub fn fill(&mut self, bytes: &[u8]) {
        assert_eq!(
            bytes.len(),
            DATA_BYTES,
            "the parameter region's size is fixed by the dimensions",
        );
        self.bytes[..bytes.len()].copy_from_slice(bytes);
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
/// carries a reason a notice can name; nothing here panics on a malformed file.
///
/// The two `reason` fields carry the text the file-format definition produced.
/// That definition is shared with the build script that writes the file, which
/// is where its wording lives; the bytes of it are taken once, in
/// [`Self::write_message`].
#[derive(Debug)]
pub enum NnueError {
    Io {
        path: PathBuf,
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

impl NnueError {
    /// This error's message, as the bytes a notice carries.
    ///
    /// The operating system's own description of an I/O failure is the one
    /// thing here that is not this project's text, and it arrives through
    /// `std::io::Error`; its error number is named instead, so composing the
    /// line needs no formatter.
    pub fn write_message(&self, out: &mut TextWriter<'_>) {
        match self {
            NnueError::Io { path, source } => {
                out.bytes(b"failed to open NNUE file ").path(path);
                out.bytes(b": errno ");
                match source.raw_os_error() {
                    Some(errno) => out.i64(i64::from(errno)),
                    None => out.bytes(b"unknown"),
                };
            }
            NnueError::SizeMismatch { expected, got } => {
                out.bytes(b"NNUE file has unexpected size: expected ")
                    .u64(*expected as u64)
                    .bytes(b" bytes, got ")
                    .u64(*got as u64);
            }
            NnueError::InvalidFormat { reason } => {
                out.bytes(b"NNUE file is malformed: ")
                    .bytes(reason.as_bytes());
            }
            NnueError::Mismatch { reason } => {
                out.bytes(b"the evaluation file is not the one this build reads: ")
                    .bytes(reason.as_bytes());
            }
            NnueError::NotLoaded => {
                out.bytes(
                    b"no NNUE network loaded; the engine reads one from the evaluation \
                      directory beside it",
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nnue_layout::net_spans;

    /// Every parameter sub-array's `(start_addr, byte_len)`.
    fn sub_arrays(net: &PtrNetwork) -> Vec<(usize, usize)> {
        let mut v = vec![
            (net.ft_biases().as_ptr() as usize, net.ft_biases().len() * 2),
            (
                net.ft_weights().as_ptr() as usize,
                net.ft_weights().len() * 2,
            ),
        ];
        for i in 0..LAYER_STACKS {
            let s = net.stack(i);
            v.push((s.fc_0_biases().as_ptr() as usize, s.fc_0_biases().len() * 4));
            v.push((s.fc_0_weights().as_ptr() as usize, s.fc_0_weights().len()));
            v.push((s.fc_1_biases().as_ptr() as usize, s.fc_1_biases().len() * 4));
            v.push((s.fc_1_weights().as_ptr() as usize, s.fc_1_weights().len()));
            v.push((s.fc_2_biases().as_ptr() as usize, s.fc_2_biases().len() * 4));
            v.push((s.fc_2_weights().as_ptr() as usize, s.fc_2_weights().len()));
        }
        v
    }

    #[test]
    fn every_sub_array_is_64_byte_aligned() {
        let owned = OwnedNetwork::zeroed();
        for (addr, _) in sub_arrays(&owned.network()) {
            assert_eq!(
                addr % SECTION_ALIGN,
                0,
                "sub-array at {addr:#x} is not 64-aligned"
            );
        }
    }

    #[test]
    fn sub_arrays_are_disjoint_and_inside_one_region() {
        let owned = OwnedNetwork::zeroed();
        let mut spans = sub_arrays(&owned.network());
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
        let (first, _) = *spans.first().unwrap();
        let (last, last_len) = *spans.last().unwrap();
        assert!((last + last_len) - first <= DATA_BYTES);
    }

    #[test]
    fn fills_are_visible_through_the_network() {
        let mut owned = OwnedNetwork::zeroed();
        owned.ft_biases_mut()[0] = -321;
        owned.ft_weights_mut()[HIDDEN_SIZE] = 99; // second feature column, lane 0
        owned.fc_0_biases_mut(1)[3] = 77;
        owned.fc_2_weights_mut(0)[1] = -5;
        let net = owned.network();
        assert_eq!(net.ft_biases()[0], -321);
        assert_eq!(net.ft_weights()[HIDDEN_SIZE], 99);
        assert_eq!(net.stack(1).fc_0_biases()[3], 77);
        assert_eq!(net.stack(0).fc_2_weights()[1], -5);
    }

    #[test]
    fn the_constant_layout_matches_the_walked_one() {
        let walked = net_spans(&NetDims::STANDARD);
        assert_eq!(SPANS.ft_biases, walked.ft_biases);
        assert_eq!(SPANS.ft_weights, walked.ft_weights);
        assert_eq!(SPANS.stacks.as_slice(), walked.stacks.as_slice());
        assert_eq!(SPANS.total_bytes, walked.total_bytes);
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

    /// An error's message as a string, for the assertions below.
    fn message(err: &NnueError) -> String {
        let mut bytes = [0u8; 512];
        let mut out = TextWriter::new(&mut bytes);
        err.write_message(&mut out);
        assert!(!out.overflowed(), "a message fits the buffer");
        String::from_utf8(out.as_bytes().to_vec()).expect("a message is ASCII")
    }

    #[test]
    fn not_loaded_has_human_readable_message() {
        let msg = message(&NnueError::NotLoaded);
        assert!(!msg.is_empty());
        assert!(msg.contains("no NNUE network loaded"));
        assert!(msg.contains("evaluation directory"));
    }

    #[test]
    fn invalid_format_carries_reason() {
        let msg = message(&NnueError::InvalidFormat {
            reason: "bad magic".to_string(),
        });
        assert!(msg.contains("bad magic"));
    }

    #[test]
    fn a_mismatch_names_what_differs() {
        let msg = message(&NnueError::Mismatch {
            reason: "made from no network file".to_string(),
        });
        assert!(msg.contains("made from no network file"), "got: {msg}");
    }

    #[test]
    fn an_io_failure_names_the_path_and_the_error_number() {
        let msg = message(&NnueError::Io {
            path: PathBuf::from("/srv/eval/nn.kernel.bin"),
            source: std::io::Error::from_raw_os_error(2),
        });
        assert!(msg.contains("/srv/eval/nn.kernel.bin"), "got: {msg}");
        assert!(msg.contains("errno 2"), "got: {msg}");
    }

    #[test]
    fn a_size_mismatch_names_both_sizes() {
        let msg = message(&NnueError::SizeMismatch {
            expected: 1024,
            got: 512,
        });
        assert!(msg.contains("expected 1024 bytes, got 512"), "got: {msg}");
    }
}
