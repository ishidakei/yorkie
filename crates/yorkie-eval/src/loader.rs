//! Reading a source network file (`nn.bin`) into a network in this process's
//! own memory.
//!
//! The engine never does this: its parameters are laid out for the kernels
//! ahead of time and read in place. What needs it is the tooling that holds the
//! two against each other — a parity run reads the same source file the build
//! converted and evaluates with it — so the reader is compiled in only where
//! that is asked for.
//!
//! The file format, its validation and the transformation into the kernels'
//! layout are the definition the build script converts with, so this reader and
//! that conversion cannot produce different parameters.

use std::path::Path;

use crate::nnue_layout::{NetDims, NetHeader};
use crate::nnue_source::convert;
use crate::types::{NnueError, OwnedNetwork};

/// Reads and validates the SFNN-1536 network file at `path`, discarding any
/// non-fatal warnings. Use [`load_network_with_warnings`] to surface them.
pub fn load_network(path: &Path) -> Result<OwnedNetwork, NnueError> {
    load_network_with_warnings(path).map(|(net, _warnings)| net)
}

/// Reads and validates the SFNN-1536 network file at `path`, returning the
/// network together with any non-fatal warning bodies. Structural problems and
/// a version mismatch still fail with an error.
pub fn load_network_with_warnings(path: &Path) -> Result<(OwnedNetwork, Vec<String>), NnueError> {
    let bytes = std::fs::read(path).map_err(|e| NnueError::Io {
        path: path.display().to_string(),
        source: e,
    })?;
    network_from_bytes(&bytes, &NetDims::STANDARD)
}

/// The network `bytes` describe, in a freshly allocated block.
fn network_from_bytes(
    bytes: &[u8],
    dims: &NetDims,
) -> Result<(OwnedNetwork, Vec<String>), NnueError> {
    let converted = convert(bytes, dims).map_err(|reason| NnueError::InvalidFormat { reason })?;
    let mut owned = OwnedNetwork::zeroed();
    owned.fill(&converted.data);
    Ok((owned, converted.warnings))
}

/// The parameters `bytes` lay out, as the kernels read them — the same region
/// the build script writes into the evaluation file.
pub fn parameter_bytes(bytes: &[u8], dims: &NetDims) -> Result<Vec<u8>, NnueError> {
    convert(bytes, dims)
        .map(|converted| converted.data)
        .map_err(|reason| NnueError::InvalidFormat { reason })
}

/// The identity fields of the network file `bytes`, without laying its
/// parameters out.
pub fn source_header(bytes: &[u8], dims: &NetDims) -> Result<NetHeader, NnueError> {
    convert(bytes, dims)
        .map(|converted| converted.net)
        .map_err(|reason| NnueError::InvalidFormat { reason })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nnue_layout::{LAYER_STACKS, NetSpans, Span, net_spans};
    use crate::nnue_source::{
        ARCH_STRING, FT_HASH, LEB128_MAGIC, NET_HASH, NNUE_HASH_VALUE, NNUE_VERSION,
        SECTION_HASH_WARNING, sha256,
    };

    /// The conversion's output held against the dimensions that produced it.
    ///
    /// The engine's own layout is a set of constants for the shipped
    /// dimensions, so a test that varies them reads the parameters through the
    /// walked spans instead.
    #[derive(Debug)]
    struct Converted {
        header: NetHeader,
        data: Vec<u8>,
        spans: NetSpans,
    }

    impl Converted {
        /// The `i16` array `span` describes, decoded byte by byte: the
        /// conversion hands back a plain `Vec<u8>`, whose start carries no
        /// alignment the elements could be read through directly.
        fn i16_array(&self, span: Span) -> Vec<i16> {
            self.data[span.offset..span.offset + span.count * 2]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&b| i16::from_le_bytes(b))
                .collect()
        }

        fn ft_biases(&self) -> Vec<i16> {
            self.i16_array(self.spans.ft_biases)
        }

        fn ft_weights(&self) -> Vec<i16> {
            self.i16_array(self.spans.ft_weights)
        }

        fn stacks(&self) -> usize {
            self.spans.stacks.len()
        }
    }

    /// [`network_from_bytes`] for arbitrary dimensions: what the conversion
    /// produced, without the engine's constant layout over it.
    fn network_from_bytes(
        bytes: &[u8],
        dims: &NetDims,
    ) -> Result<(Converted, Vec<String>), NnueError> {
        let converted =
            convert(bytes, dims).map_err(|reason| NnueError::InvalidFormat { reason })?;
        Ok((
            Converted {
                header: converted.net,
                data: converted.data,
                spans: net_spans(dims),
            },
            converted.warnings,
        ))
    }

    const TEST_DIMS: NetDims = NetDims {
        hidden_size: 4,
        num_features: 2,
        layer_stacks: LAYER_STACKS,
        fc_0_output: 2,
        fc_0_padded_input: 4,
        fc_1_output: 2,
        fc_1_padded_input: 2,
        fc_2_output: 1,
        fc_2_padded_input: 2,
    };

    fn build_valid_bytes(dims: &NetDims, arch: &str) -> Vec<u8> {
        build_valid_bytes_arch(dims, arch.as_bytes())
    }

    /// Like [`build_valid_bytes`] but takes the architecture field as raw bytes,
    /// so a test can feed a non-UTF-8 arch string.
    fn build_valid_bytes_arch(dims: &NetDims, arch: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&NNUE_VERSION.to_le_bytes());
        out.extend_from_slice(&NNUE_HASH_VALUE.to_le_bytes());
        out.extend_from_slice(&(arch.len() as u32).to_le_bytes());
        out.extend_from_slice(arch);
        out.extend_from_slice(&FT_HASH.to_le_bytes());
        append_zero_leb128_block(&mut out, dims.hidden_size);
        append_zero_leb128_block(&mut out, dims.hidden_size * dims.num_features);
        for _ in 0..dims.layer_stacks {
            out.extend_from_slice(&NET_HASH.to_le_bytes());
            append_zeros(&mut out, dims.fc_0_output * 4);
            append_zeros(&mut out, dims.fc_0_output * dims.fc_0_padded_input);
            append_zeros(&mut out, dims.fc_1_output * 4);
            append_zeros(&mut out, dims.fc_1_output * dims.fc_1_padded_input);
            append_zeros(&mut out, dims.fc_2_output * 4);
            append_zeros(&mut out, dims.fc_2_output * dims.fc_2_padded_input);
        }
        out
    }

    fn append_zero_leb128_block(out: &mut Vec<u8>, count: usize) {
        out.extend_from_slice(LEB128_MAGIC);
        out.extend_from_slice(&(count as u32).to_le_bytes());
        out.resize(out.len() + count, 0);
    }

    fn append_zeros(out: &mut Vec<u8>, n: usize) {
        out.resize(out.len() + n, 0);
    }

    fn append_signed_leb128_block(out: &mut Vec<u8>, values: &[i16]) {
        out.extend_from_slice(LEB128_MAGIC);
        let len_pos = out.len();
        out.extend_from_slice(&0u32.to_le_bytes());
        let payload_start = out.len();
        for &v in values {
            encode_signed_leb128(out, v as i64);
        }
        let bytes_left = (out.len() - payload_start) as u32;
        out[len_pos..len_pos + 4].copy_from_slice(&bytes_left.to_le_bytes());
    }

    fn encode_signed_leb128(out: &mut Vec<u8>, mut value: i64) {
        loop {
            let byte = (value as u8) & 0x7F;
            // An arithmetic shift preserves the sign bit signed LEB128 uses to
            // end the byte stream.
            value >>= 7;
            let sign_bit = byte & 0x40;
            if (value == 0 && sign_bit == 0) || (value == -1 && sign_bit != 0) {
                out.push(byte);
                return;
            }
            out.push(byte | 0x80);
        }
    }

    fn build_bytes_with_ft(dims: &NetDims, arch: &str, biases: &[i16], weights: &[i16]) -> Vec<u8> {
        assert_eq!(biases.len(), dims.hidden_size);
        assert_eq!(weights.len(), dims.hidden_size * dims.num_features);
        let mut out = Vec::new();
        out.extend_from_slice(&NNUE_VERSION.to_le_bytes());
        out.extend_from_slice(&NNUE_HASH_VALUE.to_le_bytes());
        out.extend_from_slice(&(arch.len() as u32).to_le_bytes());
        out.extend_from_slice(arch.as_bytes());
        out.extend_from_slice(&FT_HASH.to_le_bytes());
        append_signed_leb128_block(&mut out, biases);
        append_signed_leb128_block(&mut out, weights);
        for _ in 0..dims.layer_stacks {
            out.extend_from_slice(&NET_HASH.to_le_bytes());
            append_zeros(&mut out, dims.fc_0_output * 4);
            append_zeros(&mut out, dims.fc_0_output * dims.fc_0_padded_input);
            append_zeros(&mut out, dims.fc_1_output * 4);
            append_zeros(&mut out, dims.fc_1_output * dims.fc_1_padded_input);
            append_zeros(&mut out, dims.fc_2_output * 4);
            append_zeros(&mut out, dims.fc_2_output * dims.fc_2_padded_input);
        }
        out
    }

    const SCALE_DIMS: NetDims = NetDims {
        hidden_size: 5,
        num_features: 2,
        layer_stacks: LAYER_STACKS,
        fc_0_output: 2,
        fc_0_padded_input: 5,
        fc_1_output: 2,
        fc_1_padded_input: 2,
        fc_2_output: 1,
        fc_2_padded_input: 2,
    };

    #[test]
    fn valid_header_round_trips() {
        let bytes = build_valid_bytes(&TEST_DIMS, ARCH_STRING);
        let (net, warnings) = network_from_bytes(&bytes, &TEST_DIMS).expect("should parse");
        assert!(
            warnings.is_empty(),
            "clean file must not warn: {warnings:?}"
        );
        assert_eq!(net.header.version, NNUE_VERSION);
        assert_eq!(net.header.hash, NNUE_HASH_VALUE);
        assert_eq!(net.header.arch_id, ARCH_STRING);
        assert_eq!(net.ft_biases().len(), TEST_DIMS.hidden_size);
        assert_eq!(
            net.ft_weights().len(),
            TEST_DIMS.hidden_size * TEST_DIMS.num_features
        );
        assert_eq!(net.stacks(), TEST_DIMS.layer_stacks);
        let stack = net.spans.stacks[0];
        assert_eq!(stack.fc_0_biases.count, TEST_DIMS.fc_0_output);
        assert_eq!(
            stack.fc_0_weights.count,
            TEST_DIMS.fc_0_output * TEST_DIMS.fc_0_padded_input
        );
        assert_eq!(stack.fc_1_biases.count, TEST_DIMS.fc_1_output);
        assert_eq!(
            stack.fc_1_weights.count,
            TEST_DIMS.fc_1_output * TEST_DIMS.fc_1_padded_input
        );
        assert_eq!(stack.fc_2_biases.count, TEST_DIMS.fc_2_output);
        assert_eq!(
            stack.fc_2_weights.count,
            TEST_DIMS.fc_2_output * TEST_DIMS.fc_2_padded_input
        );
        assert!(net.ft_biases().iter().all(|&x| x == 0));
        assert!(net.ft_weights().iter().all(|&x| x == 0));
    }

    #[test]
    fn different_arch_string_loads_without_complaint() {
        // The architecture string is never compared, so with matching hashes a
        // different one loads cleanly.
        let bytes = build_valid_bytes(&TEST_DIMS, "SFNNwoP1024");
        let (net, warnings) = network_from_bytes(&bytes, &TEST_DIMS).expect("should parse");
        assert!(
            warnings.is_empty(),
            "arch difference must not warn: {warnings:?}"
        );
        assert_eq!(net.header.arch_id, "SFNNwoP1024");
        assert_eq!(net.stacks(), TEST_DIMS.layer_stacks);
    }

    #[test]
    fn non_utf8_arch_string_loads_without_complaint() {
        // Raw, non-UTF-8 arch bytes are rendered lossily and never rejected.
        let bytes = build_valid_bytes_arch(&TEST_DIMS, &[0xFF, 0xFE, 0x00, 0x80]);
        let (net, warnings) = network_from_bytes(&bytes, &TEST_DIMS).expect("should parse");
        assert!(
            warnings.is_empty(),
            "arch bytes must not warn: {warnings:?}"
        );
        assert!(
            net.header.arch_id.contains('\u{FFFD}'),
            "non-UTF-8 bytes should render lossily, got {:?}",
            net.header.arch_id
        );
        assert_eq!(net.stacks(), TEST_DIMS.layer_stacks);
    }

    #[test]
    fn short_read_is_rejected() {
        let bytes = build_valid_bytes(&TEST_DIMS, ARCH_STRING);
        let truncated = &bytes[..bytes.len() - 5];
        let err = network_from_bytes(truncated, &TEST_DIMS).unwrap_err();
        match err {
            NnueError::InvalidFormat { reason } => assert!(
                reason.contains("unexpected size"),
                "expected a size complaint, got: {reason}"
            ),
            other => panic!("expected InvalidFormat, got {other:?}"),
        }
    }

    #[test]
    fn oversize_trailer_is_rejected() {
        let mut bytes = build_valid_bytes(&TEST_DIMS, ARCH_STRING);
        bytes.extend_from_slice(&[0xAA, 0xBB, 0xCC]);
        let err = network_from_bytes(&bytes, &TEST_DIMS).unwrap_err();
        match err {
            NnueError::InvalidFormat { reason } => assert!(
                reason.contains("unexpected size"),
                "expected a size complaint, got: {reason}"
            ),
            other => panic!("expected InvalidFormat, got {other:?}"),
        }
    }

    #[test]
    fn wrong_version_is_hard_rejected_with_reference_message() {
        let mut bytes = build_valid_bytes(&TEST_DIMS, ARCH_STRING);
        bytes[..4].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        let err = network_from_bytes(&bytes, &TEST_DIMS).unwrap_err();
        match err {
            NnueError::InvalidFormat { reason } => {
                assert!(
                    reason.contains("NNUE header version mismatch: expected")
                        && reason.contains("got"),
                    "expected the reference version-mismatch message shape, got: {reason}"
                );
            }
            other => panic!("expected InvalidFormat, got {other:?}"),
        }
    }

    #[test]
    fn wrong_top_level_hash_loads_with_warning() {
        let mut bytes = build_valid_bytes(&TEST_DIMS, ARCH_STRING);
        // The top-level hash sits right after the 4-byte version.
        bytes[4..8].copy_from_slice(&0x1234_5678u32.to_le_bytes());
        let (net, warnings) = network_from_bytes(&bytes, &TEST_DIMS).expect("should load");
        // Exactly one warning: the file-level hash mismatch, naming both arches.
        assert_eq!(
            warnings.len(),
            1,
            "expected a single warning, got {warnings:?}"
        );
        let w = &warnings[0];
        assert!(
            w.contains("Warning: NNUE hash mismatch: expected"),
            "got: {w}"
        );
        assert!(w.contains(&format!("got {}", 0x1234_5678u32)), "got: {w}");
        assert!(
            w.contains(&format!("arch_expected={ARCH_STRING}")),
            "got: {w}"
        );
        // Parameters are still read intact.
        assert_eq!(net.ft_biases().len(), TEST_DIMS.hidden_size);
        assert_eq!(net.stacks(), TEST_DIMS.layer_stacks);
        assert!(net.ft_biases().iter().all(|&x| x == 0));
    }

    #[test]
    fn wrong_ft_hash_loads_with_section_warning() {
        let mut bytes = build_valid_bytes(&TEST_DIMS, ARCH_STRING);
        // ft_hash follows version(4) + hash(4) + arch_size(4) + arch bytes.
        let ft_hash_pos = 12 + ARCH_STRING.len();
        bytes[ft_hash_pos..ft_hash_pos + 4].copy_from_slice(&0x0BAD_F00Du32.to_le_bytes());
        let (net, warnings) = network_from_bytes(&bytes, &TEST_DIMS).expect("should load");
        assert_eq!(warnings, vec![SECTION_HASH_WARNING.to_string()]);
        assert_eq!(net.stacks(), TEST_DIMS.layer_stacks);
    }

    #[test]
    fn wrong_net_hash_loads_with_section_warning() {
        let mut bytes = build_valid_bytes(&TEST_DIMS, ARCH_STRING);
        // The first layer stack's `net_hash` is the last word before its
        // parameter blocks.
        let ft_bias_block = LEB128_MAGIC.len() + 4 + TEST_DIMS.hidden_size;
        let ft_weight_block =
            LEB128_MAGIC.len() + 4 + TEST_DIMS.hidden_size * TEST_DIMS.num_features;
        let net_hash_pos = 12 + ARCH_STRING.len() + 4 + ft_bias_block + ft_weight_block;
        bytes[net_hash_pos..net_hash_pos + 4].copy_from_slice(&0x0BAD_CAFEu32.to_le_bytes());
        let (net, warnings) = network_from_bytes(&bytes, &TEST_DIMS).expect("should load");
        assert_eq!(warnings, vec![SECTION_HASH_WARNING.to_string()]);
        assert_eq!(net.stacks(), TEST_DIMS.layer_stacks);
    }

    #[test]
    fn corrupted_leb128_magic_is_rejected() {
        let mut bytes = build_valid_bytes(&TEST_DIMS, ARCH_STRING);
        let magic_start = 12 + ARCH_STRING.len() + 4;
        bytes[magic_start] = b'X';
        let err = network_from_bytes(&bytes, &TEST_DIMS).unwrap_err();
        assert!(
            matches!(err, NnueError::InvalidFormat { .. }),
            "expected InvalidFormat, got {err:?}"
        );
    }

    #[test]
    fn ft_scale_doubles_biases_and_weights() {
        let biases: [i16; 5] = [0, 1, -1, 16_383, -16_384];
        let weight_row: [i16; 5] = [0, 1, -1, 16_383, -16_384];
        let mut weights = Vec::with_capacity(SCALE_DIMS.hidden_size * SCALE_DIMS.num_features);
        for _ in 0..SCALE_DIMS.num_features {
            weights.extend_from_slice(&weight_row);
        }

        let bytes = build_bytes_with_ft(&SCALE_DIMS, ARCH_STRING, &biases, &weights);
        let (net, _warnings) = network_from_bytes(&bytes, &SCALE_DIMS).expect("should parse");

        let expected_row: [i16; 5] = [0, 2, -2, 32_766, -32_768];
        assert_eq!(net.ft_biases(), &expected_row[..]);
        assert_eq!(net.ft_weights().len(), weights.len());
        for chunk in net.ft_weights().chunks(expected_row.len()) {
            assert_eq!(chunk, &expected_row[..]);
        }
    }

    #[test]
    fn ft_scale_rejects_overflow_bias() {
        let biases: [i16; 5] = [16_384, 0, 0, 0, 0];
        let weights = vec![0i16; SCALE_DIMS.hidden_size * SCALE_DIMS.num_features];

        let bytes = build_bytes_with_ft(&SCALE_DIMS, ARCH_STRING, &biases, &weights);
        let err = network_from_bytes(&bytes, &SCALE_DIMS).unwrap_err();

        match err {
            NnueError::InvalidFormat { reason } => {
                assert!(
                    reason.contains("16384"),
                    "expected message to name the offending value, got: {reason}"
                );
                assert!(
                    reason.contains("out of range") || reason.contains("overflow"),
                    "expected message to mention overflow / out-of-range, got: {reason}"
                );
            }
            other => panic!("expected InvalidFormat, got {other:?}"),
        }
    }

    #[test]
    fn sha256_matches_nist_empty_vector() {
        assert_eq!(
            sha256(b""),
            hex_decode("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
        );
    }

    #[test]
    fn sha256_matches_nist_abc_vector() {
        assert_eq!(
            sha256(b"abc"),
            hex_decode("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        );
    }

    #[test]
    fn sha256_matches_nist_long_vector() {
        assert_eq!(
            sha256(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            hex_decode("248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1")
        );
    }

    fn hex_decode(s: &str) -> [u8; 32] {
        assert_eq!(s.len(), 64);
        let mut out = [0u8; 32];
        for (i, chunk) in s.as_bytes().chunks(2).enumerate() {
            out[i] = u8::from_str_radix(std::str::from_utf8(chunk).unwrap(), 16).unwrap();
        }
        out
    }
}
