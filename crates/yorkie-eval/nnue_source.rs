// Reading an SFNN-1536 network file (`nn.bin`) and transforming it into the
// data region the kernels read.
//
// Shared verbatim between this crate and its build script — the build script
// performs the transformation once, and the crate's parity tooling performs it
// again against the same source to check the result — so one definition is what
// keeps the two from producing different bytes. It is `#[path]`-included rather
// than shared as a crate because a build script cannot depend on a member of
// the workspace it is building.
//
// The format is ported from `ReadHeader` / `ReadParameters`. Which failures are
// fatal follows the reference:
//
// - The **version word** is a hard failure on mismatch: the file is a different
//   serialization format, so parsing cannot continue.
// - The **file-level, feature-transformer and layer-stack hashes** are only
//   warnings; the transformation continues and the parameters are read as
//   usual. They are topology-derived, but the reference tolerates old files, so
//   this does too, carrying the warnings into the kernel-layout file's header
//   for the engine to report.
// - The **architecture string** is read but never compared. It appears only
//   inside the file-hash warning, rendered lossily so non-UTF-8 bytes never
//   fail the read.
// - **Structural failures stay hard**: a short read, a bad LEB128 magic, an
//   out-of-range value, or trailing bytes after the last stack.

use crate::nnue_layout::{NetDims, NetHeader, Span, net_spans};

pub const NNUE_VERSION: u32 = 0x7AF3_2F16;
pub const NNUE_HASH_VALUE: u32 = 0x3C20_3B32;
pub const FT_HASH: u32 = 0x5F13_4AB8;
pub const NET_HASH: u32 = 0x6333_718A;
pub const ARCH_STRING: &str = "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm(Friend)[73305->1536x2],Network=AffineTransform[1<-32](ClippedReLU[32](AffineTransform[32<-15](ClippedReLU[15](AffineTransform[15<-3072](InputSlice[3072(0:3072)]))))){LayerStack=9}";
pub const LEB128_MAGIC: &[u8; 17] = b"COMPRESSED_LEB128";

/// Warning body emitted when a feature-transformer or layer-stack hash does not
/// match — the reference's `Detail::ReadParameters` text, spacing and all.
pub const SECTION_HASH_WARNING: &str = "Warning : nn.bin hash mismatch.";

/// The result of transforming one source file: what the file said it was, what
/// was non-fatally wrong with it, and the data region itself.
pub struct Converted {
    pub net: NetHeader,
    pub warnings: Vec<String>,
    /// The parameters in the layout the kernels read, byte for byte as the
    /// engine uses them in place.
    ///
    /// Multi-byte values are little-endian, which is the byte order of the
    /// machine the file is written on and read back on: the engine is built for
    /// one machine and reads this region as typed memory without touching it.
    pub data: Vec<u8>,
}

/// Read `bytes` as a network file of `dims` and lay its parameters out for the
/// kernels.
pub fn convert(bytes: &[u8], dims: &NetDims) -> Result<Converted, String> {
    let spans = net_spans(dims);
    let mut data = vec![0u8; spans.total_bytes];
    let mut warnings = Vec::new();
    let mut reader = ByteReader::new(bytes);

    let net = read_header(&mut reader, &mut warnings)?;

    let ft_hash = reader.read_u32_le()?;
    if ft_hash != FT_HASH {
        warnings.push(SECTION_HASH_WARNING.to_string());
    }
    read_leb128_i16_into(&mut reader, &mut data, spans.ft_biases)?;
    read_leb128_i16_into(&mut reader, &mut data, spans.ft_weights)?;
    for stack in &spans.stacks {
        let net_hash = reader.read_u32_le()?;
        if net_hash != NET_HASH {
            warnings.push(SECTION_HASH_WARNING.to_string());
        }
        // In the file's `fc_0`, `fc_1`, `fc_2` order.
        read_i32_into(&mut reader, &mut data, stack.fc_0_biases)?;
        read_i8_into(&mut reader, &mut data, stack.fc_0_weights)?;
        read_i32_into(&mut reader, &mut data, stack.fc_1_biases)?;
        read_i8_into(&mut reader, &mut data, stack.fc_1_weights)?;
        read_i32_into(&mut reader, &mut data, stack.fc_2_biases)?;
        read_i8_into(&mut reader, &mut data, stack.fc_2_weights)?;
    }
    reader.assert_eof()?;

    Ok(Converted {
        net,
        warnings,
        data,
    })
}

fn read_header(reader: &mut ByteReader, warnings: &mut Vec<String>) -> Result<NetHeader, String> {
    let version = reader.read_u32_le()?;
    if version != NNUE_VERSION {
        // The reference's `ReadHeader` message shape, exactly.
        return Err(format!(
            "NNUE header version mismatch: expected {NNUE_VERSION} got {version}"
        ));
    }
    let hash = reader.read_u32_le()?;
    let arch_size = reader.read_u32_le()?;
    // `read_slice` bounds the length against the file, so a short file fails
    // structurally rather than here.
    let arch_bytes = reader.read_slice(arch_size as usize)?;
    let arch_id = String::from_utf8_lossy(arch_bytes).into_owned();
    // The message names both the in-file and the expected architecture string.
    if hash != NNUE_HASH_VALUE {
        warnings.push(format!(
            "Warning: NNUE hash mismatch: expected {NNUE_HASH_VALUE} got {hash} \
             arch_in_file={arch_id} arch_expected={ARCH_STRING}"
        ));
    }
    Ok(NetHeader {
        version,
        hash,
        arch_id,
    })
}

/// Decode one signed-LEB128 block into `span`, scaling each value by two as the
/// feature transformer's parameters are stored at half amplitude.
fn read_leb128_i16_into(
    reader: &mut ByteReader,
    data: &mut [u8],
    span: Span,
) -> Result<(), String> {
    let count = span.count;
    let magic = reader.read_slice(LEB128_MAGIC.len())?;
    if magic != LEB128_MAGIC {
        return Err(format!(
            "expected LEB128 magic {:?}, got {:?}",
            std::str::from_utf8(LEB128_MAGIC).unwrap_or("<non-utf8>"),
            String::from_utf8_lossy(magic),
        ));
    }
    let bytes_left = reader.read_u32_le()? as usize;
    // Worst-case signed LEB128 for an `i16` is 3 bytes, which bounds the read
    // against bad input.
    let upper_bound = count.saturating_mul(3);
    if bytes_left > upper_bound {
        return Err(format!(
            "LEB128 bytes_left {bytes_left} exceeds upper bound {upper_bound} for {count} i16 values"
        ));
    }
    let payload = reader.read_slice(bytes_left)?;
    let mut pos = 0usize;
    for i in 0..count {
        let v = read_signed_leb128(payload, &mut pos)?;
        if !(i16::MIN as i64..=i16::MAX as i64).contains(&v) {
            return Err(format!("LEB128 value {v} out of i16 range"));
        }
        let scaled = v * 2;
        if !(i16::MIN as i64..=i16::MAX as i64).contains(&scaled) {
            return Err(format!(
                "feature-transformer value {v} out of range: ×2 scale overflows i16 \
                 (pre-scale bound is [-16_384, 16_383])"
            ));
        }
        let at = span.offset + i * 2;
        data[at..at + 2].copy_from_slice(&(scaled as i16).to_le_bytes());
    }
    if pos != payload.len() {
        return Err(format!(
            "LEB128 block has {} unused bytes after {count} values",
            payload.len() - pos,
        ));
    }
    Ok(())
}

fn read_signed_leb128(bytes: &[u8], pos: &mut usize) -> Result<i64, String> {
    let mut result: i64 = 0;
    let mut shift: u32 = 0;
    loop {
        if *pos >= bytes.len() {
            return Err("LEB128 value truncated".to_string());
        }
        let byte = bytes[*pos];
        *pos += 1;
        if shift >= 64 {
            return Err("LEB128 value exceeds 64 bits".to_string());
        }
        result |= ((byte & 0x7F) as i64).wrapping_shl(shift);
        shift += 7;
        if byte & 0x80 == 0 {
            if shift < 64 && (byte & 0x40) != 0 {
                result |= (-1i64).wrapping_shl(shift);
            }
            return Ok(result);
        }
    }
}

fn read_i32_into(reader: &mut ByteReader, data: &mut [u8], span: Span) -> Result<(), String> {
    let bytes = reader.read_slice(span.count * 4)?;
    data[span.offset..span.offset + span.count * 4].copy_from_slice(bytes);
    Ok(())
}

fn read_i8_into(reader: &mut ByteReader, data: &mut [u8], span: Span) -> Result<(), String> {
    let bytes = reader.read_slice(span.count)?;
    data[span.offset..span.offset + span.count].copy_from_slice(bytes);
    Ok(())
}

/// A forward-only reader over the source file's bytes. Every read is
/// bounds-checked, so a truncated file is an error and never a panic.
struct ByteReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn read_slice(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| "read offset overflows usize".to_string())?;
        if end > self.bytes.len() {
            return Err(format!(
                "NNUE file has unexpected size: expected {end} bytes, got {}",
                self.bytes.len()
            ));
        }
        let s = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn read_u32_le(&mut self) -> Result<u32, String> {
        let s = self.read_slice(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    fn assert_eof(&self) -> Result<(), String> {
        if self.pos != self.bytes.len() {
            Err(format!(
                "NNUE file has unexpected size: expected {} bytes, got {}",
                self.pos,
                self.bytes.len()
            ))
        } else {
            Ok(())
        }
    }
}

/// The SHA-256 of `data` — the identity a kernel-layout file records for the
/// source it was made from.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a_2f98,
        0x7137_4491,
        0xb5c0_fbcf,
        0xe9b5_dba5,
        0x3956_c25b,
        0x59f1_11f1,
        0x923f_82a4,
        0xab1c_5ed5,
        0xd807_aa98,
        0x1283_5b01,
        0x2431_85be,
        0x550c_7dc3,
        0x72be_5d74,
        0x80de_b1fe,
        0x9bdc_06a7,
        0xc19b_f174,
        0xe49b_69c1,
        0xefbe_4786,
        0x0fc1_9dc6,
        0x240c_a1cc,
        0x2de9_2c6f,
        0x4a74_84aa,
        0x5cb0_a9dc,
        0x76f9_88da,
        0x983e_5152,
        0xa831_c66d,
        0xb003_27c8,
        0xbf59_7fc7,
        0xc6e0_0bf3,
        0xd5a7_9147,
        0x06ca_6351,
        0x1429_2967,
        0x27b7_0a85,
        0x2e1b_2138,
        0x4d2c_6dfc,
        0x5338_0d13,
        0x650a_7354,
        0x766a_0abb,
        0x81c2_c92e,
        0x9272_2c85,
        0xa2bf_e8a1,
        0xa81a_664b,
        0xc24b_8b70,
        0xc76c_51a3,
        0xd192_e819,
        0xd699_0624,
        0xf40e_3585,
        0x106a_a070,
        0x19a4_c116,
        0x1e37_6c08,
        0x2748_774c,
        0x34b0_bcb5,
        0x391c_0cb3,
        0x4ed8_aa4a,
        0x5b9c_ca4f,
        0x682e_6ff3,
        0x748f_82ee,
        0x78a5_636f,
        0x84c8_7814,
        0x8cc7_0208,
        0x90be_fffa,
        0xa450_6ceb,
        0xbef9_a3f7,
        0xc671_78f2,
    ];

    const INIT: [u32; 8] = [
        0x6a09_e667,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];

    fn compress(h: &mut [u32; 8], block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = *h;
        for i in 0..64 {
            let big_s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let temp1 = hh
                .wrapping_add(big_s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let big_s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = big_s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut h = INIT;
    let bit_len: u64 = (data.len() as u64).wrapping_mul(8);

    let full_blocks = data.len() / 64;
    for i in 0..full_blocks {
        let mut block = [0u8; 64];
        block.copy_from_slice(&data[i * 64..(i + 1) * 64]);
        compress(&mut h, &block);
    }

    let remainder = &data[full_blocks * 64..];
    let mut tail = [0u8; 128];
    tail[..remainder.len()].copy_from_slice(remainder);
    tail[remainder.len()] = 0x80;
    if remainder.len() < 56 {
        tail[56..64].copy_from_slice(&bit_len.to_be_bytes());
        let mut block = [0u8; 64];
        block.copy_from_slice(&tail[..64]);
        compress(&mut h, &block);
    } else {
        tail[120..128].copy_from_slice(&bit_len.to_be_bytes());
        let mut b1 = [0u8; 64];
        b1.copy_from_slice(&tail[..64]);
        compress(&mut h, &b1);
        let mut b2 = [0u8; 64];
        b2.copy_from_slice(&tail[64..128]);
        compress(&mut h, &b2);
    }

    let mut out = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}
