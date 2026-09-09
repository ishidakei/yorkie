// The kernel-layout evaluation file: the network's dimensions, the byte layout
// the SIMD kernels read the parameters out of, and the file's header.
//
// Shared verbatim between this crate and its build script — the build script
// writes that file and the crate reads it, so one definition is what keeps the
// two from disagreeing about a byte. It is `#[path]`-included rather than
// shared as a crate because a build script cannot depend on a member of the
// workspace it is building.
//
// Nothing here reads a file, allocates a network, or touches the environment.

/// The prose below follows the reference's naming: the **FT layer** is the
/// feature transformer, and **L1 / L2 / L3** are the dense layers after it. The
/// FT layer is never called "L1". In the identifiers, L1 is `fc_0`, L2 is
/// `fc_1` and L3 is `fc_2`.
pub const HIDDEN_SIZE: usize = 1_536;
pub const NUM_FEATURES: usize = 73_305;
pub const LAYER_STACKS: usize = 9;

// `fc_0`'s 16th output feeds only the post-`fc_2` shortcut; the first 15 feed
// both activations.
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

/// The identity fields of the source network file, carried through into the
/// kernel-layout file so the engine can still report them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NetHeader {
    pub version: u32,
    pub hash: u32,
    pub arch_id: String,
}

/// The dimensions of one SFNN network, driving both the byte layout and the
/// transformation's read sizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NetDims {
    pub hidden_size: usize,
    pub num_features: usize,
    pub layer_stacks: usize,
    pub fc_0_output: usize,
    pub fc_0_padded_input: usize,
    pub fc_1_output: usize,
    pub fc_1_padded_input: usize,
    pub fc_2_output: usize,
    pub fc_2_padded_input: usize,
}

impl NetDims {
    /// The shipped SFNN-1536 dimensions.
    pub const STANDARD: NetDims = NetDims {
        hidden_size: HIDDEN_SIZE,
        num_features: NUM_FEATURES,
        layer_stacks: LAYER_STACKS,
        fc_0_output: FC_0_OUTPUT_DIMS,
        fc_0_padded_input: FC_0_PADDED_INPUT_DIMS,
        fc_1_output: FC_1_OUTPUT_DIMS,
        fc_1_padded_input: FC_1_PADDED_INPUT_DIMS,
        fc_2_output: FC_2_OUTPUT_DIMS,
        fc_2_padded_input: FC_2_PADDED_INPUT_DIMS,
    };
}

/// Alignment every parameter array starts on inside the data region: the
/// 64-byte cache line the AVX-512 kernels' 512-bit loads assume, so no load
/// splits one.
pub const SECTION_ALIGN: usize = 64;

/// Where the data region starts in the file, and the boundary the engine's
/// in-memory copy of it sits on: 2 MiB, so a huge-page hint over the region can
/// be honoured and the file offset a mapping starts at is page-aligned.
pub const DATA_OFFSET: usize = 2 * 1024 * 1024;

/// The kernel-layout file's name inside the evaluation directory.
pub const FILE_NAME: &str = "nn.kernel.bin";

/// The first bytes of the file, so a file that is not one is refused before
/// anything in it is believed.
pub const MAGIC: [u8; 16] = *b"YORKIE-NNUE-KL\0\0";

/// The version of the layout below and of the transformation that produces it.
/// Bumped whenever either changes, or whenever what the kernels expect of the
/// bytes changes, so a file an older build wrote is refused rather than read as
/// if it said something else.
pub const LAYOUT_VERSION: u32 = 1;

/// One parameter array's place in the data region: its byte offset from the
/// start of that region, and its element count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub offset: usize,
    pub count: usize,
}

/// The six spans of one layer stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StackSpans {
    pub fc_0_biases: Span,
    pub fc_0_weights: Span,
    pub fc_1_biases: Span,
    pub fc_1_weights: Span,
    pub fc_2_biases: Span,
    pub fc_2_weights: Span,
}

/// Every parameter array's place in the data region, and the byte size of that
/// region.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NetSpans {
    pub ft_biases: Span,
    pub ft_weights: Span,
    pub stacks: Vec<StackSpans>,
    pub total_bytes: usize,
}

/// Advance `cursor` past `count` elements of `size` bytes, aligned to
/// [`SECTION_ALIGN`], and report where they start.
const fn place(cursor: usize, count: usize, size: usize) -> (usize, usize) {
    let offset = cursor.next_multiple_of(SECTION_ALIGN);
    (offset, offset + count * size)
}

/// The spans of every parameter array, in the order the data region carries
/// them: the feature transformer's biases and weights, then each layer stack's
/// six arrays.
pub fn net_spans(dims: &NetDims) -> NetSpans {
    let mut cursor = 0usize;
    let mut span = |count: usize, size: usize| {
        let (offset, next) = place(cursor, count, size);
        cursor = next;
        Span { offset, count }
    };
    let ft_biases = span(dims.hidden_size, 2);
    let ft_weights = span(dims.hidden_size * dims.num_features, 2);
    let mut stacks = Vec::with_capacity(dims.layer_stacks);
    for _ in 0..dims.layer_stacks {
        stacks.push(StackSpans {
            fc_0_biases: span(dims.fc_0_output, 4),
            fc_0_weights: span(dims.fc_0_output * dims.fc_0_padded_input, 1),
            fc_1_biases: span(dims.fc_1_output, 4),
            fc_1_weights: span(dims.fc_1_output * dims.fc_1_padded_input, 1),
            fc_2_biases: span(dims.fc_2_output, 4),
            fc_2_weights: span(dims.fc_2_output * dims.fc_2_padded_input, 1),
        });
    }
    NetSpans {
        ft_biases,
        ft_weights,
        stacks,
        total_bytes: cursor,
    }
}

/// The byte size of the data region, as a constant: the same walk
/// [`net_spans`] performs, in a form a `static`'s size can be written from. A
/// unit test holds the two against each other.
pub const fn data_bytes(dims: &NetDims) -> usize {
    let mut cursor = 0usize;
    cursor = place(cursor, dims.hidden_size, 2).1;
    cursor = place(cursor, dims.hidden_size * dims.num_features, 2).1;
    let mut stack = 0usize;
    while stack < dims.layer_stacks {
        cursor = place(cursor, dims.fc_0_output, 4).1;
        cursor = place(cursor, dims.fc_0_output * dims.fc_0_padded_input, 1).1;
        cursor = place(cursor, dims.fc_1_output, 4).1;
        cursor = place(cursor, dims.fc_1_output * dims.fc_1_padded_input, 1).1;
        cursor = place(cursor, dims.fc_2_output, 4).1;
        cursor = place(cursor, dims.fc_2_output * dims.fc_2_padded_input, 1).1;
        stack += 1;
    }
    cursor
}

/// The data region of the shipped network, in bytes.
pub const DATA_BYTES: usize = data_bytes(&NetDims::STANDARD);

/// What the source network file was, for the build that wrote a kernel-layout
/// file: its SHA-256, or the fact that no source was there to convert.
///
/// A checkout without a staged network still builds — the engine reports the
/// missing network when it is asked to be ready — so "no source" is a value the
/// header carries rather than a build failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Absent,
    Sha256([u8; 32]),
}

/// The kernel-layout file's header: what the data region after it is, and what
/// it was made from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Header {
    pub layout_version: u32,
    pub source: Source,
    /// The `target_feature` set the converting build compiled for. The kernels
    /// are selected from it, so a file made for another one describes bytes
    /// these kernels would read differently.
    pub target_features: String,
    pub dims: NetDims,
    pub data_bytes: u64,
    /// The source file's own version word and hash, and its architecture
    /// string.
    pub net: NetHeader,
    /// The non-fatal complaints the conversion had about the source file, in
    /// the order it made them, so the engine can report them when it is asked
    /// to be ready.
    pub warnings: Vec<String>,
}

impl Header {
    /// The header's bytes: magic, then the fields in declaration order,
    /// little-endian, with every string and list length-prefixed.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&self.layout_version.to_le_bytes());
        match self.source {
            Source::Absent => {
                out.push(0);
                out.extend_from_slice(&[0u8; 32]);
            }
            Source::Sha256(digest) => {
                out.push(1);
                out.extend_from_slice(&digest);
            }
        }
        put_str(&mut out, &self.target_features);
        for value in dims_words(&self.dims) {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out.extend_from_slice(&self.data_bytes.to_le_bytes());
        out.extend_from_slice(&self.net.version.to_le_bytes());
        out.extend_from_slice(&self.net.hash.to_le_bytes());
        put_str(&mut out, &self.net.arch_id);
        out.extend_from_slice(&(self.warnings.len() as u32).to_le_bytes());
        for warning in &self.warnings {
            put_str(&mut out, warning);
        }
        out
    }

    /// Read a header back out of `bytes`, which may be longer than the header
    /// itself.
    pub fn decode(bytes: &[u8]) -> Result<Header, String> {
        let mut r = Cursor { bytes, pos: 0 };
        if r.take(MAGIC.len())? != MAGIC {
            return Err("not a kernel-layout evaluation file (wrong magic)".to_string());
        }
        let layout_version = r.u32()?;
        let source = match r.u8()? {
            0 => {
                r.take(32)?;
                Source::Absent
            }
            1 => {
                let mut digest = [0u8; 32];
                digest.copy_from_slice(r.take(32)?);
                Source::Sha256(digest)
            }
            other => return Err(format!("unknown source marker {other}")),
        };
        let target_features = r.string()?;
        let mut words = [0u32; DIMS_WORDS];
        for word in words.iter_mut() {
            *word = r.u32()?;
        }
        let dims = dims_from_words(&words);
        let data_bytes = r.u64()?;
        let net = NetHeader {
            version: r.u32()?,
            hash: r.u32()?,
            arch_id: r.string()?,
        };
        let count = r.u32()? as usize;
        let mut warnings = Vec::with_capacity(count.min(64));
        for _ in 0..count {
            warnings.push(r.string()?);
        }
        Ok(Header {
            layout_version,
            source,
            target_features,
            dims,
            data_bytes,
            net,
            warnings,
        })
    }

    /// How this header differs from what a build expects to read, or `None`
    /// when it is the file that build wrote.
    ///
    /// The comparison is of the description, not of the data: the region is
    /// hundreds of mebibytes and re-hashing it at startup would cost what
    /// laying it out ahead of time saved.
    pub fn refusal(&self, expected_source: Source, expected_features: &str) -> Option<String> {
        if self.layout_version != LAYOUT_VERSION {
            return Some(format!(
                "layout version {} in the file, {LAYOUT_VERSION} in this build",
                self.layout_version
            ));
        }
        if self.source != expected_source {
            return Some(format!(
                "made from {}, this build was made from {}",
                describe_source(self.source),
                describe_source(expected_source)
            ));
        }
        if self.target_features != expected_features {
            return Some(format!(
                "made for target features `{}`, this build compiled for `{expected_features}`",
                self.target_features
            ));
        }
        if self.dims != NetDims::STANDARD {
            return Some("network dimensions are not the ones this build reads".to_string());
        }
        if self.data_bytes != DATA_BYTES as u64 {
            return Some(format!(
                "data region is {} bytes, this build reads {DATA_BYTES}",
                self.data_bytes
            ));
        }
        None
    }
}

/// Resolve a directory setting against `base`: an absolute path stands as it
/// is, a relative one is taken against `base`.
///
/// The two directory settings differ only in what `base` is — the repository
/// root for the network a build converts, the running executable's own
/// directory for the file the engine reads — so the rule itself is written
/// once.
pub fn resolve_dir(base: &std::path::Path, dir: &str) -> std::path::PathBuf {
    let raw = std::path::PathBuf::from(dir);
    if raw.is_absolute() {
        raw
    } else {
        base.join(raw)
    }
}

/// A source identity as it reads in a message.
pub fn describe_source(source: Source) -> String {
    match source {
        Source::Absent => "no network file".to_string(),
        Source::Sha256(digest) => {
            let mut s = String::from("network ");
            // The first eight bytes are enough to name which network it is.
            for byte in &digest[..8] {
                s.push(HEX[(byte >> 4) as usize] as char);
                s.push(HEX[(byte & 0xF) as usize] as char);
            }
            s
        }
    }
}

const HEX: &[u8; 16] = b"0123456789abcdef";

const DIMS_WORDS: usize = 9;

fn dims_words(dims: &NetDims) -> [u32; DIMS_WORDS] {
    [
        dims.hidden_size as u32,
        dims.num_features as u32,
        dims.layer_stacks as u32,
        dims.fc_0_output as u32,
        dims.fc_0_padded_input as u32,
        dims.fc_1_output as u32,
        dims.fc_1_padded_input as u32,
        dims.fc_2_output as u32,
        dims.fc_2_padded_input as u32,
    ]
}

fn dims_from_words(words: &[u32; DIMS_WORDS]) -> NetDims {
    NetDims {
        hidden_size: words[0] as usize,
        num_features: words[1] as usize,
        layer_stacks: words[2] as usize,
        fc_0_output: words[3] as usize,
        fc_0_padded_input: words[4] as usize,
        fc_1_output: words[5] as usize,
        fc_1_padded_input: words[6] as usize,
        fc_2_output: words[7] as usize,
        fc_2_padded_input: words[8] as usize,
    }
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

/// A forward-only reader over the header bytes. Every read is bounds-checked,
/// so a truncated or malformed header is an error and never a panic.
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| "header offset overflow".to_string())?;
        if end > self.bytes.len() {
            return Err("header is truncated".to_string());
        }
        let s = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, String> {
        let s = self.take(4)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }

    fn u64(&mut self) -> Result<u64, String> {
        let s = self.take(8)?;
        Ok(u64::from_le_bytes([
            s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
        ]))
    }

    fn string(&mut self) -> Result<String, String> {
        let len = self.u32()? as usize;
        let bytes = self.take(len)?;
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }
}
