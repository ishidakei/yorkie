//! Shared harness for driver-level session tests: a synthetic (all-zero)
//! evaluation file, a temp-dir guard, an all-at-once `drive`, a `.ybb` writer,
//! and a streaming input harness for async-hold tests.
//!
//! The engine reads its parameters from an evaluation file the build laid out
//! for the kernels, beside the binary. A test that wants a network of its own
//! writes one — the header the engine checks, then an all-zero parameter region
//! — and points the driver at the directory holding it. Nothing checks the
//! parameters against the header, which is what makes a zero-weight network
//! possible at all: it evaluates every position to the same value, which is
//! what a session test pinning a transcript wants.

#![allow(dead_code)]

use std::io::{self, BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex, MutexGuard};

use yorkie_eval::network_file;
use yorkie_protocol::UsiDriver;
use yorkie_state::{Move, Position, parse_sfen, parse_usi_move, sfen_pack};

/// The message a test that is pinned to the test config's values fails with when
/// the engine was built from another config.
const WRONG_CONFIG: &str = "this test requires the test config \
     — build with `YORKIE_CONFIG=configs/test.toml`";

/// Assert that this build compiled in the three values the suite's pinned
/// assertions were captured under.
///
/// A test whose expected transcript only holds under those values calls this
/// first, so a run that forgot `YORKIE_CONFIG` fails naming the fix rather than
/// as an unexplained transcript mismatch. It never skips.
pub fn require_test_config() {
    use yorkie_protocol::config;
    assert_eq!(config::USI_HASH, 16, "{WRONG_CONFIG}");
    assert_eq!(config::THREADS, 1, "{WRONG_CONFIG}");
    assert_eq!(config::PV_INTERVAL, 0, "{WRONG_CONFIG}");
}

/// Whether a position evaluates to the same value in every game of this build.
///
/// A build carrying the `random` feature with a non-zero `random` setting offsets
/// each position's static evaluation by an amount drawn from a per-game seed, so
/// two searches taken in different games — and any comparison against a search
/// driven outside a session, which installs no seed — are not expected to agree.
/// A test that needs them to agree asks here first and skips when they cannot.
pub fn evaluation_is_noise_free() -> bool {
    #[cfg(feature = "random")]
    {
        yorkie_protocol::config::RANDOM == 0
    }
    #[cfg(not(feature = "random"))]
    {
        true
    }
}

/// Serialises the tests that reach the transposition table.
///
/// The table is one `static` per process, and `usinewgame` — which every `go`
/// session and every `bench` runs — empties the whole of it. Two tests driving
/// sessions as threads of one binary would therefore clear and overwrite each
/// other's entries, and what a search visits, and so how many nodes it reports,
/// would depend on which other test was running beside it.
///
/// Under `cargo nextest` each test is its own process and the lock is never
/// contended; under a plain `cargo test`, where one binary's tests are threads
/// of one process, it is what makes a driven search reproducible.
static TT_LOCK: Mutex<()> = Mutex::new(());

/// Exclusive use of the process's transposition table, held until the returned
/// guard goes out of scope. A test that drives a session takes it as the first
/// thing in its body, so the guard covers every search that body runs.
///
/// A panicking test leaves the lock poisoned; the next test wants the table,
/// not the panic, and the session it drives empties the table before searching
/// anything.
pub fn serial_tt() -> MutexGuard<'static, ()> {
    TT_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// The message [`evaluation_is_noise_free`]'s callers skip with.
pub const NOISY_EVALUATION: &str =
    "skipped: this build's per-game evaluation noise makes two games incomparable";

/// The architecture string a clean network file carries — the one the source
/// reader renders into its hash-mismatch warning, and the one this harness's
/// synthetic header claims so no such warning is produced.
const ARCH_STRING: &str = "ModelType=SFNNWithoutPsqt;Features=HalfKA_hm(Friend)[73305->1536x2],Network=AffineTransform[1<-32](ClippedReLU[32](AffineTransform[32<-15](ClippedReLU[15](AffineTransform[15<-3072](InputSlice[3072(0:3072)]))))){LayerStack=9}";

/// The source network file's version word and hash for a clean file.
const NNUE_VERSION: u32 = 0x7AF3_2F16;
const NNUE_HASH_VALUE: u32 = 0x3C20_3B32;

/// A temp directory removed on drop.
pub struct TempDir {
    path: PathBuf,
}

impl TempDir {
    pub fn new(tag: &str) -> Self {
        static CTR: AtomicU32 = AtomicU32::new(0);
        let n = CTR.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "engine-book-session-{}-{}-{n}",
            std::process::id(),
            tag
        ));
        std::fs::create_dir_all(&path).expect("create temp dir");
        Self { path }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// The header of an evaluation file this build accepts: the layout, the source
/// and the target features it was made for, all as compiled in.
fn synthetic_header() -> network_file::Header {
    network_file::Header {
        layout_version: network_file::LAYOUT_VERSION,
        source: network_file::SOURCE,
        target_features: network_file::TARGET_FEATURES.to_string(),
        dims: network_file::NetDims::STANDARD,
        data_bytes: network_file::DATA_BYTES as u64,
        net: yorkie_eval::NetHeader {
            version: NNUE_VERSION,
            hash: NNUE_HASH_VALUE,
            arch_id: ARCH_STRING.to_string(),
        },
        warnings: Vec::new(),
    }
}

/// Write an evaluation file with all-zero parameters at `path`.
///
/// The parameter region is a hole rather than a few hundred mebibytes of
/// written zeros: the file system reads a hole back as zeros, which is exactly
/// the network wanted.
fn write_synthetic_evaluation_file(path: &Path) {
    use std::io::Write as _;
    let encoded = synthetic_header().encode();
    let mut file = std::fs::File::create(path).expect("create the evaluation file");
    file.write_all(&encoded).expect("write the header");
    file.set_len((network_file::DATA_OFFSET + network_file::DATA_BYTES) as u64)
        .expect("size the parameter region");
}

/// Whether the file at `path` is the synthetic network *this* build accepts:
/// the full parameter region, and a header naming the same layout, source,
/// target features and dimensions [`synthetic_header`] claims.
///
/// A header that cannot be read at all is not a match either, which covers both
/// a file that is not there and one the reader refuses outright.
fn staged_file_matches_this_build(path: &Path) -> bool {
    let bytes = (network_file::DATA_OFFSET + network_file::DATA_BYTES) as u64;
    if !std::fs::metadata(path).is_ok_and(|m| m.len() == bytes) {
        return false;
    }
    let want = synthetic_header();
    network_file::read_header(path).is_ok_and(|have| {
        have.layout_version == want.layout_version
            && have.source == want.source
            && have.target_features == want.target_features
            && have.dims == want.dims
    })
}

/// Write a synthetic evaluation file into `dir` and return its path.
pub fn write_synthetic_evaluation_file_in(dir: &Path) -> PathBuf {
    let path = dir.join(network_file::FILE_NAME);
    write_synthetic_evaluation_file(&path);
    path
}

/// Drive a full canned session in-process and return the transcript. `run` joins
/// any worker, so the buffer is complete on return.
///
/// The book / `rtime` PRNG seed is process entropy, so this is for sessions
/// whose output does not depend on book-move randomisation; book sessions use
/// [`drive_with_seed`].
pub fn drive(input: &str) -> String {
    let output = Arc::new(Mutex::new(Vec::<u8>::new()));
    driver(input.as_bytes(), Arc::clone(&output), None)
        .run()
        .expect("driver run");
    let bytes = output.lock().expect("output lock").clone();
    String::from_utf8(bytes).expect("utf-8")
}

/// A fixed seed for reproducible book / `rtime` sessions: injecting it makes
/// book-move selection deterministic.
pub const TEST_BOOK_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// Like [`drive`] but with an explicit book-PRNG session seed, so book-move
/// selection (and `rtime`) is reproducible.
pub fn drive_with_seed(input: &str, book_seed: u64) -> String {
    let output = Arc::new(Mutex::new(Vec::<u8>::new()));
    driver(input.as_bytes(), Arc::clone(&output), Some(book_seed))
        .run()
        .expect("driver run");
    let bytes = output.lock().expect("output lock").clone();
    String::from_utf8(bytes).expect("utf-8")
}

/// The directory every driver this harness builds resolves `eval_dir` against,
/// once a test has staged a network. Unset until then, which is the state the
/// tests that need the engine to find *no* network rely on.
static EVAL_ROOT: OnceLock<PathBuf> = OnceLock::new();

/// Put a synthetic network where a driver this harness builds will look for it,
/// and return its path.
///
/// `eval_dir` is a compile-time constant and no build has an option surface, so
/// what a test chooses is the directory that constant resolves against — the
/// running executable's own directory in the engine, and whatever a test names
/// here in a session it drives itself.
///
/// Safe to call from every test in a binary and from several at once: every
/// caller names the same directory and writes the same file, the write is an
/// atomic rename from a unique temporary, and a staged file is reused.
///
/// The fixture root lives under `target/`, so a file written by one run outlives
/// it and is found by the next — including a run of a build that describes a
/// different network, whose engine would refuse the file it finds. A staged file
/// is therefore reused only when its header is the one this build expects; the
/// size is the cheap part of that answer, not the whole of it.
pub fn stage_configured_eval_dir() -> PathBuf {
    let root = fixture_root("synthetic");
    let eval_dir = root.join(yorkie_protocol::config::EVAL_DIR);
    std::fs::create_dir_all(&eval_dir).expect("create fixture eval dir");
    let file = eval_dir.join(network_file::FILE_NAME);
    if !staged_file_matches_this_build(&file) {
        // Unique temporary + rename: a concurrent staging attempt from another
        // test in this process (or another test binary) either sees no file or
        // sees the complete one, never a half-written one.
        static CTR: AtomicU32 = AtomicU32::new(0);
        let tmp = eval_dir.join(format!(
            "{}.{}.{}",
            network_file::FILE_NAME,
            std::process::id(),
            CTR.fetch_add(1, Ordering::Relaxed)
        ));
        write_synthetic_evaluation_file(&tmp);
        std::fs::rename(&tmp, &file).expect("publish the synthetic evaluation file");
    }
    let _ = EVAL_ROOT.set(root);
    file
}

/// Point the drivers this harness builds at the evaluation file the build
/// itself wrote — the network the engine plays with — and report whether there
/// is one.
///
/// For the tests that need the real network, which is staged outside the build
/// tree and never committed.
pub fn stage_engine_eval_root() -> bool {
    let root = engine_eval_root();
    let present = network_file::network_path(&root).is_file();
    let _ = EVAL_ROOT.set(root);
    present
}

/// The directory holding the binaries this build produced, which is where the
/// build wrote the evaluation file: the test executable's parent's parent
/// (`<target>/<profile>/deps/<name>-<hash>`).
pub fn engine_eval_root() -> PathBuf {
    let exe = std::env::current_exe().expect("test executable path");
    exe.parent()
        .and_then(Path::parent)
        .expect("a test executable lives under <target>/<profile>/deps")
        .to_path_buf()
}

/// Build a driver over `reader`, pointed at whatever network this test staged.
fn driver<R: BufRead>(
    reader: R,
    output: Arc<Mutex<Vec<u8>>>,
    book_seed: Option<u64>,
) -> UsiDriver<R, Vec<u8>> {
    let driver = match book_seed {
        Some(seed) => UsiDriver::with_book_seed(reader, output, seed),
        None => UsiDriver::new(reader, output),
    };
    match EVAL_ROOT.get() {
        Some(root) => driver.with_eval_root(root.clone()),
        None => driver,
    }
}

/// A fixture root under the workspace `target/` directory, one per `tag`.
/// Derived from the test executable's own path
/// (`<target>/<profile>/deps/<name>-<hash>`), so it follows `CARGO_TARGET_DIR`
/// wherever it points.
fn fixture_root(tag: &str) -> PathBuf {
    let exe = std::env::current_exe().expect("test executable path");
    let target = exe
        .parent() // deps
        .and_then(Path::parent) // <profile>
        .and_then(Path::parent) // target
        .expect("test executable lives under <target>/<profile>/deps");
    target.join(format!("usi-session-fixtures-{tag}"))
}

/// The transcript a diagnostic `info string <body>` contributes in *this* build:
/// the line itself with `verbose1`, the empty string without it.
///
/// A session test that pins bytes composes its expectation through this helper
/// so it stays byte-exact in both builds.
pub fn diag_line(body: &str) -> String {
    if cfg!(feature = "verbose1") {
        format!("info string {body}\n")
    } else {
        String::new()
    }
}

/// A transcript with the per-reply statistics line taken out.
///
/// That line counts what the *process* allocated since the previous reply, so
/// two runs of one session do not agree on it. A transcript pinned byte-for-byte
/// is about what the engine decided, and the number says nothing about that.
/// Below `verbose1` there is no such line and this is the identity.
pub fn without_stats(out: &str) -> String {
    out.lines()
        .filter(|l| !l.starts_with("info string stats "))
        .map(|l| format!("{l}\n"))
        .collect()
}

pub fn bestmove_lines(out: &str) -> Vec<&str> {
    out.lines()
        .filter_map(|l| l.strip_prefix("bestmove "))
        .collect()
}

// --- `.ybb` writer (mirrors xtask capture-book's serializer). ---

/// One book move for [`write_ybb`]: `(usi, value, depth)`.
pub type YbbMove<'a> = (&'a str, i16, u16);

/// Build and write a depth-carrying `.ybb` at `path` from `(sfen, moves)`
/// records. Positions are packed with the workspace encoder; records are sorted
/// by packed key (as the format requires).
pub fn write_ybb(path: &Path, records: &[(&str, Vec<YbbMove<'_>>)]) {
    const MAGIC: &[u8; 16] = b"YANE-BINBOOK-V1\0";

    struct Rec {
        packed: [u8; 32],
        ply: u16,
        moves: Vec<(u16, i16, u16)>,
    }

    let mut recs: Vec<Rec> = records
        .iter()
        .map(|(sfen, moves)| {
            let pos = parse_sfen(sfen).expect("valid sfen");
            let moves = moves
                .iter()
                .map(|(usi, v, d)| {
                    let m16 = parse_usi_move(usi, &pos).expect("valid move").move16();
                    (m16, *v, *d)
                })
                .collect();
            Rec {
                packed: sfen_pack(&pos),
                ply: pos.ply(),
                moves,
            }
        })
        .collect();
    recs.sort_by_key(|r| r.packed);

    let mut header = Vec::new();
    header.extend_from_slice(MAGIC);
    header.extend_from_slice(&(recs.len() as u64).to_le_bytes());
    header.extend_from_slice(&1u64.to_le_bytes()); // flags: move-depth present

    let mut index = Vec::new();
    let mut moves = Vec::new();
    for r in &recs {
        let moves_offset = moves.len() as u64;
        index.extend_from_slice(&r.packed);
        index.extend_from_slice(&moves_offset.to_le_bytes());
        index.extend_from_slice(&r.ply.to_le_bytes());
        index.extend_from_slice(&(r.moves.len() as u16).to_le_bytes());
        for (m, v, d) in &r.moves {
            moves.extend_from_slice(&m.to_le_bytes());
            moves.extend_from_slice(&(*v as u16).to_le_bytes());
            moves.extend_from_slice(&d.to_le_bytes());
        }
    }
    let mut out = header;
    out.extend_from_slice(&index);
    out.extend_from_slice(&moves);
    std::fs::write(path, &out).expect("write ybb");
}

/// Copy the committed `tests/fixtures/book/sample.ybb` into `dir` under
/// `dest_name` and return its path.
pub fn stage_sample_ybb(dir: &Path, dest_name: &str) -> PathBuf {
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("tests/fixtures/book/sample.ybb");
    let dest = dir.join(dest_name);
    std::fs::copy(&src, &dest).expect("copy sample.ybb");
    dest
}

pub fn parse(sfen: &str) -> Position {
    parse_sfen(sfen).expect("valid sfen")
}

pub fn legal(pos: &Position) -> Vec<Move> {
    let mut v = Vec::new();
    pos.generate_legal_all(&mut v);
    v
}

// --- Streaming input harness (for the ponder/infinite hold tests). ---

/// A blocking [`Read`] fed line-chunks over an mpsc channel: it blocks on an
/// empty buffer until the next chunk (or EOF) arrives, so a test can feed the
/// driver commands over time and observe output between them.
pub struct BlockingReader {
    rx: Receiver<Option<Vec<u8>>>,
    buf: Vec<u8>,
    pos: usize,
    done: bool,
}

impl Read for BlockingReader {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        loop {
            if self.pos < self.buf.len() {
                let n = std::cmp::min(out.len(), self.buf.len() - self.pos);
                out[..n].copy_from_slice(&self.buf[self.pos..self.pos + n]);
                self.pos += n;
                return Ok(n);
            }
            if self.done {
                return Ok(0);
            }
            match self.rx.recv() {
                Ok(Some(chunk)) => {
                    self.buf = chunk;
                    self.pos = 0;
                }
                Ok(None) | Err(_) => {
                    self.done = true;
                    return Ok(0);
                }
            }
        }
    }
}

/// A running driver on its own thread, fed incrementally.
pub struct StreamHarness {
    tx: Sender<Option<Vec<u8>>>,
    output: Arc<Mutex<Vec<u8>>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl StreamHarness {
    /// A streaming harness seeded from process entropy (the production default).
    pub fn start() -> Self {
        Self::start_with_seed(None)
    }

    /// A streaming harness with an explicit book-PRNG session seed, for
    /// reproducible book / `rtime` sessions. `None` uses process entropy.
    pub fn start_with_seed(book_seed: Option<u64>) -> Self {
        let (tx, rx) = channel();
        let output = Arc::new(Mutex::new(Vec::<u8>::new()));
        let out2 = Arc::clone(&output);
        let handle = std::thread::spawn(move || {
            let reader = BufReader::new(BlockingReader {
                rx,
                buf: Vec::new(),
                pos: 0,
                done: false,
            });
            driver(reader, out2, book_seed).run().expect("driver run");
        });
        StreamHarness {
            tx,
            output,
            handle: Some(handle),
        }
    }

    /// Feed one command line (a `\n` is appended).
    pub fn send(&self, line: &str) {
        let mut bytes = line.as_bytes().to_vec();
        bytes.push(b'\n');
        self.tx.send(Some(bytes)).expect("send");
    }

    /// Current transcript.
    pub fn output(&self) -> String {
        String::from_utf8(self.output.lock().expect("lock").clone()).expect("utf-8")
    }

    /// Poll until `pred(output)` holds or `timeout` elapses; returns whether it
    /// became true. Uses a coarse 5ms poll (no wall-clock assertions).
    pub fn wait_until(&self, timeout_ms: u64, pred: impl Fn(&str) -> bool) -> bool {
        let mut waited = 0u64;
        loop {
            if pred(&self.output()) {
                return true;
            }
            if waited >= timeout_ms {
                return false;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
            waited += 5;
        }
    }

    /// Send `quit`, close the input, and join the driver thread.
    pub fn quit_join(mut self) -> String {
        self.send("quit");
        let _ = self.tx.send(None);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        self.output()
    }
}
