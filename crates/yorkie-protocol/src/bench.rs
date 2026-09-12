//! The `bench` command's argument parsing and default position set — a port of
//! the reference `setup_bench` and its `Defaults` list.
//!
//! The whole module exists only under the `verbose3` cargo feature, so the
//! default build has neither this parse nor the `bench` command token.
//!
//! This module owns only the semantic parse of the argument tokens into a
//! [`BenchConfig`]; the command handler consumes it.
//!
//! ```text
//! bench [ttSizeMB] [threads] [limit] [default|current|<fenFile>] [limitType]
//! ```
//!
//! The reference's defaults are `ttSize=1024`, `threads=1`, `limit=15000`,
//! `fenFile=default`, `limitType=movetime` — its *code*, not the stale comment
//! example beside it, is the ground truth. This port mirrors them exactly.
//!
//! `ttSizeMB` is checked and then dropped: the transposition table is a `static`
//! whose size the build fixed, so no command can change it, and a bench
//! measures whatever size the binary was built with. The argument keeps its
//! place because the grammar is positional — `bench 16 1 6 default depth` names
//! its thread count by being fourth from the end.
//!
//! The reference's `limitType` also accepts `perft` and `eval`. `perft` needs a
//! `go perft` path this crate does not own and `eval` needs `trace_eval`, so
//! both parse to a loud [`BenchParseError`] rather than panicking.

use std::fs;
use std::path::Path;

use yorkie_state::TextWriter;
use yorkie_state::text::{atoi_i64, atoi_u64};

use crate::engine::GoParams;

/// The reference `Defaults` position list, transcribed verbatim (every SFEN,
/// same order). Used when the position source is `default` (or omitted).
pub const BENCH_DEFAULT_POSITIONS: [&[u8]; 4] = [
    // 初期局面に近い曲面。
    b"lnsgkgsnl/1r7/p1ppp1bpp/1p3pp2/7P1/2P6/PP1PPPP1P/1B3S1R1/LNSGKG1NL b - 9",
    // 読めば読むほど後手悪いような局面
    b"l4S2l/4g1gs1/5p1p1/pr2N1pkp/4Gn3/PP3PPPP/2GPP4/1K7/L3r+s2L w BS2N5Pb 1",
    // 57同銀は詰み、みたいな。読めば読むほど先手が悪いことがわかってくる局面。
    b"6n1l/2+S1k4/2lp4p/1np1B2b1/3PP4/1N1S3rP/1P2+pPP+p1/1p1G5/3KG2r1 b GSN2L4Pgs2p 1",
    // 指し手生成祭りの局面 cf. http://d.hatena.ne.jp/ak11/20110508/p1
    b"l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w RGgsn5p 1",
];

/// The reference non-Stockfish defaults.
const DEFAULT_TT_MB: &[u8] = b"1024";
const DEFAULT_THREADS: &[u8] = b"1";
const DEFAULT_LIMIT: &[u8] = b"15000";
const DEFAULT_FEN_SOURCE: &[u8] = b"default";
const DEFAULT_LIMIT_TYPE: &[u8] = b"movetime";

/// A `bench` argument-parse failure, surfaced as an `info string` so a garbage
/// argument fails loudly without panicking.
///
/// Each variant carries what it refused — borrowed from the command line, so
/// nothing is copied out of it — and [`Self::write_message`] spells it where
/// the notice is written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchParseError<'a> {
    InvalidTtSize(&'a [u8]),
    InvalidThreads(&'a [u8]),
    InvalidLimit(&'a [u8]),
    DepthOutOfRange(u64),
    UnsupportedLimitType(&'a [u8]),
    /// A `<fenFile>` that could not be read. The operating system's own
    /// description is not this project's text, so its error number is named.
    UnreadableFile {
        path: &'a [u8],
        errno: Option<i32>,
    },
    NoPositions(&'a [u8]),
}

impl BenchParseError<'_> {
    /// Append this failure's message to `out`.
    pub fn write_message(&self, out: &mut TextWriter<'_>) {
        match self {
            Self::InvalidTtSize(token) => {
                out.bytes(b"invalid ttSizeMB `").bytes(token).bytes(b"`");
            }
            Self::InvalidThreads(token) => {
                out.bytes(b"invalid threads `").bytes(token).bytes(b"`");
            }
            Self::InvalidLimit(token) => {
                out.bytes(b"invalid limit `").bytes(token).bytes(b"`");
            }
            Self::DepthOutOfRange(limit) => {
                out.bytes(b"depth limit out of range `")
                    .u64(*limit)
                    .bytes(b"`");
            }
            Self::UnsupportedLimitType(token) => {
                out.bytes(b"unsupported limit type `")
                    .bytes(token)
                    .bytes(b"` (supported: depth, nodes, movetime)");
            }
            Self::UnreadableFile { path, errno } => {
                out.bytes(b"unable to open file `")
                    .bytes(path)
                    .bytes(b"`: errno ");
                match errno {
                    Some(errno) => out.i64(i64::from(*errno)),
                    None => out.bytes(b"unknown"),
                };
            }
            Self::NoPositions(path) => {
                out.bytes(b"no positions in file `").bytes(path).bytes(b"`");
            }
        }
    }
}

/// A fully-resolved `bench` invocation: the option values to apply, the search
/// limit for every position, and the position list to run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchConfig {
    /// The `Threads` value to `setoption`.
    pub threads: i64,
    /// The per-position search limit, applied to every position exactly as a
    /// normal `go` would consume it.
    pub limits: GoParams,
    /// The positions to search, as SFENs (each parsed by the caller).
    pub fens: Vec<Vec<u8>>,
}

/// Parse the `bench` argument tokens into a [`BenchConfig`], filling missing
/// trailing arguments with the reference defaults. `current_sfen` is the current
/// session position's SFEN, used only when the position source is `current`.
///
/// Errors (never panics):
/// - a non-integer `ttSizeMB`, `threads`, or `limit`;
/// - an unsupported `limitType` (see the scope-divergence note above);
/// - a `<fenFile>` that cannot be opened.
pub fn parse_bench<'a>(
    tokens: &[&'a [u8]],
    current_sfen: &[u8],
) -> Result<BenchConfig, BenchParseError<'a>> {
    let arg = |i: usize, default: &'static [u8]| -> &'a [u8] {
        tokens.get(i).copied().unwrap_or(default)
    };
    let tt_arg = arg(0, DEFAULT_TT_MB);
    let threads_arg = arg(1, DEFAULT_THREADS);
    let limit_arg = arg(2, DEFAULT_LIMIT);
    let fen_source = arg(3, DEFAULT_FEN_SOURCE);
    let limit_type = arg(4, DEFAULT_LIMIT_TYPE);

    // Parsed for its shape and then dropped: nothing can resize the table, but
    // a garbage argument still has to fail loudly rather than shift the
    // positional arguments behind it.
    let _tt_mb: i64 = atoi_i64(tt_arg).ok_or(BenchParseError::InvalidTtSize(tt_arg))?;
    let threads: i64 = atoi_i64(threads_arg).ok_or(BenchParseError::InvalidThreads(threads_arg))?;
    let limit: u64 = atoi_u64(limit_arg).ok_or(BenchParseError::InvalidLimit(limit_arg))?;

    let mut limits = GoParams::default();
    match limit_type {
        b"depth" => {
            let d = u32::try_from(limit).map_err(|_| BenchParseError::DepthOutOfRange(limit))?;
            limits.depth = Some(d);
        }
        b"nodes" => limits.nodes = Some(limit),
        b"movetime" => limits.movetime = Some(limit),
        other => return Err(BenchParseError::UnsupportedLimitType(other)),
    }

    let fens = match fen_source {
        b"default" => BENCH_DEFAULT_POSITIONS
            .iter()
            .map(|sfen| sfen.to_vec())
            .collect(),
        b"current" => vec![current_sfen.to_vec()],
        path => {
            let text = fs::read(path_of(path)).map_err(|e| BenchParseError::UnreadableFile {
                path,
                errno: e.raw_os_error(),
            })?;
            let fens: Vec<Vec<u8>> = text
                .split(|&b| b == b'\n')
                .map(yorkie_state::text::trim_ascii_whitespace)
                .filter(|line| !line.is_empty())
                .map(<[u8]>::to_vec)
                .collect();
            if fens.is_empty() {
                return Err(BenchParseError::NoPositions(path));
            }
            fens
        }
    };

    Ok(BenchConfig {
        threads,
        limits,
        fens,
    })
}

/// A `<fenFile>` argument as a path: its bytes are the path's own.
#[cfg(unix)]
fn path_of(bytes: &[u8]) -> &Path {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt as _;

    Path::new(OsStr::from_bytes(bytes))
}

/// Write `pos`'s SFEN into `buf` — the position source `current` names.
///
/// A thin re-export of [`yorkie_state::format_sfen`] so the caller expresses
/// intent at the call site (`bench::current_sfen(engine.position(), &mut buf)`).
pub fn current_sfen<'b>(
    pos: &yorkie_state::Position,
    buf: &'b mut yorkie_state::SfenBuf,
) -> &'b [u8] {
    yorkie_state::format_sfen(pos, buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A failure's message, so the assertions below can read it as text.
    fn rendered(err: &BenchParseError<'_>) -> Vec<u8> {
        let mut bytes = [0u8; 512];
        let mut out = TextWriter::new(&mut bytes);
        err.write_message(&mut out);
        out.as_bytes().to_vec()
    }

    fn message(err: &BenchParseError<'_>) -> String {
        String::from_utf8(rendered(err)).expect("a message is ASCII")
    }

    #[test]
    fn defaults_when_no_args() {
        let cfg = parse_bench(&[], b"startsfen").expect("defaults parse");
        assert_eq!(cfg.threads, 1);
        // Default limit type is movetime 15000 (yaneuraou's one-minute bench).
        assert_eq!(cfg.limits.movetime, Some(15000));
        assert_eq!(cfg.limits.depth, None);
        assert_eq!(cfg.fens.len(), 4);
        assert_eq!(cfg.fens[0], BENCH_DEFAULT_POSITIONS[0]);
    }

    #[test]
    fn depth_limit_type() {
        let tokens: [&[u8]; 5] = [b"16", b"1", b"6", b"default", b"depth"];
        let cfg = parse_bench(&tokens, b"x").expect("parse");
        assert_eq!(cfg.limits.depth, Some(6));
        assert_eq!(cfg.limits.movetime, None);
    }

    #[test]
    fn nodes_limit_type() {
        let tokens: [&[u8]; 5] = [b"16", b"1", b"100000", b"default", b"nodes"];
        let cfg = parse_bench(&tokens, b"x").expect("parse");
        assert_eq!(cfg.limits.nodes, Some(100000));
    }

    #[test]
    fn current_source_uses_given_sfen() {
        let tokens: [&[u8]; 5] = [b"16", b"1", b"4", b"current", b"depth"];
        let cfg = parse_bench(&tokens, b"my-sfen").expect("parse");
        assert_eq!(cfg.fens, vec![b"my-sfen".to_vec()]);
    }

    #[test]
    fn garbage_tt_size_errors() {
        let tokens: [&[u8]; 1] = [b"notanumber"];
        let err = parse_bench(&tokens, b"x").expect_err("not an integer");
        assert_eq!(message(&err), "invalid ttSizeMB `notanumber`");
    }

    /// An argument carrying bytes no `bench` line is spelled in parses as no
    /// number, which is the malformed-argument path — and the message quoting
    /// it is written out as it arrived.
    #[test]
    fn a_non_ascii_argument_errors() {
        let tokens: [&[u8]; 1] = [b"\x82\xa0"];
        let err = parse_bench(&tokens, b"x").expect_err("not an integer");
        assert_eq!(rendered(&err), b"invalid ttSizeMB `\x82\xa0`");
    }

    #[test]
    fn unsupported_limit_type_errors() {
        let tokens: [&[u8]; 5] = [b"16", b"1", b"5", b"default", b"perft"];
        let err = parse_bench(&tokens, b"x").expect_err("perft unsupported");
        let message = message(&err);
        assert!(
            message.contains("perft"),
            "message names the type: {message}"
        );
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn missing_file_errors() {
        let tokens: [&[u8]; 5] = [b"16", b"1", b"5", b"/no/such/bench/file", b"depth"];
        let err = parse_bench(&tokens, b"x").expect_err("the file is absent");
        let message = message(&err);
        assert!(
            message.starts_with("unable to open file `/no/such/bench/file`"),
            "got: {message}"
        );
    }
}
