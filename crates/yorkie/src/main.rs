use std::ffi::OsString;
use std::io::{BufReader, Write as _, stdin, stdout};
use std::os::unix::ffi::OsStrExt as _;
use std::process::ExitCode;
use std::sync::{Arc, Mutex};

use yorkie::perft;
use yorkie_protocol::UsiEngine;
use yorkie_state::text::{TextWriter, atoi_u32};
use yorkie_state::{Position, parse_sfen, parse_usi_move};

const USAGE: &[u8] = b"\
usage:
  yorkie                                              # run USI event loop on stdin/stdout
  yorkie perft startpos <depth>
  yorkie perft sfen <SFEN-LITERAL> <depth>
  yorkie perft sfen <SFEN-LITERAL> moves <m1> [<m2> ...] <depth>
";

/// Room one error message is composed in. The widest of them quotes an
/// argument, which the operating system already bounds.
const MESSAGE_BYTES: usize = 8 * 1024;

fn main() -> ExitCode {
    // The arguments arrive as the bytes the operating system holds them as:
    // this engine's own text is ASCII, and anything else is quoted back rather
    // than decoded.
    let args: Vec<OsString> = std::env::args_os().skip(1).collect();
    let borrowed: Vec<&[u8]> = args.iter().map(|arg| arg.as_bytes()).collect();
    let mut bytes = [0u8; MESSAGE_BYTES];
    let mut message = TextWriter::new(&mut bytes);
    match dispatch(&borrowed, &mut message) {
        Ok(()) => ExitCode::SUCCESS,
        Err(()) => {
            let mut err = std::io::stderr().lock();
            let _ = err.write_all(b"error: ");
            let _ = err.write_all(message.as_bytes());
            let _ = err.write_all(b"\n");
            let _ = err.write_all(USAGE);
            ExitCode::from(2)
        }
    }
}

/// Run what `args` asks for, or write why it cannot into `message`.
fn dispatch(args: &[&[u8]], message: &mut TextWriter<'_>) -> Result<(), ()> {
    let Some((&cmd, rest)) = args.split_first() else {
        // The reader stays on the main thread; the writer is shared with the
        // search worker, so it must be `Send + 'static` — use the owned `Stdout`
        // handle (a `StdoutLock` is not `Send`) behind an `Arc<Mutex<_>>`.
        let writer = Arc::new(Mutex::new(stdout()));
        return UsiEngine::new(BufReader::new(stdin()), writer)
            .run()
            .map_err(|e| {
                message.bytes(b"usi session i/o error: errno ");
                match e.raw_os_error() {
                    Some(errno) => message.i64(i64::from(errno)),
                    None => message.bytes(b"unknown"),
                };
            });
    };
    match cmd {
        b"perft" => perft_cmd(rest, message),
        other => {
            message
                .bytes(b"unknown subcommand `")
                .bytes(other)
                .bytes(b"`");
            Err(())
        }
    }
}

fn perft_cmd(args: &[&[u8]], message: &mut TextWriter<'_>) -> Result<(), ()> {
    let (mut pos, depth) = match args {
        [b"startpos", depth] => (Position::startpos(), parse_depth(depth, message)?),
        [b"sfen", sfen, depth] => {
            let pos = position_of(sfen, message)?;
            (pos, parse_depth(depth, message)?)
        }
        [b"sfen", sfen, b"moves", rest @ ..] => {
            let Some((depth_str, moves)) = rest.split_last() else {
                message
                    .bytes(b"perft sfen \xE2\x80\xA6 moves expects at least one move and a depth");
                return Err(());
            };
            let depth = parse_depth(depth_str, message)?;
            let mut pos = position_of(sfen, message)?;
            for m in moves {
                let parsed = parse_usi_move(m, &pos).map_err(|e| {
                    message.bytes(b"invalid usi move `").bytes(m).bytes(b"`: ");
                    e.write_message(message);
                })?;
                pos.do_move(parsed);
            }
            (pos, depth)
        }
        _ => {
            message.bytes(
                b"perft expects `startpos <depth>`, `sfen <SFEN> <depth>`, \
                  or `sfen <SFEN> moves <m1> ... <depth>`",
            );
            return Err(());
        }
    };
    let nodes = perft::perft(&mut pos, depth);
    let mut digits = [0u8; 32];
    let mut out = TextWriter::new(&mut digits);
    out.u64(nodes).byte(b'\n');
    let mut stdout = stdout().lock();
    stdout.write_all(out.as_bytes()).map_err(|_| ())
}

/// The position `sfen` spells.
fn position_of(sfen: &[u8], message: &mut TextWriter<'_>) -> Result<Position, ()> {
    parse_sfen(sfen).map_err(|e| {
        message.bytes(b"invalid sfen: ");
        e.write_message(message);
    })
}

fn parse_depth(text: &[u8], message: &mut TextWriter<'_>) -> Result<u32, ()> {
    atoi_u32(text).ok_or_else(|| {
        message
            .bytes(b"depth must be a non-negative integer, got `")
            .bytes(text)
            .bytes(b"`");
    })
}
