//! The connection's sending half and the client's log.
//!
//! [`Wire`] is the one place a line is sent from, by the session's thread and
//! by the search thread replying with a move alike. Every line goes out whole
//! in one write while the wire's lock is held, so a keep-alive can never land
//! inside a move. A move is sent only while one is owed ([`Wire::arm`]): a
//! search stopped because its game ended still replies, and that reply must
//! not reach a server that has moved on.
//!
//! [`Log`] is the record of the exchange: every line received and sent, each
//! with a timestamp, and the client's own notes, one line each, enough to
//! reconstruct a game afterwards. The password never reaches it.

use std::fs::File;
use std::io::{self, Write};
use std::sync::{Mutex, MutexGuard};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use yorkie_state::{CsaMoveBuf, TextWriter};

use crate::engine::Reply;

/// Room one log line is composed in. A line wider than this is cut short.
const LOG_LINE_BYTES: usize = 4096;

/// Room one sent line is composed in: the widest is a `LOGIN` with a name and
/// a password, each bounded by [`MAX_CREDENTIAL_BYTES`].
const SEND_LINE_BYTES: usize = 2 * MAX_CREDENTIAL_BYTES + 16;

/// The longest login name or password accepted.
pub(crate) const MAX_CREDENTIAL_BYTES: usize = 256;

/// Where the log goes.
pub(crate) enum Log {
    Stderr,
    File(File),
    /// Kept in memory for a test to read back.
    #[cfg(test)]
    Memory(Mutex<Vec<u8>>),
}

impl Log {
    /// One log line: the time, a tag saying what the line is (`<` received,
    /// `>` sent, `#` a note of the client's own) and the line.
    pub(crate) fn line(&self, tag: u8, parts: &[&[u8]]) {
        let mut bytes = [0u8; LOG_LINE_BYTES];
        let mut out = TextWriter::new(&mut bytes);
        write_timestamp(&mut out, SystemTime::now());
        out.byte(b' ').byte(tag).byte(b' ');
        for part in parts {
            out.bytes(part);
        }
        // The line ending is kept even when the line was cut short.
        let len = out.len().min(LOG_LINE_BYTES - 1);
        bytes[len] = b'\n';
        let line = &bytes[..=len];
        // One write per line, so lines the two threads log never interleave.
        let _ = match self {
            Self::Stderr => io::stderr().lock().write_all(line),
            Self::File(f) => (&*f).write_all(line),
            #[cfg(test)]
            Self::Memory(m) => {
                m.lock().unwrap().extend_from_slice(line);
                Ok(())
            }
        };
    }

    /// A note of the client's own.
    pub(crate) fn note(&self, parts: &[&[u8]]) {
        self.line(b'#', parts);
    }

    /// Each line of `text` as a note — for the statistics and PV lines, which
    /// are rendered as the USI layer renders them.
    #[cfg(feature = "verbose1")]
    pub(crate) fn notes(&self, text: &[u8]) {
        for line in text.split(|&b| b == b'\n') {
            if !line.is_empty() {
                self.note(&[line]);
            }
        }
    }

    /// Everything logged so far.
    #[cfg(test)]
    pub(crate) fn bytes(&self) -> Vec<u8> {
        match self {
            Self::Memory(m) => m.lock().unwrap().clone(),
            _ => Vec::new(),
        }
    }
}

/// Append `now` as UTC to the millisecond, `2026-11-27T12:34:56.789Z`.
fn write_timestamp(out: &mut TextWriter<'_>, now: SystemTime) {
    let since = now.duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = since.as_secs();
    let (y, m, d) = civil_from_days((secs / 86_400) as i64);
    let in_day = secs % 86_400;
    out.u64_padded(y as u64, 4)
        .byte(b'-')
        .u64_padded(m, 2)
        .byte(b'-')
        .u64_padded(d, 2)
        .byte(b'T')
        .u64_padded(in_day / 3600, 2)
        .byte(b':')
        .u64_padded(in_day / 60 % 60, 2)
        .byte(b':')
        .u64_padded(in_day % 60, 2)
        .byte(b'.')
        .u64_padded(u64::from(since.subsec_millis()), 3)
        .byte(b'Z');
}

/// The proleptic Gregorian date `days` after 1970-01-01.
fn civil_from_days(days: i64) -> (i64, u64, u64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u64;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u64;
    let y = yoe + era * 400 + i64::from(m <= 2);
    (y, m, d)
}

struct WireState<W> {
    /// The connection, once there is one.
    out: Option<W>,
    /// A move is owed: the next reply the engine gives is sent.
    armed: bool,
    /// When the last line went out.
    last_sent: Instant,
}

/// The sending half of the connection, shared by the session and the search.
pub(crate) struct Wire<W> {
    state: Mutex<WireState<W>>,
    pub(crate) log: Log,
}

impl<W: Write> Wire<W> {
    /// A wire with nothing to send to yet.
    pub(crate) fn new(log: Log) -> Self {
        Self {
            state: Mutex::new(WireState {
                out: None,
                armed: false,
                last_sent: Instant::now(),
            }),
            log,
        }
    }

    fn lock(&self) -> MutexGuard<'_, WireState<W>> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Send to `out` from now on.
    pub(crate) fn attach(&self, out: W) {
        let mut s = self.lock();
        s.out = Some(out);
        s.last_sent = Instant::now();
    }

    /// Write `line` and its line feed in one write.
    fn write(state: &mut WireState<W>, line: &[u8]) -> io::Result<()> {
        let mut bytes = [0u8; SEND_LINE_BYTES];
        let mut out = TextWriter::new(&mut bytes);
        out.bytes(line).byte(b'\n');
        if out.overflowed() {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        let len = out.len();
        let result = match state.out.as_mut() {
            Some(w) => w.write_all(&bytes[..len]).and_then(|()| w.flush()),
            None => Err(io::Error::from(io::ErrorKind::NotConnected)),
        };
        bytes.fill(0);
        state.last_sent = Instant::now();
        result
    }

    /// Send one line, and log it.
    pub(crate) fn send(&self, line: &[u8]) -> io::Result<()> {
        let result = Self::write(&mut self.lock(), line);
        self.log.line(b'>', &[line]);
        result
    }

    /// Send `LOGIN <name> <password>`, logging it with the password masked.
    pub(crate) fn send_login(&self, name: &[u8], password: &[u8]) -> io::Result<()> {
        let mut bytes = [0u8; SEND_LINE_BYTES];
        let mut line = TextWriter::new(&mut bytes);
        line.bytes(b"LOGIN ").bytes(name).byte(b' ').bytes(password);
        let len = line.len();
        let overflowed = line.overflowed();
        let result = if overflowed {
            Err(io::Error::from(io::ErrorKind::InvalidInput))
        } else {
            Self::write(&mut self.lock(), &bytes[..len])
        };
        bytes.fill(0);
        std::hint::black_box(&bytes);
        self.log.line(b'>', &[b"LOGIN ", name, b" ********"]);
        result
    }

    /// A move is owed: the engine's next reply goes out.
    pub(crate) fn arm(&self) {
        self.lock().armed = true;
    }

    /// No move is owed any more; a reply that arrives now is dropped.
    pub(crate) fn disarm(&self) {
        self.lock().armed = false;
    }

    /// Send the engine's reply, if a move is owed.
    pub(crate) fn send_reply(&self, reply: Reply) {
        let mut buf = CsaMoveBuf::new();
        let text: &[u8] = match reply {
            Reply::BestMove { mv } => buf.format(mv),
            Reply::Resign => b"%TORYO",
            Reply::Win => b"%KACHI",
        };
        let result = {
            let mut s = self.lock();
            if !s.armed {
                drop(s);
                self.log
                    .note(&[b"reply ", text, b" dropped: no move is owed"]);
                return;
            }
            s.armed = false;
            Self::write(&mut s, text)
        };
        self.log.line(b'>', &[text]);
        if let Err(e) = result {
            write_io_error(&self.log, b"sending the move failed", &e);
        }
    }

    /// Send a bare line feed if nothing has been sent for `idle`.
    pub(crate) fn keep_alive(&self, idle: std::time::Duration) -> io::Result<()> {
        let result = {
            let mut s = self.lock();
            if s.out.is_none() || s.last_sent.elapsed() < idle {
                return Ok(());
            }
            Self::write(&mut s, b"")
        };
        self.log.line(b'>', &[b"(keep-alive)"]);
        result
    }
}

/// Log an I/O failure, naming its errno when it has one.
pub(crate) fn write_io_error(log: &Log, what: &[u8], e: &io::Error) {
    let mut bytes = [0u8; 64];
    let mut out = TextWriter::new(&mut bytes);
    match e.raw_os_error() {
        Some(errno) => {
            out.bytes(b": errno ").i64(i64::from(errno));
        }
        None => {
            out.bytes(b": ").bytes(kind_name(e.kind()));
        }
    }
    log.note(&[what, out.as_bytes()]);
}

fn kind_name(kind: io::ErrorKind) -> &'static [u8] {
    match kind {
        io::ErrorKind::NotConnected => b"not connected",
        io::ErrorKind::InvalidInput => b"line too long",
        io::ErrorKind::UnexpectedEof => b"connection closed",
        io::ErrorKind::TimedOut => b"timed out",
        _ => b"i/o error",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use yorkie_state::{Position, parse_usi_move};

    fn wire() -> Wire<Vec<u8>> {
        let w = Wire::new(Log::Memory(Mutex::new(Vec::new())));
        w.attach(Vec::new());
        w
    }

    fn sent(w: &Wire<Vec<u8>>) -> Vec<u8> {
        w.lock().out.clone().unwrap()
    }

    #[test]
    fn the_password_is_sent_and_never_logged() {
        let w = wire();
        w.send_login(b"me", b"s3cret-pass").unwrap();
        assert_eq!(sent(&w), b"LOGIN me s3cret-pass\n");
        let log = w.log.bytes();
        assert!(!log.windows(11).any(|x| x == b"s3cret-pass"));
        assert!(log.ends_with(b" > LOGIN me ********\n"));
    }

    #[test]
    fn a_reply_goes_out_only_while_a_move_is_owed() {
        let w = wire();
        let pos = Position::startpos();
        let mv = parse_usi_move(b"7g7f", &pos).unwrap();
        w.send_reply(Reply::BestMove { mv });
        assert_eq!(sent(&w), b"");
        w.arm();
        w.send_reply(Reply::BestMove { mv });
        // Only one reply per arming.
        w.send_reply(Reply::BestMove { mv });
        assert_eq!(sent(&w), b"+7776FU\n");
        w.arm();
        w.disarm();
        w.send_reply(Reply::Resign);
        w.arm();
        w.send_reply(Reply::Resign);
        assert_eq!(sent(&w), b"+7776FU\n%TORYO\n");
    }

    #[test]
    fn a_keep_alive_is_a_bare_line_and_only_after_the_idle_time() {
        let w = wire();
        w.keep_alive(std::time::Duration::from_secs(3600)).unwrap();
        assert_eq!(sent(&w), b"");
        w.keep_alive(std::time::Duration::ZERO).unwrap();
        assert_eq!(sent(&w), b"\n");
    }

    #[test]
    fn timestamps_are_utc_to_the_millisecond() {
        let mut bytes = [0u8; 64];
        let mut out = TextWriter::new(&mut bytes);
        let t = UNIX_EPOCH + std::time::Duration::from_millis(1_795_000_496_789);
        write_timestamp(&mut out, t);
        assert_eq!(out.as_bytes(), b"2026-11-18T11:14:56.789Z");
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(19_782), (2024, 2, 29));
    }
}
