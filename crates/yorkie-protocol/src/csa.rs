//! The CSA server-protocol client (protocol version 1.2.1): it connects to a
//! game server over plain TCP, logs in, plays the games the server offers and
//! logs out.
//!
//! [`CsaEngine`] drives the same [`Engine`] the USI layer drives, through
//! [`CsaSink`], the [`EngineSink`] this client compiles. The session's own
//! thread reads the server's lines and answers them; the engine's search thread
//! sends its move itself the moment it has one, the way the USI layer writes
//! `bestmove`, so no hand-over sits between a finished search and the wire.
//!
//! The connection is blocking I/O on the session's thread, with a read timeout
//! that wakes it regularly to send the keep-alive and to look at the shutdown
//! flag. No async runtime is involved, and the search path is untouched: the
//! reply is the only thing the search hands this layer.
//!
//! What a game needs from the server is read from it: the clock is the
//! server's, from the time control in the summary and the `,T<n>` on every
//! move it relays, and a move it relays that the client's position does not
//! allow is logged and followed by the server's verdict rather than argued
//! with. The entering-king declaration, the move limit and re-login after a
//! dropped connection are not implemented; a game the engine resigns ends with
//! `%TORYO`.

mod clock;
mod session;
mod summary;
mod wire;

use std::ffi::{OsStr, OsString};
use std::io::{self, Read};
use std::net::TcpStream;
use std::os::unix::ffi::{OsStrExt as _, OsStringExt as _};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use yorkie_search::BookHit;
#[cfg(feature = "verbose3")]
use yorkie_search::causes::ReplyMarks;
#[cfg(feature = "verbose2")]
use yorkie_search::{PvInfo, PvSink};
use yorkie_state::text::atoi_u16;
use yorkie_state::{Color, TextWriter};
#[cfg(feature = "verbose1")]
use yorkie_storage::clear_alloc_count;

use crate::engine::{
    Engine, EngineSink, GoOutcome, NOTICE_BYTES, PositionSfen, ReadyOutcome, Reply,
};
use crate::usi::ENGINE_NAME;
use session::{Phase, Session, Step};
use summary::SideTime;

use wire::{Log, MAX_CREDENTIAL_BYTES, Wire, write_io_error};

/// The environment variables the connection settings are read from.
pub const HOST_VAR: &str = "YORKIE_CSA_HOST";
pub const PORT_VAR: &str = "YORKIE_CSA_PORT";
pub const ID_VAR: &str = "YORKIE_CSA_ID";
pub const PASSWORD_VAR: &str = "YORKIE_CSA_PASSWORD";
/// The file the log is appended to; standard error when unset.
pub const LOG_VAR: &str = "YORKIE_CSA_LOG";

/// How long the client stays silent before it sends a bare line feed. The
/// protocol forbids sending them more often than every 30 seconds.
const KEEP_ALIVE_IDLE: Duration = Duration::from_secs(40);

/// How often a blocked read gives the session's thread back, to send the
/// keep-alive and to look at the shutdown flag.
const READ_TIMEOUT: Duration = Duration::from_millis(500);

/// How long the client waits for the answer to its `LOGOUT` before it closes
/// the connection anyway.
const LOGOUT_WAIT: Duration = Duration::from_secs(10);

/// Room for one received line. Every line the protocol defines is far shorter;
/// a longer one is logged as such and skipped.
const LINE_BYTES: usize = 4096;

/// What the connection settings lack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingsError {
    /// The variable is not set, or is empty.
    Missing(&'static str),
    /// The variable's value is not one the setting takes.
    Invalid(&'static str),
}

impl SettingsError {
    /// This error's message.
    pub fn write_message(&self, out: &mut TextWriter<'_>) {
        match self {
            Self::Missing(var) => {
                out.bytes(var.as_bytes()).bytes(b" is not set");
            }
            Self::Invalid(var) => {
                out.bytes(var.as_bytes()).bytes(b" is not valid");
            }
        }
    }
}

/// Where to connect and who to log in as, read at start-up from the
/// environment.
///
/// The password is held only until `LOGIN` is sent, and is then overwritten:
/// re-login after a dropped connection is not implemented, so nothing needs it
/// afterwards. The variable itself is removed from the process environment as
/// it is read, so no child process and no later reader of the environment
/// table sees it; the copy the kernel placed in the process's initial stack at
/// `exec` is outside this program's reach.
pub struct CsaSettings {
    host: OsString,
    port: u16,
    id: Vec<u8>,
    password: Vec<u8>,
    log: Option<PathBuf>,
}

impl CsaSettings {
    /// Read the settings from [`HOST_VAR`], [`PORT_VAR`], [`ID_VAR`],
    /// [`PASSWORD_VAR`] and, optionally, [`LOG_VAR`].
    ///
    /// # Safety
    ///
    /// Removes [`PASSWORD_VAR`] from the environment, so no other thread may
    /// be reading or writing the environment while this runs: call it at the
    /// start of `main`, before any thread is started.
    pub unsafe fn from_env() -> Result<Self, SettingsError> {
        let var = |name: &'static str| match std::env::var_os(name) {
            Some(v) if !v.is_empty() => Ok(v),
            _ => Err(SettingsError::Missing(name)),
        };
        let host = var(HOST_VAR)?;
        let port = var(PORT_VAR)?;
        let id = var(ID_VAR)?;
        let password = var(PASSWORD_VAR)?;
        // SAFETY: the caller guarantees no other thread touches the
        // environment.
        unsafe { std::env::remove_var(PASSWORD_VAR) };
        let port = atoi_u16(port.as_bytes())
            .filter(|&p| p != 0)
            .ok_or(SettingsError::Invalid(PORT_VAR))?;
        let credential = |v: OsString, name: &'static str| {
            let bytes = v.into_vec();
            let fits = bytes.len() <= MAX_CREDENTIAL_BYTES;
            if fits && bytes.iter().all(|&b| b.is_ascii_graphic()) {
                Ok(bytes)
            } else {
                Err(SettingsError::Invalid(name))
            }
        };
        if host.to_str().is_none() {
            return Err(SettingsError::Invalid(HOST_VAR));
        }
        Ok(Self {
            host,
            port,
            id: credential(id, ID_VAR)?,
            password: credential(password, PASSWORD_VAR)?,
            log: std::env::var_os(LOG_VAR)
                .filter(|v| !v.is_empty())
                .map(PathBuf::from),
        })
    }
}

/// Raised by the `SIGTERM` handler; read by the session between lines.
static TERM_REQUESTED: AtomicBool = AtomicBool::new(false);

extern "C" fn on_sigterm(_: libc::c_int) {
    TERM_REQUESTED.store(true, Ordering::Relaxed);
}

/// Catch `SIGTERM` into [`TERM_REQUESTED`] instead of dying on it.
fn install_sigterm_handler() -> io::Result<()> {
    // SAFETY: the handler only stores to an atomic, which is
    // async-signal-safe; the struct is fully initialised before the call.
    let rc = unsafe {
        let mut action: libc::sigaction = std::mem::zeroed();
        action.sa_sigaction = on_sigterm as extern "C" fn(libc::c_int) as libc::sighandler_t;
        libc::sigemptyset(&mut action.sa_mask);
        action.sa_flags = libc::SA_RESTART;
        libc::sigaction(libc::SIGTERM, &action, std::ptr::null_mut())
    };
    if rc == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

/// Where the engine's output goes in this client: its move to the server, and
/// everything else to the log.
///
/// A clone is a handle to the same wire, not a second one.
#[derive(Clone)]
pub struct CsaSink {
    wire: Arc<Wire<TcpStream>>,
}

impl CsaSink {
    /// The per-reply statistics line, to the log.
    #[cfg(feature = "verbose1")]
    fn log_stats(&self, #[cfg(feature = "verbose3")] marks: ReplyMarks) {
        let mut bytes = [0u8; 4096];
        let mut cursor = io::Cursor::new(&mut bytes[..]);
        crate::usi::emit_stats(
            &mut cursor,
            #[cfg(feature = "verbose3")]
            marks,
        );
        let len = cursor.position() as usize;
        self.wire.log.notes(&bytes[..len]);
    }

    /// One PV line, to the log.
    #[cfg(feature = "verbose2")]
    fn log_pv(&self, info: &PvInfo) {
        let mut bytes = [0u8; crate::usi::INFO_BYTES + 64];
        let mut cursor = io::Cursor::new(&mut bytes[..]);
        let _ = crate::usi::write_pv_info(&mut cursor, info);
        let len = cursor.position() as usize;
        self.wire.log.notes(&bytes[..len]);
    }
}

impl EngineSink for CsaSink {
    #[cfg(feature = "verbose2")]
    type PvOutput = Self;

    fn notice(&self, msg: &[u8]) -> io::Result<()> {
        self.wire.log.note(&[b"info string ", msg]);
        Ok(())
    }

    #[cfg(feature = "verbose1")]
    fn diagnostic(&self, msg: &[u8]) {
        self.wire.log.note(&[b"info string ", msg]);
    }

    fn reply(&self, reply: Reply, #[cfg(feature = "verbose3")] marks: ReplyMarks) {
        // The move goes out first; what the log says about it can wait.
        self.wire.send_reply(reply);
        #[cfg(feature = "verbose1")]
        self.log_stats(
            #[cfg(feature = "verbose3")]
            marks,
        );
    }

    // The book's candidate lines are a USI analysis surface; this client's log
    // records the move it chose, which is the reply below.
    #[cfg(feature = "verbose2")]
    fn book_candidates(&self, _hit: &BookHit, _hashfull: u32, _time_ms: u64) {}

    fn book_reply(
        &self,
        hit: &BookHit,
        #[cfg(feature = "verbose2")] _hashfull: u32,
        #[cfg(feature = "verbose2")] _time_ms: u64,
    ) {
        self.wire.send_reply(Reply::BestMove { mv: hit.best });
        #[cfg(feature = "verbose1")]
        self.log_stats(
            #[cfg(feature = "verbose3")]
            ReplyMarks::NONE,
        );
    }

    #[cfg(feature = "verbose2")]
    fn pv_block(&self, infos: &[PvInfo]) {
        for info in infos {
            self.log_pv(info);
        }
    }

    #[cfg(feature = "verbose2")]
    fn pv_output(&self) -> Self {
        self.clone()
    }
}

#[cfg(feature = "verbose2")]
impl PvSink for CsaSink {
    fn emit(&mut self, info: &PvInfo) {
        self.log_pv(info);
    }
}

/// What one read of the connection produced.
enum Received {
    /// A line of that many bytes, read at that instant.
    Line(usize, Instant),
    /// A line too long to hold, skipped to its end.
    TooLong,
    /// Nothing arrived before the read timed out.
    Idle,
    /// The server closed the connection.
    Closed,
}

/// The receiving half: lines out of the byte stream, each stamped with the
/// moment the read that completed it returned.
struct LineReader<R> {
    inner: R,
    buf: Vec<u8>,
    start: usize,
    end: usize,
    /// Inside a line too long to hold, dropping bytes until its end.
    skipping: bool,
    stamp: Instant,
}

impl<R: Read> LineReader<R> {
    fn new(inner: R) -> Self {
        Self {
            inner,
            buf: vec![0; 4 * LINE_BYTES],
            start: 0,
            end: 0,
            skipping: false,
            stamp: Instant::now(),
        }
    }

    /// The next line into `line`, without its `\n` or `\r\n`.
    fn next(&mut self, line: &mut [u8; LINE_BYTES]) -> io::Result<Received> {
        loop {
            if let Some(i) = self.buf[self.start..self.end]
                .iter()
                .position(|&b| b == b'\n')
            {
                let text = &self.buf[self.start..self.start + i];
                self.start += i + 1;
                if std::mem::take(&mut self.skipping) {
                    return Ok(Received::TooLong);
                }
                let text = text.strip_suffix(b"\r").unwrap_or(text);
                if text.len() > LINE_BYTES {
                    return Ok(Received::TooLong);
                }
                line[..text.len()].copy_from_slice(text);
                return Ok(Received::Line(text.len(), self.stamp));
            }
            if self.start > 0 {
                self.buf.copy_within(self.start..self.end, 0);
                self.end -= self.start;
                self.start = 0;
            }
            if self.end == self.buf.len() {
                self.skipping = true;
                self.end = 0;
            }
            match self.inner.read(&mut self.buf[self.end..]) {
                Ok(0) => return Ok(Received::Closed),
                Ok(n) => {
                    self.end += n;
                    self.stamp = Instant::now();
                }
                Err(e)
                    if matches!(
                        e.kind(),
                        io::ErrorKind::WouldBlock
                            | io::ErrorKind::TimedOut
                            | io::ErrorKind::Interrupted
                    ) =>
                {
                    return Ok(Received::Idle);
                }
                Err(e) => return Err(e),
            }
        }
    }
}

/// The CSA client: the connection, the protocol state and the engine they
/// drive.
pub struct CsaEngine {
    settings: CsaSettings,
    wire: Arc<Wire<TcpStream>>,
    engine: Engine<CsaSink>,
    session: Session,
}

impl CsaEngine {
    /// A client for `settings`, its engine built but not yet initialised.
    /// Fails only when the log file cannot be opened.
    pub fn new(settings: CsaSettings) -> io::Result<Self> {
        let log = match &settings.log {
            Some(path) => Log::File(
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)?,
            ),
            None => Log::Stderr,
        };
        let wire = Arc::new(Wire::new(log));
        let engine = Engine::new(CsaSink {
            wire: Arc::clone(&wire),
        });
        Ok(Self {
            settings,
            wire,
            engine,
            session: Session::new(),
        })
    }

    fn log(&self) -> &Log {
        &self.wire.log
    }

    /// Initialise the engine, connect, log in and play until the server or a
    /// `SIGTERM` ends the session.
    ///
    /// A `SIGTERM` between games logs out and returns. During a game — or once
    /// a game has been agreed to — the game is played to its end first, and
    /// the client logs out as soon as it is over.
    pub fn run(mut self) -> io::Result<()> {
        install_sigterm_handler()?;
        // This thread reads the server; it runs where the host keeps its
        // non-search work.
        self.engine.pin_to_shared_cpu();
        self.log().note(&[b"starting ", ENGINE_NAME]);
        self.ready()?;
        let stream = self.connect()?;
        let mut reader = LineReader::new(stream.try_clone()?);
        self.wire.attach(stream);
        let login = self
            .wire
            .send_login(&self.settings.id, &self.settings.password);
        self.settings.password.fill(0);
        std::hint::black_box(&self.settings.password);
        self.settings.password = Vec::new();
        login?;

        let mut line = [0u8; LINE_BYTES];
        let mut logout_deadline = None;
        loop {
            if TERM_REQUESTED.load(Ordering::Relaxed)
                && logout_deadline.is_none()
                && matches!(self.session.phase(), Phase::LoggingIn | Phase::Waiting)
            {
                self.log().note(&[b"SIGTERM: logging out"]);
                self.session.logging_out();
                self.wire.send(b"LOGOUT")?;
                logout_deadline = Some(Instant::now() + LOGOUT_WAIT);
            }
            if logout_deadline.is_some_and(|d| Instant::now() >= d) {
                self.log().note(&[b"no answer to LOGOUT; closing"]);
                return Ok(());
            }
            match reader.next(&mut line)? {
                Received::Line(len, at) => {
                    let text = &line[..len];
                    self.log().line(b'<', &[text]);
                    let step = self.session.on_line(text);
                    if self.act(step, at)? {
                        self.engine.finish_search_join();
                        return Ok(());
                    }
                }
                Received::TooLong => self.log().note(&[b"received a line too long to read"]),
                Received::Idle => {}
                Received::Closed => {
                    self.wire.disarm();
                    self.engine.stop();
                    self.engine.finish_search_join();
                    if self.session.phase() == Phase::LoggingOut {
                        self.log().note(&[b"connection closed"]);
                        return Ok(());
                    }
                    self.log().note(&[b"the server closed the connection"]);
                    return Err(io::Error::from(io::ErrorKind::UnexpectedEof));
                }
            }
            if let Err(e) = self.wire.keep_alive(KEEP_ALIVE_IDLE) {
                write_io_error(self.log(), b"keep-alive failed", &e);
                return Err(e);
            }
        }
    }

    /// The heavy initialisation, before the client logs in, so that the engine
    /// is ready the moment a game is offered.
    fn ready(&mut self) -> io::Result<()> {
        let mut bytes = [0u8; NOTICE_BYTES];
        let mut out = TextWriter::new(&mut bytes);
        match self.engine.ready()? {
            ReadyOutcome::Ready => {
                // What initialisation allocated belongs to no reply.
                #[cfg(feature = "verbose1")]
                {
                    clear_alloc_count();
                    crate::stats::clear_search_counts();
                }
                self.log().note(&[b"engine ready"]);
                return Ok(());
            }
            ReadyOutcome::LoadFailed(reason) => {
                out.bytes(b"eval load failed: ");
                reason.write_message(&mut out);
            }
            ReadyOutcome::LayoutMismatch(reason) => {
                out.bytes(b"NUMA layout mismatch: ");
                reason.write_message(&mut out);
            }
            ReadyOutcome::YorkieBookFailed(path, reason) => {
                out.bytes(b"yorkie book load failed : ")
                    .path(&path)
                    .bytes(b" : ");
                reason.write_message(|fragment| {
                    out.bytes(fragment);
                });
            }
        }
        self.log().note(&[out.as_bytes()]);
        Err(io::Error::other("the engine could not be made ready"))
    }

    /// Open the connection: plain TCP, `TCP_NODELAY` so a move goes out the
    /// moment it is written, and a read timeout so the session's thread wakes
    /// up regularly. The address is not logged.
    fn connect(&self) -> io::Result<TcpStream> {
        self.log().note(&[b"connecting to the server"]);
        let host = OsStr::to_str(&self.settings.host).unwrap_or_default();
        let stream = TcpStream::connect((host, self.settings.port)).inspect_err(|e| {
            write_io_error(self.log(), b"connecting failed", e);
        })?;
        stream.set_nodelay(true)?;
        stream.set_read_timeout(Some(READ_TIMEOUT))?;
        self.log().note(&[b"connected"]);
        Ok(stream)
    }

    /// Do what `step` asks. Returns whether the session is over.
    fn act(&mut self, step: Step<'_>, at: Instant) -> io::Result<bool> {
        match step {
            Step::Nothing => {}
            Step::LoginOk => self.log().note(&[b"logged in"]),
            Step::LoginRefused => {
                self.log().note(&[b"login refused"]);
                return Err(io::Error::from(io::ErrorKind::PermissionDenied));
            }
            Step::LoggedOut => {
                self.log().note(&[b"logged out"]);
                return Ok(true);
            }
            Step::Offered => {
                self.log_offer();
                self.engine.new_game()?;
                let game = self.session.game().expect("an offered game");
                let mut bytes = [0u8; 512];
                let mut agree = TextWriter::new(&mut bytes);
                agree.bytes(b"AGREE ").bytes(&game.summary.game_id);
                if agree.overflowed() {
                    self.wire.send(b"AGREE")?;
                } else {
                    self.wire.send(agree.as_bytes())?;
                }
                self.session.agreed();
            }
            Step::Unreadable(e) => {
                let mut bytes = [0u8; 256];
                let mut out = TextWriter::new(&mut bytes);
                out.bytes(b"unreadable game summary: ");
                e.write_message(&mut out);
                self.log().note(&[out.as_bytes(), b"; refusing the game"]);
                self.wire.send(b"REJECT")?;
                self.session.refused();
            }
            Step::Rejected => self.log().note(&[b"the game was refused"]),
            Step::Started => {
                self.log_clock();
                self.move_if_owed(at);
            }
            Step::Moved { ours } => {
                self.log_clock();
                if !ours {
                    self.move_if_owed(at);
                }
            }
            Step::Unplayable(line, e) => {
                let mut bytes = [0u8; 256];
                let mut out = TextWriter::new(&mut bytes);
                e.write_message(&mut out);
                self.log().note(&[
                    b"the server relayed ",
                    line,
                    b", which this position does not allow (",
                    out.as_bytes(),
                    b"); following the server",
                ]);
                self.wire.disarm();
                self.engine.stop();
            }
            Step::GameOver { reason, result } => {
                self.wire.disarm();
                self.engine.stop();
                self.engine.finish_search_join();
                self.log().note(&[
                    b"game over: ",
                    result.name(),
                    b" (",
                    reason.map_or(&b"no reason given"[..], |r| r.name()),
                    b")",
                ]);
            }
            Step::Unexpected(_) => self.log().note(&[b"(not expected here; ignored)"]),
        }
        Ok(false)
    }

    /// Search and answer, if the engine owes the server a move.
    fn move_if_owed(&mut self, at: Instant) {
        let Self {
            session,
            engine,
            wire,
            ..
        } = self;
        let Some(game) = session.game() else {
            return;
        };
        if !game.our_move() {
            return;
        }
        let sfen = match game.start_fields() {
            None => PositionSfen::StartPos,
            Some(fields) => PositionSfen::Sfen(fields),
        };
        wire.arm();
        if engine.set_position(sfen, game.usi_moves()).is_err() {
            // The moves were each checked as they arrived, so only the length
            // of the game can be refused; there is nothing left to search.
            wire.log
                .note(&[b"the engine refused the game's position; resigning"]);
            wire.send_reply(Reply::Resign);
            return;
        }
        match engine.go_at(game.go_params(), at) {
            GoOutcome::Started => {}
            GoOutcome::NoNetwork => {
                wire.log.note(&[b"no evaluation network loaded; resigning"]);
                wire.send_reply(Reply::Resign);
            }
        }
    }

    /// Both clocks as the client now holds them, in the server's unit: the
    /// figures a game's record is checked against.
    fn log_clock(&self) {
        let Some(game) = self.session.game() else {
            return;
        };
        let mut bytes = [0u8; 128];
        let mut out = TextWriter::new(&mut bytes);
        out.bytes(b"clock +")
            .u64(game.clock.remaining_units(Color::Black))
            .bytes(b" -")
            .u64(game.clock.remaining_units(Color::White));
        self.log().note(&[out.as_bytes()]);
    }

    /// What the offered game is, for the log.
    fn log_offer(&self) {
        let Some(game) = self.session.game() else {
            return;
        };
        let s = &game.summary;
        let mut bytes = [0u8; 1024];
        let mut out = TextWriter::new(&mut bytes);
        out.bytes(b"offered ")
            .bytes(&s.game_id)
            .bytes(b": protocol ")
            .bytes(&s.protocol_version)
            .bytes(b", format ")
            .bytes(&s.format)
            .bytes(b"; playing ")
            .byte(sign(s.your_turn))
            .bytes(b" against ")
            .bytes(&s.names[s.your_turn.flip().index()]);
        for side in [Color::Black, Color::White] {
            out.bytes(b"; time ").byte(sign(side)).byte(b' ');
            write_side_time(&mut out, &s.time[side.index()]);
        }
        match s.max_moves {
            Some(n) => out.bytes(b"; max moves ").u64(n),
            None => out.bytes(b"; no move limit"),
        };
        if !s.moves.is_empty() {
            out.bytes(b"; ")
                .u64(s.moves.len() as u64)
                .bytes(b" moves already played");
        }
        self.log().note(&[out.as_bytes()]);
    }
}

fn sign(side: Color) -> u8 {
    match side {
        Color::Black => b'+',
        Color::White => b'-',
    }
}

fn write_side_time(out: &mut TextWriter<'_>, t: &SideTime) {
    out.bytes(b"total ")
        .u64(t.total)
        .bytes(b" byoyomi ")
        .u64(t.byoyomi)
        .bytes(b" increment ")
        .u64(t.increment)
        .bytes(b" least ")
        .u64(t.least)
        .bytes(b" (unit ")
        .u64(t.unit_ms)
        .bytes(b" ms)");
    if let Some(r) = t.remaining {
        out.bytes(b" remaining ").u64(r);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reader handing out its chunks one per read, then reporting a timeout,
    /// then the end of the stream.
    struct Chunks(Vec<&'static [u8]>, bool);

    impl Read for Chunks {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            if self.0.is_empty() {
                if std::mem::take(&mut self.1) {
                    return Err(io::Error::from(io::ErrorKind::WouldBlock));
                }
                return Ok(0);
            }
            let chunk = self.0[0];
            let n = chunk.len().min(buf.len());
            buf[..n].copy_from_slice(&chunk[..n]);
            if n == chunk.len() {
                self.0.remove(0);
            } else {
                self.0[0] = &chunk[n..];
            }
            Ok(n)
        }
    }

    fn lines(chunks: Vec<&'static [u8]>) -> Vec<String> {
        let mut r = LineReader::new(Chunks(chunks, true));
        let mut line = [0u8; LINE_BYTES];
        let mut out = Vec::new();
        loop {
            match r.next(&mut line).unwrap() {
                Received::Line(n, _) => out.push(String::from_utf8_lossy(&line[..n]).into_owned()),
                Received::TooLong => out.push("<too long>".into()),
                Received::Idle => out.push("<idle>".into()),
                Received::Closed => return out,
            }
        }
    }

    #[test]
    fn lines_are_cut_wherever_the_reads_split_them() {
        assert_eq!(
            lines(vec![b"LOGIN:a OK\r\n+77", b"76FU,T1\n\n-33", b"34FU,T2\n"]),
            ["LOGIN:a OK", "+7776FU,T1", "", "-3334FU,T2", "<idle>"]
        );
    }

    #[test]
    fn a_line_too_long_is_skipped_to_its_end() {
        let long: &'static [u8] = Box::leak(vec![b'x'; 5 * LINE_BYTES].into_boxed_slice());
        assert_eq!(
            lines(vec![b"a\n", long, b"y\nb\n"]),
            ["a", "<too long>", "b", "<idle>"]
        );
    }
}
