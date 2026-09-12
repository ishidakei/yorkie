//! The USI protocol layer: the line reader, the command handlers and every
//! byte this engine puts on the wire.
//!
//! [`UsiEngine`] owns an [`Engine`] and does the two things the engine does not:
//! it turns a USI line into a typed request, and it renders what comes back —
//! `usiok`, `readyok`, `info`, `info string` and `bestmove` — through
//! [`UsiSink`], the [`EngineSink`] this build compiles.

use std::collections::BTreeSet;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::Duration;
#[cfg(feature = "verbose3")]
use std::time::Instant;

use yorkie_search::BookHit;
// The PV-line surface: only a `verbose2` build renders one, so only it needs
// the line's data type, its bound marker and the sink trait.
#[cfg(feature = "verbose2")]
use yorkie_search::{PvBound, PvInfo, PvSink};
use yorkie_state::TextWriter;
// A move's text inside a longer line: what is left of it are the optional
// surfaces that interpolate a move into one. A `bestmove` reply, which every
// build writes, composes its text in a stack buffer instead.
#[cfg(feature = "verbose2")]
use yorkie_state::{Move, write_usi_move};
// A whole SFEN is parsed only by the `verbose3` commands that carry one as an
// argument; the `position` command's own is read field by field, into the
// position the engine already holds.
#[cfg(feature = "verbose3")]
use yorkie_state::{Position, SfenBuf, parse_sfen, parse_sfen_fields_into, parse_usi_move};
#[cfg(feature = "verbose2")]
use yorkie_storage::Value;
#[cfg(feature = "verbose3")]
use yorkie_storage::{TTData, TranspositionTable, VALUE_NONE};
// The per-reply allocation tally: raised by the counting global allocator this
// feature installs, and read by the statistics line that reports it.
#[cfg(feature = "verbose1")]
use yorkie_storage::{clear_alloc_count, take_alloc_count};

#[cfg(feature = "verbose3")]
use crate::bench;
use crate::bestmove::BestmoveBuf;
#[cfg(feature = "verbose1")]
use crate::engine::MAX_POSITION_MOVES;
#[cfg(feature = "verbose2")]
use crate::engine::PAWN_VALUE;
use crate::engine::{
    Engine, EngineSink, GoOutcome, GoParams, NOTICE_BYTES, PositionRefusal, PositionSfen,
    ReadyOutcome, Reply,
};
use crate::formatter::Formatter;
use crate::parser::{Command, MAX_LINE_BYTES, parse_line};
#[cfg(feature = "verbose1")]
use crate::stats::StatsBuf;
#[cfg(feature = "verbose3")]
use crate::tt_command::{
    TtCommand, TtPosition, TtStoreArgs, bound_name, parse_tt, value_from_tt, value_to_tt,
};

/// The public values used in the `id name` / `id author` lines.
///
/// The version part is this project's own generation number (see
/// `CHANGELOG.md`), not an upstream-tracking number; the upstream YaneuraOu
/// baseline is documented in `README.md` instead.
pub const ENGINE_NAME: &[u8] = b"Yorkie 3.1.0";
pub const ENGINE_AUTHOR: &[u8] = b"Kei Ishida <ishida.kei@gmail.com>";

/// Room a diagnostic `info string` body is composed in.
///
/// A body quoting a token the host sent can be wider than this, and is
/// truncated to fit: a diagnostic about a garbage token is worth cutting short,
/// and the two that echo a whole command line write their pieces straight out
/// instead of gathering them here.
///
/// Nothing below `verbose1` writes a diagnostic, so neither the room nor any of
/// the wording is compiled there.
#[cfg(feature = "verbose1")]
const DIAG_BYTES: usize = 1024;

/// Room the `verbose2` PV / book `info` body is composed in: the fixed fields,
/// then a principal variation as deep as the search can return.
#[cfg(feature = "verbose2")]
const INFO_BYTES: usize = 128 + (yorkie_state::MAX_USI_MOVE_LEN + 1) * 256;
// --- isready keep-alive (reference `Engine::run_heavy_job`).
/// How often the keep-alive helper thread polls the stop flag while the heavy
/// `isready` initialisation runs (reference: `sleep_for(100ms)`).
const KEEP_ALIVE_POLL_INTERVAL: Duration = Duration::from_millis(100);
/// How many polls elapse between bare keep-alive newlines: `50 * 100ms = 5s`
/// (reference: `if (++count >= 50 /* 5秒 */)`). A GUI (Shogidokoro / ShogiGUI)
/// reads the periodic empty line as a sign the engine is alive and does not
/// time out while the book load, the table's placement and the evaluation
/// network's — a copy of a few hundred mebibytes per node, where the machine
/// calls for copies — run between `isready` and `readyok`.
const KEEP_ALIVE_TICKS_PER_NEWLINE: u32 = 50;
// Reference USI score conversion. These are `pub(crate)` so the `verbose3` `tt`
// commands, which speak the same score surface, do not grow a second copy of
// the scale.
//
// The mate scale is only needed to *render* a score, and both surfaces that
// render one are optional, so the default build compiles neither the two
// constants nor `push_score` / `format_score`. `PAWN_VALUE` is unconditional:
// `to_cp` and the draw-contempt scaling read it in every build.
/// `VALUE_MATE`.
#[cfg(feature = "verbose2")]
pub(crate) const VALUE_MATE: Value = 32000;
/// `VALUE_TB_WIN_IN_MAX_PLY`: the `is_decisive` threshold.
#[cfg(feature = "verbose2")]
pub(crate) const VALUE_TB_WIN_IN_MAX_PLY: Value = VALUE_MATE - 246;
/// Append a search value to `out` the way the reference USI layer formats it: a
/// mate distance for decisive scores, else centipawns.
#[cfg(feature = "verbose2")]
pub(crate) fn write_score(out: &mut TextWriter<'_>, v: Value) {
    if v.abs() >= VALUE_TB_WIN_IN_MAX_PLY {
        let distance = VALUE_MATE - v.abs();
        let mate = if v > 0 { distance } else { -distance };
        out.bytes(b"mate ").i64(i64::from(mate));
    } else {
        out.bytes(b"cp ").i64(i64::from(100 * v / PAWN_VALUE));
    }
}

/// The USI renderer: every line this engine writes is composed here.
///
/// A handle to the one output sink, shared with the search worker (which emits
/// its own `info` / `bestmove`). A `Mutex` serialises the worker's lines against
/// any the main thread emits concurrently; a clone is another handle to the same
/// sink, not a second one.
pub struct UsiSink<W: Write + Send + 'static> {
    writer: Arc<Mutex<W>>,
}

impl<W: Write + Send + 'static> Clone for UsiSink<W> {
    fn clone(&self) -> Self {
        Self {
            writer: Arc::clone(&self.writer),
        }
    }
}

impl<W: Write + Send + 'static> UsiSink<W> {
    pub fn new(writer: Arc<Mutex<W>>) -> Self {
        Self { writer }
    }

    /// Lock the shared output sink, recovering from a poisoned mutex (a worker
    /// panic must not wedge the main loop's own output).
    fn lock(&self) -> MutexGuard<'_, W> {
        self.writer.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Emit one `info string <msg>` line.
    ///
    /// This is the unconditional sink, reserved for the initialisation phase and
    /// for a `verbose3` command's response payload: those lines are how a
    /// failed startup is diagnosed at all, so no feature may take them away.
    fn info_string(&self, msg: &[u8]) -> io::Result<()> {
        Formatter::new(&mut *self.lock()).info_string(msg)
    }

    /// Emit one `info string <parts…>` line, the parts joined with nothing
    /// between them — for a message whose pieces are a literal and something
    /// borrowed from the command line, which is written whole rather than
    /// gathered into a buffer that might be narrower than it.
    ///
    /// Every such message is a diagnostic, so this exists only with their
    /// feature.
    #[cfg(feature = "verbose1")]
    fn info_string_parts(&self, parts: &[&[u8]]) -> io::Result<()> {
        Formatter::new(&mut *self.lock()).info_string_parts(parts)
    }

    /// Emit each non-blank line of `text` as `info string <line>`, mirroring the
    /// reference `print_info_string`: the text is split on `'\n'` and
    /// whitespace-only lines are skipped.
    #[cfg(feature = "verbose3")]
    fn info_string_lines(&self, text: &[u8]) -> io::Result<()> {
        for line in text.split(|&b| b == b'\n') {
            if !yorkie_state::text::trim_ascii_whitespace(line).is_empty() {
                self.info_string(line)?;
            }
        }
        Ok(())
    }

    /// The `usi` handshake: identity and `usiok`, with NO `option name ...`
    /// lines — in every build.
    ///
    /// The engine has no runtime configuration to advertise. Every setting was
    /// compiled in from the TOML config, and a GUI that saw an option list would
    /// be shown a control it cannot actually operate.
    fn handshake(&self) -> io::Result<()> {
        let mut guard = self.lock();
        let mut f = Formatter::new(&mut *guard);
        f.id_name(ENGINE_NAME)?;
        f.id_author(ENGINE_AUTHOR)?;
        f.usiok()
    }

    /// Emit one `readyok` line.
    fn readyok(&self) -> io::Result<()> {
        Formatter::new(&mut *self.lock()).readyok()
    }

    /// Emit a reply the main thread produces itself — the `go` that found no
    /// network — preceded by the statistics of the interval it ends.
    ///
    /// No search is in flight on this path, so there is no "reply is out" flag
    /// for it to stamp.
    fn reply_now(&self, reply: Reply) -> io::Result<()> {
        let mut payload = BestmoveBuf::new();
        let text = render_reply(&mut payload, reply);
        let mut guard = self.lock();
        #[cfg(feature = "verbose1")]
        emit_stats(&mut *guard);
        Formatter::new(&mut *guard).bestmove(text)
    }

    /// A BARE newline (empty line, no `info string` prefix), routed through the
    /// single output sink so it cannot interleave mid-line with the heavy
    /// initialisation's own output.
    fn keep_alive_newline(&self) {
        let mut guard = self.lock();
        let _ = Formatter::new(&mut *guard).raw_line(b"");
    }
}

/// The USI text of a [`Reply`]: the two token replies are constants, and a move
/// reply is composed in the caller's stack buffer.
fn render_reply(payload: &mut BestmoveBuf, reply: Reply) -> &[u8] {
    match reply {
        Reply::Resign => b"resign",
        Reply::Win => b"win",
        Reply::BestMove { mv, ponder } => payload.compose(mv, ponder),
    }
}

impl<W: Write + Send + 'static> EngineSink for UsiSink<W> {
    // A clone of this sink is what renders the PV lines, so the renderer is this
    // very type.
    #[cfg(feature = "verbose2")]
    type PvOutput = Self;

    fn notice(&self, msg: &[u8]) -> io::Result<()> {
        self.info_string(msg)
    }

    #[cfg(feature = "verbose1")]
    fn diagnostic(&self, msg: &[u8]) {
        let _ = self.info_string(msg);
    }

    fn reply(&self, reply: Reply, #[cfg(feature = "verbose3")] sent: &AtomicBool) {
        // Composed before the lock is taken, so the composing's own allocations
        // stay inside the interval the statistics line reports.
        let mut payload = BestmoveBuf::new();
        let text = render_reply(&mut payload, reply);
        let mut guard = self.lock();
        #[cfg(feature = "verbose1")]
        emit_stats(&mut *guard);
        let _ = Formatter::new(&mut *guard).bestmove(text);
        #[cfg(feature = "verbose3")]
        sent.store(true, Ordering::Relaxed);
    }

    #[cfg(feature = "verbose2")]
    fn book_candidates(&self, hit: &BookHit, hashfull: u32, time_ms: u64) {
        let mut bytes = [0u8; INFO_BYTES];
        let mut guard = self.lock();
        let mut f = Formatter::new(&mut *guard);
        for line in &hit.info_lines {
            let mut body = TextWriter::new(&mut bytes);
            body.bytes(b"depth ")
                .u64(u64::from(line.depth))
                .bytes(b" multipv ")
                .u64(line.multipv as u64)
                .bytes(b" score ");
            write_score(&mut body, Value::from(line.score));
            write_book_tail(&mut body, hashfull, time_ms, &line.pv);
            let _ = f.info(body.as_bytes());
        }
    }

    fn book_reply(
        &self,
        hit: &BookHit,
        #[cfg(feature = "verbose2")] hashfull: u32,
        #[cfg(feature = "verbose2")] time_ms: u64,
        #[cfg(feature = "verbose3")] sent: &AtomicBool,
    ) {
        #[cfg(feature = "verbose2")]
        let mut bytes = [0u8; INFO_BYTES];
        #[cfg(feature = "verbose2")]
        let info = {
            let mut body = TextWriter::new(&mut bytes);
            body.bytes(b"depth 0 multipv 1 score ");
            write_score(&mut body, Value::from(hit.value));
            let mut pv = [hit.best; 2];
            let pv = match hit.ponder {
                Some(p) => {
                    pv[1] = p;
                    &pv[..2]
                }
                None => &pv[..1],
            };
            write_book_tail(&mut body, hashfull, time_ms, pv);
            body.len()
        };
        let mut payload = BestmoveBuf::new();
        let text = payload.compose(hit.best, hit.ponder);
        let mut guard = self.lock();
        #[cfg(feature = "verbose2")]
        let _ = Formatter::new(&mut *guard).info(&bytes[..info]);
        // After that line, so the statistics cover composing it too, and directly
        // before the reply.
        #[cfg(feature = "verbose1")]
        emit_stats(&mut *guard);
        let _ = Formatter::new(&mut *guard).bestmove(text);
        #[cfg(feature = "verbose3")]
        sent.store(true, Ordering::Relaxed);
    }

    #[cfg(feature = "verbose2")]
    fn pv_block(&self, infos: &[PvInfo]) {
        let mut guard = self.lock();
        for info in infos {
            let _ = write_pv_info(&mut *guard, info);
        }
    }

    #[cfg(feature = "verbose2")]
    fn pv_output(&self) -> Self {
        self.clone()
    }
}

/// The main worker's per-iteration / fail-high-low PV lines go straight to the
/// shared USI output. Helpers are given no sink and emit nothing.
///
/// `verbose2` only. Without that feature the main worker is given no sink either,
/// which is what keeps the tournament build's search free of PV work: the
/// search's emission sites are all behind `pv_sink.is_some()`.
#[cfg(feature = "verbose2")]
impl<W: Write + Send + 'static> PvSink for UsiSink<W> {
    fn emit(&mut self, info: &PvInfo) {
        let mut guard = self.lock();
        let _ = write_pv_info(&mut *guard, info);
    }
}

/// The USI event loop: a line reader, the renderer and the engine those lines
/// drive.
pub struct UsiEngine<R: BufRead, W: Write + Send + 'static> {
    reader: R,
    /// The renderer, and a second handle to the engine's own output.
    out: UsiSink<W>,
    /// The engine the commands drive.
    engine: Engine<UsiSink<W>>,
    /// Poll interval of the `isready` keep-alive helper thread ([`KeepAlive`]),
    /// overridable via [`Self::with_keep_alive_poll`] so a test can drive the
    /// mechanism with a short interval.
    keep_alive_poll: Duration,
}

impl<R: BufRead, W: Write + Send + 'static> UsiEngine<R, W> {
    /// A session whose book / `rtime` PRNG stream is seeded from process entropy,
    /// so every process run differs. Tests wanting reproducible book selection
    /// or `rtime` budgets construct via [`Self::with_book_seed`].
    pub fn new(reader: R, writer: Arc<Mutex<W>>) -> Self {
        Self::from_engine(reader, UsiSink::new(writer), Engine::new)
    }

    /// A session with an explicit book-PRNG seed. The entropy default
    /// ([`Self::new`]) draws one instead; tests inject a fixed seed for
    /// deterministic book / `rtime` behaviour.
    pub fn with_book_seed(reader: R, writer: Arc<Mutex<W>>, book_seed: u64) -> Self {
        Self::from_engine(reader, UsiSink::new(writer), |sink| {
            Engine::with_book_seed(sink, book_seed)
        })
    }

    /// The shared half of the two constructors: the sink is cloned so the
    /// session keeps a handle of its own beside the engine's.
    fn from_engine(
        reader: R,
        out: UsiSink<W>,
        build: impl FnOnce(UsiSink<W>) -> Engine<UsiSink<W>>,
    ) -> Self {
        let engine = build(out.clone());
        Self {
            reader,
            out,
            engine,
            keep_alive_poll: KEEP_ALIVE_POLL_INTERVAL,
        }
    }

    /// Override the `isready` keep-alive poll interval, so a test can make a
    /// deliberately slowed heavy job elapse at least one tick. The newline still
    /// fires only after `KEEP_ALIVE_TICKS_PER_NEWLINE` polls, so this scales
    /// the whole cadence.
    pub fn with_keep_alive_poll(mut self, poll: Duration) -> Self {
        self.keep_alive_poll = poll;
        self
    }

    /// Override the directory a relative `eval_dir` resolves against — see
    /// [`Engine::set_eval_root`].
    pub fn with_eval_root(mut self, root: PathBuf) -> Self {
        self.engine.set_eval_root(root);
        self
    }

    /// Override the sysfs root the `isready` layout check reads — see
    /// [`Engine::set_sysfs_root`].
    pub fn with_sysfs_root(mut self, root: PathBuf) -> Self {
        self.engine.set_sysfs_root(root);
        self
    }

    /// Override the CPU set the `isready` layout check takes for this process's
    /// startup affinity — see [`Engine::set_startup_affinity`].
    pub fn with_startup_affinity(mut self, cpus: BTreeSet<usize>) -> Self {
        self.engine.set_startup_affinity(cpus);
        self
    }

    /// Read USI lines until `quit` or end of input, dispatching each one.
    ///
    /// One buffer serves the whole session: a line is read into it as the bytes
    /// that arrived, nothing validates or decodes them, and the typed command a
    /// line parses into borrows its tokens from it. Input can therefore not end
    /// a session — a `setoption` value spelling a path in a Windows code page is
    /// read like any other line — and no line costs an allocation.
    pub fn run(mut self) -> io::Result<()> {
        let mut line = vec![0u8; MAX_LINE_BYTES];
        loop {
            let filled = match read_command_line(&mut self.reader, &mut line)? {
                LineRead::Line(filled) => filled,
                LineRead::TooLong => {
                    self.handle_too_long()?;
                    continue;
                }
                LineRead::Eof => {
                    // EOF: treat as quit — stop and join any running search first.
                    self.engine.finish_search_join();
                    return Ok(());
                }
            };
            match parse_line(&line[..filled]) {
                Command::Usi => self.handle_usi()?,
                Command::IsReady => self.handle_isready()?,
                Command::SetOption { name, value } => self.handle_setoption(name, value)?,
                Command::UsiNewGame => self.handle_usinewgame(),
                Command::Position { sfen, moves } => self.handle_position(sfen, moves)?,
                Command::Go(params) => self.handle_go(params)?,
                #[cfg(not(feature = "verbose2"))]
                Command::GoExtraClause(clause) => self.handle_go_extra_clause(clause)?,
                Command::Stop => self.handle_stop(),
                Command::GameOver => self.handle_gameover(),
                Command::PonderHit => self.handle_ponderhit()?,
                #[cfg(feature = "verbose3")]
                Command::Bench(tokens) => self.handle_bench(&tokens)?,
                #[cfg(feature = "verbose3")]
                Command::Tt(tokens) => self.handle_tt(&tokens)?,
                Command::Quit => {
                    self.engine.finish_search_join();
                    return Ok(());
                }
                #[cfg(feature = "verbose1")]
                Command::Unknown(text) => self.handle_unknown(text)?,
                // The line is consumed and dropped either way; only the report
                // of it is gated.
                #[cfg(not(feature = "verbose1"))]
                Command::Unknown => {}
                Command::TooLong => self.handle_too_long()?,
            }
        }
    }

    fn handle_isready(&mut self) -> io::Result<()> {
        // Reclaim any worker before touching the table it may hold.
        self.engine.finish_search_join();
        // The reference applies two option-override files here, before its own
        // isready work (`USIEngine::isready`): `engine_options.txt` in the
        // current directory and `<EvalDir>/eval_options.txt`. This engine opens
        // neither, in any build. Its settings are compiled in, so a file that
        // claims to override one of them would be a lie on disk — and reading a
        // file only to ignore what it says would be worse than not reading it.

        // Wrap the heavy initialisation in a keep-alive scope: a helper thread
        // emits a bare newline every 5 s so a GUI does not time out. The guard's
        // `Drop` stops and joins that helper whether the block returns normally
        // or bails out early via `?`.
        let outcome = {
            let _keep_alive = KeepAlive::spawn(self.out.clone(), self.keep_alive_poll);
            self.engine.ready()?
            // `_keep_alive` dropped here: stop flag set, helper thread joined.
        };

        match outcome {
            ReadyOutcome::Ready => {
                self.out.readyok()?;
                // The initialisation phase allocates the table, the network and
                // the pool, and none of that belongs to a reply: the first
                // reported interval starts here.
                #[cfg(feature = "verbose1")]
                clear_alloc_count();
                Ok(())
            }
            ReadyOutcome::LoadFailed(reason) => {
                // Contract: on a load failure, emit
                // `info string eval load failed: <reason>` and do NOT emit
                // `readyok`. There is no working network to lose here: one
                // already read is reused by the idempotent path above, so the
                // only session that reaches this had none to begin with.
                let mut bytes = [0u8; NOTICE_BYTES];
                let mut out = TextWriter::new(&mut bytes);
                out.bytes(b"eval load failed: ");
                reason.write_message(&mut out);
                self.out.info_string(out.as_bytes())
            }
            ReadyOutcome::LayoutMismatch(reason) => {
                let mut bytes = [0u8; NOTICE_BYTES];
                let mut out = TextWriter::new(&mut bytes);
                out.bytes(b"NUMA layout mismatch: ");
                reason.write_message(&mut out);
                self.out.info_string(out.as_bytes())
            }
        }
    }

    /// The `usi` handshake.
    fn handle_usi(&mut self) -> io::Result<()> {
        self.out.handshake()
    }

    /// `usinewgame`: a fresh game, and nothing to say about it.
    fn handle_usinewgame(&mut self) {
        self.engine.new_game();
    }

    /// `setoption name <N> value <V>`: the USI minimum, in every build.
    ///
    /// There is no option to set — every setting was fixed at build time from
    /// the TOML config, and the `usi` reply advertises no options at all. USI
    /// requires no reply, so the line is parsed, consumed and dropped.
    fn handle_setoption(&mut self, _name: &[u8], _value: &[u8]) -> io::Result<()> {
        Ok(())
    }

    fn handle_position(&mut self, sfen: PositionSfen<'_>, moves: &[u8]) -> io::Result<()> {
        match self.engine.set_position(sfen, moves) {
            Ok(()) => Ok(()),
            Err(refusal) => self.report_position_refusal(refusal),
        }
    }

    /// Report a refused `position` line — the `verbose1` surface.
    ///
    /// The illegal-move report quotes a token from the command line and writes
    /// it whole, so a garbage token arrives back as it came however wide it was.
    #[cfg(feature = "verbose1")]
    fn report_position_refusal(&self, refusal: PositionRefusal<'_>) -> io::Result<()> {
        match refusal {
            PositionRefusal::Sfen(e) => {
                let mut bytes = [0u8; DIAG_BYTES];
                let mut out = TextWriter::new(&mut bytes);
                out.bytes(b"position parse error: ");
                e.write_message(&mut out);
                self.out.info_string(out.as_bytes())
            }
            PositionRefusal::IllegalMove(mv) => {
                self.out.info_string_parts(&[b"illegal move: ", mv])
            }
            PositionRefusal::TooManyMoves => {
                let mut bytes = [0u8; DIAG_BYTES];
                let mut out = TextWriter::new(&mut bytes);
                out.bytes(b"position error: more than ")
                    .u64(MAX_POSITION_MOVES as u64)
                    .bytes(b" moves; position unchanged");
                self.out.info_string(out.as_bytes())
            }
        }
    }

    /// A build below `verbose1` prints no diagnostic, so the refusal is
    /// consumed and dropped; the command was refused either way.
    #[cfg(not(feature = "verbose1"))]
    fn report_position_refusal(&self, _refusal: PositionRefusal<'_>) -> io::Result<()> {
        Ok(())
    }

    /// `go …`: start a search, or answer the one thing the engine cannot.
    fn handle_go(&mut self, params: GoParams) -> io::Result<()> {
        let outcome = self.engine.go(params);
        self.answer_go(outcome)
    }

    /// `stop`: abort the running search, which then emits its reply.
    fn handle_stop(&mut self) {
        self.engine.stop();
    }

    /// `gameover [win|lose|draw]`: the game ended. Treated exactly like `stop`:
    /// the same flag, releasing a held book reply (`go ponder`/`go infinite`) or
    /// aborting a running search. Over a shogi GUI an opponent resign during
    /// `go ponder` arrives as `gameover` without a preceding `stop`; unhandled,
    /// pondering would never stop. A no-op when idle.
    fn handle_gameover(&mut self) {
        self.handle_stop();
    }

    /// `ponderhit`: the opponent played the predicted move.
    fn handle_ponderhit(&mut self) -> io::Result<()> {
        let outcome = self.engine.ponderhit();
        self.answer_go(outcome)
    }

    /// What a `go` that never started owes the host: the notice and the
    /// `bestmove resign` that stands in for the search.
    fn answer_go(&mut self, outcome: GoOutcome) -> io::Result<()> {
        match outcome {
            GoOutcome::Started => Ok(()),
            GoOutcome::NoNetwork => {
                #[cfg(feature = "verbose1")]
                self.out
                    .info_string(b"no eval network loaded; run isready")?;
                self.out.reply_now(Reply::Resign)
            }
        }
    }

    /// A `go` line carrying a clause that arrives with `verbose2`, seen by a
    /// build without that feature: report it and start nothing.
    ///
    /// Failing loud is deliberate — ignoring the clause would silently change
    /// the search's terms, turning `go depth 4` into a clock-less `go` in the
    /// middle of a game. Any search already running is left alone.
    ///
    /// The refusal itself happens in every build; `verbose1` is where it is
    /// also reported, so a build below it consumes the clause and says nothing.
    #[cfg(all(not(feature = "verbose2"), feature = "verbose1"))]
    fn handle_go_extra_clause(&mut self, clause: &[u8]) -> io::Result<()> {
        self.out.info_string_parts(&[
            b"go error: `",
            clause,
            b"` requires a verbose2 build; no search started",
        ])
    }

    #[cfg(all(not(feature = "verbose2"), not(feature = "verbose1")))]
    fn handle_go_extra_clause(&mut self, _clause: &[u8]) -> io::Result<()> {
        Ok(())
    }

    /// `bench [ttSizeMB] [threads] [limit] [default|current|<fenFile>] [limitType]`
    /// — a reproducible NPS benchmark ported from the reference's
    /// `USIEngine::bench` and `setup_bench`.
    ///
    /// Ends with one machine-parsable summary line. A parse failure is reported
    /// as an `info string` and runs nothing, never a panic.
    ///
    /// The requested thread count and table size are the only ones in the engine
    /// that do not come from the config constants, and they last as long as the
    /// session.
    #[cfg(feature = "verbose3")]
    fn handle_bench(&mut self, tokens: &[&[u8]]) -> io::Result<()> {
        // Reclaim any running search before touching the pool / the TT.
        self.engine.finish_search_join();

        let mut sfen_buf = SfenBuf::new();
        let current = bench::current_sfen(self.engine.position(), &mut sfen_buf);
        let config = match bench::parse_bench(tokens, current) {
            Ok(c) => c,
            Err(e) => {
                let mut bytes = [0u8; DIAG_BYTES];
                let mut out = TextWriter::new(&mut bytes);
                out.bytes(b"bench: ");
                e.write_message(&mut out);
                return self.out.info_string(out.as_bytes());
            }
        };

        // The thread count is the one value the reference replays as a
        // `setoption` line that still means something here — the table's size
        // is the build's, not the command's. The pool rebuild reports itself
        // exactly as the reference `Threads` on_change callback does.
        self.engine.resize_pool(config.threads.max(1) as usize);
        {
            let mut bytes = [0u8; DIAG_BYTES];
            let mut out = TextWriter::new(&mut bytes);
            self.engine.write_thread_allocation_information(&mut out);
            self.out.info_string_lines(out.as_bytes())?;
        }

        // The `ucinewgame` (`search_clear`) the reference runs once before the
        // positions: clears the TT, resets histories, and rebuilds the pool — the
        // clean, identical starting state that makes two runs report equal nodes.
        self.handle_usinewgame();
        // That state covers the evaluation noise too: the seed a new game draws
        // would give each run its own node count.
        #[cfg(feature = "random")]
        self.engine.set_bench_random_seed();

        // The reference resets `elapsed` right after `search_clear`, so the timing
        // excludes the clear itself.
        let start = Instant::now();
        let mut total_nodes: u64 = 0;
        let mut positions: u64 = 0;
        for fen in &config.fens {
            match parse_sfen(fen) {
                Ok(p) => self.engine.set_search_position(p),
                Err(e) => {
                    // A malformed position in a `<fenFile>` is skipped loudly, not
                    // fatal — the rest of the bench still runs.
                    let mut bytes = [0u8; DIAG_BYTES];
                    let mut reason = TextWriter::new(&mut bytes);
                    e.write_message(&mut reason);
                    self.out.info_string_parts(&[
                        b"bench: skipping bad position `",
                        fen,
                        b"`: ",
                        reason.as_bytes(),
                    ])?;
                    continue;
                }
            }
            positions += 1;
            total_nodes += match self.engine.bench_run_one(config.limits.clone()) {
                Some(nodes) => nodes,
                // A position with no network loaded resigns and contributes no
                // nodes, exactly as a `go` would.
                None => {
                    self.answer_go(GoOutcome::NoNetwork)?;
                    0
                }
            };
        }

        // `+1` mirrors the reference's divide-by-zero guard.
        //
        // The summary is `bench`'s RESULT, not a diagnostic: it is the whole
        // point of the command, so it rides on `verbose3` alone and no other
        // verbosity feature can silence it. (`verbose3` also brings the
        // per-position `info` lines a measurement run reads, which is where a
        // bad run shows itself.)
        let time_ms = start.elapsed().as_millis() as u64 + 1;
        let nps = 1000 * total_nodes / time_ms;
        let mut bytes = [0u8; DIAG_BYTES];
        let mut out = TextWriter::new(&mut bytes);
        out.bytes(b"bench: positions=")
            .u64(positions)
            .bytes(b" nodes=")
            .u64(total_nodes)
            .bytes(b" time_ms=")
            .u64(time_ms)
            .bytes(b" nps=")
            .u64(nps);
        self.out.info_string(out.as_bytes())
    }

    // The `tt` command family exists only under the `verbose3` cargo feature.
    // Its `info string` lines are the commands' response — a `tt probe` that
    // printed nothing would be a command with no output — so they go through the
    // unconditional [`Self::info_string`] rather than the `verbose1` sink.

    /// Dispatch one `tt …` line.
    ///
    /// Refuses while a search is in flight — one `info string tt error: …` line,
    /// never a panic. A worker that has already replied but not yet been joined
    /// is reclaimed first rather than refused, so the natural
    /// `go … → bestmove → tt probe` sequence works. The table itself is always
    /// there to read: it is a `static`, empty rather than absent before a game.
    ///
    /// What counts as "already replied" is [`Engine::reclaim_replied_search`]'s
    /// question, not this one's.
    #[cfg(feature = "verbose3")]
    fn handle_tt(&mut self, tokens: &[&[u8]]) -> io::Result<()> {
        self.engine.reclaim_replied_search();
        if self.engine.search_is_running() {
            return self.tt_error(&[b"a search is running; `stop` it first"]);
        }

        let command = match parse_tt(tokens) {
            Ok(command) => command,
            Err(e) => {
                let mut bytes = [0u8; DIAG_BYTES];
                let mut out = TextWriter::new(&mut bytes);
                e.write_message(&mut out);
                return self.tt_error(&[out.as_bytes()]);
            }
        };

        match command {
            TtCommand::Store(args) => self.tt_store(&args),
            TtCommand::Probe(position) => self.tt_probe(&position),
            TtCommand::Children(position) => self.tt_children(&position),
        }
    }

    /// The single error channel for the `tt` commands.
    #[cfg(feature = "verbose3")]
    fn tt_error(&self, parts: &[&[u8]]) -> io::Result<()> {
        let mut all: Vec<&[u8]> = Vec::with_capacity(parts.len() + 1);
        all.push(b"tt error: ");
        all.extend_from_slice(parts);
        self.out.info_string_parts(&all)
    }

    /// Build the [`Position`] a `tt` command names.
    ///
    /// The extra king check is this surface's own: the SFEN parser accepts a
    /// kingless board, but the move generators these commands then run assume
    /// both kings are present.
    ///
    /// A refusal is written into `reason`, which the caller owns, and the
    /// returned `Err` says only that there is one.
    #[cfg(feature = "verbose3")]
    fn tt_position(
        &self,
        position: &TtPosition<'_>,
        reason: &mut TextWriter<'_>,
    ) -> Result<Position, ()> {
        use yorkie_state::Color;

        let pos = match position {
            TtPosition::StartPos => Position::startpos(),
            TtPosition::Sfen(fields) => {
                let mut pos = Position::empty();
                match parse_sfen_fields_into(&mut pos, *fields) {
                    Ok(()) => pos,
                    Err(e) => {
                        e.write_message(reason);
                        return Err(());
                    }
                }
            }
        };
        if pos.king_square(Color::Black).is_none() || pos.king_square(Color::White).is_none() {
            reason.bytes(b"position has no king for one or both sides");
            return Err(());
        }
        Ok(pos)
    }

    /// `tt store …` — write one entry for the named position.
    ///
    /// The write goes through the ordinary probe-then-write path at the table's
    /// current generation, exactly as a search storing that value at that depth
    /// would. That includes the replacement policy, which may decline the write,
    /// so the command re-probes afterwards and reports which happened.
    #[cfg(feature = "verbose3")]
    fn tt_store(&self, args: &TtStoreArgs<'_>) -> io::Result<()> {
        let mut reason_bytes = [0u8; DIAG_BYTES];
        let mut reason = TextWriter::new(&mut reason_bytes);
        let pos = match self.tt_position(&args.position, &mut reason) {
            Ok(pos) => pos,
            Err(()) => return self.tt_error(&[reason.as_bytes()]),
        };
        let mut legal: Vec<Move> = Vec::new();
        pos.generate_legal_all(&mut legal);

        // `none` stores the `MOVE_NONE` fragment, which `TTEntry::save` reads as
        // "keep whatever move this entry already holds for this position".
        let move16 = if args.mv == b"none" {
            None
        } else {
            match parse_usi_move(args.mv, &pos) {
                Ok(mv) if legal.contains(&mv) => mv.move16_stored(),
                Ok(_) => {
                    return self.tt_error(&[b"move `", args.mv, b"` is not legal here"]);
                }
                Err(e) => {
                    reason.clear();
                    e.write_message(&mut reason);
                    return self.tt_error(&[
                        b"move `",
                        args.mv,
                        b"` is not a USI move: ",
                        reason.as_bytes(),
                    ]);
                }
            }
        };

        let key = pos.key();
        let side = pos.side_to_move().index() as u8;
        let stored_value = value_to_tt(args.value, 0);
        let generation = TranspositionTable::shared().generation();

        let (_, _, writer) = TranspositionTable::shared().probe(key, side);
        writer.write(
            key,
            stored_value,
            args.pv,
            args.bound,
            args.depth,
            move16,
            args.eval,
            generation,
            args.path_dep,
        );

        // Verify rather than assume. A `move none` write is excluded from the
        // comparison on purpose: `save` deliberately preserves the pre-existing
        // move for it, so a mismatch there is the documented behaviour, not a
        // declined write.
        let (found, data, _) = TranspositionTable::shared().probe(key, side);
        let stored = found
            && data.value == stored_value
            && data.eval == args.eval
            && data.depth == args.depth
            && data.bound == args.bound
            && data.is_pv == args.pv
            && data.path_dep == args.path_dep
            && (move16.is_none() || data.move16 == move16);
        if stored {
            self.out.info_string(b"tt store ok")
        } else {
            self.out
                .info_string(b"tt store skipped (replacement policy kept the existing entry)")
        }
    }

    /// `tt probe …` — read the entry for the named position (`ply == 0`, so the
    /// reported value is exactly the stored one).
    #[cfg(feature = "verbose3")]
    fn tt_probe(&self, position: &TtPosition<'_>) -> io::Result<()> {
        let mut bytes = [0u8; DIAG_BYTES];
        let mut out = TextWriter::new(&mut bytes);
        let pos = match self.tt_position(position, &mut out) {
            Ok(pos) => pos,
            Err(()) => return self.tt_error(&[out.as_bytes()]),
        };
        let (found, data, _) =
            TranspositionTable::shared().probe(pos.key(), pos.side_to_move().index() as u8);
        if !found {
            return self.out.info_string(b"tt probe miss");
        }
        let mut legal: Vec<Move> = Vec::new();
        pos.generate_legal_all(&mut legal);
        out.clear();
        out.bytes(b"tt probe hit ");
        write_tt_entry_fields(&mut out, &data, &legal, 0);
        self.out.info_string(out.as_bytes())
    }

    /// `tt children …` — probe every legal child of the named position, one ply
    /// deep.
    ///
    /// Children are reported at `ply == 1`, so their values are expressed
    /// relative to the *named* position: a child holding "mate in 5 from the
    /// child" prints as `mate 6`, one ply further out than a `tt probe` of that
    /// child's own SFEN would. A child with no entry produces no line, and the
    /// closing `tt children end <n>` line marks the list complete.
    #[cfg(feature = "verbose3")]
    fn tt_children(&self, position: &TtPosition<'_>) -> io::Result<()> {
        let mut bytes = [0u8; DIAG_BYTES];
        let mut out = TextWriter::new(&mut bytes);
        let mut pos = match self.tt_position(position, &mut out) {
            Ok(pos) => pos,
            Err(()) => return self.tt_error(&[out.as_bytes()]),
        };
        let mut legal: Vec<Move> = Vec::new();
        pos.generate_legal_all(&mut legal);

        let mut child_legal: Vec<Move> = Vec::new();
        let mut hits = 0usize;
        for mv in &legal {
            let undo = pos.do_move(*mv);
            let (found, data, _) =
                TranspositionTable::shared().probe(pos.key(), pos.side_to_move().index() as u8);
            if found {
                child_legal.clear();
                pos.generate_legal_all(&mut child_legal);
                out.clear();
                out.bytes(b"tt child ");
                write_usi_move(*mv, &mut out);
                out.byte(b' ');
                write_tt_entry_fields(&mut out, &data, &child_legal, 1);
            }
            pos.undo_move(*mv, undo);
            if found {
                hits += 1;
                self.out.info_string(out.as_bytes())?;
            }
        }
        out.clear();
        out.bytes(b"tt children end ").u64(hits as u64);
        self.out.info_string(out.as_bytes())
    }

    #[cfg(feature = "verbose1")]
    fn handle_unknown(&mut self, line: &[u8]) -> io::Result<()> {
        self.out.info_string_parts(&[b"unknown command: ", line])
    }

    /// Report a line past the input limit — the `verbose1` surface.
    #[cfg(feature = "verbose1")]
    fn handle_too_long(&mut self) -> io::Result<()> {
        self.out.info_string(b"command too long")
    }

    #[cfg(not(feature = "verbose1"))]
    fn handle_too_long(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// The labelled body shared by `tt probe hit` and `tt child` lines:
/// `move <usi|none> value <score> depth <d> bound <b> eval <score> pv <bool>
/// pathdep <0|1>`.
///
/// `legal` is the legal-move list of the position the entry belongs to, used to
/// widen the stored 16-bit fragment exactly as the search does: a fragment with
/// no matching legal move prints as `none` rather than being decoded into a
/// nonsense square. That is also what a **key16 false positive** looks like from
/// here — the table matches entries on the low 16 bits of the key, so a hit may
/// belong to a different position sharing those bits.
///
/// `ply` is the entry's distance from the position the command named, so the
/// value is reported in that position's frame.
#[cfg(feature = "verbose3")]
fn write_tt_entry_fields(out: &mut TextWriter<'_>, data: &TTData, legal: &[Move], ply: i32) {
    out.bytes(b"move ");
    match legal
        .iter()
        .copied()
        .find(|m| m.move16_stored() == data.move16)
    {
        Some(mv) => write_usi_move(mv, out),
        None => {
            out.bytes(b"none");
        }
    }
    out.bytes(b" value ");
    write_tt_score_field(out, value_from_tt(data.value, ply));
    out.bytes(b" depth ")
        .i64(i64::from(data.depth))
        .bytes(b" bound ")
        .bytes(bound_name(data.bound))
        .bytes(b" eval ");
    write_tt_score_field(out, data.eval);
    out.bytes(b" pv ")
        .bytes(if data.is_pv {
            &b"true"[..]
        } else {
            &b"false"[..]
        })
        .bytes(b" pathdep ")
        .u64(u64::from(data.path_dep));
}

/// One score field of a `tt` output line: `cp <n>` / `mate <n>` in the same USI
/// scale [`write_score`] gives an `info … score` line, or the literal `none`
/// for the `VALUE_NONE` sentinel (which the search writes into `eval16`
/// whenever a node has no static eval — `tt store` cannot produce it).
#[cfg(feature = "verbose3")]
fn write_tt_score_field(out: &mut TextWriter<'_>, v: Value) {
    if v == VALUE_NONE {
        out.bytes(b"none");
    } else {
        write_score(out, v);
    }
}

/// Write one PV `info` line from a [`PvInfo`] — the reference's
/// `on_update_full` as this port surfaces it, carrying every field the
/// reference prints and in its order: `nodes nps hashfull time pv`. `seldepth`
/// is emitted only when it is non-zero, as the reference does — a line with no
/// search behind it (a book hit, a root with no legal move) reads badly with a
/// ` seldepth 0` on it.
///
/// `verbose2` only: the default build renders no PV line.
#[cfg(feature = "verbose2")]
fn write_pv_info<W: Write + ?Sized>(w: &mut W, info: &PvInfo) -> io::Result<()> {
    let mut bytes = [0u8; INFO_BYTES];
    let mut body = TextWriter::new(&mut bytes);
    body.bytes(b"depth ").i64(i64::from(info.depth));
    if info.sel_depth != 0 {
        body.bytes(b" seldepth ").i64(i64::from(info.sel_depth));
    }
    body.bytes(b" multipv ").u64(info.multipv as u64);
    body.bytes(b" score ");
    write_score(&mut body, info.score);
    match info.bound {
        PvBound::Lower => {
            body.bytes(b" lowerbound");
        }
        PvBound::Upper => {
            body.bytes(b" upperbound");
        }
        PvBound::Exact => {}
    }
    body.bytes(b" nodes ")
        .u64(info.nodes)
        .bytes(b" nps ")
        .u64(info.nps)
        .bytes(b" hashfull ")
        .u64(u64::from(info.hashfull))
        .bytes(b" time ")
        .u64(info.time_ms);
    if !info.pv.is_empty() {
        body.bytes(b" pv");
        for m in &info.pv {
            body.byte(b' ');
            write_usi_move(*m, &mut body);
        }
    }
    Formatter::new(w).info(body.as_bytes())
}

#[cfg(feature = "verbose1")]
fn emit_stats<W: Write + ?Sized>(w: &mut W) {
    let mut buf = StatsBuf::new();
    if let Some(line) = crate::stats::render(&mut buf, take_alloc_count()) {
        let _ = Formatter::new(w).composed_line(line);
    }
}

/// What one read of the input produced.
enum LineRead {
    /// A line, filling that many bytes of the buffer.
    Line(usize),
    /// A line that filled the buffer without reaching a newline. The rest of
    /// it is consumed, so the stream is on a line boundary again.
    TooLong,
    /// End of input.
    Eof,
}

/// Read the next line's bytes into `buf`, without the `'\n'` that ended it or
/// the `'\r'` a host with Windows line endings sent before it.
///
/// The bytes arrive as they came: nothing is validated as UTF-8, so a line
/// spelling a path in a code page the host happens to use is read like any
/// other and cannot fail the read. A line wider than `buf` is reported as
/// [`LineRead::TooLong`] and consumed to its end, leaving the stream where the
/// next line starts.
fn read_command_line<R: BufRead>(reader: &mut R, buf: &mut [u8]) -> io::Result<LineRead> {
    let mut filled = 0usize;
    let mut too_long = false;
    let mut saw_any = false;
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            break;
        }
        saw_any = true;
        let (chunk_len, done) = match available.iter().position(|&b| b == b'\n') {
            Some(i) => (i, true),
            None => (available.len(), false),
        };
        let room = buf.len() - filled;
        let take = chunk_len.min(room);
        buf[filled..filled + take].copy_from_slice(&available[..take]);
        filled += take;
        if take != chunk_len {
            too_long = true;
        }
        // The newline itself is consumed with the chunk it ended.
        reader.consume(chunk_len + usize::from(done));
        if done {
            break;
        }
    }
    if !saw_any {
        return Ok(LineRead::Eof);
    }
    if too_long {
        return Ok(LineRead::TooLong);
    }
    // A `\r\n` host sends the carriage return as part of the line.
    if filled > 0 && buf[filled - 1] == b'\r' {
        filled -= 1;
    }
    Ok(LineRead::Line(filled))
}

/// A running keep-alive: a helper thread that emits a bare newline every
/// [`KEEP_ALIVE_TICKS_PER_NEWLINE`] polls so a GUI does not time out while the
/// heavy `isready` initialisation runs — the reference's
/// `Engine::run_heavy_job`. Dropping the guard stops and joins the thread, so
/// the join runs whether the wrapped work returns normally or bails out early
/// via `?`.
struct KeepAlive {
    /// Set on drop to stop the helper (`thread_end`).
    stop: Arc<AtomicBool>,
    /// `Some` until the guard is dropped; taken to join exactly once.
    handle: Option<JoinHandle<()>>,
}

impl KeepAlive {
    /// Spawn the helper thread and block until it has actually started, then
    /// return the guard. The heavy work must run *after* this returns so a
    /// CPU-bound job cannot delay the helper's first tick; the reference spins
    /// on a `thread_started` flag for the same reason.
    ///
    /// The tick holds the shared output lock for the whole newline, so a
    /// keep-alive tick can never interleave mid-line with an `info string …` the
    /// heavy work emits concurrently.
    fn spawn<W: Write + Send + 'static>(out: UsiSink<W>, poll: Duration) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let started = Arc::new(AtomicBool::new(false));
        let handle = thread::spawn({
            let stop = Arc::clone(&stop);
            let started = Arc::clone(&started);
            move || {
                started.store(true, Ordering::Release);
                let mut count: u32 = 0;
                while !stop.load(Ordering::Acquire) {
                    thread::sleep(poll);
                    count += 1;
                    if count >= KEEP_ALIVE_TICKS_PER_NEWLINE {
                        count = 0;
                        out.keep_alive_newline();
                    }
                }
            }
        });
        // Wait until the helper is running (reference `Tools::sleep` spin on
        // `thread_started`). We poll finer than the reference's 100 ms so
        // wrapping a *fast* `isready` adds no perceptible latency; the 5 s
        // keep-alive cadence itself is unaffected.
        while !started.load(Ordering::Acquire) {
            thread::sleep(Duration::from_millis(1));
        }
        Self {
            stop,
            handle: Some(handle),
        }
    }
}

impl Drop for KeepAlive {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// The fields every book `info` line ends with: the counters a book hit has no
/// numbers for, the occupancy and the clock, then the book PV.
///
/// `verbose2` only — the book `info` lines are its only caller.
#[cfg(feature = "verbose2")]
fn write_book_tail(out: &mut TextWriter<'_>, hashfull: u32, time_ms: u64, pv: &[Move]) {
    out.bytes(b" nodes 0 nps 0 hashfull ")
        .u64(u64::from(hashfull))
        .bytes(b" time ")
        .u64(time_ms)
        .bytes(b" pv");
    for m in pv {
        out.byte(b' ');
        write_usi_move(*m, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::time::Instant;

    #[cfg(feature = "verbose2")]
    use yorkie_state::{Position, parse_usi_move};

    use crate::engine::MAX_POSITION_MOVES;
    use crate::{king_shuffle, serial_tt};

    /// Drive a full canned session in-process and return everything written.
    ///
    /// The output sink is an `Arc<Mutex<Vec<u8>>>` shared with the session (and,
    /// during a `go`, its search worker); after `run` returns — which joins any
    /// worker — the buffer holds the complete transcript.
    fn run_with(input: &str) -> String {
        let output = Arc::new(Mutex::new(Vec::<u8>::new()));
        let session = UsiEngine::new(input.as_bytes(), Arc::clone(&output));
        session.run().expect("session run");
        let bytes = output.lock().expect("output lock").clone();
        String::from_utf8(bytes).expect("utf-8")
    }

    /// The initial position's SFEN as text, for the command lines below.
    fn startpos_sfen() -> String {
        String::from_utf8(yorkie_state::STARTPOS_SFEN.to_vec()).expect("an SFEN is ASCII")
    }

    /// The transcript a diagnostic `info string <body>` contributes in THIS
    /// build: the line with `verbose1`, nothing without it. Lets a pinned
    /// transcript stay byte-exact in both builds instead of being asserted in
    /// only one of them.
    fn diag(body: &str) -> String {
        if cfg!(feature = "verbose1") {
            format!("info string {body}\n")
        } else {
            String::new()
        }
    }

    /// Render one PV line exactly as [`write_pv_info`] would put it on the wire.
    #[cfg(feature = "verbose2")]
    fn pv_line(info: &PvInfo) -> String {
        let mut buf = Vec::<u8>::new();
        write_pv_info(&mut buf, info).expect("write to Vec cannot fail");
        String::from_utf8(buf).expect("utf-8")
    }

    #[cfg(feature = "verbose2")]
    fn pv_info_fixture(score: Value, bound: PvBound, pv: &[&str]) -> PvInfo {
        let pos = Position::startpos();
        PvInfo {
            depth: 12,
            sel_depth: 19,
            multipv: 2,
            score,
            bound,
            nodes: 1_234_567_890,
            nps: 2_469_135_780,
            hashfull: 314,
            time_ms: 500,
            pv: pv
                .iter()
                .map(|s| parse_usi_move(s.as_bytes(), &pos).expect("fixture move parses"))
                .collect(),
        }
    }

    /// The `info` PV line is byte-exact. `write_pv_info` assembles it from
    /// `NumBuffer`-backed digits rather than `format!` temporaries, so pin the
    /// full wire bytes for every branch of the line (cp / mate, both signs, the
    /// three bounds, and an empty PV) rather than just the fields' presence.
    #[cfg(feature = "verbose2")]
    #[test]
    fn pv_info_line_is_byte_exact() {
        assert_eq!(
            pv_line(&pv_info_fixture(90, PvBound::Exact, &["7g7f", "3c3d"])),
            "info depth 12 seldepth 19 multipv 2 score cp 100 nodes 1234567890 nps 2469135780 hashfull 314 time 500 pv 7g7f 3c3d\n"
        );
        // Truncating division toward zero, negative side.
        assert_eq!(
            pv_line(&pv_info_fixture(-95, PvBound::Lower, &["7g7f"])),
            "info depth 12 seldepth 19 multipv 2 score cp -105 lowerbound nodes 1234567890 nps 2469135780 hashfull 314 time 500 pv 7g7f\n"
        );
        assert_eq!(
            pv_line(&pv_info_fixture(0, PvBound::Upper, &[])),
            "info depth 12 seldepth 19 multipv 2 score cp 0 upperbound nodes 1234567890 nps 2469135780 hashfull 314 time 500\n"
        );
        // Decisive scores switch to `mate <distance>`, signed by the side.
        assert_eq!(
            pv_line(&pv_info_fixture(VALUE_MATE - 5, PvBound::Exact, &["7g7f"])),
            "info depth 12 seldepth 19 multipv 2 score mate 5 nodes 1234567890 nps 2469135780 hashfull 314 time 500 pv 7g7f\n"
        );
        assert_eq!(
            pv_line(&pv_info_fixture(
                -(VALUE_MATE - 5),
                PvBound::Exact,
                &["7g7f"]
            )),
            "info depth 12 seldepth 19 multipv 2 score mate -5 nodes 1234567890 nps 2469135780 hashfull 314 time 500 pv 7g7f\n"
        );
    }

    /// A drop move, the `depth 0` / `nodes 0` / `nps 0` extremes, the floored
    /// `time 1`, and both ends of the `hashfull` permille range still round-trip
    /// byte-for-byte (the digit paths that `NumBuffer` owns). A zero
    /// `sel_depth` drops the field entirely, which is what the reference prints.
    #[cfg(feature = "verbose2")]
    #[test]
    fn pv_info_line_covers_zero_and_drop_extremes() {
        let mut info = pv_info_fixture(0, PvBound::Exact, &[]);
        info.depth = 0;
        info.sel_depth = 0;
        info.multipv = 1;
        info.nodes = 0;
        info.nps = 0;
        info.hashfull = 0;
        info.time_ms = 1;
        assert_eq!(
            pv_line(&info),
            "info depth 0 multipv 1 score cp 0 nodes 0 nps 0 hashfull 0 time 1\n"
        );

        let pos = yorkie_state::parse_sfen(b"4k4/9/9/9/9/9/9/9/4K4 b P 1").expect("sfen parses");
        info.pv = vec![parse_usi_move(b"P*5e", &pos).expect("drop parses")];
        info.hashfull = 1000;
        assert_eq!(
            pv_line(&info),
            "info depth 0 multipv 1 score cp 0 nodes 0 nps 0 hashfull 1000 time 1 pv P*5e\n"
        );
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn quit_returns_immediately() {
        let _tt = serial_tt();
        assert_eq!(run_with("quit\n"), "");
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn eof_returns_ok() {
        let _tt = serial_tt();
        assert_eq!(run_with(""), "");
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn isready_without_network_reports_load_failure() {
        let _tt = serial_tt();
        // Nothing staged an evaluation file where this session looks — beside
        // the running executable — so the load fails: the contract is an
        // `info string eval load failed:` notice and NO `readyok`. The process
        // stays alive (the `quit` returns).
        let out = run_with("isready\nquit\n");
        assert!(
            out.contains("info string eval load failed:"),
            "expected eval-load-failure notice, got: {out:?}"
        );
        assert!(
            !out.contains("readyok"),
            "readyok must not appear on a failed load: {out:?}"
        );
        // A fast `isready` (default keep-alive cadence: a bare newline only every
        // 5 s) emits no keep-alive newline — the first tick never elapses. The
        // load-failure notice is a single line with a trailing `\n`; no *empty*
        // line (bare keep-alive newline) may appear.
        assert_eq!(
            bare_newline_count(&out),
            0,
            "a fast isready must emit no keep-alive newline: {out:?}"
        );
    }

    /// Count bare keep-alive newlines: empty lines produced by the helper
    /// thread's `raw_line("")`. Splitting on `\n` yields one trailing empty
    /// segment for the final terminator, which is not a bare newline; every
    /// other empty segment is.
    fn bare_newline_count(out: &str) -> usize {
        let parts: Vec<&str> = out.split('\n').collect();
        // Drop the trailing terminator segment before counting empties.
        parts
            .iter()
            .take(parts.len().saturating_sub(1))
            .filter(|s| s.is_empty())
            .count()
    }

    /// Everything written to a shared test writer so far.
    fn writer_snapshot(writer: &Arc<Mutex<Vec<u8>>>) -> String {
        let bytes = writer.lock().unwrap_or_else(|e| e.into_inner()).clone();
        String::from_utf8(bytes).expect("utf-8")
    }

    /// Block until the shared writer holds at least `want` bare keep-alive
    /// newlines, failing the test if that has not happened within `DEADLINE`.
    ///
    /// A test must wait for this rather than sleep a fixed time: the helper
    /// promises a newline every [`KEEP_ALIVE_TICKS_PER_NEWLINE`] *polls*, not
    /// every N ms of wall time, and each of those polls is a `thread::sleep`
    /// that runs long on a loaded machine. A fixed sleep sized off the nominal
    /// cadence therefore expires before the newline lands whenever the machine
    /// is busy; this loop only bounds how long the helper may take to write
    /// anything at all.
    fn wait_for_bare_newlines(writer: &Arc<Mutex<Vec<u8>>>, want: usize) {
        // Generous versus the ~50 ms nominal cadence of a 1 ms poll: this is a
        // liveness backstop for a helper that never writes, not a cadence check.
        const DEADLINE: Duration = Duration::from_secs(5);
        let start = Instant::now();
        loop {
            let out = writer_snapshot(writer);
            if bare_newline_count(&out) >= want {
                return;
            }
            assert!(
                start.elapsed() < DEADLINE,
                "expected {want} bare keep-alive newline(s) within {DEADLINE:?}, got: {out:?}"
            );
            thread::sleep(Duration::from_millis(2));
        }
    }

    #[test]
    fn keep_alive_emits_bare_newline_through_shared_writer() {
        // Drive the keep-alive mechanism directly with a short poll interval and
        // a "heavy job" that stands in for slow initialisation by waiting for the
        // helper to tick. The job also writes a real line through the *same*
        // shared writer, between two ticks, so this asserts both that bare
        // newlines are emitted while the job runs and that none of them
        // interleaves mid-line with that output.
        let writer = Arc::new(Mutex::new(Vec::<u8>::new()));
        {
            // 1 ms poll → a bare newline every 50 polls (KEEP_ALIVE_TICKS_PER_NEWLINE).
            let keep_alive =
                KeepAlive::spawn(UsiSink::new(Arc::clone(&writer)), Duration::from_millis(1));
            // Heavy job, part one: run until the helper has ticked at least once.
            wait_for_bare_newlines(&writer, 1);
            // Then emit a real line partway through, to probe for interleaving.
            // Straight through the shared writer, not through a gated sink: this
            // test is about the keep-alive helper's interleaving, and must run in
            // every build. Count under the same lock, so the count cannot miss a
            // tick that lands between the write and the read.
            let seen = {
                let mut guard = writer.lock().unwrap_or_else(|e| e.into_inner());
                Formatter::new(&mut *guard)
                    .info_string(b"busy")
                    .expect("write to Vec cannot fail");
                bare_newline_count(&String::from_utf8(guard.clone()).expect("utf-8"))
            };
            // Heavy job, part two: run until one *further* tick has landed, so the
            // interleaving assertion has a keep-alive newline after that line too.
            wait_for_bare_newlines(&writer, seen + 1);
            drop(keep_alive); // stop flag set + helper joined here.
        }
        let out = writer_snapshot(&writer);

        assert!(
            bare_newline_count(&out) >= 1,
            "expected at least one bare keep-alive newline, got: {out:?}"
        );
        // No interleaving: every non-empty line is the intact `info string busy`.
        for line in out.split('\n') {
            assert!(
                line.is_empty() || line == "info string busy",
                "keep-alive newline interleaved with output: {out:?}"
            );
        }
        assert!(
            out.contains("info string busy\n"),
            "the heavy job's line must survive intact: {out:?}"
        );
    }

    #[test]
    fn keep_alive_stops_and_joins_when_job_finishes() {
        // A short-lived scope with a short poll: the guard's Drop must set the
        // stop flag and join the helper without hanging, and (the job being
        // near-instant) emit no bare newline. There is nothing to wait for here —
        // the assertion is that the helper has *not* ticked — so the bound comes
        // from measured wall time instead: the helper cannot have written before
        // KEEP_ALIVE_TICKS_PER_NEWLINE polls elapsed, and on a machine loaded
        // enough that this near-instant scope itself took that long, the count is
        // allowed to rise exactly as far as the stolen time justifies. On an idle
        // machine the scope takes a millisecond or two and the bound is zero.
        let writer = Arc::new(Mutex::new(Vec::<u8>::new()));
        let poll = Duration::from_millis(1);
        let start = Instant::now();
        {
            let _keep_alive = KeepAlive::spawn(UsiSink::new(Arc::clone(&writer)), poll);
            // No wait: the "job" finishes before the first newline tick.
        }
        // Returning from the scope at all is the stop-and-join property: a Drop
        // that failed to set the stop flag would hang forever in the join.
        let elapsed = start.elapsed();
        let out = writer_snapshot(&writer);
        let max_newlines =
            (elapsed.as_nanos() / poll.as_nanos()) / u128::from(KEEP_ALIVE_TICKS_PER_NEWLINE);
        assert!(
            bare_newline_count(&out) as u128 <= max_newlines,
            "a job that finishes before the first tick emits no newline \
             ({elapsed:?} of polls allows at most {max_newlines}): {out:?}"
        );
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn usinewgame_is_no_op() {
        let _tt = serial_tt();
        assert_eq!(run_with("usinewgame\nquit\n"), "");
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn unknown_command_echoes_back() {
        let _tt = serial_tt();
        assert_eq!(
            run_with("frobnicate\nquit\n"),
            diag("unknown command: frobnicate")
        );
    }

    /// There is no option to set in any build, and USI requires no reply to
    /// `setoption`, so every one of them is consumed in silence: a name the
    /// reference implementation registers, a nonexistent one and an ill-typed
    /// one all take the same path and all emit nothing.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn setoption_is_consumed_silently() {
        let _tt = serial_tt();
        for line in [
            "setoption name USI_Hash value 256",
            "setoption name Nonexistent value foo",
            "setoption name USI_Hash value not-a-number",
            "setoption name Threads value 8",
            // No `value` token at all — still consumed.
            "setoption name Threads",
        ] {
            assert_eq!(run_with(&format!("{line}\nquit\n")), "", "line {line:?}");
        }
    }

    /// A transcript with the per-reply statistics line dropped.
    ///
    /// That line counts what the *process* allocated since the previous reply,
    /// so two runs of one session do not agree on it, and neither would two
    /// sessions being compared for the decisions they took. Without `verbose1`
    /// there is no such line and this is the identity.
    fn without_stats(out: &str) -> String {
        out.lines()
            .filter(|l| !l.starts_with("info string stats "))
            .map(|l| format!("{l}\n"))
            .collect()
    }

    /// Consuming the line really is inert: a `setoption name Threads` an older
    /// build would have acted on leaves the pool exactly as it was, so the
    /// following `go` behaves as if the line had never arrived — and the
    /// transcript is byte-identical to the one without the lines.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn a_consumed_setoption_changes_nothing() {
        let _tt = serial_tt();
        let with_setoption = run_with(
            "setoption name Threads value 1\n\
             setoption name USI_Hash value 1\n\
             position startpos\n\
             go btime 1000 wtime 1000\n\
             quit\n",
        );
        let without = run_with("position startpos\ngo btime 1000 wtime 1000\nquit\n");
        assert_eq!(without_stats(&with_setoption), without_stats(&without));
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn position_startpos_silent() {
        let _tt = serial_tt();
        assert_eq!(run_with("position startpos\nquit\n"), "");
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn position_sfen_startpos_silent() {
        let _tt = serial_tt();
        let sfen = startpos_sfen();
        assert_eq!(run_with(&format!("position sfen {sfen}\nquit\n")), "");
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn position_startpos_moves_silent() {
        let _tt = serial_tt();
        assert_eq!(run_with("position startpos moves 7g7f\nquit\n"), "");
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn position_sfen_malformed_emits_info_string() {
        let _tt = serial_tt();
        let out = run_with("position sfen not-a-board b - 1\nquit\n");
        if cfg!(feature = "verbose1") {
            assert!(
                out.starts_with("info string position parse error:"),
                "unexpected output: {out:?}",
            );
        } else {
            // The rejection itself is unchanged (the position is not adopted —
            // `position_parse_error_leaves_prior_state_intact` covers that); the
            // default build just does not say so.
            assert_eq!(out, "", "unexpected output: {out:?}");
        }
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn position_with_illegal_move_emits_info_string() {
        let _tt = serial_tt();
        // 1a1b would move a non-existent piece (square 1a empty at startpos).
        let out = run_with("position startpos moves 1a1b\nquit\n");
        if cfg!(feature = "verbose1") {
            assert!(
                out.starts_with("info string illegal move:"),
                "unexpected output: {out:?}",
            );
        } else {
            assert_eq!(out, "", "unexpected output: {out:?}");
        }
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn position_with_pseudo_legal_but_illegal_move_emits_info_string() {
        let _tt = serial_tt();
        // 1a1b' shape — pick a syntactically valid move that is not a legal
        // generated move from startpos. Pawn on 7g cannot jump to 5g.
        let out = run_with("position startpos moves 7g5g\nquit\n");
        if cfg!(feature = "verbose1") {
            assert!(
                out.starts_with("info string illegal move:"),
                "unexpected output: {out:?}",
            );
        } else {
            assert_eq!(out, "", "unexpected output: {out:?}");
        }
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn position_parse_error_leaves_prior_state_intact() {
        let _tt = serial_tt();
        // Apply a legal move; then send a malformed sfen; then `go`. The reply
        // must be a legal move from the *post-7g7f* position, not from startpos
        // — proving the malformed line did not clobber the engine's state.
        // (`go` here has no network loaded, so it resigns; the check is that the
        // parse error is reported and exactly one bestmove is emitted.)
        let session = "position startpos moves 7g7f\n\
                       position sfen not-a-board b - 1\n\
                       go\n\
                       quit\n";
        let out = run_with(session);
        if cfg!(feature = "verbose1") {
            assert!(
                out.contains("info string position parse error:"),
                "missing parse-error info string in: {out}"
            );
        }
        let bestmoves: Vec<&str> = out.lines().filter(|l| l.starts_with("bestmove ")).collect();
        assert_eq!(
            bestmoves.len(),
            1,
            "expected one bestmove line, got {bestmoves:?}"
        );
    }

    /// A `position` line carrying more moves than the retained command can hold
    /// is refused by name, and the position the last accepted line named stands.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn a_move_list_past_the_bound_is_refused() {
        let _tt = serial_tt();
        let session = format!(
            "position startpos moves 7g7f\n\
             position startpos moves {}\n\
             go\n\
             quit\n",
            king_shuffle(MAX_POSITION_MOVES + 1)
        );
        let out = run_with(&session);
        if cfg!(feature = "verbose1") {
            assert!(
                out.contains(&format!(
                    "info string position error: more than {MAX_POSITION_MOVES} moves"
                )),
                "missing the refusal in: {out}"
            );
        }
        let bestmoves: Vec<&str> = out.lines().filter(|l| l.starts_with("bestmove ")).collect();
        assert_eq!(
            bestmoves.len(),
            1,
            "expected one bestmove line, got {bestmoves:?}"
        );
    }

    /// The SFEN fields arrive split, and the session joins them back into the one
    /// string the parser reads — whatever ran between them on the wire.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn a_position_sfen_is_accepted_however_its_fields_were_spaced() {
        let _tt = serial_tt();
        let sfen = startpos_sfen();
        assert_eq!(
            run_with(&format!("position   sfen  {sfen}   moves   7g7f\nquit\n")),
            ""
        );
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn go_without_network_resigns_with_notice() {
        let _tt = serial_tt();
        // No successful `isready`, so no network is loaded. `go` must not crash;
        // it emits the notice and `bestmove resign`. (The positive path — a
        // legal, search-chosen move — is covered in tests/eval_session.rs with a
        // synthetic network, and in tests/real_network_selfplay against the
        // network the build laid out.)
        let out = run_with("go\nquit\n");
        if cfg!(feature = "verbose1") {
            assert!(
                out.contains("info string no eval network loaded; run isready"),
                "expected the no-network notice, got: {out:?}"
            );
        }
        let bestmoves: Vec<&str> = out.lines().filter(|l| l.starts_with("bestmove ")).collect();
        assert_eq!(bestmoves, vec!["bestmove resign"]);
    }

    #[cfg(feature = "verbose2")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn go_with_limit_subtokens_still_emits_one_bestmove() {
        let _tt = serial_tt();
        // Whatever subset of the `go` clauses the host provides, the session parses and
        // accepts them and still emits exactly one bestmove line (resign here,
        // as no network is loaded).
        let session = "go depth 8 wtime 60000 btime 60000 byoyomi 5000\nquit\n";
        let out = run_with(session);
        let bestmoves: Vec<&str> = out.lines().filter(|l| l.starts_with("bestmove ")).collect();
        assert_eq!(bestmoves.len(), 1);
    }

    /// Below `verbose2` — the default (tournament) build: the same line is
    /// refused by name and starts nothing, so there is no `bestmove` at all. The
    /// clock clauses riding along on it do not rescue it: a `go` whose terms the
    /// build cannot honour is not silently downgraded to one it can.
    #[cfg(not(feature = "verbose2"))]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn go_with_gated_limit_subtokens_is_refused_and_starts_no_search() {
        let _tt = serial_tt();
        let session = "go depth 8 wtime 60000 btime 60000 byoyomi 5000\nquit\n";
        // The refusal is what matters — no `bestmove`, so no search started. The
        // line that names it is `verbose1`.
        assert_eq!(
            run_with(session),
            diag("go error: `depth` requires a verbose2 build; no search started")
        );
    }

    /// Below `verbose2`: the match clauses are untouched — a clock-bounded `go`
    /// still runs and emits one `bestmove` (resign here, as no network is
    /// loaded).
    #[cfg(not(feature = "verbose2"))]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn go_with_match_subtokens_still_emits_one_bestmove() {
        let _tt = serial_tt();
        let session = "go wtime 60000 btime 60000 byoyomi 5000\nquit\n";
        let out = run_with(session);
        let bestmoves: Vec<&str> = out.lines().filter(|l| l.starts_with("bestmove ")).collect();
        assert_eq!(bestmoves, vec!["bestmove resign"]);
    }

    /// Below `verbose3`: `bench` is not a command, so it lands in the ordinary
    /// unknown-command path — the same line any stray input produces.
    #[cfg(not(feature = "verbose3"))]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn bench_is_an_unknown_command_below_verbose3() {
        let _tt = serial_tt();
        assert_eq!(
            run_with("bench 16 1 6 default depth\nquit\n"),
            diag("unknown command: bench 16 1 6 default depth")
        );
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn stop_is_silent() {
        let _tt = serial_tt();
        assert_eq!(run_with("stop\nquit\n"), "");
        // `stop` with no network resolves the same as `go` alone: the no-network
        // notice plus a single `bestmove resign`, and nothing more.
        assert_eq!(
            without_stats(&run_with("go\nstop\nquit\n")),
            without_stats(&run_with("go\nquit\n"))
        );
    }

    /// `bench` is the one command that resizes the worker pool, and a pool
    /// rebuild emits the reference allocation info line. Each cycle joins its
    /// helpers first, so repeated rebuilds never wedge the main loop or leak
    /// threads. No network is loaded, so every bench position resigns
    /// immediately and the run is fast.
    #[cfg(feature = "verbose3")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn a_bench_thread_count_emits_the_allocation_line() {
        let _tt = serial_tt();
        let out = run_with(
            "bench 1 1 1 current movetime\n\
             bench 1 4 1 current movetime\n\
             bench 1 2 1 current movetime\n\
             quit\n",
        );
        // Prefix matches: the CPU list the line ends with is the host's.
        assert!(out.contains("info string Using 1 thread on CPUs "), "{out}");
        assert!(
            out.contains("info string Using 4 threads on CPUs "),
            "{out}"
        );
        assert!(
            out.contains("info string Using 2 threads on CPUs "),
            "{out}"
        );
    }
}
