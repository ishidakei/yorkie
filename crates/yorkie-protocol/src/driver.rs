// Only the score / PV renderers use it, and both are optional surfaces.
#[cfg(feature = "verbose2")]
use core::fmt::NumBuffer;
use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use yorkie_eval::{NnueError, network_file};
use yorkie_numa::{NumaConfig, NumaIndex, NumaLayout, mempolicy};
use yorkie_search::{
    BookConfig, BookHit, EnteringKingConfig, EnteringKingRule, PonderSignal, Prng, QSearch,
    RootMove, Search, SearchControl, SharedHistories, TimeControl, TimeInput, TimeManagement,
    WorkerHistories, WorkerResult, WorkerVote, declaration_win, generate_root_moves, probe_book,
    select_best_worker,
};
// The PV-line surface: only a `verbose2` build renders one, so only it needs
// the line's data type, its bound marker, the sink trait and the output config.
#[cfg(feature = "verbose2")]
use yorkie_search::{PvBound, PvInfo, PvOutputConfig, PvSink};
// The per-game evaluation-noise seed: drawn here, read only by the search.
#[cfg(feature = "random")]
use yorkie_search::new_game_seed;
use yorkie_state::{Move, Position, format_usi_move, parse_sfen, parse_usi_move};
use yorkie_storage::{Book, TranspositionTable, Value};
#[cfg(feature = "verbose3")]
use yorkie_storage::{TTData, VALUE_NONE};
// The per-reply allocation tally: raised by the counting global allocator this
// feature installs, and read by the statistics line that reports it.
#[cfg(feature = "verbose1")]
use yorkie_storage::{clear_alloc_count, take_alloc_count};

#[cfg(feature = "verbose3")]
use crate::bench;
use crate::formatter::Formatter;
#[cfg(feature = "verbose2")]
use crate::parser::MATE_UNLIMITED_MS;
use crate::parser::{Command, GoLimits, PositionSfen, parse_line};
#[cfg(feature = "random")]
use crate::settings::RANDOM_AMPLITUDE;
use crate::settings::Settings;
#[cfg(feature = "verbose1")]
use crate::stats::StatsBuf;
#[cfg(feature = "verbose3")]
use crate::tt_command::{
    TtCommand, TtPosition, TtStoreArgs, bound_name, parse_tt, value_from_tt, value_to_tt,
};

/// Emit one diagnostic `info string` line through a [`UsiDriver`] — the
/// `verbose1` surface — and compile the message only into a build that has that
/// surface.
///
/// A macro rather than a plain method call because the message text is part of
/// the surface: below `verbose1` the expansion drops the format string and the
/// formatting with it, keeping the text out of the binary, and yields the same
/// `Ok(())` so every call site's `?` / `return` shape is identical in both
/// builds. Arguments are still named once each, so a value computed only for the
/// message cannot be left behind as an unused binding.
///
/// Pass interpolated values as trailing arguments (`"illegal move: {}", s`)
/// rather than as inline captures at any call site a build below `verbose1`
/// still compiles: an inline capture is invisible to the expansion that drops
/// the message, so the binding it names would go unused there.
macro_rules! diag {
    ($driver:expr, $fmt:literal $(, $arg:expr)* $(,)?) => {{
        #[cfg(feature = "verbose1")]
        {
            $driver.info_string_diag(format_args!($fmt $(, $arg)*))
        }
        #[cfg(not(feature = "verbose1"))]
        {
            let _ = &$driver;
            $(let _ = &$arg;)*
            Ok::<(), io::Error>(())
        }
    }};
}

/// The public values used in the `id name` / `id author` lines.
///
/// The version part is this project's own generation number (see
/// `CHANGELOG.md`), not an upstream-tracking number; the upstream YaneuraOu
/// baseline is documented in `README.md` instead.
pub const ENGINE_NAME: &str = "Yorkie 3.1.0";
pub const ENGINE_AUTHOR: &str = "Kei Ishida <ishida.kei@gmail.com>";

// The transposition table is a `static` sized from the `usi_hash` config
// constant (the reference's `USI_Hash` option — the depth-1 fixture capture
// condition) when the binary is built, so no command and no reply can change
// how big it is. What `isready` still decides is where its pages live; see
// [`UsiDriver::place_transposition_table`].

/// The largest iterative-deepening depth a `go` ever requests. `run_root`'s own
/// `rootDepth + 1 < MAX_PLY` guard (`MAX_PLY == 246`) is the real ceiling; this
/// is the value passed for a time-/stop-bounded `go` (no explicit `depth`), and
/// the clamp for an out-of-range `go depth N`. It sits one below `MAX_PLY` so
/// the loop guard never has to truncate it.
const SEARCH_MAX_DEPTH: i32 = 245;

/// Where the running machine's NUMA layout is read from, for the one check that
/// reads it: `isready`'s, which holds the machine against the layout this binary
/// was built for.
const SYSFS_ROOT: &str = "/sys";

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

/// The evaluation-noise seed a `bench` runs under. What the command reports is a
/// node count two runs — and two processes — have to agree on, so the clean
/// starting state it builds for itself fixes the seed as well as the table and
/// the histories. A game draws its own.
#[cfg(all(feature = "verbose3", feature = "random"))]
const BENCH_RANDOM_SEED: u64 = 0;

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
/// `Eval::PawnValue` / `NormalizeToPawnValue`.
pub(crate) const PAWN_VALUE: Value = 90;
/// `VALUE_INFINITE`: the pre-search `rootMoves[0].score` sentinel the
/// `ResignValue` guard excludes.
const VALUE_INFINITE: Value = 32001;

/// The reference `USIEngine::to_cp`: `100 * v / NormalizeToPawnValue`, with
/// C++-style truncating division (Rust truncates toward zero, matching). Used
/// by the `ResignValue` check; unlike [`format_score`] it does not special-case
/// mate scores (the reference `to_cp` applies the same linear map to all
/// values).
fn to_cp(v: Value) -> Value {
    100 * v / PAWN_VALUE
}

/// Append a search value to `out` the way the reference USI layer formats it: a
/// mate distance for decisive scores, else centipawns.
///
/// Appending in place keeps the `info` PV path free of the `String` temporary
/// [`format_score`] hands back; [`format_score`] itself stays for the two book
/// call sites that need an owned value.
#[cfg(feature = "verbose2")]
fn push_score(out: &mut String, v: Value) {
    let mut digits = NumBuffer::new();
    if v.abs() >= VALUE_TB_WIN_IN_MAX_PLY {
        let distance = VALUE_MATE - v.abs();
        let mate = if v > 0 { distance } else { -distance };
        out.push_str("mate ");
        out.push_str(mate.format_into(&mut digits));
    } else {
        out.push_str("cp ");
        out.push_str((100 * v / PAWN_VALUE).format_into(&mut digits));
    }
}

/// [`push_score`] into a fresh `String`.
#[cfg(feature = "verbose2")]
pub(crate) fn format_score(v: Value) -> String {
    let mut out = String::new();
    push_score(&mut out, v);
    out
}

/// The evaluation network in the memory it was read into, paired with the file
/// path it came from.
///
/// The path is retained so `isready` is idempotent: a repeat reuses what is
/// already there instead of reading the file again.
///
/// How many instances there are is the machine's answer, decided when the
/// binary was built. On a single-node machine there is one, mapped from the
/// file and shared by every worker — and by every other engine process on the
/// machine, the pages being the page cache's. On a multi-node machine there is
/// one copy per *system* NUMA node the thread plan's workers run on, each in
/// memory belonging to that node. The granularity is the system node, not the
/// possibly L3-bundled logical node, so logical nodes that share a system node
/// share one copy.
struct LoadedEval {
    path: PathBuf,
    /// The network instances, in the order they were placed.
    instances: Vec<Arc<Search>>,
    /// System node → the instance its workers read. Empty when there is one
    /// instance for everything, which is every single-node machine and any plan
    /// that binds no worker.
    by_node: BTreeMap<NumaIndex, usize>,
}

/// The result of the heavy `isready` initialisation, produced inside the
/// [`KeepAlive`] scope and consumed by [`UsiDriver::handle_isready`] once the
/// keep-alive helper has stopped: the network is ready (`readyok`), the load
/// failed, or the machine is not the one this binary plans its threads for.
/// Each failure is one `info string` and no `readyok`.
enum IsreadyOutcome {
    Ready,
    LoadFailed(String),
    LayoutMismatch(String),
}

/// The opened opening books plus the `IgnoreBookPly` value captured at load
/// time.
///
/// `books` is the Multiple Book priority list, and a probe consults it in order
/// and takes the first hit. The reference captures `IgnoreBookPly` into the
/// book at `read_book` time, so changing it requires a reload.
///
/// Only the coordinator probes, once per `go` before helpers start: the
/// on-the-fly read path is not thread-safe by design, and this single-prober
/// discipline is what preserves that.
struct LoadedBook {
    books: Vec<Book>,
    ignore_book_ply: bool,
}

/// A search worker running on its own thread. The main thread
/// keeps reading USI lines while this runs; `stop` / `quit` set [`Self::stop`],
/// which the search polls at the reference `check_time` granularity. The worker
/// emits its own `info` / `bestmove`, then returns the session-owned
/// [`SearchState`] so the driver can reclaim it for the next `go`.
struct ActiveSearch {
    handle: JoinHandle<SearchState>,
    stop: Arc<AtomicBool>,
    /// The shared `go ponder` state (`Some` only for a `go ponder`). A plain
    /// `ponderhit` clears it (`set_ponderhit(false)`), turning the pondering
    /// search into a normal time-managed one; `None` means this was not a ponder
    /// search, and a stray `ponderhit` falls back to a `stop`.
    ponder: Option<Arc<PonderSignal>>,
    /// Suppresses the coordinator's `bestmove` (and final PV) for the
    /// Stochastic_Ponder ponderhit teardown, which stops the rewound search
    /// without emitting anything.
    suppress: Arc<AtomicBool>,
    /// Set by the coordinator *inside* the critical section that writes
    /// `bestmove`, so this search counts as finished from the moment its reply
    /// is on the wire.
    ///
    /// [`JoinHandle::is_finished`] is not that moment: the coordinator emits
    /// `bestmove` and only then unwinds, so a host that reads `bestmove` and
    /// immediately sends the next command can land in the window where the
    /// thread has not yet returned. A flag stamped under the output lock closes
    /// it — any reader that has seen the `bestmove` line took the same lock
    /// afterwards, so it cannot see this as unset.
    ///
    /// The `tt` commands' idle check is its only reader, and they are
    /// `verbose3`, so a build without that feature neither carries nor raises it.
    #[cfg(feature = "verbose3")]
    bestmove_sent: Arc<AtomicBool>,
    /// The root game ply this search ran at (`rootPos.game_ply()`), carried so
    /// a completed real search updates the driver's `last_game_ply`.
    game_ply: i32,
}

/// The session-owned search state a `go` lends to its worker and reclaims when
/// the worker finishes: the game-scoped worker history tables, which persist
/// across `go`s within one game and are reset by `usinewgame`.
///
/// The transposition table is not part of this handover — it lives behind an
/// [`Arc`], so the worker gets a clone and never hands it back.
struct SearchState {
    histories: WorkerHistories,
    /// The chosen worker's reported score / average score and the main worker's
    /// final `timeReduction`, carried back so the driver seeds the next `go`'s
    /// time management.
    ///
    /// Always `Some`: the reference runs this bookkeeping on *every* path,
    /// including the search-skipping short-circuits, which carry the unsearched
    /// `-VALUE_INFINITE` defaults and the current ply. The third element is
    /// `Some(tr)` only when a real search produced a fresh `timeReduction`; on a
    /// short-circuit the reference never touches `previousTimeReduction`, so the
    /// driver's persisted value is left unchanged.
    time_state: Option<(Value, Value, Option<f64>)>,
}

pub struct UsiDriver<R: BufRead, W: Write + Send + 'static> {
    reader: R,
    /// The output sink, shared with the search worker (which writes its own
    /// `info` / `bestmove`). A `Mutex` serialises the worker's lines against any
    /// the main thread emits concurrently.
    writer: Arc<Mutex<W>>,
    /// Where every setting comes from: the compile-time constants generated
    /// from the TOML config, in every build. See [`crate::settings`].
    settings: Settings,
    pos: Position,
    /// The loaded network holder, present only after a successful `isready`.
    /// `go` before this is set replies `bestmove resign`.
    eval: Option<LoadedEval>,
    /// The shared transposition table the root search runs against: the one
    /// `static`, whose size this binary was built with.
    ///
    /// Cleared on `usinewgame` and advanced per `go` by the search itself — the
    /// driver never bumps the generation. Every worker holds this same
    /// reference, so there is nothing to hand out and nothing to reclaim.
    tt: &'static TranspositionTable,
    /// Whether `isready` has placed the table's pages ([`Self::place_transposition_table`]).
    /// Once per session: the policy and the huge-page hint are properties of the
    /// address range, and repeating them would say the same thing again.
    tt_placed: bool,
    /// Game-scoped worker histories. `None` only while a worker
    /// holds them mid-search.
    histories: Option<WorkerHistories>,
    /// The loaded opening books, present only after an `isready` opened at least
    /// one readable `.ybb`. `None` means bookless (default `BookFile=no_book`, or
    /// every listed book failed / was unsupported). Behind an [`Arc`] so a `go`
    /// hands its coordinator a cheap clone.
    book: Option<Arc<LoadedBook>>,
    /// The `(resolved-name-list, on-the-fly, ignore-book-ply)` signature of the
    /// last book load — the Multiple Book priority list, not a single name.
    /// `isready` reloads only when this changes — the reference's reload-skip.
    book_signature: Option<(Vec<PathBuf>, bool, bool)>,
    /// A session-scoped seed advanced per `go`, driving both the book-selection
    /// PRNG and the `rtime` PRNG. Seeded from process entropy by default; tests
    /// pin it via [`UsiDriver::with_book_seed`].
    book_seed: u64,
    /// The evaluation-noise seed for the game in progress. Drawn from the
    /// operating system's randomness at construction and again at every
    /// `usinewgame`, and handed unchanged to every worker of every `go` in
    /// between: one game evaluates a position the same way throughout, and the
    /// next game evaluates it differently. It sits beside the transposition
    /// table rather than inside the Zobrist tables, which stay fixed, so
    /// nothing about position identity moves with it.
    #[cfg(feature = "random")]
    random_seed: u64,
    /// The in-flight search worker, if any.
    search: Option<ActiveSearch>,
    /// Time-management state that persists across `go`s within a game and is
    /// reset by `usinewgame`: the previous move's reported score / average score
    /// and its final `timeReduction`. Fed into each `go`'s [`TimeControl`] and
    /// refreshed on join.
    best_previous_score: Value,
    best_previous_average_score: Value,
    previous_time_reduction: f64,
    /// The root game ply of the last completed real search
    /// (`main_manager()->lastGamePly`), reset to `0` on `usinewgame`.
    ///
    /// At the next search start an odd `last_game_ply - game_ply` means the side
    /// to move alternated, which flips the sign of the persisted previous scores
    /// before they seed the next search.
    last_game_ply: i32,
    /// The last `position` command in parsed form (`last_position_cmd_string`),
    /// retained so a Stochastic_Ponder `go ponder` can rewind it by one move
    /// and a Stochastic_Ponder `ponderhit` can re-apply the real position.
    last_position: (PositionSfen, Vec<String>),
    /// The last `go` command's limits (`last_go_cmd_string`), retained so a
    /// Stochastic_Ponder `ponderhit` can re-issue it with `ponder` stripped.
    last_go: Option<GoLimits>,
    /// The worker thread pool: a main-worker slot plus `Threads − 1` persistent
    /// helper threads, each parked until a `go` dispatches it a job. The main
    /// worker is the per-`go` coordinator thread [`Self::handle_go`] spawns.
    pool: ThreadPool,
    /// The pool size the next (re)build uses. Always the `threads` config
    /// constant, except while a `verbose3` `bench` runs its own thread count
    /// — the one command that carries a worker count as an argument. Nothing
    /// on the match path ever writes it.
    pool_threads: usize,
    /// The active NUMA layout, rebuilt at construction from the layout constants
    /// this binary was built with — no `/sys` read, and no topology decision, on
    /// any path a game touches. Never replaced: the layout is a constant.
    numa_config: NumaConfig,
    /// The current worker → NUMA-node binding assignment, empty when binding is
    /// inactive. Recomputed at every pool (re)build and stable until the next
    /// one. Slot 0 is the per-`go` coordinator; `1..` are the helper threads.
    numa_bound: Vec<NumaIndex>,
    /// The shareable form of the binding assignment, including the worker →
    /// *system*-node map the memory policy is indexed by. `None` when binding is
    /// inactive. Rebuilt with the pool.
    numa_plan: Option<Arc<NumaBindPlan>>,
    /// Per-worker handles to the node-shared correction / pawn tables, rebuilt
    /// at every pool (re)build from [`Self::numa_bound`]. Length equals the pool
    /// size, so `[0]` is the coordinator's and `[1..]` the helpers'.
    worker_shared: Vec<Arc<SharedHistories>>,
    /// Per-worker handles to the NNUE network the worker evaluates with — a
    /// clone of its *system* NUMA node's copy. Empty until a network is loaded;
    /// otherwise its length equals the pool size, so `[0]` is the coordinator's
    /// and `[1..]` the helpers'.
    worker_networks: Vec<Arc<Search>>,
    /// The directory a relative `eval_dir` resolves against — the running
    /// executable's own, overridable via [`Self::with_eval_root`] so a test can
    /// present a directory of its own rather than the one it runs from.
    eval_root: PathBuf,
    /// Poll interval of the `isready` keep-alive helper thread ([`KeepAlive`]),
    /// overridable via [`Self::with_keep_alive_poll`] so a test can drive the
    /// mechanism with a short interval.
    keep_alive_poll: Duration,
    /// The sysfs root the `isready` layout check reads the running machine from
    /// — `/sys`, overridable via [`Self::with_sysfs_root`] so a test can hold
    /// the binary against a machine other than the one it is running on.
    sysfs_root: PathBuf,
    /// The CPUs this process may run on, captured at startup and held against
    /// the CPUs the compiled thread plan pins workers to by the `isready` check
    /// — which asks nothing of a build that pins none. Overridable via
    /// [`Self::with_startup_affinity`] so a test can present a confined process
    /// without confining the test process itself.
    startup_affinity: BTreeSet<usize>,
}

impl<R: BufRead, W: Write + Send + 'static> UsiDriver<R, W> {
    /// A driver whose book / `rtime` PRNG stream is seeded from process entropy,
    /// so every process run differs. Tests wanting reproducible book selection
    /// or `rtime` budgets construct via [`Self::with_book_seed`].
    pub fn new(reader: R, writer: Arc<Mutex<W>>) -> Self {
        Self::with_book_seed(reader, writer, Prng::random_seed())
    }

    /// A driver with an explicit book-PRNG session seed. The entropy default
    /// ([`Self::new`]) delegates here with [`Prng::random_seed`]; tests inject a
    /// fixed seed for deterministic book / `rtime` behaviour.
    pub fn with_book_seed(reader: R, writer: Arc<Mutex<W>>, book_seed: u64) -> Self {
        let settings = Settings::new();
        let threads = settings.threads();
        // Rebuild the active NUMA layout from the constants this binary was
        // built with: the machine was mapped to logical nodes under
        // `numa_policy` when the binary was, so nothing here reads `/sys` and
        // nothing decides a topology. Whether the machine still matches is the
        // `isready` check's question, asked once, before a game.
        let numa_config =
            NumaConfig::from_const(settings.numa_node_cpus(), settings.numa_custom_affinity());
        let numa_bound = compute_numa_binding(&numa_config, settings.numa_policy(), threads);
        let numa_plan = bind_plan(&numa_config, &numa_bound, settings.numa_system_nodes());
        let pool = ThreadPool::with_binding(threads, numa_plan.clone());
        // Build the per-node shared correction / pawn tables and give the
        // coordinator (worker 0) its node's set.
        let worker_shared = build_worker_shared(&numa_config, &numa_bound, threads);
        let histories = Some(WorkerHistories::with_shared(Arc::clone(&worker_shared[0])));
        // The coordinator's bundle is built (and filled) right here, on the USI
        // thread — so place it explicitly; see `place_coordinator_histories`.
        place_coordinator_histories(
            histories.as_ref(),
            coordinator_system_node(numa_plan.as_ref()),
        );
        Self {
            reader,
            writer,
            settings,
            pos: Position::startpos(),
            eval: None,
            tt: TranspositionTable::shared(),
            tt_placed: false,
            histories,
            book: None,
            book_signature: None,
            book_seed,
            #[cfg(feature = "random")]
            random_seed: new_game_seed(),
            search: None,
            best_previous_score: VALUE_INFINITE,
            best_previous_average_score: VALUE_INFINITE,
            previous_time_reduction: 0.85,
            last_game_ply: 0,
            last_position: (PositionSfen::StartPos, Vec::new()),
            last_go: None,
            pool,
            pool_threads: threads,
            numa_config,
            numa_bound,
            numa_plan,
            worker_shared,
            // No network loaded yet; populated by the first `isready`.
            worker_networks: Vec::new(),
            eval_root: network_file::executable_directory(),
            keep_alive_poll: KEEP_ALIVE_POLL_INTERVAL,
            sysfs_root: PathBuf::from(SYSFS_ROOT),
            startup_affinity: yorkie_numa::startup_affinity().clone(),
        }
    }

    /// Override the `isready` keep-alive poll interval, so a test can make a
    /// deliberately slowed heavy job elapse at least one tick. The newline still
    /// fires only after [`KEEP_ALIVE_TICKS_PER_NEWLINE`] polls, so this scales
    /// the whole cadence.
    pub fn with_keep_alive_poll(mut self, poll: Duration) -> Self {
        self.keep_alive_poll = poll;
        self
    }

    /// Override the directory a relative `eval_dir` resolves against.
    ///
    /// An engine finds its evaluation file beside itself, which is the one
    /// place a session cannot move: `eval_dir` is compiled in, and no build has
    /// an option surface to point it elsewhere. A test that has to drive a
    /// session against a network of its own — a synthetic one, or none at all —
    /// names the directory here, the same way it names a machine through
    /// [`Self::with_sysfs_root`].
    pub fn with_eval_root(mut self, root: PathBuf) -> Self {
        self.eval_root = root;
        self
    }

    /// Override the sysfs root the `isready` layout check reads, so a test can
    /// present a machine other than the one it runs on and see the check refuse
    /// it.
    pub fn with_sysfs_root(mut self, root: PathBuf) -> Self {
        self.sysfs_root = root;
        self
    }

    /// Override the CPU set the `isready` layout check takes for this process's
    /// startup affinity, so a test can drive a session as a confined process
    /// while the test process itself stays where it is.
    pub fn with_startup_affinity(mut self, cpus: BTreeSet<usize>) -> Self {
        self.startup_affinity = cpus;
        self
    }

    pub fn run(mut self) -> io::Result<()> {
        let mut buf = String::new();
        loop {
            buf.clear();
            let n = self.reader.read_line(&mut buf)?;
            if n == 0 {
                // EOF: treat as quit — stop and join any running search first.
                self.finish_search_join();
                return Ok(());
            }
            match parse_line(&buf) {
                Command::Usi => self.handle_usi()?,
                Command::IsReady => self.handle_isready()?,
                Command::SetOption { name, value } => self.handle_setoption(&name, &value)?,
                Command::UsiNewGame => self.handle_usinewgame(),
                Command::Position { sfen, moves } => self.handle_position(sfen, &moves)?,
                Command::Go(limits) => self.handle_go(limits)?,
                #[cfg(not(feature = "verbose2"))]
                Command::GoExtraClause(clause) => self.handle_go_extra_clause(&clause)?,
                Command::Stop => self.handle_stop(),
                Command::GameOver => self.handle_gameover(),
                Command::PonderHit => self.handle_ponderhit()?,
                #[cfg(feature = "verbose3")]
                Command::Bench(tokens) => self.handle_bench(&tokens)?,
                #[cfg(feature = "verbose3")]
                Command::Tt(tokens) => self.handle_tt(&tokens)?,
                Command::Quit => {
                    self.finish_search_join();
                    return Ok(());
                }
                #[cfg(feature = "verbose1")]
                Command::Unknown(line) => self.handle_unknown(&line)?,
                // The line is consumed and dropped either way; only the report
                // of it is gated.
                #[cfg(not(feature = "verbose1"))]
                Command::Unknown => {}
                Command::TooLong => self.handle_too_long()?,
            }
        }
    }

    /// Lock the shared output sink, recovering from a poisoned mutex (a worker
    /// panic must not wedge the main loop's own output).
    fn lock_writer(&self) -> MutexGuard<'_, W> {
        self.writer.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Emit one `info string <msg>` line.
    ///
    /// This is the unconditional sink, reserved for the initialisation phase and
    /// for a `verbose3` command's response payload: those lines are how a
    /// failed startup is diagnosed at all, so no feature may take them away.
    /// Everything else goes through [`diag!`].
    fn info_string(&self, msg: &str) -> io::Result<()> {
        Formatter::new(&mut *self.lock_writer()).info_string(msg)
    }

    /// Emit one diagnostic `info string` line — the `verbose1` surface, which
    /// carries every `info string` produced outside the initialisation phase.
    ///
    /// Callers pass `format_args!`, not a `String`, so the message is composed
    /// only if it is going to be written. They reach this through [`diag!`],
    /// which is what keeps the message text itself out of a build that cannot
    /// print it.
    #[cfg(feature = "verbose1")]
    fn info_string_diag(&self, body: std::fmt::Arguments<'_>) -> io::Result<()> {
        Formatter::new(&mut *self.lock_writer()).info_string_fmt(body)
    }

    /// Emit one `bestmove <mv>` line, preceded by the statistics of the interval
    /// it ends.
    fn bestmove(&self, mv: &str) -> io::Result<()> {
        let mut guard = self.lock_writer();
        #[cfg(feature = "verbose1")]
        emit_stats(&mut *guard);
        Formatter::new(&mut *guard).bestmove(mv)
    }

    /// Emit one `readyok` line.
    fn readyok(&self) -> io::Result<()> {
        Formatter::new(&mut *self.lock_writer()).readyok()
    }

    /// If a search worker is running, request its stop and join it, reclaiming
    /// the session-owned histories. Idempotent: a no-op when idle.
    ///
    /// Joining is also what leaves this thread alone with the shared
    /// transposition table, which is what a `usinewgame` clear wants.
    fn finish_search_join(&mut self) {
        if let Some(active) = self.search.take() {
            active.stop.store(true, Ordering::Relaxed);
            let state = active
                .handle
                .join()
                .expect("search worker thread must not panic");
            self.histories = Some(state.histories);
            // Carry the finished search's time-management outputs forward. The
            // reference runs this bookkeeping on every path, so the
            // short-circuits carry too, but with `tr == None`, since
            // `previousTimeReduction` is only written by a real
            // `iterative_deepening` run.
            if let Some((score, avg, tr)) = state.time_state {
                self.best_previous_score = score;
                self.best_previous_average_score = avg;
                if let Some(tr) = tr {
                    self.previous_time_reduction = tr;
                }
                // Remember the ply this search ran at so the next search can
                // detect a side-to-move flip.
                self.last_game_ply = active.game_ply;
            }
        }
    }

    /// Place the shared transposition table's pages, once per session.
    ///
    /// The table is a `static` whose size the build fixed; what is left to
    /// decide is where its pages come from, and both halves are asked for
    /// before anything faults one in:
    ///
    /// 1. A memory policy naming exactly the system NUMA nodes the compiled
    ///    thread plan's workers run on ([`table_placement`]): `MPOL_INTERLEAVE`
    ///    over the set when they span several, so no node's memory controller
    ///    carries the whole engine's probe traffic, and a preference for the one
    ///    node when they share it. The policy goes on first, because it decides
    ///    where the *first touch* of every page lands. A plan that pins no
    ///    worker gets no policy at all: the process's own is the better answer
    ///    there, since it is the one an operator confining the process to a node
    ///    already gave.
    /// 2. `madvise(MADV_HUGEPAGE)`, so the region a huge-page boundary starts is
    ///    actually backed by huge pages.
    ///
    /// Both are best-effort. A kernel without `CONFIG_NUMA`, a seccomp filter or
    /// a restricted cgroup refuses one or both, and the table then keeps the
    /// process default policy and ordinary pages — slower, never wrong. The
    /// outcome is reported rather than assumed, since a tournament host silently
    /// falling back is exactly what an operator wants to see before the game and
    /// not after it — so the line is an initialisation-phase `info string`,
    /// present in every build like the rest of them. It names the size and the
    /// nodes but not the address, which differs from run to run and would make
    /// every transcript differ with it.
    fn place_transposition_table(&mut self) -> io::Result<()> {
        if self.tt_placed {
            return Ok(());
        }
        self.tt_placed = true;

        // The span handed to the kernel is the whole `static`, alignment tail
        // included; the size reported is the clusters, which is what the
        // `usi_hash` setting asked for.
        let (addr, span) = self.tt.backing_region();
        let outcome = |accepted: bool| if accepted { "applied" } else { "refused" };
        let placement = match table_placement(&self.numa_bound, self.settings.numa_system_nodes()) {
            TablePlacement::OnNode(node) => format!(
                "preferred on node {node} {}",
                outcome(mempolicy::prefer_region_on_node(addr, span, node))
            ),
            TablePlacement::AcrossNodes(nodes) => format!(
                "interleave on nodes {} {}",
                yorkie_numa::format_cpu_list(nodes.iter().copied()),
                outcome(mempolicy::interleave_region_over_nodes(addr, span, &nodes))
            ),
            TablePlacement::ProcessDefault => "process default policy".to_string(),
        };
        let huge = yorkie_storage::advise_huge_pages(addr, span);

        self.info_string(&format!(
            "transposition table: {} MiB; {placement}; huge pages {}",
            yorkie_storage::TABLE_BYTES / (1024 * 1024),
            outcome(huge),
        ))
    }

    /// Recompute the worker → NUMA-node binding for the current pool size and
    /// the configured mapping policy, and rebuild the worker pool with it.
    /// Every pool (re)build routes through here so [`Self::numa_bound`] stays
    /// consistent with the live pool. Helper threads bind once at spawn; the
    /// per-`go` coordinator binds at each `go`.
    ///
    /// Callers must have joined any running search first, since a resize
    /// destroys and recreates the helper threads.
    fn rebuild_pool(&mut self) {
        let requested = self.pool_threads;
        let policy = self.settings.numa_policy();
        self.numa_bound = compute_numa_binding(&self.numa_config, policy, requested);
        // Rebuild the per-node shared correction / pawn tables from the fresh
        // binding assignment, so every pool rebuild resets them as the
        // reference does. The coordinator's own game-scoped per-worker tables
        // persist, so only its shared handle is swapped.
        self.worker_shared = build_worker_shared(&self.numa_config, &self.numa_bound, requested);
        if let Some(h) = self.histories.as_mut() {
            h.set_shared(Arc::clone(&self.worker_shared[0]));
        }
        self.numa_plan = bind_plan(
            &self.numa_config,
            &self.numa_bound,
            self.settings.numa_system_nodes(),
        );
        // The coordinator's per-worker tables outlive a pool rebuild and were
        // faulted on the USI thread, so re-assert their placement for the fresh
        // assignment. Helpers need nothing here: they are respawned and each
        // allocates its own bundle on-thread after pinning.
        place_coordinator_histories(
            self.histories.as_ref(),
            coordinator_system_node(self.numa_plan.as_ref()),
        );
        self.pool
            .set_with_binding(requested, self.numa_plan.clone());
        // Re-resolve the per-worker network handles for the fresh binding /
        // pool size: the reference forces replication right after
        // `resize_threads` (`ensure_network_replicated`).
        self.rebuild_networks();
    }

    /// Resolve the per-worker [`Self::worker_networks`] handles for the current
    /// pool size and binding assignment: each worker gets the instance its own
    /// *system* NUMA node's memory holds.
    ///
    /// Done at configuration time, so nothing about which network a worker
    /// reads is decided on the search path. Nothing is copied here: the
    /// instances were placed when the file was read, and this only hands out
    /// [`Arc`]s to them.
    fn rebuild_networks(&mut self) {
        let requested = self.pool.size().max(1);
        // Worker `i`'s system node — the reference's `get_discriminator` per
        // worker. Read before `self.eval` is borrowed.
        let sys_nodes = if self.numa_bound.is_empty() {
            Vec::new()
        } else {
            worker_system_nodes(&self.numa_bound, self.settings.numa_system_nodes())
        };

        let Some(eval) = self.eval.as_ref() else {
            // No network loaded; `go` before an `isready` resigns anyway.
            self.worker_networks = Vec::new();
            return;
        };

        self.worker_networks =
            resolve_worker_networks(&eval.instances, &eval.by_node, &sys_nodes, requested);
    }

    /// Emit each non-blank line of `text` as `info string <line>` through the
    /// single output sink, mirroring the reference `print_info_string`: the
    /// text is split on `'\n'` and whitespace-only lines are skipped.
    #[cfg(feature = "verbose3")]
    fn emit_info_string_lines(&self, text: &str) -> io::Result<()> {
        for line in text.split('\n') {
            if !line.trim().is_empty() {
                self.info_string(line)?;
            }
        }
        Ok(())
    }

    /// Emit the `Using N thread[s][ with NUMA node thread binding: ...]` line.
    #[cfg(feature = "verbose3")]
    fn emit_thread_allocation_information(&self) -> io::Result<()> {
        self.emit_info_string_lines(&thread_allocation_information_as_string(
            self.pool.size(),
            &self.numa_config,
            &self.numa_bound,
        ))
    }

    /// The `usi` handshake: identity and `usiok`, with NO `option name ...`
    /// lines — in every build.
    ///
    /// The engine has no runtime configuration to advertise. Every setting was
    /// compiled in from the TOML config, and a GUI that saw an option list would
    /// be shown a control it cannot actually operate.
    fn handle_usi(&mut self) -> io::Result<()> {
        let mut guard = self.lock_writer();
        let mut f = Formatter::new(&mut *guard);
        f.id_name(ENGINE_NAME)?;
        f.id_author(ENGINE_AUTHOR)?;
        f.usiok()
    }

    /// The absolute path a `<BookDir>/<BookFile>` pair resolves to.
    ///
    /// Mirrors the reference `get_book_name`: join `BookDir` onto the binary's
    /// folder, then `BookFile`. An absolute `BookDir` (as tests use) wins over
    /// the binary folder — `Path::join` semantics match `Path::Combine`.
    /// `BookDir` / `BookFile` are data paths opened as-is via `std::fs`, with
    /// no shell or metacharacter interpretation.
    fn book_path(&self, book_dir: &str, book_file: &str) -> PathBuf {
        let base = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(Path::to_path_buf))
            .unwrap_or_else(|| PathBuf::from("."));
        base.join(book_dir).join(book_file)
    }

    /// (Re)load the opening books from the current options — the reference's
    /// `BookMoveSelector::read_book`. Reloads only when the `(name list,
    /// on-the-fly, IgnoreBookPly)` capture changed.
    ///
    /// `no_book`, an unsupported format, or an open failure all leave that name
    /// out of the list without panicking; a `.db` whose file is absent falls
    /// back to the `.ybb` sibling with the reference's fallback info string.
    fn reload_book(&mut self) -> io::Result<()> {
        let book_file = self.settings.book_file().to_string();
        let book_dir = self.settings.book_dir().to_string();
        let on_the_fly = self.settings.book_on_the_fly();
        let ignore_book_ply = self.settings.ignore_book_ply();
        let base = self.book_path(&book_dir, &book_file);

        // Enumerate the priority series now: the resolved name list is half of
        // the reload-skip capture, so a numbered file appearing (or vanishing)
        // between two `isready`s is itself a reason to reload.
        let (names, notices) = book_names(&base);

        let signature = (names.clone(), on_the_fly, ignore_book_ply);
        if self.book_signature.as_ref() == Some(&signature) {
            return Ok(());
        }
        self.book_signature = Some(signature);
        self.book = None;

        // `no_book` → bookless, silently. (`book_name_without_extension` yields
        // an empty stem for it, so `names` is just the base name anyway.)
        if book_file == "no_book" {
            return Ok(());
        }

        // The "priority book file exists twice" notices from the enumeration,
        // verbatim from the reference.
        for notice in &notices {
            self.info_string(notice)?;
        }

        let mut books: Vec<Book> = Vec::new();
        for name in &names {
            // Resolve a `.db` whose file is absent to its `.ybb` sibling. The
            // pin applies this per name inside `MemoryBook::read_book`; for a
            // numbered name it is always a no-op (the enumeration already
            // proved the file exists).
            let resolved = resolve_book_filename_with_ybb_fallback(name);
            if &resolved != name {
                self.info_string(&format!(
                    "book file fallback : {} -> {}",
                    name.display(),
                    resolved.display()
                ))?;
            }

            // Divergence from the reference: only `.ybb` is supported. Anything
            // else — including a `.db` the reference's two-extension resolution
            // picked for a numbered slot — behaves as no-book after an
            // info-string notice, never a panic and never a SILENT skip: a
            // silent skip would hide a book the reference would have used.
            if !has_book_ext(&resolved, BOOK_EXT_YBB) {
                self.info_string(&format!("unsupported book format : {}", resolved.display()))?;
                continue;
            }

            let opened = if on_the_fly {
                Book::open_on_the_fly(&resolved)
            } else {
                Book::open_in_memory(&resolved)
            };
            match opened {
                Ok(book) => {
                    let count = book.record_count();
                    books.push(book);
                    self.info_string(&format!("book loaded : {count} positions"))?;
                }
                Err(e) => {
                    // Mirrors the reference's open/validate failure → this name is left
                    // out of the priority list.
                    self.info_string(&format!("book load failed : {e}"))?;
                }
            }
        }

        if !books.is_empty() {
            self.book = Some(Arc::new(LoadedBook {
                books,
                ignore_book_ply,
            }));
        }
        Ok(())
    }

    fn handle_isready(&mut self) -> io::Result<()> {
        // Reclaim any worker before touching the table it may hold.
        self.finish_search_join();
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
            let _keep_alive = KeepAlive::spawn(Arc::clone(&self.writer), self.keep_alive_poll);
            self.isready_heavy_job()?
            // `_keep_alive` dropped here: stop flag set, helper thread joined.
        };

        match outcome {
            IsreadyOutcome::Ready => {
                self.readyok()?;
                // The initialisation phase allocates the table, the network and
                // the pool, and none of that belongs to a reply: the first
                // reported interval starts here.
                #[cfg(feature = "verbose1")]
                clear_alloc_count();
                Ok(())
            }
            IsreadyOutcome::LoadFailed(reason) => {
                // Contract: on a load failure, emit
                // `info string eval load failed: <reason>` and do NOT emit
                // `readyok`. There is no working network to lose here: one
                // already read is reused by the idempotent path above, so the
                // only session that reaches this had none to begin with.
                self.info_string(&format!("eval load failed: {reason}"))
            }
            IsreadyOutcome::LayoutMismatch(reason) => {
                self.info_string(&format!("NUMA layout mismatch: {reason}"))
            }
        }
    }

    /// The heavy `isready` initialisation, run inside the [`KeepAlive`] scope of
    /// [`Self::handle_isready`]. Returns the outcome so the caller emits
    /// `readyok` / the load-failure notice *after* the keep-alive helper has
    /// stopped — the terminal reply never races the keep-alive newlines.
    fn isready_heavy_job(&mut self) -> io::Result<IsreadyOutcome> {
        // Before anything is allocated for a machine: is this the machine? The
        // whole thread plan was folded from the layout the binary was built on,
        // so a difference here is a wrong answer that is available now, and one
        // no later stage would report.
        if let Some(reason) = self.numa_layout_refusal() {
            return Ok(IsreadyOutcome::LayoutMismatch(reason));
        }
        // The machine is the one the binary was built for, so the thread plan's
        // nodes are the ones the table belongs on. Done before anything else
        // here, so the policy is in force before the first page of it is
        // touched.
        self.place_transposition_table()?;
        // Load / reload the opening book (the reference does this in isready).
        self.reload_book()?;
        let path = self.evaluation_file_path();

        // Idempotent: a repeat `isready` reuses what is already in memory — no
        // second read of the file, and on a multi-node machine no second copy
        // into the regions the first one is being read from.
        if self.eval.as_ref().is_some_and(|e| e.path == path) {
            return Ok(IsreadyOutcome::Ready);
        }

        match self.place_evaluation_network(&path) {
            Ok((eval, warnings, placement)) => {
                // Surface the complaints the conversion had about the source
                // network (hash mismatches) as `info string` lines before
                // `readyok`, mirroring the reference `LoadAndShare` /
                // `Detail::ReadParameters` diagnostics. A clean network carries
                // none, so a correct one emits nothing new.
                for warning in &warnings {
                    self.info_string(warning)?;
                }
                self.info_string(&placement)?;
                self.eval = Some(eval);
                // Resolve the per-worker handles now, so the next `go` finds
                // every worker already pointed at the instance its node holds.
                self.rebuild_networks();
                Ok(IsreadyOutcome::Ready)
            }
            Err(e) => Ok(IsreadyOutcome::LoadFailed(e.to_string())),
        }
    }

    /// The evaluation file the engine plays with: `eval_dir` — absolute as it
    /// stands, relative resolved against the running executable's own directory
    /// — and the file's name.
    ///
    /// `eval_dir` is a data path opened as-is via `std::fs`: no shell, no
    /// metacharacter interpretation, no symlink policy beyond what the OS
    /// `open` does.
    fn evaluation_file_path(&self) -> PathBuf {
        network_file::network_path(&self.eval_root)
    }

    /// Read the evaluation file into the memory the machine calls for, and
    /// report where it went.
    ///
    /// One mapping on a single-node machine; one copy per system NUMA node the
    /// thread plan's workers run on otherwise, each region placed on its node
    /// *before* the copy so every page's first touch lands there, and hinted
    /// for huge pages either way. Every placement call is best-effort: a kernel
    /// without `CONFIG_NUMA`, a seccomp filter or a restricted cgroup refuses
    /// one, and the network then keeps the process's own policy and ordinary
    /// pages — slower, never wrong. The line says which, in every build, since
    /// a tournament host silently falling back is what an operator wants to see
    /// before the game rather than after it.
    fn place_evaluation_network(
        &mut self,
        path: &Path,
    ) -> Result<(LoadedEval, Vec<String>, String), NnueError> {
        // Nothing may still be reading the regions when they are filled: on a
        // multi-node machine they are the process's only storage for the
        // network. Any search has been joined by the caller, so dropping these
        // handles drops the last references.
        self.worker_networks = Vec::new();
        self.eval = None;

        let outcome = |accepted: bool| if accepted { "applied" } else { "refused" };
        let mut instances = Vec::new();
        let mut by_node = BTreeMap::new();
        let mut warnings = Vec::new();

        let placement = if network_file::SHARED_MAPPING {
            let (search, w) = Search::map_evaluation_file(path)?;
            let (addr, span) = search.network().parameter_region();
            let huge = yorkie_storage::advise_huge_pages(addr, span);
            warnings = w;
            instances.push(Arc::new(search));
            format!("one shared mapping; huge pages {}", outcome(huge))
        } else {
            let nodes = network_regions(&self.numa_bound, self.settings.numa_system_nodes());
            let mut reported = Vec::new();
            // A plan that binds no worker gets one region under the process's
            // own policy, which is the policy an operator confining the process
            // already chose.
            let slots: Vec<Option<NumaIndex>> = if nodes.is_empty() {
                vec![None]
            } else {
                nodes.iter().copied().map(Some).collect()
            };
            for (slot, node) in slots.iter().enumerate() {
                let (addr, span) = network_file::region_backing(slot);
                let placed = match node {
                    Some(node) => format!(
                        "node {node} {}",
                        outcome(mempolicy::migrate_region_to_node(addr, span, *node))
                    ),
                    None => "the process default policy".to_string(),
                };
                let huge = yorkie_storage::advise_huge_pages(addr, span);
                // SAFETY: every handle to a network over these regions was
                // dropped above, and a search that could have held one was
                // joined before this `isready` reached here, so nothing is
                // reading the region being filled.
                let (search, w) = unsafe { Search::load_evaluation_file_into_region(slot, path)? };
                if let Some(node) = node {
                    by_node.insert(*node, slot);
                }
                // Every region reads the same file, so its complaints are the
                // same each time; they are worth reporting once.
                warnings = w;
                instances.push(Arc::new(search));
                reported.push(format!("{placed}, huge pages {}", outcome(huge)));
            }
            format!("one copy on {}", reported.join("; "))
        };

        Ok((
            LoadedEval {
                path: path.to_path_buf(),
                instances,
                by_node,
            },
            warnings,
            format!(
                "evaluation network: {} MiB; {placement}",
                network_file::DATA_BYTES / (1024 * 1024),
            ),
        ))
    }

    /// How the machine this process runs on differs from the one the binary was
    /// built for, or `None` when they agree.
    ///
    /// The live layout is resolved exactly as the build resolved it: the same
    /// mapping policy, over every online CPU, so what is compared is machine
    /// against machine and not machine against process. A tree that cannot be
    /// read is itself a refusal — an unverifiable layout is not a matching one.
    fn numa_layout_refusal(&self) -> Option<String> {
        let opts = match yorkie_numa::machine_sysfs_options(&self.sysfs_root) {
            Ok(opts) => opts,
            Err(e) => return Some(e),
        };
        let live = match NumaConfig::from_policy(self.settings.numa_policy(), &opts) {
            Ok(cfg) => NumaLayout::of(&cfg, &opts),
            Err(e) => return Some(e),
        };
        numa_layout_difference(
            &live,
            &self.startup_affinity,
            // What a confined process hides is a CPU some worker was going to
            // pin itself to, so the question is only asked of a build whose
            // compiled plan pins one, and only of the CPUs that plan uses. A
            // build that binds nothing — a single thread, or
            // `numa_policy = "none"` — has no such CPU, and starting it under a
            // `taskset` or in a cpuset takes nothing away from it.
            &self.numa_bound,
            self.settings.numa_node_cpus(),
        )
    }

    /// `setoption name <N> value <V>`: the USI minimum, in every build.
    ///
    /// There is no option to set — every setting was fixed at build time from
    /// the TOML config, and the `usi` reply advertises no options at all. USI
    /// requires no reply, so the line is parsed, consumed and dropped.
    fn handle_setoption(&mut self, _name: &str, _value: &str) -> io::Result<()> {
        Ok(())
    }

    fn handle_position(&mut self, sfen: PositionSfen, moves: &[String]) -> io::Result<()> {
        // Build a scratch Position. On any error, emit `info string …` and
        // leave `self.pos` untouched — the input-validation contract: the prior
        // position must survive a malformed `position` line.
        let mut scratch = match &sfen {
            PositionSfen::StartPos => Position::startpos(),
            PositionSfen::Sfen(s) => match parse_sfen(s) {
                Ok(p) => p,
                Err(e) => {
                    return diag!(self, "position parse error: {}", e);
                }
            },
        };
        let mut legal_buf: Vec<Move> = Vec::new();
        for s in moves {
            let parsed = match parse_usi_move(s, &scratch) {
                Ok(m) => m,
                Err(_) => {
                    return diag!(self, "illegal move: {}", s);
                }
            };
            legal_buf.clear();
            scratch.generate_legal_all(&mut legal_buf);
            if !legal_buf.contains(&parsed) {
                return diag!(self, "illegal move: {}", s);
            }
            scratch.do_move(parsed);
        }
        self.pos = scratch;
        // Retain the parsed command for the Stochastic_Ponder rewind / re-issue
        // (`last_position_cmd_string`).
        self.last_position = (sfen, moves.to_vec());
        Ok(())
    }

    fn handle_usinewgame(&mut self) {
        // Reclaim any running search, then reset the game state: startpos, an
        // emptied table, and fresh history tables (the reference
        // `search_clear`). Clearing the table also resets its generation; the next
        // `go` bumps it again via `run_root`.
        self.finish_search_join();
        self.pos = Position::startpos();
        // The join above left this thread as the only one touching the table,
        // so the clear races nothing.
        self.tt.clear();
        // Fresh per-worker tables. The shared correction / pawn handle is swapped
        // to the freshly (re)built node table set by `rebuild_pool` below, so a
        // cheap clone of the current handle here avoids a throwaway allocation.
        self.histories = Some(WorkerHistories::with_shared(Arc::clone(
            &self.worker_shared[0],
        )));
        // A new game draws a new evaluation-noise seed, which is what makes the
        // engine play the next game differently from this one.
        #[cfg(feature = "random")]
        {
            self.random_seed = new_game_seed();
        }
        // Reset the persistent time-management inputs to their
        // first-move-of-a-game sentinels.
        self.best_previous_score = VALUE_INFINITE;
        self.best_previous_average_score = VALUE_INFINITE;
        self.previous_time_reduction = 0.85;
        // Reset the side-flip detector and the retained command state (the
        // `last_position` default is the reference's `"position startpos"`).
        self.last_game_ply = 0;
        self.last_position = (PositionSfen::StartPos, Vec::new());
        self.last_go = None;
        // Reset the helper workers' game-scoped histories too.
        // The reference `search_clear` clears every worker; here the helper
        // histories live in the pool threads, so recreating the pool gives them
        // fresh tables — the join above guarantees the helpers are idle first.
        // Routing through `rebuild_pool` keeps the NUMA binding assignment
        // consistent with the recreated helpers.
        self.rebuild_pool();
        // The pool and the history tables just rebuilt are a new game's setup,
        // not the first move's work, so they are not counted against it.
        #[cfg(feature = "verbose1")]
        clear_alloc_count();
    }

    /// Snapshot the book-selection options into a [`BookConfig`] for one `go`.
    /// `IgnoreBookPly` is not here — it is captured at load time and travels
    /// with [`LoadedBook`].
    ///
    /// Both profiles' fields are snapshotted; the probe picks between them from
    /// `book_options_v2` and the root side to move. An option the active profile
    /// did not register reads as its type's zero, which is inert on the leg that
    /// never consults it.
    fn book_config(&self) -> BookConfig {
        BookConfig {
            book_options_v2: self.settings.book_options_v2(),
            narrow_book: self.settings.narrow_book(),
            book_moves: self.settings.book_moves(),
            ignore_rate: self.settings.book_ignore_rate(),
            eval_diff: self.settings.book_eval_diff(),
            eval_black_diff: self.settings.book_eval_black_diff(),
            eval_white_diff: self.settings.book_eval_white_diff(),
            eval_black_limit: self.settings.book_eval_black_limit(),
            eval_white_limit: self.settings.book_eval_white_limit(),
            depth_limit: self.settings.book_depth_limit(),
            depth_black_limit: self.settings.book_depth_black_limit(),
            depth_white_limit: self.settings.book_depth_white_limit(),
            consider_move_count: self.settings.consider_book_move_count(),
            // Shapes the book `info` lines and nothing else, so it is read only
            // in a build that prints them.
            #[cfg(feature = "verbose2")]
            pv_moves: self.settings.book_pv_moves(),
            flipped_book: self.settings.flipped_book(),
        }
    }

    fn handle_go(&mut self, limits: GoLimits) -> io::Result<()> {
        // A new `go` supersedes any lingering search; reclaim its state first.
        self.finish_search_join();

        // Retain this `go` for a later Stochastic_Ponder re-issue.
        self.last_go = Some(limits.clone());

        // Stochastic_Ponder `go ponder`: ponder one move earlier than the
        // retained position (drop its last move); `ponderMode` stays set.
        if limits.ponder && self.settings.stochastic_ponder() {
            self.apply_stochastic_ponder_rewind();
        }

        // The ply the search actually runs at (rewound under Stochastic_Ponder),
        // carried so a completed real search updates `last_game_ply`.
        let game_ply = self.pos.ply() as i32;

        // Build the coordinator job (option-seeded limits, all per-`go`
        // snapshots). `None` means no network is loaded — notify and resign.
        let Some(job) = self.prepare_coordinator_job(
            limits,
            #[cfg(feature = "verbose2")]
            false,
        ) else {
            diag!(self, "no eval network loaded; run isready")?;
            return self.bestmove("resign");
        };

        // The handles the main loop signals on `stop` / `ponderhit` / a
        // Stochastic_Ponder teardown; cloned out of the job before it moves into
        // the worker thread.
        let stop_for_active = Arc::clone(&job.stop);
        let ponder_for_active = job.ponder.as_ref().map(Arc::clone);
        let suppress_for_active = Arc::clone(&job.suppress_bestmove);
        #[cfg(feature = "verbose3")]
        let sent_for_active = Arc::clone(&job.bestmove_sent);
        let handle = std::thread::spawn(move || {
            let outcome = run_coordinated(job);
            SearchState {
                histories: outcome.histories,
                time_state: outcome.time_state,
            }
        });

        self.search = Some(ActiveSearch {
            handle,
            stop: stop_for_active,
            ponder: ponder_for_active,
            suppress: suppress_for_active,
            #[cfg(feature = "verbose3")]
            bestmove_sent: sent_for_active,
            game_ply,
        });
        Ok(())
    }

    /// A `go` line carrying a clause that arrives with `verbose2`, seen by a
    /// build without that feature: report it and start nothing.
    ///
    /// Failing loud is deliberate — ignoring the clause would silently change
    /// the search's terms, turning `go depth 4` into a clock-less `go` in the
    /// middle of a game. Any search already running is left alone.
    #[cfg(not(feature = "verbose2"))]
    fn handle_go_extra_clause(&mut self, clause: &str) -> io::Result<()> {
        diag!(
            self,
            "go error: `{}` requires a verbose2 build; no search started",
            clause
        )
    }

    /// Stochastic_Ponder `go ponder` rewind: reconstruct the retained position
    /// with its last move dropped and install it as the search root. A
    /// best-effort trim — an empty move list (nothing to rewind) or a rebuild
    /// failure leaves the current position untouched.
    fn apply_stochastic_ponder_rewind(&mut self) {
        let (sfen, moves) = &self.last_position;
        if moves.is_empty() {
            return;
        }
        let rewound = &moves[..moves.len() - 1];
        if let Some(pos) = build_position_from(sfen, rewound) {
            self.pos = pos;
        }
    }

    /// Build the [`CoordinatorJob`] for one search — the shared preamble of both
    /// `go` and `bench`.
    ///
    /// Returns `None` when no network is loaded; the caller emits the resign or
    /// notice appropriate to its context. `disable_pv_interval` forces the
    /// per-iteration PV interval to zero so every iteration prints, and is set
    /// only by `bench`; there is no such interval to force below `verbose2`.
    fn prepare_coordinator_job(
        &mut self,
        limits: GoLimits,
        #[cfg(feature = "verbose2")] disable_pv_interval: bool,
    ) -> Option<CoordinatorJob<W>> {
        // Nothing propagates the NNUE fixed-point scale here: the reference's
        // mutable global `NNUE::FV_SCALE` is a compile-time constant in the
        // evaluation layer, generated from the same `fv_scale` config key, so
        // there is no live value for a search to pick up.

        // Seed the depth / node ceilings from the `DepthLimit` / `NodesLimit`
        // options when this `go` carries no explicit token, then let an explicit
        // token stand. A seeded depth also disables the parallel-search vote
        // below, exactly like an explicit `go depth N`: the reference's
        // `!limits.depth` guard keys off the final value regardless of source.
        // Both keys need the same feature as the clauses they seed, so a build
        // without it has neither source and reaches no ceiling at all.
        #[cfg(feature = "verbose2")]
        let limits = {
            let mut limits = limits;
            if limits.depth.is_none() {
                let dl = self.settings.depth_limit();
                if dl != 0 {
                    limits.depth = Some(dl as u32);
                }
            }
            if limits.nodes.is_none() {
                let nl = self.settings.nodes_limit();
                if nl != 0 {
                    limits.nodes = Some(nl as u64);
                }
            }
            limits
        };

        // No network loaded (a non-compliant host `go`, or a `bench` whose
        // `isready` never succeeded). The caller resigns for this position. The
        // per-worker network handles ([`Self::worker_networks`]) are resolved
        // alongside `eval` (rebuilt on every load / pool rebuild), so a loaded
        // `eval` always has a network for every worker.
        self.eval.as_ref()?;

        // Map the `go` limits + time options onto the reference
        // `TimeManagement`. `use_time_management()` is true only for a real
        // clock / `go rtime`; a `TimeControl` is installed on the main worker
        // for those and for `go movetime`, and is `None` otherwise (fixed depth
        // / nodes / infinite / mate), where the search runs unbounded by time.
        let us = self.pos.side_to_move();
        let now = Instant::now();
        // Every `go` a build without `verbose2` can be given is bounded by the
        // clock: the six clauses that bound a search any other way, and the two
        // config keys that seed two of them, all need that feature. Such a build
        // has nothing to classify, so it carries no classification at all.
        #[cfg(feature = "verbose2")]
        let use_time_management = limits.depth.is_none()
            && limits.nodes.is_none()
            && limits.mate.is_none()
            && limits.movetime.is_none()
            && !limits.infinite;
        // DELIBERATE DIVERGENCE (see `GoLimits::mate` in the parser): the reference
        // leaves `go mate`'s enforcement to a separate mate engine, but this port has
        // none, so a concrete `go mate <ms>` budget is mapped onto a `movetime`-style
        // time bound. A bare / `infinite` `go mate` (the `MATE_UNLIMITED_MS`
        // sentinel) carries no bound and runs until `stop`.
        #[cfg(feature = "verbose2")]
        let mate_budget = match limits.mate {
            Some(m) if m != MATE_UNLIMITED_MS => Some(m as i64),
            _ => None,
        };
        #[cfg(feature = "verbose2")]
        let movetime = limits.movetime.map(|m| m as i64).or(mate_budget);

        // Side-flip continuity: when the side to move alternated between the
        // last completed search and this one — an odd
        // `last_game_ply - game_ply`, as a Stochastic_Ponder rewind / re-issue
        // produces — negate the persisted previous scores (each unless it is
        // the `VALUE_INFINITE` first-move sentinel) before they seed
        // `iterValue` / `fallingEval`. On the normal same-side case the parity
        // is even and the scores pass through unchanged.
        let flip_previous = (self.last_game_ply - self.pos.ply() as i32) & 1 != 0;
        let best_prev_score = match self.best_previous_score {
            VALUE_INFINITE => VALUE_INFINITE,
            s if flip_previous => -s,
            s => s,
        };
        let best_prev_average_score = match self.best_previous_average_score {
            VALUE_INFINITE => VALUE_INFINITE,
            a if flip_previous => -a,
            a => a,
        };

        // A `go movetime` (or the `go mate <ms>` budget mapped onto one) still
        // needs a `TimeManagement` even though it switches dynamic management
        // off. Neither clause exists without `verbose2`, where every `go` is
        // clock-managed and so gets one unconditionally; the label is the early
        // exit only a `verbose2` build has.
        #[cfg_attr(not(feature = "verbose2"), allow(unused_labels))]
        let time = 'time: {
            #[cfg(feature = "verbose2")]
            if !(use_time_management || movetime.is_some()) {
                break 'time None;
            }
            let (time_opt, inc_opt) = match us {
                yorkie_state::Color::Black => (limits.btime, limits.binc),
                yorkie_state::Color::White => (limits.wtime, limits.winc),
            };
            let mmtd = remap_max_moves_to_draw(self.settings.max_moves_to_draw());
            // A distinct PRNG stream from the book selection, so `go rtime`'s
            // randomised budget never perturbs (or is perturbed by) book choice.
            // `go rtime` is the stream's only consumer, so it is drawn only in a
            // build that has the clause.
            #[cfg(feature = "verbose2")]
            let mut prng = Prng::new(self.book_seed ^ 0xA5A5_5A5A_1234_5678);
            let tm = TimeManagement::init(
                &TimeInput {
                    time_us: time_opt.unwrap_or(0) as i64,
                    inc_us: inc_opt.unwrap_or(0) as i64,
                    byoyomi_us: limits.byoyomi.unwrap_or(0) as i64,
                    #[cfg(feature = "verbose2")]
                    movetime: movetime.unwrap_or(0),
                    #[cfg(feature = "verbose2")]
                    rtime: limits.rtime.unwrap_or(0) as i64,
                    network_delay: self.settings.network_delay(),
                    network_delay2: self.settings.network_delay2(),
                    minimum_thinking_time: self.settings.minimum_thinking_time(),
                    slow_mover: self.settings.slow_mover(),
                    round_up_to_fullsecond: self.settings.round_up_to_full_second(),
                    usi_ponder: self.settings.usi_ponder(),
                    stochastic_ponder: self.settings.stochastic_ponder(),
                    ply: self.pos.ply() as i32,
                    max_moves_to_draw: mmtd,
                    start_time: now,
                },
                #[cfg(feature = "verbose2")]
                &mut prng,
            );
            if tm.mtg_error {
                let _ = diag!(self, "Error! : MaxMovesToDraw is too small.");
            }
            Some(TimeControl {
                tm,
                #[cfg(feature = "verbose2")]
                use_time_management,
                #[cfg(feature = "verbose2")]
                movetime,
                n_threads: self.pool.size(),
                best_previous_score: best_prev_score,
                best_previous_average_score: best_prev_average_score,
                previous_time_reduction: self.previous_time_reduction,
            })
        };
        // The shared `go ponder` signal, seeded active (`ponderMode`), installed on
        // the main worker's control so it (and the coordinator's hold loop) can be
        // driven by a later `ponderhit`. `None` on every non-ponder `go`.
        let ponder = limits.ponder.then(|| Arc::new(PonderSignal::new(true)));
        let control = SearchControl {
            stop: Some(Arc::new(AtomicBool::new(false))),
            ponder: ponder.as_ref().map(Arc::clone),
            #[cfg(feature = "verbose2")]
            node_limit: limits.nodes,
            time,
        };
        // `go depth N` fixes the depth (clamped to the search's own maximum); any
        // other `go` runs to that maximum and is bounded by time / `stop`. Only a
        // `verbose2` build has a source for a lower ceiling, so only it carries
        // one.
        #[cfg(feature = "verbose2")]
        let depth = match limits.depth {
            Some(d) => (d as i32).clamp(1, SEARCH_MAX_DEPTH),
            None => SEARCH_MAX_DEPTH,
        };

        // MultiPV snapshot for this `go` (read per `go`, like the other search
        // options — no global). Clamped to the legal-move count inside the
        // worker. A second PV line is reportable only through the search `info`
        // lines, so without that feature the setting does not exist and the root
        // is single-line.
        #[cfg(feature = "verbose2")]
        let multi_pv = (self.settings.multi_pv().max(1)) as usize;

        // `get_best_thread` is consulted only when no explicit `depth` was
        // given AND `MultiPV == 1` AND this is not a `go mate` search
        // (`MultiPV == 1 && !limits.depth && !limits.mate`). A `go depth N` or
        // any MultiPV > 1 always reports the main worker: a fixed-depth result
        // stays reproducible, and under MultiPV the vote is off so every PV
        // line shows. Under `go mate` the vote is off too — a mate proof lives
        // on the main worker's own line.
        #[cfg(feature = "verbose2")]
        let mate_mode = limits.mate.is_some();
        // Without `verbose2` none of the three can be set — there is no depth
        // ceiling, no MultiPV and no mate mode — so there is nothing to decide
        // and the coordinator consults the vote unconditionally.
        #[cfg(feature = "verbose2")]
        let use_voting = limits.depth.is_none() && multi_pv == 1 && !mate_mode;

        // PV-output config snapshot for this `go`. `computed_pv_interval` is
        // `0` (never suppress — every iteration prints) under `go infinite`,
        // `ConsiderationMode`, or the bench-only `disablePvInterval`; else the
        // `PvInterval` option [ms].
        //
        // Everything here decides which PV lines get printed and nothing else —
        // `MultiPV`, which shapes the search, is passed separately above — so a
        // build that prints none neither reads the settings nor carries the
        // snapshot.
        #[cfg(feature = "verbose2")]
        let pv_config = {
            let consideration_mode = self.settings.consideration_mode();
            let computed_pv_interval =
                if disable_pv_interval || limits.infinite || consideration_mode {
                    Duration::ZERO
                } else {
                    Duration::from_millis(self.settings.pv_interval().max(0) as u64)
                };
            PvOutputConfig {
                pv_interval: computed_pv_interval,
                consideration_mode,
                output_fail_lh_pv: self.settings.output_fail_lh_pv(),
                start_time: now,
            }
        };

        // The worker count and the persistent helper slots to dispatch to. The
        // pool is never resized while a coordinator runs (every resize path calls
        // `finish_search_join` first), so these stay valid for this whole `go`.
        let n_threads = self.pool.size();
        let helper_slots = self.pool.helper_slots();
        // Each helper's node-shared tables: worker `h + 1` gets
        // `worker_shared[h + 1]`, so drop the coordinator's slot-0 handle. The
        // pool never resizes mid-`go`, so `worker_shared` (rebuilt only on a pool
        // rebuild) stays aligned with these helpers for the whole search.
        let helper_shared: Vec<Arc<SharedHistories>> =
            self.worker_shared[1..].iter().map(Arc::clone).collect();
        // Each helper's per-NUMA-node network: worker `h + 1` gets
        // `worker_networks[h + 1]` — its system node's replica. Aligned with
        // `worker_shared` (same per-worker indexing, rebuilt on the same pool
        // rebuilds), so the pool never resizes mid-`go` and these stay valid.
        let helper_networks: Vec<Arc<Search>> =
            self.worker_networks[1..].iter().map(Arc::clone).collect();

        // The shared table is a `static`, so the coordinator gets the same
        // reference and nothing is handed over; the main histories are still
        // lent take-and-return and reclaimed on join. Helper histories live in
        // the pool threads.
        let tt = self.tt;
        let histories = self
            .histories
            .take()
            .expect("session histories present when idle");
        // The coordinator (worker 0) evaluates with its own system node's network
        // replica; unbound / single-node → the one shared instance.
        let search = Arc::clone(&self.worker_networks[0]);
        let writer = Arc::clone(&self.writer);
        let stop = control
            .stop
            .clone()
            .expect("stop flag installed just above");
        let pos = self.pos.clone();

        // Book state for this `go`: the loaded book (cheap `Arc` clone), the
        // `USI_OwnBook` gate, an options snapshot, a fresh seed, and whether a
        // book reply must be held for `stop`/`ponderhit` (`go ponder`/`infinite`).
        let book = self.book.as_ref().map(Arc::clone);
        let own_book = self.settings.usi_own_book();
        let book_config = self.book_config();
        self.book_seed = self
            .book_seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let book_seed = self.book_seed;
        // Whether the coordinator holds its reply (book or searched) until
        // `stop` / `ponderhit`: `go ponder` (until the ponder flag clears) or
        // `go infinite` (the SKIP_SEARCH wait loop). Only `verbose2` has the
        // second of those, so below it `go ponder` is the whole rule.
        #[cfg(feature = "verbose2")]
        let infinite = limits.infinite;
        // The Stochastic_Ponder teardown flag: when set, the coordinator emits
        // no `bestmove` (nor final PV) for this search.
        let suppress_bestmove = Arc::new(AtomicBool::new(false));
        // Raised when this `go`'s reply reaches the output sink; the `tt`
        // commands' idle check is the only reader, so only their feature has it.
        #[cfg(feature = "verbose3")]
        let bestmove_sent = Arc::new(AtomicBool::new(false));

        // Snapshot the entering-king rule for this `go` and precompute its
        // per-side thresholds from the root position, mirroring the reference
        // `set_ekr` on the root worker. The material total is invariant across
        // the search, so every worker shares this one snapshot.
        let entering_king = EnteringKingConfig::new(
            EnteringKingRule::from_option(self.settings.entering_king_rule()),
            &pos,
        );

        // Snapshot the `MaxMovesToDraw` horizon for this `go`, applying the
        // reference's `0 → 100000` remap: a set value of 0 means unlimited.
        // Passed per `go`, like the entering-king config, so every worker
        // shares one value and no global is touched.
        let max_moves_to_draw = remap_max_moves_to_draw(self.settings.max_moves_to_draw());

        // Draw contempt is `drawValueTable[REPETITION_DRAW][us]` for the root
        // side to move; the search returns `+draw_contempt` for the root side
        // and `-draw_contempt` for the opponent.
        let draw_option = match self.pos.side_to_move() {
            yorkie_state::Color::Black => self.settings.draw_value_black(),
            yorkie_state::Color::White => self.settings.draw_value_white(),
        };
        let draw_contempt: Value = (draw_option as Value) * PAWN_VALUE / 100;

        // `ResignValue`: the post-search resign threshold in centipawns.
        // Consumed on the coordinator at emit time.
        let resign_value = self.settings.resign_value() as Value;

        // `GenerateAllLegalMoves`: when true the search also considers the
        // non-promoting moves the default generator suppresses. Every worker
        // shares the flag.
        let generate_all_legal_moves = self.settings.generate_all_legal_moves();

        // The per-`go` coordinator (worker slot 0) binds itself to its assigned
        // NUMA node — and points its allocations at that node's memory — at the
        // start of every `go` when binding is active. The reference binds pool
        // thread 0 once at creation; the port's coordinator is spawned per
        // `go`, so it re-binds each time — same target node, idempotent. `None`
        // (single-node host) → no bind, no policy.
        let numa_bind = self.numa_plan.clone();

        Some(CoordinatorJob {
            search,
            tt,
            pos,
            #[cfg(feature = "verbose2")]
            depth,
            #[cfg(feature = "verbose2")]
            use_voting,
            control,
            stop,
            histories,
            helper_slots,
            helper_shared,
            helper_networks,
            n_threads,
            numa_bind,
            book,
            book_config,
            own_book,
            book_seed,
            ponder,
            #[cfg(feature = "verbose2")]
            infinite,
            suppress_bestmove,
            #[cfg(feature = "verbose3")]
            bestmove_sent,
            entering_king,
            max_moves_to_draw,
            draw_contempt,
            resign_value,
            generate_all_legal_moves,
            #[cfg(feature = "verbose2")]
            mate_mode,
            #[cfg(feature = "random")]
            random_seed: self.random_seed,
            #[cfg(feature = "verbose2")]
            multi_pv,
            #[cfg(feature = "verbose2")]
            pv_config,
            writer,
        })
    }

    /// Run one `bench` position synchronously on the calling thread and return
    /// its total searched node count across all workers.
    ///
    /// Only the driving is synchronous — bench needs each position's node total
    /// before moving on. A position with no network loaded resigns and
    /// contributes 0 nodes.
    #[cfg(feature = "verbose3")]
    fn bench_run_one(&mut self, limits: GoLimits) -> io::Result<u64> {
        // `bench` is `verbose3`, so the PV interval it disables always exists.
        let Some(job) = self.prepare_coordinator_job(limits, true) else {
            diag!(self, "no eval network loaded; run isready")?;
            self.bestmove("resign")?;
            return Ok(0);
        };
        let outcome = run_coordinated(job);
        // Return the session histories the job borrowed (the async path reclaims
        // these on join; here we hand them straight back). Bench uses fixed
        // depth / nodes / movetime, so `time_state` is irrelevant to it.
        self.histories = Some(outcome.histories);
        Ok(outcome.nodes)
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
    fn handle_bench(&mut self, tokens: &[String]) -> io::Result<()> {
        // Reclaim any running search before touching the pool / the TT.
        self.finish_search_join();

        let current = bench::current_sfen(&self.pos);
        let config = match bench::parse_bench(tokens, &current) {
            Ok(c) => c,
            Err(e) => return diag!(self, "bench: {}", e),
        };

        // The thread count is the one value the reference replays as a
        // `setoption` line that still means something here — the table's size
        // is the build's, not the command's. The pool rebuild reports itself
        // exactly as the reference `Threads` on_change callback does.
        self.pool_threads = config.threads.max(1) as usize;
        self.rebuild_pool();
        self.emit_thread_allocation_information()?;

        // The `ucinewgame` (`search_clear`) the reference runs once before the
        // positions: clears the TT, resets histories, and rebuilds the pool — the
        // clean, identical starting state that makes two runs report equal nodes.
        self.handle_usinewgame();
        // That state covers the evaluation noise too: the seed a new game draws
        // would give each run its own node count.
        #[cfg(feature = "random")]
        {
            self.random_seed = BENCH_RANDOM_SEED;
        }

        // The reference resets `elapsed` right after `search_clear`, so the timing
        // excludes the clear itself.
        let start = Instant::now();
        let mut total_nodes: u64 = 0;
        let mut positions: u64 = 0;
        for fen in &config.fens {
            match parse_sfen(fen) {
                Ok(p) => self.pos = p,
                Err(e) => {
                    // A malformed position in a `<fenFile>` is skipped loudly, not
                    // fatal — the rest of the bench still runs.
                    diag!(self, "bench: skipping bad position `{}`: {}", fen, e)?;
                    continue;
                }
            }
            positions += 1;
            total_nodes += self.bench_run_one(config.limits.clone())?;
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
        self.info_string(&format!(
            "bench: positions={positions} nodes={total_nodes} time_ms={time_ms} nps={nps}"
        ))
    }

    fn handle_stop(&mut self) {
        // Signal the running search to abort promptly; it emits its `bestmove`
        // and its state is reclaimed on the next command that needs it (or on
        // `quit`). With no search running this is a silent no-op.
        if let Some(active) = &self.search {
            active.stop.store(true, Ordering::Relaxed);
        }
    }

    /// `gameover [win|lose|draw]`: the game ended. Treated exactly like `stop`:
    /// set the same stop flag, releasing a held book reply
    /// (`go ponder`/`go infinite`) or aborting a running search. Over a shogi
    /// GUI an opponent resign during `go ponder` arrives as `gameover` without
    /// a preceding `stop`; unhandled, pondering would never stop. A no-op when
    /// idle.
    fn handle_gameover(&mut self) {
        self.handle_stop();
    }

    /// `ponderhit`: the opponent played the predicted move.
    ///
    /// Plain path: clear the ponder flag so the pondering search continues under
    /// time management; a held book reply's coordinator wait loop polls the same
    /// flag, so this releases it. Stochastic_Ponder path: tear the rewound
    /// ponder search down without emitting, restore the real position, and
    /// re-issue the retained `go` with `ponder` stripped.
    fn handle_ponderhit(&mut self) -> io::Result<()> {
        let stochastic = self.settings.stochastic_ponder()
            && self.search.as_ref().is_some_and(|a| a.ponder.is_some());
        if stochastic {
            return self.stochastic_ponderhit();
        }

        if let Some(active) = &self.search {
            match &active.ponder {
                // set_ponderhit(false): stamp the ponderhit time and clear the flag.
                Some(p) => p.ponderhit(),
                // Not a ponder search (a stray `ponderhit` during e.g. `go
                // infinite`): fall back to a stop so any held reply is released.
                None => active.stop.store(true, Ordering::Relaxed),
            }
        }
        Ok(())
    }

    /// Stochastic_Ponder `ponderhit`: suppress the rewound ponder search's
    /// output, stop and join it, re-apply the real current position, and
    /// re-issue the retained `go` without its `ponder` token — a normal timed
    /// search of which exactly one `bestmove` reaches the GUI.
    fn stochastic_ponderhit(&mut self) -> io::Result<()> {
        // Suppress the rewound search's bestmove before stopping it.
        if let Some(active) = &self.search {
            active.suppress.store(true, Ordering::Relaxed);
        }
        self.finish_search_join();

        // Re-apply the real (current) position.
        let (sfen, moves) = self.last_position.clone();
        if let Some(pos) = build_position_from(&sfen, &moves) {
            self.pos = pos;
        }

        // Re-issue the retained `go` with `ponder` stripped.
        if let Some(mut go) = self.last_go.clone() {
            go.ponder = false;
            return self.handle_go(go);
        }
        Ok(())
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
    /// "Already replied" is [`ActiveSearch::bestmove_sent`], not
    /// `JoinHandle::is_finished`: the coordinator writes `bestmove` and *then*
    /// unwinds. `is_finished` still stands beside it for the searches that end
    /// without a reply.
    #[cfg(feature = "verbose3")]
    fn handle_tt(&mut self, tokens: &[String]) -> io::Result<()> {
        if self.search.as_ref().is_some_and(|active| {
            active.bestmove_sent.load(Ordering::Relaxed) || active.handle.is_finished()
        }) {
            self.finish_search_join();
        }
        if self.search.is_some() {
            return self.tt_error("a search is running; `stop` it first");
        }

        let command = match parse_tt(tokens) {
            Ok(command) => command,
            Err(e) => return self.tt_error(&e.to_string()),
        };

        match command {
            TtCommand::Store(args) => self.tt_store(&args),
            TtCommand::Probe(position) => self.tt_probe(&position),
            TtCommand::Children(position) => self.tt_children(&position),
        }
    }

    /// The single error channel for the `tt` commands.
    #[cfg(feature = "verbose3")]
    fn tt_error(&self, msg: &str) -> io::Result<()> {
        self.info_string(&format!("tt error: {msg}"))
    }

    /// Build the [`Position`] a `tt` command names.
    ///
    /// The extra king check is this surface's own: `parse_sfen` accepts a
    /// kingless board, but the move generators these commands then run assume
    /// both kings are present.
    #[cfg(feature = "verbose3")]
    fn tt_position(&self, position: &TtPosition) -> Result<Position, String> {
        use yorkie_state::Color;

        let pos = match position {
            TtPosition::StartPos => Position::startpos(),
            TtPosition::Sfen(sfen) => parse_sfen(sfen).map_err(|e| e.to_string())?,
        };
        if pos.king_square(Color::Black).is_none() || pos.king_square(Color::White).is_none() {
            return Err("position has no king for one or both sides".to_string());
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
    fn tt_store(&self, args: &TtStoreArgs) -> io::Result<()> {
        let pos = match self.tt_position(&args.position) {
            Ok(pos) => pos,
            Err(e) => return self.tt_error(&e),
        };
        let mut legal: Vec<Move> = Vec::new();
        pos.generate_legal_all(&mut legal);

        // `none` stores the `MOVE_NONE` fragment, which `TTEntry::save` reads as
        // "keep whatever move this entry already holds for this position".
        let move16 = if args.mv == "none" {
            0
        } else {
            match parse_usi_move(&args.mv, &pos) {
                Ok(mv) if legal.contains(&mv) => mv.move16(),
                Ok(_) => return self.tt_error(&format!("move `{}` is not legal here", args.mv)),
                Err(e) => {
                    return self.tt_error(&format!("move `{}` is not a USI move: {e:?}", args.mv));
                }
            }
        };

        let key = pos.key();
        let side = pos.side_to_move().index() as u8;
        let stored_value = value_to_tt(args.value, 0);
        let generation = self.tt.generation();

        let (_, _, writer) = self.tt.probe(key, side);
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

        // Verify rather than assume. `move16 == 0` is excluded from the
        // comparison on purpose: `save` deliberately preserves the pre-existing
        // move for a `move none` write, so a mismatch there is the documented
        // behaviour, not a declined write.
        let (found, data, _) = self.tt.probe(key, side);
        let stored = found
            && data.value == stored_value
            && data.eval == args.eval
            && data.depth == args.depth
            && data.bound == args.bound
            && data.is_pv == args.pv
            && data.path_dep == args.path_dep
            && (move16 == 0 || data.move16 == move16);
        if stored {
            self.info_string("tt store ok")
        } else {
            self.info_string("tt store skipped (replacement policy kept the existing entry)")
        }
    }

    /// `tt probe …` — read the entry for the named position (`ply == 0`, so the
    /// reported value is exactly the stored one).
    #[cfg(feature = "verbose3")]
    fn tt_probe(&self, position: &TtPosition) -> io::Result<()> {
        let pos = match self.tt_position(position) {
            Ok(pos) => pos,
            Err(e) => return self.tt_error(&e),
        };
        let (found, data, _) = self.tt.probe(pos.key(), pos.side_to_move().index() as u8);
        if !found {
            return self.info_string("tt probe miss");
        }
        let mut legal: Vec<Move> = Vec::new();
        pos.generate_legal_all(&mut legal);
        self.info_string(&format!(
            "tt probe hit {}",
            tt_entry_fields(&data, &legal, 0)
        ))
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
    fn tt_children(&self, position: &TtPosition) -> io::Result<()> {
        let mut pos = match self.tt_position(position) {
            Ok(pos) => pos,
            Err(e) => return self.tt_error(&e),
        };
        let mut legal: Vec<Move> = Vec::new();
        pos.generate_legal_all(&mut legal);

        let mut child_legal: Vec<Move> = Vec::new();
        let mut hits = 0usize;
        for mv in &legal {
            let undo = pos.do_move(*mv);
            let (found, data, _) = self.tt.probe(pos.key(), pos.side_to_move().index() as u8);
            let line = found.then(|| {
                child_legal.clear();
                pos.generate_legal_all(&mut child_legal);
                format!(
                    "tt child {} {}",
                    format_usi_move(*mv),
                    tt_entry_fields(&data, &child_legal, 1)
                )
            });
            pos.undo_move(*mv, undo);
            if let Some(line) = line {
                hits += 1;
                self.info_string(&line)?;
            }
        }
        self.info_string(&format!("tt children end {hits}"))
    }

    #[cfg(feature = "verbose1")]
    fn handle_unknown(&mut self, line: &str) -> io::Result<()> {
        diag!(self, "unknown command: {}", line)
    }

    fn handle_too_long(&mut self) -> io::Result<()> {
        diag!(self, "command too long")
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
fn tt_entry_fields(data: &TTData, legal: &[Move], ply: i32) -> String {
    let mv = legal
        .iter()
        .copied()
        .find(|m| m.move16() == data.move16)
        .map_or_else(|| "none".to_string(), format_usi_move);
    format!(
        "move {mv} value {} depth {} bound {} eval {} pv {} pathdep {}",
        tt_score_field(value_from_tt(data.value, ply)),
        data.depth,
        bound_name(data.bound),
        tt_score_field(data.eval),
        data.is_pv,
        data.path_dep as u8,
    )
}

/// One score field of a `tt` output line: `cp <n>` / `mate <n>` in the same USI
/// scale [`format_score`] gives an `info … score` line, or the literal `none`
/// for the `VALUE_NONE` sentinel (which the search writes into `eval16`
/// whenever a node has no static eval — `tt store` cannot produce it).
#[cfg(feature = "verbose3")]
fn tt_score_field(v: Value) -> String {
    if v == VALUE_NONE {
        "none".to_string()
    } else {
        format_score(v)
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
    let mut ply_digits = NumBuffer::new();
    let mut index_digits = NumBuffer::new();
    let mut node_digits = NumBuffer::new();
    let mut permille_digits = NumBuffer::new();
    let mut clock_digits = NumBuffer::new();

    // Comfortably past the fixed part of the line, so only a long PV regrows.
    let mut body = String::with_capacity(64);
    body.push_str("depth ");
    body.push_str(info.depth.format_into(&mut ply_digits));
    if info.sel_depth != 0 {
        body.push_str(" seldepth ");
        body.push_str(info.sel_depth.format_into(&mut ply_digits));
    }
    body.push_str(" multipv ");
    body.push_str(info.multipv.format_into(&mut index_digits));
    body.push_str(" score ");
    push_score(&mut body, info.score);
    match info.bound {
        PvBound::Lower => body.push_str(" lowerbound"),
        PvBound::Upper => body.push_str(" upperbound"),
        PvBound::Exact => {}
    }
    body.push_str(" nodes ");
    body.push_str(info.nodes.format_into(&mut node_digits));
    body.push_str(" nps ");
    body.push_str(info.nps.format_into(&mut clock_digits));
    body.push_str(" hashfull ");
    body.push_str(info.hashfull.format_into(&mut permille_digits));
    body.push_str(" time ");
    body.push_str(info.time_ms.format_into(&mut clock_digits));
    if !info.pv.is_empty() {
        body.push_str(" pv");
        for m in &info.pv {
            body.push(' ');
            body.push_str(&format_usi_move(*m));
        }
    }
    Formatter::new(w).info(&body)
}

/// A [`PvSink`] that writes each per-iteration / fail-high-low PV line straight
/// to the shared USI output. Installed on the main worker only; helpers and the
/// fixed-depth path get no sink and emit nothing.
///
/// `verbose2` only. Without that feature the main worker is given no sink either,
/// which is what keeps the tournament build's search free of PV work: the
/// search's emission sites are all behind `pv_sink.is_some()`.
#[cfg(feature = "verbose2")]
struct WriterPvSink<W: Write + Send> {
    writer: Arc<Mutex<W>>,
}

#[cfg(feature = "verbose2")]
impl<W: Write + Send> PvSink for WriterPvSink<W> {
    fn emit(&mut self, info: &PvInfo) {
        let mut guard = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let _ = write_pv_info(&mut *guard, info);
    }
}

/// Apply the reference's `MaxMovesToDraw` remap: a set option value of `0`
/// means "unlimited" and is rewritten to `100000` internally; any other value
/// passes through. The option itself still reports `0` — only the search-side
/// horizon uses the remapped value.
fn remap_max_moves_to_draw(option_value: i64) -> i32 {
    if option_value == 0 {
        100_000
    } else {
        option_value as i32
    }
}

/// The two book extensions the reference's name resolution knows about.
const BOOK_EXT_YBB: &str = "ybb";
const BOOK_EXT_DB: &str = "db";

/// True when `path` carries extension `ext`.
///
/// The reference compares the raw suffix, so it is case-sensitive; this port
/// matches case-insensitively throughout, rather than mixing two rules inside
/// one module.
fn has_book_ext(path: &Path, ext: &str) -> bool {
    path.extension()
        .is_some_and(|e| e.eq_ignore_ascii_case(ext))
}

/// Strip a trailing `.db` / `.ybb` from a book name, returning the stem
/// (`book_name_without_extension`).
///
/// Any OTHER name — notably the `no_book` sentinel — yields `None`, which is
/// the reference's empty stem and means "this name has no numbered priority
/// series".
fn book_name_without_extension(name: &Path) -> Option<PathBuf> {
    if has_book_ext(name, BOOK_EXT_DB) || has_book_ext(name, BOOK_EXT_YBB) {
        Some(name.with_extension(""))
    } else {
        None
    }
}

/// `<stem>-<index zero-padded to 3><extension>` (`priority_book_filename`). An
/// index past 999 simply grows past three digits, exactly as the reference's
/// `while (number.size() < 3)` padding does.
fn priority_book_filename(stem: &Path, index: usize, extension: &str) -> PathBuf {
    let mut name = stem.as_os_str().to_os_string();
    name.push(format!("-{index:03}.{extension}"));
    PathBuf::from(name)
}

/// Resolve priority book `index` for `base` (`resolve_priority_book_filename`).
///
/// The primary extension is the base name's own and wins when both files exist,
/// which also produces the reference's `priority book file exists twice` notice,
/// returned as the second tuple element for the caller to emit.
///
/// `None` means neither extension exists at this index, which ends the series.
fn resolve_priority_book_filename(base: &Path, index: usize) -> Option<(PathBuf, Option<String>)> {
    let stem = book_name_without_extension(base)?;

    let (primary_ext, secondary_ext) = if has_book_ext(base, BOOK_EXT_YBB) {
        (BOOK_EXT_YBB, BOOK_EXT_DB)
    } else {
        (BOOK_EXT_DB, BOOK_EXT_YBB)
    };
    let primary = priority_book_filename(&stem, index, primary_ext);
    let secondary = priority_book_filename(&stem, index, secondary_ext);

    if primary.exists() {
        let notice = secondary.exists().then(|| {
            format!(
                "priority book file exists twice. use : {}",
                primary.display()
            )
        });
        return Some((primary, notice));
    }
    if secondary.exists() {
        return Some((secondary, None));
    }
    None
}

/// The Multiple Book priority list for `base` (`get_book_names`): `<stem>-000`,
/// `<stem>-001`, … stopping at the first index where neither extension exists,
/// so a gap ends the series and a `-003` after a missing `-002` is never
/// reached, then the plain `base` appended last.
///
/// The second tuple element carries the `info string` bodies the enumeration
/// produced, in list order, for the caller to emit.
fn book_names(base: &Path) -> (Vec<PathBuf>, Vec<String>) {
    let mut names = Vec::new();
    let mut notices = Vec::new();
    for index in 0.. {
        let Some((name, notice)) = resolve_priority_book_filename(base, index) else {
            break;
        };
        names.push(name);
        notices.extend(notice);
    }
    names.push(base.to_path_buf());
    (names, notices)
}

/// Resolve `<name>.db` whose file is absent to its `<name>.ybb` sibling
/// (`resolve_book_filename_with_ybb_fallback`). Returns the original path when
/// it exists, or when no `.ybb` sibling is present.
fn resolve_book_filename_with_ybb_fallback(requested: &Path) -> PathBuf {
    if requested.exists() {
        return requested.to_path_buf();
    }
    if has_book_ext(requested, BOOK_EXT_DB) {
        let sibling = requested.with_extension(BOOK_EXT_YBB);
        if sibling.exists() {
            return sibling;
        }
    }
    requested.to_path_buf()
}

/// Rebuild a [`Position`] from a parsed `position` command (start / SFEN plus a
/// USI-move list), returning `None` on any parse or legality failure. Used by
/// the Stochastic_Ponder rewind / re-issue paths, which reconstruct a position
/// from the retained [`UsiDriver::last_position`] without the diagnostic side
/// effects of [`UsiDriver::handle_position`].
fn build_position_from(sfen: &PositionSfen, moves: &[String]) -> Option<Position> {
    let mut pos = match sfen {
        PositionSfen::StartPos => Position::startpos(),
        PositionSfen::Sfen(s) => parse_sfen(s).ok()?,
    };
    let mut legal_buf: Vec<Move> = Vec::new();
    for s in moves {
        let parsed = parse_usi_move(s, &pos).ok()?;
        legal_buf.clear();
        pos.generate_legal_all(&mut legal_buf);
        if !legal_buf.contains(&parsed) {
            return None;
        }
        pos.do_move(parsed);
    }
    Some(pos)
}

/// Emit a bare `bestmove <mv>` for the resign / declaration-win short-circuits,
/// which produce no `info` line, preceded by the statistics of the interval it
/// ends. Best-effort: a broken pipe must not panic the coordinator.
///
/// `sent` is raised before the lock is released, so the reply becoming visible
/// and this search counting as finished are one indivisible step downstream.
/// Only the `verbose3` `tt` commands ask that question, so only they carry the
/// flag; the `bestmove` itself is written identically in every build.
fn emit_bestmove<W: Write>(
    writer: &Arc<Mutex<W>>,
    #[cfg(feature = "verbose3")] sent: &AtomicBool,
    mv: &str,
) {
    let mut guard = writer.lock().unwrap_or_else(|e| e.into_inner());
    #[cfg(feature = "verbose1")]
    emit_stats(&mut *guard);
    let _ = Formatter::new(&mut *guard).bestmove(mv);
    #[cfg(feature = "verbose3")]
    sent.store(true, Ordering::Relaxed);
}

/// Take the statistics of the interval that ends here and write their line into
/// an already-locked sink, so it lands directly before the `bestmove` the caller
/// writes next and after any final PV line already out.
///
/// Taking the counters is what starts the next interval, so every reply calls
/// this — including one whose statistics are all zero and therefore print
/// nothing, which would otherwise carry its interval into the following reply.
/// The caller composes its `bestmove` text *before* calling, so the composing's
/// own allocations stay inside the interval being reported.
///
/// Best-effort, like the `bestmove` itself: a broken pipe must not panic the
/// coordinator.
#[cfg(feature = "verbose1")]
fn emit_stats<W: Write + ?Sized>(w: &mut W) {
    let mut buf = StatsBuf::new();
    if let Some(line) = crate::stats::render(&mut buf, take_alloc_count()) {
        let _ = Formatter::new(w).composed_line(line);
    }
}

/// Emit one diagnostic `info string <msg>` from the coordinator (best-effort) —
/// the book-probe notices, which are produced on the search thread and so cannot
/// use the driver's own [`UsiDriver::info_string_diag`].
#[cfg(feature = "verbose1")]
fn emit_info_string_diag<W: Write>(writer: &Arc<Mutex<W>>, msg: &str) {
    let mut guard = writer.lock().unwrap_or_else(|e| e.into_inner());
    let _ = Formatter::new(&mut *guard).info_string(msg);
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
    /// The guard holds the shared writer lock for the whole newline, so a
    /// keep-alive tick can never interleave mid-line with an `info string …` the
    /// heavy work emits concurrently.
    fn spawn<W: Write + Send + 'static>(writer: Arc<Mutex<W>>, poll: Duration) -> Self {
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
                        // A BARE newline (empty line, no `info string` prefix),
                        // routed through the single output sink so it cannot
                        // interleave mid-line with the heavy work's own output.
                        let mut guard = writer.lock().unwrap_or_else(|e| e.into_inner());
                        let _ = Formatter::new(&mut *guard).raw_line("");
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

/// Join book PV moves into a USI ` `-separated string. `verbose2` only — the
/// book `info` lines are its only caller.
#[cfg(feature = "verbose2")]
fn pv_string(pv: &[Move]) -> String {
    pv.iter()
        .map(|m| format_usi_move(*m))
        .collect::<Vec<_>>()
        .join(" ")
}

/// The SKIP_SEARCH hold condition, shared by the book-hit and the searched
/// reply: a `go ponder` holds until its flag clears, a `go infinite` until
/// `stop`. `go infinite` arrives with `verbose2`, so without that feature the
/// ponder flag is the whole condition.
fn reply_is_held(
    ponder: Option<&Arc<PonderSignal>>,
    #[cfg(feature = "verbose2")] infinite: bool,
) -> bool {
    let held = ponder.is_some_and(|p| p.is_active());
    #[cfg(feature = "verbose2")]
    let held = held || infinite;
    held
}

/// Emit a book hit's output the way the reference does on `search_skipped`: one
/// `info` line per surviving candidate, then — after the ponder/infinite hold —
/// a final depth-0 `info` line and the `bestmove [ponder]`.
///
/// Under `go ponder` / `go infinite` the final line and `bestmove` are held
/// until `stop` or `ponderhit`, reusing the async-stop machinery rather than
/// busy-waiting. `time_ms` is stamped once, when the book answered, so the hold
/// does not inflate the elapsed time attributed to the reply; no search ran, so
/// both `nodes` and `nps` are 0 on every line and none of them carries a
/// `seldepth` (the reference's zero `selDepth`, which it omits).
///
/// Both `info` blocks are `verbose2`; the hold and the `bestmove` are not, so
/// a default build answers a book hit with the move and nothing else.
#[allow(clippy::too_many_arguments)]
fn emit_book_hit<W: Write>(
    writer: &Arc<Mutex<W>>,
    hit: &BookHit,
    #[cfg(feature = "verbose2")] hashfull: u32,
    #[cfg(feature = "verbose2")] time_ms: u64,
    ponder: Option<&Arc<PonderSignal>>,
    #[cfg(feature = "verbose2")] infinite: bool,
    stop: &AtomicBool,
    suppress_bestmove: &AtomicBool,
    #[cfg(feature = "verbose3")] sent: &AtomicBool,
) {
    // Per-candidate multipv info lines (emitted immediately, like the reference's
    // in-probe isRoot block).
    #[cfg(feature = "verbose2")]
    {
        let mut guard = writer.lock().unwrap_or_else(|e| e.into_inner());
        let mut f = Formatter::new(&mut *guard);
        for line in &hit.info_lines {
            let body = format!(
                "depth {} multipv {} score {} nodes 0 nps 0 \
                 hashfull {hashfull} time {time_ms} pv {}",
                line.depth,
                line.multipv,
                format_score(Value::from(line.score)),
                pv_string(&line.pv),
            );
            let _ = f.info(&body);
        }
    }

    // `go ponder` / `go infinite`: hold the reply until `stop`, or until a
    // `ponderhit` clears the ponder flag (the SKIP_SEARCH wait loop).
    while !stop.load(Ordering::Relaxed)
        && reply_is_held(
            ponder,
            #[cfg(feature = "verbose2")]
            infinite,
        )
    {
        std::thread::sleep(Duration::from_millis(1));
    }

    // A Stochastic_Ponder teardown suppresses all output for this reply.
    if suppress_bestmove.load(Ordering::Relaxed) {
        return;
    }

    // Final depth-0 info line + bestmove.
    #[cfg(feature = "verbose2")]
    let pv = {
        let mut pv = format_usi_move(hit.best);
        if let Some(p) = hit.ponder {
            pv.push(' ');
            pv.push_str(&format_usi_move(p));
        }
        pv
    };
    let mut bm = format_usi_move(hit.best);
    if let Some(p) = hit.ponder {
        bm.push_str(" ponder ");
        bm.push_str(&format_usi_move(p));
    }
    let mut guard = writer.lock().unwrap_or_else(|e| e.into_inner());
    #[cfg(feature = "verbose2")]
    let _ = Formatter::new(&mut *guard).info(&format!(
        "depth 0 multipv 1 score {} nodes 0 nps 0 \
         hashfull {hashfull} time {time_ms} pv {pv}",
        format_score(Value::from(hit.value)),
    ));
    // After that line, so the statistics cover composing it too, and directly
    // before the reply.
    #[cfg(feature = "verbose1")]
    emit_stats(&mut *guard);
    let _ = Formatter::new(&mut *guard).bestmove(&bm);
    #[cfg(feature = "verbose3")]
    sent.store(true, Ordering::Relaxed);
}

/// Everything one helper needs to run its own iterative deepening for a single
/// `go`. The heavy state is shared: the network, the stop flag and the
/// per-worker node counters behind [`Arc`], the transposition table as the
/// `static` every worker points at. The position and the root-move list are
/// cheap per-helper copies (the reference `start_thinking` copies the root-move
/// list to every worker).
struct HelperJob {
    /// The loaded network holder; the helper borrows `search.network()`.
    search: Arc<Search>,
    /// The shared transposition table — the one `static`, so this is a
    /// reference rather than a handle the helper has to release.
    tt: &'static TranspositionTable,
    /// The root position.
    pos: Position,
    /// This helper's own copy of the root-move list.
    root_moves: Vec<RootMove>,
    /// The iterative-deepening depth ceiling for this `go`, below the search's
    /// own maximum. Only a `verbose2` build has a source for one — see
    /// [`CoordinatorJob::depth`], which this copies.
    #[cfg(feature = "verbose2")]
    limit_depth: i32,
    /// The one shared stop flag every worker polls (the driver installs it).
    stop: Arc<AtomicBool>,
    /// Per-worker node counters; the helper publishes `nodes` to
    /// `node_slots[index]`. The aggregate they form is read by the node ceiling
    /// and by the search `info` lines, both `verbose2`.
    #[cfg(feature = "verbose2")]
    node_slots: Arc<Vec<AtomicU64>>,
    /// Per-worker best-move-change counters; the helper `fetch_add`s its own
    /// `bmc_slots[index]` at the root, and the main worker folds every slot
    /// each iteration.
    bmc_slots: Arc<Vec<AtomicU64>>,
    /// This helper's index into `node_slots` / `bmc_slots` (`>= 1`; index 0 is the
    /// main worker).
    index: usize,
    /// The entering-king declaration config snapshot for this `go`.
    entering_king: EnteringKingConfig,
    /// The `MaxMovesToDraw` horizon for this `go` (already `0 → 100000` remapped).
    max_moves_to_draw: i32,
    /// The root-side draw contempt for this `go` (already pawn-scaled).
    draw_contempt: Value,
    /// `GenerateAllLegalMoves` — expose suppressed non-promotions.
    generate_all_legal_moves: bool,
    /// `go mate` mode — disable early mate break, enable mate-found stop. Only
    /// a `verbose2` build can parse the clause that sets it.
    #[cfg(feature = "verbose2")]
    mate_mode: bool,
    /// The game's evaluation-noise seed, the same one every worker of this `go`
    /// gets, so they agree on each position's offset.
    #[cfg(feature = "random")]
    random_seed: u64,
    /// The raw `MultiPV` option value (helpers run the MultiPV loop too, but never
    /// emit — no sink). Clamped to the legal-move count inside `run_worker`.
    /// Only a build that prints the search `info` lines can report a second PV
    /// line, so only that one searches for any.
    #[cfg(feature = "verbose2")]
    multi_pv: usize,
    /// This helper's node-shared correction / pawn tables — a cheap
    /// [`Arc`] clone of `worker_shared[index]`. Stable across `go`s within a pool
    /// lifetime (the driver rebuilds it only on a pool rebuild, which recreates
    /// the helper threads), so the helper attaches it once to its persistent
    /// per-worker tables.
    shared: Arc<SharedHistories>,
}

/// The state of one helper's coordination slot. The coordinator drives
/// `Parked → Assigned` (dispatch) and `Finished → Parked` (collect); the helper
/// thread drives `Assigned → Running → Finished`. The pool sets `Exit` (only
/// ever over `Parked`, since every teardown path first joins the coordinator,
/// which returns every helper to `Parked`).
enum SlotState {
    /// Idle, waiting for a job.
    Parked,
    /// A job the coordinator posted, not yet picked up. Boxed: a `HelperJob` is
    /// large (a full root position), so keeping it behind a pointer keeps every
    /// other `SlotState` variant small.
    Assigned(Box<HelperJob>),
    /// The helper took the job and is searching.
    Running,
    /// The helper finished; the coordinator has not yet collected the result.
    Finished(WorkerResult),
    /// The pool asked the helper thread to exit.
    Exit,
}

/// One persistent helper's coordination slot: a [`SlotState`] behind a mutex and
/// a condvar both the coordinator and the helper wait on.
struct HelperSlot {
    state: Mutex<SlotState>,
    cv: Condvar,
}

impl HelperSlot {
    fn new() -> Self {
        HelperSlot {
            state: Mutex::new(SlotState::Parked),
            cv: Condvar::new(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, SlotState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Post a search job to a parked helper and wake it.
    fn assign(&self, job: HelperJob) {
        *self.lock() = SlotState::Assigned(Box::new(job));
        self.cv.notify_all();
    }

    /// Block until the helper has finished, then take its result and return the
    /// slot to `Parked`.
    fn collect(&self) -> WorkerResult {
        let mut st = self.lock();
        loop {
            if matches!(&*st, SlotState::Finished(_)) {
                break;
            }
            st = self.cv.wait(st).unwrap_or_else(|e| e.into_inner());
        }
        match std::mem::replace(&mut *st, SlotState::Parked) {
            SlotState::Finished(r) => r,
            _ => unreachable!("slot must be Finished after the wait loop"),
        }
    }
}

/// The persistent helper thread body (the reference `idle_loop`): park until a
/// job (or exit) arrives, run one iterative deepening with the helper's own
/// game-scoped histories, publish the result, and re-park. The histories persist
/// across jobs within a game — the pool is recreated to reset them.
fn helper_loop(slot: Arc<HelperSlot>) {
    // The per-worker tables persist across `go`s (game-scoped); the shared
    // correction / pawn tables are attached from the first job's `shared`
    // handle. Built lazily so the helper never allocates a throwaway
    // single-thread shared set before its node's real one arrives. The handle is
    // stable within a pool lifetime, so it is set once.
    let mut histories: Option<WorkerHistories> = None;
    loop {
        // Park until the coordinator assigns a job or the pool asks us to exit.
        let job = {
            let mut st = slot.lock();
            loop {
                match &*st {
                    SlotState::Assigned(_) => break,
                    SlotState::Exit => return,
                    _ => st = slot.cv.wait(st).unwrap_or_else(|e| e.into_inner()),
                }
            }
            match std::mem::replace(&mut *st, SlotState::Running) {
                SlotState::Assigned(job) => *job,
                _ => unreachable!("slot must be Assigned to break the wait loop"),
            }
        };

        // Attach this helper's node-shared tables on the first job (they are
        // stable within a pool lifetime), then run with the persistent
        // histories. A helper's control is stop-only (no deadlines, no node
        // ceiling): the reference runs `check_time` on the main worker alone;
        // helpers merely poll the shared stop flag.
        let histories_in =
            histories.unwrap_or_else(|| WorkerHistories::with_shared(Arc::clone(&job.shared)));
        let (result, reclaimed) = {
            let net = job.search.network();
            let mut qs = QSearch::with_histories(net, job.tt, histories_in);
            qs.set_control(SearchControl {
                stop: Some(Arc::clone(&job.stop)),
                // Helpers never run `check_time` (only the main worker ponders), so
                // they need no ponder signal — they stop when the coordinator does.
                ponder: None,
                #[cfg(feature = "verbose2")]
                node_limit: None,
                time: None,
            });
            #[cfg(feature = "verbose2")]
            qs.set_node_tally(Arc::clone(&job.node_slots), job.index);
            qs.set_best_move_tally(Arc::clone(&job.bmc_slots), job.index);
            qs.set_entering_king(job.entering_king);
            qs.set_max_moves_to_draw(job.max_moves_to_draw);
            qs.set_draw_value(job.draw_contempt);
            qs.set_generate_all_legal_moves(job.generate_all_legal_moves);
            #[cfg(feature = "verbose2")]
            qs.set_mate_mode(job.mate_mode);
            #[cfg(feature = "random")]
            qs.set_random(RANDOM_AMPLITUDE, job.random_seed);
            // Helpers run the MultiPV loop too, but with no sink they never emit.
            #[cfg(feature = "verbose2")]
            qs.set_multi_pv(job.multi_pv);
            // A ceiling below the search's own maximum has two sources, `go
            // depth N` and the `DepthLimit` key, and both need `verbose2`.
            #[cfg(feature = "verbose2")]
            let limit_depth = job.limit_depth;
            #[cfg(not(feature = "verbose2"))]
            let limit_depth = SEARCH_MAX_DEPTH;
            let result = qs.run_worker(&job.pos, job.root_moves, limit_depth);
            (result, qs.into_histories())
        };
        histories = Some(reclaimed);

        // Release every shared-`Arc` clone this helper holds BEFORE publishing
        // `Finished`, so a reclaim that follows the coordinator's `collect()`
        // finds nothing still held by a helper that has been descheduled
        // mid-teardown. Dropping here, before the `Finished` store that
        // `collect()` synchronizes on, makes the release happen-before the
        // reclaim.
        drop(job.search);
        drop(job.stop);
        #[cfg(feature = "verbose2")]
        drop(job.node_slots);

        *slot.lock() = SlotState::Finished(result);
        slot.cv.notify_all();
    }
}

/// The engine's worker thread pool — the reference `ThreadPool`: a main-worker
/// slot plus `size − 1` persistent helper threads, each parked in
/// [`helper_loop`] until a `go` dispatches it a [`HelperJob`].
///
/// Each helper owns game-scoped histories that persist across `go`s, so the
/// pool is recreated to reset them.
struct ThreadPool {
    /// One coordination slot per helper (`size − 1` of them). Shared with the
    /// coordinator (which dispatches / collects) via [`Self::helper_slots`].
    slots: Vec<Arc<HelperSlot>>,
    /// The helper threads, joined on resize / teardown.
    handles: Vec<JoinHandle<()>>,
}

impl ThreadPool {
    /// Build a pool of `size` slots (one main + `size − 1` helpers), spawning the
    /// helper threads parked and idle. No NUMA binding (the driver uses
    /// [`Self::with_binding`]; the pool unit tests use this).
    #[cfg(test)]
    fn new(size: usize) -> Self {
        Self::with_binding(size, None)
    }

    /// Build a pool of `size` slots with an optional NUMA binding plan. Each
    /// helper thread (worker `1..`) binds itself to its assigned node once at
    /// spawn (mirroring the reference per-thread bind at creation) before it
    /// parks.
    fn with_binding(size: usize, plan: Option<Arc<NumaBindPlan>>) -> Self {
        let mut pool = ThreadPool {
            slots: Vec::new(),
            handles: Vec::new(),
        };
        pool.set_with_binding(size, plan);
        pool
    }

    /// Resize to `size` slots with no binding — used only by the pool unit tests.
    #[cfg(test)]
    fn set(&mut self, size: usize) {
        self.set_with_binding(size, None);
    }

    /// Resize to `size` slots, mirroring the reference `ThreadPool::set`: it
    /// never diffs, always joining and destroying the current helpers and then
    /// recreating the requested number with fresh histories. Callers wait for
    /// any running search to finish first, so every helper is parked when this
    /// runs.
    ///
    /// When `plan` is `Some` and its assignment is non-empty, each helper binds
    /// itself to its assigned NUMA node at spawn and makes that node its
    /// preferred allocation target. The memory half matters because the
    /// affinity pin alone places nothing: under `numactl --interleave=all` the inherited
    /// process policy would still spread the helper's private history tables
    /// across every node. Setting the preference at spawn is what makes
    /// [`helper_loop`]'s lazy allocation land node-locally, and it is per-thread,
    /// so the shared transposition table's interleave is untouched.
    fn set_with_binding(&mut self, size: usize, plan: Option<Arc<NumaBindPlan>>) {
        self.shutdown();
        let size = size.max(1);
        for worker_id in 1..size {
            let slot = Arc::new(HelperSlot::new());
            let slot_for_thread = Arc::clone(&slot);
            let plan_for_thread = plan.clone();
            self.handles.push(std::thread::spawn(move || {
                if let Some(p) = &plan_for_thread
                    && !p.bound.is_empty()
                {
                    p.config.bind_current_thread_with_local_memory(
                        p.bound[worker_id],
                        p.system_nodes[worker_id],
                    );
                }
                helper_loop(slot_for_thread);
            }));
            self.slots.push(slot);
        }
    }

    /// Ask every helper to exit and join it, leaving only the main slot.
    /// Idempotent. Every helper must be parked first (guaranteed by the
    /// `finish_search_join` every caller runs before a resize / teardown), so the
    /// `Exit` is never overwritten by a late `Finished` write.
    fn shutdown(&mut self) {
        for slot in &self.slots {
            *slot.lock() = SlotState::Exit;
            slot.cv.notify_all();
        }
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
        self.slots.clear();
    }

    /// The configured pool size: the main-worker slot plus the live helpers.
    fn size(&self) -> usize {
        self.slots.len() + 1
    }

    /// Clone the helper slot handles so a coordinator can dispatch to and collect
    /// from them for one `go`.
    fn helper_slots(&self) -> Vec<Arc<HelperSlot>> {
        self.slots.iter().map(Arc::clone).collect()
    }
}

impl Drop for ThreadPool {
    /// `quit` / EOF drop the driver, which drops the pool; join every helper so
    /// no OS thread is leaked.
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// The NUMA binding plan shared with the helper threads: the active layout plus
/// the worker → node assignment. Held behind an [`Arc`] so a pool rebuild can
/// cheaply hand each helper thread a clone.
struct NumaBindPlan {
    /// The active NUMA layout, used to resolve a node index to its CPU set.
    config: NumaConfig,
    /// The worker → node assignment (index `i` = worker `i`). Empty means no
    /// binding; `set_with_binding` then leaves every helper unbound.
    bound: Vec<NumaIndex>,
    /// The worker → **system** NUMA node map, aligned with [`Self::bound`]: the
    /// node each worker's private memory should live on. Distinct from `bound`
    /// because L3-aware bundling renumbers logical nodes, while the kernel's
    /// memory policy is indexed by system node
    /// ([`NumaConfig::system_nodes_for_binding`]).
    system_nodes: Vec<NumaIndex>,
}

/// The worker → NUMA-node assignment for `requested` threads under `policy`.
/// When binding is off the assignment is empty.
fn compute_numa_binding(config: &NumaConfig, policy: &str, requested: usize) -> Vec<NumaIndex> {
    let do_bind = match policy {
        "none" => false,
        "auto" => config.suggests_binding_threads(requested),
        // "system", "hardware", or an explicit custom string.
        _ => true,
    };
    if do_bind {
        config.distribute_threads_among_numa_nodes(requested)
    } else {
        Vec::new()
    }
}

/// Wrap a non-empty binding assignment into a shareable [`NumaBindPlan`]; an
/// empty assignment yields `None`, and no thread binds.
///
/// `system_nodes` is the compiled layout's logical → *system* node map. The
/// kernel's memory policy is indexed by system node while `bound` holds
/// *logical* nodes that L3-aware bundling may have renumbered, so the plan
/// carries the per-worker system nodes beside the logical ones, both the same
/// length, and one worker id indexes both.
fn bind_plan(
    config: &NumaConfig,
    bound: &[NumaIndex],
    system_nodes: &[NumaIndex],
) -> Option<Arc<NumaBindPlan>> {
    if bound.is_empty() {
        None
    } else {
        Some(Arc::new(NumaBindPlan {
            config: config.clone(),
            bound: bound.to_vec(),
            system_nodes: worker_system_nodes(bound, system_nodes),
        }))
    }
}

/// How a machine whose layout is `live` differs from the `built` layout — one
/// line, or `None` when they agree.
///
/// Three ways to differ, in the order a reader wants them: a different number of
/// nodes, a node holding different CPUs, and a process denied a CPU the compiled
/// thread plan pins a worker to. The last is what a `taskset` or a `cpuset`
/// around the engine can produce: the machine is right, but a worker would be
/// pinned to a CPU this process is not allowed on, which the pin refuses at the
/// point of no return — inside a spawned worker, mid-game. The question is
/// therefore what `bound` assigns workers to: the CPUs of those nodes must be
/// *allowed* by `affinity`, not equal to it. A wider affinity takes nothing
/// away, since the engine narrows each worker itself; a `bound` that assigns
/// nothing has no CPU to lose and skips the comparison entirely.
///
/// The first two are asked of every build: they compare the machine with the
/// layout the binary was built for, which confining the process does not change.
fn numa_layout_difference(
    live: &NumaLayout,
    affinity: &BTreeSet<usize>,
    bound: &[NumaIndex],
    built: &[&[usize]],
) -> Option<String> {
    if live.nodes.len() != built.len() {
        return Some(format!(
            "this binary is built for {} NUMA node(s), this host has {}",
            built.len(),
            live.nodes.len()
        ));
    }
    for (n, (live_cpus, built_cpus)) in live.nodes.iter().zip(built).enumerate() {
        if live_cpus.as_slice() != *built_cpus {
            return Some(format!(
                "node {n} holds CPUs {} on this host, {} in the layout this binary is built \
                 for",
                yorkie_numa::format_cpu_list(live_cpus.iter().copied()),
                yorkie_numa::format_cpu_list(built_cpus.iter().copied())
            ));
        }
    }
    let missing: BTreeSet<usize> = bound
        .iter()
        // A logical node outside the compiled layout cannot arise: `bound` is an
        // assignment over that same layout's nodes.
        .flat_map(|&logical| built[logical].iter().copied())
        .filter(|cpu| !affinity.contains(cpu))
        .collect();
    if !missing.is_empty() {
        return Some(format!(
            "this process may not run on CPUs {}, which the thread plan uses",
            yorkie_numa::format_cpu_list(missing)
        ));
    }
    None
}

/// Where the shared transposition table's pages belong: the **system** NUMA
/// nodes the compiled thread plan's workers run on, and nothing wider.
#[derive(Debug, PartialEq, Eq)]
enum TablePlacement {
    /// Every worker sits on one node, so the whole table belongs on it.
    OnNode(NumaIndex),
    /// The workers span several nodes, so the table is spread over exactly
    /// those: no node's memory controller carries the whole engine's probe
    /// traffic, and no page lands where no worker probes from.
    AcrossNodes(Vec<NumaIndex>),
    /// Nothing pins a worker, so the engine sets no policy of its own and the
    /// pages land wherever the process's policy puts them. A process confined
    /// to one node — one of many single-thread engines sharing a machine — then
    /// first-touches its table on that node, which is where its only worker
    /// probes from.
    ProcessDefault,
}

/// The placement a binding assignment implies, resolved through the compiled
/// layout's logical → system node map. Ascending and without repeats.
///
/// A binary built for part of a machine — a `numa_policy` node string naming
/// the CPUs of some of its nodes, or a thread count the binding fits on fewer
/// nodes than the machine has — keeps its table on the part it uses, so every
/// probe stays on a node a worker runs on. An empty assignment is the unbound
/// case: no worker is pinned, so where the pages belong is not the engine's
/// question to answer and the process's own policy answers it
/// ([`TablePlacement::ProcessDefault`]).
///
/// Reading the answer off the compiled constants keeps `isready` from asking
/// the machine a second time.
fn table_placement(bound: &[NumaIndex], system_nodes: &[NumaIndex]) -> TablePlacement {
    if bound.is_empty() {
        return TablePlacement::ProcessDefault;
    }
    let mut nodes = worker_system_nodes(bound, system_nodes);
    nodes.sort_unstable();
    nodes.dedup();
    match nodes.as_slice() {
        [node] => TablePlacement::OnNode(*node),
        _ => TablePlacement::AcrossNodes(nodes),
    }
}

/// The *system* NUMA node of every worker in a binding assignment, read off the
/// compiled layout's logical → system node map.
fn worker_system_nodes(bound: &[NumaIndex], system_nodes: &[NumaIndex]) -> Vec<NumaIndex> {
    bound
        .iter()
        .map(|&logical| {
            // A logical node outside the compiled layout cannot arise: `bound`
            // is an assignment over that same layout's nodes.
            system_nodes[logical]
        })
        .collect()
}

/// Worker 0's system NUMA node under `plan`, or `None` when binding is inactive.
fn coordinator_system_node(plan: Option<&Arc<NumaBindPlan>>) -> Option<NumaIndex> {
    plan.and_then(|p| p.system_nodes.first().copied())
}

/// Move the coordinator's session-owned history tables onto worker 0's node.
///
/// This is the one per-worker bundle the engine does **not** allocate inside the
/// worker that uses it: it is built and filled on the USI thread and only then
/// lent to the per-`go` coordinator. Its pages are therefore already faulted
/// wherever the process policy put them, and no per-thread policy can move them
/// retroactively — `mbind(MPOL_BIND | MPOL_MF_MOVE)` can, so that is what this
/// does. The helpers need none of this, each allocating its own bundle after it
/// has pinned itself.
///
/// Best-effort throughout, and run at pool-(re)build time only, outside any
/// clock.
fn place_coordinator_histories(histories: Option<&WorkerHistories>, node: Option<NumaIndex>) {
    let (Some(histories), Some(node)) = (histories, node) else {
        return;
    };
    for (addr, len) in histories.backing_regions() {
        mempolicy::migrate_region_to_node(addr, len, node);
    }
}

/// Build the per-worker handles to the node-shared correction / pawn tables,
/// mirroring the reference per-node construction.
///
/// When `bound` is empty the reference pretends every thread is on node 0;
/// otherwise it counts the assignment. When binding is active the construction
/// runs *inside* a thread bound to that node so the pages first-touch there.
///
/// Returns one [`Arc`] per worker, each pointing at its node's table set, always
/// `requested.max(1)` entries long.
fn build_worker_shared(
    config: &NumaConfig,
    bound: &[NumaIndex],
    requested: usize,
) -> Vec<Arc<SharedHistories>> {
    let requested = requested.max(1);
    let counts = shared_node_counts(bound, requested);
    // Binding active ⇒ allocate + fill each node's set on that node
    // (first-touch); otherwise (single-node) build inline.
    let binding_active = !bound.is_empty();

    let mut node_shared: std::collections::BTreeMap<NumaIndex, Arc<SharedHistories>> =
        std::collections::BTreeMap::new();
    for (&node, &count) in &counts {
        let thread_count = count.next_power_of_two();
        let arc = if binding_active {
            let mut built: Option<Arc<SharedHistories>> = None;
            config.execute_on_numa_node(node, || {
                built = Some(Arc::new(SharedHistories::new(thread_count)));
            });
            built.expect("execute_on_numa_node ran the closure")
        } else {
            Arc::new(SharedHistories::new(thread_count))
        };
        node_shared.insert(node, arc);
    }

    worker_nodes(bound, requested)
        .into_iter()
        .map(|node| Arc::clone(&node_shared[&node]))
        .collect()
}

/// The node → thread-count map for the shared-history construction: when
/// `bound` is empty every thread is pretended to be on node 0
/// (`counts[0] = requested`); otherwise the assignment is counted. Pure — no
/// allocation or binding.
fn shared_node_counts(
    bound: &[NumaIndex],
    requested: usize,
) -> std::collections::BTreeMap<NumaIndex, usize> {
    let mut counts: std::collections::BTreeMap<NumaIndex, usize> =
        std::collections::BTreeMap::new();
    if bound.is_empty() {
        counts.insert(0, requested.max(1));
    } else {
        for &node in bound {
            *counts.entry(node).or_insert(0) += 1;
        }
    }
    counts
}

/// The node each worker's shared table set belongs to: `bound[i]` when binding
/// is active, else node 0 for every worker. Pure — no allocation or binding.
/// Length is `requested.max(1)` (the pool size).
fn worker_nodes(bound: &[NumaIndex], requested: usize) -> Vec<NumaIndex> {
    if bound.is_empty() {
        vec![0; requested.max(1)]
    } else {
        bound.to_vec()
    }
}

/// The *system* NUMA nodes that need their own copy of the network: the
/// distinct nodes the binding assignment's workers run on, in node order.
///
/// Empty when no worker is bound — a single-threaded build, or
/// `numa_policy = "none"` — which is the case where one copy under the
/// process's own memory policy is the right answer, and the operator confining
/// the process is the one who chose where that is.
///
/// Pure, so the set a binding produces is unit-testable without a machine that
/// has those nodes.
fn network_regions(bound: &[NumaIndex], system_nodes: &[usize]) -> Vec<NumaIndex> {
    if bound.is_empty() {
        return Vec::new();
    }
    let mut nodes: Vec<NumaIndex> = worker_system_nodes(bound, system_nodes);
    nodes.sort_unstable();
    nodes.dedup();
    nodes
}

/// Resolve the per-worker network handles for one pool configuration, factored
/// out of [`UsiDriver::rebuild_networks`] so it is unit-testable without a
/// loaded network or a live `/sys` tree.
///
/// `sys_nodes[i]` is worker `i`'s *system* NUMA node, and `by_node` maps a
/// system node to the instance its memory holds. An empty `sys_nodes` or an
/// empty `by_node` means one instance serves every worker.
///
/// A worker whose node no instance was placed for reads the first instance
/// instead: the parameters are identical, so the only cost is that its reads
/// cross to another node. That is reachable only where the worker count grew
/// after the network was placed, which is the measurement command's doing. The
/// result is always `requested` handles long, which is what the per-worker
/// indexing everywhere else assumes.
///
/// Generic over the payload so tests can drive it with a trivial stand-in.
fn resolve_worker_networks<T>(
    instances: &[Arc<T>],
    by_node: &BTreeMap<NumaIndex, usize>,
    sys_nodes: &[NumaIndex],
    requested: usize,
) -> Vec<Arc<T>> {
    let first = match instances.first() {
        Some(first) => first,
        None => return Vec::new(),
    };
    (0..requested.max(1))
        .map(|worker| {
            let index = sys_nodes
                .get(worker)
                .and_then(|sys| by_node.get(sys))
                .copied()
                .unwrap_or(0);
            Arc::clone(instances.get(index).unwrap_or(first))
        })
        .collect()
}

// The thread-allocation diagnostic is emitted by the `verbose3` `bench`
// command and nowhere else, since that is the only command that can change the
// worker count; everything else about the layout is a compile-time constant.
/// The `(bound_count, cpus_in_node)` pairs per node.
///
/// Empty when nothing is bound. Otherwise the pairs cover nodes
/// `0..=highest_bound_node`, then — since at least one thread is bound —
/// extend with `(0, cpus_in_node)` for the remaining nodes up to
/// `num_numa_nodes`.
#[cfg(feature = "verbose3")]
fn bound_thread_counts(cfg: &NumaConfig, bound: &[NumaIndex]) -> Vec<(usize, usize)> {
    if bound.is_empty() {
        return Vec::new();
    }
    let highest = bound.iter().copied().max().unwrap_or(0);
    let mut counts = vec![0usize; highest + 1];
    for &n in bound {
        counts[n] += 1;
    }
    let mut ratios: Vec<(usize, usize)> = Vec::new();
    for (n, &c) in counts.iter().enumerate() {
        ratios.push((c, cfg.num_cpus_in_numa_node(n)));
    }
    // At least one thread is bound (checked above), so extend with the remaining
    // nodes at zero bound threads.
    for n in (highest + 1)..cfg.num_numa_nodes() {
        ratios.push((0, cfg.num_cpus_in_numa_node(n)));
    }
    ratios
}

/// The `a/x:b/y:...` per-node `bound/total` string; empty when nothing is
/// bound.
#[cfg(feature = "verbose3")]
fn thread_binding_information_as_string(cfg: &NumaConfig, bound: &[NumaIndex]) -> String {
    bound_thread_counts(cfg, bound)
        .iter()
        .map(|(current, total)| format!("{current}/{total}"))
        .collect::<Vec<_>>()
        .join(":")
}

/// `"Using N thread[s]"`, plus `" with NUMA node thread binding: a/x:b/y..."`
/// when any thread is bound.
#[cfg(feature = "verbose3")]
fn thread_allocation_information_as_string(
    threads_size: usize,
    cfg: &NumaConfig,
    bound: &[NumaIndex],
) -> String {
    let mut s = format!(
        "Using {threads_size} {}",
        if threads_size > 1 {
            "threads"
        } else {
            "thread"
        }
    );
    let binding = thread_binding_information_as_string(cfg, bound);
    if binding.is_empty() {
        return s;
    }
    s.push_str(" with NUMA node thread binding: ");
    s.push_str(&binding);
    s
}

/// The bundle [`UsiDriver::handle_go`] hands its coordinator thread — grouped
/// into one struct so [`run_coordinated`] stays a single-argument call.
struct CoordinatorJob<W: Write + Send + 'static> {
    search: Arc<Search>,
    tt: &'static TranspositionTable,
    pos: Position,
    /// The iterative-deepening ceiling for this `go`, below the search's own
    /// maximum. `go depth` and the `DepthLimit` key, its only two sources, are
    /// both `verbose2`; without that feature every `go` runs to
    /// [`SEARCH_MAX_DEPTH`] and there is no ceiling to carry.
    #[cfg(feature = "verbose2")]
    depth: i32,
    /// Consult `select_best_worker` (true) or always report the main worker. A
    /// depth ceiling, MultiPV and mate mode are the only three things that turn
    /// the vote off, and all three are `verbose2`; without that feature the
    /// coordinator has no choice to carry and always votes.
    #[cfg(feature = "verbose2")]
    use_voting: bool,
    /// The main worker's full control (stop + node ceiling + deadlines).
    control: SearchControl,
    /// The one shared stop flag.
    stop: Arc<AtomicBool>,
    /// The main worker's game-scoped histories, returned to the driver on join.
    histories: WorkerHistories,
    /// The persistent helper slots to dispatch to (`n_threads − 1` of them).
    helper_slots: Vec<Arc<HelperSlot>>,
    /// Each helper's node-shared correction / pawn tables, aligned
    /// with `helper_slots`: `helper_shared[h]` is worker `h + 1`'s
    /// [`SharedHistories`]. Handed to the helper in its [`HelperJob`]. The main
    /// worker's own shared handle already lives inside `histories`.
    helper_shared: Vec<Arc<SharedHistories>>,
    /// Each helper's per-NUMA-node network replica, aligned with
    /// `helper_slots`: `helper_networks[h]` is worker `h + 1`'s [`Search`]. Handed
    /// to the helper in its [`HelperJob`]. The main worker's own replica is
    /// `search`. When replication is inactive every entry is a clone of the one
    /// loaded instance.
    helper_networks: Vec<Arc<Search>>,
    /// The worker count (main + helpers).
    n_threads: usize,
    /// The active binding plan, or `None` when binding is inactive. When set,
    /// the coordinator pins itself to worker 0's logical node and prefers that
    /// worker's system node for memory at the start of this `go`.
    numa_bind: Option<Arc<NumaBindPlan>>,
    /// The loaded opening book to probe once, if any.
    book: Option<Arc<LoadedBook>>,
    /// The book-selection config snapshot for this `go`.
    book_config: BookConfig,
    /// `USI_OwnBook` — the master gate; when off the book is never probed.
    own_book: bool,
    /// The seed for this `go`'s book PRNG (deterministic within a session).
    book_seed: u64,
    /// The shared `go ponder` signal (`Some` only for a `go ponder`): the
    /// coordinator's hold loop runs while it is active, and `bestmove` is withheld
    /// until a `ponderhit` clears it (or `stop` fires).
    ponder: Option<Arc<PonderSignal>>,
    /// `limits.infinite` — hold the reply until `stop` regardless of the clock
    /// (the SKIP_SEARCH wait loop). Only a `verbose2` build can parse the
    /// clause that sets it.
    #[cfg(feature = "verbose2")]
    infinite: bool,
    /// The Stochastic_Ponder teardown flag: when set the coordinator emits no
    /// `bestmove` (nor final PV) for this search.
    suppress_bestmove: Arc<AtomicBool>,
    /// Stamped `true` in the same output-lock critical section that writes this
    /// search's `bestmove`, so the driver can tell "reply is out" from "thread
    /// has exited" (see [`ActiveSearch::bestmove_sent`]). A suppressed reply
    /// never sets it — nothing went out. `verbose3`, like the reader.
    #[cfg(feature = "verbose3")]
    bestmove_sent: Arc<AtomicBool>,
    /// The entering-king declaration config snapshot for this `go`.
    entering_king: EnteringKingConfig,
    /// The `MaxMovesToDraw` horizon for this `go` (already `0 → 100000` remapped).
    max_moves_to_draw: i32,
    /// The root-side draw contempt `drawValueTable[REPETITION_DRAW][us]` for this
    /// `go` (`DrawValueBlack`/`DrawValueWhite`, already pawn-scaled).
    draw_contempt: Value,
    /// The `ResignValue` threshold in centipawns; a searched best score at or
    /// below `-resign_value` resigns.
    resign_value: Value,
    /// `GenerateAllLegalMoves` — expose the suppressed non-promotions to the
    /// search generators.
    generate_all_legal_moves: bool,
    /// `go mate` mode — disables the early mate break and enables the mate-found
    /// stop rule. Only a `verbose2` build can parse the clause that sets it.
    #[cfg(feature = "verbose2")]
    mate_mode: bool,
    /// The evaluation-noise seed for the game this `go` belongs to. Every worker
    /// this coordinator dispatches is given the same one.
    #[cfg(feature = "random")]
    random_seed: u64,
    /// The raw `MultiPV` option value for this `go`. It shapes the search, and
    /// only a build that prints the search `info` lines can report a second PV
    /// line, so only that one carries it.
    #[cfg(feature = "verbose2")]
    multi_pv: usize,
    /// The PV-output config for this `go` — what gets printed, in a build that
    /// prints anything.
    #[cfg(feature = "verbose2")]
    pv_config: PvOutputConfig,
    /// The shared output sink for the per-iteration / final `info` / `bestmove`.
    writer: Arc<Mutex<W>>,
}

/// The Lazy-SMP coordinator — the reference main worker's `start_searching`,
/// running on the per-`go` thread the driver spawns.
///
/// Hands back the main worker's histories for the driver to reclaim, the
/// aggregate searched-node total (0 for the short-circuits, and what `bench`
/// accumulates), and the time-management carry-forward, whose third element is
/// `None` for a short-circuited `go`.
struct CoordinatedOutcome {
    histories: WorkerHistories,
    /// `bench` is the only reader (the async `go` path takes its node total off
    /// the wire), and `bench` is `verbose3`, so a build below it neither carries
    /// nor sums the total.
    #[cfg(feature = "verbose3")]
    nodes: u64,
    time_state: Option<(Value, Value, Option<f64>)>,
}

/// The time-management carry-forward for a search-skipping short-circuit — a
/// book hit, declaration win, resign or no legal move.
///
/// The reference's `SKIP_SEARCH:` falls straight through to the same bookkeeping
/// a real search reaches, where `rootMoves[0]` is still the unsearched default,
/// so both carried scores are `-VALUE_INFINITE`. Only `previousTimeReduction` is
/// left untouched, since `iterative_deepening` — its sole writer — did not run.
fn skip_search_carry() -> Option<(Value, Value, Option<f64>)> {
    Some((-VALUE_INFINITE, -VALUE_INFINITE, None))
}

fn run_coordinated<W: Write + Send + 'static>(job: CoordinatorJob<W>) -> CoordinatedOutcome {
    let CoordinatorJob {
        search,
        tt,
        pos,
        #[cfg(feature = "verbose2")]
        depth,
        #[cfg(feature = "verbose2")]
        use_voting,
        control,
        stop,
        histories,
        helper_slots,
        helper_shared,
        helper_networks,
        n_threads,
        numa_bind,
        book,
        book_config,
        own_book,
        book_seed,
        ponder,
        #[cfg(feature = "verbose2")]
        infinite,
        suppress_bestmove,
        #[cfg(feature = "verbose3")]
        bestmove_sent,
        entering_king,
        max_moves_to_draw,
        draw_contempt,
        resign_value,
        generate_all_legal_moves,
        #[cfg(feature = "verbose2")]
        mate_mode,
        #[cfg(feature = "random")]
        random_seed,
        #[cfg(feature = "verbose2")]
        multi_pv,
        #[cfg(feature = "verbose2")]
        pv_config,
        writer,
    } = job;
    #[cfg(feature = "verbose2")]
    let multi_pv = multi_pv.max(1);

    // Bind this coordinator to its assigned NUMA node before any search work,
    // and make that node's memory its allocation preference, so everything this
    // thread allocates from here on stays node-local instead of following the
    // launcher's process-wide interleave. Idempotent across the per-`go`
    // coordinator respawns. The bundle in `histories` is placed separately, at
    // pool (re)build time, because it was already faulted on the USI thread.
    if let Some(plan) = &numa_bind {
        plan.config
            .bind_current_thread_with_local_memory(plan.bound[0], plan.system_nodes[0]);
    }

    // One TT generation bump per `go`, on the main worker, BEFORE any helper
    // starts, so the observable single-thread sequence is the reference's:
    // bump, then search.
    tt.new_search();

    // Build the root-move list once (the reference `start_thinking`). The resign
    // and declaration-win short-circuits emit and return before any helper is
    // dispatched, exactly as `start_searching` exits before `threads.start_searching()`.
    let root_moves = generate_root_moves(&pos, generate_all_legal_moves);
    if root_moves.is_empty() {
        emit_bestmove(
            &writer,
            #[cfg(feature = "verbose3")]
            &bestmove_sent,
            "resign",
        );
        return CoordinatedOutcome {
            histories,
            #[cfg(feature = "verbose3")]
            nodes: 0,
            time_state: skip_search_carry(),
        };
    }
    // Rule-aware declaration shortcut. Point / `None` rules yield
    // `Move::win()` (emitted as the bare `win` token); `TryRule`
    // yields the actual king move onto the try square, which must be emitted
    // verbatim so the host plays it.
    if let Some(mv) = declaration_win(&pos, &entering_king) {
        if mv == Move::win() {
            emit_bestmove(
                &writer,
                #[cfg(feature = "verbose3")]
                &bestmove_sent,
                "win",
            );
        } else {
            emit_bestmove(
                &writer,
                #[cfg(feature = "verbose3")]
                &bestmove_sent,
                &format_usi_move(mv),
            );
        }
        return CoordinatedOutcome {
            histories,
            #[cfg(feature = "verbose3")]
            nodes: 0,
            time_state: skip_search_carry(),
        };
    }

    // Opening-book probe — once, on the coordinator, BEFORE any helper starts
    // (the on-the-fly read path is not thread-safe by design). The
    // `USI_OwnBook` gate and a loaded book are both required. On a hit we emit
    // and return without searching, holding the reply for `go ponder` /
    // `go infinite`.
    if own_book && let Some(loaded) = &book {
        let mut prng = Prng::new(book_seed);
        let probed = probe_book(
            &loaded.books,
            loaded.ignore_book_ply,
            &pos,
            &book_config,
            &mut prng,
        );
        #[cfg(feature = "verbose1")]
        for diag in &probed.diagnostics {
            emit_info_string_diag(&writer, diag);
        }
        if let Some(hit) = probed.hit {
            // `tm.elapsed_time()` at the moment the book answered, floored at 1
            // — the `time` the reply's `info` lines carry.
            #[cfg(feature = "verbose2")]
            let book_time_ms = (Instant::now()
                .saturating_duration_since(pv_config.start_time)
                .as_millis() as u64)
                .max(1);
            emit_book_hit(
                &writer,
                &hit,
                #[cfg(feature = "verbose2")]
                tt.hashfull(0),
                #[cfg(feature = "verbose2")]
                book_time_ms,
                ponder.as_ref(),
                #[cfg(feature = "verbose2")]
                infinite,
                &stop,
                &suppress_bestmove,
                #[cfg(feature = "verbose3")]
                &bestmove_sent,
            );
            return CoordinatedOutcome {
                histories,
                #[cfg(feature = "verbose3")]
                nodes: 0,
                time_state: skip_search_carry(),
            };
        }
    }

    // Per-worker node counters (index 0 = main, 1.. = helpers) for the aggregate
    // `go nodes N` ceiling and the final aggregated `info ... nodes`. Both
    // readers are `verbose2` — the ceiling with the clauses and keys that set
    // it, the `info` line with every search line — and each worker's own
    // `nodes`, which the search itself reads, is a counter of its own.
    #[cfg(feature = "verbose2")]
    let node_slots: Arc<Vec<AtomicU64>> =
        Arc::new((0..n_threads).map(|_| AtomicU64::new(0)).collect());
    // Per-worker best-move-change counters, same slot-per-worker shape: each
    // worker bumps its own slot at the root, the main worker folds them all
    // each iteration. Fresh (all-zero) per `go`.
    let bmc_slots: Arc<Vec<AtomicU64>> =
        Arc::new((0..n_threads).map(|_| AtomicU64::new(0)).collect());

    // Dispatch a job to every helper (index h in `helper_slots` → worker h + 1).
    for (h, slot) in helper_slots.iter().enumerate() {
        slot.assign(HelperJob {
            // Worker `h + 1`'s own system node's network replica.
            search: Arc::clone(&helper_networks[h]),
            tt,
            pos: pos.clone(),
            root_moves: root_moves.clone(),
            #[cfg(feature = "verbose2")]
            limit_depth: depth,
            stop: Arc::clone(&stop),
            #[cfg(feature = "verbose2")]
            node_slots: Arc::clone(&node_slots),
            bmc_slots: Arc::clone(&bmc_slots),
            index: h + 1,
            entering_king,
            max_moves_to_draw,
            draw_contempt,
            generate_all_legal_moves,
            #[cfg(feature = "verbose2")]
            mate_mode,
            #[cfg(feature = "random")]
            random_seed,
            #[cfg(feature = "verbose2")]
            multi_pv,
            shared: Arc::clone(&helper_shared[h]),
        });
    }

    // The main worker is the only one given a PV sink, and only a `verbose2`
    // build has one to give. `MultiPV` rides on the same feature for a different
    // reason: it shapes the search, but the extra lines it searches are
    // reportable only through the `info` lines that feature brings. Without it
    // the root is single-line and the emission sites are not compiled at all, so
    // the search of the first line is identical in all three build shapes.
    let net = search.network();
    let mut qs = QSearch::with_histories(net, tt, histories);
    qs.set_control(control);
    #[cfg(feature = "verbose2")]
    qs.set_node_tally(Arc::clone(&node_slots), 0);
    qs.set_best_move_tally(Arc::clone(&bmc_slots), 0);
    qs.set_entering_king(entering_king);
    qs.set_max_moves_to_draw(max_moves_to_draw);
    qs.set_draw_value(draw_contempt);
    qs.set_generate_all_legal_moves(generate_all_legal_moves);
    #[cfg(feature = "verbose2")]
    qs.set_mate_mode(mate_mode);
    #[cfg(feature = "random")]
    qs.set_random(RANDOM_AMPLITUDE, random_seed);
    #[cfg(feature = "verbose2")]
    qs.set_multi_pv(multi_pv);
    #[cfg(feature = "verbose2")]
    qs.set_pv_output(
        pv_config,
        Box::new(WriterPvSink {
            writer: Arc::clone(&writer),
        }),
    );
    // As in the helper loop: without `verbose2` no `go` carries a ceiling below
    // the search's own maximum, because neither source of one exists.
    #[cfg(not(feature = "verbose2"))]
    let depth = SEARCH_MAX_DEPTH;
    let main_result = qs.run_worker(&pos, root_moves, depth);

    // Ponder / infinite hold (the SKIP_SEARCH wait loop): do not emit
    // `bestmove` while still pondering or under `go infinite`. A plain
    // `ponderhit` clears the ponder flag mid-search, so the main worker usually
    // returns already un-pondering; this catches the case where the search
    // finished (mate found / depth ceiling) while a `ponderhit` had not yet
    // arrived.
    while !stop.load(Ordering::Relaxed)
        && reply_is_held(
            ponder.as_ref(),
            #[cfg(feature = "verbose2")]
            infinite,
        )
    {
        std::thread::sleep(Duration::from_millis(1));
    }

    // Signal the helpers the search is over, then wait for and collect each one.
    // They observe the shared stop at their next checkpoint and finish promptly.
    stop.store(true, Ordering::Relaxed);
    let mut results: Vec<WorkerResult> = Vec::with_capacity(n_threads);
    results.push(main_result);
    for slot in &helper_slots {
        results.push(slot.collect());
    }

    // Aggregated node count for the `info` line and for `bench` — every worker's
    // exact final count (the reference `threads.nodes_searched()`). Both readers
    // are gated, so a build with neither does not sum.
    #[cfg(feature = "verbose2")]
    let total_nodes: u64 = results.iter().map(|r| r.nodes).sum();

    // Choose the reported worker: the main worker for a `go depth N`, else the
    // thread vote (`get_best_thread`). Nothing turns the vote off in a build
    // without `verbose2`, which reaches it for every `go`; the label is the
    // early exit only a `verbose2` build has.
    #[cfg_attr(not(feature = "verbose2"), allow(unused_labels))]
    let chosen = 'vote: {
        #[cfg(feature = "verbose2")]
        if !use_voting {
            break 'vote 0;
        }
        let votes: Vec<WorkerVote> = results
            .iter()
            .map(|r| WorkerVote {
                score: r.best.score,
                pv0: r.best.pv[0],
                pv_len: r.best.pv.len(),
                completed_depth: r.completed_depth,
            })
            .collect();
        select_best_worker(&votes)
    };
    let chosen_result = &results[chosen];
    let mut best = chosen_result.best.clone();
    let ponder_candidate = chosen_result.ponder_candidate;
    // The chosen worker's MultiPV lines and completed depth feed the final `info`
    // PV block and nothing else, so a default build neither clones nor keeps
    // them.
    #[cfg(feature = "verbose2")]
    let (mut pv_lines, completed_depth) = (
        chosen_result.pv_lines.clone(),
        chosen_result.completed_depth.max(1),
    );

    // Time-management outputs carried back for the next `go`: the chosen
    // worker's score / average score become the next move's `bestPrevious*`,
    // and the main worker's final `timeReduction` becomes
    // `previousTimeReduction`.
    let out_best_previous_score = chosen_result.best.score;
    let out_best_previous_average_score = chosen_result.best.average_score;
    let out_previous_time_reduction = results[0].time_reduction;

    // Ponder-extend the CHOSEN worker's length-1 PV via the shared TT.
    #[cfg(feature = "verbose2")]
    let ponder_before = best.pv.len();
    let mut work = pos.clone();
    qs.extract_ponder(&mut work, &mut best, ponder_candidate);
    #[cfg(feature = "verbose2")]
    let ponder_extended = best.pv.len() != ponder_before;

    // `ResignValue` is decided *before* the final PV output, because a
    // resign-by-value forces that PV out so the GUI can see the score behind the
    // decision. The reference judges on `rootMoves[0].uciScore` normalized to
    // centipawns, not on the raw internal score, and an unset `uciScore` maps to
    // `VALUE_ZERO` rather than resigning outright.
    let resign_by_value = best.score != -VALUE_INFINITE && {
        let resign_score = if best.uci_score == -VALUE_INFINITE {
            0
        } else {
            best.uci_score
        };
        to_cp(resign_score) <= -resign_value
    };

    // Final PV output before `bestmove`. `pv_idx == lines.len()` makes every
    // line exact, matching the reference's `pv()` after the MultiPV loop.
    //
    // `verbose2` only; `resign_by_value` above is not gated, since it decides
    // the reply itself and a build that prints no PV still resigns on the same
    // score.
    #[cfg(feature = "verbose2")]
    {
        // `uciPvSent` is the main worker's flag; it is cleared when the chosen
        // PV was ponder-extended, or when the chosen worker is not the main one
        // and so never emitted its PV.
        let uci_pv_sent = results[0].uci_pv_sent && !ponder_extended && chosen == 0;
        if !uci_pv_sent || resign_by_value {
            // Reflect the (possibly ponder-extended) chosen line back into line 0
            // so this re-emits the exact PV that `bestmove [ponder]` will play.
            if let Some(line0) = pv_lines.get_mut(0) {
                *line0 = best.clone();
            }
            let n = pv_lines.len();
            let infos = qs.build_pv_infos(&pos, &pv_lines, n, completed_depth, n, total_nodes);
            let mut guard = writer.lock().unwrap_or_else(|e| e.into_inner());
            for info in &infos {
                let _ = write_pv_info(&mut *guard, info);
            }
        }
    }

    // `bestmove [ponder]` — the ponder move is the chosen line's second PV move.
    let mut bm = format_usi_move(best.mv);
    if best.pv.len() >= 2 {
        bm.push_str(" ponder ");
        bm.push_str(&format_usi_move(best.pv[1]));
    }

    // Resigning replaces the whole reply (the reference makes the search look
    // skipped and stacks `Move::resign()`), so it carries no ponder move.
    if resign_by_value {
        bm = "resign".to_string();
    }

    // A Stochastic_Ponder teardown stops the rewound search without emitting
    // its `bestmove`; the fresh re-issued `go` produces the single reply the
    // GUI sees. The `time_state` below is still returned so the rewound
    // search's score / ply seed the re-issue's side-flip continuity.
    if !suppress_bestmove.load(Ordering::Relaxed) {
        emit_bestmove(
            &writer,
            #[cfg(feature = "verbose3")]
            &bestmove_sent,
            &bm,
        );
    }

    // Consume the driver (ending the `&tt` / `&net` borrows) and reclaim the main
    // worker's histories for the driver, paired with the aggregate node total for
    // the `bench` accumulation and the time-management carry-forward.
    CoordinatedOutcome {
        histories: qs.into_histories(),
        #[cfg(feature = "verbose3")]
        nodes: total_nodes,
        time_state: Some((
            out_best_previous_score,
            out_best_previous_average_score,
            // A real search produced a fresh `timeReduction`
            // (`mainThread->previousTimeReduction`).
            Some(out_previous_time_reduction),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive a full canned session in-process and return everything written.
    ///
    /// The output sink is an `Arc<Mutex<Vec<u8>>>` shared with the driver (and,
    /// during a `go`, its search worker); after `run` returns — which joins any
    /// worker — the buffer holds the complete transcript.
    fn run_with(input: &str) -> String {
        let output = Arc::new(Mutex::new(Vec::<u8>::new()));
        let driver = UsiDriver::new(input.as_bytes(), Arc::clone(&output));
        driver.run().expect("driver run");
        let bytes = output.lock().expect("output lock").clone();
        String::from_utf8(bytes).expect("utf-8")
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
                .map(|s| parse_usi_move(s, &pos).expect("fixture move parses"))
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

        let pos = parse_sfen("4k4/9/9/9/9/9/9/9/4K4 b P 1").expect("sfen parses");
        info.pv = vec![parse_usi_move("P*5e", &pos).expect("drop parses")];
        info.hashfull = 1000;
        assert_eq!(
            pv_line(&info),
            "info depth 0 multipv 1 score cp 0 nodes 0 nps 0 hashfull 1000 time 1 pv P*5e\n"
        );
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn quit_returns_immediately() {
        assert_eq!(run_with("quit\n"), "");
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn eof_returns_ok() {
        assert_eq!(run_with(""), "");
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn isready_without_network_reports_load_failure() {
        // Nothing staged an evaluation file where this driver looks — beside
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
            let keep_alive = KeepAlive::spawn(Arc::clone(&writer), Duration::from_millis(1));
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
                    .info_string("busy")
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
            let _keep_alive = KeepAlive::spawn(Arc::clone(&writer), poll);
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
        assert_eq!(run_with("usinewgame\nquit\n"), "");
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn unknown_command_echoes_back() {
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
        assert_eq!(run_with("position startpos\nquit\n"), "");
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn position_sfen_startpos_silent() {
        let sfen = yorkie_state::STARTPOS_SFEN;
        assert_eq!(run_with(&format!("position sfen {sfen}\nquit\n")), "");
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn position_startpos_moves_silent() {
        assert_eq!(run_with("position startpos moves 7g7f\nquit\n"), "");
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn position_sfen_malformed_emits_info_string() {
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
        // Apply a legal move; then send a malformed sfen; then `go`. The reply
        // must be a legal move from the *post-7g7f* position, not from startpos
        // — proving the malformed line did not clobber the driver's state.
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

    #[cfg_attr(miri, ignore)]
    #[test]
    fn go_without_network_resigns_with_notice() {
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
        // Whatever subset of GoLimits the host provides, the driver parses and
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
        assert_eq!(
            run_with("bench 16 1 6 default depth\nquit\n"),
            diag("unknown command: bench 16 1 6 default depth")
        );
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn stop_is_silent() {
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
        let out = run_with(
            "bench 1 1 1 current movetime\n\
             bench 1 4 1 current movetime\n\
             bench 1 2 1 current movetime\n\
             quit\n",
        );
        // Prefix matches: on a machine whose layout suggests binding, the line
        // carries a `with NUMA node thread binding: ...` suffix.
        assert!(out.contains("info string Using 1 thread"), "{out}");
        assert!(out.contains("info string Using 4 threads"), "{out}");
        assert!(out.contains("info string Using 2 threads"), "{out}");
    }

    // -- the compiled layout, and the machine it is held against ---------

    /// A layout with `nodes` as its logical nodes, every one of them its own
    /// system node.
    fn layout(nodes: &[&[usize]]) -> NumaLayout {
        NumaLayout {
            nodes: nodes.iter().map(|cpus| cpus.to_vec()).collect(),
            system_nodes: (0..nodes.len()).collect(),
            custom_affinity: false,
        }
    }

    fn affinity(cpus: &[usize]) -> BTreeSet<usize> {
        cpus.iter().copied().collect()
    }

    #[test]
    fn a_machine_matching_the_compiled_layout_is_no_difference() {
        let built: &[&[usize]] = &[&[0, 1], &[2, 3]];
        assert_eq!(
            numa_layout_difference(&layout(built), &affinity(&[0, 1, 2, 3]), &[0, 1], built),
            None
        );
    }

    #[test]
    fn a_machine_with_another_node_count_is_reported_with_both_counts() {
        let built: &[&[usize]] = &[&[0, 1], &[2, 3]];
        let live = layout(&[&[0, 1, 2, 3]]);
        let msg = numa_layout_difference(&live, &affinity(&[0, 1, 2, 3]), &[0, 1], built)
            .expect("one node is not two");
        assert!(msg.contains("built for 2 NUMA node(s)"), "message: {msg}");
        assert!(msg.contains("this host has 1"), "message: {msg}");
    }

    #[test]
    fn a_node_holding_other_cpus_is_reported_with_both_cpu_lists() {
        let built: &[&[usize]] = &[&[0, 1], &[2, 3]];
        let live = layout(&[&[0, 1], &[2, 3, 4]]);
        let msg = numa_layout_difference(&live, &affinity(&[0, 1, 2, 3, 4]), &[0, 1], built)
            .expect("node 1 grew a CPU");
        assert!(msg.contains("node 1"), "message: {msg}");
        assert!(msg.contains("2-4"), "message: {msg}");
        assert!(msg.contains("2-3"), "message: {msg}");
    }

    #[test]
    fn a_process_denied_a_cpu_the_plan_uses_is_refused_with_those_cpus() {
        // The machine is the one the binary was built for; the *process* is not
        // allowed on every CPU the plan pins a worker to — a `taskset` around
        // the engine, whose workers would then pin themselves to CPUs they may
        // not run on. Only the CPUs it is missing are named.
        let built: &[&[usize]] = &[&[0, 1], &[2, 3]];
        let msg = numa_layout_difference(&layout(built), &affinity(&[0, 1]), &[0, 1], built)
            .expect("half the plan's CPUs are hidden");
        assert!(msg.contains("may not run on CPUs 2-3"), "message: {msg}");
        assert!(msg.contains("which the thread plan uses"), "message: {msg}");

        let msg = numa_layout_difference(&layout(built), &affinity(&[0, 1, 2]), &[0, 1], built)
            .expect("one of the plan's CPUs is hidden");
        assert!(msg.contains("may not run on CPUs 3"), "message: {msg}");
    }

    #[test]
    fn an_affinity_covering_the_plans_cpus_is_no_difference() {
        // A binary built for part of the machine, started with no confinement:
        // its plan uses node 0 and the process may run on everything. The CPUs
        // beyond the plan take nothing away, since the worker narrows itself to
        // the node it was assigned. An affinity holding exactly the plan's CPUs
        // is equally fine.
        let built: &[&[usize]] = &[&[0, 1], &[2, 3]];
        assert_eq!(
            numa_layout_difference(&layout(built), &affinity(&[0, 1, 2, 3]), &[0, 0], built),
            None
        );
        assert_eq!(
            numa_layout_difference(&layout(built), &affinity(&[0, 1]), &[0, 0], built),
            None
        );
    }

    #[test]
    fn a_confined_process_is_accepted_when_the_plan_pins_no_worker() {
        // The build that binds nothing — one thread, or `numa_policy = "none"`
        // — has no worker to pin, so the CPUs the process was denied are CPUs
        // it was never going to use. An empty assignment is how the caller says
        // so, and the machine comparisons are unaffected by it: a wrong machine
        // is still refused, confined or not.
        let built: &[&[usize]] = &[&[0, 1], &[2, 3]];
        assert_eq!(
            numa_layout_difference(&layout(built), &affinity(&[0]), &[], built),
            None
        );

        let live = layout(&[&[0, 1, 2, 3]]);
        let msg = numa_layout_difference(&live, &affinity(&[0]), &[], built)
            .expect("one node is still not two nodes");
        assert!(msg.contains("built for 2 NUMA node(s)"), "message: {msg}");
    }

    #[cfg(feature = "verbose3")]
    #[test]
    fn info_strings_exact_formats() {
        let cfg = NumaConfig::from_string("0-3,8:16-31").unwrap();

        // No binding → bare `Using N thread[s]`, singular/plural.
        assert_eq!(
            thread_allocation_information_as_string(1, &cfg, &[]),
            "Using 1 thread"
        );
        assert_eq!(
            thread_allocation_information_as_string(2, &cfg, &[]),
            "Using 2 threads"
        );

        // Binding across two equal 2-CPU nodes → `a/x:b/y` suffix.
        let two = NumaConfig::from_string("0-1:2-3").unwrap();
        assert_eq!(
            thread_allocation_information_as_string(2, &two, &[0, 1]),
            "Using 2 threads with NUMA node thread binding: 1/2:1/2"
        );

        // Both workers bound to node 0 of three nodes → the trailing nodes are
        // extended with `0/total`.
        let three = NumaConfig::from_string("0-1:2-3:4-5").unwrap();
        assert_eq!(
            thread_allocation_information_as_string(2, &three, &[0, 0]),
            "Using 2 threads with NUMA node thread binding: 2/2:0/2:0/2"
        );
    }

    #[test]
    fn thread_pool_new_sizes_to_main_plus_helpers() {
        let pool = ThreadPool::new(4);
        assert_eq!(pool.size(), 4, "4 slots = 1 main + 3 helpers");
        let single = ThreadPool::new(1);
        assert_eq!(single.size(), 1, "1 slot = main only, no helpers");
    }

    #[test]
    fn thread_pool_set_rebuilds_without_leaking() {
        // Each `set` joins the previous generation of helpers before spawning
        // the next, so cycling sizes never leaks OS threads. We can only assert
        // the resulting slot count here; the no-leak property is what the join
        // in `shutdown` guarantees.
        let mut pool = ThreadPool::new(2);
        assert_eq!(pool.size(), 2);
        pool.set(1);
        assert_eq!(pool.size(), 1);
        pool.set(4);
        assert_eq!(pool.size(), 4);
        pool.set(4);
        assert_eq!(pool.size(), 4, "a same-size set still rebuilds cleanly");
        // Dropping the pool joins the remaining helpers.
    }

    #[test]
    fn thread_pool_zero_is_clamped_to_one() {
        // The driver never passes 0 (the option min is 1), but the pool clamps
        // defensively so `size − 1` never underflows.
        let pool = ThreadPool::new(0);
        assert_eq!(pool.size(), 1);
    }

    // --- shared-history node mapping --------------------------------------

    #[test]
    fn shared_node_counts_unbound_and_bound() {
        // Unbound: every thread pretended on node 0, count == requested.
        let c = shared_node_counts(&[], 5);
        assert_eq!(c.len(), 1);
        assert_eq!(c[&0], 5);

        // Bound: per-node counts.
        let c = shared_node_counts(&[0, 1, 0, 1, 0], 5);
        assert_eq!(c[&0], 3);
        assert_eq!(c[&1], 2);
    }

    #[test]
    fn worker_nodes_selects_each_worker_node() {
        // Unbound: every worker on node 0.
        assert_eq!(worker_nodes(&[], 3), vec![0, 0, 0]);
        // Bound: worker `i` → `bound[i]`.
        assert_eq!(worker_nodes(&[1, 0, 1], 3), vec![1, 0, 1]);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn build_worker_shared_unbound_shares_one_set() {
        let cfg = NumaConfig::from_string("0-3").unwrap();
        let ws = build_worker_shared(&cfg, &[], 4);
        assert_eq!(ws.len(), 4, "one handle per worker");
        // All workers on the single node share the SAME table set.
        for i in 1..4 {
            assert!(
                Arc::ptr_eq(&ws[0], &ws[i]),
                "unbound: every worker points at one shared set"
            );
        }
        // Sized to `next_power_of_two(pool size)`.
        assert_eq!(ws[0].thread_count(), 4);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn build_worker_shared_unbound_rounds_thread_count_up() {
        let cfg = NumaConfig::from_string("0-3").unwrap();
        // 3 workers → next_power_of_two(3) == 4 (matches the reference helper).
        let ws = build_worker_shared(&cfg, &[], 3);
        assert_eq!(ws.len(), 3);
        assert_eq!(ws[0].thread_count(), 4);
        // Single worker → thread_count 1.
        let ws1 = build_worker_shared(&cfg, &[], 1);
        assert_eq!(ws1.len(), 1);
        assert_eq!(ws1[0].thread_count(), 1);
    }

    // --- NUMA memory placement --------------------------------------------

    /// A NUMA node string that forces binding on any machine: a one-node
    /// custom config over a single CPU the process is definitely allowed on.
    /// `custom_affinity` short-circuits `suggests_binding_threads` to true, so
    /// this turns the whole pin-and-place path on even where `auto` would never
    /// bind — and picking the CPU from the live affinity keeps the fail-loud
    /// `sched_setaffinity` inside it from ever seeing a forbidden CPU.
    #[cfg(target_os = "linux")]
    fn forced_binding_policy() -> String {
        let cpu = yorkie_numa::startup_affinity()
            .iter()
            .next()
            .copied()
            .unwrap_or(0);
        cpu.to_string()
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn bind_plan_is_absent_when_binding_is_inactive() {
        // The single-node host case: no assignment, so no plan, so no thread
        // ever pins itself and nothing touches a memory policy.
        let cfg = NumaConfig::from_string("0-3").unwrap();
        assert!(bind_plan(&cfg, &[], &[0]).is_none());
        assert_eq!(coordinator_system_node(None), None);
    }

    #[test]
    fn table_placement_covers_the_nodes_the_workers_sit_on_and_no_others() {
        // Every worker on one node: the whole table belongs on that node, and
        // it is the node the layout maps the logical one to, not the logical
        // one.
        assert_eq!(
            table_placement(&[0, 0, 0, 0], &[2]),
            TablePlacement::OnNode(2)
        );
        // Two nodes under the workers: spread over exactly those two.
        assert_eq!(
            table_placement(&[0, 1, 0, 1], &[0, 1]),
            TablePlacement::AcrossNodes(vec![0, 1])
        );
        // A four-node machine whose binding uses two of them: the other two
        // hold no worker, so no page of the table may land on them.
        assert_eq!(
            table_placement(&[1, 3, 1], &[0, 1, 2, 3]),
            TablePlacement::AcrossNodes(vec![1, 3])
        );
        // No binding: nothing pins a worker, so the engine names no node set of
        // its own and the process's policy places the pages — whatever the
        // layout the binary was built for covers.
        assert_eq!(
            table_placement(&[], &[0, 1, 2, 3]),
            TablePlacement::ProcessDefault
        );
        assert_eq!(table_placement(&[], &[0]), TablePlacement::ProcessDefault);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn bind_plan_carries_one_system_node_per_worker() {
        // The plan's system-node vector is what both the pin site and the
        // placement calls index with a worker id, so it is one entry per worker
        // — the *logical* node each worker sits on, resolved through the
        // layout's logical → system map, which L3 bundling makes a many-to-one.
        let cfg = NumaConfig::from_string("0:1:2").unwrap();
        let bound = vec![0, 1, 2, 0];
        let plan =
            bind_plan(&cfg, &bound, &[3, 3, 7]).expect("a non-empty assignment yields a plan");
        assert_eq!(plan.bound, bound);
        assert_eq!(
            plan.system_nodes,
            vec![3, 3, 7, 3],
            "one system node per worker, through the layout's map"
        );
        assert_eq!(coordinator_system_node(Some(&plan)), Some(3));
    }

    /// With a NUMA layout that forces binding, every large-page block behind the
    /// coordinator's history tables must be governed by an `MPOL_BIND` policy
    /// naming worker 0's system node.
    ///
    /// Still meaningful on a single-node host: the *placement* answer there is
    /// node 0 either way, but the *policy* over those pages is `MPOL_DEFAULT`
    /// until something binds it.
    ///
    /// The layout is injected rather than selected, since `numa_policy` is a
    /// compile-time constant and no checked-in config would force binding on an
    /// arbitrary host. Everything downstream of it is the production path.
    #[cfg(target_os = "linux")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn forced_binding_places_the_coordinator_histories_on_worker_zero_node() {
        let output = Arc::new(Mutex::new(Vec::<u8>::new()));
        let mut driver = UsiDriver::new(&b""[..], Arc::clone(&output));

        // A custom node string always suggests binding, so this turns the
        // binding on even on a single-node host. The node is the one CPU the
        // process is certain to be allowed on, so the (fail-loud) pin the
        // rebuild's helper threads perform cannot hit a forbidden CPU.
        driver.numa_config = NumaConfig::from_string(&forced_binding_policy())
            .expect("a one-CPU custom node string is a valid config");
        driver.rebuild_pool();

        let node = coordinator_system_node(driver.numa_plan.as_ref())
            .expect("a custom NumaPolicy binds, so a plan exists");
        let regions = driver
            .histories
            .as_ref()
            .expect("the coordinator bundle survives a pool rebuild")
            .backing_regions();
        assert!(!regions.is_empty(), "the bundle owns large-page blocks");

        for (addr, len) in regions {
            let Some(policy) = mempolicy::policy_at_address(addr) else {
                continue; // `get_mempolicy` unavailable — nothing to assert.
            };
            if policy.mode == mempolicy::MODE_DEFAULT {
                // The kernel refused the `mbind` (no CONFIG_NUMA, seccomp, a
                // restricted cgroup). Best-effort: today's behaviour stands.
                continue;
            }
            assert_eq!(
                policy.mode,
                mempolicy::MODE_BIND,
                "region {addr:#x}+{len} must be bound, not merely preferred"
            );
            assert_eq!(
                policy.nodes,
                vec![node],
                "region {addr:#x}+{len} must name worker 0's system node"
            );
        }
    }

    // A short-circuit carries the reference's SKIP_SEARCH bookkeeping: both
    // persisted scores become `-VALUE_INFINITE`, `last_game_ply` advances, and
    // `previous_time_reduction` is left untouched.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn skip_search_carry_updates_scores_and_ply_but_not_time_reduction() {
        assert_eq!(
            skip_search_carry(),
            Some((-VALUE_INFINITE, -VALUE_INFINITE, None)),
            "the short-circuit carry is the -VALUE_INFINITE sentinel with no tr"
        );

        let output = Arc::new(Mutex::new(Vec::<u8>::new()));
        let mut driver = UsiDriver::new(&b""[..], Arc::clone(&output));
        // Seed distinctive "previous real search" state.
        driver.best_previous_score = 123;
        driver.best_previous_average_score = 456;
        driver.previous_time_reduction = 0.42;
        driver.last_game_ply = 7;

        // A synthetic short-circuited search that "ran" at ply 20 and hands back
        // the book / declaration / resign carry.
        let handle = std::thread::spawn(|| SearchState {
            histories: WorkerHistories::new(),
            time_state: skip_search_carry(),
        });
        driver.search = Some(ActiveSearch {
            handle,
            stop: Arc::new(AtomicBool::new(false)),
            ponder: None,
            suppress: Arc::new(AtomicBool::new(false)),
            #[cfg(feature = "verbose3")]
            bestmove_sent: Arc::new(AtomicBool::new(true)),
            game_ply: 20,
        });
        driver.finish_search_join();

        assert_eq!(driver.best_previous_score, -VALUE_INFINITE);
        assert_eq!(driver.best_previous_average_score, -VALUE_INFINITE);
        assert_eq!(
            driver.last_game_ply, 20,
            "ply advances to the short-circuit's"
        );
        assert_eq!(
            driver.previous_time_reduction, 0.42,
            "previousTimeReduction is left untouched on a short-circuit"
        );
    }

    // A real search's carry (`Some(tr)`) *does* overwrite `previous_time_reduction`
    // — the complement of the short-circuit case above.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn real_search_carry_overwrites_time_reduction() {
        let output = Arc::new(Mutex::new(Vec::<u8>::new()));
        let mut driver = UsiDriver::new(&b""[..], Arc::clone(&output));
        driver.previous_time_reduction = 0.42;

        let handle = std::thread::spawn(|| SearchState {
            histories: WorkerHistories::new(),
            time_state: Some((10, 20, Some(1.25))),
        });
        driver.search = Some(ActiveSearch {
            handle,
            stop: Arc::new(AtomicBool::new(false)),
            ponder: None,
            suppress: Arc::new(AtomicBool::new(false)),
            #[cfg(feature = "verbose3")]
            bestmove_sent: Arc::new(AtomicBool::new(true)),
            game_ply: 3,
        });
        driver.finish_search_join();

        assert_eq!(driver.best_previous_score, 10);
        assert_eq!(driver.best_previous_average_score, 20);
        assert_eq!(driver.previous_time_reduction, 1.25);
        assert_eq!(driver.last_game_ply, 3);
    }

    // These exercise the per-worker handle resolution and the region set with a
    // trivial stand-in payload, so they need neither a loaded network nor
    // multi-node hardware.

    #[test]
    fn every_worker_reads_the_one_instance_when_nothing_is_bound() {
        // A single-node machine, a single-threaded build, or
        // `numa_policy = "none"`: one instance, and every worker points at it.
        let instances = vec![Arc::new(1000u32)];
        let workers = resolve_worker_networks(&instances, &BTreeMap::new(), &[], 3);
        assert_eq!(workers.len(), 3);
        for w in &workers {
            assert!(Arc::ptr_eq(w, &instances[0]));
        }
    }

    #[test]
    fn each_worker_reads_the_instance_its_system_node_holds() {
        // Four workers over two system nodes, alternating: each gets its own
        // node's copy, and nothing is cloned to arrange it.
        let instances = vec![Arc::new(10u32), Arc::new(20u32)];
        let by_node: BTreeMap<NumaIndex, usize> = [(0usize, 0usize), (1, 1)].into_iter().collect();
        let workers = resolve_worker_networks(&instances, &by_node, &[0, 1, 0, 1], 4);
        assert_eq!(
            workers.iter().map(|w| **w).collect::<Vec<u32>>(),
            vec![10, 20, 10, 20]
        );
    }

    #[test]
    fn two_logical_nodes_on_one_system_node_read_one_copy() {
        // The copies are per system node, so workers on logical nodes that an
        // L3-aware mapping split out of one system node share a copy.
        let instances = vec![Arc::new(10u32)];
        let by_node: BTreeMap<NumaIndex, usize> = [(0usize, 0usize)].into_iter().collect();
        let workers = resolve_worker_networks(&instances, &by_node, &[0, 0], 2);
        assert!(Arc::ptr_eq(&workers[0], &workers[1]));
    }

    #[test]
    fn a_worker_on_a_node_with_no_copy_reads_the_first_one() {
        // The worker count grew past the plan the network was placed for, so a
        // worker's node has no copy of its own: it reads another node's, which
        // holds the same parameters.
        let instances = vec![Arc::new(10u32), Arc::new(20u32)];
        let by_node: BTreeMap<NumaIndex, usize> = [(0usize, 0usize), (1, 1)].into_iter().collect();
        let workers = resolve_worker_networks(&instances, &by_node, &[1, 7], 2);
        assert_eq!(
            workers.iter().map(|w| **w).collect::<Vec<u32>>(),
            vec![20, 10]
        );
    }

    #[test]
    fn no_instance_means_no_worker_handles() {
        // Before an `isready` has read the file there is nothing to hand out,
        // and a `go` in that state resigns.
        let none: Vec<Arc<u32>> = Vec::new();
        assert!(resolve_worker_networks(&none, &BTreeMap::new(), &[0], 2).is_empty());
    }

    #[test]
    fn a_plan_that_binds_nothing_needs_no_per_node_region() {
        assert!(network_regions(&[], &[0, 1]).is_empty());
    }

    #[test]
    fn one_region_per_distinct_system_node_the_plan_uses() {
        // Four logical nodes, two per system node — an L3-subdivided layout.
        let system_nodes = [0usize, 0, 1, 1];
        assert_eq!(network_regions(&[0, 1, 2, 3], &system_nodes), vec![0, 1]);
        // A plan confined to one system node's logical nodes needs one region.
        assert_eq!(network_regions(&[2, 3, 2], &system_nodes), vec![1]);
        // The set is in node order whatever order the workers landed in.
        assert_eq!(network_regions(&[3, 0, 2, 1], &system_nodes), vec![0, 1]);
    }

    // -- Multiple Book name resolution ------------------------------------

    /// A fresh empty directory under `$TMPDIR`; the caller removes it.
    fn book_name_fixture_dir(tag: &str) -> PathBuf {
        static CTR: AtomicU64 = AtomicU64::new(0);
        let n = CTR.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "engine-book-names-{}-{tag}-{n}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("mkdir book-name fixture");
        root
    }

    /// Touch an empty file (only existence matters to the name resolution).
    fn touch(path: &Path) {
        std::fs::write(path, b"").expect("touch");
    }

    /// The file names (not full paths) of a resolved list, for readable asserts.
    fn file_names(paths: &[PathBuf]) -> Vec<String> {
        paths
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect()
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn priority_series_stops_at_the_first_gap_and_appends_the_base_last() {
        let dir = book_name_fixture_dir("series");
        let base = dir.join("user_book1.ybb");
        touch(&base);
        touch(&dir.join("user_book1-000.ybb"));
        touch(&dir.join("user_book1-001.ybb"));
        // `-002` is absent, so `-003` is never reached: a gap ends the series.
        touch(&dir.join("user_book1-003.ybb"));

        let (names, notices) = book_names(&base);
        assert_eq!(
            file_names(&names),
            vec![
                "user_book1-000.ybb",
                "user_book1-001.ybb",
                "user_book1.ybb", // the plain base name comes LAST
            ]
        );
        assert!(
            notices.is_empty(),
            "no duplicate-extension notice: {notices:?}"
        );

        // The index is zero-padded to three digits.
        let stem = book_name_without_extension(&base).expect("stem");
        assert_eq!(
            priority_book_filename(&stem, 7, "ybb")
                .file_name()
                .unwrap()
                .to_string_lossy(),
            "user_book1-007.ybb"
        );
        assert_eq!(
            priority_book_filename(&stem, 42, "db")
                .file_name()
                .unwrap()
                .to_string_lossy(),
            "user_book1-042.db"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn no_numbered_files_yields_just_the_base_name() {
        let dir = book_name_fixture_dir("bare");
        let base = dir.join("user_book1.ybb");
        touch(&base);
        let (names, notices) = book_names(&base);
        assert_eq!(names, vec![base]);
        assert!(notices.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn no_book_has_an_empty_series() {
        let dir = book_name_fixture_dir("nobook");
        let base = dir.join("no_book");
        // Even a stray `no_book-000.ybb` cannot start a series: the sentinel has
        // no `.db` / `.ybb` extension, so its stem is empty.
        touch(&dir.join("no_book-000.ybb"));
        assert_eq!(book_name_without_extension(&base), None);
        assert!(resolve_priority_book_filename(&base, 0).is_none());
        let (names, notices) = book_names(&base);
        assert_eq!(names, vec![base], "only the base name, no series");
        assert!(notices.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn cross_extension_resolution_prefers_the_bases_own_extension() {
        let dir = book_name_fixture_dir("crossext");

        // `.ybb` base: primary `.ybb` wins over a co-existing `.db`, with the
        // reference's verbatim notice.
        let ybb_base = dir.join("user_book1.ybb");
        touch(&ybb_base);
        touch(&dir.join("user_book1-000.ybb"));
        touch(&dir.join("user_book1-000.db"));
        let (names, notices) = book_names(&ybb_base);
        assert_eq!(
            file_names(&names),
            vec!["user_book1-000.ybb", "user_book1.ybb"]
        );
        assert_eq!(
            notices,
            vec![format!(
                "priority book file exists twice. use : {}",
                dir.join("user_book1-000.ybb").display()
            )]
        );

        // `.db` base: primary `.db` wins over a co-existing `.ybb`.
        let db_base = dir.join("user_book2.db");
        touch(&dir.join("user_book2-000.ybb"));
        touch(&dir.join("user_book2-000.db"));
        let (names, notices) = book_names(&db_base);
        assert_eq!(
            file_names(&names),
            vec!["user_book2-000.db", "user_book2.db"]
        );
        assert_eq!(
            notices,
            vec![format!(
                "priority book file exists twice. use : {}",
                dir.join("user_book2-000.db").display()
            )]
        );

        // Secondary-only: a `.ybb` base with just a `.db` at index 0 resolves to
        // the `.db` (which `reload_book` then routes to the fail-loud path).
        let solo = dir.join("user_book3.ybb");
        touch(&dir.join("user_book3-000.db"));
        let (names, notices) = book_names(&solo);
        assert_eq!(
            file_names(&names),
            vec!["user_book3-000.db", "user_book3.ybb"]
        );
        assert!(notices.is_empty(), "one file only → no notice: {notices:?}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn db_names_still_resolve_to_their_ybb_sibling() {
        // The `BookFile` combo advertises only `.ybb` names, so this fallback is
        // unreachable from the option surface — but `reload_book` routes every
        // enumerated name through it. Pin the behaviour here so it cannot rot.
        let dir = book_name_fixture_dir("fallback");

        // An absent `.db` whose `.ybb` sibling exists → the sibling.
        let ybb = dir.join("user_book1.ybb");
        touch(&ybb);
        assert_eq!(
            resolve_book_filename_with_ybb_fallback(&dir.join("user_book1.db")),
            ybb
        );

        // An existing file is returned untouched, whatever its extension.
        let db = dir.join("user_book2.db");
        touch(&db);
        touch(&dir.join("user_book2.ybb"));
        assert_eq!(resolve_book_filename_with_ybb_fallback(&db), db);

        // No `.ybb` sibling → the request is returned as-is (the caller then
        // reports the load failure).
        let missing = dir.join("user_book3.db");
        assert_eq!(resolve_book_filename_with_ybb_fallback(&missing), missing);

        // Only a `.db` request is rewritten: an absent `.ybb` stays absent.
        let absent_ybb = dir.join("user_book4.ybb");
        assert_eq!(
            resolve_book_filename_with_ybb_fallback(&absent_ybb),
            absent_ybb
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
