//! The engine proper: the game state, the worker pool and the search, with no
//! protocol text anywhere in it.
//!
//! What a host says and what the engine does are two different things, and this
//! is the second of them. An [`Engine`] holds the position, the worker pool and
//! its handles, the placed transposition table and evaluation network, the
//! opening book, the per-game seeds and the time-management carry-forward; it
//! offers [`Engine::set_position`], [`Engine::go`], [`Engine::stop`],
//! [`Engine::ponderhit`], [`Engine::new_game`] and [`Engine::ready`] as plain
//! functions over typed arguments and typed results. Nothing here knows what a
//! reply line looks like.
//!
//! Where the engine has something to say — a reply, an initialisation notice, a
//! diagnostic — it says it through [`EngineSink`], a generic parameter rather
//! than a runtime-chosen sink: a build compiles exactly one protocol layer, so
//! the type is known when the binary is, every call is a direct one, and nothing
//! on the game path is dispatched through a pointer.

use std::collections::BTreeSet;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use yorkie_eval::{NetworkParams, NnueError, network_file};
use yorkie_numa::{NumaIndex, NumaLayout, mempolicy};
use yorkie_search::{
    BookConfig, BookHit, EnteringKingConfig, PonderSignal, Prng, QSearch, RootMove, SearchControl,
    SharedHistories, TimeControl, TimeInput, TimeManagement, WorkerHistories, WorkerResult,
    WorkerVote, declaration_win, generate_root_moves, probe_book, select_best_worker,
};
// The PV-line surface: only a `verbose2` build renders one, so only it needs
// the line's data type, the sink trait the search emits through and the output
// config.
#[cfg(feature = "verbose2")]
use yorkie_search::{PvInfo, PvOutputConfig, PvSink};
// The per-game evaluation-noise seed: drawn here, read only by the search.
#[cfg(feature = "random")]
use yorkie_search::new_game_seed;
use yorkie_state::{ExtMove, Move, Position, SfenError, parse_sfen_into, parse_usi_move};
use yorkie_storage::{Book, TranspositionTable, Value};
// The per-reply allocation tally: raised by the counting global allocator this
// feature installs, and cleared where an interval starts.
#[cfg(feature = "verbose1")]
use yorkie_storage::clear_alloc_count;

use crate::config::{self, with_eval_network};
#[cfg(feature = "random")]
use crate::settings::RANDOM_AMPLITUDE;
use crate::settings::Settings;

// The transposition table is a `static` sized from the `usi_hash` config
// constant (the reference's `USI_Hash` option — the depth-1 fixture capture
// condition) when the binary is built, so no command and no reply can change
// how big it is. What the readiness handshake still decides is where its pages
// live; see
// [`Engine::place_transposition_table`].

/// The largest iterative-deepening depth a `go` ever requests. `run_root`'s own
/// `rootDepth + 1 < MAX_PLY` guard (`MAX_PLY == 246`) is the real ceiling; this
/// is the value passed for a time-/stop-bounded `go` (no explicit `depth`), and
/// the clamp for an out-of-range `go depth N`. It sits one below `MAX_PLY` so
/// the loop guard never has to truncate it.
const SEARCH_MAX_DEPTH: i32 = 245;

/// Where the running machine's NUMA layout is read from, for the one check that
/// reads it: [`Engine::ready`]'s, which holds the machine against the layout
/// this binary was built for.
const SYSFS_ROOT: &str = "/sys";

/// Moves one `position` command may carry. A shogi game reaches its result in a
/// few hundred plies — a build that sets a draw ply cap stops one far sooner —
/// so the bound is out of a game's reach, and a command that exceeds it is
/// refused rather than allowed to grow the buffer that holds it.
pub(crate) const MAX_POSITION_MOVES: usize = 1024;

/// Room the retained SFEN buffer starts with: enough for the longest SFEN a
/// board can spell — every square occupied by a promoted piece, both hands
/// full, a five-digit ply — so a game's `position` commands are written into
/// the buffer already there.
const SFEN_CAPACITY: usize = 256;

/// Legal moves the widest shogi position offers, which is what the replay's
/// legality buffer is built to hold (the reference's `MAX_MOVES`).
const MAX_LEGAL_MOVES: usize = 600;

/// A position command in the form the engine replays it from: where the game
/// starts, and the moves that reached the position it names, already parsed and
/// verified legal in sequence.
///
/// Both buffers are filled in place and never handed back to the allocator, so
/// a game's worth of `position` commands reaches it only for the first one.
struct RetainedPosition {
    /// The command named `startpos` rather than an explicit SFEN, in which case
    /// [`Self::sfen`] is empty.
    startpos: bool,
    /// The four SFEN fields joined by single spaces — the form
    /// [`parse_sfen_into`] reads.
    sfen: String,
    /// The moves in the order they were applied.
    moves: Vec<Move>,
}

impl RetainedPosition {
    /// `position startpos`, with room for a whole game ahead of it.
    fn new() -> Self {
        Self {
            startpos: true,
            sfen: String::with_capacity(SFEN_CAPACITY),
            moves: Vec::with_capacity(MAX_POSITION_MOVES),
        }
    }

    /// Back to `position startpos`, every buffer keeping its room.
    fn reset(&mut self) {
        self.startpos = true;
        self.sfen.clear();
        self.moves.clear();
    }

    /// Take the starting position of a freshly parsed command, joining an
    /// explicit SFEN's four fields into the buffer already held. The move list
    /// is emptied for the replay that follows.
    fn set_start(&mut self, sfen: PositionSfen<'_>) {
        self.reset();
        if let PositionSfen::Sfen(fields) = sfen {
            self.startpos = false;
            for (i, field) in fields.iter().enumerate() {
                if i > 0 {
                    self.sfen.push(' ');
                }
                self.sfen.push_str(field);
            }
        }
    }

    /// Build the starting position into `pos`, reusing the buffers it holds.
    fn start_into(&self, pos: &mut Position) -> Result<(), SfenError> {
        if self.startpos {
            pos.reset_startpos();
            Ok(())
        } else {
            parse_sfen_into(pos, &self.sfen)
        }
    }
}

/// Why a position command was refused. Each carries what a diagnostic would
/// name, borrowed from the command line itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PositionRefusal<'a> {
    Sfen(SfenError),
    IllegalMove(&'a str),
    TooManyMoves,
}

/// Where a position starts: the implicit initial position, or an explicit SFEN
/// whose four fields — board, side to move, hands, ply — are borrowed from the
/// command that named them rather than copied out of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PositionSfen<'a> {
    StartPos,
    Sfen([&'a str; 4]),
}

/// Everything that bounds one search, captured verbatim — including what a
/// given build does not act on, so a protocol layer's parse is lossless.
///
/// Six of the bounds are `verbose2`: `depth`, `nodes`, `movetime`, `infinite`,
/// `mate` and `rtime`. A build without that feature has no source for any of
/// them — the two config keys that seed the first two need the same feature —
/// so the fields carry it rather than standing unfillable.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GoParams {
    #[cfg(feature = "verbose2")]
    pub depth: Option<u32>,
    #[cfg(feature = "verbose2")]
    pub nodes: Option<u64>,
    #[cfg(feature = "verbose2")]
    pub movetime: Option<u64>,
    pub wtime: Option<u64>,
    pub btime: Option<u64>,
    pub winc: Option<u64>,
    pub binc: Option<u64>,
    pub byoyomi: Option<u64>,
    #[cfg(feature = "verbose2")]
    pub infinite: bool,
    /// Think on the predicted position; hold the reply until the prediction is
    /// confirmed or the search is stopped.
    pub ponder: bool,
    /// Mate-search mode with a time budget in milliseconds, where
    /// [`MATE_UNLIMITED_MS`] stands for unlimited.
    #[cfg(feature = "verbose2")]
    pub mate: Option<u64>,
    /// A randomised minimum-thinking-time budget used for self-play variety.
    /// `init_` seeds all three time bounds from it (plus a decaying random bump)
    /// and returns early. `None` means no such budget.
    #[cfg(feature = "verbose2")]
    pub rtime: Option<u64>,
}

/// The unlimited mate-search budget (`limits.mate = INT32_MAX`).
#[cfg(feature = "verbose2")]
pub const MATE_UNLIMITED_MS: u64 = i32::MAX as u64;
/// The evaluation-noise seed a `bench` runs under. What the command reports is a
/// node count two runs — and two processes — have to agree on, so the clean
/// starting state it builds for itself fixes the seed as well as the table and
/// the histories. A game draws its own.
#[cfg(all(feature = "verbose3", feature = "random"))]
const BENCH_RANDOM_SEED: u64 = 0;
/// `Eval::PawnValue` / `NormalizeToPawnValue`.
pub(crate) const PAWN_VALUE: Value = 90;
/// `VALUE_INFINITE`: the pre-search `rootMoves[0].score` sentinel the
/// `ResignValue` guard excludes.
const VALUE_INFINITE: Value = 32001;

/// `ResignValue`: the post-search resign threshold in centipawns. A searched
/// best score at or below `-RESIGN_VALUE` resigns.
const RESIGN_VALUE: Value = crate::config::RESIGN_VALUE as Value;
/// The book-selection settings, as one value rather than eighteen reads at every
/// `go`. `IgnoreBookPly` is not here — it is captured at book-load time and
/// travels with [`LoadedBook`].
///
/// Both profiles' fields are present; the probe picks between them from
/// `book_options_v2` and the root side to move. A setting the active profile
/// does not own reads as its type's zero, which is inert on the leg that never
/// consults it.
const BOOK_CONFIG: BookConfig = BookConfig {
    book_options_v2: crate::config::BOOK_OPTIONS_V2,
    narrow_book: !crate::config::BOOK_OPTIONS_V2 && crate::config::NARROW_BOOK,
    book_moves: crate::config::BOOK_MOVES,
    ignore_rate: crate::config::BOOK_IGNORE_RATE,
    eval_diff: if crate::config::BOOK_OPTIONS_V2 {
        0
    } else {
        crate::config::BOOK_EVAL_DIFF
    },
    eval_black_diff: if crate::config::BOOK_OPTIONS_V2 {
        crate::config::BOOK_EVAL_BLACK_DIFF
    } else {
        0
    },
    eval_white_diff: if crate::config::BOOK_OPTIONS_V2 {
        crate::config::BOOK_EVAL_WHITE_DIFF
    } else {
        0
    },
    eval_black_limit: crate::config::BOOK_EVAL_BLACK_LIMIT,
    eval_white_limit: crate::config::BOOK_EVAL_WHITE_LIMIT,
    depth_limit: if crate::config::BOOK_OPTIONS_V2 {
        0
    } else {
        crate::config::BOOK_DEPTH_LIMIT
    },
    depth_black_limit: if crate::config::BOOK_OPTIONS_V2 {
        crate::config::BOOK_DEPTH_BLACK_LIMIT
    } else {
        0
    },
    depth_white_limit: if crate::config::BOOK_OPTIONS_V2 {
        crate::config::BOOK_DEPTH_WHITE_LIMIT
    } else {
        0
    },
    consider_move_count: !crate::config::BOOK_OPTIONS_V2 && crate::config::CONSIDER_BOOK_MOVE_COUNT,
    // Shapes the book `info` lines and nothing else, so it is read only in a
    // build that prints them.
    #[cfg(feature = "verbose2")]
    pv_moves: crate::config::BOOK_PV_MOVES,
    flipped_book: crate::config::FLIPPED_BOOK,
};

/// The reference `USIEngine::to_cp`: `100 * v / NormalizeToPawnValue`, with
/// C++-style truncating division (Rust truncates toward zero, matching). Used
/// by the `ResignValue` check; unlike `format_score` it does not special-case
/// mate scores (the reference `to_cp` applies the same linear map to all
/// values).
fn to_cp(v: Value) -> Value {
    100 * v / PAWN_VALUE
}

/// The file the evaluation network in memory was read from.
///
/// The path is retained so [`Engine::ready`] is idempotent: a repeat reuses what is
/// already there instead of reading the file again. There is nothing else to
/// record — a region is a `static` this binary declares, which region a worker
/// reads follows from the compiled assignment, and both are known before the
/// process starts.
///
/// How many regions are filled is the machine's answer, decided when the binary
/// was built. On a single-node machine there is one, the file's own pages
/// mapped onto it and shared by every worker — and by every other engine
/// process on the machine, the pages being the page cache's. On a multi-node
/// machine there is one copy per *system* NUMA node the assignment puts workers
/// on, each in memory belonging to that node. The granularity is the system
/// node, not the possibly L3-bundled logical node, so logical nodes that share
/// a system node share one copy.
struct LoadedEval {
    path: PathBuf,
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
/// keeps reading commands while this runs; a stop request sets [`Self::stop`],
/// which the search polls at the reference `check_time` granularity. The worker
/// emits its own progress and its reply, then returns the session-owned
/// [`SearchState`] so the engine can reclaim it for the next `go`.
struct ActiveSearch {
    handle: JoinHandle<SearchState>,
    stop: Arc<AtomicBool>,
    /// The shared `go ponder` state (`Some` only for a `go ponder`). A plain
    /// `ponderhit` clears it (`set_ponderhit(false)`), turning the pondering
    /// search into a normal time-managed one; `None` means this was not a ponder
    /// search, and a stray `ponderhit` falls back to a `stop`.
    ponder: Option<Arc<PonderSignal>>,
    /// Suppresses the coordinator's reply (and final PV) for the
    /// Stochastic_Ponder ponderhit teardown, which stops the rewound search
    /// without emitting anything.
    suppress: Arc<AtomicBool>,
    /// Set by the coordinator *inside* the critical section that writes the
    /// reply, so this search counts as finished from the moment that reply is
    /// on the wire.
    ///
    /// [`JoinHandle::is_finished`] is not that moment: the coordinator emits its
    /// reply and only then unwinds, so a host that reads the reply and
    /// immediately sends the next command can land in the window where the
    /// thread has not yet returned. A flag stamped under the output lock closes
    /// it — any reader that has seen the reply took the same lock afterwards, so
    /// it cannot see this as unset.
    ///
    /// The `tt` commands' idle check is its only reader, and they are
    /// `verbose3`, so a build without that feature neither carries nor raises it.
    #[cfg(feature = "verbose3")]
    reply_sent: Arc<AtomicBool>,
    /// The root game ply this search ran at (`rootPos.game_ply()`), carried so
    /// a completed real search updates the engine's `last_game_ply`.
    game_ply: i32,
}

/// The session-owned search state a `go` lends to its worker and reclaims when
/// the worker finishes: the game-scoped worker history tables, which persist
/// across `go`s within one game and are reset by [`Engine::new_game`].
///
/// The transposition table is not part of this handover — it lives behind an
/// [`Arc`], so the worker gets a clone and never hands it back.
struct SearchState {
    histories: WorkerHistories,
    /// The chosen worker's reported score / average score and the main worker's
    /// final `timeReduction`, carried back so the engine seeds the next `go`'s
    /// time management.
    ///
    /// Always `Some`: the reference runs this bookkeeping on *every* path,
    /// including the search-skipping short-circuits, which carry the unsearched
    /// `-VALUE_INFINITE` defaults and the current ply. The third element is
    /// `Some(tr)` only when a real search produced a fresh `timeReduction`; on a
    /// short-circuit the reference never touches `previousTimeReduction`, so the
    /// engine's persisted value is left unchanged.
    time_state: Option<(Value, Value, Option<f64>)>,
    /// The collection buffers the coordinator filled, handed back for the next
    /// `go` to fill again.
    vote_buffers: VoteBuffers,
}

/// The handles a `go` hands its coordinator that carry no value from one search
/// to the next: the flags the main loop signals on and the per-worker counters
/// the workers tally into.
///
/// They belong to the session rather than to one search, and each `go` clears
/// them ([`Self::arm`]) instead of building new ones, so starting a search
/// reaches the allocator for none of them. Nothing races that reset: every `go`
/// joins the previous coordinator first, and a joined coordinator has already
/// collected every helper it dispatched.
struct SearchHandles {
    /// The one shared stop flag every worker polls.
    stop: Arc<AtomicBool>,
    /// The `go ponder` state. Handed to a search only when the `go` carries
    /// `ponder`; every other `go` leaves it behind, inert.
    ponder: Arc<PonderSignal>,
    /// The Stochastic_Ponder teardown flag: set to drop a rewound search's
    /// reply.
    suppress_reply: Arc<AtomicBool>,
    /// Raised when a search's reply reaches the output sink. Its only reader is
    /// the `verbose3` table-inspection commands' idle check.
    #[cfg(feature = "verbose3")]
    reply_sent: Arc<AtomicBool>,
    /// Per-worker node counters (index 0 = main, `1..` = helpers), read by the
    /// aggregate node ceiling and the final aggregated `info ... nodes` — both
    /// `verbose2`, like the counters themselves.
    #[cfg(feature = "verbose2")]
    node_slots: Arc<Vec<AtomicU64>>,
    /// Per-worker best-move-change counters: each worker bumps its own slot at
    /// the root and the main worker folds them all each iteration.
    bmc_slots: Arc<Vec<AtomicU64>>,
}

impl SearchHandles {
    /// The handles for a pool of `n_threads` workers.
    fn new(n_threads: usize) -> Self {
        SearchHandles {
            stop: Arc::new(AtomicBool::new(false)),
            ponder: Arc::new(PonderSignal::new(false)),
            suppress_reply: Arc::new(AtomicBool::new(false)),
            #[cfg(feature = "verbose3")]
            reply_sent: Arc::new(AtomicBool::new(false)),
            #[cfg(feature = "verbose2")]
            node_slots: new_tally(n_threads),
            bmc_slots: new_tally(n_threads),
        }
    }

    /// Give the counters one slot per worker of a freshly (re)built pool. Only a
    /// pool rebuild changes the worker count, and no search runs across one.
    fn fit_to_pool(&mut self, n_threads: usize) {
        #[cfg(feature = "verbose2")]
        {
            self.node_slots = new_tally(n_threads);
        }
        self.bmc_slots = new_tally(n_threads);
    }

    /// Clear every flag and zero every counter for a search about to start, and
    /// seed the ponder state from whether this `go` is a `go ponder`.
    ///
    /// Returns the ponder handle to install on the search, which is `None` for
    /// every `go` that is not pondering — the distinction a stray `ponderhit`
    /// falls back on.
    fn arm(&mut self, ponder_mode: bool) -> Option<Arc<PonderSignal>> {
        self.stop.store(false, Ordering::Relaxed);
        self.suppress_reply.store(false, Ordering::Relaxed);
        #[cfg(feature = "verbose3")]
        self.reply_sent.store(false, Ordering::Relaxed);
        #[cfg(feature = "verbose2")]
        for slot in self.node_slots.iter() {
            slot.store(0, Ordering::Relaxed);
        }
        for slot in self.bmc_slots.iter() {
            slot.store(0, Ordering::Relaxed);
        }
        if !ponder_mode {
            return None;
        }
        // The signal times a `ponderhit` from the `go` that started pondering,
        // so a reused one is restarted here rather than merely re-flagged. The
        // previous search has been joined, which leaves this handle the only
        // one — and a signal that somehow still had a reader would be unsafe to
        // rewind, so that case takes a fresh one.
        match Arc::get_mut(&mut self.ponder) {
            Some(signal) => signal.restart(true),
            None => self.ponder = Arc::new(PonderSignal::new(true)),
        }
        Some(Arc::clone(&self.ponder))
    }
}

/// One zeroed counter per worker.
fn new_tally(n_threads: usize) -> Arc<Vec<AtomicU64>> {
    Arc::new((0..n_threads).map(|_| AtomicU64::new(0)).collect())
}

/// The buffers a coordinator collects its workers into: one [`WorkerResult`] per
/// worker, and the votes derived from them.
///
/// Session-owned and lent to each search exactly as the histories are, so the
/// room for a pool's worth of results is won once rather than at every `go`.
struct VoteBuffers {
    results: Vec<WorkerResult>,
    votes: Vec<WorkerVote>,
}

impl VoteBuffers {
    fn for_pool(n_threads: usize) -> Self {
        VoteBuffers {
            results: Vec::with_capacity(n_threads),
            votes: Vec::with_capacity(n_threads),
        }
    }

    /// Make room for a freshly (re)built pool's workers. What the last search
    /// left is dropped here rather than kept: a rebuilt pool's results have
    /// nothing to do with the retired pool's.
    fn fit_to_pool(&mut self, n_threads: usize) {
        self.results.clear();
        self.votes.clear();
        self.results.reserve(n_threads);
        self.votes.reserve(n_threads);
    }
}

/// What the engine plays, as the engine states it: the move it chose and the
/// move it expects in reply, or one of the two token replies a shogi engine can
/// give instead.
///
/// The protocol layer renders it; nothing about the wording of a reply is
/// decided here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reply {
    /// The chosen move, and the move the engine expects the opponent to answer
    /// with when it has one.
    BestMove { mv: Move, ponder: Option<Move> },
    /// The position is lost — the reference's `ResignValue` verdict, a root with
    /// no legal move, or a `go` with no network loaded.
    Resign,
    /// The entering-king declaration succeeds under a rule whose win is declared
    /// rather than played.
    Win,
}

/// What a `go` did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GoOutcome {
    /// A search is running, and its reply will reach the sink.
    Started,
    /// No evaluation network is loaded, so nothing was started and the caller
    /// owes the host a reply.
    NoNetwork,
}

/// The result of the heavy [`Engine::ready`] initialisation, produced inside
/// whatever keep-alive scope the protocol layer wraps it in and consumed once
/// that helper has stopped: the network is ready, the load failed, or the
/// machine is not the one this binary plans its threads for.
///
/// Each failure carries its reason for the caller to render, and no build
/// answers a failed one with a readiness reply.
pub enum ReadyOutcome {
    Ready,
    LoadFailed(String),
    LayoutMismatch(String),
}

/// Where an [`Engine`]'s output goes: the one protocol layer this build was
/// compiled with, as a type rather than as a pointer.
///
/// Every method takes what the engine knows — a [`Reply`], a book hit, a PV
/// line, a message body — and leaves the wording to the implementor. A build
/// carries exactly one implementation, so each call is direct and monomorphised;
/// there is no trait object anywhere on the game path.
///
/// The search runs on a thread of its own and emits from there, so a sink is
/// cloned into each `go`'s coordinator — hence `Clone + Send + 'static`. A clone
/// is a handle to the same output, not a second one.
pub trait EngineSink: Clone + Send + 'static {
    /// The per-iteration PV renderer this protocol layer installs on the main
    /// worker — a concrete type, like the sink itself.
    #[cfg(feature = "verbose2")]
    type PvOutput: PvSink + 'static;

    /// One initialisation-phase notice — where the table and the network went,
    /// which CPUs the workers took, what the book load found. Present in every
    /// build: they are how a failed startup is diagnosed at all.
    fn notice(&self, msg: &str) -> io::Result<()>;

    /// One diagnostic notice, the `verbose1` surface. Best-effort: it may be
    /// produced on the search thread, where a broken pipe must not panic.
    #[cfg(feature = "verbose1")]
    fn diagnostic(&self, msg: &str);

    /// The engine's reply to a search request.
    ///
    /// `sent` is raised before the output lock is released, so the reply
    /// becoming visible and the search counting as finished are one indivisible
    /// step downstream; its only reader is the `verbose3` table-inspection
    /// commands' idle check.
    fn reply(&self, reply: Reply, #[cfg(feature = "verbose3")] sent: &AtomicBool);

    /// A book hit's surviving candidates, reported as the probe answers and
    /// before any hold.
    #[cfg(feature = "verbose2")]
    fn book_candidates(&self, hit: &BookHit, hashfull: u32, time_ms: u64);

    /// A book hit's terminal output: the summary of the chosen line, then the
    /// reply itself, in one indivisible step for the same reason
    /// [`Self::reply`]'s `sent` is.
    fn book_reply(
        &self,
        hit: &BookHit,
        #[cfg(feature = "verbose2")] hashfull: u32,
        #[cfg(feature = "verbose2")] time_ms: u64,
        #[cfg(feature = "verbose3")] sent: &AtomicBool,
    );

    /// The final PV block a completed search emits before its reply.
    #[cfg(feature = "verbose2")]
    fn pv_block(&self, infos: &[PvInfo]);

    /// A renderer for the per-iteration PV lines, for the main worker to emit
    /// through. Built once per `go`, and only in a build that prints them.
    #[cfg(feature = "verbose2")]
    fn pv_output(&self) -> Self::PvOutput;
}

/// The engine: the position, the pool, the search and everything they need,
/// with the protocol layer as a type parameter rather than a dependency.
pub struct Engine<P: EngineSink> {
    /// Where replies, notices and diagnostics go. A concrete type, fixed when
    /// the binary was built, and cloned into each `go`'s coordinator so the
    /// search thread emits through the same output.
    sink: P,
    /// Where every setting comes from: the compile-time constants generated
    /// from the TOML config, in every build. See [`crate::settings`].
    settings: Settings,
    pos: Position,
    /// The loaded network holder, present only after a successful
    /// [`Self::ready`]. A `go` before this is set starts nothing.
    eval: Option<LoadedEval>,
    /// The shared transposition table the root search runs against: the one
    /// `static`, whose size this binary was built with.
    ///
    /// Cleared by [`Self::new_game`] and advanced per `go` by the search itself — the
    /// engine never bumps the generation. Every worker holds this same
    /// reference, so there is nothing to hand out and nothing to reclaim.
    tt: &'static TranspositionTable,
    /// Whether [`Self::ready`] has placed the table's pages ([`Self::place_transposition_table`]).
    /// Once per session: the policy and the huge-page hint are properties of the
    /// address range, and repeating them would say the same thing again.
    tt_placed: bool,
    /// Game-scoped worker histories. `None` only while a worker
    /// holds them mid-search.
    histories: Option<WorkerHistories>,
    /// The loaded opening books, present only after a [`Self::ready`] opened at least
    /// one readable `.ybb`. `None` means bookless (default `BookFile=no_book`, or
    /// every listed book failed / was unsupported). Behind an [`Arc`] so a `go`
    /// hands its coordinator a cheap clone.
    book: Option<Arc<LoadedBook>>,
    /// The `(resolved-name-list, on-the-fly, ignore-book-ply)` signature of the
    /// last book load — the Multiple Book priority list, not a single name.
    /// [`Self::ready`] reloads only when this changes — the reference's reload-skip.
    book_signature: Option<(Vec<PathBuf>, bool, bool)>,
    /// A session-scoped seed advanced per `go`, driving both the book-selection
    /// PRNG and the `rtime` PRNG. Seeded from process entropy by default; tests
    /// pin it via [`Engine::with_book_seed`].
    book_seed: u64,
    /// The evaluation-noise seed for the game in progress. Drawn from the
    /// operating system's randomness at construction and again at every
    /// [`Self::new_game`], and handed unchanged to every worker of every `go` in
    /// between: one game evaluates a position the same way throughout, and the
    /// next game evaluates it differently. It sits beside the transposition
    /// table rather than inside the Zobrist tables, which stay fixed, so
    /// nothing about position identity moves with it.
    #[cfg(feature = "random")]
    random_seed: u64,
    /// The in-flight search worker, if any.
    search: Option<ActiveSearch>,
    /// Time-management state that persists across `go`s within a game and is
    /// reset by [`Self::new_game`]: the previous move's reported score / average score
    /// and its final `timeReduction`. Fed into each `go`'s [`TimeControl`] and
    /// refreshed on join.
    best_previous_score: Value,
    best_previous_average_score: Value,
    previous_time_reduction: f64,
    /// The root game ply of the last completed real search
    /// (`main_manager()->lastGamePly`), reset to `0` by [`Self::new_game`].
    ///
    /// At the next search start an odd `last_game_ply - game_ply` means the side
    /// to move alternated, which flips the sign of the persisted previous scores
    /// before they seed the next search.
    last_game_ply: i32,
    /// The last position command in parsed form (`last_position_cmd_string`),
    /// retained so a Stochastic_Ponder `go ponder` can rewind it by one move
    /// and a Stochastic_Ponder `ponderhit` can re-apply the real position.
    last_position: RetainedPosition,
    /// Where the position command being read is assembled. A malformed line
    /// must leave both the current position and the retained command exactly as
    /// they were, so nothing is written through until the whole line has been
    /// verified; the two are then swapped, which hands this the previous
    /// command's buffers for the next line.
    pending_position: RetainedPosition,
    /// The position a command is replayed into, holding whatever the
    /// previous command left there. Swapped with [`Self::pos`] once the replay
    /// has succeeded, so neither position is ever rebuilt from an empty buffer.
    scratch_pos: Position,
    /// The legal moves of the ply being replayed, against which the command's
    /// next move is checked, and the pseudo-legal moves they are filtered from.
    /// Both are built once, to the widest position's move count.
    legal_buf: Vec<Move>,
    pseudo_buf: Vec<ExtMove>,
    /// The last `go` request (`last_go_cmd_string`), retained so a
    /// Stochastic_Ponder `ponderhit` can re-issue it with `ponder` stripped.
    last_go: Option<GoParams>,
    /// The worker thread pool: a main-worker slot plus `Threads − 1` persistent
    /// helper threads, each parked until a `go` dispatches it a job. The main
    /// worker is the per-`go` coordinator thread [`Self::go`] spawns.
    pool: ThreadPool,
    /// The pool size the next (re)build uses. Always the `threads` config
    /// constant, except while a `verbose3` `bench` runs its own thread count
    /// — the one command that carries a worker count as an argument. Nothing
    /// on the match path ever writes it.
    pool_threads: usize,
    /// The NUMA layout of the machine this binary was built for, rebuilt at
    /// construction from the compiled constants — no `/sys` read, and no
    /// topology decision, on any path a game touches. Never replaced: the layout
    /// is a constant.
    numa_layout: NumaLayout,
    /// Which CPU each worker pins itself to and which system node its memory
    /// belongs on, for the current pool size. Rebuilt with the pool. Slot 0 is
    /// the per-`go` coordinator; `1..` are the helper threads.
    worker_plan: Arc<WorkerPlan>,
    /// Per-worker handles to the node-shared correction / pawn tables, rebuilt
    /// at every pool (re)build from [`Self::worker_plan`]. Length equals the pool
    /// size, so `[0]` is the coordinator's and `[1..]` the helpers'.
    worker_shared: Vec<Arc<SharedHistories>>,
    /// [`Self::worker_shared`] without the coordinator's slot-0 handle — worker
    /// `h + 1`'s set at index `h` — behind an [`Arc`] so a `go` hands its
    /// coordinator the list rather than a fresh copy of it. Rebuilt with the
    /// pool, alongside its source.
    helper_shared: Arc<Vec<Arc<SharedHistories>>>,
    /// The flags and counters every `go` reuses.
    handles: SearchHandles,
    /// The coordinator's collection buffers. `None` only while a search holds
    /// them, exactly like [`Self::histories`].
    vote_buffers: Option<VoteBuffers>,
    /// The directory a relative `eval_dir` resolves against — the running
    /// executable's own, overridable via [`Self::set_eval_root`] so a test can
    /// present a directory of its own rather than the one it runs from.
    eval_root: PathBuf,
    /// The sysfs root the [`Self::ready`] layout check reads the running machine from
    /// — `/sys`, overridable via [`Self::set_sysfs_root`] so a test can hold
    /// the binary against a machine other than the one it is running on.
    sysfs_root: PathBuf,
    /// The CPUs this process may run on, captured at startup and held against
    /// the CPUs the compiled thread plan pins workers to by the [`Self::ready`] check
    /// — which asks nothing of a build that pins none. Overridable via
    /// [`Self::set_startup_affinity`] so a test can present a confined process
    /// without confining the test process itself.
    startup_affinity: BTreeSet<usize>,
}

impl<P: EngineSink> Engine<P> {
    /// An engine whose book / `rtime` PRNG stream is seeded from process
    /// entropy, so every process run differs. Callers wanting reproducible book
    /// selection or `rtime` budgets construct via [`Self::with_book_seed`].
    pub fn new(sink: P) -> Self {
        Self::with_book_seed(sink, Prng::random_seed())
    }

    /// An engine with an explicit book-PRNG session seed. The entropy default
    /// ([`Self::new`]) delegates here with [`Prng::random_seed`]; tests inject a
    /// fixed seed for deterministic book / `rtime` behaviour.
    pub fn with_book_seed(sink: P, book_seed: u64) -> Self {
        let settings = Settings::new();
        let threads = settings.threads();
        // Rebuild the machine's NUMA layout and the worker → CPU assignment from
        // the constants this binary was built with: both were decided when the
        // binary was, so nothing here reads `/sys` and nothing decides a
        // topology. Whether the machine still matches is the readiness check's
        // question, asked once, before a game.
        let numa_layout =
            NumaLayout::from_const(settings.numa_node_cpus(), settings.numa_system_nodes());
        let worker_plan = Arc::new(WorkerPlan::of(
            settings.worker_cpus(),
            settings.worker_system_nodes(),
            threads,
        ));
        let pool = ThreadPool::with_binding(threads, Arc::clone(&worker_plan));
        // The pool decides the worker count — it refuses a size below one — so
        // what the counters and buffers are sized by is the pool, not the
        // setting it was asked for.
        let workers = pool.size();
        // Build the per-node shared correction / pawn tables and give the
        // coordinator (worker 0) its node's set.
        let worker_shared = build_worker_shared(&worker_plan);
        let helper_shared = helper_slice(&worker_shared);
        let histories = Some(WorkerHistories::with_shared(Arc::clone(&worker_shared[0])));
        // The coordinator's bundle is built (and filled) right here, on the
        // command thread — so place it explicitly; see
        // `place_coordinator_histories`.
        place_coordinator_histories(histories.as_ref(), worker_plan.system_nodes[0]);
        Self {
            sink,
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
            last_position: RetainedPosition::new(),
            pending_position: RetainedPosition::new(),
            scratch_pos: Position::startpos(),
            legal_buf: Vec::with_capacity(MAX_LEGAL_MOVES),
            pseudo_buf: Vec::with_capacity(MAX_LEGAL_MOVES),
            last_go: None,
            pool,
            pool_threads: threads,
            numa_layout,
            worker_plan,
            worker_shared,
            helper_shared,
            handles: SearchHandles::new(workers),
            vote_buffers: Some(VoteBuffers::for_pool(workers)),
            eval_root: network_file::executable_directory(),
            sysfs_root: PathBuf::from(SYSFS_ROOT),
            startup_affinity: yorkie_numa::startup_affinity().clone(),
        }
    }

    /// The position a search would run from.
    pub fn position(&self) -> &Position {
        &self.pos
    }

    /// The shared transposition table this session searches against.
    pub fn transposition_table(&self) -> &'static TranspositionTable {
        self.tt
    }

    /// Override the directory a relative `eval_dir` resolves against.
    ///
    /// An engine finds its evaluation file beside itself, which is the one
    /// place a session cannot move: `eval_dir` is compiled in, and no build has
    /// an option surface to point it elsewhere. A test that has to drive a
    /// session against a network of its own — a synthetic one, or none at all —
    /// names the directory here, the same way it names a machine through
    /// [`Self::set_sysfs_root`].
    pub fn set_eval_root(&mut self, root: PathBuf) {
        self.eval_root = root;
    }

    /// Override the sysfs root the [`Self::ready`] layout check reads, so a test
    /// can present a machine other than the one it runs on and see the check
    /// refuse it.
    pub fn set_sysfs_root(&mut self, root: PathBuf) {
        self.sysfs_root = root;
    }

    /// Override the CPU set the [`Self::ready`] layout check takes for this
    /// process's startup affinity, so a test can drive a session as a confined
    /// process while the test process itself stays where it is.
    pub fn set_startup_affinity(&mut self, cpus: BTreeSet<usize>) {
        self.startup_affinity = cpus;
    }

    /// Whether a search is still in flight.
    pub fn search_is_running(&self) -> bool {
        self.search.is_some()
    }

    /// Reclaim a search that has already replied but not yet been joined, so the
    /// natural `go … → reply → inspect` sequence finds the session idle.
    ///
    /// "Already replied" is [`ActiveSearch::reply_sent`], not
    /// [`JoinHandle::is_finished`]: the coordinator writes its reply and *then*
    /// unwinds. `is_finished` still stands beside it for the searches that end
    /// without a reply.
    #[cfg(feature = "verbose3")]
    pub fn reclaim_replied_search(&mut self) {
        if self.search.as_ref().is_some_and(|active| {
            active.reply_sent.load(Ordering::Relaxed) || active.handle.is_finished()
        }) {
            self.finish_search_join();
        }
    }

    /// Install `pos` as the search root directly, for the measurement command
    /// that walks a list of positions rather than replaying a game.
    #[cfg(feature = "verbose3")]
    pub fn set_search_position(&mut self, pos: Position) {
        self.pos = pos;
    }

    /// Fix the evaluation-noise seed for a measurement run, whose reported node
    /// count two runs — and two processes — have to agree on.
    #[cfg(all(feature = "verbose3", feature = "random"))]
    pub fn set_bench_random_seed(&mut self) {
        self.random_seed = BENCH_RANDOM_SEED;
    }

    /// Rebuild the worker pool at `threads` workers — the one value a
    /// measurement command carries that still means something here.
    #[cfg(feature = "verbose3")]
    pub fn resize_pool(&mut self, threads: usize) {
        self.pool_threads = threads;
        self.rebuild_pool();
    }

    /// `"Using N thread[s] on CPUs <list>"` for the live pool.
    #[cfg(feature = "verbose3")]
    pub fn thread_allocation_information(&self) -> String {
        thread_allocation_information_as_string(self.pool.size(), &self.worker_plan)
    }

    /// If a search worker is running, request its stop and join it, reclaiming
    /// the session-owned state. Idempotent: a no-op when idle.
    ///
    /// Joining is also what leaves this thread alone with the shared
    /// transposition table, which is what a new game's clear wants.
    pub fn finish_search_join(&mut self) {
        if let Some(active) = self.search.take() {
            active.stop.store(true, Ordering::Relaxed);
            let state = active
                .handle
                .join()
                .expect("search worker thread must not panic");
            self.histories = Some(state.histories);
            self.vote_buffers = Some(state.vote_buffers);
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
    /// 1. A memory policy naming exactly the system NUMA nodes this binary's
    ///    workers run on ([`table_placement`]): `MPOL_INTERLEAVE` over the set
    ///    when they span several, so no node's memory controller carries the
    ///    whole engine's probe traffic, and a preference for the one node when
    ///    they share it. The policy goes on first, because it decides where the
    ///    *first touch* of every page lands.
    /// 2. `madvise(MADV_HUGEPAGE)`, so the region a huge-page boundary starts is
    ///    actually backed by huge pages.
    ///
    /// Both are best-effort. A kernel without `CONFIG_NUMA`, a seccomp filter or
    /// a restricted cgroup refuses one or both, and the table then keeps the
    /// process default policy and ordinary pages — slower, never wrong. The
    /// outcome is reported rather than assumed, since a tournament host silently
    /// falling back is exactly what an operator wants to see before the game and
    /// not after it — so the line is an initialisation-phase notice, present in
    /// every build like the rest of them. It names the size and the
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
        let placement = match table_placement(&self.worker_plan) {
            TablePlacement::OnNode(node) => format!(
                "preferred on node {node} {}",
                outcome(mempolicy::prefer_region_on_node(addr, span, node))
            ),
            TablePlacement::AcrossNodes(nodes) => format!(
                "interleave on nodes {} {}",
                yorkie_numa::format_cpu_list(nodes.iter().copied()),
                outcome(mempolicy::interleave_region_over_nodes(addr, span, &nodes))
            ),
        };
        let huge = yorkie_storage::advise_huge_pages(addr, span);

        self.sink.notice(&format!(
            "transposition table: {} MiB; {placement}; huge pages {}",
            yorkie_storage::TABLE_BYTES / (1024 * 1024),
            outcome(huge),
        ))
    }

    /// Recompute the worker → CPU assignment for the current pool size and
    /// rebuild the worker pool with it. Every pool (re)build routes through here
    /// so [`Self::worker_plan`] stays consistent with the live pool. Helper
    /// threads pin once at spawn; the per-`go` coordinator pins at each `go`.
    ///
    /// Callers must have joined any running search first, since a resize
    /// destroys and recreates the helper threads.
    fn rebuild_pool(&mut self) {
        let requested = self.pool_threads;
        self.worker_plan = Arc::new(WorkerPlan::of(
            self.settings.worker_cpus(),
            self.settings.worker_system_nodes(),
            requested,
        ));
        // Rebuild the per-node shared correction / pawn tables from the fresh
        // assignment, so every pool rebuild resets them. The coordinator's own
        // game-scoped per-worker tables persist, so only its shared handle is
        // swapped.
        self.worker_shared = build_worker_shared(&self.worker_plan);
        self.helper_shared = helper_slice(&self.worker_shared);
        if let Some(h) = self.histories.as_mut() {
            h.set_shared(Arc::clone(&self.worker_shared[0]));
        }
        // The coordinator's per-worker tables outlive a pool rebuild and were
        // faulted on the command thread, so re-assert their placement for the fresh
        // assignment. Helpers need nothing here: they are respawned and each
        // allocates its own bundle on-thread after pinning.
        place_coordinator_histories(self.histories.as_ref(), self.worker_plan.system_nodes[0]);
        self.pool
            .set_with_binding(requested, Arc::clone(&self.worker_plan));
        // The per-worker counters and the coordinator's collection buffers are
        // sized by the worker count, so the rebuilt pool is where they are
        // resized — nowhere else changes it, and no search runs across a
        // rebuild.
        let size = self.pool.size();
        self.handles.fit_to_pool(size);
        if let Some(buffers) = self.vote_buffers.as_mut() {
            buffers.fit_to_pool(size);
        }
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
        // between two readiness handshakes is itself a reason to reload.
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
            self.sink.notice(notice)?;
        }

        let mut books: Vec<Book> = Vec::new();
        for name in &names {
            // Resolve a `.db` whose file is absent to its `.ybb` sibling. The
            // pin applies this per name inside `MemoryBook::read_book`; for a
            // numbered name it is always a no-op (the enumeration already
            // proved the file exists).
            let resolved = resolve_book_filename_with_ybb_fallback(name);
            if &resolved != name {
                self.sink.notice(&format!(
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
                self.sink
                    .notice(&format!("unsupported book format : {}", resolved.display()))?;
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
                    self.sink
                        .notice(&format!("book loaded : {count} positions"))?;
                }
                Err(e) => {
                    // Mirrors the reference's open/validate failure → this name is left
                    // out of the priority list.
                    self.sink.notice(&format!("book load failed : {e}"))?;
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

    /// Make the engine ready to play: hold the machine against the layout this
    /// binary was built for, place the transposition table, load the opening
    /// book and read the evaluation network into the memory the machine calls
    /// for.
    ///
    /// The heavy half of a readiness handshake, and the caller is expected to
    /// keep the host alive across it. Returning the outcome rather than
    /// answering it is what lets the caller emit its terminal reply *after* any
    /// keep-alive helper has stopped, so the reply never races it.
    pub fn ready(&mut self) -> io::Result<ReadyOutcome> {
        // Before anything is allocated for a machine: is this the machine? The
        // whole thread plan was folded from the layout the binary was built on,
        // so a difference here is a wrong answer that is available now, and one
        // no later stage would report.
        if let Some(reason) = self.numa_layout_refusal() {
            return Ok(ReadyOutcome::LayoutMismatch(reason));
        }
        // Which CPUs this binary plays on. An operator running several engines
        // on one machine has to be able to see, from the engine itself, that
        // each got the CPUs meant for it — so the line is an
        // initialisation-phase notice, present in every build.
        self.sink.notice(&format!(
            "workers on CPUs {}",
            yorkie_numa::format_cpu_list(self.worker_plan.distinct_cpus())
        ))?;
        // The machine is the one the binary was built for, so the workers'
        // nodes are the ones the table belongs on. Done before anything else
        // here, so the policy is in force before the first page of it is
        // touched.
        self.place_transposition_table()?;
        // Load / reload the opening book (the reference does this in `isready`).
        self.reload_book()?;
        let path = self.evaluation_file_path();

        // Idempotent: a repeat reuses what is already in memory — no
        // second read of the file, and on a multi-node machine no second copy
        // into the regions the first one is being read from.
        if self.eval.as_ref().is_some_and(|e| e.path == path) {
            return Ok(ReadyOutcome::Ready);
        }

        match self.place_evaluation_network(&path) {
            Ok((eval, warnings, placement)) => {
                // Surface the complaints the conversion had about the source
                // network (hash mismatches) as notices before the readiness
                // reply, mirroring the reference `LoadAndShare` /
                // `Detail::ReadParameters` diagnostics. A clean network carries
                // none, so a correct one emits nothing new.
                for warning in &warnings {
                    self.sink.notice(warning)?;
                }
                self.sink.notice(&placement)?;
                self.eval = Some(eval);
                Ok(ReadyOutcome::Ready)
            }
            Err(e) => Ok(ReadyOutcome::LoadFailed(e.to_string())),
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
    /// One mapping on a single-node machine; one copy per system NUMA node this
    /// binary's workers run on otherwise — [`config::EVAL_REGION_NODES`], the
    /// same list the worker's network type is selected from — each region
    /// placed on its node *before* the copy so every page's first touch lands
    /// there, and hinted for huge pages either way. Every placement call is
    /// best-effort: a kernel without `CONFIG_NUMA`, a seccomp filter or a
    /// restricted cgroup refuses one, and the network then keeps the process's
    /// own policy and ordinary pages — slower, never wrong. The line says
    /// which, in every build, since a tournament host silently falling back is
    /// what an operator wants to see before the game rather than after it.
    fn place_evaluation_network(
        &mut self,
        path: &Path,
    ) -> Result<(LoadedEval, Vec<String>, String), NnueError> {
        // Nothing may still be reading the regions when they are filled: they
        // are the process's only storage for the network. Any search has been
        // joined by the caller; clearing this here is what leaves the engine in
        // the "no network loaded" state until the fill succeeds.
        self.eval = None;

        let outcome = |accepted: bool| if accepted { "applied" } else { "refused" };
        let mut warnings = Vec::new();

        let placement = if network_file::SHARED_MAPPING {
            // SAFETY: nothing is pointed at the region — no search that could
            // have read it survives, each one having been joined before the
            // readiness handshake reached here.
            warnings = unsafe { network_file::map_shared(path)? };
            let (addr, span) = network_file::Region::<0>::new().parameter_region();
            let huge = yorkie_storage::advise_huge_pages(addr, span);
            format!("one shared mapping; huge pages {}", outcome(huge))
        } else {
            let mut reported = Vec::new();
            for (slot, node) in config::EVAL_REGION_NODES.iter().copied().enumerate() {
                let (addr, span) = network_file::region_backing(slot);
                let placed = format!(
                    "node {node} {}",
                    outcome(mempolicy::migrate_region_to_node(addr, span, node))
                );
                let huge = yorkie_storage::advise_huge_pages(addr, span);
                // SAFETY: as the shared mapping above.
                let w = unsafe { network_file::load_into_region(slot, path)? };
                // Every region reads the same file, so its complaints are the
                // same each time; they are worth reporting once.
                warnings = w;
                reported.push(format!("{placed}, huge pages {}", outcome(huge)));
            }
            format!("one copy on {}", reported.join("; "))
        };

        Ok((
            LoadedEval {
                path: path.to_path_buf(),
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
    /// The live layout is resolved exactly as the build resolved it, over every
    /// online CPU, so what is compared is machine against machine and not
    /// machine against process. A tree that cannot be read is itself a refusal —
    /// an unverifiable layout is not a matching one.
    fn numa_layout_refusal(&self) -> Option<String> {
        let opts = match yorkie_numa::machine_sysfs_options(&self.sysfs_root) {
            Ok(opts) => opts,
            Err(e) => return Some(e),
        };
        machine_refusal(
            &NumaLayout::of_machine(&opts),
            &self.startup_affinity,
            &self.worker_plan,
            &self.numa_layout,
        )
    }

    /// Replay one position command into the scratch buffers and, if every move
    /// of it was legal, install the result as the current position and the
    /// retained command.
    ///
    /// A refused command leaves both untouched — the input-validation contract:
    /// the prior position must survive a malformed line — which is why the work
    /// happens in the scratch pair and the two installs are the last thing done.
    /// The refusal names what it refused, borrowed from the caller's own text,
    /// so a protocol layer can report it without copying anything.
    pub fn set_position<'a>(
        &mut self,
        sfen: PositionSfen<'a>,
        moves: &'a str,
    ) -> Result<(), PositionRefusal<'a>> {
        let Self {
            pending_position: pending,
            scratch_pos: scratch,
            legal_buf,
            pseudo_buf,
            ..
        } = self;
        pending.set_start(sfen);
        pending.start_into(scratch).map_err(PositionRefusal::Sfen)?;
        for mv in moves.split_whitespace() {
            if pending.moves.len() == MAX_POSITION_MOVES {
                return Err(PositionRefusal::TooManyMoves);
            }
            let parsed =
                parse_usi_move(mv, scratch).map_err(|_| PositionRefusal::IllegalMove(mv))?;
            legal_buf.clear();
            scratch.generate_legal_all_with(pseudo_buf, legal_buf);
            if !legal_buf.contains(&parsed) {
                return Err(PositionRefusal::IllegalMove(mv));
            }
            scratch.do_move(parsed);
            pending.moves.push(parsed);
        }
        // Both installs are a swap: what they displace becomes the next
        // command's scratch, buffers and all. The retained command
        // (`last_position_cmd_string`) is what the Stochastic_Ponder rewind /
        // re-issue replays.
        std::mem::swap(&mut self.pos, &mut self.scratch_pos);
        std::mem::swap(&mut self.last_position, &mut self.pending_position);
        Ok(())
    }

    /// Rebuild the retained `position` command's first `plies` moves and install
    /// the result as the current position — the Stochastic_Ponder rewind and
    /// re-issue, which reach for a position the command list already describes.
    ///
    /// The retained moves were verified when the command arrived, so only its
    /// SFEN can still be refused; that leaves the current position untouched.
    fn install_retained_position(&mut self, plies: usize) {
        let Self {
            last_position: retained,
            scratch_pos: scratch,
            ..
        } = self;
        if retained.start_into(scratch).is_err() {
            return;
        }
        for &mv in &retained.moves[..plies] {
            scratch.do_move(mv);
        }
        std::mem::swap(&mut self.pos, &mut self.scratch_pos);
    }

    /// Start a new game: reclaim any search, empty the table, rebuild the
    /// history tables and the pool, and reset every per-game carry-forward (the
    /// reference `search_clear`).
    pub fn new_game(&mut self) {
        // Reclaim any running search, then reset the game state: startpos, an
        // emptied table, and fresh history tables (the reference
        // `search_clear`). Clearing the table also resets its generation; the next
        // `go` bumps it again via `run_root`.
        self.finish_search_join();
        self.pos.reset_startpos();
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
        self.last_position.reset();
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

    /// Start a search under `params`.
    ///
    /// Returns as soon as the coordinator thread is running: the reply reaches
    /// the sink from there. [`GoOutcome::NoNetwork`] means nothing was started,
    /// and the caller owes the host the reply itself.
    pub fn go(&mut self, params: GoParams) -> GoOutcome {
        // A new `go` supersedes any lingering search; reclaim its state first.
        self.finish_search_join();

        // Retain this `go` for a later Stochastic_Ponder re-issue.
        self.last_go = Some(params.clone());

        // Stochastic_Ponder `go ponder`: ponder one move earlier than the
        // retained position (drop its last move); `ponderMode` stays set.
        if params.ponder && self.settings.stochastic_ponder() {
            self.apply_stochastic_ponder_rewind();
        }

        // The ply the search actually runs at (rewound under Stochastic_Ponder),
        // carried so a completed real search updates `last_game_ply`.
        let game_ply = self.pos.ply() as i32;

        // Build the coordinator job (option-seeded limits, all per-`go`
        // snapshots). `None` means no network is loaded, which is the caller's
        // to answer.
        let Some(job) = self.prepare_coordinator_job(
            params,
            #[cfg(feature = "verbose2")]
            false,
        ) else {
            return GoOutcome::NoNetwork;
        };

        // The handles the main loop signals on `stop` / `ponderhit` / a
        // Stochastic_Ponder teardown; cloned out of the job before it moves into
        // the worker thread.
        let stop_for_active = Arc::clone(&job.stop);
        let ponder_for_active = job.ponder.as_ref().map(Arc::clone);
        let suppress_for_active = Arc::clone(&job.suppress_reply);
        #[cfg(feature = "verbose3")]
        let sent_for_active = Arc::clone(&job.reply_sent);
        // The coordinator's own system node, resolved before the thread starts
        // so the network type it searches with is selected once per `go`.
        let node = self.worker_plan.system_nodes[0];
        let handle = std::thread::spawn(move || {
            let outcome = with_eval_network!(node, |net| run_coordinated(net, job));
            SearchState {
                histories: outcome.histories,
                time_state: outcome.time_state,
                vote_buffers: outcome.vote_buffers,
            }
        });

        self.search = Some(ActiveSearch {
            handle,
            stop: stop_for_active,
            ponder: ponder_for_active,
            suppress: suppress_for_active,
            #[cfg(feature = "verbose3")]
            reply_sent: sent_for_active,
            game_ply,
        });
        GoOutcome::Started
    }

    /// Stochastic_Ponder `go ponder` rewind: reconstruct the retained position
    /// with its last move dropped and install it as the search root. A
    /// best-effort trim — an empty move list (nothing to rewind) or a rebuild
    /// failure leaves the current position untouched.
    fn apply_stochastic_ponder_rewind(&mut self) {
        let plies = self.last_position.moves.len();
        if plies == 0 {
            return;
        }
        self.install_retained_position(plies - 1);
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
        limits: GoParams,
        #[cfg(feature = "verbose2")] disable_pv_interval: bool,
    ) -> Option<CoordinatorJob<P>> {
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

        // No network loaded (a non-compliant host `go`, or a measurement run
        // whose readiness handshake never succeeded). The caller answers for
        // this position.
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
        // DELIBERATE DIVERGENCE (see [`GoParams::mate`]): the reference
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
                    ply: self.pos.ply() as i32,
                    start_time: now,
                },
                #[cfg(feature = "verbose2")]
                &mut prng,
            );
            #[cfg(feature = "verbose1")]
            if tm.mtg_error {
                self.sink
                    .diagnostic("Error! : MaxMovesToDraw is too small.");
            }
            Some(TimeControl {
                tm,
                #[cfg(feature = "verbose2")]
                use_time_management,
                #[cfg(feature = "verbose2")]
                movetime,
                best_previous_score: best_prev_score,
                best_previous_average_score: best_prev_average_score,
                previous_time_reduction: self.previous_time_reduction,
            })
        };
        // Clear the session's flags and counters for this search, and take the
        // `go ponder` signal (`ponderMode`) it hands the main worker's control,
        // so that it — and the coordinator's hold loop — can be driven by a
        // later `ponderhit`. `None` on every non-ponder `go`.
        let ponder = self.handles.arm(limits.ponder);
        let control = SearchControl {
            stop: Some(Arc::clone(&self.handles.stop)),
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

        // The persistent helper slots to dispatch to. The pool is never resized
        // while a coordinator runs (every resize path calls
        // `finish_search_join` first), so these stay valid for this whole `go`.
        let helper_slots = self.pool.helper_slots();
        // Each helper's node-shared tables: worker `h + 1` gets
        // `worker_shared[h + 1]`, which is why the list is held without the
        // coordinator's slot-0 handle. It is rebuilt only with the pool, which
        // never happens mid-`go`, so it stays aligned with these helpers for the
        // whole search.
        let helper_shared = Arc::clone(&self.helper_shared);

        // The shared table is a `static`, so the coordinator gets the same
        // reference and nothing is handed over; the main histories are still
        // lent take-and-return and reclaimed on join. Helper histories live in
        // the pool threads.
        let tt = self.tt;
        let histories = self
            .histories
            .take()
            .expect("session histories present when idle");
        let vote_buffers = self
            .vote_buffers
            .take()
            .expect("session collection buffers present when idle");
        let sink = self.sink.clone();
        let stop = control
            .stop
            .clone()
            .expect("stop flag installed just above");
        let pos = self.pos.clone();

        // Book state for this `go`: the loaded book (cheap `Arc` clone), the
        // `USI_OwnBook` gate, a fresh seed, and whether a book reply must be
        // held for `stop`/`ponderhit` (`go ponder`/`infinite`). The selection
        // settings are [`BOOK_CONFIG`] and travel with no job.
        let book = self.book.as_ref().map(Arc::clone);
        let own_book = self.settings.usi_own_book();
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
        // no reply (nor final PV) for this search. Raised alongside it, the flag
        // saying this `go`'s reply reached the output sink; the table-inspection
        // commands' idle check is that one's only reader, so only their feature
        // has it. Both were cleared for this search by `arm` above.
        let suppress_reply = Arc::clone(&self.handles.suppress_reply);
        #[cfg(feature = "verbose3")]
        let reply_sent = Arc::clone(&self.handles.reply_sent);

        // Precompute the entering-king thresholds from the root position,
        // mirroring the reference `set_ekr` on the root worker. The rule itself
        // is compiled in; only its two handicap-aware forms read a position at
        // all. The material total is invariant across the search, so every
        // worker shares this one snapshot.
        let entering_king = EnteringKingConfig::new(&pos);

        // The per-`go` coordinator (worker slot 0) pins itself to its assigned
        // CPU — and points its allocations at that CPU's node — at the start of
        // every `go`. It is spawned per `go`, so it re-pins each time: same
        // target CPU, idempotent.
        let worker_plan = Arc::clone(&self.worker_plan);

        Some(CoordinatorJob {
            tt,
            pos,
            #[cfg(feature = "verbose2")]
            depth,
            #[cfg(feature = "verbose2")]
            use_voting,
            control,
            stop,
            histories,
            vote_buffers,
            #[cfg(feature = "verbose2")]
            node_slots: Arc::clone(&self.handles.node_slots),
            bmc_slots: Arc::clone(&self.handles.bmc_slots),
            helper_slots,
            helper_shared,
            worker_plan,
            book,
            own_book,
            book_seed,
            ponder,
            #[cfg(feature = "verbose2")]
            infinite,
            suppress_reply,
            #[cfg(feature = "verbose3")]
            reply_sent,
            entering_king,
            #[cfg(feature = "verbose2")]
            mate_mode,
            #[cfg(feature = "random")]
            random_seed: self.random_seed,
            #[cfg(feature = "verbose2")]
            multi_pv,
            #[cfg(feature = "verbose2")]
            pv_config,
            sink,
        })
    }

    /// Run one measurement position synchronously on the calling thread and
    /// return its total searched node count across all workers.
    ///
    /// Only the driving is synchronous — a measurement run needs each position's
    /// node total before moving on. `None` means no network is loaded, which is
    /// the caller's to answer, exactly as for [`Self::go`].
    #[cfg(feature = "verbose3")]
    pub fn bench_run_one(&mut self, params: GoParams) -> Option<u64> {
        // The measurement command is `verbose3`, so the PV interval it disables
        // always exists.
        let job = self.prepare_coordinator_job(params, true)?;
        let node = self.worker_plan.system_nodes[0];
        let outcome = with_eval_network!(node, |net| run_coordinated(net, job));
        // Return the session state the job borrowed (the async path reclaims
        // this on join; here we hand it straight back). Bench uses fixed
        // depth / nodes / movetime, so `time_state` is irrelevant to it.
        self.histories = Some(outcome.histories);
        self.vote_buffers = Some(outcome.vote_buffers);
        Some(outcome.nodes)
    }

    /// Ask the running search to abort promptly, releasing a held reply.
    ///
    /// It emits its reply and its state is reclaimed on the next command that
    /// needs it. With no search running this is a silent no-op.
    pub fn stop(&mut self) {
        if let Some(active) = &self.search {
            active.stop.store(true, Ordering::Relaxed);
        }
    }

    /// The opponent played the predicted move.
    ///
    /// Plain path: clear the ponder flag so the pondering search continues under
    /// time management; a held book reply's coordinator wait loop polls the same
    /// flag, so this releases it. Stochastic_Ponder path: tear the rewound
    /// ponder search down without emitting, restore the real position, and
    /// re-issue the retained `go` with `ponder` stripped — which is the one way
    /// this can report [`GoOutcome::NoNetwork`], from the re-issue.
    pub fn ponderhit(&mut self) -> GoOutcome {
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
        GoOutcome::Started
    }

    /// Stochastic_Ponder `ponderhit`: suppress the rewound ponder search's
    /// output, stop and join it, re-apply the real current position, and
    /// re-issue the retained `go` without its `ponder` token — a normal timed
    /// search of which exactly one reply reaches the host.
    fn stochastic_ponderhit(&mut self) -> GoOutcome {
        // Suppress the rewound search's reply before stopping it.
        if let Some(active) = &self.search {
            active.suppress.store(true, Ordering::Relaxed);
        }
        self.finish_search_join();

        // Re-apply the real (current) position.
        self.install_retained_position(self.last_position.moves.len());

        // Re-issue the retained `go` with `ponder` stripped.
        if let Some(mut go) = self.last_go.clone() {
            go.ponder = false;
            return self.go(go);
        }
        GoOutcome::Started
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

/// Answer a book hit the way the reference does on `search_skipped`: the
/// surviving candidates first, then — after the ponder/infinite hold — the
/// terminal reply.
///
/// Under `go ponder` / `go infinite` the reply is held until `stop` or a
/// `ponderhit`, reusing the async-stop machinery rather than busy-waiting.
/// `time_ms` is stamped once, when the book answered, so the hold does not
/// inflate the elapsed time attributed to the reply.
///
/// The candidate report is `verbose2`; the hold and the reply are not, so a
/// default build answers a book hit with the move and nothing else.
#[allow(clippy::too_many_arguments)]
fn emit_book_hit<P: EngineSink>(
    sink: &P,
    hit: &BookHit,
    #[cfg(feature = "verbose2")] hashfull: u32,
    #[cfg(feature = "verbose2")] time_ms: u64,
    ponder: Option<&Arc<PonderSignal>>,
    #[cfg(feature = "verbose2")] infinite: bool,
    stop: &AtomicBool,
    suppress_reply: &AtomicBool,
    #[cfg(feature = "verbose3")] sent: &AtomicBool,
) {
    // Reported immediately, like the reference's in-probe isRoot block.
    #[cfg(feature = "verbose2")]
    sink.book_candidates(hit, hashfull, time_ms);

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
    if suppress_reply.load(Ordering::Relaxed) {
        return;
    }

    sink.book_reply(
        hit,
        #[cfg(feature = "verbose2")]
        hashfull,
        #[cfg(feature = "verbose2")]
        time_ms,
        #[cfg(feature = "verbose3")]
        sent,
    );
}

/// Everything one helper needs to run its own iterative deepening for a single
/// `go`. The heavy state is shared: the stop flag and the per-worker node
/// counters behind [`Arc`], the transposition table as the one `static` every
/// worker reads. The network is not here at all — which one this helper reads
/// was decided when its thread was spawned, and is the type its search was
/// compiled for. The position and the root-move list are cheap per-helper
/// copies (the reference `start_thinking` copies the root-move list to every
/// worker).
struct HelperJob {
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
    /// The one shared stop flag every worker polls (the engine installs it).
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
    /// The entering-king declaration thresholds snapshot for this `go`.
    entering_king: EnteringKingConfig,
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
    /// lifetime (the engine rebuilds it only on a pool rebuild, which recreates
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
fn helper_loop<N: NetworkParams>(net: N, slot: Arc<HelperSlot>) {
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
    /// One coordination slot per helper (`size − 1` of them). Behind an [`Arc`]
    /// so the coordinator that dispatches to and collects from them is handed
    /// the list itself rather than a copy of it per `go`; the list is replaced
    /// only by a rebuild.
    slots: Arc<Vec<Arc<HelperSlot>>>,
    /// The helper threads, joined on resize / teardown.
    handles: Vec<JoinHandle<()>>,
}

impl ThreadPool {
    /// Build a pool of `size` slots (one main + `size − 1` helpers), spawning the
    /// helper threads parked and idle, each pinned to the CPU a plan built for
    /// this process's own affinity gives it — the pool unit tests use this, so
    /// they never pin a test thread to a CPU the test runner may not use.
    #[cfg(test)]
    fn new(size: usize) -> Self {
        Self::with_binding(size, Arc::new(WorkerPlan::of_allowed_cpus(size)))
    }

    /// Build a pool of `size` slots with a worker plan. Each helper thread
    /// (worker `1..`) pins itself to its assigned CPU once at spawn, before it
    /// parks.
    fn with_binding(size: usize, plan: Arc<WorkerPlan>) -> Self {
        let mut pool = ThreadPool {
            slots: Arc::new(Vec::new()),
            handles: Vec::new(),
        };
        pool.set_with_binding(size, plan);
        pool
    }

    /// Resize to `size` slots over this process's own CPUs — used only by the
    /// pool unit tests.
    #[cfg(test)]
    fn set(&mut self, size: usize) {
        self.set_with_binding(size, Arc::new(WorkerPlan::of_allowed_cpus(size)));
    }

    /// Resize to `size` slots, mirroring the reference `ThreadPool::set`: it
    /// never diffs, always joining and destroying the current helpers and then
    /// recreating the requested number with fresh histories. Callers wait for
    /// any running search to finish first, so every helper is parked when this
    /// runs.
    ///
    /// Each helper pins itself to its assigned CPU at spawn and makes that CPU's
    /// node its preferred allocation target. The memory half matters because the
    /// affinity pin alone places nothing: under `numactl --interleave=all` the
    /// inherited process policy would still spread the helper's private history
    /// tables across every node. Setting the preference at spawn is what makes
    /// [`helper_loop`]'s lazy allocation land node-locally, and it is per-thread,
    /// so the shared transposition table's interleave is untouched.
    fn set_with_binding(&mut self, size: usize, plan: Arc<WorkerPlan>) {
        self.shutdown();
        let size = size.max(1);
        let mut slots = Vec::with_capacity(size - 1);
        for worker_id in 1..size {
            let slot = Arc::new(HelperSlot::new());
            let slot_for_thread = Arc::clone(&slot);
            let plan_for_thread = Arc::clone(&plan);
            self.handles.push(std::thread::spawn(move || {
                let node = plan_for_thread.system_nodes[worker_id];
                yorkie_numa::pin_current_thread_to_cpu_with_local_memory(
                    plan_for_thread.cpus[worker_id],
                    node,
                );
                // The one place this helper's node becomes the network type its
                // search is compiled for. Once per thread, before it parks.
                with_eval_network!(node, |net| helper_loop(net, slot_for_thread));
            }));
            slots.push(slot);
        }
        self.slots = Arc::new(slots);
    }

    /// Ask every helper to exit and join it, leaving only the main slot.
    /// Idempotent. Every helper must be parked first (guaranteed by the
    /// `finish_search_join` every caller runs before a resize / teardown), so the
    /// `Exit` is never overwritten by a late `Finished` write.
    fn shutdown(&mut self) {
        for slot in self.slots.iter() {
            *slot.lock() = SlotState::Exit;
            slot.cv.notify_all();
        }
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
        self.slots = Arc::new(Vec::new());
    }

    /// The configured pool size: the main-worker slot plus the live helpers.
    fn size(&self) -> usize {
        self.slots.len() + 1
    }

    /// The helper slot list a coordinator dispatches to and collects from for one
    /// `go`.
    fn helper_slots(&self) -> Arc<Vec<Arc<HelperSlot>>> {
        Arc::clone(&self.slots)
    }
}

impl Drop for ThreadPool {
    /// A session ending drops the engine, which drops the pool; join every helper so
    /// no OS thread is leaked.
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Which CPU each worker runs on, and which system NUMA node its private memory
/// belongs to. Held behind an [`Arc`] so a pool rebuild can cheaply hand each
/// helper thread a clone.
///
/// Both vectors are one entry per worker and are indexed by worker id, worker 0
/// being the per-`go` search coordinator.
struct WorkerPlan {
    cpus: Vec<usize>,
    system_nodes: Vec<NumaIndex>,
}

impl WorkerPlan {
    /// The plan for `requested` workers over the compiled assignment.
    ///
    /// The assignment holds one CPU per configured worker. Asking for more than
    /// that is only reachable through the measurement command that carries its
    /// own worker count, and the extra workers wrap around onto the same CPUs —
    /// oversubscribing the CPUs this binary owns rather than spilling onto ones
    /// another binary was given.
    fn of(cpus: &[usize], system_nodes: &[NumaIndex], requested: usize) -> Self {
        let n = cpus.len().max(1);
        let pick = |worker: usize| worker % n;
        WorkerPlan {
            cpus: (0..requested.max(1))
                .map(|w| cpus.get(pick(w)).copied().unwrap_or(0))
                .collect(),
            system_nodes: (0..requested.max(1))
                .map(|w| system_nodes.get(pick(w)).copied().unwrap_or(0))
                .collect(),
        }
    }

    /// A plan over the CPUs this process is actually allowed on, all of them on
    /// one node — what a unit test spawning a pool needs, so its helper threads
    /// pin themselves somewhere the test runner permits.
    #[cfg(test)]
    fn of_allowed_cpus(requested: usize) -> Self {
        let cpus: Vec<usize> = yorkie_numa::startup_affinity().iter().copied().collect();
        // The node has to be one this binary declares a region for, since a
        // spawned helper selects its network from it.
        let nodes = vec![config::EVAL_REGION_NODES[0]; cpus.len().max(1)];
        Self::of(&cpus, &nodes, requested)
    }

    /// The CPUs the plan uses, ascending and without repeats.
    fn distinct_cpus(&self) -> BTreeSet<usize> {
        self.cpus.iter().copied().collect()
    }

    /// The system NUMA nodes the plan's workers sit on, ascending and without
    /// repeats.
    fn distinct_system_nodes(&self) -> Vec<NumaIndex> {
        let set: BTreeSet<NumaIndex> = self.system_nodes.iter().copied().collect();
        set.into_iter().collect()
    }
}

/// How the machine this process runs on differs from the one the binary was
/// built for — one line, or `None` when they agree.
///
/// Four ways to differ, in the order a reader wants them: a different number of
/// nodes, a node holding different CPUs, a process denied a CPU a worker pins
/// itself to, and a CPU that has moved to another node.
///
/// The third is what a `taskset` or a `cpuset` around the engine can produce:
/// the machine is right, but a worker would be pinned to a CPU this process is
/// not allowed on, which the pin refuses at the point of no return — inside a
/// spawned worker, mid-game. The question is inclusion, not equality: a wider
/// affinity takes nothing away, since each worker narrows itself to its own CPU.
///
/// The fourth catches a machine that reports the same node count and CPU lists
/// in a different order, which would leave every worker's memory placed on a
/// node its CPU is not on.
fn machine_refusal(
    live: &NumaLayout,
    affinity: &BTreeSet<usize>,
    plan: &WorkerPlan,
    built: &NumaLayout,
) -> Option<String> {
    if live.nodes.len() != built.nodes.len() {
        return Some(format!(
            "this binary is built for {} NUMA node(s), this host has {}",
            built.nodes.len(),
            live.nodes.len()
        ));
    }
    for (n, (live_cpus, built_cpus)) in live.nodes.iter().zip(&built.nodes).enumerate() {
        if live_cpus != built_cpus {
            return Some(format!(
                "node {n} holds CPUs {} on this host, {} in the layout this binary is built \
                 for",
                yorkie_numa::format_cpu_list(live_cpus.iter().copied()),
                yorkie_numa::format_cpu_list(built_cpus.iter().copied())
            ));
        }
    }
    let missing: BTreeSet<usize> = plan
        .distinct_cpus()
        .into_iter()
        .filter(|cpu| !affinity.contains(cpu))
        .collect();
    if !missing.is_empty() {
        return Some(format!(
            "this process may not run on CPUs {}, which its workers are pinned to",
            yorkie_numa::format_cpu_list(missing)
        ));
    }
    for (worker, (&cpu, &node)) in plan.cpus.iter().zip(&plan.system_nodes).enumerate() {
        let live_node = live.system_node_of_cpu(cpu);
        if live_node != Some(node) {
            return Some(match live_node {
                Some(live_node) => format!(
                    "worker {worker} runs on CPU {cpu}, which is on node {live_node} on this \
                     host and on node {node} in the layout this binary is built for"
                ),
                None => {
                    format!("worker {worker} runs on CPU {cpu}, which no node of this host holds")
                }
            });
        }
    }
    None
}

/// Where the shared transposition table's pages belong: the **system** NUMA
/// nodes this binary's workers run on, and nothing wider.
#[derive(Debug, PartialEq, Eq)]
enum TablePlacement {
    /// Every worker sits on one node, so the whole table belongs on it.
    OnNode(NumaIndex),
    /// The workers span several nodes, so the table is spread over exactly
    /// those: no node's memory controller carries the whole engine's probe
    /// traffic, and no page lands where no worker probes from.
    AcrossNodes(Vec<NumaIndex>),
}

/// The placement a worker plan implies.
///
/// A binary built for part of a machine — a `cpu_assignment` naming CPUs of some
/// of its nodes, or a thread count that fits on fewer nodes than the machine has
/// — keeps its table on the part it uses, so every probe stays on a node a
/// worker runs on.
///
/// Reading the answer off the compiled constants keeps the readiness handshake
/// from asking the machine a second time.
fn table_placement(plan: &WorkerPlan) -> TablePlacement {
    let nodes = plan.distinct_system_nodes();
    match nodes.as_slice() {
        [node] => TablePlacement::OnNode(*node),
        _ => TablePlacement::AcrossNodes(nodes),
    }
}

/// Move the coordinator's session-owned history tables onto worker 0's node.
///
/// This is the one per-worker bundle the engine does **not** allocate inside the
/// worker that uses it: it is built and filled on the command thread and only then
/// lent to the per-`go` coordinator. Its pages are therefore already faulted
/// wherever the process policy put them, and no per-thread policy can move them
/// retroactively — `mbind(MPOL_BIND | MPOL_MF_MOVE)` can, so that is what this
/// does. The helpers need none of this, each allocating its own bundle after it
/// has pinned itself.
///
/// Best-effort throughout, and run at pool-(re)build time only, outside any
/// clock.
fn place_coordinator_histories(histories: Option<&WorkerHistories>, node: NumaIndex) {
    let Some(histories) = histories else {
        return;
    };
    for (addr, len) in histories.backing_regions() {
        mempolicy::migrate_region_to_node(addr, len, node);
    }
}

/// Build the per-worker handles to the node-shared correction / pawn tables, one
/// set per system NUMA node the plan's workers sit on.
///
/// Each node's set is allocated and filled *inside* a thread pinned to one of
/// that node's worker CPUs, so the pages first-touch there.
///
/// Returns one [`Arc`] per worker, each pointing at its node's table set.
fn build_worker_shared(plan: &WorkerPlan) -> Vec<Arc<SharedHistories>> {
    let counts = shared_node_counts(&plan.system_nodes);

    let mut node_shared: std::collections::BTreeMap<NumaIndex, Arc<SharedHistories>> =
        std::collections::BTreeMap::new();
    for (&node, &count) in &counts {
        let thread_count = count.next_power_of_two();
        // A CPU of that node: the first worker the plan puts there.
        let cpu = plan
            .system_nodes
            .iter()
            .position(|&n| n == node)
            .map(|worker| plan.cpus[worker])
            .expect("every counted node holds at least one worker");
        let mut built: Option<Arc<SharedHistories>> = None;
        yorkie_numa::execute_on_cpu(cpu, || {
            built = Some(Arc::new(SharedHistories::new(thread_count)));
        });
        node_shared.insert(node, built.expect("execute_on_cpu ran the closure"));
    }

    plan.system_nodes
        .iter()
        .map(|node| Arc::clone(&node_shared[node]))
        .collect()
}

/// The helpers' share of a per-worker handle list: everything past worker 0,
/// the coordinator, and nothing at all for a list that holds no worker.
///
/// The result is what a `go` hands its coordinator, so it is built where its
/// source is and shared from there.
fn helper_slice<T: Clone>(all: &[T]) -> Arc<Vec<T>> {
    Arc::new(all.get(1..).unwrap_or_default().to_vec())
}

/// The node → worker-count map the shared-history construction sizes each set
/// from. Pure — no allocation and no pinning.
fn shared_node_counts(worker_nodes: &[NumaIndex]) -> std::collections::BTreeMap<NumaIndex, usize> {
    let mut counts: std::collections::BTreeMap<NumaIndex, usize> =
        std::collections::BTreeMap::new();
    for &node in worker_nodes {
        *counts.entry(node).or_insert(0) += 1;
    }
    counts
}

// The thread-allocation diagnostic is emitted by the `verbose3` `bench`
// command and nowhere else, since that is the only command that can change the
// worker count; everything else about the assignment is a compile-time constant.
/// `"Using N thread[s] on CPUs <list>"` — the pool size, and the CPUs its
/// workers are pinned to.
#[cfg(feature = "verbose3")]
fn thread_allocation_information_as_string(threads_size: usize, plan: &WorkerPlan) -> String {
    format!(
        "Using {threads_size} {} on CPUs {}",
        if threads_size > 1 {
            "threads"
        } else {
            "thread"
        },
        yorkie_numa::format_cpu_list(plan.distinct_cpus())
    )
}

/// The bundle [`Engine::go`] hands its coordinator thread — grouped
/// into one struct so [`run_coordinated`] stays a single-argument call.
struct CoordinatorJob<P: EngineSink> {
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
    /// The main worker's game-scoped histories, returned to the engine on join.
    histories: WorkerHistories,
    /// The session's collection buffers, returned to the engine on join beside
    /// the histories. What the last search left is cleared before this one
    /// fills them; nothing here resizes them, since their room is the pool's
    /// property.
    vote_buffers: VoteBuffers,
    /// Per-worker node counters (index 0 = main, `1..` = helpers), zeroed for
    /// this `go` by the engine. The aggregate node ceiling and the final
    /// aggregated `info ... nodes` are the readers, both `verbose2`.
    #[cfg(feature = "verbose2")]
    node_slots: Arc<Vec<AtomicU64>>,
    /// Per-worker best-move-change counters, same slot-per-worker shape and
    /// zeroed on the same terms: each worker bumps its own slot at the root and
    /// the main worker folds them all each iteration.
    bmc_slots: Arc<Vec<AtomicU64>>,
    /// The persistent helper slots to dispatch to, one per helper.
    helper_slots: Arc<Vec<Arc<HelperSlot>>>,
    /// Each helper's node-shared correction / pawn tables, aligned
    /// with `helper_slots`: `helper_shared[h]` is worker `h + 1`'s
    /// [`SharedHistories`]. Handed to the helper in its [`HelperJob`]. The main
    /// worker's own shared handle already lives inside `histories`.
    helper_shared: Arc<Vec<Arc<SharedHistories>>>,
    /// The active worker plan: the coordinator pins itself to worker 0's CPU and
    /// prefers that CPU's node for memory at the start of this `go`.
    worker_plan: Arc<WorkerPlan>,
    /// The loaded opening book to probe once, if any.
    book: Option<Arc<LoadedBook>>,
    /// `USI_OwnBook` — the master gate; when off the book is never probed.
    own_book: bool,
    /// The seed for this `go`'s book PRNG (deterministic within a session).
    book_seed: u64,
    /// The shared `go ponder` signal (`Some` only for a `go ponder`): the
    /// coordinator's hold loop runs while it is active, and the reply is withheld
    /// until a ponderhit clears it (or the stop flag fires).
    ponder: Option<Arc<PonderSignal>>,
    /// `limits.infinite` — hold the reply until `stop` regardless of the clock
    /// (the SKIP_SEARCH wait loop). Only a `verbose2` build can parse the
    /// clause that sets it.
    #[cfg(feature = "verbose2")]
    infinite: bool,
    /// The Stochastic_Ponder teardown flag: when set the coordinator emits no
    /// reply (nor final PV) for this search.
    suppress_reply: Arc<AtomicBool>,
    /// Stamped `true` in the same output-lock critical section that writes this
    /// search's reply, so the engine can tell "reply is out" from "thread
    /// has exited" (see [`ActiveSearch::reply_sent`]). A suppressed reply
    /// never sets it — nothing went out. `verbose3`, like the reader.
    #[cfg(feature = "verbose3")]
    reply_sent: Arc<AtomicBool>,
    /// The entering-king declaration thresholds snapshot for this `go`.
    entering_king: EnteringKingConfig,
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
    /// Where this search's progress and reply go: a handle to the session's own
    /// output, cloned for the thread that will emit through it.
    sink: P,
}

/// The Lazy-SMP coordinator — the reference main worker's `start_searching`,
/// running on the per-`go` thread [`Engine::go`] spawns.
///
/// Hands back the main worker's histories for the engine to reclaim, the
/// aggregate searched-node total (0 for the short-circuits, and what `bench`
/// accumulates), and the time-management carry-forward, whose third element is
/// `None` for a short-circuited `go`.
struct CoordinatedOutcome {
    histories: WorkerHistories,
    /// The collection buffers this search filled, on their way back to the
    /// session that lent them.
    vote_buffers: VoteBuffers,
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

fn run_coordinated<P: EngineSink, N: NetworkParams>(
    net: N,
    job: CoordinatorJob<P>,
) -> CoordinatedOutcome {
    let CoordinatorJob {
        tt,
        pos,
        #[cfg(feature = "verbose2")]
        depth,
        #[cfg(feature = "verbose2")]
        use_voting,
        control,
        stop,
        histories,
        vote_buffers,
        #[cfg(feature = "verbose2")]
        node_slots,
        bmc_slots,
        helper_slots,
        helper_shared,
        worker_plan,
        book,
        own_book,
        book_seed,
        ponder,
        #[cfg(feature = "verbose2")]
        infinite,
        suppress_reply,
        #[cfg(feature = "verbose3")]
        reply_sent,
        entering_king,
        #[cfg(feature = "verbose2")]
        mate_mode,
        #[cfg(feature = "random")]
        random_seed,
        #[cfg(feature = "verbose2")]
        multi_pv,
        #[cfg(feature = "verbose2")]
        pv_config,
        sink,
    } = job;
    #[cfg(feature = "verbose2")]
    let multi_pv = multi_pv.max(1);

    // Pin this coordinator to its assigned CPU before any search work, and make
    // that CPU's node the allocation preference, so everything this thread
    // allocates from here on stays node-local instead of following the
    // launcher's process-wide interleave. Idempotent across the per-`go`
    // coordinator respawns. The bundle in `histories` is placed separately, at
    // pool (re)build time, because it was already faulted on the command thread.
    yorkie_numa::pin_current_thread_to_cpu_with_local_memory(
        worker_plan.cpus[0],
        worker_plan.system_nodes[0],
    );

    // One TT generation bump per `go`, on the main worker, BEFORE any helper
    // starts, so the observable single-thread sequence is the reference's:
    // bump, then search.
    tt.new_search();

    // Build the root-move list once (the reference `start_thinking`). The resign
    // and declaration-win short-circuits emit and return before any helper is
    // dispatched, exactly as `start_searching` exits before `threads.start_searching()`.
    let root_moves = generate_root_moves(&pos);
    if root_moves.is_empty() {
        sink.reply(
            Reply::Resign,
            #[cfg(feature = "verbose3")]
            &reply_sent,
        );
        return CoordinatedOutcome {
            histories,
            vote_buffers,
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
        let declared = if mv == Move::win() {
            Reply::Win
        } else {
            Reply::BestMove { mv, ponder: None }
        };
        sink.reply(
            declared,
            #[cfg(feature = "verbose3")]
            &reply_sent,
        );
        return CoordinatedOutcome {
            histories,
            vote_buffers,
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
            &BOOK_CONFIG,
            &mut prng,
        );
        #[cfg(feature = "verbose1")]
        for diag in &probed.diagnostics {
            sink.diagnostic(diag);
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
                &sink,
                &hit,
                #[cfg(feature = "verbose2")]
                tt.hashfull(0),
                #[cfg(feature = "verbose2")]
                book_time_ms,
                ponder.as_ref(),
                #[cfg(feature = "verbose2")]
                infinite,
                &stop,
                &suppress_reply,
                #[cfg(feature = "verbose3")]
                &reply_sent,
            );
            return CoordinatedOutcome {
                histories,
                vote_buffers,
                #[cfg(feature = "verbose3")]
                nodes: 0,
                time_state: skip_search_carry(),
            };
        }
    }

    // Dispatch a job to every helper (index h in `helper_slots` → worker h + 1).
    for (h, slot) in helper_slots.iter().enumerate() {
        slot.assign(HelperJob {
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
    let mut qs = QSearch::with_histories(net, tt, histories);
    qs.set_control(control);
    #[cfg(feature = "verbose2")]
    qs.set_node_tally(Arc::clone(&node_slots), 0);
    qs.set_best_move_tally(Arc::clone(&bmc_slots), 0);
    qs.set_entering_king(entering_king);
    #[cfg(feature = "verbose2")]
    qs.set_mate_mode(mate_mode);
    #[cfg(feature = "random")]
    qs.set_random(RANDOM_AMPLITUDE, random_seed);
    #[cfg(feature = "verbose2")]
    qs.set_multi_pv(multi_pv);
    #[cfg(feature = "verbose2")]
    qs.set_pv_output(pv_config, Box::new(sink.pv_output()));
    // As in the helper loop: without `verbose2` no `go` carries a ceiling below
    // the search's own maximum, because neither source of one exists.
    #[cfg(not(feature = "verbose2"))]
    let depth = SEARCH_MAX_DEPTH;
    let main_result = qs.run_worker(&pos, root_moves, depth);

    // Ponder / infinite hold (the SKIP_SEARCH wait loop): do not emit
    // the reply while still pondering or under `go infinite`. A plain
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
    let VoteBuffers {
        mut results,
        mut votes,
    } = vote_buffers;
    // The previous search's results are dropped here, at the point their room
    // is about to be filled again.
    results.clear();
    results.push(main_result);
    for slot in helper_slots.iter() {
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
        votes.clear();
        votes.extend(results.iter().map(|r| WorkerVote {
            score: r.best.score,
            pv0: r.best.pv[0],
            pv_len: r.best.pv.len(),
            completed_depth: r.completed_depth,
        }));
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
        to_cp(resign_score) <= -RESIGN_VALUE
    };

    // Final PV output before the reply. `pv_idx == lines.len()` makes every
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
            // so this re-emits the exact PV the reply will play.
            if let Some(line0) = pv_lines.get_mut(0) {
                *line0 = best.clone();
            }
            let n = pv_lines.len();
            let infos = qs.build_pv_infos(&pos, &pv_lines, n, completed_depth, n, total_nodes);
            sink.pv_block(&infos);
        }
    }

    // The reply — the ponder move is the chosen line's second PV move.
    // Resigning replaces the whole reply (the reference makes the search look
    // skipped and stacks `Move::resign()`), so it carries no ponder move.
    let reply = if resign_by_value {
        Reply::Resign
    } else {
        Reply::BestMove {
            mv: best.mv,
            ponder: (best.pv.len() >= 2).then(|| best.pv[1]),
        }
    };

    // A Stochastic_Ponder teardown stops the rewound search without emitting
    // its reply; the fresh re-issued `go` produces the single reply the
    // GUI sees. The `time_state` below is still returned so the rewound
    // search's score / ply seed the re-issue's side-flip continuity.
    if !suppress_reply.load(Ordering::Relaxed) {
        sink.reply(
            reply,
            #[cfg(feature = "verbose3")]
            &reply_sent,
        );
    }

    // Consume the search (ending the `&tt` / `&net` borrows) and reclaim the main
    // worker's histories for the engine, paired with the aggregate node total for
    // the `bench` accumulation and the time-management carry-forward.
    CoordinatedOutcome {
        histories: qs.into_histories(),
        vote_buffers: VoteBuffers { results, votes },
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

    use crate::usi::UsiSink;
    use crate::{king_shuffle, serial_tt};

    /// An engine writing into a buffer nothing reads back — for the state a test
    /// drives directly rather than through a session.
    fn idle_engine() -> Engine<UsiSink<Vec<u8>>> {
        Engine::new(UsiSink::new(Arc::new(Mutex::new(Vec::new()))))
    }

    /// What a refused command must leave behind, read off the engine itself: the
    /// position of the last accepted one, and that command still retained for
    /// the Stochastic_Ponder paths that replay it.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn a_refused_position_leaves_the_accepted_one_in_place() {
        let _tt = serial_tt();
        let mut engine = idle_engine();
        engine
            .set_position(PositionSfen::StartPos, "7g7f 3c3d")
            .expect("both moves are legal");
        let accepted_ply = engine.pos.ply();

        for refused in [
            "7g7f 1a1b",                                   // an illegal move mid-list
            king_shuffle(MAX_POSITION_MOVES + 1).as_str(), // past the bound
        ] {
            let _ = engine.set_position(PositionSfen::StartPos, refused);
            assert_eq!(engine.pos.ply(), accepted_ply);
            assert_eq!(engine.last_position.moves.len(), 2);
        }

        // A malformed SFEN is refused before any move is looked at.
        let _ = engine.set_position(PositionSfen::Sfen(["not-a-board", "b", "-", "1"]), "");
        assert_eq!(engine.pos.ply(), accepted_ply);
        assert_eq!(engine.last_position.moves.len(), 2);
    }

    /// A game's position commands are written into the buffers the engine was
    /// built with: whatever the game's length, none of them is regrown, which is
    /// what keeps the command path away from the allocator.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn a_games_position_commands_reuse_the_buffers() {
        let _tt = serial_tt();
        let mut engine = idle_engine();
        let sfen = [
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL",
            "b",
            "-",
            "1",
        ];
        for plies in 0..64 {
            engine
                .set_position(PositionSfen::StartPos, &king_shuffle(plies))
                .expect("a king shuffle is legal");
            engine
                .set_position(PositionSfen::Sfen(sfen), &king_shuffle(plies))
                .expect("a king shuffle is legal");
        }
        assert_eq!(engine.last_position.moves.len(), 63);
        for retained in [&engine.last_position, &engine.pending_position] {
            assert_eq!(retained.moves.capacity(), MAX_POSITION_MOVES);
            assert_eq!(retained.sfen.capacity(), SFEN_CAPACITY);
        }
        assert_eq!(engine.legal_buf.capacity(), MAX_LEGAL_MOVES);
    }

    // -- the compiled layout, and the machine it is held against ---------

    /// A layout with `nodes` as its nodes, each carrying its own position as its
    /// system node number.
    fn layout(nodes: &[&[usize]]) -> NumaLayout {
        NumaLayout::from_const(nodes, &(0..nodes.len()).collect::<Vec<_>>())
    }

    fn affinity(cpus: &[usize]) -> BTreeSet<usize> {
        cpus.iter().copied().collect()
    }

    /// A plan pinning one worker to each of `cpus`, on the node `built` puts it
    /// on.
    fn plan_over(cpus: &[usize], built: &NumaLayout) -> WorkerPlan {
        let nodes: Vec<NumaIndex> = cpus
            .iter()
            .map(|&c| built.system_node_of_cpu(c).expect("a CPU of the layout"))
            .collect();
        WorkerPlan::of(cpus, &nodes, cpus.len())
    }

    #[test]
    fn a_machine_matching_the_compiled_layout_is_no_difference() {
        let built = layout(&[&[0, 1], &[2, 3]]);
        let plan = plan_over(&[0, 2], &built);
        assert_eq!(
            machine_refusal(&built, &affinity(&[0, 1, 2, 3]), &plan, &built),
            None
        );
    }

    #[test]
    fn a_machine_with_another_node_count_is_reported_with_both_counts() {
        let built = layout(&[&[0, 1], &[2, 3]]);
        let plan = plan_over(&[0, 2], &built);
        let live = layout(&[&[0, 1, 2, 3]]);
        let msg = machine_refusal(&live, &affinity(&[0, 1, 2, 3]), &plan, &built)
            .expect("one node is not two");
        assert!(msg.contains("built for 2 NUMA node(s)"), "message: {msg}");
        assert!(msg.contains("this host has 1"), "message: {msg}");
    }

    #[test]
    fn a_node_holding_other_cpus_is_reported_with_both_cpu_lists() {
        let built = layout(&[&[0, 1], &[2, 3]]);
        let plan = plan_over(&[0, 2], &built);
        let live = layout(&[&[0, 1], &[2, 3, 4]]);
        let msg = machine_refusal(&live, &affinity(&[0, 1, 2, 3, 4]), &plan, &built)
            .expect("node 1 grew a CPU");
        assert!(msg.contains("node 1"), "message: {msg}");
        assert!(msg.contains("2-4"), "message: {msg}");
        assert!(msg.contains("2-3"), "message: {msg}");
    }

    #[test]
    fn a_process_denied_a_cpu_a_worker_is_pinned_to_is_refused_with_those_cpus() {
        // The machine is the one the binary was built for; the *process* is not
        // allowed on every CPU a worker pins itself to — a `taskset` around the
        // engine, whose workers would then pin themselves to CPUs they may not
        // run on. Only the CPUs it is missing are named.
        let built = layout(&[&[0, 1], &[2, 3]]);
        let plan = plan_over(&[0, 2, 3], &built);
        let msg = machine_refusal(&built, &affinity(&[0, 1]), &plan, &built)
            .expect("two of the plan's CPUs are hidden");
        assert!(msg.contains("may not run on CPUs 2-3"), "message: {msg}");
        assert!(
            msg.contains("which its workers are pinned to"),
            "message: {msg}"
        );

        let msg = machine_refusal(&built, &affinity(&[0, 1, 2]), &plan, &built)
            .expect("one of the plan's CPUs is hidden");
        assert!(msg.contains("may not run on CPUs 3"), "message: {msg}");
    }

    #[test]
    fn an_affinity_covering_the_workers_cpus_is_no_difference() {
        // A binary built for part of the machine, started with no confinement:
        // its workers sit on node 0 and the process may run on everything. The
        // CPUs beyond them take nothing away, since each worker narrows itself
        // to its own CPU. An affinity holding exactly those CPUs is equally
        // fine.
        let built = layout(&[&[0, 1], &[2, 3]]);
        let plan = plan_over(&[0, 1], &built);
        assert_eq!(
            machine_refusal(&built, &affinity(&[0, 1, 2, 3]), &plan, &built),
            None
        );
        assert_eq!(
            machine_refusal(&built, &affinity(&[0, 1]), &plan, &built),
            None
        );
    }

    #[test]
    fn a_cpu_that_moved_node_is_refused_with_both_nodes() {
        // Node count and CPU lists agree, but the two nodes have swapped their
        // CPUs — every worker's memory would be placed on a node its CPU is not
        // on, so the machine is not the one this binary was built for.
        let built = layout(&[&[0, 1], &[2, 3]]);
        let plan = plan_over(&[0, 2], &built);
        let live = NumaLayout::from_const(&[&[0, 1], &[2, 3]], &[1, 0]);
        let msg = machine_refusal(&live, &affinity(&[0, 1, 2, 3]), &plan, &built)
            .expect("CPU 0 is on the other node now");
        assert!(msg.contains("worker 0 runs on CPU 0"), "message: {msg}");
        assert!(msg.contains("node 1 on this host"), "message: {msg}");
        assert!(msg.contains("node 0 in the layout"), "message: {msg}");
    }

    #[cfg(feature = "verbose3")]
    #[test]
    fn info_strings_exact_formats() {
        // Singular and plural, and the CPU list in the shortened form.
        let one = WorkerPlan::of(&[5], &[0], 1);
        assert_eq!(
            thread_allocation_information_as_string(1, &one),
            "Using 1 thread on CPUs 5"
        );
        let four = WorkerPlan::of(&[0, 1, 2, 8], &[0, 0, 0, 1], 4);
        assert_eq!(
            thread_allocation_information_as_string(4, &four),
            "Using 4 threads on CPUs 0-2,8"
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
    fn a_go_is_handed_the_pools_own_helper_list() {
        // Nothing about dispatching to the helpers copies the list: every `go`
        // gets the pool's, which only a rebuild replaces.
        let mut pool = ThreadPool::new(3);
        let first = pool.helper_slots();
        assert!(Arc::ptr_eq(&first, &pool.helper_slots()));
        pool.set(2);
        assert!(
            !Arc::ptr_eq(&first, &pool.helper_slots()),
            "a rebuilt pool hands out its new slots"
        );
        assert_eq!(pool.helper_slots().len(), 1);
    }

    // --- the handles and buffers a `go` reuses ----------------------------

    #[test]
    fn arming_a_search_clears_what_the_last_one_left() {
        let mut handles = SearchHandles::new(3);
        let stop = Arc::clone(&handles.stop);
        let bmc = Arc::clone(&handles.bmc_slots);
        handles.stop.store(true, Ordering::Relaxed);
        handles.suppress_reply.store(true, Ordering::Relaxed);
        handles.bmc_slots[2].store(9, Ordering::Relaxed);
        #[cfg(feature = "verbose2")]
        handles.node_slots[1].store(9, Ordering::Relaxed);

        assert!(
            handles.arm(false).is_none(),
            "no ponder signal for a plain go"
        );

        assert!(!handles.stop.load(Ordering::Relaxed));
        assert!(!handles.suppress_reply.load(Ordering::Relaxed));
        assert!(
            handles
                .bmc_slots
                .iter()
                .all(|s| s.load(Ordering::Relaxed) == 0),
            "every best-move-change counter starts at zero"
        );
        #[cfg(feature = "verbose2")]
        assert!(
            handles
                .node_slots
                .iter()
                .all(|s| s.load(Ordering::Relaxed) == 0),
            "every node counter starts at zero"
        );
        assert!(
            Arc::ptr_eq(&stop, &handles.stop) && Arc::ptr_eq(&bmc, &handles.bmc_slots),
            "the flags and counters are cleared in place, not rebuilt"
        );
    }

    #[test]
    fn each_ponder_go_starts_from_an_unhit_signal() {
        let mut handles = SearchHandles::new(1);
        let first = handles.arm(true).expect("a `go ponder` carries the signal");
        assert!(first.is_active());
        first.ponderhit();
        assert!(!first.is_active());
        drop(first);

        let second = handles.arm(true).expect("a `go ponder` carries the signal");
        assert!(
            second.is_active(),
            "the next ponder search starts pondering rather than inheriting a hit"
        );
    }

    #[test]
    fn the_collection_buffers_hold_a_pools_worth_of_room() {
        let mut buffers = VoteBuffers::for_pool(4);
        assert!(buffers.results.capacity() >= 4 && buffers.votes.capacity() >= 4);
        buffers.fit_to_pool(9);
        assert!(buffers.results.capacity() >= 9 && buffers.votes.capacity() >= 9);
        // A smaller pool keeps the room already won: nothing here ever shrinks.
        buffers.fit_to_pool(2);
        assert!(buffers.results.capacity() >= 9 && buffers.votes.capacity() >= 9);
    }

    #[test]
    fn thread_pool_zero_is_clamped_to_one() {
        // The engine never passes 0 (the option min is 1), but the pool clamps
        // defensively so `size − 1` never underflows.
        let pool = ThreadPool::new(0);
        assert_eq!(pool.size(), 1);
    }

    // --- shared-history node mapping --------------------------------------

    #[test]
    fn shared_node_counts_are_the_workers_per_node() {
        let c = shared_node_counts(&[0, 1, 0, 1, 0]);
        assert_eq!(c[&0], 3);
        assert_eq!(c[&1], 2);

        // Every worker on one node: one set, sized for all of them.
        let c = shared_node_counts(&[2, 2, 2]);
        assert_eq!(c.len(), 1);
        assert_eq!(c[&2], 3);
    }

    /// A plan over the CPUs this process is allowed on, so the pin inside
    /// `build_worker_shared` cannot hit a forbidden CPU.
    fn allowed_plan(requested: usize) -> WorkerPlan {
        WorkerPlan::of_allowed_cpus(requested)
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn build_worker_shared_shares_one_set_per_node() {
        let ws = build_worker_shared(&allowed_plan(4));
        assert_eq!(ws.len(), 4, "one handle per worker");
        // The plan puts every worker on node 0, so all four share one set.
        for i in 1..4 {
            assert!(
                Arc::ptr_eq(&ws[0], &ws[i]),
                "workers on one node point at one shared set"
            );
        }
        // Sized to `next_power_of_two(pool size)`.
        assert_eq!(ws[0].thread_count(), 4);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn build_worker_shared_rounds_thread_count_up() {
        // 3 workers → next_power_of_two(3) == 4.
        let ws = build_worker_shared(&allowed_plan(3));
        assert_eq!(ws.len(), 3);
        assert_eq!(ws[0].thread_count(), 4);
        // Single worker → thread_count 1.
        let ws1 = build_worker_shared(&allowed_plan(1));
        assert_eq!(ws1.len(), 1);
        assert_eq!(ws1[0].thread_count(), 1);
    }

    // --- NUMA memory placement --------------------------------------------

    #[test]
    fn a_plan_wraps_extra_workers_onto_the_assigned_cpus() {
        // The assignment holds one CPU per configured worker; a measurement
        // command asking for more wraps around rather than spilling onto CPUs
        // this binary was not given.
        let plan = WorkerPlan::of(&[4, 9], &[0, 1], 5);
        assert_eq!(plan.cpus, vec![4, 9, 4, 9, 4]);
        assert_eq!(plan.system_nodes, vec![0, 1, 0, 1, 0]);
        assert_eq!(plan.distinct_cpus(), BTreeSet::from([4, 9]));
        assert_eq!(plan.distinct_system_nodes(), vec![0, 1]);
    }

    #[test]
    fn table_placement_covers_the_nodes_the_workers_sit_on_and_no_others() {
        // Every worker on one node: the whole table belongs on that node, and it
        // is the system node the layout names, not the position in the table.
        assert_eq!(
            table_placement(&WorkerPlan::of(&[0, 1], &[2, 2], 2)),
            TablePlacement::OnNode(2)
        );
        // Two nodes under the workers: spread over exactly those two.
        assert_eq!(
            table_placement(&WorkerPlan::of(&[0, 4, 1, 5], &[0, 1, 0, 1], 4)),
            TablePlacement::AcrossNodes(vec![0, 1])
        );
        // A four-node machine whose workers use two of them: the other two hold
        // no worker, so no page of the table may land on them.
        assert_eq!(
            table_placement(&WorkerPlan::of(&[0, 4, 8], &[1, 3, 1], 3)),
            TablePlacement::AcrossNodes(vec![1, 3])
        );
    }

    /// Every large-page block behind the coordinator's history tables must be
    /// governed by an `MPOL_BIND` policy naming worker 0's system node.
    ///
    /// Still meaningful on a single-node host: the *placement* answer there is
    /// node 0 either way, but the *policy* over those pages is `MPOL_DEFAULT`
    /// until something binds it.
    #[cfg(target_os = "linux")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn a_pool_rebuild_places_the_coordinator_histories_on_worker_zero_node() {
        let mut engine = idle_engine();
        engine.rebuild_pool();

        let node = engine.worker_plan.system_nodes[0];
        let regions = engine
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

        let mut engine = idle_engine();
        // Seed distinctive "previous real search" state.
        engine.best_previous_score = 123;
        engine.best_previous_average_score = 456;
        engine.previous_time_reduction = 0.42;
        engine.last_game_ply = 7;

        // A synthetic short-circuited search that "ran" at ply 20 and hands back
        // the book / declaration / resign carry.
        let handle = std::thread::spawn(|| SearchState {
            histories: WorkerHistories::new(),
            time_state: skip_search_carry(),
            vote_buffers: VoteBuffers::for_pool(1),
        });
        engine.search = Some(ActiveSearch {
            handle,
            stop: Arc::new(AtomicBool::new(false)),
            ponder: None,
            suppress: Arc::new(AtomicBool::new(false)),
            #[cfg(feature = "verbose3")]
            reply_sent: Arc::new(AtomicBool::new(true)),
            game_ply: 20,
        });
        engine.finish_search_join();

        assert_eq!(engine.best_previous_score, -VALUE_INFINITE);
        assert_eq!(engine.best_previous_average_score, -VALUE_INFINITE);
        assert_eq!(
            engine.last_game_ply, 20,
            "ply advances to the short-circuit's"
        );
        assert_eq!(
            engine.previous_time_reduction, 0.42,
            "previousTimeReduction is left untouched on a short-circuit"
        );
    }

    // A real search's carry (`Some(tr)`) *does* overwrite `previous_time_reduction`
    // — the complement of the short-circuit case above.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn real_search_carry_overwrites_time_reduction() {
        let mut engine = idle_engine();
        engine.previous_time_reduction = 0.42;

        let handle = std::thread::spawn(|| SearchState {
            histories: WorkerHistories::new(),
            time_state: Some((10, 20, Some(1.25))),
            vote_buffers: VoteBuffers::for_pool(1),
        });
        engine.search = Some(ActiveSearch {
            handle,
            stop: Arc::new(AtomicBool::new(false)),
            ponder: None,
            suppress: Arc::new(AtomicBool::new(false)),
            #[cfg(feature = "verbose3")]
            reply_sent: Arc::new(AtomicBool::new(true)),
            game_ply: 3,
        });
        engine.finish_search_join();

        assert_eq!(engine.best_previous_score, 10);
        assert_eq!(engine.best_previous_average_score, 20);
        assert_eq!(engine.previous_time_reduction, 1.25);
        assert_eq!(engine.last_game_ply, 3);
    }

    // The evaluation regions this binary declares, and which of them a worker
    // reads, are both compile-time facts; these hold them against each other.

    #[test]
    fn every_worker_sits_on_a_node_a_region_was_declared_for() {
        for &node in config::WORKER_SYSTEM_NODES {
            assert!(
                config::EVAL_REGION_NODES.contains(&node),
                "worker node {node} has no region: {:?}",
                config::EVAL_REGION_NODES
            );
        }
    }

    #[test]
    fn the_region_nodes_are_the_workers_nodes_ascending_and_without_repeats() {
        let mut expected: Vec<usize> = config::WORKER_SYSTEM_NODES.to_vec();
        expected.sort_unstable();
        expected.dedup();
        assert_eq!(config::EVAL_REGION_NODES, expected.as_slice());
    }

    #[test]
    fn the_binary_declares_a_region_for_every_node_the_workers_use() {
        assert!(
            config::EVAL_REGION_NODES.len() <= network_file::REGION_COUNT,
            "{} nodes want a region, {} are declared",
            config::EVAL_REGION_NODES.len(),
            network_file::REGION_COUNT
        );
    }

    #[test]
    fn a_pool_over_more_workers_than_cpus_stays_on_those_nodes() {
        // The measurement command is the one caller that asks for more workers
        // than the assignment has CPUs; the extra ones wrap around, so no
        // worker lands on a node without a region.
        let plan = WorkerPlan::of(
            config::WORKER_CPUS,
            config::WORKER_SYSTEM_NODES,
            config::WORKER_CPUS.len() * 3 + 1,
        );
        for node in plan.system_nodes {
            assert!(config::EVAL_REGION_NODES.contains(&node));
        }
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
