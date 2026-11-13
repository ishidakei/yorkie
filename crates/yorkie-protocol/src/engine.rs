//! The engine proper: the game state, the worker pool and the search, with no
//! protocol text anywhere in it.
//!
//! What a host says and what the engine does are two different things, and this
//! is the second of them. An [`Engine`] holds the position, the worker pool and
//! its handles, the placed transposition table and evaluation network, the
//! opening book, the per-game seeds and the time-management carry-forward; it
//! offers [`Engine::set_position`], [`Engine::go`], [`Engine::stop`],
//! [`Engine::new_game`] and [`Engine::ready`] as plain functions over typed
//! arguments and typed results. Nothing here knows what a reply line looks
//! like.
//!
//! Where the engine has something to say — a reply, an initialisation notice, a
//! diagnostic — it says it through [`EngineSink`], a generic parameter rather
//! than a runtime-chosen sink: a build compiles exactly one protocol layer, so
//! the type is known when the binary is, every call is a direct one, and nothing
//! on the game path is dispatched through a pointer.

use std::collections::BTreeSet;
use std::ffi::OsStr;
use std::io;
#[cfg(feature = "verbose2")]
use std::num::NonZeroU64;
use std::os::unix::ffi::OsStrExt as _;
use std::path::{Path, PathBuf};
#[cfg(low_priority_worker)]
use std::sync::TryLockError;
#[cfg(low_priority_worker)]
use std::sync::atomic::AtomicI32;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
// The engine waits on a clock in two places: the hold of a `go infinite` reply,
// a clause that arrives with `verbose2`, and the bounded wait for a
// low-priority worker.
#[cfg(any(feature = "verbose2", low_priority_worker))]
use std::time::Duration;
use std::time::Instant;

use yorkie_eval::{NetworkParams, NnueError, network_file};
use yorkie_numa::{CpuIndex, NumaIndex, NumaLayout, SysfsError, mempolicy};
use yorkie_search::{
    BookConfig, BookHit, EnteringKingConfig, Prng, QSearch, RootMove, RootWidth, SearchControl,
    SharedHistories, TallySlot, TimeControl, TimeInput, TimeManagement, VALUE_INFINITE,
    WorkerHistories, WorkerResult, WorkerVote, YorkieProbe, declaration_win, generate_root_moves,
    probe_book, select_best_worker,
};
// The PV-line surface: only a `verbose2` build renders one, so only it needs
// the line's data type, the sink trait the search emits through and the output
// config.
#[cfg(feature = "verbose2")]
use yorkie_search::{PvInfo, PvOutputConfig, PvSink};
// The per-game evaluation-noise seed: drawn here, read only by the search.
#[cfg(feature = "random")]
use yorkie_search::new_game_seed;
use yorkie_state::{
    ExtMove, Move, PackedSfen, Position, SfenError, TextWriter, parse_sfen_into, parse_usi_move,
    sfen_pack,
};
#[cfg(feature = "verbose3")]
use yorkie_storage::ValueMarks;
use yorkie_storage::{Book, TranspositionTable, Value, YorkieBook, YorkieBookError};
// The per-reply allocation tally: raised by the counting global allocator this
// feature installs, and cleared where an interval starts.
#[cfg(feature = "verbose1")]
use yorkie_storage::clear_alloc_count;

use crate::config::{self, with_eval_network};
use crate::settings::PONDER_ROOT_WIDTH;
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

/// Positions one game path holds: the starting position and one after each of
/// at most [`MAX_POSITION_MOVES`] moves. A command longer than that is refused,
/// so a path never needs more room than this.
const GAME_PATH_CAPACITY: usize = MAX_POSITION_MOVES + 1;

/// Room the retained SFEN buffer starts with: enough for the longest SFEN a
/// board can spell — every square occupied by a promoted piece, both hands
/// full, a five-digit ply — so a game's `position` commands are written into
/// the buffer already there.
const SFEN_CAPACITY: usize = yorkie_state::SFEN_CAPACITY;

/// Room an initialisation-phase notice is composed in.
///
/// The longest of them is a filesystem path with a sentence around it, so the
/// buffer is sized for a path the operating system will still open rather than
/// for the sentence.
pub(crate) const NOTICE_BYTES: usize = 8 * 1024;

/// `"applied"` or `"refused"` — how a best-effort placement call went, which is
/// what every placement notice reports.
fn outcome(accepted: bool) -> &'static [u8] {
    if accepted { b"applied" } else { b"refused" }
}

/// Room one book-probe notice is composed in: its widest form is two option
/// names, two values and two move counts.
#[cfg(feature = "verbose1")]
const BOOK_DIAGNOSTIC_BYTES: usize = 256;

/// Legal moves the widest shogi position offers, which is what the replay's
/// legality buffer is built to hold (the reference's `MAX_MOVES`).
const MAX_LEGAL_MOVES: usize = 600;

/// The game ply past which the search adjudicates a draw, with the reference's
/// `0 → unlimited` remap — the same horizon the search itself compiles in.
///
/// A position past it is a game end, so there is nothing to think ahead about
/// from there.
const DRAW_HORIZON: i32 = if config::MAX_MOVES_TO_DRAW == 0 {
    100_000
} else {
    config::MAX_MOVES_TO_DRAW as i32
};

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
    sfen: Vec<u8>,
    /// The moves in the order they were applied.
    moves: Vec<Move>,
    /// Every position the replay passed through, packed: the starting position,
    /// then the one after each move. Recorded only while a yorkie book is read,
    /// and empty otherwise; its room is reserved when that book is loaded.
    path: Vec<PackedSfen>,
}

impl RetainedPosition {
    /// `position startpos`, with room for a whole game ahead of it.
    fn new() -> Self {
        Self {
            startpos: true,
            sfen: Vec::with_capacity(SFEN_CAPACITY),
            moves: Vec::with_capacity(MAX_POSITION_MOVES),
            path: Vec::new(),
        }
    }

    /// Back to `position startpos`, every buffer keeping its room.
    fn reset(&mut self) {
        self.startpos = true;
        self.sfen.clear();
        self.moves.clear();
        self.path.clear();
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
                    self.sfen.push(b' ');
                }
                self.sfen.extend_from_slice(field);
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
    IllegalMove(&'a [u8]),
    TooManyMoves,
}

/// Where a position starts: the implicit initial position, or an explicit SFEN
/// whose four fields — board, side to move, hands, ply — are borrowed from the
/// command that named them rather than copied out of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PositionSfen<'a> {
    StartPos,
    Sfen([&'a [u8]; 4]),
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
///
/// A loaded book is read-only for the rest of the process and every `go`'s
/// coordinator reads it, so it is held as a `&'static` borrow made when it is
/// loaded, and a `go` hands its coordinator that reference. The borrow is never
/// given back: a book is loaded once per process unless the files the book
/// setting names change between two readiness handshakes, and a reload keeps
/// the books it replaces in memory.
struct LoadedBook {
    books: Vec<Book>,
    ignore_book_ply: bool,
}

/// The yorkie book one `go` probes: none when no yorkie book is loaded or
/// `USI_OwnBook` is off, and `fill` writing the game path into [`GAME_PATH`]
/// only for a book that will read it.
fn yorkie_for_go(
    own_book: bool,
    loaded: Option<&YorkieBook>,
    fill: impl FnOnce(&mut Vec<PackedSfen>),
) -> Option<&YorkieBook> {
    let book = loaded.filter(|_| own_book)?;
    fill(&mut GAME_PATH.lock().unwrap_or_else(PoisonError::into_inner));
    Some(book)
}

/// A search worker running on its own thread. The main thread
/// keeps reading commands while this runs; a stop request sets [`STOP`], which
/// the search polls at the reference `check_time` granularity. The worker
/// emits its own progress and its reply, then returns the session-owned
/// [`SearchState`] so the engine can reclaim it for the next `go`.
struct ActiveSearch {
    handle: JoinHandle<SearchState>,
    /// The root game ply this search ran at (`rootPos.game_ply()`), carried so
    /// a completed real search updates the engine's `last_game_ply`.
    game_ply: i32,
}

/// The session-owned search state a `go` lends to its worker and reclaims when
/// the worker finishes: the game-scoped worker history tables, which persist
/// across `go`s within one game and are reset by [`Engine::new_game`].
///
/// The transposition table is not part of this handover — it is a `static`,
/// which the worker reaches directly and never hands back.
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
    /// What the search past the reply left for the next `go`'s clock, if one
    /// ran.
    ponder: Option<PonderRecord>,
}

/// The workers a pool can hold: the `threads` setting, which is the size of
/// every pool a game runs.
///
/// The measurement command carries a worker count of its own, so a build that
/// has it (`verbose3`) holds room for `BENCH_POOL_CAPACITY` workers as well,
/// and refuses a count beyond that. The unit tests build pools of their own
/// sizes and get the same room.
#[cfg(not(any(test, feature = "verbose3")))]
const POOL_CAPACITY: usize = config::THREADS as usize;
#[cfg(any(test, feature = "verbose3"))]
const POOL_CAPACITY: usize = if (config::THREADS as usize) > BENCH_POOL_CAPACITY {
    config::THREADS as usize
} else {
    BENCH_POOL_CAPACITY
};

/// The most workers a measurement run may ask for in a build whose own
/// `threads` setting is smaller.
#[cfg(any(test, feature = "verbose3"))]
const BENCH_POOL_CAPACITY: usize = 256;

const _: () = assert!(POOL_CAPACITY >= 1);

// The flags the main loop signals a search on and the per-worker counters the
// workers tally into. They carry no value from one search to the next, so they
// belong to the session rather than to a search, and like the transposition
// table there is one session per process: each is a `static`, which a worker
// reaches at a constant address rather than through a handle it was given.
// Every `go` clears them ([`arm_search`]) before its search starts, and nothing
// races that reset: every `go` joins the previous coordinator first, and a
// joined coordinator has already collected every full-priority helper it
// dispatched. The low-priority helper may still be finishing the last search
// then; it polls [`LOW_STOP`], which that reset leaves raised.

/// The one stop flag every worker of a search polls, but the low-priority one.
static STOP: AtomicBool = AtomicBool::new(false);

/// The stop flag of the low-priority helper (`low_priority_worker`).
///
/// It is the coordinator's alone to raise and lower. The coordinator raises it
/// as soon as its own search returns, which is when it raises [`STOP`] for the
/// rest, and lowers it only when it hands that helper a new job — which it
/// does only once the helper has handed back the last one. A helper the
/// scheduler kept off its CPU past the end of a search therefore still sees the
/// flag raised when it runs again, however many searches have started since,
/// and does not carry on with a search nobody will collect.
#[cfg(low_priority_worker)]
static LOW_STOP: AtomicBool = AtomicBool::new(false);

/// How long a coordinator waits, from the end of its own search, for the
/// low-priority helper's result. A result that has not arrived by then is left
/// out of this search's vote, and the reply goes out without it.
///
/// A helper that is on its CPU sees its stop flag within microseconds, so this
/// is spent only when something else holds that CPU; it is the most that
/// worker can add to the time a `bestmove` takes.
#[cfg(low_priority_worker)]
const LATE_HELPER_WAIT: Duration = Duration::from_millis(5);

/// How often a coordinator looks at the low-priority helper's slot while it
/// waits for it. It polls rather than blocks, so that a helper descheduled
/// while it holds its slot's lock cannot hold the coordinator with it.
#[cfg(low_priority_worker)]
const LATE_HELPER_POLL: Duration = Duration::from_micros(100);

/// What the low-priority helper made of lowering its own priority: still
/// [`LOWERING_PENDING`], [`LOWERED`], or the `errno` of the refusal (`-1` when
/// the platform has no per-thread priority).
#[cfg(low_priority_worker)]
static LOW_PRIORITY_OUTCOME: AtomicI32 = AtomicI32::new(LOWERING_PENDING);
#[cfg(low_priority_worker)]
const LOWERING_PENDING: i32 = i32::MIN;
#[cfg(low_priority_worker)]
const LOWERED: i32 = 0;

/// Lower the calling helper thread to the lowest priority and publish the
/// outcome in [`LOW_PRIORITY_OUTCOME`]. Run first thing in the helper, so the
/// outcome is published whatever its CPU pin does after.
#[cfg(low_priority_worker)]
fn lower_this_helper() {
    let outcome = match yorkie_numa::lower_current_thread_priority() {
        Ok(()) => LOWERED,
        Err(errno) if errno > 0 => errno,
        Err(_) => -1,
    };
    LOW_PRIORITY_OUTCOME.store(outcome, Ordering::Release);
}

/// Why the low-priority helper of the live pool runs at normal priority after
/// all — the `errno` of the refusal, `-1` for a platform without per-thread
/// priorities — or `None` when it was lowered or there is none.
#[cfg(low_priority_worker)]
fn low_priority_refusal() -> Option<i32> {
    match LOW_PRIORITY_OUTCOME.load(Ordering::Acquire) {
        LOWERING_PENDING | LOWERED => None,
        refusal => Some(refusal),
    }
}

/// Append to the `workers on CPUs` / `workers unpinned` line which worker gives
/// way to the rest of the machine: by the CPU it shares with the non-search
/// threads, or by its number in a build that pins nothing. Nothing without one.
#[cfg(low_priority_worker)]
fn write_low_priority_suffix(plan: &WorkerPlan, out: &mut TextWriter<'_>) {
    if let Some(cpu) = plan.low_priority_cpu() {
        out.bytes(b", low priority on CPU ").u64(cpu as u64);
    } else if let Some(worker) = plan.low_priority_worker() {
        out.bytes(b", low priority on worker ").u64(worker as u64);
    }
}

/// Each helper's kernel thread id, written as it starts, so a test can ask the
/// kernel what priority a given worker runs at.
#[cfg(all(test, low_priority_worker))]
static HELPER_TIDS: [AtomicI32; POOL_CAPACITY - 1] =
    [const { AtomicI32::new(0) }; POOL_CAPACITY - 1];

/// The stop flag of the thinking-ahead search a coordinator runs once it has
/// replied, which is a different search and so takes a flag of its own:
/// [`STOP`] is already set by the time the reply is out, both by the
/// coordinator ending its helpers and by a search that stopped itself.
static PONDER_STOP: AtomicBool = AtomicBool::new(false);

/// Raised by the coordinator while it is thinking ahead, so the command that
/// ends that search knows to join the coordinator rather than leave it running.
static PONDERING: AtomicBool = AtomicBool::new(false);

/// Per-worker node counters (index 0 = main, `1..` = helpers), read by the
/// aggregate node ceiling and the final aggregated `info ... nodes` — both
/// `verbose2`, like the counters themselves.
#[cfg(feature = "verbose2")]
static NODE_SLOTS: [TallySlot; POOL_CAPACITY] = [const { TallySlot::new() }; POOL_CAPACITY];

/// Per-worker best-move-change counters: each worker bumps its own slot at the
/// root and the main worker folds them all each iteration.
static BMC_SLOTS: [TallySlot; POOL_CAPACITY] = [const { TallySlot::new() }; POOL_CAPACITY];

/// Every position of the game so far, which a `go`'s yorkie-book probe reads
/// ([`Engine::fill_game_path`]).
///
/// It is the session's rather than handed over so that a `go` neither
/// allocates nor waits for it to come back: it is written only between
/// searches, after the previous coordinator has been joined, and read only by
/// the coordinator's book probe. Its room is reserved when a yorkie book is
/// loaded.
static GAME_PATH: Mutex<Vec<PackedSfen>> = Mutex::new(Vec::new());

/// The part of a game path — a `position` command's starting position, then
/// the position after each of its moves — that counts as having occurred once
/// its first `skip_moves` moves are left out: the position after move
/// `skip_moves` and every one after it. A path of fewer moves than that keeps
/// its last position alone, and an empty path stays empty.
///
/// The left-out moves are a preamble a tournament server plays before the game
/// proper; a book built for the game proper never meant the positions it
/// passes through to count.
fn seen_positions(path: &[PackedSfen], skip_moves: usize) -> &[PackedSfen] {
    &path[skip_moves.min(path.len().saturating_sub(1))..]
}

/// The node counters of a pool of `workers`, one slot per worker.
#[cfg(feature = "verbose2")]
fn node_tally(workers: usize) -> &'static [TallySlot] {
    &NODE_SLOTS[..workers]
}

/// The best-move-change counters of a pool of `workers`, one slot per worker.
fn bmc_tally(workers: usize) -> &'static [TallySlot] {
    &BMC_SLOTS[..workers]
}

/// Clear every flag, and zero the counters of a pool of `workers`, for a search
/// about to start.
fn arm_search(workers: usize) {
    STOP.store(false, Ordering::Relaxed);
    PONDER_STOP.store(false, Ordering::Relaxed);
    PONDERING.store(false, Ordering::Relaxed);
    #[cfg(feature = "verbose2")]
    for slot in node_tally(workers) {
        slot.store(0, Ordering::Relaxed);
    }
    for slot in bmc_tally(workers) {
        slot.store(0, Ordering::Relaxed);
    }
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
    /// The chosen move.
    BestMove { mv: Move },
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
/// that helper has stopped: the network is ready, the load failed, the machine
/// is not the one this binary plans its threads for, or the configured yorkie
/// book could not be loaded.
///
/// Each failure carries its reason for the caller to render, and no build
/// answers a failed one with a readiness reply.
pub enum ReadyOutcome {
    Ready,
    LoadFailed(NnueError),
    LayoutMismatch(LayoutRefusal),
    /// The yorkie book file named in the config, and why it was refused.
    YorkieBookFailed(PathBuf, YorkieBookError),
}

/// Why the machine this process runs on is not the one the binary was built
/// for — one of four ways to differ, in the order a reader wants them, plus the
/// sysfs tree that could not be read at all.
///
/// Each variant carries what differs rather than a sentence about it;
/// [`Self::write_message`] spells it where the notice is written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutRefusal {
    /// The tree does not describe a machine whose layout can be read. An
    /// unverifiable layout is not a matching one.
    Unreadable(SysfsError),
    /// A different number of NUMA nodes.
    NodeCount { built: usize, live: usize },
    /// A node holding different CPUs.
    NodeCpus {
        node: usize,
        live: Vec<CpuIndex>,
        built: Vec<CpuIndex>,
    },
    /// CPUs a worker pins itself to that this process may not run on — what a
    /// `taskset` or a `cpuset` around the engine produces.
    DeniedCpus(Vec<CpuIndex>),
    /// A CPU that sits on another node than the built layout puts it on, which
    /// would leave a worker's memory placed away from the CPU reading it.
    CpuNode {
        worker: usize,
        cpu: CpuIndex,
        live_node: Option<NumaIndex>,
        built_node: NumaIndex,
    },
}

impl LayoutRefusal {
    /// Append this refusal's message to `out`.
    pub fn write_message(&self, out: &mut TextWriter<'_>) {
        match self {
            Self::Unreadable(e) => e.write_message(|fragment| {
                out.bytes(fragment);
            }),
            Self::NodeCount { built, live } => {
                out.bytes(b"this binary is built for ")
                    .u64(*built as u64)
                    .bytes(b" NUMA node(s), this host has ")
                    .u64(*live as u64);
            }
            Self::NodeCpus { node, live, built } => {
                out.bytes(b"node ").u64(*node as u64).bytes(b" holds CPUs ");
                write_cpu_list(live, out);
                out.bytes(b" on this host, ");
                write_cpu_list(built, out);
                out.bytes(b" in the layout this binary is built for");
            }
            Self::DeniedCpus(cpus) => {
                out.bytes(b"this process may not run on CPUs ");
                write_cpu_list(cpus, out);
                out.bytes(b", which its workers are pinned to");
            }
            Self::CpuNode {
                worker,
                cpu,
                live_node,
                built_node,
            } => {
                out.bytes(b"worker ")
                    .u64(*worker as u64)
                    .bytes(b" runs on CPU ")
                    .u64(*cpu as u64);
                match live_node {
                    Some(live_node) => {
                        out.bytes(b", which is on node ")
                            .u64(*live_node as u64)
                            .bytes(b" on this host and on node ")
                            .u64(*built_node as u64)
                            .bytes(b" in the layout this binary is built for");
                    }
                    None => {
                        out.bytes(b", which no node of this host holds");
                    }
                }
            }
        }
    }
}

/// Append `cpus` to `out` in the shortened form the sysfs files use.
fn write_cpu_list<'c>(cpus: impl IntoIterator<Item = &'c CpuIndex>, out: &mut TextWriter<'_>) {
    yorkie_numa::write_cpu_list(cpus.into_iter().copied(), |fragment| {
        out.bytes(fragment);
    });
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
/// is a handle to the same output, not a second one; the sink the engine plays
/// with writes to the process's standard output and is zero-sized, so the clone
/// is nothing at all.
pub trait EngineSink: Clone + Send + 'static {
    /// The per-iteration PV renderer this protocol layer installs on the main
    /// worker — a concrete type, like the sink itself.
    #[cfg(feature = "verbose2")]
    type PvOutput: PvSink + 'static;

    /// One initialisation-phase notice — where the table and the network went,
    /// which CPUs the workers took, what the book load found. Present in every
    /// build: they are how a failed startup is diagnosed at all.
    fn notice(&self, msg: &[u8]) -> io::Result<()>;

    /// One diagnostic notice, the `verbose1` surface. Best-effort: it may be
    /// produced on the search thread, where a broken pipe must not panic.
    #[cfg(feature = "verbose1")]
    fn diagnostic(&self, msg: &[u8]);

    /// The engine's reply to a search request.
    ///
    /// `marks` are the value marks of the score the reply rests on — what the
    /// played move's value was derived through — for the statistics line to
    /// report, so a caller reading the stream learns them without a second probe
    /// of a root entry that may since have been replaced.
    fn reply(&self, reply: Reply, #[cfg(feature = "verbose3")] marks: ValueMarks);

    /// A book hit's surviving candidates, reported as the probe answers and
    /// before any hold.
    #[cfg(feature = "verbose2")]
    fn book_candidates(&self, hit: &BookHit, hashfull: u32, time_ms: u64);

    /// A book hit's terminal output: the summary of the chosen line, then the
    /// reply itself.
    fn book_reply(
        &self,
        hit: &BookHit,
        #[cfg(feature = "verbose2")] hashfull: u32,
        #[cfg(feature = "verbose2")] time_ms: u64,
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
    /// Whether [`Self::ready`] has placed the table's pages ([`Self::place_transposition_table`]).
    /// Once per session: the policy and the huge-page hint are properties of the
    /// address range, and repeating them would say the same thing again.
    tt_placed: bool,
    /// Game-scoped worker histories. `None` only while a worker
    /// holds them mid-search.
    histories: Option<WorkerHistories>,
    /// The loaded opening books, present only after a [`Self::ready`] opened at least
    /// one readable `.ybb`. `None` means bookless (default `BookFile=no_book`, or
    /// every listed book failed / was unsupported). A `'static` borrow — see
    /// [`LoadedBook`] — so a `go` hands its coordinator the reference itself.
    book: Option<&'static LoadedBook>,
    /// The `(resolved-name-list, on-the-fly, ignore-book-ply)` signature of the
    /// last book load — the Multiple Book priority list, not a single name.
    /// [`Self::ready`] reloads only when this changes — the reference's reload-skip.
    book_signature: Option<(Vec<PathBuf>, bool, bool)>,
    /// The yorkie books this session reads, in the config's order; empty when
    /// the yorkie book is disabled. Resolved once, at construction, the way
    /// [`book_path`] resolves the `.ybb` name.
    yorkie_choices: Vec<YorkieBookChoice>,
    /// The yorkie books [`Self::ready`] loaded, aligned with
    /// [`Self::yorkie_choices`], and the files they came from, so a repeat
    /// readiness handshake reuses them. `None` until a load succeeds. A
    /// `'static` borrow on the terms of [`LoadedBook`], so choosing a book is
    /// an index and a `go` hands its coordinator the reference itself.
    yorkie_books: Option<(Vec<PathBuf>, &'static [YorkieBook])>,
    /// The index into [`Self::yorkie_choices`] of the book the game in
    /// progress plays from: drawn at every [`Self::new_game`], and the first
    /// entry until the first one.
    yorkie_choice: usize,
    /// How many leading moves of the `position` command [`Self::fill_game_path`]
    /// leaves out of the game path: the config's `yorkie_book_path_skip_moves`,
    /// overridable through [`Self::set_yorkie_book_path_skip_moves`].
    yorkie_path_skip: usize,
    /// The generator [`Self::new_game`] draws the yorkie book from, seeded from
    /// the session seed so [`Engine::with_book_seed`] pins it too.
    yorkie_prng: Prng,
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
    /// What the last search past a reply left behind, kept from the join that
    /// ended it until the next `go` or new game.
    ponder_record: Option<PonderRecord>,
    /// The time in ms [`Self::ponder_record`] spent on the reply the current
    /// position plays, `0` when the position is not its root plus one reply.
    /// Decided by each `position`, and consumed by the `go` after it, which
    /// counts `ponder_credit` percent of it as already elapsed.
    pondered_ms: i64,
    /// The root game ply of the last completed real search
    /// (`main_manager()->lastGamePly`), reset to `0` by [`Self::new_game`].
    ///
    /// At the next search start an odd `last_game_ply - game_ply` means the side
    /// to move alternated, which flips the sign of the persisted previous scores
    /// before they seed the next search.
    last_game_ply: i32,
    /// The last position command in parsed form (`last_position_cmd_string`).
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
    worker_plan: WorkerPlan,
    /// Per-worker handles to the node-shared correction / pawn tables, rebuilt
    /// at every pool (re)build from [`Self::worker_plan`]. Length equals the pool
    /// size, so `[0]` is the coordinator's and `[1..]` the helpers', each of
    /// which is handed its own when the pool spawns it.
    worker_shared: Vec<Arc<SharedHistories>>,
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
    /// Whether this session has already said that its low-priority worker
    /// could not be lowered.
    #[cfg(low_priority_worker)]
    low_priority_refusal_reported: bool,
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
        let worker_plan = WorkerPlan::of(
            settings.worker_cpus(),
            settings.worker_system_nodes(),
            threads,
        );
        #[cfg(low_priority_worker)]
        let worker_plan = worker_plan.with_low_priority(settings.low_priority_worker());
        // Build the per-node shared correction / pawn tables: the helpers are
        // handed theirs as the pool spawns them, and the coordinator (worker 0)
        // gets its node's set here.
        let worker_shared = build_worker_shared(&worker_plan);
        let pool = ThreadPool::with_binding(threads, worker_plan, &worker_shared);
        // The pool decides the worker count — it refuses a size below one — so
        // what the buffers are sized by is the pool, not the setting it was
        // asked for.
        let workers = pool.size();
        let histories = Some(WorkerHistories::with_shared(Arc::clone(&worker_shared[0])));
        // The coordinator's bundle is built (and filled) right here, on the
        // command thread — so place it explicitly; see
        // `place_coordinator_histories`.
        place_coordinator_histories(histories.as_ref(), &worker_plan);
        let yorkie_choices = settings
            .yorkie_book_files()
            .iter()
            .map(|&(file, weight)| YorkieBookChoice {
                name: PathBuf::from(file),
                path: book_path(settings.book_dir(), file),
                weight,
            })
            .collect();
        let yorkie_path_skip = settings.yorkie_book_path_skip_moves();
        Self {
            sink,
            settings,
            yorkie_choices,
            yorkie_books: None,
            yorkie_choice: 0,
            yorkie_path_skip,
            yorkie_prng: Prng::new(book_seed ^ YORKIE_DRAW_SEED_MIX),
            pos: Position::startpos(),
            eval: None,
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
            ponder_record: None,
            pondered_ms: 0,
            last_game_ply: 0,
            last_position: RetainedPosition::new(),
            pending_position: RetainedPosition::new(),
            scratch_pos: Position::startpos(),
            legal_buf: Vec::with_capacity(MAX_LEGAL_MOVES),
            pseudo_buf: Vec::with_capacity(MAX_LEGAL_MOVES),
            pool,
            pool_threads: threads,
            numa_layout,
            worker_plan,
            worker_shared,
            vote_buffers: Some(VoteBuffers::for_pool(workers)),
            eval_root: network_file::executable_directory(),
            sysfs_root: PathBuf::from(SYSFS_ROOT),
            startup_affinity: yorkie_numa::startup_affinity().clone(),
            #[cfg(low_priority_worker)]
            low_priority_refusal_reported: false,
        }
    }

    /// Pin the calling thread — the session's command reader — to the CPU the
    /// host keeps its non-search work on (the building machine's lowest online
    /// CPU), so the engine's own non-search threads stay off every other
    /// worker's CPU. The threads it starts later inherit the pin: the
    /// keep-alive timer of a readiness handshake keeps it, and the workers each
    /// pin themselves to their own CPU. With a low-priority worker this is that
    /// worker's CPU, so those threads take their time from it and from no other.
    ///
    /// The pin is what keeps these threads where they belong even when the
    /// process starts with a wide affinity. A launcher has to hand the engine
    /// every CPU a worker pins itself to (the readiness check refuses less), so
    /// a bridge confined to the shared CPU cannot pass its own mask down.
    ///
    /// Nothing happens in a build that pins no worker (there is no CPU to share
    /// then), or when this process may not run on that CPU, which the
    /// readiness check reports instead. Called once the engine has recorded the
    /// process's startup affinity, which the pin narrows for this thread. A
    /// thread's CPU affinity leaves its memory policy alone, so where the
    /// table and the per-node copies are placed does not change.
    pub fn pin_to_shared_cpu(&self) {
        if let Some(cpu) = self.shared_cpu()
            && self.startup_affinity.contains(&cpu)
        {
            yorkie_numa::pin_current_thread_to_cpu(cpu);
        }
    }

    /// The CPU [`Self::pin_to_shared_cpu`] pins to: the low-priority worker's
    /// when the plan has one, and otherwise the build's shared CPU.
    fn shared_cpu(&self) -> Option<usize> {
        #[cfg(low_priority_worker)]
        if let Some(cpu) = self.worker_plan.low_priority_cpu() {
            return Some(cpu);
        }
        if self.worker_plan.pinned() {
            crate::config::SHARED_CPU
        } else {
            None
        }
    }

    /// The position a search would run from.
    pub fn position(&self) -> &Position {
        &self.pos
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

    /// Override the yorkie book list with `(file, weight)` pairs, an empty list
    /// disabling it. The `info string` that reports a game's book names the
    /// file as it is given here.
    ///
    /// `yorkie_book_files` is compiled in like every other setting, so a test
    /// that drives a session against books of its own names the files here.
    pub fn set_yorkie_books(&mut self, books: Vec<(PathBuf, u32)>) {
        self.yorkie_choices = books
            .into_iter()
            .map(|(path, weight)| YorkieBookChoice {
                name: path.clone(),
                path,
                weight,
            })
            .collect();
    }

    /// Override how many leading moves of the `position` command the yorkie
    /// book's game path leaves out: the starting position and the positions
    /// after the first `moves - 1` moves do not count as having occurred.
    ///
    /// `yorkie_book_path_skip_moves` is compiled in like every other setting,
    /// so a test that drives a session under another count names it here.
    pub fn set_yorkie_book_path_skip_moves(&mut self, moves: usize) {
        self.yorkie_path_skip = moves;
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
    /// measurement command carries that still means something here. At most
    /// [`Self::pool_capacity`].
    #[cfg(feature = "verbose3")]
    pub fn resize_pool(&mut self, threads: usize) {
        self.pool_threads = threads;
        self.rebuild_pool();
    }

    /// The most workers [`Self::resize_pool`] can be asked for: the room this
    /// build holds for a pool's flags, counters and helper slots.
    #[cfg(feature = "verbose3")]
    pub fn pool_capacity() -> usize {
        POOL_CAPACITY
    }

    /// Append `"Using N thread[s] on CPUs <list>"` for the live pool to `out`.
    #[cfg(feature = "verbose3")]
    pub fn write_thread_allocation_information(&self, out: &mut TextWriter<'_>) {
        write_thread_allocation_information(self.pool.size(), &self.worker_plan, out);
    }

    /// If a search worker is running, request its stop and join it, reclaiming
    /// the session-owned state. Idempotent: a no-op when idle.
    ///
    /// Joining is also what leaves this thread alone with the shared
    /// transposition table, which is what a new game's clear wants.
    pub fn finish_search_join(&mut self) {
        if let Some(active) = self.search.take() {
            // Both flags: the worker may be searching for its reply, or thinking
            // ahead past one it has already given, and the join below waits for
            // whichever it is.
            STOP.store(true, Ordering::Relaxed);
            PONDER_STOP.store(true, Ordering::Relaxed);
            let state = active
                .handle
                .join()
                .expect("search worker thread must not panic");
            self.histories = Some(state.histories);
            self.vote_buffers = Some(state.vote_buffers);
            self.ponder_record = state.ponder;
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
        let (addr, span) = TranspositionTable::shared().backing_region();
        let mut bytes = [0u8; NOTICE_BYTES];
        let mut out = TextWriter::new(&mut bytes);
        out.bytes(b"transposition table: ")
            .u64((yorkie_storage::TABLE_BYTES / (1024 * 1024)) as u64)
            .bytes(b" MiB; ");
        match table_placement(&self.worker_plan) {
            TablePlacement::OnNode(node) => {
                out.bytes(b"preferred on node ").u64(node as u64).byte(b' ');
                out.bytes(outcome(mempolicy::prefer_region_on_node(addr, span, node)));
            }
            TablePlacement::AcrossNodes(nodes) => {
                out.bytes(b"interleave on nodes ");
                write_cpu_list(&nodes, &mut out);
                out.byte(b' ');
                out.bytes(outcome(mempolicy::interleave_region_over_nodes(
                    addr, span, &nodes,
                )));
            }
            TablePlacement::ProcessPolicy => {
                out.bytes(b"no node policy");
            }
        }
        let huge = yorkie_storage::advise_huge_pages(addr, span);
        out.bytes(b"; huge pages ").bytes(outcome(huge));

        self.sink.notice(out.as_bytes())
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
        self.worker_plan = WorkerPlan::of(
            self.settings.worker_cpus(),
            self.settings.worker_system_nodes(),
            requested,
        );
        #[cfg(low_priority_worker)]
        {
            self.worker_plan = self
                .worker_plan
                .with_low_priority(self.settings.low_priority_worker());
        }
        // Rebuild the per-node shared correction / pawn tables from the fresh
        // assignment, so every pool rebuild resets them. The coordinator's own
        // game-scoped per-worker tables persist, so only its shared handle is
        // swapped.
        self.worker_shared = build_worker_shared(&self.worker_plan);
        if let Some(h) = self.histories.as_mut() {
            h.set_shared(Arc::clone(&self.worker_shared[0]));
        }
        // The coordinator's per-worker tables outlive a pool rebuild and were
        // faulted on the command thread, so re-assert their placement for the fresh
        // assignment. Helpers need nothing here: they are respawned and each
        // allocates its own bundle on-thread after pinning.
        place_coordinator_histories(self.histories.as_ref(), &self.worker_plan);
        self.pool
            .set_with_binding(requested, self.worker_plan, &self.worker_shared);
        // The coordinator's collection buffers are sized by the worker count,
        // so the rebuilt pool is where they are resized — nowhere else changes
        // it, and no search runs across a rebuild.
        let size = self.pool.size();
        if let Some(buffers) = self.vote_buffers.as_mut() {
            buffers.fit_to_pool(size);
        }
    }

    /// (Re)load the opening books from the current options — the reference's
    /// `BookMoveSelector::read_book`. Reloads only when the `(name list,
    /// on-the-fly, IgnoreBookPly)` capture changed.
    ///
    /// `no_book`, an unsupported format, or an open failure all leave that name
    /// out of the list without panicking; a `.db` whose file is absent falls
    /// back to the `.ybb` sibling with the reference's fallback info string.
    fn reload_book(&mut self) -> io::Result<()> {
        let book_file = self.settings.book_file();
        let book_dir = self.settings.book_dir();
        let on_the_fly = self.settings.book_on_the_fly();
        let ignore_book_ply = self.settings.ignore_book_ply();
        let base = book_path(book_dir, book_file);

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
        if book_file == NO_BOOK {
            return Ok(());
        }

        // The "priority book file exists twice" notices from the enumeration,
        // verbatim from the reference.
        let mut bytes = [0u8; NOTICE_BYTES];
        for notice in &notices {
            let mut out = TextWriter::new(&mut bytes);
            out.bytes(b"priority book file exists twice. use : ")
                .path(notice);
            self.sink.notice(out.as_bytes())?;
        }

        let mut books: Vec<Book> = Vec::new();
        for name in &names {
            // Resolve a `.db` whose file is absent to its `.ybb` sibling. The
            // pin applies this per name inside `MemoryBook::read_book`; for a
            // numbered name it is always a no-op (the enumeration already
            // proved the file exists).
            let resolved = resolve_book_filename_with_ybb_fallback(name);
            if &resolved != name {
                let mut out = TextWriter::new(&mut bytes);
                out.bytes(b"book file fallback : ")
                    .path(name)
                    .bytes(b" -> ")
                    .path(&resolved);
                self.sink.notice(out.as_bytes())?;
            }

            // Divergence from the reference: only `.ybb` is supported. Anything
            // else — including a `.db` the reference's two-extension resolution
            // picked for a numbered slot — behaves as no-book after an
            // info-string notice, never a panic and never a SILENT skip: a
            // silent skip would hide a book the reference would have used.
            if !has_book_ext(&resolved, BOOK_EXT_YBB) {
                let mut out = TextWriter::new(&mut bytes);
                out.bytes(b"unsupported book format : ").path(&resolved);
                self.sink.notice(out.as_bytes())?;
                continue;
            }

            let opened = if on_the_fly {
                Book::open_on_the_fly(&resolved)
            } else {
                Book::open_in_memory(&resolved)
            };
            let mut out = TextWriter::new(&mut bytes);
            match opened {
                Ok(book) => {
                    let count = book.record_count();
                    books.push(book);
                    out.bytes(b"book loaded : ").u64(count).bytes(b" positions");
                }
                Err(e) => {
                    // Mirrors the reference's open/validate failure → this name is left
                    // out of the priority list.
                    out.bytes(b"book load failed : ");
                    e.write_message(|fragment| {
                        out.bytes(fragment);
                    });
                }
            }
            self.sink.notice(out.as_bytes())?;
        }

        if !books.is_empty() {
            self.book = Some(Box::leak(Box::new(LoadedBook {
                books,
                ignore_book_ply,
            })));
        }
        Ok(())
    }

    /// Load every configured yorkie book, unless the ones already loaded are
    /// they.
    ///
    /// Unlike a `.ybb` that fails to open, a yorkie book that fails is not
    /// passed over: it is named in the config to be played from, so the first
    /// failure comes back for the readiness handshake to refuse on, and none of
    /// the list is loaded. An empty list loads nothing.
    fn load_yorkie_books(&mut self) -> io::Result<Option<(PathBuf, YorkieBookError)>> {
        if self.yorkie_choices.is_empty() {
            self.yorkie_books = None;
            return Ok(None);
        }
        if self.yorkie_books.as_ref().is_some_and(|(loaded, _)| {
            loaded
                .iter()
                .eq(self.yorkie_choices.iter().map(|choice| &choice.path))
        }) {
            return Ok(None);
        }
        self.yorkie_books = None;
        let mut books = Vec::with_capacity(self.yorkie_choices.len());
        for choice in &self.yorkie_choices {
            let book = match YorkieBook::open(&choice.path) {
                Ok(book) => book,
                Err(e) => return Ok(Some((choice.path.clone(), e))),
            };
            let mut bytes = [0u8; NOTICE_BYTES];
            let mut out = TextWriter::new(&mut bytes);
            out.bytes(b"yorkie book loaded : ")
                .u64(book.shared_count() as u64)
                .bytes(b" shared + ")
                .u64(book.path_dependent_count() as u64)
                .bytes(b" path-dependent entries");
            self.sink.notice(out.as_bytes())?;
            books.push(book);
        }
        let paths = self
            .yorkie_choices
            .iter()
            .map(|choice| choice.path.clone())
            .collect();
        self.yorkie_books = Some((paths, Vec::leak(books)));
        self.reserve_game_path();
        Ok(None)
    }

    /// Draw the book this game plays from and say which it is, at the start of
    /// a game. Nothing is drawn or said when there is no yorkie book.
    fn draw_yorkie_book(&mut self) -> io::Result<()> {
        if self.yorkie_choices.is_empty() {
            return Ok(());
        }
        self.yorkie_choice = draw_weighted(&self.yorkie_choices, &mut self.yorkie_prng);
        let mut bytes = [0u8; NOTICE_BYTES];
        let mut out = TextWriter::new(&mut bytes);
        out.bytes(b"yorkie book: ")
            .path(&self.yorkie_choices[self.yorkie_choice].name);
        self.sink.notice(out.as_bytes())
    }

    /// The yorkie book the game in progress plays from, if one is loaded.
    fn chosen_yorkie_book(&self) -> Option<&'static YorkieBook> {
        let (_, books) = self.yorkie_books.as_ref()?;
        let books: &'static [YorkieBook] = books;
        books.get(self.yorkie_choice)
    }

    /// Give every game-path buffer room for the longest game a `position`
    /// command may name, so that recording and handing over a path allocates
    /// nothing once the game has started.
    fn reserve_game_path(&mut self) {
        let shared = &mut *GAME_PATH.lock().unwrap_or_else(PoisonError::into_inner);
        for path in [
            &mut self.last_position.path,
            &mut self.pending_position.path,
            shared,
        ] {
            path.reserve_exact(GAME_PATH_CAPACITY - path.len());
        }
    }

    /// Write into `out` every position of the game the last `position` command
    /// named: its starting position, then the position after each of its
    /// moves, the current one last — less the leading positions
    /// [`seen_positions`] trims away. What a path-dependent yorkie-book entry
    /// is matched against.
    ///
    /// A command read before the book was loaded recorded no path, and the
    /// current position then stands alone.
    fn fill_game_path(&self, out: &mut Vec<PackedSfen>) {
        out.clear();
        if self.last_position.path.is_empty() {
            out.push(sfen_pack(&self.pos));
        } else {
            out.extend_from_slice(seen_positions(
                &self.last_position.path,
                self.yorkie_path_skip,
            ));
        }
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
        // initialisation-phase notice, present in every build. A build that
        // pins nothing says so, since its CPUs are whatever runs it allows.
        {
            let mut bytes = [0u8; NOTICE_BYTES];
            let mut out = TextWriter::new(&mut bytes);
            if self.worker_plan.pinned() {
                out.bytes(b"workers on CPUs ");
                write_cpu_list(&self.worker_plan.distinct_cpus(), &mut out);
            } else {
                out.bytes(b"workers unpinned");
            }
            #[cfg(low_priority_worker)]
            write_low_priority_suffix(&self.worker_plan, &mut out);
            self.sink.notice(out.as_bytes())?;
        }
        // A worker that was to give way and could not is said once a session,
        // and the engine plays on with it at normal priority.
        #[cfg(low_priority_worker)]
        if self.worker_plan.low_priority_worker().is_some()
            && !self.low_priority_refusal_reported
            && let Some(refusal) = low_priority_refusal()
        {
            self.low_priority_refusal_reported = true;
            let mut bytes = [0u8; NOTICE_BYTES];
            let mut out = TextWriter::new(&mut bytes);
            out.bytes(b"low-priority worker not lowered (");
            if refusal > 0 {
                out.bytes(b"errno ").u64(refusal as u64);
            } else {
                out.bytes(b"no per-thread priority on this platform");
            }
            out.bytes(b"); it runs at normal priority");
            self.sink.notice(out.as_bytes())?;
        }
        // The machine is the one the binary was built for, so the workers'
        // nodes are the ones the table belongs on. Done before anything else
        // here, so the policy is in force before the first page of it is
        // touched.
        self.place_transposition_table()?;
        // Load / reload the opening book (the reference does this in `isready`).
        self.reload_book()?;
        if let Some((path, reason)) = self.load_yorkie_books()? {
            return Ok(ReadyOutcome::YorkieBookFailed(path, reason));
        }
        let path = self.evaluation_file_path();

        // Idempotent: a repeat reuses what is already in memory — no
        // second read of the file, and on a multi-node machine no second copy
        // into the regions the first one is being read from.
        if self.eval.as_ref().is_some_and(|e| e.path == path) {
            return Ok(ReadyOutcome::Ready);
        }

        let mut bytes = [0u8; NOTICE_BYTES];
        let mut placement = TextWriter::new(&mut bytes);
        match self.place_evaluation_network(&path, &mut placement) {
            Ok((eval, warnings)) => {
                // Surface the complaints the conversion had about the source
                // network (hash mismatches) as notices before the readiness
                // reply, mirroring the reference `LoadAndShare` /
                // `Detail::ReadParameters` diagnostics. A clean network carries
                // none, so a correct one emits nothing new.
                for warning in &warnings {
                    self.sink.notice(warning)?;
                }
                self.sink.notice(placement.as_bytes())?;
                self.eval = Some(eval);
                Ok(ReadyOutcome::Ready)
            }
            Err(e) => Ok(ReadyOutcome::LoadFailed(e)),
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
        notice: &mut TextWriter<'_>,
    ) -> Result<(LoadedEval, Vec<Vec<u8>>), NnueError> {
        // Nothing may still be reading the regions when they are filled: they
        // are the process's only storage for the network. Any search has been
        // joined by the caller; clearing this here is what leaves the engine in
        // the "no network loaded" state until the fill succeeds.
        self.eval = None;

        let mut warnings = Vec::new();
        notice
            .bytes(b"evaluation network: ")
            .u64((network_file::DATA_BYTES / (1024 * 1024)) as u64)
            .bytes(b" MiB; ");

        if network_file::SHARED_MAPPING {
            // SAFETY: nothing is pointed at the region — no search that could
            // have read it survives, each one having been joined before the
            // readiness handshake reached here.
            warnings = unsafe { network_file::map_shared(path)? };
            let (addr, span) = network_file::Region::<0>::new().parameter_region();
            let huge = yorkie_storage::advise_huge_pages(addr, span);
            notice
                .bytes(b"one shared mapping; huge pages ")
                .bytes(outcome(huge));
        } else {
            let pinned = self.worker_plan.pinned();
            notice.bytes(if pinned {
                &b"one copy on "[..]
            } else {
                &b"one copy, "[..]
            });
            for (slot, node) in config::EVAL_REGION_NODES.iter().copied().enumerate() {
                if slot != 0 {
                    notice.bytes(b"; ");
                }
                let (addr, span) = network_file::region_backing(slot);
                if pinned {
                    notice.bytes(b"node ").u64(node as u64).byte(b' ');
                    notice.bytes(outcome(mempolicy::migrate_region_to_node(addr, span, node)));
                } else {
                    notice.bytes(b"no node policy");
                }
                let huge = yorkie_storage::advise_huge_pages(addr, span);
                // SAFETY: as the shared mapping above.
                let w = unsafe { network_file::load_into_region(slot, path)? };
                // Every region reads the same file, so its complaints are the
                // same each time; they are worth reporting once.
                warnings = w;
                notice.bytes(b", huge pages ").bytes(outcome(huge));
            }
        }

        Ok((
            LoadedEval {
                path: path.to_path_buf(),
            },
            warnings,
        ))
    }

    /// How the machine this process runs on differs from the one the binary was
    /// built for, or `None` when they agree.
    ///
    /// The live layout is resolved exactly as the build resolved it, over every
    /// online CPU, so what is compared is machine against machine and not
    /// machine against process. A tree that cannot be read is itself a refusal —
    /// an unverifiable layout is not a matching one.
    fn numa_layout_refusal(&self) -> Option<LayoutRefusal> {
        let opts = match yorkie_numa::machine_sysfs_options(&self.sysfs_root) {
            Ok(opts) => opts,
            Err(e) => return Some(LayoutRefusal::Unreadable(e)),
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
    ///
    /// A worker still thinking ahead from the previous move is ended first: the
    /// game has moved on, and the next `go` must not have to wait for it.
    pub fn set_position<'a>(
        &mut self,
        sfen: PositionSfen<'a>,
        moves: &'a [u8],
    ) -> Result<(), PositionRefusal<'a>> {
        self.stop_pondering();
        let record_path = self.yorkie_books.is_some() && self.settings.usi_own_book();
        let Self {
            pending_position: pending,
            scratch_pos: scratch,
            legal_buf,
            pseudo_buf,
            ..
        } = self;
        pending.set_start(sfen);
        pending.start_into(scratch).map_err(PositionRefusal::Sfen)?;
        if record_path {
            pending.path.push(sfen_pack(scratch));
        }
        // The position the command's last move is played from, and that move:
        // what decides whether this command continues the game the search past
        // the reply was thinking about.
        let mut before_last = None;
        for mv in yorkie_state::text::tokens(moves) {
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
            before_last = Some((scratch.key(), parsed));
            scratch.do_move(parsed);
            pending.moves.push(parsed);
            if record_path {
                pending.path.push(sfen_pack(scratch));
            }
        }
        // Both installs are a swap: what they displace becomes the next
        // command's scratch, buffers and all, so a game's `position` commands
        // reach the allocator only for the first of them.
        std::mem::swap(&mut self.pos, &mut self.scratch_pos);
        std::mem::swap(&mut self.last_position, &mut self.pending_position);
        // Only a position that is the thought-about root plus the one reply the
        // opponent played is credited; any other — a new game, a position the
        // host edited — is budgeted from its own clock alone.
        self.pondered_ms = match (&self.ponder_record, before_last) {
            (Some(record), Some((key, reply))) if key == record.key => record.spent_on(reply),
            _ => 0,
        };
        Ok(())
    }

    /// Start a new game: reclaim any search, empty the table, rebuild the
    /// history tables and the pool, and reset every per-game carry-forward (the
    /// reference `search_clear`); then draw the yorkie book the game plays from.
    pub fn new_game(&mut self) -> io::Result<()> {
        // Reclaim any running search, then reset the game state: startpos, an
        // emptied table, and fresh history tables (the reference
        // `search_clear`). Clearing the table also resets its generation; the next
        // `go` bumps it again via `run_root`.
        self.finish_search_join();
        // A low-priority helper the last search replied without may still be
        // finishing: wait for it, since nothing here is on a clock.
        #[cfg(low_priority_worker)]
        self.pool.settle();
        self.pos.reset_startpos();
        // The join above left this thread as the only one touching the table,
        // so the clear races nothing.
        TranspositionTable::shared().clear();
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
        // What the last game's search past a reply left is no part of this one,
        // even where a position of this game repeats its root.
        self.ponder_record = None;
        self.pondered_ms = 0;
        // Reset the side-flip detector and the retained command state (the
        // `last_position` default is the reference's `"position startpos"`).
        self.last_game_ply = 0;
        self.last_position.reset();
        // Reset the helper workers' game-scoped histories too.
        // The reference `search_clear` clears every worker; here the helper
        // histories live in the pool threads, so recreating the pool gives them
        // fresh tables — the join above guarantees the helpers are idle first.
        // Routing through `rebuild_pool` keeps the NUMA binding assignment
        // consistent with the recreated helpers.
        self.rebuild_pool();
        self.draw_yorkie_book()?;
        // The pool and the history tables just rebuilt are a new game's setup,
        // not the first move's work, so they are not counted against it. The
        // join above ended any search of the last game, whose counts go the
        // same way.
        #[cfg(feature = "verbose1")]
        {
            clear_alloc_count();
            crate::stats::clear_search_counts();
        }
        Ok(())
    }

    /// Start a search under `params`.
    ///
    /// Returns as soon as the coordinator thread is running: the reply reaches
    /// the sink from there. [`GoOutcome::NoNetwork`] means nothing was started,
    /// and the caller owes the host the reply itself.
    pub fn go(&mut self, params: GoParams) -> GoOutcome {
        // The clock starts when the request arrives, so it is stamped before
        // anything this `go` does — including ending a search left thinking
        // ahead, whose stop and join are then spent on this move's own budget
        // rather than quietly outside it.
        let received = Instant::now();
        // A new `go` supersedes any lingering search; reclaim its state first.
        self.finish_search_join();
        // What was pondered on this position was decided when the position was
        // set, from the search that command ended. A record a later join left
        // belongs to no position, and this `go` starts the search that will
        // leave the next one.
        self.ponder_record = None;
        let pondered = std::mem::take(&mut self.pondered_ms);

        // Whether this `go` is answered with a move and then thought past: a
        // clocked request in a build that thinks ahead at all.
        let ponder_after_reply = self.settings.ponder() && is_a_game_move(&params);
        // Only a game move runs on a clock that time spent earlier can stand in
        // for; an analysis is bounded by something else.
        let credit = if is_a_game_move(&params) {
            pondered * config::PONDER_CREDIT / 100
        } else {
            0
        };

        // The ply the search runs at, carried so a completed real search updates
        // `last_game_ply`.
        let game_ply = self.pos.ply() as i32;

        // Build the coordinator job (option-seeded limits, all per-`go`
        // snapshots). `None` means no network is loaded, which is the caller's
        // to answer.
        let Some(job) = self.prepare_coordinator_job(
            params,
            received,
            credit,
            ponder_after_reply,
            #[cfg(feature = "verbose2")]
            false,
        ) else {
            return GoOutcome::NoNetwork;
        };

        // The coordinator's own system node, resolved before the thread starts
        // so the network type it searches with is selected once per `go`.
        let node = self.worker_plan.system_node(0);
        let handle = std::thread::spawn(move || {
            let outcome = with_eval_network!(node, |net| run_coordinated(net, job));
            SearchState {
                histories: outcome.histories,
                time_state: outcome.time_state,
                vote_buffers: outcome.vote_buffers,
                ponder: outcome.ponder,
            }
        });

        self.search = Some(ActiveSearch { handle, game_ply });
        GoOutcome::Started
    }

    /// End the thinking-ahead search, if one is running, before the command
    /// that arrived is handled.
    ///
    /// Raising the flag is what ends the search; the join is what leaves the
    /// session idle, and is taken only once the worker has said it is thinking
    /// ahead — during a real search this must not wait for the reply. A worker
    /// that raises the flag just after this looked stops on the flag anyway and
    /// is reclaimed by the next command that needs its state.
    pub fn stop_pondering(&mut self) {
        if self.search.is_some() {
            PONDER_STOP.store(true, Ordering::Relaxed);
            if PONDERING.load(Ordering::Relaxed) {
                self.finish_search_join();
            }
        }
    }

    /// Build the [`CoordinatorJob`] for one search — the shared preamble of both
    /// `go` and `bench`.
    ///
    /// Returns `None` when no network is loaded; the caller emits the resign or
    /// notice appropriate to its context. `received` is the instant the request
    /// arrived, which is where its clock starts, and `credit_ms` the time that
    /// clock counts as spent before it ([`TimeManagement::set_credit`]).
    /// `ponder_after_reply` asks the
    /// coordinator to keep searching once it has replied — never for a
    /// measurement run, which has to end when its position does.
    /// `disable_pv_interval` forces the per-iteration PV interval to zero so
    /// every iteration prints, and is set only by `bench`; there is no such
    /// interval to force below `verbose2`.
    fn prepare_coordinator_job(
        &mut self,
        limits: GoParams,
        received: Instant,
        credit_ms: i64,
        ponder_after_reply: bool,
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
        let now = received;
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
        // `last_game_ply - game_ply` — negate the persisted previous scores
        // (each unless it is the `VALUE_INFINITE` first-move sentinel) before
        // they seed `iterValue` / `fallingEval`. On the normal same-side case
        // the parity is even and the scores pass through unchanged.
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
            let mut tm = TimeManagement::init(
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
            tm.set_credit(credit_ms);
            #[cfg(feature = "verbose1")]
            if tm.mtg_error {
                self.sink
                    .diagnostic(b"Error! : MaxMovesToDraw is too small.");
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
        // Clear the session's flags and counters for this search.
        let workers = self.pool.size();
        arm_search(workers);
        let control = SearchControl {
            stop: Some(&STOP),
            // A ceiling of zero nodes and a ceiling of one end the same
            // search: the first completed iteration precedes the first
            // consultation of the ceiling either way.
            #[cfg(feature = "verbose2")]
            node_limit: limits
                .nodes
                .map(|n| NonZeroU64::new(n).unwrap_or(NonZeroU64::MIN)),
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
        // `finish_search_join` first), so these are the live helpers' for this
        // whole `go`.
        let helpers = self.pool.helpers();

        // The shared table is a `static`, so nothing is handed over for it; the
        // main histories are still lent take-and-return and reclaimed on join.
        // Helper histories live in the pool threads.
        let histories = self
            .histories
            .take()
            .expect("session histories present when idle");
        let vote_buffers = self
            .vote_buffers
            .take()
            .expect("session collection buffers present when idle");
        let sink = self.sink.clone();
        let pos = self.pos.clone();

        // Book state for this `go`: the loaded book, the `USI_OwnBook` gate and
        // a fresh seed. The selection settings are [`BOOK_CONFIG`] and travel
        // with no job.
        let book = self.book;
        let own_book = self.settings.usi_own_book();
        let yorkie = yorkie_for_go(own_book, self.chosen_yorkie_book(), |out| {
            self.fill_game_path(out)
        });
        self.book_seed = self
            .book_seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let book_seed = self.book_seed;
        // Whether the coordinator holds its reply (book or searched) until
        // `stop`: `go infinite` (the SKIP_SEARCH wait loop), which is the only
        // request that holds one and arrives with `verbose2`.
        #[cfg(feature = "verbose2")]
        let infinite = limits.infinite;

        // The per-worker counters of this pool, cleared by `arm_search` above.
        #[cfg(feature = "verbose2")]
        let node_slots = node_tally(workers);
        let bmc_slots = bmc_tally(workers);

        // What the coordinator keeps searching on once its reply is out. Its
        // flags are the session's own, cleared by `arm_search` like the rest.
        let ponder = PonderJob {
            enabled: ponder_after_reply,
            helpers,
            #[cfg(feature = "verbose2")]
            node_slots,
            bmc_slots,
            #[cfg(feature = "random")]
            random_seed: self.random_seed,
            #[cfg(feature = "verbose2")]
            multi_pv,
        };

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
        let worker_plan = self.worker_plan;

        Some(CoordinatorJob {
            pos,
            #[cfg(feature = "verbose2")]
            depth,
            #[cfg(feature = "verbose2")]
            use_voting,
            control,
            histories,
            vote_buffers,
            #[cfg(feature = "verbose2")]
            node_slots,
            bmc_slots,
            helpers,
            worker_plan,
            book,
            yorkie,
            own_book,
            book_seed,
            ponder,
            #[cfg(feature = "verbose2")]
            infinite,
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
        // always exists. A measurement position is answered and left: a search
        // that outlived it would be measured into the next one.
        let job = self.prepare_coordinator_job(params, Instant::now(), 0, false, true)?;
        let node = self.worker_plan.system_node(0);
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
    /// A search that has not replied yet emits its reply and its state is
    /// reclaimed on the next command that needs it. A worker thinking ahead
    /// stops with nothing to say and is joined here, so `stop` leaves the
    /// session idle. With no search running this is a silent no-op.
    pub fn stop(&mut self) {
        if self.search.is_some() {
            STOP.store(true, Ordering::Relaxed);
        }
        self.stop_pondering();
    }
}

impl<P: EngineSink> Drop for Engine<P> {
    /// The flags, counters and helper slots a search runs on are the process's
    /// rather than this engine's, so an engine going away takes its search with
    /// it: a coordinator left running would still be reading them when the next
    /// engine arms them for a search of its own.
    fn drop(&mut self) {
        self.finish_search_join();
    }
}

/// Whether `params` asks for a move in a game rather than an analysis: it names
/// a clock, and no clause that bounds the search by something else.
///
/// The distinction is what decides whether the engine keeps searching once it
/// has answered — the opponent's clock is only running after a real move.
fn is_a_game_move(params: &GoParams) -> bool {
    let clocked = params.btime.is_some()
        || params.wtime.is_some()
        || params.byoyomi.is_some()
        || params.binc.is_some()
        || params.winc.is_some();
    // The clauses that bound a search by something other than the clock all
    // arrive with `verbose2`; without it a `go` naming a clock is a game move
    // and there is nothing further to ask.
    #[cfg(feature = "verbose2")]
    let clocked = clocked
        && params.depth.is_none()
        && params.nodes.is_none()
        && params.mate.is_none()
        && params.movetime.is_none()
        && !params.infinite;
    clocked
}

/// The `book_file` name that means no book.
const NO_BOOK: &str = "no_book";

/// One entry of the yorkie-book list: the file as the config names it, the
/// path it resolves to, and its selection weight.
struct YorkieBookChoice {
    name: PathBuf,
    path: PathBuf,
    weight: u32,
}

/// Mixed into the session seed for the yorkie-book draw, so the draw does not
/// replay the stream the book-move selection takes from the same seed.
const YORKIE_DRAW_SEED_MIX: u64 = 0x5DEE_CE66_D1CE_B00C;

/// The index of one of `choices`, each drawn with probability its weight over
/// the sum of all weights. `choices` is not empty.
fn draw_weighted(choices: &[YorkieBookChoice], prng: &mut Prng) -> usize {
    let total: u64 = choices.iter().map(|choice| u64::from(choice.weight)).sum();
    let mut ticket = prng.rand(total);
    for (i, choice) in choices.iter().enumerate() {
        let weight = u64::from(choice.weight);
        if ticket < weight {
            return i;
        }
        ticket -= weight;
    }
    choices.len() - 1
}

/// The absolute path a `<BookDir>/<BookFile>` pair resolves to.
///
/// Mirrors the reference `get_book_name`: join `BookDir` onto the binary's
/// folder, then `BookFile`. An absolute `BookDir` (as tests use) wins over
/// the binary folder — `Path::join` semantics match `Path::Combine`.
/// `BookDir` / `BookFile` are data paths opened as-is via `std::fs`, with
/// no shell or metacharacter interpretation.
fn book_path(book_dir: &str, book_file: &str) -> PathBuf {
    let base = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."));
    base.join(book_dir).join(book_file)
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
    // The suffix is composed as bytes, and joined to the stem's own: a file name
    // is bytes on the platform this plays on, and nothing about it is text.
    let mut bytes = [0u8; 1 + 20 + 1 + 8];
    let mut suffix = TextWriter::new(&mut bytes);
    suffix
        .byte(b'-')
        .u64_padded(index as u64, 3)
        .byte(b'.')
        .bytes(extension.as_bytes());
    let mut name = stem.as_os_str().to_os_string();
    name.push(OsStr::from_bytes(suffix.as_bytes()));
    PathBuf::from(name)
}

/// Resolve priority book `index` for `base` (`resolve_priority_book_filename`).
///
/// The primary extension is the base name's own and wins when both files exist,
/// which also produces the reference's `priority book file exists twice` notice
/// — the second tuple element is the name that notice quotes, for the caller to
/// emit.
///
/// `None` means neither extension exists at this index, which ends the series.
fn resolve_priority_book_filename(base: &Path, index: usize) -> Option<(PathBuf, Option<PathBuf>)> {
    let stem = book_name_without_extension(base)?;

    let (primary_ext, secondary_ext) = if has_book_ext(base, BOOK_EXT_YBB) {
        (BOOK_EXT_YBB, BOOK_EXT_DB)
    } else {
        (BOOK_EXT_DB, BOOK_EXT_YBB)
    };
    let primary = priority_book_filename(&stem, index, primary_ext);
    let secondary = priority_book_filename(&stem, index, secondary_ext);

    if primary.exists() {
        let notice = secondary.exists().then(|| primary.clone());
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
/// The second tuple element carries the names the enumeration's `info string`s
/// quote, in list order, for the caller to emit.
fn book_names(base: &Path) -> (Vec<PathBuf>, Vec<PathBuf>) {
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

/// The SKIP_SEARCH hold, shared by the book-hit and the searched reply: a `go
/// infinite` withholds its reply until `stop`, reusing the async-stop machinery
/// rather than busy-waiting.
///
/// `go infinite` is the only request that holds a reply and it arrives with
/// `verbose2`, so a build without that feature never waits here at all.
#[cfg(feature = "verbose2")]
fn hold_reply(stop: &AtomicBool, infinite: bool) {
    while infinite && !stop.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// Answer a book hit the way the reference does on `search_skipped`: the
/// surviving candidates first, then — after the `go infinite` hold — the
/// terminal reply.
///
/// `time_ms` is stamped once, when the book answered, so the hold does not
/// inflate the elapsed time attributed to the reply.
///
/// The candidate report is `verbose2`; the reply is not, so a default build
/// answers a book hit with the move and nothing else.
fn emit_book_hit<P: EngineSink>(
    sink: &P,
    hit: &BookHit,
    #[cfg(feature = "verbose2")] hashfull: u32,
    #[cfg(feature = "verbose2")] time_ms: u64,
    #[cfg(feature = "verbose2")] infinite: bool,
    #[cfg(feature = "verbose2")] stop: &AtomicBool,
) {
    // Reported immediately, like the reference's in-probe isRoot block.
    #[cfg(feature = "verbose2")]
    sink.book_candidates(hit, hashfull, time_ms);

    #[cfg(feature = "verbose2")]
    hold_reply(stop, infinite);

    sink.book_reply(
        hit,
        #[cfg(feature = "verbose2")]
        hashfull,
        #[cfg(feature = "verbose2")]
        time_ms,
    );
}

/// Everything one helper needs to run its own iterative deepening for a single
/// `go`. The heavy state is shared: the stop flag and the per-worker counters
/// are the session's `static`s, of which the job carries the references, and
/// the transposition table is the one `static` every worker reads. The
/// node-shared history tables were handed to the helper when the pool spawned
/// it. The network is not here at all — which one this helper reads
/// was decided when its thread was spawned, and is the type its search was
/// compiled for. The position and the root-move list are cheap per-helper
/// copies (the reference `start_thinking` copies the root-move list to every
/// worker).
struct HelperJob {
    /// The root position.
    pos: Position,
    /// This helper's own copy of the root-move list.
    root_moves: Vec<RootMove>,
    /// The iterative-deepening depth ceiling for this `go`, below the search's
    /// own maximum. Only a `verbose2` build has a source for one — see
    /// [`CoordinatorJob::depth`], which this copies.
    #[cfg(feature = "verbose2")]
    limit_depth: i32,
    /// The stop flag every worker of this search polls: [`STOP`] for a `go`,
    /// [`PONDER_STOP`] for the search that thinks past its reply.
    stop: &'static AtomicBool,
    /// Whether this search is the one that thinks past the reply, whose root has
    /// the opponent to move.
    thinking_ahead: bool,
    /// Per-worker node counters; the helper publishes `nodes` to
    /// `node_slots[index]`. The aggregate they form is read by the node ceiling
    /// and by the search `info` lines, both `verbose2`.
    #[cfg(feature = "verbose2")]
    node_slots: &'static [TallySlot],
    /// Per-worker best-move-change counters; the helper `fetch_add`s its own
    /// `bmc_slots[index]` at the root, and the main worker folds every slot
    /// each iteration.
    bmc_slots: &'static [TallySlot],
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
    /// How wide the helper reads its root: single-line for a `go`, the
    /// configured width for the search that thinks past the reply.
    root_width: RootWidth,
}

/// The state of one helper's coordination slot. The coordinator drives
/// `Parked → Assigned` (dispatch) and `Finished → Parked` (collect); the helper
/// thread drives `Assigned → Running → Finished`. The pool sets `Exit` (only
/// ever over `Parked`, since every teardown path first joins the coordinator,
/// which returns every helper to `Parked`).
///
/// `Finished` holds the result inline: a slot is written once per helper per
/// search, and boxing the result would put a heap allocation on that path in
/// every build. Only a build carrying the most per-worker counts makes the
/// variant large enough for the lint, so it is allowed rather than expected.
#[cfg_attr(lazy_smp_defer, allow(clippy::large_enum_variant))]
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
///
/// The slots are the session's, like its flags: [`HELPER_SLOTS`] holds one per
/// helper a pool can have, and a pool of `n` workers runs its helpers on the
/// first `n − 1`.
struct HelperSlot {
    state: Mutex<SlotState>,
    cv: Condvar,
}

/// The helper slots of every pool this process builds, worker `h + 1` parked on
/// slot `h`.
static HELPER_SLOTS: [HelperSlot; POOL_CAPACITY - 1] =
    [const { HelperSlot::new() }; POOL_CAPACITY - 1];

impl HelperSlot {
    const fn new() -> Self {
        HelperSlot {
            state: Mutex::new(SlotState::Parked),
            cv: Condvar::new(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, SlotState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The slot's lock if it is free right now, without waiting for it: the
    /// helper holds it only for a moment, but a helper the scheduler takes off
    /// its CPU in that moment holds it until it runs again.
    #[cfg(low_priority_worker)]
    fn try_lock(&self) -> Option<MutexGuard<'_, SlotState>> {
        match self.state.try_lock() {
            Ok(guard) => Some(guard),
            Err(TryLockError::Poisoned(e)) => Some(e.into_inner()),
            Err(TryLockError::WouldBlock) => None,
        }
    }

    /// Post a search job to a parked helper and wake it.
    fn assign(&self, job: HelperJob) {
        *self.lock() = SlotState::Assigned(Box::new(job));
        self.cv.notify_all();
    }

    /// Post the job `job` builds if the helper is free to take one now, without
    /// waiting for it, and say whether it was posted.
    ///
    /// A helper is free when it is parked, or finished with a job whose result
    /// nobody collected — a search that has already replied without it, whose
    /// result is dropped here. A helper still searching, or whose slot is
    /// locked, is left out of the search this job is for, and `job` is not
    /// called.
    #[cfg(low_priority_worker)]
    fn try_assign(&self, job: impl FnOnce() -> HelperJob) -> bool {
        let Some(mut st) = self.try_lock() else {
            return false;
        };
        if !matches!(&*st, SlotState::Parked | SlotState::Finished(_)) {
            return false;
        }
        *st = SlotState::Assigned(Box::new(job()));
        drop(st);
        self.cv.notify_all();
        true
    }

    /// Take the helper's result if it arrives by `deadline`, and return the slot
    /// to `Parked`; `None` if it does not.
    ///
    /// A late helper's slot is left as it is, to be settled by whoever next
    /// needs it ([`Self::try_assign`], [`Self::settle`]). The one exception is a
    /// job the helper never picked up, which is taken back at the deadline, so
    /// that it does not start a search that has already ended.
    #[cfg(low_priority_worker)]
    fn collect_by(&self, deadline: Instant) -> Option<WorkerResult> {
        loop {
            let now = Instant::now();
            if let Some(mut st) = self.try_lock() {
                match &*st {
                    SlotState::Finished(_) => {
                        let SlotState::Finished(r) = std::mem::replace(&mut *st, SlotState::Parked)
                        else {
                            unreachable!("the slot was just seen Finished");
                        };
                        return Some(r);
                    }
                    SlotState::Assigned(_) if now >= deadline => {
                        *st = SlotState::Parked;
                        return None;
                    }
                    _ => {}
                }
            }
            if now >= deadline {
                return None;
            }
            std::thread::sleep(LATE_HELPER_POLL.min(deadline - now));
        }
    }

    /// Wait for as long as it takes until the helper is searching nothing, drop
    /// whatever result it left, and leave the slot `Parked` (or `Exit`, if it
    /// is already that).
    ///
    /// What a new game's table clear and a pool teardown need: no helper still
    /// running. Only the low-priority helper can be running here, finishing a
    /// search that already replied, with its stop flag raised.
    #[cfg(low_priority_worker)]
    fn settle(&self) {
        let mut st = self.lock();
        loop {
            match &*st {
                SlotState::Running => st = self.cv.wait(st).unwrap_or_else(|e| e.into_inner()),
                SlotState::Exit => return,
                _ => break,
            }
        }
        *st = SlotState::Parked;
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

/// The helpers one search dispatches to: the live pool's slots, and which of
/// them, if any, is the low-priority helper (`low_priority_worker`).
///
/// The low-priority helper is the one worker a search does not wait for. It is
/// handed a job only when it has handed back the last one, it stops on
/// `LOW_STOP` rather than on the search's own flag, and its result is
/// collected for at most `LATE_HELPER_WAIT`. Every other helper is waited for
/// as it always is: each runs alone on its own CPU at normal priority, and sees
/// the stop within microseconds.
///
/// Without `low_priority_worker` (the config key, or the Cargo feature of the
/// same name) none of that is compiled in: the searches dispatch to and wait
/// for every slot alike, in loops of their own.
#[derive(Clone, Copy)]
struct Helpers {
    slots: &'static [HelperSlot],
    /// The index into [`Self::slots`] of the low-priority helper.
    #[cfg(low_priority_worker)]
    low: Option<usize>,
}

#[cfg(low_priority_worker)]
impl Helpers {
    /// Hand every helper the job `job(h)` builds for helper `h` (worker
    /// `h + 1`), and say whether the low-priority helper took one: it is left
    /// out of this search while it is still finishing an earlier one.
    fn dispatch(&self, mut job: impl FnMut(usize) -> HelperJob) -> bool {
        let mut low_dispatched = false;
        for (h, slot) in self.slots.iter().enumerate() {
            if Some(h) == self.low {
                low_dispatched = slot.try_assign(|| {
                    LOW_STOP.store(false, Ordering::Relaxed);
                    HelperJob {
                        stop: &LOW_STOP,
                        ..job(h)
                    }
                });
            } else {
                slot.assign(job(h));
            }
        }
        low_dispatched
    }

    /// Collect every helper's result into `take`, in worker order.
    ///
    /// The caller has already raised the stop flag the helpers poll. Each
    /// full-priority helper is waited for until it finishes. The low-priority
    /// one has its own flag raised here and is collected only if it was
    /// dispatched and finishes within [`LATE_HELPER_WAIT`] of this call.
    fn collect(&self, low_dispatched: bool, mut take: impl FnMut(WorkerResult)) {
        LOW_STOP.store(true, Ordering::Relaxed);
        let deadline = Instant::now() + LATE_HELPER_WAIT;
        for (h, slot) in self.slots.iter().enumerate() {
            if Some(h) != self.low {
                take(slot.collect());
            } else if low_dispatched && let Some(result) = slot.collect_by(deadline) {
                take(result);
            }
        }
    }
}

/// The persistent helper thread body (the reference `idle_loop`): park until a
/// job (or exit) arrives, run one iterative deepening with the helper's own
/// game-scoped histories, publish the result, and re-park. The histories persist
/// across jobs within a game — the pool is recreated to reset them.
///
/// `shared` is this helper's node's shared correction / pawn tables, which the
/// pool hands over when it spawns the helper. A pool built by a unit test that
/// never dispatches a job hands over none.
fn helper_loop<N: NetworkParams>(
    net: N,
    slot: &'static HelperSlot,
    mut shared: Option<Arc<SharedHistories>>,
) {
    // The per-worker tables persist across `go`s (game-scoped) and are built on
    // the first job, attached to the node's shared tables, so a pool that is
    // rebuilt before it searches allocates none.
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

        // Run with the persistent histories, built on the first job. A helper's
        // control is stop-only (no deadlines, no node ceiling): the reference
        // runs `check_time` on the main worker alone; helpers merely poll the
        // shared stop flag.
        let histories_in = histories.unwrap_or_else(|| {
            WorkerHistories::with_shared(
                shared
                    .take()
                    .expect("the pool hands every helper it dispatches to its node's tables"),
            )
        });
        let (result, reclaimed) = {
            let mut qs = QSearch::with_histories(net, histories_in);
            qs.set_control(SearchControl {
                stop: Some(job.stop),
                #[cfg(feature = "verbose2")]
                node_limit: None,
                time: None,
            });
            #[cfg(feature = "verbose2")]
            qs.set_node_tally(job.node_slots, job.index);
            qs.set_best_move_tally(job.bmc_slots, job.index);
            qs.set_entering_king(job.entering_king);
            qs.set_thinking_ahead(job.thinking_ahead);
            #[cfg(feature = "verbose2")]
            qs.set_mate_mode(job.mate_mode);
            #[cfg(feature = "random")]
            qs.set_random(RANDOM_AMPLITUDE, job.random_seed);
            // Helpers run the MultiPV loop too, but with no sink they never emit.
            #[cfg(feature = "verbose2")]
            qs.set_multi_pv(job.multi_pv);
            qs.set_root_width(job.root_width);
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
///
/// The helpers park on the session's [`HELPER_SLOTS`], one process holding one
/// pool at a time, so what the pool owns is its threads and how many there are.
struct ThreadPool {
    /// The live helpers (`size − 1`), parked on the first that many slots of
    /// [`HELPER_SLOTS`].
    helpers: usize,
    /// The slot of the helper that runs at low priority, if the plan has one.
    #[cfg(low_priority_worker)]
    low: Option<usize>,
    /// The helper threads, joined on resize / teardown.
    handles: Vec<JoinHandle<()>>,
}

impl ThreadPool {
    /// Build a pool of `size` slots (one main + `size − 1` helpers), spawning the
    /// helper threads parked and idle, each pinned to the CPU a plan built for
    /// this process's own affinity gives it — the pool unit tests use this, so
    /// they never pin a test thread to a CPU the test runner may not use. No job
    /// is dispatched to such a pool, so its helpers are handed no shared tables.
    #[cfg(test)]
    fn new(size: usize) -> Self {
        Self::with_binding(size, WorkerPlan::of_allowed_cpus(size), &[])
    }

    /// Build a pool of `size` slots with a worker plan. Each helper thread
    /// (worker `1..`) pins itself to its assigned CPU once at spawn, before it
    /// parks, and takes worker `w`'s node-shared tables `shared[w]` with it.
    fn with_binding(size: usize, plan: WorkerPlan, shared: &[Arc<SharedHistories>]) -> Self {
        let mut pool = ThreadPool {
            helpers: 0,
            #[cfg(low_priority_worker)]
            low: None,
            handles: Vec::new(),
        };
        pool.set_with_binding(size, plan, shared);
        pool
    }

    /// Resize to `size` slots over this process's own CPUs — used only by the
    /// pool unit tests.
    #[cfg(test)]
    fn set(&mut self, size: usize) {
        self.set_with_binding(size, WorkerPlan::of_allowed_cpus(size), &[]);
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
    ///
    /// `size` is at most [`POOL_CAPACITY`]: the measurement command, the one
    /// caller with a size of its own, refuses a larger one before it gets here.
    fn set_with_binding(&mut self, size: usize, plan: WorkerPlan, shared: &[Arc<SharedHistories>]) {
        self.shutdown();
        let size = size.max(1);
        assert!(
            size <= POOL_CAPACITY,
            "a pool of {size} workers is past this build's {POOL_CAPACITY}"
        );
        #[cfg(low_priority_worker)]
        let low = plan.low_priority_worker().filter(|&w| w < size);
        #[cfg(low_priority_worker)]
        if low.is_some() {
            LOW_PRIORITY_OUTCOME.store(LOWERING_PENDING, Ordering::Relaxed);
        }
        for (worker_id, slot) in (1..size).zip(&HELPER_SLOTS) {
            // The slot is left `Exit` by the pool that used it last.
            *slot.lock() = SlotState::Parked;
            let shared = shared.get(worker_id).map(Arc::clone);
            #[cfg(low_priority_worker)]
            let lowered = low == Some(worker_id);
            self.handles.push(std::thread::spawn(move || {
                #[cfg(low_priority_worker)]
                {
                    #[cfg(test)]
                    HELPER_TIDS[worker_id - 1]
                        .store(yorkie_numa::current_thread_id(), Ordering::Release);
                    if lowered {
                        lower_this_helper();
                    }
                }
                let node = plan.system_node(worker_id);
                pin_worker(&plan, worker_id);
                // The one place this helper's node becomes the network type its
                // search is compiled for. Once per thread, before it parks.
                with_eval_network!(node, |net| helper_loop(net, slot, shared));
            }));
        }
        self.helpers = size - 1;
        #[cfg(low_priority_worker)]
        {
            self.low = low.map(|w| w - 1);
            // The pool is built between games, never on a clock, so it is ready
            // only once the readiness handshake can say how the lowering went.
            while low.is_some() && LOW_PRIORITY_OUTCOME.load(Ordering::Acquire) == LOWERING_PENDING
            {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }

    /// The helpers a search dispatches to.
    fn helpers(&self) -> Helpers {
        Helpers {
            slots: self.helper_slots(),
            #[cfg(low_priority_worker)]
            low: self.low,
        }
    }

    /// Wait until no helper is still finishing a search that has already
    /// replied — which only the low-priority helper can be — so the caller is
    /// alone with the shared table. Off any clock: a new game and a teardown
    /// are the callers.
    #[cfg(low_priority_worker)]
    fn settle(&self) {
        for slot in self.helper_slots() {
            slot.settle();
        }
    }

    /// Ask every helper to exit and join it, leaving only the main slot.
    /// Idempotent. Every full-priority helper is parked first (guaranteed by
    /// the `finish_search_join` every caller runs before a resize / teardown),
    /// and the low-priority one is waited for, so the `Exit` is never
    /// overwritten by a late `Finished` write.
    fn shutdown(&mut self) {
        for slot in self.helper_slots() {
            #[cfg(low_priority_worker)]
            slot.settle();
            *slot.lock() = SlotState::Exit;
            slot.cv.notify_all();
        }
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
        self.helpers = 0;
        #[cfg(low_priority_worker)]
        {
            self.low = None;
        }
    }

    /// The configured pool size: the main-worker slot plus the live helpers.
    fn size(&self) -> usize {
        self.helpers + 1
    }

    /// The slots of the live helpers, which a coordinator dispatches to and
    /// collects from for one `go`.
    fn helper_slots(&self) -> &'static [HelperSlot] {
        &HELPER_SLOTS[..self.helpers]
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
/// belongs to, worker 0 being the per-`go` search coordinator.
///
/// The assignment itself is the compiled one, so the plan borrows it and adds
/// only how many workers it covers: a small value, copied wherever a thread
/// needs it.
///
/// A plan with no CPUs is an unpinned build's: no worker is pinned and nothing
/// is placed on a node, so every worker runs wherever the process may and its
/// memory follows the process's own policy.
#[derive(Clone, Copy)]
struct WorkerPlan {
    /// One CPU per worker of the assignment, indexed by worker id; empty when
    /// the build pins nothing.
    cpus: &'static [usize],
    /// The system node of each of [`Self::cpus`].
    system_nodes: &'static [NumaIndex],
    /// The workers the plan covers, at least one.
    workers: usize,
    /// The worker that runs at low priority on the CPU the engine's non-search
    /// threads share (`low_priority_worker`), if any.
    #[cfg(low_priority_worker)]
    low_priority: Option<usize>,
}

impl WorkerPlan {
    /// The plan for `requested` workers over the assignment `cpus` /
    /// `system_nodes`.
    ///
    /// The assignment holds one CPU per configured worker. Asking for more than
    /// that is only reachable through the measurement command that carries its
    /// own worker count, and the extra workers wrap around onto the same CPUs —
    /// oversubscribing the CPUs this binary owns rather than spilling onto ones
    /// another binary was given.
    fn of(cpus: &'static [usize], system_nodes: &'static [NumaIndex], requested: usize) -> Self {
        WorkerPlan {
            cpus,
            system_nodes,
            workers: requested.max(1),
            #[cfg(low_priority_worker)]
            low_priority: None,
        }
    }

    /// The same plan with worker `worker` at low priority. Never the main
    /// worker (worker 0), which keeps the clock and decides when the search
    /// stops.
    #[cfg(low_priority_worker)]
    fn with_low_priority(self, worker: Option<usize>) -> Self {
        debug_assert_ne!(worker, Some(0), "the main worker is never lowered");
        WorkerPlan {
            low_priority: worker.filter(|&w| w != 0),
            ..self
        }
    }

    /// The worker that runs at low priority, if the plan has one and covers
    /// it.
    #[cfg(low_priority_worker)]
    fn low_priority_worker(&self) -> Option<usize> {
        self.low_priority.filter(|&w| w < self.workers)
    }

    /// The CPU the low-priority worker shares with the engine's non-search
    /// threads; `None` without one, or in a build that pins nothing.
    #[cfg(low_priority_worker)]
    fn low_priority_cpu(&self) -> Option<usize> {
        self.low_priority_worker().and_then(|w| self.cpu(w))
    }

    /// A plan over the CPUs this process is actually allowed on, all of them on
    /// one node — what a unit test spawning a pool needs, so its helper threads
    /// pin themselves somewhere the test runner permits.
    #[cfg(test)]
    fn of_allowed_cpus(requested: usize) -> Self {
        static ALLOWED: std::sync::OnceLock<(Vec<usize>, Vec<NumaIndex>)> =
            std::sync::OnceLock::new();
        let (cpus, nodes) = ALLOWED.get_or_init(|| {
            let cpus: Vec<usize> = yorkie_numa::startup_affinity().iter().copied().collect();
            // The node has to be one this binary declares a region for, since a
            // spawned helper selects its network from it.
            let nodes = vec![config::EVAL_REGION_NODES[0]; cpus.len().max(1)];
            (cpus, nodes)
        });
        Self::of(cpus, nodes, requested)
    }

    /// Whether the workers pin themselves to CPUs at all.
    fn pinned(&self) -> bool {
        !self.cpus.is_empty()
    }

    /// Worker `worker`'s slot in the assignment, wrapping past its end.
    fn pick(&self, worker: usize) -> usize {
        worker % self.system_nodes.len().max(1)
    }

    /// The CPU worker `worker` pins itself to, or `None` in a build that pins
    /// nothing.
    fn cpu(&self, worker: usize) -> Option<usize> {
        self.cpus.get(self.pick(worker)).copied()
    }

    /// The system node worker `worker`'s memory belongs on.
    fn system_node(&self, worker: usize) -> NumaIndex {
        self.system_nodes
            .get(self.pick(worker))
            .copied()
            .unwrap_or(0)
    }

    /// Every worker's CPU, in worker order; none in a build that pins nothing.
    fn cpus(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.workers).filter_map(|w| self.cpu(w))
    }

    /// Every worker's system node, in worker order.
    fn system_nodes(&self) -> impl Iterator<Item = NumaIndex> + '_ {
        (0..self.workers).map(|w| self.system_node(w))
    }

    /// The CPUs the plan uses, ascending and without repeats.
    fn distinct_cpus(&self) -> BTreeSet<usize> {
        self.cpus().collect()
    }

    /// The system NUMA nodes the plan's workers sit on, ascending and without
    /// repeats.
    fn distinct_system_nodes(&self) -> Vec<NumaIndex> {
        let set: BTreeSet<NumaIndex> = self.system_nodes().collect();
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
) -> Option<LayoutRefusal> {
    if live.nodes.len() != built.nodes.len() {
        return Some(LayoutRefusal::NodeCount {
            built: built.nodes.len(),
            live: live.nodes.len(),
        });
    }
    for (node, (live_cpus, built_cpus)) in live.nodes.iter().zip(&built.nodes).enumerate() {
        if live_cpus != built_cpus {
            return Some(LayoutRefusal::NodeCpus {
                node,
                live: live_cpus.clone(),
                built: built_cpus.clone(),
            });
        }
    }
    let missing: Vec<CpuIndex> = plan
        .distinct_cpus()
        .into_iter()
        .filter(|cpu| !affinity.contains(cpu))
        .collect();
    if !missing.is_empty() {
        return Some(LayoutRefusal::DeniedCpus(missing));
    }
    for (worker, (cpu, node)) in plan.cpus().zip(plan.system_nodes()).enumerate() {
        let live_node = live.system_node_of_cpu(cpu);
        if live_node != Some(node) {
            return Some(LayoutRefusal::CpuNode {
                worker,
                cpu,
                live_node,
                built_node: node,
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
    /// The workers are pinned nowhere, so no node is theirs and the table
    /// keeps the process's own policy.
    ProcessPolicy,
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
    if !plan.pinned() {
        return TablePlacement::ProcessPolicy;
    }
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
/// clock. A build that pins no worker places nothing here either.
fn place_coordinator_histories(histories: Option<&WorkerHistories>, plan: &WorkerPlan) {
    let Some(histories) = histories else {
        return;
    };
    if !plan.pinned() {
        return;
    }
    let node = plan.system_node(0);
    for (addr, len) in histories.backing_regions() {
        mempolicy::migrate_region_to_node(addr, len, node);
    }
}

/// Pin the calling thread to worker `worker`'s CPU and make that CPU's node its
/// allocation preference — or, in a build that pins no worker, leave the thread
/// and its memory policy as they are.
fn pin_worker(plan: &WorkerPlan, worker: usize) {
    if let Some(cpu) = plan.cpu(worker) {
        yorkie_numa::pin_current_thread_to_cpu_with_local_memory(cpu, plan.system_node(worker));
    }
}

/// Build the per-worker handles to the node-shared correction / pawn tables, one
/// set per system NUMA node the plan's workers sit on.
///
/// Each node's set is allocated and filled *inside* a thread pinned to one of
/// that node's worker CPUs, so the pages first-touch there. A build that pins
/// no worker allocates its one set on the calling thread.
///
/// Returns one [`Arc`] per worker, each pointing at its node's table set.
fn build_worker_shared(plan: &WorkerPlan) -> Vec<Arc<SharedHistories>> {
    let counts = shared_node_counts(plan.system_nodes());

    let mut node_shared: std::collections::BTreeMap<NumaIndex, Arc<SharedHistories>> =
        std::collections::BTreeMap::new();
    for (&node, &count) in &counts {
        let thread_count = count.next_power_of_two();
        // A CPU of that node: the first worker the plan puts there.
        let worker = plan
            .system_nodes()
            .position(|n| n == node)
            .expect("every counted node holds at least one worker");
        let built = match plan.cpu(worker) {
            Some(cpu) => {
                let mut built: Option<Arc<SharedHistories>> = None;
                yorkie_numa::execute_on_cpu(cpu, || {
                    built = Some(Arc::new(SharedHistories::new(thread_count)));
                });
                built.expect("execute_on_cpu ran the closure")
            }
            None => Arc::new(SharedHistories::new(thread_count)),
        };
        node_shared.insert(node, built);
    }

    plan.system_nodes()
        .map(|node| Arc::clone(&node_shared[&node]))
        .collect()
}

/// The node → worker-count map the shared-history construction sizes each set
/// from. Pure — no allocation and no pinning.
fn shared_node_counts(
    worker_nodes: impl IntoIterator<Item = NumaIndex>,
) -> std::collections::BTreeMap<NumaIndex, usize> {
    let mut counts: std::collections::BTreeMap<NumaIndex, usize> =
        std::collections::BTreeMap::new();
    for node in worker_nodes {
        *counts.entry(node).or_insert(0) += 1;
    }
    counts
}

// The thread-allocation diagnostic is emitted by the `verbose3` `bench`
// command and nowhere else, since that is the only command that can change the
// worker count; everything else about the assignment is a compile-time constant.
/// `"Using N thread[s] on CPUs <list>"` — the pool size, and the CPUs its
/// workers are pinned to; `"Using N thread[s] unpinned"` in a build that pins
/// none.
#[cfg(feature = "verbose3")]
fn write_thread_allocation_information(
    threads_size: usize,
    plan: &WorkerPlan,
    out: &mut TextWriter<'_>,
) {
    out.bytes(b"Using ")
        .u64(threads_size as u64)
        .byte(b' ')
        .bytes(if threads_size > 1 {
            &b"threads"[..]
        } else {
            &b"thread"[..]
        })
        .byte(b' ');
    if plan.pinned() {
        out.bytes(b"on CPUs ");
        write_cpu_list(&plan.distinct_cpus(), out);
    } else {
        out.bytes(b"unpinned");
    }
}

/// Thinking ahead: what the coordinator searches once it has answered, and the
/// workers that search runs on.
///
/// The engine plays a move and the opponent's clock starts; the position that
/// move reaches is the one position certain to matter next, so the same workers
/// keep searching it — with no limit — until the next command arrives. Nothing
/// comes out of that search: what it leaves behind is a warm transposition
/// table and warm history tables, which is what the next `go` starts from.
///
/// It runs under [`PONDER_STOP`] and raises [`PONDERING`] while it does.
struct PonderJob {
    /// Whether this `go` is thought past at all: the `ponder` setting, and a
    /// request that was a move in a game ([`is_a_game_move`]).
    enabled: bool,
    /// The helpers, dispatched to again for this search.
    helpers: Helpers,
    /// The per-worker node counters, published to as in any other search. What
    /// they hold has already been read by the reply this search follows.
    #[cfg(feature = "verbose2")]
    node_slots: &'static [TallySlot],
    /// The per-worker best-move-change counters.
    bmc_slots: &'static [TallySlot],
    /// The game's evaluation-noise seed, the same one the reply was searched
    /// under.
    #[cfg(feature = "random")]
    random_seed: u64,
    /// The `MultiPV` value the engine's own searches run at: apart from the
    /// width of its root, this is the search a `go` gets, aimed at a position
    /// one ply further on.
    #[cfg(feature = "verbose2")]
    multi_pv: usize,
}

/// What a search past a reply leaves for the next `go`'s clock: how long it
/// ran, and how its main worker spread that time over the opponent's replies.
///
/// Nothing in it is allocated for the purpose: the root moves are the list the
/// search ran on, each carrying the nodes spent below it.
struct PonderRecord {
    /// The position key of the search's root, the position the engine's own
    /// move reached.
    key: u64,
    /// When the search started and when its main worker stopped.
    started: Instant,
    stopped: Instant,
    /// The main worker's root moves, and the best move it moved out of them if
    /// it was stopped before its first iteration completed
    /// ([`QSearch::run_worker_on`]).
    root_moves: Vec<RootMove>,
    best: RootMove,
    /// The main worker's node count, the whole that the efforts are shares of.
    nodes: u64,
}

impl PonderRecord {
    /// The time in ms the search spent on `reply`: its duration, in the share of
    /// the main worker's nodes spent below that move. A reply the search never
    /// reached has no effort and was spent nothing on.
    fn spent_on(&self, reply: Move) -> i64 {
        let effort = self
            .root_moves
            .iter()
            .chain([&self.best])
            .find(|rm| rm.mv == reply)
            .map_or(0, |rm| rm.effort);
        let duration = self
            .stopped
            .saturating_duration_since(self.started)
            .as_millis() as i64;
        // Wide enough that an hour's search over a billion nodes per second
        // cannot overflow the product.
        (i128::from(duration) * i128::from(effort) / i128::from(self.nodes.max(1))) as i64
    }
}

impl PonderJob {
    /// Search on from `pos` with `played` applied, until the stop flag is
    /// raised or the search ends by itself, and hand the histories back with
    /// the record of what was searched, if anything was.
    ///
    /// Nothing is emitted here but the two `verbose1` markers: this search has
    /// no reply to give, and a `bestmove` out of it would be a second answer to
    /// a question the host already had answered.
    fn run<P: EngineSink, N: NetworkParams>(
        self,
        net: N,
        histories: WorkerHistories,
        sink: &P,
        pos: &Position,
        played: Move,
    ) -> (WorkerHistories, Option<PonderRecord>) {
        // The sink is here for those markers alone, so a build that prints
        // neither touches it at all.
        #[cfg(not(feature = "verbose1"))]
        let _ = sink;
        if !self.enabled || !played.is_ok() {
            return (histories, None);
        }
        // The command that ends this search may have arrived while the reply
        // was being written, in which case there is nothing to start.
        if PONDER_STOP.load(Ordering::Relaxed) {
            return (histories, None);
        }
        let mut root = pos.clone();
        root.do_move(played);
        // A position the engine reads as a game end has nothing to think about:
        // no legal reply to the move just played, or past the draw horizon.
        if root.ply() as i32 > DRAW_HORIZON {
            return (histories, None);
        }
        let mut root_moves = generate_root_moves(&root);
        if root_moves.is_empty() {
            return (histories, None);
        }

        let started = Instant::now();
        PONDERING.store(true, Ordering::Relaxed);
        #[cfg(feature = "verbose1")]
        sink.diagnostic(b"ponder start");
        // The generation is advanced exactly as it is for a `go`: this is a
        // search of its own, and what it stores belongs to it.
        TranspositionTable::shared().new_search();
        let entering_king = EnteringKingConfig::new(&root);

        macro_rules! ponder_job {
            ($h:expr) => {
                HelperJob {
                    pos: root.clone(),
                    root_moves: root_moves.clone(),
                    #[cfg(feature = "verbose2")]
                    limit_depth: SEARCH_MAX_DEPTH,
                    stop: &PONDER_STOP,
                    thinking_ahead: true,
                    #[cfg(feature = "verbose2")]
                    node_slots: self.node_slots,
                    bmc_slots: self.bmc_slots,
                    index: $h + 1,
                    entering_king,
                    #[cfg(feature = "verbose2")]
                    mate_mode: false,
                    #[cfg(feature = "random")]
                    random_seed: self.random_seed,
                    #[cfg(feature = "verbose2")]
                    multi_pv: self.multi_pv,
                    root_width: PONDER_ROOT_WIDTH,
                }
            };
        }
        #[cfg(not(low_priority_worker))]
        for (h, slot) in self.helpers.slots.iter().enumerate() {
            slot.assign(ponder_job!(h));
        }
        #[cfg(low_priority_worker)]
        let low_dispatched = self.helpers.dispatch(|h| ponder_job!(h));

        let mut qs = QSearch::with_histories(net, histories);
        // No clock and no ceiling: the stop flag is the whole of this search's
        // control, exactly as `go infinite` is.
        qs.set_control(SearchControl {
            stop: Some(&PONDER_STOP),
            #[cfg(feature = "verbose2")]
            node_limit: None,
            time: None,
        });
        #[cfg(feature = "verbose2")]
        qs.set_node_tally(self.node_slots, 0);
        qs.set_best_move_tally(self.bmc_slots, 0);
        qs.set_entering_king(entering_king);
        // The engine's side is the one that just moved, not the one to move at
        // this root.
        qs.set_thinking_ahead(true);
        #[cfg(feature = "verbose2")]
        qs.set_mate_mode(false);
        #[cfg(feature = "random")]
        qs.set_random(RANDOM_AMPLITUDE, self.random_seed);
        #[cfg(feature = "verbose2")]
        qs.set_multi_pv(self.multi_pv);
        // The one thing this search does differently from a `go`'s: the
        // opponent picks the move at its root, so it reads several of them to
        // full depth rather than one.
        qs.set_root_width(PONDER_ROOT_WIDTH);
        // No PV sink: the main worker of this search prints nothing either.
        // The reply this search follows is already out, so what each of its
        // workers did counts towards the next one.
        let result = qs.run_worker_on(&root, &mut root_moves, SEARCH_MAX_DEPTH);
        let stopped = Instant::now();
        #[cfg(feature = "verbose1")]
        crate::stats::add_search([&result]);

        // Whatever ended this search ends the helpers with it, and each is
        // collected before the thread returns — a slot left holding a result
        // belongs to no search. The one exception is a low-priority helper
        // that does not finish in time, which the next command must not wait
        // for: it is left to be settled by the next search that needs it.
        PONDER_STOP.store(true, Ordering::Relaxed);
        #[cfg(not(low_priority_worker))]
        for slot in self.helpers.slots {
            #[cfg(feature = "verbose1")]
            crate::stats::add_search([&slot.collect()]);
            #[cfg(not(feature = "verbose1"))]
            slot.collect();
        }
        #[cfg(low_priority_worker)]
        self.helpers.collect(low_dispatched, |result| {
            #[cfg(feature = "verbose1")]
            crate::stats::add_search([&result]);
            #[cfg(not(feature = "verbose1"))]
            let _ = result;
        });
        // The marker goes out before the flag is lowered, so a session that
        // reads "no longer thinking ahead" has already been told it stopped.
        #[cfg(feature = "verbose1")]
        sink.diagnostic(b"ponder stop");
        PONDERING.store(false, Ordering::Relaxed);
        let record = PonderRecord {
            key: root.key(),
            started,
            stopped,
            root_moves,
            best: result.best,
            nodes: result.nodes,
        };
        (qs.into_histories(), Some(record))
    }
}

/// The bundle [`Engine::go`] hands its coordinator thread — grouped
/// into one struct so [`run_coordinated`] stays a single-argument call.
struct CoordinatorJob<P: EngineSink> {
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
    /// The main worker's full control ([`STOP`] + node ceiling + deadlines).
    control: SearchControl,
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
    node_slots: &'static [TallySlot],
    /// Per-worker best-move-change counters, same slot-per-worker shape and
    /// zeroed on the same terms: each worker bumps its own slot at the root and
    /// the main worker folds them all each iteration.
    bmc_slots: &'static [TallySlot],
    /// The persistent helpers to dispatch to.
    helpers: Helpers,
    /// The active worker plan: the coordinator pins itself to worker 0's CPU and
    /// prefers that CPU's node for memory at the start of this `go`.
    worker_plan: WorkerPlan,
    /// The loaded opening book to probe once, if any.
    book: Option<&'static LoadedBook>,
    /// The loaded yorkie book, present only when one is loaded and
    /// `USI_OwnBook` is on. The game it is probed for is in [`GAME_PATH`].
    yorkie: Option<&'static YorkieBook>,
    /// `USI_OwnBook` — the master gate; when off the book is never probed.
    own_book: bool,
    /// The seed for this `go`'s book PRNG (deterministic within a session).
    book_seed: u64,
    /// What this coordinator keeps searching on once its reply is out.
    ponder: PonderJob,
    /// `limits.infinite` — hold the reply until `stop` regardless of the clock
    /// (the SKIP_SEARCH wait loop). Only a `verbose2` build can parse the
    /// clause that sets it.
    #[cfg(feature = "verbose2")]
    infinite: bool,
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
    /// What the search past the reply left, if one ran.
    ponder: Option<PonderRecord>,
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
        pos,
        #[cfg(feature = "verbose2")]
        depth,
        #[cfg(feature = "verbose2")]
        use_voting,
        control,
        histories,
        vote_buffers,
        #[cfg(feature = "verbose2")]
        node_slots,
        bmc_slots,
        helpers,
        worker_plan,
        book,
        yorkie,
        own_book,
        book_seed,
        ponder,
        #[cfg(feature = "verbose2")]
        infinite,
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
    pin_worker(&worker_plan, 0);

    // One TT generation bump per `go`, on the main worker, BEFORE any helper
    // starts, so the observable single-thread sequence is the reference's:
    // bump, then search.
    TranspositionTable::shared().new_search();

    // Build the root-move list once (the reference `start_thinking`). The resign
    // and declaration-win short-circuits emit and return before any helper is
    // dispatched, exactly as `start_searching` exits before `threads.start_searching()`.
    let root_moves = generate_root_moves(&pos);
    if root_moves.is_empty() {
        sink.reply(
            Reply::Resign,
            // Nothing was searched and no rule was consulted: the position
            // simply has no move.
            #[cfg(feature = "verbose3")]
            ValueMarks::NONE,
        );
        return CoordinatedOutcome {
            histories,
            vote_buffers,
            #[cfg(feature = "verbose3")]
            nodes: 0,
            time_state: skip_search_carry(),
            ponder: None,
        };
    }

    // Rule-aware declaration shortcut. Point / `None` rules yield
    // `Move::win()` (emitted as the bare `win` token); `TryRule`
    // yields the actual king move onto the try square, which must be emitted
    // verbatim so the host plays it. Either way the game is decided by the
    // reply, so nothing is thought past it.
    if let Some(mv) = declaration_win(&pos, &entering_king) {
        let declared = if mv == Move::win() {
            Reply::Win
        } else {
            Reply::BestMove { mv }
        };
        sink.reply(
            declared,
            // The reply *is* the declaration rule's outcome, so it carries that
            // rule's mark the way any value derived through the check does.
            #[cfg(feature = "verbose3")]
            ValueMarks {
                decl_rule: true,
                ..ValueMarks::NONE
            },
        );
        return CoordinatedOutcome {
            histories,
            vote_buffers,
            #[cfg(feature = "verbose3")]
            nodes: 0,
            time_state: skip_search_carry(),
            ponder: None,
        };
    }

    // Opening-book probe — once, on the coordinator, BEFORE any helper starts
    // (the on-the-fly read path is not thread-safe by design). The
    // `USI_OwnBook` gate and a loaded book — yorkie or `.ybb` — are both
    // required. On a hit we emit and return without searching, holding the
    // reply for `go infinite`.
    if own_book && (book.is_some() || yorkie.is_some()) {
        let mut prng = Prng::new(book_seed);
        let path = yorkie.map(|_| GAME_PATH.lock().unwrap_or_else(PoisonError::into_inner));
        let probe = yorkie
            .zip(path.as_deref())
            .map(|(book, path)| YorkieProbe { book, path });
        let probed = probe_book(
            probe.as_ref(),
            book.map_or(&[], |loaded| &loaded.books[..]),
            book.is_some_and(|loaded| loaded.ignore_book_ply),
            &pos,
            &BOOK_CONFIG,
            &mut prng,
        );
        // The path is not held through the thinking-ahead search a hit starts.
        drop(path);
        #[cfg(feature = "verbose1")]
        {
            let mut bytes = [0u8; BOOK_DIAGNOSTIC_BYTES];
            for diag in &probed.diagnostics {
                let mut out = TextWriter::new(&mut bytes);
                diag.write_message(&mut out);
                sink.diagnostic(out.as_bytes());
            }
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
                TranspositionTable::shared().hashfull(0),
                #[cfg(feature = "verbose2")]
                book_time_ms,
                #[cfg(feature = "verbose2")]
                infinite,
                #[cfg(feature = "verbose2")]
                &STOP,
            );
            // A book move is thought past like any other: the opponent's reply
            // may leave the book, and a warm table costs nothing.
            let (histories, ponder) = ponder.run(net, histories, &sink, &pos, hit.best);
            return CoordinatedOutcome {
                histories,
                vote_buffers,
                #[cfg(feature = "verbose3")]
                nodes: 0,
                time_state: skip_search_carry(),
                ponder,
            };
        }
    }

    // Dispatch a job to every helper (helper h → worker h + 1).
    macro_rules! go_job {
        ($h:expr) => {
            HelperJob {
                pos: pos.clone(),
                root_moves: root_moves.clone(),
                #[cfg(feature = "verbose2")]
                limit_depth: depth,
                stop: &STOP,
                thinking_ahead: false,
                #[cfg(feature = "verbose2")]
                node_slots,
                bmc_slots,
                index: $h + 1,
                entering_king,
                #[cfg(feature = "verbose2")]
                mate_mode,
                #[cfg(feature = "random")]
                random_seed,
                #[cfg(feature = "verbose2")]
                multi_pv,
                root_width: RootWidth::SINGLE_LINE,
            }
        };
    }
    #[cfg(not(low_priority_worker))]
    for (h, slot) in helpers.slots.iter().enumerate() {
        slot.assign(go_job!(h));
    }
    #[cfg(low_priority_worker)]
    let low_dispatched = helpers.dispatch(|h| go_job!(h));

    // The main worker is the only one given a PV sink, and only a `verbose2`
    // build has one to give. `MultiPV` rides on the same feature for a different
    // reason: it shapes the search, but the extra lines it searches are
    // reportable only through the `info` lines that feature brings. Without it
    // the root is single-line and the emission sites are not compiled at all, so
    // the search of the first line is identical in all three build shapes.
    let mut qs = QSearch::with_histories(net, histories);
    qs.set_control(control);
    #[cfg(feature = "verbose2")]
    qs.set_node_tally(node_slots, 0);
    qs.set_best_move_tally(bmc_slots, 0);
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

    // The `go infinite` hold (the SKIP_SEARCH wait loop): the reply waits for
    // `stop`, which catches the search that ended by itself — a mate found, the
    // depth ceiling reached — before one arrived.
    #[cfg(feature = "verbose2")]
    hold_reply(&STOP, infinite);

    // Signal the helpers the search is over, then wait for and collect each one.
    // They observe the shared stop at their next checkpoint and finish promptly.
    // A low-priority helper is waited for only briefly: when the scheduler is
    // holding it back, the reply goes out without its result, which then takes
    // no part in the vote below.
    STOP.store(true, Ordering::Relaxed);
    let VoteBuffers {
        mut results,
        mut votes,
    } = vote_buffers;
    // The previous search's results are dropped here, at the point their room
    // is about to be filled again.
    results.clear();
    results.push(main_result);
    #[cfg(not(low_priority_worker))]
    for slot in helpers.slots {
        results.push(slot.collect());
    }
    #[cfg(low_priority_worker)]
    helpers.collect(low_dispatched, |result| results.push(result));
    // Every worker's counts go to the interval the reply below ends.
    #[cfg(feature = "verbose1")]
    crate::stats::add_search(&results);
    #[cfg(feature = "dupstats")]
    crate::stats::add_spread(&results);

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
    let best = chosen_result.best.clone();
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
        // worker is not the main one and so never emitted its PV.
        let uci_pv_sent = results[0].uci_pv_sent && chosen == 0;
        if !uci_pv_sent || resign_by_value {
            // Reflect the chosen line back into line 0 so this re-emits the exact
            // PV the reply will play.
            if let Some(line0) = pv_lines.get_mut(0) {
                *line0 = best.clone();
            }
            let n = pv_lines.len();
            let infos = qs.build_pv_infos(&pos, &pv_lines, n, completed_depth, n, total_nodes);
            sink.pv_block(&infos);
        }
    }

    // Resigning replaces the whole reply (the reference makes the search look
    // skipped and stacks `Move::resign()`).
    let reply = if resign_by_value {
        Reply::Resign
    } else {
        Reply::BestMove { mv: best.mv }
    };

    sink.reply(
        reply,
        // The chosen worker's root move, whose score the reply plays on — a
        // resign by value included, since that decision reads the same score.
        #[cfg(feature = "verbose3")]
        best.marks,
    );

    // Consume the search (ending the `&tt` / `&net` borrows) and reclaim the main
    // worker's histories, which the thinking-ahead search below searches on with
    // — the reply is out and flushed, so nothing it does delays the host.
    // Resigning ends the game, so only a played move is thought past.
    let histories = qs.into_histories();
    let (histories, ponder) = match reply {
        Reply::BestMove { mv } => ponder.run(net, histories, &sink, &pos, mv),
        Reply::Resign | Reply::Win => (histories, None),
    };

    CoordinatedOutcome {
        histories,
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
        ponder,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::usi::UsiSink;
    use crate::{Buffer, king_shuffle, serial_session};

    /// An engine writing into a buffer nothing reads back — for the state a test
    /// drives directly rather than through a session. The caller holds
    /// [`serial_session`], since the engine's pool and flags are the process's.
    fn idle_engine() -> Engine<UsiSink<Buffer>> {
        Engine::new(UsiSink::new(Buffer::default()))
    }

    /// An empty yorkie book: a header pointing the path-dependent offset table
    /// at the end of the header, then the trailer.
    fn empty_yorkie_book() -> YorkieBook {
        let mut image = vec![0u8; 40];
        image[..8].copy_from_slice(b"YKBOOK01");
        image[8] = 2;
        image[32] = 40;
        let digest = yorkie_storage::yorkie_book::sha256(&image);
        image.extend_from_slice(&digest);
        image.extend_from_slice(b"YKBOOKED");
        YorkieBook::from_bytes(image).expect("valid empty book")
    }

    #[test]
    fn usi_own_book_off_leaves_the_yorkie_book_unprobed() {
        let book = empty_yorkie_book();
        let probed = yorkie_for_go(true, Some(&book), |_| {});
        assert!(probed.is_some(), "on with a book loaded → probed");
        let probed = yorkie_for_go(false, Some(&book), |_| {
            panic!("no game path is built for a book that is not read")
        });
        assert!(probed.is_none(), "USI_OwnBook off → not probed");
        assert!(yorkie_for_go(true, None, |_| {}).is_none());
    }

    /// A yorkie-book list of `weights`, the files named by their index.
    fn weighted(weights: &[u32]) -> Vec<YorkieBookChoice> {
        weights
            .iter()
            .enumerate()
            .map(|(i, &weight)| YorkieBookChoice {
                name: PathBuf::from(i.to_string()),
                path: PathBuf::from(i.to_string()),
                weight,
            })
            .collect()
    }

    /// With the seed pinned, the draws are one fixed sequence.
    #[test]
    fn a_pinned_seed_draws_a_fixed_sequence() {
        let choices = weighted(&[70, 30]);
        let mut prng = Prng::new(TEST_DRAW_SEED ^ YORKIE_DRAW_SEED_MIX);
        let drawn: Vec<usize> = (0..16)
            .map(|_| draw_weighted(&choices, &mut prng))
            .collect();
        assert_eq!(
            drawn,
            [0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            "the sequence a pinned seed draws"
        );
    }

    /// The seed `a_pinned_seed_draws_a_fixed_sequence` pins.
    const TEST_DRAW_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

    /// Over many draws each entry comes up in proportion to its weight, and a
    /// one-entry list always draws that entry, whatever its weight.
    #[test]
    fn draws_follow_the_weights() {
        const DRAWS: usize = 10_000;
        let choices = weighted(&[70, 30]);
        let mut prng = Prng::new(TEST_DRAW_SEED);
        let mut counts = [0usize; 2];
        for _ in 0..DRAWS {
            counts[draw_weighted(&choices, &mut prng)] += 1;
        }
        assert_eq!(counts[0] + counts[1], DRAWS);
        // Three standard deviations of a 70/30 split over 10,000 draws is
        // about 137.
        assert!(
            counts[0].abs_diff(7_000) <= 150,
            "70/30 drew {counts:?} over {DRAWS}"
        );

        let choices = weighted(&[3]);
        let mut prng = Prng::new(TEST_DRAW_SEED);
        assert!((0..100).all(|_| draw_weighted(&choices, &mut prng) == 0));
    }

    /// The game path a `go` reads.
    fn game_path(engine: &Engine<UsiSink<Buffer>>) -> Vec<PackedSfen> {
        let mut path = Vec::new();
        engine.fill_game_path(&mut path);
        path
    }

    /// The game path is every position the last `position` command passed
    /// through, the current one last; a session with no `position` has the
    /// initial position alone. The commands of a whole game are recorded in
    /// the room reserved when the book was loaded.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn the_game_path_follows_the_position_command() {
        let _session = serial_session();
        let mut engine = idle_engine();
        engine.yorkie_books = Some((
            vec![PathBuf::from("book.ykb")],
            Vec::leak(vec![empty_yorkie_book()]),
        ));
        engine.reserve_game_path();
        assert_eq!(game_path(&engine), vec![sfen_pack(&Position::startpos())]);

        engine
            .set_position(PositionSfen::StartPos, b"7g7f 3c3d")
            .expect("both moves are legal");
        let path = game_path(&engine);
        assert_eq!(path.len(), 3);
        assert_eq!(path[0], sfen_pack(&Position::startpos()));
        assert_eq!(path[2], sfen_pack(&engine.pos));

        assert_eq!(
            engine
                .set_position(PositionSfen::StartPos, b"7g7f 9z9z")
                .err(),
            Some(PositionRefusal::IllegalMove(b"9z9z")),
        );
        assert_eq!(
            game_path(&engine),
            path,
            "a refused command records nothing"
        );

        for retained in [&engine.last_position, &engine.pending_position] {
            assert_eq!(retained.path.capacity(), GAME_PATH_CAPACITY);
        }
    }

    /// A game path of `moves` moves of the four-move preamble's king shuffle
    /// and on: the starting position, then the position after each move.
    fn path_of(moves: usize) -> Vec<PackedSfen> {
        const MOVES: [&[u8]; 6] = [b"5i5h", b"5a5b", b"5h5i", b"5b5a", b"7g7f", b"3c3d"];
        let mut pos = Position::startpos();
        let mut path = vec![sfen_pack(&pos)];
        for mv in &MOVES[..moves] {
            pos.do_move(parse_usi_move(mv, &pos).expect("legal"));
            path.push(sfen_pack(&pos));
        }
        path
    }

    /// Leaving out no moves keeps the whole path.
    #[test]
    fn skipping_no_moves_keeps_the_whole_path() {
        let path = path_of(6);
        assert_eq!(seen_positions(&path, 0), &path[..]);
        let start = path_of(0);
        assert_eq!(seen_positions(&start, 0), &start[..]);
    }

    /// With more moves than are left out, the path starts at the position
    /// after the last move left out.
    #[test]
    fn skipping_fewer_moves_than_the_command_has_starts_after_the_last_skipped() {
        let path = path_of(6);
        assert_eq!(seen_positions(&path, 4), &path[4..]);
        assert_eq!(seen_positions(&path, 4).len(), 3);
        assert_eq!(seen_positions(&path, 1), &path[1..]);
    }

    /// Leaving out exactly as many moves as the command has keeps the current
    /// position alone.
    #[test]
    fn skipping_as_many_moves_as_the_command_has_keeps_the_current_position() {
        let path = path_of(4);
        assert_eq!(seen_positions(&path, 4), &path[4..]);
        assert_eq!(seen_positions(&path, 4), &[path[4]]);
    }

    /// Leaving out more moves than the command has keeps the current position
    /// alone too, and an empty path stays empty.
    #[test]
    fn skipping_more_moves_than_the_command_has_keeps_the_current_position() {
        for moves in 0..4 {
            let path = path_of(moves);
            assert_eq!(seen_positions(&path, 4), &[path[moves]], "{moves} moves");
        }
        assert_eq!(seen_positions(&[], 4), &[] as &[PackedSfen]);
        assert_eq!(seen_positions(&[], 0), &[] as &[PackedSfen]);
    }

    /// The engine trims the path it hands a `go` by the count it was given.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn the_game_path_leaves_out_the_skipped_moves() {
        let _session = serial_session();
        let mut engine = idle_engine();
        engine.yorkie_books = Some((
            vec![PathBuf::from("book.ykb")],
            Vec::leak(vec![empty_yorkie_book()]),
        ));
        engine.reserve_game_path();
        engine.set_yorkie_book_path_skip_moves(4);
        engine
            .set_position(PositionSfen::StartPos, b"5i5h 5a5b 5h5i 5b5a 7g7f 3c3d")
            .expect("every move is legal");
        assert_eq!(game_path(&engine), path_of(6)[4..].to_vec());
        engine
            .set_position(PositionSfen::StartPos, b"5i5h 5a5b")
            .expect("both moves are legal");
        assert_eq!(game_path(&engine), vec![sfen_pack(&engine.pos)]);
    }

    /// Without a yorkie book nothing is recorded, and no room is taken for it.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn no_game_path_is_recorded_without_a_yorkie_book() {
        let _session = serial_session();
        let mut engine = idle_engine();
        engine
            .set_position(PositionSfen::StartPos, b"7g7f 3c3d")
            .expect("both moves are legal");
        assert!(engine.last_position.path.is_empty());
        assert_eq!(engine.last_position.path.capacity(), 0);
    }

    /// A record of a 2000 ms search past `7g7f` whose main worker spent 300 of
    /// its 1000 nodes below `3c3d` and none below any other reply. Its `best` is
    /// the copy a completed iteration leaves, of a move still in the list.
    fn record_past_7g7f() -> PonderRecord {
        let mut root = Position::startpos();
        root.do_move(parse_usi_move(b"7g7f", &root).expect("legal"));
        let reply = parse_usi_move(b"3c3d", &root).expect("legal");
        let mut root_moves = generate_root_moves(&root);
        for rm in &mut root_moves {
            rm.effort = if rm.mv == reply { 300 } else { 0 };
        }
        let best = root_moves
            .iter()
            .find(|rm| rm.mv != reply)
            .expect("more than one reply")
            .clone();
        let started = Instant::now();
        PonderRecord {
            key: root.key(),
            started,
            stopped: started + std::time::Duration::from_millis(2000),
            root_moves,
            best,
            nodes: 1000,
        }
    }

    /// What the record above spent on `3c3d`: 2000 ms × 300 / 1000.
    const SPENT_ON_3C3D: i64 = 600;

    /// The time spent on a reply is the search's time in the share of its nodes
    /// the reply took, whether the reply is still in the list or was moved out
    /// of it as the best move so far.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn a_reply_takes_its_share_of_the_search_past_the_move() {
        let record = record_past_7g7f();
        let mut root = Position::startpos();
        root.do_move(parse_usi_move(b"7g7f", &root).expect("legal"));
        let reply = parse_usi_move(b"3c3d", &root).expect("legal");
        let unsearched = parse_usi_move(b"8c8d", &root).expect("legal");
        assert_eq!(record.spent_on(reply), SPENT_ON_3C3D);
        assert_eq!(record.spent_on(unsearched), 0, "no effort, no time");

        let mut moved_out = record_past_7g7f();
        let at = moved_out
            .root_moves
            .iter()
            .position(|rm| rm.mv == reply)
            .expect("3c3d is a root move");
        moved_out.best = moved_out.root_moves.swap_remove(at);
        assert_eq!(moved_out.spent_on(reply), SPENT_ON_3C3D);
    }

    /// Only the root plus one reply is credited, a refused command leaves the
    /// standing position's credit, and a new game discards what the last one
    /// left.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn only_the_root_plus_one_reply_is_credited() {
        let _session = serial_session();
        let mut engine = idle_engine();
        engine.ponder_record = Some(record_past_7g7f());

        for (moves, pondered) in [
            (&b"7g7f 3c3d"[..], SPENT_ON_3C3D),
            (b"7g7f 8c8d", 0),
            (b"7g7f", 0),
            (b"7g7f 3c3d 2g2f", 0),
            (b"2g2f 3c3d", 0),
            (b"7g7f 3c3d", SPENT_ON_3C3D),
        ] {
            engine
                .set_position(PositionSfen::StartPos, moves)
                .expect("legal moves");
            assert_eq!(
                engine.pondered_ms,
                pondered,
                "position startpos moves {}",
                String::from_utf8_lossy(moves)
            );
        }

        let _ = engine.set_position(PositionSfen::StartPos, b"7g7f 9z9z");
        assert_eq!(engine.pondered_ms, SPENT_ON_3C3D);

        engine.new_game().expect("a new game starts");
        assert!(engine.ponder_record.is_none());
        assert_eq!(engine.pondered_ms, 0);
        engine
            .set_position(PositionSfen::StartPos, b"7g7f 3c3d")
            .expect("legal moves");
        assert_eq!(
            engine.pondered_ms, 0,
            "the last game's search counts for nothing"
        );
    }

    /// What a refused command must leave behind, read off the engine itself: the
    /// position of the last accepted one, and that command still retained in the
    /// buffers the next one is replayed into.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn a_refused_position_leaves_the_accepted_one_in_place() {
        let _session = serial_session();
        let mut engine = idle_engine();
        engine
            .set_position(PositionSfen::StartPos, b"7g7f 3c3d")
            .expect("both moves are legal");
        let accepted_ply = engine.pos.ply();

        let past_the_bound = king_shuffle(MAX_POSITION_MOVES + 1);
        for refused in [
            &b"7g7f 1a1b"[..],         // an illegal move mid-list
            past_the_bound.as_bytes(), // past the bound
        ] {
            let _ = engine.set_position(PositionSfen::StartPos, refused);
            assert_eq!(engine.pos.ply(), accepted_ply);
            assert_eq!(engine.last_position.moves.len(), 2);
        }

        // A malformed SFEN is refused before any move is looked at.
        let _ = engine.set_position(PositionSfen::Sfen([b"not-a-board", b"b", b"-", b"1"]), b"");
        assert_eq!(engine.pos.ply(), accepted_ply);
        assert_eq!(engine.last_position.moves.len(), 2);
    }

    /// A game's position commands are written into the buffers the engine was
    /// built with: whatever the game's length, none of them is regrown, which is
    /// what keeps the command path away from the allocator.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn a_games_position_commands_reuse_the_buffers() {
        let _session = serial_session();
        let mut engine = idle_engine();
        let sfen: [&[u8]; 4] = [
            b"lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL",
            b"b",
            b"-",
            b"1",
        ];
        for plies in 0..64 {
            let moves = king_shuffle(plies);
            engine
                .set_position(PositionSfen::StartPos, moves.as_bytes())
                .expect("a king shuffle is legal");
            engine
                .set_position(PositionSfen::Sfen(sfen), moves.as_bytes())
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

    /// A refusal's message as a string, so the assertions below read as text.
    fn refusal_message(refusal: &LayoutRefusal) -> String {
        let mut bytes = [0u8; NOTICE_BYTES];
        let mut out = TextWriter::new(&mut bytes);
        refusal.write_message(&mut out);
        String::from_utf8(out.as_bytes().to_vec()).expect("a message is ASCII")
    }

    /// A plan pinning one worker to each of `cpus`, on `nodes`, which must be
    /// the nodes `built` puts those CPUs on.
    fn plan_over(
        cpus: &'static [usize],
        nodes: &'static [NumaIndex],
        built: &NumaLayout,
    ) -> WorkerPlan {
        for (&cpu, &node) in cpus.iter().zip(nodes) {
            assert_eq!(built.system_node_of_cpu(cpu), Some(node), "CPU {cpu}");
        }
        WorkerPlan::of(cpus, nodes, cpus.len())
    }

    #[test]
    fn a_machine_matching_the_compiled_layout_is_no_difference() {
        let built = layout(&[&[0, 1], &[2, 3]]);
        let plan = plan_over(&[0, 2], &[0, 1], &built);
        assert_eq!(
            machine_refusal(&built, &affinity(&[0, 1, 2, 3]), &plan, &built),
            None
        );
    }

    #[test]
    fn a_machine_with_another_node_count_is_reported_with_both_counts() {
        let built = layout(&[&[0, 1], &[2, 3]]);
        let plan = plan_over(&[0, 2], &[0, 1], &built);
        let live = layout(&[&[0, 1, 2, 3]]);
        let msg = machine_refusal(&live, &affinity(&[0, 1, 2, 3]), &plan, &built)
            .expect("one node is not two");
        let msg = refusal_message(&msg);
        assert!(msg.contains("built for 2 NUMA node(s)"), "message: {msg}");
        assert!(msg.contains("this host has 1"), "message: {msg}");
    }

    #[test]
    fn a_node_holding_other_cpus_is_reported_with_both_cpu_lists() {
        let built = layout(&[&[0, 1], &[2, 3]]);
        let plan = plan_over(&[0, 2], &[0, 1], &built);
        let live = layout(&[&[0, 1], &[2, 3, 4]]);
        let msg = machine_refusal(&live, &affinity(&[0, 1, 2, 3, 4]), &plan, &built)
            .expect("node 1 grew a CPU");
        let msg = refusal_message(&msg);
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
        let plan = plan_over(&[0, 2, 3], &[0, 1, 1], &built);
        let msg = machine_refusal(&built, &affinity(&[0, 1]), &plan, &built)
            .expect("two of the plan's CPUs are hidden");
        let msg = refusal_message(&msg);
        assert!(msg.contains("may not run on CPUs 2-3"), "message: {msg}");
        assert!(
            msg.contains("which its workers are pinned to"),
            "message: {msg}"
        );

        let msg = machine_refusal(&built, &affinity(&[0, 1, 2]), &plan, &built)
            .expect("one of the plan's CPUs is hidden");
        let msg = refusal_message(&msg);
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
        let plan = plan_over(&[0, 1], &[0, 0], &built);
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
        let plan = plan_over(&[0, 2], &[0, 1], &built);
        let live = NumaLayout::from_const(&[&[0, 1], &[2, 3]], &[1, 0]);
        let msg = machine_refusal(&live, &affinity(&[0, 1, 2, 3]), &plan, &built)
            .expect("CPU 0 is on the other node now");
        let msg = refusal_message(&msg);
        assert!(msg.contains("worker 0 runs on CPU 0"), "message: {msg}");
        assert!(msg.contains("node 1 on this host"), "message: {msg}");
        assert!(msg.contains("node 0 in the layout"), "message: {msg}");
    }

    #[cfg(feature = "verbose3")]
    #[test]
    fn info_strings_exact_formats() {
        let rendered = |threads: usize, plan: &WorkerPlan| {
            let mut bytes = [0u8; NOTICE_BYTES];
            let mut out = TextWriter::new(&mut bytes);
            write_thread_allocation_information(threads, plan, &mut out);
            out.as_bytes().to_vec()
        };
        // Singular and plural, and the CPU list in the shortened form.
        let one = WorkerPlan::of(&[5], &[0], 1);
        assert_eq!(rendered(1, &one), b"Using 1 thread on CPUs 5");
        let four = WorkerPlan::of(&[0, 1, 2, 8], &[0, 0, 0, 1], 4);
        assert_eq!(rendered(4, &four), b"Using 4 threads on CPUs 0-2,8");
    }

    #[test]
    fn thread_pool_new_sizes_to_main_plus_helpers() {
        let _session = serial_session();
        let pool = ThreadPool::new(4);
        assert_eq!(pool.size(), 4, "4 slots = 1 main + 3 helpers");
        drop(pool);
        let single = ThreadPool::new(1);
        assert_eq!(single.size(), 1, "1 slot = main only, no helpers");
    }

    #[test]
    fn thread_pool_set_rebuilds_without_leaking() {
        // Each `set` joins the previous generation of helpers before spawning
        // the next, so cycling sizes never leaks OS threads. We can only assert
        // the resulting slot count here; the no-leak property is what the join
        // in `shutdown` guarantees.
        let _session = serial_session();
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
    fn a_go_is_handed_the_static_helper_slots() {
        // Nothing about dispatching to the helpers builds a list: every `go`
        // gets the session's own slots, as many as the pool has helpers.
        let _session = serial_session();
        let mut pool = ThreadPool::new(3);
        let first = pool.helper_slots();
        assert_eq!(first.len(), 2);
        assert!(std::ptr::eq(first.as_ptr(), HELPER_SLOTS.as_ptr()));
        pool.set(2);
        assert_eq!(pool.helper_slots().len(), 1, "a rebuilt pool's helpers");
        assert!(std::ptr::eq(
            pool.helper_slots().as_ptr(),
            HELPER_SLOTS.as_ptr()
        ));
    }

    // --- the flags, counters and buffers a `go` reuses --------------------

    #[test]
    fn arming_a_search_clears_what_the_last_one_left() {
        let _session = serial_session();
        STOP.store(true, Ordering::Relaxed);
        PONDER_STOP.store(true, Ordering::Relaxed);
        PONDERING.store(true, Ordering::Relaxed);
        bmc_tally(3)[2].store(9, Ordering::Relaxed);
        #[cfg(feature = "verbose2")]
        node_tally(3)[1].store(9, Ordering::Relaxed);

        arm_search(3);

        assert!(!STOP.load(Ordering::Relaxed));
        assert!(!PONDER_STOP.load(Ordering::Relaxed));
        assert!(!PONDERING.load(Ordering::Relaxed));
        assert!(
            bmc_tally(3).iter().all(|s| s.load(Ordering::Relaxed) == 0),
            "every best-move-change counter starts at zero"
        );
        #[cfg(feature = "verbose2")]
        assert!(
            node_tally(3).iter().all(|s| s.load(Ordering::Relaxed) == 0),
            "every node counter starts at zero"
        );
    }

    #[test]
    fn a_pool_hands_its_workers_one_counter_each() {
        assert_eq!(bmc_tally(1).len(), 1);
        assert_eq!(bmc_tally(POOL_CAPACITY).len(), POOL_CAPACITY);
        #[cfg(feature = "verbose2")]
        assert_eq!(node_tally(4).len(), 4);
        assert_eq!(HELPER_SLOTS.len(), POOL_CAPACITY - 1);
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
        let _session = serial_session();
        let pool = ThreadPool::new(0);
        assert_eq!(pool.size(), 1);
    }

    // --- shared-history node mapping --------------------------------------

    #[test]
    fn shared_node_counts_are_the_workers_per_node() {
        let c = shared_node_counts([0, 1, 0, 1, 0]);
        assert_eq!(c[&0], 3);
        assert_eq!(c[&1], 2);

        // Every worker on one node: one set, sized for all of them.
        let c = shared_node_counts([2, 2, 2]);
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
        assert_eq!(plan.cpus().collect::<Vec<_>>(), vec![4, 9, 4, 9, 4]);
        assert_eq!(plan.system_nodes().collect::<Vec<_>>(), vec![0, 1, 0, 1, 0]);
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
        let _session = serial_session();
        let mut engine = idle_engine();
        engine.rebuild_pool();

        let node = engine.worker_plan.system_node(0);
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

        let _session = serial_session();
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
            ponder: None,
        });
        engine.search = Some(ActiveSearch {
            handle,
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
        let _session = serial_session();
        let mut engine = idle_engine();
        engine.previous_time_reduction = 0.42;

        let handle = std::thread::spawn(|| SearchState {
            histories: WorkerHistories::new(),
            time_state: Some((10, 20, Some(1.25))),
            vote_buffers: VoteBuffers::for_pool(1),
            ponder: None,
        });
        engine.search = Some(ActiveSearch {
            handle,
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
        for node in plan.system_nodes() {
            assert!(config::EVAL_REGION_NODES.contains(&node));
        }
    }

    // -- Multiple Book name resolution ------------------------------------

    /// A fresh empty directory under `$TMPDIR`; the caller removes it.
    fn book_name_fixture_dir(tag: &str) -> PathBuf {
        use std::sync::atomic::AtomicU64;

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
        assert_eq!(notices, vec![dir.join("user_book1-000.ybb")]);

        // `.db` base: primary `.db` wins over a co-existing `.ybb`.
        let db_base = dir.join("user_book2.db");
        touch(&dir.join("user_book2-000.ybb"));
        touch(&dir.join("user_book2-000.db"));
        let (names, notices) = book_names(&db_base);
        assert_eq!(
            file_names(&names),
            vec!["user_book2-000.db", "user_book2.db"]
        );
        assert_eq!(notices, vec![dir.join("user_book2-000.db")]);

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

    /// The low-priority worker: its priority, and the stop path that must not
    /// wait for it.
    #[cfg(low_priority_worker)]
    mod low_priority {
        use super::*;

        use std::sync::mpsc;

        /// Scheduling noise a timing assertion allows on top of the bound it
        /// checks: the test machine runs other tests beside this one.
        const SLACK: Duration = Duration::from_millis(50);

        /// A pool of `size` workers over this process's own CPUs, its last
        /// worker at low priority, with every helper's thread id cleared first.
        fn pool_with_low_priority_last(size: usize) -> ThreadPool {
            for tid in &HELPER_TIDS[..size - 1] {
                tid.store(0, Ordering::Relaxed);
            }
            let plan = WorkerPlan::of_allowed_cpus(size).with_low_priority(Some(size - 1));
            ThreadPool::with_binding(size, plan, &[])
        }

        /// Helper `h`'s kernel thread id, once the thread has written it.
        fn helper_tid(h: usize) -> yorkie_numa::ThreadId {
            loop {
                let tid = HELPER_TIDS[h].load(Ordering::Acquire);
                if tid != 0 {
                    return tid;
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        }

        fn own_nice() -> i32 {
            yorkie_numa::thread_nice(yorkie_numa::current_thread_id())
                .expect("the kernel reports this thread's priority")
        }

        /// A job for helper 1 of a two-worker search that no helper will run.
        fn idle_job() -> HelperJob {
            let pos = Position::startpos();
            HelperJob {
                entering_king: EnteringKingConfig::new(&pos),
                pos,
                root_moves: Vec::new(),
                #[cfg(feature = "verbose2")]
                limit_depth: SEARCH_MAX_DEPTH,
                stop: &STOP,
                thinking_ahead: false,
                #[cfg(feature = "verbose2")]
                node_slots: node_tally(2),
                bmc_slots: bmc_tally(2),
                index: 1,
                #[cfg(feature = "verbose2")]
                mate_mode: false,
                #[cfg(feature = "random")]
                random_seed: 0,
                #[cfg(feature = "verbose2")]
                multi_pv: 1,
                root_width: RootWidth::SINGLE_LINE,
            }
        }

        /// The first helper slot, seen as the low-priority helper of a
        /// two-worker search. No pool is live while a test holds the session.
        fn lone_low_helper() -> Helpers {
            Helpers {
                slots: &HELPER_SLOTS[..1],
                low: Some(0),
            }
        }

        #[cfg_attr(miri, ignore)]
        #[test]
        fn the_second_of_two_workers_runs_at_the_lowest_priority() {
            let _session = serial_session();
            let own = own_nice();
            let pool = pool_with_low_priority_last(2);
            assert_eq!(
                yorkie_numa::thread_nice(helper_tid(0)),
                Some(yorkie_numa::LOWEST_NICE),
                "worker 1 reads back the lowered priority"
            );
            assert_eq!(low_priority_refusal(), None, "the lowering was not refused");
            // The main worker is the per-`go` coordinator: a thread the command
            // thread spawns, which pins itself and lowers nothing.
            let plan = WorkerPlan::of_allowed_cpus(2).with_low_priority(Some(1));
            let main = std::thread::spawn(move || {
                pin_worker(&plan, 0);
                own_nice()
            })
            .join()
            .expect("the main worker's thread runs");
            assert_eq!(main, own, "the main worker keeps the session's priority");
            assert_eq!(own_nice(), own, "the command thread keeps its priority");
            drop(pool);
        }

        #[cfg_attr(miri, ignore)]
        #[test]
        fn only_the_last_worker_is_lowered() {
            let _session = serial_session();
            let own = own_nice();
            let pool = pool_with_low_priority_last(3);
            assert_eq!(yorkie_numa::thread_nice(helper_tid(0)), Some(own));
            assert_eq!(
                yorkie_numa::thread_nice(helper_tid(1)),
                Some(yorkie_numa::LOWEST_NICE)
            );
            assert_eq!(pool.helpers().low, Some(1), "the last helper's slot");
            drop(pool);
        }

        #[test]
        fn a_plan_never_lowers_the_main_worker() {
            let plan = WorkerPlan::of(&[0, 1, 2], &[0, 0, 0], 3);
            assert_eq!(
                plan.with_low_priority(Some(2)).low_priority_worker(),
                Some(2)
            );
            assert_eq!(plan.with_low_priority(None).low_priority_worker(), None);
            // A worker the plan does not cover has nothing to lower.
            let two = WorkerPlan::of(&[0, 1, 2], &[0, 0, 0], 2);
            assert_eq!(two.with_low_priority(Some(2)).low_priority_worker(), None);
        }

        #[test]
        fn the_cpu_line_names_the_low_priority_worker() {
            let rendered = |plan: &WorkerPlan| {
                let mut bytes = [0u8; NOTICE_BYTES];
                let mut out = TextWriter::new(&mut bytes);
                write_low_priority_suffix(plan, &mut out);
                out.as_bytes().to_vec()
            };
            let pinned = WorkerPlan::of(&[1, 2, 0], &[0, 0, 0], 3);
            assert_eq!(
                rendered(&pinned.with_low_priority(Some(2))),
                b", low priority on CPU 0"
            );
            let unpinned = WorkerPlan::of(&[], &[0, 0, 0], 3);
            assert_eq!(
                rendered(&unpinned.with_low_priority(Some(2))),
                b", low priority on worker 2"
            );
            assert_eq!(rendered(&pinned), b"", "nothing without one");
        }

        #[cfg_attr(miri, ignore)]
        #[test]
        fn a_helper_still_searching_does_not_hold_the_reply() {
            let _session = serial_session();
            let slot = &HELPER_SLOTS[0];
            // Dispatched, and then kept off its CPU: it never finishes.
            *slot.lock() = SlotState::Running;
            LOW_STOP.store(false, Ordering::Relaxed);
            let start = Instant::now();
            let mut taken = 0;
            lone_low_helper().collect(true, |_| taken += 1);
            let waited = start.elapsed();
            assert_eq!(taken, 0, "the reply goes out without its result");
            assert!(waited >= LATE_HELPER_WAIT, "waited {waited:?}");
            assert!(waited < LATE_HELPER_WAIT + SLACK, "waited {waited:?}");
            assert!(LOW_STOP.load(Ordering::Relaxed), "it is told to stop");
            assert!(matches!(*slot.lock(), SlotState::Running));
            // The next search neither waits for it nor hands it a job.
            assert!(!lone_low_helper().dispatch(|_| unreachable!("no job is built")));
            assert!(LOW_STOP.load(Ordering::Relaxed), "its flag stays raised");
            *slot.lock() = SlotState::Parked;
        }

        #[cfg_attr(miri, ignore)]
        #[test]
        fn a_helper_holding_its_lock_does_not_hold_the_reply() {
            let _session = serial_session();
            let slot = &HELPER_SLOTS[0];
            *slot.lock() = SlotState::Running;
            // A helper descheduled inside its slot's critical section.
            let (held_tx, held_rx) = mpsc::channel();
            let (release_tx, release_rx) = mpsc::channel::<()>();
            let holder = std::thread::spawn(move || {
                let _guard = slot.lock();
                held_tx.send(()).expect("the test waits for this");
                let _ = release_rx.recv();
            });
            held_rx.recv().expect("the holder takes the lock");
            let start = Instant::now();
            assert!(slot.collect_by(start + LATE_HELPER_WAIT).is_none());
            let waited = start.elapsed();
            assert!(waited < LATE_HELPER_WAIT + SLACK, "waited {waited:?}");
            assert!(
                !slot.try_assign(|| unreachable!("no job is built")),
                "dispatch does not wait on the lock either"
            );
            release_tx.send(()).expect("the holder waits for this");
            holder.join().expect("the holder runs");
            *slot.lock() = SlotState::Parked;
        }

        #[cfg_attr(miri, ignore)]
        #[test]
        fn a_job_the_helper_never_picked_up_is_taken_back() {
            let _session = serial_session();
            let slot = &HELPER_SLOTS[0];
            *slot.lock() = SlotState::Parked;
            LOW_STOP.store(true, Ordering::Relaxed);
            // No thread parks on the slot, so nothing picks the job up.
            assert!(lone_low_helper().dispatch(|_| idle_job()));
            assert!(
                !LOW_STOP.load(Ordering::Relaxed),
                "a dispatched helper may search"
            );
            match &*slot.lock() {
                SlotState::Assigned(job) => {
                    assert!(std::ptr::eq(job.stop, &LOW_STOP), "it polls its own flag");
                }
                _ => panic!("the job is posted"),
            }
            let start = Instant::now();
            let mut taken = 0;
            lone_low_helper().collect(true, |_| taken += 1);
            assert!(start.elapsed() < LATE_HELPER_WAIT + SLACK);
            assert_eq!(taken, 0);
            assert!(
                matches!(*slot.lock(), SlotState::Parked),
                "the job is taken back, so it never starts a search that has ended"
            );
        }

        #[cfg_attr(miri, ignore)]
        #[test]
        fn a_new_game_waits_for_a_late_helper() {
            let _session = serial_session();
            let slot = &HELPER_SLOTS[0];
            *slot.lock() = SlotState::Running;
            let finisher = std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(20));
                // What the helper's loop does once its search returns, less
                // the result.
                *slot.lock() = SlotState::Parked;
                slot.cv.notify_all();
            });
            let start = Instant::now();
            slot.settle();
            assert!(start.elapsed() >= Duration::from_millis(20));
            assert!(matches!(*slot.lock(), SlotState::Parked));
            finisher.join().expect("the finisher runs");
        }
    }
}
