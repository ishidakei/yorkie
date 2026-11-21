//! Quiescence search, ported from `Search::YaneuraOuWorker::qsearch`.
//!
//! Everything here follows the reference's **code**, not the stale design note
//! above that function, which describes a `DEPTH_QS_CHECKS` /
//! `DEPTH_QS_RECAPTURES` design the code does not implement.
//!
//! The NNUE accumulator is derived eagerly on every `do_move` rather than
//! lazily on first evaluation as the reference does; both reach the same
//! accumulator. qsearch reads the worker history tables and never writes them,
//! so no history update occurs inside a pure-qsearch subtree.
//!
//! The reference holds one `TTWriter` from the Step-3 probe and writes through
//! it, including the *tail* write after the whole move loop. In safe Rust a
//! writer borrowing the table cannot outlive the recursive calls that also
//! mutate it, so the node captures the entry's *location* at Step 3 and every
//! write site targets that slot. Re-probing instead would re-run the
//! replacement selection against a cluster a child has since churned, and could
//! pick a different slot.

use std::num::{NonZeroI32, NonZeroU16, NonZeroU32};
// The node ceiling and the PV-output interval: both are `verbose2`, so a build
// without that feature names neither type.
#[cfg(feature = "verbose2")]
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
#[cfg(feature = "verbose2")]
use std::time::Duration;
use std::time::Instant;

use yorkie_eval::{Accumulator, FinnyCache, MoveDelta, NetworkParams, evaluate_with};
use yorkie_state::{Color, Move, Piece, PieceKind, Position, RepetitionState, piece_value};
#[cfg(feature = "verbose3")]
use yorkie_storage::ValueMarks;

#[cfg(feature = "verbose3")]
use crate::causes::PathCauses;

/// A node's marks and their causes, set aside across a sub-search that
/// overwrites the live ones.
#[cfg(feature = "verbose3")]
#[derive(Clone, Copy)]
struct Held {
    marks: ValueMarks,
    causes: PathCauses,
}
use yorkie_storage::{Bound, TranspositionTable, TtSlot, Value};

#[cfg(feature = "verbose2")]
use crate::config::GENERATE_ALL_LEGAL_MOVES;
use crate::history::{
    ContPlane, ContinuationCorrectionHistory, ContinuationHistory, CorrChannel, CorrPlane,
};
use crate::movepick::{MovePicker, PickerScratch};
use crate::params;
use crate::root::{
    EnteringKingConfig, RootKind, RootMove, RootOutcome, declaration_win, generate_root_moves,
};
use crate::timeman::TimeManagement;
use crate::update::{
    SEARCHED_LIST_CAPACITY, SearchStackCell, SearchedList, WorkerHistories, update_all_stats,
    update_continuation_histories, update_correction_history, update_quiet_histories,
};

// Value / depth constants and helpers.

/// `MAX_PLY`: the standard engine build value.
const MAX_PLY: i32 = 246;
/// Whether a forced mate the engine delivers is scored [`VALUE_MATE_FLAT`]
/// rather than by its distance (the `flatten_winning_mate` setting).
///
/// The reference scores every mate by distance, so a shorter one is always
/// worth more and the search keeps hunting for it after one is proven. With
/// this on, the engine's own wins are all worth the same, while a mate against
/// the engine keeps the gradient, so the longest defence is still preferred.
const FLATTEN: bool = crate::config::FLATTEN_WINNING_MATE;
/// `VALUE_MATE`.
const VALUE_MATE: Value = 32000;
/// The single score of a forced mate the engine delivers, in a build with the
/// `flatten_winning_mate` setting on: above the whole `mate_in` gradient, so no
/// distance-scored mate outranks it, and below [`VALUE_INFINITE`].
pub const VALUE_MATE_FLAT: Value = VALUE_MATE + 1;
/// `VALUE_INFINITE`: the reference's, moved up one in a build with the
/// `flatten_winning_mate` setting on, to make room for [`VALUE_MATE_FLAT`].
pub const VALUE_INFINITE: Value = infinite_for(FLATTEN);
/// `VALUE_NONE`: one above [`VALUE_INFINITE`], in either build.
const VALUE_NONE: Value = none_for(FLATTEN);
// The table's miss sentinel is the search's, and the search's orderings all
// hold in both builds.
const _: () = assert!(VALUE_NONE == yorkie_storage::VALUE_NONE);
const _: () = assert!(VALUE_MATE < VALUE_MATE_FLAT && VALUE_MATE_FLAT <= VALUE_INFINITE);
const _: () = assert!(!FLATTEN || VALUE_MATE_FLAT < VALUE_INFINITE);

/// [`VALUE_INFINITE`] in a build with or without [`FLATTEN`].
const fn infinite_for(flat: bool) -> Value {
    if flat {
        VALUE_MATE_FLAT + 1
    } else {
        VALUE_MATE + 1
    }
}

/// [`VALUE_NONE`] in a build with or without [`FLATTEN`].
const fn none_for(flat: bool) -> Value {
    infinite_for(flat) + 1
}
/// `VALUE_MATE_IN_MAX_PLY` == `VALUE_TB_WIN_IN_MAX_PLY`.
const VALUE_TB_WIN_IN_MAX_PLY: Value = VALUE_MATE - MAX_PLY; // 31754
/// `VALUE_MAX_EVAL` == `VALUE_SUPERIOR`.
const VALUE_MAX_EVAL: Value = VALUE_TB_WIN_IN_MAX_PLY - 1; // 31753
/// `VALUE_DRAW`.
const VALUE_DRAW: Value = 0;
/// The `rootDelta` a search that has not opened an aspiration window reduces
/// against: the full window's width, as [`window_width`] measures it.
const ROOT_DELTA_FULL: NonZeroU32 =
    NonZeroU32::new(window_width(-VALUE_INFINITE, VALUE_INFINITE) as u32)
        .expect("the full window has a positive width");
/// The reference's `VALUE_INFINITE`, which [`VALUE_INFINITE`] is in a build
/// without [`FLATTEN`].
const GRADIENT_INFINITE: Value = infinite_for(false);

/// The width of the window `(alpha, beta)` as the reductions read it.
///
/// Under [`FLATTEN`] infinity sits one higher, and a width measured against it
/// would move the reductions a full window feeds by a rounding step, so a search
/// with no flat mate anywhere in it would still differ from the gradient
/// build's. Measured against [`GRADIENT_INFINITE`], a window has the width it
/// has there.
#[inline(always)]
const fn window_width(alpha: Value, beta: Value) -> Value {
    if FLATTEN {
        min_value(beta, GRADIENT_INFINITE) - max_value(alpha, -GRADIENT_INFINITE)
    } else {
        beta - alpha
    }
}

/// `min` in a `const` context, where [`Ord::min`] is not available.
#[inline(always)]
const fn min_value(a: Value, b: Value) -> Value {
    if a < b { a } else { b }
}

/// `max` in a `const` context, where [`Ord::max`] is not available.
#[inline(always)]
const fn max_value(a: Value, b: Value) -> Value {
    if a > b { a } else { b }
}
/// `DEPTH_QS`.
const DEPTH_QS: i32 = 0;
/// `DEPTH_UNSEARCHED`.
const DEPTH_UNSEARCHED: i32 = -2;
/// The `MaxMovesToDraw` limit this binary plays under, when one is configured:
/// the game ply past which every interior / qsearch node adjudicates an
/// unconditional draw. A configured `0` means unlimited and compiles to `None`.
///
/// The reference expresses "unlimited" as a ply count no game reaches
/// (`100000`) and compares against it like any other limit. That compare is
/// unreachable but not removable: the game ply is a run-time value, so every
/// node pays for a test whose answer the setting already fixed. `None` fixes it
/// where the compiler can see it — see [`is_past_move_limit`].
const MOVE_LIMIT: Option<NonZeroI32> = NonZeroI32::new(crate::config::MAX_MOVES_TO_DRAW as i32);

/// Whether `game_ply` is past the configured [`MOVE_LIMIT`], so that the node
/// holding that position is an unconditional draw.
///
/// With no limit configured this is a constant `false`, and the draw clauses
/// calling it fold away to the search's own `MAX_PLY` horizon.
#[inline(always)]
const fn is_past_move_limit(game_ply: u16) -> bool {
    match MOVE_LIMIT {
        Some(limit) => game_ply as i32 > limit.get(),
        None => false,
    }
}

/// Whether a mate delivered by the side to move at `game_ply` lands on a ply
/// the configured [`MOVE_LIMIT`] still plays.
///
/// The mated position sits one ply further on, and past the limit every node
/// adjudicates a draw before it looks for a mate, so a shortcut that claims the
/// mate one ply early must ask this first or it scores a game the full search
/// calls drawn.
#[inline(always)]
const fn mate_lands_inside_move_limit(game_ply: u16) -> bool {
    mate_lands_inside(MOVE_LIMIT, game_ply)
}

/// [`mate_lands_inside_move_limit`] against an explicit `limit`.
#[inline(always)]
const fn mate_lands_inside(limit: Option<NonZeroI32>, game_ply: u16) -> bool {
    match limit {
        Some(limit) => (game_ply as i32) < limit.get(),
        None => true,
    }
}

/// Entries in the `reductions[]` table — `1..600` are filled, `[0]` is 0.
const REDUCTIONS_LEN: usize = 600;

/// `reductions[i] = int(2763/128.0 * ln(i))`, the reference's per-`go`
/// `init_reductions` table with the depth-independent factor it always uses.
/// Nothing about it varies at run time, so it is one `static` of fixed length
/// and [`QSearch::reduction`] indexes it against a literal bound.
static REDUCTIONS: [i32; REDUCTIONS_LEN] = {
    let mut table = [0i32; REDUCTIONS_LEN];
    let mut i = 1;
    while i < REDUCTIONS_LEN {
        table[i] = (2763.0 / 128.0 * const_ln(i as f64)) as i32;
        i += 1;
    }
    table
};

/// `ln(x)` for `x >= 1`, in a form the constant evaluator can run: halve `x`
/// into `[0.75, 1.5)` — exact, since it only changes the exponent — and sum the
/// `atanh` series there, where `|s| <= 0.2` makes the tail negligible after the
/// terms kept. `f64::ln` itself is not available in a `const` context.
const fn const_ln(x: f64) -> f64 {
    let mut m = x;
    let mut halvings = 0i32;
    while m >= 1.5 {
        m /= 2.0;
        halvings += 1;
    }
    // ln(m) = 2·atanh((m-1)/(m+1)) = 2·(s + s³/3 + s⁵/5 + …).
    let s = (m - 1.0) / (m + 1.0);
    let s2 = s * s;
    let mut term = s;
    let mut sum = 0.0f64;
    let mut n = 1i32;
    while n <= 25 {
        sum += term / n as f64;
        term *= s2;
        n += 2;
    }
    2.0 * sum + halvings as f64 * std::f64::consts::LN_2
}

/// `mate_in(ply)`.
fn mate_in(ply: i32) -> Value {
    VALUE_MATE - ply
}

/// `mated_in(ply)`.
fn mated_in(ply: i32) -> Value {
    -VALUE_MATE + ply
}

/// `is_valid(v)`.
fn is_valid(v: Value) -> bool {
    v != VALUE_NONE
}

/// `is_win(v)`.
fn is_win(v: Value) -> bool {
    v >= VALUE_TB_WIN_IN_MAX_PLY
}

/// `is_loss(v)`.
fn is_loss(v: Value) -> bool {
    v <= -VALUE_TB_WIN_IN_MAX_PLY
}

/// `is_decisive(v)`.
fn is_decisive(v: Value) -> bool {
    is_win(v) || is_loss(v)
}

/// `value_to_tt(v, ply)`: shift a mate score away from the root before storing.
#[inline(always)]
fn value_to_tt(v: Value, ply: i32) -> Value {
    value_to_tt_for(FLATTEN, v, ply)
}

/// [`value_to_tt`] in a build with or without [`FLATTEN`].
///
/// Under [`FLATTEN`] the flat mate carries no distance, so it is stored as it
/// is, and every other decisive value is clamped to `[-VALUE_MATE,
/// VALUE_MATE]`. The clamp is what keeps the two apart: the repetition
/// sentinels and the infinity bound are not distances either, and shifted by a
/// ply they can step onto `±VALUE_MATE_FLAT` and come back from the table as a
/// proven mate nothing proved.
#[inline(always)]
fn value_to_tt_for(flat: bool, v: Value, ply: i32) -> Value {
    if flat {
        if v.abs() == VALUE_MATE_FLAT {
            v
        } else if is_win(v) {
            (v + ply).min(VALUE_MATE)
        } else if is_loss(v) {
            (v - ply).max(-VALUE_MATE)
        } else {
            v
        }
    } else if is_win(v) {
        v + ply
    } else if is_loss(v) {
        v - ply
    } else {
        v
    }
}

/// `value_from_tt(v, ply)` (non-`STOCKFISH` variant — no rule50 downgrade):
/// shift a stored mate score back toward the root.
#[inline(always)]
fn value_from_tt(v: Value, ply: i32) -> Value {
    value_from_tt_for(FLATTEN, v, ply)
}

/// [`value_from_tt`] in a build with or without [`FLATTEN`]; the flat mate
/// comes back as it was stored.
#[inline(always)]
fn value_from_tt_for(flat: bool, v: Value, ply: i32) -> Value {
    if v == none_for(flat) {
        none_for(flat)
    } else if flat && v.abs() == VALUE_MATE_FLAT {
        v
    } else if is_win(v) {
        v - ply
    } else if is_loss(v) {
        v + ply
    } else {
        v
    }
}

/// `value_draw(nodes)`: a ±1 dither keyed on bit 1 of the node counter,
/// `VALUE_DRAW - 1 + (nodes & 0x2)`. Deterministic given the counter.
fn value_draw(nodes: u64) -> Value {
    VALUE_DRAW - 1 + (nodes & 0x2) as Value
}

/// `RootMove::operator<` as a stable-sort comparator: descending by `score`,
/// then descending by `previousScore`. Used for both the active-tail re-sort
/// and the finished-head sort.
fn root_move_order(a: &RootMove, b: &RootMove) -> std::cmp::Ordering {
    if a.score != b.score {
        b.score.cmp(&a.score)
    } else {
        b.previous_score.cmp(&a.previous_score)
    }
}

/// The reference's fail-high/low PV-output gate, as a pure predicate so it can
/// be unit-tested. The `nodes > 10_000_000` conjunct and the `root_depth < 3`
/// disjunct are easy to drop when reading the condition informally; both are
/// the reference's.
///
/// It decides whether a PV line is printed and nothing else, so it exists only
/// in a build that prints one.
#[cfg(feature = "verbose2")]
#[expect(clippy::too_many_arguments)]
pub fn fail_lh_pv_gate(
    main_thread: bool,
    multi_pv: usize,
    best_value: Value,
    alpha: Value,
    beta: Value,
    nodes: u64,
    root_depth: i32,
    interval_elapsed: bool,
    output_fail_lh_pv: bool,
) -> bool {
    main_thread
        && multi_pv == 1
        && (best_value <= alpha || best_value >= beta)
        && nodes > 10_000_000
        && (root_depth < 3 || interval_elapsed)
        && output_fail_lh_pv
}

/// `to_corrected_static_eval(v, cv)`.
fn to_corrected_static_eval(v: Value, cv: i32) -> Value {
    (v + cv / 131072).clamp(-VALUE_MAX_EVAL, VALUE_MAX_EVAL)
}

/// The low-16-bit move fragment stored in the TT (`Move16`): the reference
/// stores `Move::to_move16()`, which is the low 16 bits of the packed move.
/// `None` is what the entry holds as its zero, so an absent best move and a
/// fragment that is all zeros land on the same stored value.
fn move16_of(m: Option<Move>) -> Option<NonZeroU16> {
    m.and_then(Move::move16_stored)
}

/// `ttData.bound & (want_lower ? BOUND_LOWER : BOUND_UPPER)` as a bool
/// (`BOUND_LOWER == 2`, `BOUND_UPPER == 1`, `BOUND_EXACT == 3`).
fn bound_matches(bound: Bound, want_lower: bool) -> bool {
    let mask = if want_lower {
        Bound::Lower as u8
    } else {
        Bound::Upper as u8
    };
    (bound as u8 & mask) != 0
}

// Search stack.

/// Sentinel entries before ply 0, matching the reference's `ss = stack + 7`.
/// Only `(ss - 1)` and the `(ss - 2)` / `(ss - 4)` planes `correction_value`
/// reads are ever touched; the rest keep the layout the reference's.
const STACK_BASE: usize = 7;

/// Length of the fixed-size search stack: [`STACK_BASE`] sentinels, `MAX_PLY`
/// live plies, then two trailing cells so the deepest node's `(ss+2)` write is
/// in range. A compile-time constant rather than a `Vec` length, so the
/// optimizer can range-analyze the ply indexes and drop the per-access bounds
/// checks that otherwise dominate the hot function's panic paths.
const STACK_LEN: usize = STACK_BASE + MAX_PLY as usize + 2;

/// Length of the differential-NNUE accumulator stack: one slot per reachable
/// do/undo depth, plus headroom. Boxed, so the large `Accumulator` slots stay
/// on the heap while the length remains a compile-time constant.
const ACC_LEN: usize = MAX_PLY as usize + 8;

/// Number of [`PickerScratch`] buffers: one per [`MovePicker`] that can be live
/// at once, which is two per ply.
///
/// A node's own picker is the first. The second is the singular extension's:
/// it re-enters `search` on the node's *own* ply while that picker is still
/// yielding, and the re-entry builds a picker of its own. Nothing adds a third,
/// because the re-entry carries an excluded move and a node with one takes
/// neither the singular nor a further re-entry path. The ProbCut picker needs
/// no slot of its own — it is finished before the node's own picker is built —
/// and a node only reaches either with `ply < MAX_PLY`.
const PICKER_SCRATCH_LEN: usize = 2 * MAX_PLY as usize + 4;

/// Outcome of a top-level [`QSearch::run`].
#[derive(Debug, Clone)]
pub struct QSearchOutcome {
    /// Search value from the root side-to-move's point of view.
    pub value: Value,
    /// Number of `do_move` calls made — the reference's node-count semantics.
    pub nodes: u64,
    /// Principal variation collected at the root (empty for a non-PV run or a
    /// fail-low).
    pub pv: Vec<Move>,
    /// Maximum selective depth reached (`selDepth`), counting from 1.
    pub sel_depth: i32,
}

/// How many `check_time` calls elapse between two real clock, node or flag
/// checks — the reference's `SearchManager::callsCnt` reset value.
///
/// The counter is seeded to this value at search start, rather than the
/// reference's `0`, so the **first** checkpoint lands after a full interval. A
/// short fixed-depth search then never reaches a checkpoint, and an
/// asynchronously-set stop flag cannot perturb it.
const CHECK_INTERVAL: i32 = 512;

/// Width of a cache line on the processors the engine is built for. Two fields
/// this far apart are never written into the same line, which is what keeps one
/// worker's stores out of another's way.
pub const CACHE_LINE: usize = 64;

/// One worker's counter in a per-worker tally, on a cache line of its own.
///
/// The workers of a search publish into the same array — each to its own slot,
/// every checkpoint — and one line holds eight bare counters, so eight workers
/// packed together would take turns invalidating each other's copy of that line
/// for stores that never actually meet. The alignment spends 56 bytes per worker
/// to make each store land where no other worker is reading or writing.
#[derive(Debug)]
#[repr(align(64))]
pub struct TallySlot(AtomicU64);

// One counter per line, so no two workers' stores meet.
const _: () = assert!(size_of::<TallySlot>() == CACHE_LINE);
const _: () = assert!(align_of::<TallySlot>() == CACHE_LINE);

impl TallySlot {
    /// A zeroed slot.
    pub const fn new() -> Self {
        TallySlot(AtomicU64::new(0))
    }
}

impl Default for TallySlot {
    fn default() -> Self {
        Self::new()
    }
}

impl std::ops::Deref for TallySlot {
    type Target = AtomicU64;

    fn deref(&self) -> &AtomicU64 {
        &self.0
    }
}

/// How wide a search reads its root: how many root moves skip the late-move
/// reduction, and how far below the best one a move may score and still do so.
///
/// A search that thinks past the engine's own reply does not know which of the
/// opponent's moves will be played, and only the subtree of the one that is
/// will be of use afterwards. Widening its root searches each plausible reply
/// to full depth instead of one predicted reply deep and the rest reduced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RootWidth {
    /// Root moves, in the root's move order, searched without late-move
    /// reduction; `0` is every one. `1` is the single-line search, where only
    /// the first move is never reduced.
    pub unreduced: usize,
    /// A root move found more than this many value units below the best one is
    /// reduced as usual for the rest of the search, whatever `unreduced` says.
    /// Read only when `unreduced` is not `1`.
    pub margin: Value,
}

impl RootWidth {
    /// The single-line search every `go` runs.
    pub const SINGLE_LINE: Self = Self {
        unreduced: 1,
        margin: 1,
    };

    /// Whether this is anything but [`Self::SINGLE_LINE`]'s shape.
    fn widens(self) -> bool {
        self.unreduced != 1
    }
}

impl Default for RootWidth {
    fn default() -> Self {
        Self::SINGLE_LINE
    }
}

/// Time, node and stop controls for one search, mapped by the driver from the
/// `go` limits. With every field absent there is no early termination, and the
/// search runs to its depth limit exactly as a build without this machinery
/// would: the flag is only read at a checkpoint, and only acted on when a limit
/// is configured.
#[derive(Clone, Default)]
pub struct SearchControl {
    /// Shared abort flag, set asynchronously by the driver on `stop` / `quit`
    /// and polled at the `CHECK_INTERVAL` granularity. `None` (the default)
    /// means no external stop is possible. A flag outlives every search that
    /// polls it, so the driver keeps it in a `static`.
    pub stop: Option<&'static AtomicBool>,
    /// Hard node ceiling (`go nodes N`): abort once the node counter reaches it.
    /// `go nodes` and the `NodesLimit` config key, its only two sources, are
    /// both `verbose2`; without that feature a search is bounded by its clock,
    /// its depth and `stop` alone. The counter the ceiling reads is not gated —
    /// the search decides with it (`QSearch::nodes`) — only the ceiling is.
    /// A ceiling of zero is not a distinguishable search — the first iteration
    /// completes before the ceiling is ever consulted — so the zero pattern is
    /// free to carry the absent case.
    #[cfg(feature = "verbose2")]
    pub node_limit: Option<NonZeroU64>,
    /// The reference `TimeManagement` state plus the limit classification the
    /// search-side time control needs. `Some` only on the main worker of a `go`
    /// that has a time budget — which, without `verbose2`, is every `go`, since
    /// the six clauses that bound a search any other way all need that feature.
    pub time: Option<TimeControl>,
}

// The node ceiling is a bare `u64` in the struct, not a tagged one.
#[cfg(feature = "verbose2")]
const _: () = assert!(size_of::<Option<NonZeroU64>>() == 8);

/// The main worker's time-management state for one `go`: the reference
/// `TimeManagement` (mutated in place — its `search_end` is written by the
/// iterative-deepening budget block and `check_time`) plus the pieces of the
/// reference `LimitsType` / `MainManager` the search-side control consults.
#[derive(Clone)]
pub struct TimeControl {
    /// The reference `TimeManagement` for this `go`.
    pub tm: TimeManagement,
    /// `limits.use_time_management()`: true only for a real clock / `go rtime`,
    /// i.e. not `movetime` / `depth` / `nodes` / `infinite` / `mate`. Gates the
    /// dynamic optimum-time block and the `maximum()` stop. The five shapes it
    /// excludes all need `verbose2`, so without that feature every `go` is
    /// clock-managed and the classification does not exist.
    #[cfg(feature = "verbose2")]
    pub use_time_management: bool,
    /// `limits.movetime` in ms (`Some` only for `go movetime`): `check_time` stops
    /// the search once `elapsed >= movetime`. `go movetime` and `go mate <ms>`,
    /// its only two sources, are `verbose2` clauses.
    #[cfg(feature = "verbose2")]
    pub movetime: Option<i64>,
    /// `main_manager()->bestPreviousScore` — the previous `go`'s reported score
    /// (`VALUE_INFINITE` for the first move of a game), used to seed
    /// `iterValue`.
    pub best_previous_score: Value,
    /// `main_manager()->bestPreviousAverageScore` — the previous `go`'s
    /// reported average score, used in `fallingEval`.
    pub best_previous_average_score: Value,
    /// `main_manager()->previousTimeReduction` — the previous `go`'s final
    /// `timeReduction` (`0.85` for the first move), used in `reduction`.
    pub previous_time_reduction: f64,
}

/// The search driver, owning the network, the search stack and the one live
/// set of worker history tables. The transposition table is not among them: it
/// is one `static` whose size the build fixed, so every probe and write reaches
/// it through [`TranspositionTable::shared`] at the point of use rather than
/// through a field this struct would have to load.
///
/// Despite the name it drives the whole search: [`Self::run_root`] is iterative
/// deepening, `search` the interior body, and `qsearch` the leaf they recurse
/// into.
///
pub struct QSearch<N: NetworkParams> {
    net: N,
    /// `nodes` counter — bumped once per `do_move`. A search input in every
    /// build, not an output: the repetition returns dither their draw value on
    /// bit 1 of it (`value_draw`), the interior search's root bookkeeping runs on
    /// `(nodes & 14) == 0`, each root move accumulates its `effort` as a
    /// difference of it, and the time-management block divides by this worker's
    /// own count. The ceilings that read it are gated; the count itself is not.
    nodes: u64,
    /// Transposition-table stores this worker made whose replacement condition
    /// held, since the search started. Reported, never read by the search.
    #[cfg(feature = "verbose1")]
    tt_updates: u64,
    /// Bare static-eval stores this worker made into an entry that already held
    /// the same position's static eval, since the search started. Reported,
    /// never read by the search.
    #[cfg(feature = "verbose1")]
    eval_dups: u64,
    /// Static evaluations this worker computed for a node, since the search
    /// started. Reported, never read by the search.
    #[cfg(feature = "verbose1")]
    evals: u64,
    /// Nodes this worker gave the static eval stored in their table entry
    /// instead of evaluating, since the search started. Reported, never read
    /// by the search.
    #[cfg(feature = "verbose1")]
    eval_tt_hits: u64,
    /// How this worker's search repeated work already done, since the search
    /// started. Reported, never read by the search.
    #[cfg(feature = "dupstats")]
    dup: crate::dupstats::DupCounts,
    /// This worker's index in its pool, as the duplication statistics record
    /// it. Zero on the single-worker path.
    #[cfg(feature = "dupstats")]
    dup_worker: u8,
    /// When this worker finished each root iteration of the search. Reported,
    /// never read by the search.
    #[cfg(feature = "dupstats")]
    timeline: Box<crate::timeline::Timeline>,
    /// Where this worker's nodes are, since the search started. Reported, never
    /// read by the search.
    #[cfg(feature = "dupstats")]
    node_hist: Box<crate::dupstats::NodeHist>,
    /// What this worker takes from the table and how its root searches went,
    /// written to the record file when its search ends. Never read by the
    /// search.
    #[cfg(feature = "dupstats")]
    channels: Box<crate::channels::Record>,
    /// When the search this worker belongs to started, the clock every worker
    /// of it reads its iterations against; `None` starts it when the worker
    /// starts.
    #[cfg(feature = "dupstats")]
    search_start: Option<Instant>,
    /// The latch that ends a measurement search at the first worker to
    /// finish; `None` for every other search.
    #[cfg(all(feature = "dupstats", feature = "verbose3"))]
    first_finish: Option<&'static crate::timeline::FirstFinish>,
    /// This worker's move-order spread: its duplicate count and the moves each
    /// ply has pulled out of its order without searching them yet.
    #[cfg(lazy_smp_spread)]
    spread: crate::spread::Spread<Move>,
    /// Whether this worker puts off a move another worker is searching, and
    /// the moves each ply has put off.
    #[cfg(lazy_smp_defer)]
    defer: crate::defer::Defer<Move>,
    /// The ply of the last node that returned the transposition table's value
    /// without searching, so a node can tell whether a deferred move's child
    /// did. Reported, never read by the search.
    #[cfg(all(lazy_smp_defer, feature = "verbose1"))]
    tt_cut_ply: i32,
    /// What the tests read off the nodes whose TT move went through a singular
    /// search.
    #[cfg(all(lazy_smp_spread, test))]
    singular_audit: SingularAudit,
    /// How many plies back this worker looks for a repetition, `0` for no
    /// limit: the config's `repetition_look_back` unless a test set another.
    /// The index of the game before the root that the window reaches into is
    /// the histories bundle's (`WorkerHistories::history_index`).
    #[cfg(repetition_index)]
    repetition_look_back: u32,
    /// `selDepth`.
    sel_depth: i32,
    /// `nmpMinPly`: while a null-move verification search runs, the ply below
    /// which null-move pruning stays disabled. Zero means none is in flight.
    /// The reference's per-`go` reset is dead code in the live build, so this
    /// is seeded once and restored to 0 by the verification block itself.
    nmp_min_ply: i32,

    /// Whether this run is a PV search (`nodeType == PV`). Invariant down the
    /// recursion.
    pv_node: bool,
    /// Whether TT hits are honoured (`ReadTT`). Invariant down the recursion.
    /// The engine always honours them; the `ReadTT == false` instantiation
    /// exists to be exercised, so only a test build carries the state and only
    /// there is [`Self::reads_tt`] anything but the constant `true`.
    #[cfg(test)]
    read_tt: bool,

    /// Root side-to-move, used by [`Self::draw_value`] to reproduce the
    /// contempt-signed `drawValueTable[REPETITION_DRAW]` (set once per search
    /// from the root side). The contempt's magnitude is compiled in
    /// ([`DRAW_CONTEMPT_BLACK`] / [`DRAW_CONTEMPT_WHITE`]); this is what gives
    /// it its sign.
    root_us: Color,

    /// The persistent search stack (`STACK_BASE` sentinels + `MAX_PLY` + 1),
    /// a fixed-size boxed array (length [`STACK_LEN`]) mirroring the reference's
    /// `Stack stack[MAX_PLY + 10]`. The constant length lets the optimizer
    /// prove `si(ply)` indexes in range and elide bounds checks in the hot
    /// search function.
    stack: Box<[SearchStackCell; STACK_LEN]>,

    /// This worker's private differential-NNUE accumulator stack, whose top is
    /// always the accumulator for the position the current node is searching.
    /// Never shared, and preallocated so a node never allocates; the constant
    /// length lets the optimizer drop the per-access bounds check.
    ///
    /// Each slot carries its two `i16` rows inline, so the whole stack is one
    /// contiguous block and a row is the base address plus a constant: the depth
    /// scales the slot size, and nothing on the way to the lanes is loaded from
    /// memory.
    acc_stack: Box<[Accumulator; ACC_LEN]>,
    /// Index of the current node's accumulator within [`Self::acc_stack`] (the
    /// live top of the do/undo stack). Incremented per `do_move`, decremented per
    /// `undo_move`; `0` at every root node.
    acc_depth: usize,
    /// This worker's private finny table. Allocated once and never shared, so
    /// it needs no synchronisation.
    finny: Box<FinnyCache>,

    /// The [`MovePicker`] buffers, [`PICKER_SCRATCH_LEN`] of them, in one
    /// block allocated once here so no node ever reaches the allocator.
    ///
    /// The recursion hands each picker a `&mut` to one of them and searches on
    /// with the rest, which is a borrow of this array that outlives the `&mut
    /// self` of the search call it spans — so the array is lent out for the
    /// length of one run ([`Self::lend_picker_scratch`]) and this field is
    /// `None` while that run is in flight. Nothing reads it there.
    picker_scratch: Option<Box<[PickerScratch; PICKER_SCRATCH_LEN]>>,
    /// Test-only: when set, [`Self::static_eval`] asserts the differential
    /// accumulator equals a from-scratch [`yorkie_eval::evaluate`] at every
    /// evaluation point (the accumulator-equivalence test). Enabled via
    /// [`Self::set_verify_accumulator`] by that test alone, so a build that is
    /// not the test build neither carries the flag nor compiles the check.
    #[cfg(test)]
    verify_accumulator: bool,
    /// Test-only: the root-move list as the last [`Self::run_worker`] left it,
    /// for the tests that read what a search did per root move.
    #[cfg(test)]
    last_root_moves: Vec<RootMove>,

    /// The single set of live worker history tables, read **everywhere** in the
    /// search tree, so that an interior update is visible to a later leaf
    /// qsearch — the reference's contract.
    histories: WorkerHistories,
    /// `rootDelta` — the width `beta - alpha` of the *root* aspiration window,
    /// read by [`Self::reduction`]. [`Self::set_root_delta`] is the only way in;
    /// [`Self::run_root`] calls it before each `search<Root>` call, and the
    /// default is the full-window width `2 * VALUE_INFINITE`.
    ///
    /// A width, not a [`Value`]: `reduction` divides by it once per move at
    /// every node, and a divisor a signed division cannot fault on has to be
    /// positive — neither `0` nor `-1`, the two cases the language checks for
    /// and panics on. Unsigned and non-zero is exactly that, and
    /// [`Self::root_delta_divisor`] carries the range across to the `i32` the
    /// division wants.
    root_delta: NonZeroU32,
    /// `rootDepth` — the current iterative-deepening depth, read by the Step-20
    /// nodes tie-break (`ss->ply + 2 >= rootDepth`). [`Self::run_root`] sets it
    /// per iteration; default `1`.
    root_depth: i32,
    /// `lastIterationPV` — the previous iteration's PV, consulted to seed
    /// `ss->followPV` from iteration 2 on. Cleared per `go`, then assigned at the
    /// end of each iterative-deepening iteration.
    last_iteration_pv: Vec<Move>,

    /// Time / node / stop controls. Empty by default (the
    /// fixed-depth parity path); the driver sets it via [`Self::set_control`]
    /// before an under-clock `go`.
    control: SearchControl,
    /// The entering-king declaration thresholds for this `go`, read by the two
    /// in-search `declaration_win` checks (`run_root`'s root shortcut and the
    /// interior Step-5 check). The rule itself is compiled in, and only its two
    /// handicap-aware forms make a threshold depend on the root position; under
    /// every other rule the checks read the compiled pair and never this
    /// snapshot. The driver installs it per `go` via
    /// [`Self::set_entering_king`].
    entering_king: EnteringKingConfig,
    /// `go mate` mode (`limits.mate != 0`). When set, the iterative-deepening
    /// early mate/mated break is disabled (the search keeps proving within its
    /// time budget) and the mate-found stop rule is armed so a proven mate
    /// terminates promptly. `false` on every non-mate `go`, so the parity path
    /// is unchanged. Set per `go` via [`Self::set_mate_mode`]. Only a `verbose2`
    /// build can parse the clause that sets it.
    #[cfg(feature = "verbose2")]
    mate_mode: bool,
    /// The reference `callsCnt` down-counter: `check_time` fires its real check
    /// once this reaches zero, then reloads it (see [`CHECK_INTERVAL`]).
    calls_cnt: i32,
    /// Latched abort state — set once a checkpoint observes the stop flag, the
    /// node ceiling, or the hard deadline. Every stop-check site early-returns on
    /// this; it stays `false` for a limit-free search, so those sites are inert.
    stopped: bool,
    /// `completedDepth`, published so `check_time` can gate the time / node
    /// stops on "at least one iteration finished". Reset to `0` per `go`; set
    /// to `rootDepth` at each completed iteration.
    completed_depth: i32,
    /// This worker's own `bestMoveChanges` since the last iteration. Used only
    /// on the single-worker path; under Lazy-SMP the shared per-worker slot
    /// takes its place, so the main worker can sum every worker's count.
    best_move_changes: f64,
    /// Lazy-SMP shared node counters, one slot per worker, from this worker's
    /// own slot (always the first here) to the pool's last — the whole pool for
    /// the main worker. Each checkpoint publishes to its own slot, and the main
    /// worker's node ceiling sums them, reproducing `threads.nodes_searched()`.
    /// `None` on the single-worker path, where `self.nodes` is authoritative.
    ///
    /// The aggregate has two readers, the node ceiling and the `nodes` field of
    /// a search `info` line, and both are `verbose2`; the search itself decides
    /// on this worker's own [`Self::nodes`], never on the sum.
    ///
    /// A slice and nothing else, so the field is as wide as the handle it
    /// stands for and every field laid out after it keeps its offset; which
    /// worker this is lives in [`Self::main_tally`].
    #[cfg(feature = "verbose2")]
    node_tally: Option<&'static [TallySlot]>,
    /// Lazy-SMP shared best-move-change counters, in the same shape as
    /// `node_tally`. Every worker adds to its own slot; only the main worker
    /// folds *every* slot into `totBestMoveChanges` and zeroes them, at each
    /// iteration end. Relaxed atomics, because the reference's cross-thread
    /// reads here are benign races.
    best_move_tally: Option<&'static [TallySlot]>,

    /// `pvIdx` — the current MultiPV line index. Read at the root by the
    /// interior-search hooks; `0` on the first line, so those hooks are no-ops
    /// and the single-PV search stays bit-identical.
    ///
    /// A second PV line can only be reported by a build that prints search
    /// `info` lines, so without that feature the root is single-line, this index
    /// is the constant 0, and the hooks that read it are not compiled.
    #[cfg(feature = "verbose2")]
    pv_idx: usize,
    /// `min(options["MultiPV"], rootMoves.size())` input — the raw `MultiPV`
    /// option value. `1` by default (the fixed-depth parity path runs one PV
    /// line). Set per `go` by [`Self::set_multi_pv`].
    #[cfg(feature = "verbose2")]
    multi_pv: usize,
    /// The per-iteration / final PV output sink (`main_manager()->pv()`). `None`
    /// on every worker but the main one, so helpers and the parity path emit
    /// nothing.
    #[cfg(feature = "verbose2")]
    pv_sink: Option<Box<dyn PvSink>>,
    /// The main worker's PV-output configuration for this `go` (`None` elsewhere).
    #[cfg(feature = "verbose2")]
    pv_config: Option<PvOutputConfig>,
    /// `lastPvInfoTime` — the last time a PV was emitted (refreshed at each
    /// `pv()` call). Seeded to `pv_config.start_time` per `go`; unused when
    /// `pv_config` is `None`.
    #[cfg(feature = "verbose2")]
    last_pv_info_time: Instant,

    /// The value marks carried on the return path: on return from
    /// [`Self::search`] or [`Self::qsearch`] they are the marks of the value
    /// that call returned, and while a node runs they are that node's marks so
    /// far.
    ///
    /// A node clears them on entry, sets one where its value comes out of the
    /// corresponding rule check or out of a marked table entry, and folds in
    /// the marks of every child that moved its window. [`Self::tt_store`]
    /// writes whatever they hold at the moment of the store. A sub-search
    /// overwrites them, so a caller that must keep its own across one saves
    /// them and restores them through [`Self::child_marks`].
    ///
    /// **No search decision reads them.** They exist so the reply can say which
    /// rules the value it rests on is behind.
    #[cfg(feature = "verbose3")]
    marks: ValueMarks,
    /// The positions before the current node that the repetition judgements
    /// behind [`Self::marks`]' `pathdep` matched (see [`crate::causes`]).
    /// They ride exactly as the marks do, and no search decision reads them
    /// either.
    #[cfg(feature = "verbose3")]
    causes: PathCauses,

    /// The evaluation-noise amplitude for this game, in the unit the search
    /// scores in: [`Self::static_eval`] offsets every position by
    /// [`noise`] of this width. `0` — the default, and what a search that never
    /// installs one keeps — leaves each evaluation exactly as the network
    /// produced it.
    #[cfg(feature = "random")]
    random_amplitude: Value,
    /// The seed [`noise`] draws against. Set per `go` from the driver's
    /// game-scoped seed, so every worker of a search agrees on each position's
    /// offset and no search ever redraws one mid-game.
    #[cfg(feature = "random")]
    random_seed: u64,
    /// Whether this worker's tallies start at the pool's first slot — the main
    /// worker, which alone reads every slot. Declared last, so it takes a byte
    /// of the struct's tail rather than moving a field before it.
    main_tally: bool,
    /// The side the engine plays in the game, set from the root at the start of
    /// every search: the side to move there, or the other one when the search
    /// thinks ahead ([`Self::set_thinking_ahead`]). Under [`FLATTEN`] it decides
    /// which side's mates are flat, so it has to name the same side in every
    /// search of a game: a table entry the search after the engine's own move
    /// stores is read by the next `go`'s.
    ///
    /// This and the next field are arrays of length one under [`FLATTEN`] and
    /// of length zero without it, so a build that scores by distance carries
    /// neither and lays the struct out exactly as it would with no such
    /// setting. Read through [`Self::is_engine`].
    engine_us: [Color; FLATTEN as usize],
    /// Whether the root is the position after the engine's own move — the
    /// opponent to move — rather than one the engine is to move in.
    thinking_ahead: [bool; FLATTEN as usize],
    /// How wide this search reads its root. Set per search by
    /// [`Self::set_root_width`]; [`RootWidth::SINGLE_LINE`] otherwise.
    root_width: HeldWidth,
}

/// What a node's store found in the table entry and who stored, read before
/// the store so it can be told afterwards what the store did to the entry.
#[cfg(feature = "dupstats")]
#[derive(Clone, Copy)]
struct DupStore {
    /// The worker's table-update count before the store: a store whose
    /// replacement condition held raises it.
    updates_before: u64,
    /// The depth the entry held the node's position at, and whether the
    /// running search wrote it; `None` for another position or nothing.
    found: Option<(i32, bool)>,
    /// Who last wrote the entry, when the side table still knows.
    previous: Option<crate::dupstats::WriterId>,
    /// The store's depth.
    depth: i32,
    /// The node's histogram buckets.
    at: crate::dupstats::Where,
}

/// A [`RootWidth`] as the search holds it: three bytes of alignment one with no
/// niche, which the compiler lays out after every other field, in the struct's
/// tail. Any wider field, or one it sorts in among the four-byte ones, moves the
/// node path's flags to other offsets, and that alone costs measurable speed.
#[derive(Clone, Copy)]
struct HeldWidth {
    unreduced: u8,
    margin: [u8; 2],
}

const _: () = assert!(size_of::<HeldWidth>() == 3 && align_of::<HeldWidth>() == 1);

impl HeldWidth {
    /// `width` with its count and margin saturated to what the bytes hold: a
    /// count above 255 reads as 255, and a margin wider than the whole value
    /// range as the widest one there is.
    fn new(width: RootWidth) -> Self {
        let unreduced = u8::try_from(width.unreduced).unwrap_or(u8::MAX);
        let margin = u16::try_from(width.margin.max(1)).unwrap_or(u16::MAX);
        Self {
            unreduced,
            margin: margin.to_ne_bytes(),
        }
    }

    fn get(self) -> RootWidth {
        RootWidth {
            unreduced: self.unreduced as usize,
            margin: u16::from_ne_bytes(self.margin) as Value,
        }
    }
}

/// The nodes whose TT move went through a singular search, and how many moves
/// such a node counted that it had counted before.
#[cfg(all(lazy_smp_spread, test))]
#[derive(Clone, Copy, Debug, Default)]
struct SingularAudit {
    nodes: u64,
    repeats: u64,
}

/// The result one worker's iterative deepening produces, consumed by
/// the driver's Lazy-SMP orchestration to vote for and report a single result.
#[derive(Clone, Debug)]
pub struct WorkerResult {
    /// The last completed iteration's `rootMoves[0]` (or, on an abort before any
    /// iteration completed, the best-so-far after the partial iteration's sort).
    pub best: RootMove,
    /// `completedDepth` — the last fully-completed iterative-deepening depth (`0`
    /// if none completed). Reported as `info depth` for the chosen worker and fed
    /// to the thread vote.
    pub completed_depth: i32,
    /// This worker's own node count (`do_move` calls). The driver sums every
    /// worker's count for the aggregated `info ... nodes` output.
    pub nodes: u64,
    /// This worker's transposition-table stores whose replacement condition
    /// held, for the per-reply statistics line.
    #[cfg(feature = "verbose1")]
    pub tt_updates: u64,
    /// This worker's bare static-eval stores into an entry that already held the
    /// same position's static eval, for the per-reply statistics line.
    #[cfg(feature = "verbose1")]
    pub eval_dups: u64,
    /// This worker's static evaluations of a node, for the per-reply
    /// statistics line.
    #[cfg(feature = "verbose1")]
    pub evals: u64,
    /// This worker's nodes that took their static eval from the table entry
    /// instead of evaluating, for the per-reply statistics line.
    #[cfg(feature = "verbose1")]
    pub eval_tt_hits: u64,
    /// How this worker's search repeated work already done, for the per-reply
    /// statistics line. Boxed: the counts are a few kilobytes, and a result is
    /// moved between threads.
    #[cfg(feature = "dupstats")]
    pub dup: Box<crate::dupstats::DupCounts>,
    /// The root move this worker was searching when the search was stopped, or
    /// `None` when its last iteration ran to its end.
    #[cfg(feature = "dupstats")]
    pub stopped_on: Option<Move>,
    /// When this worker finished each root iteration, and the one it was
    /// stopped inside. Boxed for the same reason as `dup`.
    #[cfg(feature = "dupstats")]
    pub timeline: Box<crate::timeline::Timeline>,
    /// Where this worker's nodes are: by ply, remaining depth and node type.
    /// Boxed for the same reason as `dup`.
    #[cfg(feature = "dupstats")]
    pub node_hist: Box<crate::dupstats::NodeHist>,
    /// What this worker's move-order spread counted, for the per-reply
    /// statistics line.
    #[cfg(all(lazy_smp_spread, feature = "verbose1"))]
    pub spread: crate::spread::SpreadCounts,
    /// What this worker counted about the moves it put off, for the per-reply
    /// statistics line.
    #[cfg(all(lazy_smp_defer, feature = "verbose1"))]
    pub defer: crate::defer::DeferCounts,
    /// Whether this worker (only meaningful for the main worker) already
    /// emitted the last completed iteration's final PV during iterative
    /// deepening — the reference `uciPvSent` return of `iterative_deepening`.
    /// The coordinator's final-PV fallback keys off this so a fully throttled
    /// search still emits one PV before `bestmove`. Always `false` for a helper
    /// (helpers never emit).
    #[cfg(feature = "verbose2")]
    pub uci_pv_sent: bool,
    /// The last completed iteration's top-`multiPV` root moves, in score order —
    /// the lines the coordinator re-emits when the final PV was throttled
    /// (`uci_pv_sent == false`). `[best]` for the fixed-depth / helper paths.
    #[cfg(feature = "verbose2")]
    pub pv_lines: Vec<RootMove>,
    /// `timeReduction` after iterative deepening — the value the driver stores
    /// as `previousTimeReduction` for the next `go`. `1.0` when the
    /// time-management block never ran (fixed-depth / helper).
    pub time_reduction: f64,
}

/// USI `info` bound marker for one PV line — the reference `pv()` `isExact` /
/// `scoreLowerbound` / `scoreUpperbound` logic.
///
/// One field of a PV line, so it exists only in a build that prints one.
#[cfg(feature = "verbose2")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PvBound {
    /// An exact score (no `lowerbound` / `upperbound` marker).
    Exact,
    /// A fail-high lower bound (`lowerbound`).
    Lower,
    /// A fail-low upper bound (`upperbound`).
    Upper,
}

/// One PV line's data for a USI `info` output — the reference `InfoFull` as
/// this port surfaces it. The Protocol layer formats it into the wire line; the
/// `Value`→`cp`/`mate` and `Move`→USI conversions live there.
///
/// The whole PV-output surface below — this type, its bound marker, the sink
/// trait, the configuration and every method that assembles or emits a line —
/// exists only to produce those `info` lines, so a build that prints none
/// compiles none of it.
#[cfg(feature = "verbose2")]
#[derive(Clone, Debug)]
pub struct PvInfo {
    /// `info.depth`.
    pub depth: i32,
    /// `info.selDepth`.
    pub sel_depth: i32,
    /// `info.multiPV` — the **1-based** PV index (`i + 1`).
    pub multipv: usize,
    /// `info.score`, still a raw search [`Value`].
    pub score: Value,
    /// Whether the score is exact or a fail-high/low bound.
    pub bound: PvBound,
    /// `info.nodes`.
    pub nodes: u64,
    /// `info.nps` — `nodes * 1000 / time_ms`.
    pub nps: u64,
    /// The transposition table's occupancy in permille at the moment the line
    /// was assembled (`TranspositionTable::hashfull`, `maxAge = 0`).
    pub hashfull: u32,
    /// `info.timeMs` — milliseconds since the `go` that started this search
    /// (`tm.elapsed_time()`), floored at 1 so [`Self::nps`] has a divisor.
    pub time_ms: u64,
    /// `info.pv` as moves (the Protocol layer joins them into USI text).
    pub pv: Vec<Move>,
}

/// Sink for per-iteration / final PV `info` output (the reference
/// `main_manager()->pv()` call sites). Only the **main** worker is given one;
/// helper workers and the direct fixed-depth [`QSearch::run_root`] path leave it
/// unset, so they never emit and the parity path is untouched.
#[cfg(feature = "verbose2")]
pub trait PvSink: Send {
    /// Emit one PV line.
    fn emit(&mut self, info: &PvInfo);
}

/// The PV-output configuration snapshot for one `go`, as the reference derives
/// it per search. Installed on the main worker only, via
/// [`QSearch::set_pv_output`].
///
/// `MultiPV` is not here: it shapes the search itself, so the driver passes it
/// separately and every build applies it.
#[cfg(feature = "verbose2")]
#[derive(Clone)]
pub struct PvOutputConfig {
    /// `computed_pv_interval`: `0` (never suppress) when `go infinite` or
    /// `ConsiderationMode` is on, else the `PvInterval` option as a duration.
    pub pv_interval: Duration,
    /// `ConsiderationMode`: collect each PV from the transposition table
    /// instead of the searched PV array.
    pub consideration_mode: bool,
    /// `OutputFailLHPV`: emit a PV on a fail-high/low re-search.
    pub output_fail_lh_pv: bool,
    /// `limits.startTime` — the `lastPvInfoTime` seed.
    pub start_time: Instant,
}

/// `Eval::PawnValue`, used to scale the contempt setting.
const PAWN_VALUE: i32 = 90;

/// `drawValueTable[REPETITION_DRAW][BLACK]`: the `DrawValueBlack` setting scaled
/// by [`PAWN_VALUE`], truncated toward zero as the reference's integer division
/// is — with the default `-2` that is `-2 * 90 / 100 == -1`.
const DRAW_CONTEMPT_BLACK: Value =
    (crate::config::DRAW_VALUE_BLACK * PAWN_VALUE as i64 / 100) as Value;
/// `drawValueTable[REPETITION_DRAW][WHITE]`, from `DrawValueWhite`.
const DRAW_CONTEMPT_WHITE: Value =
    (crate::config::DRAW_VALUE_WHITE * PAWN_VALUE as i64 / 100) as Value;

/// A fresh evaluation-noise seed for one game.
///
/// `RandomState`'s keys come from the operating system's randomness, drawn once
/// per thread and advanced per instance, so two processes draw different seeds
/// and one process's successive draws differ from each other.
#[cfg(feature = "random")]
pub fn new_game_seed() -> u64 {
    use std::hash::{BuildHasher, Hasher, RandomState};

    RandomState::new().build_hasher().finish()
}

/// splitmix64's 64-bit finaliser: an avalanche over the whole word, so two keys
/// differing in a single bit land far apart in the low bits [`noise`] reads.
/// Without it, positions one move apart would draw neighbouring offsets.
#[cfg(feature = "random")]
const fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The evaluation offset a position carries throughout one game: a value in
/// `-amplitude / 2 ..= amplitude - 1 - amplitude / 2`, determined by the
/// position's Zobrist `key` and the game's `seed` alone, so the same position
/// always draws the same offset within a game and a different one in the next.
///
/// `amplitude == 0` is no noise at all, which is what makes the setting behind
/// it a true off switch rather than a small perturbation.
#[cfg(feature = "random")]
pub fn noise(key: u64, seed: u64, amplitude: Value) -> Value {
    if amplitude <= 0 {
        return 0;
    }
    (mix(key ^ seed) % amplitude as u64) as Value - amplitude / 2
}

impl<N: NetworkParams> QSearch<N> {
    /// Create a driver over `net` with fresh history tables. The reference
    /// re-fills `lowPlyHistory` to 98 per `go`; [`WorkerHistories::new`] leaves
    /// it zero, so it is seeded here so a bare [`Self::run`] /
    /// [`Self::run_search`] (which do not run the per-`go` refill) sees the
    /// reference value.
    pub fn new(net: N) -> Self {
        let histories = {
            let mut h = WorkerHistories::new();
            h.low_ply.fill(98);
            h
        };
        Self::with_histories(net, histories)
    }

    /// Create a driver over `net`, taking ownership of externally-held
    /// `histories`.
    ///
    /// The game-scoped path: histories persist across `go`s within a game, as
    /// the reference's do, and are reclaimed afterwards with
    /// [`Self::into_histories`].
    pub fn with_histories(net: N, histories: WorkerHistories) -> Self {
        Self {
            net,
            nodes: 0,
            #[cfg(feature = "verbose1")]
            tt_updates: 0,
            #[cfg(feature = "verbose1")]
            eval_dups: 0,
            #[cfg(feature = "verbose1")]
            evals: 0,
            #[cfg(feature = "verbose1")]
            eval_tt_hits: 0,
            #[cfg(feature = "dupstats")]
            dup: crate::dupstats::DupCounts::ZERO,
            #[cfg(feature = "dupstats")]
            dup_worker: 0,
            #[cfg(feature = "dupstats")]
            timeline: Box::default(),
            #[cfg(feature = "dupstats")]
            node_hist: Box::default(),
            #[cfg(feature = "dupstats")]
            channels: Box::default(),
            #[cfg(feature = "dupstats")]
            search_start: None,
            #[cfg(all(feature = "dupstats", feature = "verbose3"))]
            first_finish: None,
            #[cfg(lazy_smp_spread)]
            // Alone until a pool says otherwise (`set_best_move_tally`).
            spread: crate::spread::Spread::new(STACK_LEN, crate::config::LAZY_SMP_SPREAD, 1),
            #[cfg(lazy_smp_defer)]
            // Alone until a pool says otherwise (`set_best_move_tally`).
            defer: crate::defer::Defer::new(STACK_LEN, crate::config::LAZY_SMP_DEFER, 1),
            #[cfg(all(lazy_smp_defer, feature = "verbose1"))]
            tt_cut_ply: -1,
            #[cfg(all(lazy_smp_spread, test))]
            singular_audit: SingularAudit::default(),
            #[cfg(repetition_index)]
            repetition_look_back: crate::config::REPETITION_LOOK_BACK as u32,
            sel_depth: 0,
            nmp_min_ply: 0,
            pv_node: false,
            #[cfg(test)]
            read_tt: true,
            root_us: Color::Black,
            // A fixed-size boxed array; each cell's `pv` is preallocated so a
            // PV update never grows the buffer on the hot path. Built as a
            // `Vec` (each cell distinct — `SearchStackCell` is not `Copy`) then
            // converted to the boxed array without a stack copy.
            stack: Vec::from_fn(STACK_LEN, |_| {
                let mut cell = SearchStackCell::default();
                cell.pv.reserve_exact(MAX_PLY as usize + 1);
                cell
            })
            .try_into()
                .map_err(|_| ())
                .expect("STACK_LEN cells collected"),
            // One accumulator slot per reachable do/undo depth (bounded by
            // `MAX_PLY`, plus headroom); `Accumulator` is not `Clone`, so build
            // the slots individually. A boxed fixed-size array keeps the large
            // slots on the heap (no stack copy) with a compile-time length, and
            // is the whole stack's one allocation: a slot's rows are part of it.
            acc_stack: Vec::from_fn(ACC_LEN, |_| Accumulator::new())
                .try_into()
                .map_err(|_| ())
                .expect("ACC_LEN slots collected"),
            acc_depth: 0,
            finny: FinnyCache::new(),
            // One block for the whole set: the buffers carry their storage
            // inline, so this is the only allocation any of them costs.
            picker_scratch: Some(
                Vec::from_fn(PICKER_SCRATCH_LEN, |_| PickerScratch::new())
                    .try_into()
                    .map_err(|_| ())
                    .expect("PICKER_SCRATCH_LEN buffers collected"),
            ),
            #[cfg(test)]
            verify_accumulator: false,
            #[cfg(test)]
            last_root_moves: Vec::new(),
            histories,
            root_delta: ROOT_DELTA_FULL,
            root_depth: 1,
            // Reserved to the longest PV a root move can hold, so the copy each
            // completed iteration makes into it never grows the buffer.
            last_iteration_pv: Vec::with_capacity(MAX_PLY as usize + 1),
            control: SearchControl::default(),
            entering_king: EnteringKingConfig::default(),
            #[cfg(feature = "verbose2")]
            mate_mode: false,
            calls_cnt: CHECK_INTERVAL,
            stopped: false,
            completed_depth: 0,
            best_move_changes: 0.0,
            #[cfg(feature = "verbose2")]
            node_tally: None,
            best_move_tally: None,
            main_tally: false,
            #[cfg(feature = "verbose2")]
            pv_idx: 0,
            #[cfg(feature = "verbose2")]
            multi_pv: 1,
            #[cfg(feature = "verbose2")]
            pv_sink: None,
            #[cfg(feature = "verbose2")]
            pv_config: None,
            #[cfg(feature = "verbose2")]
            last_pv_info_time: Instant::now(),
            #[cfg(feature = "verbose3")]
            marks: ValueMarks::NONE,
            #[cfg(feature = "verbose3")]
            causes: PathCauses::NONE,
            #[cfg(feature = "random")]
            random_amplitude: 0,
            #[cfg(feature = "random")]
            random_seed: 0,
            engine_us: [Color::Black; FLATTEN as usize],
            thinking_ahead: [false; FLATTEN as usize],
            root_width: HeldWidth::new(RootWidth::SINGLE_LINE),
        }
    }

    /// Test-only (the accumulator-equivalence test): when enabled, every
    /// evaluation point re-checks the differential accumulator against a
    /// from-scratch [`yorkie_eval::evaluate`]. Never enabled on a production
    /// search, so the refresh entry point is never called there.
    #[cfg(test)]
    pub fn set_verify_accumulator(&mut self, verify: bool) {
        self.verify_accumulator = verify;
    }

    /// Whether TT hits are honoured at the node about to probe. The engine
    /// always honours them, so this is the constant `true` outside the test
    /// build that exercises the `ReadTT == false` instantiation.
    #[inline]
    fn reads_tt(&self) -> bool {
        #[cfg(test)]
        {
            self.read_tt
        }
        #[cfg(not(test))]
        {
            true
        }
    }

    /// Set whether TT hits are honoured. Every descent into a child restores
    /// `true`, mirroring the reference's `ReadTT` template argument; with no
    /// state behind it that restoration compiles to nothing.
    #[inline]
    fn set_read_tt(&mut self, read_tt: bool) {
        #[cfg(test)]
        {
            self.read_tt = read_tt;
        }
        #[cfg(not(test))]
        let _ = read_tt;
    }

    /// Install the time / node / stop [`SearchControl`] for the coming search.
    /// With the default (empty) control every stop-check site is inert and the
    /// search runs the fixed-depth parity path.
    pub fn set_control(&mut self, control: SearchControl) {
        self.control = control;
    }

    /// Install the entering-king declaration thresholds for this `go`. Called by
    /// the driver after computing them from the root position; every worker
    /// (main + helpers) gets the same snapshot, since the material total is
    /// invariant across the search.
    pub fn set_entering_king(&mut self, config: EnteringKingConfig) {
        self.entering_king = config;
    }

    /// Declare whether the coming search's root is the position the engine's
    /// own move just reached, with the opponent to move: the search that thinks
    /// past a reply. The engine's side is then the one not to move at the root.
    /// `false` (the default) is every other search.
    pub fn set_thinking_ahead(&mut self, thinking_ahead: bool) {
        if let Some(slot) = self.thinking_ahead.first_mut() {
            *slot = thinking_ahead;
        }
    }

    /// Fix the two sides a search scores against, from the side to move at its
    /// root: that side itself for the draw contempt, and the engine's side for
    /// the flat mate.
    fn set_root_side(&mut self, root_side: Color) {
        self.root_us = root_side;
        if let Some(slot) = self.engine_us.first_mut() {
            *slot = if self.thinking_ahead.first() == Some(&true) {
                root_side.flip()
            } else {
                root_side
            };
        }
    }

    /// Whether `c` is the engine's side; always `false` without [`FLATTEN`],
    /// which keeps no side.
    #[inline(always)]
    fn is_engine(&self, c: Color) -> bool {
        self.engine_us.first() == Some(&c)
    }

    /// `mate_in(ply)` for a mate `us` delivers: [`VALUE_MATE_FLAT`] when `us` is
    /// the engine under [`FLATTEN`].
    #[inline(always)]
    fn mate_for(&self, us: Color, ply: i32) -> Value {
        if FLATTEN && self.is_engine(us) {
            VALUE_MATE_FLAT
        } else {
            mate_in(ply)
        }
    }

    /// `mated_in(ply)` for a mate `us` suffers: `-VALUE_MATE_FLAT` when `us` is
    /// the engine's opponent under [`FLATTEN`].
    #[inline(always)]
    fn mated_for(&self, us: Color, ply: i32) -> Value {
        if FLATTEN && !self.is_engine(us) {
            -VALUE_MATE_FLAT
        } else {
            mated_in(ply)
        }
    }

    /// Install this game's evaluation noise: `amplitude` in the unit the search
    /// scores in, and the game's `seed`. Every worker of a `go` is given the
    /// same pair, so they evaluate a shared position identically, and the pair
    /// is stable for as long as the game is.
    #[cfg(feature = "random")]
    pub fn set_random(&mut self, amplitude: Value, seed: u64) {
        self.random_amplitude = amplitude;
        self.random_seed = seed;
    }

    /// Install `go mate` mode for this `go`: disables the
    /// iterative-deepening early mate/mated break and arms the mate-found stop
    /// rule. Every worker gets the same flag; `false` (the default) leaves the
    /// iterative-deepening loop bit-identical to the parity path.
    #[cfg(feature = "verbose2")]
    pub fn set_mate_mode(&mut self, mate: bool) {
        self.mate_mode = mate;
    }

    /// Install the Lazy-SMP shared node counters: `slots` holds one
    /// [`TallySlot`] per worker and `index` is this worker's slot. Each
    /// `check_time` checkpoint publishes `self.nodes` there so the main worker's
    /// `go nodes N` ceiling can sum every worker's count (the reference
    /// `threads.nodes_searched()`). Leave unset for the single-worker path.
    #[cfg(feature = "verbose2")]
    pub fn set_node_tally(&mut self, slots: &'static [TallySlot], index: usize) {
        self.node_tally = Some(&slots[index..]);
        self.main_tally = index == 0;
    }

    /// Install the Lazy-SMP shared best-move-change counters, in the same
    /// shape as the node counters (`verbose2`). Leave unset for the single-worker
    /// path, where `best_move_changes` carries the count.
    pub fn set_best_move_tally(&mut self, slots: &'static [TallySlot], index: usize) {
        self.best_move_tally = Some(&slots[index..]);
        self.main_tally = index == 0;
        #[cfg(feature = "dupstats")]
        {
            self.dup_worker = index.min(u8::MAX as usize) as u8;
        }
        // `slots` holds one slot per worker of the pool this search runs, which
        // can be smaller than the build's `threads`.
        #[cfg(lazy_smp_spread)]
        {
            self.spread.seed(index);
            self.spread.set_workers(slots.len());
        }
        #[cfg(lazy_smp_defer)]
        self.defer.set_workers(slots.len());
    }

    /// Turn this worker's move-order spread on or off, for a pool of `workers`.
    /// A worker starts with the switch as the config's `lazy_smp_spread` sets
    /// it, and alone; [`Self::set_best_move_tally`] tells it the pool's size.
    /// One worker never spreads.
    #[cfg(lazy_smp_spread)]
    pub fn set_spread(&mut self, on: bool, workers: usize) {
        self.spread.set(on, workers);
    }

    /// Turn this worker's putting off of busy moves on or off, for a pool of
    /// `workers`. A worker starts with the switch as the config's
    /// `lazy_smp_defer` sets it, and alone; [`Self::set_best_move_tally`] tells
    /// it the pool's size. One worker never puts a move off.
    #[cfg(lazy_smp_defer)]
    pub fn set_defer(&mut self, on: bool, workers: usize) {
        self.defer.set(on, workers);
    }

    /// Look `plies` back for a repetition from the next search on, `0` for no
    /// limit. A worker starts with the config's `repetition_look_back`.
    #[cfg(repetition_index)]
    pub fn set_repetition_look_back(&mut self, plies: u32) {
        self.repetition_look_back = plies;
    }

    /// Index the game up to `root`, the root of the search about to start, for
    /// [`Self::repetition`]. A window of sixteen is the plain walk, which never
    /// reads the index, so then nothing is built.
    #[cfg(repetition_index)]
    fn index_root(&mut self, root: &Position) {
        if self.repetition_look_back != 16 {
            self.histories.history_index.rebuild(root);
        }
    }

    /// The repetition verdict on `pos`, over the window this build looks back
    /// across.
    #[inline(always)]
    fn repetition(&self, pos: &Position) -> RepetitionState {
        #[cfg(repetition_index)]
        {
            pos.is_repetition_within(self.repetition_look_back, &self.histories.history_index)
        }
        #[cfg(not(repetition_index))]
        {
            pos.is_repetition()
        }
    }

    /// Install the raw `MultiPV` option value for this `go`. Every worker gets
    /// it, since it shapes the search; the main worker's PV *output* is
    /// configured separately by [`Self::set_pv_output`]. Both clamp to
    /// `rootMoves.size()` inside [`Self::run_worker`]. Leave unset (`1`) for the
    /// fixed-depth parity path.
    ///
    /// A build that prints no search `info` line has no way to report a second
    /// PV, so without that feature the root is single-line and there is no value
    /// to install.
    #[cfg(feature = "verbose2")]
    pub fn set_multi_pv(&mut self, multi_pv: usize) {
        self.multi_pv = multi_pv.max(1);
    }

    /// Install how wide this search reads its root. Every worker of a search
    /// gets the same width. A `go` leaves it unset, which is
    /// [`RootWidth::SINGLE_LINE`]; the search that thinks past the engine's
    /// reply installs the configured one.
    pub fn set_root_width(&mut self, width: RootWidth) {
        self.root_width = HeldWidth::new(width);
    }

    /// Install the PV-output configuration and sink on the **main** worker (the
    /// reference `main_manager()->pv()` owner). This sets the computed PV
    /// interval and the consideration / fail-LH flags, and seeds `lastPvInfoTime`
    /// to the search start; the sink receives each emitted line.
    #[cfg(feature = "verbose2")]
    pub fn set_pv_output(&mut self, config: PvOutputConfig, sink: Box<dyn PvSink>) {
        self.last_pv_info_time = config.start_time;
        self.pv_config = Some(config);
        self.pv_sink = Some(sink);
    }

    /// Consume the driver and return its history tables so the session can carry
    /// them into the next `go`.
    pub fn into_histories(self) -> WorkerHistories {
        self.histories
    }

    /// The `callsCnt` reload value: the standard [`CHECK_INTERVAL`], but capped
    /// tighter when a small node ceiling is set so the check rate stays at
    /// least ~0.1% of the ceiling.
    fn calls_reset(&self) -> i32 {
        // No node ceiling can be set without `verbose2`, so nothing tightens the
        // checkpoint rate there.
        #[cfg(feature = "verbose2")]
        if let Some(n) = self.control.node_limit {
            return CHECK_INTERVAL.min((n.get() / 1024) as i32).max(1);
        }
        CHECK_INTERVAL
    }

    /// The aggregate node count against the `go nodes` ceiling
    /// (`worker.threads.nodes_searched()`). With a tally installed, sum every
    /// worker's published slot; without one (single worker), this worker's own
    /// `nodes` is the whole search, so the two forms agree bit-for-bit.
    ///
    /// Only the main worker is given a ceiling, and its tally is the whole pool.
    #[cfg(feature = "verbose2")]
    fn counted_nodes(&self) -> u64 {
        match self.node_tally {
            Some(slots) => slots.iter().map(|s| s.load(Ordering::Relaxed)).sum(),
            None => self.nodes,
        }
    }

    /// Fold this iteration's best-move changes into `tot` and reset for the
    /// next. Under Lazy-SMP only the main worker sums and zeroes every slot; a
    /// helper reaches this too, but takes the no-op arm and leaves its slot for
    /// the main worker to read and clear.
    fn fold_best_move_changes(&mut self, tot: &mut f64) {
        match self.best_move_tally {
            Some(slots) if self.main_tally => {
                for s in slots {
                    *tot += s.swap(0, Ordering::Relaxed) as f64;
                }
            }
            Some(_) => {}
            None => {
                *tot += self.best_move_changes;
                self.best_move_changes = 0.0;
            }
        }
    }

    /// The reference `SearchManager::check_time`: count down and, once per
    /// [`Self::calls_reset`] calls, consult the stop flag, then — only once the
    /// first iteration has completed — the `movetime`, node and `search_end`
    /// stops, or set `search_end` from `maximum()` under time management.
    fn check_time(&mut self) {
        if self.stopped {
            return;
        }
        self.calls_cnt -= 1;
        if self.calls_cnt > 0 {
            return;
        }
        self.calls_cnt = self.calls_reset();

        // Publish this worker's node count so the main worker's aggregate ceiling
        // (and the final aggregated `nodes` output) can sum every worker — the
        // reference reads a per-worker atomic `nodes` each checkpoint
        // (`threads.nodes_searched()`); this port publishes at the checkpoint
        // rather than per node. Inert (no slot) on the single-worker path, and
        // absent entirely where neither reader of the sum exists.
        #[cfg(feature = "verbose2")]
        if let Some(slots) = self.node_tally {
            slots[0].store(self.nodes, Ordering::Relaxed);
        }

        // The external stop flag (GUI `stop` / `gameover`, the coordinator's
        // end-of-search signal, a sibling worker's abort) is honoured
        // unconditionally — it is how every worker terminates.
        if let Some(flag) = &self.control.stop
            && flag.load(Ordering::Relaxed)
        {
            self.stopped = true;
            return;
        }

        // The reference gates every time / node stop on completedDepth >= 1 so a
        // `bestmove` is always backed by at least one finished iteration.
        if self.completed_depth < 1 {
            return;
        }

        // 3./4. movetime elapsed or node ceiling reached ⇒ stop immediately.
        #[cfg(feature = "verbose2")]
        if let Some(limit) = self.control.node_limit
            && self.counted_nodes() >= limit.get()
        {
            self.request_abort();
            return;
        }
        let Some(tc) = self.control.time.as_ref() else {
            return;
        };
        let now = Instant::now();
        let elapsed = tc.tm.elapsed_from(now);
        #[cfg(feature = "verbose2")]
        if let Some(movetime) = tc.movetime
            && elapsed >= movetime
        {
            self.request_abort();
            return;
        }
        // 5. the TimeManagement-decided end time has arrived ⇒ stop immediately.
        if tc.tm.search_end != 0 {
            if tc.tm.search_end <= elapsed {
                self.request_abort();
            }
            return;
        }
        // 1./2. the maximum think time is exceeded ⇒ round up to a whole second
        // via set_search_end rather than stopping now. Every `go` a build
        // without `verbose2` accepts is clock-managed, so there the budget alone
        // decides. The budget is measured in credited time; the end time it
        // fixes is real.
        let credited = tc.tm.credited_elapsed_from(now);
        #[cfg(feature = "verbose2")]
        let past_budget = tc.use_time_management && credited > tc.tm.maximum();
        #[cfg(not(feature = "verbose2"))]
        let past_budget = credited > tc.tm.maximum();
        if past_budget {
            self.control
                .time
                .as_mut()
                .expect("time control present")
                .tm
                .set_search_end(credited);
        }
    }

    /// Latch the abort and, if a shared stop flag is installed, publish it so an
    /// external observer (the driver's next `stop`/`quit`, a sibling reader) sees
    /// the search has terminated itself — the reference's
    /// `threads.stop = threads.abortedSearch = true`.
    fn request_abort(&mut self) {
        self.stopped = true;
        if let Some(flag) = &self.control.stop {
            flag.store(true, Ordering::Relaxed);
        }
    }

    /// Run quiescence search on `pos` from ply 0.
    ///
    /// `alpha < beta` must hold, and `pv_node || alpha == beta - 1` (the
    /// reference's `ASSERT_LV3`). `read_tt == false` reproduces the `ReadTT ==
    /// false` template instantiation (TT hits are ignored, but writes still
    /// happen).
    pub fn run(
        &mut self,
        pos: &mut Position,
        alpha: Value,
        beta: Value,
        pv_node: bool,
        read_tt: bool,
    ) -> QSearchOutcome {
        debug_assert!(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE);
        debug_assert!(pv_node || alpha == beta - 1);

        self.nodes = 0;
        self.sel_depth = 0;
        self.pv_node = pv_node;
        self.set_read_tt(read_tt);
        self.set_root_side(pos.side_to_move());
        self.stopped = false;
        self.calls_cnt = CHECK_INTERVAL;

        for cell in self.stack.iter_mut() {
            cell.current_move = None;
            cell.tt_pv = false;
            cell.pv.clear();
        }

        // Seed the root accumulator; every node below derives from it.
        self.seed_accumulator(pos);
        #[cfg(repetition_index)]
        self.index_root(pos);

        let value =
            self.lend_picker_scratch(|this, scratch| this.qsearch(pos, 0, alpha, beta, scratch));

        QSearchOutcome {
            value,
            nodes: self.nodes,
            pv: self.stack[Self::si(0)].pv.clone(),
            sel_depth: self.sel_depth,
        }
    }

    /// Stack index for search ply `ply`.
    #[inline]
    fn si(ply: i32) -> usize {
        STACK_BASE + ply as usize
    }

    /// Take the picker buffer for one node, leaving the rest for whatever it
    /// searches. The lender sizes the array so a live picker always finds one.
    #[inline]
    fn split_picker_scratch(
        scratch: &mut [PickerScratch],
    ) -> (&mut PickerScratch, &mut [PickerScratch]) {
        scratch
            .split_first_mut()
            .expect("a picker buffer per live picker")
    }

    /// Run `f` with the picker buffers (see [`Self::picker_scratch`]) lent to
    /// it, and take them back when it returns. Every entry into the recursion
    /// goes through here.
    fn lend_picker_scratch<R>(
        &mut self,
        f: impl FnOnce(&mut Self, &mut [PickerScratch]) -> R,
    ) -> R {
        let mut scratch = self
            .picker_scratch
            .take()
            .expect("the picker buffers are lent to one run at a time");
        let value = f(self, &mut scratch[..]);
        self.picker_scratch = Some(scratch);
        value
    }

    /// The current node's live accumulator (`acc_stack[acc_depth]`), read by the
    /// six evaluation sites via [`yorkie_eval::evaluate_with`].
    #[inline]
    fn acc(&self) -> &Accumulator {
        &self.acc_stack[self.acc_depth]
    }

    /// The node's NNUE static evaluation, read through the differentially
    /// maintained accumulator. Every evaluation site in the search routes
    /// through here.
    ///
    /// When `verify_accumulator` is set — a test-only knob — it also
    /// asserts the differential result equals a from-scratch
    /// [`yorkie_eval::evaluate`] of the current position.
    ///
    /// Being the one site every evaluation reaches is also what makes it the
    /// place the per-game `noise` is added: a value the search derives for
    /// itself — a mate distance, a repetition verdict, the superior / inferior
    /// scores — never comes through here, and so carries no noise. For the
    /// same reason it is where the evaluations are counted.
    #[inline]
    fn static_eval(&mut self, pos: &Position) -> Value {
        #[cfg(feature = "verbose1")]
        {
            self.evals += 1;
        }
        let value = evaluate_with(self.net, self.acc(), pos);
        #[cfg(test)]
        if self.verify_accumulator {
            assert_eq!(
                value,
                yorkie_eval::evaluate(self.net, pos),
                "differential NNUE accumulator diverged from a full refresh",
            );
        }
        #[cfg(feature = "random")]
        let value = value + noise(pos.key(), self.random_seed, self.random_amplitude);
        value
    }

    /// Full-refresh the root accumulator into slot `0` and reset the do/undo
    /// depth. Every deeper node's accumulator is derived incrementally from
    /// here, so this must run once at the start of every fresh search.
    #[inline]
    fn seed_accumulator(&mut self, pos: &Position) {
        let net = self.net;
        self.acc_depth = 0;
        self.acc_stack[0].refresh(net, pos);
    }

    /// Derive the child accumulator from the current top and advance the depth,
    /// mirroring a `do_move`. `post_pos` is the position *after* the move, used
    /// only to rebuild a perspective whose own king moved; `delta` is the
    /// add/sub feature delta captured from the pre-move position.
    ///
    /// `mv` is the move just made; with `post_pos` it addresses the shared
    /// history entries the child reads, which are prefetched here.
    #[inline]
    fn push_accumulator(&mut self, post_pos: &Position, mv: Move, delta: &MoveDelta) {
        // The reference prefetches the child's TT cluster and its shared
        // history entries inside `do_move`, the moment the post-move keys are
        // known. This port's `Position` cannot reach either table, so the hints
        // go here instead — the accumulator seam every real-search `do_move`
        // funnels through, later than the reference but still well before the
        // child probes.
        TranspositionTable::shared()
            .prefetch(post_pos.key(), post_pos.side_to_move().index() as u8);
        self.histories
            .shared
            .prefetch_child(post_pos, mv.moved_piece_after(), mv.to_sq());

        let net = self.net;
        let d = self.acc_depth;
        let (parent_slots, child_slots) = self.acc_stack.split_at_mut(d + 1);
        Accumulator::derive_into_cached(
            &parent_slots[d],
            &mut child_slots[0],
            net,
            post_pos,
            delta,
            &mut self.finny,
        );
        self.acc_depth = d + 1;
    }

    /// Prefetch the two continuation-correction entries the child of `mv`,
    /// made from stack cell `s`, reads for its corrected evaluation. The child
    /// reads its `(ss-2)` and `(ss-4)` planes at the moved piece and its
    /// destination; before the move those cells are `s - 1` and `s - 3`, and
    /// both planes are already set, so this can run before `do_move`.
    ///
    /// The cells are bounds-checked indexes into the stack, and the entry is a
    /// bounds-checked index into the table.
    #[inline]
    fn prefetch_child_corrections(&self, s: usize, mv: Move) {
        let (pc, to) = (mv.moved_piece_after(), mv.to_sq());
        let cc = &self.histories.continuation_correction;
        cc.prefetch_at(self.stack[s - 1].cont_corr, pc, to);
        cc.prefetch_at(self.stack[s - 3].cont_corr, pc, to);
    }

    /// Pop the child accumulator, mirroring an `undo_move`. The parent slot is
    /// never mutated by [`Self::push_accumulator`], so the parent accumulator is
    /// restored simply by dropping back to it.
    #[inline]
    fn pop_accumulator(&mut self) {
        self.acc_depth -= 1;
    }

    /// `drawValueTable[rs][c]` (defaults, with the `REPETITION_DRAW` row
    /// overwritten from contempt at search start).
    ///
    /// A win by the opponent's perpetual check is a mate like any other, so
    /// under [`FLATTEN`] the engine's is flat; a flat value passes through the
    /// callers' [`value_from_tt`] unshifted.
    fn draw_value(&self, rs: RepetitionState, c: Color) -> Value {
        match rs {
            RepetitionState::None => VALUE_DRAW,
            RepetitionState::Win if FLATTEN && self.is_engine(c) => VALUE_MATE_FLAT,
            RepetitionState::Lose if FLATTEN && !self.is_engine(c) => -VALUE_MATE_FLAT,
            RepetitionState::Win => VALUE_MATE,
            RepetitionState::Lose => -VALUE_MATE,
            RepetitionState::Draw => {
                // The magnitude is the root side's compiled contempt; the side
                // asking only decides its sign.
                let contempt = match self.root_us {
                    Color::Black => DRAW_CONTEMPT_BLACK,
                    Color::White => DRAW_CONTEMPT_WHITE,
                };
                if c == self.root_us {
                    contempt
                } else {
                    -contempt
                }
            }
            // VALUE_SUPERIOR == VALUE_MAX_EVAL.
            RepetitionState::Superior => VALUE_MAX_EVAL,
            RepetitionState::Inferior => -VALUE_MAX_EVAL,
        }
    }

    /// Write a TT entry to the exact `slot` captured by the node's initial
    /// [`TranspositionTable::locate`] probe, reproducing the reference's single-
    /// `TTWriter`-per-node discipline. Writing by a fixed slot (rather than a
    /// re-probe) keeps the stored TT state bit-identical to the reference even
    /// when a child has since churned the cluster — the write-slot-drift fix.
    ///
    /// The entry takes the storing node's current `marks`, so they travel with
    /// the value they belong to.
    #[expect(clippy::too_many_arguments)]
    fn tt_store(
        &mut self,
        slot: TtSlot,
        key: u64,
        value: Value,
        pv: bool,
        bound: Bound,
        depth: i32,
        mv: Option<NonZeroU16>,
        eval: Value,
    ) {
        let generation = TranspositionTable::shared().generation();
        TranspositionTable::shared().write_at(
            slot,
            key,
            value,
            pv,
            bound,
            depth,
            mv,
            eval,
            generation,
            #[cfg(feature = "verbose3")]
            self.marks,
            #[cfg(feature = "verbose1")]
            &mut self.tt_updates,
        );
    }

    /// Count the bare static-eval store about to go into `slot` as a duplicate
    /// when the entry already holds this position's static eval — read before
    /// the store, so it is the entry as the store finds it.
    #[cfg(feature = "verbose1")]
    #[inline]
    fn count_eval_dup(&mut self, slot: TtSlot, key: u64) {
        self.eval_dups += u64::from(TranspositionTable::shared().holds_eval_at(slot, key));
    }

    /// Let the spread see the store of depth `depth` about to end a node: what
    /// `slot` holds for the position `key` before the store overwrites it.
    #[cfg(lazy_smp_spread)]
    #[inline]
    fn spread_note_store(&mut self, slot: TtSlot, key: u64, depth: i32) {
        if self.spread.active() {
            let found = TranspositionTable::shared().depth_at(slot, key);
            self.spread.note_store(key, found, depth, self.root_depth);
        }
    }

    /// The moment the search this worker belongs to started, which every
    /// worker of it times its iterations from.
    #[cfg(feature = "dupstats")]
    pub fn set_search_start(&mut self, start: Instant) {
        self.search_start = Some(start);
    }

    /// End the search when the first of its workers finishes, through `latch`,
    /// which every worker of the search is handed.
    #[cfg(all(feature = "dupstats", feature = "verbose3"))]
    pub fn set_first_finish(&mut self, latch: &'static crate::timeline::FirstFinish) {
        self.first_finish = Some(latch);
    }

    /// Who had stored the root of `pos` at `depth` or deeper, read before an
    /// iteration to `depth` starts. The read changes nothing in the table.
    #[cfg(feature = "dupstats")]
    fn root_entry(&self, pos: &Position, depth: i32) -> crate::timeline::RootEntry {
        let key = pos.key();
        let tt = TranspositionTable::shared();
        let (_, _, slot) = tt.locate(key, pos.side_to_move().index() as u8, |m16| {
            pos.to_move(m16).is_some()
        });
        let found = tt.depth_at(slot, key);
        let writer = found.and_then(|_| crate::dupstats::writer_of(key));
        crate::timeline::RootEntry::of(found, depth, writer, self.dup_writer_id())
    }

    /// Who this worker is, as the duplication statistics' writer table records
    /// a store.
    #[cfg(feature = "dupstats")]
    fn dup_writer_id(&self) -> crate::dupstats::WriterId {
        crate::dupstats::WriterId {
            generation: TranspositionTable::shared().generation(),
            worker: self.dup_worker,
            iteration: self.root_depth.clamp(0, u8::MAX as i32) as u8,
        }
    }

    /// Count the use of the entry the interior node of remaining depth
    /// `depth` found its position `key` in (`hit`), and note the entry's move
    /// for the comparison with a lone search. `cut` is whether the entry ends
    /// the node.
    #[cfg(feature = "dupstats")]
    fn channel_use(
        &mut self,
        key: u64,
        hit: bool,
        data: &yorkie_storage::TTData,
        depth: i32,
        cut: bool,
    ) {
        let move16 = if hit {
            data.move16.map_or(0, NonZeroU16::get)
        } else {
            0
        };
        self.channels
            .tt
            .note_move(self.root_depth, key, hit, move16);
        if hit {
            self.channels.tt.count(&crate::channels::Use {
                iteration: self.root_depth,
                depth,
                stored_depth: data.depth,
                bound: data.bound,
                cut,
                writer: crate::dupstats::writer_of(key),
                me: self.dup_writer_id(),
            });
        }
    }

    /// Enter the node in `at`, at remaining depth `depth`, whose position `key`
    /// names: push its path frame, take its place in the presence table when it
    /// is deep enough, and count what that found. The two holds are released
    /// when the returned guards drop — on whichever return leaves the node.
    #[cfg(feature = "dupstats")]
    fn dup_enter(
        &mut self,
        key: u64,
        depth: i32,
        at: crate::dupstats::Where,
    ) -> (
        crate::dupstats::PathFrame,
        Option<crate::dupstats::PresenceHold>,
    ) {
        use crate::dupstats::{PRESENCE_MIN_DEPTH, PresenceKind, enter_path, enter_presence};
        let frame = enter_path(self.nodes);
        let hold = (depth >= PRESENCE_MIN_DEPTH).then(|| {
            let (seen, hold) = enter_presence(key, self.dup_worker, depth);
            self.dup.count_presence(PresenceKind::Entered, at);
            if let Some(kind) = seen.kind() {
                self.dup.count_presence(kind, at);
            }
            hold
        });
        (frame, hold)
    }

    /// Classify the store of depth `depth` about to end the node in `at`, whose
    /// probe found its position when `probe_hit`, by what `slot` holds, and
    /// credit the node's subtree when another worker had already stored this
    /// position at least as deep. Returns what [`Self::dup_stored`] needs to
    /// tell, once the store is made, what it did to the entry.
    #[cfg(feature = "dupstats")]
    fn dup_store(
        &mut self,
        slot: TtSlot,
        key: u64,
        depth: i32,
        probe_hit: bool,
        at: crate::dupstats::Where,
    ) -> DupStore {
        use crate::dupstats::{StoreKind, Writer, classify, credit_subtree, writer_of};
        let found = TranspositionTable::shared().depth_and_age_at(slot, key);
        let previous = found.and_then(|_| writer_of(key));
        let kind = classify(
            found.map(|(d, _)| d),
            depth,
            probe_hit,
            previous,
            self.dup_writer_id(),
        );
        self.dup.count_store(kind, at);
        if kind == StoreKind::AlreadyThere(Writer::OtherWorker) {
            self.dup.dup_nodes += credit_subtree(self.nodes);
        }
        DupStore {
            updates_before: self.tt_updates,
            found,
            previous,
            depth,
            at,
        }
    }

    /// After a store [`Self::dup_store`] classified: count what it did to a
    /// deeper entry of the running search, and when it wrote the entry, record
    /// this worker as the entry's writer.
    #[cfg(feature = "dupstats")]
    fn dup_stored(&mut self, key: u64, before: DupStore) {
        let wrote = self.tt_updates != before.updates_before;
        let me = self.dup_writer_id();
        if let Some(overwrite) = crate::dupstats::classify_overwrite(
            before.found,
            before.depth,
            wrote,
            before.previous,
            me,
        ) {
            self.dup.count_overwrite(overwrite, before.at);
        }
        if wrote {
            let replaced = crate::dupstats::set_writer(key, me);
            self.dup.writer_replaced += u64::from(replaced);
        }
    }

    /// Hand back the marks of the value a sub-search just returned, restoring
    /// `node_marks` — the marks the calling node had built up before it — as
    /// the live ones.
    #[cfg(feature = "verbose3")]
    #[inline]
    fn child_marks(&mut self, node_marks: Held) -> Held {
        let child = self.held();
        self.restore_held(node_marks);
        child
    }

    /// The live marks and their causes, to set aside across a sub-search.
    #[cfg(feature = "verbose3")]
    #[inline]
    fn held(&self) -> Held {
        Held {
            marks: self.marks,
            causes: self.causes,
        }
    }

    /// Make `held` the live marks and causes again.
    #[cfg(feature = "verbose3")]
    #[inline]
    fn restore_held(&mut self, held: Held) {
        self.marks = held.marks;
        self.causes = held.causes;
    }

    /// Start a node unmarked and with no cause.
    #[cfg(feature = "verbose3")]
    #[inline]
    fn clear_held(&mut self) {
        self.marks = ValueMarks::NONE;
        self.causes = PathCauses::NONE;
    }

    /// Fold in the marks and causes of a child that moved the window of the
    /// node at `pos`. Only the causes standing before the node are kept: one at
    /// the node or below it is a repetition closed inside the node's subtree.
    #[cfg(feature = "verbose3")]
    #[inline]
    fn merge_held(&mut self, child: Held, pos: &Position) {
        self.marks.merge(child.marks);
        self.causes.merge_below(&child.causes, pos.timeline_index());
    }

    /// The value at `pos` is a repetition verdict: mark it, and record the
    /// earlier position the verdict matched.
    #[cfg(feature = "verbose3")]
    fn mark_repetition(&mut self, pos: &Position) {
        self.marks.path_dep = true;
        let found = PathCauses::of_repetition(pos, self.repetition_window());
        self.causes.merge_below(&found, usize::MAX);
    }

    /// The window [`Self::repetition`] judges over, as
    /// [`Position::repetition_origin`] takes it.
    #[cfg(feature = "verbose3")]
    fn repetition_window(&self) -> u32 {
        #[cfg(repetition_index)]
        {
            self.repetition_look_back
        }
        #[cfg(not(repetition_index))]
        {
            16
        }
    }

    /// A table entry's value became the node's: take the entry's marks. An
    /// entry stores no causes, so a path-dependent one makes the node's causes
    /// open — unknown, whichever search wrote the entry.
    #[cfg(feature = "verbose3")]
    fn take_entry_marks(&mut self, marks: ValueMarks) {
        self.marks = marks;
        self.causes = if marks.path_dep {
            PathCauses::OPEN
        } else {
            PathCauses::NONE
        };
    }

    /// Widen a stored 16-bit move against `pos` and validate it: the unique
    /// **legal** move whose `Move16` equals `move16`, or `None`.
    ///
    /// A generate-and-match form, kept as the oracle for the O(1)
    /// [`Position::to_move`] + [`Position::pseudo_legal`] chain production
    /// widens through. Both are repetition-blind, so the two agree on every
    /// move, including one that continues a perpetual check.
    #[cfg(test)]
    fn widen_tt_move(pos: &Position, move16: NonZeroU16) -> Option<Move> {
        let mut legal: Vec<Move> = Vec::new();
        pos.generate_legal_all(&mut legal);
        Self::select_tt_move(&legal, move16)
    }

    /// Select, from a position's already-generated `legal` moves, the unique one
    /// whose `Move16` fragment equals `move16`, or `None`. This is the only step
    /// that consumes the raw fragment, and it is a total `move16_of` comparison
    /// (no bit is decoded into a `Move`). Split out from [`widen_tt_move`] so the
    /// torn-entry totality test can drive all 65536 fragments
    /// against one generated list without re-generating per pattern.
    #[cfg(test)]
    fn select_tt_move(legal: &[Move], move16: NonZeroU16) -> Option<Move> {
        legal
            .iter()
            .copied()
            .find(|&m| m.move16_stored() == Some(move16))
    }

    /// Enter [`Self::qsearch`] at `ply` with the picker buffers lent to it —
    /// the entry the tests that drive a single node use.
    #[cfg(test)]
    fn qsearch_at(&mut self, pos: &mut Position, ply: i32, alpha: Value, beta: Value) -> Value {
        self.lend_picker_scratch(|this, scratch| this.qsearch(pos, ply, alpha, beta, scratch))
    }

    /// The core recursive qsearch. `scratch` holds the picker buffers still
    /// free: this node takes the first for its own picker and the moves it
    /// searches get the rest.
    fn qsearch(
        &mut self,
        pos: &mut Position,
        ply: i32,
        mut alpha: Value,
        beta: Value,
        scratch: &mut [PickerScratch],
    ) -> Value {
        let pv_node = self.pv_node;

        // The node starts unmarked; every return below leaves this holding the
        // marks of the value it returns.
        #[cfg(feature = "verbose3")]
        {
            self.clear_held();
        }

        // Poll the stop flag / hard deadline at the reference `check_time`
        // granularity. The reference only checks in the interior
        // `search`, but polling in qsearch too bounds the abort latency inside a
        // deep leaf tree. On abort the return value is discarded — the interior
        // caller unwinds immediately via its own post-move stop check — so any
        // value is fine. Inert on the parity path (`stopped` never latches).
        self.check_time();
        if self.stopped {
            return VALUE_DRAW;
        }

        // Step 1. Initialize node.
        if pv_node {
            self.stack[Self::si(ply)].pv.clear();
        }
        let in_check = pos.in_check();
        let mut move_count = 0;
        if pv_node && self.sel_depth < ply + 1 {
            self.sel_depth = ply + 1;
        }

        // Step 2. Immediate draw / max ply.
        let us = pos.side_to_move();
        let draw_type = self.repetition(pos);
        if draw_type != RepetitionState::None {
            // The judgement read the move history, so the value is path
            // dependent by construction — this is where the mark originates.
            #[cfg(feature = "verbose3")]
            {
                self.mark_repetition(pos);
            }
            if draw_type == RepetitionState::Draw {
                // Ordinary repetition: the ±1 dither.
                return self.draw_value(RepetitionState::Draw, us) + value_draw(self.nodes);
            } else {
                // Superior / inferior / perpetual-check: map to a
                // root-relative score.
                return value_from_tt(self.draw_value(draw_type, us), ply);
            }
        }
        // The reference's `depth <= -16 → draw` measure is `#if 0`: not ported.
        if ply >= MAX_PLY || is_past_move_limit(pos.ply()) {
            // The move limit is a rule of the tournament and could be set
            // differently; the `MAX_PLY` search horizon is not, so only the
            // first marks the value it produces.
            #[cfg(feature = "verbose3")]
            if ply < MAX_PLY {
                self.marks.move_limit = true;
            }
            return self.draw_value(RepetitionState::Draw, us) + value_draw(self.nodes);
        }

        // Step 3. Transposition table lookup.
        let pos_key = pos.key();
        let side = us.index() as u8;
        // Capture the entry location once (like the reference's Step-3
        // `ttWriter`); every write below targets this exact slot.
        let (found, tt_data, tt_slot) =
            TranspositionTable::shared().locate(pos_key, side, |m16| pos.to_move(m16).is_some());
        #[cfg(feature = "dupstats")]
        let tt_data = crate::channels::mask_pv(tt_data);
        let tt_hit = found && self.reads_tt();
        // Widened O(1), without legal-move generation; the MovePicker's TT
        // stage re-validates with `pseudo_legal` + `is_legal`.
        let tt_move = if tt_hit {
            tt_data.move16.and_then(|m16| pos.to_move(m16))
        } else {
            None
        };
        let tt_value = if tt_hit {
            value_from_tt(tt_data.value, ply)
        } else {
            VALUE_NONE
        };
        let pv_hit = tt_hit && tt_data.is_pv;

        #[cfg(feature = "dupstats")]
        let tt_value_kept = tt_value;
        #[cfg(feature = "dupstats")]
        let tt_value = crate::channels::mask_cut_value(tt_value);
        if !pv_node
            && tt_data.depth >= DEPTH_QS
            && is_valid(tt_value)
            && bound_matches(tt_data.bound, tt_value >= beta)
        {
            // The stored value re-enters the search as this node's value, so
            // it brings its marks with it.
            #[cfg(feature = "verbose3")]
            {
                self.take_entry_marks(tt_data.marks);
            }
            return tt_value;
        }
        #[cfg(feature = "dupstats")]
        let (tt_move, _, tt_value, tt_data) =
            crate::channels::mask_after_cut(tt_move, false, tt_value_kept, tt_data);

        // Step 4. Static evaluation.
        let mut unadjusted_static_eval = VALUE_NONE;
        let mut best_value: Value;
        let futility_base: Value;

        if in_check {
            // Every evasion is generated, so start from -infinity.
            best_value = -VALUE_INFINITE;
            futility_base = -VALUE_INFINITE;
        } else {
            let correction_value = self.correction_value(pos, ply);

            if tt_hit {
                unadjusted_static_eval = tt_data.eval;
                #[cfg(feature = "verbose1")]
                {
                    self.eval_tt_hits += u64::from(is_valid(unadjusted_static_eval) && !pv_node);
                }
                if !is_valid(unadjusted_static_eval) {
                    unadjusted_static_eval = self.static_eval(pos);
                } else if pv_node {
                    // The reference build has USE_CLASSIC_EVAL active, which
                    // re-evaluates on PV nodes.
                    unadjusted_static_eval = self.static_eval(pos);
                }
                best_value = to_corrected_static_eval(unadjusted_static_eval, correction_value);

                if is_valid(tt_value)
                    && !is_decisive(tt_value)
                    && bound_matches(tt_data.bound, tt_value > best_value)
                {
                    best_value = tt_value;
                }
            } else {
                // 🌈 1-ply mate check, only when the TT missed. Under
                // `FLATTEN` the mate must also land inside the move limit, as
                // the flat value claims a game won rather than a distance.
                if (!FLATTEN || mate_lands_inside_move_limit(pos.ply()))
                    && let Some(mate_move) = pos.mate_1ply()
                {
                    best_value = self.mate_for(us, ply + 1);
                    // The returned value counts plies from the root; the
                    // stored one counts them from this position, so a probe
                    // at any other ply reads back the same mate in one. The
                    // quiescence search never sets its stack cell's `tt_pv`,
                    // so the PV flag comes from this node's own probe.
                    self.tt_store(
                        tt_slot,
                        pos_key,
                        value_to_tt(best_value, ply),
                        pv_hit,
                        Bound::Exact,
                        DEPTH_QS,
                        move16_of(Some(mate_move)),
                        unadjusted_static_eval,
                    );
                    return best_value;
                }

                unadjusted_static_eval = self.static_eval(pos);
                best_value = to_corrected_static_eval(unadjusted_static_eval, correction_value);
            }

            if best_value >= beta {
                if !is_decisive(best_value) {
                    best_value = (best_value + beta) / 2;
                }
                if !tt_hit {
                    #[cfg(feature = "verbose1")]
                    self.count_eval_dup(tt_slot, pos_key);
                    self.tt_store(
                        tt_slot,
                        pos_key,
                        value_to_tt(best_value, ply),
                        false,
                        Bound::Lower,
                        DEPTH_UNSEARCHED,
                        None,
                        unadjusted_static_eval,
                    );
                }
                return best_value;
            }

            if best_value > alpha {
                alpha = best_value;
            }

            // The reference's `futilityBase` is built from `ss->staticEval`,
            // which the `ttValue` refinement above does not touch — so it is
            // recomputed here rather than read back off `best_value`.
            futility_base = to_corrected_static_eval(unadjusted_static_eval, correction_value)
                + params::QS_FUTILITY_MARGIN.get();
        }

        // Step 5-8. Move loop.
        //
        // `(ss-1)` down to `(ss-6)` are read here, so one window carries the
        // bound for the whole group: the window's length is the constant 6,
        // against which the offsets into it are checked at compile time.
        let s = Self::si(ply);
        let recent = &self.stack[s - 6..s];
        let prev_sq = recent[5]
            .current_move
            .filter(|m| m.is_ok())
            .map(Move::to_sq);

        // `contHist[] = {(ss-1)->continuationHistory}`. The qsearch evasion
        // score reads plane `[0]`, the previous ply's REAL continuation plane;
        // with untouched tables it holds the uniform -523 fill, so the depth-1
        // evasion ordering is a constant shift.
        let cont_planes: [ContPlane; 6] = std::array::from_fn(|i| recent[5 - i].cont_hist);
        let (slot, scratch) = Self::split_picker_scratch(scratch);
        #[cfg(feature = "dupstats")]
        let picker_tt_move = crate::channels::mask_order(tt_move, false);
        #[cfg(not(feature = "dupstats"))]
        let picker_tt_move = tt_move;
        let mut mp = MovePicker::new_qsearch(pos, picker_tt_move, cont_planes, slot);

        let mut best_move: Option<Move> = None;

        while let Some(mv) = mp.next_move(pos, &self.histories) {
            let gives_check = pos.gives_check(mv);
            // The reference's `capture_stage` is plain `capture` here.
            let capture = !mv.is_drop() && pos.board().get(mv.to_sq()).is_some();
            move_count += 1;

            // Step 6. Pruning, only while not already losing.
            if !is_loss(best_value) {
                if !gives_check && Some(mv.to_sq()) != prev_sq && !is_loss(futility_base) {
                    if move_count > 2 {
                        continue;
                    }
                    let futility_value =
                        futility_base + pos.board().get(mv.to_sq()).map_or(0, piece_value);
                    if futility_value <= alpha {
                        best_value = best_value.max(futility_value);
                        continue;
                    }
                    if !pos.see_ge(mv, alpha - futility_base) {
                        best_value = best_value.max(alpha.min(futility_base));
                        continue;
                    }
                }

                // Skipping non-captures also drops a quiet TT move.
                if !capture {
                    continue;
                }

                if !pos.see_ge(mv, params::QS_SEE_MARGIN.get()) {
                    continue;
                }
            }

            // Step 7. Make and search. The accumulator delta must be taken
            // from the pre-move position.
            let acc_delta = MoveDelta::from_move(pos, mv);
            self.nodes += 1; // the search's sole node-count increment
            #[cfg(feature = "dupstats")]
            self.node_hist
                .count(ply + 1, 0, crate::dupstats::NodeKind::Quiescence);
            self.prefetch_child_corrections(Self::si(ply), mv);
            let undo = pos.do_move_with_check(mv, gives_check);
            // The reference sets these inside `do_move`, for every move,
            // qsearch's included. A deeper node reads the
            // continuation-correction plane at `(ss-2)` / `(ss-4)`, so a
            // qsearch ply that left them stale would feed a wrong `cntcv` — and
            // through it a wrong corrected eval — once the tables warm up.
            let moved = mv.moved_piece_after();
            self.stack[Self::si(ply)].current_move = Some(mv);
            self.stack[Self::si(ply)].cont_hist =
                ContinuationHistory::plane_index(in_check, capture, moved, mv.to_sq());
            self.stack[Self::si(ply)].cont_corr =
                ContinuationCorrectionHistory::plane_index(moved, mv.to_sq());
            self.push_accumulator(pos, mv, &acc_delta);
            #[cfg(feature = "verbose3")]
            let node_marks = self.held();
            let value = -self.qsearch(pos, ply + 1, -beta, -alpha, scratch);
            #[cfg(feature = "verbose3")]
            let child_marks = self.child_marks(node_marks);
            pos.undo_move(mv, undo);
            self.pop_accumulator();

            // An abort that fired inside this child's subtree makes `value`
            // untrustworthy; bail immediately rather than fold it in and keep
            // looping (the interior caller discards this node's value on abort).
            // Inert on the parity path.
            if self.stopped {
                return best_value;
            }

            // Step 8. New best move.
            if value > best_value {
                best_value = value;
                if value > alpha {
                    // The child moved this node's window, so its marks become
                    // this node's too.
                    #[cfg(feature = "verbose3")]
                    self.merge_held(child_marks, pos);
                    best_move = Some(mv);
                    if pv_node {
                        self.update_pv(ply, mv);
                    }
                    if value < beta {
                        alpha = value;
                    } else {
                        break; // fail high
                    }
                    // Nothing outranks a flat mate; see the interior search.
                    if FLATTEN && value == VALUE_MATE_FLAT {
                        break;
                    }
                } else if FLATTEN && pv_node && value == -VALUE_MATE_FLAT {
                    // A first move the engine mates, kept as the PV; see the
                    // interior search.
                    self.update_pv(ply, mv);
                }
            }
        }

        // Step 9. Mate check + tail write. In check with no legal move is
        // checkmate; the reference uses the `moveCount == 0` form, not
        // Stockfish's `bestValue == -VALUE_INFINITE`.
        if in_check && move_count == 0 {
            return self.mated_for(us, ply);
        }

        if !is_decisive(best_value) && best_value > beta {
            best_value = (best_value + beta) / 2;
        }

        self.tt_store(
            tt_slot,
            pos_key,
            value_to_tt(best_value, ply),
            pv_hit,
            if best_value >= beta {
                Bound::Lower
            } else {
                Bound::Upper
            },
            DEPTH_QS,
            move16_of(best_move),
            unadjusted_static_eval,
        );

        best_value
    }

    /// `ss->pv->update(move, (ss+1)->pv)`: the node's PV becomes `move`
    /// followed by the child's PV.
    fn update_pv(&mut self, ply: i32, mv: Move) {
        // `split_at_mut` hands out disjoint borrows of this node's cell and
        // its child's, so the child PV is copied into the parent's
        // preallocated buffer without allocating.
        let parent = Self::si(ply);
        let (head, tail) = self.stack.split_at_mut(parent + 1);
        let cell = &mut head[parent];
        let child_pv = &tail[0].pv;
        cell.pv.clear();
        cell.pv.push(mv);
        cell.pv.extend_from_slice(child_pv);
    }
}

// `run_root` drives the reference's `iterative_deepening` loop, entering the
// shared [`QSearch::search`] body at the root for each iteration and aspiration
// re-search. It reuses the same `QSearch`, so the interior search and the child
// qsearch it recurses into share all of that state.

impl<N: NetworkParams> QSearch<N> {
    /// Run the single-threaded, `MultiPV == 1`, book-free root path on `pos` to
    /// `limit_depth`, reproducing the reference's `start_searching` →
    /// `iterative_deepening` control flow. The caller must have sized the
    /// transposition table.
    ///
    /// This is the fixed-depth entry point: the caller names the depth, and the
    /// reference-captured parity fixtures are compared against it at each of
    /// theirs. That is a source for a ceiling in every build, which is why
    /// `limit_depth` here and on [`Self::run_worker`] carries no feature — unlike
    /// a `go`'s ceiling, which arrives only from clauses and config keys the
    /// Protocol layer gates.
    pub fn run_root(&mut self, pos: &Position, limit_depth: i32) -> RootOutcome {
        // Advance the TT generation before searching. The Lazy-SMP driver
        // hoists this bump out — one per `go`, before any helper launches — and
        // calls [`Self::run_worker`] directly; on this single-worker path it
        // stays inline.
        TranspositionTable::shared().new_search();

        // The root-move list is built once from the legal moves (the reference's
        // `start_thinking`). No legal move ⇒ `bestmove resign` with mated_in(1).
        let root_moves = generate_root_moves(pos);
        if root_moves.is_empty() {
            return RootOutcome {
                best_move: Move::resign(),
                score: mated_in(1),
                nodes: 0,
                pv: vec![Move::resign()],
                depth: 0,
                sel_depth: 0,
                kind: RootKind::Resign,
            };
        }

        // Declaration win at the root ⇒ `bestmove win` with mate_in(1). (1-ply
        // mate is intentionally *not* probed at the root.) This direct path is
        // the fixed-depth parity route and is only ever driven with the point /
        // `None` rules, for which `declaration_win` returns `Move::win()` or
        // nothing; the `TryRule` root shortcut (which emits an actual king move)
        // is owned by the driver's coordinator (`run_coordinated`), not here.
        if let Some(mv) = declaration_win(pos, &self.entering_king) {
            debug_assert_eq!(
                mv,
                Move::win(),
                "run_root is only driven with point/None entering-king rules"
            );
            // The score below is the declaration rule's, so it carries that
            // rule's mark like any other value derived through the check.
            #[cfg(feature = "verbose3")]
            {
                self.marks = ValueMarks {
                    decl_rule: true,
                    ..ValueMarks::NONE
                };
                self.causes = PathCauses::NONE;
            }
            self.set_root_side(pos.side_to_move());
            return RootOutcome {
                best_move: Move::win(),
                score: self.mate_for(pos.side_to_move(), 1),
                nodes: 0,
                pv: vec![Move::win()],
                depth: 0,
                sel_depth: 0,
                kind: RootKind::DeclarationWin,
            };
        }
        // (No book probe: no book.)

        // The single worker runs the whole iterative deepening.
        let result = self.run_worker(pos, root_moves, limit_depth);

        // --- back in start_searching (bestThread == this for a single worker) ---
        let best = result.best;

        RootOutcome {
            best_move: best.mv,
            score: best.uci_score,
            nodes: result.nodes,
            pv: best.pv.clone(),
            // At least 1: an abort during iteration 1 leaves `completed_depth`
            // zero, but a `bestmove` still went out at (partial) depth 1.
            depth: result.completed_depth.max(1),
            sel_depth: best.sel_depth,
            kind: RootKind::Normal,
        }
    }

    /// Run one Lazy-SMP worker's iterative deepening. The main worker and every
    /// helper call this on their own `QSearch`, each with its own copy of
    /// `root_moves`.
    ///
    /// **The caller must already have bumped the TT generation exactly once for
    /// this `go`** and built `root_moves`. Every time-management block is gated
    /// on the installed [`SearchControl`], and a helper's is stop-only.
    ///
    /// `limit_depth` is the caller's ceiling and is not feature-gated; see
    /// [`Self::run_root`] for why. A driver that has no ceiling to impose passes
    /// the search's own maximum, which the `rootDepth + 1 < MAX_PLY` guard below
    /// reaches on its own.
    pub fn run_worker(
        &mut self,
        root_pos: &Position,
        mut root_moves: Vec<RootMove>,
        limit_depth: i32,
    ) -> WorkerResult {
        self.run_worker_on(root_pos, &mut root_moves, limit_depth)
    }

    /// [`Self::run_worker`] on a root-move list the caller keeps: once the
    /// search is over, each move in it carries the `effort` spent on its
    /// subtree.
    ///
    /// The list is left in the search's order, with one exception: when the
    /// search was stopped before its first iteration completed, the best move
    /// so far is moved out of it into the result's `best`, its effort with it.
    pub fn run_worker_on(
        &mut self,
        root_pos: &Position,
        root_moves: &mut Vec<RootMove>,
        limit_depth: i32,
    ) -> WorkerResult {
        self.set_root_side(root_pos.side_to_move());
        self.set_read_tt(true);
        self.nodes = 0;
        #[cfg(feature = "verbose1")]
        {
            self.tt_updates = 0;
            self.eval_dups = 0;
            self.evals = 0;
            self.eval_tt_hits = 0;
        }
        #[cfg(feature = "dupstats")]
        {
            self.dup = crate::dupstats::DupCounts::ZERO;
            crate::dupstats::clear_path();
            let start = self.search_start.unwrap_or_else(Instant::now);
            self.timeline.reset(self.dup_worker, start);
            *self.node_hist = crate::dupstats::NodeHist::ZERO;
            yorkie_storage::cut::set_worker(self.dup_worker);
            self.channels.reset(self.dup_worker);
        }
        #[cfg(lazy_smp_spread)]
        self.spread.reset();
        #[cfg(lazy_smp_defer)]
        self.defer.reset();
        self.sel_depth = 0;
        self.last_iteration_pv.clear();
        // Fresh abort state for this `go`; the counter is seeded so the first
        // checkpoint lands a full interval in (see [`CHECK_INTERVAL`]).
        self.stopped = false;
        self.calls_cnt = CHECK_INTERVAL;
        // Fresh time-management bookkeeping for this `go` (reference
        // `pre_start_searching` / per-thread reset).
        self.completed_depth = 0;
        self.best_move_changes = 0.0;
        // lowPlyHistory is refilled to 98 per `go`.
        self.histories.low_ply.fill(98);

        // Full search-stack reset: a `static_eval` left over from a previous
        // `go` would flip `improving`, so every cell returns to the sentinel
        // state the interior search's improving / hindsight / continuation
        // logic assumes.
        for cell in self.stack.iter_mut() {
            // Reset to the pristine sentinel state while preserving the
            // preallocated `pv` buffer so a later PV update never reallocates.
            let mut pv = std::mem::take(&mut cell.pv);
            pv.clear();
            *cell = SearchStackCell::default();
            cell.pv = pv;
        }
        for (i, cell) in self.stack.iter_mut().enumerate().skip(STACK_BASE) {
            cell.ply = (i - STACK_BASE) as i32;
        }

        // A private working copy; the caller's position is never mutated.
        let mut work = root_pos.clone();
        #[cfg(repetition_index)]
        self.index_root(&work);

        // Seed this worker's accumulator stack from the root once. `search<Root>`
        // restores `work` to the root after every iteration and never mutates
        // slot 0 (it is only ever the derivation source), so this stays valid for
        // the whole iterative-deepening loop; `acc_depth` is reset to 0 before
        // each root search below for good measure.
        self.seed_accumulator(&work);

        // `searchAgainCounter` is incremented only when `increaseDepth` is
        // false, and `increaseDepth` is cleared only inside the time-managed
        // block, which a fixed-depth `go` never enters — so on the parity path
        // both stay at their initial values.
        let mut search_again_counter: i32 = 0;
        let mut increase_depth = true;
        // Reference iterative-deepening locals: the eval-stability time
        // reduction, the aged best-move-change statistic, the per-iteration
        // best-value ring, and the depth at which the best move last changed.
        let mut time_reduction: f64 = 1.0;
        let mut tot_best_move_changes: f64 = 0.0;
        let mut iter_idx: usize = 0;
        let mut last_best_move_depth: i32 = 0;

        // Main-thread persistent inputs for the time-management block, carried
        // in via the time control; the fixed-depth / helper path uses the
        // first-move-of-a-game sentinels, but never reads them (the block is
        // time-gated).
        let (best_previous_score, best_previous_average_score, previous_time_reduction) =
            match &self.control.time {
                Some(tc) => (
                    tc.best_previous_score,
                    tc.best_previous_average_score,
                    tc.previous_time_reduction,
                ),
                None => (VALUE_INFINITE, VALUE_INFINITE, 0.85),
            };
        // `threads.size()` — the divisor of the best-move instability factor,
        // and the worker count this binary was built with. The one command that
        // rebuilds the pool at another size searches to a fixed depth, node
        // count or move time, none of which reaches the block below.
        let n_threads = crate::config::THREADS.max(1) as usize;
        let iter_seed = if best_previous_score == VALUE_INFINITE {
            0
        } else {
            best_previous_score
        };
        let mut iter_value = [iter_seed; 4];

        // The reference's Skill-driven `max(multiPV, 4)` bump is skipped:
        // Skill is disabled in the non-Stockfish build.
        #[cfg(feature = "verbose2")]
        let multi_pv = self.multi_pv.min(root_moves.len()).max(1);

        // The last *completed* iteration's root move — the stable result an
        // aborted search rolls back to. Every iteration after the first copies
        // into the buffer this one is holding.
        let mut completed_best: Option<RootMove> = None;
        // The coordinator's final-PV fallback is their only reader, so a build
        // that prints no PV neither keeps nor copies them.
        #[cfg(feature = "verbose2")]
        let mut completed_lines: Vec<RootMove> = Vec::new();
        let mut completed_depth = 0;
        // Whether the current iteration's final PV was emitted to the GUI.
        #[cfg(feature = "verbose2")]
        let mut uci_pv_sent = false;

        let mut root_depth = 0;
        // The reference guards on the *post-increment* depth
        // (`while (++rootDepth < MAX_PLY && ... && rootDepth <=
        // limits.depth)`), hence the off-by-one shape of the bounds here.
        while root_depth + 1 < MAX_PLY && !self.stopped && root_depth < limit_depth {
            root_depth += 1;

            // Age out the PV-variability metric each iteration and, when the
            // time block cleared `increase_depth`, count a repeated depth.
            tot_best_move_changes /= 2.0;
            if !increase_depth {
                search_again_counter += 1;
            }

            #[cfg(feature = "dupstats")]
            {
                let entry = self.root_entry(root_pos, root_depth);
                self.timeline.begin(root_depth, entry);
            }

            // The aspiration seed for the next iteration; all
            // `-VALUE_INFINITE` on the first.
            for rm in root_moves.iter_mut() {
                rm.previous_score = rm.score;
            }

            #[cfg(feature = "verbose2")]
            {
                uci_pv_sent = false;
            }

            // The iteration's last search value, read by the early-mate break
            // below.
            #[cfg(feature = "verbose2")]
            let mut iter_best_value = -VALUE_INFINITE;

            // The MultiPV loop. A second PV line can only be reported by a
            // build that prints search `info` lines, so without that feature the
            // root is single-line: one aspiration search, on line 0, with no
            // index and no finished head to step over.
            #[cfg(feature = "verbose2")]
            for pv_idx in 0..multi_pv {
                self.pv_idx = pv_idx;
                iter_best_value = self.aspiration_search(
                    &mut work,
                    root_moves,
                    root_depth,
                    search_again_counter,
                    root_pos,
                    pv_idx,
                    multi_pv,
                );

                // A later line may have out-scored an earlier one.
                root_moves[..pv_idx + 1].sort_by(root_move_order);

                if self.pv_sink.is_some()
                    && (self.stopped
                        || pv_idx + 1 == multi_pv
                        || self.aggregate_nodes() > 10_000_000)
                    && !(self.stopped && is_loss(root_moves[0].uci_score))
                    && self.pv_interval_elapsed()
                {
                    self.emit_pv(root_pos, root_moves, pv_idx, root_depth, multi_pv);
                    self.last_pv_info_time = Instant::now();
                    uci_pv_sent = pv_idx + 1 == multi_pv;
                }

                if self.stopped {
                    break;
                }
            } // MultiPV loop
            #[cfg(not(feature = "verbose2"))]
            let iter_best_value =
                self.aspiration_search(&mut work, root_moves, root_depth, search_again_counter);

            #[cfg(feature = "verbose2")]
            {
                self.pv_idx = 0;
            }

            // The iteration is incomplete, so discard it and keep the last
            // completed ordering.
            if self.stopped {
                #[cfg(feature = "dupstats")]
                self.timeline
                    .end(false, self.nodes, root_moves[0].mv, root_moves[0].score);
                break;
            }
            #[cfg(feature = "dupstats")]
            self.timeline
                .end(true, self.nodes, root_moves[0].mv, root_moves[0].score);

            // End of a completed iteration (the reference's `if
            // (!threads.stop)` block).
            if self.last_iteration_pv.is_empty() || root_moves[0].pv[0] != self.last_iteration_pv[0]
            {
                last_best_move_depth = root_depth;
            }
            self.last_iteration_pv.clear();
            self.last_iteration_pv.extend_from_slice(&root_moves[0].pv);
            match completed_best.as_mut() {
                Some(slot) => slot.clone_from(&root_moves[0]),
                None => completed_best = Some(root_moves[0].clone()),
            }
            #[cfg(feature = "verbose2")]
            {
                completed_lines = root_moves[..multi_pv].to_vec();
            }
            completed_depth = root_depth;
            self.completed_depth = root_depth;

            // An exact flat mate is the engine's win proven, and no deeper
            // iteration can improve on it, so deepening stops — under `go
            // infinite` and while thinking ahead as well. When the reply goes
            // out is still the caller's stop rules' to decide. With several
            // lines, one line's mate must not end the others' search.
            #[cfg(feature = "verbose2")]
            let single_line = multi_pv == 1;
            #[cfg(not(feature = "verbose2"))]
            let single_line = true;
            if FLATTEN && single_line && iter_best_value.abs() == VALUE_MATE_FLAT {
                break;
            }

            // Early mate / mated termination: stop once the search depth
            // outruns 2.5× the mate distance. Suppressed for MultiPV > 1, where
            // one PV finding a mate must not stop the others, and under `go
            // mate`, where the search keeps proving within its time budget.
            #[cfg(feature = "verbose2")]
            let break_on_mate = multi_pv == 1 && !self.mate_mode;
            #[cfg(not(feature = "verbose2"))]
            let break_on_mate = true;
            if break_on_mate {
                if iter_best_value >= VALUE_TB_WIN_IN_MAX_PLY
                    && (VALUE_MATE - iter_best_value + 2) * 5 / 2 < root_depth
                {
                    break;
                }
                if iter_best_value <= -VALUE_TB_WIN_IN_MAX_PLY
                    && (iter_best_value + VALUE_MATE + 2) * 5 / 2 < root_depth
                {
                    break;
                }
            }

            // The reference keeps its mate-found stop under `#if STOCKFISH`,
            // deferring mate proofs to a separate engine. This port has none,
            // so without this branch a bare `go mate` would hang. In USI
            // `limits.mate` is a millisecond budget, so the reference's
            // `VALUE_MATE - score <= 2 * limits.mate` distance bound is
            // degenerate and reduces to stopping on a decisive score.
            #[cfg(feature = "verbose2")]
            if self.mate_mode && (is_win(iter_best_value) || is_loss(iter_best_value)) {
                break;
            }

            self.fold_best_move_changes(&mut tot_best_move_changes);

            // Whether there is time for another iteration. Only the main
            // worker under active time management, and only until the end time
            // is fixed. Without `verbose2` a `TimeControl` is only ever the
            // active kind, so holding one is the whole condition.
            #[cfg(feature = "verbose2")]
            let time_managed = self
                .control
                .time
                .as_ref()
                .is_some_and(|tc| tc.use_time_management && tc.tm.search_end == 0);
            #[cfg(not(feature = "verbose2"))]
            let time_managed = self
                .control
                .time
                .as_ref()
                .is_some_and(|tc| tc.tm.search_end == 0);
            if time_managed && !self.stopped {
                // `nodesEffort` divides by this worker's OWN node counter,
                // not the all-worker aggregate.
                let own_nodes = self.nodes.max(1);
                let effort = root_moves[0].effort;
                let single_root_move = root_moves.len() == 1;
                let tc = self.control.time.as_ref().expect("time control present");
                let optimum = tc.tm.optimum() as f64;
                let maximum = tc.tm.maximum() as f64;
                // Credited: what thinking ahead spent on this position counts
                // against the budget too.
                let elapsed = tc.tm.credited_elapsed_from(Instant::now());

                let nodes_effort = effort * 100_000 / own_nodes;

                // `falling_eval` is > 1 when the score is dropping against
                // the previous iterations and < 1 when rising.
                let falling_eval = ((11.325
                    + 2.115 * (best_previous_average_score - iter_best_value) as f64
                    + 0.987 * (iter_value[iter_idx] - iter_best_value) as f64)
                    / 100.0)
                    .clamp(0.5688, 1.5698);

                // Shorter when the best move is stable across iterations.
                let k = 0.5189;
                let center = last_best_move_depth as f64 + 11.57;
                time_reduction =
                    0.723 + 0.79 / (1.104 + (-k * (completed_depth as f64 - center)).exp());
                let reduction = (1.455 + previous_time_reduction) / (2.2375 * time_reduction);

                let best_move_instability =
                    1.04 + 1.8956 * tot_best_move_changes / n_threads as f64;
                let high_best_move_effort = if completed_depth >= 10 && nodes_effort >= 92425 {
                    0.666
                } else {
                    1.0
                };

                let mut total_time = optimum
                    * falling_eval
                    * reduction
                    * best_move_instability
                    * high_best_move_effort;

                if single_root_move {
                    total_time = total_time.min(502.0);
                }

                // Over budget: fix the end time rather than stopping now.
                if elapsed as f64 > total_time.min(maximum) {
                    self.control
                        .time
                        .as_mut()
                        .expect("time control present")
                        .tm
                        .set_search_end(elapsed);
                } else {
                    increase_depth = elapsed as f64 <= total_time * 0.503;
                }
            }

            iter_value[iter_idx] = iter_best_value;
            iter_idx = (iter_idx + 1) & 3;
        }

        // A worker that ran its deepening to its end, in a search that ends at
        // the first worker to finish: the first one stops the others.
        #[cfg(all(feature = "dupstats", feature = "verbose3"))]
        if !self.stopped
            && let Some(latch) = self.first_finish
            && latch.finish(usize::from(self.dup_worker))
            && let Some(stop) = self.control.stop
        {
            stop.store(true, Ordering::Relaxed);
        }

        #[cfg(feature = "dupstats")]
        self.channels.flush(self.dup_worker, self.nodes);

        // Only an abort during iteration 1 leaves `completed_best` empty, and
        // then `root_moves[0]` — the best-so-far after the partial iteration's
        // sort — is still a legal move. It is taken out of the list rather than
        // copied, so this end of the search never reaches the allocator.
        #[cfg(test)]
        {
            self.last_root_moves = root_moves.clone();
        }
        let best = completed_best.unwrap_or_else(|| root_moves.swap_remove(0));
        #[cfg(feature = "verbose2")]
        if completed_lines.is_empty() {
            completed_lines = vec![best.clone()];
        }

        WorkerResult {
            best,
            completed_depth,
            nodes: self.nodes,
            #[cfg(feature = "verbose1")]
            tt_updates: self.tt_updates,
            #[cfg(feature = "verbose1")]
            eval_dups: self.eval_dups,
            #[cfg(feature = "verbose1")]
            evals: self.evals,
            #[cfg(feature = "verbose1")]
            eval_tt_hits: self.eval_tt_hits,
            #[cfg(feature = "dupstats")]
            dup: Box::new(self.dup),
            #[cfg(feature = "dupstats")]
            stopped_on: if self.stopped {
                self.stack[Self::si(0)].current_move
            } else {
                None
            },
            #[cfg(feature = "dupstats")]
            timeline: self.timeline.clone(),
            #[cfg(feature = "dupstats")]
            node_hist: self.node_hist.clone(),
            #[cfg(all(lazy_smp_spread, feature = "verbose1"))]
            spread: self.spread.counts(),
            #[cfg(all(lazy_smp_defer, feature = "verbose1"))]
            defer: self.defer.counts(),
            #[cfg(feature = "verbose2")]
            uci_pv_sent,
            #[cfg(feature = "verbose2")]
            pv_lines: completed_lines,
            time_reduction,
        }
    }

    /// One root line's aspiration search: widen the window around the line's
    /// running average until the value lands inside it, re-sorting the
    /// still-active tail of `root_moves` after each attempt. Returns the line's
    /// last search value.
    ///
    /// Shogi uses no tbRank banding, so the finished head is `[0..pv_idx]` and
    /// the active tail `[pv_idx..]`. Below `verbose2` the root is single-line:
    /// the searched line is `root_moves[0]`, the head is empty, and the whole
    /// list is the active tail.
    // The argument count crosses the lint's threshold only in a build that
    // compiles the `verbose2` reporting parameters, so this cannot be an
    // `expect`.
    #[allow(clippy::too_many_arguments)]
    fn aspiration_search(
        &mut self,
        work: &mut Position,
        root_moves: &mut Vec<RootMove>,
        root_depth: i32,
        search_again_counter: i32,
        #[cfg(feature = "verbose2")] root_pos: &Position,
        #[cfg(feature = "verbose2")] pv_idx: usize,
        #[cfg(feature = "verbose2")] multi_pv: usize,
    ) -> Value {
        self.sel_depth = 0;

        // Aspiration window. On iteration 1 the seed sentinels make `delta`
        // huge, so alpha/beta clamp to the full window.
        #[cfg(feature = "verbose2")]
        let line = &root_moves[pv_idx];
        #[cfg(not(feature = "verbose2"))]
        let line = &root_moves[0];
        let mut delta: Value = params::ASPIRATION_DELTA.get()
            + (line.mean_squared_score.unsigned_abs() / 9000) as Value;
        let avg = line.average_score;
        let mut alpha = (avg - delta).max(-VALUE_INFINITE);
        let mut beta = (avg + delta).min(VALUE_INFINITE);

        // Each fail high shaves a ply off `adjusted_depth`.
        let mut failed_high_cnt = 0;

        let mut best_value;

        #[cfg(feature = "dupstats")]
        {
            self.channels.pass = 0;
        }
        loop {
            let adjusted_depth =
                1.max(root_depth - failed_high_cnt - 3 * (search_again_counter + 1) / 4);
            #[cfg(feature = "dupstats")]
            let nodes_before = self.nodes;
            self.set_root_delta(window_width(alpha, beta));
            self.root_depth = root_depth;
            // Slot 0 still holds the root refresh seeded by the caller.
            self.acc_depth = 0;
            best_value = self.lend_picker_scratch(|this, scratch| {
                this.search(
                    work,
                    0,
                    alpha,
                    beta,
                    adjusted_depth,
                    false,
                    true,
                    None,
                    Some(&mut *root_moves),
                    scratch,
                )
            });

            // Non-PV moves carry `-VALUE_INFINITE`, so this stable sort only
            // raises the PV and leaves the rest in order.
            #[cfg(feature = "verbose2")]
            root_moves[pv_idx..].sort_by(root_move_order);
            #[cfg(not(feature = "verbose2"))]
            root_moves.sort_by(root_move_order);

            #[cfg(feature = "dupstats")]
            {
                use crate::channels::{Asp, AspResult};
                let result = if self.stopped {
                    AspResult::Stopped
                } else if best_value <= alpha {
                    AspResult::FailLow
                } else if best_value >= beta {
                    AspResult::FailHigh
                } else {
                    AspResult::Exact
                };
                self.channels.asp.push(Asp {
                    depth: root_depth,
                    pass: self.channels.pass,
                    searched_depth: adjusted_depth,
                    alpha,
                    beta,
                    value: best_value,
                    result,
                    nodes: self.nodes - nodes_before,
                });
                self.channels.pass += 1;
            }

            // The sort above still leaves `root_moves[0]` a legal best-so-far,
            // since it ordered the previous iteration's scores.
            if self.stopped {
                break;
            }

            #[cfg(feature = "verbose2")]
            if self.should_output_fail_lh(multi_pv, best_value, alpha, beta, root_depth) {
                self.emit_pv(root_pos, root_moves, pv_idx, root_depth, multi_pv);
                self.last_pv_info_time = Instant::now();
            }

            // The reference's fail-low widening is `beta = alpha;
            // alpha = bestValue - delta`, not Stockfish's `(alpha + beta) / 2`.
            if best_value <= alpha {
                beta = alpha;
                alpha = (best_value - delta).max(-VALUE_INFINITE);
                failed_high_cnt = 0;
                #[cfg(feature = "dupstats")]
                self.timeline.fail_low();
            } else if best_value >= beta {
                alpha = (beta - delta).max(alpha);
                beta = (best_value + delta).min(VALUE_INFINITE);
                failed_high_cnt += 1;
                #[cfg(feature = "dupstats")]
                self.timeline.fail_high();
            } else {
                break;
            }
            delta += delta / 3;
        }

        best_value
    }

    /// The aggregate node count across every worker: with a Lazy-SMP tally
    /// installed, the helpers' last-published slots plus this worker's live
    /// `self.nodes`.
    ///
    /// Only the PV-output path reads it — the `go nodes N` ceiling has its own
    /// [`Self::counted_nodes`] — so it is compiled only where a PV is printed,
    /// and only the main worker prints one: its tally is the whole pool, its
    /// own slot first.
    #[cfg(feature = "verbose2")]
    fn aggregate_nodes(&self) -> u64 {
        match self.node_tally {
            Some(slots) => {
                let mut total = self.nodes;
                for s in slots.iter().skip(1) {
                    total += s.load(Ordering::Relaxed);
                }
                total
            }
            None => self.nodes,
        }
    }

    /// Whether the PV-output interval has elapsed since the last emit.
    /// Without a PV config — helpers and the parity path — vacuously true.
    #[cfg(feature = "verbose2")]
    fn pv_interval_elapsed(&self) -> bool {
        match &self.pv_config {
            Some(cfg) => self.last_pv_info_time + cfg.pv_interval <= Instant::now(),
            None => true,
        }
    }

    /// The reference's fail-high/low PV-output gate, bound to this worker's
    /// live state. Only the main worker owns a sink and PV config, so only it
    /// can pass.
    #[cfg(feature = "verbose2")]
    fn should_output_fail_lh(
        &self,
        multi_pv: usize,
        best_value: Value,
        alpha: Value,
        beta: Value,
        root_depth: i32,
    ) -> bool {
        let Some(cfg) = &self.pv_config else {
            return false;
        };
        fail_lh_pv_gate(
            self.pv_sink.is_some(),
            multi_pv,
            best_value,
            alpha,
            beta,
            self.aggregate_nodes(),
            root_depth,
            self.last_pv_info_time + cfg.pv_interval <= Instant::now(),
            cfg.output_fail_lh_pv,
        )
    }

    /// Build and emit the per-line PV `info` output — the reference's
    /// `main_manager()->pv()`. A no-op when no sink is installed.
    #[cfg(feature = "verbose2")]
    fn emit_pv(
        &mut self,
        root_pos: &Position,
        root_moves: &[RootMove],
        pv_idx: usize,
        depth: i32,
        multi_pv: usize,
    ) {
        if self.pv_sink.is_none() {
            return;
        }
        let nodes = self.aggregate_nodes();
        let infos = self.build_pv_infos(root_pos, root_moves, pv_idx, depth, multi_pv, nodes);
        if let Some(sink) = self.pv_sink.as_mut() {
            for info in &infos {
                sink.emit(info);
            }
        }
    }

    /// Assemble the `info` lines for `pv()`: one per PV index, carrying the
    /// reported depth, score, bound and PV.
    ///
    /// Public so the coordinator can build the final-PV fallback lines from the
    /// chosen worker's result, passing `pv_idx == lines.len()` to make every
    /// line exact.
    #[cfg(feature = "verbose2")]
    pub fn build_pv_infos(
        &self,
        root_pos: &Position,
        root_moves: &[RootMove],
        pv_idx: usize,
        depth: i32,
        multi_pv: usize,
        nodes: u64,
    ) -> Vec<PvInfo> {
        let consideration = self
            .pv_config
            .as_ref()
            .is_some_and(|c| c.consideration_mode);
        // The reference reads the occupancy again for each line it prints, but
        // no search runs between the lines of one call, so one read serves them
        // all. The clock is read once for the same reason.
        let hashfull = TranspositionTable::shared().hashfull(0);
        // `tm.elapsed_time()`, floored at 1 ms so a search that answered inside
        // one tick still has a divisor for `nps`. The origin is the PV
        // configuration's `limits.startTime`, the instant the `go` arrived and
        // the one time management measures from.
        let time_ms = self
            .pv_config
            .as_ref()
            .map_or(0, |c| {
                Instant::now()
                    .saturating_duration_since(c.start_time)
                    .as_millis() as u64
            })
            .max(1);
        let nps = nodes * 1000 / time_ms;
        let mut out = Vec::with_capacity(multi_pv);
        for (i, rm) in root_moves.iter().enumerate().take(multi_pv) {
            let updated = rm.score != -VALUE_INFINITE;
            if depth == 1 && !updated && i > 0 {
                continue;
            }
            let d = if updated { depth } else { 1.max(depth - 1) };
            let mut v = if updated {
                rm.uci_score
            } else {
                rm.previous_score
            };
            if v == -VALUE_INFINITE {
                v = VALUE_DRAW;
            }
            // Only the currently-searched line can carry a fail bound.
            let is_exact = i != pv_idx || !updated;
            let bound = if is_exact {
                PvBound::Exact
            } else if rm.score_lowerbound {
                PvBound::Lower
            } else if rm.score_upperbound {
                PvBound::Upper
            } else {
                PvBound::Exact
            };
            let pv = if consideration {
                self.consideration_pv(root_pos, &rm.pv)
            } else {
                rm.pv.clone()
            };
            out.push(PvInfo {
                depth: d,
                sel_depth: rm.sel_depth,
                multipv: i + 1,
                score: v,
                bound,
                nodes,
                nps,
                hashfull,
                time_ms,
                pv,
            });
        }
        out
    }

    /// The ConsiderationMode PV collector: walk the root PV as far as it goes,
    /// then extend from the transposition table, stopping at a repetition after
    /// ply 0, a TT miss, or an unusable TT move.
    ///
    /// The reference appends a repetition or terminal text marker to the PV
    /// string; this surfaces the PV as moves only and stops at the same points.
    #[cfg(feature = "verbose2")]
    fn consideration_pv(&self, root_pos: &Position, root_pv: &[Move]) -> Vec<Move> {
        let mut pos = root_pos.clone();
        let mut moves: Vec<Move> = Vec::new();
        let mut applied: Vec<(Move, yorkie_state::Undo)> = Vec::new();
        let mut ply = 0usize;
        while ply < MAX_PLY as usize {
            if ply >= 1 && self.repetition(&pos) != RepetitionState::None {
                break;
            }
            let m = if ply < root_pv.len() {
                root_pv[ply]
            } else {
                let key = pos.key();
                let side = pos.side_to_move().index() as u8;
                let (found, data, _writer) =
                    TranspositionTable::shared().probe(key, side, |m16| pos.to_move(m16).is_some());
                if !found {
                    break;
                }
                // A hit only extends the PV when the stored move is playable
                // here.
                match data.move16.and_then(|m16| pos.to_move(m16)) {
                    Some(mm)
                        if mm.is_ok()
                            && pos.pseudo_legal::<GENERATE_ALL_LEGAL_MOVES>(mm)
                            && pos.is_legal(mm) =>
                    {
                        mm
                    }
                    _ => break,
                }
            };
            // A resign/win sentinel is not playable.
            if !m.is_ok() {
                break;
            }
            moves.push(m);
            let undo = pos.do_move(m);
            applied.push((m, undo));
            ply += 1;
        }
        while let Some((m, undo)) = applied.pop() {
            pos.undo_move(m, undo);
        }
        moves
    }
}

// The interior main search: the reference's shared `search` body, entered at
// the root and recursed into at `PV` / `NonPV` until the depth reaches 0 and it
// dives into qsearch.

/// `NO_PIECE` continuation plane (`continuationHistory[0][0][NO_PIECE][SQ_ZERO]`):
/// the null-move sentinel plane, index `0` in this port's flat layout.
const NULL_MOVE_CONT_PLANE: ContPlane = ContPlane::SENTINEL;

impl<N: NetworkParams> QSearch<N> {
    /// Whether `m` is a plain capture in `pos`. In this engine the
    /// reference's `capture_stage` and `capture` coincide.
    fn is_capture(pos: &Position, m: Move) -> bool {
        !m.is_drop() && pos.board().get(m.to_sq()).is_some()
    }

    /// `is_shuffling(move, ss, pos)`: whether `move` merely shuffles a piece
    /// back and forth, so its singular extension should be suppressed. Shogi
    /// has no 50-move rule and a drop is not a round trip, so captures and
    /// drops are excluded outright.
    fn is_shuffling(&self, mv: Move, capture: bool, ply: i32, pos: &Position) -> bool {
        if capture || mv.is_drop() {
            return false;
        }
        if pos.plies_from_null() <= 6 || ply < 18 {
            return false;
        }
        let s = Self::si(ply);
        let (Some(move2), Some(move4)) = (
            self.stack[s - 2].current_move,
            self.stack[s - 4].current_move,
        ) else {
            return false;
        };
        if !move2.is_ok() || !move4.is_ok() || move2.is_drop() || move4.is_drop() {
            return false;
        }
        mv.from_sq() == move2.to_sq() && move2.from_sq() == move4.to_sq()
    }

    /// `ss->statScore` for one move. `s` is the current stack index; `captured`
    /// is the victim, present iff `capture`.
    fn move_stat_score(
        &self,
        us: Color,
        moved_piece: Piece,
        mv: Move,
        s: usize,
        capture: bool,
        captured: Option<Piece>,
    ) -> i32 {
        if let (true, Some(cap_piece)) = (capture, captured) {
            863 * piece_value(cap_piece) / 128
                + self
                    .histories
                    .capture
                    .get(moved_piece, mv.to_sq(), cap_piece)
        } else if capture {
            // A capture always carries a victim, so this arm is unreachable;
            // returning `0` keeps it panic-free instead of an `expect`.
            0
        } else {
            2 * self.histories.main.get(us, mv)
                + self.histories.continuation.get_at(
                    self.stack[s - 1].cont_hist,
                    moved_piece,
                    mv.to_sq(),
                )
                + self.histories.continuation.get_at(
                    self.stack[s - 2].cont_hist,
                    moved_piece,
                    mv.to_sq(),
                )
        }
    }

    /// Install `width` as the `rootDelta` [`Self::reduction`] reduces against.
    /// A window's width is positive — every caller passes `beta - alpha` with
    /// `beta` above `alpha` — and the floor is where that becomes a fact of the
    /// stored type rather than a property of the callers.
    fn set_root_delta(&mut self, width: Value) {
        self.root_delta = NonZeroU32::new(width.max(1) as u32).unwrap_or(NonZeroU32::MIN);
    }

    /// The `rootDelta` divisor in the `1..=i32::MAX` the signed division in
    /// [`Self::reduction`] needs: positive by the stored type, and narrowed
    /// here to the signed range. A width never reaches that ceiling — the
    /// widest is the full window's `2 * VALUE_INFINITE` — so the narrowing
    /// changes no value the search ever holds.
    fn root_delta_divisor(&self) -> i32 {
        self.root_delta.get().min(i32::MAX as u32) as i32
    }

    /// `reduction(i, d, mn, delta)`, scaled by 1024.
    fn reduction(&self, improving: bool, d: i32, mn: i32, delta: i32) -> i32 {
        let reduction_scale = REDUCTIONS[d as usize] * REDUCTIONS[mn as usize];
        reduction_scale - delta * params::LMR_DELTA.get() / self.root_delta_divisor()
            + (!improving as i32) * reduction_scale * 206 / 512
            + params::LMR_BASE.get()
    }

    /// `correction_value(*this, pos, ss)`: a weighted sum of the side-to-move
    /// channel reads keyed by the position's partial keys, plus the `(ss-2)` /
    /// `(ss-4)` continuation-correction reads when `(ss-1)`'s move is ok.
    ///
    /// On fresh tables the sum is small enough that
    /// [`to_corrected_static_eval`] divides it away to zero.
    fn correction_value(&self, pos: &Position, ply: i32) -> i32 {
        let us = pos.side_to_move();
        let pcv = self
            .histories
            .shared
            .correction_get(pos.pawn_key(), us, CorrChannel::Pawn);
        let micv =
            self.histories
                .shared
                .correction_get(pos.minor_piece_key(), us, CorrChannel::Minor);
        let wnpcv = self.histories.shared.correction_get(
            pos.non_pawn_key(Color::White),
            us,
            CorrChannel::NonPawnWhite,
        );
        let bnpcv = self.histories.shared.correction_get(
            pos.non_pawn_key(Color::Black),
            us,
            CorrChannel::NonPawnBlack,
        );

        // `(ss-1)`, `(ss-2)` and `(ss-4)` all sit in the four cells below this
        // one, so one window carries the bound for all three: the window's
        // length is the constant 4, against which the offsets into it are
        // checked at compile time.
        let s = Self::si(ply);
        let recent = &self.stack[s - 4..s];
        let (prev1, prev2, prev4) = (&recent[3], &recent[2], &recent[0]);
        let cntcv = if let Some(prev_move) = prev1.current_move.filter(|m| m.is_ok()) {
            let to = prev_move.to_sq();
            match pos.board().get(to) {
                Some(pc) => {
                    self.histories
                        .continuation_correction
                        .get_at(prev2.cont_corr, pc, to)
                        + self
                            .histories
                            .continuation_correction
                            .get_at(prev4.cont_corr, pc, to)
                }
                None => 8,
            }
        } else {
            8
        };

        12153 * pcv + 8620 * micv + 12355 * (wnpcv + bnpcv) + 7982 * cntcv
    }

    /// Drive the shared search body directly, at ply 0 but as a *non-root*
    /// node. The real search enters through [`Self::run_root`] instead; this is
    /// for tests and smoke checks.
    pub fn run_search(
        &mut self,
        pos: &mut Position,
        alpha: Value,
        beta: Value,
        depth: i32,
        cut_node: bool,
        pv_node: bool,
    ) -> Value {
        self.run_search_at(pos, 0, alpha, beta, depth, cut_node, pv_node)
    }

    /// [`Self::run_search`] with the node placed at `ply`: the values it
    /// returns count from a root `ply` plies above it, the ones it stores
    /// count from the node itself.
    #[expect(clippy::too_many_arguments)]
    fn run_search_at(
        &mut self,
        pos: &mut Position,
        ply: i32,
        alpha: Value,
        beta: Value,
        depth: i32,
        cut_node: bool,
        pv_node: bool,
    ) -> Value {
        self.nodes = 0;
        self.sel_depth = 0;
        self.set_root_side(pos.side_to_move());
        self.set_read_tt(true);
        self.set_root_delta(window_width(alpha, beta));
        self.root_depth = depth;
        self.last_iteration_pv.clear();
        self.stopped = false;
        self.calls_cnt = CHECK_INTERVAL;
        for cell in self.stack.iter_mut() {
            // Preserve each cell's preallocated `pv` buffer across the reset.
            let mut pv = std::mem::take(&mut cell.pv);
            pv.clear();
            *cell = SearchStackCell::default();
            cell.pv = pv;
        }
        self.seed_accumulator(pos);
        #[cfg(repetition_index)]
        self.index_root(pos);
        self.lend_picker_scratch(|this, scratch| {
            this.search(
                pos, ply, alpha, beta, depth, cut_node, pv_node, None, None, scratch,
            )
        })
    }

    /// The shared `search<Root/PV/NonPV>` body. `prior_captured` is the piece
    /// the move that reached this node captured.
    ///
    /// `root_moves` is `Some` **only** for the root call; its presence is the
    /// `rootNode` flag. `scratch` holds the picker buffers still free: this
    /// node takes one for each picker it builds and passes the rest to
    /// everything it searches, the same-ply singular re-entry included.
    #[expect(clippy::too_many_arguments)]
    fn search(
        &mut self,
        pos: &mut Position,
        ply: i32,
        mut alpha: Value,
        mut beta: Value,
        mut depth: i32,
        cut_node: bool,
        pv_node: bool,
        prior_captured: Option<Piece>,
        mut root_moves: Option<&mut Vec<RootMove>>,
        scratch: &mut [PickerScratch],
    ) -> Value {
        // `Root` is a PV node, so `pv_node` is also true at the root.
        let root_node = root_moves.is_some();
        let all_node = !(pv_node || cut_node);

        // qsearch runs at the *same* ply, not `ply + 1`. It clears and then
        // publishes the mark itself, and its value is this node's, so the two
        // need nothing done between them.
        if depth <= 0 {
            self.pv_node = pv_node;
            self.set_read_tt(true);
            return self.qsearch(pos, ply, alpha, beta, scratch);
        }

        // The node starts unmarked; every return below leaves this holding the
        // marks of the value it returns.
        #[cfg(feature = "verbose3")]
        {
            self.clear_held();
        }

        depth = depth.min(MAX_PLY - 1);
        debug_assert!(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE);
        debug_assert!(pv_node || alpha == beta - 1);
        debug_assert!(0 < depth && depth < MAX_PLY);
        debug_assert!(!(pv_node && cut_node));

        let s = Self::si(ply);

        self.check_time();

        // Step 1. Initialize node.
        let in_check = pos.in_check();
        self.stack[s].in_check = in_check;
        let prior_capture = prior_captured.is_some();
        let us = pos.side_to_move();
        self.stack[s].move_count = 0;
        self.stack[s].ply = ply;
        let mut best_value = -VALUE_INFINITE;

        let follow_pv = root_node
            || (ply >= 1
                && self.stack[s - 1].follow_pv
                && ((ply - 1) as usize) < self.last_iteration_pv.len()
                && self.stack[s - 1].current_move
                    == Some(self.last_iteration_pv[(ply - 1) as usize]));
        self.stack[s].follow_pv = follow_pv;

        if pv_node && self.sel_depth < ply + 1 {
            self.sel_depth = ply + 1;
        }

        // The root never draws or mate-distance-prunes.
        if !root_node {
            // Step 2. Immediate draw / max ply.
            let draw_type = self.repetition(pos);
            if draw_type != RepetitionState::None {
                // The judgement read the move history, so the value is path
                // dependent by construction — this is where the mark
                // originates.
                #[cfg(feature = "verbose3")]
                {
                    self.mark_repetition(pos);
                }
                if draw_type == RepetitionState::Draw {
                    return self.draw_value(RepetitionState::Draw, us) + value_draw(self.nodes);
                }
                return value_from_tt(self.draw_value(draw_type, us), ply);
            }
            // The reference folds `threads.stop` into this return: an
            // aborted non-root node yields the draw score without touching the
            // TT.
            if self.stopped || ply >= MAX_PLY || is_past_move_limit(pos.ply()) {
                // The move limit is a rule of the tournament and could be set
                // differently; neither an abort nor the `MAX_PLY` search
                // horizon is, so only the first marks the value it produces.
                #[cfg(feature = "verbose3")]
                if !self.stopped && ply < MAX_PLY {
                    self.marks.move_limit = true;
                }
                return self.draw_value(RepetitionState::Draw, us) + value_draw(self.nodes);
            }

            // Step 3. Mate distance pruning. Under `FLATTEN` the bound on the
            // engine's loss — the floor where the engine is to move, the
            // ceiling where its opponent is — is the reference's. The other
            // bound is a distance the engine's own mates do not carry: it still
            // narrows the window as long as the window stays open, so a search
            // with no flat mate in it runs exactly as the gradient build's, but
            // it never closes the window, since pruning on it would cut a flat
            // mate off as if it were a longer one. In its place the flat mate
            // itself bounds the window, as nothing outranks it.
            if FLATTEN && self.is_engine(us) {
                alpha = mated_in(ply).max(alpha);
                let ceiling = mate_in(ply + 1);
                if ceiling > alpha {
                    beta = ceiling.min(beta);
                }
                if alpha >= VALUE_MATE_FLAT {
                    return alpha;
                }
            } else if FLATTEN {
                beta = mate_in(ply + 1).min(beta);
                let floor = mated_in(ply);
                if floor < beta {
                    alpha = floor.max(alpha);
                }
                if beta <= -VALUE_MATE_FLAT {
                    return -VALUE_MATE_FLAT;
                }
            } else {
                alpha = mated_in(ply).max(alpha);
                beta = mate_in(ply + 1).min(beta);
            }
            if alpha >= beta {
                return alpha;
            }
        }

        let prev_sq = self.stack[s - 1]
            .current_move
            .filter(|m| m.is_ok())
            .map(Move::to_sq);
        let mut best_move: Option<Move> = None;
        let prior_reduction = self.stack[s - 1].reduction;
        self.stack[s - 1].reduction = 0;
        self.stack[s].stat_score = 0;
        self.stack[s + 2].cutoff_cnt = 0;

        // Step 4. Transposition table lookup.
        let excluded_move = self.stack[s].excluded_move;
        let pos_key = pos.key();
        let side = us.index() as u8;
        // Capture the entry location once (the reference's Step-4 `ttWriter`);
        // every write in this node targets this slot, not a re-probe.
        let (tt_hit, tt_data, tt_slot) =
            TranspositionTable::shared().locate(pos_key, side, |m16| pos.to_move(m16).is_some());
        #[cfg(feature = "dupstats")]
        let tt_data = crate::channels::mask_pv(tt_data);
        self.stack[s].tt_hit = tt_hit;
        // At the root the current PV line's move is treated as the TT move
        // regardless of what the probe returned; the rest of `tt_data` is still
        // consumed as usual.
        let tt_move = if let Some(rms) = root_moves.as_deref() {
            // The line the root is currently searching; single-line below
            // `verbose2`, so line 0.
            #[cfg(feature = "verbose2")]
            let line = self.pv_idx;
            #[cfg(not(feature = "verbose2"))]
            let line = 0;
            Some(rms[line].pv[0])
        } else if tt_hit {
            // The MovePicker's TT stage re-validates this.
            tt_data.move16.and_then(|m16| pos.to_move(m16))
        } else {
            None
        };
        let tt_value = if tt_hit {
            value_from_tt(tt_data.value, ply)
        } else {
            VALUE_NONE
        };
        if !excluded_move.is_some_and(Move::is_ok) {
            self.stack[s].tt_pv = pv_node || (tt_hit && tt_data.is_pv);
        }
        // Snapshot of `ss->ttPv` for the pre-move-loop readers, refreshed
        // after Step 9's verification search, which can flip the live field.
        // Never read past the start of the move loop.
        let mut ttpv = self.stack[s].tt_pv;
        let tt_capture = tt_move.is_some_and(|m| Self::is_capture(pos, m));

        // A measurement that takes the cutoff away from the main worker hides
        // the entry's value from the cutoff alone; every later use sees it.
        #[cfg(feature = "dupstats")]
        let tt_value_kept = tt_value;
        #[cfg(feature = "dupstats")]
        let tt_value = crate::channels::mask_cut_value(tt_value);
        #[cfg(feature = "dupstats")]
        if !root_node {
            let cut = !pv_node
                && !excluded_move.is_some_and(Move::is_ok)
                && tt_data.depth > depth - (tt_value <= beta) as i32
                && is_valid(tt_value)
                && bound_matches(tt_data.bound, tt_value >= beta)
                && (cut_node == (tt_value >= beta) || depth > 5);
            self.channel_use(pos_key, tt_hit, &tt_data, depth, cut);
        }

        // The node's buckets, and whether it is deep enough for the counts of
        // what the table and the move loop did — taken here, before the table
        // cutoff, so a node the cutoff ends is counted among the rest.
        #[cfg(feature = "dupstats")]
        let dup_at = crate::dupstats::Where::new(
            depth,
            ply,
            pv_node,
            cut_node,
            self.stack[s - 1].move_count,
        );
        #[cfg(feature = "dupstats")]
        let dup_deep = depth >= crate::dupstats::PRESENCE_MIN_DEPTH;
        #[cfg(feature = "dupstats")]
        if dup_deep {
            use crate::dupstats::Event;
            // The cutoff's condition below, split: whether the entry is deep
            // enough, and whether everything else allows the cutoff. The
            // condition itself is left as it is, so the playing builds' code is
            // what it is without the counts.
            let tt_deep = tt_data.depth > depth - (tt_value <= beta) as i32;
            let tt_cut = !pv_node
                && !excluded_move.is_some_and(Move::is_ok)
                && tt_deep
                && is_valid(tt_value)
                && bound_matches(tt_data.bound, tt_value >= beta)
                && (cut_node == (tt_value >= beta) || depth > 5);
            let event = if tt_cut {
                Event::TtCut
            } else if !tt_hit {
                Event::TtMiss
            } else if tt_deep {
                Event::TtDeepNoCut
            } else {
                Event::TtShallow
            };
            self.dup.count(event, dup_at);
        }
        if !pv_node
            && !excluded_move.is_some_and(Move::is_ok)
            && tt_data.depth > depth - (tt_value <= beta) as i32
            && is_valid(tt_value)
            && bound_matches(tt_data.bound, tt_value >= beta)
            && (cut_node == (tt_value >= beta) || depth > 5)
        {
            if let Some(ttm) = tt_move
                && tt_value >= beta
            {
                if !tt_capture {
                    update_quiet_histories(
                        &mut self.histories,
                        pos,
                        &self.stack[..],
                        s,
                        ttm,
                        (130 * depth - 71).min(1043),
                    );
                }
                if let Some(psq) = prev_sq
                    && self.stack[s - 1].move_count <= 4
                    && !prior_capture
                    && let Some(pc) = pos.board().get(psq)
                {
                    update_continuation_histories(
                        &mut self.histories,
                        &self.stack[..],
                        s - 1,
                        pc,
                        psq,
                        -2142,
                    );
                }
            }
            // The stored value re-enters the search as this node's value, so
            // it brings its marks with it.
            #[cfg(feature = "verbose3")]
            {
                self.take_entry_marks(tt_data.marks);
            }
            #[cfg(all(lazy_smp_defer, feature = "verbose1"))]
            {
                self.tt_cut_ply = ply;
            }
            return tt_value;
        }
        #[cfg(feature = "dupstats")]
        let tt_value = tt_value_kept;
        #[cfg(feature = "dupstats")]
        let (tt_move, tt_capture, tt_value, tt_data) = if root_node {
            (tt_move, tt_capture, tt_value, tt_data)
        } else {
            crate::channels::mask_after_cut(tt_move, tt_capture, tt_value, tt_data)
        };

        // From here on the node searches. Held to its end: dropping them
        // leaves its place on the path and in the presence table.
        #[cfg(feature = "dupstats")]
        let (_dup_frame, _dup_presence) = self.dup_enter(pos_key, depth, dup_at);

        // Step 5. Mate-in-1 and declaration win.
        let mut unadjusted_static_eval = VALUE_NONE;
        if !root_node
            && !tt_hit
            && !excluded_move.is_some_and(Move::is_ok)
            && !in_check
            && (!FLATTEN || mate_lands_inside_move_limit(pos.ply()))
            && let Some(mate_move) = pos.mate_1ply()
        {
            best_value = self.mate_for(us, ply + 1);
            #[cfg(feature = "dupstats")]
            let dup_updates = self.dup_store(
                tt_slot,
                pos_key,
                (MAX_PLY - 1).min(depth + 6),
                tt_hit,
                dup_at,
            );
            // Returned root-relative, stored position-relative, as every
            // decisive value in the table is.
            self.tt_store(
                tt_slot,
                pos_key,
                value_to_tt(best_value, ply),
                ttpv,
                Bound::Exact,
                (MAX_PLY - 1).min(depth + 6),
                move16_of(Some(mate_move)),
                unadjusted_static_eval,
            );
            #[cfg(feature = "dupstats")]
            self.dup_stored(pos_key, dup_updates);
            return best_value;
        }
        if (tt_move.is_none() || pv_node) && declaration_win(pos, &self.entering_king).is_some() {
            // The value is the declaration rule's; a node that asked and was
            // told no is not marked.
            #[cfg(feature = "verbose3")]
            {
                self.marks.decl_rule = true;
            }
            return self.mate_for(us, ply + 1);
        }

        // Step 6. Static evaluation.
        let correction_value = self.correction_value(pos, ply);
        let eval: Value;
        // Mirrors `ss->staticEval`, and so must be re-synced after Step 9's
        // verification search, which re-enters this node and rewrites the
        // stack cell.
        let mut static_eval: Value;
        let mut improving: bool;

        if in_check {
            static_eval = self.stack[s - 2].static_eval;
            self.stack[s].static_eval = static_eval;
            improving = false;
            // The reference's `goto moves_loop`: Steps 6b-11, the only
            // readers of `eval`, are skipped in check.
        } else {
            if excluded_move.is_some_and(Move::is_ok) {
                // The reference uses the outer search's `ss->staticEval`.
                static_eval = self.stack[s].static_eval;
                unadjusted_static_eval = static_eval;
                eval = static_eval;
            } else if tt_hit {
                unadjusted_static_eval = tt_data.eval;
                #[cfg(feature = "verbose1")]
                {
                    self.eval_tt_hits += u64::from(is_valid(unadjusted_static_eval) && !pv_node);
                }
                if !is_valid(unadjusted_static_eval) {
                    unadjusted_static_eval = self.static_eval(pos);
                } else if pv_node {
                    // The reference build has USE_CLASSIC_EVAL active.
                    unadjusted_static_eval = self.static_eval(pos);
                }
                let corrected = to_corrected_static_eval(unadjusted_static_eval, correction_value);
                static_eval = corrected;
                eval = if is_valid(tt_value) && bound_matches(tt_data.bound, tt_value > corrected) {
                    tt_value
                } else {
                    corrected
                };
            } else {
                unadjusted_static_eval = self.static_eval(pos);
                static_eval = to_corrected_static_eval(unadjusted_static_eval, correction_value);
                eval = static_eval;
                #[cfg(feature = "verbose1")]
                self.count_eval_dup(tt_slot, pos_key);
                self.tt_store(
                    tt_slot,
                    pos_key,
                    VALUE_NONE,
                    ttpv,
                    Bound::None,
                    DEPTH_UNSEARCHED,
                    None,
                    unadjusted_static_eval,
                );
            }
            self.stack[s].static_eval = static_eval;

            if let Some(prev_move) = self.stack[s - 1].current_move.filter(|m| m.is_ok())
                && !self.stack[s - 1].in_check
                && !prior_capture
            {
                let eval_diff =
                    (-(self.stack[s - 1].static_eval + static_eval)).clamp(-214, 171) + 60;
                self.histories
                    .main
                    .update(us.flip(), prev_move, eval_diff * 10);
                if !tt_hit
                    && let Some(psq) = prev_sq
                    && let Some(pc) = pos.board().get(psq)
                {
                    let not_pawn = pc.kind != PieceKind::Pawn || pc.promoted;
                    if not_pawn && !prev_move.is_promote() {
                        self.histories
                            .shared
                            .pawn_update(pos.pawn_key(), pc, psq, eval_diff * 12);
                    }
                }
            }

            improving = static_eval > self.stack[s - 2].static_eval;
            let opponent_worsening = static_eval > -self.stack[s - 1].static_eval;

            if prior_reduction >= 3 && !opponent_worsening {
                depth += 1;
            }
            if prior_reduction >= 2
                && depth >= 2
                && static_eval + self.stack[s - 1].static_eval > 173
            {
                depth -= 1;
            }

            // Step 7. Razoring.
            if !pv_node
                && eval
                    < alpha - params::RAZOR_BASE.get() - params::RAZOR_DEPTH.get() * depth * depth
            {
                self.pv_node = false;
                self.set_read_tt(true);
                return self.qsearch(pos, ply, alpha, beta, scratch);
            }

            // Step 8. Futility pruning.
            let futility_mult = params::FUTILITY_MULT.get() - 21 * (!tt_hit) as i32;
            let margin = futility_mult * depth
                - (2686 * improving as i32 + 362 * opponent_worsening as i32) * futility_mult
                    / 1024
                + correction_value.abs() / 180600;
            if !ttpv
                && depth < 15
                && eval >= beta
                && eval - margin >= beta
                && (tt_move.is_none() || tt_capture)
                && !is_loss(beta)
                && !is_win(eval)
            {
                return (2 * beta + eval) / 3;
            }

            // Step 9. Null-move search with verification search. The
            // reference's `pos.non_pawn_material(us)` term stays inside
            // `#if STOCKFISH`, so it is absent here; `ss->ply >= nmpMinPly` is
            // what disables the pass while a verification search is in flight.
            if cut_node
                && static_eval
                    >= beta - 16 * depth - 53 * improving as i32 + params::NMP_MARGIN.get()
                && !excluded_move.is_some_and(Move::is_ok)
                && ply >= self.nmp_min_ply
                && !is_loss(beta)
            {
                let r = params::NMP_REDUCTION.get() + depth / 3;
                self.stack[s].current_move = Some(Move::null());
                self.stack[s].cont_hist = NULL_MOVE_CONT_PLANE;
                self.stack[s].cont_corr = CorrPlane::SENTINEL;
                pos.do_null_move();
                // A null move touches no accumulator, so it bypasses
                // `push_accumulator`'s prefetch and needs this one, matching
                // the reference's prefetch inside `do_null_move`.
                TranspositionTable::shared().prefetch(pos.key(), pos.side_to_move().index() as u8);
                #[cfg(feature = "verbose3")]
                let node_marks = self.held();
                let null_value = -self.search(
                    pos,
                    ply + 1,
                    -beta,
                    -beta + 1,
                    depth - r,
                    false,
                    false,
                    None,
                    None,
                    scratch,
                );
                // The null child is the only thing a returned `null_value`
                // comes from; the verification search below only gates it, so
                // its own marks go nowhere.
                #[cfg(feature = "verbose3")]
                let null_marks = self.child_marks(node_marks);
                pos.undo_null_move();
                if null_value >= beta && !is_win(null_value) {
                    if self.nmp_min_ply != 0 || depth < 16 {
                        #[cfg(feature = "verbose3")]
                        self.merge_held(null_marks, pos);
                        return null_value;
                    }
                    debug_assert_eq!(
                        self.nmp_min_ply, 0,
                        "recursive null-move verification is not allowed"
                    );

                    // Verify by re-searching this *same* node — same ply,
                    // same stack cell, no `do_move`.
                    self.nmp_min_ply = ply + 3 * (depth - r) / 4;
                    #[cfg(feature = "verbose3")]
                    let node_marks = self.held();
                    let v = self.search(
                        pos,
                        ply,
                        beta - 1,
                        beta,
                        depth - r,
                        false,
                        false,
                        prior_captured,
                        None,
                        scratch,
                    );
                    // A re-entry on this node's own ply: it overwrites the live
                    // marks, which are restored here rather than kept, since
                    // `v` gates the return without being returned.
                    #[cfg(feature = "verbose3")]
                    {
                        self.restore_held(node_marks);
                    }
                    self.nmp_min_ply = 0;

                    if v >= beta {
                        #[cfg(feature = "verbose3")]
                        self.merge_held(null_marks, pos);
                        return null_value;
                    }
                }
            }
            // The verification search above re-entered on this node's own
            // `ss`, so `ss->staticEval` / `ss->ttPv` may have been rewritten
            // under us; every reference read from here on is a live `ss->`
            // read, so the snapshots are refreshed rather than carried.
            static_eval = self.stack[s].static_eval;
            ttpv = self.stack[s].tt_pv;
            improving |= static_eval >= beta;

            // Step 10. Internal iterative reduction.
            if !self.stack[s].follow_pv
                && !all_node
                && depth >= 6
                && tt_move.is_none()
                && prior_reduction <= 3
            {
                depth -= 1;
            }

            // Step 11. ProbCut.
            let prob_cut_beta = beta + params::PROBCUT_MARGIN.get() - 61 * improving as i32;
            if depth >= 3 && !is_decisive(beta) && !(is_valid(tt_value) && tt_value < prob_cut_beta)
            {
                let prob_cut_depth = depth - 4;
                // This picker is done before the node's own is built, so the
                // buffer it takes is the one that picker will take.
                let (slot, scratch) = Self::split_picker_scratch(scratch);
                #[cfg(feature = "dupstats")]
                let picker_tt_move = crate::channels::mask_order(tt_move, root_node);
                #[cfg(not(feature = "dupstats"))]
                let picker_tt_move = tt_move;
                let mut mp =
                    MovePicker::new_probcut(pos, picker_tt_move, prob_cut_beta - static_eval, slot);
                while let Some(mv) = mp.next_move(pos, &self.histories) {
                    if Some(mv) == excluded_move || !pos.is_legal(mv) {
                        continue;
                    }
                    let acc_delta = MoveDelta::from_move(pos, mv);
                    self.nodes += 1;
                    #[cfg(feature = "dupstats")]
                    self.node_hist.count(
                        ply + 1,
                        depth,
                        crate::dupstats::NodeKind::of(pv_node, cut_node),
                    );
                    self.prefetch_child_corrections(s, mv);
                    let undo = pos.do_move(mv);
                    let moved = mv.moved_piece_after();
                    self.stack[s].current_move = Some(mv);
                    self.stack[s].cont_hist =
                        ContinuationHistory::plane_index(in_check, true, moved, mv.to_sq());
                    self.stack[s].cont_corr =
                        ContinuationCorrectionHistory::plane_index(moved, mv.to_sq());
                    self.push_accumulator(pos, mv, &acc_delta);
                    self.pv_node = false;
                    self.set_read_tt(true);
                    #[cfg(feature = "verbose3")]
                    let node_marks = self.held();
                    let mut value =
                        -self.qsearch(pos, ply + 1, -prob_cut_beta, -prob_cut_beta + 1, scratch);
                    if value >= prob_cut_beta && prob_cut_depth > 0 {
                        value = -self.search(
                            pos,
                            ply + 1,
                            -prob_cut_beta,
                            -prob_cut_beta + 1,
                            prob_cut_depth,
                            !cut_node,
                            false,
                            undo.captured(),
                            None,
                            scratch,
                        );
                    }
                    #[cfg(feature = "verbose3")]
                    let child_marks = self.child_marks(node_marks);
                    pos.undo_move(mv, undo);
                    self.pop_accumulator();
                    if value >= prob_cut_beta {
                        // The child cut this node off, so the entry stored from
                        // its value and the value returned below both carry its
                        // marks.
                        #[cfg(feature = "verbose3")]
                        self.merge_held(child_marks, pos);
                        #[cfg(feature = "dupstats")]
                        let dup_updates =
                            self.dup_store(tt_slot, pos_key, prob_cut_depth + 1, tt_hit, dup_at);
                        self.tt_store(
                            tt_slot,
                            pos_key,
                            value_to_tt(value, ply),
                            ttpv,
                            Bound::Lower,
                            prob_cut_depth + 1,
                            move16_of(Some(mv)),
                            unadjusted_static_eval,
                        );
                        #[cfg(feature = "dupstats")]
                        self.dup_stored(pos_key, dup_updates);
                        if !is_decisive(value) {
                            return value - (prob_cut_beta - beta);
                        }
                    }
                }
            }
        }

        // moves_loop: in-check nodes resume here.

        // Step 12. Small ProbCut.
        let prob_cut_beta = beta + params::SMALL_PROBCUT_MARGIN.get();
        if bound_matches(tt_data.bound, true)
            && tt_data.depth >= depth - 4
            && tt_value >= prob_cut_beta
            && !is_decisive(beta)
            && is_valid(tt_value)
            && !is_decisive(tt_value)
        {
            return prob_cut_beta;
        }

        // Held as flat plane indices into the live `continuationHistory`, not
        // as snapshots, so a plane updated by an earlier move's subtree is seen
        // when a later stage scores against it.
        let cont_planes: [ContPlane; 6] = std::array::from_fn(|i| self.stack[s - 1 - i].cont_hist);
        let (slot, scratch) = Self::split_picker_scratch(scratch);
        #[cfg(feature = "dupstats")]
        let picker_tt_move = crate::channels::mask_order(tt_move, root_node);
        #[cfg(not(feature = "dupstats"))]
        let picker_tt_move = tt_move;
        let mut mp =
            MovePicker::new_main_search(pos, picker_tt_move, depth, ply, cont_planes, slot);

        let mut move_count = 0i32;
        let mut quiets_searched = SearchedList::new();
        let mut captures_searched = SearchedList::new();

        // The next move to search: the picker's, or, while this worker's spread
        // is shifting, one from a little further down the picker's order.
        #[cfg(not(lazy_smp_spread))]
        macro_rules! picked_move {
            ($this:expr) => {
                mp.next_move(pos, &$this.histories)
            };
        }
        #[cfg(lazy_smp_spread)]
        self.spread.begin_node(s);
        #[cfg(lazy_smp_spread)]
        macro_rules! picked_move {
            ($this:expr) => {
                $this.spread.pick(s, move_count == 0, || {
                    mp.next_move(pos, &$this.histories)
                        .map(|mv| (mv, mp.in_quiet_stage()))
                })
            };
        }
        // The picked move, unless another worker is searching it: then it is
        // put off until the node's other moves are done.
        #[cfg(not(lazy_smp_defer))]
        macro_rules! next_move {
            ($this:expr) => {
                picked_move!($this)
            };
        }
        #[cfg(lazy_smp_defer)]
        self.defer.begin_node(s);
        #[cfg(lazy_smp_defer)]
        macro_rules! next_move {
            ($this:expr) => {
                $this.defer.next(
                    s,
                    depth,
                    move_count == 0,
                    || picked_move!($this),
                    |mv: Move| {
                        crate::defer::BusyTable::shared()
                            .probe(crate::defer::word(pos_key, mv.move16()))
                    },
                )
            };
        }

        #[cfg(all(lazy_smp_spread, test))]
        let mut audit_counted: Vec<Move> = Vec::new();
        #[cfg(all(lazy_smp_spread, test))]
        let mut audit_repeats = 0u64;
        #[cfg(all(lazy_smp_spread, test))]
        let mut audit_singular = false;

        // Step 13. Loop through the moves.
        while let Some(mv) = next_move!(self) {
            if Some(mv) == excluded_move {
                continue;
            }
            // The MovePicker has already applied the reference's legality
            // filter, so every move here is legal.
            //
            // At the root, skip moves outside the still-active tail
            // `root_moves[pv_idx..]`, which the earlier PV lines have fixed.
            // Below `verbose2` the root is single-line, so the whole list is
            // active and nothing is skipped.
            if let Some(rms) = root_moves.as_deref() {
                #[cfg(feature = "verbose2")]
                let active = &rms[self.pv_idx..];
                #[cfg(not(feature = "verbose2"))]
                let active = rms;
                if !active.iter().any(|rm| rm.mv == mv) {
                    continue;
                }
            }
            move_count += 1;
            self.stack[s].move_count = move_count;
            if pv_node {
                self.stack[s + 1].pv.clear();
            }
            #[cfg(all(lazy_smp_spread, test))]
            {
                audit_repeats += u64::from(audit_counted.contains(&mv));
                audit_counted.push(mv);
            }

            let mut extension = 0;
            let capture = Self::is_capture(pos, mv);
            let moved_piece = mv.moved_piece_after();
            let gives_check = pos.gives_check(mv);
            let mut new_depth = depth - 1;
            let delta = window_width(alpha, beta);
            let mut r = self.reduction(improving, depth, move_count, delta);
            // `ss->ttPv` is read live here and at every in-loop site below:
            // an earlier move's singular re-entry can flip it, and the flip
            // persists for the rest of this node's move loop. The pre-loop
            // `ttpv` snapshot must NOT be used past this point.
            if self.stack[s].tt_pv {
                r += 1013;
            }

            // Step 14. Pruning at shallow depths. Skipped at the root, where
            // every move must be searched.
            if !root_node && !is_loss(best_value) {
                if move_count >= (3 + depth * depth) / (2 - improving as i32) {
                    mp.skip_quiet_moves();
                    #[cfg(lazy_smp_spread)]
                    self.spread.skip_quiets(s);
                }
                let mut lmr_depth = new_depth - r / 1024;
                if capture || gives_check {
                    let victim = pos.board().get(mv.to_sq());
                    let capt_hist = match victim {
                        Some(v) => self.histories.capture.get(moved_piece, mv.to_sq(), v),
                        None => self.histories.capture.get_empty(moved_piece, mv.to_sq()),
                    };
                    if !gives_check && lmr_depth < 7 {
                        let futility_value = static_eval
                            + 218
                            + 223 * lmr_depth
                            + victim.map_or(0, piece_value)
                            + 131 * capt_hist / 1024;
                        if futility_value <= alpha {
                            continue;
                        }
                    }
                    let margin = (167 * depth + capt_hist * 34 / 1024).max(0);
                    if alpha >= VALUE_DRAW && !pos.see_ge(mv, -margin) {
                        continue;
                    }
                } else if !self.stack[s].follow_pv || !pv_node {
                    let mut history = self.histories.continuation.get_at(
                        self.stack[s - 1].cont_hist,
                        moved_piece,
                        mv.to_sq(),
                    ) + self.histories.continuation.get_at(
                        self.stack[s - 2].cont_hist,
                        moved_piece,
                        mv.to_sq(),
                    ) + self.histories.shared.pawn_get(
                        pos.pawn_key(),
                        moved_piece,
                        mv.to_sq(),
                    );
                    if history < -4097 * depth {
                        continue;
                    }
                    history += 71 * self.histories.main.get(us, mv) / 32;
                    lmr_depth += history / 3220;
                    let futility_value = static_eval
                        + 42
                        + 151 * (!best_move.is_some_and(Move::is_ok)) as i32
                        + params::QUIET_FUTILITY_DEPTH.get() * lmr_depth
                        + 86 * (static_eval > alpha) as i32;
                    if !in_check && lmr_depth < 13 && futility_value <= alpha {
                        if best_value <= futility_value
                            && !is_decisive(best_value)
                            && !is_win(futility_value)
                        {
                            best_value = futility_value;
                        }
                        continue;
                    }
                    lmr_depth = lmr_depth.max(0);
                    if !pos.see_ge(mv, -25 * lmr_depth * lmr_depth) {
                        continue;
                    }
                }
            }

            // Step 15. Singular extension. `(ttData.bound & BOUND_LOWER)` is a
            // BIT test, so Exact passes it too. The reference reassigns
            // `ttData.value = value_from_tt(...)` at Step 4, so every
            // `ttData.value` here is `tt_value`, not the raw `tt_data.value`.
            if !root_node
                && Some(mv) == tt_move
                && !excluded_move.is_some_and(Move::is_ok)
                && depth >= 6 + self.stack[s].tt_pv as i32
                && is_valid(tt_value)
                && !is_decisive(tt_value)
                && bound_matches(tt_data.bound, true)
                && tt_data.depth >= depth - 3
                && !self.is_shuffling(mv, capture, ply, pos)
            {
                let singular_beta = tt_value
                    - (params::SINGULAR_MARGIN.get()
                        + 66 * (self.stack[s].tt_pv && !pv_node) as i32)
                        * depth
                        / 55;
                let singular_depth = new_depth / 2;
                #[cfg(feature = "dupstats")]
                self.dup.count(crate::dupstats::Event::SeTried, dup_at);

                // Re-enter on the *same* node with `move` excluded. Any `ss`
                // field the inner search overwrites is intentionally shared,
                // exactly as the reference's re-entry does.
                self.stack[s].excluded_move = Some(mv);
                self.pv_node = false;
                self.set_read_tt(true);
                #[cfg(feature = "verbose3")]
                let node_marks = self.held();
                // The re-entry shares this node's ply of pending moves too, and
                // may cut off with some of its own still there.
                #[cfg(lazy_smp_spread)]
                let held = self.spread.hold(s);
                #[cfg(lazy_smp_defer)]
                let held_deferred = self.defer.hold(s);
                let s_value = self.search(
                    pos,
                    ply,
                    singular_beta - 1,
                    singular_beta,
                    singular_depth,
                    cut_node,
                    false,
                    prior_captured,
                    None,
                    scratch,
                );
                // A re-entry on this node's own ply, so its marks are held aside
                // and folded in only where `s_value` becomes this node's value.
                #[cfg(feature = "verbose3")]
                let singular_marks = self.child_marks(node_marks);
                self.stack[s].excluded_move = None;
                #[cfg(lazy_smp_spread)]
                self.spread.restore(s, held);
                #[cfg(lazy_smp_defer)]
                self.defer.restore(s, held_deferred);
                #[cfg(all(lazy_smp_spread, test))]
                {
                    audit_singular = true;
                }

                if s_value < singular_beta {
                    let corr_val_adj = correction_value.abs() / 210590;
                    let double_margin = -4 + 212 * pv_node as i32
                        - 182 * (!tt_capture) as i32
                        - corr_val_adj
                        - 906 * self.histories.tt_move.get() / 116517
                        - (ply > self.root_depth) as i32 * 44;
                    // `92 * ss->ttPv` reads `ss->ttPv` live, after the
                    // re-entry, which failed low here and so applied
                    // `ss->ttPv |= (ss-1)->ttPv`. The guard and `singular_beta`
                    // above ran first and see the original value.
                    let triple_margin = 73 + 320 * pv_node as i32 - 218 * (!tt_capture) as i32
                        + 92 * self.stack[s].tt_pv as i32
                        - corr_val_adj
                        - (ply > self.root_depth) as i32 * 45;

                    extension = 1
                        + (s_value < singular_beta - double_margin) as i32
                        + (s_value < singular_beta - triple_margin) as i32;

                    // The remaining moves' LMR and the final TT-store depth
                    // observe this bump.
                    depth += 1;
                }
                // Multi-cut pruning: the ttMove is assumed to fail high, so
                // if excluding it still fails high over beta this is not a
                // singular node.
                else if s_value >= beta && !is_decisive(s_value) {
                    self.histories
                        .tt_move
                        .update((-424 - 107 * depth).max(-3375));
                    #[cfg(feature = "verbose3")]
                    self.merge_held(singular_marks, pos);
                    #[cfg(feature = "dupstats")]
                    self.dup.count(crate::dupstats::Event::SeMulticut, dup_at);
                    return s_value;
                } else if tt_value >= beta {
                    extension = -3;
                } else if cut_node {
                    extension = -2;
                }
                // Every branch but the multi-cut's leaves its outcome in
                // `extension`, which is 0 until this block sets it.
                #[cfg(feature = "dupstats")]
                {
                    use crate::dupstats::Event;
                    let event = match extension {
                        1 => Event::SeExt1,
                        2.. => Event::SeExt2Plus,
                        0 => Event::SeNone,
                        _ => Event::SeNegative,
                    };
                    self.dup.count(event, dup_at);
                }
            }

            // Every move after the node's first is entered as this worker's,
            // for the others to put off while it searches it; the entry spans
            // every search of the move below, re-searches included.
            #[cfg(lazy_smp_defer)]
            let busy_word = (self.defer.active()
                && move_count > 1
                && depth >= crate::defer::MIN_DEPTH)
                .then(|| {
                    let word = crate::defer::word(pos_key, mv.move16());
                    crate::defer::BusyTable::shared().enter(word);
                    #[cfg(feature = "verbose1")]
                    self.defer.note_entry();
                    word
                });
            #[cfg(all(lazy_smp_defer, feature = "verbose1"))]
            {
                self.tt_cut_ply = -1;
            }

            // Step 16. Make the move. The accumulator delta must be taken
            // from the pre-move position.
            let acc_delta = MoveDelta::from_move(pos, mv);
            self.nodes += 1;
            #[cfg(feature = "dupstats")]
            self.node_hist.count(
                ply + 1,
                depth,
                crate::dupstats::NodeKind::of(pv_node, cut_node),
            );
            self.prefetch_child_corrections(s, mv);
            let undo = pos.do_move_with_check(mv, gives_check);
            self.stack[s].current_move = Some(mv);
            self.stack[s].cont_hist =
                ContinuationHistory::plane_index(in_check, capture, moved_piece, mv.to_sq());
            self.stack[s].cont_corr =
                ContinuationCorrectionHistory::plane_index(moved_piece, mv.to_sq());
            self.push_accumulator(pos, mv, &acc_delta);
            new_depth += extension;

            // Read *after* this move's node increment, so `rm.effort` sums
            // only the subtree below the move.
            let node_count = self.nodes;

            if self.stack[s].tt_pv {
                r -= 2819
                    + pv_node as i32 * 973
                    + (tt_value > alpha) as i32 * 905
                    + (tt_data.depth >= depth) as i32 * (935 + cut_node as i32 * 959);
            }
            r += 691;
            r -= move_count * 65;
            r -= correction_value.abs() / 25600;
            if cut_node {
                r += params::LMR_CUT_NODE.get() + 985 * tt_move.is_none() as i32;
            }
            if tt_capture {
                r += 1054;
            }
            if self.stack[s + 1].cutoff_cnt > 1 {
                r +=
                    251 + 1124 * (self.stack[s + 1].cutoff_cnt > 2) as i32 + 1042 * all_node as i32;
            }
            if Some(mv) == tt_move {
                r -= 2239;
            }
            let stat_score = self.move_stat_score(us, moved_piece, mv, s, capture, undo.captured());
            self.stack[s].stat_score = stat_score;
            r -= stat_score * 428 / 4096;
            if all_node {
                r += r * 273 / (256 * depth + 260);
            }

            // A widened root reads each candidate move unreduced, with the
            // window `(floor, alpha)` in place of the null window: a value inside
            // it is the move's score, one at or below `floor` puts the move more
            // than the margin below the best and marks it far, and one at or
            // above `alpha` is re-searched with the full window. Taking the
            // window needs a PV child; `floor` is below `alpha` whenever `wide`
            // holds.
            let mut wide = false;
            let mut floor = alpha;
            if root_node && move_count > 1 {
                let width = self.root_width.get();
                if width.widens()
                    && (width.unreduced == 0 || move_count as usize <= width.unreduced)
                    && alpha > -VALUE_INFINITE
                    && let Some(rm) = root_moves
                        .as_deref()
                        .and_then(|rms| rms.iter().find(|rm| rm.mv == mv))
                    && !rm.far
                {
                    wide = true;
                    floor = (alpha - width.margin).max(-VALUE_INFINITE);
                }
            }

            // Step 17. Late-move reduction / extension.
            //
            // Several searches of this one move can run below; each overwrites
            // the live marks, so the last one — the one `value` comes from — is
            // the one left standing when the move's result is folded in.
            #[cfg(feature = "verbose3")]
            let node_marks = self.held();
            let mut value: Value = best_value;
            if depth >= 2 && move_count > 1 && !wide {
                let d = ((new_depth - r / 1024).min(new_depth + 2)).max(1) + pv_node as i32;
                self.stack[s].reduction = new_depth - d;
                self.pv_node = false;
                self.set_read_tt(true);
                value = -self.search(
                    pos,
                    ply + 1,
                    -(alpha + 1),
                    -alpha,
                    d,
                    true,
                    false,
                    undo.captured(),
                    None,
                    scratch,
                );
                self.stack[s].reduction = 0;
                if value > alpha {
                    let do_deeper = d < new_depth && value > best_value + 48;
                    let do_shallower = value < best_value + 9;
                    new_depth += do_deeper as i32 - do_shallower as i32;
                    if new_depth > d {
                        value = -self.search(
                            pos,
                            ply + 1,
                            -(alpha + 1),
                            -alpha,
                            new_depth,
                            !cut_node,
                            false,
                            undo.captured(),
                            None,
                            scratch,
                        );
                    }
                    update_continuation_histories(
                        &mut self.histories,
                        &self.stack[..],
                        s,
                        moved_piece,
                        mv.to_sq(),
                        1426,
                    );
                }
            }
            // Step 18. Full-depth search when LMR is skipped.
            else if wide {
                self.stack[s + 1].pv.clear();
                self.pv_node = true;
                self.set_read_tt(true);
                value = -self.search(
                    pos,
                    ply + 1,
                    -alpha,
                    -floor,
                    new_depth,
                    false,
                    true,
                    undo.captured(),
                    None,
                    scratch,
                );
            } else if !pv_node || move_count > 1 {
                if tt_move.is_none() {
                    r += 1057;
                }
                let nd = new_depth - (r > 4628) as i32 - (r > 5772 && new_depth > 2) as i32;
                self.pv_node = false;
                self.set_read_tt(true);
                value = -self.search(
                    pos,
                    ply + 1,
                    -(alpha + 1),
                    -alpha,
                    nd,
                    !cut_node,
                    false,
                    undo.captured(),
                    None,
                    scratch,
                );
            }

            if pv_node && (move_count == 1 || value > alpha || (wide && value >= alpha)) {
                self.stack[s + 1].pv.clear();
                if Some(mv) == tt_move
                    && ((is_valid(tt_value) && is_decisive(tt_value) && tt_data.depth > 0)
                        || tt_data.depth > 1)
                {
                    new_depth = new_depth.max(1);
                }
                self.pv_node = true;
                self.set_read_tt(true);
                value = -self.search(
                    pos,
                    ply + 1,
                    -beta,
                    -alpha,
                    new_depth,
                    false,
                    true,
                    undo.captured(),
                    None,
                    scratch,
                );
            }

            #[cfg(feature = "verbose3")]
            let child_marks = self.child_marks(node_marks);

            // Step 19. Undo move.
            pos.undo_move(mv, undo);
            self.pop_accumulator();

            // Cleared before the stop check below, so a stopped search leaves
            // no entry behind.
            #[cfg(lazy_smp_defer)]
            if let Some(word) = busy_word {
                crate::defer::BusyTable::shared().clear(word);
            }
            // The only nodes at `ply + 1` since the mark was reset are this
            // move's child, searched once or more, and a node that returns on
            // its table entry has searched nothing below it.
            #[cfg(all(lazy_smp_defer, feature = "verbose1"))]
            if self.tt_cut_ply == ply + 1 && self.defer.draining(s) {
                self.defer.note_tt_cut();
            }

            #[cfg(feature = "dupstats")]
            if root_node {
                self.timeline
                    .root_move(move_count == 1, mv, self.nodes - node_count);
                self.channels.root.push(crate::channels::RootSearch {
                    depth: self.root_depth,
                    pass: self.channels.pass,
                    move_count,
                    mv,
                    alpha,
                    beta,
                    value,
                    nodes: self.nodes - node_count,
                    best: move_count == 1 || value > alpha,
                });
            }

            // A stop that fired inside this move's subtree makes the returned
            // value untrustworthy, so bail out without updating the best move,
            // the PV, the root-move list or the TT. The caller unwinds the same
            // way, and `run_root` keeps the last completed iteration's order.
            if self.stopped {
                return VALUE_DRAW;
            }

            // Step 20. Fold this move's result into its RootMove before the
            // generic best-move logic. A searched root move is always in the
            // list, so `find` never misses; the `if let` keeps that
            // panic-free.
            if let Some(rms) = root_moves.as_deref_mut()
                && let Some(rm) = rms.iter_mut().find(|rm| rm.mv == mv)
            {
                let ms_init: i64 = -(VALUE_INFINITE as i64 * VALUE_INFINITE as i64);
                rm.effort += self.nodes - node_count;
                if wide && value <= floor {
                    rm.far = true;
                }
                rm.average_score = if rm.average_score != -VALUE_INFINITE {
                    (value + rm.average_score) / 2
                } else {
                    value
                };
                let value_sq = value as i64 * (value as i64).abs();
                rm.mean_squared_score = if rm.mean_squared_score != ms_init {
                    (value_sq + rm.mean_squared_score) / 2
                } else {
                    value_sq
                };

                if move_count == 1 || value > alpha {
                    rm.score = value;
                    rm.uci_score = value;
                    // The marks of the value just recorded, kept with it: the
                    // reply reports the marks of the move it plays, which is
                    // this record rather than whatever the live marks hold once
                    // the search has finished.
                    #[cfg(feature = "verbose3")]
                    {
                        rm.marks = child_marks.marks;
                        rm.causes = child_marks.causes.below(pos.timeline_index());
                    }
                    rm.sel_depth = self.sel_depth;
                    // The two flags decide only which bound marker a PV line
                    // carries; the `uci_score` clamps beside them feed the
                    // resign decision, so they happen in every build.
                    #[cfg(feature = "verbose2")]
                    {
                        rm.score_lowerbound = false;
                        rm.score_upperbound = false;
                    }
                    if value >= beta {
                        #[cfg(feature = "verbose2")]
                        {
                            rm.score_lowerbound = true;
                        }
                        rm.uci_score = beta;
                    } else if value <= alpha {
                        #[cfg(feature = "verbose2")]
                        {
                            rm.score_upperbound = true;
                        }
                        rm.uci_score = alpha;
                    }
                    rm.pv.clear();
                    rm.pv.push(mv);
                    rm.pv.extend(self.stack[s + 1].pv.iter().copied());

                    // How often the best move changes within an iteration,
                    // for time management. Only the first PV line counts, and
                    // only a *change*. Under Lazy-SMP each worker bumps its own
                    // shared slot for the main worker to sum. Below `verbose2`
                    // the first line is the only one.
                    #[cfg(feature = "verbose2")]
                    let counts = move_count > 1 && self.pv_idx == 0;
                    #[cfg(not(feature = "verbose2"))]
                    let counts = move_count > 1;
                    if counts {
                        match self.best_move_tally {
                            Some(slots) => {
                                slots[0].fetch_add(1, Ordering::Relaxed);
                            }
                            None => self.best_move_changes += 1.0,
                        }
                    }
                } else {
                    // Sink to the lowest value; the stable sort in `run_root`
                    // keeps its position.
                    rm.score = -VALUE_INFINITE;
                }
            }

            // Step 20 (cont). Check for a new best move.
            let inc = ((value == best_value)
                && (ply + 2 >= self.root_depth)
                && ((self.nodes as i32) & 14) == 0
                && !is_win(value.abs() + 1)) as i32;
            if value + inc > best_value {
                best_value = value;
                if value + inc > alpha {
                    // The child moved this node's window — it raised `alpha`,
                    // or cut the node off below — so its marks become this
                    // node's too.
                    #[cfg(feature = "verbose3")]
                    self.merge_held(child_marks, pos);
                    best_move = Some(mv);
                    // Update the node PV even on a fail high, but not at the
                    // root, whose PV is the RootMove's.
                    if pv_node && !root_node {
                        self.update_pv(ply, mv);
                    }
                    if value >= beta {
                        self.stack[s].cutoff_cnt += ((extension < 2) || pv_node) as i32;
                        #[cfg(feature = "dupstats")]
                        if dup_deep {
                            use crate::dupstats::Event;
                            let event = match move_count {
                                ..=1 => Event::CutFirst,
                                2..=4 => Event::Cut2To4,
                                _ => Event::CutLater,
                            };
                            self.dup.count(event, dup_at);
                            if event == Event::CutFirst && Some(mv) == tt_move {
                                self.dup.count(Event::CutFirstTtMove, dup_at);
                            }
                        }
                        break;
                    }
                    if depth > 2 && depth < 14 && !is_decisive(value) {
                        depth -= 2;
                    }
                    alpha = value;
                    // Nothing outranks a flat mate, so the remaining moves
                    // cannot change this node's value.
                    if FLATTEN && value == VALUE_MATE_FLAT {
                        break;
                    }
                } else if FLATTEN && pv_node && !root_node && value == -VALUE_MATE_FLAT {
                    // The first move of a node the engine mates whatever it
                    // plays. Nothing scores below the flat mate, so this
                    // value is exact although it fails low, and its line is
                    // kept: the reported mate length is read off the PV.
                    self.update_pv(ply, mv);
                }
            }
            if Some(mv) != best_move && move_count <= SEARCHED_LIST_CAPACITY as i32 {
                if capture {
                    captures_searched.push(mv);
                } else {
                    quiets_searched.push(mv);
                }
            }
        }
        #[cfg(all(lazy_smp_spread, test))]
        if audit_singular {
            self.singular_audit.nodes += 1;
            self.singular_audit.repeats += audit_repeats;
        }

        // Counted apart from the cutoff itself, by the value the loop ended
        // with, so the two can be checked against each other.
        #[cfg(feature = "dupstats")]
        if dup_deep && move_count != 0 && best_value >= beta {
            self.dup.count(crate::dupstats::Event::Cuts, dup_at);
        }

        // Step 21-23. Mate check, stat updates, TT write.
        if best_value >= beta && !is_decisive(best_value) && !is_decisive(alpha) {
            // The blend weighs this node's depth against one ply of beta, and
            // the depth here is at least 1: a node at 0 or below returned from
            // the qsearch hand-off at the top and never reached this. The floor
            // and the saturating step change no weight the search ever holds;
            // between them they leave the divisor in `2..=i32::MAX`, where the
            // signed division has neither a zero nor an `i32::MIN / -1` case to
            // check for and fault on.
            let weight = depth.max(1);
            best_value = (best_value * weight + beta) / weight.saturating_add(1);
        }

        if move_count == 0 {
            best_value = if excluded_move.is_some_and(Move::is_ok) {
                alpha
            } else {
                self.mated_for(us, ply)
            };
        } else if let Some(best) = best_move.filter(|m| m.is_ok()) {
            update_all_stats(
                &mut self.histories,
                pos,
                &self.stack[..],
                s,
                best,
                prev_sq,
                quiets_searched.as_slice(),
                captures_searched.as_slice(),
                depth,
                tt_move,
                prior_capture,
            );
            if !pv_node {
                self.histories
                    .tt_move
                    .update(if Some(best) == tt_move { 805 } else { -787 });
            }
        } else if !prior_capture && let Some(psq) = prev_sq {
            let mut bonus_scale = -232;
            bonus_scale -= self.stack[s - 1].stat_score / 108;
            bonus_scale += (59 * depth).min(454);
            bonus_scale += 169 * (self.stack[s - 1].move_count > 8) as i32;
            bonus_scale += 145 * (!in_check && best_value <= static_eval - 110) as i32;
            bonus_scale += 154
                * (!self.stack[s - 1].in_check && best_value <= -self.stack[s - 1].static_eval - 73)
                    as i32;
            bonus_scale = bonus_scale.max(0);
            let scaled_bonus = (135 * depth - 80).min(1400) * bonus_scale;
            if let Some(pc) = pos.board().get(psq) {
                update_continuation_histories(
                    &mut self.histories,
                    &self.stack[..],
                    s - 1,
                    pc,
                    psq,
                    scaled_bonus * 221 / 16384,
                );
                if let Some(prev_move) = self.stack[s - 1].current_move {
                    self.histories
                        .main
                        .update(us.flip(), prev_move, scaled_bonus * 235 / 32768);
                }
                let not_pawn = pc.kind != PieceKind::Pawn || pc.promoted;
                if not_pawn && !self.stack[s - 1].current_move.is_some_and(Move::is_promote) {
                    self.histories.shared.pawn_update(
                        pos.pawn_key(),
                        pc,
                        psq,
                        scaled_bonus * 290 / 8192,
                    );
                }
            }
        } else if prior_capture
            && let Some(psq) = prev_sq
            && let Some(pc) = pos.board().get(psq)
            && let Some(cap) = prior_captured
        {
            self.histories.capture.update(pc, psq, cap, 1018);
        }

        if best_value <= alpha {
            self.stack[s].tt_pv = self.stack[s].tt_pv || self.stack[s - 1].tt_pv;
        }

        // At the root, PV lines beyond the first must NOT overwrite the TT:
        // their reduced windows would poison the entry the first line wrote.
        // Below `verbose2` the root has no line beyond the first.
        #[cfg(feature = "verbose2")]
        let skip_tt_write =
            excluded_move.is_some_and(Move::is_ok) || (root_node && self.pv_idx != 0);
        #[cfg(not(feature = "verbose2"))]
        let skip_tt_write = excluded_move.is_some_and(Move::is_ok);
        if !skip_tt_write {
            let bound = if best_value >= beta {
                Bound::Lower
            } else if pv_node && best_move.is_some_and(Move::is_ok) {
                Bound::Exact
            } else {
                Bound::Upper
            };
            let store_depth = if move_count != 0 {
                depth
            } else {
                (MAX_PLY - 1).min(depth + 6)
            };
            #[cfg(feature = "dupstats")]
            let dup_updates = self.dup_store(tt_slot, pos_key, store_depth, tt_hit, dup_at);
            #[cfg(lazy_smp_spread)]
            self.spread_note_store(tt_slot, pos_key, store_depth);
            self.tt_store(
                tt_slot,
                pos_key,
                value_to_tt(best_value, ply),
                self.stack[s].tt_pv,
                bound,
                store_depth,
                move16_of(best_move),
                unadjusted_static_eval,
            );
            #[cfg(feature = "dupstats")]
            self.dup_stored(pos_key, dup_updates);
        }

        // The clamp bound is `CORRECTION_HISTORY_LIMIT / 4`.
        let best_move_ok = best_move.is_some_and(Move::is_ok);
        if !(in_check || best_move.is_some_and(|m| m.is_ok() && Self::is_capture(pos, m)))
            && (best_value > static_eval) == best_move_ok
        {
            let sign = if best_move_ok { 12 } else { 17 };
            let bonus = ((best_value - static_eval) * depth * sign / 128).clamp(-256, 256);
            update_correction_history(
                &mut self.histories,
                pos,
                &self.stack[..],
                s,
                1069 * bonus / 1024,
            );
        }

        best_value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use yorkie_eval::{
        FC_0_PADDED_INPUT_DIMS, HIDDEN_SIZE, HIDDEN1_DIMS, LAYER_STACKS, OwnedNetwork, PtrNetwork,
    };
    use yorkie_state::{Piece, PieceKind, Square};

    use crate::text_str::parse_sfen;
    use yorkie_storage::{Bound, TTData, TranspositionTable};

    /// The instantiation these tests drive: a network they made up, so its
    /// address is chosen while they run. The associated functions that read no
    /// network at all still have to be named through some instantiation, and
    /// this is the one.
    type TestQSearch = QSearch<PtrNetwork>;

    const LANE_A: usize = 0;
    const LANE_B: usize = HIDDEN_SIZE / 2;

    fn zero_net() -> OwnedNetwork {
        // The FT stays zero, so every position evaluates to 0 and the qsearch
        // arithmetic can be exercised against a known static eval.
        let mut b = OwnedNetwork::zeroed();
        let row = HIDDEN1_DIMS * FC_0_PADDED_INPUT_DIMS;
        for s in 0..LAYER_STACKS {
            let w = b.fc_0_weights_mut(s);
            w[row + LANE_A] = 1;
            w[row + LANE_B] = 1;
        }
        b
    }

    fn pos(sfen: &str) -> Position {
        parse_sfen(sfen).expect("valid SFEN")
    }

    /// Serialises the tests that use the transposition table.
    ///
    /// There is one table in the process — a `static` — so two tests running as
    /// threads of one binary would clear and overwrite each other's entries.
    /// Under `cargo nextest` each test is its own process and the lock is never
    /// contended; under a plain `cargo test` it is what makes these tests
    /// deterministic.
    static TT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Exclusive use of the emptied table, released when the test's binding
    /// goes out of scope. Derefs to the table, so a caller writes `&tt` where a
    /// `&TranspositionTable` is wanted.
    struct TtHandle {
        _guard: std::sync::MutexGuard<'static, ()>,
        tt: &'static TranspositionTable,
    }

    impl std::ops::Deref for TtHandle {
        type Target = TranspositionTable;

        fn deref(&self) -> &TranspositionTable {
            self.tt
        }
    }

    fn fresh_tt() -> TtHandle {
        // A panicking test leaves the lock poisoned; the next test wants the
        // table, not the panic, and clears it before reading anything.
        let guard = TT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let tt = TranspositionTable::shared();
        tt.clear();
        TtHandle { _guard: guard, tt }
    }

    fn legal_moves(p: &Position) -> Vec<Move> {
        let mut v = Vec::new();
        p.generate_legal_all(&mut v);
        v
    }

    fn captures(p: &Position) -> Vec<Move> {
        legal_moves(p)
            .into_iter()
            .filter(|&m| !m.is_drop() && p.board().get(m.to_sq()).is_some())
            .collect()
    }

    #[expect(clippy::too_many_arguments)]
    fn prewrite(
        table: &TranspositionTable,
        p: &Position,
        value: Value,
        pv: bool,
        bound: Bound,
        depth: i32,
        mv: Option<NonZeroU16>,
        eval: Value,
    ) {
        let key = p.key();
        let side = p.side_to_move().index() as u8;
        let generation = table.generation();
        let (_f, _d, w) = table.probe(key, side, |m16| p.to_move(m16).is_some());
        w.write(
            key,
            value,
            pv,
            bound,
            depth,
            mv,
            eval,
            generation,
            #[cfg(feature = "verbose3")]
            ValueMarks::NONE,
            #[cfg(feature = "verbose1")]
            &mut 0,
        );
    }

    fn probe_root(table: &TranspositionTable, p: &Position) -> (bool, TTData) {
        let (f, d, _w) = table.probe(p.key(), p.side_to_move().index() as u8, |m16| {
            p.to_move(m16).is_some()
        });
        (f, d)
    }

    /// Path to the real (never-committed) SFNN-1536 network, mirroring the
    /// `real_network.rs` integration test.
    fn real_nn_bin() -> std::path::PathBuf {
        let crate_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        crate_dir.join("../../eval/nn.bin")
    }

    /// Runs a real search with the accumulator self-check armed, so
    /// `evaluate_with(acc) == evaluate(refresh)` is asserted at every
    /// evaluation point. Needs the real network's nonzero weights, so a wrong
    /// accumulator changes the eval; skipped when `nn.bin` is absent.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn differential_accumulator_matches_refresh_at_every_eval_site() {
        let path = real_nn_bin();
        if !path.exists() {
            eprintln!("skipping: {} not present", path.display());
            return;
        }
        let net = yorkie_eval::load_network(&path).expect("real nn.bin loads");
        let _tt = fresh_tt();

        // Start position plus hand-heavy / sparse fixtures that exercise drops,
        // promoted pieces, and frequent king moves (the refresh path).
        const FIXTURES: [&str; 3] = [
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
            "k8/1P7/G8/1N2P4/9/9/9/9/8K b 2PG2pg 1",
            "2k6/9/4+R4/8P/9/1n7/9/9/6K2 b B2Pgp 1",
        ];
        for sfen in FIXTURES {
            let mut p = pos(sfen);
            let mut q = QSearch::new(net.network());
            q.set_verify_accumulator(true);
            // A small fixed depth is enough to cover qsearch, the interior move
            // loop, null moves, and ProbCut — every one of the six eval sites.
            q.run_search(&mut p, -VALUE_INFINITE, VALUE_INFINITE, 5, false, true);
        }
    }

    // Value / depth helper unit tests.

    #[test]
    fn value_tt_roundtrip_shifts_mate_scores_by_ply() {
        // Non-mate values are unchanged.
        assert_eq!(value_to_tt(123, 5), 123);
        assert_eq!(value_from_tt(123, 5), 123);
        // A win is pushed away from the root on store, pulled back on load.
        let win = mate_in(20); // 31980
        assert!(is_win(win));
        assert_eq!(value_to_tt(win, 5), win + 5);
        assert_eq!(value_from_tt(value_to_tt(win, 5), 5), win);
        // A loss symmetrically.
        let loss = mated_in(20); // -31980
        assert!(is_loss(loss));
        assert_eq!(value_to_tt(loss, 5), loss - 5);
        assert_eq!(value_from_tt(value_to_tt(loss, 5), 5), loss);
        // VALUE_NONE never validates.
        assert_eq!(value_from_tt(VALUE_NONE, 5), VALUE_NONE);
    }

    #[test]
    fn decisive_boundaries_match_the_pin() {
        assert!(is_win(VALUE_TB_WIN_IN_MAX_PLY));
        assert!(!is_win(VALUE_TB_WIN_IN_MAX_PLY - 1)); // == VALUE_MAX_EVAL
        assert!(is_loss(-VALUE_TB_WIN_IN_MAX_PLY));
        assert!(!is_loss(-VALUE_TB_WIN_IN_MAX_PLY + 1));
        assert!(is_decisive(mate_in(1)));
        assert!(!is_decisive(0));
        assert_eq!(VALUE_MAX_EVAL, 31753);
        assert_eq!(mate_in(1), 31999);
        assert_eq!(mated_in(0), -32000);
    }

    #[test]
    fn the_flat_mate_sits_between_the_gradient_and_infinity() {
        assert_eq!(VALUE_MATE_FLAT, VALUE_MATE + 1);
        assert_eq!(infinite_for(false), 32001);
        assert_eq!(none_for(false), 32002);
        assert!(VALUE_MATE_FLAT < infinite_for(true));
        assert!(infinite_for(true) < none_for(true));
        assert_eq!(VALUE_INFINITE, infinite_for(FLATTEN));
        assert_eq!(VALUE_NONE, none_for(FLATTEN));
        // Every value both builds' checks read as a mate still reads as one.
        assert!(is_win(VALUE_MATE_FLAT) && is_loss(-VALUE_MATE_FLAT));
    }

    /// Every value the search stores that is not a flat mate — the mate
    /// gradient, the repetition sentinels, the threshold values and the
    /// infinity bound — stored at any ply and read back at any ply under the
    /// flat build, must never come back as `±VALUE_MATE_FLAT`: that would be a
    /// mate proven by nothing. The flat mate itself must come back unchanged.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn no_stored_sentinel_comes_back_as_a_flat_mate_at_any_ply() {
        let flat_infinite = infinite_for(true);
        let mut sentinels = vec![
            VALUE_MAX_EVAL,
            VALUE_TB_WIN_IN_MAX_PLY,
            VALUE_MATE,
            flat_infinite,
            VALUE_DRAW,
        ];
        sentinels.extend((0..=MAX_PLY).map(mate_in));
        let negated: Vec<Value> = sentinels.iter().map(|v| -v).collect();
        sentinels.extend(negated);

        for store_ply in 0..=MAX_PLY {
            for &v in &sentinels {
                let stored = value_to_tt_for(true, v, store_ply);
                assert!(
                    (-VALUE_MATE..=VALUE_MATE).contains(&stored),
                    "{v} stored at ply {store_ply} left the gradient range: {stored}"
                );
                for load_ply in 0..=MAX_PLY {
                    let loaded = value_from_tt_for(true, stored, load_ply);
                    assert_ne!(
                        loaded.abs(),
                        VALUE_MATE_FLAT,
                        "{v} stored at ply {store_ply}, loaded at ply {load_ply}"
                    );
                }
            }
            for flat in [VALUE_MATE_FLAT, -VALUE_MATE_FLAT] {
                let stored = value_to_tt_for(true, flat, store_ply);
                assert_eq!(stored, flat);
                for load_ply in 0..=MAX_PLY {
                    assert_eq!(value_from_tt_for(true, stored, load_ply), flat);
                }
            }
            // The repetition rows reach the search through `value_from_tt`
            // without a store first.
            for v in [VALUE_MATE, -VALUE_MATE, VALUE_MAX_EVAL, -VALUE_MAX_EVAL] {
                assert_ne!(value_from_tt_for(true, v, store_ply).abs(), VALUE_MATE_FLAT);
            }
        }
        // The miss sentinel survives, and the gradient build is the
        // reference's shift, unclamped.
        assert_eq!(value_from_tt_for(true, none_for(true), 7), none_for(true));
        assert_eq!(value_to_tt_for(false, -infinite_for(false), 3), -32004);
    }

    #[test]
    fn a_mate_lands_inside_the_move_limit_only_below_it() {
        let limit = NonZeroI32::new(50);
        assert!(mate_lands_inside(limit, 0));
        assert!(mate_lands_inside(limit, 49));
        // At the limit the move is still played, but the position it mates is
        // past the limit and adjudicated a draw.
        assert!(!mate_lands_inside(limit, 50));
        assert!(!mate_lands_inside(limit, 51));
        assert!(mate_lands_inside(None, u16::MAX));
        // The compiled limit agrees with the draw rule one ply on.
        for ply in [0u16, 1, 49, 50, 51, 100, 1000] {
            assert_eq!(
                mate_lands_inside_move_limit(ply),
                !is_past_move_limit(ply + 1)
            );
        }
    }

    #[test]
    fn value_draw_dither_is_keyed_on_bit_one() {
        assert_eq!(value_draw(0), -1);
        assert_eq!(value_draw(1), -1); // bit 1 clear
        assert_eq!(value_draw(2), 1); // bit 1 set
        assert_eq!(value_draw(3), 1);
        assert_eq!(value_draw(4), -1);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn draw_value_table_rows_other_than_the_draw_are_fixed() {
        let net = zero_net();
        let _tt = fresh_tt();
        let mut q = QSearch::new(net.network());
        // Emulate a root search with Black to move.
        q.set_root_side(Color::Black);

        // Every row but REPETITION_DRAW is a fixed `drawValueTable` value, the
        // same for either side — except, under `FLATTEN`, the engine's win by
        // its opponent's perpetual check, which is a flat mate.
        for c in [Color::Black, Color::White] {
            let engine = c == Color::Black;
            let win = if FLATTEN && engine {
                VALUE_MATE_FLAT
            } else {
                VALUE_MATE
            };
            let lose = if FLATTEN && !engine {
                -VALUE_MATE_FLAT
            } else {
                -VALUE_MATE
            };
            assert_eq!(q.draw_value(RepetitionState::Win, c), win);
            assert_eq!(q.draw_value(RepetitionState::Lose, c), lose);
            assert_eq!(q.draw_value(RepetitionState::Superior, c), VALUE_MAX_EVAL);
            assert_eq!(q.draw_value(RepetitionState::Inferior, c), -VALUE_MAX_EVAL);
            assert_eq!(q.draw_value(RepetitionState::None, c), VALUE_DRAW);
        }
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn the_repetition_draw_row_is_the_compiled_contempt_signed_per_side() {
        // The REPETITION_DRAW row is `+contempt` for the root side and
        // `-contempt` for the opponent (the reference's symmetric
        // `drawValueTable[REPETITION_DRAW][us] = +dv`, `[~us] = -dv`), where
        // the contempt is the root side's `DrawValue` setting scaled by
        // `PawnValue` and truncated toward zero.
        assert_eq!(
            DRAW_CONTEMPT_BLACK,
            (crate::config::DRAW_VALUE_BLACK * PAWN_VALUE as i64 / 100) as Value
        );
        assert_eq!(
            DRAW_CONTEMPT_WHITE,
            (crate::config::DRAW_VALUE_WHITE * PAWN_VALUE as i64 / 100) as Value
        );

        let net = zero_net();
        let _tt = fresh_tt();
        let mut q = QSearch::new(net.network());

        q.root_us = Color::Black;
        let black = DRAW_CONTEMPT_BLACK;
        assert_eq!(q.draw_value(RepetitionState::Draw, Color::Black), black);
        assert_eq!(q.draw_value(RepetitionState::Draw, Color::White), -black);

        // White to move at the root: the same mechanism, opposite root side.
        q.root_us = Color::White;
        let white = DRAW_CONTEMPT_WHITE;
        assert_eq!(q.draw_value(RepetitionState::Draw, Color::White), white);
        assert_eq!(q.draw_value(RepetitionState::Draw, Color::Black), -white);
    }

    // Stand-pat.

    const TWO_KINGS: &str = "4k4/9/9/9/9/9/9/9/4K4 b - 1";

    #[cfg_attr(miri, ignore)]
    #[test]
    fn stand_pat_cutoff_adjusts_and_writes_unsearched_lower_bound() {
        let net = zero_net();
        let table = fresh_tt();
        let p = pos(TWO_KINGS);
        assert!(!p.in_check());

        // best_value == eval == 0 >= beta (-4); not decisive → (0 + -4)/2 = -2.
        let out = {
            let mut q = QSearch::new(net.network());
            q.run(&mut p.clone(), -5, -4, false, true)
        };
        assert_eq!(out.value, -2);
        assert_eq!(out.nodes, 0, "stand pat returns before any do_move");

        let (found, data) = probe_root(&table, &p);
        assert!(found);
        assert_eq!(data.depth, DEPTH_UNSEARCHED);
        assert_eq!(data.bound, Bound::Lower);
        assert_eq!(data.value, value_to_tt(-2, 0));
        assert_eq!(data.eval, 0);
        assert!(!data.is_pv);
    }

    // Futility / SEE pruning — one position, three alpha regimes.

    // Black lance on 9c captures the white pawn on 9b, which is defended by the
    // white gold on 9a: SEE == PawnValue - LanceValue == 90 - 315 == -225. It
    // is the only capture, is non-checking, and there is no 1-ply mate.
    const LANCE_SEE: &str = "g7k/p8/L8/9/9/9/9/9/8K b - 1";

    fn lance_capture(p: &Position) -> Move {
        // Landing on the enemy 2nd rank, `generate_legal_all` lists both the
        // promoting and non-promoting variants; they share `from`, `to`, SEE
        // and victim, so either stands for the capture qsearch searches.
        let target = Square::new(8, 1).unwrap();
        captures(p)
            .into_iter()
            .find(|&m| m.to_sq() == target)
            .expect("the lance capture of 9b must exist")
    }

    #[test]
    fn futility_see_preconditions_hold() {
        let p = pos(LANCE_SEE);
        assert!(!p.in_check());
        assert!(p.mate_1ply().is_none());
        let m = lance_capture(&p);
        // futilityBase == staticEval(0) + 328 == 328; the victim is a pawn.
        assert_eq!(p.board().get(m.to_sq()).map(piece_value), Some(90));
        assert!(!p.see_ge(m, -73), "SEE(-225) must fail the -73 gate");
        assert!(p.see_ge(m, -328), "SEE(-225) must clear a -328 gate");
        assert!(!p.see_ge(m, -128), "SEE(-225) must fail a -128 gate");
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn futility_first_prune_floors_bestvalue_at_futility_value() {
        // alpha == 418 == futilityBase + PawnValue(90): the first futility
        // test `futilityValue <= alpha` fires, flooring bestValue at 418.
        let net = zero_net();
        let _tt = fresh_tt();
        let out = {
            let mut q = QSearch::new(net.network());
            q.run(&mut pos(LANCE_SEE), 418, 419, false, true)
        };
        assert_eq!(out.value, 418);
        assert_eq!(out.nodes, 0, "the only capture is futility-pruned");
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn futility_see_branch_floors_bestvalue_at_min_alpha_base() {
        // alpha == 200: futilityValue > alpha, but SEE(-225) < alpha -
        // futilityBase (== -128), so the SEE branch fires with the floor
        // `min(alpha, futilityBase) == min == 200`.
        let net = zero_net();
        let _tt = fresh_tt();
        let out = {
            let mut q = QSearch::new(net.network());
            q.run(&mut pos(LANCE_SEE), 200, 201, false, true)
        };
        assert_eq!(out.value, 200);
        assert_eq!(out.nodes, 0);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn see_minus_73_skips_a_losing_capture() {
        // alpha == 0: futility does not prune (futilityValue 418 > 0, SEE clears
        // alpha - base == -328), so the capture reaches the `!see_ge(m, -73)`
        // gate and is skipped there (SEE -225 < -73). No node is searched.
        let net = zero_net();
        let _tt = fresh_tt();
        let out = {
            let mut q = QSearch::new(net.network());
            q.run(&mut pos(LANCE_SEE), 0, 1, false, true)
        };
        assert_eq!(out.value, 0);
        assert_eq!(out.nodes, 0, "the losing capture is SEE(-73)-skipped");
    }

    // moveCount pruning + its prevSq / givesCheck exemptions.

    // Black rook on 5e has three MVV-distinct captures (bishop 855, gold 540,
    // pawn 90), none checking, and no 1-ply mate. After either of the two
    // top-MVV captures White has no capture, so those children stand pat
    // immediately.
    const THREE_CAPTURES: &str = "6k2/9/9/9/b3R3g/9/4p4/9/K8 b - 1";

    #[cfg_attr(miri, ignore)]
    #[test]
    fn movecount_prune_drops_the_third_capture() {
        let p = pos(THREE_CAPTURES);
        assert!(!p.in_check());
        assert!(p.mate_1ply().is_none());
        assert_eq!(captures(&p).len(), 3, "fixture must offer three captures");

        let net = zero_net();
        let _tt = fresh_tt();
        let out = {
            let mut q = QSearch::new(net.network());
            q.run(&mut p.clone(), 0, 1, false, true)
        };
        // Two captures searched (each child stand-pats: no deeper do_move); the
        // 3rd (lowest-MVV pawn) is pruned by `moveCount > 2` — it is NOT
        // SEE/futility-pruned (its SEE and futilityValue both clear the gates),
        // so `nodes == 2` isolates the moveCount prune.
        assert_eq!(out.nodes, 2);
        assert_eq!(out.value, 0);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn prevsq_exemption_lets_a_recapture_past_movecount_pruning() {
        // Same position, but drive qsearch directly with `(ss-1)->currentMove`
        // set so `prevSq == 5g` (the pawn-capture's target). The 3rd capture is
        // then a "recapture" (to == prevSq), exempt from the moveCount block, so
        // it is searched too: `nodes == 3` rather than 2.
        let net = zero_net();
        let _tt = fresh_tt();
        let mut p = pos(THREE_CAPTURES);
        let mut q = QSearch::new(net.network());
        q.root_us = p.side_to_move();
        q.pv_node = false;
        q.nodes = 0;
        // A dummy previous move whose destination is 5g (internal (4,6)).
        let prev = Move::make(
            Square::new(4, 5).unwrap(),
            Square::new(4, 6).unwrap(),
            Piece::new(PieceKind::Pawn, Color::White),
        );
        q.stack[STACK_BASE - 1].current_move = Some(prev);
        let _ = q.qsearch_at(&mut p, 0, 0, 1);
        assert_eq!(q.nodes, 3, "the recapture is exempt from moveCount pruning");
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn correction_value_is_eval_neutral_on_fresh_tables() {
        // Checked both at ply 0, where the previous move is not ok, and at a
        // deeper ply with a real previous move.
        let net = zero_net();
        let _tt = fresh_tt();
        let mut q = QSearch::new(net.network());

        const SFENS: &[&str] = &[
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
            "4k4/9/9/4p4/4G4/9/9/9/4K4 b - 1",
            "l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1",
        ];
        for sfen in SFENS {
            let p = pos(sfen);

            // Ply 0: (ss-1) has no move ⇒ cntcv == 8.
            q.stack[TestQSearch::si(0) - 1].current_move = None;
            let cv0 = q.correction_value(&p, 0);
            assert_eq!(cv0 / 131072, 0, "cv/131072 must be 0 at ply 0 (`{sfen}`)");

            // Any previous move whose destination holds a piece works.
            let occupied = (0..81u8)
                .filter_map(yorkie_state::Square::from_index)
                .find(|&sq| p.board().get(sq).is_some())
                .unwrap();
            let piece = p.board().get(occupied).unwrap();
            let from = (0..81u8)
                .filter_map(yorkie_state::Square::from_index)
                .find(|&sq| p.board().get(sq).is_none())
                .unwrap();
            let prev = Move::make(from, occupied, piece);
            q.stack[TestQSearch::si(2) - 1].current_move = Some(prev);
            let cv2 = q.correction_value(&p, 2);
            assert_eq!(cv2 / 131072, 0, "cv/131072 must be 0 at ply 2 (`{sfen}`)");

            // And the corrected eval equals the uncorrected one for a spread of
            // static evals.
            for v in [-31000, -500, 0, 123, 30000] {
                assert_eq!(
                    to_corrected_static_eval(v, cv0),
                    to_corrected_static_eval(v, 0),
                    "corrected eval must be unchanged (`{sfen}`, v={v})",
                );
                assert_eq!(
                    to_corrected_static_eval(v, cv2),
                    to_corrected_static_eval(v, 0),
                );
            }
        }
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn leaf_qsearch_correction_value_reads_live_worker_update() {
        // A leaf `correction_value` read must reflect an update made to the one
        // live worker correction table *before* the call — there is
        // no qsearch-private correction duplicate to fall out of sync.
        let net = zero_net();
        let _tt = fresh_tt();
        let mut q = QSearch::new(net.network());
        let p = pos("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");
        q.stack[TestQSearch::si(0) - 1].current_move = None;

        let before = q.correction_value(&p, 0);
        let us = p.side_to_move();
        // Gravity-update the pawn channel to its limit (`+1024`), read live and
        // weighted by 12153 in `correction_value`.
        q.histories
            .shared
            .correction_update(p.pawn_key(), us, CorrChannel::Pawn, 1_000_000);
        let after = q.correction_value(&p, 0);
        assert_eq!(
            after - before,
            12153 * 1024,
            "correction_value must read the live worker pawn channel (+1024 × weight 12153)"
        );
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn leaf_qsearch_value_reflects_worker_correction_update() {
        // End-to-end: a quiet, not-in-check startpos qsearch stand-pats to the
        // corrected static eval. With the zero-eval network the uncorrected eval
        // is 0; a live worker correction update made before `run` must shift the
        // returned value, proving the leaf reads the one live table set.
        let net = zero_net();
        let p = pos("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");

        let base = {
            let _tt = fresh_tt();
            let mut q = QSearch::new(net.network());
            q.run(&mut p.clone(), -VALUE_INFINITE, VALUE_INFINITE, true, true)
                .value
        };

        let bumped = {
            let _tt = fresh_tt();
            let mut q = QSearch::new(net.network());
            let us = p.side_to_move();
            // Saturate the pawn channel so `cv / 131072` is a nonzero shift.
            for _ in 0..64 {
                q.histories.shared.correction_update(
                    p.pawn_key(),
                    us,
                    CorrChannel::Pawn,
                    1_000_000,
                );
            }
            q.run(&mut p.clone(), -VALUE_INFINITE, VALUE_INFINITE, true, true)
                .value
        };

        assert_eq!(base, 0, "fresh tables ⇒ eval-neutral correction");
        assert_ne!(
            base, bumped,
            "leaf qsearch value must reflect the live worker correction update"
        );
    }

    // Black gold on 9c captures the white pawn on 9b and thereby delivers
    // checkmate (the black lance on 9i backs it; the white king on 9a cannot
    // escape or capture). The capturing move gives check.
    const CAPTURE_MATE: &str = "k8/p8/G8/9/9/9/9/9/L7K b - 1";

    #[cfg_attr(miri, ignore)]
    #[test]
    fn givescheck_exemption_searches_a_checking_capture_under_futility() {
        let p = pos(CAPTURE_MATE);
        assert!(!p.in_check());
        let caps = captures(&p);
        assert_eq!(caps.len(), 1);
        let m = caps[0];
        assert!(p.gives_check(m), "the mating capture gives check");
        assert!(p.see_ge(m, -73), "the mating capture clears the -73 gate");

        // The pre-written TT entry makes the node a TT hit, skipping the
        // 1-ply mate short-circuit so the mate must be found through the move
        // loop. A non-checking capture would be futility-pruned at this alpha;
        // the givesCheck exemption is what lets the capture through.
        let net = zero_net();
        let table = fresh_tt();
        prewrite(&table, &p, 0, false, Bound::None, DEPTH_UNSEARCHED, None, 0);
        let out = {
            let mut q = QSearch::new(net.network());
            q.run(&mut p.clone(), 418, 419, false, true)
        };
        let want = if FLATTEN { VALUE_MATE_FLAT } else { mate_in(1) };
        assert_eq!(out.value, want, "the checking capture mates");
        assert_eq!(out.nodes, 1, "only the mating capture is searched");
    }

    // Quiet-move skip (a quiet TT move dropped by `!capture`).

    const QUIET_CHECK: &str = "k8/9/9/9/4R4/9/9/9/8K b - 1";

    #[cfg_attr(miri, ignore)]
    #[test]
    fn quiet_tt_move_is_dropped_by_the_capture_filter() {
        let p = pos(QUIET_CHECK);
        assert!(!p.in_check());
        assert!(captures(&p).is_empty(), "no captures in this position");
        // Rook 5e→9e: a quiet move that checks the white king on 9a.
        let quiet_check = Move::make(
            Square::new(4, 4).unwrap(),
            Square::new(8, 4).unwrap(),
            Piece::new(PieceKind::Rook, Color::Black),
        );
        assert!(p.gives_check(quiet_check));
        assert!(legal_moves(&p).contains(&quiet_check));

        let net = zero_net();
        let table = fresh_tt();
        // Store the quiet check as the TT move (non-cutoff entry).
        prewrite(
            &table,
            &p,
            0,
            false,
            Bound::None,
            DEPTH_UNSEARCHED,
            move16_of(Some(quiet_check)),
            0,
        );
        let out = {
            let mut q = QSearch::new(net.network());
            q.run(&mut p.clone(), 0, 1, false, true)
        };
        // The MovePicker yields the quiet check first; givesCheck skips the
        // futility block, then `!capture` drops it. Nothing is searched.
        assert_eq!(out.nodes, 0);
        assert_eq!(out.value, 0);
    }

    // Mate paths.

    // Black to move and checkmated: a head-gold wall backed by the white king.
    const BLACK_MATED: &str = "4K4/3ggg3/4k4/9/9/9/9/9/9 b - 1";

    #[cfg_attr(miri, ignore)]
    #[test]
    fn no_evasion_in_check_returns_mated_in_ply() {
        let p = pos(BLACK_MATED);
        assert!(p.in_check());
        assert!(legal_moves(&p).is_empty(), "fixture must be checkmate");

        let net = zero_net();
        let _tt = fresh_tt();
        let out = {
            let mut q = QSearch::new(net.network());
            q.run(&mut p.clone(), -VALUE_INFINITE, VALUE_INFINITE, true, true)
        };
        assert_eq!(out.value, mated_in(0)); // -VALUE_MATE
        assert_eq!(out.nodes, 0);
    }

    // Black to move with a 1-ply gold-drop mate (G*8a).
    const MATE_IN_1: &str = "k8/9/G1N6/9/9/9/9/9/8K b G 1";

    #[cfg_attr(miri, ignore)]
    #[test]
    fn mate_1ply_short_circuits_with_exact_tt_write() {
        let p = pos(MATE_IN_1);
        assert!(!p.in_check());
        assert!(p.mate_1ply().is_some());

        let net = zero_net();
        let table = fresh_tt();
        let out = {
            let mut q = QSearch::new(net.network());
            q.run(&mut p.clone(), -VALUE_INFINITE, VALUE_INFINITE, true, true)
        };
        // mate_in(ss->ply + 1) at ply 0, or the flat mate: the root side is
        // the engine's.
        let want = if FLATTEN { VALUE_MATE_FLAT } else { mate_in(1) };
        assert_eq!(out.value, want);
        assert_eq!(out.nodes, 0, "the mate is found before any do_move");

        let (found, data) = probe_root(&table, &p);
        assert!(found);
        assert_eq!(data.bound, Bound::Exact);
        assert_eq!(data.depth, DEPTH_QS);
        assert_eq!(
            data.value, want,
            "at ply 0 the stored value is the returned one"
        );
        assert_eq!(data.eval, VALUE_NONE, "unadjustedStaticEval is still NONE");
        assert_eq!(data.move16, move16_of(p.mate_1ply()));
    }

    /// A quiescence node whose stack cell holds `tt_pv` left behind by another
    /// node at that ply.
    fn qsearch_mate_node_with_stale_tt_pv(net: &OwnedNetwork, read_tt: bool) -> Value {
        let p = pos(MATE_IN_1);
        let mut q = QSearch::new(net.network());
        q.set_root_side(p.side_to_move().flip());
        q.pv_node = false;
        q.set_read_tt(read_tt);
        q.stack[STACK_BASE + 4].tt_pv = true;
        q.qsearch_at(&mut p.clone(), 4, -1, 0)
    }

    /// The quiescence search's mate-in-one store takes its PV flag from this
    /// node's probe, not from whatever its stack cell holds.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn qsearch_one_ply_mate_is_stored_with_the_probe_pv_flag() {
        let p = pos(MATE_IN_1);
        let net = zero_net();
        let table = fresh_tt();

        // An empty table: the probe misses, so the entry is not PV.
        assert_eq!(qsearch_mate_node_with_stale_tt_pv(&net, true), mate_in(5));
        let (found, data) = probe_root(&table, &p);
        assert!(found);
        assert_eq!(data.value, VALUE_MATE - 1);
        assert_eq!(data.move16, move16_of(p.mate_1ply()));
        assert!(!data.is_pv, "the stale stack flag does not reach the entry");

        // A PV entry the node does not take as a hit: the mate is stored, and
        // a probe that is not a hit is not a PV hit either.
        table.clear();
        prewrite(
            &table,
            &p,
            0,
            true,
            Bound::Upper,
            DEPTH_QS,
            None,
            VALUE_NONE,
        );
        assert_eq!(qsearch_mate_node_with_stale_tt_pv(&net, false), mate_in(5));
        let (_, data) = probe_root(&table, &p);
        assert_eq!(data.value, VALUE_MATE - 1);
        assert!(!data.is_pv);

        // A PV entry the node does take as a hit: the mate-in-one store is
        // reached only on a miss, so the entry keeps the PV flag its own
        // probe found.
        table.clear();
        prewrite(
            &table,
            &p,
            0,
            true,
            Bound::Upper,
            DEPTH_QS,
            None,
            VALUE_NONE,
        );
        assert_ne!(qsearch_mate_node_with_stale_tt_pv(&net, true), mate_in(5));
        let (_, data) = probe_root(&table, &p);
        assert_ne!(data.value, VALUE_MATE - 1);
        assert!(data.is_pv);
    }

    /// The entry a one-ply mate leaves behind holds "mate in one from here",
    /// `VALUE_MATE - 1`, and a qsearch probe of it at any ply reads back
    /// `mate_in(ply + 1)`. The engine plays the side that is mated, so the
    /// distance is kept under `FLATTEN` too.
    fn assert_one_ply_mate_entry(table: &TranspositionTable, p: &Position) {
        let (found, data) = probe_root(table, p);
        assert!(found);
        assert_eq!(data.bound, Bound::Exact);
        assert_eq!(data.value, VALUE_MATE - 1, "position-relative mate in one");

        let net = zero_net();
        for probe_ply in [2, 7] {
            assert_eq!(value_from_tt(data.value, probe_ply), mate_in(probe_ply + 1));
            let mut q = QSearch::new(net.network());
            q.set_root_side(p.side_to_move().flip());
            q.pv_node = false;
            q.set_read_tt(true);
            let v = q.qsearch_at(&mut p.clone(), probe_ply, -1, 0);
            assert_eq!(v, mate_in(probe_ply + 1), "probed at ply {probe_ply}");
            assert_eq!(q.nodes, 0, "the probe is answered by the table");
        }
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn qsearch_one_ply_mate_above_the_root_is_stored_position_relative() {
        let p = pos(MATE_IN_1);
        let net = zero_net();
        let table = fresh_tt();
        let mut q = QSearch::new(net.network());
        q.set_root_side(p.side_to_move().flip());
        q.pv_node = false;
        q.set_read_tt(true);
        let v = q.qsearch_at(&mut p.clone(), 4, -1, 0);
        assert_eq!(v, mate_in(5), "the returned value counts from the root");
        assert_eq!(probe_root(&table, &p).1.depth, DEPTH_QS);
        assert_one_ply_mate_entry(&table, &p);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn search_one_ply_mate_above_the_root_is_stored_position_relative() {
        let p = pos(MATE_IN_1);
        let net = zero_net();
        let table = fresh_tt();
        let mut q = QSearch::new(net.network());
        // The root is taken to be the opponent's, so under `FLATTEN` the mate
        // is scored against the engine and keeps its distance.
        q.set_thinking_ahead(true);
        let v = q.run_search_at(&mut p.clone(), 4, -1, 0, 1, false, false);
        assert_eq!(v, mate_in(5), "the returned value counts from the root");
        assert_eq!(q.nodes, 0, "the mate is found before any do_move");
        assert_eq!(probe_root(&table, &p).1.depth, 7);
        assert_one_ply_mate_entry(&table, &p);
    }

    /// White to move with one legal move, K9a-8a, after which black mates with
    /// G*8b.
    const ONE_MOVE_TO_MATE: &str = "k8/9/1G7/9/9/9/9/9/8K w G 1";
    /// The position after that move.
    const MATED_NEXT: &str = "1k7/9/1G7/9/9/9/9/9/8K b G 2";

    /// The mate-in-one node is first met four plies below a root and stored
    /// there; a later search meets it two plies below the root, takes its value
    /// from the table, and reports the mate at its true distance from that root.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn a_one_ply_mate_met_at_another_ply_keeps_its_distance() {
        let parent = pos(ONE_MOVE_TO_MATE);
        assert_eq!(legal_moves(&parent).len(), 1, "the parent has one move");
        assert!(parent.mate_1ply().is_none());
        let child = pos(MATED_NEXT);
        assert!(child.mate_1ply().is_some());
        let mut reached = parent.clone();
        reached.do_move(legal_moves(&parent)[0]);
        assert_eq!(
            reached.key(),
            child.key(),
            "the one move reaches the mate node"
        );

        let net = zero_net();
        let _tt = fresh_tt();
        let mut first = QSearch::new(net.network());
        first.set_thinking_ahead(true);
        assert_eq!(
            first.run_search_at(&mut child.clone(), 4, -1, 0, 1, false, false),
            mate_in(5)
        );

        // A window the null move and the static prunings leave alone, so the
        // single move is searched and the child answers from the table.
        let mut second = QSearch::new(net.network());
        let v = second.run_search_at(&mut parent.clone(), 1, 100, 101, 2, false, false);
        assert_eq!(v, mated_in(3), "white is mated two plies below the parent");
    }

    /// The search after the engine's own move has the opponent to move at its
    /// root, so a mate the root side delivers there is the opponent's and keeps
    /// the gradient, and a mate the root side suffers is the engine's.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn thinking_ahead_scores_the_mates_for_the_side_that_just_moved() {
        let net = zero_net();
        let tt = fresh_tt();
        let mut q = QSearch::new(net.network());
        q.set_thinking_ahead(true);
        let out = q.run(
            &mut pos(MATE_IN_1),
            -VALUE_INFINITE,
            VALUE_INFINITE,
            true,
            true,
        );
        assert_eq!(out.value, mate_in(1));
        drop(tt);

        let _tt = fresh_tt();
        let mut q = QSearch::new(net.network());
        q.set_thinking_ahead(true);
        let out = q.run(
            &mut pos(BLACK_MATED),
            -VALUE_INFINITE,
            VALUE_INFINITE,
            true,
            true,
        );
        let want = if FLATTEN {
            -VALUE_MATE_FLAT
        } else {
            mated_in(0)
        };
        assert_eq!(out.value, want);
    }

    /// The root search scores the engine's mate flat and, the win being proven
    /// exactly, deepens no further than the iteration that proved it. The
    /// gradient build keeps deepening until the depth outruns the distance.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn run_root_stops_deepening_at_an_exact_flat_mate() {
        let net = zero_net();
        let _tt = fresh_tt();
        let out = {
            let mut q = QSearch::new(net.network());
            q.run_root(&pos(MATE_IN_1), 8)
        };
        assert_eq!(out.pv.len(), 1);
        if FLATTEN {
            assert_eq!(out.score, VALUE_MATE_FLAT);
            assert_eq!(out.depth, 1);
        } else {
            assert_eq!(out.score, mate_in(1));
            assert!(out.depth > 1);
        }
    }

    /// A mate-in-one found at the move limit's own ply mates a position past
    /// the limit, which every node adjudicates a draw. Under `FLATTEN` the two
    /// shortcuts that claim a mate one ply early — qsearch's and the interior
    /// search's — must not claim it there. Without a limit the same position is
    /// a plain mate.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn the_one_ply_mate_shortcuts_respect_the_move_limit_boundary() {
        let game_ply = MOVE_LIMIT.map_or(100, |limit| limit.get() as u16);
        let sfen = format!("k8/9/G1N6/9/9/9/9/9/8K b G {game_ply}");
        let net = zero_net();

        let tt = fresh_tt();
        let qsearch_value = {
            let mut q = QSearch::new(net.network());
            q.run(&mut pos(&sfen), -VALUE_INFINITE, VALUE_INFINITE, true, true)
                .value
        };
        drop(tt);
        let _tt = fresh_tt();
        let search_value = {
            let mut q = QSearch::new(net.network());
            q.run_search(
                &mut pos(&sfen),
                -VALUE_INFINITE,
                VALUE_INFINITE,
                1,
                false,
                true,
            )
        };

        for value in [qsearch_value, search_value] {
            if mate_lands_inside_move_limit(game_ply) {
                assert!(is_win(value), "a mate inside the limit, got {value}");
            } else if FLATTEN {
                assert!(
                    !is_decisive(value),
                    "a mate past the limit must not be claimed, got {value}"
                );
            }
        }
    }

    // Repetition draws + the ±1 dither.

    #[cfg_attr(miri, ignore)]
    #[test]
    fn max_ply_returns_draw_with_dither() {
        let net = zero_net();
        let _tt = fresh_tt();
        let mut p = pos(TWO_KINGS);
        let mut q = QSearch::new(net.network());
        q.root_us = p.side_to_move(); // Black
        q.pv_node = false;

        // At ply == MAX_PLY the node returns draw_value(DRAW, us) + dither.
        q.nodes = 0;
        assert_eq!(q.qsearch_at(&mut p, MAX_PLY, -1, 0), -1 + value_draw(0)); // -2
        q.nodes = 2;
        assert_eq!(q.qsearch_at(&mut p, MAX_PLY, -1, 0), -1 + value_draw(2)); // 0
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn repetition_draw_is_detected_with_dither() {
        let net = zero_net();
        let _tt = fresh_tt();
        // Shuffling both kings gives a period-4 cycle, so the position after
        // six plies repeats the one after two: its second occurrence, which is
        // scored as an ordinary draw.
        let mut p = pos(TWO_KINGS);
        let bk = Piece::new(PieceKind::King, Color::Black);
        let wk = Piece::new(PieceKind::King, Color::White);
        let step = |from: (u8, u8), to: (u8, u8), pc: Piece| {
            Move::make(
                Square::new(from.0, from.1).unwrap(),
                Square::new(to.0, to.1).unwrap(),
                pc,
            )
        };
        p.do_move(step((4, 8), (3, 8), bk));
        p.do_move(step((4, 0), (3, 0), wk));
        p.do_move(step((3, 8), (4, 8), bk));
        p.do_move(step((3, 0), (4, 0), wk));
        // Two more plies open the next cycle so ply 6 == ply 2 (dist 4).
        p.do_move(step((4, 8), (3, 8), bk));
        p.do_move(step((4, 0), (3, 0), wk));
        assert_eq!(
            p.is_repetition(),
            RepetitionState::Draw,
            "a position's second occurrence is an ordinary draw"
        );

        let mut q = QSearch::new(net.network());
        q.root_us = p.side_to_move(); // Black
        q.pv_node = false;
        q.nodes = 0;
        // draw_value(DRAW, Black=root_us) + value_draw(0) == -1 + -1 == -2.
        assert_eq!(q.qsearch_at(&mut p, 6, -1, 0), -2);
    }

    /// Both kings of [`TWO_KINGS`] walk a ten-square ring in step, so a
    /// position first recurs twenty plies on: the moves of `laps` laps.
    #[cfg(repetition_index)]
    fn twenty_ply_laps(laps: usize) -> Vec<Move> {
        let ring = |rank_home: u8, rank_away: u8| {
            let mut squares = Vec::new();
            for file in (0..=4).rev() {
                squares.push(Square::new(file, rank_home).unwrap());
            }
            for file in 0..=4 {
                squares.push(Square::new(file, rank_away).unwrap());
            }
            squares
        };
        let black = ring(8, 7);
        let white = ring(0, 1);
        let bk = Piece::new(PieceKind::King, Color::Black);
        let wk = Piece::new(PieceKind::King, Color::White);
        let mut moves = Vec::new();
        for _ in 0..laps {
            for i in 0..10 {
                moves.push(Move::make(black[i], black[(i + 1) % 10], bk));
                moves.push(Move::make(white[i], white[(i + 1) % 10], wk));
            }
        }
        moves
    }

    /// A twenty-ply cycle is a draw to a search with no look-back limit,
    /// whether its earlier occurrence lies in the search or in the game before
    /// the root, and nothing to one with the default window.
    #[cfg(repetition_index)]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn a_twenty_ply_cycle_is_a_draw_only_without_the_look_back_limit() {
        let net = zero_net();
        let _tt = fresh_tt();
        let moves = twenty_ply_laps(2);
        // The node stands 22 plies in, its earlier occurrence 2 plies in; the
        // root is the start (that occurrence in the search) or 21 plies in (it
        // is in the game before the root).
        for root in [0, 21] {
            for (look_back, draw) in [(0, true), (16, false), (20, true), (19, false)] {
                let mut p = pos(TWO_KINGS);
                for &m in &moves[..root] {
                    p.do_move(m);
                }
                let mut q = QSearch::new(net.network());
                q.set_repetition_look_back(look_back);
                q.index_root(&p);
                for &m in &moves[root..22] {
                    p.do_move(m);
                }
                q.root_us = p.side_to_move();
                q.pv_node = false;
                q.nodes = 0;
                let value = q.qsearch_at(&mut p, 22 - root as i32, -1, 0);
                assert_eq!(
                    value == -2,
                    draw,
                    "root {root}, look-back {look_back}: value {value}",
                );
            }
        }
    }

    /// The `verbose3` value marks: where each comes from, how they ride through
    /// the table, and the rule by which a child hands them to its parent.
    #[cfg(feature = "verbose3")]
    mod value_marks {
        use super::*;

        /// White to move, its king on 9a in check from the black rook on 9e,
        /// with room to step aside. In check every evasion is searched and
        /// nothing is pruned, so the node's value comes from the children and
        /// from nothing else.
        const IN_CHECK_WITH_EVASIONS: &str = "k8/9/9/9/R8/9/9/9/8K w - 1";

        /// Black's king on 5b with twelve pieces in the enemy field and a rook
        /// in hand — 32 points, past either point rule's threshold — against a
        /// lone White king in the far corner. White is to move and can declare
        /// nothing (one piece in its own enemy field), so a search of this
        /// position meets the declaration only one ply down, whatever White
        /// plays.
        const WHITE_TO_MOVE_BEFORE_A_BLACK_DECLARATION: &str =
            "+R+R+B+B5/3GKG3/2SGGGS2/9/9/9/9/9/8k w R 1";

        /// Each flag on its own, so a test cannot pass by carrying the wrong
        /// one.
        const EVERY_MARK: [ValueMarks; 3] = [
            ValueMarks {
                path_dep: true,
                decl_rule: false,
                move_limit: false,
            },
            ValueMarks {
                path_dep: false,
                decl_rule: true,
                move_limit: false,
            },
            ValueMarks {
                path_dep: false,
                decl_rule: false,
                move_limit: true,
            },
        ];

        /// [`prewrite`] with the entry's value marks.
        fn prewrite_marked(
            table: &TranspositionTable,
            p: &Position,
            value: Value,
            bound: Bound,
            marks: ValueMarks,
        ) {
            let key = p.key();
            let side = p.side_to_move().index() as u8;
            let generation = table.generation();
            let (_f, _d, w) = table.probe(key, side, |m16| p.to_move(m16).is_some());
            w.write(
                key, value, false, bound, DEPTH_QS, None, 0, generation, marks, &mut 0,
            );
        }

        /// Seed every child of `p` with the same entry, so which evasion the
        /// MovePicker happens to yield first does not decide the outcome.
        fn seed_every_child(
            table: &TranspositionTable,
            p: &Position,
            value: Value,
            bound: Bound,
            marks: ValueMarks,
        ) {
            let mut child = p.clone();
            for mv in legal_moves(p) {
                let undo = child.do_move(mv);
                prewrite_marked(table, &child, value, bound, marks);
                child.undo_move(mv, undo);
            }
        }

        /// The two kings' period-4 shuffle, as [`super::repetition_draw_is_detected_with_dither`]
        /// builds it: after six plies the position repeats the one after two.
        fn shuffled_two_kings() -> Position {
            let mut p = pos(TWO_KINGS);
            let bk = Piece::new(PieceKind::King, Color::Black);
            let wk = Piece::new(PieceKind::King, Color::White);
            let step = |from: (u8, u8), to: (u8, u8), pc: Piece| {
                Move::make(
                    Square::new(from.0, from.1).unwrap(),
                    Square::new(to.0, to.1).unwrap(),
                    pc,
                )
            };
            for _ in 0..2 {
                p.do_move(step((4, 8), (3, 8), bk));
                p.do_move(step((4, 0), (3, 0), wk));
                p.do_move(step((3, 8), (4, 8), bk));
                p.do_move(step((3, 0), (4, 0), wk));
            }
            p
        }

        /// A repetition judgement is the only source of the path-dependence
        /// mark, and it sets that one alone.
        #[cfg_attr(miri, ignore)]
        #[test]
        fn a_repetition_judgement_marks_the_value_it_produces() {
            let net = zero_net();
            let _tt = fresh_tt();
            let mut p = shuffled_two_kings();
            assert_eq!(p.is_repetition(), RepetitionState::Draw);

            let mut q = QSearch::new(net.network());
            q.root_us = p.side_to_move();
            q.pv_node = false;

            q.nodes = 0;
            q.qsearch_at(&mut p, 6, -1, 0);
            assert_eq!(
                q.marks,
                ValueMarks {
                    path_dep: true,
                    ..ValueMarks::NONE
                },
                "the repetition draw is a path-dependent value and nothing else"
            );
            // Eight plies in, the nearest earlier occurrence is four back.
            assert_eq!(p.timeline_index(), 8);
            assert_eq!(q.causes.plies(), &[4], "the cause is the matched position");
            assert!(!q.causes.is_open());

            // The same board before the shuffle, where nothing has recurred: an
            // ordinary node, and an unmarked value.
            let mut fresh = pos(TWO_KINGS);
            assert_eq!(fresh.is_repetition(), RepetitionState::None);
            q.nodes = 0;
            q.qsearch_at(&mut fresh, 4, -1, 0);
            assert_eq!(q.marks, ValueMarks::NONE);
            assert_eq!(q.causes, PathCauses::NONE);
        }

        /// The causes a node keeps from a child are the ones before the node:
        /// a repetition whose earlier occurrence is the node itself closed
        /// inside its subtree. Searched from two plies before the repeating
        /// position, the matched position (four plies back from it) is the
        /// node's own line index, so the node lists nothing; searched from the
        /// ply before it, the matched position is earlier and stays.
        #[cfg_attr(miri, ignore)]
        #[test]
        fn a_repetition_closed_at_or_below_the_node_lists_nothing() {
            let net = zero_net();
            let _tt = fresh_tt();
            let full = shuffled_two_kings();
            let mut q = QSearch::new(net.network());
            q.root_us = full.side_to_move();
            q.pv_node = false;

            let mut found = PathCauses::NONE;
            found.insert(4);
            let child = Held {
                marks: ValueMarks {
                    path_dep: true,
                    ..ValueMarks::NONE
                },
                causes: found,
            };
            for (at, kept) in [(4usize, &[][..]), (5, &[4][..])] {
                let mut node = pos(TWO_KINGS);
                let mut replay = shuffled_two_kings_moves().into_iter();
                for mv in replay.by_ref().take(at) {
                    node.do_move(mv);
                }
                assert_eq!(node.timeline_index(), at);
                q.clear_held();
                q.merge_held(child, &node);
                assert!(q.marks.path_dep, "the mark reaches the node either way");
                assert_eq!(q.causes.plies(), kept, "from line index {at}");
            }
        }

        /// The superior and inferior verdicts name the earlier position with
        /// the same board and other hands. White drops a pawn the black king
        /// takes, and both kings walk back: six plies on, the board is the
        /// start's and Black holds a pawn it did not — superior for Black, to
        /// move — and, the other way round, a line where Black dropped the pawn
        /// White's king took leaves Black inferior.
        #[cfg_attr(miri, ignore)]
        #[test]
        fn a_hand_verdict_names_the_position_with_the_other_hands() {
            use crate::text_str::parse_usi_move;
            let net = zero_net();
            let _tt = fresh_tt();
            let cases = [
                (
                    "4k4/9/9/9/9/9/9/9/4K4 b p 1",
                    "5i4h P*5h 4h5h 5a5b 5h5i 5b5a",
                    RepetitionState::Superior,
                ),
                (
                    "4k4/9/9/9/9/9/9/9/4K4 b P 1",
                    "P*5b 5a5b 5i5h 5b4a 5h5i 4a5a",
                    RepetitionState::Inferior,
                ),
            ];
            for (sfen, line, verdict) in cases {
                let mut p = pos(sfen);
                for usi in line.split(' ') {
                    let mv = parse_usi_move(usi, &p).expect("a legal move");
                    p.do_move(mv);
                }
                assert_eq!(p.is_repetition(), verdict, "{line}");
                let mut q = QSearch::new(net.network());
                q.root_us = p.side_to_move();
                q.pv_node = false;
                q.qsearch_at(&mut p, 6, -1, 0);
                assert!(q.marks.path_dep, "{verdict:?}");
                assert_eq!(q.causes.plies(), &[0], "{verdict:?} names the start");
                assert!(!q.causes.is_open());
            }
        }

        /// [`shuffled_two_kings`]' moves, in order.
        fn shuffled_two_kings_moves() -> Vec<Move> {
            let bk = Piece::new(PieceKind::King, Color::Black);
            let wk = Piece::new(PieceKind::King, Color::White);
            let step = |from: (u8, u8), to: (u8, u8), pc: Piece| {
                Move::make(
                    Square::new(from.0, from.1).unwrap(),
                    Square::new(to.0, to.1).unwrap(),
                    pc,
                )
            };
            let cycle = [
                step((4, 8), (3, 8), bk),
                step((4, 0), (3, 0), wk),
                step((3, 8), (4, 8), bk),
                step((3, 0), (4, 0), wk),
            ];
            cycle.iter().chain(cycle.iter()).copied().collect()
        }

        /// The move limit is a rule of the tournament, so the draw it forces is
        /// marked; the `MAX_PLY` search horizon in the same condition is not a
        /// rule and leaves the value unmarked.
        #[cfg_attr(miri, ignore)]
        #[test]
        fn the_move_limit_draw_is_marked_and_the_ply_horizon_is_not() {
            let net = zero_net();
            let _tt = fresh_tt();
            let mut q = QSearch::new(net.network());
            q.pv_node = false;

            // The horizon draw, at a game ply well inside any configured move
            // limit, so only `ply >= MAX_PLY` can have produced it.
            let mut p = pos(TWO_KINGS);
            q.root_us = p.side_to_move();
            q.nodes = 0;
            q.qsearch_at(&mut p, MAX_PLY, -1, 0);
            assert_eq!(
                q.marks,
                ValueMarks::NONE,
                "the search's own ply horizon is not a rule of the game"
            );

            // The move-limit draw. `max_moves_to_draw = 0` means unlimited, and
            // then no game ply reaches a limit at all: this branch is compiled
            // out and the node is an ordinary one.
            const GAME_PLY: u16 = 60;
            let mut past = pos("4k4/9/9/9/9/9/9/9/4K4 b - 60");
            q.root_us = past.side_to_move();
            q.nodes = 0;
            q.qsearch_at(&mut past, 0, -1, 0);
            if is_past_move_limit(GAME_PLY) {
                assert_eq!(
                    q.marks,
                    ValueMarks {
                        move_limit: true,
                        ..ValueMarks::NONE
                    },
                    "past the limit the draw carries the move-limit mark alone"
                );
            } else {
                assert_eq!(
                    q.marks,
                    ValueMarks::NONE,
                    "an unreached limit leaves an ordinary, unmarked node"
                );
            }
        }

        /// A marked entry read back as a node's value re-enters the search
        /// marked, and the node that took it hands the mark to its parent when
        /// the child moved the parent's window. Here the child cuts the parent
        /// off, and the parent's own entry is stored marked — a round trip of
        /// every flag through the table.
        #[cfg_attr(miri, ignore)]
        #[test]
        fn a_marked_child_that_moves_the_window_marks_the_node() {
            let net = zero_net();
            let p = pos(IN_CHECK_WITH_EVASIONS);
            assert!(p.in_check(), "the fixture must be in check");
            assert!(!legal_moves(&p).is_empty(), "and must have an evasion");

            for marks in EVERY_MARK {
                let table = fresh_tt();
                // Each child cuts off on an upper bound of -500, so the node
                // sees +500 — above its beta of 1.
                seed_every_child(&table, &p, -500, Bound::Upper, marks);

                let (value, node_marks, node_causes) = {
                    let mut q = QSearch::new(net.network());
                    let out = q.run(&mut p.clone(), 0, 1, false, true);
                    (out.value, q.marks, q.causes)
                };
                assert!(value > 0, "the node fails high on the marked child");
                assert_eq!(node_marks, marks, "and returns a value marked {marks:?}");
                // A table entry carries no causes, so a path-dependent value
                // taken from one rests on positions nobody can name any more.
                let expected = if marks.path_dep {
                    PathCauses::OPEN
                } else {
                    PathCauses::NONE
                };
                assert_eq!(node_causes, expected, "the causes of {marks:?}");

                let (found, data) = probe_root(&table, &p);
                assert!(found);
                assert_eq!(
                    data.marks, marks,
                    "the stored entry carries the node's marks"
                );
            }
        }

        /// The complement of the rule: a marked child whose value stays at or
        /// below the alpha of its moment leaves the parent unmarked.
        #[cfg_attr(miri, ignore)]
        #[test]
        fn a_marked_child_at_or_below_alpha_leaves_the_node_unmarked() {
            let net = zero_net();
            let p = pos(IN_CHECK_WITH_EVASIONS);

            for marks in EVERY_MARK {
                let table = fresh_tt();
                // Each child cuts off on a lower bound of +500, so the node
                // sees -500 — never above its alpha of 0.
                seed_every_child(&table, &p, 500, Bound::Lower, marks);

                let (value, node_marks) = {
                    let mut q = QSearch::new(net.network());
                    let out = q.run(&mut p.clone(), 0, 1, false, true);
                    (out.value, q.marks)
                };
                assert!(value < 0, "every child fails low");
                assert_eq!(
                    node_marks,
                    ValueMarks::NONE,
                    "no child's {marks:?} reaches the node"
                );

                let (found, data) = probe_root(&table, &p);
                assert!(found);
                assert_eq!(data.marks, ValueMarks::NONE);
            }
        }

        /// The root shortcut answers out of the declaration rule, so the score
        /// it hands back is marked as that rule's.
        #[cfg_attr(miri, ignore)]
        #[test]
        fn a_root_declaration_marks_the_score_it_returns() {
            let net = zero_net();
            let _tt = fresh_tt();
            let p = pos("+R+R+B+B5/3GKG3/2SGGGS2/9/9/9/9/9/4k4 b R 1");
            let mut q = QSearch::new(net.network());
            let out = q.run_root(&p, 1);
            assert_eq!(out.kind, RootKind::DeclarationWin);
            assert_eq!(
                q.marks,
                ValueMarks {
                    decl_rule: true,
                    ..ValueMarks::NONE
                }
            );
        }

        /// The in-search check one ply down: the child's value is the
        /// declaration rule's, and it reaches the root's entry because it moved
        /// the root's window.
        #[cfg_attr(miri, ignore)]
        #[test]
        fn a_declaration_below_the_root_marks_the_entry_above_it() {
            let net = zero_net();
            let table = fresh_tt();
            let p = pos(WHITE_TO_MOVE_BEFORE_A_BLACK_DECLARATION);
            assert!(
                declaration_win(&p, &EnteringKingConfig::new(&p)).is_none(),
                "the side to move must have nothing to declare"
            );

            let out = {
                let mut q = QSearch::new(net.network());
                q.run_root(&p, 2)
            };
            assert_eq!(out.kind, RootKind::Normal);

            let (found, data) = probe_root(&table, &p);
            assert!(found, "the root stores its entry");
            assert!(
                data.marks.decl_rule,
                "the declaration one ply down reaches the root's entry"
            );
            assert!(!data.marks.path_dep);
        }
    }

    // ReadTT=false ignores hits; determinism.

    #[cfg_attr(miri, ignore)]
    #[test]
    fn read_tt_false_ignores_a_cutoff_entry() {
        let net = zero_net();
        let table = fresh_tt();
        let p = pos(TWO_KINGS);
        // A lower-bound entry at DEPTH_QS with value 500 >= beta triggers the
        // non-PV early cutoff when ReadTT is honoured.
        prewrite(&table, &p, 500, false, Bound::Lower, DEPTH_QS, None, 0);

        let with_tt = {
            let mut q = QSearch::new(net.network());
            q.run(&mut p.clone(), 399, 400, false, true)
        };
        assert_eq!(with_tt.value, 500, "TT cutoff returns the stored bound");
        assert_eq!(with_tt.nodes, 0);

        let without_tt = {
            let mut q = QSearch::new(net.network());
            q.run(&mut p.clone(), 399, 400, false, false)
        };
        assert_eq!(without_tt.value, 0, "ReadTT=false ignores the entry");
    }

    /// A stored move that does not rebuild in `p`: a board move of another
    /// position whose origin square is empty in `p`.
    fn foreign_move16(p: &Position) -> NonZeroU16 {
        let m = legal_moves(&pos(THREE_CAPTURES))
            .into_iter()
            .find(|m| !m.is_drop() && p.board().get(m.from_sq()).is_none())
            .expect("a board move from a square empty in `p`");
        let m16 = move16_of(Some(m)).expect("a storable move");
        assert!(p.to_move(m16).is_none(), "the move must not rebuild in `p`");
        m16
    }

    /// An entry whose move does not rebuild in the position is what another
    /// position colliding on the same tag leaves: the probe skips it, so
    /// qsearch evaluates instead of cutting on its value. The same entry with
    /// no move still cuts.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn qsearch_evaluates_past_an_entry_whose_move_does_not_fit() {
        let net = zero_net();
        let table = fresh_tt();
        let p = pos(TWO_KINGS);
        let run = |mv: Option<NonZeroU16>| {
            table.clear();
            prewrite(&table, &p, 500, false, Bound::Lower, DEPTH_QS, mv, 0);
            let mut q = QSearch::new(net.network());
            q.run(&mut p.clone(), 399, 400, false, true).value
        };
        assert_eq!(run(None), 500, "an entry with no move is a hit");
        assert_eq!(run(Some(foreign_move16(&p))), 0, "the stand pat, not 500");
    }

    /// [`qsearch_evaluates_past_an_entry_whose_move_does_not_fit`] for the
    /// main search's step-5 cutoff.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn search_evaluates_past_an_entry_whose_move_does_not_fit() {
        let net = zero_net();
        let table = fresh_tt();
        let p = pos(TWO_KINGS);
        let run = |mv: Option<NonZeroU16>| {
            table.clear();
            prewrite(&table, &p, 500, false, Bound::Lower, 10, mv, 0);
            let mut q = QSearch::new(net.network());
            let v = q.run_search(&mut p.clone(), 399, 400, 3, true, false);
            (v, q.nodes)
        };
        assert_eq!(run(None), (500, 0), "an entry with no move is a hit");
        let (v, nodes) = run(Some(foreign_move16(&p)));
        assert_ne!(v, 500, "the colliding entry's value is not taken");
        assert!(nodes > 0, "the node is searched");
        let (found, data) = probe_root(&table, &p);
        assert!(found, "the node stores its own entry over the skipped one");
        assert_ne!(data.value, 500);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn search_is_deterministic_across_runs() {
        let net = zero_net();
        let run_once = || {
            let _tt = fresh_tt();
            let mut q = QSearch::new(net.network());
            let out = q.run(&mut pos(THREE_CAPTURES), 0, 1, false, true);
            (out.value, out.nodes)
        };
        assert_eq!(run_once(), run_once());
    }

    // The reference folds EVERY worker's `bestMoveChanges` into
    // `totBestMoveChanges` and zeroes each, on the main thread only.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn fold_best_move_changes_sums_and_zeroes_every_slot() {
        let net = zero_net();
        let _tt = fresh_tt();

        // Four workers; the main worker (slot 0) folds them all.
        static SLOTS: [TallySlot; 4] = [const { TallySlot::new() }; 4];
        let slots: &'static [TallySlot] = &SLOTS;
        slots[0].store(2, Ordering::Relaxed);
        slots[1].store(3, Ordering::Relaxed);
        slots[2].store(0, Ordering::Relaxed);
        slots[3].store(5, Ordering::Relaxed);

        let mut main = QSearch::new(net.network());
        main.set_best_move_tally(slots, 0);
        let mut tot = 1.0; // a pre-existing aged statistic is added to, not replaced
        main.fold_best_move_changes(&mut tot);
        assert_eq!(tot, 1.0 + 10.0, "main folds every worker's count (2+3+0+5)");
        for (i, s) in slots.iter().enumerate() {
            assert_eq!(s.load(Ordering::Relaxed), 0, "slot {i} reset by the fold");
        }

        // A helper (slot != 0) neither folds nor resets — the main worker owns that.
        slots[1].store(7, Ordering::Relaxed);
        let mut helper = QSearch::new(net.network());
        helper.set_best_move_tally(slots, 1);
        let mut htot = 4.0;
        helper.fold_best_move_changes(&mut htot);
        assert_eq!(htot, 4.0, "a helper folds nothing");
        assert_eq!(
            slots[1].load(Ordering::Relaxed),
            7,
            "a helper leaves its slot for the main worker to read+zero"
        );

        // The single-worker path (no tally) folds its own scalar and zeroes it,
        // reaching the same total as the multi-worker fold.
        let mut solo = QSearch::new(net.network());
        solo.best_move_changes = 6.0;
        let mut stot = 2.0;
        solo.fold_best_move_changes(&mut stot);
        assert_eq!(stot, 8.0);
        assert_eq!(solo.best_move_changes, 0.0);
    }

    // Depth-1 root — the pre-search skipped exits. Both return before any
    // evaluation or do_move, so a synthetic zero network suffices.

    #[cfg_attr(miri, ignore)]
    #[test]
    fn run_root_resigns_with_no_legal_move() {
        let net = zero_net();
        let _tt = fresh_tt();
        // Black is checkmated: no legal move ⇒ bestmove resign.
        let p = pos("4K4/3ggg3/4k4/9/9/9/9/9/9 b - 1");
        let out = {
            let mut q = QSearch::new(net.network());
            q.run_root(&p, 1)
        };
        assert_eq!(out.kind, RootKind::Resign);
        assert_eq!(out.best_move, Move::resign());
        assert_eq!(out.score, mated_in(1));
        assert_eq!(out.nodes, 0);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn run_root_declares_a_nyugyoku_win() {
        let net = zero_net();
        let _tt = fresh_tt();
        // Black king on 5b inside the enemy field with a 32-point entering-king
        // score and legal moves available (so resign does not take precedence).
        let p = pos("+R+R+B+B5/3GKG3/2SGGGS2/9/9/9/9/9/4k4 b R 1");
        let out = {
            let mut q = QSearch::new(net.network());
            q.run_root(&p, 1)
        };
        assert_eq!(out.kind, RootKind::DeclarationWin);
        assert_eq!(out.best_move, Move::win());
        let want = if FLATTEN { VALUE_MATE_FLAT } else { mate_in(1) };
        assert_eq!(out.score, want);
        assert_eq!(out.nodes, 0);
    }

    // MaxMovesToDraw — the compiled setting. The fixtures below sit at a game
    // ply the configured limit may or may not be under, so each asserts the
    // behaviour of its own side of it.

    #[test]
    fn an_unlimited_setting_leaves_no_limit_to_reach() {
        if crate::config::MAX_MOVES_TO_DRAW == 0 {
            assert_eq!(MOVE_LIMIT, None);
            // No ply a position can carry is adjudicated a draw by a limit,
            // including the widest one the counter can hold.
            assert!(!is_past_move_limit(0));
            assert!(!is_past_move_limit(u16::MAX));
        } else {
            let limit = MOVE_LIMIT
                .expect("a non-zero setting configures a limit")
                .get();
            assert_eq!(i64::from(limit), crate::config::MAX_MOVES_TO_DRAW);
            // The limit is the last ply still played; the one past it draws.
            // A limit a `u16` ply counter cannot reach has no such pair.
            if let Some(past) = u16::try_from(limit).ok().and_then(|l| l.checked_add(1)) {
                assert!(!is_past_move_limit(past - 1));
                assert!(is_past_move_limit(past));
            }
        }
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn qsearch_adjudicates_a_draw_past_the_move_limit() {
        // Startpos at game ply 60, Black to move. Past the limit the ply-0
        // qsearch node adjudicates an unconditional draw before any eval or
        // `do_move`, worth `draw contempt + value_draw(0)`; below it the node
        // runs a real qsearch, which at startpos has no capture and stands pat
        // on the zero-eval network.
        const GAME_PLY: u16 = 60;
        let net = zero_net();
        let _tt = fresh_tt();
        let mut p = pos("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 60");
        let out = {
            let mut q = QSearch::new(net.network());
            q.run(&mut p, -VALUE_INFINITE, VALUE_INFINITE, true, true)
        };
        if is_past_move_limit(GAME_PLY) {
            assert_eq!(
                out.value,
                DRAW_CONTEMPT_BLACK + value_draw(0),
                "forced-draw value = draw contempt + value_draw(0)"
            );
            assert_eq!(
                out.nodes, 0,
                "the move-limit draw returns before any do_move"
            );
        } else {
            assert_eq!(
                out.value, 0,
                "an unreached limit ⇒ zero-eval stand-pat, not the draw exit"
            );
        }
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn run_root_sees_a_mate_only_below_the_move_limit() {
        // A gold-drop head mate (`G*8a`) at game ply 100: the White king on 9a
        // is not in check at the root, and dropping the supported gold beside it
        // is mate. Past the limit every interior node adjudicates a draw before
        // the mate is seen.
        const GAME_PLY: u16 = 100;
        let net = zero_net();
        let p = pos("k8/9/G1N6/9/9/9/9/9/8K b G 100");
        let _tt = fresh_tt();
        let out = {
            let mut q = QSearch::new(net.network());
            q.run_root(&p, 2)
        };
        if is_past_move_limit(GAME_PLY) {
            assert!(
                !is_decisive(out.score),
                "the move limit must suppress the mate, got score {}",
                out.score
            );
            assert!(
                out.score.abs() <= DRAW_CONTEMPT_BLACK.abs() + 1,
                "capped score must be in the draw band, got {}",
                out.score
            );
        } else {
            assert!(
                is_win(out.score),
                "an unreached limit must find the mate, got score {}",
                out.score
            );
        }
    }

    // Interior main search (`QSearch::search`). Values here are hand-computed
    // against the reference's formulas, driven by the zero-eval network so every
    // static eval is 0.

    #[cfg_attr(miri, ignore)]
    #[test]
    fn reductions_table_and_reduction_formula() {
        let net = zero_net();
        let _tt = fresh_tt();
        let mut q = QSearch::new(net.network());

        // REDUCTIONS[i] == int(2763/128.0 * ln(i)), REDUCTIONS[0] == 0.
        assert_eq!(REDUCTIONS[0], 0);
        assert_eq!(REDUCTIONS[1], 0); // ln(1) == 0
        assert_eq!(REDUCTIONS[2], 14);
        assert_eq!(REDUCTIONS[3], 23);
        assert_eq!(REDUCTIONS[4], 29);
        assert_eq!(REDUCTIONS[8], 44);
        assert_eq!(REDUCTIONS[10], 49);

        // The table is built by the constant evaluator, which cannot call
        // `f64::ln`; every entry must still be what that call would have given.
        for (i, &r) in REDUCTIONS.iter().enumerate().skip(1) {
            assert_eq!(
                r,
                (2763.0 / 128.0 * (i as f64).ln()) as i32,
                "REDUCTIONS[{i}] diverges from the library logarithm"
            );
        }

        // reduction(i, d, mn, delta) = rs - delta*585/rootDelta
        //                            + (!i)*rs*206/512 + 1133, rs = red[d]*red[mn].
        q.set_root_delta(1000);
        let rs = REDUCTIONS[8] * REDUCTIONS[4];
        assert_eq!(q.reduction(true, 8, 4, 100), rs - 100 * 585 / 1000 + 1133);
        assert_eq!(
            q.reduction(false, 8, 4, 100),
            rs - 100 * 585 / 1000 + rs * 206 / 512 + 1133,
        );
        q.set_root_delta(200);
        let rs2 = REDUCTIONS[10] * REDUCTIONS[2];
        assert_eq!(
            q.reduction(false, 10, 2, 50),
            rs2 - 50 * 585 / 200 + rs2 * 206 / 512 + 1133,
        );
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn a_width_no_window_could_have_reduces_against_one() {
        let net = zero_net();
        let _tt = fresh_tt();
        let mut q = QSearch::new(net.network());

        // No aspiration window closes to zero or inverts, and the width the
        // divisor is built from stays positive when one does: `reduction`
        // divides, and a zero or negative divisor is where that division would
        // fault rather than reduce.
        for width in [0, -1, Value::MIN] {
            q.set_root_delta(width);
            assert_eq!(q.root_delta_divisor(), 1, "width {width}");
            let rs = REDUCTIONS[8] * REDUCTIONS[4];
            assert_eq!(q.reduction(true, 8, 4, 100), rs - 100 * 585 + 1133);
        }

        q.set_root_delta(2 * VALUE_INFINITE);
        assert_eq!(q.root_delta_divisor(), 2 * VALUE_INFINITE);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn move_stat_score_capture_and_quiet() {
        let net = zero_net();
        let _tt = fresh_tt();
        let q = QSearch::new(net.network());
        let s = TestQSearch::si(2); // sentinels (ss-1)/(ss-2) below it exist.

        // Quiet: 2*mainHistory(0) + contHist[0] + contHist[1]. Both continuation
        // planes are the sentinel plane 0, filled -523.
        let bp = Piece::new(PieceKind::Pawn, Color::Black);
        let quiet = Move::make(Square::new(4, 4).unwrap(), Square::new(4, 3).unwrap(), bp);
        // 2*mainHistory(0) == 0, plus the two sentinel continuation planes.
        assert_eq!(
            q.move_stat_score(Color::Black, bp, quiet, s, false, None),
            -523 + -523,
        );

        // Capture: 863*PieceValue[pawn]/128 + captureHistory init(-678).
        let wp = Piece::new(PieceKind::Pawn, Color::White);
        let rook = Piece::new(PieceKind::Rook, Color::Black);
        let cap = Move::make(Square::new(4, 4).unwrap(), Square::new(4, 3).unwrap(), rook);
        assert_eq!(
            q.move_stat_score(Color::Black, rook, cap, s, true, Some(wp)),
            863 * 90 / 128 - 678,
        );
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn razoring_boundary_returns_qsearch_without_do_move() {
        let net = zero_net();
        // eval == 0, depth 1 ⇒ razoring fires when alpha - 502 - 306 > 0, i.e.
        // alpha > 808. It returns qsearch<NonPV>, which for two bare kings makes
        // no do_move.
        let tt = fresh_tt();
        let fired_nodes = {
            let mut q = QSearch::new(net.network());
            q.run_search(&mut pos(TWO_KINGS), 809, 810, 1, false, false);
            q.nodes
        };
        assert_eq!(fired_nodes, 0, "razoring returns qsearch with no do_move");

        // At the boundary (alpha 808 ⇒ 0 < 0 is false) razoring does not fire and
        // the move loop searches the king moves. The same table, emptied: the
        // second half must start where the first did.
        tt.clear();
        let not_nodes = {
            let mut q = QSearch::new(net.network());
            q.run_search(&mut pos(TWO_KINGS), 808, 809, 1, false, false);
            q.nodes
        };
        assert!(not_nodes > 0, "at the boundary razoring must not fire");
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn futility_boundary_returns_two_beta_plus_eval_over_three() {
        let net = zero_net();
        // depth 1, !ttHit ⇒ futilityMult 55, improving false, opponentWorsening
        // true ⇒ margin = 55 - 362*55/1024 = 36. Futility fires when eval >= beta
        // and eval - 36 >= beta; with eval 0 that is beta <= -36. It returns
        // (2*beta + eval)/3.
        let tt = fresh_tt();
        let (v, n) = {
            let mut q = QSearch::new(net.network());
            let v = q.run_search(&mut pos(TWO_KINGS), -37, -36, 1, false, false);
            (v, q.nodes)
        };
        assert_eq!(v, (2 * -36) / 3); // -24
        assert_eq!(n, 0, "futility returns before any do_move");

        // beta == -35: eval - margin (-36) < beta, so futility does not fire.
        // The same table, emptied.
        tt.clear();
        let n2 = {
            let mut q = QSearch::new(net.network());
            q.run_search(&mut pos(TWO_KINGS), -36, -35, 1, false, false);
            q.nodes
        };
        assert!(n2 > 0, "at the boundary futility must not fire");
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn null_move_boundary_returns_null_value() {
        let net = zero_net();
        let p = pos(TWO_KINGS);
        // A quiet TT move suppresses the futility return (its `!ttMove` term is
        // false), letting the null-move step be reached. Stored non-cutoff
        // (BOUND_NONE, VALUE_NONE) so Step 4 does not cut and the eval is a fresh
        // evaluate() == 0.
        let quiet = legal_moves(&p)
            .into_iter()
            .find(|&m| !m.is_drop() && p.board().get(m.to_sq()).is_none())
            .expect("a quiet king move exists");

        // depth 1, improving false ⇒ null fires when 0 >= beta + 378 - 16, i.e.
        // beta <= -362. R = 7, depth-R < 0 ⇒ the child is qsearch (no do_move).
        let tt = fresh_tt();
        prewrite(
            &tt,
            &p,
            VALUE_NONE,
            false,
            Bound::None,
            DEPTH_UNSEARCHED,
            move16_of(Some(quiet)),
            VALUE_NONE,
        );
        let (v, n) = {
            let mut q = QSearch::new(net.network());
            let v = q.run_search(&mut p.clone(), -363, -362, 1, true, false);
            (v, q.nodes)
        };
        assert_eq!(v, 0, "null search of two kings returns 0");
        assert_eq!(n, 0, "null move + qsearch make no counted do_move");

        // beta == -361: 0 >= -361 - 16 + 378 == 1 is false ⇒ null does not fire.
        // The same table, emptied and re-seeded.
        tt.clear();
        prewrite(
            &tt,
            &p,
            VALUE_NONE,
            false,
            Bound::None,
            DEPTH_UNSEARCHED,
            move16_of(Some(quiet)),
            VALUE_NONE,
        );
        let n2 = {
            let mut q = QSearch::new(net.network());
            q.run_search(&mut p.clone(), -362, -361, 1, true, false);
            q.nodes
        };
        assert!(n2 > 0, "at the boundary null move must not fire");
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn probcut_returns_value_minus_margin() {
        let net = zero_net();
        // Black rook (5e) can capture the undefended white pawn (5d); the white
        // king (1a) is far, so SEE(Rxp) == +90 and there is no recapture. A
        // ttPv-marked non-cutoff TT entry (no move) makes ss->ttPv true so Step 8
        // futility is skipped, letting ProbCut run.
        let p = pos("8k/9/9/4p4/4R4/9/9/9/K8 b - 1");
        assert!(!p.in_check());
        assert_eq!(captures(&p).len(), 1, "exactly one capture");

        let tt = fresh_tt();
        prewrite(
            &tt,
            &p,
            VALUE_NONE,
            true, // is_pv ⇒ ss->ttPv true
            Bound::None,
            DEPTH_UNSEARCHED,
            None,
            VALUE_NONE,
        );
        // depth 4 ⇒ probCutDepth 0 (no verification search); improving becomes
        // true (0 >= beta), so probCutBeta = beta + 224 - 61 = -61. The capture's
        // qsearch value is 0 >= -61, so ProbCut returns value - (probCutBeta -
        // beta) = 0 - (-61 - (-224)) = -163, having made exactly one do_move.
        let (v, n) = {
            let mut q = QSearch::new(net.network());
            let v = q.run_search(&mut p.clone(), -225, -224, 4, false, false);
            (v, q.nodes)
        };
        assert_eq!(v, -163);
        assert_eq!(n, 1, "one ProbCut capture is searched");
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn fail_low_writes_the_documented_bonuses() {
        let net = zero_net();
        let _tt = fresh_tt();
        let mut q = QSearch::new(net.network());

        // Drive `search` directly at ply 1 with a hand-set (ss-1) cell so the
        // fail-low branch (bestMove none, prevSq real, !priorCapture) fires. White
        // to move; the previous (black) move landed the black king (uppercase `K`)
        // on 5a, so `piece_on(prevSq)` is that black king.
        let mut p = pos("4K4/9/9/9/9/9/9/9/4k4 w - 1");
        q.nodes = 0;
        q.sel_depth = 0;
        q.root_us = Color::White;
        q.set_root_delta(2 * VALUE_INFINITE);
        q.root_depth = 1;

        let bk = Piece::new(PieceKind::King, Color::Black);
        let prev_sq = Square::new(4, 0).unwrap(); // 5a, where the black king sits
        let prev = Move::make(Square::new(3, 0).unwrap(), prev_sq, bk);
        let s0 = TestQSearch::si(0);
        q.stack[s0].current_move = Some(prev);
        q.stack[s0].in_check = true; // suppresses the Step-6 eval-diff main update
        q.stack[s0].stat_score = -20000;
        q.stack[s0].move_count = 0;

        let pawn_key = p.pawn_key();
        assert_eq!(q.histories.main.get(Color::Black, prev), 0);
        let pawn_before = q.histories.shared.pawn_get(pawn_key, bk, prev_sq);
        let corr_before =
            q.histories
                .shared
                .correction_get(pawn_key, Color::White, CorrChannel::Pawn);

        // A small zero-window above 0 makes every white king move (value 0) fail
        // low without tripping razoring (which needs eval < alpha - 808).
        let v = q.lend_picker_scratch(|this, scratch| {
            this.search(&mut p, 1, 1, 2, 1, false, false, None, None, scratch)
        });
        assert_eq!(v, 0);

        // bonusScale = -232 - (-20000/108) + 59 = 12; scaledBonus = 55*12 = 660.
        // mainHistory[~White=Black][prev] << 660*235/32768 == 4 (from 0).
        assert_eq!(q.histories.main.get(Color::Black, prev), 4);
        // pawn plane [blackKing][5a] << 660*290/8192 == 23 (moves off -1238).
        assert_ne!(
            q.histories.shared.pawn_get(pawn_key, bk, prev_sq),
            pawn_before
        );

        // Correction-history guard: !inCheck, bestMove is none so
        // (bestValue > staticEval) == false == bool(bestMove); the guard fires but
        // with bestValue == staticEval the bonus is 0, leaving the table unchanged.
        assert_eq!(
            q.histories
                .shared
                .correction_get(pawn_key, Color::White, CorrChannel::Pawn),
            corr_before,
        );
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn interior_smoke_runs_and_leaves_state_balanced() {
        let net = zero_net();
        const SFEN: &str =
            "l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1";
        for depth in [1, 2] {
            let orig = pos(SFEN);
            let mut work = orig.clone();
            let tt = fresh_tt();
            let (v, nodes) = {
                let mut q = QSearch::new(net.network());
                let v = q.run_search(
                    &mut work,
                    -VALUE_INFINITE,
                    VALUE_INFINITE,
                    depth,
                    false,
                    true,
                );
                (v, q.nodes)
            };
            assert!(
                -VALUE_INFINITE < v && v < VALUE_INFINITE,
                "depth {depth}: value {v} out of range"
            );
            assert!(nodes > 0, "depth {depth}: interior body must search moves");
            assert_eq!(work, orig, "depth {depth}: the position stack must balance");
            let (found, _data) = probe_root(&tt, &orig);
            assert!(found, "depth {depth}: the root node writes a TT entry");
        }
    }

    // Torn-entry totality: under Lazy SMP several workers share the TT through
    // relaxed atomics, so a decoded `TTData` can pair a stale key fragment with
    // a `move16` written for a different position — leaving the stored `move16`
    // an arbitrary `u16`. These tests drive every pattern the widen gate can
    // receive — all but zero, which the entry reads as "no move" and which
    // therefore never reaches the gate — and require no panic and every
    // accepted move to be legal.

    /// The 65535 fragments a torn entry can hand the widen gate.
    fn torn_fragments() -> impl Iterator<Item = NonZeroU16> {
        (1u32..=0xFFFF).map(|bits| NonZeroU16::new(bits as u16).expect("the range starts at one"))
    }

    /// The 6 parity-fixture SFENs, covering an in-check position and several
    /// hand-heavy ones; the test asserts that coverage explicitly below.
    const TORN_ENTRY_SFENS: &[&str] = &[
        // startpos
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        // drop-heavy — pieces in hand for both sides
        "k8/1P7/G8/1N2P4/9/9/9/9/8K b 2PG2pg 1",
        // mid-game-tactical — pieces in hand, dense board
        "l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1",
        // check-evasion — side to move is IN CHECK, several pieces in hand
        "4k4/9/4r4/9/9/9/4K3B/9/9 b RG2gs2n3p 1",
        // promotion-zone-edges — promoted pieces, near the back ranks
        "4k4/3P3+PL/2N2PR2/1L2BNS2/4N4/9/9/9/4K4 b - 1",
        // sennichite base — 18 pawns in hand
        "9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1",
    ];

    fn total_hand_count(p: &Position) -> u32 {
        let h = p.hand(p.side_to_move());
        [
            PieceKind::Pawn,
            PieceKind::Lance,
            PieceKind::Knight,
            PieceKind::Silver,
            PieceKind::Gold,
            PieceKind::Bishop,
            PieceKind::Rook,
        ]
        .iter()
        .map(|&k| h.count(k) as u32)
        .sum()
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn torn_tt_move_decode_is_total_over_all_patterns() {
        let positions: Vec<Position> = TORN_ENTRY_SFENS.iter().map(|s| pos(s)).collect();

        // The set must genuinely exercise both hard cases: a position where the
        // side to move is in check, and one with several pieces in hand.
        assert!(
            positions.iter().any(|p| p.in_check()),
            "position set must include an in-check case"
        );
        assert!(
            positions.iter().any(|p| total_hand_count(p) >= 3),
            "position set must include a hand-heavy case"
        );

        let hist = WorkerHistories::new();

        for (p, sfen) in positions.iter().zip(TORN_ENTRY_SFENS) {
            let mut legal = Vec::new();
            p.generate_legal_all(&mut legal);
            let legal_set: std::collections::HashSet<Move> = legal.iter().copied().collect();

            // `select_tt_move` is the exact code `widen_tt_move` runs after
            // generation, so sweeping it over one generated list covers every
            // pattern.
            let mut accepted: std::collections::HashSet<Move> = std::collections::HashSet::new();
            for m16 in torn_fragments() {
                if let Some(m) = TestQSearch::select_tt_move(&legal, m16) {
                    assert!(
                        legal_set.contains(&m),
                        "{sfen}: select_tt_move accepted {m:?} for move16={m16:#06x}, not legal"
                    );
                    assert_eq!(
                        move16_of(Some(m)),
                        Some(m16),
                        "{sfen}: select_tt_move({m16:#06x}) returned a move with a different fragment"
                    );
                    accepted.insert(m);
                }
            }

            // Generation is pattern-independent, so agreeing with the real
            // `widen_tt_move` on every accepted fragment and a strided sample
            // of rejecting ones transfers the sweep's totality to the real
            // decode without 65535 re-generations.
            let mut sample: Vec<NonZeroU16> =
                accepted.iter().filter_map(|&m| m.move16_stored()).collect();
            sample.extend(torn_fragments().step_by(97));
            for m16 in sample {
                assert_eq!(
                    TestQSearch::widen_tt_move(p, m16),
                    TestQSearch::select_tt_move(&legal, m16),
                    "{sfen}: widen_tt_move and select_tt_move disagree on move16={m16:#06x}"
                );
            }

            // The MovePicker TT-move stage is a pure function of
            // `(pos, widened_move)`, and `widened_move` ranges over exactly
            // `{None} ∪ accepted`, so that finite input set covers every
            // pattern.
            let mut inputs: Vec<Option<Move>> = vec![None];
            inputs.extend(accepted.iter().map(|&m| Some(m)));
            let mut q_scratch = PickerScratch::new();
            let mut main_scratch = PickerScratch::new();
            for tt_move in inputs {
                for mut mp in [
                    MovePicker::new_qsearch(p, tt_move, [ContPlane::SENTINEL; 6], &mut q_scratch),
                    MovePicker::new_main_search(
                        p,
                        tt_move,
                        6,
                        0,
                        [ContPlane::SENTINEL; 6],
                        &mut main_scratch,
                    ),
                ] {
                    while let Some(m) = mp.next_move(p, &hist) {
                        assert!(
                            legal_set.contains(&m),
                            "{sfen}: MovePicker yielded illegal {m:?} for tt_move={tt_move:?}"
                        );
                    }
                }
            }
        }
    }

    // Production widens a TT fragment through the reference chain
    // `Position::to_move` → `Position::pseudo_legal` → `Position::is_legal`.
    // These tests pin it against the generate-and-match `widen_tt_move`, kept
    // below as the `#[cfg(test)]` oracle.

    /// The strict search-generated legal set — the moves `pseudo_legal(_, false)`
    /// admits, i.e. the moves the search generators actually produce and thus
    /// the only moves that can ever be stored as a TT move. Unlike
    /// [`Position::generate_legal_all`] it prunes the "useless" non-promotions
    /// and drops onto cannot-move squares.
    fn strict_search_legal(p: &Position) -> Vec<Move> {
        let mut pseudo: Vec<yorkie_state::ExtMove> = Vec::new();
        if p.in_check() {
            p.generate_evasions::<false, _>(&mut pseudo);
        } else {
            p.generate_captures::<false, _>(&mut pseudo);
            p.generate_quiets::<false, _>(&mut pseudo);
        }
        pseudo
            .into_iter()
            .map(|e| e.mv)
            .filter(|&m| p.is_legal(m))
            .collect()
    }

    /// The round-trip oracle at one position: every strict search-legal move
    /// (the only moves that reach the TT) round-trips through `to_move`, is
    /// `pseudo_legal(all=false)`, and the production chain accepts exactly it.
    fn legal_move_chain_oracle(p: &Position, ctx: &str) {
        for m in strict_search_legal(p) {
            let f = m
                .move16_stored()
                .expect("a real move has a non-zero fragment");
            assert_eq!(
                p.to_move(f),
                Some(m),
                "{ctx}: to_move({f:#06x}) != its own move"
            );
            assert!(
                p.pseudo_legal::<false>(m),
                "{ctx}: strict-legal {m:?} is not pseudo_legal(all=false)"
            );
            assert!(
                p.pseudo_legal::<true>(m),
                "{ctx}: strict-legal {m:?} is not pseudo_legal(all=true)"
            );
            assert!(p.is_legal(m), "{ctx}: strict-legal {m:?} is not legal");
            // The production chain (all=false) accepts exactly `m` for `f`.
            let accepted = p
                .to_move(f)
                .filter(|&mm| mm.is_ok() && p.pseudo_legal::<false>(mm) && p.is_legal(mm));
            assert_eq!(
                accepted,
                Some(m),
                "{ctx}: chain does not accept {m:?} for {f:#06x}"
            );
        }
    }

    /// The totality check and the round-trip oracle at one position.
    fn widen_chain_full_gates(p: &Position, ctx: &str) {
        legal_move_chain_oracle(p, ctx);

        // The all-legal set is exactly what the generate-and-match widen oracle
        // accepted, and the strict set is what the production chain admits.
        let mut perft_legal = Vec::new();
        p.generate_legal_all(&mut perft_legal);
        let perft_set: std::collections::HashSet<Move> = perft_legal.iter().copied().collect();
        let strict: std::collections::HashSet<Move> = strict_search_legal(p).into_iter().collect();

        // A torn drop fragment carrying a stray promote bit widens to the same
        // clean drop, so `move16` is not asserted to round-trip over the full
        // sweep — only over real moves, in `legal_move_chain_oracle`.
        fn acceptances<const ALL: bool>(p: &Position) -> usize {
            torn_fragments()
                .filter(|&m16| {
                    p.to_move(m16)
                        .is_some_and(|m| m.is_ok() && p.pseudo_legal::<ALL>(m) && p.is_legal(m))
                })
                .count()
        }
        for (all, accepted) in [
            (false, acceptances::<false>(p)),
            (true, acceptances::<true>(p)),
        ] {
            assert!(
                accepted >= strict.len(),
                "{ctx}: {accepted} acceptances (all={all}) < {} strict-legal moves",
                strict.len()
            );
        }

        // `select_tt_move` accepts perft-legal moves, under lenient promotion
        // rules, so the comparison runs under `all == true`.
        for m16 in torn_fragments() {
            if let Some(old) = TestQSearch::select_tt_move(&perft_legal, m16) {
                let new = p
                    .to_move(m16)
                    .filter(|&m| m.is_ok() && p.pseudo_legal::<true>(m) && p.is_legal(m));
                assert_eq!(
                    new,
                    Some(old),
                    "{ctx}: select_tt_move accepted {old:?} for {m16:#06x}, the to_move chain gives {new:?}"
                );
            }
        }

        // Both sets are repetition-blind, so the only reason `strict` is
        // smaller is its promotion / cannot-move pruning.
        let missing: Vec<Move> = strict.difference(&perft_set).copied().collect();
        assert!(
            missing.is_empty(),
            "{ctx}: strict search-legal moves absent from the all-legal set: {missing:?}"
        );
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn to_move_widen_chain_totality_and_oracle() {
        for sfen in TORN_ENTRY_SFENS {
            let p = pos(sfen);
            widen_chain_full_gates(&p, sfen);
        }
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn to_move_widen_chain_oracle_over_playouts() {
        // The round-trip oracle must hold along a deterministic
        // playout of >= 30 plies from each fixture. The move choice rotates by
        // ply so the line advances rather than shuffling in place.
        for sfen in TORN_ENTRY_SFENS {
            let mut p = pos(sfen);
            for ply in 0..30usize {
                legal_move_chain_oracle(&p, &format!("{sfen} @ ply {ply}"));
                let mut legal = Vec::new();
                p.generate_legal_all(&mut legal);
                if legal.is_empty() {
                    break; // terminal position (mate / stalemate)
                }
                let pick = legal[ply % legal.len()];
                let _ = p.do_move(pick);
            }
        }
    }

    /// The per-game evaluation noise: what the offset a position draws is
    /// allowed to be, and that it is the same one every time within a game.
    #[cfg(feature = "random")]
    mod evaluation_noise {
        use super::*;

        /// A spread of keys and seeds, including the degenerate ones (0, all
        /// ones) and a pair differing in a single bit.
        const KEYS: &[u64] = &[
            0,
            1,
            2,
            u64::MAX,
            0x9E37_79B9_7F4A_7C15,
            0x0123_4567_89AB_CDEF,
        ];
        const SEEDS: &[u64] = &[0, 1, u64::MAX, 0xDEAD_BEEF_CAFE_F00D];

        #[test]
        fn one_key_and_seed_always_give_the_same_offset() {
            for &key in KEYS {
                for &seed in SEEDS {
                    let first = noise(key, seed, 90);
                    for _ in 0..8 {
                        assert_eq!(noise(key, seed, 90), first);
                    }
                }
            }
        }

        #[test]
        fn the_offset_stays_inside_the_amplitude() {
            for amplitude in [1, 2, 3, 9, 45, 90, 900] {
                let (lo, hi) = (-(amplitude / 2), amplitude - 1 - amplitude / 2);
                for &key in KEYS {
                    for &seed in SEEDS {
                        let n = noise(key, seed, amplitude);
                        assert!(
                            (lo..=hi).contains(&n),
                            "amplitude {amplitude}: {n} outside [{lo}, {hi}]"
                        );
                    }
                }
            }
        }

        /// An amplitude of one leaves exactly one offset to draw, and it is
        /// zero — so the narrowest noise the setting can ask for is still no
        /// bias in either direction.
        #[test]
        fn an_amplitude_of_one_is_always_zero() {
            for &key in KEYS {
                for &seed in SEEDS {
                    assert_eq!(noise(key, seed, 1), 0);
                }
            }
        }

        /// Zero amplitude is the off switch: no key, no seed, no offset.
        #[test]
        fn a_zero_amplitude_is_no_noise() {
            for &key in KEYS {
                for &seed in SEEDS {
                    assert_eq!(noise(key, seed, 0), 0);
                }
            }
        }

        /// The seed is what separates one game from the next, so keys that
        /// share it must not share an offset, and one key must not keep its
        /// offset across seeds. A distribution this coarse (`u64` down to 90
        /// values) collides sometimes; what is asserted is that it does not
        /// collapse.
        #[test]
        fn the_seed_and_the_key_both_move_the_offset() {
            let spread = |values: Vec<Value>| {
                let mut v = values;
                v.sort_unstable();
                v.dedup();
                v.len()
            };
            let across_keys = spread((0u64..64).map(|k| noise(k, 12345, 90)).collect());
            assert!(across_keys > 16, "keys collapsed to {across_keys} offsets");
            let across_seeds = spread((0u64..64).map(|s| noise(0x5A5A, s, 90)).collect());
            assert!(
                across_seeds > 16,
                "seeds collapsed to {across_seeds} offsets"
            );
        }

        /// Neighbouring keys — which is what positions one move apart are —
        /// must not draw neighbouring offsets. That is what the finaliser in
        /// front of the modulus is for.
        #[test]
        fn keys_one_bit_apart_are_not_offsets_one_step_apart() {
            let base = noise(0x0123_4567_89AB_CDEF, 7, 90);
            let neighbours: Vec<Value> = (0..64)
                .map(|b| noise(0x0123_4567_89AB_CDEF ^ (1u64 << b), 7, 90))
                .collect();
            let adjacent = neighbours
                .iter()
                .filter(|n| (**n - base).abs() <= 1)
                .count();
            assert!(
                adjacent < 8,
                "{adjacent} of 64 one-bit neighbours landed within one of {base}"
            );
        }

        /// What the noise looks like from a search: the zero network evaluates
        /// every position at 0, so whatever a search returns here is the offset
        /// and nothing else. A search given no amplitude returns the plain 0 —
        /// which is what leaves every existing fixed-depth path in this crate
        /// unchanged when the feature is compiled in.
        #[cfg_attr(miri, ignore)]
        #[test]
        fn a_search_carries_the_offset_of_the_amplitude_and_seed_it_was_given() {
            let net = zero_net();
            let p = pos("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");
            let run = |amplitude: Value, seed: u64| {
                let _tt = fresh_tt();
                let mut q = QSearch::new(net.network());
                q.set_random(amplitude, seed);
                let mut work = p.clone();
                q.run(&mut work, -VALUE_INFINITE, VALUE_INFINITE, true, true)
                    .value
            };

            assert_eq!(run(0, 0x1234_5678), 0, "no amplitude is no noise");
            assert_eq!(
                run(90, 0x1234_5678),
                run(90, 0x1234_5678),
                "one seed evaluates one position one way"
            );
            assert_ne!(
                run(90, 0x1234_5678),
                run(90, 0x8765_4321),
                "a new seed is what makes the next game differ"
            );
        }
    }

    /// The widened root of the search that thinks past the engine's reply,
    /// against the real network: a zero network scores every root move alike,
    /// so no move could ever be told apart as far below the rest. Skipped when
    /// `nn.bin` is absent.
    mod root_width {
        use super::*;
        use crate::text_str::{format_usi_move, parse_usi_move};

        const STARTPOS: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

        /// The opening after 7g7f 3c3d: the two bishops face each other, and
        /// Black's on 8h is guarded by the silver on 7i alone. Moving that
        /// silver away hangs the bishop to 2b8h+; the other moves keep it.
        const HANGING_BISHOP: &str =
            "lnsgkgsnl/1r5b1/pppppp1pp/6p2/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL b - 3";
        const HANGING_MOVES: [&str; 2] = ["7i6h", "7i7h"];
        /// Moves that keep the bishop guarded.
        const SAFE_MOVES: [&str; 2] = ["2g2f", "6i7h"];

        /// Deep enough that a root move's subtree is more than a handful of
        /// nodes, so a full-depth one stands apart from a reduced one.
        const DEPTH: i32 = 10;

        /// No margin a position can open: nothing is ever far.
        const NO_MARGIN: Value = 30_000;
        /// Five pawns: less than the bishop a hanging move loses, more than
        /// what the other moves of the opening differ by.
        const BISHOP_MARGIN: Value = 450;

        struct Searched {
            best: Move,
            nodes: u64,
            root_moves: Vec<RootMove>,
        }

        impl Searched {
            fn far(&self) -> Vec<String> {
                let mut far: Vec<String> = self
                    .root_moves
                    .iter()
                    .filter(|rm| rm.far)
                    .map(|rm| format_usi_move(rm.mv))
                    .collect();
                far.sort();
                far
            }

            fn effort(&self, usi: &str) -> u64 {
                self.root_moves
                    .iter()
                    .find(|rm| format_usi_move(rm.mv) == usi)
                    .map(|rm| rm.effort)
                    .expect("a root move")
            }

            /// The median node count of a root move's subtree: what the moves
            /// behind the best one are given.
            fn median_effort(&self) -> u64 {
                let mut efforts: Vec<u64> = self.root_moves.iter().map(|rm| rm.effort).collect();
                efforts.sort_unstable();
                efforts[efforts.len() / 2]
            }
        }

        fn search<N: NetworkParams>(net: N, sfen: &str, width: Option<RootWidth>) -> Searched {
            let _tt = fresh_tt();
            let mut q = QSearch::new(net);
            if let Some(width) = width {
                q.set_root_width(width);
            }
            let out = q.run_root(&pos(sfen), DEPTH);
            assert_eq!(out.depth, DEPTH, "{sfen}: every iteration completes");
            Searched {
                best: out.best_move,
                nodes: out.nodes,
                root_moves: std::mem::take(&mut q.last_root_moves),
            }
        }

        fn width(unreduced: usize, margin: Value) -> RootWidth {
            RootWidth { unreduced, margin }
        }

        fn real_net() -> Option<OwnedNetwork> {
            let path = real_nn_bin();
            if !path.exists() {
                eprintln!("skipping: {} not present", path.display());
                return None;
            }
            Some(yorkie_eval::load_network(&path).expect("real nn.bin loads"))
        }

        #[test]
        fn the_named_moves_are_legal() {
            let p = pos(HANGING_BISHOP);
            for usi in HANGING_MOVES.iter().chain(&SAFE_MOVES) {
                parse_usi_move(usi, &p).expect("a legal move");
            }
            assert_eq!(RootWidth::default(), RootWidth::SINGLE_LINE);
            assert_eq!(RootWidth::SINGLE_LINE.unreduced, 1);
        }

        /// One unreduced root move is the search a `go` runs, whatever the
        /// margin: the same nodes, the same move, nothing marked.
        #[cfg_attr(miri, ignore)]
        #[test]
        fn a_width_of_one_is_the_single_line_search() {
            let Some(net) = real_net() else { return };
            for sfen in [HANGING_BISHOP, STARTPOS] {
                let go = search(net.network(), sfen, None);
                for margin in [BISHOP_MARGIN, NO_MARGIN] {
                    let one = search(net.network(), sfen, Some(width(1, margin)));
                    assert_eq!((one.best, one.nodes), (go.best, go.nodes), "{sfen}");
                    assert!(one.far().is_empty(), "{sfen}");
                }
            }
        }

        /// Reading every root move to full depth costs more nodes and gives the
        /// moves behind the best one many times the subtree the reduced search
        /// gives them, for the same best move. Two unreduced moves are a search
        /// of their own, wider than the single line.
        #[cfg_attr(miri, ignore)]
        #[test]
        fn unreduced_root_moves_are_searched_wider_for_the_same_best_move() {
            let Some(net) = real_net() else { return };
            let one = search(net.network(), STARTPOS, None);
            let two = search(net.network(), STARTPOS, Some(width(2, NO_MARGIN)));
            let every = search(net.network(), STARTPOS, Some(width(0, NO_MARGIN)));

            assert!(
                one.nodes < every.nodes && one.nodes < two.nodes,
                "nodes: one {}, two {}, every {}",
                one.nodes,
                two.nodes,
                every.nodes
            );
            assert_eq!(every.best, one.best);
            assert!(
                every.median_effort() >= 3 * one.median_effort(),
                "median subtree: one {}, every {}",
                one.median_effort(),
                every.median_effort()
            );
            assert!(every.far().is_empty() && two.far().is_empty());
        }

        /// A move that hangs the bishop scores more than five pawns below the
        /// best move, so a search with that margin marks it far and spends on
        /// it no more than about what the single-line search does, while it
        /// goes on reading the moves that lose nothing. A margin wider than any
        /// gap marks nothing and reads the hanging moves to full depth too.
        #[cfg_attr(miri, ignore)]
        #[test]
        fn a_move_far_below_the_best_is_reduced_unless_the_margin_covers_it() {
            let Some(net) = real_net() else { return };
            let one = search(net.network(), HANGING_BISHOP, None);
            let narrow = search(net.network(), HANGING_BISHOP, Some(width(0, BISHOP_MARGIN)));
            let wide = search(net.network(), HANGING_BISHOP, Some(width(0, NO_MARGIN)));

            let far = narrow.far();
            for usi in HANGING_MOVES {
                assert!(far.iter().any(|m| m == usi), "{usi} not in {far:?}");
            }
            for usi in SAFE_MOVES {
                assert!(!far.iter().any(|m| m == usi), "{usi} in {far:?}");
            }
            assert!(wide.far().is_empty(), "{:?}", wide.far());
            assert!(
                narrow.nodes < wide.nodes,
                "nodes: narrow {}, wide {}",
                narrow.nodes,
                wide.nodes
            );

            for usi in HANGING_MOVES {
                let (one, narrow, wide) = (one.effort(usi), narrow.effort(usi), wide.effort(usi));
                assert!(
                    narrow < wide && narrow <= 2 * one.max(1),
                    "{usi}: subtree single-line {one}, narrow {narrow}, wide {wide}"
                );
            }
        }
    }

    /// The move-order spread with two workers, at the nodes whose TT move went
    /// through a singular search: that search re-enters on the node's own ply,
    /// and the node must not inherit what it left pending there.
    #[cfg(lazy_smp_spread)]
    mod spread_singular {
        use super::*;
        use crate::generate_root_moves;

        const POSITIONS: [&str; 3] = [
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
            "l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1",
            "4k4/3P3+PL/2N2PR2/1L2BNS2/4N4/9/9/9/4K4 b - 1",
        ];
        const DEPTH: i32 = 11;

        static SLOTS: [TallySlot; 2] = [const { TallySlot::new() }; 2];

        #[cfg_attr(miri, ignore)]
        #[test]
        fn no_move_is_searched_twice_after_a_singular_search() {
            let path = real_nn_bin();
            if !path.exists() {
                eprintln!("skipping: {} not present", path.display());
                return;
            }
            let owned = yorkie_eval::load_network(&path).expect("real nn.bin loads");
            let net = owned.network();
            let _tt = fresh_tt();
            let mut total = SingularAudit::default();
            for sfen in POSITIONS {
                TranspositionTable::shared().clear();
                TranspositionTable::shared().new_search();
                let p = pos(sfen);
                let p = &p;
                let audits = std::thread::scope(|scope| {
                    [0, 1]
                        .map(|index| {
                            scope.spawn(move || {
                                let mut search = QSearch::new(net);
                                search.set_best_move_tally(&SLOTS, index);
                                search.set_spread(true, 2);
                                search.run_worker(p, generate_root_moves(p), DEPTH);
                                search.singular_audit
                            })
                        })
                        .map(|w| w.join().expect("a worker finishes"))
                });
                for audit in audits {
                    total.nodes += audit.nodes;
                    total.repeats += audit.repeats;
                }
            }
            assert!(total.nodes > 0, "some singular searches ran");
            assert_eq!(total.repeats, 0, "moves searched twice: {total:?}");
        }
    }
}
