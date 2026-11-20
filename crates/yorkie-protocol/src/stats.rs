//! The per-reply statistics line — `info string stats <key>=<value> …`, written
//! immediately before the `bestmove` that ends a search.
//!
//! One line carries everything a reply reports about itself, as space-separated
//! `key=value` items after a fixed prefix. An item whose value is zero is
//! omitted, and a line left with no item is not written at all: a reply that has
//! nothing to report says nothing rather than saying that nothing happened.
//!
//! Every build that writes the line has six counted items, each covering the
//! interval the reply ends: `alloc`, the heap allocations, then `nodes`,
//! `tt_update`, `eval_dup`, `evals` and `eval_tt_hits`, what the searches of
//! that interval did on every thread (see [`SearchCounts`]). A build with the
//! workers' move-order spread compiled in follows them with `spread_dups`,
//! `spread_shifts` and `spread_peak`, and one with the workers' putting off of
//! busy moves compiled in with `defer_moves`, `defer_tt_cuts` and
//! `defer_entries`. A `verbose3` build adds three more: the
//! value marks of the score the reply rests on.
//!
//! A `verbose3` build's `bench` also writes one line after its summary, in the
//! same format, carrying the search counts summed over every position of the
//! run (see [`take_bench_counts`]). A `dupstats` build's `bench` follows it with
//! the same counts for each position on its own, and with where each worker's
//! nodes were in each position's search (see [`take_position_counts`] and
//! [`node_hist_line`]).
//!
//! What the line reports is the interval it ends, so composing it must not
//! disturb what it measures. It is rendered into a stack buffer sized for the
//! longest line it can produce and handed to the writer as those bytes, so
//! nothing on this path reaches the heap.

#[cfg(feature = "dupstats")]
use std::sync::Mutex;
#[cfg(all(feature = "dupstats", feature = "verbose3"))]
use std::sync::PoisonError;
use std::sync::atomic::{AtomicU64, Ordering};

use yorkie_search::WorkerResult;
#[cfg(feature = "verbose3")]
use yorkie_search::causes::{CAUSE_CAPACITY, ReplyMarks};
#[cfg(lazy_smp_defer)]
use yorkie_search::defer::DeferCounts;
#[cfg(feature = "dupstats")]
use yorkie_search::dupstats::{
    DupCounts, EVENT_KEYS, Event, PRESENCE_KEYS, PresenceKind, STORE_KEYS, StoreKind,
};
#[cfg(all(feature = "dupstats", feature = "verbose3"))]
use yorkie_search::dupstats::{NODE_TYPE_NAMES, NodeHist};
#[cfg(lazy_smp_spread)]
use yorkie_search::spread::SpreadCounts;
#[cfg(all(feature = "dupstats", feature = "verbose3"))]
use yorkie_search::timeline::{Iteration, Timeline};
use yorkie_state::TextWriter;
#[cfg(all(test, feature = "verbose3"))]
use yorkie_storage::ValueMarks;

/// What every line starts with.
const PREFIX: &[u8] = b"info string stats";

/// The four counted items' keys, in the order they are written, each with the
/// space that separates it from what comes before.
const COUNT_KEYS: [&[u8]; 6] = [
    b" alloc=",
    b" nodes=",
    b" tt_update=",
    b" eval_dup=",
    b" evals=",
    b" eval_tt_hits=",
];

/// The decimal width of the widest `u64`.
const U64_DIGITS: usize = 20;

/// The counted items at their widest: every key with a 20-digit value.
const COUNT_ITEMS_MAX: usize = COUNT_KEYS[0].len()
    + COUNT_KEYS[1].len()
    + COUNT_KEYS[2].len()
    + COUNT_KEYS[3].len()
    + COUNT_KEYS[4].len()
    + COUNT_KEYS[5].len()
    + COUNT_KEYS.len() * U64_DIGITS;

/// The three mark items, in the order they are written.
#[cfg(feature = "verbose3")]
const MARK_ITEMS: [&[u8]; 3] = [b" pathdep=1", b" declrule=1", b" movelimit=1"];

/// The key of the list of positions before the root a `pathdep` value rests
/// on, written right after `pathdep=1`.
#[cfg(feature = "verbose3")]
const PATHDEP_PLIES: &[u8] = b" pathdep_plies=";

/// Written after that list when the value may rest on positions it does not
/// name.
#[cfg(feature = "verbose3")]
const PATHDEP_OPEN: &[u8] = b" pathdep_open=1";

/// The decimal width of the widest `u32`, the type a listed ply is held in.
#[cfg(feature = "verbose3")]
const U32_DIGITS: usize = 10;

/// The two items of a `pathdep` value's causes at their widest: a full list of
/// ten-digit plies, comma-separated, and the open item.
#[cfg(feature = "verbose3")]
const CAUSE_ITEMS_MAX: usize =
    PATHDEP_PLIES.len() + CAUSE_CAPACITY * (U32_DIGITS + 1) - 1 + PATHDEP_OPEN.len();

/// The duplication items' keys after the store-outcome, presence and event
/// ones, in the order they are written.
#[cfg(feature = "dupstats")]
const DUP_TAIL_KEYS: [&str; 7] = [
    "dup_nodes",
    "wt_replaced",
    "sp_searches",
    "sp_workers",
    "sp_behind",
    "sp_stopped",
    "sp_moves",
];

/// How many duplication items there are.
#[cfg(feature = "dupstats")]
const DUP_ITEMS: usize =
    STORE_KEYS.len() + PRESENCE_KEYS.len() + EVENT_KEYS.len() + DUP_TAIL_KEYS.len();

/// The total length of `keys`, each with the ` ` before it and the `=` after.
#[cfg(feature = "dupstats")]
const fn keys_len(keys: &[&str]) -> usize {
    let mut len = 0;
    let mut i = 0;
    while i < keys.len() {
        len += keys[i].len() + 2;
        i += 1;
    }
    len
}

/// The duplication items at their widest: every key with a 20-digit value.
#[cfg(feature = "dupstats")]
const DUP_ITEMS_MAX: usize = keys_len(&STORE_KEYS)
    + keys_len(&PRESENCE_KEYS)
    + keys_len(&EVENT_KEYS)
    + keys_len(&DUP_TAIL_KEYS)
    + DUP_ITEMS * U64_DIGITS;

/// The longest line [`render`] can produce without the duplication items: the
/// prefix, then every item at its widest.
#[cfg(not(feature = "verbose3"))]
const LINE_MAX: usize = PREFIX.len() + COUNT_ITEMS_MAX;

/// The longest line [`render`] can produce without the duplication items, with
/// the three mark items of a `verbose3` build after the counted ones. Each is
/// written only when its mark is set, so the widest line is the one that carries
/// all three, and the `pathdep` mark's causes at their widest.
#[cfg(feature = "verbose3")]
const LINE_MAX: usize = PREFIX.len()
    + COUNT_ITEMS_MAX
    + MARK_ITEMS[0].len()
    + MARK_ITEMS[1].len()
    + MARK_ITEMS[2].len()
    + CAUSE_ITEMS_MAX;

/// The move-order spread's items, in the order they are written, each with the
/// space that separates it from what comes before.
#[cfg(lazy_smp_spread)]
const SPREAD_KEYS: [&[u8]; 3] = [b" spread_dups=", b" spread_shifts=", b" spread_peak="];

/// The spread items at their widest: every key with a 20-digit value.
#[cfg(lazy_smp_spread)]
const SPREAD_ITEMS_MAX: usize =
    SPREAD_KEYS[0].len() + SPREAD_KEYS[1].len() + SPREAD_KEYS[2].len() + 3 * U64_DIGITS;

/// A build without the move-order spread writes none of its items.
#[cfg(not(lazy_smp_spread))]
const SPREAD_ITEMS_MAX: usize = 0;

/// The items of the workers' putting off of busy moves, in the order they are
/// written, each with the space that separates it from what comes before.
#[cfg(lazy_smp_defer)]
const DEFER_KEYS: [&[u8]; 3] = [b" defer_moves=", b" defer_tt_cuts=", b" defer_entries="];

/// The putting-off items at their widest: every key with a 20-digit value.
#[cfg(lazy_smp_defer)]
const DEFER_ITEMS_MAX: usize =
    DEFER_KEYS[0].len() + DEFER_KEYS[1].len() + DEFER_KEYS[2].len() + 3 * U64_DIGITS;

/// A build without the putting off of busy moves writes none of its items.
#[cfg(not(lazy_smp_defer))]
const DEFER_ITEMS_MAX: usize = 0;

/// The longest line [`render`] can produce.
#[cfg(not(feature = "dupstats"))]
const MAX_LINE: usize = LINE_MAX + SPREAD_ITEMS_MAX + DEFER_ITEMS_MAX;

/// The longest line [`render`] can produce, the duplication items included.
#[cfg(feature = "dupstats")]
const MAX_LINE: usize = LINE_MAX + SPREAD_ITEMS_MAX + DEFER_ITEMS_MAX + DUP_ITEMS_MAX;

/// What the searches of one interval did, summed over every thread that
/// searched in it.
///
/// - `nodes`: the search nodes — the count a search `info` line reports.
/// - `tt_update`: the transposition-table stores whose replacement condition
///   held, at any depth and into any entry. A store that only rewrites the
///   entry's move is not one.
/// - `eval_dup`: the stores of a freshly computed static eval after a probe
///   miss that found the same position's static eval already in the entry,
///   whether or not the store then replaced it.
/// - `evals`: the static evaluations computed for a node.
/// - `eval_tt_hits`: the nodes that took their static eval from the
///   transposition-table entry instead of evaluating.
///
/// Each worker counts into plain integers of its own while it searches; they
/// are added to the interval's tally once per worker, when its search ends, so
/// nothing on the node path is shared.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SearchCounts {
    pub(crate) nodes: u64,
    pub(crate) tt_update: u64,
    pub(crate) eval_dup: u64,
    pub(crate) evals: u64,
    pub(crate) eval_tt_hits: u64,
    /// How the workers repeated each other's work, summed over every worker.
    #[cfg(feature = "dupstats")]
    pub(crate) dup: DupCounts,
    /// How the workers of each search were spread when it ended.
    #[cfg(feature = "dupstats")]
    pub(crate) spread: Spread,
    /// What the workers' move-order spread counted: the duplicates and shifted
    /// picks summed over every worker, the peak the highest any worker reached.
    #[cfg(lazy_smp_spread)]
    pub(crate) smp: SpreadCounts,
    /// What the workers counted about the moves they put off, summed over
    /// every worker.
    #[cfg(lazy_smp_defer)]
    pub(crate) defer: DeferCounts,
}

/// How the workers of the searches of an interval were spread when each search
/// ended, summed over those searches. A search that thinks ahead past the reply
/// is not among them: its workers are collected one at a time.
#[cfg(feature = "dupstats")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Spread {
    /// The searches.
    pub(crate) searches: u64,
    /// Their workers.
    pub(crate) workers: u64,
    /// Over every worker, how many root iterations its completed depth was
    /// behind the deepest worker of its search.
    pub(crate) behind: u64,
    /// The workers by that distance: 0, 1, 2, 3 or more.
    pub(crate) behind_hist: [u64; 4],
    /// The workers that were inside an iteration when the search was stopped.
    pub(crate) stopped: u64,
    /// Over every search, the number of distinct root moves its stopped workers
    /// were searching.
    pub(crate) root_moves: u64,
}

#[cfg(feature = "dupstats")]
impl Spread {
    /// The spread of one search's `results`.
    pub(crate) fn of(results: &[WorkerResult]) -> Self {
        let deepest = results.iter().map(|r| r.completed_depth).max().unwrap_or(0);
        let mut spread = Self {
            searches: u64::from(!results.is_empty()),
            workers: results.len() as u64,
            ..Self::default()
        };
        for (i, r) in results.iter().enumerate() {
            let behind = (deepest - r.completed_depth).max(0) as u64;
            spread.behind += behind;
            spread.behind_hist[behind.min(3) as usize] += 1;
            if let Some(mv) = r.stopped_on {
                spread.stopped += 1;
                let seen = results[..i].iter().any(|o| o.stopped_on == Some(mv));
                spread.root_moves += u64::from(!seen);
            }
        }
        spread
    }

    fn add(&mut self, other: &Self) {
        self.searches += other.searches;
        self.workers += other.workers;
        self.behind += other.behind;
        for (mine, theirs) in self.behind_hist.iter_mut().zip(other.behind_hist) {
            *mine += theirs;
        }
        self.stopped += other.stopped;
        self.root_moves += other.root_moves;
    }

    /// The items the statistics line writes after the duplication counts, in
    /// [`DUP_TAIL_KEYS`] order from `sp_searches` on.
    fn items(&self) -> [u64; 5] {
        [
            self.searches,
            self.workers,
            self.behind,
            self.stopped,
            self.root_moves,
        ]
    }
}

/// The duplication counts and the spread of an interval, as they stand.
#[cfg(feature = "dupstats")]
struct DupTally {
    dup: DupCounts,
    spread: Spread,
}

/// [`SearchCounts`] as they stand, one sum per item.
struct Tally {
    nodes: AtomicU64,
    tt_updates: AtomicU64,
    eval_dups: AtomicU64,
    evals: AtomicU64,
    eval_tt_hits: AtomicU64,
    #[cfg(lazy_smp_spread)]
    spread_dups: AtomicU64,
    #[cfg(lazy_smp_spread)]
    spread_shifts: AtomicU64,
    /// The highest of the workers' peaks, not a sum.
    #[cfg(lazy_smp_spread)]
    spread_peak: AtomicU64,
    #[cfg(lazy_smp_defer)]
    defer_moves: AtomicU64,
    #[cfg(lazy_smp_defer)]
    defer_tt_cuts: AtomicU64,
    #[cfg(lazy_smp_defer)]
    defer_entries: AtomicU64,
    /// Too many counts for one atomic each, and added once per worker per
    /// search, so a lock costs nothing that matters.
    #[cfg(feature = "dupstats")]
    dup: Mutex<DupTally>,
}

impl Tally {
    const fn new() -> Self {
        Self {
            nodes: AtomicU64::new(0),
            tt_updates: AtomicU64::new(0),
            eval_dups: AtomicU64::new(0),
            evals: AtomicU64::new(0),
            eval_tt_hits: AtomicU64::new(0),
            #[cfg(lazy_smp_spread)]
            spread_dups: AtomicU64::new(0),
            #[cfg(lazy_smp_spread)]
            spread_shifts: AtomicU64::new(0),
            #[cfg(lazy_smp_spread)]
            spread_peak: AtomicU64::new(0),
            #[cfg(lazy_smp_defer)]
            defer_moves: AtomicU64::new(0),
            #[cfg(lazy_smp_defer)]
            defer_tt_cuts: AtomicU64::new(0),
            #[cfg(lazy_smp_defer)]
            defer_entries: AtomicU64::new(0),
            #[cfg(feature = "dupstats")]
            dup: Mutex::new(DupTally {
                dup: DupCounts::ZERO,
                spread: Spread {
                    searches: 0,
                    workers: 0,
                    behind: 0,
                    behind_hist: [0; 4],
                    stopped: 0,
                    root_moves: 0,
                },
            }),
        }
    }

    fn add(&self, r: &WorkerResult) {
        self.nodes.fetch_add(r.nodes, Ordering::Relaxed);
        self.tt_updates.fetch_add(r.tt_updates, Ordering::Relaxed);
        self.eval_dups.fetch_add(r.eval_dups, Ordering::Relaxed);
        self.evals.fetch_add(r.evals, Ordering::Relaxed);
        self.eval_tt_hits
            .fetch_add(r.eval_tt_hits, Ordering::Relaxed);
        #[cfg(lazy_smp_spread)]
        {
            self.spread_dups.fetch_add(r.spread.dups, Ordering::Relaxed);
            self.spread_shifts
                .fetch_add(r.spread.shifts, Ordering::Relaxed);
            self.spread_peak.fetch_max(r.spread.peak, Ordering::Relaxed);
        }
        #[cfg(lazy_smp_defer)]
        {
            self.defer_moves
                .fetch_add(r.defer.deferred, Ordering::Relaxed);
            self.defer_tt_cuts
                .fetch_add(r.defer.tt_cuts, Ordering::Relaxed);
            self.defer_entries
                .fetch_add(r.defer.entries, Ordering::Relaxed);
        }
        #[cfg(feature = "dupstats")]
        self.dup_tally().dup.add(&r.dup);
    }

    #[cfg(feature = "dupstats")]
    fn dup_tally(&self) -> std::sync::MutexGuard<'_, DupTally> {
        self.dup
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    #[cfg(feature = "dupstats")]
    fn add_spread(&self, spread: &Spread) {
        self.dup_tally().spread.add(spread);
    }

    fn take(&self) -> SearchCounts {
        #[cfg(feature = "dupstats")]
        let (dup, spread) = {
            let mut tally = self.dup_tally();
            let dup = std::mem::take(&mut tally.dup);
            (dup, std::mem::take(&mut tally.spread))
        };
        SearchCounts {
            nodes: self.nodes.swap(0, Ordering::Relaxed),
            tt_update: self.tt_updates.swap(0, Ordering::Relaxed),
            eval_dup: self.eval_dups.swap(0, Ordering::Relaxed),
            evals: self.evals.swap(0, Ordering::Relaxed),
            eval_tt_hits: self.eval_tt_hits.swap(0, Ordering::Relaxed),
            #[cfg(lazy_smp_spread)]
            smp: SpreadCounts {
                dups: self.spread_dups.swap(0, Ordering::Relaxed),
                shifts: self.spread_shifts.swap(0, Ordering::Relaxed),
                peak: self.spread_peak.swap(0, Ordering::Relaxed),
            },
            #[cfg(lazy_smp_defer)]
            defer: DeferCounts {
                deferred: self.defer_moves.swap(0, Ordering::Relaxed),
                tt_cuts: self.defer_tt_cuts.swap(0, Ordering::Relaxed),
                entries: self.defer_entries.swap(0, Ordering::Relaxed),
            },
            #[cfg(feature = "dupstats")]
            dup,
            #[cfg(feature = "dupstats")]
            spread,
        }
    }

    fn clear(&self) {
        self.take();
    }
}

/// The tally of the interval the next reply ends.
static INTERVAL: Tally = Tally::new();

/// The tally of the `bench` run in progress. Every reply takes the interval's
/// tally, and a run answers once per position, so the run's sum is kept apart.
#[cfg(feature = "verbose3")]
static BENCH_RUN: Tally = Tally::new();

/// The tally of the position a `bench` run is searching: the run's tally, taken
/// after each position.
#[cfg(all(feature = "dupstats", feature = "verbose3"))]
static BENCH_POSITION: Tally = Tally::new();

/// Add the counts of finished workers to the interval's tally.
pub(crate) fn add_search<'a>(results: impl IntoIterator<Item = &'a WorkerResult>) {
    for r in results {
        INTERVAL.add(r);
        #[cfg(feature = "verbose3")]
        BENCH_RUN.add(r);
        #[cfg(all(feature = "dupstats", feature = "verbose3"))]
        BENCH_POSITION.add(r);
    }
}

/// Add how the workers of one finished search were spread — `results` holds
/// every one of its workers — to the interval's tally.
#[cfg(feature = "dupstats")]
pub(crate) fn add_spread(results: &[WorkerResult]) {
    let spread = Spread::of(results);
    INTERVAL.add_spread(&spread);
    #[cfg(feature = "verbose3")]
    BENCH_RUN.add_spread(&spread);
    #[cfg(all(feature = "dupstats", feature = "verbose3"))]
    BENCH_POSITION.add_spread(&spread);
}

/// What a `bench` run keeps of one worker of one search: when it finished each
/// iteration, and where its nodes were.
#[cfg(all(feature = "dupstats", feature = "verbose3"))]
pub(crate) struct BenchWorker {
    pub(crate) timeline: Box<Timeline>,
    pub(crate) node_hist: Box<NodeHist>,
}

/// The timelines and node histograms of every worker of every search of the
/// `bench` run in progress, in the order the searches ended; `None` while no
/// run is.
#[cfg(all(feature = "dupstats", feature = "verbose3"))]
static BENCH_TIMELINES: Mutex<Option<Vec<BenchWorker>>> = Mutex::new(None);

/// Keep the timelines and node histograms of the workers of one finished
/// search, `results` holding every one of them, when a `bench` run is in
/// progress.
#[cfg(all(feature = "dupstats", feature = "verbose3"))]
pub(crate) fn add_timelines(results: &[WorkerResult]) {
    let mut held = BENCH_TIMELINES
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    if let Some(kept) = held.as_mut() {
        kept.extend(results.iter().map(|r| BenchWorker {
            timeline: r.timeline.clone(),
            node_hist: r.node_hist.clone(),
        }));
    }
}

/// Start keeping the timelines of a `bench` run's searches.
#[cfg(all(feature = "dupstats", feature = "verbose3"))]
pub(crate) fn start_bench_timelines() {
    *BENCH_TIMELINES
        .lock()
        .unwrap_or_else(PoisonError::into_inner) = Some(Vec::new());
}

/// The workers kept since the run started or since the last take.
#[cfg(all(feature = "dupstats", feature = "verbose3"))]
pub(crate) fn take_bench_timelines() -> Vec<BenchWorker> {
    BENCH_TIMELINES
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .as_mut()
        .map(std::mem::take)
        .unwrap_or_default()
}

/// Stop keeping timelines: the `bench` run is over.
#[cfg(all(feature = "dupstats", feature = "verbose3"))]
pub(crate) fn stop_bench_timelines() {
    *BENCH_TIMELINES
        .lock()
        .unwrap_or_else(PoisonError::into_inner) = None;
}

/// The widest `iter` line body: the lead, every key and a 20-digit value for
/// each, a move and a root-entry word, and the root-move split's moves and
/// values.
#[cfg(all(feature = "dupstats", feature = "verbose3"))]
const ITER_LINE: usize = 1024;

/// Hand `emit` the body of the `iter` line of iteration `it` of worker `worker`
/// in the search of position `position` (its index in the run's list): what
/// follows `info string `.
///
/// `iter pos=<p> worker=<w> depth=<d> done=<0|1> us=<µs> nodes=<n> fl=<n>
/// fh=<n> move=<usi> value=<v> changed=<0|1> first=<n> rest=<n>
/// root=<none|self|other|prev|unknown> split=<usi>:<n>,… other=<n>`
///
/// `split` is the iteration's [`yorkie_search::timeline::RootSplit`]: each of
/// the first root moves it searched with the nodes under it, in the order it
/// first searched them (`none` when it searched none), and `other` the nodes
/// under every root move past those.
#[cfg(all(feature = "dupstats", feature = "verbose3"))]
pub(crate) fn iteration_line<E>(
    position: u64,
    worker: u8,
    it: &Iteration,
    emit: impl FnOnce(&[u8]) -> Result<(), E>,
) -> Result<(), E> {
    let mut bytes = [0u8; ITER_LINE];
    let mut out = TextWriter::new(&mut bytes);
    out.bytes(b"iter pos=")
        .u64(position)
        .bytes(b" worker=")
        .u64(u64::from(worker))
        .bytes(b" depth=")
        .i64(i64::from(it.depth))
        .bytes(b" done=")
        .u64(u64::from(it.completed))
        .bytes(b" us=")
        .u64(it.micros)
        .bytes(b" nodes=")
        .u64(it.nodes)
        .bytes(b" fl=")
        .u64(u64::from(it.fail_lows))
        .bytes(b" fh=")
        .u64(u64::from(it.fail_highs))
        .bytes(b" move=");
    match it.best {
        Some(mv) => yorkie_state::write_usi_move(mv, &mut out),
        None => {
            out.bytes(b"none");
        }
    }
    out.bytes(b" value=")
        .i64(i64::from(it.value))
        .bytes(b" changed=")
        .u64(u64::from(it.best_changed))
        .bytes(b" first=")
        .u64(it.first_move_nodes)
        .bytes(b" rest=")
        .u64(it.other_move_nodes)
        .bytes(b" root=")
        .bytes(it.root_entry.key().as_bytes())
        .bytes(b" split=");
    let mut named = 0;
    for (mv, nodes) in it.split.named() {
        if named != 0 {
            out.byte(b',');
        }
        yorkie_state::write_usi_move(mv, &mut out);
        out.byte(b':').u64(nodes);
        named += 1;
    }
    if named == 0 {
        out.bytes(b"none");
    }
    out.bytes(b" other=").u64(it.split.rest);
    debug_assert!(!out.overflowed(), "the widest line fits ITER_LINE");
    emit(out.as_bytes())
}

/// Read the interval's tally and reset it to zero, so what is returned covers
/// exactly the interval since the previous take or clear.
pub(crate) fn take_search_counts() -> SearchCounts {
    INTERVAL.take()
}

/// Drop the interval's tally, so what came before is attributed to no reply.
pub(crate) fn clear_search_counts() {
    INTERVAL.clear();
}

/// Read the `bench` run's tally — every search since the previous
/// [`clear_bench_counts`] — and reset it to zero.
#[cfg(feature = "verbose3")]
pub(crate) fn take_bench_counts() -> SearchCounts {
    BENCH_RUN.take()
}

/// Start a `bench` run's tally from zero.
#[cfg(feature = "verbose3")]
pub(crate) fn clear_bench_counts() {
    BENCH_RUN.clear();
    #[cfg(feature = "dupstats")]
    BENCH_POSITION.clear();
}

/// Read the tally of the `bench` run's searches since the previous take — one
/// position's, when taken after each — and reset it to zero.
#[cfg(all(feature = "dupstats", feature = "verbose3"))]
pub(crate) fn take_position_counts() -> SearchCounts {
    BENCH_POSITION.take()
}

/// The items of a line [`render`] composed: what follows its prefix, starting
/// with the space before the first item.
#[cfg(all(feature = "dupstats", feature = "verbose3"))]
pub(crate) fn line_items(line: &[u8]) -> &[u8] {
    line.strip_prefix(PREFIX).unwrap_or(line)
}

/// The widest `nodehist` line body: the lead, three keys and their 20-digit
/// values, then every bucket of the three rows as a 20-digit value and a comma.
#[cfg(all(feature = "dupstats", feature = "verbose3"))]
const NODE_HIST_LINE: usize = 64
    + 4 * U64_DIGITS
    + (yorkie_search::dupstats::NODE_PLY_BUCKETS
        + yorkie_search::dupstats::NODE_DEPTH_BUCKETS
        + NODE_TYPE_NAMES.len())
        * (U64_DIGITS + 1);

/// Hand `emit` the body of the `nodehist` line of worker `worker` in the search
/// of position `position`: what follows `info string `.
///
/// `nodehist pos=<p> worker=<w> nodes=<n> ply=<n>,… depth=<n>,…
/// type=<pv>,<cut>,<all>,<qs>`
///
/// `ply` has one value per ply from 0, `depth` one per remaining depth from 0
/// (0 is the quiescence search); each list stops at its last non-zero bucket,
/// and the last bucket of [`NodeHist`] holds every deeper one.
#[cfg(all(feature = "dupstats", feature = "verbose3"))]
pub(crate) fn node_hist_line<E>(
    position: u64,
    worker: u8,
    hist: &NodeHist,
    emit: impl FnOnce(&[u8]) -> Result<(), E>,
) -> Result<(), E> {
    fn row(out: &mut TextWriter<'_>, name: &[u8], values: &[u64]) {
        let len = values.iter().rposition(|&n| n != 0).map_or(1, |i| i + 1);
        out.byte(b' ').bytes(name).byte(b'=');
        for (i, n) in values[..len].iter().enumerate() {
            if i != 0 {
                out.byte(b',');
            }
            out.u64(*n);
        }
    }
    let mut bytes = [0u8; NODE_HIST_LINE];
    let mut out = TextWriter::new(&mut bytes);
    out.bytes(b"nodehist pos=")
        .u64(position)
        .bytes(b" worker=")
        .u64(u64::from(worker))
        .bytes(b" nodes=")
        .u64(hist.total());
    row(&mut out, b"ply", &hist.ply);
    row(&mut out, b"depth", &hist.depth);
    out.bytes(b" type=");
    for (i, n) in hist.node_type.iter().enumerate() {
        if i != 0 {
            out.byte(b',');
        }
        out.u64(*n);
    }
    debug_assert!(!out.overflowed(), "the widest line fits NODE_HIST_LINE");
    emit(out.as_bytes())
}

/// The stack buffer a line is composed in.
pub(crate) struct StatsBuf {
    bytes: [u8; MAX_LINE],
}

impl StatsBuf {
    pub(crate) const fn new() -> Self {
        Self {
            bytes: [0; MAX_LINE],
        }
    }
}

/// Compose the statistics line for one reply into `buf`, or `None` when every
/// statistic is zero and so no line is written.
///
/// `alloc` and `search` are the counts of the interval the reply ends.
///
/// `marks` are the marks of the value the reply rests on — the score of the root
/// move the engine plays. Each is an item that appears only when it is set, like
/// every other item on the line. Right after `pathdep=1` come its causes:
/// `pathdep_plies=<n>,<n>,…`, the positions before the root its repetition
/// judgements matched, when there are any, and `pathdep_open=1` when there may
/// be others the list does not name.
///
/// [`MAX_LINE`] is the widest line the items can make, so nothing here can run
/// out of room — and a truncated line would be worse than no line, so the
/// impossible case is asserted rather than discarded.
pub(crate) fn render(
    buf: &mut StatsBuf,
    alloc: u64,
    search: SearchCounts,
    #[cfg(feature = "verbose3")] marks: impl Into<ReplyMarks>,
) -> Option<&[u8]> {
    let mut out = TextWriter::new(&mut buf.bytes);
    out.bytes(PREFIX);
    let mut carried = false;
    let counts = [
        alloc,
        search.nodes,
        search.tt_update,
        search.eval_dup,
        search.evals,
        search.eval_tt_hits,
    ];
    for (key, value) in COUNT_KEYS.iter().zip(counts) {
        if value != 0 {
            out.bytes(key).u64(value);
            carried = true;
        }
    }
    #[cfg(lazy_smp_spread)]
    for (key, value) in
        SPREAD_KEYS
            .iter()
            .zip([search.smp.dups, search.smp.shifts, search.smp.peak])
    {
        if value != 0 {
            out.bytes(key).u64(value);
            carried = true;
        }
    }
    #[cfg(lazy_smp_defer)]
    for (key, value) in DEFER_KEYS.iter().zip([
        search.defer.deferred,
        search.defer.tt_cuts,
        search.defer.entries,
    ]) {
        if value != 0 {
            out.bytes(key).u64(value);
            carried = true;
        }
    }
    #[cfg(feature = "dupstats")]
    for (key, value) in dup_items(&search) {
        if value != 0 {
            out.byte(b' ').bytes(key.as_bytes()).byte(b'=').u64(value);
            carried = true;
        }
    }
    #[cfg(feature = "verbose3")]
    {
        let ReplyMarks { marks, causes } = marks.into();
        for (item, set) in
            MARK_ITEMS
                .iter()
                .zip([marks.path_dep, marks.decl_rule, marks.move_limit])
        {
            if !set {
                continue;
            }
            out.bytes(item);
            carried = true;
            if *item == MARK_ITEMS[0] {
                if let Some((first, rest)) = causes.plies().split_first() {
                    out.bytes(PATHDEP_PLIES).u64(u64::from(*first));
                    for at in rest {
                        out.byte(b',').u64(u64::from(*at));
                    }
                }
                if causes.is_open() {
                    out.bytes(PATHDEP_OPEN);
                }
            }
        }
    }
    debug_assert!(!out.overflowed(), "the widest line fits MAX_LINE");
    let len = out.len();
    carried.then_some(&buf.bytes[..len])
}

/// The duplication items of `search`, `(key, value)` in the order the line
/// writes them: each store outcome's, presence outcome's and event's total,
/// then [`DUP_TAIL_KEYS`].
#[cfg(feature = "dupstats")]
fn dup_items(search: &SearchCounts) -> impl Iterator<Item = (&'static str, u64)> + '_ {
    let stores = StoreKind::ALL.map(|kind| search.dup.stores_of(kind));
    let presence = PresenceKind::ALL.map(|kind| search.dup.presence_of(kind));
    let events = Event::ALL.map(|event| search.dup.events_of(event));
    let tail = [search.dup.dup_nodes, search.dup.writer_replaced]
        .into_iter()
        .chain(search.spread.items());
    STORE_KEYS
        .into_iter()
        .zip(stores)
        .chain(PRESENCE_KEYS.into_iter().zip(presence))
        .chain(EVENT_KEYS.into_iter().zip(events))
        .chain(DUP_TAIL_KEYS.into_iter().zip(tail))
}

/// The widest histogram line: the `duphist ` lead and a key, then each
/// dimension's name and its buckets as 20-digit values.
#[cfg(all(feature = "dupstats", feature = "verbose3"))]
const HIST_LINE: usize = 1024;

/// Hand `emit` the body of each histogram line of `search`: first the bucket
/// labels of every dimension (`duphist buckets depth=1-2,3-5,… ply=… lost=…`),
/// then one line per store outcome, presence outcome and event, with that count
/// broken down over each dimension (`duphist <key> depth=<n>,<n>,… ply=…`) —
/// the three overwrites of a deeper entry also over the depth they lost
/// (`lost=<1>,<2>,<3>,<4+>`) — and last the workers' spread
/// (`duphist spread behind=<0>,<1>,<2>,<3+>`). A body is what follows
/// `info string `.
#[cfg(all(feature = "dupstats", feature = "verbose3"))]
pub(crate) fn histogram_lines<E>(
    search: &SearchCounts,
    mut emit: impl FnMut(&[u8]) -> Result<(), E>,
) -> Result<(), E> {
    use yorkie_search::dupstats::{BUCKET_NAMES, DIM_NAMES, LOST_BUCKET_NAMES, LOST_NAME};

    let mut bytes = [0u8; HIST_LINE];
    let mut out = TextWriter::new(&mut bytes);
    out.bytes(b"duphist buckets");
    let lost: &[&str] = &LOST_BUCKET_NAMES;
    for (name, labels) in DIM_NAMES
        .iter()
        .zip(BUCKET_NAMES)
        .chain([(&LOST_NAME, lost)])
    {
        out.byte(b' ').bytes(name.as_bytes()).byte(b'=');
        for (i, label) in labels.iter().enumerate() {
            if i != 0 {
                out.byte(b',');
            }
            out.bytes(label.as_bytes());
        }
    }
    debug_assert!(!out.overflowed(), "the widest line fits HIST_LINE");
    emit(out.as_bytes())?;

    let lost_rows = Event::ALL.map(|event| event.lost_row().map(|row| &search.dup.ow_lost[row]));
    let hists = STORE_KEYS
        .iter()
        .zip(&search.dup.stores)
        .map(|kh| (kh, None))
        .chain(
            PRESENCE_KEYS
                .iter()
                .zip(&search.dup.presence)
                .map(|kh| (kh, None)),
        )
        .chain(EVENT_KEYS.iter().zip(&search.dup.events).zip(lost_rows));
    for ((key, hist), lost) in hists {
        out.clear();
        out.bytes(b"duphist ").bytes(key.as_bytes());
        for ((name, labels), row) in DIM_NAMES.iter().zip(BUCKET_NAMES).zip(hist) {
            out.byte(b' ').bytes(name.as_bytes()).byte(b'=');
            for (i, n) in row[..labels.len()].iter().enumerate() {
                if i != 0 {
                    out.byte(b',');
                }
                out.u64(*n);
            }
        }
        if let Some(lost) = lost {
            out.byte(b' ').bytes(LOST_NAME.as_bytes()).byte(b'=');
            for (i, n) in lost.iter().enumerate() {
                if i != 0 {
                    out.byte(b',');
                }
                out.u64(*n);
            }
        }
        debug_assert!(!out.overflowed(), "the widest line fits HIST_LINE");
        emit(out.as_bytes())?;
    }

    out.clear();
    out.bytes(b"duphist spread behind=");
    for (i, n) in search.spread.behind_hist.iter().enumerate() {
        if i != 0 {
            out.byte(b',');
        }
        out.u64(*n);
    }
    emit(out.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reply whose value went through none of the marked rules — every build's
    /// ordinary case, and the only shape a build without `verbose3` has.
    fn unmarked(buf: &mut StatsBuf, alloc: u64) -> Option<&[u8]> {
        counted(buf, alloc, SearchCounts::default())
    }

    /// Like [`unmarked`], with search counts as well.
    fn counted(buf: &mut StatsBuf, alloc: u64, search: SearchCounts) -> Option<&[u8]> {
        render(
            buf,
            alloc,
            search,
            #[cfg(feature = "verbose3")]
            ValueMarks::NONE,
        )
    }

    fn search(nodes: u64, tt_update: u64, eval_dup: u64) -> SearchCounts {
        SearchCounts {
            nodes,
            tt_update,
            eval_dup,
            ..SearchCounts::default()
        }
    }

    fn evaluated(evals: u64, eval_tt_hits: u64) -> SearchCounts {
        SearchCounts {
            evals,
            eval_tt_hits,
            ..SearchCounts::default()
        }
    }

    fn all(value: u64) -> SearchCounts {
        SearchCounts {
            nodes: value,
            tt_update: value,
            eval_dup: value,
            evals: value,
            eval_tt_hits: value,
            #[cfg(lazy_smp_spread)]
            smp: SpreadCounts {
                dups: value,
                shifts: value,
                peak: value,
            },
            #[cfg(lazy_smp_defer)]
            defer: DeferCounts {
                deferred: value,
                tt_cuts: value,
                entries: value,
            },
            #[cfg(feature = "dupstats")]
            dup: DupCounts::ZERO,
            #[cfg(feature = "dupstats")]
            spread: Spread::default(),
        }
    }

    #[test]
    fn an_all_zero_reply_writes_no_line() {
        let mut buf = StatsBuf::new();
        assert_eq!(unmarked(&mut buf, 0), None);
    }

    #[test]
    fn a_counted_reply_is_the_prefix_and_the_item() {
        let mut buf = StatsBuf::new();
        assert_eq!(
            unmarked(&mut buf, 1),
            Some(&b"info string stats alloc=1"[..])
        );
        assert_eq!(
            unmarked(&mut buf, 40_321),
            Some(&b"info string stats alloc=40321"[..])
        );
    }

    #[test]
    fn the_counted_items_come_in_order_and_zero_ones_are_omitted() {
        let mut buf = StatsBuf::new();
        assert_eq!(
            counted(&mut buf, 3, search(1200, 450, 7)),
            Some(&b"info string stats alloc=3 nodes=1200 tt_update=450 eval_dup=7"[..])
        );
        assert_eq!(
            counted(&mut buf, 0, search(1200, 0, 7)),
            Some(&b"info string stats nodes=1200 eval_dup=7"[..])
        );
        assert_eq!(
            counted(&mut buf, 0, search(0, 0, 1)),
            Some(&b"info string stats eval_dup=1"[..])
        );
        assert_eq!(counted(&mut buf, 0, search(0, 0, 0)), None);
    }

    #[test]
    fn the_evaluation_items_follow_eval_dup_and_zero_ones_are_omitted() {
        let mut buf = StatsBuf::new();
        assert_eq!(
            counted(
                &mut buf,
                3,
                SearchCounts {
                    evals: 900,
                    eval_tt_hits: 250,
                    ..search(1200, 450, 7)
                }
            ),
            Some(
                &b"info string stats alloc=3 nodes=1200 tt_update=450 eval_dup=7 \
                   evals=900 eval_tt_hits=250"[..]
            )
        );
        assert_eq!(
            counted(&mut buf, 0, evaluated(900, 0)),
            Some(&b"info string stats evals=900"[..])
        );
        assert_eq!(
            counted(&mut buf, 0, evaluated(0, 250)),
            Some(&b"info string stats eval_tt_hits=250"[..])
        );
        assert_eq!(
            counted(
                &mut buf,
                0,
                SearchCounts {
                    nodes: 8,
                    ..evaluated(5, 2)
                }
            ),
            Some(&b"info string stats nodes=8 evals=5 eval_tt_hits=2"[..])
        );
        assert_eq!(counted(&mut buf, 0, evaluated(0, 0)), None);
    }

    #[cfg(not(any(
        feature = "verbose3",
        feature = "dupstats",
        lazy_smp_spread,
        lazy_smp_defer
    )))]
    #[test]
    fn the_widest_line_fits_the_buffer() {
        let mut buf = StatsBuf::new();
        let line =
            counted(&mut buf, u64::MAX, all(u64::MAX)).expect("a non-zero count is reported");
        assert_eq!(
            line,
            &b"info string stats alloc=18446744073709551615 nodes=18446744073709551615 \
               tt_update=18446744073709551615 eval_dup=18446744073709551615 \
               evals=18446744073709551615 eval_tt_hits=18446744073709551615"[..]
        );
        assert_eq!(line.len(), MAX_LINE);
    }

    #[test]
    fn a_buffer_is_reusable_and_carries_nothing_over() {
        let mut buf = StatsBuf::new();
        assert_eq!(
            unmarked(&mut buf, 123_456),
            Some(&b"info string stats alloc=123456"[..])
        );
        assert_eq!(
            unmarked(&mut buf, 7),
            Some(&b"info string stats alloc=7"[..])
        );
        assert_eq!(unmarked(&mut buf, 0), None);
    }

    /// The tally sums whatever is added to it and is emptied by a take or a
    /// clear, so nothing carries into the next interval.
    #[test]
    fn the_search_tally_is_emptied_by_a_take_and_by_a_clear() {
        let tally = Tally::new();
        assert_eq!(tally.take(), SearchCounts::default());
        tally.nodes.fetch_add(10, Ordering::Relaxed);
        tally.tt_updates.fetch_add(4, Ordering::Relaxed);
        tally.nodes.fetch_add(5, Ordering::Relaxed);
        tally.eval_dups.fetch_add(1, Ordering::Relaxed);
        tally.evals.fetch_add(9, Ordering::Relaxed);
        tally.eval_tt_hits.fetch_add(6, Ordering::Relaxed);
        tally.evals.fetch_add(2, Ordering::Relaxed);
        assert_eq!(
            tally.take(),
            SearchCounts {
                evals: 11,
                eval_tt_hits: 6,
                ..search(15, 4, 1)
            }
        );
        assert_eq!(tally.take(), SearchCounts::default());
        tally.nodes.fetch_add(3, Ordering::Relaxed);
        tally.eval_tt_hits.fetch_add(3, Ordering::Relaxed);
        tally.clear();
        assert_eq!(tally.take(), SearchCounts::default());
    }

    /// The three items a `verbose3` build adds: the marks of the value the reply
    /// rests on.
    #[cfg(feature = "verbose3")]
    mod value_marks {
        use super::*;
        use yorkie_search::causes::PathCauses;

        fn marks(path_dep: bool, decl_rule: bool, move_limit: bool) -> ValueMarks {
            ValueMarks {
                path_dep,
                decl_rule,
                move_limit,
            }
        }

        /// Every combination of the three, each written only when it is set and
        /// always in the same order.
        #[test]
        fn each_combination_writes_the_items_that_are_set() {
            let expected: [&[u8]; 8] = [
                b"info string stats alloc=9",
                b"info string stats alloc=9 pathdep=1",
                b"info string stats alloc=9 declrule=1",
                b"info string stats alloc=9 pathdep=1 declrule=1",
                b"info string stats alloc=9 movelimit=1",
                b"info string stats alloc=9 pathdep=1 movelimit=1",
                b"info string stats alloc=9 declrule=1 movelimit=1",
                b"info string stats alloc=9 pathdep=1 declrule=1 movelimit=1",
            ];
            let mut buf = StatsBuf::new();
            for (bits, want) in expected.iter().enumerate() {
                let m = marks(bits & 1 != 0, bits & 2 != 0, bits & 4 != 0);
                assert_eq!(
                    render(&mut buf, 9, SearchCounts::default(), m),
                    Some(*want),
                    "marks {m:?}"
                );
            }
        }

        /// A mark is an item like any other, so it alone is enough to make a line
        /// a reply with nothing else to report would not have written.
        #[test]
        fn a_mark_alone_writes_the_line() {
            let mut buf = StatsBuf::new();
            assert_eq!(
                render(
                    &mut buf,
                    0,
                    SearchCounts::default(),
                    marks(false, true, false)
                ),
                Some(&b"info string stats declrule=1"[..])
            );
            assert_eq!(
                render(&mut buf, 0, SearchCounts::default(), ValueMarks::NONE),
                None
            );
        }

        /// The marks follow the counted items.
        #[test]
        fn the_marks_come_after_the_counted_items() {
            let mut buf = StatsBuf::new();
            assert_eq!(
                render(&mut buf, 0, search(8, 0, 2), marks(true, false, false)),
                Some(&b"info string stats nodes=8 eval_dup=2 pathdep=1"[..])
            );
        }

        #[cfg(not(any(feature = "dupstats", lazy_smp_spread, lazy_smp_defer)))]
        #[test]
        fn the_widest_line_fits_the_buffer() {
            let mut buf = StatsBuf::new();
            let widest = ReplyMarks {
                marks: marks(true, true, true),
                causes: causes(
                    &(0..CAUSE_CAPACITY)
                        .map(|i| u32::MAX as usize - i)
                        .collect::<Vec<_>>(),
                    true,
                ),
            };
            let line = render(&mut buf, u64::MAX, all(u64::MAX), widest)
                .expect("a marked reply is reported");
            assert_eq!(
                line,
                &b"info string stats alloc=18446744073709551615 nodes=18446744073709551615 \
                   tt_update=18446744073709551615 eval_dup=18446744073709551615 \
                   evals=18446744073709551615 eval_tt_hits=18446744073709551615 \
                   pathdep=1 pathdep_plies=4294967288,4294967289,4294967290,4294967291,\
                   4294967292,4294967293,4294967294,4294967295 pathdep_open=1 \
                   declrule=1 movelimit=1"[..]
            );
            assert_eq!(line.len(), MAX_LINE);
        }

        /// A set listing `plies`, open when `open` is.
        fn causes(plies: &[usize], open: bool) -> PathCauses {
            let mut out = PathCauses::NONE;
            for &at in plies {
                out.insert(at);
            }
            if open {
                out.set_open();
            }
            out
        }

        /// The causes of a `pathdep` value follow `pathdep=1` directly: the list
        /// ascending and comma-separated, then the open item; an empty list is
        /// left off, and a set that is not open writes no open item.
        #[test]
        fn the_causes_follow_the_pathdep_mark() {
            let mut buf = StatsBuf::new();
            let cases: [(&[usize], bool, &[u8]); 4] = [
                (&[], false, b"info string stats pathdep=1 declrule=1"),
                (
                    &[12, 4],
                    false,
                    b"info string stats pathdep=1 pathdep_plies=4,12 declrule=1",
                ),
                (
                    &[],
                    true,
                    b"info string stats pathdep=1 pathdep_open=1 declrule=1",
                ),
                (
                    &[7],
                    true,
                    b"info string stats pathdep=1 pathdep_plies=7 pathdep_open=1 declrule=1",
                ),
            ];
            for (plies, open, want) in cases {
                let report = ReplyMarks {
                    marks: marks(true, true, false),
                    causes: causes(plies, open),
                };
                assert_eq!(
                    render(&mut buf, 0, SearchCounts::default(), report),
                    Some(want),
                    "causes {plies:?}, open {open}"
                );
            }
        }
    }

    /// The items a build with the move-order spread adds.
    #[cfg(lazy_smp_spread)]
    mod move_order_spread {
        use super::*;

        fn spread(dups: u64, shifts: u64, peak: u64) -> SearchCounts {
            SearchCounts {
                smp: SpreadCounts { dups, shifts, peak },
                ..search(100, 0, 0)
            }
        }

        /// The three items follow the counted ones, and a zero one is omitted.
        #[test]
        fn the_items_follow_the_counted_ones() {
            let mut buf = StatsBuf::new();
            assert_eq!(
                counted(&mut buf, 0, spread(12, 30, 2)),
                Some(
                    &b"info string stats nodes=100 spread_dups=12 spread_shifts=30 spread_peak=2"[..]
                )
            );
            assert_eq!(
                counted(&mut buf, 0, spread(0, 0, 1)),
                Some(&b"info string stats nodes=100 spread_peak=1"[..])
            );
        }

        /// The tally sums the duplicates and shifts and keeps the highest peak.
        #[test]
        fn the_tally_sums_the_counts_and_keeps_the_highest_peak() {
            let tally = Tally::new();
            for (dups, shifts, peak) in [(3, 5, 2), (4, 1, 1)] {
                tally.spread_dups.fetch_add(dups, Ordering::Relaxed);
                tally.spread_shifts.fetch_add(shifts, Ordering::Relaxed);
                tally.spread_peak.fetch_max(peak, Ordering::Relaxed);
            }
            assert_eq!(tally.take().smp, spread(7, 6, 2).smp);
            assert_eq!(tally.take().smp, SpreadCounts::default());
        }

        #[cfg(not(any(feature = "verbose3", feature = "dupstats")))]
        #[test]
        fn the_widest_line_fits_the_buffer() {
            let mut buf = StatsBuf::new();
            let line =
                counted(&mut buf, u64::MAX, all(u64::MAX)).expect("a non-zero count is reported");
            assert_eq!(line.len(), MAX_LINE);
        }
    }

    /// The items a build with the putting off of busy moves adds.
    #[cfg(lazy_smp_defer)]
    mod busy_move_defer {
        use super::*;

        fn defer(deferred: u64, tt_cuts: u64, entries: u64) -> SearchCounts {
            SearchCounts {
                defer: DeferCounts {
                    deferred,
                    tt_cuts,
                    entries,
                },
                ..search(100, 0, 0)
            }
        }

        /// The three items follow the counted ones, and a zero one is omitted.
        #[test]
        fn the_items_follow_the_counted_ones() {
            let mut buf = StatsBuf::new();
            assert_eq!(
                counted(&mut buf, 0, defer(12, 3, 40)),
                Some(
                    &b"info string stats nodes=100 defer_moves=12 defer_tt_cuts=3 defer_entries=40"
                        [..]
                )
            );
            assert_eq!(
                counted(&mut buf, 0, defer(0, 0, 7)),
                Some(&b"info string stats nodes=100 defer_entries=7"[..])
            );
        }

        /// The tally sums every worker's counts.
        #[test]
        fn the_tally_sums_the_counts() {
            let tally = Tally::new();
            for (deferred, tt_cuts, entries) in [(3, 1, 9), (4, 0, 2)] {
                tally.defer_moves.fetch_add(deferred, Ordering::Relaxed);
                tally.defer_tt_cuts.fetch_add(tt_cuts, Ordering::Relaxed);
                tally.defer_entries.fetch_add(entries, Ordering::Relaxed);
            }
            assert_eq!(tally.take().defer, defer(7, 1, 11).defer);
            assert_eq!(tally.take().defer, DeferCounts::default());
        }

        #[cfg(not(any(feature = "verbose3", feature = "dupstats")))]
        #[test]
        fn the_widest_line_fits_the_buffer() {
            let mut buf = StatsBuf::new();
            let line =
                counted(&mut buf, u64::MAX, all(u64::MAX)).expect("a non-zero count is reported");
            assert_eq!(line.len(), MAX_LINE);
        }
    }

    /// The items a `dupstats` build adds: how the workers repeated each other's
    /// work, and how they were spread when each search ended.
    #[cfg(feature = "dupstats")]
    mod duplication {
        use super::*;
        use yorkie_search::dupstats::{Where, Writer};

        fn render_plain(buf: &mut StatsBuf, search: SearchCounts) -> Option<&[u8]> {
            render(
                buf,
                0,
                search,
                #[cfg(feature = "verbose3")]
                ValueMarks::NONE,
            )
        }

        #[test]
        fn the_items_follow_the_counted_ones_and_zero_ones_are_omitted() {
            let mut dup = DupCounts::ZERO;
            let at = Where::new(4, 2, false, true, 1);
            dup.count_store(StoreKind::New, at);
            dup.count_store(StoreKind::New, at);
            dup.count_store(StoreKind::AlreadyThere(Writer::OtherWorker), at);
            dup.count_presence(PresenceKind::Entered, at);
            dup.count(Event::TtMiss, at);
            dup.count(Event::CutFirst, at);
            dup.dup_nodes = 40;
            let search = SearchCounts {
                dup,
                spread: Spread {
                    searches: 1,
                    workers: 2,
                    behind: 1,
                    behind_hist: [1, 1, 0, 0],
                    stopped: 1,
                    root_moves: 1,
                },
                ..search(100, 0, 0)
            };
            let mut buf = StatsBuf::new();
            assert_eq!(
                render_plain(&mut buf, search),
                Some(
                    &b"info string stats nodes=100 st_new=2 st_there_other=1 pr_enter=1 \
                       tt_miss=1 cut_first=1 dup_nodes=40 sp_searches=1 sp_workers=2 sp_behind=1 sp_stopped=1 \
                       sp_moves=1"[..]
                )
            );
        }

        #[test]
        fn the_widest_line_fits_the_buffer() {
            let mut dup = DupCounts::ZERO;
            for hist in dup
                .stores
                .iter_mut()
                .chain(dup.presence.iter_mut())
                .chain(dup.events.iter_mut())
            {
                hist[0][0] = u64::MAX;
            }
            dup.dup_nodes = u64::MAX;
            dup.writer_replaced = u64::MAX;
            let spread = Spread {
                searches: u64::MAX,
                workers: u64::MAX,
                behind: u64::MAX,
                behind_hist: [0; 4],
                stopped: u64::MAX,
                root_moves: u64::MAX,
            };
            let search = SearchCounts {
                dup,
                spread,
                ..all(u64::MAX)
            };
            let mut buf = StatsBuf::new();
            #[cfg(feature = "verbose3")]
            let marks = {
                let mut causes = yorkie_search::causes::PathCauses::NONE;
                for i in 0..CAUSE_CAPACITY {
                    causes.insert(u32::MAX as usize - i);
                }
                causes.set_open();
                ReplyMarks {
                    marks: ValueMarks {
                        path_dep: true,
                        decl_rule: true,
                        move_limit: true,
                    },
                    causes,
                }
            };
            let line = render(
                &mut buf,
                u64::MAX,
                search,
                #[cfg(feature = "verbose3")]
                marks,
            )
            .expect("a non-zero count is reported");
            assert_eq!(line.len(), MAX_LINE);
        }

        /// The spread of one search: how far behind the deepest worker each one
        /// stopped, and how many distinct root moves the stopped ones were on.
        #[test]
        fn the_spread_of_a_search_counts_depths_behind_and_distinct_moves() {
            use yorkie_search::{RootMove, generate_root_moves};
            let pos = yorkie_state::Position::startpos();
            let moves: Vec<RootMove> = generate_root_moves(&pos);
            let worker = |depth: i32, on: Option<usize>| WorkerResult {
                best: moves[0].clone(),
                completed_depth: depth,
                nodes: 0,
                tt_updates: 0,
                eval_dups: 0,
                evals: 0,
                eval_tt_hits: 0,
                dup: Box::new(DupCounts::ZERO),
                stopped_on: on.map(|i| moves[i].mv),
                timeline: Box::default(),
                node_hist: Box::default(),
                #[cfg(lazy_smp_spread)]
                spread: SpreadCounts::default(),
                #[cfg(lazy_smp_defer)]
                defer: DeferCounts::default(),
                #[cfg(feature = "verbose2")]
                uci_pv_sent: false,
                #[cfg(feature = "verbose2")]
                pv_lines: Vec::new(),
                time_reduction: 1.0,
            };
            let results = [
                worker(12, None),
                worker(11, Some(0)),
                worker(12, Some(0)),
                worker(9, Some(3)),
            ];
            assert_eq!(
                Spread::of(&results),
                Spread {
                    searches: 1,
                    workers: 4,
                    behind: 4,
                    behind_hist: [2, 1, 0, 1],
                    stopped: 3,
                    root_moves: 2,
                }
            );
            assert_eq!(Spread::of(&[]), Spread::default());
        }

        #[test]
        fn the_histogram_lines_carry_every_outcome_and_the_spread() {
            #[cfg(feature = "verbose3")]
            {
                use yorkie_search::dupstats::Overwrite;
                let mut dup = DupCounts::ZERO;
                dup.count_store(StoreKind::New, Where::new(4, 2, false, true, 1));
                dup.count_overwrite(
                    Overwrite::Shallower {
                        by: Event::OwShallowOther,
                        lost: 2,
                    },
                    Where::new(7, 9, true, false, 0),
                );
                dup.count(Event::TtCut, Where::new(3, 1, false, false, 5));
                let search = SearchCounts {
                    dup,
                    ..SearchCounts::default()
                };
                let mut lines: Vec<String> = Vec::new();
                histogram_lines::<()>(&search, |l| {
                    lines.push(String::from_utf8(l.to_vec()).expect("text"));
                    Ok(())
                })
                .expect("collected");
                assert_eq!(
                    lines.len(),
                    1 + STORE_KEYS.len() + PRESENCE_KEYS.len() + EVENT_KEYS.len() + 1
                );
                assert_eq!(
                    lines[0],
                    "duphist buckets depth=1-2,3-5,6-9,10+ ply=0-3,4-7,8-11,12-19,20+ \
                     type=pv,cut,all move=none,1,2-4,5+ lost=1,2,3,4+"
                );
                assert_eq!(
                    lines[1],
                    "duphist st_new depth=0,1,0,0 ply=1,0,0,0,0 type=0,1,0 move=0,1,0,0"
                );
                let line = |key: &str| {
                    let lead = format!("duphist {key} ");
                    lines
                        .iter()
                        .find(|l| l.starts_with(&lead))
                        .cloned()
                        .unwrap_or_default()
                };
                assert_eq!(
                    line("ow_shallow_other"),
                    "duphist ow_shallow_other depth=0,0,1,0 ply=0,0,1,0,0 type=1,0,0 \
                     move=1,0,0,0 lost=0,1,0,0"
                );
                assert_eq!(
                    line("ow_shallow_self"),
                    "duphist ow_shallow_self depth=0,0,0,0 ply=0,0,0,0,0 type=0,0,0 \
                     move=0,0,0,0 lost=0,0,0,0"
                );
                assert_eq!(
                    line("tt_cut"),
                    "duphist tt_cut depth=0,1,0,0 ply=1,0,0,0,0 type=0,0,1 move=0,0,0,1"
                );
                assert_eq!(lines[lines.len() - 1], "duphist spread behind=0,0,0,0");
            }
        }
    }
}
