//! How parallel search workers repeat each other's work: the counts a
//! `dupstats` build keeps beside the search, and the two statistics-only tables
//! they are read from.
//!
//! Every worker of a parallel search shares one transposition table and nothing
//! else, so two workers can search the same subtree, one after the other or both
//! at once. Three kinds of count say where and how much:
//!
//! - **Store outcomes.** At each store that ends a node, what the table entry
//!   held before it: another position or nothing, or only what the node itself
//!   put there after its probe missed ([`StoreKind::New`]); this position with
//!   no searched depth — only a static eval or a quiescence result — that was
//!   there when the node probed ([`StoreKind::Unsearched`]); this position at a
//!   shallower depth
//!   ([`StoreKind::Deepened`]); this position at an equal or greater depth
//!   ([`StoreKind::AlreadyThere`]). The last two are split by who wrote the
//!   entry ([`Writer`]). The writer is not in the table entry; it is kept in a
//!   side table of its own, indexed by the position key.
//! - **Concurrent presence.** A second side table holds the positions some
//!   worker is searching right now, at a remaining depth of at least
//!   [`PRESENCE_MIN_DEPTH`]. A node entry that finds another worker already
//!   inside the same position is counted, by how that worker's remaining depth
//!   compares with its own ([`PresenceKind`]).
//! - **Work under a duplicated subtree.** When a store finds the entry already
//!   there and written by another worker, the node count of the subtree under the
//!   storing node is added to [`DupCounts::dup_nodes`] — once, for the outermost
//!   such node on the path ([`credit_subtree`]).
//!
//! Each store-outcome and presence count is kept as a small histogram over four
//! dimensions of the node it was taken at ([`Where`]).
//!
//! Nothing here is read by the search: its decisions, node counts and best
//! moves are those of a build without the feature. The side tables do cost
//! memory traffic, so the build is slower, and its timings are not comparable
//! with a build without it.

use std::cell::RefCell;
use std::cmp::Ordering as Cmp;
use std::sync::atomic::{AtomicU64, Ordering};

/// The dimensions a count is broken down by: the node's remaining depth on
/// entry, its ply from the root, its node type, and the index of the move that
/// reached it in its parent's move loop.
pub const DIMS: usize = 4;
/// The most buckets any one dimension has.
pub const BUCKETS: usize = 5;
/// Each dimension's name, as the histogram output writes it.
pub const DIM_NAMES: [&str; DIMS] = ["depth", "ply", "type", "move"];
/// Each dimension's bucket labels. The move dimension's `none` is a node no move
/// loop reached: the root, and a node of the null-move or ProbCut search.
pub const BUCKET_NAMES: [&[&str]; DIMS] = [
    &["1-2", "3-5", "6-9", "10+"],
    &["0-3", "4-7", "8-11", "12-19", "20+"],
    &["pv", "cut", "all"],
    &["none", "1", "2-4", "5+"],
];

/// One count broken down over each dimension separately: `[dimension][bucket]`.
/// Every dimension's row sums to the same total.
pub type Hist = [[u64; BUCKETS]; DIMS];

/// The histogram buckets one node falls in, one per dimension.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Where([u8; DIMS]);

impl Where {
    /// The buckets of a node of remaining depth `depth` at `ply`, of the node
    /// type `pv_node` / `cut_node` give, reached by the `move_index`-th move of
    /// its parent's loop (`0` when no loop reached it).
    pub fn new(depth: i32, ply: i32, pv_node: bool, cut_node: bool, move_index: i32) -> Self {
        let depth = match depth {
            ..=2 => 0,
            3..=5 => 1,
            6..=9 => 2,
            _ => 3,
        };
        let ply = match ply {
            ..=3 => 0,
            4..=7 => 1,
            8..=11 => 2,
            12..=19 => 3,
            _ => 4,
        };
        let node_type = if pv_node {
            0
        } else if cut_node {
            1
        } else {
            2
        };
        let move_index = match move_index {
            ..=0 => 0,
            1 => 1,
            2..=4 => 2,
            _ => 3,
        };
        Self([depth, ply, node_type, move_index])
    }

    fn record(self, hist: &mut Hist) {
        for (row, bucket) in hist.iter_mut().zip(self.0) {
            row[bucket as usize] += 1;
        }
    }
}

/// The total a histogram counts.
pub fn total(hist: &Hist) -> u64 {
    hist[0].iter().sum()
}

/// Who wrote the table entry a store found its own position in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Writer {
    /// This worker, in the root iteration it is in now — an aspiration or PVS
    /// re-search, or a transposition within the iteration.
    ThisIteration,
    /// This worker, in an earlier root iteration of the same search.
    EarlierIteration,
    /// Another worker of the same search.
    OtherWorker,
    /// Any worker of an earlier search.
    EarlierSearch,
    /// Not known: the side table lost the record to another position.
    Unknown,
}

impl Writer {
    const ALL: [Writer; 5] = [
        Writer::ThisIteration,
        Writer::EarlierIteration,
        Writer::OtherWorker,
        Writer::EarlierSearch,
        Writer::Unknown,
    ];

    const fn index(self) -> usize {
        match self {
            Writer::ThisIteration => 0,
            Writer::EarlierIteration => 1,
            Writer::OtherWorker => 2,
            Writer::EarlierSearch => 3,
            Writer::Unknown => 4,
        }
    }
}

/// What a store that ends a node found in the table entry before it wrote.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoreKind {
    /// Another position, or nothing, or only the static eval the node itself
    /// stored after its probe missed.
    New,
    /// This position with no searched depth — only a static eval, or a
    /// quiescence result — already there when the node probed.
    Unsearched,
    /// This position at a shallower searched depth than the store's.
    Deepened(Writer),
    /// This position at a depth equal to the store's or greater.
    AlreadyThere(Writer),
}

/// Each [`StoreKind`]'s item key on the statistics line, in
/// [`StoreKind::index`] order.
pub const STORE_KEYS: [&str; StoreKind::COUNT] = [
    "st_new",
    "st_unsearched",
    "st_deep_iter",
    "st_deep_earlier",
    "st_deep_other",
    "st_deep_prev",
    "st_deep_unknown",
    "st_there_iter",
    "st_there_earlier",
    "st_there_other",
    "st_there_prev",
    "st_there_unknown",
];

/// Each [`PresenceKind`]'s item key on the statistics line, in
/// [`PresenceKind::index`] order.
pub const PRESENCE_KEYS: [&str; PresenceKind::COUNT] = [
    "pr_enter",
    "pr_deeper",
    "pr_equal",
    "pr_shallower",
    "pr_busy",
];

impl StoreKind {
    /// How many kinds there are.
    pub const COUNT: usize = 12;

    /// Every kind, in [`Self::index`] order.
    pub const ALL: [StoreKind; Self::COUNT] = {
        let mut all = [StoreKind::New; Self::COUNT];
        all[1] = StoreKind::Unsearched;
        let mut i = 0;
        while i < Writer::ALL.len() {
            all[2 + i] = StoreKind::Deepened(Writer::ALL[i]);
            all[2 + Writer::ALL.len() + i] = StoreKind::AlreadyThere(Writer::ALL[i]);
            i += 1;
        }
        all
    };

    /// The kind's position in [`DupCounts::stores`].
    pub const fn index(self) -> usize {
        match self {
            StoreKind::New => 0,
            StoreKind::Unsearched => 1,
            StoreKind::Deepened(w) => 2 + w.index(),
            StoreKind::AlreadyThere(w) => 2 + Writer::ALL.len() + w.index(),
        }
    }
}

/// What a node entry found in the presence table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresenceKind {
    /// Every node entry the table was consulted for — a node of remaining
    /// depth [`PRESENCE_MIN_DEPTH`] or more that went on past the
    /// transposition-table cutoff to search: the denominator.
    Entered,
    /// Another worker was inside the position at a greater remaining depth.
    Deeper,
    /// Another worker was inside the position at the same remaining depth.
    Equal,
    /// Another worker was inside the position at a smaller remaining depth.
    Shallower,
    /// The table slot was held by another position, so nothing could be told.
    Busy,
}

impl PresenceKind {
    /// How many kinds there are.
    pub const COUNT: usize = 5;

    /// Every kind, in [`Self::index`] order.
    pub const ALL: [PresenceKind; Self::COUNT] = [
        PresenceKind::Entered,
        PresenceKind::Deeper,
        PresenceKind::Equal,
        PresenceKind::Shallower,
        PresenceKind::Busy,
    ];

    /// The kind's position in [`DupCounts::presence`].
    pub const fn index(self) -> usize {
        match self {
            PresenceKind::Entered => 0,
            PresenceKind::Deeper => 1,
            PresenceKind::Equal => 2,
            PresenceKind::Shallower => 3,
            PresenceKind::Busy => 4,
        }
    }
}

/// One worker's counts over one search, or a sum of several.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DupCounts {
    /// One histogram per [`StoreKind`], at [`StoreKind::index`].
    pub stores: [Hist; StoreKind::COUNT],
    /// One histogram per [`PresenceKind`], at [`PresenceKind::index`].
    pub presence: [Hist; PresenceKind::COUNT],
    /// The nodes searched under a subtree whose root store found the entry
    /// already there, written by another worker — each node once.
    pub dup_nodes: u64,
    /// Writer records that replaced another position's record in the side table:
    /// how often the side table itself collided.
    pub writer_replaced: u64,
}

impl DupCounts {
    /// Nothing counted.
    pub const ZERO: Self = Self {
        stores: [[[0; BUCKETS]; DIMS]; StoreKind::COUNT],
        presence: [[[0; BUCKETS]; DIMS]; PresenceKind::COUNT],
        dup_nodes: 0,
        writer_replaced: 0,
    };

    /// Count one store of `kind` at a node in `at`.
    pub fn count_store(&mut self, kind: StoreKind, at: Where) {
        at.record(&mut self.stores[kind.index()]);
    }

    /// Count one presence outcome of `kind` at a node in `at`.
    pub fn count_presence(&mut self, kind: PresenceKind, at: Where) {
        at.record(&mut self.presence[kind.index()]);
    }

    /// Add `other` into these counts.
    pub fn add(&mut self, other: &Self) {
        let pairs = self
            .stores
            .iter_mut()
            .zip(&other.stores)
            .chain(self.presence.iter_mut().zip(&other.presence));
        for (mine, theirs) in pairs {
            for (row, other_row) in mine.iter_mut().zip(theirs) {
                for (cell, n) in row.iter_mut().zip(other_row) {
                    *cell += n;
                }
            }
        }
        self.dup_nodes += other.dup_nodes;
        self.writer_replaced += other.writer_replaced;
    }

    /// The total of the store outcome `kind`.
    pub fn stores_of(&self, kind: StoreKind) -> u64 {
        total(&self.stores[kind.index()])
    }

    /// The total of the presence outcome `kind`.
    pub fn presence_of(&self, kind: PresenceKind) -> u64 {
        total(&self.presence[kind.index()])
    }
}

impl Default for DupCounts {
    fn default() -> Self {
        Self::ZERO
    }
}

/// Who a store is made by: which search, which worker, which root iteration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WriterId {
    /// The transposition table's generation, which every worker of one search
    /// shares and the next search moves on.
    pub generation: u8,
    /// The worker's index in its pool.
    pub worker: u8,
    /// The root depth of the iteration the worker is in.
    pub iteration: u8,
}

/// Classify a store of depth `store_depth` that found its own position in the
/// entry at depth `found` (`None`: another position or nothing), last written
/// by `previous` (`None`: not known), made by `me`. `probe_hit` is whether the
/// node's own probe found the position when it was entered.
///
/// A node whose probe missed writes its static eval into the entry before it
/// searches, at no searched depth; the store that ends it then finds that, which
/// is the node's own doing and so as new as a miss. Only an entry with no
/// searched depth that was there before the node is [`StoreKind::Unsearched`].
pub fn classify(
    found: Option<i32>,
    store_depth: i32,
    probe_hit: bool,
    previous: Option<WriterId>,
    me: WriterId,
) -> StoreKind {
    let Some(depth) = found else {
        return StoreKind::New;
    };
    if depth < 1 {
        return if probe_hit {
            StoreKind::Unsearched
        } else {
            StoreKind::New
        };
    }
    let writer = match previous {
        None => Writer::Unknown,
        Some(w) if w.generation != me.generation => Writer::EarlierSearch,
        Some(w) if w.worker != me.worker => Writer::OtherWorker,
        Some(w) if w.iteration == me.iteration => Writer::ThisIteration,
        Some(_) => Writer::EarlierIteration,
    };
    if depth >= store_depth {
        StoreKind::AlreadyThere(writer)
    } else {
        StoreKind::Deepened(writer)
    }
}

/// The part of a position key the side tables keep to tell positions apart: the
/// low 32 bits. Their slot is chosen by the high bits, so the two are
/// independent.
const fn tag(key: u64) -> u64 {
    key & 0xffff_ffff
}

/// A record's flag bit: set in every record, so an empty slot (zero) is never
/// mistaken for one.
const LIVE: u64 = 1 << 63;

/// `log2` of the writer table's slot count. 2^23 slots of 8 bytes are 64 MiB,
/// several times the number of positions a search of a few million nodes stores
/// at a searched depth, so a record is rarely lost to another position before it
/// is read; [`DupCounts::writer_replaced`] and [`Writer::Unknown`] say how often
/// it was.
const WRITER_BITS: u32 = 23;

/// The writer table: one record per slot — the key tag, then
/// [`WriterId::generation`], [`WriterId::worker`] and [`WriterId::iteration`] in
/// one byte each — written after each store that wrote its entry.
static WRITERS: [AtomicU64; 1 << WRITER_BITS] = [const { AtomicU64::new(0) }; 1 << WRITER_BITS];

const fn writer_slot(key: u64) -> usize {
    (key >> (64 - WRITER_BITS)) as usize
}

const fn writer_record(key: u64, id: WriterId) -> u64 {
    tag(key)
        | (id.generation as u64) << 32
        | (id.worker as u64) << 40
        | (id.iteration as u64) << 48
        | LIVE
}

/// Who last wrote the entry of the position `key` names, or `None` when the
/// side table holds no record of it.
pub fn writer_of(key: u64) -> Option<WriterId> {
    let record = WRITERS[writer_slot(key)].load(Ordering::Relaxed);
    (record & LIVE != 0 && tag(record) == tag(key)).then_some(WriterId {
        generation: (record >> 32) as u8,
        worker: (record >> 40) as u8,
        iteration: (record >> 48) as u8,
    })
}

/// Record `id` as the writer of the position `key` names. Returns whether the
/// record replaced another position's.
pub fn set_writer(key: u64, id: WriterId) -> bool {
    let old = WRITERS[writer_slot(key)].swap(writer_record(key, id), Ordering::Relaxed);
    old & LIVE != 0 && tag(old) != tag(key)
}

/// The smallest remaining depth at which a node enters the presence table. The
/// shallowest nodes are the most numerous and the cheapest to repeat, and
/// keeping them out keeps the table small.
pub const PRESENCE_MIN_DEPTH: i32 = 3;

/// `log2` of the presence table's slot count. It holds at most one node per ply
/// on each worker's path, a few thousand at the most, so 2^16 slots leave it
/// nearly empty; [`PresenceKind::Busy`] says how often a slot was taken.
const PRESENCE_BITS: u32 = 16;

/// The presence table: one record per slot — the key tag, then the worker
/// index and the remaining depth in one byte each — for the node a worker is
/// inside right now.
static PRESENCE: [AtomicU64; 1 << PRESENCE_BITS] =
    [const { AtomicU64::new(0) }; 1 << PRESENCE_BITS];

const fn presence_slot(key: u64) -> usize {
    (key >> (64 - PRESENCE_BITS)) as usize
}

const fn presence_record(key: u64, worker: u8, depth: u8) -> u64 {
    tag(key) | (worker as u64) << 32 | (depth as u64) << 40 | LIVE
}

/// The number of presence-table slots holding a record. Zero whenever no
/// search is running.
pub fn presence_occupancy() -> usize {
    PRESENCE
        .iter()
        .filter(|slot| slot.load(Ordering::Relaxed) != 0)
        .count()
}

/// What [`enter_presence`] found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Seen {
    /// Nobody else: the slot was free and this worker now holds it.
    Alone,
    /// This worker itself is already inside the position — a re-search of the
    /// same node on the same ply. Neither counted nor registered.
    Itself,
    /// Another worker is inside the position; its remaining depth compared with
    /// this one's.
    Other(Cmp),
    /// The slot holds another position.
    Busy,
}

impl Seen {
    /// The presence count this outcome adds beside [`PresenceKind::Entered`].
    pub fn kind(self) -> Option<PresenceKind> {
        match self {
            Seen::Alone | Seen::Itself => None,
            Seen::Other(Cmp::Greater) => Some(PresenceKind::Deeper),
            Seen::Other(Cmp::Equal) => Some(PresenceKind::Equal),
            Seen::Other(Cmp::Less) => Some(PresenceKind::Shallower),
            Seen::Busy => Some(PresenceKind::Busy),
        }
    }
}

/// A worker's hold on a presence-table slot, released when dropped. Holds
/// nothing when the entry did not register.
#[derive(Debug)]
pub struct PresenceHold {
    slot: usize,
    record: u64,
}

impl Drop for PresenceHold {
    fn drop(&mut self) {
        if self.record != 0 {
            // Only the holder ever clears its own record, so this cannot fail.
            let _ = PRESENCE[self.slot].compare_exchange(
                self.record,
                0,
                Ordering::Relaxed,
                Ordering::Relaxed,
            );
        }
    }
}

/// Enter the position `key` names as `worker`, at remaining depth `depth`:
/// register in the presence table when the slot is free, and say who, if
/// anyone, was inside it already.
pub fn enter_presence(key: u64, worker: u8, depth: i32) -> (Seen, PresenceHold) {
    let slot = presence_slot(key);
    let depth8 = depth.clamp(0, u8::MAX as i32) as u8;
    let mine = presence_record(key, worker, depth8);
    let current =
        match PRESENCE[slot].compare_exchange(0, mine, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return (Seen::Alone, PresenceHold { slot, record: mine }),
            Err(current) => current,
        };
    let seen = if tag(current) != tag(key) {
        Seen::Busy
    } else if (current >> 32) as u8 == worker {
        Seen::Itself
    } else {
        Seen::Other(((current >> 40) as u8).cmp(&depth8))
    };
    (seen, PresenceHold { slot, record: 0 })
}

/// One node on a worker's current path, for [`credit_subtree`].
#[derive(Clone, Copy, Debug)]
struct Frame {
    /// The worker's node count when the node was entered.
    entry_nodes: u64,
    /// How many nodes of its subtree are already in [`DupCounts::dup_nodes`].
    credited: u64,
}

/// Room for a path of every ply plus the same-ply re-searches on it, so a push
/// never reaches the allocator.
const PATH_CAPACITY: usize = 512;

thread_local! {
    /// The calling worker's current path, one frame per node from the root
    /// down. A worker runs on one thread for the whole of a search.
    static PATH: RefCell<Vec<Frame>> = RefCell::new(Vec::with_capacity(PATH_CAPACITY));
}

/// A node's frame on the calling thread's path, popped when dropped. Dropping
/// it hands what its subtree had credited up to the node above.
#[derive(Debug)]
pub struct PathFrame(());

impl Drop for PathFrame {
    fn drop(&mut self) {
        PATH.with_borrow_mut(|path| {
            if let Some(frame) = path.pop()
                && let Some(parent) = path.last_mut()
            {
                parent.credited += frame.credited;
            }
        });
    }
}

/// Push a frame for a node entered when the worker's node count stood at
/// `nodes`.
pub fn enter_path(nodes: u64) -> PathFrame {
    PATH.with_borrow_mut(|path| {
        path.push(Frame {
            entry_nodes: nodes,
            credited: 0,
        });
    });
    PathFrame(())
}

/// The node at the top of the calling thread's path is a duplicated subtree, and
/// the worker's node count stands at `nodes`: mark the whole subtree under it
/// credited, and return how many of its nodes were not already — the amount to
/// add to [`DupCounts::dup_nodes`]. A duplicated subtree inside a duplicated
/// subtree is so counted once, and a node credited twice adds only what it
/// searched in between.
pub fn credit_subtree(nodes: u64) -> u64 {
    PATH.with_borrow_mut(|path| {
        let Some(frame) = path.last_mut() else {
            return 0;
        };
        let subtree = nodes - frame.entry_nodes;
        let added = subtree.saturating_sub(frame.credited);
        frame.credited = frame.credited.max(subtree);
        added
    })
}

/// How many frames the calling thread's path holds. Zero whenever the thread is
/// not inside a search.
pub fn path_len() -> usize {
    PATH.with_borrow(Vec::len)
}

/// Drop every frame on the calling thread's path.
pub fn clear_path() {
    PATH.with_borrow_mut(Vec::clear);
}

#[cfg(test)]
mod tests {
    use super::*;

    const ME: WriterId = WriterId {
        generation: 3,
        worker: 1,
        iteration: 9,
    };

    #[test]
    fn a_store_into_another_position_or_nothing_is_new() {
        assert_eq!(classify(None, 5, false, None, ME), StoreKind::New);
        assert_eq!(classify(None, 5, true, Some(ME), ME), StoreKind::New);
    }

    #[test]
    fn an_entry_with_no_searched_depth_is_unsearched_when_the_probe_found_it() {
        assert_eq!(
            classify(Some(0), 5, true, Some(ME), ME),
            StoreKind::Unsearched
        );
        assert_eq!(classify(Some(-2), 1, true, None, ME), StoreKind::Unsearched);
        // The node's own static-eval store after a miss.
        assert_eq!(classify(Some(-2), 1, false, None, ME), StoreKind::New);
    }

    #[test]
    fn the_writer_splits_by_search_worker_and_iteration() {
        let earlier_iteration = WriterId { iteration: 8, ..ME };
        let other_worker = WriterId { worker: 0, ..ME };
        let earlier_search = WriterId {
            generation: 2,
            ..ME
        };
        let cases = [
            (Some(ME), Writer::ThisIteration),
            (Some(earlier_iteration), Writer::EarlierIteration),
            (Some(other_worker), Writer::OtherWorker),
            (Some(earlier_search), Writer::EarlierSearch),
            (None, Writer::Unknown),
        ];
        for (previous, writer) in cases {
            // The probe's outcome does not matter once a searched depth is
            // there: another worker may have stored it since.
            for probe_hit in [false, true] {
                assert_eq!(
                    classify(Some(4), 5, probe_hit, previous, ME),
                    StoreKind::Deepened(writer),
                    "shallower, {previous:?}"
                );
                assert_eq!(
                    classify(Some(5), 5, probe_hit, previous, ME),
                    StoreKind::AlreadyThere(writer),
                    "equal, {previous:?}"
                );
                assert_eq!(
                    classify(Some(9), 5, probe_hit, previous, ME),
                    StoreKind::AlreadyThere(writer),
                    "deeper, {previous:?}"
                );
            }
        }
    }

    #[test]
    fn every_store_kind_has_its_own_index() {
        for (i, kind) in StoreKind::ALL.iter().enumerate() {
            assert_eq!(kind.index(), i, "{kind:?}");
        }
        for (i, kind) in PresenceKind::ALL.iter().enumerate() {
            assert_eq!(kind.index(), i, "{kind:?}");
        }
    }

    #[test]
    fn the_writer_table_returns_what_was_set_and_tells_a_collision() {
        let key = 0x1234_5678_9abc_def0;
        assert_eq!(writer_of(key), None);
        assert!(!set_writer(key, ME));
        assert_eq!(writer_of(key), Some(ME));
        // Same slot (same high bits), another tag.
        let rival = key ^ 0x1;
        assert_eq!(writer_of(rival), None);
        assert!(set_writer(rival, ME));
        assert_eq!(writer_of(key), None);
    }

    #[test]
    fn presence_is_entered_found_and_left() {
        let key = 0x0f0f_0000_0000_0042;
        let before = presence_occupancy();
        let (seen, hold) = enter_presence(key, 0, 6);
        assert_eq!(seen, Seen::Alone);
        assert_eq!(presence_occupancy(), before + 1);
        {
            let (seen, _h) = enter_presence(key, 0, 4);
            assert_eq!(seen, Seen::Itself);
            assert_eq!(seen.kind(), None);
            let (seen, _h) = enter_presence(key, 1, 5);
            assert_eq!(seen, Seen::Other(Cmp::Greater));
            assert_eq!(seen.kind(), Some(PresenceKind::Deeper));
            let (seen, _h) = enter_presence(key, 2, 6);
            assert_eq!(seen.kind(), Some(PresenceKind::Equal));
            let (seen, _h) = enter_presence(key, 3, 7);
            assert_eq!(seen.kind(), Some(PresenceKind::Shallower));
            let (seen, _h) = enter_presence(key ^ 0x10, 3, 7);
            assert_eq!(seen, Seen::Busy);
        }
        // The holds that did not register release nothing.
        assert_eq!(presence_occupancy(), before + 1);
        drop(hold);
        assert_eq!(presence_occupancy(), before);
        let (seen, _hold) = enter_presence(key, 1, 5);
        assert_eq!(seen, Seen::Alone);
    }

    #[test]
    fn a_duplicated_subtree_inside_a_duplicated_subtree_is_counted_once() {
        clear_path();
        let outer = enter_path(100);
        {
            let _sibling = enter_path(101);
            // Not duplicated: nothing credited.
        }
        let mut total = 0;
        {
            let _inner = enter_path(110);
            {
                let _innermost = enter_path(115);
                total += credit_subtree(125);
                assert_eq!(total, 10);
            }
            // The inner node is duplicated too: only what is not already
            // counted below it.
            total += credit_subtree(150);
            assert_eq!(total, 40);
        }
        {
            let _later = enter_path(160);
            total += credit_subtree(170);
            assert_eq!(total, 50);
        }
        // The outer node covers everything under it once.
        total += credit_subtree(200);
        assert_eq!(total, 100);
        // Credited again after searching on: only the new part.
        total += credit_subtree(230);
        assert_eq!(total, 130);
        drop(outer);
        assert_eq!(path_len(), 0);
    }

    #[test]
    fn a_node_falls_in_one_bucket_per_dimension() {
        let mut counts = DupCounts::ZERO;
        counts.count_store(StoreKind::New, Where::new(1, 0, true, false, 0));
        counts.count_store(StoreKind::New, Where::new(7, 13, false, true, 3));
        counts.count_store(StoreKind::New, Where::new(30, 40, false, false, 9));
        let h = counts.stores[StoreKind::New.index()];
        assert_eq!(h[0], [1, 0, 1, 1, 0]);
        assert_eq!(h[1], [1, 0, 0, 1, 1]);
        assert_eq!(h[2], [1, 1, 1, 0, 0]);
        assert_eq!(h[3], [1, 0, 1, 1, 0]);
        assert_eq!(counts.stores_of(StoreKind::New), 3);
        let mut sum = DupCounts::ZERO;
        sum.add(&counts);
        sum.add(&counts);
        assert_eq!(sum.stores_of(StoreKind::New), 6);
    }
}
