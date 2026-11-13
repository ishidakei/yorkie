//! Parallel workers putting off a move another worker is searching right now.
//!
//! The workers of a parallel search share one small table, apart from the
//! transposition table, of the (position, move) pairs some worker is in the
//! middle of searching: the [`BusyTable`]. At a node with a remaining depth of
//! at least [`MIN_DEPTH`], every move after the first one the node searches is
//! looked up before it is searched. A move the table says another worker is on
//! goes to the node's list of deferred moves and the node goes on with its next
//! move; any other move is entered in the table, searched, and removed from it
//! again. Once the node's own order has run out the deferred moves come back,
//! in the order they were put off and without being looked up again, by when
//! the other worker's result is usually in the transposition table. A node that
//! cuts off before then searches none of them. This is the scheme published as
//! "Simplified ABDADA".
//!
//! The first move a node searches is never put off, since searching it first is
//! what makes the rest of the node cheap; nor is it entered, so no other worker
//! puts it off either. A move searched from the deferred list is entered like
//! any other, so the workers that reach the pair after it put it off in turn.
//!
//! The table takes no locks. A lost race costs one move searched twice or one
//! move put off needlessly, never a wrong result: the table decides only the
//! order a node searches its moves in, and every move a node puts off it still
//! searches unless it cuts off first. Every entry is removed by the worker that
//! made it as soon as the move's search returns, a stopped search's included,
//! so no entry outlives the search that made it.

use std::sync::atomic::{AtomicU64, Ordering};

/// The least remaining depth at which a node looks its moves up and enters
/// them. Below it a move's subtree is too small for another worker's result to
/// be worth waiting for, and the table traffic would cost more than it saves.
pub const MIN_DEPTH: i32 = 3;

/// The number of bits of a word that pick its bucket.
const BUCKET_BITS: u32 = 13;

/// The number of buckets: 8192 of [`WAYS`] words each, 256 KiB in all. The
/// entries live at any one time are at most one per worker per ply at
/// [`MIN_DEPTH`] or deeper, a few thousand even with 192 workers, so the table
/// is mostly empty at this size and a bucket rarely fills. A smaller table than
/// the published 32768 buckets keeps fewer cache lines moving between the
/// workers, which on a machine whose workers span two NUMA nodes means between
/// the nodes' caches.
pub const BUCKETS: usize = 1 << BUCKET_BITS;

/// The words of one bucket.
pub const WAYS: usize = 4;

/// The most moves one node puts off. A node whose list is full searches its
/// further moves when it reaches them, busy or not.
pub const DEFERRED_MAX: usize = 32;

/// One bucket: 32 bytes, aligned so it never straddles a cache line.
#[repr(C, align(32))]
struct Bucket([AtomicU64; WAYS]);

/// The (position, move) pairs some worker is searching, as [`word`]s. Zero
/// marks an empty way.
pub struct BusyTable {
    buckets: [Bucket; BUCKETS],
}

static SHARED: BusyTable = BusyTable::new();

impl BusyTable {
    /// An empty table.
    pub const fn new() -> Self {
        Self {
            buckets: [const { Bucket([const { AtomicU64::new(0) }; WAYS]) }; BUCKETS],
        }
    }

    /// The table every worker of the process shares.
    #[inline]
    pub fn shared() -> &'static Self {
        &SHARED
    }

    #[inline]
    fn bucket(&self, word: u64) -> &[AtomicU64; WAYS] {
        &self.buckets[(word >> (u64::BITS - BUCKET_BITS)) as usize].0
    }

    /// Whether some worker has entered `word` and not yet cleared it.
    #[inline]
    pub fn probe(&self, word: u64) -> bool {
        self.bucket(word)
            .iter()
            .any(|w| w.load(Ordering::Relaxed) == word)
    }

    /// Enter `word`: into an empty way of its bucket, or, when every way holds
    /// a word, in place of the one its low bits name. A full bucket never keeps
    /// a pair out; the pair it replaces is only searched without being put off.
    #[inline]
    pub fn enter(&self, word: u64) {
        let bucket = self.bucket(word);
        for way in bucket {
            if way
                .compare_exchange(0, word, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return;
            }
        }
        bucket[word as usize % WAYS].store(word, Ordering::Relaxed);
    }

    /// Remove one copy of `word`, if the bucket still holds one: another
    /// worker's entry may have replaced it, or another worker that entered the
    /// same pair may have removed it.
    #[inline]
    pub fn clear(&self, word: u64) {
        for way in self.bucket(word) {
            if way.load(Ordering::Relaxed) == word
                && way
                    .compare_exchange(word, 0, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok()
            {
                return;
            }
        }
    }

    /// The number of entries the table holds.
    pub fn occupancy(&self) -> usize {
        self.buckets
            .iter()
            .flat_map(|b| &b.0)
            .filter(|w| w.load(Ordering::Relaxed) != 0)
            .count()
    }
}

impl Default for BusyTable {
    fn default() -> Self {
        Self::new()
    }
}

/// The table's word for the move `mv` (its 16-bit form) in the position `key`.
/// Both steps are one-to-one, so two moves of one position never share a word;
/// a word of zero, which would read as an empty way, becomes one.
#[inline]
pub fn word(key: u64, mv: u16) -> u64 {
    let w = splitmix(key ^ u64::from(mv).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    if w == 0 { 1 } else { w }
}

/// What a worker counted about the moves it put off, for the statistics line.
#[cfg(feature = "verbose1")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeferCounts {
    /// Moves put off because another worker was searching them.
    pub deferred: u64,
    /// Moves searched from the deferred list whose child node returned the
    /// transposition table's value without searching.
    pub tt_cuts: u64,
    /// Entries made in the table.
    pub entries: u64,
}

#[cfg(feature = "verbose1")]
impl DeferCounts {
    /// Fold `other` in.
    pub fn add(&mut self, other: &Self) {
        self.deferred += other.deferred;
        self.tt_cuts += other.tt_cuts;
        self.entries += other.entries;
    }
}

/// The moves one node has put off, in the order it put them off, and how far
/// the node has come through them. Only the first `len` slots are live.
#[derive(Clone, Copy, Debug)]
pub struct Deferred<T: Copy> {
    moves: [Option<T>; DEFERRED_MAX],
    len: usize,
    /// The next deferred move to hand out, once `draining`.
    next: usize,
    /// Whether the node's own order has run out, so that its moves now come
    /// from the deferred list.
    draining: bool,
}

impl<T: Copy> Deferred<T> {
    const EMPTY: Self = Self {
        moves: [None; DEFERRED_MAX],
        len: 0,
        next: 0,
        draining: false,
    };
}

/// One worker's side of the scheme: whether it is on, and the moves each ply
/// has put off.
pub struct Defer<T: Copy> {
    on: bool,
    active: bool,
    deferred: Box<[Deferred<T>]>,
    #[cfg(feature = "verbose1")]
    counts: DeferCounts,
}

impl<T: Copy> Defer<T> {
    /// A worker's state over `plies` plies, turned on when `on` and there is
    /// more than one worker.
    pub fn new(plies: usize, on: bool, workers: usize) -> Self {
        let mut defer = Self {
            on: false,
            active: false,
            deferred: vec![Deferred::EMPTY; plies].into_boxed_slice(),
            #[cfg(feature = "verbose1")]
            counts: DeferCounts::default(),
        };
        defer.set(on, workers);
        defer
    }

    /// Turn the scheme on or off for a pool of `workers`. One worker has no one
    /// to wait for, so it is off whatever `on` says.
    pub fn set(&mut self, on: bool, workers: usize) {
        self.on = on;
        self.set_workers(workers);
    }

    /// Size the pool the worker searches in, leaving the switch as it is.
    pub fn set_workers(&mut self, workers: usize) {
        self.active = self.on && workers > 1;
    }

    /// Whether the worker puts moves off and enters them.
    #[inline]
    pub fn active(&self) -> bool {
        self.active
    }

    /// Start a search afresh: nothing counted.
    pub fn reset(&mut self) {
        #[cfg(feature = "verbose1")]
        {
            self.counts = DeferCounts::default();
        }
    }

    /// What this worker counted since [`Self::reset`].
    #[cfg(feature = "verbose1")]
    pub fn counts(&self) -> DeferCounts {
        self.counts
    }

    /// Note an entry made in the table.
    #[cfg(feature = "verbose1")]
    #[inline]
    pub fn note_entry(&mut self) {
        self.counts.entries += 1;
    }

    /// Note a deferred move whose child node returned the table's value.
    #[cfg(feature = "verbose1")]
    #[inline]
    pub fn note_tt_cut(&mut self) {
        self.counts.tt_cuts += 1;
    }

    /// Whether the node at stack index `s` is searching its deferred moves.
    #[cfg(feature = "verbose1")]
    #[inline]
    pub fn draining(&self, s: usize) -> bool {
        self.deferred[s].draining
    }

    /// Forget what the node at stack index `s` put off. Called as a node starts
    /// its move loop, since the node that last used the ply may have cut off
    /// with moves still deferred.
    #[inline]
    pub fn begin_node(&mut self, s: usize) {
        let d = &mut self.deferred[s];
        d.len = 0;
        d.next = 0;
        d.draining = false;
    }

    /// What the node at stack index `s` has put off, to hand back with
    /// [`Self::restore`] once a search that re-enters on the same index — a
    /// singular search — has returned.
    #[inline]
    pub fn hold(&self, s: usize) -> Deferred<T> {
        self.deferred[s]
    }

    /// Put back what [`Self::hold`] took for the node at stack index `s`.
    #[inline]
    pub fn restore(&mut self, s: usize, held: Deferred<T>) {
        self.deferred[s] = held;
    }

    /// The next move the node at stack index `s` searches. `pull` hands out the
    /// node's order, `busy` says whether another worker is searching a move,
    /// `depth` is the node's remaining depth and `first` says the node has not
    /// counted a move yet. Off, below [`MIN_DEPTH`] and for the first move this
    /// is the order's next move; otherwise a busy move is put off, and once the
    /// order has run out the deferred moves come out in the order they went in.
    pub fn next(
        &mut self,
        s: usize,
        depth: i32,
        first: bool,
        mut pull: impl FnMut() -> Option<T>,
        mut busy: impl FnMut(T) -> bool,
    ) -> Option<T> {
        let d = &mut self.deferred[s];
        if !d.draining {
            let looks = self.active && !first && depth >= MIN_DEPTH;
            while let Some(mv) = pull() {
                if looks && d.len < DEFERRED_MAX && busy(mv) {
                    d.moves[d.len] = Some(mv);
                    d.len += 1;
                    #[cfg(feature = "verbose1")]
                    {
                        self.counts.deferred += 1;
                    }
                    continue;
                }
                return Some(mv);
            }
            d.draining = true;
        }
        if d.next < d.len {
            d.next += 1;
            d.moves[d.next - 1]
        } else {
            None
        }
    }
}

/// The `splitmix64` finalizer: one-to-one, and spreads every input bit over
/// the output.
const fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> Box<BusyTable> {
        Box::default()
    }

    #[test]
    fn an_entry_is_found_until_it_is_cleared() {
        let t = table();
        let w = word(0x0123_4567_89AB_CDEF, 0x1234);
        assert!(!t.probe(w));
        t.enter(w);
        assert!(t.probe(w));
        assert_eq!(t.occupancy(), 1);
        t.clear(w);
        assert!(!t.probe(w));
        assert_eq!(t.occupancy(), 0);
        // Clearing what is not there touches nothing.
        t.enter(w);
        t.clear(w ^ 1 << 3);
        assert!(t.probe(w));
    }

    /// Two workers entering one pair leave two copies; each clear removes one.
    #[test]
    fn each_clear_removes_one_copy() {
        let t = table();
        let w = word(42, 7);
        t.enter(w);
        t.enter(w);
        t.clear(w);
        assert!(t.probe(w));
        t.clear(w);
        assert!(!t.probe(w));
        assert_eq!(t.occupancy(), 0);
    }

    /// Words of one bucket: the bucket bits fixed, the rest varied.
    fn same_bucket(n: u64) -> impl Iterator<Item = u64> {
        (1..=n).map(|i| (0x5A5u64 << (u64::BITS - BUCKET_BITS)) | (i * 0x1_0001))
    }

    #[test]
    fn a_full_bucket_replaces_rather_than_blocks() {
        let t = table();
        let words: Vec<u64> = same_bucket(WAYS as u64 + 1).collect();
        for &w in &words[..WAYS] {
            t.enter(w);
        }
        assert!(words[..WAYS].iter().all(|&w| t.probe(w)));
        let extra = words[WAYS];
        t.enter(extra);
        assert!(t.probe(extra), "the new pair is entered");
        assert_eq!(
            words[..WAYS].iter().filter(|&&w| t.probe(w)).count(),
            WAYS - 1,
            "one pair made way for it"
        );
        assert_eq!(t.occupancy(), WAYS);
    }

    /// Every move of a position has a word of its own: a set of keys, each
    /// with every 16-bit move form, gives as many words as pairs.
    #[test]
    fn two_moves_of_one_position_do_not_alias() {
        let keys = [
            0,
            1,
            0xFFFF_FFFF_FFFF_FFFF,
            0x0123_4567_89AB_CDEF,
            0x8000_0000_0000_0000,
        ];
        let moves = if cfg!(miri) { 0x200u32 } else { 0x1_0000 };
        for key in keys {
            let mut words: Vec<u64> = (0..moves).map(|mv| word(key, mv as u16)).collect();
            assert!(words.iter().all(|&w| w != 0));
            words.sort_unstable();
            words.dedup();
            assert_eq!(words.len(), moves as usize, "key {key:#x}");
        }
    }

    /// Drive one node's loop: `moves` in order, `busy` the stub that says
    /// which another worker is on, `cut_after` the number of searched moves
    /// after which the node cuts off. Returns the searched moves in order.
    fn run_node(
        defer: &mut Defer<u32>,
        moves: &[u32],
        depth: i32,
        busy: impl Fn(u32) -> bool,
        cut_after: Option<usize>,
    ) -> Vec<u32> {
        defer.begin_node(2);
        let mut order = moves.iter().copied();
        let mut searched = Vec::new();
        while let Some(mv) = defer.next(2, depth, searched.is_empty(), || order.next(), &busy) {
            searched.push(mv);
            if cut_after == Some(searched.len()) {
                break;
            }
        }
        searched
    }

    fn on() -> Defer<u32> {
        Defer::new(4, true, 2)
    }

    #[test]
    fn a_deferred_move_is_searched_once_after_the_others() {
        let moves: Vec<u32> = (0..10).collect();
        let busy = |mv: u32| mv == 3 || mv == 6;
        let mut defer = on();
        let searched = run_node(&mut defer, &moves, MIN_DEPTH, busy, None);
        assert_eq!(searched, [0, 1, 2, 4, 5, 7, 8, 9, 3, 6]);
        #[cfg(feature = "verbose1")]
        assert_eq!(defer.counts().deferred, 2);
    }

    #[test]
    fn a_cutoff_before_the_deferred_pass_searches_none_of_them() {
        let moves: Vec<u32> = (0..10).collect();
        let mut defer = on();
        let searched = run_node(&mut defer, &moves, 8, |mv| mv % 2 == 1, Some(4));
        assert_eq!(searched, [0, 2, 4, 6]);
        // The next node on the ply starts with nothing deferred.
        let searched = run_node(&mut defer, &moves, 8, |_| false, None);
        assert_eq!(searched, moves);
    }

    #[test]
    fn the_first_move_is_never_deferred() {
        let moves: Vec<u32> = (0..5).collect();
        let mut defer = on();
        let searched = run_node(&mut defer, &moves, 8, |_| true, None);
        assert_eq!(
            searched, moves,
            "the first is searched, the rest come back in order"
        );
    }

    #[test]
    fn off_alone_or_shallow_the_order_is_kept() {
        let moves: Vec<u32> = (0..10).collect();
        let busy = |mv: u32| mv % 3 == 1;
        for mut defer in [Defer::new(4, false, 2), Defer::new(4, true, 1)] {
            assert!(!defer.active());
            assert_eq!(run_node(&mut defer, &moves, 8, busy, None), moves);
        }
        let mut shallow = on();
        assert_eq!(
            run_node(&mut shallow, &moves, MIN_DEPTH - 1, busy, None),
            moves
        );
        let mut pooled: Defer<u32> = Defer::new(4, true, 1);
        pooled.set_workers(2);
        assert!(pooled.active());
    }

    #[test]
    fn a_full_list_searches_further_moves_at_once() {
        let moves: Vec<u32> = (0..(DEFERRED_MAX as u32 + 5)).collect();
        let mut defer = on();
        let searched = run_node(&mut defer, &moves, 8, |_| true, None);
        let mut expected: Vec<u32> = vec![0];
        expected.extend(DEFERRED_MAX as u32 + 1..DEFERRED_MAX as u32 + 5);
        expected.extend(1..=DEFERRED_MAX as u32);
        assert_eq!(searched, expected);
    }

    /// What a node has put off survives a search that re-entered on its index.
    #[test]
    fn a_held_list_comes_back_whole() {
        let mut defer = on();
        defer.begin_node(2);
        let mut outer = 0..6u32;
        let busy = |mv: u32| mv == 2;
        assert_eq!(defer.next(2, 8, true, || outer.next(), busy), Some(0));
        assert_eq!(defer.next(2, 8, false, || outer.next(), busy), Some(1));
        assert_eq!(defer.next(2, 8, false, || outer.next(), busy), Some(3));
        let held = defer.hold(2);
        let inner = run_node(&mut defer, &[100, 101, 102], 8, |_| true, None);
        assert_eq!(inner, [100, 101, 102]);
        defer.restore(2, held);
        let rest: Vec<u32> =
            std::iter::from_fn(|| defer.next(2, 8, false, || outer.next(), busy)).collect();
        assert_eq!(rest, [4, 5, 2]);
    }
}
