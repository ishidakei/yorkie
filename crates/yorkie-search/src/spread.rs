//! Parallel workers spreading their move order apart.
//!
//! Every worker of a parallel search runs the same iterative deepening over the
//! same move order and shares only the transposition table, so much of what one
//! worker searches another has already searched. Here each worker keeps a
//! **duplicate count** of its own. A store at the end of a node that finds the
//! position's entry already holding a result at least as deep, put there by
//! someone other than this worker in this iteration, raises the count by one,
//! up to the pool size; any other such store lowers it by one, down to zero.
//!
//! While the count `c` is above zero the worker does not take the next move of
//! a node's order but one `k` places further on, `k` drawn uniformly from
//! `0..n` with `n` the smallest of `2^c`, [`WINDOW_MAX`] and the moves left. The
//! moves passed over stay in their order and come out later, each exactly once.
//! The first move a node searches is exempt ([`FIRST_MOVE_EXEMPT`]): it is the
//! table's move, or the order's best guess, and searching it first is what
//! makes the rest of the node cheap. A move the node passes over without
//! counting it — the move a singular search excludes, a root move outside the
//! lines still being searched — does not use the exemption up. Collisions push
//! the workers apart; quiet lets them drift back to the plain order.
//!
//! The move order generates its moves stage by stage, and a shifted pick pulls
//! up to `n` moves out of it, into later stages when the current one runs out;
//! "the moves left" is how many it could pull, so the window never reaches past
//! the end of the order. When the node stops searching quiet moves, the quiet
//! moves already pulled out are dropped with the rest.
//!
//! Each worker owns its state outright: nothing here is shared or atomic.

/// The widest window a shifted pick draws from, and so the length of the
/// per-ply buffer of moves pulled out of the order and not yet searched.
pub const WINDOW_MAX: usize = 16;

/// Whether a node's first move is always searched first, whatever the count.
pub const FIRST_MOVE_EXEMPT: bool = true;

/// Whether a store that finds an entry this worker itself wrote in the current
/// iteration is left uncounted. A worker meets its own entries whenever it
/// searches a subtree again — an aspiration window that failed, a null-window
/// search that has to be repeated with the full one — and those are not
/// another worker's work.
pub const IGNORE_OWN_ENTRIES: bool = true;

/// How much deeper than the store the entry has to be for the store to count:
/// `0` counts an entry as deep as the store, `1` only a deeper one.
pub const MIN_LEAD: i32 = 0;

/// The number of slots of a worker's record of its own stores. Direct-mapped
/// by the low bits of the position key; a slot overwritten by a later store
/// forgets the earlier one, which can only make an own entry count.
const OWN_SLOTS: usize = 4096;

/// What a worker counted about its spread, for the statistics line.
#[cfg(feature = "verbose1")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpreadCounts {
    /// Stores that found another worker's result already there.
    pub dups: u64,
    /// Moves taken from further down the order than the next one.
    pub shifts: u64,
    /// The highest duplicate count reached.
    pub peak: u64,
}

#[cfg(feature = "verbose1")]
impl SpreadCounts {
    /// Fold `other` in: the counts add, the peak is the higher of the two.
    pub fn add(&mut self, other: &Self) {
        self.dups += other.dups;
        self.shifts += other.shifts;
        self.peak = self.peak.max(other.peak);
    }
}

/// The moves one node pulled out of its order and has not searched yet, in
/// the order's order, each with whether the order handed it out as a quiet
/// move. Only the first `len` slots are live.
#[derive(Clone, Copy, Debug)]
pub struct Pending<T: Copy> {
    moves: [Option<(T, bool)>; WINDOW_MAX],
    len: usize,
}

impl<T: Copy> Pending<T> {
    const EMPTY: Self = Self {
        moves: [None; WINDOW_MAX],
        len: 0,
    };

    /// Take the `k`-th pending move out, keeping the others in order.
    fn take(&mut self, k: usize) -> T {
        debug_assert!(k < self.len);
        let (taken, _) = self.moves[k].expect("a pending slot below len holds a move");
        self.moves.copy_within(k + 1..self.len, k);
        self.len -= 1;
        taken
    }

    fn push(&mut self, mv: T, quiet: bool) {
        self.moves[self.len] = Some((mv, quiet));
        self.len += 1;
    }

    /// Drop the quiet moves, keeping the others in order.
    fn drop_quiets(&mut self) {
        let mut kept = 0;
        for i in 0..self.len {
            if let Some((_, false)) = self.moves[i] {
                self.moves[kept] = self.moves[i];
                kept += 1;
            }
        }
        self.len = kept;
    }
}

/// One worker's spread: its duplicate count, the generator its draws come
/// from, the record of its own stores and the per-ply pending moves.
pub struct Spread<T: Copy> {
    on: bool,
    active: bool,
    count: u32,
    cap: u32,
    rng: u64,
    own: Box<[u32; OWN_SLOTS]>,
    pending: Box<[Pending<T>]>,
    #[cfg(feature = "verbose1")]
    counts: SpreadCounts,
}

impl<T: Copy> Spread<T> {
    /// A spread over `plies` plies of pending moves, its count bounded by
    /// `workers`, turned on when `active` and there is more than one worker.
    pub fn new(plies: usize, active: bool, workers: usize) -> Self {
        let mut spread = Self {
            on: false,
            active: false,
            count: 0,
            cap: 0,
            rng: 0,
            own: Box::new([0; OWN_SLOTS]),
            pending: vec![Pending::EMPTY; plies].into_boxed_slice(),
            #[cfg(feature = "verbose1")]
            counts: SpreadCounts::default(),
        };
        spread.set(active, workers);
        spread.seed(0);
        spread
    }

    /// Turn the spread on or off for a pool of `workers`. One worker has no one
    /// to spread from, so it is off whatever `on` says.
    pub fn set(&mut self, on: bool, workers: usize) {
        self.on = on;
        self.set_workers(workers);
    }

    /// Size the pool the worker searches in, leaving the switch as it is. The
    /// pool a search runs can be smaller than the build's worker count, and a
    /// worker alone in its pool does not spread.
    pub fn set_workers(&mut self, workers: usize) {
        self.active = self.on && workers > 1;
        self.cap = u32::try_from(workers).unwrap_or(u32::MAX);
    }

    /// Whether the spread is on.
    #[inline]
    pub fn active(&self) -> bool {
        self.active
    }

    /// Seed the generator from the worker's index, so every worker draws its
    /// own sequence.
    pub fn seed(&mut self, worker: usize) {
        self.rng = splitmix(worker as u64 + 1) | 1;
    }

    /// The duplicate count.
    pub fn count(&self) -> u32 {
        self.count
    }

    /// Start a search afresh: count zero, no own stores remembered, nothing
    /// counted. The generator carries on.
    pub fn reset(&mut self) {
        self.count = 0;
        self.own.fill(0);
        #[cfg(feature = "verbose1")]
        {
            self.counts = SpreadCounts::default();
        }
    }

    /// What this worker counted since [`Self::reset`].
    #[cfg(feature = "verbose1")]
    pub fn counts(&self) -> SpreadCounts {
        self.counts
    }

    /// A uniform draw from `0..n`, `n > 0`.
    fn below(&mut self, n: usize) -> usize {
        // xorshift64*: nothing here needs more than independent-looking draws.
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        let r = self.rng.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32;
        ((r * n as u64) >> 32) as usize
    }

    /// The window a shifted pick draws from at the current count, before the
    /// moves left bound it.
    fn window(&self) -> usize {
        (1usize << self.count.min(usize::BITS - 1)).min(WINDOW_MAX)
    }

    /// Forget the moves the node at stack index `s` left pending. Called as a
    /// node starts its move loop, since the node that last used the ply may have
    /// cut off with moves still pending.
    #[inline]
    pub fn begin_node(&mut self, s: usize) {
        self.pending[s].len = 0;
    }

    /// The moves the node at stack index `s` has pending, to hand back with
    /// [`Self::restore`] once a search that re-enters on the same index — a
    /// singular search — has returned, leaving its own pending moves there.
    #[inline]
    pub fn hold(&self, s: usize) -> Pending<T> {
        self.pending[s]
    }

    /// Put back what [`Self::hold`] took for the node at stack index `s`.
    #[inline]
    pub fn restore(&mut self, s: usize, held: Pending<T>) {
        self.pending[s] = held;
    }

    /// The node at stack index `s` searches no more quiet moves: drop the ones
    /// it has pending.
    #[inline]
    pub fn skip_quiets(&mut self, s: usize) {
        self.pending[s].drop_quiets();
    }

    /// The next move the node at stack index `s` searches, `pull` handing out
    /// the order's next move and whether it is a quiet one. `first` says the
    /// node has not counted a move yet. With the spread off, at count zero and
    /// for an exempt first move this is the order's next move, a pending one
    /// first.
    pub fn pick(
        &mut self,
        s: usize,
        first: bool,
        mut pull: impl FnMut() -> Option<(T, bool)>,
    ) -> Option<T> {
        if !self.active || self.count == 0 || (first && FIRST_MOVE_EXEMPT) {
            let pending = &mut self.pending[s];
            return if pending.len > 0 {
                Some(pending.take(0))
            } else {
                pull().map(|(mv, _)| mv)
            };
        }
        let n = self.window();
        let pending = &mut self.pending[s];
        while pending.len < n {
            match pull() {
                Some((mv, quiet)) => pending.push(mv, quiet),
                None => break,
            }
        }
        let len = pending.len;
        if len == 0 {
            return None;
        }
        let k = self.below(len);
        #[cfg(feature = "verbose1")]
        {
            self.counts.shifts += u64::from(k != 0);
        }
        Some(self.pending[s].take(k))
    }

    /// Note the store about to end a node: the position `key`, stored at
    /// `depth` in iteration `iteration`, finds `found`, the depth the entry
    /// holds that position at, if it does. Moves the count and returns whether
    /// the store found another worker's result.
    pub fn note_store(&mut self, key: u64, found: Option<i32>, depth: i32, iteration: i32) -> bool {
        if !self.active {
            return false;
        }
        let slot = (key as usize) % OWN_SLOTS;
        let tag = own_tag(key, iteration);
        let own = IGNORE_OWN_ENTRIES && self.own[slot] == tag;
        self.own[slot] = tag;
        let dup = !own && found.is_some_and(|d| d >= depth + MIN_LEAD);
        if dup {
            self.count = (self.count + 1).min(self.cap);
            #[cfg(feature = "verbose1")]
            {
                self.counts.dups += 1;
                self.counts.peak = self.counts.peak.max(u64::from(self.count));
            }
        } else {
            self.count = self.count.saturating_sub(1);
        }
        dup
    }
}

/// The own-store record's entry for `key` stored in `iteration`: the key's top
/// 24 bits over the iteration's low 8, never zero, so an empty slot matches
/// nothing.
fn own_tag(key: u64, iteration: i32) -> u32 {
    (((key >> 40) as u32) << 8 | (iteration as u32 & 0xff)) | 0x8000_0000
}

/// The `splitmix64` finalizer: spreads a small seed over every bit.
const fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run one node of `moves` through the spread at count `count` (held there
    /// for the whole node), returning the order the moves came out in.
    fn drain(spread: &mut Spread<u32>, moves: &[u32], count: u32) -> Vec<u32> {
        spread.count = count;
        spread.begin_node(3);
        let mut order = moves.iter().copied();
        let mut out = Vec::new();
        let mut first = true;
        while let Some(mv) = spread.pick(3, first, || order.next().map(|mv| (mv, false))) {
            first = false;
            out.push(mv);
        }
        out
    }

    fn on(workers: usize) -> Spread<u32> {
        Spread::new(8, true, workers)
    }

    #[test]
    fn every_move_comes_out_once_and_the_first_move_first() {
        for len in [0usize, 1, 2, 3, 5, 16, 17, 40] {
            let moves: Vec<u32> = (100..100 + len as u32).collect();
            for count in 0..=8 {
                for worker in 0..20 {
                    let mut spread = on(64);
                    spread.seed(worker);
                    let out = drain(&mut spread, &moves, count);
                    let mut sorted = out.clone();
                    sorted.sort_unstable();
                    assert_eq!(sorted, moves, "len {len} count {count} worker {worker}");
                    if len > 0 {
                        assert_eq!(out[0], moves[0], "the first move is first");
                    }
                }
            }
        }
    }

    #[test]
    fn at_count_zero_the_order_is_the_pickers() {
        let moves: Vec<u32> = (0..40).collect();
        let mut spread = on(4);
        assert_eq!(drain(&mut spread, &moves, 0), moves);
    }

    #[test]
    fn a_spread_that_is_off_keeps_the_order() {
        let moves: Vec<u32> = (0..40).collect();
        let mut off = Spread::new(8, false, 4);
        assert_eq!(drain(&mut off, &moves, 5), moves);
        let mut one_worker = Spread::new(8, true, 1);
        assert!(!one_worker.active());
        assert_eq!(drain(&mut one_worker, &moves, 5), moves);
    }

    /// The switch takes effect only in a pool of more than one worker, and a
    /// pool resized at run time decides it again.
    #[test]
    fn the_pool_size_decides_whether_the_switch_takes_effect() {
        let mut spread: Spread<u32> = Spread::new(8, true, 1);
        assert!(!spread.active(), "alone until a pool says otherwise");
        spread.set_workers(2);
        assert!(spread.active());
        spread.set_workers(1);
        assert!(!spread.active(), "a pool shrunk to one worker");
        let mut off: Spread<u32> = Spread::new(8, false, 1);
        off.set_workers(4);
        assert!(!off.active(), "a pool does not turn the switch on");
    }

    #[test]
    fn a_positive_count_shifts_some_picks() {
        let moves: Vec<u32> = (0..40).collect();
        let mut spread = on(4);
        assert_ne!(drain(&mut spread, &moves, 3), moves);
        #[cfg(feature = "verbose1")]
        assert!(spread.counts().shifts > 0);
    }

    /// Skipping quiets drops the pending quiet moves; the others still come
    /// out, each once and in their order.
    #[test]
    fn skipping_quiets_drops_the_pending_quiet_moves() {
        // Even moves are captures, odd ones quiet.
        let moves: Vec<u32> = (0..40).collect();
        let mut spread = on(64);
        spread.count = 4;
        spread.begin_node(3);
        let mut order = moves.iter().map(|&mv| (mv, mv % 2 == 1));
        let mut out = vec![spread.pick(3, true, || order.next()).unwrap()];
        out.push(spread.pick(3, false, || order.next()).unwrap());
        spread.skip_quiets(3);
        spread.count = 0;
        // The second pick pulled moves 1 to 16 and searched one of them.
        let mut pending = Vec::new();
        for _ in 0..spread.pending[3].len {
            pending.push(spread.pick(3, false, || None).unwrap());
        }
        assert!(pending.iter().all(|mv| mv % 2 == 0), "{pending:?}");
        assert!(pending.is_sorted(), "{pending:?}");
        let expected: Vec<u32> = (1..17)
            .filter(|mv| mv % 2 == 0 && !out.contains(mv))
            .collect();
        assert_eq!(pending, expected, "{out:?} searched");
    }

    /// What a node holds pending survives a search that re-entered on its
    /// index, and only it comes out.
    #[test]
    fn a_held_buffer_comes_back_whole() {
        let mut spread = on(64);
        spread.count = 4;
        spread.begin_node(3);
        let mut outer = (0..40u32).map(|mv| (mv, false));
        spread.pick(3, false, || outer.next()).unwrap();
        let held = spread.hold(3);
        spread.begin_node(3);
        let mut inner = (100..140u32).map(|mv| (mv, false));
        spread.pick(3, false, || inner.next()).unwrap();
        assert!(
            spread.pending[3].len > 0,
            "the inner search left moves pending"
        );
        spread.restore(3, held);
        spread.count = 0;
        let mut rest: Vec<u32> =
            std::iter::from_fn(|| spread.pick(3, false, || outer.next())).collect();
        rest.sort_unstable();
        assert!(rest.iter().all(|&mv| mv < 40), "{rest:?}");
        assert_eq!(rest.len(), 39);
    }

    #[test]
    fn a_shift_stays_inside_the_window() {
        let moves: Vec<u32> = (0..40).collect();
        for count in 1..=6u32 {
            let mut spread = on(64);
            spread.seed(count as usize);
            let out = drain(&mut spread, &moves, count);
            let window = (1usize << count).min(WINDOW_MAX);
            // A move comes out no earlier than `window - 1` places before its
            // place in the order.
            for (at, mv) in out.iter().enumerate() {
                assert!((*mv as usize) < at + window, "count {count}: {out:?}");
            }
        }
    }

    #[test]
    fn the_count_stays_within_zero_and_the_pool_size() {
        let mut spread = on(3);
        let mut state = 0x1234_5678u64;
        for i in 0..10_000 {
            state = splitmix(state);
            let dup = state & 1 == 1;
            // A fresh key every time, so no store is this worker's own.
            let key = splitmix(i) | 1;
            let counted = spread.note_store(key, dup.then_some(9), 5, 1);
            assert_eq!(counted, dup);
            assert!(spread.count() <= 3);
        }
        for i in 0..10 {
            spread.note_store(splitmix(100_000 + i), Some(9), 5, 1);
        }
        assert_eq!(spread.count(), 3, "capped at the pool size");
        for i in 0..10 {
            spread.note_store(splitmix(200_000 + i), None, 5, 1);
        }
        assert_eq!(spread.count(), 0, "floored at zero");
    }

    #[test]
    fn a_store_meeting_this_workers_own_entry_is_not_counted() {
        let mut spread = on(4);
        let key = 0xDEAD_BEEF_0123_4567;
        // The first search of the node stores into an empty entry...
        assert!(!spread.note_store(key, None, 6, 7));
        // ...and its re-search in the same iteration finds that entry.
        assert!(!spread.note_store(key, Some(6), 6, 7));
        assert_eq!(spread.count(), 0);
        // The same entry in the next iteration is no longer this iteration's.
        assert!(spread.note_store(key, Some(6), 6, 8));
        assert_eq!(spread.count(), 1);
        // Another position's entry found as deep counts; a shallower one does not.
        assert!(spread.note_store(key ^ 0xFFFF_0000_0000_0000, Some(6), 6, 8));
        assert!(!spread.note_store(key ^ 0x0F0F_0000_0000_0000, Some(5), 6, 8));
    }

    #[test]
    fn a_reset_forgets_the_count_and_the_own_stores() {
        let mut spread = on(4);
        let key = 0x0123_4567_89AB_CDEF;
        spread.note_store(key, None, 3, 2);
        spread.note_store(key ^ 1 << 50, Some(9), 3, 2);
        assert_eq!(spread.count(), 1);
        spread.reset();
        assert_eq!(spread.count(), 0);
        assert!(spread.note_store(key, Some(3), 3, 2));
    }
}
