//! When each worker of a search finished each root iteration: the per-worker
//! record a `dupstats` build keeps of its iterative deepening.
//!
//! A worker writes one [`Iteration`] per root iteration it completes, and one
//! more for the iteration it is inside when the search stops it. Each carries
//! the time since the search started — one clock for every worker of a search,
//! read once per iteration — the worker's own node count at that moment, how
//! often the iteration was searched again at the root, the best move it ended
//! on, how its nodes split between the first root move and the rest — and
//! over the first [`ROOT_SPLIT`] distinct root moves it searched, move by move —
//! and whether the transposition table already held the root deep enough when the
//! iteration started, and who had written it there.
//!
//! The records live in a fixed array indexed by depth, which the worker owns
//! and writes once per iteration, so nothing is allocated while a search runs
//! and nothing is shared between workers. Nothing here is read by the search:
//! its decisions, node counts and best moves are those of a build without the
//! feature.
//!
//! A `verbose3` build also has [`FirstFinish`]: the latch a measurement uses to
//! end a fixed-depth search when the first of its workers, rather than the main
//! one, has finished.

#[cfg(feature = "verbose3")]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use yorkie_state::Move;

use crate::dupstats::WriterId;

/// How many depths a timeline has room for: every depth a root iteration can
/// reach.
pub const DEPTHS: usize = 256;

/// How many distinct root moves an iteration's [`RootSplit`] names; the nodes
/// of any further one are summed in [`RootSplit::rest`].
pub const ROOT_SPLIT: usize = 8;

/// An iteration's nodes, root move by root move: the first [`ROOT_SPLIT`]
/// distinct root moves the iteration searched, in the order it first searched
/// each, with the nodes under every search of it (an aspiration re-search
/// searches a move again), and the nodes under every other root move together.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RootSplit {
    /// The moves, in the order first searched; `None` past the last one.
    pub moves: [Option<Move>; ROOT_SPLIT],
    /// The nodes under each of [`Self::moves`].
    pub nodes: [u64; ROOT_SPLIT],
    /// The nodes under every root move past the first [`ROOT_SPLIT`].
    pub rest: u64,
}

impl RootSplit {
    const EMPTY: Self = Self {
        moves: [None; ROOT_SPLIT],
        nodes: [0; ROOT_SPLIT],
        rest: 0,
    };

    fn add(&mut self, mv: Move, nodes: u64) {
        for (slot, n) in self.moves.iter_mut().zip(&mut self.nodes) {
            match slot {
                Some(m) if *m == mv => {
                    *n += nodes;
                    return;
                }
                Some(_) => {}
                None => {
                    *slot = Some(mv);
                    *n = nodes;
                    return;
                }
            }
        }
        self.rest += nodes;
    }

    /// The named moves with their nodes, in the order first searched.
    pub fn named(&self) -> impl Iterator<Item = (Move, u64)> + '_ {
        self.moves
            .iter()
            .zip(&self.nodes)
            .map_while(|(m, n)| m.map(|m| (m, *n)))
    }
}

/// Who had stored the root at the iteration's depth or deeper when the
/// iteration started.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RootEntry {
    /// The table did not hold the root at that depth: another position, a
    /// shallower entry, or nothing.
    Absent,
    /// This worker, in an earlier iteration or an earlier root search of this
    /// one.
    ThisWorker,
    /// Another worker of the same search.
    OtherWorker,
    /// Any worker of an earlier search.
    EarlierSearch,
    /// Not known: the writer side table lost the record to another position.
    Unknown,
}

impl RootEntry {
    /// The entry held the root at `found` depth (`None`: it held something
    /// else), last written by `writer` (`None`: not known); the iteration
    /// searches to `depth` and is made by `me`.
    pub fn of(found: Option<i32>, depth: i32, writer: Option<WriterId>, me: WriterId) -> Self {
        match found {
            Some(d) if d >= depth => match writer {
                None => RootEntry::Unknown,
                Some(w) if w.generation != me.generation => RootEntry::EarlierSearch,
                Some(w) if w.worker != me.worker => RootEntry::OtherWorker,
                Some(_) => RootEntry::ThisWorker,
            },
            _ => RootEntry::Absent,
        }
    }

    /// The word an `iter` line writes for it.
    pub const fn key(self) -> &'static str {
        match self {
            RootEntry::Absent => "none",
            RootEntry::ThisWorker => "self",
            RootEntry::OtherWorker => "other",
            RootEntry::EarlierSearch => "prev",
            RootEntry::Unknown => "unknown",
        }
    }
}

/// One root iteration of one worker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Iteration {
    /// The root depth; zero for a depth the worker never reached.
    pub depth: i32,
    /// Whether the iteration ran to its end; `false` for the one the search
    /// stopped the worker inside.
    pub completed: bool,
    /// Microseconds from the start of the search to the end of the iteration,
    /// or to the moment the worker saw the stop.
    pub micros: u64,
    /// The worker's own node count at that moment.
    pub nodes: u64,
    /// Aspiration searches of the root that failed low and were searched again.
    pub fail_lows: u32,
    /// Aspiration searches of the root that failed high and were searched again.
    pub fail_highs: u32,
    /// The root move first in the worker's order at the end of the iteration.
    pub best: Option<Move>,
    /// That move's score, in the search's internal units.
    pub value: i32,
    /// Whether that move is not the one the previous completed iteration ended
    /// on. The first iteration has none to differ from.
    pub best_changed: bool,
    /// Nodes searched under the first root move of each root search in the
    /// iteration.
    pub first_move_nodes: u64,
    /// Nodes searched under every other root move.
    pub other_move_nodes: u64,
    /// The same nodes, root move by root move.
    pub split: RootSplit,
    /// Who had stored the root at this depth or deeper when it started.
    pub root_entry: RootEntry,
}

impl Iteration {
    const EMPTY: Self = Self {
        depth: 0,
        completed: false,
        micros: 0,
        nodes: 0,
        fail_lows: 0,
        fail_highs: 0,
        best: None,
        value: 0,
        best_changed: false,
        first_move_nodes: 0,
        other_move_nodes: 0,
        split: RootSplit::EMPTY,
        root_entry: RootEntry::Absent,
    };
}

/// One worker's iterations over one search.
#[derive(Clone, Debug)]
pub struct Timeline {
    /// The worker's index in its pool.
    pub worker: u8,
    /// Indexed by depth; an entry of depth zero was not reached.
    iterations: [Iteration; DEPTHS],
    /// The iteration in progress.
    current: Iteration,
    /// The best move of the last completed iteration.
    previous_best: Option<Move>,
    /// When the search started, for every worker of it.
    start: Instant,
}

impl Timeline {
    /// An empty timeline of worker 0, its search started now.
    pub fn new() -> Self {
        Self {
            worker: 0,
            iterations: [Iteration::EMPTY; DEPTHS],
            current: Iteration::EMPTY,
            previous_best: None,
            start: Instant::now(),
        }
    }

    /// Forget every iteration, for a search of `worker` that started at
    /// `start`.
    pub fn reset(&mut self, worker: u8, start: Instant) {
        self.worker = worker;
        self.iterations.fill(Iteration::EMPTY);
        self.current = Iteration::EMPTY;
        self.previous_best = None;
        self.start = start;
    }

    /// An iteration to `depth` starts, the root found as `root_entry` says.
    pub fn begin(&mut self, depth: i32, root_entry: RootEntry) {
        self.current = Iteration {
            depth,
            root_entry,
            ..Iteration::EMPTY
        };
    }

    /// The root search failed low and is searched again.
    pub fn fail_low(&mut self) {
        self.current.fail_lows += 1;
    }

    /// The root search failed high and is searched again.
    pub fn fail_high(&mut self) {
        self.current.fail_highs += 1;
    }

    /// The root move `mv` was searched, under which `nodes` nodes were; `first`
    /// when it was the first move of its root search.
    pub fn root_move(&mut self, first: bool, mv: Move, nodes: u64) {
        self.current.split.add(mv, nodes);
        if first {
            self.current.first_move_nodes += nodes;
        } else {
            self.current.other_move_nodes += nodes;
        }
    }

    /// The iteration in progress ends — `completed`, or stopped — with the
    /// worker at `nodes` nodes and `best` scored `value` first in its order.
    pub fn end(&mut self, completed: bool, nodes: u64, best: Move, value: i32) {
        let micros = self.start.elapsed().as_micros();
        let mut it = self.current;
        it.completed = completed;
        it.micros = u64::try_from(micros).unwrap_or(u64::MAX);
        it.nodes = nodes;
        it.best = Some(best);
        it.value = value;
        it.best_changed = self.previous_best.is_some_and(|prev| prev != best);
        if completed {
            self.previous_best = Some(best);
        }
        if let Ok(at) = usize::try_from(it.depth)
            && at < DEPTHS
        {
            self.iterations[at] = it;
        }
    }

    /// The iterations the worker reached, by depth.
    pub fn iterations(&self) -> impl Iterator<Item = &Iteration> {
        self.iterations.iter().filter(|it| it.depth != 0)
    }
}

impl Default for Timeline {
    fn default() -> Self {
        Self::new()
    }
}

/// The latch that ends a fixed-depth measurement search when its first worker
/// finishes, rather than its main one.
///
/// Every worker of such a search is handed the latch, and a worker that runs
/// its iterative deepening to its end without being stopped calls
/// [`Self::finish`]. The first to do so raises the search's stop flag, which
/// ends every other worker inside the iteration it is in, and is the worker
/// whose result the search reports. A search not handed the latch ends as it
/// always does: when the main worker finishes, a helper that finished first
/// waiting for it.
#[cfg(feature = "verbose3")]
#[derive(Debug)]
pub struct FirstFinish {
    /// The first worker to finish, or [`Self::NONE`].
    winner: AtomicUsize,
}

#[cfg(feature = "verbose3")]
impl FirstFinish {
    const NONE: usize = usize::MAX;

    /// A latch no worker has finished on.
    pub const fn new() -> Self {
        Self {
            winner: AtomicUsize::new(Self::NONE),
        }
    }

    /// Open the latch again, for the next search.
    pub fn reset(&self) {
        self.winner.store(Self::NONE, Ordering::Relaxed);
    }

    /// Worker `worker` finished: whether it is the first to.
    pub fn finish(&self, worker: usize) -> bool {
        self.winner
            .compare_exchange(Self::NONE, worker, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    /// The first worker to finish, if one has.
    pub fn winner(&self) -> Option<usize> {
        let w = self.winner.load(Ordering::Acquire);
        (w != Self::NONE).then_some(w)
    }
}

#[cfg(feature = "verbose3")]
impl Default for FirstFinish {
    fn default() -> Self {
        Self::new()
    }
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
    fn the_root_entry_counts_only_at_the_iterations_depth_or_deeper() {
        let other = WriterId { worker: 0, ..ME };
        let earlier = WriterId {
            generation: 2,
            ..ME
        };
        let mine_before = WriterId { iteration: 4, ..ME };
        assert_eq!(RootEntry::of(None, 5, Some(ME), ME), RootEntry::Absent);
        assert_eq!(RootEntry::of(Some(4), 5, Some(ME), ME), RootEntry::Absent);
        assert_eq!(
            RootEntry::of(Some(5), 5, Some(other), ME),
            RootEntry::OtherWorker
        );
        assert_eq!(
            RootEntry::of(Some(9), 5, Some(earlier), ME),
            RootEntry::EarlierSearch
        );
        assert_eq!(
            RootEntry::of(Some(5), 5, Some(mine_before), ME),
            RootEntry::ThisWorker
        );
        assert_eq!(RootEntry::of(Some(5), 5, None, ME), RootEntry::Unknown);
    }

    fn mv(n: u32) -> Move {
        Move::from_bits(n).expect("a non-zero move")
    }

    #[test]
    fn an_iteration_is_written_once_at_its_depth() {
        let mut t = Timeline::new();
        t.reset(2, Instant::now());
        t.begin(1, RootEntry::Absent);
        t.root_move(true, mv(7), 10);
        t.root_move(false, mv(8), 5);
        t.root_move(false, mv(9), 7);
        t.end(true, 30, mv(100), 40);
        t.begin(2, RootEntry::ThisWorker);
        t.fail_low();
        t.fail_high();
        t.fail_high();
        t.end(true, 90, mv(101), -12);
        t.begin(3, RootEntry::OtherWorker);
        t.end(false, 95, mv(101), -12);

        let its: Vec<&Iteration> = t.iterations().collect();
        assert_eq!(t.worker, 2);
        assert_eq!(its.len(), 3);
        assert_eq!(
            (
                its[0].depth,
                its[0].first_move_nodes,
                its[0].other_move_nodes
            ),
            (1, 10, 12)
        );
        assert_eq!(
            its[0].split.named().collect::<Vec<_>>(),
            vec![(mv(7), 10), (mv(8), 5), (mv(9), 7)]
        );
        assert_eq!(its[1].split.named().count(), 0);
        assert!(!its[0].best_changed, "the first has nothing to differ from");
        assert_eq!((its[1].fail_lows, its[1].fail_highs), (1, 2));
        assert!(its[1].best_changed);
        assert_eq!(its[1].root_entry, RootEntry::ThisWorker);
        assert!(its[0].completed && its[1].completed && !its[2].completed);
        assert!(!its[2].best_changed);
        assert!(its[0].micros <= its[1].micros && its[1].micros <= its[2].micros);

        t.reset(0, Instant::now());
        assert_eq!(t.iterations().count(), 0);
    }

    #[test]
    fn a_root_move_searched_again_adds_to_its_own_slot_and_the_ninth_goes_to_the_rest() {
        let mut t = Timeline::new();
        t.begin(5, RootEntry::Absent);
        t.root_move(true, mv(1), 100);
        t.root_move(false, mv(2), 3);
        // An aspiration re-search: the first move again.
        t.root_move(true, mv(1), 50);
        for n in 3..=ROOT_SPLIT as u32 + 2 {
            t.root_move(false, mv(n), 1);
        }
        t.end(true, 200, mv(1), 0);
        let it = t.iterations().next().expect("one iteration");
        let named: Vec<_> = it.split.named().collect();
        assert_eq!(named.len(), ROOT_SPLIT);
        assert_eq!(named[0], (mv(1), 150));
        assert_eq!(named[1], (mv(2), 3));
        assert_eq!(it.split.rest, 2);
        assert_eq!(
            named.iter().map(|(_, n)| n).sum::<u64>() + it.split.rest,
            it.first_move_nodes + it.other_move_nodes
        );
    }

    /// One stub worker: how many ticks each of its iterations takes.
    #[cfg(feature = "verbose3")]
    struct Stub {
        ticks: Vec<u32>,
    }

    /// Drive stub workers through a search to `depth` one tick at a time, the
    /// way the coordinator drives real ones: every worker steps its iteration
    /// on each tick until it has finished its depth or seen the stop, and the
    /// search ends on the tick the main worker (worker 0) returns, the stop
    /// raised for the others. With `latch`, a worker that finishes calls it,
    /// and the first to finish raises the stop. Returns the tick the search
    /// ended on, the worker it reports, and each worker's completed depth.
    #[cfg(feature = "verbose3")]
    fn run_stubs(
        workers: &[Stub],
        depth: usize,
        latch: Option<&FirstFinish>,
    ) -> (u32, usize, Vec<usize>) {
        if let Some(l) = latch {
            l.reset();
        }
        let mut done = vec![0usize; workers.len()];
        let mut spent = vec![0u32; workers.len()];
        let mut finished = vec![false; workers.len()];
        let mut stop = false;
        let mut tick = 0;
        loop {
            tick += 1;
            for (w, stub) in workers.iter().enumerate() {
                if finished[w] || stop {
                    continue;
                }
                spent[w] += 1;
                if spent[w] == stub.ticks[done[w]] {
                    done[w] += 1;
                    spent[w] = 0;
                    if done[w] == depth {
                        finished[w] = true;
                        if latch.is_some_and(|l| l.finish(w)) {
                            stop = true;
                        }
                    }
                }
            }
            if finished[0] || stop {
                let chosen = latch.and_then(FirstFinish::winner).unwrap_or(0);
                return (tick, chosen, done);
            }
        }
    }

    #[cfg(feature = "verbose3")]
    #[test]
    fn stopping_at_the_first_worker_ends_no_later_than_at_the_main_one() {
        let latch = FirstFinish::new();
        // The helper is faster: under the main rule it finishes and waits; under
        // the first rule its finish ends the search, the main worker stopped
        // inside its last iteration.
        let fast_helper = [
            Stub {
                ticks: vec![2, 4, 8],
            },
            Stub {
                ticks: vec![1, 2, 4],
            },
        ];
        let (main_end, main_chosen, main_done) = run_stubs(&fast_helper, 3, None);
        let (first_end, first_chosen, first_done) = run_stubs(&fast_helper, 3, Some(&latch));
        assert_eq!((main_end, main_chosen, main_done), (14, 0, vec![3, 3]));
        assert_eq!((first_end, first_chosen, first_done), (7, 1, vec![2, 3]));

        // The main worker is faster: both rules end on its finish.
        let fast_main = [
            Stub {
                ticks: vec![1, 1, 1],
            },
            Stub {
                ticks: vec![2, 2, 2],
            },
        ];
        let main = run_stubs(&fast_main, 3, None);
        let first = run_stubs(&fast_main, 3, Some(&latch));
        assert_eq!(main, (3, 0, vec![3, 1]));
        assert_eq!(first, main);

        for stubs in [&fast_helper[..], &fast_main[..]] {
            assert!(run_stubs(stubs, 3, Some(&latch)).0 <= run_stubs(stubs, 3, None).0);
        }
    }

    #[cfg(feature = "verbose3")]
    #[test]
    fn the_latch_takes_the_first_finish_only() {
        let latch = FirstFinish::new();
        assert_eq!(latch.winner(), None);
        assert!(latch.finish(3));
        assert!(!latch.finish(0));
        assert_eq!(latch.winner(), Some(3));
        latch.reset();
        assert!(latch.finish(0));
        assert_eq!(latch.winner(), Some(0));
    }
}
