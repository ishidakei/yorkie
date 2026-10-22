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
//! that interval did on every thread (see [`SearchCounts`]). A `verbose3` build
//! adds three more: the value marks of the score the reply rests on.
//!
//! A `verbose3` build's `bench` also writes one line after its summary, in the
//! same format, carrying the search counts summed over every position of the
//! run (see [`take_bench_counts`]).
//!
//! What the line reports is the interval it ends, so composing it must not
//! disturb what it measures. It is rendered into a stack buffer sized for the
//! longest line it can produce and handed to the writer as those bytes, so
//! nothing on this path reaches the heap.

use std::sync::atomic::{AtomicU64, Ordering};

use yorkie_search::WorkerResult;
use yorkie_state::TextWriter;
#[cfg(feature = "verbose3")]
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

/// The longest line [`render`] can produce: the prefix, then every item at its
/// widest.
#[cfg(not(feature = "verbose3"))]
const MAX_LINE: usize = PREFIX.len() + COUNT_ITEMS_MAX;

/// The longest line [`render`] can produce, with the three mark items of a
/// `verbose3` build after the counted ones. Each is written only when its mark
/// is set, so the widest line is the one that carries all three.
#[cfg(feature = "verbose3")]
const MAX_LINE: usize = PREFIX.len()
    + COUNT_ITEMS_MAX
    + MARK_ITEMS[0].len()
    + MARK_ITEMS[1].len()
    + MARK_ITEMS[2].len();

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
}

/// [`SearchCounts`] as they stand, one sum per item.
struct Tally {
    nodes: AtomicU64,
    tt_updates: AtomicU64,
    eval_dups: AtomicU64,
    evals: AtomicU64,
    eval_tt_hits: AtomicU64,
}

impl Tally {
    const fn new() -> Self {
        Self {
            nodes: AtomicU64::new(0),
            tt_updates: AtomicU64::new(0),
            eval_dups: AtomicU64::new(0),
            evals: AtomicU64::new(0),
            eval_tt_hits: AtomicU64::new(0),
        }
    }

    fn add(&self, r: &WorkerResult) {
        self.nodes.fetch_add(r.nodes, Ordering::Relaxed);
        self.tt_updates.fetch_add(r.tt_updates, Ordering::Relaxed);
        self.eval_dups.fetch_add(r.eval_dups, Ordering::Relaxed);
        self.evals.fetch_add(r.evals, Ordering::Relaxed);
        self.eval_tt_hits
            .fetch_add(r.eval_tt_hits, Ordering::Relaxed);
    }

    fn take(&self) -> SearchCounts {
        SearchCounts {
            nodes: self.nodes.swap(0, Ordering::Relaxed),
            tt_update: self.tt_updates.swap(0, Ordering::Relaxed),
            eval_dup: self.eval_dups.swap(0, Ordering::Relaxed),
            evals: self.evals.swap(0, Ordering::Relaxed),
            eval_tt_hits: self.eval_tt_hits.swap(0, Ordering::Relaxed),
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

/// Add the counts of finished workers to the interval's tally.
pub(crate) fn add_search<'a>(results: impl IntoIterator<Item = &'a WorkerResult>) {
    for r in results {
        INTERVAL.add(r);
        #[cfg(feature = "verbose3")]
        BENCH_RUN.add(r);
    }
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
/// every other item on the line.
///
/// [`MAX_LINE`] is the widest line the items can make, so nothing here can run
/// out of room — and a truncated line would be worse than no line, so the
/// impossible case is asserted rather than discarded.
pub(crate) fn render(
    buf: &mut StatsBuf,
    alloc: u64,
    search: SearchCounts,
    #[cfg(feature = "verbose3")] marks: ValueMarks,
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
    #[cfg(feature = "verbose3")]
    for (item, set) in MARK_ITEMS
        .iter()
        .zip([marks.path_dep, marks.decl_rule, marks.move_limit])
    {
        if set {
            out.bytes(item);
            carried = true;
        }
    }
    debug_assert!(!out.overflowed(), "the widest line fits MAX_LINE");
    let len = out.len();
    carried.then_some(&buf.bytes[..len])
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

    #[cfg(not(feature = "verbose3"))]
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

        #[test]
        fn the_widest_line_fits_the_buffer() {
            let mut buf = StatsBuf::new();
            let line = render(&mut buf, u64::MAX, all(u64::MAX), marks(true, true, true))
                .expect("a marked reply is reported");
            assert_eq!(
                line,
                &b"info string stats alloc=18446744073709551615 nodes=18446744073709551615 \
                   tt_update=18446744073709551615 eval_dup=18446744073709551615 \
                   evals=18446744073709551615 eval_tt_hits=18446744073709551615 \
                   pathdep=1 declrule=1 movelimit=1"[..]
            );
            assert_eq!(line.len(), MAX_LINE);
        }
    }
}
