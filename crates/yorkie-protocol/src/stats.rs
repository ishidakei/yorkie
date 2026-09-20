//! The per-reply statistics line — `info string stats <key>=<value> …`, written
//! immediately before the `bestmove` that ends a search.
//!
//! One line carries everything a reply reports about itself, as space-separated
//! `key=value` items after a fixed prefix. An item whose value is zero is
//! omitted, and a line left with no item is not written at all: a reply that has
//! nothing to report says nothing rather than saying that nothing happened.
//!
//! `alloc` — the allocations of the interval the reply ends — is the item every
//! build that writes the line has. A `verbose3` build adds three more: the value
//! marks of the score the reply rests on.
//!
//! What the line reports is the interval it ends, so composing it must not
//! disturb what it measures. It is rendered into a stack buffer sized for the
//! longest line it can produce and handed to the writer as those bytes, so
//! nothing on this path reaches the heap.

use yorkie_state::TextWriter;
#[cfg(feature = "verbose3")]
use yorkie_storage::ValueMarks;

/// What every line starts with.
const PREFIX: &[u8] = b"info string stats";

/// The three mark items, in the order they are written. The keys are the ones
/// `tt probe` spells, so a reader needs one vocabulary for the marks of a value
/// in the table and the marks of the value a reply rests on.
#[cfg(feature = "verbose3")]
const MARK_ITEMS: [&[u8]; 3] = [b" pathdep=1", b" declrule=1", b" movelimit=1"];

/// The longest line [`render`] can produce: the prefix, then every item at its
/// widest. `alloc` is a `u64`, whose decimal form runs to 20 digits.
#[cfg(not(feature = "verbose3"))]
const MAX_LINE: usize = PREFIX.len() + " alloc=".len() + 20;

/// The longest line [`render`] can produce, with the three mark items of a
/// `verbose3` build after `alloc`. Each is written only when its mark is set, so
/// the widest line is the one that carries all three.
#[cfg(feature = "verbose3")]
const MAX_LINE: usize = PREFIX.len()
    + " alloc=".len()
    + 20
    + MARK_ITEMS[0].len()
    + MARK_ITEMS[1].len()
    + MARK_ITEMS[2].len();

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
    #[cfg(feature = "verbose3")] marks: ValueMarks,
) -> Option<&[u8]> {
    let mut out = TextWriter::new(&mut buf.bytes);
    out.bytes(PREFIX);
    let mut carried = false;
    if alloc != 0 {
        out.bytes(b" alloc=").u64(alloc);
        carried = true;
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
        render(
            buf,
            alloc,
            #[cfg(feature = "verbose3")]
            ValueMarks::NONE,
        )
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

    #[cfg(not(feature = "verbose3"))]
    #[test]
    fn the_widest_line_fits_the_buffer() {
        let mut buf = StatsBuf::new();
        let line = unmarked(&mut buf, u64::MAX).expect("a non-zero count is reported");
        assert_eq!(line, b"info string stats alloc=18446744073709551615");
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
        /// always in the `tt probe` order.
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
                assert_eq!(render(&mut buf, 9, m), Some(*want), "marks {m:?}");
            }
        }

        /// A mark is an item like any other, so it alone is enough to make a line
        /// a reply with nothing else to report would not have written.
        #[test]
        fn a_mark_alone_writes_the_line() {
            let mut buf = StatsBuf::new();
            assert_eq!(
                render(&mut buf, 0, marks(false, true, false)),
                Some(&b"info string stats declrule=1"[..])
            );
            assert_eq!(render(&mut buf, 0, ValueMarks::NONE), None);
        }

        #[test]
        fn the_widest_line_fits_the_buffer() {
            let mut buf = StatsBuf::new();
            let line = render(&mut buf, u64::MAX, marks(true, true, true))
                .expect("a marked reply is reported");
            assert_eq!(
                line,
                b"info string stats alloc=18446744073709551615 pathdep=1 declrule=1 movelimit=1"
            );
            assert_eq!(line.len(), MAX_LINE);
        }
    }
}
