//! The per-reply statistics line — `info string stats <key>=<value> …`, written
//! immediately before the `bestmove` that ends a search.
//!
//! One line carries every statistic a reply reports, as space-separated
//! `key=value` items after a fixed prefix. An item whose value is zero is
//! omitted, and a line left with no item is not written at all: a reply that has
//! nothing to report says nothing rather than saying that nothing happened.
//!
//! What the line reports is the interval it ends, so composing it must not
//! disturb what it measures. It is rendered into a stack buffer sized for the
//! longest line it can produce and handed to the writer as those bytes, so
//! nothing on this path reaches the heap.

use yorkie_state::TextWriter;

/// What every line starts with.
const PREFIX: &[u8] = b"info string stats";

/// The longest line [`render`] can produce: the prefix, then every item at its
/// widest. `alloc` is a `u64`, whose decimal form runs to 20 digits.
const MAX_LINE: usize = PREFIX.len() + " alloc=".len() + 20;

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
/// [`MAX_LINE`] is the widest line the items can make, so nothing here can run
/// out of room — and a truncated line would be worse than no line, so the
/// impossible case is asserted rather than discarded.
pub(crate) fn render(buf: &mut StatsBuf, alloc: u64) -> Option<&[u8]> {
    let mut out = TextWriter::new(&mut buf.bytes);
    out.bytes(PREFIX);
    let mut carried = false;
    if alloc != 0 {
        out.bytes(b" alloc=").u64(alloc);
        carried = true;
    }
    debug_assert!(!out.overflowed(), "the widest line fits MAX_LINE");
    let len = out.len();
    if carried {
        Some(&buf.bytes[..len])
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_all_zero_reply_writes_no_line() {
        let mut buf = StatsBuf::new();
        assert_eq!(render(&mut buf, 0), None);
    }

    #[test]
    fn a_counted_reply_is_the_prefix_and_the_item() {
        let mut buf = StatsBuf::new();
        assert_eq!(render(&mut buf, 1), Some(&b"info string stats alloc=1"[..]));
        assert_eq!(
            render(&mut buf, 40_321),
            Some(&b"info string stats alloc=40321"[..])
        );
    }

    #[test]
    fn the_widest_line_fits_the_buffer() {
        let mut buf = StatsBuf::new();
        let line = render(&mut buf, u64::MAX).expect("a non-zero count is reported");
        assert_eq!(line, b"info string stats alloc=18446744073709551615");
        assert_eq!(line.len(), MAX_LINE);
    }

    #[test]
    fn a_buffer_is_reusable_and_carries_nothing_over() {
        let mut buf = StatsBuf::new();
        assert_eq!(
            render(&mut buf, 123_456),
            Some(&b"info string stats alloc=123456"[..])
        );
        assert_eq!(render(&mut buf, 7), Some(&b"info string stats alloc=7"[..]));
        assert_eq!(render(&mut buf, 0), None);
    }
}
