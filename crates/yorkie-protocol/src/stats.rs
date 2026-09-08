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

use core::fmt::{self, Write};

/// What every line starts with.
const PREFIX: &str = "info string stats";

/// The longest line [`render`] can produce: the prefix, then every item at its
/// widest. `alloc` is a `u64`, whose decimal form runs to 20 digits.
const MAX_LINE: usize = PREFIX.len() + " alloc=".len() + 20;

/// The stack buffer a line is composed in.
pub(crate) struct StatsBuf {
    bytes: [u8; MAX_LINE],
    len: usize,
}

impl StatsBuf {
    pub(crate) const fn new() -> Self {
        Self {
            bytes: [0; MAX_LINE],
            len: 0,
        }
    }

    fn as_str(&self) -> &str {
        // Everything written went through `write_str`, which appends whole
        // `&str`s, so the filled prefix is valid UTF-8 by construction.
        core::str::from_utf8(&self.bytes[..self.len]).expect("composed of &str fragments")
    }
}

impl Write for StatsBuf {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let end = self.len + s.len();
        if end > self.bytes.len() {
            return Err(fmt::Error);
        }
        self.bytes[self.len..end].copy_from_slice(s.as_bytes());
        self.len = end;
        Ok(())
    }
}

/// Compose the statistics line for one reply into `buf`, or `None` when every
/// statistic is zero and so no line is written.
///
/// [`MAX_LINE`] is the widest line the items can make, so nothing here can run
/// out of room — and a truncated line would be worse than no line, so the
/// impossible case panics rather than being discarded.
pub(crate) fn render(buf: &mut StatsBuf, alloc: u64) -> Option<&str> {
    buf.len = 0;
    buf.write_str(PREFIX).expect("the prefix fits");
    let mut carried = false;
    if alloc != 0 {
        write!(buf, " alloc={alloc}").expect("the widest item fits");
        carried = true;
    }
    if carried { Some(buf.as_str()) } else { None }
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
        assert_eq!(render(&mut buf, 1), Some("info string stats alloc=1"));
        assert_eq!(
            render(&mut buf, 40_321),
            Some("info string stats alloc=40321")
        );
    }

    #[test]
    fn the_widest_line_fits_the_buffer() {
        let mut buf = StatsBuf::new();
        let line = render(&mut buf, u64::MAX).expect("a non-zero count is reported");
        assert_eq!(line, format!("info string stats alloc={}", u64::MAX));
        assert_eq!(line.len(), MAX_LINE);
    }

    #[test]
    fn a_buffer_is_reusable_and_carries_nothing_over() {
        let mut buf = StatsBuf::new();
        assert_eq!(
            render(&mut buf, 123_456),
            Some("info string stats alloc=123456")
        );
        assert_eq!(render(&mut buf, 7), Some("info string stats alloc=7"));
        assert_eq!(render(&mut buf, 0), None);
    }
}
