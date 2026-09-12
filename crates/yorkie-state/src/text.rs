//! ASCII text as bytes: the small reading and writing routines every layer
//! above this one composes its text from.
//!
//! Every byte this engine exchanges with a host or reads from a file is ASCII by
//! specification — USI commands, SFEN, USI moves, the evaluation file's header,
//! the opening book, sysfs — so text here is `&[u8]` and a fixed byte buffer.
//! Nothing is validated as UTF-8 on the way in and nothing is a `str` on the way
//! out, which is also why a line carrying bytes that are not valid UTF-8 cannot
//! end a session.
//!
//! Numbers are read and written digit by digit rather than through the
//! formatting machinery, so composing a line reaches no allocator and pulls in
//! no formatter.

/// The widest decimal a `u64` spells, which is also the widest an `i64` spells
/// without its sign.
const MAX_U64_DIGITS: usize = 20;

/// A fixed-capacity buffer ASCII text is composed into.
///
/// The buffer is the caller's — an array on its stack, or a field of something
/// it already holds — so composing a line allocates nothing. Writing past the
/// capacity keeps the prefix and drops the rest: a diagnostic quoting a token as
/// long as the whole command line is worth truncating, and a line the engine
/// owes a host is composed into a buffer its widest form provably fits.
pub struct TextWriter<'a> {
    buf: &'a mut [u8],
    len: usize,
    /// Whether a write ran out of room, so a caller that must not truncate can
    /// hold itself to that.
    overflowed: bool,
}

impl<'a> TextWriter<'a> {
    /// A writer filling `buf` from its start.
    pub fn new(buf: &'a mut [u8]) -> Self {
        Self {
            buf,
            len: 0,
            overflowed: false,
        }
    }

    /// Drop everything written so far, keeping the room.
    pub fn clear(&mut self) {
        self.len = 0;
        self.overflowed = false;
    }

    /// The bytes written so far.
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf[..self.len]
    }

    /// How many bytes are written.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether nothing is written yet.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether a write was dropped for want of room, so a caller that must not
    /// truncate can assert it did not.
    pub fn overflowed(&self) -> bool {
        self.overflowed
    }

    /// Append `bytes`.
    pub fn bytes(&mut self, bytes: &[u8]) -> &mut Self {
        let room = self.buf.len() - self.len;
        let take = bytes.len().min(room);
        if take != bytes.len() {
            self.overflowed = true;
        }
        self.buf[self.len..self.len + take].copy_from_slice(&bytes[..take]);
        self.len += take;
        self
    }

    /// Append one byte.
    pub fn byte(&mut self, byte: u8) -> &mut Self {
        self.bytes(&[byte])
    }

    /// Append `value` as ASCII decimal digits.
    pub fn u64(&mut self, value: u64) -> &mut Self {
        let mut digits = [0u8; MAX_U64_DIGITS];
        self.bytes(decimal(value, &mut digits))
    }

    /// Append `value` as ASCII decimal digits, preceded by `-` when negative.
    pub fn i64(&mut self, value: i64) -> &mut Self {
        if value < 0 {
            self.byte(b'-');
        }
        self.u64(value.unsigned_abs())
    }

    /// Append `value` as exactly `width` ASCII decimal digits, zero-padded on
    /// the left; a value too wide for the field keeps all of its digits.
    pub fn u64_padded(&mut self, value: u64, width: usize) -> &mut Self {
        let mut digits = [0u8; MAX_U64_DIGITS];
        let text = decimal(value, &mut digits);
        for _ in text.len()..width {
            self.byte(b'0');
        }
        self.bytes(text)
    }

    /// Append `value` as lower-case hexadecimal digits, zero-padded on the left
    /// to `width`.
    pub fn hex_padded(&mut self, value: u64, width: usize) -> &mut Self {
        let mut digits = [0u8; 16];
        let mut i = digits.len();
        let mut v = value;
        loop {
            i -= 1;
            digits[i] = HEX_DIGITS[(v & 0xF) as usize];
            v >>= 4;
            if v == 0 {
                break;
            }
        }
        let text = &digits[i..];
        for _ in text.len()..width {
            self.byte(b'0');
        }
        self.bytes(text)
    }

    /// Append one byte the way a diagnostic quotes an input byte: `'c'` for a
    /// printable one, `0xNN` for anything else, so a line the engine writes
    /// stays ASCII whatever arrived.
    pub fn quoted_byte(&mut self, byte: u8) -> &mut Self {
        if is_printable(byte) {
            self.byte(b'\'').byte(byte).byte(b'\'')
        } else {
            self.bytes(b"0x").hex_padded(u64::from(byte), 2)
        }
    }

    /// Append a filesystem path's own bytes.
    #[cfg(unix)]
    pub fn path(&mut self, path: &std::path::Path) -> &mut Self {
        use std::os::unix::ffi::OsStrExt as _;

        self.bytes(path.as_os_str().as_bytes())
    }
}

/// The lower-case hexadecimal digits, in value order.
const HEX_DIGITS: [u8; 16] = *b"0123456789abcdef";

/// Whether `byte` is one a USI line may carry inside a token: printable ASCII,
/// which is every byte the protocol, SFEN and USI move notation are spelled in.
pub const fn is_printable(byte: u8) -> bool {
    byte.is_ascii_graphic()
}

/// Whether `byte` separates two tokens of a command line.
pub const fn is_token_separator(byte: u8) -> bool {
    byte == b' ' || byte == b'\t'
}

/// `value`'s ASCII decimal digits, written into the tail of `digits`.
fn decimal(value: u64, digits: &mut [u8; MAX_U64_DIGITS]) -> &[u8] {
    let mut i = digits.len();
    let mut v = value;
    loop {
        i -= 1;
        digits[i] = b'0' + (v % 10) as u8;
        v /= 10;
        if v == 0 {
            break;
        }
    }
    &digits[i..]
}

/// The unsigned decimal `text` spells, or `None` when it is empty, carries a
/// byte that is not a digit, or names a value past `u64`.
pub fn atoi_u64(text: &[u8]) -> Option<u64> {
    if text.is_empty() {
        return None;
    }
    let mut value: u64 = 0;
    for &byte in text {
        if !byte.is_ascii_digit() {
            return None;
        }
        value = value.checked_mul(10)?.checked_add(u64::from(byte - b'0'))?;
    }
    Some(value)
}

/// The signed decimal `text` spells, with an optional leading `-` or `+`.
pub fn atoi_i64(text: &[u8]) -> Option<i64> {
    let (negative, digits) = match text.split_first() {
        Some((b'-', rest)) => (true, rest),
        Some((b'+', rest)) => (false, rest),
        _ => (false, text),
    };
    let magnitude = atoi_u64(digits)?;
    if negative {
        if magnitude > 1 << 63 {
            return None;
        }
        Some((magnitude as i64).wrapping_neg())
    } else {
        i64::try_from(magnitude).ok()
    }
}

/// [`atoi_u64`] narrowed to `u32`.
pub fn atoi_u32(text: &[u8]) -> Option<u32> {
    u32::try_from(atoi_u64(text)?).ok()
}

/// [`atoi_u64`] narrowed to `u16`.
pub fn atoi_u16(text: &[u8]) -> Option<u16> {
    u16::try_from(atoi_u64(text)?).ok()
}

/// [`atoi_u64`] narrowed to `usize`.
pub fn atoi_usize(text: &[u8]) -> Option<usize> {
    usize::try_from(atoi_u64(text)?).ok()
}

/// The leading token of `text` and what follows it: what precedes the first run
/// of token separators, and what follows that run. Both halves are empty once
/// nothing is left, and the tail keeps its own inner spacing, so a caller can
/// hand it on whole.
pub fn split_token(text: &[u8]) -> (&[u8], &[u8]) {
    match text.iter().position(|&b| is_token_separator(b)) {
        Some(i) => (&text[..i], trim_leading_separators(&text[i..])),
        None => (text, &text[text.len()..]),
    }
}

/// `text` without its leading token separators.
fn trim_leading_separators(text: &[u8]) -> &[u8] {
    let start = text
        .iter()
        .position(|&b| !is_token_separator(b))
        .unwrap_or(text.len());
    &text[start..]
}

/// `text` without leading or trailing ASCII whitespace.
pub fn trim_ascii_whitespace(text: &[u8]) -> &[u8] {
    let start = text
        .iter()
        .position(|&b| !b.is_ascii_whitespace())
        .unwrap_or(text.len());
    let end = text
        .iter()
        .rposition(|&b| !b.is_ascii_whitespace())
        .map_or(start, |i| i + 1);
    &text[start..end]
}

/// The tokens of `text`, separated by runs of ASCII whitespace and skipping
/// empty runs — what `split_whitespace` accepted for ASCII input.
pub fn tokens(text: &[u8]) -> impl Iterator<Item = &[u8]> {
    text.split(|b| b.is_ascii_whitespace())
        .filter(|token| !token.is_empty())
}

/// The text forms this crate's own unit tests read through.
///
/// Fixtures and assertion messages are written as text, and these turn one into
/// the bytes the engine speaks — and back. Nothing the engine runs goes through
/// them, which is why they exist only in a test build.
#[cfg(test)]
pub(crate) mod test_text {
    use crate::move_::{Move, UsiMoveBuf};
    use crate::position::Position;
    use crate::sfen::{SfenBuf, SfenError, format_sfen, parse_sfen};

    /// [`parse_sfen`] for an SFEN written as text — or, so one helper covers
    /// both, already as bytes.
    pub(crate) fn parse_sfen_str(sfen: impl AsRef<[u8]>) -> Result<Position, SfenError> {
        parse_sfen(sfen.as_ref())
    }

    /// A position's SFEN as an owned string.
    pub(crate) fn sfen_string(pos: &Position) -> String {
        let mut buf = SfenBuf::new();
        String::from_utf8(format_sfen(pos, &mut buf).to_vec()).expect("an SFEN is ASCII")
    }

    /// A move's USI text as an owned string.
    pub(crate) fn format_usi_move(m: Move) -> String {
        let mut buf = UsiMoveBuf::new();
        String::from_utf8(buf.format(m).to_vec()).expect("a USI move is ASCII")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn written(compose: impl FnOnce(&mut TextWriter<'_>)) -> String {
        let mut bytes = [0u8; 128];
        let mut w = TextWriter::new(&mut bytes);
        compose(&mut w);
        String::from_utf8(w.as_bytes().to_vec()).expect("ascii")
    }

    #[test]
    fn decimals_cover_zero_and_the_widest_value() {
        assert_eq!(
            written(|w| {
                w.u64(0);
            }),
            "0"
        );
        assert_eq!(
            written(|w| {
                w.u64(u64::MAX);
            }),
            "18446744073709551615"
        );
        assert_eq!(
            written(|w| {
                w.i64(-1);
            }),
            "-1"
        );
        assert_eq!(
            written(|w| {
                w.i64(i64::MIN);
            }),
            "-9223372036854775808"
        );
        assert_eq!(
            written(|w| {
                w.i64(0);
            }),
            "0"
        );
    }

    #[test]
    fn padded_fields_keep_every_digit_of_a_wide_value() {
        assert_eq!(
            written(|w| {
                w.u64_padded(7, 3);
            }),
            "007"
        );
        assert_eq!(
            written(|w| {
                w.u64_padded(1234, 3);
            }),
            "1234"
        );
        assert_eq!(
            written(|w| {
                w.hex_padded(0x82, 2);
            }),
            "82"
        );
        assert_eq!(
            written(|w| {
                w.hex_padded(0xf, 4);
            }),
            "000f"
        );
    }

    #[test]
    fn an_input_byte_is_quoted_as_ascii_whatever_it_is() {
        assert_eq!(
            written(|w| {
                w.quoted_byte(b'G');
            }),
            "'G'"
        );
        assert_eq!(
            written(|w| {
                w.quoted_byte(0x82);
            }),
            "0x82"
        );
        assert_eq!(
            written(|w| {
                w.quoted_byte(b' ');
            }),
            "0x20"
        );
    }

    #[test]
    fn a_full_buffer_keeps_the_prefix_and_reports_the_loss() {
        let mut bytes = [0u8; 4];
        let mut w = TextWriter::new(&mut bytes);
        w.bytes(b"abc");
        assert!(!w.overflowed());
        w.bytes(b"de");
        assert_eq!(w.as_bytes(), b"abcd");
        assert!(w.overflowed());
    }

    #[test]
    fn integers_parse_from_digits_alone() {
        assert_eq!(atoi_u64(b"0"), Some(0));
        assert_eq!(atoi_u64(b"18446744073709551615"), Some(u64::MAX));
        assert_eq!(atoi_u64(b"18446744073709551616"), None);
        assert_eq!(atoi_u64(b""), None);
        assert_eq!(atoi_u64(b"12a"), None);
        assert_eq!(atoi_u64(b"-1"), None);
        assert_eq!(atoi_u64(b"\x82\xa0"), None);
        assert_eq!(atoi_i64(b"-1"), Some(-1));
        assert_eq!(atoi_i64(b"+12"), Some(12));
        assert_eq!(atoi_i64(b"9223372036854775807"), Some(i64::MAX));
        assert_eq!(atoi_i64(b"-9223372036854775808"), Some(i64::MIN));
        assert_eq!(atoi_i64(b"9223372036854775808"), None);
        assert_eq!(atoi_i64(b"-9223372036854775809"), None);
        assert_eq!(atoi_u16(b"65536"), None);
        assert_eq!(atoi_u32(b"4294967295"), Some(u32::MAX));
    }

    #[test]
    fn a_token_splits_off_its_run_of_separators() {
        assert_eq!(split_token(b"usi"), (&b"usi"[..], &b""[..]));
        assert_eq!(
            split_token(b"position  startpos moves"),
            (&b"position"[..], &b"startpos moves"[..])
        );
        assert_eq!(split_token(b""), (&b""[..], &b""[..]));
        assert_eq!(split_token(b"go\tponder"), (&b"go"[..], &b"ponder"[..]));
    }

    #[test]
    fn whitespace_trims_from_both_ends() {
        assert_eq!(trim_ascii_whitespace(b"  usi \r\n"), b"usi");
        assert_eq!(trim_ascii_whitespace(b"   "), b"");
        assert_eq!(trim_ascii_whitespace(b""), b"");
    }

    #[test]
    fn tokens_skip_empty_runs() {
        let split: Vec<&[u8]> = tokens(b"  7g7f\t\t8c8d  ").collect();
        assert_eq!(split, vec![&b"7g7f"[..], &b"8c8d"[..]]);
        assert_eq!(tokens(b"   ").count(), 0);
    }
}
