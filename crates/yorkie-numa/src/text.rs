//! ASCII text as bytes, for the sysfs files this crate reads and the index
//! lists it renders.
//!
//! sysfs is ASCII, so a file read from it is a `Vec<u8>` and the index lists in
//! it are scanned digit by digit. This layer sits below the one that owns the
//! engine's text writer, so it carries the few routines it needs itself rather
//! than reaching up for them.

use std::path::Path;

/// The widest decimal a `usize` spells.
const MAX_DIGITS: usize = 20;

/// One index rendered as ASCII decimal digits, held where a caller can borrow
/// them for as long as it takes to write them out.
pub struct Decimal {
    bytes: [u8; MAX_DIGITS],
    start: usize,
}

impl Decimal {
    /// `value`'s digits.
    pub fn of(value: usize) -> Self {
        let mut bytes = [0u8; MAX_DIGITS];
        let mut start = MAX_DIGITS;
        let mut v = value;
        loop {
            start -= 1;
            bytes[start] = b'0' + (v % 10) as u8;
            v /= 10;
            if v == 0 {
                break;
            }
        }
        Self { bytes, start }
    }

    /// The digits.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[self.start..]
    }
}

/// The unsigned decimal `text` spells, or `None` when it is empty, carries a
/// byte that is not a digit, or names a value past `usize`.
pub fn atoi_usize(text: &[u8]) -> Option<usize> {
    if text.is_empty() {
        return None;
    }
    let mut value: usize = 0;
    for &byte in text {
        if !byte.is_ascii_digit() {
            return None;
        }
        value = value
            .checked_mul(10)?
            .checked_add(usize::from(byte - b'0'))?;
    }
    Some(value)
}

/// `bytes` as one path component.
///
/// A path's bytes are its own on Unix, which is the only platform whose sysfs
/// tree this reads.
#[cfg(unix)]
pub fn path_segment(bytes: &[u8]) -> Option<&Path> {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt as _;

    Some(Path::new(OsStr::from_bytes(bytes)))
}

/// Off Unix there is no way to name a path by its bytes, and no sysfs tree to
/// name: every reader treats `None` as a file it cannot read, which is the
/// answer this platform has.
#[cfg(not(unix))]
pub fn path_segment(_bytes: &[u8]) -> Option<&Path> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimals_cover_zero_and_the_widest_value() {
        assert_eq!(Decimal::of(0).as_bytes(), b"0");
        assert_eq!(Decimal::of(7).as_bytes(), b"7");
        assert_eq!(Decimal::of(1023).as_bytes(), b"1023");
        assert_eq!(
            Decimal::of(usize::MAX).as_bytes(),
            usize::MAX.to_string().as_bytes()
        );
    }

    #[test]
    fn indices_parse_from_digits_alone() {
        assert_eq!(atoi_usize(b"0"), Some(0));
        assert_eq!(atoi_usize(b"1023"), Some(1023));
        assert_eq!(atoi_usize(b""), None);
        assert_eq!(atoi_usize(b"12a"), None);
        assert_eq!(atoi_usize(b"-1"), None);
        assert_eq!(atoi_usize(b" 1"), None);
    }
}
