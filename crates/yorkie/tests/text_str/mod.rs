//! The text forms these tests read through.
//!
//! The engine speaks bytes: an SFEN, a USI move and a command line are `&[u8]`
//! everywhere it handles them. Fixtures and assertion messages here are written
//! as text instead, so this is the one place a fixture becomes the bytes a
//! parser takes and a formatted position becomes a string an assertion can
//! print.

#![allow(dead_code)]

use yorkie_state::{Move, Position, SfenBuf, SfenError, UsiMoveBuf};

/// [`yorkie_state::parse_sfen`] for an SFEN written as text — or, so one helper
/// covers both, already as bytes.
pub fn parse_sfen(sfen: impl AsRef<[u8]>) -> Result<Position, SfenError> {
    yorkie_state::parse_sfen(sfen.as_ref())
}

/// [`yorkie_state::parse_usi_move`] for a move written as text, likewise.
pub fn parse_usi_move(
    usi: impl AsRef<[u8]>,
    pos: &Position,
) -> Result<Move, yorkie_state::UsiMoveParseError> {
    yorkie_state::parse_usi_move(usi.as_ref(), pos)
}

/// A position's SFEN as an owned string.
pub fn format_sfen(pos: &Position) -> String {
    let mut buf = SfenBuf::new();
    String::from_utf8(yorkie_state::format_sfen(pos, &mut buf).to_vec()).expect("an SFEN is ASCII")
}

/// A move's USI text as an owned string.
pub fn format_usi_move(m: Move) -> String {
    let mut buf = UsiMoveBuf::new();
    String::from_utf8(buf.format(m).to_vec()).expect("a USI move is ASCII")
}
