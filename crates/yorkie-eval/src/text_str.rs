#![cfg(test)]

use yorkie_state::{Move, Position, SfenError, UsiMoveBuf, UsiMoveParseError};

/// [`yorkie_state::parse_sfen`] for an SFEN written as text.
pub(crate) fn parse_sfen(sfen: &str) -> Result<Position, SfenError> {
    yorkie_state::parse_sfen(sfen.as_bytes())
}

/// [`yorkie_state::parse_usi_move`] for a move written as text.
pub(crate) fn parse_usi_move(usi: &str, pos: &Position) -> Result<Move, UsiMoveParseError> {
    yorkie_state::parse_usi_move(usi.as_bytes(), pos)
}

/// A move's USI text as an owned string.
pub(crate) fn format_usi_move(m: Move) -> String {
    let mut buf = UsiMoveBuf::new();
    String::from_utf8(buf.format(m).to_vec()).expect("a USI move is ASCII")
}
