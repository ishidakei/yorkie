//! The payload of a `bestmove` line — the move the engine plays, and nothing
//! after it.
//!
//! A reply is the last thing a search does, and what it costs is measured
//! against the search it ends, so the text is composed in a stack buffer sized
//! for the widest payload rather than in a `String`. The two token replies,
//! `resign` and `win`, are constants and need no buffer at all.

use yorkie_state::{MAX_USI_MOVE_LEN, Move, TextWriter, UsiMoveBuf};

/// The longest payload: one move at its widest.
const MAX_PAYLOAD: usize = MAX_USI_MOVE_LEN;

/// The stack buffer a payload is composed in.
pub(crate) struct BestmoveBuf {
    bytes: [u8; MAX_PAYLOAD],
    len: usize,
}

impl BestmoveBuf {
    pub(crate) const fn new() -> Self {
        Self {
            bytes: [0; MAX_PAYLOAD],
            len: 0,
        }
    }

    /// Compose `<mv>` and return the text.
    pub(crate) fn compose(&mut self, best: Move) -> &[u8] {
        let mut mv = UsiMoveBuf::new();
        let mut out = TextWriter::new(&mut self.bytes);
        out.bytes(mv.format(best));
        debug_assert!(!out.overflowed(), "the widest payload fits MAX_PAYLOAD");
        self.len = out.len();
        &self.bytes[..self.len]
    }
}

#[cfg(test)]
mod tests {
    use yorkie_state::{Color, Piece, PieceKind, Square};

    use super::*;

    fn mv(from: (u8, u8), to: (u8, u8)) -> Move {
        Move::make(
            Square::new(from.0, from.1).unwrap(),
            Square::new(to.0, to.1).unwrap(),
            Piece::new(PieceKind::Pawn, Color::Black),
        )
    }

    /// The one thing that widens a move: a promotion marker.
    fn promoting() -> Move {
        Move::make_promote(
            Square::new(0, 0).unwrap(),
            Square::new(8, 8).unwrap(),
            Piece::new(PieceKind::Pawn, Color::Black),
        )
    }

    #[test]
    fn a_reply_is_the_move_alone() {
        let m = mv((6, 6), (6, 5));
        let mut buf = BestmoveBuf::new();
        let mut expected = UsiMoveBuf::new();
        assert_eq!(buf.compose(m), expected.format(m));
    }

    #[test]
    fn the_widest_payload_fits_the_buffer() {
        let mut buf = BestmoveBuf::new();
        let payload = buf.compose(promoting());
        assert_eq!(payload, b"1a9i+");
        assert_eq!(payload.len(), MAX_PAYLOAD);
    }

    #[test]
    fn a_reused_buffer_carries_nothing_over() {
        let mut buf = BestmoveBuf::new();
        assert_eq!(buf.compose(promoting()), b"1a9i+");
        assert_eq!(buf.compose(mv((2, 2), (2, 3))), b"3c3d");
    }
}
