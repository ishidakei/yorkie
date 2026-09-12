//! The payload of a `bestmove` line — the move the engine plays, and the move
//! it expects in reply.
//!
//! A reply is the last thing a search does, and what it costs is measured
//! against the search it ends, so the text is composed in a stack buffer sized
//! for the widest payload rather than in a `String`. The two token replies,
//! `resign` and `win`, are constants and need no buffer at all.

use yorkie_state::{MAX_USI_MOVE_LEN, Move, TextWriter, UsiMoveBuf};

/// What sits between the two moves of a pondering reply.
const PONDER: &[u8] = b" ponder ";

/// The longest payload: both moves at their widest, with the keyword between.
const MAX_PAYLOAD: usize = MAX_USI_MOVE_LEN + PONDER.len() + MAX_USI_MOVE_LEN;

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

    /// Compose `<mv>`, or `<mv> ponder <mv>` when the reply names a move to
    /// ponder on, and return the text.
    pub(crate) fn compose(&mut self, best: Move, ponder: Option<Move>) -> &[u8] {
        let mut mv = UsiMoveBuf::new();
        let mut out = TextWriter::new(&mut self.bytes);
        out.bytes(mv.format(best));
        if let Some(p) = ponder {
            out.bytes(PONDER);
            out.bytes(mv.format(p));
        }
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

    #[test]
    fn a_plain_reply_is_the_move_alone() {
        let m = mv((6, 6), (6, 5));
        let mut buf = BestmoveBuf::new();
        let mut expected = UsiMoveBuf::new();
        assert_eq!(buf.compose(m, None), expected.format(m));
    }

    #[test]
    fn a_pondering_reply_names_both_moves() {
        let (best, ponder) = (mv((6, 6), (6, 5)), mv((2, 2), (2, 3)));
        let mut buf = BestmoveBuf::new();
        assert_eq!(buf.compose(best, Some(ponder)), b"7g7f ponder 3c3d");
    }

    #[test]
    fn the_widest_payload_fits_the_buffer() {
        // Both moves promoting, which is the one thing that widens a move.
        let promoting = Move::make_promote(
            Square::new(0, 0).unwrap(),
            Square::new(8, 8).unwrap(),
            Piece::new(PieceKind::Pawn, Color::Black),
        );
        let mut buf = BestmoveBuf::new();
        let payload = buf.compose(promoting, Some(promoting));
        assert_eq!(payload, b"1a9i+ ponder 1a9i+");
        assert_eq!(payload.len(), MAX_PAYLOAD);
    }

    #[test]
    fn a_reused_buffer_carries_nothing_over() {
        let (best, ponder) = (mv((6, 6), (6, 5)), mv((2, 2), (2, 3)));
        let mut buf = BestmoveBuf::new();
        assert_eq!(buf.compose(best, Some(ponder)), b"7g7f ponder 3c3d");
        assert_eq!(buf.compose(ponder, None), b"3c3d");
    }
}
