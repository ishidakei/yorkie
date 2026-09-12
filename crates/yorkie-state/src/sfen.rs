use crate::color::Color;
use crate::piece::{Piece, PieceKind};
use crate::position::Position;
use crate::square::Square;
use crate::text::{TextWriter, atoi_u16};

pub const STARTPOS_SFEN: &[u8] = b"lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

/// Room the widest SFEN needs: every square occupied by a promoted piece, both
/// hands full, and a five-digit ply.
pub const SFEN_CAPACITY: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SfenError {
    UnexpectedEnd,
    InvalidPiece(u8),
    UnexpectedPromoteMarker,
    NonPromotablePromoted(u8),
    BoardCursorOverflow,
    BoardCursorIncomplete,
    InvalidSideToMove(u8),
    MissingHandPieceAfterCount,
    InvalidHandPiece(u8),
    InvalidPly,
    UnexpectedTrailing,
}

impl SfenError {
    /// This error's message, as the bytes a diagnostic line carries.
    ///
    /// An input byte is quoted the way [`TextWriter::quoted_byte`] quotes one,
    /// so a refusal naming a byte that arrived from a host stays ASCII whatever
    /// that byte was.
    pub fn write_message(&self, out: &mut TextWriter<'_>) {
        out.bytes(b"sfen: ");
        match self {
            SfenError::UnexpectedEnd => {
                out.bytes(b"unexpected end of input");
            }
            SfenError::InvalidPiece(b) => {
                out.bytes(b"invalid piece char ").quoted_byte(*b);
            }
            SfenError::UnexpectedPromoteMarker => {
                out.bytes(b"unexpected '+' marker");
            }
            SfenError::NonPromotablePromoted(b) => {
                out.bytes(b"piece ")
                    .quoted_byte(*b)
                    .bytes(b" cannot be promoted");
            }
            SfenError::BoardCursorOverflow => {
                out.bytes(b"board cursor advanced past file 1");
            }
            SfenError::BoardCursorIncomplete => {
                out.bytes(b"board section ended with cursor not at end");
            }
            SfenError::InvalidSideToMove(b) => {
                out.bytes(b"invalid side-to-move ").quoted_byte(*b);
            }
            SfenError::MissingHandPieceAfterCount => {
                out.bytes(b"hand count without following piece");
            }
            SfenError::InvalidHandPiece(b) => {
                out.bytes(b"invalid hand piece ").quoted_byte(*b);
            }
            SfenError::InvalidPly => {
                out.bytes(b"invalid ply");
            }
            SfenError::UnexpectedTrailing => {
                out.bytes(b"unexpected trailing input");
            }
        }
    }
}

pub fn parse_sfen(s: &[u8]) -> Result<Position, SfenError> {
    let mut pos = Position::empty();
    parse_sfen_into(&mut pos, s)?;
    Ok(pos)
}

/// Parse `s` into `pos`, reusing the buffers it already holds instead of
/// building a position of its own.
///
/// A rejected SFEN leaves `pos` holding whatever part of it had already been
/// read, so a caller with a position to protect parses into a scratch and
/// installs it only once this has returned `Ok`.
pub fn parse_sfen_into(pos: &mut Position, s: &[u8]) -> Result<(), SfenError> {
    let mut fields = s.split(|&b| b == b' ');
    let board_field = fields.next().ok_or(SfenError::UnexpectedEnd)?;
    let stm_field = fields.next().ok_or(SfenError::UnexpectedEnd)?;
    let hand_field = fields.next().ok_or(SfenError::UnexpectedEnd)?;
    let ply_field = fields.next().ok_or(SfenError::UnexpectedEnd)?;
    if fields.next().is_some() {
        return Err(SfenError::UnexpectedTrailing);
    }

    parse_sfen_fields_into(pos, [board_field, stm_field, hand_field, ply_field])
}

/// [`parse_sfen_into`] for an SFEN that arrived already split into its four
/// fields — board, side to move, hands, ply — which is how a `position sfen`
/// command carries one, so nothing has to be joined to read it.
pub fn parse_sfen_fields_into(pos: &mut Position, fields: [&[u8]; 4]) -> Result<(), SfenError> {
    let [board_field, stm_field, hand_field, ply_field] = fields;

    pos.reset_empty();
    parse_board(board_field, pos)?;
    parse_side_to_move(stm_field, pos)?;
    parse_hands(hand_field, pos)?;
    parse_ply(ply_field, pos)?;
    // The board / hand / side mutations above go through the direct setters,
    // which bypass incremental key maintenance; seed the keys once here.
    pos.refresh_keys();
    Ok(())
}

fn parse_board(field: &[u8], pos: &mut Position) -> Result<(), SfenError> {
    let mut file: i8 = (Square::FILES as i8) - 1;
    let mut rank: i8 = 0;
    let mut promote = false;

    for &byte in field {
        match byte {
            b'/' => {
                if file != -1 {
                    return Err(SfenError::BoardCursorIncomplete);
                }
                if rank == (Square::RANKS as i8) - 1 {
                    return Err(SfenError::BoardCursorOverflow);
                }
                if promote {
                    return Err(SfenError::UnexpectedPromoteMarker);
                }
                file = (Square::FILES as i8) - 1;
                rank += 1;
            }
            b'+' => {
                if promote {
                    return Err(SfenError::UnexpectedPromoteMarker);
                }
                promote = true;
            }
            b'1'..=b'9' => {
                if promote {
                    return Err(SfenError::UnexpectedPromoteMarker);
                }
                let skip = (byte - b'0') as i8;
                if file - skip < -1 {
                    return Err(SfenError::BoardCursorOverflow);
                }
                file -= skip;
            }
            _ => {
                let (kind, color) = parse_piece_byte(byte)?;
                if file < 0 || rank < 0 || rank >= Square::RANKS as i8 {
                    return Err(SfenError::BoardCursorOverflow);
                }
                let piece = if promote {
                    Piece::promoted(kind, color)
                        .ok_or(SfenError::NonPromotablePromoted(byte.to_ascii_uppercase()))?
                } else {
                    Piece::new(kind, color)
                };
                let sq = Square::new(file as u8, rank as u8).unwrap();
                pos.board_mut().set(sq, Some(piece));
                file -= 1;
                promote = false;
            }
        }
    }

    if promote {
        return Err(SfenError::UnexpectedPromoteMarker);
    }
    if rank != (Square::RANKS as i8) - 1 || file != -1 {
        return Err(SfenError::BoardCursorIncomplete);
    }
    Ok(())
}

fn parse_side_to_move(field: &[u8], pos: &mut Position) -> Result<(), SfenError> {
    let (&token, rest) = field.split_first().ok_or(SfenError::UnexpectedEnd)?;
    if !rest.is_empty() {
        return Err(SfenError::InvalidSideToMove(token));
    }
    let color = match token {
        b'b' => Color::Black,
        b'w' => Color::White,
        other => return Err(SfenError::InvalidSideToMove(other)),
    };
    pos.set_side_to_move(color);
    Ok(())
}

fn parse_hands(field: &[u8], pos: &mut Position) -> Result<(), SfenError> {
    if field.is_empty() {
        return Err(SfenError::UnexpectedEnd);
    }
    if field == b"-" {
        return Ok(());
    }

    let mut count: u32 = 0;
    let mut count_started = false;
    for &byte in field {
        if byte.is_ascii_digit() {
            count = count * 10 + u32::from(byte - b'0');
            count_started = true;
        } else {
            let (kind, color) = parse_piece_byte(byte)?;
            if !is_hand_kind(kind) {
                return Err(SfenError::InvalidHandPiece(byte));
            }
            let n = if count_started { count } else { 1 };
            for _ in 0..n {
                pos.hand_mut(color).increment(kind);
            }
            count = 0;
            count_started = false;
        }
    }

    if count_started {
        return Err(SfenError::MissingHandPieceAfterCount);
    }
    Ok(())
}

fn parse_ply(field: &[u8], pos: &mut Position) -> Result<(), SfenError> {
    let n = atoi_u16(field).ok_or(SfenError::InvalidPly)?;
    pos.set_ply(n);
    Ok(())
}

fn parse_piece_byte(byte: u8) -> Result<(PieceKind, Color), SfenError> {
    let color = if byte.is_ascii_uppercase() {
        Color::Black
    } else if byte.is_ascii_lowercase() {
        Color::White
    } else {
        return Err(SfenError::InvalidPiece(byte));
    };
    let kind = match byte.to_ascii_uppercase() {
        b'P' => PieceKind::Pawn,
        b'L' => PieceKind::Lance,
        b'N' => PieceKind::Knight,
        b'S' => PieceKind::Silver,
        b'G' => PieceKind::Gold,
        b'B' => PieceKind::Bishop,
        b'R' => PieceKind::Rook,
        b'K' => PieceKind::King,
        _ => return Err(SfenError::InvalidPiece(byte)),
    };
    Ok((kind, color))
}

const fn is_hand_kind(kind: PieceKind) -> bool {
    !matches!(kind, PieceKind::King)
}

const HAND_OUTPUT_ORDER: [PieceKind; 7] = [
    PieceKind::Rook,
    PieceKind::Bishop,
    PieceKind::Gold,
    PieceKind::Silver,
    PieceKind::Knight,
    PieceKind::Lance,
    PieceKind::Pawn,
];

/// A buffer one SFEN is written into — [`SFEN_CAPACITY`] bytes, which the
/// widest one a board can spell fits.
pub struct SfenBuf {
    bytes: [u8; SFEN_CAPACITY],
}

impl SfenBuf {
    pub const fn new() -> Self {
        Self {
            bytes: [0; SFEN_CAPACITY],
        }
    }
}

impl Default for SfenBuf {
    fn default() -> Self {
        Self::new()
    }
}

/// Write `pos`'s SFEN into `buf` and return the text — the inverse of
/// [`parse_sfen`].
pub fn format_sfen<'b>(pos: &Position, buf: &'b mut SfenBuf) -> &'b [u8] {
    let mut out = TextWriter::new(&mut buf.bytes);
    write_sfen(pos, &mut out);
    debug_assert!(!out.overflowed(), "an SFEN fits SFEN_CAPACITY");
    let len = out.len();
    &buf.bytes[..len]
}

/// Append `pos`'s SFEN to `out`, for a caller composing a longer line around it.
pub fn write_sfen(pos: &Position, out: &mut TextWriter<'_>) {
    write_board(pos, out);
    out.byte(b' ');
    out.byte(match pos.side_to_move() {
        Color::Black => b'b',
        Color::White => b'w',
    });
    out.byte(b' ');
    write_hands(pos, out);
    out.byte(b' ');
    out.u64(u64::from(pos.ply()));
}

fn write_board(pos: &Position, out: &mut TextWriter<'_>) {
    for rank in 0..Square::RANKS {
        if rank != 0 {
            out.byte(b'/');
        }
        let mut empty: u8 = 0;
        for file in (0..Square::FILES).rev() {
            let sq = Square::new(file, rank).unwrap();
            match pos.board().get(sq) {
                None => empty += 1,
                Some(piece) => {
                    if empty > 0 {
                        out.byte(b'0' + empty);
                        empty = 0;
                    }
                    if piece.promoted {
                        out.byte(b'+');
                    }
                    out.byte(piece_byte(piece.kind, piece.color));
                }
            }
        }
        if empty > 0 {
            out.byte(b'0' + empty);
        }
    }
}

fn write_hands(pos: &Position, out: &mut TextWriter<'_>) {
    let mut wrote = false;
    for color in [Color::Black, Color::White] {
        for &kind in &HAND_OUTPUT_ORDER {
            let n = pos.hand(color).count(kind);
            if n == 0 {
                continue;
            }
            wrote = true;
            if n != 1 {
                out.u64(u64::from(n));
            }
            out.byte(piece_byte(kind, color));
        }
    }
    if !wrote {
        out.byte(b'-');
    }
}

fn piece_byte(kind: PieceKind, color: Color) -> u8 {
    let upper = match kind {
        PieceKind::Pawn => b'P',
        PieceKind::Lance => b'L',
        PieceKind::Knight => b'N',
        PieceKind::Silver => b'S',
        PieceKind::Gold => b'G',
        PieceKind::Bishop => b'B',
        PieceKind::Rook => b'R',
        PieceKind::King => b'K',
    };
    match color {
        Color::Black => upper,
        Color::White => upper.to_ascii_lowercase(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sfen_of(pos: &Position) -> Vec<u8> {
        let mut buf = SfenBuf::new();
        format_sfen(pos, &mut buf).to_vec()
    }

    #[test]
    fn startpos_round_trips_byte_for_byte() {
        let pos = parse_sfen(STARTPOS_SFEN).unwrap();
        assert_eq!(sfen_of(&pos), STARTPOS_SFEN);
    }

    #[test]
    fn startpos_places_kings_at_5a_and_5e() {
        let pos = parse_sfen(STARTPOS_SFEN).unwrap();
        let black_king_sq = Square::new(4, 8).unwrap();
        let white_king_sq = Square::new(4, 0).unwrap();
        assert_eq!(
            pos.board().get(black_king_sq),
            Some(Piece::new(PieceKind::King, Color::Black))
        );
        assert_eq!(
            pos.board().get(white_king_sq),
            Some(Piece::new(PieceKind::King, Color::White))
        );
    }

    #[test]
    fn side_to_move_b_and_w_parse() {
        let flipped = b"lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1";
        let pos = parse_sfen(flipped).unwrap();
        assert_eq!(pos.side_to_move(), Color::White);
        let pos2 = parse_sfen(STARTPOS_SFEN).unwrap();
        assert_eq!(pos2.side_to_move(), Color::Black);
    }

    #[test]
    fn empty_run_digits_place_piece_at_correct_file() {
        let sfen = b"9/9/9/9/4P4/9/9/9/9 b - 1";
        let pos = parse_sfen(sfen).unwrap();
        let sq = Square::new(4, 4).unwrap();
        assert_eq!(
            pos.board().get(sq),
            Some(Piece::new(PieceKind::Pawn, Color::Black))
        );
        assert_eq!(sfen_of(&pos), sfen);
    }

    #[test]
    fn promoted_piece_round_trips() {
        let sfen = b"9/9/9/9/4+P4/9/9/9/9 b - 1";
        let pos = parse_sfen(sfen).unwrap();
        let sq = Square::new(4, 4).unwrap();
        let piece = pos.board().get(sq).unwrap();
        assert!(piece.promoted);
        assert_eq!(piece.kind, PieceKind::Pawn);
        assert_eq!(piece.color, Color::Black);
        assert_eq!(sfen_of(&pos), sfen);
    }

    #[test]
    fn hand_with_mixed_counts_round_trips() {
        let sfen = b"9/9/9/9/9/9/9/9/9 b P2p 1";
        let pos = parse_sfen(sfen).unwrap();
        assert_eq!(pos.hand(Color::Black).count(PieceKind::Pawn), 1);
        assert_eq!(pos.hand(Color::White).count(PieceKind::Pawn), 2);
        assert_eq!(sfen_of(&pos), sfen);
    }

    #[test]
    fn hand_dash_round_trips() {
        let sfen = b"9/9/9/9/9/9/9/9/9 b - 1";
        let pos = parse_sfen(sfen).unwrap();
        assert_eq!(sfen_of(&pos), sfen);
    }

    #[test]
    fn hand_multi_digit_count_round_trips() {
        let sfen = b"9/9/9/9/9/9/9/9/9 b 18P 1";
        let pos = parse_sfen(sfen).unwrap();
        assert_eq!(pos.hand(Color::Black).count(PieceKind::Pawn), 18);
        assert_eq!(sfen_of(&pos), sfen);
    }

    #[test]
    fn hand_full_canonical_order() {
        let sfen = b"9/9/9/9/9/9/9/9/9 b RBGSNLPrbgsnlp 1";
        let pos = parse_sfen(sfen).unwrap();
        assert_eq!(sfen_of(&pos), sfen);
    }

    #[test]
    fn rejects_promoted_gold() {
        let sfen = b"9/9/9/9/4+G4/9/9/9/9 b - 1";
        match parse_sfen(sfen) {
            Err(SfenError::NonPromotablePromoted(b'G')) => {}
            other => panic!("expected NonPromotablePromoted(b'G'), got {other:?}"),
        }
    }

    #[test]
    fn rejects_promote_marker_before_digit() {
        let sfen = b"9/9/9/9/+1P6/9/9/9/9 b - 1";
        match parse_sfen(sfen) {
            Err(SfenError::UnexpectedPromoteMarker) => {}
            other => panic!("expected UnexpectedPromoteMarker, got {other:?}"),
        }
    }

    #[test]
    fn rejects_invalid_side_to_move() {
        let sfen = b"9/9/9/9/9/9/9/9/9 x - 1";
        match parse_sfen(sfen) {
            Err(SfenError::InvalidSideToMove(b'x')) => {}
            other => panic!("expected InvalidSideToMove(b'x'), got {other:?}"),
        }
    }

    #[test]
    fn rejects_king_in_hand() {
        let sfen = b"9/9/9/9/9/9/9/9/9 b K 1";
        match parse_sfen(sfen) {
            Err(SfenError::InvalidHandPiece(b'K')) => {}
            other => panic!("expected InvalidHandPiece(b'K'), got {other:?}"),
        }
    }

    #[test]
    fn rejects_count_without_piece() {
        let sfen = b"9/9/9/9/9/9/9/9/9 b 5 1";
        match parse_sfen(sfen) {
            Err(SfenError::MissingHandPieceAfterCount) => {}
            other => panic!("expected MissingHandPieceAfterCount, got {other:?}"),
        }
    }

    #[test]
    fn rejects_overflowing_empty_run() {
        let sfen = b"9/9/9/9/8P1/9/9/9/9 b - 1";
        match parse_sfen(sfen) {
            Err(SfenError::BoardCursorOverflow) => {}
            other => panic!("expected BoardCursorOverflow, got {other:?}"),
        }
    }

    #[test]
    fn rejects_short_rank() {
        let sfen = b"8/9/9/9/9/9/9/9/9 b - 1";
        match parse_sfen(sfen) {
            Err(SfenError::BoardCursorIncomplete) => {}
            other => panic!("expected BoardCursorIncomplete, got {other:?}"),
        }
    }

    #[test]
    fn rejects_trailing_field() {
        let sfen = b"9/9/9/9/9/9/9/9/9 b - 1 extra";
        match parse_sfen(sfen) {
            Err(SfenError::UnexpectedTrailing) => {}
            other => panic!("expected UnexpectedTrailing, got {other:?}"),
        }
    }

    /// A field carrying a byte no SFEN is spelled in — a CP932 lead byte, say —
    /// is refused like any other malformed field, and the message it renders is
    /// still ASCII.
    #[test]
    fn rejects_a_non_ascii_board_byte_and_reports_it_in_ascii() {
        let sfen = b"\x82\xa0/9/9/9/9/9/9/9/9 b - 1";
        let err = parse_sfen(sfen).expect_err("a non-ASCII board byte is refused");
        assert_eq!(err, SfenError::InvalidPiece(0x82));
        let mut bytes = [0u8; 64];
        let mut out = TextWriter::new(&mut bytes);
        err.write_message(&mut out);
        assert_eq!(out.as_bytes(), b"sfen: invalid piece char 0x82");
        assert!(out.as_bytes().is_ascii());
    }

    #[test]
    fn rejects_a_non_numeric_ply() {
        assert_eq!(
            parse_sfen(b"9/9/9/9/9/9/9/9/9 b - x"),
            Err(SfenError::InvalidPly)
        );
        assert_eq!(
            parse_sfen(b"9/9/9/9/9/9/9/9/9 b - "),
            Err(SfenError::InvalidPly)
        );
    }

    #[test]
    fn startpos_helper_matches_parse() {
        let from_helper = Position::startpos();
        let from_parse = parse_sfen(STARTPOS_SFEN).unwrap();
        assert_eq!(from_helper, from_parse);
    }

    #[test]
    fn the_widest_board_fits_the_buffer() {
        // Every square a promoted pawn and both hands as full as the pieces
        // allow: the widest board text, and the widest hand text beside it.
        let widest = b"+p+p+p+p+p+p+p+p+p/+p+p+p+p+p+p+p+p+p/+p+p+p+p+p+p+p+p+p/\
+p+p+p+p+p+p+p+p+p/+p+p+p+p+p+p+p+p+p/+p+p+p+p+p+p+p+p+p/\
+p+p+p+p+p+p+p+p+p/+p+p+p+p+p+p+p+p+p/+p+p+p+p+p+p+p+p+p b - 65535";
        let pos = parse_sfen(widest).expect("the widest board parses");
        let mut buf = SfenBuf::new();
        assert_eq!(format_sfen(&pos, &mut buf), widest);
    }
}
