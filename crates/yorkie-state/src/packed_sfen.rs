//! PackedSfen — the 32-byte Huffman position encoding, ported bit-for-bit from
//! the reference.
//!
//! A `.ybb` opening-book index keys positions by this exact encoding, so any
//! single-bit divergence from the reference writer makes the index binary
//! search miss.
//!
//! Bits are packed least-significant-first within each byte, bytes ascending.
//! The serialization order is side to move, both king squares, the board in
//! square-index order with the kings skipped, both hands in Apery order, then
//! the "piece box" of every piece on neither the board nor in a hand. The box
//! pads the leftover pieces, so the width is a fixed 256 bits for any position
//! within the standard piece complement.
//!
//! The two kings are the only pieces whose squares are written outright, and
//! the 256 bits budget for exactly two of them: a position missing a king needs
//! one more board bit than fits, so it has no encoding here, and the decoder
//! rejects the square value that would claim one.

use core::fmt;

use crate::color::Color;
use crate::piece::{Piece, PieceKind};
use crate::position::Position;
use crate::square::Square;

/// Length in bytes of a [`PackedSfen`].
pub const PACKED_SFEN_LEN: usize = 32;

/// Width in bits of a [`PackedSfen`]: the encoding is exactly this long for
/// every position it can represent, so it doubles as the end-of-stream mark
/// that terminates the hand and piece-box run.
const PACKED_SFEN_BITS: usize = PACKED_SFEN_LEN * 8;

/// Widest board code in the Huffman table.
const MAX_BOARD_CODE_BITS: u32 = 6;

/// Widest hand code — a board code with its low bit dropped.
const MAX_HAND_CODE_BITS: u32 = MAX_BOARD_CODE_BITS - 1;

/// A shogi position packed into 32 bytes.
///
/// The encoding does **not** cover the game ply, so two positions differing
/// only in ply pack to identical bytes.
pub type PackedSfen = [u8; PACKED_SFEN_LEN];

/// Piece kinds in the reference's Apery order (`to_apery_pieces[]`), which
/// [`PieceKind`]'s discriminants already follow.
const HAND_ORDER: [PieceKind; 7] = [
    PieceKind::Pawn,
    PieceKind::Lance,
    PieceKind::Knight,
    PieceKind::Silver,
    PieceKind::Gold,
    PieceKind::Bishop,
    PieceKind::Rook,
];

/// Starting piece-box counts. A king is never boxed.
const PIECE_BOX_START: [i32; 7] = [18, 4, 4, 4, 4, 2, 2];

/// LSB-first bit writer over the fixed 32-byte buffer.
struct BitWriter {
    data: PackedSfen,
    cursor: usize,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            data: [0; PACKED_SFEN_LEN],
            cursor: 0,
        }
    }

    #[inline]
    fn write_one_bit(&mut self, b: bool) {
        let byte = self.cursor / 8;
        // An over-populated illegal position would otherwise index past the
        // end. Dropping the overflow bits keeps the encoder total; its output
        // is only meaningful for legal inputs anyway.
        if b && byte < PACKED_SFEN_LEN {
            self.data[byte] |= 1 << (self.cursor & 7);
        }
        self.cursor += 1;
    }

    #[inline]
    fn write_n_bit(&mut self, d: u32, n: u32) {
        for i in 0..n {
            self.write_one_bit((d & (1 << i)) != 0);
        }
    }
}

/// LSB-first bit reader over the fixed 32-byte buffer, the inverse of
/// [`BitWriter`].
struct BitReader<'a> {
    data: &'a PackedSfen,
    cursor: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a PackedSfen) -> Self {
        Self { data, cursor: 0 }
    }

    #[inline]
    fn read_one_bit(&mut self) -> Result<bool, PackedSfenError> {
        if self.cursor >= PACKED_SFEN_BITS {
            return Err(PackedSfenError::StreamOverrun);
        }
        let bit = (self.data[self.cursor / 8] >> (self.cursor & 7)) & 1;
        self.cursor += 1;
        Ok(bit != 0)
    }

    fn read_n_bit(&mut self, n: u32) -> Result<u32, PackedSfenError> {
        let mut value = 0;
        for i in 0..n {
            if self.read_one_bit()? {
                value |= 1 << i;
            }
        }
        Ok(value)
    }

    #[inline]
    fn at_end(&self) -> bool {
        self.cursor >= PACKED_SFEN_BITS
    }
}

/// The on-board and hand Huffman code of a raw piece kind (`huffman_table[]`).
fn huffman_board(kind: PieceKind) -> (u32, u32) {
    match kind {
        PieceKind::Pawn => (0x01, 2),
        PieceKind::Lance => (0x03, 4),
        PieceKind::Knight => (0x0b, 4),
        PieceKind::Silver => (0x07, 4),
        PieceKind::Gold => (0x0f, 5),
        PieceKind::Bishop => (0x1f, 6),
        PieceKind::Rook => (0x3f, 6),
        // A king is encoded by its square, not the Huffman stream, and never
        // sits in a hand.
        PieceKind::King => unreachable!("king is not Huffman-coded as a board/hand piece"),
    }
}

/// The piece-box Huffman code of a raw piece kind
/// (`huffman_table_piecebox[]`).
fn huffman_piece_box(kind: PieceKind) -> (u32, u32) {
    match kind {
        PieceKind::Pawn => (0x02, 2),
        PieceKind::Lance => (0x09, 4),
        PieceKind::Knight => (0x0d, 4),
        PieceKind::Silver => (0x0b, 4),
        PieceKind::Gold => (0x1b, 5),
        PieceKind::Bishop => (0x2f, 6),
        PieceKind::Rook => (0x3f, 6),
        PieceKind::King => unreachable!("king is never in the piece box"),
    }
}

/// A board piece, Huffman-coded (`write_board_piece_to_stream`): code, then a
/// promote bit except for gold, then a colour bit.
fn write_board_piece(w: &mut BitWriter, piece: Piece) {
    let (code, bits) = huffman_board(piece.kind);
    w.write_n_bit(code, bits);
    if piece.kind != PieceKind::Gold {
        w.write_one_bit(piece.promoted);
    }
    w.write_one_bit(piece.color == Color::White);
}

/// A hand piece, Huffman-coded (`write_hand_piece_to_stream`): the board code
/// with its low bit dropped, then a forced-unpromoted bit except for gold, then
/// a colour bit.
fn write_hand_piece(w: &mut BitWriter, kind: PieceKind, color: Color) {
    let (code, bits) = huffman_board(kind);
    w.write_n_bit(code >> 1, bits - 1);
    if kind != PieceKind::Gold {
        w.write_one_bit(false);
    }
    w.write_one_bit(color == Color::White);
}

/// A piece-box piece, Huffman-coded (`write_piecebox_piece_to_stream`): the
/// piece-box code, then a zero colour bit except for gold, which encodes its
/// colour implicitly.
fn write_piece_box_piece(w: &mut BitWriter, kind: PieceKind) {
    let (code, bits) = huffman_piece_box(kind);
    w.write_n_bit(code, bits);
    if kind != PieceKind::Gold {
        w.write_one_bit(false);
    }
}

/// The kind a board code identifies, if `(code, bits)` is one. Every board code
/// has bit 0 set, so a leading zero bit — the empty-square code — cannot begin
/// one, and the caller reads it before asking.
fn board_kind_of_code(code: u32, bits: u32) -> Option<PieceKind> {
    HAND_ORDER
        .into_iter()
        .find(|&kind| huffman_board(kind) == (code, bits))
}

/// The kind a hand code identifies, if `(code, bits)` is one.
fn hand_kind_of_code(code: u32, bits: u32) -> Option<PieceKind> {
    HAND_ORDER.into_iter().find(|&kind| {
        let (board_code, board_bits) = huffman_board(kind);
        (board_code >> 1, board_bits - 1) == (code, bits)
    })
}

/// One board square, the inverse of [`write_board_piece`]. `None` is an empty
/// square.
fn read_board_piece(r: &mut BitReader) -> Result<Option<Piece>, PackedSfenError> {
    let mut code = 0;
    let mut bits = 0;
    let kind = loop {
        if r.read_one_bit()? {
            code |= 1 << bits;
        }
        bits += 1;
        if code == 0 {
            return Ok(None);
        }
        if let Some(kind) = board_kind_of_code(code, bits) {
            break kind;
        }
        if bits >= MAX_BOARD_CODE_BITS {
            return Err(PackedSfenError::UnknownCode { code, bits });
        }
    };

    let promoted = kind != PieceKind::Gold && r.read_one_bit()?;
    let color = read_color(r)?;
    Ok(Some(if promoted {
        Piece::promoted(kind, color).expect("a promotable kind, gold having no promote bit")
    } else {
        Piece::new(kind, color)
    }))
}

/// One hand or piece-box entry, the inverse of [`write_hand_piece`] and
/// [`write_piece_box_piece`]. The promote bit a hand piece writes as zero is
/// what tells the two apart.
fn read_hand_piece(r: &mut BitReader) -> Result<HandEntry, PackedSfenError> {
    let mut code = 0;
    let mut bits = 0;
    let kind = loop {
        if r.read_one_bit()? {
            code |= 1 << bits;
        }
        bits += 1;
        if let Some(kind) = hand_kind_of_code(code, bits) {
            break kind;
        }
        if bits >= MAX_HAND_CODE_BITS {
            return Err(PackedSfenError::UnknownCode { code, bits });
        }
    };

    let in_piece_box = kind != PieceKind::Gold && r.read_one_bit()?;
    let color = read_color(r)?;
    Ok(HandEntry {
        kind,
        color,
        in_piece_box,
    })
}

/// A hand or piece-box entry as the stream spells it. A boxed gold arrives as a
/// promoted white silver: gold has no promote bit of its own, so the piece box
/// spells it with the silver code and the colour bit that a boxed silver leaves
/// zero. [`Self::piece_box_kind`] undoes that.
struct HandEntry {
    kind: PieceKind,
    color: Color,
    in_piece_box: bool,
}

impl HandEntry {
    fn piece_box_kind(&self) -> PieceKind {
        if self.kind == PieceKind::Silver && self.color == Color::White {
            PieceKind::Gold
        } else {
            self.kind
        }
    }
}

fn read_color(r: &mut BitReader) -> Result<Color, PackedSfenError> {
    Ok(if r.read_one_bit()? {
        Color::White
    } else {
        Color::Black
    })
}

/// Locate a colour's king, or `SQ_NB` when it is absent.
fn king_square(pos: &Position, color: Color) -> u32 {
    for index in 0..Square::COUNT as u8 {
        let sq = Square::from_index(index).expect("index < COUNT");
        if let Some(p) = pos.board().get(sq)
            && p.kind == PieceKind::King
            && p.color == color
        {
            return sq.index() as u32;
        }
    }
    Square::COUNT as u32
}

/// Pack a [`Position`] into its 32-byte [`PackedSfen`], bit-identical to the
/// reference `Position::sfen_pack`.
pub fn sfen_pack(pos: &Position) -> PackedSfen {
    let mut w = BitWriter::new();

    w.write_one_bit(pos.side_to_move() == Color::White);

    w.write_n_bit(king_square(pos, Color::Black), 7);
    w.write_n_bit(king_square(pos, Color::White), 7);

    // Board pieces, the kings already emitted, plus piece-box bookkeeping.
    let mut box_count = PIECE_BOX_START;
    for index in 0..Square::COUNT as u8 {
        let sq = Square::from_index(index).expect("index < COUNT");
        match pos.board().get(sq) {
            Some(p) if p.kind == PieceKind::King => {}
            Some(p) => {
                write_board_piece(&mut w, p);
                box_count[p.kind.index()] -= 1;
            }
            None => w.write_one_bit(false),
        }
    }

    for color in [Color::Black, Color::White] {
        for kind in HAND_ORDER {
            let n = pos.hand(color).count(kind);
            for _ in 0..n {
                write_hand_piece(&mut w, kind, color);
            }
            box_count[kind.index()] -= i32::from(n);
        }
    }

    // The piece box: everything left over.
    for kind in HAND_ORDER {
        let leftover = box_count[kind.index()].max(0);
        for _ in 0..leftover {
            write_piece_box_piece(&mut w, kind);
        }
    }

    debug_assert_eq!(
        w.cursor, 256,
        "packed sfen must be exactly 256 bits for a legal position"
    );
    w.data
}

/// Why a [`PackedSfen`] does not decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackedSfenError {
    /// A king-square field naming no square. The two kings are budgeted for,
    /// so the reference's "this side has no king" value 81 is rejected here
    /// along with everything above it.
    KingSquareOutOfRange(u32),
    /// Both kings on one square.
    KingSquaresCollide(u32),
    /// Bits that begin no code in the Huffman table. Both tables are complete
    /// over their widths — every bit path ends at a piece — so this is what
    /// keeps a reader total should one stop being, not something a stream can
    /// say today.
    UnknownCode { code: u32, bits: u32 },
    /// A read past the last bit of the stream.
    StreamOverrun,
    /// More copies of a kind than the piece complement holds.
    TooManyPieces(PieceKind),
    /// Fewer copies of a kind than the piece complement holds — pieces the
    /// stream accounts for neither on the board, nor in a hand, nor in the
    /// piece box.
    MissingPieces(PieceKind),
}

impl fmt::Display for PackedSfenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PackedSfenError::KingSquareOutOfRange(sq) => {
                write!(f, "packed sfen: king square {sq} is not a board square")
            }
            PackedSfenError::KingSquaresCollide(sq) => {
                write!(f, "packed sfen: both kings on square {sq}")
            }
            PackedSfenError::UnknownCode { code, bits } => {
                write!(f, "packed sfen: {bits}-bit prefix {code:#x} is not a code")
            }
            PackedSfenError::StreamOverrun => {
                f.write_str("packed sfen: read past the end of the stream")
            }
            PackedSfenError::TooManyPieces(kind) => {
                write!(f, "packed sfen: more {kind:?} than the piece set holds")
            }
            PackedSfenError::MissingPieces(kind) => {
                write!(f, "packed sfen: fewer {kind:?} than the piece set holds")
            }
        }
    }
}

impl std::error::Error for PackedSfenError {}

/// Charge one copy of `kind` against what the piece set has left.
fn take_piece(remaining: &mut [i32; 7], kind: PieceKind) -> Result<(), PackedSfenError> {
    let left = &mut remaining[kind.index()];
    if *left == 0 {
        return Err(PackedSfenError::TooManyPieces(kind));
    }
    *left -= 1;
    Ok(())
}

/// Rebuild the [`Position`] a [`PackedSfen`] encodes, the inverse of
/// [`sfen_pack`] (the reference `Position::set_from_packed_sfen`, without its
/// file-flipping `mirror` argument).
///
/// The ply is not part of the 32 bytes, so the result carries ply 1; a caller
/// that needs the real one keeps it outside the code. Everything else — the
/// keys and the piece sets the board maintains — is rebuilt, so the result is
/// indistinguishable from the same position parsed from SFEN.
pub fn position_from_packed_sfen(packed: &PackedSfen) -> Result<Position, PackedSfenError> {
    let mut r = BitReader::new(packed);
    let side_to_move = read_color(&mut r)?;

    let mut pos = Position::empty();
    let mut remaining = PIECE_BOX_START;
    {
        // One guard over the whole board phase: it recomputes the check info on
        // drop, which is worth doing once rather than once per square.
        let mut board = pos.board_mut();

        for color in [Color::Black, Color::White] {
            let index = r.read_n_bit(7)?;
            let sq = u8::try_from(index)
                .ok()
                .and_then(Square::from_index)
                .ok_or(PackedSfenError::KingSquareOutOfRange(index))?;
            if board.get(sq).is_some() {
                return Err(PackedSfenError::KingSquaresCollide(index));
            }
            board.set(sq, Some(Piece::new(PieceKind::King, color)));
        }

        for index in 0..Square::COUNT as u8 {
            let sq = Square::from_index(index).expect("index < COUNT");
            // The kings are already placed, and the stream skips their squares.
            if board.get(sq).is_some() {
                continue;
            }
            if let Some(piece) = read_board_piece(&mut r)? {
                take_piece(&mut remaining, piece.kind)?;
                board.set(sq, Some(piece));
            }
        }
    }

    while !r.at_end() {
        let entry = read_hand_piece(&mut r)?;
        if entry.in_piece_box {
            take_piece(&mut remaining, entry.piece_box_kind())?;
            continue;
        }
        take_piece(&mut remaining, entry.kind)?;
        pos.hand_mut(entry.color).increment(entry.kind);
    }

    if let Some(kind) = HAND_ORDER
        .into_iter()
        .find(|kind| remaining[kind.index()] != 0)
    {
        return Err(PackedSfenError::MissingPieces(kind));
    }

    pos.set_side_to_move(side_to_move);
    // The board / hand / side mutations above go through the direct setters,
    // which bypass incremental key maintenance; seed the keys once here.
    pos.refresh_keys();
    Ok(pos)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::move_::Move;
    use crate::sfen::{format_sfen, parse_sfen};

    /// Parse an SFEN that may omit the trailing ply field, as the reference's
    /// own test vectors do.
    fn parse(sfen: &str) -> Position {
        let with_ply = if sfen.split(' ').count() == 3 {
            format!("{sfen} 1")
        } else {
            sfen.to_string()
        };
        parse_sfen(&with_ply).expect("valid sfen")
    }

    /// The SFEN minus its ply field — everything the 32 bytes encode.
    fn sfen_without_ply(pos: &Position) -> String {
        let sfen = format_sfen(pos);
        sfen.rsplit_once(' ')
            .expect("format_sfen always writes the ply field")
            .0
            .to_string()
    }

    fn assert_round_trips(pos: &Position, what: &str) {
        let decoded = position_from_packed_sfen(&sfen_pack(pos))
            .unwrap_or_else(|e| panic!("{what} failed to decode: {e}"));
        assert_eq!(sfen_without_ply(&decoded), sfen_without_ply(pos), "{what}");
        assert_eq!(decoded.key(), pos.key(), "key of {what}");
        assert_eq!(decoded.ply(), 1, "ply of {what}");
    }

    /// The eight standard handicaps, plus the two one-lance ones, as the
    /// weaker side sees them: the odds pieces sit in the piece box, which the
    /// even game leaves empty.
    const HANDICAP_SFENS: [&str; 10] = [
        "lnsgkgsn1/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
        "1nsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
        "lnsgkgsnl/1r7/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
        "lnsgkgsnl/7b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
        "lnsgkgsn1/7b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
        "lnsgkgsnl/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
        "1nsgkgsn1/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
        "2sgkgs2/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
        "3gkg3/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
        "4k4/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
    ];

    /// Mate-problem shapes: sparse boards, promoted pieces, both hands full,
    /// and — the two-king budget being what it is — a king for each side.
    const TSUME_SFENS: [&str; 6] = [
        "4k4/9/4r4/9/9/9/4K3B/9/9 b RG2gs2n3p 1",
        "k8/1P7/G8/1N2P4/9/9/9/9/8K b 2PG2pg 1",
        "4k4/3P3+PL/2N2PR2/1L2BNS2/4N4/9/9/9/4K4 b - 1",
        "9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1",
        "4k4/9/9/9/9/9/9/9/4K4 b 2R2B2G2S2N2L9P 1",
        "4k4/3+P+P+P3/9/9/9/9/9/4K4/9 b GSNLRBrb 1",
    ];

    /// Vectors transcribed from the reference's own PackedSfen unit test, whose
    /// bytes came from a third implementation. They exercise the board, both
    /// hands, and the piece box.
    #[test]
    fn matches_reference_cshogi_vectors() {
        let cases: [(&str, [u8; 32]); 4] = [
            (
                "lnsgkgsnl/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w -",
                [
                    89, 164, 81, 34, 12, 171, 68, 252, 44, 167, 68, 56, 94, 137, 240, 72, 132, 87,
                    34, 60, 167, 68, 56, 86, 137, 248, 88, 70, 137, 48, 188, 126,
                ],
            ),
            (
                "lns1kgsnl/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w -",
                [
                    89, 164, 81, 34, 12, 171, 68, 252, 44, 167, 68, 56, 94, 137, 240, 72, 4, 18,
                    225, 57, 37, 194, 177, 74, 196, 199, 50, 74, 132, 97, 191, 126,
                ],
            ),
            (
                "lnsgkgsnl/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGK4 w -",
                [
                    89, 164, 81, 34, 88, 37, 226, 199, 41, 17, 188, 18, 129, 68, 120, 37, 194, 115,
                    74, 132, 99, 149, 136, 143, 101, 148, 8, 67, 106, 107, 191, 126,
                ],
            ),
            (
                "lnsgk4/9/ppppppppp/9/9/9/PPPPPPPPP/9/LNSGK4 w GBRgbr",
                [
                    89, 36, 18, 1, 137, 128, 68, 64, 34, 144, 8, 175, 68, 120, 78, 137, 112, 172,
                    18, 97, 25, 37, 194, 112, 30, 159, 251, 252, 166, 212, 218, 90,
                ],
            ),
        ];

        for (sfen, expected) in cases {
            let pos = parse(sfen);
            let packed = sfen_pack(&pos);
            assert_eq!(packed, expected, "packed sfen mismatch for {sfen}");
        }
    }

    #[test]
    fn ply_does_not_affect_packing() {
        let a =
            parse_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1").unwrap();
        let b =
            parse_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 99").unwrap();
        assert_eq!(sfen_pack(&a), sfen_pack(&b));
    }

    #[test]
    fn side_to_move_flips_bit0() {
        let black =
            parse_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1").unwrap();
        let white =
            parse_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1").unwrap();
        assert_eq!(sfen_pack(&black)[0] & 1, 0);
        assert_eq!(sfen_pack(&white)[0] & 1, 1);
    }

    #[test]
    fn fixture_sfens_pack_to_full_width() {
        // The parity fixtures reach promoted board pieces, both-colour hands
        // and sparse boards, which the transcribed vectors do not.
        for sfen in [
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
            "4k4/9/4r4/9/9/9/4K3B/9/9 b RG2gs2n3p 1",
            "k8/1P7/G8/1N2P4/9/9/9/9/8K b 2PG2pg 1",
            "l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1",
            "4k4/3P3+PL/2N2PR2/1L2BNS2/4N4/9/9/9/4K4 b - 1",
            "9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1",
        ] {
            let pos = parse_sfen(sfen).unwrap();
            let packed = sfen_pack(&pos);
            assert_eq!(packed.len(), 32, "sfen {sfen}");
        }
    }

    /// The bit-identity check in the decoding direction: the transcribed bytes
    /// rebuild the positions they were transcribed for.
    #[test]
    fn reference_cshogi_vectors_decode_back() {
        let cases: [(&str, [u8; 32]); 4] = [
            (
                "lnsgkgsnl/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w -",
                [
                    89, 164, 81, 34, 12, 171, 68, 252, 44, 167, 68, 56, 94, 137, 240, 72, 132, 87,
                    34, 60, 167, 68, 56, 86, 137, 248, 88, 70, 137, 48, 188, 126,
                ],
            ),
            (
                "lns1kgsnl/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w -",
                [
                    89, 164, 81, 34, 12, 171, 68, 252, 44, 167, 68, 56, 94, 137, 240, 72, 4, 18,
                    225, 57, 37, 194, 177, 74, 196, 199, 50, 74, 132, 97, 191, 126,
                ],
            ),
            (
                "lnsgkgsnl/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGK4 w -",
                [
                    89, 164, 81, 34, 88, 37, 226, 199, 41, 17, 188, 18, 129, 68, 120, 37, 194, 115,
                    74, 132, 99, 149, 136, 143, 101, 148, 8, 67, 106, 107, 191, 126,
                ],
            ),
            (
                "lnsgk4/9/ppppppppp/9/9/9/PPPPPPPPP/9/LNSGK4 w GBRgbr",
                [
                    89, 36, 18, 1, 137, 128, 68, 64, 34, 144, 8, 175, 68, 120, 78, 137, 112, 172,
                    18, 97, 25, 37, 194, 112, 30, 159, 251, 252, 166, 212, 218, 90,
                ],
            ),
        ];

        for (sfen, packed) in cases {
            let decoded = position_from_packed_sfen(&packed).expect("a reference vector decodes");
            // Compared through the parser rather than against the literal: the
            // vectors write a hand in a different order than `format_sfen`.
            assert_eq!(
                sfen_without_ply(&decoded),
                sfen_without_ply(&parse(sfen)),
                "decoded {sfen}"
            );
        }
    }

    #[test]
    fn start_handicap_and_tsume_positions_round_trip() {
        assert_round_trips(&Position::startpos(), "the start position");
        for sfen in HANDICAP_SFENS.into_iter().chain(TSUME_SFENS) {
            assert_round_trips(&parse_sfen(sfen).unwrap(), sfen);
        }
    }

    /// Every position reachable in `depth` plies, the root included.
    fn walk(pos: &mut Position, depth: u32, visit: &mut impl FnMut(&Position)) {
        visit(pos);
        if depth == 0 {
            return;
        }
        let mut moves: Vec<Move> = Vec::with_capacity(64);
        pos.generate_legal_all(&mut moves);
        for m in moves {
            let undo = pos.do_move(m);
            walk(pos, depth - 1, visit);
            pos.undo_move(m, undo);
        }
    }

    /// The property the opening book rests on, over every position three plies
    /// from the start: the code round trips, and two positions never share one.
    #[test]
    #[cfg_attr(miri, ignore)] // ~25k positions packed and decoded; far over the miri budget.
    fn perft_positions_round_trip_and_pack_distinctly() {
        let mut coded: HashMap<PackedSfen, String> = HashMap::new();
        let mut visited = 0;
        let mut visit = |pos: &Position| {
            visited += 1;
            let sfen = sfen_without_ply(pos);
            assert_round_trips(pos, &sfen);
            if let Some(other) = coded.insert(sfen_pack(pos), sfen.clone()) {
                assert_eq!(other, sfen, "two positions share one code");
            }
        };
        walk(&mut Position::startpos(), 3, &mut visit);
        // The perft counts of the first three plies, the root included.
        assert_eq!(visited, 1 + 30 + 900 + 25_470);
    }

    /// Both code tables are complete over their widths — every bit path ends at
    /// a piece — so the readers' `UnknownCode` guard stands unreached. It stays
    /// because the alternative to a total decoder is a panic on a table that
    /// stops being complete.
    #[test]
    fn every_bit_path_decodes_to_a_piece() {
        for width in [MAX_BOARD_CODE_BITS, MAX_HAND_CODE_BITS] {
            let board = width == MAX_BOARD_CODE_BITS;
            for path in 0..(1u32 << width) {
                let mut code = 0;
                let mut decoded = false;
                for bits in 1..=width {
                    code |= path & (1 << (bits - 1));
                    if (board && code == 0)
                        || if board {
                            board_kind_of_code(code, bits).is_some()
                        } else {
                            hand_kind_of_code(code, bits).is_some()
                        }
                    {
                        decoded = true;
                        break;
                    }
                }
                assert!(decoded, "no code on the {width}-bit path {path:#08b}");
            }
        }
    }

    #[test]
    fn code_width_bounds_hold_over_the_table() {
        for kind in HAND_ORDER {
            let (_, bits) = huffman_board(kind);
            assert!(bits <= MAX_BOARD_CODE_BITS, "{kind:?} board code");
            assert!(bits - 1 <= MAX_HAND_CODE_BITS, "{kind:?} hand code");
        }
    }

    /// The board and piece-box streams a king-square field of 81 would open —
    /// the reference's "this side has no king" — need a 257th bit, so the
    /// decoder turns the value away rather than reading into a stream that
    /// cannot be there.
    #[test]
    fn rejects_a_king_square_off_the_board() {
        let mut w = BitWriter::new();
        w.write_one_bit(false);
        w.write_n_bit(0, 7);
        w.write_n_bit(Square::COUNT as u32, 7);
        assert_eq!(
            position_from_packed_sfen(&w.data),
            Err(PackedSfenError::KingSquareOutOfRange(81))
        );
    }

    #[test]
    fn rejects_both_kings_on_one_square() {
        let mut w = BitWriter::new();
        w.write_one_bit(false);
        w.write_n_bit(40, 7);
        w.write_n_bit(40, 7);
        assert_eq!(
            position_from_packed_sfen(&w.data),
            Err(PackedSfenError::KingSquaresCollide(40))
        );
    }

    #[test]
    fn rejects_an_all_ones_buffer() {
        assert_eq!(
            position_from_packed_sfen(&[0xff; PACKED_SFEN_LEN]),
            Err(PackedSfenError::KingSquareOutOfRange(127))
        );
    }

    #[test]
    fn rejects_a_nineteenth_pawn() {
        let mut w = BitWriter::new();
        w.write_one_bit(false);
        w.write_n_bit(0, 7);
        w.write_n_bit(1, 7);
        for _ in 0..Square::COUNT - 2 {
            w.write_one_bit(false);
        }
        for _ in 0..19 {
            write_hand_piece(&mut w, PieceKind::Pawn, Color::Black);
        }
        assert_eq!(
            position_from_packed_sfen(&w.data),
            Err(PackedSfenError::TooManyPieces(PieceKind::Pawn))
        );
    }

    #[test]
    fn rejects_a_truncated_stream() {
        let mut packed = sfen_pack(&Position::startpos());
        packed[PACKED_SFEN_LEN / 2..].fill(0);
        assert!(position_from_packed_sfen(&packed).is_err());
    }

    /// Whatever the bits say, the answer is a `Result`. The king squares are
    /// seeded with real ones so the random bits are spent on the board, the
    /// hands and the piece box rather than on the field that guards them.
    #[test]
    #[cfg_attr(miri, ignore)] // 2k decodes; over the miri budget.
    fn random_streams_decode_or_err_but_never_panic() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next_bit = move || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545_f491_4f6c_dd1d) & (1 << 33) != 0
        };

        for seed in 0..2000u32 {
            let mut w = BitWriter::new();
            w.write_one_bit(next_bit());
            w.write_n_bit(seed % Square::COUNT as u32, 7);
            w.write_n_bit((seed + 1 + seed / 81) % Square::COUNT as u32, 7);
            while w.cursor < PACKED_SFEN_BITS {
                w.write_one_bit(next_bit());
            }
            let _ = position_from_packed_sfen(&w.data);
        }
    }
}
