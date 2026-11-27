//! CSA notation: the move and position text of the CSA server protocol
//! (version 1.2.1), converted to and from [`Move`] and [`Position`].
//!
//! A move is seven bytes, `+7776FU`: the mover's sign, the from-square, the
//! to-square and the code of the piece standing on the to-square *after* the
//! move. A drop spells its from-square `00` (`-0055KA`), and a promotion shows
//! the promoted code (`+2822UM`), so telling a promotion from a move of a piece
//! that is already promoted takes the position the move is played in.
//!
//! Squares are written file then rank, each `1..=9`; file `1` is the right-hand
//! file from Black's side and rank `1` the top rank, the same orientation as
//! USI's `1a`.
//!
//! A position is the block of lines a game summary carries between
//! `BEGIN Position` and `END Position`. The standard record format spells the
//! board three ways and this reads all of them: `PI` (the initial position,
//! optionally followed by the pieces taken off it for a handicap), the nine
//! rank lines `P1`..`P9`, and `P+` / `P-` lines placing single pieces — on a
//! square, or in hand as `00<piece>`, or, with `00AL`, every piece not placed
//! elsewhere. A line holding only `+` or `-` names the side to move and ends
//! the position. The `shogi-server` implementation of the protocol writes the
//! nine rank lines, lists each piece in hand as its own `00<piece>` item, and
//! follows the side-to-move line with the moves already played in the game
//! (each as a move with a `,T<seconds>` time suffix); those move lines are
//! handed back to the caller unread, because playing them is the game
//! record's business, not the notation's.
//!
//! Nothing here allocates: input is a byte slice, a move renders into a
//! seven-byte [`CsaMoveBuf`] or onto a [`TextWriter`], and a position is
//! parsed into a caller's [`Position`], reusing what it already holds.

use crate::color::Color;
use crate::move_::Move;
use crate::piece::{Piece, PieceKind};
use crate::position::Position;
use crate::square::Square;
use crate::text::TextWriter;

/// Width of a move's text: sign, two squares, piece code.
pub const CSA_MOVE_LEN: usize = 7;

/// What was wrong with a piece of CSA text.
///
/// A two-byte field — a square's digits, a piece code — is carried as the bytes
/// that arrived, so the message can quote them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CsaError {
    /// A move's text is not [`CSA_MOVE_LEN`] bytes wide.
    MoveLength(usize),
    /// A move or a piece cell starts with neither `+` nor `-`.
    InvalidSign(u8),
    /// A move is signed for the side that is not to move.
    WrongSide,
    /// Two digits that name no board square (or, for a move's from-square, not
    /// the `00` of a drop either).
    InvalidSquare([u8; 2]),
    /// A piece code that is none of the fourteen.
    UnknownPiece([u8; 2]),
    /// A move from a square holding no piece.
    EmptyFromSquare,
    /// A move from a square holding the opponent's piece.
    NotOwnPiece,
    /// A move whose piece code is neither the piece on its from-square nor that
    /// piece promoted.
    PieceMismatch([u8; 2]),
    /// A drop or a piece in hand naming a promoted piece or a king.
    NotDroppable([u8; 2]),
    /// A well-formed move the position does not allow: the move is spelled
    /// correctly and its piece is there, but the rules forbid playing it.
    IllegalMove,
    /// A position line of no form the block allows.
    MalformedLine,
    /// A rank line given twice, or a rank line beside a `PI` line.
    DuplicateBoard,
    /// A board given with some but not all of its nine rank lines.
    MissingRank(u8),
    /// A piece placed on a square that already holds one.
    SquareOccupied([u8; 2]),
    /// A `PI` line taking off a piece the initial position does not have there.
    PieceNotThere([u8; 2]),
    /// More pieces of one kind than a set holds, more than one king a side, or
    /// a second `00AL`.
    TooManyPieces,
    /// The block ended before its side-to-move line.
    MissingSideToMove,
    /// Something other than blank or comment lines after the side-to-move line,
    /// where the caller asked for the position alone.
    TrailingLines,
}

impl CsaError {
    /// This error's message, as the bytes a diagnostic line carries. An input
    /// byte is quoted the way [`TextWriter::quoted_byte`] quotes one, so the
    /// message stays ASCII whatever arrived.
    pub fn write_message(&self, out: &mut TextWriter<'_>) {
        out.bytes(b"csa: ");
        match self {
            Self::MoveLength(n) => {
                out.bytes(b"move is ")
                    .u64(*n as u64)
                    .bytes(b" bytes (expected 7)");
            }
            Self::InvalidSign(b) => {
                out.bytes(b"invalid sign ").quoted_byte(*b);
            }
            Self::WrongSide => {
                out.bytes(b"move is signed for the side not to move");
            }
            Self::InvalidSquare(f) => {
                out.bytes(b"square out of range ");
                quote_pair(out, f);
            }
            Self::UnknownPiece(f) => {
                out.bytes(b"unknown piece code ");
                quote_pair(out, f);
            }
            Self::EmptyFromSquare => {
                out.bytes(b"from-square is empty");
            }
            Self::NotOwnPiece => {
                out.bytes(b"from-square holds the opponent's piece");
            }
            Self::PieceMismatch(f) => {
                out.bytes(b"piece code ");
                quote_pair(out, f);
                out.bytes(b" does not match the piece on the from-square");
            }
            Self::NotDroppable(f) => {
                out.bytes(b"piece ");
                quote_pair(out, f);
                out.bytes(b" cannot be in hand");
            }
            Self::IllegalMove => {
                out.bytes(b"move is not legal in the position");
            }
            Self::MalformedLine => {
                out.bytes(b"malformed position line");
            }
            Self::DuplicateBoard => {
                out.bytes(b"board given more than once");
            }
            Self::MissingRank(r) => {
                out.bytes(b"rank line P")
                    .u64(u64::from(*r))
                    .bytes(b" missing");
            }
            Self::SquareOccupied(f) => {
                out.bytes(b"square ");
                quote_pair(out, f);
                out.bytes(b" already holds a piece");
            }
            Self::PieceNotThere(f) => {
                out.bytes(b"initial position has no such piece on ");
                quote_pair(out, f);
            }
            Self::TooManyPieces => {
                out.bytes(b"more pieces than a set holds");
            }
            Self::MissingSideToMove => {
                out.bytes(b"side-to-move line missing");
            }
            Self::TrailingLines => {
                out.bytes(b"unexpected lines after the side to move");
            }
        }
    }
}

fn quote_pair(out: &mut TextWriter<'_>, pair: &[u8; 2]) {
    if pair.iter().all(|&b| crate::text::is_printable(b)) {
        out.byte(b'\'').bytes(pair).byte(b'\'');
    } else {
        out.quoted_byte(pair[0]).byte(b' ').quoted_byte(pair[1]);
    }
}

/// The piece kind and promotion a two-letter code names.
fn parse_piece_code(code: [u8; 2]) -> Option<(PieceKind, bool)> {
    Some(match &code {
        b"FU" => (PieceKind::Pawn, false),
        b"KY" => (PieceKind::Lance, false),
        b"KE" => (PieceKind::Knight, false),
        b"GI" => (PieceKind::Silver, false),
        b"KI" => (PieceKind::Gold, false),
        b"KA" => (PieceKind::Bishop, false),
        b"HI" => (PieceKind::Rook, false),
        b"OU" => (PieceKind::King, false),
        b"TO" => (PieceKind::Pawn, true),
        b"NY" => (PieceKind::Lance, true),
        b"NK" => (PieceKind::Knight, true),
        b"NG" => (PieceKind::Silver, true),
        b"UM" => (PieceKind::Bishop, true),
        b"RY" => (PieceKind::Rook, true),
        _ => return None,
    })
}

/// The two-letter code of `kind`, promoted or not. Only a promotable kind has a
/// promoted code; asking for a promoted gold or king gets the plain one.
const fn piece_code(kind: PieceKind, promoted: bool) -> &'static [u8; 2] {
    match (kind, promoted) {
        (PieceKind::Pawn, false) => b"FU",
        (PieceKind::Lance, false) => b"KY",
        (PieceKind::Knight, false) => b"KE",
        (PieceKind::Silver, false) => b"GI",
        (PieceKind::Gold, _) => b"KI",
        (PieceKind::Bishop, false) => b"KA",
        (PieceKind::Rook, false) => b"HI",
        (PieceKind::King, _) => b"OU",
        (PieceKind::Pawn, true) => b"TO",
        (PieceKind::Lance, true) => b"NY",
        (PieceKind::Knight, true) => b"NK",
        (PieceKind::Silver, true) => b"NG",
        (PieceKind::Bishop, true) => b"UM",
        (PieceKind::Rook, true) => b"RY",
    }
}

const fn sign_byte(color: Color) -> u8 {
    match color {
        Color::Black => b'+',
        Color::White => b'-',
    }
}

fn parse_sign(byte: u8) -> Result<Color, CsaError> {
    match byte {
        b'+' => Ok(Color::Black),
        b'-' => Ok(Color::White),
        other => Err(CsaError::InvalidSign(other)),
    }
}

/// The board square two digits name, `None` for `00` and anything else that is
/// not one.
fn board_square(file: u8, rank: u8) -> Option<Square> {
    if !(b'1'..=b'9').contains(&file) || !(b'1'..=b'9').contains(&rank) {
        return None;
    }
    Square::new(file - b'1', rank - b'1')
}

/// `sq`'s rank counted from `color`'s far side: `0` is the rank a pawn of that
/// colour can never leave.
const fn relative_rank(sq: Square, color: Color) -> u8 {
    match color {
        Color::Black => sq.rank(),
        Color::White => Square::RANKS - 1 - sq.rank(),
    }
}

/// Whether an unpromoted `kind` of `color` standing on `sq` would still have a
/// move — the rule that keeps a pawn or lance off the last rank and a knight
/// off the last two, whether it arrives by a move or a drop.
const fn has_a_move_from(kind: PieceKind, color: Color, sq: Square) -> bool {
    let r = relative_rank(sq, color);
    match kind {
        PieceKind::Pawn | PieceKind::Lance => r >= 1,
        PieceKind::Knight => r >= 2,
        _ => true,
    }
}

/// Whether the side to move may play `m`.
///
/// [`Position::pseudo_legal`] is the search's guard for a move out of a table,
/// and it leaves to the generators two rules a move from the outside has to be
/// held to as well: a promotion needs its from- or to-square in the enemy
/// camp, and an unpromoted piece may not land where it would never move again.
/// Those are checked here; the rest — the piece's reach, check, pins, nifu,
/// a pawn-drop mate — is the search's own predicates.
fn is_legal_move(pos: &Position, m: Move) -> bool {
    let us = pos.side_to_move();
    let to = m.to_sq();
    let after = m.moved_piece_after();
    if m.is_drop() {
        return has_a_move_from(after.kind, us, to) && pos.pseudo_legal::<true>(m);
    }
    if m.is_promote() {
        if relative_rank(m.from_sq(), us) > 2 && relative_rank(to, us) > 2 {
            return false;
        }
    } else if !after.promoted && !has_a_move_from(after.kind, us, to) {
        return false;
    }
    pos.pseudo_legal::<true>(m) && pos.is_legal(m)
}

/// Parse a move's text such as `+7776FU`, `-0055KA` or `+2822UM`, played in
/// `pos`, into the [`Move`] it is.
///
/// The text is the move alone: a time suffix (`,T12`) that a server appends is
/// the caller's to split off. A move that is spelled correctly but that the
/// rules do not allow in `pos` is [`CsaError::IllegalMove`]; the other errors
/// say what about the text does not fit the position.
pub fn parse_csa_move(text: &[u8], pos: &Position) -> Result<Move, CsaError> {
    let &[sign, f0, r0, f1, r1, c0, c1] = text else {
        return Err(CsaError::MoveLength(text.len()));
    };
    let color = parse_sign(sign)?;
    if color != pos.side_to_move() {
        return Err(CsaError::WrongSide);
    }
    let to = board_square(f1, r1).ok_or(CsaError::InvalidSquare([f1, r1]))?;
    let code = [c0, c1];
    let (kind, promoted) = parse_piece_code(code).ok_or(CsaError::UnknownPiece(code))?;

    let m = if (f0, r0) == (b'0', b'0') {
        if promoted || kind == PieceKind::King {
            return Err(CsaError::NotDroppable(code));
        }
        Move::make_drop(kind, color, to)
    } else {
        let from = board_square(f0, r0).ok_or(CsaError::InvalidSquare([f0, r0]))?;
        let piece = pos.board().get(from).ok_or(CsaError::EmptyFromSquare)?;
        if piece.color != color {
            return Err(CsaError::NotOwnPiece);
        }
        if piece.kind != kind {
            return Err(CsaError::PieceMismatch(code));
        }
        match (piece.promoted, promoted) {
            (false, true) => Move::make_promote(from, to, piece),
            (true, false) => return Err(CsaError::PieceMismatch(code)),
            _ => Move::make(from, to, piece),
        }
    };

    if !is_legal_move(pos, m) {
        return Err(CsaError::IllegalMove);
    }
    Ok(m)
}

/// A buffer one CSA move is written into.
pub struct CsaMoveBuf {
    bytes: [u8; CSA_MOVE_LEN],
}

impl CsaMoveBuf {
    pub const fn new() -> Self {
        Self {
            bytes: [0; CSA_MOVE_LEN],
        }
    }

    /// Write `m` into this buffer and return the text — the inverse of
    /// [`parse_csa_move`].
    ///
    /// A [`Move`] carries its mover and the piece it leaves on the to-square,
    /// which is everything the text spells, so no position is needed. Behaviour
    /// on a non-move sentinel ([`Move::null`], [`Move::resign`], [`Move::win`])
    /// is unspecified: those have no notation as a move.
    pub fn format(&mut self, m: Move) -> &[u8] {
        let after = m.moved_piece_after();
        self.bytes[0] = sign_byte(after.color);
        if m.is_drop() {
            self.bytes[1] = b'0';
            self.bytes[2] = b'0';
        } else {
            let from = m.from_sq();
            self.bytes[1] = b'1' + from.file();
            self.bytes[2] = b'1' + from.rank();
        }
        let to = m.to_sq();
        self.bytes[3] = b'1' + to.file();
        self.bytes[4] = b'1' + to.rank();
        self.bytes[5..7].copy_from_slice(piece_code(after.kind, after.promoted));
        &self.bytes
    }
}

impl Default for CsaMoveBuf {
    fn default() -> Self {
        Self::new()
    }
}

/// Append `m`'s CSA text to `out`, as [`CsaMoveBuf::format`] spells it.
pub fn write_csa_move(m: Move, out: &mut TextWriter<'_>) {
    let mut buf = CsaMoveBuf::new();
    out.bytes(buf.format(m));
}

/// How many of each kind one set holds, indexed by [`PieceKind::index`]. A side
/// has one king; the set's two are counted per colour.
const SET: [u8; PieceKind::COUNT] = [18, 4, 4, 4, 4, 2, 2, 1];

/// The kinds that can be held, in the order a hand is written.
const HAND_ORDER: [PieceKind; 7] = [
    PieceKind::Rook,
    PieceKind::Bishop,
    PieceKind::Gold,
    PieceKind::Silver,
    PieceKind::Knight,
    PieceKind::Lance,
    PieceKind::Pawn,
];

/// The initial position's back rank, from file 9 to file 1.
const BACK_RANK: [PieceKind; 9] = [
    PieceKind::Lance,
    PieceKind::Knight,
    PieceKind::Silver,
    PieceKind::Gold,
    PieceKind::King,
    PieceKind::Gold,
    PieceKind::Silver,
    PieceKind::Knight,
    PieceKind::Lance,
];

/// A position being read: the board, the hands and the counts that keep it to
/// one set, built up line by line before anything is written to the caller's
/// [`Position`].
struct Builder {
    board: [Option<Piece>; Square::COUNT],
    hands: [[u8; PieceKind::COUNT]; Color::COUNT],
    /// Pieces of each kind but the king placed so far, on the board and in
    /// hand together.
    placed: [u8; PieceKind::COUNT],
    kings: [u8; Color::COUNT],
    /// Rank lines seen, bit `r` for `P<r+1>`.
    ranks: u16,
    initial: bool,
    all_rest: Option<Color>,
}

impl Builder {
    const fn new() -> Self {
        Self {
            board: [None; Square::COUNT],
            hands: [[0; PieceKind::COUNT]; Color::COUNT],
            placed: [0; PieceKind::COUNT],
            kings: [0; Color::COUNT],
            ranks: 0,
            initial: false,
            all_rest: None,
        }
    }

    fn count(&mut self, kind: PieceKind, color: Color) -> Result<(), CsaError> {
        let slot = if kind == PieceKind::King {
            &mut self.kings[color.index()]
        } else {
            &mut self.placed[kind.index()]
        };
        if *slot >= SET[kind.index()] {
            return Err(CsaError::TooManyPieces);
        }
        *slot += 1;
        Ok(())
    }

    fn place(&mut self, sq: Square, piece: Piece, field: [u8; 2]) -> Result<(), CsaError> {
        let cell = &mut self.board[sq.index() as usize];
        if cell.is_some() {
            return Err(CsaError::SquareOccupied(field));
        }
        *cell = Some(piece);
        self.count(piece.kind, piece.color)
    }

    fn hold(&mut self, kind: PieceKind, color: Color) -> Result<(), CsaError> {
        self.count(kind, color)?;
        self.hands[color.index()][kind.index()] += 1;
        Ok(())
    }

    fn initial_position(&mut self, removals: &[u8]) -> Result<(), CsaError> {
        if self.initial || self.ranks != 0 {
            return Err(CsaError::DuplicateBoard);
        }
        self.initial = true;
        for color in [Color::Black, Color::White] {
            let rank = |r: u8| match color {
                Color::Black => Square::RANKS - 1 - r,
                Color::White => r,
            };
            // From Black's side the rook is on file 2 and the bishop on file 8,
            // so file index 1 and 7; White's mirror them.
            let (rook_file, bishop_file) = match color {
                Color::Black => (1, 7),
                Color::White => (7, 1),
            };
            for (i, &kind) in BACK_RANK.iter().enumerate() {
                let sq = Square::new(8 - i as u8, rank(0)).unwrap();
                self.place(sq, Piece::new(kind, color), [0; 2])?;
            }
            let rook = Square::new(rook_file, rank(1)).unwrap();
            self.place(rook, Piece::new(PieceKind::Rook, color), [0; 2])?;
            let bishop = Square::new(bishop_file, rank(1)).unwrap();
            self.place(bishop, Piece::new(PieceKind::Bishop, color), [0; 2])?;
            for file in 0..Square::FILES {
                let sq = Square::new(file, rank(2)).unwrap();
                self.place(sq, Piece::new(PieceKind::Pawn, color), [0; 2])?;
            }
        }

        for &[f, r, c0, c1] in items_of(removals)? {
            let field = [f, r];
            let sq = board_square(f, r).ok_or(CsaError::InvalidSquare(field))?;
            let code = [c0, c1];
            let (kind, promoted) = parse_piece_code(code).ok_or(CsaError::UnknownPiece(code))?;
            let cell = &mut self.board[sq.index() as usize];
            match *cell {
                Some(p) if p.kind == kind && p.promoted == promoted => {
                    *cell = None;
                    if kind == PieceKind::King {
                        self.kings[p.color.index()] -= 1;
                    } else {
                        self.placed[kind.index()] -= 1;
                    }
                }
                _ => return Err(CsaError::PieceNotThere(field)),
            }
        }
        Ok(())
    }

    fn rank_line(&mut self, rank: u8, cells: &[u8]) -> Result<(), CsaError> {
        if self.initial || self.ranks & (1 << rank) != 0 {
            return Err(CsaError::DuplicateBoard);
        }
        self.ranks |= 1 << rank;
        // Nine three-byte cells. A writer that trims trailing blanks leaves the
        // last empty cell as ` *`.
        let full = match cells.len() {
            27 => cells,
            26 if cells.ends_with(b" *") => cells,
            _ => return Err(CsaError::MalformedLine),
        };
        for i in 0..Square::FILES as usize {
            let cell = &full[i * 3..(i * 3 + 3).min(full.len())];
            if cell == b" * " || cell == b" *" {
                continue;
            }
            let color = parse_sign(cell[0])?;
            let code = [cell[1], cell[2]];
            let (kind, promoted) = parse_piece_code(code).ok_or(CsaError::UnknownPiece(code))?;
            let piece = piece_of(kind, color, promoted);
            let sq = Square::new(8 - i as u8, rank).unwrap();
            self.place(sq, piece, [b'9' - i as u8, b'1' + rank])?;
        }
        Ok(())
    }

    fn pieces_line(&mut self, color: Color, items: &[u8]) -> Result<(), CsaError> {
        for &[f, r, c0, c1] in items_of(items)? {
            let field = [f, r];
            let code = [c0, c1];
            if field == *b"00" {
                if code == *b"AL" {
                    if self.all_rest.is_some() {
                        return Err(CsaError::TooManyPieces);
                    }
                    self.all_rest = Some(color);
                    continue;
                }
                let (kind, promoted) =
                    parse_piece_code(code).ok_or(CsaError::UnknownPiece(code))?;
                if promoted || kind == PieceKind::King {
                    return Err(CsaError::NotDroppable(code));
                }
                self.hold(kind, color)?;
            } else {
                let sq = board_square(f, r).ok_or(CsaError::InvalidSquare(field))?;
                let (kind, promoted) =
                    parse_piece_code(code).ok_or(CsaError::UnknownPiece(code))?;
                self.place(sq, piece_of(kind, color, promoted), field)?;
            }
        }
        Ok(())
    }

    fn finish(mut self, pos: &mut Position, side_to_move: Color) -> Result<(), CsaError> {
        if self.ranks != 0 && self.ranks != 0x1FF {
            let missing = (0..Square::RANKS).find(|r| self.ranks & (1 << r) == 0);
            return Err(CsaError::MissingRank(missing.unwrap_or(0) + 1));
        }
        if let Some(color) = self.all_rest {
            for kind in HAND_ORDER {
                let rest = SET[kind.index()] - self.placed[kind.index()];
                self.hands[color.index()][kind.index()] += rest;
                self.placed[kind.index()] += rest;
            }
        }

        pos.reset_empty();
        {
            let mut board = pos.board_mut();
            for (i, &piece) in self.board.iter().enumerate() {
                board.set(Square::from_index(i as u8).unwrap(), piece);
            }
        }
        for color in [Color::Black, Color::White] {
            let hand = pos.hand_mut(color);
            for kind in HAND_ORDER {
                for _ in 0..self.hands[color.index()][kind.index()] {
                    hand.increment(kind);
                }
            }
        }
        pos.set_side_to_move(side_to_move);
        pos.set_ply(1);
        // The direct setters above bypass incremental key maintenance.
        pos.refresh_keys();
        Ok(())
    }
}

/// The four-byte `<square><piece>` items a `PI`, `P+` or `P-` line lists.
fn items_of(text: &[u8]) -> Result<&[[u8; 4]], CsaError> {
    match text.as_chunks::<4>() {
        (items, []) => Ok(items),
        _ => Err(CsaError::MalformedLine),
    }
}

const fn piece_of(kind: PieceKind, color: Color, promoted: bool) -> Piece {
    Piece {
        kind,
        color,
        promoted,
    }
}

/// The lines of `block`, each without its line ending (`\n` or `\r\n`).
fn lines(block: &[u8]) -> impl Iterator<Item = &[u8]> {
    block
        .split(|&b| b == b'\n')
        .map(|line| line.strip_suffix(b"\r").unwrap_or(line))
}

/// Whether a line carries nothing a reader must act on: blank, or a `'`
/// comment.
fn is_ignorable(line: &[u8]) -> bool {
    line.is_empty() || line[0] == b'\''
}

/// Parse a game summary's position block — the lines between
/// `BEGIN Position` and `END Position` — into `pos`, and return the part of the
/// block after the side-to-move line: the moves already played, which a
/// server lists there when a game resumes from a position past its start.
///
/// The position is ply 1 with no history. A block with no `PI` and no rank
/// lines starts from an empty board, as the standard format has it. Every
/// piece counts against one set, so a block placing more than a set holds is
/// refused rather than clamped.
///
/// A rejected block leaves `pos` untouched: the block is read whole before
/// `pos` is written.
pub fn parse_csa_position_into<'a>(
    pos: &mut Position,
    block: &'a [u8],
) -> Result<&'a [u8], CsaError> {
    let mut builder = Builder::new();
    let mut rest = block;
    while !rest.is_empty() {
        let (line, after) = match rest.iter().position(|&b| b == b'\n') {
            Some(i) => (&rest[..i], &rest[i + 1..]),
            None => (rest, &rest[rest.len()..]),
        };
        rest = after;
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if is_ignorable(line) {
            continue;
        }
        match line {
            b"+" => {
                builder.finish(pos, Color::Black)?;
                return Ok(rest);
            }
            b"-" => {
                builder.finish(pos, Color::White)?;
                return Ok(rest);
            }
            [b'P', b'I', removals @ ..] => builder.initial_position(removals)?,
            [b'P', b'+', items @ ..] => builder.pieces_line(Color::Black, items)?,
            [b'P', b'-', items @ ..] => builder.pieces_line(Color::White, items)?,
            [b'P', r @ b'1'..=b'9', cells @ ..] => builder.rank_line(r - b'1', cells)?,
            _ => return Err(CsaError::MalformedLine),
        }
    }
    Err(CsaError::MissingSideToMove)
}

/// [`parse_csa_position_into`] for a block that holds a position and nothing
/// after its side-to-move line but blank or comment lines. Those lines are
/// checked after `pos` is written, so a [`CsaError::TrailingLines`] refusal
/// leaves the block's position in `pos`.
pub fn parse_csa_position(pos: &mut Position, block: &[u8]) -> Result<(), CsaError> {
    let rest = parse_csa_position_into(pos, block)?;
    if !lines(rest).all(is_ignorable) {
        return Err(CsaError::TrailingLines);
    }
    Ok(())
}

/// Room [`write_csa_position`] needs for any position: nine 30-byte rank lines,
/// two hand lines of `P±` and at most 38 four-byte items with their line ends,
/// and the side-to-move line.
pub const CSA_POSITION_CAPACITY: usize = 9 * 30 + 2 * 3 + 38 * 4 + 2;

/// Append `pos` as a position block: the nine rank lines, a `P+` / `P-` line
/// for each side holding anything (each piece in hand its own `00` item), and
/// the side-to-move line, every line ended by `\n` — the form `shogi-server`
/// writes.
pub fn write_csa_position(pos: &Position, out: &mut TextWriter<'_>) {
    for rank in 0..Square::RANKS {
        out.byte(b'P').byte(b'1' + rank);
        for file in (0..Square::FILES).rev() {
            match pos.board().get(Square::new(file, rank).unwrap()) {
                None => {
                    out.bytes(b" * ");
                }
                Some(p) => {
                    out.byte(sign_byte(p.color))
                        .bytes(piece_code(p.kind, p.promoted));
                }
            }
        }
        out.byte(b'\n');
    }
    for color in [Color::Black, Color::White] {
        let hand = pos.hand(color);
        if HAND_ORDER.iter().all(|&k| hand.count(k) == 0) {
            continue;
        }
        out.byte(b'P').byte(sign_byte(color));
        for kind in HAND_ORDER {
            for _ in 0..hand.count(kind) {
                out.bytes(b"00").bytes(piece_code(kind, false));
            }
        }
        out.byte(b'\n');
    }
    out.byte(sign_byte(pos.side_to_move())).byte(b'\n');
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sfen::{STARTPOS_SFEN, parse_sfen};

    /// The initial position as `shogi-server` writes it, every empty cell
    /// three bytes wide.
    const STARTPOS_CSA: &[u8] = concat!(
        "P1-KY-KE-GI-KI-OU-KI-GI-KE-KY\n",
        "P2 * -HI *  *  *  *  * -KA * \n",
        "P3-FU-FU-FU-FU-FU-FU-FU-FU-FU\n",
        "P4 *  *  *  *  *  *  *  *  * \n",
        "P5 *  *  *  *  *  *  *  *  * \n",
        "P6 *  *  *  *  *  *  *  *  * \n",
        "P7+FU+FU+FU+FU+FU+FU+FU+FU+FU\n",
        "P8 * +KA *  *  *  *  * +HI * \n",
        "P9+KY+KE+GI+KI+OU+KI+GI+KE+KY\n",
        "+\n",
    )
    .as_bytes();

    fn csa_position(block: &[u8]) -> Result<Position, CsaError> {
        let mut pos = Position::empty();
        parse_csa_position(&mut pos, block)?;
        Ok(pos)
    }

    fn render_position(pos: &Position) -> Vec<u8> {
        let mut bytes = [0u8; CSA_POSITION_CAPACITY];
        let mut out = TextWriter::new(&mut bytes);
        write_csa_position(pos, &mut out);
        assert!(!out.overflowed());
        out.as_bytes().to_vec()
    }

    fn render_move(m: Move) -> [u8; CSA_MOVE_LEN] {
        let mut buf = CsaMoveBuf::new();
        buf.format(m).try_into().unwrap()
    }

    fn legal_moves(pos: &Position) -> Vec<Move> {
        let mut out = Vec::new();
        pos.generate_legal_all(&mut out);
        out
    }

    fn message(err: CsaError) -> Vec<u8> {
        let mut bytes = [0u8; 128];
        let mut out = TextWriter::new(&mut bytes);
        err.write_message(&mut out);
        assert!(out.as_bytes().is_ascii());
        out.as_bytes().to_vec()
    }

    /// Positions covering every piece kind on the move for both colours:
    /// openings, promotions waiting on both sides, drops from full hands,
    /// a check to answer, and pawn drops that nifu and a pawn-drop mate forbid.
    const SFENS: &[&[u8]] = &[
        STARTPOS_SFEN,
        b"lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
        // Black's bishop can take on 2b, promoting or not.
        b"lnsgkgsnl/1r5b1/pppppp1pp/6p2/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL b - 5",
        // Promoted pieces of every kind for both sides, and hands.
        b"9/1+P1+L1+N1+S1/4k4/1+B5+R1/9/1+r5+b1/4K4/1+s1+n1+l1+p1/9 b GPgp 1",
        b"9/1+P1+L1+N1+S1/4k4/1+B5+R1/9/1+r5+b1/4K4/1+s1+n1+l1+p1/9 w GPgp 1",
        // Pawns, lances and knights about to reach the last ranks.
        b"4k4/PL7/2P6/3N5/9/5n3/6p2/7lp/4K4 b - 1",
        b"4k4/PL7/2P6/3N5/9/5n3/6p2/7lp/4K4 w - 1",
        // Full hands of every droppable kind.
        b"4k4/9/9/9/9/9/9/9/4K4 b RBGSNLP2rb3g3s3n3l15p 1",
        b"4k4/9/9/9/9/9/9/9/4K4 w RBGSNLP2rb3g3s3n3l15p 1",
        // Black in check from a rook, with drops and moves to interpose.
        b"4r3k/9/9/9/9/9/9/3G1G3/4K4 b GSp 1",
        // A pawn drop on 1b would mate (forbidden); a pawn already on file 2.
        b"7nk/7s1/8G/9/7P1/9/9/9/K8 b P 1",
    ];

    fn positions() -> Vec<Position> {
        SFENS.iter().map(|s| parse_sfen(s).unwrap()).collect()
    }

    #[test]
    fn known_moves_render_as_the_protocol_spells_them() {
        let pos = Position::startpos();
        let m = parse_csa_move(b"+7776FU", &pos).unwrap();
        assert_eq!(&render_move(m), b"+7776FU");
        assert_eq!(m, crate::parse_usi_move(b"7g7f", &pos).unwrap());

        let pos = parse_sfen(SFENS[2]).unwrap();
        let promote = parse_csa_move(b"+8822UM", &pos).unwrap();
        assert!(promote.is_promote());
        assert_eq!(promote, crate::parse_usi_move(b"8h2b+", &pos).unwrap());
        let plain = parse_csa_move(b"+8822KA", &pos).unwrap();
        assert!(!plain.is_promote());

        let pos = parse_sfen(SFENS[8]).unwrap();
        let drop = parse_csa_move(b"-0055KA", &pos).unwrap();
        assert!(drop.is_drop());
        assert_eq!(&render_move(drop), b"-0055KA");
        assert_eq!(drop, crate::parse_usi_move(b"B*5e", &pos).unwrap());
    }

    #[test]
    fn a_promoted_piece_moves_under_its_promoted_code() {
        let pos = parse_sfen(SFENS[3]).unwrap();
        let m = parse_csa_move(b"+8483UM", &pos).unwrap();
        assert!(!m.is_promote());
        assert_eq!(&render_move(m), b"+8483UM");
        assert_eq!(
            parse_csa_move(b"+8483KA", &pos),
            Err(CsaError::PieceMismatch(*b"KA"))
        );
    }

    /// The fourteen piece codes.
    const CODES: [&[u8; 2]; 14] = [
        b"FU", b"KY", b"KE", b"GI", b"KI", b"KA", b"HI", b"OU", b"TO", b"NY", b"NK", b"NG", b"UM",
        b"RY",
    ];

    /// Every legal move of every test position comes back from its own text,
    /// and the moves cover every piece code for both colours, promotions and
    /// drops.
    #[test]
    fn every_legal_move_round_trips() {
        let mut seen = [[false; Color::COUNT]; CODES.len()];
        let mut promotions = 0;
        let mut drops = 0;
        for pos in positions() {
            for m in legal_moves(&pos) {
                let text = render_move(m);
                assert_eq!(parse_csa_move(&text, &pos), Ok(m), "{text:?}");
                let code = CODES.iter().position(|c| c[..] == text[5..7]).unwrap();
                seen[code][pos.side_to_move().index()] = true;
                promotions += usize::from(!m.is_drop() && m.is_promote());
                drops += usize::from(m.is_drop());
            }
        }
        for (code, colours) in CODES.iter().zip(seen) {
            assert_eq!(colours, [true, true], "{code:?} not moved by both colours");
        }
        assert!(promotions > 0 && drops > 0);
    }

    /// Every seven-byte text a side to move could send — each of its pieces to
    /// every square under every code, and every drop — is accepted exactly when
    /// it is one of the position's legal moves.
    #[test]
    #[cfg_attr(miri, ignore = "tens of thousands of candidates per position")]
    fn only_legal_moves_parse() {
        for pos in positions() {
            let legal = legal_moves(&pos);
            let us = pos.side_to_move();
            let mut froms = vec![*b"00"];
            for i in 0..Square::COUNT as u8 {
                let sq = Square::from_index(i).unwrap();
                if pos.board().get(sq).is_some_and(|p| p.color == us) {
                    froms.push([b'1' + sq.file(), b'1' + sq.rank()]);
                }
            }
            let mut accepted = 0;
            for from in &froms {
                for to in 0..Square::COUNT as u8 {
                    let to = Square::from_index(to).unwrap();
                    for code in CODES {
                        let text = [
                            sign_byte(us),
                            from[0],
                            from[1],
                            b'1' + to.file(),
                            b'1' + to.rank(),
                            code[0],
                            code[1],
                        ];
                        if let Ok(m) = parse_csa_move(&text, &pos) {
                            assert!(legal.contains(&m), "{text:?} parsed but is not legal");
                            accepted += 1;
                        }
                    }
                }
            }
            assert_eq!(accepted, legal.len());
        }
    }

    #[test]
    fn each_move_error_is_reported() {
        let start = Position::startpos();
        let cases: &[(&[u8], CsaError)] = &[
            (b"+7776F", CsaError::MoveLength(6)),
            (b"+7776FUX", CsaError::MoveLength(8)),
            (b"*7776FU", CsaError::InvalidSign(b'*')),
            (b"-3334FU", CsaError::WrongSide),
            (b"+7770FU", CsaError::InvalidSquare(*b"70")),
            (b"+7076FU", CsaError::InvalidSquare(*b"70")),
            (b"+7776XX", CsaError::UnknownPiece(*b"XX")),
            (b"+5556FU", CsaError::EmptyFromSquare),
            (b"+3334FU", CsaError::NotOwnPiece),
            (b"+7776KY", CsaError::PieceMismatch(*b"KY")),
            (b"+0055FU", CsaError::IllegalMove),
            (b"+7775FU", CsaError::IllegalMove),
            (b"+7776TO", CsaError::IllegalMove),
            (b"+5949OU", CsaError::IllegalMove),
        ];
        for &(text, expected) in cases {
            assert_eq!(parse_csa_move(text, &start), Err(expected), "{text:?}");
            assert!(message(expected).starts_with(b"csa: "));
        }

        let hands = parse_sfen(SFENS[7]).unwrap();
        assert_eq!(
            parse_csa_move(b"+0055TO", &hands),
            Err(CsaError::NotDroppable(*b"TO"))
        );
        assert_eq!(
            parse_csa_move(b"+0055OU", &hands),
            Err(CsaError::NotDroppable(*b"OU"))
        );
        // A pawn or lance drop on the last rank, a knight on the last two.
        assert_eq!(
            parse_csa_move(b"+0011FU", &hands),
            Err(CsaError::IllegalMove)
        );
        assert_eq!(
            parse_csa_move(b"+0012KE", &hands),
            Err(CsaError::IllegalMove)
        );
        assert_eq!(
            parse_csa_move(b"+0011KY", &hands),
            Err(CsaError::IllegalMove)
        );
        // A promotion with neither square in the enemy camp.
        let start_bishop = parse_sfen(SFENS[2]).unwrap();
        assert_eq!(
            parse_csa_move(b"+8877UM", &start_bishop),
            Err(CsaError::IllegalMove)
        );
        // Nifu and a pawn-drop mate.
        let pawn = parse_sfen(SFENS[10]).unwrap();
        assert_eq!(
            parse_csa_move(b"+0023FU", &pawn),
            Err(CsaError::IllegalMove)
        );
        assert_eq!(
            parse_csa_move(b"+0012FU", &pawn),
            Err(CsaError::IllegalMove)
        );
        assert!(parse_csa_move(b"+0033FU", &pawn).is_ok());
    }

    #[test]
    fn a_non_ascii_field_is_quoted_in_ascii() {
        let err = parse_csa_move(b"+7776\x82\xa0", &Position::startpos()).unwrap_err();
        assert_eq!(err, CsaError::UnknownPiece([0x82, 0xa0]));
        assert_eq!(message(err), b"csa: unknown piece code 0x82 0xa0");
        assert_eq!(
            message(CsaError::UnknownPiece(*b"XX")),
            b"csa: unknown piece code 'XX'"
        );
    }

    #[test]
    fn the_start_position_reads_and_writes_as_the_server_spells_it() {
        let pos = csa_position(STARTPOS_CSA).unwrap();
        assert_eq!(pos, Position::startpos());
        assert_eq!(pos.key(), Position::startpos().key());
        assert_eq!(render_position(&pos), STARTPOS_CSA);

        let pi = csa_position(b"PI\n+\n").unwrap();
        assert_eq!(pi, Position::startpos());
    }

    #[test]
    fn pi_takes_off_the_named_pieces() {
        let pos = csa_position(b"PI82HI22KA\n-\n").unwrap();
        let expected =
            parse_sfen(b"lnsgkgsnl/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1").unwrap();
        assert_eq!(pos, expected);
        assert_eq!(pos.key(), expected.key());
        assert_eq!(
            csa_position(b"PI82KA\n-\n"),
            Err(CsaError::PieceNotThere(*b"82"))
        );
    }

    #[test]
    fn an_arbitrary_position_with_pieces_in_hand_round_trips() {
        let sfen: &[u8] =
            b"ln1g3+Rl/2s1kg3/p1pppp2p/6p2/1p7/2P6/PP1PPPP1P/1B1K2S2/LN1G1G1NL w RBSPsnp 1";
        let expected = parse_sfen(sfen).unwrap();
        let text = render_position(&expected);
        let pos = csa_position(&text).unwrap();
        assert_eq!(pos, expected);
        assert_eq!(pos.key(), expected.key());
        assert_eq!(render_position(&pos), text);
        for m in legal_moves(&pos) {
            assert_eq!(parse_csa_move(&render_move(m), &pos), Ok(m));
        }
    }

    #[test]
    fn single_pieces_and_al_fill_the_hands() {
        // A mating-problem shape: three pieces placed, the attacker holding a
        // gold, and every other piece but the kings to the defender.
        let pos = csa_position(b"P-51OU\nP+53TO59OU\nP+00KI\nP-00AL\n+\n").unwrap();
        let expected = parse_sfen(b"4k4/9/4+P4/9/9/9/9/9/4K4 b G2r2b3g4s4n4l17p 1").unwrap();
        assert_eq!(pos, expected);
        assert_eq!(pos.key(), expected.key());
    }

    #[test]
    fn the_rest_of_the_block_is_handed_back() {
        let mut block = STARTPOS_CSA.to_vec();
        block.extend_from_slice(b"+2726FU,T12\n-3334FU,T6\n");
        let mut pos = Position::empty();
        let rest = parse_csa_position_into(&mut pos, &block).unwrap();
        assert_eq!(rest, b"+2726FU,T12\n-3334FU,T6\n");
        assert_eq!(
            parse_csa_position(&mut pos, &block),
            Err(CsaError::TrailingLines)
        );
        let mut commented = STARTPOS_CSA.to_vec();
        commented.extend_from_slice(b"' a comment\n\n");
        assert!(parse_csa_position(&mut pos, &commented).is_ok());
    }

    #[test]
    fn crlf_comments_and_a_trimmed_last_cell_are_read() {
        let mut text = Vec::new();
        text.extend_from_slice(b"' start\r\n");
        for line in lines(STARTPOS_CSA) {
            text.extend_from_slice(line.strip_suffix(b" ").unwrap_or(line));
            text.extend_from_slice(b"\r\n");
        }
        assert_eq!(csa_position(&text).unwrap(), Position::startpos());
    }

    #[test]
    fn each_position_error_is_reported() {
        let empty_rank = b" *  *  *  *  *  *  *  *  * ";
        let mut nine_ranks = Vec::new();
        for r in 1..=9u8 {
            nine_ranks.extend_from_slice(&[b'P', b'0' + r]);
            nine_ranks.extend_from_slice(empty_rank);
            nine_ranks.push(b'\n');
        }
        let with = |extra: &[u8]| {
            let mut v = nine_ranks.clone();
            v.extend_from_slice(extra);
            v
        };
        let cases: Vec<(Vec<u8>, CsaError)> = vec![
            (b"PX\n+\n".to_vec(), CsaError::MalformedLine),
            (b"+7776FU\n+\n".to_vec(), CsaError::MalformedLine),
            (b"P1 *  * \n+\n".to_vec(), CsaError::MalformedLine),
            (b"P+00F\n+\n".to_vec(), CsaError::MalformedLine),
            (b"PI82H\n+\n".to_vec(), CsaError::MalformedLine),
            (b"P+00XX\n+\n".to_vec(), CsaError::UnknownPiece(*b"XX")),
            (b"P+0aFU\n+\n".to_vec(), CsaError::InvalidSquare(*b"0a")),
            (b"P+50FU\n+\n".to_vec(), CsaError::InvalidSquare(*b"50")),
            (b"P+00TO\n+\n".to_vec(), CsaError::NotDroppable(*b"TO")),
            (b"P+00OU\n+\n".to_vec(), CsaError::NotDroppable(*b"OU")),
            (
                b"P+55FU55FU\n+\n".to_vec(),
                CsaError::SquareOccupied(*b"55"),
            ),
            (b"P+59OU58OU\n+\n".to_vec(), CsaError::TooManyPieces),
            (b"P+00KA00KA00KA\n+\n".to_vec(), CsaError::TooManyPieces),
            (b"P+00AL\nP-00AL\n+\n".to_vec(), CsaError::TooManyPieces),
            (b"PI\nPI\n+\n".to_vec(), CsaError::DuplicateBoard),
            (with(b"PI\n+\n"), CsaError::DuplicateBoard),
            (
                with(b"P5 *  *  *  *  *  *  *  *  * \n+\n"),
                CsaError::DuplicateBoard,
            ),
            (
                b"P1 *  *  *  *  *  *  *  *  * \n+\n".to_vec(),
                CsaError::MissingRank(2),
            ),
            (
                b"P1 *  *  *  * *FU *  *  *  * \n+\n".to_vec(),
                CsaError::InvalidSign(b'*'),
            ),
            (with(b""), CsaError::MissingSideToMove),
            (b"PI55FU\n+\n".to_vec(), CsaError::PieceNotThere(*b"55")),
        ];
        for (block, expected) in cases {
            let mut pos = Position::startpos();
            assert_eq!(
                parse_csa_position(&mut pos, &block),
                Err(expected),
                "{:?}",
                String::from_utf8_lossy(&block)
            );
            assert_eq!(
                pos,
                Position::startpos(),
                "a refused block leaves pos alone"
            );
            assert!(message(expected).starts_with(b"csa: "));
        }
    }

    #[test]
    fn every_error_has_a_message() {
        let all = [
            CsaError::MoveLength(3),
            CsaError::InvalidSign(b'x'),
            CsaError::WrongSide,
            CsaError::InvalidSquare(*b"0a"),
            CsaError::UnknownPiece(*b"ZZ"),
            CsaError::EmptyFromSquare,
            CsaError::NotOwnPiece,
            CsaError::PieceMismatch(*b"KY"),
            CsaError::NotDroppable(*b"OU"),
            CsaError::IllegalMove,
            CsaError::MalformedLine,
            CsaError::DuplicateBoard,
            CsaError::MissingRank(4),
            CsaError::SquareOccupied(*b"55"),
            CsaError::PieceNotThere(*b"82"),
            CsaError::TooManyPieces,
            CsaError::MissingSideToMove,
            CsaError::TrailingLines,
        ];
        for err in all {
            assert!(message(err).len() > b"csa: ".len(), "{err:?}");
        }
        assert_eq!(
            message(CsaError::MissingRank(4)),
            b"csa: rank line P4 missing"
        );
    }
}
