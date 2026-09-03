use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::color::Color;
use crate::hand::Hand;
use crate::move_::Move;
use crate::movegen::try_find_king;
use crate::piece::{Piece, PieceKind};
use crate::square::Square;

/// The seven piece kinds that can be held in hand.
const HAND_KINDS: [PieceKind; 7] = [
    PieceKind::Pawn,
    PieceKind::Lance,
    PieceKind::Knight,
    PieceKind::Silver,
    PieceKind::Gold,
    PieceKind::Bishop,
    PieceKind::Rook,
];

/// Maximum look-back, in plies, for repetition detection
/// (`Position::max_repetition_ply`, `position.cpp`). Searching further back is
/// rarely productive and measurably slows the engine, so the walk is capped
/// here regardless of `plies_from_null`.
const MAX_REPETITION_PLY: i32 = 16;

/// Classification of a repeated position (`RepetitionState`, `types.h`).
///
/// `Win` and `Lose` are from the viewpoint of the **side to move** in the
/// position asked about, not of whoever delivered the checks. Getting this
/// backwards flips the sign of the score in search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RepetitionState {
    /// Not a repetition.
    None,
    /// Perpetual check delivered by the opponent, so the side to move wins.
    Win,
    /// Perpetual check delivered by the side to move, which therefore loses.
    Lose,
    /// Ordinary (non-perpetual-check) fourfold-style repetition.
    Draw,
    /// Same board, side to move holds the strictly superior hand.
    Superior,
    /// Same board, side to move holds the strictly inferior hand.
    Inferior,
}

/// `true` iff `superior` holds an equal-or-greater count of **every** hand
/// piece kind than `inferior` (`hand_is_equal_or_superior`, `types.h`).
fn hand_is_equal_or_superior(superior: &Hand, inferior: &Hand) -> bool {
    HAND_KINDS
        .iter()
        .all(|&k| superior.count(k) >= inferior.count(k))
}

/// The piece sitting on `from` *before* a board move `m`, given the piece that
/// will occupy `to` after it. Drops have no `from` square and never reach here.
fn piece_before_move(m: Move, piece_after: Piece) -> Piece {
    if m.is_promote() {
        Piece::new(piece_after.kind, piece_after.color)
    } else {
        piece_after
    }
}

/// Is `side_to_move`'s king attacked on the just-updated board? An absent king
/// — a pseudo-legal probe move captured it — is `false`, matching the
/// predicate this checks.
#[cfg(debug_assertions)]
fn post_move_gives_check(board: &Board, side_to_move: Color) -> bool {
    match try_find_king(board, side_to_move) {
        Some(king_sq) => crate::movegen::is_attacked_by(board, king_sq, side_to_move.flip()),
        None => false,
    }
}

/// Snapshot pushed onto `Position::history` after every `do_move`, porting the
/// reference `StateInfo` (`position.cpp`).
///
/// It stores no board copy: `undo_move` reconstructs the board by
/// reverse-applying the move, and position identity is carried by `board_key`.
/// That inherits the reference's accepted hash-collision risk — two boards
/// hashing equal with equal hands compare equal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StateInfo {
    pub hands: [Hand; Color::COUNT],
    pub side_to_move: Color,
    /// Whether the side that just moved put the now-side-to-move in check.
    pub gives_check: bool,
    pub board_key: u64,
    /// Plies since the last null move or the setup root. Bounds the repetition
    /// look-back to `min(16, plies_from_null)`, so detection never crosses a
    /// null move.
    pub plies_from_null: i32,
    /// The run, counted in twos, of consecutive checking moves by each colour
    /// ending at this state.
    pub continuous_check: [u16; Color::COUNT],
    /// Signed ply distance to the previous occurrence of this position, `0` if
    /// none. Negative once `repetition_times` reaches the forced fourfold.
    pub repetition: i32,
    pub repetition_times: i32,
    pub repetition_type: RepetitionState,
}

#[derive(Debug, Clone)]
pub struct Position {
    board: Board,
    hands: [Hand; Color::COUNT],
    side_to_move: Color,
    ply: u16,
    history: Vec<StateInfo>,
    /// Snapshot of the position before the first move of the current line.
    /// `history` records only post-move states, so the root is not otherwise
    /// recoverable, yet `is_repetition` must be able to look back to it.
    /// Captured lazily on the first `do_move` of an empty history.
    root: Option<StateInfo>,
    /// Zobrist board hash: XOR of per-`(piece, square)` terms plus the
    /// side-to-move term. See [`crate::key`] for the scheme.
    board_key: u64,
    /// Zobrist hand hash: sum of per-`(color, kind)` steps, one per held copy.
    hand_key: u64,
    /// Partial position key over board pawns only — a promoted pawn is not a
    /// pawn here, and pieces in hand never contribute
    /// (`StateInfo::pawnKey`). Seeded with the non-zero [`crate::key::no_pawns`]
    /// value.
    pawn_key: u64,
    /// Partial position key over board minor pieces, per
    /// [`crate::key::is_minor_piece`] (`StateInfo::minorPieceKey`).
    minor_piece_key: u64,
    /// Partial position keys over each colour's non-pawn board pieces, king
    /// included (`StateInfo::nonPawnKey`). Indexed by [`Color::index`].
    non_pawn_key: [u64; Color::COUNT],
    // The reference also maintains an additive `materialKey`, but nothing in
    // its search or move ordering reads it, so it is not ported.
    /// Per-state check info of the current position
    /// (`StateInfo::{checkSquares, blockersForKing}`), read by the predicates in
    /// [`crate::search_movegen`]. Recomputed on every state change. Excluded
    /// from [`PartialEq`] as a derived cache.
    check_info: crate::search_movegen::CheckInfo,
    /// The stack of parent [`Self::check_info`] values, so `undo_move` /
    /// `undo_null_move` restore the parent's in O(1). Excluded from
    /// [`PartialEq`] as auxiliary cache state.
    check_info_stack: Vec<crate::search_movegen::CheckInfo>,
}

/// Structural equality over the primary state alone. The keys are a derived
/// cache, so a position built by the direct setters — which leave it stale until
/// `refresh_keys` — still compares equal to the same state built through
/// `do_move`. `root` is excluded for the same reason: a position that has taken
/// a move-then-undo round trip must compare equal to the untouched snapshot it
/// was cloned from.
impl PartialEq for Position {
    fn eq(&self, other: &Self) -> bool {
        self.board == other.board
            && self.hands == other.hands
            && self.side_to_move == other.side_to_move
            && self.ply == other.ply
            && self.history == other.history
    }
}

impl Eq for Position {}

impl Position {
    pub const fn empty() -> Self {
        // An empty board with Black (the side whose term is *not* XORed in) to
        // move and no hand pieces hashes to zero on both halves. The partial
        // keys carry their empty-board values: `pawn_key` seeds with the
        // non-zero `noPawns` constant, the others with zero.
        Self {
            board: Board::empty(),
            hands: [Hand::empty(), Hand::empty()],
            side_to_move: Color::Black,
            ply: 1,
            history: Vec::new(),
            root: None,
            board_key: 0,
            hand_key: 0,
            pawn_key: crate::key::NO_PAWNS_SEED,
            minor_piece_key: 0,
            non_pawn_key: [0; Color::COUNT],
            // The empty board's check info (no kings, no attacks); recomputed
            // eagerly on the first state change (SFEN parse / `do_move`).
            check_info: crate::search_movegen::CheckInfo::EMPTY,
            check_info_stack: Vec::new(),
        }
    }

    pub fn startpos() -> Self {
        crate::sfen::parse_sfen(crate::sfen::STARTPOS_SFEN)
            .expect("STARTPOS_SFEN is a hard-coded valid sfen")
    }

    pub const fn board(&self) -> &Board {
        &self.board
    }

    /// Direct mutable access to the board, through a [`BoardMut`] guard.
    ///
    /// **Warning:** a mutation through this guard leaves the cached `board_key`
    /// stale; call `refresh_keys` afterwards, as SFEN parsing does. The guard
    /// does recompute the check info on drop.
    pub fn board_mut(&mut self) -> BoardMut<'_> {
        BoardMut { pos: self }
    }

    pub const fn hand(&self, color: Color) -> &Hand {
        &self.hands[color.index()]
    }

    /// Direct mutable access to a color's hand.
    ///
    /// **Warning:** a mutation through this reference leaves the cached
    /// `hand_key` stale; call `refresh_keys` afterwards.
    pub fn hand_mut(&mut self, color: Color) -> &mut Hand {
        &mut self.hands[color.index()]
    }

    pub const fn side_to_move(&self) -> Color {
        self.side_to_move
    }

    /// Locate `color`'s king, or `None` when that side has no king on the board
    /// (`king_square`).
    pub fn king_square(&self, color: Color) -> Option<Square> {
        try_find_king(&self.board, color)
    }

    /// Set the side to move directly.
    ///
    /// **Warning:** the side-to-move term is part of `board_key`, so this leaves
    /// the cached key stale; call `refresh_keys` afterwards.
    pub fn set_side_to_move(&mut self, color: Color) {
        self.side_to_move = color;
        self.check_info = self.compute_check_info();
    }

    pub const fn ply(&self) -> u16 {
        self.ply
    }

    /// Set the ply counter directly. Unlike the other direct mutators this needs
    /// no `refresh_keys`: `ply` is not part of the Zobrist key.
    pub const fn set_ply(&mut self, ply: u16) {
        self.ply = ply;
    }

    /// Plies since the last null move or the setup root, `0` before any move
    /// (`state()->pliesFromNull`).
    pub fn plies_from_null(&self) -> i32 {
        self.current_plies_from_null()
    }

    /// Full 64-bit Zobrist key: `board_key ^ hand_key`. A pure function of
    /// `(board, hands, side_to_move)`.
    pub const fn key(&self) -> u64 {
        self.board_key ^ self.hand_key
    }

    /// Board half of the key: XOR of per-`(piece, square)` terms plus the
    /// side-to-move term.
    pub const fn board_key(&self) -> u64 {
        self.board_key
    }

    /// The per-state [`crate::search_movegen::CheckInfo`] of the current
    /// position.
    pub(crate) fn check_info(&self) -> &crate::search_movegen::CheckInfo {
        debug_assert_eq!(
            self.check_info.board_key,
            self.board_key(),
            "cached check_info is stale relative to the current board_key",
        );
        &self.check_info
    }

    /// Hand half of the key: sum of per-`(color, kind)` steps.
    pub const fn hand_key(&self) -> u64 {
        self.hand_key
    }

    /// Partial key over board pawns only.
    pub const fn pawn_key(&self) -> u64 {
        self.pawn_key
    }

    /// Partial key over board minor pieces.
    pub const fn minor_piece_key(&self) -> u64 {
        self.minor_piece_key
    }

    /// Partial key over `color`'s board non-pawn pieces, king included.
    pub const fn non_pawn_key(&self, color: Color) -> u64 {
        self.non_pawn_key[color.index()]
    }

    /// XOR `piece`'s `psq` term into the partial keys it belongs to
    /// (`xor_piece_for_partial_key`, `position.cpp`). XOR is self-inverse, so
    /// the same call both places and removes a piece.
    fn xor_piece_partial(&mut self, piece: Piece, sq: Square) {
        let term = crate::key::psq(piece, sq);
        if piece.kind == PieceKind::Pawn && !piece.promoted {
            self.pawn_key ^= term;
        } else {
            if crate::key::is_minor_piece(piece) {
                self.minor_piece_key ^= term;
            }
            self.non_pawn_key[piece.color.index()] ^= term;
        }
    }

    /// Recompute `(board_key, hand_key)` from scratch over the current board,
    /// hands and side-to-move.
    fn recomputed_keys(&self) -> (u64, u64) {
        let mut board_key = 0u64;
        for index in 0..Square::COUNT as u8 {
            let sq = Square::from_index(index).unwrap();
            if let Some(piece) = self.board.get(sq) {
                board_key ^= crate::key::psq(piece, sq);
            }
        }
        if self.side_to_move == Color::White {
            board_key ^= crate::key::side();
        }

        let mut hand_key = 0u64;
        for color in [Color::Black, Color::White] {
            for kind in HAND_KINDS {
                let n = self.hands[color.index()].count(kind) as u64;
                hand_key =
                    hand_key.wrapping_add(crate::key::hand_step(color, kind).wrapping_mul(n));
            }
        }
        (board_key, hand_key)
    }

    /// Recompute the three partial keys from scratch over the current board,
    /// mirroring the per-piece walk in `Position::set` (`position.cpp`).
    fn recomputed_partial_keys(&self) -> (u64, u64, [u64; Color::COUNT]) {
        let mut scratch = Position::empty();
        for index in 0..Square::COUNT as u8 {
            let sq = Square::from_index(index).unwrap();
            if let Some(piece) = self.board.get(sq) {
                scratch.xor_piece_partial(piece, sq);
            }
        }
        (
            scratch.pawn_key,
            scratch.minor_piece_key,
            scratch.non_pawn_key,
        )
    }

    /// Reseed the stored keys and check info from a from-scratch recomputation.
    /// The direct board / hand / side mutators leave them stale, so anything
    /// that uses one must call this before the position is read.
    pub(crate) fn refresh_keys(&mut self) {
        let (board_key, hand_key) = self.recomputed_keys();
        self.board_key = board_key;
        self.hand_key = hand_key;
        let (pawn_key, minor_piece_key, non_pawn_key) = self.recomputed_partial_keys();
        self.pawn_key = pawn_key;
        self.minor_piece_key = minor_piece_key;
        self.non_pawn_key = non_pawn_key;
        self.check_info = self.compute_check_info();
    }

    /// Play `m`, deciding check status from the parent's cached check info
    /// before touching the board — `do_move(m, newSt)` (`position.h`).
    pub fn do_move(&mut self, m: Move) -> Undo {
        let gc = self.gives_check(m);
        self.do_move_with_check(m, gc)
    }

    /// Play `m` given the pre-computed `gives_check` predicate
    /// (`do_move(m, newSt, givesCheck)`, `position.h`). `gives_check` must equal
    /// `self.gives_check(m)` evaluated from the pre-move position.
    pub fn do_move_with_check(&mut self, m: Move, gives_check: bool) -> Undo {
        // Snapshot the pre-move state as the lookback root, so it always
        // reflects the actual head of the current line.
        if self.history.is_empty() {
            self.root = Some(self.root_state());
        }

        // The state this move is played from — the look-back "previous" whose
        // `plies_from_null` / `continuous_check` the new state extends. It is the
        // top of the history stack, or the freshly-captured root when empty.
        let prev = self
            .history
            .last()
            .cloned()
            .unwrap_or_else(|| self.root.as_ref().expect("root captured above").clone());

        let mover = self.side_to_move;
        let to = m.to_sq();

        // Whether the mover's opponent was already in check before the move.
        // Impossible in legal play, but contrived fixtures and movegen scratch
        // loops reach it; the debug oracles below weaken there, because the
        // pre-existing checker is one the differential `checkersBB` neither can
        // nor needs to reproduce.
        #[cfg(debug_assertions)]
        let enemy_already_in_check = post_move_gives_check(&self.board, mover.flip());

        let undo = if m.is_drop() {
            let kind = m.dropped_piece_kind();
            let piece = Piece::new(kind, mover);
            self.board.set(to, Some(piece));
            self.hands[mover.index()].decrement(kind);
            self.board_key ^= crate::key::psq(piece, to);
            self.hand_key = self
                .hand_key
                .wrapping_sub(crate::key::hand_step(mover, kind));
            // The copy that left the hand contributed to no partial key.
            self.xor_piece_partial(piece, to);
            Undo { captured: None }
        } else {
            let from = m.from_sq();
            let captured = self.board.get(to);
            let piece_after = m.moved_piece_after();
            let piece_before = piece_before_move(m, piece_after);
            self.board.set(to, Some(piece_after));
            self.board.set(from, None);
            if let Some(cap) = captured {
                self.hands[mover.index()].increment(cap.kind);
                self.board_key ^= crate::key::psq(cap, to);
                self.hand_key = self
                    .hand_key
                    .wrapping_add(crate::key::hand_step(mover, cap.kind));
                // The captured piece's move into the hand touches no partial key.
                self.xor_piece_partial(cap, to);
            }
            self.board_key ^= crate::key::psq(piece_before, from);
            self.board_key ^= crate::key::psq(piece_after, to);
            // A promoting pawn thereby leaves `pawn_key` for `non_pawn_key`.
            self.xor_piece_partial(piece_before, from);
            self.xor_piece_partial(piece_after, to);
            Undo { captured }
        };

        self.side_to_move = mover.flip();
        self.ply = self.ply.wrapping_add(1);
        self.board_key ^= crate::key::side();

        // Only the `gives_check ⟹ king-attacked` direction is asserted, as in
        // the reference's `ASSERT_LV3`. The converse would wrongly fire where
        // the side not to move was already in check by some other piece: the
        // predicate correctly reports no *new* check while the full probe still
        // sees the pre-existing checker.
        #[cfg(debug_assertions)]
        if gives_check {
            debug_assert!(
                post_move_gives_check(&self.board, self.side_to_move),
                "gives_check flag was set for move {m:?} but the post-move attack \
                 probe finds no checker on the now-side-to-move's king",
            );
        }

        let mut continuous_check = prev.continuous_check;
        continuous_check[mover.index()] = if gives_check {
            prev.continuous_check[mover.index()] + 2
        } else {
            0
        };

        // `self.check_info` is still the parent value here; the board and side
        // are already final.
        let child_checkers = if gives_check {
            self.differential_child_checkers(m, mover, &self.check_info)
        } else {
            Bitboard::EMPTY
        };

        #[cfg(debug_assertions)]
        if gives_check {
            let oks = try_find_king(&self.board, self.side_to_move)
                .expect("gives_check implies the now-side-to-move has a king");
            let probed = crate::movegen::attackers_bb(&self.board, oks, mover);
            debug_assert!(
                (child_checkers & !probed).is_empty(),
                "differential checkers {child_checkers:?} are not a subset of the \
                 probed attackers {probed:?} for move {m:?}",
            );
            if !enemy_already_in_check {
                debug_assert_eq!(
                    child_checkers, probed,
                    "differential checkers disagree with the full probe for move {m:?}",
                );
            }
        }

        self.check_info_stack.push(self.check_info);
        self.check_info =
            self.compute_check_info_with_in_check_and_checkers(gives_check, child_checkers);

        self.history.push(StateInfo {
            hands: self.hands,
            side_to_move: self.side_to_move,
            gives_check,
            board_key: self.board_key,
            plies_from_null: prev.plies_from_null + 1,
            continuous_check,
            // Filled in by `store_current_repetition` below.
            repetition: 0,
            repetition_times: 0,
            repetition_type: RepetitionState::None,
        });

        self.store_current_repetition();

        undo
    }

    /// The setup-root `StateInfo` — the `set` / `set_state` initial state
    /// (`position.cpp`).
    fn root_state(&self) -> StateInfo {
        StateInfo {
            hands: self.hands,
            side_to_move: self.side_to_move,
            gives_check: false,
            board_key: self.board_key,
            plies_from_null: 0,
            continuous_check: [0; Color::COUNT],
            repetition: 0,
            repetition_times: 0,
            repetition_type: RepetitionState::None,
        }
    }

    /// Compute the repetition info for the current state and store it into that
    /// `StateInfo`. Must run after the `do_move` push.
    fn store_current_repetition(&mut self) {
        let (repetition, repetition_times, repetition_type) = self.compute_repetition();
        if let Some(top) = self.history.last_mut() {
            top.repetition = repetition;
            top.repetition_times = repetition_times;
            top.repetition_type = repetition_type;
        }
    }

    pub fn undo_move(&mut self, m: Move, undo: Undo) {
        // Pop history first, so that `do_move` then `undo_move` leaves an
        // element-by-element identical `Position`.
        let _ = self.history.pop();
        // An empty stack — an `undo_move` with no matching `do_move` — falls
        // back to a fresh compute.
        self.check_info = match self.check_info_stack.pop() {
            Some(ci) => ci,
            None => self.compute_check_info(),
        };
        self.side_to_move = self.side_to_move.flip();
        self.ply = self.ply.wrapping_sub(1);
        self.board_key ^= crate::key::side();
        let mover = self.side_to_move;
        let to = m.to_sq();

        if m.is_drop() {
            let kind = m.dropped_piece_kind();
            let piece = Piece::new(kind, mover);
            self.board.set(to, None);
            self.hands[mover.index()].increment(kind);
            self.board_key ^= crate::key::psq(piece, to);
            self.hand_key = self
                .hand_key
                .wrapping_add(crate::key::hand_step(mover, kind));
            self.xor_piece_partial(piece, to);
            return;
        }

        let from = m.from_sq();
        let after = m.moved_piece_after();
        let piece_before = piece_before_move(m, after);
        self.board.set(from, Some(piece_before));
        self.board.set(to, undo.captured);
        self.board_key ^= crate::key::psq(after, to);
        self.board_key ^= crate::key::psq(piece_before, from);
        self.xor_piece_partial(after, to);
        self.xor_piece_partial(piece_before, from);
        if let Some(cap) = undo.captured {
            self.hands[mover.index()].decrement(cap.kind);
            self.board_key ^= crate::key::psq(cap, to);
            self.hand_key = self
                .hand_key
                .wrapping_sub(crate::key::hand_step(mover, cap.kind));
            self.xor_piece_partial(cap, to);
        }
    }

    /// Play a null move: pass the turn without touching the board or hands
    /// (`Position::do_null_move`, `position.cpp`). Undo with
    /// [`Self::undo_null_move`].
    ///
    /// `plies_from_null` resets to 0, so repetition detection never looks back
    /// across the null move and rebuilds its window one ply at a time.
    pub fn do_null_move(&mut self) {
        if self.history.is_empty() {
            self.root = Some(self.root_state());
        }

        let prev = self
            .history
            .last()
            .cloned()
            .unwrap_or_else(|| self.root.as_ref().expect("root captured above").clone());

        let null_mover = self.side_to_move;
        self.side_to_move = self.side_to_move.flip();
        self.ply = self.ply.wrapping_add(1);
        self.board_key ^= crate::key::side();

        // A null move is never a check, so it breaks the passing side's streak.
        // The new side to move's run is already 0: you cannot null out of check.
        let mut continuous_check = prev.continuous_check;
        continuous_check[null_mover.index()] = 0;

        // The board is unchanged, but passing the turn swaps which king is the
        // enemy, so the check info differs; save the parent value for O(1) undo
        // and compute the new side's info eagerly. A null move never gives check
        // and is only tried when the side to move is not already in check, so the
        // child's `in_check` is `false` (injected, not probed).
        self.check_info_stack.push(self.check_info);
        self.check_info = self.compute_check_info_with_in_check(false);

        self.history.push(StateInfo {
            hands: self.hands,
            side_to_move: self.side_to_move,
            gives_check: false,
            board_key: self.board_key,
            // `st->pliesFromNull = 0` (`position.cpp`).
            plies_from_null: 0,
            continuous_check,
            // `st->repetition = st->repetition_times = 0`; a state reached
            // directly by a null move is never itself a repetition.
            repetition: 0,
            repetition_times: 0,
            repetition_type: RepetitionState::None,
        });
    }

    /// Undo the [`Self::do_null_move`] that reached the current position.
    pub fn undo_null_move(&mut self) {
        let _ = self.history.pop();
        self.check_info = match self.check_info_stack.pop() {
            Some(ci) => ci,
            None => self.compute_check_info(),
        };
        self.side_to_move = self.side_to_move.flip();
        self.ply = self.ply.wrapping_sub(1);
        self.board_key ^= crate::key::side();
    }

    /// True iff `s`'s `(board_key, hands, side_to_move)` matches the current
    /// position's.
    fn matches_current_state(&self, s: &StateInfo) -> bool {
        s.board_key == self.board_key
            && s.hands == self.hands
            && s.side_to_move == self.side_to_move
    }

    /// Number of `history` entries matching the current position. The entry
    /// `do_move` just pushed counts, so a state reached once returns 1.
    pub fn position_occurrences(&self) -> usize {
        self.history
            .iter()
            .filter(|s| self.matches_current_state(s))
            .count()
    }

    /// Yield the history entries after the most-recent prior occurrence of the
    /// current position, oldest-to-newest, so consumers can pair entries with
    /// the moves that produced them by index. Empty if there is no prior
    /// occurrence.
    #[cfg(test)]
    pub(crate) fn history_since_last_distinct(&self) -> impl Iterator<Item = &StateInfo> + '_ {
        let len = self.history.len();
        let start = if len == 0 {
            len
        } else {
            match self.history[..len - 1]
                .iter()
                .rposition(|s| self.matches_current_state(s))
            {
                Some(p) => p + 1,
                None => len,
            }
        };
        self.history[start..len].iter()
    }

    /// The state exactly `dist` plies back from the current position, or `None`
    /// if that reaches past the recorded root. The timeline is
    /// `[root, history[0], …, history[len - 1] == current]`.
    fn state_back(&self, dist: usize) -> Option<&StateInfo> {
        let len = self.history.len();
        match len.cmp(&dist) {
            std::cmp::Ordering::Less => None,
            std::cmp::Ordering::Equal => self.root.as_ref(),
            std::cmp::Ordering::Greater => Some(&self.history[len - 1 - dist]),
        }
    }

    /// `plies_from_null` of the current state, `0` before any move.
    fn current_plies_from_null(&self) -> i32 {
        self.history.last().map_or(0, |s| s.plies_from_null)
    }

    /// `continuous_check[color]` of the current state, `0` before any move.
    fn current_continuous_check(&self, color: Color) -> i32 {
        self.history
            .last()
            .map_or(0, |s| s.continuous_check[color.index()] as i32)
    }

    /// Compute the repetition triple for the current position, walking the
    /// `StateInfo` chain back `min(16, plies_from_null)` plies in steps of two
    /// (`position.cpp`).
    ///
    /// It reads the *stored* `repetition_times` / `repetition_type` of the
    /// nearest prior occurrence, so the chain it builds on must already carry
    /// them — every state reached through `do_move` does.
    fn compute_repetition(&self) -> (i32, i32, RepetitionState) {
        let end = MAX_REPETITION_PLY.min(self.current_plies_from_null());
        // A repetition needs at least a four-ply cycle to return.
        if end < 4 {
            return (0, 0, RepetitionState::None);
        }

        let stm = self.side_to_move;
        let cc_stm = self.current_continuous_check(stm);
        let cc_opp = self.current_continuous_check(stm.flip());

        let mut i = 4i32;
        while i <= end {
            if let Some(prev) = self.state_back(i as usize) {
                // No board copy is stored, so a `board_key` hit is trusted
                // directly and the stored hands then decide draw vs
                // superior/inferior. This inherits the reference's accepted
                // hash-collision risk.
                if prev.board_key == self.board_key {
                    if prev.hands == self.hands {
                        let times = prev.repetition_times + 1;
                        // Negative marks the forced fourfold.
                        let repetition = if times >= 3 { -i } else { i };
                        let mut typ = if i <= cc_stm {
                            RepetitionState::Lose
                        } else if i <= cc_opp {
                            RepetitionState::Win
                        } else {
                            RepetitionState::Draw
                        };
                        // A cycle that was perpetual check only part of the way
                        // is an ordinary draw.
                        if prev.repetition_times != 0 && typ != prev.repetition_type {
                            typ = RepetitionState::Draw;
                        }
                        return (repetition, times, typ);
                    }
                    // Superior / inferior is decided on the side to move's hand
                    // alone: by piece conservation the opponent's hand moves
                    // oppositely.
                    let cur_hand = &self.hands[stm.index()];
                    let prev_hand = &prev.hands[stm.index()];
                    if hand_is_equal_or_superior(cur_hand, prev_hand) {
                        return (i, 0, RepetitionState::Superior);
                    }
                    if hand_is_equal_or_superior(prev_hand, cur_hand) {
                        return (i, 0, RepetitionState::Inferior);
                    }
                    // Neither dominates: a coincidental board match, keep going.
                }
            }
            i += 2;
        }

        (0, 0, RepetitionState::None)
    }

    /// Classify the current position as a repetition from the search's viewpoint
    /// at search distance `ply` from the root (`Position::is_repetition`,
    /// `position.cpp`).
    ///
    /// The `repetition < ply` gate reports a twofold or threefold only when the
    /// earlier occurrence lies after the search root, while a forced fourfold —
    /// carrying a negative `repetition` — always passes, so it is seen even when
    /// it sits entirely in the game history before the root.
    pub fn is_repetition(&self, ply: u16) -> RepetitionState {
        match self.history.last() {
            Some(s) if s.repetition != 0 && s.repetition < ply as i32 => s.repetition_type,
            _ => RepetitionState::None,
        }
    }
}

/// The mutable-board guard returned by [`Position::board_mut`].
pub struct BoardMut<'a> {
    pos: &'a mut Position,
}

impl std::ops::Deref for BoardMut<'_> {
    type Target = Board;

    fn deref(&self) -> &Board {
        &self.pos.board
    }
}

impl std::ops::DerefMut for BoardMut<'_> {
    fn deref_mut(&mut self) -> &mut Board {
        &mut self.pos.board
    }
}

impl Drop for BoardMut<'_> {
    fn drop(&mut self) {
        self.pos.check_info = self.pos.compute_check_info();
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Undo {
    pub(crate) captured: Option<Piece>,
}

impl Undo {
    /// The piece captured by the move this `Undo` reverses, or `None` for a
    /// non-capturing move (`pos.captured_piece()`).
    pub fn captured(self) -> Option<Piece> {
        self.captured
    }
}

impl Default for Position {
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::piece::{Piece, PieceKind};
    use crate::square::Square;

    #[test]
    fn empty_has_black_to_move_and_ply_one() {
        let p = Position::empty();
        assert_eq!(p.side_to_move(), Color::Black);
        assert_eq!(p.ply(), 1);
    }

    #[test]
    fn empty_has_no_pieces_or_hands() {
        let p = Position::empty();
        for index in 0..Square::COUNT as u8 {
            let sq = Square::from_index(index).unwrap();
            assert!(p.board().get(sq).is_none());
        }
        for color in [Color::Black, Color::White] {
            for kind in [
                PieceKind::Pawn,
                PieceKind::Lance,
                PieceKind::Knight,
                PieceKind::Silver,
                PieceKind::Gold,
                PieceKind::Bishop,
                PieceKind::Rook,
            ] {
                assert_eq!(p.hand(color).count(kind), 0);
            }
        }
    }

    #[test]
    fn mutating_one_hand_does_not_disturb_the_other() {
        let mut p = Position::empty();
        p.hand_mut(Color::Black).increment(PieceKind::Pawn);
        assert_eq!(p.hand(Color::Black).count(PieceKind::Pawn), 1);
        assert_eq!(p.hand(Color::White).count(PieceKind::Pawn), 0);
    }

    #[test]
    fn board_mut_writes_through_to_board() {
        let mut p = Position::empty();
        let sq = Square::new(0, 0).unwrap();
        let piece = Piece::new(PieceKind::King, Color::Black);
        p.board_mut().set(sq, Some(piece));
        assert_eq!(p.board().get(sq), Some(piece));
    }

    #[test]
    fn do_undo_round_trips_pawn_push_at_startpos() {
        let mut p = Position::startpos();
        let snapshot = p.clone();
        let from = Square::new(6, 6).unwrap(); // 7g
        let to = Square::new(6, 5).unwrap(); // 7f
        let pawn = Piece::new(PieceKind::Pawn, Color::Black);
        let m = Move::make(from, to, pawn);
        let undo = p.do_move(m);
        assert_eq!(p.side_to_move(), Color::White);
        assert_eq!(p.ply(), 2);
        assert_eq!(p.board().get(to), Some(pawn));
        assert!(p.board().get(from).is_none());
        p.undo_move(m, undo);
        assert_eq!(p, snapshot);
    }

    #[test]
    fn do_undo_round_trips_capture() {
        let mut p = Position::empty();
        let from = Square::new(4, 4).unwrap();
        let to = Square::new(4, 3).unwrap();
        let attacker = Piece::new(PieceKind::Rook, Color::Black);
        let victim = Piece::new(PieceKind::Pawn, Color::White);
        let bk = Piece::new(PieceKind::King, Color::Black);
        let wk = Piece::new(PieceKind::King, Color::White);
        p.board_mut().set(from, Some(attacker));
        p.board_mut().set(to, Some(victim));
        p.board_mut().set(Square::new(0, 8).unwrap(), Some(bk));
        p.board_mut().set(Square::new(0, 0).unwrap(), Some(wk));
        let snapshot = p.clone();
        let m = Move::make(from, to, attacker);
        let undo = p.do_move(m);
        assert_eq!(p.board().get(to), Some(attacker));
        assert!(p.board().get(from).is_none());
        assert_eq!(p.hand(Color::Black).count(PieceKind::Pawn), 1);
        p.undo_move(m, undo);
        assert_eq!(p, snapshot);
    }

    #[test]
    fn do_undo_round_trips_promotion() {
        let mut p = Position::empty();
        let from = Square::new(4, 3).unwrap(); // rank 3, just outside promotion zone
        let to = Square::new(4, 2).unwrap(); // rank 2, in promotion zone
        let pawn = Piece::new(PieceKind::Pawn, Color::Black);
        p.board_mut().set(from, Some(pawn));
        p.board_mut().set(
            Square::new(0, 8).unwrap(),
            Some(Piece::new(PieceKind::King, Color::Black)),
        );
        p.board_mut().set(
            Square::new(0, 0).unwrap(),
            Some(Piece::new(PieceKind::King, Color::White)),
        );
        let snapshot = p.clone();
        let m = Move::make_promote(from, to, pawn);
        let undo = p.do_move(m);
        let placed = p.board().get(to).unwrap();
        assert_eq!(placed.kind, PieceKind::Pawn);
        assert!(placed.promoted);
        p.undo_move(m, undo);
        assert_eq!(p, snapshot);
    }

    #[test]
    fn do_undo_round_trips_drop() {
        let mut p = Position::empty();
        p.hand_mut(Color::Black).increment(PieceKind::Pawn);
        p.board_mut().set(
            Square::new(0, 8).unwrap(),
            Some(Piece::new(PieceKind::King, Color::Black)),
        );
        p.board_mut().set(
            Square::new(0, 0).unwrap(),
            Some(Piece::new(PieceKind::King, Color::White)),
        );
        let snapshot = p.clone();
        let to = Square::new(4, 4).unwrap();
        let m = Move::make_drop(PieceKind::Pawn, Color::Black, to);
        let undo = p.do_move(m);
        assert_eq!(
            p.board().get(to),
            Some(Piece::new(PieceKind::Pawn, Color::Black))
        );
        assert_eq!(p.hand(Color::Black).count(PieceKind::Pawn), 0);
        p.undo_move(m, undo);
        assert_eq!(p, snapshot);
    }

    /// Two-king board with no other pieces, so both kings can step sideways
    /// without check geometry getting in the way.
    fn setup_king_shuffle_pos() -> Position {
        let mut p = Position::empty();
        p.board_mut().set(
            Square::new(0, 8).unwrap(),
            Some(Piece::new(PieceKind::King, Color::Black)),
        );
        p.board_mut().set(
            Square::new(0, 0).unwrap(),
            Some(Piece::new(PieceKind::King, Color::White)),
        );
        p
    }

    /// Four moves that, applied in order from `setup_king_shuffle_pos`, return
    /// the position to its starting state.
    fn shuffle_cycle() -> [Move; 4] {
        let bk = Piece::new(PieceKind::King, Color::Black);
        let wk = Piece::new(PieceKind::King, Color::White);
        [
            Move::make(Square::new(0, 8).unwrap(), Square::new(1, 8).unwrap(), bk),
            Move::make(Square::new(0, 0).unwrap(), Square::new(1, 0).unwrap(), wk),
            Move::make(Square::new(1, 8).unwrap(), Square::new(0, 8).unwrap(), bk),
            Move::make(Square::new(1, 0).unwrap(), Square::new(0, 0).unwrap(), wk),
        ]
    }

    #[test]
    fn do_undo_round_trips_capture_promotion() {
        // A Black pawn captures a White knight on the last rank, where the
        // promotion is forced.
        let mut p = Position::empty();
        let bk = Piece::new(PieceKind::King, Color::Black);
        let wk = Piece::new(PieceKind::King, Color::White);
        let pawn = Piece::new(PieceKind::Pawn, Color::Black);
        let target = Piece::new(PieceKind::Knight, Color::White);
        p.board_mut().set(Square::new(8, 8).unwrap(), Some(bk));
        p.board_mut().set(Square::new(0, 0).unwrap(), Some(wk));
        p.board_mut().set(Square::new(4, 1).unwrap(), Some(pawn));
        p.board_mut().set(Square::new(4, 0).unwrap(), Some(target));
        let snapshot = p.clone();
        let from = Square::new(4, 1).unwrap();
        let to = Square::new(4, 0).unwrap();
        let m = Move::make_promote(from, to, pawn);
        let undo = p.do_move(m);
        let placed = p.board().get(to).unwrap();
        assert_eq!(placed.kind, PieceKind::Pawn);
        assert!(placed.promoted);
        assert_eq!(p.hand(Color::Black).count(PieceKind::Knight), 1);
        p.undo_move(m, undo);
        assert_eq!(p, snapshot);
    }

    #[test]
    fn do_move_grows_history_by_one() {
        let mut p = Position::startpos();
        let prior_len = p.history.len();
        let pawn = Piece::new(PieceKind::Pawn, Color::Black);
        let m = Move::make(Square::new(6, 6).unwrap(), Square::new(6, 5).unwrap(), pawn);
        p.do_move(m);
        assert_eq!(p.history.len(), prior_len + 1);
    }

    #[test]
    fn undo_move_shrinks_history_by_one() {
        let mut p = Position::startpos();
        let pawn = Piece::new(PieceKind::Pawn, Color::Black);
        let m = Move::make(Square::new(6, 6).unwrap(), Square::new(6, 5).unwrap(), pawn);
        let undo = p.do_move(m);
        let post_do_len = p.history.len();
        p.undo_move(m, undo);
        assert_eq!(p.history.len(), post_do_len - 1);
    }

    #[test]
    fn startpos_history_is_empty() {
        assert_eq!(Position::startpos().history.len(), 0);
        assert_eq!(Position::empty().history.len(), 0);
        let parsed = crate::sfen::parse_sfen(crate::sfen::STARTPOS_SFEN).unwrap();
        assert_eq!(parsed.history.len(), 0);
    }

    #[test]
    fn position_occurrences_zero_in_fresh_state() {
        assert_eq!(Position::startpos().position_occurrences(), 0);
        assert_eq!(Position::empty().position_occurrences(), 0);
    }

    #[test]
    fn position_occurrences_after_round_trip_cycle() {
        let mut p = setup_king_shuffle_pos();
        for m in shuffle_cycle() {
            p.do_move(m);
        }
        assert_eq!(p.position_occurrences(), 1);
    }

    #[test]
    fn gives_check_true_for_check_move() {
        // A Black rook slides onto the file of the white king at (4,0).
        let mut p = Position::empty();
        p.board_mut().set(
            Square::new(8, 8).unwrap(),
            Some(Piece::new(PieceKind::King, Color::Black)),
        );
        p.board_mut().set(
            Square::new(4, 0).unwrap(),
            Some(Piece::new(PieceKind::King, Color::White)),
        );
        p.board_mut().set(
            Square::new(0, 8).unwrap(),
            Some(Piece::new(PieceKind::Rook, Color::Black)),
        );
        let rook = Piece::new(PieceKind::Rook, Color::Black);
        let m = Move::make(Square::new(0, 8).unwrap(), Square::new(4, 8).unwrap(), rook);
        p.do_move(m);
        assert!(
            p.history.last().unwrap().gives_check,
            "rook to file 4 should put white king in check",
        );
    }

    #[test]
    fn gives_check_false_for_quiet_move() {
        let mut p = Position::startpos();
        let pawn = Piece::new(PieceKind::Pawn, Color::Black);
        let m = Move::make(Square::new(6, 6).unwrap(), Square::new(6, 5).unwrap(), pawn);
        p.do_move(m);
        assert!(
            !p.history.last().unwrap().gives_check,
            "7g7f is a quiet move",
        );
    }

    #[test]
    fn gives_check_true_for_discovered_check() {
        // A Black knight blocks a Black rook from the white king on file 4;
        // stepping the knight off-file uncovers the rook without the knight
        // itself attacking the king.
        let mut p = Position::empty();
        p.board_mut().set(
            Square::new(0, 8).unwrap(),
            Some(Piece::new(PieceKind::King, Color::Black)),
        );
        p.board_mut().set(
            Square::new(4, 0).unwrap(),
            Some(Piece::new(PieceKind::King, Color::White)),
        );
        p.board_mut().set(
            Square::new(4, 8).unwrap(),
            Some(Piece::new(PieceKind::Rook, Color::Black)),
        );
        p.board_mut().set(
            Square::new(4, 5).unwrap(),
            Some(Piece::new(PieceKind::Knight, Color::Black)),
        );
        let knight = Piece::new(PieceKind::Knight, Color::Black);
        let m = Move::make(
            Square::new(4, 5).unwrap(),
            Square::new(3, 3).unwrap(),
            knight,
        );
        p.do_move(m);
        assert!(
            p.history.last().unwrap().gives_check,
            "knight move should uncover rook discovered-check",
        );
    }

    #[test]
    fn history_since_last_distinct_walks_short_cycle() {
        // Two king-shuffle cycles, so the start state appears twice in history
        // and the iterator should yield only the most-recent cycle.
        let mut p = setup_king_shuffle_pos();
        let moves = shuffle_cycle();
        for _ in 0..2 {
            for m in moves {
                p.do_move(m);
            }
        }
        assert_eq!(p.position_occurrences(), 2);

        // Replay one cycle from a fresh position for the expected sequence.
        let mut expected = setup_king_shuffle_pos();
        let mut expected_states: Vec<(u64, Color)> = Vec::new();
        for m in moves {
            expected.do_move(m);
            expected_states.push((expected.board_key(), expected.side_to_move()));
        }

        let cycle: Vec<&StateInfo> = p.history_since_last_distinct().collect();
        assert_eq!(cycle.len(), 4, "cycle: {cycle:?}");
        for (i, s) in cycle.iter().enumerate() {
            assert_eq!(
                (s.board_key, s.side_to_move),
                expected_states[i],
                "cycle[{i}] does not match expected post-move state",
            );
        }
        assert_eq!(cycle[3].board_key, p.board_key());
        assert_eq!(cycle[3].side_to_move, p.side_to_move());
    }

    #[test]
    fn history_since_last_distinct_is_empty_when_no_prior_occurrence() {
        let mut p = Position::startpos();
        let pawn = Piece::new(PieceKind::Pawn, Color::Black);
        let m = Move::make(Square::new(6, 6).unwrap(), Square::new(6, 5).unwrap(), pawn);
        p.do_move(m);
        let cycle: Vec<&StateInfo> = p.history_since_last_distinct().collect();
        assert!(cycle.is_empty(), "expected empty cycle, got {cycle:?}");
    }

    use crate::sfen::parse_sfen;

    const FIXTURE_SFENS: &[&str] = &[
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1", // startpos
        "4k4/9/4r4/9/9/9/4K3B/9/9 b RG2gs2n3p 1",                          // check-evasion
        "k8/1P7/G8/1N2P4/9/9/9/9/8K b 2PG2pg 1",                           // drop-heavy
        "l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1", // mid-game-tactical
        "4k4/3P3+PL/2N2PR2/1L2BNS2/4N4/9/9/9/4K4 b - 1", // promotion-zone-edges
        "9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1",                // sennichite
    ];

    /// Deterministic xorshift64*, so a failing playout replays.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }

        fn pick(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    /// Assert the incrementally-maintained keys equal a from-scratch
    /// recomputation, and that `key() == board_key ^ hand_key`.
    fn assert_key_consistent(p: &Position, ctx: &str) {
        let (board_key, hand_key) = p.recomputed_keys();
        assert_eq!(p.board_key(), board_key, "{ctx}: board_key diverged");
        assert_eq!(p.hand_key(), hand_key, "{ctx}: hand_key diverged");
        assert_eq!(
            p.key(),
            board_key ^ hand_key,
            "{ctx}: key() != board_key ^ hand_key",
        );

        let (pawn_key, minor_piece_key, non_pawn_key) = p.recomputed_partial_keys();
        assert_eq!(p.pawn_key(), pawn_key, "{ctx}: pawn_key diverged");
        assert_eq!(
            p.minor_piece_key(),
            minor_piece_key,
            "{ctx}: minor_piece_key diverged",
        );
        assert_eq!(
            p.non_pawn_key(Color::Black),
            non_pawn_key[Color::Black.index()],
            "{ctx}: non_pawn_key(Black) diverged",
        );
        assert_eq!(
            p.non_pawn_key(Color::White),
            non_pawn_key[Color::White.index()],
            "{ctx}: non_pawn_key(White) diverged",
        );
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn incremental_key_matches_from_scratch_on_random_playouts() {
        const MIN_PLIES: usize = 30;

        for (fi, sfen) in FIXTURE_SFENS.iter().enumerate() {
            let mut pos = parse_sfen(sfen).expect("fixture sfen parses");
            assert_key_consistent(&pos, &format!("fixture {fi} root"));

            let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ (fi as u64).wrapping_add(1));
            let mut stack: Vec<(Move, Undo)> = Vec::new();
            let mut legal: Vec<Move> = Vec::new();

            let mut plies = 0usize;
            while plies < MIN_PLIES {
                legal.clear();
                pos.generate_legal_all(&mut legal);
                if legal.is_empty() {
                    // Terminal: unwind fully and restart from the root.
                    while let Some((m, u)) = stack.pop() {
                        pos.undo_move(m, u);
                        assert_key_consistent(&pos, &format!("fixture {fi} restart undo"));
                    }
                    continue;
                }
                let m = legal[rng.pick(legal.len())];
                let u = pos.do_move(m);
                assert_key_consistent(&pos, &format!("fixture {fi} ply {plies} after do"));
                stack.push((m, u));
                plies += 1;
            }

            while let Some((m, u)) = stack.pop() {
                pos.undo_move(m, u);
                assert_key_consistent(&pos, &format!("fixture {fi} after undo"));
            }
        }
    }

    #[test]
    fn transposition_reaches_identical_key() {
        // Two pawn pushes a side, interleaved in two turn-respecting orders.
        let line_a = ["2g2f", "8c8d", "7g7f", "3c3d"];
        let line_b = ["7g7f", "3c3d", "2g2f", "8c8d"];

        let play = |line: &[&str]| {
            let mut p = Position::startpos();
            for usi in line {
                let m = crate::move_::parse_usi_move(usi, &p).expect("legal move");
                p.do_move(m);
            }
            p
        };

        let a = play(&line_a);
        let b = play(&line_b);

        assert_eq!(a.board(), b.board());
        assert_eq!(a.side_to_move(), b.side_to_move());
        for color in [Color::Black, Color::White] {
            assert_eq!(a.hand(color), b.hand(color));
        }

        assert_eq!(a.key(), b.key(), "transposed positions must share a key");
        assert_key_consistent(&a, "transposition line a");
        assert_key_consistent(&b, "transposition line b");
    }

    #[test]
    fn hand_key_capture_then_drop_cancels() {
        // A black rook captures a white pawn into hand, White waits, then Black
        // drops the pawn back out.
        let mut p = parse_sfen("k8/9/9/4p4/4R4/9/9/9/8K b - 1").unwrap();
        let h0 = p.hand_key();
        assert_key_consistent(&p, "capture/drop root");

        let rook = Piece::new(PieceKind::Rook, Color::Black);
        let capture = Move::make(Square::new(4, 4).unwrap(), Square::new(4, 3).unwrap(), rook);
        p.do_move(capture);
        assert_ne!(p.hand_key(), h0, "capture should push a pawn into hand");
        assert_key_consistent(&p, "after capture");

        let wk = Piece::new(PieceKind::King, Color::White);
        let wait = Move::make(Square::new(8, 0).unwrap(), Square::new(7, 0).unwrap(), wk);
        p.do_move(wait);
        assert_key_consistent(&p, "after white wait");

        let drop = Move::make_drop(PieceKind::Pawn, Color::Black, Square::new(4, 4).unwrap());
        p.do_move(drop);
        assert_key_consistent(&p, "after drop");
        assert_eq!(
            p.hand_key(),
            h0,
            "capture then drop of the same kind must restore hand_key",
        );
    }

    #[test]
    fn hand_count_changes_the_key() {
        let one = parse_sfen("9/9/9/9/9/9/9/9/9 b P 1").unwrap();
        let two = parse_sfen("9/9/9/9/9/9/9/9/9 b 2P 1").unwrap();
        assert_ne!(one.key(), two.key(), "differing hand counts must differ");
        assert_ne!(one.hand_key(), two.hand_key());
        assert_eq!(one.board_key(), two.board_key());
    }

    #[test]
    fn side_to_move_changes_the_key() {
        let black = parse_sfen("9/9/9/9/4P4/9/9/9/9 b - 1").unwrap();
        let white = parse_sfen("9/9/9/9/4P4/9/9/9/9 w - 1").unwrap();
        assert_ne!(black.key(), white.key(), "side-to-move must change the key");
        assert_ne!(black.board_key(), white.board_key());
        // The side term lives in `board_key`.
        assert_eq!(black.hand_key(), white.hand_key());
    }

    #[test]
    fn partial_keys_ignore_hand_contents() {
        let a = parse_sfen("9/9/9/9/4P4/9/9/9/9 b - 1").unwrap();
        let b = parse_sfen("9/9/9/9/4P4/9/9/9/9 b 2Rb 1").unwrap();
        assert_eq!(a.pawn_key(), b.pawn_key(), "pawn_key must ignore hands");
        assert_eq!(
            a.minor_piece_key(),
            b.minor_piece_key(),
            "minor_piece_key must ignore hands",
        );
        assert_eq!(a.non_pawn_key(Color::Black), b.non_pawn_key(Color::Black));
        assert_eq!(a.non_pawn_key(Color::White), b.non_pawn_key(Color::White));
        assert_ne!(a.key(), b.key());
    }

    #[test]
    fn pawn_key_empty_when_no_board_pawns() {
        let p = parse_sfen("4k4/9/9/9/9/9/9/9/4K4 b - 1").unwrap();
        assert_eq!(p.pawn_key(), crate::key::NO_PAWNS_SEED);
        let q = parse_sfen("4k4/9/9/9/4P4/9/9/9/4K4 b - 1").unwrap();
        assert_ne!(q.pawn_key(), crate::key::NO_PAWNS_SEED);
    }

    #[test]
    fn promoting_pawn_leaves_pawn_key_and_enters_minor_and_nonpawn() {
        // The board's only pawn, on 5c, promotes forward to 5b.
        let mut p = parse_sfen("4k4/9/4P4/9/9/9/9/9/4K4 b - 1").unwrap();
        let pawn = Piece::new(PieceKind::Pawn, Color::Black);
        let from = Square::new(4, 2).unwrap();
        let to = Square::new(4, 1).unwrap();
        let minor_before = p.minor_piece_key();
        let nonpawn_before = p.non_pawn_key(Color::Black);
        assert_ne!(p.pawn_key(), crate::key::NO_PAWNS_SEED, "pawn present");

        let m = Move::make_promote(from, to, pawn);
        let undo = p.do_move(m);
        assert_key_consistent(&p, "after promoting pawn push");
        assert_eq!(
            p.pawn_key(),
            crate::key::NO_PAWNS_SEED,
            "promoted pawn must leave pawn_key",
        );
        assert_ne!(
            p.minor_piece_key(),
            minor_before,
            "promoted pawn (a minor piece) must enter minor_piece_key",
        );
        assert_ne!(
            p.non_pawn_key(Color::Black),
            nonpawn_before,
            "promoted pawn must enter Black's non_pawn_key",
        );

        p.undo_move(m, undo);
        assert_key_consistent(&p, "after undo of promoting pawn push");
        assert_ne!(p.pawn_key(), crate::key::NO_PAWNS_SEED, "pawn restored");
        assert_eq!(p.minor_piece_key(), minor_before);
        assert_eq!(p.non_pawn_key(Color::Black), nonpawn_before);
    }

    #[test]
    fn pawn_drop_enters_pawn_key() {
        // A pawn drop places a board pawn, so `pawn_key` must change (the copy
        // that left the hand contributed to no partial key). Round-trips exact.
        let mut p = parse_sfen("4k4/9/9/9/9/9/9/9/4K4 b P 1").unwrap();
        assert_eq!(p.pawn_key(), crate::key::NO_PAWNS_SEED, "no board pawn yet");
        let to = Square::new(4, 4).unwrap();
        let m = Move::make_drop(PieceKind::Pawn, Color::Black, to);
        let undo = p.do_move(m);
        assert_key_consistent(&p, "after pawn drop");
        assert_ne!(
            p.pawn_key(),
            crate::key::NO_PAWNS_SEED,
            "dropped pawn must enter pawn_key",
        );
        p.undo_move(m, undo);
        assert_key_consistent(&p, "after undo pawn drop");
        assert_eq!(p.pawn_key(), crate::key::NO_PAWNS_SEED);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn key_agrees_with_repetition_equality_on_sennichite() {
        let text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/perft/sennichite.json",
        ))
        .expect("read sennichite fixture");
        let json: serde_json::Value = serde_json::from_str(&text).expect("parse fixture");
        let sfen = json["sfen"].as_str().expect("fixture sfen");
        let moves: Vec<String> = json["moves"]
            .as_array()
            .expect("fixture moves")
            .iter()
            .map(|m| m.as_str().expect("move string").to_string())
            .collect();

        let mut pos = parse_sfen(sfen).expect("fixture sfen parses");
        type Signature = (Board, [Hand; Color::COUNT], Color);
        let mut seen: Vec<(Signature, u64)> = Vec::new();
        let mut any_recurrence = false;

        for usi in &moves {
            let m = crate::move_::parse_usi_move(usi, &pos).expect("fixture move parses");
            pos.do_move(m);
            assert_key_consistent(&pos, &format!("sennichite `{usi}`"));

            let sig: Signature = (
                *pos.board(),
                [*pos.hand(Color::Black), *pos.hand(Color::White)],
                pos.side_to_move(),
            );
            let key = pos.key();

            let priors: Vec<u64> = seen
                .iter()
                .filter(|(s, _)| *s == sig)
                .map(|(_, k)| *k)
                .collect();
            assert_eq!(
                pos.position_occurrences(),
                priors.len() + 1,
                "occurrence count vs structural signature disagree at `{usi}`",
            );
            for prior_key in &priors {
                any_recurrence = true;
                assert_eq!(
                    *prior_key, key,
                    "recurrence at `{usi}` has a different key than its earlier occurrence",
                );
            }
            seen.push((sig, key));
        }

        assert!(
            any_recurrence,
            "sennichite fixture should exercise at least one recurrence",
        );
    }

    /// Load the sennichite perft fixture's `(sfen, moves)`.
    fn load_sennichite() -> (String, Vec<String>) {
        let text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/perft/sennichite.json",
        ))
        .expect("read sennichite fixture");
        let json: serde_json::Value = serde_json::from_str(&text).expect("parse fixture");
        let sfen = json["sfen"].as_str().expect("fixture sfen").to_string();
        let moves = json["moves"]
            .as_array()
            .expect("fixture moves")
            .iter()
            .map(|m| m.as_str().expect("move string").to_string())
            .collect();
        (sfen, moves)
    }

    /// Build a `StateInfo` whose `board_key` is computed the way the
    /// incremental path would, so hand-assembled histories agree with
    /// `Position::board_key`. `assemble` fills in the repetition fields.
    fn make_state(
        board: Board,
        hands: [Hand; Color::COUNT],
        side: Color,
        gives_check: bool,
    ) -> StateInfo {
        let mut scratch = Position::empty();
        *scratch.board_mut() = board;
        scratch.hands = hands;
        scratch.set_side_to_move(side);
        scratch.refresh_keys();
        StateInfo {
            hands,
            side_to_move: side,
            gives_check,
            board_key: scratch.board_key(),
            plies_from_null: 0,
            continuous_check: [0; Color::COUNT],
            repetition: 0,
            repetition_times: 0,
            repetition_type: RepetitionState::None,
        }
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn is_repetition_none_before_and_draw_at_sennichite_recurrence() {
        // The fixture is a four-ply king shuffle, so the root recurs every four
        // plies — the first recurrence landing straight on it, which is
        // findable only through the root snapshot.
        let (sfen, moves) = load_sennichite();
        let mut pos = parse_sfen(&sfen).expect("fixture sfen parses");
        for (idx, usi) in moves.iter().enumerate() {
            let m = crate::move_::parse_usi_move(usi, &pos).expect("fixture move parses");
            pos.do_move(m);
            let depth = idx + 1;
            let rep = pos.is_repetition(16);
            if depth < 4 {
                assert_eq!(
                    rep,
                    RepetitionState::None,
                    "no repetition should be reported at depth {depth}",
                );
            } else {
                assert_eq!(
                    rep,
                    RepetitionState::Draw,
                    "plain sennichite must classify as DRAW at depth {depth}",
                );
            }
        }
    }

    #[test]
    fn is_repetition_lose_when_side_to_move_is_perpetual_checker() {
        // A Black rook oscillates giving check to a shuffling White king, and
        // the position returns to the root after four plies with Black to move.
        let mut p = Position::empty();
        let bk = Piece::new(PieceKind::King, Color::Black);
        let wk = Piece::new(PieceKind::King, Color::White);
        let br = Piece::new(PieceKind::Rook, Color::Black);
        p.board_mut().set(Square::new(0, 8).unwrap(), Some(bk)); // Black king, far corner
        p.board_mut().set(Square::new(4, 0).unwrap(), Some(wk)); // White king K0
        p.board_mut().set(Square::new(8, 1).unwrap(), Some(br)); // Black rook A
        p.refresh_keys();

        // A checks a king on rank 1, B one on rank 0.
        let a = Square::new(8, 1).unwrap();
        let b = Square::new(8, 0).unwrap();
        let k0 = Square::new(4, 0).unwrap();
        let k1 = Square::new(4, 1).unwrap();
        p.do_move(Move::make(a, b, br)); // Black: rook A→B, checks WK@K0
        p.do_move(Move::make(k0, k1, wk)); // White: king escapes to K1
        p.do_move(Move::make(b, a, br)); // Black: rook B→A, checks WK@K1
        p.do_move(Move::make(k1, k0, wk)); // White: king back to K0 (== root)

        assert_eq!(
            p.is_repetition(16),
            RepetitionState::Lose,
            "the perpetually-checking side to move must be classified LOSE",
        );
    }

    #[test]
    fn is_repetition_win_when_opponent_is_perpetual_checker() {
        // The mirror of the LOSE case: a White rook perpetually checks a
        // shuffling Black king, and the position recurs with Black to move.
        let mut p = Position::empty();
        let bk = Piece::new(PieceKind::King, Color::Black);
        let wk = Piece::new(PieceKind::King, Color::White);
        let wr = Piece::new(PieceKind::Rook, Color::White);
        p.board_mut().set(Square::new(0, 8).unwrap(), Some(wk)); // White king, far corner
        p.board_mut().set(Square::new(4, 0).unwrap(), Some(bk)); // Black king K0 (in check)
        p.board_mut().set(Square::new(8, 0).unwrap(), Some(wr)); // White rook B, checks K0
        p.set_side_to_move(Color::Black);
        p.refresh_keys();

        let a = Square::new(8, 1).unwrap();
        let b = Square::new(8, 0).unwrap();
        let k0 = Square::new(4, 0).unwrap();
        let k1 = Square::new(4, 1).unwrap();
        p.do_move(Move::make(k0, k1, bk)); // Black: king escapes to K1
        p.do_move(Move::make(b, a, wr)); // White: rook B→A, checks BK@K1
        p.do_move(Move::make(k1, k0, bk)); // Black: king back to K0
        p.do_move(Move::make(a, b, wr)); // White: rook A→B, checks BK@K0 (== root)

        assert_eq!(
            p.is_repetition(16),
            RepetitionState::Win,
            "the checked side to move must be classified WIN",
        );
    }

    /// The two-king board shared by every superior/inferior fixture state, so
    /// they differ only in their hands.
    fn superiority_board() -> Board {
        let mut board = Board::empty();
        board.set(
            Square::new(0, 8).unwrap(),
            Some(Piece::new(PieceKind::King, Color::Black)),
        );
        board.set(
            Square::new(0, 0).unwrap(),
            Some(Piece::new(PieceKind::King, Color::White)),
        );
        board
    }

    /// A [`superiority_board`] state plus the given per-color hand-pawn counts,
    /// Black to move.
    fn superiority_state(black_pawns: u8, white_pawns: u8) -> StateInfo {
        let board = superiority_board();
        let mut hands = [Hand::empty(), Hand::empty()];
        for _ in 0..black_pawns {
            hands[Color::Black.index()].increment(PieceKind::Pawn);
        }
        for _ in 0..white_pawns {
            hands[Color::White.index()].increment(PieceKind::Pawn);
        }
        make_state(board, hands, Color::Black, false)
    }

    /// Assemble a `Position` from a hand-built history, bypassing `do_move` so
    /// a controlled superior/inferior chain can be fed to `is_repetition`.
    /// `board` is separate because the per-state records carry no board copy.
    ///
    /// Each state's `plies_from_null` is set to its distance from the root, so
    /// the top state's look-back reaches the whole assembled chain.
    fn assemble(board: Board, states: &[StateInfo], root: StateInfo) -> Position {
        let last = states.last().expect("non-empty history").clone();
        let mut p = Position::empty();
        *p.board_mut() = board;
        p.hands = last.hands;
        p.set_side_to_move(last.side_to_move);
        p.refresh_keys();
        p.history = states.to_vec();
        for (idx, s) in p.history.iter_mut().enumerate() {
            s.plies_from_null = idx as i32 + 1;
        }
        p.root = Some(root);
        p.store_current_repetition();
        p
    }

    #[test]
    fn is_repetition_superior_when_side_to_move_gained_a_hand_pawn() {
        // Same board as the root four plies back, but Black now holds a hand
        // pawn it did not hold then.
        let root = superiority_state(0, 1);
        let filler = superiority_state(9, 9); // never inspected; only dist 4 is
        let current = superiority_state(1, 0);
        let p = assemble(
            superiority_board(),
            &[filler.clone(), filler.clone(), filler, current],
            root,
        );
        assert_eq!(p.is_repetition(16), RepetitionState::Superior);
    }

    #[test]
    fn is_repetition_inferior_when_side_to_move_lost_a_hand_pawn() {
        // The mirror of the superior case.
        let root = superiority_state(1, 0);
        let filler = superiority_state(9, 9);
        let current = superiority_state(0, 1);
        let p = assemble(
            superiority_board(),
            &[filler.clone(), filler.clone(), filler, current],
            root,
        );
        assert_eq!(p.is_repetition(16), RepetitionState::Inferior);
    }

    #[test]
    fn is_repetition_gated_by_search_ply() {
        // Both kings walk a triangle and return to the root only after six
        // plies, so `repetition == 6` and the gate turns on `ply > 6`.
        let mut p = Position::empty();
        let bk = Piece::new(PieceKind::King, Color::Black);
        let wk = Piece::new(PieceKind::King, Color::White);
        p.board_mut().set(Square::new(0, 8).unwrap(), Some(bk));
        p.board_mut().set(Square::new(0, 0).unwrap(), Some(wk));
        p.refresh_keys();

        let bmoves = [
            (Square::new(0, 8).unwrap(), Square::new(1, 8).unwrap()),
            (Square::new(1, 8).unwrap(), Square::new(0, 7).unwrap()),
            (Square::new(0, 7).unwrap(), Square::new(0, 8).unwrap()),
        ];
        let wmoves = [
            (Square::new(0, 0).unwrap(), Square::new(1, 0).unwrap()),
            (Square::new(1, 0).unwrap(), Square::new(0, 1).unwrap()),
            (Square::new(0, 1).unwrap(), Square::new(0, 0).unwrap()),
        ];
        for i in 0..3 {
            p.do_move(Move::make(bmoves[i].0, bmoves[i].1, bk));
            p.do_move(Move::make(wmoves[i].0, wmoves[i].1, wk));
        }

        assert_eq!(
            p.is_repetition(6),
            RepetitionState::None,
            "a recurrence reaching the root (repetition == ply) is not an immediate draw",
        );
        assert_eq!(
            p.is_repetition(7),
            RepetitionState::Draw,
            "with the root strictly inside the search window the recurrence is a draw",
        );
        assert_eq!(p.is_repetition(16), RepetitionState::Draw);
    }

    #[test]
    fn is_repetition_forced_fourfold_is_seen_across_the_search_root() {
        // Three four-ply king shuffles, so the root occurs a fourth time.
        let mut p = setup_king_shuffle_pos();
        let cycle = shuffle_cycle();
        for _ in 0..3 {
            for m in cycle {
                p.do_move(m);
            }
        }

        let top = p.history.last().expect("history is non-empty");
        assert_eq!(
            top.repetition_times, 3,
            "the fourth occurrence chains repetition_times to 3",
        );
        assert!(
            top.repetition < 0,
            "a forced fourfold marks repetition negative (got {})",
            top.repetition,
        );

        // `ply == 1` places the entire cycle before the search root.
        assert_eq!(
            p.is_repetition(1),
            RepetitionState::Draw,
            "a forced fourfold is reported regardless of the search ply",
        );
    }

    #[test]
    fn is_repetition_two_fold_before_root_is_not_reported() {
        // One four-ply cycle, so the root recurs once at a positive distance.
        // Only the fourfold path crosses the search root.
        let mut p = setup_king_shuffle_pos();
        for m in shuffle_cycle() {
            p.do_move(m);
        }
        let top = p.history.last().expect("history is non-empty");
        assert_eq!(top.repetition, 4, "second occurrence: positive distance");
        assert_eq!(top.repetition_times, 1);

        assert_eq!(
            p.is_repetition(4),
            RepetitionState::None,
            "a two-fold reaching the root (ply == distance) is not an immediate draw",
        );
        assert_eq!(
            p.is_repetition(5),
            RepetitionState::Draw,
            "with the earlier occurrence strictly after the root it is a draw",
        );
    }

    #[test]
    fn plies_from_null_blocks_repetition_detection_across_a_null_move() {
        // A four-ply king shuffle that returns to the root — ordinarily a draw.
        let mut p = setup_king_shuffle_pos();
        for m in shuffle_cycle() {
            p.do_move(m);
        }
        assert_eq!(
            p.is_repetition(16),
            RepetitionState::Draw,
            "the closed four-ply cycle is an ordinary draw",
        );

        // Two null moves return the board to the root position yet again, but
        // collapse the look-back window to zero.
        p.do_null_move(); // Black passes
        p.do_null_move(); // White passes → root board, Black to move
        assert_eq!(
            p.current_plies_from_null(),
            0,
            "a null move resets plies_from_null",
        );
        assert_eq!(
            p.is_repetition(16),
            RepetitionState::None,
            "plies_from_null == 0 blocks detection across the null move",
        );

        for m in shuffle_cycle() {
            p.do_move(m);
        }
        assert_eq!(
            p.is_repetition(16),
            RepetitionState::Draw,
            "four real plies past the null restore detection",
        );
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn gives_check_matches_stored_history_flag_over_playouts() {
        // Lives here rather than in `search_movegen` so it can read the private
        // `history`.
        const MIN_PLIES: usize = 30;
        for (fi, sfen) in FIXTURE_SFENS.iter().enumerate() {
            let mut pos = parse_sfen(sfen).expect("fixture sfen parses");
            let mut rng = Rng(0x2360_ED05_1FC6_5DA4 ^ (fi as u64).wrapping_add(1));
            let mut stack: Vec<(Move, Undo)> = Vec::new();
            let mut legal: Vec<Move> = Vec::new();
            let mut plies = 0usize;
            while plies < MIN_PLIES {
                legal.clear();
                pos.generate_legal_all(&mut legal);
                if legal.is_empty() {
                    while let Some((m, u)) = stack.pop() {
                        pos.undo_move(m, u);
                    }
                    continue;
                }
                for &m in &legal {
                    let predicted = pos.gives_check(m);
                    let u = pos.do_move(m);
                    let stored = pos.history.last().unwrap().gives_check;
                    pos.undo_move(m, u);
                    assert_eq!(
                        predicted, stored,
                        "fixture {fi} ply {plies}: gives_check(m) != stored flag",
                    );
                }
                let m = legal[rng.pick(legal.len())];
                let u = pos.do_move(m);
                stack.push((m, u));
                plies += 1;
            }
        }
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn is_repetition_survives_do_undo_round_trip() {
        let (sfen, moves) = load_sennichite();
        let mut pos = parse_sfen(&sfen).expect("fixture sfen parses");
        let mut stack: Vec<(Move, Undo)> = Vec::new();
        let mut forward: Vec<RepetitionState> = Vec::new();
        for usi in &moves {
            let m = crate::move_::parse_usi_move(usi, &pos).expect("fixture move parses");
            let u = pos.do_move(m);
            forward.push(pos.is_repetition(16));
            stack.push((m, u));
        }

        while let Some((m, u)) = stack.pop() {
            pos.undo_move(m, u);
            let depth = stack.len();
            let expected = if depth >= 1 {
                forward[depth - 1]
            } else {
                RepetitionState::None
            };
            assert_eq!(
                pos.is_repetition(16),
                expected,
                "is_repetition mismatch after unwinding to depth {depth}",
            );
        }
    }

    /// Assert two positions are identical element-by-element, including the
    /// derived piece sets and every key — which both `Board`'s and
    /// `Position`'s `PartialEq` exclude.
    fn assert_positions_identical(a: &Position, b: &Position, ctx: &str) {
        for index in 0..Square::COUNT as u8 {
            let sq = Square::from_index(index).unwrap();
            assert_eq!(a.board().get(sq), b.board().get(sq), "{ctx}: square {sq:?}");
        }
        assert_eq!(
            a.board().occupied(),
            b.board().occupied(),
            "{ctx}: occupied set",
        );
        for color in [Color::Black, Color::White] {
            assert_eq!(
                a.board().pieces_color(color),
                b.board().pieces_color(color),
                "{ctx}: by_color {color:?}",
            );
            for pat in 0..crate::board::PATTERN_COUNT {
                assert_eq!(
                    a.board().pieces_pattern(color, pat),
                    b.board().pieces_pattern(color, pat),
                    "{ctx}: by_pattern {color:?} {pat}",
                );
            }
        }
        assert_eq!(a.key(), b.key(), "{ctx}: key");
        assert_eq!(a.board_key(), b.board_key(), "{ctx}: board_key");
        assert_eq!(a.hand_key(), b.hand_key(), "{ctx}: hand_key");
        assert_eq!(a.pawn_key(), b.pawn_key(), "{ctx}: pawn_key");
        assert_eq!(
            a.minor_piece_key(),
            b.minor_piece_key(),
            "{ctx}: minor_piece_key",
        );
        for color in [Color::Black, Color::White] {
            assert_eq!(
                a.non_pawn_key(color),
                b.non_pawn_key(color),
                "{ctx}: non_pawn_key {color:?}",
            );
        }
        assert_eq!(a.side_to_move(), b.side_to_move(), "{ctx}: side_to_move");
        assert_eq!(a.ply(), b.ply(), "{ctx}: ply");
        for color in [Color::Black, Color::White] {
            assert_eq!(a.hand(color), b.hand(color), "{ctx}: hand {color:?}");
        }
        // `root` is not compared: it is left populated across an undo by
        // design. Every repetition-relevant field lives in `history`.
        assert_eq!(a.history, b.history, "{ctx}: history");
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn do_undo_round_trip_is_identity_over_playouts() {
        const MIN_PLIES: usize = 40;

        for (fi, sfen) in FIXTURE_SFENS.iter().enumerate() {
            let mut pos = parse_sfen(sfen).expect("fixture sfen parses");
            let mut rng = Rng(0x1D87_2B41_9C6F_00A5 ^ (fi as u64).wrapping_add(1));
            let mut stack: Vec<(Move, Undo)> = Vec::new();
            let mut legal: Vec<Move> = Vec::new();

            let mut plies = 0usize;
            while plies < MIN_PLIES {
                if !pos.in_check() {
                    let snapshot = pos.clone();
                    pos.do_null_move();
                    pos.undo_null_move();
                    assert_positions_identical(
                        &pos,
                        &snapshot,
                        &format!("fixture {fi} ply {plies} null round-trip"),
                    );
                }

                legal.clear();
                pos.generate_legal_all(&mut legal);
                if legal.is_empty() {
                    // Terminal: unwind fully and restart.
                    while let Some((m, u)) = stack.pop() {
                        pos.undo_move(m, u);
                    }
                    continue;
                }

                let m = legal[rng.pick(legal.len())];

                let snapshot = pos.clone();
                let u = pos.do_move(m);
                pos.undo_move(m, u);
                assert_positions_identical(
                    &pos,
                    &snapshot,
                    &format!("fixture {fi} ply {plies} do/undo round-trip"),
                );

                let u = pos.do_move(m);
                stack.push((m, u));
                plies += 1;
            }

            while let Some((m, u)) = stack.pop() {
                pos.undo_move(m, u);
            }
        }
    }
}
