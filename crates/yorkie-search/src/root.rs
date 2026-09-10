//! Root search data types and helpers, ported from the reference's
//! `start_searching` / `iterative_deepening` / `search<Root>`.
//!
//! This module holds the *pure* pieces: the [`RootMove`] / [`RootOutcome`]
//! value types, the [`generate_root_moves`] list builder, and the
//! [`declaration_win`] nyugyoku predicate. The search glue lives on
//! [`crate::qsearch::QSearch::run_root`] instead, so it shares the node
//! counter, transposition table, search stack, history tables and network with
//! the interior and quiescence searches it recurses into.

use yorkie_state::{Color, ExtMove, Move, PieceKind, Position, Square};
use yorkie_storage::Value;

use crate::config::GENERATE_ALL_LEGAL_MOVES;

/// `VALUE_INFINITE`, duplicated from [`crate::qsearch`] so the pure root-move
/// initialisation does not reach into that module's private constants.
const VALUE_INFINITE: Value = 32001;

/// `MAX_PLY`, used by the proven-win / proven-loss thresholds below.
const MAX_PLY: Value = 246;
/// `VALUE_MATE`.
const VALUE_MATE: Value = 32000;
/// `VALUE_TB_WIN_IN_MAX_PLY`: the `is_win` / `is_loss` threshold,
/// `VALUE_MATE - MAX_PLY`.
const VALUE_TB_WIN_IN_MAX_PLY: Value = VALUE_MATE - MAX_PLY;

/// `is_win(v)`: a proven-win (mate/TB-win) score.
fn is_win(v: Value) -> bool {
    v >= VALUE_TB_WIN_IN_MAX_PLY
}

/// `is_loss(v)`: a proven-loss (mated/TB-loss) score.
fn is_loss(v: Value) -> bool {
    v <= -VALUE_TB_WIN_IN_MAX_PLY
}

/// `RootMove::meanSquaredScore` initial value. The reference holds this in an
/// `int`; it is `i64` here so the `value * abs(value)` moving average cannot
/// overflow. Only the first iteration's `delta` reads it, so the widening is
/// unobservable.
const MEAN_SQUARED_INIT: i64 = -(VALUE_INFINITE as i64 * VALUE_INFINITE as i64);

/// Capacity every [`RootMove::pv`] is built and copied with: the root move plus
/// the deepest line below it the search can return, which is the same bound the
/// search stack's own PV cells hold. A PV update then writes into the buffer the
/// move already owns instead of growing one.
const PV_CAPACITY: usize = MAX_PLY as usize + 1;

/// One root move and the per-iteration statistics `search<Root>` maintains for
/// it. Only the fields the depth-1 path reads or writes are kept.
#[derive(Debug)]
pub struct RootMove {
    /// `pv[0]` — the move itself.
    pub mv: Move,
    /// Principal variation: `mv` followed by the searched child PV.
    pub pv: Vec<Move>,
    /// This iteration's search score (`-VALUE_INFINITE` when not the PV / not
    /// alpha-improving; the stable sort keeps such moves in place).
    pub score: Value,
    /// The USI-reported score (equals `score`, or the clamped window bound on a
    /// fail high/low).
    pub uci_score: Value,
    /// Previous iteration's `score` (the sort tie-break). All `-VALUE_INFINITE`
    /// in the single depth-1 iteration.
    pub previous_score: Value,
    /// Moving average of `score` (aspiration input; not read at depth 1).
    pub average_score: Value,
    /// Moving average of `score * |score|` (aspiration `delta` input).
    pub mean_squared_score: i64,
    /// `selDepth` recorded when this move became the PV.
    pub sel_depth: i32,
    /// `uciScore` is a lower bound (fail high). Read only to mark a PV line
    /// `lowerbound`, so it exists only in a build that prints one.
    #[cfg(feature = "verbose2")]
    pub score_lowerbound: bool,
    /// `uciScore` is an upper bound (fail low). Read only to mark a PV line
    /// `upperbound`, so it exists only in a build that prints one.
    #[cfg(feature = "verbose2")]
    pub score_upperbound: bool,
    /// Nodes spent searching this root move's subtree.
    pub effort: u64,
}

/// `Vec::clone` allocates exactly the length it copies, so a derived clone would
/// hand the copy a PV buffer that grows again on its first update; `clone` here
/// keeps [`PV_CAPACITY`] instead. `clone_from` goes further and writes into the
/// destination's existing buffer, which is what lets a search record its best
/// move once per iteration without reaching the allocator at all.
impl Clone for RootMove {
    fn clone(&self) -> Self {
        let mut copy = Self::new(self.mv);
        copy.clone_from(self);
        copy
    }

    fn clone_from(&mut self, source: &Self) {
        self.mv = source.mv;
        self.pv.clear();
        self.pv.extend_from_slice(&source.pv);
        self.score = source.score;
        self.uci_score = source.uci_score;
        self.previous_score = source.previous_score;
        self.average_score = source.average_score;
        self.mean_squared_score = source.mean_squared_score;
        self.sel_depth = source.sel_depth;
        #[cfg(feature = "verbose2")]
        {
            self.score_lowerbound = source.score_lowerbound;
            self.score_upperbound = source.score_upperbound;
        }
        self.effort = source.effort;
    }
}

impl RootMove {
    /// A fresh root move: `pv == [m]`, every score at its `-VALUE_INFINITE` /
    /// `MEAN_SQUARED_INIT` sentinel.
    pub fn new(m: Move) -> Self {
        let mut pv = Vec::with_capacity(PV_CAPACITY);
        pv.push(m);
        Self {
            mv: m,
            pv,
            score: -VALUE_INFINITE,
            uci_score: -VALUE_INFINITE,
            previous_score: -VALUE_INFINITE,
            average_score: -VALUE_INFINITE,
            mean_squared_score: MEAN_SQUARED_INIT,
            sel_depth: 0,
            #[cfg(feature = "verbose2")]
            score_lowerbound: false,
            #[cfg(feature = "verbose2")]
            score_upperbound: false,
            effort: 0,
        }
    }
}

/// How the root search resolved (the reference's three `search_skipped`
/// pre-search exits plus the normal search).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RootKind {
    /// A normal search produced the result.
    Normal,
    /// No legal move: `bestmove resign` with `mated_in(1)`.
    Resign,
    /// `Position::DeclarationWin()` succeeded: `bestmove win` with `mate_in(1)`.
    DeclarationWin,
}

/// Outcome of a depth-1 root search — the fields the USI `info` / `bestmove`
/// output (and the parity tests) consume.
#[derive(Clone, Debug)]
pub struct RootOutcome {
    /// The chosen move. A real move for [`RootKind::Normal`]; the
    /// [`Move::resign`] / [`Move::win`] sentinel otherwise.
    pub best_move: Move,
    /// Score from the root side-to-move's point of view (`rootMoves[0].uciScore`,
    /// or `mated_in(1)` / `mate_in(1)` for the skipped exits).
    pub score: Value,
    /// Total nodes searched (`do_move` count); `0` for the skipped exits.
    pub nodes: u64,
    /// Principal variation (`rootMoves[0].pv`, possibly ponder-extended).
    pub pv: Vec<Move>,
    /// Iterative-deepening depth the reported result was completed at, so a
    /// time-aborted search reports the depth it finished rather than the one it
    /// was cut off in. `0` for the resign / declaration-win pre-search exits.
    pub depth: i32,
    /// Maximum selective depth reached.
    pub sel_depth: i32,
    /// Which resolution produced this outcome.
    pub kind: RootKind,
}

/// The per-worker facts the Lazy-SMP thread vote consumes: the last iteration's
/// raw `rootMoves[0].score` (NOT the USI-clamped `uciScore`), the first PV move
/// it votes for, the PV length (the truncated-PV guard reads `pv.len() > 2`),
/// and the worker's `completedDepth` (`0` if it never completed an iteration).
#[derive(Clone, Debug)]
pub struct WorkerVote {
    /// `rootMoves[0].score` — the raw search score the vote weights by.
    pub score: Value,
    /// `rootMoves[0].pv[0]` — the move this worker would play.
    pub pv0: Move,
    /// `rootMoves[0].pv.size()` — read by the truncated-PV tiebreak.
    pub pv_len: usize,
    /// `completedDepth` — the last fully-completed iterative-deepening depth.
    pub completed_depth: i32,
}

/// `YaneuraOuWorker::get_best_thread`: pick the index of the worker whose
/// result the engine reports.
///
/// `workers[0]` must be the main worker, and the slice is never empty.
pub fn select_best_worker(workers: &[WorkerVote]) -> usize {
    debug_assert!(!workers.is_empty());

    // `minScore = min over workers of rootMoves[0].score`. Seeded from
    // VALUE_NONE like the reference, but every worker has a valid score so the
    // seed is always superseded.
    let min_score = workers.iter().map(|w| w.score).min().unwrap_or(0);

    // `voting_value(w) = (score - minScore + 14) * completedDepth`, widened to
    // i64 so the product cannot overflow `Value`'s i32 range.
    let voting_value = |w: &WorkerVote| -> i64 {
        (w.score as i64 - min_score as i64 + 14) * w.completed_depth as i64
    };

    // `votes[pv[0]] += voting_value(w)` summed over workers. A move's total is
    // summed where it is read rather than tallied into a map first: only the two
    // moves being compared are ever asked for, and the candidates are at most
    // one per worker, so the tally would be scanned as a short list anyway — and
    // this way the vote holds no buffer at all.
    let vote_total = |mv: Move| -> i64 {
        workers
            .iter()
            .filter(|w| w.pv0 == mv)
            .map(&voting_value)
            .sum()
    };

    // Scan all workers keeping the best. `best` starts at the main worker
    // (index 0); comparing it against itself is a no-op.
    let mut best = 0usize;
    for i in 0..workers.len() {
        let cur = &workers[best];
        let th = &workers[i];

        let best_score = cur.score;
        let new_score = th.score;

        let best_vote = vote_total(cur.pv0);
        let new_vote = vote_total(th.pv0);

        let best_in_win = is_win(best_score);
        let new_in_win = is_win(new_score);
        let best_in_loss = best_score != -VALUE_INFINITE && is_loss(best_score);
        let new_in_loss = new_score != -VALUE_INFINITE && is_loss(new_score);

        // The truncated-PV guard: a vote total only counts when the PV has more
        // than two moves (`pv.size() > 2`).
        let better_voting_value =
            voting_value(th) * (th.pv_len > 2) as i64 > voting_value(cur) * (cur.pv_len > 2) as i64;

        if best_in_win {
            // Keep the shortest mate: switch only to a higher proven-win score.
            if new_score > best_score {
                best = i;
            }
        } else if best_in_loss {
            // Keep the longest defence: switch only to a lower proven loss.
            if new_in_loss && new_score < best_score {
                best = i;
            }
        } else if new_in_win
            || new_in_loss
            || (!is_loss(new_score)
                && (new_vote > best_vote || (new_vote == best_vote && better_voting_value)))
        {
            best = i;
        }
    }

    best
}

/// Build the root-move list in `MoveList<LEGAL>` order — the reference's
/// `ThreadPool::start_thinking`.
///
/// The legality compaction of `generateMoves<LEGAL>` overwrites an illegal move
/// at the cursor with the current last move and shrinks the list. That is *not*
/// an order-preserving filter, and reproducing it exactly is what fixes
/// `root_moves[0]` — the move the root search treats as its TT move.
pub fn generate_root_moves(pos: &Position) -> Vec<RootMove> {
    // The generators emit `ExtMove`; the root list only needs the
    // `Move`, read via `.mv` in the legality compaction and `RootMove::new`.
    let mut pseudo: Vec<ExtMove> = Vec::new();
    if pos.in_check() {
        pos.generate_evasions::<GENERATE_ALL_LEGAL_MOVES>(&mut pseudo);
    } else {
        pos.generate_non_evasions::<GENERATE_ALL_LEGAL_MOVES>(&mut pseudo);
    }

    // `while (cur != last) if (!legal(*cur)) *cur = *(--last); else ++cur;`
    let mut cur = 0usize;
    let mut last = pseudo.len();
    while cur != last {
        if pos.is_legal(pseudo[cur].mv) {
            cur += 1;
        } else {
            pseudo[cur] = pseudo[last - 1];
            last -= 1;
        }
    }
    pseudo.truncate(last);

    pseudo.into_iter().map(|e| RootMove::new(e.mv)).collect()
}

/// The entering-king (nyugyoku) declaration rule, selected by the
/// `EnteringKingRule` USI option. Order and semantics mirror the reference
/// `enum EnteringKingRule`; the option-string spellings are `EKR_STRINGS`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnteringKingRule {
    /// `EKR_NONE` / `NoEnteringKing` — declaration disabled everywhere.
    None,
    /// `EKR_24_POINT` / `CSARule24` — 24-point law (declare at 31+, both sides).
    Point24,
    /// `EKR_24_POINT_H` / `CSARule24H` — 24-point law, handicap-aware.
    Point24H,
    /// `EKR_27_POINT` / `CSARule27` — CSA 27-point law (Black 28, White 27).
    Point27,
    /// `EKR_27_POINT_H` / `CSARule27H` — CSA 27-point law, handicap-aware.
    Point27H,
    /// `EKR_TRY_RULE` / `TryRule` — reach the opponent king's initial square.
    Try,
}

impl EnteringKingRule {
    /// The `EKR_STRINGS` choice list in the reference's exact order — the `var`
    /// values of the `EnteringKingRule` combo option.
    pub const STRINGS: [&'static str; 6] = [
        "NoEnteringKing",
        "CSARule24",
        "CSARule24H",
        "CSARule27",
        "CSARule27H",
        "TryRule",
    ];

    /// Map an option string to its rule. An unrecognised string falls back to
    /// [`EnteringKingRule::None`]; the config schema only accepts a declared
    /// spelling, so the fallback is unreachable in practice.
    ///
    /// `const` because the rule the engine plays under is compiled in, and
    /// resolving it here is what lets the node-side check be the one branch of
    /// that rule.
    pub const fn from_option(s: &str) -> Self {
        // A `const fn` cannot `match` on a `&str`, so the choice list is walked
        // by byte comparison instead.
        if str_eq(s, "NoEnteringKing") {
            Self::None
        } else if str_eq(s, "CSARule24") {
            Self::Point24
        } else if str_eq(s, "CSARule24H") {
            Self::Point24H
        } else if str_eq(s, "CSARule27") {
            Self::Point27
        } else if str_eq(s, "CSARule27H") {
            Self::Point27H
        } else if str_eq(s, "TryRule") {
            Self::Try
        } else {
            Self::None
        }
    }

    /// Whether this rule charges White the material a handicap removed, which is
    /// the only thing that makes a threshold depend on the position.
    const fn is_handicap_aware(self) -> bool {
        matches!(self, Self::Point24H | Self::Point27H)
    }
}

/// Byte-wise `&str` equality, for [`EnteringKingRule::from_option`].
const fn str_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// The entering-king rule this binary plays under, resolved from the config at
/// build time. Every declaration check below branches on it, so the compiler
/// keeps exactly the chosen rule's arm.
pub const ENTERING_KING_RULE: EnteringKingRule =
    EnteringKingRule::from_option(crate::config::ENTERING_KING_RULE);

/// The compiled rule's per-side thresholds before any handicap adjustment —
/// `Position::update_entering_point`'s starting pair, which for every rule but
/// the two handicap-aware ones is also its final one.
const BASE_ENTERING_KING_POINTS: [i32; Color::COUNT] = match ENTERING_KING_RULE {
    EnteringKingRule::Point24 | EnteringKingRule::Point24H => [31, 31],
    EnteringKingRule::Point27 | EnteringKingRule::Point27H => [28, 27],
    // `EKR_NONE` / `EKR_TRY_RULE` never consult the thresholds.
    EnteringKingRule::None | EnteringKingRule::Try => [0, 0],
};

/// The per-side point thresholds the compiled rule needs, precomputed once per
/// `go` from the root position — the reference's `set_ekr` state.
///
/// Only the handicap-aware rules read a position at all; under every other rule
/// this is [`BASE_ENTERING_KING_POINTS`] and the node-side check never touches
/// the snapshot. Total material on the board and in both hands is invariant
/// across a game, since captures only move pieces to hands, so a snapshot taken
/// from the root is exact for every node of that search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EnteringKingConfig {
    /// `enteringKingPoint[]`, indexed by [`Color::index`] (Black 0, White 1).
    points: [i32; Color::COUNT],
}

impl EnteringKingConfig {
    /// Precompute `enteringKingPoint[]` against `root`
    /// (`Position::update_entering_point`).
    pub fn new(root: &Position) -> Self {
        Self {
            points: entering_king_points(ENTERING_KING_RULE, root),
        }
    }
}

impl Default for EnteringKingConfig {
    /// The compiled rule's unadjusted thresholds. Only the handicap-aware rules
    /// move off them, and they move by an amount a root position states, which
    /// a default has none of — the fixed-depth path (`QSearch::run_root`) relies
    /// on this.
    fn default() -> Self {
        Self {
            points: BASE_ENTERING_KING_POINTS,
        }
    }
}

/// `Position::update_entering_point`: the per-side entering-king point
/// thresholds for `rule`, computed from `pos`.
fn entering_king_points(rule: EnteringKingRule, pos: &Position) -> [i32; Color::COUNT] {
    let mut points = match rule {
        EnteringKingRule::Point24 | EnteringKingRule::Point24H => [31, 31],
        EnteringKingRule::Point27 | EnteringKingRule::Point27H => [28, 27],
        // `EKR_NONE` / `EKR_TRY_RULE` never consult the thresholds.
        EnteringKingRule::None | EnteringKingRule::Try => return [0, 0],
    };

    if rule.is_handicap_aware() {
        // Total material points: every piece on the board scores 1 (kings
        // included in this popcount), big pieces (bishop/rook family, promotions
        // included) score 4 more; hands add small 1 / big 5 (kings never in
        // hand). A full set is 56.
        let mut p: i32 = 0;
        for sq in (0..Square::COUNT as u8).filter_map(Square::from_index) {
            if let Some(pc) = pos.board().get(sq) {
                p += 1;
                if matches!(pc.kind, PieceKind::Bishop | PieceKind::Rook) {
                    p += 4;
                }
            }
        }
        for color in [Color::Black, Color::White] {
            let h = pos.hand(color);
            let c = |k: PieceKind| h.count(k) as i32;
            p += c(PieceKind::Pawn)
                + c(PieceKind::Lance)
                + c(PieceKind::Knight)
                + c(PieceKind::Silver)
                + c(PieceKind::Gold)
                + (c(PieceKind::Bishop) + c(PieceKind::Rook)) * 5;
        }
        // The deficit from a full set is charged to White only — the handicap
        // giver is treated as White (AobaZero convention).
        if p != 56 {
            points[Color::White.index()] -= 56 - p;
        }
    }

    points
}

/// `Position::DeclarationWin()` under the compiled [`ENTERING_KING_RULE`]:
///
/// * [`EnteringKingRule::None`] — always `None`.
/// * point rules — [`Move::win`] when the side to move may declare: its king is
///   inside the enemy three ranks, it is not in check, it has at least 11
///   pieces (king included) in the enemy field, and its point total reaches the
///   per-side threshold.
/// * [`EnteringKingRule::Try`] — the king move onto the opponent king's initial
///   square, when that square is adjacent to our king, holds no own piece, and
///   is unattacked once our king vacates its square.
///
/// The rule is a constant, so what a build compiles here is the one arm it
/// selects — and, under every rule but the two handicap-aware ones, the
/// thresholds are constants too and `config` is never read.
pub fn declaration_win(pos: &Position, config: &EnteringKingConfig) -> Option<Move> {
    match ENTERING_KING_RULE {
        EnteringKingRule::None => None,
        EnteringKingRule::Point24 | EnteringKingRule::Point27 => {
            declaration_win_points(pos, BASE_ENTERING_KING_POINTS)
        }
        EnteringKingRule::Point24H | EnteringKingRule::Point27H => {
            declaration_win_points(pos, config.points)
        }
        EnteringKingRule::Try => declaration_win_try(pos),
    }
}

/// The CSA point-law branch with the per-side threshold taken from the
/// precomputed `points` (`enteringKingPoint[us]`).
fn declaration_win_points(pos: &Position, points: [i32; Color::COUNT]) -> Option<Move> {
    let us = pos.side_to_move();

    // (e) not in check.
    if pos.in_check() {
        return None;
    }

    // The enemy field is the three ranks nearest the opponent: ranks 0..=2 for
    // Black (which advances toward rank 0), ranks 6..=8 for White.
    let in_enemy_field = |sq: Square| match us {
        Color::Black => sq.rank() <= 2,
        Color::White => sq.rank() >= 6,
    };

    // (b) the king is inside the enemy field.
    let king_sq = find_king(pos, us)?;
    if !in_enemy_field(king_sq) {
        return None;
    }

    // (d) at least 11 own pieces (king included) in the enemy field, and the
    // count of big pieces (bishop/rook incl. horse/dragon) there.
    let mut p1: i32 = 0;
    let mut p2: i32 = 0;
    for sq in (0..Square::COUNT as u8).filter_map(Square::from_index) {
        if let Some(p) = pos.board().get(sq)
            && p.color == us
            && in_enemy_field(sq)
        {
            p1 += 1;
            if matches!(p.kind, PieceKind::Bishop | PieceKind::Rook) {
                p2 += 1;
            }
        }
    }
    if p1 < 11 {
        return None;
    }

    // (c) point total: small pieces 1, big pieces 5, king excluded (`- 1`).
    let h = pos.hand(us);
    let count = |k: PieceKind| h.count(k) as i32;
    let score = p1 + p2 * 4 - 1
        + count(PieceKind::Pawn)
        + count(PieceKind::Lance)
        + count(PieceKind::Knight)
        + count(PieceKind::Silver)
        + count(PieceKind::Gold)
        + (count(PieceKind::Bishop) + count(PieceKind::Rook)) * 5;

    if score < points[us.index()] {
        return None;
    }

    Some(Move::win())
}

/// The try-rule branch: return the king move onto the opponent king's initial
/// square when the three try conditions hold.
fn declaration_win_try(pos: &Position) -> Option<Move> {
    let us = pos.side_to_move();

    // The opponent king's initial square: 5a for Black, 5i for White. Both are
    // fixed on-board coordinates, so `Square::new` never returns `None`; `?`
    // keeps the impossible case panic-free.
    let try_sq = match us {
        Color::Black => Square::new(4, 0),
        Color::White => Square::new(4, 8),
    }?;

    let king_sq = find_king(pos, us)?;

    // 1) our king can step onto the try square (kingEffect adjacency).
    if !king_adjacent(king_sq, try_sq) {
        return None;
    }

    // 2) no *own* piece occupies the try square (an enemy piece there is a
    //    capture-try and does not block).
    if matches!(pos.board().get(try_sq), Some(p) if p.color == us) {
        return None;
    }

    // 3) the opponent has no effect on the try square once our king leaves
    //    king_sq (`effected_to(~us, king_try_sq, king_sq)`).
    if pos.is_attacked_discounting(try_sq, us.flip(), king_sq) {
        return None;
    }

    // The king move onto the try square, encoded exactly as the generator
    // would. `king_sq` came from `find_king`, so it always holds our king; `?`
    // keeps the impossible case panic-free.
    let king_piece = pos.board().get(king_sq)?;
    Some(Move::make(king_sq, try_sq, king_piece))
}

/// The side-to-move-agnostic king locator shared by the declaration branches.
fn find_king(pos: &Position, us: Color) -> Option<Square> {
    (0..Square::COUNT as u8)
        .filter_map(Square::from_index)
        .find(|&sq| {
            matches!(pos.board().get(sq), Some(p) if p.color == us && p.kind == PieceKind::King)
        })
}

/// True iff `sq` lies on one of `king_sq`'s eight king steps (`kingEffect`).
fn king_adjacent(king_sq: Square, sq: Square) -> bool {
    let df = (king_sq.file() as i8 - sq.file() as i8).abs();
    let dr = (king_sq.rank() as i8 - sq.rank() as i8).abs();
    df <= 1 && dr <= 1 && (df, dr) != (0, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use yorkie_state::parse_sfen;

    fn pos(sfen: &str) -> Position {
        parse_sfen(sfen).expect("valid SFEN")
    }

    const STARTPOS: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
    const IN_CHECK: &str = "4k4/9/4r4/9/9/9/4K3B/9/9 b RG2gs2n3p 1";
    const CHECKMATE: &str = "4K4/3ggg3/4k4/9/9/9/9/9/9 b - 1";
    // Black king on 5b inside the enemy field, 12 own pieces there (2 dragons +
    // 2 horses = 4 big) plus a rook in hand: 12 + 4*4 - 1 + 5 = 32 points, a CSA
    // 27-point win, and (unlike a fully walled-in king) it still has legal moves.
    const NYUGYOKU: &str = "+R+R+B+B5/3GKG3/2SGGGS2/9/9/9/9/9/4k4 b R 1";

    #[test]
    fn root_moves_equal_legal_captures_and_quiets_when_not_in_check() {
        let p = pos(STARTPOS);
        assert!(!p.in_check());
        let rm: HashSet<Move> = generate_root_moves(&p).into_iter().map(|r| r.mv).collect();
        assert!(!rm.is_empty());

        let mut caps: Vec<ExtMove> = Vec::new();
        p.generate_captures::<ALL>(&mut caps);
        let mut quiets: Vec<ExtMove> = Vec::new();
        p.generate_quiets::<ALL>(&mut quiets);
        let legal: HashSet<Move> = caps
            .into_iter()
            .chain(quiets)
            .map(|e| e.mv)
            .filter(|&m| p.is_legal(m))
            .collect();
        assert_eq!(rm, legal);
    }

    #[test]
    fn root_moves_equal_legal_evasions_when_in_check() {
        let p = pos(IN_CHECK);
        assert!(p.in_check());
        let rm: HashSet<Move> = generate_root_moves(&p).into_iter().map(|r| r.mv).collect();

        let mut ev: Vec<ExtMove> = Vec::new();
        p.generate_evasions::<ALL>(&mut ev);
        let legal: HashSet<Move> = ev
            .into_iter()
            .map(|e| e.mv)
            .filter(|&m| p.is_legal(m))
            .collect();
        assert_eq!(rm, legal);
    }

    #[test]
    fn root_moves_empty_when_checkmated() {
        let p = pos(CHECKMATE);
        assert!(p.in_check());
        assert!(generate_root_moves(&p).is_empty());
    }

    #[test]
    fn every_generated_root_move_starts_with_a_full_pv_buffer() {
        for rm in generate_root_moves(&pos(STARTPOS)) {
            assert_eq!(rm.pv.capacity(), PV_CAPACITY);
        }
    }

    #[test]
    fn a_root_move_carries_its_pv_buffer_and_every_field_through_a_copy() {
        let list = generate_root_moves(&pos(STARTPOS));
        let (a, b) = (list[0].mv, list[1].mv);
        let mut src = RootMove::new(a);
        src.pv.push(b);
        src.score = 12;
        src.uci_score = 13;
        src.previous_score = 14;
        src.average_score = 15;
        src.mean_squared_score = 16;
        src.sel_depth = 17;
        src.effort = 18;
        #[cfg(feature = "verbose2")]
        {
            src.score_lowerbound = true;
            src.score_upperbound = true;
        }

        let copy = src.clone();
        assert_eq!(copy.pv.capacity(), PV_CAPACITY);
        assert_eq!(copy.pv, src.pv);

        let mut slot = RootMove::new(b);
        let buffer = slot.pv.as_ptr();
        slot.clone_from(&src);
        assert_eq!(
            slot.pv.as_ptr(),
            buffer,
            "the copy replaced the buffer instead of writing into it"
        );
        assert_eq!(slot.mv, a);
        assert_eq!(slot.pv, src.pv);
        assert_eq!(slot.score, 12);
        assert_eq!(slot.uci_score, 13);
        assert_eq!(slot.previous_score, 14);
        assert_eq!(slot.average_score, 15);
        assert_eq!(slot.mean_squared_score, 16);
        assert_eq!(slot.sel_depth, 17);
        assert_eq!(slot.effort, 18);
        #[cfg(feature = "verbose2")]
        {
            assert!(slot.score_lowerbound);
            assert!(slot.score_upperbound);
        }
    }

    /// The `GenerateAllLegalMoves` setting this binary compiled, which is what
    /// [`generate_root_moves`] generates under.
    const ALL: bool = GENERATE_ALL_LEGAL_MOVES;

    // Rule-aware declaration. Positions are hand-built so a single
    // point of the score / count / threshold formula is exercised at a time.
    // `declaration_win` itself runs under the one rule this binary compiled, so
    // the per-rule cases drive that rule's branch directly.

    /// The declaration `rule` reaches at `p`, with that rule's thresholds taken
    /// from the position.
    fn declaration_under(rule: EnteringKingRule, p: &Position) -> Option<Move> {
        match rule {
            EnteringKingRule::None => None,
            EnteringKingRule::Try => declaration_win_try(p),
            _ => declaration_win_points(p, entering_king_points(rule, p)),
        }
    }

    #[test]
    fn each_declared_spelling_maps_to_its_own_rule() {
        use EnteringKingRule::*;
        const RULES: [EnteringKingRule; 6] = [None, Point24, Point24H, Point27, Point27H, Try];
        for (s, rule) in EnteringKingRule::STRINGS.iter().zip(RULES) {
            assert_eq!(EnteringKingRule::from_option(s), rule, "spelling `{s}`");
        }
        // An undeclared spelling cannot enable a declaration.
        assert_eq!(EnteringKingRule::from_option("CSARule28"), None);
    }

    #[test]
    fn declaration_win_takes_the_compiled_rules_branch() {
        // Both a declarable and an undeclarable position, so a rule that never
        // declares is distinguished from one that does.
        for sfen in [NYUGYOKU, STARTPOS, IN_CHECK] {
            let p = pos(sfen);
            assert_eq!(
                declaration_win(&p, &EnteringKingConfig::new(&p)),
                declaration_under(ENTERING_KING_RULE, &p),
                "compiled rule {ENTERING_KING_RULE:?} on `{sfen}`"
            );
        }
    }

    #[test]
    fn point_rules_decline_a_normal_position() {
        assert!(declaration_under(EnteringKingRule::Point27, &pos(STARTPOS)).is_none());
        // In check ⇒ no declaration even with pieces forward.
        assert!(declaration_under(EnteringKingRule::Point27, &pos(IN_CHECK)).is_none());
    }

    #[test]
    fn point27_detects_a_nyugyoku_position() {
        let p = pos(NYUGYOKU);
        assert!(!p.in_check());
        assert_eq!(
            declaration_under(EnteringKingRule::Point27, &p),
            Some(Move::win())
        );
    }

    #[test]
    fn point27_black_threshold_is_28() {
        // Black king + 10 golds in the enemy field (p1 = 11, board score 10) plus
        // a hand of 18 pawns → 28, exactly the Black threshold. (Hand counts are
        // capped at their real per-piece maxima — e.g. a rook holds at most 2 —
        // so pawns are used to dial the score freely.)
        let declare = pos("GGGGKGGGG/GG7/9/9/9/9/9/9/8k b 18P 1");
        assert_eq!(
            declaration_under(EnteringKingRule::Point27, &declare),
            Some(Move::win())
        );
        // One point of hand less (17P → 17) → 27 < 28, no declaration.
        let decline = pos("GGGGKGGGG/GG7/9/9/9/9/9/9/8k b 17P 1");
        assert!(declaration_under(EnteringKingRule::Point27, &decline).is_none());
    }

    #[test]
    fn point27_white_threshold_is_27() {
        // Mirror image: White king + 10 golds in White's enemy field (ranks g-i),
        // hand of 17 pawns → 27, exactly the White threshold.
        let declare = pos("8K/9/9/9/9/9/9/gg7/ggggkgggg w 17p 1");
        assert_eq!(
            declaration_under(EnteringKingRule::Point27, &declare),
            Some(Move::win())
        );
        // 16p → 26 < 27, no declaration.
        let decline = pos("8K/9/9/9/9/9/9/gg7/ggggkgggg w 16p 1");
        assert!(declaration_under(EnteringKingRule::Point27, &decline).is_none());
    }

    #[test]
    fn point24_threshold_is_31_both_colors() {
        // 24-point law: both sides need 31. Board score 10 + hand 21 (2R + 11P) = 31.
        let black_declare = pos("GGGGKGGGG/GG7/9/9/9/9/9/9/8k b 2R11P 1");
        assert_eq!(
            declaration_under(EnteringKingRule::Point24, &black_declare),
            Some(Move::win())
        );
        let black_decline = pos("GGGGKGGGG/GG7/9/9/9/9/9/9/8k b 2R10P 1"); // 30
        assert!(declaration_under(EnteringKingRule::Point24, &black_decline).is_none());

        let white_declare = pos("8K/9/9/9/9/9/9/gg7/ggggkgggg w 2r11p 1");
        assert_eq!(
            declaration_under(EnteringKingRule::Point24, &white_declare),
            Some(Move::win())
        );
        let white_decline = pos("8K/9/9/9/9/9/9/gg7/ggggkgggg w 2r10p 1"); // 30
        assert!(declaration_under(EnteringKingRule::Point24, &white_decline).is_none());
    }

    #[test]
    fn piece_count_gate_needs_eleven_in_the_enemy_field() {
        // The hand alone makes the score pass for both fields, so the only
        // thing that can decline the 10-piece case is the count gate.
        let eleven = pos("GGGGKGGGG/GG7/9/9/9/9/9/9/8k b 2R9P 1"); // king + 10 golds
        assert_eq!(
            declaration_under(EnteringKingRule::Point27, &eleven),
            Some(Move::win())
        );
        let ten = pos("GGGGKGGGG/G8/9/9/9/9/9/9/8k b 2R9P 1"); // king + 9 golds
        assert!(declaration_under(EnteringKingRule::Point27, &ten).is_none());
    }

    #[test]
    fn king_outside_enemy_field_never_declares() {
        // Black king on rank d (outside the enemy three ranks) with a large hand.
        let p = pos("9/9/9/4K4/9/9/9/9/8k b 2R4P 1");
        assert!(declaration_under(EnteringKingRule::Point27, &p).is_none());
    }

    #[test]
    fn big_piece_on_board_counts_five() {
        // king + 8 golds + 1 dragon + 1 gold in the enemy field: p1 = 11, one big
        // piece → board score 11 + 4·1 − 1 = 14, plus hand 14 (2R4P) = 28. The
        // dragon's +4 over a small piece is the only reason it reaches 28.
        let with_dragon = pos("GGGGKGGGG/+RG7/9/9/9/9/9/9/8k b 2R4P 1");
        assert_eq!(
            declaration_under(EnteringKingRule::Point27, &with_dragon),
            Some(Move::win())
        );
        // Same shape with the dragon replaced by a small piece → 24 < 28.
        let all_small = pos("GGGGKGGGG/GG7/9/9/9/9/9/9/8k b 2R4P 1");
        assert!(declaration_under(EnteringKingRule::Point27, &all_small).is_none());
    }

    #[test]
    fn handicap_variants_lower_white_threshold_by_the_deficit() {
        // Startpos with White's rook and bishop removed: full-set 56 minus two big
        // pieces (5 + 5) → material total 46, a deficit of 10.
        let handicap = pos("lnsgkgsnl/9/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");

        // `_H` rules charge the whole deficit to White; Black never moves.
        assert_eq!(
            entering_king_points(EnteringKingRule::Point27H, &handicap),
            [28, 17]
        );
        assert_eq!(
            entering_king_points(EnteringKingRule::Point24H, &handicap),
            [31, 21]
        );
        // The non-handicap rules ignore the deficit entirely.
        assert_eq!(
            entering_king_points(EnteringKingRule::Point27, &handicap),
            [28, 27]
        );
        assert_eq!(
            entering_king_points(EnteringKingRule::Point24, &handicap),
            [31, 31]
        );
        // A complete set leaves every threshold at its base value, even under `_H`.
        let full = pos(STARTPOS);
        assert_eq!(
            entering_king_points(EnteringKingRule::Point27H, &full),
            [28, 27]
        );
        assert_eq!(
            entering_king_points(EnteringKingRule::Point24H, &full),
            [31, 31]
        );
    }

    #[test]
    fn no_entering_king_never_declares() {
        // A clearly declarable 27-point position yields nothing under `None`,
        // whose thresholds are the inert pair.
        let p = pos(NYUGYOKU);
        assert!(declaration_under(EnteringKingRule::None, &p).is_none());
        assert_eq!(entering_king_points(EnteringKingRule::None, &p), [0, 0]);
    }

    // --- TryRule. ---

    fn sq(file: u8, rank: u8) -> Square {
        Square::new(file, rank).expect("on-board square")
    }

    #[test]
    fn try_rule_black_declares_king_move_to_5a() {
        // Black king on 5b, the try square 5a empty and unattacked.
        let p = pos("9/4K4/9/9/9/9/9/9/8k b - 1");
        let king_sq = sq(4, 1);
        let try_sq = sq(4, 0);
        let expected = Move::make(king_sq, try_sq, p.board().get(king_sq).unwrap());
        assert_eq!(declaration_under(EnteringKingRule::Try, &p), Some(expected));
        // The returned move is exactly what the normal generator produces.
        let mut legal = Vec::new();
        p.generate_legal_all(&mut legal);
        assert!(legal.contains(&expected));
    }

    #[test]
    fn try_rule_white_declares_king_move_to_5i() {
        // White king on 5h, the try square 5i empty and unattacked.
        let p = pos("8K/9/9/9/9/9/9/4k4/9 w - 1");
        let king_sq = sq(4, 7);
        let try_sq = sq(4, 8);
        let expected = Move::make(king_sq, try_sq, p.board().get(king_sq).unwrap());
        assert_eq!(declaration_under(EnteringKingRule::Try, &p), Some(expected));
    }

    #[test]
    fn try_rule_requires_king_adjacency() {
        // Black king on 5c is two ranks from 5a → not adjacent → no declaration.
        let p = pos("9/9/4K4/9/9/9/9/9/8k b - 1");
        assert!(declaration_under(EnteringKingRule::Try, &p).is_none());
    }

    #[test]
    fn try_rule_own_piece_on_the_square_blocks() {
        // Black gold sits on 5a → the try square holds an own piece → blocked.
        let p = pos("4G4/4K4/9/9/9/9/9/9/8k b - 1");
        assert!(declaration_under(EnteringKingRule::Try, &p).is_none());
    }

    #[test]
    fn try_rule_enemy_piece_on_the_square_is_a_capture_try() {
        // A White knight on 5a does not defend its own square and (from 5a) does
        // not attack 5b, so the king may capture-try onto it.
        let p = pos("4n4/4K4/9/9/9/9/9/9/8k b - 1");
        let expected = Move::make(sq(4, 1), sq(4, 0), p.board().get(sq(4, 1)).unwrap());
        assert_eq!(declaration_under(EnteringKingRule::Try, &p), Some(expected));
    }

    #[test]
    fn try_rule_attacked_square_blocks() {
        // A White rook on 9a rakes rank a, so the try square 5a is attacked
        // (independently of our king) → blocked.
        let p = pos("r8/4K4/9/9/9/9/9/9/8k b - 1");
        assert!(declaration_under(EnteringKingRule::Try, &p).is_none());
    }

    #[test]
    fn try_rule_attack_test_discounts_the_moving_king() {
        // A White rook on 5c is blocked from 5a by our own king on 5b. The attack
        // test removes the moving king from the occupancy (the reference's
        // `pieces() ^ kingSq`), revealing the rook's attack on 5a → blocked. A
        // naive check that left the king in place would wrongly allow the try.
        let p = pos("9/4K4/4r4/9/9/9/9/9/8k b - 1");
        let king_sq = sq(4, 1);
        let try_sq = sq(4, 0);
        assert!(declaration_under(EnteringKingRule::Try, &p).is_none());
        // Discounting the king reveals the rook; discounting an unrelated empty
        // square leaves the king blocking it — pinpointing the king as the cause.
        assert!(p.is_attacked_discounting(try_sq, Color::White, king_sq));
        assert!(!p.is_attacked_discounting(try_sq, Color::White, sq(8, 8)));
    }

    // Lazy-SMP thread vote (`select_best_worker` / `get_best_thread`).

    /// Two distinct legal moves from startpos, used as the workers' `pv[0]` votes.
    fn two_moves() -> (Move, Move) {
        let rm = generate_root_moves(&pos(STARTPOS));
        (rm[0].mv, rm[1].mv)
    }

    fn vote(score: Value, pv0: Move, pv_len: usize, completed_depth: i32) -> WorkerVote {
        WorkerVote {
            score,
            pv0,
            pv_len,
            completed_depth,
        }
    }

    #[test]
    fn vote_single_worker_degenerates_to_main() {
        let (a, _) = two_moves();
        assert_eq!(select_best_worker(&[vote(100, a, 1, 5)]), 0);
    }

    #[test]
    fn vote_plain_winner_is_the_higher_vote_total() {
        let (a, b) = two_moves();
        // Equal scores; worker 1 searched deeper, so its move's vote total is
        // higher: votes[a] = 14*5 = 70, votes[b] = 14*10 = 140.
        let workers = [vote(100, a, 3, 5), vote(100, b, 3, 10)];
        assert_eq!(select_best_worker(&workers), 1);
        // Reversed depths ⇒ the main worker (index 0) keeps the higher total.
        let workers = [vote(100, a, 3, 10), vote(100, b, 3, 5)];
        assert_eq!(select_best_worker(&workers), 0);
    }

    #[test]
    fn vote_proven_win_prefers_the_shorter_mate() {
        let (a, b) = two_moves();
        // Incumbent (index 0) is a proven win (mate in 5); a higher proven-win
        // score (mate in 3) is a shorter mate and wins.
        let mate_in = |ply: Value| VALUE_MATE - ply;
        let workers = [vote(mate_in(5), a, 1, 8), vote(mate_in(3), b, 1, 8)];
        assert_eq!(select_best_worker(&workers), 1);
        // A *longer* mate (lower proven-win score) never displaces the incumbent.
        let workers = [vote(mate_in(5), a, 1, 8), vote(mate_in(7), b, 1, 8)];
        assert_eq!(select_best_worker(&workers), 0);
    }

    #[test]
    fn vote_proven_loss_prefers_the_lower_score() {
        let (a, b) = two_moves();
        // Incumbent (index 0) is a proven loss; a *lower* proven-loss score
        // wins (`newThreadScore < bestThreadScore`).
        let mated_in = |ply: Value| -VALUE_MATE + ply;
        let workers = [vote(mated_in(7), a, 1, 8), vote(mated_in(3), b, 1, 8)];
        assert_eq!(select_best_worker(&workers), 1);
        // A higher proven-loss score does not displace the incumbent.
        let workers = [vote(mated_in(3), a, 1, 8), vote(mated_in(7), b, 1, 8)];
        assert_eq!(select_best_worker(&workers), 0);
    }

    #[test]
    fn vote_truncated_pv_breaks_a_vote_tie() {
        let (a, b) = two_moves();
        // Equal scores and equal depths ⇒ equal vote totals for the two moves.
        // The tie is broken by the truncated-PV guard: the worker whose PV has
        // more than two moves wins.
        let workers = [vote(50, a, 2, 5), vote(50, b, 3, 5)];
        assert_eq!(select_best_worker(&workers), 1);
        // If the incumbent has the longer PV instead, the tie leaves it in place
        // (the challenger's `pv.len() > 2` factor is 0, so it cannot exceed).
        let workers = [vote(50, a, 3, 5), vote(50, b, 2, 5)];
        assert_eq!(select_best_worker(&workers), 0);
    }
}
