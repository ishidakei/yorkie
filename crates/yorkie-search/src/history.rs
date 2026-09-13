//! History-table skeletons the [`MovePicker`](crate::MovePicker) consults for
//! move ordering.
//!
//! Each table is either zero-filled (via `new`) or filled once with the
//! reference's `clear()` init constant (via
//! [`fill`](CapturePieceToHistory::fill)). The init constants matter for
//! parity: before any update lands, a uniformly filled table contributes a
//! constant to every score, so first-search move ordering is a pure function
//! of those constants and the static terms (MVV for captures, the capture bias
//! for evasions) — which is what makes the depth-1 node counts reproducible.
//!
//! The `pawnHistory` and `correctionHistory` tables do not live here: they are
//! shared between the worker threads of one NUMA node, so they sit in
//! [`SharedHistories`] with atomic entries. At `thread_count == 1` that type is
//! byte-identical to a per-worker copy, so single-thread node counts are
//! unaffected by the sharing.
//!
//! Entries are `i16` to match the reference's `StatsEntry` width, and each
//! table's heap array goes through the shared huge-page allocator, mirroring
//! the reference's `make_unique_large_page`. The allocator affects placement
//! only.
//!
//! Three of the update methods carry `#[inline(always)]`. Their caller is the
//! main search function, which is large enough that the ordinary `#[inline]`
//! hint leaves the call standing; forcing the inline removes it and leaves the
//! binary's text section smaller than it was. Every other accessor here is
//! inlined without an attribute.

use std::sync::atomic::{AtomicI16, Ordering};

use yorkie_state::{Color, Move, Piece, Square};
use yorkie_storage::{LargePageArray, LargePageBox, Zeroable};

/// Index of one plane of an `N`-plane continuation-style history table.
///
/// Every way of making one bounds the value below `N`, and the field is private,
/// so a table lookup addresses its plane straight away: the plane number a
/// search stack cell carries has otherwise lost the bound its `plane_index`
/// computation had.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PlaneIndex<const N: usize>(usize);

impl<const N: usize> PlaneIndex<N> {
    /// The `[NO_PIECE][0]` plane the reference seeds pre-root cells with.
    /// `NO_PIECE` has piece code `0` in this port's dense encoding, so it is
    /// plane `0`.
    pub const SENTINEL: Self = {
        assert!(N > 0, "a history table has at least one plane");
        Self(0)
    };

    /// Plane `i`.
    ///
    /// # Panics
    /// Panics unless `i < N`.
    pub fn new(i: usize) -> Self {
        assert!(i < N, "plane index {i} is past the table's {N} planes");
        Self(i)
    }

    /// The plane number, known to the compiler to be below `N`.
    fn get(self) -> usize {
        debug_assert!(self.0 < N);
        // SAFETY: the field is private and every constructor bounds it below
        // `N`, so this holds for every value of the type that exists.
        unsafe { core::hint::assert_unchecked(self.0 < N) };
        self.0
    }
}

/// Number of low-ply planes the reference keeps (`LOW_PLY_HISTORY_SIZE`):
/// `lowPlyHistory` is indexed by `ply` for `ply < 5`.
pub const LOW_PLY_HISTORY_SIZE: usize = 5;

/// Per-table gravity limits `D` (the `StatsEntry<T, D>` template parameter),
/// one constant per history table. Every update is clamped and pulled toward
/// zero relative to its table's `D` — see [`apply_gravity`].
pub const MAIN_HISTORY_D: i32 = 7183;
/// `lowPlyHistory` gravity limit (`ButterflyHistory` shares `D = 7183`).
pub const LOW_PLY_HISTORY_D: i32 = 7183;
/// `captureHistory` gravity limit.
pub const CAPTURE_HISTORY_D: i32 = 10692;
/// `continuationHistory` plane gravity limit (`PieceToHistory`).
pub const CONTINUATION_HISTORY_D: i32 = 30000;
/// `pawnHistory` gravity limit.
pub const PAWN_HISTORY_D: i32 = 8192;
/// Correction-history gravity limit `CORRECTION_HISTORY_LIMIT`.
pub const CORRECTION_HISTORY_D: i32 = 1024;
/// `ttMoveHistory` gravity limit.
pub const TT_MOVE_HISTORY_D: i32 = 8192;

/// Number of pawn-structure planes in [`SharedHistories`]'s pawn table
/// (`PAWN_HISTORY_BASE_SIZE`; a power of two, thread count 1 × base 8192). The
/// reference multiplies this by the thread count; this engine shares one table
/// per NUMA node instead, so the base size stands.
pub const PAWN_HISTORY_BASE_SIZE: usize = 8192;

/// Number of correction-history slots (`CORRHIST_BASE_SIZE`; `u16::MAX + 1`, a
/// power of two, thread count 1).
pub const CORRHIST_BASE_SIZE: usize = 65536;

/// The reference `StatsEntry::operator<<` gravity update, in integer arithmetic
/// throughout. The result satisfies `|entry| <= d`, and every `d` here is below
/// `i16::MAX`, so it always fits back into `i16`.
pub fn apply_gravity(entry: i16, bonus: i32, d: i32) -> i16 {
    debug_assert!(d > 0);
    let clamped = bonus.clamp(-d, d);
    let val = entry as i32;
    // `val * clamped.abs()` peaks at d² ≈ 9·10⁸ for the largest table, well
    // within i32.
    let updated = val + clamped - val * clamped.abs() / d;
    debug_assert!(updated.abs() <= d, "gravity result {updated} exceeds D={d}");
    updated as i16
}

/// Number of distinct colored piece codes: `(kind, promoted, color)`.
/// `kind` (0..8) + `promoted` (×8) + `color` (×16) ⇒ 0..32.
const PIECE_NB: usize = 32;
/// Number of board squares.
const SQ_NB: usize = Square::COUNT;
/// Distinct captured-piece type codes: the empty target at `0`, then one per
/// `(kind, promoted)` pair — `kind` (0..8) + `promoted` (×8), shifted up by the
/// empty slot, so `0..17`. A promoted king is not a piece any position holds,
/// but giving its code a slot is what keeps [`captured_code`]'s range inside the
/// table's without a case for it.
const CAPTURED_NB: usize = 17;

/// Dense index for a colored, possibly-promoted piece (the `pc` index). Unique
/// per `(kind, color, promoted)`, in `0..PIECE_NB`.
fn piece_code(p: Piece) -> usize {
    p.kind.index() + if p.promoted { 8 } else { 0 } + if p.color == Color::White { 16 } else { 0 }
}

/// Dense index for a captured piece's *type*, collapsing colour but keeping the
/// promoted distinction, in `1..CAPTURED_NB`. Index `0` is reserved for an
/// empty target, which the main search reads for a non-capturing check's
/// `captHist`.
fn captured_code(p: Piece) -> usize {
    1 + p.kind.index() + if p.promoted { 8 } else { 0 }
}

/// `captureHistory[pc][to][type_of(captured)]` — the capture-ordering bonus for
/// "moved piece `pc` captures a `captured`-type piece on square `to`".
pub struct CapturePieceToHistory {
    /// The dimensions are part of the type, so an entry's address is one
    /// computation from the block's base and each index carries its own bound.
    table: LargePageBox<[[[i16; CAPTURED_NB]; SQ_NB]; PIECE_NB]>,
}

impl Default for CapturePieceToHistory {
    fn default() -> Self {
        Self {
            table: LargePageBox::zeroed(),
        }
    }
}

impl CapturePieceToHistory {
    /// A fresh, zero-filled table.
    pub fn new() -> Self {
        Self::default()
    }

    /// The `(address, byte length)` of this table's large-page block — see
    /// [`WorkerHistories::backing_regions`](crate::WorkerHistories::backing_regions).
    /// A [`LargePageBox`] always owns a block, so this is never `None`.
    pub fn backing_region(&self) -> (usize, usize) {
        self.table.backing_region()
    }

    /// Overwrite every entry with `v` — the reference's
    /// `captureHistory.fill(v)` (init `-678`).
    pub fn fill(&mut self, v: i16) {
        self.table
            .iter_mut()
            .flatten()
            .flatten()
            .for_each(|e| *e = v);
    }

    /// The bonus for `moved` capturing `captured` on `to`. `0` for the
    /// zero-filled table.
    pub fn get(&self, moved: Piece, to: Square, captured: Piece) -> i32 {
        self.table[piece_code(moved)][to.index() as usize][captured_code(captured)] as i32
    }

    /// The entry for `moved` moving to an **empty** `to` — the `NO_PIECE`
    /// (index `0`) victim slot the main search reads for a non-capturing check
    /// (`captureHistory[movedPiece][to][type_of(NO_PIECE)]`).
    pub fn get_empty(&self, moved: Piece, to: Square) -> i32 {
        self.table[piece_code(moved)][to.index() as usize][0] as i32
    }

    /// Gravity-update the entry for `moved` capturing `captured` on `to`
    /// (`D = 10692`).
    #[inline(always)]
    pub fn update(&mut self, moved: Piece, to: Square, captured: Piece, bonus: i32) {
        let cell = &mut self.table[piece_code(moved)][to.index() as usize][captured_code(captured)];
        *cell = apply_gravity(*cell, bonus, CAPTURE_HISTORY_D);
    }
}

/// Plane count of [`ContinuationHistory`]: `[in_check][capture][pc][to]`.
pub const CONT_PLANES: usize = 2 * 2 * PIECE_NB * SQ_NB;
/// Plane count of [`ContinuationCorrectionHistory`]: `[pc][to]`.
pub const CONT_CORR_PLANES: usize = PIECE_NB * SQ_NB;

/// A [`ContinuationHistory`] plane index.
pub type ContPlane = PlaneIndex<CONT_PLANES>;
/// A [`ContinuationCorrectionHistory`] plane index.
pub type CorrPlane = PlaneIndex<CONT_CORR_PLANES>;

/// The low 16 bits of the packed move — the butterfly tables' move dimension.
fn move16(m: Move) -> usize {
    (m.to_bits() & 0xFFFF) as usize
}

/// `mainHistory[us][move]` — the butterfly (from-to) quiet-move history for the
/// side to move. Indexed by the low 16 bits of the packed move, which encode
/// exactly the `(from-or-dropped-type, to)` pair the reference's `move.raw()`
/// uses.
pub struct ButterflyHistory {
    /// Layout `[us][move16]`, both dimensions compile-time.
    table: LargePageBox<[[i16; 1 << 16]; Color::COUNT]>,
}

impl Default for ButterflyHistory {
    fn default() -> Self {
        Self {
            table: LargePageBox::zeroed(),
        }
    }
}

impl ButterflyHistory {
    /// A fresh, zero-filled table.
    pub fn new() -> Self {
        Self::default()
    }

    /// The `(address, byte length)` of this table's large-page block — see
    /// [`WorkerHistories::backing_regions`](crate::WorkerHistories::backing_regions).
    /// A [`LargePageBox`] always owns a block, so this is never `None`.
    pub fn backing_region(&self) -> (usize, usize) {
        self.table.backing_region()
    }

    /// Overwrite every entry with `v` — the reference's `mainHistory.fill(v)`
    /// (init `0`).
    pub fn fill(&mut self, v: i16) {
        self.table.iter_mut().flatten().for_each(|e| *e = v);
    }

    /// The quiet-move history for `m` played by `us`. `0` for the zero-filled
    /// table.
    pub fn get(&self, us: Color, m: Move) -> i32 {
        self.table[us.index()][move16(m)] as i32
    }

    /// Gravity-update `mainHistory[us][move.raw16]` (`D = 7183`).
    #[inline(always)]
    pub fn update(&mut self, us: Color, m: Move, bonus: i32) {
        let cell = &mut self.table[us.index()][move16(m)];
        *cell = apply_gravity(*cell, bonus, MAIN_HISTORY_D);
    }
}

/// `PieceToHistory[pc][to]` — one continuation-history plane. The qsearch
/// evasion quiet score reads only `continuationHistory[0]`, i.e. one such
/// plane.
///
/// The entries are inline, so a plane held inside a larger table is part of
/// that table's block and an entry is one address computation away from it.
pub struct PieceToHistory {
    table: [[i16; SQ_NB]; PIECE_NB],
}

// SAFETY: an all-zero `[[i16; SQ_NB]; PIECE_NB]` is the zero-filled plane, a
// valid value, and the plane needs no drop glue.
unsafe impl Zeroable for PieceToHistory {}

impl Default for PieceToHistory {
    fn default() -> Self {
        Self {
            table: [[0i16; SQ_NB]; PIECE_NB],
        }
    }
}

impl PieceToHistory {
    /// A fresh, zero-filled plane.
    pub fn new() -> Self {
        Self::default()
    }

    /// Overwrite every entry with `v` — the reference fills each
    /// `continuationHistory` plane with `-523`.
    pub fn fill(&mut self, v: i16) {
        self.table.iter_mut().flatten().for_each(|e| *e = v);
    }

    /// The continuation bonus for piece `pc` moving to `to`. `0` for the
    /// zero-filled plane.
    pub fn get(&self, pc: Piece, to: Square) -> i32 {
        self.table[piece_code(pc)][to.index() as usize] as i32
    }

    /// Gravity-update the `[pc][to]` continuation entry (`D = 30000`).
    pub fn update(&mut self, pc: Piece, to: Square, bonus: i32) {
        let cell = &mut self.table[piece_code(pc)][to.index() as usize];
        *cell = apply_gravity(*cell, bonus, CONTINUATION_HISTORY_D);
    }
}

/// `lowPlyHistory[ply][move]` — the near-root quiet bonus. Re-filled per `go`
/// through [`fill`](LowPlyHistory::fill) at the root.
pub struct LowPlyHistory {
    /// Layout `[ply][move16]`, both dimensions compile-time.
    table: LargePageBox<[[i16; 1 << 16]; LOW_PLY_HISTORY_SIZE]>,
}

impl Default for LowPlyHistory {
    fn default() -> Self {
        Self {
            table: LargePageBox::zeroed(),
        }
    }
}

impl LowPlyHistory {
    /// A fresh, zero-filled table.
    pub fn new() -> Self {
        Self::default()
    }

    /// The `(address, byte length)` of this table's large-page block — see
    /// [`WorkerHistories::backing_regions`](crate::WorkerHistories::backing_regions).
    /// A [`LargePageBox`] always owns a block, so this is never `None`.
    pub fn backing_region(&self) -> (usize, usize) {
        self.table.backing_region()
    }

    /// Overwrite every entry with `v` — the reference's `lowPlyHistory.fill(v)`
    /// (init `98`).
    pub fn fill(&mut self, v: i16) {
        self.table.iter_mut().flatten().for_each(|e| *e = v);
    }

    /// The near-root bonus for `m` at `ply`. The caller guarantees
    /// `ply < LOW_PLY_HISTORY_SIZE`. `0` for the zero-filled table.
    pub fn get(&self, ply: usize, m: Move) -> i32 {
        self.table[ply][move16(m)] as i32
    }

    /// Gravity-update `lowPlyHistory[ply][move.raw16]` (`D = 7183`). The caller
    /// guarantees `ply < LOW_PLY_HISTORY_SIZE`.
    pub fn update(&mut self, ply: usize, m: Move, bonus: i32) {
        let cell = &mut self.table[ply][move16(m)];
        *cell = apply_gravity(*cell, bonus, LOW_PLY_HISTORY_D);
    }
}

/// The four correction channels of a `CorrectionBundle`. Each
/// unified-correction slot holds one `i16` per channel, per side to move.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CorrChannel {
    /// By color and pawn structure (`pawn_key`-keyed slot).
    Pawn = 0,
    /// By color and minor-piece positions (`minor_piece_key`-keyed slot).
    Minor = 1,
    /// By White non-pawn positions (`non_pawn_key(WHITE)`-keyed slot).
    NonPawnWhite = 2,
    /// By Black non-pawn positions (`non_pawn_key(BLACK)`-keyed slot).
    NonPawnBlack = 3,
}

impl CorrChannel {
    const COUNT: usize = 4;
    fn index(self) -> usize {
        self as usize
    }
}

/// Reference `SharedHistories::clear()` init values: correction entries `0`,
/// pawn entries `-1238`.
const CORRECTION_INIT: i16 = 0;
const PAWN_INIT: i16 = -1238;

/// The atomic gravity update: identical arithmetic to [`apply_gravity`], but a
/// plain relaxed load-modify-store rather than a CAS loop. The reference does
/// not guard the read-modify-write against a concurrent update either, so two
/// threads updating the same entry can lose one update — accepted by design,
/// since the histories are a heuristic.
fn apply_gravity_atomic(cell: &AtomicI16, bonus: i32, d: i32) {
    debug_assert!(d > 0);
    let clamped = bonus.clamp(-d, d);
    let val = cell.load(Ordering::Relaxed) as i32;
    let updated = val + clamped - val * clamped.abs() / d;
    debug_assert!(updated.abs() <= d, "gravity result {updated} exceeds D={d}");
    cell.store(updated as i16, Ordering::Relaxed);
}

/// Number of `i16` channels a correction slot holds: one [`CorrChannel`] per
/// side to move (`MultiArray<CorrectionBundle, COLOR_NB>`).
const CORR_SLOT_LEN: usize = Color::COUNT * CorrChannel::COUNT;
/// Number of `i16` entries a pawn-history slot holds (`[pc][to]`).
const PAWN_SLOT_LEN: usize = PIECE_NB * SQ_NB;

/// The reference `SharedHistories`: the unified correction history and the pawn
/// history, shared between the worker threads of one NUMA node and sized by
/// that node's thread count, so a larger node gets a proportionally larger,
/// less-contended table.
///
/// `thread_count` must be a non-zero power of two (asserted) so slot selection
/// is a single mask over the full 64-bit key.
pub struct SharedHistories {
    /// The node's thread count (a power of two); the slot multiplier of both
    /// tables.
    thread_count: usize,
    /// Correction table: `thread_count * CORRHIST_BASE_SIZE` slots, each
    /// [`CORR_SLOT_LEN`] atomic `i16`. Layout `[slot][color][channel]`, flat.
    correction: LargePageArray<AtomicI16>,
    /// `correctionHistory.get_size() - 1` — the correction slot mask.
    corr_mask: usize,
    /// Pawn table: `thread_count * PAWN_HISTORY_BASE_SIZE` slots, each
    /// [`PAWN_SLOT_LEN`] atomic `i16`. Layout `[plane][pc][to]`, flat.
    pawn: LargePageArray<AtomicI16>,
    /// `pawnHistory.get_size() - 1` — the pawn slot mask.
    pawn_mask: usize,
}

impl SharedHistories {
    /// Build the shared tables for a node of `thread_count` workers, which must
    /// be a non-zero power of two.
    ///
    /// The allocation **and** the initial fill run here, so calling this inside
    /// a node-bound thread lets first-touch place every page on that node.
    pub fn new(thread_count: usize) -> Self {
        assert!(
            thread_count.is_power_of_two() && thread_count != 0,
            "SharedHistories thread_count must be a non-zero power of two, got {thread_count}"
        );
        let corr_slots = thread_count * CORRHIST_BASE_SIZE;
        let pawn_slots = thread_count * PAWN_HISTORY_BASE_SIZE;
        let out = Self {
            thread_count,
            correction: LargePageArray::zeroed(corr_slots * CORR_SLOT_LEN),
            corr_mask: corr_slots - 1,
            pawn: LargePageArray::zeroed(pawn_slots * PAWN_SLOT_LEN),
            pawn_mask: pawn_slots - 1,
        };
        // The correction fill of `0` is a no-op on the zeroed allocation, but is
        // written explicitly so every page is first-touched on the (bound) node.
        out.fill_correction(CORRECTION_INIT);
        out.fill_pawn(PAWN_INIT);
        out
    }

    /// The node's thread count (the slot multiplier).
    pub fn thread_count(&self) -> usize {
        self.thread_count
    }

    /// Number of correction slots (`thread_count * CORRHIST_BASE_SIZE`).
    pub fn correction_slots(&self) -> usize {
        self.corr_mask + 1
    }

    /// Number of pawn slots (`thread_count * PAWN_HISTORY_BASE_SIZE`).
    pub fn pawn_slots(&self) -> usize {
        self.pawn_mask + 1
    }

    /// Overwrite every correction entry with `v` (atomic RELAXED stores).
    pub fn fill_correction(&self, v: i16) {
        for e in self.correction.iter() {
            e.store(v, Ordering::Relaxed);
        }
    }

    /// Overwrite every pawn entry with `v` (atomic RELAXED stores).
    pub fn fill_pawn(&self, v: i16) {
        for e in self.pawn.iter() {
            e.store(v, Ordering::Relaxed);
        }
    }

    /// Flat index of the `channel` entry for side-to-move `color` in the slot
    /// keyed by `key`: `slot = key & corr_mask`, then `[color][channel]`.
    fn corr_index(&self, key: u64, color: Color, channel: CorrChannel) -> usize {
        let slot = (key as usize) & self.corr_mask;
        (slot * Color::COUNT + color.index()) * CorrChannel::COUNT + channel.index()
    }

    /// The `channel` entry for side-to-move `color` in the slot keyed by `key`.
    ///
    /// Both tables are sized `slots * SLOT_LEN` and masked with `slots - 1` in
    /// [`Self::new`], which is the only place either is built, so a masked index
    /// is always inside the block. The length is a run-time value, so the bound
    /// has to be stated rather than derived from the type.
    fn corr_cell(&self, key: u64, color: Color, channel: CorrChannel) -> &AtomicI16 {
        let i = self.corr_index(key, color, channel);
        debug_assert!(i < self.correction.len());
        // SAFETY: see the doc comment — `key & corr_mask < corr_mask + 1` and
        // the table holds `(corr_mask + 1) * CORR_SLOT_LEN` entries.
        unsafe { self.correction.get_unchecked(i) }
    }

    /// The `channel` correction value for side-to-move `color` in the slot keyed
    /// by `key`. `0` on a fresh table.
    pub fn correction_get(&self, key: u64, color: Color, channel: CorrChannel) -> i32 {
        self.corr_cell(key, color, channel).load(Ordering::Relaxed) as i32
    }

    /// Gravity-update the `channel` entry for side-to-move `color` in the slot
    /// keyed by `key` (`D = 1024`).
    pub fn correction_update(&self, key: u64, color: Color, channel: CorrChannel, bonus: i32) {
        apply_gravity_atomic(
            self.corr_cell(key, color, channel),
            bonus,
            CORRECTION_HISTORY_D,
        );
    }

    /// Flat index of the `[pc][to]` entry in the plane keyed by `pawn_key`:
    /// `plane = pawn_key & pawn_mask`, then `[pc][to]`.
    fn pawn_index(&self, pawn_key: u64, pc: Piece, to: Square) -> usize {
        let plane = (pawn_key as usize) & self.pawn_mask;
        (plane * PIECE_NB + piece_code(pc)) * SQ_NB + to.index() as usize
    }

    /// The `[pc][to]` entry in the plane keyed by `pawn_key`. See
    /// [`Self::corr_cell`] for why the masked index needs no check.
    fn pawn_cell(&self, pawn_key: u64, pc: Piece, to: Square) -> &AtomicI16 {
        let i = self.pawn_index(pawn_key, pc, to);
        debug_assert!(i < self.pawn.len());
        // SAFETY: `pawn_key & pawn_mask < pawn_mask + 1` and the table holds
        // `(pawn_mask + 1) * PAWN_SLOT_LEN` entries.
        unsafe { self.pawn.get_unchecked(i) }
    }

    /// The pawn-structure bonus for piece `pc` moving to `to` in the plane keyed
    /// by `pawn_key`. `-1238` on a fresh table.
    pub fn pawn_get(&self, pawn_key: u64, pc: Piece, to: Square) -> i32 {
        self.pawn_cell(pawn_key, pc, to).load(Ordering::Relaxed) as i32
    }

    /// Gravity-update the `[pc][to]` entry in the plane keyed by `pawn_key`
    /// (`D = 8192`).
    #[inline(always)]
    pub fn pawn_update(&self, pawn_key: u64, pc: Piece, to: Square, bonus: i32) {
        apply_gravity_atomic(self.pawn_cell(pawn_key, pc, to), bonus, PAWN_HISTORY_D);
    }
}

/// `CorrectionHistory<Continuation>`: a `[pc][to]` table whose every cell is
/// itself a `[pc][to]` `i16` table.
///
/// The outer `[pc][to]` selects a *plane* — the one a search stack cell's
/// `continuationCorrectionHistory` points at — and the inner `[pc][to]` indexes
/// within it. [`CorrPlane::SENTINEL`] is what pre-root cells are seeded with.
pub struct ContinuationCorrectionHistory {
    /// Layout: `[plane = outer_pc*SQ_NB + outer_to][inner_pc][inner_to]`. Every
    /// dimension is fixed so each access carries a compile-time length.
    table: LargePageBox<[[[i16; SQ_NB]; PIECE_NB]; CONT_CORR_PLANES]>,
}

impl Default for ContinuationCorrectionHistory {
    fn default() -> Self {
        // Huge-page-backed, zero-initialised; the compile-time dimensions are
        // preserved by [`LargePageBox`].
        Self {
            table: LargePageBox::zeroed(),
        }
    }
}

impl ContinuationCorrectionHistory {
    /// A fresh, zero-filled table. Use [`Self::fill`] to apply the reference's
    /// init value of `6`.
    pub fn new() -> Self {
        Self::default()
    }

    /// The `(address, byte length)` of this table's large-page block — see
    /// [`WorkerHistories::backing_regions`](crate::WorkerHistories::backing_regions).
    /// A [`LargePageBox`] always owns a block, so this is never `None`.
    pub fn backing_region(&self) -> (usize, usize) {
        self.table.backing_region()
    }

    /// Overwrite every entry with `v` — the reference fills each continuation
    /// correction plane with `6`.
    pub fn fill(&mut self, v: i16) {
        self.table
            .iter_mut()
            .flatten()
            .flatten()
            .for_each(|e| *e = v);
    }

    /// The plane index selected by the outer `[pc][to]` (the plane a search
    /// stack cell's `continuationCorrectionHistory` points at).
    pub fn plane_index(pc: Piece, to: Square) -> CorrPlane {
        CorrPlane::new(piece_code(pc) * SQ_NB + to.index() as usize)
    }

    /// The inner `[pc][to]` value in `plane`. The fill value on a filled table.
    pub fn get_at(&self, plane: CorrPlane, pc: Piece, to: Square) -> i32 {
        self.table[plane.get()][piece_code(pc)][to.index() as usize] as i32
    }

    /// Gravity-update the inner `[pc][to]` entry in `plane` (`D = 1024`).
    pub fn update_at(&mut self, plane: CorrPlane, pc: Piece, to: Square, bonus: i32) {
        let cell = &mut self.table[plane.get()][piece_code(pc)][to.index() as usize];
        *cell = apply_gravity(*cell, bonus, CORRECTION_HISTORY_D);
    }
}

/// The worker's `continuationHistory`: planes keyed by
/// `(in_check, capture, moved_piece, to)`, each plane a `[pc][to]` `i16` table.
///
/// A search stack cell's `continuationHistory` points at one such plane, and
/// the update primitives write the current move's `[pc][to]` within it.
pub struct ContinuationHistory {
    /// Layout: `[plane][inner_pc][inner_to]`, `plane` from [`Self::plane_index`].
    /// The planes are inline, so their length is part of the type and an entry's
    /// address is one computation from the block's base.
    table: LargePageBox<[PieceToHistory; CONT_PLANES]>,
}

impl Default for ContinuationHistory {
    fn default() -> Self {
        Self {
            table: LargePageBox::zeroed(),
        }
    }
}

impl ContinuationHistory {
    /// A fresh, zero-filled table. Use [`Self::fill`] for the reference init.
    pub fn new() -> Self {
        Self::default()
    }

    /// The `(address, byte length)` of this table's large-page block — see
    /// [`WorkerHistories::backing_regions`](crate::WorkerHistories::backing_regions).
    /// This is the big one: ~54 MiB, more than three quarters of a worker's
    /// private footprint. A [`LargePageBox`] always owns a block, so this is
    /// never `None`.
    pub fn backing_region(&self) -> (usize, usize) {
        self.table.backing_region()
    }

    /// Overwrite every entry with `v` — the reference fills each continuation
    /// plane with `-523`.
    pub fn fill(&mut self, v: i16) {
        self.table.iter_mut().for_each(|plane| plane.fill(v));
    }

    /// The plane index selected by `(in_check, capture, pc, to)` — the plane a
    /// search stack cell's `continuationHistory` points at after a move.
    pub fn plane_index(in_check: bool, capture: bool, pc: Piece, to: Square) -> ContPlane {
        let ic = in_check as usize;
        let cap = capture as usize;
        ContPlane::new(((ic * 2 + cap) * PIECE_NB + piece_code(pc)) * SQ_NB + to.index() as usize)
    }

    /// The inner `[pc][to]` value in `plane`.
    pub fn get_at(&self, plane: ContPlane, pc: Piece, to: Square) -> i32 {
        self.table[plane.get()].get(pc, to)
    }

    /// Gravity-update the inner `[pc][to]` entry in `plane` (`D = 30000`).
    pub fn update_at(&mut self, plane: ContPlane, pc: Piece, to: Square, bonus: i32) {
        self.table[plane.get()].update(pc, to, bonus);
    }

    /// Borrow `plane` whole, so a search that keeps its continuation table in
    /// this multi-plane form can hand the picker the six planes its `contHist`
    /// array names.
    pub fn plane(&self, plane: ContPlane) -> &PieceToHistory {
        &self.table[plane.get()]
    }
}

/// `TTMoveHistory`: a single gravity entry with `D = 8192`, init `0`.
#[derive(Default)]
pub struct TtMoveHistory {
    entry: i16,
}

impl TtMoveHistory {
    /// A fresh, zero entry.
    pub fn new() -> Self {
        Self::default()
    }

    /// The current value.
    pub fn get(&self) -> i32 {
        self.entry as i32
    }

    /// Gravity-update the entry (`D = 8192`).
    pub fn update(&mut self, bonus: i32) {
        self.entry = apply_gravity(self.entry, bonus, TT_MOVE_HISTORY_D);
    }
}

#[cfg(test)]
mod plane_tests {
    use super::*;
    use yorkie_state::PieceKind;

    fn bp() -> Piece {
        Piece::new(PieceKind::Pawn, Color::Black)
    }
    fn to() -> Square {
        Square::new(4, 3).unwrap()
    }

    /// A borrowed plane is part of the table's own block, so a continuation
    /// lookup is an offset into it and never a pointer load of its own.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn every_plane_lies_inside_the_tables_block() {
        let hist = ContinuationHistory::new();
        let (base, len) = hist.backing_region();
        for n in [0, 1, CONT_PLANES - 1] {
            let plane = ContPlane::new(n);
            let at = hist.plane(plane) as *const PieceToHistory as usize;
            assert!(
                at >= base && at + size_of::<PieceToHistory>() <= base + len,
                "plane {n} at {at:#x} is outside the block [{base:#x}, +{len})",
            );
        }
    }

    /// A plane index past the table's last plane is rejected at construction.
    #[test]
    #[should_panic(expected = "past the table's")]
    fn a_plane_index_past_the_last_plane_is_rejected() {
        let _ = ContPlane::new(CONT_PLANES);
    }

    /// A borrowed plane reads what the table reads at the same cell.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn a_borrowed_plane_reads_what_the_table_reads() {
        let mut hist = ContinuationHistory::new();
        hist.fill(-523);
        let seven = ContPlane::new(7);
        let eight = ContPlane::new(8);
        hist.update_at(seven, bp(), to(), 4_000);
        assert_eq!(
            hist.plane(seven).get(bp(), to()),
            hist.get_at(seven, bp(), to())
        );
        assert_ne!(hist.plane(seven).get(bp(), to()), -523);
        assert_eq!(hist.plane(eight).get(bp(), to()), -523);
    }
}

#[cfg(test)]
mod shared_tests {
    use super::*;
    use std::sync::Arc;
    use yorkie_state::PieceKind;

    fn bp() -> Piece {
        Piece::new(PieceKind::Pawn, Color::Black)
    }
    fn to() -> Square {
        Square::new(4, 3).unwrap()
    }
    const CHANNELS: [CorrChannel; 4] = [
        CorrChannel::Pawn,
        CorrChannel::Minor,
        CorrChannel::NonPawnWhite,
        CorrChannel::NonPawnBlack,
    ];

    /// Table sizing scales the slot count by `thread_count` for both tables.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn sizing_scales_with_thread_count() {
        for &tc in &[1usize, 2, 4, 8] {
            let sh = SharedHistories::new(tc);
            assert_eq!(sh.thread_count(), tc);
            assert_eq!(sh.correction_slots(), tc * CORRHIST_BASE_SIZE);
            assert_eq!(sh.pawn_slots(), tc * PAWN_HISTORY_BASE_SIZE);
        }
    }

    /// The `thread_count` power-of-two assert.
    #[test]
    #[should_panic(expected = "power of two")]
    fn rejects_non_power_of_two() {
        let _ = SharedHistories::new(3);
    }

    #[test]
    #[should_panic(expected = "power of two")]
    fn rejects_zero_thread_count() {
        let _ = SharedHistories::new(0);
    }

    /// Init values: pawn `-1238`, correction `0`, across distinct slots.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn init_values() {
        let sh = SharedHistories::new(1);
        assert_eq!(sh.pawn_get(0, bp(), to()), -1238);
        assert_eq!(sh.pawn_get(0xFFFF, bp(), to()), -1238);
        for &color in &[Color::Black, Color::White] {
            for &ch in &CHANNELS {
                assert_eq!(sh.correction_get(0, color, ch), 0);
                assert_eq!(sh.correction_get(0xFFFF, color, ch), 0);
            }
        }
    }

    /// Mask widening: two pawn keys equal mod `8192` but different mod `16384`
    /// alias at `thread_count == 1` (mask `8191`) and separate at `thread_count
    /// == 2` (mask `16383`).
    #[cfg_attr(miri, ignore)]
    #[test]
    fn pawn_mask_widens_with_thread_count() {
        let key_a = 0u64;
        let key_b = PAWN_HISTORY_BASE_SIZE as u64; // 8192: equal mod 8192, differ mod 16384

        let one = SharedHistories::new(1);
        let b_before = one.pawn_get(key_b, bp(), to());
        one.pawn_update(key_a, bp(), to(), 1_000_000);
        assert_ne!(
            one.pawn_get(key_b, bp(), to()),
            b_before,
            "thread_count 1: keys 0 and 8192 must share one plane"
        );

        let two = SharedHistories::new(2);
        let b_before = two.pawn_get(key_b, bp(), to());
        two.pawn_update(key_a, bp(), to(), 1_000_000);
        assert_ne!(
            two.pawn_get(key_a, bp(), to()),
            b_before,
            "thread_count 2: the updated plane still moved"
        );
        assert_eq!(
            two.pawn_get(key_b, bp(), to()),
            b_before,
            "thread_count 2: keys 0 and 8192 must be in distinct planes"
        );
    }

    /// The same widening for the correction table (`65536` slot base).
    #[cfg_attr(miri, ignore)]
    #[test]
    fn correction_mask_widens_with_thread_count() {
        let key_a = 0u64;
        let key_b = CORRHIST_BASE_SIZE as u64; // 65536

        let one = SharedHistories::new(1);
        one.correction_update(key_a, Color::Black, CorrChannel::Pawn, 1_000_000);
        assert_ne!(
            one.correction_get(key_b, Color::Black, CorrChannel::Pawn),
            0,
            "thread_count 1: correction keys 0 and 65536 share a slot"
        );

        let two = SharedHistories::new(2);
        two.correction_update(key_a, Color::Black, CorrChannel::Pawn, 1_000_000);
        assert_ne!(
            two.correction_get(key_a, Color::Black, CorrChannel::Pawn),
            0
        );
        assert_eq!(
            two.correction_get(key_b, Color::Black, CorrChannel::Pawn),
            0,
            "thread_count 2: correction keys 0 and 65536 are in distinct slots"
        );
    }

    /// Atomicity smoke: two threads hammering the same entry with `<<`
    /// terminates and leaves a value within `[-D, D]` (the reference accepts
    /// lost updates but the gravity bound always holds).
    #[cfg_attr(miri, ignore)]
    #[test]
    fn concurrent_updates_terminate_within_limit() {
        let sh = Arc::new(SharedHistories::new(2));
        let key = 12_345u64;
        let handles: Vec<_> = (0..2)
            .map(|t| {
                let sh = Arc::clone(&sh);
                std::thread::spawn(move || {
                    for i in 0..20_000i32 {
                        let bonus = if (i + t) % 2 == 0 { 5000 } else { -5000 };
                        sh.pawn_update(key, bp(), to(), bonus);
                        sh.correction_update(key, Color::White, CorrChannel::Minor, bonus);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("update thread must not panic");
        }
        let pv = sh.pawn_get(key, bp(), to());
        assert!(pv.abs() <= PAWN_HISTORY_D, "pawn value {pv} exceeds D");
        let cv = sh.correction_get(key, Color::White, CorrChannel::Minor);
        assert!(
            cv.abs() <= CORRECTION_HISTORY_D,
            "correction value {cv} exceeds D"
        );
    }
}
