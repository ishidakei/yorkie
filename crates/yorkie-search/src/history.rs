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
//! Entries are `i16` to match the reference's `StatsEntry` width. A table is
//! either in large-page static storage laid out when the binary is built, or
//! on the heap through the shared huge-page allocator, mirroring the
//! reference's `make_unique_large_page`; either affects placement only.
//!
//! Three of the update methods carry `#[inline(always)]`. Their caller is the
//! main search function, which is large enough that the ordinary `#[inline]`
//! hint leaves the call standing; forcing the inline removes it and leaves the
//! binary's text section smaller than it was. Every other accessor here is
//! inlined without an attribute.

use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicI16, Ordering};

use yorkie_state::{Color, Move, Piece, Square};
use yorkie_storage::{LARGE_PAGE_ALIGN, LargePageAlloc, Zeroable};

use crate::worker_block::WorkerAlloc;

/// The `(address, byte length)` of the large-page block behind a heap table:
/// the table's bytes rounded up to whole large pages, which is what the
/// allocator handed out. A table inside a worker's block is placed with the
/// block instead (`BlockClaim::backing_region`).
#[expect(
    clippy::borrowed_box,
    reason = "only a box from the large-page policy names a rounded block"
)]
fn heap_region<T>(table: &Box<T, WorkerAlloc>) -> (usize, usize) {
    (
        Box::as_ptr(table).addr(),
        size_of::<T>().next_multiple_of(LARGE_PAGE_ALIGN),
    )
}

/// Ask the CPU to bring the cache line holding `entry` into every cache level.
/// A no-op off x86-64.
///
/// The address is taken from a reference, so it is always that of a live,
/// in-bounds entry.
#[inline(always)]
fn prefetch_t0<T>(entry: &T) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: `_mm_prefetch` is a hardware hint with no architectural effect:
    // it neither reads nor writes the memory observably and cannot fault. The
    // pointer comes from a reference, so it is valid besides.
    unsafe {
        use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
        _mm_prefetch::<_MM_HINT_T0>(std::ptr::from_ref(entry).cast::<i8>());
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = entry;
}

/// Index of one plane of an `N`-plane continuation-style history table.
///
/// Every way of making one bounds the value below `N`, and the field is private,
/// so a table lookup addresses its plane straight away: the plane number a
/// search stack cell carries has otherwise lost the bound its `plane_index`
/// computation had.
///
/// The plane number is held in a `u16` rather than a `usize`. Six of these ride
/// in a [`MovePicker`](crate::MovePicker) and two in every search stack cell, so
/// the six bytes a plane number does not need are six bytes of those two
/// per-node structures. A table with more planes than a `u16` can address fails
/// the build rather than truncating a plane number.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PlaneIndex<const N: usize>(u16);

impl<const N: usize> PlaneIndex<N> {
    /// A table has at least one plane and no more than a `u16` can address, so a
    /// table that outgrew the field is a build failure rather than a silently
    /// truncated plane number. Every constructor evaluates it.
    const PLANES_FIT: () = assert!(
        N > 0 && N <= u16::MAX as usize + 1,
        "a history table has at least one plane and at most u16::MAX + 1",
    );

    /// The `[NO_PIECE][0]` plane the reference seeds pre-root cells with.
    /// `NO_PIECE` has piece code `0` in this port's dense encoding, so it is
    /// plane `0`.
    pub const SENTINEL: Self = {
        let () = Self::PLANES_FIT;
        Self(0)
    };

    /// Plane `i`.
    ///
    /// # Panics
    /// Panics unless `i < N`.
    pub fn new(i: usize) -> Self {
        let () = Self::PLANES_FIT;
        assert!(i < N, "plane index {i} is past the table's {N} planes");
        Self(i as u16)
    }

    /// The plane number, known to the compiler to be below `N`.
    fn get(self) -> usize {
        let i = self.0 as usize;
        debug_assert!(i < N);
        // SAFETY: the field is private and every constructor bounds it below
        // `N`, so this holds for every value of the type that exists.
        unsafe { core::hint::assert_unchecked(i < N) };
        i
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

/// The `[pc][to]` position of one entry within a `[pc][to]` plane, as one flat
/// number.
///
/// A quiet move reads the same `[pc][to]` entry from six planes (five
/// continuation planes and the pawn plane), so the index is computed once per
/// move and each plane is then one load away. Like [`PlaneIndex`], the field is
/// private and bounded by its constructor, so a plane lookup needs no check.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PieceSquare(u16);

impl PieceSquare {
    /// The entry of piece `pc` moving to `to`.
    pub fn new(pc: Piece, to: Square) -> Self {
        let i = piece_code(pc) * SQ_NB + to.index() as usize;
        debug_assert!(i < PIECE_NB * SQ_NB);
        Self(i as u16)
    }

    /// The flat index, known to the compiler to be inside a plane.
    fn get(self) -> usize {
        let i = self.0 as usize;
        debug_assert!(i < PIECE_NB * SQ_NB);
        // SAFETY: the field is private and `new` builds it from a piece code
        // below `PIECE_NB` and a square index below `SQ_NB`.
        unsafe { core::hint::assert_unchecked(i < PIECE_NB * SQ_NB) };
        i
    }
}

/// The storage of a [`CapturePieceToHistory`].
pub(crate) type CaptureTable = [[[i16; CAPTURED_NB]; SQ_NB]; PIECE_NB];

/// `captureHistory[pc][to][type_of(captured)]` — the capture-ordering bonus for
/// "moved piece `pc` captures a `captured`-type piece on square `to`".
pub struct CapturePieceToHistory {
    /// The dimensions are part of the type, so an entry's address is one
    /// computation from the block's base and each index carries its own bound.
    table: Box<CaptureTable, WorkerAlloc>,
}

impl Default for CapturePieceToHistory {
    fn default() -> Self {
        Self {
            table: WorkerAlloc::zeroed(),
        }
    }
}

#[cfg(test)]
impl CapturePieceToHistory {
    /// The table's entries as bytes, for a test comparing two tables whole.
    pub(crate) fn as_bytes(&self) -> &[u8] {
        // SAFETY: the table is plain `i16` data with no padding, so every one
        // of its bytes is initialised.
        unsafe {
            std::slice::from_raw_parts(Box::as_ptr(&self.table).cast(), size_of_val(&*self.table))
        }
    }
}

impl CapturePieceToHistory {
    /// A fresh, zero-filled table.
    pub fn new() -> Self {
        Self::default()
    }

    /// The table over `table`, as it stands — a worker's block hands its
    /// storage over unfilled.
    pub(crate) fn over(table: Box<CaptureTable, WorkerAlloc>) -> Self {
        Self { table }
    }

    /// The `(address, byte length)` of this table's large-page block — see
    /// [`WorkerHistories::backing_regions`](crate::WorkerHistories::backing_regions).
    /// The table is not zero-sized, so it always owns a block.
    pub fn backing_region(&self) -> (usize, usize) {
        heap_region(&self.table)
    }

    /// Overwrite every entry with `v` — the reference's
    /// `captureHistory.fill(v)` (init `-678` by default).
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

// A plane index is the bare plane number and nothing else: six of them sit in a
// per-node `MovePicker` and two in every search stack cell, so a field added
// here would be paid for at every node.
const _: () = assert!(size_of::<ContPlane>() == 2);
const _: () = assert!(size_of::<CorrPlane>() == 2);

/// The low 16 bits of the packed move — the butterfly tables' move dimension.
fn move16(m: Move) -> usize {
    (m.to_bits() & 0xFFFF) as usize
}

/// The storage of a [`ButterflyHistory`].
pub(crate) type ButterflyTable = [[i16; 1 << 16]; Color::COUNT];

/// `mainHistory[us][move]` — the butterfly (from-to) quiet-move history for the
/// side to move. Indexed by the low 16 bits of the packed move, which encode
/// exactly the `(from-or-dropped-type, to)` pair the reference's `move.raw()`
/// uses.
pub struct ButterflyHistory {
    /// Layout `[us][move16]`, both dimensions compile-time.
    table: Box<ButterflyTable, WorkerAlloc>,
}

impl Default for ButterflyHistory {
    fn default() -> Self {
        Self {
            table: WorkerAlloc::zeroed(),
        }
    }
}

#[cfg(test)]
impl ButterflyHistory {
    /// The table's entries as bytes, for a test comparing two tables whole.
    pub(crate) fn as_bytes(&self) -> &[u8] {
        // SAFETY: the table is plain `i16` data with no padding, so every one
        // of its bytes is initialised.
        unsafe {
            std::slice::from_raw_parts(Box::as_ptr(&self.table).cast(), size_of_val(&*self.table))
        }
    }
}

impl ButterflyHistory {
    /// A fresh, zero-filled table.
    pub fn new() -> Self {
        Self::default()
    }

    /// The table over `table`, as it stands — a worker's block hands its
    /// storage over unfilled.
    pub(crate) fn over(table: Box<ButterflyTable, WorkerAlloc>) -> Self {
        Self { table }
    }

    /// The `(address, byte length)` of this table's large-page block — see
    /// [`WorkerHistories::backing_regions`](crate::WorkerHistories::backing_regions).
    /// The table is not zero-sized, so it always owns a block.
    pub fn backing_region(&self) -> (usize, usize) {
        heap_region(&self.table)
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

    /// The side-to-move `us` row, for a caller that reads many moves of one
    /// side; its [`MoveRow::get`] equals [`Self::get`] with `us`.
    pub fn row(&self, us: Color) -> MoveRow<'_> {
        MoveRow {
            entries: &self.table[us.index()],
        }
    }

    /// Gravity-update `mainHistory[us][move.raw16]` (`D = 7183`).
    #[inline(always)]
    pub fn update(&mut self, us: Color, m: Move, bonus: i32) {
        let cell = &mut self.table[us.index()][move16(m)];
        *cell = apply_gravity(*cell, bonus, MAIN_HISTORY_D);
    }
}

/// One row of a butterfly-shaped table (`mainHistory[us]`, `lowPlyHistory[ply]`),
/// indexed by the move's low 16 bits.
#[derive(Clone, Copy)]
pub struct MoveRow<'a> {
    entries: &'a [i16; 1 << 16],
}

impl MoveRow<'_> {
    /// The entry for `m`.
    pub fn get(self, m: Move) -> i32 {
        self.entries[move16(m)] as i32
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

// The plane is its entries and nothing else, so a plane held inside a larger
// table is reached by multiplying the plane number by this constant.
const _: () = assert!(size_of::<PieceToHistory>() == PIECE_NB * SQ_NB * size_of::<i16>());
const _: () = assert!(align_of::<PieceToHistory>() == align_of::<i16>());

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

    /// The entry at `ps`; equal to [`Self::get`] of the piece and square `ps`
    /// was built from.
    pub fn get_ps(&self, ps: PieceSquare) -> i32 {
        self.table.as_flattened()[ps.get()] as i32
    }

    /// Gravity-update the `[pc][to]` continuation entry (`D = 30000`).
    pub fn update(&mut self, pc: Piece, to: Square, bonus: i32) {
        let cell = &mut self.table[piece_code(pc)][to.index() as usize];
        *cell = apply_gravity(*cell, bonus, CONTINUATION_HISTORY_D);
    }
}

/// The storage of a [`LowPlyHistory`].
pub(crate) type LowPlyTable = [[i16; 1 << 16]; LOW_PLY_HISTORY_SIZE];

/// `lowPlyHistory[ply][move]` — the near-root quiet bonus. Re-filled per `go`
/// through [`fill`](LowPlyHistory::fill) at the root.
pub struct LowPlyHistory {
    /// Layout `[ply][move16]`, both dimensions compile-time.
    table: Box<LowPlyTable, WorkerAlloc>,
}

impl Default for LowPlyHistory {
    fn default() -> Self {
        Self {
            table: WorkerAlloc::zeroed(),
        }
    }
}

#[cfg(test)]
impl LowPlyHistory {
    /// The table's entries as bytes, for a test comparing two tables whole.
    pub(crate) fn as_bytes(&self) -> &[u8] {
        // SAFETY: the table is plain `i16` data with no padding, so every one
        // of its bytes is initialised.
        unsafe {
            std::slice::from_raw_parts(Box::as_ptr(&self.table).cast(), size_of_val(&*self.table))
        }
    }
}

impl LowPlyHistory {
    /// A fresh, zero-filled table.
    pub fn new() -> Self {
        Self::default()
    }

    /// The table over `table`, as it stands — a worker's block hands its
    /// storage over unfilled.
    pub(crate) fn over(table: Box<LowPlyTable, WorkerAlloc>) -> Self {
        Self { table }
    }

    /// The `(address, byte length)` of this table's large-page block — see
    /// [`WorkerHistories::backing_regions`](crate::WorkerHistories::backing_regions).
    /// The table is not zero-sized, so it always owns a block.
    pub fn backing_region(&self) -> (usize, usize) {
        heap_region(&self.table)
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

    /// The `ply` row, or `None` for a ply the table does not cover; its
    /// [`MoveRow::get`] equals [`Self::get`] with `ply`.
    pub fn row(&self, ply: usize) -> Option<MoveRow<'_>> {
        self.table.get(ply).map(|entries| MoveRow { entries })
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
/// pawn entries the registered `pawn_history_init`, whose range lies inside
/// `i16`.
const CORRECTION_INIT: i16 = 0;
fn pawn_init() -> i16 {
    crate::params::PAWN_HISTORY_INIT.get() as i16
}

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
/// The thread count must be a non-zero power of two (asserted) so slot
/// selection is a single mask over the full 64-bit key.
///
/// A value of this type is one worker's handle to its node's tables: each
/// table's base address and slot mask, which a lookup reads straight out of
/// the worker's own state, with nothing else on the path to an entry. Every
/// worker of a node holds a handle of its own, a clone of the node's; the
/// entries the workers write live in the tables these handles address, not in
/// the handle, and the tables are sized by the node's thread count exactly so
/// that contention for an entry thins out as the node grows.
///
/// A table is either in static storage, which lives as long as the program, or
/// on the heap, kept alive by the reference count every handle over it holds.
/// Static storage is never freed, so a set wholly in static storage counts
/// nothing.
#[derive(Clone)]
pub struct SharedHistories {
    /// Correction table: `(corr_mask + 1) * CORR_SLOT_LEN` atomic `i16`.
    /// Layout `[slot][color][channel]`, flat.
    correction: NonNull<AtomicI16>,
    /// `correctionHistory.get_size() - 1` — the correction slot mask.
    corr_mask: usize,
    /// Pawn table: `(pawn_mask + 1) * PAWN_SLOT_LEN` atomic `i16`. Layout
    /// `[plane][pc][to]`, flat.
    pawn: NonNull<AtomicI16>,
    /// `pawnHistory.get_size() - 1` — the pawn slot mask.
    pawn_mask: usize,
    /// Those of the two tables that are on the heap, held by every handle over
    /// them; `None` when both are in static storage.
    _heap: Option<Arc<HeapTables>>,
    /// The node's tables, which this worker's set reads and writes through for
    /// each table it does not keep a private copy of; `None` on a node's own
    /// set. See [`Self::private_over`].
    #[cfg(feature = "dupstats")]
    delegate: Option<Box<SharedHistories>>,
    /// Whether this set keeps its correction table to itself.
    #[cfg(feature = "dupstats")]
    corr_own: bool,
    /// Whether this set keeps its pawn table to itself.
    #[cfg(feature = "dupstats")]
    pawn_own: bool,
}

// Two table addresses, two masks and the heap handle. A worker holds the
// handle inline, so this is what it adds to the worker's own state.
#[cfg(not(feature = "dupstats"))]
const _: () = assert!(size_of::<SharedHistories>() == 40);

// SAFETY: the handle holds the addresses of `AtomicI16` entries, which are
// `Sync`, and the entries live for as long as the handle does: static storage
// for the whole program, and heap tables for as long as `heap`, an `Arc` that
// is itself `Send + Sync`. Every access through the handle is a shared
// reference to an atomic.
unsafe impl Send for SharedHistories {}
// SAFETY: as for `Send`.
unsafe impl Sync for SharedHistories {}

/// The tables of a set that are on the heap, behind every handle over them.
/// The boxes are never moved out or touched after the set is built: the
/// handles reach the entries through the addresses taken then.
struct HeapTables {
    _correction: Option<Box<[AtomicI16], LargePageAlloc>>,
    _pawn: Option<Box<[AtomicI16], LargePageAlloc>>,
}

/// The base of `table`, which must hold `len` entries.
fn table_base(table: &[AtomicI16], len: usize, name: &str) -> NonNull<AtomicI16> {
    assert_eq!(
        table.len(),
        len,
        "the {name} table does not hold the entries its thread count gives"
    );
    NonNull::from(table).cast()
}

impl SharedHistories {
    /// Entries in a correction table for a node of `thread_count` workers.
    pub const fn correction_len(thread_count: usize) -> usize {
        thread_count * CORRHIST_BASE_SIZE * CORR_SLOT_LEN
    }

    /// Entries in a pawn table for a node of `thread_count` workers.
    pub const fn pawn_len(thread_count: usize) -> usize {
        thread_count * PAWN_HISTORY_BASE_SIZE * PAWN_SLOT_LEN
    }

    /// Build the shared tables for a node of `thread_count` workers, which must
    /// be a non-zero power of two, both on the heap.
    ///
    /// The allocation **and** the initial fill run here, so calling this inside
    /// a node-bound thread lets first-touch place every page on that node.
    pub fn new(thread_count: usize) -> Self {
        Self::build(thread_count, None, None)
    }

    /// Build the shared tables for a node of `thread_count` workers, which must
    /// be a non-zero power of two: each table over the static storage given
    /// for it, which must hold exactly [`Self::correction_len`] or
    /// [`Self::pawn_len`] entries, and on the heap when none is given.
    ///
    /// Every entry of both tables is written here, so calling this inside a
    /// node-bound thread lets first-touch place every page on that node, and a
    /// table over static storage starts exactly as a fresh heap table does,
    /// whatever the storage held before.
    pub fn build(
        thread_count: usize,
        correction: Option<&'static [AtomicI16]>,
        pawn: Option<&'static [AtomicI16]>,
    ) -> Self {
        assert!(
            thread_count.is_power_of_two() && thread_count != 0,
            "SharedHistories thread_count must be a non-zero power of two, got {thread_count}"
        );
        let corr_len = Self::correction_len(thread_count);
        let pawn_len = Self::pawn_len(thread_count);
        let heap_correction = correction
            .is_none()
            .then(|| LargePageAlloc::zeroed_slice(corr_len));
        let heap_pawn = pawn
            .is_none()
            .then(|| LargePageAlloc::zeroed_slice(pawn_len));
        let heap = (heap_correction.is_some() || heap_pawn.is_some()).then(|| {
            Arc::new(HeapTables {
                _correction: heap_correction,
                _pawn: heap_pawn,
            })
        });
        let heap_tables = heap.as_deref();
        let correction = correction
            .or_else(|| heap_tables.and_then(|h| h._correction.as_deref()))
            .expect("a correction table without static storage is on the heap");
        let pawn = pawn
            .or_else(|| heap_tables.and_then(|h| h._pawn.as_deref()))
            .expect("a pawn table without static storage is on the heap");
        let correction = table_base(correction, corr_len, "correction");
        let pawn = table_base(pawn, pawn_len, "pawn");
        let out = Self {
            correction,
            corr_mask: thread_count * CORRHIST_BASE_SIZE - 1,
            pawn,
            pawn_mask: thread_count * PAWN_HISTORY_BASE_SIZE - 1,
            _heap: heap,
            #[cfg(feature = "dupstats")]
            delegate: None,
            #[cfg(feature = "dupstats")]
            corr_own: true,
            #[cfg(feature = "dupstats")]
            pawn_own: true,
        };
        // The correction fill of `0` is a no-op on a zeroed heap table, but is
        // written explicitly so every page is first-touched on the (bound)
        // node, and so static storage an earlier set left values in starts
        // over.
        out.fill_correction(CORRECTION_INIT);
        out.fill_pawn(pawn_init());
        out
    }

    /// One worker's set of the tables, for a measurement that cuts the sharing
    /// of one or both: the correction table (when `corr_own`) and the pawn
    /// table (when `pawn_own`) are the worker's own, on the heap and sized for
    /// one worker as a lone search's are, and every other read and write goes
    /// to `node`.
    #[cfg(feature = "dupstats")]
    pub fn private_over(node: SharedHistories, corr_own: bool, pawn_own: bool) -> Self {
        let mut own = Self::new(1);
        own.delegate = Some(Box::new(node));
        own.corr_own = corr_own;
        own.pawn_own = pawn_own;
        own
    }

    /// The node's thread count (the slot multiplier).
    pub fn thread_count(&self) -> usize {
        (self.corr_mask + 1) / CORRHIST_BASE_SIZE
    }

    /// Number of correction slots (`thread_count * CORRHIST_BASE_SIZE`).
    pub fn correction_slots(&self) -> usize {
        self.corr_mask + 1
    }

    /// Number of pawn slots (`thread_count * PAWN_HISTORY_BASE_SIZE`).
    pub fn pawn_slots(&self) -> usize {
        self.pawn_mask + 1
    }

    /// The correction table's `(address, byte length)`.
    pub fn correction_region(&self) -> (usize, usize) {
        let entries = self.correction_entries();
        (entries.as_ptr().addr(), size_of_val(entries))
    }

    /// The pawn table's `(address, byte length)`.
    pub fn pawn_region(&self) -> (usize, usize) {
        let entries = self.pawn_entries();
        (entries.as_ptr().addr(), size_of_val(entries))
    }

    /// Whether `self` and `other` address the same two tables.
    pub fn same_tables(&self, other: &Self) -> bool {
        self.correction == other.correction && self.pawn == other.pawn
    }

    /// Every correction entry.
    fn correction_entries(&self) -> &[AtomicI16] {
        // SAFETY: `correction` addresses `(corr_mask + 1) * CORR_SLOT_LEN`
        // entries, set together in `build`, that live as long as the handle
        // does (see the `Send` impl); an entry is only ever reached through a
        // shared reference, and an atomic allows any number of those.
        unsafe {
            std::slice::from_raw_parts(
                self.correction.as_ptr(),
                (self.corr_mask + 1) * CORR_SLOT_LEN,
            )
        }
    }

    /// Every pawn entry.
    fn pawn_entries(&self) -> &[AtomicI16] {
        // SAFETY: as in `Self::correction_entries`, for the pawn table's
        // `(pawn_mask + 1) * PAWN_SLOT_LEN` entries.
        unsafe {
            std::slice::from_raw_parts(self.pawn.as_ptr(), (self.pawn_mask + 1) * PAWN_SLOT_LEN)
        }
    }

    /// Overwrite every correction entry with `v` (atomic RELAXED stores).
    pub fn fill_correction(&self, v: i16) {
        for e in self.correction_entries() {
            e.store(v, Ordering::Relaxed);
        }
    }

    /// Overwrite every pawn entry with `v` (atomic RELAXED stores).
    pub fn fill_pawn(&self, v: i16) {
        for e in self.pawn_entries() {
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
    /// Both tables hold `slots * SLOT_LEN` entries and are masked with
    /// `slots - 1`, both set together in [`Self::build`], which is the only
    /// place a set is made, so a masked index is always inside the table. The
    /// length is a run-time value, so the bound has to be stated rather than
    /// derived from the type.
    fn corr_cell(&self, key: u64, color: Color, channel: CorrChannel) -> &AtomicI16 {
        #[cfg(feature = "dupstats")]
        if !self.corr_own
            && let Some(node) = &self.delegate
        {
            return node.corr_cell(key, color, channel);
        }
        let i = self.corr_index(key, color, channel);
        debug_assert!(i < (self.corr_mask + 1) * CORR_SLOT_LEN);
        // SAFETY: see the doc comment — `key & corr_mask < corr_mask + 1` and
        // the table holds `(corr_mask + 1) * CORR_SLOT_LEN` entries, which live
        // as long as the handle; an entry is only reached by shared reference.
        unsafe { &*self.correction.as_ptr().add(i) }
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
        #[cfg(feature = "dupstats")]
        if !self.pawn_own
            && let Some(node) = &self.delegate
        {
            return node.pawn_cell(pawn_key, pc, to);
        }
        let i = self.pawn_index(pawn_key, pc, to);
        debug_assert!(i < (self.pawn_mask + 1) * PAWN_SLOT_LEN);
        // SAFETY: `pawn_key & pawn_mask < pawn_mask + 1` and the table holds
        // `(pawn_mask + 1) * PAWN_SLOT_LEN` entries, which live as long as the
        // handle; an entry is only reached by shared reference.
        unsafe { &*self.pawn.as_ptr().add(i) }
    }

    /// The pawn-structure bonus for piece `pc` moving to `to` in the plane keyed
    /// by `pawn_key`. `-1238` on a fresh table by default.
    pub fn pawn_get(&self, pawn_key: u64, pc: Piece, to: Square) -> i32 {
        self.pawn_cell(pawn_key, pc, to).load(Ordering::Relaxed) as i32
    }

    /// Gravity-update the `[pc][to]` entry in the plane keyed by `pawn_key`
    /// (`D = 8192`).
    #[inline(always)]
    pub fn pawn_update(&self, pawn_key: u64, pc: Piece, to: Square, bonus: i32) {
        apply_gravity_atomic(self.pawn_cell(pawn_key, pc, to), bonus, PAWN_HISTORY_D);
    }

    /// Prefetch the entries of these tables that the position `child` will
    /// read, `child` having just been reached by moving `pc` to `to`: its four
    /// correction channels for its side to move, and the pawn-history entry
    /// `[pc][to]` in its pawn plane — the same set the reference prefetches
    /// inside its `do_move`.
    ///
    /// Each address is that of the cell the matching read would load, formed
    /// by the same masked lookup, so it is always inside the table.
    #[inline]
    pub fn prefetch_child(&self, child: &yorkie_state::Position, pc: Piece, to: Square) {
        let us = child.side_to_move();
        prefetch_t0(self.pawn_cell(child.pawn_key(), pc, to));
        prefetch_t0(self.corr_cell(child.pawn_key(), us, CorrChannel::Pawn));
        prefetch_t0(self.corr_cell(child.minor_piece_key(), us, CorrChannel::Minor));
        prefetch_t0(self.corr_cell(
            child.non_pawn_key(Color::White),
            us,
            CorrChannel::NonPawnWhite,
        ));
        prefetch_t0(self.corr_cell(
            child.non_pawn_key(Color::Black),
            us,
            CorrChannel::NonPawnBlack,
        ));
    }

    /// The whole plane keyed by `pawn_key`, for a caller that reads many
    /// entries of one position's plane: the key is masked, and the table's
    /// base read, once rather than per entry.
    pub fn pawn_plane(&self, pawn_key: u64) -> PawnPlane<'_> {
        #[cfg(feature = "dupstats")]
        if !self.pawn_own
            && let Some(node) = &self.delegate
        {
            return node.pawn_plane(pawn_key);
        }
        let start = ((pawn_key as usize) & self.pawn_mask) * PAWN_SLOT_LEN;
        debug_assert!(start + PAWN_SLOT_LEN <= (self.pawn_mask + 1) * PAWN_SLOT_LEN);
        // SAFETY: as in `Self::pawn_cell`, the masked plane number is below
        // `pawn_mask + 1` and the table holds `(pawn_mask + 1) * PAWN_SLOT_LEN`
        // entries, so the `PAWN_SLOT_LEN` entries from `start` are in the
        // table; an array of `AtomicI16` has the layout of that run.
        let entries = unsafe {
            &*self
                .pawn
                .as_ptr()
                .add(start)
                .cast::<[AtomicI16; PAWN_SLOT_LEN]>()
        };
        PawnPlane { entries }
    }
}

/// One plane of the pawn history, resolved from a pawn key by
/// [`SharedHistories::pawn_plane`].
#[derive(Clone, Copy)]
pub struct PawnPlane<'a> {
    entries: &'a [AtomicI16; PAWN_SLOT_LEN],
}

impl PawnPlane<'_> {
    /// The entry at `ps`; equal to [`SharedHistories::pawn_get`] with this
    /// plane's key and the piece and square `ps` was built from.
    pub fn get(self, ps: PieceSquare) -> i32 {
        self.entries[ps.get()].load(Ordering::Relaxed) as i32
    }
}

/// The storage of a [`ContinuationCorrectionHistory`].
pub(crate) type ContCorrTable = [[[i16; SQ_NB]; PIECE_NB]; CONT_CORR_PLANES];

/// `CorrectionHistory<Continuation>`: a `[pc][to]` table whose every cell is
/// itself a `[pc][to]` `i16` table.
///
/// The outer `[pc][to]` selects a *plane* — the one a search stack cell's
/// `continuationCorrectionHistory` points at — and the inner `[pc][to]` indexes
/// within it. [`CorrPlane::SENTINEL`] is what pre-root cells are seeded with.
pub struct ContinuationCorrectionHistory {
    /// Layout: `[plane = outer_pc*SQ_NB + outer_to][inner_pc][inner_to]`. Every
    /// dimension is fixed so each access carries a compile-time length.
    table: Box<ContCorrTable, WorkerAlloc>,
}

impl Default for ContinuationCorrectionHistory {
    fn default() -> Self {
        // Huge-page-backed, zero-initialised; the compile-time dimensions are
        // preserved by the `Box`.
        Self {
            table: WorkerAlloc::zeroed(),
        }
    }
}

#[cfg(test)]
impl ContinuationCorrectionHistory {
    /// The table's entries as bytes, for a test comparing two tables whole.
    pub(crate) fn as_bytes(&self) -> &[u8] {
        // SAFETY: the table is plain `i16` data with no padding, so every one
        // of its bytes is initialised.
        unsafe {
            std::slice::from_raw_parts(Box::as_ptr(&self.table).cast(), size_of_val(&*self.table))
        }
    }
}

impl ContinuationCorrectionHistory {
    /// A fresh, zero-filled table. Use [`Self::fill`] to apply the reference's
    /// init value of `6`.
    pub fn new() -> Self {
        Self::default()
    }

    /// The table over `table`, as it stands — a worker's block hands its
    /// storage over unfilled.
    pub(crate) fn over(table: Box<ContCorrTable, WorkerAlloc>) -> Self {
        Self { table }
    }

    /// The `(address, byte length)` of this table's large-page block — see
    /// [`WorkerHistories::backing_regions`](crate::WorkerHistories::backing_regions).
    /// The table is not zero-sized, so it always owns a block.
    pub fn backing_region(&self) -> (usize, usize) {
        heap_region(&self.table)
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

    /// Prefetch the inner `[pc][to]` entry in `plane`, the one [`Self::get_at`]
    /// with the same arguments reads. The address is a bounds-checked index.
    #[inline]
    pub fn prefetch_at(&self, plane: CorrPlane, pc: Piece, to: Square) {
        prefetch_t0(&self.table[plane.get()][piece_code(pc)][to.index() as usize]);
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
    table: Box<[PieceToHistory; CONT_PLANES], WorkerAlloc>,
}

impl Default for ContinuationHistory {
    fn default() -> Self {
        Self {
            table: WorkerAlloc::zeroed(),
        }
    }
}

#[cfg(test)]
impl ContinuationHistory {
    /// The table's entries as bytes, for a test comparing two tables whole.
    pub(crate) fn as_bytes(&self) -> &[u8] {
        // SAFETY: the table is plain `i16` data with no padding, so every one
        // of its bytes is initialised.
        unsafe {
            std::slice::from_raw_parts(Box::as_ptr(&self.table).cast(), size_of_val(&*self.table))
        }
    }
}

impl ContinuationHistory {
    /// A fresh, zero-filled table on the heap. Use [`Self::fill`] for the
    /// reference init.
    pub fn new() -> Self {
        Self::default()
    }

    /// The table over `table`, as it stands — a worker's block hands its
    /// storage over unfilled.
    pub(crate) fn over(table: Box<[PieceToHistory; CONT_PLANES], WorkerAlloc>) -> Self {
        Self { table }
    }

    /// The `(address, byte length)` of this table's large-page block — see
    /// [`WorkerHistories::backing_regions`](crate::WorkerHistories::backing_regions).
    /// This is the big one: ~54 MiB, more than three quarters of a worker's
    /// private footprint. The table is not zero-sized, so it always owns a
    /// block.
    pub fn backing_region(&self) -> (usize, usize) {
        heap_region(&self.table)
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

    /// The inner entry `ps` in `plane`; equal to [`Self::get_at`] of the piece
    /// and square `ps` was built from.
    pub fn get_at_ps(&self, plane: ContPlane, ps: PieceSquare) -> i32 {
        self.table[plane.get()].get_ps(ps)
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

// One gravity entry, at the width every other history entry has.
const _: () = assert!(size_of::<TtMoveHistory>() == size_of::<i16>());

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

    /// A resolved pawn plane reads the entries `pawn_get` reads, for keys that
    /// land on the first plane, the last one and planes between, at two table
    /// sizes.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn pawn_plane_reads_the_entries_pawn_get_reads() {
        let pieces = [
            bp(),
            Piece::new(PieceKind::Gold, Color::White),
            Piece {
                kind: PieceKind::Rook,
                color: Color::White,
                promoted: true,
            },
        ];
        let squares = [Square::new(0, 0).unwrap(), to(), Square::new(8, 8).unwrap()];
        for tc in [1usize, 2] {
            let sh = SharedHistories::new(tc);
            let keys = [0u64, 1, 0x1234_5678_9abc_def0, u64::MAX];
            for (k, &key) in keys.iter().enumerate() {
                for (i, &pc) in pieces.iter().enumerate() {
                    for (j, &sq) in squares.iter().enumerate() {
                        let bonus = 1000 * (k as i32 + 1) + 100 * i as i32 + 10 * j as i32;
                        sh.pawn_update(key, pc, sq, bonus);
                    }
                }
            }
            for &key in &keys {
                let plane = sh.pawn_plane(key);
                for &pc in &pieces {
                    for &sq in &squares {
                        assert_eq!(
                            plane.get(PieceSquare::new(pc, sq)),
                            sh.pawn_get(key, pc, sq),
                            "tc {tc} key {key:#x}"
                        );
                    }
                }
            }
        }
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
        let sh = SharedHistories::new(2);
        let key = 12_345u64;
        let handles: Vec<_> = (0..2)
            .map(|t| {
                let sh = sh.clone();
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

    /// A clone of a handle addresses the tables the original does, so an
    /// update through one is read through the other, and the heap tables stay
    /// alive while any handle over them does; a second set is a second pair
    /// of tables.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn a_clone_addresses_the_same_tables_and_keeps_them_alive() {
        let first = SharedHistories::new(1);
        let second = first.clone();
        assert!(first.same_tables(&second));
        assert_eq!(first.correction_region(), second.correction_region());
        assert_eq!(first.pawn_region(), second.pawn_region());
        drop(first);
        second.pawn_update(7, bp(), to(), 4000);
        second.correction_update(7, Color::Black, CorrChannel::Minor, 500);
        let third = second.clone();
        assert_eq!(
            third.pawn_get(7, bp(), to()),
            second.pawn_get(7, bp(), to())
        );
        assert_ne!(third.pawn_get(7, bp(), to()), pawn_init() as i32);
        assert_ne!(third.correction_get(7, Color::Black, CorrChannel::Minor), 0);
        let other = SharedHistories::new(1);
        assert!(!other.same_tables(&second));
        let (a, alen) = other.pawn_region();
        let (b, blen) = second.pawn_region();
        assert!(a + alen <= b || b + blen <= a, "two sets overlap");
    }

    /// A table given static storage is built over that storage, its every
    /// entry written to the initial value whatever it held, and reads and
    /// writes through it exactly as a heap table does.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn a_table_over_static_storage_is_filled_and_used_in_place() {
        let corr: &'static [AtomicI16] = Box::leak(
            (0..SharedHistories::correction_len(1))
                .map(|_| AtomicI16::new(77))
                .collect(),
        );
        let sh = SharedHistories::build(1, Some(corr), None);
        assert_eq!(
            sh.correction_region(),
            (corr.as_ptr().addr(), size_of_val(corr))
        );
        assert!(corr.iter().all(|e| e.load(Ordering::Relaxed) == 0));
        sh.correction_update(3, Color::White, CorrChannel::Pawn, 300);
        let i = sh.corr_index(3, Color::White, CorrChannel::Pawn);
        assert_eq!(
            corr[i].load(Ordering::Relaxed) as i32,
            sh.correction_get(3, Color::White, CorrChannel::Pawn)
        );
        assert_ne!(corr[i].load(Ordering::Relaxed), 0);
        assert_eq!(sh.pawn_get(3, bp(), to()), pawn_init() as i32);
    }

    /// Static storage of the wrong length is refused.
    #[test]
    #[should_panic(expected = "does not hold the entries")]
    fn static_storage_of_the_wrong_length_is_refused() {
        static SHORT: [AtomicI16; 8] = [const { AtomicI16::new(0) }; 8];
        let _ = SharedHistories::build(1, Some(&SHORT), Some(&SHORT));
    }
}
