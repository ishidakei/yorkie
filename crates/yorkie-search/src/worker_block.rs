//! Each search worker's fixed-size tables, in one block of static storage per
//! worker.
//!
//! The worker count is a build setting, and so is the size of every table a
//! worker keeps to itself, so the storage for all of them is laid out when the
//! binary is built: a region of [`WORKER_BLOCKS`] consecutive [`WorkerBlock`]s
//! in the binary's BSS, stated to the assembler through
//! [`yorkie_storage::large_page_bss!`] exactly as the transposition table's is.
//! The region is zero at process start and occupies no bytes of the file.
//!
//! # Placement
//!
//! A block is [`LARGE_PAGE_ALIGN`]-aligned and its size is a whole number of
//! large pages, so the region starts on a large-page boundary and no large page
//! holds tables of two workers. The region is hinted for huge pages once, when
//! the first block is claimed. Nothing writes a block until its owner does:
//! claiming one writes nothing, and every table in it is first written by the
//! thread that runs the worker, after that thread has been pinned. A page
//! therefore faults in on the node of the worker that uses it, as a block the
//! worker allocated for itself would.
//!
//! # Ownership
//!
//! [`BlockClaim::claim`] is the one way to a block. It hands block `w` to at
//! most one owner at a time — a second claim while the first is held is
//! refused — and with the claim, once, the history tables carved out of it,
//! each an ordinary `Box` over its own disjoint part of the block. The tables
//! one search works in are lent out of the block by the claim for each search
//! ([`BlockClaim::lend_search_tables`]), written fresh as a heap allocation
//! for that search would have been — all but the accumulator stack, whose
//! old contents no search can see (see [`SearchTables::acc_stack`]). The next
//! owner of the block treats whatever it finds there as garbage to overwrite.
//!
//! A block is held for as long as anything over it is alive: the claim, and
//! every box over the block. Each block keeps a count of them, set when the
//! block is claimed, raised by each lending and lowered by dropping the claim
//! and by freeing each box, and the block is given back only when that count
//! reaches zero. In which order the claim and the boxes go therefore does not
//! matter: a box kept after its claim is dropped keeps the block held, so no
//! second claim can hand out boxes over the same memory while it lives, and a
//! box that is leaked leaves the block held for good rather than shared.
//!
//! The boxes use [`WorkerAlloc`], whose `deallocate` leaves memory inside the
//! region alone — it only lowers the count of the block the memory is in —
//! and returns anything else to the large-page allocator. That is
//! what lets the same table type sit either in a worker's block or on the heap:
//! a worker whose index has no block (a measurement pool larger than the
//! configured worker count, or a second engine in a test process) and every
//! table a test builds on its own get heap boxes, with the code that reads them
//! unchanged. A box is a pointer either way, so a table access compiles to the
//! same load through the same field it always did; nothing indexes the region
//! by worker at an access.
//!
//! Under miri, which can neither assemble the region nor take its address, the
//! region is one zeroed heap allocation made on first use and never freed.

use std::alloc::{AllocError, Allocator, Layout, handle_alloc_error};
use std::mem::MaybeUninit;
use std::num::NonZeroU32;
use std::ptr::NonNull;
use std::sync::Once;
use std::sync::atomic::{AtomicUsize, Ordering};

use yorkie_eval::{Accumulator, FinnyCache};
use yorkie_storage::{LARGE_PAGE_ALIGN, LargePageAlloc, Zeroable};

use crate::history::{
    ButterflyTable, CONT_PLANES, CaptureTable, ContCorrTable, LowPlyTable, PieceToHistory,
};
use crate::movepick::PickerScratch;
use crate::qsearch::{
    ACC_LEN, AccStack, MAX_PLY, PICKER_SCRATCH_LEN, PickerStack, STACK_LEN, SearchStack,
};
use crate::update::SearchStackCell;

/// Blocks in the region: one per worker the build is configured for.
pub const WORKER_BLOCKS: usize = crate::config::THREADS as usize;

const _: () = assert!(WORKER_BLOCKS >= 1);

/// One worker's tables, each at a fixed offset on a 64 KiB boundary
/// (see `Spaced`).
///
/// The type's alignment is the large-page size, so its size is rounded up to a
/// whole number of large pages: that padding is what keeps two workers' tables
/// off one large page.
///
/// The history tables and the accumulator stack are integer arrays, valid
/// whatever bytes they hold. The search's other tables, behind `MaybeUninit`,
/// are written fresh each time a search borrows them
/// ([`BlockClaim::lend_search_tables`]), exactly as they were allocated fresh
/// for each search on the heap.
#[repr(C, align(2097152))]
pub struct WorkerBlock {
    finny: Spaced<MaybeUninit<FinnyCache>>,
    stack: Spaced<MaybeUninit<SearchStack>>,
    picker_scratch: Spaced<MaybeUninit<PickerStack>>,
    acc_stack: Spaced<AccStack>,
    capture: Spaced<CaptureTable>,
    low_ply: Spaced<LowPlyTable>,
    main: Spaced<ButterflyTable>,
    continuation_correction: Spaced<ContCorrTable>,
    continuation: Spaced<[PieceToHistory; CONT_PLANES]>,
}

/// One table of a [`WorkerBlock`], started on a 64 KiB boundary.
///
/// The search reads and writes several of these tables at once, node after
/// node, and the low bits of their addresses decide which cache sets they
/// share. Packed end to end, the tables' starts fall wherever the sizes before
/// them leave them, and with the code's alignment held fixed that measured
/// about 1% slower in a single-thread search than the same tables each in an
/// allocation of its own, which the global allocator starts on a 64 KiB
/// boundary. This boundary puts them where those allocations do, and costs the
/// block nothing: the padding fits in the large pages the block already rounds
/// up to.
#[repr(C, align(65536))]
struct Spaced<T>(T);

const _: () = assert!(align_of::<WorkerBlock>() == 2 * 1024 * 1024);
const _: () = assert!(align_of::<WorkerBlock>().is_multiple_of(LARGE_PAGE_ALIGN));
const _: () = assert!(size_of::<WorkerBlock>().is_multiple_of(LARGE_PAGE_ALIGN));
const _: () = assert!(size_of::<WorkerBlock>() > 0);

/// The region: [`WORKER_BLOCKS`] blocks, back to back.
#[repr(C)]
pub struct WorkerRegion {
    blocks: [WorkerBlock; WORKER_BLOCKS],
}

/// Bytes the region occupies.
pub const REGION_BYTES: usize = size_of::<WorkerRegion>();

const _: () = assert!(REGION_BYTES == WORKER_BLOCKS * size_of::<WorkerBlock>());
const _: () = assert!(align_of::<WorkerRegion>() == align_of::<WorkerBlock>());

yorkie_storage::large_page_bss! {
    /// The region's storage.
    fn region_storage() -> *mut WorkerRegion = far "yorkie_worker_region";
}

/// The region's base under miri.
#[cfg(miri)]
struct MiriRegion(*mut WorkerBlock);

// SAFETY: the pointer is only an address the claims compute blocks from; who
// may touch a block is decided by `HOLDERS`, not by holding this.
#[cfg(miri)]
unsafe impl Send for MiriRegion {}
// SAFETY: as for `Send`.
#[cfg(miri)]
unsafe impl Sync for MiriRegion {}

/// The region's stand-in under miri, allocated on first use.
#[cfg(miri)]
static MIRI_REGION: std::sync::OnceLock<MiriRegion> = std::sync::OnceLock::new();

/// The base of the region.
fn region() -> *mut WorkerBlock {
    #[cfg(not(miri))]
    {
        region_storage().cast()
    }
    #[cfg(miri)]
    {
        MIRI_REGION
            .get_or_init(|| {
                let layout = Layout::new::<WorkerRegion>();
                // SAFETY: the layout is non-zero-sized.
                let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
                if ptr.is_null() {
                    handle_alloc_error(layout);
                }
                MiriRegion(ptr.cast())
            })
            .0
    }
}

/// The index of the block `ptr` lies in, or `None` when it lies outside the
/// region. Under miri a region that was never allocated contains nothing, and
/// asking does not allocate it.
fn block_of(ptr: NonNull<u8>) -> Option<usize> {
    #[cfg(not(miri))]
    let base = region().addr();
    #[cfg(miri)]
    let base = MIRI_REGION.get().map(|r| r.0.addr())?;
    let offset = ptr.addr().get().checked_sub(base)?;
    (offset < REGION_BYTES).then(|| offset / size_of::<WorkerBlock>())
}

/// What holds each block: its claim and every box over it, counted. Zero is a
/// free block.
///
/// A claim takes a block by moving its count off zero with `Acquire`, and each
/// holder lets go with a `Release` decrement. The decrements are
/// read-modify-writes, so the one that reaches zero carries all of them, and
/// the next claim, reading that zero, sees every write any holder made.
static HOLDERS: [AtomicUsize; WORKER_BLOCKS] = [const { AtomicUsize::new(0) }; WORKER_BLOCKS];

/// Boxes [`BlockClaim::claim`] carves: the fields of [`BlockTables`].
const CARVED_BOXES: usize = 5;

/// Boxes [`BlockClaim::lend_search_tables`] lends: the fields of
/// [`SearchTables`].
const LENT_BOXES: usize = 4;

/// Let go of one hold on block `block`.
fn release(block: usize) {
    let before = HOLDERS[block].fetch_sub(1, Ordering::Release);
    debug_assert!(before > 0, "block {block} released more often than held");
}

/// The huge-page hint over the whole region, given once.
static ADVISED: Once = Once::new();

/// The allocator behind every per-worker table's box.
///
/// It allocates from [`LargePageAlloc`], and frees there everything except
/// memory inside the worker region, which belongs to a block rather than to an
/// allocation and is left as it is.
#[derive(Clone, Copy, Debug, Default)]
pub struct WorkerAlloc;

impl WorkerAlloc {
    /// A zeroed `T` on the heap, its base on a [`LARGE_PAGE_ALIGN`] boundary.
    /// Nothing writes the bytes, so no page is touched until the caller first
    /// writes one.
    pub fn zeroed<T: Zeroable>() -> Box<T, Self> {
        let layout = Layout::new::<T>();
        let ptr = Self
            .allocate_zeroed(layout)
            .unwrap_or_else(|_| handle_alloc_error(layout))
            .cast::<T>();
        // SAFETY: `ptr` was allocated by `WorkerAlloc` with `Layout::new::<T>()`,
        // the layout a `Box<T>` frees with, outside the region, so it is owned
        // by nothing else. Its bytes are zero, which `T: Zeroable` makes a
        // valid `T`.
        unsafe { Box::from_non_null_in(ptr, Self) }
    }

    /// A heap `T` whose value `init` writes in place, so a large value never
    /// passes through the stack.
    ///
    /// # Safety
    ///
    /// `init` must leave a valid `T` at the pointer it is given, which is valid
    /// for writes of one `T` and aligned for it.
    unsafe fn new_with<T>(init: impl FnOnce(*mut T)) -> Box<T, Self> {
        let layout = Layout::new::<T>();
        let ptr = Self
            .allocate(layout)
            .unwrap_or_else(|_| handle_alloc_error(layout))
            .cast::<T>();
        init(ptr.as_ptr());
        // SAFETY: `ptr` was allocated by `WorkerAlloc` with `Layout::new::<T>()`
        // and is owned by nothing else, and the caller guarantees `init` left a
        // valid `T` there.
        unsafe { Box::from_non_null_in(ptr, Self) }
    }
}

// SAFETY: `allocate` and `allocate_zeroed` forward to `LargePageAlloc`, which
// never hands out memory inside the region (the region is static storage, not
// heap), so `deallocate` returns every block they produced to the allocator
// that produced it, with the caller's layout, exactly as `LargePageAlloc`
// requires. A pointer inside the region came from `BlockClaim`, which counted
// the box it made as one of the block's holders; freeing it is not a
// deallocation but that holder letting go, and the memory stays with the
// block.
unsafe impl Allocator for WorkerAlloc {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        LargePageAlloc.allocate(layout)
    }

    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        LargePageAlloc.allocate_zeroed(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if let Some(block) = block_of(ptr) {
            release(block);
            return;
        }
        // SAFETY: outside the region every pointer this allocator handed out
        // came from `LargePageAlloc` under the same layout (see the impl's
        // safety comment), and the caller guarantees it is freed once.
        unsafe { LargePageAlloc.deallocate(ptr, layout) }
    }
}

/// A worker's history tables as [`BlockClaim::claim`] carves them out of its
/// block, each a box over its own field of it. Each holds what the block was
/// left holding: zero in a block nobody held before, the last owner's values
/// otherwise.
pub struct BlockTables {
    /// `captureHistory`.
    pub capture: Box<CaptureTable, WorkerAlloc>,
    /// `lowPlyHistory`.
    pub low_ply: Box<LowPlyTable, WorkerAlloc>,
    /// `mainHistory`.
    pub main: Box<ButterflyTable, WorkerAlloc>,
    /// `continuationCorrectionHistory`.
    pub continuation_correction: Box<ContCorrTable, WorkerAlloc>,
    /// `continuationHistory`.
    pub continuation: Box<[PieceToHistory; CONT_PLANES], WorkerAlloc>,
}

/// The tables one search works in: built fresh for every search, and kept
/// only as long as it runs.
pub struct SearchTables {
    /// The search stack, every cell at its default with its `pv` reserved to
    /// the longest line, so a PV update never grows it on the hot path.
    pub stack: Box<SearchStack, WorkerAlloc>,
    /// The accumulator stack: every slot zero on the heap, and as the last
    /// search left it in a block. A search reads no slot it has not written
    /// first — it refreshes slot 0 from the root position before anything
    /// reads it, and derives each deeper slot whole from the one below before
    /// the node that owns it reads it — so what a slot held before the search
    /// is never seen, and the block lends the stack without writing it.
    pub acc_stack: Box<AccStack, WorkerAlloc>,
    /// The move-picker buffers, every one empty. The buffers carry their
    /// storage inline, so this is the only block any of them costs.
    pub picker_scratch: Box<PickerStack, WorkerAlloc>,
    /// The finny table, every entry cold. Its feature lists stay heap vectors.
    pub finny: Box<FinnyCache, WorkerAlloc>,
}

impl SearchTables {
    /// Fresh tables on the heap, for a search whose histories have no block.
    pub fn heap() -> Self {
        // SAFETY: each writer leaves a valid value at the pointer it is given,
        // which `new_with` makes valid for writes and aligned.
        unsafe {
            Self {
                stack: WorkerAlloc::new_with(|p| write_search_stack(p)),
                acc_stack: WorkerAlloc::new_with(|p| write_acc_stack(p)),
                picker_scratch: WorkerAlloc::new_with(|p| write_picker_stack(p)),
                finny: WorkerAlloc::new_with(|p| FinnyCache::write_new(p)),
            }
        }
    }
}

/// Write a fresh cell into every slot of `p`.
///
/// # Safety
///
/// `p` must be valid for writes of one [`SearchStack`] and aligned for it.
unsafe fn write_search_stack(p: *mut SearchStack) {
    for slot in 0..STACK_LEN {
        let mut cell = SearchStackCell::default();
        cell.pv.reserve_exact(MAX_PLY as usize + 1);
        // SAFETY: `slot < STACK_LEN`, so the slot is inside the storage the
        // caller hands over.
        unsafe { (&raw mut (*p)[slot]).write(cell) };
    }
}

/// Write a zeroed accumulator into every slot of `p`.
///
/// # Safety
///
/// `p` must be valid for writes of one [`AccStack`] and aligned for it.
unsafe fn write_acc_stack(p: *mut AccStack) {
    for slot in 0..ACC_LEN {
        // SAFETY: `slot < ACC_LEN`, so the slot is inside the storage the
        // caller hands over.
        unsafe { (&raw mut (*p)[slot]).write(Accumulator::new()) };
    }
}

/// Write an empty buffer into every slot of `p`.
///
/// # Safety
///
/// `p` must be valid for writes of one [`PickerStack`] and aligned for it.
unsafe fn write_picker_stack(p: *mut PickerStack) {
    for slot in 0..PICKER_SCRATCH_LEN {
        // SAFETY: `slot < PICKER_SCRATCH_LEN`, so the slot is inside the
        // storage the caller hands over.
        unsafe { (&raw mut (*p)[slot]).write(PickerScratch::new()) };
    }
}

/// The hold on one worker's block, the one value that can lend the block's
/// search tables.
///
/// The claim is one of the block's holders, as is every box over the block —
/// the tables [`Self::claim`] carved and those [`Self::lend_search_tables`]
/// lent. The block is given back when the last of them is dropped, in
/// whichever order they go (see the module documentation).
#[derive(Debug)]
pub struct BlockClaim {
    /// The worker's index plus one: a non-zero `u32`, so `Option<BlockClaim>`
    /// is four bytes and fits in the tail of the bundle that holds it.
    slot: NonZeroU32,
}

impl BlockClaim {
    /// Hold worker `worker`'s block, and carve its tables out of it.
    ///
    /// `None` when the build has no block for `worker`, or when the block is
    /// still held — by an earlier claim, or by any box over it that outlives
    /// that claim. Nothing in the block is written.
    pub fn claim(worker: usize) -> Option<(Self, BlockTables)> {
        let holders = HOLDERS.get(worker)?;
        if holders
            .compare_exchange(0, 1 + CARVED_BOXES, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return None;
        }
        let base = region();
        ADVISED.call_once(|| {
            yorkie_storage::advise_huge_pages(base.addr(), REGION_BYTES);
        });
        // SAFETY: `worker < WORKER_BLOCKS`, so the block is inside the region.
        let block = unsafe { base.add(worker) };
        // SAFETY: the count just moved off zero makes this the only holder of
        // the block — no box over it from an earlier owner is left — and the
        // boxes are carved from distinct fields of it, so none aliases another
        // or anything else. Each field is an array of integers, valid whatever
        // bytes it holds — the zeros BSS starts as, or what the last owner
        // left — and correctly aligned at its offset in an aligned block. The
        // count already includes these boxes, and freeing one only lowers it.
        let tables = unsafe {
            BlockTables {
                capture: Box::from_raw_in(&raw mut (*block).capture.0, WorkerAlloc),
                low_ply: Box::from_raw_in(&raw mut (*block).low_ply.0, WorkerAlloc),
                main: Box::from_raw_in(&raw mut (*block).main.0, WorkerAlloc),
                continuation_correction: Box::from_raw_in(
                    &raw mut (*block).continuation_correction.0,
                    WorkerAlloc,
                ),
                continuation: Box::from_raw_in(&raw mut (*block).continuation.0, WorkerAlloc),
            }
        };
        let slot = u32::try_from(worker + 1)
            .ok()
            .and_then(NonZeroU32::new)
            .expect("a block index fits in u32");
        Some((Self { slot }, tables))
    }

    /// The block's search tables, written fresh in place, for one search.
    ///
    /// # Safety
    ///
    /// No box an earlier call returned may be left: each call writes the same
    /// storage over again and hands out new boxes over it.
    pub unsafe fn lend_search_tables(&mut self) -> SearchTables {
        // This claim is alive, so the count is above zero and the block stays
        // held; the new boxes join its holders before they exist.
        HOLDERS[self.worker()].fetch_add(LENT_BOXES, Ordering::Relaxed);
        // SAFETY: the block is this claim's alone, and the caller guarantees no
        // earlier box over these fields is left, so writing them races and
        // aliases nothing. Each writer leaves a valid value in its field, and
        // the accumulator stack is integer data valid whatever bytes it holds;
        // every field is aligned at its offset in an aligned block, and
        // freeing one of the boxes only lowers the block's count.
        unsafe {
            let block = self.block();
            let stack = (&raw mut (*block).stack.0).cast::<SearchStack>();
            write_search_stack(stack);
            let acc_stack = &raw mut (*block).acc_stack.0;
            let picker_scratch = (&raw mut (*block).picker_scratch.0).cast::<PickerStack>();
            write_picker_stack(picker_scratch);
            let finny = (&raw mut (*block).finny.0).cast::<FinnyCache>();
            FinnyCache::write_new(finny);
            SearchTables {
                stack: Box::from_raw_in(stack, WorkerAlloc),
                acc_stack: Box::from_raw_in(acc_stack, WorkerAlloc),
                picker_scratch: Box::from_raw_in(picker_scratch, WorkerAlloc),
                finny: Box::from_raw_in(finny, WorkerAlloc),
            }
        }
    }

    /// The block itself.
    fn block(&self) -> *mut WorkerBlock {
        // SAFETY: `worker < WORKER_BLOCKS` for every claim, so the block is
        // inside the region.
        unsafe { region().add(self.worker()) }
    }

    /// The worker whose block this is.
    pub fn worker(&self) -> usize {
        self.slot.get() as usize - 1
    }

    /// The `(address, byte length)` of the block, for a caller placing it: a
    /// whole number of large pages, naming no other worker's page.
    pub fn backing_region(&self) -> (usize, usize) {
        let addr = region().addr() + self.worker() * size_of::<WorkerBlock>();
        (addr, size_of::<WorkerBlock>())
    }
}

impl Drop for BlockClaim {
    fn drop(&mut self) {
        release(self.worker());
    }
}

/// Serialises the tests that claim blocks: a test binary run as one process
/// runs its tests on parallel threads, and the region is the process's.
#[cfg(test)]
pub(crate) fn claim_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every table starts on a multiple of its own alignment, and the tables
    /// do not overlap.
    #[test]
    fn every_table_sits_at_its_alignment_without_overlap() {
        macro_rules! span {
            ($field:ident, $ty:ty) => {
                (
                    core::mem::offset_of!(WorkerBlock, $field),
                    size_of::<$ty>(),
                    align_of::<$ty>(),
                )
            };
        }
        let mut spans = [
            span!(stack, SearchStack),
            span!(acc_stack, AccStack),
            span!(picker_scratch, PickerStack),
            span!(finny, FinnyCache),
            span!(capture, CaptureTable),
            span!(low_ply, LowPlyTable),
            span!(main, ButterflyTable),
            span!(continuation_correction, ContCorrTable),
            span!(continuation, [PieceToHistory; CONT_PLANES]),
        ];
        for &(offset, _, align) in &spans {
            assert!(offset.is_multiple_of(align));
        }
        spans.sort_unstable();
        for pair in spans.windows(2) {
            assert!(pair[0].0 + pair[0].1 <= pair[1].0, "{pair:?} overlap");
        }
        let (last, len, _) = spans[spans.len() - 1];
        assert!(last + len <= size_of::<WorkerBlock>());
    }

    /// The blocks tile the region: each a whole number of large pages, back
    /// to back, `WORKER_BLOCKS` of them.
    #[test]
    fn the_region_is_its_blocks_back_to_back() {
        assert!(size_of::<WorkerBlock>().is_multiple_of(LARGE_PAGE_ALIGN));
        assert_eq!(REGION_BYTES, WORKER_BLOCKS * size_of::<WorkerBlock>());
        assert!(region().addr().is_multiple_of(LARGE_PAGE_ALIGN));
    }

    /// The region is far storage: the linker places it after the
    /// transposition table, which is near, so the region never lies between
    /// the code and the table's start however many blocks it holds.
    #[cfg(not(miri))]
    #[test]
    fn the_region_lies_past_the_end_of_the_transposition_table() {
        let table = std::ptr::from_ref(yorkie_storage::TranspositionTable::shared());
        let table_end = table.addr() + size_of::<yorkie_storage::TranspositionTable>();
        assert!(
            table_end <= region().addr(),
            "the worker region at {:#x} lies before the end of the transposition table at {table_end:#x}",
            region().addr()
        );
    }

    /// A held block is refused to a second claimer, a worker the build has no
    /// block for gets none, and a block given back can be claimed again —
    /// over the same memory, disjoint from every other worker's block.
    #[test]
    fn a_block_has_one_holder_at_a_time() {
        let _lock = claim_lock();
        let (first, tables) = BlockClaim::claim(0).expect("block 0 is free");
        assert!(BlockClaim::claim(0).is_none(), "block 0 is already held");
        assert!(BlockClaim::claim(WORKER_BLOCKS).is_none());
        let (addr, len) = first.backing_region();
        assert!(addr.is_multiple_of(LARGE_PAGE_ALIGN));
        assert_eq!(len, size_of::<WorkerBlock>());
        let at = std::ptr::from_ref(&*tables.continuation).addr();
        assert!(at >= addr && at + size_of_val(&*tables.continuation) <= addr + len);
        for other in (0..WORKER_BLOCKS).skip(1) {
            let (other_claim, _other_tables) = BlockClaim::claim(other).expect("free");
            let (o, olen) = other_claim.backing_region();
            assert!(
                o + olen <= addr || addr + len <= o,
                "blocks 0 and {other} overlap"
            );
        }
        drop(tables);
        drop(first);
        let (again, _tables) = BlockClaim::claim(0).expect("block 0 was given back");
        assert_eq!(again.backing_region(), (addr, len));
    }

    /// A search's tables are lent out of the block itself, those that are not
    /// plain integer data written fresh each time, and the block can lend them
    /// again once they are given back.
    #[test]
    fn a_blocks_search_tables_are_lent_fresh_from_the_block() {
        let _lock = claim_lock();
        let (mut claim, _tables) = BlockClaim::claim(0).expect("block 0 is free");
        let (addr, len) = claim.backing_region();
        for _ in 0..2 {
            // SAFETY: no earlier lent table is left: each round's are dropped
            // at its end.
            let mut lent = unsafe { claim.lend_search_tables() };
            let inside = |at: usize, bytes: usize| at >= addr && at + bytes <= addr + len;
            assert!(inside(
                std::ptr::from_ref(&*lent.acc_stack).addr(),
                size_of::<AccStack>()
            ));
            assert!(inside(
                std::ptr::from_ref(&*lent.picker_scratch).addr(),
                size_of::<PickerStack>()
            ));
            assert!(lent.picker_scratch.iter().all(|b| b.is_empty()));
            for buffer in lent.picker_scratch.iter_mut() {
                buffer.push_for_test();
            }
            assert!(inside(
                std::ptr::from_ref(&*lent.stack).addr(),
                size_of::<SearchStack>()
            ));
            for cell in lent.stack.iter_mut() {
                assert_eq!(cell.static_eval, yorkie_storage::VALUE_NONE);
                assert!(cell.pv.is_empty() && cell.pv.capacity() > MAX_PLY as usize);
                cell.static_eval = 5;
            }
            assert!(inside(
                std::ptr::from_ref(&*lent.finny).addr(),
                size_of::<FinnyCache>()
            ));
        }
    }

    /// A box over the block frees nothing when dropped, and a heap box from
    /// the same allocator frees normally.
    #[test]
    fn a_block_box_is_not_freed_and_a_heap_box_is() {
        let heap = WorkerAlloc::zeroed::<[i16; 64]>();
        assert_eq!(block_of(NonNull::from(&*heap).cast()), None);
        drop(heap);
    }

    /// A box over the block that outlives the claim keeps the block held, so
    /// no second claim hands out boxes over the same memory while it lives;
    /// once the last box goes, the block can be claimed again.
    #[test]
    fn a_box_that_outlives_its_claim_keeps_the_block_held() {
        let _lock = claim_lock();
        let (mut claim, tables) = BlockClaim::claim(0).expect("block 0 is free");
        // SAFETY: no earlier lent table is left: this is the first lending.
        let lent = unsafe { claim.lend_search_tables() };
        let BlockTables {
            capture,
            low_ply,
            main,
            continuation_correction,
            continuation,
        } = tables;
        drop((capture, low_ply, main, continuation_correction));
        drop(claim);
        assert!(
            BlockClaim::claim(0).is_none(),
            "a carved table and the lent tables still hold block 0"
        );
        drop(lent);
        assert!(
            BlockClaim::claim(0).is_none(),
            "the continuation table still holds block 0"
        );
        let addr = std::ptr::from_ref(&*continuation).addr();
        drop(continuation);
        let (again, _tables) = BlockClaim::claim(0).expect("nothing holds block 0");
        let (block, len) = again.backing_region();
        assert!(addr >= block && addr < block + len);
    }
}
