//! Shared huge-page-backed allocator, a port of the reference's
//! `aligned_large_pages_alloc` / `make_unique_large_page`.
//!
//! The history tables and the loaded NNUE parameters allocate through here, so
//! the huge-page policy lives in one place: base alignment
//! [`LARGE_PAGE_ALIGN`], the byte size rounded **up** to a whole multiple of it,
//! and — on Linux — `MADV_HUGEPAGE`-hinted best-effort across the whole rounded
//! region. [`advise_huge_pages`] is that last step on its own, for a region this
//! module did not allocate: the transposition table's `static`, which the
//! linker placed.
//!
//! [`LargePageAlloc`] is that policy as an [`Allocator`], so the blocks are
//! plain `Box<[T], LargePageAlloc>` and `Box<T, LargePageAlloc>`.
//! [`LargePageAlloc::zeroed_slice`] and [`LargePageAlloc::zeroed`] build them
//! zero-initialised, so the element type must have a valid all-zero bit
//! pattern — the [`Zeroable`] marker.

use std::alloc::{AllocError, Allocator, Global, Layout, handle_alloc_error};
use std::ptr::NonNull;

/// Base alignment of a large-page allocation: a 2 MiB huge-page boundary on
/// Linux so a `MADV_HUGEPAGE` hint can back the region with transparent huge
/// pages, and a plain page boundary elsewhere.
#[cfg(target_os = "linux")]
pub const LARGE_PAGE_ALIGN: usize = 2 * 1024 * 1024;
/// See the Linux definition; 4 KiB elsewhere.
#[cfg(not(target_os = "linux"))]
pub const LARGE_PAGE_ALIGN: usize = 4096;

/// Types whose all-zero bit pattern is a valid, fully initialised value, so a
/// zeroed allocation is a valid `Self`.
///
/// # Safety
///
/// An all-zero byte pattern of `size_of::<Self>()` bytes must be a valid value
/// of `Self`.
pub unsafe trait Zeroable {}

// SAFETY: `0` is a valid value of each of these integer types.
unsafe impl Zeroable for i8 {}
unsafe impl Zeroable for i16 {}
unsafe impl Zeroable for i32 {}
unsafe impl Zeroable for u8 {}

// SAFETY: `AtomicI16` has the same layout as `i16`, and its all-zero pattern is
// the integer `0`.
unsafe impl Zeroable for core::sync::atomic::AtomicI16 {}

// SAFETY: an array is valid when every element is, and an all-zero array is an
// array of all-zero (valid) elements.
unsafe impl<T: Zeroable, const N: usize> Zeroable for [T; N] {}

/// Round `min_bytes` up to a whole multiple of [`LARGE_PAGE_ALIGN`].
pub(crate) const fn rounded_size(min_bytes: usize) -> usize {
    min_bytes.next_multiple_of(LARGE_PAGE_ALIGN)
}

/// Ask the kernel to back the byte range `[addr, addr + len)` with transparent
/// huge pages (`madvise(MADV_HUGEPAGE)`).
///
/// The hint every block [`LargePageAlloc`] hands out carries, and the one a
/// caller applies to a region it did not allocate here — the transposition
/// table's `static`, whose address the linker fixed. `addr` must be
/// page-aligned, which both of those are, being [`LARGE_PAGE_ALIGN`]-aligned.
///
/// Returns whether the kernel accepted it. `false` — off Linux, or where the
/// kernel was built or configured without transparent huge pages — leaves the
/// range with whatever backing it had, which costs TLB misses and nothing else.
///
/// No memory is read or written: the range is handed to the kernel as a
/// descriptor, which is why this is safe despite taking a raw address.
pub fn advise_huge_pages(addr: usize, len: usize) -> bool {
    // miri rejects `madvise` with any advice beyond MADV_NORMAL / RANDOM /
    // SEQUENTIAL / WILLNEED, which would abort every miri test that allocates
    // here. The hint has no observable effect on the range, so dropping it
    // under miri still leaves the allocation, the aliasing and the drop path
    // covered.
    #[cfg(all(target_os = "linux", not(miri)))]
    {
        // SAFETY: `madvise` only adjusts kernel paging policy for the range —
        // it neither reads nor writes the memory and cannot invalidate it, so
        // an address it rejects (unmapped, unaligned) is an error return rather
        // than an unsound access.
        let rc = unsafe { libc::madvise(addr as *mut libc::c_void, len, libc::MADV_HUGEPAGE) };
        rc == 0
    }
    #[cfg(not(all(target_os = "linux", not(miri))))]
    {
        let _ = (addr, len);
        false
    }
}

/// An [`Allocator`] that hands out blocks under the huge-page policy: the
/// request's alignment raised to [`LARGE_PAGE_ALIGN`], its size rounded up to a
/// whole multiple of it, and the whole block `MADV_HUGEPAGE`-hinted. The
/// memory itself comes from [`Global`], so a zeroed request reaches the global
/// allocator's own zeroed path, which can hand out fresh OS pages without
/// writing them.
///
/// [`deallocate`](Allocator::deallocate) receives the caller's layout, not the
/// block's, so it applies the same raise and round-up again before returning
/// the block to [`Global`]. Both are pure functions of the layout, which is
/// what makes the two agree; a caller that frees with any size the block fits
/// (from its request up to the rounded length it was handed) lands on the same
/// block layout, since the rounding of a whole multiple is itself.
#[derive(Clone, Copy, Debug, Default)]
pub struct LargePageAlloc;

impl LargePageAlloc {
    /// The layout of the block that serves a `layout` request.
    fn block_layout(layout: Layout) -> Result<Layout, AllocError> {
        let raised = layout.align_to(LARGE_PAGE_ALIGN).map_err(|_| AllocError)?;
        let size = raised
            .size()
            .checked_next_multiple_of(LARGE_PAGE_ALIGN)
            .ok_or(AllocError)?;
        Layout::from_size_align(size, raised.align()).map_err(|_| AllocError)
    }

    /// Hint a block [`Global`] just handed out, and pass it on.
    fn advised(block: Result<NonNull<[u8]>, AllocError>) -> Result<NonNull<[u8]>, AllocError> {
        let block = block?;
        if !block.is_empty() {
            // Best-effort, as the reference's is: a refused hint leaves the
            // block exactly as it was.
            advise_huge_pages(block.as_ptr().addr(), block.len());
        }
        Ok(block)
    }

    /// A zeroed block for `layout`, aborting through [`handle_alloc_error`] if
    /// the allocation fails.
    fn zeroed_block(layout: Layout) -> NonNull<u8> {
        Self.allocate_zeroed(layout)
            .unwrap_or_else(|_| handle_alloc_error(layout))
            .cast()
    }

    /// A zeroed `T`, its base on a [`LARGE_PAGE_ALIGN`] boundary.
    ///
    /// The bytes are zero because the allocator returned them zeroed; nothing
    /// writes them here, so no page is touched until the caller first writes
    /// one.
    pub fn zeroed<T: Zeroable>() -> Box<T, Self> {
        let ptr = Self::zeroed_block(Layout::new::<T>()).cast::<T>();
        // SAFETY: `ptr` was allocated by `LargePageAlloc` with
        // `Layout::new::<T>()`, the layout a `Box<T>` frees with, so it is
        // aligned for `T` and owned by nothing else. Its bytes are zero, which
        // `T: Zeroable` makes a valid `T`.
        unsafe { Box::from_non_null_in(ptr, Self) }
    }

    /// A zeroed `[T]` of `len` elements, its base on a [`LARGE_PAGE_ALIGN`]
    /// boundary when `len * size_of::<T>()` is non-zero.
    ///
    /// The bytes are zero because the allocator returned them zeroed; nothing
    /// writes them here, so no page is touched until the caller first writes
    /// one.
    pub fn zeroed_slice<T: Zeroable>(len: usize) -> Box<[T], Self> {
        let layout = Layout::array::<T>(len).expect("large-page array byte size overflows");
        let ptr = NonNull::slice_from_raw_parts(Self::zeroed_block(layout).cast::<T>(), len);
        // SAFETY: `ptr` was allocated by `LargePageAlloc` with
        // `Layout::array::<T>(len)`, the layout a `Box<[T]>` of `len` elements
        // frees with, so it is aligned for `T` and owned by nothing else. Its
        // `len` elements are all-zero bytes, which `T: Zeroable` makes valid.
        unsafe { Box::from_non_null_in(ptr, Self) }
    }

    /// The `(address, byte length)` of the block behind `boxed`, or `None`
    /// when the value is zero-sized and so occupies no block.
    ///
    /// The length is the *rounded* block size, not the value's size, so a NUMA
    /// placement call names the block whole and does not spill policy onto a
    /// neighbouring allocation. The address is a `usize` because the consumer
    /// hands it to the kernel as a range descriptor and never dereferences it.
    #[expect(
        clippy::borrowed_box,
        reason = "only a box from this allocator names a block it rounded"
    )]
    pub fn backing_region<T: ?Sized>(boxed: &Box<T, Self>) -> Option<(usize, usize)> {
        let bytes = size_of_val::<T>(boxed);
        (bytes != 0).then(|| (Box::as_ptr(boxed).addr(), rounded_size(bytes)))
    }
}

// SAFETY: every method forwards to `Global` with the layout replaced by
// `block_layout`, a pure function of the caller's layout that maps every size
// the returned block fits to the same block layout, so a block goes back to
// `Global` with exactly the layout it was allocated with. `Global` upholds the
// rest of the contract; the raised alignment and the longer block only
// strengthen what the caller asked for, and the returned slice reports the
// whole block. The huge-page hint neither reads nor writes the block.
unsafe impl Allocator for LargePageAlloc {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Self::advised(Global.allocate(Self::block_layout(layout)?))
    }

    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Self::advised(Global.allocate_zeroed(Self::block_layout(layout)?))
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        let block =
            Self::block_layout(layout).expect("the layout was rounded when it was allocated");
        // SAFETY: the caller guarantees `ptr` came from this allocator with a
        // layout it fits, so `Global` allocated it with `block`.
        unsafe { Global.deallocate(ptr, block) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A request no machine can satisfy comes back as an error rather than a
    /// pointer to write through: the allocator returns `Result`, and the
    /// zeroed constructors turn the error into `handle_alloc_error`, which ends
    /// the process before anything is written.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn a_failed_allocation_is_an_error_not_a_pointer() {
        let layout = Layout::from_size_align(1 << 62, 64).expect("a valid layout");
        assert!(LargePageAlloc.allocate_zeroed(layout).is_err());
        assert!(LargePageAlloc.allocate(layout).is_err());
    }

    #[test]
    fn rounded_size_rounds_up_to_alignment() {
        assert_eq!(rounded_size(1), LARGE_PAGE_ALIGN);
        assert_eq!(rounded_size(LARGE_PAGE_ALIGN - 1), LARGE_PAGE_ALIGN);
        assert_eq!(rounded_size(LARGE_PAGE_ALIGN), LARGE_PAGE_ALIGN);
        assert_eq!(rounded_size(LARGE_PAGE_ALIGN + 1), 2 * LARGE_PAGE_ALIGN);
        assert_eq!(rounded_size(2 * LARGE_PAGE_ALIGN), 2 * LARGE_PAGE_ALIGN);
    }

    #[test]
    fn a_block_is_rounded_and_frees_with_any_layout_it_fits() {
        // Under miri a free whose layout differs from the allocation's is
        // reported, so this also pins that `deallocate` rounds as `allocate`
        // did.
        let request = Layout::from_size_align(7, 2).unwrap();
        for free_size in [request.size(), LARGE_PAGE_ALIGN] {
            let block = LargePageAlloc.allocate_zeroed(request).expect("allocation");
            assert_eq!(
                block.len(),
                LARGE_PAGE_ALIGN,
                "the block is the rounded size"
            );
            assert_eq!(block.as_ptr().addr() % LARGE_PAGE_ALIGN, 0);
            let free = Layout::from_size_align(free_size, request.align()).unwrap();
            // SAFETY: `block` came from `LargePageAlloc` for `request`, `free`
            // has its alignment and a size the block fits, and it is freed
            // once.
            unsafe { LargePageAlloc.deallocate(block.cast(), free) };
        }
    }

    #[test]
    fn block_layout_matches_the_rounding_the_regions_report() {
        for size in [
            0,
            1,
            7,
            LARGE_PAGE_ALIGN - 1,
            LARGE_PAGE_ALIGN,
            LARGE_PAGE_ALIGN + 1,
        ] {
            let block = LargePageAlloc::block_layout(Layout::from_size_align(size, 8).unwrap())
                .expect("small layouts round");
            assert_eq!(block.size(), rounded_size(size));
            assert_eq!(block.align(), LARGE_PAGE_ALIGN);
        }
        let over = Layout::from_size_align(1, 2 * LARGE_PAGE_ALIGN).unwrap();
        assert_eq!(
            LargePageAlloc::block_layout(over).unwrap().align(),
            2 * LARGE_PAGE_ALIGN,
            "a stricter alignment than the policy's is kept",
        );
        assert!(
            LargePageAlloc::block_layout(Layout::from_size_align(usize::MAX / 2, 1).unwrap())
                .is_err()
        );
    }

    // Ignored under miri: the largest lengths run to millions of elements, and
    // walking them all takes minutes there. The small-length tests below cover
    // the same alignment and round-up logic.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn array_is_aligned_zeroed_and_correct_length() {
        // A length that does not divide the alignment, and one spanning
        // several alignment units.
        for &len in &[1usize, 7, 4096, LARGE_PAGE_ALIGN / 2 + 3, LARGE_PAGE_ALIGN] {
            let buf = LargePageAlloc::zeroed_slice::<i16>(len);
            assert_eq!(buf.len(), len);
            assert_eq!(
                buf.as_ptr().addr() % LARGE_PAGE_ALIGN,
                0,
                "base pointer not {LARGE_PAGE_ALIGN}-aligned for len {len}",
            );
            assert!(
                buf.iter().all(|&x| x == 0),
                "buffer not zeroed for len {len}"
            );
        }
    }

    #[test]
    fn empty_array_is_valid_and_aligned() {
        let buf = LargePageAlloc::zeroed_slice::<i16>(0);
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
        assert_eq!(buf.as_ptr().addr() % align_of::<i16>(), 0);
    }

    #[test]
    fn array_is_mutable_in_place() {
        let mut buf = LargePageAlloc::zeroed_slice::<i32>(64);
        for (i, slot) in buf.iter_mut().enumerate() {
            *slot = i as i32;
        }
        assert_eq!(buf[63], 63);
        assert_eq!(buf.as_ptr().addr() % LARGE_PAGE_ALIGN, 0);
    }

    #[test]
    fn box_is_aligned_and_zeroed() {
        let boxed = LargePageAlloc::zeroed::<[[i16; 4]; 2]>();
        assert_eq!(
            std::ptr::from_ref(&*boxed).addr() % LARGE_PAGE_ALIGN,
            0,
            "box base pointer not {LARGE_PAGE_ALIGN}-aligned",
        );
        assert_eq!(*boxed, [[0i16; 4]; 2]);
    }

    #[test]
    fn backing_region_names_the_whole_rounded_block() {
        // A length that does not fill a whole alignment unit, so the region
        // must report the rounded allocation rather than the element bytes.
        let buf = LargePageAlloc::zeroed_slice::<i16>(7);
        let (addr, bytes) =
            LargePageAlloc::backing_region(&buf).expect("a non-empty array owns a block");
        assert_eq!(addr, buf.as_ptr().addr());
        assert_eq!(addr % LARGE_PAGE_ALIGN, 0, "base must be page-aligned");
        assert_eq!(bytes, rounded_size(7 * size_of::<i16>()));
        assert_eq!(bytes, LARGE_PAGE_ALIGN);

        // An empty array allocates nothing, so there is no region to place.
        assert_eq!(
            LargePageAlloc::backing_region(&LargePageAlloc::zeroed_slice::<i16>(0)),
            None
        );
    }

    #[test]
    fn box_backing_region_names_the_whole_rounded_block() {
        let boxed = LargePageAlloc::zeroed::<[[i16; 4]; 2]>();
        let (addr, bytes) =
            LargePageAlloc::backing_region(&boxed).expect("a non-ZST box owns a block");
        assert_eq!(addr, std::ptr::from_ref(&*boxed).addr());
        assert_eq!(addr % LARGE_PAGE_ALIGN, 0, "base must be page-aligned");
        assert_eq!(bytes, rounded_size(size_of::<[[i16; 4]; 2]>()));
    }

    #[test]
    fn box_is_mutable_and_keeps_fixed_dimensions() {
        let mut boxed = LargePageAlloc::zeroed::<[[i16; 4]; 2]>();
        boxed[1][3] = 42;
        assert_eq!(boxed[1][3], 42);
        assert_eq!(boxed[0][0], 0);
    }

    /// A zeroed block comes zeroed from the allocator, not filled by the
    /// constructing thread: a fill would fault every page in on that thread
    /// and fix which NUMA node it lands on before the block's user ever
    /// touches it. `mincore` reports which of the block's pages are resident;
    /// with mimalloc installed only the first [`LARGE_PAGE_ALIGN`] bytes are,
    /// exactly as for a direct `std::alloc::alloc_zeroed` of the same layout.
    ///
    /// Linux only, and not under miri, which cannot call `mincore`.
    #[cfg(all(target_os = "linux", not(miri)))]
    #[test]
    fn a_zeroed_block_is_not_touched_by_its_construction() {
        const BYTES: usize = 64 * LARGE_PAGE_ALIGN;
        let buf = LargePageAlloc::zeroed_slice::<u8>(BYTES);
        let page = 4096;
        let mut resident = vec![0u8; BYTES / page];
        // SAFETY: the range is the block `buf` owns, page-aligned, and
        // `resident` holds one byte per 4 KiB page of it; `mincore` only writes
        // that vector.
        let rc =
            unsafe { libc::mincore(buf.as_ptr().cast_mut().cast(), BYTES, resident.as_mut_ptr()) };
        assert_eq!(rc, 0, "mincore failed");
        // The global allocator may write the block's first word itself, and a
        // huge page then backs that whole first unit; a fill would show past
        // it.
        let past_first_unit = &resident[LARGE_PAGE_ALIGN / page..];
        let touched = past_first_unit.iter().filter(|&&r| r & 1 != 0).count();
        assert_eq!(
            touched, 0,
            "{touched} of the block's pages past its first {LARGE_PAGE_ALIGN} bytes are resident \
             before the first write"
        );
        drop(buf);
    }
}
