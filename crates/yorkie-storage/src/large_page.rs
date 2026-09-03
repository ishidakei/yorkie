//! Shared huge-page-backed allocator, a port of the reference's
//! `aligned_large_pages_alloc` / `make_unique_large_page` (`memory.cpp`).
//!
//! The transposition table, the history tables and the loaded NNUE parameters
//! all allocate through here, so the huge-page policy lives in one place: base
//! alignment [`LARGE_PAGE_ALIGN`], the byte size rounded **up** to a whole
//! multiple of it, zero-initialised, and — on Linux — `MADV_HUGEPAGE`-hinted
//! best-effort across the whole rounded region.
//!
//! [`LargePageArray<T>`] and [`LargePageBox<T>`] are the `Box<[T]>` and `Box<T>`
//! analogues on top of it. Both construct their storage zeroed, so the element
//! type must have a valid all-zero bit pattern — the [`Zeroable`] marker.

use std::alloc::{Layout, alloc_zeroed, dealloc, handle_alloc_error};
use std::fmt;
use std::mem::{align_of, size_of};
use std::ops::{Deref, DerefMut};
use std::ptr::{self, NonNull};
use std::slice;

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
/// of `Self`, and `Self` must need no drop glue: the containers below free the
/// backing bytes directly and never run element destructors.
pub unsafe trait Zeroable {}

// SAFETY: `0` is a valid value of each of these integer types, and all are
// trivially destructible.
unsafe impl Zeroable for i8 {}
unsafe impl Zeroable for i16 {}
unsafe impl Zeroable for i32 {}
unsafe impl Zeroable for u8 {}

// SAFETY: `AtomicI16` has the same layout as `i16`, its all-zero pattern is the
// integer `0`, and it has no `Drop` glue.
unsafe impl Zeroable for core::sync::atomic::AtomicI16 {}

// SAFETY: an array is valid when every element is, and an all-zero array is an
// array of all-zero (valid) elements; an array needs no drop glue when its
// elements need none.
unsafe impl<T: Zeroable, const N: usize> Zeroable for [T; N] {}

/// Round `min_bytes` up to a whole multiple of [`LARGE_PAGE_ALIGN`].
pub(crate) const fn rounded_size(min_bytes: usize) -> usize {
    min_bytes.div_ceil(LARGE_PAGE_ALIGN) * LARGE_PAGE_ALIGN
}

/// Allocate a zeroed block of at least `min_bytes` under the module's
/// huge-page policy. `min_bytes` must be non-zero. The returned [`Layout`] must
/// be replayed to [`free_large`] to free the block.
pub(crate) fn alloc_zeroed_large(min_bytes: usize) -> (NonNull<u8>, Layout) {
    debug_assert!(min_bytes > 0, "large-page allocation must be non-empty");

    let size = rounded_size(min_bytes);
    let layout = Layout::from_size_align(size, LARGE_PAGE_ALIGN)
        .expect("large-page allocation size/alignment is valid");

    // SAFETY: `layout` has non-zero size, since `min_bytes >= 1` rounds up to
    // at least `LARGE_PAGE_ALIGN`.
    let raw = unsafe { alloc_zeroed(layout) };
    let ptr = match NonNull::new(raw) {
        Some(p) => p,
        None => handle_alloc_error(layout),
    };

    // miri rejects `madvise` with any advice beyond MADV_NORMAL / RANDOM /
    // SEQUENTIAL / WILLNEED, which would abort every miri test that allocates
    // here. The hint has no observable effect on the returned block, so
    // dropping it under miri still leaves the allocation, the aliasing and the
    // drop path covered.
    #[cfg(all(target_os = "linux", not(miri)))]
    {
        // SAFETY: `ptr` and `size` describe the live allocation just returned,
        // and `madvise` only adjusts kernel paging policy for that range — it
        // neither reads nor writes the memory and cannot invalidate it. The
        // return value is discarded because the hint is best-effort, as the
        // reference discards it too.
        unsafe {
            libc::madvise(ptr.as_ptr() as *mut libc::c_void, size, libc::MADV_HUGEPAGE);
        }
    }

    (ptr, layout)
}

/// Free a block returned by [`alloc_zeroed_large`].
///
/// # Safety
///
/// `ptr` must have come from [`alloc_zeroed_large`] with exactly `layout`, and
/// must not be freed more than once.
pub(crate) unsafe fn free_large(ptr: NonNull<u8>, layout: Layout) {
    // SAFETY: contract delegated to the caller (see the doc comment).
    unsafe { dealloc(ptr.as_ptr(), layout) }
}

/// A large-page-backed, zero-initialised `[T]` — the analogue of `Box<[T]>`.
/// The exposed slice covers exactly `len` elements; the round-up tail of the
/// allocation is not part of it.
pub struct LargePageArray<T: Zeroable> {
    /// Base of the aligned allocation. Dangling (never dereferenced) when
    /// `len == 0`; always [`LARGE_PAGE_ALIGN`]-aligned otherwise.
    ptr: NonNull<T>,
    /// Number of live elements the slice exposes.
    len: usize,
    /// Exact [`Layout`] the block was allocated with, replayed to [`free_large`]
    /// on drop. Meaningless (zero-sized) when `len == 0`.
    layout: Layout,
}

// SAFETY: `LargePageArray<T>` uniquely owns its heap block of `T`, freed once
// in `Drop`, just like `Box<[T]>`, so it is `Send`/`Sync` exactly when `T` is.
unsafe impl<T: Zeroable + Send> Send for LargePageArray<T> {}
unsafe impl<T: Zeroable + Sync> Sync for LargePageArray<T> {}

impl<T: Zeroable> LargePageArray<T> {
    /// A zeroed array of `len` elements. `len == 0` allocates nothing.
    pub fn zeroed(len: usize) -> Self {
        debug_assert!(
            size_of::<T>() > 0,
            "LargePageArray element must not be a ZST"
        );
        debug_assert!(align_of::<T>() <= LARGE_PAGE_ALIGN);

        if len == 0 {
            return Self {
                ptr: NonNull::dangling(),
                len: 0,
                // Zero-sized, correctly aligned; never handed to `free_large`.
                layout: Layout::from_size_align(0, LARGE_PAGE_ALIGN)
                    .expect("LARGE_PAGE_ALIGN is a valid power-of-two alignment"),
            };
        }

        let bytes = len
            .checked_mul(size_of::<T>())
            .expect("LargePageArray byte size overflows usize");
        let (raw, layout) = alloc_zeroed_large(bytes);
        // `raw` is `LARGE_PAGE_ALIGN`-aligned, hence aligned for `T`.
        Self {
            ptr: raw.cast(),
            len,
            layout,
        }
    }

    /// The stored base pointer, carrying the allocation's original raw,
    /// write-capable provenance rather than a reference reborrow.
    ///
    /// `as_ptr` / `as_mut_ptr` resolve through `Deref` to the slice methods and
    /// so stamp the pointer with a temporary reference's provenance, making a
    /// later write through it undefined behaviour under Stacked or Tree
    /// Borrows. An arena view that carves sub-array pointers must use this.
    pub(crate) fn base_nonnull(&self) -> NonNull<T> {
        self.ptr
    }

    /// The `(address, byte length)` of the backing block, or `None` when the
    /// array is empty and so allocates nothing.
    ///
    /// The length is the *rounded* allocation size, not `len * size_of::<T>()`,
    /// so a NUMA placement call names the block whole and does not spill policy
    /// onto a neighbouring allocation. The address is a `usize` because the
    /// consumer hands it to the kernel as a range descriptor and never
    /// dereferences it.
    pub fn backing_region(&self) -> Option<(usize, usize)> {
        (self.len != 0).then(|| (self.ptr.as_ptr() as usize, self.layout.size()))
    }

    /// Copy `src` into a fresh large-page-backed array of the same length.
    pub fn from_slice(src: &[T]) -> Self
    where
        T: Copy,
    {
        let buf = Self::zeroed(src.len());
        if !src.is_empty() {
            // SAFETY: `buf` holds `src.len()` elements in one allocation,
            // disjoint from `src`, and `T: Copy` makes a bytewise copy valid.
            unsafe {
                ptr::copy_nonoverlapping(src.as_ptr(), buf.ptr.as_ptr(), src.len());
            }
        }
        buf
    }
}

impl<T: Zeroable> Deref for LargePageArray<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        // SAFETY: for `len > 0`, `ptr` addresses `len` contiguous, zeroed `T`
        // within one allocation. For `len == 0` this yields an empty slice, for
        // which `from_raw_parts` accepts any aligned non-null pointer. The
        // borrow is tied to `&self`, so no `&mut` can coexist.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Zeroable> DerefMut for LargePageArray<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: as `deref`, but `&mut self` guarantees exclusivity.
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Zeroable + fmt::Debug> fmt::Debug for LargePageArray<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T: Zeroable> Drop for LargePageArray<T> {
    fn drop(&mut self) {
        if self.len != 0 {
            // SAFETY: `ptr` came from `alloc_zeroed_large` with exactly
            // `layout`, and is freed exactly once because `LargePageArray`
            // uniquely owns the block. `T: Zeroable` needs no drop glue, so
            // freeing the raw bytes is sufficient.
            unsafe {
                free_large(self.ptr.cast(), self.layout);
            }
        }
    }
}

/// A large-page-backed, zero-initialised single `T` — the analogue of `Box<T>`.
/// Keeps a fixed-shape array's compile-time dimensions, so its per-access index
/// maths stay bounds-check-free.
pub struct LargePageBox<T: Zeroable> {
    /// Base of the aligned allocation, holding one initialised `T`.
    ptr: NonNull<T>,
    /// Exact [`Layout`] the block was allocated with, replayed to [`free_large`]
    /// on drop.
    layout: Layout,
}

// SAFETY: `LargePageBox<T>` uniquely owns one heap `T`, freed once in `Drop`,
// just like `Box<T>`, so it is `Send`/`Sync` exactly when `T` is.
unsafe impl<T: Zeroable + Send> Send for LargePageBox<T> {}
unsafe impl<T: Zeroable + Sync> Sync for LargePageBox<T> {}

impl<T: Zeroable> LargePageBox<T> {
    /// A zeroed `T`, its base pointer on a [`LARGE_PAGE_ALIGN`] boundary.
    pub fn zeroed() -> Self {
        debug_assert!(size_of::<T>() > 0, "LargePageBox value must not be a ZST");
        debug_assert!(align_of::<T>() <= LARGE_PAGE_ALIGN);

        let (raw, layout) = alloc_zeroed_large(size_of::<T>());
        Self {
            ptr: raw.cast(),
            layout,
        }
    }

    /// The `(address, byte length)` of the backing block, as
    /// [`LargePageArray::backing_region`]. Always present, since a
    /// [`LargePageBox`] holds one non-ZST `T`.
    pub fn backing_region(&self) -> (usize, usize) {
        (self.ptr.as_ptr() as usize, self.layout.size())
    }
}

impl<T: Zeroable> Deref for LargePageBox<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: `ptr` addresses one zeroed `T` within a live allocation, and
        // the borrow is tied to `&self`, so no `&mut` can coexist.
        unsafe { self.ptr.as_ref() }
    }
}

impl<T: Zeroable> DerefMut for LargePageBox<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: as `deref`, but `&mut self` guarantees exclusivity.
        unsafe { self.ptr.as_mut() }
    }
}

impl<T: Zeroable + fmt::Debug> fmt::Debug for LargePageBox<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T: Zeroable> Drop for LargePageBox<T> {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from `alloc_zeroed_large` with exactly `layout`,
        // and is freed exactly once because `LargePageBox` uniquely owns the
        // block. `T: Zeroable` needs no drop glue.
        unsafe {
            free_large(self.ptr.cast(), self.layout);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounded_size_rounds_up_to_alignment() {
        assert_eq!(rounded_size(1), LARGE_PAGE_ALIGN);
        assert_eq!(rounded_size(LARGE_PAGE_ALIGN - 1), LARGE_PAGE_ALIGN);
        assert_eq!(rounded_size(LARGE_PAGE_ALIGN), LARGE_PAGE_ALIGN);
        assert_eq!(rounded_size(LARGE_PAGE_ALIGN + 1), 2 * LARGE_PAGE_ALIGN);
        assert_eq!(rounded_size(2 * LARGE_PAGE_ALIGN), 2 * LARGE_PAGE_ALIGN);
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
            let buf = LargePageArray::<i16>::zeroed(len);
            assert_eq!(buf.len(), len);
            assert_eq!(
                buf.as_ptr() as usize % LARGE_PAGE_ALIGN,
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
        let buf = LargePageArray::<i16>::zeroed(0);
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
        assert_eq!(buf.as_ptr() as usize % align_of::<i16>(), 0);
    }

    #[test]
    fn array_is_mutable_in_place() {
        let mut buf = LargePageArray::<i32>::zeroed(64);
        for (i, slot) in buf.iter_mut().enumerate() {
            *slot = i as i32;
        }
        assert_eq!(buf[63], 63);
        assert_eq!(buf.as_ptr() as usize % LARGE_PAGE_ALIGN, 0);
    }

    #[test]
    fn from_slice_preserves_bytes_and_aligns() {
        let src: Vec<i16> = (0..1000i32).map(|i| (i - 500) as i16).collect();
        let buf = LargePageArray::<i16>::from_slice(&src);
        assert_eq!(buf.as_ptr() as usize % LARGE_PAGE_ALIGN, 0);
        assert_eq!(&*buf, &src[..]);
    }

    #[test]
    fn box_is_aligned_and_zeroed() {
        let boxed = LargePageBox::<[[i16; 4]; 2]>::zeroed();
        assert_eq!(
            (&*boxed as *const _ as usize) % LARGE_PAGE_ALIGN,
            0,
            "box base pointer not {LARGE_PAGE_ALIGN}-aligned",
        );
        assert_eq!(*boxed, [[0i16; 4]; 2]);
    }

    #[test]
    fn backing_region_names_the_whole_rounded_block() {
        // A length that does not fill a whole alignment unit, so the region
        // must report the rounded allocation rather than the element bytes.
        let buf = LargePageArray::<i16>::zeroed(7);
        let (addr, bytes) = buf
            .backing_region()
            .expect("a non-empty array owns a block");
        assert_eq!(addr, buf.as_ptr() as usize);
        assert_eq!(addr % LARGE_PAGE_ALIGN, 0, "base must be page-aligned");
        assert_eq!(bytes, rounded_size(7 * size_of::<i16>()));
        assert_eq!(bytes, LARGE_PAGE_ALIGN);

        // An empty array allocates nothing, so there is no region to place.
        assert_eq!(LargePageArray::<i16>::zeroed(0).backing_region(), None);
    }

    #[test]
    fn box_backing_region_names_the_whole_rounded_block() {
        let boxed = LargePageBox::<[[i16; 4]; 2]>::zeroed();
        let (addr, bytes) = boxed.backing_region();
        assert_eq!(addr, &*boxed as *const _ as usize);
        assert_eq!(addr % LARGE_PAGE_ALIGN, 0, "base must be page-aligned");
        assert_eq!(bytes, rounded_size(size_of::<[[i16; 4]; 2]>()));
    }

    #[test]
    fn box_is_mutable_and_keeps_fixed_dimensions() {
        let mut boxed = LargePageBox::<[[i16; 4]; 2]>::zeroed();
        boxed[1][3] = 42;
        assert_eq!(boxed[1][3], 42);
        assert_eq!(boxed[0][0], 0);
    }
}
