//! The process-wide global allocator: [mimalloc].
//!
//! The search is allocation-light in its inner loop but not allocation-free —
//! every worker churns small, short-lived blocks — and glibc's `malloc`
//! serialises those on a small set of arenas, at a cost that grows with the
//! worker count. mimalloc's per-thread heaps buy that back. Defaults only: no
//! `secure` feature, whose guard pages and randomised freelists are a hardening
//! trade this engine does not need, and no option tuning.
//!
//! A `#[global_allocator]` is a whole-program property: exactly one may exist
//! in a linked crate graph, and it applies to every binary linking the crate
//! that declares it. Declaring it here rather than in the engine binary is what
//! puts the test binaries on mimalloc too — including
//! [`crate::large_page`]'s alignment tests, which are precisely the ones that
//! must hold under it. The crates that do not depend on this one keep the
//! system allocator; none is on the allocation-hot path.
//!
//! [`crate::large_page`]'s `alloc_zeroed_large` requests an over-aligned, zeroed
//! block through `std::alloc::alloc_zeroed`, and the
//! [`GlobalAlloc`](std::alloc::GlobalAlloc) contract requires the returned
//! pointer to satisfy the requested alignment whichever allocator is installed.
//! `global_allocator_honours_large_page_alignment` asserts it directly.
//!
//! On peak RSS mimalloc cuts both ways: it returns freed memory to the OS
//! lazily, which inflates RSS, but it can also satisfy a zeroed request with
//! fresh, already-zero OS pages where the system allocator must over-align and
//! then memset — touching every page. The large-page path issues exactly that
//! kind of request for the biggest blocks in the process, so here the second
//! effect dominates. Neither direction is a leak; both are allocator policy.
//!
//! mimalloc is a C library reached through FFI, which miri cannot execute, so a
//! `#[global_allocator]` bound to it would abort every miri test in every crate
//! linking this one. `#[cfg(not(miri))]` drops the declaration there, leaving
//! the standard allocator in place. It gates the *static* only, so the set of
//! tests miri executes is unchanged.
//!
//! With `verbose1` the installed allocator is `CountingAlloc` wrapping
//! mimalloc, which tallies the blocks the process is handed. The design rule is
//! that a game allocates nothing on the heap — everything is taken at
//! initialisation and reused — and the tally is how far the engine still is from
//! that, measured per reply. Without the feature mimalloc is installed directly:
//! no wrapper, no counter, no branch on the allocation path.

#[cfg(feature = "verbose1")]
use std::sync::atomic::{AtomicU64, Ordering};

/// The process-wide allocation tally the installed [`CountingAlloc`] raises.
#[cfg(feature = "verbose1")]
static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);

/// The process-wide allocator.
#[cfg(all(not(miri), not(feature = "verbose1")))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// The process-wide allocator, counting the blocks it hands out.
#[cfg(all(not(miri), feature = "verbose1"))]
#[global_allocator]
static GLOBAL: CountingAlloc<mimalloc::MiMalloc> =
    CountingAlloc::new(mimalloc::MiMalloc, &ALLOC_COUNT);

/// An allocator that raises a counter for every block it hands out and forwards
/// every call to the allocator it wraps.
///
/// Three of the four [`GlobalAlloc`](std::alloc::GlobalAlloc) entry points
/// produce a block and are counted: `alloc`, `alloc_zeroed` and `realloc`.
/// `dealloc` is not — what the tally answers is how many blocks were taken, and
/// a free says nothing about that. `alloc_zeroed` is forwarded rather than left
/// to the trait's default (which would allocate and then memset), so the inner
/// allocator's fresh-zero-page path stays intact; that path is what the
/// over-aligned [`crate::large_page`] requests, the biggest in the process,
/// depend on.
///
/// The counter is borrowed rather than fixed, so a test can weigh a wrapper of
/// its own against a counter nothing else touches.
#[cfg(feature = "verbose1")]
pub struct CountingAlloc<A> {
    inner: A,
    count: &'static AtomicU64,
}

#[cfg(feature = "verbose1")]
impl<A> CountingAlloc<A> {
    pub const fn new(inner: A, count: &'static AtomicU64) -> Self {
        Self { inner, count }
    }
}

// SAFETY: every method forwards to `inner`, whose own `GlobalAlloc` impl upholds
// the contract; the wrapper returns exactly what it returns and passes each
// pointer and layout through unchanged. The added counter increment is a relaxed
// read-modify-write on an unrelated atomic, which allocates nothing and so
// cannot re-enter the allocator.
#[cfg(feature = "verbose1")]
unsafe impl<A: std::alloc::GlobalAlloc> std::alloc::GlobalAlloc for CountingAlloc<A> {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        self.count.fetch_add(1, Ordering::Relaxed);
        // SAFETY: `layout` is the caller's, forwarded unchanged.
        unsafe { self.inner.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8 {
        self.count.fetch_add(1, Ordering::Relaxed);
        // SAFETY: `layout` is the caller's, forwarded unchanged.
        unsafe { self.inner.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        // SAFETY: `ptr` was handed out by `inner` under `layout`, since every
        // allocating method forwards to it.
        unsafe { self.inner.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, new_size: usize) -> *mut u8 {
        self.count.fetch_add(1, Ordering::Relaxed);
        // SAFETY: `ptr` was handed out by `inner` under `layout`, and `new_size`
        // is the caller's, forwarded unchanged.
        unsafe { self.inner.realloc(ptr, layout, new_size) }
    }
}

/// Read the allocation tally and reset it to zero in one step, so the value
/// returned covers exactly the interval since the previous take (or the last
/// [`clear_alloc_count`]) and nothing spills into the next one.
#[cfg(feature = "verbose1")]
pub fn take_alloc_count() -> u64 {
    ALLOC_COUNT.swap(0, Ordering::Relaxed)
}

/// Drop the allocation tally, so what came before is attributed to no interval.
#[cfg(feature = "verbose1")]
pub fn clear_alloc_count() {
    ALLOC_COUNT.store(0, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use std::alloc::Layout;

    use crate::large_page::LARGE_PAGE_ALIGN;

    /// Ignored under miri: the sizes below are whole 2 MiB regions and the zero
    /// check walks every byte, which miri interprets one access at a time.
    /// `large_page`'s small-length tests cover the same contract there.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn global_allocator_honours_large_page_alignment() {
        for size in [LARGE_PAGE_ALIGN, 2 * LARGE_PAGE_ALIGN, 8 * LARGE_PAGE_ALIGN] {
            let layout = Layout::from_size_align(size, LARGE_PAGE_ALIGN).unwrap();
            // SAFETY: `layout` has non-zero size, and the block is freed
            // exactly once below with the same layout.
            let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
            assert!(!ptr.is_null(), "allocation of {size} bytes failed");
            assert_eq!(
                ptr as usize % LARGE_PAGE_ALIGN,
                0,
                "global allocator returned a pointer not {LARGE_PAGE_ALIGN}-aligned for {size} bytes",
            );
            // SAFETY: `ptr` addresses `size` readable bytes just returned by
            // `alloc_zeroed`.
            let bytes = unsafe { std::slice::from_raw_parts(ptr, size) };
            assert!(bytes.iter().all(|&b| b == 0), "block not zeroed");
            // SAFETY: `ptr` came from `alloc_zeroed` with exactly `layout`.
            unsafe { std::alloc::dealloc(ptr, layout) };
        }
    }

    /// Weighed against a counter of its own, over the system allocator, so the
    /// tally is exactly what this test asked for: the process-wide counter moves
    /// under every other test in the binary, and mimalloc is FFI that miri cannot
    /// run.
    #[cfg(feature = "verbose1")]
    #[test]
    fn the_counting_allocator_counts_each_block_handed_out_and_no_free() {
        use std::alloc::{GlobalAlloc, System};
        use std::sync::atomic::{AtomicU64, Ordering};

        use super::CountingAlloc;

        static COUNT: AtomicU64 = AtomicU64::new(0);
        let alloc = CountingAlloc::new(System, &COUNT);
        let small = Layout::from_size_align(64, 8).unwrap();
        let grown = Layout::from_size_align(128, 8).unwrap();

        // SAFETY: `small` has non-zero size.
        let ptr = unsafe { alloc.alloc(small) };
        assert!(!ptr.is_null(), "alloc failed");
        assert_eq!(COUNT.load(Ordering::Relaxed), 1, "alloc must count");

        // SAFETY: `ptr` came from this allocator under `small`, and the new size
        // is non-zero and rounds up to no more than `isize::MAX`.
        let ptr = unsafe { alloc.realloc(ptr, small, grown.size()) };
        assert!(!ptr.is_null(), "realloc failed");
        assert_eq!(COUNT.load(Ordering::Relaxed), 2, "realloc must count");

        // SAFETY: `ptr` came from `realloc` and so is held under `grown`.
        unsafe { alloc.dealloc(ptr, grown) };
        assert_eq!(
            COUNT.load(Ordering::Relaxed),
            2,
            "a free hands out no block and must not count"
        );

        // SAFETY: `small` has non-zero size.
        let zeroed = unsafe { alloc.alloc_zeroed(small) };
        assert!(!zeroed.is_null(), "alloc_zeroed failed");
        assert_eq!(
            COUNT.load(Ordering::Relaxed),
            3,
            "alloc_zeroed hands out a block and must count"
        );
        // SAFETY: `zeroed` came from this allocator under exactly `small`.
        unsafe { alloc.dealloc(zeroed, small) };
        assert_eq!(COUNT.load(Ordering::Relaxed), 3);
    }
}
