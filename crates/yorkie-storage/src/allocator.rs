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
//! [`crate::large_page::alloc_zeroed_large`] requests an over-aligned, zeroed
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

/// The process-wide allocator.
#[cfg(not(miri))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

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
}
