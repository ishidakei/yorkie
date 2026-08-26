//! The process-wide global allocator: [mimalloc].
//!
//! # Why an allocator swap at all
//!
//! The search is allocation-light in its inner loop by construction, but it is
//! not allocation-free: every worker thread churns small, short-lived blocks
//! (root move lists, PV buffers, the per-thread scratch the move picker hands
//! around), and glibc's `malloc` serialises those on a small set of arenas. The
//! cost is invisible at one thread and grows with the worker count. The
//! previous, apery_rust-based generation of this engine measured **+1.69% NPS
//! at 64 threads** from exactly this change (A/B, node-count-fixed bench on one
//! machine, median of alternating runs), with no measurable single-thread
//! effect. mimalloc's per-thread heaps are what buy that back.
//!
//! Defaults only. No `secure` feature (it adds guard pages and randomised
//! freelists — a hardening trade this engine does not need and pays for in the
//! hot path), and no option tuning (huge-page or purge settings): the tuning
//! surface is deliberately left alone so the measurement above stays
//! attributable to the allocator itself.
//!
//! # Why the declaration lives in *this* crate
//!
//! A `#[global_allocator]` is a whole-program property: exactly one may exist
//! in a linked crate graph, and it applies to every binary that links the crate
//! declaring it. Placing it in the `yorkie` binary root would cover the
//! tournament binary and nothing else — every test binary in the workspace
//! would keep running on the system allocator, including the large-page
//! alignment tests in [`crate::large_page`], which are precisely the ones that
//! have to hold *under mimalloc* (see below). Storage is instead the lowest
//! layer the allocation-heavy crates share: `yorkie` → `yorkie-protocol` →
//! `yorkie-storage`, and `yorkie-eval` / `yorkie-search` / `yorkie-protocol`
//! all depend on it directly. Declaring it here therefore covers the tournament
//! binary *and* the test binaries of every crate that allocates through the
//! large-page path, with no new dependency edges and no violation of the
//! layering rule in this crate's manifest (mimalloc is a utility crate, not a
//! Yorkie layer).
//!
//! The remaining members — `yorkie-state`, `yorkie-numa`, `xtask` — are leaves
//! that do not depend on Storage, so their test binaries keep the system
//! allocator. That is deliberate: adding a Storage edge to a leaf purely to
//! install an allocator would invert the layering for no measurable gain, and
//! none of the three is on the allocation-hot path.
//!
//! # Interaction with the large-page allocator
//!
//! [`crate::large_page::alloc_zeroed_large`] requests a 2 MiB-aligned,
//! zero-initialised block through `std::alloc::alloc_zeroed` and then
//! `madvise(MADV_HUGEPAGE)`-hints it. With mimalloc installed those requests
//! route through mimalloc rather than glibc. Nothing about that path changes:
//! the [`GlobalAlloc`](std::alloc::GlobalAlloc) contract requires the returned
//! pointer to satisfy the requested alignment, mimalloc honours it (over its
//! page size it falls through to an aligned-allocation path), and the `madvise`
//! call is made on the pointer the allocator returned, whichever allocator that
//! is. The alignment guarantee is asserted directly — by this module's own
//! `global_allocator_honours_large_page_alignment`, and by
//! [`crate::large_page`]'s existing alignment tests, all of which, because of
//! the placement decision above, execute against mimalloc.
//!
//! On peak RSS, mimalloc cuts both ways: it reserves arena regions up front and
//! returns freed memory to the OS lazily (which inflates RSS), but it can also
//! satisfy a zeroed request with fresh, already-zero OS pages, where
//! `std::alloc::System` must `posix_memalign` and then memset an over-aligned
//! block — touching every page of it. The large-page path issues exactly that
//! kind of request, for the biggest blocks in the process, so on this engine
//! the second effect dominates: a one-thread `bench 64 1 200000 default nodes`
//! measured **1525 MiB peak RSS on glibc and 621 MiB on mimalloc**. Do not read
//! that as a universal property — it is what this allocation mix produces, and
//! the lazy-purge direction can still show up on other workloads. Either way it
//! is allocator policy, not a leak.
//!
//! # The miri carve-out
//!
//! mimalloc is a C library reached through FFI. miri interprets Rust MIR and
//! cannot execute the foreign code behind `mi_malloc`, so a `#[global_allocator]`
//! bound to it would abort every miri test in every crate that links Storage.
//! `#[cfg(not(miri))]` drops the declaration under miri only, leaving the
//! standard allocator in place there. It gates the *static*, not any test or
//! `cfg`-dependent behaviour, so the set of tests miri executes is unchanged —
//! the same carve-out shape [`crate::large_page`] already uses for its
//! `madvise` hint.

/// The process-wide allocator.
///
/// Absent under miri (see the module docs); miri then uses the standard
/// allocator, which it can model.
#[cfg(not(miri))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg(test)]
mod tests {
    use std::alloc::Layout;

    use crate::large_page::LARGE_PAGE_ALIGN;

    /// The over-aligned request the large-page path makes must come back
    /// correctly aligned from *whatever* allocator is installed — the
    /// `GlobalAlloc` contract, exercised here through the global `alloc_zeroed`
    /// shim (i.e. mimalloc in a normal run, the standard allocator under miri).
    ///
    /// `miri, ignore`: the sizes below are whole 2 MiB regions and the zero
    /// check walks every byte, which miri interprets one access at a time.
    /// `large_page`'s own small-length alignment tests cover the same contract
    /// under miri.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn global_allocator_honours_large_page_alignment() {
        for size in [LARGE_PAGE_ALIGN, 2 * LARGE_PAGE_ALIGN, 8 * LARGE_PAGE_ALIGN] {
            let layout = Layout::from_size_align(size, LARGE_PAGE_ALIGN).unwrap();
            // SAFETY: `layout` has non-zero size; the block is freed exactly
            // once below with the same layout.
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
