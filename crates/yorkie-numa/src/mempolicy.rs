//! Linux NUMA **memory**-policy calls — the placement half of the NUMA work
//! ([`NumaConfig`](crate::NumaConfig) owns the thread-pinning half).
//!
//! The recommended launch line on a many-core host wraps the engine in
//! `numactl --interleave=all`. That process-wide policy is right for the one
//! transposition table every worker shares, but it is inherited by every thread,
//! so a worker's *private* working set is round-robined across all nodes even
//! though it was pinned to one. Under `MPOL_INTERLEAVE` the kernel's first-touch
//! rule does not apply, so allocating on the "right" thread places nothing.
//!
//! The three-layer policy this module makes possible:
//!
//! 1. **Process default — untouched**, so the TT keeps interleaving and the USI
//!    thread keeps the launcher's policy.
//! 2. **Explicit per-region placement** — [`migrate_region_to_node`], for a
//!    block another thread already allocated *and faulted*, so first-touch is no
//!    longer available as a lever.
//! 3. **Worker-scope preference** — [`set_current_thread_preferred_node`], set
//!    by a worker right after it pins itself. Being per-thread, it can never
//!    disturb the TT's interleave or another thread's placement.
//!
//! The kernel's nodemask is indexed by **system** NUMA node, while a
//! [`NumaIndex`] is a *logical* node that L3-aware bundling can renumber, so
//! callers must map through
//! [`NumaConfig::system_node_of_logical`](crate::NumaConfig::system_node_of_logical)
//! before calling anything here.
//!
//! Every syscall wrapper returns a plain `bool` / `Option` and never panics: a
//! kernel that refuses the call simply leaves today's placement in force.
//!
//! Off Linux the syscall wrappers are compiled-out no-ops, and so they are under
//! miri, which models neither `set_mempolicy` nor `mbind`. The mask and
//! page-rounding logic below is platform-independent.

use crate::NumaIndex;

/// Bits carried by one nodemask word. The kernel's `set_mempolicy` / `mbind` /
/// `get_mempolicy` nodemask is an array of `unsigned long`, which on every Linux
/// ABI is exactly pointer-width — hence `usize`.
const MASK_WORD_BITS: usize = usize::BITS as usize;

/// Largest node index this module will build a mask for. The kernel caps
/// `MAX_NUMNODES` at 1024 in its largest configuration and rejects a wider
/// nodemask with `EINVAL`; refusing here keeps a nonsense index from turning
/// into a multi-megabyte mask allocation.
const MAX_NODE_INDEX: NumaIndex = 1023;

/// `MPOL_DEFAULT` — no policy of its own; fall back to the next policy up
/// (`linux/mempolicy.h`).
pub const MODE_DEFAULT: i32 = 0;
/// `MPOL_PREFERRED` — allocate on the named node when it has memory, else fall
/// back to the nearest node that does.
pub const MODE_PREFERRED: i32 = 1;
/// `MPOL_BIND` — allocate only from the named node(s).
pub const MODE_BIND: i32 = 2;
/// `MPOL_INTERLEAVE` — round-robin allocations across the named node(s). This is
/// what `numactl --interleave=all` installs as the process default.
pub const MODE_INTERLEAVE: i32 = 3;
/// `MPOL_LOCAL` — allocate on the node the faulting CPU belongs to.
pub const MODE_LOCAL: i32 = 4;

/// `MPOL_MF_MOVE` — `mbind` should also *migrate* already-faulted pages in the
/// range that do not conform to the new policy. Not defined by the `libc` crate,
/// so it is spelled out here from `linux/mempolicy.h`. `MPOL_MF_STRICT` is
/// deliberately **not** used: a page that cannot be moved (shared, pinned,
/// unmovable) must be left where it is rather than fail the call.
#[cfg(all(target_os = "linux", not(miri)))]
const MPOL_MF_MOVE: libc::c_ulong = 1 << 1;

/// `MPOL_F_ADDR` — `get_mempolicy` should report the policy governing the given
/// address rather than the calling thread's own policy (`linux/mempolicy.h`).
#[cfg(all(target_os = "linux", not(miri)))]
const MPOL_F_ADDR: libc::c_ulong = 1 << 1;

/// A memory policy as reported by `get_mempolicy`: the mode and the node set it
/// names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemPolicy {
    /// The policy mode: [`MODE_DEFAULT`], [`MODE_PREFERRED`], [`MODE_BIND`],
    /// [`MODE_INTERLEAVE`] or [`MODE_LOCAL`].
    pub mode: i32,
    /// The nodes the mode names, ascending. Empty for [`MODE_DEFAULT`] /
    /// [`MODE_LOCAL`].
    pub nodes: Vec<NumaIndex>,
}

/// Build the kernel nodemask naming exactly `node`, plus the `maxnode` argument
/// that goes with it.
///
/// `maxnode` is `words * MASK_WORD_BITS + 1`, the convention libnuma uses: the
/// kernel decrements the argument before deriving both the word count and the
/// mask of valid bits in the final word, so passing the bit count itself would
/// silently drop the mask's top bit — which, for a node on a word boundary, is
/// the only bit set.
///
/// Returns `None` for an index past [`MAX_NODE_INDEX`], which the kernel would
/// reject anyway.
pub(crate) fn node_mask(node: NumaIndex) -> Option<(Vec<usize>, usize)> {
    if node > MAX_NODE_INDEX {
        return None;
    }
    let words = node / MASK_WORD_BITS + 1;
    let mut mask = vec![0usize; words];
    mask[node / MASK_WORD_BITS] = 1usize << (node % MASK_WORD_BITS);
    Some((mask, words * MASK_WORD_BITS + 1))
}

/// Decode a kernel nodemask — the little-endian bit array `node_mask` builds and
/// `get_mempolicy` fills — into ascending node indices.
pub fn nodes_from_mask(mask: &[usize]) -> Vec<NumaIndex> {
    let mut nodes = Vec::new();
    for (w, &word) in mask.iter().enumerate() {
        for bit in 0..MASK_WORD_BITS {
            if word & (1usize << bit) != 0 {
                nodes.push(w * MASK_WORD_BITS + bit);
            }
        }
    }
    nodes
}

/// The page-aligned span covering the byte range `[addr, addr + len)`, given a
/// `page` size (a power of two). Returns `(start, length)`.
///
/// `mbind` rejects a base that is not page-aligned and rounds the length up
/// internally; doing both explicitly here makes visible at the call site that
/// the policy also covers whatever else shares the first and last page.
///
/// Returns `None` for an empty range, or one whose rounding would overflow.
pub(crate) fn page_span(page: usize, addr: usize, len: usize) -> Option<(usize, usize)> {
    debug_assert!(page.is_power_of_two(), "page size must be a power of two");
    if len == 0 {
        return None;
    }
    let start = addr & !(page - 1);
    let end = addr.checked_add(len)?;
    // Round the exclusive end up to the next page boundary.
    let end = end.checked_add(page - 1)? & !(page - 1);
    Some((start, end - start))
}

/// The system page size (`sysconf(_SC_PAGESIZE)`), falling back to 4 KiB when
/// the query fails or reports a nonsensical value. Off Linux this is the 4 KiB
/// constant — nothing there consumes it beyond the pure [`page_span`] tests.
pub fn page_size() -> usize {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: `sysconf` takes an integer name and returns a `long`. It reads
        // no caller memory; an unsupported name yields -1, which the check below
        // rejects in favour of the fallback.
        let raw = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if raw > 0 {
            let size = raw as usize;
            if size.is_power_of_two() {
                return size;
            }
        }
    }
    4096
}

/// Make the **calling thread** prefer `system_node` for every subsequent
/// allocation (`set_mempolicy(MPOL_PREFERRED, {system_node})`).
///
/// `MPOL_PREFERRED` rather than `MPOL_BIND` keeps it a preference: if the node
/// runs out of memory the allocation still succeeds elsewhere instead of
/// provoking an OOM kill mid-search.
///
/// `system_node` is a **system** NUMA node index, not a logical [`NumaIndex`].
///
/// Returns whether the kernel accepted the policy. A `false` is not an error:
/// the thread simply keeps the policy it inherited.
pub fn set_current_thread_preferred_node(system_node: NumaIndex) -> bool {
    match node_mask(system_node) {
        Some((mask, maxnode)) => sys_set_preferred(&mask, maxnode),
        None => false,
    }
}

/// Place the byte range `[addr, addr + len)` on `system_node`, migrating pages
/// that are already faulted elsewhere
/// (`mbind(MPOL_BIND, {system_node}, MPOL_MF_MOVE)`).
///
/// The remedy for a block another thread allocated *and* first-touched, where no
/// per-thread policy can fix the placement after the fact. `MPOL_MF_STRICT` is
/// not passed, so a page that cannot be migrated is left in place and the call
/// still succeeds.
///
/// The range is widened to whole pages ([`page_span`]); callers pass
/// large-page-backed blocks, which are already 2 MiB-aligned and -sized.
///
/// `system_node` is a **system** NUMA node index. Returns whether the kernel
/// accepted the call; `false` leaves the range exactly as it was.
///
/// No memory is read or written here: the address is handed to the kernel as a
/// range descriptor, which is why this is safe despite taking a raw address.
pub fn migrate_region_to_node(addr: usize, len: usize, system_node: NumaIndex) -> bool {
    let Some((mask, maxnode)) = node_mask(system_node) else {
        return false;
    };
    let Some((start, span)) = page_span(page_size(), addr, len) else {
        return false;
    };
    sys_mbind(start, span, &mask, maxnode)
}

/// The calling thread's own memory policy (`get_mempolicy(…, NULL, 0)`) — the
/// read-back counterpart of [`set_current_thread_preferred_node`].
pub fn current_thread_policy() -> Option<MemPolicy> {
    sys_get_mempolicy(None)
}

/// The memory policy governing `addr` (`get_mempolicy(…, addr, MPOL_F_ADDR)`).
///
/// The read-back counterpart of [`migrate_region_to_node`]. `addr` must be a
/// mapped address; an unmapped one yields `None`. No memory is dereferenced.
pub fn policy_at_address(addr: usize) -> Option<MemPolicy> {
    sys_get_mempolicy(Some(addr))
}

/// `set_mempolicy(MPOL_PREFERRED, mask, maxnode)` on the calling thread.
#[cfg(all(target_os = "linux", not(miri)))]
fn sys_set_preferred(mask: &[usize], maxnode: usize) -> bool {
    // SAFETY: the syscall takes (mode, nodemask, maxnode). `mask` is a live
    // buffer the kernel only reads, and `maxnode` is derived from its length by
    // `node_mask`. The call changes the calling thread's allocation policy and
    // nothing else — it cannot invalidate any pointer or reference.
    let rc = unsafe {
        libc::syscall(
            libc::SYS_set_mempolicy,
            libc::MPOL_PREFERRED as libc::c_long,
            mask.as_ptr(),
            maxnode as libc::c_ulong,
        )
    };
    rc == 0
}

/// Off Linux (and under miri) there is no memory policy to set.
#[cfg(not(all(target_os = "linux", not(miri))))]
fn sys_set_preferred(_mask: &[usize], _maxnode: usize) -> bool {
    false
}

/// `mbind(start, span, MPOL_BIND, mask, maxnode, MPOL_MF_MOVE)`.
#[cfg(all(target_os = "linux", not(miri)))]
fn sys_mbind(start: usize, span: usize, mask: &[usize], maxnode: usize) -> bool {
    // SAFETY: the syscall takes (addr, len, mode, nodemask, maxnode, flags).
    // `mask` is a live buffer the kernel only reads. `start`/`span` describe a
    // page-aligned range that the kernel validates against this process's
    // address space, returning an error for anything unmapped rather than
    // touching it. `mbind` adjusts placement only: it changes neither the
    // contents, nor the mapping, nor the addressability of the range, so no live
    // reference into it is invalidated.
    let rc = unsafe {
        libc::syscall(
            libc::SYS_mbind,
            start as *const libc::c_void,
            span as libc::c_ulong,
            libc::MPOL_BIND as libc::c_long,
            mask.as_ptr(),
            maxnode as libc::c_ulong,
            MPOL_MF_MOVE,
        )
    };
    rc == 0
}

/// Off Linux (and under miri) there is no placement to change.
#[cfg(not(all(target_os = "linux", not(miri))))]
fn sys_mbind(_start: usize, _span: usize, _mask: &[usize], _maxnode: usize) -> bool {
    false
}

/// `get_mempolicy` for either the calling thread (`addr == None`) or the policy
/// governing an address (`addr == Some(a)`, `MPOL_F_ADDR`).
#[cfg(all(target_os = "linux", not(miri)))]
fn sys_get_mempolicy(addr: Option<usize>) -> Option<MemPolicy> {
    // 1024 bits — `MAX_NUMNODES` in the kernel's largest configuration, so the
    // buffer can hold whatever the kernel reports. `maxnode` follows the same
    // `bits + 1` convention as `node_mask`.
    const MASK_WORDS: usize = (MAX_NODE_INDEX + 1) / MASK_WORD_BITS;
    let mut mask = [0usize; MASK_WORDS];
    let maxnode = MASK_WORDS * MASK_WORD_BITS + 1;
    let mut mode: i32 = 0;

    let (ptr, flags) = match addr {
        Some(a) => (a as *const libc::c_void, MPOL_F_ADDR),
        None => (std::ptr::null::<libc::c_void>(), 0),
    };

    // SAFETY: the syscall takes (mode_out, nodemask_out, maxnode, addr, flags).
    // `mode` and `mask` are live, exclusively borrowed out-parameters sized to
    // `maxnode`; the kernel writes only into them. `ptr` is used purely as a
    // lookup key into this process's address space and is never dereferenced
    // here.
    let rc = unsafe {
        libc::syscall(
            libc::SYS_get_mempolicy,
            &mut mode as *mut i32,
            mask.as_mut_ptr(),
            maxnode as libc::c_ulong,
            ptr,
            flags,
        )
    };
    if rc != 0 {
        return None;
    }
    Some(MemPolicy {
        mode,
        nodes: nodes_from_mask(&mask),
    })
}

/// Off Linux (and under miri) there is no memory policy to report.
#[cfg(not(all(target_os = "linux", not(miri))))]
fn sys_get_mempolicy(_addr: Option<usize>) -> Option<MemPolicy> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- node mask construction -------------------------------------------

    #[test]
    fn node_mask_sets_exactly_one_bit() {
        for node in [0usize, 1, 5, 63, 64, 65, 127, 128, 1023] {
            let (mask, _) = node_mask(node).expect("node is within range");
            assert_eq!(
                nodes_from_mask(&mask),
                vec![node],
                "the mask for node {node} names exactly that node"
            );
        }
    }

    #[test]
    fn node_mask_word_count_grows_with_the_index() {
        let w = MASK_WORD_BITS;
        assert_eq!(node_mask(0).unwrap().0.len(), 1);
        assert_eq!(node_mask(w - 1).unwrap().0.len(), 1);
        assert_eq!(node_mask(w).unwrap().0.len(), 2);
        assert_eq!(node_mask(2 * w - 1).unwrap().0.len(), 2);
        assert_eq!(node_mask(2 * w).unwrap().0.len(), 3);
    }

    #[test]
    fn node_mask_maxnode_follows_the_libnuma_plus_one_convention() {
        // `maxnode` must exceed the mask's bit count by one: the kernel
        // decrements it before deriving the valid-bit mask of the last word, so
        // passing the bit count itself would drop that word's top bit —
        // precisely the only bit set for a node on a word boundary.
        let w = MASK_WORD_BITS;
        assert_eq!(node_mask(0).unwrap().1, w + 1);
        assert_eq!(node_mask(w - 1).unwrap().1, w + 1);
        assert_eq!(node_mask(w).unwrap().1, 2 * w + 1);
    }

    #[test]
    fn node_mask_rejects_an_out_of_range_index() {
        assert!(node_mask(MAX_NODE_INDEX).is_some());
        assert!(node_mask(MAX_NODE_INDEX + 1).is_none());
        assert!(node_mask(usize::MAX).is_none());
    }

    #[test]
    fn nodes_from_mask_decodes_several_bits_in_order() {
        let w = MASK_WORD_BITS;
        let mask = vec![0b1001usize, 0b10usize];
        assert_eq!(nodes_from_mask(&mask), vec![0, 3, w + 1]);
        assert!(nodes_from_mask(&[0usize; 4]).is_empty());
    }

    // -- page rounding ----------------------------------------------------

    #[test]
    fn page_span_rounds_the_base_down_and_the_end_up() {
        // A range starting mid-page covers the whole page it starts in.
        assert_eq!(page_span(4096, 4096 + 10, 1), Some((4096, 4096)));
        // …and spills into the next one once it crosses the boundary.
        assert_eq!(page_span(4096, 4096 + 10, 4096), Some((4096, 8192)));
        // An already-aligned, whole-page range is unchanged.
        assert_eq!(page_span(4096, 8192, 4096), Some((8192, 4096)));
        assert_eq!(page_span(4096, 8192, 8192), Some((8192, 8192)));
        // One byte past a page boundary pulls in the next page.
        assert_eq!(page_span(4096, 8192, 4097), Some((8192, 8192)));
    }

    #[test]
    fn page_span_handles_a_huge_page_size() {
        const HUGE: usize = 2 * 1024 * 1024;
        // The large-page allocator's blocks are HUGE-aligned and HUGE-rounded,
        // so their span is exactly the block.
        assert_eq!(
            page_span(HUGE, 4 * HUGE, 3 * HUGE),
            Some((4 * HUGE, 3 * HUGE))
        );
        // A 4 KiB-aligned base inside a huge page still rounds out to the page.
        assert_eq!(page_span(HUGE, 4 * HUGE + 4096, 1), Some((4 * HUGE, HUGE)));
    }

    #[test]
    fn page_span_rejects_an_empty_range() {
        assert_eq!(page_span(4096, 4096, 0), None);
        assert_eq!(page_span(4096, 0, 0), None);
    }

    #[test]
    fn page_span_rejects_an_overflowing_range() {
        assert_eq!(page_span(4096, usize::MAX - 1, 8), None);
    }

    // -- page size --------------------------------------------------------

    #[test]
    fn page_size_is_a_sane_power_of_two() {
        let p = page_size();
        assert!(p.is_power_of_two(), "page size {p} must be a power of two");
        assert!(p >= 4096, "page size {p} must be at least 4 KiB");
    }

    // -- guard clauses (hold on every platform, and under miri) -----------

    #[test]
    fn an_empty_or_out_of_range_request_is_refused_before_any_syscall() {
        assert!(!migrate_region_to_node(0x1000, 0, 0));
        assert!(!migrate_region_to_node(0x1000, 4096, MAX_NODE_INDEX + 1));
        assert!(!set_current_thread_preferred_node(MAX_NODE_INDEX + 1));
    }

    // -- live syscalls (best-effort; assert nothing changed when refused) --

    /// Still meaningful on a single-node host: every *placement* question has
    /// the same answer there, but a thread's *policy* stays whatever it
    /// inherited until something sets it, so reading it back is a real
    /// assertion.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn preferred_node_policy_is_installed_on_the_calling_thread_only() {
        // Run in a spawned thread so the test harness's own policy is untouched.
        std::thread::spawn(|| {
            let before = current_thread_policy();
            if !set_current_thread_preferred_node(0) {
                // No CONFIG_NUMA, a seccomp filter, or a restricted cgroup: the
                // best-effort contract says "leave today's behaviour alone".
                assert_eq!(
                    current_thread_policy(),
                    before,
                    "a refused set_mempolicy must not change the policy"
                );
                return;
            }
            let after = current_thread_policy().expect("get_mempolicy after a successful set");
            assert_eq!(after.mode, MODE_PREFERRED);
            assert_eq!(after.nodes, vec![0]);
        })
        .join()
        .expect("the policy thread must not panic");

        // The policy is per-thread: this thread is unaffected by the above.
        if let Some(here) = current_thread_policy() {
            assert_ne!(
                (here.mode, here.nodes.as_slice()),
                (MODE_PREFERRED, [0usize].as_slice()),
                "set_mempolicy must not leak out of the thread that called it"
            );
        }
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn migrating_a_region_installs_a_bind_policy_over_it() {
        // A page-aligned, two-page owned allocation, written before and after so
        // a botched `mbind` would show up as corrupted or unmapped memory.
        let page = page_size();
        let layout = std::alloc::Layout::from_size_align(2 * page, page).expect("valid layout");
        // SAFETY: `layout` has non-zero size; the block is freed exactly once
        // below and is never aliased.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "the allocation must succeed");
        // SAFETY: `ptr` addresses `2 * page` live, initialised bytes, and this is
        // the only reference to them.
        let bytes = unsafe { std::slice::from_raw_parts_mut(ptr, 2 * page) };
        bytes[0] = 0xAB;
        bytes[2 * page - 1] = 0xCD;

        let addr = ptr as usize;
        let before = policy_at_address(addr);
        if migrate_region_to_node(addr, 2 * page, 0) {
            let after = policy_at_address(addr).expect("get_mempolicy over a mapped address");
            assert_eq!(after.mode, MODE_BIND);
            assert_eq!(after.nodes, vec![0]);
        } else {
            assert_eq!(
                policy_at_address(addr),
                before,
                "a refused mbind must not change the range's policy"
            );
        }
        // The bytes survive either way — `mbind` moves pages, it does not change
        // their contents.
        assert_eq!(bytes[0], 0xAB);
        assert_eq!(bytes[2 * page - 1], 0xCD);

        // SAFETY: `ptr` / `layout` are exactly what `alloc_zeroed` returned, and
        // the block is freed once.
        unsafe { std::alloc::dealloc(ptr, layout) };
    }
}
