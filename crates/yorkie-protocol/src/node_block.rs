//! The node-shared history tables, in one block of static storage per NUMA
//! node that holds a worker.
//!
//! Which node each worker sits on is fixed when the binary is built
//! ([`config::WORKER_SYSTEM_NODES`]), and so is the size of each node's shared
//! tables: they scale with the node's worker count rounded up to a power of
//! two. The storage for every node's tables is therefore laid out at build
//! time ([`RegionLayout`]): a region of one block per node, in ascending node
//! order, each block sized for its own node, in the binary's BSS, stated to the
//! assembler through [`yorkie_storage::large_page_bss!`] as the per-worker
//! blocks are. The region is zero at process start and occupies no bytes of
//! the file.
//!
//! # Placement
//!
//! The region starts on a large-page boundary, every table in a block starts
//! on one, and every block is a whole number of large pages, so no large page
//! holds tables of two nodes. The region is hinted for huge pages once, when
//! it is claimed. Nothing writes a block until its node's set is built:
//! claiming the region writes nothing, and the engine builds each node's set —
//! which writes every entry of its tables — on a thread pinned to one of that
//! node's worker CPUs, before any worker of the node searches. A page
//! therefore faults in on the node whose workers use it.
//!
//! # Sharing
//!
//! Every entry is an atomic, and the region is only ever reached through
//! shared references to its entries, so any number of workers may read and
//! write one table at once, and a table may be handed out again while earlier
//! handles over it are alive: they all address the same atomics, which static
//! storage keeps for the life of the program. That is what makes the hand-out
//! safe; [`region`] is the one place a reference into the region is made.
//!
//! What a claim adds is that one engine at a time builds over the region: a
//! second engine in the same process (a test binary that runs several) is
//! refused it and keeps its tables on the heap, so two engines' searches do
//! not write each other's tables.
//!
//! Under miri, which can neither assemble the region nor take its address, the
//! region is one zeroed heap allocation made on first use and never freed.

use std::sync::Once;
use std::sync::atomic::{AtomicBool, AtomicI16, Ordering};

use yorkie_numa::NumaIndex;
use yorkie_search::SharedHistories;
use yorkie_storage::LARGE_PAGE_ALIGN;

use crate::config;

/// Bytes of one table entry.
const ENTRY_BYTES: usize = size_of::<AtomicI16>();

/// Entries in one large page: every table starts on a multiple of this.
const PAGE_ENTRIES: usize = LARGE_PAGE_ALIGN / ENTRY_BYTES;

/// One table's place in the region, in entries from the region's start.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TableSpan {
    /// The index of the table's first entry.
    pub(crate) start: usize,
    /// The table's entry count.
    pub(crate) len: usize,
}

impl TableSpan {
    /// The index one past the table's last entry.
    pub(crate) const fn end(self) -> usize {
        self.start + self.len
    }
}

/// One node's block of the region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct NodeBlock {
    /// The system node whose workers use the block.
    pub(crate) node: NumaIndex,
    /// The workers the build puts on the node.
    pub(crate) workers: usize,
    /// `workers` rounded up to a power of two: what the tables are sized by.
    pub(crate) thread_count: usize,
    /// The block's first byte, from the region's start.
    pub(crate) offset: usize,
    /// The block's length in bytes, a whole number of large pages.
    pub(crate) bytes: usize,
    /// The correction table's entries.
    pub(crate) correction: TableSpan,
    /// The pawn table's entries.
    pub(crate) pawn: TableSpan,
}

impl NodeBlock {
    /// A placeholder for an unused slot of [`RegionLayout::blocks`].
    const UNUSED: Self = Self {
        node: 0,
        workers: 0,
        thread_count: 0,
        offset: 0,
        bytes: 0,
        correction: TableSpan { start: 0, len: 0 },
        pawn: TableSpan { start: 0, len: 0 },
    };
}

/// Where every node's block lies in the region, for a build whose workers sit
/// on the nodes a table names; room for at most `N` nodes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RegionLayout<const N: usize> {
    /// The blocks in ascending node order; the first `len` are used.
    blocks: [NodeBlock; N],
    /// The number of nodes that hold a worker.
    len: usize,
    /// The region's length in bytes.
    bytes: usize,
}

impl<const N: usize> RegionLayout<N> {
    /// The layout for workers on the system nodes `worker_nodes`, one entry per
    /// worker: one block for each node named, sized by the number of times it
    /// is named.
    pub(crate) const fn of(worker_nodes: &[NumaIndex]) -> Self {
        let mut nodes = [0; N];
        let mut counts = [0usize; N];
        let mut len = 0;
        let mut w = 0;
        while w < worker_nodes.len() {
            let node = worker_nodes[w];
            let mut i = 0;
            while i < len && nodes[i] < node {
                i += 1;
            }
            if i < len && nodes[i] == node {
                counts[i] += 1;
            } else {
                assert!(
                    len < N,
                    "the workers sit on more nodes than the layout holds"
                );
                let mut j = len;
                while j > i {
                    nodes[j] = nodes[j - 1];
                    counts[j] = counts[j - 1];
                    j -= 1;
                }
                nodes[i] = node;
                counts[i] = 1;
                len += 1;
            }
            w += 1;
        }

        let mut blocks = [NodeBlock::UNUSED; N];
        let mut at = 0;
        let mut i = 0;
        while i < len {
            let thread_count = counts[i].next_power_of_two();
            let start = at;
            let correction = TableSpan {
                start: at,
                len: SharedHistories::correction_len(thread_count),
            };
            at = correction.end().next_multiple_of(PAGE_ENTRIES);
            let pawn = TableSpan {
                start: at,
                len: SharedHistories::pawn_len(thread_count),
            };
            at = pawn.end().next_multiple_of(PAGE_ENTRIES);
            blocks[i] = NodeBlock {
                node: nodes[i],
                workers: counts[i],
                thread_count,
                offset: start * ENTRY_BYTES,
                bytes: (at - start) * ENTRY_BYTES,
                correction,
                pawn,
            };
            i += 1;
        }
        Self {
            blocks,
            len,
            bytes: at * ENTRY_BYTES,
        }
    }

    /// The blocks, in ascending node order.
    pub(crate) const fn blocks(&self) -> &[NodeBlock] {
        self.blocks.split_at(self.len).0
    }

    /// The region's length in bytes.
    pub(crate) const fn bytes(&self) -> usize {
        self.bytes
    }

    /// The block of system node `node`, if a worker sits there.
    pub(crate) fn block(&self, node: NumaIndex) -> Option<&NodeBlock> {
        self.blocks().iter().find(|b| b.node == node)
    }
}

/// At most one node per worker.
const MAX_NODES: usize = config::THREADS as usize;

/// This build's layout.
pub(crate) const LAYOUT: RegionLayout<MAX_NODES> = RegionLayout::of(config::WORKER_SYSTEM_NODES);

/// Entries in the region.
const REGION_ENTRIES: usize = LAYOUT.bytes() / ENTRY_BYTES;

const _: () = assert!(LAYOUT.bytes().is_multiple_of(LARGE_PAGE_ALIGN));
const _: () = assert!(REGION_ENTRIES > 0);
const _: () = {
    let blocks = LAYOUT.blocks();
    let mut at = 0;
    let mut i = 0;
    while i < blocks.len() {
        let b = blocks[i];
        assert!(b.offset == at, "the blocks lie back to back");
        assert!(b.bytes.is_multiple_of(LARGE_PAGE_ALIGN));
        assert!(b.thread_count == b.workers.next_power_of_two());
        assert!(b.correction.start * ENTRY_BYTES == b.offset);
        assert!(b.correction.end() <= b.pawn.start);
        assert!((b.pawn.start * ENTRY_BYTES).is_multiple_of(LARGE_PAGE_ALIGN));
        assert!(b.pawn.end() * ENTRY_BYTES <= b.offset + b.bytes);
        at += b.bytes;
        i += 1;
    }
    assert!(at == LAYOUT.bytes(), "the blocks tile the region");
};

/// The region's storage: [`REGION_ENTRIES`] entries on a large-page boundary.
#[repr(C, align(2097152))]
struct NodeRegion([AtomicI16; REGION_ENTRIES]);

const _: () = assert!(align_of::<NodeRegion>().is_multiple_of(LARGE_PAGE_ALIGN));
const _: () = assert!(size_of::<NodeRegion>() == LAYOUT.bytes());

yorkie_storage::large_page_bss! {
    /// The region's storage.
    fn region_storage() -> *mut NodeRegion = far "yorkie_node_region";
}

/// The region's base under miri.
#[cfg(miri)]
struct MiriRegion(*mut NodeRegion);

// SAFETY: the pointer is only the address `region` makes its shared slice
// from; every access through that slice is atomic.
#[cfg(miri)]
unsafe impl Send for MiriRegion {}
// SAFETY: as for `Send`.
#[cfg(miri)]
unsafe impl Sync for MiriRegion {}

/// The region's stand-in under miri, allocated on first use.
#[cfg(miri)]
static MIRI_REGION: std::sync::OnceLock<MiriRegion> = std::sync::OnceLock::new();

/// Every entry of the region.
fn region() -> &'static [AtomicI16] {
    #[cfg(not(miri))]
    let base = region_storage();
    #[cfg(miri)]
    let base = MIRI_REGION
        .get_or_init(|| {
            let layout = std::alloc::Layout::new::<NodeRegion>();
            // SAFETY: the layout is non-zero-sized.
            let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            MiriRegion(ptr.cast())
        })
        .0;
    // SAFETY: `base` addresses `REGION_ENTRIES` entries that live for the
    // whole program — static storage, or under miri an allocation that is
    // never freed — aligned for `AtomicI16` and zero until written, which is a
    // valid `AtomicI16`. Nothing makes a unique reference into the region or
    // reaches it other than through this slice, whose every access is atomic,
    // so any number of these shared slices may coexist on any threads.
    unsafe { std::slice::from_raw_parts(base.cast::<AtomicI16>(), REGION_ENTRIES) }
}

/// The entries `span` names in `region`.
fn carve(region: &[AtomicI16], span: TableSpan) -> &[AtomicI16] {
    &region[span.start..span.end()]
}

/// The correction table of `block`, sized for `thread_count` workers (its own
/// thread count or a smaller power of two: a prefix of the table).
fn correction_in<'a>(
    region: &'a [AtomicI16],
    block: &NodeBlock,
    thread_count: usize,
) -> Option<&'a [AtomicI16]> {
    (thread_count <= block.thread_count).then(|| {
        let table = carve(region, block.correction);
        &table[..SharedHistories::correction_len(thread_count)]
    })
}

/// The pawn table of `block`, sized for `thread_count` workers. See
/// [`correction_in`].
fn pawn_in<'a>(
    region: &'a [AtomicI16],
    block: &NodeBlock,
    thread_count: usize,
) -> Option<&'a [AtomicI16]> {
    (thread_count <= block.thread_count).then(|| {
        let table = carve(region, block.pawn);
        &table[..SharedHistories::pawn_len(thread_count)]
    })
}

/// Whether some engine holds the region.
static CLAIMED: AtomicBool = AtomicBool::new(false);

/// The huge-page hint over the whole region, given once.
static ADVISED: Once = Once::new();

/// One engine's hold on the region: the one value that hands out its tables.
#[derive(Debug)]
pub(crate) struct NodeClaim(());

impl NodeClaim {
    /// Hold the region, or `None` while another engine holds it. Nothing in
    /// the region is written.
    pub(crate) fn claim() -> Option<Self> {
        CLAIMED
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .ok()?;
        ADVISED.call_once(|| {
            let entries = region();
            yorkie_storage::advise_huge_pages(entries.as_ptr().addr(), size_of_val(entries));
        });
        Some(Self(()))
    }

    /// Node `node`'s correction table, sized for `thread_count` workers: `None`
    /// when no worker of the build sits on `node`, or when its block is sized
    /// for fewer workers.
    pub(crate) fn correction(
        &self,
        node: NumaIndex,
        thread_count: usize,
    ) -> Option<&'static [AtomicI16]> {
        correction_in(region(), LAYOUT.block(node)?, thread_count)
    }

    /// Node `node`'s pawn table, sized for `thread_count` workers. See
    /// [`Self::correction`].
    pub(crate) fn pawn(
        &self,
        node: NumaIndex,
        thread_count: usize,
    ) -> Option<&'static [AtomicI16]> {
        pawn_in(region(), LAYOUT.block(node)?, thread_count)
    }

    /// The `(address, byte length)` of node `node`'s block.
    #[cfg(test)]
    pub(crate) fn backing_region(&self, node: NumaIndex) -> Option<(usize, usize)> {
        let block = LAYOUT.block(node)?;
        Some((region().as_ptr().addr() + block.offset, block.bytes))
    }
}

impl Drop for NodeClaim {
    fn drop(&mut self) {
        CLAIMED.store(false, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two nodes of different worker counts, named out of order and with a
    /// node between them that holds none.
    const TWO_NODES: RegionLayout<4> = RegionLayout::of(&[3, 1, 3, 3]);

    const _: () = assert!(TWO_NODES.blocks().len() == 2);
    const _: () = assert!(TWO_NODES.bytes().is_multiple_of(LARGE_PAGE_ALIGN));

    /// Each block follows its own node's worker count, rounded up to a power of
    /// two, rather than the largest node's.
    #[test]
    fn each_block_is_sized_for_its_own_node() {
        let [one, three] = TWO_NODES.blocks() else {
            panic!("two blocks");
        };
        assert_eq!((one.node, one.workers, one.thread_count), (1, 1, 1));
        assert_eq!((three.node, three.workers, three.thread_count), (3, 3, 4));
        assert_eq!(one.correction.len, SharedHistories::correction_len(1));
        assert_eq!(three.correction.len, SharedHistories::correction_len(4));
        assert_eq!(one.pawn.len, SharedHistories::pawn_len(1));
        assert_eq!(three.pawn.len, SharedHistories::pawn_len(4));
        assert!(one.bytes < three.bytes);
        assert_eq!(TWO_NODES.bytes(), one.bytes + three.bytes);
        assert_eq!(TWO_NODES.block(2), None);
    }

    /// The blocks lie back to back without overlapping, each a whole number of
    /// large pages, with every table on a large-page boundary inside its own
    /// block.
    #[test]
    fn the_blocks_tile_the_region_on_large_pages() {
        let mut at = 0;
        for b in TWO_NODES.blocks() {
            assert_eq!(b.offset, at);
            assert!(b.offset.is_multiple_of(LARGE_PAGE_ALIGN));
            assert!(b.bytes.is_multiple_of(LARGE_PAGE_ALIGN));
            assert!(b.correction.end() <= b.pawn.start);
            for table in [b.correction, b.pawn] {
                let start = table.start * ENTRY_BYTES;
                assert!(start.is_multiple_of(LARGE_PAGE_ALIGN));
                assert!(start >= b.offset && table.end() * ENTRY_BYTES <= b.offset + b.bytes);
            }
            at += b.bytes;
        }
        assert_eq!(at, TWO_NODES.bytes());
    }

    /// Workers of one node are handed the same tables, and workers of
    /// different nodes disjoint ones, over a region laid out for two nodes of
    /// different worker counts.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn workers_of_one_node_share_tables_and_of_two_nodes_do_not() {
        let storage =
            yorkie_storage::LargePageAlloc::zeroed_slice::<AtomicI16>(TWO_NODES.bytes() / 2);
        let worker_nodes = [3, 1, 3, 3];
        let tables: Vec<&[AtomicI16]> = worker_nodes
            .iter()
            .flat_map(|&node| {
                let block = TWO_NODES.block(node).expect("a worker's node has a block");
                [
                    correction_in(&storage, block, block.thread_count),
                    pawn_in(&storage, block, block.thread_count),
                ]
                .map(|t| t.expect("sized for its node"))
            })
            .collect();
        let worker_nodes = worker_nodes.map(|n| [n, n]).as_flattened().to_vec();
        // Two tables per worker: equal ones of one node are the same table.
        let table_of = |i: usize| (worker_nodes[i], i % 2);
        let span = |t: &[AtomicI16]| (t.as_ptr().addr(), t.as_ptr().addr() + size_of_val(t));
        for (i, a) in tables.iter().enumerate() {
            for (j, b) in tables.iter().enumerate() {
                let ((a0, a1), (b0, b1)) = (span(a), span(b));
                if table_of(i) == table_of(j) {
                    assert_eq!((a0, a1), (b0, b1), "tables {i} and {j} are one table");
                } else {
                    assert!(a1 <= b0 || b1 <= a0, "tables {i} and {j} overlap");
                }
            }
        }
    }

    /// A smaller set over a block is a prefix of its table, and a larger one
    /// than the block was sized for is refused.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn a_smaller_set_takes_a_prefix_and_a_larger_one_none() {
        let three = *TWO_NODES.block(3).expect("node 3 holds workers");
        let storage = yorkie_storage::LargePageAlloc::zeroed_slice::<AtomicI16>(three.pawn.end());
        let whole = correction_in(&storage, &three, 4).expect("its own size");
        let one = correction_in(&storage, &three, 1).expect("a smaller size");
        assert_eq!(one.as_ptr(), whole.as_ptr());
        assert_eq!(one.len(), SharedHistories::correction_len(1));
        assert!(correction_in(&storage, &three, 8).is_none());
        let whole = pawn_in(&storage, &three, 4).expect("its own size");
        let one = pawn_in(&storage, &three, 1).expect("a smaller size");
        assert_eq!(one.as_ptr(), whole.as_ptr());
        assert_eq!(one.len(), SharedHistories::pawn_len(1));
        assert!(pawn_in(&storage, &three, 8).is_none());
    }

    /// This build's region: one block per node a worker sits on, the region on
    /// a large-page boundary, each table inside its node's block, and one
    /// engine at a time holding it.
    #[test]
    fn the_build_region_is_held_by_one_claim_at_a_time() {
        let _session = crate::serial_session();
        let region = region();
        assert!(region.as_ptr().addr().is_multiple_of(LARGE_PAGE_ALIGN));
        assert_eq!(size_of_val(region), LAYOUT.bytes());
        let claim = NodeClaim::claim().expect("no engine holds the region");
        assert!(NodeClaim::claim().is_none(), "the region is held");
        for &node in config::WORKER_SYSTEM_NODES {
            let block = LAYOUT.block(node).expect("every worker's node has a block");
            let (addr, len) = claim.backing_region(node).expect("a block");
            let correction = claim
                .correction(node, block.thread_count)
                .expect("the node's own size");
            let pawn = claim
                .pawn(node, block.thread_count)
                .expect("the node's own size");
            for table in [correction, pawn] {
                let at = table.as_ptr().addr();
                assert!(at >= addr && at + size_of_val(table) <= addr + len);
            }
            let ends = |t: &[AtomicI16]| t.as_ptr().addr() + size_of_val(t);
            assert!(ends(correction) <= pawn.as_ptr().addr());
        }
        drop(claim);
        assert!(NodeClaim::claim().is_some(), "the region was given back");
    }
}
