//! Per-NUMA-node replication of the loaded NNUE network (Linux-only, `numa` feature).
//! Each node gets a leaked, node-bound (`mbind`) `mmap` copy of the read-only weights so workers read node-local, off the shared `Arc`/`RwLock` eval path.

use std::ptr::NonNull;
use std::sync::RwLock;

use super::aligned::Aligned64;
use super::types::{NetworkStack, NnueNetwork};

// `set_mempolicy(2)` numbers; not re-exported by `libc`, so spell them out.
const MPOL_BIND: libc::c_int = 2;
const MPOL_MF_STRICT: libc::c_uint = 1;
const MPOL_MF_MOVE: libc::c_uint = 2;

/// Cache-line / SIMD alignment for NNUE buffers (matches `aligned::ALIGN`; `mmap` base is page-aligned).
const ALIGN: usize = 64;

/// Node → replica table, populated once per network load by [`build_replicas`].
struct Replicas {
    by_node: Vec<(u32, &'static NnueNetwork)>,
}

static REPLICAS: RwLock<Option<Replicas>> = RwLock::new(None);

/// Builds one node-local replica of `src` per NUMA node; a failed node falls back to the shared global. Called at `isready`; reload leaks the previous replicas.
pub fn build_replicas(src: &NnueNetwork) {
    let mut built: Vec<(u32, &'static NnueNetwork)> = Vec::new();
    for node in crate::numa::node_ids() {
        if let Some(net) = replicate_to_node(src, node) {
            built.push((node, net));
        }
    }
    *REPLICAS.write().expect("nnue::numa REPLICAS write lock poisoned") = Some(Replicas { by_node: built });
}

/// The node-local replica for `node`, or `None` to fall back to the shared global.
pub fn replica_for_node(node: u32) -> Option<&'static NnueNetwork> {
    REPLICAS
        .read()
        .expect("nnue::numa REPLICAS read lock poisoned")
        .as_ref()
        .and_then(|r| r.by_node.iter().find(|(n, _)| *n == node).map(|(_, net)| *net))
}

/// Bytes a buffer of `len` `T`s occupies in the arena, rounded up to [`ALIGN`].
fn aligned_bytes<T>(len: usize) -> usize {
    let bytes = len * std::mem::size_of::<T>();
    (bytes + ALIGN - 1) & !(ALIGN - 1)
}

/// Total arena size needed to hold every weight/bias buffer of `net` (each 64-aligned).
fn arena_bytes(net: &NnueNetwork) -> usize {
    let mut total = aligned_bytes::<i16>(net.ft_biases.len()) + aligned_bytes::<i16>(net.ft_weights.len());
    for s in &net.stacks {
        total += aligned_bytes::<i32>(s.fc_0_biases.len());
        total += aligned_bytes::<i8>(s.fc_0_weights.len());
        total += aligned_bytes::<i32>(s.fc_1_biases.len());
        total += aligned_bytes::<i8>(s.fc_1_weights.len());
        total += aligned_bytes::<i32>(s.fc_2_biases.len());
        total += aligned_bytes::<i8>(s.fc_2_weights.len());
    }
    total
}

/// Node-bound `mmap` region with a 64-aligned bump pointer; never unmapped (process-lifetime).
struct NodeArena {
    base: *mut u8,
    len: usize,
    used: usize,
}

impl NodeArena {
    /// `mmap`s `size` bytes and binds them to `node`. `None` only if `mmap` fails; `mbind` is best-effort (an unbound copy still wins by staying off the shared `Arc`/`RwLock`).
    fn new(node: u32, size: usize) -> Option<NodeArena> {
        if size == 0 {
            return None;
        }
        // SAFETY: `sysconf` and `mmap` are plain libc calls with valid arguments.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) }.max(1) as usize;
        let len = (size + page - 1) & !(page - 1);
        // SAFETY: anonymous private mapping, no fd, standard flags.
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if base == libc::MAP_FAILED {
            return None;
        }
        let base = base as *mut u8;
        // SAFETY: base/len from the successful mmap. Best-effort THP hint issued before the arena
        // is faulted, so the NNUE weights fault in as 2 MiB hugepages (fewer dTLB misses).
        unsafe { libc::madvise(base as *mut libc::c_void, len, libc::MADV_HUGEPAGE) };
        // SAFETY: `base`/`len` from the successful `mmap`; binding is best-effort, return ignored.
        let _bound = unsafe { mbind_region(base, len, node) };
        Some(NodeArena { base, len, used: 0 })
    }

    /// Reserves `bytes` of 64-aligned space and returns its base, or `None` if exhausted.
    fn alloc(&mut self, bytes: usize) -> Option<*mut u8> {
        let off = (self.used + ALIGN - 1) & !(ALIGN - 1);
        let end = off.checked_add(bytes)?;
        if end > self.len {
            return None;
        }
        self.used = (end + ALIGN - 1) & !(ALIGN - 1);
        // SAFETY: `off < self.len` and the region is `self.len` bytes from `base`.
        Some(unsafe { self.base.add(off) })
    }

    /// Prefetches the whole region onto its node. Best-effort; ignored on failure.
    fn prefetch(&self) {
        // SAFETY: `base`/`len` describe the live mapping.
        unsafe { libc::madvise(self.base as *mut libc::c_void, self.len, libc::MADV_WILLNEED) };
    }
}

/// Binds `[addr, addr+len)` to `node` (strict, move faulted pages); returns whether `mbind` succeeded.
unsafe fn mbind_region(addr: *mut u8, len: usize, node: u32) -> bool {
    // Node bitmask with a single bit set; `maxnode` counts the bits the kernel scans.
    let words = (node as usize / 64) + 1;
    let mut mask = vec![0u64; words];
    mask[node as usize / 64] = 1u64 << (node as usize % 64);
    let maxnode = (words * 64) as libc::c_ulong;
    // SAFETY: arguments match the `mbind(2)` ABI; `mask` lives across the call.
    let ret = unsafe {
        libc::syscall(
            libc::SYS_mbind,
            addr as *mut libc::c_void,
            len as libc::c_ulong,
            MPOL_BIND,
            mask.as_ptr() as *const libc::c_ulong,
            maxnode,
            MPOL_MF_STRICT | MPOL_MF_MOVE,
        )
    };
    ret == 0
}

/// Copies `src` into the arena and returns a non-owning [`Aligned64`] view (empty uses an owned sentinel).
unsafe fn copy_buf<T: Copy>(arena: &mut NodeArena, src: &[T]) -> Option<Aligned64<T>> {
    if src.is_empty() {
        return Some(Aligned64::from_slice(src));
    }
    let dst = arena.alloc(std::mem::size_of_val(src))? as *mut T;
    // SAFETY: `dst` has room for `src.len()` 64-aligned `T`s, disjoint from `src`.
    unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len()) };
    let ptr = NonNull::new(dst)?;
    // SAFETY: `ptr` is 64-aligned, non-null, holds `src.len()` init `T`, and is leaked by the caller.
    Some(unsafe { Aligned64::from_borrowed_raw(ptr, src.len()) })
}

/// Builds a node-local replica of `src` bound to `node`, leaked to `&'static`; `None` if the arena `mmap`/space fails.
fn replicate_to_node(src: &NnueNetwork, node: u32) -> Option<&'static NnueNetwork> {
    let mut arena = NodeArena::new(node, arena_bytes(src))?;

    // SAFETY (all `copy_buf` calls): each `Aligned64` borrows `arena` and is `Box::leak`ed below, never dropped.
    let ft_biases = unsafe { copy_buf(&mut arena, &src.ft_biases) }?;
    let ft_weights = unsafe { copy_buf(&mut arena, &src.ft_weights) }?;

    let mut stacks = Vec::with_capacity(src.stacks.len());
    for s in &src.stacks {
        stacks.push(NetworkStack {
            fc_0_biases: unsafe { copy_buf(&mut arena, &s.fc_0_biases) }?,
            fc_0_weights: unsafe { copy_buf(&mut arena, &s.fc_0_weights) }?,
            fc_1_biases: unsafe { copy_buf(&mut arena, &s.fc_1_biases) }?,
            fc_1_weights: unsafe { copy_buf(&mut arena, &s.fc_1_weights) }?,
            fc_2_biases: unsafe { copy_buf(&mut arena, &s.fc_2_biases) }?,
            fc_2_weights: unsafe { copy_buf(&mut arena, &s.fc_2_weights) }?,
        });
    }

    arena.prefetch();
    // Arena is never unmapped: `NodeArena` has no `Drop` and `replica` is leaked below (process-lifetime).

    let replica = NnueNetwork {
        header: src.header.clone(),
        ft_biases,
        ft_weights,
        stacks,
        sha256: src.sha256,
    };
    Some(Box::leak(Box::new(replica)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_bytes_rounds_each_buffer_to_alignment() {
        // 3 i16 = 6 bytes → 64; 1 i16 = 2 bytes → 64. Empty stacks contribute nothing.
        let net = NnueNetwork {
            header: super::super::types::NetHeader {
                version: 0,
                hash: 0,
                arch_id: "t".to_string(),
            },
            ft_biases: Aligned64::from_slice(&[1i16, 2, 3]),
            ft_weights: Aligned64::from_slice(&[9i16]),
            stacks: Vec::new(),
            sha256: [0u8; 32],
        };
        assert_eq!(arena_bytes(&net), 128);
    }

    #[test]
    fn replicate_node0_matches_source_bytes() {
        // Node 0 always exists; a successful replica must be byte-identical to the source.
        let net = NnueNetwork {
            header: super::super::types::NetHeader {
                version: 7,
                hash: 0xABCD,
                arch_id: "replica-test".to_string(),
            },
            ft_biases: Aligned64::from_slice(&[10i16, -20, 30, -40]),
            ft_weights: (0..2048i32).map(|i| (i % 97 - 48) as i16).collect(),
            stacks: vec![NetworkStack {
                fc_0_biases: Aligned64::from_slice(&[1i32, 2, 3]),
                fc_0_weights: (0..64i32).map(|i| (i - 32) as i8).collect(),
                fc_1_biases: Aligned64::from_slice(&[5i32]),
                fc_1_weights: Aligned64::from_slice(&[-7i8, 7]),
                fc_2_biases: Aligned64::from_slice(&[42i32]),
                fc_2_weights: Aligned64::from_slice(&[1i8]),
            }],
            sha256: [0xEE; 32],
        };

        let replica = replicate_to_node(&net, 0).expect("node 0 replica should build");
        assert_eq!(&*replica.ft_biases, &*net.ft_biases);
        assert_eq!(&*replica.ft_weights, &*net.ft_weights);
        assert_eq!(replica.header.hash, net.header.hash);
        assert_eq!(replica.sha256, net.sha256);
        assert_eq!(replica.stacks.len(), 1);
        assert_eq!(&*replica.stacks[0].fc_0_weights, &*net.stacks[0].fc_0_weights);
        assert_eq!(&*replica.stacks[0].fc_2_biases, &*net.stacks[0].fc_2_biases);
        // The replica's weight buffers must be 64-byte aligned (mmap base is page-aligned).
        assert_eq!(replica.ft_weights.as_ptr() as usize % ALIGN, 0);
        assert_eq!(replica.stacks[0].fc_0_weights.as_ptr() as usize % ALIGN, 0);
    }
}
