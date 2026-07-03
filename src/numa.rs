//! NUMA topology discovery and CPU-affinity pinning (Linux-only, `numa` feature).
//! Compact-by-node core assignment per worker index; topology cached from `/sys`, `YORKIE_NUMA_RESERVE` cores held back for the OS.
//! Memory placement layers: process default (TT follows the launcher), explicit per-node `mbind` for pre-faulted memory (NNUE replicas, `Thread` structs), and per-worker `set_mempolicy(MPOL_PREFERRED)`. All syscalls best-effort.

use std::sync::OnceLock;

// `set_mempolicy(2)` / `mbind(2)` numbers; not re-exported by `libc`, so spell them out.
const MPOL_PREFERRED: libc::c_int = 1;
const MPOL_BIND: libc::c_int = 2;
const MPOL_MF_STRICT: libc::c_uint = 1;
const MPOL_MF_MOVE: libc::c_uint = 2;

/// Cores held back from the assignable pool for the OS (overridable via `YORKIE_NUMA_RESERVE`).
pub const DEFAULT_RESERVE: usize = 2;

/// Environment variable that overrides [`DEFAULT_RESERVE`] at runtime.
const RESERVE_ENV: &str = "YORKIE_NUMA_RESERVE";

/// A worker thread's core/node placement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Assignment {
    /// Logical CPU (core) id to pin the thread to.
    pub cpu: usize,
    /// NUMA node the core belongs to (selects the node-local NNUE replica).
    pub node: u32,
}

/// One NUMA node and the logical CPUs it owns (sorted ascending).
#[derive(Clone, Debug)]
struct NodeInfo {
    id: u32,
    cpus: Vec<usize>,
}

/// Whole-machine topology plus the derived assignable pool.
#[derive(Clone, Debug)]
struct Topology {
    /// All NUMA nodes, sorted by id; always non-empty. Read via [`node_ids`] to build NNUE replicas.
    #[allow(dead_code)]
    nodes: Vec<NodeInfo>,
    /// Compact-by-node `(cpu, node)` order minus reserved cores; worker `i` maps to `assignable[i % len]`.
    assignable: Vec<Assignment>,
}

static TOPOLOGY: OnceLock<Topology> = OnceLock::new();

fn topology() -> &'static Topology {
    TOPOLOGY.get_or_init(build_topology)
}

/// Reserve count from `YORKIE_NUMA_RESERVE`, else [`DEFAULT_RESERVE`].
fn reserve() -> usize {
    std::env::var(RESERVE_ENV)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_RESERVE)
}

fn build_topology() -> Topology {
    let nodes = read_sys_nodes().unwrap_or_else(fallback_nodes);
    // Defensive: never let an empty/garbage topology produce an empty pool.
    let nodes = if nodes.iter().all(|n| n.cpus.is_empty()) {
        fallback_nodes()
    } else {
        nodes
    };

    // Compact-by-node order: all of node 0's cpus, then node 1's, ...
    let mut order: Vec<Assignment> = Vec::new();
    for node in &nodes {
        for &cpu in &node.cpus {
            order.push(Assignment { cpu, node: node.id });
        }
    }

    // Hold back the last `reserve` cores for the OS, but keep at least one assignable.
    let reserve = reserve().min(order.len().saturating_sub(1));
    order.truncate(order.len() - reserve);

    Topology {
        nodes,
        assignable: order,
    }
}

/// Reads each node's `cpulist` from `/sys`; `None` when absent, so the caller falls back to a flat layout.
fn read_sys_nodes() -> Option<Vec<NodeInfo>> {
    let dir = std::fs::read_dir("/sys/devices/system/node").ok()?;
    let mut nodes: Vec<NodeInfo> = Vec::new();
    for entry in dir.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let Some(id_str) = name.strip_prefix("node") else {
            continue;
        };
        let Ok(id) = id_str.parse::<u32>() else {
            continue;
        };
        let cpulist = std::fs::read_to_string(entry.path().join("cpulist")).ok()?;
        nodes.push(NodeInfo {
            id,
            cpus: parse_cpulist(&cpulist),
        });
    }
    if nodes.is_empty() {
        return None;
    }
    nodes.sort_by_key(|n| n.id);
    Some(nodes)
}

/// Single-node fallback covering CPUs `0..nproc`, used when `/sys` has no NUMA view.
fn fallback_nodes() -> Vec<NodeInfo> {
    let nproc = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    vec![NodeInfo {
        id: 0,
        cpus: (0..nproc).collect(),
    }]
}

/// Parses a Linux `cpulist` such as `"0-3,5,7-8"` into a sorted, de-duplicated list.
fn parse_cpulist(s: &str) -> Vec<usize> {
    let mut cpus = Vec::new();
    for part in s.trim().split(',').filter(|p| !p.is_empty()) {
        match part.split_once('-') {
            Some((lo, hi)) => {
                if let (Ok(lo), Ok(hi)) = (lo.trim().parse::<usize>(), hi.trim().parse::<usize>()) {
                    cpus.extend(lo..=hi);
                }
            }
            None => {
                if let Ok(cpu) = part.trim().parse::<usize>() {
                    cpus.push(cpu);
                }
            }
        }
    }
    cpus.sort_unstable();
    cpus.dedup();
    cpus
}

/// Core/node placement for worker index `idx` (wraps modulo the assignable pool).
pub fn assignment_for_idx(idx: usize) -> Assignment {
    let pool = &topology().assignable;
    pool[idx % pool.len()]
}

/// NUMA node ids, sorted ascending; used to build one NNUE replica per node.
#[allow(dead_code)]
pub fn node_ids() -> Vec<u32> {
    topology().nodes.iter().map(|n| n.id).collect()
}

/// Pins the current thread to `cpu`; `false` if rejected (best-effort, never fatal).
pub fn pin_current_thread(cpu: usize) -> bool {
    core_affinity::set_for_current(core_affinity::CoreId { id: cpu })
}

/// Node bitmask with only `node`'s bit set, plus the `maxnode` bit count the kernel scans.
fn node_mask(node: u32) -> (Vec<u64>, libc::c_ulong) {
    let words = (node as usize / 64) + 1;
    let mut mask = vec![0u64; words];
    mask[node as usize / 64] = 1u64 << (node as usize % 64);
    let maxnode = (words * 64) as libc::c_ulong;
    (mask, maxnode)
}

/// Rounds `[addr, addr + len)` outward to `page` boundaries (edge pages may cover neighbouring bytes).
fn page_span(addr: usize, len: usize, page: usize) -> (usize, usize) {
    let start = addr & !(page - 1);
    let end = addr + len;
    (start, (end - start + page - 1) & !(page - 1))
}

/// Sets the calling thread's task policy to prefer `node` for first-touch (`MPOL_PREFERRED`, best-effort). Worker threads only — USI/master keep the inherited policy so the TT stays on the launcher's.
pub fn prefer_node_for_current_thread(node: u32) -> bool {
    let (mask, maxnode) = node_mask(node);
    // SAFETY: arguments match the `set_mempolicy(2)` ABI; `mask` lives across the call.
    let ret = unsafe {
        libc::syscall(
            libc::SYS_set_mempolicy,
            MPOL_PREFERRED,
            mask.as_ptr() as *const libc::c_ulong,
            maxnode,
        )
    };
    ret == 0
}

/// Binds `[addr, addr + len)` to `node` (`MPOL_BIND` + `MPOL_MF_MOVE`; adds `MPOL_MF_STRICT` when `strict`), outward-rounded to pages; best-effort.
pub unsafe fn mbind_region(addr: *mut u8, len: usize, node: u32, strict: bool) -> bool {
    if len == 0 {
        return false;
    }
    // SAFETY: `sysconf` is a plain libc call with a valid argument.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) }.max(1) as usize;
    let (start, span) = page_span(addr as usize, len, page);
    let (mask, maxnode) = node_mask(node);
    let flags = if strict { MPOL_MF_STRICT | MPOL_MF_MOVE } else { MPOL_MF_MOVE };
    // SAFETY: arguments match the `mbind(2)` ABI; `mask` lives across the call.
    let ret = unsafe {
        libc::syscall(
            libc::SYS_mbind,
            start as *mut libc::c_void,
            span as libc::c_ulong,
            MPOL_BIND,
            mask.as_ptr() as *const libc::c_ulong,
            maxnode,
            flags,
        )
    };
    ret == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_cpulist_handles_ranges_singletons_and_gaps() {
        assert_eq!(parse_cpulist("0-3,5,7-8"), vec![0, 1, 2, 3, 5, 7, 8]);
        assert_eq!(parse_cpulist("4"), vec![4]);
        assert_eq!(parse_cpulist(""), Vec::<usize>::new());
        // Unsorted / duplicate input is normalised.
        assert_eq!(parse_cpulist("3,1,1,2"), vec![1, 2, 3]);
    }

    #[test]
    fn compact_by_node_order_keeps_threads_on_their_node() {
        let nodes = vec![
            NodeInfo {
                id: 0,
                cpus: vec![0, 1, 2, 3],
            },
            NodeInfo {
                id: 1,
                cpus: vec![4, 5, 6, 7],
            },
        ];
        let mut order = Vec::new();
        for node in &nodes {
            for &cpu in &node.cpus {
                order.push(Assignment { cpu, node: node.id });
            }
        }
        // First k indices stay on node 0, next k on node 1 (compact, not scattered).
        assert_eq!(order[0], Assignment { cpu: 0, node: 0 });
        assert_eq!(order[3], Assignment { cpu: 3, node: 0 });
        assert_eq!(order[4], Assignment { cpu: 4, node: 1 });
        assert_eq!(order[7], Assignment { cpu: 7, node: 1 });
    }

    #[test]
    fn assignment_pool_is_non_empty_and_wraps() {
        // Exercises the real machine topology: the pool must always be usable.
        let a0 = assignment_for_idx(0);
        let wrapped = assignment_for_idx(usize::MAX);
        assert!(node_ids().contains(&a0.node));
        assert!(node_ids().contains(&wrapped.node));
    }

    #[test]
    fn node_mask_sets_exactly_the_node_bit() {
        // First word.
        assert_eq!(node_mask(0), (vec![1u64], 64));
        assert_eq!(node_mask(5), (vec![1u64 << 5], 64));
        assert_eq!(node_mask(63), (vec![1u64 << 63], 64));
        // Word boundary: node 64 is bit 0 of the second word.
        assert_eq!(node_mask(64), (vec![0, 1u64], 128));
        assert_eq!(node_mask(65), (vec![0, 1u64 << 1], 128));
    }

    #[test]
    fn page_span_rounds_outward_to_page_boundaries() {
        const PAGE: usize = 4096;
        // Already aligned: unchanged start, len rounded up.
        assert_eq!(page_span(PAGE, PAGE, PAGE), (PAGE, PAGE));
        assert_eq!(page_span(PAGE, 1, PAGE), (PAGE, PAGE));
        // Unaligned start: rounded down, span still covers the last byte.
        assert_eq!(page_span(PAGE + 10, 100, PAGE), (PAGE, PAGE));
        // Range straddling a boundary grows to cover both pages.
        assert_eq!(page_span(PAGE - 6, 10, PAGE), (0, 2 * PAGE));
        assert_eq!(page_span(PAGE + PAGE - 1, 2, PAGE), (PAGE, 2 * PAGE));
    }

    // Best-effort wrappers: assert only that the calls are well-formed (no crash/UB); either outcome is valid.
    #[test]
    fn prefer_node_for_current_thread_is_best_effort() {
        // Scratch thread so the test runner's own task policy is left untouched.
        std::thread::spawn(|| {
            let node = node_ids()[0];
            let _ = prefer_node_for_current_thread(node);
        })
        .join()
        .unwrap();
    }

    #[test]
    fn mbind_region_accepts_an_unaligned_heap_range() {
        let node = node_ids()[0];
        let buf = vec![0u8; 3 * 4096];
        // Unaligned interior range exercises the helper's page rounding.
        // SAFETY: the range lies inside the live `buf` allocation owned by this test.
        let _ = unsafe { mbind_region(buf.as_ptr().add(100) as *mut u8, 200, node, false) };
        drop(buf);
    }
}
