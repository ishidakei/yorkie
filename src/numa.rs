//! NUMA topology discovery and CPU-affinity pinning (Linux-only, `numa` feature).
//! Compact-by-node core assignment per worker index; topology cached from `/sys`, `YORKIE_NUMA_RESERVE` cores held back for the OS.

use std::sync::OnceLock;

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
}
