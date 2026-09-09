//! NUMA topology discovery and per-CPU worker pinning.
//!
//! This crate describes the machine's NUMA layout — the system nodes, the CPUs
//! each one holds, and the L3 cache domain every CPU belongs to — and, on Linux,
//! pins the calling thread to a single logical CPU.
//!
//! Pinning a worker to a CPU says nothing about where that worker's memory
//! lives: under `numactl --interleave=all` the process default policy
//! round-robins every page, first-touch included. The [`mempolicy`] module adds
//! the Linux wrappers that let a pinned worker keep its *private* working set on
//! its own node without disturbing the process default, which the shared
//! transposition table depends on.
//!
//! Linux-only by design. The pure parsing and topology code compiles and runs
//! everywhere, and the real-syscall pieces degrade to an "all system threads"
//! fallback elsewhere so the crate stays buildable and testable on non-Linux
//! hosts.
//!
//! All sysfs readers take an injectable root path via [`SysfsOptions`], so tests
//! run against fixture directories rather than the live `/sys` tree.

pub mod mempolicy;

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// A processor (CPU) index, as numbered by the operating system.
pub type CpuIndex = usize;

/// A NUMA-node index.
///
/// A *system* node index is the number the kernel gives the node; a *logical*
/// index is the node's position in [`NumaLayout::nodes`]. The two differ
/// whenever the online node numbers have a gap, which is why a layout carries
/// [`NumaLayout::system_nodes`] beside its CPU lists.
pub type NumaIndex = usize;

/// The machine a sysfs tree describes: where to read it, and which CPUs it
/// reports online.
#[derive(Debug, Clone)]
pub struct SysfsOptions {
    /// Root under which the `devices/system/...` sysfs hierarchy lives
    /// (`/sys` in production).
    pub root: PathBuf,
    /// Every logical CPU the machine reports online.
    pub online_cpus: BTreeSet<CpuIndex>,
}

/// The [`SysfsOptions`] describing the whole machine under `root`.
///
/// Fail-loud: a tree without the two `online` files is not a machine whose
/// layout can be read, and answering "one node" for it would be a wrong answer
/// where a missing one is called for.
pub fn machine_sysfs_options(root: &Path) -> Result<SysfsOptions, String> {
    let missing = |rel: &str| format!("cannot read `{}`", root.join(rel).display());
    let cpu_online = "devices/system/cpu/online";
    let node_online = "devices/system/node/online";
    let online = read_sysfs(root, cpu_online).ok_or_else(|| missing(cpu_online))?;
    if read_sysfs(root, node_online).is_none() {
        return Err(missing(node_online));
    }
    let online_cpus: BTreeSet<CpuIndex> = parse_cpu_list(&remove_whitespace(&online))
        .into_iter()
        .collect();
    if online_cpus.is_empty() {
        return Err(format!(
            "`{}` lists no online CPU",
            root.join(cpu_online).display()
        ));
    }
    Ok(SysfsOptions {
        root: root.to_path_buf(),
        online_cpus,
    })
}

/// A machine's NUMA layout, in the form a binary carries it: the ordered CPU
/// list of every node and the system-node number each of them has.
///
/// Resolving the layout once and carrying it as data is what lets a program
/// decide its plan before it runs: the machine is described here, so nothing
/// downstream of it consults `/sys` again.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumaLayout {
    /// The ascending CPU indices of each node, in logical order.
    pub nodes: Vec<Vec<CpuIndex>>,
    /// The kernel's own number for each node, aligned with [`Self::nodes`].
    /// This is what the memory policy is indexed by.
    pub system_nodes: Vec<NumaIndex>,
}

impl NumaLayout {
    /// The layout of the machine `opts` describes.
    ///
    /// Reads `devices/system/node/online` and every node's `cpulist`, keeping
    /// only online CPUs and dropping nodes left empty. A node whose `cpulist`
    /// cannot be read leaves the whole tree untrustworthy, so the result is the
    /// one node holding every online CPU rather than a partial topology.
    pub fn of_machine(opts: &SysfsOptions) -> Self {
        let single = || NumaLayout {
            nodes: vec![opts.online_cpus.iter().copied().collect()],
            system_nodes: vec![0],
        };

        let Some(node_ids) = read_sysfs(&opts.root, "devices/system/node/online") else {
            return single();
        };
        let mut nodes: Vec<Vec<CpuIndex>> = Vec::new();
        let mut system_nodes: Vec<NumaIndex> = Vec::new();
        for n in parse_cpu_list(&remove_whitespace(&node_ids)) {
            let path = format!("devices/system/node/node{n}/cpulist");
            let Some(cpu_ids) = read_sysfs(&opts.root, &path) else {
                return single();
            };
            let cpus: Vec<CpuIndex> = parse_cpu_list(&remove_whitespace(&cpu_ids))
                .into_iter()
                .filter(|c| opts.online_cpus.contains(c))
                .collect();
            if !cpus.is_empty() {
                nodes.push(cpus);
                system_nodes.push(n);
            }
        }
        if nodes.is_empty() {
            return single();
        }
        NumaLayout {
            nodes,
            system_nodes,
        }
    }

    /// Rebuild a layout resolved earlier. Reads nothing, so this is the
    /// constructor a program uses when its layout was decided before the process
    /// started.
    ///
    /// # Panics
    /// Panics on a layout no detection run could have produced: a length
    /// mismatch, an empty node, or a CPU claimed by two nodes. Each would
    /// silently renumber or lose a node, which is the one outcome a compiled-in
    /// layout must not have.
    pub fn from_const(nodes: &[&[CpuIndex]], system_nodes: &[NumaIndex]) -> Self {
        assert_eq!(
            nodes.len(),
            system_nodes.len(),
            "a layout has one system node per node"
        );
        let mut seen: BTreeSet<CpuIndex> = BTreeSet::new();
        for (n, cpus) in nodes.iter().enumerate() {
            assert!(!cpus.is_empty(), "NUMA node {n} of the layout is empty");
            for &c in *cpus {
                assert!(
                    seen.insert(c),
                    "CPU {c} belongs to more than one NUMA node of the layout"
                );
            }
        }
        NumaLayout {
            nodes: nodes.iter().map(|cpus| cpus.to_vec()).collect(),
            system_nodes: system_nodes.to_vec(),
        }
    }

    /// The per-node CPU lists in the borrowed form [`Self::from_const`] takes.
    pub fn node_slices(&self) -> Vec<&[CpuIndex]> {
        self.nodes.iter().map(Vec::as_slice).collect()
    }

    /// The number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// The *system* NUMA node owning `cpu`, if any node does.
    pub fn system_node_of_cpu(&self, cpu: CpuIndex) -> Option<NumaIndex> {
        self.nodes
            .iter()
            .position(|cpus| cpus.contains(&cpu))
            .map(|logical| self.system_nodes[logical])
    }

    /// Every CPU the layout covers, ascending.
    pub fn cpus(&self) -> BTreeSet<CpuIndex> {
        self.nodes.iter().flatten().copied().collect()
    }
}

/// A system L3 cache domain: the CPUs sharing one L3, tagged with the system
/// NUMA node they sit on.
///
/// On a chiplet CPU this is the CCD, which is the granularity a build spreads
/// its workers over.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct L3Domain {
    /// The system NUMA node the domain's CPUs belong to.
    pub system_node: NumaIndex,
    /// The domain's online CPUs, ascending.
    pub cpus: Vec<CpuIndex>,
}

/// The machine's L3 cache domains, ordered by system node and then by their
/// lowest CPU.
///
/// Every online CPU lands in exactly one domain. A CPU whose
/// `cache/index3/shared_cpu_list` cannot be read forms a domain of its own,
/// which is the answer that treats an unreported cache as no sharing rather
/// than as sharing with everything.
pub fn l3_domains(opts: &SysfsOptions, layout: &NumaLayout) -> Vec<L3Domain> {
    let mut domains: Vec<L3Domain> = Vec::new();
    let mut placed: BTreeSet<CpuIndex> = BTreeSet::new();

    for &cpu in &opts.online_cpus {
        if placed.contains(&cpu) {
            continue;
        }
        let path = format!("devices/system/cpu/cpu{cpu}/cache/index3/shared_cpu_list");
        let siblings = read_sysfs(&opts.root, &path).unwrap_or_default();
        let mut cpus: Vec<CpuIndex> = parse_cpu_list(&remove_whitespace(&siblings))
            .into_iter()
            .filter(|c| opts.online_cpus.contains(c) && !placed.contains(c))
            .collect();
        if !cpus.contains(&cpu) {
            cpus.push(cpu);
        }
        cpus.sort_unstable();
        cpus.dedup();
        placed.extend(cpus.iter().copied());
        domains.push(L3Domain {
            system_node: layout.system_node_of_cpu(cpu).unwrap_or(0),
            cpus,
        });
    }

    domains.sort_by_key(|d| (d.system_node, d.cpus[0]));
    domains
}

/// Renders an ascending CPU sequence in the shortened form the sysfs files use:
/// `','` between entries, `"a-b"` for a run of consecutive indices.
///
/// The inverse of [`parse_cpu_list`], so a rendered list parses back to the
/// sequence it came from.
pub fn format_cpu_list(cpus: impl IntoIterator<Item = CpuIndex>) -> String {
    let v: Vec<CpuIndex> = cpus.into_iter().collect();
    let mut out = String::new();
    let mut range_start = 0usize; // index into `v`
    for i in 0..v.len() {
        let at_range_end = i + 1 == v.len() || v[i + 1] != v[i] + 1;
        if !at_range_end {
            continue;
        }
        if range_start != 0 {
            out.push(',');
        }
        if i != range_start {
            let _ = write!(out, "{}-{}", v[range_start], v[i]);
        } else {
            let _ = write!(out, "{}", v[i]);
        }
        range_start = i + 1;
    }
    out
}

/// Expands the shortened index-list syntax into a flat list of indices: `','`
/// separates entries, each either a single index or an inclusive `"a-b"` range.
/// Empty entries are skipped, and an entry that is not an index at all
/// contributes nothing.
pub fn parse_cpu_list(s: &str) -> Vec<CpuIndex> {
    let mut indices = Vec::new();

    if s.is_empty() {
        return indices;
    }

    for ss in s.split(',') {
        if ss.is_empty() {
            continue;
        }

        let parts: Vec<&str> = ss.split('-').collect();
        match parts.as_slice() {
            [single] => {
                if let Some(c) = parse_index(single) {
                    indices.push(c);
                }
            }
            [first, last] => {
                if let (Some(cfirst), Some(clast)) = (parse_index(first), parse_index(last)) {
                    for c in cfirst..=clast {
                        indices.push(c);
                    }
                }
            }
            // Entries with 0 or 3+ dash-separated parts describe no index.
            _ => {}
        }
    }

    indices
}

/// Reads a sysfs file under `root`, returning its contents, or `None` if it
/// cannot be read.
fn read_sysfs(root: &Path, rel: &str) -> Option<String> {
    std::fs::read_to_string(root.join(rel)).ok()
}

/// Removes all ASCII whitespace from `s`.
fn remove_whitespace(s: &str) -> String {
    s.chars().filter(|c| !c.is_ascii_whitespace()).collect()
}

/// Parses a single decimal index, tolerating surrounding whitespace and
/// trailing non-digits. Returns `None` when no leading digits are present.
fn parse_index(s: &str) -> Option<CpuIndex> {
    let t = s.trim_start();
    let digits: String = t.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    digits.parse::<CpuIndex>().ok()
}

/// The number of usable hardware threads, at least 1.
pub fn system_threads() -> CpuIndex {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .max(1)
}

/// The set of CPUs the process was allowed to run on **at startup**, captured
/// once.
///
/// A deliberate startup snapshot, so a check does not change its answer as the
/// live affinity changes over time. On non-Linux targets it degrades to all
/// system threads.
pub fn startup_affinity() -> &'static BTreeSet<CpuIndex> {
    static STARTUP: OnceLock<BTreeSet<CpuIndex>> = OnceLock::new();
    STARTUP.get_or_init(capture_process_affinity)
}

#[cfg(target_os = "linux")]
fn capture_process_affinity() -> BTreeSet<CpuIndex> {
    // A fixed 1024-CPU `cpu_set_t`. A machine that exceeds it fails loud at pin
    // time rather than silently mis-pinning.
    let mut cpus = BTreeSet::new();
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        let size = std::mem::size_of::<libc::cpu_set_t>();
        let status = libc::sched_getaffinity(0, size, &mut set as *mut libc::cpu_set_t);
        if status != 0 {
            // Soft error: assume all system threads are available rather than
            // aborting the process from within a library.
            return (0..system_threads()).collect();
        }
        for c in 0..(size * 8) {
            if libc::CPU_ISSET(c, &set) {
                cpus.insert(c);
            }
        }
    }
    cpus
}

#[cfg(not(target_os = "linux"))]
fn capture_process_affinity() -> BTreeSet<CpuIndex> {
    (0..system_threads()).collect()
}

/// Pin the *current* thread to the single logical CPU `cpu`, so nothing else
/// about its affinity remains. A no-op on non-Linux targets.
///
/// # Panics
/// Fail-loud:
/// * if `cpu >= 1024` — the fixed `cpu_set_t` this crate uses cannot name it,
///   and a CPU index that would not fit is rejected rather than truncated;
/// * if `sched_setaffinity` fails, which is what a process denied that CPU
///   produces.
pub fn pin_current_thread_to_cpu(cpu: CpuIndex) {
    pin_current_thread(cpu);
}

/// Pin the current thread to `cpu` **and** point its private allocations at
/// `system_node` — the pin-then-prefer pair every worker runs once, right after
/// it is spawned.
///
/// The pin alone only constrains where the thread *runs*: under
/// `numactl --interleave=all` the inherited process policy still spreads
/// everything it allocates across all nodes.
/// [`mempolicy::set_current_thread_preferred_node`] closes that gap, and being
/// per-thread it cannot perturb the shared transposition table's interleave.
///
/// The pin is fail-loud; the memory policy is best-effort, and returning `false`
/// merely leaves today's placement in force.
pub fn pin_current_thread_to_cpu_with_local_memory(cpu: CpuIndex, system_node: NumaIndex) -> bool {
    pin_current_thread_to_cpu(cpu);
    mempolicy::set_current_thread_preferred_node(system_node)
}

/// Run `f` on a temporary thread pinned to `cpu`, then join it, so that an
/// allocation `f` makes and faults is placed on that CPU's node by the kernel's
/// first-touch policy.
///
/// A scoped thread, so `f`'s captures may be borrowed for the duration and it
/// can write its result back into a caller-owned slot. On non-Linux targets the
/// pin is a no-op but the closure still runs on the temporary thread, so the
/// control flow is identical across platforms.
pub fn execute_on_cpu<F>(cpu: CpuIndex, f: F)
where
    F: FnOnce() + Send,
{
    std::thread::scope(|scope| {
        scope.spawn(|| {
            pin_current_thread_to_cpu(cpu);
            f();
        });
    });
}

/// The Linux affinity-setting core of [`pin_current_thread_to_cpu`].
///
/// The trailing `sched_yield` is a defensive re-schedule, so the thread lands on
/// the newly-allowed CPU promptly.
#[cfg(target_os = "linux")]
fn pin_current_thread(cpu: CpuIndex) {
    assert!(
        cpu < 1024,
        "pin_current_thread_to_cpu: CPU index {cpu} exceeds this crate's fixed \
         1024-CPU cpu_set_t capacity"
    );
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(cpu, &mut set);
        let size = std::mem::size_of::<libc::cpu_set_t>();
        let status = libc::sched_setaffinity(0, size, &set as *const libc::cpu_set_t);
        if status != 0 {
            panic!(
                "pin_current_thread_to_cpu: sched_setaffinity({cpu}) failed: {}",
                std::io::Error::last_os_error()
            );
        }
        // Defensive re-schedule; allowed because this is not performance critical.
        libc::sched_yield();
    }
}

/// Non-Linux no-op counterpart of [`pin_current_thread`]: this port does not pin
/// threads off Linux.
#[cfg(not(target_os = "linux"))]
fn pin_current_thread(_cpu: CpuIndex) {}

#[cfg(test)]
mod tests {
    use super::*;

    // -- shortened-list parsing -------------------------------------------

    #[test]
    fn parse_simple_list_and_range() {
        assert_eq!(parse_cpu_list("0-3,8"), vec![0, 1, 2, 3, 8]);
        assert_eq!(parse_cpu_list("5"), vec![5]);
        assert_eq!(parse_cpu_list("2-2"), vec![2]);
    }

    #[test]
    fn parse_empty_and_empty_entries() {
        assert_eq!(parse_cpu_list(""), Vec::<CpuIndex>::new());
        // Empty entries between commas are skipped.
        assert_eq!(parse_cpu_list("0,,3"), vec![0, 3]);
    }

    #[test]
    fn parse_tolerates_whitespace() {
        // sysfs content is passed through `remove_whitespace` first.
        assert_eq!(
            parse_cpu_list(&remove_whitespace(" 0-3 , 8 \n")),
            vec![0, 1, 2, 3, 8]
        );
        assert_eq!(parse_cpu_list("0-3\n"), vec![0, 1, 2, 3]);
    }

    #[test]
    fn descending_range_is_empty() {
        assert_eq!(parse_cpu_list("5-3"), Vec::<CpuIndex>::new());
    }

    // -- CPU-list rendering -----------------------------------------------

    #[test]
    fn cpu_list_rendering_compresses_runs() {
        assert_eq!(format_cpu_list([0, 1, 2, 3, 8]), "0-3,8");
        assert_eq!(format_cpu_list([5]), "5");
        assert_eq!(format_cpu_list([]), "");
        assert_eq!(format_cpu_list([1, 3, 5]), "1,3,5");
        // The inverse of the parser it renders for.
        assert_eq!(
            parse_cpu_list(&format_cpu_list([0, 1, 2, 7, 8])),
            vec![0, 1, 2, 7, 8]
        );
    }

    // -- the layout a binary carries --------------------------------------

    #[test]
    fn from_const_rebuilds_the_nodes_and_the_system_map() {
        let layout = NumaLayout::from_const(&[&[0, 1, 2, 3], &[8, 9]], &[0, 2]);
        assert_eq!(layout.num_nodes(), 2);
        assert_eq!(layout.nodes[0], vec![0, 1, 2, 3]);
        assert_eq!(layout.system_nodes, vec![0, 2]);
        assert_eq!(layout.system_node_of_cpu(9), Some(2));
        assert_eq!(layout.system_node_of_cpu(3), Some(0));
        assert_eq!(layout.system_node_of_cpu(7), None);
        assert_eq!(layout.cpus().len(), 6);
    }

    #[test]
    #[should_panic(expected = "is empty")]
    fn from_const_rejects_an_empty_node() {
        NumaLayout::from_const(&[&[0, 1], &[], &[2]], &[0, 1, 2]);
    }

    #[test]
    #[should_panic(expected = "more than one NUMA node")]
    fn from_const_rejects_a_repeated_cpu() {
        NumaLayout::from_const(&[&[0, 1], &[1, 2]], &[0, 1]);
    }

    #[test]
    #[should_panic(expected = "one system node per node")]
    fn from_const_rejects_a_length_mismatch() {
        NumaLayout::from_const(&[&[0, 1], &[2]], &[0]);
    }

    // -- pinning ----------------------------------------------------------

    #[cfg(target_os = "linux")]
    fn current_thread_affinity() -> BTreeSet<CpuIndex> {
        let mut cpus = BTreeSet::new();
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            let size = std::mem::size_of::<libc::cpu_set_t>();
            let status = libc::sched_getaffinity(0, size, &mut set as *mut libc::cpu_set_t);
            assert_eq!(status, 0, "sched_getaffinity failed in test");
            for c in 0..(size * 8) {
                if libc::CPU_ISSET(c, &set) {
                    cpus.insert(c);
                }
            }
        }
        cpus
    }

    /// One CPU this process is certainly allowed on, so the fail-loud pin cannot
    /// hit a forbidden one.
    #[cfg(target_os = "linux")]
    fn an_allowed_cpu() -> CpuIndex {
        current_thread_affinity()
            .into_iter()
            .next()
            .expect("the test thread must have >= 1 CPU")
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn pinning_leaves_exactly_one_cpu_allowed() {
        // Run in a spawned thread so the test harness's own affinity is never
        // perturbed.
        let cpu = an_allowed_cpu();
        std::thread::spawn(move || {
            pin_current_thread_to_cpu(cpu);
            assert_eq!(current_thread_affinity(), BTreeSet::from([cpu]));
        })
        .join()
        .expect("pin test thread must not panic");
    }

    /// The pin-and-place pair: the affinity half is fail-loud and observable via
    /// `sched_getaffinity`, the memory half is best-effort and observable via
    /// `get_mempolicy`. On a single-node host the *placement* is node 0 either
    /// way, but the thread's *policy* is whatever it inherited until this call
    /// changes it.
    #[cfg(target_os = "linux")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn pin_with_local_memory_sets_both_the_affinity_and_the_policy() {
        let cpu = an_allowed_cpu();
        std::thread::spawn(move || {
            let took = pin_current_thread_to_cpu_with_local_memory(cpu, 0);
            assert_eq!(
                current_thread_affinity(),
                BTreeSet::from([cpu]),
                "the pin half is unconditional"
            );
            if took {
                let policy = mempolicy::current_thread_policy()
                    .expect("get_mempolicy after a successful set");
                assert_eq!(policy.mode, mempolicy::MODE_PREFERRED);
                assert_eq!(policy.nodes, vec![0]);
            }
        })
        .join()
        .expect("pin test thread must not panic");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn execute_on_cpu_runs_the_closure_pinned() {
        let cpu = an_allowed_cpu();
        let mut observed: BTreeSet<CpuIndex> = BTreeSet::new();
        execute_on_cpu(cpu, || observed = current_thread_affinity());
        assert_eq!(observed, BTreeSet::from([cpu]));
    }
}
