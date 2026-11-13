// The CPU-assignment half of `build.rs`: decide which logical CPU each worker
// thread is pinned to, and record the choice in the machine-local ledger so the
// next build on the same machine picks different CPUs.
//
// One worker, one CPU. `cpu_assignment = "auto"` picks them round-robin over the
// machine's L3 domains, skipping every CPU the ledger says an earlier build
// took; an explicit CPU list names them outright and touches no ledger, so
// whoever writes such a list is the one keeping it clear of the other builds on
// the machine. Either way the choice is made once, when the binary is built, and
// nothing at run time revisits it. `"none"` is the third form: no worker is
// pinned and no ledger is touched, so any number of processes of the one binary
// can share a machine, each placed by whatever runs it.
//
// Split from `build_config.rs` because this is the part that consults the
// machine and the ledger rather than the config file, and split from
// `build_numa.rs` because exactly one build script per binary may take CPUs —
// the crates that only read the same config file must not.

use std::collections::BTreeSet;
use std::fs::OpenOptions;
use std::io::{Read as _, Seek as _, SeekFrom, Write as _};

// `NumaLayout` is spelled out at each use, and `format_cpu_list` is the helper
// beside it: `build.rs` includes this file beside `build_numa.rs`, which
// imports the one and defines the other.
use yorkie_numa::{CpuIndex, L3Domain, NumaIndex};

/// The machine a build assigns CPUs on: every online CPU, the L3 domains they
/// are grouped into, and the NUMA layout that says which node each one is on.
struct Machine {
    online: BTreeSet<CpuIndex>,
    domains: Vec<L3Domain>,
    layout: yorkie_numa::NumaLayout,
}

impl Machine {
    /// The CPU the machine's non-search work is kept on: its lowest-numbered
    /// online CPU.
    ///
    /// This is the one place the rule is written down. A build with
    /// `low_priority_worker = true` puts its low-priority worker here, so that
    /// worker is the only one sharing a CPU with anything; a host that keeps
    /// its other processes off the workers' CPUs keeps them on this one.
    fn shared_cpu(&self) -> CpuIndex {
        self.online.first().copied().unwrap_or(0)
    }
}

/// The `cpu_assignment` value that pins no worker.
const UNPINNED: &str = "none";

/// Move the machine's [shared CPU](Machine::shared_cpu) to the end of an
/// explicit list, so the last worker — the low-priority one, never the main
/// worker — is the one pinned to it.
///
/// The list has to name that CPU: the low-priority worker exists to give way to
/// whatever else runs there, and lowering a worker on any other CPU would slow
/// the search down for nothing.
fn shared_cpu_last(cpus: &mut Vec<CpuIndex>, machine: &Machine) -> Result<(), String> {
    let shared = machine.shared_cpu();
    let Some(at) = cpus.iter().position(|&c| c == shared) else {
        return Err(format!(
            "`low_priority_worker` = true puts the low-priority worker on CPU {shared}, the \
             machine's lowest online CPU and the one its non-search threads share, and \
             `cpu_assignment` does not name it — add CPU {shared} to the list, or set \
             `low_priority_worker` = false"
        ));
    };
    let cpu = cpus.remove(at);
    cpus.push(cpu);
    Ok(())
}

/// Which build a ledger line belongs to.
///
/// Two builds writing into different directories are different binaries even
/// when they read the same config, so each gets its own CPUs; a rebuild of the
/// same one keeps the CPUs it already has, so rebuilding never eats into the
/// machine.
#[derive(Debug, Clone, PartialEq, Eq)]
struct BuildIdentity {
    /// The directory the built binary is written to.
    out_dir: String,
    /// The config file the build read.
    config: String,
    /// The crate version the build carries.
    version: String,
}

/// One recorded allocation: the CPUs a build holds, and which build holds them.
#[derive(Debug, Clone, PartialEq, Eq)]
struct LedgerEntry {
    cpus: Vec<CpuIndex>,
    identity: BuildIdentity,
}

/// The comment a freshly created ledger opens with, so the file explains itself
/// to whoever opens it.
const LEDGER_HEADER: &str = "\
# Logical CPUs each build here has taken, one build per line:
#
#     <cpu list>\\t<output directory>\\t<config>\\t<version>
#
# A build takes the CPUs its workers are pinned to and leaves them to itself;
# rebuilding the same binary keeps its line. Delete this file to hand every CPU
# back and start the allocation over.
";

/// The field separator of a ledger line.
const LEDGER_SEP: char = '\t';

/// Parse a ledger file. Blank lines and `#` comments are skipped; anything else
/// must be a well-formed line, since a line that cannot be read is a build
/// holding CPUs this build must not take.
fn parse_ledger(text: &str, label: &str) -> Result<Vec<LedgerEntry>, String> {
    let mut entries = Vec::new();
    for (idx, raw) in text.lines().enumerate() {
        let line = raw.trim_end();
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split(LEDGER_SEP).collect();
        let [cpus, out_dir, config, version] = fields.as_slice() else {
            return Err(format!(
                "{label}:{}: expected four tab-separated fields, found {}",
                idx + 1,
                fields.len()
            ));
        };
        let cpus = parse_explicit_cpus(cpus)
            .map_err(|e| format!("{label}:{}: {e}", idx + 1))?;
        if cpus.is_empty() {
            return Err(format!("{label}:{}: names no CPU", idx + 1));
        }
        entries.push(LedgerEntry {
            cpus,
            identity: BuildIdentity {
                out_dir: (*out_dir).to_string(),
                config: (*config).to_string(),
                version: (*version).to_string(),
            },
        });
    }
    Ok(entries)
}

/// Render a ledger file: the header, then one line per entry.
fn render_ledger(entries: &[LedgerEntry]) -> String {
    let mut out = String::from(LEDGER_HEADER);
    for e in entries {
        let _ = writeln!(
            out,
            "{}{LEDGER_SEP}{}{LEDGER_SEP}{}{LEDGER_SEP}{}",
            format_cpu_list(e.cpus.iter().copied()),
            e.identity.out_dir,
            e.identity.config,
            e.identity.version
        );
    }
    out
}

/// Parse an explicit CPU list — `0,2,4-7` — refusing anything that is not one,
/// rather than silently dropping the part it cannot read.
fn parse_explicit_cpus(spec: &str) -> Result<Vec<CpuIndex>, String> {
    let mut cpus = Vec::new();
    for token in spec.split(',') {
        let token = token.trim();
        if token.is_empty() {
            return Err(format!("`{spec}` has an empty entry"));
        }
        let parts: Vec<&str> = token.split('-').collect();
        match parts.as_slice() {
            [one] => cpus.push(cpu_index(one)?),
            [low, high] => {
                let (low, high) = (cpu_index(low)?, cpu_index(high)?);
                if high < low {
                    return Err(format!("`{token}` counts down"));
                }
                cpus.extend(low..=high);
            }
            _ => return Err(format!("`{token}` is not a CPU or a CPU range")),
        }
    }
    Ok(cpus)
}

fn cpu_index(s: &str) -> Result<CpuIndex, String> {
    s.parse::<CpuIndex>()
        .map_err(|_| format!("`{s}` is not a CPU index"))
}

/// The CPUs `threads` workers take under `"auto"`: round-robin over the L3
/// domains, each domain's free CPUs in ascending order, skipping every CPU
/// `taken` holds.
///
/// One worker lands on the first free CPU of the first domain, the next on the
/// first free CPU of the second, and so on, wrapping around — so a build asking
/// for many workers spreads them evenly over the chiplets, and a machine's worth
/// of single-worker builds fills one domain after another rather than crowding
/// the first.
///
/// `Err` carries how many CPUs were actually free, which is what a shortfall has
/// to say.
fn round_robin_cpus(
    domains: &[L3Domain],
    taken: &BTreeSet<CpuIndex>,
    threads: usize,
) -> Result<Vec<CpuIndex>, usize> {
    let free: Vec<Vec<CpuIndex>> = domains
        .iter()
        .map(|d| {
            d.cpus
                .iter()
                .copied()
                .filter(|c| !taken.contains(c))
                .collect()
        })
        .collect();
    let available: usize = free.iter().map(Vec::len).sum();
    if available < threads {
        return Err(available);
    }

    let mut cursors = vec![0usize; free.len()];
    let mut picked = Vec::with_capacity(threads);
    let mut domain = 0usize;
    while picked.len() < threads {
        if let Some(&cpu) = free[domain].get(cursors[domain]) {
            cursors[domain] += 1;
            picked.push(cpu);
        }
        domain = (domain + 1) % free.len();
    }
    Ok(picked)
}

/// The CPUs an explicit `cpu_assignment` names, held against the machine alone:
/// exactly `threads` of them, every one online, none named twice.
///
/// What another build on the machine holds is deliberately not consulted. A list
/// says which CPUs this binary runs on, and the operator who wrote it is the one
/// deciding how the machine is divided.
fn explicit_cpus(
    spec: &str,
    machine: &Machine,
    threads: usize,
) -> Result<Vec<CpuIndex>, String> {
    let cpus = parse_explicit_cpus(spec).map_err(|e| format!("`cpu_assignment`: {e}"))?;
    if cpus.len() != threads {
        return Err(format!(
            "`cpu_assignment` names {} CPU(s) but `threads` is {threads} — a build pins one \
             worker to one CPU, so the list has exactly as many entries as there are workers",
            cpus.len()
        ));
    }
    let mut seen = BTreeSet::new();
    for &cpu in &cpus {
        if !seen.insert(cpu) {
            return Err(format!("`cpu_assignment` names CPU {cpu} twice"));
        }
        if !machine.online.contains(&cpu) {
            return Err(format!(
                "`cpu_assignment` names CPU {cpu}, which the machine does not report online \
                 (online: {})",
                format_cpu_list(machine.online.iter().copied())
            ));
        }
    }
    Ok(cpus)
}

/// The CPUs this build's workers are pinned to.
///
/// Under `"auto"` the pick comes out of `ledger_path`, under an exclusive lock
/// held for the whole read-decide-write cycle so two builds on one machine
/// cannot pick the same CPU. A build already in the ledger keeps its line: the
/// CPUs it holds are recomputed against the same free set the first build saw,
/// so rebuilding is idempotent and eats nothing.
///
/// An explicit list is the other case, and it neither opens nor writes the
/// ledger: the CPUs are the ones written down, held against the machine only.
/// `"none"` takes no CPU at all and leaves the ledger alone too.
fn take_cpus(
    request: &AssignmentRequest,
    machine: &Machine,
    ledger_path: &Path,
    identity: &BuildIdentity,
) -> Result<Vec<CpuIndex>, String> {
    let threads = usize::try_from(request.threads).unwrap_or(usize::MAX);
    if threads > machine.online.len() {
        return Err(format!(
            "`threads` = {threads} exceeds the {} logical CPU(s) the machine reports online — \
             a build pins one worker to one CPU, so it cannot ask for more workers than the \
             machine has",
            machine.online.len()
        ));
    }

    if request.low_priority && threads < 2 {
        return Err(format!(
            "`low_priority_worker` = true lowers the last worker and never the main one, so it \
             needs `threads` of at least 2; this config has {threads}"
        ));
    }

    if request.spec == UNPINNED {
        return Ok(Vec::new());
    }
    if request.spec != "auto" {
        let mut cpus = explicit_cpus(request.spec, machine, threads)?;
        if request.low_priority {
            shared_cpu_last(&mut cpus, machine)?;
        }
        return Ok(cpus);
    }

    let label = ledger_path.display().to_string();
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(ledger_path)
        .map_err(|e| format!("cannot open the CPU ledger `{label}`: {e}"))?;
    let _lock = LedgerLock::acquire(&file, &label)?;

    let mut text = String::new();
    file.read_to_string(&mut text)
        .map_err(|e| format!("cannot read the CPU ledger `{label}`: {e}"))?;
    let mut entries = parse_ledger(&text, &label)?;

    let mine = entries.iter().position(|e| &e.identity == identity);
    let taken: BTreeSet<CpuIndex> = entries
        .iter()
        .enumerate()
        .filter(|(i, _)| Some(*i) != mine)
        .flat_map(|(_, e)| e.cpus.iter().copied())
        .collect();

    // The low-priority worker's CPU is fixed rather than picked: it is the one
    // the machine's other work shares, so the pick for the full-priority
    // workers goes round it and that worker comes last.
    let shared = request.low_priority.then(|| machine.shared_cpu());
    let mut excluded = taken;
    if let Some(cpu) = shared {
        if excluded.contains(&cpu) {
            return Err(format!(
                "`low_priority_worker` = true puts the low-priority worker on CPU {cpu}, the \
                 machine's lowest online CPU and the one its non-search threads share, and \
                 `{label}` records CPU {cpu} as another build's. Delete that build's line, or \
                 set `low_priority_worker` = false"
            ));
        }
        excluded.insert(cpu);
    }
    let picked = threads - usize::from(shared.is_some());
    let mut cpus = round_robin_cpus(&machine.domains, &excluded, picked).map_err(|free| {
        format!(
            "`cpu_assignment` = \"auto\" needs {picked} free CPU(s){} and `{label}` leaves \
             {free}: {} short. Delete the lines of the builds that no longer run, or the \
             whole file to start the allocation over",
            shared.map_or(String::new(), |cpu| format!(
                " besides CPU {cpu}, the low-priority worker's"
            )),
            picked - free
        )
    })?;
    cpus.extend(shared);

    let entry = LedgerEntry {
        cpus: cpus.clone(),
        identity: identity.clone(),
    };
    match mine {
        Some(i) => entries[i] = entry,
        None => entries.push(entry),
    }

    let rendered = render_ledger(&entries);
    file.set_len(0)
        .and_then(|()| file.seek(SeekFrom::Start(0)))
        .and_then(|_| file.write_all(rendered.as_bytes()))
        .map_err(|e| format!("cannot write the CPU ledger `{label}`: {e}"))?;
    Ok(cpus)
}

/// An exclusive `flock` on the ledger, released when the guard drops.
struct LedgerLock {
    fd: std::os::fd::RawFd,
}

impl LedgerLock {
    fn acquire(file: &std::fs::File, label: &str) -> Result<Self, String> {
        use std::os::fd::AsRawFd as _;
        let fd = file.as_raw_fd();
        // SAFETY: `fd` is the open ledger's descriptor, valid for as long as
        // the caller's `File` lives, which outlives this guard.
        if unsafe { libc::flock(fd, libc::LOCK_EX) } != 0 {
            return Err(format!(
                "cannot lock the CPU ledger `{label}`: {}",
                std::io::Error::last_os_error()
            ));
        }
        Ok(LedgerLock { fd })
    }
}

impl Drop for LedgerLock {
    fn drop(&mut self) {
        // SAFETY: the descriptor is still open — the `File` the lock was taken
        // on outlives this guard.
        unsafe { libc::flock(self.fd, libc::LOCK_UN) };
    }
}

/// The generated items naming the assigned CPUs and the node each one sits on.
///
/// An unpinned build has no CPUs to name, so `cpus` is empty while
/// `system_nodes` still holds one node per worker.
fn render_cpu_assignment(
    cpus: &[CpuIndex],
    system_nodes: &[NumaIndex],
    shared: Option<CpuIndex>,
) -> String {
    let list = |v: &[usize]| {
        v.iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    };
    format!(
        "
/// Whether every worker thread pins itself to a CPU of its own. `false` only
/// under `cpu_assignment = \"none\"`, where [`WORKER_CPUS`] is empty: no
/// worker is pinned, nothing places memory on a node, and `isready` asks
/// nothing of the process's affinity.
pub const WORKERS_PINNED: bool = {pinned};

/// The logical CPU every worker thread pins itself to, indexed by worker id:
/// `{shape}`.
///
/// Chosen when this binary was built, so nothing at run time decides where a
/// worker runs. Worker 0 is the search coordinator. With
/// `low_priority_worker` on, the last entry is the machine's lowest online
/// CPU, which its non-search threads share with the low-priority worker.
pub const WORKER_CPUS: &[usize] = &[{cpus}];

/// The CPU the engine's own non-search threads (the USI reader and the threads
/// it starts) are pinned to: the building machine's lowest online CPU, the one
/// a host keeps its other work on. `None` in a build that pins no worker.
pub const SHARED_CPU: Option<usize> = {shared};

/// The system NUMA node of each worker's CPU, one per worker, indexed like
/// [`WORKER_CPUS`].
///
/// This is what the per-worker memory placement and the shared tables are keyed
/// by, and `isready` holds it against the running machine before a game. An
/// unpinned build names the machine's first node for every worker: the node
/// whose evaluation region and shared tables they all use, with nothing placing
/// memory there.
pub const WORKER_SYSTEM_NODES: &[usize] = &[{nodes}];
{regions}",
        pinned = !cpus.is_empty(),
        shape = if cpus.is_empty() {
            "none".to_string()
        } else {
            let mut sorted = cpus.to_vec();
            sorted.sort_unstable();
            format_cpu_list(sorted)
        },
        cpus = list(cpus),
        shared = shared.map_or_else(|| "None".to_string(), |c| format!("Some({c})")),
        nodes = list(system_nodes),
        regions = render_eval_regions(system_nodes),
    )
}

/// The generated items pairing each system NUMA node a worker sits on with the
/// evaluation region that node's workers read.
///
/// A worker's node is known when the binary is built, so which region it reads
/// is too, and so is the network type that addresses it. What the macro
/// produces is the one place the node number turns into that type: a `match`
/// over a constant, run once when a worker thread is spawned and once when a
/// `go` starts its coordinator, and never again. Every arm names a distinct
/// `Region<_>`, so the search is compiled once per region the machine needs and
/// no evaluation loads a base address from anywhere.
fn render_eval_regions(system_nodes: &[NumaIndex]) -> String {
    let mut distinct: Vec<NumaIndex> = system_nodes.to_vec();
    distinct.sort_unstable();
    distinct.dedup();

    let nodes = distinct
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let arms = distinct
        .iter()
        .enumerate()
        .map(|(slot, node)| {
            format!(
                "            {node}usize => {{\n                \
                 let $net = ::yorkie_eval::Region::<{slot}usize>::new();\n                \
                 $body\n            }}\n"
            )
        })
        .collect::<String>();

    format!(
        "
/// The system NUMA nodes whose workers read a network of their own, ascending:
/// evaluation region `i` is the one node `EVAL_REGION_NODES[i]`'s workers read.
///
/// One entry on a machine whose workers all sit on one node, which is every
/// single-node machine; otherwise one per node the assignment spreads onto.
pub const EVAL_REGION_NODES: &[usize] = &[{nodes}];

/// Bind `$net` to the network a worker on system NUMA node `$node` reads, and
/// run `$body` with it.
///
/// `$node` is one of [`EVAL_REGION_NODES`] — every worker's node is, since the
/// assignment is what that list is built from — and an unknown one is a
/// contradiction of the constants rather than a case to handle.
macro_rules! with_eval_network {{
    ($node:expr, |$net:ident| $body:expr) => {{
        match $node {{
{arms}            other => unreachable!(
                \"no worker sits on system NUMA node {{other}}: this binary was built for {nodes}\"
            ),
        }}
    }};
}}
pub(crate) use with_eval_network;"
    )
}

/// Resolve `cpu_assignment` against the building machine and its ledger, and
/// render the constants naming the result.
fn resolve_cpu_assignment(
    request: &AssignmentRequest,
    identity: &BuildIdentity,
    repo_root: &Path,
) -> Result<ResolvedAssignment, String> {
    let mut ledger = PathBuf::from(request.ledger);
    if request.spec == "auto" {
        if ledger.as_os_str().is_empty() {
            return Err(
                "`cpu_assignment` = \"auto\" needs a ledger to pick from, and `cpu_ledger` is \
                 empty. Name a file for `cpu_ledger`, or name this build's CPUs outright as a \
                 list such as \"0,2,4-7\", which takes nothing from any ledger"
                    .to_string(),
            );
        }
        if ledger.is_relative() {
            ledger = repo_root.join(ledger);
        }
    }

    let opts = building_machine()?;
    let layout = yorkie_numa::NumaLayout::of_machine(&opts);
    let machine = Machine {
        online: opts.online_cpus.clone(),
        domains: yorkie_numa::l3_domains(&opts, &layout),
        layout,
    };

    let cpus = take_cpus(request, &machine, &ledger, identity)?;
    let system_nodes: Vec<NumaIndex> = if request.spec == UNPINNED {
        let first = machine.layout.system_nodes.first().copied().unwrap_or(0);
        vec![first; usize::try_from(request.threads).unwrap_or(1).max(1)]
    } else {
        cpus.iter()
            .map(|&cpu| machine.layout.system_node_of_cpu(cpu).unwrap_or(0))
            .collect()
    };
    Ok(ResolvedAssignment {
        items: render_cpu_assignment(
            &cpus,
            &system_nodes,
            (request.spec != UNPINNED).then(|| machine.shared_cpu()),
        ),
    })
}
