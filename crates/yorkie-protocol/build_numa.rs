// The NUMA half of `build.rs`: read the building machine's layout, and render
// it as the constants the engine plans its worker binding from.
//
// Split from `build_config.rs` because this is the only part of the config
// pipeline that consults the machine rather than the file, and the only part
// that needs the `yorkie-numa` crate — which the crate compiling the layout in
// takes as a build dependency, while the other crates reading the same config
// do not.

use yorkie_numa::{NumaConfig, NumaLayout, format_cpu_list, machine_sysfs_options};

/// The sysfs root a build reads the machine's layout from.
const SYSFS_ROOT: &str = "/sys";

/// Resolve the building machine's logical NUMA layout under `policy`, and
/// render it.
///
/// The resolution is the one the engine would otherwise have performed at
/// startup, taken over the whole machine: every online CPU, rather than the CPUs
/// this build process happens to be allowed on, because what is compiled in
/// describes the machine. A build confined to part of it still yields the layout
/// of the whole; a *run* so confined is what the engine's startup check catches.
fn resolve_numa_layout(policy: &str) -> Result<ResolvedLayout, String> {
    let opts = machine_sysfs_options(Path::new(SYSFS_ROOT))?;
    let config = NumaConfig::from_policy(policy, &opts)?;
    let layout = NumaLayout::of(&config, &opts);
    Ok(ResolvedLayout {
        nodes: layout.nodes.len(),
        items: render_numa_layout(&layout),
    })
}

/// The generated items describing `layout`, without a trailing newline.
fn render_numa_layout(layout: &NumaLayout) -> String {
    let node_cpus: Vec<String> = layout
        .nodes
        .iter()
        .map(|cpus| {
            let list: Vec<String> = cpus.iter().map(usize::to_string).collect();
            format!("&[{}]", list.join(", "))
        })
        .collect();
    let system_nodes: Vec<String> = layout.system_nodes.iter().map(usize::to_string).collect();
    let shape: Vec<String> = layout
        .nodes
        .iter()
        .map(|cpus| format_cpu_list(cpus.iter().copied()))
        .collect();

    format!(
        "
/// The system CPU indices of every logical NUMA node, in node order: `{shape}`.
///
/// Resolved from the machine this binary was built on. The engine distributes
/// and binds its workers over these, so a machine laid out differently is one it
/// refuses to play on: `isready` reports the difference and withholds `readyok`.
pub const NUMA_NODE_CPUS: &[&[usize]] = &[{node_cpus}];

/// The *system* NUMA node each logical node belongs to, aligned with
/// [`NUMA_NODE_CPUS`].
///
/// Coarser than the logical node wherever the mapping policy subdivides a system
/// node by L3 domain, and it is this map — not the logical one — that the
/// per-worker memory placement and the network replica set are keyed by.
pub const NUMA_SYSTEM_NODES: &[usize] = &[{system_nodes}];

/// Whether the resolved layout may not match the CPUs the process is allowed to
/// run on, which makes every worker bind and every node hold its own memory.
pub const NUMA_CUSTOM_AFFINITY: bool = {custom};",
        shape = shape.join(":"),
        node_cpus = node_cpus.join(", "),
        system_nodes = system_nodes.join(", "),
        custom = layout.custom_affinity,
    )
}
