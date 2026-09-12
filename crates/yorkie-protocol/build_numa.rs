// The NUMA half of `build.rs`: read the building machine's layout, and render it
// as the constants the engine places its memory from.
//
// Split from `build_config.rs` because this is the only part of the config
// pipeline that consults the machine rather than the file, and the only part
// that needs the `yorkie-numa` crate — which the crate compiling the layout in
// takes as a build dependency, while the other crates reading the same config do
// not.

use yorkie_numa::{NumaLayout, machine_sysfs_options};

/// The sysfs root a build reads the machine from.
const SYSFS_ROOT: &str = "/sys";

/// A CPU list in the shortened sysfs form, as the string this script renders
/// into the generated items.
///
/// `yorkie_numa` hands the list over as byte fragments, because the engine
/// composes it into a buffer it owns; a build script is free to gather them.
fn format_cpu_list(cpus: impl IntoIterator<Item = usize>) -> String {
    let mut out = Vec::new();
    yorkie_numa::write_cpu_list(cpus, |fragment| out.extend_from_slice(fragment));
    String::from_utf8(out).expect("a CPU list is ASCII")
}

/// The machine this build runs on, or the reason it cannot be read.
///
/// Every online CPU, rather than the CPUs this build process happens to be
/// allowed on, because what is compiled in describes the machine. A build
/// confined to part of it still yields the layout of the whole; a *run* so
/// confined is what the engine's startup check catches.
fn building_machine() -> Result<yorkie_numa::SysfsOptions, String> {
    machine_sysfs_options(Path::new(SYSFS_ROOT)).map_err(|e| {
        let mut out = Vec::new();
        e.write_message(|fragment| out.extend_from_slice(fragment));
        String::from_utf8(out).expect("a refusal is ASCII")
    })
}

/// Resolve the building machine's NUMA layout, and render it.
fn resolve_numa_layout() -> Result<ResolvedLayout, String> {
    let layout = NumaLayout::of_machine(&building_machine()?);
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
/// The system CPU indices of every NUMA node, in node order: `{shape}`.
///
/// Resolved from the machine this binary was built on. The engine pins its
/// workers to CPUs of these nodes and places its memory on them, so a machine
/// laid out differently is one it refuses to play on: `isready` reports the
/// difference and withholds `readyok`.
pub const NUMA_NODE_CPUS: &[&[usize]] = &[{node_cpus}];

/// The kernel's own number for each node, aligned with [`NUMA_NODE_CPUS`].
///
/// The node numbers the machine reports need not be contiguous, and it is this
/// map — not the position in the table — that the per-worker memory placement
/// and the network replica set are keyed by.
pub const NUMA_SYSTEM_NODES: &[usize] = &[{system_nodes}];",
        shape = shape.join(":"),
        node_cpus = node_cpus.join(", "),
        system_nodes = system_nodes.join(", "),
    )
}
