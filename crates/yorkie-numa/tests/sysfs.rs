//! Fixture-backed tests for the sysfs-driven detection path, plus a best-effort
//! smoke test against the live `/sys` tree on Linux.
//!
//! Each fixture is a committed miniature sysfs tree under `tests/fixtures/`; the
//! injectable [`SysfsOptions::root`] points at one, so no test touches the real
//! machine topology (except the explicitly Linux-gated smoke test at the end).
//!
//! Every test here is `#[cfg_attr(miri, ignore)]`: reading a fixture is a real
//! `std::fs` call, which miri's isolation rejects outright. The parsing and
//! topology logic these fixtures drive has no unsafe code, and the crate's unit
//! tests take their sysfs contents as in-memory strings, so they keep running
//! under miri.

use std::collections::BTreeSet;
use std::path::PathBuf;

use yorkie_numa::{CpuIndex, L3Domain, NumaLayout, SysfsOptions};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn all_cpus(n: CpuIndex) -> BTreeSet<CpuIndex> {
    (0..n).collect()
}

fn opts(name: &str, online: BTreeSet<CpuIndex>) -> SysfsOptions {
    SysfsOptions {
        root: fixture(name),
        online_cpus: online,
    }
}

// -- system-NUMA parse from a fixture sysfs tree --------------------------

#[cfg_attr(miri, ignore)]
#[test]
fn a_two_node_machine_reads_back_as_two_nodes() {
    let layout = NumaLayout::of_machine(&opts("two_node_l3", all_cpus(8)));
    assert_eq!(layout.nodes, vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]]);
    assert_eq!(layout.system_nodes, vec![0, 1]);
    assert_eq!(layout.system_node_of_cpu(5), Some(1));
}

#[cfg_attr(miri, ignore)]
#[test]
fn an_offline_cpu_is_left_out_of_its_node() {
    // The node `cpulist` names CPUs 4-7; only 4 and 5 are online, so the node
    // holds those two and nothing else.
    let online: BTreeSet<CpuIndex> = (0..6).collect();
    let layout = NumaLayout::of_machine(&opts("two_node_l3", online));
    assert_eq!(layout.nodes, vec![vec![0, 1, 2, 3], vec![4, 5]]);
}

// -- missing-file fallbacks ----------------------------------------------

#[cfg_attr(miri, ignore)]
#[test]
fn a_node_without_a_cpulist_falls_back_to_one_node() {
    // node1/cpulist is absent, so the partial topology is discarded in favour of
    // the single node holding every online CPU.
    let layout = NumaLayout::of_machine(&opts("missing_cpulist", all_cpus(8)));
    assert_eq!(layout.nodes, vec![(0..8).collect::<Vec<_>>()]);
    assert_eq!(layout.system_nodes, vec![0]);
}

#[cfg_attr(miri, ignore)]
#[test]
fn a_tree_without_a_node_list_falls_back_to_one_node() {
    let layout = NumaLayout::of_machine(&opts("no_online", all_cpus(4)));
    assert_eq!(layout.nodes, vec![(0..4).collect::<Vec<_>>()]);
}

// -- L3 domains -----------------------------------------------------------

#[cfg_attr(miri, ignore)]
#[test]
fn l3_domains_are_ordered_by_system_node_then_lowest_cpu() {
    let o = opts("two_node_l3", all_cpus(8));
    let layout = NumaLayout::of_machine(&o);
    assert_eq!(
        yorkie_numa::l3_domains(&o, &layout),
        vec![
            L3Domain {
                system_node: 0,
                cpus: vec![0, 1]
            },
            L3Domain {
                system_node: 0,
                cpus: vec![2, 3]
            },
            L3Domain {
                system_node: 1,
                cpus: vec![4, 5]
            },
            L3Domain {
                system_node: 1,
                cpus: vec![6, 7]
            },
        ]
    );
}

#[cfg_attr(miri, ignore)]
#[test]
fn a_cpu_without_a_reported_l3_is_a_domain_of_its_own() {
    // The fixture reports no cache tree at all, so each online CPU shares an L3
    // with nothing and stands alone.
    let o = opts("missing_cpulist", all_cpus(3));
    let layout = NumaLayout::of_machine(&o);
    let domains = yorkie_numa::l3_domains(&o, &layout);
    assert_eq!(domains.len(), 3);
    for (n, domain) in domains.iter().enumerate() {
        assert_eq!(domain.cpus, vec![n]);
    }
}

// -- the machine behind a sysfs root --------------------------------------

#[cfg_attr(miri, ignore)]
#[test]
fn machine_options_describe_every_online_cpu() {
    let o = yorkie_numa::machine_sysfs_options(&fixture("two_node_l3")).expect("a full tree");
    assert_eq!(o.online_cpus, all_cpus(8));
    assert_eq!(NumaLayout::of_machine(&o).num_nodes(), 2);
}

#[cfg_attr(miri, ignore)]
#[test]
fn machine_options_refuse_a_root_without_sysfs() {
    // A tree that cannot say which CPUs or which nodes exist gets no answer at
    // all, rather than the guessed single node the detection path falls back to.
    let err =
        yorkie_numa::machine_sysfs_options(&fixture("missing_cpulist")).expect_err("no cpu list");
    let message = refusal_message(&err);
    assert!(
        message.contains("devices/system/cpu/online"),
        "message: {message}"
    );
    let err = yorkie_numa::machine_sysfs_options(&fixture("no_online")).expect_err("no node list");
    let message = refusal_message(&err);
    assert!(
        message.contains("devices/system/node/online"),
        "message: {message}"
    );
}

/// A refusal's message as a string, for the assertions above.
fn refusal_message(err: &yorkie_numa::SysfsError) -> String {
    let mut out = Vec::new();
    err.write_message(|fragment| out.extend_from_slice(fragment));
    String::from_utf8(out).expect("a message is ASCII")
}

// -- best-effort Linux smoke ----------------------------------------------

#[cfg(target_os = "linux")]
#[cfg_attr(miri, ignore)]
#[test]
fn smoke_the_real_machine_parses() {
    // Structure-only assertions: the real /sys parses without error and yields
    // at least one non-empty node covering every online CPU, and every one of
    // those CPUs lands in exactly one L3 domain. We do NOT assert any values,
    // since those are machine-specific.
    let o = yorkie_numa::machine_sysfs_options(std::path::Path::new("/sys"))
        .expect("a Linux host reports a topology");
    let layout = NumaLayout::of_machine(&o);
    assert!(layout.num_nodes() >= 1);
    assert_eq!(layout.cpus(), o.online_cpus);

    let domains = yorkie_numa::l3_domains(&o, &layout);
    let covered: BTreeSet<CpuIndex> = domains
        .iter()
        .flat_map(|d| d.cpus.iter().copied())
        .collect();
    assert_eq!(covered, o.online_cpus);
    assert_eq!(
        covered.len(),
        domains.iter().map(|d| d.cpus.len()).sum::<usize>(),
        "no CPU belongs to two L3 domains"
    );
}
