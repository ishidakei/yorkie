//! Starting a session as a process confined to part of the machine.
//!
//! A `taskset` or a cgroup cpuset around the engine hides CPUs from the process.
//! Every worker of every build pins itself to one logical CPU chosen when the
//! binary was built, so a confinement that hides one of those CPUs takes a
//! worker's seat away — `isready` names the difference and withholds `readyok`
//! rather than letting the pin fail mid-game inside a spawned worker. A
//! confinement that leaves them alone costs nothing, and the session readies as
//! usual: that is what a set of engines pinned to individual CPUs of one
//! many-core machine relies on.
//!
//! A build with `cpu_assignment = "none"` pins no worker, so no confinement takes
//! anything from it: it readies under any `taskset` at all, which is what lets
//! many processes of that one binary share a machine.
//!
//! The confinement is presented to the driver rather than imposed on the test
//! process, which shares its CPUs with the rest of the suite.

mod common;

use std::collections::BTreeSet;
use std::path::PathBuf;

use common::{Transcript, stage_configured_eval_dir};
use yorkie_protocol::{UsiEngine, config};

/// The CPUs this build's workers are pinned to.
fn worker_cpus() -> BTreeSet<usize> {
    config::WORKER_CPUS.iter().copied().collect()
}

/// Every CPU the compiled layout holds.
fn compiled_cpus() -> BTreeSet<usize> {
    config::NUMA_NODE_CPUS
        .iter()
        .flat_map(|cpus| cpus.iter().copied())
        .collect()
}

/// Drive a session as a process allowed on `cpus` alone, against the network
/// `eval_root` holds.
fn drive_confined(input: &str, cpus: BTreeSet<usize>, eval_root: PathBuf) -> String {
    let output = Transcript::default();
    let driver = UsiEngine::new(input.as_bytes(), output.clone())
        .with_startup_affinity(cpus)
        .with_eval_root(eval_root);
    driver.run().expect("driver run");
    output.text()
}

/// The directory the staged network sits under, which the driver reads
/// `eval_dir` against.
fn staged_eval_root() -> PathBuf {
    stage_configured_eval_dir()
        .parent()
        .and_then(std::path::Path::parent)
        .expect("the staged network sits under <root>/<eval_dir>")
        .to_path_buf()
}

#[cfg_attr(miri, ignore)]
#[test]
fn a_start_denied_a_workers_cpu_is_refused() {
    let _session = common::serial_session();
    if !config::WORKERS_PINNED {
        eprintln!("skipped: this build pins no worker, so no confinement denies one a CPU");
        return;
    }
    let workers = worker_cpus();
    let allowed: BTreeSet<usize> = compiled_cpus().difference(&workers).copied().collect();
    if allowed.is_empty() {
        // The workers cover every CPU the machine has, so there is no
        // confinement left to present that still names a real CPU.
        eprintln!("skipped: this build's workers cover every CPU of the machine");
        return;
    }

    let out = drive_confined("isready\nquit\n", allowed, staged_eval_root());
    assert!(
        out.contains("info string NUMA layout mismatch:"),
        "a process denied a worker's CPU must be refused, got: {out:?}"
    );
    assert!(
        out.contains("which its workers are pinned to"),
        "the refusal must say what the missing CPUs are for, got: {out:?}"
    );
    assert!(
        !out.contains("readyok"),
        "readyok must not follow a refused layout: {out:?}"
    );
}

#[cfg_attr(miri, ignore)]
#[test]
fn a_start_confined_to_exactly_the_workers_cpus_readies() {
    let _session = common::serial_session();
    if !config::WORKERS_PINNED {
        eprintln!("skipped: this build pins no worker, so it has no CPUs to be confined to");
        return;
    }
    // The narrowest confinement that still leaves every worker its seat: nothing
    // beyond the assignment is needed, since each worker narrows itself to one
    // CPU anyway.
    let out = drive_confined("isready\nquit\n", worker_cpus(), staged_eval_root());
    assert!(
        !out.contains("NUMA layout mismatch"),
        "an affinity covering every worker's CPU takes nothing away, got: {out:?}"
    );
    assert!(
        out.contains("readyok"),
        "the confined session must ready, got: {out:?}"
    );
    assert!(
        out.contains(&format!(
            "info string workers on CPUs {}",
            cpu_list(worker_cpus())
        )),
        "the session names the CPUs it plays on, got: {out:?}"
    );
}

/// Confined to a single CPU — each of the machine's lowest and highest in turn,
/// so every CPU but one is hidden and a worker's CPU is among the hidden ones
/// whenever a worker has one — a session readies exactly when no worker is
/// pinned to a hidden CPU: always, in a build that pins none.
#[cfg_attr(miri, ignore)]
#[test]
fn a_single_cpu_confinement_readies_unless_it_hides_a_workers_cpu() {
    let _session = common::serial_session();
    let compiled = compiled_cpus();
    if compiled.len() < 2 {
        eprintln!("skipped: a one-CPU machine has no confinement that hides a CPU");
        return;
    }
    let ends = [
        *compiled.first().expect("non-empty"),
        *compiled.last().expect("non-empty"),
    ];
    for cpu in ends {
        let allowed = BTreeSet::from([cpu]);
        let out = drive_confined("isready\nquit\n", allowed.clone(), staged_eval_root());
        let readies = !config::WORKERS_PINNED || worker_cpus().is_subset(&allowed);
        assert_eq!(
            out.contains("readyok"),
            readies,
            "confined to CPU {cpu}, pinned = {}: {out:?}",
            config::WORKERS_PINNED
        );
        assert_eq!(
            out.contains("info string NUMA layout mismatch:"),
            !readies,
            "a session that does not ready says why: {out:?}"
        );
        if !config::WORKERS_PINNED {
            assert!(
                out.contains("info string workers unpinned"),
                "an unpinned build says so: {out:?}"
            );
        }
    }
}

/// A CPU list in the shortened sysfs form, as the notice above writes it.
fn cpu_list(cpus: impl IntoIterator<Item = usize>) -> String {
    let mut out = Vec::new();
    yorkie_numa::write_cpu_list(cpus, |fragment| out.extend_from_slice(fragment));
    String::from_utf8(out).expect("a CPU list is ASCII")
}
