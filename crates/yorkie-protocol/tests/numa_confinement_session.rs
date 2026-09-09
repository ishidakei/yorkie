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
//! The confinement is presented to the driver rather than imposed on the test
//! process, which shares its CPUs with the rest of the suite.

mod common;

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use common::stage_configured_eval_dir;
use yorkie_protocol::{UsiDriver, config};

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
    let output = Arc::new(Mutex::new(Vec::<u8>::new()));
    let driver = UsiDriver::new(input.as_bytes(), Arc::clone(&output))
        .with_startup_affinity(cpus)
        .with_eval_root(eval_root);
    driver.run().expect("driver run");
    String::from_utf8(output.lock().expect("output lock").clone()).expect("utf-8")
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
    let _tt = common::serial_tt();
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
    let _tt = common::serial_tt();
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
            yorkie_numa::format_cpu_list(worker_cpus())
        )),
        "the session names the CPUs it plays on, got: {out:?}"
    );
}
