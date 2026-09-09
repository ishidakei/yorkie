//! Starting a session as a process confined to part of the machine.
//!
//! A `taskset` or a cgroup cpuset around the engine hides CPUs from the process,
//! and what that costs depends on the compiled thread plan. A plan that pins
//! workers to nodes loses the CPUs it was going to pin them to, so `isready`
//! names the difference and withholds `readyok`. A plan that pins nothing — a
//! single worker, or `numa_policy = "none"` — loses nothing, so the session
//! readies as usual and the transposition table keeps the process's own memory
//! policy. The second is what a set of single-thread engines pinned to
//! individual CPUs of one many-core machine relies on.
//!
//! The confinement is presented to the driver rather than imposed on the test
//! process, which shares its CPUs with the rest of the suite.

mod common;

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use common::stage_configured_eval_dir;
use yorkie_numa::NumaConfig;
use yorkie_protocol::{UsiDriver, config};

/// Whether this build's compiled thread plan pins a worker, which is what
/// decides whether hiding CPUs from the process takes anything away from it.
fn compiled_plan_binds() -> bool {
    let cfg = NumaConfig::from_const(config::NUMA_NODE_CPUS, config::NUMA_CUSTOM_AFFINITY);
    match config::NUMA_POLICY {
        "none" => false,
        "auto" => cfg.suggests_binding_threads(config::THREADS as usize),
        _ => true,
    }
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

#[cfg_attr(miri, ignore)]
#[test]
fn a_confined_start_is_refused_only_where_the_plan_pins_a_worker() {
    let all = compiled_cpus();
    let narrowed: BTreeSet<usize> = all.iter().copied().take(1).collect();
    if narrowed.len() == all.len() {
        // A one-CPU layout cannot be narrowed, so there is no confinement to
        // present. Nothing is skipped anywhere else.
        eprintln!("skipped: the compiled layout holds a single CPU");
        return;
    }

    let staged = stage_configured_eval_dir();
    let eval_root = staged
        .parent()
        .and_then(std::path::Path::parent)
        .expect("the staged network sits under <root>/<eval_dir>")
        .to_path_buf();
    let out = drive_confined("isready\nquit\n", narrowed, eval_root);

    if compiled_plan_binds() {
        assert!(
            out.contains("info string NUMA layout mismatch:"),
            "a plan that pins workers must refuse a process denied their CPUs, got: {out:?}"
        );
        assert!(
            !out.contains("readyok"),
            "readyok must not follow a refused layout: {out:?}"
        );
    } else {
        assert!(
            !out.contains("NUMA layout mismatch"),
            "a plan that pins nothing loses nothing to a confined start, got: {out:?}"
        );
        assert!(
            out.contains("readyok"),
            "the confined session must ready, got: {out:?}"
        );
        assert!(
            out.contains("info string transposition table: ")
                && out.contains("; process default policy;"),
            "with no worker pinned the table takes the process's own policy, got: {out:?}"
        );
    }
}
