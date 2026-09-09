//! Which logical CPU each worker is pinned to, and the machine-local ledger the
//! choice is taken from.
//!
//! `build_cpus.rs` is `include!`d here exactly as `build.rs` includes it, so
//! these tests exercise the same code that decides where a binary's workers run.
//! The promise it rests on: two binaries built on one machine never take the
//! same CPU, and rebuilding one of them takes nothing more.

#![allow(dead_code)]

include!(concat!(env!("CARGO_MANIFEST_DIR"), "/build_config.rs"));
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/build_numa.rs"));
include!(concat!(env!("CARGO_MANIFEST_DIR"), "/build_cpus.rs"));

use std::sync::atomic::{AtomicU64, Ordering};

/// One L3 domain, in the form the machine reader hands them over.
fn domain(system_node: usize, cpus: &[usize]) -> L3Domain {
    L3Domain {
        system_node,
        cpus: cpus.to_vec(),
    }
}

/// A machine with two NUMA nodes of four CPUs each, every node split into two
/// L3 domains — the smallest shape in which "round-robin over the domains" and
/// "in node order" are two different answers.
fn two_node_machine() -> Machine {
    Machine {
        online: (0..8).collect(),
        domains: vec![
            domain(0, &[0, 1]),
            domain(0, &[2, 3]),
            domain(1, &[4, 5]),
            domain(1, &[6, 7]),
        ],
        layout: yorkie_numa::NumaLayout::from_const(&[&[0, 1, 2, 3], &[4, 5, 6, 7]], &[0, 1]),
    }
}

fn taken(cpus: &[usize]) -> BTreeSet<usize> {
    cpus.iter().copied().collect()
}

fn request<'a>(spec: &'a str, ledger: &'a str, threads: i64) -> AssignmentRequest<'a> {
    AssignmentRequest {
        spec,
        ledger,
        threads,
    }
}

fn identity(out_dir: &str) -> BuildIdentity {
    BuildIdentity {
        out_dir: out_dir.to_string(),
        config: "configs/default.toml".to_string(),
        version: "1.2.3".to_string(),
    }
}

/// A fresh ledger path under `$TMPDIR` that no test shares.
fn ledger_path(tag: &str) -> PathBuf {
    static CTR: AtomicU64 = AtomicU64::new(0);
    let n = CTR.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "yorkie-cpu-ledger-{}-{tag}-{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    path
}

// --- The pick: deterministic, and spread over the L3 domains ---------------

#[cfg_attr(miri, ignore)]
#[test]
fn the_pick_goes_round_the_l3_domains_in_order() {
    let m = two_node_machine();
    // One worker per domain, first CPU of each, domains in node-then-lowest-CPU
    // order.
    assert_eq!(
        round_robin_cpus(&m.domains, &taken(&[]), 4),
        Ok(vec![0, 2, 4, 6])
    );
    // Past one round it wraps, so the whole machine comes out in the order the
    // domains were visited rather than in CPU order.
    assert_eq!(
        round_robin_cpus(&m.domains, &taken(&[]), 8),
        Ok(vec![0, 2, 4, 6, 1, 3, 5, 7])
    );
    // A single worker lands on the first free CPU of the first domain.
    assert_eq!(round_robin_cpus(&m.domains, &taken(&[]), 1), Ok(vec![0]));
}

#[cfg_attr(miri, ignore)]
#[test]
fn the_pick_is_the_same_answer_every_time() {
    let m = two_node_machine();
    let once = round_robin_cpus(&m.domains, &taken(&[]), 3);
    for _ in 0..8 {
        assert_eq!(round_robin_cpus(&m.domains, &taken(&[]), 3), once);
    }
}

#[cfg_attr(miri, ignore)]
#[test]
fn the_pick_skips_cpus_the_ledger_holds_and_still_spreads() {
    let m = two_node_machine();
    // The first domain has nothing left, so the round-robin passes over it and
    // comes back to the others.
    assert_eq!(
        round_robin_cpus(&m.domains, &taken(&[0, 1]), 4),
        Ok(vec![2, 4, 6, 3])
    );
    // Half of every domain gone: what is left is still visited domain by
    // domain.
    assert_eq!(
        round_robin_cpus(&m.domains, &taken(&[0, 2, 4, 6]), 4),
        Ok(vec![1, 3, 5, 7])
    );
}

#[cfg_attr(miri, ignore)]
#[test]
fn a_shortfall_reports_how_many_cpus_were_free() {
    let m = two_node_machine();
    assert_eq!(
        round_robin_cpus(&m.domains, &taken(&[0, 1, 2, 3]), 5),
        Err(4)
    );
    assert_eq!(
        round_robin_cpus(&m.domains, &taken(&(0..8).collect::<Vec<_>>()), 1),
        Err(0)
    );
}

// --- The ledger file -------------------------------------------------------

#[cfg_attr(miri, ignore)]
#[test]
fn a_ledger_round_trips_through_its_own_format() {
    let entries = vec![
        LedgerEntry {
            cpus: vec![0, 1, 2, 3],
            identity: identity("/build/a"),
        },
        LedgerEntry {
            cpus: vec![7],
            identity: identity("/build/b"),
        },
    ];
    let text = render_ledger(&entries);
    assert!(
        text.contains("0-3\t/build/a\t"),
        "compressed CPU list: {text}"
    );
    assert_eq!(parse_ledger(&text, "ledger"), Ok(entries));
}

#[cfg_attr(miri, ignore)]
#[test]
fn a_ledger_line_that_cannot_be_read_is_an_error() {
    // A line holding CPUs this build must not take, that cannot be read, is the
    // one case where guessing would hand a CPU to two binaries.
    for (body, want) in [
        (
            "0-3\t/build/a\tconfigs/default.toml\n",
            "four tab-separated",
        ),
        (
            "zero\t/build/a\tconfigs/default.toml\t1.2.3\n",
            "is not a CPU index",
        ),
        ("\t/build/a\tconfigs/default.toml\t1.2.3\n", "empty entry"),
    ] {
        let err = parse_ledger(body, "ledger").expect_err("must fail on: {body}");
        assert!(err.contains(want), "for {body:?} expected {want:?}: {err}");
        assert!(err.starts_with("ledger:1: "), "message: {err}");
    }
}

#[cfg_attr(miri, ignore)]
#[test]
fn comments_and_blank_lines_are_not_allocations() {
    let text = "# a header\n\n0-1\t/build/a\tconfigs/default.toml\t1.2.3\n\n";
    let entries = parse_ledger(text, "ledger").expect("parses");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].cpus, vec![0, 1]);
}

// --- Taking CPUs, under the lock -------------------------------------------

#[cfg_attr(miri, ignore)]
#[test]
fn the_first_build_takes_its_cpus_and_the_ledger_gains_a_line() {
    let m = two_node_machine();
    let path = ledger_path("first");
    let cpus = take_cpus(&request("auto", "", 2), &m, &path, &identity("/build/a"))
        .expect("a fresh ledger has every CPU free");
    assert_eq!(cpus, vec![0, 2]);

    let text = std::fs::read_to_string(&path).expect("the ledger was written");
    let entries = parse_ledger(&text, "ledger").expect("parses");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].cpus, vec![0, 2]);
    assert_eq!(entries[0].identity, identity("/build/a"));
    let _ = std::fs::remove_file(&path);
}

#[cfg_attr(miri, ignore)]
#[test]
fn a_fresh_ledger_gives_the_same_build_the_same_cpus() {
    let m = two_node_machine();
    let first = ledger_path("deterministic-1");
    let second = ledger_path("deterministic-2");
    let a = take_cpus(&request("auto", "", 2), &m, &first, &identity("/build/a")).expect("takes");
    let b = take_cpus(&request("auto", "", 2), &m, &second, &identity("/build/a")).expect("takes");
    assert_eq!(a, b, "the pick has no randomness in it");
    let _ = std::fs::remove_file(&first);
    let _ = std::fs::remove_file(&second);
}

#[cfg_attr(miri, ignore)]
#[test]
fn rebuilding_keeps_a_builds_line_and_takes_nothing_more() {
    let m = two_node_machine();
    let path = ledger_path("rebuild");
    let me = identity("/build/a");
    let first = take_cpus(&request("auto", "", 3), &m, &path, &me).expect("takes");
    for _ in 0..3 {
        assert_eq!(
            take_cpus(&request("auto", "", 3), &m, &path, &me).expect("takes"),
            first,
            "a rebuild keeps the CPUs it already holds"
        );
        let entries = parse_ledger(&std::fs::read_to_string(&path).unwrap(), "l").unwrap();
        assert_eq!(entries.len(), 1, "and adds no line");
    }
    let _ = std::fs::remove_file(&path);
}

#[cfg_attr(miri, ignore)]
#[test]
fn another_build_on_the_same_machine_gets_other_cpus() {
    let m = two_node_machine();
    let path = ledger_path("share");
    let a = take_cpus(&request("auto", "", 2), &m, &path, &identity("/build/a")).expect("takes");
    let b = take_cpus(&request("auto", "", 2), &m, &path, &identity("/build/b")).expect("takes");
    assert_eq!(a, vec![0, 2]);
    // The second build starts its own round at the first domain again, and the
    // CPUs the first build holds are simply not free any more.
    assert_eq!(b, vec![1, 3]);
    assert!(
        a.iter().all(|cpu| !b.contains(cpu)),
        "no CPU is handed out twice"
    );
    let _ = std::fs::remove_file(&path);
}

#[cfg_attr(miri, ignore)]
#[test]
fn a_build_that_does_not_fit_in_what_is_left_is_refused() {
    let m = two_node_machine();
    let path = ledger_path("shortfall");
    take_cpus(&request("auto", "", 6), &m, &path, &identity("/build/a")).expect("takes");
    let err = take_cpus(&request("auto", "", 4), &m, &path, &identity("/build/b"))
        .expect_err("two CPUs are left and four are wanted");
    assert!(err.contains("needs 4 free CPU(s)"), "message: {err}");
    assert!(err.contains("leaves 2"), "message: {err}");
    assert!(err.contains("2 short"), "message: {err}");
    assert!(
        err.contains(&path.display().to_string()),
        "the message must name the ledger: {err}"
    );
    let _ = std::fs::remove_file(&path);
}

#[cfg_attr(miri, ignore)]
#[test]
fn more_workers_than_the_machine_has_cpus_is_refused() {
    let m = two_node_machine();
    let path = ledger_path("oversubscribed");
    let err = take_cpus(&request("auto", "", 9), &m, &path, &identity("/build/a"))
        .expect_err("eight CPUs cannot hold nine workers");
    assert!(err.contains("`threads` = 9"), "message: {err}");
    assert!(
        err.contains("8 logical CPU(s) the machine reports online"),
        "message: {err}"
    );
    let _ = std::fs::remove_file(&path);
}

// --- An explicit CPU list --------------------------------------------------

#[cfg_attr(miri, ignore)]
#[test]
fn an_explicit_list_is_taken_as_written() {
    let m = two_node_machine();
    let path = ledger_path("explicit");
    let cpus = take_cpus(&request("5,1,4", "", 3), &m, &path, &identity("/build/a"))
        .expect("three online CPUs");
    assert_eq!(cpus, vec![5, 1, 4], "worker order is the order written");
    let _ = std::fs::remove_file(&path);
}

#[cfg_attr(miri, ignore)]
#[test]
fn an_explicit_list_neither_reads_nor_writes_the_ledger() {
    let m = two_node_machine();
    let path = ledger_path("explicit-ignores-ledger");

    // A ledger recording CPU 3 as another build's does not stand in the way of a
    // list naming CPU 3: keeping the two builds apart is whoever wrote the list's
    // business, not the ledger's.
    take_cpus(&request("auto", "", 8), &m, &path, &identity("/build/a")).expect("takes them all");
    let before = std::fs::read_to_string(&path).expect("the ledger was written");
    assert_eq!(
        take_cpus(&request("3", "", 1), &m, &path, &identity("/build/b")).expect("takes CPU 3"),
        vec![3]
    );
    assert_eq!(
        std::fs::read_to_string(&path).expect("the ledger is still there"),
        before,
        "an explicit list adds no line"
    );

    // And with no ledger at all it is still a legal assignment: the file the path
    // names is never created.
    let absent = ledger_path("explicit-no-ledger");
    assert_eq!(
        take_cpus(&request("3", "", 1), &m, &absent, &identity("/build/c")).expect("takes CPU 3"),
        vec![3]
    );
    assert!(!absent.exists(), "no ledger was created");
    let _ = std::fs::remove_file(&path);
}

#[cfg_attr(miri, ignore)]
#[test]
fn auto_without_a_ledger_is_a_build_error() {
    let Err(err) = resolve_cpu_assignment(
        &request("auto", "", 1),
        &identity("/build/a"),
        Path::new("/repo"),
    ) else {
        panic!("`auto` has nothing to pick from");
    };
    assert!(err.contains("needs a ledger"), "message: {err}");
    assert!(err.contains("`cpu_ledger` is empty"), "message: {err}");
}

#[cfg_attr(miri, ignore)]
#[test]
fn an_explicit_list_the_machine_refuses_is_a_build_error() {
    let m = two_node_machine();

    let path = ledger_path("explicit-count");
    let err = take_cpus(&request("0-3", "", 2), &m, &path, &identity("/build/a"))
        .expect_err("four CPUs for two workers");
    assert!(err.contains("names 4 CPU(s) but `threads` is 2"), "{err}");

    let path = ledger_path("explicit-dup");
    let err = take_cpus(&request("1,1", "", 2), &m, &path, &identity("/build/a"))
        .expect_err("one CPU cannot hold two workers");
    assert!(err.contains("names CPU 1 twice"), "{err}");

    let path = ledger_path("explicit-offline");
    let err = take_cpus(&request("9", "", 1), &m, &path, &identity("/build/a"))
        .expect_err("CPU 9 is not on the fixture machine");
    assert!(err.contains("does not report online"), "{err}");
    assert!(
        err.contains("0-7"),
        "the message names what is online: {err}"
    );

    let path = ledger_path("explicit-malformed");
    let err = take_cpus(&request("0,two", "", 2), &m, &path, &identity("/build/a"))
        .expect_err("`two` is not a CPU index");
    assert!(err.contains("is not a CPU index"), "{err}");
}

// --- The lock --------------------------------------------------------------

/// Two builds racing for one ledger must serialise: the lock is held across the
/// whole read-decide-write cycle, so the second sees the first's line and picks
/// around it. Without the lock both would read an empty ledger and take the same
/// CPU.
#[cfg_attr(miri, ignore)]
#[test]
fn concurrent_builds_do_not_take_the_same_cpu() {
    let m = two_node_machine();
    let path = ledger_path("race");
    let takers: Vec<std::thread::JoinHandle<Vec<usize>>> = (0..8)
        .map(|n| {
            let path = path.clone();
            let machine = two_node_machine();
            std::thread::spawn(move || {
                take_cpus(
                    &request("auto", "", 1),
                    &machine,
                    &path,
                    &identity(&format!("/build/{n}")),
                )
                .expect("eight builds fit on eight CPUs")
            })
        })
        .collect();
    let mut got: Vec<usize> = takers
        .into_iter()
        .flat_map(|h| h.join().expect("taker thread must not panic"))
        .collect();
    got.sort_unstable();
    assert_eq!(
        got,
        m.online.iter().copied().collect::<Vec<_>>(),
        "eight single-worker builds fill the machine exactly once"
    );
    let _ = std::fs::remove_file(&path);
}
