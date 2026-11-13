//! Driver-level session tests for the `bench` command.
//!
//! The syntax tests run *without* a network: each position resigns instantly, so
//! they exercise the argument parse and summary plumbing without a real search.
//! The determinism and multi-thread tests stage a synthetic all-zero network and
//! run a small fixed-depth bench.
//!
//! Gated on `verbose3`, so without that feature this file compiles to nothing.
//! The complementary "command absent" assertions live in `src/parser.rs` and
//! `src/usi.rs`, compiled only without it.
//!
//! The `bench:` summary line is the command's result, not a diagnostic, so it is
//! asserted unconditionally; the argument-rejection lines are diagnostics and
//! are asserted only under `verbose1`.

#![cfg(feature = "verbose3")]

mod common;

use common::{drive, stage_configured_eval_dir};

/// Extract the `nodes=` field from the single `bench:` summary line in `out`.
fn bench_summary_nodes(out: &str) -> u64 {
    let line = out
        .lines()
        .find(|l| l.contains("bench: positions="))
        .unwrap_or_else(|| panic!("no bench summary line in:\n{out}"));
    let field = line
        .split_whitespace()
        .find_map(|t| t.strip_prefix("nodes="))
        .unwrap_or_else(|| panic!("no nodes= field in bench summary: {line:?}"));
    field.parse().expect("nodes= is an integer")
}

/// The `positions=` field from the `bench:` summary line.
fn bench_summary_positions(out: &str) -> u64 {
    let line = out
        .lines()
        .find(|l| l.contains("bench: positions="))
        .unwrap_or_else(|| panic!("no bench summary line in:\n{out}"));
    line.split_whitespace()
        .find_map(|t| t.strip_prefix("positions="))
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| panic!("no positions= field in: {line:?}"))
}

// Syntax variants (hermetic — no network, so each position resigns).

#[cfg_attr(miri, ignore)]
#[test]
fn bare_bench_runs_the_four_default_positions() {
    let _session = common::serial_session();
    // No network loaded → each of the four default positions resigns instantly,
    // and the summary reports positions=4, nodes=0. This proves the default
    // position list and the summary plumbing without a 60-second real search.
    let out = drive("bench\nquit\n");
    assert_eq!(
        bench_summary_positions(&out),
        4,
        "bare bench runs the four default positions:\n{out}"
    );
    assert_eq!(
        bench_summary_nodes(&out),
        0,
        "no network → zero nodes:\n{out}"
    );
    let bestmoves = common::bestmove_lines(&out);
    assert_eq!(
        bestmoves,
        vec!["resign"; 4],
        "one resign per position:\n{out}"
    );
}

/// An argument rejection is a diagnostic, so the line that names it is
/// `verbose1`. What "loudly" protects — that a bad argument runs no positions
/// rather than benching something else — is asserted in every build.
#[cfg_attr(miri, ignore)]
#[test]
fn garbage_argument_fails_loudly_without_panicking() {
    let _session = common::serial_session();
    // A non-integer TT size is a loud parse error, not a panic and not a search.
    let out = drive("bench notanumber\nquit\n");
    if cfg!(feature = "verbose1") {
        assert!(
            out.contains("info string bench: invalid ttSizeMB"),
            "expected a loud parse error:\n{out}"
        );
    }
    assert!(
        !out.contains("bench: positions="),
        "a parse error runs no positions:\n{out}"
    );
}

#[cfg(feature = "verbose1")]
#[cfg_attr(miri, ignore)]
#[test]
fn unsupported_limit_type_fails_loudly() {
    let _session = common::serial_session();
    // `perft` / `eval` are out of NPS-bench scope; they are reported, not run.
    let out = drive("bench 16 1 5 default perft\nquit\n");
    assert!(
        out.contains("info string bench: unsupported limit type `perft`"),
        "expected the scope-divergence notice:\n{out}"
    );
}

#[cfg_attr(miri, ignore)]
#[test]
fn current_source_benches_the_set_position() {
    let _session = common::serial_session();
    // `current` benches exactly one position (the session position). With no
    // network it resigns, but the summary must report positions=1.
    let session = "position startpos moves 7g7f\n\
                   bench 16 1 4 current depth\n\
                   quit\n";
    let out = drive(session);
    assert_eq!(
        bench_summary_positions(&out),
        1,
        "current source = one position:\n{out}"
    );
}

// Determinism — identical total nodes across runs (threads=1).

/// The standard bench session against a staged synthetic network: a small
/// fixed-depth default bench so the test stays fast. Returns the full transcript.
fn bench_session(bench_line: &str) -> String {
    stage_configured_eval_dir();
    let input = format!(
        "usi\n\
         isready\n\
         {bench_line}\n\
         quit\n"
    );
    drive(&input)
}

#[cfg_attr(miri, ignore)]
#[test]
fn two_runs_in_one_process_report_identical_nodes() {
    let _session = common::serial_session();
    stage_configured_eval_dir();

    // Two bench runs in ONE process (one network load). Each resets the TT and
    // histories at its start, so run 2 sees the same clean state as run 1.
    let input = "usi\n\
                 isready\n\
                 bench 16 1 3 default depth\n\
                 bench 16 1 3 default depth\n\
                 quit\n";
    let out = drive(input);
    let summaries: Vec<&str> = out
        .lines()
        .filter(|l| l.contains("bench: positions="))
        .collect();
    assert_eq!(summaries.len(), 2, "two bench summaries:\n{out}");

    let nodes0 = bench_summary_nodes(summaries[0]);
    let nodes1 = bench_summary_nodes(summaries[1]);
    assert!(
        nodes0 > 0,
        "the synthetic-network bench searched real nodes:\n{out}"
    );
    assert_eq!(
        nodes0, nodes1,
        "two in-process bench runs must report identical total nodes:\n{out}"
    );
}

#[cfg_attr(miri, ignore)]
#[test]
fn two_process_launches_report_identical_nodes() {
    let _session = common::serial_session();
    // Two independent driver runs (separate "process launches") with identical
    // input must report the same total nodes — determinism does not depend on
    // in-process carry-over.
    let a = bench_session("bench 16 1 3 default depth");
    let b = bench_session("bench 16 1 3 default depth");
    let na = bench_summary_nodes(&a);
    let nb = bench_summary_nodes(&b);
    assert!(na > 0, "real search:\n{a}");
    assert_eq!(
        na, nb,
        "two process launches must agree on total nodes\nA:\n{a}\nB:\n{b}"
    );
}

/// After the summary, one statistics line sums the search counts over every
/// position of the run, so its `nodes` is the summary's.
#[cfg_attr(miri, ignore)]
#[test]
fn the_summary_is_followed_by_the_runs_statistics() {
    let _session = common::serial_session();
    let out = bench_session("bench 16 1 6 default depth");
    let lines: Vec<&str> = out.lines().collect();
    let at = lines
        .iter()
        .position(|l| l.contains("bench: positions="))
        .unwrap_or_else(|| panic!("no bench summary line in:\n{out}"));
    let stats = lines
        .get(at + 1)
        .and_then(|l| l.strip_prefix("info string stats "))
        .unwrap_or_else(|| panic!("a statistics line must follow the summary in:\n{out}"));
    let nodes: u64 = stats
        .split(' ')
        .find_map(|item| item.strip_prefix("nodes="))
        .unwrap_or_else(|| panic!("no nodes= item in {stats:?}"))
        .parse()
        .expect("nodes= is an integer");
    let summary = bench_summary_nodes(&out);
    assert!(summary > 0, "real search:\n{out}");
    assert_eq!(nodes, summary, "in:\n{out}");
    assert!(
        !stats.split(' ').any(|item| item.starts_with("alloc=")),
        "the run's line carries the search counts only: {stats:?}"
    );
}

/// A `dupstats` build follows the run's statistics line with the histograms of
/// its duplication counts: the bucket labels, one line per store and presence
/// outcome, and the workers' spread, one search per position.
#[cfg(feature = "dupstats")]
#[cfg_attr(miri, ignore)]
#[test]
fn the_runs_statistics_are_followed_by_the_duplication_histograms() {
    let _session = common::serial_session();
    let out = bench_session("bench 16 2 3 default depth");
    let lines: Vec<&str> = out.lines().collect();
    let at = lines
        .iter()
        .position(|l| l.contains("bench: positions="))
        .unwrap_or_else(|| panic!("no bench summary line in:\n{out}"));
    let stats = lines[at + 1]
        .strip_prefix("info string stats ")
        .unwrap_or_else(|| panic!("a statistics line must follow the summary in:\n{out}"));
    let count = |key: &str| -> u64 {
        stats
            .split(' ')
            .find_map(|item| item.strip_prefix(key)?.strip_prefix('='))
            .map_or(0, |n| n.parse().expect("an integer"))
    };
    let positions = bench_summary_positions(&out);
    assert_eq!(count("sp_searches"), positions, "{stats:?}");
    assert!(count("st_new") > 0, "{stats:?}");
    assert!(count("pr_enter") > 0, "{stats:?}");

    let hist: Vec<&str> = lines[at + 2..]
        .iter()
        .map_while(|l| l.strip_prefix("info string duphist "))
        .collect();
    assert_eq!(hist.len(), 1 + 12 + 5 + 1, "in:\n{out}");
    assert!(hist[0].starts_with("buckets depth="), "{:?}", hist[0]);
    assert!(hist[1].starts_with("st_new depth="), "{:?}", hist[1]);
    assert!(hist[18].starts_with("spread behind="), "{:?}", hist[18]);
    let behind: u64 = hist[18]["spread behind=".len()..]
        .split(',')
        .map(|n| n.parse::<u64>().expect("an integer"))
        .sum();
    assert_eq!(behind, count("sp_workers"), "every worker is in the spread");
}

// A threads=2 bench completes and reports a summary (no determinism).

#[cfg_attr(miri, ignore)]
#[test]
fn threads_two_bench_completes_and_reports() {
    let _session = common::serial_session();
    let out = bench_session("bench 16 2 3 default depth");
    assert_eq!(
        bench_summary_positions(&out),
        4,
        "a threads=2 bench still runs all four default positions:\n{out}"
    );
    // No node-count assertion: multi-thread node totals vary run to run.
}

/// One `info string iter` line of a `dupstats` bench, its fields by name.
#[cfg(feature = "dupstats")]
#[derive(Debug)]
struct IterLine {
    pos: u64,
    worker: u64,
    depth: u64,
    done: bool,
    us: u64,
    nodes: u64,
}

/// Every `iter` line of `out`, in the order written.
#[cfg(feature = "dupstats")]
fn iter_lines(out: &str) -> Vec<IterLine> {
    out.lines()
        .filter_map(|l| l.strip_prefix("info string iter "))
        .map(|body| {
            let field = |key: &str| -> &str {
                body.split(' ')
                    .find_map(|item| item.strip_prefix(key)?.strip_prefix('='))
                    .unwrap_or_else(|| panic!("no {key}= in {body:?}"))
            };
            let num = |key: &str| -> u64 { field(key).parse().expect("an integer") };
            for key in ["fl", "fh", "first", "rest", "value"] {
                field(key).parse::<i64>().expect("an integer");
            }
            assert!(
                ["none", "self", "other", "prev", "unknown"].contains(&field("root")),
                "{body:?}"
            );
            assert!(!field("move").is_empty(), "{body:?}");
            assert!(["0", "1"].contains(&field("changed")), "{body:?}");
            IterLine {
                pos: num("pos"),
                worker: num("worker"),
                depth: num("depth"),
                done: field("done") == "1",
                us: num("us"),
                nodes: num("nodes"),
            }
        })
        .collect()
}

/// The lines of one worker in one position's search hold every depth it
/// completed, from 1 and in order, then at most the one it was stopped
/// inside, marked not done — at times that never go back and node counts that
/// only grow. Returns the deepest completed depth, and whether the worker was
/// stopped inside the next one.
#[cfg(feature = "dupstats")]
fn check_worker_lines(lines: &[&IterLine]) -> (u64, bool) {
    let completed = lines.iter().filter(|l| l.done).count() as u64;
    for (i, l) in lines.iter().enumerate() {
        assert_eq!(l.depth, i as u64 + 1, "{lines:?}");
        assert_eq!(l.done, l.depth <= completed, "{lines:?}");
    }
    assert!(lines.len() as u64 <= completed + 1, "{lines:?}");
    for pair in lines.windows(2) {
        assert!(pair[0].us <= pair[1].us, "{pair:?}");
        assert!(pair[0].nodes < pair[1].nodes, "{pair:?}");
    }
    (completed, lines.len() as u64 > completed)
}

/// The deepest completed depth of each worker of position `pos`'s search, and
/// whether it was stopped inside the next one.
#[cfg(feature = "dupstats")]
fn completed_by_worker(lines: &[IterLine], pos: u64, workers: u64) -> Vec<(u64, bool)> {
    (0..workers)
        .map(|w| {
            let mine: Vec<&IterLine> = lines
                .iter()
                .filter(|l| l.pos == pos && l.worker == w)
                .collect();
            check_worker_lines(&mine)
        })
        .collect()
}

/// At one thread a `dupstats` bench writes one `iter` line per completed depth
/// of every position, all of them the main worker's, after the histograms: up
/// to the bench depth, or — under a config whose node limit ends the search
/// first — up to the depth that limit stopped it inside.
#[cfg(feature = "dupstats")]
#[cfg_attr(miri, ignore)]
#[test]
fn one_thread_writes_one_iteration_line_per_completed_depth() {
    let _session = common::serial_session();
    let out = bench_session("bench 16 1 4 default depth");
    let lines = iter_lines(&out);
    assert!(lines.iter().all(|l| l.worker == 0), "{lines:?}");
    for pos in 0..4 {
        let [(completed, stopped)] = completed_by_worker(&lines, pos, 1)[..] else {
            unreachable!("one worker");
        };
        assert!(stopped || completed == 4, "position {pos}:\n{out}");
    }
    let all: Vec<&str> = out.lines().collect();
    let last_hist = all
        .iter()
        .rposition(|l| l.starts_with("info string duphist "))
        .expect("histograms");
    let first_iter = all
        .iter()
        .position(|l| l.starts_with("info string iter "))
        .expect("iteration lines");
    assert!(
        last_hist < first_iter,
        "the iterations follow the histograms"
    );
}

/// At two threads every completed depth of every worker has a line, a worker
/// stopped inside an iteration has that one marked not done, and the main
/// worker reaches the bench depth in every position's search it was not
/// stopped in by a node limit.
#[cfg(feature = "dupstats")]
#[cfg_attr(miri, ignore)]
#[test]
fn two_threads_write_every_workers_iterations() {
    let _session = common::serial_session();
    let out = bench_session("bench 16 2 5 default depth");
    let lines = iter_lines(&out);
    for pos in 0..4 {
        let (main_completed, main_stopped) = completed_by_worker(&lines, pos, 2)[0];
        assert!(
            main_stopped || main_completed == 5,
            "position {pos}:\n{out}"
        );
    }
}

/// Stopping at the first worker: in every position's search some worker
/// reached the bench depth, and the others' lines end where they were stopped
/// — unless a node limit stopped every worker first.
#[cfg(feature = "dupstats")]
#[cfg_attr(miri, ignore)]
#[test]
fn stopping_at_the_first_worker_ends_on_a_worker_at_the_depth() {
    let _session = common::serial_session();
    let out = bench_session("bench 16 2 5 default depth first");
    assert_eq!(bench_summary_positions(&out), 4, "in:\n{out}");
    let lines = iter_lines(&out);
    for pos in 0..4 {
        let workers = completed_by_worker(&lines, pos, 2);
        let finished = workers.iter().filter(|&&(d, stopped)| d == 5 && !stopped);
        assert!(
            finished.count() >= 1 || workers.iter().all(|&(_, stopped)| stopped),
            "position {pos}: {workers:?}\n{out}"
        );
    }
}
