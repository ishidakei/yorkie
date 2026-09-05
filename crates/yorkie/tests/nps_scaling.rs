//! NPS scaling measurement (Lazy SMP).
//!
//! Measures aggregate nodes-per-second at one and two workers for three
//! positions. The worker count is a compile-time constant, so the measurement
//! runs through `bench`, the one command that carries its own worker count as an
//! argument.
//!
//! This is a **measurement, not a threshold**: it prints a table and never
//! asserts a scaling factor. On a host whose two logical CPUs are SMT siblings a
//! ratio well under 2× is normal, so any threshold would encode the machine
//! rather than the engine.
//!
//! `#[ignore]`-gated, and it needs the real network staged locally; when the
//! network is absent it prints a notice and passes. Run it in a release build:
//!
//! ```text
//! cargo test --release -p yorkie --test nps_scaling -- --ignored --nocapture
//! ```
//!
//! Gated on `verbose2`: the session drives analysis-only `go` clauses, and this
//! is an NPS measurement payload that reads `nodes` back out of the engine's
//! transcript — the level at which the whole measurement surface (per-iteration
//! `info` lines included) is the one a bench round reports on. The spawned
//! `yorkie` binary reaches that level only when the test binary does.

#![cfg(feature = "verbose2")]

mod common;

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use common::{engine_cwd_with_eval_dir, eval_dir};

const MOVETIME_MS: u64 = 5000;
/// The table size the bench runs allocate, in MiB.
const BENCH_TT_MB: u64 = 1024;
const RUNS: usize = 3;

/// (label, SFEN). Startpos plus two positions from the existing depth-5 parity
/// fixtures.
const POSITIONS: &[(&str, &str)] = &[
    (
        "startpos",
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
    ),
    (
        "mid-game-tactical",
        "l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1",
    ),
    ("check-evasion", "4k4/9/4r4/9/9/9/4K3B/9/9 b RG2gs2n3p 1"),
];

fn send(stdin: &mut ChildStdin, cmd: &str) {
    stdin.write_all(cmd.as_bytes()).expect("write engine stdin");
    stdin.write_all(b"\n").expect("write newline");
    stdin.flush().expect("flush engine stdin");
}

fn read_until<F: Fn(&str) -> bool>(reader: &mut BufReader<ChildStdout>, pred: F) -> Option<String> {
    let mut line = String::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line).expect("read engine stdout");
        if n == 0 {
            return None;
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if pred(trimmed) {
            return Some(trimmed.to_string());
        }
    }
}

/// Parse the `nodes=<N>` field out of a `bench:` summary line.
fn parse_bench_nodes(summary: &str) -> Option<u64> {
    summary
        .split_whitespace()
        .find_map(|t| t.strip_prefix("nodes="))
        .and_then(|n| n.parse::<u64>().ok())
}

/// Run one `bench … <threads> … current movetime` for `sfen` and return the
/// aggregated node count off its summary line.
fn measure_nodes(
    stdin: &mut ChildStdin,
    reader: &mut BufReader<ChildStdout>,
    sfen: &str,
    threads: u32,
) -> u64 {
    send(stdin, &format!("position sfen {sfen}"));
    send(
        stdin,
        &format!("bench {BENCH_TT_MB} {threads} {MOVETIME_MS} current movetime"),
    );
    let summary = read_until(reader, |l| l.contains("bench: positions="))
        .expect("a bench summary line must arrive");
    parse_bench_nodes(&summary)
        .unwrap_or_else(|| panic!("no nodes= field in bench summary: {summary:?}"))
}

fn median3(mut v: [u64; RUNS]) -> u64 {
    v.sort_unstable();
    v[RUNS / 2]
}

fn start_engine() -> (Child, ChildStdin, BufReader<ChildStdout>) {
    let dir = eval_dir();
    let exe = env!("CARGO_BIN_EXE_yorkie");
    let mut child: Child = Command::new(exe)
        .current_dir(engine_cwd_with_eval_dir(&dir))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn engine");
    let mut stdin = child.stdin.take().expect("stdin piped");
    let mut reader = BufReader::new(child.stdout.take().expect("stdout piped"));

    send(&mut stdin, "usi");
    read_until(&mut reader, |l| l == "usiok").expect("usiok");
    send(&mut stdin, "isready");
    let ack = read_until(&mut reader, |l| {
        l == "readyok" || l.starts_with("info string eval load failed")
    })
    .expect("readyok or load failure");
    assert_eq!(ack, "readyok", "real network must load");
    (child, stdin, reader)
}

#[test]
#[ignore = "spawns the engine and searches about 90 s; run explicitly"]
fn nps_threads1_vs_threads2() {
    let dir = eval_dir();
    if !dir.join("nn.bin").exists() {
        eprintln!(
            "skipping nps_threads1_vs_threads2: {} is not present (obtained out-of-band)",
            dir.join("nn.bin").display()
        );
        return;
    }

    // median NPS per (position index, threads-1 or threads-2).
    let mut medians: Vec<(&str, f64, f64)> = Vec::new();

    for &(label, sfen) in POSITIONS {
        let nps_for = |threads: u32| -> f64 {
            let (mut child, mut stdin, mut reader) = start_engine();
            let mut runs = [0u64; RUNS];
            for r in runs.iter_mut() {
                *r = measure_nodes(&mut stdin, &mut reader, sfen, threads);
            }
            send(&mut stdin, "quit");
            drop(stdin);
            let _ = child.wait();
            median3(runs) as f64 / (MOVETIME_MS as f64 / 1000.0)
        };
        let nps1 = nps_for(1);
        let nps2 = nps_for(2);
        medians.push((label, nps1, nps2));
    }

    eprintln!("\nNPS scaling (median of {RUNS} runs, bench movetime {MOVETIME_MS}):");
    eprintln!(
        "{:<20} {:>14} {:>14} {:>8}",
        "position", "1-worker NPS", "2-worker NPS", "ratio"
    );
    for (label, nps1, nps2) in &medians {
        let ratio = if *nps1 > 0.0 { nps2 / nps1 } else { 0.0 };
        eprintln!("{label:<20} {nps1:>14.0} {nps2:>14.0} {ratio:>8.2}");
    }

    // Measurement only — assert we actually got numbers, never a scaling factor.
    assert!(
        medians.iter().all(|(_, n1, n2)| *n1 > 0.0 && *n2 > 0.0),
        "every measurement must produce a positive node count"
    );
}
