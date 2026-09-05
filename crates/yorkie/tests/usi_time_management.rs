//! End-to-end USI session tests for depth wiring, asynchronous stop, and time
//! management, driving the built `yorkie` binary as a subprocess.
//!
//! These pin the USI layer to the already-gated search: `go depth 2` / `go
//! depth 3` must reproduce the reference fixtures through the real driver, and
//! the time-managed forms (`go infinite` + `stop`, `go movetime`, a Fischer
//! mini-game) must each terminate promptly with exactly one `bestmove`.
//!
//! The whole file is skipped with a notice when `nn.bin` is absent.
//!
//! Gated on `verbose2`: the session drives analysis-only `go` clauses, every
//! assertion reads the search `info depth …` line (its `nodes` and `score`
//! fields), and the spawned `yorkie` binary reaches that level only when the
//! test binary does.

#![cfg(feature = "verbose2")]

mod common;

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::{Duration, Instant};

use common::{engine_cwd_with_eval_dir, eval_dir};
use serde::Deserialize;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures")
}

/// The asserted subset of a search fixture. The depth-2/3 test asserts `bestmove`
/// plus the `nodes` count and the centipawn `score` parsed from the final `info`
/// line — the class of drift a transposition-table refactor could introduce.
#[derive(Debug, Deserialize)]
struct Fixture {
    #[serde(default)]
    moves: Vec<String>,
    bestmove: String,
    nodes: u64,
    score: Score,
}

/// The `score` object of a fixture. Every fixture is a non-mate centipawn
/// score, so only the `cp` arm is modelled.
#[derive(Debug, Deserialize)]
struct Score {
    cp: i64,
}

fn load_fixture(rel: &str) -> Fixture {
    let path = fixtures_dir().join(rel);
    let raw =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse {rel}: {e}"))
}

/// A live USI session over the spawned engine binary.
struct Engine {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Engine {
    /// Spawn the engine in a working directory whose `EvalDir` links to the
    /// staged network, and complete the `usi` / `isready` handshake. Returns
    /// `None` (with a printed notice) when the network is absent, so callers can
    /// skip.
    fn start() -> Option<Self> {
        let dir = eval_dir();
        if !dir.join("nn.bin").exists() {
            eprintln!(
                "skipping usi_time_management: {} is not present (obtained out-of-band)",
                dir.join("nn.bin").display()
            );
            return None;
        }

        let exe = env!("CARGO_BIN_EXE_yorkie");
        let mut child = Command::new(exe)
            .current_dir(engine_cwd_with_eval_dir(&dir))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn engine");
        let stdin = child.stdin.take().expect("stdin piped");
        let stdout = BufReader::new(child.stdout.take().expect("stdout piped"));

        let mut eng = Engine {
            child,
            stdin,
            stdout,
        };
        eng.send("usi");
        eng.read_until(|l| l == "usiok");
        eng.send("isready");
        eng.read_until(|l| l == "readyok")
            .expect("network must load (readyok)");
        Some(eng)
    }

    fn send(&mut self, cmd: &str) {
        self.stdin
            .write_all(cmd.as_bytes())
            .expect("write engine stdin");
        self.stdin.write_all(b"\n").expect("write newline");
        self.stdin.flush().expect("flush engine stdin");
    }

    /// Read (trimmed) lines until one satisfies `pred`; returns it, or `None` on
    /// EOF.
    fn read_until<F: Fn(&str) -> bool>(&mut self, pred: F) -> Option<String> {
        let mut line = String::new();
        loop {
            line.clear();
            let n = self
                .stdout
                .read_line(&mut line)
                .expect("read engine stdout");
            if n == 0 {
                return None;
            }
            let trimmed = line.trim_end_matches(['\r', '\n']);
            if pred(trimmed) {
                return Some(trimmed.to_string());
            }
        }
    }

    /// Read the `info` line of the given iterative-deepening `depth` and pull out
    /// its `nodes` count and centipawn `score cp` value. With `PvInterval 0` every
    /// completed iteration prints a line, not just the last one, so callers must
    /// target the depth they mean explicitly.
    fn read_info_nodes_cp(&mut self, depth: u32) -> (u64, i64) {
        let prefix = format!("info depth {depth} ");
        let line = self
            .read_until(|l| l.starts_with(&prefix) && l.contains(" nodes "))
            .expect("a search info line at the target depth must arrive");
        let toks: Vec<&str> = line.split_whitespace().collect();
        let after = |key: &str| -> &str {
            let i = toks
                .iter()
                .position(|&t| t == key)
                .unwrap_or_else(|| panic!("`{key}` token missing in info line: {line:?}"));
            toks.get(i + 1)
                .unwrap_or_else(|| panic!("value after `{key}` missing in info line: {line:?}"))
        };
        // `score cp <n>`: guard that the score is centipawns, not `mate`.
        assert_eq!(
            after("score"),
            "cp",
            "fixture scores are centipawn scores, got: {line:?}"
        );
        let nodes = after("nodes").parse().expect("nodes is a u64");
        let cp = after("cp").parse().expect("cp is an i64");
        (nodes, cp)
    }

    /// Read the next `bestmove` line and return just the move token (dropping any
    /// ` ponder …` suffix).
    fn read_bestmove(&mut self) -> String {
        let line = self
            .read_until(|l| l.starts_with("bestmove "))
            .expect("a bestmove must arrive");
        line.strip_prefix("bestmove ")
            .unwrap()
            .split_whitespace()
            .next()
            .expect("bestmove token")
            .to_string()
    }

    fn quit(mut self) {
        self.send("quit");
        let status = self.child.wait().expect("wait engine");
        assert!(status.success(), "engine exited non-zero: {status:?}");
    }
}

#[cfg_attr(miri, ignore)]
#[test]
fn go_depth_2_and_3_match_fixtures_via_binary() {
    // The fixture assertions need the single worker and the unthrottled PV the
    // test configs compile in: helpers sharing the TT would move the node count,
    // and a PV throttle would decide which iterations printed by wall clock.
    common::require_test_config();
    let Some(mut eng) = Engine::start() else {
        return;
    };

    // depth 2: position startpos moves 7g7f, go depth 2. Assert the info line's
    // nodes and cp score as well as the bestmove.
    let d2 = load_fixture("search-depth2/startpos-7g7f.json");
    eng.send("usinewgame");
    eng.send(&format!("position startpos moves {}", d2.moves.join(" ")));
    eng.send("go depth 2");
    let (d2_nodes, d2_cp) = eng.read_info_nodes_cp(2);
    assert_eq!(
        d2_nodes, d2.nodes,
        "go depth 2 nodes must match the fixture"
    );
    assert_eq!(
        d2_cp, d2.score.cp,
        "go depth 2 cp score must match the fixture"
    );
    assert_eq!(
        eng.read_bestmove(),
        d2.bestmove,
        "go depth 2 bestmove must match the reference fixture"
    );

    // depth 3: position startpos, go depth 3.
    let d3 = load_fixture("search/startpos.json");
    eng.send("usinewgame");
    eng.send("position startpos");
    eng.send("go depth 3");
    let (d3_nodes, d3_cp) = eng.read_info_nodes_cp(3);
    assert_eq!(
        d3_nodes, d3.nodes,
        "go depth 3 nodes must match the fixture"
    );
    assert_eq!(
        d3_cp, d3.score.cp,
        "go depth 3 cp score must match the fixture"
    );
    assert_eq!(
        eng.read_bestmove(),
        d3.bestmove,
        "go depth 3 bestmove must match the reference fixture"
    );

    eng.quit();
}

#[cfg_attr(miri, ignore)]
#[test]
fn repeated_searches_in_one_session_keep_matching_the_fixture() {
    // The worker pool is built once and reused for every `go` in a session, so a
    // second and third search over the same position must reproduce the first's
    // fixture numbers: a helper left in a dirty state, or histories leaking
    // across `usinewgame`, would show up as drift here.
    common::require_test_config();
    let Some(mut eng) = Engine::start() else {
        return;
    };

    let d2 = load_fixture("search-depth2/startpos-7g7f.json");
    for round in 0..3 {
        eng.send("usinewgame");
        eng.send(&format!("position startpos moves {}", d2.moves.join(" ")));
        eng.send("go depth 2");
        let (nodes, cp) = eng.read_info_nodes_cp(2);
        assert_eq!(
            nodes, d2.nodes,
            "round {round}: node count must match fixture"
        );
        assert_eq!(
            cp, d2.score.cp,
            "round {round}: cp score must match fixture"
        );
        let best = eng.read_bestmove();
        assert_eq!(
            best, d2.bestmove,
            "round {round}: bestmove must match fixture"
        );
        assert_legal_move_after(&d2.moves, &best);
    }

    eng.quit();
}

/// Session smoke tests over the compiled-in worker pool.
/// Each drives the spawned binary; each asserts exactly one *legal* bestmove and
/// prompt termination.
#[cfg_attr(miri, ignore)]
#[test]
fn go_movetime_and_depth_and_infinite_and_fischer_all_terminate() {
    let Some(mut eng) = Engine::start() else {
        return;
    };
    // The deadline is polled only at ~512-node `check_time` checkpoints, and in
    // an unoptimised test build with two workers contending for cores a single
    // checkpoint can be several seconds. These bounds are therefore deliberately
    // loose: they prove the search self-terminates near its budget rather than
    // running to the depth ceiling.
    let bound = Duration::from_secs(10);

    // (a) go movetime 300 → one legal bestmove within a generous bound.
    eng.send("usinewgame");
    eng.send("position startpos");
    let t = Instant::now();
    eng.send("go movetime 300");
    let best = eng.read_bestmove();
    assert!(t.elapsed() < bound, "go movetime 300 took too long");
    assert_legal_move_after(&[], &best);

    // (c) go depth 3 → a legal bestmove (result nondeterministic under SMP).
    eng.send("usinewgame");
    eng.send("position startpos");
    eng.send("go depth 3");
    let best = eng.read_bestmove();
    assert_legal_move_after(&[], &best);

    // (d) go infinite + stop → one prompt bestmove.
    eng.send("usinewgame");
    eng.send("position startpos");
    eng.send("go infinite");
    std::thread::sleep(Duration::from_millis(250));
    let t = Instant::now();
    eng.send("stop");
    let best = eng.read_bestmove();
    assert!(t.elapsed() < bound, "bestmove after stop took too long");
    assert_legal_move_after(&[], &best);

    // (b) Fischer mini-game → every move meets its deadline with one bestmove.
    const CLOCK: u64 = 300;
    const INC: u64 = 200;
    eng.send("usinewgame");
    let mut moves: Vec<String> = Vec::new();
    for ply in 0..6 {
        if moves.is_empty() {
            eng.send("position startpos");
        } else {
            eng.send(&format!("position startpos moves {}", moves.join(" ")));
        }
        let t = Instant::now();
        eng.send(&format!(
            "go btime {CLOCK} wtime {CLOCK} binc {INC} winc {INC}"
        ));
        let best = eng.read_bestmove();
        assert!(t.elapsed() < bound, "ply {ply}: missed deadline");
        if best == "resign" || best == "win" {
            break;
        }
        assert_legal_move_after(&moves, &best);
        moves.push(best);
    }
    assert!(
        moves.len() >= 2,
        "mini-game should play several moves, got {moves:?}"
    );

    eng.quit();
}

/// Assert `best` is a legal move in the position reached by applying `setup`
/// (USI move strings) from startpos. `resign` / `win` are *rejected* here: every
/// call site breaks out of its loop on a terminal reply before reaching this
/// guard, so a `resign` / `win` arriving here is an unexpected result and fails
/// the assertion below.
fn assert_legal_move_after(setup: &[String], best: &str) {
    assert!(
        !best.is_empty() && best != "resign" && best != "win",
        "expected a real move, got {best:?}"
    );
    let mut pos = yorkie_state::parse_sfen(yorkie_state::STARTPOS_SFEN).expect("startpos SFEN");
    for m in setup {
        let mv = yorkie_state::parse_usi_move(m, &pos).expect("legal setup move");
        pos.do_move(mv);
    }
    let mv = yorkie_state::parse_usi_move(best, &pos)
        .unwrap_or_else(|_| panic!("bestmove {best:?} is not a well-formed USI move"));
    let mut legal = Vec::new();
    pos.generate_legal_all(&mut legal);
    assert!(legal.contains(&mv), "bestmove {best:?} is not legal");
}

#[cfg_attr(miri, ignore)]
#[test]
fn go_infinite_then_stop_yields_one_prompt_bestmove() {
    // Single-worker timing test: the bound below holds at the one worker the
    // test config compiles in, where the ~512-node `check_time` cadence is not
    // slowed by helper CPU contention. Timing under a bigger pool is covered by
    // `go_movetime_and_depth_and_infinite_and_fischer_all_terminate`, whose
    // bounds are sized for contention.
    common::require_test_config();
    let Some(mut eng) = Engine::start() else {
        return;
    };

    eng.send("usinewgame");
    eng.send("position startpos");
    eng.send("go infinite");

    // Let the search run, then ask it to stop and time how long bestmove takes.
    std::thread::sleep(Duration::from_millis(250));
    let t = Instant::now();
    eng.send("stop");
    let best = eng.read_bestmove();
    let elapsed = t.elapsed();

    assert!(
        !best.is_empty() && best != "resign",
        "go infinite on startpos must return a real move, got {best:?}"
    );
    // Bounded by the ~512-node check granularity, not by an iteration boundary.
    // Generous for a debug build under load.
    assert!(
        elapsed < Duration::from_secs(3),
        "bestmove after stop took too long: {elapsed:?}"
    );

    eng.quit();
}

#[cfg_attr(miri, ignore)]
#[test]
fn go_movetime_returns_within_a_generous_bound() {
    // Single-worker timing test (see the note in the go-infinite test above).
    common::require_test_config();
    let Some(mut eng) = Engine::start() else {
        return;
    };

    eng.send("usinewgame");
    eng.send("position startpos");
    let t = Instant::now();
    eng.send("go movetime 300");
    let best = eng.read_bestmove();
    let elapsed = t.elapsed();

    assert!(
        !best.is_empty() && best != "resign",
        "go movetime on startpos must return a real move, got {best:?}"
    );
    // 300 ms budget; allow ample slack for process scheduling in a debug build.
    assert!(
        elapsed < Duration::from_secs(3),
        "go movetime 300 took too long: {elapsed:?}"
    );

    eng.quit();
}

#[cfg_attr(miri, ignore)]
#[test]
fn fischer_mini_game_makes_every_deadline_with_one_bestmove_each() {
    // Single-worker timing test (see the note in the go-infinite test above).
    common::require_test_config();
    let Some(mut eng) = Engine::start() else {
        return;
    };

    // Small Fischer budgets: each move's hard deadline is remaining + increment,
    // so a bestmove must arrive well within that. We replay the engine's own
    // choices to walk a short game.
    const BTIME: u64 = 300;
    const WTIME: u64 = 300;
    const INC: u64 = 200;
    // A generous ceiling on the per-move wall clock: the deadline is polled only
    // at ~512-node `check_time` checkpoints, and in an unoptimised test build a
    // checkpoint is roughly half a second of compute, so a move can overshoot
    // its nominal budget by up to one checkpoint.
    let per_move_bound = Duration::from_secs(3);

    eng.send("usinewgame");
    let mut moves: Vec<String> = Vec::new();

    for ply in 0..6 {
        if moves.is_empty() {
            eng.send("position startpos");
        } else {
            eng.send(&format!("position startpos moves {}", moves.join(" ")));
        }
        let t = Instant::now();
        eng.send(&format!(
            "go btime {BTIME} wtime {WTIME} binc {INC} winc {INC}"
        ));
        let best = eng.read_bestmove();
        let elapsed = t.elapsed();

        assert!(
            elapsed < per_move_bound,
            "ply {ply}: missed deadline ({elapsed:?} >= {per_move_bound:?})"
        );

        if best == "resign" || best == "win" {
            break; // a terminal result ends the mini-game early.
        }
        moves.push(best);
    }

    // At least a couple of real plies were played under the clock.
    assert!(
        moves.len() >= 2,
        "expected the mini-game to play several moves, got {moves:?}"
    );

    eng.quit();
}

#[cfg_attr(miri, ignore)]
#[test]
fn byoyomi_mini_game_makes_every_deadline_with_one_bestmove_each() {
    // Single-worker timing test (see the note in the go-infinite test above).
    common::require_test_config();
    let Some(mut eng) = Engine::start() else {
        return;
    };

    // A byoyomi game with the main clock exhausted, so every move has only the
    // byoyomi period. This is the reference's "final push" shape, where
    // `time[us] < byoyomi * 1.2` makes the manager spend the byoyomi.
    //
    // The per-move wall bound is loose for the same checkpoint-granularity
    // reason as the Fischer test above.
    const BYOYOMI: u64 = 1000;
    let per_move_bound = Duration::from_secs(3);

    eng.send("usinewgame");
    let mut moves: Vec<String> = Vec::new();

    for ply in 0..6 {
        if moves.is_empty() {
            eng.send("position startpos");
        } else {
            eng.send(&format!("position startpos moves {}", moves.join(" ")));
        }
        let t = Instant::now();
        eng.send(&format!("go btime 0 wtime 0 byoyomi {BYOYOMI}"));
        let best = eng.read_bestmove();
        let elapsed = t.elapsed();

        assert!(
            elapsed < per_move_bound,
            "ply {ply}: byoyomi move missed deadline ({elapsed:?} >= {per_move_bound:?})"
        );

        if best == "resign" || best == "win" {
            break; // a terminal result ends the mini-game early.
        }
        moves.push(best);
    }

    assert!(
        moves.len() >= 2,
        "expected the byoyomi mini-game to play several moves, got {moves:?}"
    );

    eng.quit();
}
