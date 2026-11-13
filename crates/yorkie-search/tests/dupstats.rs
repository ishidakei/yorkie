//! The duplication statistics of a `dupstats` build, from the outside: what one
//! worker and two workers count over the depth-5 search fixtures, what is left
//! in the statistics-only tables once a search has ended, and the record each
//! worker keeps of when it finished each root iteration.
//!
//! With one worker the counts must not change the search — the fixtures' node
//! counts and best moves are the plain build's — and nothing can be another
//! worker's. With two, both workers finish the fixtures and each finds work the
//! other did. Skipped with a notice when `nn.bin` is absent.

#![cfg(feature = "dupstats")]

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use serde::Deserialize;
use yorkie_search::dupstats::{
    DupCounts, PresenceKind, StoreKind, Writer, path_len, presence_occupancy,
};
use yorkie_search::timeline::{Iteration, RootEntry};
use yorkie_search::{QSearch, SearchControl, TallySlot, WorkerResult, generate_root_moves};
mod text_str;

#[cfg(not(feature = "tt-entry16"))]
use text_str::format_usi_move;
use text_str::{parse_sfen, parse_usi_move};
use yorkie_state::Position;
use yorkie_storage::TranspositionTable;

const FIXTURES: &[&str] = &[
    "startpos.json",
    "drop-heavy.json",
    "mid-game-tactical.json",
    "check-evasion.json",
    "promotion-zone-edges.json",
    "sennichite.json",
];

#[derive(Debug, Deserialize)]
struct FixtureJson {
    sfen: String,
    #[serde(default)]
    moves: Vec<String>,
    depth: i32,
    // Read only where the reference's numbers are this build's.
    #[cfg_attr(feature = "tt-entry16", allow(dead_code))]
    bestmove: String,
    #[cfg_attr(feature = "tt-entry16", allow(dead_code))]
    nodes: u64,
}

fn nn_bin_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../eval/nn.bin")
}

fn network() -> Option<yorkie_eval::OwnedNetwork> {
    let path = nn_bin_path();
    if !path.exists() {
        eprintln!("skipping: {} is not present", path.display());
        return None;
    }
    Some(yorkie_eval::load_network(&path).expect("real nn.bin should load and validate"))
}

fn load_fixture(name: &str) -> FixtureJson {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/search-depth5")
        .join(name);
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read fixture {}: {e}", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse fixture {name}: {e}"))
}

fn setup(fixture: &FixtureJson) -> Position {
    let mut pos = parse_sfen(&fixture.sfen).expect("valid fixture SFEN");
    for usi in &fixture.moves {
        let m = parse_usi_move(usi, &pos).unwrap_or_else(|e| panic!("bad move {usi}: {e:?}"));
        pos.do_move(m);
    }
    pos
}

/// The counts that can only come from another worker.
fn other_worker_counts(dup: &DupCounts) -> [u64; 6] {
    [
        dup.stores_of(StoreKind::Deepened(Writer::OtherWorker)),
        dup.stores_of(StoreKind::AlreadyThere(Writer::OtherWorker)),
        dup.presence_of(PresenceKind::Deeper),
        dup.presence_of(PresenceKind::Equal),
        dup.presence_of(PresenceKind::Shallower),
        dup.dup_nodes,
    ]
}

fn stores(dup: &DupCounts) -> u64 {
    StoreKind::ALL.iter().map(|k| dup.stores_of(*k)).sum()
}

/// Check the timeline of a worker that completed `completed` iterations and,
/// when `stopped`, was stopped inside the next one: one completed record per
/// depth from 1, then the stopped one marked not completed, at times that never
/// go back and node counts that only grow, the last at the worker's own count.
fn check_timeline(r: &WorkerResult, completed: i32, stopped: bool, what: &str) {
    let its: Vec<&Iteration> = r.timeline.iterations().collect();
    let want = completed as usize + usize::from(stopped);
    assert_eq!(its.len(), want, "{what}: {its:?}");
    for (i, it) in its.iter().enumerate() {
        assert_eq!(it.depth, i as i32 + 1, "{what}: {it:?}");
        assert_eq!(it.completed, it.depth <= completed, "{what}: {it:?}");
        assert!(it.best.is_some(), "{what}: {it:?}");
        assert!(
            it.first_move_nodes + it.other_move_nodes <= it.nodes,
            "{what}: {it:?}"
        );
    }
    for pair in its.windows(2) {
        assert!(pair[0].micros <= pair[1].micros, "{what}: {pair:?}");
        assert!(pair[0].nodes < pair[1].nodes, "{what}: {pair:?}");
    }
    if let Some(last) = its.last() {
        assert_eq!(last.nodes, r.nodes, "{what}: the last record is the end");
    }
}

/// One worker, as the fixed-depth search runs: the fixtures come out exactly
/// as the plain build's, and no count is another worker's.
#[cfg_attr(miri, ignore)]
#[test]
fn one_worker_matches_the_fixtures_and_sees_no_other_worker() {
    let Some(owned) = network() else { return };
    let net = owned.network();
    let mut first_move_nodes = 0;
    for name in FIXTURES {
        let fixture = load_fixture(name);
        TranspositionTable::shared().clear();
        TranspositionTable::shared().new_search();
        let pos = setup(&fixture);
        let mut search = QSearch::new(net);
        let result = search.run_worker(&pos, generate_root_moves(&pos), fixture.depth);

        // A wide table finds other hits, so the reference's numbers are not
        // its numbers.
        #[cfg(not(feature = "tt-entry16"))]
        {
            assert_eq!(result.nodes, fixture.nodes, "{name}: nodes");
            assert_eq!(
                format_usi_move(result.best.mv),
                fixture.bestmove,
                "{name}: bestmove"
            );
        }
        assert!(stores(&result.dup) > 0, "{name}: {:?}", result.dup);
        assert!(
            result.dup.presence_of(PresenceKind::Entered) > 0,
            "{name}: {:?}",
            result.dup
        );
        assert_eq!(other_worker_counts(&result.dup), [0; 6], "{name}");
        assert_eq!(result.stopped_on, None, "{name}: ran to its depth");
        check_timeline(&result, fixture.depth, false, name);
        for it in result.timeline.iterations() {
            assert_ne!(it.root_entry, RootEntry::OtherWorker, "{name}: {it:?}");
            first_move_nodes += it.first_move_nodes;
        }
        assert_eq!(presence_occupancy(), 0, "{name}");
        assert_eq!(path_len(), 0, "{name}");
    }
    // A root answered without a move loop (a mate found before it) splits
    // nothing, but the fixtures between them search their root moves.
    assert!(first_move_nodes > 0);
}

static SLOTS: [TallySlot; 2] = [TallySlot::new(), TallySlot::new()];

/// Run two workers over `pos` to `depth` on the shared table, each on its own
/// thread, the way the pool runs them.
fn two_workers<N: yorkie_eval::NetworkParams + Copy + Send>(
    net: N,
    pos: &Position,
    depth: i32,
    control: impl Fn() -> SearchControl + Sync,
) -> [WorkerResult; 2] {
    TranspositionTable::shared().new_search();
    std::thread::scope(|scope| {
        let workers = [0, 1].map(|index| {
            let control = &control;
            scope.spawn(move || {
                let mut search = QSearch::new(net);
                search.set_best_move_tally(&SLOTS, index);
                search.set_control(control());
                let result = search.run_worker(pos, generate_root_moves(pos), depth);
                assert_eq!(path_len(), 0, "worker {index} left frames on its path");
                result
            })
        });
        workers.map(|w| w.join().expect("a worker finishes"))
    })
}

/// Two workers both finish every fixture, and between them find work the
/// other one did.
#[cfg_attr(miri, ignore)]
#[test]
fn two_workers_finish_the_fixtures_and_find_each_others_work() {
    let Some(owned) = network() else { return };
    let net = owned.network();
    let mut total = DupCounts::ZERO;
    for name in FIXTURES {
        let fixture = load_fixture(name);
        TranspositionTable::shared().clear();
        let pos = setup(&fixture);
        let results = two_workers(net, &pos, fixture.depth, SearchControl::default);
        for r in &results {
            assert_eq!(r.completed_depth, fixture.depth, "{name}");
            total.add(&r.dup);
            check_timeline(r, fixture.depth, false, name);
        }
        assert_eq!(results[0].timeline.worker, 0, "{name}");
        assert_eq!(results[1].timeline.worker, 1, "{name}");
        assert_eq!(presence_occupancy(), 0, "{name}");
    }
    let other: u64 = other_worker_counts(&total)[..2].iter().sum();
    assert!(other > 0, "{total:?}");
}

static STOP: AtomicBool = AtomicBool::new(false);

/// A search stopped by the clock unwinds from wherever its workers were, and
/// leaves no worker registered in the presence table and no frame on any path.
#[cfg_attr(miri, ignore)]
#[test]
fn a_search_stopped_by_time_leaves_the_tables_empty() {
    let Some(owned) = network() else { return };
    let net = owned.network();
    TranspositionTable::shared().clear();
    let pos = Position::startpos();
    STOP.store(false, Ordering::Relaxed);
    let results = std::thread::scope(|scope| {
        scope.spawn(|| {
            std::thread::sleep(Duration::from_millis(300));
            STOP.store(true, Ordering::Relaxed);
        });
        two_workers(net, &pos, 64, || SearchControl {
            stop: Some(&STOP),
            ..SearchControl::default()
        })
    });
    for r in &results {
        assert!(r.completed_depth < 64);
        assert!(r.stopped_on.is_some(), "stopped inside an iteration");
        check_timeline(r, r.completed_depth, true, "stopped by time");
    }
    assert_eq!(presence_occupancy(), 0);
}
