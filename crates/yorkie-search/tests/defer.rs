//! Parallel workers putting off a move another worker is searching, from the
//! outside: one worker with it on searches exactly as the build without it, and
//! two workers with it on — with the move-order spread too, in a build that has
//! both — finish the depth-5 search fixtures with a legal move and leave the
//! table of busy moves empty, also when a search is stopped in the middle.
//! Skipped with a notice when `nn.bin` is absent.

#![cfg(lazy_smp_defer)]

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use serde::Deserialize;
use yorkie_search::defer::BusyTable;
use yorkie_search::{QSearch, SearchControl, TallySlot, WorkerResult, generate_root_moves};
mod serial;
mod text_str;

#[cfg(not(feature = "tt-entry16"))]
use text_str::format_usi_move;
use text_str::{parse_sfen, parse_usi_move};
use yorkie_state::{Move, Position};
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

fn is_legal(pos: &Position, mv: Move) -> bool {
    let mut legal = Vec::new();
    pos.generate_legal_all(&mut legal);
    legal.contains(&mv)
}

/// The switch turned on in a pool of one worker — what a build with the config
/// key on and `threads = 1` constructs — leaves the search exactly the build
/// without it: the fixtures' node counts and best moves.
#[cfg_attr(miri, ignore)]
#[test]
fn one_worker_with_defer_on_matches_the_fixtures() {
    let _search = serial::serial_search();
    let Some(owned) = network() else { return };
    let net = owned.network();
    for name in FIXTURES {
        let fixture = load_fixture(name);
        TranspositionTable::shared().clear();
        TranspositionTable::shared().new_search();
        let pos = setup(&fixture);
        let mut search = QSearch::new(net);
        search.set_defer(true, 1);
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
        #[cfg(feature = "verbose1")]
        assert_eq!(
            result.defer,
            yorkie_search::defer::DeferCounts::default(),
            "{name}: one worker counts nothing"
        );
        assert!(is_legal(&pos, result.best.mv), "{name}");
        assert_eq!(BusyTable::shared().occupancy(), 0, "{name}");
    }
}

static SLOTS: [TallySlot; 2] = [TallySlot::new(), TallySlot::new()];

/// Which of the parallel-search switches the two workers turn on.
#[derive(Clone, Copy, Debug)]
enum Switches {
    Defer,
    #[cfg(lazy_smp_spread)]
    DeferAndSpread,
}

/// Run two workers with `switches` on over `pos` to `depth` on the shared
/// table, each on its own thread, the way the pool runs them.
fn two_workers<N: yorkie_eval::NetworkParams + Copy + Send>(
    net: N,
    pos: &Position,
    depth: i32,
    switches: Switches,
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
                search.set_defer(true, 2);
                match switches {
                    Switches::Defer => {}
                    #[cfg(lazy_smp_spread)]
                    Switches::DeferAndSpread => search.set_spread(true, 2),
                }
                search.run_worker(pos, generate_root_moves(pos), depth)
            })
        });
        workers.map(|w| w.join().expect("a worker finishes"))
    })
}

/// Two workers with `switches` on finish every fixture with a legal move and
/// leave no entry in the table; between them they enter moves.
fn two_workers_finish_the_fixtures(switches: Switches) {
    let _search = serial::serial_search();
    let Some(owned) = network() else { return };
    let net = owned.network();
    #[cfg(feature = "verbose1")]
    let mut total = yorkie_search::defer::DeferCounts::default();
    for name in FIXTURES {
        let fixture = load_fixture(name);
        TranspositionTable::shared().clear();
        let pos = setup(&fixture);
        for r in two_workers(net, &pos, fixture.depth, switches, SearchControl::default) {
            assert_eq!(r.completed_depth, fixture.depth, "{name}");
            assert!(is_legal(&pos, r.best.mv), "{name}");
            #[cfg(feature = "verbose1")]
            total.add(&r.defer);
        }
        assert_eq!(BusyTable::shared().occupancy(), 0, "{name}");
    }
    // Whether a move is put off at all depends on the two threads running at
    // once, which a loaded machine does not promise, so only what holds under
    // any scheduling is checked.
    #[cfg(feature = "verbose1")]
    {
        assert!(total.entries > 0, "{switches:?}: {total:?}");
        assert!(total.tt_cuts <= total.deferred, "{switches:?}: {total:?}");
    }
}

static STOP: AtomicBool = AtomicBool::new(false);

/// A search stopped from outside unwinds from wherever its workers were — the
/// clock's stop takes the same path — and leaves no entry in the table.
fn a_stopped_search_leaves_the_table_empty(switches: Switches) {
    let _search = serial::serial_search();
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
        two_workers(net, &pos, 64, switches, || SearchControl {
            stop: Some(&STOP),
            ..SearchControl::default()
        })
    });
    for r in &results {
        assert!(r.completed_depth < 64, "{switches:?}");
        assert!(is_legal(&pos, r.best.mv), "{switches:?}");
    }
    assert_eq!(BusyTable::shared().occupancy(), 0, "{switches:?}");
}

#[cfg_attr(miri, ignore)]
#[test]
fn two_workers_with_defer_on_finish_the_fixtures() {
    two_workers_finish_the_fixtures(Switches::Defer);
}

#[cfg_attr(miri, ignore)]
#[test]
fn a_stopped_search_with_defer_on_leaves_the_table_empty() {
    a_stopped_search_leaves_the_table_empty(Switches::Defer);
}

#[cfg(lazy_smp_spread)]
#[cfg_attr(miri, ignore)]
#[test]
fn two_workers_with_defer_and_spread_on_finish_the_fixtures() {
    two_workers_finish_the_fixtures(Switches::DeferAndSpread);
}

#[cfg(lazy_smp_spread)]
#[cfg_attr(miri, ignore)]
#[test]
fn a_stopped_search_with_defer_and_spread_on_leaves_the_table_empty() {
    a_stopped_search_leaves_the_table_empty(Switches::DeferAndSpread);
}
