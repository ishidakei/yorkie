//! The move-order spread of parallel workers, from the outside: one worker
//! with the spread on searches exactly as the build without it, and two
//! workers with it on both finish the depth-5 search fixtures with a legal
//! move, having found each other's results and shifted picks because of them.
//! Skipped with a notice when `nn.bin` is absent.

#![cfg(lazy_smp_spread)]

use std::path::PathBuf;

use serde::Deserialize;
use yorkie_search::{QSearch, TallySlot, WorkerResult, generate_root_moves};
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

/// The spread turned on in a pool of one worker — what a build with the config
/// key on and `threads = 1` constructs — leaves the search exactly the build
/// without it: the fixtures' node counts and best moves.
#[cfg_attr(miri, ignore)]
#[test]
fn one_worker_with_the_spread_on_matches_the_fixtures() {
    let Some(owned) = network() else { return };
    let net = owned.network();
    for name in FIXTURES {
        let fixture = load_fixture(name);
        TranspositionTable::shared().clear();
        TranspositionTable::shared().new_search();
        let pos = setup(&fixture);
        let mut search = QSearch::new(net);
        search.set_spread(true, 1);
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
            result.spread,
            yorkie_search::spread::SpreadCounts::default(),
            "{name}: one worker counts nothing"
        );
        assert!(is_legal(&pos, result.best.mv), "{name}");
    }
}

static SLOTS: [TallySlot; 2] = [TallySlot::new(), TallySlot::new()];

/// Run two workers with the spread on over `pos` to `depth` on the shared
/// table, each on its own thread, the way the pool runs them.
fn two_workers<N: yorkie_eval::NetworkParams + Copy + Send>(
    net: N,
    pos: &Position,
    depth: i32,
) -> [WorkerResult; 2] {
    TranspositionTable::shared().new_search();
    std::thread::scope(|scope| {
        let workers = [0, 1].map(|index| {
            scope.spawn(move || {
                let mut search = QSearch::new(net);
                search.set_best_move_tally(&SLOTS, index);
                search.set_spread(true, 2);
                search.run_worker(pos, generate_root_moves(pos), depth)
            })
        });
        workers.map(|w| w.join().expect("a worker finishes"))
    })
}

/// Two workers with the spread on both finish every fixture with a legal
/// move, and between them count duplicates, shifted picks and a peak.
#[cfg_attr(miri, ignore)]
#[test]
fn two_workers_with_the_spread_on_finish_the_fixtures() {
    let Some(owned) = network() else { return };
    let net = owned.network();
    #[cfg(feature = "verbose1")]
    let mut total = yorkie_search::spread::SpreadCounts::default();
    for name in FIXTURES {
        let fixture = load_fixture(name);
        TranspositionTable::shared().clear();
        let pos = setup(&fixture);
        for r in two_workers(net, &pos, fixture.depth) {
            assert_eq!(r.completed_depth, fixture.depth, "{name}");
            assert!(is_legal(&pos, r.best.mv), "{name}");
            #[cfg(feature = "verbose1")]
            {
                assert!(r.spread.peak <= 2, "{name}: {:?}", r.spread);
                total.add(&r.spread);
            }
        }
    }
    #[cfg(feature = "verbose1")]
    {
        assert!(total.dups > 0, "{total:?}");
        assert!(total.shifts > 0, "{total:?}");
        assert!(total.peak > 0, "{total:?}");
    }
}
