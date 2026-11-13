//! The transposition table's replacement depth margin (the
//! `tt_replace_depth_margin` setting), from the outside: two workers sharing
//! the table finish the depth-5 search fixtures at their depth with a legal
//! move at the margin the build carries. The margin is a compile-time
//! constant, so a margin other than the config's is tested by building these
//! tests with a config that sets it (`0`, say). Skipped with a notice when
//! `nn.bin` is absent.

use std::path::PathBuf;

use serde::Deserialize;
use yorkie_search::{QSearch, TallySlot, WorkerResult, generate_root_moves};
mod text_str;

use text_str::{parse_sfen, parse_usi_move};
use yorkie_state::{Move, Position};
use yorkie_storage::{REPLACE_DEPTH_MARGIN, TranspositionTable};

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

static SLOTS: [TallySlot; 2] = [TallySlot::new(), TallySlot::new()];

/// Run two workers over `pos` to `depth` on the shared table, each on its own
/// thread, the way the pool runs them.
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
                search.run_worker(pos, generate_root_moves(pos), depth)
            })
        });
        workers.map(|w| w.join().expect("a worker finishes"))
    })
}

#[test]
fn the_margin_is_in_the_accepted_range() {
    assert!((0..=8).contains(&REPLACE_DEPTH_MARGIN));
}

/// Two workers finish every fixture at its depth with a legal move.
#[cfg_attr(miri, ignore)]
#[test]
fn two_workers_finish_the_fixtures_at_the_built_margin() {
    let Some(owned) = network() else { return };
    let net = owned.network();
    for name in FIXTURES {
        let fixture = load_fixture(name);
        TranspositionTable::shared().clear();
        let pos = setup(&fixture);
        for r in two_workers(net, &pos, fixture.depth) {
            assert_eq!(
                r.completed_depth, fixture.depth,
                "{name} at margin {REPLACE_DEPTH_MARGIN}"
            );
            assert!(
                is_legal(&pos, r.best.mv),
                "{name} at margin {REPLACE_DEPTH_MARGIN}"
            );
        }
    }
}
