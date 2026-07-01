//! NNUE evaluation function (SFNNwoP1536), derived from YaneuraOu
//! (https://github.com/yaneurao/YaneuraOu).
#![allow(dead_code)]

pub mod aligned;
pub mod bucket;
pub mod features;
pub mod loader;
pub mod network;
#[cfg(feature = "numa")]
pub mod numa;
pub mod simd;
pub mod transformer;
pub mod types;

#[cfg(test)]
pub(crate) mod test_fixtures;

#[allow(unused_imports)]
pub use types::{Accumulator, FEATURE_LIST_CAPACITY, FeatureIndex, FeatureList, HIDDEN_SIZE, NetHeader, NnueError, NnueNetwork};

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

#[cfg(test)]
use aligned::Aligned64;

use crate::position::*;
use crate::search::{Stack, get_stack_mut};
use crate::types::*;

struct LoadedNetwork {
    net: Arc<NnueNetwork>,
    path: PathBuf,
}

static NETWORK: RwLock<Option<LoadedNetwork>> = RwLock::new(None);

// Node-local NNUE replica for this worker (`numa` feature), set at `worker()` entry; the eval hot path reads it instead of the shared `Arc`+`RwLock`. Null = fall back to global `NETWORK`.
#[cfg(feature = "numa")]
thread_local! {
    static LOCAL_NET: std::cell::Cell<*const NnueNetwork> = const { std::cell::Cell::new(std::ptr::null()) };
}

/// Points this worker at the node-local replica for `node`, else the shared global. Called at `worker()` entry.
#[cfg(feature = "numa")]
pub fn set_local_replica_for_node(node: u32) {
    let ptr = numa::replica_for_node(node).map_or(std::ptr::null(), |net| net as *const NnueNetwork);
    LOCAL_NET.with(|c| c.set(ptr));
}

/// The current thread's node-local replica if set (`numa`); `pub(crate)` so the search's incremental path uses the same replica as full-eval.
#[cfg(feature = "numa")]
pub(crate) fn local_replica() -> Option<&'static NnueNetwork> {
    let ptr = LOCAL_NET.with(|c| c.get());
    // SAFETY: `ptr` is null or a `&'static` replica leaked by `numa::build_replicas` (read-only, process-lifetime).
    if ptr.is_null() { None } else { Some(unsafe { &*ptr }) }
}

pub fn evaluate(pos: &mut Position, stack: &mut [Stack]) -> Value {
    #[cfg(feature = "numa")]
    if let Some(net) = local_replica() {
        return evaluate_with(net, pos, stack);
    }
    let net_arc = match current_network() {
        Some(n) => n,
        None => {
            debug_assert!(false, "nnue::evaluate called without a loaded network");
            return Value::ZERO;
        }
    };
    evaluate_with(&net_arc, pos, stack)
}

fn evaluate_with(net: &NnueNetwork, pos: &mut Position, stack: &mut [Stack]) -> Value {
    let stm = pos.side_to_move();
    let bucket = bucket::select(pos);
    let slot = get_stack_mut(stack, 0);
    debug_assert!(
        slot.accumulator.computed,
        "nnue::evaluate reached with uncomputed accumulator"
    );
    network::forward(&slot.accumulator, net, stm, bucket)
}

pub fn evaluate_at_root(pos: &Position, stack: &mut [Stack]) -> Value {
    #[cfg(feature = "numa")]
    if let Some(net) = local_replica() {
        return evaluate_at_root_with(net, pos, stack);
    }
    let net_arc = match current_network() {
        Some(n) => n,
        None => {
            debug_assert!(false, "nnue::evaluate_at_root called without a loaded network");
            return Value::ZERO;
        }
    };
    evaluate_at_root_with(&net_arc, pos, stack)
}

fn evaluate_at_root_with(net: &NnueNetwork, pos: &Position, stack: &mut [Stack]) -> Value {
    let stm = pos.side_to_move();
    let bucket = bucket::select(pos);
    let slot = get_stack_mut(stack, 0);
    transformer::refresh(&mut slot.accumulator, net, pos);
    network::forward(&slot.accumulator, net, stm, bucket)
}

pub(crate) fn current_network() -> Option<Arc<NnueNetwork>> {
    NETWORK
        .read()
        .expect("nnue::NETWORK read lock poisoned")
        .as_ref()
        .map(|loaded| Arc::clone(&loaded.net))
}

pub fn is_loaded() -> bool {
    NETWORK.read().expect("nnue::NETWORK read lock poisoned").is_some()
}

pub fn loaded_path() -> Option<PathBuf> {
    NETWORK
        .read()
        .expect("nnue::NETWORK read lock poisoned")
        .as_ref()
        .map(|n| n.path.clone())
}

pub fn loaded_sha256_hex() -> Option<String> {
    NETWORK
        .read()
        .expect("nnue::NETWORK read lock poisoned")
        .as_ref()
        .map(|n| hex_lower(&n.net.sha256))
}

pub fn load_network_from_path(path: &Path) -> Result<(), NnueError> {
    let net = Arc::new(loader::load_network(path)?);
    // Build node-local replicas before publishing the global, so workers see a ready table at `isready`.
    #[cfg(feature = "numa")]
    numa::build_replicas(&net);
    let mut slot = NETWORK.write().expect("nnue::NETWORK write lock poisoned");
    *slot = Some(LoadedNetwork {
        net,
        path: path.to_path_buf(),
    });
    Ok(())
}

fn hex_lower(bytes: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0F) as usize] as char);
    }
    out
}

#[cfg(test)]
pub(crate) static TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
pub(crate) fn set_loaded_for_test(net: Arc<NnueNetwork>, path: PathBuf) {
    let mut slot = NETWORK.write().expect("nnue::NETWORK write lock poisoned");
    *slot = Some(LoadedNetwork { net, path });
}

#[cfg(test)]
pub(crate) fn clear_loaded_for_test() {
    let mut slot = NETWORK.write().expect("nnue::NETWORK write lock poisoned");
    *slot = None;
}

#[cfg(all(test, feature = "numa"))]
pub(crate) fn set_local_replica_for_test(net: &'static NnueNetwork) {
    LOCAL_NET.with(|c| c.set(net as *const NnueNetwork));
}

#[cfg(all(test, feature = "numa"))]
pub(crate) fn clear_local_replica_for_test() {
    LOCAL_NET.with(|c| c.set(std::ptr::null()));
}

#[cfg(test)]
pub(crate) fn make_placeholder_network(sha256: [u8; 32]) -> NnueNetwork {
    NnueNetwork {
        header: NetHeader {
            version: 0,
            hash: 0,
            arch_id: "placeholder".to_string(),
        },
        ft_biases: Aligned64::zeroed(0),
        ft_weights: Aligned64::zeroed(0),
        stacks: Vec::new(),
        sha256,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loaded_accessors_track_slot_state() {
        let _guard = TEST_MUTEX.lock().expect("TEST_MUTEX poisoned");
        clear_loaded_for_test();

        assert!(!is_loaded());
        assert_eq!(loaded_path(), None);
        assert_eq!(loaded_sha256_hex(), None);

        let mut sha = [0u8; 32];
        sha[0] = 0xAB;
        sha[31] = 0xCD;
        let net = Arc::new(make_placeholder_network(sha));
        let first_path = PathBuf::from("/tmp/first.nnue");
        set_loaded_for_test(net, first_path.clone());

        assert!(is_loaded());
        assert_eq!(loaded_path(), Some(first_path.clone()));
        let hex = loaded_sha256_hex().expect("sha256 present");
        assert_eq!(hex.len(), 64);
        assert!(hex.starts_with("ab"));
        assert!(hex.ends_with("cd"));
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));

        clear_loaded_for_test();
        assert!(!is_loaded());
        assert_eq!(loaded_path(), None);
        assert_eq!(loaded_sha256_hex(), None);
    }

    #[test]
    fn hex_lower_encodes_full_range() {
        let mut bytes = [0u8; 32];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = i as u8;
        }
        let s = hex_lower(&bytes);
        assert_eq!(&s[..6], "000102");
        assert_eq!(&s[62..], "1f");
        assert_eq!(s.len(), 64);
    }

    #[test]
    fn evaluate_is_deterministic() {
        use crate::search::CURRENT_STACK_INDEX;
        test_fixtures::run_with_large_stack(|| {
            let _guard = TEST_MUTEX.lock().expect("TEST_MUTEX poisoned");
            clear_loaded_for_test();
            set_loaded_for_test(test_fixtures::synthetic_net_arc(), PathBuf::from("synthetic"));

            let mut pos = Position::new();
            let mut stack_a: Vec<Stack> = (0..CURRENT_STACK_INDEX + 2).map(|_| Stack::new()).collect();
            let mut stack_b: Vec<Stack> = (0..CURRENT_STACK_INDEX + 2).map(|_| Stack::new()).collect();

            evaluate_at_root(&pos, &mut stack_a);
            evaluate_at_root(&pos, &mut stack_b);

            let v1 = evaluate(&mut pos, &mut stack_a);
            let v2 = evaluate(&mut pos, &mut stack_b);
            assert_eq!(v1, v2, "nnue::evaluate must be deterministic for identical (pos, net)");

            clear_loaded_for_test();
        });
    }

    // With `numa` on, eval must read the TLS replica not global `NETWORK`: the global is cleared, so a completed eval proves the TLS replica was used.
    #[cfg(feature = "numa")]
    #[test]
    fn evaluate_reads_thread_local_replica_not_global() {
        use crate::search::CURRENT_STACK_INDEX;
        test_fixtures::run_with_large_stack(|| {
            let _guard = TEST_MUTEX.lock().expect("TEST_MUTEX poisoned");
            clear_loaded_for_test(); // global is empty on purpose
            set_local_replica_for_test(test_fixtures::synthetic_net());
            assert!(local_replica().is_some(), "TLS replica pointer must be set");

            let mut pos = Position::new();
            let mut stack: Vec<Stack> = (0..CURRENT_STACK_INDEX + 2).map(|_| Stack::new()).collect();

            let root = evaluate_at_root(&pos, &mut stack);
            let leaf = evaluate(&mut pos, &mut stack);
            // Same (pos, net) ⇒ root refresh and leaf forward agree, and repeat is stable.
            assert_eq!(root, leaf, "root and leaf eval must agree for the replica network");
            assert_eq!(leaf, evaluate(&mut pos, &mut stack), "replica eval must be deterministic");

            clear_local_replica_for_test();
            clear_loaded_for_test();
        });
    }

    // Incremental-path companion: global cleared, only the TLS replica set, so driving moves through `local_replica()` proves the incremental accumulator runs off the node-local replica and matches a fresh refresh.
    #[cfg(feature = "numa")]
    #[test]
    fn incremental_path_reads_thread_local_replica_not_global() {
        use crate::movetypes::Move;
        use crate::search::do_move_with_accumulator;
        test_fixtures::run_with_large_stack(|| {
            let _guard = TEST_MUTEX.lock().expect("TEST_MUTEX poisoned");
            clear_loaded_for_test(); // global is empty on purpose
            set_local_replica_for_test(test_fixtures::synthetic_net());

            // The incremental path takes its net from the node-local replica, as `iterative_deepening_loop` resolves `nnue_net_ptr`.
            let net = local_replica().expect("TLS replica pointer must be set");

            let mut pos = Position::new();
            let mut stack: Vec<Stack> = (0..10).map(|_| Stack::new()).collect();
            transformer::refresh(&mut stack[0].accumulator, net, &pos);

            let usi_moves = ["7g7f", "3c3d", "2g2f", "8c8d"];
            for (ply, usi) in usi_moves.iter().enumerate() {
                let mv = Move::new_from_usi_str(usi, &pos).expect("legal move");
                let gives_check = pos.gives_check(mv);
                do_move_with_accumulator(&mut stack, ply, &mut pos, mv, gives_check, net);
                assert!(
                    stack[ply + 1].accumulator.computed,
                    "incremental update must mark the slot computed"
                );
            }

            // Same replica net ⇒ the incrementally-maintained accumulator equals a fresh refresh.
            let mut expected = Accumulator::zeroed();
            transformer::refresh(&mut expected, net, &pos);
            assert_eq!(
                stack[usi_moves.len()].accumulator.us,
                expected.us,
                "incremental `.us` must match fresh refresh"
            );
            assert_eq!(
                stack[usi_moves.len()].accumulator.them,
                expected.them,
                "incremental `.them` must match fresh refresh"
            );

            clear_local_replica_for_test();
            clear_loaded_for_test();
        });
    }

    #[test]
    fn evaluate_at_root_refreshes_uncomputed_slot() {
        use crate::search::CURRENT_STACK_INDEX;
        test_fixtures::run_with_large_stack(|| {
            let _guard = TEST_MUTEX.lock().expect("TEST_MUTEX poisoned");
            clear_loaded_for_test();
            set_loaded_for_test(test_fixtures::synthetic_net_arc(), PathBuf::from("synthetic"));

            let pos = Position::new();
            let mut stack: Vec<Stack> = (0..CURRENT_STACK_INDEX + 2).map(|_| Stack::new()).collect();

            assert!(
                !stack[CURRENT_STACK_INDEX].accumulator.computed,
                "fresh stack slot must start uncomputed"
            );

            let v1 = evaluate_at_root(&pos, &mut stack);

            assert!(
                stack[CURRENT_STACK_INDEX].accumulator.computed,
                "evaluate_at_root must refresh the slot"
            );

            let v2 = evaluate_at_root(&pos, &mut stack);
            assert_eq!(v1, v2, "evaluate_at_root must be deterministic for the same (pos, net)");

            clear_loaded_for_test();
        });
    }
}
