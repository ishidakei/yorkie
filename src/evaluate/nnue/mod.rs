//! NNUE evaluation function (SFNNwoP1536).
//!
//! Derived from YaneuraOu (https://github.com/yaneurao/YaneuraOu).
#![allow(dead_code)]

pub mod bucket;
pub mod features;
pub mod loader;
pub mod network;
pub mod simd;
pub mod transformer;
pub mod types;

#[cfg(test)]
pub(crate) mod test_fixtures;

#[allow(unused_imports)]
pub use types::{Accumulator, FEATURE_LIST_CAPACITY, FeatureIndex, FeatureList, HIDDEN_SIZE, NetHeader, NnueError, NnueNetwork};

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use crate::position::*;
use crate::search::{Stack, get_stack_mut};
use crate::types::*;

struct LoadedNetwork {
    net: Arc<NnueNetwork>,
    path: PathBuf,
}

static NETWORK: RwLock<Option<LoadedNetwork>> = RwLock::new(None);

pub fn evaluate(pos: &mut Position, stack: &mut [Stack]) -> Value {
    let net_arc = match current_network() {
        Some(n) => n,
        None => {
            debug_assert!(false, "nnue::evaluate called without a loaded network");
            return Value::ZERO;
        }
    };
    let net: &NnueNetwork = &net_arc;

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
    let net_arc = match current_network() {
        Some(n) => n,
        None => {
            debug_assert!(false, "nnue::evaluate_at_root called without a loaded network");
            return Value::ZERO;
        }
    };
    let net: &NnueNetwork = &net_arc;

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

#[cfg(test)]
pub(crate) fn make_placeholder_network(sha256: [u8; 32]) -> NnueNetwork {
    NnueNetwork {
        header: NetHeader {
            version: 0,
            hash: 0,
            arch_id: "placeholder".to_string(),
        },
        ft_biases: Vec::<i16>::new().into_boxed_slice(),
        ft_weights: Vec::<i16>::new().into_boxed_slice(),
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
