//! The compile-time configuration: that it really is compile-time.
//!
//! Every generated value is used in a `const` context, none of which compiles
//! against a value read at run time — so this file fails to *build*, rather than
//! fails to pass, if the mechanism ever regresses into a lookup.
//!
//! Every assertion is written against the constants themselves, so it holds for
//! a binary built from any config file, not just the checked-in play one.

use yorkie_protocol::config;

// An array length is the least forgiving const context there is.
const MULTI_PV_SLOTS: [u8; config::MULTI_PV as usize] = [0; config::MULTI_PV as usize];
const BOOK_PV_SLOTS: [u16; config::BOOK_PV_MOVES as usize] = [0; config::BOOK_PV_MOVES as usize];

/// The transposition table's byte size, folded at compile time out of the
/// configured MiB — the shape the engine's own sizing arithmetic has.
const HASH_BYTES: u64 = (config::USI_HASH as u64) * 1024 * 1024;

/// A `const fn` fed the configured values: if any of them were a runtime read,
/// this initializer would not be accepted.
const fn clamp_positive(v: i64) -> i64 {
    if v < 1 { 1 } else { v }
}
const POOL_SIZE: usize = clamp_positive(config::THREADS) as usize;

/// Constant booleans drive constant branches — the branch every build wants
/// folded away rather than evaluated once per node.
const PV_ON_FAIL: &str = if config::OUTPUT_FAIL_LH_PV {
    "emit"
} else {
    "suppress"
};

/// A `&'static str` constant used where only a constant is accepted.
const EVAL_SUBPATH: &str = config::EVAL_DIR;

// Const-context assertions: evaluated by the compiler, so a run-time value
// could not be substituted for any of these operands.
const _: () = assert!(HASH_BYTES >= 1024 * 1024);
const _: () = assert!(POOL_SIZE >= 1);
const _: () = assert!(MULTI_PV_SLOTS.len() == config::MULTI_PV as usize);
const _: () = assert!(!EVAL_SUBPATH.is_empty());
const _: () = assert!(config::NODES_LIMIT >= 0);

#[test]
fn generated_values_are_usable_as_compile_time_constants() {
    // The `const` items above are the actual proof — they were evaluated by the
    // compiler before this test existed at run time. Reading them back here
    // keeps them from being dead code and pins the arithmetic.
    assert_eq!(MULTI_PV_SLOTS.len(), config::MULTI_PV as usize);
    assert_eq!(BOOK_PV_SLOTS.len(), config::BOOK_PV_MOVES as usize);
    assert_eq!(HASH_BYTES, config::USI_HASH as u64 * 1024 * 1024);
    assert_eq!(POOL_SIZE, config::THREADS.max(1) as usize);
    assert_eq!(
        PV_ON_FAIL,
        if config::OUTPUT_FAIL_LH_PV {
            "emit"
        } else {
            "suppress"
        }
    );
    assert_eq!(EVAL_SUBPATH, config::EVAL_DIR);

    // A local `const` context, so the proof is not only at module level.
    const RESIGN_IS_REACHABLE: bool = config::RESIGN_VALUE < 99_999;
    const DRAW_SPREAD: i64 = config::DRAW_VALUE_BLACK - config::DRAW_VALUE_WHITE;
    assert_eq!(RESIGN_IS_REACHABLE, config::RESIGN_VALUE < 99_999);
    assert_eq!(
        DRAW_SPREAD,
        config::DRAW_VALUE_BLACK - config::DRAW_VALUE_WHITE
    );
}

#[test]
fn the_binary_records_which_config_it_was_built_with() {
    assert!(
        !config::CONFIG_NAME.is_empty(),
        "every build records its config instance"
    );
}
