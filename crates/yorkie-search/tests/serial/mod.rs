//! Exclusive use of the process's search state for one test.
//!
//! A search runs on state that is the process's, not the search's: the shared
//! transposition table, the table of moves a worker is busy with, and the
//! statics a test hands its workers (best-move tally slots, a stop flag). A test
//! binary run as one process runs its tests on parallel threads, and two of them
//! searching at once would clear, fill and stop each other's searches. Under
//! `cargo nextest` each test is its own process and the lock is never
//! contended; under a plain `cargo test` it is what keeps them apart.

use std::sync::{Mutex, MutexGuard};

static SEARCH_STATE: Mutex<()> = Mutex::new(());

/// Held until the returned guard goes out of scope; a test that searches takes
/// it as the first thing in its body. A panicking test leaves the lock
/// poisoned; the next test wants the state, not the panic, and clears the
/// table before it searches.
pub fn serial_search() -> MutexGuard<'static, ()> {
    SEARCH_STATE.lock().unwrap_or_else(|e| e.into_inner())
}
