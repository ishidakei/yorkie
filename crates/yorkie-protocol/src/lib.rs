/// The feature-gated `bench` command — present only under `verbose3`.
#[cfg(feature = "verbose3")]
pub mod bench;
pub(crate) mod bestmove;
pub mod config;
pub mod engine;
pub mod formatter;
pub mod parser;
pub(crate) mod settings;
/// The per-reply statistics line, which only a `verbose1` build writes.
#[cfg(feature = "verbose1")]
pub(crate) mod stats;
/// The feature-gated `tt` command family — present only under `verbose3`.
#[cfg(feature = "verbose3")]
pub mod tt_command;
pub mod usi;

#[cfg(feature = "verbose3")]
pub use bench::{BENCH_DEFAULT_POSITIONS, BenchConfig, BenchParseError, parse_bench};
pub use engine::{Engine, EngineSink, GoOutcome, GoParams, PositionSfen, ReadyOutcome, Reply};
pub use formatter::Formatter;
pub use parser::{Command, parse_line};
#[cfg(feature = "verbose3")]
pub use tt_command::{TtCommand, TtParseError, TtPosition, TtStoreArgs, parse_tt};
pub use usi::UsiEngine;

/// Serialises the tests that drive a session.
///
/// The transposition table is one `static` per process, and every session that
/// reaches `usinewgame` empties the whole of it. Two such tests running as
/// threads of one binary would clear the table under each other. Under
/// `cargo nextest` each test is its own process and the lock is never contended;
/// under a plain `cargo test` it is what keeps them apart.
#[cfg(test)]
static TT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Exclusive use of the process's transposition table, held until the returned
/// guard goes out of scope. A panicking test leaves the lock poisoned; the next
/// test wants the table, not the panic, and the session it drives empties the
/// table before searching anything.
#[cfg(test)]
pub(crate) fn serial_tt() -> std::sync::MutexGuard<'static, ()> {
    TT_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// `n` legal moves from the initial position: both kings stepping onto the
/// square in front of them and back, for as long as asked.
#[cfg(test)]
pub(crate) fn king_shuffle(n: usize) -> String {
    const CYCLE: [&str; 4] = ["5i5h", "5a5b", "5h5i", "5b5a"];
    let mut line = String::new();
    for i in 0..n {
        if i > 0 {
            line.push(' ');
        }
        line.push_str(CYCLE[i % CYCLE.len()]);
    }
    line
}
