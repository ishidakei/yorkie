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
pub mod usi;

#[cfg(feature = "verbose3")]
pub use bench::{BENCH_DEFAULT_POSITIONS, BenchConfig, BenchParseError, parse_bench};
pub use engine::{Engine, EngineSink, GoOutcome, GoParams, PositionSfen, ReadyOutcome, Reply};
pub use formatter::Formatter;
pub use parser::{Command, parse_line};
pub use usi::{Stdout, UsiEngine, UsiOutput};

/// Serialises the tests that build an engine or a worker pool.
///
/// A session's state is the process's: the transposition table, which every
/// session that reaches `usinewgame` empties whole, and the flags, counters and
/// helper slots a search runs on, which every `go` arms and every pool parks its
/// helpers on. Two such tests running as threads of one binary would clear and
/// overwrite them under each other. Under `cargo nextest` each test is its own
/// process and the lock is never contended; under a plain `cargo test` it is
/// what keeps them apart.
#[cfg(test)]
static SESSION_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Exclusive use of the process's session state, held until the returned guard
/// goes out of scope. A panicking test leaves the lock poisoned; the next test
/// wants the state, not the panic, and the session it drives arms everything it
/// uses before searching anything.
#[cfg(test)]
pub(crate) fn serial_session() -> std::sync::MutexGuard<'static, ()> {
    SESSION_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// A session's output kept in memory for a test to read back: the
/// buffer-backed [`UsiOutput`].
#[cfg(test)]
#[derive(Clone, Default)]
pub(crate) struct Buffer(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

#[cfg(test)]
impl Buffer {
    /// Everything written so far.
    pub(crate) fn bytes(&self) -> Vec<u8> {
        self.lock().clone()
    }
}

#[cfg(test)]
impl UsiOutput for Buffer {
    type Writer = Vec<u8>;
    type Guard<'a> = std::sync::MutexGuard<'a, Vec<u8>>;

    /// A writer that panicked mid-line leaves the lock poisoned; the test
    /// reading the transcript back still wants it.
    fn lock(&self) -> Self::Guard<'_> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }
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
