/// The feature-gated `bench` command — present only under `verbose3`.
#[cfg(feature = "verbose3")]
pub mod bench;
pub mod config;
pub mod driver;
pub mod formatter;
pub mod parser;
pub(crate) mod settings;
/// The feature-gated `tt` command family — present only under `verbose3`.
#[cfg(feature = "verbose3")]
pub mod tt_command;

#[cfg(feature = "verbose3")]
pub use bench::{BENCH_DEFAULT_POSITIONS, BenchConfig, BenchParseError, parse_bench};
pub use driver::UsiDriver;
pub use formatter::Formatter;
pub use parser::{Command, parse_line};
#[cfg(feature = "verbose3")]
pub use tt_command::{TtCommand, TtParseError, TtPosition, TtStoreArgs, parse_tt};
