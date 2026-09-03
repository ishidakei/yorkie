//! Storage layer: a substrate that may depend on std and utility crates only,
//! never on the State, Evaluation, Search, or Protocol layers.
//!
//! It hosts the transposition table ([`tt`]), the `.ybb` opening-book reader
//! ([`book`]), and the huge-page-backed allocator ([`large_page`]) the other
//! layers use for their big allocations. It also declares the process-wide
//! `#[global_allocator]` ([`allocator`]) — a whole-program property rather than
//! a Storage concern, placed here for the reason those module docs give.

pub mod allocator;
pub mod arena;
pub mod book;
pub mod large_page;
pub mod tt;

pub use arena::{ARENA_SUB_ALIGN, ArenaLayout, ArenaSlice, LargePageArena, Section};
pub use book::{Book, BookError, BookMove};
pub use large_page::{LARGE_PAGE_ALIGN, LargePageArray, LargePageBox, Zeroable};
pub use tt::{
    Bound, DEFAULT_HASH_MB, DEPTH_NONE, Depth, TTData, TTWriter, TranspositionTable, TtSlot,
    VALUE_NONE, Value,
};
