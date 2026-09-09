//! Storage layer: a substrate that may depend on std and utility crates only,
//! never on the State, Evaluation, Search, or Protocol layers.
//!
//! It hosts the transposition table ([`tt`]), the `.ybb` opening-book reader
//! ([`book`]), the huge-page-backed allocator ([`large_page`]) the other layers
//! use for their big allocations, and the huge-page-aligned read-only file
//! mapping ([`mapped`]) the evaluation network is read through. It also
//! declares the process-wide
//! `#[global_allocator]` ([`allocator`]) — a whole-program property rather than
//! a Storage concern, placed here for the reason those module docs give.
//!
//! The transposition table's size is a setting, and the table is a `static`,
//! so this crate compiles that setting in the same way the rest of the engine
//! does: `build.rs` reads the one TOML config and generates the `config`
//! module.

pub mod allocator;
pub mod arena;
pub mod book;
mod config;
pub mod large_page;
pub mod mapped;
pub mod tt;

#[cfg(feature = "verbose1")]
pub use allocator::{CountingAlloc, clear_alloc_count, take_alloc_count};
pub use arena::{ARENA_SUB_ALIGN, ArenaLayout, ArenaSlice, LargePageArena, Section};
pub use book::{Book, BookError, BookMove};
pub use large_page::{LARGE_PAGE_ALIGN, LargePageArray, LargePageBox, Zeroable, advise_huge_pages};
pub use mapped::MappedRegion;
pub use tt::{
    Bound, CLUSTER_COUNT, DEPTH_NONE, Depth, TABLE_BYTES, TT_ALIGN, TTData, TTWriter,
    TranspositionTable, TtSlot, VALUE_NONE, Value,
};
