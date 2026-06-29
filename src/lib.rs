#![allow(
        clippy::cognitive_complexity,
        clippy::too_many_arguments,
        clippy::new_without_default,
        clippy::branches_sharing_code, // This feature suggests incorrect code.
    )]
#[macro_use]
extern crate custom_derive;
#[macro_use]
extern crate derive_more;
#[macro_use]
extern crate enum_derive;
#[macro_use]
extern crate static_assertions;
// Guard: the tournament build must never link the `tsumeshogi-solver` mate dependency (owned by `mate`).
#[cfg(all(feature = "tournament", feature = "mate"))]
compile_error!(
    "features `tournament` and `mate` are mutually exclusive — the tournament build must not link the mate-search solver"
);

mod authors;
mod bitboard;
mod book;
mod engine_name;
mod evaluate;
mod file_to_vec;
mod hand;
mod huffman_code;
mod learn;
mod movegen;
mod movepick;
mod movetypes;
/// NUMA topology + CPU-affinity support (Linux-only, `numa` feature); off = no module, pins nothing.
#[cfg(feature = "numa")]
mod numa;
mod piecevalue;
mod position;
mod search;
mod sfen;
pub mod stack_size;
mod thread;
mod timeman;
/// Compile-time tournament consts from `build.rs` (config TOML); present only under the `tournament` feature.
#[cfg(feature = "tournament")]
mod tournament {
    include!(concat!(env!("OUT_DIR"), "/tournament_consts.rs"));
}
mod tt;
mod types;
pub mod usi;
mod usioption;
