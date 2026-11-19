//! The packed feature-transformer weights give the accumulators and the
//! evaluations of the 16-bit ones, bit for bit, at every position of a game.
//!
//! A build compiling the AVX-512 BW kernels stores the feature transformer's
//! weights a byte each, plus a correction for each weight that is not twice
//! its byte, and its accumulator kernels read that form. This plays a hundred
//! pseudo-random games and, at every position, holds the accumulator those
//! kernels derive — incrementally, through the king-move cache too — against
//! one rebuilt from scratch out of the source network's 16-bit weights, and
//! the evaluation against the 16-bit one.
//!
//! Where the build packed its own network and the file it wrote is beside the
//! test, the packed side is that network as the engine reads it; otherwise the
//! source network is packed in memory the way the build packs it, so the packed
//! kernels of whichever backend this build has are still checked.
//!
//! Compiled only where the source reader is (`source-network`), and skipped
//! with a notice when the network is not staged.

#![cfg(feature = "source-network")]

use std::path::PathBuf;

use yorkie_eval::network_file;
use yorkie_eval::{
    Accumulator, FinnyCache, FtColumns, MoveDelta, NetworkParams, PtrNetwork, evaluate,
    evaluate_with, load_network,
};
use yorkie_state::{Color, Move, Position};

mod common;
mod text_str;

use text_str::{format_usi_move, parse_sfen};

const GAMES: usize = 100;
const MAX_PLIES: usize = 256;
const STARTPOS: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

fn source_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../eval/nn.bin")
}

/// Deterministic xorshift64*, so a failing game replays.
struct Rng(u64);

impl Rng {
    fn pick(&mut self, n: usize) -> usize {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) % n as u64) as usize
    }
}

fn assert_same(packed: &Accumulator, wide: &Accumulator, ctx: &str) {
    for color in [Color::Black, Color::White] {
        assert_eq!(
            packed.perspective(color),
            wide.perspective(color),
            "{ctx}: {color:?} half differs from the 16-bit refresh",
        );
    }
}

/// Play the games, checking every position; returns the number of moves made.
fn play<P: NetworkParams>(packed: P, wide: PtrNetwork) -> usize {
    assert!(matches!(packed.ft_columns(), FtColumns::Packed(_)));
    assert!(matches!(wide.ft_columns(), FtColumns::Wide(_)));

    let mut cache = FinnyCache::new();
    let mut legal: Vec<Move> = Vec::new();
    let mut moves = 0usize;
    for game in 0..GAMES {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ (game as u64 + 1).wrapping_mul(0xA24B_AED4));
        let mut pos: Position = parse_sfen(STARTPOS).expect("startpos parses");

        let mut cached = Accumulator::new();
        cached.refresh(packed, &pos);
        let mut wide_root = Accumulator::new();
        wide_root.refresh(wide, &pos);
        assert_same(&cached, &wide_root, &format!("game {game} root"));
        let mut plain = Accumulator::new();
        plain.refresh(packed, &pos);

        for ply in 0..MAX_PLIES {
            legal.clear();
            pos.generate_legal_all(&mut legal);
            if legal.is_empty() {
                break;
            }
            let mv = legal[rng.pick(legal.len())];
            let ctx = format!("game {game} ply {ply} `{}`", format_usi_move(mv));

            let delta = MoveDelta::from_move(&pos, mv);
            let _undo = pos.do_move(mv);
            moves += 1;

            let mut next_cached = Accumulator::new();
            Accumulator::derive_into_cached(
                &cached,
                &mut next_cached,
                packed,
                &pos,
                &delta,
                &mut cache,
            );
            let mut next_plain = Accumulator::new();
            Accumulator::derive_into(&plain, &mut next_plain, packed, &pos, &delta);
            let mut fresh = Accumulator::new();
            fresh.refresh(wide, &pos);

            assert_same(&next_cached, &fresh, &format!("{ctx} (cached)"));
            assert_same(&next_plain, &fresh, &format!("{ctx} (derived)"));
            assert_eq!(
                evaluate_with(packed, &next_cached, &pos),
                evaluate(wide, &pos),
                "{ctx}: the evaluations differ",
            );
            cached = next_cached;
            plain = next_plain;
        }
    }
    moves
}

#[cfg_attr(miri, ignore)]
#[test]
fn packed_accumulators_match_16_bit_refreshes_over_a_hundred_games() {
    let source = source_path();
    if !source.is_file() {
        eprintln!(
            "skipping packed_accumulators_match_16_bit_refreshes_over_a_hundred_games: \
             the network is obtained out-of-band and is not staged here"
        );
        return;
    }
    let wide = load_network(&source).expect("the source network is readable");

    let engine = if network_file::FT_PACKED {
        common::engine_network()
    } else {
        None
    };
    let moves = match engine {
        Some(engine) => play(engine, wide.network()),
        None => {
            let mut packed = load_network(&source).expect("the source network is readable");
            let records = packed
                .pack_ft()
                .expect("the shipped network needs few enough corrections to be packed");
            eprintln!("packed in memory with {records} correction records");
            play(packed.network(), wide.network())
        }
    };
    eprintln!("{GAMES} games, {moves} moves, every position bit-identical");
}
