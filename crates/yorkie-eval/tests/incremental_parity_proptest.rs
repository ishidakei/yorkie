//! Property gate: the incrementally-updated accumulator equals a from-scratch
//! refresh at **every** step of a random legal line.
//!
//! This is the randomized sibling of `incremental_parity.rs`. That suite drives
//! two fixed corpora — the recorded fixture lines and six deterministic
//! xorshift playouts. This one hands the line choice to proptest: a root SFEN
//! index plus a vector of legal-move selectors, shrunk automatically to a
//! minimal failing line when an invariant breaks.
//!
//! At each ply the incremental accumulator (threaded through a do/undo stack) is
//! compared bit-for-bit — both perspectives, `i16` equality — against a fresh
//! [`Accumulator::refresh`], and [`evaluate_with`] over the incremental
//! accumulator is checked against the full-refresh [`evaluate`]. The line is
//! then unwound, re-checking the parent accumulator after every undo.
//!
//! **Kernel backend.** This suite drives only the crate's public, pure-Rust
//! `Accumulator` API — no SIMD intrinsic is called from here, and nothing is
//! compared against the reference C++ engine (that is `eval_parity.rs`'s job).
//! Which feature-transformer *kernels* that API dispatches to is fixed at
//! **compile** time from the CPU features the build enables (see
//! `yorkie_eval::Backend`), so a test binary cannot select one at run time: an
//! AVX-512 build exercises the AVX-512 kernels through this same property, and
//! any other build exercises the scalar ones. `report_active_backend` below
//! prints which. The scalar-versus-AVX-512 bit-equality of the kernels
//! themselves stays pinned by the in-crate backend tests.
//!
//! The network file is staged locally at
//! `eval/nn.bin` and is never committed. When it is
//! absent the property body is a no-op and the test passes, matching
//! `incremental_parity.rs` so the default `cargo test` run stays green
//! everywhere.
//!
//! `#[cfg_attr(miri, ignore)]` and default failure persistence match the
//! `yorkie-state` proptest suites.

use std::path::PathBuf;

use proptest::prelude::*;
use proptest::test_runner::TestCaseResult;
use yorkie_eval::{
    Accumulator, NnueNetwork, active_backend, evaluate, evaluate_with, load_network,
};
use yorkie_state::{Color, Move, Position, Undo, format_usi_move, parse_sfen};

/// Roots the random lines start from — the same six positions
/// `incremental_parity.rs` seeds its playouts with, covering the opening,
/// check-evasion, drop-heavy, mid-game-tactical, promotion-zone and
/// bare-kings-with-pawns shapes.
const ROOT_SFENS: &[&str] = &[
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1", // startpos
    "4k4/9/4r4/9/9/9/4K3B/9/9 b RG2gs2n3p 1",                          // check-evasion
    "k8/1P7/G8/1N2P4/9/9/9/9/8K b 2PG2pg 1",                           // drop-heavy
    "l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1", // mid-game-tactical
    "4k4/3P3+PL/2N2PR2/1L2BNS2/4N4/9/9/9/4K4 b - 1", // promotion-zone-edges
    "9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1",                // sennichite
];

/// Plies per generated line. Every ply costs two full refreshes and two full
/// evaluations, so this is kept short to hold the standard test gate fast.
const MAX_PLIES: usize = 12;

fn workspace_relative(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(rel)
}

fn nn_bin_path() -> PathBuf {
    workspace_relative("eval/nn.bin")
}

thread_local! {
    /// The 113 MB network, loaded at most once per test thread — proptest runs
    /// every case of a property on the same thread, so re-loading it per case
    /// would dominate the runtime. `None` means the file is not staged here.
    static NETWORK: Option<NnueNetwork> = {
        let path = nn_bin_path();
        if path.exists() {
            Some(load_network(&path).expect("real nn.bin should load and validate"))
        } else {
            None
        }
    };
}

/// A root index plus a line of legal-move selectors; each selector picks
/// `legal[i % legal.len()]` at the position it is applied to.
fn arb_line() -> impl Strategy<Value = (usize, Vec<u16>)> {
    (
        0..ROOT_SFENS.len(),
        proptest::collection::vec(any::<u16>(), 0..=MAX_PLIES),
    )
}

fn refreshed(net: &NnueNetwork, pos: &Position) -> Accumulator {
    let mut acc = Accumulator::new();
    acc.refresh(net, pos);
    acc
}

/// Both accumulator halves must be bit-identical to a from-scratch refresh of
/// `pos`, and `evaluate_with` must agree with the full-refresh `evaluate`.
fn check_matches_refresh(
    net: &NnueNetwork,
    acc: &Accumulator,
    pos: &Position,
    ctx: &str,
) -> TestCaseResult {
    let fresh = refreshed(net, pos);
    for color in [Color::Black, Color::White] {
        prop_assert_eq!(
            acc.perspective(color),
            fresh.perspective(color),
            "{}: {:?} half diverged from refresh",
            ctx,
            color,
        );
    }
    prop_assert_eq!(
        evaluate_with(net, acc, pos),
        evaluate(net, pos),
        "{}: evaluate paths disagree",
        ctx,
    );
    Ok(())
}

/// One do/undo frame: the move, its `Undo` token, and the incremental
/// accumulator for the resulting child position.
struct Frame {
    mv: Move,
    undo: Undo,
    acc: Accumulator,
}

fn legal_moves(pos: &Position) -> Vec<Move> {
    let mut buf = Vec::with_capacity(64);
    pos.generate_legal_all(&mut buf);
    buf
}

fn run_line(net: &NnueNetwork, root_index: usize, selectors: &[u16]) -> TestCaseResult {
    let sfen = ROOT_SFENS[root_index];
    let mut pos = parse_sfen(sfen).expect("root sfen parses");
    let root = refreshed(net, &pos);
    let mut frames: Vec<Frame> = Vec::new();

    for (ply, sel) in selectors.iter().enumerate() {
        let legal = legal_moves(&pos);
        if legal.is_empty() {
            // Terminal position: nothing left to play, so stop extending.
            break;
        }
        let mv = legal[*sel as usize % legal.len()];
        let ctx = format!(
            "root {root_index} ply {ply} `{}` [after do]",
            format_usi_move(mv)
        );

        let parent = frames.last().map_or(&root, |f| &f.acc);
        let acc = parent.update_after_move(net, &mut pos, mv);
        let undo = pos.do_move(mv);
        check_matches_refresh(net, &acc, &pos, &ctx)?;
        frames.push(Frame { mv, undo, acc });
    }

    // Unwind, re-checking the now-current parent accumulator after every undo.
    while let Some(frame) = frames.pop() {
        pos.undo_move(frame.mv, frame.undo);
        let parent = frames.last().map_or(&root, |f| &f.acc);
        let ctx = format!(
            "root {root_index} unwind `{}` [after undo]",
            format_usi_move(frame.mv)
        );
        check_matches_refresh(net, parent, &pos, &ctx)?;
    }

    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 32,
        ..ProptestConfig::default()
    })]

    #[cfg_attr(miri, ignore)]
    #[test]
    fn incremental_accumulator_matches_refresh_on_random_lines(
        (root_index, selectors) in arb_line()
    ) {
        NETWORK.with(|net| match net {
            // Not staged in this checkout: the property is vacuous, exactly as
            // `incremental_parity.rs` skips.
            None => Ok(()),
            Some(net) => run_line(net, root_index, &selectors),
        })?;
    }
}

/// Reports which kernel backend the property above actually exercised, so a run
/// that silently compiled the scalar path on an AVX-512 host is visible in the
/// log rather than invisible.
#[test]
fn report_active_backend() {
    eprintln!(
        "incremental_parity_proptest: accumulator kernels = {:?}",
        active_backend()
    );
}
