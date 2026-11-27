//! A root child that repeats a board from the game before the root, with the
//! side to move holding less in hand, is searched like any other move.
//!
//! The two inputs reach the same position. Input A plays `4e4f 6h4f P*4e` from
//! its start, so the root move `4f6h` puts the bishop back where it stood at
//! the start: the child has the start's board with White to move, and Black
//! holds the pawn White dropped. The position-level judgement walks back past
//! the root, finds the start four plies earlier and returns the inferior
//! verdict for White. Taken at face value, that verdict would give `4f6h` the
//! maximal non-mate value and the root would play it without searching. The
//! search leaves the inferior verdict out at the root's children, so input A
//! gets the ordinary value input B gets. Input B starts one ply later, after
//! `4e4f`; its start has the pawn on 4f, not 4e, so no earlier board matches.
//!
//! The reference engine takes the inferior verdict at the root's children too,
//! and plays `4f6h` for input A; this engine does not.
//!
//! Skipped with a notice when `nn.bin` is absent.

use std::path::PathBuf;

use yorkie_search::QSearch;
mod text_str;

use text_str::{format_sfen, format_usi_move, parse_sfen, parse_usi_move};
use yorkie_state::{Position, RepetitionState};
use yorkie_storage::TranspositionTable;

/// `VALUE_MATE`.
const VALUE_MATE: i32 = 32000;
/// `VALUE_TB_WIN_IN_MAX_PLY`: the `is_decisive` threshold.
const VALUE_TB_WIN_IN_MAX_PLY: i32 = VALUE_MATE - 246;

const SFEN_A: &str = "l5snl/3k5/1pnsg1b2/p1ppp1g1p/5pp2/PPP1P4/2SP5/K1GB5/LNG4+r1 w 2Prsnl3p 78";
const MOVES_A: &str = "4e4f 6h4f P*4e";
const SFEN_B: &str = "l5snl/3k5/1pnsg1b2/p1ppp1g1p/6p2/PPP1Pp3/2SP5/K1GB5/LNG4+r1 b 2Prsnl3p 79";
const MOVES_B: &str = "6h4f P*4e";

const DEPTH: i32 = 4;

fn nn_bin_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../eval/nn.bin")
}

fn play(pos: &mut Position, moves: &str) {
    for usi in moves.split(' ') {
        let m = parse_usi_move(usi, pos).unwrap_or_else(|e| panic!("bad move {usi}: {e:?}"));
        pos.do_move(m);
    }
}

fn setup(sfen: &str, moves: &str) -> Position {
    let mut pos = parse_sfen(sfen).expect("valid SFEN");
    play(&mut pos, moves);
    pos
}

/// The board and the side to move of an SFEN, without hands and ply.
fn board_and_side(sfen: &str) -> String {
    sfen.split(' ').take(2).collect::<Vec<_>>().join(" ")
}

#[test]
fn the_root_child_of_input_a_repeats_its_start_with_white_inferior() {
    let root = setup(SFEN_A, MOVES_A);
    let mut child = root.clone();
    play(&mut child, "4f6h");
    assert_eq!(
        board_and_side(&format_sfen(&child)),
        board_and_side(SFEN_A),
        "the child has the start's board and side to move"
    );
    assert_eq!(child.is_repetition(), RepetitionState::Inferior);
    assert_eq!(
        child.is_repetition_without_inferior(),
        RepetitionState::None,
        "nothing beyond the start matches"
    );

    let mut child_b = setup(SFEN_B, MOVES_B);
    play(&mut child_b, "4f6h");
    assert_eq!(
        board_and_side(&format_sfen(&child_b)),
        board_and_side(&format_sfen(&child)),
        "both inputs reach the same child board"
    );
    assert_eq!(child_b.is_repetition(), RepetitionState::None);
}

#[cfg_attr(miri, ignore)]
#[test]
fn both_inputs_search_the_root_move_and_get_an_ordinary_value() {
    let path = nn_bin_path();
    if !path.exists() {
        eprintln!(
            "skipping both_inputs_search_the_root_move_and_get_an_ordinary_value: {} is not present (obtained out-of-band)",
            path.display()
        );
        return;
    }
    let owned = yorkie_eval::load_network(&path).expect("real nn.bin should load and validate");
    let net = owned.network();

    TranspositionTable::shared().clear();
    let a = QSearch::new(net).run_root(&setup(SFEN_A, MOVES_A), DEPTH);
    TranspositionTable::shared().clear();
    let b = QSearch::new(net).run_root(&setup(SFEN_B, MOVES_B), DEPTH);
    for (name, out) in [("A", &a), ("B", &b)] {
        assert!(
            out.score.abs() < VALUE_TB_WIN_IN_MAX_PLY,
            "input {name} gets an ordinary value, got {}",
            out.score
        );
    }
    assert_eq!(a.score, b.score, "both inputs score the root alike");
    assert_eq!(
        format_usi_move(a.best_move),
        format_usi_move(b.best_move),
        "both inputs choose the same move"
    );
}
