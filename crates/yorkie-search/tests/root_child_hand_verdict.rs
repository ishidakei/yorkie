//! A root child that repeats a board from the game before the root, with other
//! hands, takes the superior / inferior verdict against that earlier position.
//!
//! The two inputs reach the same position. Input A plays `4e4f 6h4f P*4e` from
//! its start, so the root move `4f6h` puts the bishop back where it stood at
//! the start: the child has the start's board with White to move, and Black
//! holds the pawn White dropped. The repetition judgement walks back past the
//! root, finds the start four plies earlier and returns the inferior verdict
//! for White, so the root takes `4f6h` with the maximal non-mate value. Input B
//! starts one ply later, after `4e4f`; its start has the pawn on 4f, not 4e,
//! so no earlier board matches and the same move gets an ordinary value.
//!
//! The reference engine scores the two inputs the same way. These tests pin
//! that behaviour: a change to how far before the root the hand verdict may
//! reach changes input A's value and has to update them.
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
/// `VALUE_SUPERIOR`, the value of a superior verdict.
const VALUE_SUPERIOR: i32 = VALUE_TB_WIN_IN_MAX_PLY - 1;

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
fn input_a_takes_the_hand_verdict_and_input_b_does_not() {
    let path = nn_bin_path();
    if !path.exists() {
        eprintln!(
            "skipping input_a_takes_the_hand_verdict_and_input_b_does_not: {} is not present (obtained out-of-band)",
            path.display()
        );
        return;
    }
    let owned = yorkie_eval::load_network(&path).expect("real nn.bin should load and validate");
    let net = owned.network();

    TranspositionTable::shared().clear();
    let a = QSearch::new(net).run_root(&setup(SFEN_A, MOVES_A), DEPTH);
    assert_eq!(format_usi_move(a.best_move), "4f6h");
    assert_eq!(a.score, VALUE_SUPERIOR, "input A scores the inferior child");

    TranspositionTable::shared().clear();
    let b = QSearch::new(net).run_root(&setup(SFEN_B, MOVES_B), DEPTH);
    assert!(
        b.score.abs() < VALUE_TB_WIN_IN_MAX_PLY,
        "input B gets an ordinary value, got {}",
        b.score
    );
}
