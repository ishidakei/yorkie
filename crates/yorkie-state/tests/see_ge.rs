//! Hand-built exchange scenarios for `Position::see_ge`, each with its expected
//! value worked out below, plus a seeded-playout sweep over the perft fixtures.

use yorkie_state::board::Board;
use yorkie_state::color::Color;
use yorkie_state::move_::{Move, parse_usi_move};
use yorkie_state::piece::{Piece, PieceKind};
use yorkie_state::position::Position;
use yorkie_state::sfen::parse_sfen;
use yorkie_state::square::Square;

/// Apery material values, so the expected SEE values read in named terms.
const PAWN: i32 = 90;
const LANCE: i32 = 315;
const SILVER: i32 = 495;
const GOLD: i32 = 540;
const ROOK: i32 = 990;
const DRAGON: i32 = 1395;

fn sq(file: u8, rank: u8) -> Square {
    Square::new(file, rank).unwrap()
}

fn set(pos: &mut Position, file: u8, rank: u8, kind: PieceKind, color: Color) {
    pos.board_mut()
        .set(sq(file, rank), Some(Piece::new(kind, color)));
}

fn set_promoted(pos: &mut Position, file: u8, rank: u8, kind: PieceKind, color: Color) {
    pos.board_mut()
        .set(sq(file, rank), Some(Piece::promoted(kind, color).unwrap()));
}

/// Only the two kings, in opposite corners, clear of the file-4 and rank-4
/// exchange geometry the scenarios below use.
fn two_king_board(mover: Color) -> Position {
    let mut pos = Position::empty();
    set(&mut pos, 0, 8, PieceKind::King, Color::Black);
    set(&mut pos, 0, 0, PieceKind::King, Color::White);
    pos.set_side_to_move(mover);
    pos
}

/// Assert the SEE value of `m` is exactly `value`, by probing the threshold on
/// both sides of the boundary — the reference's own `see_ge_th` check.
fn assert_see_value(pos: &Position, m: Move, value: i32) {
    assert!(
        pos.see_ge(m, value),
        "see_ge(m, {value}) should be true (threshold == SEE value)",
    );
    assert!(
        pos.see_ge(m, value - 1),
        "see_ge(m, {}) should be true (threshold below SEE value)",
        value - 1,
    );
    assert!(
        !pos.see_ge(m, value + 1),
        "see_ge(m, {}) should be false (threshold above SEE value)",
        value + 1,
    );
}

#[test]
fn undefended_capture_wins_the_victim() {
    // A Black rook captures an undefended White pawn: SEE = +Pawn.
    let mut pos = two_king_board(Color::Black);
    set(&mut pos, 4, 8, PieceKind::Rook, Color::Black);
    set(&mut pos, 4, 4, PieceKind::Pawn, Color::White);
    let m = Move::make(
        sq(4, 8),
        sq(4, 4),
        Piece::new(PieceKind::Rook, Color::Black),
    );
    assert_see_value(&pos, m, PAWN);
}

#[test]
fn defended_pawn_capture_by_rook_is_a_loss() {
    // A Black rook captures a gold-defended White pawn, and is recaptured:
    // SEE = Pawn - Rook.
    let mut pos = two_king_board(Color::Black);
    set(&mut pos, 4, 8, PieceKind::Rook, Color::Black);
    set(&mut pos, 4, 4, PieceKind::Pawn, Color::White);
    set(&mut pos, 3, 3, PieceKind::Gold, Color::White);
    let m = Move::make(
        sq(4, 8),
        sq(4, 4),
        Piece::new(PieceKind::Rook, Color::Black),
    );
    assert_see_value(&pos, m, PAWN - ROOK);
    assert!(!pos.see_ge(m, 0));
}

#[test]
fn three_deep_recapture_chain_nets_a_pawn() {
    // A Black pawn takes a defended White pawn with a Black lance x-raying
    // behind it: +Pawn, -Pawn, +Pawn leaves the lance safe at SEE = +Pawn.
    let mut pos = two_king_board(Color::Black);
    set(&mut pos, 4, 4, PieceKind::Pawn, Color::White);
    set(&mut pos, 4, 5, PieceKind::Pawn, Color::Black);
    set(&mut pos, 4, 6, PieceKind::Lance, Color::Black);
    set(&mut pos, 4, 3, PieceKind::Pawn, Color::White);
    let m = Move::make(
        sq(4, 5),
        sq(4, 4),
        Piece::new(PieceKind::Pawn, Color::Black),
    );
    assert_see_value(&pos, m, PAWN);
}

#[test]
fn xray_rook_behind_rook_flips_loss_to_win() {
    // Doubled Black rooks over a rook-defended White pawn. Without the rear
    // rook SEE = Pawn - Rook; with it the recapture is itself recaptured and
    // SEE = +Pawn, so the rear rook flips the sign.
    let base = |with_rear: bool| {
        let mut pos = two_king_board(Color::Black);
        set(&mut pos, 4, 4, PieceKind::Pawn, Color::White);
        set(&mut pos, 4, 5, PieceKind::Rook, Color::Black);
        set(&mut pos, 2, 4, PieceKind::Rook, Color::White);
        if with_rear {
            set(&mut pos, 4, 6, PieceKind::Rook, Color::Black);
        }
        pos
    };
    let m = Move::make(
        sq(4, 5),
        sq(4, 4),
        Piece::new(PieceKind::Rook, Color::Black),
    );

    let without = base(false);
    assert_see_value(&without, m, PAWN - ROOK);
    assert!(!without.see_ge(m, 0), "without the x-ray rook it is a loss");

    let with = base(true);
    assert_see_value(&with, m, PAWN - ROOK + ROOK);
    assert!(with.see_ge(m, 0), "the x-ray rook flips it to a win");
}

#[test]
fn xray_lance_behind_lance_changes_the_verdict() {
    // Doubled Black lances over a rook-defended White pawn. Without the rear
    // lance SEE = Pawn - Lance; with it White declines to recapture, since the
    // rear lance would take the rook, and SEE = +Pawn.
    let base = |with_rear: bool| {
        let mut pos = two_king_board(Color::Black);
        set(&mut pos, 4, 4, PieceKind::Pawn, Color::White);
        set(&mut pos, 4, 5, PieceKind::Lance, Color::Black);
        set(&mut pos, 2, 4, PieceKind::Rook, Color::White);
        if with_rear {
            set(&mut pos, 4, 6, PieceKind::Lance, Color::Black);
        }
        pos
    };
    let m = Move::make(
        sq(4, 5),
        sq(4, 4),
        Piece::new(PieceKind::Lance, Color::Black),
    );

    let without = base(false);
    assert_see_value(&without, m, PAWN - LANCE);
    assert!(
        !without.see_ge(m, 0),
        "without the x-ray lance it is a loss"
    );

    let with = base(true);
    assert_see_value(&with, m, PAWN);
    assert!(with.see_ge(m, 0), "the x-ray lance flips it to a win");
}

#[test]
fn promoted_victim_contributes_its_promoted_value() {
    // An undefended White dragon is worth `DRAGON`, not `ROOK`.
    let mut pos = two_king_board(Color::Black);
    set(&mut pos, 4, 8, PieceKind::Rook, Color::Black);
    set_promoted(&mut pos, 4, 4, PieceKind::Rook, Color::White);
    let m = Move::make(
        sq(4, 8),
        sq(4, 4),
        Piece::new(PieceKind::Rook, Color::Black),
    );
    assert_see_value(&pos, m, DRAGON);
}

#[test]
fn move_promotion_is_not_credited() {
    // A Black pawn captures a gold-defended White pawn inside the promotion
    // zone, where promoting is optional. The recapture is valued at the pawn's
    // own value either way, so SEE = 0 for both variants. Were the promotion
    // credited, the recaptured piece would be a tokin and the values differ.
    let mut pos = two_king_board(Color::Black);
    set(&mut pos, 4, 3, PieceKind::Pawn, Color::Black);
    set(&mut pos, 4, 2, PieceKind::Pawn, Color::White);
    set(&mut pos, 3, 1, PieceKind::Gold, Color::White);
    let pawn = Piece::new(PieceKind::Pawn, Color::Black);
    let quiet = Move::make(sq(4, 3), sq(4, 2), pawn);
    let promote = Move::make_promote(sq(4, 3), sq(4, 2), pawn);

    assert_see_value(&pos, quiet, 0);
    for th in [-GOLD, -PAWN, -1, 0, 1, PAWN, GOLD] {
        assert_eq!(
            pos.see_ge(quiet, th),
            pos.see_ge(promote, th),
            "promotion flag must not change see_ge at threshold {th}",
        );
    }
}

#[test]
fn pinned_defender_cannot_recapture() {
    // A Black rook captures a White pawn whose only defender, a gold, is
    // pinned to its king by a Black lance. Pinned it cannot recapture, so
    // SEE = +Pawn; remove the pin and SEE = Pawn - Rook.
    let base = |with_pin: bool| {
        let mut pos = two_king_board(Color::Black);
        // The White king moves onto the 4-file so the lance can pin.
        pos.board_mut().set(sq(0, 0), None);
        set(&mut pos, 5, 0, PieceKind::King, Color::White);
        set(&mut pos, 4, 8, PieceKind::Rook, Color::Black);
        set(&mut pos, 4, 4, PieceKind::Pawn, Color::White);
        set(&mut pos, 5, 3, PieceKind::Gold, Color::White);
        if with_pin {
            set(&mut pos, 5, 8, PieceKind::Lance, Color::Black);
        }
        pos
    };
    let m = Move::make(
        sq(4, 8),
        sq(4, 4),
        Piece::new(PieceKind::Rook, Color::Black),
    );

    let pinned = base(true);
    assert_see_value(&pinned, m, PAWN);
    assert!(
        pinned.see_ge(m, 0),
        "pinned defender makes the capture safe"
    );

    let free = base(false);
    assert_see_value(&free, m, PAWN - ROOK);
    assert!(!free.see_ge(m, 0), "an unpinned gold recaptures — a loss");
}

#[test]
fn threshold_semantics_sweep_around_a_known_exchange() {
    // The defended-rook-capture loss, over a spread of thresholds either side
    // of the exact boundary.
    let mut pos = two_king_board(Color::Black);
    set(&mut pos, 4, 8, PieceKind::Rook, Color::Black);
    set(&mut pos, 4, 4, PieceKind::Pawn, Color::White);
    set(&mut pos, 3, 3, PieceKind::Gold, Color::White);
    let m = Move::make(
        sq(4, 8),
        sq(4, 4),
        Piece::new(PieceKind::Rook, Color::Black),
    );
    let see = PAWN - ROOK; // -900
    for th in [-2000, -1000, see - 1, see, see + 1, 0, 90, 500, 2000] {
        assert_eq!(
            pos.see_ge(m, th),
            th <= see,
            "see_ge(m, {th}) should equal ({th} <= {see})",
        );
    }
}

#[test]
fn reference_pos1move_bishop_capture_promote_is_see_zero() {
    // The reference's own "pos1move" case: a bishop-takes-bishop promotion
    // recaptured by the silver, so the material is even and SEE = 0.
    let mut pos = Position::startpos();
    for usi in ["7g7f", "3c3d"] {
        let m = parse_usi_move(usi, &pos).expect("startpos move parses");
        pos.do_move(m);
    }
    let m = parse_usi_move("8h2b+", &pos).expect("bishop capture-promote parses");
    assert_see_value(&pos, m, 0);
}

/// The reference `see_ge` test's position P2: a Black horse on 2b that White
/// has declined to recapture, Black to move.
fn reference_pos2() -> Position {
    let mut pos = Position::startpos();
    for usi in ["7g7f", "3c3d", "8h2b+", "8c8d"] {
        let m = parse_usi_move(usi, &pos).expect("reference-sequence move parses");
        pos.do_move(m);
    }
    pos
}

#[test]
fn reference_pos2move_horse_to_31_is_horse_for_silver() {
    // The reference's "pos2move": the horse into 3a is met by the gold,
    // trading it for the silver it captured. SEE = -Horse + Silver.
    let pos = reference_pos2();
    let m = parse_usi_move("2b3a", &pos).expect("horse move parses");
    assert_see_value(&pos, m, -945 + SILVER);
}

#[test]
fn reference_pos2drop_bishop_drop_is_bishop_for_knight() {
    // The reference's "pos2drop": B*3c is answered by the knight and then
    // recaptured by the horse. SEE = -Bishop + Knight.
    let pos = reference_pos2();
    let m = parse_usi_move("B*3c", &pos).expect("bishop drop parses");
    assert_see_value(&pos, m, -855 + 405);
}

#[test]
fn reference_pos2move_horse_to_33_is_a_free_loss() {
    // The reference's second "pos2move": 2b3c hangs the horse to the knight
    // for nothing. SEE = -Horse.
    let pos = reference_pos2();
    let m = parse_usi_move("2b3c", &pos).expect("horse move parses");
    assert_see_value(&pos, m, -945);
}

#[test]
fn silver_recapture_is_the_least_valuable_attacker() {
    // A Black rook captures a White silver defended by both a silver and a
    // gold, so the cheaper silver recaptures: SEE = Silver - Rook.
    let mut pos = two_king_board(Color::Black);
    set(&mut pos, 4, 8, PieceKind::Rook, Color::Black);
    set(&mut pos, 4, 4, PieceKind::Silver, Color::White);
    set(&mut pos, 3, 3, PieceKind::Silver, Color::White);
    set(&mut pos, 5, 3, PieceKind::Gold, Color::White);
    let m = Move::make(
        sq(4, 8),
        sq(4, 4),
        Piece::new(PieceKind::Rook, Color::Black),
    );
    assert_see_value(&pos, m, SILVER - ROOK);
}

/// The perft-fixture SFENs, one deterministic playout each.
const FIXTURE_SFENS: &[&str] = &[
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1", // startpos
    "4k4/9/4r4/9/9/9/4K3B/9/9 b RG2gs2n3p 1",                          // check-evasion
    "k8/1P7/G8/1N2P4/9/9/9/9/8K b 2PG2pg 1",                           // drop-heavy
    "l7l/1r1sg2k1/2nppgsp1/p1p3p1p/1p2N4/2P1P1P2/PPSP1PB1P/3GG1SR1/LN2K3L b BNPp 1", // mid-game-tactical
    "4k4/3P3+PL/2N2PR2/1L2BNS2/4N4/9/9/9/4K4 b - 1", // promotion-zone-edges
    "9/4k4/9/9/9/9/9/4K4/9 b 9P9p 1",                // sennichite
];

/// Deterministic xorshift64*, so a failing playout replays.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn pick(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// A capture is a non-drop move landing on an occupied square.
fn is_capture(board: &Board, m: Move) -> bool {
    !m.is_drop() && board.get(m.to_sq()).is_some()
}

/// Thresholds swept per capture, ascending.
const THRESHOLDS: [i32; 9] = [-3000, -900, -90, -1, 0, 1, 90, 900, 3000];

#[cfg_attr(miri, ignore)]
#[test]
fn see_ge_is_deterministic_and_panic_free_on_fixture_playouts() {
    const MIN_PLIES: usize = 40;

    for (fi, sfen) in FIXTURE_SFENS.iter().enumerate() {
        let mut pos = parse_sfen(sfen).expect("fixture sfen parses");
        let mut rng = Rng(0x1234_5678_9ABC_DEF0 ^ (fi as u64).wrapping_add(1));
        let mut legal: Vec<Move> = Vec::new();

        let mut plies = 0usize;
        while plies < MIN_PLIES {
            legal.clear();
            pos.generate_legal_all(&mut legal);
            if legal.is_empty() {
                break;
            }

            for &m in &legal {
                if !is_capture(pos.board(), m) {
                    continue;
                }
                // `see_ge` is monotone non-increasing in the threshold, so a
                // `true` here implies `true` at every lower one visited.
                let mut prev: Option<bool> = None;
                for &th in &THRESHOLDS {
                    let a = pos.see_ge(m, th);
                    let b = pos.see_ge(m, th);
                    assert_eq!(a, b, "see_ge not deterministic (fixture {fi}, th {th})");
                    if let Some(pv) = prev {
                        assert!(
                            !a || pv,
                            "see_ge true at higher threshold but false at a lower one (fixture {fi})",
                        );
                    }
                    prev = Some(a);
                }
            }

            let m = legal[rng.pick(legal.len())];
            pos.do_move(m);
            plies += 1;
        }
    }
}
