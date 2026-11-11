//! Games that open with a fixed preamble: four king moves, `5i5h 5a5b 5h5i
//! 5b5a`, which return the board to the initial position, so the first position
//! the engine is asked about is the initial board at ply 5. The preamble arrives
//! as ordinary moves of the `position` command.
//!
//! Every setting these tests rely on is named and checked in the test that
//! relies on it, rather than taken on trust from whichever config the build
//! read. The yorkie books are named per session through
//! [`yorkie_protocol::UsiEngine::with_yorkie_books`]; the network is the
//! synthetic all-zero one ([`common::stage_configured_eval_dir`]).
//!
//! The `.ybb` book a session reads is fixed when the engine is built, and the
//! test configs build none, so where the yorkie book and a `.ybb` book meet,
//! the probe the engine runs for a `go` is driven directly, with its settings
//! spelled out.

mod common;

use common::{
    TEST_BOOK_SEED, TempDir, YkEntry, bestmove_lines, drive_with_yorkie_book,
    drive_with_yorkie_books, legal, serial_session, stage_configured_eval_dir, startpos_after,
    write_ybb, yorkie_book_bytes,
};
use yorkie_protocol::config;
use yorkie_search::book::{BookConfig, Prng, YorkieProbe, probe_book};
use yorkie_state::{Move, PackedSfen, Position, sfen_pack};
use yorkie_storage::{Book, YorkieBook};

/// The four king moves a server plays before the game proper.
const PREAMBLE: [&str; 4] = ["5i5h", "5a5b", "5h5i", "5b5a"];

/// The preamble followed by `moves`.
fn after_preamble<'a>(moves: &[&'a str]) -> Vec<&'a str> {
    PREAMBLE
        .iter()
        .copied()
        .chain(moves.iter().copied())
        .collect()
}

/// The `position` command for the game `moves`.
fn position(moves: &[&str]) -> String {
    if moves.is_empty() {
        "position startpos\n".to_string()
    } else {
        format!("position startpos moves {}\n", moves.join(" "))
    }
}

const GO: &str = "go btime 10000 wtime 10000 byoyomi 0\n";

/// The move of each `bestmove` line, without any `ponder` that follows it.
fn played(out: &str) -> Vec<&str> {
    bestmove_lines(out)
        .into_iter()
        .map(|line| line.split(' ').next().unwrap_or(line))
        .collect()
}

fn write(dir: &TempDir, name: &str, bytes: &[u8]) -> std::path::PathBuf {
    let path = dir.path().join(name);
    std::fs::write(&path, bytes).expect("write book");
    path
}

/// The settings that let a loaded yorkie book answer a `go`: the book is
/// consulted at all, and never skipped at random.
fn require_book_is_consulted() {
    // A run-time check rather than a `const` one: a build of another config
    // fails this test alone, not the whole binary.
    if !config::USI_OWN_BOOK {
        panic!("these sessions need `usi_own_book = true`");
    }
    assert_eq!(
        config::BOOK_IGNORE_RATE,
        0,
        "these sessions need `book_ignore_rate = 0`"
    );
}

/// The setting that leaves a session with no `.ybb` book.
fn require_no_ybb_book() {
    assert_eq!(
        config::BOOK_FILE,
        "no_book",
        "these sessions need `book_file = \"no_book\"`"
    );
}

/// The preamble returns the board to the initial position, side to move
/// included; only the ply differs.
#[test]
fn the_preamble_returns_to_the_initial_board_at_ply_5() {
    let start = Position::startpos();
    let after = startpos_after(&PREAMBLE);
    assert_eq!(sfen_pack(&after), sfen_pack(&start));
    assert_eq!(start.ply(), 1);
    assert_eq!(after.ply(), 5);
}

/// A shared entry for the initial board answers at ply 5, behind the preamble,
/// as it does at ply 1.
#[cfg_attr(miri, ignore)]
#[test]
fn a_shared_yorkie_book_entry_for_the_initial_board_answers_after_the_preamble() {
    require_book_is_consulted();
    require_no_ybb_book();
    let _session = serial_session();
    stage_configured_eval_dir();
    let dir = TempDir::new("preamble-shared");
    let shared: Vec<YkEntry<'_>> = vec![(&[], vec![("2g2f", 10_000)], vec![])];
    let path = write(&dir, "shared.ykb", &yorkie_book_bytes(&shared, &[]));

    let out = drive_with_yorkie_book(
        &format!(
            "isready\n\
             usinewgame\n\
             {}{GO}\
             usinewgame\n\
             {}{GO}\
             quit\n",
            position(&[]),
            position(&PREAMBLE),
        ),
        TEST_BOOK_SEED,
        &path,
    );
    assert!(out.contains("readyok\n"), "{out}");
    assert_eq!(played(&out), vec!["2g2f", "2g2f"], "{out}");
}

/// A king step no search prefers there, so playing it means the entry fired.
const PATH_DEPENDENT_MOVE: &str = "5a4b";

/// A path-dependent entry whose support is a position reached after the
/// preamble applies in a game that passed through that position, and not in
/// one that reaches the entry's position by a transposition that skips it.
#[cfg_attr(miri, ignore)]
#[test]
fn a_path_dependent_yorkie_book_entry_applies_once_its_support_occurred_after_the_preamble() {
    require_book_is_consulted();
    require_no_ybb_book();
    let _session = serial_session();
    stage_configured_eval_dir();
    let dir = TempDir::new("preamble-pathdep");
    let target = after_preamble(&["7g7f", "3c3d", "2g2f"]);
    let support = after_preamble(&["7g7f"]);
    let pathdep: Vec<YkEntry<'_>> =
        vec![(&target, vec![(PATH_DEPENDENT_MOVE, 10_000)], vec![&support])];
    let path = write(&dir, "pathdep.ykb", &yorkie_book_bytes(&[], &pathdep));

    let transposed = after_preamble(&["2g2f", "3c3d", "7g7f"]);
    assert_eq!(
        sfen_pack(&startpos_after(&transposed)),
        sfen_pack(&startpos_after(&target)),
        "both orders reach the entry's position"
    );
    let out = drive_with_yorkie_book(
        &format!(
            "isready\n\
             usinewgame\n\
             {}{GO}\
             usinewgame\n\
             {}{GO}\
             quit\n",
            position(&target),
            position(&transposed),
        ),
        TEST_BOOK_SEED,
        &path,
    );
    assert!(
        out.contains("info string yorkie book loaded : 0 shared + 1 path-dependent entries\n"),
        "{out}"
    );
    let played = played(&out);
    assert_eq!(played.len(), 2, "{out}");
    assert_eq!(
        played[0], PATH_DEPENDENT_MOVE,
        "the support occurred: {out}"
    );
    assert_ne!(
        played[1], PATH_DEPENDENT_MOVE,
        "the support never occurred: {out}"
    );
}

/// The `.ybb` selection settings, every one spelled out: the book is consulted
/// up to ply `book_moves`, never skipped, and no entry below is filtered away.
fn book_config(book_moves: i64) -> BookConfig {
    BookConfig {
        book_options_v2: false,
        narrow_book: false,
        book_moves,
        ignore_rate: 0,
        eval_diff: 30,
        eval_black_diff: 0,
        eval_white_diff: 0,
        eval_black_limit: 0,
        eval_white_limit: 0,
        depth_limit: 0,
        depth_black_limit: 0,
        depth_white_limit: 0,
        consider_move_count: false,
        #[cfg(feature = "verbose2")]
        pv_moves: 8,
        flipped_book: false,
    }
}

/// The initial board in SFEN, at `ply`.
fn initial_board_at(ply: u16) -> String {
    format!("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - {ply}")
}

/// Every position the game `moves` reaches, the initial one included, as the
/// engine hands them to the yorkie book.
fn game_path(moves: &[&str]) -> Vec<PackedSfen> {
    (0..=moves.len())
        .map(|n| sfen_pack(&startpos_after(&moves[..n])))
        .collect()
}

/// What the probe a `go` runs plays at the position the game `moves` reaches,
/// given the yorkie book `yorkie` and the `.ybb` book `ybb`, with the
/// ply-matching rule `ignore_book_ply` and the ply limit `book_moves`.
fn probe_game(
    moves: &[&str],
    yorkie: Option<&YorkieBook>,
    ybb: &Book,
    ignore_book_ply: bool,
    book_moves: i64,
) -> Option<Move> {
    let pos = startpos_after(moves);
    let path = game_path(moves);
    let probe = yorkie.map(|book| YorkieProbe { book, path: &path });
    probe_book(
        probe.as_ref(),
        std::slice::from_ref(ybb),
        ignore_book_ply,
        &pos,
        &book_config(book_moves),
        &mut Prng::new(TEST_BOOK_SEED),
    )
    .hit
    .map(|hit| hit.best)
}

/// [`probe_game`] at the position behind the preamble, consulting the `.ybb`
/// book up to ply 16.
fn probe_after_preamble(
    yorkie: Option<&YorkieBook>,
    ybb: &Book,
    ignore_book_ply: bool,
) -> Option<Move> {
    probe_game(&PREAMBLE, yorkie, ybb, ignore_book_ply, 16)
}

/// A `.ybb` book whose one entry is the initial board recorded at `ply`,
/// answering `7g7f`.
fn initial_board_ybb(dir: &TempDir, ply: u16) -> Book {
    let path = dir.path().join(format!("ply{ply}.ybb"));
    write_ybb(&path, &[(&initial_board_at(ply), vec![("7g7f", 0, 0)])]);
    Book::open_in_memory(&path).expect("open the .ybb book")
}

/// The move `usi` in the position behind the preamble.
fn preamble_move(usi: &str) -> Move {
    let pos = startpos_after(&PREAMBLE);
    yorkie_state::parse_usi_move(usi.as_bytes(), &pos).expect("legal move")
}

/// Behind the preamble, where the yorkie book and a `.ybb` book both answer the
/// initial board, the yorkie book's move is the one played; where only the
/// `.ybb` book answers, its move is. The `.ybb` book answers through an entry
/// recorded at ply 5 with the ply checked, and through an entry recorded at
/// ply 1 with the ply ignored.
#[cfg_attr(miri, ignore)]
#[test]
fn behind_the_preamble_the_yorkie_book_is_probed_ahead_of_the_ybb_book() {
    let dir = TempDir::new("preamble-ybb");
    let at_ply_5 = initial_board_ybb(&dir, 5);
    let at_ply_1 = initial_board_ybb(&dir, 1);

    let answering: Vec<YkEntry<'_>> = vec![(&[], vec![("2g2f", 10_000)], vec![])];
    let answering = YorkieBook::from_bytes(yorkie_book_bytes(&answering, &[]))
        .expect("a well-formed yorkie book");
    // A yorkie book with an entry, but none for the initial board.
    let silent: Vec<YkEntry<'_>> = vec![(&["5g5f"], vec![("8c8d", 10_000)], vec![])];
    let silent =
        YorkieBook::from_bytes(yorkie_book_bytes(&silent, &[])).expect("a well-formed yorkie book");

    let yorkie_move = preamble_move("2g2f");
    let ybb_move = preamble_move("7g7f");
    for (ybb, ignore_book_ply, what) in [
        (&at_ply_5, false, "a ply-5 entry, ply checked"),
        (&at_ply_1, true, "a ply-1 entry, ply ignored"),
    ] {
        assert_eq!(
            probe_after_preamble(Some(&answering), ybb, ignore_book_ply),
            Some(yorkie_move),
            "both books answer ({what}): the yorkie book's move"
        );
        assert_eq!(
            probe_after_preamble(Some(&silent), ybb, ignore_book_ply),
            Some(ybb_move),
            "only the .ybb book answers ({what}): its move"
        );
        assert_eq!(
            probe_after_preamble(None, ybb, ignore_book_ply),
            Some(ybb_move),
            "no yorkie book ({what}): the .ybb book's move"
        );
    }
    // The ply-1 entry is not the ply-5 position's while the ply is checked.
    assert_eq!(probe_after_preamble(None, &at_ply_1, false), None);
}

/// Behind the preamble, with the `.ybb` book consulted at every ply
/// (`book_moves = 10000`), an entry recorded for the initial board at ply 1
/// answers the initial board at ply 5 when the recorded ply is ignored
/// (`ignore_book_ply = true`), and misses when it is compared
/// (`ignore_book_ply = false`).
#[cfg_attr(miri, ignore)]
#[test]
fn behind_the_preamble_a_ply_1_ybb_entry_answers_only_with_ignore_book_ply() {
    let dir = TempDir::new("preamble-ignore-ply");
    let at_ply_1 = initial_board_ybb(&dir, 1);
    let ybb_move = preamble_move("7g7f");

    assert_eq!(
        probe_game(&PREAMBLE, None, &at_ply_1, true, 10_000),
        Some(ybb_move),
        "ignore_book_ply = true: the ply-1 entry answers at ply 5"
    );
    assert_eq!(
        probe_game(&PREAMBLE, None, &at_ply_1, false, 10_000),
        None,
        "ignore_book_ply = false: the ply-1 entry misses at ply 5"
    );
}

/// The preamble played five times over: twenty king moves that leave the
/// initial board at ply 21.
fn five_preambles() -> Vec<&'static str> {
    PREAMBLE.iter().copied().cycle().take(20).collect()
}

/// A `.ybb` entry is found at a game ply above 16 when the book is consulted at
/// every ply (`book_moves = 10000`), and not when it is consulted only up to
/// ply 16 (`book_moves = 16`) — whether the entry matches through its recorded
/// ply (`ignore_book_ply = false`, an entry recorded at ply 21) or regardless
/// of it (`ignore_book_ply = true`, an entry recorded at ply 1).
#[cfg_attr(miri, ignore)]
#[test]
fn a_ybb_entry_answers_past_ply_16_only_with_book_moves_10000() {
    let moves = five_preambles();
    let pos = startpos_after(&moves);
    assert_eq!(pos.ply(), 21);
    assert_eq!(sfen_pack(&pos), sfen_pack(&Position::startpos()));
    let ybb_move = yorkie_state::parse_usi_move("7g7f".as_bytes(), &pos).expect("legal move");

    let dir = TempDir::new("ybb-past-ply-16");
    for (ybb, ignore_book_ply, what) in [
        (
            initial_board_ybb(&dir, 21),
            false,
            "a ply-21 entry, ply checked",
        ),
        (
            initial_board_ybb(&dir, 1),
            true,
            "a ply-1 entry, ply ignored",
        ),
    ] {
        assert_eq!(
            probe_game(&moves, None, &ybb, ignore_book_ply, 10_000),
            Some(ybb_move),
            "book_moves = 10000 ({what}): found at ply 21"
        );
        assert_eq!(
            probe_game(&moves, None, &ybb, ignore_book_ply, 16),
            None,
            "book_moves = 16 ({what}): not consulted at ply 21"
        );
    }
}

/// Whether `reply` is a legal move in the position the game `moves` reaches.
fn is_legal_after(moves: &[&str], reply: &str) -> bool {
    let pos = startpos_after(moves);
    yorkie_state::parse_usi_move(reply.as_bytes(), &pos).is_ok_and(|mv| legal(&pos).contains(&mv))
}

/// With no book at all, a `go` behind the preamble searches and replies with a
/// legal move — not a resignation, not a win declaration — for the side to
/// move at ply 5, and, one move later, for the other side.
#[cfg_attr(miri, ignore)]
#[test]
fn without_a_book_a_go_behind_the_preamble_searches_a_legal_move_for_either_side() {
    require_no_ybb_book();
    let _session = serial_session();
    stage_configured_eval_dir();

    let black = PREAMBLE.to_vec();
    let white = after_preamble(&["7g7f"]);
    let out = drive_with_yorkie_books(
        &format!(
            "isready\n\
             usinewgame\n\
             {}{GO}\
             {}{GO}\
             quit\n",
            position(&black),
            position(&white),
        ),
        TEST_BOOK_SEED,
        &[],
    );
    assert!(out.contains("readyok\n"), "{out}");
    assert!(
        !out.contains("info string yorkie book loaded"),
        "no yorkie book: {out}"
    );
    let played = played(&out);
    assert_eq!(played.len(), 2, "{out}");
    for (moves, reply, side) in [(&black, played[0], "black"), (&white, played[1], "white")] {
        assert_ne!(reply, "resign", "{side}: {out}");
        assert_ne!(reply, "win", "{side}: {out}");
        assert!(
            is_legal_after(moves, reply),
            "{side}: `{reply}` is not a legal move: {out}"
        );
    }
}
