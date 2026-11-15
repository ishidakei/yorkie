//! Driver-level sessions that tell which book each reply was played from.
//!
//! From `verbose1` up a book reply is preceded by `info string book: yorkie`
//! or `info string book: ybb`, naming the book that answered; a searched reply
//! carries neither. A tournament build prints no such line.
//!
//! The yorkie book and the `.ybb` book are named per session through
//! [`yorkie_protocol::UsiEngine::with_yorkie_books`] and
//! [`yorkie_protocol::UsiEngine::with_book_file`]; the network is the synthetic
//! all-zero one ([`common::stage_configured_eval_dir`]).

mod common;

use common::{
    TEST_BOOK_SEED, TempDir, YkEntry, bestmove_lines, drive_with_books, drive_with_seed,
    reply_books, serial_session, stage_configured_eval_dir, write_ybb, yorkie_book_bytes,
};
use yorkie_protocol::config;

const GO: &str = "go btime 10000 wtime 10000 byoyomi 0\n";

/// The position `2g2f 8c8d` reaches, at its ply.
const AFTER_2G2F_8C8D: &str = "lnsgkgsnl/1r5b1/p1ppppppp/1p7/9/7P1/PPPPPPP1P/1B5R1/LNSGKGSNL b - 3";

/// The settings that let a loaded book answer a `go` in these sessions.
fn require_book_is_consulted() {
    if !config::USI_OWN_BOOK {
        panic!("these sessions need `usi_own_book = true`");
    }
    assert_eq!(
        config::BOOK_IGNORE_RATE,
        0,
        "these sessions need `book_ignore_rate = 0`"
    );
    if config::BOOK_MOVES < 3 {
        panic!("these sessions need the `.ybb` book consulted at ply 3");
    }
}

/// A yorkie book answering the initial position with `2g2f`, and a `.ybb`
/// book answering the position `2g2f 8c8d` reaches with `2f2e`, staged in
/// `dir`.
fn stage_books(dir: &TempDir) -> (std::path::PathBuf, std::path::PathBuf) {
    let shared: Vec<YkEntry<'_>> = vec![(&[], vec![("2g2f", 10_000)], vec![])];
    let yorkie = dir.path().join("opening.ykb");
    std::fs::write(&yorkie, yorkie_book_bytes(&shared, &[])).expect("write yorkie book");
    let ybb = dir.path().join("user_book1.ybb");
    // A deep, even entry, so no depth floor or eval cutoff drops it.
    write_ybb(&ybb, &[(AFTER_2G2F_8C8D, vec![("2f2e", 0, 64)])]);
    (yorkie, ybb)
}

/// The yorkie book answers the first position, the `.ybb` book the second, and
/// the third, which neither holds, is searched: each reply names its book, or
/// none.
#[cfg_attr(miri, ignore)]
#[test]
fn each_book_reply_names_the_book_it_came_from() {
    require_book_is_consulted();
    let _session = serial_session();
    stage_configured_eval_dir();
    let dir = TempDir::new("book-source");
    let (yorkie, ybb) = stage_books(&dir);

    let out = drive_with_books(
        &format!(
            "isready\n\
             usinewgame\n\
             position startpos\n\
             {GO}\
             position startpos moves 2g2f 8c8d\n\
             {GO}\
             position startpos moves 7g7f 3c3d\n\
             {GO}\
             quit\n"
        ),
        TEST_BOOK_SEED,
        &ybb,
        &[(&yorkie, 1)],
    );
    assert!(out.contains("readyok\n"), "{out}");
    let played = bestmove_lines(&out);
    assert_eq!(played.len(), 3, "{out}");
    assert_eq!(&played[..2], ["2g2f", "2f2e"], "{out}");
    if cfg!(feature = "verbose1") {
        assert_eq!(
            reply_books(&out),
            vec![Some("yorkie"), Some("ybb"), None],
            "{out}"
        );
    } else {
        assert!(!out.contains("info string book: "), "{out}");
    }
}

/// With no book loaded no reply names one, whatever the build prints.
#[cfg_attr(miri, ignore)]
#[test]
fn a_session_without_a_book_names_none() {
    let _session = serial_session();
    stage_configured_eval_dir();
    assert_eq!(
        config::BOOK_FILE,
        "no_book",
        "this session needs no `.ybb` book"
    );
    assert!(
        config::YORKIE_BOOK_FILES.is_empty(),
        "this session needs no yorkie book"
    );
    let out = drive_with_seed(
        &format!("isready\nusinewgame\nposition startpos\n{GO}quit\n"),
        TEST_BOOK_SEED,
    );
    assert_eq!(bestmove_lines(&out).len(), 1, "{out}");
    assert!(!out.contains("info string book: "), "{out}");
}
