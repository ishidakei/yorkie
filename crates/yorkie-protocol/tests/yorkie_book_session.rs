//! Driver-level sessions against a yorkie book: the readiness handshake that
//! loads it, the `usinewgame` that draws one from the list, and the `go` it
//! answers.
//!
//! `yorkie_book_files` is compiled in, so each session names its own book files
//! through [`yorkie_protocol::UsiEngine::with_yorkie_books`]. The network is the
//! synthetic all-zero one ([`common::stage_configured_eval_dir`]).

mod common;

use common::{
    TEST_BOOK_SEED, TempDir, YkEntry, bestmove_lines, drive_with_yorkie_book,
    drive_with_yorkie_books, reseal_yorkie_book, seal_yorkie_book, serial_session,
    stage_configured_eval_dir, yorkie_book_bytes,
};

/// Two shared entries — the initial position answers `2g2f`, the position after
/// `2g2f 8c8d` answers `2f2e` — and a path-dependent entry for the position
/// `7g7f 3c3d 2g2f` reaches, which applies only if the post-`7g7f` position
/// occurred.
fn book_image() -> Vec<u8> {
    let shared: Vec<YkEntry<'_>> = vec![
        (&[], vec![("2g2f", 10_000)], vec![]),
        (&["2g2f", "8c8d"], vec![("2f2e", 10_000)], vec![]),
    ];
    let pathdep: Vec<YkEntry<'_>> = vec![(
        &["7g7f", "3c3d", "2g2f"],
        vec![("8c8d", 10_000)],
        vec![&["7g7f"]],
    )];
    yorkie_book_bytes(&shared, &pathdep)
}

fn write(dir: &TempDir, name: &str, bytes: &[u8]) -> std::path::PathBuf {
    let path = dir.path().join(name);
    std::fs::write(&path, bytes).expect("write yorkie book");
    path
}

const GO: &str = "go btime 10000 wtime 10000 byoyomi 0\n";

#[cfg_attr(miri, ignore)]
#[test]
fn a_loaded_yorkie_book_answers_shared_and_path_dependent_positions() {
    let _session = serial_session();
    stage_configured_eval_dir();
    let dir = TempDir::new("yorkie-hit");
    let path = write(&dir, "best.ykb", &book_image());

    let out = drive_with_yorkie_book(
        &format!(
            "isready\n\
             usinewgame\n\
             position startpos\n\
             {GO}\
             position startpos moves 2g2f 8c8d\n\
             {GO}\
             position startpos moves 7g7f 3c3d 2g2f\n\
             {GO}\
             quit\n"
        ),
        TEST_BOOK_SEED,
        &path,
    );
    assert!(out.contains("readyok\n"), "{out}");
    assert!(
        out.contains("info string yorkie book loaded : 2 shared + 1 path-dependent entries\n"),
        "{out}"
    );
    // Past `book_moves` or not, every answer is the book's: the yorkie book
    // has no ply limit, and the third position is only in its path-dependent
    // section.
    assert_eq!(bestmove_lines(&out), vec!["2g2f", "2f2e", "8c8d"], "{out}");
}

/// A book whose only entry answers the initial position with `mv`.
fn opening_book(mv: &str) -> Vec<u8> {
    let shared: Vec<YkEntry<'_>> = vec![(&[], vec![(mv, 10_000)], vec![])];
    yorkie_book_bytes(&shared, &[])
}

/// The files the session's `info string yorkie book:` lines name, in order.
fn drawn_books(out: &str) -> Vec<&str> {
    out.lines()
        .filter_map(|l| l.strip_prefix("info string yorkie book: "))
        .collect()
}

/// Of two books that answer the initial position differently, each game plays
/// from the one its `usinewgame` draw names: with the seed pinned, the two
/// games below draw different files and play different first moves.
#[cfg_attr(miri, ignore)]
#[test]
fn each_game_plays_from_the_yorkie_book_it_draws() {
    let _session = serial_session();
    stage_configured_eval_dir();
    let dir = TempDir::new("yorkie-draw");
    let left = write(&dir, "left.ykb", &opening_book("2g2f"));
    let right = write(&dir, "right.ykb", &opening_book("7g7f"));

    let out = drive_with_yorkie_books(
        &format!(
            "isready\n\
             usinewgame\n\
             position startpos\n\
             {GO}\
             usinewgame\n\
             position startpos\n\
             {GO}\
             quit\n"
        ),
        DRAW_SEED,
        &[(&left, 1), (&right, 1)],
    );
    assert!(out.contains("readyok\n"), "{out}");
    let drawn = drawn_books(&out);
    assert_eq!(drawn.len(), 2, "one report per game: {out}");
    assert_ne!(drawn[0], drawn[1], "the seed draws each file once: {out}");
    let played = bestmove_lines(&out);
    assert_eq!(played.len(), 2, "{out}");
    for (file, played) in drawn.iter().zip(&played) {
        let book_move = if file.ends_with("left.ykb") {
            "2g2f"
        } else {
            assert!(file.ends_with("right.ykb"), "{file}");
            "7g7f"
        };
        assert_eq!(*played, book_move, "the game drawing {file}: {out}");
    }
}

/// The session seed under which [`each_game_plays_from_the_yorkie_book_it_draws`]
/// draws a different file for each of its two games.
const DRAW_SEED: u64 = TEST_BOOK_SEED;

/// A session that starts its first game without `usinewgame` plays from the
/// first entry, however light it is, and reports no draw.
#[cfg_attr(miri, ignore)]
#[test]
fn a_game_without_usinewgame_plays_from_the_first_yorkie_book() {
    let _session = serial_session();
    stage_configured_eval_dir();
    let dir = TempDir::new("yorkie-first");
    let first = write(&dir, "first.ykb", &opening_book("2g2f"));
    let heavy = write(&dir, "heavy.ykb", &opening_book("7g7f"));

    let out = drive_with_yorkie_books(
        &format!("isready\nposition startpos\n{GO}quit\n"),
        TEST_BOOK_SEED,
        &[(&first, 1), (&heavy, 1_000)],
    );
    assert!(drawn_books(&out).is_empty(), "{out}");
    assert_eq!(bestmove_lines(&out), vec!["2g2f"], "{out}");
}

/// Every listed file is checked: a faulty one withholds `readyok` wherever it
/// stands in the list, and the refusal names it.
#[cfg_attr(miri, ignore)]
#[test]
fn a_faulty_file_anywhere_in_the_list_withholds_readyok() {
    let _session = serial_session();
    stage_configured_eval_dir();
    let dir = TempDir::new("yorkie-list-reject");
    let good = write(&dir, "good.ykb", &book_image());
    let missing = dir.path().join("missing.ykb");

    let out = drive_with_yorkie_books(
        "isready\nquit\n",
        TEST_BOOK_SEED,
        &[(&good, 1), (&missing, 1)],
    );
    assert!(!out.contains("readyok"), "readyok in:\n{out}");
    let line = out
        .lines()
        .find(|l| l.starts_with("info string yorkie book load failed : "))
        .unwrap_or_else(|| panic!("no refusal in:\n{out}"));
    assert!(line.contains("missing.ykb"), "the file is named: {line}");
}

/// The isready refusal of the yorkie book at `path`: the `info string` that
/// names it, with `readyok` withheld.
fn refusal(path: &std::path::Path) -> String {
    let out = drive_with_yorkie_book("isready\nquit\n", TEST_BOOK_SEED, path);
    assert!(!out.contains("readyok"), "readyok in:\n{out}");
    out.lines()
        .find(|l| l.starts_with("info string yorkie book load failed : "))
        .unwrap_or_else(|| panic!("no refusal in:\n{out}"))
        .to_string()
}

/// A well-formed version 1 file — the 64-byte header that declares its own
/// length and offsets for both tables, and no entries — is refused by its
/// version, not read as some other fault.
#[cfg_attr(miri, ignore)]
#[test]
fn a_version_1_yorkie_book_is_refused_by_its_version() {
    let _session = serial_session();
    let dir = TempDir::new("yorkie-v1");
    let mut header = vec![0u8; 64];
    header[..8].copy_from_slice(b"YKBOOK01");
    header[8..10].copy_from_slice(&1u16.to_le_bytes());
    header[12..16].copy_from_slice(&64u32.to_le_bytes());
    header[32..40].copy_from_slice(&64u64.to_le_bytes());
    header[40..48].copy_from_slice(&64u64.to_le_bytes());
    let path = write(&dir, "v1.ykb", &seal_yorkie_book(header));

    let line = refusal(&path);
    assert!(line.contains("v1.ykb"), "the file is named: {line}");
    assert!(
        line.contains("unsupported version 1 (this reader parses version 2)"),
        "{line}"
    );
}

/// Every way a configured yorkie book is refused: an `info string` naming
/// the fault, and no `readyok`.
#[cfg_attr(miri, ignore)]
#[test]
fn a_faulty_yorkie_book_withholds_readyok() {
    let _session = serial_session();
    let dir = TempDir::new("yorkie-reject");
    let good = book_image();

    let mut bad_magic = good.clone();
    bad_magic[0] = b'X';
    let mut bad_trailer = good.clone();
    let n = bad_trailer.len();
    bad_trailer[n - 1] = b'X';
    let mut version = good.clone();
    version[8] = 3;
    let mut checksum = good.clone();
    checksum[70] ^= 1;
    let mut offsets = good.clone();
    offsets[16] = 0xFF;
    // Swap the two shared offsets, so the keys run backwards.
    let mut order = good.clone();
    let (first, second) = order[40..56].split_at_mut(8);
    first.swap_with_slice(second);

    let cases: Vec<(&str, Option<Vec<u8>>, &str)> = vec![
        ("missing.ykb", None, "i/o error"),
        ("magic.ykb", Some(bad_magic), "bad magic"),
        ("trailer.ykb", Some(bad_trailer), "bad trailer magic"),
        (
            "version.ykb",
            Some(reseal_yorkie_book(version)),
            "unsupported version 3",
        ),
        ("checksum.ykb", Some(checksum), "SHA-256 mismatch"),
        (
            "offsets.ykb",
            Some(reseal_yorkie_book(offsets)),
            "outside the file",
        ),
        (
            "order.ykb",
            Some(reseal_yorkie_book(order)),
            "keys out of order",
        ),
    ];
    for (name, bytes, reason) in cases {
        let path = match bytes {
            Some(bytes) => write(&dir, name, &bytes),
            None => dir.path().join(name),
        };
        let out = drive_with_yorkie_book("isready\nquit\n", TEST_BOOK_SEED, &path);
        assert!(!out.contains("readyok"), "{name}: readyok in:\n{out}");
        let line = out
            .lines()
            .find(|l| l.starts_with("info string yorkie book load failed : "))
            .unwrap_or_else(|| panic!("{name}: no refusal in:\n{out}"));
        assert!(line.contains(name), "{name}: the file is named: {line}");
        assert!(line.contains(reason), "{name}: `{reason}` in: {line}");
    }
}

/// A book the game below never hits: a shared entry for a position it does not
/// reach, and a path-dependent entry for a position it does reach, by a move
/// order other than the one the entry's support names.
#[cfg(feature = "verbose1")]
fn missed_book_image() -> Vec<u8> {
    let shared: Vec<YkEntry<'_>> = vec![(&["5g5f"], vec![("8c8d", 10_000)], vec![])];
    let pathdep: Vec<YkEntry<'_>> = vec![(
        &["2g2f", "3c3d", "7g7f"],
        vec![("8c8d", 10_000)],
        vec![&["2g2f"]],
    )];
    yorkie_book_bytes(&shared, &pathdep)
}

/// The `alloc=` count of every `info string stats` line in `out`, an omitted
/// item counting as zero.
#[cfg(feature = "verbose1")]
fn reply_allocations(out: &str) -> Vec<u64> {
    out.lines()
        .filter_map(|l| l.strip_prefix("info string stats "))
        .map(|items| {
            items
                .split(' ')
                .find_map(|item| item.strip_prefix("alloc="))
                .map_or(0, |n| n.parse().expect("a count"))
        })
        .collect()
}

/// Recording and handing over the game path takes no allocation once the game
/// has started: over one game, every reply of a session reading a yorkie book
/// allocates no more than the same reply without one.
#[cfg(feature = "verbose1")]
#[cfg_attr(miri, ignore)]
#[test]
fn a_loaded_yorkie_book_adds_no_allocation_to_a_reply() {
    // Each game draws its evaluation noise from process entropy, so two noisy
    // sessions run different searches and their counts are not comparable.
    #[cfg(feature = "random")]
    if yorkie_protocol::config::RANDOM != 0 {
        eprintln!("skipped: this build's evaluation noise differs between the two sessions");
        return;
    }
    let _session = serial_session();
    stage_configured_eval_dir();
    let dir = TempDir::new("yorkie-alloc");
    let path = write(&dir, "missed.ykb", &missed_book_image());

    let game = [
        "",
        " moves 7g7f",
        " moves 7g7f 3c3d",
        " moves 7g7f 3c3d 2g2f",
        " moves 7g7f 3c3d 2g2f 8c8d",
        " moves 7g7f 3c3d 2g2f 8c8d 2f2e",
        " moves 7g7f 3c3d 2g2f 8c8d 2f2e 8d8e",
    ];
    let mut input = String::from("isready\nusinewgame\n");
    for moves in game {
        input.push_str(&format!("position startpos{moves}\n{GO}"));
    }
    input.push_str("quit\n");

    let with_book = common::drive_counting_allocations(&input, TEST_BOOK_SEED, Some(&path));
    let without = common::drive_counting_allocations(&input, TEST_BOOK_SEED, None);
    assert!(
        with_book
            .contains("info string yorkie book loaded : 1 shared + 1 path-dependent entries\n"),
        "{with_book}"
    );
    assert_eq!(
        bestmove_lines(&with_book),
        bestmove_lines(&without),
        "a book the game never hits leaves every reply as it was"
    );
    let (with_book, without) = (reply_allocations(&with_book), reply_allocations(&without));
    assert_eq!(with_book.len(), game.len(), "one count per reply");
    assert_eq!(without.len(), game.len(), "one count per reply");
    for (reply, (with_book, without)) in with_book.iter().zip(&without).enumerate() {
        assert!(
            with_book <= without,
            "reply {reply}: {with_book} allocations with the yorkie book, {without} without"
        );
    }
}
