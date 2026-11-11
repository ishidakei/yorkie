//! The yorkie book's game path leaving out the first moves of each `position`
//! command, for games whose server plays a fixed preamble first: four king
//! moves, `5i5h 5a5b 5h5i 5b5a`, which return the board to the initial
//! position.
//!
//! How many moves are left out is the compile-time `yorkie_book_path_skip_moves`
//! setting, so each session here names its own count through
//! [`yorkie_protocol::UsiEngine::with_yorkie_book_path_skip_moves`], as it names
//! its yorkie book through [`yorkie_protocol::UsiEngine::with_yorkie_books`].
//! The network is the synthetic all-zero one
//! ([`common::stage_configured_eval_dir`]).

mod common;

use common::{
    TEST_BOOK_SEED, TempDir, YkEntry, bestmove_lines, drive_with_yorkie_book_skipping,
    serial_session, stage_configured_eval_dir, startpos_after, yorkie_book_bytes,
};
use yorkie_protocol::config;
use yorkie_state::sfen_pack;

/// The four king moves a server plays before the game proper.
const PREAMBLE: [&str; 4] = ["5i5h", "5a5b", "5h5i", "5b5a"];

/// The preamble's length: the count a tournament that plays it sets.
const PREAMBLE_SKIP: usize = PREAMBLE.len();

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
/// consulted at all, never skipped at random, and no `.ybb` book stands behind
/// it.
fn require_book_settings() {
    // A run-time check rather than a `const` one: a build of another config
    // fails these tests alone, not the whole binary.
    if !config::USI_OWN_BOOK {
        panic!("these sessions need `usi_own_book = true`");
    }
    assert_eq!(
        config::BOOK_IGNORE_RATE,
        0,
        "these sessions need `book_ignore_rate = 0`"
    );
    assert_eq!(
        config::BOOK_FILE,
        "no_book",
        "these sessions need `book_file = \"no_book\"`"
    );
}

/// A king step no search prefers there, so playing it means the entry fired.
const PATH_DEPENDENT_MOVE: &str = "5a4b";

/// The game the path-dependent entries below sit at the end of.
fn target() -> Vec<&'static str> {
    after_preamble(&["7g7f", "3c3d", "2g2f"])
}

/// One game, `moves`, played from a single path-dependent entry at the end of
/// [`target`] whose support is the position `support` reaches, with the first
/// `skip_moves` moves of the command left out of the game path. Returns the
/// move played.
fn play_with_support(support: &[&str], skip_moves: usize) -> String {
    let dir = TempDir::new(&format!("path-skip-{}-{skip_moves}", support.len()));
    let target = target();
    let pathdep: Vec<YkEntry<'_>> =
        vec![(&target, vec![(PATH_DEPENDENT_MOVE, 10_000)], vec![support])];
    let path = write(&dir, "pathdep.ykb", &yorkie_book_bytes(&[], &pathdep));
    let out = drive_with_yorkie_book_skipping(
        &format!(
            "isready\n\
             usinewgame\n\
             {}{GO}\
             quit\n",
            position(&target),
        ),
        TEST_BOOK_SEED,
        &path,
        skip_moves,
    );
    assert!(
        out.contains("info string yorkie book loaded : 0 shared + 1 path-dependent entries\n"),
        "{out}"
    );
    let played = played(&out);
    assert_eq!(played.len(), 1, "{out}");
    played[0].to_string()
}

/// With the preamble left out, an entry whose support is a position only the
/// preamble passes through — the one after `5i5h` — does not apply; with
/// nothing left out it does, as it always has.
#[cfg_attr(miri, ignore)]
#[test]
fn a_support_inside_the_preamble_counts_only_when_nothing_is_skipped() {
    require_book_settings();
    let _session = serial_session();
    stage_configured_eval_dir();
    let support = ["5i5h"];
    assert_ne!(
        sfen_pack(&startpos_after(&support)),
        sfen_pack(&startpos_after(&target()[..PREAMBLE_SKIP])),
        "the support is not the board the preamble ends on"
    );

    assert_ne!(
        play_with_support(&support, PREAMBLE_SKIP),
        PATH_DEPENDENT_MOVE,
        "a preamble position does not count once the preamble is skipped"
    );
    assert_eq!(
        play_with_support(&support, 0),
        PATH_DEPENDENT_MOVE,
        "with nothing skipped every position of the command counts"
    );
}

/// With the preamble left out, an entry whose support is a position reached
/// after the preamble still applies.
#[cfg_attr(miri, ignore)]
#[test]
fn a_support_after_the_preamble_counts_when_the_preamble_is_skipped() {
    require_book_settings();
    let _session = serial_session();
    stage_configured_eval_dir();
    let support = after_preamble(&["7g7f"]);
    assert_eq!(
        play_with_support(&support, PREAMBLE_SKIP),
        PATH_DEPENDENT_MOVE,
        "the support occurred after the preamble"
    );
}

/// A shared entry reads no game path, so it answers at the end of the
/// preamble, and after it, whether the preamble is left out or not.
#[cfg_attr(miri, ignore)]
#[test]
fn a_shared_entry_answers_whatever_is_skipped() {
    require_book_settings();
    let _session = serial_session();
    stage_configured_eval_dir();
    let dir = TempDir::new("path-skip-shared");
    let after_two = after_preamble(&["7g7f", "3c3d"]);
    let shared: Vec<YkEntry<'_>> = vec![
        (&[], vec![("2g2f", 10_000)], vec![]),
        (&after_two, vec![("8h2b+", 10_000)], vec![]),
    ];
    let path = write(&dir, "shared.ykb", &yorkie_book_bytes(&shared, &[]));

    for skip_moves in [0, PREAMBLE_SKIP] {
        let out = drive_with_yorkie_book_skipping(
            &format!(
                "isready\n\
                 usinewgame\n\
                 {}{GO}\
                 {}{GO}\
                 quit\n",
                position(&PREAMBLE),
                position(&after_two),
            ),
            TEST_BOOK_SEED,
            &path,
            skip_moves,
        );
        assert!(out.contains("readyok\n"), "{out}");
        assert_eq!(
            played(&out),
            vec!["2g2f", "8h2b+"],
            "skipping {skip_moves} moves: {out}"
        );
    }
}
