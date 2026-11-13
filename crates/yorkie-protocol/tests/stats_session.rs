//! The per-reply statistics line, from the outside: where it sits in a session's
//! output and what one reply's number covers.
//!
//! Gated on `verbose1`, which is the feature that installs the counting
//! allocator and writes the line at all; a build without it is covered by
//! `info_gating.rs`, which pins that no such line appears.
//!
//! Beside `alloc` the line counts what the searches of the interval did —
//! `nodes`, `tt_update`, `eval_dup`, `evals` and `eval_tt_hits` — and leaves
//! out any of them that is zero, as a reply that ran no search shows.
//!
//! The line is written by taking the counter, so what the previous reply
//! reported can never be reported again — the two numbers below being drawn from
//! two separate searches is the observable form of that.
//!
//! The three items a `verbose3` build adds — the value marks of the score the
//! reply rests on — are in the `value_marks` module at the end, gated on that
//! feature.

#![cfg(feature = "verbose1")]

mod common;

use common::{drive, stage_configured_eval_dir};

const STATS_PREFIX: &str = "info string stats ";

/// The `key=value` items of an `info string stats` line, or `None` for any
/// other line.
fn stats_items(line: &str) -> Option<Vec<&str>> {
    Some(line.strip_prefix(STATS_PREFIX)?.split(' ').collect())
}

/// The six counted items, in the order the line writes them.
const COUNT_KEYS: [&str; 6] = [
    "alloc",
    "nodes",
    "tt_update",
    "eval_dup",
    "evals",
    "eval_tt_hits",
];

/// The value of the `key` item on the statistics line `line`. An omitted item
/// is a zero one.
fn stats_count(line: &str, key: &str) -> u64 {
    let items = stats_items(line).unwrap_or_else(|| panic!("{line:?} is no statistics line"));
    items
        .iter()
        .find_map(|item| item.strip_prefix(key)?.strip_prefix('='))
        .map_or(0, |n| {
            n.parse()
                .unwrap_or_else(|e| panic!("{line:?} must carry a count for {key}: {e}"))
        })
}

/// The `(index, alloc count)` of every `info string stats` line in `out`.
fn stats_lines(out: &str) -> Vec<(usize, u64)> {
    out.lines()
        .enumerate()
        .filter(|(_, l)| l.starts_with(STATS_PREFIX))
        .map(|(i, l)| (i, stats_count(l, "alloc")))
        .collect()
}

/// The counted items of a statistics line, as `(key, value)` in the order they
/// were written.
fn counted_items(line: &str) -> Vec<(&str, &str)> {
    stats_items(line)
        .unwrap_or_else(|| panic!("{line:?} is no statistics line"))
        .into_iter()
        .filter_map(|item| item.split_once('='))
        .filter(|(key, _)| COUNT_KEYS.contains(key))
        .collect()
}

fn bestmove_indices(out: &str) -> Vec<usize> {
    out.lines()
        .enumerate()
        .filter(|(_, l)| l.starts_with("bestmove "))
        .map(|(i, _)| i)
        .collect()
}

#[cfg_attr(miri, ignore)]
#[test]
fn every_reply_reports_the_allocations_of_its_own_interval() {
    let _session = common::serial_session();
    stage_configured_eval_dir();
    let out = drive(
        "usi\n\
         isready\n\
         usinewgame\n\
         position startpos moves 7g7f 3c3d\n\
         go byoyomi 1000\n\
         go byoyomi 1000\n\
         quit\n",
    );
    assert!(
        out.contains("readyok\n"),
        "the network must load in:\n{out}"
    );

    let bestmoves = bestmove_indices(&out);
    assert_eq!(bestmoves.len(), 2, "expected two replies in:\n{out}");

    let stats = stats_lines(&out);
    assert_eq!(
        stats.len(),
        2,
        "expected one statistics line per reply in:\n{out}"
    );

    for (&(stats_at, count), &bestmove_at) in stats.iter().zip(&bestmoves) {
        assert_eq!(
            stats_at + 1,
            bestmove_at,
            "the statistics line must be the line immediately before its \
             bestmove in:\n{out}"
        );
        assert!(
            count > 0,
            "a reply that searched allocated something, got {count} in:\n{out}"
        );
    }
}

/// A reply that searched reports what its search did: nodes searched and
/// entries written. Every item the line carries is a non-zero count, in the
/// line's order — a zero one is left out rather than written.
#[cfg_attr(miri, ignore)]
#[test]
fn a_searched_reply_reports_its_nodes_and_table_updates() {
    let _session = common::serial_session();
    stage_configured_eval_dir();
    let out = drive(
        "usi\n\
         isready\n\
         usinewgame\n\
         position startpos moves 7g7f 3c3d\n\
         go byoyomi 1000\n\
         quit\n",
    );
    assert!(
        out.contains("readyok\n"),
        "the network must load in:\n{out}"
    );
    let line = out
        .lines()
        .find(|l| l.starts_with(STATS_PREFIX))
        .unwrap_or_else(|| panic!("a statistics line must be in:\n{out}"));

    assert!(stats_count(line, "nodes") > 0, "{line:?} in:\n{out}");
    assert!(stats_count(line, "tt_update") > 0, "{line:?} in:\n{out}");

    let items = counted_items(line);
    let mut order = items
        .iter()
        .map(|(key, _)| COUNT_KEYS.iter().position(|k| k == key));
    assert!(
        order.clone().is_sorted(),
        "the counted items come in the order {COUNT_KEYS:?}, got {line:?}"
    );
    assert!(
        order.all(|at| at.is_some()),
        "only known items, got {line:?}"
    );
    for (key, value) in items {
        assert_ne!(value, "0", "a zero {key} is omitted, got {line:?}");
    }
}

/// With one search thread, the reply's `nodes` is the count the search's own
/// last `info` line reported: the line before it reports the whole search, and
/// the statistics line reports every search of the interval, which here is
/// that one.
#[cfg(feature = "verbose2")]
#[cfg_attr(miri, ignore)]
#[test]
fn a_single_threaded_replys_nodes_are_its_last_info_lines() {
    use common::StreamHarness;
    use yorkie_protocol::config;

    assert_eq!(config::THREADS, 1, "this pins a single-threaded search");
    let _session = common::serial_session();
    stage_configured_eval_dir();
    let h = StreamHarness::start();
    h.send("usi");
    h.send("isready");
    assert!(
        h.wait_until(30_000, |o| o.contains("readyok")),
        "network must load and ack readyok"
    );
    h.send("position startpos moves 7g7f 3c3d");
    h.send("go depth 4");
    assert!(
        h.wait_until(60_000, |o| o.contains("bestmove ")),
        "the search must finish; got:\n{}",
        h.output()
    );
    let out = h.quit_join();

    let info_nodes = out
        .lines()
        .filter(|l| l.starts_with("info depth "))
        .filter_map(|l| {
            let mut words = l.split(' ');
            words.find(|w| *w == "nodes")?;
            words.next()?.parse::<u64>().ok()
        })
        .next_back()
        .unwrap_or_else(|| panic!("the search must report its nodes in:\n{out}"));
    let line = out
        .lines()
        .find(|l| l.starts_with(STATS_PREFIX))
        .unwrap_or_else(|| panic!("a statistics line must be in:\n{out}"));
    assert_eq!(stats_count(line, "nodes"), info_nodes, "in:\n{out}");
}

/// A searched reply counts the static evaluations its search made and the ones
/// it took from the table. With one search thread no other worker can write a
/// node's eval between its probe and its store, and this search leaves no other
/// way for a store to find one, so no store is a duplicate.
#[cfg_attr(miri, ignore)]
#[test]
fn a_single_threaded_reply_counts_its_evaluations_and_no_duplicate() {
    use yorkie_protocol::config;

    assert_eq!(config::THREADS, 1, "this pins a single-threaded search");
    let _session = common::serial_session();
    stage_configured_eval_dir();
    // A fixed node count where the build accepts the clause; a build without
    // `verbose2` refuses it, so there the search is bounded by the clock.
    let go = if cfg!(feature = "verbose2") {
        "go nodes 20000"
    } else {
        "go byoyomi 1000"
    };
    let out = drive(&format!(
        "usi\n\
         isready\n\
         usinewgame\n\
         position startpos moves 7g7f 3c3d\n\
         {go}\n\
         quit\n"
    ));
    assert!(
        out.contains("readyok\n"),
        "the network must load in:\n{out}"
    );
    let line = out
        .lines()
        .find(|l| l.starts_with(STATS_PREFIX))
        .unwrap_or_else(|| panic!("a statistics line must be in:\n{out}"));
    assert!(line.contains(" evals="), "{line:?} in:\n{out}");
    assert!(line.contains(" eval_tt_hits="), "{line:?} in:\n{out}");
    assert!(
        stats_count(line, "evals") + stats_count(line, "eval_tt_hits") > 0,
        "{line:?} in:\n{out}"
    );
    assert!(!line.contains("eval_dup"), "{line:?} in:\n{out}");
}

/// A `dupstats` build's line also totals how the workers repeated each other's
/// work. With one search thread every store finds its own work or none, and no
/// node entry finds another worker inside its position, so every count that
/// names another worker is zero and left off the line.
#[cfg(feature = "dupstats")]
#[cfg_attr(miri, ignore)]
#[test]
fn a_single_threaded_reply_counts_its_stores_and_no_other_worker() {
    use yorkie_protocol::config;

    assert_eq!(config::THREADS, 1, "this pins a single-threaded search");
    let _session = common::serial_session();
    stage_configured_eval_dir();
    let out = drive(
        "usi\n\
         isready\n\
         usinewgame\n\
         position startpos moves 7g7f 3c3d\n\
         go byoyomi 1000\n\
         quit\n",
    );
    let line = out
        .lines()
        .find(|l| l.starts_with(STATS_PREFIX))
        .unwrap_or_else(|| panic!("a statistics line must be in:\n{out}"));
    for key in ["st_new", "pr_enter", "sp_searches", "sp_workers"] {
        assert!(stats_count(line, key) > 0, "{key} in {line:?}");
    }
    assert_eq!(stats_count(line, "sp_searches"), 1, "{line:?}");
    assert_eq!(stats_count(line, "sp_workers"), 1, "{line:?}");
    for key in [
        "st_deep_other",
        "st_there_other",
        "pr_deeper",
        "pr_equal",
        "pr_shallower",
        "dup_nodes",
        "sp_behind",
        "ow_shallow_other",
    ] {
        assert!(
            !line.contains(&format!(" {key}=")),
            "{key} must be zero in {line:?}"
        );
    }
}

#[cfg_attr(miri, ignore)]
#[test]
fn a_reply_that_starts_no_search_still_reports_its_interval() {
    let _session = common::serial_session();
    // No network staged for this session — the working directory is wherever the
    // previous test left it, so this asserts only the shape a `bestmove` line
    // comes in, not which move it is.
    let out = drive(
        "position startpos\n\
         go byoyomi 1000\n\
         quit\n",
    );
    let bestmoves = bestmove_indices(&out);
    assert_eq!(bestmoves.len(), 1, "expected one reply in:\n{out}");
    let stats = stats_lines(&out);
    assert_eq!(stats.len(), 1, "expected one statistics line in:\n{out}");
    assert_eq!(
        stats[0].0 + 1,
        bestmoves[0],
        "the statistics line must directly precede the bestmove in:\n{out}"
    );
    // No search ran, so its counts are zero and left off the line.
    let line = out.lines().nth(stats[0].0).expect("the statistics line");
    for key in ["nodes", "tt_update", "eval_dup", "evals", "eval_tt_hits"] {
        assert_eq!(stats_count(line, key), 0, "{line:?} in:\n{out}");
        assert!(!line.contains(key), "a zero {key} is omitted, got {line:?}");
    }
}

/// The three items a `verbose3` build adds: what the value the reply rests on —
/// the score of the root move the engine plays — was derived through. A caller
/// recording the reply keeps them beside the move, so the positions a change of
/// repetition path, declaration rule or move limit would put back in doubt can
/// be searched again later.
#[cfg(feature = "verbose3")]
mod value_marks {
    use super::*;
    use common::StreamHarness;
    use yorkie_protocol::config;

    /// A Black rook checks a shuffling White king along the far file, and the
    /// four-ply cycle has already run twice, so White's escape back to 5a makes
    /// the position a fourfold one ply into the search. The perpetually-checking
    /// side loses that repetition, which makes the escape White's winning move
    /// and its value a repetition judgement's.
    const PERPETUAL_CHECK: &str = "4k4/R8/9/9/9/9/9/9/8K b - 1";
    const PERPETUAL_CYCLE: &str = "9b9a 5a5b 9a9b 5b5a 9b9a 5a5b 9a9b 5b5a 9b9a 5a5b 9a9b";

    /// Black's king on 5b with twelve pieces in the enemy field and a rook in
    /// hand — 32 points, past either point rule's threshold — against a lone
    /// White king in the far corner, with Black to move. The reply is the
    /// declaration itself.
    const BLACK_TO_MOVE_AND_DECLARING: &str = "+R+R+B+B5/3GKG3/2SGGGS2/9/9/9/9/9/8k b R 1";

    /// Two bare kings at a game ply past `configs/test-limits.toml`'s move
    /// limit, where every move the search can make draws by that rule.
    const TWO_KINGS_AT_PLY_60: &str = "4k4/9/9/9/9/9/9/9/4K4 b - 60";

    /// Bring up a session with a synthetic (all-zero) network, blocking until
    /// `readyok`.
    fn ready_harness() -> StreamHarness {
        stage_configured_eval_dir();
        let h = StreamHarness::start();
        h.send("usi");
        h.send("isready");
        assert!(
            h.wait_until(30_000, |o| o.contains("readyok")),
            "network must load and ack readyok"
        );
        h
    }

    /// Set `position`, search it to `depth` and block until the reply is out.
    fn search(h: &StreamHarness, position: &str, depth: u32) {
        h.send(position);
        h.send(&format!("go depth {depth}"));
        assert!(
            h.wait_until(30_000, |o| o.contains("bestmove ")),
            "the search must finish; got:\n{}",
            h.output()
        );
    }

    /// The statistics line that precedes the first `bestmove` of `out`, as its
    /// items.
    fn reply_items(out: &str) -> Vec<&str> {
        let at = out
            .lines()
            .position(|l| l.starts_with("bestmove "))
            .unwrap_or_else(|| panic!("a reply must be in:\n{out}"));
        let line = out.lines().nth(at - 1).expect("a line before the reply");
        stats_items(line)
            .unwrap_or_else(|| panic!("{line:?} must be the statistics line in:\n{out}"))
    }

    /// Whether `key` is one of the counts a `dupstats` build adds — how the
    /// search workers repeated each other's work, and what each one's search
    /// did with the table and in its move loop. No key is, in a build without
    /// it.
    fn is_dup_key(key: &str) -> bool {
        const TABLE_KEYS: [&str; 4] = ["tt_cut", "tt_deep_nocut", "tt_shallow", "tt_miss"];
        cfg!(feature = "dupstats")
            && (["st_", "pr_", "sp_", "ow_", "se_", "cut_"]
                .iter()
                .any(|p| key.starts_with(p))
                || TABLE_KEYS.contains(&key)
                || key == "cuts"
                || key == "dup_nodes"
                || key == "wt_replaced")
    }

    /// Whether the reply's statistics line carries each mark, in the order
    /// `pathdep`, `declrule`, `movelimit`.
    fn reply_marks(out: &str) -> [bool; 3] {
        let items = reply_items(out);
        ["pathdep=1", "declrule=1", "movelimit=1"].map(|item| items.contains(&item))
    }

    /// A reply whose move was chosen on a value a repetition judgement produced
    /// says so.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn a_repetition_behind_the_played_move_marks_the_reply() {
        let _session = common::serial_session();
        let h = ready_harness();
        search(
            &h,
            &format!("position sfen {PERPETUAL_CHECK} moves {PERPETUAL_CYCLE}"),
            2,
        );
        let out = h.quit_join();

        assert_eq!(
            reply_marks(&out),
            [true, false, false],
            "the played move's value came through a repetition, in:\n{out}"
        );
    }

    /// `bestmove win` is the declaration rule's own outcome, so it carries that
    /// rule's mark.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn a_declaration_marks_the_reply_it_is() {
        let _session = common::serial_session();
        let h = ready_harness();
        search(
            &h,
            &format!("position sfen {BLACK_TO_MOVE_AND_DECLARING}"),
            2,
        );
        let out = h.quit_join();

        assert!(
            out.contains("bestmove win\n"),
            "this position is declared, not searched, in:\n{out}"
        );
        assert_eq!(
            reply_marks(&out),
            [false, true, false],
            "a declaration reports the declaration rule, in:\n{out}"
        );
    }

    /// The move-limit draw, which only a build whose `MaxMovesToDraw` is below
    /// the fixture's game ply can reach: `0` means unlimited, and under such a
    /// config the same search meets no rule at all.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn a_reply_past_the_move_limit_marks_it() {
        let _session = common::serial_session();
        let horizon = config::MAX_MOVES_TO_DRAW;
        let h = ready_harness();
        search(&h, &format!("position sfen {TWO_KINGS_AT_PLY_60}"), 2);
        let out = h.quit_join();

        let expected = horizon != 0 && horizon < 60;
        assert_eq!(
            reply_marks(&out),
            [false, false, expected],
            "movelimit under MaxMovesToDraw {horizon}, in:\n{out}"
        );
    }

    /// A reply whose value met none of the three prints the line every build
    /// prints: the items exist, but an unset one is written no more than a zero
    /// count is.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn a_reply_that_met_no_rule_prints_the_line_unchanged() {
        let _session = common::serial_session();
        let h = ready_harness();
        search(&h, "position startpos", 2);
        let out = h.quit_join();

        let items = reply_items(&out);
        assert!(
            items.iter().any(|item| item.starts_with("alloc=")),
            "the line carries its counts, in:\n{out}"
        );
        assert!(
            items.iter().all(|item| item
                .split_once('=')
                .is_some_and(|(key, _)| COUNT_KEYS.contains(&key) || is_dup_key(key))),
            "and nothing but its counts, in:\n{out}"
        );
    }
}
