//! The per-reply statistics line, from the outside: where it sits in a session's
//! output and what one reply's number covers.
//!
//! Gated on `verbose1`, which is the feature that installs the counting
//! allocator and writes the line at all; a build without it is covered by
//! `info_gating.rs`, which pins that no such line appears.
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

/// The `(index, count)` of every `info string stats` line in `out`. An omitted
/// item is a zero one, `alloc` included.
fn stats_lines(out: &str) -> Vec<(usize, u64)> {
    out.lines()
        .enumerate()
        .filter_map(|(i, l)| {
            let items = stats_items(l)?;
            let n = items
                .iter()
                .find_map(|item| item.strip_prefix("alloc="))
                .map_or(0, |n| {
                    n.parse()
                        .unwrap_or_else(|e| panic!("{l:?} must carry a count: {e}"))
                });
            Some((i, n))
        })
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
    let _tt = common::serial_tt();
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

#[cfg_attr(miri, ignore)]
#[test]
fn a_reply_that_starts_no_search_still_reports_its_interval() {
    let _tt = common::serial_tt();
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
    /// Where [`PERPETUAL_CYCLE`] leaves the game: White to move, in check, one
    /// escape away from the repetition.
    const PERPETUAL_ROOT: &str = "9/R3k4/9/9/9/9/9/9/8K w - 12";

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

    /// Whether the reply's statistics line carries each mark, in the order
    /// `tt probe` spells them.
    fn reply_marks(out: &str) -> [bool; 3] {
        let items = reply_items(out);
        ["pathdep=1", "declrule=1", "movelimit=1"].map(|item| items.contains(&item))
    }

    /// The `pathdep` / `declrule` / `movelimit` triple of the `tt probe` reply
    /// in `out`, or `None` when the probe missed.
    fn probed_marks(out: &str) -> Option<[bool; 3]> {
        let line = out
            .lines()
            .find(|l| l.starts_with("info string tt probe "))
            .unwrap_or_else(|| panic!("a probe reply must be in:\n{out}"))
            .to_string();
        if !line.contains(" pathdep ") {
            return None;
        }
        let items: Vec<&str> = line.split(' ').collect();
        Some(["pathdep", "declrule", "movelimit"].map(|key| {
            let at = items
                .iter()
                .position(|item| *item == key)
                .unwrap_or_else(|| panic!("{key} must be in: {line}"));
            items[at + 1] == "1"
        }))
    }

    /// A reply whose move was chosen on a value a repetition judgement produced
    /// says so, and the root entry the same search stored says the same thing.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn a_repetition_behind_the_played_move_marks_the_reply() {
        let _tt = common::serial_tt();
        let h = ready_harness();
        search(
            &h,
            &format!("position sfen {PERPETUAL_CHECK} moves {PERPETUAL_CYCLE}"),
            2,
        );
        h.send(&format!("tt probe sfen {PERPETUAL_ROOT}"));
        let out = h.quit_join();

        assert_eq!(
            reply_marks(&out),
            [true, false, false],
            "the played move's value came through a repetition, in:\n{out}"
        );
        let probed = probed_marks(&out).expect("the root's entry survived the search");
        assert_eq!(
            probed,
            reply_marks(&out),
            "the root's entry must say what the reply said, in:\n{out}"
        );
    }

    /// `bestmove win` is the declaration rule's own outcome, so it carries that
    /// rule's mark.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn a_declaration_marks_the_reply_it_is() {
        let _tt = common::serial_tt();
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
        let _tt = common::serial_tt();
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
        let _tt = common::serial_tt();
        let h = ready_harness();
        search(&h, "position startpos", 2);
        let out = h.quit_join();

        let items = reply_items(&out);
        assert_eq!(items.len(), 1, "one item only, in:\n{out}");
        assert!(
            items[0].starts_with("alloc="),
            "and that item is the count, in:\n{out}"
        );
    }
}
