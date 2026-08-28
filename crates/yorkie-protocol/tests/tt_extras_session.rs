//! Driver-level session tests for the `usi-extras` `tt` commands: `tt store` /
//! `tt probe` round trips (including the mate convention and the centipawn
//! quantisation), the one-ply `tt children` sweep, and the error surface.
//!
//! The whole file is gated on the feature — with `usi-extras` off it compiles to
//! nothing, matching the default build where the `tt` token is not a command at
//! all. The complementary "feature off" assertion lives in
//! `src/parser.rs::tests::tt_is_not_a_command_without_usi_extras`, which is
//! compiled only when the feature is OFF; run `cargo nextest run -p
//! yorkie-protocol` (default features, i.e. without `--all-features`) to execute
//! it.
//!
//! No network is needed for most of them: a `bench` carries its own table size
//! as a command argument, so `bench 1 1 1 current movetime` allocates a 1 MiB
//! table on its own and (finding no network loaded) resigns each position
//! immediately. These sessions therefore never touch `isready` / `nn.bin`.

#![cfg(feature = "usi-extras")]

mod common;

use common::{StreamHarness, drive, stage_configured_eval_dir};
use yorkie_state::{format_sfen, parse_sfen, parse_usi_move};

const STARTPOS: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

/// Run `script` against a driver whose table is already sized, and return every
/// `tt …` reply with the `info string ` prefix stripped.
fn tt_session(script: &[&str]) -> Vec<String> {
    let mut input = String::from("bench 1 1 1 current movetime\n");
    for line in script {
        input.push_str(line);
        input.push('\n');
    }
    input.push_str("quit\n");
    replies(&drive(&input))
}

/// Like [`tt_session`] but WITHOUT the `bench` line, so the table is still
/// unsized (the state right after process start, before `isready`).
fn tt_session_unsized(script: &[&str]) -> Vec<String> {
    let mut input = String::new();
    for line in script {
        input.push_str(line);
        input.push('\n');
    }
    input.push_str("quit\n");
    replies(&drive(&input))
}

fn replies(out: &str) -> Vec<String> {
    out.lines()
        .filter_map(|l| l.strip_prefix("info string tt ").map(str::to_string))
        .collect()
}

/// The SFEN reached from the start position by `moves`.
fn sfen_after(moves: &[&str]) -> String {
    let mut pos = parse_sfen(STARTPOS).expect("startpos sfen");
    for m in moves {
        let mv = parse_usi_move(m, &pos).expect("well-formed setup move");
        pos.do_move(mv);
    }
    format_sfen(&pos)
}

fn store_at(sfen: &str, rest: &str) -> String {
    format!("tt store sfen {sfen} {rest}")
}

// -------------------------------------------------------------------------
// 1. `tt store` → `tt probe` round trip.
// -------------------------------------------------------------------------

#[cfg_attr(miri, ignore)]
#[test]
fn store_then_probe_returns_the_same_entry() {
    let got = tt_session(&[
        "tt store startpos move 7g7f value 100 depth 12 bound exact eval 50 pv",
        "tt probe startpos",
    ]);
    assert_eq!(
        got,
        vec![
            "store ok".to_string(),
            // 100 cp → 90 internal → 100 cp; 50 cp → 45 internal → 50 cp. Both
            // survive the USI PawnValue scale exactly.
            "probe hit move 7g7f value cp 100 depth 12 bound exact eval cp 50 pv true".to_string(),
        ]
    );
}

#[cfg_attr(miri, ignore)]
#[test]
fn omitting_pv_stores_a_non_pv_entry() {
    let got = tt_session(&[
        "tt store startpos move 2g2f value 0 depth 3 bound lower eval 0",
        "tt probe startpos",
    ]);
    assert_eq!(
        got,
        vec![
            "store ok".to_string(),
            "probe hit move 2g2f value cp 0 depth 3 bound lower eval cp 0 pv false".to_string(),
        ]
    );
}

/// The entry format's quantisation, stated explicitly.
///
/// Two independent lossy steps sit between the command surface and the stored
/// bits, and this pins both:
///
/// 1. **The USI centipawn scale.** `cp` maps to the engine's internal value
///    through `PawnValue == 90` (`v = cp * 90 / 100`, and back `cp = 100 * v /
///    90`), with truncating division in each direction — so a `cp` argument is
///    quantised to a multiple of the 9/10 ratio and small values collapse.
/// 2. **`value16` / `eval16`.** Both are `i16` in the entry — in either entry
///    layout, since `tt-entry16` spends its extra bytes entirely on the key and
///    leaves the payload fields alone; every value that survives step 1 fits,
///    so step 1 is the only *observable* loss here.
///
/// `mate` arguments are exact — they bypass the centipawn scale entirely.
#[cfg_attr(miri, ignore)]
#[test]
fn centipawn_round_trip_quantises_to_the_usi_pawn_scale() {
    // 101 cp → 101*90/100 = 90 internal → 100*90/90 = 100 cp.
    // 109 cp →  98 internal → 108 cp. 5 cp → 4 internal → 4 cp.
    for (given, expected) in [(100, 100), (101, 100), (109, 108), (5, 4), (-101, -100)] {
        let got = tt_session(&[
            &format!("tt store startpos move none value {given} depth 1 bound exact eval {given}"),
            "tt probe startpos",
        ]);
        assert_eq!(
            got,
            vec![
                "store ok".to_string(),
                format!(
                    "probe hit move none value cp {expected} depth 1 bound exact eval cp {expected} pv false"
                ),
            ],
            "cp {given} must quantise to cp {expected}"
        );
    }
}

// -------------------------------------------------------------------------
// 2. The mate / root-relative ply convention.
// -------------------------------------------------------------------------

#[cfg_attr(miri, ignore)]
#[test]
fn mate_scores_round_trip_through_store_and_probe() {
    let got = tt_session(&[
        "tt store startpos move 7g7f value mate 5 depth 30 bound exact eval 0 pv",
        "tt probe startpos",
    ]);
    assert_eq!(
        got,
        vec![
            "store ok".to_string(),
            "probe hit move 7g7f value mate 5 depth 30 bound exact eval cp 0 pv true".to_string(),
        ]
    );

    let got = tt_session(&[
        "tt store startpos move 7g7f value mate -5 depth 30 bound exact eval 0",
        "tt probe startpos",
    ]);
    assert_eq!(
        got,
        vec![
            "store ok".to_string(),
            "probe hit move 7g7f value mate -5 depth 30 bound exact eval cp 0 pv false".to_string(),
        ]
    );
}

/// The named position is the root. A mate written against a *child*'s own SFEN
/// reads back unchanged when that child is itself the named position, and one
/// ply further out when it is reported as a child of its parent — the shift the
/// reference's `value_from_tt(v, ply)` applies, here with `ply == 1`.
#[cfg_attr(miri, ignore)]
#[test]
fn children_report_values_relative_to_the_named_position() {
    let child = sfen_after(&["7g7f"]);
    let got = tt_session(&[
        &store_at(
            &child,
            "move 3c3d value mate 5 depth 30 bound exact eval 0 pv",
        ),
        &format!("tt probe sfen {child}"),
        "tt children startpos",
    ]);
    assert_eq!(
        got,
        vec![
            "store ok".to_string(),
            // Named as the root: exactly what was stored.
            "probe hit move 3c3d value mate 5 depth 30 bound exact eval cp 0 pv true".to_string(),
            // Named as a child of startpos: one ply further from the root.
            "child 7g7f move 3c3d value mate 6 depth 30 bound exact eval cp 0 pv true".to_string(),
            "children end 1".to_string(),
        ]
    );
}

/// Non-mate values carry no ply term, so a centipawn child reads back identically
/// through either route.
#[cfg_attr(miri, ignore)]
#[test]
fn children_do_not_shift_non_mate_values() {
    let child = sfen_after(&["2g2f"]);
    let got = tt_session(&[
        &store_at(&child, "move 8c8d value 90 depth 7 bound upper eval -90"),
        &format!("tt probe sfen {child}"),
        "tt children startpos",
    ]);
    assert_eq!(
        got,
        vec![
            "store ok".to_string(),
            "probe hit move 8c8d value cp 90 depth 7 bound upper eval cp -90 pv false".to_string(),
            "child 2g2f move 8c8d value cp 90 depth 7 bound upper eval cp -90 pv false".to_string(),
            "children end 1".to_string(),
        ]
    );
}

// -------------------------------------------------------------------------
// 3. `tt children`.
// -------------------------------------------------------------------------

#[cfg_attr(miri, ignore)]
#[test]
fn children_lists_only_seeded_children() {
    let seeded = ["7g7f", "2g2f", "6i7h"];
    // A fourth legal move deliberately left unseeded.
    let unseeded = "1g1f";

    let mut script: Vec<String> = seeded
        .iter()
        .enumerate()
        .map(|(i, mv)| {
            store_at(
                &sfen_after(&[mv]),
                &format!(
                    "move none value {} depth {} bound lower eval 0",
                    (i as i32 + 1) * 100,
                    i + 4
                ),
            )
        })
        .collect();
    script.push("tt children startpos".to_string());
    let refs: Vec<&str> = script.iter().map(String::as_str).collect();
    let got = tt_session(&refs);

    // Children come out in the position's legal-move order, which is movegen's
    // business and not part of this command's contract — compare as a set.
    let mut children: Vec<&str> = got
        .iter()
        .map(String::as_str)
        .filter(|l| l.starts_with("child "))
        .collect();
    children.sort_unstable();
    let mut expected = vec![
        "child 7g7f move none value cp 100 depth 4 bound lower eval cp 0 pv false",
        "child 2g2f move none value cp 200 depth 5 bound lower eval cp 0 pv false",
        "child 6i7h move none value cp 300 depth 6 bound lower eval cp 0 pv false",
    ];
    expected.sort_unstable();
    assert_eq!(children, expected, "got: {got:?}");
    assert_eq!(got.last().map(String::as_str), Some("children end 3"));
    assert!(
        !got.iter().any(|l| l.contains(unseeded)),
        "an unseeded child must produce no line; got: {got:?}"
    );
}

#[cfg_attr(miri, ignore)]
#[test]
fn children_of_an_untouched_position_lists_nothing() {
    let got = tt_session(&["tt children startpos"]);
    assert_eq!(got, vec!["children end 0".to_string()]);
}

#[cfg_attr(miri, ignore)]
#[test]
fn probe_of_an_untouched_position_is_a_miss() {
    let got = tt_session(&["tt probe startpos"]);
    assert_eq!(got, vec!["probe miss".to_string()]);
}

// -------------------------------------------------------------------------
// 4. Error surface — every one a single line, none a panic.
// -------------------------------------------------------------------------

#[cfg_attr(miri, ignore)]
#[test]
fn malformed_input_yields_one_error_line_each() {
    let cases: &[&str] = &[
        // Subcommand.
        "tt",
        "tt frobnicate startpos",
        // Position clause.
        "tt probe",
        "tt probe nonsense",
        "tt probe sfen a b c",
        "tt probe sfen zzz9/9/9/9/9/9/9/9/9 b - 1",
        "tt probe sfen 9/9/9/9/9/9/9/9/9 x - 1",
        "tt children sfen 9/9/9/9/9/9/9/9/9 b - 1",
        "tt probe startpos and more",
        // Store clauses.
        "tt store startpos move 7g7f value 0 depth 1 bound exact",
        "tt store startpos move 7g7f value 0 depth 1 bound sideways eval 0",
        "tt store startpos move 7g7f value nonsense depth 1 bound exact eval 0",
        "tt store startpos move 7g7f value mate 999 depth 1 bound exact eval 0",
        "tt store startpos move 7g7f value 999999 depth 1 bound exact eval 0",
        "tt store startpos move 7g7f value 0 depth 9999 bound exact eval 0",
        "tt store startpos move 7g7f value 0 depth -3 bound exact eval 0",
        // Move token: syntactically broken, and legal-but-not-here.
        "tt store startpos move zzzz value 0 depth 1 bound exact eval 0",
        "tt store startpos move 1a1b value 0 depth 1 bound exact eval 0",
        "tt store startpos move 7g7e value 0 depth 1 bound exact eval 0",
    ];
    for case in cases {
        let got = tt_session(&[case]);
        assert_eq!(got.len(), 1, "`{case}` must produce one reply; got {got:?}");
        assert!(
            got[0].starts_with("error: "),
            "`{case}` must be an error; got {got:?}"
        );
    }
}

/// A kingless board parses as an SFEN but has no meaningful move generation, so
/// it is rejected before movegen rather than risking a panic there.
#[cfg_attr(miri, ignore)]
#[test]
fn a_position_without_kings_is_rejected() {
    let got = tt_session(&["tt probe sfen 9/9/9/9/9/9/9/9/9 b - 1"]);
    assert_eq!(got.len(), 1);
    assert!(got[0].contains("no king"), "got {got:?}");
}

/// Before `isready` (or an explicit `USI_Hash`) the table has zero clusters, and
/// probing it would panic. Every subcommand must report that instead.
#[cfg_attr(miri, ignore)]
#[test]
fn an_unallocated_table_is_a_clear_error() {
    for case in [
        "tt probe startpos",
        "tt children startpos",
        "tt store startpos move 7g7f value 0 depth 1 bound exact eval 0",
    ] {
        let got = tt_session_unsized(&[case]);
        assert_eq!(got.len(), 1, "`{case}` → {got:?}");
        assert!(
            got[0].contains("not allocated"),
            "`{case}` must report the unsized table; got {got:?}"
        );
    }
}

/// A malformed line must never disturb an entry that is already there.
#[cfg_attr(miri, ignore)]
#[test]
fn a_rejected_store_leaves_the_existing_entry_intact() {
    let got = tt_session(&[
        "tt store startpos move 7g7f value 100 depth 12 bound exact eval 50 pv",
        "tt store startpos move 7g7e value 999 depth 12 bound exact eval 0",
        "tt probe startpos",
    ]);
    assert_eq!(got.len(), 3);
    assert_eq!(got[0], "store ok");
    assert!(got[1].starts_with("error: "), "got {got:?}");
    assert_eq!(
        got[2],
        "probe hit move 7g7f value cp 100 depth 12 bound exact eval cp 50 pv true"
    );
}

// -------------------------------------------------------------------------
// 5. Replacement policy / generation.
// -------------------------------------------------------------------------

/// `tt store` goes through `TTEntry::save` exactly as a search would, so the
/// replacement policy applies — and a write it declines is reported as skipped
/// rather than silently claimed. A shallower non-exact re-store of the same
/// position within one generation is the reference's declined case.
#[cfg_attr(miri, ignore)]
#[test]
fn a_declined_write_is_reported_not_silently_dropped() {
    let got = tt_session(&[
        "tt store startpos move 7g7f value 100 depth 40 bound exact eval 0 pv",
        "tt store startpos move 2g2f value 200 depth 1 bound lower eval 0",
        "tt probe startpos",
    ]);
    assert_eq!(
        got,
        vec![
            "store ok".to_string(),
            "store skipped (replacement policy kept the existing entry)".to_string(),
            // The deeper exact entry survives — except for its move, which
            // `TTEntry::save` always refreshes when a new one is supplied.
            "probe hit move 2g2f value cp 100 depth 40 bound exact eval cp 0 pv true".to_string(),
        ]
    );
}

// -------------------------------------------------------------------------
// 6. Idle-only admission.
// -------------------------------------------------------------------------

/// Bring up a session with a synthetic (all-zero) network and the compiled-in
/// table, blocking until `readyok`.
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

/// The commands read and write the table the search workers are churning, so
/// they are refused mid-search. No sleep is needed to make this deterministic:
/// the driver's read loop handles one line at a time, so by the time the `tt`
/// line is read the `go infinite` worker is already registered.
#[cfg_attr(miri, ignore)]
#[test]
fn a_running_search_refuses_the_tt_commands() {
    let h = ready_harness();

    h.send("position startpos");
    h.send("go infinite");
    h.send("tt probe startpos");
    assert!(
        h.wait_until(30_000, |o| o.contains("info string tt error: ")),
        "a mid-search `tt probe` must report an error; got:\n{}",
        h.output()
    );
    h.send("stop");
    let out = h.quit_join();

    let replies = replies(&out);
    assert_eq!(replies.len(), 1, "got {replies:?}");
    assert!(
        replies[0].contains("a search is running"),
        "got {replies:?}"
    );
}

/// A finished-but-unjoined worker is not "searching": the normal
/// `go … → bestmove → tt probe` sequence must work without an intervening
/// `stop`.
#[cfg_attr(miri, ignore)]
#[test]
fn a_finished_search_does_not_block_the_tt_commands() {
    let h = ready_harness();

    h.send("position startpos");
    h.send("go depth 1");
    assert!(
        h.wait_until(30_000, |o| o.contains("bestmove ")),
        "the depth-1 search must finish; got:\n{}",
        h.output()
    );
    h.send("tt probe startpos");
    let out = h.quit_join();

    let replies = replies(&out);
    assert_eq!(replies.len(), 1, "got {replies:?}");
    assert!(
        replies[0].starts_with("probe hit ") || replies[0] == "probe miss",
        "expected a probe result, got {replies:?}"
    );
}

/// `usinewgame` clears the table (and its generation), so entries do not leak
/// across games.
#[cfg_attr(miri, ignore)]
#[test]
fn usinewgame_clears_stored_entries() {
    let got = tt_session(&[
        "tt store startpos move 7g7f value 100 depth 12 bound exact eval 0",
        "usinewgame",
        "tt probe startpos",
    ]);
    assert_eq!(got, vec!["store ok".to_string(), "probe miss".to_string()]);
}

// -------------------------------------------------------------------------
// 6. `usi-extras` × `tt-entry16` — the command family over the wide table.
// -------------------------------------------------------------------------

/// The two features are independent and compose: with both on, the whole `tt`
/// family drives a table of 16-byte, full-64-bit-key entries. Every test above
/// already runs in that configuration under `--all-features`; this one exists to
/// name the combination explicitly and to exercise all three subcommands over
/// one wide table in a single session, at a scale (a dozen distinct positions,
/// each with its own payload) that a single narrow assertion would not reach.
///
/// What it deliberately does *not* try to do is discriminate the two layouts.
/// That takes hand-chosen colliding keys, which this surface cannot produce —
/// a `tt` command names a position and the key falls out of it — so the
/// aliasing proofs live at the Storage layer, in
/// `yorkie-storage/tests/tt_basic.rs::wide_key_identity`, where keys are
/// constructed bit by bit.
#[cfg(feature = "tt-entry16")]
#[cfg_attr(miri, ignore)]
#[test]
fn the_tt_commands_round_trip_through_the_wide_table() {
    // Twelve distinct legal first moves, hence twelve distinct child positions
    // and twelve distinct 64-bit keys.
    const CHILDREN: [&str; 12] = [
        "7g7f", "2g2f", "1g1f", "9g9f", "3g3f", "4g4f", "5g5f", "6g6f", "8g8f", "6i7h", "4i3h",
        "3i4h",
    ];

    // Store one entry per child, each with its own value and depth.
    let mut script: Vec<String> = CHILDREN
        .iter()
        .enumerate()
        .map(|(i, mv)| {
            store_at(
                &sfen_after(&[mv]),
                &format!(
                    "move none value {} depth {} bound lower eval 0",
                    (i as i32 + 1) * 100,
                    i + 4
                ),
            )
        })
        .collect();
    // Then probe each child on its own SFEN (ply 0) …
    script.extend(
        CHILDREN
            .iter()
            .map(|mv| format!("tt probe sfen {}", sfen_after(&[mv]))),
    );
    // … and sweep them all as children of the start position (ply 1).
    script.push("tt children startpos".to_string());

    let refs: Vec<&str> = script.iter().map(String::as_str).collect();
    let got = tt_session(&refs);

    // Every store landed: a fresh 1 MiB table has room for all twelve, and no
    // two of them are the same position.
    let stores: Vec<&String> = got.iter().filter(|l| l.starts_with("store ")).collect();
    assert_eq!(stores.len(), CHILDREN.len());
    assert!(
        stores.iter().all(|l| *l == "store ok"),
        "every distinct position must get its own entry; got: {got:?}"
    );

    // Every probe reads back that position's own payload, unshifted.
    let probes: Vec<&str> = got
        .iter()
        .map(String::as_str)
        .filter(|l| l.starts_with("probe "))
        .collect();
    let expected_probes: Vec<String> = (0..CHILDREN.len())
        .map(|i| {
            format!(
                "probe hit move none value cp {} depth {} bound lower eval cp 0 pv false",
                (i + 1) * 100,
                i + 4
            )
        })
        .collect();
    assert_eq!(probes, expected_probes, "got: {got:?}");

    // `tt children startpos` finds exactly the twelve seeded children (the
    // start position has more legal moves than that, and the rest stay silent).
    let mut children: Vec<&str> = got
        .iter()
        .map(String::as_str)
        .filter(|l| l.starts_with("child "))
        .collect();
    children.sort_unstable();
    let mut expected_children: Vec<String> = CHILDREN
        .iter()
        .enumerate()
        .map(|(i, mv)| {
            format!(
                "child {mv} move none value cp {} depth {} bound lower eval cp 0 pv false",
                (i + 1) * 100,
                i + 4
            )
        })
        .collect();
    expected_children.sort_unstable();
    assert_eq!(children, expected_children, "got: {got:?}");
    assert_eq!(
        got.last().map(String::as_str),
        Some(&format!("children end {}", CHILDREN.len())[..])
    );
}
