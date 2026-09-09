//! The transposition table, against the semantics of the reference.
//!
//! # The one table
//!
//! The table is a `static` sized when the binary was built, so these tests do
//! not make one: they take the shared table, empty it, and hold a lock for as
//! long as they read what they wrote. Under `cargo nextest` each test is its
//! own process and the lock is never contended; under a plain `cargo test` it
//! is what keeps two tests from clearing each other's entries.
//!
//! # The addressing model these tests use
//!
//! The table holds [`CLUSTER_COUNT`] clusters — a power of two for every
//! `usi_hash` the configs use — so with `CLUSTER_BITS = log2(CLUSTER_COUNT)`,
//!
//! ```text
//! cluster_index_pre_side = mul_hi64(key, CLUSTER_COUNT) = key >> (64 - CLUSTER_BITS)
//! in_cluster_key_frag    = key & 0xffff
//! ```
//!
//! The top `CLUSTER_BITS` bits select the cluster and the low 16 are the stored
//! key fragment — disjoint ranges, so [`key`] sets each independently. The side
//! to move is OR-ed into cluster-index bit 0, so fixing `hi` **and** `side`
//! keeps a family of keys in one cluster while the fragment varies.
//!
//! The bits between are read by neither, which is why [`key_mid`] exists: a key
//! varying only there is indistinguishable from its sibling to the default
//! build and distinguishable to a `tt-entry16` one. That is the whole
//! observable difference between the two layouts.
//!
//! A cluster holds 3 entries by default and 2 under `tt-entry16`, so the two
//! tests whose walk-through only reads correctly for one count carry a
//! `cfg`-selected pair of bodies rather than skipping under the other.
//!
//! Every test here is ignored under miri: clearing the table alone is millions
//! of interpreted atomic writes — tens of minutes per test. What the `static`
//! itself has to be is covered there by the crate's own unit tests, over a
//! prefix miri can finish.

use std::ops::Deref;
use std::sync::{Mutex, MutexGuard};

use yorkie_storage::{Bound, CLUSTER_COUNT, DEPTH_NONE, TT_ALIGN, TTData, TranspositionTable};

/// Entries per 32-byte cluster, restating the crate-private `CLUSTER_SIZE`.
const CLUSTER_ENTRIES: usize = if cfg!(feature = "tt-entry16") { 2 } else { 3 };

/// How many key bits select a cluster.
const CLUSTER_BITS: u32 = CLUSTER_COUNT.trailing_zeros();

/// Where the cluster-selecting bits start.
const CLUSTER_SHIFT: u32 = 64 - CLUSTER_BITS;

/// Serialises the tests against the one shared table (see the module docs).
static TT_LOCK: Mutex<()> = Mutex::new(());

/// Exclusive use of the emptied table, released when the binding goes out of
/// scope. Derefs to the table, so a test writes `&tt` where a
/// `&TranspositionTable` is wanted.
struct TtHandle {
    _guard: MutexGuard<'static, ()>,
    tt: &'static TranspositionTable,
}

impl Deref for TtHandle {
    type Target = TranspositionTable;

    fn deref(&self) -> &TranspositionTable {
        self.tt
    }
}

fn fresh_tt() -> TtHandle {
    // A panicking test leaves the lock poisoned; the next test wants the table,
    // not the panic, and empties it before reading anything.
    let guard = TT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let tt = TranspositionTable::shared();
    tt.clear();
    TtHandle { _guard: guard, tt }
}

/// Build a key that lands in cluster `hi` (before the side fold) with
/// in-cluster fragment `frag`. Requires `hi < 2^CLUSTER_BITS`.
fn key(hi: u64, frag: u16) -> u64 {
    key_mid(hi, 0, frag)
}

/// [`key`] with the otherwise-unused middle bits set to `mid`.
fn key_mid(hi: u64, mid: u64, frag: u16) -> u64 {
    assert!(
        CLUSTER_COUNT.is_power_of_two(),
        "this addressing model needs a power-of-two cluster count, got {CLUSTER_COUNT}"
    );
    assert!(hi < (1 << CLUSTER_BITS));
    assert!(mid < (1 << (CLUSTER_SHIFT - 16)));
    (hi << CLUSTER_SHIFT) | (mid << 16) | frag as u64
}

/// Probe `k` and store through the returned writer, at the table's current
/// generation, leaving the entry unmarked.
#[allow(clippy::too_many_arguments)]
fn store(
    tt: &TranspositionTable,
    k: u64,
    side: u8,
    value: i32,
    pv: bool,
    bound: Bound,
    depth: i32,
    mv: u16,
    eval: i32,
) {
    let generation = tt.generation();
    let (_, _, w) = tt.probe(k, side);
    w.write(
        k,
        value,
        pv,
        bound,
        depth,
        mv,
        eval,
        generation,
        #[cfg(feature = "verbose3")]
        false,
    );
}

#[cfg_attr(miri, ignore)]
#[test]
fn store_probe_round_trip_every_field() {
    let tt = fresh_tt();

    let k = key(100, 0x1234);
    let side = 0;

    // Miss on an empty table.
    let (found, data, w) = tt.probe(k, side);
    assert!(!found);
    assert_eq!(data, miss_sentinel());
    w.write(
        k,
        -321,
        true,
        Bound::Lower,
        17,
        0x0abc,
        -654,
        tt_generation_zero(),
        #[cfg(feature = "verbose3")]
        false,
    );

    // Hit: every field survives the round trip.
    let (found, data, _) = tt.probe(k, side);
    assert!(found);
    assert_eq!(data.value, -321);
    assert_eq!(data.eval, -654);
    assert_eq!(data.depth, 17);
    assert_eq!(data.bound, Bound::Lower);
    assert!(data.is_pv);
    assert_eq!(data.move16, 0x0abc);
}

#[cfg_attr(miri, ignore)]
#[test]
fn every_bound_and_pv_combination_round_trips() {
    let tt = fresh_tt();
    let side = 1;

    for (i, bound) in [Bound::None, Bound::Upper, Bound::Lower, Bound::Exact]
        .into_iter()
        .enumerate()
    {
        for pv in [false, true] {
            // Distinct fragment per case so they don't overwrite each other.
            let frag = 0x100 + (i as u16) * 2 + pv as u16;
            let k = key(7, frag);
            store(&tt, k, side, 10 + i as i32, pv, bound, 5, frag, -10);
            let (found, data, _) = tt.probe(k, side);
            assert!(found, "bound={bound:?} pv={pv} should be found");
            assert_eq!(data.bound, bound);
            assert_eq!(data.is_pv, pv);
            assert_eq!(data.value, 10 + i as i32);
        }
    }
}

#[cfg_attr(miri, ignore)]
#[test]
fn miss_on_wrong_key() {
    let tt = fresh_tt();
    let side = 0;

    let stored = key(42, 0xBEEF);
    store(&tt, stored, side, 100, false, Bound::Exact, 8, 0x0111, 100);

    // Same cluster, different fragment → miss.
    let (found, data, _) = tt.probe(key(42, 0xBEE0), side);
    assert!(!found);
    assert_eq!(data, miss_sentinel());

    // Different cluster entirely → miss.
    let (found, _, _) = tt.probe(key(99, 0xBEEF), side);
    assert!(!found);

    // Same key but opposite side lands in a different cluster → miss.
    let (found, _, _) = tt.probe(stored, 1);
    assert!(!found);
}

#[cfg_attr(miri, ignore)]
#[test]
#[cfg(not(feature = "tt-entry16"))]
fn replacement_evicts_lowest_priority_entry() {
    // Fill one three-entry cluster, all written at generation 0 so every
    // relative_age is 0 and replace_priority == depth8 == depth − DEPTH_NONE.
    //
    //   slot 0: frag 1, depth 10 → depth8 13, priority 13
    //   slot 1: frag 2, depth  5 → depth8  8, priority  8   ← lowest
    //   slot 2: frag 3, depth 20 → depth8 23, priority 23
    //
    // A miss then replaces the lowest-priority entry, i.e. slot 1 (frag 2).
    let tt = fresh_tt();
    let side = 0;
    let hi = 100;

    store(&tt, key(hi, 1), side, 0, false, Bound::Lower, 10, 1, 0);
    store(&tt, key(hi, 2), side, 0, false, Bound::Lower, 5, 2, 0);
    store(&tt, key(hi, 3), side, 0, false, Bound::Lower, 20, 3, 0);

    // All three present before the eviction.
    assert!(tt.probe(key(hi, 1), side).0);
    assert!(tt.probe(key(hi, 2), side).0);
    assert!(tt.probe(key(hi, 3), side).0);

    // Miss on frag 4 → writer targets the evicted slot; write frag 4 there.
    store(&tt, key(hi, 4), side, 0, false, Bound::Lower, 1, 4, 0);

    // frag 2 (depth 5) was the least valuable and is gone; frags 1, 3, 4 remain.
    assert!(
        !tt.probe(key(hi, 2), side).0,
        "frag 2 should have been evicted"
    );
    assert!(tt.probe(key(hi, 1), side).0, "frag 1 should survive");
    assert!(tt.probe(key(hi, 3), side).0, "frag 3 should survive");
    assert!(tt.probe(key(hi, 4), side).0, "frag 4 should now be present");
}

/// The `tt-entry16` counterpart: the same replacement scan over a **2**-entry
/// cluster. Two entries fill it, and the third store evicts the shallower one.
#[cfg_attr(miri, ignore)]
#[test]
#[cfg(feature = "tt-entry16")]
fn replacement_evicts_lowest_priority_entry() {
    // Both written at generation 0, so every relative_age is 0 and
    // replace_priority == depth8 == depth − DEPTH_NONE.
    //
    //   slot 0: frag 2, depth 10 → depth8 13, priority 13
    //   slot 1: frag 4, depth  5 → depth8  8, priority  8   ← lowest
    //
    // The cluster is now full, so a miss replaces slot 1 (frag 4).
    //
    // The fragments are spaced by two because key bit 0 is the
    // path-dependence mark at `verbose3`, so consecutive ones would name the
    // same entry there.
    let tt = fresh_tt();
    let side = 0;
    let hi = 100;

    store(&tt, key(hi, 2), side, 0, false, Bound::Lower, 10, 1, 0);
    store(&tt, key(hi, 4), side, 0, false, Bound::Lower, 5, 2, 0);

    // Both present before the eviction.
    assert!(tt.probe(key(hi, 2), side).0);
    assert!(tt.probe(key(hi, 4), side).0);

    // Miss on frag 6 → writer targets the evicted slot; write frag 6 there.
    store(&tt, key(hi, 6), side, 0, false, Bound::Lower, 1, 3, 0);

    assert!(
        !tt.probe(key(hi, 4), side).0,
        "frag 4 (depth 5) should have been evicted"
    );
    assert!(tt.probe(key(hi, 2), side).0, "frag 2 should survive");
    assert!(tt.probe(key(hi, 6), side).0, "frag 6 should now be present");
}

#[cfg_attr(miri, ignore)]
#[test]
#[cfg(not(feature = "tt-entry16"))]
fn generation_aging_lowers_replacement_priority() {
    // A deep-but-old entry loses to a shallow-but-fresh one once enough
    // generations pass, because replace_priority = depth8 − 8·relative_age.
    //
    // After three new_search() bumps the table is at generation 3:
    //   P: frag 1, depth 20, gen 0 → depth8 23, age 3, priority 23 − 24 = −1  ← lowest
    //   Q: frag 2, depth  3, gen 3 → depth8  6, age 0, priority  6
    //   R: frag 3, depth  8, gen 3 → depth8 11, age 0, priority 11
    //
    // Without aging P's priority would be 23 (highest, never evicted); aging
    // flips it to the lowest, so the miss evicts P.
    let tt = fresh_tt();
    let side = 0;
    let hi = 200;

    // P written at generation 0.
    assert_eq!(tt.generation(), 0);
    store(&tt, key(hi, 1), side, 0, false, Bound::Lower, 20, 1, 0);

    // Advance to generation 3, then write Q and R.
    tt.new_search();
    tt.new_search();
    tt.new_search();
    assert_eq!(tt.generation(), 3);
    store(&tt, key(hi, 2), side, 0, false, Bound::Lower, 3, 2, 0);
    store(&tt, key(hi, 3), side, 0, false, Bound::Lower, 8, 3, 0);

    // Sanity: all three occupy the cluster.
    assert!(tt.probe(key(hi, 1), side).0);
    assert!(tt.probe(key(hi, 2), side).0);
    assert!(tt.probe(key(hi, 3), side).0);

    // Miss → evicts the aged, deep entry P (frag 1).
    store(&tt, key(hi, 4), side, 0, false, Bound::Lower, 1, 4, 3);
    assert!(
        !tt.probe(key(hi, 1), side).0,
        "aged deep entry P should be evicted"
    );
    assert!(
        tt.probe(key(hi, 2), side).0,
        "fresh shallow entry Q should survive"
    );
    assert!(tt.probe(key(hi, 3), side).0, "fresh entry R should survive");
}

/// The `tt-entry16` counterpart: the same generation arithmetic over a
/// **2**-entry cluster. `relative_age` and the `− 8·age` weighting are shared
/// code, so what this pins is that the smaller scan still applies them.
#[cfg_attr(miri, ignore)]
#[test]
#[cfg(feature = "tt-entry16")]
fn generation_aging_lowers_replacement_priority() {
    // After three new_search() bumps the table is at generation 3:
    //   P: frag 2, depth 20, gen 0 → depth8 23, age 3, priority 23 − 24 = −1  ← lowest
    //   Q: frag 4, depth  3, gen 3 → depth8  6, age 0, priority  6
    //
    // Without aging P's priority would be 23 — higher than Q's 6, so Q would be
    // the victim. Aging flips the order and the miss evicts P instead.
    //
    // The fragments are spaced by two because key bit 0 is the
    // path-dependence mark at `verbose3`, so consecutive ones would name the
    // same entry there.
    let tt = fresh_tt();
    let side = 0;
    let hi = 200;

    // P written at generation 0.
    assert_eq!(tt.generation(), 0);
    store(&tt, key(hi, 2), side, 0, false, Bound::Lower, 20, 1, 0);

    // Advance to generation 3, then write Q.
    tt.new_search();
    tt.new_search();
    tt.new_search();
    assert_eq!(tt.generation(), 3);
    store(&tt, key(hi, 4), side, 0, false, Bound::Lower, 3, 2, 0);

    // Sanity: both occupy the cluster.
    assert!(tt.probe(key(hi, 2), side).0);
    assert!(tt.probe(key(hi, 4), side).0);

    // Miss → evicts the aged, deep entry P (frag 2), not the shallow fresh Q.
    store(&tt, key(hi, 6), side, 0, false, Bound::Lower, 1, 3, 3);
    assert!(
        !tt.probe(key(hi, 2), side).0,
        "aged deep entry P should be evicted"
    );
    assert!(
        tt.probe(key(hi, 4), side).0,
        "fresh shallow entry Q should survive"
    );
    assert!(tt.probe(key(hi, 6), side).0, "frag 6 should now be present");
}

/// The entry-count half of the `tt-entry16` trade, as behaviour rather than a
/// layout constant.
#[cfg_attr(miri, ignore)]
#[test]
fn a_cluster_holds_exactly_cluster_size_positions() {
    let tt = fresh_tt();
    let side = 0;
    let hi = 555;

    // Fragments are spaced by two: under `verbose3` the wide layout spends key
    // bit 0 on the path-dependence mark, so a family of consecutive fragments
    // would hold two positions that are one and the same entry there.
    let frag = |i: u16| i * 2;

    // Equal depth throughout, so nothing is preferentially retained and the
    // test turns purely on capacity.
    for i in 1..=CLUSTER_ENTRIES as u16 {
        let f = frag(i);
        store(&tt, key(hi, f), side, 0, false, Bound::Lower, 7, f, 0);
    }
    for i in 1..=CLUSTER_ENTRIES as u16 {
        let f = frag(i);
        assert!(
            tt.probe(key(hi, f), side).0,
            "a full cluster must retain all {CLUSTER_ENTRIES} of its entries (frag {f} missing)"
        );
    }

    // One more distinct position than the cluster can hold: it is stored, and
    // exactly one of the previous occupants is gone.
    let extra = frag(CLUSTER_ENTRIES as u16 + 1);
    store(
        &tt,
        key(hi, extra),
        side,
        0,
        false,
        Bound::Lower,
        7,
        extra,
        0,
    );
    assert!(tt.probe(key(hi, extra), side).0, "the new entry is present");
    let survivors = (1..=CLUSTER_ENTRIES as u16)
        .filter(|&i| tt.probe(key(hi, frag(i)), side).0)
        .count();
    assert_eq!(
        survivors,
        CLUSTER_ENTRIES - 1,
        "storing a {}th position must displace exactly one",
        CLUSTER_ENTRIES + 1
    );
}

#[cfg_attr(miri, ignore)]
#[test]
fn save_preserves_move_when_new_move_absent() {
    // The reference keeps the old move when the incoming one is none and the
    // key still matches.
    let tt = fresh_tt();
    let side = 0;
    let k = key(300, 0x55);

    store(&tt, k, side, 1, false, Bound::Lower, 10, 0x0777, 1);
    // Same key, mv = 0, deeper: refreshes value, keeps move.
    store(&tt, k, side, 2, false, Bound::Lower, 12, 0, 2);

    let (found, data, _) = tt.probe(k, side);
    assert!(found);
    assert_eq!(
        data.move16, 0x0777,
        "old move retained when new move is absent"
    );
    assert_eq!(data.value, 2, "value refreshed");
    assert_eq!(data.depth, 12, "depth refreshed");
}

/// The table's size and placement, as a caller outside the crate sees them:
/// the reference's `clusterCount = mb · 1024 · 1024 / sizeof(Cluster = 32)`,
/// on a huge-page boundary, with nothing able to move either.
#[cfg_attr(miri, ignore)]
#[test]
fn the_table_is_sized_and_aligned_before_anything_runs() {
    let tt = fresh_tt();
    let (addr, bytes) = tt.backing_region();

    assert_eq!(
        CLUSTER_COUNT % 32_768,
        0,
        "a whole number of MiB of clusters"
    );
    assert_eq!(addr % TT_ALIGN, 0, "the table starts on a huge page");
    assert_eq!(bytes % TT_ALIGN, 0, "and covers whole ones");
    assert!(bytes >= CLUSTER_COUNT * 32);
}

#[cfg_attr(miri, ignore)]
#[test]
fn clear_zeroes_entries_and_generation() {
    let tt = fresh_tt();
    tt.new_search();
    let k = key(11, 0x22);
    store(&tt, k, 0, 5, false, Bound::Exact, 9, 0x22, 5);
    assert!(tt.probe(k, 0).0);

    tt.clear();
    assert_eq!(tt.generation(), 0, "clear resets generation");
    assert!(!tt.probe(k, 0).0, "clear empties every entry");
}

#[cfg_attr(miri, ignore)]
#[test]
fn new_search_wraps_within_five_bits() {
    let tt = fresh_tt();
    // 32 bumps wrap 0 → 0 (generation is 5 bits: 0..=31).
    for _ in 0..31 {
        tt.new_search();
    }
    assert_eq!(tt.generation(), 31);
    tt.new_search();
    assert_eq!(tt.generation(), 0, "generation wraps at 2^5");
}

/// Compares whole tables byte for byte through `checksum`, so it exists only in
/// a build that has one.
#[cfg(feature = "verbose3")]
#[cfg_attr(miri, ignore)]
#[test]
fn determinism_identical_sequences_yield_identical_tables() {
    fn run(tt: &TranspositionTable) -> u64 {
        tt.clear();
        for round in 0..4 {
            tt.new_search();
            for f in 1..=6u16 {
                let k = key((f as u64) * 3, f.wrapping_mul(37).wrapping_add(1));
                store(
                    tt,
                    k,
                    (f & 1) as u8,
                    (round * 100 + f as i32) - 250,
                    f % 2 == 0,
                    Bound::Lower,
                    (f as i32) + round,
                    f,
                    round * 10,
                );
            }
        }
        tt.checksum()
    }

    // The same sequence twice on the one table, each from the emptied state.
    let tt = fresh_tt();
    assert_eq!(run(&tt), run(&tt));
}

#[cfg_attr(miri, ignore)]
#[test]
fn a_cleared_table_reads_back_all_misses() {
    let tt = fresh_tt();
    for hi in 0..2048u64 {
        for side in 0..2u8 {
            // A fragment that is nonzero *above bit 0* cannot match a zeroed
            // entry's `key == 0`, so the probe takes the true miss path. Bit 0
            // is excluded because the `verbose3` wide layout stores the
            // path-dependence mark there rather than the hash.
            let k = key(hi, (hi as u16).wrapping_mul(7) | 2);
            let (found, data, _) = tt.probe(k, side);
            assert!(!found, "cleared table entry occupied at hi={hi}");
            assert_eq!(data, miss_sentinel());
        }
    }
}

#[cfg_attr(miri, ignore)]
#[test]
fn many_clear_and_store_cycles_preserve_semantics() {
    // A game after a game after a game: each `usinewgame` empties the table and
    // the next search fills it again.
    let tt = fresh_tt();
    for i in 0..40u64 {
        tt.clear();
        let k = key(i, i as u16);
        store(&tt, k, 0, 3, false, Bound::Exact, 5, i as u16, 3);
        assert!(tt.probe(k, 0).0);
        if i > 0 {
            let previous = key(i - 1, (i - 1) as u16);
            assert!(!tt.probe(previous, 0).0, "the clear emptied the last game");
        }
    }
}

/// A diagnostic, not an assertion: report the table's `AnonHugePages` figure
/// from `/proc/self/smaps`. Never fails, since transparent huge pages may be
/// disabled — and the `static` carries no hint of its own until a session's
/// `isready` advises one, so a bare test run is expected to read zero.
#[cfg_attr(miri, ignore)]
#[test]
#[cfg(target_os = "linux")]
fn thp_uptake_diagnostic() {
    use std::fs;

    let tt = fresh_tt();
    let (base, _bytes) = tt.backing_region();
    assert_ne!(base, 0);

    let smaps = match fs::read_to_string("/proc/self/smaps") {
        Ok(s) => s,
        Err(e) => {
            eprintln!("THP diagnostic: /proc/self/smaps unreadable ({e}); skipping");
            return;
        }
    };

    // smaps is a sequence of blocks, each headed by `start-end perms ...`.
    fn header_contains(line: &str, base: u64) -> Option<bool> {
        let range = line
            .split_once(' ')
            .map(|(r, _)| r)
            .filter(|r| r.contains('-'))?;
        let (start, end) = range.split_once('-')?;
        let s = u64::from_str_radix(start, 16).ok()?;
        let e = u64::from_str_radix(end, 16).ok()?;
        Some((s..e).contains(&base))
    }

    let mut in_region = false;
    let mut anon_huge_kb: Option<u64> = None;
    for line in smaps.lines() {
        if let Some(contains) = header_contains(line, base as u64) {
            in_region = contains;
        } else if in_region && let Some(rest) = line.strip_prefix("AnonHugePages:") {
            anon_huge_kb = rest
                .trim()
                .strip_suffix(" kB")
                .and_then(|n| n.trim().parse::<u64>().ok());
            break;
        }
    }

    match anon_huge_kb {
        Some(kb) => eprintln!(
            "THP diagnostic: TT region at {base:#x} backed by {kb} kB AnonHugePages \
             (0 means THP disabled or not yet faulted in)"
        ),
        None => eprintln!(
            "THP diagnostic: no AnonHugePages line found for TT region at {base:#x} \
             (kernel without THP accounting); skipping"
        ),
    }
    // No assertion: huge-page availability is environmental.
}

/// The `TTData` a miss returns.
fn miss_sentinel() -> TTData {
    TTData {
        move16: 0,
        value: yorkie_storage::VALUE_NONE,
        eval: yorkie_storage::VALUE_NONE,
        depth: DEPTH_NONE,
        bound: Bound::None,
        is_pv: false,
        #[cfg(feature = "verbose3")]
        path_dep: false,
    }
}

/// The generation a freshly-resized table starts at.
fn tt_generation_zero() -> u8 {
    0
}

// The two modules below assert the same scenarios against the two layouts, so
// a regression in either direction is a red test rather than a still-green
// suite. Only the aliasing verdict differs: where the wide layout keeps two
// entries apart, the narrow one merges them.

/// Identity under `tt-entry16`: the entry stores the whole 64-bit key, so a hit
/// is exact and no two distinct positions can be mistaken for one another.
#[cfg(feature = "tt-entry16")]
mod wide_key_identity {
    use super::*;

    /// Keys sharing their low 16 bits but landing in **different** clusters:
    /// neither can be reached by probing the other.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn equal_low_16_bits_in_different_clusters_do_not_alias() {
        let tt = fresh_tt();
        let side = 0;

        let a = key(10, 0xBEEF);
        let b = key(20, 0xBEEF);
        assert_eq!(a as u16, b as u16, "the pair shares its low 16 bits");

        store(&tt, a, side, 111, false, Bound::Exact, 9, 0x11, 111);
        assert!(!tt.probe(b, side).0, "b must not read a's entry");

        store(&tt, b, side, 222, false, Bound::Exact, 9, 0x22, 222);
        assert_eq!(tt.probe(a, side).1.value, 111, "a keeps its own payload");
        assert_eq!(tt.probe(b, side).1.value, 222, "b keeps its own payload");
    }

    /// Keys sharing their low 16 bits **and** their cluster, differing only in
    /// the middle bits neither the index nor a 16-bit fragment reads.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn same_cluster_and_low_16_bits_do_not_alias() {
        let tt = fresh_tt();
        let side = 0;

        let a = key_mid(300, 0, 0xABCD);
        let b = key_mid(300, 1, 0xABCD);
        assert_ne!(a, b);
        assert_eq!(a as u16, b as u16, "the pair shares its low 16 bits");
        assert_eq!(a >> CLUSTER_SHIFT, b >> CLUSTER_SHIFT, "and its cluster");

        store(&tt, a, side, 111, false, Bound::Exact, 9, 0x11, 111);
        assert!(
            !tt.probe(b, side).0,
            "a 64-bit key must not be matched by a sibling that differs above bit 16"
        );

        store(&tt, b, side, 222, true, Bound::Lower, 12, 0x22, 222);

        let (found_a, data_a, _) = tt.probe(a, side);
        assert!(found_a, "a survives — b took the cluster's other entry");
        assert_eq!(data_a.value, 111);
        assert_eq!(data_a.move16, 0x11);
        assert_eq!(data_a.depth, 9);

        let (found_b, data_b, _) = tt.probe(b, side);
        assert!(found_b);
        assert_eq!(data_b.value, 222);
        assert_eq!(data_b.move16, 0x22);
        assert_eq!(data_b.depth, 12);
    }

    /// Every middle bit is load-bearing, one at a time.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn every_middle_bit_participates_in_identity() {
        let tt = fresh_tt();
        let side = 0;
        let hi = 4000;

        for bit in 16..CLUSTER_SHIFT {
            // Earlier iterations may still occupy the cluster's other entry,
            // but every key here has a zero middle part while every sibling has
            // a nonzero one, so no leftover can be mistaken for a sibling.
            let base = key_mid(hi, 0, (bit as u16).wrapping_mul(7) | 1);
            let sibling = base | (1u64 << bit);
            assert_ne!(base, sibling);
            assert_eq!(base as u16, sibling as u16);
            assert_eq!(base >> CLUSTER_SHIFT, sibling >> CLUSTER_SHIFT);

            store(&tt, base, side, 5, false, Bound::Exact, 9, 0x33, 5);
            assert!(
                !tt.probe(sibling, side).0,
                "key bit {bit} must take part in the identity check"
            );
        }
    }
}

/// Identity in the default layout, where a hit is a 16-bit match and a
/// same-cluster sibling *does* alias. This is the reference's behaviour, and
/// the reason the search validates every TT move against the position.
#[cfg(not(feature = "tt-entry16"))]
mod narrow_key_identity {
    use super::*;

    /// Keys sharing their low 16 bits but landing in different clusters still
    /// do not alias: this half of identity is layout-independent.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn equal_low_16_bits_in_different_clusters_do_not_alias() {
        let tt = fresh_tt();
        let side = 0;

        let a = key(10, 0xBEEF);
        let b = key(20, 0xBEEF);
        assert_eq!(a as u16, b as u16, "the pair shares its low 16 bits");

        store(&tt, a, side, 111, false, Bound::Exact, 9, 0x11, 111);
        assert!(!tt.probe(b, side).0, "b must not read a's entry");

        store(&tt, b, side, 222, false, Bound::Exact, 9, 0x22, 222);
        assert_eq!(tt.probe(a, side).1.value, 111, "a keeps its own payload");
        assert_eq!(tt.probe(b, side).1.value, 222, "b keeps its own payload");
    }

    /// Keys sharing their cluster and their low 16 bits **do** alias here —
    /// the ~1/65536-per-entry false hit the reference lives with.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn same_cluster_and_low_16_bits_alias() {
        let tt = fresh_tt();
        let side = 0;

        let a = key_mid(300, 0, 0xABCD);
        let b = key_mid(300, 1, 0xABCD);
        assert_ne!(a, b);

        store(&tt, a, side, 111, false, Bound::Exact, 9, 0x11, 111);

        let (found_b, data_b, _) = tt.probe(b, side);
        assert!(found_b, "a 16-bit key cannot tell b from a");
        assert_eq!(data_b.value, 111, "b reads a's payload");

        // Writing through b lands in the same entry, so a now reads b's data.
        store(&tt, b, side, 222, true, Bound::Lower, 12, 0x22, 222);
        let (found_a, data_a, _) = tt.probe(a, side);
        assert!(found_a);
        assert_eq!(data_a.value, 222, "the two share one entry");
        assert_eq!(data_a.move16, 0x22);
    }
}

/// The per-entry path-dependence mark, which exists only at `verbose3`. The
/// search's rule for *when* an entry is marked is the Search layer's; what is
/// pinned here is that a mark written for one entry comes back for that entry
/// and no other, follows the payload through the replacement policy, and is
/// part of what [`TranspositionTable::checksum`] summarises.
#[cfg(feature = "verbose3")]
mod path_dependence_mark {
    use super::*;

    /// [`store`], with the mark it leaves unmarked spelled out.
    #[allow(clippy::too_many_arguments)]
    fn store_marked(
        tt: &TranspositionTable,
        k: u64,
        side: u8,
        value: i32,
        pv: bool,
        bound: Bound,
        depth: i32,
        mv: u16,
        eval: i32,
        path_dep: bool,
    ) {
        let generation = tt.generation();
        let (_, _, w) = tt.probe(k, side);
        w.write(k, value, pv, bound, depth, mv, eval, generation, path_dep);
    }

    /// `CLUSTER_ENTRIES` keys that share one cluster, spaced so no two of them
    /// differ only in the key bit the wide layout spends on the mark.
    fn cluster_family(hi: u64) -> Vec<u64> {
        (0..CLUSTER_ENTRIES)
            .map(|i| key(hi, 0x10 + (i as u16) * 0x10))
            .collect()
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn a_stored_mark_reads_back_and_a_rewrite_replaces_it() {
        let tt = fresh_tt();
        let side = 0;
        let k = key(64, 0x0102);

        store_marked(&tt, k, side, 10, false, Bound::Exact, 8, 0x11, 0, true);
        assert!(tt.probe(k, side).1.path_dep);

        // The same slot, rewritten unmarked: the mark follows the payload
        // rather than accumulating.
        store_marked(&tt, k, side, 20, false, Bound::Exact, 8, 0x11, 0, false);
        let (found, data, _) = tt.probe(k, side);
        assert!(found);
        assert_eq!(data.value, 20);
        assert!(!data.path_dep);
    }

    /// A miss carries no mark, whatever the entry it would replace holds.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn the_miss_sentinel_is_unmarked() {
        let tt = fresh_tt();
        let side = 0;

        store_marked(
            &tt,
            key(70, 0x0001),
            side,
            10,
            false,
            Bound::Exact,
            8,
            0x11,
            0,
            true,
        );
        let (found, data, _) = tt.probe(key(70, 0x0002), side);
        assert!(!found);
        assert_eq!(data, miss_sentinel());
    }

    /// Every entry of one cluster carries its own mark. In the default layout
    /// they share a single word, so this is what says the per-slot writes do
    /// not clobber each other.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn entries_sharing_a_cluster_keep_their_own_marks() {
        let tt = fresh_tt();
        let side = 1;
        let family = cluster_family(88);

        // Alternate the marks, so neither an all-set nor an all-clear word
        // would pass.
        for (i, &k) in family.iter().enumerate() {
            store_marked(
                &tt,
                k,
                side,
                i as i32,
                false,
                Bound::Exact,
                8,
                0x20 + i as u16,
                0,
                i % 2 == 0,
            );
        }
        for (i, &k) in family.iter().enumerate() {
            let (found, data, _) = tt.probe(k, side);
            assert!(found, "entry {i} must still be there");
            assert_eq!(data.value, i as i32);
            assert_eq!(data.path_dep, i % 2 == 0, "entry {i} carries its own mark");
        }

        // Rewriting one entry's mark leaves its neighbours' alone.
        store_marked(
            &tt,
            family[0],
            side,
            0,
            false,
            Bound::Exact,
            8,
            0x20,
            0,
            false,
        );
        for (i, &k) in family.iter().enumerate() {
            let expected = i != 0 && i % 2 == 0;
            assert_eq!(tt.probe(k, side).1.path_dep, expected, "entry {i}");
        }
    }

    /// A write the replacement policy declines leaves the mark as it is, like
    /// every other field of the entry it kept.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn a_declined_write_leaves_the_mark_alone() {
        let tt = fresh_tt();
        let side = 0;
        let k = key(96, 0x0303);

        store_marked(&tt, k, side, 10, true, Bound::Exact, 40, 0x11, 0, true);
        // Shallow, non-exact, same position and generation: declined.
        store_marked(&tt, k, side, 99, false, Bound::Lower, 1, 0x22, 0, false);

        let (found, data, _) = tt.probe(k, side);
        assert!(found);
        assert_eq!(data.value, 10, "the declined write kept the deep entry");
        assert!(data.path_dep, "and with it the mark");
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn clear_drops_the_marks_with_the_entries() {
        let tt = fresh_tt();
        let side = 0;
        let empty = tt.checksum();

        for &k in &cluster_family(112) {
            store_marked(&tt, k, side, 1, false, Bound::Exact, 8, 0x11, 0, true);
        }
        tt.clear();
        assert_eq!(
            tt.checksum(),
            empty,
            "a cleared table must be indistinguishable from a fresh one"
        );
    }

    /// The checksum summarises everything the table stores, the mark included,
    /// so two runs that differ in nothing else still differ in it.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn the_checksum_covers_the_mark() {
        let checksum_with = |path_dep: bool| {
            let tt = fresh_tt();
            store_marked(
                &tt,
                key(128, 0x0404),
                0,
                7,
                false,
                Bound::Exact,
                8,
                0x11,
                0,
                path_dep,
            );
            tt.checksum()
        };
        assert_eq!(checksum_with(false), checksum_with(false));
        assert_ne!(checksum_with(false), checksum_with(true));
    }

    /// Under `tt-entry16` the mark is bit 0 of the stored key, so identity is
    /// the remaining 63 bits: two keys differing only there share an entry.
    /// That is the whole price of putting it in the key.
    #[cfg(feature = "tt-entry16")]
    #[cfg_attr(miri, ignore)]
    #[test]
    fn the_wide_layout_spends_key_bit_zero_on_the_mark() {
        let tt = fresh_tt();
        let side = 0;
        let a = key(144, 0x0500);
        let b = a | 1;

        store_marked(&tt, a, side, 11, false, Bound::Exact, 8, 0x11, 0, true);
        let (found, data, _) = tt.probe(b, side);
        assert!(found, "bit 0 is the mark, not part of the identity");
        assert_eq!(data.value, 11);
        assert!(data.path_dep);
    }
}
