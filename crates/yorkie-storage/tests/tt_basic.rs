//! The transposition table, against the semantics of `tt.cpp`.
//!
//! # The addressing model these tests use
//!
//! With `resize(1)` the table holds `2¹⁵` clusters, so
//!
//! ```text
//! cluster_index_pre_side = mul_hi64(key, 32768) = key >> 49
//! in_cluster_key_frag    = key & 0xffff
//! ```
//!
//! The top 15 bits select the cluster and the low 16 are the stored key
//! fragment — disjoint ranges, so [`key`] sets each independently. The side to
//! move is OR-ed into cluster-index bit 0, so fixing `hi` **and** `side` keeps
//! a family of keys in one cluster while the fragment varies.
//!
//! Bits `16..49` are read by neither, which is why [`key_mid`] exists: a key
//! varying only there is indistinguishable from its sibling to the default
//! build and distinguishable to a `tt-entry16` one. That is the whole
//! observable difference between the two layouts.
//!
//! A cluster holds 3 entries by default and 2 under `tt-entry16`, so the two
//! tests whose walk-through only reads correctly for one count carry a
//! `cfg`-selected pair of bodies rather than skipping under the other.
//!
//! Every test here is ignored under miri: the smallest table they can address
//! is one MiB of atomics, and miri interprets every zeroing write and probe of
//! it — tens of minutes per test. The table's unsafe surface is covered there
//! by the crate's unit tests, at sizes miri can finish.

use yorkie_storage::{Bound, DEPTH_NONE, TTData, TranspositionTable};

/// Entries per 32-byte cluster, restating the crate-private `CLUSTER_SIZE`.
const CLUSTER_ENTRIES: usize = if cfg!(feature = "tt-entry16") { 2 } else { 3 };

/// Build a key that lands in cluster `hi` (before the side fold) with
/// in-cluster fragment `frag`. Requires `hi < 2¹⁵`.
fn key(hi: u64, frag: u16) -> u64 {
    key_mid(hi, 0, frag)
}

/// [`key`] with the otherwise-unused middle bits set to `mid`. Requires
/// `hi < 2¹⁵` and `mid < 2³³`.
fn key_mid(hi: u64, mid: u64, frag: u16) -> u64 {
    assert!(hi < (1 << 15));
    assert!(mid < (1 << 33));
    (hi << 49) | (mid << 16) | frag as u64
}

/// Probe `k` and store through the returned writer, at the table's current
/// generation.
#[allow(clippy::too_many_arguments)]
fn store(
    tt: &mut TranspositionTable,
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
    w.write(k, value, pv, bound, depth, mv, eval, generation);
}

#[cfg_attr(miri, ignore)]
#[test]
fn store_probe_round_trip_every_field() {
    let mut tt = TranspositionTable::new();
    tt.resize(1);

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
    let mut tt = TranspositionTable::new();
    tt.resize(1);
    let side = 1;

    for (i, bound) in [Bound::None, Bound::Upper, Bound::Lower, Bound::Exact]
        .into_iter()
        .enumerate()
    {
        for pv in [false, true] {
            // Distinct fragment per case so they don't overwrite each other.
            let frag = 0x100 + (i as u16) * 2 + pv as u16;
            let k = key(7, frag);
            store(&mut tt, k, side, 10 + i as i32, pv, bound, 5, frag, -10);
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
    let mut tt = TranspositionTable::new();
    tt.resize(1);
    let side = 0;

    let stored = key(42, 0xBEEF);
    store(
        &mut tt,
        stored,
        side,
        100,
        false,
        Bound::Exact,
        8,
        0x0111,
        100,
    );

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
    let mut tt = TranspositionTable::new();
    tt.resize(1);
    let side = 0;
    let hi = 100;

    store(&mut tt, key(hi, 1), side, 0, false, Bound::Lower, 10, 1, 0);
    store(&mut tt, key(hi, 2), side, 0, false, Bound::Lower, 5, 2, 0);
    store(&mut tt, key(hi, 3), side, 0, false, Bound::Lower, 20, 3, 0);

    // All three present before the eviction.
    assert!(tt.probe(key(hi, 1), side).0);
    assert!(tt.probe(key(hi, 2), side).0);
    assert!(tt.probe(key(hi, 3), side).0);

    // Miss on frag 4 → writer targets the evicted slot; write frag 4 there.
    store(&mut tt, key(hi, 4), side, 0, false, Bound::Lower, 1, 4, 0);

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
    //   slot 0: frag 1, depth 10 → depth8 13, priority 13
    //   slot 1: frag 2, depth  5 → depth8  8, priority  8   ← lowest
    //
    // The cluster is now full, so a miss replaces slot 1 (frag 2).
    let mut tt = TranspositionTable::new();
    tt.resize(1);
    let side = 0;
    let hi = 100;

    store(&mut tt, key(hi, 1), side, 0, false, Bound::Lower, 10, 1, 0);
    store(&mut tt, key(hi, 2), side, 0, false, Bound::Lower, 5, 2, 0);

    // Both present before the eviction.
    assert!(tt.probe(key(hi, 1), side).0);
    assert!(tt.probe(key(hi, 2), side).0);

    // Miss on frag 3 → writer targets the evicted slot; write frag 3 there.
    store(&mut tt, key(hi, 3), side, 0, false, Bound::Lower, 1, 3, 0);

    assert!(
        !tt.probe(key(hi, 2), side).0,
        "frag 2 (depth 5) should have been evicted"
    );
    assert!(tt.probe(key(hi, 1), side).0, "frag 1 should survive");
    assert!(tt.probe(key(hi, 3), side).0, "frag 3 should now be present");
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
    let mut tt = TranspositionTable::new();
    tt.resize(1);
    let side = 0;
    let hi = 200;

    // P written at generation 0.
    assert_eq!(tt.generation(), 0);
    store(&mut tt, key(hi, 1), side, 0, false, Bound::Lower, 20, 1, 0);

    // Advance to generation 3, then write Q and R.
    tt.new_search();
    tt.new_search();
    tt.new_search();
    assert_eq!(tt.generation(), 3);
    store(&mut tt, key(hi, 2), side, 0, false, Bound::Lower, 3, 2, 0);
    store(&mut tt, key(hi, 3), side, 0, false, Bound::Lower, 8, 3, 0);

    // Sanity: all three occupy the cluster.
    assert!(tt.probe(key(hi, 1), side).0);
    assert!(tt.probe(key(hi, 2), side).0);
    assert!(tt.probe(key(hi, 3), side).0);

    // Miss → evicts the aged, deep entry P (frag 1).
    store(&mut tt, key(hi, 4), side, 0, false, Bound::Lower, 1, 4, 3);
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
    //   P: frag 1, depth 20, gen 0 → depth8 23, age 3, priority 23 − 24 = −1  ← lowest
    //   Q: frag 2, depth  3, gen 3 → depth8  6, age 0, priority  6
    //
    // Without aging P's priority would be 23 — higher than Q's 6, so Q would be
    // the victim. Aging flips the order and the miss evicts P instead.
    let mut tt = TranspositionTable::new();
    tt.resize(1);
    let side = 0;
    let hi = 200;

    // P written at generation 0.
    assert_eq!(tt.generation(), 0);
    store(&mut tt, key(hi, 1), side, 0, false, Bound::Lower, 20, 1, 0);

    // Advance to generation 3, then write Q.
    tt.new_search();
    tt.new_search();
    tt.new_search();
    assert_eq!(tt.generation(), 3);
    store(&mut tt, key(hi, 2), side, 0, false, Bound::Lower, 3, 2, 0);

    // Sanity: both occupy the cluster.
    assert!(tt.probe(key(hi, 1), side).0);
    assert!(tt.probe(key(hi, 2), side).0);

    // Miss → evicts the aged, deep entry P (frag 1), not the shallow fresh Q.
    store(&mut tt, key(hi, 3), side, 0, false, Bound::Lower, 1, 3, 3);
    assert!(
        !tt.probe(key(hi, 1), side).0,
        "aged deep entry P should be evicted"
    );
    assert!(
        tt.probe(key(hi, 2), side).0,
        "fresh shallow entry Q should survive"
    );
    assert!(tt.probe(key(hi, 3), side).0, "frag 3 should now be present");
}

/// The entry-count half of the `tt-entry16` trade, as behaviour rather than a
/// layout constant.
#[cfg_attr(miri, ignore)]
#[test]
fn a_cluster_holds_exactly_cluster_size_positions() {
    let mut tt = TranspositionTable::new();
    tt.resize(1);
    let side = 0;
    let hi = 555;

    // Equal depth throughout, so nothing is preferentially retained and the
    // test turns purely on capacity.
    for f in 1..=CLUSTER_ENTRIES as u16 {
        store(&mut tt, key(hi, f), side, 0, false, Bound::Lower, 7, f, 0);
    }
    for f in 1..=CLUSTER_ENTRIES as u16 {
        assert!(
            tt.probe(key(hi, f), side).0,
            "a full cluster must retain all {CLUSTER_ENTRIES} of its entries (frag {f} missing)"
        );
    }

    // One more distinct position than the cluster can hold: it is stored, and
    // exactly one of the previous occupants is gone.
    let extra = CLUSTER_ENTRIES as u16 + 1;
    store(
        &mut tt,
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
        .filter(|&f| tt.probe(key(hi, f), side).0)
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
    let mut tt = TranspositionTable::new();
    tt.resize(1);
    let side = 0;
    let k = key(300, 0x55);

    store(&mut tt, k, side, 1, false, Bound::Lower, 10, 0x0777, 1);
    // Same key, mv = 0, deeper: refreshes value, keeps move.
    store(&mut tt, k, side, 2, false, Bound::Lower, 12, 0, 2);

    let (found, data, _) = tt.probe(k, side);
    assert!(found);
    assert_eq!(
        data.move16, 0x0777,
        "old move retained when new move is absent"
    );
    assert_eq!(data.value, 2, "value refreshed");
    assert_eq!(data.depth, 12, "depth refreshed");
}

#[cfg_attr(miri, ignore)]
#[test]
fn resize_sizes_by_formula_and_clears() {
    let mut tt = TranspositionTable::new();
    assert_eq!(tt.cluster_count(), 0, "fresh table is empty");

    // clusterCount = mb · 1024 · 1024 / sizeof(Cluster=32) = mb · 32768.
    tt.resize(1);
    assert_eq!(tt.cluster_count(), 32_768);
    tt.resize(4);
    assert_eq!(tt.cluster_count(), 4 * 32_768);

    // Store, then a *different* size reallocates and clears.
    let k = key(1, 0x1);
    store(&mut tt, k, 0, 5, false, Bound::Exact, 9, 0x1, 5);
    assert!(tt.probe(k, 0).0);

    tt.resize(2);
    assert_eq!(tt.cluster_count(), 2 * 32_768);
    assert!(!tt.probe(k, 0).0, "resize to a new size clears the table");

    // Re-store works after resize.
    store(&mut tt, k, 0, 7, false, Bound::Exact, 9, 0x1, 7);
    assert_eq!(tt.probe(k, 0).1.value, 7);
}

#[cfg_attr(miri, ignore)]
#[test]
fn resize_to_same_size_is_a_no_op() {
    let mut tt = TranspositionTable::new();
    tt.resize(1);
    let k = key(5, 0x9);
    store(&mut tt, k, 0, 5, false, Bound::Exact, 9, 0x9, 5);
    let before = tt.checksum();

    // The same MiB yields the same cluster count, so the reference's early
    // return leaves the table untouched.
    tt.resize(1);
    assert_eq!(tt.cluster_count(), 32_768);
    assert_eq!(
        tt.checksum(),
        before,
        "same-size resize leaves the table intact"
    );
    assert!(tt.probe(k, 0).0);
}

#[cfg_attr(miri, ignore)]
#[test]
fn clear_zeroes_entries_and_generation() {
    let mut tt = TranspositionTable::new();
    tt.resize(1);
    tt.new_search();
    let k = key(11, 0x22);
    store(&mut tt, k, 0, 5, false, Bound::Exact, 9, 0x22, 5);
    assert!(tt.probe(k, 0).0);

    tt.clear();
    assert_eq!(tt.generation(), 0, "clear resets generation");
    assert!(!tt.probe(k, 0).0, "clear empties every entry");
}

#[cfg_attr(miri, ignore)]
#[test]
fn new_search_wraps_within_five_bits() {
    let mut tt = TranspositionTable::new();
    tt.resize(1);
    // 32 bumps wrap 0 → 0 (generation is 5 bits: 0..=31).
    for _ in 0..31 {
        tt.new_search();
    }
    assert_eq!(tt.generation(), 31);
    tt.new_search();
    assert_eq!(tt.generation(), 0, "generation wraps at 2^5");
}

#[cfg_attr(miri, ignore)]
#[test]
fn determinism_identical_sequences_yield_identical_tables() {
    fn run() -> TranspositionTable {
        let mut tt = TranspositionTable::new();
        tt.resize(1);
        for round in 0..4 {
            tt.new_search();
            for f in 1..=6u16 {
                let k = key((f as u64) * 3, f.wrapping_mul(37).wrapping_add(1));
                store(
                    &mut tt,
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
        tt
    }

    assert_eq!(run().checksum(), run().checksum());
}

/// The private `TT_ALLOC_ALIGN`, restated.
const EXPECTED_TT_ALIGN: usize = if cfg!(target_os = "linux") {
    2 * 1024 * 1024
} else {
    4096
};

#[cfg_attr(miri, ignore)]
#[test]
fn resized_table_base_pointer_is_page_aligned() {
    let mut tt = TranspositionTable::new();
    assert_eq!(tt.backing_ptr_addr(), 0, "unsized table reports no address");

    for &mb in &[1usize, 2, 8, 64] {
        tt.resize(mb);
        let addr = tt.backing_ptr_addr();
        assert_ne!(addr, 0);
        assert_eq!(
            addr % EXPECTED_TT_ALIGN,
            0,
            "TT base for {mb} MiB not {EXPECTED_TT_ALIGN}-aligned",
        );
    }
}

#[cfg_attr(miri, ignore)]
#[test]
fn fresh_resize_reads_back_all_misses() {
    // The allocation is zeroed, so a freshly resized table is unoccupied
    // without a separate clear.
    let mut tt = TranspositionTable::new();
    tt.resize(2);
    for hi in 0..2048u64 {
        for side in 0..2u8 {
            // A nonzero fragment cannot match a zeroed entry's `key == 0`, so
            // the probe takes the true miss path.
            let k = key(hi & 0x7fff, (hi as u16).wrapping_mul(7) | 1);
            let (found, data, _) = tt.probe(k, side);
            assert!(!found, "fresh table entry occupied at hi={hi}");
            assert_eq!(data, miss_sentinel());
        }
    }
}

#[cfg_attr(miri, ignore)]
#[test]
fn resize_grow_shrink_same_cycles_preserve_semantics() {
    // Grow, shrink and repeat a size, across several of them.
    let mut tt = TranspositionTable::new();
    let k = key(3, 0xabcd);

    for &mb in &[1usize, 4, 2, 8, 8, 1, 1, 16] {
        let prev_count = tt.cluster_count();
        let prev_sum = tt.checksum();
        tt.resize(mb);
        assert_eq!(tt.cluster_count(), mb * 32_768);

        if tt.cluster_count() == prev_count {
            // Same size → untouched (no realloc, no clear).
            assert_eq!(tt.checksum(), prev_sum, "same-size resize must be a no-op");
        } else {
            // Changed size → aligned, cleared allocation.
            assert_eq!(tt.backing_ptr_addr() % EXPECTED_TT_ALIGN, 0);
            assert!(!tt.probe(k, 0).0, "grow/shrink clears the table");
            // Store survives until the next size change.
            store(&mut tt, k, 0, 42, false, Bound::Exact, 9, 0xabcd, 42);
            assert_eq!(tt.probe(k, 0).1.value, 42);
        }
    }
}

#[cfg_attr(miri, ignore)]
#[test]
fn many_resizes_do_not_leak_or_corrupt() {
    // A double free or mismatched layout would trip the allocator here.
    let mut tt = TranspositionTable::new();
    for i in 0..40u64 {
        let mb = 1 + (i % 4) as usize; // cycles 1,2,3,4 → forces real reallocs
        tt.resize(mb);
        let k = key(i & 0x7fff, i as u16);
        store(&mut tt, k, 0, 3, false, Bound::Exact, 5, i as u16, 3);
        assert!(tt.probe(k, 0).0);
    }
}

/// A diagnostic, not an assertion: report the TT mapping's `AnonHugePages` figure
/// from `/proc/self/smaps`. Never fails, since transparent huge pages may be
/// disabled.
#[cfg_attr(miri, ignore)]
#[test]
#[cfg(target_os = "linux")]
fn thp_uptake_diagnostic_over_64mib() {
    use std::fs;

    let mut tt = TranspositionTable::new();
    tt.resize(64); // large enough that the huge-page path applies
    let base = tt.backing_ptr_addr();
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
        let mut tt = TranspositionTable::new();
        tt.resize(1);
        let side = 0;

        let a = key(10, 0xBEEF);
        let b = key(20, 0xBEEF);
        assert_eq!(a as u16, b as u16, "the pair shares its low 16 bits");

        store(&mut tt, a, side, 111, false, Bound::Exact, 9, 0x11, 111);
        assert!(!tt.probe(b, side).0, "b must not read a's entry");

        store(&mut tt, b, side, 222, false, Bound::Exact, 9, 0x22, 222);
        assert_eq!(tt.probe(a, side).1.value, 111, "a keeps its own payload");
        assert_eq!(tt.probe(b, side).1.value, 222, "b keeps its own payload");
    }

    /// Keys sharing their low 16 bits **and** their cluster, differing only in
    /// the middle bits neither the index nor a 16-bit fragment reads.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn same_cluster_and_low_16_bits_do_not_alias() {
        let mut tt = TranspositionTable::new();
        tt.resize(1);
        let side = 0;

        let a = key_mid(300, 0, 0xABCD);
        let b = key_mid(300, 1, 0xABCD);
        assert_ne!(a, b);
        assert_eq!(a as u16, b as u16, "the pair shares its low 16 bits");
        assert_eq!(a >> 49, b >> 49, "and its cluster");

        store(&mut tt, a, side, 111, false, Bound::Exact, 9, 0x11, 111);
        assert!(
            !tt.probe(b, side).0,
            "a 64-bit key must not be matched by a sibling that differs above bit 16"
        );

        store(&mut tt, b, side, 222, true, Bound::Lower, 12, 0x22, 222);

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
        let mut tt = TranspositionTable::new();
        tt.resize(1);
        let side = 0;
        let hi = 4000;

        for bit in 16..49u32 {
            // Earlier iterations may still occupy the cluster's other entry,
            // but every key here has a zero middle part while every sibling has
            // a nonzero one, so no leftover can be mistaken for a sibling.
            let base = key_mid(hi, 0, (bit as u16).wrapping_mul(7) | 1);
            let sibling = base | (1u64 << bit);
            assert_ne!(base, sibling);
            assert_eq!(base as u16, sibling as u16);
            assert_eq!(base >> 49, sibling >> 49);

            store(&mut tt, base, side, 5, false, Bound::Exact, 9, 0x33, 5);
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
        let mut tt = TranspositionTable::new();
        tt.resize(1);
        let side = 0;

        let a = key(10, 0xBEEF);
        let b = key(20, 0xBEEF);
        assert_eq!(a as u16, b as u16, "the pair shares its low 16 bits");

        store(&mut tt, a, side, 111, false, Bound::Exact, 9, 0x11, 111);
        assert!(!tt.probe(b, side).0, "b must not read a's entry");

        store(&mut tt, b, side, 222, false, Bound::Exact, 9, 0x22, 222);
        assert_eq!(tt.probe(a, side).1.value, 111, "a keeps its own payload");
        assert_eq!(tt.probe(b, side).1.value, 222, "b keeps its own payload");
    }

    /// Keys sharing their cluster and their low 16 bits **do** alias here —
    /// the ~1/65536-per-entry false hit the reference lives with.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn same_cluster_and_low_16_bits_alias() {
        let mut tt = TranspositionTable::new();
        tt.resize(1);
        let side = 0;

        let a = key_mid(300, 0, 0xABCD);
        let b = key_mid(300, 1, 0xABCD);
        assert_ne!(a, b);

        store(&mut tt, a, side, 111, false, Bound::Exact, 9, 0x11, 111);

        let (found_b, data_b, _) = tt.probe(b, side);
        assert!(found_b, "a 16-bit key cannot tell b from a");
        assert_eq!(data_b.value, 111, "b reads a's payload");

        // Writing through b lands in the same entry, so a now reads b's data.
        store(&mut tt, b, side, 222, true, Bound::Lower, 12, 0x22, 222);
        let (found_a, data_a, _) = tt.probe(a, side);
        assert!(found_a);
        assert_eq!(data_a.value, 222, "the two share one entry");
        assert_eq!(data_a.move16, 0x22);
    }
}
