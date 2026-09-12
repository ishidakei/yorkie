//! Finny tables: a per-worker cache of refreshed accumulator halves, keyed by
//! the perspective's own-king square.
//!
//! A perspective whose own king moved cannot be updated differentially, so its
//! half is rebuilt from the FT biases plus all [`MAX_ACTIVE_FEATURES`] active
//! columns — roughly an order of magnitude more work than a one-piece diff, on
//! a move shogi trees are full of. But within one search the king revisits the
//! same squares over and over, so keeping one refreshed accumulator per
//! (perspective, king square) alongside the feature list it was built from
//! turns most of those rebuilds into a handful of columns.
//!
//! The reference ships the same structure dormant behind an undefined
//! `USE_FINNY_TABLES`, so this adapts code a default build never compiles.
//!
//! # The invariant
//!
//! For every initialised entry:
//!
//! ```text
//! entry.accumulation == ft_biases + sum over entry.active of ft_weights[column]
//! ```
//!
//! That mentions no position, only a feature multiset, which is why an entry
//! stays valid across nodes, searches and whole games. Only the weights
//! changing underneath could invalidate it, and they cannot: the parameters are
//! placed once, before any worker exists, and are read-only from then on.
//!
//! Applying `changed_indices(entry.active, new_active)` to the entry preserves
//! the invariant and re-establishes the accumulator identity for the new
//! position **exactly**: the accumulator is a sum of `i16` columns under
//! wrapping arithmetic, so any decomposition of the same multiset of adds and
//! subs is bit-identical.
//!
//! One boxed cache per search worker, allocated at worker setup and shared with
//! nobody, so Lazy SMP needs no locking here.

use yorkie_state::{Color, Position, Square};

use crate::features::{
    DiffScratch, FeatureIndex, MAX_ACTIVE_FEATURES, active_features_into, changed_indices_into,
};
use crate::transformer::{apply_diff, refresh_perspective};
use crate::types::{HIDDEN_SIZE, NetworkParams};

/// One cached refreshed half: the accumulation and the active-feature list it
/// was built from.
///
/// The accumulation is held inline and the entry is 64-byte aligned, so the
/// table is one contiguous run of cache-line-aligned rows rather than a row per
/// allocation, and the rebuild reaches one through the table's base address
/// alone.
#[derive(Debug)]
#[repr(C, align(64))]
struct FinnyEntry {
    /// `ft_biases + sum(ft_weights[c] for c in active)`, valid only while
    /// [`Self::initialized`] is set.
    accumulation: [i16; HIDDEN_SIZE],
    /// The active-feature multiset [`Self::accumulation`] corresponds to.
    active: Vec<FeatureIndex>,
    /// Whether [`Self::accumulation`] currently satisfies the module invariant.
    initialized: bool,
}

impl FinnyEntry {
    fn new() -> Self {
        FinnyEntry {
            accumulation: [0; HIDDEN_SIZE],
            active: Vec::with_capacity(MAX_ACTIVE_FEATURES),
            initialized: false,
        }
    }
}

/// A worker-private finny table: one `FinnyEntry` per (perspective, own-king
/// square).
#[derive(Debug)]
pub struct FinnyCache {
    /// `[perspective][own king square]`. The key is the untouched king square,
    /// not the mirrored `sq_k_code`: two mirror-equivalent king squares
    /// generate different index sets and must not share an entry.
    entries: [[FinnyEntry; Square::COUNT]; Color::COUNT],
    /// Reusable buffer for the post-move active-feature list.
    scratch_active: Vec<FeatureIndex>,
    /// Reusable buffers for the entry-vs-position feature diff.
    diff: DiffScratch,
}

impl FinnyCache {
    /// Allocate an empty cache. Boxed because the entries carry ~0.5 MiB of
    /// accumulation rows, so this belongs at worker setup, not on the search
    /// path.
    ///
    /// Each entry is written straight into the heap allocation. Building the
    /// table as a value first would put that half-mebibyte on the stack, and a
    /// worker thread's stack is not that large.
    pub fn new() -> Box<Self> {
        let mut cache = Box::<FinnyCache>::new_uninit();
        let base = cache.as_mut_ptr();
        // SAFETY: `base` addresses one uninitialised, suitably aligned
        // `FinnyCache` this call owns; every field is written exactly once, and
        // through a raw pointer, so nothing reads the memory before it holds a
        // value. `assume_init` runs only once all of them have.
        unsafe {
            for perspective in 0..Color::COUNT {
                for bucket in 0..Square::COUNT {
                    (&raw mut (*base).entries[perspective][bucket]).write(FinnyEntry::new());
                }
            }
            (&raw mut (*base).scratch_active).write(Vec::with_capacity(MAX_ACTIVE_FEATURES));
            (&raw mut (*base).diff).write(DiffScratch::default());
            cache.assume_init()
        }
    }

    /// Rebuild `perspective`'s half of the accumulator for `pos` into `dst`,
    /// going through this cache.
    ///
    /// # Panics
    /// Panics if `pos` is missing `perspective`'s king.
    pub(crate) fn refresh_into<N: NetworkParams>(
        &mut self,
        net: N,
        pos: &Position,
        perspective: Color,
        dst: &mut [i16; HIDDEN_SIZE],
    ) {
        // Destructure so the entry borrow and the scratch borrows are disjoint.
        let FinnyCache {
            entries,
            scratch_active,
            diff,
            ..
        } = self;

        let bucket = pos
            .king_square(perspective)
            .expect("position must have the perspective's king")
            .index() as usize;
        let entry = &mut entries[perspective.index()][bucket];

        active_features_into(pos, perspective, scratch_active);

        if entry.initialized {
            // Within one bucket the own-king square is identical on both sides,
            // so the padding feature cancels in the multiset diff exactly as it
            // does for an ordinary incremental update.
            changed_indices_into(&entry.active, scratch_active, diff);
            apply_diff(
                &mut entry.accumulation,
                net.ft_weights(),
                &diff.added,
                &diff.removed,
            );
        } else {
            // A cold entry pays the from-scratch rebuild once per bucket.
            refresh_perspective(
                &mut entry.accumulation,
                net.ft_biases(),
                net.ft_weights(),
                scratch_active,
            );
            entry.initialized = true;
        }

        *dst = entry.accumulation;
        // The old list becomes the next call's scratch, so the steady state
        // allocates nothing.
        std::mem::swap(&mut entry.active, scratch_active);
    }
}

#[cfg(test)]
impl FinnyCache {
    /// Whether the next rebuild in that bucket would be a warm hit.
    pub(crate) fn is_warm(&self, perspective: Color, king_sq: Square) -> bool {
        self.entries[perspective.index()][king_sq.index() as usize].initialized
    }

    /// Recompute `biases + sum(columns)` from `entry.active` and compare, for
    /// every initialised entry.
    pub(crate) fn assert_invariant<N: NetworkParams>(&self, net: N) {
        for per_color in self.entries.iter() {
            for entry in per_color.iter() {
                if !entry.initialized {
                    continue;
                }
                let mut expected: Vec<i16> = net.ft_biases().to_vec();
                for &idx in &entry.active {
                    let base = idx as usize * HIDDEN_SIZE;
                    let col = &net.ft_weights()[base..base + HIDDEN_SIZE];
                    for (a, &w) in expected.iter_mut().zip(col.iter()) {
                        *a = a.wrapping_add(w);
                    }
                }
                assert_eq!(
                    entry.accumulation.as_slice(),
                    expected.as_slice(),
                    "finny entry violates the biases+columns invariant",
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_cache_is_one_arena_of_cache_line_aligned_empty_entries() {
        /// Compared against whole, rather than lane by lane: this test runs
        /// under miri, where a quarter of a million interpreted loads is the
        /// difference between a fast test and a slow one.
        const ZEROED: [i16; HIDDEN_SIZE] = [0; HIDDEN_SIZE];

        let cache = FinnyCache::new();
        assert!(
            size_of::<FinnyEntry>().is_multiple_of(64),
            "an entry that is not a whole number of cache lines wide pushes the next one off a line",
        );

        let mut previous: Option<usize> = None;
        for perspective in [Color::Black, Color::White] {
            for bucket in 0..Square::COUNT {
                let entry = &cache.entries[perspective.index()][bucket];
                assert!(!entry.initialized);
                assert!(entry.accumulation == ZEROED, "entry is not zeroed");
                assert!(entry.active.is_empty());

                let row = entry.accumulation.as_ptr() as usize;
                assert_eq!(
                    row % 64,
                    0,
                    "{perspective:?} bucket {bucket} is not 64-byte aligned"
                );
                if let Some(previous) = previous {
                    assert_eq!(
                        row - previous,
                        size_of::<FinnyEntry>(),
                        "{perspective:?} bucket {bucket} does not follow its predecessor",
                    );
                }
                previous = Some(row);
            }
        }
    }
}
