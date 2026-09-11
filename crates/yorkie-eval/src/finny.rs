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

use crate::aligned::Aligned64;
use crate::features::{
    DiffScratch, FeatureIndex, MAX_ACTIVE_FEATURES, active_features_into, changed_indices_into,
};
use crate::transformer::{apply_diff, refresh_perspective};
use crate::types::{HIDDEN_SIZE, NetworkParams};

/// One cached refreshed half: the accumulation and the active-feature list it
/// was built from.
#[derive(Debug)]
struct FinnyEntry {
    /// `ft_biases + sum(ft_weights[c] for c in active)`, valid only while
    /// [`Self::initialized`] is set.
    accumulation: Aligned64<i16>,
    /// The active-feature multiset [`Self::accumulation`] corresponds to.
    active: Vec<FeatureIndex>,
    /// Whether [`Self::accumulation`] currently satisfies the module invariant.
    initialized: bool,
}

impl FinnyEntry {
    fn new() -> Self {
        FinnyEntry {
            accumulation: Aligned64::<i16>::zeroed(HIDDEN_SIZE),
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
    /// Allocate an empty cache. Boxed because the entries own ~0.5 MiB of
    /// accumulation buffers, so this belongs at worker setup, not on the search
    /// path.
    pub fn new() -> Box<Self> {
        Box::new(FinnyCache {
            entries: std::array::from_fn(|_| std::array::from_fn(|_| FinnyEntry::new())),
            scratch_active: Vec::with_capacity(MAX_ACTIVE_FEATURES),
            diff: DiffScratch::default(),
        })
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
        dst: &mut [i16],
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

        dst.copy_from_slice(&entry.accumulation);
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
                    &*entry.accumulation,
                    expected.as_slice(),
                    "finny entry violates the biases+columns invariant",
                );
            }
        }
    }
}
