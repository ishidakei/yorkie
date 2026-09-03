//! Feature-transformer accumulator and its output transform.
//!
//! Ported from `nnue_feature_transformer.h`. The per-lane kernels go through
//! [`crate::simd`], which picks the AVX-512 backend or the scalar baseline at
//! compile time — bit-identical either way. The layer-stack forward pass lives
//! in [`crate::network`].
//!
//! Because the accumulator is a linear sum of `i16` weight columns,
//! `prev - removed + added` equals a from-scratch refresh of the post-move
//! position bit for bit under wrapping `i16`. That is what lets a perspective be
//! updated incrementally; one whose own king moved is rebuilt instead, since its
//! whole index space shifts.
//!
//! Index `[Color::Black]` always holds Black's half and `[Color::White]` always
//! White's. The side-to-move reordering happens only in the output transform,
//! which folds each perspective's `HIDDEN_SIZE` values to `HIDDEN_SIZE / 2`
//! bytes by pairing lane `j` with lane `j + HIDDEN_SIZE/2`, clamping each to
//! `[0, 254]`, multiplying, and shifting right by 9.

use yorkie_state::{Color, Move, Position};

use crate::aligned::Aligned64;
use crate::features::{
    FeatureIndex, MoveDelta, active_features, changed_indices, requires_full_refresh,
};
use crate::finny::FinnyCache;
use crate::simd::{post_ft_kernel, transformer_kernel};
use crate::types::{FC_0_INPUT_DIMS, HIDDEN_SIZE, NnueNetwork};

/// Width of the byte buffer the output transform fills: one lane per fc_0
/// input, i.e. `HIDDEN_SIZE/2` per perspective across the two perspectives.
pub const FT_OUTPUT_DIMS: usize = FC_0_INPUT_DIMS;

/// Per-perspective feature-transformer accumulator: one `i16` vector of
/// [`HIDDEN_SIZE`] per perspective, each cache-line aligned.
#[derive(Debug)]
pub struct Accumulator {
    /// Indexed by [`Color::index`], never by side to move.
    perspectives: [Aligned64<i16>; Color::COUNT],
}

impl Accumulator {
    /// Allocates a zeroed accumulator (both perspectives all-zero).
    pub fn new() -> Self {
        Accumulator {
            perspectives: [
                Aligned64::<i16>::zeroed(HIDDEN_SIZE),
                Aligned64::<i16>::zeroed(HIDDEN_SIZE),
            ],
        }
    }

    /// Read-only view of one perspective's accumulator half.
    pub fn perspective(&self, color: Color) -> &[i16] {
        &self.perspectives[color.index()]
    }

    /// Rebuild both perspectives from the FT biases plus each perspective's
    /// active feature columns, added with wrapping `i16`: the downstream
    /// clipped output transform saturates, so overflow here is intended.
    ///
    /// # Panics
    /// Panics if `pos` is missing either king.
    pub fn refresh(&mut self, net: &NnueNetwork, pos: &Position) {
        for color in [Color::Black, Color::White] {
            let feats = active_features(pos, color);
            refresh_perspective(
                &mut self.perspectives[color.index()],
                &net.ft_biases,
                &net.ft_weights,
                &feats,
            );
        }
    }

    /// Incrementally derive the accumulator *after* `mv` from `self`, the
    /// accumulator for the position *before* it.
    ///
    /// `pos` must be the pre-move position, and is left unchanged: this does
    /// its own `do_move` / `undo_move` internally to read the post-move active
    /// features, so the caller applies the real move itself afterwards.
    ///
    /// # Panics
    /// Panics if `pos`, before or after `mv`, is missing either king.
    pub fn update_after_move(
        &self,
        net: &NnueNetwork,
        pos: &mut Position,
        mv: Move,
    ) -> Accumulator {
        let refresh = [
            requires_full_refresh(mv, Color::Black),
            requires_full_refresh(mv, Color::White),
        ];

        // A perspective about to refresh needs no pre-move feature scan.
        let before: [Vec<FeatureIndex>; Color::COUNT] = [
            if refresh[Color::Black.index()] {
                Vec::new()
            } else {
                active_features(pos, Color::Black)
            },
            if refresh[Color::White.index()] {
                Vec::new()
            } else {
                active_features(pos, Color::White)
            },
        ];

        let undo = pos.do_move(mv);

        let mut next = Accumulator::new();
        for color in [Color::Black, Color::White] {
            let i = color.index();
            if refresh[i] {
                refresh_perspective(
                    &mut next.perspectives[i],
                    &net.ft_biases,
                    &net.ft_weights,
                    &active_features(pos, color),
                );
            } else {
                let after = active_features(pos, color);
                let (removed, added) = changed_indices(&before[i], &after);
                next.perspectives[i].copy_from_slice(&self.perspectives[i]);
                apply_diff(&mut next.perspectives[i], &net.ft_weights, &added, &removed);
            }
        }

        pos.undo_move(mv, undo);
        next
    }

    /// Write the post-move accumulator into `dst`, deriving each perspective
    /// from `src` — the pre-move accumulator — plus a [`MoveDelta`]. Writing
    /// into a caller-owned `dst` lets the search reuse a preallocated per-worker
    /// stack instead of allocating a buffer per node.
    ///
    /// `post_pos` must be the position **after** the move, and is read only to
    /// full-refresh a perspective whose own king moved.
    ///
    /// # Panics
    /// Panics if `post_pos` is missing either king.
    pub fn derive_into(
        src: &Accumulator,
        dst: &mut Accumulator,
        net: &NnueNetwork,
        post_pos: &Position,
        delta: &MoveDelta,
    ) {
        for color in [Color::Black, Color::White] {
            let i = color.index();
            match delta.half(color) {
                Some(pd) => {
                    dst.perspectives[i].copy_from_slice(&src.perspectives[i]);
                    apply_diff(
                        &mut dst.perspectives[i],
                        &net.ft_weights,
                        pd.added(),
                        pd.removed(),
                    );
                }
                None => {
                    refresh_perspective(
                        &mut dst.perspectives[i],
                        &net.ft_biases,
                        &net.ft_weights,
                        &active_features(post_pos, color),
                    );
                }
            }
        }
    }

    /// [`Accumulator::derive_into`] with the king-move rebuild routed through a
    /// worker-private finny table. Rather than rebuilding the half from the
    /// biases plus every active column, the rebuild arm diffs the post-move
    /// feature list against the cached half for that king square. The cache
    /// stores a *sum of columns*, and wrapping `i16` addition does not care how
    /// that sum is decomposed, so the result is unchanged.
    ///
    /// The reference ships its finny tables dormant behind an undefined
    /// `USE_FINNY_TABLES`; this adopts them for the speed. It is
    /// value-invariant, so it changes no search behaviour and no node count.
    ///
    /// # Panics
    /// Panics if `post_pos` is missing either king.
    pub fn derive_into_cached(
        src: &Accumulator,
        dst: &mut Accumulator,
        net: &NnueNetwork,
        post_pos: &Position,
        delta: &MoveDelta,
        cache: &mut FinnyCache,
    ) {
        for color in [Color::Black, Color::White] {
            let i = color.index();
            match delta.half(color) {
                Some(pd) => {
                    dst.perspectives[i].copy_from_slice(&src.perspectives[i]);
                    apply_diff(
                        &mut dst.perspectives[i],
                        &net.ft_weights,
                        pd.added(),
                        pd.removed(),
                    );
                }
                None => cache.refresh_into(net, post_pos, color, &mut dst.perspectives[i]),
            }
        }
    }

    /// Pack the accumulator into `out`, the byte input buffer for `fc_0`, as
    /// `[stm-half | ~stm-half]`. See the module docs for the fold.
    pub fn output_transform(&self, stm: Color, out: &mut [u8; FT_OUTPUT_DIMS]) {
        const HALF: usize = HIDDEN_SIZE / 2;
        let stm_half = &self.perspectives[stm.index()];
        let other_half = &self.perspectives[stm.flip().index()];
        post_ft_kernel::ewm_one_perspective(stm_half, &mut out[..HALF]);
        post_ft_kernel::ewm_one_perspective(other_half, &mut out[HALF..]);
    }
}

impl Default for Accumulator {
    fn default() -> Self {
        Self::new()
    }
}

/// Rebuild one perspective's half: copy the biases, then add each active
/// feature's weight column. `ft_weights` is row-major `[feature][lane]`.
pub(crate) fn refresh_perspective(
    out: &mut [i16],
    ft_biases: &[i16],
    ft_weights: &[i16],
    indices: &[FeatureIndex],
) {
    out.copy_from_slice(ft_biases);
    transformer_kernel::add_features(out, ft_weights, indices);
}

/// Apply a feature delta to an already-copied perspective half: subtract every
/// removed column and add every added column. Under wrapping `i16` the two
/// passes commute.
///
/// The common single-add arities route to fused kernels, one accumulator
/// round-trip per lane; any other shape falls back to a separate sub-then-add.
pub(crate) fn apply_diff(
    out: &mut [i16],
    weights: &[i16],
    added: &[FeatureIndex],
    removed: &[FeatureIndex],
) {
    match (added.len(), removed.len()) {
        (1, 1) => transformer_kernel::add_sub_features(out, weights, added, removed),
        (1, 2) => transformer_kernel::add_sub_sub_features(
            out,
            weights,
            &added[..1],
            &removed[..1],
            &removed[1..2],
        ),
        _ => {
            transformer_kernel::sub_features(out, weights, removed);
            transformer_kernel::add_features(out, weights, added);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{NetHeader, NnueNetworkBuilder};
    use yorkie_state::{Move, PieceKind, format_usi_move, parse_sfen, parse_usi_move};

    const STARTPOS: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
    const HALF: usize = HIDDEN_SIZE / 2;

    // The output-transform clamp ceiling and shift, restated so these tests
    // transcribe the reference formula independently.
    const EWM_CLAMP: i32 = 127 * 2;
    const EWM_SHIFT: i32 = 9;

    #[test]
    fn new_is_zeroed_and_cache_line_aligned() {
        let acc = Accumulator::new();
        for color in [Color::Black, Color::White] {
            let half = acc.perspective(color);
            assert_eq!(half.len(), HIDDEN_SIZE);
            assert!(half.iter().all(|&x| x == 0));
            assert_eq!(
                half.as_ptr() as usize % 64,
                0,
                "{color:?} half is not 64-byte aligned"
            );
        }
    }

    #[test]
    fn refresh_perspective_sums_bias_and_active_columns() {
        // biases[i] = i; weights[column c][lane i] = c*100 + i. With active
        // indices {2, 5}: out[i] = i + (200 + i) + (500 + i) = 700 + 3*i.
        const N_FEATURES: usize = 8;
        let biases: Vec<i16> = (0..HIDDEN_SIZE as i16).collect();
        let mut weights = vec![0i16; N_FEATURES * HIDDEN_SIZE];
        for c in 0..N_FEATURES {
            for i in 0..HIDDEN_SIZE {
                weights[c * HIDDEN_SIZE + i] = (c as i16) * 100 + i as i16;
            }
        }

        let mut out = vec![0i16; HIDDEN_SIZE];
        refresh_perspective(&mut out, &biases, &weights, &[2, 5]);

        assert_eq!(out[0], 700);
        assert_eq!(out[1], 703);
        assert_eq!(out[10], 730);
        assert_eq!(out[HIDDEN_SIZE - 1], 700 + 3 * (HIDDEN_SIZE as i16 - 1));
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, 700 + 3 * i as i16, "lane {i}");
        }
    }

    #[test]
    fn refresh_perspective_with_no_active_features_is_the_bias_vector() {
        let biases: Vec<i16> = (0..HIDDEN_SIZE).map(|i| (i as i16) % 17 - 8).collect();
        let weights = vec![123i16; 4 * HIDDEN_SIZE];
        let mut out = vec![0i16; HIDDEN_SIZE];
        refresh_perspective(&mut out, &biases, &weights, &[]);
        assert_eq!(out, biases);
    }

    #[test]
    fn add_features_uses_wrapping_i16_arithmetic() {
        // A bias of `i16::MAX` plus a column of ones wraps to `i16::MIN`.
        let mut out = vec![i16::MAX; HIDDEN_SIZE];
        let weights = vec![1i16; HIDDEN_SIZE];
        transformer_kernel::add_features(&mut out, &weights, &[0]);
        assert!(out.iter().all(|&x| x == i16::MIN));
    }

    /// A full-size network whose weight columns are seeded only where the given
    /// position references them, so the ~215 MiB block stays on lazy zero pages.
    fn synthetic_net_for(pos: &Position) -> NnueNetwork {
        synthetic_net_covering(std::slice::from_ref(pos))
    }

    /// [`synthetic_net_for`] seeded from *any* of `positions`. An
    /// incremental-update test passes both the pre- and post-move positions, so
    /// the delta lands on nonzero columns rather than untouched zero pages,
    /// where its add side would be a trivial no-op.
    fn synthetic_net_covering(positions: &[Position]) -> NnueNetwork {
        synthetic_net_covering_salted(positions, 0)
    }

    /// [`synthetic_net_covering`] with every bias and weight shifted by `salt`,
    /// so two nets over the same positions hold different parameters.
    fn synthetic_net_covering_salted(positions: &[Position], salt: i16) -> NnueNetwork {
        let header = NetHeader {
            version: 0,
            hash: 0,
            arch_id: "synthetic".to_string(),
        };
        // All parameters live in one arena; the FC arrays stay zero.
        let mut builder = NnueNetworkBuilder::new(header, [0u8; 32]);
        for (i, slot) in builder.ft_biases_mut().iter_mut().enumerate() {
            *slot = ((i as i16) % 17 - 8).wrapping_add(salt);
        }
        {
            let ft_weights = builder.ft_weights_mut();
            for pos in positions {
                for color in [Color::Black, Color::White] {
                    for idx in active_features(pos, color) {
                        let base = idx as usize * HIDDEN_SIZE;
                        for (i, slot) in ft_weights[base..base + HIDDEN_SIZE].iter_mut().enumerate()
                        {
                            *slot = (((idx as i32).wrapping_mul(31).wrapping_add(i as i32 * 7) % 23
                                - 11) as i16)
                                .wrapping_add(salt);
                        }
                    }
                }
            }
        }
        builder.build()
    }

    /// Recompute one perspective without sharing code with `refresh`.
    fn expected_half(net: &NnueNetwork, pos: &Position, color: Color) -> Vec<i16> {
        let mut acc: Vec<i16> = net.ft_biases.to_vec();
        for idx in active_features(pos, color) {
            let base = idx as usize * HIDDEN_SIZE;
            let col = &net.ft_weights[base..base + HIDDEN_SIZE];
            for (a, &w) in acc.iter_mut().zip(col.iter()) {
                *a = a.wrapping_add(w);
            }
        }
        acc
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn refresh_matches_independent_recomputation() {
        let pos = parse_sfen(STARTPOS).unwrap();
        let net = synthetic_net_for(&pos);
        let mut acc = Accumulator::new();
        acc.refresh(&net, &pos);
        for color in [Color::Black, Color::White] {
            assert_eq!(
                acc.perspective(color),
                expected_half(&net, &pos, color).as_slice(),
                "{color:?} perspective mismatch",
            );
        }
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn refresh_is_deterministic() {
        let pos = parse_sfen(STARTPOS).unwrap();
        let net = synthetic_net_for(&pos);
        let mut a = Accumulator::new();
        let mut b = Accumulator::new();
        a.refresh(&net, &pos);
        b.refresh(&net, &pos);
        for color in [Color::Black, Color::White] {
            assert_eq!(a.perspective(color), b.perspective(color));
        }
    }

    /// Assert the incremental update of `pos` by `mv` is bit-identical to a
    /// from-scratch refresh of the post-move position, and leaves `pos` alone.
    fn assert_incremental_matches_refresh(sfen: &str, usi: &str) {
        let mut pos = parse_sfen(sfen).unwrap();
        let mv: Move = parse_usi_move(usi, &pos).unwrap();

        let mut after_pos = pos.clone();
        after_pos.do_move(mv);
        let net = synthetic_net_covering(&[pos.clone(), after_pos]);

        let mut prev = Accumulator::new();
        prev.refresh(&net, &pos);

        let pos_snapshot = pos.clone();
        let next = prev.update_after_move(&net, &mut pos, mv);
        assert_eq!(pos, pos_snapshot, "update_after_move mutated the position");

        let undo = pos.do_move(mv);
        let mut expected = Accumulator::new();
        expected.refresh(&net, &pos);
        pos.undo_move(mv, undo);

        for color in [Color::Black, Color::White] {
            assert_eq!(
                next.perspective(color),
                expected.perspective(color),
                "{color:?}: incremental update != refresh for `{usi}` from `{sfen}`",
            );
        }
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn update_after_quiet_move_matches_refresh() {
        assert_incremental_matches_refresh(STARTPOS, "7g7f");
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn update_after_capture_matches_refresh() {
        assert_incremental_matches_refresh("4k4/1p7/9/9/9/9/9/1R7/4K4 b - 1", "8h8b");
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn update_after_drop_matches_refresh() {
        assert_incremental_matches_refresh("4k4/9/9/9/9/9/9/9/4K4 b P 1", "P*5e");
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn update_after_promotion_matches_refresh() {
        assert_incremental_matches_refresh("4k4/9/9/1P7/9/9/9/9/4K4 b - 1", "8d8c+");
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn update_after_black_king_move_matches_refresh() {
        // Black's own king moves -> Black refreshes, White stays incremental.
        assert_incremental_matches_refresh("4k4/9/9/9/9/9/9/9/4K4 b - 1", "5i5h");
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn update_after_white_king_move_matches_refresh() {
        // White to move, White's own king moves -> White refreshes, Black
        // stays incremental.
        assert_incremental_matches_refresh("4k4/9/9/9/9/9/9/9/4K4 w - 1", "5a5b");
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn update_after_move_sequence_stays_bit_exact() {
        // A multi-ply line threaded through incremental updates.
        const LINE: [&str; 5] = ["7g7f", "3c3d", "8h2b+", "3a2b", "5i5h"];

        // Collect every position the line visits, so the synthetic net covers
        // the feature columns the deltas touch.
        let mut walk = parse_sfen(STARTPOS).unwrap();
        let mut visited = vec![walk.clone()];
        for usi in LINE {
            let mv = parse_usi_move(usi, &walk).unwrap();
            walk.do_move(mv);
            visited.push(walk.clone());
        }
        let net = synthetic_net_covering(&visited);

        let mut pos = parse_sfen(STARTPOS).unwrap();
        let mut base = Accumulator::new();
        base.refresh(&net, &pos);

        let mut stack: Vec<Accumulator> = vec![base];
        for usi in LINE {
            let mv = parse_usi_move(usi, &pos).unwrap();
            let next = stack.last().unwrap().update_after_move(&net, &mut pos, mv);
            stack.push(next);
            pos.do_move(mv);

            let mut expected = Accumulator::new();
            expected.refresh(&net, &pos);
            for color in [Color::Black, Color::White] {
                assert_eq!(
                    stack.last().unwrap().perspective(color),
                    expected.perspective(color),
                    "{color:?}: drift after `{usi}`",
                );
            }
        }
    }

    use crate::features::MoveDelta;

    /// Drive [`Accumulator::derive_into`] as the search does — snapshot the
    /// [`MoveDelta`] pre-move, apply the move, derive the child from the parent
    /// — and compare against both a refresh and
    /// [`Accumulator::update_after_move`].
    fn assert_derive_matches_refresh(sfen: &str, usi: &str) {
        let mut pos = parse_sfen(sfen).unwrap();
        let mv: Move = parse_usi_move(usi, &pos).unwrap();

        let mut after_pos = pos.clone();
        after_pos.do_move(mv);
        let net = synthetic_net_covering(&[pos.clone(), after_pos]);

        let mut parent = Accumulator::new();
        parent.refresh(&net, &pos);

        // The scan-based form runs its own do/undo and leaves `pos` intact.
        let oracle = parent.update_after_move(&net, &mut pos, mv);

        let delta = MoveDelta::from_move(&pos, mv);
        let undo = pos.do_move(mv);
        let mut child = Accumulator::new();
        Accumulator::derive_into(&parent, &mut child, &net, &pos, &delta);

        let mut expected = Accumulator::new();
        expected.refresh(&net, &pos);
        pos.undo_move(mv, undo);

        // `derive_into` writes only `dst`, so a search pop that discards the
        // child trivially restores the parent.
        for color in [Color::Black, Color::White] {
            assert_eq!(
                child.perspective(color),
                expected.perspective(color),
                "{color:?}: derive_into != refresh for `{usi}` from `{sfen}`",
            );
            assert_eq!(
                child.perspective(color),
                oracle.perspective(color),
                "{color:?}: derive_into != update_after_move for `{usi}` from `{sfen}`",
            );
        }
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn derive_after_quiet_move_matches_refresh() {
        assert_derive_matches_refresh(STARTPOS, "7g7f");
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn derive_after_capture_matches_refresh() {
        assert_derive_matches_refresh("4k4/1p7/9/9/9/9/9/1R7/4K4 b - 1", "8h8b");
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn derive_after_drop_matches_refresh() {
        assert_derive_matches_refresh("4k4/9/9/9/9/9/9/9/4K4 b P 1", "P*5e");
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn derive_after_promotion_matches_refresh() {
        assert_derive_matches_refresh("4k4/9/9/1P7/9/9/9/9/4K4 b - 1", "8d8c+");
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn derive_after_capture_promotion_matches_refresh() {
        // A bishop captures and promotes: the two-add, two-sub shape, with a
        // promoted board plane.
        assert_derive_matches_refresh("4k4/2r6/9/9/9/9/9/1B7/4K4 b - 1", "8h7b+");
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn derive_after_capturing_a_promoted_piece_matches_refresh() {
        // A captured dragon must revert to a bare rook in hand, not stay on the
        // promoted plane.
        assert_derive_matches_refresh("4k4/2+r6/9/9/9/9/9/1B7/4K4 b - 1", "8h7b+");
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn derive_after_black_king_move_matches_refresh() {
        // Black's own king moves -> Black refreshes, White stays incremental.
        assert_derive_matches_refresh("4k4/9/9/9/9/9/9/9/4K4 b - 1", "5i5h");
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn derive_after_white_king_move_matches_refresh() {
        assert_derive_matches_refresh("4k4/9/9/9/9/9/9/9/4K4 w - 1", "5a5b");
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn derive_after_king_capture_matches_refresh() {
        // The Black king captures, so for White — which stays incremental —
        // the mover is on the shared king plane and a capture happens at once.
        assert_derive_matches_refresh("4k4/9/9/9/9/9/9/4r4/4K4 b - 1", "5i5h");
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn null_move_leaves_the_accumulator_unchanged() {
        // A null move changes no pieces and no king squares, so the search's
        // reuse of the parent accumulator across one is exact.
        let mut pos = parse_sfen(STARTPOS).unwrap();
        let net = synthetic_net_for(&pos);
        let mut acc = Accumulator::new();
        acc.refresh(&net, &pos);

        pos.do_null_move();
        let mut expected = Accumulator::new();
        expected.refresh(&net, &pos);
        for color in [Color::Black, Color::White] {
            assert_eq!(
                acc.perspective(color),
                expected.perspective(color),
                "{color:?}: null move changed the accumulator",
            );
        }
    }

    /// A pure function of the ply and position, so the coverage pass and the
    /// accumulator pass walk the same line. It rotates a preference for
    /// promotions and captures so those move types are exercised, falling back
    /// to an LCG index for breadth.
    fn choose(ply: usize, pos: &Position, moves: &[Move]) -> Move {
        let idx = ((ply as u64).wrapping_mul(2_654_435_761) >> 13) as usize % moves.len();
        match ply % 4 {
            0 => moves.iter().copied().find(|m| m.is_promote()),
            1 => moves
                .iter()
                .copied()
                .find(|m| !m.is_drop() && pos.board().get(m.to_sq()).is_some()),
            _ => None,
        }
        .unwrap_or(moves[idx])
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn derive_into_stays_bit_exact_over_a_long_playout() {
        const PLIES: usize = 48;

        // Collect every visited position, so the synthetic net seeds the
        // feature columns the deltas touch.
        let mut walk = parse_sfen(STARTPOS).unwrap();
        let mut visited = vec![walk.clone()];
        let mut line: Vec<Move> = Vec::new();
        for ply in 0..PLIES {
            let mut moves: Vec<Move> = Vec::new();
            walk.generate_legal_all(&mut moves);
            if moves.is_empty() {
                break;
            }
            let mv = choose(ply, &walk, &moves);
            line.push(mv);
            walk.do_move(mv);
            visited.push(walk.clone());
        }
        assert!(line.len() >= 30, "playout too short: {}", line.len());
        let net = synthetic_net_covering(&visited);

        let mut pos = parse_sfen(STARTPOS).unwrap();
        let mut base = Accumulator::new();
        base.refresh(&net, &pos);
        let mut stack: Vec<Accumulator> = vec![base];

        let mut saw_capture = false;
        let mut saw_promotion = false;
        let mut saw_drop = false;
        let mut saw_king_move = false;

        for (i, &mv) in line.iter().enumerate() {
            saw_capture |= !mv.is_drop() && pos.board().get(mv.to_sq()).is_some();
            saw_promotion |= mv.is_promote();
            saw_drop |= mv.is_drop();
            saw_king_move |= !mv.is_drop() && mv.moved_piece_after().kind == PieceKind::King;

            let delta = MoveDelta::from_move(&pos, mv);
            pos.do_move(mv);
            let mut child = Accumulator::new();
            Accumulator::derive_into(stack.last().unwrap(), &mut child, &net, &pos, &delta);

            let mut expected = Accumulator::new();
            expected.refresh(&net, &pos);
            for color in [Color::Black, Color::White] {
                assert_eq!(
                    child.perspective(color),
                    expected.perspective(color),
                    "{color:?}: drift at ply {i} after `{}`",
                    format_usi_move(mv),
                );
            }
            stack.push(child);
        }

        assert!(saw_capture, "playout exercised no capture");
        assert!(saw_promotion, "playout exercised no promotion");
        assert!(saw_drop, "playout exercised no drop");
        assert!(saw_king_move, "playout exercised no king move");
    }

    // The finny cache is checked in every state: cold, warm on the same
    // position, warm from a different position sharing the king square, and
    // across a network change, which must reset it rather than serve a stale
    // half.

    use crate::finny::FinnyCache;

    /// Rebuild `color`'s half through `cache`, compare against the independent
    /// recomputation, then re-check the cache's own invariant.
    fn assert_cached_refresh_matches(
        cache: &mut FinnyCache,
        net: &NnueNetwork,
        pos: &Position,
        color: Color,
        what: &str,
    ) {
        let mut dst = vec![0i16; HIDDEN_SIZE];
        cache.refresh_into(net, pos, color, &mut dst);
        assert_eq!(
            dst,
            expected_half(net, pos, color),
            "{color:?}: cached rebuild != from-scratch refresh ({what})",
        );
        cache.assert_invariant(net);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn cached_refresh_matches_a_cold_entry_then_a_warm_one() {
        let pos = parse_sfen(STARTPOS).unwrap();
        let net = synthetic_net_for(&pos);
        let mut cache = FinnyCache::new();

        for color in [Color::Black, Color::White] {
            let king = pos.king_square(color).unwrap();
            assert!(!cache.is_warm(color, king), "entry warm before any rebuild");
            assert_cached_refresh_matches(&mut cache, &net, &pos, color, "cold entry");
            assert!(cache.is_warm(color, king), "entry not warmed by a rebuild");
            // A second pass over the same position leaves the diff empty.
            assert_cached_refresh_matches(
                &mut cache,
                &net,
                &pos,
                color,
                "warm entry, same position",
            );
        }
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn cached_refresh_absorbs_a_stale_entry_from_a_different_position() {
        // The same black king square in both positions but wildly different
        // piece sets, so the second rebuild hits a warm entry whose cached half
        // belongs to the first and must be diffed all the way across.
        let a = parse_sfen(STARTPOS).unwrap();
        let b = parse_sfen("4k4/9/2+R6/9/9/1n2s4/9/9/4K4 b GPl2p 1").unwrap();
        assert_eq!(a.king_square(Color::Black), b.king_square(Color::Black));

        let net = synthetic_net_covering(&[a.clone(), b.clone()]);
        let mut cache = FinnyCache::new();

        assert_cached_refresh_matches(&mut cache, &net, &a, Color::Black, "cold entry");
        assert_cached_refresh_matches(
            &mut cache,
            &net,
            &b,
            Color::Black,
            "stale entry, same bucket",
        );
        // ...and back again, so the diff runs in both directions.
        assert_cached_refresh_matches(&mut cache, &net, &a, Color::Black, "stale entry, reversed");
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn cached_refresh_invalidates_on_a_network_change() {
        let pos = parse_sfen(STARTPOS).unwrap();
        let net_a = synthetic_net_covering_salted(std::slice::from_ref(&pos), 0);
        let net_b = synthetic_net_covering_salted(std::slice::from_ref(&pos), 5);
        // Both nets are live at once, so they cannot share an identity token.
        assert_ne!(
            net_a.ft_biases.to_vec(),
            net_b.ft_biases.to_vec(),
            "the two synthetic nets must actually differ",
        );

        let mut cache = FinnyCache::new();
        assert_cached_refresh_matches(&mut cache, &net_a, &pos, Color::Black, "net A, cold");

        // Rebuilding the same position against a different network must reset
        // the cache: reusing the entry would return the first net's half.
        assert_cached_refresh_matches(&mut cache, &net_b, &pos, Color::Black, "net B after net A");
        assert!(
            !cache.is_warm(Color::White, pos.king_square(Color::White).unwrap()),
            "the reset must clear every entry, not just the one rebuilt",
        );
        // And back again, which must reset once more.
        assert_cached_refresh_matches(&mut cache, &net_a, &pos, Color::Black, "net A after net B");
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn derive_into_cached_matches_derive_into_on_a_king_move() {
        // A move that rebuilds, through the cached entry point.
        let mut pos = parse_sfen("4k4/9/9/9/9/9/9/9/4K4 b - 1").unwrap();
        let mv = parse_usi_move("5i5h", &pos).unwrap();
        let mut after_pos = pos.clone();
        after_pos.do_move(mv);
        let net = synthetic_net_covering(&[pos.clone(), after_pos]);

        let mut parent = Accumulator::new();
        parent.refresh(&net, &pos);

        let delta = MoveDelta::from_move(&pos, mv);
        assert!(
            delta.half(Color::Black).is_none(),
            "this fixture must exercise the rebuild arm",
        );
        pos.do_move(mv);

        let mut uncached = Accumulator::new();
        Accumulator::derive_into(&parent, &mut uncached, &net, &pos, &delta);

        let mut cache = FinnyCache::new();
        let mut cached = Accumulator::new();
        Accumulator::derive_into_cached(&parent, &mut cached, &net, &pos, &delta, &mut cache);

        let mut expected = Accumulator::new();
        expected.refresh(&net, &pos);

        for color in [Color::Black, Color::White] {
            assert_eq!(
                cached.perspective(color),
                expected.perspective(color),
                "{color:?}: derive_into_cached != refresh",
            );
            assert_eq!(
                cached.perspective(color),
                uncached.perspective(color),
                "{color:?}: derive_into_cached != derive_into",
            );
        }
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn derive_into_cached_stays_bit_exact_over_a_long_playout() {
        // The `derive_into` playout re-run through one persistent finny cache,
        // with real king moves and warm-entry hits along the way.
        const PLIES: usize = 48;

        let mut walk = parse_sfen(STARTPOS).unwrap();
        let mut visited = vec![walk.clone()];
        let mut line: Vec<Move> = Vec::new();
        for ply in 0..PLIES {
            let mut moves: Vec<Move> = Vec::new();
            walk.generate_legal_all(&mut moves);
            if moves.is_empty() {
                break;
            }
            // The derive loop below visits sibling king-move successors, so
            // their feature columns must be seeded too.
            for &m in moves
                .iter()
                .filter(|m| !m.is_drop() && m.moved_piece_after().kind == PieceKind::King)
            {
                let mut side = walk.clone();
                side.do_move(m);
                visited.push(side);
            }
            let mv = choose(ply, &walk, &moves);
            line.push(mv);
            walk.do_move(mv);
            visited.push(walk.clone());
        }
        assert!(line.len() >= 30, "playout too short: {}", line.len());
        let net = synthetic_net_covering(&visited);

        let mut cache = FinnyCache::new();
        let mut king_moves = 0usize;
        let mut warm_hits = 0usize;

        // A single linear playout rarely revisits a king square, where a real
        // search does so constantly. The sibling loop below models trying
        // several king moves at one node; the repeated pass models re-searching
        // the same king move from many.
        for pass in 0..2 {
            let mut pos = parse_sfen(STARTPOS).unwrap();
            let mut base = Accumulator::new();
            base.refresh(&net, &pos);
            let mut stack: Vec<Accumulator> = vec![base];

            for (i, &mv) in line.iter().enumerate() {
                // Every legal king move here, derived and discarded.
                let mut siblings: Vec<Move> = Vec::new();
                pos.generate_legal_all(&mut siblings);
                siblings.retain(|m| {
                    *m != mv && !m.is_drop() && m.moved_piece_after().kind == PieceKind::King
                });
                for &sib in siblings.iter().chain(std::iter::once(&mv)) {
                    let delta = MoveDelta::from_move(&pos, sib);
                    let undo = pos.do_move(sib);

                    // Classified before the rebuild: a rebuild arm whose entry
                    // is already initialised is exactly a warm hit.
                    for color in [Color::Black, Color::White] {
                        if delta.half(color).is_none() {
                            king_moves += 1;
                            if cache.is_warm(color, pos.king_square(color).unwrap()) {
                                warm_hits += 1;
                            }
                        }
                    }

                    let mut child = Accumulator::new();
                    Accumulator::derive_into_cached(
                        stack.last().unwrap(),
                        &mut child,
                        &net,
                        &pos,
                        &delta,
                        &mut cache,
                    );

                    let mut expected = Accumulator::new();
                    expected.refresh(&net, &pos);
                    for color in [Color::Black, Color::White] {
                        assert_eq!(
                            child.perspective(color),
                            expected.perspective(color),
                            "{color:?}: drift on pass {pass} at ply {i} after `{}`",
                            format_usi_move(sib),
                        );
                    }
                    cache.assert_invariant(&net);

                    pos.undo_move(sib, undo);
                    if sib == mv {
                        pos.do_move(mv);
                        stack.push(child);
                    }
                }
            }
        }

        assert!(king_moves > 0, "playout exercised no king move");
        assert!(warm_hits > 0, "playout never hit a warm cache entry");
    }

    #[test]
    fn output_transform_lays_out_stm_then_other() {
        // us(Black) lane0 pair -> (200*100)>>9 = 39; them(White) -> (250*50)>>9 = 24.
        let mut acc = Accumulator::new();
        acc.perspectives[Color::Black.index()][0] = 200;
        acc.perspectives[Color::Black.index()][HALF] = 100;
        acc.perspectives[Color::White.index()][0] = 250;
        acc.perspectives[Color::White.index()][HALF] = 50;

        let mut out = [0u8; FT_OUTPUT_DIMS];
        acc.output_transform(Color::Black, &mut out);
        assert_eq!(out[0], 39);
        assert_eq!(out[HALF], 24);
        assert!(out[1..HALF].iter().all(|&x| x == 0));
        assert!(out[HALF + 1..].iter().all(|&x| x == 0));

        let mut out_w = [0u8; FT_OUTPUT_DIMS];
        acc.output_transform(Color::White, &mut out_w);
        assert_eq!(out_w[0], 24);
        assert_eq!(out_w[HALF], 39);
    }

    #[test]
    fn output_transform_clamps_min_zero_and_max() {
        let mut acc = Accumulator::new();
        let black = &mut acc.perspectives[Color::Black.index()];
        // lane 0: negative s0 clamps to 0 -> product 0.
        black[0] = -1_000;
        black[HALF] = 200;
        // lane 1: s1 = 0 -> product 0.
        black[1] = 254;
        black[HALF + 1] = 0;
        // lane 2: both above the ceiling clamp to 254 -> (254*254)>>9 = 126.
        black[2] = 30_000;
        black[HALF + 2] = 30_000;
        // lane 3: exact ceiling values, same result.
        black[3] = EWM_CLAMP as i16;
        black[HALF + 3] = EWM_CLAMP as i16;
        // lane 4: a mid value, hand-checked: (100*50)>>9 = 9.
        black[4] = 100;
        black[HALF + 4] = 50;

        let mut out = [0u8; FT_OUTPUT_DIMS];
        acc.output_transform(Color::Black, &mut out);
        assert_eq!(out[0], 0, "negative clamps to zero");
        assert_eq!(out[1], 0, "zero factor yields zero");
        assert_eq!(out[2], 126, "saturated pair");
        assert_eq!(out[3], 126, "exact ceiling pair");
        assert_eq!(out[4], 9, "mid-range pair");
    }

    #[test]
    fn output_transform_full_row_matches_scalar_formula() {
        // Against a direct transcription of the reference's scalar formula.
        let mut acc = Accumulator::new();
        for color in [Color::Black, Color::White] {
            let half = &mut acc.perspectives[color.index()];
            for i in 0..HIDDEN_SIZE {
                half[i] = ((i as i32 * 5 + color.index() as i32 * 3) % 400 - 50) as i16;
            }
        }

        for stm in [Color::Black, Color::White] {
            let mut out = [0u8; FT_OUTPUT_DIMS];
            acc.output_transform(stm, &mut out);

            let order = [stm, stm.flip()];
            for (p, persp) in order.iter().enumerate() {
                let half = acc.perspective(*persp);
                for j in 0..HALF {
                    let s0 = (half[j] as i32).clamp(0, EWM_CLAMP);
                    let s1 = (half[j + HALF] as i32).clamp(0, EWM_CLAMP);
                    let expected = ((s0 * s1) >> EWM_SHIFT) as u8;
                    assert_eq!(out[p * HALF + j], expected, "stm {stm:?} half {p} lane {j}");
                }
            }
        }
    }
}
