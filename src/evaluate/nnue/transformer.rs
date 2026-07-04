use super::features;
use super::simd::transformer_kernel;
use super::types::{Accumulator, FeatureIndex, FeatureList, NnueNetwork};
use crate::position::Position;
use crate::types::Color;

pub fn refresh(acc: &mut Accumulator, net: &NnueNetwork, pos: &Position) {
    refresh_perspective(&mut acc.us, net, pos, Color::BLACK);
    refresh_perspective(&mut acc.them, net, pos, Color::WHITE);
    acc.computed = true;
}

// Rebuilds one perspective's half from the bias vector plus its active features in `pos`.
fn refresh_perspective(half: &mut [i16], net: &NnueNetwork, pos: &Position, perspective: Color) {
    half.copy_from_slice(&net.ft_biases);
    let feats = features::active_features(pos, perspective);
    transformer_kernel::add_features(half, &net.ft_weights, feats.as_slice());
}

/// King-move update: rebuild `moving`'s own perspective (own-king-relative features all invalidated); the other perspective stays incremental via its pre-move `feature_diff`.
pub fn update_on_king_move(
    acc_prev: &Accumulator,
    acc_next: &mut Accumulator,
    net: &NnueNetwork,
    pos: &Position,
    moving: Color,
    diff_removed_other: &FeatureList,
    diff_added_other: &FeatureList,
) {
    if moving == Color::BLACK {
        refresh_perspective(&mut acc_next.us, net, pos, Color::BLACK);
        acc_next.them.copy_from_slice(&acc_prev.them);
        apply_diff(&mut acc_next.them, &net.ft_weights, diff_added_other, diff_removed_other);
    } else {
        refresh_perspective(&mut acc_next.them, net, pos, Color::WHITE);
        acc_next.us.copy_from_slice(&acc_prev.us);
        apply_diff(&mut acc_next.us, &net.ft_weights, diff_added_other, diff_removed_other);
    }
    acc_next.computed = true;
}

pub fn update_on_move(
    acc_prev: &Accumulator,
    acc_next: &mut Accumulator,
    net: &NnueNetwork,
    diff_removed_us: &FeatureList,
    diff_added_us: &FeatureList,
    diff_removed_them: &FeatureList,
    diff_added_them: &FeatureList,
) {
    acc_next.us.copy_from_slice(&acc_prev.us);
    acc_next.them.copy_from_slice(&acc_prev.them);

    apply_diff(&mut acc_next.us, &net.ft_weights, diff_added_us, diff_removed_us);
    apply_diff(&mut acc_next.them, &net.ft_weights, diff_added_them, diff_removed_them);

    acc_next.computed = true;
}

fn apply_diff(out: &mut [i16], weights: &[i16], added: &FeatureList, removed: &FeatureList) {
    let added: &[FeatureIndex] = added.as_slice();
    let removed: &[FeatureIndex] = removed.as_slice();
    match (added.len(), removed.len()) {
        (1, 1) => transformer_kernel::add_sub_features(out, weights, added, removed),
        (1, 2) => transformer_kernel::add_sub_sub_features(out, weights, &added[..1], &removed[..1], &removed[1..2]),
        _ => {
            transformer_kernel::sub_features(out, weights, removed);
            transformer_kernel::add_features(out, weights, added);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_fixtures::{run_with_large_stack, synthetic_net};
    use super::super::types::HIDDEN_SIZE;
    use super::*;
    use crate::movetypes::Move;

    fn start_pos() -> Position {
        Position::new()
    }

    #[test]
    fn update_on_move_with_empty_diffs_equals_copy() {
        run_with_large_stack(|| {
            let net = synthetic_net();
            let pos = start_pos();

            let mut acc_prev = Accumulator::zeroed();
            refresh(&mut acc_prev, net, &pos);

            let mut acc_next = Accumulator::zeroed();
            let empty = FeatureList::new();
            update_on_move(&acc_prev, &mut acc_next, net, &empty, &empty, &empty, &empty);

            assert!(acc_next.computed);
            assert_eq!(acc_next.us, acc_prev.us);
            assert_eq!(acc_next.them, acc_prev.them);
        });
    }

    #[test]
    fn refresh_with_empty_feature_list_equals_bias_vector() {
        run_with_large_stack(|| {
            let net = synthetic_net();
            let mut acc = Accumulator::zeroed();

            acc.us.copy_from_slice(&net.ft_biases);
            acc.them.copy_from_slice(&net.ft_biases);
            let empty: FeatureList = FeatureList::new();
            transformer_kernel::add_features(&mut acc.us, &net.ft_weights, empty.as_slice());
            transformer_kernel::add_features(&mut acc.them, &net.ft_weights, empty.as_slice());

            for i in 0..HIDDEN_SIZE {
                assert_eq!(acc.us[i], net.ft_biases[i]);
                assert_eq!(acc.them[i], net.ft_biases[i]);
            }
        });
    }

    // Drives `update_on_king_move` as the search does, then asserts both halves are bit-identical to a fresh refresh.
    fn assert_king_move_update_matches_refresh(pos: &mut Position, usi: &str) {
        let net = synthetic_net();
        let mv = Move::new_from_usi_str(usi, pos).expect("legal move");

        let refresh_black = features::requires_full_refresh(pos, mv, Color::BLACK);
        let refresh_white = features::requires_full_refresh(pos, mv, Color::WHITE);
        assert!(
            refresh_black ^ refresh_white,
            "test move must be a king move of exactly one side"
        );
        let moving = if refresh_black { Color::BLACK } else { Color::WHITE };

        let (removed_other, added_other) = features::feature_diff(pos, mv, moving.inverse());

        let mut acc_prev = Accumulator::zeroed();
        refresh(&mut acc_prev, net, pos);

        let gives_check = pos.gives_check(mv);
        pos.do_move(mv, gives_check);

        let mut acc_next = Accumulator::zeroed();
        update_on_king_move(&acc_prev, &mut acc_next, net, pos, moving, &removed_other, &added_other);
        assert!(acc_next.computed);

        let mut expected = Accumulator::zeroed();
        refresh(&mut expected, net, pos);
        assert_eq!(
            acc_next.us, expected.us,
            "BLACK perspective: king-move update != fresh refresh"
        );
        assert_eq!(
            acc_next.them, expected.them,
            "WHITE perspective: king-move update != fresh refresh"
        );
    }

    #[test]
    fn update_on_black_king_move_matches_refresh() {
        run_with_large_stack(|| {
            let mut pos = start_pos();
            assert_king_move_update_matches_refresh(&mut pos, "5i5h");
        });
    }

    #[test]
    fn update_on_white_king_move_matches_refresh() {
        run_with_large_stack(|| {
            let mut pos = start_pos();
            let mv = Move::new_from_usi_str("7g7f", &pos).expect("legal move");
            let gives_check = pos.gives_check(mv);
            pos.do_move(mv, gives_check);
            assert_king_move_update_matches_refresh(&mut pos, "5a5b");
        });
    }

    #[test]
    fn refresh_equals_incremental_after_move_sequence() {
        run_with_large_stack(|| {
            let net = synthetic_net();
            let mut pos = start_pos();

            let usi_moves = ["7g7f", "3c3d", "8h2b+", "3a2b"];

            let mut acc = Accumulator::zeroed();
            refresh(&mut acc, net, &pos);

            for usi in usi_moves.iter() {
                let mv = Move::new_from_usi_str(usi, &pos).expect("legal move");
                assert!(!features::requires_full_refresh(&pos, mv, Color::BLACK));
                assert!(!features::requires_full_refresh(&pos, mv, Color::WHITE));

                let (removed_black, added_black) = features::feature_diff(&mut pos, mv, Color::BLACK);
                let (removed_white, added_white) = features::feature_diff(&mut pos, mv, Color::WHITE);

                let prev = acc.clone();
                update_on_move(
                    &prev,
                    &mut acc,
                    net,
                    &removed_black,
                    &added_black,
                    &removed_white,
                    &added_white,
                );
                assert!(acc.computed);

                let gives_check = pos.gives_check(mv);
                pos.do_move(mv, gives_check);
            }

            let mut expected = Accumulator::zeroed();
            refresh(&mut expected, net, &pos);
            assert_eq!(acc.us, expected.us, "BLACK perspective: incremental != fresh refresh");
            assert_eq!(acc.them, expected.them, "WHITE perspective: incremental != fresh refresh");
        });
    }
}
