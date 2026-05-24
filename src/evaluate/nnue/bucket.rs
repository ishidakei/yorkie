use super::types::LAYER_STACKS;
use crate::position::Position;
use crate::types::{Color, Rank};

const F_RANK_TO_INDEX: [usize; 9] = [0, 0, 0, 3, 3, 3, 6, 6, 6];
const E_RANK_TO_INDEX: [usize; 9] = [0, 0, 0, 1, 1, 1, 2, 2, 2];

pub fn select(pos: &Position) -> usize {
    let stm = pos.side_to_move();
    let f_king = pos.king_square(stm);
    let e_king = pos.king_square(stm.inverse());

    let f_rank = match stm {
        Color::BLACK => Rank::new(f_king),
        _ => Rank::new(f_king).inverse(),
    };
    let e_rank = match stm {
        Color::BLACK => Rank::new(e_king).inverse(),
        _ => Rank::new(e_king),
    };

    let idx = F_RANK_TO_INDEX[f_rank.0 as usize] + E_RANK_TO_INDEX[e_rank.0 as usize];
    idx.min(LAYER_STACKS - 1)
}

#[cfg(test)]
mod tests {
    use super::super::test_fixtures::run_with_large_stack;
    use super::*;

    fn ya_bucket_index(sfen: &str) -> Option<usize> {
        const K_TO_IDX: [usize; 9] = [0, 0, 0, 3, 3, 3, 6, 6, 6];
        const E_TO_IDX: [usize; 9] = [0, 0, 0, 1, 1, 1, 2, 2, 2];

        let mut parts = sfen.split_whitespace();
        let board = parts.next()?;
        let stm = parts.next()?;

        let mut black_king_rank: Option<usize> = None;
        let mut white_king_rank: Option<usize> = None;

        for (rank_idx, rank_str) in board.split('/').enumerate() {
            if rank_idx >= 9 {
                return None;
            }
            let mut file_idx = 0usize;
            let mut chars = rank_str.chars().peekable();
            while let Some(c) = chars.next() {
                match c {
                    '1'..='9' => file_idx += (c as u8 - b'0') as usize,
                    '+' => {
                        let _ = chars.next();
                        file_idx += 1;
                    }
                    'K' => {
                        black_king_rank = Some(rank_idx);
                        file_idx += 1;
                    }
                    'k' => {
                        white_king_rank = Some(rank_idx);
                        file_idx += 1;
                    }
                    _ => file_idx += 1,
                }
            }
            if file_idx > 9 {
                return None;
            }
        }

        let bk = black_king_rank?;
        let wk = white_king_rank?;
        if bk > 8 || wk > 8 {
            return None;
        }

        let (f_rank, e_rank) = match stm {
            "b" => (bk, 8 - wk),
            "w" => (8 - wk, bk),
            _ => return None,
        };
        Some((K_TO_IDX[f_rank] + E_TO_IDX[e_rank]).min(LAYER_STACKS - 1))
    }

    fn select_from_sfen(sfen: &str) -> usize {
        let pos = Position::new_from_sfen(sfen).expect("fixture SFEN must parse");
        select(&pos)
    }

    const BUCKET_FIXTURES: &[(usize, &str)] = &[
        (0, "8K/9/9/9/9/9/9/9/8k b - 1"),
        (1, "8K/9/9/9/9/8k/9/9/9 b - 1"),
        (2, "8K/9/k8/9/9/9/9/9/9 b - 1"),
        (3, "9/9/9/8K/9/9/9/9/8k b - 1"),
        (4, "9/9/9/8K/9/8k/9/9/9 b - 1"),
        (5, "9/9/k8/8K/9/9/9/9/9 b - 1"),
        (6, "9/9/9/9/9/9/8K/9/8k b - 1"),
        (7, "9/9/9/9/9/k8/8K/9/9 b - 1"),
        (8, "9/9/8k/9/9/9/8K/9/9 b - 1"),
    ];

    #[test]
    fn select_matches_oracle_on_each_bucket() {
        run_with_large_stack(|| {
            for (expected, sfen) in BUCKET_FIXTURES {
                let oracle = ya_bucket_index(sfen).unwrap_or_else(|| panic!("oracle could not parse `{sfen}`"));
                assert_eq!(
                    oracle, *expected,
                    "fixture `{sfen}` claims bucket {expected} but the SFEN-oracle says {oracle}"
                );
                let got = select_from_sfen(sfen);
                assert_eq!(
                    got, *expected,
                    "select(pos) mismatch on `{sfen}`: expected {expected}, got {got}"
                );
            }
        });
    }

    #[test]
    fn select_returns_in_range() {
        run_with_large_stack(|| {
            for (_, sfen) in BUCKET_FIXTURES {
                let got = select_from_sfen(sfen);
                assert!(got < LAYER_STACKS, "bucket {got} >= LAYER_STACKS ({LAYER_STACKS})");
            }
        });
    }

    #[test]
    fn select_white_to_move_mirrors_ranks() {
        run_with_large_stack(|| {
            let sfen = "8k/9/9/9/9/9/9/9/8K w - 1";
            assert_eq!(ya_bucket_index(sfen), Some(8));
            assert_eq!(select_from_sfen(sfen), 8);
        });
    }

    #[test]
    fn select_black_stm_rank_boundaries() {
        run_with_large_stack(|| {
            let just_below = "9/9/8K/9/9/9/9/9/8k b - 1";
            let just_above = "9/9/9/8K/9/9/9/9/8k b - 1";
            assert_eq!(select_from_sfen(just_below), 0);
            assert_eq!(select_from_sfen(just_above), 3);

            let r6 = "9/9/9/9/9/8K/9/9/8k b - 1";
            let r7 = "9/9/9/9/9/9/8K/9/8k b - 1";
            assert_eq!(select_from_sfen(r6), 3);
            assert_eq!(select_from_sfen(r7), 6);
        });
    }

    #[test]
    fn select_handles_real_startpos() {
        run_with_large_stack(|| {
            let startpos = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
            assert_eq!(ya_bucket_index(startpos), Some(8));
            assert_eq!(select_from_sfen(startpos), 8);
        });
    }
}
