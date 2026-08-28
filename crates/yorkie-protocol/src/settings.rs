//! The one place that decides where a setting's value comes from.
//!
//! There is exactly one answer, in every build: the compile-time constant
//! [`crate::config`] generated from the TOML config. The optimizer sees a
//! literal, the engine cannot be reconfigured while it is running, and it never
//! reads a constant through a variable. No build has a runtime option surface,
//! so there is nothing else a setting could come from — no declaration table, no
//! `setoption`, no option-override file, no profile file.
//!
//! The accessors stay because the driver reads settings by name
//! (`self.settings.usi_hash()`), and keeping the name → constant mapping in one
//! file is what makes "which constant does `USI_Hash` mean" answerable in one
//! place.
//!
//! # Profile-dependent book options
//!
//! Eight book settings come in two halves: four belong to the V1 book profile
//! (`NarrowBook`, `ConsiderBookMoveCount`, `BookEvalDiff`, `BookDepthLimit`) and
//! four to V2 (`BookEvalBlackDiff`, `BookEvalWhiteDiff`, `BookDepthBlackLimit`,
//! `BookDepthWhiteLimit`). The config schema is one fixed key set, so both
//! halves are always present; the accessors for those eight mask against
//! [`crate::config::BOOK_OPTIONS_V2`], and the half the selected profile does
//! not own reads as its type's zero. `BOOK_OPTIONS_V2` is itself a constant, so
//! the mask folds away with everything else.

/// The engine's settings: the generated constants, addressed by name.
///
/// A zero-sized type — there is no settings state to hold. It exists so the
/// driver has one object to ask, and so the name → constant mapping lives in
/// one file rather than being spelled out at each call site.
pub(crate) struct Settings;

impl Settings {
    /// The settings for a session. Always the same ones: the constants this
    /// binary was built with.
    pub(crate) fn new() -> Self {
        Self
    }

    /// The worker-pool size (`Threads`), always at least 1.
    pub(crate) fn threads(&self) -> usize {
        crate::config::THREADS.max(1) as usize
    }

    /// Whether the book options follow the V2 profile — the reference
    /// `OptionsMap::book_options_v2()`.
    pub(crate) fn book_options_v2(&self) -> bool {
        crate::config::BOOK_OPTIONS_V2
    }
}

/// Define the `spin`-valued accessors: each is its generated constant.
macro_rules! spin_accessors {
    ($( $(#[$attr:meta])* $name:ident => $konst:ident; )*) => {
        impl Settings {
            $(
                $(#[$attr])*
                pub(crate) fn $name(&self) -> i64 {
                    crate::config::$konst
                }
            )*
        }
    };
}

/// Define the `check`-valued accessors.
macro_rules! check_accessors {
    ($( $(#[$attr:meta])* $name:ident => $konst:ident; )*) => {
        impl Settings {
            $(
                $(#[$attr])*
                pub(crate) fn $name(&self) -> bool {
                    crate::config::$konst
                }
            )*
        }
    };
}

/// Define the string-valued accessors.
macro_rules! text_accessors {
    ($( $(#[$attr:meta])* $name:ident => $konst:ident; )*) => {
        impl Settings {
            $(
                $(#[$attr])*
                pub(crate) fn $name(&self) -> &str {
                    crate::config::$konst
                }
            )*
        }
    };
}

spin_accessors! {
    /// Transposition-table size in MiB (`USI_Hash`).
    usi_hash => USI_HASH;
    /// Principal variations reported per search (`MultiPV`).
    multi_pv => MULTI_PV;
    /// Ply past which the book is no longer consulted (`BookMoves`).
    book_moves => BOOK_MOVES;
    /// Percentage chance of ignoring the book on a probe (`BookIgnoreRate`).
    book_ignore_rate => BOOK_IGNORE_RATE;
    /// Absolute book-value floor with Black to move (`BookEvalBlackLimit`).
    book_eval_black_limit => BOOK_EVAL_BLACK_LIMIT;
    /// Absolute book-value floor with White to move (`BookEvalWhiteLimit`).
    book_eval_white_limit => BOOK_EVAL_WHITE_LIMIT;
    /// How many book moves to emit as a PV (`BookPvMoves`).
    book_pv_moves => BOOK_PV_MOVES;
    /// Per-`go` search-depth ceiling, `0` unlimited (`DepthLimit`).
    depth_limit => DEPTH_LIMIT;
    /// Per-`go` node ceiling, `0` unlimited (`NodesLimit`).
    nodes_limit => NODES_LIMIT;
    /// Ply past which the search adjudicates a draw (`MaxMovesToDraw`).
    max_moves_to_draw => MAX_MOVES_TO_DRAW;
    /// PV-output throttle in milliseconds (`PvInterval`).
    pv_interval => PV_INTERVAL;
    /// Draw score with Black to move, in centipawns (`DrawValueBlack`).
    draw_value_black => DRAW_VALUE_BLACK;
    /// Draw score with White to move, in centipawns (`DrawValueWhite`).
    draw_value_white => DRAW_VALUE_WHITE;
    /// Post-search resign threshold in centipawns (`ResignValue`).
    resign_value => RESIGN_VALUE;
    /// Average GUI round-trip margin in milliseconds (`NetworkDelay`).
    network_delay => NETWORK_DELAY;
    /// Worst-case GUI round-trip margin in milliseconds (`NetworkDelay2`).
    network_delay2 => NETWORK_DELAY2;
    /// Floor on a move's optimum time in milliseconds (`MinimumThinkingTime`).
    minimum_thinking_time => MINIMUM_THINKING_TIME;
    /// Percentage multiplier on the optimum time (`SlowMover`).
    slow_mover => SLOW_MOVER;
}

check_accessors! {
    /// Consult the book at all (`USI_OwnBook`).
    usi_own_book => USI_OWN_BOOK;
    /// Stream the book from disk rather than loading it (`BookOnTheFly`).
    book_on_the_fly => BOOK_ON_THE_FLY;
    /// Match book positions ignoring their recorded ply (`IgnoreBookPly`).
    ignore_book_ply => IGNORE_BOOK_PLY;
    /// Also probe the mirrored position (`FlippedBook`).
    flipped_book => FLIPPED_BOOK;
    /// Collect each PV from the transposition table (`ConsiderationMode`).
    consideration_mode => CONSIDERATION_MODE;
    /// Emit a PV on a fail-high / fail-low (`OutputFailLHPV`).
    output_fail_lh_pv => OUTPUT_FAIL_LH_PV;
    /// Also consider the suppressed non-promoting moves
    /// (`GenerateAllLegalMoves`).
    generate_all_legal_moves => GENERATE_ALL_LEGAL_MOVES;
    /// Use the clock up to each whole second (`RoundUpToFullSecond`).
    round_up_to_full_second => ROUND_UP_TO_FULL_SECOND;
    /// Ponder toggle feeding the optimum-time bonus (`USI_Ponder`).
    usi_ponder => USI_PONDER;
    /// Stochastic-ponder toggle (`Stochastic_Ponder`).
    stochastic_ponder => STOCHASTIC_PONDER;
}

text_accessors! {
    /// Directory holding `nn.bin` (`EvalDir`).
    eval_dir => EVAL_DIR;
    /// Book file name, `no_book` for bookless (`BookFile`).
    book_file => BOOK_FILE;
    /// Book directory (`BookDir`).
    book_dir => BOOK_DIR;
    /// Entering-king declaration rule (`EnteringKingRule`).
    entering_king_rule => ENTERING_KING_RULE;
    /// NUMA mapping / binding policy (`NumaPolicy`).
    numa_policy => NUMA_POLICY;
}

// --- Profile-dependent book options (see the module docs for the masking). ---

impl Settings {
    /// V1 only (`NarrowBook`): drop book moves played only once. `false` under
    /// V2, which does not own this setting.
    pub(crate) fn narrow_book(&self) -> bool {
        !crate::config::BOOK_OPTIONS_V2 && crate::config::NARROW_BOOK
    }

    /// V1 only (`ConsiderBookMoveCount`): weight book selection by play count.
    pub(crate) fn consider_book_move_count(&self) -> bool {
        !crate::config::BOOK_OPTIONS_V2 && crate::config::CONSIDER_BOOK_MOVE_COUNT
    }

    /// V1 only (`BookEvalDiff`): the un-split book value-difference filter.
    pub(crate) fn book_eval_diff(&self) -> i64 {
        if crate::config::BOOK_OPTIONS_V2 {
            0
        } else {
            crate::config::BOOK_EVAL_DIFF
        }
    }

    /// V1 only (`BookDepthLimit`): the un-split book depth floor.
    pub(crate) fn book_depth_limit(&self) -> i64 {
        if crate::config::BOOK_OPTIONS_V2 {
            0
        } else {
            crate::config::BOOK_DEPTH_LIMIT
        }
    }

    /// V2 only (`BookEvalBlackDiff`): value-difference filter, Black to move.
    pub(crate) fn book_eval_black_diff(&self) -> i64 {
        if crate::config::BOOK_OPTIONS_V2 {
            crate::config::BOOK_EVAL_BLACK_DIFF
        } else {
            0
        }
    }

    /// V2 only (`BookEvalWhiteDiff`): value-difference filter, White to move.
    pub(crate) fn book_eval_white_diff(&self) -> i64 {
        if crate::config::BOOK_OPTIONS_V2 {
            crate::config::BOOK_EVAL_WHITE_DIFF
        } else {
            0
        }
    }

    /// V2 only (`BookDepthBlackLimit`): book depth floor, Black to move.
    pub(crate) fn book_depth_black_limit(&self) -> i64 {
        if crate::config::BOOK_OPTIONS_V2 {
            crate::config::BOOK_DEPTH_BLACK_LIMIT
        } else {
            0
        }
    }

    /// V2 only (`BookDepthWhiteLimit`): book depth floor, White to move.
    pub(crate) fn book_depth_white_limit(&self) -> i64 {
        if crate::config::BOOK_OPTIONS_V2 {
            crate::config::BOOK_DEPTH_WHITE_LIMIT
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The masking above reproduces what an engine built under the other book
    /// profile would not see: the half of the book group the active profile does
    /// not own reads as its type's zero, and the half it does own reads its
    /// configured value.
    ///
    /// Written against `config::BOOK_OPTIONS_V2` rather than against a fixed
    /// profile, so it holds for a binary built from either kind of config file.
    #[test]
    fn profile_dependent_book_options_mask_to_their_types_zero() {
        use crate::config;

        let s = Settings::new();
        assert_eq!(s.book_options_v2(), config::BOOK_OPTIONS_V2);

        if config::BOOK_OPTIONS_V2 {
            // V1-only: not owned under V2.
            assert!(!s.narrow_book());
            assert!(!s.consider_book_move_count());
            assert_eq!(s.book_eval_diff(), 0);
            assert_eq!(s.book_depth_limit(), 0);
            // V2-only: live.
            assert_eq!(s.book_eval_black_diff(), config::BOOK_EVAL_BLACK_DIFF);
            assert_eq!(s.book_eval_white_diff(), config::BOOK_EVAL_WHITE_DIFF);
            assert_eq!(s.book_depth_black_limit(), config::BOOK_DEPTH_BLACK_LIMIT);
            assert_eq!(s.book_depth_white_limit(), config::BOOK_DEPTH_WHITE_LIMIT);
        } else {
            // V1-only: live.
            assert_eq!(s.narrow_book(), config::NARROW_BOOK);
            assert_eq!(
                s.consider_book_move_count(),
                config::CONSIDER_BOOK_MOVE_COUNT
            );
            assert_eq!(s.book_eval_diff(), config::BOOK_EVAL_DIFF);
            assert_eq!(s.book_depth_limit(), config::BOOK_DEPTH_LIMIT);
            // V2-only: not owned under V1.
            assert_eq!(s.book_eval_black_diff(), 0);
            assert_eq!(s.book_eval_white_diff(), 0);
            assert_eq!(s.book_depth_black_limit(), 0);
            assert_eq!(s.book_depth_white_limit(), 0);
        }

        // Settings owned by both profiles are never masked.
        assert_eq!(s.book_moves(), config::BOOK_MOVES);
        assert_eq!(s.book_eval_white_limit(), config::BOOK_EVAL_WHITE_LIMIT);
        assert_eq!(s.flipped_book(), config::FLIPPED_BOOK);
    }
}
