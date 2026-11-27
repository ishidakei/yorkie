//! The search parameters a tuning run varies: the pruning, reduction and
//! extension constants that shape the tree, the history updates and move
//! scores that order it, and the coefficients of the time allocation that
//! decide how long a search runs, each named once here and read through
//! `Param::get` wherever the search uses it.
//!
//! Every parameter is an integer. A coefficient the arithmetic uses as a
//! floating-point number is registered scaled by a power of ten and read
//! through `Param::scaled`: the default divided by its scale is the
//! correctly rounded quotient, which is the same `f64` the decimal literal
//! spells, so the default allocates bit for bit as the literal would.
//!
//! In every build but `tune` a parameter is a `const`, and `get` on it is the
//! constant: the search compiles to what a literal at the use site would give.
//! In a `tune` build each one is a `static` that `apply` writes once at
//! startup from a `name=value,name=value` list, before any search thread
//! exists; the names left out keep the defaults below, so a `tune` build given
//! no list searches exactly as the plain build of the same config.
//!
//! Each parameter carries the range of values the search tolerates: the
//! arithmetic it enters stays inside `i32` over the whole depth range, and the
//! invariant the code around it relies on still holds. That range is wider than
//! any range worth tuning over; it only keeps a mistyped value from producing a
//! search that hangs or overflows.

#[cfg(feature = "tune")]
use std::sync::atomic::{AtomicI32, Ordering};

#[cfg(feature = "tune")]
use yorkie_state::text::{TextWriter, atoi_i64};

/// One search parameter, a compile-time constant.
#[cfg(not(feature = "tune"))]
#[derive(Clone, Copy)]
pub(crate) struct Param(i32);

#[cfg(not(feature = "tune"))]
impl Param {
    const fn new(value: i32) -> Self {
        Self(value)
    }

    #[inline(always)]
    pub(crate) const fn get(self) -> i32 {
        self.0
    }

    /// The value divided by `scale`, for a coefficient registered scaled.
    #[inline(always)]
    pub(crate) fn scaled(self, scale: f64) -> f64 {
        f64::from(self.0) / scale
    }
}

/// One search parameter, written at startup by [`apply`].
///
/// A relaxed load is enough to read it: the only store happens on the main
/// thread before any search thread is spawned, and spawning a thread
/// synchronises with everything the spawning thread did before it, so every
/// search thread observes the stored value. Nothing writes it afterwards.
#[cfg(feature = "tune")]
pub(crate) struct Param(AtomicI32);

#[cfg(feature = "tune")]
impl Param {
    const fn new(value: i32) -> Self {
        Self(AtomicI32::new(value))
    }

    #[inline(always)]
    pub(crate) fn get(&self) -> i32 {
        self.0.load(Ordering::Relaxed)
    }

    /// The value divided by `scale`, for a coefficient registered scaled.
    #[inline(always)]
    pub(crate) fn scaled(&self, scale: f64) -> f64 {
        f64::from(self.get()) / scale
    }
}

/// One registered parameter: its name in a tuning list, the parameter itself,
/// and the values the search tolerates.
#[cfg(feature = "tune")]
struct Entry {
    name: &'static [u8],
    param: &'static Param,
    min: i32,
    max: i32,
}

macro_rules! search_params {
    ($(
        $(#[doc = $doc:literal])*
        $id:ident = $name:literal, $default:literal, $min:literal..=$max:literal;
    )+) => {
        $(
            $(#[doc = $doc])*
            #[cfg(not(feature = "tune"))]
            pub(crate) const $id: Param = Param::new($default);
            $(#[doc = $doc])*
            #[cfg(feature = "tune")]
            pub(crate) static $id: Param = Param::new($default);
            const _: () = assert!($min <= $default && $default <= $max);
            // The read the search makes, evaluated where only a compile-time
            // constant is accepted: the build fails if a parameter is not one.
            #[cfg(not(feature = "tune"))]
            const _: i32 = $id.get();
        )+

        /// Every parameter a tuning list can name.
        #[cfg(feature = "tune")]
        static REGISTRY: &[Entry] = &[$(
            Entry { name: $name, param: &$id, min: $min, max: $max },
        )+];
    };
}

search_params! {
    /// Razoring: the margin below alpha at depth 0, before the depth term.
    RAZOR_BASE = b"razor_base", 502, 0..=2000;
    /// Razoring: the margin's coefficient on the squared depth.
    RAZOR_DEPTH = b"razor_depth", 306, 0..=1000;
    /// Futility pruning: the per-ply margin with a transposition-table hit. The
    /// margin without one is this less `futility_tt_miss`, at most 39, so the
    /// floor keeps it positive.
    FUTILITY_MULT = b"futility_mult", 76, 40..=500;
    /// Null-move pruning: how far the static evaluation may sit below beta,
    /// before the depth and improving terms, for the null move to be tried.
    NMP_MARGIN = b"nmp_margin", 378, 0..=2000;
    /// Null-move pruning: the depth reduction before the `depth / 3` term. At
    /// least one, so the null-move search is shallower than the node.
    NMP_REDUCTION = b"nmp_reduction", 7, 1..=20;
    /// ProbCut: how far above beta the cut is sought, before the improving
    /// term, `probcut_improving`, at most 100, is taken off it. The floor keeps
    /// the ProbCut beta above beta.
    PROBCUT_MARGIN = b"probcut_margin", 224, 101..=2000;
    /// The transposition-table ProbCut at the head of the move loop: how far
    /// above beta a stored lower bound must sit to cut the node.
    SMALL_PROBCUT_MARGIN = b"small_probcut_margin", 416, 0..=2000;
    /// Singular extension: the singular beta's margin below the stored value,
    /// per 55 plies of depth, before the off-PV term.
    SINGULAR_MARGIN = b"singular_margin", 60, 0..=1000;
    /// Late-move reduction: the constant term of the base reduction, in
    /// 1024ths of a ply.
    LMR_BASE = b"lmr_base", 1133, -5000..=10000;
    /// Late-move reduction: the weight of the window's width against the root
    /// window's, subtracted from the base reduction.
    LMR_DELTA = b"lmr_delta", 585, 0..=5000;
    /// Late-move reduction: the extra reduction at a cut node, in 1024ths of a
    /// ply.
    LMR_CUT_NODE = b"lmr_cut_node", 3611, 0..=10000;
    /// Quiet-move futility pruning at shallow depth: the margin per ply of the
    /// reduced depth.
    QUIET_FUTILITY_DEPTH = b"quiet_futility_depth", 120, 0..=1000;
    /// Quiescence search: the futility margin over the static evaluation.
    QS_FUTILITY_MARGIN = b"qs_futility_margin", 328, 0..=2000;
    /// Quiescence search: the static-exchange threshold below which a capture
    /// is skipped.
    QS_SEE_MARGIN = b"qs_see_margin", -73, -2000..=2000;
    /// The aspiration window's half-width before the root score's spread is
    /// added. At least one, so each fail widens the window.
    ASPIRATION_DELTA = b"aspiration_delta", 5, 1..=1000;

    // The divisors of the formulas above. A large one is registered as it is.
    // A small one, `x / d`, is read as `x * 1024 / k` with the default
    // `k = 1024 * d`: for integer `x` that is the same quotient as `x / d`,
    // and a step of one on `k` moves the quotient by a thousandth rather than
    // by a whole share of `d`. Every lower bound keeps the divisor positive.

    /// The aspiration window's half-width: the divisor of the root score's
    /// mean square. A mean square is at most the infinite score's square, so
    /// the floor keeps the half-width near a million.
    ASPIRATION_SPREAD_DIV = b"aspiration_spread_div", 9000, 1000..=100000;
    /// The aspiration window: each fail widens the half-width by its own
    /// `1024 / this` (a third at the default).
    ASPIRATION_WIDEN_DIV = b"aspiration_widen_div", 3072, 1024..=16384;
    /// The aspiration loop: a root depth searched again is shallower by
    /// `3 * (repeats + 1) * 1024 / this` plies (three quarters a repeat at the
    /// default).
    SEARCH_AGAIN_DIV = b"search_again_div", 4096, 1024..=16384;
    /// Futility pruning: the divisor of the correction value's magnitude
    /// added to the margin. A correction value stays under 2^27 in
    /// magnitude, so the floor keeps the term under 14000.
    FUTILITY_CORRECTION_DIV = b"futility_correction_div", 180600, 10000..=2000000;
    /// Null-move pruning: the depth reduction's depth term is
    /// `depth * 1024 / this` (a third of the depth at the default).
    NMP_DEPTH_DIV = b"nmp_depth_div", 3072, 512..=16384;
    /// Null-move pruning: the verification search disables the null move for
    /// `3 * (depth - R) * 1024 / this` plies (three quarters of the reduced
    /// depth at the default).
    NMP_VERIFY_DIV = b"nmp_verify_div", 4096, 1024..=16384;
    /// Singular extension: the divisor of the margin times the depth.
    SINGULAR_MARGIN_DIV = b"singular_margin_div", 55, 20..=500;
    /// Singular extension: the excluded-move search's depth is
    /// `new_depth * 1024 / this` (half at the default). The floor keeps it at
    /// most the node's own.
    SINGULAR_DEPTH_DIV = b"singular_depth_div", 2048, 1024..=8192;
    /// Quiet-move pruning at shallow depth: the divisor of the history added
    /// to the reduced depth. The history sum stays under 2^17 in magnitude,
    /// so the floor keeps the reduced depth's square, and the SEE threshold it
    /// sets, inside `i32`.
    LMR_HISTORY_DIV = b"lmr_history_div", 3220, 100..=100000;
    /// Late-move reduction: the divisor of the correction value's magnitude
    /// taken off the reduction. The floor keeps the reduction small enough
    /// that its all-node scaling stays inside `i32`.
    LMR_CORRECTION_DIV = b"lmr_correction_div", 25600, 1000..=1000000;

    // The time allocation. None of these can make a search outlast the clock:
    // the optimum and the maximum are clamped to the remaining time after the
    // terms below, and the search loop's per-iteration target is capped at the
    // maximum, so a range only has to keep the arithmetic defined (no zero
    // divisor, no inverted clamp, no factor that turns the target into NaN).

    /// The planning horizon in plies once out of the opening. With the
    /// smallest opening terms below it still leaves at least two own moves to
    /// plan for, so the spend-everything branch for the last move before the
    /// draw horizon is reached only there.
    TM_MOVE_HORIZON = b"tm_move_horizon", 160, 100..=400;
    /// On a clock with neither increment nor byoyomi, how many plies the
    /// horizon is wider by at the start; it narrows by one a ply to the base.
    TM_OPENING_FORFEIT = b"tm_opening_forfeit", 40, 0..=100;
    /// On a clock with an increment or byoyomi, how many plies the horizon is
    /// wider by at the start, before the narrowing below.
    TM_OPENING_INC = b"tm_opening_inc", 20, 0..=100;
    /// On a clock with an increment or byoyomi, the ply up to which the
    /// horizon narrows by one a ply. The ceiling keeps the narrowed horizon at
    /// least four plies above zero with the smallest base horizon.
    TM_OPENING_INC_PLIES = b"tm_opening_inc_plies", 80, 0..=96;
    /// The maximum time's ratio to the per-move share, in thousandths. At
    /// least one, so the uncapped maximum is never below the optimum.
    TM_MAX_RATIO = b"tm_max_ratio", 5000, 1000..=20000;
    /// The share of the estimated remaining time the maximum is capped at, in
    /// thousandths.
    TM_MAX_SHARE = b"tm_max_share", 300, 100..=600;
    /// The falling-evaluation factor: its constant term, in thousandths of a
    /// percent.
    TM_FALLING_BASE = b"tm_falling_base", 11325, 0..=50000;
    /// The falling-evaluation factor: the weight of the drop from the previous
    /// move's average score, in thousandths of a percent per unit of score.
    TM_FALLING_AVERAGE = b"tm_falling_average", 2115, 0..=10000;
    /// The falling-evaluation factor: the weight of the drop from the score
    /// four iterations back, in thousandths of a percent per unit of score.
    TM_FALLING_ITERATION = b"tm_falling_iteration", 987, 0..=10000;
    /// The falling-evaluation factor's floor, in ten-thousandths. Positive,
    /// and at most the ceiling's floor, so the clamp is never inverted.
    TM_FALLING_MIN = b"tm_falling_min", 5688, 1000..=10000;
    /// The falling-evaluation factor's ceiling, in ten-thousandths.
    TM_FALLING_MAX = b"tm_falling_max", 15698, 10000..=30000;
    /// The best-move-stability curve: its steepness per ply of depth, in
    /// ten-thousandths. Positive, so the curve rises with depth.
    TM_REDUCTION_K = b"tm_reduction_k", 5189, 1000..=20000;
    /// The best-move-stability curve: how many plies past the depth at which
    /// the best move last changed its midpoint sits, in hundredths.
    TM_REDUCTION_CENTER = b"tm_reduction_center", 1157, 0..=3000;
    /// The best-move-stability curve: its floor, in thousandths. Positive, so
    /// the reduction it divides stays finite.
    TM_REDUCTION_BASE = b"tm_reduction_base", 723, 100..=2000;
    /// The best-move-stability curve: its height above the floor, in
    /// thousandths.
    TM_REDUCTION_AMPLITUDE = b"tm_reduction_amplitude", 790, 0..=3000;
    /// The term added to the previous move's curve value before the
    /// reduction divides by this move's, in thousandths.
    TM_REDUCTION_PREVIOUS = b"tm_reduction_previous", 1455, 0..=5000;
    /// The best-move-instability factor's constant term, in thousandths.
    /// Positive, so the target time is never zero.
    TM_INSTABILITY_BASE = b"tm_instability_base", 1040, 100..=3000;
    /// The best-move-instability factor's weight on the aged count of
    /// best-move changes per thread, in ten-thousandths.
    TM_INSTABILITY_WEIGHT = b"tm_instability_weight", 18956, 0..=60000;
    /// The factor on the target time when most of the nodes went under the
    /// best move, in thousandths. Positive, so the target time is never zero.
    TM_EFFORT_FACTOR = b"tm_effort_factor", 666, 100..=1500;
    /// The share of the target time an iteration may start under and still
    /// deepen, in thousandths; past it the next iteration repeats the depth.
    TM_DEEPEN_SHARE = b"tm_deepen_share", 503, 100..=1000;

    // The pruning constants beside the ones above: the terms of the futility,
    // null-move, ProbCut and shallow-depth margins, and the depth and move-count
    // gates that switch each pruning on. A gate is a small integer and moves by
    // one; its floor keeps the depth the pruning searches or stores at least what
    // the code after it relies on.

    /// Futility pruning: how much smaller the per-ply margin is without a
    /// transposition-table hit. At most 39, below `futility_mult`'s floor.
    FUTILITY_TT_MISS = b"futility_tt_miss", 21, 0..=39;
    /// Futility pruning: the margin taken off when the evaluation improves, in
    /// 1024ths of the per-ply margin.
    FUTILITY_IMPROVING = b"futility_improving", 2686, 0..=8192;
    /// Futility pruning: the margin taken off when the opponent's evaluation
    /// worsens, in 1024ths of the per-ply margin.
    FUTILITY_WORSENING = b"futility_worsening", 362, 0..=4096;
    /// Futility pruning: the depth it applies below.
    FUTILITY_MAX_DEPTH = b"futility_max_depth", 15, 1..=256;
    /// Null-move pruning: how much the evaluation margin narrows per ply of
    /// depth.
    NMP_DEPTH_MARGIN = b"nmp_depth_margin", 16, 0..=100;
    /// Null-move pruning: how much the evaluation margin narrows when the
    /// evaluation improves.
    NMP_IMPROVING = b"nmp_improving", 53, 0..=500;
    /// Null-move pruning: the depth from which a null-move cutoff is verified
    /// by a reduced search of the node itself.
    NMP_VERIFY_DEPTH = b"nmp_verify_depth", 16, 0..=256;
    /// Internal iterative reduction: the depth from which a node without a
    /// transposition-table move is searched a ply shallower. At least 2, so the
    /// reduced node is still a full-width one.
    IIR_MIN_DEPTH = b"iir_min_depth", 6, 2..=256;
    /// Internal iterative reduction: the largest reduction of the move into the
    /// node with which it still applies.
    IIR_MAX_PRIOR_REDUCTION = b"iir_max_prior_reduction", 3, -10..=64;
    /// The prior reduction from which a node whose opponent's evaluation did not
    /// worsen is searched a ply deeper.
    HINDSIGHT_EXTEND = b"hindsight_extend", 3, 0..=64;
    /// The prior reduction from which a node whose two evaluations sum above
    /// `hindsight_eval_sum` is searched a ply shallower.
    HINDSIGHT_REDUCE = b"hindsight_reduce", 2, 0..=64;
    /// The sum of this node's and its parent's static evaluations above which
    /// the node is searched a ply shallower after a reduced move.
    HINDSIGHT_EVAL_SUM = b"hindsight_eval_sum", 173, -2000..=2000;
    /// ProbCut: how far the margin narrows when the evaluation improves. At
    /// most 100, below `probcut_margin`'s floor.
    PROBCUT_IMPROVING = b"probcut_improving", 61, 0..=100;
    /// ProbCut: the depth from which it applies. At least 3, so with the
    /// largest reduction the depth it stores is at least -1.
    PROBCUT_MIN_DEPTH = b"probcut_min_depth", 3, 3..=256;
    /// ProbCut: how many plies shallower than the node the verification search
    /// is.
    PROBCUT_REDUCTION = b"probcut_reduction", 4, 1..=5;
    /// The transposition-table ProbCut: how much shallower than the node the
    /// stored lower bound may be.
    SMALL_PROBCUT_DEPTH = b"small_probcut_depth", 4, 0..=64;
    /// The transposition-table cutoff: the depth above which an entry cuts a
    /// node whose kind does not match the bound.
    TT_CUTOFF_DEPTH = b"tt_cutoff_depth", 5, 0..=256;
    /// Move-count pruning: the constant term of the move count past which the
    /// remaining quiet moves are skipped, before the squared depth.
    MOVE_COUNT_PRUNE_BASE = b"move_count_prune_base", 3, 0..=100;
    /// Futility pruning of a capture or check at shallow depth: the margin's
    /// constant term.
    CAPTURE_FUTILITY_BASE = b"capture_futility_base", 218, -1000..=2000;
    /// Futility pruning of a capture or check: the margin per ply of the
    /// reduced depth.
    CAPTURE_FUTILITY_DEPTH = b"capture_futility_depth", 223, 0..=2000;
    /// Futility pruning of a capture or check: the weight of its capture
    /// history, in 1024ths.
    CAPTURE_FUTILITY_HISTORY = b"capture_futility_history", 131, 0..=1024;
    /// Futility pruning of a capture: the reduced depth it applies below.
    CAPTURE_FUTILITY_MAX_DEPTH = b"capture_futility_max_depth", 7, 0..=64;
    /// SEE pruning of a capture or check: the threshold per ply of depth.
    CAPTURE_SEE_DEPTH = b"capture_see_depth", 167, 0..=1000;
    /// SEE pruning of a capture or check: the weight of its capture history in
    /// the threshold, in 1024ths.
    CAPTURE_SEE_HISTORY = b"capture_see_history", 34, 0..=1024;
    /// Quiet-move pruning at shallow depth: the continuation and pawn history
    /// per ply of depth below which a quiet move is skipped.
    CONT_HISTORY_PRUNE = b"cont_history_prune", -4097, -20000..=0;
    /// Quiet-move pruning: the weight of the main history added to the history
    /// that adjusts the reduced depth, in 32nds.
    MAIN_HISTORY_PRUNE = b"main_history_prune", 71, 0..=256;
    /// Quiet-move futility pruning: the margin's constant term.
    QUIET_FUTILITY_BASE = b"quiet_futility_base", 42, -1000..=1000;
    /// Quiet-move futility pruning: the margin added while the node has no
    /// best move.
    QUIET_FUTILITY_NO_BEST = b"quiet_futility_no_best", 151, 0..=1000;
    /// Quiet-move futility pruning: the margin added when the static evaluation
    /// is above alpha.
    QUIET_FUTILITY_EVAL = b"quiet_futility_eval", 86, 0..=1000;
    /// Quiet-move futility pruning: the reduced depth it applies below.
    QUIET_FUTILITY_MAX_DEPTH = b"quiet_futility_max_depth", 13, 0..=64;
    /// SEE pruning of a quiet move: the threshold per squared ply of the
    /// reduced depth.
    QUIET_SEE = b"quiet_see", 25, 0..=200;
    /// The depth above which a move that raises alpha takes two plies off the
    /// node's remaining moves. At least 2, so the depth stays positive.
    ALPHA_RAISE_MIN_DEPTH = b"alpha_raise_min_depth", 2, 2..=256;
    /// The depth below which a move that raises alpha takes two plies off the
    /// node's remaining moves.
    ALPHA_RAISE_MAX_DEPTH = b"alpha_raise_max_depth", 14, 0..=256;
    /// Quiescence search: the move count past which a quiet-looking move that
    /// does not check or recapture is skipped.
    QS_MOVE_COUNT = b"qs_move_count", 2, 0..=100;

    // The late-move reduction's terms, in 1024ths of a ply. The reduction stays
    // under 2^20 in magnitude before the all-node scaling, whose factor is at most
    // five here, so the scaled reduction stays inside `i32`.

    /// The share of the base reduction added when the evaluation does not
    /// improve, in 512ths.
    LMR_NOT_IMPROVING = b"lmr_not_improving", 206, 0..=1024;
    /// The reduction added at a node on a transposition-table PV, before the
    /// move is made.
    LMR_TTPV = b"lmr_ttpv", 1013, 0..=5000;
    /// The reduction taken off at a node on a transposition-table PV.
    LMR_TTPV_BASE = b"lmr_ttpv_base", 2819, 0..=10000;
    /// On a transposition-table PV: the further reduction taken off at a PV
    /// node.
    LMR_TTPV_PV_NODE = b"lmr_ttpv_pv_node", 973, 0..=5000;
    /// On a transposition-table PV: the further reduction taken off when the
    /// stored value is above alpha.
    LMR_TTPV_TT_VALUE = b"lmr_ttpv_tt_value", 905, 0..=5000;
    /// On a transposition-table PV: the further reduction taken off when the
    /// entry is at least as deep as the node.
    LMR_TTPV_TT_DEPTH = b"lmr_ttpv_tt_depth", 935, 0..=5000;
    /// On a transposition-table PV with a deep enough entry: the further
    /// reduction taken off at a cut node.
    LMR_TTPV_CUT_NODE = b"lmr_ttpv_cut_node", 959, 0..=5000;
    /// The reduction added to every move once it is made.
    LMR_OFFSET = b"lmr_offset", 691, -5000..=5000;
    /// The reduction taken off per move searched so far.
    LMR_MOVE_COUNT = b"lmr_move_count", 65, 0..=500;
    /// At a cut node: the further reduction added without a
    /// transposition-table move.
    LMR_CUT_NODE_NO_TT_MOVE = b"lmr_cut_node_no_tt_move", 985, 0..=5000;
    /// The reduction added when the transposition-table move is a capture.
    LMR_TT_CAPTURE = b"lmr_tt_capture", 1054, 0..=5000;
    /// The reduction added when the child's earlier siblings cut off more than
    /// once.
    LMR_CUTOFF = b"lmr_cutoff", 251, 0..=5000;
    /// The further reduction added when they cut off more than twice.
    LMR_CUTOFF_MANY = b"lmr_cutoff_many", 1124, 0..=5000;
    /// The further reduction added for those cutoffs at an all node.
    LMR_CUTOFF_ALL_NODE = b"lmr_cutoff_all_node", 1042, 0..=5000;
    /// The reduction taken off the transposition-table move.
    LMR_TT_MOVE = b"lmr_tt_move", 2239, 0..=10000;
    /// A capture's statistics score: the weight of the captured piece's value,
    /// in 128ths. At the ceiling the score stays under 2^17.
    LMR_CAPTURE_STAT = b"lmr_capture_stat", 863, 0..=4096;
    /// The weight of the move's statistics score taken off the reduction, in
    /// 4096ths.
    LMR_STAT_SCORE = b"lmr_stat_score", 428, 0..=2000;
    /// At an all node, the reduction is scaled up by this over
    /// `256 * depth + lmr_all_node_offset`.
    LMR_ALL_NODE = b"lmr_all_node", 273, 0..=1000;
    /// At an all node: the constant term of the scaling's denominator. Never
    /// negative, so the denominator is at least 256.
    LMR_ALL_NODE_OFFSET = b"lmr_all_node_offset", 260, 0..=2000;
    /// After a reduced search fails high: how far above the best value it must
    /// land for the re-search to go a ply deeper.
    LMR_DEEPER_MARGIN = b"lmr_deeper_margin", 48, -1000..=1000;
    /// After a reduced search fails high: how close to the best value it may
    /// land for the re-search to go a ply shallower.
    LMR_SHALLOWER_MARGIN = b"lmr_shallower_margin", 9, -1000..=1000;
    /// A move searched without a reduction: the reduction added without a
    /// transposition-table move, for the thresholds below.
    LMR_FULL_NO_TT_MOVE = b"lmr_full_no_tt_move", 1057, 0..=5000;
    /// A move searched without a reduction: the reduction above which it is
    /// searched a ply shallower.
    LMR_FULL_THRESHOLD = b"lmr_full_threshold", 4628, 0..=20000;
    /// A move searched without a reduction: the reduction above which it is
    /// searched a further ply shallower, when that leaves more than two.
    LMR_FULL_THRESHOLD_DEEP = b"lmr_full_threshold_deep", 5772, 0..=20000;

    // The singular extension's gates and margins. A correction value is under
    // 2^27 in magnitude and the TT-move history under 8192, so with the divisors'
    // floors every margin stays far inside `i32`.

    /// The depth from which the transposition-table move is tested for
    /// singularity, before the PV term.
    SINGULAR_MIN_DEPTH = b"singular_min_depth", 6, 4..=256;
    /// The plies the minimum depth is raised by on a transposition-table PV.
    SINGULAR_TTPV_DEPTH = b"singular_ttpv_depth", 1, 0..=10;
    /// How much shallower than the node the entry may be for the test.
    SINGULAR_TT_DEPTH_MARGIN = b"singular_tt_depth_margin", 3, 0..=64;
    /// The singular beta's margin added on a transposition-table PV off the
    /// PV, before the depth term.
    SINGULAR_OFF_PV = b"singular_off_pv", 66, 0..=1000;
    /// The divisor of the correction value's magnitude taken off both the
    /// double and the triple extension's margins.
    SINGULAR_CORRECTION_DIV = b"singular_correction_div", 210590, 1000..=10000000;
    /// The double extension's margin below the singular beta: its constant
    /// term.
    DOUBLE_EXT_BASE = b"double_ext_base", -4, -1000..=1000;
    /// The double extension's margin: the term at a PV node.
    DOUBLE_EXT_PV_NODE = b"double_ext_pv_node", 212, 0..=2000;
    /// The double extension's margin: the term taken off for a quiet
    /// transposition-table move.
    DOUBLE_EXT_QUIET = b"double_ext_quiet", 182, 0..=2000;
    /// The double extension's margin: the weight of the TT-move history, over
    /// the divisor below.
    DOUBLE_EXT_TT_MOVE_HISTORY = b"double_ext_tt_move_history", 906, 0..=4000;
    /// The double extension's margin: the divisor of the TT-move history term.
    DOUBLE_EXT_TT_MOVE_HISTORY_DIV = b"double_ext_tt_move_history_div", 116517, 1000..=10000000;
    /// The double extension's margin: the term taken off past the root depth's
    /// ply.
    DOUBLE_EXT_LATE_PLY = b"double_ext_late_ply", 44, 0..=1000;
    /// The triple extension's margin below the singular beta: its constant
    /// term.
    TRIPLE_EXT_BASE = b"triple_ext_base", 73, -1000..=1000;
    /// The triple extension's margin: the term at a PV node.
    TRIPLE_EXT_PV_NODE = b"triple_ext_pv_node", 320, 0..=2000;
    /// The triple extension's margin: the term taken off for a quiet
    /// transposition-table move.
    TRIPLE_EXT_QUIET = b"triple_ext_quiet", 218, 0..=2000;
    /// The triple extension's margin: the term on a transposition-table PV.
    TRIPLE_EXT_TTPV = b"triple_ext_ttpv", 92, 0..=1000;
    /// The triple extension's margin: the term taken off past the root depth's
    /// ply.
    TRIPLE_EXT_LATE_PLY = b"triple_ext_late_ply", 45, 0..=1000;
    /// The extension, negative, of a transposition-table move that is not
    /// singular while its stored value is at least beta.
    NEG_EXT_TT_FAIL_HIGH = b"neg_ext_tt_fail_high", -3, -10..=0;
    /// The extension, negative, of a transposition-table move that is not
    /// singular at a cut node.
    NEG_EXT_CUT_NODE = b"neg_ext_cut_node", -2, -10..=0;

    // The history updates. Every bonus that reaches a continuation-history update
    // stays under 2^20 in magnitude and every weight it meets there under 2048, so
    // the products stay inside `i32`; the tables' gravity clamps each update to the
    // table's bound after that.

    /// The best move's history bonus: per ply of depth.
    STAT_BONUS_DEPTH = b"stat_bonus_depth", 128, 0..=1024;
    /// The best move's history bonus: the constant term.
    STAT_BONUS_OFFSET = b"stat_bonus_offset", -77, -1000..=1000;
    /// The best move's history bonus: the cap on the depth term.
    STAT_BONUS_MAX = b"stat_bonus_max", 1529, 0..=4096;
    /// The best move's history bonus: the term when it is the
    /// transposition-table move.
    STAT_BONUS_TT_MOVE = b"stat_bonus_tt_move", 353, 0..=2048;
    /// The best move's history bonus: the divisor of the parent's statistics
    /// score added to it.
    STAT_BONUS_PARENT_DIV = b"stat_bonus_parent_div", 32, 8..=1024;
    /// The other moves' history malus: per ply of depth.
    STAT_MALUS_DEPTH = b"stat_malus_depth", 882, 0..=2048;
    /// The other moves' history malus: the constant term.
    STAT_MALUS_OFFSET = b"stat_malus_offset", -204, -2000..=1000;
    /// The other moves' history malus: the cap.
    STAT_MALUS_MAX = b"stat_malus_max", 2122, 0..=4096;
    /// The share of the bonus a quiet best move receives, in 1024ths.
    QUIET_BONUS_SCALE = b"quiet_bonus_scale", 806, 0..=2048;
    /// The share of the malus the other quiet moves receive, in 1024ths.
    QUIET_MALUS_SCALE = b"quiet_malus_scale", 1113, 0..=2048;
    /// How the malus decays from one quiet move to the next, in 1024ths.
    QUIET_MALUS_DECAY = b"quiet_malus_decay", 977, 0..=1024;
    /// The share of the bonus a capture best move receives, in 1024ths.
    CAPTURE_BONUS_SCALE = b"capture_bonus_scale", 1286, 0..=2048;
    /// The share of the malus the other captures receive, in 1024ths.
    CAPTURE_MALUS_SCALE = b"capture_malus_scale", 1559, 0..=3072;
    /// The share of the malus given to an early quiet parent move that was
    /// refuted, in 1024ths.
    REFUTATION_PENALTY = b"refutation_penalty", 616, 0..=2048;
    /// A quiet update: the low-ply history's share, in 1024ths.
    LOW_PLY_BONUS = b"low_ply_bonus", 761, 0..=2048;
    /// A quiet update: the continuation histories' share, in 1024ths.
    CONT_BONUS = b"cont_bonus", 955, 0..=2048;
    /// A quiet update: the pawn history's share of a positive bonus, in
    /// 1024ths.
    PAWN_BONUS = b"pawn_bonus", 850, 0..=2048;
    /// A quiet update: the pawn history's share of a malus, in 1024ths.
    PAWN_MALUS = b"pawn_malus", 550, 0..=2048;
    /// A continuation-history update: the weight of the plane 1 ply back,
    /// in 1024ths.
    CONT_WEIGHT_1 = b"cont_weight_1", 1157, 0..=2048;
    /// A continuation-history update: the weight of the plane 2 plies back,
    /// in 1024ths.
    CONT_WEIGHT_2 = b"cont_weight_2", 648, 0..=2048;
    /// A continuation-history update: the weight of the plane 3 plies back,
    /// in 1024ths.
    CONT_WEIGHT_3 = b"cont_weight_3", 288, 0..=2048;
    /// A continuation-history update: the weight of the plane 4 plies back,
    /// in 1024ths.
    CONT_WEIGHT_4 = b"cont_weight_4", 576, 0..=2048;
    /// A continuation-history update: the weight of the plane 5 plies back,
    /// in 1024ths.
    CONT_WEIGHT_5 = b"cont_weight_5", 140, 0..=2048;
    /// A continuation-history update: the weight of the plane 6 plies back,
    /// in 1024ths.
    CONT_WEIGHT_6 = b"cont_weight_6", 441, 0..=2048;
    /// A continuation-history update: the constant added to the plane one ply
    /// back.
    CONT_NEAR_PLY = b"cont_near_ply", 88, -1000..=1000;
    /// A transposition-table cutoff by a quiet move: its history bonus per ply.
    TT_CUT_BONUS_DEPTH = b"tt_cut_bonus_depth", 130, 0..=1024;
    /// A transposition-table cutoff by a quiet move: the bonus's constant term.
    TT_CUT_BONUS_OFFSET = b"tt_cut_bonus_offset", -71, -1000..=1000;
    /// A transposition-table cutoff by a quiet move: the bonus's cap.
    TT_CUT_BONUS_MAX = b"tt_cut_bonus_max", 1043, 0..=4096;
    /// A transposition-table cutoff: the parent's move count up to which the
    /// parent's quiet move is penalised.
    TT_CUT_PARENT_MOVE_COUNT = b"tt_cut_parent_move_count", 4, 0..=600;
    /// A transposition-table cutoff: the continuation-history penalty of the
    /// parent's move.
    TT_CUT_CONT_PENALTY = b"tt_cut_cont_penalty", -2142, -10000..=0;
    /// The continuation-history bonus of a reduced move whose search failed
    /// high.
    RESEARCH_CONT_BONUS = b"research_cont_bonus", 1426, 0..=4096;
    /// The TT-move history's update at a non-PV node whose best move is the
    /// transposition-table move.
    TT_MOVE_BONUS = b"tt_move_bonus", 805, 0..=4096;
    /// The TT-move history's update when it is not.
    TT_MOVE_MALUS = b"tt_move_malus", -787, -4096..=0;
    /// Multi-cut: the TT-move history's update, constant term.
    MULTICUT_TT_MOVE_BASE = b"multicut_tt_move_base", -424, -4096..=0;
    /// Multi-cut: the TT-move history's update, per ply of depth.
    MULTICUT_TT_MOVE_DEPTH = b"multicut_tt_move_depth", 107, 0..=1000;
    /// Multi-cut: the TT-move history's update, its floor.
    MULTICUT_TT_MOVE_MIN = b"multicut_tt_move_min", -3375, -8192..=0;
    /// A node that fails low after a capture: the capture history's bonus for
    /// that capture.
    PRIOR_CAPTURE_BONUS = b"prior_capture_bonus", 1018, 0..=4096;
    /// A node that fails low after a quiet move: the bonus scale's constant
    /// term.
    PRIOR_BASE = b"prior_base", -232, -2000..=500;
    /// A node that fails low after a quiet move: the divisor of the parent's
    /// statistics score taken off the scale. A statistics score is under 2^17
    /// in magnitude, so the floor keeps the term under 2500.
    PRIOR_STAT_DIV = b"prior_stat_div", 108, 54..=10000;
    /// The bonus scale per ply of depth.
    PRIOR_DEPTH = b"prior_depth", 59, 0..=500;
    /// The cap on the bonus scale's depth term.
    PRIOR_DEPTH_MAX = b"prior_depth_max", 454, 0..=1000;
    /// The parent's move count above which the bonus scale grows.
    PRIOR_MOVE_COUNT = b"prior_move_count", 8, 0..=600;
    /// How much the bonus scale grows above that move count.
    PRIOR_MOVE_COUNT_BONUS = b"prior_move_count_bonus", 169, 0..=500;
    /// How much the bonus scale grows when the best value is well below the
    /// static evaluation.
    PRIOR_EVAL_BONUS = b"prior_eval_bonus", 145, 0..=500;
    /// How far below the static evaluation the best value must be for that.
    PRIOR_EVAL_MARGIN = b"prior_eval_margin", 110, -1000..=1000;
    /// How much the bonus scale grows when the best value is well below the
    /// parent's negated static evaluation.
    PRIOR_PARENT_EVAL_BONUS = b"prior_parent_eval_bonus", 154, 0..=500;
    /// How far below the parent's negated static evaluation it must be.
    PRIOR_PARENT_EVAL_MARGIN = b"prior_parent_eval_margin", 73, -1000..=1000;
    /// The scaled bonus's depth factor: per ply.
    PRIOR_SCALED_DEPTH = b"prior_scaled_depth", 135, 0..=1000;
    /// The scaled bonus's depth factor: its constant term.
    PRIOR_SCALED_OFFSET = b"prior_scaled_offset", -80, -2000..=2000;
    /// The scaled bonus's depth factor: its cap. With the bonus scale's
    /// ceilings the scaled bonus stays under 2^24 in magnitude.
    PRIOR_SCALED_MAX = b"prior_scaled_max", 1400, 0..=2800;
    /// The continuation histories' share of the scaled bonus, in 16384ths.
    PRIOR_CONT_SCALE = b"prior_cont_scale", 221, 0..=1000;
    /// The main history's share of the scaled bonus, in 32768ths.
    PRIOR_MAIN_SCALE = b"prior_main_scale", 235, 0..=1000;
    /// The pawn history's share of the scaled bonus, in 8192ths.
    PRIOR_PAWN_SCALE = b"prior_pawn_scale", 290, 0..=1000;
    /// The parent's quiet move's history update from the change in static
    /// evaluation: the change's floor. Never positive and the ceiling never
    /// negative, so the clamp is never inverted.
    EVAL_DIFF_MIN = b"eval_diff_min", -214, -2000..=0;
    /// The evaluation-change update: the change's ceiling.
    EVAL_DIFF_MAX = b"eval_diff_max", 171, 0..=2000;
    /// The evaluation-change update: the constant added to the clamped change.
    EVAL_DIFF_OFFSET = b"eval_diff_offset", 60, -500..=500;
    /// The evaluation-change update: the main history's factor.
    EVAL_DIFF_MAIN = b"eval_diff_main", 10, 0..=100;
    /// The evaluation-change update: the pawn history's factor.
    EVAL_DIFF_PAWN = b"eval_diff_pawn", 12, 0..=100;
    /// The correction value: the pawn channel's weight. Each channel's entry is
    /// under 1024 in magnitude, so with every weight at most 16384 the
    /// correction value stays under 2^27.
    CORR_PAWN = b"corr_pawn", 12153, 0..=16384;
    /// The correction value: the minor-piece channel's weight.
    CORR_MINOR = b"corr_minor", 8620, 0..=16384;
    /// The correction value: the weight of each side's non-pawn channel.
    CORR_NON_PAWN = b"corr_non_pawn", 12355, 0..=16384;
    /// The correction value: the continuation-correction weight.
    CORR_CONT = b"corr_cont", 7982, 0..=16384;
    /// The correction value: the continuation reading without a previous
    /// move. At most the two readings it stands for.
    CORR_CONT_NO_MOVE = b"corr_cont_no_move", 8, 0..=2048;
    /// A correction update: the minor-piece channel's share, in 128ths.
    CORR_UPDATE_MINOR = b"corr_update_minor", 153, 0..=512;
    /// A correction update: each non-pawn channel's share, in 128ths.
    CORR_UPDATE_NON_PAWN = b"corr_update_non_pawn", 187, 0..=512;
    /// A correction update: the share of the continuation plane two plies
    /// back, in 128ths.
    CORR_UPDATE_CONT_2 = b"corr_update_cont_2", 126, 0..=512;
    /// A correction update: the share of the plane four plies back, in 128ths.
    CORR_UPDATE_CONT_4 = b"corr_update_cont_4", 63, 0..=512;
    /// The correction bonus per ply and unit of error with a best move, in
    /// 128ths. The ceiling keeps the product inside `i32`.
    CORR_BONUS_BEST = b"corr_bonus_best", 12, 0..=64;
    /// The correction bonus without a best move, in 128ths.
    CORR_BONUS_NO_BEST = b"corr_bonus_no_best", 17, 0..=64;
    /// The share of the clamped correction bonus applied, in 1024ths.
    CORR_BONUS_SCALE = b"corr_bonus_scale", 1069, 0..=4096;
    /// The main history's value on a cleared table. Every init is inside its
    /// table's bound.
    MAIN_HISTORY_INIT = b"main_history_init", 0, -2000..=2000;
    /// The capture history's value on a cleared table.
    CAPTURE_HISTORY_INIT = b"capture_history_init", -678, -5000..=5000;
    /// The continuation history's value on a cleared table.
    CONTINUATION_HISTORY_INIT = b"continuation_history_init", -523, -5000..=5000;
    /// The continuation-correction history's value on a cleared table.
    CONT_CORRECTION_INIT = b"cont_correction_init", 6, -512..=512;
    /// The pawn history's value on a cleared table.
    PAWN_HISTORY_INIT = b"pawn_history_init", -1238, -5000..=5000;

    // The move ordering. A weight under 256 in the comment is read as `x * k / 256`
    // with `k` registered, which at the default is the multiple of `x` it
    // replaces exactly, and moves in 256ths of it.

    /// A capture's score: the weight of the captured piece's value, in 256ths.
    MP_CAPTURE_WEIGHT = b"mp_capture_weight", 1792, 0..=8192;
    /// A quiet move's score: the main history's weight, in 256ths.
    MP_MAIN_WEIGHT = b"mp_main_weight", 512, 0..=2048;
    /// A quiet move's score: the pawn history's weight, in 256ths.
    MP_PAWN_WEIGHT = b"mp_pawn_weight", 512, 0..=2048;
    /// A quiet move's score: the low-ply history's weight before its division
    /// by the ply, in 256ths.
    MP_LOW_PLY_WEIGHT = b"mp_low_ply_weight", 2048, 0..=8192;
    /// A quiet move's score: the SEE threshold a direct check must pass for the
    /// check bonus.
    MP_CHECK_SEE = b"mp_check_see", -75, -2000..=2000;
    /// A quiet move's score: the bonus for a direct check.
    MP_CHECK_BONUS = b"mp_check_bonus", 16384, 0..=65536;
    /// The quiet moves are sorted down to this per ply of depth.
    MP_QUIET_SORT = b"mp_quiet_sort", -3560, -20000..=0;
    /// A capture is good when its SEE passes minus its score times 1024 over
    /// this (an eighteenth at the default). The floor keeps the threshold no
    /// larger than the score.
    MP_GOOD_CAPTURE_SEE_DIV = b"mp_good_capture_see_div", 18432, 1024..=1000000;
    /// The score above which a quiet move is tried before the bad captures.
    MP_GOOD_QUIET_THRESHOLD = b"mp_good_quiet_threshold", -14000, -60000..=60000;
}

/// Why a tuning list was refused. Each variant borrows the part of the list it
/// is about.
#[cfg(feature = "tune")]
#[derive(Debug, PartialEq, Eq)]
pub enum TuneError<'a> {
    /// An item that is not `name=value`.
    Malformed(&'a [u8]),
    /// A name no parameter has.
    UnknownName(&'a [u8]),
    /// A value that is not a decimal integer in the parameter's range.
    BadValue { name: &'a [u8], value: &'a [u8] },
    /// A name given more than once.
    Repeated(&'a [u8]),
}

#[cfg(feature = "tune")]
impl TuneError<'_> {
    /// The refusal as one line of text.
    pub fn write_message(&self, out: &mut TextWriter<'_>) {
        match *self {
            Self::Malformed(item) => {
                out.bytes(b"not name=value: `").bytes(item).bytes(b"`");
            }
            Self::UnknownName(name) => {
                out.bytes(b"unknown parameter `").bytes(name).bytes(b"`");
            }
            Self::BadValue { name, value } => {
                out.bytes(b"bad value `").bytes(value).bytes(b"` for `");
                out.bytes(name).bytes(b"`");
                if let Some(entry) = REGISTRY.iter().find(|e| e.name == name) {
                    out.bytes(b" (an integer in ").i64(i64::from(entry.min));
                    out.bytes(b"..=").i64(i64::from(entry.max)).bytes(b")");
                }
            }
            Self::Repeated(name) => {
                out.bytes(b"parameter `")
                    .bytes(name)
                    .bytes(b"` given twice");
            }
        }
    }
}

/// The assignments a tuning list spells, as indices into the registry with
/// their values, or the first thing wrong with it. An empty list assigns
/// nothing; spaces around an item, a name or a value are ignored.
#[cfg(feature = "tune")]
fn parse(list: &[u8]) -> Result<Vec<(usize, i32)>, TuneError<'_>> {
    let mut assignments: Vec<(usize, i32)> = Vec::new();
    if list.trim_ascii().is_empty() {
        return Ok(assignments);
    }
    for item in list.split(|&b| b == b',') {
        let Some(eq) = item.iter().position(|&b| b == b'=') else {
            return Err(TuneError::Malformed(item.trim_ascii()));
        };
        let name = item[..eq].trim_ascii();
        let value = item[eq + 1..].trim_ascii();
        if name.is_empty() {
            return Err(TuneError::Malformed(item.trim_ascii()));
        }
        let Some(index) = REGISTRY.iter().position(|e| e.name == name) else {
            return Err(TuneError::UnknownName(name));
        };
        let entry = &REGISTRY[index];
        let parsed = atoi_i64(value)
            .and_then(|v| i32::try_from(v).ok())
            .filter(|v| (entry.min..=entry.max).contains(v));
        let Some(parsed) = parsed else {
            return Err(TuneError::BadValue { name, value });
        };
        if assignments.iter().any(|&(i, _)| i == index) {
            return Err(TuneError::Repeated(name));
        }
        assignments.push((index, parsed));
    }
    Ok(assignments)
}

/// Set the parameters a tuning list names, leaving the rest at their defaults.
/// Nothing is written unless the whole list is good.
///
/// Called once, on the main thread, before any search thread is spawned —
/// `Param` says why that is what makes the values visible to the search.
#[cfg(feature = "tune")]
pub fn apply(list: &[u8]) -> Result<(), TuneError<'_>> {
    for (index, value) in parse(list)? {
        REGISTRY[index].param.0.store(value, Ordering::Relaxed);
    }
    Ok(())
}

/// The scaled time coefficients at their defaults are the very `f64`s (and,
/// for the maximum ratio, the `f32`) the reference's literals spell, so the
/// per-iteration time scaling computes what the literals computed.
#[cfg(test)]
mod scaled_defaults {
    use super::*;

    #[test]
    #[allow(clippy::float_cmp)]
    fn each_scaled_default_is_the_literal() {
        for (got, literal) in [
            (TM_MAX_SHARE.scaled(1000.0), 0.3),
            (TM_FALLING_BASE.scaled(1000.0), 11.325),
            (TM_FALLING_AVERAGE.scaled(1000.0), 2.115),
            (TM_FALLING_ITERATION.scaled(1000.0), 0.987),
            (TM_FALLING_MIN.scaled(10000.0), 0.5688),
            (TM_FALLING_MAX.scaled(10000.0), 1.5698),
            (TM_REDUCTION_K.scaled(10000.0), 0.5189),
            (TM_REDUCTION_CENTER.scaled(100.0), 11.57),
            (TM_REDUCTION_BASE.scaled(1000.0), 0.723),
            (TM_REDUCTION_AMPLITUDE.scaled(1000.0), 0.79),
            (TM_REDUCTION_PREVIOUS.scaled(1000.0), 1.455),
            (TM_INSTABILITY_BASE.scaled(1000.0), 1.04),
            (TM_INSTABILITY_WEIGHT.scaled(10000.0), 1.8956),
            (TM_EFFORT_FACTOR.scaled(1000.0), 0.666),
            (TM_DEEPEN_SHARE.scaled(1000.0), 0.503),
        ] {
            assert_eq!(
                got.to_bits(),
                f64::to_bits(literal),
                "{got} against {literal}"
            );
        }
        assert_eq!(TM_MAX_RATIO.get() as f32 / 1000.0, 5.0f32);
    }

    /// A small divisor read as `x * 1024 / k` gives at its default the
    /// quotient of the literal `x / d` it replaces, negative `x` included.
    #[test]
    fn each_small_divisor_at_its_default_is_the_literal() {
        for (k, d, wide) in [
            (ASPIRATION_WIDEN_DIV.get(), 3, true),
            (SEARCH_AGAIN_DIV.get(), 4, false),
            (NMP_DEPTH_DIV.get(), 3, false),
            (NMP_VERIFY_DIV.get(), 4, false),
            (SINGULAR_DEPTH_DIV.get(), 2, false),
            (MP_GOOD_CAPTURE_SEE_DIV.get(), 18, false),
        ] {
            assert_eq!(k, 1024 * d);
            let reach = if cfg!(miri) { 100 } else { 100_000 };
            for x in -reach..=reach {
                assert_eq!(x * 1024 / k, x / d, "{x} / {d}");
            }
            if wide {
                for x in [i64::from(i32::MAX), 1 << 40, -(1 << 40)] {
                    assert_eq!(x * 1024 / i64::from(k), x / i64::from(d), "{x} / {d}");
                }
            }
        }
    }

    /// A move-ordering weight read as `x * k / 256` gives at its default the
    /// multiple of `x` it replaces, negative `x` included.
    #[test]
    fn each_move_ordering_weight_at_its_default_is_the_literal() {
        for (k, m) in [
            (MP_CAPTURE_WEIGHT.get(), 7),
            (MP_MAIN_WEIGHT.get(), 2),
            (MP_PAWN_WEIGHT.get(), 2),
            (MP_LOW_PLY_WEIGHT.get(), 8),
        ] {
            assert_eq!(k, 256 * m);
            let reach = if cfg!(miri) { 100 } else { 100_000 };
            for x in -reach..=reach {
                assert_eq!(x * k / 256, m * x, "{m} * {x}");
            }
        }
    }
}

#[cfg(test)]
#[cfg(feature = "tune")]
mod tests {
    use super::*;

    fn index(name: &[u8]) -> usize {
        REGISTRY.iter().position(|e| e.name == name).unwrap()
    }

    #[test]
    fn a_good_list_assigns_every_named_parameter() {
        let got = parse(b"razor_base=480, qs_see_margin=-60 ,aspiration_delta=+7").unwrap();
        assert_eq!(
            got,
            vec![
                (index(b"razor_base"), 480),
                (index(b"qs_see_margin"), -60),
                (index(b"aspiration_delta"), 7),
            ]
        );
    }

    #[test]
    fn an_empty_list_assigns_nothing() {
        assert_eq!(parse(b""), Ok(vec![]));
        assert_eq!(parse(b"  "), Ok(vec![]));
    }

    #[test]
    fn an_unknown_name_is_refused() {
        assert_eq!(
            parse(b"razor_base=480,razor_bias=3"),
            Err(TuneError::UnknownName(b"razor_bias"))
        );
    }

    #[test]
    fn a_bad_value_is_refused() {
        for (list, value) in [
            (&b"razor_base=4.5"[..], &b"4.5"[..]),
            (b"razor_base=", b""),
            (b"razor_base=abc", b"abc"),
            (b"razor_base=99999999999", b"99999999999"),
            // Outside the range the search tolerates.
            (b"razor_base=2001", b"2001"),
            (b"aspiration_delta=0", b"0"),
        ] {
            let name: &[u8] = &list[..list.iter().position(|&b| b == b'=').unwrap()];
            assert_eq!(
                parse(list),
                Err(TuneError::BadValue { name, value }),
                "{}",
                String::from_utf8_lossy(list)
            );
        }
    }

    #[test]
    fn a_malformed_or_repeated_item_is_refused() {
        assert_eq!(
            parse(b"razor_base"),
            Err(TuneError::Malformed(b"razor_base"))
        );
        assert_eq!(parse(b"razor_base=1,"), Err(TuneError::Malformed(b"")));
        assert_eq!(parse(b"=5"), Err(TuneError::Malformed(b"=5")));
        assert_eq!(
            parse(b"razor_base=1,razor_base=2"),
            Err(TuneError::Repeated(b"razor_base"))
        );
    }

    #[test]
    fn a_refusal_names_what_was_wrong() {
        let mut bytes = [0u8; 256];
        let mut out = TextWriter::new(&mut bytes);
        parse(b"aspiration_delta=0")
            .unwrap_err()
            .write_message(&mut out);
        assert_eq!(
            out.as_bytes(),
            b"bad value `0` for `aspiration_delta` (an integer in 1..=1000)"
        );
    }

    #[test]
    fn the_names_are_distinct() {
        for (i, a) in REGISTRY.iter().enumerate() {
            assert!(REGISTRY[..i].iter().all(|b| b.name != a.name));
        }
    }
}
