//! The search parameters a tuning run varies: the pruning and reduction
//! constants that shape the tree, and the coefficients of the time allocation
//! that decide how long a search runs, each named once here and read through
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
    /// margin without one is this less 21, so the floor keeps it positive.
    FUTILITY_MULT = b"futility_mult", 76, 22..=500;
    /// Null-move pruning: how far the static evaluation may sit below beta,
    /// before the depth and improving terms, for the null move to be tried.
    NMP_MARGIN = b"nmp_margin", 378, 0..=2000;
    /// Null-move pruning: the depth reduction before the `depth / 3` term. At
    /// least one, so the null-move search is shallower than the node.
    NMP_REDUCTION = b"nmp_reduction", 7, 1..=20;
    /// ProbCut: how far above beta the cut is sought, before the improving
    /// term, which takes 61 off it. The floor keeps the ProbCut beta above beta.
    PROBCUT_MARGIN = b"probcut_margin", 224, 62..=2000;
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
    /// added to the margin. A correction value stays under 2^26 in
    /// magnitude, so the floor keeps the term under 7000.
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
    /// to the reduced depth. The history sum stays under 90000 in magnitude,
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
