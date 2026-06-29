use crate::search::*;
use crate::types::*;
use crate::usioption::*;

// Think-start time-management inputs, read once per `init`. Tournament build folds compile-time
// consts baked by `build.rs`, else the runtime USI options.

/// `Move_Overhead` (ms): per-move communication/round-trip overhead used by `compute`.
#[cfg(feature = "tournament")]
#[inline]
fn move_overhead(_usi_options: &UsiOptions) -> i64 {
    crate::tournament::MOVE_OVERHEAD
}

#[cfg(not(feature = "tournament"))]
#[inline]
fn move_overhead(usi_options: &UsiOptions) -> i64 {
    usi_options.get_i64(UsiOptions::MOVE_OVERHEAD)
}

/// `Opening_Time_Weight` (percent): opening time-allocation weight used by `compute`.
#[cfg(feature = "tournament")]
#[inline]
fn opening_time_weight(_usi_options: &UsiOptions) -> i64 {
    crate::tournament::OPENING_TIME_WEIGHT
}

#[cfg(not(feature = "tournament"))]
#[inline]
fn opening_time_weight(usi_options: &UsiOptions) -> i64 {
    usi_options.get_i64(UsiOptions::OPENING_TIME_WEIGHT)
}

/// `Timeout_Safety_Margin` (ms): near-deadline cushion used by the low-time cap in `compute`.
#[cfg(feature = "tournament")]
#[inline]
fn timeout_safety_margin(_usi_options: &UsiOptions) -> i64 {
    crate::tournament::TIMEOUT_SAFETY_MARGIN
}

#[cfg(not(feature = "tournament"))]
#[inline]
fn timeout_safety_margin(usi_options: &UsiOptions) -> i64 {
    usi_options.get_i64(UsiOptions::TIMEOUT_SAFETY_MARGIN)
}

/// `Minimum_Thinking_Time` (ms): floor on the committed `optimum` think time in `compute`.
#[cfg(feature = "tournament")]
#[inline]
fn minimum_thinking_time(_usi_options: &UsiOptions) -> i64 {
    crate::tournament::MINIMUM_THINKING_TIME
}

#[cfg(not(feature = "tournament"))]
#[inline]
fn minimum_thinking_time(usi_options: &UsiOptions) -> i64 {
    usi_options.get_i64(UsiOptions::MINIMUM_THINKING_TIME)
}

#[derive(Clone)]
pub struct TimeManagement {
    start_time: Option<std::time::Instant>,
    optimum_time_milli: std::time::Duration,
    maximum_time_milli: std::time::Duration,
}

impl TimeManagement {
    pub fn new() -> TimeManagement {
        TimeManagement {
            start_time: None,
            optimum_time_milli: std::time::Duration::from_millis(0),
            maximum_time_milli: std::time::Duration::from_millis(0),
        }
    }
    /// Floor on `maximum` think time (ms) so a legal move is still produced near the deadline.
    const MAXIMUM_FLOOR_MILLI: i64 = 50;

    pub fn init(&mut self, usi_optoins: &UsiOptions, limits: &mut LimitsType, us: Color, ply: i32) {
        self.start_time = limits.start_time;
        // Read the time-management inputs once via the coexistence accessors (USI options, or consts under `tournament`).
        let move_overhead = move_overhead(usi_optoins);
        let opening_time_weight = opening_time_weight(usi_optoins);
        let timeout_safety_margin = timeout_safety_margin(usi_optoins);
        let minimum_thinking_time = minimum_thinking_time(usi_optoins);
        let our_time = limits.time[us.0 as usize].as_millis() as i64;
        let our_inc = limits.inc[us.0 as usize].as_millis() as i64;
        let (optimum, maximum) = Self::compute(
            our_time,
            our_inc,
            ply,
            move_overhead,
            opening_time_weight,
            timeout_safety_margin,
            minimum_thinking_time,
        );
        self.optimum_time_milli = std::time::Duration::from_millis(optimum);
        self.maximum_time_milli = std::time::Duration::from_millis(maximum);
    }

    /// Pure `(optimum, maximum)` think-time allocation (ms), side-effect-free for unit testing.
    fn compute(
        our_time: i64,
        our_inc: i64,
        ply: i32,
        move_overhead: i64,
        opening_time_weight: i64,
        timeout_safety_margin: i64,
        minimum_thinking_time: i64,
    ) -> (u64, u64) {
        let moves_to_go = 50;
        let time_left = std::cmp::max(1, our_time + our_inc * (moves_to_go - 1) - move_overhead * (2 + moves_to_go));
        let time_left = time_left * opening_time_weight / 100;

        let opt_scale = ((0.8 + ply as f64 / 128.0) / moves_to_go as f64).min(0.8 * our_time as f64 / time_left as f64);
        let max_scale = 6.3f64.min(1.5 + 0.11 * moves_to_go as f64);

        let optimum = opt_scale * time_left as f64;
        let maximum = (0.8 * our_time as f64 - move_overhead as f64).min(max_scale * optimum);

        // Low-time regime: keep `move_overhead + timeout_safety_margin` in reserve so a move can never self-timeout.
        let maximum = if our_time < 5 * timeout_safety_margin {
            let safe_cap = (our_time - (move_overhead + timeout_safety_margin)).max(Self::MAXIMUM_FLOOR_MILLI) as f64;
            maximum.min(safe_cap)
        } else {
            maximum
        };

        // Floor the committed think time, but never above `maximum`.
        let optimum = optimum.max(minimum_thinking_time as f64).min(maximum);

        (optimum.max(0.0) as u64, maximum.max(0.0) as u64)
    }
    pub fn optimum_millis(&self) -> i64 {
        self.optimum_time_milli.as_millis() as i64
    }
    pub fn maximum_millis(&self) -> i64 {
        self.maximum_time_milli.as_millis() as i64
    }
    pub fn elapsed(&self) -> i64 {
        let duration = self.start_time.unwrap().elapsed();
        (duration.as_secs() * 1000 + u64::from(duration.subsec_millis())) as i64
    }
}

#[cfg(test)]
mod tests {
    use super::TimeManagement;

    // Option defaults used by the tests.
    const OVERHEAD: i64 = 10;
    const WEIGHT: i64 = 100;
    const MARGIN: i64 = 2_000;
    const MIN_THINK: i64 = 1_000;

    // 120s + 2s Fischer at a full clock: optimum <= maximum, and no single move takes the whole clock.
    #[test]
    fn fast_fischer_full_clock() {
        let (opt, max) = TimeManagement::compute(120_000, 2_000, 1, OVERHEAD, WEIGHT, MARGIN, MIN_THINK);
        assert!(opt <= max, "optimum {opt} must not exceed maximum {max}");
        assert!(max < 120_000, "a single move must never be allowed the whole clock");
        assert!(max > 0);
    }

    // Self-preservation: maximum think time is always strictly less than the remaining clock.
    #[test]
    fn never_allows_a_self_timeout() {
        for &t in &[100i64, 500, 1_000, 2_000, 2_010, 3_000, 5_000, 10_000, 30_000, 120_000] {
            let (_opt, max) = TimeManagement::compute(t, 2_000, 40, OVERHEAD, WEIGHT, MARGIN, MIN_THINK);
            assert!((max as i64) < t, "t={t}: maximum {max} must be < remaining {t}");
        }
    }

    // Low clock vs increment: think time stays a sustainable chunk, under the remaining clock.
    #[test]
    fn low_clock_settles_to_sustainable_chunk() {
        let (_opt, max) = TimeManagement::compute(4_000, 2_000, 40, OVERHEAD, WEIGHT, MARGIN, MIN_THINK);
        assert!((max as i64) < 4_000, "must leave time on a 4s clock, got {max}");
        assert!((max as i64) > 1_000, "should still use a sustainable chunk, got {max}");
    }

    // Long TC: the safety cap is inert, so optimum and maximum match the base formula.
    #[test]
    fn long_tc_unchanged() {
        let (opt, max) = TimeManagement::compute(600_000, 10_000, 1, OVERHEAD, WEIGHT, MARGIN, MIN_THINK);
        assert!(
            (17_400..17_800).contains(&(opt as i64)),
            "optimum {opt} drifted from the formula value"
        );
        assert!(
            (110_000..111_500).contains(&(max as i64)),
            "maximum {max} drifted from the formula value"
        );
    }

    // Increasing the move overhead reduces both optimum and maximum (monotonicity).
    #[test]
    fn move_overhead_is_monotonic() {
        let (opt_small, max_small) = TimeManagement::compute(120_000, 2_000, 40, 10, WEIGHT, MARGIN, MIN_THINK);
        let (opt_large, max_large) = TimeManagement::compute(120_000, 2_000, 40, 300, WEIGHT, MARGIN, MIN_THINK);
        assert!(
            opt_large < opt_small,
            "larger overhead should shrink optimum: {opt_large} !< {opt_small}"
        );
        assert!(
            max_large < max_small,
            "larger overhead should shrink maximum: {max_large} !< {max_small}"
        );
    }

    // Threshold at `our_time < 5 * margin`: inert above, binds at `our_time - (overhead + margin)` below.
    #[test]
    fn timeout_safety_margin_threshold_boundary() {
        // Above the threshold: the margin is inert.
        let (_o1, max_inert_a) = TimeManagement::compute(100_000, 2_000, 40, OVERHEAD, WEIGHT, 2_000, MIN_THINK);
        let (_o2, max_inert_b) = TimeManagement::compute(100_000, 2_000, 40, OVERHEAD, WEIGHT, 8_000, MIN_THINK);
        assert_eq!(
            max_inert_a, max_inert_b,
            "above the threshold the safety margin must be inert"
        );

        // Below the threshold (our_time = 9_000 < 5 * 2_000) the cap binds exactly.
        let our_time = 9_000i64;
        let (_o3, max_bind) = TimeManagement::compute(our_time, 2_000, 40, OVERHEAD, WEIGHT, MARGIN, MIN_THINK);
        assert_eq!(
            max_bind as i64,
            our_time - (OVERHEAD + MARGIN),
            "below the threshold the cap must bind at our_time - (overhead + margin)"
        );
    }

    // In the binding (low-time) regime, a larger Timeout_Safety_Margin yields a smaller maximum.
    #[test]
    fn timeout_safety_margin_is_monotonic() {
        // our_time = 4_000 is below 5 * margin for both 1_000 and 2_000, so both bind.
        let (_o_s, max_small) = TimeManagement::compute(4_000, 2_000, 40, OVERHEAD, WEIGHT, 1_000, MIN_THINK);
        let (_o_l, max_large) = TimeManagement::compute(4_000, 2_000, 40, OVERHEAD, WEIGHT, 2_000, MIN_THINK);
        assert!(
            max_large < max_small,
            "larger safety margin should shrink maximum: {max_large} !< {max_small}"
        );
    }

    // Minimum_Thinking_Time floors optimum, monotonically, never past maximum.
    #[test]
    fn minimum_thinking_time_floors_optimum() {
        // our_time = 5_000: base optimum sits below maximum, so the floor is observable.
        let (opt_low, _m1) = TimeManagement::compute(5_000, 2_000, 40, OVERHEAD, WEIGHT, MARGIN, 500);
        let (opt_high, max_high) = TimeManagement::compute(5_000, 2_000, 40, OVERHEAD, WEIGHT, MARGIN, 2_500);
        assert!(opt_high > opt_low, "a larger Minimum_Thinking_Time should raise optimum");
        assert_eq!(opt_high as i64, 2_500, "optimum should be floored at Minimum_Thinking_Time");
        assert!(opt_high <= max_high, "optimum must never exceed maximum");

        // An absurdly large floor is clamped to `maximum`, never beyond it.
        let (opt_capped, max_capped) = TimeManagement::compute(5_000, 2_000, 40, OVERHEAD, WEIGHT, MARGIN, 100_000);
        assert_eq!(
            opt_capped, max_capped,
            "Minimum_Thinking_Time must not push optimum past maximum"
        );
    }

    // At shipped defaults the only low-time reserve is Move_Overhead + Timeout_Safety_Margin.
    #[test]
    fn effective_reserve_is_overhead_plus_safety_margin_at_defaults() {
        const DEFAULT_OVERHEAD: i64 = 200;
        const DEFAULT_SAFETY_MARGIN: i64 = 1_200;
        for &reported in &[2_000i64, 3_000, 5_000] {
            let (opt, max) = TimeManagement::compute(reported, 2_000, 40, DEFAULT_OVERHEAD, 100, DEFAULT_SAFETY_MARGIN, 1_000);
            assert_eq!(
                max as i64,
                reported - (DEFAULT_OVERHEAD + DEFAULT_SAFETY_MARGIN),
                "reported={reported}: the only reserve must be Move_Overhead + Timeout_Safety_Margin"
            );
            assert!(opt <= max);
        }
    }

    // Opening_Time_Weight scales allocated time (larger weight -> larger optimum).
    #[test]
    fn opening_time_weight_is_monotonic() {
        let (opt_100, _m1) = TimeManagement::compute(600_000, 10_000, 1, OVERHEAD, 100, MARGIN, MIN_THINK);
        let (opt_200, _m2) = TimeManagement::compute(600_000, 10_000, 1, OVERHEAD, 200, MARGIN, MIN_THINK);
        assert!(
            opt_200 > opt_100,
            "larger Opening_Time_Weight should raise optimum: {opt_200} !> {opt_100}"
        );
    }
}

// Tournament build: the think-start time-management inputs are the compile-time consts baked by build.rs, not runtime USI options.
#[cfg(all(test, feature = "tournament"))]
mod tournament_const_tests {
    use super::*;

    #[test]
    fn time_management_consts_are_fixed() {
        // `const` context: each compiles only if it is a genuine compile-time const.
        const MOVE_OVERHEAD: i64 = crate::tournament::MOVE_OVERHEAD;
        const TIMEOUT_SAFETY_MARGIN: i64 = crate::tournament::TIMEOUT_SAFETY_MARGIN;
        const MINIMUM_THINKING_TIME: i64 = crate::tournament::MINIMUM_THINKING_TIME;
        const OPENING_TIME_WEIGHT: i64 = crate::tournament::OPENING_TIME_WEIGHT;
        assert_eq!(MOVE_OVERHEAD, 200, "v1 tournament config bakes Move_Overhead = 200");
        assert_eq!(
            TIMEOUT_SAFETY_MARGIN, 1_200,
            "v1 tournament config bakes Timeout_Safety_Margin = 1200"
        );
        assert_eq!(
            MINIMUM_THINKING_TIME, 1_000,
            "v1 tournament config bakes Minimum_Thinking_Time = 1000"
        );
        assert_eq!(
            OPENING_TIME_WEIGHT, 100,
            "v1 tournament config bakes Opening_Time_Weight = 100"
        );
    }

    // The accessors resolve to the baked const under `tournament`; the `UsiOptions` argument is ignored.
    #[test]
    fn accessors_return_baked_consts() {
        let opts = UsiOptions::new();
        assert_eq!(move_overhead(&opts), crate::tournament::MOVE_OVERHEAD);
        assert_eq!(opening_time_weight(&opts), crate::tournament::OPENING_TIME_WEIGHT);
        assert_eq!(timeout_safety_margin(&opts), crate::tournament::TIMEOUT_SAFETY_MARGIN);
        assert_eq!(minimum_thinking_time(&opts), crate::tournament::MINIMUM_THINKING_TIME);
    }
}
