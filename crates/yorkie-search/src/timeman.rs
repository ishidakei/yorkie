//! Time management — a port of the reference `TimeManagement`, compiled under
//! `USE_TIME_MANAGEMENT` in a non-DEEP build.
//!
//! For one `go` this computes an **optimum**, **maximum** and **minimum** think
//! time in milliseconds. The search then shrinks the deadline toward `optimum`
//! based on eval stability and stops at [`TimeManagement::search_end`].
//!
//! All internal arithmetic is in `i64` milliseconds, and the same `(int)`
//! truncation and `f32` / `f64` points as the C++ code are preserved so the
//! computed budgets match bit-for-bit given the same inputs.
//!
//! The budget is the same whether or not the engine keeps searching after its
//! move, so nothing here is worth a bonus that a measurement has not asked for.
//! What thinking ahead does change is how much of the budget is left: the time
//! that search spent on the reply the opponent then played is handed in as a
//! credit ([`TimeManagement::set_credit`]), which every stop decision counts as
//! already elapsed. The clamps to the real remaining time keep reading the real
//! clock, so a credit can only make a search stop sooner.

use std::time::Instant;

#[cfg(feature = "verbose2")]
use crate::book::Prng;
use crate::params;

/// The raw inputs [`TimeManagement::init`] needs, extracted by the Protocol
/// layer from a `go` request and the root position. Keeping the input primitive
/// (rather than that layer's own `GoParams`) preserves the layering rule that
/// Search never depends on Protocol.
///
/// Only what a `go` brings is here. The settings the budget also depends on —
/// the two network delays, the minimum think time, the slow-mover percentage,
/// the whole-second rounding and the draw horizon — are compiled in, so
/// [`TimeManagement::init`] reads them itself instead of being handed the same
/// values again at every `go`.
#[derive(Clone, Copy, Debug)]
pub struct TimeInput {
    /// `limits.time[us]` — the side-to-move's remaining main clock in ms.
    pub time_us: i64,
    /// `limits.inc[us]` — the side-to-move's Fischer increment in ms.
    pub inc_us: i64,
    /// `limits.byoyomi[us]` — the side-to-move's byoyomi in ms.
    pub byoyomi_us: i64,
    /// `limits.movetime` in ms (`0` when not a `go movetime`). `go movetime` and
    /// `go mate <ms>`, its only two sources, are `verbose2` clauses.
    #[cfg(feature = "verbose2")]
    pub movetime: i64,
    /// `limits.rtime` in ms (`0` when not a `go rtime`). `go rtime` is a
    /// `verbose2` clause and nothing else seeds it.
    #[cfg(feature = "verbose2")]
    pub rtime: i64,
    /// `ply` — the root's game ply (`rootPos.game_ply()`; 1 at the hirate start).
    pub ply: i32,
    /// The instant the `go` arrived (`limits.startTime`), the origin for
    /// [`TimeManagement::elapsed`].
    pub start_time: Instant,
}

/// The half of the budget's inputs that a `go` does not bring: the settings the
/// engine compiles in. [`COMPILED`] is the one instance a search ever uses; the
/// type exists so the arithmetic below can be exercised at values other than the
/// ones this binary was built with.
#[derive(Clone, Copy, Debug)]
struct TimeSettings {
    /// `options["NetworkDelay"]` in ms.
    network_delay: i64,
    /// `options["NetworkDelay2"]` in ms.
    network_delay2: i64,
    /// `options["MinimumThinkingTime"]` in ms.
    minimum_thinking_time: i64,
    /// `options["SlowMover"]` — percentage multiplier on the optimum time.
    slow_mover: i64,
    /// `options["RoundUpToFullSecond"]`.
    round_up_to_fullsecond: bool,
    /// The game ply past which a draw is adjudicated, with the reference's
    /// `0 → 100000` unlimited remap applied.
    max_moves_to_draw: i32,
}

/// The settings this binary plays under — assembled by the compiler, not at
/// each `go`.
const COMPILED: TimeSettings = TimeSettings {
    network_delay: crate::config::NETWORK_DELAY,
    network_delay2: crate::config::NETWORK_DELAY2,
    minimum_thinking_time: crate::config::MINIMUM_THINKING_TIME,
    slow_mover: crate::config::SLOW_MOVER,
    round_up_to_fullsecond: crate::config::ROUND_UP_TO_FULL_SECOND,
    max_moves_to_draw: if crate::config::MAX_MOVES_TO_DRAW == 0 {
        100_000
    } else {
        crate::config::MAX_MOVES_TO_DRAW as i32
    },
};

/// The reference `TimeManagement` state for one `go`.
#[derive(Clone, Debug)]
pub struct TimeManagement {
    /// `startTime` — the origin for [`Self::elapsed`] (`now() - startTime`).
    pub start_time: Instant,
    /// `search_end` [ms from `start_time`, real time]: `0` means "not yet
    /// decided"; once set, the search stops when `search_end <= elapsed`, the
    /// credit not counted.
    pub search_end: i64,
    /// True only for the `MTG <= 0` error path, so the driver can emit the
    /// reference `info string Error!` diagnostic.
    pub mtg_error: bool,

    minimum_time: i64,
    optimum_time: i64,
    maximum_time: i64,

    minimum_thinking_time: i64,
    network_delay: i64,
    remain_time: i64,
    round_up_to_fullsecond: bool,
    /// The time in ms counted as already elapsed before `start_time`. A `u32`
    /// rather than an `i64` so it sits in the tail padding the fields above
    /// leave: the struct is part of every search's state, and a wider one would
    /// move the fields laid out after it.
    credit: u32,
}

const _: () = assert!(size_of::<TimeManagement>() == 80);

impl TimeManagement {
    /// Compute the think-time budget for one `go`. The PRNG draws the decaying
    /// bump on top of a `go rtime` budget and is consulted nowhere else, so it
    /// is a parameter only where that clause exists.
    pub fn init(input: &TimeInput, #[cfg(feature = "verbose2")] prng: &mut Prng) -> TimeManagement {
        Self::init_with(
            input,
            &COMPILED,
            #[cfg(feature = "verbose2")]
            prng,
        )
    }

    /// [`Self::init`] against an explicit settings half.
    fn init_with(
        input: &TimeInput,
        settings: &TimeSettings,
        #[cfg(feature = "verbose2")] prng: &mut Prng,
    ) -> TimeManagement {
        let &TimeInput {
            time_us,
            inc_us,
            byoyomi_us,
            #[cfg(feature = "verbose2")]
            movetime,
            #[cfg(feature = "verbose2")]
            rtime,
            ply,
            start_time,
        } = input;
        let &TimeSettings {
            network_delay,
            network_delay2,
            minimum_thinking_time,
            slow_mover,
            round_up_to_fullsecond,
            max_moves_to_draw,
        } = settings;

        let mut tm = TimeManagement {
            start_time,
            search_end: 0,
            mtg_error: false,
            minimum_time: 0,
            optimum_time: 0,
            maximum_time: 0,
            minimum_thinking_time,
            network_delay,
            remain_time: 0,
            round_up_to_fullsecond,
            credit: 0,
        };

        // Byoyomi is folded in here because it is available for *this* move;
        // the Fischer increment is deliberately not, since it is credited only
        // after the move has been played and spending it now would overdraw the
        // clock. Floored so a spent clock cannot self-destruct.
        let mut remain_time = time_us + byoyomi_us - network_delay2;
        remain_time = remain_time.max(if round_up_to_fullsecond { 100 } else { 1 });
        tm.remain_time = remain_time;

        // `go rtime`: a randomised minimum-think budget, decaying with ply,
        // used for self-play variety.
        #[cfg(feature = "verbose2")]
        if rtime != 0 {
            let mut r = rtime;
            if ply != 0 {
                let bound = (r as f32 * 0.5).min(r as f32 * 10.0 / ply as f32) as i64;
                r += prng.rand(bound.max(0) as u64) as i64;
            }
            tm.remain_time = r;
            tm.minimum_time = r;
            tm.optimum_time = r;
            tm.maximum_time = r;
            return tm;
        }

        // `go movetime`: spend exactly the given time.
        #[cfg(feature = "verbose2")]
        if movetime != 0 {
            tm.remain_time = movetime;
            tm.minimum_time = movetime;
            tm.optimum_time = movetime;
            tm.maximum_time = movetime;
            return tm;
        }

        // Time-forfeit (sudden death): neither increment nor byoyomi.
        let time_forfeit = inc_us == 0 && byoyomi_us == 0;

        // The planning horizon (the reference `MoveHorizon`, the plies assumed
        // still to play), wider early and narrower once out of the opening.
        let move_horizon = params::TM_MOVE_HORIZON.get();
        let move_horizon = if time_forfeit {
            let opening = params::TM_OPENING_FORFEIT.get();
            move_horizon + opening - ply.min(opening)
        } else {
            move_horizon + params::TM_OPENING_INC.get()
                - ply.min(params::TM_OPENING_INC_PLIES.get())
        };

        // Own remaining moves until the draw horizon.
        let mtg = (max_moves_to_draw - ply + 2).min(move_horizon) / 2;

        if mtg <= 0 {
            // Should be unreachable given a sane MaxMovesToDraw; guard anyway.
            tm.mtg_error = true;
            tm.minimum_time = 500;
            tm.optimum_time = 500;
            tm.maximum_time = 500;
            return tm;
        }
        if mtg == 1 {
            // Last move before the horizon: spend everything.
            tm.minimum_time = remain_time;
            tm.optimum_time = remain_time;
            tm.maximum_time = remain_time;
            return tm;
        }

        // Minimum think time floor.
        let minimum_time = (minimum_thinking_time - network_delay).max(if round_up_to_fullsecond {
            1000
        } else {
            1
        });
        tm.minimum_time = minimum_time;

        // Time estimated still available across the remaining moves.
        let mut remain_estimate = time_us + inc_us * mtg as i64 + byoyomi_us * mtg as i64;
        if round_up_to_fullsecond {
            remain_estimate -= (mtg as i64 + 1) * 1000;
        }
        remain_estimate = remain_estimate.max(0);

        // optimum candidate.
        let t1 = minimum_time + remain_estimate / mtg as i64;

        // maximum candidate: up to `max_ratio`× the optimum, capped at a share
        // (30% by default) of the remaining estimate.
        // In `f32`, as the reference computes it: the ratio in thousandths is
        // divided in `f32` so that 5000 gives exactly the reference's 5.0.
        let mut max_ratio = params::TM_MAX_RATIO.get() as f32 / 1000.0;
        if time_forfeit {
            max_ratio = max_ratio.min((time_us as f32 / (60.0 * 1000.0)).max(1.0));
        }
        let mut t2 = minimum_time + (remain_estimate as f32 * max_ratio / mtg as f32) as i64;
        t2 = t2.min((remain_estimate as f64 * params::TM_MAX_SHARE.scaled(1000.0)) as i64);

        // Fold in SlowMover and clamp to the remaining time.
        tm.optimum_time = t1.min(remain_time) * slow_mover / 100;
        tm.maximum_time = t2.min(remain_time);

        // Byoyomi with (almost) no main clock: spend it all this move.
        if byoyomi_us != 0 && time_us < (byoyomi_us as f64 * 1.2) as i64 {
            let v = byoyomi_us + time_us;
            tm.minimum_time = v;
            tm.optimum_time = v;
            tm.maximum_time = v;
        }

        // Final clamps: round up minimum/maximum and never exceed remain_time.
        tm.minimum_time = tm.round_up(tm.minimum_time).min(remain_time);
        tm.optimum_time = tm.optimum_time.min(remain_time);
        tm.maximum_time = tm.round_up(tm.maximum_time).min(remain_time);

        tm
    }

    /// Round `t0` up to a whole second (subtracting the network delay), floored
    /// at `MinimumThinkingTime` and capped at `remain_time`. A no-op rounding
    /// when `RoundUpToFullSecond` is off.
    pub fn round_up(&self, t0: i64) -> i64 {
        self.round_up_over(t0, self.minimum_thinking_time)
    }

    /// [`Self::round_up`] with `floor` in place of `MinimumThinkingTime`.
    fn round_up_over(&self, t0: i64, floor: i64) -> i64 {
        if self.round_up_to_fullsecond {
            let mut t = (((t0 + 999) / 1000) * 1000).max(floor);
            t -= self.network_delay;
            if t < t0 {
                t += 1000;
            }
            t.min(self.remain_time)
        } else {
            let mut t = t0.max(floor);
            t -= self.network_delay;
            t.min(self.remain_time)
        }
    }

    /// Fix the search end time from the elapsed time `e` in ms at which the
    /// search decided to stop, rounding the used time up to a full second and
    /// storing it as an offset from `start_time`.
    ///
    /// `e` is the credited elapsed time ([`Self::credited_elapsed_from`]). With
    /// no credit the floor is [`Self::minimum`]: a search that decided to stop
    /// before the minimum think time still spends it. With one, the search
    /// stands on time spent before this `go`, so there is no floor; the rounding
    /// still applies to the real elapsed time, because the part of the current
    /// wall second the clock charges anyway is still the engine's to use.
    pub fn set_search_end(&mut self, e: i64) {
        self.search_end = if self.credit == 0 {
            self.round_up(e.max(self.minimum_time))
        } else {
            self.round_up_over(e - i64::from(self.credit), 0)
        };
    }

    /// Count `ms` as already elapsed when this `go` arrived. Clamped at zero,
    /// and at the largest credit the field holds, which is weeks.
    pub fn set_credit(&mut self, ms: i64) {
        self.credit = ms.clamp(0, i64::from(u32::MAX)) as u32;
    }

    /// The credit [`Self::set_credit`] installed, in ms.
    pub fn credit(&self) -> i64 {
        i64::from(self.credit)
    }

    /// Elapsed time in ms since `startTime`, measured against `now`: the real
    /// time this `go` has run, which the clock checks read.
    pub fn elapsed_from(&self, now: Instant) -> i64 {
        now.saturating_duration_since(self.start_time).as_millis() as i64
    }

    /// The elapsed time every stop decision reads: [`Self::elapsed_from`] plus
    /// the credit.
    pub fn credited_elapsed_from(&self, now: Instant) -> i64 {
        self.elapsed_from(now) + i64::from(self.credit)
    }

    /// Elapsed time in ms since `startTime`, measured now.
    pub fn elapsed(&self) -> i64 {
        self.elapsed_from(Instant::now())
    }

    /// `optimum()` — the target think time in ms.
    pub fn optimum(&self) -> i64 {
        self.optimum_time
    }
    /// `maximum()` — the hard think-time ceiling in ms.
    pub fn maximum(&self) -> i64 {
        self.maximum_time
    }
    /// `minimum()` — the minimum think time in ms.
    pub fn minimum(&self) -> i64 {
        self.minimum_time
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `TimeInput` with no clock; individual tests override the fields they
    /// exercise. The budgets the tests below pin are the reference's under the
    /// option defaults, which is what the compiled settings are asserted to be
    /// first, so a config that moved one of them fails there and not in a wall
    /// of unexplained numbers.
    fn base() -> TimeInput {
        TimeInput {
            time_us: 0,
            inc_us: 0,
            byoyomi_us: 0,
            #[cfg(feature = "verbose2")]
            movetime: 0,
            #[cfg(feature = "verbose2")]
            rtime: 0,
            ply: 1,
            start_time: Instant::now(),
        }
    }

    #[test]
    fn the_compiled_settings_are_the_reference_option_defaults() {
        assert_eq!(
            (
                COMPILED.network_delay,
                COMPILED.network_delay2,
                COMPILED.minimum_thinking_time,
                COMPILED.slow_mover,
                COMPILED.round_up_to_fullsecond,
            ),
            (120, 1120, 2000, 100, true),
        );
    }

    fn init(input: &TimeInput) -> TimeManagement {
        init_settings(input, &COMPILED)
    }

    /// The budget at settings other than the compiled ones — how the arithmetic
    /// answers a setting is a property of the arithmetic, not of the one value
    /// this binary happens to carry.
    fn init_settings(input: &TimeInput, settings: &TimeSettings) -> TimeManagement {
        #[cfg(feature = "verbose2")]
        let mut prng = Prng::new(1);
        TimeManagement::init_with(
            input,
            settings,
            #[cfg(feature = "verbose2")]
            &mut prng,
        )
    }

    /// `go movetime` exists only from `verbose2` up, so the budget it pins does
    /// too.
    #[cfg(feature = "verbose2")]
    #[test]
    fn movetime_sets_all_three_to_movetime() {
        let tm = init(&TimeInput {
            movetime: 3000,
            ..base()
        });
        assert_eq!(tm.minimum(), 3000);
        assert_eq!(tm.optimum(), 3000);
        assert_eq!(tm.maximum(), 3000);
    }

    #[test]
    fn byoyomi_10min_plus_10s() {
        // 10 min main + 10 s byoyomi, no increment. Hand-computed against the
        // reference with the default options and ply 1.
        let input = TimeInput {
            time_us: 600_000,
            byoyomi_us: 10_000,
            ply: 1,
            ..base()
        };
        let tm = init(&input);

        // A byoyomi clock, so this is not the time-forfeit path and the
        // 5.0 max ratio applies. Neither the 0.3 remaining-time cap nor the
        // final-push branch binds here, so t1 / t2 pass through unchanged and
        // only the round-up-to-a-second step moves `maximum`.
        assert_eq!(tm.minimum(), 1880);
        assert_eq!(tm.optimum(), 17610);
        assert_eq!(tm.maximum(), 80880);
    }

    #[test]
    fn fischer_time_plus_increment() {
        // 5 min + 5 s increment, ply 20.
        let input = TimeInput {
            time_us: 300_000,
            inc_us: 5_000,
            ply: 20,
            ..base()
        };
        let tm = init(&input);

        // A Fischer clock: the increment is not folded into `remain_time`, but
        // it does raise `remain_estimate` over the move horizon.
        assert_eq!(tm.minimum(), 1880);
        assert_eq!(tm.optimum(), 9617);
        assert_eq!(tm.maximum(), 40880);
    }

    #[test]
    fn increment_is_excluded_from_remain_time() {
        // The Fischer increment is credited only *after* the move is played, so
        // it must not enter `remain_time`. A tiny main clock with a huge
        // increment makes the difference visible: `remain_time` binds both
        // optimum and maximum, and it comes from the main clock alone.
        let input = TimeInput {
            time_us: 5_000,
            inc_us: 60_000,
            ply: 1,
            ..base()
        };
        let tm = init(&input);

        // `remain_time` is 3880 here, and would be 63880 if the increment were
        // folded in; both t1 and t2 exceed it, so it binds optimum and maximum.
        assert_eq!(tm.minimum(), 1880);
        assert_eq!(tm.optimum(), 3880);
        assert_eq!(tm.maximum(), 3880);
    }

    #[test]
    fn sudden_death_ratio_uncapped_above_five_minutes() {
        // Sudden death, time >= 5 min: max_ratio stays 5.0.
        let input = TimeInput {
            time_us: 600_000,
            ply: 1,
            ..base()
        };
        let tm = init(&input);

        // No byoyomi and no increment, so this is the time-forfeit path: the
        // move horizon widens to 199 and `max_ratio` is clamped to 5.0.
        assert_eq!(tm.optimum(), 6930);
        assert_eq!(tm.maximum(), 27880);
    }

    #[test]
    fn sudden_death_ratio_clamped_under_one_minute() {
        // Sudden death, time < 1 min: max_ratio clamps to 1.0.
        let input = TimeInput {
            time_us: 30_000,
            ply: 1,
            ..base()
        };
        let tm = init(&input);

        // The clock is short enough that `remain_estimate` floors at 0, so t1
        // and t2 collapse to `minimum_time` and every budget is 1880.
        assert_eq!(tm.minimum(), 1880);
        assert_eq!(tm.optimum(), 1880);
        assert_eq!(tm.maximum(), 1880);
    }

    #[test]
    fn final_push_when_time_under_byoyomi_1_2() {
        // Byoyomi 1000, main clock 500 < 1000*1.2 = 1200: the final push spends
        // byoyomi + time.
        let input = TimeInput {
            time_us: 500,
            byoyomi_us: 1_000,
            ply: 50,
            ..base()
        };
        let tm = init(&input);

        // The main clock is under 1.2× the byoyomi, so the final-push branch
        // fires and every budget collapses to `remain_time`.
        assert_eq!(tm.minimum(), 380);
        assert_eq!(tm.optimum(), 380);
        assert_eq!(tm.maximum(), 380);
    }

    #[test]
    fn mtg_equal_one_spends_remaining() {
        // At the draw horizon (MTG == 1) all three times equal remain_time.
        // `max_moves_to_draw - ply + 2 == 2` gives MTG = 1, which is the ply
        // standing on the horizon itself.
        let input = TimeInput {
            time_us: 60_000,
            byoyomi_us: 5_000,
            ply: COMPILED.max_moves_to_draw,
            ..base()
        };
        let tm = init(&input);
        // remain_time = 60000 + 5000 - 1120 = 63880.
        assert_eq!(tm.minimum(), 63880);
        assert_eq!(tm.optimum(), 63880);
        assert_eq!(tm.maximum(), 63880);
    }

    #[test]
    fn mtg_non_positive_sets_error_and_500() {
        // `max_moves_to_draw - ply + 2 <= 0` -> MTG <= 0 error path: a ply past
        // the horizon, which the search itself would already have adjudicated.
        let input = TimeInput {
            time_us: 60_000,
            ply: COMPILED.max_moves_to_draw + 2,
            ..base()
        };
        let tm = init(&input);
        assert!(tm.mtg_error);
        assert_eq!(tm.minimum(), 500);
        assert_eq!(tm.optimum(), 500);
        assert_eq!(tm.maximum(), 500);
    }

    #[test]
    fn round_up_to_fullsecond_false_uses_units_of_one() {
        // With RoundUpToFullSecond off the minimum floor drops to 1 and round_up
        // does not round to a second.
        let input = TimeInput {
            time_us: 300_000,
            inc_us: 5_000,
            ply: 20,
            ..base()
        };
        let tm = init_settings(
            &input,
            &TimeSettings {
                round_up_to_fullsecond: false,
                ..COMPILED
            },
        );

        // With the per-move reserve off, `remain_estimate` keeps the whole
        // `(MTG + 1) * 1000`, and with round-up off `maximum` skips the
        // whole-second step.
        assert_eq!(tm.minimum(), 1880);
        assert_eq!(tm.optimum(), 10630);
        assert_eq!(tm.maximum(), 45510);
    }

    #[test]
    fn slow_mover_scales_optimum_only() {
        let baseline = init(&TimeInput {
            time_us: 600_000,
            byoyomi_us: 10_000,
            ply: 1,
            ..base()
        });
        let slow = init_settings(
            &TimeInput {
                time_us: 600_000,
                byoyomi_us: 10_000,
                ply: 1,
                ..base()
            },
            &TimeSettings {
                slow_mover: 200,
                ..COMPILED
            },
        );
        // optimum doubles (t1.min(remain) * 200/100), maximum unchanged.
        assert_eq!(slow.optimum(), baseline.optimum() * 2);
        assert_eq!(slow.maximum(), baseline.maximum());
    }

    /// `go rtime` exists only from `verbose2` up, so the budget it draws does
    /// too.
    #[cfg(feature = "verbose2")]
    #[test]
    fn rtime_result_within_bounds() {
        // rtime r plus a decaying random increment in [0, min(r/2, r*10/ply)).
        // ply 40 -> bound = min = 1000, so result in [4000, 5000).
        let r: i64 = 4000;
        let ply = 40;
        for seed in 1..50u64 {
            let mut prng = Prng::new(seed);
            let tm = TimeManagement::init(
                &TimeInput {
                    rtime: r,
                    ply,
                    ..base()
                },
                &mut prng,
            );
            assert!(
                tm.optimum() >= r && tm.optimum() < r + 1000,
                "rtime out of bounds: {} not in [{}, {})",
                tm.optimum(),
                r,
                r + 1000
            );
            assert_eq!(tm.minimum(), tm.optimum());
            assert_eq!(tm.maximum(), tm.optimum());
        }
    }

    /// A 3-minute (black) / 10-minute (white) clock with a 2-second increment
    /// and a 512-move cap, under the time settings above. Every case below is
    /// a Fischer clock, so the move horizon is `180 - min(ply, 80)`, and the
    /// minimum is `max(2000 - 120, 1000) = 1880`, which `round_up` keeps:
    /// `max(2000, 2000) - 120 = 1880`.
    mod match_clock {
        use super::*;

        const MATCH: TimeSettings = TimeSettings {
            network_delay: 120,
            network_delay2: 1120,
            minimum_thinking_time: 2000,
            slow_mover: 100,
            round_up_to_fullsecond: true,
            max_moves_to_draw: 512,
        };

        fn budget(time_us: i64, ply: i32) -> (i64, i64, i64) {
            let tm = init_settings(
                &TimeInput {
                    time_us,
                    inc_us: 2_000,
                    ply,
                    ..base()
                },
                &MATCH,
            );
            (tm.minimum(), tm.optimum(), tm.maximum())
        }

        #[test]
        fn black_first_move() {
            // remain_time = 180000 - 1120 = 178880; horizon 179;
            // MTG = min(512 - 1 + 2, 179) / 2 = 89.
            // estimate = 180000 + 2000*89 - 90*1000 = 268000.
            // t1 = 1880 + 268000/89 (3011) = 4891.
            // t2 = 1880 + 268000*5/89 (15056) = 16936, under 0.3 cap 80400.
            // maximum = round_up(16936): 17000 - 120 = 16880 < 16936 -> 17880.
            assert_eq!(budget(180_000, 1), (1880, 4891, 17880));
        }

        #[test]
        fn black_base_nearly_spent_in_the_middlegame() {
            // remain_time = 10000 - 1120 = 8880; horizon 100;
            // MTG = min(433, 100) / 2 = 50.
            // estimate = 10000 + 2000*50 - 51*1000 = 59000.
            // t1 = 1880 + 59000/50 (1180) = 3060.
            // t2 = 1880 + 59000*5/50 (5900) = 7780, under 0.3 cap 17700.
            // maximum = round_up(7780): 8000 - 120 = 7880.
            assert_eq!(budget(10_000, 81), (1880, 3060, 7880));
        }

        #[test]
        fn black_near_the_cap() {
            // remain_time = 4000 - 1120 = 2880; horizon 100;
            // MTG = min(512 - 501 + 2, 100) / 2 = 6.
            // estimate = 4000 + 2000*6 - 7*1000 = 9000.
            // t1 = 1880 + 9000/6 (1500) = 3380 -> capped at remain_time 2880.
            // t2 = 1880 + 9000*5/6 (7500) = 9380 -> 0.3 cap 2700.
            // maximum = round_up(2700): 3000 - 120 = 2880, capped at 2880.
            assert_eq!(budget(4_000, 501), (1880, 2880, 2880));
        }

        #[test]
        fn white_first_move() {
            // remain_time = 600000 - 1120 = 598880; horizon 178;
            // MTG = min(512, 178) / 2 = 89.
            // estimate = 600000 + 2000*89 - 90*1000 = 688000.
            // t1 = 1880 + 688000/89 (7730) = 9610.
            // t2 = 1880 + 688000*5/89 (38651) = 40531, under 0.3 cap 206400.
            // maximum = round_up(40531): 41000 - 120 = 40880.
            assert_eq!(budget(600_000, 2), (1880, 9610, 40880));
        }

        #[test]
        fn white_in_the_middlegame() {
            // remain_time = 300000 - 1120 = 298880; horizon 100;
            // MTG = min(432, 100) / 2 = 50.
            // estimate = 300000 + 2000*50 - 51*1000 = 349000.
            // t1 = 1880 + 349000/50 (6980) = 8860.
            // t2 = 1880 + 349000*5/50 (34900) = 36780, under 0.3 cap 104700.
            // maximum = round_up(36780): 37000 - 120 = 36880.
            assert_eq!(budget(300_000, 82), (1880, 8860, 36880));
        }

        #[test]
        fn white_near_the_cap() {
            // remain_time = 20000 - 1120 = 18880; horizon 100;
            // MTG = min(512 - 502 + 2, 100) / 2 = 6.
            // estimate = 20000 + 2000*6 - 7*1000 = 25000.
            // t1 = 1880 + 25000/6 (4166) = 6046.
            // t2 = 1880 + 25000*5/6 (20833) = 22713 -> 0.3 cap 7500.
            // maximum = round_up(7500): 8000 - 120 = 7880.
            assert_eq!(budget(20_000, 502), (1880, 6046, 7880));
        }
    }

    #[test]
    fn round_up_boundaries_full_second_branch() {
        // `round_up` reads only the option defaults (minimum_thinking_time 2000,
        // network_delay 120) and `remain_time`. A large clock keeps the
        // `remain_time` cap from binding, so the rounding alone is under test.
        let tm = init(&TimeInput {
            time_us: 1_000_000,
            ..base()
        });
        // Exactly on a whole second: 3000 -> ((3000+999)/1000)*1000 = 3000;
        // max=3000; -120 = 2880; 2880 < 3000 -> +1000 = 3880.
        assert_eq!(tm.round_up(3000), 3880);
        // Just over: 3001 -> ceil to 4000; -120 = 3880; 3880 >= 3001 -> 3880.
        assert_eq!(tm.round_up(3001), 3880);
        // Below the minimum floor: 100 -> ceil 1000; max=2000; -120 = 1880;
        // 1880 >= 100 -> 1880.
        assert_eq!(tm.round_up(100), 1880);
    }

    #[test]
    fn round_up_no_round_branch() {
        let tm = init_settings(
            &TimeInput {
                time_us: 1_000_000,
                ..base()
            },
            &TimeSettings {
                round_up_to_fullsecond: false,
                ..COMPILED
            },
        );
        // max=3001; -120 = 2881.
        assert_eq!(tm.round_up(3001), 2881);
        // max=2000; -120 = 1880.
        assert_eq!(tm.round_up(100), 1880);
    }

    #[test]
    fn set_search_end_rounds_elapsed() {
        // search_end = round_up(max(elapsed, minimum())).
        let mut tm = init(&TimeInput {
            time_us: 600_000,
            byoyomi_us: 10_000,
            ply: 1,
            ..base()
        });
        // minimum() == 1880 here. Elapsed 5000 > minimum, so search_end =
        // round_up: ceil 5000; -120 = 4880; 4880<5000 -> 5880.
        tm.set_search_end(5000);
        assert_eq!(tm.search_end, 5880);

        // Elapsed below minimum uses minimum(): max = 1880; round_up = 1880.
        tm.search_end = 0;
        tm.set_search_end(500);
        assert_eq!(tm.search_end, 1880);
    }

    #[test]
    fn set_search_end_in_byoyomi_is_capped_by_the_remaining_time() {
        // Byoyomi with (almost) no main clock: the whole of it is this move's
        // budget, and `round_up`'s cap is what keeps the rounding inside it.
        let mut tm = init(&TimeInput {
            time_us: 500,
            byoyomi_us: 1_000,
            ply: 50,
            ..base()
        });
        // minimum() == 380 (clamped to remain_time). Elapsed 100 < minimum, so
        // max = 380; round_up: ceil 1000; max=2000; -120=1880; 1880<380? no;
        // min(1880, remain_time=380) = 380.
        tm.set_search_end(100);
        assert_eq!(tm.search_end, 380);
    }

    /// The 5 min + 5 s, ply 20 budget of [`fischer_time_plus_increment`]
    /// (minimum 1880, optimum 9617, maximum 40880), credited with `credit` ms,
    /// and an instant `real` ms after its `go` arrived.
    fn credited(credit: i64, real: u64) -> (TimeManagement, Instant) {
        let mut tm = init(&TimeInput {
            time_us: 300_000,
            inc_us: 5_000,
            ply: 20,
            ..base()
        });
        tm.set_credit(credit);
        let now = tm.start_time + std::time::Duration::from_millis(real);
        (tm, now)
    }

    #[test]
    fn a_credit_counts_as_elapsed_and_leaves_the_budget_alone() {
        let (plain, plain_now) = credited(0, 700);
        let (tm, now) = credited(3_000, 700);
        assert_eq!(
            (tm.minimum(), tm.optimum(), tm.maximum()),
            (plain.minimum(), plain.optimum(), plain.maximum()),
            "the budget and its real-time clamps do not read the credit"
        );
        assert_eq!(
            tm.elapsed_from(now),
            700,
            "the real clock is the real clock"
        );
        assert_eq!(tm.credited_elapsed_from(now), 3_700);
        assert_eq!(plain.credited_elapsed_from(plain_now), 700);

        let mut clamped = tm.clone();
        clamped.set_credit(-5);
        assert_eq!(clamped.credit(), 0);
        clamped.set_credit(i64::MAX);
        assert_eq!(clamped.credit(), i64::from(u32::MAX));
    }

    /// The two checks the search makes — past the optimum, past the maximum —
    /// read the credited time, so each fires `credit` ms of real time sooner.
    #[test]
    fn a_credit_brings_the_optimum_and_maximum_checks_forward_by_itself() {
        const C: i64 = 4_000;
        for limit in [9617, 40880] {
            let fires = |credit: i64, real: i64| {
                let (tm, now) = credited(credit, real as u64);
                tm.credited_elapsed_from(now) > limit
            };
            assert!(
                !fires(0, limit) && fires(0, limit + 1),
                "uncredited at {limit}"
            );
            assert!(
                !fires(C, limit - C) && fires(C, limit - C + 1),
                "credited {C} ms at {limit}"
            );
        }
    }

    /// A credit beyond the maximum makes the very first check stop the search:
    /// the end it fixes rounds the real time up to the wall second the clock
    /// charges anyway, with no minimum think time under it.
    #[test]
    fn a_credit_past_the_maximum_ends_the_search_at_the_first_check() {
        let (mut tm, now) = credited(50_000, 30);
        let e = tm.credited_elapsed_from(now);
        assert!(e > tm.maximum());
        tm.set_search_end(e);
        // Real 30: ceil 1000; -120 = 880; not below 30.
        assert_eq!(tm.search_end, 880);
        assert!(
            tm.search_end < tm.minimum(),
            "no floor under a credited end"
        );

        // Real 1500: ceil 2000; -120 = 1880.
        let (mut tm, now) = credited(50_000, 1500);
        tm.set_search_end(tm.credited_elapsed_from(now));
        assert_eq!(tm.search_end, 1880);
        // Real 1950: ceil 2000; -120 = 1880 < 1950, so 2880.
        let (mut tm, now) = credited(50_000, 1950);
        tm.set_search_end(tm.credited_elapsed_from(now));
        assert_eq!(tm.search_end, 2880);
    }

    /// The allocation with the reference's literals written in, where the
    /// engine reads its time parameters: what [`TimeManagement::init_with`]
    /// gives at the parameters' defaults, for a clock, the move-number terms
    /// and the settings, outside `go movetime` and `go rtime`.
    fn literal_budget(input: &TimeInput, s: &TimeSettings) -> (i64, i64, i64) {
        let (time_us, inc_us, byoyomi_us, ply) =
            (input.time_us, input.inc_us, input.byoyomi_us, input.ply);
        let remain_time = (time_us + byoyomi_us - s.network_delay2)
            .max(if s.round_up_to_fullsecond { 100 } else { 1 });
        let time_forfeit = inc_us == 0 && byoyomi_us == 0;
        let move_horizon = if time_forfeit {
            160 + 40 - ply.min(40)
        } else {
            160 + 20 - ply.min(80)
        };
        let mtg = (s.max_moves_to_draw - ply + 2).min(move_horizon) / 2;
        if mtg <= 0 {
            return (500, 500, 500);
        }
        if mtg == 1 {
            return (remain_time, remain_time, remain_time);
        }
        let minimum_time = (s.minimum_thinking_time - s.network_delay)
            .max(if s.round_up_to_fullsecond { 1000 } else { 1 });
        let mut remain_estimate = time_us + inc_us * mtg as i64 + byoyomi_us * mtg as i64;
        if s.round_up_to_fullsecond {
            remain_estimate -= (mtg as i64 + 1) * 1000;
        }
        remain_estimate = remain_estimate.max(0);
        let t1 = minimum_time + remain_estimate / mtg as i64;
        let mut max_ratio = 5.0f32;
        if time_forfeit {
            max_ratio = max_ratio.min((time_us as f32 / (60.0 * 1000.0)).max(1.0));
        }
        let mut t2 = minimum_time + (remain_estimate as f32 * max_ratio / mtg as f32) as i64;
        t2 = t2.min((remain_estimate as f64 * 0.3) as i64);
        let (mut minimum, mut optimum, mut maximum) = (
            minimum_time,
            t1.min(remain_time) * s.slow_mover / 100,
            t2.min(remain_time),
        );
        if byoyomi_us != 0 && time_us < (byoyomi_us as f64 * 1.2) as i64 {
            let v = byoyomi_us + time_us;
            (minimum, optimum, maximum) = (v, v, v);
        }
        let round_up = |t0: i64| {
            let t = if s.round_up_to_fullsecond {
                let mut t =
                    (((t0 + 999) / 1000) * 1000).max(s.minimum_thinking_time) - s.network_delay;
                if t < t0 {
                    t += 1000;
                }
                t
            } else {
                t0.max(s.minimum_thinking_time) - s.network_delay
            };
            t.min(remain_time)
        };
        (
            round_up(minimum).min(remain_time),
            optimum.min(remain_time),
            round_up(maximum).min(remain_time),
        )
    }

    /// Every clock shape (sudden death, increment, byoyomi, a spent main
    /// clock), from the first move to past the draw horizon, under the
    /// compiled settings and under a match-like set without the whole-second
    /// rounding.
    fn assert_the_defaults_allocate_as_the_literals() {
        let local = TimeSettings {
            network_delay: 20,
            network_delay2: 100,
            minimum_thinking_time: 100,
            slow_mover: 100,
            round_up_to_fullsecond: false,
            max_moves_to_draw: 320,
        };
        let clocks = [
            (0, 0, 0),
            (500, 0, 1_000),
            (5_000, 60_000, 0),
            (10_000, 100, 0),
            (30_000, 0, 0),
            (60_000, 0, 5_000),
            (180_000, 2_000, 0),
            (300_000, 5_000, 0),
            (600_000, 0, 0),
            (600_000, 0, 10_000),
            (900_000, 10_000, 0),
            (3_600_000, 0, 0),
        ];
        let plies = [
            1, 2, 20, 39, 40, 41, 79, 80, 81, 150, 250, 318, 319, 320, 322,
        ];
        for settings in [COMPILED, local] {
            for (time_us, inc_us, byoyomi_us) in clocks {
                for ply in plies {
                    let input = TimeInput {
                        time_us,
                        inc_us,
                        byoyomi_us,
                        ply,
                        ..base()
                    };
                    let tm = init_settings(&input, &settings);
                    assert_eq!(
                        (tm.minimum(), tm.optimum(), tm.maximum()),
                        literal_budget(&input, &settings),
                        "{settings:?} {time_us}+{inc_us}/{byoyomi_us} at ply {ply}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_time_parameters_defaults_allocate_as_the_literals() {
        assert_the_defaults_allocate_as_the_literals();
    }

    /// No credit is today's arithmetic, floor included.
    #[test]
    fn no_credit_ends_the_search_as_it_always_has() {
        for real in [30, 700, 5000] {
            let (mut tm, now) = credited(0, real);
            let mut reference = tm.clone();
            tm.set_search_end(tm.credited_elapsed_from(now));
            let e = reference.elapsed_from(now);
            reference.search_end = reference.round_up(e.max(reference.minimum()));
            assert_eq!(tm.search_end, reference.search_end, "real {real}");
        }
        let (mut tm, now) = credited(0, 30);
        tm.set_search_end(tm.credited_elapsed_from(now));
        assert_eq!(tm.search_end, 1880, "the minimum think time is the floor");
    }
}
