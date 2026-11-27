//! Both sides' clocks through a game, kept the way the server keeps them.
//!
//! The server is the authority on time: every move it relays carries `,T<n>`,
//! the time it charged that move in its own unit, and the client's clock is
//! that figure applied to the time control, for the engine's own moves as much
//! as for the opponent's. `shogi-server` charges a move by adding the increment
//! to the mover's clock, taking the charged time off it and holding it at zero;
//! byoyomi is never added to the clock, it is the time a move may run past it.
//! [`Clock::charge`] is that arithmetic.

use yorkie_state::Color;

use super::summary::SideTime;
use crate::engine::GoParams;

/// Both sides' remaining time and their time controls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Clock {
    sides: [SideClock; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SideClock {
    time: SideTime,
    /// What is left on the main clock, in the server's unit.
    remaining: u64,
}

impl Clock {
    /// The clocks at the start of a game under `time`, indexed by
    /// [`Color::index`]: each side's total, or the remaining time a resumed game
    /// gave.
    pub(crate) fn new(time: &[SideTime; 2]) -> Self {
        let side = |t: SideTime| SideClock {
            time: t,
            remaining: t.remaining.unwrap_or(t.total),
        };
        Self {
            sides: [side(time[0]), side(time[1])],
        }
    }

    /// Charge `side` for a move the server timed at `units`.
    pub(crate) fn charge(&mut self, side: Color, units: u64) {
        let c = &mut self.sides[side.index()];
        c.remaining = (c.remaining + c.time.increment).saturating_sub(units);
    }

    /// What is left on `side`'s main clock, in the server's unit.
    pub(crate) fn remaining_units(&self, side: Color) -> u64 {
        self.sides[side.index()].remaining
    }

    /// What is left on `side`'s main clock, in milliseconds.
    pub(crate) fn remaining_ms(&self, side: Color) -> u64 {
        let c = &self.sides[side.index()];
        c.remaining.saturating_mul(c.time.unit_ms)
    }

    /// The limits a USI `go btime … wtime … binc … winc … byoyomi …` would carry
    /// for a move by `to_move`: the clocks as they stand, each side's
    /// increment, and the byoyomi of the side to move — USI has one byoyomi
    /// value, and it is the mover's that bounds this move. A clause the time
    /// control does not use is left out, as a host leaves it out of a `go`.
    pub(crate) fn go_params(&self, to_move: Color) -> GoParams {
        let ms =
            |side: Color, units: u64| units.saturating_mul(self.sides[side.index()].time.unit_ms);
        let inc = |side: Color| {
            let i = self.sides[side.index()].time.increment;
            (i > 0).then(|| ms(side, i))
        };
        let byoyomi = self.sides[to_move.index()].time.byoyomi;
        GoParams {
            btime: Some(self.remaining_ms(Color::Black)),
            wtime: Some(self.remaining_ms(Color::White)),
            binc: inc(Color::Black),
            winc: inc(Color::White),
            byoyomi: (byoyomi > 0).then(|| ms(to_move, byoyomi)),
            ..GoParams::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn side(unit_ms: u64, total: u64, byoyomi: u64, increment: u64) -> SideTime {
        SideTime {
            unit_ms,
            total,
            byoyomi,
            increment,
            least: 0,
            remaining: None,
        }
    }

    #[test]
    fn an_increment_is_added_before_the_charge_is_taken() {
        let mut c = Clock::new(&[side(1000, 180, 0, 2), side(1000, 600, 0, 2)]);
        c.charge(Color::Black, 5);
        assert_eq!(c.remaining_units(Color::Black), 177);
        c.charge(Color::White, 0);
        assert_eq!(c.remaining_units(Color::White), 602);
        assert_eq!(c.remaining_ms(Color::White), 602_000);
    }

    #[test]
    fn the_clock_holds_at_zero_and_byoyomi_is_never_added() {
        let mut c = Clock::new(&[side(1000, 10, 30, 0), side(1000, 10, 30, 0)]);
        c.charge(Color::Black, 25);
        assert_eq!(c.remaining_units(Color::Black), 0);
        c.charge(Color::Black, 29);
        assert_eq!(c.remaining_units(Color::Black), 0);
    }

    #[test]
    fn a_resumed_game_starts_from_the_remaining_time() {
        let mut t = side(1000, 600, 0, 10);
        t.remaining = Some(321);
        let c = Clock::new(&[t, side(1000, 600, 0, 10)]);
        assert_eq!(c.remaining_units(Color::Black), 321);
        assert_eq!(c.remaining_units(Color::White), 600);
    }

    #[test]
    fn go_carries_both_clocks_both_increments_and_the_movers_byoyomi() {
        let c = Clock::new(&[side(1000, 180, 0, 2), side(1000, 600, 0, 2)]);
        let p = c.go_params(Color::White);
        assert_eq!(p.btime, Some(180_000));
        assert_eq!(p.wtime, Some(600_000));
        assert_eq!(p.binc, Some(2000));
        assert_eq!(p.winc, Some(2000));
        assert_eq!(p.byoyomi, None);

        let c = Clock::new(&[side(1000, 600, 10, 0), side(1, 0, 5000, 0)]);
        let p = c.go_params(Color::White);
        assert_eq!(p.btime, Some(600_000));
        assert_eq!(p.wtime, Some(0));
        assert_eq!(p.binc, None);
        assert_eq!(p.winc, None);
        assert_eq!(p.byoyomi, Some(5000));
        assert_eq!(c.go_params(Color::Black).byoyomi, Some(10_000));
    }

    #[test]
    fn a_minute_unit_scales_to_milliseconds() {
        let mut c = Clock::new(&[side(60_000, 15, 0, 0), side(60_000, 15, 0, 0)]);
        c.charge(Color::Black, 2);
        assert_eq!(c.remaining_ms(Color::Black), 13 * 60_000);
    }
}
