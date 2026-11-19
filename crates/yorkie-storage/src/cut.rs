//! Switches that cut one channel between the workers of a parallel search, for
//! measuring which shared state changes the main worker's tree.
//!
//! The switches are read once, from the `YORKIE_DUPSTATS_CUT` environment
//! variable: a comma-separated list of [`Cut`] names. With the variable unset
//! or empty every switch is off and a `dupstats` build searches as a build
//! without the feature does. The module exists only in a `dupstats` build.
//!
//! A switch can change what a lone search does too (a private table region is
//! smaller than the whole table; a main-worker restriction applies to a lone
//! search, whose one worker is the main one), so a measurement compares a
//! many-worker search with a one-worker search under the same switches.

use std::cell::Cell;
use std::sync::OnceLock;

/// One channel cut, or one restriction on what the main worker takes from a
/// table entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cut {
    /// Every worker probes and stores in a region of the table of its own:
    /// worker `w` in the `w`-th of [`REGIONS`] equal slices.
    TtPrivate,
    /// Workers other than the main one store nothing in the table.
    HelperNoStore,
    /// The shared correction and pawn tables are sized for one worker, as a
    /// lone search's are, instead of for the node's worker count.
    HistSize1,
    /// Every worker has a correction table of its own, sized for one worker.
    CorrPrivate,
    /// Every worker has a pawn-history table of its own, sized for one worker.
    PawnPrivate,
    /// The main worker takes no table cutoff, in the main search or in the
    /// quiescence search.
    MainNoCut,
    /// The main worker uses no entry's move past the cutoff: no move to order
    /// first, no "the entry's move is a capture", no singular extension, and
    /// every other condition on the move reads "no move".
    MainNoMove,
    /// The main worker's move pickers are given no entry move; every other use
    /// of the move stays.
    MainNoOrder,
    /// The main worker uses no entry's value past the cutoff: not as the
    /// node's evaluation, nor in ProbCut, the singular extension or any
    /// reduction.
    MainNoValue,
    /// The main worker ignores the static evaluation an entry stores and
    /// evaluates the position itself.
    MainNoEval,
    /// The main worker reads every entry's "was a PV node" flag as clear.
    MainNoPv,
}

impl Cut {
    const ALL: [(Cut, &'static str); 11] = [
        (Cut::TtPrivate, "tt_private"),
        (Cut::HelperNoStore, "helper_nostore"),
        (Cut::HistSize1, "hist_size1"),
        (Cut::CorrPrivate, "corr_private"),
        (Cut::PawnPrivate, "pawn_private"),
        (Cut::MainNoCut, "main_no_cut"),
        (Cut::MainNoMove, "main_no_move"),
        (Cut::MainNoOrder, "main_no_order"),
        (Cut::MainNoValue, "main_no_value"),
        (Cut::MainNoEval, "main_no_eval"),
        (Cut::MainNoPv, "main_no_pv"),
    ];

    const fn bit(self) -> u32 {
        1 << self as u32
    }

    /// The switch named `name`, as `YORKIE_DUPSTATS_CUT` spells it.
    pub fn named(name: &str) -> Option<Cut> {
        Self::ALL.iter().find(|(_, n)| *n == name).map(|(c, _)| *c)
    }
}

/// The number of equal slices [`Cut::TtPrivate`] divides the table into, one
/// per worker; a worker past the last takes its index modulo this.
pub const REGIONS: usize = 64;

static CUTS: OnceLock<u32> = OnceLock::new();

/// The switches `spec` names, as a bit set. Panics on a name that is not one,
/// so a misspelt switch cannot pass for a measurement with it off.
pub fn parse(spec: &str) -> u32 {
    spec.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            Cut::named(s)
                .unwrap_or_else(|| panic!("YORKIE_DUPSTATS_CUT names no switch `{s}`"))
                .bit()
        })
        .fold(0, |a, b| a | b)
}

fn cuts() -> u32 {
    *CUTS.get_or_init(|| parse(&std::env::var("YORKIE_DUPSTATS_CUT").unwrap_or_default()))
}

/// Whether switch `cut` is on.
#[inline]
pub fn on(cut: Cut) -> bool {
    cuts() & cut.bit() != 0
}

/// Whether switch `cut` is on and the calling thread is running the main
/// worker's search.
#[inline]
pub fn on_main(cut: Cut) -> bool {
    on(cut) && worker() == 0
}

thread_local! {
    static WORKER: Cell<u8> = const { Cell::new(0) };
}

/// Record that the calling thread runs worker `index`'s search, until it is
/// told otherwise.
pub fn set_worker(index: u8) {
    WORKER.with(|w| w.set(index));
}

/// The worker the calling thread last said it runs; `0` (the main worker) on a
/// thread that never said.
#[inline]
pub fn worker() -> u8 {
    WORKER.with(Cell::get)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_list_sets_the_bits_it_names_and_nothing_else() {
        let bits = parse("tt_private, main_no_cut,");
        assert_eq!(bits, Cut::TtPrivate.bit() | Cut::MainNoCut.bit());
        assert_eq!(parse(""), 0);
    }

    #[test]
    #[should_panic(expected = "names no switch")]
    fn a_misspelt_switch_is_refused() {
        parse("tt_privat");
    }

    #[test]
    fn every_switch_has_a_distinct_name_and_bit() {
        for (i, (c, n)) in Cut::ALL.iter().enumerate() {
            assert_eq!(Cut::named(n), Some(*c));
            assert_eq!(c.bit(), 1 << i);
        }
    }
}
