//! The one place that decides where a setting's value comes from.
//!
//! There is exactly one answer, in every build: the compile-time constant
//! [`crate::config`] generated from the TOML config. No build has a runtime
//! option surface — no declaration table, no `setoption`, no option-override
//! file, no profile file.
//!
//! The accessors stay because the driver reads settings by name, and keeping the
//! name → constant mapping in one file is what makes "which constant does
//! `MultiPV` mean" answerable in one place.
//!
//! Only the settings the driver still asks for by name are here. A setting the
//! search or the storage layer reads has no accessor: those layers compile the
//! same constant in for themselves, so nothing hands the value over at run time.
//! `usi_hash` is the clearest case — the transposition table is a `static` whose
//! length is that value — and the book-selection group, the draw and resign
//! values, the entering-king rule and the time-management settings are the same
//! shape, folded into [`crate::driver::BOOK_CONFIG`], the driver's own constants
//! and the search layer's.

/// The evaluation-noise amplitude, in the unit the search scores in.
///
/// The `random` setting is in centipawns and a search value is in `PawnValue`
/// units, so the conversion is `random * PawnValue / 100` — the inverse of the
/// centipawn rendering, which makes `random = 100` a whole pawn wide. A `const`
/// rather than an accessor because the search's noise bound is fixed when the
/// binary is built, like every other setting here.
#[cfg(feature = "random")]
pub(crate) const RANDOM_AMPLITUDE: yorkie_storage::Value =
    (crate::config::RANDOM * crate::driver::PAWN_VALUE as i64 / 100) as yorkie_storage::Value;

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

    /// The CPUs of every NUMA node, in node order: the layout this binary was
    /// built for.
    pub(crate) fn numa_node_cpus(&self) -> &'static [&'static [usize]] {
        crate::config::NUMA_NODE_CPUS
    }

    /// The kernel's node number for each node, aligned with
    /// [`Self::numa_node_cpus`].
    pub(crate) fn numa_system_nodes(&self) -> &'static [usize] {
        crate::config::NUMA_SYSTEM_NODES
    }

    /// The logical CPU each worker is pinned to, indexed by worker id.
    pub(crate) fn worker_cpus(&self) -> &'static [usize] {
        crate::config::WORKER_CPUS
    }

    /// The system NUMA node of each worker's CPU, aligned with
    /// [`Self::worker_cpus`].
    pub(crate) fn worker_system_nodes(&self) -> &'static [usize] {
        crate::config::WORKER_SYSTEM_NODES
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
    /// Principal variations reported per search (`MultiPV`). Read only where a
    /// second PV line can be reported, which is where the search `info` lines
    /// are; without that feature the root is single-line and the constant does
    /// not exist.
    #[cfg(feature = "verbose2")]
    multi_pv => MULTI_PV;
    /// Per-`go` search-depth ceiling, `0` unlimited (`DepthLimit`). A ceiling
    /// bounds a search by something other than the clock, which is what an
    /// analysis session asks for and a rated game never does, so it shares the
    /// feature of the `go depth` clause that also sets it; without that feature
    /// the engine has no depth ceiling and the constant does not exist.
    #[cfg(feature = "verbose2")]
    depth_limit => DEPTH_LIMIT;
    /// Per-`go` node ceiling, `0` unlimited (`NodesLimit`). Gated for the same
    /// reason as the depth ceiling above, with `go nodes` as its other source.
    #[cfg(feature = "verbose2")]
    nodes_limit => NODES_LIMIT;
    /// PV-output throttle in milliseconds (`PvInterval`). Read only where a PV
    /// is printed.
    #[cfg(feature = "verbose2")]
    pv_interval => PV_INTERVAL;
}

check_accessors! {
    /// Consult the book at all (`USI_OwnBook`).
    usi_own_book => USI_OWN_BOOK;
    /// Stream the book from disk rather than loading it (`BookOnTheFly`).
    book_on_the_fly => BOOK_ON_THE_FLY;
    /// Match book positions ignoring their recorded ply (`IgnoreBookPly`).
    ignore_book_ply => IGNORE_BOOK_PLY;
    /// Collect each PV from the transposition table (`ConsiderationMode`). Both
    /// this and the fail-high/low toggle below shape a printed PV only, so they
    /// are read only where one is printed.
    #[cfg(feature = "verbose2")]
    consideration_mode => CONSIDERATION_MODE;
    /// Emit a PV on a fail-high / fail-low (`OutputFailLHPV`).
    #[cfg(feature = "verbose2")]
    output_fail_lh_pv => OUTPUT_FAIL_LH_PV;
    /// Stochastic-ponder toggle (`Stochastic_Ponder`), which also decides
    /// whether a `go ponder` rewinds the retained position by one move.
    stochastic_ponder => STOCHASTIC_PONDER;
}

text_accessors! {
    /// Book file name, `no_book` for bookless (`BookFile`).
    book_file => BOOK_FILE;
    /// Book directory (`BookDir`).
    book_dir => BOOK_DIR;
}
