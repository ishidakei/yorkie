//! Pairs of functions that show a search parameter compiles to its constant.
//!
//! Each pair has one function that reads a parameter the way the search does,
//! through `params::NAME.get()` or `.scaled(..)`, and one that does the same
//! arithmetic with the parameter's value written in. The two compile to the
//! same machine code: a divisor becomes a multiply and shift, a comparison
//! folds into the branch, and nothing is loaded from memory. The
//! `param-probes` binary, built in release mode, carries both functions of
//! every pair, and disassembling it shows whether their instructions are
//! identical.
//!
//! The arithmetic of each pair mirrors one use site in the search, so the
//! types and the operations are the ones the search compiles.
//!
//! The value written in is the integer the build generated for the parameter
//! from its SPSA settings file, a bare `i32` constant item: the written-in
//! function binds it to a local and computes with that local, which the
//! compiler sees as the literal itself. So the pair compares the read through
//! `Param` with the literal whatever value the build was given.
//!
//! Every function is exported under its own unmangled name. The compiler
//! merges two functions it finds identical into one body, and without the
//! export the link-time optimiser would then drop the second name; exported, it
//! stays as a second symbol at the same address, so a merged pair is visible as
//! one.

use crate::params;

macro_rules! probe {
    (
        $(#[doc = $doc:literal])*
        $read:ident, $written:ident: $param:ident;
        fn($($arg:ident: $ty:ty),*) -> $ret:ty;
        read = $read_body:expr;
        written($k:ident) = $written_body:expr;
    ) => {
        $(#[doc = $doc])*
        #[inline(never)]
        #[unsafe(no_mangle)]
        pub fn $read($($arg: $ty),*) -> $ret {
            $read_body
        }

        $(#[doc = $doc])*
        ///
        /// The parameter's value written in.
        #[inline(never)]
        #[unsafe(no_mangle)]
        pub fn $written($($arg: $ty),*) -> $ret {
            let $k: i32 = params::values::$param;
            $written_body
        }
    };
}

probe! {
    /// An integer division by a divisor parameter: the aspiration window's
    /// half-width from the root score's mean square.
    probe_divide_read, probe_divide_written: ASPIRATION_SPREAD_DIV;
    fn(mean_square: u64) -> u64;
    read = mean_square / u64::from(params::ASPIRATION_SPREAD_DIV.get().unsigned_abs());
    written(k) = mean_square / u64::from(k.unsigned_abs());
}

probe! {
    /// A comparison against a parameter that decides a branch: the depth
    /// futility pruning applies below.
    probe_compare_read, probe_compare_written: FUTILITY_MAX_DEPTH;
    fn(depth: i32, eval: i32, margin: i32) -> i32;
    read = if depth < params::FUTILITY_MAX_DEPTH.get() {
        eval.wrapping_sub(margin.wrapping_mul(depth))
    } else {
        eval
    };
    written(k) = if depth < k {
        eval.wrapping_sub(margin.wrapping_mul(depth))
    } else {
        eval
    };
}

probe! {
    /// A coefficient read as a fraction: the share of the remaining time the
    /// maximum is capped at.
    probe_scaled_read, probe_scaled_written: TM_MAX_SHARE;
    fn(remain: i64, t: i64) -> i64;
    read = t.min((remain as f64 * params::TM_MAX_SHARE.scaled(1000.0)) as i64);
    written(k) = t.min((remain as f64 * (f64::from(k) / 1000.0)) as i64);
}

probe! {
    /// A small divisor read as `x * 1024 / k`: the depth term of the null-move
    /// reduction.
    probe_fraction_read, probe_fraction_written: NMP_DEPTH_DIV;
    fn(depth: i32) -> i32;
    read = depth * 1024 / params::NMP_DEPTH_DIV.get();
    written(k) = depth * 1024 / k;
}
