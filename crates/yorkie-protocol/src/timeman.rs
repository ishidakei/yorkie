//! Not a module: this file is deliberately absent from `lib.rs`, so nothing in
//! it is compiled.
//!
//! Time management is not a Protocol-layer concern. The faithful reference
//! `TimeManagement` port lives in `yorkie_search::timeman`; the USI driver builds
//! an `yorkie_search::TimeControl` from the `go` limits and the time options
//! (`NetworkDelay` / `NetworkDelay2` / `MinimumThinkingTime` / `SlowMover` /
//! `RoundUpToFullSecond`) and hands it to the search.
