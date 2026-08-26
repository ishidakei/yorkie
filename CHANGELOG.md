# Changelog

## 3.1.0

Global allocator switched to mimalloc
(~+2% NPS measured at 1-2 threads; the
allocator-contention benefit grows with
thread count). Toolchain pinned to Rust
1.98.0. Test coverage expanded with
property tests (do/undo round-trip,
move-generation invariants,
incremental-vs-refresh eval parity);
quality gates tightened. Module layout
modernized (sibling-file form,
lint-enforced).

## 3.0.0

The tree is replaced wholesale with the
YaneuraOu Rust port, imported from
attic-shogi. The apery_rust-based tree
it replaces is preserved as v2.0.0 in this
repository's history.

## 2.0.0

apery_rust-based engine. Entered
Denryu-sen 7 TSEC part 1.
