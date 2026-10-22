# Changelog

## 3.2.0

Settings are compiled in from a TOML
file under `configs/` (a plain build
reads the play config); the engine has
no USI options, reads no option files,
and ignores `setoption`. Commands and
output are chosen by three Cargo
features, `verbose1` to `verbose3`: the
build with none of them is the match
build and prints only `bestmove`;
MultiPV and the analysis `go` limits
need `verbose2`, `bench` needs
`verbose3`. `bestmove` carries the move
alone. USI pondering is replaced by a
`ponder` setting that keeps searching
after the reply until the next command.
Search `info` lines report `nps`,
`time` and `hashfull`, and drop a zero
`seldepth`; `verbose1` adds a per-reply
statistics line. The NUMA layout and
each worker's logical CPU are fixed at
build time, and `isready` refuses a
machine that differs. The network is
converted at build time into a
kernel-ready file; the transposition
table is a build-time-sized static. New
off-by-default features: `tt-entry16`
(full 64-bit table keys) and `random`
(per-game evaluation noise). Hot-path
work (inline buffers, no allocation per
`go` or `position`, settings read as
constants, `-C jump-tables=no`) raises
search throughput on unchanged node
counts. Release builds keep the line
table. Toolchain pinned to Rust 1.98.1.

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
