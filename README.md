# Yorkie

Yorkie is a Shogi AI engine whose upstream is apery_rust (https://github.com/HiraokaTakuya/apery_rust). It also features capabilities ported from YaneuraOu (https://github.com/yaneurao/YaneuraOu).

## Building (NNUE release, Ubuntu 24.04 / x64)

Two scripts under `scripts/` build an optimized NNUE release binary tuned for the host CPU:

- **`scripts/setup.sh`** — the one-command path from a fresh `git clone` on a bare host. It
  installs the prerequisites a plain Ubuntu lacks (a C toolchain via `build-essential`, plus
  `curl`, and the Rust stable toolchain via rustup), then runs `build.sh`. It modifies the
  system (uses `apt` via `sudo` when not root) and is idempotent — if Rust is already present,
  the rustup install is skipped.
- **`scripts/build.sh`** — just the build, for hosts that already have the Rust toolchain (e.g.
  CI). Run this directly when the prerequisites are already in place.

```sh
scripts/setup.sh      # fresh host: install prerequisites, then build
# or
scripts/build.sh      # toolchain already installed: build only
```

Both produce `./target/release/yorkie` via
`cargo build --release --no-default-features --features nnue` with
`RUSTFLAGS="-C target-cpu=native"`. The release profile enables LTO, and
`-C target-cpu=native` selects the SIMD (AVX-512 / VNNI) fast paths at build time while
keeping the portable scalar fallback. Both exit non-zero with an actionable message if a
prerequisite is missing or the build fails.

The scripts only provision and build the engine. The runtime assets (the NNUE network and a
YaneuraOu-format opening book) are not downloaded, generated, or wired up by them — they are
located at runtime through USI options (`Eval_Dir` for the network directory, `Book_File` for
the book) when the engine starts.
