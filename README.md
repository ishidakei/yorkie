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

### Tournament (production) build

A tournament binary is built with one command, naming the event config TOML:

```sh
YORKIE_TOURNAMENT_CONFIG=configs/tsec7-part1.toml cargo tournament
```

The `tournament` alias (defined in `.cargo/config.toml`) expands to a release build with
features `nnue,tournament,numa` and `-C target-cpu=native`. The startup-fixed values from
the named config TOML are baked in as compile-time consts, search info output is compiled
out (the engine prints `bestmove` only), and MultiPV, the JSON book backend, and the mate
solver are excluded from the binary.

The event is chosen by the TOML path alone — the alias has no default. Running
`cargo tournament` without `YORKIE_TOURNAMENT_CONFIG` fails at build time with
`` the `tournament` feature requires YORKIE_TOURNAMENT_CONFIG=configs/<event>.toml ``.

**Memory warning**: `configs/tsec7-part1.toml` bakes a transposition table of roughly
288 GiB, sized for a 384 GiB-class host. On smaller machines, write your own
`configs/<event>.toml` (see `configs/ci.toml` for a minimal example) and point
`YORKIE_TOURNAMENT_CONFIG` at it.

Note: a real `RUSTFLAGS` environment variable set by the caller overrides the alias's
`build.rustflags`, replacing `-C target-cpu=native`.
