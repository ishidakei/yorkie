#!/usr/bin/env sh
#
# build.sh — build Yorkie as an optimized NNUE release binary from a fresh clone.
#
# Target platform : Ubuntu 24.04 on an x64 CPU.
# What it does    : builds the engine with the `nnue` evaluation feature as a release
#                   build optimized for the host CPU (`-C target-cpu=native`), so the
#                   SIMD (AVX-512 / VNNI) fast paths are selected at build time while the
#                   portable scalar fallback is retained. On success the binary is at
#                   ./target/release/yorkie.
# What it is NOT  : it does not install the toolchain, nor download, generate, or wire up
#                   the runtime assets (the NNUE network and the opening book). The Rust
#                   toolchain is a prerequisite (run scripts/setup.sh on a fresh host to
#                   provision it); the assets are located at runtime via USI options
#                   (Eval_Dir, Book_File) when the engine starts.
#
# Usage:
#   scripts/build.sh
#
# Exit status is non-zero with an actionable message if the toolchain is missing or the
# build fails.

set -eu

# Resolve the repository root from this script's location, so the script works regardless
# of the directory it is invoked from.
script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH='' cd -- "${script_dir}/.." && pwd)
cd "${repo_root}"

binary="target/release/yorkie"

die() {
    echo "error: $*" >&2
    exit 1
}

# --- Toolchain check -------------------------------------------------------------------
command -v cargo >/dev/null 2>&1 || die "cargo not found. Install a Rust stable toolchain \
(e.g. via rustup: https://rustup.rs) and re-run. The repo pins the channel via rust-toolchain."

# --- Build -----------------------------------------------------------------------------
# nnue is one of three mutually-exclusive eval features, so default features are disabled.
echo "==> Building release (nnue, target-cpu=native) ..."
if ! RUSTFLAGS="-C target-cpu=native" \
        cargo build --release --no-default-features --features nnue; then
    die "build failed. See the cargo output above."
fi
[ -x "${binary}" ] || die "build reported success but ${binary} is missing."

echo "==> Build OK: ${repo_root}/${binary}"
