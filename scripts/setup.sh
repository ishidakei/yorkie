#!/usr/bin/env sh
#
# setup.sh — provision a fresh Ubuntu 24.04 (x64) host, then build Yorkie.
#
# A bare Ubuntu has neither a C toolchain (needed to link a Rust binary) nor Rust itself.
# This script installs those prerequisites and then delegates to build.sh to produce the
# optimized NNUE release binary. It is the one-command path from a fresh `git clone`.
#
# It is idempotent: if Rust (cargo) is already available, the rustup install is skipped.
# Any arguments are forwarded to build.sh.
#
# Note: this step MODIFIES THE SYSTEM. It installs system packages via apt (using sudo
# when not run as root) and runs the official rustup installer. On a host that already has
# Rust set up, run scripts/build.sh directly instead.
#
# Usage:
#   scripts/setup.sh

set -eu

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)

die() {
    echo "error: $*" >&2
    exit 1
}

# Run a command with root privileges, using sudo only when the current user is not root.
run_root() {
    if [ "$(id -u)" -eq 0 ]; then
        "$@"
    else
        sudo "$@"
    fi
}

# --- Preconditions ---------------------------------------------------------------------
if [ "$(id -u)" -ne 0 ] && ! command -v sudo >/dev/null 2>&1; then
    die "root privileges are required to install system packages (build-essential, curl), \
but the current user is not root and sudo is unavailable. Install those packages and the \
Rust toolchain manually, then run scripts/build.sh directly."
fi

# --- C toolchain + curl ----------------------------------------------------------------
# build-essential provides the linker Rust needs; curl fetches the rustup installer.
echo "==> Installing system packages (build-essential, curl) ..."
run_root apt-get update
run_root apt-get install -y --no-install-recommends build-essential curl

# --- Rust (rustup + stable toolchain) --------------------------------------------------
if command -v cargo >/dev/null 2>&1; then
    echo "==> Rust already present; skipping rustup install."
else
    echo "==> Installing Rust via rustup ..."
    curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs | sh -s -- -y
    # Put cargo on PATH for this shell (and the build.sh child) without a fresh login.
    # rustup honours the repo's rust-toolchain pin (stable) on first cargo use.
    # shellcheck disable=SC1091
    . "${HOME}/.cargo/env"
fi

# --- Build -----------------------------------------------------------------------------
exec "${script_dir}/build.sh" "$@"
