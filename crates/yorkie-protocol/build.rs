//! Compile the engine's settings in.
//!
//! The engine carries no runtime configuration: every setting it has is read
//! from one TOML file at build time and emitted as a `pub const` into
//! `$OUT_DIR/engine_config.rs`, which `src/config.rs` includes as
//! `yorkie_protocol::config`. Every build reads those constants and nothing
//! else, so the optimizer sees literals: `if config::CONSIDERATION_MODE` is
//! folded away rather than branched on, and the design rule "never read a
//! constant through a variable" holds for the whole of a game.
//!
//! Which file is read is chosen by the `YORKIE_CONFIG` environment variable (a
//! path; a relative one resolves against the repository root). Unset, the build
//! reads the checked-in play config `configs/default.toml`.
//!
//! Every failure is a hard build error with a pointed message. There is no
//! fallback value for a missing key, no default for a malformed one, and no
//! tolerance for a key outside the schema: a config that does not say exactly
//! what the engine will do must not produce a binary at all.
//!
//! This file is the impure half — environment, filesystem, exit code. The
//! schema, the parser, the code generator and the config-path resolution live in
//! `build_config.rs`, which is `include!`d here and, separately, by
//! `tests/config_schema.rs`, so the fail-loud behaviour is covered by ordinary
//! tests rather than only by breaking a build on purpose.

include!("build_config.rs");

/// Report a build-stopping configuration error and exit. `process::exit` rather
/// than `panic!` so the message cargo surfaces is the message, without a
/// backtrace header wrapped around it.
fn fail(msg: &str) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(1);
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=build_config.rs");
    println!("cargo:rerun-if-env-changed={CONFIG_ENV}");

    let repo_root = repo_root();
    let path = config_path(&repo_root, std::env::var_os(CONFIG_ENV));
    println!("cargo:rerun-if-changed={}", path.display());

    let contents = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(e) => fail(&format!(
            "cannot read the engine config `{}`: {e}\n       \
             set {CONFIG_ENV} to a readable config file, or leave it unset to use \
             `{DEFAULT_CONFIG}`",
            path.display()
        )),
    };

    let label = path.display().to_string();
    let generated = match compile_config(
        &contents,
        &label,
        &display_source(&repo_root, &path),
        &config_name(&path),
    ) {
        Ok(g) => g,
        Err(e) => fail(&e),
    };

    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR is set by cargo"))
        .join("engine_config.rs");
    if let Err(e) = std::fs::write(&out, generated) {
        fail(&format!(
            "cannot write the generated config `{}`: {e}",
            out.display()
        ));
    }
}

/// The repository root: two levels above this crate (`crates/yorkie-protocol`).
fn repo_root() -> PathBuf {
    let manifest = PathBuf::from(
        std::env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set by cargo"),
    );
    manifest
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or(manifest)
}
