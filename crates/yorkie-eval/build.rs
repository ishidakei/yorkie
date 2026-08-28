//! Compile the engine's settings into the evaluation layer.
//!
//! The evaluation layer needs exactly one of them — `fv_scale`, the NNUE
//! fixed-point output scale the reference keeps in the mutable global
//! `NNUE::FV_SCALE`. The engine has no runtime configuration, so there is no
//! live value to be told about: the scale is read from the same TOML config
//! every other setting comes from and emitted as a `pub const` into
//! `$OUT_DIR/engine_config.rs`, which `src/config.rs` includes.
//!
//! The schema, the parser, the code generator and the `YORKIE_CONFIG` path
//! resolution are shared verbatim with the protocol crate's build script by
//! `include!`ing its `build_config.rs`, so the two builds cannot read the same
//! file and disagree about what it says. (A build script cannot depend on a
//! member of the workspace it is building, which is why this is an `include!`
//! and not a crate.)
//!
//! The generated module carries every schema key, not just `fv_scale`: the
//! generator renders the schema as a whole, and `src/config.rs` allows the
//! unused ones rather than growing a second, divergent code path here.

include!("../yorkie-protocol/build_config.rs");

/// Report a build-stopping configuration error and exit. `process::exit` rather
/// than `panic!` so the message cargo surfaces is the message, without a
/// backtrace header wrapped around it.
fn fail(msg: &str) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(1);
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../yorkie-protocol/build_config.rs");
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

/// The repository root: two levels above this crate (`crates/yorkie-eval`).
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
