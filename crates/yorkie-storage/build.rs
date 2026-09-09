//! Compile the engine's settings into the storage layer.
//!
//! The storage layer needs exactly one setting, `usi_hash`, which is the size of
//! the transposition table. The table is a `static`, so its cluster count is an
//! array length — a value that has to exist before the crate is compiled, not
//! one a caller can hand over.
//!
//! The schema, parser, code generator and path resolution are shared verbatim
//! with the protocol crate's build script, so the two builds cannot read the
//! same file and disagree about what it says. That sharing is an `include!`
//! because a build script cannot depend on a member of the workspace it is
//! building.
//!
//! The generated module carries every schema key it can render on its own, not
//! just `usi_hash`: the generator renders the schema as a whole, and allowing
//! the unused ones is cheaper than a second, divergent code path here. The one
//! it leaves out is the NUMA layout, which is not a value in the file but the
//! machine's own, and which only the crate that plans thread binding compiles
//! in.

include!("../yorkie-protocol/build_config.rs");

/// Report a build-stopping configuration error and exit. `process::exit` rather
/// than `panic!`, so cargo surfaces the message without a backtrace header
/// wrapped around it.
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
        // This crate reads only `usi_hash`, which no feature gates; the crate
        // that declares the gating features is where a config a build cannot
        // honour is refused or reported.
        &Gating::Absent,
        // Nothing here places memory per node, so this crate compiles in no NUMA
        // layout and reads none.
        &Layout::Absent,
        // Exactly one build script per binary may take CPUs from the ledger, and
        // it is not this one.
        &Assignment::Absent,
    ) {
        Ok(g) => g,
        Err(e) => fail(&e),
    };

    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR is set by cargo"))
        .join("engine_config.rs");
    if let Err(e) = std::fs::write(&out, generated.code) {
        fail(&format!(
            "cannot write the generated config `{}`: {e}",
            out.display()
        ));
    }
}

/// The repository root: two levels above this crate (`crates/yorkie-storage`).
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
