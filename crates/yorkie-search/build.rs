//! Compile the engine's settings into the search layer.
//!
//! The search reads the settings that shape a node: the draw horizon and the
//! draw contempt it scores a repetition with, whether the generators expose the
//! suppressed non-promotions, which entering-king rule decides a declaration,
//! whether the engine's own mates are scored flat, and how far back a
//! repetition is looked for. Every one of them is fixed when the binary is
//! built, so the node path reads a constant rather than a field a caller filled
//! in.
//!
//! The schema, parser, code generator and path resolution are shared verbatim
//! with the protocol crate's build script, so the two builds cannot read the
//! same file and disagree about what it says. That sharing is an `include!`
//! because a build script cannot depend on a member of the workspace it is
//! building.
//!
//! Three settings reshape the search rather than setting a value in it: with
//! `lazy_smp_spread` on, this script sets the `cfg` of the same name, behind
//! which the workers' move-order spread is compiled, with `lazy_smp_defer`
//! on, the one behind which a worker puts off a move another worker is
//! searching, and with a `repetition_look_back` other than 16, the
//! `repetition_index` behind which each worker indexes the game before its
//! root; off, none of it exists.
//!
//! The generated module carries every schema key it can render on its own, not
//! just the ones read here: the generator renders the schema as a whole, and
//! allowing the unused ones is cheaper than a second, divergent code path. The
//! one it leaves out is the NUMA layout, which is not a value in the file but the
//! machine's own, and which only the crate that plans thread binding compiles
//! in.
//!
//! The script also compiles in the values of the search and time-allocation
//! parameters `src/params.rs` registers. They come from an SPSA settings file:
//! the one the `YORKIE_SPSA_CONFIG` environment variable names, or, when it is
//! unset, the one the engine config's `spsa_config` key names (the play config
//! names `configs/spsa/yorkie.toml`, the test configs
//! `configs/spsa/yaneuraou.toml`). A relative path is taken against the
//! repository root. The file is checked against the registry in `params.rs` —
//! every parameter given once, nothing else, each value an integer inside the
//! parameter's range — and the build stops on the first thing wrong, naming
//! the file and the parameter.

include!("../yorkie-protocol/build_config.rs");
include!("build_spsa.rs");

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
        // The keys this crate reads are ungated; the crate that declares the
        // gating features is where a config a build cannot honour is refused or
        // reported.
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
    emit_switch_cfgs(&generated.cfgs);
    emit_repetition_index_cfg(&generated.cfgs);

    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR is set by cargo"))
        .join("engine_config.rs");
    if let Err(e) = std::fs::write(&out, generated.code) {
        fail(&format!(
            "cannot write the generated config `{}`: {e}",
            out.display()
        ));
    }

    compile_spsa_values(&repo_root, &contents, &label);
}

/// Write the values of the registered search parameters, read from the SPSA
/// settings file this build selects, to `$OUT_DIR/spsa_values.rs`.
fn compile_spsa_values(repo_root: &Path, config_contents: &str, config_label: &str) {
    println!("cargo:rerun-if-changed=build_spsa.rs");
    println!("cargo:rerun-if-changed=src/params.rs");
    println!("cargo:rerun-if-env-changed={SPSA_CONFIG_ENV}");

    // The engine config compiled cleanly above, so the key is there and is a
    // string.
    let from_config = match parse_config(config_contents, config_label)
        .ok()
        .and_then(|mut entries| entries.remove("spsa_config"))
    {
        Some(Entry {
            value: Value::Str(s),
            ..
        }) => s,
        _ => fail(&format!(
            "{config_label}: the `spsa_config` key does not name an SPSA settings file"
        )),
    };
    let path = spsa_config_path(repo_root, std::env::var_os(SPSA_CONFIG_ENV), &from_config);
    println!("cargo:rerun-if-changed={}", path.display());

    let contents = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(e) => fail(&format!(
            "cannot read the SPSA settings file `{}`: {e}\n       \
             set {SPSA_CONFIG_ENV}, or the engine config's `spsa_config` key, to a readable \
             SPSA settings file",
            path.display()
        )),
    };
    let params_rs = match std::fs::read_to_string("src/params.rs") {
        Ok(c) => c,
        Err(e) => fail(&format!("cannot read `src/params.rs`: {e}")),
    };
    let label = path.display().to_string();
    let code = match compile_spsa(
        &contents,
        &label,
        &display_source(repo_root, &path),
        &params_rs,
    ) {
        Ok(code) => code,
        Err(e) => fail(&e),
    };

    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR is set by cargo"))
        .join("spsa_values.rs");
    if let Err(e) = std::fs::write(&out, code) {
        fail(&format!(
            "cannot write the generated parameter values `{}`: {e}",
            out.display()
        ));
    }
}

/// Set the `cfg` of every [`CFG_SWITCHES`] entry the config turns on, or whose
/// same-named Cargo feature (`lazy_smp_spread` ↔ `lazy-smp-spread`) this build
/// carries. The feature compiles a switch's code in without the config turning
/// it on, which is how the tests reach code no test config enables.
fn emit_switch_cfgs(on: &[&str]) {
    for (_, cfg) in CFG_SWITCHES {
        println!("cargo::rustc-check-cfg=cfg({cfg})");
        let feature = format!("CARGO_FEATURE_{}", cfg.to_ascii_uppercase());
        if on.contains(cfg) || std::env::var_os(feature).is_some() {
            println!("cargo::rustc-cfg={cfg}");
        }
    }
}

/// Set the `cfg` behind which the search carries the index of the game before
/// its root when the config's `repetition_look_back` is not the plain
/// sixteen-ply walk's, or when the build carries the `repetition-index`
/// feature, which compiles the same code in under any config.
fn emit_repetition_index_cfg(on: &[&str]) {
    println!("cargo::rustc-check-cfg=cfg({REPETITION_INDEX_CFG})");
    if on.contains(&REPETITION_INDEX_CFG)
        || std::env::var_os("CARGO_FEATURE_REPETITION_INDEX").is_some()
    {
        println!("cargo::rustc-cfg={REPETITION_INDEX_CFG}");
    }
}

/// The repository root: two levels above this crate (`crates/yorkie-search`).
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
