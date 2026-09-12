//! Compile the engine's settings into the evaluation layer, and convert the
//! network into the layout the kernels read.
//!
//! The evaluation layer needs exactly one setting, `fv_scale`, which is read
//! from the same TOML config every other setting comes from and emitted as a
//! `pub const` into `$OUT_DIR/engine_config.rs`.
//!
//! The schema, parser, code generator and path resolution are shared verbatim
//! with the protocol crate's build script, so the two builds cannot read the
//! same file and disagree about what it says. That sharing is an `include!`
//! because a build script cannot depend on a member of the workspace it is
//! building.
//!
//! The generated module carries every schema key it can render on its own, not
//! just `fv_scale`: the generator renders the schema as a whole, and allowing
//! the unused ones is cheaper than a second, divergent code path here. The two
//! it leaves out are the NUMA layout and the worker → CPU assignment, which are
//! not values in the file but the machine's own. The assignment in particular
//! takes CPUs from a ledger shared across builds, which exactly one build script
//! per binary may do.
//!
//! The other half of the work here is the network. `original_eval_dir/nn.bin`
//! is decoded, scaled and laid out exactly as the kernels read it, once, and
//! written to `eval_dir` as a kernel-layout file — so starting the engine costs
//! an open and a mapping rather than a parse, a permutation and a heap
//! allocation of a few hundred mebibytes. The conversion is skipped when the
//! file already there was made from the same source by the same layout and the
//! same target features. A checkout with no network staged still builds: the
//! generated constants say there was no source, and the engine reports the
//! missing network when it is asked to be ready.

include!("../yorkie-protocol/build_config.rs");

#[path = "nnue_layout.rs"]
#[allow(dead_code)]
mod nnue_layout;
#[path = "nnue_source.rs"]
#[allow(dead_code)]
mod nnue_source;

use nnue_layout::{Header, NetDims, Source, resolve_dir};

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
    println!("cargo:rerun-if-changed=nnue_layout.rs");
    println!("cargo:rerun-if-changed=nnue_source.rs");
    println!("cargo:rerun-if-env-changed={CONFIG_ENV}");
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");

    println!("cargo::rustc-check-cfg=cfg(build_targets_host_cpu)");
    if targets_host_cpu() {
        println!("cargo::rustc-cfg=build_targets_host_cpu");
    }

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
        // This crate declares none of the gating features and reads only
        // `fv_scale`, which no feature gates; the crate that declares them is
        // where a config a build cannot honour is refused or reported.
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

    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR is set by cargo"));
    let out = out_dir.join("engine_config.rs");
    if let Err(e) = std::fs::write(&out, generated.code) {
        fail(&format!(
            "cannot write the generated config `{}`: {e}",
            out.display()
        ));
    }

    let entries = match parse_config(&contents, &label) {
        Ok(entries) => entries,
        Err(e) => fail(&e),
    };
    let network = prepare_network(&repo_root, &out_dir, &entries);
    let network_out = out_dir.join("eval_network.rs");
    if let Err(e) = std::fs::write(&network_out, network) {
        fail(&format!(
            "cannot write the generated network constants `{}`: {e}",
            network_out.display()
        ));
    }
}

/// Convert the network into the kernel-layout file the engine reads, and render
/// the constants describing what was written.
fn prepare_network(repo_root: &Path, out_dir: &Path, entries: &BTreeMap<String, Entry>) -> String {
    let source_dir = resolve_dir(repo_root, &text_key(entries, "original_eval_dir"));
    let source_path = source_dir.join(SOURCE_FILE_NAME);
    println!("cargo:rerun-if-changed={}", source_path.display());

    let eval_dir = resolve_dir(&exe_dir(out_dir), &text_key(entries, "eval_dir"));
    let target = eval_dir.join(nnue_layout::FILE_NAME);
    // The file this writes is also an input: deleting it is how an operator
    // asks for it to be made again, and without this cargo would see nothing
    // changed and leave the engine with no network to read.
    println!("cargo:rerun-if-changed={}", target.display());
    let features = target_features();

    let source = match std::fs::read(&source_path) {
        Ok(bytes) => Some(bytes),
        // A checkout with no network staged builds, and says so through the
        // constants: every other error naming the file the config points at is
        // the operator's to see.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => fail(&format!(
            "cannot read the network `{}`: {e}",
            source_path.display()
        )),
    };

    let source_id = match &source {
        None => Source::Absent,
        Some(bytes) => Source::Sha256(nnue_source::sha256(bytes)),
    };

    if let Some(bytes) = source {
        if already_written(&target, source_id, &features) {
            // The file on disk was made from this source, by this layout, for
            // these target features: its bytes cannot differ from what a
            // conversion would produce.
        } else {
            write_kernel_layout_file(&target, &bytes, source_id, &features);
        }
    }

    render_constants(source_id, &features)
}

/// The source network's file name inside `original_eval_dir` — the name the
/// reference engine gives it.
const SOURCE_FILE_NAME: &str = "nn.bin";

/// Whether `target` is already the file this build would write.
fn already_written(target: &Path, source: Source, features: &str) -> bool {
    let Ok(mut file) = std::fs::File::open(target) else {
        return false;
    };
    // The header sits at the front of the data offset; reading that much is a
    // few pages, not the region behind it.
    let mut head = vec![0u8; HEADER_READ_BYTES];
    let read = read_head(&mut file, &mut head);
    let Ok(header) = Header::decode(&head[..read]) else {
        return false;
    };
    header.refusal(source, features).is_none()
}

/// How much of a kernel-layout file is read to recover its header: the whole
/// region before the data, so a header can grow without a size to keep in step.
const HEADER_READ_BYTES: usize = nnue_layout::DATA_OFFSET;

fn read_head(file: &mut std::fs::File, buf: &mut [u8]) -> usize {
    use std::io::Read as _;
    let mut filled = 0;
    while filled < buf.len() {
        match file.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(_) => break,
        }
    }
    filled
}

/// Transform `bytes` and publish the result at `target`.
///
/// Written to a unique temporary in the same directory and renamed, so a build
/// interrupted mid-write leaves the previous file rather than half of a new
/// one, and a concurrent build of another profile never observes a partial
/// file.
fn write_kernel_layout_file(target: &Path, bytes: &[u8], source: Source, features: &str) {
    let converted = match nnue_source::convert(bytes, &NetDims::STANDARD) {
        Ok(c) => c,
        Err(e) => fail(&format!("cannot read the network: {e}")),
    };
    let header = Header {
        layout_version: nnue_layout::LAYOUT_VERSION,
        source,
        target_features: features.to_string(),
        dims: NetDims::STANDARD,
        data_bytes: converted.data.len() as u64,
        net: converted.net,
        warnings: converted.warnings,
    };
    let encoded = header.encode();
    if encoded.len() > nnue_layout::DATA_OFFSET {
        fail("the kernel-layout header does not fit before the data region");
    }

    let dir = target.parent().unwrap_or(Path::new("."));
    if let Err(e) = std::fs::create_dir_all(dir) {
        fail(&format!(
            "cannot create the evaluation directory `{}`: {e}",
            dir.display()
        ));
    }
    let temporary = dir.join(format!(
        "{}.{}.tmp",
        nnue_layout::FILE_NAME,
        std::process::id()
    ));

    let write = || -> std::io::Result<()> {
        use std::io::Write as _;
        let mut file = std::io::BufWriter::new(std::fs::File::create(&temporary)?);
        file.write_all(&encoded)?;
        file.write_all(&vec![0u8; nnue_layout::DATA_OFFSET - encoded.len()])?;
        file.write_all(&converted.data)?;
        file.flush()?;
        Ok(())
    };
    if let Err(e) = write() {
        let _ = std::fs::remove_file(&temporary);
        fail(&format!(
            "cannot write the evaluation file `{}`: {e}",
            temporary.display()
        ));
    }
    if let Err(e) = std::fs::rename(&temporary, target) {
        let _ = std::fs::remove_file(&temporary);
        fail(&format!(
            "cannot publish the evaluation file `{}`: {e}",
            target.display()
        ));
    }
}

/// The generated constants: what the engine holds the file it opens against.
fn render_constants(source: Source, features: &str) -> String {
    let mut out = String::new();
    out.push_str("// @generated by build.rs. Do not edit.\n\n");
    match source {
        Source::Absent => {
            out.push_str(
                "/// Whether a network was there to convert when this binary was built.\n\
                 pub const SOURCE_PRESENT: bool = false;\n\n\
                 /// The SHA-256 of the network this binary was built from; all zero \
                 when there was none.\n\
                 pub const SOURCE_SHA256: [u8; 32] = [0; 32];\n",
            );
        }
        Source::Sha256(digest) => {
            let bytes: Vec<String> = digest.iter().map(|b| b.to_string()).collect();
            let _ = write!(
                out,
                "/// Whether a network was there to convert when this binary was built.\n\
                 pub const SOURCE_PRESENT: bool = true;\n\n\
                 /// The SHA-256 of the network this binary was built from.\n\
                 pub const SOURCE_SHA256: [u8; 32] = [{}];\n",
                bytes.join(", ")
            );
        }
    }
    let _ = write!(
        out,
        "\n/// The `target_feature` set this binary was compiled for. The kernels are \
         selected\n/// from it, so a file laid out for another set is not one they can read.\n\
         pub const TARGET_FEATURES: &str = \"{}\";\n\n\
         /// The number of NUMA nodes the machine this binary was built on has — the \
         number\n/// of memory regions the network can need at most, one per node whose \
         workers\n/// read it.\n\
         pub const MACHINE_NODES: usize = {};\n",
        escape(features),
        machine_nodes(),
    );
    out
}

/// The `target_feature` set this build compiles for, as cargo reports it.
fn target_features() -> String {
    std::env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default()
}

/// The value of a string-typed schema key. The config was validated above, so a
/// key that is missing or is not a string cannot reach here.
fn text_key(entries: &BTreeMap<String, Entry>, key: &str) -> String {
    match entries.get(key).map(|e| &e.value) {
        Some(Value::Str(s)) => s.clone(),
        _ => fail(&format!("`{key}` must be a string")),
    }
}

/// The directory the engine binary this build produces will sit in, which is
/// where its relative `eval_dir` resolves: three levels above `OUT_DIR`
/// (`<target>/[<triple>/]<profile>/build/<pkg>-<hash>/out`).
fn exe_dir(out_dir: &Path) -> PathBuf {
    out_dir
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| out_dir.to_path_buf())
}

/// How many NUMA nodes the building machine has.
///
/// What it decides is whether the network can be one mapping every process on
/// the machine shares, or has to be a copy per node the workers read it from. A
/// host whose sysfs reports no topology is one node, which is the answer that
/// makes the engine share one mapping.
fn machine_nodes() -> usize {
    let opts = match yorkie_numa::machine_sysfs_options(Path::new("/sys")) {
        Ok(opts) => opts,
        Err(e) => {
            let mut reason = Vec::new();
            e.write_message(|fragment| reason.extend_from_slice(fragment));
            let reason = String::from_utf8(reason).expect("a refusal is ASCII");
            fail(&format!(
                "cannot read this host's NUMA topology: {reason}\n       \
                 the engine compiles the layout of the machine it is built on into the \
                 binary, so building it needs a Linux host whose sysfs reports one"
            ))
        }
    };
    yorkie_numa::NumaLayout::of_machine(&opts)
        .num_nodes()
        .max(1)
}

/// Whether this build compiles for the CPU of the machine doing the building,
/// which is what `-C target-cpu=native` asks for and what the default build of
/// this crate gets.
///
/// The kernel backend is selected from the target features the build enables,
/// so only such a build can be held against the building host's own CPU
/// features; one that pins a portable target CPU compiles the scalar arm even
/// where the host could have run the SIMD one, and means to. rustc takes the
/// last `target-cpu` it is given, so this does too.
fn targets_host_cpu() -> bool {
    let Ok(flags) = std::env::var("CARGO_ENCODED_RUSTFLAGS") else {
        return false;
    };
    flags
        .split('\u{1f}')
        .filter_map(|flag| {
            flag.strip_prefix("-C")
                .unwrap_or(flag)
                .strip_prefix("target-cpu=")
        })
        .next_back()
        .is_some_and(|cpu| cpu == "native")
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
