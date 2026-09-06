// The engine config's schema, parser and code generator — the pure half of
// `build.rs`.
//
// Nothing here touches the environment, the filesystem, or the process; every
// failure is a returned `Err(String)`, which is what lets
// `tests/config_schema.rs` assert on the fail-loud behaviour instead of only
// observing it by breaking a build on purpose.
//
// It is `include!`d rather than shared as a crate because a build script cannot
// depend on a member of the workspace it is building.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// The environment variable naming the config file to compile in.
const CONFIG_ENV: &str = "YORKIE_CONFIG";

/// The config compiled in when [`CONFIG_ENV`] is unset, relative to the
/// repository root.
const DEFAULT_CONFIG: &str = "configs/default.toml";

/// `Threads`' real ceiling is machine-dependent (`max(1024, 4 · cores)`), so it
/// cannot be checked exactly at build time. This is a sanity bound: it catches a
/// slipped digit while leaving room for any machine the engine could plausibly
/// run on. A value the target machine cannot actually honour is the operator's
/// to get right.
const MAX_THREADS_SANITY: i64 = 4096;

/// The `EnteringKingRule` choice list (`EKR_STRINGS`, in the reference's order).
const ENTERING_KING_RULE_CHOICES: &[&str] = &[
    "NoEnteringKing",
    "CSARule24",
    "CSARule24H",
    "CSARule27",
    "CSARule27H",
    "TryRule",
];

/// The `BookFile` choice list — the same set the runtime combo advertises.
const BOOK_FILE_CHOICES: &[&str] = &[
    "no_book",
    "standard_book.ybb",
    "yaneura_book1.ybb",
    "yaneura_book2.ybb",
    "yaneura_book3.ybb",
    "yaneura_book4.ybb",
    "user_book1.ybb",
    "user_book2.ybb",
    "user_book3.ybb",
    "book.ybb",
];

/// What a schema key accepts, and what it generates.
enum Kind {
    /// `i64`, inside an inclusive range. The bounds mirror the corresponding
    /// USI spin option's `min` / `max`.
    Int { min: i64, max: i64 },
    /// `bool`.
    Bool,
    /// `&'static str`, any content.
    Text,
    /// `&'static str`, restricted to a fixed list — the corresponding USI combo
    /// option's choices.
    Choice(&'static [&'static str]),
}

/// Every Cargo feature the schema conditions a constant on: the verbosity
/// features, each implying the one before it, then the features orthogonal to
/// that axis. Which of them are on is the caller's to determine — cargo tells a
/// build script through its `CARGO_FEATURE_*` variables.
///
/// Each including build script uses the half of the gating machinery its crate
/// needs — one declares these features, the other declares none — so an item
/// only the other one constructs is not dead code.
#[allow(dead_code)]
const GATE_FEATURES: &[&str] = &["verbose1", "verbose2", "verbose3", "random"];

/// How a build script treats the feature-gated keys (see [`Gate`]).
#[allow(dead_code)]
enum Gating<'a> {
    /// The crate declares the gating features: those in the list are on, the
    /// gated key's constant is generated behind its feature's `cfg`, and a value
    /// a build without that feature cannot honour meets the gate's own rule.
    Features(&'a [&'a str]),
    /// The crate declares none of the gating features, so it has no `cfg` to
    /// name and no build shape to judge a value against. Every key is rendered
    /// plainly; such a crate reads only ungated ones, and the crate that owns
    /// the features judges the config the whole build shares.
    Absent,
}

/// A setting whose constant is conditioned on a Cargo feature: the generated
/// constant carries that feature's `cfg`, so a build without the feature does
/// not merely leave the setting unread — it does not have it. The variants
/// differ in what such a build may put in the config file.
enum Gate {
    /// A setting the engine implements only with a verbosity feature. A build
    /// without it accepts only the one value its code is fixed to, so a setting
    /// that could not take effect is refused rather than silently ignored.
    Verbosity {
        /// The Cargo feature the key needs.
        feature: &'static str,
        /// The only value a build without `feature` accepts.
        fixed: i64,
        /// Why the lower build is fixed to that value, as the second sentence of
        /// the error message.
        because: &'static str,
    },
    /// A setting an off-by-default feature owns, orthogonal to the verbosity
    /// axis. One config file has to build every shape, so a build without the
    /// feature takes any value the range allows; anything but the inert one is
    /// reported as a build warning, since nothing in that build reads it.
    Feature {
        /// The Cargo feature the key needs.
        feature: &'static str,
        /// The value that makes the setting a no-op, and so the one value a
        /// build without the feature ignores without losing anything.
        inert: i64,
    },
}

impl Gate {
    /// The Cargo feature the generated constant's `cfg` names.
    fn feature(&self) -> &'static str {
        match self {
            Gate::Verbosity { feature, .. } | Gate::Feature { feature, .. } => feature,
        }
    }
}

/// One schema entry. `key` is the TOML key; the generated constant's name is
/// that key upper-cased. `usi` names the runtime option the constant replaces,
/// and is carried into the generated doc comment so the mapping is readable
/// from either side.
struct Spec {
    key: &'static str,
    usi: &'static str,
    kind: Kind,
    gate: Option<Gate>,
}

const fn int(key: &'static str, usi: &'static str, min: i64, max: i64) -> Spec {
    Spec {
        key,
        usi,
        kind: Kind::Int { min, max },
        gate: None,
    }
}

/// An `int` key that only takes effect with `feature` (see [`Gate::Verbosity`]).
const fn gated_int(
    key: &'static str,
    usi: &'static str,
    min: i64,
    max: i64,
    feature: &'static str,
    fixed: i64,
    because: &'static str,
) -> Spec {
    Spec {
        key,
        usi,
        kind: Kind::Int { min, max },
        gate: Some(Gate::Verbosity {
            feature,
            fixed,
            because,
        }),
    }
}

/// An `int` key an orthogonal feature owns (see [`Gate::Feature`]).
const fn feature_int(
    key: &'static str,
    usi: &'static str,
    min: i64,
    max: i64,
    feature: &'static str,
    inert: i64,
) -> Spec {
    Spec {
        key,
        usi,
        kind: Kind::Int { min, max },
        gate: Some(Gate::Feature { feature, inert }),
    }
}

const fn boolean(key: &'static str, usi: &'static str) -> Spec {
    Spec {
        key,
        usi,
        kind: Kind::Bool,
        gate: None,
    }
}

const fn text(key: &'static str, usi: &'static str) -> Spec {
    Spec {
        key,
        usi,
        kind: Kind::Text,
        gate: None,
    }
}

const fn choice(key: &'static str, usi: &'static str, choices: &'static [&'static str]) -> Spec {
    Spec {
        key,
        usi,
        kind: Kind::Choice(choices),
        gate: None,
    }
}

/// The exact key set a config file must carry — no more, no less.
///
/// Ranges are the runtime declarations' own `min` / `max`, so a value this
/// schema accepts is a value a `setoption`-driven engine would have accepted,
/// and the compiled-in setting cannot drift outside what the option allows.
const SCHEMA: &[Spec] = &[
    // --- Engine core.
    int("usi_hash", "USI_Hash", 1, 33_554_432),
    int("threads", "Threads", 1, MAX_THREADS_SANITY),
    gated_int(
        "multi_pv",
        "MultiPV",
        1,
        600,
        "verbose2",
        1,
        "a build that prints no search `info` line has no way to report a second principal \
         variation, so its root search is single-line",
    ),
    text("eval_dir", "EvalDir"),
    int("fv_scale", "FV_SCALE", 1, 128),
    text("numa_policy", "NumaPolicy"),
    boolean("usi_ponder", "USI_Ponder"),
    boolean("stochastic_ponder", "Stochastic_Ponder"),
    // --- Opening book. `book_options_v2` is the one key with no USI option
    // behind it: it stands in for the reference's pre-handshake profile file,
    // which no build of this engine reads.
    boolean("book_options_v2", "(none: the book-option profile selector)"),
    boolean("usi_own_book", "USI_OwnBook"),
    boolean("narrow_book", "NarrowBook"),
    int("book_moves", "BookMoves", 0, 10_000),
    int("book_ignore_rate", "BookIgnoreRate", 0, 100),
    choice("book_file", "BookFile", BOOK_FILE_CHOICES),
    text("book_dir", "BookDir"),
    int("book_eval_diff", "BookEvalDiff", 0, 99_999),
    int("book_eval_black_diff", "BookEvalBlackDiff", 0, 99_999),
    int("book_eval_white_diff", "BookEvalWhiteDiff", 0, 99_999),
    int(
        "book_eval_black_limit",
        "BookEvalBlackLimit",
        -99_999,
        99_999,
    ),
    int(
        "book_eval_white_limit",
        "BookEvalWhiteLimit",
        -99_999,
        99_999,
    ),
    int("book_depth_limit", "BookDepthLimit", 0, 99_999),
    int("book_depth_black_limit", "BookDepthBlackLimit", 0, 99_999),
    int("book_depth_white_limit", "BookDepthWhiteLimit", 0, 99_999),
    boolean("book_on_the_fly", "BookOnTheFly"),
    boolean("consider_book_move_count", "ConsiderBookMoveCount"),
    int("book_pv_moves", "BookPvMoves", 1, 246),
    boolean("ignore_book_ply", "IgnoreBookPly"),
    boolean("flipped_book", "FlippedBook"),
    // --- Search behaviour.
    choice(
        "entering_king_rule",
        "EnteringKingRule",
        ENTERING_KING_RULE_CHOICES,
    ),
    int("depth_limit", "DepthLimit", 0, 2_147_483_647),
    int("nodes_limit", "NodesLimit", 0, i64::MAX),
    int("max_moves_to_draw", "MaxMovesToDraw", 0, 100_000),
    int("pv_interval", "PvInterval", 0, 100_000_000),
    boolean("consideration_mode", "ConsiderationMode"),
    boolean("output_fail_lh_pv", "OutputFailLHPV"),
    int("draw_value_black", "DrawValueBlack", -30_000, 30_000),
    int("draw_value_white", "DrawValueWhite", -30_000, 30_000),
    int("resign_value", "ResignValue", 0, 99_999),
    boolean("generate_all_legal_moves", "GenerateAllLegalMoves"),
    feature_int(
        "random",
        "(none: the static-evaluation noise amplitude)",
        0,
        100,
        "random",
        0,
    ),
    // --- Time management.
    int("network_delay", "NetworkDelay", 0, 10_000),
    int("network_delay2", "NetworkDelay2", 0, 10_000),
    int("minimum_thinking_time", "MinimumThinkingTime", 1, 100_000),
    int("slow_mover", "SlowMover", 1, 1_000),
    boolean("round_up_to_full_second", "RoundUpToFullSecond"),
];

/// A parsed scalar, with the source line it came from for diagnostics.
struct Entry {
    value: Value,
    line: usize,
}

enum Value {
    Int(i64),
    Bool(bool),
    Str(String),
}

impl Value {
    fn type_name(&self) -> &'static str {
        match self {
            Value::Int(_) => "integer",
            Value::Bool(_) => "boolean",
            Value::Str(_) => "string",
        }
    }
}

/// Parse the accepted TOML subset into `key -> entry`: flat `key = value` pairs
/// whose values are integers, booleans or basic strings, plus `#` comments and
/// blank lines.
///
/// Anything else is an error rather than something quietly skipped, because a
/// line the engine config does not understand is a setting the operator believes
/// they made.
///
/// `label` is the file name used in messages.
fn parse_config(contents: &str, label: &str) -> Result<BTreeMap<String, Entry>, String> {
    let mut entries: BTreeMap<String, Entry> = BTreeMap::new();

    for (idx, raw) in contents.lines().enumerate() {
        let line_no = idx + 1;
        let line = strip_comment(raw).trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') {
            return Err(at(
                label,
                line_no,
                "table headers are not supported; the engine config is a flat key = value list",
            ));
        }
        let Some((key, value)) = line.split_once('=') else {
            return Err(at(label, line_no, "expected `key = value`"));
        };
        let key = key.trim();
        if key.is_empty() || !key.bytes().all(is_key_byte) {
            return Err(at(
                label,
                line_no,
                &format!("`{key}` is not a valid key (lowercase letters, digits and `_` only)"),
            ));
        }
        let value = parse_value(value.trim(), label, line_no)?;
        if let Some(prev) = entries.insert(
            key.to_string(),
            Entry {
                value,
                line: line_no,
            },
        ) {
            return Err(at(
                label,
                line_no,
                &format!("key `{key}` is set twice (first at line {})", prev.line),
            ));
        }
    }

    Ok(entries)
}

fn is_key_byte(b: u8) -> bool {
    b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'
}

/// Drop a trailing `#` comment, leaving `#` inside a basic string alone.
fn strip_comment(line: &str) -> &str {
    let mut in_string = false;
    for (i, &b) in line.as_bytes().iter().enumerate() {
        match b {
            b'"' => in_string = !in_string,
            b'#' if !in_string => return &line[..i],
            _ => {}
        }
    }
    line
}

fn parse_value(raw: &str, label: &str, line_no: usize) -> Result<Value, String> {
    if raw.is_empty() {
        return Err(at(label, line_no, "missing value after `=`"));
    }
    if let Some(rest) = raw.strip_prefix('"') {
        let Some(body) = rest.strip_suffix('"') else {
            return Err(at(label, line_no, "unterminated string value"));
        };
        if body.contains('"') {
            return Err(at(label, line_no, "a string value may not contain a quote"));
        }
        if body.contains('\\') {
            return Err(at(
                label,
                line_no,
                "escape sequences are not supported in a string value",
            ));
        }
        return Ok(Value::Str(body.to_string()));
    }
    match raw {
        "true" => return Ok(Value::Bool(true)),
        "false" => return Ok(Value::Bool(false)),
        _ => {}
    }
    let digits: String = raw.chars().filter(|c| *c != '_').collect();
    match digits.parse::<i64>() {
        Ok(v) => Ok(Value::Int(v)),
        Err(_) => Err(at(
            label,
            line_no,
            &format!("`{raw}` is not an integer, a boolean, or a quoted string"),
        )),
    }
}

fn at(label: &str, line: usize, msg: &str) -> String {
    format!("{label}:{line}: {msg}")
}

/// The pure pipeline's output.
#[derive(Debug)]
struct Generated {
    /// The generated Rust module.
    code: String,
    /// Settings this build compiles but cannot honour, for the including build
    /// script to surface. Only the build script whose crate declares the gating
    /// features can produce, or has to report, any.
    #[allow(dead_code)]
    warnings: Vec<String>,
}

/// Type-check and range-check every schema key against the parsed file, reject
/// anything left over, and render the generated module.
///
/// `label` names the file in messages, `source` is the path recorded in the
/// generated header, and `name` is the config's identity (its file stem).
/// `gating` says how this build treats the feature-gated keys.
fn generate(
    entries: &BTreeMap<String, Entry>,
    label: &str,
    source: &str,
    name: &str,
    gating: &Gating,
) -> Result<Generated, String> {
    // Unknown keys first: a typo'd key would otherwise be reported as the schema
    // key it was meant to be, going missing.
    let unknown: Vec<&str> = entries
        .keys()
        .map(String::as_str)
        .filter(|k| !SCHEMA.iter().any(|s| s.key == *k))
        .collect();
    if !unknown.is_empty() {
        return Err(format!(
            "{label}: unknown key(s): {}\n       \
             the engine config schema is exact — remove them, or add them to SCHEMA in \
             this crate's build_config.rs",
            unknown.join(", ")
        ));
    }

    let mut warnings = Vec::new();
    let mut out = String::new();
    let _ = write!(
        out,
        "// @generated from `{source}`. Do not edit: edit the config and rebuild.\n\n\
         /// The file-stem of the config instance compiled into this binary.\n\
         pub const CONFIG_NAME: &str = \"{}\";\n\n",
        escape(name)
    );

    for spec in SCHEMA {
        let Some(entry) = entries.get(spec.key) else {
            return Err(format!("{label}: required key `{}` is missing", spec.key));
        };
        let const_name = spec.key.to_ascii_uppercase();
        if let Gating::Features(active) = gating
            && let Some(gate) = &spec.gate
            && let Value::Int(v) = &entry.value
            && !active.contains(&gate.feature())
        {
            match gate {
                Gate::Verbosity {
                    feature,
                    fixed,
                    because,
                } if *v != *fixed => {
                    return Err(at(
                        label,
                        entry.line,
                        &format!(
                            "`{}` = {v} needs a `{feature}` build: {because}. This build does not \
                             have that feature, so the only value it accepts is {fixed} — set the \
                             key to that, or build with `--features {feature}`",
                            spec.key
                        ),
                    ));
                }
                Gate::Feature { feature, inert } if *v != *inert => {
                    warnings.push(at(
                        label,
                        entry.line,
                        &format!(
                            "`{}` = {v} has no effect: the `{feature}` feature is off in this \
                             build, so nothing reads the setting — build with \
                             `--features {feature}`, or set the key to {inert}",
                            spec.key
                        ),
                    ));
                }
                _ => {}
            }
        }
        let rendered = match (&spec.kind, &entry.value) {
            (Kind::Int { min, max }, Value::Int(v)) => {
                if v < min || v > max {
                    return Err(at(
                        label,
                        entry.line,
                        &format!("`{}` = {v} is outside [{min}, {max}]", spec.key),
                    ));
                }
                format!("pub const {const_name}: i64 = {v};")
            }
            (Kind::Bool, Value::Bool(v)) => format!("pub const {const_name}: bool = {v};"),
            (Kind::Text, Value::Str(v)) => {
                format!("pub const {const_name}: &str = \"{}\";", escape(v))
            }
            (Kind::Choice(choices), Value::Str(v)) => {
                if !choices.contains(&v.as_str()) {
                    return Err(at(
                        label,
                        entry.line,
                        &format!(
                            "`{}` = \"{v}\" is not one of: {}",
                            spec.key,
                            choices.join(", ")
                        ),
                    ));
                }
                format!("pub const {const_name}: &str = \"{}\";", escape(v))
            }
            (kind, value) => {
                return Err(at(
                    label,
                    entry.line,
                    &format!(
                        "`{}` must be {}, got {}",
                        spec.key,
                        expected(kind),
                        value.type_name()
                    ),
                ));
            }
        };
        let _ = writeln!(out, "/// USI option `{}`.", spec.usi);
        if let Gating::Features(_) = gating
            && let Some(gate) = &spec.gate
        {
            let _ = writeln!(out, "#[cfg(feature = \"{}\")]", gate.feature());
        }
        let _ = writeln!(out, "{rendered}");
    }

    Ok(Generated {
        code: out,
        warnings,
    })
}

fn expected(kind: &Kind) -> &'static str {
    match kind {
        Kind::Int { .. } => "an integer",
        Kind::Bool => "a boolean",
        Kind::Text | Kind::Choice(_) => "a string",
    }
}

fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Parse and render in one step — the whole pure pipeline, for a caller that has
/// already read the bytes.
fn compile_config(
    contents: &str,
    label: &str,
    source: &str,
    name: &str,
    gating: &Gating,
) -> Result<Generated, String> {
    let entries = parse_config(contents, label)?;
    generate(&entries, label, source, name, gating)
}

// The config-path resolution is kept here, and kept pure — the caller reads the
// environment — so it is covered by the same tests as the schema. Getting it
// wrong is the quiet failure mode the whole mechanism has to avoid: a build that
// silently reads a config the operator did not mean to select.

/// Resolve the config file to read from the raw `YORKIE_CONFIG` value. Unset
/// selects the checked-in play config.
///
/// A relative path is taken against the repository root, so
/// `YORKIE_CONFIG=configs/test.toml` means the same thing from any working
/// directory: cargo runs a build script with the package directory as its cwd,
/// so a cwd-relative rule would name a different file depending on which crate
/// triggered the build.
fn config_path(repo_root: &Path, raw: Option<OsString>) -> PathBuf {
    let raw = raw
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_CONFIG));
    if raw.is_absolute() {
        raw
    } else {
        repo_root.join(raw)
    }
}

/// The config's file stem — the identity a build records for itself.
fn config_name(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unnamed".to_string())
}

/// The config path as it appears in the generated header: repository-relative
/// when it lives inside the repository, so the generated file does not bake an
/// absolute build-machine path into an otherwise reproducible artefact.
fn display_source(repo_root: &Path, path: &Path) -> String {
    path.strip_prefix(repo_root)
        .unwrap_or(path)
        .display()
        .to_string()
}
