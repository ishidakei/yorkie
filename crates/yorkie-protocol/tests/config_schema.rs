//! The engine config's schema and parser: what it accepts, and how loudly it
//! refuses everything else.
//!
//! `build_config.rs` is `include!`d here exactly as `build.rs` includes it, so
//! these tests exercise the same code that decides whether a build happens. The
//! whole design rests on one promise: a config the engine cannot make sense of
//! never becomes a binary.

#![allow(dead_code)]

include!(concat!(env!("CARGO_MANIFEST_DIR"), "/build_config.rs"));

/// The play config — the real thing, read off disk, so these tests are a live
/// check that `configs/default.toml` still satisfies the schema.
fn default_config_text() -> String {
    config_text("configs/default.toml")
}

fn config_text(rel: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(rel);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{rel} is readable: {e}"))
}

/// Every checked-in config, by repository-relative path. A new one must be added
/// here so the two whole-file tests below cover it.
const CHECKED_IN_CONFIGS: &[&str] = &[
    "configs/default.toml",
    "configs/test.toml",
    "configs/test-limits.toml",
];

/// Compile a config body, returning the generated Rust or the error message.
fn compile(text: &str) -> Result<String, String> {
    compile_config(text, "test.toml", "test.toml", "test")
}

/// The checked-in config with one key's line replaced (or removed, if
/// `replacement` is `None`).
fn default_with(key: &str, replacement: Option<&str>) -> String {
    let text = default_config_text();
    let mut out = String::new();
    let mut hit = false;
    for line in text.lines() {
        let is_target = line
            .split_once('=')
            .is_some_and(|(k, _)| k.trim() == key && !line.trim_start().starts_with('#'));
        if is_target {
            hit = true;
            if let Some(r) = replacement {
                out.push_str(r);
                out.push('\n');
            }
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    assert!(hit, "key `{key}` not found in configs/default.toml");
    out
}

// --- The checked-in configs are valid -------------------------------------

#[cfg_attr(miri, ignore)]
#[test]
fn every_checked_in_config_compiles() {
    for rel in CHECKED_IN_CONFIGS {
        let out = compile(&config_text(rel)).unwrap_or_else(|e| panic!("{rel} must compile: {e}"));
        assert!(out.contains("pub const USI_HASH: i64 ="), "{rel}");
        assert!(
            out.contains("pub const CONFIG_NAME: &str = \"test\";"),
            "{rel}"
        );
    }
}

/// The configs carry different values — the play config is sized for a real
/// machine, and the second test config diverges on purpose — but their *schema*
/// never may, and the day one does this is the test that says so.
#[cfg_attr(miri, ignore)]
#[test]
fn every_checked_in_config_declares_the_same_key_set() {
    let base = parse_config(&default_config_text(), "default").expect("default parses");
    let want: Vec<&str> = base.keys().map(String::as_str).collect();
    for rel in CHECKED_IN_CONFIGS {
        let got = parse_config(&config_text(rel), rel).unwrap_or_else(|e| panic!("{rel}: {e}"));
        let got: Vec<&str> = got.keys().map(String::as_str).collect();
        assert_eq!(got, want, "{rel} declares a different key set");
    }
}

/// Every generated constant is a `pub const` of the type the accessor layer
/// expects — one line per schema key, plus the header and `CONFIG_NAME`.
#[cfg_attr(miri, ignore)]
#[test]
fn generation_covers_the_whole_schema() {
    let out = compile(&default_config_text()).expect("compiles");
    for spec in SCHEMA {
        let name = spec.key.to_ascii_uppercase();
        assert!(
            out.contains(&format!("pub const {name}: ")),
            "constant {name} was not generated"
        );
        assert!(
            out.contains(&format!("/// USI option `{}`.", spec.usi)),
            "constant {name} lost its option-name doc comment"
        );
    }
}

// --- Fail-loud: the schema is exact ---------------------------------------

#[cfg_attr(miri, ignore)]
#[test]
fn a_missing_key_is_an_error() {
    let err = compile(&default_with("slow_mover", None)).expect_err("missing key must fail");
    assert!(
        err.contains("required key `slow_mover` is missing"),
        "unhelpful message: {err}"
    );
}

#[cfg_attr(miri, ignore)]
#[test]
fn an_unknown_key_is_an_error() {
    let text = format!("{}\nnot_a_setting = 1\n", default_config_text());
    let err = compile(&text).expect_err("unknown key must fail");
    assert!(
        err.contains("unknown key(s): not_a_setting"),
        "unhelpful message: {err}"
    );
}

/// A typo'd key is the interesting case: it is reported as the unknown key it
/// is, not as the schema key it displaced going missing.
#[cfg_attr(miri, ignore)]
#[test]
fn a_typoed_key_is_reported_as_unknown() {
    let err =
        compile(&default_with("slow_mover", Some("slowmover = 100"))).expect_err("typo must fail");
    assert!(
        err.contains("unknown key(s): slowmover"),
        "a typo must name itself, not the key it displaced: {err}"
    );
}

#[cfg_attr(miri, ignore)]
#[test]
fn a_duplicate_key_is_an_error() {
    let text = format!("{}\nslow_mover = 200\n", default_config_text());
    let err = compile(&text).expect_err("duplicate key must fail");
    assert!(err.contains("`slow_mover` is set twice"), "message: {err}");
}

// --- Fail-loud: types and ranges ------------------------------------------

#[cfg_attr(miri, ignore)]
#[test]
fn a_mistyped_value_is_an_error() {
    for (line, want) in [
        (
            "usi_hash = true",
            "`usi_hash` must be an integer, got boolean",
        ),
        (
            "usi_hash = \"1024\"",
            "`usi_hash` must be an integer, got string",
        ),
    ] {
        let err = compile(&default_with("usi_hash", Some(line))).expect_err("must fail");
        assert!(err.contains(want), "expected {want:?}, got {err}");
    }

    let err =
        compile(&default_with("flipped_book", Some("flipped_book = 1"))).expect_err("must fail");
    assert!(
        err.contains("`flipped_book` must be a boolean, got integer"),
        "message: {err}"
    );

    let err = compile(&default_with("eval_dir", Some("eval_dir = 3"))).expect_err("must fail");
    assert!(
        err.contains("`eval_dir` must be a string, got integer"),
        "message: {err}"
    );
}

#[cfg_attr(miri, ignore)]
#[test]
fn an_out_of_range_value_is_an_error() {
    for (line, want) in [
        ("usi_hash = 0", "`usi_hash` = 0 is outside [1, 33554432]"),
        ("usi_hash = 33554433", "is outside [1, 33554432]"),
    ] {
        let err = compile(&default_with("usi_hash", Some(line))).expect_err("must fail");
        assert!(err.contains(want), "expected {want:?}, got {err}");
    }

    let err = compile(&default_with("threads", Some("threads = 0"))).expect_err("must fail");
    assert!(err.contains("`threads` = 0 is outside"), "message: {err}");

    let err = compile(&default_with(
        "draw_value_black",
        Some("draw_value_black = -30001"),
    ))
    .expect_err("must fail");
    assert!(err.contains("is outside [-30000, 30000]"), "message: {err}");
}

#[cfg_attr(miri, ignore)]
#[test]
fn an_unlisted_combo_choice_is_an_error() {
    let err = compile(&default_with(
        "book_file",
        Some("book_file = \"my_book.db\""),
    ))
    .expect_err("must fail");
    assert!(
        err.contains("`book_file` = \"my_book.db\" is not one of: no_book,"),
        "the message must list the choices: {err}"
    );

    let err = compile(&default_with(
        "entering_king_rule",
        Some("entering_king_rule = \"CSARule99\""),
    ))
    .expect_err("must fail");
    assert!(
        err.contains("is not one of: NoEnteringKing,"),
        "message: {err}"
    );
}

// --- Fail-loud: the accepted TOML subset ----------------------------------

#[cfg_attr(miri, ignore)]
#[test]
fn syntax_outside_the_accepted_subset_is_an_error() {
    for (body, want) in [
        (
            "[engine]\nusi_hash = 1024\n",
            "table headers are not supported",
        ),
        ("usi_hash\n", "expected `key = value`"),
        ("usi_hash =\n", "missing value after `=`"),
        (
            "usi_hash = 12abc\n",
            "is not an integer, a boolean, or a quoted string",
        ),
        ("eval_dir = \"eval\n", "unterminated string value"),
        (
            "eval_dir = \"a\\\\b\"\n",
            "escape sequences are not supported",
        ),
        ("USI_HASH = 1\n", "is not a valid key"),
        ("eval-dir = \"eval\"\n", "is not a valid key"),
        (
            "book_file = [\"a\"]\n",
            "is not an integer, a boolean, or a quoted string",
        ),
    ] {
        let err = compile(body).expect_err("must fail on: {body}");
        assert!(
            err.contains(want),
            "for {body:?} expected {want:?}, got {err}"
        );
    }
}

/// The line number in a message is the line the operator has to go and fix.
#[cfg_attr(miri, ignore)]
#[test]
fn errors_carry_the_source_line() {
    let err = compile("# a comment\n\nusi_hash = true\n").expect_err("must fail");
    assert!(err.starts_with("test.toml:3: "), "message: {err}");
}

// --- The accepted subset really is accepted -------------------------------

#[cfg_attr(miri, ignore)]
#[test]
fn comments_underscores_and_spacing_are_accepted() {
    // A `#` inside a string is content, not a comment; digit separators are
    // stripped; whitespace around `=` is free.
    let text = default_with(
        "book_dir",
        Some("book_dir = \"book#1\"   # trailing comment"),
    );
    // Rewrite whatever `usi_hash` the checked-in file carries, with a digit
    // separator and no spacing around `=`.
    let text = text
        .lines()
        .map(|line| {
            let is_target = line.split_once('=').is_some_and(|(k, _)| {
                k.trim() == "usi_hash" && !line.trim_start().starts_with('#')
            });
            if is_target { "usi_hash=1_024" } else { line }
        })
        .collect::<Vec<_>>()
        .join("\n");
    let out = compile(&text).expect("must compile");
    assert!(out.contains("pub const BOOK_DIR: &str = \"book#1\";"));
    assert!(out.contains("pub const USI_HASH: i64 = 1024;"));
}

#[cfg_attr(miri, ignore)]
#[test]
fn negative_values_round_trip() {
    let out = compile(&default_config_text()).expect("compiles");
    assert!(out.contains("pub const BOOK_EVAL_WHITE_LIMIT: i64 = -140;"));
    assert!(out.contains("pub const DRAW_VALUE_BLACK: i64 = -2;"));
}

// --- Which file a build selects -------------------------------------------

#[test]
fn config_path_resolution() {
    use std::ffi::OsString;
    use std::path::{Path, PathBuf};

    let root = Path::new("/repo");

    // Unset: the checked-in play config, under the repository root.
    assert_eq!(
        config_path(root, None),
        PathBuf::from("/repo/configs/default.toml")
    );

    // Relative: resolved against the repository root, NOT the build script's
    // working directory (which is the package directory, and would therefore
    // name a different file).
    assert_eq!(
        config_path(root, Some(OsString::from("configs/test.toml"))),
        PathBuf::from("/repo/configs/test.toml")
    );

    // Absolute: used as given, so a config outside the tree is selectable.
    assert_eq!(
        config_path(root, Some(OsString::from("/etc/yorkie/event.toml"))),
        PathBuf::from("/etc/yorkie/event.toml")
    );
}

#[test]
fn the_generated_header_records_a_repository_relative_source() {
    use std::path::Path;

    let root = Path::new("/repo");
    assert_eq!(
        display_source(root, Path::new("/repo/configs/test.toml")),
        "configs/test.toml"
    );
    // A config outside the repository keeps its full path — there is no shorter
    // way to say which file it was.
    assert_eq!(
        display_source(root, Path::new("/etc/yorkie/event.toml")),
        "/etc/yorkie/event.toml"
    );
    assert_eq!(config_name(Path::new("/repo/configs/test.toml")), "test");
}
