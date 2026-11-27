//! The SPSA settings files and their reader: the checked-in files against the
//! parameters `params.rs` registers, and the refusals a build makes of a file
//! that does not fit them.
//!
//! `build_spsa.rs` is `include!`d here exactly as `build.rs` includes it, so
//! these tests exercise the code that decides whether a build happens.

#![allow(dead_code)]

include!(concat!(env!("CARGO_MANIFEST_DIR"), "/build_spsa.rs"));

/// The two checked-in SPSA settings files, by repository-relative path.
const SPSA_FILES: &[&str] = &["configs/spsa/yaneuraou.toml", "configs/spsa/yorkie.toml"];

fn repo_file(rel: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(rel);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{rel} is readable: {e}"))
}

fn params_rs() -> String {
    repo_file("crates/yorkie-search/src/params.rs")
}

/// A small registry, so the refusals below do not depend on the real one.
const REGISTRY: &str = "\
search_params! {
    /// A doc comment.
    ALPHA = b\"alpha\", 0..=100;

    // A comment.
    BETA = b\"beta\", -50..=50;
}
";

fn compile(contents: &str) -> Result<String, String> {
    compile_spsa(contents, "x.toml", "x.toml", REGISTRY)
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_registry_reads_every_parameter_of_params_rs() {
    let registered = registered_params(&params_rs()).unwrap();
    assert_eq!(
        registered.len(),
        params_rs().matches(" = b\"").count(),
        "every registration line is read"
    );
    for (i, r) in registered.iter().enumerate() {
        assert!(r.min <= r.max, "{}", r.name);
        assert!(
            registered[..i]
                .iter()
                .all(|o| o.name != r.name && o.id != r.id),
            "{} is registered once",
            r.name
        );
        assert_eq!(r.id, r.name.to_ascii_uppercase(), "{}", r.name);
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn every_checked_in_spsa_file_compiles_against_the_registry() {
    let params = params_rs();
    for rel in SPSA_FILES {
        if let Err(e) = compile_spsa(&repo_file(rel), rel, rel, &params) {
            panic!("{rel} does not compile: {e}");
        }
    }
}

/// The two files name the same parameters. The test above already implies it,
/// as each must name every registered parameter and nothing else; this states
/// it on its own, as the names of the two files.
#[test]
#[cfg_attr(miri, ignore)]
fn the_two_spsa_files_name_the_same_parameters() {
    let names = |rel: &str| {
        let mut names: Vec<String> = parse_spsa(&repo_file(rel), rel)
            .unwrap()
            .into_iter()
            .map(|(name, _, _)| name)
            .collect();
        names.sort();
        names
    };
    assert_eq!(names(SPSA_FILES[0]), names(SPSA_FILES[1]));
}

#[test]
fn a_good_file_generates_one_constant_per_parameter_in_registry_order() {
    let code = compile("# a comment\nbeta = -7 # trailing\n\nalpha = +100\n").unwrap();
    assert!(code.starts_with("// @generated from `x.toml`."));
    assert!(
        code.ends_with("pub(crate) const ALPHA: i32 = 100;\npub(crate) const BETA: i32 = -7;\n")
    );
}

#[test]
fn a_missing_parameter_is_refused_by_name() {
    let e = compile("alpha = 1\n").unwrap_err();
    assert!(e.starts_with("x.toml: parameter(s) missing: beta"), "{e}");
}

#[test]
fn an_unknown_parameter_is_refused_by_name() {
    let e = compile("alpha = 1\nbeta = 2\ngamma = 3\n").unwrap_err();
    assert!(e.starts_with("x.toml:3: unknown parameter `gamma`"), "{e}");
}

#[test]
fn a_repeated_parameter_is_refused_by_name() {
    let e = compile("alpha = 1\nbeta = 2\nalpha = 3\n").unwrap_err();
    assert_eq!(
        e,
        "x.toml:3: parameter `alpha` is given twice (first at line 1)"
    );
}

#[test]
fn a_value_that_is_not_an_integer_is_refused_by_name() {
    for value in [
        "4.5",
        "",
        "abc",
        "\"5\"",
        "1_000",
        "true",
        "- 5",
        "0x10",
        "99999999999999999999",
    ] {
        let e = compile(&format!("alpha = {value}\nbeta = 0\n")).unwrap_err();
        assert_eq!(
            e,
            format!("x.toml:1: parameter `alpha`: `{value}` is not an integer"),
            "{value}"
        );
    }
}

#[test]
fn a_value_outside_the_range_is_refused_by_name() {
    for value in ["101", "-1"] {
        let e = compile(&format!("alpha = {value}\nbeta = 0\n")).unwrap_err();
        assert_eq!(
            e,
            format!(
                "x.toml:1: parameter `alpha` = {value} is outside the range the search \
                 tolerates, 0..=100"
            )
        );
    }
    assert!(compile("alpha = 0\nbeta = -50\n").is_ok());
    assert!(compile("alpha = 100\nbeta = 50\n").is_ok());
}

#[test]
fn a_line_that_is_not_name_value_is_refused() {
    let e = compile("alpha = 1\n[param]\n").unwrap_err();
    assert!(e.starts_with("x.toml:2: expected `name = value`"), "{e}");
    let e = compile("Alpha = 1\n").unwrap_err();
    assert!(
        e.starts_with("x.toml:1: `Alpha` is not a parameter name"),
        "{e}"
    );
}

#[test]
fn a_registration_line_the_reader_cannot_read_is_refused() {
    let e = registered_params("search_params! {\n    ALPHA = b\"alpha\", 0, 0..=100;\n}\n")
        .unwrap_err();
    assert!(e.starts_with("params.rs:2: expected"), "{e}");
    assert!(registered_params("fn main() {}\n").is_err());
    assert!(registered_params("search_params! {\n    ALPHA = b\"alpha\", 0..=1;\n").is_err());
}

#[test]
fn the_environment_variable_takes_precedence_over_the_config_key() {
    let root = std::path::Path::new("/repo");
    assert_eq!(
        spsa_config_path(root, None, "configs/spsa/yorkie.toml"),
        std::path::PathBuf::from("/repo/configs/spsa/yorkie.toml")
    );
    assert_eq!(
        spsa_config_path(
            root,
            Some("configs/spsa/yaneuraou.toml".into()),
            "configs/spsa/yorkie.toml"
        ),
        std::path::PathBuf::from("/repo/configs/spsa/yaneuraou.toml")
    );
    assert_eq!(
        spsa_config_path(
            root,
            Some("/elsewhere/s.toml".into()),
            "configs/spsa/yorkie.toml"
        ),
        std::path::PathBuf::from("/elsewhere/s.toml")
    );
    assert_eq!(
        spsa_config_path(root, None, "/elsewhere/s.toml"),
        std::path::PathBuf::from("/elsewhere/s.toml")
    );
}
