// The SPSA settings file's reader and code generator — the pure half of the
// parameter values `build.rs` compiles into the search.
//
// The search parameters are registered in `src/params.rs`: each one's name, the
// constant the search reads it through, its doc comment and the range of values
// the search tolerates. Their values are not there. They come from an SPSA
// settings file, one `name = value` line per registered parameter, and this
// file turns that into one `const` per parameter for `params.rs` to include.
//
// Nothing here touches the environment, the filesystem or the process; every
// failure is a returned `Err(String)`, so `tests/spsa_settings.rs` can assert
// on the refusals and check the checked-in files against the registry. It is
// `include!`d for the reason `build_config.rs` is, and it names every `std`
// item by its full path so that it can sit beside that file's imports.

/// The environment variable naming the SPSA settings file to compile in. It
/// takes precedence over the engine config's `spsa_config` key.
const SPSA_CONFIG_ENV: &str = "YORKIE_SPSA_CONFIG";

/// One parameter `params.rs` registers.
#[derive(Debug, PartialEq, Eq)]
struct Registered {
    /// The constant the search reads it through.
    id: String,
    /// Its name in an SPSA settings file and in a tuning list.
    name: String,
    /// The smallest value the search tolerates.
    min: i64,
    /// The largest value the search tolerates.
    max: i64,
}

/// The parameters the `search_params!` invocation of `params.rs` registers, in
/// its order. Each is one line, `ID = b"name", MIN..=MAX;`, among doc comments,
/// comments and blank lines; anything else in the invocation is refused rather
/// than skipped, so a parameter cannot drop out of the check unseen.
fn registered_params(source: &str) -> Result<Vec<Registered>, String> {
    let mut lines = source.lines().enumerate();
    if !lines.any(|(_, l)| l == "search_params! {") {
        return Err("params.rs: no `search_params! {` invocation".to_string());
    }
    let mut params = Vec::new();
    for (idx, raw) in lines {
        let line_no = idx + 1;
        if raw == "}" {
            return Ok(params);
        }
        let line = raw.trim();
        if line.is_empty() || line.starts_with("//") {
            continue;
        }
        let parsed = line.strip_suffix(';').and_then(|l| {
            let (id, rest) = l.split_once(" = b\"")?;
            let (name, rest) = rest.split_once("\", ")?;
            let (min, max) = rest.split_once("..=")?;
            Some((id, name, min.parse::<i64>().ok()?, max.parse::<i64>().ok()?))
        });
        let Some((id, name, min, max)) = parsed else {
            return Err(format!(
                "params.rs:{line_no}: expected `ID = b\"name\", MIN..=MAX;`, got `{line}`"
            ));
        };
        params.push(Registered {
            id: id.to_string(),
            name: name.to_string(),
            min,
            max,
        });
    }
    Err("params.rs: the `search_params!` invocation is not closed".to_string())
}

/// The `name = value` lines of an SPSA settings file, each with the line it is
/// on, in the file's order. `#` starts a comment; blank lines are skipped.
/// The value is kept as written: whether it is an integer is judged against
/// the registry, so that the message can name the parameter.
fn parse_spsa(contents: &str, label: &str) -> Result<Vec<(String, String, usize)>, String> {
    let mut lines = Vec::new();
    for (idx, raw) in contents.lines().enumerate() {
        let line_no = idx + 1;
        let line = raw.split_once('#').map_or(raw, |(l, _)| l).trim();
        if line.is_empty() {
            continue;
        }
        let Some((name, value)) = line.split_once('=') else {
            return Err(format!(
                "{label}:{line_no}: expected `name = value`, got `{line}`"
            ));
        };
        let name = name.trim();
        let valid = !name.is_empty()
            && name
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_');
        if !valid {
            return Err(format!(
                "{label}:{line_no}: `{name}` is not a parameter name (lowercase letters, \
                 digits and `_` only)"
            ));
        }
        lines.push((name.to_string(), value.trim().to_string(), line_no));
    }
    Ok(lines)
}

/// A decimal integer with an optional sign, and nothing else.
fn spsa_integer(value: &str) -> Option<i64> {
    let digits = value.strip_prefix(['-', '+']).unwrap_or(value);
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    value.parse().ok()
}

/// The value of every registered parameter, in the registry's order, or the
/// first thing wrong with the file: a line naming a parameter the registry does
/// not have, a parameter named twice, a value that is not an integer or is
/// outside the parameter's range, a registered parameter the file leaves out.
fn spsa_values(
    contents: &str,
    label: &str,
    registered: &[Registered],
) -> Result<Vec<i64>, String> {
    let mut values: Vec<Option<(i64, usize)>> = vec![None; registered.len()];
    for (name, value, line_no) in parse_spsa(contents, label)? {
        let Some(index) = registered.iter().position(|r| r.name == name) else {
            return Err(format!(
                "{label}:{line_no}: unknown parameter `{name}`: the engine registers no \
                 parameter of that name"
            ));
        };
        if let Some((_, first)) = values[index] {
            return Err(format!(
                "{label}:{line_no}: parameter `{name}` is given twice (first at line {first})"
            ));
        }
        let entry = &registered[index];
        let Some(parsed) = spsa_integer(&value) else {
            return Err(format!(
                "{label}:{line_no}: parameter `{name}`: `{value}` is not an integer"
            ));
        };
        if parsed < entry.min || parsed > entry.max {
            return Err(format!(
                "{label}:{line_no}: parameter `{name}` = {parsed} is outside the range the \
                 search tolerates, {}..={}",
                entry.min, entry.max
            ));
        }
        values[index] = Some((parsed, line_no));
    }
    let missing: Vec<&str> = registered
        .iter()
        .zip(&values)
        .filter(|(_, v)| v.is_none())
        .map(|(r, _)| r.name.as_str())
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "{label}: parameter(s) missing: {}\n       \
             an SPSA settings file gives a value to every parameter the engine registers",
            missing.join(", ")
        ));
    }
    Ok(values.into_iter().flatten().map(|(v, _)| v).collect())
}

/// The generated module: one `const` per registered parameter, named as the
/// parameter's constant, holding the file's value.
fn generate_spsa(registered: &[Registered], values: &[i64], source: &str) -> String {
    let mut out = format!(
        "// @generated from `{source}`. Do not edit: edit that file and rebuild.\n\n"
    );
    for (entry, value) in registered.iter().zip(values) {
        out.push_str(&format!("pub(crate) const {}: i32 = {value};\n", entry.id));
    }
    out
}

/// Read the registry out of `params.rs`, check the SPSA settings file against
/// it and render the module: the whole pure pipeline.
fn compile_spsa(
    contents: &str,
    label: &str,
    source: &str,
    params_rs: &str,
) -> Result<String, String> {
    let registered = registered_params(params_rs)?;
    let values = spsa_values(contents, label, &registered)?;
    Ok(generate_spsa(&registered, &values, source))
}

/// The SPSA settings file to read: the one [`SPSA_CONFIG_ENV`] names when it is
/// set, otherwise the one the engine config's `spsa_config` key names. A
/// relative path is taken against the repository root, as the engine config's
/// own path is.
fn spsa_config_path(
    repo_root: &std::path::Path,
    raw: Option<std::ffi::OsString>,
    from_config: &str,
) -> std::path::PathBuf {
    let raw = raw.map_or_else(
        || std::path::PathBuf::from(from_config),
        std::path::PathBuf::from,
    );
    if raw.is_absolute() {
        raw
    } else {
        repo_root.join(raw)
    }
}
