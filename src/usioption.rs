use crate::search::*;
use crate::thread::*;
use crate::tt::*;

#[derive(Clone)]
enum UsiOptionValue {
    #[allow(dead_code)]
    String {
        default: String,
        current: String,
    },
    #[allow(dead_code)]
    Filename {
        default: std::path::PathBuf,
        current: std::path::PathBuf,
    },
    Spin {
        default: i64,
        current: i64,
        min: i64,
        max: i64,
    },
    Check {
        default: bool,
        current: bool,
    },
    Button,
}

impl UsiOptionValue {
    #[allow(dead_code)]
    fn string(default: &str) -> UsiOptionValue {
        UsiOptionValue::String {
            default: default.to_string(),
            current: default.to_string(),
        }
    }
    #[allow(dead_code)]
    fn filename(default: &str) -> UsiOptionValue {
        UsiOptionValue::Filename {
            default: default.into(),
            current: default.into(),
        }
    }
    fn spin(default: i64, min: i64, max: i64) -> UsiOptionValue {
        UsiOptionValue::Spin {
            default,
            current: default,
            min,
            max,
        }
    }
    fn check(default: bool) -> UsiOptionValue {
        UsiOptionValue::Check {
            default,
            current: default,
        }
    }
}

#[derive(Clone)]
struct UsiOptionEntry {
    value: UsiOptionValue,
    fixed: bool,
}

impl UsiOptionEntry {
    fn new(value: UsiOptionValue) -> Self {
        Self { value, fixed: false }
    }
}

#[derive(Clone)]
pub struct UsiOptions {
    v: std::collections::HashMap<&'static str, UsiOptionEntry>,
}

impl UsiOptions {
    pub const BOOK_ENABLE: &'static str = "Book_Enable";
    pub const BOOK_FILE: &'static str = "Book_File";
    pub const BOOK_MOVES: &'static str = "BookMoves";
    pub const IGNORE_BOOK_PLY: &'static str = "IgnoreBookPly";
    pub const BOOK_DEPTH_LIMIT: &'static str = "BookDepthLimit";
    pub const BOOK_EVAL_DIFF: &'static str = "BookEvalDiff";
    pub const FLIPPED_BOOK: &'static str = "FlippedBook";
    pub const BOOK_ON_THE_FLY: &'static str = "BookOnTheFly";
    pub const BYOYOMI_MARGIN: &'static str = "Byoyomi_Margin";
    const CLEAR_HASH: &'static str = "Clear_Hash";
    pub const EVAL_DIR: &'static str = "Eval_Dir";
    #[cfg(feature = "nnue")]
    pub const FV_SCALE: &'static str = "FV_SCALE";
    pub const MINIMUM_THINKING_TIME: &'static str = "Minimum_Thinking_Time";
    pub const MOVE_OVERHEAD: &'static str = "Move_Overhead";
    pub const MULTI_PV: &'static str = "MultiPV";
    pub const OPENING_TIME_WEIGHT: &'static str = "Opening_Time_Weight";
    pub const THREADS: &'static str = "Threads";
    pub const TIMEOUT_SAFETY_MARGIN: &'static str = "Timeout_Safety_Margin";
    pub const USI_HASH: &'static str = "USI_Hash";
    pub const USI_PONDER: &'static str = "USI_Ponder";

    pub fn new() -> UsiOptions {
        let mut options = std::collections::HashMap::new();

        // The following are all options.
        options.insert(Self::BOOK_ENABLE, UsiOptionEntry::new(UsiOptionValue::check(false)));
        options.insert(
            Self::BOOK_FILE,
            UsiOptionEntry::new(UsiOptionValue::filename("book/20191216/book.json")),
        );
        // YaneuraOu-compatible opening-book probe options, snapshotted into `BookOptions`.
        const SPIN_MAX: i64 = i32::MAX as i64;
        options.insert(Self::BOOK_MOVES, UsiOptionEntry::new(UsiOptionValue::spin(16, 0, SPIN_MAX)));
        options.insert(Self::IGNORE_BOOK_PLY, UsiOptionEntry::new(UsiOptionValue::check(false)));
        options.insert(
            Self::BOOK_DEPTH_LIMIT,
            UsiOptionEntry::new(UsiOptionValue::spin(0, 0, SPIN_MAX)),
        );
        options.insert(
            Self::BOOK_EVAL_DIFF,
            UsiOptionEntry::new(UsiOptionValue::spin(30, 0, SPIN_MAX)),
        );
        options.insert(Self::FLIPPED_BOOK, UsiOptionEntry::new(UsiOptionValue::check(true)));
        options.insert(Self::BOOK_ON_THE_FLY, UsiOptionEntry::new(UsiOptionValue::check(false)));
        // Transmission reserve per byoyomi period (ms): the pure-byoyomi path bypasses the time
        // manager, so this is its one margin.
        options.insert(
            Self::BYOYOMI_MARGIN,
            UsiOptionEntry::new(UsiOptionValue::spin(500, 0, i64::MAX)),
        );
        options.insert(Self::CLEAR_HASH, UsiOptionEntry::new(UsiOptionValue::Button));
        #[cfg(feature = "nnue")]
        const EVAL_DIR_DEFAULT: &str = "eval/nnue";
        #[cfg(feature = "material")]
        const EVAL_DIR_DEFAULT: &str = "";
        options.insert(Self::EVAL_DIR, UsiOptionEntry::new(UsiOptionValue::string(EVAL_DIR_DEFAULT)));
        // .nnue files do not carry the post-accumulator divisor; default 16
        #[cfg(feature = "nnue")]
        options.insert(Self::FV_SCALE, UsiOptionEntry::new(UsiOptionValue::spin(16, 1, 128)));
        options.insert(Self::MULTI_PV, UsiOptionEntry::new(UsiOptionValue::spin(1, 1, 500)));
        // Time-management options (see `timeman::compute`). Per-move round-trip overhead (ms).
        options.insert(Self::MOVE_OVERHEAD, UsiOptionEntry::new(UsiOptionValue::spin(200, 0, 10000)));
        // Near-deadline cushion (ms): caps a single move in the low-time regime so it never times out.
        options.insert(
            Self::TIMEOUT_SAFETY_MARGIN,
            UsiOptionEntry::new(UsiOptionValue::spin(1200, 0, 60000)),
        );
        // Floor on the committed think time (ms).
        options.insert(
            Self::MINIMUM_THINKING_TIME,
            UsiOptionEntry::new(UsiOptionValue::spin(1000, 0, 60000)),
        );
        // Opening-phase time weight (percent): scales allocated time; 100 = neutral.
        options.insert(
            Self::OPENING_TIME_WEIGHT,
            UsiOptionEntry::new(UsiOptionValue::spin(100, 10, 1000)),
        );
        options.insert(Self::THREADS, UsiOptionEntry::new(UsiOptionValue::spin(1, 1, 8192)));
        const MAX_HASH_MB: usize = 0x200_0000;
        options.insert(
            Self::USI_HASH,
            UsiOptionEntry::new(UsiOptionValue::spin(256, 1, MAX_HASH_MB as i64)),
        );
        options.insert(Self::USI_PONDER, UsiOptionEntry::new(UsiOptionValue::check(true)));

        UsiOptions { v: options }
    }
    pub fn push_button(&self, key: &str, tt: &mut TranspositionTable) {
        match self.v.get(key).map(|entry| &entry.value) {
            None => {
                println!("Error: illegal option name: {}", key);
            }
            Some(UsiOptionValue::Button) => match key {
                Self::CLEAR_HASH => {
                    tt.clear();
                }
                _ => unreachable!(),
            },
            _ => {
                println!(r#"Error: The option "{}" isn't button type"#, key);
            }
        }
    }
    pub(crate) fn fixed_warning(name: &str) -> String {
        format!(
            "info string Error: option {} is fixed by eval_options.txt; setoption ignored",
            name
        )
    }

    pub fn set(
        &mut self,
        key: &str,
        value: &str,
        thread_pool: &mut ThreadPool,
        tt: &mut TranspositionTable,
        reductions: &mut Reductions,
        is_ready: &mut bool,
    ) {
        if let Some(entry) = self.v.get(key)
            && entry.fixed
        {
            println!("{}", Self::fixed_warning(key));
            return;
        }
        self.set_internal_unchecked(key, value, thread_pool, tt, reductions, is_ready);
    }

    fn set_internal_unchecked(
        &mut self,
        key: &str,
        value: &str,
        thread_pool: &mut ThreadPool,
        tt: &mut TranspositionTable,
        reductions: &mut Reductions,
        is_ready: &mut bool,
    ) {
        match self.v.get_mut(key).map(|entry| &mut entry.value) {
            None => {
                println!("Error: illegal option name: {}", key);
            }
            Some(UsiOptionValue::String { current, .. }) => {
                *current = value.to_string();
                if key == Self::EVAL_DIR {
                    *is_ready = false;
                }
            }
            Some(UsiOptionValue::Filename { current, .. }) => {
                *current = value.into();
                if key == Self::BOOK_FILE {
                    *is_ready = false;
                }
            }
            Some(UsiOptionValue::Spin { current, min, max, .. }) => match value.parse::<i64>() {
                Ok(n) => {
                    let n = std::cmp::min(n, *max);
                    let n = std::cmp::max(n, *min);
                    *current = n;
                    match key {
                        Self::THREADS => thread_pool.set(n as usize, tt, reductions),
                        Self::USI_HASH => tt.resize(n as usize, thread_pool),
                        #[cfg(feature = "nnue")]
                        Self::FV_SCALE => crate::evaluate::nnue::network::set_fv_scale(n as i32),
                        _ => {}
                    }
                }
                Err(err) => {
                    println!("{:?}", err);
                }
            },
            Some(UsiOptionValue::Check { current, .. }) => {
                let prev = *current;
                match value {
                    "true" => *current = true,
                    "false" => *current = false,
                    _ => println!("Error: illegal option value: {}", value),
                }
                let false_to_true = !prev && *current;
                if false_to_true && key == Self::BOOK_ENABLE {
                    *is_ready = false;
                }
            }
            Some(UsiOptionValue::Button) => println!(r#"Error: The option "{}" is button type. You can't set value to it."#, key),
        }
    }

    fn mark_fixed(&mut self, key: &str) {
        if let Some(entry) = self.v.get_mut(key) {
            entry.fixed = true;
        }
    }

    pub fn read_eval_options(
        &mut self,
        path: &std::path::Path,
        thread_pool: &mut ThreadPool,
        tt: &mut TranspositionTable,
        reductions: &mut Reductions,
        is_ready: &mut bool,
    ) -> Vec<String> {
        use std::io::BufRead;

        let mut lines = Vec::new();
        let file = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return lines,
            Err(e) => {
                lines.push(format!("info string Error: failed to open {}: {}", path.display(), e));
                return lines;
            }
        };
        lines.push(format!("info string read engine options, path = {}", path.display()));

        for raw_line in std::io::BufReader::new(file).lines() {
            let raw = match raw_line {
                Ok(s) => s,
                Err(e) => {
                    lines.push(format!("info string Error: read failure: {}", e));
                    continue;
                }
            };
            let trimmed = raw.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            // dlshogi-style `X=Y` is normalised to the simplified `X Y` form
            let normalised = trimmed.replace('=', " ");
            let tokens: Vec<&str> = normalised.split_whitespace().collect();
            if tokens.is_empty() {
                continue;
            }

            let parsed = if tokens[0] == "option" {
                parse_full_usi_form(&tokens)
            } else if tokens.len() >= 2 {
                Some((tokens[0].to_string(), tokens[1].to_string()))
            } else {
                None
            };
            let (name, raw_value) = match parsed {
                Some(pair) => pair,
                None => {
                    lines.push(format!("info string Error: malformed line: {}", trimmed));
                    continue;
                }
            };

            if !self.v.contains_key(name.as_str()) {
                lines.push(format!("info string Error: option name not found: {}", name));
                continue;
            }

            // Pre-clamp so the emitted override line reflects the stored value.
            let value_to_apply = match self.v.get(name.as_str()).map(|e| &e.value) {
                Some(UsiOptionValue::Spin { min, max, .. }) => match raw_value.parse::<i64>() {
                    Ok(n) => std::cmp::min(std::cmp::max(n, *min), *max).to_string(),
                    Err(err) => {
                        lines.push(format!("info string Error: invalid spin value for {}: {:?}", name, err));
                        continue;
                    }
                },
                _ => raw_value.clone(),
            };

            self.set_internal_unchecked(&name, &value_to_apply, thread_pool, tt, reductions, is_ready);
            self.mark_fixed(&name);

            lines.push(format!(
                "info string engine option override. name = {} , value = {}",
                name, value_to_apply
            ));
        }
        lines
    }
    pub fn to_usi_string(&self) -> String {
        let mut s = self
            .v
            .iter()
            .map(|(key, entry)| match &entry.value {
                UsiOptionValue::String { default, .. } => {
                    format!("option name {} type string default {}", key, default)
                }
                UsiOptionValue::Filename { default, .. } => {
                    format!("option name {} type filename default {}", key, default.to_string_lossy())
                }
                UsiOptionValue::Spin { default, min, max, .. } => {
                    format!("option name {} type spin default {} min {} max {}", key, default, min, max)
                }
                UsiOptionValue::Check { default, .. } => {
                    format!("option name {} type check default {}", key, default)
                }
                UsiOptionValue::Button => format!("option name {} type button", key),
            })
            .collect::<Vec<_>>();
        s.sort_unstable();
        s.join("\n") // The last line has no "\n".
    }
    pub fn get_i64(&self, key: &str) -> i64 {
        match self.v.get(key).map(|entry| &entry.value) {
            Some(UsiOptionValue::Spin { current, .. }) => *current,
            _ => panic!("Error: illegal option name: {}", key),
        }
    }
    #[allow(dead_code)]
    pub fn get_string(&self, key: &str) -> String {
        match self.v.get(key).map(|entry| &entry.value) {
            Some(UsiOptionValue::String { current, .. }) => current.clone(),
            _ => panic!("Error: illegal option name: {}", key),
        }
    }
    #[allow(dead_code)]
    pub fn get_filename(&self, key: &str) -> std::path::PathBuf {
        match self.v.get(key).map(|entry| &entry.value) {
            Some(UsiOptionValue::Filename { current, .. }) => current.clone(),
            _ => panic!("Error: illegal option name: {}", key),
        }
    }
    pub fn get_bool(&self, key: &str) -> bool {
        match self.v.get(key).map(|entry| &entry.value) {
            Some(UsiOptionValue::Check { current, .. }) => *current,
            _ => panic!("Error: illegal option name: {}", key),
        }
    }
}

// Walks `option name <X> type <T> default <Y> [min …] [max …] [var …]` and
// returns `(X, Y)`, or None if either anchor is missing.
fn parse_full_usi_form(tokens: &[&str]) -> Option<(String, String)> {
    let mut name: Option<String> = None;
    let mut value: Option<String> = None;
    let mut i = 1; // skip leading "option"
    while i < tokens.len() {
        match tokens[i] {
            "name" if i + 1 < tokens.len() => {
                name = Some(tokens[i + 1].to_string());
                i += 2;
            }
            "default" if i + 1 < tokens.len() => {
                value = Some(tokens[i + 1].to_string());
                i += 2;
            }
            "type" | "min" | "max" | "var" if i + 1 < tokens.len() => {
                i += 2;
            }
            _ => i += 1,
        }
    }
    match (name, value) {
        (Some(n), Some(v)) => Some((n, v)),
        _ => None,
    }
}

#[cfg(all(test, feature = "nnue"))]
mod fv_scale_option_tests {
    use super::*;

    fn set_fv_scale_via_usi(opts: &mut UsiOptions, value: &str, is_ready: &mut bool) {
        let mut thread_pool = ThreadPool::new();
        let mut tt = TranspositionTable::new();
        let mut reductions = Reductions::new();
        opts.set(
            UsiOptions::FV_SCALE,
            value,
            &mut thread_pool,
            &mut tt,
            &mut reductions,
            is_ready,
        );
    }

    #[test]
    fn fv_scale_default_is_16_and_advertised() {
        let opts = UsiOptions::new();
        let advertised = opts.to_usi_string();
        assert!(
            advertised.contains("option name FV_SCALE type spin default 16 min 1 max 128"),
            "expected FV_SCALE advertised at default 16 / [1, 128]; got:\n{advertised}"
        );
        assert_eq!(opts.get_i64(UsiOptions::FV_SCALE), 16);
    }

    #[test]
    fn fv_scale_set_clamps_out_of_range() {
        let _guard = crate::evaluate::nnue::TEST_MUTEX.lock().expect("TEST_MUTEX poisoned");

        let mut opts = UsiOptions::new();
        let mut is_ready = true;

        for (input, expected) in [("0", 1i64), ("-1", 1), ("129", 128), (&i64::MAX.to_string(), 128)] {
            set_fv_scale_via_usi(&mut opts, input, &mut is_ready);
            assert_eq!(
                opts.get_i64(UsiOptions::FV_SCALE),
                expected,
                "value {input} should clamp to {expected}",
            );
        }

        crate::evaluate::nnue::network::set_fv_scale(16);
    }

    #[test]
    fn fv_scale_set_rejects_non_integer() {
        let _guard = crate::evaluate::nnue::TEST_MUTEX.lock().expect("TEST_MUTEX poisoned");

        let mut opts = UsiOptions::new();
        let mut is_ready = true;
        set_fv_scale_via_usi(&mut opts, "32", &mut is_ready);
        assert_eq!(opts.get_i64(UsiOptions::FV_SCALE), 32);

        set_fv_scale_via_usi(&mut opts, "not-a-number", &mut is_ready);
        assert_eq!(
            opts.get_i64(UsiOptions::FV_SCALE),
            32,
            "non-integer input must leave the previous value untouched",
        );

        crate::evaluate::nnue::network::set_fv_scale(16);
    }

    #[test]
    fn fv_scale_set_does_not_toggle_is_ready() {
        let _guard = crate::evaluate::nnue::TEST_MUTEX.lock().expect("TEST_MUTEX poisoned");

        let mut opts = UsiOptions::new();
        let mut is_ready = true;
        set_fv_scale_via_usi(&mut opts, "28", &mut is_ready);
        assert!(
            is_ready,
            "FV_SCALE setter must not request an `isready` round-trip — the network bytes are unchanged",
        );
        assert_eq!(opts.get_i64(UsiOptions::FV_SCALE), 28);

        crate::evaluate::nnue::network::set_fv_scale(16);
    }
}

#[cfg(test)]
mod time_management_option_tests {
    use super::*;

    #[test]
    fn advertises_behavior_named_time_options() {
        let opts = UsiOptions::new();
        let advertised = opts.to_usi_string();
        for expected in [
            "option name Move_Overhead type spin default 200 min 0 max 10000",
            "option name Timeout_Safety_Margin type spin default 1200 min 0 max 60000",
            "option name Minimum_Thinking_Time type spin default 1000 min 0 max 60000",
            "option name Opening_Time_Weight type spin default 100 min 10 max 1000",
        ] {
            assert!(
                advertised.contains(expected),
                "missing advertised option line: {expected}\n{advertised}"
            );
        }
        // The old apery name is gone, and no time-loss is ever called a "flag".
        assert!(
            !advertised.contains("Slow_Mover"),
            "Slow_Mover must be renamed to Opening_Time_Weight"
        );
        // Time_Margin is retired: Move_Overhead + Timeout_Safety_Margin own the main-clock reserve.
        assert!(
            !advertised.contains("Time_Margin"),
            "Time_Margin must not be advertised: Move_Overhead + Timeout_Safety_Margin own its role"
        );
        assert!(
            advertised.contains("option name Byoyomi_Margin type spin default 500"),
            "Byoyomi_Margin must remain for the pure-byoyomi path"
        );
        assert!(
            !advertised.to_lowercase().contains("flag"),
            "no option string may use 'flag' wording (a time-loss is a timeout)"
        );

        assert_eq!(opts.get_i64(UsiOptions::MOVE_OVERHEAD), 200);
        assert_eq!(opts.get_i64(UsiOptions::TIMEOUT_SAFETY_MARGIN), 1200);
        assert_eq!(opts.get_i64(UsiOptions::MINIMUM_THINKING_TIME), 1000);
        assert_eq!(opts.get_i64(UsiOptions::OPENING_TIME_WEIGHT), 100);
    }
}

#[cfg(test)]
mod fixed_flag_tests {
    use super::*;

    fn set_multi_pv_via_usi(opts: &mut UsiOptions, value: &str) {
        let mut thread_pool = ThreadPool::new();
        let mut tt = TranspositionTable::new();
        let mut reductions = Reductions::new();
        let mut is_ready = true;
        opts.set(
            UsiOptions::MULTI_PV,
            value,
            &mut thread_pool,
            &mut tt,
            &mut reductions,
            &mut is_ready,
        );
    }

    #[test]
    fn set_short_circuits_when_option_is_fixed() {
        let mut opts = UsiOptions::new();
        set_multi_pv_via_usi(&mut opts, "4");
        assert_eq!(opts.get_i64(UsiOptions::MULTI_PV), 4);

        opts.mark_fixed(UsiOptions::MULTI_PV);
        set_multi_pv_via_usi(&mut opts, "8");
        assert_eq!(opts.get_i64(UsiOptions::MULTI_PV), 4, "set() must not mutate a fixed option",);
    }

    #[test]
    fn set_still_mutates_non_fixed_options() {
        let mut opts = UsiOptions::new();
        opts.mark_fixed(UsiOptions::USI_HASH);
        set_multi_pv_via_usi(&mut opts, "7");
        assert_eq!(
            opts.get_i64(UsiOptions::MULTI_PV),
            7,
            "marking USI_Hash fixed must not affect MultiPV",
        );
    }
}

#[cfg(test)]
mod eval_options_tests {
    use super::*;
    use std::path::{Path, PathBuf};

    struct TempFile {
        path: PathBuf,
    }
    impl TempFile {
        fn with(content: &str, tag: &str) -> Self {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let path = std::env::temp_dir().join(format!("yorkie-eval-options-{}-{}-{}.txt", tag, std::process::id(), nanos,));
            std::fs::write(&path, content).expect("write temp eval_options");
            Self { path }
        }
        fn path(&self) -> &Path {
            &self.path
        }
    }
    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    fn read_eval_options(opts: &mut UsiOptions, path: &Path) -> Vec<String> {
        let mut thread_pool = ThreadPool::new();
        let mut tt = TranspositionTable::new();
        let mut reductions = Reductions::new();
        let mut is_ready = true;
        opts.read_eval_options(path, &mut thread_pool, &mut tt, &mut reductions, &mut is_ready)
    }

    fn set_multi_pv_via_usi(opts: &mut UsiOptions, value: &str) {
        let mut thread_pool = ThreadPool::new();
        let mut tt = TranspositionTable::new();
        let mut reductions = Reductions::new();
        let mut is_ready = true;
        opts.set(
            UsiOptions::MULTI_PV,
            value,
            &mut thread_pool,
            &mut tt,
            &mut reductions,
            &mut is_ready,
        );
    }

    #[test]
    fn missing_file_returns_empty_and_emits_nothing() {
        let mut opts = UsiOptions::new();
        let missing = std::env::temp_dir().join("yorkie-eval-options-no-such.txt");
        let _ = std::fs::remove_file(&missing);
        let lines = read_eval_options(&mut opts, &missing);
        assert!(lines.is_empty(), "missing file must emit no info-strings; got {lines:?}");
    }

    #[test]
    fn simplified_form_applies_and_emits_canonical_lines() {
        let tmp = TempFile::with("MultiPV 4\n", "simplified");
        let mut opts = UsiOptions::new();
        let lines = read_eval_options(&mut opts, tmp.path());

        assert_eq!(lines.len(), 2, "expected exactly the read + override lines; got {lines:?}");
        assert_eq!(
            lines[0],
            format!("info string read engine options, path = {}", tmp.path().display()),
        );
        assert_eq!(lines[1], "info string engine option override. name = MultiPV , value = 4");
        assert_eq!(opts.get_i64(UsiOptions::MULTI_PV), 4);
    }

    #[test]
    fn equals_form_normalises_to_simplified() {
        let tmp = TempFile::with("MultiPV=4\n", "equals");
        let mut opts = UsiOptions::new();
        let lines = read_eval_options(&mut opts, tmp.path());
        assert_eq!(lines[1], "info string engine option override. name = MultiPV , value = 4");
        assert_eq!(opts.get_i64(UsiOptions::MULTI_PV), 4);
    }

    #[test]
    fn full_usi_form_picks_default_value() {
        let tmp = TempFile::with("option name MultiPV type spin default 4 min 1 max 500\n", "fullusi");
        let mut opts = UsiOptions::new();
        let lines = read_eval_options(&mut opts, tmp.path());
        assert_eq!(lines[1], "info string engine option override. name = MultiPV , value = 4");
        assert_eq!(opts.get_i64(UsiOptions::MULTI_PV), 4);
    }

    #[test]
    fn unknown_option_emits_error_and_continues() {
        let tmp = TempFile::with("BOGUS_OPTION 99\nMultiPV 4\n", "unknown");
        let mut opts = UsiOptions::new();
        let lines = read_eval_options(&mut opts, tmp.path());
        assert!(
            lines
                .iter()
                .any(|l| l == "info string Error: option name not found: BOGUS_OPTION"),
            "expected unknown-option error info-string; got {lines:?}",
        );
        assert_eq!(
            opts.get_i64(UsiOptions::MULTI_PV),
            4,
            "valid line after error must still apply",
        );
    }

    #[test]
    fn comments_and_blanks_are_skipped() {
        let tmp = TempFile::with("\n   \n# comment\n   # indented comment\nMultiPV 4\n", "comments");
        let mut opts = UsiOptions::new();
        let lines = read_eval_options(&mut opts, tmp.path());
        assert_eq!(lines.len(), 2, "comments/blanks must not emit info-strings; got {lines:?}");
        assert_eq!(opts.get_i64(UsiOptions::MULTI_PV), 4);
    }

    #[test]
    fn out_of_range_spin_clamps_in_emitted_line() {
        let tmp = TempFile::with("MultiPV 9999\n", "clamp");
        let mut opts = UsiOptions::new();
        let lines = read_eval_options(&mut opts, tmp.path());
        assert_eq!(lines[1], "info string engine option override. name = MultiPV , value = 500");
        assert_eq!(opts.get_i64(UsiOptions::MULTI_PV), 500);
    }

    #[test]
    fn successful_override_flips_fixed_flag() {
        let tmp = TempFile::with("MultiPV 4\n", "pinned");
        let mut opts = UsiOptions::new();
        let _ = read_eval_options(&mut opts, tmp.path());
        set_multi_pv_via_usi(&mut opts, "7");
        assert_eq!(
            opts.get_i64(UsiOptions::MULTI_PV),
            4,
            "read_eval_options must mark the entry fixed so subsequent set() is a no-op",
        );
    }

    #[cfg(feature = "nnue")]
    fn set_fv_scale_via_usi(opts: &mut UsiOptions, value: &str) {
        let mut thread_pool = ThreadPool::new();
        let mut tt = TranspositionTable::new();
        let mut reductions = Reductions::new();
        let mut is_ready = true;
        opts.set(
            UsiOptions::FV_SCALE,
            value,
            &mut thread_pool,
            &mut tt,
            &mut reductions,
            &mut is_ready,
        );
    }

    #[cfg(feature = "nnue")]
    #[test]
    fn fv_scale_simplified_form_applies_and_emits_canonical_lines() {
        let _guard = crate::evaluate::nnue::TEST_MUTEX.lock().expect("TEST_MUTEX poisoned");
        let tmp = TempFile::with("FV_SCALE 28\n", "fv-scale-simplified");
        let mut opts = UsiOptions::new();
        let lines = read_eval_options(&mut opts, tmp.path());

        assert_eq!(lines.len(), 2, "expected exactly the read + override lines; got {lines:?}");
        assert_eq!(
            lines[0],
            format!("info string read engine options, path = {}", tmp.path().display()),
        );
        assert_eq!(lines[1], "info string engine option override. name = FV_SCALE , value = 28");
        assert_eq!(opts.get_i64(UsiOptions::FV_SCALE), 28);

        crate::evaluate::nnue::network::set_fv_scale(16);
    }

    #[cfg(feature = "nnue")]
    #[test]
    fn fv_scale_equals_form_normalises_to_simplified() {
        let _guard = crate::evaluate::nnue::TEST_MUTEX.lock().expect("TEST_MUTEX poisoned");
        let tmp = TempFile::with("FV_SCALE=28\n", "fv-scale-equals");
        let mut opts = UsiOptions::new();
        let lines = read_eval_options(&mut opts, tmp.path());

        assert_eq!(lines[1], "info string engine option override. name = FV_SCALE , value = 28");
        assert_eq!(opts.get_i64(UsiOptions::FV_SCALE), 28);

        crate::evaluate::nnue::network::set_fv_scale(16);
    }

    #[cfg(feature = "nnue")]
    #[test]
    fn fv_scale_full_usi_form_picks_default_value() {
        let _guard = crate::evaluate::nnue::TEST_MUTEX.lock().expect("TEST_MUTEX poisoned");
        let tmp = TempFile::with(
            "option name FV_SCALE type spin default 28 min 1 max 128\n",
            "fv-scale-fullusi",
        );
        let mut opts = UsiOptions::new();
        let lines = read_eval_options(&mut opts, tmp.path());

        assert_eq!(lines[1], "info string engine option override. name = FV_SCALE , value = 28");
        assert_eq!(opts.get_i64(UsiOptions::FV_SCALE), 28);

        crate::evaluate::nnue::network::set_fv_scale(16);
    }

    #[cfg(feature = "nnue")]
    #[test]
    fn fv_scale_comments_and_blanks_are_skipped() {
        let _guard = crate::evaluate::nnue::TEST_MUTEX.lock().expect("TEST_MUTEX poisoned");
        let tmp = TempFile::with(
            "\n   \n# pinned FV_SCALE override\n   # indented comment\nFV_SCALE 28\n",
            "fv-scale-comments",
        );
        let mut opts = UsiOptions::new();
        let lines = read_eval_options(&mut opts, tmp.path());

        assert_eq!(lines.len(), 2, "comments/blanks must not emit info-strings; got {lines:?}");
        assert_eq!(lines[1], "info string engine option override. name = FV_SCALE , value = 28");
        assert_eq!(opts.get_i64(UsiOptions::FV_SCALE), 28);

        crate::evaluate::nnue::network::set_fv_scale(16);
    }

    #[cfg(feature = "nnue")]
    #[test]
    fn fv_scale_unknown_option_emits_error_and_continues() {
        let _guard = crate::evaluate::nnue::TEST_MUTEX.lock().expect("TEST_MUTEX poisoned");
        let tmp = TempFile::with("BOGUS_OPTION 99\nFV_SCALE 28\n", "fv-scale-unknown");
        let mut opts = UsiOptions::new();
        let lines = read_eval_options(&mut opts, tmp.path());

        assert!(
            lines
                .iter()
                .any(|l| l == "info string Error: option name not found: BOGUS_OPTION"),
            "expected unknown-option error info-string; got {lines:?}",
        );
        assert_eq!(
            opts.get_i64(UsiOptions::FV_SCALE),
            28,
            "valid line after error must still apply",
        );

        crate::evaluate::nnue::network::set_fv_scale(16);
    }

    #[cfg(feature = "nnue")]
    #[test]
    fn fv_scale_out_of_range_spin_clamps_in_emitted_line() {
        let _guard = crate::evaluate::nnue::TEST_MUTEX.lock().expect("TEST_MUTEX poisoned");
        let tmp = TempFile::with("FV_SCALE 999\n", "fv-scale-clamp");
        let mut opts = UsiOptions::new();
        let lines = read_eval_options(&mut opts, tmp.path());

        assert_eq!(lines[1], "info string engine option override. name = FV_SCALE , value = 128");
        assert_eq!(opts.get_i64(UsiOptions::FV_SCALE), 128);

        crate::evaluate::nnue::network::set_fv_scale(16);
    }

    #[cfg(feature = "nnue")]
    #[test]
    fn setoption_emits_warning_when_option_is_fixed() {
        let _guard = crate::evaluate::nnue::TEST_MUTEX.lock().expect("TEST_MUTEX poisoned");

        let tmp = TempFile::with("FV_SCALE 28\n", "fv-scale-fixed-warning");
        let mut opts = UsiOptions::new();
        let lines = read_eval_options(&mut opts, tmp.path());
        assert_eq!(lines.len(), 2);
        assert_eq!(opts.get_i64(UsiOptions::FV_SCALE), 28);

        assert_eq!(
            UsiOptions::fixed_warning(UsiOptions::FV_SCALE),
            "info string Error: option FV_SCALE is fixed by eval_options.txt; setoption ignored",
        );

        set_fv_scale_via_usi(&mut opts, "30");
        assert_eq!(
            opts.get_i64(UsiOptions::FV_SCALE),
            28,
            "setoption must not override a value pinned by eval_options.txt",
        );

        crate::evaluate::nnue::network::set_fv_scale(16);
    }
}
