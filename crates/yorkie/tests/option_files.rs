//! That the engine does NOT read the option-override files on disk.
//!
//! `engine_options.txt` (current directory) and `<EvalDir>/eval_options.txt` were
//! the reference's way to reconfigure an engine without a rebuild. This engine
//! takes every setting from the TOML config compiled into it, in every build, so
//! it must not read either file: a file on disk that claims to set `USI_Hash` has
//! no authority here, and reading it — even to ignore it — would blur where the
//! engine's settings actually come from.
//!
//! `engine_options.txt` would resolve against the process working directory,
//! which a single-process test cannot change safely. So this spawns the built
//! binary with its own working directory instead, which is where the file would
//! have been read from.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// A temp directory removed on drop.
struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new(tag: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("yorkie-option-files-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("create temp dir");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// Run the engine binary in `cwd`, feed it `input`, return its stdout.
fn drive_in(cwd: &Path, input: &[u8]) -> String {
    let exe = env!("CARGO_BIN_EXE_yorkie");
    let mut child = Command::new(exe)
        .current_dir(cwd)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn engine");
    {
        let stdin = child.stdin.as_mut().expect("stdin piped");
        stdin.write_all(input).expect("write stdin");
    }
    let out = child.wait_with_output().expect("wait");
    assert!(
        out.status.success(),
        "engine exited non-zero: {:?}",
        out.status
    );
    String::from_utf8(out.stdout).expect("utf-8 stdout")
}

/// An override file the reference implementation would visibly act on.
fn write_engine_options(dir: &Path) {
    std::fs::write(dir.join("engine_options.txt"), "USI_Hash 16\nMultiPV 3\n")
        .expect("write engine_options.txt");
}

/// No build opens the file, so none says anything about it — and the `isready`
/// that follows is byte-identical to the one produced in a directory where no
/// such file exists.
#[cfg_attr(miri, ignore)]
#[test]
fn engine_options_txt_is_not_read() {
    let with_file = TempDir::new("with-file");
    write_engine_options(with_file.path());
    let out = drive_in(with_file.path(), b"isready\nquit\n");

    assert!(
        !out.contains("read engine options"),
        "no build may read engine_options.txt:\n{out}"
    );
    assert!(
        !out.contains("engine option override"),
        "no build may apply an override:\n{out}"
    );

    let without_file = TempDir::new("without-file");
    assert_eq!(
        out,
        drive_in(without_file.path(), b"isready\nquit\n"),
        "the presence of engine_options.txt must make no difference at all"
    );
}
