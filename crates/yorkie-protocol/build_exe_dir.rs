// Where the engine binary sits, as the build scripts that need it find it.
//
// Split from `build_config.rs` because only the two build scripts that place
// something beside the binary or identify a build by it use this, while every
// crate reading the config includes that file. `include!`d for the same reason
// that file is: a build script cannot depend on a member of the workspace it is
// building. It expects `std::path::{Path, PathBuf}` in scope.

/// The directory the engine binary a build produces will sit in, found from the
/// build script's `OUT_DIR`: the parent of the nearest ancestor named `build`.
///
/// Cargo puts `OUT_DIR` at `<target>/[<triple>/]<profile>/build/<pkg>-<hash>/out`,
/// and some nightly cargos one level deeper, at
/// `<target>/[<triple>/]<profile>/build/<pkg>/<hash>/out`; the binary sits in
/// `<target>/[<triple>/]<profile>/` under both, so the depth is searched for
/// rather than counted. An `OUT_DIR` with no `build` ancestor is a layout this
/// does not know and is an error: a guess would put the converted network, or
/// the build's identity, somewhere the binary does not look, and the build would
/// still succeed.
fn exe_dir(out_dir: &Path) -> Result<PathBuf, String> {
    out_dir
        .ancestors()
        .skip(1)
        .find(|dir| dir.file_name().is_some_and(|name| name == "build"))
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .ok_or_else(|| {
            format!(
                "OUT_DIR is `{}`, which is not a layout this build script knows: none of its \
                 ancestors is the `build` directory cargo puts build-script output under, so \
                 the directory the engine binary sits in cannot be found from it",
                out_dir.display()
            )
        })
}
