#[cfg(feature = "kppt")]
pub mod kppt;
#[cfg(feature = "material")]
pub mod material;
#[cfg(feature = "nnue")]
pub mod nnue;

#[cfg(any(
    all(feature = "kppt", feature = "material"),
    all(feature = "kppt", feature = "nnue"),
    all(feature = "material", feature = "nnue"),
))]
compile_error!("features `kppt`, `material`, and `nnue` are mutually exclusive — enable exactly one");

#[cfg(not(any(feature = "kppt", feature = "material", feature = "nnue")))]
compile_error!("one of `kppt`, `material`, or `nnue` must be enabled — the default is `kppt`");

#[cfg(all(feature = "nnue", not(target_arch = "x86_64")))]
compile_error!("feature `nnue` is supported on `x86_64` targets only");
