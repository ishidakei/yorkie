#[cfg(feature = "material")]
pub mod material;
#[cfg(feature = "nnue")]
pub mod nnue;

#[cfg(all(feature = "material", feature = "nnue"))]
compile_error!("features `material` and `nnue` are mutually exclusive — enable exactly one");

#[cfg(not(any(feature = "material", feature = "nnue")))]
compile_error!("one of `material` or `nnue` must be enabled — the default is `nnue`");

#[cfg(all(feature = "nnue", not(target_arch = "x86_64")))]
compile_error!("feature `nnue` is supported on `x86_64` targets only");
