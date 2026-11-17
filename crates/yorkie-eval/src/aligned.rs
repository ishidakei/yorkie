//! 64-byte-aligning allocator: a `Vec<T, Align64>` has a 64-byte base, so the
//! AVX-512 kernels' unaligned 512-bit loads from it never split a cache line.
//!
//! This is what the kernel parity tests hold a made-up weight or input buffer
//! in, so that what they feed a kernel sits on the cache lines the engine's own
//! parameters and accumulators do.

use std::alloc::{AllocError, Allocator, Global, Layout};
use std::ptr::NonNull;

/// Cache-line size assumed by the AVX-512 NNUE kernels.
const ALIGN: usize = 64;

/// An [`Allocator`] that forwards to [`Global`] with every request's alignment
/// raised to 64 bytes.
///
/// A zero-size request reports a non-null, 64-byte-aligned base, as `Global`
/// does for any zero-size layout. A `Vec<T, Align64>` that holds no capacity
/// never asks the allocator, so only a buffer with elements is guaranteed the
/// 64-byte base — which is every buffer a kernel reads.
#[derive(Clone, Copy, Debug, Default)]
pub struct Align64;

impl Align64 {
    /// `layout` with its alignment raised to [`ALIGN`]; the same raise is
    /// applied on allocation and on deallocation, so the two layouts match.
    fn raised(layout: Layout) -> Result<Layout, AllocError> {
        layout.align_to(ALIGN).map_err(|_| AllocError)
    }
}

/// Copies `src` into a fresh 64-byte-aligned buffer of the same length.
pub fn copy_of<T: Copy>(src: &[T]) -> Vec<T, Align64> {
    let mut buf = Vec::with_capacity_in(src.len(), Align64);
    buf.extend_from_slice(src);
    buf
}

// SAFETY: every method forwards to `Global` with the layout raised by
// `raised`, which is a pure function of the caller's layout, so a block is
// returned to `Global` with the exact layout it was allocated with; `Global`
// upholds the rest of the `Allocator` contract, and the raised alignment only
// strengthens what the caller asked for.
unsafe impl Allocator for Align64 {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Global.allocate(Self::raised(layout)?)
    }

    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Global.allocate_zeroed(Self::raised(layout)?)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        let raised = Self::raised(layout).expect("layout was raised when it was allocated");
        // SAFETY: the caller guarantees `ptr` came from this allocator with
        // `layout`, so `Global` allocated it with `raised`.
        unsafe { Global.deallocate(ptr, raised) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_aligned<T>(ptr: *const T) {
        assert_eq!(ptr.addr() % ALIGN, 0, "base pointer is not 64-byte aligned");
    }

    #[test]
    fn filled_is_aligned_and_zero() {
        for len in [1usize, 7, 15, 16, 17, 1_536, 24_576] {
            let mut buf = Vec::with_capacity_in(len, Align64);
            buf.resize(len, 0i16);
            assert_eq!(buf.len(), len);
            assert_aligned(buf.as_ptr());
            assert!(buf.iter().all(|&x| x == 0));
        }
    }

    #[test]
    fn allocate_zeroed_is_aligned_and_zero_at_every_length() {
        for len in [0usize, 1, 7, 15, 16, 17, 1_536, 24_576] {
            let layout = Layout::array::<i16>(len).expect("small array layout");
            let block = Align64.allocate_zeroed(layout).expect("allocation");
            assert_aligned(block.as_ptr().cast::<u8>());
            // SAFETY: `block` is a live allocation of at least `layout.size()` bytes.
            let bytes = unsafe { block.as_ref() };
            assert!(bytes.len() >= layout.size());
            assert!(bytes.iter().all(|&b| b == 0));
            // SAFETY: `block` came from `Align64` with `layout` and is freed once.
            unsafe { Align64.deallocate(block.cast(), layout) };
        }
    }

    #[test]
    fn copy_of_slice_preserves_bytes_and_aligns() {
        let src: Vec<i8> = (0..200i32).map(|i| (i - 100) as i8).collect();
        let buf = copy_of(&src);
        assert_aligned(buf.as_ptr());
        assert_eq!(&buf[..], &src[..]);
    }

    #[test]
    fn extend_from_iter_preserves_values_and_aligns() {
        let v: Vec<i32> = (0..50).map(|i| i * 3 - 7).collect();
        let mut buf = Vec::new_in(Align64);
        buf.extend(v.iter().copied());
        assert_aligned(buf.as_ptr());
        assert_eq!(&buf[..], &v[..]);
    }

    #[test]
    fn empty_is_valid() {
        let buf: Vec<i16, Align64> = Vec::new_in(Align64);
        assert!(buf.is_empty());
        let from_empty = copy_of::<i16>(&[]);
        assert!(from_empty.is_empty());
    }

    #[test]
    fn in_place_fill() {
        let mut buf = Vec::with_capacity_in(64, Align64);
        buf.resize(64, 0i32);
        for (i, slot) in buf.iter_mut().enumerate() {
            *slot = i as i32;
        }
        assert_aligned(buf.as_ptr());
        assert_eq!(buf[63], 63);
    }
}
