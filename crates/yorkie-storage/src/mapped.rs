//! Putting a file's own pages at an address the caller chose.
//!
//! What the evaluation network is read through when every worker on the machine
//! can share one copy of it: the parameters are laid out in the file exactly as
//! the kernels read them, so mapping the file is loading the network, and the
//! pages are the page cache's — every process on the machine that maps the same
//! file shares them.
//!
//! The address is the caller's, not the kernel's: the network lives at a place
//! this binary declares, and the mapping goes *there*, over the anonymous pages
//! that were reserved for it. That is what lets the parameters be reached at a
//! fixed address plus a constant offset while still being shared.

use std::fs::File;
use std::io;
use std::path::Path;

/// Replace the pages of `[addr, addr + len)` with `len` bytes of `path`
/// starting at `offset`, shared read-only with every other process that maps
/// them.
///
/// `addr` and `offset` must both be page-aligned, and the caller must own the
/// whole range — it is unmapped and replaced, so a range holding anything else
/// would lose it.
///
/// Fails when the file is shorter than the range asked for, rather than leaving
/// a mapping whose tail faults on the first read. Returns whether the pages were
/// replaced: `false` where this build cannot map a file at all, leaving the
/// range as it was for the caller to read the bytes into.
///
/// # Safety
/// `addr` must be the start of a range of at least `len` bytes this process owns
/// and nothing is reading, since every page of it is thrown away.
pub unsafe fn map_file_onto(addr: usize, len: usize, path: &Path, offset: u64) -> io::Result<bool> {
    let file = File::open(path)?;
    let size = file.metadata()?.len();
    if size < offset + len as u64 {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!(
                "file holds {size} bytes, {} were asked for",
                offset + len as u64
            ),
        ));
    }
    // SAFETY: forwarded to the caller, who owns the same obligation.
    unsafe { map(&file, addr, len, offset) }
}

#[cfg(all(unix, not(miri)))]
unsafe fn map(file: &File, addr: usize, len: usize, offset: u64) -> io::Result<bool> {
    use std::os::fd::AsRawFd;

    let page = page_size();
    assert!(
        addr.is_multiple_of(page) && offset.is_multiple_of(page as u64),
        "a file maps onto a page boundary, from a page boundary",
    );
    // Whole pages, so the mapping covers every byte asked for. The last one
    // runs past the end of the file only within its own page, which the kernel
    // zero-fills; nothing reads it.
    let span = len.next_multiple_of(page);

    // SAFETY: the caller vouches that this process owns `[addr, addr + span)`
    // and that nothing is reading it, so replacing those pages affects no other
    // mapping.
    let mapped = unsafe {
        libc::mmap(
            addr as *mut libc::c_void,
            span,
            libc::PROT_READ,
            libc::MAP_SHARED | libc::MAP_FIXED,
            file.as_raw_fd(),
            offset as libc::off_t,
        )
    };
    if mapped == libc::MAP_FAILED {
        return Err(io::Error::last_os_error());
    }
    Ok(true)
}

#[cfg(not(all(unix, not(miri))))]
unsafe fn map(file: &File, addr: usize, len: usize, offset: u64) -> io::Result<bool> {
    let _ = (file, addr, len, offset);
    Ok(false)
}

/// The kernel's page size, which a mapping's address and file offset both have
/// to be a multiple of.
#[cfg(all(unix, not(miri)))]
fn page_size() -> usize {
    // SAFETY: `sysconf` reads a kernel parameter and touches no memory of ours.
    let reported = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if reported > 0 {
        reported as usize
    } else {
        4096
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::large_page::{LARGE_PAGE_ALIGN, LargePageArray};

    fn temp_file(tag: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "yorkie-mapped-{}-{tag}-{:?}",
            std::process::id(),
            std::thread::current().id(),
        ));
        std::fs::write(&path, bytes).expect("write the fixture file");
        path
    }

    // Both tests read a file, which the interpreter running the crate's tests
    // does not do.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn the_file_lands_at_the_address_it_was_given() {
        let mut bytes = vec![0u8; LARGE_PAGE_ALIGN];
        bytes.extend((0..1000u32).map(|i| i as u8));
        let path = temp_file("range", &bytes);

        // A block this test owns, standing in for the region the engine's
        // network is declared at.
        let target = LargePageArray::<u8>::zeroed(LARGE_PAGE_ALIGN);
        let addr = target.as_ptr() as usize;
        // SAFETY: `target` is this test's own block, nothing else reads it, and
        // it is large-page aligned and long enough.
        let mapped =
            unsafe { map_file_onto(addr, 1000, &path, LARGE_PAGE_ALIGN as u64).expect("map") };
        assert!(mapped, "a unix build maps the file");

        // SAFETY: the mapping is live and covers these bytes.
        let seen = unsafe { std::slice::from_raw_parts(addr as *const u8, 1000) };
        assert_eq!(seen, &bytes[LARGE_PAGE_ALIGN..]);

        // Put anonymous pages back, so the block can be freed as it was
        // allocated.
        // SAFETY: exactly the range replaced above, which this test owns.
        unsafe {
            libc::mmap(
                addr as *mut libc::c_void,
                LARGE_PAGE_ALIGN,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_FIXED,
                -1,
                0,
            );
        }
        let _ = std::fs::remove_file(&path);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn a_file_shorter_than_the_range_is_refused() {
        let path = temp_file("short", &[0u8; 64]);
        let target = LargePageArray::<u8>::zeroed(LARGE_PAGE_ALIGN);
        // SAFETY: `target` is this test's own block and long enough for the
        // range asked for; the call refuses before touching it.
        let err = unsafe { map_file_onto(target.as_ptr() as usize, 4096, &path, 0) }
            .expect_err("must refuse");
        assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);
        let _ = std::fs::remove_file(&path);
    }
}
