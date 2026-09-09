//! A read-only mapping of part of a file, on a huge-page boundary.
//!
//! What the evaluation network is read through when every worker on the machine
//! can share one copy of it: the parameters are laid out in the file exactly as
//! the kernels read them, so mapping the file is loading the network, and the
//! pages are the page cache's — every process on the machine that maps the same
//! file shares them.
//!
//! The mapping starts on a [`LARGE_PAGE_ALIGN`] boundary, which a plain `mmap`
//! does not promise: an aligned span is reserved first and the file is mapped
//! over it, so a huge-page hint over the region can be honoured.

use std::fmt;
use std::fs::File;
use std::io;
use std::path::Path;

use crate::large_page::LARGE_PAGE_ALIGN;
#[cfg(not(all(unix, not(miri))))]
use crate::large_page::LargePageArray;

/// A mapped byte range of a file, unmapped on drop.
pub struct MappedRegion {
    backing: Backing,
    len: usize,
}

/// Where the bytes actually live.
enum Backing {
    /// The file's own pages, shared with every other process that maps them.
    #[cfg(all(unix, not(miri)))]
    Mapped { addr: usize, span: usize },
    /// A copy in this process's memory, on the targets and under the
    /// interpreter that cannot map a file. The bytes, their alignment and their
    /// lifetime are the same; only the sharing is lost.
    #[cfg(not(all(unix, not(miri))))]
    Copied(LargePageArray<u8>),
}

impl fmt::Debug for MappedRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MappedRegion")
            .field("addr", &self.addr())
            .field("len", &self.len)
            .finish()
    }
}

// SAFETY: the region is read-only for its whole life and owned solely by this
// value, so sharing a reference to it across threads exposes nothing a `&[u8]`
// would not.
unsafe impl Send for MappedRegion {}
unsafe impl Sync for MappedRegion {}

impl MappedRegion {
    /// Map `len` bytes of `path` starting at `offset`, which must be a multiple
    /// of [`LARGE_PAGE_ALIGN`] so the mapped bytes land on such a boundary.
    ///
    /// Fails when the file is shorter than the range asked for, rather than
    /// leaving a mapping whose tail faults on the first read.
    pub fn open(path: &Path, offset: u64, len: usize) -> io::Result<Self> {
        assert!(
            offset.is_multiple_of(LARGE_PAGE_ALIGN as u64),
            "a mapped range starts on a large-page boundary",
        );
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
        Self::map(&file, offset, len)
    }

    /// The address of the first mapped byte. Handed to the kernel as a range
    /// descriptor, and used as the base the typed views are carved from.
    pub fn addr(&self) -> usize {
        match &self.backing {
            #[cfg(all(unix, not(miri)))]
            Backing::Mapped { addr, .. } => *addr,
            #[cfg(not(all(unix, not(miri))))]
            Backing::Copied(bytes) => bytes.as_ptr() as usize,
        }
    }

    /// The number of bytes mapped.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the range is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[cfg(all(unix, not(miri)))]
    fn map(file: &File, offset: u64, len: usize) -> io::Result<Self> {
        use std::os::fd::AsRawFd;

        let span = len.next_multiple_of(page_size());
        // A reservation one alignment unit longer than the mapping, so an
        // aligned start is inside it whatever the kernel picks.
        let reserved = span + LARGE_PAGE_ALIGN;
        // SAFETY: an anonymous, unreadable reservation. It maps no file, reads
        // and writes nothing, and either returns a fresh range or fails.
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                reserved,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if base == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        let base = base as usize;
        let addr = base.next_multiple_of(LARGE_PAGE_ALIGN);

        // SAFETY: `addr .. addr + span` lies inside the reservation this call
        // owns, so replacing it affects no other mapping in the process.
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
            let error = io::Error::last_os_error();
            // SAFETY: unmapping exactly the reservation made above.
            unsafe { libc::munmap(base as *mut libc::c_void, reserved) };
            return Err(error);
        }

        // Give back the reservation's slack, so the process holds the mapping
        // and nothing else.
        for (start, bytes) in [
            (base, addr - base),
            (addr + span, base + reserved - addr - span),
        ] {
            if bytes > 0 {
                // SAFETY: both ranges are parts of the reservation that the
                // file mapping does not cover.
                unsafe { libc::munmap(start as *mut libc::c_void, bytes) };
            }
        }

        Ok(Self {
            backing: Backing::Mapped { addr, span },
            len,
        })
    }

    #[cfg(not(all(unix, not(miri))))]
    fn map(file: &File, offset: u64, len: usize) -> io::Result<Self> {
        use std::io::{Read as _, Seek as _, SeekFrom};

        let mut file = file.try_clone()?;
        file.seek(SeekFrom::Start(offset))?;
        let mut bytes = LargePageArray::<u8>::zeroed(len.max(1));
        file.read_exact(&mut bytes[..len])?;
        Ok(Self {
            backing: Backing::Copied(bytes),
            len,
        })
    }
}

#[cfg(all(unix, not(miri)))]
impl Drop for MappedRegion {
    fn drop(&mut self) {
        let Backing::Mapped { addr, span } = self.backing;
        // SAFETY: exactly the range `map` left mapped, which this value owns
        // and nothing else names.
        unsafe { libc::munmap(addr as *mut libc::c_void, span) };
    }
}

/// The kernel's page size, which every unmapped range has to be a multiple of.
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
    use crate::large_page::LARGE_PAGE_ALIGN;

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
    fn maps_the_range_after_the_offset_on_a_large_page_boundary() {
        let mut bytes = vec![0u8; LARGE_PAGE_ALIGN];
        bytes.extend((0..1000u32).map(|i| i as u8));
        let path = temp_file("range", &bytes);

        let region = MappedRegion::open(&path, LARGE_PAGE_ALIGN as u64, 1000).expect("map");
        assert_eq!(region.len(), 1000);
        assert_eq!(
            region.addr() % LARGE_PAGE_ALIGN,
            0,
            "the mapped bytes start on a large-page boundary",
        );
        // SAFETY: the region is live and covers these bytes.
        let seen = unsafe { std::slice::from_raw_parts(region.addr() as *const u8, 1000) };
        assert_eq!(seen, &bytes[LARGE_PAGE_ALIGN..]);

        let _ = std::fs::remove_file(&path);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn a_file_shorter_than_the_range_is_refused() {
        let path = temp_file("short", &[0u8; 64]);
        let err = MappedRegion::open(&path, 0, 4096).expect_err("must refuse");
        assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);
        let _ = std::fs::remove_file(&path);
    }
}
