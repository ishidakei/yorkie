//! Opening book support. [`Book::from_file`] auto-detects the YaneuraOu or legacy JSON backend
//! by a short content sniff (not the file extension). Under `tournament` the JSON and on-the-fly
//! backends are removed: the YaneuraOu marker is required and the book is loaded fully in memory.

#[cfg(not(feature = "tournament"))]
mod json;
mod yaneuraou;

#[cfg(not(feature = "tournament"))]
pub use self::json::JsonBook;
pub use self::yaneuraou::YaneuraouBook;

use crate::movetypes::Move;
use crate::position::Position;
use crate::usioption::UsiOptions;
use anyhow::Result;
use rand::prelude::ThreadRng;
use std::path::Path;

/// Leading bytes inspected to detect the book format.
const SNIFF_BYTES: usize = 256;

/// Snapshot of the YaneuraOu-compatible book options, so backends never read the option store.
#[derive(Clone, Copy, Debug)]
pub struct BookOptions {
    /// Only probe while the position's ply is within this limit.
    pub book_moves: i32,
    /// Match positions ignoring the trailing ply field of the SFEN.
    pub ignore_book_ply: bool,
    /// Skip entries whose recorded depth is below this limit.
    pub book_depth_limit: i32,
    /// Keep only candidates whose value is within this margin of the best candidate.
    pub book_eval_diff: i32,
    /// On a primary miss, retry against the horizontally mirrored position.
    pub flipped_book: bool,
    /// Stream a large book from disk instead of materializing all move payloads in memory.
    /// Removed under `tournament` (the book is always fully loaded into memory there).
    #[cfg(not(feature = "tournament"))]
    pub book_on_the_fly: bool,
}

impl BookOptions {
    /// Reads the current YaneuraOu-compatible book option values from the USI option store.
    pub fn from_usi_options(options: &UsiOptions) -> BookOptions {
        BookOptions {
            book_moves: options.get_i64(UsiOptions::BOOK_MOVES) as i32,
            ignore_book_ply: options.get_bool(UsiOptions::IGNORE_BOOK_PLY),
            book_depth_limit: options.get_i64(UsiOptions::BOOK_DEPTH_LIMIT) as i32,
            book_eval_diff: options.get_i64(UsiOptions::BOOK_EVAL_DIFF) as i32,
            flipped_book: options.get_bool(UsiOptions::FLIPPED_BOOK),
            #[cfg(not(feature = "tournament"))]
            book_on_the_fly: options.get_bool(UsiOptions::BOOK_ON_THE_FLY),
        }
    }
}

/// A loaded opening book: either the legacy JSON backend or the YaneuraOu backend. Under
/// `tournament` only the YaneuraOu backend exists.
pub enum Book {
    #[cfg(not(feature = "tournament"))]
    Json(JsonBook),
    Yaneuraou(YaneuraouBook),
}

impl Book {
    /// Loads an opening book from `path`, auto-detecting the format by content.
    #[cfg_attr(feature = "tournament", allow(unused_variables))]
    pub fn from_file<P>(path: P, options: &BookOptions) -> Result<Book>
    where
        P: AsRef<Path>,
    {
        let path = path.as_ref();
        if sniff_is_yaneuraou(path)? {
            #[cfg(not(feature = "tournament"))]
            let book = YaneuraouBook::from_path(path, options.book_on_the_fly)?;
            #[cfg(feature = "tournament")]
            let book = YaneuraouBook::from_path(path, false)?;
            Ok(Book::Yaneuraou(book))
        } else {
            #[cfg(not(feature = "tournament"))]
            {
                let file = std::fs::File::open(path)?;
                let book = JsonBook::from_reader(std::io::BufReader::new(file))?;
                Ok(Book::Json(book))
            }
            #[cfg(feature = "tournament")]
            {
                anyhow::bail!("not a YaneuraOu-format book: {}", path.display())
            }
        }
    }

    /// Probes the book for a move in `pos`; the `options` filters apply only to the YaneuraOu backend.
    pub fn probe(&self, pos: &Position, options: &BookOptions, rng: &mut ThreadRng) -> Option<Move> {
        match self {
            #[cfg(not(feature = "tournament"))]
            Book::Json(book) => book.probe(pos, rng),
            Book::Yaneuraou(book) => book.probe(pos, options, rng),
        }
    }
}

/// Returns `true` if the file's first non-whitespace bytes start with the YaneuraOu marker.
fn sniff_is_yaneuraou(path: &Path) -> Result<bool> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut head = [0u8; SNIFF_BYTES];
    let n = file.read(&mut head)?;
    let head = &head[..n];
    let trimmed = match head.iter().position(|b| !b.is_ascii_whitespace()) {
        Some(start) => &head[start..],
        None => return Ok(false),
    };
    Ok(trimmed.starts_with(yaneuraou::HEADER_MARKER.as_bytes()))
}

#[cfg(test)]
impl BookOptions {
    /// Permissive defaults for tests: no filtering, no flip, in-memory loader.
    pub fn for_test() -> BookOptions {
        BookOptions {
            book_moves: i32::MAX,
            ignore_book_ply: false,
            book_depth_limit: 0,
            book_eval_diff: i32::MAX,
            flipped_book: false,
            #[cfg(not(feature = "tournament"))]
            book_on_the_fly: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEADER: &str = "#YANEURAOU-DB2016 1.00";
    const START_SFEN: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

    struct TempBook {
        path: std::path::PathBuf,
    }
    impl TempBook {
        fn with(content: &str, tag: &str) -> Self {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let path = std::env::temp_dir().join(format!("yorkie-book-{}-{}-{}.db", tag, std::process::id(), nanos));
            std::fs::write(&path, content).expect("write temp book");
            Self { path }
        }
    }
    impl Drop for TempBook {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    fn run<F: FnOnce() + Send + 'static>(f: F) {
        std::thread::Builder::new()
            .stack_size(crate::stack_size::STACK_SIZE)
            .spawn(f)
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn from_file_detects_yaneuraou_by_content() {
        run(|| {
            // Leading blank lines before the marker must still be detected.
            let content = format!("\n\n{HEADER}\nsfen {START_SFEN}\n7g7f none 50 32 1\n");
            let tmp = TempBook::with(&content, "detect-yane");
            let book = Book::from_file(&tmp.path, &BookOptions::for_test()).unwrap();
            assert!(
                matches!(book, Book::Yaneuraou(_)),
                "marker file should select the YaneuraOu backend"
            );
        });
    }

    // Without `tournament`, JSON content selects the JSON backend via content sniffing.
    #[cfg(not(feature = "tournament"))]
    #[test]
    fn from_file_detects_json_by_content() {
        run(|| {
            let tmp = TempBook::with(r#"{"some/sfen b - 1":{"7g7f":{"value":1,"win":1,"lose":1}}}"#, "detect-json");
            let book = Book::from_file(&tmp.path, &BookOptions::for_test()).unwrap();
            assert!(matches!(book, Book::Json(_)), "JSON content should select the JSON backend");
        });
    }

    // Under `tournament` the JSON backend is gone, so JSON content is no longer a valid book: with
    // no YaneuraOu marker, `from_file` errors instead of falling back to a JSON parse.
    #[cfg(feature = "tournament")]
    #[test]
    fn from_file_rejects_json_content_under_tournament() {
        run(|| {
            let tmp = TempBook::with(r#"{"some/sfen b - 1":{"7g7f":{"value":1,"win":1,"lose":1}}}"#, "reject-json");
            assert!(
                Book::from_file(&tmp.path, &BookOptions::for_test()).is_err(),
                "JSON content must be rejected when the JSON backend is removed"
            );
        });
    }

    #[test]
    fn from_file_unknown_format_errors() {
        run(|| {
            let tmp = TempBook::with("this is not a book at all\n", "unknown");
            // Non-marker content errors in both builds (JSON parse failure, or missing YaneuraOu marker).
            assert!(Book::from_file(&tmp.path, &BookOptions::for_test()).is_err());
        });
    }

    #[test]
    fn probe_returns_book_move() {
        run(|| {
            let content = format!("{HEADER}\nsfen {START_SFEN}\n7g7f 3c3d 50 32 1\n");
            let tmp = TempBook::with(&content, "probe-move");
            let book = Book::from_file(&tmp.path, &BookOptions::for_test()).unwrap();
            let pos = Position::new_from_sfen(START_SFEN).unwrap();
            let mv = book.probe(&pos, &BookOptions::for_test(), &mut rand::thread_rng());
            assert_eq!(mv.unwrap().to_usi_string(), "7g7f");
        });
    }
}
