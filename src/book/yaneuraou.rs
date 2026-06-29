//! YaneuraOu-format opening book backend (`#YANEURAOU-DB2016`).
//! In-memory loader and an on-the-fly reader that binary-searches the SFEN-sorted file on disk.

use super::BookOptions;
use crate::movetypes::*;
use crate::position::*;
use rand::prelude::*;
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
// `Read`/`Seek`/`SeekFrom` and `PathBuf` are only used by the streaming on-the-fly reader, which is
// removed under `tournament`.
#[cfg(not(feature = "tournament"))]
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
#[cfg(not(feature = "tournament"))]
use std::path::PathBuf;

/// Marker that identifies the YaneuraOu book format (first non-blank line).
pub const HEADER_MARKER: &str = "#YANEURAOU-DB2016";

/// Parse error for YaneuraOu book input, with 1-based line numbers.
#[derive(thiserror::Error, Debug)]
pub enum BookError {
    #[error("opening book is missing the {HEADER_MARKER} header")]
    MissingHeader,
    #[error("move entry on line {line} appears before any `sfen` line")]
    EntryBeforeSfen { line: usize },
    #[error("malformed `sfen` line {line}")]
    MalformedSfen { line: usize },
    #[error("malformed move entry on line {line}: {reason}")]
    MalformedMoveEntry { line: usize, reason: String },
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// One candidate move stored for a position.
#[derive(Clone, Debug)]
struct MoveEntry {
    /// Best move in USI notation.
    best: String,
    #[allow(dead_code)]
    ponder: Option<String>,
    value: i32,
    depth: i32,
    num: u64,
}

/// The candidate moves recorded for a single `sfen` block, tagged with that block's ply.
#[derive(Clone, Debug)]
struct PositionEntry {
    ply: i32,
    moves: Vec<MoveEntry>,
}

/// Returns the body of a `sfen <body>` line, or `None` if the first token is not `sfen`.
fn sfen_body(line: &str) -> Option<&str> {
    let mut it = line.splitn(2, char::is_whitespace);
    if it.next()? == "sfen" {
        Some(it.next().unwrap_or("").trim())
    } else {
        None
    }
}

/// Splits an SFEN body into its ply-less key and ply (a missing ply defaults to 1).
fn split_sfen_key(body: &str) -> Option<(String, i32)> {
    let toks: Vec<&str> = body.split_whitespace().collect();
    if toks.len() < 3 {
        return None;
    }
    let trimmed = format!("{} {} {}", toks[0], toks[1], toks[2]);
    let ply = toks.get(3).and_then(|t| t.parse::<i32>().ok()).unwrap_or(1);
    Some((trimmed, ply))
}

fn parse_field<T: std::str::FromStr>(tok: Option<&&str>, name: &str, line: usize) -> Result<Option<T>, BookError> {
    match tok {
        Some(s) => s.parse::<T>().map(Some).map_err(|_| BookError::MalformedMoveEntry {
            line,
            reason: format!("invalid {name}: {s}"),
        }),
        None => Ok(None),
    }
}

/// Parses a single move-entry line. Returns `Ok(None)` for a `none` best move (no candidate).
fn parse_move_entry(line: &str, lineno: usize) -> Result<Option<MoveEntry>, BookError> {
    let toks: Vec<&str> = line.split_whitespace().collect();
    let best = match toks.first() {
        Some(b) => *b,
        None => return Ok(None),
    };
    if best == "none" {
        return Ok(None);
    }
    let ponder = toks.get(1).filter(|s| **s != "none").map(|s| s.to_string());
    let value = parse_field::<i32>(toks.get(2), "value", lineno)?.unwrap_or(0);
    let depth = parse_field::<i32>(toks.get(3), "depth", lineno)?.unwrap_or(0);
    let num = parse_field::<u64>(toks.get(4), "num", lineno)?.unwrap_or(1);
    Ok(Some(MoveEntry {
        best: best.to_string(),
        ponder,
        value,
        depth,
        num,
    }))
}

/// Parses the whole database into a ply-keyed map, validating structure with typed errors.
fn parse_lines<I>(lines: I) -> Result<BTreeMap<String, Vec<PositionEntry>>, BookError>
where
    I: Iterator<Item = std::io::Result<String>>,
{
    let mut map: BTreeMap<String, Vec<PositionEntry>> = BTreeMap::new();
    let mut header_seen = false;
    let mut cur: Option<(String, PositionEntry)> = None;
    let mut lineno = 0usize;

    let flush = |map: &mut BTreeMap<String, Vec<PositionEntry>>, cur: &mut Option<(String, PositionEntry)>| {
        if let Some((key, entry)) = cur.take() {
            map.entry(key).or_default().push(entry);
        }
    };

    for line in lines {
        let line = line?;
        lineno += 1;
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if !header_seen {
            if t.starts_with(HEADER_MARKER) {
                header_seen = true;
                continue;
            }
            return Err(BookError::MissingHeader);
        }
        if t.starts_with("//") || t.starts_with('#') {
            continue;
        }
        if let Some(body) = sfen_body(t) {
            flush(&mut map, &mut cur);
            let (trimmed, ply) = split_sfen_key(body).ok_or(BookError::MalformedSfen { line: lineno })?;
            cur = Some((trimmed, PositionEntry { ply, moves: Vec::new() }));
        } else {
            match &mut cur {
                Some((_, entry)) => {
                    if let Some(mv) = parse_move_entry(t, lineno)? {
                        entry.moves.push(mv);
                    }
                }
                None => return Err(BookError::EntryBeforeSfen { line: lineno }),
            }
        }
    }
    flush(&mut map, &mut cur);

    if !header_seen {
        return Err(BookError::MissingHeader);
    }
    Ok(map)
}

/// Fully in-memory YaneuraOu book.
#[derive(Debug)]
struct MemBook {
    map: BTreeMap<String, Vec<PositionEntry>>,
}

impl MemBook {
    fn candidates(&self, sfen: &str, ignore_ply: bool) -> Option<Vec<MoveEntry>> {
        let (trimmed, ply) = split_sfen_key(sfen)?;
        let group = self.map.get(&trimmed)?;
        let mut out = Vec::new();
        for entry in group {
            if ignore_ply || entry.ply == ply {
                out.extend(entry.moves.iter().cloned());
            }
        }
        if out.is_empty() { None } else { Some(out) }
    }
}

/// A located `sfen` block: its body and the byte offset where its move entries begin.
#[cfg(not(feature = "tournament"))]
#[derive(Debug)]
struct SfenHit {
    body: String,
    moves_start: u64,
}

/// First `sfen` line at or after byte `p` (a partial line straddling `p` is skipped); `Ok(None)` at EOF.
#[cfg(not(feature = "tournament"))]
fn first_sfen_from(file: &mut std::fs::File, p: u64) -> std::io::Result<Option<SfenHit>> {
    // If `p` isn't at a line boundary, discard the straddling partial line.
    let discard_partial = if p == 0 {
        false
    } else {
        file.seek(SeekFrom::Start(p - 1))?;
        let mut one = [0u8; 1];
        let n = file.read(&mut one)?;
        !(n == 1 && one[0] == b'\n')
    };
    file.seek(SeekFrom::Start(p))?;
    let mut reader = BufReader::new(&mut *file);
    let mut offset = p;
    let mut buf = String::new();
    if discard_partial {
        offset += reader.read_line(&mut buf)? as u64;
    }
    loop {
        buf.clear();
        let n = reader.read_line(&mut buf)?;
        if n == 0 {
            return Ok(None);
        }
        offset += n as u64;
        if let Some(body) = sfen_body(buf.trim()) {
            return Ok(Some(SfenHit {
                body: body.to_string(),
                moves_start: offset,
            }));
        }
    }
}

/// Reads one block's move entries starting at `moves_start`, plus the next `sfen` block if any.
#[cfg(not(feature = "tournament"))]
fn read_block_moves(file: &mut std::fs::File, moves_start: u64) -> std::io::Result<(Vec<MoveEntry>, Option<SfenHit>)> {
    file.seek(SeekFrom::Start(moves_start))?;
    let mut reader = BufReader::new(&mut *file);
    let mut offset = moves_start;
    let mut buf = String::new();
    let mut out = Vec::new();
    loop {
        buf.clear();
        let n = reader.read_line(&mut buf)?;
        if n == 0 {
            return Ok((out, None));
        }
        offset += n as u64;
        let t = buf.trim();
        if t.is_empty() || t.starts_with("//") || t.starts_with('#') {
            continue;
        }
        if let Some(body) = sfen_body(t) {
            return Ok((
                out,
                Some(SfenHit {
                    body: body.to_string(),
                    moves_start: offset,
                }),
            ));
        }
        // Best-effort parse: a malformed move line in an on-the-fly book is skipped, not fatal.
        if let Ok(Some(mv)) = parse_move_entry(t, 0) {
            out.push(mv);
        }
    }
}

/// Binary-searches the SFEN-sorted file for the first block whose ply-less key is `>= target`.
#[cfg(not(feature = "tournament"))]
fn lower_bound(file: &mut std::fs::File, file_size: u64, target: &str) -> std::io::Result<Option<SfenHit>> {
    let mut s = 0u64;
    let mut e = file_size;
    while s < e {
        let m = s + (e - s) / 2;
        match first_sfen_from(file, m)? {
            // No `sfen` line at or after `m`: the answer, if any, lies to the left.
            None => e = m,
            Some(hit) => match split_sfen_key(&hit.body) {
                Some((key, _)) if key.as_str() < target => s = hit.moves_start,
                Some(_) => e = m,
                // Malformed `sfen` body: step past it (offsets still advance, so this terminates).
                None => s = hit.moves_start,
            },
        }
    }
    first_sfen_from(file, s)
}

/// On-the-fly YaneuraOu book: binary-searches the SFEN-sorted file on disk instead of loading it.
#[cfg(not(feature = "tournament"))]
#[derive(Debug)]
struct OnTheFlyBook {
    path: PathBuf,
    file_size: u64,
}

#[cfg(not(feature = "tournament"))]
impl OnTheFlyBook {
    /// Opens the book and validates only its header line; the body is never scanned.
    fn open(path: &Path) -> Result<OnTheFlyBook, BookError> {
        let file = std::fs::File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut buf = String::new();
        loop {
            buf.clear();
            let n = reader.read_line(&mut buf)?;
            if n == 0 {
                return Err(BookError::MissingHeader);
            }
            let t = buf.trim();
            if t.is_empty() {
                continue;
            }
            if t.starts_with(HEADER_MARKER) {
                break;
            }
            return Err(BookError::MissingHeader);
        }
        let file_size = std::fs::metadata(path)?.len();
        Ok(OnTheFlyBook {
            path: path.to_path_buf(),
            file_size,
        })
    }

    fn candidates(&self, sfen: &str, ignore_ply: bool) -> Option<Vec<MoveEntry>> {
        let (target, ply) = split_sfen_key(sfen)?;
        let mut file = std::fs::File::open(&self.path).ok()?;
        let mut hit = lower_bound(&mut file, self.file_size, &target).ok()??;
        let mut out = Vec::new();
        // Blocks sharing a ply-less key are contiguous; scan forward while the key matches.
        while let Some((key, block_ply)) = split_sfen_key(&hit.body) {
            if key != target {
                break;
            }
            let (moves, following) = match read_block_moves(&mut file, hit.moves_start) {
                Ok(parts) => parts,
                Err(_) => break,
            };
            if ignore_ply || block_ply == ply {
                out.extend(moves);
            }
            match following {
                Some(next) => hit = next,
                None => break,
            }
        }
        if out.is_empty() { None } else { Some(out) }
    }
}

/// A loaded YaneuraOu opening book, either fully in memory or streamed from disk.
#[derive(Debug)]
pub struct YaneuraouBook {
    inner: Inner,
}

#[derive(Debug)]
enum Inner {
    Mem(MemBook),
    #[cfg(not(feature = "tournament"))]
    OnTheFly(OnTheFlyBook),
}

impl YaneuraouBook {
    /// Loads a book from `path`; when `on_the_fly`, binary-searches the file on disk instead of loading it.
    #[cfg_attr(feature = "tournament", allow(unused_variables))]
    pub fn from_path<P: AsRef<Path>>(path: P, on_the_fly: bool) -> Result<YaneuraouBook, BookError> {
        let path = path.as_ref();
        #[cfg(not(feature = "tournament"))]
        let inner = if on_the_fly {
            Inner::OnTheFly(OnTheFlyBook::open(path)?)
        } else {
            let file = std::fs::File::open(path)?;
            let map = parse_lines(BufReader::new(file).lines())?;
            Inner::Mem(MemBook { map })
        };
        #[cfg(feature = "tournament")]
        let inner = {
            let file = std::fs::File::open(path)?;
            let map = parse_lines(BufReader::new(file).lines())?;
            Inner::Mem(MemBook { map })
        };
        Ok(YaneuraouBook { inner })
    }

    fn candidates(&self, sfen: &str, ignore_ply: bool) -> Option<Vec<MoveEntry>> {
        match &self.inner {
            Inner::Mem(b) => b.candidates(sfen, ignore_ply),
            #[cfg(not(feature = "tournament"))]
            Inner::OnTheFly(b) => b.candidates(sfen, ignore_ply),
        }
    }

    /// Probes the book for the current position, honoring the YaneuraOu-compatible options.
    pub fn probe(&self, pos: &Position, options: &BookOptions, rng: &mut ThreadRng) -> Option<Move> {
        // `BookMoves` gating: stop consulting the book once we are past the limit.
        if pos.ply() > options.book_moves {
            return None;
        }
        let sfen = pos.to_sfen();
        if let Some(entries) = self.candidates(&sfen, options.ignore_book_ply)
            && let Some(mv) = select_from_entries(&entries, pos, options, false, rng)
        {
            return Some(mv);
        }
        // `FlippedBook`: retry against the horizontally mirrored position on a primary miss.
        if options.flipped_book
            && let Some(mirrored) = mirror_sfen(&sfen)
            && let Some(entries) = self.candidates(&mirrored, options.ignore_book_ply)
            && let Some(mv) = select_from_entries(&entries, pos, options, true, rng)
        {
            return Some(mv);
        }
        None
    }
}

/// Filters by depth/eval and picks a `num`-weighted random legal move (mapping mirrored moves back).
fn select_from_entries(
    entries: &[MoveEntry],
    pos: &Position,
    options: &BookOptions,
    mirrored: bool,
    rng: &mut ThreadRng,
) -> Option<Move> {
    let mut candidates: Vec<&MoveEntry> = entries.iter().filter(|e| e.depth >= options.book_depth_limit).collect();
    if candidates.is_empty() {
        return None;
    }
    let best_value = candidates.iter().map(|e| e.value).max()?;
    candidates.retain(|e| e.value >= best_value.saturating_sub(options.book_eval_diff));

    let mut weighted: Vec<(Move, f64)> = Vec::new();
    for e in &candidates {
        let usi = if mirrored {
            match mirror_usi_move(&e.best) {
                Some(s) => s,
                None => continue,
            }
        } else {
            e.best.clone()
        };
        if let Some(mv) = Move::new_from_usi_str(&usi, pos) {
            weighted.push((mv, e.num.max(1) as f64));
        }
    }
    if weighted.is_empty() {
        return None;
    }
    let dist = rand::distributions::WeightedIndex::new(weighted.iter().map(|x| x.1)).ok()?;
    Some(weighted[dist.sample(rng)].0)
}

/// Mirrors a file digit `d` (1..=9) to `10 - d`; returns `None` for any other character.
fn mirror_file_char(c: char) -> Option<char> {
    let d = c.to_digit(10)?;
    if (1..=9).contains(&d) {
        std::char::from_digit(10 - d, 10)
    } else {
        None
    }
}

/// Horizontally mirrors a USI move string (file `d → 10 − d`; ranks/promotion unchanged).
fn mirror_usi_move(usi: &str) -> Option<String> {
    let chars: Vec<char> = usi.chars().collect();
    if chars.len() >= 2 && chars[1] == '*' {
        // Drop move: `<piece>*<file><rank>`.
        if chars.len() != 4 {
            return None;
        }
        let file = mirror_file_char(chars[2])?;
        Some(format!("{}*{}{}", chars[0], file, chars[3]))
    } else {
        // Board move: `<file><rank><file><rank>[+]`.
        if chars.len() < 4 {
            return None;
        }
        let file_from = mirror_file_char(chars[0])?;
        let file_to = mirror_file_char(chars[2])?;
        let mut s = format!("{}{}{}{}", file_from, chars[1], file_to, chars[3]);
        if chars.len() == 5 && chars[4] == '+' {
            s.push('+');
        }
        Some(s)
    }
}

/// Horizontally mirrors the board of an SFEN string; side-to-move, hand, and ply are unchanged.
fn mirror_sfen(sfen: &str) -> Option<String> {
    let mut parts = sfen.splitn(2, ' ');
    let board = parts.next()?;
    let rest = parts.next().unwrap_or("");
    let mirrored = mirror_board(board)?;
    if rest.is_empty() {
        Some(mirrored)
    } else {
        Some(format!("{mirrored} {rest}"))
    }
}

/// Reverses the file order of each rank in an SFEN board field.
fn mirror_board(board: &str) -> Option<String> {
    let mut out = Vec::new();
    for rank in board.split('/') {
        let mut units: Vec<String> = Vec::new();
        let mut chars = rank.chars();
        while let Some(c) = chars.next() {
            if c == '+' {
                let piece = chars.next()?; // A promoted piece must follow `+`.
                units.push(format!("+{piece}"));
            } else {
                units.push(c.to_string());
            }
        }
        units.reverse();
        out.push(units.concat());
    }
    Some(out.join("/"))
}

#[cfg(test)]
mod tests {
    use super::*;
    // `super::*` re-exports `PathBuf` only without `tournament` (it is gated there); the test
    // helpers need it in both builds.
    #[cfg(feature = "tournament")]
    use std::path::PathBuf;

    const HEADER: &str = "#YANEURAOU-DB2016 1.00";
    const START_SFEN: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

    fn run<F: FnOnce() + Send + 'static>(f: F) {
        std::thread::Builder::new()
            .stack_size(crate::stack_size::STACK_SIZE)
            .spawn(f)
            .unwrap()
            .join()
            .unwrap();
    }

    struct TempBook {
        path: PathBuf,
    }
    impl TempBook {
        fn with(content: &str, tag: &str) -> Self {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let path = std::env::temp_dir().join(format!("yorkie-yane-{}-{}-{}.db", tag, std::process::id(), nanos));
            std::fs::write(&path, content).expect("write temp book");
            Self { path }
        }
    }
    impl Drop for TempBook {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    fn opts() -> BookOptions {
        BookOptions::for_test()
    }

    // ----- parser / error handling -----

    #[test]
    fn empty_file_is_missing_header() {
        let tmp = TempBook::with("", "empty");
        let err = YaneuraouBook::from_path(&tmp.path, false).unwrap_err();
        assert!(matches!(err, BookError::MissingHeader), "got {err:?}");
    }

    #[test]
    fn missing_header_errors() {
        let tmp = TempBook::with("sfen something b - 1\n7g7f none 0 1 1\n", "noheader");
        let err = YaneuraouBook::from_path(&tmp.path, false).unwrap_err();
        assert!(matches!(err, BookError::MissingHeader), "got {err:?}");
    }

    #[test]
    fn entry_before_sfen_errors_with_line() {
        // header on line 1, blank line 2, move entry on line 3 before any sfen line.
        let tmp = TempBook::with(&format!("{HEADER}\n\n7g7f none 0 1 1\n"), "early-entry");
        let err = YaneuraouBook::from_path(&tmp.path, false).unwrap_err();
        assert!(matches!(err, BookError::EntryBeforeSfen { line: 3 }), "got {err:?}");
    }

    #[test]
    fn malformed_numeric_errors_with_line() {
        let content = format!("{HEADER}\nsfen {START_SFEN}\n7g7f none not_a_number 1 1\n");
        let tmp = TempBook::with(&content, "bad-value");
        let err = YaneuraouBook::from_path(&tmp.path, false).unwrap_err();
        assert!(matches!(err, BookError::MalformedMoveEntry { line: 3, .. }), "got {err:?}");
    }

    #[test]
    fn malformed_sfen_errors_with_line() {
        let content = format!("{HEADER}\nsfen too few\n7g7f none 0 1 1\n");
        let tmp = TempBook::with(&content, "bad-sfen");
        let err = YaneuraouBook::from_path(&tmp.path, false).unwrap_err();
        assert!(matches!(err, BookError::MalformedSfen { line: 2 }), "got {err:?}");
    }

    #[test]
    fn comments_blanks_and_none_moves_are_tolerated() {
        run(|| {
            let content = format!(
                "{HEADER}\n\n// leading comment\n# another comment\nsfen {START_SFEN}\nnone none 0 0 0\n7g7f none 50 32 100\n"
            );
            let tmp = TempBook::with(&content, "tolerant");
            let book = YaneuraouBook::from_path(&tmp.path, false).unwrap();
            let pos = Position::new_from_sfen(START_SFEN).unwrap();
            let mv = book.probe(&pos, &opts(), &mut rand::thread_rng()).unwrap();
            assert_eq!(mv.to_usi_string(), "7g7f", "the `none` entry is skipped, leaving 7g7f");
        });
    }

    // ----- probe filters -----

    #[test]
    fn book_moves_gating() {
        run(|| {
            let content = format!("{HEADER}\nsfen {START_SFEN}\n7g7f none 50 32 1\n");
            let tmp = TempBook::with(&content, "gating");
            let book = YaneuraouBook::from_path(&tmp.path, false).unwrap();
            let pos = Position::new_from_sfen(START_SFEN).unwrap();

            let mut zero = opts();
            zero.book_moves = 0; // ply 1 > 0 → no book move
            assert!(book.probe(&pos, &zero, &mut rand::thread_rng()).is_none());

            let mut wide = opts();
            wide.book_moves = 16;
            assert!(book.probe(&pos, &wide, &mut rand::thread_rng()).is_some());
        });
    }

    #[test]
    fn ignore_book_ply_matches_regardless_of_ply() {
        run(|| {
            // Stored under ply 99; the real position is ply 1.
            let stored = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 99";
            let content = format!("{HEADER}\nsfen {stored}\n7g7f none 50 32 1\n");
            let tmp = TempBook::with(&content, "ignore-ply");
            let book = YaneuraouBook::from_path(&tmp.path, false).unwrap();
            let pos = Position::new_from_sfen(START_SFEN).unwrap();

            let mut strict = opts();
            strict.ignore_book_ply = false;
            assert!(
                book.probe(&pos, &strict, &mut rand::thread_rng()).is_none(),
                "ply mismatch must miss"
            );

            let mut ignore = opts();
            ignore.ignore_book_ply = true;
            assert_eq!(
                book.probe(&pos, &ignore, &mut rand::thread_rng()).unwrap().to_usi_string(),
                "7g7f",
            );
        });
    }

    #[test]
    fn book_depth_limit_filters_shallow_entries() {
        run(|| {
            // Only the shallow move 2g2f (depth 4) and a deep move 7g7f (depth 40) exist.
            let content = format!("{HEADER}\nsfen {START_SFEN}\n2g2f none 0 4 1\n7g7f none 0 40 1\n");
            let tmp = TempBook::with(&content, "depth");
            let book = YaneuraouBook::from_path(&tmp.path, false).unwrap();
            let pos = Position::new_from_sfen(START_SFEN).unwrap();

            let mut deep = opts();
            deep.book_depth_limit = 10; // drops 2g2f
            let mv = book.probe(&pos, &deep, &mut rand::thread_rng()).unwrap();
            assert_eq!(mv.to_usi_string(), "7g7f");

            let mut too_deep = opts();
            too_deep.book_depth_limit = 100; // drops everything
            assert!(book.probe(&pos, &too_deep, &mut rand::thread_rng()).is_none());
        });
    }

    #[test]
    fn book_eval_diff_keeps_only_near_best() {
        run(|| {
            // best value 50 (7g7f) and a far-worse 2g2f (value -100).
            let content = format!("{HEADER}\nsfen {START_SFEN}\n7g7f none 50 32 1\n2g2f none -100 32 1\n");
            let tmp = TempBook::with(&content, "evaldiff");
            let book = YaneuraouBook::from_path(&tmp.path, false).unwrap();
            let pos = Position::new_from_sfen(START_SFEN).unwrap();

            let mut tight = opts();
            tight.book_eval_diff = 10; // keep only within 10 of best → just 7g7f
            for _ in 0..50 {
                let mv = book.probe(&pos, &tight, &mut rand::thread_rng()).unwrap();
                assert_eq!(mv.to_usi_string(), "7g7f", "low-value move must be filtered out");
            }
        });
    }

    #[test]
    fn flipped_book_mirrors_and_maps_move_back() {
        run(|| {
            // Book key is the horizontally mirrored start position; stored move 3g3f is the
            // mirror of the real legal move 7g7f.
            let mirrored = "lnsgkgsnl/1b5r1/ppppppppp/9/9/9/PPPPPPPPP/1R5B1/LNSGKGSNL b - 1";
            let content = format!("{HEADER}\nsfen {mirrored}\n3g3f none 0 1 1\n");
            let tmp = TempBook::with(&content, "flip");
            let book = YaneuraouBook::from_path(&tmp.path, false).unwrap();
            let pos = Position::new_from_sfen(START_SFEN).unwrap();

            let mut no_flip = opts();
            no_flip.flipped_book = false;
            assert!(
                book.probe(&pos, &no_flip, &mut rand::thread_rng()).is_none(),
                "without flip it misses"
            );

            let mut flip = opts();
            flip.flipped_book = true;
            let mv = book.probe(&pos, &flip, &mut rand::thread_rng()).unwrap();
            assert_eq!(mv.to_usi_string(), "7g7f", "mirrored 3g3f maps back to 7g7f");
        });
    }

    // ----- on-the-fly parity (removed under `tournament`) -----

    #[cfg(not(feature = "tournament"))]
    #[test]
    fn on_the_fly_matches_in_memory_candidates() {
        let content = format!("{HEADER}\nsfen {START_SFEN}\n7g7f 3c3d 50 32 100\n2g2f none 40 32 60\n6i7h none 5 8 10\n");
        let tmp = TempBook::with(&content, "parity");
        let mem = YaneuraouBook::from_path(&tmp.path, false).unwrap();
        let otf = YaneuraouBook::from_path(&tmp.path, true).unwrap();

        let mut a: Vec<_> = mem
            .candidates(START_SFEN, false)
            .unwrap()
            .into_iter()
            .map(|e| (e.best, e.value, e.depth, e.num))
            .collect();
        let mut b: Vec<_> = otf
            .candidates(START_SFEN, false)
            .unwrap()
            .into_iter()
            .map(|e| (e.best, e.value, e.depth, e.num))
            .collect();
        a.sort();
        b.sort();
        assert_eq!(a, b, "on-the-fly candidate set must equal the in-memory set");
    }

    #[cfg(not(feature = "tournament"))]
    #[test]
    fn on_the_fly_probe_returns_legal_move() {
        run(|| {
            let content = format!("{HEADER}\nsfen {START_SFEN}\n7g7f none 50 32 1\n");
            let tmp = TempBook::with(&content, "otf-probe");
            let otf = YaneuraouBook::from_path(&tmp.path, true).unwrap();
            let pos = Position::new_from_sfen(START_SFEN).unwrap();
            let mv = otf.probe(&pos, &opts(), &mut rand::thread_rng()).unwrap();
            assert_eq!(mv.to_usi_string(), "7g7f");
        });
    }

    #[test]
    fn second_block_sfen_matches_engine() {
        run(|| {
            // Confirms the fixture's post-7g7f block SFEN equals the engine's to_sfen after 7g7f.
            let mut pos = Position::new_from_sfen(START_SFEN).unwrap();
            let m = Move::new_from_usi_str("7g7f", &pos).unwrap();
            let gives_check = pos.gives_check(m);
            pos.do_move(m, gives_check);

            let book = YaneuraouBook::from_path("test/yaneuraou_book.db", false).unwrap();
            let mv = book.probe(&pos, &opts(), &mut rand::thread_rng());
            assert!(mv.is_some(), "fixture must contain the post-7g7f position: {}", pos.to_sfen());
        });
    }

    /// Dumps a position's candidate moves as sorted `(best, value, depth, num)` tuples, or `None`.
    #[cfg(not(feature = "tournament"))]
    fn dump_candidates(book: &YaneuraouBook, sfen: &str, ignore_ply: bool) -> Option<Vec<(String, i32, i32, u64)>> {
        book.candidates(sfen, ignore_ply).map(|v| {
            let mut t: Vec<_> = v.into_iter().map(|e| (e.best, e.value, e.depth, e.num)).collect();
            t.sort();
            t
        })
    }

    #[cfg(not(feature = "tournament"))]
    #[test]
    fn on_the_fly_binary_search_matches_in_memory_across_blocks() {
        // Four positions in ascending SFEN order; the on-disk binary search must find any of them,
        // and miss for keys before the first, between blocks, and after the last.
        let content = format!(
            "{HEADER}\n\
             sfen k1/9/9/9/9/9/9/9/9 b - 1\n7g7f none 10 20 1\n\
             sfen k2/9/9/9/9/9/9/9/9 b - 1\n2g2f none 20 20 2\n6i7h none 5 8 3\n\
             sfen k3/9/9/9/9/9/9/9/9 b - 1\n3g3f none 30 20 4\n\
             sfen k4/9/9/9/9/9/9/9/9 b - 1\n5g5f none 40 20 5\n"
        );
        let tmp = TempBook::with(&content, "otf-multi");
        let mem = YaneuraouBook::from_path(&tmp.path, false).unwrap();
        let otf = YaneuraouBook::from_path(&tmp.path, true).unwrap();

        for query in [
            "k1/9/9/9/9/9/9/9/9 b - 1",  // first block
            "k2/9/9/9/9/9/9/9/9 b - 1",  // middle block (two moves)
            "k4/9/9/9/9/9/9/9/9 b - 1",  // last block
            "k0/9/9/9/9/9/9/9/9 b - 1",  // before the first key → miss
            "k25/9/9/9/9/9/9/9/9 b - 1", // strictly between k2 and k3 → miss
            "k9/9/9/9/9/9/9/9/9 b - 1",  // after the last key → miss
        ] {
            assert_eq!(
                dump_candidates(&mem, query, false),
                dump_candidates(&otf, query, false),
                "on-the-fly disagrees with in-memory for {query}",
            );
        }
    }

    #[cfg(not(feature = "tournament"))]
    #[test]
    fn on_the_fly_ignore_book_ply_gathers_adjacent_ply_blocks() {
        // Same ply-less key at ply 1 and ply 7, adjacent and ascending in the file.
        let content = format!(
            "{HEADER}\n\
             sfen 9/9/9/9/9/9/9/9/9 b - 1\n7g7f none 10 20 1\n\
             sfen 9/9/9/9/9/9/9/9/9 b - 7\n2g2f none 20 20 2\n"
        );
        let tmp = TempBook::with(&content, "otf-ply");
        let mem = YaneuraouBook::from_path(&tmp.path, false).unwrap();
        let otf = YaneuraouBook::from_path(&tmp.path, true).unwrap();
        let query = "9/9/9/9/9/9/9/9/9 b - 1";

        // Strict ply matching sees only the ply-1 block; ignoring ply gathers both adjacent blocks.
        assert_eq!(dump_candidates(&otf, query, false), dump_candidates(&mem, query, false));
        assert_eq!(dump_candidates(&otf, query, false).unwrap().len(), 1);
        assert_eq!(dump_candidates(&otf, query, true), dump_candidates(&mem, query, true));
        assert_eq!(dump_candidates(&otf, query, true).unwrap().len(), 2);
    }

    #[cfg(not(feature = "tournament"))]
    #[test]
    fn on_the_fly_probes_first_and_last_fixture_blocks() {
        run(|| {
            let otf = YaneuraouBook::from_path("test/yaneuraou_book.db", true).unwrap();
            // The initial position is the last block in SFEN order.
            let start = Position::new_from_sfen(START_SFEN).unwrap();
            assert!(otf.probe(&start, &opts(), &mut rand::thread_rng()).is_some());
            // The post-7g7f position is the first block in SFEN order.
            let mut pos = Position::new_from_sfen(START_SFEN).unwrap();
            let m = Move::new_from_usi_str("7g7f", &pos).unwrap();
            let gives_check = pos.gives_check(m);
            pos.do_move(m, gives_check);
            assert!(otf.probe(&pos, &opts(), &mut rand::thread_rng()).is_some());
        });
    }

    #[cfg(not(feature = "tournament"))]
    #[test]
    fn on_the_fly_missing_header_errors() {
        let tmp = TempBook::with("sfen 9/9/9/9/9/9/9/9/9 b - 1\n7g7f none 0 1 1\n", "otf-noheader");
        let err = YaneuraouBook::from_path(&tmp.path, true).unwrap_err();
        assert!(matches!(err, BookError::MissingHeader), "got {err:?}");
    }

    // ----- mirroring units -----

    #[test]
    fn mirror_usi_move_units() {
        assert_eq!(mirror_usi_move("7g7f").as_deref(), Some("3g3f"));
        assert_eq!(mirror_usi_move("2g2f").as_deref(), Some("8g8f"));
        assert_eq!(mirror_usi_move("8h2b+").as_deref(), Some("2h8b+"));
        assert_eq!(mirror_usi_move("P*5e").as_deref(), Some("P*5e"));
        assert_eq!(mirror_usi_move("P*7f").as_deref(), Some("P*3f"));
    }

    #[test]
    fn mirror_board_is_an_involution_on_start() {
        let board = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL";
        let once = mirror_board(board).unwrap();
        assert_eq!(once, "lnsgkgsnl/1b5r1/ppppppppp/9/9/9/PPPPPPPPP/1R5B1/LNSGKGSNL");
        assert_eq!(mirror_board(&once).unwrap(), board, "mirroring twice is the identity");
    }
}
