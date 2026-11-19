//! What a worker of a parallel search takes from the state the other workers
//! write, recorded per root iteration and written to a file at the end of
//! each worker's search.
//!
//! Three records, kept per worker:
//!
//! - **Table use** ([`TtUse`]). Every interior node whose probe found its
//!   position, by who wrote the entry (as the writer side table of
//!   [`crate::dupstats`] knows it), by the entry's depth against the node's
//!   remaining depth, by the entry's bound, and by whether the entry cut the
//!   node off. When a reference file of a lone search is loaded, also whether
//!   the entry's move is the move the lone search found in the same position in
//!   the same iteration.
//! - **Aspiration searches**: per root search, its window, the depth it was
//!   searched at, its value and its nodes.
//! - **Root moves**: per root move searched, the aspiration search it was in,
//!   its place in the move loop, the window it was searched with, its value,
//!   its nodes and whether it became the best move.
//!
//! Three environment variables steer it, each read once: `YORKIE_DUPSTATS_LOG`
//! names the file the records are appended to (nothing is written without
//! it); `YORKIE_DUPSTATS_REF_OUT` names a file the main worker writes, for
//! every interior position it visited in each iteration, the move its first
//! probe found there; `YORKIE_DUPSTATS_REF_IN` names such a file to compare
//! against. The module exists only in a `dupstats` build, and nothing in it is
//! read by the search.

use std::collections::HashMap;
use std::io::Write as _;
use std::sync::{Mutex, OnceLock};

use yorkie_state::text::{atoi_u16, atoi_u64};
use yorkie_state::{Move, TextWriter, write_usi_move};
use yorkie_storage::cut::{Cut, on_main};
use yorkie_storage::{Bound, TTData, VALUE_NONE, Value};

use crate::dupstats::WriterId;

/// The entry `data` as the main worker reads it under
/// [`Cut::MainNoPv`]: with its PV flag clear.
#[inline]
pub fn mask_pv(data: TTData) -> TTData {
    if on_main(Cut::MainNoPv) {
        TTData {
            is_pv: false,
            ..data
        }
    } else {
        data
    }
}

/// The entry's value `value` as the main worker's table cutoff reads it under
/// [`Cut::MainNoCut`]: none, so the cutoff never holds.
#[inline]
pub fn mask_cut_value(value: Value) -> Value {
    if on_main(Cut::MainNoCut) {
        VALUE_NONE
    } else {
        value
    }
}

/// What an interior node of the main worker reads, past the table cutoff, of
/// its entry's move `mv` (and whether that move is a capture, `capture`), its
/// value `value` and its stored data `data`, under [`Cut::MainNoMove`],
/// [`Cut::MainNoValue`] and [`Cut::MainNoEval`].
#[inline]
pub fn mask_after_cut(
    mv: Option<Move>,
    capture: bool,
    value: Value,
    data: TTData,
) -> (Option<Move>, bool, Value, TTData) {
    let (mv, capture) = if on_main(Cut::MainNoMove) {
        (None, false)
    } else {
        (mv, capture)
    };
    let value = if on_main(Cut::MainNoValue) {
        VALUE_NONE
    } else {
        value
    };
    let data = if on_main(Cut::MainNoEval) {
        TTData {
            eval: VALUE_NONE,
            ..data
        }
    } else {
        data
    };
    (mv, capture, value, data)
}

/// The move the main worker's move picker is given at a node whose entry move
/// is `mv`, under [`Cut::MainNoOrder`]; the root's move is never withheld.
#[inline]
pub fn mask_order(mv: Option<Move>, root: bool) -> Option<Move> {
    if !root && on_main(Cut::MainNoOrder) {
        None
    } else {
        mv
    }
}

/// Root iterations a record keeps apart; a deeper one shares the last slot.
pub const ITERATIONS: usize = 64;

/// Who wrote an entry a node used.
const WRITERS: [&str; 5] = [
    "self_iter",
    "self_earlier",
    "other",
    "earlier_search",
    "unknown",
];
/// The entry's stored depth against the node's remaining depth.
const RELS: [&str; 3] = ["deeper", "equal", "shallower"];
/// The entry's bound.
const BOUNDS: [&str; 4] = ["none", "upper", "lower", "exact"];
/// How the entry's move compares with the lone search's in the same position.
const REFS: [&str; 6] = [
    "absent",
    "both_none",
    "same",
    "differs",
    "lone_none",
    "main_none",
];

const USE_CELLS: usize = WRITERS.len() * RELS.len() * BOUNDS.len() * 2;

/// The table-use counts of one worker.
pub struct TtUse {
    counts: Box<[[u64; USE_CELLS]; ITERATIONS]>,
    refs: Box<[[u64; REFS.len()]; ITERATIONS]>,
    /// The lone search's record being written, when this worker writes one.
    recording: Option<HashMap<(u8, u64), u16>>,
}

fn iteration_slot(iteration: i32) -> usize {
    (iteration.max(0) as usize).min(ITERATIONS - 1)
}

/// What a probe that found its position names about the entry.
pub struct Use {
    /// The root iteration.
    pub iteration: i32,
    /// The node's remaining depth.
    pub depth: i32,
    /// The entry's stored depth.
    pub stored_depth: i32,
    /// The entry's bound.
    pub bound: Bound,
    /// Whether the entry cut the node off.
    pub cut: bool,
    /// Who wrote the entry, as the writer side table knows it.
    pub writer: Option<WriterId>,
    /// The worker using the entry, in the search running now.
    pub me: WriterId,
}

impl TtUse {
    /// Empty counts; a worker given index `0` while a reference file is to be
    /// written records its probes for it.
    pub fn new() -> Self {
        Self {
            counts: Box::new([[0; USE_CELLS]; ITERATIONS]),
            refs: Box::new([[0; REFS.len()]; ITERATIONS]),
            recording: None,
        }
    }

    /// Clear the counts for a new search by worker `worker`.
    pub fn reset(&mut self, worker: u8) {
        self.counts.iter_mut().for_each(|r| r.fill(0));
        self.refs.iter_mut().for_each(|r| r.fill(0));
        self.recording = (worker == 0 && env().ref_out.is_some()).then(HashMap::new);
    }

    /// Count one use of an entry.
    pub fn count(&mut self, u: &Use) {
        let writer = match u.writer {
            None => 4,
            Some(w) if w.generation != u.me.generation => 3,
            Some(w) if w.worker != u.me.worker => 2,
            Some(w) if w.iteration == u.me.iteration => 0,
            Some(_) => 1,
        };
        let rel = match u.stored_depth.cmp(&u.depth) {
            std::cmp::Ordering::Greater => 0,
            std::cmp::Ordering::Equal => 1,
            std::cmp::Ordering::Less => 2,
        };
        let bound = match u.bound {
            Bound::None => 0,
            Bound::Upper => 1,
            Bound::Lower => 2,
            Bound::Exact => 3,
        };
        let cell = ((writer * RELS.len() + rel) * BOUNDS.len() + bound) * 2 + usize::from(u.cut);
        self.counts[iteration_slot(u.iteration)][cell] += 1;
    }

    /// Note the move `move16` (`0` for none) the probe of the position `key`
    /// found in `iteration`, against the lone search's record and into the one
    /// being written. `hit` is whether the probe found the position.
    pub fn note_move(&mut self, iteration: i32, key: u64, hit: bool, move16: u16) {
        let it = iteration_slot(iteration) as u8;
        if let Some(rec) = &mut self.recording {
            rec.entry((it, key)).or_insert(move16);
        }
        if !hit {
            return;
        }
        if let Some(lone) = reference() {
            let kind = match lone.get(&(it, key)) {
                None => 0,
                Some(0) if move16 == 0 => 1,
                Some(&m) if m == move16 => 2,
                Some(0) => 4,
                Some(_) if move16 == 0 => 5,
                Some(_) => 3,
            };
            self.refs[it as usize][kind] += 1;
        }
    }
}

impl Default for TtUse {
    fn default() -> Self {
        Self::new()
    }
}

/// How one aspiration search ended.
#[derive(Clone, Copy)]
pub enum AspResult {
    /// At or below alpha.
    FailLow,
    /// At or above beta.
    FailHigh,
    /// Inside the window.
    Exact,
    /// Stopped before it finished.
    Stopped,
}

impl AspResult {
    fn name(self) -> &'static [u8] {
        match self {
            AspResult::FailLow => b"fail_low",
            AspResult::FailHigh => b"fail_high",
            AspResult::Exact => b"exact",
            AspResult::Stopped => b"stopped",
        }
    }
}

/// One aspiration search at the root.
pub struct Asp {
    /// The iteration.
    pub depth: i32,
    /// The search's place among the iteration's aspiration searches, from 0.
    pub pass: u32,
    /// The depth it searched to.
    pub searched_depth: i32,
    /// The window.
    pub alpha: i32,
    /// The window.
    pub beta: i32,
    /// The value it returned.
    pub value: i32,
    /// How it ended.
    pub result: AspResult,
    /// Its nodes.
    pub nodes: u64,
}

/// One root move searched.
pub struct RootSearch {
    /// The iteration.
    pub depth: i32,
    /// The aspiration search it was in.
    pub pass: u32,
    /// Its place in the root move loop, from 1.
    pub move_count: i32,
    /// The move.
    pub mv: Move,
    /// The window at the root when the move was searched.
    pub alpha: i32,
    /// The window at the root when the move was searched.
    pub beta: i32,
    /// The value its last search returned.
    pub value: i32,
    /// The nodes its searches took.
    pub nodes: u64,
    /// Whether it became the root's best move.
    pub best: bool,
}

/// Everything one worker records, written out at the end of its search.
#[derive(Default)]
pub struct Record {
    /// The table-use counts.
    pub tt: TtUse,
    /// Every aspiration search.
    pub asp: Vec<Asp>,
    /// Every root move searched.
    pub root: Vec<RootSearch>,
    /// The aspiration search the root is in now.
    pub pass: u32,
}

impl Record {
    /// Clear for a new search by worker `worker`.
    pub fn reset(&mut self, worker: u8) {
        self.tt.reset(worker);
        self.asp.clear();
        self.root.clear();
        self.pass = 0;
    }

    /// Append this worker's records to the log file, if one is named, and
    /// write the reference file, if this worker records one. `nodes` is the
    /// worker's node count for the whole search.
    pub fn flush(&mut self, worker: u8, nodes: u64) {
        let env = env();
        if let (Some(path), Some(rec)) = (&env.ref_out, self.tt.recording.take()) {
            let mut text = Vec::with_capacity(rec.len() * 24);
            for ((it, key), m) in rec {
                writeln!(text, "{it}\t{key:x}\t{m}").expect("a Vec takes every write");
            }
            std::fs::write(path, text).expect("the reference file can be written");
        }
        let Some(path) = &env.log else {
            return;
        };
        let mut out = Vec::new();
        let w = worker;
        writeln!(out, "worker\t{w}\t{nodes}").expect("a Vec takes every write");
        for (it, row) in self.tt.counts.iter().enumerate() {
            for (cell, &n) in row.iter().enumerate() {
                if n == 0 {
                    continue;
                }
                let cut = cell % 2;
                let b = (cell / 2) % BOUNDS.len();
                let r = (cell / 2 / BOUNDS.len()) % RELS.len();
                let wr = cell / 2 / BOUNDS.len() / RELS.len();
                writeln!(
                    out,
                    "ttuse\t{w}\t{it}\t{}\t{}\t{}\t{cut}\t{n}",
                    WRITERS[wr], RELS[r], BOUNDS[b]
                )
                .expect("a Vec takes every write");
            }
        }
        for (it, row) in self.tt.refs.iter().enumerate() {
            for (kind, &n) in row.iter().enumerate() {
                if n != 0 {
                    writeln!(out, "ttref\t{w}\t{it}\t{}\t{n}", REFS[kind])
                        .expect("a Vec takes every write");
                }
            }
        }
        for a in &self.asp {
            write!(
                out,
                "asp\t{w}\t{}\t{}\t{}\t{}\t{}\t{}\t",
                a.depth, a.pass, a.searched_depth, a.alpha, a.beta, a.value
            )
            .expect("a Vec takes every write");
            out.extend_from_slice(a.result.name());
            writeln!(out, "\t{}", a.nodes).expect("a Vec takes every write");
        }
        let mut buf = [0u8; 16];
        for r in &self.root {
            let mut mv = TextWriter::new(&mut buf);
            write_usi_move(r.mv, &mut mv);
            write!(
                out,
                "root\t{w}\t{}\t{}\t{}\t",
                r.depth, r.pass, r.move_count
            )
            .expect("a Vec takes every write");
            out.extend_from_slice(mv.as_bytes());
            writeln!(
                out,
                "\t{}\t{}\t{}\t{}\t{}",
                r.alpha,
                r.beta,
                r.value,
                r.nodes,
                u8::from(r.best)
            )
            .expect("a Vec takes every write");
        }
        let _guard = LOG_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .and_then(|mut f| f.write_all(&out))
            .expect("the record file can be appended to");
    }
}

static LOG_LOCK: Mutex<()> = Mutex::new(());

struct Env {
    log: Option<std::path::PathBuf>,
    ref_out: Option<std::path::PathBuf>,
    ref_in: Option<std::path::PathBuf>,
}

fn env() -> &'static Env {
    static ENV: OnceLock<Env> = OnceLock::new();
    ENV.get_or_init(|| {
        let path = |name| {
            std::env::var_os(name)
                .filter(|v| !v.is_empty())
                .map(Into::into)
        };
        Env {
            log: path("YORKIE_DUPSTATS_LOG"),
            ref_out: path("YORKIE_DUPSTATS_REF_OUT"),
            ref_in: path("YORKIE_DUPSTATS_REF_IN"),
        }
    })
}

/// The lone search's record, when `YORKIE_DUPSTATS_REF_IN` names one.
fn reference() -> Option<&'static HashMap<(u8, u64), u16>> {
    static REF: OnceLock<Option<HashMap<(u8, u64), u16>>> = OnceLock::new();
    REF.get_or_init(|| {
        let path = env().ref_in.as_ref()?;
        let text = std::fs::read(path).expect("the reference file can be read");
        Some(
            text.split(|&b| b == b'\n')
                .map(|line| line.strip_suffix(b"\r").unwrap_or(line))
                .filter(|line| !line.is_empty())
                .map(parse_ref_line)
                .collect(),
        )
    })
    .as_ref()
}

fn parse_ref_line(line: &[u8]) -> ((u8, u64), u16) {
    let mut f = line.split(|&b| b == b'\t');
    let mut next = || f.next().expect("a reference line has three fields");
    let it = atoi_u64(next())
        .and_then(|v| u8::try_from(v).ok())
        .expect("an iteration");
    let key = parse_hex(next()).expect("a key");
    let m = atoi_u16(next()).expect("a move");
    ((it, key), m)
}

/// The unsigned hexadecimal `text` spells, or `None` when it is empty, carries
/// a byte that is not a hex digit, or names a value past `u64`.
fn parse_hex(text: &[u8]) -> Option<u64> {
    if text.is_empty() {
        return None;
    }
    let mut value: u64 = 0;
    for &byte in text {
        let digit = match byte {
            b'0'..=b'9' => byte - b'0',
            b'a'..=b'f' => byte - b'a' + 10,
            b'A'..=b'F' => byte - b'A' + 10,
            _ => return None,
        };
        value = value.checked_mul(16)?.checked_add(u64::from(digit))?;
    }
    Some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn me() -> WriterId {
        WriterId {
            generation: 3,
            worker: 0,
            iteration: 9,
        }
    }

    fn total(t: &TtUse) -> u64 {
        t.counts.iter().flatten().sum()
    }

    #[test]
    fn a_use_lands_in_one_cell_of_its_iteration() {
        let mut t = TtUse::new();
        let other = WriterId { worker: 1, ..me() };
        t.count(&Use {
            iteration: 9,
            depth: 5,
            stored_depth: 7,
            bound: Bound::Lower,
            cut: true,
            writer: Some(other),
            me: me(),
        });
        assert_eq!(total(&t), 1);
        let cell = ((2 * RELS.len()) * BOUNDS.len() + 2) * 2 + 1;
        assert_eq!(t.counts[9][cell], 1);
    }

    #[test]
    fn a_reference_line_reads_back_as_written() {
        assert_eq!(
            parse_ref_line(b"12\tdeadbeef\t513"),
            ((12, 0xdead_beef), 513)
        );
    }
}
