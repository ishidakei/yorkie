//! Transposition table, ported from the reference and specialised to the
//! engine's default build configuration.
//!
//! The reference is heavily `#define`-parameterised; this mirrors its default
//! build: a plain 64-bit position key, and clusters of three 10-byte entries
//! whose stored key fragment is the position key's low 16 bits. Those choices
//! are load-bearing for search-node parity — the cluster size and the
//! `clusterCount = mb·2²⁰ / sizeof(Cluster)` arithmetic decide which positions
//! collide, and the in-cluster replacement policy decides which entry survives,
//! both of which feed into how many nodes qsearch visits.
//!
//! # The `tt-entry16` feature
//!
//! The `tt-entry16` cargo feature, off by default, is the one sanctioned
//! departure from that layout: it widens the entry to 16 bytes by storing the
//! whole 64-bit key in place of the 16-bit fragment.
//!
//! | | default | `tt-entry16` |
//! |---|---|---|
//! | `TteKey` (the stored key) | `u16` — the hash's low 16 bits | `u64` — the whole hash |
//! | `size_of::<TTEntry>()` | 10 | 16 |
//! | `CLUSTER_SIZE` | 3 | 2 |
//! | `size_of::<Cluster>()` | 32 (`3 × 10 + 2` padding) | 32 (`2 × 16`, no padding) |
//!
//! Position identity is then full 64-bit equality rather than a 16-bit compare,
//! so a hit is exact instead of carrying a ~1/65536 per-entry false-hit rate.
//! The payload fields are byte-for-byte the same, the cluster stays 32 bytes,
//! and the same byte budget therefore holds two thirds as many entries.
//!
//! **A `tt-entry16` build's search output is tied to neither the default
//! build's nor the reference's.** Both exact identity and the smaller entry
//! count move which probes hit, and nothing here promises otherwise.
//!
//! # Layering
//!
//! The Storage layer must not depend on the State layer, so this module speaks
//! in primitives: the position key is a `u64`, the side to move a `u8`, and the
//! best move its low-16-bit fragment. Widening that fragment back to a full
//! move, and validating it against the actual position, belongs to the caller.
//!
//! # One table, sized before the binary exists
//!
//! The table is one array of [`CLUSTER_COUNT`] clusters in the binary's BSS,
//! its length folded from the `usi_hash` setting the binary was built with.
//! Nothing allocates it, nothing resizes it, and its length is a literal at
//! every probe. [`TranspositionTable::shared`] hands out the `&'static` handle
//! every worker holds; measuring with another table size means building with
//! another config.
//!
//! Its storage is a zero-filled symbol declared to the assembler, not a Rust
//! `static` with an array initialiser, because rustc const-evaluates such an
//! initialiser by materialising every one of its bytes: minutes of compile time
//! on every build at a gigabyte, and out of reach of any machine at the sizes a
//! large-memory host is configured for. An assembler's `.zero` states the size
//! without producing the bytes, and a `nobits` section keeps them out of the
//! file as well. A `#[link_section]` attribute on an initialised `static` does
//! not help, since the initialiser is still evaluated. The layout a `static`
//! declaration would guarantee on its own is asserted here against the typed
//! view of the symbol instead, at compile time and in the tests.
//!
//! Only the cluster array needs that treatment; the generation counter is a
//! `static` of its own, so the symbol holds clusters and nothing else.
//!
//! The array is [`TT_ALIGN`]-aligned, so it starts on a huge-page boundary and
//! a `MADV_HUGEPAGE` hint over it can be honoured. BSS is zero at process
//! start and an all-zero cluster is a valid empty one, so the table is ready
//! before anything touches it.
//!
//! Miri interprets Rust and cannot resolve a symbol that only an assembler
//! defines, so a miri build gets the same table as a Rust `static` array. Its
//! initialiser is const-evaluated byte by byte and every access to it is
//! interpreted, both of which scale with the cluster count, so that build fixes
//! [`CLUSTER_COUNT`] at a small count of its own instead of folding it from
//! `usi_hash`.
//!
//! # Threading
//!
//! The read/write path takes `&self`, so every worker hits the one table at
//! once. Each entry field is an atomic accessed with [`Ordering::Relaxed`] —
//! the Rust-sound equivalent of the reference's racy, lock-free `TTEntry`. On
//! x86-64 that lowers to a plain `MOV`.
//!
//! **An entry can tear.** Relaxed atomics make each *field* access indivisible,
//! but an entry is six fields with no cross-field atomicity, so a concurrent
//! write can land between this thread's field reads and `TTEntry::read` return
//! an old key fragment beside a new value. The reference tolerates exactly this:
//! a mismatched fragment reads back as a miss or a wrong-position hit, and the
//! caller validates every TT move against the actual position anyway.
//!
//! [`TranspositionTable::clear`] writes through the same shared reference, one
//! relaxed store per field, so it is sound whatever else is running. It is
//! still meant for the gap between games, when nothing else is: emptying
//! entries a search is reading costs it nodes, not correctness.
//!
//! # The path-dependence mark (`verbose3`)
//!
//! A stored value can depend on the move history by which its position was
//! reached, because a repetition judgement reads that history. At `verbose3`
//! each entry carries one bit recording that its value and best move were
//! derived through such a judgement, so an outside inspector can tell those
//! entries apart. Nothing in the table reads the bit: it is not part of an
//! entry's identity, it does not enter the replacement priority, and a probe
//! hits or misses regardless of it.
//!
//! The two layouts keep it in different places, because only one of them has a
//! spare byte:
//!
//! | | default | `tt-entry16` |
//! |---|---|---|
//! | where the bit lives | the cluster's two padding bytes, as an `AtomicU16` whose bit `i` belongs to `entry[i]` | bit 0 of the stored key |
//! | what it costs | nothing — the bytes were padding | the key drops to 63 bits, so the false-hit rate is 2⁻⁶³ |
//!
//! In the default layout the bit and the entry are separate words, so a reader
//! can briefly see a new entry beside the old bit under contention — the same
//! tearing the six entry fields already permit, and acceptable for a
//! diagnostic. Under `tt-entry16` the bit is written with the key, in one
//! store.

use std::mem::{offset_of, size_of};
#[cfg(feature = "tt-entry16")]
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicI16, AtomicU8, AtomicU16, Ordering};

/// The single memory ordering used for every entry field access.
const REL: Ordering = Ordering::Relaxed;

/// Search value, mirroring the reference's `int` `Value`. Truncated to `i16`
/// in an entry, where every real value fits.
pub type Value = i32;

/// Search depth, mirroring the reference's `int` `Depth`. Stored in an entry
/// offset by [`DEPTH_NONE`] and truncated to `u8`.
pub type Depth = i32;

/// `DEPTH_NONE`. Entries store `depth8 = depth − DEPTH_NONE`, so an all-zero
/// entry reads back as `DEPTH_NONE` and counts as unoccupied.
pub const DEPTH_NONE: Depth = -3;

/// `VALUE_NONE`, the sentinel returned for a miss.
pub const VALUE_NONE: Value = 32002;

// The `genBound8` bit layout: `generation (5) | bound (2) << 5 | pv (1) << 7`.
const GENERATION_BITS: u8 = 5;
const GENERATION_MASK: u8 = (1 << GENERATION_BITS) - 1;
const BOUND_SHIFT: u8 = GENERATION_BITS;
const BOUND_MASK: u8 = 0b11 << BOUND_SHIFT;
const PV_SHIFT: u8 = BOUND_SHIFT + 2;
const PV_MASK: u8 = 1 << PV_SHIFT;

// Everything the two entry layouts differ in is declared here as a
// `cfg`-selected type or constant, so the rest of the module reads `TteKey` and
// `CLUSTER_SIZE` and carries no `cfg` of its own.

/// What position identity is checked on: the hash's low 16 bits, so a match is
/// a 16-bit compare with a ~1/65536 per-entry false-hit rate.
#[cfg(not(feature = "tt-entry16"))]
type TteKey = u16;
/// The atomic wrapper for [`TteKey`]; same size and alignment as the plain type.
#[cfg(not(feature = "tt-entry16"))]
type AtomicTteKey = AtomicU16;
/// Narrow `key` to what an entry stores — the low 16 bits.
#[cfg(not(feature = "tt-entry16"))]
#[inline]
fn tte_key(key: u64) -> TteKey {
    key as TteKey
}
/// A stored key widened back to 64 bits, for [`TranspositionTable::checksum`],
/// and compiled only where that is.
#[cfg(all(not(feature = "tt-entry16"), feature = "verbose3"))]
#[inline]
fn key_bits(k: TteKey) -> u64 {
    k as u64
}
/// Number of entries per cluster (`TT_CLUSTER_SIZE == 3`).
#[cfg(not(feature = "tt-entry16"))]
const CLUSTER_SIZE: usize = 3;
/// Trailing bytes that pad a [`Cluster`] out to 32 (`3 × 10 + 2`).
#[cfg(not(feature = "tt-entry16"))]
const CLUSTER_PADDING: usize = 2;

/// What position identity is checked on: under `tt-entry16` the whole 64-bit
/// hash, so a match is exact.
#[cfg(feature = "tt-entry16")]
type TteKey = u64;
/// The atomic wrapper for [`TteKey`]; same size and alignment as the plain type.
#[cfg(feature = "tt-entry16")]
type AtomicTteKey = AtomicU64;
/// Narrow `key` to what an entry stores. That is the whole key, except at
/// `verbose3`, where bit 0 of the stored key is the path-dependence mark and
/// the identity is the remaining 63 bits.
#[cfg(feature = "tt-entry16")]
#[inline]
fn tte_key(key: u64) -> TteKey {
    #[cfg(feature = "verbose3")]
    {
        key & !1
    }
    #[cfg(not(feature = "verbose3"))]
    {
        key
    }
}
/// A stored key widened back to 64 bits — already that wide here. Compiled only
/// where [`TranspositionTable::checksum`] is.
#[cfg(all(feature = "tt-entry16", feature = "verbose3"))]
#[inline]
fn key_bits(k: TteKey) -> u64 {
    k
}
/// Entries per cluster: a 16-byte entry fits twice in the 32-byte cluster.
#[cfg(feature = "tt-entry16")]
const CLUSTER_SIZE: usize = 2;
/// None needed — `2 × 16` is exactly 32.
#[cfg(feature = "tt-entry16")]
const CLUSTER_PADDING: usize = 0;

/// Whether a stored key identifies the same position as `k`.
///
/// Plain equality, except where the stored key's bit 0 is the path-dependence
/// mark rather than part of the hash, in which case both sides are compared
/// with it cleared.
#[inline]
fn key_matches(stored: TteKey, k: TteKey) -> bool {
    #[cfg(all(feature = "tt-entry16", feature = "verbose3"))]
    {
        stored & !1 == k & !1
    }
    #[cfg(not(all(feature = "tt-entry16", feature = "verbose3")))]
    {
        stored == k
    }
}

/// Bound type of a stored value. The discriminants are load-bearing:
/// `Exact == Upper | Lower`, and the value packs into `genBound8`.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Bound {
    /// No bound — used when only a move / static eval is stored.
    None = 0,
    /// Upper bound: the true value is `≤` the stored value (fail-low).
    Upper = 1,
    /// Lower bound: the true value is `≥` the stored value (fail-high).
    Lower = 2,
    /// Exact value (PV node, non-mate).
    Exact = 3,
}

impl Bound {
    /// Recover a `Bound` from its 2-bit encoding. Every value `0..=3` is valid.
    #[inline]
    fn from_u8(v: u8) -> Bound {
        match v & 0b11 {
            0 => Bound::None,
            1 => Bound::Upper,
            2 => Bound::Lower,
            _ => Bound::Exact,
        }
    }
}

/// A decoded copy of an entry's payload — the reference's `TTData`. By value:
/// nothing here borrows the table.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TTData {
    /// Best move for this position, as a 16-bit fragment (`0` = none).
    pub move16: u16,
    /// Search value returned at this node.
    pub value: Value,
    /// Static / qsearch evaluation at this node.
    pub eval: Value,
    /// Depth the value was searched to.
    pub depth: Depth,
    /// Bound type of `value`.
    pub bound: Bound,
    /// Whether this was a PV node.
    pub is_pv: bool,
    /// Whether [`Self::value`] and [`Self::move16`] were derived through at
    /// least one repetition judgement on the path the search took to this
    /// position. A sufficient sign of path dependence, not an exact one: an
    /// unmarked value may still differ on another path, and a marked one may
    /// not.
    #[cfg(feature = "verbose3")]
    pub path_dep: bool,
}

impl TTData {
    /// The miss sentinel a failed [`TranspositionTable::probe`] returns.
    #[inline]
    fn none() -> TTData {
        TTData {
            move16: 0,
            value: VALUE_NONE,
            eval: VALUE_NONE,
            depth: DEPTH_NONE,
            bound: Bound::None,
            is_pv: false,
            #[cfg(feature = "verbose3")]
            path_dep: false,
        }
    }
}

/// This entry's age relative to `curr_generation` (`TTEntry::relative_age`).
///
/// Generations count like clock hours, so `0 − 1 == 31`. Wrapping subtraction
/// then masking to 5 bits gives that borrow regardless of the pv and bound bits
/// packed alongside.
#[inline]
fn relative_age(gen_bound8: u8, curr_generation: u8) -> u8 {
    curr_generation.wrapping_sub(gen_bound8) & GENERATION_MASK
}

/// A single transposition-table entry — 10 bytes by default, 16 under
/// `tt-entry16` — laid out field-for-field like the reference's `TTEntry`.
/// `#[repr(C)]` pins the layout so a [`Cluster`] is exactly 32 bytes and the
/// `clusterCount` arithmetic matches the reference. Every field is an atomic
/// of the same size and alignment as its plain counterpart, so the entry can be
/// shared across threads without changing its bytes.
#[repr(C)]
struct TTEntry {
    /// The stored position key: the hash's low 16 bits (`TTE_KEY_TYPE`) by
    /// default, the whole 64-bit hash under `tt-entry16`.
    key: AtomicTteKey,
    /// `depth − DEPTH_NONE`; `0` means unoccupied.
    depth8: AtomicU8,
    /// Packed `generation | bound << 5 | pv << 7`.
    gen_bound8: AtomicU8,
    /// Best move fragment (`Move16`).
    move16: AtomicU16,
    /// Search value.
    value16: AtomicI16,
    /// Static / qsearch eval.
    eval16: AtomicI16,
}

impl TTEntry {
    /// An unoccupied entry. A `const fn` rather than a `Default` impl, because
    /// the miri build's table is a `static` whose initialiser is a constant
    /// expression and `Default::default` is not one. Every other build states
    /// the table's zeros to the assembler and needs no initialiser at all.
    #[cfg(miri)]
    const fn empty() -> TTEntry {
        TTEntry {
            key: AtomicTteKey::new(0),
            depth8: AtomicU8::new(0),
            gen_bound8: AtomicU8::new(0),
            move16: AtomicU16::new(0),
            value16: AtomicI16::new(0),
            eval16: AtomicI16::new(0),
        }
    }

    /// Decode the packed bitfields into external types (`TTEntry::read`). The
    /// six loads are independent, so under contention they can straddle a
    /// concurrent write and yield a torn payload.
    #[inline]
    fn read(&self) -> TTData {
        let gen_bound8 = self.gen_bound8.load(REL);
        TTData {
            move16: self.move16.load(REL),
            value: self.value16.load(REL) as Value,
            eval: self.eval16.load(REL) as Value,
            depth: DEPTH_NONE + self.depth8.load(REL) as Depth,
            bound: Bound::from_u8((gen_bound8 & BOUND_MASK) >> BOUND_SHIFT),
            is_pv: (gen_bound8 & PV_MASK) != 0,
            // The mark is not an entry field; [`Cluster::read`] fills it in
            // from wherever the layout keeps it.
            #[cfg(feature = "verbose3")]
            path_dep: false,
        }
    }

    /// `TTEntry::is_occupied`: the external depth is not `DEPTH_NONE`.
    #[inline]
    fn is_occupied(&self) -> bool {
        self.depth8.load(REL) != 0
    }

    /// Replacement priority used by `probe`: `depth8 − 8 · relative_age`, lower
    /// being more replaceable. In `i32` to match the reference's `int`
    /// promotion, since the subtraction can go negative.
    #[inline]
    fn replace_priority(&self, curr_generation: u8) -> i32 {
        let depth8 = self.depth8.load(REL) as i32;
        let gen_bound8 = self.gen_bound8.load(REL);
        depth8 - 8 * relative_age(gen_bound8, curr_generation) as i32
    }

    /// Zero every field, one relaxed store each — the same shared reference
    /// every other write goes through, since the table is a `static` and no
    /// caller can hold it exclusively.
    #[inline]
    fn reset(&self) {
        self.key.store(0, REL);
        self.depth8.store(0, REL);
        self.gen_bound8.store(0, REL);
        self.move16.store(0, REL);
        self.value16.store(0, REL);
        self.eval16.store(0, REL);
    }
}

/// A cluster of [`CLUSTER_SIZE`] entries, padded to a round 32 bytes. Entries
/// in one cluster share a hash slot; a collision spills into the following
/// entries and is resolved by the replacement policy in `probe`.
#[repr(C)]
struct Cluster {
    entry: [TTEntry; CLUSTER_SIZE],
    /// Trailing bytes with nothing in them. An all-zero `Cluster` is a valid
    /// empty one, so they stay zero for the life of the table.
    #[cfg(any(not(feature = "verbose3"), feature = "tt-entry16"))]
    _padding: [u8; CLUSTER_PADDING],
    /// The entries' path-dependence marks, bit `i` belonging to `entry[i]`.
    /// It occupies the two bytes the cluster has to spare, and is 2-aligned
    /// there, so the cluster is still 32 bytes and no entry moves.
    #[cfg(all(feature = "verbose3", not(feature = "tt-entry16")))]
    marks: AtomicU16,
}

impl Cluster {
    /// An empty cluster: the all-zero bit pattern the table starts as. Needed
    /// only where the table is a Rust `static` — see [`TTEntry::empty`].
    #[cfg(miri)]
    const fn empty() -> Cluster {
        Cluster {
            entry: [const { TTEntry::empty() }; CLUSTER_SIZE],
            #[cfg(any(not(feature = "verbose3"), feature = "tt-entry16"))]
            _padding: [0; CLUSTER_PADDING],
            #[cfg(all(feature = "verbose3", not(feature = "tt-entry16")))]
            marks: AtomicU16::new(0),
        }
    }

    /// Decode `slot`'s payload (`TTEntry::read`), including the mark, which is
    /// not an entry field.
    #[inline]
    fn read(&self, slot: usize) -> TTData {
        #[cfg(not(feature = "verbose3"))]
        {
            self.entry[slot].read()
        }
        #[cfg(feature = "verbose3")]
        {
            let mut data = self.entry[slot].read();
            data.path_dep = self.path_dep(slot);
            data
        }
    }

    /// Zero every entry, and with it every mark.
    #[inline]
    fn reset(&self) {
        for entry in &self.entry {
            entry.reset();
        }
        #[cfg(all(feature = "verbose3", not(feature = "tt-entry16")))]
        self.marks.store(0, REL);
    }

    /// Store a new node's data into `slot`, possibly overwriting an older
    /// position (`TTEntry::save`). `curr_generation` is an argument rather than
    /// read from the table, because the reference lets learners pass a
    /// per-thread one.
    ///
    /// The old fields are each read once before any store, so the replacement
    /// decision sees the pre-save entry state. `path_dep` follows the payload:
    /// it is written exactly when the payload is, and left alone when the
    /// replacement condition declines the store.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn save(
        &self,
        slot: usize,
        k: TteKey,
        v: Value,
        pv: bool,
        b: Bound,
        d: Depth,
        m: u16,
        ev: Value,
        curr_generation: u8,
        #[cfg(feature = "verbose3")] path_dep: bool,
    ) {
        // The stored key's bit 0 is the mark here, and `tte_key` has already
        // cleared it out of `k`.
        #[cfg(all(feature = "verbose3", feature = "tt-entry16"))]
        let k = k | path_dep as TteKey;

        let entry = &self.entry[slot];
        let old_key = entry.key.load(REL);
        let old_depth8 = entry.depth8.load(REL);
        let old_gen_bound8 = entry.gen_bound8.load(REL);

        // Preserve the old move if we don't have a new one for this position.
        if m != 0 || !key_matches(old_key, k) {
            entry.move16.store(m, REL);
        }

        // The depth comparison is in `i32` to match the reference's `int`
        // promotion: `depth8 - 4` may be negative.
        if b == Bound::Exact
            || !key_matches(old_key, k)
            || d - DEPTH_NONE + 2 * pv as Depth > old_depth8 as Depth - 4
            || relative_age(old_gen_bound8, curr_generation) != 0
        {
            debug_assert!(d > DEPTH_NONE);
            debug_assert!(d - DEPTH_NONE < 256);
            debug_assert!(curr_generation <= GENERATION_MASK);

            entry.key.store(k, REL);
            entry.depth8.store((d - DEPTH_NONE) as u8, REL);
            entry.gen_bound8.store(
                curr_generation | (b as u8) << BOUND_SHIFT | (pv as u8) << PV_SHIFT,
                REL,
            );
            entry.value16.store(v as i16, REL);
            entry.eval16.store(ev as i16, REL);
            #[cfg(all(feature = "verbose3", not(feature = "tt-entry16")))]
            self.set_path_dep(slot, path_dep);
        }
    }
}

#[cfg(feature = "verbose3")]
impl Cluster {
    /// `slot`'s path-dependence mark.
    #[inline]
    fn path_dep(&self, slot: usize) -> bool {
        #[cfg(not(feature = "tt-entry16"))]
        {
            self.marks.load(REL) & (1 << slot) != 0
        }
        #[cfg(feature = "tt-entry16")]
        {
            self.entry[slot].key.load(REL) & 1 != 0
        }
    }

    /// Write `slot`'s mark, leaving its neighbours' bits alone: every entry in
    /// the cluster shares this word, and their writers race.
    #[cfg(not(feature = "tt-entry16"))]
    #[inline]
    fn set_path_dep(&self, slot: usize, path_dep: bool) {
        let bit = 1u16 << slot;
        if path_dep {
            self.marks.fetch_or(bit, REL);
        } else {
            self.marks.fetch_and(!bit, REL);
        }
    }
}

// The `clusterCount` arithmetic assumes a 32-byte cluster, and `tt-entry16`'s
// contract is precisely that its six extra bytes all go to the key and leave
// the payload offsets unmoved. Both are statements about layout, so they are
// checked at compile time, in every build.
const _: () = assert!(size_of::<Cluster>() == 32);
const _: () = assert!(size_of::<TTEntry>() * CLUSTER_SIZE + CLUSTER_PADDING == 32);

/// Size of the stored key, and hence the offset of every payload field after
/// it: 2 by default, 8 under `tt-entry16`.
const KEY_SIZE: usize = size_of::<TteKey>();

const _: () = assert!(size_of::<TTEntry>() == if cfg!(feature = "tt-entry16") { 16 } else { 10 });
const _: () = assert!(KEY_SIZE == if cfg!(feature = "tt-entry16") { 8 } else { 2 });
const _: () = assert!(CLUSTER_SIZE == if cfg!(feature = "tt-entry16") { 2 } else { 3 });

// The payload fields pack against the key with no interior padding in either
// layout: under `tt-entry16` the `u64` key already leaves `move16` 2-aligned,
// so `#[repr(C)]` inserts nothing.
const _: () = assert!(offset_of!(TTEntry, key) == 0);
const _: () = assert!(offset_of!(TTEntry, depth8) == KEY_SIZE);
const _: () = assert!(offset_of!(TTEntry, gen_bound8) == KEY_SIZE + 1);
const _: () = assert!(offset_of!(TTEntry, move16) == KEY_SIZE + 2);
const _: () = assert!(offset_of!(TTEntry, value16) == KEY_SIZE + 4);
const _: () = assert!(offset_of!(TTEntry, eval16) == KEY_SIZE + 6);

// The mark word sits in the cluster's spare bytes, where its 2-byte alignment
// is satisfied, so the cluster neither grows nor shifts an entry.
#[cfg(all(feature = "verbose3", not(feature = "tt-entry16")))]
const _: () = assert!(offset_of!(Cluster, marks) == size_of::<TTEntry>() * CLUSTER_SIZE);

/// Base alignment of the table: a 2 MiB huge-page boundary, so the array starts
/// where a `MADV_HUGEPAGE` hint over it can be honoured. The linker places the
/// `static` on it, which is why nothing here rounds an address up.
///
/// Written as a literal because `#[repr(align(N))]` takes no expression; the
/// assertion below holds it against the huge-page policy the rest of the crate
/// allocates under.
pub const TT_ALIGN: usize = 2 * 1024 * 1024;
const _: () = assert!(TT_ALIGN >= crate::large_page::LARGE_PAGE_ALIGN);

/// Clusters in the table — the reference's `clusterCount`, here a constant:
/// `usi_hash · 2²⁰ / sizeof(Cluster)`, the same arithmetic the reference runs
/// when it allocates.
///
/// It is always even, which is what lets the side to move be folded into
/// cluster-index bit 0 and stay in range.
#[cfg(not(miri))]
pub const CLUSTER_COUNT: usize =
    crate::config::USI_HASH as usize * 1024 * 1024 / size_of::<Cluster>();

/// Clusters in the table under miri, which const-evaluates the `static`'s
/// initialiser and interprets every access to it — both linear in this count,
/// and minutes of it at the count `usi_hash` folds to. 8192 clusters, 256 KiB,
/// is the smallest count that leaves every property the storage tests read off
/// the table intact: it is even, it is well past the 1000 clusters `hashfull`
/// samples, it is a power of two like every configured count, and the two
/// 4096-cluster ends the emptiness check walks still meet without overlapping.
#[cfg(miri)]
pub const CLUSTER_COUNT: usize = 8192;

const _: () = assert!(CLUSTER_COUNT.is_multiple_of(2));
// `hashfull` samples the first 1000 clusters, and the smallest `usi_hash` the
// schema accepts is 1 MiB, which is 32768 of them.
const _: () = assert!(CLUSTER_COUNT >= 1000);

/// Bytes of clusters — what `usi_hash` asked for, and what a report of the
/// table's size means. The `static` holding them is a little larger, since its
/// alignment rounds the object up; see [`TranspositionTable::backing_region`],
/// which is the span to hand the kernel.
pub const TABLE_BYTES: usize = CLUSTER_COUNT * size_of::<Cluster>();

/// High 64 bits of the 128-bit product `a · b` (`mul_hi64`), which maps a key
/// onto `0..clusterCount` without a power-of-two table size.
#[inline]
fn mul_hi64(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) >> 64) as u64
}

/// Cluster index for `key` with `side_to_move` folded into bit 0
/// (`TranspositionTable::first_entry`), so the two sides never share a cluster.
///
/// [`CLUSTER_COUNT`] is a constant, so the multiply is against a literal and
/// nothing is loaded to find the table's size.
#[inline]
fn cluster_index(key: u64, side_to_move: u8) -> usize {
    let index = mul_hi64(key, CLUSTER_COUNT as u64) as usize;
    (index & !1) | (side_to_move as usize & 1)
}

/// The engine's transposition table — [`CLUSTER_COUNT`] contiguous `Cluster`s,
/// on a [`TT_ALIGN`] boundary.
///
/// There is exactly one, and [`Self::shared`] is the only way to reach it. A
/// type this large is never a value: the private field keeps anything but that
/// one table from existing.
#[repr(C, align(2097152))]
pub struct TranspositionTable {
    clusters: [Cluster; CLUSTER_COUNT],
}

/// Bytes the table's storage occupies: the clusters, rounded up to a whole
/// number of [`TT_ALIGN`] units by the type's alignment — up to one spare huge
/// page of BSS, never touched and never resident. The rounding is what lets
/// [`TranspositionTable::backing_region`] hand a whole object to the kernel
/// without naming a neighbour's page.
const TABLE_STORAGE_BYTES: usize = size_of::<TranspositionTable>();

// The one transposition table's storage: `TABLE_STORAGE_BYTES` zero bytes on a
// `TT_ALIGN` boundary, stated to the assembler rather than built by the
// constant evaluator (see the module docs). The symbol is hidden, so a
// reference to it resolves within this binary instead of through the dynamic
// symbol table.
//
// The section is writable and `nobits` like `.bss`, but is deliberately not
// named `.bss.*`: a linker script folds those into `.bss` in input order, and
// this object landing anywhere but last there pushes the ordinary
// zero-initialised objects — every `static` in the program, standard library
// included — beyond the ±2 GiB a `R_X86_64_PC32` reference can reach, which
// fails the link outright at the sizes a large-memory host is configured for.
// Under its own name it is an orphan section, placed after the last section
// like it, so only the table itself sits far from the code and nothing has to
// reach past it.
#[cfg(not(miri))]
core::arch::global_asm!(
    ".section .yorkie_transposition_table,\"aw\",@nobits",
    ".balign {align}",
    ".globl yorkie_transposition_table",
    ".hidden yorkie_transposition_table",
    ".type yorkie_transposition_table, @object",
    "yorkie_transposition_table:",
    ".zero {bytes}",
    ".size yorkie_transposition_table, {bytes}",
    ".previous",
    align = const TT_ALIGN,
    bytes = const TABLE_STORAGE_BYTES,
);

// The typed view of that storage. Zero at process start, which is an empty
// table — the first game finds it ready without anything clearing or faulting
// it in.
#[cfg(not(miri))]
unsafe extern "C" {
    #[link_name = "yorkie_transposition_table"]
    static TABLE: TranspositionTable;
}

/// The same table for a miri build, which cannot see the symbol above.
#[cfg(miri)]
static TABLE: TranspositionTable = TranspositionTable {
    clusters: [const { Cluster::empty() }; CLUSTER_COUNT],
};

/// Generation counter, bumped once per [`TranspositionTable::new_search`]. Only
/// the low [`GENERATION_BITS`] bits are significant. Atomic so the searcher can
/// bump it through the shared `&self`.
///
/// A `static` beside the table rather than a field in it, so the table's
/// storage is clusters and nothing else.
static GENERATION8: AtomicU8 = AtomicU8::new(0);

// The clusters lead the object and its size is a whole number of alignment
// units, so the table's address and size describe the huge-page-aligned region
// a placement call names, with no arithmetic in between. These hold whatever
// declares the storage, and the byte count the assembler is given is this same
// size.
const _: () = assert!(align_of::<TranspositionTable>() == TT_ALIGN);
const _: () = assert!(offset_of!(TranspositionTable, clusters) == 0);
const _: () = assert!(TABLE_STORAGE_BYTES.is_multiple_of(TT_ALIGN));
const _: () = assert!(TABLE_STORAGE_BYTES >= TABLE_BYTES);

impl TranspositionTable {
    /// The one table. Every worker, and the driver, hold this same reference.
    #[inline]
    pub fn shared() -> &'static TranspositionTable {
        #[cfg(not(miri))]
        {
            // SAFETY: the symbol is defined in this binary, by the
            // `global_asm!` above, as `size_of::<TranspositionTable>()` zero
            // bytes aligned to the type's alignment. Every byte of that type is
            // an atomic or padding, so any bit pattern — the zeros BSS starts
            // as included — is a valid value of it, and every access to it goes
            // through a shared reference. Nothing else in the process names the
            // symbol, so no `&mut` to it can exist.
            unsafe { &TABLE }
        }
        #[cfg(miri)]
        {
            &TABLE
        }
    }

    /// The `(address, byte length)` of the table, for a caller placing it: a
    /// NUMA policy over its pages, a huge-page hint, or a `/proc/self/smaps`
    /// region to read the huge-page uptake back from.
    ///
    /// The length covers the whole `static`, which is a whole number of
    /// [`TT_ALIGN`] units, so the range names no neighbour's page. The address
    /// is a `usize` because the consumer hands it to the kernel as a range
    /// descriptor and never dereferences it.
    #[inline]
    pub fn backing_region(&self) -> (usize, usize) {
        (std::ptr::from_ref(self) as usize, TABLE_STORAGE_BYTES)
    }

    /// Zero every entry and reset the generation (`TranspositionTable::clear`).
    ///
    /// Runs over the whole table, which is what `usinewgame` asks for; there is
    /// no smaller region to clear, since the size is fixed for the process.
    pub fn clear(&self) {
        GENERATION8.store(0, REL);
        for cluster in &self.clusters {
            cluster.reset();
        }
    }

    /// Bump the generation at the start of a root search
    /// (`TranspositionTable::new_search`). Wraps within `GENERATION_BITS`, so
    /// it never spills into the bound or pv bits of `genBound8`.
    pub fn new_search(&self) {
        let next = GENERATION8.load(REL).wrapping_add(1) & GENERATION_MASK;
        GENERATION8.store(next, REL);
    }

    /// The current generation (`TranspositionTable::generation`).
    #[inline]
    pub fn generation(&self) -> u8 {
        GENERATION8.load(REL)
    }

    /// Approximate table occupancy in permille, counting only entries younger
    /// than `max_age` (`TranspositionTable::hashfull`). Samples the first 1000
    /// clusters; the table must hold at least that many.
    ///
    /// Its only consumer is the search `info` line's `hashfull` field, so it is
    /// compiled only into a build that prints one.
    #[cfg(feature = "verbose2")]
    pub fn hashfull(&self, max_age: u8) -> u32 {
        let generation = GENERATION8.load(REL);
        let mut cnt = 0u32;
        for cluster in self.clusters.iter().take(1000) {
            for entry in &cluster.entry {
                if entry.is_occupied()
                    && relative_age(entry.gen_bound8.load(REL), generation) <= max_age
                {
                    cnt += 1;
                }
            }
        }
        cnt / CLUSTER_SIZE as u32
    }

    /// Software-prefetch the cluster [`Self::probe`] would select for
    /// `(key, side_to_move)`. A no-op off x86-64.
    ///
    /// The reference issues this mid-`do_move`, because its position holds a TT
    /// pointer. The layering rules forbid that here, so the hint comes from the
    /// caller just after `do_move` returns — later, but still well ahead of the
    /// child's probe, and a prefetch has no architectural semantics anyway.
    #[inline]
    pub fn prefetch(&self, key: u64, side_to_move: u8) {
        #[cfg(target_arch = "x86_64")]
        {
            let ci = cluster_index(key, side_to_move);
            // SAFETY: `_mm_prefetch` is a pure hardware hint — it neither reads
            // nor writes the pointed-to memory observably and cannot fault, so
            // it carries no preconditions. `ci` is in `0..CLUSTER_COUNT` by
            // `cluster_index`, so the pointer is in bounds regardless.
            unsafe {
                use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
                let ptr = self.clusters.as_ptr().add(ci) as *const i8;
                _mm_prefetch::<_MM_HINT_T0>(ptr);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = (key, side_to_move);
        }
    }

    /// Look up `key` (`TranspositionTable::probe`).
    ///
    /// Returns `(found, data, writer)`, where `writer` targets the matching
    /// entry on a hit and the least-valuable entry to replace on a miss. By
    /// default `found` may be a 16-bit key collision; under `tt-entry16` the
    /// 64-bit match makes it exact.
    pub fn probe(&self, key: u64, side_to_move: u8) -> (bool, TTData, TTWriter<'_>) {
        let ci = cluster_index(key, side_to_move);
        let k = tte_key(key);
        let generation = GENERATION8.load(REL);
        let cluster = &self.clusters[ci];

        if let Some(i) = (0..CLUSTER_SIZE).find(|&i| key_matches(cluster.entry[i].key.load(REL), k))
        {
            let found = cluster.entry[i].is_occupied();
            let data = cluster.read(i);
            return (found, data, TTWriter::new(cluster, i));
        }

        // Miss: pick the least-valuable entry to replace.
        let mut replace = 0;
        for i in 1..CLUSTER_SIZE {
            if cluster.entry[replace].replace_priority(generation)
                > cluster.entry[i].replace_priority(generation)
            {
                replace = i;
            }
        }

        (false, TTData::none(), TTWriter::new(cluster, replace))
    }

    /// Like [`Self::probe`] but returns the chosen entry's location rather than
    /// a borrowing [`TTWriter`], so the caller can hold it across the recursive
    /// search calls that also mutate the table and still write to the **same**
    /// physical entry through [`Self::write_at`]. Re-probing at write time
    /// would re-run the replacement selection against a cluster the children
    /// have since churned, and could land on a different slot.
    pub fn locate(&self, key: u64, side_to_move: u8) -> (bool, TTData, TtSlot) {
        let ci = cluster_index(key, side_to_move);
        let k = tte_key(key);
        let generation = GENERATION8.load(REL);
        let cluster = &self.clusters[ci];

        if let Some(i) = (0..CLUSTER_SIZE).find(|&i| key_matches(cluster.entry[i].key.load(REL), k))
        {
            let found = cluster.entry[i].is_occupied();
            let data = cluster.read(i);
            return (
                found,
                data,
                TtSlot {
                    cluster: ci,
                    entry: i,
                },
            );
        }

        let mut replace = 0;
        for i in 1..CLUSTER_SIZE {
            if cluster.entry[replace].replace_priority(generation)
                > cluster.entry[i].replace_priority(generation)
            {
                replace = i;
            }
        }

        (
            false,
            TTData::none(),
            TtSlot {
                cluster: ci,
                entry: replace,
            },
        )
    }

    /// Store into the exact entry [`Self::locate`] captured, under the same
    /// replacement policy as [`TTWriter::write`]. Addressed by index rather
    /// than re-selected, so it lands on that slot even when a child has since
    /// overwritten the entry.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn write_at(
        &self,
        slot: TtSlot,
        key: u64,
        value: Value,
        pv: bool,
        bound: Bound,
        depth: Depth,
        mv: u16,
        eval: Value,
        generation: u8,
        #[cfg(feature = "verbose3")] path_dep: bool,
    ) {
        self.clusters[slot.cluster].save(
            slot.entry,
            tte_key(key),
            value,
            pv,
            bound,
            depth,
            mv,
            eval,
            generation,
            #[cfg(feature = "verbose3")]
            path_dep,
        );
    }

    /// A stable checksum over the whole table's raw bytes.
    ///
    /// Nothing a game plays reads it: it exists to pin the table's contents
    /// while the table is being inspected from the outside, which is what the
    /// `verbose3` feature is for, so it is compiled only there.
    ///
    /// The path-dependence marks are part of what the table stores and so are
    /// mixed in: in the default layout as the cluster's mark word, and under
    /// `tt-entry16` inside the key each entry already contributes.
    #[cfg(feature = "verbose3")]
    pub fn checksum(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut mix = |x: u64| {
            h ^= x;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        };
        for cluster in self.clusters.iter() {
            for e in &cluster.entry {
                mix(key_bits(e.key.load(REL)));
                mix(e.depth8.load(REL) as u64);
                mix(e.gen_bound8.load(REL) as u64);
                mix(e.move16.load(REL) as u64);
                mix(e.value16.load(REL) as u16 as u64);
                mix(e.eval16.load(REL) as u16 as u64);
            }
            #[cfg(not(feature = "tt-entry16"))]
            mix(cluster.marks.load(REL) as u64);
        }
        mix(GENERATION8.load(REL) as u64);
        h
    }
}

/// The location of a resolved TT entry. Holds no borrow, so it survives the
/// recursive search calls between a node's probe and its writes.
#[derive(Clone, Copy, Debug)]
pub struct TtSlot {
    cluster: usize,
    entry: usize,
}

/// A single-use handle for writing one entry (`TTWriter`). It addresses the
/// cluster and a slot in it rather than the entry alone, because an entry's
/// path-dependence mark can live outside the entry.
pub struct TTWriter<'a> {
    cluster: &'a Cluster,
    slot: usize,
}

impl<'a> TTWriter<'a> {
    #[inline]
    fn new(cluster: &'a Cluster, slot: usize) -> Self {
        TTWriter { cluster, slot }
    }

    /// Store data into the targeted entry, subject to the replacement policy
    /// (`TTWriter::write`). `key` is the full 64-bit position key, of which the
    /// entry keeps whatever its layout stores.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn write(
        self,
        key: u64,
        value: Value,
        pv: bool,
        bound: Bound,
        depth: Depth,
        mv: u16,
        eval: Value,
        generation: u8,
        #[cfg(feature = "verbose3")] path_dep: bool,
    ) {
        self.cluster.save(
            self.slot,
            tte_key(key),
            value,
            pv,
            bound,
            depth,
            mv,
            eval,
            generation,
            #[cfg(feature = "verbose3")]
            path_dep,
        );
    }
}

#[cfg(test)]
mod static_table_tests {
    //! What the table's storage itself has to be, which the public API cannot
    //! show: its placement, its size, and that BSS really did give it an empty
    //! table over the whole span the storage was declared with.
    //!
    //! Nothing here writes to the table, so these run beside any other test of
    //! the process without a lock.

    use super::*;

    /// [`CLUSTER_COUNT`], recomputed from the setting rather than read from the
    /// constant, so a change to either arithmetic fails this.
    #[cfg(not(miri))]
    const EXPECTED_CLUSTERS: usize = crate::config::USI_HASH as usize * 1024 * 1024 / 32;

    /// The count a miri build fixes instead of folding from the setting,
    /// restated rather than read from the constant for the same reason.
    #[cfg(miri)]
    const EXPECTED_CLUSTERS: usize = 8192;

    #[test]
    fn cluster_count_follows_the_configured_hash_size() {
        assert_eq!(CLUSTER_COUNT, EXPECTED_CLUSTERS);
        assert_eq!(CLUSTER_COUNT % 2, 0, "the side fold needs an even count");
    }

    #[test]
    fn the_table_sits_on_a_huge_page_boundary() {
        let (addr, bytes) = TranspositionTable::shared().backing_region();
        assert_eq!(addr % TT_ALIGN, 0, "the table must start on a huge page");
        assert_eq!(bytes % TT_ALIGN, 0, "and cover whole ones");

        // The clusters, rounded up to whole alignment units — nothing else is
        // in there.
        assert_eq!(bytes, TABLE_STORAGE_BYTES);
        assert!(bytes >= TABLE_BYTES);
        assert!(bytes - TABLE_BYTES < TT_ALIGN);
    }

    /// The first and last 4096 clusters: enough to show the storage was zero at
    /// process start, and that it really spans the clusters it was declared
    /// with — the far end of it is the part a byte count stated to the
    /// assembler could get wrong. Short enough to finish under miri.
    #[test]
    fn the_table_starts_empty_from_end_to_end() {
        let tt = TranspositionTable::shared();
        let edges = tt
            .clusters
            .iter()
            .take(4096)
            .chain(tt.clusters.iter().skip(CLUSTER_COUNT - 4096));
        for cluster in edges {
            for e in &cluster.entry {
                assert!(!e.is_occupied());
                assert_eq!(e.key.load(REL), 0);
                assert_eq!(e.gen_bound8.load(REL), 0);
                assert_eq!(e.move16.load(REL), 0);
                assert_eq!(e.value16.load(REL), 0);
                assert_eq!(e.eval16.load(REL), 0);
            }
            #[cfg(any(not(feature = "verbose3"), feature = "tt-entry16"))]
            assert_eq!(cluster._padding, [0u8; CLUSTER_PADDING]);
            #[cfg(all(feature = "verbose3", not(feature = "tt-entry16")))]
            assert_eq!(cluster.marks.load(REL), 0);
        }
    }

    #[test]
    fn every_cluster_index_is_in_range_and_carries_the_side() {
        for key in [
            0u64,
            1,
            u64::MAX,
            0x8000_0000_0000_0000,
            0x1234_5678_9abc_def0,
        ] {
            for side in 0..2u8 {
                let ci = cluster_index(key, side);
                assert!(ci < CLUSTER_COUNT, "index {ci} out of range for {key:#x}");
                assert_eq!(ci & 1, side as usize, "bit 0 carries the side to move");
            }
        }
    }
}
