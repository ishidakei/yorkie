use crate::movetypes::*;
use crate::position::*;
use crate::thread::*;
use crate::types::*;
use rayon::prelude::*;
use std::ops::{Deref, DerefMut};
#[cfg(target_os = "linux")]
use std::ptr::NonNull;

#[derive(Clone, Copy)]
pub struct TtEntry {
    key16: u16,
    mv16: u16,
    value16: i16,
    eval16: i16,
    genbound8: u8,
    depth8: u8,
}

impl TtEntry {
    fn new() -> Self {
        Self {
            key16: 0,
            mv16: 0,
            value16: 0,
            eval16: 0,
            genbound8: 0,
            depth8: 0,
        }
    }
    pub fn mv(&self, pos: &Position) -> Option<Move> {
        // This can be illegal move.
        let m = Move(std::num::NonZeroU32::new(u32::from(self.mv16))?);
        let m = if !Some(m).is_normal_move() || m.is_drop() {
            m
        } else {
            Move(unsafe {
                std::num::NonZeroU32::new_unchecked(m.0.get() | ((pos.piece_on(m.from()).0 as u32) << Move::MOVED_PIECE_SHIFT))
            })
        };
        if pos.pseudo_legal::<SearchingType>(m) { Some(m) } else { None }
    }
    pub fn value(&self) -> Value {
        Value(i32::from(self.value16))
    }
    pub fn eval(&self) -> Value {
        Value(i32::from(self.eval16))
    }
    pub fn depth(&self) -> Depth {
        Depth(i32::from(self.depth8)) + Depth::OFFSET
    }
    pub fn is_pv(&self) -> bool {
        (self.genbound8 & 0x4) != 0
    }
    pub fn bound(&self) -> Bound {
        Bound(i32::from(self.genbound8) & 0x3)
    }
    #[allow(dead_code)]
    pub fn generation(&self) -> u8 {
        self.genbound8 & GENERATION_MASK as u8
    }
    pub fn save(
        &mut self,
        key: Key,
        value: Value,
        pv: bool,
        bound: Bound,
        depth: Depth,
        mv: Option<Move>,
        eval: Value,
        generation: u8,
    ) {
        let key = key.excluded_turn().0 as u16;
        if let Some(mv) = mv {
            self.mv16 = u32::from(mv.0) as u16;
        } else if key != self.key16 {
            self.mv16 = 0;
        }

        if bound == Bound::EXACT || key != self.key16 || depth.0 - Depth::OFFSET.0 > i32::from(self.depth8) - 4 {
            debug_assert!(depth > Depth::OFFSET);
            debug_assert!(depth.0 < 256 + Depth::OFFSET.0);
            self.key16 = key;
            self.value16 = value.0 as i16;
            self.eval16 = eval.0 as i16;
            self.genbound8 = (i32::from(generation) | (i32::from(pv) << 2) | bound.0) as u8;
            self.depth8 = (depth.0 - Depth::OFFSET.0) as u8;
        }
    }
}

const CLUSTER_SIZE: usize = 3;

const GENERATION_BITS: u32 = 3;
const GENERATION_DELTA: u8 = 1 << GENERATION_BITS;
const GENERATION_CYCLE: i32 = 255 + (1 << GENERATION_BITS);
const GENERATION_MASK: i32 = (0xff << GENERATION_BITS) & 0xff;

#[repr(align(32))]
#[derive(Clone, Copy)]
struct TtCluster {
    entry: [TtEntry; CLUSTER_SIZE],
    _padding: [u8; 2],
}

impl TtCluster {
    fn new() -> Self {
        Self {
            entry: [TtEntry::new(); CLUSTER_SIZE],
            _padding: [0; 2],
        }
    }
}

// Backing store for the TT clusters. On Linux, mmap + madvise(MADV_HUGEPAGE) *before* faulting so pages come in as 2 MiB hugepages (random probes thrash the dTLB otherwise; khugepaged won't collapse a multi-GiB TT mid-search). Deref to `[TtCluster]` keeps probe sites unchanged; `Heap` is the non-Linux / mmap-failure fallback.
enum TtStorage {
    // Anonymous mmap (hugepage-hinted before faulting); `len` is the cluster count. Page-aligned base satisfies `align(32)`.
    #[cfg(target_os = "linux")]
    Mmap {
        ptr: NonNull<TtCluster>,
        len: usize,
    },
    Heap(Vec<TtCluster>),
}

impl TtStorage {
    // Allocate `cluster_count` zeroed clusters, faulted in parallel: hugepage-hinted mmap on Linux, heap `Vec` otherwise.
    fn new(cluster_count: usize) -> Self {
        #[cfg(target_os = "linux")]
        if let Some(storage) = Self::new_mmap(cluster_count) {
            return storage;
        }
        // Fallback (non-Linux / mmap failed): parallel collect faults the buffer in as 4 KiB pages.
        Self::Heap((0..cluster_count).into_par_iter().map(|_| TtCluster::new()).collect())
    }

    // Linux hugepage-hinted alloc: mmap → madvise(MADV_HUGEPAGE) → parallel fault-in as 2 MiB pages. `None` (caller falls back to `Heap`) on empty request or mmap failure.
    #[cfg(target_os = "linux")]
    fn new_mmap(cluster_count: usize) -> Option<Self> {
        let len = cluster_count.checked_mul(std::mem::size_of::<TtCluster>())?;
        if len == 0 {
            return None; // let the Heap path own the empty table
        }
        // SAFETY: standard anonymous private mapping request; mmap returns MAP_FAILED (not null) on error.
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if base == libc::MAP_FAILED {
            return None;
        }
        // Best-effort THP hint issued BEFORE faulting, so the parallel init installs 2 MiB hugepages directly (failure is harmless — falls back to 4 KiB).
        // SAFETY: `base`/`len` describe the live mapping just returned by `mmap`.
        unsafe {
            libc::madvise(base, len, libc::MADV_HUGEPAGE);
        }
        let ptr = match NonNull::new(base as *mut TtCluster) {
            Some(ptr) => ptr,
            None => {
                // Unreachable in practice (a successful mmap is non-null), but stay sound.
                // SAFETY: `base`/`len` are the live mapping from `mmap` above.
                unsafe {
                    libc::munmap(base, len);
                }
                return None;
            }
        };
        // Fault the mapping in (parallel), writing zeroed clusters to trigger the 2 MiB fault-ins; anonymous mmap memory is already zeroed, so the slice is a valid `[TtCluster]`.
        // SAFETY: `ptr` points at `cluster_count` consecutive, writable, zero-initialized `TtCluster`s.
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr(), cluster_count) };
        slice.par_iter_mut().for_each(|c| *c = TtCluster::new());
        Some(Self::Mmap { ptr, len: cluster_count })
    }
}

impl Deref for TtStorage {
    type Target = [TtCluster];
    fn deref(&self) -> &[TtCluster] {
        match self {
            #[cfg(target_os = "linux")]
            // SAFETY: the mapping holds `len` initialized `TtCluster`s for as long as `self` lives.
            TtStorage::Mmap { ptr, len } => unsafe { std::slice::from_raw_parts(ptr.as_ptr(), *len) },
            TtStorage::Heap(v) => v,
        }
    }
}

impl DerefMut for TtStorage {
    fn deref_mut(&mut self) -> &mut [TtCluster] {
        match self {
            #[cfg(target_os = "linux")]
            // SAFETY: the mapping holds `len` initialized `TtCluster`s; `&mut self` gives exclusive access for its lifetime.
            TtStorage::Mmap { ptr, len } => unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr(), *len) },
            TtStorage::Heap(v) => v,
        }
    }
}

impl Drop for TtStorage {
    fn drop(&mut self) {
        #[cfg(target_os = "linux")]
        if let TtStorage::Mmap { ptr, len } = self {
            let bytes = *len * std::mem::size_of::<TtCluster>();
            // SAFETY: `ptr`/`bytes` are the exact base/length from `mmap` in `new_mmap`; sole owner, dropped once.
            unsafe {
                libc::munmap(ptr.as_ptr() as *mut libc::c_void, bytes);
            }
        }
    }
}

// Tournament build: TT size is a compile-time const from the baked `TT_BYTES`, not a runtime field, dropping the per-probe field load (indexing arithmetic unchanged).
#[cfg(feature = "tournament")]
const TT_CLUSTER_COUNT: usize = crate::tournament::TT_BYTES / std::mem::size_of::<TtCluster>();
// Multiply-high indexing packs the side-to-move into the low bit, so the cluster count must be even; enforce it at compile time.
#[cfg(feature = "tournament")]
const _: () = assert!(
    TT_CLUSTER_COUNT & 1 == 0,
    "tournament tt_bytes must yield an even TT cluster count"
);

pub struct TranspositionTable {
    table: TtStorage,
    // Non-tournament: runtime cluster count (USI_Hash-driven), read on every probe; tournament folds the compile-time const instead.
    #[cfg(not(feature = "tournament"))]
    cluster_count: usize,
    generation8: u8,
}

impl TranspositionTable {
    pub fn new() -> TranspositionTable {
        TranspositionTable {
            table: TtStorage::Heap(vec![]),
            #[cfg(not(feature = "tournament"))]
            cluster_count: 0,
            generation8: 0,
        }
    }
    pub fn resize(&mut self, mega_byte_size: usize, thread_pool: &mut ThreadPool) {
        thread_pool.wait_for_search_finished();
        // Tournament build ignores `mega_byte_size` (USI_Hash) and uses the compile-time `TT_CLUSTER_COUNT`; non-tournament sizes from USI_Hash.
        #[cfg(feature = "tournament")]
        let cluster_count = {
            let _ = mega_byte_size;
            TT_CLUSTER_COUNT
        };
        #[cfg(not(feature = "tournament"))]
        let cluster_count = {
            self.cluster_count = mega_byte_size * 1024 * 1024 / std::mem::size_of::<TtCluster>();
            debug_assert!(self.cluster_count & 1 == 0);
            self.cluster_count
        };
        // Free the old backing before allocating the new one, so peak memory is max(old, new) not old + new.
        self.table = TtStorage::Heap(vec![]);
        self.table = TtStorage::new(cluster_count);
    }
    // parallel zero clearing.
    pub fn clear(&mut self) {
        self.table.par_iter_mut().for_each(|x| {
            *x = unsafe { std::mem::zeroed() };
        });
    }
    pub fn new_search(&mut self) {
        self.generation8 = self.generation8.wrapping_add(GENERATION_DELTA);
    }
    fn cluster_index(&self, key: Key) -> usize {
        fn mul_hi64(l: u64, r: u64) -> u64 {
            ((u128::from(l) * u128::from(r)) >> 64) as u64
        }
        // Stockfish-style multiply-high (NOT masking). Operand is the compile-time const (tournament) or the runtime field; the arithmetic is identical.
        #[cfg(feature = "tournament")]
        let cluster_count = TT_CLUSTER_COUNT;
        #[cfg(not(feature = "tournament"))]
        let cluster_count = self.cluster_count;
        let index = mul_hi64(key.excluded_turn().0, cluster_count as u64); // [0, cluster_count / 2 - 1]
        ((index << 1) | key.turn_bit()) as usize // [0, cluster_count - 1]
    }
    fn get_mut_cluster(&mut self, index: usize) -> &mut TtCluster {
        debug_assert!(index < self.table.len());
        unsafe { self.table.get_unchecked_mut(index) }
    }
    pub fn probe(&mut self, key: Key) -> (&mut TtEntry, bool) {
        let generation8 = self.generation8;
        let key16 = key.excluded_turn().0 as u16;
        let cluster = self.get_mut_cluster(self.cluster_index(key));
        for i in 0..cluster.entry.len() {
            if cluster.entry[i].key16 == key16 || i32::from(cluster.entry[i].depth8) == 0 {
                cluster.entry[i].genbound8 = generation8 | (cluster.entry[i].genbound8 & (GENERATION_DELTA - 1)); // refresh
                let found = i32::from(cluster.entry[i].depth8) != 0;
                return (&mut cluster.entry[i], found);
            }
        }
        let replace = cluster
            .entry
            .iter_mut()
            .min_by(|x, y| {
                let left = i32::from(x.depth8)
                    - ((GENERATION_CYCLE + i32::from(generation8) - i32::from(x.genbound8)) & GENERATION_MASK);
                let right = i32::from(y.depth8)
                    - ((GENERATION_CYCLE + i32::from(generation8) - i32::from(y.genbound8)) & GENERATION_MASK);
                left.cmp(&right)
            })
            .unwrap();
        let found = false;
        (replace, found)
    }
    pub fn generation(&self) -> u8 {
        self.generation8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size() {
        assert_eq!(std::mem::size_of::<TtEntry>(), 10);
        assert_eq!(std::mem::size_of::<TtCluster>(), 32);
        assert_eq!(std::mem::size_of::<[TtCluster; 4]>(), 128);
    }

    // Runtime-sizing test: asserts against the runtime `cluster_count` field, absent in the tournament build (covered by `tournament_tests`).
    #[cfg(not(feature = "tournament"))]
    #[test]
    fn test_cluster_index() {
        use crate::search::*;
        std::thread::Builder::new()
            .stack_size(crate::stack_size::STACK_SIZE)
            .spawn(|| {
                let mut thread_pool = ThreadPool::new();
                let mut tt = TranspositionTable::new();
                let mut reductions = Reductions::new();
                thread_pool.set(1, &mut tt, &mut reductions);
                tt.resize(1, &mut thread_pool);

                // If key is all 1 bits, index is max.
                let key = Key(0xffff_ffff_ffff_ffff);
                let index = tt.cluster_index(key);
                assert_eq!(index, tt.cluster_count - 1);
            })
            .unwrap()
            .join()
            .unwrap();
    }

    // Runtime-sizing test: runs non-tournament only (`resize(1)` would allocate the full fixed-size table under `tournament`).
    #[cfg(not(feature = "tournament"))]
    #[test]
    fn test_probe() {
        use crate::search::*;
        std::thread::Builder::new()
            .stack_size(crate::stack_size::STACK_SIZE)
            .spawn(|| {
                let mut thread_pool = ThreadPool::new();
                let mut tt = TranspositionTable::new();
                let mut reductions = Reductions::new();
                thread_pool.set(1, &mut tt, &mut reductions);
                tt.resize(1, &mut thread_pool);
                let pv = false;
                let gen8 = tt.generation8;

                use rand::prelude::*;
                let mut rand: StdRng = SeedableRng::seed_from_u64(123);
                let key = Key(0x0123_4567_89ab_cdef);
                let cluster_index = tt.cluster_index(key);

                let (tte, found) = tt.probe(key);
                assert!(!found);
                let (d2_val, d2) = (Value(20), Depth(2));
                tte.save(key, d2_val, pv, Bound::EXACT, d2, None, Value(0), gen8); // cluster: [(d2, gen_old), 0, 0]
                let (_, found) = tt.probe(key);
                assert!(found);

                fn gen_same_cluster_index_key(rng: &mut StdRng, cluster_index: usize, tt: &TranspositionTable) -> Key {
                    loop {
                        let key = Key(rng.r#gen());
                        let c_index = tt.cluster_index(key);
                        if c_index == cluster_index {
                            return key;
                        }
                    }
                }
                let key = gen_same_cluster_index_key(&mut rand, cluster_index, &tt);
                let (tte, found) = tt.probe(key);
                assert!(!found);
                let (d1_val, d1) = (Value(10), Depth(1));
                tte.save(key, d1_val, pv, Bound::EXACT, d1, None, Value(0), gen8); // cluster: [(d2, gen_old), (d1, gen_old), 0]
                let (_, found) = tt.probe(key);
                assert!(found);

                let key = gen_same_cluster_index_key(&mut rand, cluster_index, &tt);
                let (tte, found) = tt.probe(key);
                assert!(!found);
                let (d9_val, d9) = (Value(90), Depth(9));
                tte.save(key, d9_val, pv, Bound::EXACT, d9, None, Value(0), gen8); // cluster: [(d2, gen_old), (d1, gen_old), (d9, gen_old)]
                let (_, found) = tt.probe(key);
                assert!(found);

                tt.new_search();
                let gen8 = tt.generation8;

                let key = gen_same_cluster_index_key(&mut rand, cluster_index, &tt);
                let (tte, found) = tt.probe(key);
                assert!(!found);
                assert_eq!(tte.value(), d1_val); // the entry is most shallow depth
                let (d1_val, d1) = (Value(10), Depth(1));
                tte.save(key, d1_val, pv, Bound::EXACT, d1, None, Value(0), gen8); // cluster: [(d2, gen_old), (d1, gen_new), (d9, gen_old)]
                let (_, found) = tt.probe(key);
                assert!(found);

                let key = gen_same_cluster_index_key(&mut rand, cluster_index, &tt);
                let (tte, found) = tt.probe(key);
                assert!(!found);
                assert_eq!(tte.value(), d2_val); // old and shallow entry.
                let (d3_val, d3) = (Value(30), Depth(3));
                tte.save(key, d3_val, pv, Bound::EXACT, d3, None, Value(0), gen8); // cluster: [d3, gen_new), (d1, gen_new), (d9, gen_old)]
                let (_, found) = tt.probe(key);
                assert!(found);

                let key = gen_same_cluster_index_key(&mut rand, cluster_index, &tt);
                let (tte, found) = tt.probe(key);
                assert!(!found);
                assert_eq!(tte.value(), d1_val); // d9 entry has very deep depth. d9 isn't chosen.
                let (d2_val, d2) = (Value(20), Depth(2));
                tte.save(key, d2_val, pv, Bound::EXACT, d2, None, Value(0), gen8); // cluster: [d3, gen_new), (d2, gen_new), (d9, gen_old)]
                let (_, found) = tt.probe(key);
                assert!(found);
            })
            .unwrap()
            .join()
            .unwrap();
    }

    // Tournament build: prove `TT_CLUSTER_COUNT` is a genuine compile-time even const and that the indexing arithmetic is unchanged.
    #[cfg(feature = "tournament")]
    mod tournament_tests {
        use super::*;

        #[test]
        fn cluster_count_const_is_compile_time_and_even() {
            // `const` contexts: compile only if `TT_CLUSTER_COUNT` is a genuine compile-time const.
            const _: usize = TT_CLUSTER_COUNT;
            const _: () = assert!(TT_CLUSTER_COUNT & 1 == 0);
            assert_ne!(TT_CLUSTER_COUNT, 0, "a tournament TT must hold at least one cluster");
            assert_eq!(
                TT_CLUSTER_COUNT & 1,
                0,
                "multiply-high indexing requires an even cluster count"
            );
            // v1 placeholder: 256 MiB / 32-byte clusters.
            assert_eq!(
                TT_CLUSTER_COUNT,
                256 * 1024 * 1024 / std::mem::size_of::<TtCluster>(),
                "v1 tournament config bakes tt_bytes = 256 MiB",
            );
        }

        #[test]
        fn cluster_index_matches_runtime_formula() {
            // Independent reference: the pre-const arithmetic; `cluster_index` (reading the const) must agree with it.
            fn reference_index(key: Key, cluster_count: usize) -> usize {
                fn mul_hi64(l: u64, r: u64) -> u64 {
                    ((u128::from(l) * u128::from(r)) >> 64) as u64
                }
                let index = mul_hi64(key.excluded_turn().0, cluster_count as u64);
                ((index << 1) | key.turn_bit()) as usize
            }

            // `cluster_index` reads only the const, not the table, so no allocation is needed.
            let tt = TranspositionTable::new();
            for raw in [
                0x0000_0000_0000_0000u64,
                0x0000_0000_0000_0001,
                0x0123_4567_89ab_cdef,
                0xdead_beef_0000_0001,
                0xffff_ffff_ffff_ffff,
            ] {
                let key = Key(raw);
                let got = tt.cluster_index(key);
                assert_eq!(got, reference_index(key, TT_CLUSTER_COUNT), "key {raw:#018x}");
                assert!(got < TT_CLUSTER_COUNT, "index must stay within the const-sized table");
            }

            // All-ones key selects the last cluster, matching the runtime test's invariant.
            assert_eq!(tt.cluster_index(Key(0xffff_ffff_ffff_ffff)), TT_CLUSTER_COUNT - 1);
        }
    }
}
