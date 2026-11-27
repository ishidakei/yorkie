//! The CSA conversions allocate nothing: a game reads and writes a move per
//! ply and a position per game through them, and that path holds no heap
//! traffic.
//!
//! The counting allocator is this test binary's global allocator, so the file
//! holds a single test: a second one running on another thread would have its
//! allocations counted here.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use yorkie_state::{
    CSA_POSITION_CAPACITY, CsaMoveBuf, Move, Position, TextWriter, parse_csa_move,
    parse_csa_position_into, write_csa_move, write_csa_position,
};

struct Counting;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every call is forwarded to `System` unchanged; the counter is a side
// effect that touches no memory the allocation hands out.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: the caller's contract for `alloc` is passed on as it is.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `System.alloc` with this `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: the caller's contract for `realloc` is passed on as it is.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

const BLOCK: &[u8] = b"P1-KY-KE-GI-KI-OU-KI-GI-KE-KY\n\
P2 * -HI *  *  *  *  * -KA * \n\
P3-FU-FU-FU-FU-FU-FU-FU-FU-FU\n\
P4 *  *  *  *  *  *  *  *  * \n\
P5 *  *  *  *  *  *  *  *  * \n\
P6 *  *  *  *  *  *  *  *  * \n\
P7+FU+FU+FU+FU+FU+FU+FU+FU+FU\n\
P8 * +KA *  *  *  *  * +HI * \n\
P9+KY+KE+GI+KI+OU+KI+GI+KE+KY\n\
P-00AL\n\
+\n\
+7776FU,T3\n";

#[test]
fn parsing_and_writing_allocate_nothing() {
    let mut pos = Position::startpos();
    let mut legal = Vec::new();
    let mut start = Position::startpos();
    let mut text = [0u8; CSA_POSITION_CAPACITY];
    let mut buf = CsaMoveBuf::new();

    COUNTING.store(true, Ordering::Relaxed);
    let rest = parse_csa_position_into(&mut pos, BLOCK).expect("the block parses");
    let first = parse_csa_move(&rest[..7], &pos).expect("the listed move parses");
    let mut out = TextWriter::new(&mut text);
    write_csa_position(&pos, &mut out);
    write_csa_move(first, &mut out);
    let rendered: Move = parse_csa_move(buf.format(first), &pos).expect("its text parses");
    let _ = parse_csa_move(b"+7775FU", &pos);
    let _ = parse_csa_position_into(&mut start, b"P1\n+\n");
    COUNTING.store(false, Ordering::Relaxed);

    assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), 0);
    assert_eq!(rendered, first);
    pos.generate_legal_all(&mut legal);
    assert!(legal.contains(&first));
}
