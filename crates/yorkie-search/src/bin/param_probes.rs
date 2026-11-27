//! Calls both functions of every pair in `yorkie_search::probes`, so that a
//! release build of this binary carries each of them as a function of its own
//! whose machine code can be disassembled and compared. The arguments pass
//! through `black_box`, so no call is folded into a constant, and the two
//! results of each pair are checked to agree.

use std::hint::black_box;

use yorkie_search::probes;

fn main() {
    for x in [0_i64, 1, 7, 1000, 123_456_789, -3, -40_000] {
        let u = x.unsigned_abs();
        assert_eq!(
            probes::probe_divide_read(black_box(u)),
            probes::probe_divide_written(black_box(u))
        );

        let d = (x % 64) as i32;
        let e = (x % 30_000) as i32;
        assert_eq!(
            probes::probe_compare_read(black_box(d), black_box(e), black_box(76)),
            probes::probe_compare_written(black_box(d), black_box(e), black_box(76)),
        );

        assert_eq!(
            probes::probe_scaled_read(black_box(x), black_box(x / 2)),
            probes::probe_scaled_written(black_box(x), black_box(x / 2)),
        );

        assert_eq!(
            probes::probe_fraction_read(black_box(d)),
            probes::probe_fraction_written(black_box(d))
        );
    }
}
