//! Yorkie-book reader — the `YKBOOK01` opening book, whose entries may be
//! path-dependent.
//!
//! A `.ybb` keys an entry by the position alone. A yorkie book carries two
//! sections: **shared** entries, keyed by the position alone, and
//! **path-dependent** entries, which apply only when every position in their
//! support list has already occurred in the game. When a position has a
//! shared entry this reader answers with it, and otherwise with the first
//! path-dependent entry whose whole support list has been seen.
//!
//! The file is read whole and validated once, at open: magics, version,
//! checksum, every offset, every entry's extent and the key order. A book that
//! passes is then probed without a bounds or order check of its own — the
//! validation is what makes every slice the probe takes in range.
//!
//! Like the `.ybb` reader, this layer speaks only primitives: a packed position
//! is a raw `[u8; 32]` and a move is its 16-bit fragment. Packing a position and
//! widening a fragment into a legal move both live above.
//!
//! # Layout (version 2)
//!
//! `specs/yorkie-book-format.md` is the full specification;
//! this is its outline. All integers little-endian, offsets from the start of the file: a 40-byte
//! header (magic, version, padding, `n_shared`, `n_pathdep`,
//! `pathdep_index_off`); the shared offset table (`n_shared` × u64), directly
//! after the header, and the shared entries, sorted by key; the path-dependent
//! offset table (`n_pathdep` × u64) at `pathdep_index_off` and its entries,
//! sorted by key with equal keys contiguous; a 40-byte trailer holding the
//! SHA-256 of every byte before it and the ASCII `YKBOOKED`.
//!
//! An entry is its key (a packed position), `n_moves: u16` and `n_support:
//! u16`, then `n_moves` × `{ move16: u16, weight: u16 }` with the weights
//! summing to 10,000 in descending order, then `n_support` packed positions —
//! none in the shared section, at least one in the path-dependent one.

use std::cmp::Ordering;
use std::path::Path;

/// The header's leading magic.
const MAGIC: &[u8; 8] = b"YKBOOK01";
/// The trailer's closing magic.
const TRAILER_MAGIC: &[u8; 8] = b"YKBOOKED";
/// The only format version this reader parses.
const VERSION: u16 = 2;
/// Header size in bytes; the shared offset table starts here.
const HEADER_LEN: usize = 40;
/// Trailer size: the SHA-256 digest, then the closing magic.
const TRAILER_LEN: usize = 40;
/// The fixed part of an entry, both sections: the key and the two counts.
const ENTRY_PREFIX_LEN: usize = 36;
/// Where an entry's `n_moves` sits.
const N_MOVES_AT: usize = 32;
/// Where an entry's `n_support` sits.
const N_SUPPORT_AT: usize = 34;
/// One packed position.
const KEY_LEN: usize = 32;
/// One `{ move16, weight }` record.
const MOVE_LEN: usize = 4;
/// What the weights of one entry sum to.
pub const WEIGHT_TOTAL: u32 = 10_000;

/// Why a yorkie book was refused at open.
#[derive(Debug)]
pub enum YorkieBookError {
    /// The file could not be read.
    Io(std::io::Error),
    /// The file is shorter than a header and a trailer.
    Truncated,
    /// The leading 8 bytes are not `YKBOOK01`.
    BadMagic,
    /// The closing 8 bytes are not `YKBOOKED`.
    BadTrailerMagic,
    /// A format version this reader does not parse.
    UnsupportedVersion(u16),
    /// The trailer's digest is not the SHA-256 of the bytes before it.
    ChecksumMismatch,
    /// An offset table, or an entry it points at, runs outside the body.
    OffsetOutOfRange,
    /// An entry's own counts break the format: no moves, weights that do not
    /// sum to 10,000, a support list in the shared section or none in the
    /// path-dependent one.
    MalformedEntry,
    /// The shared keys are not strictly increasing, or the path-dependent keys
    /// decrease.
    KeysOutOfOrder,
}

impl YorkieBookError {
    /// Append this failure's message to `push`, one fragment at a time — the
    /// same shape as the `.ybb` reader's, for the same reason: the buffer the
    /// notice goes into belongs to the layer above.
    pub fn write_message(&self, mut push: impl FnMut(&[u8])) {
        match self {
            Self::Io(e) => {
                push(b"i/o error: errno ");
                match e.raw_os_error() {
                    Some(errno) => push_decimal(&mut push, errno.unsigned_abs().into()),
                    None => push(b"unknown"),
                }
            }
            Self::Truncated => push(b"file shorter than a header and a trailer"),
            Self::BadMagic => push(b"bad magic (not a YKBOOK01 file)"),
            Self::BadTrailerMagic => push(b"bad trailer magic (not YKBOOKED)"),
            Self::UnsupportedVersion(v) => {
                push(b"unsupported version ");
                push_decimal(&mut push, u64::from(*v));
                push(b" (this reader parses version 2)");
            }
            Self::ChecksumMismatch => push(b"SHA-256 mismatch"),
            Self::OffsetOutOfRange => push(b"an offset points outside the file"),
            Self::MalformedEntry => push(b"an entry's move or support counts are malformed"),
            Self::KeysOutOfOrder => push(b"keys out of order"),
        }
    }
}

impl From<std::io::Error> for YorkieBookError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Push `value`'s decimal digits.
fn push_decimal(push: &mut impl FnMut(&[u8]), value: u64) {
    let mut bytes = [0u8; 20];
    let mut start = bytes.len();
    let mut v = value;
    loop {
        start -= 1;
        bytes[start] = b'0' + (v % 10) as u8;
        v /= 10;
        if v == 0 {
            break;
        }
    }
    push(&bytes[start..]);
}

/// One move of an entry's move set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct YorkieMove {
    /// 16-bit move fragment (YaneuraOu `Move16`).
    pub move16: u16,
    /// Probability × 10,000 of drawing this move.
    pub weight: u16,
}

/// An entry the probe found applicable.
#[derive(Debug, Clone, Copy)]
pub struct YorkieEntry<'a> {
    /// Whether the entry came from the path-dependent section.
    pub path_dependent: bool,
    /// The move records, `n_moves` × 4 bytes.
    moves: &'a [u8],
}

impl YorkieEntry<'_> {
    /// The move set, in the file's order (descending weight).
    pub fn moves(&self) -> impl ExactSizeIterator<Item = YorkieMove> + '_ {
        self.moves
            .as_chunks::<MOVE_LEN>()
            .0
            .iter()
            .map(|m| YorkieMove {
                move16: u16_at(m, 0),
                weight: u16_at(m, 2),
            })
    }
}

/// An opened, validated yorkie book.
pub struct YorkieBook {
    data: Vec<u8>,
    n_shared: usize,
    n_pathdep: usize,
    pathdep_index_off: usize,
}

impl YorkieBook {
    /// Read the file at `path` whole and validate it.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, YorkieBookError> {
        Self::from_bytes(std::fs::read(path)?)
    }

    /// Validate an already-loaded byte image.
    pub fn from_bytes(data: Vec<u8>) -> Result<Self, YorkieBookError> {
        if data.len() < HEADER_LEN + TRAILER_LEN {
            return Err(YorkieBookError::Truncated);
        }
        if &data[..8] != MAGIC {
            return Err(YorkieBookError::BadMagic);
        }
        let body_end = data.len() - TRAILER_LEN;
        if &data[data.len() - 8..] != TRAILER_MAGIC {
            return Err(YorkieBookError::BadTrailerMagic);
        }
        let version = u16_at(&data, 8);
        if version != VERSION {
            return Err(YorkieBookError::UnsupportedVersion(version));
        }
        if sha256(&data[..body_end])[..] != data[body_end..body_end + 32] {
            return Err(YorkieBookError::ChecksumMismatch);
        }

        let n_shared = u64_at(&data, 16);
        let n_pathdep = u64_at(&data, 24);
        let pathdep_index_off = u64_at(&data, 32);
        let n_shared = index_table(n_shared, HEADER_LEN as u64, body_end)?;
        let n_pathdep = index_table(n_pathdep, pathdep_index_off, body_end)?;

        let book = YorkieBook {
            data,
            n_shared,
            n_pathdep,
            pathdep_index_off: pathdep_index_off as usize,
        };
        book.validate_section(false, body_end)?;
        book.validate_section(true, body_end)?;
        Ok(book)
    }

    /// Number of shared entries.
    pub fn shared_count(&self) -> usize {
        self.n_shared
    }

    /// Number of path-dependent entries.
    pub fn path_dependent_count(&self) -> usize {
        self.n_pathdep
    }

    /// The entry that applies to `key`, given which positions the game has
    /// already seen: the shared entry if there is one, otherwise the first
    /// path-dependent entry for `key` whose whole support list `seen` accepts.
    pub fn probe(
        &self,
        key: &[u8; 32],
        seen: impl Fn(&[u8; 32]) -> bool,
    ) -> Option<YorkieEntry<'_>> {
        if let Some(i) = self.lower_bound(false, key)
            && self.key(false, i) == key
        {
            return Some(self.entry(false, i));
        }
        let first = self.lower_bound(true, key)?;
        (first..self.n_pathdep)
            .take_while(|&i| self.key(true, i) == key)
            .find(|&i| self.support(i).all(&seen))
            .map(|i| self.entry(true, i))
    }

    /// The first index in a section whose key is not below `key`, or `None`
    /// when every key is.
    fn lower_bound(&self, pathdep: bool, key: &[u8; 32]) -> Option<usize> {
        let n = self.count(pathdep);
        let (mut lo, mut hi) = (0, n);
        while lo < hi {
            let mid = usize::midpoint(lo, hi);
            if self.key(pathdep, mid) < key {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        (lo < n).then_some(lo)
    }

    fn count(&self, pathdep: bool) -> usize {
        if pathdep {
            self.n_pathdep
        } else {
            self.n_shared
        }
    }

    /// Byte offset of entry `i` of a section.
    fn entry_off(&self, pathdep: bool, i: usize) -> usize {
        let table = if pathdep {
            self.pathdep_index_off
        } else {
            HEADER_LEN
        };
        u64_at(&self.data, table + i * 8) as usize
    }

    fn key(&self, pathdep: bool, i: usize) -> &[u8; 32] {
        let off = self.entry_off(pathdep, i);
        self.data[off..off + KEY_LEN]
            .first_chunk()
            .expect("32 bytes")
    }

    fn entry(&self, pathdep: bool, i: usize) -> YorkieEntry<'_> {
        let off = self.entry_off(pathdep, i);
        let n_moves = usize::from(u16_at(&self.data, off + N_MOVES_AT));
        let moves_off = off + ENTRY_PREFIX_LEN;
        YorkieEntry {
            path_dependent: pathdep,
            moves: &self.data[moves_off..moves_off + n_moves * MOVE_LEN],
        }
    }

    /// The support list of path-dependent entry `i`.
    fn support(&self, i: usize) -> impl Iterator<Item = &[u8; 32]> {
        let off = self.entry_off(true, i);
        let n_moves = usize::from(u16_at(&self.data, off + N_MOVES_AT));
        let n_support = usize::from(u16_at(&self.data, off + N_SUPPORT_AT));
        let start = off + ENTRY_PREFIX_LEN + n_moves * MOVE_LEN;
        self.data[start..start + n_support * KEY_LEN]
            .as_chunks::<KEY_LEN>()
            .0
            .iter()
    }

    /// Check every entry of a section lies inside the body, carries a
    /// well-formed move set and support count, and follows its predecessor in
    /// key order.
    fn validate_section(&self, pathdep: bool, body_end: usize) -> Result<(), YorkieBookError> {
        let mut previous: Option<&[u8; 32]> = None;
        for i in 0..self.count(pathdep) {
            let off = self.entry_off(pathdep, i);
            let header_end = off
                .checked_add(ENTRY_PREFIX_LEN)
                .filter(|&end| off >= HEADER_LEN && end <= body_end)
                .ok_or(YorkieBookError::OffsetOutOfRange)?;
            let h = &self.data[off..header_end];
            let n_moves = usize::from(u16_at(h, N_MOVES_AT));
            let n_support = usize::from(u16_at(h, N_SUPPORT_AT));
            let end = header_end + n_moves * MOVE_LEN + n_support * KEY_LEN;
            if end > body_end {
                return Err(YorkieBookError::OffsetOutOfRange);
            }
            if n_moves == 0 || (n_support == 0) == pathdep {
                return Err(YorkieBookError::MalformedEntry);
            }
            let moves = &self.data[header_end..header_end + n_moves * MOVE_LEN];
            let total: u32 = moves
                .as_chunks::<MOVE_LEN>()
                .0
                .iter()
                .map(|m| u32::from(u16_at(m, 2)))
                .sum();
            if total != WEIGHT_TOTAL {
                return Err(YorkieBookError::MalformedEntry);
            }

            let key = self.key(pathdep, i);
            if let Some(prev) = previous {
                let in_order = match prev.cmp(key) {
                    Ordering::Less => true,
                    Ordering::Equal => pathdep,
                    Ordering::Greater => false,
                };
                if !in_order {
                    return Err(YorkieBookError::KeysOutOfOrder);
                }
            }
            previous = Some(key);
        }
        Ok(())
    }
}

/// Check an offset table of `count` u64s at `off` lies inside the body, and
/// return its length as a `usize`.
fn index_table(count: u64, off: u64, body_end: usize) -> Result<usize, YorkieBookError> {
    let end = count
        .checked_mul(8)
        .and_then(|bytes| bytes.checked_add(off))
        .ok_or(YorkieBookError::OffsetOutOfRange)?;
    if off < HEADER_LEN as u64 || end > body_end as u64 {
        return Err(YorkieBookError::OffsetOutOfRange);
    }
    Ok(count as usize)
}

fn u16_at(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(*b[at..at + 2].first_chunk().expect("2 bytes"))
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(*b[at..at + 8].first_chunk().expect("8 bytes"))
}

/// SHA-256 (FIPS 180-4) of `bytes`, as the trailer stores it.
pub fn sha256(bytes: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    let compress = |h: &mut [u32; 8], block: &[u8; 64]| {
        let mut w = [0u32; 64];
        for (t, word) in block.as_chunks::<4>().0.iter().enumerate() {
            w[t] = u32::from_be_bytes(*word);
        }
        for t in 16..64 {
            let s0 = w[t - 15].rotate_right(7) ^ w[t - 15].rotate_right(18) ^ (w[t - 15] >> 3);
            let s1 = w[t - 2].rotate_right(17) ^ w[t - 2].rotate_right(19) ^ (w[t - 2] >> 10);
            w[t] = w[t - 16]
                .wrapping_add(s0)
                .wrapping_add(w[t - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = *h;
        for t in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[t])
                .wrapping_add(w[t]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (x, v) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *x = x.wrapping_add(v);
        }
    };

    let (blocks, rest) = bytes.as_chunks::<64>();
    for block in blocks {
        compress(&mut h, block);
    }
    let mut tail = [0u8; 128];
    tail[..rest.len()].copy_from_slice(rest);
    tail[rest.len()] = 0x80;
    let tail_len = if rest.len() < 56 { 64 } else { 128 };
    let bit_len = (bytes.len() as u64).wrapping_mul(8);
    tail[tail_len - 8..tail_len].copy_from_slice(&bit_len.to_be_bytes());
    for block in tail[..tail_len].as_chunks::<64>().0 {
        compress(&mut h, block);
    }

    let mut out = [0u8; 32];
    for (chunk, word) in out.as_chunks_mut::<4>().0.iter_mut().zip(h) {
        *chunk = word.to_be_bytes();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn sha256_matches_the_standard_vectors() {
        assert_eq!(
            hex(&sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            hex(&sha256(
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
            )),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    /// `(key, moves as (move16, weight), support)`.
    type RawEntry = ([u8; 32], Vec<(u16, u16)>, Vec<[u8; 32]>);

    /// A minimal well-formed image: entries are written in the order given.
    fn build(shared: &[RawEntry], pathdep: &[RawEntry]) -> Vec<u8> {
        let mut out = vec![0u8; HEADER_LEN];
        out[..8].copy_from_slice(MAGIC);
        out[8..10].copy_from_slice(&VERSION.to_le_bytes());
        out[16..24].copy_from_slice(&(shared.len() as u64).to_le_bytes());
        out[24..32].copy_from_slice(&(pathdep.len() as u64).to_le_bytes());
        for (is_pathdep, entries) in [(false, shared), (true, pathdep)] {
            let table = out.len();
            if is_pathdep {
                out[32..40].copy_from_slice(&(table as u64).to_le_bytes());
            }
            out.resize(table + entries.len() * 8, 0);
            for (i, (key, moves, support)) in entries.iter().enumerate() {
                let off = out.len() as u64;
                out[table + i * 8..table + i * 8 + 8].copy_from_slice(&off.to_le_bytes());
                out.extend_from_slice(key);
                out.extend_from_slice(&(moves.len() as u16).to_le_bytes());
                out.extend_from_slice(&(support.len() as u16).to_le_bytes());
                for (m, w) in moves {
                    out.extend_from_slice(&m.to_le_bytes());
                    out.extend_from_slice(&w.to_le_bytes());
                }
                for s in support {
                    out.extend_from_slice(s);
                }
            }
        }
        seal(out)
    }

    fn seal(mut body: Vec<u8>) -> Vec<u8> {
        let digest = sha256(&body);
        body.extend_from_slice(&digest);
        body.extend_from_slice(TRAILER_MAGIC);
        body
    }

    fn key(first: u8) -> [u8; 32] {
        let mut k = [0u8; 32];
        k[0] = first;
        k
    }

    fn sample() -> Vec<u8> {
        build(
            &[
                (key(1), vec![(0x11, 10_000)], vec![]),
                (key(3), vec![(0x31, 7_000), (0x32, 3_000)], vec![]),
            ],
            &[
                (key(5), vec![(0x51, 10_000)], vec![key(8), key(9)]),
                (key(5), vec![(0x52, 10_000)], vec![key(8)]),
            ],
        )
    }

    #[test]
    fn shared_and_path_dependent_entries_probe_as_specified() {
        let book = YorkieBook::from_bytes(sample()).expect("valid");
        assert_eq!(book.shared_count(), 2);
        assert_eq!(book.path_dependent_count(), 2);

        let hit = book.probe(&key(3), |_| false).expect("shared hit");
        assert!(!hit.path_dependent);
        let moves: Vec<_> = hit.moves().collect();
        assert_eq!(
            moves,
            vec![
                YorkieMove {
                    move16: 0x31,
                    weight: 7_000
                },
                YorkieMove {
                    move16: 0x32,
                    weight: 3_000
                },
            ]
        );
        assert!(book.probe(&key(2), |_| true).is_none());
        assert!(book.probe(&key(9), |_| true).is_none());

        // Both supports seen: the first entry in file order wins.
        let both = book.probe(&key(5), |_| true).expect("hit");
        assert!(both.path_dependent);
        assert_eq!(both.moves().next().unwrap().move16, 0x51);
        // Only key 8 seen: the first entry fails, the second applies.
        let one = book.probe(&key(5), |k| *k == key(8)).expect("hit");
        assert_eq!(one.moves().next().unwrap().move16, 0x52);
        // Nothing seen: no entry applies.
        assert!(book.probe(&key(5), |_| false).is_none());
    }

    fn reject(bytes: Vec<u8>) -> YorkieBookError {
        match YorkieBook::from_bytes(bytes) {
            Ok(_) => panic!("accepted a malformed book"),
            Err(e) => e,
        }
    }

    /// Re-seal `bytes` after editing the body, so only the edit is wrong.
    fn reseal(mut bytes: Vec<u8>) -> Vec<u8> {
        bytes.truncate(bytes.len() - TRAILER_LEN);
        seal(bytes)
    }

    #[test]
    fn every_structural_fault_is_refused() {
        let mut b = sample();
        b[0] = b'X';
        assert!(matches!(reject(b), YorkieBookError::BadMagic));

        let mut b = sample();
        let n = b.len();
        b[n - 1] = b'X';
        assert!(matches!(reject(b), YorkieBookError::BadTrailerMagic));

        for version in [1u8, 3] {
            let mut b = sample();
            b[8] = version;
            assert!(matches!(
                reject(reseal(b)),
                YorkieBookError::UnsupportedVersion(v) if v == u16::from(version)
            ));
        }

        let mut b = sample();
        b[HEADER_LEN + 40] ^= 1;
        assert!(matches!(reject(b), YorkieBookError::ChecksumMismatch));

        // A shared offset table claiming far more entries than the file holds.
        let mut b = sample();
        b[16] = 200;
        assert!(matches!(
            reject(reseal(b)),
            YorkieBookError::OffsetOutOfRange
        ));

        // A path-dependent offset table starting at the trailer.
        let mut b = sample();
        let at_trailer = (b.len() - TRAILER_LEN) as u64;
        b[32..40].copy_from_slice(&at_trailer.to_le_bytes());
        assert!(matches!(
            reject(reseal(b)),
            YorkieBookError::OffsetOutOfRange
        ));

        // An entry offset pointing into the trailer.
        let mut b = sample();
        let table = HEADER_LEN;
        let past = (b.len() - TRAILER_LEN - 10) as u64;
        b[table..table + 8].copy_from_slice(&past.to_le_bytes());
        assert!(matches!(
            reject(reseal(b)),
            YorkieBookError::OffsetOutOfRange
        ));

        assert!(matches!(reject(vec![0u8; 79]), YorkieBookError::Truncated));

        let swapped = build(
            &[
                (key(3), vec![(0x31, 10_000)], vec![]),
                (key(1), vec![(0x11, 10_000)], vec![]),
            ],
            &[],
        );
        assert!(matches!(reject(swapped), YorkieBookError::KeysOutOfOrder));
        let duplicate = build(
            &[
                (key(1), vec![(0x11, 10_000)], vec![]),
                (key(1), vec![(0x12, 10_000)], vec![]),
            ],
            &[],
        );
        assert!(matches!(reject(duplicate), YorkieBookError::KeysOutOfOrder));
        let pathdep_down = build(
            &[],
            &[
                (key(5), vec![(0x51, 10_000)], vec![key(1)]),
                (key(4), vec![(0x41, 10_000)], vec![key(1)]),
            ],
        );
        assert!(matches!(
            reject(pathdep_down),
            YorkieBookError::KeysOutOfOrder
        ));

        let light = build(&[(key(1), vec![(0x11, 9_999)], vec![])], &[]);
        assert!(matches!(reject(light), YorkieBookError::MalformedEntry));
        let unsupported = build(&[], &[(key(1), vec![(0x11, 10_000)], vec![])]);
        assert!(matches!(
            reject(unsupported),
            YorkieBookError::MalformedEntry
        ));
    }

    #[test]
    fn an_empty_book_is_valid_and_misses_everything() {
        let book = YorkieBook::from_bytes(build(&[], &[])).expect("valid");
        assert!(book.probe(&key(1), |_| true).is_none());
    }

    /// The format specification published beside the source.
    const FORMAT_DOC: &str = include_str!("../../../specs/yorkie-book-format.md");

    /// The bytes of the specification's example file: the code block after the
    /// heading `heading`, one line per 16 bytes, each led by its offset.
    fn documented_example(doc: &str, heading: &str) -> Vec<u8> {
        let mut lines = doc.lines().skip_while(|l| *l != heading).skip(1);
        lines
            .by_ref()
            .find(|l| l.starts_with("```"))
            .expect("a code block after the heading");
        let mut bytes = Vec::new();
        for line in lines.take_while(|l| !l.starts_with("```")) {
            let mut fields = line.split_whitespace();
            let offset = fields.next().expect("an offset");
            assert_eq!(
                usize::from_str_radix(offset, 16).expect("a hex offset"),
                bytes.len(),
                "offset column of {line:?}"
            );
            bytes.extend(fields.map(|b| u8::from_str_radix(b, 16).expect("a hex byte")));
        }
        bytes
    }

    /// `lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1`.
    const EXAMPLE_START: [u8; 32] = [
        0x58, 0xa4, 0x51, 0x22, 0x0c, 0xeb, 0x67, 0x22, 0x7e, 0x96, 0x53, 0x22, 0x1c, 0xaf, 0x44,
        0x78, 0x24, 0xc2, 0x2b, 0x11, 0x9e, 0x53, 0x22, 0x1c, 0xeb, 0x6f, 0x22, 0x3e, 0x96, 0x51,
        0x22, 0x0c,
    ];
    /// The start position after `2g2f`.
    const EXAMPLE_AFTER_2G2F: [u8; 32] = [
        0x59, 0xa4, 0x51, 0x22, 0x0c, 0xeb, 0x67, 0x12, 0x7e, 0x96, 0x53, 0x22, 0x1c, 0xaf, 0x44,
        0x78, 0x24, 0xc2, 0x2b, 0x11, 0x9e, 0x53, 0x22, 0x1c, 0xeb, 0x6f, 0x22, 0x3e, 0x96, 0x51,
        0x22, 0x0c,
    ];
    /// The start position after `7g7f`.
    const EXAMPLE_AFTER_7G7F: [u8; 32] = [
        0x59, 0xa4, 0x51, 0x22, 0x0c, 0xeb, 0x67, 0x22, 0x7e, 0x96, 0x53, 0x22, 0x1c, 0xaf, 0x44,
        0x78, 0x24, 0xc2, 0x2b, 0x11, 0x9e, 0x53, 0x12, 0x1c, 0xeb, 0x6f, 0x22, 0x3e, 0x96, 0x51,
        0x22, 0x0c,
    ];
    /// The start position after `7g7f 3c3d 2g2f`, which `2g2f 3c3d 7g7f` also
    /// reaches.
    const EXAMPLE_P: [u8; 32] = [
        0x59, 0xa4, 0x51, 0x22, 0x0c, 0xeb, 0x67, 0x12, 0x7e, 0x96, 0x93, 0x24, 0x1c, 0xaf, 0x44,
        0x78, 0x24, 0xc2, 0x2b, 0x11, 0x9e, 0x53, 0x12, 0x1c, 0xeb, 0x6f, 0x22, 0x3e, 0x96, 0x51,
        0x22, 0x0c,
    ];

    /// The specification's example, as the builder writes it: `7g7f` and
    /// `2g2f` at the start position; `8c8d` at `EXAMPLE_P` once the position
    /// after `2g2f` has occurred, else `4a3b` once the one after `7g7f` has.
    fn example() -> Vec<u8> {
        build(
            &[(
                EXAMPLE_START,
                vec![(0x1e3b, 6_000), (0x078e, 4_000)],
                vec![],
            )],
            &[
                (EXAMPLE_P, vec![(0x20c2, 10_000)], vec![EXAMPLE_AFTER_2G2F]),
                (EXAMPLE_P, vec![(0x0d93, 10_000)], vec![EXAMPLE_AFTER_7G7F]),
            ],
        )
    }

    #[test]
    fn the_documented_example_is_byte_for_byte_the_builder_output() {
        let expected = example();
        let documented = documented_example(FORMAT_DOC, "#### 16 進ダンプ");
        let dump: String = expected
            .chunks(16)
            .enumerate()
            .map(|(i, row)| {
                let bytes: Vec<String> = row.iter().map(|b| format!("{b:02x}")).collect();
                format!("{:08x}  {}\n", i * 16, bytes.join(" "))
            })
            .collect();
        assert_eq!(documented, expected, "the dump should read:\n{dump}");
    }

    #[test]
    fn the_documented_example_answers_as_its_entries_describe() {
        let book = YorkieBook::from_bytes(documented_example(FORMAT_DOC, "#### 16 進ダンプ"))
            .expect("valid");
        assert_eq!((book.shared_count(), book.path_dependent_count()), (1, 2));

        let start = book.probe(&EXAMPLE_START, |_| false).expect("shared hit");
        assert!(!start.path_dependent);
        let moves: Vec<_> = start.moves().map(|m| (m.move16, m.weight)).collect();
        assert_eq!(moves, vec![(0x1e3b, 6_000), (0x078e, 4_000)]);

        let via_2g2f = [EXAMPLE_START, EXAMPLE_AFTER_2G2F, EXAMPLE_P];
        let hit = book
            .probe(&EXAMPLE_P, |k| via_2g2f.contains(k))
            .expect("first entry applies");
        assert!(hit.path_dependent);
        assert_eq!(hit.moves().next().map(|m| m.move16), Some(0x20c2));

        let via_7g7f = [EXAMPLE_START, EXAMPLE_AFTER_7G7F, EXAMPLE_P];
        let hit = book
            .probe(&EXAMPLE_P, |k| via_7g7f.contains(k))
            .expect("second entry applies");
        assert_eq!(hit.moves().next().map(|m| m.move16), Some(0x0d93));

        assert!(book.probe(&EXAMPLE_P, |k| *k == EXAMPLE_P).is_none());
        assert!(book.probe(&EXAMPLE_AFTER_7G7F, |_| true).is_none());
    }

    #[test]
    fn the_format_document_has_a_section_for_the_version_read() {
        let heading = format!("## `YKBOOK01` バージョン {VERSION}");
        assert!(
            FORMAT_DOC.lines().any(|l| l == heading),
            "specs/yorkie-book-format.md has no heading {heading:?}"
        );
    }
}
