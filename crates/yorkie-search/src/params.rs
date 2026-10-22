//! The search parameters a tuning run varies: the pruning and reduction
//! constants that shape the tree, each named once here and read through
//! `Param::get` wherever the search uses it.
//!
//! In every build but `tune` a parameter is a `const`, and `get` on it is the
//! constant: the search compiles to what a literal at the use site would give.
//! In a `tune` build each one is a `static` that `apply` writes once at
//! startup from a `name=value,name=value` list, before any search thread
//! exists; the names left out keep the defaults below, so a `tune` build given
//! no list searches exactly as the plain build of the same config.
//!
//! Each parameter carries the range of values the search tolerates: the
//! arithmetic it enters stays inside `i32` over the whole depth range, and the
//! invariant the code around it relies on still holds. That range is wider than
//! any range worth tuning over; it only keeps a mistyped value from producing a
//! search that hangs or overflows.

#[cfg(feature = "tune")]
use std::sync::atomic::{AtomicI32, Ordering};

#[cfg(feature = "tune")]
use yorkie_state::text::{TextWriter, atoi_i64};

/// One search parameter, a compile-time constant.
#[cfg(not(feature = "tune"))]
#[derive(Clone, Copy)]
pub(crate) struct Param(i32);

#[cfg(not(feature = "tune"))]
impl Param {
    const fn new(value: i32) -> Self {
        Self(value)
    }

    #[inline(always)]
    pub(crate) const fn get(self) -> i32 {
        self.0
    }
}

/// One search parameter, written at startup by [`apply`].
///
/// A relaxed load is enough to read it: the only store happens on the main
/// thread before any search thread is spawned, and spawning a thread
/// synchronises with everything the spawning thread did before it, so every
/// search thread observes the stored value. Nothing writes it afterwards.
#[cfg(feature = "tune")]
pub(crate) struct Param(AtomicI32);

#[cfg(feature = "tune")]
impl Param {
    const fn new(value: i32) -> Self {
        Self(AtomicI32::new(value))
    }

    #[inline(always)]
    pub(crate) fn get(&self) -> i32 {
        self.0.load(Ordering::Relaxed)
    }
}

/// One registered parameter: its name in a tuning list, the parameter itself,
/// and the values the search tolerates.
#[cfg(feature = "tune")]
struct Entry {
    name: &'static [u8],
    param: &'static Param,
    min: i32,
    max: i32,
}

macro_rules! search_params {
    ($(
        $(#[doc = $doc:literal])*
        $id:ident = $name:literal, $default:literal, $min:literal..=$max:literal;
    )+) => {
        $(
            $(#[doc = $doc])*
            #[cfg(not(feature = "tune"))]
            pub(crate) const $id: Param = Param::new($default);
            $(#[doc = $doc])*
            #[cfg(feature = "tune")]
            pub(crate) static $id: Param = Param::new($default);
            const _: () = assert!($min <= $default && $default <= $max);
        )+

        /// Every parameter a tuning list can name.
        #[cfg(feature = "tune")]
        static REGISTRY: &[Entry] = &[$(
            Entry { name: $name, param: &$id, min: $min, max: $max },
        )+];
    };
}

search_params! {
    /// Razoring: the margin below alpha at depth 0, before the depth term.
    RAZOR_BASE = b"razor_base", 502, 0..=2000;
    /// Razoring: the margin's coefficient on the squared depth.
    RAZOR_DEPTH = b"razor_depth", 306, 0..=1000;
    /// Futility pruning: the per-ply margin with a transposition-table hit. The
    /// margin without one is this less 21, so the floor keeps it positive.
    FUTILITY_MULT = b"futility_mult", 76, 22..=500;
    /// Null-move pruning: how far the static evaluation may sit below beta,
    /// before the depth and improving terms, for the null move to be tried.
    NMP_MARGIN = b"nmp_margin", 378, 0..=2000;
    /// Null-move pruning: the depth reduction before the `depth / 3` term. At
    /// least one, so the null-move search is shallower than the node.
    NMP_REDUCTION = b"nmp_reduction", 7, 1..=20;
    /// ProbCut: how far above beta the cut is sought, before the improving
    /// term, which takes 61 off it. The floor keeps the ProbCut beta above beta.
    PROBCUT_MARGIN = b"probcut_margin", 224, 62..=2000;
    /// The transposition-table ProbCut at the head of the move loop: how far
    /// above beta a stored lower bound must sit to cut the node.
    SMALL_PROBCUT_MARGIN = b"small_probcut_margin", 416, 0..=2000;
    /// Singular extension: the singular beta's margin below the stored value,
    /// per 55 plies of depth, before the off-PV term.
    SINGULAR_MARGIN = b"singular_margin", 60, 0..=1000;
    /// Late-move reduction: the constant term of the base reduction, in
    /// 1024ths of a ply.
    LMR_BASE = b"lmr_base", 1133, -5000..=10000;
    /// Late-move reduction: the weight of the window's width against the root
    /// window's, subtracted from the base reduction.
    LMR_DELTA = b"lmr_delta", 585, 0..=5000;
    /// Late-move reduction: the extra reduction at a cut node, in 1024ths of a
    /// ply.
    LMR_CUT_NODE = b"lmr_cut_node", 3611, 0..=10000;
    /// Quiet-move futility pruning at shallow depth: the margin per ply of the
    /// reduced depth.
    QUIET_FUTILITY_DEPTH = b"quiet_futility_depth", 120, 0..=1000;
    /// Quiescence search: the futility margin over the static evaluation.
    QS_FUTILITY_MARGIN = b"qs_futility_margin", 328, 0..=2000;
    /// Quiescence search: the static-exchange threshold below which a capture
    /// is skipped.
    QS_SEE_MARGIN = b"qs_see_margin", -73, -2000..=2000;
    /// The aspiration window's half-width before the root score's spread is
    /// added. At least one, so each fail widens the window.
    ASPIRATION_DELTA = b"aspiration_delta", 5, 1..=1000;
}

/// Why a tuning list was refused. Each variant borrows the part of the list it
/// is about.
#[cfg(feature = "tune")]
#[derive(Debug, PartialEq, Eq)]
pub enum TuneError<'a> {
    /// An item that is not `name=value`.
    Malformed(&'a [u8]),
    /// A name no parameter has.
    UnknownName(&'a [u8]),
    /// A value that is not a decimal integer in the parameter's range.
    BadValue { name: &'a [u8], value: &'a [u8] },
    /// A name given more than once.
    Repeated(&'a [u8]),
}

#[cfg(feature = "tune")]
impl TuneError<'_> {
    /// The refusal as one line of text.
    pub fn write_message(&self, out: &mut TextWriter<'_>) {
        match *self {
            Self::Malformed(item) => {
                out.bytes(b"not name=value: `").bytes(item).bytes(b"`");
            }
            Self::UnknownName(name) => {
                out.bytes(b"unknown parameter `").bytes(name).bytes(b"`");
            }
            Self::BadValue { name, value } => {
                out.bytes(b"bad value `").bytes(value).bytes(b"` for `");
                out.bytes(name).bytes(b"`");
                if let Some(entry) = REGISTRY.iter().find(|e| e.name == name) {
                    out.bytes(b" (an integer in ").i64(i64::from(entry.min));
                    out.bytes(b"..=").i64(i64::from(entry.max)).bytes(b")");
                }
            }
            Self::Repeated(name) => {
                out.bytes(b"parameter `")
                    .bytes(name)
                    .bytes(b"` given twice");
            }
        }
    }
}

/// The assignments a tuning list spells, as indices into the registry with
/// their values, or the first thing wrong with it. An empty list assigns
/// nothing; spaces around an item, a name or a value are ignored.
#[cfg(feature = "tune")]
fn parse(list: &[u8]) -> Result<Vec<(usize, i32)>, TuneError<'_>> {
    let mut assignments: Vec<(usize, i32)> = Vec::new();
    if list.trim_ascii().is_empty() {
        return Ok(assignments);
    }
    for item in list.split(|&b| b == b',') {
        let Some(eq) = item.iter().position(|&b| b == b'=') else {
            return Err(TuneError::Malformed(item.trim_ascii()));
        };
        let name = item[..eq].trim_ascii();
        let value = item[eq + 1..].trim_ascii();
        if name.is_empty() {
            return Err(TuneError::Malformed(item.trim_ascii()));
        }
        let Some(index) = REGISTRY.iter().position(|e| e.name == name) else {
            return Err(TuneError::UnknownName(name));
        };
        let entry = &REGISTRY[index];
        let parsed = atoi_i64(value)
            .and_then(|v| i32::try_from(v).ok())
            .filter(|v| (entry.min..=entry.max).contains(v));
        let Some(parsed) = parsed else {
            return Err(TuneError::BadValue { name, value });
        };
        if assignments.iter().any(|&(i, _)| i == index) {
            return Err(TuneError::Repeated(name));
        }
        assignments.push((index, parsed));
    }
    Ok(assignments)
}

/// Set the parameters a tuning list names, leaving the rest at their defaults.
/// Nothing is written unless the whole list is good.
///
/// Called once, on the main thread, before any search thread is spawned —
/// `Param` says why that is what makes the values visible to the search.
#[cfg(feature = "tune")]
pub fn apply(list: &[u8]) -> Result<(), TuneError<'_>> {
    for (index, value) in parse(list)? {
        REGISTRY[index].param.0.store(value, Ordering::Relaxed);
    }
    Ok(())
}

#[cfg(test)]
#[cfg(feature = "tune")]
mod tests {
    use super::*;

    fn index(name: &[u8]) -> usize {
        REGISTRY.iter().position(|e| e.name == name).unwrap()
    }

    #[test]
    fn a_good_list_assigns_every_named_parameter() {
        let got = parse(b"razor_base=480, qs_see_margin=-60 ,aspiration_delta=+7").unwrap();
        assert_eq!(
            got,
            vec![
                (index(b"razor_base"), 480),
                (index(b"qs_see_margin"), -60),
                (index(b"aspiration_delta"), 7),
            ]
        );
    }

    #[test]
    fn an_empty_list_assigns_nothing() {
        assert_eq!(parse(b""), Ok(vec![]));
        assert_eq!(parse(b"  "), Ok(vec![]));
    }

    #[test]
    fn an_unknown_name_is_refused() {
        assert_eq!(
            parse(b"razor_base=480,razor_bias=3"),
            Err(TuneError::UnknownName(b"razor_bias"))
        );
    }

    #[test]
    fn a_bad_value_is_refused() {
        for (list, value) in [
            (&b"razor_base=4.5"[..], &b"4.5"[..]),
            (b"razor_base=", b""),
            (b"razor_base=abc", b"abc"),
            (b"razor_base=99999999999", b"99999999999"),
            // Outside the range the search tolerates.
            (b"razor_base=2001", b"2001"),
            (b"aspiration_delta=0", b"0"),
        ] {
            let name: &[u8] = &list[..list.iter().position(|&b| b == b'=').unwrap()];
            assert_eq!(
                parse(list),
                Err(TuneError::BadValue { name, value }),
                "{}",
                String::from_utf8_lossy(list)
            );
        }
    }

    #[test]
    fn a_malformed_or_repeated_item_is_refused() {
        assert_eq!(
            parse(b"razor_base"),
            Err(TuneError::Malformed(b"razor_base"))
        );
        assert_eq!(parse(b"razor_base=1,"), Err(TuneError::Malformed(b"")));
        assert_eq!(parse(b"=5"), Err(TuneError::Malformed(b"=5")));
        assert_eq!(
            parse(b"razor_base=1,razor_base=2"),
            Err(TuneError::Repeated(b"razor_base"))
        );
    }

    #[test]
    fn a_refusal_names_what_was_wrong() {
        let mut bytes = [0u8; 256];
        let mut out = TextWriter::new(&mut bytes);
        parse(b"aspiration_delta=0")
            .unwrap_err()
            .write_message(&mut out);
        assert_eq!(
            out.as_bytes(),
            b"bad value `0` for `aspiration_delta` (an integer in 1..=1000)"
        );
    }

    #[test]
    fn the_names_are_distinct() {
        for (i, a) in REGISTRY.iter().enumerate() {
            assert!(REGISTRY[..i].iter().all(|b| b.name != a.name));
        }
    }
}
