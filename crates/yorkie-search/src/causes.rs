//! The positions a path-dependent value rests on (`verbose3`).
//!
//! A value carries the `pathdep` mark when it was derived through a repetition
//! judgement (see [`yorkie_storage::ValueMarks`]). Beside the mark the search
//! carries the *causes*: where on the game's line stand the earlier positions
//! those judgements matched, as [`Position::timeline_index`] values — `0` for
//! the position the `position` command set up, `n` for the one after its
//! `n`-th move. They travel exactly as the mark does — out of the judgement,
//! up through every child that moved its parent's window, and through a
//! table cutoff — so the causes a reply reports belong to exactly the marks
//! that reached its root.
//!
//! A node keeps only the causes that stand before it on the line: a
//! repetition whose earlier occurrence is the node itself or lies below it
//! closed inside the node's own subtree, and the node's value does not depend
//! on how the node was reached because of it. At the root that is what makes a
//! repetition closed entirely inside the search tree list nothing.
//!
//! A set holds at most [`CAUSE_CAPACITY`] positions. Where the causes are not
//! known — the set overflowed, or a value came out of a table entry whose own
//! causes cannot be recovered — the set is *open*: the positions it lists are
//! causes, but there may be others.
//!
//! **No search decision reads them.** They exist only to be reported.
//!
//! [`Position::timeline_index`]: yorkie_state::Position::timeline_index

use yorkie_state::Position;
use yorkie_storage::ValueMarks;

/// What the value a reply rests on was derived through: its marks, and the
/// positions before the root its `pathdep` mark rests on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplyMarks {
    /// The value's marks.
    pub marks: ValueMarks,
    /// The causes of `marks.path_dep`, as line indices of the game before the
    /// root.
    pub causes: PathCauses,
}

impl ReplyMarks {
    /// No mark and no cause.
    pub const NONE: ReplyMarks = ReplyMarks {
        marks: ValueMarks::NONE,
        causes: PathCauses::NONE,
    };
}

impl From<ValueMarks> for ReplyMarks {
    fn from(marks: ValueMarks) -> Self {
        ReplyMarks {
            marks,
            causes: PathCauses::NONE,
        }
    }
}

/// How many positions one value's set of causes holds before it is open.
pub const CAUSE_CAPACITY: usize = 8;

/// The earlier positions of the game a path-dependent value rests on: a sorted
/// set of line indices, and whether there may be causes it does not list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PathCauses {
    plies: [u32; CAUSE_CAPACITY],
    len: u8,
    open: bool,
}

impl Default for PathCauses {
    fn default() -> Self {
        Self::NONE
    }
}

impl PathCauses {
    /// No cause, and none unknown: what a value no repetition judgement
    /// reached carries.
    pub const NONE: PathCauses = PathCauses {
        plies: [0; CAUSE_CAPACITY],
        len: 0,
        open: false,
    };

    /// No known cause, and possibly some unknown ones.
    pub const OPEN: PathCauses = PathCauses {
        open: true,
        ..PathCauses::NONE
    };

    /// The listed causes, ascending.
    pub fn plies(&self) -> &[u32] {
        &self.plies[..usize::from(self.len)]
    }

    /// Whether there may be causes the set does not list.
    pub fn is_open(&self) -> bool {
        self.open
    }

    /// Mark the set as possibly incomplete.
    pub fn set_open(&mut self) {
        self.open = true;
    }

    /// Add the position at line index `at`. A set already full, or an index
    /// too large to hold, opens instead.
    pub fn insert(&mut self, at: usize) {
        let Ok(at) = u32::try_from(at) else {
            self.open = true;
            return;
        };
        let len = usize::from(self.len);
        let pos = match self.plies[..len].binary_search(&at) {
            Ok(_) => return,
            Err(pos) => pos,
        };
        if len == CAUSE_CAPACITY {
            self.open = true;
            return;
        }
        self.plies.copy_within(pos..len, pos + 1);
        self.plies[pos] = at;
        self.len += 1;
    }

    /// Fold in `other`'s causes that stand before line index `below`, and its
    /// openness.
    pub fn merge_below(&mut self, other: &PathCauses, below: usize) {
        for &at in other.plies() {
            if (at as usize) < below {
                self.insert(at as usize);
            }
        }
        self.open |= other.open;
    }

    /// This set without the causes at or after line index `below`.
    pub fn below(&self, below: usize) -> PathCauses {
        let mut out = PathCauses {
            open: self.open,
            ..PathCauses::NONE
        };
        out.merge_below(self, below);
        out
    }

    /// The cause a repetition verdict at `pos` over a `look_back`-ply window
    /// rests on: the earlier occurrence it matched, or an open set if that
    /// cannot be found (which a verdict other than "none" rules out). With
    /// `inferior` unset, the verdict is one that leaves the inferior verdict
    /// out.
    pub fn of_repetition(pos: &Position, look_back: u32, inferior: bool) -> PathCauses {
        let mut out = PathCauses::NONE;
        match pos.repetition_origin(look_back, inferior) {
            Some(at) => out.insert(at),
            None => out.open = true,
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_keeps_the_set_sorted_and_unique() {
        let mut c = PathCauses::NONE;
        for at in [7, 3, 9, 3, 0] {
            c.insert(at);
        }
        assert_eq!(c.plies(), &[0, 3, 7, 9]);
        assert!(!c.is_open());
    }

    #[test]
    fn a_full_set_opens_rather_than_dropping_silently() {
        let mut c = PathCauses::NONE;
        for at in 0..CAUSE_CAPACITY {
            c.insert(at * 2);
        }
        assert!(!c.is_open());
        c.insert(1);
        assert!(c.is_open());
        assert_eq!(c.plies().len(), CAUSE_CAPACITY);
        // A cause already listed is no overflow.
        let mut d = PathCauses::NONE;
        for at in 0..CAUSE_CAPACITY {
            d.insert(at);
        }
        d.insert(0);
        assert!(!d.is_open());
    }

    #[test]
    fn merge_below_keeps_only_the_earlier_causes_and_the_openness() {
        let mut child = PathCauses::NONE;
        for at in [2, 5, 6, 10] {
            child.insert(at);
        }
        child.set_open();
        let mut node = PathCauses::NONE;
        node.insert(1);
        node.merge_below(&child, 6);
        assert_eq!(node.plies(), &[1, 2, 5]);
        assert!(node.is_open());
        assert_eq!(child.below(3).plies(), &[2]);
        assert!(child.below(0).is_open());
    }
}
