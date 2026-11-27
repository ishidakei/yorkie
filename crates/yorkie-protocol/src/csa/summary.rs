//! The game summary: the block a server sends between `BEGIN Game_Summary` and
//! `END Game_Summary` to offer a game, read into a [`GameSummary`].
//!
//! The block is a list of `Key:Value` lines with nested sections. What is read:
//! `Protocol_Version`, `Format`, `Game_ID`, `Name+`, `Name-`, `Your_Turn`,
//! `To_Move` and `Max_Moves` at the top; the time section `BEGIN Time` …
//! `END Time`, which applies to both sides, and `BEGIN Time+` / `BEGIN Time-`,
//! which apply to one side and win over the shared one key by key; and the
//! position section `BEGIN Position` … `END Position`, with the moves already
//! played after its side-to-move line. A key the reader does not know is
//! skipped, and so is a section it does not know, nested or not.

use yorkie_state::text::atoi_u64;
use yorkie_state::{Color, CsaError, Move, Position, parse_csa_move, parse_csa_position_into};

/// Longest a summary may be, in bytes. A real one is well under a kilobyte, and
/// one carrying a whole game's moves stays far below this.
pub(crate) const MAX_SUMMARY_BYTES: usize = 64 * 1024;

/// One side's time control, in the server's own unit ([`Self::unit_ms`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SideTime {
    /// Milliseconds per unit: `Time_Unit:1sec` is 1000.
    pub(crate) unit_ms: u64,
    /// `Total_Time`: the main clock at the start of the game.
    pub(crate) total: u64,
    /// `Byoyomi`: the time a move may take once the main clock is spent.
    pub(crate) byoyomi: u64,
    /// `Increment`: what each move adds to the mover's clock.
    pub(crate) increment: u64,
    /// `Least_Time_Per_Move`: the least a move is charged, however fast.
    pub(crate) least: u64,
    /// `Remaining_Time+` / `Remaining_Time-`: the clock where a summary resumes
    /// a game already under way, which then stands in for [`Self::total`].
    pub(crate) remaining: Option<u64>,
}

impl SideTime {
    /// No time limit at all: what a summary that names no time gives.
    const UNTIMED: Self = Self {
        unit_ms: 1000,
        total: 0,
        byoyomi: 0,
        increment: 0,
        least: 0,
        remaining: None,
    };
}

/// The time keys a section gave, before the shared and the per-side sections
/// are merged.
#[derive(Clone, Copy, Default)]
struct TimeKeys {
    unit_ms: Option<u64>,
    total: Option<u64>,
    byoyomi: Option<u64>,
    increment: Option<u64>,
    least: Option<u64>,
}

impl TimeKeys {
    /// These keys over `base`, each one given here winning.
    fn over(self, base: Self) -> Self {
        Self {
            unit_ms: self.unit_ms.or(base.unit_ms),
            total: self.total.or(base.total),
            byoyomi: self.byoyomi.or(base.byoyomi),
            increment: self.increment.or(base.increment),
            least: self.least.or(base.least),
        }
    }

    fn resolve(self, remaining: Option<u64>) -> SideTime {
        let d = SideTime::UNTIMED;
        SideTime {
            unit_ms: self.unit_ms.unwrap_or(d.unit_ms),
            total: self.total.unwrap_or(d.total),
            byoyomi: self.byoyomi.unwrap_or(d.byoyomi),
            increment: self.increment.unwrap_or(d.increment),
            least: self.least.unwrap_or(d.least),
            remaining,
        }
    }
}

/// A game the server offers, as its summary describes it.
#[derive(Clone, Debug)]
pub(crate) struct GameSummary {
    pub(crate) protocol_version: Vec<u8>,
    pub(crate) format: Vec<u8>,
    pub(crate) game_id: Vec<u8>,
    /// `Name+` and `Name-`, indexed by [`Color::index`].
    pub(crate) names: [Vec<u8>; 2],
    /// The side this client plays.
    pub(crate) your_turn: Color,
    pub(crate) max_moves: Option<u64>,
    /// Each side's time control, indexed by [`Color::index`].
    pub(crate) time: [SideTime; 2],
    /// The position the game starts from, before [`Self::moves`].
    pub(crate) start: Position,
    /// The moves already played from [`Self::start`], each with the time the
    /// server listed for it.
    pub(crate) moves: Vec<(Move, Option<u64>)>,
}

/// What made a summary unreadable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SummaryError {
    /// A required key is missing; the name of it.
    Missing(&'static [u8]),
    /// A key whose value is not of the form it takes; the name of it.
    BadValue(&'static [u8]),
    /// A section that is never closed, or an `END` closing no open section.
    Unbalanced,
    /// No position section.
    NoPosition,
    /// The position section does not spell a position.
    Position(CsaError),
    /// A move after the position that it does not allow; its index from zero.
    Move(usize, CsaError),
    /// `To_Move` disagrees with the position section's side-to-move line.
    ToMoveMismatch,
    /// The summary is longer than [`MAX_SUMMARY_BYTES`].
    TooLong,
}

impl SummaryError {
    /// This error's message, as a log line carries it.
    pub(crate) fn write_message(&self, out: &mut yorkie_state::TextWriter<'_>) {
        match self {
            Self::Missing(key) => {
                out.bytes(b"missing ").bytes(key);
            }
            Self::BadValue(key) => {
                out.bytes(b"bad value for ").bytes(key);
            }
            Self::Unbalanced => {
                out.bytes(b"unbalanced BEGIN / END");
            }
            Self::NoPosition => {
                out.bytes(b"no position section");
            }
            Self::Position(e) => e.write_message(out),
            Self::Move(i, e) => {
                out.bytes(b"move ").u64(*i as u64 + 1).bytes(b": ");
                e.write_message(out);
            }
            Self::ToMoveMismatch => {
                out.bytes(b"To_Move disagrees with the position");
            }
            Self::TooLong => {
                out.bytes(b"summary too long");
            }
        }
    }
}

/// The lines of `block` with their byte offsets, each without its `\n` or
/// `\r\n`.
fn lines_with_offsets(block: &[u8]) -> impl Iterator<Item = (usize, usize, &[u8])> {
    let mut at = 0usize;
    std::iter::from_fn(move || {
        if at >= block.len() {
            return None;
        }
        let start = at;
        let end = block[at..]
            .iter()
            .position(|&b| b == b'\n')
            .map_or(block.len(), |i| at + i);
        at = end + 1;
        let line = &block[start..end];
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        Some((start, end.min(block.len()), line))
    })
}

/// `+` or `-` as the side it names.
pub(crate) fn parse_side(text: &[u8]) -> Option<Color> {
    match text {
        b"+" => Some(Color::Black),
        b"-" => Some(Color::White),
        _ => None,
    }
}

/// A `Time_Unit` value — a count and `msec`, `sec` or `min` — in milliseconds.
pub(crate) fn parse_time_unit(text: &[u8]) -> Option<u64> {
    let digits = text.iter().take_while(|b| b.is_ascii_digit()).count();
    let count = if digits == 0 {
        1
    } else {
        atoi_u64(&text[..digits])?
    };
    let per = match &text[digits..] {
        b"msec" => 1,
        b"sec" => 1000,
        b"min" => 60_000,
        _ => return None,
    };
    let ms = count.checked_mul(per)?;
    (ms > 0).then_some(ms)
}

/// Which section a line sits in.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Section {
    Top,
    /// `BEGIN Time`, or one side's, indexed by [`Color::index`].
    Time(Option<Color>),
    /// A section this reader does not know, at this nesting depth below the
    /// section that holds it.
    Skipped(usize),
}

/// Read a summary block — the lines between `BEGIN Game_Summary` and
/// `END Game_Summary`, neither of them included.
pub(crate) fn parse_game_summary(block: &[u8]) -> Result<GameSummary, SummaryError> {
    if block.len() > MAX_SUMMARY_BYTES {
        return Err(SummaryError::TooLong);
    }
    let mut protocol_version = Vec::new();
    let mut format = Vec::new();
    let mut game_id = None;
    let mut names = [Vec::new(), Vec::new()];
    let mut your_turn = None;
    let mut to_move = None;
    let mut max_moves = None;
    let mut shared = TimeKeys::default();
    let mut per_side = [TimeKeys::default(); 2];
    let mut remaining = [None; 2];
    let mut position = None;

    let mut section = Section::Top;
    let mut lines = lines_with_offsets(block);
    while let Some((_, end, line)) = lines.next() {
        if line.is_empty() || line[0] == b'\'' {
            continue;
        }
        if let Section::Skipped(depth) = section {
            if line.starts_with(b"BEGIN ") {
                section = Section::Skipped(depth + 1);
            } else if line.starts_with(b"END ") {
                section = match depth {
                    0 => Section::Top,
                    d => Section::Skipped(d - 1),
                };
            }
            continue;
        }
        if let Some(name) = line.strip_prefix(b"BEGIN ") {
            if section != Section::Top {
                // Nothing known nests inside a time section.
                section = Section::Skipped(0);
                continue;
            }
            section = match name {
                b"Time" => Section::Time(None),
                b"Time+" => Section::Time(Some(Color::Black)),
                b"Time-" => Section::Time(Some(Color::White)),
                b"Position" => {
                    let body_start = end + 1;
                    let mut body_end = None;
                    for (start, _, inner) in lines.by_ref() {
                        if inner == b"END Position" {
                            body_end = Some(start);
                            break;
                        }
                    }
                    let body_end = body_end.ok_or(SummaryError::Unbalanced)?;
                    position = Some(&block[body_start.min(body_end)..body_end]);
                    Section::Top
                }
                _ => Section::Skipped(0),
            };
            continue;
        }
        if line.starts_with(b"END ") {
            if section == Section::Top {
                return Err(SummaryError::Unbalanced);
            }
            section = Section::Top;
            continue;
        }
        let Some(colon) = line.iter().position(|&b| b == b':') else {
            continue;
        };
        let (key, value) = (&line[..colon], &line[colon + 1..]);
        match section {
            Section::Top => match key {
                b"Protocol_Version" => protocol_version = value.to_vec(),
                b"Format" => format = value.to_vec(),
                b"Game_ID" => game_id = Some(value.to_vec()),
                b"Name+" => names[0] = value.to_vec(),
                b"Name-" => names[1] = value.to_vec(),
                b"Your_Turn" => {
                    your_turn =
                        Some(parse_side(value).ok_or(SummaryError::BadValue(b"Your_Turn"))?);
                }
                b"To_Move" => {
                    to_move = Some(parse_side(value).ok_or(SummaryError::BadValue(b"To_Move"))?);
                }
                b"Max_Moves" => {
                    max_moves = Some(atoi_u64(value).ok_or(SummaryError::BadValue(b"Max_Moves"))?);
                }
                _ => {}
            },
            Section::Time(side) => {
                let keys = match side {
                    None => &mut shared,
                    Some(c) => &mut per_side[c.index()],
                };
                read_time_key(keys, &mut remaining, key, value)?;
            }
            Section::Skipped(_) => unreachable!("skipped sections are consumed above"),
        }
    }
    if section != Section::Top {
        return Err(SummaryError::Unbalanced);
    }

    let game_id = game_id.ok_or(SummaryError::Missing(b"Game_ID"))?;
    let your_turn = your_turn.ok_or(SummaryError::Missing(b"Your_Turn"))?;
    let position = position.ok_or(SummaryError::NoPosition)?;
    let mut start = Position::startpos();
    let rest = parse_csa_position_into(&mut start, position).map_err(SummaryError::Position)?;
    if to_move.is_some_and(|c| c != start.side_to_move()) {
        return Err(SummaryError::ToMoveMismatch);
    }
    let moves = read_moves(&start, rest)?;
    let time = [
        per_side[0].over(shared).resolve(remaining[0]),
        per_side[1].over(shared).resolve(remaining[1]),
    ];
    Ok(GameSummary {
        protocol_version,
        format,
        game_id,
        names,
        your_turn,
        max_moves,
        time,
        start,
        moves,
    })
}

/// One `Key:Value` line of a time section, into `keys` (or, for a remaining
/// time, into the side it names).
fn read_time_key(
    keys: &mut TimeKeys,
    remaining: &mut [Option<u64>; 2],
    key: &[u8],
    value: &[u8],
) -> Result<(), SummaryError> {
    let number = |name: &'static [u8]| atoi_u64(value).ok_or(SummaryError::BadValue(name));
    match key {
        b"Time_Unit" => {
            keys.unit_ms =
                Some(parse_time_unit(value).ok_or(SummaryError::BadValue(b"Time_Unit"))?);
        }
        b"Total_Time" => keys.total = Some(number(b"Total_Time")?),
        b"Byoyomi" => keys.byoyomi = Some(number(b"Byoyomi")?),
        b"Increment" => keys.increment = Some(number(b"Increment")?),
        b"Least_Time_Per_Move" => keys.least = Some(number(b"Least_Time_Per_Move")?),
        b"Remaining_Time+" => remaining[0] = Some(number(b"Remaining_Time+")?),
        b"Remaining_Time-" => remaining[1] = Some(number(b"Remaining_Time-")?),
        _ => {}
    }
    Ok(())
}

/// Split a move line such as `+7776FU,T12` into the move and its time. A line
/// with no `,T` part has no time; anything after the move other than a `,T`
/// and a number is `None`.
pub(crate) fn split_move_line(line: &[u8]) -> Option<(&[u8], Option<u64>)> {
    match line.iter().position(|&b| b == b',') {
        None => Some((line, None)),
        Some(i) => {
            let time = line[i + 1..].strip_prefix(b"T")?;
            Some((&line[..i], Some(atoi_u64(time)?)))
        }
    }
}

/// The moves listed after the position's side-to-move line, played from
/// `start` to check each is legal where it stands.
fn read_moves(start: &Position, rest: &[u8]) -> Result<Vec<(Move, Option<u64>)>, SummaryError> {
    let mut pos = start.clone();
    let mut moves = Vec::new();
    for (_, _, line) in lines_with_offsets(rest) {
        if line.is_empty() || line[0] == b'\'' {
            continue;
        }
        let i = moves.len();
        let (text, time) =
            split_move_line(line).ok_or(SummaryError::Move(i, CsaError::MalformedLine))?;
        let m = parse_csa_move(text, &pos).map_err(|e| SummaryError::Move(i, e))?;
        pos.do_move(m);
        moves.push((m, time));
    }
    Ok(moves)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The summary `shogi-server` proposes to the side playing first, byte for
    /// byte the shape it writes.
    const SHOGI_SERVER: &str = "\
Protocol_Version:1.2
Protocol_Mode:Server
Format:Shogi 1.0
Declaration:Jishogi 1.1
Game_ID:wdoor+floodgate-300-10F+a+b+20261127120000
Name+:alice
Name-:bob
Your_Turn:-
Rematch_On_Draw:NO
To_Move:+
Max_Moves:256
BEGIN Time
Time_Unit:1sec
Total_Time:300
Byoyomi:0
Increment:10
Least_Time_Per_Move:0
END Time
BEGIN Position
P1-KY-KE-GI-KI-OU-KI-GI-KE-KY
P2 * -HI *  *  *  *  * -KA *
P3-FU-FU-FU-FU-FU-FU-FU-FU-FU
P4 *  *  *  *  *  *  *  *  *
P5 *  *  *  *  *  *  *  *  *
P6 *  *  *  *  *  *  *  *  *
P7+FU+FU+FU+FU+FU+FU+FU+FU+FU
P8 * +KA *  *  *  *  * +HI *
P9+KY+KE+GI+KI+OU+KI+GI+KE+KY
+
END Position
";

    fn parse(text: &str) -> Result<GameSummary, SummaryError> {
        parse_game_summary(text.as_bytes())
    }

    #[test]
    fn reads_the_summary_shogi_server_proposes() {
        let s = parse(SHOGI_SERVER).unwrap();
        assert_eq!(s.protocol_version, b"1.2");
        assert_eq!(s.format, b"Shogi 1.0");
        assert_eq!(s.game_id, b"wdoor+floodgate-300-10F+a+b+20261127120000");
        assert_eq!(s.names, [b"alice".to_vec(), b"bob".to_vec()]);
        assert_eq!(s.your_turn, Color::White);
        assert_eq!(s.start.side_to_move(), Color::Black);
        assert_eq!(s.max_moves, Some(256));
        let side = SideTime {
            unit_ms: 1000,
            total: 300,
            byoyomi: 0,
            increment: 10,
            least: 0,
            remaining: None,
        };
        assert_eq!(s.time, [side, side]);
        assert_eq!(s.start, Position::startpos());
        assert!(s.moves.is_empty());
    }

    #[test]
    fn per_side_sections_win_over_the_shared_one_key_by_key() {
        let text = "\
Game_ID:g
Your_Turn:+
BEGIN Time
Time_Unit:1sec
Byoyomi:5
END Time
BEGIN Time+
Total_Time:180
Increment:2
END Time+
BEGIN Time-
Time_Unit:1msec
Total_Time:600000
Increment:2000
END Time-
BEGIN Position
PI
+
END Position
";
        let s = parse(text).unwrap();
        assert_eq!(
            s.time[0],
            SideTime {
                unit_ms: 1000,
                total: 180,
                byoyomi: 5,
                increment: 2,
                least: 0,
                remaining: None
            }
        );
        assert_eq!(
            s.time[1],
            SideTime {
                unit_ms: 1,
                total: 600_000,
                byoyomi: 5,
                increment: 2000,
                least: 0,
                remaining: None
            }
        );
    }

    #[test]
    fn the_moves_after_the_position_are_played_and_timed() {
        let text = "\
Game_ID:g
Your_Turn:+
BEGIN Position
PI
+
+7776FU,T3
-3334FU,T1
'a comment
+8822UM
END Position
";
        let s = parse(text).unwrap();
        assert_eq!(s.moves.len(), 3);
        assert_eq!(s.moves[0].1, Some(3));
        assert_eq!(s.moves[1].1, Some(1));
        assert_eq!(s.moves[2].1, None);
        assert_eq!(s.start, Position::startpos());
    }

    #[test]
    fn an_illegal_move_after_the_position_is_refused() {
        let text = "Game_ID:g\nYour_Turn:+\nBEGIN Position\nPI\n+\n+7775FU,T3\nEND Position\n";
        assert!(matches!(parse(text), Err(SummaryError::Move(0, _))));
    }

    #[test]
    fn unknown_keys_and_sections_are_skipped() {
        let text = "\
Game_ID:g
Some_Future_Key:whatever
Your_Turn:+
BEGIN Jishogi
Declaration:24
BEGIN Nested
Anything:1
END Nested
END Jishogi
BEGIN Time
Time_Unit:1min
Total_Time:10
Time_Roundup:NO
END Time
BEGIN Position
PI
+
END Position
";
        let s = parse(text).unwrap();
        assert_eq!(s.time[0].unit_ms, 60_000);
        assert_eq!(s.time[1].total, 10);
    }

    #[test]
    fn a_resumed_game_carries_the_remaining_times() {
        let text = "\
Game_ID:g
Your_Turn:+
BEGIN Time
Total_Time:600
Remaining_Time+:432
Remaining_Time-:517
END Time
BEGIN Position
PI
+
END Position
";
        let s = parse(text).unwrap();
        assert_eq!(s.time[0].remaining, Some(432));
        assert_eq!(s.time[1].remaining, Some(517));
    }

    #[test]
    fn what_a_summary_cannot_do_without_is_named() {
        assert_eq!(
            parse("Your_Turn:+\nBEGIN Position\nPI\n+\nEND Position\n").unwrap_err(),
            SummaryError::Missing(b"Game_ID")
        );
        assert_eq!(
            parse("Game_ID:g\nBEGIN Position\nPI\n+\nEND Position\n").unwrap_err(),
            SummaryError::Missing(b"Your_Turn")
        );
        assert_eq!(
            parse("Game_ID:g\nYour_Turn:+\n").unwrap_err(),
            SummaryError::NoPosition
        );
        assert_eq!(
            parse("Game_ID:g\nYour_Turn:x\n").unwrap_err(),
            SummaryError::BadValue(b"Your_Turn")
        );
        assert_eq!(
            parse("Game_ID:g\nYour_Turn:+\nBEGIN Time\nTotal_Time:abc\nEND Time\n").unwrap_err(),
            SummaryError::BadValue(b"Total_Time")
        );
        assert_eq!(
            parse("Game_ID:g\nYour_Turn:+\nBEGIN Time\nTotal_Time:1\n").unwrap_err(),
            SummaryError::Unbalanced
        );
        assert_eq!(
            parse("Game_ID:g\nYour_Turn:+\nTo_Move:-\nBEGIN Position\nPI\n+\nEND Position\n")
                .unwrap_err(),
            SummaryError::ToMoveMismatch
        );
    }

    #[test]
    fn time_units_read_as_milliseconds() {
        assert_eq!(parse_time_unit(b"1sec"), Some(1000));
        assert_eq!(parse_time_unit(b"sec"), Some(1000));
        assert_eq!(parse_time_unit(b"1min"), Some(60_000));
        assert_eq!(parse_time_unit(b"100msec"), Some(100));
        assert_eq!(parse_time_unit(b"0sec"), None);
        assert_eq!(parse_time_unit(b"1hour"), None);
    }

    #[test]
    fn move_lines_split_into_move_and_time() {
        assert_eq!(
            split_move_line(b"+7776FU,T12"),
            Some((&b"+7776FU"[..], Some(12)))
        );
        assert_eq!(split_move_line(b"+7776FU"), Some((&b"+7776FU"[..], None)));
        assert_eq!(split_move_line(b"+7776FU,X"), None);
    }
}
