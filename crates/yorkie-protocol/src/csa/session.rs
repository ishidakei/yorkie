//! The client's side of the protocol as a state machine over received lines.
//!
//! [`Session::on_line`] reads one line from the server and says what it means
//! ([`Step`]); the caller does what that asks of the engine and of the
//! connection. Nothing here touches either, so a whole exchange can be played
//! through it from a transcript.

use yorkie_state::{Color, CsaError, Move, Position, SfenBuf, format_sfen, parse_csa_move};

use super::clock::Clock;
use super::summary::{
    GameSummary, MAX_SUMMARY_BYTES, SummaryError, parse_game_summary, split_move_line,
};
use crate::engine::GoParams;
use yorkie_search::EnteringKingRule;
use yorkie_state::STARTPOS_SFEN;
use yorkie_state::write_usi_move;

/// Where the exchange with the server stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Phase {
    /// `LOGIN` sent, its answer not yet read.
    LoggingIn,
    /// Logged in, no game offered.
    Waiting,
    /// Inside a `BEGIN Game_Summary` block.
    Summary,
    /// The offered game agreed to, `START` not yet read.
    Agreed,
    /// A game under way.
    Playing,
    /// `LOGOUT` sent.
    LoggingOut,
}

/// Why a game ended, from the line the server sends before the result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EndReason {
    Resign,
    TimeUp,
    IllegalMove,
    Sennichite,
    OuteSennichite,
    Jishogi,
    MaxMoves,
    /// A `#` line of no form this client knows.
    Other,
}

impl EndReason {
    fn of(line: &[u8]) -> Self {
        match line {
            b"#RESIGN" => Self::Resign,
            b"#TIME_UP" => Self::TimeUp,
            b"#ILLEGAL_MOVE" => Self::IllegalMove,
            b"#SENNICHITE" => Self::Sennichite,
            b"#OUTE_SENNICHITE" => Self::OuteSennichite,
            b"#JISHOGI" => Self::Jishogi,
            b"#MAX_MOVES" => Self::MaxMoves,
            _ => Self::Other,
        }
    }

    pub(crate) fn name(self) -> &'static [u8] {
        match self {
            Self::Resign => b"resign",
            Self::TimeUp => b"time up",
            Self::IllegalMove => b"illegal move",
            Self::Sennichite => b"sennichite",
            Self::OuteSennichite => b"oute sennichite",
            Self::Jishogi => b"jishogi",
            Self::MaxMoves => b"max moves",
            Self::Other => b"other",
        }
    }
}

/// How a game ended for this client.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GameResult {
    Win,
    Lose,
    Draw,
    /// `#CENSORED`: the game ended without a result, as at the move limit.
    Censored,
    /// `#CHUDAN`: the server broke the game off.
    Chudan,
}

impl GameResult {
    fn of(line: &[u8]) -> Option<Self> {
        Some(match line {
            b"#WIN" => Self::Win,
            b"#LOSE" => Self::Lose,
            b"#DRAW" => Self::Draw,
            b"#CENSORED" => Self::Censored,
            b"#CHUDAN" => Self::Chudan,
            _ => return None,
        })
    }

    pub(crate) fn name(self) -> &'static [u8] {
        match self {
            Self::Win => b"win",
            Self::Lose => b"lose",
            Self::Draw => b"draw",
            Self::Censored => b"censored",
            Self::Chudan => b"chudan",
        }
    }
}

/// The rules this build plays a game under, which a server's summary is held
/// against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BuildRules {
    /// The ply past which the search scores every position a draw
    /// (`max_moves_to_draw`); `None` for no limit.
    pub(crate) move_limit: Option<u64>,
    /// The rule the engine checks before it declares (`entering_king_rule`).
    pub(crate) rule: EnteringKingRule,
}

impl BuildRules {
    /// The rules this binary was built with.
    pub(crate) const COMPILED: Self = Self {
        move_limit: match crate::config::MAX_MOVES_TO_DRAW {
            0 => None,
            n => Some(n as u64),
        },
        rule: EnteringKingRule::from_option(crate::config::ENTERING_KING_RULE.as_bytes()),
    };
}

/// The server's move limit against the build's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LimitCheck {
    Same,
    /// The server ends the game before the build's limit: the engine plans
    /// for moves the game will not have, and loses nothing by it.
    ServerShorter,
    /// The server plays on past the build's limit, into positions the engine
    /// scores as drawn whatever they are: the game is refused.
    ServerLonger,
}

impl LimitCheck {
    fn of(server: Option<u64>, build: Option<u64>) -> Self {
        match (server, build) {
            (s, b) if s == b => Self::Same,
            (_, None) => Self::ServerShorter,
            (Some(s), Some(b)) if s < b => Self::ServerShorter,
            _ => Self::ServerLonger,
        }
    }
}

/// The declaration rule the summary names, against the build's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RuleCheck {
    /// The summary names the build's rule.
    Same,
    /// The summary names another rule, which this one is: a declaration the
    /// engine makes under its own rule may be refused, and lose the game.
    Differs(EnteringKingRule),
    /// The summary names a rule this client does not know.
    Unknown,
    /// The summary names no rule.
    NotNamed,
}

impl RuleCheck {
    fn of(declaration: &[u8], build: EnteringKingRule) -> Self {
        let server = match declaration {
            b"" => return Self::NotNamed,
            // The CSA's own declaration rule, version 1.1: the 27-point rule.
            b"Jishogi 1.1" => EnteringKingRule::Point27,
            _ => return Self::Unknown,
        };
        if server == build {
            Self::Same
        } else {
            Self::Differs(server)
        }
    }
}

/// What one received line means for the caller.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Step<'a> {
    /// Nothing to do: a blank keep-alive line, a line of a summary still being
    /// read, an extended-mode `##` line, or one the client has no use for.
    Nothing,
    LoginOk,
    LoginRefused,
    LoggedOut,
    /// A game is offered and its summary was read: prepare the engine and
    /// agree to it. What the summary says about the move limit and the
    /// declaration rule, against the build's, is for the log.
    Offered {
        limit: LimitCheck,
        rule: RuleCheck,
    },
    /// A game is offered whose move limit is longer than the build's, or that
    /// has none where the build has one: refuse it.
    LimitTooLong,
    /// A summary that cannot be read: refuse the game.
    Unreadable(SummaryError),
    /// The offered game was refused, by either side.
    Rejected,
    /// The game started.
    Started,
    /// A move was played, by this client when `ours`.
    Moved {
        ours: bool,
    },
    /// The server relayed a move the client's position does not allow. The
    /// server is the authority on the game, so the client stops playing it and
    /// waits for the server to end it.
    Unplayable(&'a [u8], CsaError),
    /// The game ended.
    GameOver {
        reason: Option<EndReason>,
        result: GameResult,
        /// The game ended on a declaration this client made.
        declared: bool,
    },
    /// A line this client did not expect where it came.
    Unexpected(&'a [u8]),
}

/// The game being played: everything the client keeps to answer a move.
pub(crate) struct Game {
    pub(crate) summary: GameSummary,
    pub(crate) clock: Clock,
    /// The position after every move played so far.
    pub(crate) pos: Position,
    /// The game's starting position as an SFEN, or empty for the initial
    /// position.
    start_sfen: Vec<u8>,
    /// Every move played from the start, as USI moves separated by spaces: the
    /// move list a USI `position` command would carry.
    usi_moves: Vec<u8>,
    /// The server relayed a move this client could not play, so it no longer
    /// knows the position.
    lost_track: bool,
}

/// Room for a whole game's move list: the engine refuses one longer than
/// this many moves anyway.
const USI_MOVES_CAPACITY: usize = crate::engine::MAX_POSITION_MOVES * 6;

impl Game {
    fn new(summary: GameSummary) -> Self {
        let mut buf = SfenBuf::new();
        let sfen = format_sfen(&summary.start, &mut buf);
        let start_sfen = if sfen == STARTPOS_SFEN {
            Vec::new()
        } else {
            sfen.to_vec()
        };
        let mut game = Self {
            clock: Clock::new(&summary.time),
            pos: summary.start.clone(),
            start_sfen,
            usi_moves: Vec::with_capacity(USI_MOVES_CAPACITY),
            lost_track: false,
            summary,
        };
        for i in 0..game.summary.moves.len() {
            let m = game.summary.moves[i].0;
            game.play(m);
        }
        game
    }

    fn play(&mut self, m: Move) {
        self.pos.do_move(m);
        if !self.usi_moves.is_empty() {
            self.usi_moves.push(b' ');
        }
        let mut bytes = [0u8; yorkie_state::MAX_USI_MOVE_LEN];
        let mut out = yorkie_state::TextWriter::new(&mut bytes);
        write_usi_move(m, &mut out);
        self.usi_moves.extend_from_slice(out.as_bytes());
    }

    /// The side this client plays.
    pub(crate) fn us(&self) -> Color {
        self.summary.your_turn
    }

    /// Whether the engine owes the server a move now.
    pub(crate) fn our_move(&self) -> bool {
        !self.lost_track && self.pos.side_to_move() == self.us()
    }

    /// The starting position: `None` for the initial one, else the four SFEN
    /// fields.
    pub(crate) fn start_fields(&self) -> Option<[&[u8]; 4]> {
        if self.start_sfen.is_empty() {
            return None;
        }
        let mut fields = [&b""[..]; 4];
        for (slot, field) in fields.iter_mut().zip(self.start_sfen.split(|&b| b == b' ')) {
            *slot = field;
        }
        Some(fields)
    }

    /// The moves played so far, as a USI `position` command lists them.
    pub(crate) fn usi_moves(&self) -> &[u8] {
        &self.usi_moves
    }

    /// The limits for the engine's next move.
    pub(crate) fn go_params(&self) -> GoParams {
        self.clock.go_params(self.pos.side_to_move())
    }
}

/// The protocol state: the phase, the summary being read and the game being
/// played.
pub(crate) struct Session {
    build: BuildRules,
    phase: Phase,
    summary: Vec<u8>,
    summary_too_long: bool,
    game: Option<Game>,
    reason: Option<EndReason>,
    /// The `%KACHI` the server relayed was this client's.
    declared: bool,
}

impl Session {
    /// A session whose `LOGIN` has been sent, for a build of `build`.
    pub(crate) fn new(build: BuildRules) -> Self {
        Self {
            build,
            phase: Phase::LoggingIn,
            summary: Vec::with_capacity(4096),
            summary_too_long: false,
            game: None,
            reason: None,
            declared: false,
        }
    }

    pub(crate) fn phase(&self) -> Phase {
        self.phase
    }

    /// The game offered or under way.
    pub(crate) fn game(&self) -> Option<&Game> {
        self.game.as_ref()
    }

    /// `LOGOUT` has been sent.
    pub(crate) fn logging_out(&mut self) {
        self.phase = Phase::LoggingOut;
    }

    /// The offered game has been agreed to.
    pub(crate) fn agreed(&mut self) {
        self.phase = Phase::Agreed;
    }

    /// The offered game has been refused by this client.
    pub(crate) fn refused(&mut self) {
        self.game = None;
        self.phase = Phase::Waiting;
    }

    /// Read one line, without its line ending.
    pub(crate) fn on_line<'a>(&mut self, line: &'a [u8]) -> Step<'a> {
        if self.phase == Phase::Summary {
            return self.summary_line(line);
        }
        if line.is_empty() || line.starts_with(b"##") {
            return Step::Nothing;
        }
        if line == b"LOGOUT:completed" {
            self.phase = Phase::LoggingOut;
            return Step::LoggedOut;
        }
        match self.phase {
            Phase::LoggingIn => {
                if line.starts_with(b"LOGIN:") {
                    if line.ends_with(b" OK") {
                        self.phase = Phase::Waiting;
                        return Step::LoginOk;
                    }
                    return Step::LoginRefused;
                }
                Step::Unexpected(line)
            }
            Phase::Waiting | Phase::Agreed => {
                if line == b"BEGIN Game_Summary" {
                    self.phase = Phase::Summary;
                    self.summary.clear();
                    self.summary_too_long = false;
                    self.game = None;
                    return Step::Nothing;
                }
                if self.phase == Phase::Agreed {
                    if line.starts_with(b"START:") {
                        self.phase = Phase::Playing;
                        self.reason = None;
                        self.declared = false;
                        return Step::Started;
                    }
                    if line.starts_with(b"REJECT:") {
                        self.refused();
                        return Step::Rejected;
                    }
                }
                Step::Unexpected(line)
            }
            Phase::Playing => self.game_line(line),
            Phase::Summary | Phase::LoggingOut => Step::Nothing,
        }
    }

    fn summary_line<'a>(&mut self, line: &'a [u8]) -> Step<'a> {
        if line != b"END Game_Summary" {
            if self.summary.len() + line.len() + 1 > MAX_SUMMARY_BYTES {
                self.summary_too_long = true;
            } else {
                self.summary.extend_from_slice(line);
                self.summary.push(b'\n');
            }
            return Step::Nothing;
        }
        self.phase = Phase::Waiting;
        if self.summary_too_long {
            return Step::Unreadable(SummaryError::TooLong);
        }
        match parse_game_summary(&self.summary) {
            Ok(summary) => {
                let limit = LimitCheck::of(summary.max_moves, self.build.move_limit);
                let rule = RuleCheck::of(&summary.declaration, self.build.rule);
                self.game = Some(Game::new(summary));
                if limit == LimitCheck::ServerLonger {
                    return Step::LimitTooLong;
                }
                Step::Offered { limit, rule }
            }
            Err(e) => Step::Unreadable(e),
        }
    }

    fn game_line<'a>(&mut self, line: &'a [u8]) -> Step<'a> {
        let game = self.game.as_mut().expect("a game is under way");
        match line[0] {
            b'+' | b'-' => {
                if game.lost_track {
                    return Step::Nothing;
                }
                let Some((text, time)) = split_move_line(line) else {
                    game.lost_track = true;
                    return Step::Unplayable(line, CsaError::MalformedLine);
                };
                let mover = game.pos.side_to_move();
                match parse_csa_move(text, &game.pos) {
                    Ok(m) => {
                        game.play(m);
                        game.clock.charge(mover, time.unwrap_or(0));
                        Step::Moved {
                            ours: mover == game.us(),
                        }
                    }
                    Err(e) => {
                        game.lost_track = true;
                        Step::Unplayable(line, e)
                    }
                }
            }
            // `%TORYO` and `%KACHI` repeat the move that ends the game; the `#`
            // lines after them say how it ended. The declaration is the side
            // to move's, since a declaration is made in place of a move.
            b'%' => {
                if line == b"%KACHI" {
                    self.declared = !game.lost_track && game.pos.side_to_move() == game.us();
                }
                Step::Nothing
            }
            b'#' => match GameResult::of(line) {
                Some(result) => {
                    self.phase = Phase::Waiting;
                    Step::GameOver {
                        reason: self.reason.take(),
                        result,
                        declared: std::mem::take(&mut self.declared),
                    }
                }
                None => {
                    self.reason = Some(EndReason::of(line));
                    Step::Nothing
                }
            },
            _ => Step::Unexpected(line),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SUMMARY: &str = "\
BEGIN Game_Summary
Protocol_Version:1.2
Protocol_Mode:Server
Format:Shogi 1.0
Game_ID:test-game
Name+:black
Name-:white
Your_Turn:+
To_Move:+
BEGIN Time+
Time_Unit:1sec
Total_Time:180
Increment:2
END Time+
BEGIN Time-
Time_Unit:1sec
Total_Time:600
Increment:2
END Time-
BEGIN Position
PI
+
END Position
END Game_Summary";

    /// A build with no move limit, declaring under the 27-point rule.
    const UNLIMITED: BuildRules = BuildRules {
        move_limit: None,
        rule: EnteringKingRule::Point27,
    };

    /// What [`SUMMARY`] offers a build of [`UNLIMITED`].
    const OFFERED: &str = "Offered { limit: Same, rule: NotNamed }";

    /// Feed `transcript` line by line and collect what each line asked for,
    /// leaving out the lines that ask for nothing.
    fn run(session: &mut Session, transcript: &str) -> Vec<String> {
        let mut steps = Vec::new();
        for line in transcript.lines() {
            let step = session.on_line(line.as_bytes());
            if step != Step::Nothing {
                steps.push(format!("{step:?}"));
            }
        }
        steps
    }

    fn started() -> Session {
        let mut s = Session::new(UNLIMITED);
        run(&mut s, "LOGIN:black OK");
        run(&mut s, SUMMARY);
        s.agreed();
        assert_eq!(run(&mut s, "START:test-game"), ["Started"]);
        s
    }

    #[test]
    fn login_answers() {
        let mut s = Session::new(UNLIMITED);
        assert_eq!(run(&mut s, "LOGIN:incorrect"), ["LoginRefused"]);
        assert_eq!(s.phase(), Phase::LoggingIn);
        assert_eq!(run(&mut s, "LOGIN:black OK"), ["LoginOk"]);
        assert_eq!(s.phase(), Phase::Waiting);
        s.logging_out();
        assert_eq!(run(&mut s, "LOGOUT:completed"), ["LoggedOut"]);
    }

    #[test]
    fn a_summary_offers_a_game() {
        let mut s = Session::new(UNLIMITED);
        run(&mut s, "LOGIN:black OK");
        assert_eq!(run(&mut s, SUMMARY), [OFFERED]);
        let g = s.game().unwrap();
        assert_eq!(g.summary.game_id, b"test-game");
        assert!(g.our_move());
        assert_eq!(g.start_fields(), None);
        let p = g.go_params();
        assert_eq!((p.btime, p.wtime), (Some(180_000), Some(600_000)));
    }

    #[test]
    fn a_rejected_offer_goes_back_to_waiting() {
        let mut s = Session::new(UNLIMITED);
        run(&mut s, "LOGIN:black OK");
        run(&mut s, SUMMARY);
        s.agreed();
        assert_eq!(run(&mut s, "REJECT:test-game by white"), ["Rejected"]);
        assert_eq!(s.phase(), Phase::Waiting);
        assert!(s.game().is_none());
        // And a fresh offer after it is read as one.
        assert_eq!(run(&mut s, SUMMARY), [OFFERED]);
    }

    #[test]
    fn an_offer_that_never_starts_is_replaced_by_the_next() {
        let mut s = Session::new(UNLIMITED);
        run(&mut s, "LOGIN:black OK");
        run(&mut s, SUMMARY);
        s.agreed();
        assert_eq!(run(&mut s, SUMMARY), [OFFERED]);
        assert_eq!(s.phase(), Phase::Waiting);
    }

    #[test]
    fn an_unreadable_summary_is_said_so() {
        let mut s = Session::new(UNLIMITED);
        run(&mut s, "LOGIN:black OK");
        let steps = run(
            &mut s,
            "BEGIN Game_Summary\nGame_ID:x\nBEGIN Position\nPI\n+\nEND Position\nEND Game_Summary",
        );
        assert_eq!(
            steps,
            ["Unreadable(Missing([89, 111, 117, 114, 95, 84, 117, 114, 110]))"]
        );
        assert_eq!(s.phase(), Phase::Waiting);
    }

    #[test]
    fn moves_are_played_and_both_clocks_charged() {
        let mut s = started();
        assert_eq!(
            run(&mut s, "+7776FU,T3\n\n-3334FU,T7"),
            ["Moved { ours: true }", "Moved { ours: false }"]
        );
        let g = s.game().unwrap();
        assert_eq!(g.clock.remaining_units(Color::Black), 179);
        assert_eq!(g.clock.remaining_units(Color::White), 595);
        assert_eq!(g.usi_moves(), b"7g7f 3c3d");
        assert!(g.our_move());
        let p = g.go_params();
        assert_eq!((p.btime, p.wtime), (Some(179_000), Some(595_000)));
    }

    #[test]
    fn a_move_the_position_does_not_allow_stops_play() {
        let mut s = started();
        let steps = run(&mut s, "+7775FU,T3");
        assert_eq!(steps.len(), 1);
        assert!(steps[0].starts_with("Unplayable"));
        assert!(!s.game().unwrap().our_move());
        // The server's later lines are still read to the end of the game.
        assert_eq!(run(&mut s, "-3334FU,T1"), Vec::<String>::new());
        assert_eq!(
            run(&mut s, "#ILLEGAL_MOVE\n#WIN"),
            ["GameOver { reason: Some(IllegalMove), result: Win, declared: false }"]
        );
    }

    /// Every way `shogi-server` ends a game, as the lines it sends this client.
    #[test]
    fn every_game_end_is_read() {
        let cases: &[(&str, &str)] = &[
            ("%TORYO\n#RESIGN\n#WIN", "Some(Resign), result: Win"),
            ("%TORYO\n#RESIGN\n#LOSE", "Some(Resign), result: Lose"),
            ("#TIME_UP\n#WIN", "Some(TimeUp), result: Win"),
            ("#TIME_UP\n#LOSE", "Some(TimeUp), result: Lose"),
            ("#ILLEGAL_MOVE\n#LOSE", "Some(IllegalMove), result: Lose"),
            (
                "%KACHI\n#ILLEGAL_MOVE\n#LOSE",
                "Some(IllegalMove), result: Lose",
            ),
            ("#SENNICHITE\n#DRAW", "Some(Sennichite), result: Draw"),
            (
                "#OUTE_SENNICHITE\n#WIN",
                "Some(OuteSennichite), result: Win",
            ),
            (
                "#OUTE_SENNICHITE\n#LOSE",
                "Some(OuteSennichite), result: Lose",
            ),
            ("%KACHI\n#JISHOGI\n#WIN", "Some(Jishogi), result: Win"),
            ("%KACHI\n#JISHOGI\n#LOSE", "Some(Jishogi), result: Lose"),
            ("#MAX_MOVES\n#CENSORED", "Some(MaxMoves), result: Censored"),
            ("#CHUDAN", "None, result: Chudan"),
            ("#SOMETHING_NEW\n#DRAW", "Some(Other), result: Draw"),
        ];
        for (lines, want) in cases {
            let mut s = started();
            run(&mut s, "+7776FU,T1");
            let steps = run(&mut s, lines);
            assert_eq!(
                steps,
                [format!("GameOver {{ reason: {want}, declared: false }}")],
                "{lines}"
            );
            assert_eq!(s.phase(), Phase::Waiting, "{lines}");
            // The next game is offered and read as usual.
            assert_eq!(run(&mut s, SUMMARY), [OFFERED], "{lines}");
        }
    }

    #[test]
    fn a_game_resumed_past_its_start_carries_its_moves() {
        let mut s = Session::new(UNLIMITED);
        run(&mut s, "LOGIN:black OK");
        let summary = SUMMARY.replace("PI\n+\n", "PI\n+\n+7776FU,T1\n-3334FU,T1\n");
        assert_eq!(run(&mut s, &summary), [OFFERED]);
        let g = s.game().unwrap();
        assert_eq!(g.usi_moves(), b"7g7f 3c3d");
        assert!(g.our_move());
        // The server starts the clocks afresh at `START`.
        assert_eq!(g.clock.remaining_units(Color::Black), 180);
    }

    #[test]
    fn a_position_other_than_the_initial_one_is_named_by_its_sfen() {
        let mut s = Session::new(UNLIMITED);
        run(&mut s, "LOGIN:black OK");
        // The initial position without Black's bishop.
        let summary = SUMMARY.replace("PI\n", "PI88KA\n");
        assert_eq!(run(&mut s, &summary), [OFFERED]);
        let fields = s.game().unwrap().start_fields().unwrap();
        assert_eq!(
            fields[0],
            b"lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/7R1/LNSGKGSNL"
        );
        assert_eq!(&fields[1..], [&b"b"[..], b"-", b"1"]);
    }

    /// [`SUMMARY`] with `Max_Moves` and `Declaration` lines, each left out
    /// where `None`.
    fn summary_with(max_moves: Option<&str>, declaration: Option<&str>) -> String {
        let mut extra = String::new();
        if let Some(n) = max_moves {
            extra.push_str(&format!("Max_Moves:{n}\n"));
        }
        if let Some(d) = declaration {
            extra.push_str(&format!("Declaration:{d}\n"));
        }
        SUMMARY.replace("To_Move:+\n", &format!("To_Move:+\n{extra}"))
    }

    fn offer(build: BuildRules, summary: &str) -> (Session, Vec<String>) {
        let mut s = Session::new(build);
        run(&mut s, "LOGIN:black OK");
        let steps = run(&mut s, summary);
        (s, steps)
    }

    #[test]
    fn the_servers_move_limit_is_held_against_the_builds() {
        let limited = BuildRules {
            move_limit: Some(256),
            ..UNLIMITED
        };
        let cases: &[(BuildRules, Option<&str>, &str)] = &[
            (
                limited,
                Some("256"),
                "Offered { limit: Same, rule: NotNamed }",
            ),
            (
                limited,
                Some("200"),
                "Offered { limit: ServerShorter, rule: NotNamed }",
            ),
            (limited, Some("300"), "LimitTooLong"),
            (limited, None, "LimitTooLong"),
            (
                UNLIMITED,
                Some("256"),
                "Offered { limit: ServerShorter, rule: NotNamed }",
            ),
            (UNLIMITED, None, "Offered { limit: Same, rule: NotNamed }"),
        ];
        for &(build, max_moves, want) in cases {
            let (s, steps) = offer(build, &summary_with(max_moves, None));
            assert_eq!(steps, [want], "{build:?} {max_moves:?}");
            assert_eq!(s.phase(), Phase::Waiting);
        }
    }

    #[test]
    fn a_game_refused_for_its_move_limit_leaves_the_client_waiting_for_the_next() {
        let limited = BuildRules {
            move_limit: Some(256),
            ..UNLIMITED
        };
        let (mut s, steps) = offer(limited, &summary_with(Some("300"), None));
        assert_eq!(steps, ["LimitTooLong"]);
        assert_eq!(s.game().unwrap().summary.max_moves, Some(300));
        s.refused();
        assert!(s.game().is_none());
        assert_eq!(
            run(&mut s, &summary_with(Some("256"), None)),
            ["Offered { limit: Same, rule: NotNamed }"]
        );
    }

    #[test]
    fn the_declaration_rule_the_summary_names_is_held_against_the_builds() {
        let point24 = BuildRules {
            rule: EnteringKingRule::Point24,
            ..UNLIMITED
        };
        let cases: &[(BuildRules, Option<&str>, &str)] = &[
            (UNLIMITED, Some("Jishogi 1.1"), "Same"),
            (point24, Some("Jishogi 1.1"), "Differs(Point27)"),
            (UNLIMITED, Some("Something 9.9"), "Unknown"),
            (UNLIMITED, None, "NotNamed"),
        ];
        for &(build, declaration, want) in cases {
            let (_, steps) = offer(build, &summary_with(None, declaration));
            assert_eq!(
                steps,
                [format!("Offered {{ limit: Same, rule: {want} }}")],
                "{declaration:?}"
            );
        }
    }

    #[test]
    fn a_declaration_of_ours_the_server_accepts() {
        let mut s = started();
        assert!(s.game().unwrap().our_move());
        assert_eq!(
            run(&mut s, "%KACHI\n#JISHOGI\n#WIN"),
            ["GameOver { reason: Some(Jishogi), result: Win, declared: true }"]
        );
    }

    #[test]
    fn a_declaration_of_ours_the_server_refuses() {
        let mut s = started();
        assert_eq!(
            run(&mut s, "%KACHI\n#ILLEGAL_MOVE\n#LOSE"),
            ["GameOver { reason: Some(IllegalMove), result: Lose, declared: true }"]
        );
    }

    #[test]
    fn a_declaration_of_the_opponents_is_not_ours() {
        let mut s = started();
        run(&mut s, "+7776FU,T1");
        assert_eq!(
            run(&mut s, "%KACHI\n#JISHOGI\n#LOSE"),
            ["GameOver { reason: Some(Jishogi), result: Lose, declared: false }"]
        );
    }

    #[test]
    fn a_declaration_belongs_to_its_own_game_only() {
        let mut s = started();
        run(&mut s, "%KACHI\n#JISHOGI\n#WIN");
        run(&mut s, SUMMARY);
        s.agreed();
        run(&mut s, "START:test-game");
        assert_eq!(
            run(&mut s, "#TIME_UP\n#LOSE"),
            ["GameOver { reason: Some(TimeUp), result: Lose, declared: false }"]
        );
    }
}
