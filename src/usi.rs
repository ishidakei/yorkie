use crate::book::*;
#[cfg(feature = "material")]
use crate::evaluate::material::*;
#[cfg(feature = "nnue")]
use crate::evaluate::nnue::evaluate_at_root;
use crate::file_to_vec::*;
use crate::huffman_code::*;
use crate::learn::*;
use crate::movegen::*;
use crate::movetypes::*;
use crate::position::*;
use crate::search::*;
use crate::sfen::START_SFEN;
use crate::thread::*;
use crate::tt::*;
use crate::types::*;
use crate::usioption::*;
use anyhow::{Context, Result, anyhow};
use std::io::prelude::*;

/// `Byoyomi_Margin` (ms) subtracted off a `byoyomi` period; tournament build bakes a compile-time const.
#[cfg(feature = "tournament")]
#[inline]
fn byoyomi_margin(_usi_options: &UsiOptions) -> i64 {
    crate::tournament::BYOYOMI_MARGIN
}

#[cfg(not(feature = "tournament"))]
#[inline]
fn byoyomi_margin(usi_options: &UsiOptions) -> i64 {
    usi_options.get_i64(UsiOptions::BYOYOMI_MARGIN)
}

/// Parse the argument list of a USI `go` command into search limits and whether `ponder` was set.
/// Kept separate from [`go`] so the margin behaviour is unit-testable.
fn parse_go_limits(usi_options: &UsiOptions, args: &[&str]) -> Result<(LimitsType, bool)> {
    let mut limits = LimitsType::new();
    limits.start_time = Some(std::time::Instant::now());
    let mut iter = args.iter();
    fn next_num<T: std::str::FromStr>(limit_type: &str, iter: &mut std::slice::Iter<'_, &str>) -> Result<T>
    where
        <T as std::str::FromStr>::Err: std::error::Error + Send + Sync + 'static,
    {
        let item = iter.next().with_context(|| format!("no token after {}.", limit_type))?;
        item.parse::<T>().map_err(|e| anyhow!("{}: {}", e, item))
    }
    let mut ponder_mode = false;
    while let Some(&limit_type) = iter.next() {
        match limit_type {
            "btime" | "wtime" => {
                let color = if limit_type == "btime" { Color::BLACK } else { Color::WHITE };
                let n: u64 = next_num(limit_type, &mut iter)?;
                limits.time[color.0 as usize] = std::time::Duration::from_millis(n);
            }
            "binc" | "winc" => {
                let color = if limit_type == "binc" { Color::BLACK } else { Color::WHITE };
                let n = next_num(limit_type, &mut iter)?;
                limits.inc[color.0 as usize] = std::time::Duration::from_millis(n);
            }
            "byoyomi" | "movetime" => {
                let n: u64 = next_num(limit_type, &mut iter)?;
                let margin = if limit_type == "byoyomi" {
                    byoyomi_margin(usi_options) as u64
                } else {
                    0
                };
                limits.movetime = Some(std::time::Duration::from_millis(n.saturating_sub(margin)));
            }
            "depth" => {
                let n = next_num(limit_type, &mut iter)?;
                limits.depth = Some(n);
            }
            "infinite" => limits.infinite = Some(()),
            "nodes" => {
                let n = next_num(limit_type, &mut iter)?;
                limits.nodes = Some(n);
            }
            "ponder" => {
                ponder_mode = true;
            }
            "perft" => {
                let n = next_num(limit_type, &mut iter)?;
                limits.perft = Some(n);
            }
            "mate" => {
                let mate_limit = iter.next().with_context(|| format!("no token after {}.", limit_type))?;
                let n = match *mate_limit {
                    "infinite" => 0,
                    _ => mate_limit.parse().map_err(|e| anyhow!("{}: {}", e, mate_limit))?,
                };
                limits.mate = Some(n);
            }
            invalid_token => return Err(anyhow!("invalid token: {}", invalid_token)),
        }
    }
    Ok((limits, ponder_mode))
}

fn go(
    thread_pool: &mut ThreadPool,
    tt: &mut TranspositionTable,
    usi_options: &UsiOptions,
    pos: &Position,
    args: &[&str],
) -> Result<()> {
    let (limits, ponder_mode) = parse_go_limits(usi_options, args)?;
    let hide_all_output = false;
    thread_pool.start_thinking(pos, tt, limits, usi_options, ponder_mode, hide_all_output);
    Ok(())
}

fn isready(
    is_ready: &mut bool,
    usi_options: &mut UsiOptions,
    thread_pool: &mut ThreadPool,
    tt: &mut TranspositionTable,
    reductions: &mut Reductions,
) {
    fn isready_impl(
        is_ready: &mut bool,
        usi_options: &mut UsiOptions,
        thread_pool: &mut ThreadPool,
        tt: &mut TranspositionTable,
        reductions: &mut Reductions,
    ) -> Result<()> {
        if *is_ready {
            return Ok(());
        }
        // Apply eval_options.txt before any evaluator initialisation so its
        // overrides reach the upcoming nn.bin load.
        let eval_dir = usi_options.get_string(UsiOptions::EVAL_DIR);
        if !eval_dir.is_empty() {
            let path = format!("{}/eval_options.txt", eval_dir);
            let lines = usi_options.read_eval_options(std::path::Path::new(&path), thread_pool, tt, reductions, is_ready);
            for line in lines {
                println!("{}", line);
            }
        }
        #[cfg(feature = "nnue")]
        {
            // Re-read in case eval_options.txt rewrote Eval_Dir.
            let eval_dir = usi_options.get_string(UsiOptions::EVAL_DIR);
            for line in nnue_isready_lines(&eval_dir) {
                println!("{}", line);
            }
        }
        if usi_options.get_bool(UsiOptions::BOOK_ENABLE) {
            let file_name = usi_options.get_filename(UsiOptions::BOOK_FILE);
            let book_options = BookOptions::from_usi_options(usi_options);
            let book =
                Book::from_file(&file_name, &book_options).map_err(|e| anyhow!("{}: {}", e, file_name.to_string_lossy()))?;
            thread_pool.book = Some(book);
        }
        tt.resize(usi_options.get_i64(UsiOptions::USI_HASH) as usize, thread_pool);
        *is_ready = true;
        Ok(())
    }
    match isready_impl(is_ready, usi_options, thread_pool, tt, reductions) {
        Ok(()) => println!("readyok"),
        Err(e) => println!("info {}", e),
    }
}

#[cfg(feature = "nnue")]
fn nnue_isready_lines(eval_dir: &str) -> Vec<String> {
    if eval_dir.is_empty() {
        return vec!["info string nnue: Eval_Dir is not set".to_string()];
    }
    let requested = std::path::PathBuf::from(format!("{}/nn.bin", eval_dir));
    if crate::evaluate::nnue::loaded_path().as_deref() == Some(requested.as_path()) {
        return Vec::new();
    }
    match crate::evaluate::nnue::load_network_from_path(&requested) {
        Ok(()) => {
            let sha = crate::evaluate::nnue::loaded_sha256_hex().unwrap_or_default();
            vec![format!("info string nnue: loaded {} sha256 {}", requested.display(), sha)]
        }
        Err(e) => vec![format!("info string nnue: {}", e)],
    }
}

fn usi_new_game(thread_pool: &mut ThreadPool, _tt: &mut TranspositionTable) {
    thread_pool.wait_for_search_finished();
    thread_pool.clear();
    // Is tt.clear() disturbed at the continuous match?
    //_tt.clear();
}

fn self_move(thread_pool: &mut ThreadPool, tt: &mut TranspositionTable, usi_options: &UsiOptions, pos: &Position) {
    let start_sfen = &pos.to_sfen();
    loop {
        let mut pos = Position::new_from_sfen(start_sfen).unwrap();
        let mut record = pos.to_sfen();
        let mut pos_map = std::collections::HashMap::new();
        loop {
            println!("position sfen {}", record);
            let key = pos.key().0;
            *pos_map.entry(key).or_insert(0) += 1;
            if *pos_map.get(&key).unwrap() == 4 {
                break;
            }
            let mut limits = LimitsType::new();
            limits.start_time = Some(std::time::Instant::now());
            limits.movetime = Some(std::time::Duration::from_millis(4000));
            let ponder_mode = false;
            let hide_all_output = false;
            thread_pool.start_thinking(&pos, tt, limits, usi_options, ponder_mode, hide_all_output);
            thread_pool.wait_for_search_finished();
            let m = thread_pool.last_best_root_move.lock().unwrap().as_ref().unwrap().pv[0];
            if m == Move::RESIGN {
                break;
            } else {
                pos.do_move(m, pos.gives_check(m));
                record += &format!(" {}", m.to_usi_string());
            }
        }
    }
}

fn position(pos: &mut Position, args: &[&str]) {
    fn position_impl(pos: &mut Position, args: &[&str]) -> Result<()> {
        if args.is_empty() {
            return Err(anyhow!(
                r#"invalid position command. expected: "startpos" or "sfen". but found nothing"#,
            ));
        }
        let mut tmp_pos;
        let args = match args[0] {
            "startpos" => {
                tmp_pos = Position::new();
                &args[1..]
            }
            "sfen" => {
                // &args[1..]:  skip "sfen".
                tmp_pos = Position::new_from_sfen_args(&args[1..]).map_err(|e| anyhow!("sfen error: {}", e))?;
                &args[5..]
            }
            _ => {
                return Err(anyhow!(
                    r#"invalid position command. expected: "startpos" or "sfen". found: "{}""#,
                    args[0]
                ));
            }
        };
        if args.is_empty() {
            *pos = tmp_pos;
            pos.reserve_states();
            return Ok(());
        }
        if args[0] != "moves" {
            return Err(anyhow!(
                r#"invalid position command. expected: "moves". found: "{}""#,
                args[0]
            ));
        }
        for arg in &args[1..] {
            let m = Move::new_from_usi_str(arg, &tmp_pos)
                .with_context(|| anyhow!("invalid move: {}, position: {}", arg, tmp_pos.to_sfen()))?;
            let gives_check = tmp_pos.gives_check(m);
            tmp_pos.do_move(m, gives_check);
        }
        *pos = tmp_pos;
        pos.reserve_states();
        Ok(())
    }

    if let Err(e) = position_impl(pos, args) {
        println!("info {}", e);
    }
}

pub fn setoption(
    args: &[&str],
    usi_options: &mut UsiOptions,
    thread_pool: &mut ThreadPool,
    tt: &mut TranspositionTable,
    reductions: &mut Reductions,
    is_ready: &mut bool,
) {
    fn setoption_impl(
        args: &[&str],
        usi_options: &mut UsiOptions,
        thread_pool: &mut ThreadPool,
        tt: &mut TranspositionTable,
        reductions: &mut Reductions,
        is_ready: &mut bool,
    ) -> Result<()> {
        if !args.is_empty() && args[0] != "name" {
            return Err(anyhow!(r#"invalid token: expected: "name", found: "{}""#, args[0]));
        }
        match args.len() {
            2 => {
                let name = args[1];
                usi_options.push_button(name, tt);
            }
            4 => {
                if args[2] != "value" {
                    return Err(anyhow!(r#"invalid token: expected: "value", found: "{}""#, args[2]));
                }
                let name = args[1];
                let value = args[3];
                usi_options.set(name, value, thread_pool, tt, reductions, is_ready);
            }
            _ => {
                return Err(anyhow!(
                    "invalid number of sections. expected: name <option-name> value <option-value> found: {}",
                    args.join(" ")
                ));
            }
        }
        Ok(())
    }

    if let Err(e) = setoption_impl(args, usi_options, thread_pool, tt, reductions, is_ready) {
        println!("info {}", e);
    }
}

fn legal_moves(pos: &Position) {
    let mut mlist = MoveList::new();
    mlist.generate::<LegalType>(pos, 0);
    for i in 0..mlist.size {
        print!("{} ", unsafe { (*mlist.ext_moves[i].as_ptr()).mv.to_usi_string() });
    }
    println!();
}

fn legal_all_moves(pos: &Position) {
    let mut mlist = MoveList::new();
    mlist.generate::<LegalAllType>(pos, 0);
    for i in 0..mlist.size {
        print!("{} ", unsafe { (*mlist.ext_moves[i].as_ptr()).mv.to_usi_string() });
    }
    println!();
}

fn bench_movegen(pos: &Position) {
    let start = std::time::Instant::now();
    let max = 5_000_000;
    let mut mlist = MoveList::new();
    for _ in 0..max {
        mlist.size = 0;
        mlist.generate::<CaptureOrPawnPromotionsType>(pos, 0);
        let size = mlist.size;
        mlist.generate::<QuietsWithoutPawnPromotionsType>(pos, size);
    }
    let end = start.elapsed();
    let elapsed = (end.as_secs() * 1000) as i64 + i64::from(end.subsec_millis());
    println!("elapsed: {} [msec]", elapsed);
    println!("times/s: {} [times/sec]", if elapsed == 0 { 0 } else { max * 1000 / elapsed });
    println!("num of moves: {}", mlist.size);
}

fn read_sfen_and_output_hcp(args: &[&str]) {
    fn read_sfen_and_output_hcp_impl(args: &[&str]) -> Result<()> {
        if args.len() != 2 {
            return Err(anyhow!("expected: <input-path> <output-path> found: {}", args.join(" ")));
        }
        let input_path = args[0];
        let output_path = args[1];
        let mut set = std::collections::HashSet::new();
        let mut v = Vec::new();
        let input_file = std::fs::File::open(input_path).map_err(|e| anyhow!("{}: {}", e, input_path))?;
        for line in std::io::BufReader::new(input_file).lines() {
            let line = line.unwrap();
            let args = line.split_whitespace().collect::<Vec<&str>>();
            if args.is_empty() {
                continue;
            }
            let mut pos;
            let args = match args[0] {
                "startpos" => {
                    pos = Position::new();
                    &args[1..]
                }
                "sfen" => {
                    // &args[1..]:  skip "sfen".
                    match Position::new_from_sfen_args(&args[1..]) {
                        Ok(tmp_pos) => pos = tmp_pos,
                        Err(e) => {
                            println!("info sfen error: {}", e);
                            continue;
                        }
                    }
                    &args[5..]
                }
                _ => {
                    println!(
                        r#"info invalid position command. expected: "startpos" or "sfen". found: "{}""#,
                        args[0]
                    );
                    continue;
                }
            };
            if args.is_empty() {
                pos.reserve_states();
                continue;
            }
            if args[0] != "moves" {
                println!(r#"info invalid position command. expected: "moves". found: "{}""#, args[0]);
                continue;
            }

            if !set.contains(&pos.key()) {
                set.insert(pos.key());
                v.push(HuffmanCodedPosition::from(&pos));
            }
            for arg in &args[1..] {
                if let Some(m) = Move::new_from_usi_str(arg, &pos) {
                    let gives_check = pos.gives_check(m);
                    pos.do_move(m, gives_check);
                    if !set.contains(&pos.key()) {
                        set.insert(pos.key());
                        v.push(HuffmanCodedPosition::from(&pos));
                    }
                } else {
                    println!("info invalid move: {}, position: {}", arg, pos.to_sfen());
                    break;
                }
            }
        }
        let mut output_file =
            std::io::BufWriter::new(std::fs::File::create(output_path).map_err(|e| anyhow!("{}: {}", e, output_path))?);
        let slice: &[u8] = unsafe {
            std::slice::from_raw_parts(
                v.as_slice().as_ptr() as *const u8,
                std::mem::size_of::<HuffmanCodedPosition>() * v.len(),
            )
        };
        output_file.write_all(slice).unwrap();
        Ok(())
    }

    if let Err(e) = read_sfen_and_output_hcp_impl(args) {
        println!("info {}", e);
    }
}

// debug code
fn read_hcp(args: &[&str]) {
    fn read_hcp_impl(args: &[&str]) -> Result<()> {
        if args.len() != 2 {
            return Err(anyhow!(
                "read_hcp error. expected: <input-path> <output-path> found: {}",
                args.join(" ")
            ));
        }
        let input_path = args[0];
        let output_path = args[1];
        let v = file_to_vec(input_path).map_err(|e| anyhow!("{}: {}", e, input_path))?;
        let mut output_file =
            std::io::BufWriter::new(std::fs::File::create(output_path).map_err(|e| anyhow!("{}: {}", e, output_path))?);
        for item in v {
            let pos = Position::new_from_huffman_coded_position(&item)?;
            writeln!(output_file, "{}", pos.to_sfen())?;
        }
        Ok(())
    }

    if let Err(e) = read_hcp_impl(args) {
        println!("{}", e);
    }
}

fn read_csa_dirs_and_output_sfen(dir_paths: &[&str]) {
    for dir_path in dir_paths.iter() {
        for path in std::fs::read_dir(dir_path).unwrap() {
            let path = path.unwrap().path().display().to_string();
            let mut f = std::fs::File::open(&path).unwrap();
            let mut buf = Vec::new();
            f.read_to_end(&mut buf).unwrap();
            if let Ok(sfen) = csa_record_to_sfen(&buf) {
                println!("{}", sfen);
            }
        }
    }
}

fn csa_record_to_sfen(csa: &[u8]) -> Result<String> {
    enum Phase {
        InitialPositionAndOptionalInformation,
        Moves,
    }
    let mut phase = Phase::InitialPositionAndOptionalInformation;
    let mut _version = None;
    let mut _player_black = None;
    let mut _player_white = None;
    let mut _event = None;
    let mut _site = None;
    let mut _start_time = None;
    let mut _end_time = None;
    let mut _time_limit = None;
    let mut _opening = None;
    let mut pos = Position::new();
    let mut s = format!("sfen {} moves", START_SFEN);
    for line in csa.split(|num_as_ascii| *num_as_ascii == b'\n') {
        match phase {
            Phase::InitialPositionAndOptionalInformation => {
                if line.starts_with(b"'") {
                    // line is a comment.
                    continue;
                } else if line.starts_with(b"V") {
                    _version = Some(line);
                } else if line.starts_with(b"N+") {
                    _player_black = Some(line);
                } else if line.starts_with(b"N-") {
                    _player_white = Some(line);
                } else if line.starts_with(b"$EVENT:") {
                    _event = Some(line);
                } else if line.starts_with(b"$SITE:") {
                    _site = Some(line);
                } else if line.starts_with(b"$START_TIME:") {
                    _start_time = Some(line);
                } else if line.starts_with(b"$END_TIME:") {
                    _end_time = Some(line);
                } else if line.starts_with(b"$TIME_LIMIT:") {
                    _time_limit = Some(line);
                } else if line.starts_with(b"$OPENING:") {
                    _opening = Some(line);
                } else if line.starts_with(b"P") {
                    // start position
                    // todo: allow any position.
                } else if line == b"+" || line == b"-" {
                    phase = Phase::Moves;
                }
            }
            Phase::Moves => {
                if line.starts_with(b"'") {
                    // line is a comment.
                    continue;
                } else if line.starts_with(b"%") {
                    // game end.
                } else if line.starts_with(b"+") || line.starts_with(b"-") {
                    // black or white player's move
                    let line = std::str::from_utf8(&line[1..])?;
                    let m = Move::new_from_csa_str(line, &pos).context("illegal move")?;
                    s += &format!(" {}", m.to_usi_string());
                    let gives_check = pos.gives_check(m);
                    pos.do_move(m, gives_check);
                } else if line.starts_with(b"T") {
                    // consumption time
                }
            }
        }
    }
    Ok(s)
}

pub fn cmd_loop() {
    let mut tt = TranspositionTable::new();
    let mut reductions = Reductions::new();
    let mut thread_pool = ThreadPool::new();
    thread_pool.set(1, &mut tt, &mut reductions);
    let mut usi_options = UsiOptions::new();
    let mut pos = Position::new();
    let mut is_ready = false;
    loop {
        let cmd = if std::env::args().len() == 1 {
            let mut cmd = String::new();
            // std::io::stdin().read_line() includes "\n"
            match std::io::stdin().read_line(&mut cmd) {
                Ok(0) | Err(_) => cmd = String::from("quit"), // if read EOF, be Ok(0).
                Ok(_) => cmd = cmd.trim().to_string(),
            }
            cmd
        } else {
            let mut cmd = String::new();
            for arg in std::env::args().skip(1) {
                cmd.push_str(&arg);
                cmd.push(' ');
            }
            cmd
        };
        let args: Vec<&str> = cmd.split_whitespace().collect();
        let token = if args.is_empty() { "" } else { args[0] }; // if read "\n", args is empty.

        match token {
            // Required commands as USI protocol.
            "gameover" | "quit" | "stop" => {
                thread_pool.stop.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            "go" => {
                if is_ready {
                    if let Err(err) = go(&mut thread_pool, &mut tt, &usi_options, &pos, &args[1..]) {
                        println!("info {}", err);
                    }
                } else {
                    println!(r#"info error. "isready" command is needed in advance."#);
                }
            }
            "isready" => {
                isready(&mut is_ready, &mut usi_options, &mut thread_pool, &mut tt, &mut reductions);
            }
            "ponderhit" => {
                thread_pool.ponder.store(false, std::sync::atomic::Ordering::Relaxed);
            }
            "position" => position(&mut pos, &args[1..]),
            "setoption" => setoption(
                &args[1..],
                &mut usi_options,
                &mut thread_pool,
                &mut tt,
                &mut reductions,
                &mut is_ready,
            ),
            "usi" => {
                let mut s = format!("id name {}", crate::engine_name::ENGINE_NAME);
                s += &format!("\nid author {}", crate::authors::AUTHORS);
                s += &format!("\n{}", usi_options.to_usi_string());
                s += "\nusiok";
                println!("{}", s);
            }
            "usinewgame" => usi_new_game(&mut thread_pool, &mut tt),
            // Not required commands as USI protocol.
            "bench_movegen" => bench_movegen(&pos),
            "d" => pos.print(),
            "eval" => {
                if is_ready {
                    let mut stack = vec![Stack::new(); CURRENT_STACK_INDEX + 1];
                    println!("{}", evaluate_at_root(&pos, &mut stack).0);
                } else {
                    println!(r#"info error. "isready" command is needed in advance."#);
                }
            }
            "generate_teachers" => {
                if is_ready {
                    if let Err(e) = generate_teachers(&args[1..]) {
                        println!("info {}", e);
                    }
                } else {
                    println!(r#"info error. "isready" command is needed in advance."#);
                }
            }
            "key" => println!("{}", pos.key().0),
            "legal_moves" => legal_moves(&pos),
            "legal_all_moves" => legal_all_moves(&pos),
            "self_move" => self_move(&mut thread_pool, &mut tt, &usi_options, &pos),
            "read_csa_dirs_and_output_sfen" => read_csa_dirs_and_output_sfen(&args[1..]),
            "read_hcp" => read_hcp(&args[1..]),
            "read_sfen_and_output_hcp" => read_sfen_and_output_hcp(&args[1..]),
            "wait" => thread_pool.wait_for_search_finished(),
            _ => println!("info unknown command: {}", cmd),
        }
        if std::env::args().len() > 1 || token == "quit" {
            break;
        }
    }
}

#[cfg(test)]
mod parse_go_limits_tests {
    use super::*;

    // The reported main clock must reach the time manager untrimmed (no parse-time haircut).
    #[test]
    fn fischer_clock_arrives_untrimmed() {
        let opts = UsiOptions::new();
        let (limits, ponder) =
            parse_go_limits(&opts, &["btime", "120000", "wtime", "118000", "binc", "2000", "winc", "2000"]).unwrap();
        assert_eq!(limits.time[Color::BLACK.0 as usize].as_millis(), 120_000);
        assert_eq!(limits.time[Color::WHITE.0 as usize].as_millis(), 118_000);
        assert_eq!(limits.inc[Color::BLACK.0 as usize].as_millis(), 2_000);
        assert_eq!(limits.inc[Color::WHITE.0 as usize].as_millis(), 2_000);
        assert!(limits.movetime.is_none(), "no byoyomi token must mean no movetime cap");
        assert!(
            limits.use_time_management(),
            "the Fischer line must take the time-managed path"
        );
        assert!(!ponder);
    }

    // The byoyomi path bypasses the time manager, so Byoyomi_Margin is its one margin.
    #[test]
    fn byoyomi_is_margined() {
        let opts = UsiOptions::new();
        let (limits, _) = parse_go_limits(&opts, &["btime", "0", "wtime", "0", "byoyomi", "10000"]).unwrap();
        assert_eq!(limits.movetime.unwrap().as_millis(), 9_500);
        assert!(!limits.use_time_management());

        // A period shorter than the margin clamps to zero instead of underflowing.
        let (limits, _) = parse_go_limits(&opts, &["byoyomi", "300"]).unwrap();
        assert_eq!(limits.movetime.unwrap().as_millis(), 0);
    }

    // `movetime` is a GUI-fixed think time, not a clock deadline: honoured exactly, no margin.
    #[test]
    fn movetime_is_exact() {
        let opts = UsiOptions::new();
        let (limits, _) = parse_go_limits(&opts, &["movetime", "5000"]).unwrap();
        assert_eq!(limits.movetime.unwrap().as_millis(), 5_000);
    }

    #[test]
    fn rejects_bad_tokens() {
        let opts = UsiOptions::new();
        assert!(parse_go_limits(&opts, &["btime"]).is_err(), "missing number");
        assert!(parse_go_limits(&opts, &["btime", "abc"]).is_err(), "non-numeric");
        assert!(parse_go_limits(&opts, &["unknown_token"]).is_err(), "unknown token");
    }
}

// Tournament build: the byoyomi margin subtracted during `go` parsing is the compile-time const
// baked by build.rs, not the runtime USI option.
#[cfg(all(test, feature = "tournament"))]
mod tournament_const_tests {
    use super::*;

    #[test]
    fn byoyomi_margin_const_is_fixed() {
        // `const` context: compiles only if this is a genuine compile-time const.
        const BYOYOMI_MARGIN: i64 = crate::tournament::BYOYOMI_MARGIN;
        assert_eq!(BYOYOMI_MARGIN, 500, "v1 tournament config bakes Byoyomi_Margin = 500");
    }

    // The accessor resolves to the baked const under `tournament`; the `UsiOptions` argument is
    // ignored, so the byoyomi period is margined by the const regardless of the runtime option.
    #[test]
    fn byoyomi_margin_accessor_returns_const() {
        let opts = UsiOptions::new();
        assert_eq!(byoyomi_margin(&opts), crate::tournament::BYOYOMI_MARGIN);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csa_record_to_sfne() {
        use std::fs::File;
        use std::io::prelude::*;
        let mut f = File::open("test/example.csa").unwrap();
        let mut buf = Vec::new();
        f.read_to_end(&mut buf).unwrap();
        if let Ok(sfen) = csa_record_to_sfen(&buf) {
            assert_eq!(
                sfen,
                "sfen lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1 moves 2g2f 3c3d"
            );
        }
    }

    #[cfg(feature = "nnue")]
    #[test]
    fn nnue_isready_lines_reports_empty_eval_dir() {
        let _guard = crate::evaluate::nnue::TEST_MUTEX.lock().expect("TEST_MUTEX poisoned");
        crate::evaluate::nnue::clear_loaded_for_test();

        let lines = nnue_isready_lines("");
        assert_eq!(lines, vec!["info string nnue: Eval_Dir is not set".to_string()]);
        assert!(!crate::evaluate::nnue::is_loaded());
    }

    #[cfg(feature = "nnue")]
    #[test]
    fn nnue_isready_lines_surfaces_loader_error_without_mutating_slot() {
        let _guard = crate::evaluate::nnue::TEST_MUTEX.lock().expect("TEST_MUTEX poisoned");
        crate::evaluate::nnue::clear_loaded_for_test();

        let missing_dir = "/definitely-does-not-exist-28-wire";
        let lines = nnue_isready_lines(missing_dir);
        assert_eq!(lines.len(), 1, "expected exactly one info line, got {:?}", lines);
        assert!(
            lines[0].starts_with("info string nnue: "),
            "line does not carry the nnue prefix: {}",
            lines[0]
        );
        assert!(!crate::evaluate::nnue::is_loaded());
    }

    #[cfg(feature = "nnue")]
    #[test]
    fn nnue_isready_lines_is_idempotent_for_same_path() {
        use std::sync::Arc;
        let _guard = crate::evaluate::nnue::TEST_MUTEX.lock().expect("TEST_MUTEX poisoned");
        crate::evaluate::nnue::clear_loaded_for_test();

        let eval_dir = "/tmp/yorkie-nnue-idempotent";
        let path = std::path::PathBuf::from(format!("{}/nn.bin", eval_dir));
        let mut sha = [0u8; 32];
        sha[0] = 0x01;
        sha[31] = 0xEE;
        let net = Arc::new(crate::evaluate::nnue::make_placeholder_network(sha));
        crate::evaluate::nnue::set_loaded_for_test(net, path.clone());

        let lines = nnue_isready_lines(eval_dir);
        assert!(lines.is_empty(), "expected no lines for already-loaded path, got {:?}", lines);
        assert!(crate::evaluate::nnue::is_loaded());

        crate::evaluate::nnue::clear_loaded_for_test();
    }
}
