use crate::book::*;
#[cfg(feature = "material")]
use crate::evaluate::material::*;
#[cfg(feature = "nnue")]
use crate::evaluate::nnue::{self, current_network, evaluate, evaluate_at_root};
use crate::movegen::*;
use crate::movepick::*;
use crate::movetypes::*;
use crate::piecevalue::*;
use crate::position::*;
use crate::search::*;
use crate::timeman::*;
use crate::tt::*;
use crate::types::*;
use crate::usioption::*;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// `USI_Ponder` (bool): whether to emit a `ponder` move after `bestmove`; tournament build folds the const, else the runtime USI option.
#[cfg(feature = "tournament")]
#[inline]
fn usi_ponder(_usi_options: &UsiOptions) -> bool {
    crate::tournament::USI_PONDER
}

#[cfg(not(feature = "tournament"))]
#[inline]
fn usi_ponder(usi_options: &UsiOptions) -> bool {
    usi_options.get_bool(UsiOptions::USI_PONDER)
}

pub struct StatsType;
impl StatsType {
    pub const NUM: usize = 2;
}

pub struct InCheckType;
impl InCheckType {
    pub const NUM: usize = 2;
}

struct Thread {
    idx: usize,
    // Root PV index advanced by the multi-PV loop; non-tournament only (tournament folds `pv_idx` to const 0).
    #[cfg(not(feature = "tournament"))]
    pv_idx: usize,
    tt_hit_average: u64,
    sel_depth: i32,
    null_move_pruning_min_ply: i32,
    null_move_pruning_color: Color,
    // The engine's own game side (set per search in start_thinking, ponder-aware): its mates are flat, mates against it stay graded. Pinning to color, not root side, keeps TT entries consistent across ponder and normal searches.
    engine_color: Color,
    position: Position,
    root_moves: RootMoves,
    root_depth: Depth,
    completed_depth: Depth,
    counter_moves: CounterMoveHistory,
    main_history: ButterflyHistory,
    low_ply_history: LowPlyHistory,
    capture_history: CapturePieceToHistory,
    continuation_history: [[ContinuationHistory; StatsType::NUM]; InCheckType::NUM],
    limits: LimitsType, // Clone from ThreadPool for fast access.
    // Game-ply draw horizon (maximum-moves rule); snapshot per search, 0 = unlimited = i32::MAX.
    // Non-tournament only: under `tournament` the horizon is a compile-time const.
    #[cfg(not(feature = "tournament"))]
    max_moves_to_draw: i32,
    // Per-side draw contempt (Value units), snapshot of Draw_Contempt once per search for a plain field read.
    #[cfg(not(feature = "tournament"))]
    draw_contempt: i32,
    tt: *mut TranspositionTable,
    timeman: Arc<Mutex<TimeManagement>>, // shold I use pointer for speedup?
    reductions: *mut Reductions,
    usi_options: UsiOptions,
    best_move_changes: Arc<AtomicU64>,
    best_move_changess: Vec<Arc<AtomicU64>>,

    nodes: Arc<AtomicI64>,
    // Cached per search so descent sites avoid re-locking the process-wide RwLock; populated in iterative_deepening_loop.
    #[cfg(feature = "nnue")]
    nnue_network: Option<Arc<nnue::types::NnueNetwork>>,
    // Node-local NNUE replica for the incremental path (raw pointer for a disjoint-field read alongside `&mut self.position`); set per search in `iterative_deepening_loop`, null until then.
    #[cfg(all(feature = "nnue", feature = "numa"))]
    nnue_net_ptr: *const nnue::types::NnueNetwork,
    // following variables are shared one object that ThreadPool has.
    best_previous_score: Arc<Mutex<Value>>,
    iter_values: Arc<Mutex<[Value; 4]>>,
    increase_depth: Arc<AtomicBool>,
    // following variables are used only main thread.
    previous_time_reduction: f64,
    calls_count: i32,
    stop_on_ponderhit: Arc<AtomicBool>,
    ponder: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    // Non-tournament only: guards the in-search info output (tournament compiles it out; bestmove stays guarded by `ThreadPool::hide_all_output`).
    #[cfg(not(feature = "tournament"))]
    hide_all_output: Arc<AtomicBool>,
    nodess: Vec<Arc<AtomicI64>>,
}

unsafe impl std::marker::Send for Thread {} // for Thread::tt

struct ThreadPoolBase {
    threads: Vec<Arc<Mutex<Thread>>>,
}

pub struct ThreadPool {
    thread_pool_base: Arc<Mutex<ThreadPoolBase>>,
    nodess: Vec<Arc<AtomicI64>>,
    pub book: Option<Book>,
    timeman: Arc<Mutex<TimeManagement>>,
    best_previous_score: Arc<Mutex<Value>>,
    iter_values: Arc<Mutex<[Value; 4]>>,
    best_move_changess: Vec<Arc<AtomicU64>>,
    stop_on_ponderhit: Arc<AtomicBool>,
    pub ponder: Arc<AtomicBool>,
    pub stop: Arc<AtomicBool>,
    increase_depth: Arc<AtomicBool>,
    pub hide_all_output: Arc<AtomicBool>,
    pub limits: LimitsType,
    pub last_best_root_move: Arc<Mutex<Option<RootMove>>>, // Not for usi engine. For debug or some tools.
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Thread {
    fn is_main(&self) -> bool {
        self.idx == 0
    }
    fn clear(&mut self) {
        self.calls_count = 0;
        self.counter_moves.fill(None);
        self.main_history.fill(0);
        self.low_ply_history.fill(0);
        self.capture_history.fill(0);

        self.continuation_history.iter_mut().for_each(|x| {
            x.iter_mut().for_each(|y| {
                y.fill(0);
                y.v[Piece::EMPTY.0 as usize][0].fill(COUNTER_MOVE_PRUNE_THRESHOLD - 1);
            })
        });
    }
    /// Game-ply draw horizon (maximum-moves rule); tournament build folds a compile-time const, else the per-search `Max_Moves_To_Draw` snapshot.
    #[cfg(feature = "tournament")]
    #[inline]
    fn max_moves_to_draw(&self) -> i32 {
        crate::tournament::MAX_MOVES_TO_DRAW
    }

    #[cfg(not(feature = "tournament"))]
    #[inline]
    fn max_moves_to_draw(&self) -> i32 {
        self.max_moves_to_draw
    }

    /// Per-side draw contempt (`Value` units) for [`crate::search::draw_value`]; tournament build folds the const, else the per-search `Draw_Contempt` snapshot.
    #[cfg(feature = "tournament")]
    #[inline]
    fn draw_contempt(&self) -> i32 {
        crate::tournament::DRAW_CONTEMPT
    }

    #[cfg(not(feature = "tournament"))]
    #[inline]
    fn draw_contempt(&self) -> i32 {
        self.draw_contempt
    }

    /// Index of the root PV being searched; tournament folds this to const `0`, else the runtime `pv_idx` field.
    #[cfg(feature = "tournament")]
    #[inline]
    fn pv_idx(&self) -> usize {
        0
    }

    #[cfg(not(feature = "tournament"))]
    #[inline]
    fn pv_idx(&self) -> usize {
        self.pv_idx
    }

    fn iterative_deepening_loop(&mut self) {
        // Snapshot the draw horizon once per search from the runtime USI option (0 = unlimited = i32::MAX);
        // tournament build reads a compile-time const instead, so there is no field to populate.
        #[cfg(not(feature = "tournament"))]
        {
            self.max_moves_to_draw = match self.usi_options.get_i64(UsiOptions::MAX_MOVES_TO_DRAW) {
                0 => i32::MAX, // 0 = unlimited
                limit => limit as i32,
            };
            // Per-side draw contempt; default 0 keeps a draw neutral (today's behavior).
            self.draw_contempt = self.usi_options.get_i64(UsiOptions::DRAW_CONTEMPT) as i32;
        }
        let mut stack: [Stack; MAX_PLY as usize + 10] = std::array::from_fn(|_| Stack::new());
        let mut best_value = -Value::INFINITE;
        let mut last_best_move = None;
        let mut last_best_move_depth = Depth::ZERO; // not Option<Depth>
        let mut delta = -Value::INFINITE;
        let mut alpha = -Value::INFINITE;
        let mut beta = Value::INFINITE;
        let mut time_reduction = 1.0;
        let mut total_best_move_changes = 0.0f64;
        // Non-tournament only: paces the in-search info output (tournament compiles it out).
        #[cfg(not(feature = "tournament"))]
        let mut last_info_time: Option<std::time::Instant> = None;
        let mut iter_index = 0;
        for item in stack.iter_mut().take(CURRENT_STACK_INDEX) {
            item.continuation_history = self.continuation_history[0][0].sentinel();
        }
        for i in 0..=MAX_PLY + 2 {
            get_stack_mut(&mut stack, i64::from(i)).ply = i;
        }
        if self.is_main() {
            let best_previous_score = *self.best_previous_score.lock().unwrap();
            if best_previous_score == Value::INFINITE {
                for item in self.iter_values.lock().unwrap().iter_mut() {
                    *item = Value::ZERO;
                }
            } else {
                for item in self.iter_values.lock().unwrap().iter_mut() {
                    *item = best_previous_score;
                }
            }
        }

        self.low_ply_history.keep_data_from_previous_search();

        // Number of root PVs to search: the `MultiPV` option clamped to legal-move count, or a single PV under `tournament`.
        #[cfg(not(feature = "tournament"))]
        let multi_pv = std::cmp::min(self.usi_options.get_i64(UsiOptions::MULTI_PV) as usize, self.root_moves.len());
        self.tt_hit_average = TT_HIT_AVERAGE_WINDOW * TT_HIT_AVERAGE_RESOLUTION / 2;

        let mut search_again_counter = 0;

        #[cfg(feature = "nnue")]
        {
            self.nnue_network = Some(current_network().expect("nnue network must be loaded before iterative_deepening_loop"));
            // Point the incremental path at this node's replica, else the shared Arc's inner pointer (kept alive by `nnue_network`).
            #[cfg(feature = "numa")]
            {
                self.nnue_net_ptr =
                    nnue::local_replica().map_or_else(|| Arc::as_ptr(self.nnue_network.as_ref().unwrap()), |r| r as *const _);
            }
        }

        evaluate_at_root(&self.position, &mut stack);
        while {
            self.root_depth += Depth::ONE_PLY;
            self.root_depth
        } < Depth::MAX
            && !self.stop.load(Ordering::Relaxed)
            && !(self.limits.depth.is_some() && self.is_main() && self.root_depth.0 > Depth(self.limits.depth.unwrap() as i32).0)
        {
            if self.idx > 0 {
                let i = (self.idx - 1) % 20;
                if ((self.root_depth.0 + SKIP_PHASE[i]) / SKIP_SIZE[i]) % 2 != 0 {
                    continue;
                }
            }

            if self.is_main() {
                total_best_move_changes /= 2.0;
            }

            for rm in self.root_moves.iter_mut() {
                rm.previous_score = rm.score;
            }

            #[cfg(not(feature = "tournament"))]
            {
                self.pv_idx = 0;
            }

            if !self.increase_depth.load(Ordering::Relaxed) {
                search_again_counter += 1;
            }
            // Root PV search: loops `pv_idx` over the `multi_pv` best moves; under `tournament` it runs once (single PV) and the trailing break collapses the loop (hence the `never_loop` allow).
            #[cfg_attr(feature = "tournament", allow(clippy::never_loop))]
            loop {
                if self.stop.load(Ordering::Relaxed) {
                    break;
                }
                #[cfg(not(feature = "tournament"))]
                if self.pv_idx >= multi_pv {
                    break;
                }
                self.sel_depth = 0;
                if self.root_depth >= Depth(4) {
                    let previous_score = self.root_moves[self.pv_idx()].previous_score;
                    delta = Value(17);
                    alpha = std::cmp::max(previous_score - delta, -Value::INFINITE);
                    beta = std::cmp::min(previous_score + delta, Value::INFINITE);
                }

                let mut failed_high_count = 0;
                loop {
                    let adjusted_depth = std::cmp::max(
                        Depth::ONE_PLY,
                        self.root_depth - Depth(failed_high_count + search_again_counter),
                    );
                    best_value = self.search::<RootType>(&mut stack, alpha, beta, adjusted_depth, false);
                    let pv_idx = self.pv_idx();
                    self.root_moves[pv_idx..].sort_by(|x, y| y.cmp(x));
                    if self.stop.load(Ordering::Relaxed) {
                        break;
                    }
                    // In-search info output on an aspiration fail high/low; tournament compiles it out.
                    #[cfg(not(feature = "tournament"))]
                    if self.is_main()
                        && multi_pv == 1
                        && (best_value <= alpha || beta <= best_value)
                        && self.timeman.lock().unwrap().elapsed() > 3000
                        && (self.root_depth < Depth(10)
                            || last_info_time.is_none()
                            || last_info_time.unwrap().elapsed().as_millis() > 200)
                    {
                        last_info_time = Some(std::time::Instant::now());
                        if !self.hide_all_output.load(Ordering::Relaxed) {
                            println!(
                                "{}",
                                self.pv_info_to_usi_string(self.nodes_searched(), multi_pv, self.root_depth, alpha, beta, false,)
                            );
                        }
                    }
                    if best_value <= alpha {
                        beta = (alpha + beta) / 2;
                        alpha = std::cmp::max(best_value - delta, -Value::INFINITE);

                        failed_high_count = 0;
                        if self.is_main() {
                            self.stop_on_ponderhit.store(false, Ordering::Relaxed);
                        }
                    } else if beta <= best_value {
                        beta = std::cmp::min(best_value + delta, Value::INFINITE);
                        failed_high_count += 1;
                    } else {
                        break;
                    }

                    delta += delta / 4 + Value(5);
                    debug_assert!(-Value::INFINITE <= alpha && beta <= Value::INFINITE);
                }

                let pv_idx = self.pv_idx();
                self.root_moves[0..=pv_idx].sort_by(|x, y| y.cmp(x));

                // Per-iteration PV info output; tournament compiles it out.
                #[cfg(not(feature = "tournament"))]
                if self.is_main()
                    && (self.stop.load(Ordering::Relaxed)
                        || self.pv_idx() + 1 == multi_pv
                        || self.timeman.lock().unwrap().elapsed() > 3000)
                    && (self.root_depth < Depth(10)
                        || last_info_time.is_none()
                        || last_info_time.unwrap().elapsed().as_millis() > 200)
                {
                    last_info_time = Some(std::time::Instant::now());
                    if !self.hide_all_output.load(Ordering::Relaxed) {
                        println!(
                            "{}",
                            self.pv_info_to_usi_string(self.nodes_searched(), multi_pv, self.root_depth, alpha, beta, false,)
                        );
                    }
                }

                #[cfg(not(feature = "tournament"))]
                {
                    self.pv_idx += 1;
                }
                #[cfg(feature = "tournament")]
                break;
            }

            if !self.stop.load(Ordering::Relaxed) {
                self.completed_depth = self.root_depth;
            }

            if last_best_move.is_none() || last_best_move.non_zero_unwrap_unchecked() != self.root_moves[0].pv[0] {
                last_best_move = Some(self.root_moves[0].pv[0]);
                last_best_move_depth = self.root_depth;
            }

            if let Some(mate) = self.limits.mate
                && best_value >= Value::MATE_IN_MAX_PLY
                && Value::MATE - best_value <= Value(mate as i32)
            {
                self.stop.store(true, Ordering::Relaxed);
            }

            // Flat-win termination: a proven flat win is the global maximum, so stop rather than spend more time (but `go infinite` keeps searching, and while pondering defer via stop_on_ponderhit).
            if best_value == Value::MATE_FLAT && self.limits.infinite.is_none() && !self.stop.load(Ordering::Relaxed) {
                if self.ponder.load(Ordering::Relaxed) {
                    self.stop_on_ponderhit.store(true, Ordering::Relaxed);
                } else {
                    self.stop.store(true, Ordering::Relaxed);
                }
            }

            if !self.is_main() {
                continue;
            }

            if self.limits.use_time_management()
                && !self.stop.load(Ordering::Relaxed)
                && !self.stop_on_ponderhit.load(Ordering::Relaxed)
            {
                let falling_eval = f64::from(
                    318 + 6 * (self.best_previous_score.lock().unwrap().0 - best_value.0)
                        + 6 * (self.iter_values.lock().unwrap()[iter_index].0 - best_value.0),
                ) / 825.0;
                let falling_eval = num::clamp(falling_eval, 0.5, 1.5);
                time_reduction = if last_best_move_depth.0 + 9 < self.completed_depth.0 {
                    1.92
                } else {
                    0.95
                };
                let reduction = (1.47 + self.previous_time_reduction) / (2.32 * time_reduction);
                for best_move_changes in self.best_move_changess.iter() {
                    total_best_move_changes += best_move_changes.load(Ordering::Relaxed) as f64;
                    best_move_changes.store(0, Ordering::Relaxed);
                }
                let best_move_instability = 1.073
                    + 1.0f64.max(2.25 - 9.9 / self.root_depth.0 as f64) * total_best_move_changes
                        / self.best_move_changess.len() as f64;
                let (elapsed, optimum_millis) = {
                    let timeman = self.timeman.lock().unwrap();
                    (timeman.elapsed(), timeman.optimum_millis())
                };
                let total_time = {
                    let total_time = (optimum_millis as f64 * falling_eval * reduction * best_move_instability) as i64;
                    if self.root_moves.len() == 1 {
                        std::cmp::min(500, total_time)
                    } else {
                        total_time
                    }
                };
                if elapsed > total_time {
                    if self.ponder.load(Ordering::Relaxed) {
                        self.stop_on_ponderhit.store(true, Ordering::Relaxed);
                    } else {
                        self.stop.store(true, Ordering::Relaxed);
                    }
                } else if self.increase_depth.load(Ordering::Relaxed)
                    && self.ponder.load(Ordering::Relaxed)
                    && elapsed as f64 > total_time as f64 * 0.58
                {
                    self.increase_depth.store(false, Ordering::Relaxed);
                } else {
                    self.increase_depth.store(true, Ordering::Relaxed);
                }
            }
            self.iter_values.lock().unwrap()[iter_index] = best_value;
            iter_index = (iter_index + 1) & 3;
        }

        if !self.is_main() {
            return;
        }

        self.previous_time_reduction = time_reduction;
    }
    fn search<NT: NodeTypeTrait>(
        &mut self,
        stack: &mut [Stack],
        alpha: Value,
        beta: Value,
        mut depth: Depth,
        cut_node: bool,
    ) -> Value {
        let pv_node = NT::NODE_TYPE != NON_PV;
        let root_node = NT::NODE_TYPE == ROOT;
        let max_next_depth = if root_node { depth } else { depth + Depth::ONE_PLY };

        if depth < Depth::ONE_PLY {
            // Is there a better way?
            // I want to set a generic parameters in compile-time calculations.
            return if pv_node {
                self.qsearch::<PvType>(stack, alpha, beta, Depth::ZERO)
            } else {
                self.qsearch::<NonPvType>(stack, alpha, beta, Depth::ZERO)
            };
        }

        debug_assert!(-Value::INFINITE <= alpha && alpha < beta && beta <= Value::INFINITE);
        debug_assert!(pv_node || (alpha == beta - Value(1)));
        debug_assert!(Depth::ZERO < depth && depth < Depth::MAX);
        debug_assert!(!(pv_node && cut_node));

        // Step 1
        get_stack_mut(stack, 0).in_check = self.position.in_check();
        let prior_capture = self.position.captured_piece();
        let us = self.position.side_to_move();
        let mut best_value = -Value::INFINITE;
        let max_value = Value::INFINITE;

        if self.is_main() {
            self.check_time();
        }

        if pv_node && self.sel_depth < get_stack(stack, 0).ply + 1 {
            self.sel_depth = get_stack(stack, 0).ply + 1;
        }

        let mut alpha = alpha;
        let mut beta = beta;
        if !root_node {
            // Step 2
            match self.position.is_repetition() {
                Repetition::Not => {
                    if self.stop.load(Ordering::Relaxed) || get_stack(stack, 0).ply >= MAX_PLY {
                        return if get_stack(stack, 0).ply >= MAX_PLY && !get_stack(stack, 0).in_check {
                            evaluate(&mut self.position, stack)
                        } else {
                            value_draw(self.nodes.load(Ordering::Relaxed))
                        };
                    }
                    // Maximum-moves rule: past the limit is a draw; checked after repetition so a boundary perpetual check stays a loss.
                    if self.position.ply() > self.max_moves_to_draw() {
                        return draw_value(self.draw_contempt(), self.position.side_to_move());
                    }
                }
                Repetition::Draw => return draw_value(self.draw_contempt(), self.position.side_to_move()),
                // Perpetual-check outcomes are immediate here (within the horizon): flat when the engine's side wins, graded when it loses.
                Repetition::Win => {
                    return if us == self.engine_color {
                        Value::MATE_FLAT
                    } else {
                        Value::mate_in(get_stack(stack, 0).ply)
                    };
                }
                Repetition::Lose => {
                    return if us == self.engine_color {
                        Value::mated_in(get_stack(stack, 0).ply)
                    } else {
                        Value::MATED_FLAT
                    };
                }
                Repetition::Superior => {
                    if get_stack(stack, 0).ply != 2 {
                        return Value::MATE_IN_MAX_PLY;
                    }
                }
                Repetition::Inferior => {
                    if get_stack(stack, 0).ply != 2 {
                        return Value::MATED_IN_MAX_PLY;
                    }
                }
            }

            // Step 3: mate-distance pruning, split by side because only the graded direction has a distance to clamp (the flat win/loss already gives the cutoff).
            if us == self.engine_color {
                alpha = std::cmp::max(Value::mated_in(get_stack(stack, 0).ply), alpha);
            } else {
                beta = std::cmp::min(Value::mate_in(get_stack(stack, 0).ply + 1), beta);
            }
            if alpha >= beta {
                return alpha;
            }
        }

        debug_assert!(0 <= get_stack(stack, 0).ply && get_stack(stack, 0).ply < MAX_PLY);

        get_stack_mut(stack, 1).tt_pv = false;
        let mut best_move: Option<Move> = None;
        get_stack_mut(stack, 1).excluded_move = None;
        get_stack_mut(stack, 2).killers[0] = None;
        get_stack_mut(stack, 2).killers[1] = None;
        get_stack_mut(stack, 0).double_extensions = get_stack(stack, -1).double_extensions;

        // get_stack(stack, -1).current_move can be None. None => prev_sq: Square(0)
        let prev_sq = get_stack(stack, -1).current_move.non_zero_unwrap_unchecked().to(); // todo: Move::NULL

        if !root_node {
            get_stack_mut(stack, 2).stat_score = 0;
        }

        // Step 4
        let excluded_move = get_stack(stack, 0).excluded_move;
        let key = if let Some(excluded_move) = excluded_move {
            Key(self.position.key().0 ^ Key::make_key(u64::from(excluded_move.0.get())).0)
        } else {
            self.position.key()
        };
        let tte = {
            let (tte, tt_hit) = unsafe { (*self.tt).probe(key) };
            get_stack_mut(stack, 0).tt_hit = tt_hit;
            tte
        };
        let tt_value = if get_stack(stack, 0).tt_hit {
            value_from_tt(tte.value(), get_stack(stack, 0).ply)
        } else {
            Value::NONE
        };
        let tt_move = if root_node {
            Some(self.root_moves[self.pv_idx()].pv[0])
        } else if get_stack(stack, 0).tt_hit {
            tte.mv(&self.position)
        } else {
            None
        };
        if excluded_move.is_none() {
            get_stack_mut(stack, 0).tt_pv = pv_node || (get_stack(stack, 0).tt_hit && tte.is_pv());
        }

        if get_stack(stack, 0).tt_pv
            && depth > Depth(12)
            && get_stack(stack, 0).ply - 1 < LowPlyHistory::MAX_LPH as i32
            && prior_capture == Piece::EMPTY
            && get_stack(stack, -1).current_move.is_normal_move()
        {
            self.low_ply_history.update(
                get_stack(stack, 0).ply - 1,
                get_stack(stack, -1).current_move.non_zero_unwrap_unchecked(),
                stat_bonus(depth - Depth(5)),
            );
        }

        self.tt_hit_average = (TT_HIT_AVERAGE_WINDOW - 1) * self.tt_hit_average / TT_HIT_AVERAGE_WINDOW
            + TT_HIT_AVERAGE_RESOLUTION * u64::from(get_stack(stack, 0).tt_hit);

        if !pv_node
            && get_stack(stack, 0).tt_hit
            && tte.depth() >= depth
            && tt_value != Value::NONE
            && if tt_value >= beta {
                tte.bound().include_lower()
            } else {
                tte.bound().include_upper()
            }
        {
            if let Some(tt_move) = tt_move {
                if tt_value >= beta {
                    if !tt_move.is_capture_or_pawn_promotion(&self.position) {
                        debug_assert!(self.position.pseudo_legal::<SearchingType>(tt_move));
                        // tt_move is guaranteed to be pseudo_legal.
                        // If tt_move isn't checked for pseudo_legal,
                        // tt_move can be promotion move for the piece that can't promotion.
                        // Then can be as follows.
                        //     tt_move.piece_moved_after_move().0 >= Piece::NUM
                        // It causes "index out of bounds" in update_continuation_histoies() in in update_quiet_stats().
                        self.update_quiet_stats(stack, tt_move, stat_bonus(depth), depth);
                    }

                    if get_stack(stack, -1).move_count <= 2
                        && !(prior_capture == Piece::EMPTY // prev is capture
                             || get_stack(stack, -1)
                                .current_move
                                .non_zero_unwrap_unchecked()
                                .is_pawn_promotion())
                    {
                        update_continuation_histories(
                            stack,
                            self.position.piece_on(prev_sq),
                            prev_sq,
                            -stat_bonus(depth + Depth::ONE_PLY),
                        );
                    }
                } else if !(tt_move.is_capture(&self.position)/*|| tt_move.is_pawn_promotion()*/) {
                    let penalty = -stat_bonus(depth);
                    self.main_history.update(us, tt_move, penalty);
                    update_continuation_histories(&mut stack[1..], tt_move.piece_moved_after_move(), tt_move.to(), penalty);
                }
            }
            return tt_value;
        }

        // Step 5
        if self.position.is_entering_king_win() {
            // The declaration win is at this node, within the horizon: flat for the engine's side, graded for the opponent.
            best_value = if us == self.engine_color {
                Value::MATE_FLAT
            } else {
                Value::mate_in(get_stack(stack, 0).ply)
            };
            if tt_move.is_none() || tt_move.non_zero_unwrap_unchecked() != Move::WIN {
                get_stack_mut(stack, 0).static_eval = best_value; // is this necessary?
                tte.save(
                    key,
                    value_to_tt(best_value, get_stack(stack, 0).ply),
                    get_stack(stack, 0).tt_pv,
                    Bound::EXACT,
                    depth,
                    Some(Move::WIN),
                    best_value,
                    unsafe { (*self.tt).generation() },
                );
            }
            return best_value;
        }

        if !root_node
            && !get_stack(stack, 0).in_check
            // A 1-ply mate lands on game ply ply() + 1, so at the boundary (ply() == limit) it is past the horizon and scores a draw, not a mate.
            && self.position.ply() < self.max_moves_to_draw()
            && let Some(mate_move) = self.position.mate_move_in_1ply()
        {
            best_value = if us == self.engine_color {
                Value::MATE_FLAT
            } else {
                Value::mate_in(get_stack(stack, 0).ply)
            };
            get_stack_mut(stack, 0).static_eval = best_value; // is this necessary?
            tte.save(
                key,
                value_to_tt(best_value, get_stack(stack, 0).ply),
                get_stack(stack, 0).tt_pv,
                Bound::EXACT,
                depth,
                Some(mate_move),
                best_value,
                unsafe { (*self.tt).generation() },
            );
            return best_value;
        }

        let pure_static_eval = if root_node {
            evaluate_at_root(&self.position, stack)
        } else {
            evaluate(&mut self.position, stack)
        };
        let improving;
        // Step 6
        if get_stack(stack, 0).in_check {
            get_stack_mut(stack, 0).static_eval = pure_static_eval;
            improving = false;
        } else {
            let mut eval;
            if get_stack(stack, 0).tt_hit {
                eval = tte.eval();
                get_stack_mut(stack, 0).static_eval = eval;
                if eval == Value::NONE {
                    eval = pure_static_eval;
                    get_stack_mut(stack, 0).static_eval = eval;
                }
                if eval == Value::NONE {
                    eval = value_draw(self.nodes.load(Ordering::Relaxed));
                }
                if tt_value != Value::NONE
                    && if tt_value > eval {
                        tte.bound().include_lower()
                    } else {
                        tte.bound().include_upper()
                    }
                {
                    eval = tt_value;
                }
            } else {
                if get_stack(stack, -1).current_move.is_some() {
                    eval = pure_static_eval;
                    get_stack_mut(stack, 0).static_eval = eval;
                } else {
                    eval = -get_stack(stack, -1).static_eval;
                    get_stack_mut(stack, 0).static_eval = eval;
                }

                if excluded_move.is_none() {
                    tte.save(
                        key,
                        Value::NONE,
                        get_stack(stack, 0).tt_pv,
                        Bound::BOUND_NONE,
                        Depth::NONE,
                        None,
                        eval,
                        unsafe { (*self.tt).generation() },
                    );
                }
            }

            if get_stack(stack, -1).current_move.is_normal_move()
                && !get_stack(stack, -1).in_check
                && prior_capture == Piece::EMPTY
            {
                let bonus = num::clamp(
                    -depth.0 * 4 * (get_stack(stack, -1).static_eval.0 + get_stack(stack, 0).static_eval.0),
                    -1000,
                    1000,
                );
                self.main_history.update(
                    us.inverse(),
                    get_stack(stack, -1).current_move.non_zero_unwrap_unchecked(),
                    bonus,
                );
            }

            improving = if get_stack(stack, -2).static_eval == Value::NONE {
                get_stack(stack, 0).static_eval > get_stack(stack, -4).static_eval
                    || get_stack(stack, -4).static_eval == Value::NONE
            } else {
                get_stack(stack, 0).static_eval > get_stack(stack, -2).static_eval
            };

            // Step 7
            if !pv_node && depth.0 < 9 && eval - futility_margin(depth) >= beta && eval < Value::KNOWN_WIN {
                return eval;
            }

            // Step 8
            if !pv_node
                && get_stack(stack, -1).current_move.is_some()
                && get_stack(stack, -1).stat_score < 23767
                && eval >= beta
                && eval >= get_stack(stack, 0).static_eval
                && get_stack(stack, 0).static_eval.0
                    >= beta.0 - 20 * depth.0 - 22 * i32::from(improving) + 168 * i32::from(get_stack(stack, 0).tt_pv) + 177
                && excluded_move.is_none()
                && (get_stack(stack, 0).ply >= self.null_move_pruning_min_ply || us != self.null_move_pruning_color)
            {
                debug_assert!(eval - beta >= Value(0));
                let r = Depth(std::cmp::min((eval.0 - beta.0) / 205, 3) + depth.0 / 3 + 4);
                get_stack_mut(stack, 0).current_move = Some(Move::NULL);
                get_stack_mut(stack, 0).continuation_history = self.continuation_history[0][0].sentinel();

                self.position.do_null_move();
                let mut null_value = -self.search::<NonPvType>(&mut stack[1..], -beta, -beta + Value(1), depth - r, !cut_node);
                self.position.undo_null_move();

                if null_value >= beta {
                    if null_value >= Value::MATE_IN_MAX_PLY {
                        null_value = beta;
                    }

                    if self.null_move_pruning_min_ply != 0 || (beta.0.abs() < Value::KNOWN_WIN.0 && depth.0 < 14) {
                        return null_value;
                    }

                    debug_assert!(self.null_move_pruning_min_ply == 0);
                    self.null_move_pruning_min_ply = get_stack(stack, 0).ply + 3 * (depth.0 - r.0) / 4;
                    self.null_move_pruning_color = us;

                    let v = self.search::<NonPvType>(stack, beta - Value(1), beta, depth - r, false);

                    self.null_move_pruning_min_ply = 0;
                    if v >= beta {
                        return null_value;
                    }
                }
            }

            let prob_cut_beta = Value(beta.0 + 209 - 44 * i32::from(improving));

            // Step 9
            if !pv_node
                && depth.0 > 4
                && beta.0.abs() < Value::MATE_IN_MAX_PLY.0
                && !(get_stack(stack, 0).tt_hit
                    && tte.depth() >= depth - Depth(3)
                    && tt_value != Value::NONE
                    && tt_value < prob_cut_beta)
            {
                let raised_beta = Value(beta.0 + 176 - 49 * i32::from(improving));
                debug_assert!(raised_beta < Value::INFINITE);
                let mut mp = MovePickerForProbCut::new(
                    &self.position,
                    tt_move,
                    raised_beta - get_stack(stack, 0).static_eval,
                    &self.capture_history,
                );
                let mut prob_cut_count = 0;
                let tt_pv = get_stack(stack, 0).tt_pv;
                get_stack_mut(stack, 0).tt_pv = false;
                while let Some(m) = mp.next_move(&self.position) {
                    if prob_cut_count >= 2 + 2 * i32::from(cut_node) {
                        break;
                    }
                    if m != excluded_move.non_zero_unwrap_unchecked() && self.position.legal(m) {
                        prob_cut_count += 1;
                        get_stack_mut(stack, 0).current_move = Some(m);
                        get_stack_mut(stack, 0).continuation_history = self.continuation_history
                            [usize::from(get_stack(stack, 0).in_check)][usize::from(prior_capture != Piece::EMPTY)]
                        .get_mut(m.piece_moved_after_move(), m.to());
                        debug_assert!(depth.0 >= 5);

                        let gives_check = self.position.gives_check(m);
                        #[cfg(feature = "nnue")]
                        {
                            // SAFETY (numa): `nnue_net_ptr` points at process-lifetime replica weights (kept alive by `nnue_network`); deref isn't tied to `&self`, so it coexists with the `&mut self.position` borrow.
                            #[cfg(feature = "numa")]
                            let net_ref = unsafe { &*self.nnue_net_ptr };
                            #[cfg(not(feature = "numa"))]
                            let net_ref = self.nnue_network.as_ref().expect("nnue network must be loaded before search");
                            do_move_with_accumulator(stack, CURRENT_STACK_INDEX, &mut self.position, m, gives_check, net_ref);
                        }
                        #[cfg(not(feature = "nnue"))]
                        self.position.do_move(m, gives_check);
                        let mut value =
                            -self.qsearch::<NonPvType>(&mut stack[1..], -prob_cut_beta, -prob_cut_beta + Value(1), Depth::ZERO);
                        if value >= prob_cut_beta {
                            value = -self.search::<NonPvType>(
                                &mut stack[1..],
                                -prob_cut_beta,
                                -prob_cut_beta + Value(1),
                                Depth(depth.0 - 4),
                                !cut_node,
                            );
                        }
                        self.position.undo_move(m);

                        if value >= prob_cut_beta {
                            if !(get_stack(stack, 0).tt_hit && tte.depth() >= depth - Depth(3) && tt_value != Value::NONE) {
                                tte.save(
                                    key,
                                    value_to_tt(value, get_stack(stack, 0).ply),
                                    tt_pv,
                                    Bound::LOWER,
                                    depth - Depth(3),
                                    Some(m),
                                    get_stack(stack, 0).static_eval,
                                    unsafe { (*self.tt).generation() },
                                );
                            }
                            return value;
                        }
                    }
                }
                get_stack_mut(stack, 0).tt_pv = tt_pv;
            }

            // Step 10
            if pv_node && depth >= Depth(6) && tt_move.is_none() {
                depth -= Depth(2);
            }

            if cut_node && depth >= Depth(9) && tt_move.is_none() {
                depth -= Depth::ONE_PLY;
            }
        }

        let tt_capture = tt_move.is_some()
            && tt_move
                .non_zero_unwrap_unchecked()
                .is_capture_or_pawn_promotion(&self.position);

        // Step 11
        let prob_cut_beta = beta + Value(409);
        if get_stack(stack, 0).in_check
            && !pv_node
            && depth >= Depth(4)
            && tt_capture
            && tte.bound().include_lower()
            && tte.depth() >= depth - Depth(3)
            && tt_value >= prob_cut_beta
            && tt_value.0.abs() <= Value::KNOWN_WIN.0
            && beta.0.abs() <= Value::KNOWN_WIN.0
        {
            return prob_cut_beta;
        }

        let cont_hists = [
            get_stack(stack, -1).continuation_history as *const PieceToHistory,
            get_stack(stack, -2).continuation_history as *const PieceToHistory,
            std::ptr::null(),
            get_stack(stack, -4).continuation_history as *const PieceToHistory,
            std::ptr::null(),
            get_stack(stack, -6).continuation_history as *const PieceToHistory,
        ];

        let counter_move = self.counter_moves.get(prev_sq, self.position.piece_on(prev_sq));

        let mut mp = MovePickerForMainSearch::new(
            &self.position,
            tt_move,
            depth,
            &self.main_history,
            &self.low_ply_history,
            &self.capture_history,
            &cont_hists,
            counter_move,
            &get_stack(stack, 0).killers,
            get_stack(stack, 0).ply,
        );

        let mut value = best_value;
        let mut move_count_pruning = false;
        let mut singular_quiet_lmr = false;
        let mut double_extension = false;
        //let likely_fail_low = pv_node
        //    && tt_move.is_some()
        //    && tte.bound().include_upper()
        //    && tte.depth() >= depth;

        // Step 12
        let mut move_count = 0;
        const CAPTURES_SEARCHED_NUM: usize = 32;
        const QUIETS_SEARCHED_NUM: usize = 64;
        let mut captures_searched = arrayvec::ArrayVec::<_, CAPTURES_SEARCHED_NUM>::new();
        let mut quiets_searched = arrayvec::ArrayVec::<_, QUIETS_SEARCHED_NUM>::new();
        while let Some(m) = mp.next_move(&self.position, move_count_pruning) {
            debug_assert!(Some(m).is_normal_move());

            if m == excluded_move.non_zero_unwrap_unchecked() {
                continue;
            }

            if root_node && !self.root_moves.iter().skip(self.pv_idx()).any(|x| x.pv[0] == m) {
                continue;
            }

            if !root_node && !self.position.legal(m) {
                continue;
            }

            move_count += 1;
            get_stack_mut(stack, 0).move_count = move_count;

            let mut extension = Depth::ZERO;
            let is_capture_or_pawn_promotion = m.is_capture_or_pawn_promotion(&self.position);
            let piece_moved_after_move = m.piece_moved_after_move();
            let gives_check = self.position.gives_check(m);

            let new_depth = depth - Depth::ONE_PLY;
            let to = m.to();

            // Step 13
            if !root_node && best_value > Value::MATED_IN_MAX_PLY {
                move_count_pruning = move_count >= futility_move_count(improving, depth.0);
                let lmr_depth = std::cmp::max(
                    new_depth - unsafe { (*self.reductions).get(improving, depth, move_count) },
                    Depth::ZERO,
                );
                if is_capture_or_pawn_promotion || gives_check {
                    if !gives_check
                        && lmr_depth < Depth::ONE_PLY
                        && self
                            .capture_history
                            .get(piece_moved_after_move, to, PieceType::new(self.position.piece_on(to)))
                            < 0
                    {
                        continue;
                    }

                    if !self.position.see_ge(m, Value(-218 * depth.0)) {
                        continue;
                    }
                } else {
                    if lmr_depth.0 < 5
                        && unsafe { (*cont_hists[0]).get(to, piece_moved_after_move) }
                            + unsafe { (*cont_hists[1]).get(to, piece_moved_after_move) }
                            + unsafe { (*cont_hists[3]).get(to, piece_moved_after_move) }
                            < -3000 * depth.0 + 3000
                    {
                        continue;
                    }
                    if !get_stack(stack, 0).in_check
                        && lmr_depth.0 < 7
                        && get_stack(stack, 0).static_eval.0 + 172 + 157 * lmr_depth.0 <= alpha.0
                        // This process is not done by Stockfish.
                        && unsafe { (*cont_hists[0]).get(to, piece_moved_after_move) }
                            + unsafe { (*cont_hists[1]).get(to, piece_moved_after_move) }
                            + unsafe { (*cont_hists[3]).get(to, piece_moved_after_move) }
                            + unsafe { (*cont_hists[5]).get(to, piece_moved_after_move) } / 3
                            < 28255
                    {
                        continue;
                    }
                    if !self
                        .position
                        .see_ge(m, Value(-21 * lmr_depth.0 * lmr_depth.0 - 21 * lmr_depth.0))
                    {
                        continue;
                    }
                }
            }

            // Step 14
            if !root_node
                && depth.0 >= 7
                && m == tt_move.non_zero_unwrap_unchecked()
                && excluded_move.is_none()
                && tt_value.0.abs() < Value::KNOWN_WIN.0
                && tte.bound().include_lower()
                && tte.depth().0 >= depth.0 - 3
            {
                let singular_beta = Value(tt_value.0 - 2 * depth.0);
                let singular_depth = Depth((depth.0 - 1) / 2);
                get_stack_mut(stack, 0).excluded_move = Some(m);
                value = self.search::<NonPvType>(stack, singular_beta - Value(1), singular_beta, singular_depth, cut_node);
                get_stack_mut(stack, 0).excluded_move = None;
                if value < singular_beta {
                    extension = Depth::ONE_PLY;
                    singular_quiet_lmr = !tt_capture;
                    if !pv_node && value < singular_beta - Value(93) && get_stack(stack, 0).double_extensions < 3 {
                        extension = Depth(2);
                        double_extension = true;
                    }
                } else if singular_beta >= beta {
                    return singular_beta;
                } else if tt_value >= beta {
                    get_stack_mut(stack, 0).excluded_move = Some(m);
                    value = self.search::<NonPvType>(stack, beta - Value(1), beta, Depth((depth.0 + 3) / 2), cut_node);
                    get_stack_mut(stack, 0).excluded_move = None;

                    if value >= beta {
                        return beta;
                    }
                }
            } else if ((pv_node || cut_node) && is_capture_or_pawn_promotion && move_count != 1)
                || (gives_check && depth > Depth(6) && get_stack(stack, 0).static_eval.0.abs() > 100)
            {
                extension = Depth::ONE_PLY;
            }

            let new_depth = new_depth + extension;

            get_stack_mut(stack, 0).double_extensions = get_stack(stack, -1).double_extensions + i32::from(extension == Depth(2));
            get_stack_mut(stack, 0).current_move = Some(m);
            get_stack_mut(stack, 0).continuation_history = self.continuation_history[usize::from(get_stack(stack, 0).in_check)]
                [usize::from(is_capture_or_pawn_promotion)]
            .get_mut(piece_moved_after_move, to);

            // Step 15
            #[cfg(feature = "nnue")]
            {
                // SAFETY (numa): `nnue_net_ptr` points at process-lifetime read-only replica weights (kept alive by `nnue_network`); the deref isn't tied to `&self`, so it coexists with the `&mut self.position` borrow.
                #[cfg(feature = "numa")]
                let net_ref = unsafe { &*self.nnue_net_ptr };
                #[cfg(not(feature = "numa"))]
                let net_ref = self.nnue_network.as_ref().expect("nnue network must be loaded before search");
                do_move_with_accumulator(stack, CURRENT_STACK_INDEX, &mut self.position, m, gives_check, net_ref);
            }
            #[cfg(not(feature = "nnue"))]
            self.position.do_move(m, gives_check);

            // Step 16
            let (do_full_depth_search, did_lmr) = if depth.0 >= 3
                && move_count > 1 + 2 * i32::from(root_node)
                && (!is_capture_or_pawn_promotion
                    || (cut_node && get_stack(stack, -1).move_count > 1)
                    || !get_stack(stack, 0).tt_pv)
                && (!pv_node || get_stack(stack, 0).ply > 1 || self.idx % 4 != 3)
            {
                let mut r = unsafe { (*self.reductions).get(improving, depth, move_count) };

                if pv_node {
                    r -= Depth::ONE_PLY;
                }

                if self.tt_hit_average > 537 * TT_HIT_AVERAGE_RESOLUTION * TT_HIT_AVERAGE_WINDOW / 1024 {
                    r -= Depth::ONE_PLY;
                }

                //if get_stack(stack, 0).tt_pv && !likely_fail_low {
                //    r -= Depth(2);
                //}

                if (root_node || !pv_node) && self.best_move_changes.load(Ordering::Relaxed) <= 2 {
                    r += Depth::ONE_PLY;
                }

                //if get_stack(stack, -1).move_count > 13 {
                //    r -= Depth::ONE_PLY;
                //}

                if singular_quiet_lmr {
                    r -= Depth(1);
                }

                if cut_node && Some(m) != get_stack(stack, 0).killers[0] {
                    r += Depth(2);
                }

                if tt_capture {
                    r += Depth::ONE_PLY;
                }

                get_stack_mut(stack, 0).stat_score = self.main_history.get(us, m)
                    + unsafe { (*cont_hists[0]).get(to, piece_moved_after_move) }
                    + unsafe { (*cont_hists[1]).get(to, piece_moved_after_move) }
                    + unsafe { (*cont_hists[3]).get(to, piece_moved_after_move) }
                    - 4923;

                r -= Depth(get_stack(stack, 0).stat_score / 14721);

                let d = num::clamp(
                    new_depth - r,
                    Depth::ONE_PLY,
                    new_depth
                        + Depth(i32::from(
                            r.0 < -1 && (move_count <= 5 || (depth.0 > 6 && pv_node)) && !double_extension,
                        )),
                );
                value = -self.search::<NonPvType>(&mut stack[1..], -(alpha + Value(1)), -alpha, d, true);
                (value > alpha && d < new_depth, true)
            } else {
                (!pv_node || move_count > 1, false)
            };

            // Step 17
            if do_full_depth_search {
                value = -self.search::<NonPvType>(&mut stack[1..], -(alpha + Value(1)), -alpha, new_depth, !cut_node);

                if did_lmr && !is_capture_or_pawn_promotion {
                    let bonus = if value > alpha {
                        stat_bonus(new_depth)
                    } else {
                        -stat_bonus(new_depth)
                    };
                    update_continuation_histories(stack, piece_moved_after_move, to, bonus);
                }
            }
            if pv_node && (move_count == 1 || (value > alpha && (root_node || value < beta))) {
                value = -self.search::<PvType>(
                    &mut stack[1..],
                    -beta,
                    -alpha,
                    std::cmp::min(max_next_depth, new_depth),
                    false,
                );
            }

            // Step 18
            self.position.undo_move(m);

            debug_assert!(-Value::INFINITE < value && value < Value::INFINITE);

            // Step 19
            if self.stop.load(Ordering::Relaxed) {
                return Value::ZERO;
            }

            if root_node {
                let rm: &mut RootMove = self.root_moves.iter_mut().find(|x| x.pv[0] == m).unwrap();
                if move_count == 1 || value > alpha {
                    rm.score = value;
                    rm.sel_depth = self.sel_depth;
                    rm.pv.truncate(1);
                    rm.extract_pv_from_tt(&mut self.position, self.tt);
                    if move_count > 1 {
                        self.best_move_changes.fetch_add(1, Ordering::Relaxed);
                    }
                } else {
                    rm.score = -Value::INFINITE;
                }
            }

            if value > best_value {
                best_value = value;
                if value > alpha {
                    best_move = Some(m);
                    if pv_node && !root_node {
                        // todo: update_pv
                    }
                    if pv_node && value < beta {
                        alpha = value;
                    } else {
                        debug_assert!(value >= beta); // fail high
                        break;
                    }
                }
            }

            if m != best_move.non_zero_unwrap_unchecked() {
                if is_capture_or_pawn_promotion {
                    let _ = captures_searched.try_push(m);
                } else if !is_capture_or_pawn_promotion {
                    let _ = quiets_searched.try_push(m);
                }
            }
        }

        // Step 20
        fn legal_moves_size(pos: &Position) -> usize {
            let mut mlist = MoveList::new();
            let current_size = 0;
            mlist.generate::<LegalType>(pos, current_size);
            mlist.size
        }
        debug_assert!(
            move_count != 0 || !get_stack(stack, 0).in_check || excluded_move.is_some() || legal_moves_size(&self.position) == 0
        );

        if move_count == 0 {
            // Mated at this node, within the horizon: graded when the engine's side is mated, flat when the opponent is.
            best_value = if excluded_move.is_some() {
                alpha
            } else if us == self.engine_color {
                Value::mated_in(get_stack(stack, 0).ply)
            } else {
                Value::MATED_FLAT
            };
        } else if let Some(best_move) = best_move {
            self.update_all_stats(
                stack,
                best_move,
                best_value,
                beta,
                prev_sq,
                &quiets_searched[..],
                &captures_searched[..],
                depth,
            );
        } else if (pv_node || depth.0 >= 3) && prior_capture == Piece::EMPTY {
            update_continuation_histories(
                stack,
                self.position.piece_on(prev_sq),
                prev_sq,
                stat_bonus(depth) * (1 + i32::from(pv_node || cut_node)),
            );
        }

        if pv_node {
            best_value = std::cmp::min(best_value, max_value);
        }

        if best_value <= alpha {
            get_stack_mut(stack, 0).tt_pv = get_stack(stack, 0).tt_pv || (get_stack(stack, -1).tt_pv && depth > Depth(3));
        } else if depth > Depth(3) {
            get_stack_mut(stack, 0).tt_pv = get_stack(stack, 0).tt_pv && get_stack(stack, 1).tt_pv;
        }

        if excluded_move.is_none() && !(root_node && self.pv_idx() != 0) {
            tte.save(
                key,
                value_to_tt(best_value, get_stack(stack, 0).ply),
                get_stack(stack, 0).tt_pv,
                if best_value >= beta {
                    Bound::LOWER
                } else if pv_node && best_move.is_some() {
                    Bound::EXACT
                } else {
                    Bound::UPPER
                },
                depth,
                best_move,
                get_stack(stack, 0).static_eval,
                unsafe { (*self.tt).generation() },
            );
        }

        debug_assert!(-Value::INFINITE < best_value && best_value < Value::INFINITE);

        best_value
    }
    fn qsearch<NT: NodeTypeTrait>(&mut self, stack: &mut [Stack], alpha: Value, beta: Value, depth: Depth) -> Value {
        let pv_node = NT::NODE_TYPE == PV;
        let mut alpha = alpha;

        let old_alpha = alpha;
        get_stack_mut(stack, 0).current_move = None;
        get_stack_mut(stack, 0).continuation_history = self.continuation_history[0][0].sentinel();
        let mut best_move: Option<Move> = None;
        get_stack_mut(stack, 0).in_check = self.position.in_check();
        let mut move_count = 0;

        // We don't have to check repetition.
        // Because qsearch use only capture-moves, promotion-moves, and evasion-moves.
        // Their moves don't reach repetition positions.
        if get_stack_mut(stack, 0).ply >= MAX_PLY {
            return Value::DRAW;
        }
        // Maximum-moves rule: past the limit is a draw, returned before the TT probe and 1-ply-mate check.
        if self.position.ply() > self.max_moves_to_draw() {
            return draw_value(self.draw_contempt(), self.position.side_to_move());
        }

        debug_assert!(0 <= get_stack(stack, 0).ply && get_stack(stack, 0).ply < MAX_PLY);

        let tt_depth = if get_stack(stack, 0).in_check || depth >= Depth::QS_CHECKS {
            Depth::QS_CHECKS
        } else {
            Depth::QS_NO_CHECKS
        };
        let key = self.position.key();
        let tte = {
            let (tte, tt_hit) = unsafe { (*self.tt).probe(key) };
            get_stack_mut(stack, 0).tt_hit = tt_hit;
            tte
        };
        let tt_value = if get_stack(stack, 0).tt_hit {
            value_from_tt(tte.value(), get_stack(stack, 0).ply)
        } else {
            Value::NONE
        };
        let tt_move = if get_stack(stack, 0).tt_hit {
            tte.mv(&self.position)
        } else {
            None
        };
        let pv_hit = get_stack(stack, 0).tt_hit && tte.is_pv();

        if !pv_node
            && get_stack(stack, 0).tt_hit
            && tte.depth() >= tt_depth
            && tt_value != Value::NONE // Only in case of TT access race
            && if tt_value >= beta {
                tte.bound().include_lower()
            } else {
                tte.bound().include_upper()
            }
        {
            return tt_value;
        }

        let mut best_value;
        let futility_base;
        if get_stack(stack, 0).in_check {
            get_stack_mut(stack, 0).static_eval = Value::NONE;
            futility_base = -Value::INFINITE;
            best_value = -Value::INFINITE;
        } else {
            // Same horizon boundary as the main-search shortcut: at ply() == limit the 1-ply mate is past the horizon, a draw.
            if self.position.ply() < self.max_moves_to_draw()
                && let Some(_mate_move) = self.position.mate_move_in_1ply()
            {
                return if self.position.side_to_move() == self.engine_color {
                    Value::MATE_FLAT
                } else {
                    Value::mate_in(get_stack(stack, 0).ply)
                };
            }
            if get_stack(stack, 0).tt_hit {
                best_value = tte.eval();
                get_stack_mut(stack, 0).static_eval = best_value;
                if best_value == Value::NONE {
                    best_value = evaluate(&mut self.position, stack);
                    get_stack_mut(stack, 0).static_eval = best_value;
                }
                if tt_value != Value::NONE
                    && if tt_value > best_value {
                        tte.bound().include_lower()
                    } else {
                        tte.bound().include_upper()
                    }
                {
                    best_value = tt_value;
                }
            } else {
                best_value = if get_stack(stack, -1).current_move.non_zero_unwrap_unchecked() != Move::NULL {
                    evaluate(&mut self.position, stack)
                } else {
                    -get_stack(stack, -1).static_eval
                };
                get_stack_mut(stack, 0).static_eval = best_value;
            }

            if best_value >= beta {
                if !get_stack(stack, 0).tt_hit {
                    tte.save(
                        key,
                        value_to_tt(best_value, get_stack(stack, 0).ply),
                        false,
                        Bound::LOWER,
                        Depth::NONE,
                        None,
                        get_stack(stack, 0).static_eval,
                        unsafe { (*self.tt).generation() },
                    );
                }
                return best_value;
            }

            if pv_node && best_value > alpha {
                alpha = best_value;
            }

            futility_base = best_value + Value(155);
        }

        let cont_hists = [
            get_stack(stack, -1).continuation_history as *const PieceToHistory,
            get_stack(stack, -2).continuation_history as *const PieceToHistory,
            std::ptr::null(),
            get_stack(stack, -4).continuation_history as *const PieceToHistory,
            std::ptr::null(),
            get_stack(stack, -6).continuation_history as *const PieceToHistory,
        ];
        let mut mp = MovePickerForQSearch::new(
            &self.main_history,
            &self.capture_history,
            &cont_hists,
            &self.position,
            get_stack(stack, -1).current_move.non_zero_unwrap_unchecked().to(),
            tt_move,
            depth,
        );

        evaluate(&mut self.position, stack); // for difference calculation
        while let Some(m) = mp.next_move(&self.position) {
            debug_assert!(m != Move::NULL);

            if !self.position.legal(m) {
                continue;
            }

            let gives_check = self.position.gives_check(m);
            let is_capture_or_pawn_promotion = m.is_capture_or_pawn_promotion(&self.position);
            move_count += 1;
            if best_value > Value::MATED_IN_MAX_PLY && !gives_check && futility_base > -Value::KNOWN_WIN {
                if move_count > 2 {
                    continue;
                }
                let futility_value = futility_base
                    + capture_piece_value(self.position.piece_on(m.to()))
                    + if m.is_promotion() {
                        promote_piece_type_value(PieceType::new(m.piece_moved_before_move()))
                    } else {
                        Value::ZERO
                    };

                if futility_value <= alpha {
                    best_value = std::cmp::max(best_value, futility_value);
                    continue;
                }

                if futility_base <= alpha && !self.position.see_ge(m, Value(1)) {
                    best_value = std::cmp::max(best_value, futility_base);
                    continue;
                }
            }

            if best_value > Value::MATED_IN_MAX_PLY && !self.position.see_ge(m, Value::ZERO) {
                continue;
            }

            let piece_moved_after_move = m.piece_moved_after_move();
            let to = m.to();
            get_stack_mut(stack, 0).current_move = Some(m);
            get_stack_mut(stack, 0).continuation_history = self.continuation_history[usize::from(get_stack(stack, 0).in_check)]
                [usize::from(is_capture_or_pawn_promotion)]
            .get_mut(piece_moved_after_move, to);

            if !is_capture_or_pawn_promotion
                && best_value > Value::MATED_IN_MAX_PLY
                && unsafe { (*cont_hists[0]).get(to, piece_moved_after_move) } < i32::from(COUNTER_MOVE_PRUNE_THRESHOLD)
                && unsafe { (*cont_hists[1]).get(to, piece_moved_after_move) } < i32::from(COUNTER_MOVE_PRUNE_THRESHOLD)
            {
                continue;
            }

            #[cfg(feature = "nnue")]
            {
                // SAFETY (numa): `nnue_net_ptr` points at process-lifetime read-only replica weights (kept alive by `nnue_network`); the deref isn't tied to `&self`, so it coexists with the `&mut self.position` borrow.
                #[cfg(feature = "numa")]
                let net_ref = unsafe { &*self.nnue_net_ptr };
                #[cfg(not(feature = "numa"))]
                let net_ref = self.nnue_network.as_ref().expect("nnue network must be loaded before search");
                do_move_with_accumulator(stack, CURRENT_STACK_INDEX, &mut self.position, m, gives_check, net_ref);
            }
            #[cfg(not(feature = "nnue"))]
            self.position.do_move(m, gives_check);
            let value = -self.qsearch::<NT>(&mut stack[1..], -beta, -alpha, depth - Depth::ONE_PLY);
            self.position.undo_move(m);

            debug_assert!(-Value::INFINITE < value && value < Value::INFINITE);

            if value > best_value {
                best_value = value;

                if value > alpha {
                    best_move = Some(m);
                    if pv_node {
                        // todo: update_pv
                    }

                    if pv_node && value < beta {
                        alpha = value;
                    } else {
                        break; // fail high
                    }
                }
            }
        }

        if get_stack(stack, 0).in_check && best_value == -Value::INFINITE {
            debug_assert_eq!(
                {
                    let mut mlist = MoveList::new();
                    mlist.generate::<LegalType>(&self.position, 0);
                    mlist.size
                },
                0
            );
            // Mated at this node, within the horizon: graded for the engine's side, flat when the opponent is mated.
            return if self.position.side_to_move() == self.engine_color {
                Value::mated_in(get_stack(stack, 0).ply)
            } else {
                Value::MATED_FLAT
            };
        }

        tte.save(
            key,
            value_to_tt(best_value, get_stack(stack, 0).ply),
            pv_hit,
            if best_value >= beta {
                Bound::LOWER
            } else if pv_node && best_value > old_alpha {
                Bound::EXACT
            } else {
                Bound::UPPER
            },
            tt_depth,
            best_move,
            get_stack(stack, 0).static_eval,
            unsafe { (*self.tt).generation() },
        );

        debug_assert!(-Value::INFINITE < best_value && best_value < Value::INFINITE);

        best_value
    }
    fn nodes_searched(&self) -> i64 {
        debug_assert!(self.is_main());
        self.nodess.iter().fold(0, |sum, nodes| sum + nodes.load(Ordering::Relaxed))
    }
    fn check_time(&mut self) {
        self.calls_count -= 1;
        if self.calls_count > 0 {
            return;
        }
        self.calls_count = match self.limits.nodes {
            Some(nodes) => std::cmp::min(1024, nodes / 1024) as i32,
            None => 1024,
        };

        if self.ponder.load(Ordering::Relaxed) {
            return;
        }

        let elapsed = self.limits.start_time.unwrap().elapsed();

        if (self.limits.use_time_management()
            && (elapsed.as_millis() as i64 > self.timeman.lock().unwrap().maximum_millis() - 10
                || self.stop_on_ponderhit.load(Ordering::Relaxed)))
            || (self.limits.movetime.is_some() && elapsed >= self.limits.movetime.unwrap())
            || (self.limits.nodes.is_some() && self.nodes_searched() >= self.limits.nodes.unwrap() as i64)
        {
            self.stop.store(true, Ordering::Relaxed);
        }
    }
    fn update_all_stats(
        &mut self,
        stack: &mut [Stack],
        best_move: Move,
        best_value: Value,
        beta: Value,
        prev_sq: Square,
        quiets_searched: &[Move],
        captures_searched: &[Move],
        depth: Depth,
    ) {
        let us = self.position.side_to_move();
        let moved_piece = best_move.piece_moved_after_move();
        let captured = PieceType::new(self.position.piece_on(best_move.to()));
        let bonus1 = stat_bonus(depth + Depth::ONE_PLY);
        let bonus2 = if best_value > beta + piece_type_value(PieceType::PAWN) {
            bonus1
        } else {
            std::cmp::min(bonus1, stat_bonus(depth))
        };
        if !best_move.is_capture_or_pawn_promotion(&self.position) {
            self.update_quiet_stats(stack, best_move, bonus2, depth);
            for &quiet_move in quiets_searched {
                self.main_history.update(us, quiet_move, -bonus2);
                update_continuation_histories(&mut stack[1..], quiet_move.piece_moved_after_move(), quiet_move.to(), -bonus2);
            }
        } else {
            let capture_history = &mut self.capture_history;
            capture_history.update(moved_piece, best_move.to(), captured, bonus1);
        }

        if (get_stack(stack, -1).move_count == 1 + i32::from(get_stack(stack, -1).tt_hit)
            || get_stack(stack, -1).current_move == get_stack(stack, -1).killers[0])
            && self.position.captured_piece() == Piece::EMPTY
        {
            update_continuation_histories(stack, self.position.piece_on(prev_sq), prev_sq, -bonus1);
        }

        for &capture_move in captures_searched {
            let moved_piece = capture_move.piece_moved_after_move();
            let captured = PieceType::new(self.position.piece_on(capture_move.to()));
            self.capture_history.update(moved_piece, capture_move.to(), captured, -bonus1);
        }
    }
    fn update_quiet_stats(&mut self, stack: &mut [Stack], m: Move, bonus: i32, depth: Depth) {
        if get_stack(stack, 0).killers[0].non_zero_unwrap_unchecked() != m {
            let ss = get_stack_mut(stack, 0);
            ss.killers[1] = ss.killers[0];
            ss.killers[0] = Some(m);
        }
        let us = self.position.side_to_move();
        self.main_history.update(us, m, bonus);
        update_continuation_histories(&mut stack[1..], m.piece_moved_after_move(), m.to(), bonus);

        if PieceType::new(m.piece_moved_after_move()) == PieceType::PAWN {
            self.main_history.update(us, m, -bonus);
        }

        let prev_move = get_stack(stack, -1).current_move;
        if prev_move.is_normal_move() {
            let prev_sq = prev_move.non_zero_unwrap_unchecked().to();
            self.counter_moves.set(prev_sq, self.position.piece_on(prev_sq), m);
        }
        if depth.0 > 11 && get_stack(stack, 0).ply < LowPlyHistory::MAX_LPH as i32 {
            self.low_ply_history
                .update(get_stack(stack, 0).ply, m, stat_bonus(depth - Depth(7)));
        }
    }
    // Only off-`tournament` and for tests: tournament emits no search info.
    #[cfg(any(not(feature = "tournament"), test))]
    fn pv_info_to_usi_string(
        &self,
        nodes_searched: i64,
        multi_pv: usize,
        depth: Depth,
        alpha: Value,
        beta: Value,
        reverse: bool, // for Shogidokoro Graph
    ) -> String {
        let elapsed_millis = self.limits.start_time.unwrap().elapsed().as_millis() as i64 + 1; // "+ 1": avoid dividing by 0
        let info_with_multi_pv_index = |i: usize, rm: &RootMove| -> Option<String> {
            let updated = rm.score != -Value::INFINITE;
            if depth == Depth::ONE_PLY && !updated && i > 0 {
                return None;
            }
            let (d, mut v) = if updated {
                (depth, rm.score)
            } else {
                (std::cmp::max(Depth::ONE_PLY, depth - Depth::ONE_PLY), rm.previous_score)
            };
            if v == -Value::INFINITE {
                v = Value::ZERO;
            }
            let bound = if v >= beta {
                "lowerbound "
            } else if v <= alpha {
                "upperbound "
            } else {
                ""
            };
            let pv = rm.pv.iter().map(|m| m.to_usi_string()).collect::<Vec<_>>().join(" ");
            let nps = nodes_searched * 1000 / elapsed_millis;
            // The tournament build emits a single PV and drops the `multipv` token entirely.
            #[cfg(not(feature = "tournament"))]
            let line = format!(
                "info depth {depth} seldepth {seldepth} multipv {multipv} score {score} {bound}nodes {nodes} nps {nps} time {time} pv {pv}",
                depth = d.0,
                seldepth = rm.sel_depth,
                multipv = i + 1,
                score = v.to_usi(),
                nodes = nodes_searched,
                time = elapsed_millis,
            );
            #[cfg(feature = "tournament")]
            let line = format!(
                "info depth {depth} seldepth {seldepth} score {score} {bound}nodes {nodes} nps {nps} time {time} pv {pv}",
                depth = d.0,
                seldepth = rm.sel_depth,
                score = v.to_usi(),
                nodes = nodes_searched,
                time = elapsed_millis,
            );
            Some(line)
        };
        let mut lines = self
            .root_moves
            .iter()
            .take(multi_pv)
            .enumerate()
            .flat_map(|(i, rm)| info_with_multi_pv_index(i, rm))
            .collect::<Vec<_>>();
        if reverse {
            lines.reverse();
        }
        lines.join("\n")
    }
}

/// The benchmark build's one search info line (cumulative nodes/nps/time), printed once before `bestmove` for NPS tooling.
#[cfg(all(feature = "tournament", feature = "emit-nps"))]
fn bench_nps_info_string(nodes_searched: i64, elapsed_millis: i64) -> String {
    format!(
        "info nodes {nodes_searched} nps {nps} time {elapsed_millis}",
        nps = nodes_searched * 1000 / elapsed_millis,
    )
}

impl ThreadPool {
    pub fn new() -> ThreadPool {
        ThreadPool {
            thread_pool_base: Arc::new(Mutex::new(ThreadPoolBase { threads: vec![] })),
            nodess: vec![],
            book: None,
            timeman: Arc::new(Mutex::new(TimeManagement::new())),
            best_previous_score: Arc::new(Mutex::new(Value::INFINITE)),
            iter_values: Arc::new(Mutex::new([Value::ZERO; 4])),
            best_move_changess: vec![],
            stop_on_ponderhit: Arc::new(AtomicBool::new(false)),
            ponder: Arc::new(AtomicBool::new(false)),
            stop: Arc::new(AtomicBool::new(false)),
            increase_depth: Arc::new(AtomicBool::new(true)),
            hide_all_output: Arc::new(AtomicBool::new(false)),
            limits: LimitsType::new(),
            last_best_root_move: Arc::new(Mutex::new(None)),
            handle: None,
        }
    }
    pub fn clear(&mut self) {
        for th in self.thread_pool_base.lock().unwrap().threads.iter() {
            th.lock().unwrap().clear();
        }
        *self.last_best_root_move.lock().unwrap() = None;

        let thread_pool_base = self.thread_pool_base.lock().unwrap();
        let mut main_thread = thread_pool_base.threads[0].lock().unwrap();
        main_thread.calls_count = 0;
        *main_thread.best_previous_score.lock().unwrap() = Value::INFINITE;
        main_thread.previous_time_reduction = 1.0;
    }
    pub fn set(&mut self, requested: usize, tt: &mut TranspositionTable, reductions: &mut Reductions) {
        if let Some(handle) = self.handle.take() {
            handle.join().unwrap();
            self.thread_pool_base.lock().unwrap().threads.clear();
            self.nodess.clear();
        }
        self.thread_pool_base.lock().unwrap().threads.clear();
        self.nodess = (0..requested).map(|_| Arc::new(AtomicI64::new(0))).collect();
        self.best_move_changess = (0..requested).map(|_| Arc::new(AtomicU64::new(0))).collect();
        *reductions = Reductions::new();
        self.thread_pool_base.lock().unwrap().threads = (0..requested)
            .map(|i| {
                Arc::new(Mutex::new(Thread {
                    idx: i,
                    #[cfg(not(feature = "tournament"))]
                    pv_idx: 0,
                    tt_hit_average: 0,
                    sel_depth: 0,
                    null_move_pruning_min_ply: 0,
                    null_move_pruning_color: Color::BLACK,
                    // Re-populated per search in start_thinking (ponder-aware).
                    engine_color: Color::BLACK,
                    position: Position::new(),
                    root_moves: RootMoves::new(),
                    root_depth: Depth::ZERO,
                    completed_depth: Depth::ZERO,
                    counter_moves: CounterMoveHistory::new(),
                    main_history: ButterflyHistory::new(),
                    low_ply_history: LowPlyHistory::new(),
                    capture_history: CapturePieceToHistory::new(),
                    continuation_history: [
                        [ContinuationHistory::new(), ContinuationHistory::new()],
                        [ContinuationHistory::new(), ContinuationHistory::new()],
                    ],
                    limits: self.limits.clone(),
                    // Non-tournament only: gate the initializer to match the field (tournament reads the const accessor).
                    #[cfg(not(feature = "tournament"))]
                    max_moves_to_draw: i32::MAX,
                    // Re-snapshotted per search in iterative_deepening_loop; 0 = no contempt.
                    #[cfg(not(feature = "tournament"))]
                    draw_contempt: 0,
                    tt,
                    timeman: self.timeman.clone(),
                    reductions,
                    usi_options: UsiOptions::new(),
                    best_move_changes: self.best_move_changess[i].clone(),
                    best_move_changess: self.best_move_changess.clone(),
                    nodes: self.nodess[i].clone(),
                    #[cfg(feature = "nnue")]
                    nnue_network: None,
                    // Set per search in iterative_deepening_loop; null until then.
                    #[cfg(all(feature = "nnue", feature = "numa"))]
                    nnue_net_ptr: std::ptr::null(),
                    best_previous_score: self.best_previous_score.clone(),
                    iter_values: self.iter_values.clone(),
                    increase_depth: self.increase_depth.clone(),
                    previous_time_reduction: 1.0,
                    calls_count: 0,
                    stop_on_ponderhit: self.stop_on_ponderhit.clone(),
                    ponder: self.ponder.clone(),
                    stop: self.stop.clone(),
                    #[cfg(not(feature = "tournament"))]
                    hide_all_output: self.hide_all_output.clone(),
                    nodess: vec![],
                }))
            })
            .collect();
        // NUMA: migrate each Thread struct onto its worker's node. Its ~51 MB of history tables are already faulted on this USI thread and a later worker-side policy cannot move them, so bind+migrate here using the same `assignment_for_idx` mapping the pin uses. Best-effort.
        #[cfg(feature = "numa")]
        for (i, th) in self.thread_pool_base.lock().unwrap().threads.iter().enumerate() {
            let node = crate::numa::assignment_for_idx(i).node;
            let addr = Arc::as_ptr(th) as *mut u8;
            let len = std::mem::size_of::<Mutex<Thread>>();
            // SAFETY: `addr`/`len` describe the live `Mutex<Thread>` payload of an Arc we hold.
            unsafe { crate::numa::mbind_region(addr, len, node, false) };
        }
        // Main thread has other thread's nodes.
        self.thread_pool_base.lock().unwrap().threads[0]
            .lock()
            .unwrap()
            .nodess
            .clone_from(&self.nodess);
    }
    pub fn start_thinking(
        &mut self,
        pos: &Position,
        tt: &mut TranspositionTable,
        limits: LimitsType,
        usi_options: &UsiOptions,
        ponder_mode: bool,
        hide_all_output: bool,
    ) {
        let mut limits = limits;
        if let Some(perft) = limits.perft {
            Perft::new(pos).go(perft);
            return;
        } else if let Some(mate) = limits.mate {
            #[cfg(feature = "mate")]
            {
                Mate::new(pos).go(mate);
            }
            // Without the `mate` feature, answer `go mate` with `checkmate notimplemented` per the USI spec.
            #[cfg(not(feature = "mate"))]
            {
                let _ = mate;
                println!("checkmate notimplemented");
            }
            return;
        }
        self.wait_for_search_finished();
        self.stop.store(false, Ordering::Relaxed);
        self.stop_on_ponderhit.store(false, Ordering::Relaxed);
        self.ponder.store(ponder_mode, Ordering::Relaxed);
        // The engine's own game side, inverted while pondering; mate flattening (see Thread::engine_color) is relative to this.
        let engine_color = if ponder_mode {
            pos.side_to_move().inverse()
        } else {
            pos.side_to_move()
        };
        self.hide_all_output.store(hide_all_output, Ordering::Relaxed);
        self.timeman
            .lock()
            .unwrap()
            .init(usi_options, &mut limits, pos.side_to_move(), pos.ply());
        tt.new_search();
        self.limits = limits.clone();
        let root_moves = {
            let mut mlist = MoveList::new();
            mlist.generate::<LegalType>(pos, 0);
            let mut root_moves = RootMoves::new();
            let book_move = if usi_options.get_bool(UsiOptions::BOOK_ENABLE) {
                match &self.book {
                    Some(book) => {
                        let book_options = BookOptions::from_usi_options(usi_options);
                        book.probe(pos, &book_options, &mut rand::thread_rng())
                    }
                    None => None,
                }
            } else {
                None
            };
            match book_move {
                Some(book_move) => {
                    root_moves.push(RootMove::new(book_move));
                }
                None => {
                    for m in mlist.slice(0) {
                        root_moves.push(RootMove::new(m.mv));
                    }
                }
            }
            root_moves
        };
        let dummy_nodes = Arc::new(AtomicI64::new(0)); // This isn't used.
        let pos = Position::new_from_position(pos, dummy_nodes);
        let nodess_cloned = self.nodess.clone();
        let timeman_cloned = self.timeman.clone();
        let previous_score_cloned = self.best_previous_score.clone();
        let thread_pool_base_cloned = self.thread_pool_base.clone();
        let stop_cloned = self.stop.clone();
        let ponder_cloned = self.ponder.clone();
        let hide_all_output_cloned = self.hide_all_output.clone();
        let usi_options_cloned = usi_options.clone();
        let last_best_root_move_cloned = self.last_best_root_move.clone();
        self.handle = Some(
            std::thread::Builder::new()
                .stack_size(crate::stack_size::STACK_SIZE)
                .spawn(move || {
                    if root_moves.is_empty() || pos.is_entering_king_win() {
                        while !stop_cloned.load(Ordering::Relaxed)
                            && (ponder_cloned.load(Ordering::Relaxed) || limits.infinite.is_some())
                        {
                            std::thread::sleep(std::time::Duration::from_millis(1));
                        }
                        let m = if root_moves.is_empty() {
                            *last_best_root_move_cloned.lock().unwrap() = Some(RootMove::new(Move::RESIGN));
                            "resign"
                        } else {
                            *last_best_root_move_cloned.lock().unwrap() = Some(RootMove::new(Move::WIN));
                            "win"
                        };
                        if !hide_all_output_cloned.load(Ordering::Relaxed) {
                            println!("bestmove {}", m);
                        }
                        return;
                    }
                    let mut v = vec![];
                    for (i, thread) in thread_pool_base_cloned
                        .lock()
                        .unwrap()
                        .threads
                        .iter_mut()
                        .enumerate()
                        // i == 0 => not using a worker thread.
                        .rev()
                    {
                        let nodes_cloned = nodess_cloned[i].clone();
                        let pos = Position::new_from_position(&pos, nodes_cloned.clone());
                        nodes_cloned.store(0, Ordering::Relaxed);
                        let root_moves_cloned = root_moves.clone();
                        let thread_cloned = thread.clone();
                        let limits_cloned = limits.clone();
                        let usi_options_cloned = usi_options_cloned.clone();
                        let timeman_cloned = timeman_cloned.clone();
                        let worker = move || {
                            let mut th = thread_cloned.lock().unwrap();
                            // NUMA: pin this worker to its compact-by-node core and select that node's NNUE replica (feature off = no pinning, shared global). Must run on the worker's own OS thread.
                            #[cfg(feature = "numa")]
                            {
                                let assignment = crate::numa::assignment_for_idx(th.idx);
                                crate::numa::pin_current_thread(assignment.cpu);
                                // Prefer the pinned core's node for everything this worker
                                // first-touches from here on (its OS thread stack, per-search
                                // allocations): pinning alone leaves the inherited task policy in
                                // place, which under an interleaving launcher would scatter this
                                // thread-private memory across all nodes. MPOL_PREFERRED, not BIND,
                                // so allocation falls back to other nodes when this one is full.
                                // For idx 0 this closure runs inline on the search-master thread
                                // and the policy is deliberately left in place afterwards: past the
                                // search that thread only joins helpers and prints bestmove, then
                                // exits. The USI thread never changes policy, so the TT (allocated
                                // and resized there) keeps following the process default.
                                crate::numa::prefer_node_for_current_thread(assignment.node);
                                #[cfg(feature = "nnue")]
                                crate::evaluate::nnue::set_local_replica_for_node(assignment.node);
                            }
                            th.best_move_changes.store(0, Ordering::Relaxed);
                            th.engine_color = engine_color;
                            th.limits = limits_cloned;
                            th.nodes = nodes_cloned;
                            th.root_depth = Depth::ZERO;
                            th.root_moves = root_moves_cloned;
                            th.position = pos;
                            th.usi_options = usi_options_cloned;
                            th.timeman = timeman_cloned;
                            th.iterative_deepening_loop();
                        };
                        if i == 0 {
                            worker(); // The main thread doesn't use std::thread::spawn().
                        } else {
                            v.push(
                                std::thread::Builder::new()
                                    .stack_size(crate::stack_size::STACK_SIZE)
                                    .spawn(worker)
                                    .unwrap(),
                            );
                        }
                    }
                    while !stop_cloned.load(Ordering::Relaxed)
                        && (ponder_cloned.load(Ordering::Relaxed) || limits.infinite.is_some())
                    {
                        // nop
                    }
                    // main thread finished.
                    // stop the other threads.
                    stop_cloned.store(true, Ordering::Relaxed);
                    for handle in v {
                        handle.join().unwrap();
                    }

                    // Single PV under `tournament` (MultiPV removed); otherwise the clamped option.
                    #[cfg(not(feature = "tournament"))]
                    let multi_pv = std::cmp::min(usi_options_cloned.get_i64(UsiOptions::MULTI_PV) as usize, root_moves.len());
                    #[cfg(feature = "tournament")]
                    let multi_pv = 1usize;
                    let best_thread = if multi_pv == 1 && limits.depth.is_none() && !root_moves.is_empty() {
                        let mut votes = std::collections::BTreeMap::new();
                        let min_score: Value = thread_pool_base_cloned
                            .lock()
                            .unwrap()
                            .threads
                            .iter()
                            .map(|x| x.lock().unwrap().root_moves[0].score)
                            .min()
                            .unwrap();

                        for th in thread_pool_base_cloned.lock().unwrap().threads.iter() {
                            let th = th.lock().unwrap();
                            *votes.entry(th.root_moves[0].pv[0].0.get()).or_insert(0) +=
                                i64::from((th.root_moves[0].score.0 - min_score.0 + 14) * th.completed_depth.0);
                        }

                        thread_pool_base_cloned
                            .lock()
                            .unwrap()
                            .threads
                            .iter()
                            // get first "max" score.
                            .min_by(|x, y| {
                                let x_score = x.lock().unwrap().root_moves[0].score;
                                let y_score = y.lock().unwrap().root_moves[0].score;
                                if x_score >= Value::MATE_IN_MAX_PLY || y_score >= Value::MATE_IN_MAX_PLY {
                                    y_score.cmp(&x_score)
                                } else {
                                    let x_vote_score = *votes.get(&x.lock().unwrap().root_moves[0].pv[0].0.get()).unwrap();
                                    let y_vote_score = *votes.get(&y.lock().unwrap().root_moves[0].pv[0].0.get()).unwrap();
                                    y_vote_score.cmp(&x_vote_score)
                                }
                            })
                            .unwrap()
                            .clone()
                    } else {
                        thread_pool_base_cloned.lock().unwrap().threads[0].clone()
                    };

                    *previous_score_cloned.lock().unwrap() = best_thread.lock().unwrap().root_moves[0].score;

                    // Snapshot before the `best_thread` lock: `best_thread` is usually `threads[0]` and `Mutex` isn't reentrant, so locking it inside would self-deadlock.
                    #[cfg(any(not(feature = "tournament"), feature = "emit-nps"))]
                    let nodes_searched = thread_pool_base_cloned.lock().unwrap().threads[0]
                        .lock()
                        .unwrap()
                        .nodes_searched();
                    if let Ok(best_thread) = best_thread.lock()
                        && !hide_all_output_cloned.load(Ordering::Relaxed)
                    {
                        // Always send again PV info; tournament compiles this out.
                        #[cfg(not(feature = "tournament"))]
                        println!(
                            "{}",
                            best_thread.pv_info_to_usi_string(
                                nodes_searched,
                                multi_pv,
                                best_thread.completed_depth,
                                -Value::INFINITE,
                                Value::INFINITE,
                                true,
                            )
                        );
                        // The benchmark build's single emission point, just before `bestmove`; only with `emit-nps` on top of `tournament`.
                        #[cfg(all(feature = "tournament", feature = "emit-nps"))]
                        {
                            // "+ 1": avoid dividing by 0, like pv_info_to_usi_string.
                            let elapsed_millis = limits.start_time.unwrap().elapsed().as_millis() as i64 + 1;
                            println!("{}", bench_nps_info_string(nodes_searched, elapsed_millis));
                        }
                        let mut s = format!("bestmove {}", best_thread.root_moves[0].pv[0].to_usi_string(),);
                        if usi_ponder(&usi_options_cloned) && best_thread.root_moves[0].pv.len() >= 2 {
                            s += &format!(" ponder {}", best_thread.root_moves[0].pv[1].to_usi_string());
                        }
                        println!("{}", s);
                    }
                    *last_best_root_move_cloned.lock().unwrap() = Some(best_thread.lock().unwrap().root_moves[0].clone());
                })
                .unwrap(),
        );
    }
    pub fn wait_for_search_finished(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.join().unwrap();
        }
    }
    #[allow(dead_code)]
    fn nodes_searched(&self) -> i64 {
        self.nodess.iter().fold(0, |sum, nodes| sum + nodes.load(Ordering::Relaxed))
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.wait_for_search_finished();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_start_thinking() {
        std::thread::Builder::new()
            .stack_size(crate::stack_size::STACK_SIZE)
            .spawn(|| {
                let mut thread_pool = ThreadPool::new();
                let mut tt = TranspositionTable::new();
                tt.resize(16, &mut thread_pool);
                // No evaluation funciton binaries.
                // We want to do "cargo test" without evaluation function binaries.
                // Then we do nothing and pass this test.
                // todo: Is there a more better way?
            })
            .unwrap()
            .join()
            .unwrap();
    }

    // Search-behavior tests need an eval without binaries (material) and the runtime horizon
    // (not tournament, where the horizon is a compile-time const).
    #[cfg(all(feature = "material", not(feature = "tournament")))]
    mod max_moves_to_draw_tests {
        use super::super::*;

        /// Deterministic single-thread fixed-depth search on a big-stack thread; returns best root move and node count.
        fn run_search(sfen: &str, max_moves_to_draw: &str, depth: u32) -> (RootMove, i64) {
            run_search_with_contempt(sfen, max_moves_to_draw, "0", depth)
        }

        fn run_search_with_contempt(sfen: &str, max_moves_to_draw: &str, draw_contempt: &str, depth: u32) -> (RootMove, i64) {
            let sfen = sfen.to_string();
            let max_moves_to_draw = max_moves_to_draw.to_string();
            let draw_contempt = draw_contempt.to_string();
            std::thread::Builder::new()
                .stack_size(crate::stack_size::STACK_SIZE)
                .spawn(move || run_search_impl(&sfen, &max_moves_to_draw, &draw_contempt, depth))
                .unwrap()
                .join()
                .unwrap()
        }

        fn run_search_impl(sfen: &str, max_moves_to_draw: &str, draw_contempt: &str, depth: u32) -> (RootMove, i64) {
            let mut reductions = Reductions::new();
            let mut thread_pool = ThreadPool::new();
            let mut tt = TranspositionTable::new();
            thread_pool.set(1, &mut tt, &mut reductions);
            tt.resize(16, &mut thread_pool);
            let mut usi_options = UsiOptions::new();
            let mut is_ready = true;
            usi_options.set(
                UsiOptions::MAX_MOVES_TO_DRAW,
                max_moves_to_draw,
                &mut thread_pool,
                &mut tt,
                &mut reductions,
                &mut is_ready,
            );
            usi_options.set(
                UsiOptions::DRAW_CONTEMPT,
                draw_contempt,
                &mut thread_pool,
                &mut tt,
                &mut reductions,
                &mut is_ready,
            );
            let pos = Position::new_from_sfen(sfen).unwrap();
            let mut limits = LimitsType::new();
            limits.depth = Some(depth);
            limits.start_time = Some(std::time::Instant::now());
            thread_pool.start_thinking(&pos, &mut tt, limits, &usi_options, false, true);
            thread_pool.wait_for_search_finished();
            let best = thread_pool
                .last_best_root_move
                .lock()
                .unwrap()
                .clone()
                .expect("search must produce a best root move");
            (best, thread_pool.nodes_searched())
        }

        // Black to move, gold in hand: G*5b is mate.
        const MATE_IN_ONE_SFEN_BODY: &str = "4k4/9/4P4/9/9/9/9/9/4K4 b G";

        #[test]
        fn win_before_horizon_is_preferred() {
            // Mate inside the horizon (ply 15, limit 16): must be scored as mate, not draw.
            let sfen = format!("{} 15", MATE_IN_ONE_SFEN_BODY);
            let (best, _) = run_search(&sfen, "16", 6);
            assert_eq!(
                best.score,
                Value::MATE_FLAT,
                "a mate inside the horizon for the engine's own side scores the flat win; got {}",
                best.score.0
            );
            let pos = Position::new_from_sfen(&sfen).unwrap();
            let mate_move = Move::new_from_usi_str("G*5b", &pos).unwrap();
            assert_eq!(best.pv[0], mate_move, "the mating move must head the pv");
        }

        #[test]
        fn mate_beyond_horizon_scores_as_draw() {
            // Mate lands beyond the horizon (limit 14): must score as a draw, not a win.
            let sfen = format!("{} 15", MATE_IN_ONE_SFEN_BODY);
            let (best, _) = run_search(&sfen, "14", 6);
            assert_eq!(
                best.score,
                Value::DRAW,
                "a mate that lands beyond the horizon is a draw; got {}",
                best.score.0
            );

            // Root already past the limit (game ply 17 > 16): every move scores as a draw.
            let sfen = format!("{} 17", MATE_IN_ONE_SFEN_BODY);
            let (best, _) = run_search(&sfen, "16", 6);
            assert_eq!(
                best.score,
                Value::DRAW,
                "past the limit everything is a draw; got {}",
                best.score.0
            );

            // Non-zero Draw_Contempt: Black to move at the root scores the horizon draw as `-contempt`.
            const C: i32 = 300;
            let expected = draw_value(C, Color::BLACK);
            assert_eq!(expected, Value(-C), "Black to move: a draw is bad");
            let sfen = format!("{} 15", MATE_IN_ONE_SFEN_BODY);
            let (best, _) = run_search_with_contempt(&sfen, "14", "300", 6);
            assert_eq!(
                best.score, expected,
                "a beyond-horizon draw must carry Black's draw contempt; got {}",
                best.score.0
            );
            let sfen = format!("{} 17", MATE_IN_ONE_SFEN_BODY);
            let (best, _) = run_search_with_contempt(&sfen, "16", "300", 6);
            assert_eq!(
                best.score, expected,
                "past the limit every move is a draw at Black's contempt; got {}",
                best.score.0
            );
        }

        // White to move, mated by R*1b as soon as black gets to move.
        const LOSING_SIDE_SFEN_BODY: &str = "6G1k/9/p7G/9/9/9/9/9/4K4 w R";

        #[test]
        fn losing_side_steers_toward_the_horizon() {
            // White's move is move 16 = the limit, so black's mate lands beyond the horizon: a draw.
            let sfen = format!("{} 16", LOSING_SIDE_SFEN_BODY);
            let (best, _) = run_search(&sfen, "16", 6);
            assert_eq!(
                best.score,
                Value::DRAW,
                "the losing side reaches the horizon and saves the draw; got {}",
                best.score.0
            );

            // Sanity check of the same position without a limit: white is simply mated.
            let (best, _) = run_search(&sfen, "0", 6);
            assert!(
                best.score <= Value::MATED_IN_MAX_PLY,
                "without a horizon the losing side is mated; got {}",
                best.score.0
            );
        }

        // A mate for the engine's own side scores the flat Value::MATE_FLAT (no ply gradient); a mate against it keeps the graded -MATE + ply.

        #[test]
        fn engine_mate_scores_flat_and_bestmove_mates() {
            let sfen = format!("{} 1", MATE_IN_ONE_SFEN_BODY);
            let (best, _) = run_search(&sfen, "0", 6);
            assert_eq!(
                best.score,
                Value::MATE_FLAT,
                "the engine's own forced mate must score the flat win value; got {}",
                best.score.0
            );
            let pos = Position::new_from_sfen(&sfen).unwrap();
            let mate_move = Move::new_from_usi_str("G*5b", &pos).unwrap();
            assert_eq!(best.pv[0], mate_move, "bestmove must still be a mating move");
        }

        // Once the flat win is proven the iterative-deepening loop stops, so a higher depth limit must not change the node count.
        #[test]
        fn proven_flat_win_terminates_iterative_deepening() {
            let sfen = format!("{} 1", MATE_IN_ONE_SFEN_BODY);
            let (best6, nodes6) = run_search(&sfen, "0", 6);
            let (best20, nodes20) = run_search(&sfen, "0", 20);
            assert_eq!(best6.score, Value::MATE_FLAT);
            assert_eq!(best20.score, Value::MATE_FLAT);
            assert_eq!(
                nodes6, nodes20,
                "after the mate is proven the search must stop, so a higher depth limit adds no nodes"
            );
        }

        // The losing engine keeps the graded score and must prefer the longer resistance (a spite check) over the immediate loss.
        #[test]
        fn losing_side_keeps_gradient_and_prefers_longer_resistance() {
            // Immediate loss: the only move is the pawn push, then R*1b mates.
            let immediate = format!("{} 1", LOSING_SIDE_SFEN_BODY);
            let (best_immediate, _) = run_search(&immediate, "0", 6);
            assert!(
                best_immediate.score <= Value::MATED_IN_MAX_PLY && Value::MATED_FLAT < best_immediate.score,
                "a mate against the engine must stay in the graded range, not flat; got {}",
                best_immediate.score.0
            );

            // Same position plus a knight in hand: the spite check N*4g delays the mate by two plies.
            let delayed = "6G1k/9/p7G/9/9/9/9/9/4K4 w Rn 1";
            let (best_delayed, _) = run_search(delayed, "0", 6);
            assert!(
                best_delayed.score <= Value::MATED_IN_MAX_PLY && Value::MATED_FLAT < best_delayed.score,
                "the delayed loss must also stay in the graded range; got {}",
                best_delayed.score.0
            );
            assert!(
                best_delayed.score > best_immediate.score,
                "the graded score must reward the longer resistance ({} vs {})",
                best_delayed.score.0,
                best_immediate.score.0
            );
            let pos = Position::new_from_sfen(delayed).unwrap();
            let spite_check = Move::new_from_usi_str("N*4g", &pos).unwrap();
            assert_eq!(
                best_delayed.pv[0], spite_check,
                "the losing engine must pick the delaying spite check"
            );
        }

        // The 1-ply-mate shortcut must not claim a mate landing past the horizon (game ply == limit, so the mate would land on limit + 1, a draw).
        #[test]
        fn one_ply_mate_exactly_at_horizon_boundary_is_a_draw() {
            // At limit 15, after the forced 9c9d the black node sits at the boundary (ply 15): its R*1b "mate" would land on ply 16, so everything is a draw.
            let sfen = format!("{} 14", LOSING_SIDE_SFEN_BODY);
            let (best, _) = run_search(&sfen, "15", 6);
            assert_eq!(
                best.score,
                Value::DRAW,
                "a 1-ply mate landing past the horizon must not be claimed; got {}",
                best.score.0
            );

            // One more move of room (limit 16) and the same mate counts again.
            let (best, _) = run_search(&sfen, "16", 6);
            assert!(
                best.score <= Value::MATED_IN_MAX_PLY,
                "with the mate inside the horizon the losing side is mated again; got {}",
                best.score.0
            );
        }

        // `numa` OFF/ON invariance: a single-thread fixed-depth search on the start position must give the same bestmove and node count either way (the feature only pins and picks a replica). Literals captured on the OFF build.
        #[test]
        fn numa_off_on_search_is_invariant_at_threads_1() {
            let startpos = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/9/1B5R1/LNSGKGSNL b - 1";
            let (best, nodes) = run_search(startpos, "0", 6);
            let pos = Position::new_from_sfen(startpos).unwrap();
            let expected_best = Move::new_from_usi_str("2h4h", &pos).unwrap();
            assert_eq!(best.pv[0], expected_best, "single-thread bestmove must be NUMA-invariant");
            assert_eq!(nodes, 1610, "single-thread node count must be NUMA-invariant");
        }

        #[test]
        fn zero_means_unlimited_and_changes_nothing() {
            // 0 = unlimited: away from any limit the search is identical node for node.
            let startpos = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/9/1B5R1/LNSGKGSNL b - 1";
            let (best_unlimited, nodes_unlimited) = run_search(startpos, "0", 6);
            for limit in ["512", "100000"] {
                let (best, nodes) = run_search(startpos, limit, 6);
                assert_eq!(best.score, best_unlimited.score, "limit {limit}: score must not change");
                assert_eq!(best.pv[0], best_unlimited.pv[0], "limit {limit}: bestmove must not change");
                assert_eq!(nodes, nodes_unlimited, "limit {limit}: node count must not change");
            }
        }
    }

    // Tournament build: draw-horizon snapshot is the compile-time const from build.rs, not the runtime USI option.
    #[cfg(feature = "tournament")]
    mod tournament_const_tests {
        #[test]
        fn max_moves_to_draw_const_is_fixed() {
            // `const` context: compiles only if this is a genuine compile-time const.
            const _: i32 = crate::tournament::MAX_MOVES_TO_DRAW;
            assert_eq!(
                crate::tournament::MAX_MOVES_TO_DRAW,
                512,
                "v1 tournament config bakes Max_Moves_To_Draw = 512",
            );
        }

        #[test]
        fn draw_contempt_const_is_fixed() {
            // `const` context: compiles only if this is a genuine compile-time const.
            const _: i32 = crate::tournament::DRAW_CONTEMPT;
            assert_eq!(
                crate::tournament::DRAW_CONTEMPT,
                300,
                "v1 tournament config bakes draw_contempt = 300",
            );
        }

        #[test]
        fn usi_ponder_const_is_fixed() {
            use super::super::*;
            // `const` context: compiles only if this is a genuine compile-time const.
            const _: bool = crate::tournament::USI_PONDER;
            // Value check via the accessor (a fn call); under `tournament` it returns the baked const.
            let opts = UsiOptions::new();
            assert!(usi_ponder(&opts), "v1 tournament config bakes USI_Ponder = true");
        }

        // `usi_ponder` under `tournament` returns the baked const and ignores the runtime option. Big-stack thread because `ThreadPool::set` builds the large `Thread` struct on the stack.
        #[test]
        fn usi_ponder_accessor_ignores_runtime_option() {
            use super::super::*;
            std::thread::Builder::new()
                .stack_size(crate::stack_size::STACK_SIZE)
                .spawn(|| {
                    let mut reductions = Reductions::new();
                    let mut thread_pool = ThreadPool::new();
                    let mut tt = TranspositionTable::new();
                    thread_pool.set(1, &mut tt, &mut reductions);
                    let mut usi_options = UsiOptions::new();
                    let mut is_ready = true;
                    // Flip the runtime option to the opposite of the baked const (true -> false).
                    usi_options.set(
                        UsiOptions::USI_PONDER,
                        "false",
                        &mut thread_pool,
                        &mut tt,
                        &mut reductions,
                        &mut is_ready,
                    );
                    assert!(
                        !usi_options.get_bool(UsiOptions::USI_PONDER),
                        "runtime option must now be false"
                    );
                    assert!(
                        usi_ponder(&usi_options),
                        "the tournament accessor must return the baked USI_PONDER const, ignoring the runtime option",
                    );
                })
                .unwrap()
                .join()
                .unwrap();
        }
    }

    // Tournament build: single PV, so the info line carries no `multipv` token. `pv_info_to_usi_string` needs no eval network, so this runs in the `nnue,tournament` CI row. Big-stack thread for `ThreadPool::set`.
    #[cfg(feature = "tournament")]
    mod tournament_single_pv_tests {
        use super::super::*;

        #[test]
        fn single_pv_info_has_no_multipv_token() {
            std::thread::Builder::new()
                .stack_size(crate::stack_size::STACK_SIZE)
                .spawn(|| {
                    let mut reductions = Reductions::new();
                    let mut thread_pool = ThreadPool::new();
                    let mut tt = TranspositionTable::new();
                    thread_pool.set(1, &mut tt, &mut reductions);

                    // Two legal root moves, yet the single-PV emitter must print exactly one PV line and no `multipv` token.
                    let pos = Position::new_from_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1").unwrap();
                    let root_moves = ["7g7f", "2g2f"]
                        .iter()
                        .map(|s| RootMove::new(Move::new_from_usi_str(s, &pos).unwrap()))
                        .collect::<Vec<_>>();

                    let pool_base = thread_pool.thread_pool_base.lock().unwrap();
                    let mut th = pool_base.threads[0].lock().unwrap();
                    th.root_moves = root_moves;
                    th.limits.start_time = Some(std::time::Instant::now());

                    let info = th.pv_info_to_usi_string(0, 1, Depth(5), -Value::INFINITE, Value::INFINITE, false);
                    assert!(
                        !info.contains("multipv"),
                        "tournament info output must not carry a `multipv` token; got:\n{info}"
                    );
                    assert_eq!(
                        info.lines().count(),
                        1,
                        "the single-PV root search must emit exactly one PV line; got:\n{info}"
                    );
                })
                .unwrap()
                .join()
                .unwrap();
        }
    }

    // Benchmark build (`tournament,emit-nps`): the search-end line must carry the `nps <integer>` token and no PV payload.
    #[cfg(all(feature = "tournament", feature = "emit-nps"))]
    mod emit_nps_tests {
        use super::super::*;

        #[test]
        fn bench_nps_line_carries_the_nps_token_and_no_pv() {
            let line = bench_nps_info_string(1_234_567, 1001);
            assert_eq!(line, "info nodes 1234567 nps 1233333 time 1001");

            // The exact token shape benchmark tooling greps for.
            let re = regex::Regex::new(r"\bnps\s+(\d+)\b").unwrap();
            let caps = re
                .captures(&line)
                .expect("the bench line must carry an `nps <integer>` token");
            assert_eq!(caps[1].parse::<i64>().unwrap(), 1_234_567 * 1000 / 1001);

            assert!(
                !line.contains("depth") && !line.contains("pv") && !line.contains("score"),
                "the bench line must stay minimal (no PV/depth/score payload); got: {line}"
            );
        }
    }
}
