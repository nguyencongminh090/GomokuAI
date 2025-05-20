"""
Main Board class for the Gomoku engine, managing game state and operations.
Based on Rapfi's board.h and board.cpp.
"""
import sys
import math
from typing import List, Tuple, Optional, Iterator, Callable, Any, Dict # Added Iterator, Callable
from enum import IntEnum # Added IntEnum

from .types import (Pattern, Pattern4, Color, Rule, PatternCode, Score, Value,
                    SIDE_NB, PATTERN_NB, PATTERN4_NB, RULE_NB, CandidateRange)
from .pos import (Pos, Direction, FULL_BOARD_SIZE, FULL_BOARD_CELL_COUNT, MAX_BOARD_SIZE,
                  BOARD_BOUNDARY, DIRECTIONS, RANGE_SQUARE2, RANGE_SQUARE2_LINE3,
                  RANGE_SQUARE3, RANGE_SQUARE3_LINE4, RANGE_SQUARE4) # Import specific ranges
from .hash_utils import zobrist_table, zobrist_side, lc_hash # Zobrist tables and LCHash
from .board_utils import Cell, StateInfo, CandArea # Helper classes
from .pattern_utils import (Pattern2x, lookup_pattern_from_luts, get_pcode_from_patterns,
                            fuse_key, get_half_line_len, init_pattern_config)

from . import config as engine_config
SearchThreadBase = Any 
EvaluatorBase = Any


def rotr64(val: int, shift: int, bits: int = 64) -> int:
    shift &= (bits - 1) 
    mask = (1 << bits) - 1
    return ((val >> shift) | (val << (bits - shift))) & mask

class Board:
    class MoveType(IntEnum): 
        NORMAL = 0        
        NO_EVALUATOR = 1  
        NO_EVAL = 2       
        NO_EVAL_MULTI = 3 

    def __init__(self, board_size: int,
                 cand_range_enum: CandidateRange = engine_config.DEFAULT_CANDIDATE_RANGE,
                 search_thread: Optional[SearchThreadBase] = None): 
        
        if not (0 < board_size <= MAX_BOARD_SIZE):
            raise ValueError(f"Board size {board_size} out of range (1-{MAX_BOARD_SIZE})")

        self.board_size: int = board_size
        self.board_cell_count: int = board_size * board_size
        self.move_count: int = 0 
        self.pass_count: List[int] = [0] * SIDE_NB 
        self.current_side: Color = Color.BLACK
        self.current_zobrist_key: int = 0 

        self.candidate_range_offsets: Optional[Tuple[int, ...]] = None
        self.candidate_range_size: int = 0
        self.cand_area_expand_dist: int = 0 

        self._setup_candidate_range(cand_range_enum)

        self.evaluator_instance: Optional[EvaluatorBase] = None 
        self.this_thread: Optional[SearchThreadBase] = search_thread
        
        self.cells: List[Cell] = [Cell() for _ in range(FULL_BOARD_CELL_COUNT)]
        
        self.bit_key0: List[int] = [0] * FULL_BOARD_SIZE 
        self.bit_key1: List[int] = [0] * FULL_BOARD_SIZE 
        self.bit_key2: List[int] = [0] * (FULL_BOARD_SIZE * 2 - 1) 
        self.bit_key3: List[int] = [0] * (FULL_BOARD_SIZE * 2 - 1) 
        
        max_game_plies = self.board_cell_count + 10 
        self.state_infos: List[StateInfo] = [StateInfo() for _ in range(max_game_plies + 1)] 
        self.update_cache: List[Dict[int, Tuple]] = [{} for _ in range(max_game_plies + 1)] 


    def _setup_candidate_range(self, cand_range_enum: CandidateRange):
        if cand_range_enum == CandidateRange.SQUARE2:
            self.candidate_range_offsets = RANGE_SQUARE2
            self.cand_area_expand_dist = 2
        elif cand_range_enum == CandidateRange.SQUARE2_LINE3:
            self.candidate_range_offsets = RANGE_SQUARE2_LINE3
            self.cand_area_expand_dist = 3
        elif cand_range_enum == CandidateRange.SQUARE3:
            self.candidate_range_offsets = RANGE_SQUARE3
            self.cand_area_expand_dist = 3
        elif cand_range_enum == CandidateRange.SQUARE3_LINE4:
            self.candidate_range_offsets = RANGE_SQUARE3_LINE4
            self.cand_area_expand_dist = 3
        elif cand_range_enum == CandidateRange.SQUARE4:
            self.candidate_range_offsets = RANGE_SQUARE4
            self.cand_area_expand_dist = 4
        elif cand_range_enum == CandidateRange.FULL_BOARD:
            self.candidate_range_offsets = None 
            self.cand_area_expand_dist = self.board_size 
        else: 
            self.candidate_range_offsets = RANGE_SQUARE3_LINE4 
            self.cand_area_expand_dist = 3

        if self.candidate_range_offsets:
            self.candidate_range_size = len(self.candidate_range_offsets)
        else:
            self.candidate_range_size = 0 

    def _iter_all_playable_positions(self) -> Iterator[Pos]:
        for y in range(self.board_size):
            for x in range(self.board_size):
                yield Pos(x, y)

    def _iter_all_raw_positions(self) -> Iterator[Pos]:
        current_pos_val = Pos.FULL_BOARD_START._pos
        end_pos_val = Pos.FULL_BOARD_END._pos
        while current_pos_val < end_pos_val:
            yield Pos(current_pos_val)
            current_pos_val += 1
            
    def iter_empty_positions(self) -> Iterator[Pos]:
        for pos in self._iter_all_playable_positions():
            if self.cells[pos._pos].piece == Color.EMPTY:
                yield pos
                
    def iter_candidate_area_positions(self) -> Iterator[Pos]:
        current_cand_area = self.state_infos[self.move_count].cand_area
        if current_cand_area.is_empty():
            return 

        for y_coord in range(current_cand_area.y0, current_cand_area.y1 + 1):
            for x_coord in range(current_cand_area.x0, current_cand_area.x1 + 1):
                yield Pos(x_coord, y_coord)

    def iter_candidate_moves(self) -> Iterator[Pos]:
        for pos in self.iter_candidate_area_positions():
            cell_obj = self.cells[pos._pos]
            if cell_obj.piece == Color.EMPTY and cell_obj.is_candidate():
                yield pos

    def _set_bit_key_at_pos(self, pos: Pos, color: Color):
        raw_x = pos.x + BOARD_BOUNDARY
        raw_y = pos.y + BOARD_BOUNDARY
        color_bits = (0x1 + color.value) 
        
        mask_to_xor = color_bits << (2 * raw_x)
        self.bit_key0[raw_y] ^= mask_to_xor
        
        mask_to_xor = color_bits << (2 * raw_y)
        self.bit_key1[raw_x] ^= mask_to_xor
        
        mask_to_xor = color_bits << (2 * raw_x) 
        self.bit_key2[raw_x + raw_y] ^= mask_to_xor
        self.bit_key3[FULL_BOARD_SIZE - 1 - raw_x + raw_y] ^= mask_to_xor

    def _initialize_empty_bitkey_at_pos(self, pos: Pos):
        raw_x = pos.x + BOARD_BOUNDARY
        raw_y = pos.y + BOARD_BOUNDARY
        
        clear_mask_x = ~ (0b11 << (2 * raw_x))
        self.bit_key0[raw_y] &= clear_mask_x
        
        clear_mask_y = ~ (0b11 << (2 * raw_y))
        self.bit_key1[raw_x] &= clear_mask_y
        
        self.bit_key2[raw_x + raw_y] &= clear_mask_x
        self.bit_key3[FULL_BOARD_SIZE - 1 - raw_x + raw_y] &= clear_mask_x


    def get_key_segment_at(self, rule: Rule, pos: Pos, dir_idx: int) -> int:
        half_len = get_half_line_len(rule)
        raw_x = pos.x + BOARD_BOUNDARY
        raw_y = pos.y + BOARD_BOUNDARY
        val = 0
        shift = 0
        if dir_idx == 0: 
            val = self.bit_key0[raw_y]
            shift = 2 * (raw_x - half_len)
        elif dir_idx == 1: 
            val = self.bit_key1[raw_x]
            shift = 2 * (raw_y - half_len)
        elif dir_idx == 2: 
            val = self.bit_key2[raw_x + raw_y]
            shift = 2 * (raw_x - half_len) 
        elif dir_idx == 3: 
            val = self.bit_key3[FULL_BOARD_SIZE - 1 - raw_x + raw_y]
            shift = 2 * (raw_x - half_len) 
        else:
            raise ValueError("Invalid dir_idx for getKeyAt")
        return rotr64(val, shift)

    def new_game(self, rule: Rule):
        for i in range(FULL_BOARD_CELL_COUNT):
            self.cells[i] = Cell() 
        self.bit_key0 = [0] * FULL_BOARD_SIZE
        self.bit_key1 = [0] * FULL_BOARD_SIZE
        self.bit_key2 = [0] * (FULL_BOARD_SIZE * 2 - 1)
        self.bit_key3 = [0] * (FULL_BOARD_SIZE * 2 - 1)

        self.move_count = 0
        self.pass_count = [0,0]
        self.current_side = Color.BLACK
        self.current_zobrist_key = zobrist_side[Color.BLACK.value] 

        for pos_raw_val in range(Pos.FULL_BOARD_START._pos, Pos.FULL_BOARD_END._pos):
            pos = Pos(pos_raw_val)
            cell_obj = self.cells[pos_raw_val]
            if pos.is_on_board(self.board_size, self.board_size):
                cell_obj.piece = Color.EMPTY
                self._initialize_empty_bitkey_at_pos(pos)
            else:
                cell_obj.piece = Color.WALL
                self._set_bit_key_at_pos(pos, Color.BLACK) 
                self._set_bit_key_at_pos(pos, Color.WHITE) 

        st = self.state_infos[0]
        st.__init__() 
        
        current_value_black = 0 
        for pos in self._iter_all_playable_positions():
            cell_obj = self.cells[pos._pos] 
            if cell_obj.piece == Color.EMPTY:
                for dir_idx in range(4): 
                    key_segment = self.get_key_segment_at(rule, pos, dir_idx)
                    fused = fuse_key(rule, key_segment)
                    cell_obj.pattern2x[dir_idx] = lookup_pattern_from_luts(rule, fused)

                pcode_b = cell_obj.get_pcode(Color.BLACK)
                pcode_w = cell_obj.get_pcode(Color.WHITE)
                
                cell_obj.update_pattern4_and_score(rule, pcode_b, pcode_w)
                st.p4_count[Color.BLACK.value][cell_obj.pattern4[Color.BLACK.value].value] += 1
                st.p4_count[Color.WHITE.value][cell_obj.pattern4[Color.WHITE.value].value] += 1
                
                cell_obj.value_black = engine_config.get_value_black(rule, pcode_b, pcode_w)
                current_value_black += cell_obj.value_black
        
        st.value_black = current_value_black
        st.cand_area = CandArea() 

        if self.candidate_range_offsets is None: 
            self.expand_cand_area(self.center_pos(), self.board_size // 2 + 1, 0)
        
        if self.evaluator_instance and hasattr(self.evaluator_instance, 'init_empty_board'):
            self.evaluator_instance.init_empty_board()

    def make_move(self, rule: Rule, pos: Pos, move_type: MoveType = MoveType.NORMAL):
        if pos == Pos.PASS:
            if self.pass_count[self.current_side.value] + self.pass_count[(~self.current_side).value] >= self.board_cell_count * 2:
                pass 

            self.move_count += 1
            st_new = self.state_infos[self.move_count]
            st_prev = self.state_infos[self.move_count - 1]
            
            st_new.cand_area = CandArea(st_prev.cand_area.x0, st_prev.cand_area.y0, st_prev.cand_area.x1, st_prev.cand_area.y1)
            st_new.last_flex4_attack_move = list(st_prev.last_flex4_attack_move)
            st_new.last_pattern4_move = [list(sub) for sub in st_prev.last_pattern4_move]
            st_new.p4_count = [list(sub) for sub in st_prev.p4_count]
            st_new.value_black = st_prev.value_black

            st_new.last_move = Pos.PASS
            self.pass_count[self.current_side.value] += 1
            
            self.current_zobrist_key ^= zobrist_side[self.current_side.value] 
            self.current_side = ~self.current_side
            self.current_zobrist_key ^= zobrist_side[self.current_side.value] 

            if move_type == Board.MoveType.NORMAL and self.evaluator_instance and \
               hasattr(self.evaluator_instance, 'after_pass'):
                self.evaluator_instance.after_pass(self)
            return

        if not (pos.is_on_board(self.board_size, self.board_size) and self.cells[pos._pos].piece == Color.EMPTY):
            raise ValueError(f"Invalid move: {pos} on board_size {self.board_size} or cell not empty.")

        if move_type == Board.MoveType.NORMAL and self.evaluator_instance and \
           hasattr(self.evaluator_instance, 'before_move'):
            self.evaluator_instance.before_move(self, pos)

        current_ply_cache = self.update_cache[self.move_count] 
        current_ply_cache.clear()

        self.move_count += 1
        st_new = self.state_infos[self.move_count]
        st_prev = self.state_infos[self.move_count - 1]

        st_new.cand_area = CandArea(st_prev.cand_area.x0, st_prev.cand_area.y0, st_prev.cand_area.x1, st_prev.cand_area.y1)
        st_new.last_flex4_attack_move = list(st_prev.last_flex4_attack_move) 
        st_new.last_pattern4_move = [list(sub) for sub in st_prev.last_pattern4_move] 
        st_new.p4_count = [list(sub) for sub in st_prev.p4_count] 
        st_new.value_black = st_prev.value_black 
        
        st_new.last_move = pos
        st_new.cand_area.expand(pos, self.board_size, self.cand_area_expand_dist)
        
        placed_cell = self.cells[pos._pos]
        current_ply_cache[pos._pos] = (
            list(placed_cell.pattern4), 
            list(placed_cell.score),    
            placed_cell.value_black,
            placed_cell.piece 
        )
        
        if move_type != Board.MoveType.NO_EVAL_MULTI : 
            st_new.p4_count[Color.BLACK.value][placed_cell.pattern4[Color.BLACK.value].value] -= 1
            st_new.p4_count[Color.WHITE.value][placed_cell.pattern4[Color.WHITE.value].value] -= 1
        
        placed_cell.piece = self.current_side
        self.current_zobrist_key ^= zobrist_table[self.current_side.value][pos._pos]
        self._set_bit_key_at_pos(pos, self.current_side) 

        delta_value_black = 0
        if move_type == Board.MoveType.NORMAL or move_type == Board.MoveType.NO_EVALUATOR:
            delta_value_black -= placed_cell.value_black 
        
        f4_count_before_move = [st_prev.p4_count[c.value][Pattern4.B_FLEX4.value] for c in [Color.BLACK, Color.WHITE]]

        half_len_rule = get_half_line_len(rule)
        affected_positions_set = {pos} 
        for d_enum in DIRECTIONS: 
            for i in range(-half_len_rule, half_len_rule + 1):
                if i == 0 and d_enum != DIRECTIONS[0]: 
                    pass 
                current_eval_pos = pos + d_enum * i
                if current_eval_pos.is_on_board(self.board_size, self.board_size):
                    affected_positions_set.add(current_eval_pos)

        for affected_pos in affected_positions_set:
            affected_cell = self.cells[affected_pos._pos]
            if affected_cell.piece == Color.EMPTY : 
                if affected_pos != pos:
                    current_ply_cache[affected_pos._pos] = (
                        list(affected_cell.pattern4), list(affected_cell.score),
                        affected_cell.value_black, affected_cell.piece 
                    )
                
                if move_type == Board.MoveType.NORMAL or move_type == Board.MoveType.NO_EVALUATOR:
                    delta_value_black -= affected_cell.value_black 

                if move_type != Board.MoveType.NO_EVAL_MULTI:
                    st_new.p4_count[Color.BLACK.value][affected_cell.pattern4[Color.BLACK.value].value] -= 1
                    st_new.p4_count[Color.WHITE.value][affected_cell.pattern4[Color.WHITE.value].value] -= 1

                for dir_idx, _ in enumerate(DIRECTIONS):
                    key_segment = self.get_key_segment_at(rule, affected_pos, dir_idx)
                    fused = fuse_key(rule, key_segment)
                    affected_cell.pattern2x[dir_idx] = lookup_pattern_from_luts(rule, fused)
                
                pcode_b = affected_cell.get_pcode(Color.BLACK)
                pcode_w = affected_cell.get_pcode(Color.WHITE)
                affected_cell.update_pattern4_and_score(rule, pcode_b, pcode_w)

                if move_type != Board.MoveType.NO_EVAL_MULTI:
                    st_new.p4_count[Color.BLACK.value][affected_cell.pattern4[Color.BLACK.value].value] += 1
                    st_new.p4_count[Color.WHITE.value][affected_cell.pattern4[Color.WHITE.value].value] += 1
                
                if move_type == Board.MoveType.NORMAL or move_type == Board.MoveType.NO_EVALUATOR:
                    affected_cell.value_black = engine_config.get_value_black(rule, pcode_b, pcode_w)
                    delta_value_black += affected_cell.value_black 
                
                for c_idx, p_color in enumerate([Color.BLACK, Color.WHITE]):
                    p4_val = affected_cell.pattern4[c_idx]
                    if Pattern4.C_BLOCK4_FLEX3.value <= p4_val.value <= Pattern4.A_FIVE.value:
                        p4_offset = p4_val.value - Pattern4.C_BLOCK4_FLEX3.value
                        st_new.last_pattern4_move[c_idx][p4_offset] = affected_pos
            
        if move_type == Board.MoveType.NORMAL or move_type == Board.MoveType.NO_EVALUATOR:
            st_new.value_black += delta_value_black 

        if move_type != Board.MoveType.NO_EVAL_MULTI:
            self.current_zobrist_key ^= zobrist_side[self.current_side.value] 
            self.current_side = ~self.current_side
            self.current_zobrist_key ^= zobrist_side[self.current_side.value] 

        if self.candidate_range_offsets:
            for offset_val in self.candidate_range_offsets:
                cand_pos = pos + offset_val 
                if cand_pos.is_on_board(self.board_size, self.board_size):
                     self.cells[cand_pos._pos].cand += 1
        
        for c_enum in [Color.BLACK, Color.WHITE]:
            if f4_count_before_move[c_enum.value] == 0 and \
               st_new.p4_count[c_enum.value][Pattern4.B_FLEX4.value] > 0:
                st_new.last_flex4_attack_move[c_enum.value] = pos

        if move_type == Board.MoveType.NORMAL and self.evaluator_instance and \
           hasattr(self.evaluator_instance, 'after_move'):
            self.evaluator_instance.after_move(self, pos)
            
    def undo_move(self, rule: Rule, move_type: MoveType = MoveType.NORMAL):
        if self.move_count == 0:
            raise Exception("Cannot undo from initial board state.")

        last_move_pos = self.state_infos[self.move_count].last_move

        if last_move_pos == Pos.PASS:
            if move_type != Board.MoveType.NO_EVAL_MULTI: 
                self.current_zobrist_key ^= zobrist_side[self.current_side.value] 
                self.current_side = ~self.current_side
                self.current_zobrist_key ^= zobrist_side[self.current_side.value] 
            
            self.pass_count[self.current_side.value] -= 1
            self.move_count -= 1

            if move_type == Board.MoveType.NORMAL and self.evaluator_instance and \
               hasattr(self.evaluator_instance, 'after_undo_pass'):
                self.evaluator_instance.after_undo_pass(self)
            return

        if move_type == Board.MoveType.NORMAL and self.evaluator_instance and \
           hasattr(self.evaluator_instance, 'before_undo'):
            self.evaluator_instance.before_undo(self, last_move_pos)

        if move_type != Board.MoveType.NO_EVAL_MULTI:
            self.current_zobrist_key ^= zobrist_side[self.current_side.value]
            self.current_side = ~self.current_side
            self.current_zobrist_key ^= zobrist_side[self.current_side.value]
        
        self.current_zobrist_key ^= zobrist_table[self.current_side.value][last_move_pos._pos]
        self._set_bit_key_at_pos(last_move_pos, self.current_side) 
        
        cache_to_restore = self.update_cache[self.move_count -1]
        for cached_pos_val, (old_p4s, old_scores, old_val_b, old_piece) in cache_to_restore.items():
            cell_to_restore = self.cells[cached_pos_val]
            cell_to_restore.pattern4 = list(old_p4s) 
            cell_to_restore.score = list(old_scores) 
            cell_to_restore.value_black = old_val_b
            if cached_pos_val == last_move_pos._pos: 
                cell_to_restore.piece = Color.EMPTY 
            
        half_len_rule = get_half_line_len(rule)
        affected_positions_set_undo = {last_move_pos}
        for d_enum in DIRECTIONS:
            for i in range(-half_len_rule, half_len_rule + 1):
                current_eval_pos = last_move_pos + d_enum * i
                if current_eval_pos.is_on_board(self.board_size, self.board_size):
                    affected_positions_set_undo.add(current_eval_pos)
        
        for affected_pos_undo in affected_positions_set_undo:
            cell_obj_undo = self.cells[affected_pos_undo._pos]
            if cell_obj_undo.piece == Color.EMPTY: 
                for dir_idx, _ in enumerate(DIRECTIONS):
                    key_segment = self.get_key_segment_at(rule, affected_pos_undo, dir_idx)
                    fused = fuse_key(rule, key_segment)
                    cell_obj_undo.pattern2x[dir_idx] = lookup_pattern_from_luts(rule, fused)
                
        if self.candidate_range_offsets:
            for offset_val in self.candidate_range_offsets:
                cand_pos = last_move_pos + offset_val
                if cand_pos.is_on_board(self.board_size, self.board_size):
                    self.cells[cand_pos._pos].cand -= 1
        
        self.move_count -= 1

        if move_type == Board.MoveType.NORMAL and self.evaluator_instance and \
           hasattr(self.evaluator_instance, 'after_undo'):
            self.evaluator_instance.after_undo(self, last_move_pos)

    # --- Queries ---
    def get_cell(self, pos: Pos) -> Cell:
        if not (0 <= pos._pos < FULL_BOARD_CELL_COUNT): 
            raise IndexError(f"Position {pos} raw value {pos._pos} out of bounds for cells array.")
        return self.cells[pos._pos]

    def get_piece_at(self, pos: Pos) -> Color:
        return self.get_cell(pos).piece

    def is_on_board(self, pos: Pos) -> bool:
        return pos.is_on_board(self.board_size, self.board_size)

    def is_empty(self, pos: Pos) -> bool:
        return self.get_piece_at(pos) == Color.EMPTY

    def is_legal(self, pos: Pos) -> bool:
        if pos == Pos.PASS: return True
        return pos.is_on_board(self.board_size, self.board_size) and self.is_empty(pos)

    def ply(self) -> int:
        return self.move_count
    
    # ***** ADD THIS METHOD *****
    def side_to_move(self) -> Color:
        """Returns the color of the current player to move."""
        return self.current_side
    # ***************************
    
    def get_current_state_info(self) -> StateInfo:
        return self.state_infos[self.move_count]

    def get_history_move(self, move_idx_in_game: int) -> Pos: 
        if not (0 <= move_idx_in_game < self.move_count):
            raise IndexError("Move index out of game history bounds.")
        return self.state_infos[move_idx_in_game + 1].last_move 

    def center_pos(self) -> Pos:
        return Pos(self.board_size // 2, self.board_size // 2)

    def expand_cand_area(self, pos: Pos, fill_dist: int, line_dist: int):
        current_st_info = self.get_current_state_info()
        area_dist = max(fill_dist, line_dist, self.cand_area_expand_dist)
        current_st_info.cand_area.expand(pos, self.board_size, area_dist)

        for x_offset in range(-fill_dist, fill_dist + 1):
            for y_offset in range(-fill_dist, fill_dist + 1):
                check_pos = Pos(pos.x + x_offset, pos.y + y_offset)
                if check_pos.is_on_board(self.board_size, self.board_size) and \
                   self.is_empty(check_pos) and not self.get_cell(check_pos).is_candidate():
                    self.get_cell(check_pos).cand +=1
        
        if line_dist > fill_dist:
            for d_enum in DIRECTIONS: 
                for i in range(fill_dist + 1, line_dist + 1): 
                    check_pos_plus = pos + d_enum * i
                    check_pos_minus = pos - d_enum * i
                    for p_to_check in [check_pos_plus, check_pos_minus]:
                        if p_to_check.is_on_board(self.board_size, self.board_size) and \
                           self.is_empty(p_to_check) and not self.get_cell(p_to_check).is_candidate():
                            self.get_cell(p_to_check).cand +=1
                            
    def to_string(self) -> str:
        s = []
        header = "   " + " ".join([chr(ord('A') + i) for i in range(self.board_size)])
        s.append(header)
        for y in range(self.board_size -1, -1, -1): 
            row_str = f"{str(y+1).ljust(2)} "
            for x in range(self.board_size):
                pos = Pos(x,y)
                piece = self.get_piece_at(pos)
                if piece == Color.BLACK: row_str += "X "
                elif piece == Color.WHITE: row_str += "O "
                elif piece == Color.EMPTY:
                    row_str += ". " if not self.get_cell(pos).is_candidate() else "* "
                else: row_str += "# " 
            s.append(row_str.strip())
        return "\n".join(s)

    def checkForbiddenPoint(self, pos: Pos) -> bool:
        cell_obj = self.get_cell(pos)
        if cell_obj.pattern4[Color.BLACK.value] == Pattern4.FORBID:
            return True
        return False


if __name__ == '__main__':
    print("--- Board Tests ---")
    board = Board(15, CandidateRange.SQUARE2) 
    board.new_game(Rule.FREESTYLE)
    print(board.to_string())
    print(f"Ply: {board.ply()}, Side to move: {board.side_to_move().name}") # Used side_to_move()
    print(f"Zobrist key: {board.current_zobrist_key:016x}")

    center = board.center_pos()
    board.make_move(Rule.FREESTYLE, center)
    print(f"\nAfter move at {center.x},{center.y}:")
    print(board.to_string())
    print(f"Ply: {board.ply()}, Side to move: {board.side_to_move().name}")
    print(f"Zobrist key: {board.current_zobrist_key:016x}")
    print(f"Last move in state: {board.get_current_state_info().last_move}")
    assert board.get_current_state_info().last_move == center
    assert board.get_piece_at(center) == Color.BLACK

    move2 = Pos(center.x + 1, center.y)
    board.make_move(Rule.FREESTYLE, move2)
    print(f"\nAfter move at {move2.x},{move2.y}:")
    print(board.to_string())
    print(f"Ply: {board.ply()}, Side to move: {board.side_to_move().name}")
    assert board.get_piece_at(move2) == Color.WHITE
    
    z_before_undo = board.current_zobrist_key
    s_before_undo = board.current_side

    board.undo_move(Rule.FREESTYLE)
    print(f"\nAfter undo (back to {center.x},{center.y} being last move):")
    print(board.to_string())
    print(f"Ply: {board.ply()}, Side to move: {board.side_to_move().name}")
    assert board.ply() == 1
    assert board.side_to_move() == Color.WHITE 
    assert board.get_piece_at(move2) == Color.EMPTY
    assert board.get_piece_at(center) == Color.BLACK
    
    board.undo_move(Rule.FREESTYLE)
    print(f"\nAfter second undo (empty board):")
    print(board.to_string())
    print(f"Ply: {board.ply()}, Side to move: {board.side_to_move().name}")
    assert board.ply() == 0
    assert board.side_to_move() == Color.BLACK
    assert board.get_piece_at(center) == Color.EMPTY

    print("Board tests completed (basic functionality).")