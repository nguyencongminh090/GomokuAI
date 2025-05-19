import numpy as np
from .types import Rule, Color, Pattern, Pattern4
from .config import Config

class Pattern2x:
    """Matches Pattern2x struct in pattern.h."""
    def __init__(self, pat_black=Pattern.DEAD, pat_white=Pattern.DEAD):
        self.pat_black = pat_black
        self.pat_white = pat_white

    def pat(self, side):
        return self.pat_black if side == Color.BLACK else self.pat_white

class PatternConfig:
    HALF_LINE_LEN = {Rule.FREESTYLE: 4, Rule.STANDARD: 5, Rule.RENJU: 5}
    KEY_CNT = {
        Rule.FREESTYLE: 1 << (HALF_LINE_LEN[Rule.FREESTYLE] * 4 - 2),  # 1 << 14
        Rule.STANDARD: 1 << (HALF_LINE_LEN[Rule.STANDARD] * 4 - 2),    # 1 << 18
        Rule.RENJU: 1 << (HALF_LINE_LEN[Rule.RENJU] * 4 - 2)           # 1 << 18
    }

    PATTERN2x = {
        Rule.FREESTYLE: np.array([Pattern2x() for _ in range(KEY_CNT[Rule.FREESTYLE])], dtype=object),
        Rule.STANDARD: np.array([Pattern2x() for _ in range(KEY_CNT[Rule.STANDARD])], dtype=object),
        Rule.RENJU: np.array([Pattern2x() for _ in range(KEY_CNT[Rule.RENJU])], dtype=object)
    }
    PCODE = np.zeros((Pattern.PATTERN_NB, Pattern.PATTERN_NB, Pattern.PATTERN_NB, Pattern.PATTERN_NB), dtype=np.uint16)

    @staticmethod
    def half_line_len(rule):
        return PatternConfig.HALF_LINE_LEN[rule]

    @staticmethod
    def init_patterns():
        """Initialize pattern tables and PCODE."""
        for rule in [Rule.FREESTYLE, Rule.STANDARD, Rule.RENJU]:
            half_len = PatternConfig.HALF_LINE_LEN[rule]
            key_cnt = PatternConfig.KEY_CNT[rule]
            memo_black = {}
            memo_white = {}

            def get_pattern(line, side, memo):
                code = ''.join(str(c) for c in line)
                if code in memo:
                    return memo[code]
                real_len, full_len, start, end = PatternConfig.count_line(line)
                check_overline = rule in [Rule.STANDARD, Rule.RENJU] and side == Color.BLACK
                
                if check_overline and real_len >= 6:
                    p = Pattern.OL
                elif real_len >= 5:
                    p = Pattern.F5
                elif full_len < 5:
                    p = Pattern.DEAD
                else:
                    pat_cnt = [0] * Pattern.PATTERN_NB
                    f5_idx = []
                    mid = len(line) // 2
                    for i in range(start, end + 1):
                        if line[i] == 0:  # EMPT
                            sl = PatternConfig.shift_line(line, i)
                            sl[mid] = 1  # SELF
                            slp = get_pattern(sl, side, memo)
                            if slp == Pattern.F5 and len(f5_idx) < 2:
                                f5_idx.append(i)
                            pat_cnt[slp] += 1
                    if pat_cnt[Pattern.F5] >= 2:
                        p = Pattern.F4
                        if rule == Rule.RENJU and side == Color.BLACK and len(f5_idx) == 2 and f5_idx[1] - f5_idx[0] < 5:
                            p = Pattern.OL
                    elif pat_cnt[Pattern.F5]:
                        p = Pattern.B4
                    elif pat_cnt[Pattern.F4] >= 2:
                        p = Pattern.F3S
                    elif pat_cnt[Pattern.F4]:
                        p = Pattern.F3
                    elif pat_cnt[Pattern.B4]:
                        p = Pattern.B3
                    elif pat_cnt[Pattern.F3S] + pat_cnt[Pattern.F3] >= 4:
                        p = Pattern.F2B
                    elif pat_cnt[Pattern.F3S] + pat_cnt[Pattern.F3] >= 3:
                        p = Pattern.F2A
                    elif pat_cnt[Pattern.F3S] + pat_cnt[Pattern.F3] >= 2:
                        p = Pattern.F2
                    elif pat_cnt[Pattern.B3]:
                        p = Pattern.B2
                    elif pat_cnt[Pattern.F2] + pat_cnt[Pattern.F2A] + pat_cnt[Pattern.F2B]:
                        p = Pattern.F1
                    elif pat_cnt[Pattern.B2]:
                        p = Pattern.B1
                    else:
                        p = Pattern.DEAD
                memo[code] = p
                return p

            for key in range(key_cnt):
                line_black = PatternConfig.line_from_key(key, Color.BLACK, half_len)
                line_white = PatternConfig.line_from_key(key, Color.WHITE, half_len)
                pat_black = get_pattern(line_black, Color.BLACK, memo_black)
                pat_white = get_pattern(line_white, Color.WHITE, memo_white)
                PatternConfig.PATTERN2x[rule][key] = Pattern2x(pat_black, pat_white)

        # Simplified PCODE initialization to avoid high values
        n = Pattern.PATTERN_NB
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        PatternConfig.PCODE[i, j, k, l] = (i << 12) | (j << 8) | (k << 4) | l  # Direct encoding

    @staticmethod
    def line_from_key(key, self_color, half_len):
        line = [0] * (2 * half_len + 1)  # 0: EMPT, 1: SELF, 2: OPPO
        mid = half_len
        line[mid] = 1  # SELF at center
        for i in range(half_len):
            bits = (key >> (2 * i)) & 0b11
            line[mid - i - 1] = 2 if bits == (self_color ^ 1) else (1 if bits == self_color else 0)
            if i < half_len - 1:
                line[mid + i + 1] = 2 if bits == (self_color ^ 1) else (1 if bits == self_color else 0)
        return line

    @staticmethod
    def count_line(line):
        mid = len(line) // 2
        real_len, full_len = 1, 1
        start, end = mid, mid
        real_len_inc = 1
        for i in range(mid - 1, -1, -1):
            if line[i] == 1:
                real_len += real_len_inc
            elif line[i] == 2:
                break
            else:
                real_len_inc = 0
            full_len += 1
            start = i
        real_len_inc = 1
        for i in range(mid + 1, len(line)):
            if line[i] == 1:
                real_len += real_len_inc
            elif line[i] == 2:
                break
            else:
                real_len_inc = 0
            full_len += 1
            end = i
        return real_len, full_len, start, end

    @staticmethod
    def shift_line(line, i):
        mid = len(line) // 2
        shifted = [2] * len(line)
        for j in range(len(line)):
            idx = j + i - mid
            if 0 <= idx < len(line):
                shifted[j] = line[idx]
        return shifted

    @staticmethod
    def fuse_key(key, rule):
        key = np.uint64(key)
        half_len = PatternConfig.half_line_len(rule)
        if half_len == 4:
            shifted = np.right_shift(key, np.uint64(2))
            masked = shifted & np.uint64(0x3f00)
            lower = key & np.uint64(0x00ff)
            return np.uint64(masked | lower)
        else:
            shifted = np.right_shift(key, np.uint64(2))
            masked = shifted & np.uint64(0xff800)
            lower = key & np.uint64(0x007ff)
            return np.uint64(masked | lower)

    @staticmethod
    def lookup_pattern(rule, key):
        fused_key = PatternConfig.fuse_key(key, rule)
        return PatternConfig.PATTERN2x[rule][fused_key]

    @staticmethod
    def pattern4_from_pcode(pcode):
        """Convert pcode to Pattern4 with bounds checking."""
        p1 = (pcode >> 12) & (Pattern.PATTERN_NB - 1)
        p2 = (pcode >> 8) & (Pattern.PATTERN_NB - 1)
        p3 = (pcode >> 4) & (Pattern.PATTERN_NB - 1)
        p4 = pcode & (Pattern.PATTERN_NB - 1)
        forbid = pcode >= Pattern.FORBID
        return PatternConfig.get_pattern4(p1, p2, p3, p4, forbid)

    @staticmethod
    def get_pattern4(p1, p2, p3, p4, forbid=False):
        counts = [0] * Pattern.PATTERN_NB
        counts[p1] += 1
        counts[p2] += 1
        counts[p3] += 1
        counts[p4] += 1

        if counts[Pattern.F5] >= 1:
            return Pattern4.A_FIVE
        if forbid and (counts[Pattern.OL] >= 1 or 
                       counts[Pattern.F4] + counts[Pattern.B4] >= 2 or 
                       counts[Pattern.F3] + counts[Pattern.F3S] >= 2):
            return Pattern4.FORBID
        if counts[Pattern.B4] >= 2 or counts[Pattern.F4] >= 1:
            return Pattern4.B_FLEX4
        if counts[Pattern.B4] >= 1:
            if counts[Pattern.F3] >= 1 or counts[Pattern.F3S] >= 1:
                return Pattern4.C_BLOCK4_FLEX3
            if counts[Pattern.B3] >= 1 or counts[Pattern.F2] + counts[Pattern.F2A] + counts[Pattern.F2B] >= 1:
                return Pattern4.D_BLOCK4_PLUS
            return Pattern4.E_BLOCK4
        if counts[Pattern.F3] >= 1 or counts[Pattern.F3S] >= 1:
            if counts[Pattern.F3] + counts[Pattern.F3S] >= 2:
                return Pattern4.F_FLEX3_2X
            if counts[Pattern.B3] >= 1 or counts[Pattern.F2] + counts[Pattern.F2A] + counts[Pattern.F2B] >= 1:
                return Pattern4.G_FLEX3_PLUS
            return Pattern4.H_FLEX3
        if counts[Pattern.B3] >= 1:
            if counts[Pattern.B3] >= 2 or counts[Pattern.F2] + counts[Pattern.F2A] + counts[Pattern.F2B] >= 1:
                return Pattern4.I_BLOCK3_PLUS
            return Pattern4.K_BLOCK3
        if counts[Pattern.F2] + counts[Pattern.F2A] + counts[Pattern.F2B] >= 2:
            return Pattern4.J_FLEX2_2X
        if counts[Pattern.F2] + counts[Pattern.F2A] + counts[Pattern.F2B] >= 1:
            return Pattern4.L_FLEX2
        return Pattern4.NONE

PatternConfig.init_patterns()