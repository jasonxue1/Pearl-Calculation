"""Conversion helpers for direction/rb/num and 32-bit code strings."""

from typing import Tuple

import numpy as np

_BIT_COUNT = 32
_MAX_ABS_RB = 10880
_CODE_GROUP_LENGTHS = (3, 3, 1, 4, 4, 4, 4, 1, 3, 3, 2)

# 1-based positions in the 32-bit code.
_R_MAJOR_BITS = (
    (1, 340),
    (2, 680),
    (3, 1360),
    (4, 130),
    (5, 50),
    (6, 2720),
    (32, 5440),
)
_R_TENS_BITS = ((8, 10), (9, 20), (10, 40), (11, 80))
_R_ONES_BITS = ((12, 8), (13, 4), (14, 2), (15, 1))

_B_MAJOR_BITS = (
    (25, 2720),
    (26, 50),
    (27, 130),
    (28, 1360),
    (29, 680),
    (30, 340),
    (31, 5440),
)
_B_TENS_BITS = ((20, 80), (21, 40), (22, 20), (23, 10))
_B_ONES_BITS = ((16, 1), (17, 2), (18, 4), (19, 8))

_SIGN_BIT_POS_1 = 7
_SIGN_BIT_POS_2 = 24

# direction (0/1/2/3) -> [x, y] = coeff @ [r, b]
_COEFF_TABLE = np.array(
    [
        [[1, 0], [0, 1]],  # 0 -> 00
        [[-1, 0], [0, 1]],  # 1 -> 01
        [[0, -1], [-1, 0]],  # 2 -> 10
        [[0, 1], [-1, 0]],  # 3 -> 11
    ],
    dtype=np.int64,
)


def _as_int_vector(vector: np.ndarray, size: int, name: str) -> np.ndarray:
    if not isinstance(vector, np.ndarray):
        raise TypeError(f"{name} must be a numpy vector")
    if vector.ndim != 1 or vector.shape[0] != size:
        raise ValueError(f"{name} must be a {size}-dimensional vector")
    if not np.issubdtype(vector.dtype, np.integer):
        raise TypeError(f"{name} must contain integers")
    return vector.astype(np.int64, copy=False)


def _encode_abs_to_bits(
    value: int,
    major_bits: Tuple[Tuple[int, int], ...],
    tens_bits: Tuple[Tuple[int, int], ...],
    ones_bits: Tuple[Tuple[int, int], ...],
) -> np.ndarray:
    bits = np.zeros(_BIT_COUNT, dtype=np.int8)
    major_weights = [w for _, w in major_bits]

    chosen_mask = 0
    chosen_remain = 0
    found = False
    for mask in range(1 << len(major_bits)):
        major_sum = 0
        for i, weight in enumerate(major_weights):
            if (mask >> i) & 1:
                major_sum += weight

        remain = value - major_sum
        if 0 <= remain <= 160:
            chosen_mask = mask
            chosen_remain = remain
            found = True
            break

    if not found:
        return bits

    for i, (pos, _) in enumerate(major_bits):
        if (chosen_mask >> i) & 1:
            bits[pos - 1] = 1

    if chosen_remain <= 150:
        tens_value = chosen_remain - (chosen_remain % 10)
        ones_value = chosen_remain % 10
    else:
        tens_value = 150
        ones_value = chosen_remain - 150

    tens_digit = tens_value // 10
    for pos, weight in tens_bits:
        significance = weight // 10
        bits[pos - 1] = 1 if (tens_digit & significance) else 0

    ones_repr = 10 if ones_value == 10 else ones_value
    for pos, weight in ones_bits:
        bits[pos - 1] = 1 if (ones_repr & weight) else 0

    return bits


def _decode_abs_from_bits(
    bits: np.ndarray,
    major_bits: Tuple[Tuple[int, int], ...],
    tens_bits: Tuple[Tuple[int, int], ...],
    ones_bits: Tuple[Tuple[int, int], ...],
) -> int:
    major_sum = sum(int(bits[pos - 1]) * weight for pos, weight in major_bits)
    tens_sum = sum(int(bits[pos - 1]) * weight for pos, weight in tens_bits)
    ones_raw = sum(int(bits[pos - 1]) * weight for pos, weight in ones_bits)
    return major_sum + tens_sum + min(ones_raw, 10)


def _compact_code(code: str) -> str:
    return "".join(code.split())


def _format_code(code32: str) -> str:
    groups = []
    start = 0
    for length in _CODE_GROUP_LENGTHS:
        end = start + length
        groups.append(code32[start:end])
        start = end
    return f"{' '.join(groups[:5])}  {' '.join(groups[5:10])}  {groups[10]}"


def rb2num(rb: np.ndarray) -> np.ndarray:
    """[direction, r, b] -> [x, y]."""
    rb = _as_int_vector(rb, 3, "rb")
    direction, r, b = int(rb[0]), int(rb[1]), int(rb[2])
    if direction < 0 or direction > 3:
        raise ValueError("direction must be in [0, 1, 2, 3]")
    coeff = _COEFF_TABLE[direction]
    return coeff @ np.array([r, b], dtype=np.int64)


def num2rb(num: np.ndarray) -> np.ndarray:
    """[x, y] -> [direction, r, b]."""
    num = _as_int_vector(num, 2, "num")
    x, y = int(num[0]), int(num[1])

    if x >= 0 and y >= 0:
        return np.array([0, x, y], dtype=np.int64)
    if x <= 0 and y >= 0:
        return np.array([1, -x, y], dtype=np.int64)
    if x <= 0 and y <= 0:
        return np.array([2, -y, -x], dtype=np.int64)
    return np.array([3, -y, x], dtype=np.int64)


def rb2code(rb: np.ndarray) -> str:
    """[direction, r, b] -> formatted 32-bit code string."""
    rb = _as_int_vector(rb, 3, "rb")
    direction, r, b = int(rb[0]), int(rb[1]), int(rb[2])

    if direction < 0 or direction > 3:
        raise ValueError("direction must be in [0, 1, 2, 3]")
    if r < 0 or b < 0:
        raise ValueError("r and b must be natural numbers (>= 0)")
    if r > _MAX_ABS_RB or b > _MAX_ABS_RB:
        raise ValueError("r and b must be <= 10880")

    code0 = direction // 2
    code1 = direction % 2

    bits = np.zeros(_BIT_COUNT, dtype=np.int8)
    r_bits = _encode_abs_to_bits(r, _R_MAJOR_BITS, _R_TENS_BITS, _R_ONES_BITS)
    b_bits = _encode_abs_to_bits(b, _B_MAJOR_BITS, _B_TENS_BITS, _B_ONES_BITS)
    bits = np.maximum(bits, r_bits)
    bits = np.maximum(bits, b_bits)
    bits[_SIGN_BIT_POS_1 - 1] = code0
    bits[_SIGN_BIT_POS_2 - 1] = code1

    code32 = "".join(str(int(v)) for v in bits)
    return _format_code(code32)


def code2rb(code: str) -> np.ndarray:
    """32-bit code string -> [direction, r, b]."""
    if not isinstance(code, str):
        raise TypeError("code must be a string")

    compact_code = _compact_code(code)
    if len(compact_code) != _BIT_COUNT:
        raise ValueError("code must contain exactly 32 bits")

    bits = np.fromiter(
        (ch == "1" for ch in compact_code), dtype=np.int8, count=_BIT_COUNT
    )
    code0 = int(bits[_SIGN_BIT_POS_1 - 1])
    code1 = int(bits[_SIGN_BIT_POS_2 - 1])
    direction = code0 * 2 + code1
    r = _decode_abs_from_bits(bits, _R_MAJOR_BITS, _R_TENS_BITS, _R_ONES_BITS)
    b = _decode_abs_from_bits(bits, _B_MAJOR_BITS, _B_TENS_BITS, _B_ONES_BITS)
    return np.array([direction, r, b], dtype=np.int64)


if __name__ == "__main__":
    sample_code = "111 111 1 1111 1111  1111 1111 1 111 111  11"
    print(code2rb(sample_code))
    print(rb2num(code2rb(sample_code)))
