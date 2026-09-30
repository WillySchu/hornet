"""Tests for codegen/divide_by_constant.py's magic numbers, checked against truncating division."""

import random

import pytest

from backend.common.division import is_power_of_two, magic

M64 = (1 << 64) - 1


def _s64(v: int) -> int:
    v &= M64
    return v - (1 << 64) if v >> 63 else v


def _trunc(n: int, d: int) -> int:
    q = abs(n) // abs(d)
    return q if (n < 0) == (d < 0) else -q


def _via_magic(n: int, d: int) -> int:
    m, s = magic(d)
    t = (m * n) >> 64
    if m < 0:
        t = _s64(t + n)
    t >>= s
    return _s64(t + ((t & M64) >> 63))


DIVISORS = [d for d in range(3, 400) if not is_power_of_two(d)]
DIVISORS += [1000003, 7919, 3 ** 39, 10 ** 18, 2 ** 31 - 1, 2 ** 31 + 1, 2 ** 62 + 1, 2 ** 63 - 1, 2 ** 63 - 3]


@pytest.mark.parametrize('d', DIVISORS)
def test_magic_quotient_matches_truncating_division(d):
    r = random.Random(d)
    values = [-(2 ** 63), 2 ** 63 - 1, 0, 1, -1] + [_s64(d * k + e) for k in (1, 2, 3, -1, -9, 1000) for e in (-1, 0, 1)]
    values += [r.randint(-(2 ** 63), 2 ** 63 - 1) for _ in range(200)]
    for n in values:
        assert _via_magic(n, d) == _trunc(n, d), (n, d)


def test_magic_rejects_powers_of_two_and_small_divisors():
    for bad in (0, 1, 2, 64, 1 << 62):
        with pytest.raises(AssertionError):
            magic(bad)
