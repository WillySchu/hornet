"""Signed division by a constant via multiply-high (Granlund & Montgomery; Hacker's Delight 10-1)."""

M64 = (1 << 64) - 1


def is_power_of_two(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def magic(d: int) -> tuple[int, int]:
    """(M, s) for 2 <= d < 2**63, not a power of two: q = trunc(n / d) is
    t = mulhs(M, n); if M < 0: t += n; t >>= s; q = t + (t >>> 63). M is signed 64-bit."""
    assert 2 <= d < 1 << 63 and not is_power_of_two(d)
    two63 = 1 << 63
    anc = two63 - 1 - two63 % d
    p = 63
    q1, r1 = divmod(two63, anc)
    q2, r2 = divmod(two63, d)
    while True:
        p += 1
        q1, r1 = 2 * q1, 2 * r1
        if r1 >= anc:
            q1, r1 = q1 + 1, r1 - anc
        q2, r2 = 2 * q2, 2 * r2
        if r2 >= d:
            q2, r2 = q2 + 1, r2 - d
        delta = d - r2
        if not (q1 < delta or (q1 == delta and r1 == 0)):
            break
    m = (q2 + 1) & M64
    return (m - (1 << 64) if m >> 63 else m), p - 64
