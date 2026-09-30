"""Copy coalescing: `t = <op>; x = t` with t read nowhere else becomes `x = <op>`."""

from dataclasses import replace

from ir.cfg import uses
from ir.ir import (
    IRBinOp, IRCall, IRCast, IRFunction, IRLoad, IRLocalAddress, IRMove, IRReadArgument, IRStaticDataAddress,
    IRUnOp, Temp,
)

_DEFINES = (IRMove, IRBinOp, IRUnOp, IRCast, IRLoad, IRLocalAddress, IRStaticDataAddress, IRCall, IRReadArgument)


def coalesce_copies(ir_fn: IRFunction, pinned: set) -> None:
    """Coalesce adjacent single-use copies in ir_fn.body in place. Pinned temps are left alone."""
    body = ir_fn.body
    use_sites = uses(body)
    out = []
    i = 0
    while i < len(body):
        d = body[i]
        m = body[i + 1] if i + 1 < len(body) else None
        if (isinstance(m, IRMove) and isinstance(m.src, Temp) and isinstance(d, _DEFINES)
                and d.dst is not None and d.dst == m.src and m.dst != m.src
                and use_sites.get(m.src.id) == [i + 1] and m.dst.type == m.src.type
                and m.src.id not in pinned and m.dst.id not in pinned):
            out.append(replace(d, dst=m.dst))
            i += 2
            continue
        out.append(d)
        i += 1
    ir_fn.body = out
