"""Safepoints: a check that the runtime may stop the task, on every way a program can run on without
end. In Hornet that is a loop (a cycle of a function's control flow) or recursion (a cycle of the
call graph; every call's target is known). So a check goes at the start of each loop's header (the
target of a branch back, found from the optimized code, whatever loops the source had), and at the
entry of each function in a cycle of calls. Code that is neither finishes in a time bounded by its
size.

A prototype: placed only when HORNET_SAFEPOINTS is set, for measuring what the checks cost. With
HORNET_SAFEPOINTS=2, a loop whose every time round calls a function that checks at its entry has no
check of its own."""

import os

from ir.cfg import build_blocks, flatten, reverse_postorder
from ir.ir import IRCall, IRLabel, IRProgram, IRSafepoint


def enabled() -> bool:
    return os.environ.get('HORNET_SAFEPOINTS', '') not in ('', '0')


def loop_headers(blocks: list) -> set:
    """Indices of the blocks that a branch back (to a block still being searched from) reaches.
    Every cycle of the graph has such a branch."""
    headers, on_path, seen = set(), {0}, {0}
    stack = [(0, iter(blocks[0].successors))]
    while stack:
        idx, succs = stack[-1]
        nxt = next(succs, None)
        if nxt is None:
            stack.pop()
            on_path.discard(idx)
        elif nxt in on_path:
            headers.add(nxt)
        elif nxt not in seen:
            seen.add(nxt)
            on_path.add(nxt)
            stack.append((nxt, iter(blocks[nxt].successors)))
    return headers


def dominators(blocks: list) -> dict:
    """Block index -> its immediate dominator (Cooper, Harvey and Kennedy), for blocks reachable from 0."""
    order = reverse_postorder(blocks)
    position = {b: i for i, b in enumerate(order)}
    idom = {0: 0}
    changed = True
    while changed:
        changed = False
        for b in order[1:]:
            done = [p for p in blocks[b].predecessors if p in idom]
            new = done[0]
            for p in done[1:]:
                x, y = p, new
                while x != y:
                    while position[x] > position[y]:
                        x = idom[x]
                    while position[y] > position[x]:
                        y = idom[y]
                new = x
            if idom.get(b) != new:
                idom[b] = new
                changed = True
    return idom


def _always_checked(blocks: list, header: int, idom: dict, checked: set) -> bool:
    """Whether every way round the loop at `header` calls a function in `checked`: each branch back
    to it is dominated, from the header, by a block that makes such a call."""
    def calls_checked(b):
        return any(isinstance(i, IRCall) and i.name in checked for i in blocks[b].instructions)
    latches = [p for p in blocks[header].predecessors if p in idom and _dominates(idom, header, p)]
    for latch in latches:
        b = latch
        while not calls_checked(b):
            if b == header:
                return False
            b = idom[b]
    return bool(latches)


def _dominates(idom: dict, a: int, b: int) -> bool:
    while b != a:
        if b == idom[b]:
            return False
        b = idom[b]
    return True


def recursive_functions(ir_program: IRProgram) -> set:
    """Names of the functions in a cycle of calls (Tarjan's strongly connected components)."""
    names = {fn.name for fn in ir_program.functions}
    calls = {fn.name: {i.name for i in fn.body if isinstance(i, IRCall) and i.name in names}
             for fn in ir_program.functions}
    index, low, on_stack, stack, found = {}, {}, set(), [], set()
    counter = [0]
    for root in calls:
        if root in index:
            continue
        work = [(root, iter(calls[root]))]
        index[root] = low[root] = counter[0]
        counter[0] += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, callees = work[-1]
            callee = next(callees, None)
            if callee is not None:
                if callee not in index:
                    index[callee] = low[callee] = counter[0]
                    counter[0] += 1
                    stack.append(callee)
                    on_stack.add(callee)
                    work.append((callee, iter(calls[callee])))
                elif callee in on_stack:
                    low[node] = min(low[node], index[callee])
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[node])
            if low[node] == index[node]:
                component = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                if len(component) > 1 or node in calls[node]:
                    found.update(component)
    return found


def _safepoint(ir_program: IRProgram) -> IRSafepoint:
    return IRSafepoint(ir_program.ids.new_label("safepoint"), ir_program.ids.new_label("safepoint_back"))


def place_safepoints(ir_program: IRProgram) -> None:
    recursive = recursive_functions(ir_program)
    skip = os.environ.get('HORNET_SAFEPOINTS') == '2'
    for fn in ir_program.functions:
        blocks = build_blocks(fn.body)
        if not blocks:
            continue
        idom = dominators(blocks) if skip else {}
        for idx in loop_headers(blocks):
            if skip and _always_checked(blocks, idx, idom, recursive):
                continue
            label = blocks[idx].instructions[0]
            assert isinstance(label, IRLabel), label  # (a branch's target)
            blocks[idx].instructions.insert(1, _safepoint(ir_program))
        body = flatten(blocks)
        if fn.name in recursive:
            body.insert(0, _safepoint(ir_program))
        fn.body = body
