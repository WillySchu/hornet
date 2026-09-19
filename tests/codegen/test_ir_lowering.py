"""Tests for ir_lowering.py's InstructionSelector -- currently just
the IRSliceGrow write-back ordering bug documented in codegen/
register_allocator.py's own ALLOCATABLE_REGISTERS comment. Everything
else here is otherwise only exercised indirectly, through the full
compile-and-run integration tests in tests/test_compiler.py."""

from codegen.codegen import CodeGenerator
from codegen.assembly_ast import Mov, MovQ, Register
from codegen.id_allocator import IdAllocator
from codegen.ir_lowering import InstructionSelector
from ir.ir import IRFunction, IRProgram, IRSliceGrow, Temp
from semantic import Type


def _selector_with(register_assignment: dict) -> InstructionSelector:
    """A minimal, otherwise-empty host/InstructionSelector pair --
    every Temp involved in the scenarios below is given a real
    register assignment, so lowering never needs to fall back to
    _temp_slots (the only other per-Temp state InstructionSelector
    would otherwise need). ir_program/ids still has to be real,
    though: the growth-policy arithmetic _gen_slice_grow_into always
    runs first (see IRSliceGrow's own lowering) and mints a couple of
    local labels for it regardless of which registers anything ends
    up assigned to."""
    host = CodeGenerator()
    host.ir_program = IRProgram(ids=IdAllocator())
    host._register_assignment = register_assignment
    return InstructionSelector(host, IRFunction(name='f'))


def _write_index(out: list, src: str, dst: str) -> int:
    """The index of the (Mov or MovQ) instruction moving `src` into
    `dst`, or -1 if there isn't one -- used below to check WHICH of
    two writes came first, not just that both eventually happened."""
    for i, instr in enumerate(out):
        if isinstance(instr, (Mov, MovQ)) and instr.src == Register(src) and instr.dst == Register(dst):
            return i
    return -1


def test_slice_grow_dst_cap_before_dst_ptr_when_dst_ptr_lands_on_r13d():
    """The actual bug, reproduced directly: dst_ptr and dst_cap are
    written out of the SAME fixed scratch registers (%ebx/%r13d)
    IRSliceGrow's own lowering just used internally for ptr/cap. If
    the general allocator assigns dst_ptr itself to %r13d, writing it
    out FIRST (the naive, original order) overwrites the not-yet-read
    new_cap value still sitting there with the new pointer -- dst_cap
    then reads back a corrupted (truncated-pointer) value instead of
    the real capacity. Confirmed end-to-end against benchmarks/
    programs/copy_heavy.ht (a real, reproducible segfault) before this
    fix; see register_allocator.py's own ALLOCATABLE_REGISTERS
    comment for the full incident."""
    ptr, length, cap = Temp(id=0, type=Type.INT64), Temp(id=1, type=Type.INT), Temp(id=2, type=Type.INT)
    dst_ptr, dst_cap = Temp(id=3, type=Type.INT64), Temp(id=4, type=Type.INT)
    selector = _selector_with({
        ptr.id: 'r10d', length.id: 'r11d', cap.id: 'r15d',
        dst_ptr.id: 'r13d', dst_cap.id: 'r12d',
    })
    instr = IRSliceGrow(dst_ptr=dst_ptr, dst_cap=dst_cap, ptr=ptr, length=length, cap=cap, element_width=4)
    out = selector.lower_ir([instr])

    cap_write = _write_index(out, 'r13d', 'r12d')  # dst_cap <- still-valid new_cap
    ptr_write = _write_index(out, 'rbx', 'r13')     # dst_ptr <- new ptr (widened, since INT64)
    assert cap_write != -1 and ptr_write != -1
    assert cap_write < ptr_write


def test_slice_grow_original_order_is_fine_when_dst_cap_lands_on_ebx():
    """The mirror-image assignment (dst_cap, not dst_ptr, landing on
    the OTHER fixed register) never needed reordering in the first
    place: a move only READS its source, so writing dst_ptr out of
    %ebx first can't destroy the value dst_cap's own later write puts
    there. Pinned here so this case is never "fixed" by accident into
    doing unnecessary work."""
    ptr, length, cap = Temp(id=0, type=Type.INT64), Temp(id=1, type=Type.INT), Temp(id=2, type=Type.INT)
    dst_ptr, dst_cap = Temp(id=3, type=Type.INT64), Temp(id=4, type=Type.INT)
    selector = _selector_with({
        ptr.id: 'r10d', length.id: 'r11d', cap.id: 'r15d',
        dst_ptr.id: 'r14d', dst_cap.id: 'ebx',
    })
    instr = IRSliceGrow(dst_ptr=dst_ptr, dst_cap=dst_cap, ptr=ptr, length=length, cap=cap, element_width=4)
    out = selector.lower_ir([instr])

    ptr_write = _write_index(out, 'rbx', 'r14')     # dst_ptr <- new ptr, reads %ebx (non-destructively)
    cap_write = _write_index(out, 'r13d', 'ebx')    # dst_cap <- new_cap, written into %ebx afterward
    assert ptr_write != -1 and cap_write != -1
    assert ptr_write < cap_write


def test_slice_grow_true_swap_uses_r12_as_a_temporary():
    """The genuine cycle: dst_ptr assigned %r13d AND dst_cap assigned
    %ebx at once -- each destination is the OTHER write's still-needed
    source, so no ordering alone can avoid clobbering one of them.
    %r12 is always free to use as a temporary here regardless (length
    is already dead the moment the call returns -- see _gen_slice_
    grow_into's own docstring)."""
    ptr, length, cap = Temp(id=0, type=Type.INT64), Temp(id=1, type=Type.INT), Temp(id=2, type=Type.INT)
    dst_ptr, dst_cap = Temp(id=3, type=Type.INT64), Temp(id=4, type=Type.INT)
    selector = _selector_with({
        ptr.id: 'r10d', length.id: 'r11d', cap.id: 'r15d',
        dst_ptr.id: 'r13d', dst_cap.id: 'ebx',
    })
    instr = IRSliceGrow(dst_ptr=dst_ptr, dst_cap=dst_cap, ptr=ptr, length=length, cap=cap, element_width=4)
    out = selector.lower_ir([instr])

    stash = _write_index(out, 'rbx', 'r12')          # new ptr stashed out of %ebx first
    cap_write = _write_index(out, 'r13d', 'ebx')     # dst_cap <- new_cap (still valid; %ebx not yet touched again)
    ptr_write = _write_index(out, 'r12', 'r13')      # dst_ptr <- the stashed ptr (already widened), not %ebx (already overwritten)
    assert stash != -1 and cap_write != -1 and ptr_write != -1
    assert stash < cap_write < ptr_write
