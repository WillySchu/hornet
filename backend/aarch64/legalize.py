"""Resolve frame slots to memory operands once the frame is laid out. A load or store reaches
-256..255 bytes from the frame pointer directly; farther slots are addressed through x17."""

from backend.aarch64.assembly import AddrOf, FrameSlot, Imm, Instr, Mem, Reg, Shift, FP, SP
from backend.aarch64.calling_convention import SCRATCH_ADDRESS


def materialize(dst: Reg, value: int) -> list:
    """movz/movk sequence for a non-negative `value`."""
    out, first = [], True
    for i in range(4):
        chunk = (value >> (16 * i)) & 0xFFFF
        if chunk or (first and i == 3):
            out.append(Instr('movz' if first else 'movk', (dst, Imm(chunk), Shift('lsl', 16 * i))))
            first = False
    return out


def _address_below_fp(dst: Reg, distance: int) -> list:
    """dst = x29 - distance (distance >= 0)."""
    if distance <= 4095:
        return [Instr('sub', (dst, FP, Imm(distance)))]
    return materialize(SCRATCH_ADDRESS, distance) + [Instr('sub', (dst, FP, SCRATCH_ADDRESS))]


def legalize(instrs: list, frame) -> list:
    out = []
    for instr in instrs:
        if isinstance(instr, AddrOf):
            offset = frame.offsets[instr.slot.slot] + instr.slot.extra
            out.extend(_address_below_fp(instr.dst, -offset))
            continue
        if isinstance(instr, Instr) and any(isinstance(o, FrameSlot) for o in instr.operands):
            operands = []
            for o in instr.operands:
                if isinstance(o, FrameSlot):
                    if o.slot == frame.outgoing:
                        o = Mem(SP, o.extra)  # outgoing arguments sit at the bottom of the frame
                    else:
                        offset = frame.offsets[o.slot] + o.extra
                        if -256 <= offset <= 255:
                            o = Mem(FP, offset)
                        else:
                            out.extend(_address_below_fp(SCRATCH_ADDRESS, -offset))
                            o = Mem(SCRATCH_ADDRESS)
                operands.append(o)
            instr = Instr(instr.mnemonic, tuple(operands))
        out.append(instr)
    return out
