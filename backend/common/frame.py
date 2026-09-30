"""A function's stack frame as the backend sees it: the IR's slots plus the backend's own
(outgoing call arguments, spills), laid out below a register save area."""

from ir.ir import IRFunction, Temp
from typesys import type_byte_width


class Frame:
    def __init__(self, ir_fn: IRFunction, struct_registry: dict, sum_type_registry: dict):
        self.slots: dict = dict(ir_fn.slot_widths)  # slot key -> width, in creation order
        self.spills: dict = {}  # temp id -> slot key
        self.outgoing = None  # slot for arguments passed on the stack, placed at the bottom
        self.offsets: dict = {}  # slot key -> offset from the frame pointer, after layout()
        self.size = 0  # bytes below the save area, after layout()
        self._structs = struct_registry
        self._sums = sum_type_registry

    def new_slot(self, width: int):
        """A backend-owned slot; its key never collides with the IR's integer slot ids."""
        key = ('backend', len(self.slots))
        self.slots[key] = width
        return key

    def reserve_outgoing(self, width: int) -> None:
        if width > 0:
            self.outgoing = self.new_slot(width)

    def spill_slot(self, temp: Temp):
        """The slot holding `temp` when it has no register."""
        if temp.id not in self.spills:
            self.spills[temp.id] = self.new_slot(type_byte_width(temp.type, self._structs, self._sums))
        return self.spills[temp.id]

    def layout(self, save_area: int, alignment: int = 16, used=None) -> None:
        """Offsets (negative, from the frame pointer) for every slot below `save_area` bytes of saved
        registers, with the outgoing-argument slot at the very bottom; save_area + size is a
        multiple of `alignment`. With `used`, only those slots get space: optimization often
        leaves slots nothing refers to."""
        next_offset = -save_area
        for key, width in self.slots.items():
            if key == self.outgoing or (used is not None and key not in used):
                continue
            next_offset -= width
            self.offsets[key] = next_offset
        used = -next_offset
        if self.outgoing is not None:
            used += self.slots[self.outgoing]
        self.size = (used + alignment - 1) // alignment * alignment - save_area
        if self.outgoing is not None:
            self.offsets[self.outgoing] = -(save_area + self.size)
