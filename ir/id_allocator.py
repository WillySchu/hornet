"""Whole-compilation counters for Temps, labels, and frame slots. Ids are unique, never reused, not dense."""

from ir.ir import IRFunction, Temp
from typesys import Type


class IdAllocator:
    def __init__(self):
        self._temp_count = 0
        self._label_count = 0
        self._slot_count = 0

    def new_label(self, prefix: str) -> str:
        """Fresh label `.L<prefix>_<n>`."""
        label = f".L{prefix}_{self._label_count}"
        self._label_count += 1
        return label

    def new_temp(self, t: Type) -> Temp:
        """Fresh Temp of type `t`; storage decided at lowering."""
        temp_id = self._temp_count
        self._temp_count += 1
        return Temp(id=temp_id, type=t)

    def temp_at_offset(self, t: Type, slot: int, ir_fn: IRFunction) -> Temp:
        """Fresh Temp homed in `slot` of `ir_fn`."""
        temp = self.new_temp(t)
        ir_fn.temp_homes[temp.id] = slot
        return temp

    def new_slot(self, width: int, label: str, ir_fn: IRFunction) -> int:
        """Fresh logical frame slot."""
        slot_id = self._slot_count
        self._slot_count += 1
        ir_fn.slot_widths[slot_id] = width
        ir_fn.slot_labels[slot_id] = label
        return slot_id
