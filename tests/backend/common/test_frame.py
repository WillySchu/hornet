"""backend/common/frame.py."""

from backend.common.frame import Frame
from ir.ir import IRFunction, Temp
from typesys import Type


def _frame(**slots) -> Frame:
    fn = IRFunction(name='f')
    fn.slot_widths = dict(slots)
    return Frame(fn, {}, {})


def test_layout_places_slots_below_the_save_area_in_creation_order():
    f = _frame(**{'1': 8, '2': 24})
    spill = f.spill_slot(Temp(7, Type.INT32))
    f.layout(save_area=16)
    assert f.offsets == {'1': -24, '2': -48, spill: -52}
    assert (16 + f.size) % 16 == 0 and f.size >= 52 - 16


def test_outgoing_arguments_sit_at_the_bottom():
    f = _frame(**{'1': 8})
    f.reserve_outgoing(16)
    f.layout(save_area=8)
    assert f.offsets['1'] == -16
    assert f.offsets[f.outgoing] == -(8 + f.size)
    assert (8 + f.size) % 16 == 0


def test_backend_slots_never_collide_with_ir_slots_and_spills_are_reused():
    f = _frame(**{'0': 8})
    a, b = f.spill_slot(Temp(1, Type.INT)), f.spill_slot(Temp(2, Type.INT))
    assert a != b and a not in ('0',) and f.spill_slot(Temp(1, Type.INT)) == a
    f.reserve_outgoing(0)
    assert f.outgoing is None


def test_layout_skips_slots_nothing_refers_to():
    f = _frame(**{'1': 8, '2': 24, '3': 8})
    f.layout(save_area=0, used={'3'})
    assert f.offsets == {'3': -8}
    assert f.size == 16
    f = _frame(**{'1': 8})
    f.layout(save_area=16, used=set())
    assert f.offsets == {} and f.size == 0
