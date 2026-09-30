"""backend/common/jumps.py, with a toy instruction spelling."""

from backend.common.jumps import drop_jumps_to_next, invert_branches


class Toy:
    def label(self, i):
        return i[1] if i[0] == 'label' else None

    def jump(self, i):
        return i[1] if i[0] == 'jump' else None

    def branch(self, i):
        return (i[1], i[2]) if i[0] == 'branch' else None

    def make_branch(self, cond, target):
        return ('branch', cond, target)

    def invert(self, cond):
        return 'not ' + cond


def test_branch_over_a_jump_is_inverted():
    instrs = [('branch', 'lt', 'A'), ('jump', 'B'), ('label', 'A')]
    assert invert_branches(instrs, Toy()) == [('branch', 'not lt', 'B'), ('label', 'A')]
    kept = [('branch', 'lt', 'A'), ('jump', 'B'), ('label', 'C')]
    assert invert_branches(kept, Toy()) == kept


def test_jump_to_the_next_label_is_dropped():
    assert drop_jumps_to_next([('jump', 'A'), ('label', 'A')], Toy()) == [('label', 'A')]
    assert drop_jumps_to_next([('jump', 'B'), ('label', 'A')], Toy()) == [('jump', 'B'), ('label', 'A')]
