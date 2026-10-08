"""semantic/flow.py on its own: the questions about statements that need no state, and Scopes, which
holds the names in scope and narrows sum variables by rebinding them, with no checker in sight."""

import pytest

from ops import BinaryOp, UnaryOp
from parser import Binary, IsCheck, NoneLiteral, Unary, Variable
from semantic.errors import SemanticError
from semantic.facts import Facts
from semantic.flow import Scopes, always_leaves, always_returns, conditions_after, conjuncts, contains_reachable_break
from symbols import Symbol
from tests.test_compiler import _parse
from typesys import SumTypeInfo, Type, TypeKind

SHAPE = Type(TypeKind.SUM, sum_type_name='Shape')
POINT = Type(TypeKind.STRUCT, struct_name='Point')


class _Decls:
    sum_types = {'Shape': SumTypeInfo(name='Shape', variants=[POINT, Type.INT, Type.NONE])}


def _scopes(*variables):
    """A Scopes with each (name, type) declared, ids from 1; and the facts it reads."""
    symbols = {i: Symbol(id=i, name=name, kind='local', type=t) for i, (name, t) in enumerate(variables, start=1)}
    facts = Facts()
    scopes = Scopes(symbols, _Decls, facts)
    for i, (name, t) in enumerate(variables, start=1):
        scopes.declare(name, t, None, i)
    return scopes, facts


def _body(source: str) -> list:
    return _parse("def int f(int n):\n" + source).functions[0].body


def test_names_and_scopes():
    scopes, _ = _scopes(('a', Type.INT))
    assert scopes.lookup('a') == (Type.INT, 1) and scopes.is_local('a') and not scopes.is_local('b')
    assert scopes.lookup('b') is None
    with pytest.raises(SemanticError, match="Variable 'a' is already declared in this scope"):
        scopes.declare('a', Type.STR, None, 2)
    scopes.push()
    scopes.declare('a', Type.STR, None, 2)            # an inner scope may shadow
    scopes.declare('b', Type.BOOL, None, 3)
    assert scopes.lookup('a') == (Type.STR, 2) and scopes.lookup('b') == (Type.BOOL, 3)
    scopes.pop()
    assert scopes.lookup('a') == (Type.INT, 1) and scopes.lookup('b') is None


def test_narrowing_rebinds_a_name_until_its_region_ends():
    scopes, _ = _scopes(('s', SHAPE))
    everything = frozenset([POINT, Type.INT, Type.NONE])
    assert scopes.possible_variants(1) == everything and not scopes.is_narrowed(1)
    mark = scopes.mark()
    scopes.apply({1: ('s', frozenset([POINT, Type.INT]))})         # not none: still a Shape, of two variants
    assert scopes.lookup('s') == (SHAPE, 1) and scopes.possible_variants(1) == {POINT, Type.INT}
    inner = scopes.mark()
    scopes.apply({1: ('s', frozenset([POINT, Type.NONE]))})        # and not an int either: a Point
    assert scopes.lookup('s') == (POINT, 1) and scopes.is_narrowed(1)
    scopes.end_region(inner)
    assert scopes.lookup('s') == (SHAPE, 1) and scopes.possible_variants(1) == {POINT, Type.INT}
    scopes.apply({1: ('s', frozenset([Type.NONE]))})               # nothing left it could be...
    assert scopes.possible_variants(1) == frozenset()
    scopes.end_region(mark)
    assert scopes.lookup('s') == (SHAPE, 1) and scopes.possible_variants(1) == everything and not scopes.is_narrowed(1)
    # Only `none` left has nothing to read: the variable stays its sum.
    scopes.apply({1: ('s', frozenset([Type.NONE]))})
    assert scopes.lookup('s') == (SHAPE, 1) and scopes.possible_variants(1) == {Type.NONE}


def test_assignment_and_shadowing_end_what_is_known():
    scopes, _ = _scopes(('s', SHAPE))
    scopes.apply({1: ('s', frozenset([POINT]))})
    assert scopes.lookup('s') == (POINT, 1)
    scopes.forget(1)                                               # `s = ...`
    assert scopes.lookup('s') == (SHAPE, 1) and scopes.possible_variants(1) == {POINT, Type.INT, Type.NONE}
    scopes.apply({1: ('s', frozenset([POINT]))})
    scopes.forget_assigned_in(_body("    s = 5\n    return n\n"))   # a loop whose body assigns s
    assert scopes.lookup('s') == (SHAPE, 1)
    scopes.forget_assigned_in(_body("    s += 5\n    return n\n"))  # ... `+=` is no such assignment
    scopes.apply({1: ('s', frozenset([Type.INT]))})
    scopes.forget_assigned_in(_body("    s += 5\n    return n\n"))
    assert scopes.lookup('s') == (Type.INT, 1)
    # A fact about a variable another has since shadowed doesn't touch the one in scope.
    scopes.push()
    scopes.symbols[2] = Symbol(id=2, name='s', kind='local', type=Type.STR)
    scopes.declare('s', Type.STR, None, 2)
    scopes.apply({1: ('s', frozenset([POINT]))})
    assert scopes.lookup('s') == (Type.STR, 2)


def test_what_a_condition_says_when_true_and_when_false():
    scopes, facts = _scopes(('s', SHAPE), ('n', Type.INT))

    def is_check(variant: Type) -> IsCheck:
        check = IsCheck(variable_name='s', type_name='T')
        facts.narrowed[check.nid], facts.decls[check.nid] = variant, 1       # as the checker records it
        return check
    is_point, is_int = is_check(POINT), is_check(Type.INT)
    assert scopes.when(is_point) == ({1: ('s', frozenset([POINT]))}, {1: ('s', frozenset([Type.INT, Type.NONE]))})
    negated = Unary(op=UnaryOp.NOT, operand=is_point)
    assert scopes.when(negated) == tuple(reversed(scopes.when(is_point)))
    either = Binary(op=BinaryOp.OR, left=is_point, right=is_int)
    assert scopes.when(either) == ({1: ('s', frozenset([POINT, Type.INT]))}, {1: ('s', frozenset([Type.NONE]))})
    both = Binary(op=BinaryOp.AND, left=negated, right=Unary(op=UnaryOp.NOT, operand=is_int))
    assert scopes.when(both)[0] == {1: ('s', frozenset([Type.NONE]))} and scopes.when(both)[1] == {
        1: ('s', frozenset([POINT, Type.INT]))}
    s = Variable(name='s')
    facts.decls[s.nid] = 1
    not_none = Binary(op=BinaryOp.NOT_EQUAL, left=s, right=NoneLiteral())    # `s != none` is `not (s is none)`
    assert scopes.when(not_none) == ({1: ('s', frozenset([POINT, Type.INT]))}, {1: ('s', frozenset([Type.NONE]))})
    n = Variable(name='n')
    facts.decls[n.nid] = 2
    assert scopes.when(Binary(op=BinaryOp.EQUAL, left=n, right=n)) == ({}, {})  # nothing about a sum variable


def test_the_questions_about_statements():
    assert always_returns(_body("    if n > 0:\n        return 1\n    else:\n        return 2\n"), {})
    assert not always_returns(_body("    if n > 0:\n        return 1\n    return 2\n")[:1], {})
    assert always_returns(_body("    while true:\n        n += 1\n"), {})             # never finishes
    assert not always_returns(_body("    while true:\n        break\n    return 1\n")[:1], {})
    loop = _body("    while n > 0:\n        if n == 3:\n            break\n        n -= 1\n    return n\n")[0]
    assert contains_reachable_break(loop.body) and always_leaves(loop.body[0].then_body, {})
    assert not always_leaves(loop.body, {})
    chain = _body("    if n == 1:\n        return 1\n    elif n == 2:\n        n += 1\n    else:\n        return 3\n"
                  "    return n\n")[0]
    false, true = conditions_after(chain, {})       # only the `elif` stays: it ran
    assert [c.right.value for c in false] == [1] and true.right.value == 2
    condition = _body("    if n > 0 and n < 5 and n != 3:\n        return 1\n    return 0\n")[0].condition
    assert len(conjuncts(condition)) == 3
