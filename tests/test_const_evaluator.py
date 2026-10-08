"""semantic/constants.py's ConstEvaluator on its own: it is given the facts about checked
expressions and one thing to ask (check a declaration), and needs nothing else of the analyzer."""

import pytest

from ops import BinaryOp, UnaryOp
from parser import Binary, BoolLiteral, ConstDecl, Constant, StringLiteral, Unary, Variable
from semantic.constants import ConstEvaluator
from semantic.errors import SemanticError
from semantic.facts import Facts
from typesys import Type


def _typed(facts: Facts, node, type_: Type):
    facts.types[node.nid] = type_
    return node


def _number(facts: Facts, value: int) -> Constant:
    return _typed(facts, Constant(value=value), Type.INT)


def _named(facts: Facts, key: str, type_: Type = Type.INT) -> Variable:
    """A name the checker resolved to the constant `key`."""
    node = _typed(facts, Variable(name=key), type_)
    facts.const_refs[node.nid] = key
    return node


def test_a_checked_expression_is_evaluated_from_the_facts_alone():
    facts = Facts()
    evaluator = ConstEvaluator(facts, {}, check_declaration=lambda decl: pytest.fail("no declaration is involved"))
    total = _typed(facts, Binary(op=BinaryOp.ADD, left=_number(facts, 2), right=_number(facts, 3)), Type.INT)
    assert evaluator.evaluate(total) == 5
    negated = _typed(facts, Unary(op=UnaryOp.NEGATE, operand=total), Type.INT)
    assert evaluator.evaluate(negated) == -5
    words = _typed(facts, Binary(op=BinaryOp.ADD, left=_typed(facts, StringLiteral(value='a'), Type.STR),
                                 right=_typed(facts, StringLiteral(value='b'), Type.STR)), Type.STR)
    assert evaluator.evaluate(words) == 'ab'
    before = _typed(facts, Binary(op=BinaryOp.LESS_THAN, left=words, right=words), Type.BOOL)
    assert evaluator.evaluate(before) is False
    assert evaluator.evaluate(_typed(facts, BoolLiteral(value=True), Type.BOOL)) is True
    with pytest.raises(SemanticError, match="must be built from literals, other constants"):
        evaluator.evaluate(_typed(facts, Variable(name='a_local'), Type.INT))  # a name that is no constant's
    with pytest.raises(SemanticError, match="Division by zero in a constant expression"):
        evaluator.evaluate(_typed(facts, Binary(op=BinaryOp.DIVIDE, left=_number(facts, 1), right=_number(facts, 0)),
                                  Type.INT))


def test_a_constant_is_worked_out_when_first_needed_and_only_once():
    facts, checked = Facts(), []

    def check(decl: ConstDecl) -> Type:
        checked.append(decl.name)
        return Type.INT
    evaluator = ConstEvaluator(facts, {}, check)
    # A = B + 1 is declared before B = 20, which is declared before C = B.
    a = ConstDecl(name='A', const_type='int', value=_typed(
        facts, Binary(op=BinaryOp.ADD, left=_named(facts, 'B'), right=_number(facts, 1)), Type.INT))
    b = ConstDecl(name='B', const_type='int', value=_number(facts, 20))
    c = ConstDecl(name='C', const_type='int', value=_named(facts, 'B'))
    evaluator.declare([a, b, c])
    assert evaluator.values == {} and checked == []
    assert evaluator.constant('A') == (Type.INT, 21)
    assert checked == ['A', 'B']                        # B's turn came when A needed it
    evaluator.evaluate_all()
    assert checked == ['A', 'B', 'C'] and evaluator.type_of('C') == Type.INT
    assert evaluator.values == {'A': (Type.INT, 21), 'B': (Type.INT, 20), 'C': (Type.INT, 20)}


def test_a_constant_defined_in_terms_of_itself_and_one_declared_twice():
    facts = Facts()
    evaluator = ConstEvaluator(facts, {}, lambda decl: Type.INT)
    evaluator.declare([ConstDecl(name='A', const_type='int', value=_named(facts, 'B')),
                       ConstDecl(name='B', const_type='int', value=_named(facts, 'A'))])
    with pytest.raises(SemanticError, match="Constant '[AB]' is defined in terms of itself"):
        evaluator.constant('A')
    with pytest.raises(SemanticError, match="Constant 'A' is already declared"):
        evaluator.declare([ConstDecl(name='A', const_type='int', value=_number(facts, 1))])
