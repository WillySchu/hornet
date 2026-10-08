"""Flow: what the path through a function says at each point.

Two kinds of thing. Questions about statements that need no state: whether a block always returns
or leaves, what a loop's body assigns, what a condition is a conjunction of. And Scopes: the names
in scope at a point in a function, each with its type *as known there*. An `is` check narrows a sum
variable from that point to the end of a region, which is done by rebinding its name in the scope,
and undone at the region's end; so the scopes and the narrowing are one mechanism."""

import dataclasses
from typing import Dict, List, Optional, Set, Tuple

from ops import EQUALITY_OPS, LOGICAL_OPS, BinaryOp, UnaryOp
from parser import (
    Assign, Binary, BoolLiteral, Break, Continue, ExprStmt, If, IsCheck, Match, Node, NoneLiteral, Return, Unary,
    Variable, While,
)
from scopes import display_name as shown
from semantic.errors import SemanticError
from typesys import Type, TypeKind


def always_ends(statements: List[Node], types: dict, ends: tuple) -> bool:
    """Whether control never runs off the end of `statements`: every path reaches a statement of
    one of the classes `ends`, a call to a `never` function (by `types`, the checked types by node
    number), or a `while true` without a reachable break."""
    for stmt in statements:
        if isinstance(stmt, ends):
            return True
        if isinstance(stmt, ExprStmt) and types.get(stmt.expr.nid) == Type.NEVER:
            return True
        if isinstance(stmt, Match) and all(always_ends(body, types, ends) for _, body in stmt.arms) and (
                stmt.else_body is None or always_ends(stmt.else_body, types, ends)):  # exhaustive without an else
            return True
        if isinstance(stmt, If) and stmt.else_body is not None and always_ends(stmt.then_body, types, ends) and \
                always_ends(stmt.else_body, types, ends):
            return True
        if isinstance(stmt, While) and isinstance(stmt.condition, BoolLiteral) and stmt.condition.value is True \
                and not contains_reachable_break(stmt.body):
            return True
    return False


def always_returns(statements: List[Node], types: dict) -> bool:
    """Whether every path through `statements` returns (or never finishes)."""
    return always_ends(statements, types, (Return,))


def contains_reachable_break(statements: List[Node]) -> bool:
    """Whether `statements` contain a break for this loop (not nested loops)."""
    for stmt in statements:
        if isinstance(stmt, Break):
            return True
        if isinstance(stmt, If):
            if contains_reachable_break(stmt.then_body):
                return True
            if stmt.else_body is not None and contains_reachable_break(stmt.else_body):
                return True
        if isinstance(stmt, Match):
            if any(contains_reachable_break(body) for _, body in stmt.arms):
                return True
            if stmt.else_body is not None and contains_reachable_break(stmt.else_body):
                return True
    return False


def always_leaves(statements: List[Node], types: dict) -> bool:
    """Whether every path through `statements` returns, breaks out of or continues the enclosing
    loop, or never finishes."""
    return always_ends(statements, types, (Return, Break, Continue))


def conditions_after(stmt: If, types: dict) -> Tuple[List[Node], Optional[Node]]:
    """(the conditions known false, the condition known true) once control passes the `if`/`elif`/
    `else` chain `stmt`. When every branch but one always leaves, that one ran: the conditions before
    it were false, and its own was true (it has none if it is the `else`, written or not)."""
    branches = []  # (condition, body); the `else` last, with no condition
    while True:
        branches.append((stmt.condition, stmt.then_body))
        if stmt.else_body is not None and len(stmt.else_body) == 1 and isinstance(stmt.else_body[0], If):
            stmt = stmt.else_body[0]
            continue
        branches.append((None, stmt.else_body or []))
        break
    staying = [i for i, (_, body) in enumerate(branches) if not always_leaves(body, types)]
    if len(staying) != 1:
        return [], None
    return [condition for condition, _ in branches[:staying[0]]], branches[staying[0]][0]


def _assigned_names(node) -> Set[str]:
    """The names of the variables given a new value (`v = ...`) anywhere under `node` (a statement, or a
    list of them). `v += ...` isn't one: it gives v back the variant it held."""
    names: Set[str] = set()
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, (list, tuple)):
            stack.extend(n)
        elif isinstance(n, Node):
            if isinstance(n, Assign) and isinstance(n.target, Variable) and n.op is None:
                names.add(n.target.name)
            stack.extend(getattr(n, f.name) for f in dataclasses.fields(n))
    return names


def conjuncts(condition: Node) -> List[Node]:
    """The checks a condition's top-level `and`s join, in order (the condition itself if it has none)."""
    if isinstance(condition, Binary) and condition.op == BinaryOp.AND:
        return conjuncts(condition.left) + conjuncts(condition.right)
    return [condition]


def _both(a: dict, b: dict) -> dict:
    """What is known when both of two sets of narrowing facts hold (Scopes.when)."""
    out = dict(a)
    for decl_id, (name, variants) in b.items():
        out[decl_id] = (name, variants & out[decl_id][1]) if decl_id in out else (name, variants)
    return out


def _either(a: dict, b: dict) -> dict:
    """What is known when one of two sets of narrowing facts holds, but not which."""
    return {decl_id: (name, variants | b[decl_id][1]) for decl_id, (name, variants) in a.items() if decl_id in b}


class Scopes:
    """The names in scope at a point in a function: name -> (type as known here, declaration id),
    innermost scope last. A sum variable that `is` checks have narrowed is rebound, in the scope the
    check is in, to the variant it must hold (restrict), until the end of that region (end_region)."""

    def __init__(self, symbols, decls, facts):
        self.symbols = symbols  # declaration id -> Symbol
        self.decls = decls
        self.facts = facts
        self._scopes: List[Dict[str, Tuple[Type, object]]] = [{}]
        self._declared: List[Set[str]] = [set()]  # per scope: the names declared in it (not merely narrowed)
        # Sum variables an `is` check has narrowed here: decl id -> the variants it may still hold.
        self._possible: Dict[int, frozenset] = {}
        self._undo: list = []  # what _restrict changed, undone by end_region

    # -- names

    def lookup(self, name: str) -> Optional[Tuple[Type, object]]:
        """(type, decl id) of the local `name`, innermost first; None if no local has the name."""
        for scope in reversed(self._scopes):
            if name in scope:
                return scope[name]
        return None

    def is_local(self, name: str) -> bool:
        return self.lookup(name) is not None

    def push(self) -> None:
        self._scopes.append({})
        self._declared.append(set())

    def pop(self) -> None:
        self._scopes.pop()
        self._declared.pop()

    def declare(self, name: str, type_: Type, node: Optional[Node], decl_id) -> None:
        """Declare in the innermost scope; shadowing outer scopes is allowed.
        decl_id is the declaration's Symbol.id."""
        if name in self._declared[-1]:
            raise SemanticError(f"Variable '{shown(name)}' is already declared in this scope", node)
        self._declared[-1].add(name)
        self._scopes[-1][name] = (type_, decl_id)

    # -- narrowing

    def mark(self) -> int:
        """Where a region starts, for end_region."""
        return len(self._undo)

    def is_narrowed(self, decl_id) -> bool:
        return decl_id in self._possible

    def possible_variants(self, decl_id) -> frozenset:
        """The variants the sum variable declared as `decl_id` may hold here."""
        if decl_id in self._possible:
            return self._possible[decl_id]
        return frozenset(self.decls.sum_types[self.symbols[decl_id].type.sum_type_name].variants)

    def _restrict(self, name: str, decl_id, possible: frozenset) -> None:
        """From here to the end of the region (_end_region), sum variable `name` holds one of
        `possible`: with one variant left it has that variant's type (`none` has nothing to read,
        so the variable stays its sum). Assigning to it ends that (_forget)."""
        narrowed = len(possible) == 1 and Type.NONE not in possible
        entry = (next(iter(possible)) if narrowed else self.symbols[decl_id].type, decl_id)
        scope = self._scopes[-1]
        self._undo.append((scope, name, scope.get(name), entry, decl_id, self._possible.get(decl_id)))
        scope[name] = entry
        self._possible[decl_id] = possible

    def end_region(self, mark: int) -> None:
        """Undo every _restrict since `mark` (a length of self._undo)."""
        while len(self._undo) > mark:
            scope, name, previous, entry, decl_id, possible = self._undo.pop()
            if scope.get(name) is entry:  # not since replaced by a declaration of the same name
                if previous is None:
                    del scope[name]
                else:
                    scope[name] = previous
            if possible is None:
                del self._possible[decl_id]
            else:
                self._possible[decl_id] = possible

    def when(self, condition: Node) -> Tuple[dict, dict]:
        """What `condition` (already checked) says about sum variables when it is true, and when it is
        false: each a dict, decl id -> (name, the variants the variable may then hold). `x is T` (and
        `x == none`) says so directly; `and`, `or`, and `not` combine what their operands say."""
        if isinstance(condition, Unary) and condition.op == UnaryOp.NOT:
            when_true, when_false = self.when(condition.operand)
            return when_false, when_true
        if isinstance(condition, Binary) and condition.op in LOGICAL_OPS:
            left_true, left_false = self.when(condition.left)
            right_true, right_false = self.when(condition.right)
            if condition.op == BinaryOp.AND:
                return _both(left_true, right_true), _either(left_false, right_false)
            return _either(left_true, right_true), _both(left_false, right_false)
        name = decl_id = variant = None
        if isinstance(condition, IsCheck) and condition.nid in self.facts.narrowed:
            name, decl_id, variant = condition.variable_name, self.facts.decls.get(condition.nid), \
                self.facts.narrowed[condition.nid]
        elif isinstance(condition, Binary) and condition.op in EQUALITY_OPS:
            for side, other in ((condition.left, condition.right), (condition.right, condition.left)):
                if isinstance(side, Variable) and isinstance(other, NoneLiteral):  # `x == none` is `x is none`
                    name, decl_id, variant = side.name, self.facts.decls.get(side.nid), Type.NONE
        if decl_id is None or self.symbols[decl_id].type.kind != TypeKind.SUM:
            return {}, {}
        variants = frozenset(self.decls.sum_types[self.symbols[decl_id].type.sum_type_name].variants)
        holds, excluded = {decl_id: (name, frozenset({variant}))}, {decl_id: (name, variants - {variant})}
        if isinstance(condition, Binary) and condition.op == BinaryOp.NOT_EQUAL:
            return excluded, holds
        return holds, excluded

    def apply(self, known: Optional[dict]) -> None:
        """Narrow by `known` (one of _when's results) until the end of the region."""
        for decl_id, (name, variants) in (known or {}).items():
            # An `as NAME` binding ends with its `if`, so the name may be gone, or another variable's.
            entry = self.lookup(name)
            if entry is not None and entry[1] == decl_id:
                self._restrict(name, decl_id, self.possible_variants(decl_id) & variants)

    def forget(self, decl_id) -> None:
        """The variable declared as `decl_id` has been assigned: nothing is known about it from here
        to the end of the region, whatever was known before."""
        name = self.symbols[decl_id].name
        entry = self.lookup(name)
        if decl_id in self._possible and entry is not None and entry[1] == decl_id:
            variants = self.decls.sum_types[self.symbols[decl_id].type.sum_type_name].variants
            self._restrict(name, decl_id, frozenset(variants))

    def forget_assigned_in(self, loop_syntax) -> None:
        """Before a loop: forget what is known about every variable its body assigns, since the body
        may already have run when its condition and its statements are reached."""
        for name in _assigned_names(loop_syntax):
            entry = self.lookup(name)
            if entry is not None and entry[1] is not None:
                self.forget(entry[1])
