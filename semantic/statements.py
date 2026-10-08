"""Statements: a function's body, statement by statement.

StatementChecker checks each function: its parameters, then its statements in order, keeping where
checking is (Context) up to date as blocks open and close and as conditions narrow what is known.
The expressions in them are ExpressionChecker's; nothing there calls back here."""

from typing import Dict, List, Optional

from parser import (
    ArrayLiteral, Assign, Binary, Break, Call, Continue, DictLiteral, ExprStmt, Field, For, ForIn, Function, If, Index,
    IsCheck, Match, Node, Return, Slice, SliceLiteral, StringLiteral, VarDecl, Variable, While,
)
from scopes import display_name as shown
from semantic.context import Context
from semantic.errors import SemanticError
from semantic.flow import (
    Scopes, always_ends, always_leaves, always_returns, conditions_after, conjuncts, contains_reachable_break,
)
from typesys import Type, TypeKind


_MAIN_PARAMS = ([], [Type.INT, Type(TypeKind.POINTER, element_type=Type.UINT8)])

# Types the C runtime can read as main's int exit status.
_MAIN_RETURN_TYPES = (Type.INT, Type.INT32, Type.INT8, Type.UINT8, Type.BOOL)


def _check_main_signature(fn, param_types: list, return_type: Type) -> None:
    """`main` is called by the C runtime: it returns the exit status (normally `int`) and takes no
    parameters or `(int argc, *byte argv)`."""
    if return_type not in _MAIN_RETURN_TYPES:
        raise SemanticError(
            f"'main' must return int (the program's exit status), not "
            f"{'nothing' if return_type == Type.VOID else return_type} -- declare it 'def int main()'", fn)
    if param_types not in _MAIN_PARAMS:
        raise SemanticError(
            "'main' takes no parameters, or exactly '(int argc, *byte argv)'", fn.params[0] if fn.params else fn)


class StatementChecker:
    """Checks functions, one at a time (analyze_function)."""

    def __init__(self, context: Context, decls, facts, symbols, types, expressions, module_set):
        self.context = context
        self.decls = decls
        self.facts = facts
        self.symbols = symbols
        self.types = types
        self.expressions = expressions
        self.module_set = module_set
        self.loop_depth = 0  # enclosing loop count
        self._assignments: list = []  # decl ids, as variables are assigned on paths that reach what follows

    def _enter(self, decl: Node) -> None:
        """Check `decl` (a top-level declaration, under its key) in its own file's scope."""
        self.context.scope = self.module_set.scope_of[decl.nid]

    def _type(self, type_expr, node: Node, sums: bool = True) -> Type:
        return self.types.resolve(type_expr, node, self.context.scope, sums)

    def _return_type(self, decl) -> Type:
        return self.types.return_type(decl, self.context.scope)

    def analyze_function(self, fn: Function) -> None:
        self._enter(fn)
        self.context.scopes = Scopes(self.symbols, self.decls, self.facts)
        self.context.bindable, self.context.bound = set(), set()
        self._assignments = []  # decl ids, as variables are assigned on paths that reach what follows
        self.loop_depth = 0
        # Params are locals; _declare also catches duplicates.
        for p in fn.params:
            param_type = self._type(p.type, p)
            self.facts.types[p.nid] = param_type
            self.facts.symbols[p.nid] = self.symbols.new(p.name, 'param', param_type, p)
            self.context.scopes.declare(p.name, param_type, p, self.facts.symbols[p.nid].id)
        return_type = self._return_type(fn)
        self.facts.returns[fn.nid] = return_type
        self._analyze_block(fn.body, return_type)
        if return_type == Type.NEVER and not always_ends(fn.body, self.facts.types, ()):
            raise SemanticError(
                f"Function '{shown(fn.name)}' is declared never, but can finish -- every path must end in a call "
                f"to a never function (such as panic) or in a `while true` it doesn't break out of", fn)
        # Void functions may fall off the end.
        if return_type not in (Type.VOID, Type.NEVER) and not always_returns(fn.body, self.facts.types):
            raise SemanticError(
                f"Function '{shown(fn.name)}' (declared to return {return_type}) "
                f"does not return a value on all code paths",
                fn,
            )
        if fn.name == 'main':  # (after its body: what is wrong in there is said first)
            _check_main_signature(fn, *self.decls.functions['main'])

    def _analyze_block(self, statements: List[Node], return_type: Type) -> None:
        """Check a block's statements in the current scope. What an `if` or a `while` establishes for
        the code after it narrows the rest of the block; what a statement assigns is forgotten."""
        mark = self.context.scopes.mark()
        for stmt in statements:
            first = len(self._assignments)
            self.analyze_statement(stmt, return_type)
            assigned = set(self._assignments[first:])  # here, or in nested blocks that reach what follows
            if isinstance(stmt, If):
                known_false, known_true = conditions_after(stmt, self.facts.types)
                for condition in known_false:
                    self.context.scopes.apply(self.context.scopes.when(condition)[1])
                if known_true is not None:
                    self.context.scopes.apply(self.context.scopes.when(known_true)[0])
            for decl_id in assigned:
                self.context.scopes.forget(decl_id)
            if isinstance(stmt, While) and not contains_reachable_break(stmt.body):
                # The loop ended because its condition was false.
                self.context.scopes.apply(self.context.scopes.when(stmt.condition)[1])
        self.context.scopes.end_region(mark)

    def _analyze_body(self, statements: List[Node], return_type: Type, known: Optional[dict] = None) -> None:
        """A block in a scope of its own, narrowed by `known` (one of _when's results)."""
        mark, first = self.context.scopes.mark(), len(self._assignments)
        self.context.scopes.push()
        self.context.scopes.apply(known)
        self._analyze_block(statements, return_type)
        self.context.scopes.pop()
        self.context.scopes.end_region(mark)
        if always_leaves(statements, self.facts.types):
            del self._assignments[first:]  # what follows the enclosing statement isn't reached from here

    def analyze_statement(self, stmt: Node, return_type: Type) -> None:
        if isinstance(stmt, VarDecl):
            self.analyze_var_decl(stmt)
        elif isinstance(stmt, Assign):
            self.analyze_assign(stmt)
        elif isinstance(stmt, Return):
            self.analyze_return(stmt, return_type)
        elif isinstance(stmt, If):
            self.analyze_if(stmt, return_type)
        elif isinstance(stmt, Match):
            self.analyze_match(stmt, return_type)
        elif isinstance(stmt, While):
            self.analyze_while(stmt, return_type)
        elif isinstance(stmt, For):
            self.analyze_for(stmt, return_type)
        elif isinstance(stmt, ForIn):
            self.analyze_for_in(stmt, return_type)
        elif isinstance(stmt, Break):
            self.analyze_break(stmt)
        elif isinstance(stmt, Continue):
            self.analyze_continue(stmt)
        elif isinstance(stmt, ExprStmt):
            self.expressions.check_expr_allowing_struct_literal(stmt.expr)
        else:
            raise SemanticError(f"No semantic rule for statement: {stmt!r}", stmt)

    def analyze_var_decl(self, stmt: VarDecl) -> None:
        declared_type = self._type(stmt.var_type, stmt)
        missing = self.expressions.without_zero_value(declared_type) if stmt.init is None else None
        if missing is not None:
            # Only a sum with a `none` variant has a zero value (that variant).
            what = f"{missing} has" if missing == declared_type else f"{declared_type} contains {missing}, which has"
            raise SemanticError(
                f"'{stmt.name}' (declared {declared_type}) has no initializer -- {what} no zero value, "
                f"so one is required here (a sum type's zero value is its `none` variant, if it has one)",
                stmt,
            )
        if stmt.init is not None:
            # Checked before declaring, so `int a = a` fails.
            init_type = self.expressions.check_value_flowing_into_allowing_struct_literal(stmt.init, declared_type)
            if not self.expressions.types_compatible(init_type, declared_type):
                raise SemanticError(
                    f"Cannot initialize '{stmt.name}' (declared {declared_type}) "
                    f"with a value of type {init_type}",
                    stmt,
                )
        self.facts.types[stmt.nid] = declared_type
        self.facts.symbols[stmt.nid] = self.symbols.new(stmt.name, 'local', declared_type, stmt)
        self.context.scopes.declare(stmt.name, declared_type, stmt, self.facts.symbols[stmt.nid].id)

    def analyze_assign(self, stmt: Assign) -> None:
        """`target = value` or `target op= value`, to a name, field, element, or pointee."""
        target = stmt.target
        if isinstance(target, Variable):
            if self.expressions.const_key(target.name) is not None and not self.context.scopes.is_local(target.name):
                raise SemanticError(f"Cannot assign to constant '{target.name}'", stmt)
            _, decl_id = self.expressions.resolve(target.name, stmt)
            self.facts.decls[target.nid] = decl_id
            target_type = self.symbols[decl_id].type if decl_id is not None \
                else self.expressions.lookup(target.name, stmt)
            what = f"to '{target.name}' (declared {target_type})"
        elif isinstance(target, Index):
            target_type = self.expressions.check_indexable_and_index(target.array, target.index)
            what = f"to an array element of type {target_type}"
        elif isinstance(target, Field):
            if target.nid in self.module_set.qualified:
                raise SemanticError(f"Cannot assign to constant '{target.name}'", stmt)
            target_type = self.expressions.check_struct_and_field(target.base, target.name)
            what = f"to field '{target.name}' of type {target_type}"
        else:
            pointer_type = self.expressions.check_expr(target.operand)
            if pointer_type.kind != TypeKind.POINTER:
                raise SemanticError(
                    f"Cannot dereference a value of type {pointer_type} for assignment -- '*' requires a pointer "
                    f"operand",
                    target.operand,
                )
            target_type = pointer_type.element_type
            what = f"through a pointer to {target_type}"
        self.facts.types[target.nid] = target_type
        if stmt.op is not None:
            # Checked as the operation it performs: `target op value`.
            operation = Binary(op=stmt.op, left=target, right=stmt.value, line=stmt.line, col=stmt.col, file=stmt.file)
            self.expressions.check_binary(operation)
            return
        value_type = self.expressions.check_value_flowing_into_allowing_struct_literal(stmt.value, target_type)
        if not self.expressions.types_compatible(value_type, target_type):
            raise SemanticError(f"Cannot assign a value of type {value_type} {what}", stmt)
        if isinstance(target, Variable) and self.facts.decls[target.nid] is not None:
            self._assignments.append(self.facts.decls[target.nid])  # the value was read as narrowed; no longer
            self.context.scopes.forget(self.facts.decls[target.nid])

    def analyze_return(self, stmt: Return, return_type: Type) -> None:
        """`return [expr]`; bare return only in void functions."""
        if return_type == Type.NEVER:
            raise SemanticError("Function is declared never, so it can't return", stmt)
        if stmt.value is None:
            if return_type != Type.VOID:
                raise SemanticError(
                    f"Function is declared to return {return_type}, but "
                    f"this bare 'return' returns nothing",
                    stmt,
                )
            return
        value_type = self.expressions.check_value_flowing_into_allowing_struct_literal(stmt.value, return_type)
        if return_type == Type.VOID:
            raise SemanticError(
                f"Function has no declared return type and cannot "
                f"return a value (got {value_type}) -- use a bare "
                f"'return' instead",
                stmt,
            )
        if not self.expressions.types_compatible(value_type, return_type):
            raise SemanticError(
                f"Function is declared to return {return_type}, but this "
                f"'return' statement returns {value_type}",
                stmt,
            )

    def analyze_if(self, stmt: If, return_type: Type) -> None:
        # `EXPR is T as NAME` binds NAME when it is the condition, or one of the checks its `and`s join
        # (check_is_check declares it, in evaluation order). A scope of their own holds the names.
        binders = [c for c in conjuncts(stmt.condition) if isinstance(c, IsCheck) and c.binds]
        whole = isinstance(stmt.condition, IsCheck) and stmt.condition.binds
        if binders:
            self.context.scopes.push()
            self.context.bindable.update(c.nid for c in binders)

        condition_type = self.expressions.check_expr(stmt.condition)
        if condition_type != Type.BOOL:
            raise SemanticError(
                f"'if' condition must be bool, got {condition_type} "
                f"(no implicit int-to-bool conversion -- try `x != 0` "
                f"instead of `x`)",
                stmt.condition,
            )

        # What the condition says when true narrows then_body, and what it says when false narrows
        # else_body (and what follows the `if`, when its other branches always leave: _analyze_block).
        when_true, when_false = self.context.scopes.when(stmt.condition)
        self._analyze_body(stmt.then_body, return_type, when_true)
        if binders and not whole:
            self.context.scopes.pop()  # in the `else`, a check after an `and` may never have run: its name is unset
        if stmt.else_body is not None:
            self._analyze_body(stmt.else_body, return_type, when_false)
        if whole:
            self.context.scopes.pop()

    def analyze_match(self, stmt: 'Match', return_type: Type) -> None:
        """Each arm narrows the subject within its own body, like `if NAME is T:` (the order of checks,
        and so of errors, is that of the equivalent `if`/`else` chain)."""
        first_check = stmt.arms[0][0]
        has_binding = stmt.subject is not None
        if has_binding:
            self.context.scopes.push()
            self.expressions.declare_is_binding(first_check)
        self.expressions.check_expr(first_check)
        if first_check.nid in self.facts.enum_checks:
            self._analyze_enum_match(stmt, return_type)
            if has_binding:
                self.context.scopes.pop()
            return
        tested = set()
        for i, (check, body) in enumerate(stmt.arms):
            self.expressions.check_expr(check)
            if i == 0:
                self._check_match_exhaustiveness(stmt)
            variant = self.facts.narrowed[check.nid]
            tested.add(variant)
            self._analyze_body(body, return_type, self.context.scopes.when(check)[0])
        if stmt.else_body is not None:  # the subject is none of the arms' variants
            decl_id = self.facts.decls[first_check.nid]
            self._analyze_body(stmt.else_body, return_type,
                               {decl_id: (stmt.variable_name, self.context.scopes.possible_variants(decl_id) - tested)})
        if has_binding:
            self.context.scopes.pop()

    def _analyze_enum_match(self, stmt: 'Match', return_type: Type) -> None:
        """`match` on an enum: each arm names a member, and the arms cover every member unless there is
        an `else`. Nothing is narrowed."""
        subject_type = self.facts.enum_checks[stmt.arms[0][0].nid][1]
        members = self.decls.enums[subject_type.enum_name].members
        seen = set()
        for check, _ in stmt.arms:
            self.expressions.check_expr(check)
            if check.nid not in self.facts.enum_checks:
                raise SemanticError(f"'{check.type_name}' is not a member of {subject_type}", check)
            index = self.facts.enum_checks[check.nid][0]
            if index in seen:
                raise SemanticError(
                    f"'{members[index]}' is tested more than once in this match on '{stmt.variable_name}'", check)
            seen.add(index)
        missing = [member for index, member in enumerate(members) if index not in seen]
        if missing and stmt.else_body is None:
            raise SemanticError(
                f"This match on '{stmt.variable_name}' (declared {subject_type}) doesn't cover every member -- "
                f"missing: {', '.join(missing)} (add an arm for each, or an 'else:' to cover the rest)", stmt)
        for _, body in stmt.arms:
            self._analyze_body(body, return_type)
        if stmt.else_body is not None:
            self._analyze_body(stmt.else_body, return_type)

    def _check_match_exhaustiveness(self, stmt: 'Match') -> None:
        """Reject duplicate arms; require exhaustiveness without an else."""
        subject_name = stmt.variable_name
        subject_type = self.symbols[self.facts.decls[stmt.arms[0][0].nid]].type  # as declared, not as narrowed
        sum_type_info = self.decls.sum_types[subject_type.sum_type_name]

        seen: Dict[Type, IsCheck] = {}
        for arm_condition, _ in stmt.arms:
            arm_type = self._type(arm_condition.type_name, arm_condition)
            if arm_type in seen:
                raise SemanticError(
                    f"'{arm_condition.type_name}' is tested more than once in "
                    f"this match on '{subject_name}'",
                    arm_condition,
                )
            seen[arm_type] = arm_condition

        if stmt.else_body is not None:
            return

        missing = [v for v in sum_type_info.variants if v not in seen]
        if missing:
            raise SemanticError(
                f"This match on '{subject_name}' (declared {subject_type}) "
                f"doesn't cover every variant -- missing: "
                f"{', '.join(str(v) for v in missing)} "
                f"(add an arm for each, or an 'else:' to cover the rest)",
                stmt,
            )

    def analyze_while(self, stmt: While, return_type: Type) -> None:
        """The condition narrows the body, like an `if`'s (and, when it is false, what follows a loop
        with no `break`: _analyze_block)."""
        self.context.scopes.forget_assigned_in(stmt.body)
        condition_type = self.expressions.check_expr(stmt.condition)
        if condition_type != Type.BOOL:
            raise SemanticError(
                f"'while' condition must be bool, got {condition_type} "
                f"(no implicit int-to-bool conversion -- try `x != 0` "
                f"instead of `x`)",
                stmt.condition,
            )

        self.loop_depth += 1
        self._analyze_body(stmt.body, return_type, self.context.scopes.when(stmt.condition)[0])
        self.loop_depth -= 1

    def analyze_for(self, stmt: For, return_type: Type) -> None:
        """`for init; cond; increment:`; one scope spans all clauses."""
        self.context.scopes.push()
        self.analyze_statement(stmt.init, return_type)
        self.context.scopes.forget_assigned_in([stmt.body, stmt.increment])
        condition_type = self.expressions.check_expr(stmt.condition)
        if condition_type != Type.BOOL:
            raise SemanticError(
                f"'for' condition must be bool, got {condition_type} "
                f"(no implicit int-to-bool conversion -- try `x != 0` "
                f"instead of `x`)",
                stmt.condition,
            )
        self.loop_depth += 1
        mark = self.context.scopes.mark()
        # The condition narrows the body, as a `while`'s does.
        self.context.scopes.apply(self.context.scopes.when(stmt.condition)[0])
        self._analyze_block(stmt.body, return_type)
        self.context.scopes.end_region(mark)
        self.loop_depth -= 1
        self.analyze_statement(stmt.increment, return_type)
        self.context.scopes.pop()

    def analyze_for_in(self, stmt: ForIn, return_type: Type) -> None:
        """`for a[, b] in iterable:` over arrays, slices, dicts, and strings (bytes)."""
        if not isinstance(
                stmt.iterable,
                (Variable, Field, Index, Slice, ArrayLiteral, SliceLiteral, DictLiteral, StringLiteral, Call)):
            raise SemanticError(
                f"'for ... in' requires a variable, field, index, "
                f"slice, or array/dict/str literal as its own iterable, "
                f"not a {type(stmt.iterable).__name__}",
                stmt.iterable,
            )
        if isinstance(stmt.iterable, Call) or (
                isinstance(stmt.iterable, (Field, Index, Slice))
                and self.expressions.root_variable_of(stmt.iterable) is None):
            raise SemanticError(
                f"'for ... in' does not support a function call result "
                f"as its own iterable, or anywhere in its own "
                f"iterable's chain -- assign it to a variable first",
                stmt.iterable,
            )
        iterable_type = self.expressions.check_expr(stmt.iterable)
        if iterable_type.kind not in (TypeKind.ARRAY, TypeKind.SLICE, TypeKind.DICT, TypeKind.STR):
            raise SemanticError(
                f"'for ... in' requires an array, slice, dict, or str as its own "
                f"iterable, got {iterable_type}",
                stmt.iterable,
            )
        num_bindings = len(stmt.binding_names)
        if iterable_type.kind == TypeKind.STR:
            binding_types = [Type.INT, Type.UINT8] if num_bindings == 2 else [Type.UINT8]
        elif iterable_type.kind == TypeKind.DICT:
            binding_types = [iterable_type.key_type, iterable_type.element_type][:num_bindings]
        else:
            binding_types = [
                Type.INT, iterable_type.element_type] if num_bindings == 2 else [iterable_type.element_type]
        self.context.scopes.push()
        symbols = [
            self.symbols.new(name, 'binding', t, stmt) for name, t in zip(stmt.binding_names, binding_types)
        ]
        self.facts.for_symbols[stmt.nid] = symbols
        for name, binding_type, sym in zip(stmt.binding_names, binding_types, symbols):
            self.context.scopes.declare(name, binding_type, stmt, sym.id)
        self.context.scopes.forget_assigned_in(stmt.body)
        self.loop_depth += 1
        self._analyze_block(stmt.body, return_type)
        self.loop_depth -= 1
        self.context.scopes.pop()

    def analyze_break(self, stmt: Break) -> None:
        if self.loop_depth == 0:
            raise SemanticError("'break' outside of a loop", stmt)

    def analyze_continue(self, stmt: Continue) -> None:
        if self.loop_depth == 0:
            raise SemanticError("'continue' outside of a loop", stmt)
