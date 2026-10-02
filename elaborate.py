"""Elaboration: the analyzed parser tree to the typed tree (typed_ast.py).

For now this runs after semantic.analyze() and reads its annotations; the checks themselves will
move here when later stages switch to the typed tree. Every implicit operation becomes explicit:
values flowing into a slot of a known type go through `_convert`.
"""

import parser as p
import typed_ast as t
from ops import BinaryOp, UnaryOp
from typesys import Type, TypeKind

_BYTE_SLICE = Type(TypeKind.SLICE, element_type=Type.UINT8)


class ElaborationError(Exception):
    """A tree shape the elaborator has no rule for: a compiler bug, since analysis accepted it."""


def elaborate(program: p.Program) -> t.Program:
    return _Elaborator(program).program()


class _Elaborator:
    def __init__(self, program: p.Program):
        self.ast = program
        self.symbols = program.symbols
        self.structs = program.struct_registry
        self.sum_types = program.sum_type_registry
        self.functions = program.function_registry
        self.externs = {e.name for e in program.extern_functions}
        self.intrinsics = {i.name for i in program.intrinsics}
        self.return_type = None

    def program(self) -> t.Program:
        return t.Program(tuple(self.function(fn) for fn in self.ast.functions), self.structs, self.sum_types,
                         self.symbols)

    def function(self, fn: p.Function) -> t.Function:
        self.return_type = fn.resolved_return_type
        return t.Function(fn.name, tuple(param.symbol for param in fn.params), fn.resolved_return_type,
                          self.block(fn.body))

    # -- statements

    def block(self, statements) -> tuple:
        out = []
        for stmt in statements or ():
            out.extend(self.statement(stmt))
        return tuple(out)

    def statement(self, s) -> list:
        """Typed statements for one parser statement (a narrowing binding adds its declaration)."""
        if isinstance(s, p.VarDecl):
            return [self.declare(s)]
        if isinstance(s, p.Assign):
            symbol = self.symbols[s.decl_id]
            return [t.Assign(t.Local(symbol.type, symbol), self.convert(s.value, symbol.type))]
        if isinstance(s, (p.IndexAssign, p.FieldAssign, p.DerefAssign)):
            if isinstance(s, p.IndexAssign):
                target = self.index(p.Index(array=s.array, index=s.index))
            elif isinstance(s, p.FieldAssign):
                target = self.field(p.Field(base=s.base, name=s.name))
            else:
                pointer = self.expr(s.pointer)
                target = t.Deref(pointer.type.element_type, pointer)
            if s.compound_op is not None:
                return [t.CompoundAssign(target, s.compound_op, self.convert(s.value, target.type))]
            return [t.Assign(target, self.convert(s.value, target.type))]
        if isinstance(s, p.ExprStmt):
            return [t.ExprStmt(self.expr(s.expr))]
        if isinstance(s, p.Return):
            return [t.Return(None if s.value is None else self.convert(s.value, self.return_type))]
        if isinstance(s, p.If):
            return self.if_statement(s)
        if isinstance(s, p.While):
            return [t.While(self.expr(s.condition), self.block(s.body))]
        if isinstance(s, p.For):
            init = self.statement(s.init) if s.init is not None else []
            if len(init) > 1:
                raise ElaborationError(f"for-loop initializer elaborated to {len(init)} statements")
            step = self.statement(s.increment)
            return [t.For(init[0] if init else None, self.expr(s.condition), step[0], self.block(s.body))]
        if isinstance(s, p.ForIn):
            iterable = self.expr(s.iterable)
            kind = {TypeKind.ARRAY: 'array', TypeKind.SLICE: 'slice', TypeKind.STR: 'str',
                    TypeKind.DICT: 'dict'}[iterable.type.kind]
            return [t.ForIn(kind, iterable, tuple(s.symbols), self.block(s.body))]
        if isinstance(s, p.Break):
            return [t.Break()]
        if isinstance(s, p.Continue):
            return [t.Continue()]
        raise ElaborationError(f"No elaboration for statement {type(s).__name__}")

    def declare(self, s: p.VarDecl) -> t.Declare:
        symbol = s.symbol
        init = self.zero(symbol.type) if s.init is None else self.convert(s.init, symbol.type)
        return t.Declare(symbol, init)

    def if_statement(self, s: p.If) -> list:
        if not isinstance(s.condition, p.IsCheck):
            return [t.If(self.expr(s.condition), self.block(s.then_body), self.block(s.else_body))]
        before, subject = self.narrowing_subject(s.condition)
        if s.is_match:
            arms, current = [], s
            for i in range(s.match_arm_count):
                arms.append((current.condition.narrowed_type, self.block(current.then_body)))
                if i < s.match_arm_count - 1:
                    current = current.else_body[0]
            else_body = None if current.else_body is None else self.block(current.else_body)
            return before + [t.Match(subject, tuple(arms), else_body)]
        test = t.TagTest(Type.BOOL, subject, s.condition.narrowed_type)
        return before + [t.If(test, self.block(s.then_body), self.block(s.else_body))]

    def narrowing_subject(self, check: p.IsCheck):
        """(statements to run first, the sum being tested) for `NAME is T` or `EXPR is T as NAME`."""
        if check.binding_decl is not None:
            symbol = check.binding_decl.symbol
            declare = t.Declare(symbol, self.convert(check.subject, symbol.type))
            return [declare], t.Local(symbol.type, symbol)
        symbol = self.symbols[check.decl_id]
        return [], t.Local(symbol.type, symbol)

    # -- conversions

    def zero(self, type_: Type) -> t.Expr:
        if type_.kind == TypeKind.DICT:
            return t.NewEmptyDict(type_)
        if type_.kind == TypeKind.SLICE:
            return t.EmptySlice(type_)
        return t.ZeroValue(type_)

    def convert(self, e, target: Type) -> t.Expr:
        """`e` as a value flowing into a slot of type `target`, implicit operations made explicit."""
        if isinstance(e, p.Unary) and e.boxed_sum is not None:
            return t.BoxVariant(target, t.WidenToSum(e.boxed_sum, self.expr(e.operand)))
        if isinstance(e, p.NoneLiteral):
            if target.kind == TypeKind.POINTER:
                return t.NoneLit(target)
            if target.kind == TypeKind.SUM:
                return t.WidenToSum(target, t.NoneLit(Type.NONE))
            raise ElaborationError(f"`none` flowing into {target}")
        if (target.kind == TypeKind.SLICE and isinstance(e, p.ArrayLiteral) and e.type_expr is None):
            elements = tuple(self.convert(x, target.element_type) for x in e.elements)
            return t.SliceLiteral(target, elements) if elements else t.EmptySlice(target)
        if target.kind == TypeKind.ARRAY and isinstance(e, p.ArrayLiteral) and e.type_expr is None:
            return t.ArrayLiteral(target, tuple(self.convert(x, target.element_type) for x in e.elements))
        value = self.expr(e)
        if value.type == target:
            return value
        if target.kind == TypeKind.SUM and value.type in self.sum_types[target.sum_type_name].variants:
            return t.WidenToSum(target, value)
        if value.type.kind == TypeKind.POINTER and value.type.element_type == target:
            return t.Deref(target, value)  # a method's value receiver called through a pointer
        if isinstance(value, t.IntLit) and target.kind in (TypeKind.INT, TypeKind.INT32, TypeKind.INT8, TypeKind.UINT8):
            return t.IntLit(target, value.value)
        raise ElaborationError(f"No conversion from {value.type} to {target} for {e!r}")

    # -- expressions

    def expr(self, e) -> t.Expr:
        if isinstance(e, p.Constant):
            return t.IntLit(e.resolved_type, e.value)
        if isinstance(e, p.BoolLiteral):
            return t.BoolLit(Type.BOOL, e.value)
        if isinstance(e, p.StringLiteral):
            return t.StrLit(Type.STR, e.value)
        if isinstance(e, p.ByteLiteral):
            return t.IntLit(Type.UINT8, e.value)
        if isinstance(e, p.NoneLiteral):
            return t.NoneLit(Type.NONE)
        if isinstance(e, p.Variable):
            symbol = self.symbols[e.decl_id]
            local = t.Local(symbol.type, symbol)
            if e.resolved_type is not None and e.resolved_type != symbol.type:
                return t.Payload(e.resolved_type, local)  # narrowed by an enclosing `is`
            return local
        if isinstance(e, p.ArrayLiteral):
            array_type = e.resolved_type
            return t.ArrayLiteral(array_type, tuple(self.convert(x, array_type.element_type) for x in e.elements))
        if isinstance(e, p.DictLiteral):
            d = e.resolved_type
            return t.DictLiteral(d, tuple((self.convert(k, d.key_type), self.convert(v, d.element_type))
                                          for k, v in e.entries))
        if isinstance(e, p.Index):
            return self.index(e)
        if isinstance(e, p.Slice) and isinstance(e.array, p.ArrayLiteral) and e.low is None and e.high is None:
            # A typed slice literal, `[]T[...]`: new storage holding the elements.
            elements = tuple(self.convert(x, e.resolved_type.element_type) for x in e.array.elements)
            return t.SliceLiteral(e.resolved_type, elements) if elements else t.EmptySlice(e.resolved_type)
        if isinstance(e, p.Slice):
            base = self.expr(e.array)
            kind = {TypeKind.ARRAY: 'array', TypeKind.SLICE: 'slice', TypeKind.STR: 'str'}[base.type.kind]
            low = None if e.low is None else self.expr(e.low)
            high = None if e.high is None else self.expr(e.high)
            return t.SliceOf(e.resolved_type, kind, base, low, high)
        if isinstance(e, p.Field):
            return self.field(e)
        if isinstance(e, p.Call):
            return self.call(e)
        if isinstance(e, p.Unary):
            if e.op == UnaryOp.ADDRESS_OF:
                if e.boxed_sum is not None:
                    return self.convert(e, e.resolved_type)
                operand = self.expr(e.operand)
                return t.AddressOf(e.resolved_type, operand)
            if e.op == UnaryOp.DEREFERENCE:
                pointer = self.expr(e.operand)
                return t.Deref(pointer.type.element_type, pointer)
            return t.Unary(e.resolved_type, e.op, self.expr(e.operand))
        if isinstance(e, p.Cast):
            value = self.expr(e.expr)
            if e.resolved_type == Type.STR:
                return (t.StrFromByte if value.type == Type.UINT8 else t.StrFromBytes)(Type.STR, value)
            return t.IntCast(e.resolved_type, value)
        if isinstance(e, p.Binary):
            return self.binary(e)
        raise ElaborationError(f"No elaboration for expression {type(e).__name__}")

    def index(self, e: p.Index) -> t.Expr:
        base = self.expr(e.array)
        kind = base.type.kind
        if kind == TypeKind.DICT:
            return t.DictLookup(base.type.element_type, base, self.convert(e.index, base.type.key_type))
        index = self.convert(e.index, Type.INT) if not isinstance(e.index, p.Constant) else self.expr(e.index)
        node = {TypeKind.ARRAY: t.ArrayIndex, TypeKind.SLICE: t.SliceIndex, TypeKind.STR: t.StrIndex}[kind]
        element = Type.UINT8 if kind == TypeKind.STR else base.type.element_type
        return node(element, base, index)

    def field(self, e: p.Field) -> t.FieldAccess:
        base = self.expr(e.base)
        through_pointer = base.type.kind == TypeKind.POINTER
        struct_type = base.type.element_type if through_pointer else base.type
        names = list(self.structs[struct_type.struct_name].fields)
        field_type = self.structs[struct_type.struct_name].fields[e.name]
        return t.FieldAccess(field_type, base, e.name, names.index(e.name), through_pointer)

    def call(self, e: p.Call) -> t.Expr:
        name = e.name
        if name == 'print':
            return t.Print(Type.VOID, self.expr(e.args[0]))
        if name == 'len':
            return t.Len(Type.INT, self.expr(e.args[0]))
        if name == 'append':
            s = self.expr(e.args[0])
            return t.Append(s.type, s, self.convert(e.args[1], s.type.element_type))
        if name == 'del':
            d = self.expr(e.args[0])
            return t.DictDelete(Type.VOID, d, self.convert(e.args[1], d.type.key_type))
        if name in ('bytes', 'hornet_bytes'):  # analysis rewrites bytes(s) to the runtime function
            return t.BytesFromStr(_BYTE_SLICE, self.expr(e.args[0]))
        if name in self.structs:
            field_types = self.structs[name].fields
            if e.kwargs is not None:
                given = dict(e.kwargs)
                values = tuple(self.convert(given[f], ft) if f in given else self.zero(ft)
                               for f, ft in field_types.items())
            else:
                values = tuple(self.convert(a, ft) for a, ft in zip(e.args, field_types.values()))
            return t.StructLiteral(Type(TypeKind.STRUCT, struct_name=name), values)
        param_types, return_type = self.functions[name]
        kind = 'extern' if name in self.externs else 'intrinsic' if name in self.intrinsics else 'function'
        return t.Call(return_type, name, kind, tuple(self.convert(a, pt) for a, pt in zip(e.args, param_types)))

    def binary(self, e: p.Binary) -> t.Expr:
        op, left_type, right_type = e.op, e.left.resolved_type, e.right.resolved_type
        if op == BinaryOp.IN:
            container = self.expr(e.right)
            if container.type.kind == TypeKind.DICT:
                return t.DictContains(Type.BOOL, self.convert(e.left, container.type.key_type), container)
            return t.ElementContains(Type.BOOL, self.convert(e.left, container.type.element_type), container)
        if op in (BinaryOp.EQUAL, BinaryOp.NOT_EQUAL) and Type.NONE in (left_type, right_type):
            other = e.right if left_type == Type.NONE else e.left
            value = self.expr(other)
            if value.type.kind == TypeKind.SUM:
                test = t.TagTest(Type.BOOL, value, Type.NONE)
                return test if op == BinaryOp.EQUAL else t.Unary(Type.BOOL, UnaryOp.NOT, test)
            return t.Binary(Type.BOOL, op, value, t.NoneLit(value.type))
        if left_type == Type.STR and op == BinaryOp.ADD:
            return t.StrConcat(Type.STR, self.expr(e.left), self.expr(e.right))
        if left_type == Type.STR and op in (BinaryOp.EQUAL, BinaryOp.NOT_EQUAL):
            return t.StrCompare(Type.BOOL, op, self.expr(e.left), self.expr(e.right))
        left, right = self.expr(e.left), self.expr(e.right)
        if left.type != right.type and isinstance(right, t.IntLit):
            right = t.IntLit(left.type, right.value)
        elif left.type != right.type and isinstance(left, t.IntLit):
            left = t.IntLit(right.type, left.value)
        return t.Binary(e.resolved_type, op, left, right)
