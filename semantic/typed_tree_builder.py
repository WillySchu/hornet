
import dataclasses

import parser as syntax
import typed_ast as typed
from ops import BinaryOp, UnaryOp
from typesys import Type, TypeKind


_BYTE_SLICE = Type(TypeKind.SLICE, element_type=Type.UINT8)
_EQUALITY_OPS = {BinaryOp.EQUAL, BinaryOp.NOT_EQUAL}
_ORDERING_OPS = {BinaryOp.LESS_THAN, BinaryOp.GREATER_THAN,
                 BinaryOp.LESS_THAN_OR_EQUAL, BinaryOp.GREATER_THAN_OR_EQUAL}


class ElaborationError(Exception):
    """A tree shape with no typed-tree rule: a compiler bug, since analysis accepted it."""


class TypedTreeBuilder:
    """Builds the typed tree (typed_ast.py) from a checked program: every implicit operation becomes
    explicit, and values flowing into a slot of a known type go through `convert`."""

    def __init__(self, facts, decls, consts: dict, symbols):
        """What checking learned (Facts), what the program declares (Declarations), each constant's
        (type, value) by key, and the symbol table."""
        self.facts = facts
        self.consts = consts
        self.symbols = symbols
        self.structs = decls.structs
        self.sum_types = decls.sum_types
        self.enums = decls.enums
        self.functions = decls.functions
        self.intrinsics = decls.intrinsic_original_names  # name -> the intrinsic's own name
        self.externs = decls.extern_names
        self.return_type = None

    def ty(self, e) -> Type:
        return self.facts.types[e.nid]

    def constant(self, e) -> typed.Expr:
        """A reference to a constant (`NAME` or `alias.NAME`): its value."""
        const_type, value = self.consts[self.facts.const_refs[e.nid]]
        if const_type == Type.BOOL:
            return typed.BoolLit(Type.BOOL, value)
        if const_type == Type.STR:
            return typed.StrLit(Type.STR, value)
        if const_type.kind == TypeKind.ENUM:
            return typed.EnumMember(const_type, self.enums[const_type.enum_name].members[value], value)
        return typed.IntLit(self.ty(e), value)

    def program(self, functions) -> typed.Program:
        return typed.Program(tuple(self.function(fn) for fn in functions), self.structs, self.sum_types,
                             self.symbols, self.enums)

    def function(self, fn: syntax.Function) -> typed.Function:
        self.return_type = self.facts.returns[fn.nid]
        return typed.Function(fn.name, tuple(self.facts.symbols[param.nid] for param in fn.params), self.return_type,
                              self.block(fn.body))

    # -- statements

    def block(self, statements) -> tuple:
        out = []
        for stmt in statements or ():
            out.extend(self.statement(stmt))
        return tuple(out)

    def _at(self, node, source):
        """Give typed `node` the position of parser node `source`, unless it already has one."""
        if isinstance(node, (typed.Expr, typed.Stmt)) and not node.line:
            for name in ('line', 'col', 'file'):
                object.__setattr__(node, name, getattr(source, name))
        return node

    def statement(self, s) -> list:
        """Typed statements for one parser statement (a narrowing binding adds its declaration)."""
        return [self._at(node, s) for node in self._statement(s)]

    def _statement(self, s) -> list:
        if isinstance(s, syntax.VarDecl):
            return [self.declare(s)]
        if isinstance(s, syntax.Assign):
            if isinstance(s.target, syntax.Variable) and s.op is None:
                # `v = value` replaces the whole of v, whatever an `is` check has narrowed it to.
                symbol = self.symbols[self.facts.decls[s.target.nid]]
                target = typed.Local(symbol.type, symbol)
            else:
                # `v += value` reads v first, so a narrowed v is the variant it holds, which is what
                # it is given back: the narrowing still stands afterwards.
                target = self.expr(s.target)
            value = self.convert(s.value, target.type)
            return [typed.Assign(target, value) if s.op is None else typed.CompoundAssign(target, s.op, value)]
        if isinstance(s, syntax.ExprStmt):
            return [typed.ExprStmt(self.expr(s.expr))]
        if isinstance(s, syntax.Return):
            return [typed.Return(None if s.value is None else self.convert(s.value, self.return_type))]
        if isinstance(s, syntax.If):
            return self.if_statement(s)
        if isinstance(s, syntax.Match):
            before, subject = self.narrowing_subject(s.arms[0][0])
            if s.arms[0][0].nid in self.facts.enum_checks:  # an `if`/`elif` chain of equality tests
                chain = self.block(s.else_body)
                for check, body in reversed(s.arms):
                    chain = (self._at(typed.If(self.enum_test(subject, check), self.block(body), chain), check),)
                return before + list(chain)
            arms = tuple((self.facts.narrowed[check.nid], self.block(body)) for check, body in s.arms)
            return before + [typed.Match(subject, arms, None if s.else_body is None else self.block(s.else_body))]
        if isinstance(s, syntax.While):
            return [typed.While(self.expr(s.condition), self.block(s.body))]
        if isinstance(s, syntax.For):
            init = self.statement(s.init) if s.init is not None else []
            if len(init) > 1:
                raise ElaborationError(f"for-loop initializer elaborated to {len(init)} statements")
            step = self.statement(s.increment)
            return [typed.For(init[0] if init else None, self.expr(s.condition), step[0], self.block(s.body))]
        if isinstance(s, syntax.ForIn) and s.nid in self.facts.member_loops:
            enum, member = self.facts.member_loops[s.nid], self.facts.for_symbols[s.nid][0]
            return [typed.ForMembers(enum, member, self.block(s.body))]
        if isinstance(s, syntax.ForIn):
            iterable = self.expr(s.iterable)
            kind = {TypeKind.ARRAY: 'array', TypeKind.SLICE: 'slice', TypeKind.STR: 'str',
                    TypeKind.DICT: 'dict'}[iterable.type.kind]
            return [typed.ForIn(kind, iterable, tuple(self.facts.for_symbols[s.nid]), self.block(s.body))]
        if isinstance(s, syntax.Defer):
            return self.defer(s)
        if isinstance(s, syntax.Break):
            return [typed.Break()]
        if isinstance(s, syntax.Continue):
            return [typed.Continue()]
        raise ElaborationError(f"No elaboration for statement {type(s).__name__}")

    def defer(self, s: syntax.Defer) -> list:
        """`defer CALL`: each of the call's operands is put in a variable of its own here, and the
        deferred call is of those. (A literal needs none.)"""
        call, declarations = self.expr(s.call), []

        def kept(operand: typed.Expr) -> typed.Expr:
            if isinstance(operand, (typed.IntLit, typed.BoolLit, typed.StrLit, typed.NoneLit, typed.EnumMember)):
                return operand
            symbol = self.symbols.new('deferred', 'local', operand.type, s)
            declarations.append(typed.Declare(symbol, operand))
            return typed.Local(operand.type, symbol)

        operands = {}
        for field in dataclasses.fields(call):
            value = getattr(call, field.name)
            if isinstance(value, typed.Expr):
                operands[field.name] = kept(value)
            elif isinstance(value, tuple) and value and all(isinstance(v, typed.Expr) for v in value):
                operands[field.name] = tuple(kept(v) for v in value)
        return declarations + [typed.Defer(dataclasses.replace(call, **operands))]

    def declare(self, s: syntax.VarDecl) -> typed.Declare:
        symbol = self.facts.symbols[s.nid]
        init = self.zero(symbol.type) if s.init is None else self.convert(s.init, symbol.type)
        return typed.Declare(symbol, init)

    def if_statement(self, s: syntax.If) -> list:
        if not (isinstance(s.condition, syntax.IsCheck) and s.condition.binds):
            return [typed.If(self.expr(s.condition), self.block(s.then_body), self.block(s.else_body))]
        before, subject = self.narrowing_subject(s.condition)  # declares the `as NAME` binding
        return before + [typed.If(self.is_test(subject, s.condition), self.block(s.then_body),
                                  self.block(s.else_body))]

    def is_test(self, subject: typed.Expr, check: syntax.IsCheck) -> typed.Expr:
        """`subject is T`: a tag test on a sum, an equality test on an enum."""
        if check.nid in self.facts.enum_checks:
            return self.enum_test(subject, check)
        return typed.TagTest(Type.BOOL, subject, self.facts.narrowed[check.nid])

    def enum_test(self, subject: typed.Local, check: syntax.IsCheck) -> typed.Binary:
        """`subject is Member` on an enum: `subject == Enum.Member`. The subject may be a sum's variable
        narrowed to the enum."""
        index, enum = self.facts.enum_checks[check.nid]
        value = subject if subject.type == enum else typed.Payload(enum, subject)
        member = typed.EnumMember(enum, self.enums[enum.enum_name].members[index], index)
        return typed.Binary(Type.BOOL, BinaryOp.EQUAL, value, member)

    def narrowing_subject(self, check: syntax.IsCheck):
        """(statements to run first, the sum being tested) for `NAME is T` or `EXPR is T as NAME`."""
        binding = self.facts.bindings.get(check.nid)
        if binding is not None:
            symbol = self.facts.symbols[binding.nid]
            declare = typed.Declare(symbol, self.convert(check.subject, symbol.type))
            return [declare], typed.Local(symbol.type, symbol)
        symbol = self.symbols[self.facts.decls[check.nid]]
        return [], typed.Local(symbol.type, symbol)

    # -- conversions

    def zero(self, type_: Type) -> typed.Expr:
        if type_.kind == TypeKind.DICT:
            return typed.NewEmptyDict(type_)
        if type_.kind == TypeKind.SLICE:
            return typed.EmptySlice(type_)
        return typed.ZeroValue(type_)

    def convert(self, e, target: Type) -> typed.Expr:
        """`e` as a value flowing into a slot of type `target`, implicit operations made explicit."""
        return self._at(self._convert(e, target), e)

    def _convert(self, e, target: Type) -> typed.Expr:
        if isinstance(e, syntax.Unary) and e.nid in self.facts.boxed:
            return typed.BoxVariant(target, typed.WidenToSum(self.facts.boxed[e.nid], self.expr(e.operand)))
        if isinstance(e, syntax.NoneLiteral):
            if target.kind == TypeKind.POINTER:
                return typed.NoneLit(target)
            if target.kind == TypeKind.SUM:
                return typed.WidenToSum(target, typed.NoneLit(Type.NONE))
            raise ElaborationError(f"`none` flowing into {target}")
        if (target.kind == TypeKind.SLICE and isinstance(e, syntax.ArrayLiteral) and e.type_expr is None):
            elements = tuple(self.convert(x, target.element_type) for x in e.elements)
            return typed.SliceLiteral(target, elements) if elements else typed.EmptySlice(target)
        if target.kind == TypeKind.ARRAY and isinstance(e, syntax.ArrayLiteral) and e.type_expr is None:
            return typed.ArrayLiteral(target, tuple(self.convert(x, target.element_type) for x in e.elements))
        value = self.expr(e)
        if value.type == target:
            return value
        if e.nid in self.facts.narrowed_sums:
            return typed.NarrowSum(target, value)
        if target.kind == TypeKind.SUM and value.type in self.sum_types[target.sum_type_name].variants:
            return typed.WidenToSum(target, value)
        if target.kind == TypeKind.SUM and value.type.kind == TypeKind.SUM:  # (checked to have its variants)
            return typed.WidenSum(target, value)
        if value.type.kind == TypeKind.POINTER and value.type.element_type == target:
            return typed.Deref(target, value)  # a method's value receiver called through a pointer
        if (
                isinstance(value, typed.IntLit)
                and target.kind in (TypeKind.INT, TypeKind.INT32, TypeKind.INT8, TypeKind.UINT8)
        ):
            return typed.IntLit(target, value.value)
        raise ElaborationError(f"No conversion from {value.type} to {target} for {e!r}")

    # -- expressions

    def expr(self, e) -> typed.Expr:
        return self._at(self._expr(e), e)

    def _expr(self, e) -> typed.Expr:
        if isinstance(e, syntax.Constant):
            return typed.IntLit(self.ty(e), e.value)
        if isinstance(e, syntax.BoolLiteral):
            return typed.BoolLit(Type.BOOL, e.value)
        if isinstance(e, syntax.StringLiteral):
            return typed.StrLit(Type.STR, e.value)
        if isinstance(e, syntax.ByteLiteral):
            return typed.IntLit(Type.UINT8, e.value)
        if isinstance(e, syntax.NoneLiteral):
            return typed.NoneLit(Type.NONE)
        if isinstance(e, syntax.Variable):
            decl = self.facts.decls[e.nid]
            if decl is None:
                return self.constant(e)
            symbol = self.symbols[decl]
            local = typed.Local(symbol.type, symbol)
            if self.ty(e) != symbol.type:
                return typed.Payload(self.ty(e), local)  # narrowed by an `is` check
            return local
        if isinstance(e, syntax.ArrayLiteral):
            array_type = self.ty(e)
            return typed.ArrayLiteral(array_type, tuple(self.convert(x, array_type.element_type) for x in e.elements))
        if isinstance(e, syntax.DictLiteral):
            d = self.ty(e)
            return typed.DictLiteral(d, tuple((self.convert(k, d.key_type), self.convert(v, d.element_type))
                                              for k, v in e.entries))
        if isinstance(e, syntax.Index):
            return self.index(e)
        if isinstance(e, syntax.SliceLiteral):
            elements = tuple(self.convert(x, self.ty(e).element_type) for x in e.elements)
            return typed.SliceLiteral(self.ty(e), elements) if elements else typed.EmptySlice(self.ty(e))
        if (
                isinstance(e, syntax.Slice) and isinstance(e.array, syntax.ArrayLiteral)
                and e.low is None and e.high is None
        ):
            # `[...][:]`: new storage holding the elements, like a slice literal.
            elements = tuple(self.convert(x, self.ty(e).element_type) for x in e.array.elements)
            return typed.SliceLiteral(self.ty(e), elements) if elements else typed.EmptySlice(self.ty(e))
        if isinstance(e, syntax.Slice):
            base = self.expr(e.array)
            kind = {TypeKind.ARRAY: 'array', TypeKind.SLICE: 'slice', TypeKind.STR: 'str'}[base.type.kind]
            low = None if e.low is None else self.expr(e.low)
            high = None if e.high is None else self.expr(e.high)
            return typed.SliceOf(self.ty(e), kind, base, low, high)
        if isinstance(e, syntax.Field):
            if e.nid in self.facts.enum_members:
                enum, index = self.facts.enum_members[e.nid]
                return typed.EnumMember(self.ty(e), self.enums[enum].members[index], index)
            return self.constant(e) if e.nid in self.facts.const_refs else self.field(e)
        if isinstance(e, syntax.Call):
            return self.call(e)
        if isinstance(e, syntax.Unary):
            if e.op == UnaryOp.ADDRESS_OF:
                if e.nid in self.facts.boxed:
                    return self.convert(e, self.ty(e))
                operand = self.expr(e.operand)
                return typed.AddressOf(self.ty(e), operand)
            if e.op == UnaryOp.DEREFERENCE:
                pointer = self.expr(e.operand)
                return typed.Deref(pointer.type.element_type, pointer)
            return typed.Unary(self.ty(e), e.op, self.expr(e.operand))
        if isinstance(e, syntax.Cast):
            value = self.expr(e.expr)
            if self.ty(e) == Type.STR and value.type.kind == TypeKind.ENUM:
                if isinstance(value, typed.EnumMember):  # a member or a constant: its name is known here
                    return typed.StrLit(Type.STR, value.name)
                return typed.EnumName(Type.STR, value)
            if self.ty(e) == Type.STR:
                return (typed.StrFromByte if value.type == Type.UINT8 else typed.StrFromBytes)(Type.STR, value)
            return typed.IntCast(self.ty(e), value)
        if isinstance(e, syntax.Binary):
            return self.binary(e)
        if isinstance(e, syntax.IsCheck):
            if e.variable_name is None:
                return self.is_test(self.expr(e.subject), e)
            if e.binds:  # one of a condition's checks joined by `and` (a whole condition is if_statement's)
                symbol = self.facts.symbols[self.facts.bindings[e.nid].nid]
                return typed.Bind(Type.BOOL, symbol, self.convert(e.subject, symbol.type),
                                  self.is_test(typed.Local(symbol.type, symbol), e))
            symbol = self.symbols[self.facts.decls[e.nid]]
            return self.is_test(typed.Local(symbol.type, symbol), e)
        raise ElaborationError(f"No elaboration for expression {type(e).__name__}")

    def index(self, e: syntax.Index) -> typed.Expr:
        base = self.expr(e.array)
        kind = base.type.kind
        if kind == TypeKind.DICT:
            return typed.DictLookup(base.type.element_type, base, self.convert(e.index, base.type.key_type))
        index = self.convert(e.index, Type.INT) if not isinstance(e.index, syntax.Constant) else self.expr(e.index)
        node = {TypeKind.ARRAY: typed.ArrayIndex, TypeKind.SLICE: typed.SliceIndex, TypeKind.STR: typed.StrIndex}[kind]
        element = Type.UINT8 if kind == TypeKind.STR else base.type.element_type
        return node(element, base, index)

    def field(self, e: syntax.Field) -> typed.FieldAccess:
        base = self.expr(e.base)
        through_pointer = base.type.kind == TypeKind.POINTER
        struct_type = base.type.element_type if through_pointer else base.type
        names = list(self.structs[struct_type.struct_name].fields)
        field_type = self.structs[struct_type.struct_name].fields[e.name]
        return typed.FieldAccess(field_type, base, e.name, names.index(e.name), through_pointer)

    def shown(self, e) -> typed.Expr:
        """A value given to print or format. A `none` that is no pointer's and no sum's has no type to be
        shown by: it is the text every none is shown as."""
        if self.facts.types[e.nid] == Type.NONE:
            return typed.StrLit(Type.STR, 'none')
        return self.expr(e)

    def call(self, e: syntax.Call) -> typed.Expr:
        name, args = self.facts.calls.get(e.nid, (e.name, e.args))
        if name == 'print':
            return typed.Print(Type.VOID, self.shown(args[0]))
        if name == 'len':
            if e.nid in self.facts.enum_lens:
                return typed.IntLit(Type.INT, self.facts.enum_lens[e.nid])
            return typed.Len(Type.INT, self.expr(args[0]))
        if name == 'append':
            s = self.expr(args[0])
            return typed.Append(s.type, s, self.convert(args[1], s.type.element_type))
        if name == 'del':
            d = self.expr(args[0])
            return typed.DictDelete(Type.VOID, d, self.convert(args[1], d.type.key_type))
        if name == 'bytes':
            return typed.BytesFromStr(_BYTE_SLICE, self.expr(args[0]))
        if name == 'panic':
            return typed.Panic(Type.NEVER, self.expr(args[0]))
        if name == 'format':
            template = self.facts.formats[e.nid]
            if len(args) == 1:  # nothing to fill in: the text itself
                return typed.StrLit(Type.STR, typed.format_pieces(template)[0])
            return typed.Format(Type.STR, template, tuple(self.shown(a) for a in args[1:]))
        if name in self.enums:  # `Enum(n)`
            value = self.expr(args[0])
            if isinstance(value, typed.IntLit):  # checked to be a member
                return typed.EnumMember(self.ty(e), self.enums[name].members[value.value], value.value)
            return typed.EnumFromInt(self.ty(e), value)
        if name in self.structs:
            field_types = self.structs[name].fields
            if e.kwargs is not None:
                given = dict(e.kwargs)
                values = tuple(self.convert(given[f], ft) if f in given else self.zero(ft)
                               for f, ft in field_types.items())
            else:
                values = tuple(self.convert(a, ft) for a, ft in zip(args, field_types.values()))
            return typed.StructLiteral(Type(TypeKind.STRUCT, struct_name=name), values)
        param_types, return_type = self.functions[name]
        converted = tuple(self.convert(a, pt) for a, pt in zip(args, param_types))
        if name in self.intrinsics:
            intrinsic = {'_raw_ptr': typed.StrRawPtr, '_raw_len': typed.StrRawLen,
                         '_from_raw_parts': typed.StrFromRawParts}[self.intrinsics[name]]
            return intrinsic(return_type, *converted)
        return typed.Call(return_type, name, 'extern' if name in self.externs else 'function', converted)

    def binary(self, e: syntax.Binary) -> typed.Expr:
        if e.nid in self.facts.enum_ins:
            enum = self.facts.enum_ins[e.nid]
            return typed.EnumContains(Type.BOOL, self.expr(e.left), Type(TypeKind.ENUM, enum_name=enum))
        op, left_type, right_type = e.op, self.ty(e.left), self.ty(e.right)
        if op == BinaryOp.IN:
            container = self.expr(e.right)
            if container.type.kind == TypeKind.DICT:
                return typed.DictContains(Type.BOOL, self.convert(e.left, container.type.key_type), container)
            return typed.ElementContains(Type.BOOL, self.convert(e.left, container.type.element_type), container)
        if op in (BinaryOp.EQUAL, BinaryOp.NOT_EQUAL) and Type.NONE in (left_type, right_type):
            other = e.right if left_type == Type.NONE else e.left
            value = self.expr(other)
            if value.type.kind == TypeKind.SUM:
                test = typed.TagTest(Type.BOOL, value, Type.NONE)
                return test if op == BinaryOp.EQUAL else typed.Unary(Type.BOOL, UnaryOp.NOT, test)
            return typed.Binary(Type.BOOL, op, value, typed.NoneLit(value.type))
        if e.nid in self.facts.sum_equalities:  # both sides as the one sum they are compared as
            compared = self.facts.sum_equalities[e.nid]
            return typed.Binary(Type.BOOL, op, self.convert(e.left, compared), self.convert(e.right, compared))
        if left_type == Type.STR and op == BinaryOp.ADD:
            return typed.StrConcat(Type.STR, self.expr(e.left), self.expr(e.right))
        if left_type == Type.STR and op in _EQUALITY_OPS | _ORDERING_OPS:
            return typed.StrCompare(Type.BOOL, op, self.expr(e.left), self.expr(e.right))
        return typed.Binary(self.ty(e), op, self.expr(e.left), self.expr(e.right))
