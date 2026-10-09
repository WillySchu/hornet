"""IR from the typed tree (typed_ast.py), one function at a time. NotYetPorted marks a tree shape
with no rule: a compiler bug, since every shape elaboration produces is handled.

Composite values live in memory: `address(e)` gives where one is (materializing a temporary if
needed) and `write_into(dst, e)` stores one; a str is read as (ptr, len) and a slice as
(ptr, len, cap).
"""

from dataclasses import fields, replace

import typed_ast as t
from escape_analysis import analyze_array_escapes, is_heap_allocated
from ir.panics import PanicBlocks, located, message_label
from ir.typedesc import type_descriptor
from ir.ir import (
    IRBinOp, IRBoundsCheck, IRBranch, IRCall, IRCast, IRConst, IRCopy, IRFunction, IRJump, IRLabel, IRLoad,
    IRLocalAddress, IRMove, IRNullCheck, IRReturn, IRSliceBoundsCheck, IRStaticDataAddress, IRStore, IRUnOp, Temp,
)
from ops import BinaryOp, UnaryOp
from typesys import SUM_TYPE_TAG_WIDTH, Type, TypeKind, type_byte_width

_SCALAR_KINDS = {TypeKind.INT, TypeKind.INT32, TypeKind.INT8, TypeKind.UINT8, TypeKind.BOOL,
                 TypeKind.POINTER, TypeKind.ENUM}


class NotYetPorted(Exception):
    """A typed-tree shape with no IR rule."""


def _scalar(type_: Type) -> bool:
    return type_.kind in _SCALAR_KINDS


_COMPOSITE_BY_ADDRESS = {TypeKind.ARRAY, TypeKind.STRUCT, TypeKind.SUM, TypeKind.DICT}  # passed and returned by address
_PORTED_KINDS = _SCALAR_KINDS | _COMPOSITE_BY_ADDRESS | {TypeKind.STR, TypeKind.SLICE}


def _valueless(type_: Type) -> bool:
    """Whether a call of this type yields nothing: no declared return type, or `never`."""
    return type_ in (Type.VOID, Type.NEVER)


def _ported(type_: Type) -> bool:
    return _valueless(type_) or type_.kind in _PORTED_KINDS


_PLACE_NODES = (t.Local, t.Deref, t.FieldAccess, t.ArrayIndex, t.SliceIndex, t.Payload, t.DictLookup)
_DICT_BUCKET_OCCUPIED = 1  # runtime.c's HORNET_DICT_BUCKET_OCCUPIED
_DICT_HEADER_SIZE = 32  # {buckets, count, tombstones, capacity}


def _as_int32(value):
    """`value` with an enum's type replaced by int32: an enum value is its member's index, and the
    IR and backends know it only as that integer."""
    if isinstance(value, (Temp, IRConst)) and value.type.kind == TypeKind.ENUM:
        return replace(value, type=Type.INT32)
    if isinstance(value, Type) and value.kind == TypeKind.ENUM:
        return Type.INT32
    if isinstance(value, list):
        return [_as_int32(item) for item in value]
    return value


def _without_enum_types(instr):
    """`instr` with every enum-typed operand and value type as int32 (_as_int32)."""
    return replace(instr, **{f.name: _as_int32(getattr(instr, f.name)) for f in fields(instr)})


def link_name(name: str) -> str:
    """A Hornet function's linker symbol. A key with a module or struct prefix (`m$f`, `S.f`) can't
    be a C identifier; an entry-file function's bare name gets a trailing `$`, so it never stands
    in for a libc or runtime symbol. `main` is the C entry point."""
    return name if name == 'main' or '$' in name or '.' in name else name + '$'


def _nodes(root):
    """`root` (a node, or a tuple of them) and every expression and statement under it."""
    stack = list(root) if isinstance(root, tuple) else [root]
    while stack:
        node = stack.pop()
        yield node
        for f in fields(node):
            value = getattr(node, f.name)
            for v in value if isinstance(value, tuple) else (value,):
                stack.extend(x for x in (v if isinstance(v, tuple) else (v,)) if isinstance(x, (t.Expr, t.Stmt)))


def _calls_anything(e) -> bool:
    """Whether evaluating `e` may call a function: the only way an expression can change a dictionary."""
    return any(isinstance(node, t.Call) for node in _nodes(e))


def _addressed(fn: t.Function) -> set:
    """Ids of the variables whose own address is taken (`&x`; not `&x.field`, nor `&u` of a narrowed
    `u`): the only way a whole variable changes without being assigned. So a scalar one is copied
    when read, and a sum one re-checked where an `is` check has narrowed it."""
    return {node.place.symbol.id for node in _nodes(fn.body)
            if isinstance(node, t.AddressOf) and isinstance(node.place, t.Local)}


def _variable(place):
    """The variable `place` is part of by value (through fields, elements, and a narrowed sum's
    payload), or None if it is reached through a pointer, a slice, or a dict."""
    while not isinstance(place, t.Local):
        if isinstance(place, t.FieldAccess) and not place.through_pointer:
            place = place.base
        elif isinstance(place, t.ArrayIndex):
            place = place.base
        elif isinstance(place, t.Payload):
            place = place.sum
        else:
            return None
    return place.symbol


def _exposed(fn: t.Function) -> set:
    """Ids of the variables a pointer or a slice may reach: `&x`, `&x.field`, `&x[i]`, or a slice of
    an array `x[a:b]`, anywhere in the function. Only these can change while a call runs."""
    places = [node.place if isinstance(node, t.AddressOf) else node.base for node in _nodes(fn.body)
              if isinstance(node, t.AddressOf) or (isinstance(node, t.SliceOf) and node.kind == 'array')]
    return {symbol.id for symbol in map(_variable, places) if symbol is not None}


class TypedFunctionBuilder:
    def __init__(self, ir_program):
        self.ir_program = ir_program
        self.ids = ir_program.ids

    def build(self, fn: t.Function) -> IRFunction:
        if not _ported(fn.return_type):
            raise NotYetPorted(f"returns {fn.return_type}")
        self.ir_fn = ir_fn = IRFunction(name=link_name(fn.name))
        ir_fn.return_type = fn.return_type
        self.heap_ids = analyze_array_escapes(fn, self.ir_program.struct_registry, self.ir_program.escape_summaries)
        self.addressed, self.exposed = _addressed(fn), _exposed(fn)
        self.panics = PanicBlocks(self.ir_program, "check_failed")  # for panic_when
        self.storage = {}  # symbol id -> (Temp or None, heap)
        self.loops = []  # (continue label, end label, how many blocks were open around the loop)
        self.deferred = []  # for each open block, innermost last: the calls deferred in it so far
        ir = self.params(fn)
        ir += self.block(fn.body)
        if not ir or not isinstance(ir[-1], (IRBranch, IRJump, IRReturn)):
            ir.append(IRReturn(value=None))
        ir_fn.body = [_without_enum_types(instr) for instr in ir + self.panics.blocks()]
        ir_fn.params = _as_int32(ir_fn.params)
        ir_fn.return_type = _as_int32(ir_fn.return_type)
        return ir_fn

    # -- layout and scratch

    def width(self, type_: Type) -> int:
        return type_byte_width(type_, self.ir_program.struct_registry, self.ir_program.sum_type_registry)

    def temp(self, type_: Type = Type.INT64):
        return self.ids.new_temp(type_)

    def offset(self, base, amount: int) -> tuple:
        if amount == 0:
            return [], base
        address = self.temp()
        return [IRBinOp(dst=address, op=BinaryOp.ADD, left=base, right=IRConst(amount, Type.INT64))], address

    def scratch_address(self, type_: Type) -> tuple:
        """New temporary storage for a composite value: a slot, or the heap if it is too large for the
        stack."""
        address = self.temp()
        if is_heap_allocated(type_, self.ir_program.struct_registry, self.ir_program.sum_type_registry):
            return [IRCall(dst=address, name='hornet_alloc', args=[IRConst(self.width(type_), Type.INT64)])], address
        slot = self.ids.new_slot(self.width(type_), "temporary", self.ir_fn)
        return [IRLocalAddress(dst=address, slot=slot)], address

    # -- storage

    def heap(self, symbol) -> bool:
        return (is_heap_allocated(symbol.type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
                or symbol.id in self.heap_ids)

    def bind(self, symbol):
        """Storage for `symbol`: a scalar's Temp (its value, or a pointer to it on the heap) homed in a
        slot; a composite's slot (holding the value, or a pointer to it on the heap)."""
        if symbol.id in self.storage:
            return self.storage[symbol.id]
        if not _ported(symbol.type):
            raise NotYetPorted(f"{symbol.kind} {symbol} of type {symbol.type}")
        heap = self.heap(symbol)
        label = {'binding': 'for_in', 'narrowing': 'local'}.get(symbol.kind, symbol.kind)
        slot = self.ids.new_slot(8 if heap else self.width(symbol.type), f"{label}:{symbol}", self.ir_fn)
        self.ir_fn.var_slots[symbol.id] = slot
        temp = None
        if _scalar(symbol.type):
            temp_type = Type(TypeKind.POINTER, element_type=symbol.type) if heap else symbol.type
            temp = self.ids.temp_at_offset(temp_type, slot, self.ir_fn)
        self.storage[symbol.id] = (temp, heap)
        return temp, heap

    def local_address(self, symbol) -> tuple:
        """Where a composite variable's value is."""
        _, heap = self.bind(symbol)
        slot_address = self.temp()
        ir = [IRLocalAddress(dst=slot_address, slot=self.ir_fn.var_slots[symbol.id])]
        if not heap:
            return ir, slot_address
        pointer = self.temp()
        return ir + [IRLoad(dst=pointer, address=slot_address)], pointer

    def allocate_heap(self, symbol) -> list:
        """A composite variable on the heap: new storage, its pointer kept in the variable's slot."""
        pointer, slot_address = self.temp(), self.temp()
        return [
            IRCall(dst=pointer, name='hornet_alloc', args=[IRConst(self.width(symbol.type), Type.INT64)]),
            IRLocalAddress(dst=slot_address, slot=self.ir_fn.var_slots[symbol.id]),
            IRStore(address=slot_address, value=pointer, value_type=Type.INT64)
        ]

    def initialize(self, symbol, init) -> list:
        """A variable's first value (a declaration): fresh storage, so it is written in place."""
        temp, heap = self.bind(symbol)
        if not _scalar(symbol.type):
            ir = self.allocate_heap(symbol) if heap else []
            address_ir, address = self.local_address(symbol)
            return ir + address_ir + self.write_into(address, init)
        value_ir, value = self.value(init)
        return value_ir + self.initialize_scalar(symbol, value)

    def initialize_scalar(self, symbol, value) -> list:
        temp, heap = self.bind(symbol)
        if not heap:
            return [IRMove(dst=temp, src=value)]
        box = self.temp()
        return [
            IRCall(dst=box, name='hornet_alloc', args=[IRConst(self.width(symbol.type), Type.INT64)]),
            IRStore(address=box, value=value, value_type=symbol.type),
            IRMove(dst=temp, src=box)
        ]

    def params(self, fn: t.Function) -> list:
        ir, incoming = [], []
        if not _scalar(fn.return_type) and not _valueless(fn.return_type):
            self.hidden_return = self.temp()
            self.ir_fn.params.append(self.hidden_return)
            self.ir_fn.hidden_return_ptr_slot = self.ids.new_slot(8, "hidden_return_ptr", self.ir_fn)
        for symbol in fn.params:
            if not _ported(symbol.type):
                raise NotYetPorted(f"parameter {symbol} of type {symbol.type}")
            words = self.words(symbol.type)
            self.ir_fn.params.extend(words)
            incoming.append(words)
        if self.ir_fn.hidden_return_ptr_slot is not None:
            address = self.temp()
            ir += [IRLocalAddress(dst=address, slot=self.ir_fn.hidden_return_ptr_slot),
                   IRStore(address=address, value=self.hidden_return, value_type=Type.INT64)]
        for symbol, words in zip(fn.params, incoming):
            temp, heap = self.bind(symbol)
            if _scalar(symbol.type):
                if fn.name == 'main' and symbol.type == Type.INT and words[0].type == Type.INT32:
                    widened = self.temp(Type.INT)
                    ir += [IRCast(dst=widened, src=words[0])] + self.initialize_scalar(symbol, widened)
                elif heap:
                    ir += self.initialize_scalar(symbol, words[0])
                else:  # arrives directly in its variable
                    self.ir_fn.params[self.ir_fn.params.index(words[0])] = temp
                continue
            ir += self.allocate_heap(symbol) if heap else []
            address_ir, address = self.local_address(symbol)
            ir += address_ir
            if symbol.type.kind == TypeKind.STR:
                ir += self.write_str(address, *words)
            elif symbol.type.kind == TypeKind.SLICE:
                ir += self.write_slice(address, *words)
            else:  # the caller passed its address; the parameter is a copy
                ir.append(IRCopy(dst_address=address, src_address=words[0], value_type=symbol.type))
        return ir

    def words(self, type_: Type) -> list:
        """The incoming words of a parameter of `type_` (Hornet's calling convention)."""
        if type_.kind == TypeKind.STR:
            return [self.temp(), self.temp(Type.INT)]
        if type_.kind == TypeKind.SLICE:
            return [self.temp(), self.temp(Type.INT), self.temp(Type.INT)]
        if type_.kind in _COMPOSITE_BY_ADDRESS:
            return [self.temp()]
        if self.ir_fn.name == 'main' and type_ == Type.INT and not self.ir_fn.params:
            return [self.temp(Type.INT32)]  # main's argc is a C int
        return [self.temp(type_)]

    # -- statements

    def block(self, statements) -> list:
        """A block's statements, then the calls deferred in it, for when its end is reached."""
        self.deferred.append([])
        ir = []
        for s in statements or ():
            ir += self.statement(s)
        if not (statements and isinstance(statements[-1], (t.Return, t.Break, t.Continue))):
            ir += self.leave(len(self.deferred) - 1)
        self.deferred.pop()
        return ir

    def leave(self, outermost: int) -> list:
        """The deferred calls of the open blocks from the innermost out to block number `outermost`,
        each block's last deferred first: what leaving those blocks from here does."""
        ir = []
        for calls in reversed(self.deferred[outermost:]):
            for call in reversed(calls):
                ir += self.statement(t.ExprStmt(call, line=call.line, col=call.col, file=call.file))
        return ir

    def statement(self, s) -> list:
        if isinstance(s, t.Declare):
            return self.initialize(s.symbol, s.init)
        if isinstance(s, t.Assign):
            return self.assign(s.target, s.value)
        if isinstance(s, t.CompoundAssign):
            if isinstance(s.target, t.DictLookup) and _calls_anything(s.value):
                return self.compound_assign_dict_entry(s)
            if isinstance(s.target, t.Local):  # read and written in place: `x = x op v`
                kind = t.StrConcat if s.target.type == Type.STR else lambda type_, l, r: t.Binary(
                    type_, s.op, l, r, line=s.line, col=s.col, file=s.file)
                return self.assign(s.target, kind(s.target.type, s.target, s.value))
            if s.target.type == Type.STR:  # `+=`: the place's address once, then concatenate
                ir, address = self.place_address(s.target)
                read_ir, left_ptr, left_len = self.read_str(address)
                value_ir, right_ptr, right_len = self.str_value(s.value)
                concat_ir, ptr, length = self.concat(left_ptr, left_len, right_ptr, right_len)
                len_ir, len_address = self.offset(address, 8)
                return ir + read_ir + value_ir + concat_ir + len_ir + [
                    IRStore(address=address, value=ptr, value_type=Type.INT64),
                    IRStore(address=len_address, value=length, value_type=Type.INT),
                ]
            if not _scalar(s.target.type):
                raise NotYetPorted(f"compound assignment of {s.target.type}")
            ir, address = self.place_address(s.target)
            current, result = self.temp(s.target.type), self.temp(s.target.type)
            value_ir, value = self.value(s.value)
            return ir + [IRLoad(dst=current, address=address)] + value_ir + [
                IRBinOp(dst=result, op=s.op, left=current, right=value, where=s.where),
                IRStore(address=address, value=result, value_type=s.target.type),
            ]
        if isinstance(s, t.ExprStmt):
            if s.expr.type == Type.NONE:  # a bare `none` does nothing
                return []
            if _scalar(s.expr.type) or _valueless(s.expr.type):
                ir, _ = self.value(s.expr, discard=True)
                return ir
            ir, _ = self.address(s.expr)  # evaluated for its effects (calls, checks)
            return ir
        if isinstance(s, t.Defer):
            self.deferred[-1].append(s.call)
            return []
        if isinstance(s, t.Return):  # the value first, then what every open block deferred
            if s.value is None:
                return self.leave(0) + [IRReturn(value=None)]
            if _scalar(s.value.type):
                ir, value = self.value(s.value)
                return ir + self.leave(0) + [IRReturn(value=value)]
            hidden, address = self.temp(), self.temp()
            return [IRLocalAddress(dst=address, slot=self.ir_fn.hidden_return_ptr_slot),
                    IRLoad(dst=hidden, address=address)] + self.write_into(hidden, s.value) + self.leave(0) + [
                        IRReturn(value=None)]
        if isinstance(s, t.If):
            then_label, else_label, end_label = (self.ids.new_label(x) for x in ("if_then", "if_else", "if_end"))
            ir = self.branch(s.cond, then_label, else_label) + [IRLabel(then_label)]
            ir += self.block(s.then_body) + [IRJump(end_label), IRLabel(else_label)]
            return ir + self.block(s.else_body) + [IRJump(end_label), IRLabel(end_label)]
        if isinstance(s, t.Match):
            return self.match(s)
        if isinstance(s, t.While):
            start, body, end = (self.ids.new_label(x) for x in ("while_start", "while_body", "while_end"))
            self.loops.append((start, end, len(self.deferred)))
            body_ir = self.block(s.body)
            self.loops.pop()
            return [IRJump(start), IRLabel(start)] + self.branch(s.cond, body, end) + [IRLabel(body)] + body_ir + [
                IRJump(start), IRLabel(end)]
        if isinstance(s, t.For):
            start, body, step, end = (self.ids.new_label(x) for x in ("for_start", "for_body", "for_step", "for_end"))
            ir = self.statement(s.init) if s.init is not None else []
            self.loops.append((step, end, len(self.deferred)))
            body_ir = self.block(s.body)
            self.loops.pop()
            return ir + [
                IRJump(start),
                IRLabel(start)
            ] + self.branch(s.cond, body, end) + [IRLabel(body)] + body_ir + [
                IRJump(step),
                IRLabel(step)
            ] + self.fresh_loop_variable(s) + self.statement(s.step) + [
                IRJump(start),
                IRLabel(end)
            ]
        if isinstance(s, t.ForIn):
            return self.for_in(s)
        if isinstance(s, t.Break):  # out of the loop's body, and any block open inside it
            return self.leave(self.loops[-1][2]) + [IRJump(self.loops[-1][1])]
        if isinstance(s, t.Continue):
            return self.leave(self.loops[-1][2]) + [IRJump(self.loops[-1][0])]
        raise NotYetPorted(type(s).__name__)

    def match(self, s: t.Match) -> list:
        """Compare the subject's tag with each arm's variant in turn."""
        ir, address = self.address(s.subject)
        tag = self.temp(Type.INT32)
        ir.append(IRLoad(dst=tag, address=address))
        end = self.ids.new_label("match_end")
        for variant, body in s.arms:
            arm = self.ids.new_label('match_arm')
            next_arm = self.ids.new_label('match_next')
            test_ir, is_it = self.tag_is(tag, s.subject.type, variant)
            ir += test_ir + [IRBranch(cond=is_it, true_label=arm, false_label=next_arm), IRLabel(arm)]
            ir += self.block(body) + [IRJump(end), IRLabel(next_arm)]
        return ir + self.block(s.else_body) + [IRJump(end), IRLabel(end)]

    def tag_is(self, tag, sum_type: Type, tested: Type) -> tuple:
        """(IR, bool): whether `tag`, a value of `sum_type`'s, is the number of the variant `tested`;
        or, `tested` being a sum, of any of its variants: then bit `tag` of a mask of those numbers."""
        variants = self.ir_program.sum_type_registry[sum_type.sum_type_name].variants
        result = self.temp(Type.BOOL)
        if tested.kind != TypeKind.SUM:
            return [IRBinOp(dst=result, op=BinaryOp.EQUAL, left=tag, right=IRConst(variants.index(tested), Type.INT32))
                    ], result
        numbers = [variants.index(v) for v in self.ir_program.sum_type_registry[tested.sum_type_name].variants]
        if max(numbers) > 62:
            raise NotYetPorted("a test for a sum type against a sum of more than 63 variants")
        wide, bit, kept = self.temp(Type.INT), self.temp(Type.INT), self.temp(Type.INT)
        return [
            IRCast(dst=wide, src=tag),
            IRBinOp(dst=bit, op=BinaryOp.SHIFT_LEFT, left=IRConst(1, Type.INT), right=wide),
            IRBinOp(dst=kept, op=BinaryOp.BITWISE_AND, left=bit, right=IRConst(sum(1 << n for n in numbers), Type.INT)),
            IRBinOp(dst=result, op=BinaryOp.NOT_EQUAL, left=kept, right=IRConst(0, Type.INT)),
        ], result

    def assign(self, target, value_expr) -> list:
        if _scalar(target.type):
            value_ir, value = self.value(value_expr)
            if isinstance(target, t.Local):
                temp, heap = self.bind(target.symbol)
                if heap:
                    return value_ir + [IRStore(address=temp, value=value, value_type=target.type)]
                return value_ir + [IRMove(dst=temp, src=value)]
            if isinstance(target, t.DictLookup):
                value_ir2, address = self.scratch_address(target.type)
                return value_ir + value_ir2 + [
                    IRStore(address=address, value=value, value_type=target.type)
                ] + self.dict_set(target, address)
            ir, address = self.place_address(target)
            return value_ir + ir + [IRStore(address=address, value=value, value_type=target.type)]
        # A composite. Copied directly from another place: two places of one type are the same
        # storage or disjoint (a type can't contain itself by value), so copying never reads what it
        # has already written. Unless evaluating the target can run code that changes the source,
        # or the source is a dict entry the target's insertion could move. Any other value may read
        # the target (`p = P(p.y, p.x)`), so it is built completely before the target changes.
        direct = (isinstance(value_expr, _PLACE_NODES) and not _calls_anything(target)
                  and not (isinstance(value_expr, t.DictLookup) and isinstance(target, t.DictLookup)))
        if not direct and target.type.kind in (TypeKind.STR, TypeKind.SLICE) and not isinstance(target, t.DictLookup):
            # A str or slice is computed as its words, in temps, before anything is written: no temporary.
            if target.type.kind == TypeKind.STR:
                value_ir, ptr, length = self.str_value(value_expr)
                ir, address = self.place_address(target)
                return value_ir + ir + self.write_str(address, ptr, length)
            value_ir, ptr, length, cap = self.slice_value(value_expr)
            ir, address = self.place_address(target)
            return value_ir + ir + self.write_slice(address, ptr, length, cap)
        value_ir, value_address = self.address(value_expr, fresh=not direct)
        if isinstance(target, t.DictLookup):
            return value_ir + self.dict_set(target, value_address)
        ir, address = self.place_address(target)
        return value_ir + ir + [IRCopy(dst_address=address, src_address=value_address, value_type=target.type)]

    # -- places: where a value is stored

    def place_address(self, e) -> tuple:
        """The address of an assignable place (a variable, field, element, or pointee)."""
        if isinstance(e, t.Local):
            if _scalar(e.type):
                temp, heap = self.bind(e.symbol)
                if heap:
                    return [], temp
                address = self.temp()
                return [IRLocalAddress(dst=address, slot=self.ir_fn.var_slots[e.symbol.id])], address
            return self.local_address(e.symbol)
        if isinstance(e, t.Deref):
            return self.pointer(e.pointer, e)
        if isinstance(e, t.FieldAccess):
            if e.through_pointer:
                ir, base = self.pointer(e.base, e)
            else:
                ir, base = self.address(e.base)
            offset_ir, address = self.offset(base, self.field_offset(e))
            return ir + offset_ir, address
        if isinstance(e, t.ArrayIndex):
            ir, base = self.address(e.base)
            index_ir, index = self.value(e.index)
            element_ir, address = self.element(base, index, e.type)
            return ir + index_ir + [
                IRBoundsCheck(index=index, length=IRConst(e.base.type.size, Type.INT), where=e.where)
            ] + element_ir, address
        if isinstance(e, t.SliceIndex):
            ir, ptr, length, _ = self.slice_value(e.base)
            index_ir, index = self.value(e.index)
            element_ir, address = self.element(ptr, index, e.type)
            return ir + index_ir + [IRBoundsCheck(index=index, length=length, where=e.where)] + element_ir, address
        if isinstance(e, t.Payload):
            ir, base = self.address(e.sum)
            if isinstance(e.sum, t.Local) and e.sum.symbol.id in self.addressed:
                ir = ir + self.variant_check(base, e)
            offset_ir, address = self.offset(base, SUM_TYPE_TAG_WIDTH)
            return ir + offset_ir, address
        if isinstance(e, t.DictLookup):
            ir, address = self.dict_call(e.dict, e.key, 'lookup', result_type=Type.INT64)
            return ir + self.panic_when_missing(address, "dict lookup: key not found", e.dict, e), address
        raise NotYetPorted(f"place {type(e).__name__}")

    def variant_check(self, base, e: t.Payload) -> list:
        """Panic unless the sum at `base` still holds the variant `e` was narrowed to: a pointer to
        the variable may have changed it since the `is` check."""
        variants = self.ir_program.sum_type_registry[e.sum.type.sum_type_name].variants
        tag, changed = self.temp(Type.INT32), self.temp(Type.BOOL)
        return [
            IRLoad(dst=tag, address=base),
            IRBinOp(dst=changed, op=BinaryOp.NOT_EQUAL, left=tag, right=IRConst(variants.index(e.type), Type.INT32))
        ] + self.panic_when(changed, f"'{e.sum.symbol.name}' changed variant while narrowed", e)

    def field_offset(self, e: t.FieldAccess) -> int:
        struct_type = e.base.type.element_type if e.through_pointer else e.base.type
        offset = 0
        for name, field_type in self.ir_program.struct_registry[struct_type.struct_name].fields.items():
            if name == e.name:
                return offset
            offset += self.width(field_type)
        raise NotYetPorted(f"no field {e.name}")

    def element(self, base, index, element_type) -> tuple:
        """The address of element `index` (already bounds-checked) from `base`."""
        scaled, address = self.temp(Type.INT), self.temp()
        return [
            IRBinOp(dst=scaled, op=BinaryOp.MULTIPLY, left=index, right=IRConst(self.width(element_type), Type.INT)),
            IRBinOp(dst=address, op=BinaryOp.ADD, left=base, right=scaled),
        ], address

    # -- composite values

    def address(self, e, fresh: bool = False) -> tuple:
        """Where composite value `e` is. A place is used in place unless `fresh` asks for a copy;
        anything else is built in a new temporary."""
        if not fresh and isinstance(e, _PLACE_NODES):
            return self.place_address(e)
        ir, address = self.scratch_address(e.type)
        return ir + self.write_into(address, e), address

    def changed_by(self, e, later) -> bool:
        """Whether evaluating `later` could change composite `e`, evaluated before it: `later` calls
        something, and `e` is a place a call can reach (through a pointer, slice, or dict, or in a
        variable one may reach). Its value was fixed when it was evaluated, so it is copied then."""
        if not isinstance(e, _PLACE_NODES) or not _calls_anything(later):
            return False
        symbol = _variable(e)
        return symbol is None or symbol.id in self.exposed

    def stored(self, e) -> tuple:
        """The address of a str or slice held in memory: a place, or a call's result."""
        if isinstance(e, _PLACE_NODES):
            return self.place_address(e)
        if isinstance(e, t.Call):
            ir, address = self.scratch_address(e.type)
            return ir + self.call(e, False, destination=address)[0], address
        raise NotYetPorted(f"a {e.type} from {type(e).__name__}")

    def write_into(self, dst, e) -> list:
        """Store composite value `e` at `dst` (new storage, or a place `e` can't be part of)."""
        kind = e.type.kind
        if isinstance(e, _PLACE_NODES) and kind in (TypeKind.STR, TypeKind.SLICE):  # the descriptor, as a block
            ir, source = self.place_address(e)
            return ir + [IRCopy(dst_address=dst, src_address=source, value_type=e.type)]
        if kind == TypeKind.STR:
            ir, ptr, length = self.str_value(e)
            return ir + self.write_str(dst, ptr, length)
        if kind == TypeKind.SLICE:
            ir, ptr, length, cap = self.slice_value(e)
            return ir + self.write_slice(dst, ptr, length, cap)
        if not _ported(e.type):
            raise NotYetPorted(f"a {e.type} value")
        if isinstance(e, _PLACE_NODES):
            ir, source = self.address(e)
            return ir + [IRCopy(dst_address=dst, src_address=source, value_type=e.type)]
        if isinstance(e, t.StructLiteral):
            ir, offset = [], 0
            for (name, field_type), value in zip(
                    self.ir_program.struct_registry[e.type.struct_name].fields.items(), e.fields):
                offset_ir, address = self.offset(dst, offset)
                ir += offset_ir + self.store(address, value)
                offset += self.width(field_type)
            return ir
        if isinstance(e, t.ArrayLiteral):
            ir, width = [], self.width(e.type.element_type)
            for i, value in enumerate(e.elements):
                offset_ir, address = self.offset(dst, i * width)
                ir += offset_ir + self.store(address, value)
            return ir
        if isinstance(e, t.WidenToSum):
            variants = self.ir_program.sum_type_registry[e.type.sum_type_name].variants
            ir = [IRStore(address=dst, value=IRConst(variants.index(e.value.type), Type.INT32), value_type=Type.INT32)]
            if e.value.type == Type.NONE:
                return ir
            offset_ir, payload = self.offset(dst, SUM_TYPE_TAG_WIDTH)
            return ir + offset_ir + self.store(payload, e.value)
        if isinstance(e, t.WidenSum):
            return self.widen_sum(dst, e)
        if isinstance(e, t.NarrowSum):
            return self.narrow_sum(dst, e)
        if isinstance(e, t.ZeroValue):
            return self.zero_into(dst, e.type)
        if isinstance(e, t.NewEmptyDict):
            return self.zero_into(dst, e.type)
        if isinstance(e, t.DictLiteral):
            return self.dict_literal(dst, e)
        if isinstance(e, t.Call):
            return self.call(e, False, destination=dst)[0]
        raise NotYetPorted(f"composite {type(e).__name__}")

    def widen_sum(self, dst, e: t.WidenSum) -> list:
        """A sum's value at `dst` as the wider sum e.type. The wider sum has room for it (its payload
        is at least as large), so it is copied whole; then its tag is rewritten where the two sums
        number that variant differently, through a table from the one's numbers to the other's."""
        narrow = self.ir_program.sum_type_registry[e.value.type.sum_type_name].variants
        wide = self.ir_program.sum_type_registry[e.type.sum_type_name].variants
        ir, source = self.address(e.value)
        ir = ir + [IRCopy(dst_address=dst, src_address=source, value_type=e.value.type)]
        numbers = [wide.index(variant) for variant in narrow]
        if numbers == list(range(len(narrow))):
            return ir
        tables = self.ir_program.__dict__.setdefault('_sum_tag_tables', {})
        if (e.value.type, e.type) not in tables:
            tables[(e.value.type, e.type)] = self.ids.new_label("sum_tags")
            self.ir_program.type_descriptors.append((tables[(e.value.type, e.type)], numbers))
        tag, index, offset = self.temp(Type.INT32), self.temp(Type.INT), self.temp(Type.INT)
        base, entry, number, new_tag = self.temp(), self.temp(), self.temp(Type.INT), self.temp(Type.INT32)
        return ir + [
            IRLoad(dst=tag, address=source),
            IRCast(dst=index, src=tag),
            IRBinOp(dst=offset, op=BinaryOp.MULTIPLY, left=index, right=IRConst(8, Type.INT)),
            IRStaticDataAddress(dst=base, label=tables[(e.value.type, e.type)]),
            IRBinOp(dst=entry, op=BinaryOp.ADD, left=base, right=offset),
            IRLoad(dst=number, address=entry),
            IRCast(dst=new_tag, src=number),
            IRStore(address=dst, value=new_tag, value_type=Type.INT32),
        ]

    def narrow_sum(self, dst, e: t.NarrowSum) -> list:
        """A sum variable's value at `dst` as the narrower sum e.type, which the checker has shown has
        the variant it holds. That variant fits (it is one of the narrower sum's), so the narrower
        sum's width is copied, and the tag is translated through a table, in which a variant the
        narrower sum lacks is -1. Reaching one of those means a pointer to the variable changed it
        after the `is` check, which is checked for where there can be such a pointer."""
        wide = self.ir_program.sum_type_registry[e.value.type.sum_type_name].variants
        narrow = self.ir_program.sum_type_registry[e.type.sum_type_name].variants
        numbers = [narrow.index(variant) if variant in narrow else -1 for variant in wide]
        ir, source = self.address(e.value)
        ir = ir + [IRCopy(dst_address=dst, src_address=source, value_type=e.type)]
        tables = self.ir_program.__dict__.setdefault('_sum_tag_tables', {})
        if (e.value.type, e.type) not in tables:
            tables[(e.value.type, e.type)] = self.ids.new_label("sum_tags")
            self.ir_program.type_descriptors.append((tables[(e.value.type, e.type)], numbers))
        tag, index, offset = self.temp(Type.INT32), self.temp(Type.INT), self.temp(Type.INT)
        base, entry, number, new_tag = self.temp(), self.temp(), self.temp(Type.INT), self.temp(Type.INT32)
        ir += [
            IRLoad(dst=tag, address=source),
            IRCast(dst=index, src=tag),
            IRBinOp(dst=offset, op=BinaryOp.MULTIPLY, left=index, right=IRConst(8, Type.INT)),
            IRStaticDataAddress(dst=base, label=tables[(e.value.type, e.type)]),
            IRBinOp(dst=entry, op=BinaryOp.ADD, left=base, right=offset),
            IRLoad(dst=number, address=entry),
        ]
        if isinstance(e.value, t.Local) and e.value.symbol.id in self.addressed:
            changed = self.temp(Type.BOOL)
            ir += [IRBinOp(dst=changed, op=BinaryOp.LESS_THAN, left=number, right=IRConst(0, Type.INT))]
            ir += self.panic_when(changed, f"'{e.value.symbol.name}' changed variant while narrowed", e)
        return ir + [IRCast(dst=new_tag, src=number), IRStore(address=dst, value=new_tag, value_type=Type.INT32)]

    def store(self, address, value) -> list:
        """Store any value (scalar or composite) at `address`."""
        if _scalar(value.type):
            ir, v = self.value(value)
            return ir + [IRStore(address=address, value=v, value_type=value.type)]
        return self.write_into(address, value)

    def zero_into(self, dst, type_: Type) -> list:
        kind = type_.kind
        if kind == TypeKind.SUM:  # its `none` variant
            variants = self.ir_program.sum_type_registry[type_.sum_type_name].variants
            return [IRStore(address=dst, value=IRConst(variants.index(Type.NONE), Type.INT32), value_type=Type.INT32)]
        if kind == TypeKind.STR:
            ir, ptr, length = self.str_value(t.StrLit(Type.STR, ''))
            return ir + self.write_str(dst, ptr, length)
        if kind == TypeKind.SLICE:
            return self.write_slice(dst, IRConst(0, Type.INT64), IRConst(0, Type.INT), IRConst(0, Type.INT))
        if kind == TypeKind.STRUCT:
            ir, offset = [], 0
            for field_type in self.ir_program.struct_registry[type_.struct_name].fields.values():
                offset_ir, address = self.offset(dst, offset)
                ir += offset_ir + self.zero_into(address, field_type)
                offset += self.width(field_type)
            return ir
        if kind == TypeKind.ARRAY:
            return self.zero_array(dst, type_)
        if kind == TypeKind.DICT:  # a new empty dict
            header = self.temp()
            return [
                IRCall(
                    dst=header, name='hornet_alloc_zeroed',
                    args=[IRConst(1, Type.INT64), IRConst(_DICT_HEADER_SIZE, Type.INT64)]
                ),
                IRStore(address=dst, value=header, value_type=Type.INT64),
            ]
        return [IRStore(address=dst, value=IRConst(0, type_), value_type=type_)]

    def zero_array(self, dst, type_: Type) -> list:
        i, cond, scaled, element = self.temp(Type.INT), self.temp(Type.BOOL), self.temp(Type.INT), self.temp()
        start, body, end = (self.ids.new_label(x) for x in ("zero_start", "zero_body", "zero_end"))
        width = self.width(type_.element_type)
        return [
            IRMove(dst=i, src=IRConst(0, Type.INT)),
            IRJump(start), IRLabel(start),
            IRBinOp(dst=cond, op=BinaryOp.LESS_THAN, left=i, right=IRConst(type_.size, Type.INT)),
            IRBranch(cond=cond, true_label=body, false_label=end), IRLabel(body),
            IRBinOp(dst=scaled, op=BinaryOp.MULTIPLY, left=i, right=IRConst(width, Type.INT)),
            IRBinOp(dst=element, op=BinaryOp.ADD, left=dst, right=scaled)
        ] + self.zero_into(element, type_.element_type) + [
                IRBinOp(dst=i, op=BinaryOp.ADD, left=i, right=IRConst(1, Type.INT)), IRJump(start), IRLabel(end)
        ]

    # -- strings: (ptr, len)

    def write_str(self, dst, ptr, length) -> list:
        ir, len_address = self.offset(dst, 8)
        return [IRStore(address=dst, value=ptr, value_type=Type.INT64)] + ir + [
            IRStore(address=len_address, value=length, value_type=Type.INT)]

    def read_str(self, address) -> tuple:
        ptr, length = self.temp(), self.temp(Type.INT)
        ir, len_address = self.offset(address, 8)
        return [IRLoad(dst=ptr, address=address)] + ir + [IRLoad(dst=length, address=len_address)], ptr, length

    def str_value(self, e) -> tuple:
        if isinstance(e, t.ZeroValue):
            e = t.StrLit(Type.STR, '')
        if isinstance(e, t.StrLit):
            ptr, label = self.temp(), self.ids.new_label("str")
            self.ir_program.string_literals.append((label, e.value))
            return [IRStaticDataAddress(dst=ptr, label=label)], ptr, IRConst(len(e.value), Type.INT)
        if isinstance(e, t.StrConcat):
            left_ir, left_ptr, left_len = self.str_value(e.left)
            right_ir, right_ptr, right_len = self.str_value(e.right)
            concat_ir, buffer, total = self.concat(left_ptr, left_len, right_ptr, right_len)
            return left_ir + right_ir + concat_ir, buffer, total
        if isinstance(e, t.SliceOf):
            base_ir, ptr, length = self.str_value(e.base)
            bounds_ir, low, high = self.bounds(e, length)
            new_len, new_ptr = self.temp(Type.INT), self.temp()
            return base_ir + bounds_ir + [
                IRBinOp(dst=new_len, op=BinaryOp.SUBTRACT, left=high, right=low),
                IRBinOp(dst=new_ptr, op=BinaryOp.ADD, left=ptr, right=low),
            ], new_ptr, new_len
        if isinstance(e, t.StrFromByte):
            label = getattr(self.ir_program, 'byte_table_label', None)
            if label is None:
                label = self.ids.new_label("byte_table")
                self.ir_program.string_literals.append((label, ''.join(chr(i) for i in range(256))))
                self.ir_program.byte_table_label = label
            value_ir, value = self.value(e.value)
            base, offset, ptr = self.temp(), self.temp(), self.temp()
            return value_ir + [
                IRStaticDataAddress(dst=base, label=label), IRCast(dst=offset, src=value),
                IRBinOp(dst=ptr, op=BinaryOp.ADD, left=base, right=offset),
            ], ptr, IRConst(1, Type.INT)
        if isinstance(e, t.EnumName):
            # A static table of the members' names, each a str's (ptr, len), indexed by the value.
            members = self.ir_program.enum_registry[e.value.type.enum_name].members
            tables = self.ir_program.__dict__.setdefault('_enum_name_tables', {})
            if e.value.type not in tables:
                words = []
                for member in members:
                    name_label = self.ids.new_label("enum_name")
                    self.ir_program.string_literals.append((name_label, member))
                    words += [name_label, len(member)]
                tables[e.value.type] = self.ids.new_label("enum_names")
                self.ir_program.type_descriptors.append((tables[e.value.type], words))
            value_ir, value = self.value(e.value)
            base, index, offset, entry = self.temp(), self.temp(Type.INT), self.temp(Type.INT), self.temp()
            read_ir, ptr, length = self.read_str(entry)
            return value_ir + [
                IRStaticDataAddress(dst=base, label=tables[e.value.type]),
                IRCast(dst=index, src=value),
                IRBinOp(dst=offset, op=BinaryOp.MULTIPLY, left=index, right=IRConst(16, Type.INT)),
                IRBinOp(dst=entry, op=BinaryOp.ADD, left=base, right=offset),
            ] + read_ir, ptr, length
        if isinstance(e, t.Format):
            return self.format(e)
        if isinstance(e, t.StrFromBytes):
            slice_ir, src, length, _ = self.slice_value(e.value)
            size, ptr = self.temp(Type.INT), self.temp()
            return slice_ir + [
                IRBinOp(dst=size, op=BinaryOp.BITWISE_OR, left=length, right=IRConst(1, Type.INT)),  # never malloc(0)
                IRCall(dst=ptr, name='hornet_alloc', args=[size]),
                IRCall(dst=None, name='memcpy', args=[ptr, src, length]),
            ], ptr, length
        if isinstance(e, t.StrFromRawParts):
            ptr_ir, ptr = self.value(e.ptr)
            len_ir, length = self.value(e.length)
            return ptr_ir + len_ir, ptr, length
        ir, address = self.stored(e)
        read_ir, ptr, length = self.read_str(address)
        return ir + read_ir, ptr, length

    def bounds(self, e: t.SliceOf, length, limit=None) -> tuple:
        """(IR checking low <= high <= limit, low, high) for `x[low:high]`; high defaults to the
        length, and the limit (a slice's capacity) to the length."""
        against = '' if limit is None else ' capacity'  # what a failed check names the limit
        limit = length if limit is None else limit
        ir = []
        if e.low is not None:
            low_ir, low = self.value(e.low)
            ir += low_ir
        else:
            low = IRConst(0, Type.INT)
        if e.high is not None:
            high_ir, high = self.value(e.high)
            ir += high_ir
        else:
            high = length
        return ir + [
            IRSliceBoundsCheck(value=low, bound=limit, where=e.where, part='start' + against),
            IRSliceBoundsCheck(value=high, bound=limit, where=e.where, part='end' + against),
            IRSliceBoundsCheck(value=low, bound=high, where=e.where, part='order')
        ], low, high

    # -- slices: (ptr, len, cap)

    def write_slice(self, dst, ptr, length, cap) -> list:
        len_ir, len_address = self.offset(dst, 8)
        cap_ir, cap_address = self.offset(dst, 16)
        return [
            IRStore(address=dst, value=ptr, value_type=Type.INT64)
        ] + len_ir + [
            IRStore(address=len_address, value=length, value_type=Type.INT)
        ] + cap_ir + [
            IRStore(address=cap_address, value=cap, value_type=Type.INT)
        ]

    def slice_value(self, e) -> tuple:
        if isinstance(e, t.EmptySlice):
            return [], IRConst(0, Type.INT64), IRConst(0, Type.INT), IRConst(0, Type.INT)
        if isinstance(e, t.SliceLiteral):
            element_type = e.type.element_type
            count = len(e.elements)
            ptr = self.temp()
            ir = [IRCall(dst=ptr, name='hornet_alloc', args=[IRConst(count * self.width(element_type), Type.INT64)])]
            for i, value in enumerate(e.elements):
                offset_ir, address = self.offset(ptr, i * self.width(element_type))
                ir += offset_ir + self.store(address, value)
            return ir, ptr, IRConst(count, Type.INT), IRConst(count, Type.INT)
        if isinstance(e, t.SliceOf) and e.kind in ('array', 'slice'):
            if e.kind == 'array':
                if isinstance(e.base, _PLACE_NODES):
                    ir, ptr = self.address(e.base)
                else:  # a temporary array: the slice may outlive this statement, so its storage is on the heap
                    ptr = self.temp()
                    ir = [
                             IRCall(dst=ptr, name='hornet_alloc', args=[IRConst(self.width(e.base.type), Type.INT64)])
                         ] + self.write_into(ptr, e.base)
                length = cap = IRConst(e.base.type.size, Type.INT)
            else:
                ir, ptr, length, cap = self.slice_value(e.base)
            # A slice may extend up to the capacity, as in Go (an array's is its length).
            bounds_ir, low, high = self.bounds(e, length, cap if e.kind == 'slice' else None)
            width = self.width(e.type.element_type)
            scaled = self.temp(Type.INT)
            new_ptr = self.temp()
            new_len = self.temp(Type.INT)
            new_cap = self.temp(Type.INT)
            return ir + bounds_ir + [
                IRBinOp(dst=scaled, op=BinaryOp.MULTIPLY, left=low, right=IRConst(width, Type.INT)),
                IRBinOp(dst=new_ptr, op=BinaryOp.ADD, left=ptr, right=scaled),
                IRBinOp(dst=new_len, op=BinaryOp.SUBTRACT, left=high, right=low),
                IRBinOp(dst=new_cap, op=BinaryOp.SUBTRACT, left=cap, right=low),
            ], new_ptr, new_len, new_cap
        if isinstance(e, t.Append):
            return self.append(e)
        if isinstance(e, t.BytesFromStr):
            str_ir, ptr, length = self.str_value(e.value)
            scratch_ir, address = self.scratch_address(e.type)
            ir = str_ir + scratch_ir + [IRCall(dst=None, name='hornet_bytes', args=[address, ptr, length])]
            read_ir, ptr, length, cap = self.read_slice(address)
            return ir + read_ir, ptr, length, cap
        ir, address = self.stored(e)
        ptr, length, cap = self.temp(), self.temp(Type.INT), self.temp(Type.INT)
        len_ir, len_address = self.offset(address, 8)
        cap_ir, cap_address = self.offset(address, 16)
        return ir + [IRLoad(dst=ptr, address=address)] + len_ir + [
            IRLoad(dst=length, address=len_address)
        ] + cap_ir + [IRLoad(dst=cap, address=cap_address)], ptr, length, cap

    # -- scalar values

    def pointer(self, expr, at) -> tuple:
        """A pointer about to be dereferenced by typed node `at`: checked not to be none."""
        ir, value = self.value(expr)
        return ir + [IRNullCheck(value, where=at.where)], value

    def value(self, e, discard: bool = False) -> tuple:
        """(IR, value) for a scalar expression (or a void call when `discard`)."""
        if isinstance(e, (t.StrRawPtr, t.StrRawLen)):
            ir, ptr, length = self.str_value(e.value)
            return ir, ptr if isinstance(e, t.StrRawPtr) else length
        if isinstance(e, t.Call):
            return self.call(e, discard)
        if isinstance(e, t.Print):
            return self.print_(e), None
        if isinstance(e, t.Panic):
            return self.panic(e), None
        if isinstance(e, t.DictDelete):
            ir, removed = self.dict_call(e.dict, e.key, 'delete', result_type=Type.INT)
            return ir + self.panic_when_missing(removed, "dict delete: key not found", e.dict, e), None
        if isinstance(e, t.DictContains):
            ir, found = self.dict_call(e.dict, e.key, 'contains', result_type=Type.INT)
            result = self.temp(Type.BOOL)
            return ir + [IRBinOp(dst=result, op=BinaryOp.NOT_EQUAL, left=found, right=IRConst(0, Type.INT))], result
        if isinstance(e, t.ElementContains):
            return self.element_contains(e)
        if not _scalar(e.type):
            raise NotYetPorted(f"a {e.type} value ({type(e).__name__})")
        if isinstance(e, t.IntLit):
            return [], IRConst(e.value, e.type)
        if isinstance(e, t.EnumMember):
            return [], IRConst(e.index, Type.INT32)
        if isinstance(e, t.Bind):
            test_ir, test = self.value(e.test)
            return self.initialize(e.symbol, e.value) + test_ir, test
        if isinstance(e, (t.EnumFromInt, t.EnumContains)):
            return self.enum_from_int(e) if isinstance(e, t.EnumFromInt) else self.enum_contains(e)
        if isinstance(e, t.BoolLit):
            return [], IRConst(1 if e.value else 0, Type.BOOL)
        if isinstance(e, t.NoneLit):
            return [], IRConst(0, Type.INT64)  # a null pointer is the address 0
        if isinstance(e, t.ZeroValue):
            return [], IRConst(0, Type.INT64 if e.type.kind == TypeKind.POINTER else e.type)
        if isinstance(e, t.Local):
            temp, heap = self.bind(e.symbol)
            if heap:
                loaded = self.temp(e.type)
                return [IRLoad(dst=loaded, address=temp)], loaded
            if e.symbol.id in self.addressed:  # what is evaluated later may change it through a pointer
                copy = self.temp(e.type)
                return [IRMove(dst=copy, src=temp)], copy
            return [], temp
        if isinstance(e, (t.Deref, t.FieldAccess, t.ArrayIndex, t.SliceIndex, t.Payload, t.DictLookup)):
            ir, address = self.place_address(e)
            loaded = self.temp(e.type)
            return ir + [IRLoad(dst=loaded, address=address)], loaded
        if isinstance(e, t.StrIndex):
            ir, ptr, length = self.str_value(e.base)
            index_ir, index = self.value(e.index)
            address, loaded = self.temp(), self.temp(Type.UINT8)
            return ir + index_ir + [
                IRBoundsCheck(index=index, length=length, where=e.where),
                IRBinOp(dst=address, op=BinaryOp.ADD, left=ptr, right=index),
                IRLoad(dst=loaded, address=address)
            ], loaded
        if isinstance(e, t.Len):
            kind = e.value.type.kind
            if kind == TypeKind.ARRAY:
                ir = [] if isinstance(e.value, t.Local) else self.address(e.value)[0]  # still evaluated
                return ir, IRConst(e.value.type.size, Type.INT)
            if kind == TypeKind.STR:
                ir, _, length = self.str_value(e.value)
                return ir, length
            if kind == TypeKind.SLICE:
                ir, _, length, _ = self.slice_value(e.value)
                return ir, length
            ir, header = self.dict_header(e.value)
            count_ir, count_address = self.offset(header, 8)
            length = self.temp(Type.INT)
            return ir + count_ir + [IRLoad(dst=length, address=count_address)], length
        if isinstance(e, t.TagTest):
            ir, address = self.address(e.sum)
            tag = self.temp(Type.INT32)
            test_ir, result = self.tag_is(tag, e.sum.type, e.variant)
            return ir + [IRLoad(dst=tag, address=address)] + test_ir, result
        if isinstance(e, t.StrCompare):
            return self.str_compare(e)
        if isinstance(e, t.AddressOf):
            return self.address_of(e)
        if isinstance(e, t.BoxVariant):
            box = self.temp()
            return [IRCall(dst=box, name='hornet_alloc', args=[IRConst(self.width(e.value.type), Type.INT64)])] + \
                self.write_into(box, e.value), box
        if isinstance(e, t.Unary):
            ir, operand = self.value(e.operand)
            result = self.temp(e.type)
            return ir + [IRUnOp(dst=result, op=e.op, operand=operand)], result
        if isinstance(e, t.IntCast):
            ir, operand = self.value(e.value)
            result = self.temp(e.type)
            return ir + [IRCast(dst=result, src=operand)], result
        if isinstance(e, t.Binary):
            if e.op in (BinaryOp.AND, BinaryOp.OR):
                return self.short_circuit(e)
            if not _scalar(e.left.type):
                return self.composite_equality(e)
            left_ir, left = self.value(e.left)
            right_ir, right = self.value(e.right)
            result = self.temp(e.type)
            return left_ir + right_ir + [IRBinOp(dst=result, op=e.op, left=left, right=right, where=e.where)], result
        raise NotYetPorted(type(e).__name__)

    def address_of(self, e: t.AddressOf) -> tuple:
        place = e.place
        if isinstance(place, t.StructLiteral):
            # `&S(...)`: its own storage, on the heap when the pointer may outlive this frame.
            box = self.temp()
            return [IRCall(dst=box, name='hornet_alloc', args=[IRConst(self.width(place.type), Type.INT64)])] + \
                self.write_into(box, place), box
        if isinstance(place, t.Local) and _scalar(place.type):
            temp, heap = self.bind(place.symbol)
            if heap:
                return [], temp
        return self.place_address(place)

    def str_compare(self, e: t.StrCompare) -> tuple:
        left_ir, left_ptr, left_len = self.str_value(e.left)
        right_ir, right_ptr, right_len = self.str_value(e.right)
        if e.op not in (BinaryOp.EQUAL, BinaryOp.NOT_EQUAL):
            order_ir, result = self.str_order(e.op, left_ptr, left_len, right_ptr, right_len)
            return left_ir + right_ir + order_ir, result
        same_length, cmp, result = self.temp(Type.BOOL), self.temp(Type.INT32), self.temp(Type.BOOL)
        equal_len, differ, end = (self.ids.new_label(x) for x in ("str_len_equal", "str_len_differ", "str_cmp_end"))
        return left_ir + right_ir + [
            IRBinOp(dst=same_length, op=BinaryOp.EQUAL, left=left_len, right=right_len),
            IRBranch(cond=same_length, true_label=equal_len, false_label=differ),
            IRLabel(equal_len),
            IRCall(dst=cmp, name='memcmp', args=[left_ptr, right_ptr, left_len]),
            IRBinOp(dst=result, op=e.op, left=cmp, right=IRConst(0, Type.INT32)),
            IRJump(end),
            IRLabel(differ),
            IRMove(dst=result, src=IRConst(0 if e.op == BinaryOp.EQUAL else 1, Type.BOOL)),
            IRJump(end),
            IRLabel(end)
        ], result

    def str_order(self, op, left_ptr, left_len, right_ptr, right_len) -> tuple:
        """`<`, `>`, `<=`, or `>=` of two strs, byte by byte: the first byte that differs decides
        (memcmp, over the length they share), and if none does, the shorter str comes first."""
        shared, cmp = self.temp(Type.INT), self.temp(Type.INT32)
        left_is_shorter, differ, result = self.temp(Type.BOOL), self.temp(Type.BOOL), self.temp(Type.BOOL)
        use_right, compare, by_bytes, by_length, end = (self.ids.new_label(x) for x in (
            "str_order_right", "str_order_compare", "str_order_bytes", "str_order_length", "str_order_end"))
        return [
            IRMove(dst=shared, src=left_len),
            IRBinOp(dst=left_is_shorter, op=BinaryOp.LESS_THAN_OR_EQUAL, left=left_len, right=right_len),
            IRBranch(cond=left_is_shorter, true_label=compare, false_label=use_right),
            IRLabel(use_right),
            IRMove(dst=shared, src=right_len),
            IRJump(compare),
            IRLabel(compare),
            IRCall(dst=cmp, name='memcmp', args=[left_ptr, right_ptr, shared]),
            IRBinOp(dst=differ, op=BinaryOp.NOT_EQUAL, left=cmp, right=IRConst(0, Type.INT32)),
            IRBranch(cond=differ, true_label=by_bytes, false_label=by_length),
            IRLabel(by_bytes),
            IRBinOp(dst=result, op=op, left=cmp, right=IRConst(0, Type.INT32)),
            IRJump(end),
            IRLabel(by_length),
            IRBinOp(dst=result, op=op, left=left_len, right=right_len),
            IRJump(end),
            IRLabel(end),
        ], result

    def short_circuit(self, e: t.Binary) -> tuple:
        """`and`/`or` as a value: branch on it (see `branch`), then set the result to 1 or 0."""
        yes, no, end = (self.ids.new_label(x) for x in ("logic_true", "logic_false", "logic_end"))
        result = self.temp(Type.BOOL)
        return self.branch(e, yes, no) + [
            IRLabel(yes),
            IRMove(dst=result, src=IRConst(1, Type.BOOL)),
            IRJump(end),
            IRLabel(no),
            IRMove(dst=result, src=IRConst(0, Type.BOOL)),
            IRJump(end),
            IRLabel(end)
        ], result

    def branch(self, e, if_true: str, if_false: str) -> list:
        """Jump to `if_true` or `if_false` on bool `e`, without computing its value where it's made of
        `and`, `or`, `not`, and literals: those become jumps (short-circuiting), and a comparison
        feeding the final branch is fused with it by the backends."""
        if isinstance(e, t.Binary) and e.op in (BinaryOp.AND, BinaryOp.OR):
            rhs = self.ids.new_label("logic_rhs")
            if e.op == BinaryOp.AND:
                left = self.branch(e.left, rhs, if_false)
            else:
                left = self.branch(e.left, if_true, rhs)
            return left + [IRLabel(rhs)] + self.branch(e.right, if_true, if_false)
        if isinstance(e, t.Unary) and e.op == UnaryOp.NOT:
            return self.branch(e.operand, if_false, if_true)
        if isinstance(e, t.BoolLit):
            return [IRJump(if_true if e.value else if_false)]
        if isinstance(e, t.Bind):
            return self.initialize(e.symbol, e.value) + self.branch(e.test, if_true, if_false)
        ir, cond = self.value(e)
        return ir + [IRBranch(cond=cond, true_label=if_true, false_label=if_false)]

    def call(self, e: t.Call, discard: bool, destination=None) -> tuple:
        """A call; a composite result is written to `destination` (or a new temporary)."""
        if not _ported(e.type):
            raise NotYetPorted(f"call returning {e.type}")
        ir, args = [], []
        composite_result = not _valueless(e.type) and not _scalar(e.type)
        if composite_result:
            if destination is None:
                scratch_ir, destination = self.scratch_address(e.type)
                ir += scratch_ir
            args.append(destination)
        for i, a in enumerate(e.args):
            kind = a.type.kind
            if _scalar(a.type):
                arg_ir, value = self.value(a)
                ir += arg_ir
                args.append(value)
            elif kind == TypeKind.STR:
                arg_ir, ptr, length = self.str_value(a)
                ir += arg_ir
                args += [ptr, length]
            elif kind == TypeKind.SLICE:
                arg_ir, ptr, length, cap = self.slice_value(a)
                ir += arg_ir
                args += [ptr, length, cap]
            elif kind in _COMPOSITE_BY_ADDRESS:
                # The callee copies it, on entry. A later argument that calls could change it
                # before then, so in that case its value is copied here.
                arg_ir, address = self.address(a, fresh=self.changed_by(a, e.args[i + 1:]))
                ir += arg_ir
                args.append(address)
            else:
                raise NotYetPorted(f"argument of type {a.type}")
        result = self.temp(e.type) if _scalar(e.type) else None
        ir.append(IRCall(dst=result, name=e.name if e.kind == 'extern' else link_name(e.name), args=args))
        if e.type == Type.NEVER:
            ir.append(self.never_returned(e))
        return ir, destination if composite_result else result

    def enum_range(self, value_expr, enum: Type) -> tuple:
        """(IR, the integer as an int, below zero?, past the last member?) for a value tested against
        `enum`'s members, whose values are 0 up to their count."""
        ir, value = self.value(value_expr)
        wide, negative, too_big = self.temp(Type.INT), self.temp(Type.BOOL), self.temp(Type.BOOL)
        count = len(self.ir_program.enum_registry[enum.enum_name].members)
        return (
            ir + [IRCast(dst=wide, src=value)],
            wide,
            IRBinOp(dst=negative, op=BinaryOp.LESS_THAN, left=wide, right=IRConst(0, Type.INT)),
            IRBinOp(dst=too_big, op=BinaryOp.GREATER_THAN_OR_EQUAL, left=wide, right=IRConst(count, Type.INT)),
        )

    def enum_from_int(self, e: t.EnumFromInt) -> tuple:
        """A member's value is its position, so the integer must be one: at least 0 and below the
        count, which is what a bounds check tests. Its panic names the integer."""
        ir, value = self.value(e.value)
        wide, result = self.temp(Type.INT), self.temp(e.type)
        count = len(self.ir_program.enum_registry[e.type.enum_name].members)
        return ir + [
            IRCast(dst=wide, src=value),
            IRBoundsCheck(index=wide, length=IRConst(count, Type.INT), where=e.where,
                          message=f"not a member of {e.type}", routine="hornet_panic_enum_value"),
            IRCast(dst=result, src=wide),
        ], result

    def enum_contains(self, e: t.EnumContains) -> tuple:
        ir, _, negative, too_big = self.enum_range(e.value, e.enum)
        result = self.temp(Type.BOOL)
        check_top = self.ids.new_label('in_enum_top')
        end = self.ids.new_label('in_enum_end')
        return ir + [
            negative,
            IRMove(dst=result, src=IRConst(0, Type.BOOL)),
            IRBranch(cond=negative.dst, true_label=end, false_label=check_top), IRLabel(check_top), too_big,
            IRBinOp(dst=result, op=BinaryOp.EQUAL, left=too_big.dst, right=IRConst(0, Type.BOOL)),
            IRJump(end),
            IRLabel(end)
        ], result

    def never_returned(self, at) -> IRJump:
        """What follows a call that doesn't return: a panic, should it return after all."""
        return IRJump(self.panics.label(located("a never function returned", at.where)))

    def panic(self, e: t.Panic) -> list:
        """`panic(message)`: the runtime prints this call's position and the message, and aborts."""
        ir, ptr, length = self.str_value(e.message)
        where = self.temp()
        return ir + [
            IRStaticDataAddress(dst=where, label=message_label(self.ir_program, e.where or "")),
            IRCall(dst=None, name='hornet_panic_at', args=[where, ptr, length]), self.never_returned(e)
        ]

    def format(self, e: t.Format) -> tuple:
        """One buffer, in a frame slot, that the runtime appends each piece to: the template's text,
        a str argument's bytes, any other argument as print shows it. Its first two words are then
        the result's pointer and length (the bytes are the result's to keep)."""
        buffer = self.temp()
        slot = self.ids.new_slot(24, "format", self.ir_fn)
        ir = [IRLocalAddress(dst=buffer, slot=slot), IRCall(dst=None, name='hornet_format_begin', args=[buffer])]

        def text(piece) -> list:
            piece_ir, ptr, length = self.str_value(piece)
            return piece_ir + [IRCall(dst=None, name='hornet_format_text', args=[buffer, ptr, length])]

        pieces = t.format_pieces(e.template)
        for piece, arg in zip(pieces, e.args + (None,)):
            if piece:
                ir += text(t.StrLit(Type.STR, piece))
            if arg is None:
                break
            if arg.type == Type.STR:
                ir += text(arg)
                continue
            if _scalar(arg.type):
                arg_ir, value = self.value(arg)
                slot_ir, address = self.scratch_address(arg.type)
                arg_ir += slot_ir + [IRStore(address=address, value=value, value_type=arg.type)]
            else:
                arg_ir, address = self.address(arg)
            descriptor = self.temp()
            ir += arg_ir + [
                IRStaticDataAddress(dst=descriptor, label=type_descriptor(self.ir_program, arg.type)),
                IRCall(dst=None, name='hornet_format_value', args=[buffer, address, descriptor]),
            ]
        read_ir, ptr, length = self.read_str(buffer)
        return ir + read_ir, ptr, length

    def print_(self, e: t.Print) -> list:
        value_type = e.value.type
        if _scalar(value_type):
            ir, value = self.value(e.value)
            slot_ir, address = self.scratch_address(value_type)
            ir += slot_ir + [IRStore(address=address, value=value, value_type=value_type)]
        elif _ported(value_type):
            ir, address = self.address(e.value)
        else:
            raise NotYetPorted(f"print of {value_type}")
        desc = self.temp()
        label = type_descriptor(self.ir_program, value_type)
        return ir + [
            IRStaticDataAddress(dst=desc, label=label),
            IRCall(dst=None, name='hornet_print', args=[address, desc])
        ]

    # -- loops

    def fresh_loop_variable(self, s: t.For) -> list:
        """A for-loop variable whose address may outlive an iteration gets new storage each iteration,
        carrying its value over, so a pointer taken in one iteration keeps that iteration's value."""
        if not isinstance(s.init, t.Declare) or not self.heap(s.init.symbol):
            return []
        symbol = s.init.symbol
        temp, _ = self.bind(symbol)
        box = self.temp()
        ir = [IRCall(dst=box, name='hornet_alloc', args=[IRConst(self.width(symbol.type), Type.INT64)])]
        if _scalar(symbol.type):
            current = self.temp(symbol.type)
            return ir + [
                IRLoad(dst=current, address=temp),
                IRStore(address=box, value=current, value_type=symbol.type),
                IRMove(dst=temp, src=box)
            ]
        old_ir, old = self.local_address(symbol)
        slot_address = self.temp()
        return ir + old_ir + [
            IRCopy(dst_address=box, src_address=old, value_type=symbol.type),
            IRLocalAddress(dst=slot_address, slot=self.ir_fn.var_slots[symbol.id]),
            IRStore(address=slot_address, value=box, value_type=Type.INT64)
        ]

    def bind_from(self, symbol, address) -> list:
        """A for-in binding's value for this iteration: a copy of what is at `address`, in new storage
        each iteration when its address may outlive the iteration."""
        temp, heap = self.bind(symbol)
        if _scalar(symbol.type):
            value = self.temp(symbol.type)
            return [IRLoad(dst=value, address=address)] + self.initialize_scalar(symbol, value)
        ir = self.allocate_heap(symbol) if heap else []
        dst_ir, dst = self.local_address(symbol)
        return ir + dst_ir + [IRCopy(dst_address=dst, src_address=address, value_type=symbol.type)]

    def panic_when(self, cond, message: str, at) -> list:
        """Panic with `message`, reported at typed node `at`, when `cond` holds."""
        ok = self.ids.new_label("check_ok")
        return [
            IRBranch(cond=cond, true_label=self.panics.label(located(message, at.where)), false_label=ok),
            IRLabel(ok)
        ]

    def panic_when_missing(self, found, message: str, dict_expr, at) -> list:
        """Panic with `message` and the key, when a lookup or a delete in `dict_expr` reports nothing
        `found` (a null address, a zero count). The runtime noted the key; its type says how to print it."""
        missing, ok = self.temp(Type.BOOL), self.ids.new_label("check_ok")
        key_descriptor = type_descriptor(self.ir_program, dict_expr.type.key_type)
        return [
            IRBinOp(dst=missing, op=BinaryOp.EQUAL, left=found, right=IRConst(0, found.type)),
            IRBranch(cond=missing, true_label=self.panics.label(located(message, at.where), key_descriptor),
                     false_label=ok),
            IRLabel(ok)
        ]

    def for_in(self, s: t.ForIn) -> list:
        if s.kind == 'dict':
            return self.for_in_dict(s)
        start = self.ids.new_label('for_in_start')
        body = self.ids.new_label('for_in_body')
        step = self.ids.new_label('for_in_next')
        end = self.ids.new_label('for_in_end')
        recheck = None  # (descriptor address, pointer at the start): a slice reallocated during the loop panics
        if s.kind == 'str':
            ir, base, length = self.str_value(s.iterable)
            element_type = Type.UINT8
        elif s.kind == 'array':
            ir, base = self.address(s.iterable)
            length, element_type = IRConst(s.iterable.type.size, Type.INT), s.iterable.type.element_type
        else:
            element_type = s.iterable.type.element_type
            root = s.iterable.base if isinstance(s.iterable, t.SliceOf) and s.iterable.kind == 'slice' else s.iterable
            ir, base, length, _ = self.slice_value(s.iterable)
            if isinstance(root, _PLACE_NODES):
                root_ir, descriptor = self.place_address(root)
                first = self.temp()
                ir += root_ir + [IRLoad(dst=first, address=descriptor)]
                recheck = (descriptor, first)
        i, more = self.temp(Type.INT), self.temp(Type.BOOL)
        ir += [IRMove(dst=i, src=IRConst(0, Type.INT)), IRJump(start), IRLabel(start),
               IRBinOp(dst=more, op=BinaryOp.LESS_THAN, left=i, right=length),
               IRBranch(cond=more, true_label=body, false_label=end), IRLabel(body)]
        if recheck is not None:
            now, moved = self.temp(), self.temp(Type.BOOL)
            ir += [
                IRLoad(dst=now, address=recheck[0]),
                IRBinOp(dst=moved, op=BinaryOp.NOT_EQUAL, left=now, right=recheck[1]),
            ]
            ir += self.panic_when(moved, "for ... in: slice was reallocated (e.g. by append) during iteration", s)
        element_ir, element = self.element(base, i, element_type)
        ir += element_ir
        bindings = list(s.bindings)
        if len(bindings) == 2:
            ir += self.initialize_scalar(bindings.pop(0), i)
        ir += self.bind_from(bindings[0], element)
        self.loops.append((step, end, len(self.deferred)))
        ir += self.block(s.body)
        self.loops.pop()
        return ir + [
            IRJump(step),
            IRLabel(step),
            IRBinOp(dst=i, op=BinaryOp.ADD, left=i, right=IRConst(1, Type.INT)),
            IRJump(start),
            IRLabel(end)
        ]

    def for_in_dict(self, s: t.ForIn) -> list:
        """Scan every bucket; a dict whose buckets are reallocated during the loop panics."""
        dict_type = s.iterable.type
        key_width, value_width = self.width(dict_type.key_type), self.width(dict_type.element_type)
        ir, header = self.dict_header(s.iterable)
        buckets, capacity = self.temp(), self.temp()
        cap_ir, cap_address = self.offset(header, 24)
        ir += [IRLoad(dst=buckets, address=header)] + cap_ir + [IRLoad(dst=capacity, address=cap_address)]
        start, body, entry, step, end = (self.ids.new_label(x) for x in
                                         ("for_in_start", "for_in_body", "for_in_entry", "for_in_next", "for_in_end"))
        i, more, now, moved = self.temp(), self.temp(Type.BOOL), self.temp(), self.temp(Type.BOOL)
        scaled, bucket, state, occupied = self.temp(), self.temp(), self.temp(Type.UINT8), self.temp(Type.BOOL)
        ir += [
            IRMove(dst=i, src=IRConst(0, Type.INT64)), IRJump(start), IRLabel(start),
            IRBinOp(dst=more, op=BinaryOp.LESS_THAN, left=i, right=capacity),
            IRBranch(cond=more, true_label=body, false_label=end), IRLabel(body),
            IRLoad(dst=now, address=header),
            IRBinOp(dst=moved, op=BinaryOp.NOT_EQUAL, left=now, right=buckets),
        ]
        ir += self.panic_when(moved, "for ... in: dict's own buckets were reallocated (e.g. by an insert that "
                                     "triggered growth) during iteration", s)
        ir += [
            IRBinOp(dst=scaled, op=BinaryOp.MULTIPLY, left=i, right=IRConst(1 + key_width + value_width, Type.INT64)),
            IRBinOp(dst=bucket, op=BinaryOp.ADD, left=buckets, right=scaled),
            IRLoad(dst=state, address=bucket),
            IRBinOp(dst=occupied, op=BinaryOp.EQUAL, left=state, right=IRConst(_DICT_BUCKET_OCCUPIED, Type.UINT8)),
            IRBranch(cond=occupied, true_label=entry, false_label=step),
            IRLabel(entry),
        ]
        key_ir, key_address = self.offset(bucket, 1)
        ir += key_ir + self.bind_from(s.bindings[0], key_address)
        if len(s.bindings) == 2:
            value_ir, value_address = self.offset(key_address, key_width)
            ir += value_ir + self.bind_from(s.bindings[1], value_address)
        self.loops.append((step, end, len(self.deferred)))
        ir += self.block(s.body)
        self.loops.pop()
        return ir + [
            IRJump(step),
            IRLabel(step),
            IRBinOp(dst=i, op=BinaryOp.ADD, left=i, right=IRConst(1, Type.INT64)),
            IRJump(start),
            IRLabel(end),
        ]

    # -- dicts: a dict value is a pointer to its header

    def dict_header(self, e) -> tuple:
        ir, address = self.address(e)
        header = self.temp()
        return ir + [IRLoad(dst=header, address=address)], header

    def dict_call(self, dict_expr, key, operation: str, result_type=None, extra=()) -> tuple:
        """hornet_dict_<operation>_{str,scalar}_key on `dict_expr` and `key`."""
        ir, entry = self.dict_entry(dict_expr, key)
        call_ir, result = self.dict_op(entry, operation, result_type, extra)
        return ir + call_ir, result

    def dict_entry(self, dict_expr, key) -> tuple:
        """Evaluate a dictionary and a key once, for one or more operations: (ir, entry)."""
        dict_type = dict_expr.type
        value_width = IRConst(self.width(dict_type.element_type), Type.INT64)
        ir, header = self.dict_header(dict_expr)
        if dict_type.key_type.kind == TypeKind.STR:
            key_ir, ptr, length = self.str_value(key)
            return ir + key_ir, (header, [value_width, ptr, length], 'str')
        key_ir, value = self.value(key)
        scratch_ir, key_address = self.scratch_address(dict_type.key_type)
        key_ir += scratch_ir + [IRStore(address=key_address, value=value, value_type=dict_type.key_type)]
        return ir + key_ir, (
            header, [IRConst(self.width(dict_type.key_type), Type.INT64), value_width, key_address], 'scalar',
        )

    def dict_op(self, entry, operation: str, result_type=None, extra=()) -> tuple:
        header, key_args, kind = entry
        result = None if result_type is None else self.temp(result_type)
        call = IRCall(dst=result, name=f'hornet_dict_{operation}_{kind}_key', args=[header, *key_args, *extra])
        return [call], result

    def dict_set(self, target: t.DictLookup, value_address) -> list:
        return self.dict_call(target.dict, target.key, 'set', extra=(value_address,))[0]

    def dict_literal(self, dst, e: t.DictLiteral) -> list:
        """Build the table, insert each entry, then the header; `dst` gets the header pointer."""
        key_type, value_type = e.type.key_type, e.type.element_type
        key_width, value_width = self.width(key_type), self.width(value_type)
        stride = 1 + key_width + value_width
        capacity = 8
        while capacity < len(e.entries) * 2:
            capacity *= 2
        buckets, header, count = self.temp(), self.temp(), IRConst(0, Type.INT)
        ir = [IRCall(dst=buckets, name='hornet_alloc_zeroed',
                     args=[IRConst(capacity, Type.INT64), IRConst(stride, Type.INT64)])]
        table = [buckets, IRConst(capacity, Type.INT64), IRConst(stride, Type.INT64)]
        for key, value in e.entries:
            value_ir, value_address = self.scratch_address(value_type)
            ir += value_ir + self.store(value_address, value)
            inserted = self.temp(Type.INT)
            if key_type.kind == TypeKind.STR:
                key_ir, ptr, length = self.str_value(key)
                ir += key_ir + [IRCall(dst=inserted, name='hornet_dict_insert_str_key',
                                       args=table + [ptr, length, value_address, IRConst(value_width, Type.INT64)])]
            else:
                key_ir, key_address = self.scratch_address(key_type)
                ir += key_ir + self.store(key_address, key) + [
                    IRCall(dst=inserted, name='hornet_dict_insert_scalar_key',
                           args=table + [key_address, IRConst(key_width, Type.INT64), value_address,
                                         IRConst(value_width, Type.INT64)])]
            total = self.temp(Type.INT)
            ir.append(IRBinOp(dst=total, op=BinaryOp.ADD, left=count, right=inserted))  # 1 for a new key
            count = total
        fields = [(0, buckets, Type.INT64), (8, count, Type.INT), (16, IRConst(0, Type.INT), Type.INT),
                  (24, IRConst(capacity, Type.INT64), Type.INT64)]
        ir.append(IRCall(dst=header, name='hornet_alloc', args=[IRConst(_DICT_HEADER_SIZE, Type.INT64)]))
        for offset, value, value_type_ in fields:
            offset_ir, address = self.offset(header, offset)
            ir += offset_ir + [IRStore(address=address, value=value, value_type=value_type_)]
        return ir + [IRStore(address=dst, value=header, value_type=Type.INT64)]

    # -- slices

    def read_slice(self, address) -> tuple:
        ptr, length, cap = self.temp(), self.temp(Type.INT), self.temp(Type.INT)
        len_ir, len_address = self.offset(address, 8)
        cap_ir, cap_address = self.offset(address, 16)
        return [IRLoad(dst=ptr, address=address)] + len_ir + [
            IRLoad(dst=length, address=len_address)
        ] + cap_ir + [IRLoad(dst=cap, address=cap_address)], ptr, length, cap

    def append(self, e: t.Append) -> tuple:
        """Write in place when there is spare capacity; otherwise grow (cap 0 -> 1, doubling below 256,
        then by a quarter) and copy."""
        element_type = e.type.element_type
        width = IRConst(self.width(element_type), Type.INT)
        ir, ptr, length, cap = self.slice_value(e.slice)
        out_ptr, out_len, out_cap = self.temp(), self.temp(Type.INT), self.temp(Type.INT)
        reuse = self.ids.new_label('append_reuse')
        grow = self.ids.new_label('append_grow')
        end = self.ids.new_label('append_end')
        full, new_len = self.temp(Type.BOOL), self.temp(Type.INT)
        ir += [
            IRBinOp(dst=full, op=BinaryOp.GREATER_THAN_OR_EQUAL, left=length, right=cap),
            IRBinOp(dst=new_len, op=BinaryOp.ADD, left=length, right=IRConst(1, Type.INT)),
            IRBranch(cond=full, true_label=grow, false_label=reuse),
        ]

        def write_at(base) -> list:
            element_ir, address = self.element(base, length, element_type)
            return element_ir + self.store(address, e.value)

        ir += [IRLabel(reuse)] + write_at(ptr) + [
            IRMove(dst=out_ptr, src=ptr),
            IRMove(dst=out_cap, src=cap),
            IRMove(dst=out_len, src=new_len),
            IRJump(end)
        ]
        new_cap, grown, is_zero, is_small, quarter = (
            self.temp(Type.INT), self.temp(), self.temp(Type.BOOL), self.temp(Type.BOOL), self.temp(Type.INT))
        zero = self.ids.new_label('cap_zero')
        nonzero = self.ids.new_label('cap_nonzero')
        small = self.ids.new_label('cap_small')
        large = self.ids.new_label('cap_large')
        ready = self.ids.new_label('cap_ready')
        ir += [
            IRLabel(grow),
            IRBinOp(dst=is_zero, op=BinaryOp.EQUAL, left=cap, right=IRConst(0, Type.INT)),
            IRBranch(cond=is_zero, true_label=zero, false_label=nonzero),
            IRLabel(zero), IRMove(dst=new_cap, src=IRConst(1, Type.INT)), IRJump(ready),
            IRLabel(nonzero), IRBinOp(dst=is_small, op=BinaryOp.LESS_THAN, left=cap, right=IRConst(256, Type.INT)),
            IRBranch(cond=is_small, true_label=small, false_label=large),
            IRLabel(small), IRBinOp(dst=new_cap, op=BinaryOp.ADD, left=cap, right=cap), IRJump(ready),
            IRLabel(large), IRBinOp(dst=quarter, op=BinaryOp.SHIFT_RIGHT, left=cap, right=IRConst(2, Type.INT)),
            IRBinOp(dst=new_cap, op=BinaryOp.ADD, left=cap, right=quarter), IRJump(ready),
            IRLabel(ready), IRCall(dst=grown, name='hornet_slice_grow', args=[ptr, length, new_cap, width]),
        ]
        ir += write_at(grown) + [
            IRMove(dst=out_ptr, src=grown),
            IRMove(dst=out_cap, src=new_cap),
            IRMove(dst=out_len, src=new_len),
            IRJump(end),
            IRLabel(end)
        ]
        return ir, out_ptr, out_len, out_cap

    # -- comparisons of composites

    def composite_equality(self, e: t.Binary) -> tuple:
        if e.op not in (BinaryOp.EQUAL, BinaryOp.NOT_EQUAL):
            raise NotYetPorted(f"{e.op.name} on {e.left.type}")
        left_ir, left = self.address(e.left, fresh=self.changed_by(e.left, e.right))
        right_ir, right = self.address(e.right)
        mismatch, end = self.ids.new_label("eq_mismatch"), self.ids.new_label("eq_end")
        result = self.temp(Type.BOOL)
        equal = 1 if e.op == BinaryOp.EQUAL else 0
        return left_ir + right_ir + self.equal_or_jump(left, right, e.left.type, mismatch) + [
            IRMove(dst=result, src=IRConst(equal, Type.BOOL)),
            IRJump(end),
            IRLabel(mismatch),
            IRMove(dst=result, src=IRConst(1 - equal, Type.BOOL)),
            IRJump(end),
            IRLabel(end),
        ], result

    def equal_or_jump(self, left, right, type_: Type, mismatch: str) -> list:
        """Fall through if the values at `left` and `right` are equal; jump to `mismatch` if not."""
        same = self.ids.new_label("eq_same")
        if type_.kind == TypeKind.STRUCT:
            ir, offset = [], 0
            for field_type in self.ir_program.struct_registry[type_.struct_name].fields.values():
                l_ir, l = self.offset(left, offset)
                r_ir, r = self.offset(right, offset)
                ir += l_ir + r_ir + self.equal_or_jump(l, r, field_type, mismatch)
                offset += self.width(field_type)
            return ir
        if type_.kind == TypeKind.ARRAY:
            ir = []
            for i in range(type_.size):
                l_ir, l = self.offset(left, i * self.width(type_.element_type))
                r_ir, r = self.offset(right, i * self.width(type_.element_type))
                ir += l_ir + r_ir + self.equal_or_jump(l, r, type_.element_type, mismatch)
            return ir
        if type_.kind == TypeKind.SUM:
            # The same variant, and then that variant's payloads: which comparison that is depends on
            # the tag. (The bytes can't be compared whole: those past the variant held mean nothing.)
            l_tag, r_tag, differs = self.temp(Type.INT32), self.temp(Type.INT32), self.temp(Type.BOOL)
            tags_same, done = self.ids.new_label("eq_tags_same"), self.ids.new_label("eq_sum_same")
            l_ir, l_payload = self.offset(left, SUM_TYPE_TAG_WIDTH)
            r_ir, r_payload = self.offset(right, SUM_TYPE_TAG_WIDTH)
            ir = [
                IRLoad(dst=l_tag, address=left),
                IRLoad(dst=r_tag, address=right),
                IRBinOp(dst=differs, op=BinaryOp.NOT_EQUAL, left=l_tag, right=r_tag),
                IRBranch(cond=differs, true_label=mismatch, false_label=tags_same), IRLabel(tags_same),
            ] + l_ir + r_ir
            for number, variant in enumerate(self.ir_program.sum_type_registry[type_.sum_type_name].variants):
                if variant == Type.NONE:  # nothing held: the tags were all there is
                    continue
                holds = self.temp(Type.BOOL)
                this, other = self.ids.new_label("eq_variant"), self.ids.new_label("eq_next")
                ir += [
                    IRBinOp(dst=holds, op=BinaryOp.EQUAL, left=l_tag, right=IRConst(number, Type.INT32)),
                    IRBranch(cond=holds, true_label=this, false_label=other), IRLabel(this),
                ] + self.equal_or_jump(l_payload, r_payload, variant, mismatch) + [IRJump(done), IRLabel(other)]
            return ir + [IRJump(done), IRLabel(done)]
        if type_.kind == TypeKind.STR:
            l_ir, l_ptr, l_len = self.read_str(left)
            r_ir, r_ptr, r_len = self.read_str(right)
            same_len, cmp, differs = self.temp(Type.BOOL), self.temp(Type.INT32), self.temp(Type.BOOL)
            lengths_ok = self.ids.new_label("eq_len_same")
            return l_ir + r_ir + [
                IRBinOp(dst=same_len, op=BinaryOp.EQUAL, left=l_len, right=r_len),
                IRBranch(cond=same_len, true_label=lengths_ok, false_label=mismatch), IRLabel(lengths_ok),
                IRCall(dst=cmp, name='memcmp', args=[l_ptr, r_ptr, l_len]),
                IRBinOp(dst=differs, op=BinaryOp.NOT_EQUAL, left=cmp, right=IRConst(0, Type.INT32)),
                IRBranch(cond=differs, true_label=mismatch, false_label=same), IRLabel(same)
            ]
        if not _scalar(type_):
            raise NotYetPorted(f"equality of {type_}")
        l_value, r_value, differs = self.temp(type_), self.temp(type_), self.temp(Type.BOOL)
        return [
            IRLoad(dst=l_value, address=left),
            IRLoad(dst=r_value, address=right),
            IRBinOp(dst=differs, op=BinaryOp.NOT_EQUAL, left=l_value, right=r_value),
            IRBranch(cond=differs, true_label=mismatch, false_label=same),
            IRLabel(same),
        ]

    def element_contains(self, e: t.ElementContains) -> tuple:
        """`x in a`: compare with each element in turn."""
        element_type = e.value.type
        needle_ir, needle = self.scratch_address(element_type)
        ir = needle_ir + self.store(needle, e.value)
        if e.container.type.kind == TypeKind.ARRAY:
            base_ir, base = self.address(e.container)
            length = IRConst(e.container.type.size, Type.INT)
        else:
            base_ir, base, length, _ = self.slice_value(e.container)
        ir += base_ir
        i, more, result = self.temp(Type.INT), self.temp(Type.BOOL), self.temp(Type.BOOL)
        start = self.ids.new_label('in_start')
        body = self.ids.new_label('in_body')
        nope = self.ids.new_label('in_next')
        found = self.ids.new_label('in_found')
        end = self.ids.new_label('in_end')
        element_ir, element = self.element(base, i, element_type)
        return ir + [
            IRMove(dst=i, src=IRConst(0, Type.INT)),
            IRJump(start),
            IRLabel(start),
            IRBinOp(dst=more, op=BinaryOp.LESS_THAN, left=i, right=length),
            IRBranch(cond=more, true_label=body, false_label=end),
            IRLabel(body),
        ] + element_ir + self.equal_or_jump(element, needle, element_type, nope) + [
            IRJump(found),
            IRLabel(nope),
            IRBinOp(dst=i, op=BinaryOp.ADD, left=i, right=IRConst(1, Type.INT)),
            IRJump(start),
            IRLabel(found),
            IRMove(dst=result, src=IRConst(1, Type.BOOL)),
            IRJump(start + "_done"),
            IRLabel(end),
            IRMove(dst=result, src=IRConst(0, Type.BOOL)),
            IRJump(start + "_done"),
            IRLabel(start + "_done"),
        ], result

    def concat(self, left_ptr, left_len, right_ptr, right_len) -> tuple:
        """A new string holding both: (ir, ptr, len)."""
        total, buffer, right_dst = self.temp(Type.INT), self.temp(), self.temp()
        return [
            IRBinOp(dst=total, op=BinaryOp.ADD, left=left_len, right=right_len),
            IRCall(dst=buffer, name='hornet_alloc', args=[total]),
            IRCall(dst=None, name='memcpy', args=[buffer, left_ptr, left_len]),
            IRBinOp(dst=right_dst, op=BinaryOp.ADD, left=buffer, right=left_len),
            IRCall(dst=None, name='memcpy', args=[right_dst, right_ptr, right_len])
        ], buffer, total

    def compound_assign_dict_entry(self, s: t.CompoundAssign) -> list:
        """`d[k] op= v` where evaluating `v` may change `d`: `d` and `k` once, read the entry, evaluate
        `v`, then store by key (a fresh lookup), since `v` may have grown the table or removed `k`."""
        target = s.target
        ir, entry = self.dict_entry(target.dict, target.key)
        lookup_ir, address = self.dict_op(entry, 'lookup', Type.INT64)
        scratch_ir, result_address = self.scratch_address(target.type)
        ir += lookup_ir + self.panic_when_missing(address, "dict lookup: key not found", target.dict, target) \
            + scratch_ir
        if target.type == Type.STR:
            read_ir, left_ptr, left_len = self.read_str(address)
            value_ir, right_ptr, right_len = self.str_value(s.value)
            concat_ir, ptr, length = self.concat(left_ptr, left_len, right_ptr, right_len)
            len_ir, len_address = self.offset(result_address, 8)
            ir += read_ir + value_ir + concat_ir + len_ir + [
                IRStore(address=result_address, value=ptr, value_type=Type.INT64),
                IRStore(address=len_address, value=length, value_type=Type.INT)]
        else:
            current, result = self.temp(target.type), self.temp(target.type)
            value_ir, value = self.value(s.value)
            ir += [IRLoad(dst=current, address=address)] + value_ir + [
                IRBinOp(dst=result, op=s.op, left=current, right=value, where=s.where),
                IRStore(address=result_address, value=result, value_type=target.type),
            ]
        return ir + self.dict_op(entry, 'set', extra=(result_address,))[0]
