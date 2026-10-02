"""IR from the typed tree (typed_ast.py), one function at a time. NotYetPorted marks a tree shape
with no rule: a compiler bug, since every shape elaboration produces is handled.

Composite values live in memory: `address(e)` gives where one is (materializing a temporary if
needed) and `write_into(dst, e)` stores one; a str is read as (ptr, len) and a slice as
(ptr, len, cap).
"""

import typed_ast as t
from escape_analysis import analyze_array_escapes, is_heap_allocated
from ir.typedesc import type_descriptor
from ir.ir import (
    IRBinOp, IRBoundsCheck, IRBranch, IRCall, IRCast, IRConst, IRCopy, IRFunction, IRJump, IRLabel, IRLoad,
    IRLocalAddress, IRMove, IRNullCheck, IRReturn, IRSliceBoundsCheck, IRStaticDataAddress, IRStore, IRUnOp,
)
from ops import BinaryOp
from typesys import SUM_TYPE_TAG_WIDTH, Type, TypeKind, type_byte_width

_SCALAR_KINDS = {TypeKind.INT, TypeKind.INT32, TypeKind.INT8, TypeKind.UINT8, TypeKind.BOOL,
                 TypeKind.POINTER}


class NotYetPorted(Exception):
    """A typed-tree shape with no IR rule."""


def _scalar(type_: Type) -> bool:
    return type_.kind in _SCALAR_KINDS


_COMPOSITE_BY_ADDRESS = {TypeKind.ARRAY, TypeKind.STRUCT, TypeKind.SUM, TypeKind.DICT}  # passed and returned by address
_PORTED_KINDS = _SCALAR_KINDS | _COMPOSITE_BY_ADDRESS | {TypeKind.STR, TypeKind.SLICE}


def _ported(type_: Type) -> bool:
    return type_ == Type.VOID or type_.kind in _PORTED_KINDS


_PLACE_NODES = (t.Local, t.Deref, t.FieldAccess, t.ArrayIndex, t.SliceIndex, t.Payload, t.DictLookup)
_DICT_BUCKET_OCCUPIED = 1  # runtime.c's HORNET_DICT_BUCKET_OCCUPIED
_DICT_HEADER_SIZE = 32  # {buckets, count, tombstones, capacity}


class TypedFunctionBuilder:
    def __init__(self, ir_program):
        self.ir_program = ir_program
        self.ids = ir_program.ids

    def build(self, fn: t.Function) -> IRFunction:
        if not _ported(fn.return_type):
            raise NotYetPorted(f"returns {fn.return_type}")
        self.ir_fn = ir_fn = IRFunction(name=fn.name)
        ir_fn.return_type = fn.return_type
        self.heap_ids = analyze_array_escapes(fn, self.ir_program.struct_registry, self.ir_program.escape_summaries)
        self.storage = {}  # symbol id -> (Temp or None, heap)
        self.loops = []  # (continue label, end label)
        self.scratch = {}  # name -> slot, for per-function scratch storage
        ir = self.params(fn)
        ir += self.block(fn.body)
        if not ir or not isinstance(ir[-1], (IRBranch, IRJump, IRReturn)):
            ir.append(IRReturn(value=None))
        ir_fn.body = ir
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
            return [IRCall(dst=address, name='malloc', args=[IRConst(self.width(type_), Type.INT64)])], address
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
        return [IRCall(dst=pointer, name='malloc', args=[IRConst(self.width(symbol.type), Type.INT64)]),
                IRLocalAddress(dst=slot_address, slot=self.ir_fn.var_slots[symbol.id]),
                IRStore(address=slot_address, value=pointer, value_type=Type.INT64)]

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
        return [IRCall(dst=box, name='malloc', args=[IRConst(self.width(symbol.type), Type.INT64)]),
                IRStore(address=box, value=value, value_type=symbol.type),
                IRMove(dst=temp, src=box)]

    def params(self, fn: t.Function) -> list:
        ir, incoming = [], []
        if not _scalar(fn.return_type) and fn.return_type != Type.VOID:
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
        ir = []
        for s in statements or ():
            ir += self.statement(s)
        return ir

    def statement(self, s) -> list:
        if isinstance(s, t.Declare):
            return self.initialize(s.symbol, s.init)
        if isinstance(s, t.Assign):
            return self.assign(s.target, s.value)
        if isinstance(s, t.CompoundAssign):
            if isinstance(s.target, t.Local):  # read and written in place: `x = x op v`
                kind = t.StrConcat if s.target.type == Type.STR else lambda type_, l, r: t.Binary(type_, s.op, l, r)
                return self.assign(s.target, kind(s.target.type, s.target, s.value))
            if s.target.type == Type.STR:  # `+=`: the place's address once, then concatenate
                ir, address = self.place_address(s.target)
                read_ir, left_ptr, left_len = self.read_str(address)
                value_ir, right_ptr, right_len = self.str_value(s.value)
                concat_ir, ptr, length = self.concat(left_ptr, left_len, right_ptr, right_len)
                len_ir, len_address = self.offset(address, 8)
                return ir + read_ir + value_ir + concat_ir + len_ir + [
                    IRStore(address=address, value=ptr, value_type=Type.INT64),
                    IRStore(address=len_address, value=length, value_type=Type.INT)]
            if not _scalar(s.target.type):
                raise NotYetPorted(f"compound assignment of {s.target.type}")
            ir, address = self.place_address(s.target)
            current, result = self.temp(s.target.type), self.temp(s.target.type)
            value_ir, value = self.value(s.value)
            return ir + [IRLoad(dst=current, address=address)] + value_ir + [
                IRBinOp(dst=result, op=s.op, left=current, right=value),
                IRStore(address=address, value=result, value_type=s.target.type)]
        if isinstance(s, t.ExprStmt):
            if s.expr.type == Type.NONE:  # a bare `none` does nothing
                return []
            if _scalar(s.expr.type) or s.expr.type == Type.VOID:
                ir, _ = self.value(s.expr, discard=True)
                return ir
            ir, _ = self.address(s.expr)  # evaluated for its effects (calls, checks)
            return ir
        if isinstance(s, t.Return):
            if s.value is None:
                return [IRReturn(value=None)]
            if _scalar(s.value.type):
                ir, value = self.value(s.value)
                return ir + [IRReturn(value=value)]
            hidden, address = self.temp(), self.temp()
            return [IRLocalAddress(dst=address, slot=self.ir_fn.hidden_return_ptr_slot),
                    IRLoad(dst=hidden, address=address)] + self.write_into(hidden, s.value) + [IRReturn(value=None)]
        if isinstance(s, t.If):
            then_label, else_label, end_label = (self.ids.new_label(x) for x in ("if_then", "if_else", "if_end"))
            ir, cond = self.value(s.cond)
            ir += [IRBranch(cond=cond, true_label=then_label, false_label=else_label), IRLabel(then_label)]
            ir += self.block(s.then_body) + [IRJump(end_label), IRLabel(else_label)]
            return ir + self.block(s.else_body) + [IRJump(end_label), IRLabel(end_label)]
        if isinstance(s, t.Match):
            return self.match(s)
        if isinstance(s, t.While):
            start, body, end = (self.ids.new_label(x) for x in ("while_start", "while_body", "while_end"))
            cond_ir, cond = self.value(s.cond)
            self.loops.append((start, end))
            body_ir = self.block(s.body)
            self.loops.pop()
            return [IRJump(start), IRLabel(start)] + cond_ir + [IRBranch(cond=cond, true_label=body, false_label=end),
                                                               IRLabel(body)] + body_ir + [IRJump(start), IRLabel(end)]
        if isinstance(s, t.For):
            start, body, step, end = (self.ids.new_label(x) for x in ("for_start", "for_body", "for_step", "for_end"))
            ir = self.statement(s.init) if s.init is not None else []
            cond_ir, cond = self.value(s.cond)
            self.loops.append((step, end))
            body_ir = self.block(s.body)
            self.loops.pop()
            return ir + [IRJump(start), IRLabel(start)] + cond_ir + [
                IRBranch(cond=cond, true_label=body, false_label=end), IRLabel(body)] + body_ir + [
                IRJump(step), IRLabel(step)] + self.fresh_loop_variable(s) + self.statement(s.step) + [
                IRJump(start), IRLabel(end)]
        if isinstance(s, t.ForIn):
            return self.for_in(s)
        if isinstance(s, t.Break):
            return [IRJump(self.loops[-1][1])]
        if isinstance(s, t.Continue):
            return [IRJump(self.loops[-1][0])]
        raise NotYetPorted(type(s).__name__)

    def match(self, s: t.Match) -> list:
        """Compare the subject's tag with each arm's variant in turn."""
        ir, address = self.address(s.subject)
        tag = self.temp(Type.INT32)
        ir.append(IRLoad(dst=tag, address=address))
        variants = self.ir_program.sum_type_registry[s.subject.type.sum_type_name].variants
        end = self.ids.new_label("match_end")
        for variant, body in s.arms:
            arm, next_arm, is_it = self.ids.new_label("match_arm"), self.ids.new_label("match_next"), self.temp(Type.BOOL)
            ir += [IRBinOp(dst=is_it, op=BinaryOp.EQUAL, left=tag, right=IRConst(variants.index(variant), Type.INT32)),
                   IRBranch(cond=is_it, true_label=arm, false_label=next_arm), IRLabel(arm)]
            ir += self.block(body) + [IRJump(end), IRLabel(next_arm)]
        return ir + self.block(s.else_body) + [IRJump(end), IRLabel(end)]

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
                return value_ir + value_ir2 + [IRStore(address=address, value=value, value_type=target.type)] + \
                    self.dict_set(target, address)
            ir, address = self.place_address(target)
            return value_ir + ir + [IRStore(address=address, value=value, value_type=target.type)]
        # A composite: the new value may read the target (`p = P(p.y, p.x)`), so it is built
        # completely before the target changes.
        value_ir, value_address = self.address(value_expr, fresh=True)
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
            return self.pointer(e.pointer)
        if isinstance(e, t.FieldAccess):
            if e.through_pointer:
                ir, base = self.pointer(e.base)
            else:
                ir, base = self.address(e.base)
            offset_ir, address = self.offset(base, self.field_offset(e))
            return ir + offset_ir, address
        if isinstance(e, t.ArrayIndex):
            ir, base = self.address(e.base)
            index_ir, index = self.value(e.index)
            element_ir, address = self.element(base, index, e.type)
            return ir + index_ir + [IRBoundsCheck(index=index, length=IRConst(e.base.type.size, Type.INT))] + \
                element_ir, address
        if isinstance(e, t.SliceIndex):
            ir, ptr, length, _ = self.slice_value(e.base)
            index_ir, index = self.value(e.index)
            element_ir, address = self.element(ptr, index, e.type)
            return ir + index_ir + [IRBoundsCheck(index=index, length=length)] + element_ir, address
        if isinstance(e, t.Payload):
            ir, base = self.address(e.sum)
            offset_ir, address = self.offset(base, SUM_TYPE_TAG_WIDTH)
            return ir + offset_ir, address
        if isinstance(e, t.DictLookup):
            return self.dict_call(e.dict, e.key, 'lookup', result_type=Type.INT64)
        raise NotYetPorted(f"place {type(e).__name__}")

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
        return [IRBinOp(dst=scaled, op=BinaryOp.MULTIPLY, left=index, right=IRConst(self.width(element_type), Type.INT)),
                IRBinOp(dst=address, op=BinaryOp.ADD, left=base, right=scaled)], address

    # -- composite values

    def address(self, e, fresh: bool = False) -> tuple:
        """Where composite value `e` is. A place is used in place unless `fresh` asks for a copy;
        anything else is built in a new temporary."""
        if not fresh and isinstance(e, _PLACE_NODES):
            return self.place_address(e)
        ir, address = self.scratch_address(e.type)
        return ir + self.write_into(address, e), address

    def stored(self, e) -> tuple:
        """The address of a str or slice held in memory: a place, or a call's result."""
        if isinstance(e, _PLACE_NODES):
            return self.place_address(e)
        if isinstance(e, t.Call):
            ir, address = self.scratch_address(e.type)
            return ir + self.call(e, False, destination=address)[0], address
        raise NotYetPorted(f"a {e.type} from {type(e).__name__}")

    def write_into(self, dst, e) -> list:
        """Store composite value `e` at `dst`."""
        kind = e.type.kind
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
            for (name, field_type), value in zip(self.ir_program.struct_registry[e.type.struct_name].fields.items(), e.fields):
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
        if isinstance(e, t.ZeroValue):
            return self.zero_into(dst, e.type)
        if isinstance(e, t.NewEmptyDict):
            return self.zero_into(dst, e.type)
        if isinstance(e, t.DictLiteral):
            return self.dict_literal(dst, e)
        if isinstance(e, t.Call):
            return self.call(e, False, destination=dst)[0]
        raise NotYetPorted(f"composite {type(e).__name__}")

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
            return [IRCall(dst=header, name='calloc', args=[IRConst(1, Type.INT64), IRConst(_DICT_HEADER_SIZE, Type.INT64)]),
                    IRStore(address=dst, value=header, value_type=Type.INT64)]
        return [IRStore(address=dst, value=IRConst(0, type_), value_type=type_)]

    def zero_array(self, dst, type_: Type) -> list:
        i, cond, scaled, element = self.temp(Type.INT), self.temp(Type.BOOL), self.temp(Type.INT), self.temp()
        start, body, end = (self.ids.new_label(x) for x in ("zero_start", "zero_body", "zero_end"))
        width = self.width(type_.element_type)
        return [IRMove(dst=i, src=IRConst(0, Type.INT)), IRJump(start), IRLabel(start),
                IRBinOp(dst=cond, op=BinaryOp.LESS_THAN, left=i, right=IRConst(type_.size, Type.INT)),
                IRBranch(cond=cond, true_label=body, false_label=end), IRLabel(body),
                IRBinOp(dst=scaled, op=BinaryOp.MULTIPLY, left=i, right=IRConst(width, Type.INT)),
                IRBinOp(dst=element, op=BinaryOp.ADD, left=dst, right=scaled)] + \
            self.zero_into(element, type_.element_type) + [
                IRBinOp(dst=i, op=BinaryOp.ADD, left=i, right=IRConst(1, Type.INT)), IRJump(start), IRLabel(end)]

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
            return base_ir + bounds_ir + [IRBinOp(dst=new_len, op=BinaryOp.SUBTRACT, left=high, right=low),
                                         IRBinOp(dst=new_ptr, op=BinaryOp.ADD, left=ptr, right=low)], new_ptr, new_len
        if isinstance(e, t.StrFromByte):
            label = getattr(self.ir_program, 'byte_table_label', None)
            if label is None:
                label = self.ids.new_label("byte_table")
                self.ir_program.string_literals.append((label, ''.join(chr(i) for i in range(256))))
                self.ir_program.byte_table_label = label
            value_ir, value = self.value(e.value)
            base, offset, ptr = self.temp(), self.temp(), self.temp()
            return value_ir + [IRStaticDataAddress(dst=base, label=label), IRCast(dst=offset, src=value),
                               IRBinOp(dst=ptr, op=BinaryOp.ADD, left=base, right=offset)], ptr, IRConst(1, Type.INT)
        if isinstance(e, t.StrFromBytes):
            slice_ir, src, length, _ = self.slice_value(e.value)
            size, ptr = self.temp(Type.INT), self.temp()
            return slice_ir + [
                IRBinOp(dst=size, op=BinaryOp.BITWISE_OR, left=length, right=IRConst(1, Type.INT)),  # never malloc(0)
                IRCall(dst=ptr, name='malloc', args=[size]),
                IRCall(dst=None, name='memcpy', args=[ptr, src, length])], ptr, length
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
        limit = length if limit is None else limit
        ir = []
        if e.high is not None:
            high_ir, high = self.value(e.high)
            ir += high_ir
        else:
            high = length
        if e.low is not None:
            low_ir, low = self.value(e.low)
            ir += low_ir
        else:
            low = IRConst(0, Type.INT)
        return ir + [IRSliceBoundsCheck(value=low, bound=limit), IRSliceBoundsCheck(value=high, bound=limit),
                     IRSliceBoundsCheck(value=low, bound=high)], low, high

    # -- slices: (ptr, len, cap)

    def write_slice(self, dst, ptr, length, cap) -> list:
        len_ir, len_address = self.offset(dst, 8)
        cap_ir, cap_address = self.offset(dst, 16)
        return [IRStore(address=dst, value=ptr, value_type=Type.INT64)] + len_ir + [
            IRStore(address=len_address, value=length, value_type=Type.INT)] + cap_ir + [
            IRStore(address=cap_address, value=cap, value_type=Type.INT)]

    def slice_value(self, e) -> tuple:
        if isinstance(e, t.EmptySlice):
            return [], IRConst(0, Type.INT64), IRConst(0, Type.INT), IRConst(0, Type.INT)
        if isinstance(e, t.SliceLiteral):
            element_type = e.type.element_type
            count = len(e.elements)
            ptr = self.temp()
            ir = [IRCall(dst=ptr, name='malloc', args=[IRConst(count * self.width(element_type), Type.INT64)])]
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
                    ir = [IRCall(dst=ptr, name='malloc', args=[IRConst(self.width(e.base.type), Type.INT64)])] + \
                        self.write_into(ptr, e.base)
                length = cap = IRConst(e.base.type.size, Type.INT)
            else:
                ir, ptr, length, cap = self.slice_value(e.base)
            # A slice may extend up to the capacity, as in Go.
            bounds_ir, low, high = self.bounds(e, length, cap)
            width = self.width(e.type.element_type)
            scaled, new_ptr, new_len, new_cap = self.temp(Type.INT), self.temp(), self.temp(Type.INT), self.temp(Type.INT)
            return ir + bounds_ir + [
                IRBinOp(dst=scaled, op=BinaryOp.MULTIPLY, left=low, right=IRConst(width, Type.INT)),
                IRBinOp(dst=new_ptr, op=BinaryOp.ADD, left=ptr, right=scaled),
                IRBinOp(dst=new_len, op=BinaryOp.SUBTRACT, left=high, right=low),
                IRBinOp(dst=new_cap, op=BinaryOp.SUBTRACT, left=cap, right=low)], new_ptr, new_len, new_cap
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
        return ir + [IRLoad(dst=ptr, address=address)] + len_ir + [IRLoad(dst=length, address=len_address)] + \
            cap_ir + [IRLoad(dst=cap, address=cap_address)], ptr, length, cap

    # -- scalar values

    def pointer(self, expr) -> tuple:
        """A pointer about to be dereferenced: checked not to be none."""
        ir, value = self.value(expr)
        return ir + [IRNullCheck(value)], value

    def value(self, e, discard: bool = False) -> tuple:
        """(IR, value) for a scalar expression (or a void call when `discard`)."""
        if isinstance(e, (t.StrRawPtr, t.StrRawLen)):
            ir, ptr, length = self.str_value(e.value)
            return ir, ptr if isinstance(e, t.StrRawPtr) else length
        if isinstance(e, t.Call):
            return self.call(e, discard)
        if isinstance(e, t.Print):
            return self.print_(e), None
        if isinstance(e, t.DictDelete):
            return self.dict_call(e.dict, e.key, 'delete')[0], None
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
        if isinstance(e, t.BoolLit):
            return [], IRConst(1 if e.value else 0, Type.BOOL)
        if isinstance(e, t.NoneLit):
            return [], IRConst(0, Type.INT64)  # a null pointer is the address 0
        if isinstance(e, t.ZeroValue):
            return [], IRConst(0, Type.INT64 if e.type.kind == TypeKind.POINTER else e.type)
        if isinstance(e, t.Local):
            temp, heap = self.bind(e.symbol)
            if not heap:
                return [], temp
            loaded = self.temp(e.type)
            return [IRLoad(dst=loaded, address=temp)], loaded
        if isinstance(e, (t.Deref, t.FieldAccess, t.ArrayIndex, t.SliceIndex, t.Payload, t.DictLookup)):
            ir, address = self.place_address(e)
            loaded = self.temp(e.type)
            return ir + [IRLoad(dst=loaded, address=address)], loaded
        if isinstance(e, t.StrIndex):
            ir, ptr, length = self.str_value(e.base)
            index_ir, index = self.value(e.index)
            address, loaded = self.temp(), self.temp(Type.UINT8)
            return ir + index_ir + [IRBoundsCheck(index=index, length=length),
                                    IRBinOp(dst=address, op=BinaryOp.ADD, left=ptr, right=index),
                                    IRLoad(dst=loaded, address=address)], loaded
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
            tag, result = self.temp(Type.INT32), self.temp(Type.BOOL)
            index = self.ir_program.sum_type_registry[e.sum.type.sum_type_name].variants.index(e.variant)
            return ir + [IRLoad(dst=tag, address=address),
                         IRBinOp(dst=result, op=BinaryOp.EQUAL, left=tag, right=IRConst(index, Type.INT32))], result
        if isinstance(e, t.StrCompare):
            return self.str_compare(e)
        if isinstance(e, t.AddressOf):
            return self.address_of(e)
        if isinstance(e, t.BoxVariant):
            box = self.temp()
            return [IRCall(dst=box, name='malloc', args=[IRConst(self.width(e.value.type), Type.INT64)])] + \
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
            return left_ir + right_ir + [IRBinOp(dst=result, op=e.op, left=left, right=right)], result
        raise NotYetPorted(type(e).__name__)

    def address_of(self, e: t.AddressOf) -> tuple:
        place = e.place
        if isinstance(place, t.StructLiteral):
            # `&S(...)`: its own storage, on the heap when the pointer may outlive this frame.
            box = self.temp()
            return [IRCall(dst=box, name='malloc', args=[IRConst(self.width(place.type), Type.INT64)])] + \
                self.write_into(box, place), box
        if isinstance(place, t.Local) and _scalar(place.type):
            temp, heap = self.bind(place.symbol)
            if heap:
                return [], temp
        return self.place_address(place)

    def str_compare(self, e: t.StrCompare) -> tuple:
        left_ir, left_ptr, left_len = self.str_value(e.left)
        right_ir, right_ptr, right_len = self.str_value(e.right)
        same_length, cmp, result = self.temp(Type.BOOL), self.temp(Type.INT32), self.temp(Type.BOOL)
        equal_len, differ, end = (self.ids.new_label(x) for x in ("str_len_equal", "str_len_differ", "str_cmp_end"))
        return left_ir + right_ir + [
            IRBinOp(dst=same_length, op=BinaryOp.EQUAL, left=left_len, right=right_len),
            IRBranch(cond=same_length, true_label=equal_len, false_label=differ),
            IRLabel(equal_len),
            IRCall(dst=cmp, name='memcmp', args=[left_ptr, right_ptr, left_len]),
            IRBinOp(dst=result, op=e.op, left=cmp, right=IRConst(0, Type.INT32)), IRJump(end),
            IRLabel(differ), IRMove(dst=result, src=IRConst(0 if e.op == BinaryOp.EQUAL else 1, Type.BOOL)),
            IRJump(end), IRLabel(end)], result

    def short_circuit(self, e: t.Binary) -> tuple:
        """`and`/`or`: the right operand runs only when the left doesn't decide."""
        is_and = e.op == BinaryOp.AND
        rhs, done, end = (self.ids.new_label(x) for x in ("logic_rhs", "logic_short", "logic_end"))
        result = self.temp(Type.BOOL)
        left_ir, left = self.value(e.left)
        right_ir, right = self.value(e.right)
        return left_ir + [
            IRBranch(cond=left, true_label=rhs if is_and else done, false_label=done if is_and else rhs),
            IRLabel(rhs), *right_ir, IRMove(dst=result, src=right), IRJump(end),
            IRLabel(done), IRMove(dst=result, src=IRConst(0 if is_and else 1, Type.BOOL)), IRJump(end),
            IRLabel(end)], result

    def call(self, e: t.Call, discard: bool, destination=None) -> tuple:
        """A call; a composite result is written to `destination` (or a new temporary)."""
        if not _ported(e.type):
            raise NotYetPorted(f"call returning {e.type}")
        ir, args = [], []
        composite_result = e.type != Type.VOID and not _scalar(e.type)
        if composite_result:
            if destination is None:
                scratch_ir, destination = self.scratch_address(e.type)
                ir += scratch_ir
            args.append(destination)
        for a in e.args:
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
                arg_ir, address = self.address(a)  # the callee copies it
                ir += arg_ir
                args.append(address)
            else:
                raise NotYetPorted(f"argument of type {a.type}")
        result = self.temp(e.type) if e.type != Type.VOID and _scalar(e.type) else None
        ir.append(IRCall(dst=result, name=e.name, args=args))
        return ir, destination if composite_result else result

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
        return ir + [IRStaticDataAddress(dst=desc, label=label), IRCall(dst=None, name='hornet_print', args=[address, desc])]

    # -- loops

    def fresh_loop_variable(self, s: t.For) -> list:
        """A for-loop variable whose address may outlive an iteration gets new storage each iteration,
        carrying its value over, so a pointer taken in one iteration keeps that iteration's value."""
        if not isinstance(s.init, t.Declare) or not self.heap(s.init.symbol):
            return []
        symbol = s.init.symbol
        temp, _ = self.bind(symbol)
        box = self.temp()
        ir = [IRCall(dst=box, name='malloc', args=[IRConst(self.width(symbol.type), Type.INT64)])]
        if _scalar(symbol.type):
            current = self.temp(symbol.type)
            return ir + [IRLoad(dst=current, address=temp), IRStore(address=box, value=current, value_type=symbol.type),
                         IRMove(dst=temp, src=box)]
        old_ir, old = self.local_address(symbol)
        slot_address = self.temp()
        return ir + old_ir + [IRCopy(dst_address=box, src_address=old, value_type=symbol.type),
                              IRLocalAddress(dst=slot_address, slot=self.ir_fn.var_slots[symbol.id]),
                              IRStore(address=slot_address, value=box, value_type=Type.INT64)]

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

    def panic_if(self, cond, message: str) -> list:
        """Panic with `message` when `cond` holds."""
        bad, ok = self.ids.new_label("check_failed"), self.ids.new_label("check_ok")
        label, text = self.ids.new_label("panic_msg"), self.temp()
        self.ir_program.string_literals.append((label, message))
        return [IRBranch(cond=cond, true_label=bad, false_label=ok), IRLabel(bad),
                IRStaticDataAddress(dst=text, label=label), IRCall(dst=None, name='hornet_panic', args=[text]),
                IRJump(ok), IRLabel(ok)]

    def for_in(self, s: t.ForIn) -> list:
        if s.kind == 'dict':
            return self.for_in_dict(s)
        start, body, step, end = (self.ids.new_label(x) for x in ("for_in_start", "for_in_body", "for_in_next", "for_in_end"))
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
            ir += [IRLoad(dst=now, address=recheck[0]), IRBinOp(dst=moved, op=BinaryOp.NOT_EQUAL, left=now, right=recheck[1])]
            ir += self.panic_if(moved, "for ... in: slice was reallocated (e.g. by append) during iteration")
        element_ir, element = self.element(base, i, element_type)
        ir += element_ir
        bindings = list(s.bindings)
        if len(bindings) == 2:
            ir += self.initialize_scalar(bindings.pop(0), i)
        ir += self.bind_from(bindings[0], element)
        self.loops.append((step, end))
        ir += self.block(s.body)
        self.loops.pop()
        return ir + [IRJump(step), IRLabel(step), IRBinOp(dst=i, op=BinaryOp.ADD, left=i, right=IRConst(1, Type.INT)),
                     IRJump(start), IRLabel(end)]

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
        ir += [IRMove(dst=i, src=IRConst(0, Type.INT64)), IRJump(start), IRLabel(start),
               IRBinOp(dst=more, op=BinaryOp.LESS_THAN, left=i, right=capacity),
               IRBranch(cond=more, true_label=body, false_label=end), IRLabel(body),
               IRLoad(dst=now, address=header), IRBinOp(dst=moved, op=BinaryOp.NOT_EQUAL, left=now, right=buckets)]
        ir += self.panic_if(moved, "for ... in: dict's own buckets were reallocated (e.g. by an insert that "
                                   "triggered growth) during iteration")
        ir += [IRBinOp(dst=scaled, op=BinaryOp.MULTIPLY, left=i, right=IRConst(1 + key_width + value_width, Type.INT64)),
               IRBinOp(dst=bucket, op=BinaryOp.ADD, left=buckets, right=scaled),
               IRLoad(dst=state, address=bucket),
               IRBinOp(dst=occupied, op=BinaryOp.EQUAL, left=state, right=IRConst(_DICT_BUCKET_OCCUPIED, Type.UINT8)),
               IRBranch(cond=occupied, true_label=entry, false_label=step), IRLabel(entry)]
        key_ir, key_address = self.offset(bucket, 1)
        ir += key_ir + self.bind_from(s.bindings[0], key_address)
        if len(s.bindings) == 2:
            value_ir, value_address = self.offset(key_address, key_width)
            ir += value_ir + self.bind_from(s.bindings[1], value_address)
        self.loops.append((step, end))
        ir += self.block(s.body)
        self.loops.pop()
        return ir + [IRJump(step), IRLabel(step), IRBinOp(dst=i, op=BinaryOp.ADD, left=i, right=IRConst(1, Type.INT64)),
                     IRJump(start), IRLabel(end)]

    # -- dicts: a dict value is a pointer to its header

    def dict_header(self, e) -> tuple:
        ir, address = self.address(e)
        header = self.temp()
        return ir + [IRLoad(dst=header, address=address)], header

    def dict_call(self, dict_expr, key, operation: str, result_type=None, extra=()) -> tuple:
        """hornet_dict_<operation>_{str,scalar}_key on `dict_expr` and `key`."""
        dict_type = dict_expr.type
        value_width = IRConst(self.width(dict_type.element_type), Type.INT64)
        ir, header = self.dict_header(dict_expr)
        if dict_type.key_type.kind == TypeKind.STR:
            key_ir, ptr, length = self.str_value(key)
            args = [header, value_width, ptr, length, *extra]
            name = f'hornet_dict_{operation}_str_key'
        else:
            key_ir, value = self.value(key)
            scratch_ir, key_address = self.scratch_address(dict_type.key_type)
            key_ir += scratch_ir + [IRStore(address=key_address, value=value, value_type=dict_type.key_type)]
            args = [header, IRConst(self.width(dict_type.key_type), Type.INT64), value_width, key_address, *extra]
            name = f'hornet_dict_{operation}_scalar_key'
        result = None if result_type is None else self.temp(result_type)
        return ir + key_ir + [IRCall(dst=result, name=name, args=args)], result

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
        ir = [IRCall(dst=buckets, name='calloc', args=[IRConst(capacity, Type.INT64), IRConst(stride, Type.INT64)])]
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
        ir.append(IRCall(dst=header, name='malloc', args=[IRConst(_DICT_HEADER_SIZE, Type.INT64)]))
        for offset, value, value_type_ in fields:
            offset_ir, address = self.offset(header, offset)
            ir += offset_ir + [IRStore(address=address, value=value, value_type=value_type_)]
        return ir + [IRStore(address=dst, value=header, value_type=Type.INT64)]

    # -- slices

    def read_slice(self, address) -> tuple:
        ptr, length, cap = self.temp(), self.temp(Type.INT), self.temp(Type.INT)
        len_ir, len_address = self.offset(address, 8)
        cap_ir, cap_address = self.offset(address, 16)
        return [IRLoad(dst=ptr, address=address)] + len_ir + [IRLoad(dst=length, address=len_address)] + \
            cap_ir + [IRLoad(dst=cap, address=cap_address)], ptr, length, cap

    def append(self, e: t.Append) -> tuple:
        """Write in place when there is spare capacity; otherwise grow (cap 0 -> 1, doubling below 256,
        then by a quarter) and copy."""
        element_type = e.type.element_type
        width = IRConst(self.width(element_type), Type.INT)
        ir, ptr, length, cap = self.slice_value(e.slice)
        out_ptr, out_len, out_cap = self.temp(), self.temp(Type.INT), self.temp(Type.INT)
        reuse, grow, end = (self.ids.new_label(x) for x in ("append_reuse", "append_grow", "append_end"))
        full, new_len = self.temp(Type.BOOL), self.temp(Type.INT)
        ir += [IRBinOp(dst=full, op=BinaryOp.GREATER_THAN_OR_EQUAL, left=length, right=cap),
               IRBinOp(dst=new_len, op=BinaryOp.ADD, left=length, right=IRConst(1, Type.INT)),
               IRBranch(cond=full, true_label=grow, false_label=reuse)]

        def write_at(base) -> list:
            element_ir, address = self.element(base, length, element_type)
            return element_ir + self.store(address, e.value)

        ir += [IRLabel(reuse)] + write_at(ptr) + [IRMove(dst=out_ptr, src=ptr), IRMove(dst=out_cap, src=cap),
                                                  IRMove(dst=out_len, src=new_len), IRJump(end)]
        new_cap, grown, is_zero, is_small, quarter = (self.temp(Type.INT), self.temp(), self.temp(Type.BOOL),
                                                      self.temp(Type.BOOL), self.temp(Type.INT))
        zero, nonzero, small, large, ready = (self.ids.new_label(x) for x in
                                              ("cap_zero", "cap_nonzero", "cap_small", "cap_large", "cap_ready"))
        ir += [IRLabel(grow),
               IRBinOp(dst=is_zero, op=BinaryOp.EQUAL, left=cap, right=IRConst(0, Type.INT)),
               IRBranch(cond=is_zero, true_label=zero, false_label=nonzero),
               IRLabel(zero), IRMove(dst=new_cap, src=IRConst(1, Type.INT)), IRJump(ready),
               IRLabel(nonzero), IRBinOp(dst=is_small, op=BinaryOp.LESS_THAN, left=cap, right=IRConst(256, Type.INT)),
               IRBranch(cond=is_small, true_label=small, false_label=large),
               IRLabel(small), IRBinOp(dst=new_cap, op=BinaryOp.ADD, left=cap, right=cap), IRJump(ready),
               IRLabel(large), IRBinOp(dst=quarter, op=BinaryOp.SHIFT_RIGHT, left=cap, right=IRConst(2, Type.INT)),
               IRBinOp(dst=new_cap, op=BinaryOp.ADD, left=cap, right=quarter), IRJump(ready),
               IRLabel(ready), IRCall(dst=grown, name='hornet_slice_grow', args=[ptr, length, new_cap, width])]
        ir += write_at(grown) + [IRMove(dst=out_ptr, src=grown), IRMove(dst=out_cap, src=new_cap),
                                 IRMove(dst=out_len, src=new_len), IRJump(end), IRLabel(end)]
        return ir, out_ptr, out_len, out_cap

    # -- comparisons of composites

    def composite_equality(self, e: t.Binary) -> tuple:
        if e.op not in (BinaryOp.EQUAL, BinaryOp.NOT_EQUAL):
            raise NotYetPorted(f"{e.op.name} on {e.left.type}")
        left_ir, left = self.address(e.left)
        right_ir, right = self.address(e.right)
        mismatch, end = self.ids.new_label("eq_mismatch"), self.ids.new_label("eq_end")
        result = self.temp(Type.BOOL)
        equal = 1 if e.op == BinaryOp.EQUAL else 0
        return left_ir + right_ir + self.equal_or_jump(left, right, e.left.type, mismatch) + [
            IRMove(dst=result, src=IRConst(equal, Type.BOOL)), IRJump(end),
            IRLabel(mismatch), IRMove(dst=result, src=IRConst(1 - equal, Type.BOOL)), IRJump(end), IRLabel(end)], result

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
        if type_.kind == TypeKind.STR:
            l_ir, l_ptr, l_len = self.read_str(left)
            r_ir, r_ptr, r_len = self.read_str(right)
            same_len, cmp, differs = self.temp(Type.BOOL), self.temp(Type.INT32), self.temp(Type.BOOL)
            lengths_ok = self.ids.new_label("eq_len_same")
            return l_ir + r_ir + [IRBinOp(dst=same_len, op=BinaryOp.EQUAL, left=l_len, right=r_len),
                                  IRBranch(cond=same_len, true_label=lengths_ok, false_label=mismatch), IRLabel(lengths_ok),
                                  IRCall(dst=cmp, name='memcmp', args=[l_ptr, r_ptr, l_len]),
                                  IRBinOp(dst=differs, op=BinaryOp.NOT_EQUAL, left=cmp, right=IRConst(0, Type.INT32)),
                                  IRBranch(cond=differs, true_label=mismatch, false_label=same), IRLabel(same)]
        if not _scalar(type_):
            raise NotYetPorted(f"equality of {type_}")
        l_value, r_value, differs = self.temp(type_), self.temp(type_), self.temp(Type.BOOL)
        return [IRLoad(dst=l_value, address=left), IRLoad(dst=r_value, address=right),
                IRBinOp(dst=differs, op=BinaryOp.NOT_EQUAL, left=l_value, right=r_value),
                IRBranch(cond=differs, true_label=mismatch, false_label=same), IRLabel(same)]

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
        start, body, nope, found, end = (self.ids.new_label(x) for x in ("in_start", "in_body", "in_next", "in_found", "in_end"))
        element_ir, element = self.element(base, i, element_type)
        return ir + [IRMove(dst=i, src=IRConst(0, Type.INT)), IRJump(start), IRLabel(start),
                     IRBinOp(dst=more, op=BinaryOp.LESS_THAN, left=i, right=length),
                     IRBranch(cond=more, true_label=body, false_label=end), IRLabel(body)] + element_ir + \
            self.equal_or_jump(element, needle, element_type, nope) + [IRJump(found), IRLabel(nope),
            IRBinOp(dst=i, op=BinaryOp.ADD, left=i, right=IRConst(1, Type.INT)), IRJump(start), IRLabel(found),
            IRMove(dst=result, src=IRConst(1, Type.BOOL)), IRJump(start + "_done"), IRLabel(end),
            IRMove(dst=result, src=IRConst(0, Type.BOOL)), IRJump(start + "_done"), IRLabel(start + "_done")], result

    def concat(self, left_ptr, left_len, right_ptr, right_len) -> tuple:
        """A new string holding both: (ir, ptr, len)."""
        total, buffer, right_dst = self.temp(Type.INT), self.temp(), self.temp()
        return [IRBinOp(dst=total, op=BinaryOp.ADD, left=left_len, right=right_len),
                IRCall(dst=buffer, name='malloc', args=[total]),
                IRCall(dst=None, name='memcpy', args=[buffer, left_ptr, left_len]),
                IRBinOp(dst=right_dst, op=BinaryOp.ADD, left=buffer, right=left_len),
                IRCall(dst=None, name='memcpy', args=[right_dst, right_ptr, right_len])], buffer, total
