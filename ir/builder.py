"""IRFunctionBuilder: builds one function's IR from the analyzed AST. Feature logic lives in the mixins."""

from typing import List, Optional

from escape_analysis import analyze_array_escapes, is_heap_allocated
from ir.errors import IRError
from ir.utils import COMPOSITE_KINDS, is_composite_addressable, type_of
from typesys import type_byte_width
from ir.ir import (
    IRBranch, IRCall, IRCast, IRConst, IRCopy, IRFunction, IRJump, IRLocalAddress, IRReadArgument, IRReturn, IRStore, Temp,
)
from ir.arrays_slices import ArraysSlicesMixin
from ir.dicts import DictsMixin
from ir.dispatch import DispatchMixin
from ir.pointers import PointersMixin
from ir.scalars import ScalarsMixin
from ir.statements import StatementsMixin
from ir.strings import StringsMixin
from ir.structs import StructsMixin
from ir.sum_types import SumTypesMixin
from parser import (
    ArrayLiteral,
    Assign,
    Binary,
    Call,
    ExprStmt,
    Field,
    FieldAssign,
    For,
    ForIn,
    Function,
    If,
    Index,
    IndexAssign,
    IsCheck,
    Node,
    Param,
    Return,
    Slice,
    Unary,
    VarDecl,
    Variable,
    While,
)
from ops import UnaryOp
from typesys import Type, TypeKind


class IRFunctionBuilder(
        ArraysSlicesMixin,
        DictsMixin,
        DispatchMixin,
        PointersMixin,
        ScalarsMixin,
        StatementsMixin,
        StringsMixin,
        StructsMixin,
        SumTypesMixin):
    def __init__(self, ir_program):
        self.ir_program = ir_program
        self.loop_labels: List[tuple] = []  # (continue_label, end_label), innermost last

    def gen_function_ir(self, fn: Function) -> IRFunction:
        """Build `fn`'s IRFunction: slots, params, body."""
        # id(ArrayLiteral or Call) -> slot
        self._argument_temp_slots = {}
        ir_fn = IRFunction(name=fn.name)
        self.ir_fn = ir_fn
        ir_fn.return_type = fn.resolved_return_type
        return_type = ir_fn.return_type
        param_types = [p.resolved_type for p in fn.params]

        # Declarations needing heap storage (size or escape).
        self._escaping_decl_ids = analyze_array_escapes(fn, self.ir_program.struct_registry, self.ir_program.escape_summaries)

        # Composite returns take a hidden result pointer as argument 0.
        arg_shift = 0
        if return_type.kind in COMPOSITE_KINDS:
            ir_fn.hidden_return_ptr_slot = self.ir_program.ids.new_slot(8, "hidden_return_ptr", ir_fn)
            arg_shift = 1

        # Per-function scratch slots for print/dict arguments.
        self._unnamed_slice_temp_slot = self.ir_program.ids.new_slot(24, "unnamed_slice_temp", ir_fn)

        self._unnamed_dict_temp_slot = self.ir_program.ids.new_slot(32, "unnamed_dict_temp", ir_fn)

        self._dict_key_scratch_slot = self.ir_program.ids.new_slot(8, "dict_key_scratch", ir_fn)

        self._print_scalar_temp_slot = self.ir_program.ids.new_slot(8, "print_scalar_temp", ir_fn)

        self._unnamed_str_temp_slot = self.ir_program.ids.new_slot(16, "unnamed_str_temp", ir_fn)

        self._collect_params(fn.params, ir_fn)
        self._collect_locals(fn.body, ir_fn)
        self._collect_argument_temps(fn.body, ir_fn)
        self.locals = {}

        param_setup_ir = self._ir_param_setup(fn, param_types, arg_shift, ir_fn)

        statement_ir = []
        for stmt in fn.body:
            statement_ir.extend(self.gen_statement_ir(stmt, ir_fn))
        ir_fn.body = param_setup_ir + statement_ir
        ir_fn.return_type = return_type
        if not ir_fn.body or not isinstance(ir_fn.body[-1], (IRBranch, IRJump, IRReturn)):
            # Void functions may fall off the end; unreachable trailing code needs a terminator too.
            ir_fn.body.append(IRReturn(value=None))
        return ir_fn

    def _ir_param_setup(self, fn: Function, param_types: List[Type], arg_shift: int, ir_fn: IRFunction) -> list:
        """IR for the hidden return pointer and parameters."""
        ir = []
        reg_index = arg_shift
        hidden_ptr = None
        if ir_fn.hidden_return_ptr_slot is not None:
            hidden_ptr = self.ir_program.ids.new_temp(Type.INT64)
            ir.append(IRReadArgument(dst=hidden_ptr, index=0))
        captured = []
        for p, p_type in zip(fn.params, param_types):
            if p_type.kind == TypeKind.SLICE:
                ptr_value = self.ir_program.ids.new_temp(Type.INT64)
                len_value = self.ir_program.ids.new_temp(Type.INT)
                cap_value = self.ir_program.ids.new_temp(Type.INT)
                ir.append(IRReadArgument(dst=ptr_value, index=reg_index))
                ir.append(IRReadArgument(dst=len_value, index=reg_index + 1))
                ir.append(IRReadArgument(dst=cap_value, index=reg_index + 2))
                reg_index += 3
                captured.append((ptr_value, len_value, cap_value))
            elif p_type.kind == TypeKind.STR:
                # str: {ptr, len} in two argument slots.
                ptr_value = self.ir_program.ids.new_temp(Type.INT64)
                len_value = self.ir_program.ids.new_temp(Type.INT)
                ir.append(IRReadArgument(dst=ptr_value, index=reg_index))
                ir.append(IRReadArgument(dst=len_value, index=reg_index + 1))
                reg_index += 2
                captured.append((ptr_value, len_value))
            elif p_type.kind in (TypeKind.ARRAY, TypeKind.STRUCT, TypeKind.SUM, TypeKind.DICT):
                caller_ptr = self.ir_program.ids.new_temp(Type.INT64)
                ir.append(IRReadArgument(dst=caller_ptr, index=reg_index))
                reg_index += 1
                captured.append(caller_ptr)
            else:
                self._bind_param(p, ir_fn)
                incoming = self.ir_program.ids.new_temp(p_type)
                if fn.name == 'main' and reg_index == 0 and p_type == Type.INT:
                    # C passes argc as a 32-bit int; sign-extend it.
                    argc = self.ir_program.ids.new_temp(Type.INT32)
                    ir.append(IRReadArgument(dst=argc, index=reg_index))
                    ir.append(IRCast(dst=incoming, src=argc))
                else:
                    ir.append(IRReadArgument(dst=incoming, index=reg_index))
                ir.extend(self._ir_finish_scalar_var_decl(p.name, id(p), p_type, incoming))
                reg_index += 1
                captured.append(None)

        if hidden_ptr is not None:
            hidden_ptr_addr = self.ir_program.ids.new_temp(Type.INT64)
            ir.append(IRLocalAddress(dst=hidden_ptr_addr, slot=ir_fn.hidden_return_ptr_slot))
            ir.append(IRStore(address=hidden_ptr_addr, value=hidden_ptr, value_type=Type.INT64))

        for p, p_type, cap in zip(fn.params, param_types, captured):
            if p_type.kind == TypeKind.SLICE:
                ptr_value, len_value, cap_value = cap
                slot = self._bind_param(p, ir_fn)
                if self._is_heap_allocated(id(p), p_type):
                    ir.extend(self._ir_malloc_and_store(p_type, slot))
                ir_dst, param_addr = self._ir_slice_address(Variable(name=p.name, decl_id=id(p)))
                ir.extend(ir_dst)
                ir.extend(self._ir_write_slice_descriptor_into_address(param_addr, ptr_value, len_value, cap_value))
            elif p_type.kind == TypeKind.STR:
                ptr_value, len_value = cap
                slot = self._bind_param(p, ir_fn)
                if self._is_heap_allocated(id(p), p_type):
                    size = type_byte_width(p_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
                    new_ptr = self.ir_program.ids.new_temp(Type.INT64)
                    ir.append(IRCall(dst=new_ptr, name='malloc', args=[IRConst(size, Type.INT64)]))
                    param_addr = self.ir_program.ids.new_temp(Type.INT64)
                    ir.append(IRLocalAddress(dst=param_addr, slot=slot))
                    ir.append(IRStore(address=param_addr, value=new_ptr, value_type=Type.INT64))
                    ir.extend(self._ir_write_str_descriptor_into_address(new_ptr, ptr_value, len_value))
                else:
                    param_addr = self.ir_program.ids.new_temp(Type.INT64)
                    ir.append(IRLocalAddress(dst=param_addr, slot=slot))
                    ir.extend(self._ir_write_str_descriptor_into_address(param_addr, ptr_value, len_value))
            elif p_type.kind in (TypeKind.ARRAY, TypeKind.STRUCT, TypeKind.SUM, TypeKind.DICT):
                caller_ptr = cap
                slot = self._bind_param(p, ir_fn)
                param_addr = self.ir_program.ids.new_temp(Type.INT64)
                ir.append(IRLocalAddress(dst=param_addr, slot=slot))
                if self._is_heap_allocated(id(p), p_type):
                    size = type_byte_width(p_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
                    new_ptr = self.ir_program.ids.new_temp(Type.INT64)
                    ir.append(IRCall(dst=new_ptr, name='malloc', args=[IRConst(size, Type.INT64)]))
                    ir.append(IRStore(address=param_addr, value=new_ptr, value_type=Type.INT64))
                    ir.append(IRCopy(dst_address=new_ptr, src_address=caller_ptr, value_type=p_type))
                else:
                    ir.append(IRCopy(dst_address=param_addr, src_address=caller_ptr, value_type=p_type))
        return ir

    def _collect_params(self, params: List[Param], ir_fn: IRFunction) -> None:
        """One slot per parameter; heap-allocated params get an 8-byte pointer slot."""
        for p in params:
            p_type = p.resolved_type
            width = 8 if self._is_heap_allocated(id(p), p_type) else type_byte_width(p_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
            ir_fn.var_slots[id(p)] = self.ir_program.ids.new_slot(width, f"param:{p.name}", ir_fn)

    def _bind_param(self, p: Param, ir_fn: IRFunction) -> int:
        """Bind `p` in scope to its slot and Temp."""
        slot = ir_fn.var_slots[id(p)]
        p_type = p.resolved_type
        temp_type = Type(TypeKind.POINTER, element_type=p_type) if self._is_heap_allocated(id(p), p_type) else p_type
        self.locals[id(p)] = (slot, p_type, id(p), self.ir_program.ids.temp_at_offset(temp_type, slot))
        return slot

    def _allocate_local_slot(self, stmt: VarDecl, ir_fn: IRFunction) -> None:
        """Allocate a VarDecl's slot (also used for IsCheck binding_decl)."""
        var_type = stmt.resolved_type
        width = 8 if self._is_heap_allocated(
            id(stmt), var_type) else type_byte_width(var_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
        ir_fn.var_slots[id(stmt)] = self.ir_program.ids.new_slot(width, f"local:{stmt.name}", ir_fn)

    def _allocate_for_in_binding_slot(self, stmt: ForIn, index: int, binding_type: Type, ir_fn: IRFunction) -> None:
        """Allocate a ForIn binding's slot, keyed by (id(stmt), index)."""
        width = 8 if self._is_heap_allocated(
            (id(stmt), index), binding_type) else type_byte_width(binding_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
        ir_fn.var_slots[(id(stmt), index)] = self.ir_program.ids.new_slot(
            width, f"for_in:{stmt.binding_names[index]}", ir_fn)

    def _collect_locals(self, statements: List[Node], ir_fn: IRFunction) -> None:
        """Allocate slots for every VarDecl and binding in `statements`."""
        for stmt in statements:
            if isinstance(stmt, VarDecl):
                self._allocate_local_slot(stmt, ir_fn)
            elif isinstance(stmt, If):
                if isinstance(stmt.condition, IsCheck) and stmt.condition.binding_decl is not None:
                    self._allocate_local_slot(stmt.condition.binding_decl, ir_fn)
                self._collect_locals(stmt.then_body, ir_fn)
                if stmt.else_body is not None:
                    self._collect_locals(stmt.else_body, ir_fn)
            elif isinstance(stmt, While):
                self._collect_locals(stmt.body, ir_fn)
            elif isinstance(stmt, ForIn):
                for i, binding_type in enumerate(stmt.binding_types):
                    self._allocate_for_in_binding_slot(stmt, i, binding_type, ir_fn)
                self._collect_locals(stmt.body, ir_fn)
            elif isinstance(stmt, For):
                self._collect_locals([stmt.init], ir_fn)
                self._collect_locals(stmt.body, ir_fn)

    def _collect_argument_temps(self, statements: List[Node], ir_fn: IRFunction) -> None:
        """Reserve slots for composite temporaries used as call arguments."""
        for stmt in statements:
            if isinstance(stmt, VarDecl):
                if stmt.init is not None:
                    self._collect_argument_temps_in_expr(stmt.init, ir_fn)
            elif isinstance(stmt, Assign):
                self._collect_argument_temps_in_expr(stmt.value, ir_fn)
            elif isinstance(stmt, IndexAssign):
                self._collect_argument_temps_in_expr(stmt.array, ir_fn)
                self._collect_argument_temps_in_expr(stmt.index, ir_fn)
                self._collect_argument_temps_in_expr(stmt.value, ir_fn)
            elif isinstance(stmt, FieldAssign):
                self._collect_argument_temps_in_expr(stmt.base, ir_fn)
                self._collect_argument_temps_in_expr(stmt.value, ir_fn)
            elif isinstance(stmt, Return):
                if stmt.value is not None:
                    self._collect_argument_temps_in_expr(stmt.value, ir_fn)
            elif isinstance(stmt, If):
                self._collect_argument_temps_in_expr(stmt.condition, ir_fn)
                self._collect_argument_temps(stmt.then_body, ir_fn)
                if stmt.else_body is not None:
                    self._collect_argument_temps(stmt.else_body, ir_fn)
            elif isinstance(stmt, While):
                self._collect_argument_temps_in_expr(stmt.condition, ir_fn)
                self._collect_argument_temps(stmt.body, ir_fn)
            elif isinstance(stmt, ForIn):
                self._collect_argument_temps_in_expr(stmt.iterable, ir_fn)
                self._collect_argument_temps(stmt.body, ir_fn)
            elif isinstance(stmt, For):
                self._collect_argument_temps([stmt.init], ir_fn)
                self._collect_argument_temps_in_expr(stmt.condition, ir_fn)
                self._collect_argument_temps(stmt.body, ir_fn)
                self._collect_argument_temps([stmt.increment], ir_fn)
            elif isinstance(stmt, ExprStmt):
                self._collect_argument_temps_in_expr(stmt.expr, ir_fn)
                if isinstance(stmt.expr, Call) and stmt.expr.name in self.ir_program.struct_registry:
                    # A struct literal statement still needs storage.
                    self._reserve_argument_temp(stmt.expr, type_of(stmt.expr), ir_fn)

    def _is_ordinary_composite_call(self, expr: Node) -> bool:
        """Call using the hidden-pointer convention (not struct literals or builtins)."""
        return isinstance(expr, Call) and expr.name != 'append' and expr.name not in self.ir_program.struct_registry

    def _collect_argument_temps_in_expr(self, expr: Optional[Node], ir_fn: IRFunction) -> None:
        """Expression walk for _collect_argument_temps."""
        if expr is None:
            return
        if isinstance(expr, Call):
            param_types = self.ir_program.function_registry[expr.name][0] if expr.name in self.ir_program.function_registry else None
            for i, arg in enumerate(expr.args):
                self._collect_argument_temps_in_expr(arg, ir_fn)
                arg_type = type_of(arg)
                # Struct argument to a sum parameter needs a sum-sized slot.
                reserve_type = arg_type
                if param_types is not None and arg_type.kind == TypeKind.STRUCT and param_types[i].kind == TypeKind.SUM:
                    reserve_type = param_types[i]
                if reserve_type.kind in (TypeKind.ARRAY, TypeKind.STRUCT, TypeKind.SUM, TypeKind.DICT) and not is_composite_addressable(arg):
                    self._reserve_argument_temp(arg, reserve_type, ir_fn)
        elif isinstance(expr, Binary):
            self._collect_argument_temps_in_expr(expr.left, ir_fn)
            self._collect_argument_temps_in_expr(expr.right, ir_fn)
        elif isinstance(expr, Unary):
            self._collect_argument_temps_in_expr(expr.operand, ir_fn)
            if (expr.op == UnaryOp.ADDRESS_OF and isinstance(expr.operand, Call)
                    and expr.operand.name in self.ir_program.struct_registry):
                struct_type = type_of(expr.operand)
                if not self._is_heap_allocated(id(expr.operand), struct_type):
                    width = type_byte_width(struct_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
                    self._argument_temp_slots[id(expr.operand)] = self.ir_program.ids.new_slot(width, "struct_literal_address", ir_fn)
        elif isinstance(expr, Index):
            self._collect_argument_temps_in_expr(expr.array, ir_fn)
            self._collect_argument_temps_in_expr(expr.index, ir_fn)
            array_type = type_of(expr.array)
            if array_type.kind in (TypeKind.ARRAY, TypeKind.SLICE) and (
                    self._is_ordinary_composite_call(expr.array) or isinstance(expr.array, ArrayLiteral)):
                self._reserve_argument_temp(expr.array, array_type, ir_fn)
        elif isinstance(expr, Field):
            self._collect_argument_temps_in_expr(expr.base, ir_fn)
            base_type = type_of(expr.base)
            if base_type.kind == TypeKind.STRUCT and (
                    self._is_ordinary_composite_call(expr.base)
                    or (isinstance(expr.base, Call) and expr.base.name in self.ir_program.struct_registry)):
                self._reserve_argument_temp(expr.base, base_type, ir_fn)
        elif isinstance(expr, Slice):
            self._collect_argument_temps_in_expr(expr.array, ir_fn)
            self._collect_argument_temps_in_expr(expr.low, ir_fn)
            self._collect_argument_temps_in_expr(expr.high, ir_fn)
        elif isinstance(expr, ArrayLiteral):
            for element in expr.elements:
                self._collect_argument_temps_in_expr(element, ir_fn)

    def _reserve_argument_temp(self, expr: Node, t: Type, ir_fn: IRFunction) -> None:
        """Reserve a slot for a composite temporary keyed by id(expr)."""
        if is_heap_allocated(t, self.ir_program.struct_registry, self.ir_program.sum_type_registry):
            return
        width = type_byte_width(t, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
        self._argument_temp_slots[id(expr)] = self.ir_program.ids.new_slot(width, "argument_temp", ir_fn)

    def _bind_local(self, stmt: VarDecl, ir_fn: IRFunction) -> int:
        """Bind `stmt` in scope to its slot and Temp."""
        slot = ir_fn.var_slots[id(stmt)]
        var_type = stmt.resolved_type
        temp_type = Type(TypeKind.POINTER, element_type=var_type) if self._is_heap_allocated(id(stmt), var_type) else var_type
        self.locals[id(stmt)] = (slot, var_type, id(stmt), self.ir_program.ids.temp_at_offset(temp_type, slot))
        return slot

    def _bind_for_in_binding(self, stmt: ForIn, index: int, binding_type: Type, ir_fn: IRFunction) -> int:
        """Bind a ForIn binding in scope."""
        slot = ir_fn.var_slots[(id(stmt), index)]
        temp_type = Type(TypeKind.POINTER, element_type=binding_type) if self._is_heap_allocated((id(stmt), index), binding_type) else binding_type
        self.locals[(id(stmt), index)] = (slot, binding_type, (id(stmt), index), self.ir_program.ids.temp_at_offset(temp_type, slot))
        return slot

    def _decl(self, ref) -> object:
        """Decl id of `ref`: a decl id, VarDecl, Param, or a node annotated with decl_id."""
        if isinstance(ref, (int, tuple)):
            return ref
        if isinstance(ref, (VarDecl, Param)):
            return id(ref)
        decl_id = getattr(ref, 'decl_id', None)
        if decl_id is None:
            raise IRError(f"{ref!r} has no decl_id -- semantic.analyze() must run first")
        return decl_id

    def _var_ref(self, stmt) -> Variable:
        """A Variable naming the local a VarDecl declares or an Assign targets."""
        return Variable(name=stmt.name, decl_id=self._decl(stmt))

    def _local(self, ref) -> tuple:
        """(slot, declared type, decl id, Temp) of a local."""
        decl_id = self._decl(ref)
        if decl_id not in self.locals:
            raise IRError(f"Local {ref!r} is not bound")
        return self.locals[decl_id]

    def _local_slot(self, ref) -> int:
        return self._local(ref)[0]

    def _local_type(self, ref) -> Type:
        return self._local(ref)[1]

    def _local_decl_id(self, ref) -> object:
        return self._local(ref)[2]

    def _local_temp(self, ref) -> Temp:
        return self._local(ref)[3]

    def _is_heap_allocated(self, decl_id: int, t: Type) -> bool:
        """Heap-allocated by size or escape analysis."""
        return is_heap_allocated(t, self.ir_program.struct_registry, self.ir_program.sum_type_registry) or decl_id in self._escaping_decl_ids
