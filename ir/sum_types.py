"""Sum type value semantics. A sum-typed value is a fixed-size
DISCRIMINANT TAG (SUM_TYPE_TAG_WIDTH bytes, an ordinary int -- see
ir/utils.py's own constant) followed immediately by a payload region
sized to the WIDEST variant (type_byte_width's own SUM case), which
every variant shares -- unlike a struct's own fields, which each get
their own, non-overlapping offset, only one variant's own bytes are
ever meaningful at a time, exactly which one the tag says.

There is no separate variant-CONSTRUCTOR syntax (see SumTypeDef's own
docstring in parser.py): a Circle becomes a Shape purely by being
assigned into one, checked at the semantic level by _types_compatible
(a struct is compatible with a sum type that lists it as one of its
own variants) and lowered here, at the one place every such value
actually gets written: _ir_write_composite_value_into's own dispatch,
which reaches _ir_write_sum_type_value_into below whenever value_type
is SUM-kind, regardless of whether value_expr is a bare struct literal
or an already-struct-typed value -- both are handled uniformly, by
writing the tag once here and then recursing back into that SAME
dispatcher, now with the source struct's own type, to write the
payload exactly as if it were an ordinary struct-typed target (which,
as far as THAT recursive call is concerned, it is)."""

from ir.ir import IRBinOp, IRCall, IRConst, IRLocalAddress, IRStore
from ir.utils import SUM_TYPE_TAG_WIDTH, type_byte_width, type_of
from parser import Node, BinaryOp
from semantic import Type, TypeKind


class SumTypesMixin:
    def _ir_write_sum_type_value_into(self, dst_address, value_expr: Node, sum_type: Type):
        """Builds (without lowering) a struct value widening into a
        sum-typed dst_address as real IR: the tag, then the struct's
        own value one level under it. Returns None only when the
        struct's own value is out of scope for _ir_write_composite_
        value_into (a named/partial struct literal, chiefly) -- the
        identical, mutual-recursion fallback every other composite
        case here already has.

        `sum_type` is the TARGET's own type -- sum_type_registry is
        what supplies the variant list a plain struct type alone
        can't; source_struct_type (value_expr's own type, already
        resolved by semantic.py) is what's actually being widened, and
        the ONE thing that decides which discriminant gets written:
        its own index in sum_type's variant list, the same order
        SumTypeDef.variants -- and this program's own sum_type_
        registry, copied from it unchanged -- already preserves."""
        source_struct_type = type_of(value_expr)
        variants = self.ir_program.sum_type_registry[sum_type.sum_type_name].variants
        discriminant = variants.index(source_struct_type.struct_name)

        tag_ir = [IRStore(address=dst_address, value=IRConst(discriminant, Type.INT), value_type=Type.INT)]

        payload_addr = self.ir_program.ids.new_temp(Type.INT64)
        payload_addr_ir = [IRBinOp(
            dst=payload_addr, op=BinaryOp.ADD,
            left=dst_address, right=IRConst(SUM_TYPE_TAG_WIDTH, Type.INT64),
        )]

        payload_ir = self._ir_write_composite_value_into(payload_addr, value_expr, source_struct_type)
        if payload_ir is None:
            return None
        return tag_ir + payload_addr_ir + payload_ir

    def _ir_materialize_sum_type_value(self, expr: Node, sum_type: Type):
        """Materializes a struct value WIDENING into sum_type as its
        own fresh address -- the sum-type counterpart to _ir_
        materialize_struct_literal (ir/structs.py), sharing its
        identical "reserved slot, or malloc when none was reserved"
        skeleton, but sized for sum_type itself (tag + largest
        variant) rather than expr's own (narrower) struct type, and
        written via _ir_write_sum_type_value_into rather than _ir_
        write_struct_literal_into.

        Used specifically for a struct-literal (or already-struct-
        typed) argument being passed to a sum-typed PARAMETER -- see
        _ir_call_arguments' own dispatch (ir/scalars.py) for where the
        "does this argument need widening" decision gets made, by
        consulting the callee's own declared parameter type
        (function_registry), not just type_of(expr) -- the same
        registry lookup _collect_argument_temps_in_expr (ir/builder.
        py) already has to make first, to reserve a SUM_TYPE_TAG_
        WIDTH-plus-largest-variant slot here rather than a struct-
        sized one, if this argument gets one reserved at all."""
        if id(expr) in self._argument_temp_slots:
            slot = self._argument_temp_slots[id(expr)]
            addr = self.ir_program.ids.new_temp(Type.INT64)
            addr_ir = [IRLocalAddress(dst=addr, slot=slot)]
        else:
            addr = self.ir_program.ids.new_temp(Type.INT64)
            size = type_byte_width(sum_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
            addr_ir = [IRCall(dst=addr, name='malloc', args=[IRConst(size, Type.INT64)])]
        write_ir = self._ir_write_sum_type_value_into(addr, expr, sum_type)
        if write_ir is None:
            return None
        return addr_ir + write_ir, addr
