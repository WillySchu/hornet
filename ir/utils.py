"""AST-facing helpers shared by the IR builder mixins."""

from parser import Node, Field, Index, Slice, Unary, Variable
from ops import UnaryOp
from typesys import Type, TypeKind

from ir.errors import IRError

# Composite types: copied through addresses, not held in registers.
COMPOSITE_KINDS = {TypeKind.ARRAY, TypeKind.SLICE, TypeKind.STRUCT, TypeKind.SUM, TypeKind.STR, TypeKind.DICT}


def is_composite_addressable(expr: Node) -> bool:
    """Whether `expr` has an address to copy a composite value from."""
    if isinstance(expr, (Variable, Field, Index)):
        return True
    return isinstance(expr, Unary) and expr.op == UnaryOp.DEREFERENCE

def root_variable(expr: Node):
    """Variable under an Index/Slice/Field chain, or None."""
    while isinstance(expr, (Index, Slice, Field)):
        expr = expr.base if isinstance(expr, Field) else expr.array
    return expr if isinstance(expr, Variable) else None


def type_of(expr: Node) -> Type:
    """expr.resolved_type; IRError if semantic analysis didn't run."""
    if expr.resolved_type is None:
        raise IRError(
            f"{expr!r} has no resolved type -- semantic.analyze() "
            f"must run before codegen (see compile_to_asm)"
        )
    return expr.resolved_type
