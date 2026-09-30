"""Lowers struct methods to mangled free Functions. Runs before semantic analysis."""

from parser import Function, Param, PointerTypeExpr, Program


def mangle_method_name(struct_name: str, method_name: str) -> str:
    """`Struct.method`; '.' can't appear in identifiers, so no collisions."""
    return f"{struct_name}.{method_name}"


def desugar_methods(program: Program) -> None:
    """Append one Function per method; the receiver becomes the first Param (`*S` for a pointer receiver)."""
    for sd in program.structs:
        for md in sd.methods:
            # Synthesized nodes take the MethodDef's position for error reporting.
            receiver_type = sd.name
            if md.receiver_is_pointer:
                receiver_type = PointerTypeExpr(pointee_type=sd.name, line=md.line, col=md.col, file=md.file)
            receiver_param = Param(name=md.receiver_name, type=receiver_type, line=md.line, col=md.col, file=md.file)
            program.functions.append(Function(
                name=mangle_method_name(sd.name, md.name),
                return_type=md.return_type,
                params=[receiver_param] + md.params,
                body=md.body,
                line=md.line, col=md.col, file=md.file,
            ))
