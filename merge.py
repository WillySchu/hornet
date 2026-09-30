"""Merges discovered modules into the entry Program: mangles module declarations to `module$name` and resolves qualified references and visibility, so later passes see one flat Program. Runs before desugar/analyze."""

import re
from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Set, Tuple

from diagnostics import CompileError
from modules import DiscoveredModule
from parser import (
    ArrayTypeExpr,
    Assign,
    Call,
    Field,
    ForIn,
    MethodDef,
    Param,
    VarDecl,
    Node,
    PointerTypeExpr,
    Program,
    QualifiedTypeExpr,
    SliceTypeExpr,
    Variable,
)


class MergeError(CompileError):
    """Unresolvable or private qualified reference."""

    def __init__(self, message: str, file: Optional[str] = None, line: int = 0, col: int = 0):
        clean = re.sub(rf' at line {line}\b', '', message) if line else message
        super().__init__(clean, file, line, col, legacy=message)


# Node fields holding type expressions.
_TYPE_FIELD_NAMES = frozenset({'var_type', 'return_type', 'field_type', 'target_type', 'type', 'type_name', 'const_type'})


def _mangle(canonical_module: str, name: str) -> str:
    """`module$name`; '$' can't appear in identifiers and '.' is taken by methods."""
    return f"{canonical_module}${name}"


def _own_top_level_names(program: Program) -> Set[str]:
    """Top-level names in `program` that get mangled (excludes externs)."""
    names: Set[str] = set()
    for fn in program.functions:
        names.add(fn.name)
    for sd in program.structs:
        names.add(sd.name)
    for ta in program.type_aliases:
        names.add(ta.name)
    for st in program.sum_types:
        names.add(st.name)
    for ic in program.intrinsics:
        names.add(ic.name)
    for cd in program.consts:
        names.add(cd.name)
    return names


def _own_extern_names(program: Program) -> Set[str]:
    """Extern names; never mangled, since they are linker symbols."""
    return {ext.name for ext in program.extern_functions}


def _check_visible(module: DiscoveredModule, name: str, referencing_module: Optional[str], at_line: int) -> None:
    """Raise if `name` is private (leading '_') to referencing_module (None = entry file)."""
    if name.startswith('_') and referencing_module != module.canonical_name:
        raise MergeError(
            f"'{module.canonical_name}.{name}' at line {at_line} is not visible outside "
            f"the module that defines it -- names starting with '_' are private to their "
            f"own module",
            line=at_line,
        )


@dataclass
class _MergeContext:
    modules: Dict[str, DiscoveredModule]
    all_own_names: Dict[str, Set[str]]
    all_extern_names: Dict[str, Set[str]]
    all_const_names: Dict[str, Set[str]]
    # Constants visible by bare name in the file being rewritten: local name -> merged name.
    const_map: Dict[str, str]


def _const_map(program: Program, canonical_module: Optional[str], named_imports: Dict[str, Tuple[str, str]],
               ctx: _MergeContext) -> Dict[str, str]:
    """Bare constant names visible in a file. Locals may not reuse them, so renaming every use is safe."""
    out = {cd.name: (cd.name if canonical_module is None else _mangle(canonical_module, cd.name))
           for cd in program.consts}
    for local_name, (canonical_name, original_name) in named_imports.items():
        if original_name in ctx.all_const_names[canonical_name]:
            out[local_name] = _mangle(canonical_name, original_name)
    return out


def _check_binding(name: str, node: Node, ctx: _MergeContext) -> None:
    if name in ctx.const_map:
        raise MergeError(
            f"'{name}' at line {node.line} is a constant here and can't also be a variable name",
            line=node.line, col=node.col,
        )


def _resolve_in_module(canonical_name: str, name: str, ctx: _MergeContext,
                        referencing_module: Optional[str], at_line: int) -> str:
    """`name` in canonical_name, as rewritten (externs unchanged)."""
    target = ctx.modules[canonical_name]
    if name in ctx.all_extern_names[canonical_name]:
        return name
    if name not in ctx.all_own_names[canonical_name]:
        raise MergeError(
            f"'{name}' at line {at_line} is not declared in module "
            f"{target.canonical_name!r} ({target.file_path})",
            line=at_line,
        )
    _check_visible(target, name, referencing_module, at_line)
    return _mangle(canonical_name, name)


def _resolve_qualified(
        alias: str, name: str, import_aliases: Dict[str, str], ctx: _MergeContext,
        referencing_module: Optional[str], at_line: int) -> Optional[str]:
    """Resolve `alias.name` from a file with the given import aliases."""
    canonical_name = import_aliases.get(alias)
    if canonical_name is None:
        return None
    return _resolve_in_module(canonical_name, name, ctx, referencing_module, at_line)


def _resolve_named(
        local_name: str, named_imports: Dict[str, Tuple[str, str]], ctx: _MergeContext,
        referencing_module: Optional[str], at_line: int) -> Optional[str]:
    """Resolve a bare name via the file's named imports, or None."""
    target = named_imports.get(local_name)
    if target is None:
        return None
    canonical_name, original_name = target
    return _resolve_in_module(canonical_name, original_name, ctx, referencing_module, at_line)


def _rewrite_type_expr(type_expr, own_names: Set[str], canonical_module: Optional[str],
                        import_aliases: Dict[str, str], named_imports: Dict[str, Tuple[str, str]],
                        ctx: _MergeContext):
    """Rewrite QualifiedTypeExprs, recursing through array/slice/pointer wrappers."""
    if isinstance(type_expr, QualifiedTypeExpr):
        resolved = _resolve_qualified(type_expr.module, type_expr.name, import_aliases, ctx, canonical_module,
                                       type_expr.line)
        if resolved is None:
            raise MergeError(
                f"'{type_expr.module}' at line {type_expr.line} doesn't name an imported "
                f"module",
                line=type_expr.line,
            )
        return resolved
    if isinstance(type_expr, (ArrayTypeExpr, SliceTypeExpr, PointerTypeExpr)):
        field_name = 'pointee_type' if isinstance(type_expr, PointerTypeExpr) else 'element_type'
        setattr(type_expr, field_name, _rewrite_type_expr(
            getattr(type_expr, field_name), own_names, canonical_module, import_aliases, named_imports, ctx))
        return type_expr
    if isinstance(type_expr, str) and type_expr in own_names:
        return _mangle(canonical_module, type_expr)
    if isinstance(type_expr, str):
        resolved = _resolve_named(type_expr, named_imports, ctx, canonical_module, 0)
        if resolved is not None:
            return resolved
    return type_expr


def _rewrite_node(node, own_names: Set[str], canonical_module: Optional[str], import_aliases: Dict[str, str],
                   named_imports: Dict[str, Tuple[str, str]], ctx: _MergeContext):
    """Rewrite `node` and descendants in place; returns the (possibly replaced) node."""
    if isinstance(node, Call) and node.receiver is not None and isinstance(node.receiver, Variable):
        resolved = _resolve_qualified(node.receiver.name, node.name, import_aliases, ctx, canonical_module, node.line)
        if resolved is not None:
            node.name = resolved
            node.receiver = None
            node.args = [
                _rewrite_node(a, own_names, canonical_module, import_aliases, named_imports, ctx)
                for a in node.args
            ]
            if node.kwargs is not None:
                node.kwargs = [
                    (k, _rewrite_node(v, own_names, canonical_module, import_aliases, named_imports, ctx))
                    for k, v in node.kwargs
                ]
            return node
    if isinstance(node, Field) and isinstance(node.base, Variable):
        resolved = _resolve_qualified(node.base.name, node.name, import_aliases, ctx, canonical_module, node.line)
        if resolved is not None:
            return Variable(name=resolved, line=node.line, col=node.col)
    if isinstance(node, Variable) and node.name in ctx.const_map:
        node.name = ctx.const_map[node.name]
    if isinstance(node, Assign) and node.name in ctx.const_map:
        node.name = ctx.const_map[node.name]  # semantic analysis rejects assigning to a constant
    if isinstance(node, (VarDecl, Param)):
        _check_binding(node.name, node, ctx)
    if isinstance(node, ForIn):
        for name in node.binding_names:
            _check_binding(name, node, ctx)
    if isinstance(node, MethodDef):
        _check_binding(node.receiver_name, node, ctx)
    if isinstance(node, Call) and node.receiver is None:
        if node.name in own_names:
            node.name = _mangle(canonical_module, node.name)
        else:
            resolved = _resolve_named(node.name, named_imports, ctx, canonical_module, node.line)
            if resolved is not None:
                node.name = resolved

    if isinstance(node, Node):
        for f in fields(node):
            if f.name in ('resolved_type', 'line', 'col'):
                continue
            value = getattr(node, f.name)
            if f.name in _TYPE_FIELD_NAMES:
                setattr(node, f.name, _rewrite_type_expr(
                    value, own_names, canonical_module, import_aliases, named_imports, ctx))
            elif f.name == 'variants':  # variant names, possibly qualified
                setattr(node, f.name, [
                    _rewrite_type_expr(v, own_names, canonical_module, import_aliases, named_imports, ctx)
                    for v in value
                ])
            else:
                setattr(node, f.name, _rewrite_node(
                    value, own_names, canonical_module, import_aliases, named_imports, ctx))
        return node
    if isinstance(node, list):
        return [
            _rewrite_node(item, own_names, canonical_module, import_aliases, named_imports, ctx) for item in node
        ]
    return node


def _validate_named_imports(
        program: Program, own_names: Set[str], named_imports: Dict[str, Tuple[str, str]], ctx: _MergeContext,
        referencing_module: Optional[str]) -> None:
    """Per-file checks before rewriting: alias vs local-declaration collisions, and unknown or private imported names."""
    for from_decl in program.from_imports:
        for original_name, local_alias in from_decl.names:
            if local_alias in own_names:
                raise MergeError(
                    f"'{local_alias}' at line {from_decl.line} collides with this file's "
                    f"own declaration of that name -- rename the import with 'as', or "
                    f"rename the declaration",
                    line=from_decl.line,
                )
            canonical_name, _ = named_imports[local_alias]
            _resolve_in_module(canonical_name, original_name, ctx, referencing_module, from_decl.line)


# Fixed intrinsic signatures; not user-extensible.
_RECOGNIZED_INTRINSICS: Dict[str, Tuple[object, List[object]]] = {
    '_raw_ptr': (PointerTypeExpr(pointee_type='byte'), ['str']),
    '_raw_len': ('int', ['str']),
    '_from_raw_parts': ('str', [PointerTypeExpr(pointee_type='byte'), 'int']),
}

# Spellings type_from_name treats as equal; needed because this runs before it.
_TYPE_SPELLING_ALIASES = {'byte': 'uint8'}


def _canonical_type_spelling(type_expr):
    """Canonicalize aliased spellings (e.g. `byte` -> `uint8`)."""
    if isinstance(type_expr, PointerTypeExpr):
        return PointerTypeExpr(pointee_type=_canonical_type_spelling(type_expr.pointee_type))
    if isinstance(type_expr, str):
        return _TYPE_SPELLING_ALIASES.get(type_expr, type_expr)
    return type_expr


def _signatures_match(decl: 'IntrinsicDecl', expected: Tuple[object, List[object]]) -> bool:
    """Whether decl's return and param types match `expected`."""
    expected_return, expected_params = expected
    if _canonical_type_spelling(decl.return_type) != _canonical_type_spelling(expected_return):
        return False
    if len(decl.params) != len(expected_params):
        return False
    return all(
        _canonical_type_spelling(p.type) == _canonical_type_spelling(e)
        for p, e in zip(decl.params, expected_params)
    )


def _validate_intrinsics(program: Program) -> None:
    """Reject unknown intrinsics or signature mismatches."""
    for ic in program.intrinsics:
        expected = _RECOGNIZED_INTRINSICS.get(ic.original_name)
        if expected is None:
            raise MergeError(
                f"'{ic.original_name}' at line {ic.line} isn't a recognized intrinsic -- "
                f"the compiler only implements a fixed set of these "
                f"({', '.join(sorted(_RECOGNIZED_INTRINSICS))}), not a general "
                f"extensibility mechanism",
                line=ic.line,
            )
        if not _signatures_match(ic, expected):
            expected_return, expected_params = expected
            params_str = ', '.join(str(p) for p in expected_params)
            raise MergeError(
                f"'{ic.original_name}' at line {ic.line} doesn't match its own required "
                f"signature -- expected ({params_str}) -> {expected_return}",
                line=ic.line,
            )


def merge_programs(entry_program: Program, modules: Dict[str, DiscoveredModule]) -> Program:
    """Merge modules into entry_program in place and return it."""
    current = [entry_program.file]
    try:
        return _merge(entry_program, modules, current)
    except MergeError as e:
        if e.file is None:
            e.file = current[0]
        raise


def _merge(entry_program: Program, modules: Dict[str, DiscoveredModule], current: list) -> Program:
    entry_aliases = getattr(entry_program, 'import_aliases', {})
    entry_named = getattr(entry_program, 'named_imports', {})
    ctx = _MergeContext(
        modules=modules,
        all_own_names={name: _own_top_level_names(module.program) for name, module in modules.items()},
        all_extern_names={name: _own_extern_names(module.program) for name, module in modules.items()},
        all_const_names={name: {cd.name for cd in module.program.consts} for name, module in modules.items()},
        const_map={},
    )

    _validate_intrinsics(entry_program)
    _validate_named_imports(entry_program, _own_top_level_names(entry_program), entry_named, ctx, None)
    entry_program.imports = []
    entry_program.from_imports = []
    ctx.const_map = _const_map(entry_program, None, entry_named, ctx)
    entry_program.consts = _rewrite_node(entry_program.consts, set(), None, entry_aliases, entry_named, ctx)
    entry_program.functions = _rewrite_node(entry_program.functions, set(), None, entry_aliases, entry_named, ctx)
    entry_program.structs = _rewrite_node(entry_program.structs, set(), None, entry_aliases, entry_named, ctx)
    entry_program.type_aliases = _rewrite_node(
        entry_program.type_aliases, set(), None, entry_aliases, entry_named, ctx)
    entry_program.sum_types = _rewrite_node(entry_program.sum_types, set(), None, entry_aliases, entry_named, ctx)
    entry_program.extern_functions = _rewrite_node(
        entry_program.extern_functions, set(), None, entry_aliases, entry_named, ctx)
    entry_program.intrinsics = _rewrite_node(entry_program.intrinsics, set(), None, entry_aliases, entry_named, ctx)

    for canonical_name, module in modules.items():
        own_names = ctx.all_own_names[canonical_name]
        program = module.program
        current[0] = program.file
        _validate_intrinsics(program)
        _validate_named_imports(program, own_names, module.named_imports, ctx, canonical_name)
        ctx.const_map = _const_map(program, canonical_name, module.named_imports, ctx)
        program.consts = _rewrite_node(
            program.consts, own_names, canonical_name, module.import_aliases, module.named_imports, ctx)
        program.functions = _rewrite_node(
            program.functions, own_names, canonical_name, module.import_aliases, module.named_imports, ctx)
        program.structs = _rewrite_node(
            program.structs, own_names, canonical_name, module.import_aliases, module.named_imports, ctx)
        program.type_aliases = _rewrite_node(
            program.type_aliases, own_names, canonical_name, module.import_aliases, module.named_imports, ctx)
        program.sum_types = _rewrite_node(
            program.sum_types, own_names, canonical_name, module.import_aliases, module.named_imports, ctx)
        program.extern_functions = _rewrite_node(
            program.extern_functions, own_names, canonical_name, module.import_aliases, module.named_imports, ctx)
        program.intrinsics = _rewrite_node(
            program.intrinsics, own_names, canonical_name, module.import_aliases, module.named_imports, ctx)

        for fn in program.functions:
            fn.name = _mangle(canonical_name, fn.name)
            entry_program.functions.append(fn)
        for sd in program.structs:
            sd.name = _mangle(canonical_name, sd.name)
            entry_program.structs.append(sd)
        for ta in program.type_aliases:
            ta.name = _mangle(canonical_name, ta.name)
            entry_program.type_aliases.append(ta)
        for st in program.sum_types:
            st.name = _mangle(canonical_name, st.name)
            entry_program.sum_types.append(st)
        for ext in program.extern_functions:
            entry_program.extern_functions.append(ext)
        for cd in program.consts:
            cd.name = _mangle(canonical_name, cd.name)
            entry_program.consts.append(cd)
        for ic in program.intrinsics:
            # Only name is mangled; original_name identifies the intrinsic.
            ic.name = _mangle(canonical_name, ic.name)
            entry_program.intrinsics.append(ic)

    return entry_program
