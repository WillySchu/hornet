"""Module scopes: what each name means in each file, across modules, without changing any tree.

Every top-level declaration gets a program-wide unique key: its own name in the entry file, and
`module$name` in an imported module ('$' can't appear in identifiers; '.' is taken by methods).
Externs keep their names, which are linker symbols. A file's scope maps the bare names it can use
(its own declarations and `from ... import` names) to keys; `alias.name` references are resolved
here too, recorded by node number, along with the checks on imports and on names that locals may
not reuse. semantic.py checks each declaration and function in its file's scope.
"""

import dataclasses
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from diagnostics import CompileError
from modules import DiscoveredModule
from parser import (
    Call,
    Field,
    ForIn,
    MethodDef,
    Node,
    Param,
    PointerTypeExpr,
    Program,
    QualifiedTypeExpr,
    VarDecl,
    Variable,
)


# The builtin functions (semantic.py's check_call). No declaration or named import may take one of
# these names, in any file: it would stand in for the builtin there.
BUILTIN_FUNCTION_NAMES = {'print', 'len', 'append', 'del', 'bytes', 'panic'}


class MergeError(CompileError):
    """An unresolvable or private qualified reference, or a name a local may not reuse."""

    def __init__(self, message: str, file: Optional[str] = None, line: int = 0, col: int = 0):
        clean = re.sub(rf' at line {line}\b', '', message) if line else message
        super().__init__(clean, file, line, col, legacy=message)


def mangle(module: str, name: str) -> str:
    return f"{module}${name}"


def display_name(key: str) -> str:
    """The name as declared: a key without its module prefix."""
    return key.rsplit('$', 1)[-1]


@dataclass
class Scope:
    """One file's names. `module` is the canonical module name, None for the entry file."""
    module: Optional[str]
    file: Optional[str]
    own: Dict[str, str]  # this file's top-level declarations: name -> key
    externs: Set[str]  # this file's extern declarations
    consts: Dict[str, str]  # bare constant names usable here -> key
    aliases: Dict[str, str]  # `import ... as alias` -> canonical module
    named: Dict[str, Tuple[str, str]]  # `from ... import name as local` -> (canonical module, name)
    resolved: Dict[str, str] = field(default_factory=dict)  # named imports, resolved: local -> key

    def resolve(self, name: str) -> Optional[str]:
        """The key a bare name refers to at top level here, or None (a builtin, local, or unknown)."""
        if name in self.own:
            return self.own[name]
        if name in self.externs:
            return name
        return self.resolved.get(name)


@dataclass
class ModuleSet:
    """The program as analysis sees it: the files' declarations, renamed to their keys (new nodes;
    bodies and type expressions are shared, unchanged), each with the scope of its file."""
    functions: List[Node] = field(default_factory=list)
    structs: List[Node] = field(default_factory=list)
    type_aliases: List[Node] = field(default_factory=list)
    sum_types: List[Node] = field(default_factory=list)
    extern_functions: List[Node] = field(default_factory=list)
    intrinsics: List[Node] = field(default_factory=list)
    consts: List[Node] = field(default_factory=list)
    files: List[Tuple[Program, Scope]] = field(default_factory=list)  # entry first
    scope_of: Dict[int, Scope] = field(default_factory=dict)  # declaration copy's nid -> its file's scope
    qualified: Dict[int, str] = field(default_factory=dict)  # `alias.name` node's nid -> key


_DECLARATION_KINDS = ('consts', 'functions', 'structs', 'type_aliases', 'sum_types', 'extern_functions', 'intrinsics')


def _own_names(program: Program) -> Set[str]:
    return {d.name for kind in _DECLARATION_KINDS if kind != 'extern_functions' for d in getattr(program, kind)}


def build_module_set(entry: Program, modules: Dict[str, DiscoveredModule]) -> ModuleSet:
    """Scopes for every file, with all import checks, and the renamed declarations."""
    files = [(entry, None, getattr(entry, 'import_aliases', {}), getattr(entry, 'named_imports', {}))]
    files += [(m.program, name, m.import_aliases, m.named_imports) for name, m in modules.items()]
    own = {module: _own_names(program) for program, module, _, _ in files}
    externs = {module: {ext.name for ext in program.extern_functions} for program, module, _, _ in files}
    const_names = {module: {cd.name for cd in program.consts} for program, module, _, _ in files}
    result = ModuleSet()
    current = [entry.file]

    def resolve_in(module: str, name: str, referencing: Optional[str], at: Node) -> str:
        """The key of `name` in `module`, referred to from module `referencing` at node `at`."""
        target = modules[module]
        if name in externs[module]:
            return name
        if name not in own[module]:
            raise MergeError(f"'{name}' at line {at.line} is not declared in module "
                             f"{target.canonical_name!r} ({target.file_path})", line=at.line, col=at.col)
        if name.startswith('_') and referencing != module:
            raise MergeError(f"'{module}.{name}' at line {at.line} is not visible outside the module that "
                             f"defines it -- names starting with '_' are private to their own module",
                             line=at.line, col=at.col)
        return mangle(module, name)

    try:
        for program, module, aliases, named in files:
            current[0] = program.file
            key = (lambda n: n) if module is None else (lambda n, m=module: mangle(m, n))
            consts = {cd.name: key(cd.name) for cd in program.consts}
            for local, (target, original) in named.items():
                if original in const_names[target]:
                    consts[local] = mangle(target, original)
            scope = Scope(module, program.file, {n: key(n) for n in own[module]}, externs[module], consts,
                          aliases, named)
            _validate_intrinsics(program)
            for from_decl in program.from_imports:
                for original, local in from_decl.names:
                    if local in BUILTIN_FUNCTION_NAMES:
                        raise MergeError(
                            f"'{local}' at line {from_decl.line} is a builtin and can't name an import -- choose "
                            f"another name after 'as'", line=from_decl.line, col=from_decl.col)
                    if local in own[module]:
                        raise MergeError(
                            f"'{local}' at line {from_decl.line} collides with this file's own declaration of that "
                            f"name -- rename the import with 'as', or rename the declaration",
                            line=from_decl.line, col=from_decl.col)
                    scope.resolved[local] = resolve_in(named[local][0], original, module, from_decl)
            for kind in _DECLARATION_KINDS:
                for decl in getattr(program, kind):
                    _walk(decl, scope, result.qualified, lambda a, n, at, s=scope: resolve_in(
                        s.aliases[a], n, s.module, at) if a in s.aliases else None)
            result.files.append((program, scope))
            for kind in _DECLARATION_KINDS:
                for decl in getattr(program, kind):
                    copy = decl if kind == 'extern_functions' else dataclasses.replace(decl, name=key(decl.name))
                    getattr(result, kind).append(copy)
                    result.scope_of[copy.nid] = scope
    except MergeError as e:
        if e.file is None:
            e.file = current[0]
        raise
    return result


def _walk(node, scope: Scope, qualified: Dict[int, str], resolve_qualified) -> None:
    """Resolve every `alias.name` under `node`, and check the names locals declare."""
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, (list, tuple)):
            stack.extend(reversed(n))
            continue
        if not isinstance(n, Node):
            continue
        if isinstance(n, QualifiedTypeExpr):
            key = resolve_qualified(n.module, n.name, n)
            if key is None:
                raise MergeError(f"'{n.module}' at line {n.line} doesn't name an imported module",
                                 line=n.line, col=n.col)
            qualified[n.nid] = key
            continue
        if isinstance(n, Call) and isinstance(n.receiver, Variable):
            key = resolve_qualified(n.receiver.name, n.name, n)
            if key is not None:
                qualified[n.nid] = key
                stack.extend(reversed([n.args, n.kwargs or []]))
                continue
        if isinstance(n, Field) and isinstance(n.base, Variable):
            key = resolve_qualified(n.base.name, n.name, n)
            if key is not None:
                qualified[n.nid] = key
                continue
        for name in _declared_names(n):
            if name in scope.consts:
                raise MergeError(f"'{name}' at line {n.line} is a constant here and can't also be a variable name",
                                 line=n.line, col=n.col)
            if name in scope.aliases:
                raise MergeError(f"'{name}' at line {n.line} is an imported module's name here and can't also be "
                                 f"a variable name", line=n.line, col=n.col)
        stack.extend(reversed([getattr(n, f.name) for f in dataclasses.fields(n)
                               if f.name not in ('line', 'col', 'file', 'nid')]))


def _declared_names(n: Node) -> List[str]:
    if isinstance(n, (VarDecl, Param)):
        return [n.name]
    if isinstance(n, ForIn):
        return list(n.binding_names)
    if isinstance(n, MethodDef):
        return [n.receiver_name]
    return []


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


def _signatures_match(decl, expected: Tuple[object, List[object]]) -> bool:
    expected_return, expected_params = expected
    if _canonical_type_spelling(decl.return_type) != _canonical_type_spelling(expected_return):
        return False
    if len(decl.params) != len(expected_params):
        return False
    return all(_canonical_type_spelling(p.type) == _canonical_type_spelling(e) for p, e in zip(decl.params, expected_params))


def _validate_intrinsics(program: Program) -> None:
    """Reject unknown intrinsics or signature mismatches."""
    for ic in program.intrinsics:
        expected = _RECOGNIZED_INTRINSICS.get(ic.original_name)
        if expected is None:
            raise MergeError(
                f"'{ic.original_name}' at line {ic.line} isn't a recognized intrinsic -- the compiler only "
                f"implements a fixed set of these ({', '.join(sorted(_RECOGNIZED_INTRINSICS))}), not a general "
                f"extensibility mechanism", line=ic.line, col=ic.col)
        if not _signatures_match(ic, expected):
            expected_return, expected_params = expected
            params_str = ', '.join(str(p) for p in expected_params)
            raise MergeError(f"'{ic.original_name}' at line {ic.line} doesn't match its own required signature -- "
                             f"expected ({params_str}) -> {expected_return}", line=ic.line, col=ic.col)
