"""Program merging.

Folds every module discover_modules() found into the entry file's own
Program, so semantic.py sees exactly what it already sees today for a
single file: one flat, unqualified Program, with no module concept of
its own at all. This is the MERGE model this feature's own design
discussion settled on -- see modules.py's own module docstring for
why it's what makes a circular import safe: nothing downstream of
this module needs modules folded in in any particular order, since
semantic.py's own analyze() already collects every signature across
the WHOLE (now merged) Program before checking any body, the same
two-phase process that already makes forward references and mutual
recursion within one file work today.

The one thing every discovered module's own declarations need before
they're safe to fold in: a globally-unique name, so a function in
module "utils" can never collide with a same-named one in module
"other", or with anything in the entry file itself. MANGLE below does
this the same way desugar.py's own mangle_method_name already does
for struct methods (StructName.methodName) -- except '.' is already
spoken for by that convention, so a collision between the two would
be a real, silent naming clash (a struct named "utils" with a method
"bar", and a module "utils" exporting a function "bar", would BOTH
mangle to "utils.bar") -- '$' is used here instead, a character that,
like '.', can never appear in an ordinary Hornet IDENTIFIER (see
lexer.py's own IDENTIFIER regex), so it's equally collision-free
against anything a user could type directly, while staying distinct
from '.' itself.

Every qualified reference (`module.name`, in expression position via
Call/Field's own existing receiver/base shape, or in type position via
QualifiedTypeExpr) is resolved and rewritten here too, directly into
its own mangled, plain form -- along with visibility (see VISIBLE_TO)
-- rather than leaving either concern for semantic.py to handle later.
That's what keeps semantic.py itself completely unaware that modules
exist at all: by the time it ever sees this Program, every name in it
is already a single, flat, globally-unique, ordinary identifier,
exactly like today.

Runs BEFORE desugar_methods/analyze -- both would otherwise
misinterpret an un-rewritten qualified Call/Field (Call.receiver or
Field.base set to a bare Variable naming an imported module) as an
ordinary method call or struct field access.
"""

from dataclasses import dataclass, fields
from typing import Dict, Optional, Set

from modules import DiscoveredModule
from parser import (
    ArrayTypeExpr,
    Call,
    Field,
    Node,
    PointerTypeExpr,
    Program,
    QualifiedTypeExpr,
    SliceTypeExpr,
    Variable,
)


class MergeError(Exception):
    """Raised when a qualified reference (`module.name`) can't be
    resolved: `module` doesn't name any import visible from the
    referencing file, or `name` isn't one of `module`'s own top-level
    declarations, or it is but is hidden from this referencing module
    (see VISIBLE_TO)."""


# Field names, across every Node subclass that has one, known to hold
# a type expression (a bare struct/alias name string, an ArrayTypeExpr/
# SliceTypeExpr/PointerTypeExpr, or now a QualifiedTypeExpr) rather
# than an ordinary child expression -- handled by _rewrite_type_expr
# instead of the generic per-field recursion in _rewrite_node, since a
# bare string here means something entirely different than a bare
# string anywhere else in the tree (a NAME to resolve as a type, not
# arbitrary data -- unlike, say, a StringLiteral's own `value`, or a
# Param's own `name`, which must NEVER be rewritten even when they
# happen to match some module's own declaration name).
_TYPE_FIELD_NAMES = frozenset({'var_type', 'return_type', 'field_type', 'target_type', 'type', 'type_name'})


def _mangle(canonical_module: str, name: str) -> str:
    """A module-qualified declaration's own globally-unique name, e.g.
    canonical_module="utils", name="Circle" -> "utils$Circle". See
    this module's own docstring for why '$', not '.' (already spoken
    for by mangle_method_name in desugar.py)."""
    return f"{canonical_module}${name}"


def _own_top_level_names(program: Program) -> Set[str]:
    """Every MANGLE-ELIGIBLE name program itself declares at the top
    level -- what a BARE (unqualified) reference within program's own
    body should be rewritten to the mangled form for, since it's this
    module's own declaration under a different, pre-merge name.
    Deliberately excludes extern function names -- see _own_extern_
    names' own docstring for why those are never mangled at all.
    Struct field/method names and local variable/parameter names are
    also deliberately NOT included here -- only genuinely top-level,
    module-scoped names."""
    names: Set[str] = set()
    for fn in program.functions:
        names.add(fn.name)
    for sd in program.structs:
        names.add(sd.name)
    for ta in program.type_aliases:
        names.add(ta.name)
    for st in program.sum_types:
        names.add(st.name)
    return names


def _own_extern_names(program: Program) -> Set[str]:
    """Every extern function name program itself declares -- kept
    entirely separate from _own_top_level_names, and NEVER mangled,
    unlike every other kind of top-level declaration: an extern
    function's own `name` IS the real, external C symbol the linker
    resolves against (see ExternFunctionDecl's own docstring in
    parser.py) -- mangling it to "module$name" would still let a
    qualified reference (`module.name`) resolve internally, but the
    generated assembly would then try to call an external symbol that
    doesn't exist at all, since the real C function is still named
    exactly `name`, never the mangled form. A bare, in-module
    reference to one of these stays unmangled for the identical
    reason (see _rewrite_node's own Call handling, which only mangles
    names found in own_names, not these)."""
    return {ext.name for ext in program.extern_functions}


def _check_visible(module: DiscoveredModule, name: str, referencing_module: Optional[str], at_line: int) -> None:
    """Raises MergeError if `name`, one of module's own top-level
    declarations, is hidden (a leading '_') from referencing_module --
    None for the entry file itself, which can never BE the same as
    any imported module's own canonical_name, so a leading '_' is
    always hidden from it. See this feature's own design discussion:
    visibility is a top-level-declaration-only rule for v1, struct
    field/method visibility deferred."""
    if name.startswith('_') and referencing_module != module.canonical_name:
        raise MergeError(
            f"'{module.canonical_name}.{name}' at line {at_line} is not visible outside "
            f"the module that defines it -- names starting with '_' are private to their "
            f"own module"
        )


@dataclass
class _MergeContext:
    """Everything a single rewrite pass needs about the WHOLE
    discovered set, bundled once rather than threaded as three
    separate parameters through every function below: modules (for
    looking up a qualifier's own target), and all_own_names/all_
    extern_names (every module's own declaration names, computed once
    up front -- see _resolve_qualified's own docstring for why these
    can't be recomputed fresh from a module's own, possibly-already-
    mutated Program partway through the merge loop)."""
    modules: Dict[str, DiscoveredModule]
    all_own_names: Dict[str, Set[str]]
    all_extern_names: Dict[str, Set[str]]


def _resolve_qualified(
        alias: str, name: str, import_aliases: Dict[str, str], ctx: _MergeContext,
        referencing_module: Optional[str], at_line: int) -> Optional[str]:
    """Resolves `alias.name` (as written in a file whose own import
    list is import_aliases, itself either an imported module -- pass
    its own canonical_name as referencing_module -- or the entry file
    -- pass None) to the name it refers to, or returns None when
    `alias` doesn't name an import at all (meaning this ISN'T a
    qualified reference -- an ordinary struct field/method access,
    left for semantic.py to resolve as it already does today).

    An ordinary declaration resolves to its own mangled form; an
    extern function resolves to its own BARE name instead, unchanged
    -- see _own_extern_names' own docstring for why.

    Raises MergeError if `alias` DOES name an import but `name` isn't
    one of that module's own declarations, or is but is hidden (see
    _check_visible)."""
    canonical_name = import_aliases.get(alias)
    if canonical_name is None:
        return None
    target = ctx.modules[canonical_name]
    if name in ctx.all_extern_names[canonical_name]:
        return name  # extern functions are never mangled -- see _own_extern_names
    if name not in ctx.all_own_names[canonical_name]:
        raise MergeError(
            f"'{name}' at line {at_line} is not declared in module "
            f"{target.canonical_name!r} ({target.file_path})"
        )
    _check_visible(target, name, referencing_module, at_line)
    return _mangle(canonical_name, name)


def _rewrite_type_expr(type_expr, own_names: Set[str], canonical_module: Optional[str],
                        import_aliases: Dict[str, str], ctx: _MergeContext):
    """Rewrites a type expression -- see _TYPE_FIELD_NAMES's own
    docstring for the shapes this covers -- recursively for the three
    nested wrapper kinds, and resolving a QualifiedTypeExpr into its
    own plain, mangled string form directly (a type position never
    needs to STAY a QualifiedTypeExpr past this point, unlike Call/
    Field in expression position, which still need to exist as
    themselves for an ordinary, non-qualified method call or field
    access)."""
    if isinstance(type_expr, QualifiedTypeExpr):
        resolved = _resolve_qualified(type_expr.module, type_expr.name, import_aliases, ctx, canonical_module,
                                       type_expr.line)
        if resolved is None:
            raise MergeError(
                f"'{type_expr.module}' at line {type_expr.line} doesn't name an imported "
                f"module"
            )
        return resolved
    if isinstance(type_expr, (ArrayTypeExpr, SliceTypeExpr, PointerTypeExpr)):
        field_name = 'pointee_type' if isinstance(type_expr, PointerTypeExpr) else 'element_type'
        setattr(type_expr, field_name,
                _rewrite_type_expr(getattr(type_expr, field_name), own_names, canonical_module, import_aliases, ctx))
        return type_expr
    if isinstance(type_expr, str) and type_expr in own_names:
        return _mangle(canonical_module, type_expr)
    return type_expr


def _rewrite_node(node, own_names: Set[str], canonical_module: Optional[str], import_aliases: Dict[str, str],
                   ctx: _MergeContext):
    """Rewrites node (a Node, a list, or a scalar) and every one of
    its own descendants in place, mutating and returning the same
    object except where the rewritten VALUE has to be a different kind
    entirely (a QualifiedTypeExpr resolving to a plain string; a
    qualified Call/Field resolving to a plain, receiver-less Variable/
    Call) -- those cases return the replacement, which the caller (a
    recursive call, or one of merge_programs' own top-level loops)
    assigns back into the field/list slot that held the original.

    Call and Field get their own handling before the generic per-field
    recursion below, since a qualified reference (`module.name`) is
    parsed as the SAME shape as an ordinary method call/field access
    (see Field's own docstring in parser.py) -- telling them apart
    needs this function's own module-aware context (import_aliases),
    which the generic recursion has no way to thread through a bare
    isinstance dispatch on its own. A Call with no receiver at all
    still needs checking against own_names (a bare, in-module
    reference to one of THIS module's own other top-level
    declarations), the identical reasoning _TYPE_FIELD_NAMES's own
    docstring gives for why a type name can't just be recursed into
    generically either."""
    if isinstance(node, Call) and node.receiver is not None and isinstance(node.receiver, Variable):
        resolved = _resolve_qualified(node.receiver.name, node.name, import_aliases, ctx, canonical_module, node.line)
        if resolved is not None:
            node.name = resolved
            node.receiver = None
            node.args = [_rewrite_node(a, own_names, canonical_module, import_aliases, ctx) for a in node.args]
            if node.kwargs is not None:
                # A module-qualified struct construction using named
                # fields (`module.Circle(radius=5)`) -- see parse_
                # receiver_call_args' own docstring in parser.py for
                # why a receiver-based Call can carry kwargs at all
                # (the grammar accepts them generically, for every
                # receiver-based call, since there's no symbol table
                # at parse time to tell this apart from an ordinary
                # method call -- which never actually reaches this
                # branch, since node.receiver would already have
                # resolved to None above only for a genuine qualified
                # reference; an ordinary method call's own kwargs, if
                # any, are instead rejected downstream by semantic.
                # py's own _check_method_call).
                node.kwargs = [
                    (k, _rewrite_node(v, own_names, canonical_module, import_aliases, ctx)) for k, v in node.kwargs
                ]
            return node
    if isinstance(node, Field) and isinstance(node.base, Variable):
        resolved = _resolve_qualified(node.base.name, node.name, import_aliases, ctx, canonical_module, node.line)
        if resolved is not None:
            return Variable(name=resolved, line=node.line, col=node.col)
    if isinstance(node, Call) and node.receiver is None and node.name in own_names:
        node.name = _mangle(canonical_module, node.name)

    if isinstance(node, Node):
        for f in fields(node):
            if f.name in ('resolved_type', 'line', 'col'):
                continue
            value = getattr(node, f.name)
            if f.name in _TYPE_FIELD_NAMES:
                setattr(node, f.name, _rewrite_type_expr(value, own_names, canonical_module, import_aliases, ctx))
            elif f.name == 'variants':  # SumTypeDef's own list of variant names, each possibly qualified
                setattr(node, f.name, [
                    _rewrite_type_expr(v, own_names, canonical_module, import_aliases, ctx) for v in value
                ])
            else:
                setattr(node, f.name, _rewrite_node(value, own_names, canonical_module, import_aliases, ctx))
        return node
    if isinstance(node, list):
        return [_rewrite_node(item, own_names, canonical_module, import_aliases, ctx) for item in node]
    return node


def merge_programs(entry_program: Program, modules: Dict[str, DiscoveredModule]) -> Program:
    """Folds every module's own declarations into entry_program in
    place, mangled and with every qualified reference resolved (see
    this module's own docstring), and returns entry_program itself.

    The entry file's own declarations are never renamed at all --
    nothing can ever import the entry file, so they need no
    globally-unique, mangled form; they stay exactly as parsed. Its
    own BODY is still rewritten, though, exactly like every module's
    own, for any qualified references it makes into an imported
    module.

    Every module's own top-level names are collected FIRST, before
    any rewriting or renaming starts (see _resolve_qualified's own
    docstring for why this ordering matters): two modules can freely
    reference each other (a genuine import cycle), and each has to
    see the OTHER's own original names, not whatever it's already
    been mangled to by the time this loop reaches it.

    Extern function declarations are folded in WITHOUT renaming --
    see _own_extern_names' own docstring for why their own name can
    never be mangled -- so two different modules independently
    declaring the identical extern (the same real C symbol) will
    collide as a duplicate declaration once merged; semantic.py's own,
    pre-existing "already declared" check catches that loudly rather
    than silently, and deduplicating identical extern declarations
    across modules is left as a known, narrow gap for now."""
    entry_program.imports = []  # consumed here; nothing downstream needs the raw ImportDecls again
    entry_aliases = getattr(entry_program, 'import_aliases', {})
    ctx = _MergeContext(
        modules=modules,
        all_own_names={name: _own_top_level_names(module.program) for name, module in modules.items()},
        all_extern_names={name: _own_extern_names(module.program) for name, module in modules.items()},
    )

    entry_program.functions = _rewrite_node(entry_program.functions, set(), None, entry_aliases, ctx)
    entry_program.structs = _rewrite_node(entry_program.structs, set(), None, entry_aliases, ctx)
    entry_program.type_aliases = _rewrite_node(entry_program.type_aliases, set(), None, entry_aliases, ctx)
    entry_program.sum_types = _rewrite_node(entry_program.sum_types, set(), None, entry_aliases, ctx)
    entry_program.extern_functions = _rewrite_node(entry_program.extern_functions, set(), None, entry_aliases, ctx)

    for canonical_name, module in modules.items():
        own_names = ctx.all_own_names[canonical_name]
        program = module.program
        program.functions = _rewrite_node(program.functions, own_names, canonical_name, module.import_aliases, ctx)
        program.structs = _rewrite_node(program.structs, own_names, canonical_name, module.import_aliases, ctx)
        program.type_aliases = _rewrite_node(
            program.type_aliases, own_names, canonical_name, module.import_aliases, ctx)
        program.sum_types = _rewrite_node(program.sum_types, own_names, canonical_name, module.import_aliases, ctx)
        program.extern_functions = _rewrite_node(
            program.extern_functions, own_names, canonical_name, module.import_aliases, ctx)

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
            # Never mangled -- see _own_extern_names' own docstring.
            entry_program.extern_functions.append(ext)

    return entry_program
