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
from typing import Dict, List, Optional, Set, Tuple

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
    module-scoped names.

    Includes intrinsics: unlike extern, an intrinsic's own name IS
    mangled -- see IntrinsicDecl's own docstring in parser.py for
    why -- so a bare, in-module reference to one needs the identical
    rewriting any other function/struct/alias/sum-type reference
    already gets."""
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


def _resolve_in_module(canonical_name: str, name: str, ctx: _MergeContext,
                        referencing_module: Optional[str], at_line: int) -> str:
    """Resolves `name`, one of canonical_name's own top-level
    declarations, to what it actually becomes: its own mangled form,
    or, for an extern function, its own unchanged bare name (see
    _own_extern_names' own docstring for why). The shared core both
    _resolve_qualified (given an ALIAS to look up a canonical_name
    from first) and _resolve_named (given a canonical_name directly,
    already resolved by modules.py's own discovery) delegate to, once
    each has settled on its own canonical_name by whichever route.

    Raises MergeError if `name` isn't one of that module's own
    declarations, or is but is hidden (see _check_visible)."""
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


def _resolve_qualified(
        alias: str, name: str, import_aliases: Dict[str, str], ctx: _MergeContext,
        referencing_module: Optional[str], at_line: int) -> Optional[str]:
    """Resolves `alias.name` (as written in a file whose own import
    list is import_aliases, itself either an imported module -- pass
    its own canonical_name as referencing_module -- or the entry file
    -- pass None) to the name it refers to, or returns None when
    `alias` doesn't name an import at all (meaning this ISN'T a
    qualified reference -- an ordinary struct field/method access,
    left for semantic.py to resolve as it already does today)."""
    canonical_name = import_aliases.get(alias)
    if canonical_name is None:
        return None
    return _resolve_in_module(canonical_name, name, ctx, referencing_module, at_line)


def _resolve_named(
        local_name: str, named_imports: Dict[str, Tuple[str, str]], ctx: _MergeContext,
        referencing_module: Optional[str], at_line: int) -> Optional[str]:
    """Resolves a BARE local_name (no receiver, no qualifier -- just
    the name as written) against this file's own named-import table
    (see DiscoveredModule's own docstring in modules.py for how that
    table is built) -- returns None when local_name isn't one of them
    at all, meaning it's something else entirely (this module's own
    declaration, a builtin, a local variable, or genuinely
    undeclared) -- left for own_names' own check, or ultimately
    semantic.py, to resolve or reject, exactly like an unresolved
    qualifier already falls through in _resolve_qualified."""
    target = named_imports.get(local_name)
    if target is None:
        return None
    canonical_name, original_name = target
    return _resolve_in_module(canonical_name, original_name, ctx, referencing_module, at_line)


def _rewrite_type_expr(type_expr, own_names: Set[str], canonical_module: Optional[str],
                        import_aliases: Dict[str, str], named_imports: Dict[str, Tuple[str, str]],
                        ctx: _MergeContext):
    """Rewrites a type expression -- see _TYPE_FIELD_NAMES's own
    docstring for the shapes this covers -- recursively for the three
    nested wrapper kinds, and resolving a QualifiedTypeExpr into its
    own plain, mangled string form directly (a type position never
    needs to STAY a QualifiedTypeExpr past this point, unlike Call/
    Field in expression position, which still need to exist as
    themselves for an ordinary, non-qualified method call or field
    access). A bare string not among own_names is checked against
    named_imports next, before being left alone as a builtin/genuinely
    unresolved name -- the same three-way order _rewrite_node's own
    bare-Call handling uses, for the identical reason: own_names is
    settled by merge_programs' own up-front collision check to never
    overlap with named_imports, so which is checked first can't change
    the outcome, only which error message a genuine collision would
    have produced had that check not already run first."""
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
    isinstance dispatch on its own. A Call with no receiver at all is
    checked against own_names first (a bare, in-module reference to
    one of THIS module's own other top-level declarations) and then,
    if that misses, against named_imports (a bare reference to a
    DIFFERENT module's own declaration, brought in directly by a
    `from ... import` -- see _resolve_named's own docstring) -- the
    two can never both match for the same name, since merge_programs'
    own up-front check already rejects that collision before any
    rewriting starts, so the order between them here is not a
    precedence decision, just which happens to be cheaper to check
    first."""
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
                    (k, _rewrite_node(v, own_names, canonical_module, import_aliases, named_imports, ctx))
                    for k, v in node.kwargs
                ]
            return node
    if isinstance(node, Field) and isinstance(node.base, Variable):
        resolved = _resolve_qualified(node.base.name, node.name, import_aliases, ctx, canonical_module, node.line)
        if resolved is not None:
            return Variable(name=resolved, line=node.line, col=node.col)
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
            elif f.name == 'variants':  # SumTypeDef's own list of variant names, each possibly qualified
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
    """Run once per file, before any rewriting starts, for two things
    a purely lazy, rewrite-time-only check would miss:

    1. A named import's own local alias colliding with one of this
       SAME file's own top-level declarations. _rewrite_node's own
       bare-Call/type dispatch checks own_names before named_imports,
       which would just silently prefer the own declaration every
       time rather than flagging the ambiguity -- rejected here
       instead, matching this language's existing stance against
       shadowing anywhere else (see this feature's own design
       discussion, and modules.py's own identical reasoning for why a
       plain-import qualifier and a named-import alias are checked
       against each other too).

    2. Every named import is validated -- existence in its own source
       module, and visibility -- EAGERLY, here, rather than only the
       moment some reference in this file's own body happens to use
       it. Unlike a qualified reference (`module.name`), which is
       inherently a use site the instant it's written, a named import
       is a standalone declaration that could go entirely unused --
       `from "utils" import nonexistent` would otherwise never be
       caught at all if nothing in this file ever actually calls
       `nonexistent`. This is also why _resolve_named's own call
       inside _rewrite_type_expr's bare-string branch can safely pass
       line 0 for its own error messages: by the time any rewriting
       runs, every named import already resolved successfully once,
       right here, with this function's own, real line number."""
    for from_decl in program.from_imports:
        for original_name, local_alias in from_decl.names:
            if local_alias in own_names:
                raise MergeError(
                    f"'{local_alias}' at line {from_decl.line} collides with this file's "
                    f"own declaration of that name -- rename the import with 'as', or "
                    f"rename the declaration"
                )
            canonical_name, _ = named_imports[local_alias]
            _resolve_in_module(canonical_name, original_name, ctx, referencing_module, from_decl.line)


# The complete, fixed set of intrinsics the compiler recognizes and
# knows how to substitute IR for -- see IntrinsicDecl's own docstring
# in parser.py for the whole mechanism. Never extended by anything a
# user writes: this is a closed set of compiler-implemented
# primitives, not a general extensibility mechanism, so an
# intrinsic declaration with any other original_name is rejected
# outright by _validate_intrinsics below, regardless of which file
# declares it -- there's nothing special about the standard library's
# own c.ht in particular; the check is purely name-and-signature-
# based. Each entry: (return type, [param types]), using the exact
# same string/PointerTypeExpr shapes parse_type itself would produce
# -- see _signatures_match's own docstring for why a plain shape
# comparison here is enough, no type_from_name resolution needed.
_RECOGNIZED_INTRINSICS: Dict[str, Tuple[object, List[object]]] = {
    '_raw_ptr': (PointerTypeExpr(pointee_type='byte'), ['str']),
    '_raw_len': ('int', ['str']),
    '_from_raw_parts': ('str', [PointerTypeExpr(pointee_type='byte'), 'int']),
}

# Every spelling type_from_name itself already treats as equivalent
# (semantic.py's own BUILTIN_TYPE_ALIASES-style table) -- needed here
# too, since this runs before type_from_name ever gets a chance to
# canonicalize anything: `*byte` and `*uint8` have to compare equal
# even though they're textually different strings at this stage.
_TYPE_SPELLING_ALIASES = {'byte': 'uint8'}


def _canonical_type_spelling(type_expr):
    """Normalizes one type-expression shape (a bare string, or a
    PointerTypeExpr wrapping one) through _TYPE_SPELLING_ALIASES, so
    two textually-different but semantically-identical spellings
    (`byte` and `uint8`) compare equal via plain `==` -- see
    _RECOGNIZED_INTRINSICS' own docstring for why this check happens
    here, at the plain-shape level, rather than after semantic.py's
    own, later type resolution."""
    if isinstance(type_expr, PointerTypeExpr):
        return PointerTypeExpr(pointee_type=_canonical_type_spelling(type_expr.pointee_type))
    if isinstance(type_expr, str):
        return _TYPE_SPELLING_ALIASES.get(type_expr, type_expr)
    return type_expr


def _signatures_match(decl: 'IntrinsicDecl', expected: Tuple[object, List[object]]) -> bool:
    """True if decl's own declared signature -- return type, and each
    param's own type in order -- matches expected exactly, once each
    side's own type spelling is canonicalized (see _canonical_type_
    spelling's own docstring). A plain shape/string comparison is
    enough for this: every recognized intrinsic's own signature is
    built entirely from scalar/pointer-to-scalar types, which parse_
    type already represents as a bare string or a PointerTypeExpr
    wrapping one -- neither shape needs a struct/alias registry to
    resolve, unlike a struct-typed signature would."""
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
    """Run once per file (entry or module alike -- nothing about this
    check is specific to the standard library itself), before any
    mangling starts: rejects any intrinsic declaration whose own
    original_name isn't one of _RECOGNIZED_INTRINSICS at all, or is
    but whose declared signature doesn't match that name's own fixed,
    expected one exactly. Checked here, against original_name, rather
    than left for semantic.py to catch later against the (by then
    already-mangled) name -- see IntrinsicDecl's own docstring for why
    original_name has to be the one thing mangling never touches."""
    for ic in program.intrinsics:
        expected = _RECOGNIZED_INTRINSICS.get(ic.original_name)
        if expected is None:
            raise MergeError(
                f"'{ic.original_name}' at line {ic.line} isn't a recognized intrinsic -- "
                f"the compiler only implements a fixed set of these "
                f"({', '.join(sorted(_RECOGNIZED_INTRINSICS))}), not a general "
                f"extensibility mechanism"
            )
        if not _signatures_match(ic, expected):
            expected_return, expected_params = expected
            params_str = ', '.join(str(p) for p in expected_params)
            raise MergeError(
                f"'{ic.original_name}' at line {ic.line} doesn't match its own required "
                f"signature -- expected ({params_str}) -> {expected_return}"
            )


def merge_programs(entry_program: Program, modules: Dict[str, DiscoveredModule]) -> Program:
    """Folds every module's own declarations into entry_program in
    place, mangled and with every qualified reference resolved (see
    this module's own docstring), and returns entry_program itself.

    The entry file's own declarations are never renamed at all --
    nothing can ever import the entry file, so they need no
    globally-unique, mangled form; they stay exactly as parsed. Its
    own BODY is still rewritten, though, exactly like every module's
    own, for any qualified or named references it makes into an
    imported module.

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
    across modules is left as a known, narrow gap for now.

    Intrinsic declarations, by contrast, ARE mangled, exactly like an
    ordinary function -- see IntrinsicDecl's own docstring in parser.
    py -- and are validated (see _validate_intrinsics) before that
    mangling happens, in every file alike, entry included: nothing
    about this check is specific to the standard library's own
    modules."""
    entry_aliases = getattr(entry_program, 'import_aliases', {})
    entry_named = getattr(entry_program, 'named_imports', {})
    ctx = _MergeContext(
        modules=modules,
        all_own_names={name: _own_top_level_names(module.program) for name, module in modules.items()},
        all_extern_names={name: _own_extern_names(module.program) for name, module in modules.items()},
    )

    _validate_intrinsics(entry_program)
    _validate_named_imports(entry_program, _own_top_level_names(entry_program), entry_named, ctx, None)
    entry_program.imports = []  # consumed above/below; nothing downstream needs the raw ImportDecls again
    entry_program.from_imports = []  # same
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
        _validate_intrinsics(program)
        _validate_named_imports(program, own_names, module.named_imports, ctx, canonical_name)
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
            # Never mangled -- see _own_extern_names' own docstring.
            entry_program.extern_functions.append(ext)
        for ic in program.intrinsics:
            # original_name is deliberately left untouched -- only
            # name is mangled -- see IntrinsicDecl's own docstring for
            # why the two have to stay independent.
            ic.name = _mangle(canonical_name, ic.name)
            entry_program.intrinsics.append(ic)

    return entry_program
