"""Module discovery.

Resolves and parses every file an entry .ht file transitively imports,
before any merging or semantic analysis happens (see merge.py's own
module docstring for what consumes this). A "module" is exactly one
file for v1 (see ImportDecl's own docstring in parser.py) -- discovery
never groups files together, only discovers and parses each one once.

No dependency ordering happens here, deliberately: this feature's own
design discussion settled on the MERGE model (see merge.py), where
every discovered module's own declarations get folded into one
unified Program before semantic analysis ever runs, the same two-
phase (every signature collected, THEN every body checked) process
that already makes forward references and mutual recursion within one
file work today. That's what makes a circular import (A imports B, B
imports A) safe to allow at all -- nothing downstream of this module
needs modules visited in any particular order, so discovery itself
only has to make sure each file is parsed exactly once, regardless of
how many places import it or whether a cycle exists.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

from lexer import lex
from parser import Parser, Program


class ModuleError(Exception):
    """Raised for any import-resolution failure: a path that doesn't
    resolve to a real file, two different files sharing one canonical
    module name, or two imports in one file colliding on their own
    local alias."""


@dataclass
class DiscoveredModule:
    """One transitively-imported file, fully parsed but not yet
    merged or analyzed.

    canonical_name is this module's own, globally-unique identity --
    its own file's stem (path's last component, '.ht' stripped, e.g.
    "sub/dir/utils.ht" -> "utils") -- used both to qualify every one
    of its own top-level declarations when merging (see merge.py) and
    as the mangling key every OTHER module's own qualified references
    resolve against. Distinct from whatever LOCAL alias a particular
    importer happens to use for it (see import_aliases on the
    IMPORTING module, not this one): `import "utils" as u` still
    merges utils.ht's own declarations under the canonical name
    "utils", with "u" only ever meaningful within the one file that
    wrote that alias.

    import_aliases maps each name THIS module's own `import`
    statements bring into scope (the alias actually written in ITS
    OWN source, or the default-derived one) to the OTHER module's own
    canonical_name -- what a qualified reference within this module's
    own body (`alias.name`) needs resolved against.

    named_imports maps each BARE local alias THIS module's own `from
    ... import ...` statements bring into scope to (source module's
    own canonical_name, original name in that module) -- what a bare,
    unqualified reference within this module's own body needs
    resolved against instead of import_aliases' own qualifier lookup.
    Kept as its own, separate table rather than folded into import_
    aliases: the two answer different questions (`is this name a
    MODULE qualifier` vs. `is this name a DIRECTLY-imported
    declaration`), and merge.py's own bare-reference dispatch needs to
    ask them separately, in that order."""
    canonical_name: str
    file_path: Path
    program: Program
    import_aliases: Dict[str, str] = field(default_factory=dict)
    named_imports: Dict[str, Tuple[str, str]] = field(default_factory=dict)


_STDLIB_ROOT = Path(__file__).parent / 'stdlib'


def _candidate_path(base_dir: Path, path: str) -> Path:
    """Turns an ImportDecl's own `path` into a canonical, absolute
    filesystem Path under base_dir, appending '.ht' when `path`
    doesn't already end with it (matching Python's own `import foo`
    needing no '.py'). Doesn't check the result actually exists --
    that's _resolve_import_path's own job, trying this against more
    than one base_dir in turn."""
    candidate = path if path.endswith('.ht') else f'{path}.ht'
    return (base_dir / candidate).resolve()


def _resolve_import_path(importer_dir: Path, path: str) -> Optional[Path]:
    """Resolves an ImportDecl's own `path`, trying the importing
    file's own directory FIRST, falling back to Hornet's own standard
    library location (_STDLIB_ROOT, a fixed directory bundled with the
    compiler itself -- see this feature's own design discussion for
    why fallback rather than a stricter scheme requiring a marker for
    one or the other) only if nothing local matches. Relative
    resolution takes unconditional priority: a same-named local file
    is always what a bare import reaches, never silently shadowed by a
    stdlib module of the same name -- the stdlib is only ever
    consulted for a name nothing local answers to at all.

    Returns None if NEITHER location has a matching file -- left for
    the caller to turn into its own ModuleError, with whatever message
    fits its own context."""
    local = _candidate_path(importer_dir, path)
    if local.is_file():
        return local
    stdlib_candidate = _candidate_path(_STDLIB_ROOT, path)
    if stdlib_candidate.is_file():
        return stdlib_candidate
    return None


def discover_modules(entry_path: str) -> Tuple[Program, Dict[str, DiscoveredModule]]:
    """Parses entry_path and every file it transitively imports.

    Returns (entry_program, modules): entry_program is the entry
    file's own, unqualified Program -- nothing ever imports the entry
    file itself, so its own declarations never need a canonical name
    or qualifying at all (see merge.py). modules maps every
    TRANSITIVELY-imported file's own canonical_name to its own
    DiscoveredModule, deduplicated by resolved absolute path -- a
    diamond import or a genuine cycle is parsed exactly once,
    regardless of how many places reach it.

    Raises ModuleError for: a path that doesn't resolve to a real
    file; two DIFFERENT files (by resolved path) producing the same
    canonical_name (this module's own identity, used to qualify its
    declarations, has to be globally unique across the whole
    discovered set -- an `as` rename at one import site only fixes
    the local-alias collision case below, not this one, since it
    doesn't change what the other module's own declarations get
    mangled as; the fix there is renaming one of the actual files);
    or two of THIS file's own imports (plain or named, in any
    combination) introducing the identical bare local name -- `import
    "utils"` and `from "other" import foo as utils` colliding, say.
    Plain-import qualifiers and named-import aliases are deliberately
    checked against EACH OTHER here, as one shared namespace, not two
    independent ones: nothing stops a qualifier from being used
    immediately before `.`, and a named import from being used
    anywhere else, so the two COULD coexist without a real parsing
    ambiguity -- but allowing it buys nothing and reads as confusing
    at the use site (does `foo` on its own mean the named import, or
    was `foo.` about to follow?), so it's rejected outright instead,
    matching this language's existing, consistent stance against
    shadowing anywhere else (see semantic.py's own "already declared"
    checks for functions, structs, sum types, and type aliases)."""
    entry_file = Path(entry_path).resolve()
    entry_program = Parser(lex(str(entry_file))).parse_program()

    modules: Dict[str, DiscoveredModule] = {}  # canonical_name -> DiscoveredModule, populated once a module's own _visit call returns
    by_path: Dict[Path, str] = {}  # resolved path -> canonical_name; doubles as the "already discovered (or in progress)" set

    def _visit(program: Program, importer_dir: Path) -> Tuple[Dict[str, str], Dict[str, Tuple[str, str]]]:
        """Resolves every one of program's own plain and named
        imports, recursively visiting each not-yet-discovered target.
        Returns (this program's own alias -> canonical_name mapping,
        its own local_alias -> (canonical_name, original_name)
        mapping)."""
        aliases: Dict[str, str] = {}
        named: Dict[str, Tuple[str, str]] = {}

        def _resolve_and_discover(path: str, at_line: int) -> str:
            """Shared by both loops below: resolves `path`, visiting
            and registering it in `modules` if not already discovered
            (or in progress, as an ancestor of this very call -- a
            genuine cycle), and returns its own canonical_name."""
            resolved = _resolve_import_path(importer_dir, path)
            if resolved is None:
                raise ModuleError(
                    f"Import {path!r} at line {at_line} doesn't resolve to a real "
                    f"file (looked for {_candidate_path(importer_dir, path)}, or in "
                    f"the standard library)"
                )
            if resolved in by_path:
                return by_path[resolved]
            canonical_name = resolved.stem
            if canonical_name in modules or canonical_name in by_path.values():
                raise ModuleError(
                    f"Two different files both resolve to module name "
                    f"{canonical_name!r} -- module names must be globally "
                    f"unique across every imported file; rename one of them "
                    f"(the second is {resolved})"
                )
            by_path[resolved] = canonical_name  # reserved BEFORE recursing, so a cycle terminates here next time
            sub_program = Parser(lex(str(resolved))).parse_program()
            sub_aliases, sub_named = _visit(sub_program, resolved.parent)
            modules[canonical_name] = DiscoveredModule(
                canonical_name=canonical_name, file_path=resolved, program=sub_program,
                import_aliases=sub_aliases, named_imports=sub_named,
            )
            return canonical_name

        for decl in program.imports:
            canonical_name = _resolve_and_discover(decl.path, decl.line)
            if decl.qualifier in aliases and aliases[decl.qualifier] != canonical_name:
                raise ModuleError(
                    f"Two imports at line {decl.line} both use the name "
                    f"{decl.qualifier!r} -- give one an explicit 'as' alias"
                )
            aliases[decl.qualifier] = canonical_name

        for from_decl in program.from_imports:
            canonical_name = _resolve_and_discover(from_decl.path, from_decl.line)
            for original_name, local_alias in from_decl.names:
                if local_alias in aliases:
                    raise ModuleError(
                        f"'{local_alias}' at line {from_decl.line} collides with an "
                        f"'import ... as {local_alias}' elsewhere in this file -- "
                        f"give one of them a different name"
                    )
                if local_alias in named and named[local_alias] != (canonical_name, original_name):
                    raise ModuleError(
                        f"'{local_alias}' at line {from_decl.line} is imported more than "
                        f"once under that name -- give one an explicit 'as' alias"
                    )
                named[local_alias] = (canonical_name, original_name)

        return aliases, named

    entry_aliases, entry_named = _visit(entry_program, entry_file.parent)
    entry_program.import_aliases = entry_aliases
    entry_program.named_imports = entry_named
    return entry_program, modules
