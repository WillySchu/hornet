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
from typing import Dict, Tuple

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
    own body (`alias.name`) needs resolved against."""
    canonical_name: str
    file_path: Path
    program: Program
    import_aliases: Dict[str, str] = field(default_factory=dict)


def _resolve_import_path(importer_dir: Path, path: str) -> Path:
    """Resolves an ImportDecl's own `path` relative to the importing
    file's own directory -- see this feature's own design discussion
    for why: Hornet has no project-root/manifest concept yet, so the
    importing file's own location is the only thing that exists to
    resolve against -- into a canonical, absolute filesystem Path,
    appending '.ht' when `path` doesn't already end with it (matching
    Python's own `import foo` needing no '.py')."""
    candidate = path if path.endswith('.ht') else f'{path}.ht'
    return (importer_dir / candidate).resolve()


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
    file; two imports in ONE file's own import list producing the
    identical local alias (ambiguous -- which module would `alias.
    name` even mean?); or two DIFFERENT files (by resolved path)
    producing the same canonical_name (this module's own identity,
    used to qualify its declarations, has to be globally unique across
    the whole discovered set -- an `as` rename at one import site only
    fixes the LOCAL alias collision case above, not this one, since it
    doesn't change what the other module's own declarations get
    mangled as; the fix there is renaming one of the actual files)."""
    entry_file = Path(entry_path).resolve()
    entry_program = Parser(lex(str(entry_file))).parse_program()

    modules: Dict[str, DiscoveredModule] = {}  # canonical_name -> DiscoveredModule, populated once a module's own _visit call returns
    by_path: Dict[Path, str] = {}  # resolved path -> canonical_name; doubles as the "already discovered (or in progress)" set

    def _visit(program: Program, importer_dir: Path) -> Dict[str, str]:
        """Resolves every one of program's own imports, recursively
        visiting each not-yet-discovered one. Returns THIS program's
        own alias -> canonical_name mapping."""
        aliases: Dict[str, str] = {}
        for decl in program.imports:
            resolved = _resolve_import_path(importer_dir, decl.path)
            if not resolved.is_file():
                raise ModuleError(
                    f"Import {decl.path!r} at line {decl.line} doesn't resolve to a real "
                    f"file (looked for {resolved})"
                )

            if resolved in by_path:
                # Already discovered, or mid-discovery as an ancestor
                # of this very call (a genuine cycle) -- either way,
                # reuse its own, already-settled canonical name rather
                # than re-deriving, re-parsing, or re-visiting it.
                canonical_name = by_path[resolved]
            else:
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
                sub_aliases = _visit(sub_program, resolved.parent)
                modules[canonical_name] = DiscoveredModule(
                    canonical_name=canonical_name, file_path=resolved, program=sub_program,
                    import_aliases=sub_aliases,
                )

            if decl.qualifier in aliases and aliases[decl.qualifier] != canonical_name:
                raise ModuleError(
                    f"Two imports at line {decl.line} both use the name "
                    f"{decl.qualifier!r} -- give one an explicit 'as' alias"
                )
            aliases[decl.qualifier] = canonical_name
        return aliases

    entry_aliases = _visit(entry_program, entry_file.parent)
    entry_program.import_aliases = entry_aliases
    return entry_program, modules
