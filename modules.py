"""Module discovery: resolve and parse every file the entry file transitively imports. One module per file."""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

from diagnostics import CompileError, path_text, quoted_text
from lexer import lex
from parser import Parser, Program


_MODULE_NAME = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')  # an identifier, as the lexer reads one


class ModuleError(CompileError):
    """Import resolution failure."""

    def __init__(self, message: str, file: Optional[str] = None, line: int = 0, col: int = 0):
        clean = re.sub(rf' at line {line}\b', '', message) if line else message
        super().__init__(clean, file, line, col, legacy=message)


@dataclass
class DiscoveredModule:
    """One imported file, parsed. canonical_name is the file's name without `.ht`: an identifier, unique
    among the program's modules."""
    canonical_name: str
    file_path: Path
    program: Program
    import_aliases: Dict[str, str] = field(default_factory=dict)
    named_imports: Dict[str, Tuple[str, str]] = field(default_factory=dict)


_STDLIB_ROOT = Path(__file__).parent / 'stdlib'


def _candidate_path(base_dir: Path, path: str) -> Path:
    """`path` as an absolute Path under base_dir, with '.ht' appended if missing. An import's path was
    read a byte to a character, like the rest of its file: the file system is given those bytes."""
    candidate = path if path.endswith('.ht') else f'{path}.ht'
    name = candidate.encode('latin-1').decode(sys.getfilesystemencoding(), 'surrogateescape')
    return (base_dir / name).resolve()


def _resolve_import_path(importer_dir: Path, path: str) -> Optional[Path]:
    """Resolve relative to the importing file, then fall back to _STDLIB_ROOT."""
    local = _candidate_path(importer_dir, path)
    if local.is_file():
        return local
    stdlib_candidate = _candidate_path(_STDLIB_ROOT, path)
    if stdlib_candidate.is_file():
        return stdlib_candidate
    return None


def discover_modules(entry_path: str) -> Tuple[Program, Dict[str, DiscoveredModule]]:
    """Returns (entry_program, {canonical_name: DiscoveredModule})."""
    entry_file = Path(entry_path).resolve()
    entry_program = Parser(lex(str(entry_file))).parse_program()

    modules: Dict[str, DiscoveredModule] = {}  # canonical_name -> module
    by_path: Dict[Path, str] = {}  # path -> canonical_name; also the visited/in-progress set

    def _visit(program: Program, importer_dir: Path) -> Tuple[Dict[str, str], Dict[str, Tuple[str, str]]]:
        """Returns (alias -> canonical_name, local_alias -> (canonical_name, original_name))."""
        aliases: Dict[str, str] = {}
        named: Dict[str, Tuple[str, str]] = {}

        def _resolve_and_discover(decl) -> str:
            path, at_line = decl.path, decl.line
            resolved = _resolve_import_path(importer_dir, path)
            if resolved is None:
                raise ModuleError(
                    f"Import {quoted_text(path)} at line {at_line} doesn't resolve to a real "
                    f"file (looked for {path_text(_candidate_path(importer_dir, path))}, or in "
                    f"the standard library)",
                    file=program.file, line=at_line, col=decl.col,
                )
            if resolved == entry_file:  # it would be compiled a second time, as a module
                raise ModuleError(
                    f"Import {quoted_text(path)} at line {at_line} is the program's entry file, which can't be "
                    f"imported -- move what is shared into a module of its own",
                    file=program.file, line=at_line, col=decl.col,
                )
            if resolved in by_path:
                return by_path[resolved]
            canonical_name = resolved.stem
            if not _MODULE_NAME.fullmatch(canonical_name):  # it becomes part of every symbol of the module
                raise ModuleError(
                    f"Import {quoted_text(path)} is the file {quoted_text(path_text(resolved.name))}, and a module "
                    f"is named by its file: {quoted_text(path_text(canonical_name))} must be an identifier "
                    f"(letters, digits, and underscores, not starting with a digit) -- rename the file",
                    file=program.file, line=decl.line, col=decl.col,
                )
            if canonical_name in modules or canonical_name in by_path.values():
                raise ModuleError(
                    f"Two different files both resolve to module name "
                    f"{canonical_name!r} -- module names must be globally "
                    f"unique across every imported file; rename one of them "
                    f"(the second is {path_text(resolved)})",
                    file=program.file, line=decl.line, col=decl.col,
                )
            by_path[resolved] = canonical_name  # Reserved before recursing so cycles terminate.
            sub_program = Parser(lex(str(resolved))).parse_program()
            sub_aliases, sub_named = _visit(sub_program, resolved.parent)
            modules[canonical_name] = DiscoveredModule(
                canonical_name=canonical_name, file_path=resolved, program=sub_program,
                import_aliases=sub_aliases, named_imports=sub_named,
            )
            return canonical_name

        for decl in program.imports:
            canonical_name = _resolve_and_discover(decl)
            if decl.qualifier in aliases and aliases[decl.qualifier] != canonical_name:
                raise ModuleError(
                    f"Two imports at line {decl.line} both use the name "
                    f"{decl.qualifier!r} -- give one an explicit 'as' alias",
                    file=program.file, line=decl.line, col=decl.col,
                )
            aliases[decl.qualifier] = canonical_name

        for from_decl in program.from_imports:
            canonical_name = _resolve_and_discover(from_decl)
            for original_name, local_alias in from_decl.names:
                if local_alias in aliases:
                    raise ModuleError(
                        f"'{local_alias}' at line {from_decl.line} collides with an "
                        f"'import ... as {local_alias}' elsewhere in this file -- "
                        f"give one of them a different name",
                        file=program.file, line=from_decl.line, col=from_decl.col,
                    )
                if local_alias in named and named[local_alias] != (canonical_name, original_name):
                    raise ModuleError(
                        f"'{local_alias}' at line {from_decl.line} is imported more than "
                        f"once under that name -- give one an explicit 'as' alias",
                        file=program.file, line=from_decl.line, col=from_decl.col,
                    )
                named[local_alias] = (canonical_name, original_name)

        return aliases, named

    entry_aliases, entry_named = _visit(entry_program, entry_file.parent)
    entry_program.import_aliases = entry_aliases
    entry_program.named_imports = entry_named
    return entry_program, modules
