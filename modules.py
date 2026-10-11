"""Module discovery: resolve and parse every file the entry file transitively imports. One module per
file, and a file is one module however many imports reach it, by whatever paths."""

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    """One imported file, parsed. canonical_name is what the program knows the module by, unique among
    its modules (module_names): it starts every key of the module's declarations. `name` is the
    file's own, without `.ht`: an identifier, and what the module is called to the user."""
    canonical_name: str
    file_path: Path
    program: Program
    import_aliases: Dict[str, str] = field(default_factory=dict)
    named_imports: Dict[str, Tuple[str, str]] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.file_path.stem


def _name_part(directory: str) -> str:
    """A directory's name as part of a module's: what a symbol can't hold becomes `_`."""
    part = re.sub(r'[^A-Za-z0-9_]', '_', directory)
    return part if re.match(r'[A-Za-z_]', part) else f"_{part}"


def module_names(files: List[Path]) -> Dict[Path, str]:
    """What each of a program's module files is known by: its name without `.ht`; or, where several
    files have that name, each with as many of its directories before it as tell them apart, joined
    by `$` (`net$util`, `db$util`). No two are the same, and none depends on the order of `files`."""
    same_name: Dict[str, List[Path]] = {}
    for path in files:
        same_name.setdefault(path.stem, []).append(path)
    names: Dict[Path, str] = {}
    for stem, paths in same_name.items():
        if len(paths) == 1:
            names[paths[0]] = stem
            continue
        deepest = max(len(path.parent.parts) for path in paths)
        for depth in range(1, deepest + 1):
            told_apart = {path: '$'.join([_name_part(d) for d in path.parent.parts[-depth:]] + [stem])
                          for path in paths}
            if len(set(told_apart.values())) == len(paths):
                break
        else:  # directories that differ only in what a symbol can't hold: numbered, in the paths' order
            told_apart = {path: f"{told_apart[path]}${n}" for n, path in enumerate(sorted(paths), start=1)}
        names.update(told_apart)
    return names


_STDLIB_ROOT = Path(__file__).parent / 'stdlib'
# An import's path that starts with this names a module of the standard library; any other is a file's.
STD_PREFIX = 'std:'


def _candidate_path(base_dir: Path, path: str) -> Path:
    """`path` as an absolute Path under base_dir, with '.ht' appended if missing. An import's path was
    read a byte to a character, like the rest of its file: the file system is given those bytes."""
    candidate = path if path.endswith('.ht') else f'{path}.ht'
    name = candidate.encode('latin-1').decode(sys.getfilesystemencoding(), 'surrogateescape')
    return (base_dir / name).resolve()


def _standard_module(name: str) -> Optional[Path]:
    """The standard library's module `name` (what follows `std:`), if it has one. Nothing outside the
    library is reached this way, whatever the name."""
    root = _STDLIB_ROOT.resolve()
    candidate = _candidate_path(root, name)
    return candidate if root in candidate.parents and candidate.is_file() else None


def _resolve_import_path(importer_dir: Path, path: str) -> Optional[Path]:
    """`std:name` is the standard library's; any other path is a file's, relative to the importing
    file. Neither stands in for the other."""
    if path.startswith(STD_PREFIX):
        return _standard_module(path[len(STD_PREFIX):])
    local = _candidate_path(importer_dir, path)
    return local if local.is_file() else None


def _not_found(importer_dir: Path, path: str) -> str:
    """Why `path` resolved to nothing, and what may have been meant."""
    if path.startswith(STD_PREFIX):
        name = path[len(STD_PREFIX):]
        has = sorted(file.stem for file in _STDLIB_ROOT.glob('*.ht'))
        return f"the standard library has no module {quoted_text(name)}; it has {', '.join(has)}"
    looked = f"looked for {path_text(_candidate_path(importer_dir, path))}"
    if _standard_module(path) is not None:
        return f"{looked}; the standard library's module is imported as {quoted_text(STD_PREFIX + path)}"
    return looked


def discover_modules(entry_path: str) -> Tuple[Program, Dict[str, DiscoveredModule]]:
    """Returns (entry_program, {canonical_name: DiscoveredModule})."""
    entry_file = Path(entry_path).resolve()
    entry_program = Parser(lex(str(entry_file))).parse_program()

    # Until every file is found, a module is known by its file: what it is to be called depends on
    # the others (module_names).
    found: Dict[Path, DiscoveredModule] = {}  # in the order their own imports were finished with
    visited = set()  # also those in progress, so that cycles terminate

    def _visit(program: Program, importer_dir: Path) -> Tuple[Dict[str, Path], Dict[str, Tuple[Path, str]]]:
        """Returns (alias -> module's file, local_alias -> (module's file, original_name))."""
        aliases: Dict[str, Path] = {}
        named: Dict[str, Tuple[Path, str]] = {}

        def _resolve_and_discover(decl) -> Path:
            path, at_line = decl.path, decl.line
            resolved = _resolve_import_path(importer_dir, path)
            if resolved is None:
                raise ModuleError(
                    f"Import {quoted_text(path)} at line {at_line} doesn't resolve to a real "
                    f"file ({_not_found(importer_dir, path)})",
                    file=program.file, line=at_line, col=decl.col,
                )
            if resolved == entry_file:  # it would be compiled a second time, as a module
                raise ModuleError(
                    f"Import {quoted_text(path)} at line {at_line} is the program's entry file, which can't be "
                    f"imported -- move what is shared into a module of its own",
                    file=program.file, line=at_line, col=decl.col,
                )
            if resolved in visited:
                return resolved
            if not _MODULE_NAME.fullmatch(resolved.stem):  # it becomes part of every symbol of the module
                raise ModuleError(
                    f"Import {quoted_text(path)} is the file {quoted_text(path_text(resolved.name))}, and a module "
                    f"is named by its file: {quoted_text(path_text(resolved.stem))} must be an identifier "
                    f"(letters, digits, and underscores, not starting with a digit) -- rename the file",
                    file=program.file, line=decl.line, col=decl.col,
                )
            visited.add(resolved)
            sub_program = Parser(lex(str(resolved))).parse_program()
            sub_aliases, sub_named = _visit(sub_program, resolved.parent)
            found[resolved] = DiscoveredModule(
                canonical_name='', file_path=resolved, program=sub_program,
                import_aliases=sub_aliases, named_imports=sub_named,
            )
            return resolved

        for decl in program.imports:
            module = _resolve_and_discover(decl)
            if decl.qualifier in aliases and aliases[decl.qualifier] != module:
                raise ModuleError(
                    f"Two imports at line {decl.line} both use the name "
                    f"{decl.qualifier!r} -- give one an explicit 'as' alias",
                    file=program.file, line=decl.line, col=decl.col,
                )
            aliases[decl.qualifier] = module

        for from_decl in program.from_imports:
            module = _resolve_and_discover(from_decl)
            for original_name, local_alias in from_decl.names:
                if local_alias in aliases:
                    raise ModuleError(
                        f"'{local_alias}' at line {from_decl.line} collides with an "
                        f"'import ... as {local_alias}' elsewhere in this file -- "
                        f"give one of them a different name",
                        file=program.file, line=from_decl.line, col=from_decl.col,
                    )
                if local_alias in named and named[local_alias] != (module, original_name):
                    raise ModuleError(
                        f"'{local_alias}' at line {from_decl.line} is imported more than "
                        f"once under that name -- give one an explicit 'as' alias",
                        file=program.file, line=from_decl.line, col=from_decl.col,
                    )
                named[local_alias] = (module, original_name)

        return aliases, named

    entry_aliases, entry_named = _visit(entry_program, entry_file.parent)

    # Every file is found: each gets its name, and what was known by file is known by name.
    names = module_names(list(found))

    def by_name(aliases: Dict[str, Path], named: Dict[str, Tuple[Path, str]]) -> tuple:
        return ({alias: names[file] for alias, file in aliases.items()},
                {local: (names[file], original) for local, (file, original) in named.items()})

    modules: Dict[str, DiscoveredModule] = {}
    for file, module in found.items():
        module.canonical_name = names[file]
        module.import_aliases, module.named_imports = by_name(module.import_aliases, module.named_imports)
        modules[names[file]] = module
    entry_program.import_aliases, entry_program.named_imports = by_name(entry_aliases, entry_named)
    return entry_program, modules
