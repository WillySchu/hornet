"""Tests for modules.py's own discover_modules: resolving and parsing
every file an entry .ht file transitively imports, before any merging
or semantic analysis happens.
"""

import tempfile
from pathlib import Path

import pytest

from modules import discover_modules, ModuleError


def _write(tmpdir: str, name: str, content: str) -> str:
    path = Path(tmpdir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return str(path)


def test_entry_file_with_no_imports():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(tmpdir, "main.ht", "def int main():\n    return 0\n")
        entry_program, modules = discover_modules(entry)
        assert entry_program.imports == []
        assert modules == {}


def test_single_import_is_discovered():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(tmpdir, "main.ht", "import 'utils'\n\ndef int main():\n    return 0\n")
        _write(tmpdir, "utils.ht", "def int helper():\n    return 42\n")
        entry_program, modules = discover_modules(entry)
        assert set(modules.keys()) == {"utils"}
        assert entry_program.import_aliases == {"utils": "utils"}
        assert modules["utils"].program.functions[0].name == "helper"


def test_explicit_as_alias_used_for_local_resolution_not_canonical_name(): 
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(tmpdir, "main.ht", "import 'utils' as u\n\ndef int main():\n    return 0\n")
        _write(tmpdir, "utils.ht", "def int helper():\n    return 42\n")
        entry_program, modules = discover_modules(entry)
        # The canonical name stays "utils" (derived from the FILE),
        # even though this importer locally calls it "u".
        assert set(modules.keys()) == {"utils"}
        assert entry_program.import_aliases == {"u": "utils"}


def test_transitive_import_is_discovered():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(tmpdir, "main.ht", "import 'a'\n\ndef int main():\n    return 0\n")
        _write(tmpdir, "a.ht", "import 'b'\n\ndef int fromA():\n    return 1\n")
        _write(tmpdir, "b.ht", "def int fromB():\n    return 2\n")
        entry_program, modules = discover_modules(entry)
        assert set(modules.keys()) == {"a", "b"}
        assert modules["a"].import_aliases == {"b": "b"}


def test_diamond_import_discovers_shared_dependency_exactly_once():
    """main imports both b and c, and BOTH b and c import d -- d must
    be discovered (and, crucially, parsed) exactly once, not twice."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(tmpdir, "main.ht", "import 'b'\nimport 'c'\n\ndef int main():\n    return 0\n")
        _write(tmpdir, "b.ht", "import 'd'\n\ndef int fromB():\n    return 1\n")
        _write(tmpdir, "c.ht", "import 'd'\n\ndef int fromC():\n    return 2\n")
        _write(tmpdir, "d.ht", "def int fromD():\n    return 3\n")
        entry_program, modules = discover_modules(entry)
        assert set(modules.keys()) == {"b", "c", "d"}
        # Each module object is discovered exactly once -- b's and c's
        # own alias for "d" point at the SAME canonical name, and
        # there's only one entry in `modules` for it at all.
        assert modules["b"].import_aliases["d"] == "d"
        assert modules["c"].import_aliases["d"] == "d"


def test_circular_import_terminates_and_both_sides_resolve():
    """a imports b, b imports a right back -- discovery must terminate
    (not infinite-loop) and both modules must still resolve their own
    alias for one another correctly. Whether a program built from this
    actually type-checks is semantic.py's own concern (the merge
    model's two-phase analysis is what makes it SAFE to allow this at
    all) -- this test is scoped to discovery alone: it must not hang
    or crash on a genuine cycle."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(tmpdir, "main.ht", "import 'a'\n\ndef int main():\n    return 0\n")
        _write(tmpdir, "a.ht", "import 'b'\n\ndef int fromA():\n    return 1\n")
        _write(tmpdir, "b.ht", "import 'a'\n\ndef int fromB():\n    return 2\n")
        entry_program, modules = discover_modules(entry)
        assert set(modules.keys()) == {"a", "b"}
        assert modules["a"].import_aliases == {"b": "b"}
        assert modules["b"].import_aliases == {"a": "a"}


def test_self_import_terminates():
    """A file importing itself -- the degenerate one-node cycle."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(tmpdir, "main.ht", "import 'main'\n\ndef int main():\n    return 0\n")
        entry_program, modules = discover_modules(entry)
        assert set(modules.keys()) == {"main"}
        assert entry_program.import_aliases == {"main": "main"}


def test_nested_path_import():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(tmpdir, "main.ht", "import 'sub/dir/utils'\n\ndef int main():\n    return 0\n")
        _write(tmpdir, "sub/dir/utils.ht", "def int helper():\n    return 1\n")
        entry_program, modules = discover_modules(entry)
        assert set(modules.keys()) == {"utils"}


def test_import_relative_to_importing_files_own_directory():
    """b.ht, imported from sub/, itself imports 'sibling' -- that
    resolves relative to sub/ (b's own directory), not the entry
    file's directory, even though it's b that's being imported
    transitively."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(tmpdir, "main.ht", "import 'sub/b'\n\ndef int main():\n    return 0\n")
        _write(tmpdir, "sub/b.ht", "import 'sibling'\n\ndef int fromB():\n    return 1\n")
        _write(tmpdir, "sub/sibling.ht", "def int fromSibling():\n    return 2\n")
        entry_program, modules = discover_modules(entry)
        assert set(modules.keys()) == {"b", "sibling"}


def test_import_path_may_include_ht_extension_explicitly():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(tmpdir, "main.ht", "import 'utils.ht'\n\ndef int main():\n    return 0\n")
        _write(tmpdir, "utils.ht", "def int helper():\n    return 1\n")
        entry_program, modules = discover_modules(entry)
        assert set(modules.keys()) == {"utils"}


def test_nonexistent_import_path_is_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(tmpdir, "main.ht", "import 'nonexistent'\n\ndef int main():\n    return 0\n")
        with pytest.raises(ModuleError, match="doesn't resolve to a real file"):
            discover_modules(entry)


def test_two_different_files_with_same_basename_is_rejected():
    """sub1/utils.ht and sub2/utils.ht are genuinely different files
    that would both want the canonical name "utils" -- module names
    must be globally unique, so this is a hard error regardless of
    how either is imported."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "import 'sub1/utils'\nimport 'sub2/utils'\n\ndef int main():\n    return 0\n",
        )
        _write(tmpdir, "sub1/utils.ht", "def int a():\n    return 1\n")
        _write(tmpdir, "sub2/utils.ht", "def int b():\n    return 2\n")
        with pytest.raises(ModuleError, match="module names must be globally unique"):
            discover_modules(entry)


def test_same_basename_collision_detected_even_when_transitive():
    """The collision doesn't have to be at the entry file's own import
    list -- it's detected anywhere in the discovered set."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "import 'a'\nimport 'sub/utils'\n\ndef int main():\n    return 0\n",
        )
        _write(tmpdir, "a.ht", "import 'utils'\n\ndef int fromA():\n    return 1\n")
        _write(tmpdir, "utils.ht", "def int x():\n    return 1\n")
        _write(tmpdir, "sub/utils.ht", "def int y():\n    return 2\n")
        with pytest.raises(ModuleError, match="module names must be globally unique"):
            discover_modules(entry)


def test_duplicate_local_alias_in_one_file_is_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "import 'utils'\nimport 'other' as utils\n\ndef int main():\n    return 0\n",
        )
        _write(tmpdir, "utils.ht", "def int a():\n    return 1\n")
        _write(tmpdir, "other.ht", "def int b():\n    return 2\n")
        with pytest.raises(ModuleError, match="give one an explicit 'as' alias"):
            discover_modules(entry)


def test_same_module_imported_twice_with_same_alias_is_not_a_collision():
    """Two DIFFERENT files both importing 'utils' (as its own default
    alias) isn't a collision at all -- they're two separate files each
    referring to the same, single shared module, not one file with an
    ambiguous alias."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "import 'a'\nimport 'utils'\n\ndef int main():\n    return 0\n",
        )
        _write(tmpdir, "a.ht", "import 'utils'\n\ndef int fromA():\n    return 1\n")
        _write(tmpdir, "utils.ht", "def int helper():\n    return 1\n")
        entry_program, modules = discover_modules(entry)
        assert set(modules.keys()) == {"a", "utils"}
        assert entry_program.import_aliases == {"a": "a", "utils": "utils"}
        assert modules["a"].import_aliases == {"utils": "utils"}
