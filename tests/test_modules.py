"""Tests for modules.py's own discover_modules: resolving and parsing
every file an entry .ht file transitively imports, before any merging
or semantic analysis happens.
"""

import tempfile
import modules
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


def test_basic_named_import_is_discovered():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(tmpdir, "main.ht", "from 'utils' import helper\n\ndef int main():\n    return 0\n")
        _write(tmpdir, "utils.ht", "def int helper():\n    return 42\n")
        entry_program, modules = discover_modules(entry)
        assert set(modules.keys()) == {"utils"}
        assert entry_program.named_imports == {"helper": ("utils", "helper")}


def test_named_import_with_as_renaming():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht", "from 'utils' import helper as h\n\ndef int main():\n    return 0\n")
        _write(tmpdir, "utils.ht", "def int helper():\n    return 42\n")
        entry_program, modules = discover_modules(entry)
        # The MAPPING'S OWN target is unaffected by the local rename --
        # it still points at "helper" (the name in utils.ht itself),
        # just reachable under the local key "h".
        assert entry_program.named_imports == {"h": ("utils", "helper")}


def test_multiple_named_imports_in_one_statement():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "from 'utils' import helper, other as o\n\ndef int main():\n    return 0\n",
        )
        _write(tmpdir, "utils.ht", "def int helper():\n    return 1\n\ndef int other():\n    return 2\n")
        entry_program, modules = discover_modules(entry)
        assert entry_program.named_imports == {"helper": ("utils", "helper"), "o": ("utils", "other")}


def test_named_import_alias_colliding_with_plain_import_qualifier_is_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "import 'a' as utils\nfrom 'b' import foo as utils\n\ndef int main():\n    return 0\n",
        )
        _write(tmpdir, "a.ht", "def int x():\n    return 1\n")
        _write(tmpdir, "b.ht", "def int foo():\n    return 2\n")
        with pytest.raises(ModuleError, match="collides with an 'import ... as utils'"):
            discover_modules(entry)


def test_plain_import_qualifier_colliding_with_earlier_named_import_is_rejected():
    """The identical collision, in the opposite source order -- the
    named import is written FIRST, the plain import SECOND. Confirmed
    separately from the test above since discover_modules processes
    every plain import before any named one, regardless of source
    order, and this is the direction that could plausibly be missed
    by an implementation that only checks in textual order."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "from 'b' import foo as utils\nimport 'a' as utils\n\ndef int main():\n    return 0\n",
        )
        _write(tmpdir, "a.ht", "def int x():\n    return 1\n")
        _write(tmpdir, "b.ht", "def int foo():\n    return 2\n")
        with pytest.raises(ModuleError, match="collides with an 'import ... as utils'"):
            discover_modules(entry)


def test_two_named_imports_colliding_on_alias_from_different_sources_is_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "from 'a' import x as shared\nfrom 'b' import foo as shared\n\ndef int main():\n    return 0\n",
        )
        _write(tmpdir, "a.ht", "def int x():\n    return 1\n")
        _write(tmpdir, "b.ht", "def int foo():\n    return 2\n")
        with pytest.raises(ModuleError, match="is imported more than once under that name"):
            discover_modules(entry)


def test_qualified_and_named_import_of_the_same_module_is_not_a_collision():
    """import 'utils' (a plain qualifier) and from 'utils' import ...
    (a named import) both targeting the SAME module -- this is a
    genuinely different case from the collision tests above, since
    the two don't actually share a local name at all (the qualifier
    is "utils", the named import's own local alias is "add") -- only
    a shared LOCAL NAME is ever a collision, not a shared source
    module."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "import 'utils'\nfrom 'utils' import add\n\ndef int main():\n    return 0\n",
        )
        _write(tmpdir, "utils.ht", "def int add(int a, int b):\n    return a + b\n")
        entry_program, modules = discover_modules(entry)
        assert entry_program.import_aliases == {"utils": "utils"}
        assert entry_program.named_imports == {"add": ("utils", "add")}


def test_fallback_reaches_the_stdlib_when_nothing_local_matches(monkeypatch, tmp_path):
    """The core fallback mechanism (see modules.py's own module
    docstring): a program with no local file matching the import path
    still resolves correctly once a stdlib location has a matching
    one. _STDLIB_ROOT is monkeypatched to an isolated temp directory
    for this test, deliberately independent of whatever the real,
    shipped stdlib happens to contain at any given time -- the real
    stdlib's own content gets its own, separate tests."""
    stdlib_dir = tmp_path / "fake_stdlib"
    stdlib_dir.mkdir()
    (stdlib_dir / "greet.ht").write_text("def int hello():\n    return 99\n")
    monkeypatch.setattr(modules, "_STDLIB_ROOT", stdlib_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(tmpdir, "main.ht", "import 'greet'\n\ndef int main():\n    return greet.hello()\n")
        entry_program, discovered = discover_modules(entry)
        assert set(discovered.keys()) == {"greet"}
        assert discovered["greet"].file_path == (stdlib_dir / "greet.ht").resolve()


def test_local_file_takes_priority_over_the_stdlib_even_when_both_exist():
    """The other half of fallback semantics: relative resolution
    takes UNCONDITIONAL priority. A local file with the same name as
    a stdlib module is always what a bare import reaches -- never
    silently shadowed. Uses the real _STDLIB_ROOT deliberately (not
    monkeypatched) specifically to confirm this against the real,
    shipped 'c' module: a local file also named c.ht must still win."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(tmpdir, "main.ht", "import 'c'\n\ndef int main():\n    return c.localOnly()\n")
        _write(tmpdir, "c.ht", "def int localOnly():\n    return 7\n")
        entry_program, discovered = discover_modules(entry)
        assert discovered["c"].file_path == (Path(tmpdir) / "c.ht").resolve()


def test_nonexistent_path_checks_both_locations_before_failing(monkeypatch, tmp_path):
    stdlib_dir = tmp_path / "fake_stdlib"
    stdlib_dir.mkdir()
    monkeypatch.setattr(modules, "_STDLIB_ROOT", stdlib_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(tmpdir, "main.ht", "import 'nowhere'\n\ndef int main():\n    return 0\n")
        with pytest.raises(ModuleError, match="doesn't resolve to a real file"):
            discover_modules(entry)
