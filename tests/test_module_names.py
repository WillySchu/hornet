"""Names that reach the linker or the function table from more than one file: a module is named by
its file, so the file's name must be an identifier; and an extern's name is the same program-wide,
so nothing else in the function table may have it."""

import pytest

from build import build_executable
from modules import ModuleError, discover_modules
from scopes import MergeError
from semantic import SemanticError, analyze
from target import default_target
from tests.targets import run_binary
from tests.test_compiler import GCC_SKIP

MAIN = "def int main():\n    return 0\n"


def _files(tmp_path, **files) -> str:
    """Write `name -> text` (a name's `__` is a `-`, `.ht` is added); the path of `main`."""
    for name, text in files.items():
        (tmp_path / f"{name.replace('__', '-')}.ht").write_text(text)
    return str(tmp_path / "main.ht")


def _analyze(entry: str):
    program, modules = discover_modules(entry)
    return analyze(program, modules)


@pytest.mark.parametrize("import_line", [
    "import 'my-mod' as m",  # `as` renames the qualifier, not the module
    "import 'my-mod'",
    "from 'my-mod' import f",
])
def test_a_module_file_must_be_named_as_an_identifier(tmp_path, import_line):
    entry = _files(tmp_path, my__mod="def int f():\n    return 3\n", main=f"\n{import_line}\n\n" + MAIN)
    with pytest.raises(ModuleError, match=r"is the file 'my-mod.ht'.*'my-mod' must be an identifier.*rename the file") as e:
        discover_modules(entry)
    assert (e.value.file, e.value.line, e.value.col) == (entry, 2, 1)


def test_a_module_file_name_cannot_start_with_a_digit(tmp_path):
    (tmp_path / "9lives.ht").write_text("def int f():\n    return 3\n")
    entry = _files(tmp_path, main="import '9lives' as cat\n\n" + MAIN)
    with pytest.raises(ModuleError, match="'9lives' must be an identifier"):
        discover_modules(entry)


@GCC_SKIP
def test_the_entry_file_and_keyword_named_modules_are_fine(tmp_path):
    # The entry file's name is in no symbol; a keyword is an identifier to the linker, and needs only `as`.
    (tmp_path / "for.ht").write_text("def int f():\n    return 3\n")
    (tmp_path / "my-entry.ht").write_text("import 'for' as loops\n\ndef int main():\n    return loops.f()\n")
    build_executable(str(tmp_path / "my-entry.ht"), str(tmp_path / "out"))
    assert run_binary(default_target(), [tmp_path / "out"]).returncode == 3


def test_an_entry_function_named_like_an_imported_modules_extern(tmp_path):
    entry = _files(
        tmp_path,
        lib="extern int getpid()\n\ndef int pid():\n    return getpid()\n",
        main="from 'lib' import pid\n\ndef int getpid():\n    return 1\n\n" + MAIN,
    )
    with pytest.raises(SemanticError, match=r"Function 'getpid' has the same name as an extern declared in "
                                            r"lib.ht \(line 1\) -- rename the function") as e:
        _analyze(entry)
    assert (e.value.file, e.value.line) == (entry, 3)  # at the function, which is what can be renamed


def test_two_modules_declaring_one_extern(tmp_path):
    entry = _files(
        tmp_path,
        a="extern int getpid()\n\ndef int pid_a():\n    return getpid()\n",
        b="\nextern int getpid()\n\ndef int pid_b():\n    return getpid()\n",
        main="from 'a' import pid_a\nfrom 'b' import pid_b\n\n" + MAIN,
    )
    with pytest.raises(SemanticError, match=r"Extern 'getpid' is already declared in a.ht \(line 1\) -- declare an "
                                            r"extern once and import it") as e:
        _analyze(entry)
    assert (e.value.file, e.value.line) == (str(tmp_path / "b.ht"), 2)


@pytest.mark.parametrize("main,match", [
    ("\nimport 'missing'\n", "doesn't resolve to a real file"),
    ("\nimport 'lib'\nimport 'other/lib' as lib2\n", "Two different files both resolve to module name 'lib'"),
    ("import 'lib'\nimport 'second' as lib\n", "both use the name 'lib'"),
    ("import 'lib'\nfrom 'second' import g as lib\n", "collides with an 'import ... as lib'"),
    ("from 'lib' import f\nfrom 'second' import g as f\n", "imported more than once"),
    ("\nfrom 'lib' import missing\n", "'missing' .*is not declared in module 'lib'"),
    ("\nfrom 'lib' import _hidden\n", "not visible outside the module"),
    ("import 'lib'\n\ndef int g():\n    return lib.missing()\n", "'missing' .*is not declared in module 'lib'"),
])
def test_import_errors_report_a_line_and_a_column(tmp_path, main, match):
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "lib.ht").write_text("def int h():\n    return 3\n")
    entry = _files(tmp_path, lib="def int f():\n    return 1\ndef int _hidden():\n    return 2\n",
                   second="def int g():\n    return 2\n", main=main + "\n" + MAIN)
    with pytest.raises((ModuleError, MergeError), match=match) as e:
        _analyze(entry)
    assert e.value.file == entry and e.value.line >= 2 and e.value.col >= 1
