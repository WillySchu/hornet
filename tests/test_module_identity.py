"""A module is its file. Two files with one name, in different directories, are two modules: each is
known inside the compiler by its name with as many directories before it as tell them apart
(`net$util`, `db$util`), and to the user by its name alone."""

import subprocess
from pathlib import Path

import pytest

from build import build_executable, executable_name
from compile import generate_asm
from modules import ModuleError, discover_modules, module_names
from scopes import MergeError
from semantic import SemanticError, analyze
from target import Target, default_target
from tests.test_compiler import GCC_SKIP


def _tree(root: Path, **files: str) -> str:
    """Write `files` (`dir__name` is `dir/name.ht`) under `root`; the path of main.ht."""
    for name, source in files.items():
        path = root / (name.replace('__', '/') + '.ht')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    return str(root / 'main.ht')


def _run(entry: str) -> str:
    exe = Path(entry).with_name(executable_name('main', default_target()))
    build_executable(entry, str(exe))
    return subprocess.run([str(exe)], capture_output=True, text=True).stdout


def test_what_files_are_known_by():
    names = module_names([Path(p) for p in (
        '/p/app/strings.ht', '/p/app/net/util.ht', '/p/app/db/util.ht',
        '/p/a/x/errors.ht', '/p/b/x/errors.ht', '/p/c/errors.ht',        # (one directory doesn't tell these apart)
        '/p/my-lib/2d/shape.ht', '/p/my_lib/2d/shape.ht', '/q/my lib/2d/shape.ht')])
    assert sorted(names.values()) == sorted([
        'strings', 'net$util', 'db$util', 'a$x$errors', 'b$x$errors', 'p$c$errors',
        # What a symbol can't hold is `_`; where that leaves files alike, they are numbered.
        '_$p$my_lib$_2d$shape$1', '_$p$my_lib$_2d$shape$2', '_$q$my_lib$_2d$shape$3'])
    files = [Path('/p/b/util.ht'), Path('/p/a/util.ht'), Path('/p/one.ht')]
    assert module_names(files) == module_names(files[::-1]) == {
        files[0]: 'b$util', files[1]: 'a$util', files[2]: 'one'}


@GCC_SKIP
def test_two_files_of_one_name_are_two_modules(tmp_path):
    entry = _tree(
        tmp_path,
        net__util="type Limit struct:\n    int n\n\n"
                  "def Limit limit():\n    return Limit(1)\n\n"
                  "def int _own():\n    return 10\n",
        db__util="from '../net/util' import Limit as NetLimit\n\n"
                 "type Limit struct:\n    str text\n\n"                       # a type of its own, of the same name
                 "def Limit limit():\n    return Limit('two')\n\n"
                 "def int both(NetLimit other):\n    return other.n + len(limit().text) + _own()\n\n"
                 "def int _own():\n    return 20\n",
        main="import 'net/util'\nimport 'db/util' as db\nfrom 'db/util' import both\n\n"
             "def int main():\n"
             "    print(util.limit())\n    print(db.limit())\n    print(both(util.limit()))\n"
             "    return 0\n")
    assert _run(entry) == "Limit(n: 1)\nLimit(text: 'two')\n24\n"
    entry_program, modules = discover_modules(entry)
    asm = generate_asm(analyze(entry_program, modules), target=Target('x86_64', 'linux'))
    for symbol in ('net$util$limit', 'db$util$limit', 'db$util$both', 'db$util$_own'):
        assert f"\n{symbol}:\n" in asm
    assert '\n    .asciz "limit"\n' in asm and '\n    .asciz "util$limit"\n' not in asm    # a name is what was declared


@GCC_SKIP
def test_a_file_named_like_a_standard_module_is_another_module(tmp_path):
    entry = _tree(
        tmp_path,
        errors="type Error struct:\n    int code\n\ndef Error mine():\n    return Error(7)\n",
        main="from 'errors' import mine\nfrom 'std:fmt' import int_to_str\n\n"   # (fmt uses the library's errors)
             "def int main():\n    print(int_to_str(mine().code))\n    return 0\n")
    assert _run(entry) == "7\n"
    _, modules = discover_modules(entry)
    assert {name for name in modules if name.endswith('errors')} == {f"{tmp_path.name}$errors", "stdlib$errors"}


def test_one_file_by_two_paths_is_one_module(tmp_path):
    entry = _tree(tmp_path, lib__one="def int f():\n    return 1\n",
                  main="import 'lib/one'\nimport './lib/../lib/one' as again\n\ndef int main():\n    return 0\n")
    entry_program, modules = discover_modules(entry)
    assert list(modules) == ['one'] and entry_program.import_aliases == {'one': 'one', 'again': 'one'}


@pytest.mark.parametrize("main,error,message", [
    ("import 'net/util'\nimport 'db/util'\n", ModuleError,
     "Two imports .*both use the name 'util' -- give one an explicit 'as' alias"),
    # To the user a module is called what its file is.
    ("import 'net/util'\nimport 'db/util' as db\n\ndef int f():\n    return db._own()\n", MergeError,
     "'util._own' .*is not visible outside the module that defines it"),
    ("import 'net/util'\nimport 'db/util' as db\n\ndef int f():\n    return db.missing()\n", MergeError,
     r"'missing' .*is not declared in module 'util' \(.*db.util\.ht\)"),
    # Two types of one name are two types.
    ("from 'net/util' import limit\nfrom 'db/util' import Limit\n\ndef Limit f():\n    return limit()\n", SemanticError,
     "Function is declared to return Limit, but this 'return' statement returns Limit"),
])
def test_what_is_refused(tmp_path, main, error, message):
    entry = _tree(
        tmp_path,
        net__util="type Limit struct:\n    int n\n\ndef Limit limit():\n    return Limit(1)\n",
        db__util="type Limit struct:\n    str text\n\ndef int _own():\n    return 20\n",
        main=main + "\ndef int main():\n    return 0\n")
    with pytest.raises(error, match=message):
        analyze(*discover_modules(entry))
