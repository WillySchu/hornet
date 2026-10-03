"""A builtin function's name (`print`, `len`, `append`, `del`, `bytes`, `panic`) can't be declared
or imported under, in the entry file or in a module: it would replace the builtin there."""

import pytest

from modules import discover_modules
from scopes import BUILTIN_FUNCTION_NAMES, MergeError
from semantic import SemanticError, analyze

MAIN = "def int main():\n    return 0\n"


def _analyze(tmp_path, **files):
    for name, text in files.items():
        (tmp_path / f"{name}.ht").write_text(text)
    program, modules = discover_modules(str(tmp_path / "main.ht"))
    return analyze(program, modules)


def test_the_builtins():
    assert BUILTIN_FUNCTION_NAMES == {'print', 'len', 'append', 'del', 'bytes', 'panic'}


@pytest.mark.parametrize("declaration", [
    "def print(int x):\n    return\n",
    "def never panic(str m):\n    while true:\n        m = m\n",
    "type len struct:\n    int x\n",
    "const int append = 1\n",
    "extern int bytes()\n",
])
@pytest.mark.parametrize("where", ["main", "lib"])
def test_a_declaration_cannot_take_a_builtins_name(tmp_path, declaration, where):
    files = {"main": "import 'lib'\n\n" + MAIN, "lib": "def int f():\n    return 1\n"}
    files[where] += declaration
    with pytest.raises(SemanticError, match="is a builtin and can't be"):
        _analyze(tmp_path, **files)


def test_an_import_cannot_be_named_after_a_builtin(tmp_path):
    with pytest.raises(MergeError, match="'print' .*is a builtin and can't name an import") as e:
        _analyze(tmp_path, main="\nfrom 'lib' import f as print\n\n" + MAIN, lib="def int f():\n    return 1\n")
    assert (e.value.line, e.value.col) == (2, 1)


def test_methods_and_variables_may_share_a_builtins_name(tmp_path):
    # Neither is called as `name(...)`, so neither replaces the builtin.
    _analyze(tmp_path, main=(
        "type P struct:\n"
        "    int x\n"
        "    def int len(self):\n"
        "        return self.x\n"
        "def int main():\n"
        "    int print = 4\n"
        "    P p = P(print)\n"
        "    return p.len() + len('ab')\n"
    ))
