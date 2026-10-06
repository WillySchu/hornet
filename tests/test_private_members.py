"""A struct's field or method whose name starts with `_` is private to the module that declares the
struct, like a top-level name: other modules can't read or write the field, call the method, or
give the field a value in a struct literal. Whole values still copy, compare, and print anywhere."""

import pytest

from build import build_executable
from modules import discover_modules
from semantic import SemanticError, analyze
from target import default_target
from tests.targets import run_binary
from tests.test_compiler import GCC_SKIP

COUNTER = (
    "type Counter struct:\n"
    "    int step\n"
    "    int _count\n"
    "\n"
    "    def int next(*self):\n"
    "        self._bump()\n"
    "        return self._count\n"
    "\n"
    "    def _bump(*self):\n"
    "        self._count += self.step\n"
    "\n"
    "def Counter starting_at(int n):\n"
    "    return Counter(1, n)\n"            # by position: inside the module
    "\n"
    "def int peek(Counter c):\n"            # a free function of the module sees the field too
    "    return c._count\n"
)
IMPORTS = "from 'counter' import Counter, starting_at, peek\n\n"


def _write(tmp_path, main: str, **more) -> str:
    (tmp_path / "counter.ht").write_text(COUNTER)
    for name, text in more.items():
        (tmp_path / f"{name}.ht").write_text(text)
    (tmp_path / "main.ht").write_text(main)
    return str(tmp_path / "main.ht")


@GCC_SKIP
def test_private_members_work_inside_their_module_and_values_travel_freely(tmp_path):
    entry = _write(
        tmp_path, IMPORTS +
        "type Local struct:\n"
        "    int _mine\n"
        "    def int _get(self):\n"
        "        return self._mine\n"
        "def int main():\n"
        "    Counter zero\n"                       # every field zero
        "    Counter c = starting_at(10)\n"
        "    Counter by_two = Counter(step=2)\n"   # the private field starts as zero
        "    print(c.next() + c.next())\n"
        "    print(by_two.next())\n"
        "    print(zero.step)\n"
        "    print(peek(c))\n"
        "    print(c)\n"
        "    Counter copy = c\n"
        "    print(copy == c)\n"
        "    Local l = Local(7)\n"                 # the entry file's own
        "    print(l._get() + l._mine)\n"
        "    return 0\n")
    build_executable(entry, str(tmp_path / "out"))
    result = run_binary(default_target(), [tmp_path / "out"], capture_output=True, text=True)
    assert (result.returncode, result.stdout) == (0, "23\n2\n0\n12\nCounter(step: 1, _count: 12)\ntrue\n14\n")


@pytest.mark.parametrize("statement,match", [
    ("print(c._count)", "Field '_count' of 'Counter' is not visible outside the module that defines the struct"),
    ("c._count = 5", "Field '_count' of 'Counter' is not visible"),
    ("c._count += 1", "Field '_count' of 'Counter' is not visible"),
    ("*int p = &c._count", "Field '_count' of 'Counter' is not visible"),
    ("*Counter p = &c\n    print(p._count)", "Field '_count' of 'Counter' is not visible"),
    ("c._bump()", "Method '_bump' of 'Counter' is not visible outside the module that defines the struct"),
    ("Counter d = Counter(1, 2)", r"'Counter\(...\)' gives every field by position, but _count is private"),
    ("Counter d = Counter(_count=1)", "Field '_count' of 'Counter' is not visible"),
])
def test_another_module_cannot_reach_them(tmp_path, statement, match):
    entry = _write(
        tmp_path, IMPORTS + f"def int main():\n    Counter c = starting_at(1)\n    {statement}\n    return 0\n")
    program, modules = discover_modules(entry)
    with pytest.raises(SemanticError, match=match):
        analyze(program, modules)


def test_nor_can_a_module_that_only_receives_the_value(tmp_path):
    entry = _write(
        tmp_path, "from 'user' import look\nfrom 'counter' import starting_at\n\n"
                  "def int main():\n    return look(starting_at(1))\n",
        user="from 'counter' import Counter\n\ndef int look(Counter c):\n    return c._count\n")
    program, modules = discover_modules(entry)
    with pytest.raises(SemanticError, match="Field '_count' of 'Counter' is not visible") as e:
        analyze(program, modules)
    assert e.value.file.endswith("user.ht")
