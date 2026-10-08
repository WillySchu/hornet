"""Functions `main` can't reach are checked but not compiled: importing a module costs what is used
of it. Reachability follows the calls in the typed tree, where each is resolved; nothing else in
Hornet can call a function."""

import subprocess
import sys
from pathlib import Path

import pytest

from compile import compile_to_asm, generate_asm
from dump import dump
from ir.program_builder import build_ir_program, reachable_functions
from ir.typed_builder import link_name
from modules import discover_modules
from semantic import SemanticError, analyze
from target import Target
from tests.test_compiler import GCC_SKIP, _parse, assert_program_stdout

ROOT = Path(__file__).resolve().parent.parent

SOURCE = (
    "extern int abs(int n)\n"
    "extern int labs(int n)\n"
    "type Light enum:\n"
    "    Red\n"
    "    Green\n"
    "    def Light other(self):\n"
    "        return Light.Green\n"
    "    def str unused_method(self):\n"
    "        return 'never said'\n"
    "type Counter struct:\n"
    "    int n\n"
    "    def bump(*self):\n"
    "        self.n += helper(1)\n"
    "    def int unused_total(self):\n"
    "        return labs(self.n)\n"
    "def int helper(int n):\n"                    # reached through a method
    "    return abs(n)\n"
    "def int deep(int n):\n"                      # reached through used()
    "    return n * 2\n"
    "def int used(int n):\n"
    "    return deep(n) + 1\n"
    "def int only_from_unused(int n):\n"          # called, but only by what is never called
    "    return n + 100\n"
    "def int unused(int n):\n"
    "    return only_from_unused(n)\n"
    "def int ping(int n):\n"                      # each calls the other; nothing else calls either
    "    return pong(n)\n"
    "def int pong(int n):\n"
    "    return ping(n)\n"
    "def int main():\n"
    "    Counter c = Counter(0)\n"
    "    c.bump()\n"
    "    print(used(c.n))\n"
    "    print(Light.Red.other())\n"
    "    return 0\n"
)
REACHED = ['Light.other', 'Counter.bump', 'helper', 'deep', 'used', 'main']
NOT_REACHED = ['Light.unused_method', 'Counter.unused_total', 'only_from_unused', 'unused', 'ping', 'pong']


def test_what_main_can_reach():
    program = analyze(_parse(SOURCE))
    everyone = [fn.name for fn in program.functions]
    assert set(everyone) == set(REACHED + NOT_REACHED)  # all of them are checked
    in_order = [name for name in everyone if name in REACHED]
    assert [fn.name for fn in reachable_functions(program)] == in_order  # the program's order is kept
    assert [fn.name for fn in build_ir_program(program).functions] == [link_name(name) for name in in_order]
    everything = build_ir_program(program, keep_unreachable=True)
    assert {fn.name for fn in everything.functions} == {link_name(name) for name in REACHED + NOT_REACHED}


def test_the_assembly_has_only_what_is_reached_and_what_that_uses():
    # (For one named target: how a symbol and a call are written differs between them.)
    asm = generate_asm(analyze(_parse(SOURCE)), target=Target('x86_64', 'linux'))
    for name in REACHED:
        assert f"\n{link_name(name)}:\n" in asm, name
    for name in NOT_REACHED:
        assert f"\n{link_name(name)}:\n" not in asm, name
    assert "call    abs\n" in asm and "labs" not in asm          # an extern only they call isn't referenced
    assert "never said" not in asm                                # ... nor is a string only they use


@GCC_SKIP
def test_the_program_runs_the_same():
    assert_program_stdout(SOURCE, "3\nLight.Green\n")


def test_an_unreachable_function_is_still_checked():
    with pytest.raises(SemanticError, match="undeclared variable 'missing'"):
        analyze(_parse("def int unused():\n    return missing\ndef int main():\n    return 0\n"))


def test_without_main_everything_is_kept():
    program = analyze(_parse("def int a():\n    return 1\ndef int b():\n    return a()\n"))
    assert [fn.name for fn in reachable_functions(program)] == ['a', 'b']
    assert len(build_ir_program(program).functions) == 2


def test_the_dumps(tmp_path):
    path = tmp_path / "p.ht"
    path.write_text(SOURCE)
    typed_dump, ir_dump = dump(str(path), 'typed'), dump(str(path), 'ir')
    assert "function unused(n#" in typed_dump and "function ping(n#" in typed_dump     # what was checked
    assert "function unused$" not in ir_dump and "function used$" in ir_dump           # what is compiled
    assert "function unused$" not in dump(str(path), 'optimized-ir')


def test_an_imported_module_costs_what_is_used_of_it(tmp_path):
    (tmp_path / "lib.ht").write_text(
        "def int wanted(int n):\n    return _inner(n)\n"
        "def int _inner(int n):\n    return n + 1\n"
        "def int unwanted(int n):\n    return n * 1000\n")
    (tmp_path / "main.ht").write_text("from 'lib' import wanted, unwanted\n\ndef int main():\n    return wanted(1)\n")
    asm = compile_to_asm(str(tmp_path / "main.ht"))
    assert "lib$wanted:" in asm and "lib$_inner:" in asm and "unwanted" not in asm
    # The standard library, likewise: a program that only prints through `os` has none of the rest.
    (tmp_path / "hello.ht").write_text(
        "from 'os' import write_stdout\n\ndef int main():\n    write_stdout('hi\\n')\n    return 0\n")
    hello = compile_to_asm(str(tmp_path / "hello.ht"))
    assert "os$write_stdout:" in hello and "os$read_file" not in hello and "isatty" not in hello
    assert "hornet_is_terminal" not in hello and "strings$" not in hello


def test_the_repository_s_own_programs_shrink():
    def instructions(path: str, keep: bool) -> int:
        entry, modules = discover_modules(str(ROOT / path))
        from backend import lower_to_asm
        from optimize.optimizer import optimize
        from target import default_target
        asm = lower_to_asm(optimize(build_ir_program(analyze(entry, modules), keep_unreachable=keep)), default_target())
        return sum(1 for line in asm.splitlines() if line.startswith('    ') and not line.lstrip().startswith('.'))
    for path in ('examples/wc.ht', 'tools/hfmt/main.ht'):
        assert instructions(path, keep=False) < 0.95 * instructions(path, keep=True), path
    # A program that uses all it has is unchanged.
    assert instructions('benchmarks/programs/branchy.ht', False) == instructions('benchmarks/programs/branchy.ht', True)


def test_the_command_line(tmp_path):
    path = tmp_path / "p.ht"
    path.write_text(SOURCE)
    asm = subprocess.run([sys.executable, str(ROOT / 'compile.py'), str(path)], capture_output=True, text=True).stdout
    assert "used$:" in asm and "unused$:" not in asm
