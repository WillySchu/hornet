"""`compile.py --dump STAGE` (dump.py): what each stage of the compiler produces, as text. One
small program through every stage, that a dump runs no further than its stage, and that every
program in the repository can be dumped at every stage."""

import dataclasses
import glob
import subprocess
import sys
from pathlib import Path

import pytest

import parser as syntax
from diagnostics import CompileError
from dump import STAGES, _TYPE_EXPRS, dump
from ir import ir
from ir.program_builder import build_ir_program
from lexer import lex
from modules import discover_modules
from optimize.optimizer import optimize
from semantic import analyze

ROOT = Path(__file__).resolve().parent.parent

SOURCE = (
    "def int twice(int n):\n"
    "    return n * 2\n"
    "\n"
    "def int main():\n"
    "    return twice(3) / 1\n"
)

EXPECTED = {
    'tokens': (
        "1:1 DEF\n1:5 INT\n1:9 IDENTIFIER twice\n1:14 OPEN_PAREN\n1:15 INT\n1:19 IDENTIFIER n\n"
        "1:20 CLOSE_PAREN\n1:21 COLON\n1:22 NEWLINE\n"
        "2:1 INDENT\n2:5 RETURN\n2:12 IDENTIFIER n\n2:14 STAR\n2:16 NUMBER 2\n2:17 NEWLINE\n"
        "3:1 NEWLINE\n"
        "4:1 DEDENT\n4:1 DEF\n4:5 INT\n4:9 IDENTIFIER main\n4:13 OPEN_PAREN\n4:14 CLOSE_PAREN\n4:15 COLON\n"
        "4:16 NEWLINE\n"
        "5:1 INDENT\n5:5 RETURN\n5:12 IDENTIFIER twice\n5:17 OPEN_PAREN\n5:18 NUMBER 3\n5:19 CLOSE_PAREN\n"
        "5:21 SLASH\n5:23 NUMBER 1\n5:24 NEWLINE\n"
        "6:1 DEDENT\n6:1 EOF\n"
    ),
    'tree': (
        "Program\n"
        "  functions:\n"
        "    Function name=twice return_type=int\n"
        "      params:\n"
        "        Param name=n type=int\n"
        "      body:\n"
        "        Return\n"
        "          value:\n"
        "            Binary op=MULTIPLY\n"
        "              left:\n"
        "                Variable name=n\n"
        "              right:\n"
        "                Constant value=2\n"
        "    Function name=main return_type=int\n"
        "      body:\n"
        "        Return\n"
        "          value:\n"
        "            Binary op=DIVIDE\n"
        "              left:\n"
        "                Call name=twice\n"
        "                  args:\n"
        "                    Constant value=3\n"
        "              right:\n"
        "                Constant value=1\n"
    ),
    'typed': (
        "function twice(n#0: int) -> int\n"
        "  Return\n"
        "    value:\n"
        "      Binary op=MULTIPLY : int\n"
        "        left:\n"
        "          Local symbol=n#0 : int\n"
        "        right:\n"
        "          IntLit value=2 : int\n"
        "\n"
        "function main() -> int\n"
        "  Return\n"
        "    value:\n"
        "      Binary op=DIVIDE : int\n"
        "        left:\n"
        "          Call name=twice kind=function : int\n"
        "            args:\n"
        "              IntLit value=3 : int\n"
        "        right:\n"
        "          IntLit value=1 : int\n"
    ),
    'ir': (
        "function twice$(t1: int) -> int\n"
        "  slot 0: 8 bytes (param:n#0)\n"
        "  t1 lives in slot 0\n"
        "  t2: int = multiply t1, 2\n"
        "  return t2\n"
        "\n"
        "function main() -> int\n"
        "  t3: int = call twice$(3)\n"
        "  t4: int = divide t3, 1 at p.ht:5:12\n"  # a division says where its panic would be reported
        "  return t4\n"
        "\n"
    ),
    'optimized-ir': (  # dividing by one is gone, and multiplying by two is a shift
        "function twice$(t1: int) -> int\n"
        "  slot 0: 8 bytes (param:n#0)\n"
        "  t1 lives in slot 0\n"
        "  t2: int = shift_left t1, 1\n"
        "  return t2\n"
        "\n"
        "function main() -> int\n"
        "  t4: int = call twice$(3)\n"
        "  return t4\n"
        "\n"
    ),
}


def _file(tmp_path, source: str, name: str = 'p.ht') -> str:
    (tmp_path / name).write_text(source)
    return str(tmp_path / name)


@pytest.mark.parametrize('stage', STAGES)
def test_one_program_at_each_stage(tmp_path, stage):
    assert dump(_file(tmp_path, SOURCE), stage) == EXPECTED[stage]


def test_the_command(tmp_path):
    path = _file(tmp_path, SOURCE)
    for stage in STAGES:
        command = [sys.executable, str(ROOT / 'compile.py'), path, '--dump', stage]
        assert subprocess.run(command, capture_output=True, text=True).stdout == EXPECTED[stage]
    out = tmp_path / 'p.ir'
    assert subprocess.run(command + ['-o', str(out)], capture_output=True, text=True).stdout == ''
    assert out.read_text() == EXPECTED['optimized-ir']
    for arguments in (['--dump', 'assembly'], ['--dump-typed']):  # (the flag `--dump typed` replaced)
        assert subprocess.run([sys.executable, str(ROOT / 'compile.py'), path, *arguments],
                              capture_output=True).returncode == 2


def test_a_dump_runs_no_further_than_its_stage(tmp_path):
    unparsable = _file(tmp_path, "def int main(:\n    return 1 +\n", 'a.ht')
    assert dump(unparsable, 'tokens').startswith("1:1 DEF\n1:5 INT\n1:9 IDENTIFIER main\n1:13 OPEN_PAREN\n1:14 COLON\n")
    with pytest.raises(CompileError, match="Expected a type"):
        dump(unparsable, 'tree')
    untyped = _file(tmp_path, "def int main():\n    return missing\n", 'b.ht')
    assert "Variable name=missing" in dump(untyped, 'tree')
    for stage in ('typed', 'ir', 'optimized-ir'):
        with pytest.raises(CompileError, match="undeclared variable 'missing'"):
            dump(untyped, stage)


def test_values_are_spelled_as_hornet_spells_them(tmp_path):
    path = _file(
        tmp_path,
        "type P struct:\n"
        "    int x\n"
        "def int main():\n"
        "    *P p = none\n"
        "    bool yes = true\n"
        "    byte b = \"q\"\n"
        "    str s = 'it\\'s\\ta \"line\"\\n'\n"
        "    if yes and p != none:\n"
        "        return p.x\n"
        "    return len(s)\n")
    tokens, tree, typed, ir_text = (dump(path, stage) for stage in ('tokens', 'tree', 'typed', 'ir'))
    quoted = "'it\\'s\\ta \"line\"\\n'"
    assert f"7:13 STRING {quoted}\n" in tokens and '6:14 BYTE "q"\n' in tokens
    assert f"StringLiteral value={quoted}\n" in tree and "BoolLiteral value=true\n" in tree
    assert f"StrLit value={quoted} : str\n" in typed and "BoolLit value=true : bool\n" in typed
    assert "FieldAccess name=x index=0 through_pointer=true : int\n" in typed and "True" not in typed + tree
    # (A check for none is a branch to a panic by the time the IR is built; its message says where.)
    assert f"= {quoted}\n" in ir_text and "= 'p.ht:9:16: panic: dereference of none'\n" in ir_text


def test_tokens_and_tree_are_of_one_file_and_the_rest_of_the_whole_program(tmp_path):
    _file(tmp_path, "def int helper():\n    return 7\n", 'lib.ht')
    path = _file(tmp_path, "from 'lib' import helper\n\ndef int main():\n    return helper()\n")
    assert "helper" in dump(path, 'tree') and "Function name=helper" not in dump(path, 'tree')
    assert "FromImportDecl path=lib" in dump(path, 'tree')
    assert "function lib$helper() -> int\n" in dump(path, 'typed')
    for stage in ('ir', 'optimized-ir'):
        assert "function lib$helper() -> int\n  return 7\n" in dump(path, stage)


SOURCES = sorted(path for pattern in ('stdlib/*.ht', 'tools/**/*.ht', 'examples/**/*.ht', 'benchmarks/programs/*.ht')
                 for path in glob.glob(str(ROOT / pattern), recursive=True))
PROGRAMS = [path for path in SOURCES if "def int main(" in Path(path).read_text(encoding='latin-1')]

# What no file in the repository happens to use.
EXTRA = (
    "import 'lib' as other\n"
    "type Count = int\n"
    "def other.Thing same(other.Thing t, Count c):\n"
    "    return t\n"
)


def _kinds(text: str) -> set:
    return {line.split()[0] for line in text.splitlines() if line.strip()}


def test_every_file_in_the_repository_has_tokens_and_a_tree(tmp_path):
    seen = _kinds(dump(_file(tmp_path, EXTRA), 'tree'))
    for path in SOURCES:
        assert dump(path, 'tokens').endswith(" EOF\n"), path
        tree = dump(path, 'tree')
        assert tree == dump(path, 'tree') and tree.startswith("Program\n"), path
        seen |= _kinds(tree)
    # Every kind of node was printed somewhere (a type is written inline, not as a node of its own).
    nodes = {c.__name__ for c in vars(syntax).values() if isinstance(c, type) and issubclass(c, syntax.Node)}
    assert nodes - {'Node'} - {c.__name__ for c in _TYPE_EXPRS} <= seen


@pytest.mark.parametrize('path', PROGRAMS, ids=lambda p: '-'.join(Path(p).parts[-2:]))
def test_every_program_in_the_repository_dumps_at_every_stage(path):
    for stage in ('typed', 'ir', 'optimized-ir'):
        text = dump(path, stage)
        assert text.startswith("function ") and text == dump(path, stage), stage


def test_every_kind_of_instruction_is_printed_by_some_program():
    seen = set()
    for path in PROGRAMS:
        entry, modules = discover_modules(path)
        built = build_ir_program(analyze(entry, modules))
        seen |= {type(instr) for fn in built.functions for instr in fn.body}
        seen |= {type(instr) for fn in optimize(built).functions for instr in fn.body}
    # (A null check is replaced by a branch as soon as its function is built: ir/null_checks.py.)
    instructions = {c for c in vars(ir).values() if dataclasses.is_dataclass(c) and c.__name__.startswith('IR')}
    assert instructions - {ir.IRFunction, ir.IRProgram, ir.IRConst, ir.IRNullCheck} <= seen
