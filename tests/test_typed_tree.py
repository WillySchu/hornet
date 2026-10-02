"""The typed tree (typed_ast.py) that semantic.analyze() builds."""

import glob
import subprocess
import sys
from pathlib import Path

import pytest

import typed_ast as t
from compile import typed_tree
from modules import discover_modules
from semantic import analyze
from tests.test_compiler import _parse

ROOT = Path(__file__).resolve().parents[1]

SOURCE = """\
type Num struct:
    int v
type Bin struct:
    str op
    *Expr left
    *Expr right
type Expr is Num | Bin | none
type Opts struct:
    int n
    []int xs
    dict[str]int d
    Expr e
def int eval(*Expr p):
    match *p as e:
        is Num:
            return e.v
        is Bin:
            return eval(e.left) + eval(e.right)
        is none:
            return 0
    return 0
def int main():
    Expr e = Bin('+', &Num(1), &Num(2))
    Opts o = Opts(n=3)
    []int s = [1, 2]
    []int empty = []
    dict[str]int d
    d['k'] += 1
    o.xs = append(o.xs, 4)
    int8 small = int8(5)
    str c = str(byte(65)) + 'x'
    []byte bs = bytes(c)
    if 2 in s and 'k' in d and c[0] == byte(65) and c != 'Ax':
        print(c)
    if e is Bin:
        print(e.op)
    if o.e == none:
        print(len(bs))
    for k, v in d:
        print(v)
    for x in s[0:1]:
        print(x)
    print(eval(&e) + int(small) + len(empty))
    return 0
"""


def _typed(source: str) -> t.Program:
    ast = _parse(source)
    return analyze(ast)


def _nodes(program: t.Program) -> set:
    seen = set()

    def visit(node):
        seen.add(type(node).__name__)
        for value in vars(node).values():
            for item in value if isinstance(value, tuple) else (value,):
                for x in item if isinstance(item, tuple) else (item,):
                    if isinstance(x, (t.Expr, t.Stmt)):
                        visit(x)
    for fn in program.functions:
        for stmt in fn.body:
            visit(stmt)
    return seen


def test_implicit_operations_are_explicit():
    nodes = _nodes(_typed(SOURCE))
    assert {'WidenToSum', 'BoxVariant', 'SliceLiteral', 'EmptySlice', 'NewEmptyDict', 'ZeroValue', 'StructLiteral',
            'Payload', 'TagTest', 'Match', 'Deref', 'IntCast', 'StrFromByte', 'StrConcat', 'BytesFromStr',
            'DictContains', 'ElementContains', 'StrIndex', 'StrCompare', 'DictLookup', 'CompoundAssign', 'Append',
            'SliceOf', 'ForIn', 'Len', 'Print', 'FieldAccess', 'AddressOf'} <= nodes


def test_omitted_fields_are_zero_values_in_declaration_order():
    program = _typed(SOURCE)
    main = next(f for f in program.functions if f.name == 'main')
    opts = main.body[1].init
    assert isinstance(opts, t.StructLiteral)
    assert [type(f).__name__ for f in opts.fields] == ['IntLit', 'EmptySlice', 'NewEmptyDict', 'ZeroValue']


def test_a_variable_narrowed_by_is_reads_the_payload():
    program = _typed(SOURCE)
    text = t.dump(program)
    assert "TagTest variant=Bin : bool" in text
    assert "Payload : Bin" in text


def test_dump_is_deterministic():
    assert t.dump(_typed(SOURCE)) == t.dump(_typed(SOURCE))


def _repository_programs():
    paths = sorted(glob.glob(str(ROOT / 'benchmarks' / 'programs' / '*.ht'))) + sorted(glob.glob(str(ROOT / 'examples' / '*.ht')))
    return paths + [str(ROOT / 'tools' / 'hfmt' / 'main.ht')]


@pytest.mark.parametrize('path', _repository_programs(), ids=lambda p: Path(p).stem)
def test_repository_programs_have_typed_trees(path):
    entry, modules = discover_modules(path)
    assert t.dump(analyze(entry, modules))


def test_every_node_kind_occurs_in_the_repository_and_this_file():
    defined = {name for name, cls in vars(t).items()
               if isinstance(cls, type) and issubclass(cls, (t.Expr, t.Stmt)) and cls not in (t.Expr, t.Stmt)}
    used = _nodes(_typed(SOURCE))
    for path in _repository_programs():
        entry, modules = discover_modules(path)
        used |= _nodes(analyze(entry, modules))
    used |= _nodes(_typed("def int main():\n    [2]int a = [1, 2]\n    print(a[1])\n    int y = 1\n    *int p = &y\n"
                          "    *p = 2\n    while y < 3:\n        y += 1\n        if y == 2:\n            continue\n"
                          "        break\n    for int i = 0; i < 2; i += 1:\n        y = -y\n    print(true)\n"
                          "    dict[int]int d\n    d[1] = 1\n    del(d, 1)\n    *int q = none\n    print(q == none)\n"
                          "    print(dict[int]int{1: 2})\n    print(str(bytes('a')))\n    return 0\n"))
    assert defined - used == set(), "node kinds no test produces"


def test_dump_typed_command(tmp_path):
    src = tmp_path / 'p.ht'
    src.write_text("def int main():\n    return 0\n")
    r = subprocess.run([sys.executable, str(ROOT / 'compile.py'), str(src), '--dump-typed'], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.startswith("function main() -> int\n  Return\n")
    assert r.stdout == typed_tree(str(src))


FRONT_END = {'parser', 'semantic', 'lexer', 'modules', 'scopes'}
LATER_STAGES = ['ir', 'optimize', 'backend', 'escape_analysis.py']


def test_later_stages_never_import_the_front_end():
    """The typed program is the only thing that crosses from the front end: ir/, optimize/,
    backend/, and escape analysis use typed_ast, typesys, and symbols, never the parser's tree."""
    import ast as python_ast
    offenders = []
    for root in LATER_STAGES:
        path = ROOT / root
        for source in sorted([path] if path.is_file() else path.rglob('*.py')):
            for node in python_ast.walk(python_ast.parse(source.read_text())):
                names = []
                if isinstance(node, python_ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, python_ast.ImportFrom) and node.module:
                    names = [node.module]
                offenders += [f"{source.relative_to(ROOT)}: {name}" for name in names
                              if name.split('.')[0] in FRONT_END]
    assert offenders == []


def test_intrinsics_are_their_own_nodes():
    program = _typed("intrinsic *byte _raw_ptr(str s)\nintrinsic int _raw_len(str s)\n"
                     "intrinsic str _from_raw_parts(*byte p, int n)\n"
                     "def int main():\n    str s = 'hi'\n    print(_from_raw_parts(_raw_ptr(s), _raw_len(s)))\n    return 0\n")
    assert {'StrRawPtr', 'StrRawLen', 'StrFromRawParts'} <= _nodes(program)
