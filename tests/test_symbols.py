"""symbols.py, node numbers, and the rule that the compiler never keys on object identity."""

import io
import tokenize
from pathlib import Path

import parser
from ir.program_builder import build_ir_program
from tests.test_compiler import _parse, analyze

ROOT = Path(__file__).resolve().parents[1]


def test_the_compiler_never_uses_object_identity():
    """Per-node and per-declaration facts are keyed on Node.nid and Symbol.id: object identity
    breaks silently when a pass copies nodes, varies from run to run, and has no equivalent in Hornet."""
    offenders = []
    for path in sorted(ROOT.rglob('*.py')):
        rel = path.relative_to(ROOT)
        if rel.parts[0] in ('tests', 'benchmarks', '.venv', 'htmlcov') or rel.name == 'conftest.py':
            continue
        tokens = list(tokenize.generate_tokens(io.StringIO(path.read_text()).readline))
        for k in range(1, len(tokens) - 1):
            t = tokens[k]
            if (t.type == tokenize.NAME and t.string == 'id' and tokens[k + 1].string == '('
                    and tokens[k - 1].string not in ('.', 'def')):
                offenders.append(f"{rel}:{t.start[0]}")
    assert offenders == []


SOURCE = (
    "type A struct:\n    int v\ntype B struct:\n    int w\ntype AB is A | B\n"
    "def int f(int n, AB ab):\n"
    "    int x = n\n"
    "    for i, y in [1, 2]:\n"
    "        int x = y\n"
    "        print(x + i)\n"
    "    if ab is A:\n"
    "        x = ab.v\n"
    "    match ab as m:\n"
    "        is A:\n"
    "            x = m.v\n"
    "        is B:\n"
    "            x = m.w\n"
    "    return x\n"
    "def int main():\n    return 0\n"
)


def test_every_declaration_gets_its_own_symbol():
    import typed_ast as t
    program = _parse(SOURCE)
    typed_program = analyze(program)
    fn = typed_program.functions[0]
    kinds = [(str(s), s.kind) for s in typed_program.symbols.symbols]
    assert kinds[:6] == [('n#0', 'param'), ('ab#1', 'param'), ('x#2', 'local'), ('i#3', 'binding'),
                         ('y#4', 'binding'), ('x#5', 'local')]
    assert ('m#6', 'narrowing') in kinds
    assert fn.params[0] is typed_program.symbols[0]
    outer_x, loop = fn.body[0], fn.body[1]
    inner_x = loop.body[0]
    assert isinstance(outer_x, t.Declare) and isinstance(loop, t.ForIn)
    assert outer_x.symbol is not inner_x.symbol  # shadowing is a different variable
    assert loop.bindings[1].name == 'y' and inner_x.init.symbol is loop.bindings[1]
    assert fn.body[-1].value.symbol is outer_x.symbol  # `return x` refers to the outer x


def test_node_numbers_are_unique_and_follow_creation_order():
    program = _parse(SOURCE)
    seen = []
    stack = [program]
    while stack:
        node = stack.pop()
        seen.append(node.nid)
        for f in parser.fields(node):
            value = getattr(node, f.name)
            for v in value if isinstance(value, list) else [value]:
                if isinstance(v, parser.Node):
                    stack.append(v)
    assert len(seen) == len(set(seen)) and all(n >= 0 for n in seen)
    a, b = parser.Constant(1), parser.Constant(2)
    assert b.nid == a.nid + 1


def test_slot_labels_name_their_symbol():
    program = _parse(SOURCE)
    program = analyze(program)
    labels = list(next(f for f in build_ir_program(program).functions if f.name == 'f').slot_labels.values())
    assert 'param:n#0' in labels and 'local:x#2' in labels and 'for_in:y#4' in labels
