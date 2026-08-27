"""Tests for escape_analysis.py"""
import ast
import tempfile
from pathlib import Path
    
import parser
import semantic
import codegen.escape_analysis as ea
from lexer import lex


def parse_and_analyze(source: str) -> parser.Program:
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / 'program.ht'
        src_path.write_text(source)
        tokens = lex(str(src_path))
        ast = parser.Parser(tokens).parse_program()
        semantic.analyze(ast)
        return ast


def parse_expression(source: str) -> ast.Program:
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / 'program.ht'
        src_path.write_text(source)
        tokens = lex(str(src_path))
        ast = parser.Parser(tokens).parse_expression()
        return ast


def _analyze(fn: parser.Function):
    program = parser.Program(functions=[fn], structs=[])
    semantic.analyze(program)


def test_is_heap_allocated_int():
    t = semantic.Type(kind=semantic.TypeKind.INT)
    assert not ea.is_heap_allocated(t, {})


def test_is_heap_allocated_str():
    t = semantic.Type(kind=semantic.TypeKind.STR)
    assert not ea.is_heap_allocated(t, {})


def test_is_heap_allocated_bool():
    t = semantic.Type(kind=semantic.TypeKind.BOOL)
    assert not ea.is_heap_allocated(t, {})


def test_is_heap_allocated_none():
    t = semantic.Type(kind=semantic.TypeKind.NONE)
    assert not ea.is_heap_allocated(t, {})


def test_is_heap_allocated_array_int_stack():
    size = 4096  # int = 4, 4 * 4096 = 16384
    t = semantic.Type(kind=semantic.TypeKind.ARRAY, element_type=semantic.Type(kind=semantic.TypeKind.INT), size=size)
    assert not ea.is_heap_allocated(t, {})


def test_is_heap_allocated_array_int_heap():
    size = 4097  # int = 4, 4 * 4096 = 16388
    t = semantic.Type(kind=semantic.TypeKind.ARRAY, element_type=semantic.Type(kind=semantic.TypeKind.INT), size=size)
    assert ea.is_heap_allocated(t, {})


def test_is_heap_allocated_array_str_stack():
    size = 2048  # str = 8, 4 * 4096 = 16384
    t = semantic.Type(kind=semantic.TypeKind.ARRAY, element_type=semantic.Type(kind=semantic.TypeKind.STR), size=size)
    assert not ea.is_heap_allocated(t, {})


def test_is_heap_allocated_array_str_heap():
    size = 2049  # str = 9, 4 * 4096 = 16388
    t = semantic.Type(kind=semantic.TypeKind.ARRAY, element_type=semantic.Type(kind=semantic.TypeKind.STR), size=size)
    assert ea.is_heap_allocated(t, {})


# TODO(will): Test Structs.


def test_unwrap_slices():
    tcs = [
        {
            'expr': parser.Variable(name='arr'),
            'expected': parser.Variable(name='arr'),
        },
        {
            'expr': parser.Slice(array=parser.Variable(name='arr')),
            'expected': parser.Variable(name='arr'),
        },
        # TODO(will): Expand this testing.
    ]

    for tc in tcs:
        assert tc['expected'] == ea._unwrap_slices(tc['expr'])


def test_root_variable_name():
    tcs = [
        {
            'expr': parser.VarDecl(name='name', var_type='str'),
            'name': None
        },
        {
            'expr': parser.Variable(name='sl'),
            'name': 'sl'
        },
        {
            'expr': parser.Index(array=parser.Variable(name='sl'), index=parser.Constant(1)),
            'name': 'sl'
        },
        {
            'expr': parser.Slice(array=parser.Variable(name='arr'), high=parser.Constant(3)),
            'name': 'arr'
        },
        {
            'expr': parser.Field(base=parser.Variable(name='st'), name='x'),
            'name': 'st'
        },
        {
            'expr': parser.Index(
                array=parser.Slice(
                    array=parser.Field(base=parser.Variable(name='st2'), name='x')
                ),
                index=parser.Constant(1),
            ),
            'name': 'st2'
        },
        {
            'expr': parser.Index(
                array=parser.Slice(
                    array=parser.Field(base=parser.Call(name='st2'), name='x')
                ),
                index=parser.Constant(1),
            ),
            'name': None
        },
    ]
    for tc in tcs:
        assert tc['name'] == ea.root_variable_name(tc['expr'])


def test_analyze_array_escapes_empty():
    fn = parser.Function(name='main', return_type=None)
    expected = set()
    res = ea.analyze_array_escapes(fn, [], {})
    assert expected == res


def test_analyze_array_escapes_fn_on_uninitialized_slice():
    fn = parser.Function(
        name='main',
        return_type=None,
        body=[
            parser.VarDecl(
                name='sl',
                var_type=parser.SliceTypeExpr(
                    element_type='int',
                ),
            ),
            parser.Call(
                name='fn',
                args=[parser.Variable(name='sl')],
            ),
        ],
    )
    expected = set()
    res = ea.analyze_array_escapes(fn, [], {})
    assert expected == res


def test_analyze_array_escapes_fn_on_uninitialized_slice():
    fn = parser.Function(
        name='main',
        return_type=None,
        body=[
            parser.VarDecl(
                name='sl',
                var_type=parser.SliceTypeExpr(
                    element_type='int',
                ),
            ),
            parser.Call(
                name='fn',
                args=[parser.Variable(name='sl')],
            ),
        ],
    )
    expected = set()
    res = ea.analyze_array_escapes(fn, [], {})
    assert expected == res


# TODO(will): I feel like this should escape?
def test_analyze_array_escapes_fn_on_initialized_slice():
    fn = parser.Function(
        name='main',
        return_type=None,
        body=[
            parser.VarDecl(
                name='sl',
                var_type=parser.SliceTypeExpr(
                    element_type='int',
                ),
                init=parser.ArrayLiteral(
                    elements=[parser.Constant(1), parser.Constant(2), parser.Constant(3)]
                ),
            ),
            parser.Call(
                name='fn',
                args=[parser.Variable(name='sl')],
            ),
        ],
    )
    expected = set()
    res = ea.analyze_array_escapes(fn, [], {})
    assert expected == res


# TODO(will): I feel like this should escape?
def test_analyze_array_escapes_return_initialized_slice():
    fn = parser.Function(
        name='main',
        return_type=semantic.SliceTypeExpr(element_type='int'),
        body=[
            parser.VarDecl(
                name='sl',
                var_type=parser.SliceTypeExpr(
                    element_type='int',
                ),
                init=parser.ArrayLiteral(
                    elements=[parser.Constant(1), parser.Constant(2), parser.Constant(3)]
                ),
            ),
            parser.Return(
                parser.Variable(name='sl'),
            ),
        ],
    )
    _analyze(fn)
    expected = set()
    res = ea.analyze_array_escapes(fn, [], {})
    assert expected == res


def test_analyze_array_escapes_return_sliced_array():
    fn = parser.Function(
        name='main',
        return_type=semantic.SliceTypeExpr(element_type='int'),
        body=[
            parser.VarDecl(
                name='arr',
                var_type=parser.ArrayTypeExpr(
                    element_type='int',
                    size=5,
                ),
                init=parser.ArrayLiteral(
                    elements=[parser.Constant(1), parser.Constant(2), parser.Constant(3), parser.Constant(4), parser.Constant(5)]
                ),
            ),
            parser.VarDecl(
                name='sl',
                var_type=parser.SliceTypeExpr(
                    element_type='int',
                ),
                init=parser.Slice(
                    array=parser.Variable(name='arr'),
                    low=parser.Index(array=parser.Variable(name='arr'), index=parser.Constant(value=1)),
                    high=parser.Index(array=parser.Variable(name='arr'), index=parser.Constant(value=3)),
                )
            ),
            parser.Return(
                parser.Variable(name='sl'),
            ),
        ],
    )
    _analyze(fn)
    res = ea.analyze_array_escapes(fn, [], {})
    assert len(res) == 1


# TODO(will): I feel like this should escape?
def test_test():
    source = '''
def main():
    []int ints = [1, 2, 3]
    print_ints(ints)


def print_ints([]int ints):
    print(ints)
'''
    ast = parse_and_analyze(source)
    main_res = ea.analyze_array_escapes(ast.functions[0], [], {})
    assert main_res == set()
    print_ints_res = ea.analyze_array_escapes(ast.functions[1], [semantic.Type(kind=semantic.TypeKind.SLICE)], {})
    assert print_ints_res == set()


def test_test2():
    source = (
        "def []int sliceints([5]int arr):\n"
        "    []int a = arr[:]\n"
        "    return a\n"
        "\n"
        "def int helper(int x):\n"
        "    int a = x + 1\n"
        "    int b = a + 1\n"
        "    return a + b\n"
        "\n"
        "def int main():\n"
        "    [5]int arr = [1, 2, 3, 4, 5]\n"
        "    []int sl = sliceints(arr)\n"
        "    int junk = helper(1)\n"
        "    junk = helper(2)\n"
        "    junk = helper(3)\n"
        "    print(sl)\n"
        "    return 0\n"
    )

    ast = parse_and_analyze(source)
    sliceints_res = ea.analyze_array_escapes(ast.functions[0], [semantic.Type(kind=semantic.TypeKind.ARRAY, element_type='int')], {})
    assert len(sliceints_res) == 1
    assert set() == ea.analyze_array_escapes(ast.functions[1], [semantic.Type(kind=semantic.TypeKind.INT)], {})
    assert set() == ea.analyze_array_escapes(ast.functions[2], [], {})


def test_escape_analyzer_declare():
    fn = parser.Function(name='main', return_type=None)

    tcs = [
        {
            'declarations': [],
            'expected': {
                'scopes': [{}],
                'decl_types': {},
                'array_decls': set(),
                'slice_decls': set(),
                'direct_backing': {},
                'slice_deps': {},
            }
        },
        {
            'declarations': [('var1', 1, semantic.Type(kind=semantic.TypeKind.INT))],
            'expected': {
                'scopes': [{'var1': 1}],
                'decl_types': {1: semantic.Type(kind=semantic.TypeKind.INT)},
                'array_decls': set(),
                'slice_decls': set(),
                'direct_backing': {},
                'slice_deps': {},
            }
        },
        # TODO(will): This is incredible simple, but probably throw a few more tests here.
    ]

    for tc in tcs:
        analyzer = ea.EscapeAnalyzer(fn, [], {})
        for decl in tc['declarations']:
            analyzer.declare(*decl)
        assert tc['expected']['scopes'] == analyzer.scopes
        assert tc['expected']['decl_types'] == analyzer.decl_types
        assert tc['expected']['array_decls'] == analyzer.array_decls
        assert tc['expected']['slice_decls'] == analyzer.slice_decls
        assert tc['expected']['direct_backing'] == analyzer.direct_backing
        assert tc['expected']['slice_deps'] == analyzer.slice_deps


def test_escape_analyzer_resolve():
    fn = parser.Function(name='main', return_type=None)

    tcs = [
        {
            'scopes': [],
            'expected': None,
            'name': 'var1',
        },
        {
            'scopes': [{}],
            'expected': None,
            'name': 'var1',
        },
        {
            'scopes': [{'var2': 1}],
            'expected': None,
            'name': 'var1',
        },
        {
            'scopes': [{'var1': 1}],
            'expected': 1,
            'name': 'var1',
        },
        {
            'scopes': [{'var1': 1}, {}],
            'expected': 1,
            'name': 'var1',
        },
        {
            'scopes': [{'var1': 1}, {'var2': 2}],
            'expected': 1,
            'name': 'var1',
        },
        {
            'scopes': [{'var1': 1}, {'var2': 2, 'var1': 3}],
            'expected': 3,
            'name': 'var1',
        },
    ]

    for tc in tcs:
        analyzer = ea.EscapeAnalyzer(fn, [], {})
        analyzer.scopes = tc['scopes']
        assert tc['expected'] == analyzer.resolve(tc['name'])


# TODO(will): Hand roll some functions for testing resolve to ensure that scopes are built correctly.


def test_escape_analyzer_slot_node_id():
    fn = parser.Function(name='main', return_type=None)
    analyzer = ea.EscapeAnalyzer(fn, [], {})

    tcs = [
        {
            'expected': -1,
            'id': 1,
            'slot': '1',
        },
        {
            'expected': -1,
            'id': 1,
            'slot': '1',
        },
        {
            'expected': -2,
            'id': 2,
            'slot': '1',
        },
        {
            'expected': -3,
            'id': 1,
            'slot': '2',
        },
        {
            'expected': -4,
            'id': 2,
            'slot': '2',
        },
        {
            'expected': -2,
            'id': 2,
            'slot': '1',
        },
    ]

    for tc in tcs:
        assert tc['expected'] == analyzer.slot_node_id(tc['id'], tc['slot'])


def test_escape_analyzer_contains_slice():
    fn = parser.Function(name='main', return_type=None)

    tcs = [
        {
            'type': semantic.Type(kind=semantic.TypeKind.SLICE),
            'structs': {},
            'res': True,
        },
        # TODO(will): Finish these tests.
    ]

    for tc in tcs:
        analyzer = ea.EscapeAnalyzer(fn, [], tc['structs'])
        assert tc['res'] == analyzer._contains_slice(tc['type'])


def test_escape_analyzer_whole_value_node_of_empty():
    fn = parser.Function(name='main', return_type=None)
    analyzer = ea.EscapeAnalyzer(fn, [], {})
    assert analyzer.whole_value_node_of('var1') is None


def test_escape_analyzer_whole_value_node_of_param_not_slice():
    fn = parser.Function(name='main', params=[parser.Param(name='x', type='int')], return_type=None)
    analyzer = ea.EscapeAnalyzer(fn, [semantic.Type(kind=semantic.TypeKind.INT)], {})
    assert analyzer.whole_value_node_of('x') is None


def test_escape_analyzer_whole_value_node_of_param_slice():
    fn = parser.Function(
        name='main',
        params=[parser.Param(name='x', type=parser.SliceTypeExpr(element_type='int'))],
        return_type=None,
    )
    analyzer = ea.EscapeAnalyzer(fn, [semantic.Type(kind=semantic.TypeKind.SLICE)], {})
    # TODO(will): ids aren't deterministic, but would love a better way of testing this.
    assert analyzer.whole_value_node_of('x') is not None


def test_escape_analyzer_whole_value_node_of_variable_no_slice():
    fn = parser.Function(
        name='main',
        params=[],
        body=[
            parser.VarDecl(
                name='x',
                var_type='int',
                init=parser.Constant(value=1),
            ),
            parser.Return()
        ],
        return_type=None,
    )
    analyzer = ea.EscapeAnalyzer(fn, [], {})
    assert analyzer.whole_value_node_of('x') is None
    analyzer.walk_statements([fn])
    assert analyzer.whole_value_node_of('x') is None


def test_escape_analyzer_whole_value_node_of_variable_slice():
    fn = parser.Function(
        name='main',
        params=[],
        body=[
            parser.VarDecl(
                name='sl',
                var_type=parser.SliceTypeExpr(element_type='int'),
                init=parser.ArrayLiteral(
                    elements=[
                        parser.Constant(value=1),
                        parser.Constant(value=2),
                        parser.Constant(value=3),
                    ],
                ),
            ),
            parser.Return(),
        ],
        return_type=None,
    )
    analyzer = ea.EscapeAnalyzer(fn, [], {})
    assert analyzer.whole_value_node_of('sl') is None
    analyzer.walk_statements(fn.body)
    assert analyzer.whole_value_node_of('sl') is not None


def test_escape_analyzer_whole_value_node_of_variable_struct_no_slice():
    fn = parser.Function(
        name='main',
        params=[],
        body=[
            parser.VarDecl(
                name='a',
                var_type='A',
            ),
            parser.FieldAssign(base=parser.Variable('a'), name='x', value=parser.Constant(value=1)),
            parser.FieldAssign(base=parser.Variable('a'), name='y', value=parser.StringLiteral(value='asdf')),
            parser.Return(),
        ],
        return_type=None,
    )
    structs = {'A': semantic.StructInfo(
        name='A',
        fields={
            'x': semantic.Type(kind=semantic.TypeKind.INT),
            'y': semantic.Type(kind=semantic.TypeKind.STR),
        },
    )}
    analyzer = ea.EscapeAnalyzer(fn, [], structs)
    analyzer.walk_statements(fn.body)
    assert analyzer.whole_value_node_of('a') is None


def test_escape_analyzer_whole_value_node_of_variable_struct_slice():
    source = """
struct A:
    int x
    str y
    []int sl

def helper():
    A a
    a.x = 1
    a.y = 'asdf'
    return
"""
    ast = parse_and_analyze(source)
    print(ast)
    fn = parser.Function(
        name='main',
        params=[],
        body=[
            parser.VarDecl(
                name='a',
                var_type='A',
            ),
            parser.FieldAssign(base=parser.Variable('a'), name='x', value=parser.Constant(value=1)),
            parser.FieldAssign(base=parser.Variable('a'), name='y', value=parser.StringLiteral(value='asdf')),
            parser.Return(),
        ],
        return_type=None,
    )
    structs = {'A': semantic.StructInfo(
        name='A',
        fields={
            'x': semantic.Type(kind=semantic.TypeKind.INT),
            'y': semantic.Type(kind=semantic.TypeKind.STR),
            'sl': semantic.Type(kind=semantic.TypeKind.SLICE),
        },
    )}
    analyzer = ea.EscapeAnalyzer(fn, [], structs)
    analyzer.walk_statements(fn.body)
    assert analyzer.whole_value_node_of('a') == -1


def test_escape_analyzer_field_slot_of():
    # TODO(will): Finish.
    ...


def test_escape_analyzer_contribution():
    # TODO(will): Finish.
    ...


def test_escape_analyzer_scan_expr_for_escaping_calls():
    # TODO(will): Finish.
    ...


def test_escape_analyzer_walk_statements():
    # TODO(will): Finsh.
    ...
