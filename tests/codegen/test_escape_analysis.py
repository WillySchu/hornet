"""Tests for escape_analysis.py"""
    
import codegen.escape_analysis as ea
import parser
import semantic


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
            parser.Return(
                parser.Variable(name='sl'),
            ),
        ],
    )
    expected = set()
    res = ea.analyze_array_escapes(fn, [], {})
    assert expected == res


# TODO(will): I feel like this should escape?
def xtest_analyze_array_escapes_return_sliced_array():
    fn = parser.Function(
        name='main',
        return_type=None,
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
    print(fn.pretty())
    expected = {4327349456}
    res = ea.analyze_array_escapes(fn, [], {})
    print(res)
    print(list(res)[0])
    print(type(list(res)[0]))
    assert expected == res
