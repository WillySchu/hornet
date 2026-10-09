"""`defer CALL`: the call's receiver and arguments are evaluated where the statement is; the call is
made when the block the statement is in is left, by reaching its end or by `return`, `break`, or
`continue`. A block's deferred calls run last deferred first. A panic runs none."""

import pytest

from lexer import Lexer, TokenType
from parser import Call, Defer, TokenStream, statements
from tests.test_compiler import (
    GCC_SKIP, _parse, analyze, assert_program_panics, assert_program_semantic_error, assert_program_stdout)
from typed_ast import dump

HELPERS = (
    "def note(str what):\n"
    "    print(what)\n"
    "def int loud(str what, int n):\n"
    "    print(what)\n"
    "    return n\n"
)


def _lines(*lines: str) -> str:
    return "".join(line + "\n" for line in lines)


@GCC_SKIP
def test_a_deferred_call_runs_when_its_block_is_left():
    assert_program_stdout(
        HELPERS +
        "def int nested(int n):\n"
        "    defer note('function ends')\n"
        "    if n > 0:\n"
        "        defer note('if body ends')\n"
        "        if n > 1:\n"
        "            return loud('value computed', n * 2)\n"     # from two blocks down: both run, inner first
        "        print('inside')\n"
        "    print('after the if')\n"
        "    return n\n"
        "def int main():\n"
        "    print(nested(0))\n    print(nested(1))\n    print(nested(2))\n"
        "    return 0\n",
        _lines("after the if", "function ends", "0",
               "inside", "if body ends", "after the if", "function ends", "1",
               "value computed", "if body ends", "function ends", "4"),
    )


@GCC_SKIP
def test_the_last_deferred_runs_first_and_operands_are_what_they_were():
    assert_program_stdout(
        HELPERS +
        "type Counter struct:\n"
        "    int n\n"
        "    def bump(*c):\n"
        "        c.n += 1\n"
        "def int main():\n"
        "    defer print('first deferred, last run')\n"
        "    int x = 1\n"
        "    defer print(x)\n"                                    # 1: what x was here
        "    x = 2\n"
        "    defer print(loud('evaluated at the defer', 9))\n"    # loud runs now; print at the end
        "    Counter c = Counter(0)\n"
        "    if true:\n"
        "        defer c.bump()\n"                                # a pointer receiver: c itself, when it runs
        "        defer c.bump()\n"
        "        print(c.n)\n"
        "    print(c.n)\n"
        "    return 0\n",
        _lines("evaluated at the defer", "0", "2", "9", "1", "first deferred, last run"),
    )


@GCC_SKIP
def test_loops_and_match_arms():
    assert_program_stdout(
        HELPERS +
        "type Small is int | str\n"
        "def int search([]int xs, int wanted):\n"
        "    defer note('search done')\n"
        "    for x in xs:\n"
        "        defer note(format('looked at {}', x))\n"
        "        if x == wanted:\n"
        "            return x\n"                                  # out of the loop's body and the function's
        "    return -1\n"
        "def kind(Small s):\n"
        "    match s:\n"
        "        is int:\n"
        "            defer note('arm ends')\n"
        "            print('an int')\n"
        "        is str:\n"
        "            print('a str')\n"
        "def int main():\n"
        "    for int i = 0; i < 3; i += 1:\n"
        "        defer note(format('iteration {} ends', i))\n"    # each time round, however it ends
        "        if i == 1:\n            continue\n"
        "        if i == 2:\n            break\n"
        "        print('body')\n"
        "    print(search([]int[4, 5, 6], 5))\n"
        "    kind(1)\n    kind('one')\n"
        "    dict[str]int ages = dict[str]int{'a': 1, 'b': 2}\n"
        "    for name, age in ages:\n"
        "        defer note('entry ' + name)\n"
        "        if age == 1:\n            continue\n"
        "    int n = 0\n"
        "    while n < 2:\n"
        "        defer note('while body ends')\n"
        "        n += 1\n"
        "    return 0\n",
        _lines("body", "iteration 0 ends", "iteration 1 ends", "iteration 2 ends",
               "looked at 4", "looked at 5", "search done", "5",
               "an int", "arm ends", "a str", "entry a", "entry b", "while body ends", "while body ends"),
    )


@GCC_SKIP
def test_what_is_known_of_a_variable_at_the_defer_is_what_the_call_is_of():
    assert_program_stdout(
        "type Error struct:\n"
        "    str message\n"
        "type Conn struct:\n"
        "    int id\n"
        "    def close(c):\n"
        "        print(format('closed {}', c.id))\n"
        "type ConnResult is Conn | Error\n"
        "type Point struct:\n"
        "    int x\n"
        "    int y\n"
        "def ConnResult connect(int id):\n"
        "    if id < 0:\n        return Error('refused')\n"
        "    return Conn(id)\n"
        "def str use(int id):\n"
        "    ConnResult conn = connect(id)\n"
        "    if conn is Error:\n        return conn.message\n"    # before the defer: nothing to close
        "    defer conn.close()\n"                                # a Conn here, by elimination
        "    conn = Error('replaced')\n"                          # the deferred call is of what it was
        "    return format('used {}', id)\n"                      # a str, built before the close
        "def Point corner(int n):\n"
        "    defer print('corner done')\n"
        "    return Point(n, n + 1)\n"                            # a struct
        "def ConnResult again(int id):\n"
        "    defer print('again done')\n"
        "    return connect(id)\n"                                # a sum
        "def int main():\n"
        "    print(use(7))\n    print(use(-1))\n    print(corner(3))\n    print(again(2))\n"
        "    return 0\n",
        _lines("closed 7", "used 7", "refused", "corner done", "Point(x: 3, y: 4)", "again done", "Conn(id: 2)"),
    )


@GCC_SKIP
def test_a_panic_runs_none():
    assert_program_panics(
        "def int main():\n"
        "    defer print('deferred')\n"
        "    print('before')\n"
        "    panic('stop')\n",
        "stop",
        expected_stdout="before\n",
    )


@GCC_SKIP
def test_a_function_only_ever_deferred_is_still_built():
    assert_program_stdout(
        "def only_deferred():\n    print('ran')\n"
        "def helper():\n    defer only_deferred()\n"
        "def int main():\n    helper()\n    return 0\n",
        "ran\n")


def test_how_it_parses_and_what_it_becomes():
    stream = TokenStream(Lexer("defer conn.close(1)\n", 'f.ht').tokenize())
    assert stream.current().type == TokenType.DEFER
    node = statements.parse_statement(stream)
    assert isinstance(node, Defer) and isinstance(node.call, Call) and (node.line, node.col) == (1, 1)
    tree = dump(analyze(_parse(
        "def int f(int a, int b):\n    return a + b\n"
        "def int main():\n    int x = 1\n    defer f(x, 2)\n    return 0\n")))
    # The operand that isn't a literal is put in a variable of its own; the call is of that.
    assert ("  Declare symbol=deferred#3\n"
            "    init:\n"
            "      Local symbol=x#2 : int\n"
            "  Defer\n"
            "    call:\n"
            "      Call name=f kind=function : int\n"
            "        args:\n"
            "          Local symbol=deferred#3 : int\n"
            "          IntLit value=2 : int\n") in tree


@pytest.mark.parametrize("statement,message", [
    ("defer 5", "'defer' takes a call, as in `defer file.close..`: what is deferred is calling it"),
    ("int x = 1\n    defer x", "'defer' takes a call"),
    ("int x = 1\n    defer x = 2", "'defer' takes a call"),
    ("defer", "Expected an expression, got end of line"),
    ("defer f(1) f(2)", "Expected the end of the line after this statement"),
    ("int defer = 1", "Expected a variable name"),
])
def test_what_does_not_parse(statement, message):
    from parser import ParseError
    with pytest.raises(ParseError, match=message):
        _parse(f"def int f(int n):\n    return n\ndef int main():\n    {statement}\n    return 0\n")


@pytest.mark.parametrize("statement,message", [
    ("defer Point(1)", r"'defer' takes a call to a function or a method; `Point\(...\)` makes a value"),
    ("defer stop()", "'stop' never returns, so it can't be deferred: leaving the block would end the program"),
    ("defer panic('x')", "'panic' never returns, so it can't be deferred"),
    ("defer missing()", "Call to undeclared function 'missing'"),
    ("defer f('one')", "Argument 1 to 'f' should be int, got str"),
])
def test_what_cannot_be_deferred(statement, message):
    assert_program_semantic_error(
        "type Point struct:\n    int x\n"
        "def int f(int n):\n    return n\n"
        "def never stop():\n    panic('stop')\n"
        f"def int main():\n    {statement}\n    return 0\n", match=message)
