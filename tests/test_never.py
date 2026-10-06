"""`never` as a return type: the function doesn't return, so a call to it ends its path; and the
built-in `panic(message)`, which is such a call and reports its own position."""

import signal

import pytest

from lexer import Lexer
from parser import ParseError, Parser
from tests.test_compiler import (
    GCC_SKIP, _parse, analyze, assert_program_semantic_error, assert_program_stdout, compile_and_run,
)

DECLS = (
    "type Error struct:\n"
    "    str message\n"
    "    def never raise(self):\n"
    "        panic(self.message)\n"
    "type Result is int | Error\n"
    "extern never abort()\n"
    "def never fail(str what):\n"
    "    panic('failed: ' + what)\n"
)


def test_never_is_parsed_only_as_a_return_type():
    program = Parser(Lexer(DECLS).tokenize()).parse_program()
    assert [program.functions[0].return_type, program.extern_functions[0].return_type,
            program.structs[0].methods[0].return_type] == ['never', 'never', 'never']
    with pytest.raises(ParseError):
        Parser(Lexer("def int main():\n    never x = 1\n    return 0\n").tokenize()).parse_program()
    with pytest.raises(ParseError):
        Parser(Lexer("def int f(never x):\n    return 0\n").tokenize()).parse_program()


def test_a_never_call_ends_its_path():
    # No `return` after it, in a function, an `if`/`else`, or a `match`; and it guards what follows.
    analyze(_parse(
        DECLS +
        "def int direct():\n"
        "    fail('a')\n"
        "def int branches(bool b):\n"
        "    if b:\n"
        "        return 1\n"
        "    else:\n"
        "        abort()\n"
        "def int arms(Result r):\n"
        "    match r as v:\n"
        "        is int:\n"
        "            return v\n"
        "        is Error:\n"
        "            v.raise()\n"
        "def int guarded(Result r):\n"
        "    if r is Error:\n"
        "        panic(r.message)\n"
        "    return r + 1\n"
        "def never forever():\n"
        "    while true:\n"
        "        print(1)\n"
        "def never either(bool b):\n"
        "    if b:\n"
        "        fail('x')\n"
        "    else:\n"
        "        forever()\n"
        "def int main():\n"
        "    return 0\n"
    ))


@pytest.mark.parametrize("body,match", [
    ("def never f(bool b):\n    if b:\n        fail('x')\n", "Function 'f' is declared never, but can finish"),
    ("def never f():\n    print(1)\n", "Function 'f' is declared never, but can finish"),
    ("def never f():\n    while true:\n        break\n", "Function 'f' is declared never, but can finish"),
    ("def never f(bool b):\n    if b:\n        return\n    fail('x')\n", "declared never, so it can't return"),
    ("def never main():\n    fail('x')\n", "main"),
    # A call that doesn't return has no value.
    ("def int f():\n    int x = fail('x')\n    return x\n", "with a value of type never"),
    ("def int f():\n    return fail('x')\n", "returns never"),
    ("def f():\n    print(fail('x'))\n", "there's no value there to print"),
    # A function that only may not return still needs its `return`.
    ("def int f(bool b):\n    if b:\n        fail('x')\n", "does not return a value on all code paths"),
    # panic takes one str, and is a builtin.
    ("def f():\n    panic()\n", "'panic' expects exactly 1 argument, got 0"),
    ("def f():\n    panic(5)\n", "'panic' expects a str, got int"),
    ("def panic(str message):\n    print(message)\n", "builtin"),
])
def test_what_is_an_error(body, match):
    main = "" if "main" in body else "def int main():\n    return 0\n"
    assert_program_semantic_error(DECLS + body + main, match=match)


def _run(main_body: str):
    return compile_and_run(DECLS + "def int main():\n" + main_body)


@GCC_SKIP
def test_panic_reports_its_message_and_position_after_the_programs_output():
    result = _run("    str what = 'disk'\n    print('before')\n    panic('no ' + what)\n")
    assert (result.returncode, result.stdout, result.stderr) == (
        -signal.SIGABRT, "before\n", "program.lang:12:5: panic: no disk\n")


@GCC_SKIP
def test_panic_inside_a_never_function_or_method_reports_that_call():
    assert _run("    fail('x')\n").stderr == "program.lang:8:5: panic: failed: x\n"
    assert _run("    Error e = Error('lost')\n    e.raise()\n").stderr == "program.lang:4:9: panic: lost\n"


@GCC_SKIP
def test_an_extern_declared_never_that_returns_panics():
    result = compile_and_run("extern never getpid()\ndef int main():\n    getpid()\n    return 0\n")
    assert (result.returncode, result.stderr) == (
        -signal.SIGABRT, "program.lang:3:5: panic: a never function returned\n")


@GCC_SKIP
def test_statements_after_a_never_call_are_allowed_and_never_run():
    assert_program_stdout(
        DECLS +
        "def int pick(Result r):\n"
        "    if r is Error:\n"
        "        abort()\n"
        "        print('unreachable')\n"
        "    return r\n"
        "def int main():\n"
        "    print(pick(7))\n"
        "    return 0\n",
        "7\n",
    )
