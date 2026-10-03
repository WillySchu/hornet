"""Modules: names across files (scopes.py), resolved while each file is checked.

Most of these compile and run an actual multi-file program end to end (discover_modules -> analyze ->
generate_asm -> gcc): the point is files interacting correctly.
"""

import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

import pytest

from lexer import lex
from scopes import MergeError
from build import build_executable
from tests.targets import on_every_target, run_binary
from tests.test_compiler import panic_message
from modules import discover_modules
from parser import Parser, ParseError
from semantic import analyze, SemanticError

GCC_AVAILABLE = shutil.which("gcc") is not None
pytestmark = pytest.mark.skipif(not GCC_AVAILABLE, reason="gcc not available")




def _write(tmpdir: str, name: str, content: str) -> str:
    path = Path(tmpdir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return str(path)


def _compile_and_run(entry_path: str, tmpdir: str, args: list = None, stdin: str = None) -> subprocess.CompletedProcess:
    """The full pipeline: discover, analyze, codegen,
    assemble, link, run -- returning the finished process so callers
    can assert on returncode and/or stdout. `args` (default none) are
    passed through to the compiled binary itself as its own argv[1:]
    -- for a program that reads its own command-line arguments (e.g.
    stdlib/os.ht's own get_args), not the compiler's own invocation.
    `stdin` (default none, meaning the binary's own stdin is left
    disconnected from anything -- an immediate EOF, exactly like
    running it with input redirected from /dev/null) is piped into
    the binary's own stdin -- for a program that reads it (e.g.
    stdlib/os.ht's own read_stdin)."""
    def build_and_run(target):
        binary = Path(tmpdir) / f"program-{target}"
        build_executable(entry_path, str(binary), target=target)
        return run_binary(target, [binary, *(args or [])], input=stdin or "", capture_output=True, text=True)
    return on_every_target(build_and_run)


def test_cross_module_function_call():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(tmpdir, "main.ht", "import 'utils'\n\ndef int main():\n    return utils.add(2, 3)\n")
        _write(tmpdir, "utils.ht", "def int add(int a, int b):\n    return a + b\n")
        result = _compile_and_run(entry, tmpdir)
        assert result.returncode == 5


def test_qualified_struct_type_and_construction():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "import 'shapes'\n\n"
            "def int main():\n"
            "    shapes.Circle c = shapes.Circle(5)\n"
            "    return c.radius\n",
        )
        _write(tmpdir, "shapes.ht", "type Circle struct:\n    int radius\n")
        result = _compile_and_run(entry, tmpdir)
        assert result.returncode == 5


def test_import_alias_used_for_qualification():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "import 'utils' as u\n\ndef int main():\n    return u.add(2, 3)\n",
        )
        _write(tmpdir, "utils.ht", "def int add(int a, int b):\n    return a + b\n")
        result = _compile_and_run(entry, tmpdir)
        assert result.returncode == 5


def test_in_module_bare_references_still_resolve_after_mangling():
    """utils.ht's own `outer` calls its own `inner` by bare name --
    both get mangled, but the reference between them must be rewritten
    to match, not left pointing at the now-nonexistent bare name."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(tmpdir, "main.ht", "import 'utils'\n\ndef int main():\n    return utils.outer()\n")
        _write(
            tmpdir, "utils.ht",
            "def int inner():\n    return 7\n\ndef int outer():\n    return inner() + 1\n",
        )
        result = _compile_and_run(entry, tmpdir)
        assert result.returncode == 8


def test_ordinary_struct_method_call_unaffected_by_modules():
    """A struct method call in the entry file, alongside an unrelated
    import -- confirms the Call/Field disambiguation correctly leaves
    an ordinary (non-import-alias) receiver alone."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "import 'shapes'\n\n"
            "type Box struct:\n"
            "    int width\n\n"
            "    def int area(self):\n"
            "        return self.width * self.width\n\n"
            "def int main():\n"
            "    shapes.Circle c = shapes.Circle(5)\n"
            "    Box b = Box(4)\n"
            "    return c.radius + b.area()\n",
        )
        _write(tmpdir, "shapes.ht", "type Circle struct:\n    int radius\n")
        result = _compile_and_run(entry, tmpdir)
        assert result.returncode == 21  # 5 + 4*4


def test_circular_import_compiles_and_runs():
    """The concrete claim this feature's whole design rests on: a
    genuine cycle (a imports b, b imports a right back) is safe under
    the merge model, for the same reason mutual recursion between two
    functions in one file already works -- every signature is
    collected before any body is checked. Verified end to end, not
    just "discovery doesn't hang" (already covered in test_modules.py)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(tmpdir, "main.ht", "import 'a'\n\ndef int main():\n    return a.fromA()\n")
        _write(tmpdir, "a.ht", "import 'b'\n\ndef int fromA():\n    return b.fromB() + 1\n")
        _write(tmpdir, "b.ht", "import 'a'\n\ndef int fromB():\n    return 10\n")
        result = _compile_and_run(entry, tmpdir)
        assert result.returncode == 11


def test_hidden_name_is_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(tmpdir, "main.ht", "import 'utils'\n\ndef int main():\n    return utils._secret()\n")
        _write(tmpdir, "utils.ht", "def int _secret():\n    return 42\n")
        entry_program, modules = discover_modules(entry)
        with pytest.raises(MergeError, match="not visible outside the module that defines it"):
            analyze(entry_program, modules)


def test_hidden_name_is_accessible_from_within_its_own_module():
    """The same '_secret' function, called from WITHIN utils.ht itself
    (by another function also declared there) -- not hidden from its
    own module, only from every other one."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(tmpdir, "main.ht", "import 'utils'\n\ndef int main():\n    return utils.public()\n")
        _write(
            tmpdir, "utils.ht",
            "def int _secret():\n    return 42\n\ndef int public():\n    return _secret()\n",
        )
        result = _compile_and_run(entry, tmpdir)
        assert result.returncode == 42


def test_reference_to_undeclared_name_in_module_is_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(tmpdir, "main.ht", "import 'utils'\n\ndef int main():\n    return utils.nonexistent()\n")
        _write(tmpdir, "utils.ht", "def int helper():\n    return 1\n")
        entry_program, modules = discover_modules(entry)
        with pytest.raises(MergeError, match="is not declared in module 'utils'"):
            analyze(entry_program, modules)


def test_reference_to_unknown_qualifier_falls_through_as_ordinary_field_access():
    """`nonImportedName.foo` -- nonImportedName isn't an import alias
    at all, so _resolve_qualified correctly returns None and this
    falls through to ordinary struct-field-access resolution, which
    then correctly rejects it as semantic.py's own, pre-existing
    "undeclared variable" error -- not a MergeError at all, since this
    was never a qualified reference in the first place."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "def int main():\n    return nonImportedName.foo()\n",
        )
        entry_program, modules = discover_modules(entry)
        with pytest.raises(SemanticError):
            analyze(entry_program, modules)


def test_diamond_import_shared_module_reachable_from_both_paths():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "import 'b'\nimport 'c'\n\ndef int main():\n    return b.fromB() + c.fromC()\n",
        )
        _write(tmpdir, "b.ht", "import 'd'\n\ndef int fromB():\n    return d.fromD() + 1\n")
        _write(tmpdir, "c.ht", "import 'd'\n\ndef int fromC():\n    return d.fromD() + 2\n")
        _write(tmpdir, "d.ht", "def int fromD():\n    return 10\n")
        result = _compile_and_run(entry, tmpdir)
        assert result.returncode == 23  # (10+1) + (10+2)


def test_sum_type_variants_rewritten_correctly_within_an_imported_module():
    """A sum type AND its own variant structs, all declared together
    within one imported module -- confirms SumTypeDef.variants' own
    bare-name rewriting keeps each variant pointing at its own,
    correctly-mangled struct, not the pre-mangling bare name (which
    no longer exists as a declaration once merged, and would fail
    semantic analysis outright as an unrecognized variant if the
    rewrite were wrong). Narrowing (`is`) against a QUALIFIED variant
    name isn't tested here -- neither IsCheck.type_name nor
    SumTypeDef's own variant list is parseable as a qualified name at
    all yet (_parse_if_condition and _parse_sum_type_body each only
    ever consume a single IDENTIFIER token) -- this is scoped to what
    the grammar actually accepts today."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "import 'shapes'\n\n"
            "def int main():\n"
            "    shapes.Thing t = shapes.Circle(5)\n"
            "    return 1\n",
        )
        _write(
            tmpdir, "shapes.ht",
            "type Circle struct:\n    int radius\n\n"
            "type Square struct:\n    int side\n\n"
            "type Thing is Circle | Square\n",
        )
        result = _compile_and_run(entry, tmpdir)
        assert result.returncode == 1


def test_type_alias_declared_in_an_imported_module():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "import 'utils'\n\n"
            "def int main():\n"
            "    utils.MyInt x = 5\n"
            "    return x\n",
        )
        _write(tmpdir, "utils.ht", "type MyInt = int\n")
        result = _compile_and_run(entry, tmpdir)
        assert result.returncode == 5


def test_extern_function_declared_in_an_imported_module_is_never_mangled():
    """The real, load-bearing case _own_extern_names exists for: an
    extern function's own name IS the real C symbol the linker
    resolves against -- mangling it would make the generated assembly
    call a symbol ("utils$abs") that doesn't exist at all. abs is
    already linked in by default (libc). Deliberately scalar-only
    (int in, int out) -- extern FFI interop with str is explicitly
    out of scope (see ir/strings.py's own module docstring), so this
    avoids that entanglement entirely rather than accidentally
    exercising it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "import 'utils'\n\n"
            "def int main():\n"
            "    return utils.abs(0 - 5)\n",
        )
        _write(tmpdir, "utils.ht", "extern int abs(int x)\n")
        result = _compile_and_run(entry, tmpdir)
        assert result.returncode == 5


def test_unknown_module_in_type_position_is_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "def int main():\n"
            "    notImported.Circle c = notImported.Circle(5)\n"
            "    return 0\n",
        )
        entry_program, modules = discover_modules(entry)
        with pytest.raises(MergeError, match="doesn't name an imported module"):
            analyze(entry_program, modules)


def test_array_of_qualified_struct_type():
    """Exercises ArrayTypeExpr's own nested-type rewriting -- a
    QualifiedTypeExpr wrapped inside an array type, not just a bare
    one."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "import 'shapes'\n\n"
            "def int main():\n"
            "    [2]shapes.Circle arr = [shapes.Circle(3), shapes.Circle(4)]\n"
            "    return arr[0].radius + arr[1].radius\n",
        )
        _write(tmpdir, "shapes.ht", "type Circle struct:\n    int radius\n")
        result = _compile_and_run(entry, tmpdir)
        assert result.returncode == 7


def test_qualified_struct_construction_with_named_kwargs():
    """Exercises the kwargs-rewriting branch of a qualified Call --
    struct construction using named fields, not positional args. See
    _check_method_call_with_named_kwargs_is_rejected for the OTHER
    half of this feature: an ordinary method call still can't use
    named arguments, even though the grammar accepts them generically
    for every receiver-based call (parse_receiver_call_args' own
    docstring explains why it has to)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "import 'shapes'\n\n"
            "def int main():\n"
            "    shapes.Circle c = shapes.Circle(radius=9)\n"
            "    return c.radius\n",
        )
        _write(tmpdir, "shapes.ht", "type Circle struct:\n    int radius\n")
        result = _compile_and_run(entry, tmpdir)
        assert result.returncode == 9


def test_method_call_with_named_kwargs_is_rejected():
    """The other half of the fix: parse_receiver_call_args has to
    accept named kwargs generically (no symbol table at parse time to
    tell a qualified struct construction apart from an ordinary
    method call), so this is caught downstream instead, by _check_
    method_call's own explicit rejection -- confirmed here as a clear
    SemanticError naming named arguments specifically, not the
    confusing argument-count mismatch it would otherwise silently
    fall through to (0 positional args vs. 1 expected, with the
    named one nowhere mentioned)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "type Box struct:\n"
            "    int width\n\n"
            "    def int area(self):\n"
            "        return self.width * self.width\n\n"
            "def int main():\n"
            "    Box b = Box(4)\n"
            "    return b.area(x=1)\n",
        )
        entry_program, modules = discover_modules(entry)
        with pytest.raises(SemanticError, match="not supported for method calls"):
            analyze(entry_program, modules)


def test_in_module_struct_field_referencing_another_struct_in_the_same_module():
    """A struct field whose own type is ANOTHER struct declared in
    the SAME imported module -- a bare (unqualified, in-module) type
    reference, exercising _rewrite_type_expr's own bare-string branch
    rather than QualifiedTypeExpr's."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "import 'shapes'\n\n"
            "def int main():\n"
            "    shapes.Box b = shapes.Box(shapes.Point(3, 4))\n"
            "    return b.corner.x + b.corner.y\n",
        )
        _write(
            tmpdir, "shapes.ht",
            "type Point struct:\n    int x\n    int y\n\n"
            "type Box struct:\n    Point corner\n",
        )
        result = _compile_and_run(entry, tmpdir)
        assert result.returncode == 7


def test_qualified_narrowing_with_as_binding_bare_subject():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "import 'shapes'\n\n"
            "def int main():\n"
            "    shapes.Thing t = shapes.Circle(5)\n"
            "    if t is shapes.Circle as c:\n"
            "        return c.radius\n"
            "    return 0\n",
        )
        _write(
            tmpdir, "shapes.ht",
            "type Circle struct:\n    int radius\n\n"
            "type Square struct:\n    int side\n\n"
            "type Thing is Circle | Square\n",
        )
        result = _compile_and_run(entry, tmpdir)
        assert result.returncode == 5


def test_qualified_narrowing_non_bare_subject():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "import 'shapes'\n\n"
            "def shapes.Thing makeThing():\n"
            "    return shapes.Circle(8)\n\n"
            "def int main():\n"
            "    if makeThing() is shapes.Circle as c:\n"
            "        return c.radius\n"
            "    return 0\n",
        )
        _write(
            tmpdir, "shapes.ht",
            "type Circle struct:\n    int radius\n\n"
            "type Square struct:\n    int side\n\n"
            "type Thing is Circle | Square\n",
        )
        result = _compile_and_run(entry, tmpdir)
        assert result.returncode == 8


def test_sum_type_with_qualified_variants_declared_in_the_entry_file():
    """The sum type ITSELF declared in the entry file, with its own
    variants qualified references into an imported module -- the
    mirror image of test_sum_type_variants_rewritten_correctly_
    within_an_imported_module, which keeps everything in one module."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "import 'shapes'\n\n"
            "type Thing is shapes.Circle | shapes.Square\n\n"
            "def int main():\n"
            "    Thing t = shapes.Circle(7)\n"
            "    match t:\n"
            "        is shapes.Circle:\n"
            "            return t.radius\n"
            "        is shapes.Square:\n"
            "            return t.side\n",
        )
        _write(
            tmpdir, "shapes.ht",
            "type Circle struct:\n    int radius\n\ntype Square struct:\n    int side\n",
        )
        result = _compile_and_run(entry, tmpdir)
        assert result.returncode == 7


def test_qualified_match_arm_selects_the_other_variant():
    """The same program as above, but constructing the OTHER variant
    -- confirms both qualified match arms resolve to their own,
    correctly distinct mangled struct, not both accidentally matching
    the same one."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "import 'shapes'\n\n"
            "type Thing is shapes.Circle | shapes.Square\n\n"
            "def int main():\n"
            "    Thing t = shapes.Square(9)\n"
            "    match t:\n"
            "        is shapes.Circle:\n"
            "            return t.radius\n"
            "        is shapes.Square:\n"
            "            return t.side\n",
        )
        _write(
            tmpdir, "shapes.ht",
            "type Circle struct:\n    int radius\n\ntype Square struct:\n    int side\n",
        )
        result = _compile_and_run(entry, tmpdir)
        assert result.returncode == 9


def test_unknown_module_in_qualified_narrowing_is_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "def int main():\n"
            "    int x = 5\n"
            "    if x is notImported.Circle as c:\n"
            "        return 1\n"
            "    return 0\n",
        )
        entry_program, modules = discover_modules(entry)
        with pytest.raises(MergeError, match="doesn't name an imported module"):
            analyze(entry_program, modules)


def test_hidden_struct_in_qualified_narrowing_is_rejected():
    """Visibility applies to a qualified `is` target exactly like any
    other qualified reference -- _rewrite_type_expr's own Qualified
    TypeExpr case delegates to the SAME _resolve_qualified/_check_
    visible machinery every other position already uses, so this
    isn't a separate check to maintain, just a consequence of reusing
    it. shapes.ht itself constructs the hidden variant (visible from
    within its own module) and hands it back already boxed as a
    Thing, so the ENTRY file's only qualified reference to the hidden
    name at all is the `is` check itself -- isolating that one path,
    rather than also tripping over a second violation at the
    construction site."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "import 'shapes'\n\n"
            "def int main():\n"
            "    shapes.Thing t = shapes.makeHidden()\n"
            "    if t is shapes._Circle as c:\n"
            "        return c.radius\n"
            "    return 0\n",
        )
        _write(
            tmpdir, "shapes.ht",
            "type _Circle struct:\n    int radius\n\n"
            "type Square struct:\n    int side\n\n"
            "type Thing is _Circle | Square\n\n"
            "def Thing makeHidden():\n"
            "    return _Circle(5)\n",
        )
        entry_program, modules = discover_modules(entry)
        with pytest.raises(MergeError, match="not visible outside the module that defines it"):
            analyze(entry_program, modules)



    """`utils.helper` with no call -- parses as a qualified Field
    access (see Field's own docstring), correctly rewritten to a bare
    Variable("utils$helper") by merge.py, and THEN correctly rejected
    by semantic.py's own, pre-existing "undeclared variable" check --
    since a function name used as a bare value isn't a variable at
    all. Confirms this falls through to an ordinary, existing error
    rather than crashing anywhere in the merge/rewrite pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "import 'utils'\n\n"
            "def int main():\n"
            "    return utils.helper\n",
        )
        _write(tmpdir, "utils.ht", "def int helper():\n    return 1\n")
        entry_program, modules = discover_modules(entry)
        with pytest.raises(SemanticError):
            analyze(entry_program, modules)


def test_basic_named_import_function_call():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(tmpdir, "main.ht", "from 'utils' import add\n\ndef int main():\n    return add(2, 3)\n")
        _write(tmpdir, "utils.ht", "def int add(int a, int b):\n    return a + b\n")
        result = _compile_and_run(entry, tmpdir)
        assert result.returncode == 5


def test_named_import_with_as_renaming_for_a_struct_type_and_a_function():
    """A renamed named import used both in TYPE position (as a
    VarDecl's own type and as a construction call) and in ordinary
    CALL position -- confirms _resolve_named is reached correctly from
    both _rewrite_type_expr's own bare-string branch and _rewrite_
    node's own bare-Call branch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "from 'shapes' import Circle as C, radius as getRadius\n\n"
            "def int main():\n"
            "    C c = C(7)\n"
            "    return getRadius(c)\n",
        )
        _write(
            tmpdir, "shapes.ht",
            "type Circle struct:\n    int radius\n\n"
            "def int radius(Circle c):\n    return c.radius\n",
        )
        result = _compile_and_run(entry, tmpdir)
        assert result.returncode == 7


def test_named_import_colliding_with_own_declaration_is_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "from 'utils' import helper\n\n"
            "def int helper():\n"
            "    return 1\n\n"
            "def int main():\n"
            "    return 0\n",
        )
        _write(tmpdir, "utils.ht", "def int helper():\n    return 2\n")
        entry_program, modules = discover_modules(entry)
        with pytest.raises(MergeError, match="collides with this file's own declaration"):
            analyze(entry_program, modules)


def test_unused_invalid_named_import_is_rejected_eagerly():
    """The concrete case that motivates validating named imports up
    front rather than only when a reference happens to use them:
    'nonexistent' is never referenced anywhere in main.ht's own body
    at all -- a purely lazy, rewrite-time-only check would never
    encounter it, and would silently let this compile."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "from 'utils' import nonexistent\n\ndef int main():\n    return 0\n",
        )
        _write(tmpdir, "utils.ht", "def int helper():\n    return 1\n")
        entry_program, modules = discover_modules(entry)
        with pytest.raises(MergeError, match="is not declared in module 'utils'"):
            analyze(entry_program, modules)


def test_hidden_name_via_named_import_is_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "from 'utils' import _secret\n\ndef int main():\n    return 0\n",
        )
        _write(tmpdir, "utils.ht", "def int _secret():\n    return 1\n")
        entry_program, modules = discover_modules(entry)
        with pytest.raises(MergeError, match="not visible outside the module that defines it"):
            analyze(entry_program, modules)


def test_qualified_and_named_import_of_the_same_module_both_work():
    """Per this feature's own design discussion: no good reason to
    disallow a plain and a named import of the same module coexisting,
    since they populate genuinely different namespaces (a qualifier
    prefix vs. a bare name) -- confirmed end to end, both actually
    being called."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "import 'utils'\n"
            "from 'utils' import subtract\n\n"
            "def int main():\n"
            "    return utils.add(2, 3) + subtract(10, 4)\n",
        )
        _write(
            tmpdir, "utils.ht",
            "def int add(int a, int b):\n    return a + b\n\n"
            "def int subtract(int a, int b):\n    return a - b\n",
        )
        result = _compile_and_run(entry, tmpdir)
        assert result.returncode == 11  # (2+3) + (10-4)


def test_named_import_from_a_module_also_involved_in_a_circular_import():
    """Named imports compose with the existing circular-import
    support, not just plain qualified ones -- b named-imports a
    (non-recursive) helper from a, while a and b still import each
    other directly too. Deliberately NOT named-importing fromA itself
    from within fromB -- fromB calling fromA, which itself calls
    fromB, would be genuine infinite runtime recursion, not a
    compile-time concern this test is meant to exercise at all."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(tmpdir, "main.ht", "import 'a'\n\ndef int main():\n    return a.fromA()\n")
        _write(tmpdir, "a.ht", "import 'b'\n\ndef int fromA():\n    return b.fromB() + 1\n\ndef int aValue():\n    return 100\n")
        _write(tmpdir, "b.ht", "from 'a' import aValue\n\ndef int fromB():\n    return aValue() + 10\n")
        result = _compile_and_run(entry, tmpdir)
        assert result.returncode == 111  # (100 + 10) + 1


def test_intrinsic_round_trips_a_str_through_raw_parts():
    """The three intrinsics called directly, in the entry file (never
    mangled) -- confirms both ir-building hook points (_raw_ptr/_raw_
    len in the scalar-Call dispatch, _from_raw_parts inside _ir_str_
    value's own dispatcher) work correctly together."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "intrinsic *byte _raw_ptr(str s)\n"
            "intrinsic int _raw_len(str s)\n"
            "intrinsic str _from_raw_parts(*byte p, int n)\n\n"
            "def int main():\n"
            "    str s = 'hello'\n"
            "    *byte p = _raw_ptr(s)\n"
            "    int n = _raw_len(s)\n"
            "    str s2 = _from_raw_parts(p, n)\n"
            "    if s2 == 'hello':\n"
            "        return n\n"
            "    return -1\n",
        )
        result = _compile_and_run(entry, tmpdir)
        assert result.returncode == 5


def test_intrinsic_via_a_return_statement_directly():
    """The specific bug found and fixed this arc: Return's own IR-
    building has a separate fast path for 'a composite return whose
    own value is an ordinary function call', forwarding the current
    function's own hidden return pointer straight through -- which,
    left unexcluded, would treat an intrinsic call exactly like an
    ordinary one and try to emit a real call to a symbol that has no
    compiled body anywhere, failing at LINK time, not compile time.
    This is the direct regression test for that fix: a function whose
    entire body is `return _from_raw_parts(...)`."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "intrinsic *byte _raw_ptr(str s)\n"
            "intrinsic int _raw_len(str s)\n"
            "intrinsic str _from_raw_parts(*byte p, int n)\n\n"
            "def str identity(str s):\n"
            "    return _from_raw_parts(_raw_ptr(s), _raw_len(s))\n\n"
            "def int main():\n"
            "    if identity('hello') == 'hello':\n"
            "        return 1\n"
            "    return 0\n",
        )
        result = _compile_and_run(entry, tmpdir)
        assert result.returncode == 1


def test_intrinsic_with_unrecognized_name_is_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "intrinsic int _not_a_real_intrinsic(str s)\n\ndef int main():\n    return 0\n",
        )
        entry_program, modules = discover_modules(entry)
        with pytest.raises(MergeError, match="isn't a recognized intrinsic"):
            analyze(entry_program, modules)


def test_intrinsic_with_mismatched_signature_is_rejected():
    """Right name, wrong signature -- _raw_len is supposed to take a
    str and return int; this one takes a bool instead."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "intrinsic int _raw_len(bool b)\n\ndef int main():\n    return 0\n",
        )
        entry_program, modules = discover_modules(entry)
        with pytest.raises(MergeError, match="doesn't match its own required signature"):
            analyze(entry_program, modules)


def test_intrinsic_signature_accepts_either_byte_or_uint8_spelling():
    """*byte and *uint8 are the same type (see lexer.py's own keyword
    table) but textually different spellings at the AST-shape level
    this validation runs at, before semantic.py's own type_from_name
    ever canonicalizes anything -- confirms _canonical_type_spelling
    treats them as equivalent rather than rejecting the less-common
    spelling as a signature mismatch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "intrinsic *uint8 _raw_ptr(str s)\n\ndef int main():\n    return 0\n",
        )
        entry_program, modules = discover_modules(entry)
        analyze(entry_program, modules)  # should not raise


def test_stdlib_c_module_round_trips_a_str_through_a_c_string():
    """The real, shipped stdlib/c.ht, imported the ordinary way --
    not a scratch reimplementation of it. Confirms to_cstring/from_
    cstring actually work end to end through libc's own calloc/
    memcpy/strlen, not just that the underlying intrinsics do."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "import 'c'\n\n"
            "def int main():\n"
            "    str s = 'hello world'\n"
            "    *byte cstr = c.to_cstring(s)\n"
            "    str back = c.from_cstring(cstr)\n"
            "    if back == s:\n"
            "        return 1\n"
            "    return 0\n",
        )
        result = _compile_and_run(entry, tmpdir)
        assert result.returncode == 1


def test_stdlib_c_module_round_trips_an_empty_string():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "import 'c'\n\n"
            "def int main():\n"
            "    str empty = ''\n"
            "    *byte cstr = c.to_cstring(empty)\n"
            "    str back = c.from_cstring(cstr)\n"
            "    return len(back)\n",
        )
        result = _compile_and_run(entry, tmpdir)
        assert result.returncode == 0


def test_stdlib_c_module_embedded_null_truncates_as_expected_for_c_strings():
    """Not a bug -- an inherent, documented limitation of null-
    terminated C strings, which can never represent an embedded null
    at all. Confirms the round trip truncates predictably (at the
    first null, via the real libc strlen) rather than crashing or
    silently corrupting something."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "import 'c'\n\n"
            "def int main():\n"
            "    str withNul = 'abc\\0xyz'\n"
            "    *byte cstr = c.to_cstring(withNul)\n"
            "    str back = c.from_cstring(cstr)\n"
            "    if back == 'abc':\n"
            "        return 1\n"
            "    return 0\n",
        )
        result = _compile_and_run(entry, tmpdir)
        assert result.returncode == 1


def test_stdlib_c_modules_raw_intrinsics_are_hidden():
    """Per the design decision to hide the raw three and expose only
    to_cstring/from_cstring: a qualified reference to _raw_ptr from
    OUTSIDE c.ht itself is rejected by the same visibility mechanism
    any other '_'-prefixed name already gets -- no new code, just
    ordinary leading-underscore hiding."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "import 'c'\n\n"
            "def int main():\n"
            "    str s = 'hello'\n"
            "    *byte p = c._raw_ptr(s)\n"
            "    return 0\n",
        )
        entry_program, modules = discover_modules(entry)
        with pytest.raises(MergeError, match="not visible outside the module that defines it"):
            analyze(entry_program, modules)


def test_intrinsic_with_wrong_return_type_is_rejected():
    """Right param, wrong return type -- _raw_len is supposed to
    return int; this one declares bool instead."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "intrinsic bool _raw_len(str s)\n\ndef int main():\n    return 0\n",
        )
        entry_program, modules = discover_modules(entry)
        with pytest.raises(MergeError, match="doesn't match its own required signature"):
            analyze(entry_program, modules)


def test_intrinsic_with_wrong_parameter_count_is_rejected():
    """Right name, right types, but an extra parameter -- _raw_len
    takes exactly one str, not a str plus anything else."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "intrinsic int _raw_len(str s, int extra)\n\ndef int main():\n    return 0\n",
        )
        entry_program, modules = discover_modules(entry)
        with pytest.raises(MergeError, match="doesn't match its own required signature"):
            analyze(entry_program, modules)


def test_intrinsic_colliding_with_a_struct_of_the_same_name_is_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "type _raw_len struct:\n    int x\n\n"
            "intrinsic int _raw_len(str s)\n\n"
            "def int main():\n    return 0\n",
        )
        entry_program, modules = discover_modules(entry)
        with pytest.raises(SemanticError, match="collides with a struct"):
            analyze(entry_program, modules)


def test_intrinsic_colliding_with_an_already_declared_function_is_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "def int _raw_len():\n    return 1\n\n"
            "intrinsic int _raw_len(str s)\n\n"
            "def int main():\n    return 0\n",
        )
        entry_program, modules = discover_modules(entry)
        with pytest.raises(SemanticError, match="is already declared"):
            analyze(entry_program, modules)


def test_stdlib_hash_module_hash_int64_directly_with_value_exceeding_int32_range():
    """Exercises hash_int64 with a value too large for int32 -- the
    case the int64-literal-cast bug fix protects."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "import 'hash'\n\n"
            "def int main():\n"
            "    print(hash.hash_int64(9223372036854775000))\n"
            "    return 0\n",
        )
        result = _compile_and_run(entry, tmpdir)
        assert result.stdout == "-434107886741263413\n"


def test_stdlib_hash_module_hash_str_known_answer():
    """Known-answer test against an independently-computed FNV-1a
    64 reference."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "import 'hash'\n\n"
            "def int main():\n"
            "    print(hash.hash_str('hello'))\n"
            "    return 0\n",
        )
        result = _compile_and_run(entry, tmpdir)
        assert result.stdout == "-6615550055289275125\n"


def test_stdlib_hash_module_hash_str_empty_string():
    """Empty string still hashes to FNV-1a's own offset basis."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "import 'hash'\n\n"
            "def int main():\n"
            "    print(hash.hash_str(''))\n"
            "    return 0\n",
        )
        result = _compile_and_run(entry, tmpdir)
        assert result.stdout == "-3750763034362895579\n"


def test_stdlib_hash_module_hash_int_positive_and_negative():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "import 'hash'\n\n"
            "def int main():\n"
            "    print(hash.hash_int(42))\n"
            "    print(hash.hash_int(-42))\n"
            "    return 0\n",
        )
        result = _compile_and_run(entry, tmpdir)
        assert result.stdout == "-55488592825689361\n-1247359464956196604\n"


def test_stdlib_hash_module_hash_bool():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "import 'hash'\n\n"
            "def int main():\n"
            "    print(hash.hash_bool(true))\n"
            "    print(hash.hash_bool(false))\n"
            "    return 0\n",
        )
        result = _compile_and_run(entry, tmpdir)
        assert result.stdout == "-8517097267634966620\n-6284781860667377211\n"


def test_stdlib_hash_module_hash_byte():
    """Byte 0 and int 0 hash identically; byte 255 differs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "import 'hash'\n\n"
            "def int main():\n"
            "    print(hash.hash_byte(\"\\xff\"))\n"
            "    print(hash.hash_byte(\"\\x00\"))\n"
            "    return 0\n",
        )
        result = _compile_and_run(entry, tmpdir)
        assert result.stdout == "-8064062821143829734\n-6284781860667377211\n"


def test_stdlib_hash_module_equal_inputs_hash_equal():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "import 'hash'\n\n"
            "def int main():\n"
            "    if hash.hash_str('same') == hash.hash_str('same'):\n"
            "        return 1\n"
            "    return 0\n",
        )
        result = _compile_and_run(entry, tmpdir)
        assert result.returncode == 1


def test_stdlib_hash_module_different_strings_hash_differently():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "import 'hash'\n\n"
            "def int main():\n"
            "    if hash.hash_str('abc') == hash.hash_str('abd'):\n"
            "        return 0\n"
            "    return 1\n",
        )
        result = _compile_and_run(entry, tmpdir)
        assert result.returncode == 1


def test_stdlib_os_module_get_args_count_and_values():
    """The program's own name (whatever argv[0] the OS hands it,
    unrelated to and independent of the source file's own name here)
    always occupies args[0] -- matching the C convention get_args'
    own underlying argc/argv already come from -- so three EXTRA
    arguments means len(args) == 4."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "from 'os' import get_args\n\n"
            "def int main(int argc, *byte argv):\n"
            "    []str args = get_args(argc, argv)\n"
            "    print(len(args))\n"
            "    for int i = 1; i < len(args); i += 1:\n"
            "        print(args[i])\n"
            "    return 0\n",
        )
        result = _compile_and_run(entry, tmpdir, args=["alpha", "beta", "gamma"])
        assert result.stdout == "4\nalpha\nbeta\ngamma\n"


def test_stdlib_os_module_get_args_with_no_extra_arguments():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "from 'os' import get_args\n\n"
            "def int main(int argc, *byte argv):\n"
            "    []str args = get_args(argc, argv)\n"
            "    print(len(args))\n"
            "    return 0\n",
        )
        result = _compile_and_run(entry, tmpdir)
        assert result.stdout == "1\n"


def test_stdlib_os_module_get_args_preserves_an_argument_containing_spaces():
    """A single shell-quoted argument with an embedded space must
    still arrive as ONE args[] entry, not be split on whitespace --
    get_args itself never touches an argument's own content at all
    (hornet_argv_get returns the OS's own already-split argv[i]
    unchanged), so this is really confirming the round-trip through
    from_cstring preserves it, not get_args' own indexing logic."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "from 'os' import get_args\n\n"
            "def int main(int argc, *byte argv):\n"
            "    []str args = get_args(argc, argv)\n"
            "    print(args[1])\n"
            "    return 0\n",
        )
        result = _compile_and_run(entry, tmpdir, args=["hello world"])
        assert result.stdout == "hello world\n"


def test_stdlib_os_module_read_file_reads_a_real_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = _write(tmpdir, "data.txt", "line one\nline two\nline three")
        entry = _write(
            tmpdir, "main.ht",
            "from 'os' import read_file\nfrom 'errors' import must_str\n\n"
            "def int main():\n"
            f"    str contents = must_str(read_file('{data_path}'))\n"
            "    print(len(contents))\n"
            "    print(contents)\n"
            "    return 0\n",
        )
        result = _compile_and_run(entry, tmpdir)
        assert result.stdout == "28\nline one\nline two\nline three\n"


def test_stdlib_os_module_read_file_handles_a_file_larger_than_one_chunk():
    """read_all_from_fd's own chunk size is 65536 bytes -- this file
    is deliberately several times that, to prove the loop-until-EOF
    logic actually spans multiple read() calls correctly, not just
    the single-chunk case every other test here happens to exercise."""
    with tempfile.TemporaryDirectory() as tmpdir:
        big_content = "".join(f"line {i}\n" for i in range(20000))
        data_path = _write(tmpdir, "big.txt", big_content)
        entry = _write(
            tmpdir, "main.ht",
            "from 'os' import read_file\nfrom 'errors' import must_str\n\n"
            "def int main():\n"
            f"    str contents = must_str(read_file('{data_path}'))\n"
            "    print(len(contents))\n"
            "    return 0\n",
        )
        result = _compile_and_run(entry, tmpdir)
        assert result.stdout == f"{len(big_content)}\n"


def test_stdlib_os_module_read_file_returns_an_error_for_a_missing_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        missing_path = str(Path(tmpdir) / "does_not_exist.txt")
        entry = _write(
            tmpdir, "main.ht",
            "from 'os' import read_file\nfrom 'errors' import Error, StrResult, must_str\n\n"
            "def int main():\n"
            f"    StrResult r = read_file('{missing_path}')\n"
            "    if r is Error:\n"
            "        print(r.message)\n"
            f"    str contents = must_str(read_file('{missing_path}'))\n"
            "    return 0\n",
        )
        result = _compile_and_run(entry, tmpdir)
        message = f"could not open '{missing_path}': No such file or directory"
        assert (result.returncode, result.stdout, panic_message(result.stderr)) == (
            -signal.SIGABRT, message + "\n", message + "\n")


def test_stdlib_os_module_read_stdin_reads_piped_input():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "from 'os' import read_stdin\nfrom 'errors' import must_str\n\n"
            "def int main():\n"
            "    str contents = must_str(read_stdin())\n"
            "    print(len(contents))\n"
            "    print(contents)\n"
            "    return 0\n",
        )
        result = _compile_and_run(entry, tmpdir, stdin="hello from stdin")
        assert result.stdout == "16\nhello from stdin\n"


def test_stdlib_os_module_read_stdin_with_no_input_reads_empty():
    """No stdin piped in at all -- read() sees an immediate EOF (0),
    the ordinary end of a genuinely empty stream, not an error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "from 'os' import read_stdin\nfrom 'errors' import must_str\n\n"
            "def int main():\n"
            "    str contents = must_str(read_stdin())\n"
            "    print(len(contents))\n"
            "    return 0\n",
        )
        result = _compile_and_run(entry, tmpdir)
        assert result.stdout == "0\n"


def test_stdlib_os_module_read_file_and_read_stdin_share_read_all_from_fd():
    """The same content, read once from a file and once from stdin,
    must come back identical -- read_all_from_fd's own {>0, 0, <0}
    read() loop genuinely doesn't know or care which kind of fd it
    was handed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        content = "shared content\nacross both paths\n"
        data_path = _write(tmpdir, "data.txt", content)
        entry = _write(
            tmpdir, "main.ht",
            "from 'os' import read_file, read_stdin, get_args\nfrom 'errors' import must_str\n\n"
            "def int main(int argc, *byte argv):\n"
            "    []str args = get_args(argc, argv)\n"
            "    str contents = ''\n"
            "    if len(args) > 1:\n"
            "        contents = must_str(read_file(args[1]))\n"
            "    else:\n"
            "        contents = must_str(read_stdin())\n"
            "    print(contents)\n"
            "    return 0\n",
        )
        from_file = _compile_and_run(entry, tmpdir, args=[data_path])
        from_stdin = _compile_and_run(entry, tmpdir, stdin=content)
        assert from_file.stdout == content + "\n"
        assert from_stdin.stdout == content + "\n"


def test_stdlib_fmt_module_int_to_str_known_values():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "from 'fmt' import int_to_str\n\n"
            "def int main():\n"
            "    print(int_to_str(0))\n"
            "    print(int_to_str(7))\n"
            "    print(int_to_str(42))\n"
            "    print(int_to_str(100))\n"
            "    print(int_to_str(999999999))\n"
            "    return 0\n",
        )
        result = _compile_and_run(entry, tmpdir)
        assert result.stdout == "0\n7\n42\n100\n999999999\n"


def test_stdlib_fmt_module_int_to_str_result_is_an_ordinary_str():
    """Concatenable with other strings, like any other str value --
    not a special, only-printable result."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "from 'fmt' import int_to_str\n\n"
            "def int main():\n"
            "    print(int_to_str(3) + ' lines')\n"
            "    return 0\n",
        )
        result = _compile_and_run(entry, tmpdir)
        assert result.stdout == "3 lines\n"


def test_stdlib_fmt_module_int_to_str_formats_a_negative_number():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(
            tmpdir, "main.ht",
            "from 'fmt' import int_to_str\n\n"
            "def int main():\n"
            "    print(int_to_str(0 - 5))\n"
            "    return 0\n",
        )
        result = _compile_and_run(entry, tmpdir)
        assert result.returncode == 0
        assert result.stdout == "-5\n"




def test_types_and_names_inside_literals_are_resolved_across_modules():
    """Typed array literals, dict types and literals, dict entries, and named arguments all name
    module types and constants that merging must rename."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write(
            tmpdir, "lib.ht",
            "type Q struct:\n    int y\n"
            "const int K = 5\n"
            "def int _helper():\n    return 7\n"
            "def int own():\n"
            "    []Q qs = []Q[Q(1)]\n"
            "    [1]*Q qp = [1]*Q[&qs[0]]\n"
            "    dict[str]Q d = dict[str]Q{'a': Q(2)}\n"
            "    dict[int]int e = dict[int]int{K: _helper()}\n"
            "    Q n = Q(y=K)\n"
            "    return qs[0].y + qp[0].y + d['a'].y + e[K] + n.y\n",
        )
        entry = _write(
            tmpdir, "main.ht",
            "import 'lib'\n"
            "from 'lib' import Q\n"
            "def int main():\n"
            "    []lib.Q qs = []lib.Q[lib.Q(1)]\n"
            "    [1]Q qa = [1]Q[Q(2)]\n"
            "    dict[str]lib.Q d = dict[str]lib.Q{'a': lib.Q(3)}\n"
            "    []dict[str]lib.Q ds = []dict[str]lib.Q[d]\n"
            "    dict[int]*lib.Q pm\n"
            "    print(qs[0].y + qa[0].y + d['a'].y + ds[0]['a'].y + len(pm))\n"
            "    print(lib.own())\n"
            "    return 0\n",
        )
        result = _compile_and_run(entry, tmpdir)
        assert result.stdout == "9\n16\n", result.stdout + result.stderr


def _analyze_files(tmpdir: str, files: dict):
    for name, content in files.items():
        _write(tmpdir, name, content)
    entry_program, modules = discover_modules(str(Path(tmpdir) / 'main.ht'))
    return entry_program, modules


@pytest.mark.parametrize('local', [
    "    int lib = 1\n",
    "    for lib in [1, 2]:\n        print(lib)\n",
])
def test_a_local_cant_reuse_an_import_alias(local):
    with tempfile.TemporaryDirectory() as tmpdir:
        entry_program, modules = _analyze_files(tmpdir, {
            'main.ht': "import 'lib'\ndef int main():\n" + local + "    return 0\n",
            'lib.ht': "def int f():\n    return 1\n",
        })
        with pytest.raises(MergeError, match="'lib' at line 3 is an imported module's name here and can't also be a variable name"):
            analyze(entry_program, modules)


def test_a_parameter_cant_reuse_an_import_alias():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry_program, modules = _analyze_files(tmpdir, {
            'main.ht': "import 'lib' as l\ndef int g(int l):\n    return l\ndef int main():\n    return 0\n",
            'lib.ht': "def int f():\n    return 1\n",
        })
        with pytest.raises(MergeError, match="'l' at line 2 is an imported module's name here"):
            analyze(entry_program, modules)


LIB_WITH_EXTERN = "extern int abs(int value)\ndef int f():\n    return abs(-2)\n"


def test_another_modules_extern_needs_an_import():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry_program, modules = _analyze_files(tmpdir, {
            'main.ht': "import 'lib'\ndef int main():\n    return abs(-3) + lib.f()\n",
            'lib.ht': LIB_WITH_EXTERN,
        })
        with pytest.raises(SemanticError, match="Call to undeclared function 'abs'"):
            analyze(entry_program, modules)


@pytest.mark.parametrize('main', [
    "from 'lib' import abs\ndef int main():\n    return abs(-3) + lib_f()\n",
    "import 'lib'\ndef int main():\n    return lib.abs(-3) + lib_f()\n",
])
def test_an_imported_extern_is_callable(main):
    with tempfile.TemporaryDirectory() as tmpdir:
        main = main.replace("lib_f()", "2")
        entry = _write(tmpdir, 'main.ht', main)
        _write(tmpdir, 'lib.ht', LIB_WITH_EXTERN)
        assert _compile_and_run(entry, tmpdir).returncode == 5


def test_types_from_other_modules_print_and_report_by_their_declared_names():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = _write(tmpdir, 'main.ht', "import 'shapes'\ndef int main():\n    print(shapes.Box(3))\n    return 0\n")
        _write(tmpdir, 'shapes.ht', "type Box struct:\n    int side\n")
        assert _compile_and_run(entry, tmpdir).stdout.strip() == "Box(side: 3)"
        _write(tmpdir, 'main.ht', "import 'shapes'\ndef int main():\n    shapes.Box b = 1\n    return 0\n")
        entry_program, modules = discover_modules(entry)
        with pytest.raises(SemanticError, match=r"Cannot initialize 'b' \(declared Box\) with a value of type int"):
            analyze(entry_program, modules)


def test_analysis_leaves_every_modules_tree_unchanged():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry_program, modules = _analyze_files(tmpdir, {
            'main.ht': "import 'lib'\nfrom 'lib' import K\ndef int main():\n    [lib.K]int a\n"
                       "    lib.Box b = lib.Box(K)\n    return b.get() + lib.f()\n",
            'lib.ht': "const int K = 2\ntype Box struct:\n    int n\n    def int get(self):\n        return self.n\n"
                      "def int f():\n    return K\n",
        })
        programs = [entry_program] + [m.program for m in modules.values()]
        before = [[d.pretty() for d in p.functions + p.structs + p.consts] for p in programs]
        typed_program = analyze(entry_program, modules)
        assert sorted(fn.name for fn in typed_program.functions) == ['lib$Box.get', 'lib$f', 'main']
        assert [[d.pretty() for d in p.functions + p.structs + p.consts] for p in programs] == before
