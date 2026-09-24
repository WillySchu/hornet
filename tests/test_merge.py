"""Tests for merge.py's own merge_programs: folding every discovered
module's own declarations into the entry program, mangled and with
every qualified reference resolved -- entirely before semantic.py
ever runs (see merge.py's own module docstring).

Most of these compile and run an actual multi-file program end to end
(through discover_modules -> merge_programs -> desugar_methods ->
analyze -> generate_asm -> gcc), the same way tests elsewhere in this
suite already do for a single file -- this feature's whole point is
files interacting correctly, so a merged-AST-shape check alone
wouldn't cover nearly as much of what actually matters.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from desugar import desugar_methods
from lexer import lex
from merge import merge_programs, MergeError
from modules import discover_modules
from parser import Parser, ParseError
from semantic import analyze, SemanticError

GCC_AVAILABLE = shutil.which("gcc") is not None
pytestmark = pytest.mark.skipif(not GCC_AVAILABLE, reason="gcc not available")

RUNTIME_C_PATH = Path(__file__).parent.parent / "runtime" / "runtime.c"

# Same convention tests/test_compiler.py's own HOST_IS_MACOS/ASM_
# PLATFORM already use, and build.py/compile.py's own HOST_IS_MACOS/
# DEFAULT_PLATFORM before that -- see build.py's own comment for why
# this matters: gcc's default target on Apple Silicon is arm64, which
# can't assemble this compiler's x86-64 AT&T-syntax output at all, and
# the assembly itself is platform-shaped regardless of host (a leading
# underscore on every external symbol, on macOS) -- generating Linux-
# shaped assembly unconditionally, the way an earlier version of this
# file did, fails on a macOS host for both reasons at once.
HOST_IS_MACOS = sys.platform == "darwin"
ASM_PLATFORM = "macos" if HOST_IS_MACOS else "linux"


def _write(tmpdir: str, name: str, content: str) -> str:
    path = Path(tmpdir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return str(path)


def _compile_and_run(entry_path: str, tmpdir: str) -> subprocess.CompletedProcess:
    """The full pipeline: discover, merge, desugar, analyze, codegen,
    assemble, link, run -- returning the finished process so callers
    can assert on returncode and/or stdout."""
    entry_program, modules = discover_modules(entry_path)
    merged = merge_programs(entry_program, modules)
    desugar_methods(merged)
    analyze(merged)
    from codegen.codegen import generate_asm
    asm = generate_asm(merged, platform=ASM_PLATFORM)

    asm_path = Path(tmpdir) / "program.s"
    asm_path.write_text(asm, encoding="latin-1")
    runtime_o = Path(tmpdir) / "runtime.o"
    runtime_cc_cmd = ["gcc"]
    if HOST_IS_MACOS:
        runtime_cc_cmd += ["-arch", "x86_64"]
    runtime_cc_cmd += ["-c", str(RUNTIME_C_PATH), "-o", str(runtime_o)]
    subprocess.run(runtime_cc_cmd, check=True, capture_output=True)
    binary = Path(tmpdir) / "program"
    gcc_cmd = ["gcc"]
    if HOST_IS_MACOS:
        gcc_cmd += ["-arch", "x86_64"]
    gcc_cmd += [str(asm_path), str(runtime_o), "-o", str(binary)]
    link = subprocess.run(gcc_cmd, capture_output=True, text=True)
    assert link.returncode == 0, f"link failed:\n{link.stderr}\n--- asm ---\n{asm}"
    return subprocess.run([str(binary)], capture_output=True, text=True)


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
            merge_programs(entry_program, modules)


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
            merge_programs(entry_program, modules)


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
        merged = merge_programs(entry_program, modules)
        desugar_methods(merged)
        with pytest.raises(SemanticError):
            analyze(merged)


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
            merge_programs(entry_program, modules)


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
        merged = merge_programs(entry_program, modules)
        desugar_methods(merged)
        with pytest.raises(SemanticError, match="not supported for method calls"):
            analyze(merged)


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
            merge_programs(entry_program, modules)


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
            merge_programs(entry_program, modules)



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
        merged = merge_programs(entry_program, modules)
        desugar_methods(merged)
        with pytest.raises(SemanticError):
            analyze(merged)
