"""The Windows x64 calling convention (backend/x86_64/calling_convention.py's WIN64): four argument
registers, 32 bytes of shadow space on every call, %rdi and %rsi callee-saved, and a stack probe
for a frame of a page or more. And the x86_64-windows target around it."""

import pytest

import build
from backend.x86_64.assembly_ast import CallInstr, Directive, Memory, Mov, MovQ, Pop, Push, Register, Ret, SubQ
from backend.x86_64.calling_convention import SYSV, WIN64, abi_for
from backend.x86_64.codegen import CodeGenerator
from compile import generate_asm
from ir.program_builder import build_ir_program
from ir.typed_builder import link_name
from optimize.optimizer import optimize
from target import TARGET_NAMES, Target
from tests.targets import RUNNABLE_TARGETS, run_binary
from tests.test_compiler import _parse, analyze

WINDOWS = Target('x86_64', 'windows')

SOURCE = (
    "def int six(int a, int b, int c, int d, int e, int f):\n"
    "    return a + b * 2 + c * 3 + d * 4 + e * 5 + f * 6\n"
    "def int big():\n"
    "    [1000]int table\n"           # 8,000 bytes: more than a page
    "    table[999] = 7\n"
    "    return table[999]\n"
    "def int main():\n"
    "    return six(1, 2, 3, 4, 5, 6) + big()\n"
)


def _functions(abi) -> dict:
    program = CodeGenerator(abi).generate(optimize(build_ir_program(analyze(_parse(SOURCE)))))
    return {fn.name: fn.instructions for fn in program.functions}


def _operands(instructions) -> list:
    return [getattr(i, name) for i in instructions for name in ('src', 'dst') if hasattr(i, name)]


def test_the_two_conventions():
    assert (abi_for(WINDOWS) is WIN64
            and abi_for(Target('x86_64', 'linux')) is SYSV is abi_for(Target('x86_64', 'macos')))
    assert WIN64.arg_registers_64 == ('rcx', 'rdx', 'r8', 'r9') and WIN64.shadow_space == 32
    assert {'rdi', 'rsi'} <= set(WIN64.callee_saved_registers) and not {'rdi', 'rsi'} & set(SYSV.callee_saved_registers)
    for abi in (SYSV, WIN64):  # the scratch registers are never allocated, and every pool register has one role
        assert not {'eax', 'ecx', 'edx'} & set(abi.allocatable)
        assert not set(abi.caller_saved_pool) & set(abi.callee_saved_pool)


def test_the_fifth_and_sixth_arguments_travel_on_the_stack_above_the_shadow_space():
    functions = _functions(WIN64)
    # The callee reads them above the saved %rbp, the return address, and 32 bytes of shadow space.
    callee = _operands(functions[link_name('six')])
    assert Memory('rbp', 48) in callee and Memory('rbp', 56) in callee
    # The caller writes them just above the shadow space at the bottom of its frame.
    main = [i for i in functions['main'] if not isinstance(i, Directive)]
    pushes = sum(isinstance(i, Push) for i in main[2:6])
    bottom = -(8 * pushes + next(i.src.value for i in main if isinstance(i, SubQ) and i.dst == Register('rsp')))
    stores = [i.dst.offset for i in main if isinstance(i, MovQ) and isinstance(i.dst, Memory) and i.dst.base == 'rbp']
    assert bottom % 16 == 0 and stores == [bottom + 32, bottom + 40]
    # Under SysV all six are in registers.
    sysv = _functions(SYSV)
    assert not any(isinstance(i, MovQ) and isinstance(i.dst, Memory) for i in sysv['main'])
    assert Memory('rbp', 16) not in _operands(sysv[link_name('six')])


def test_every_frame_keeps_the_shadow_space_and_a_large_one_is_probed():
    functions = _functions(WIN64)
    for name, instructions in functions.items():
        grows = [i for i in instructions[:16] if isinstance(i, SubQ) and i.dst == Register('rsp')]
        assert len(grows) == 1, name
    big = functions[link_name('big')]
    probe = next(i for i, instr in enumerate(big) if isinstance(instr, CallInstr) and instr.target == '___chkstk_ms')
    assert (isinstance(big[probe - 1], Mov)
            and big[probe - 1].dst == Register('eax')
            and big[probe - 1].src.value >= 8000)
    assert big[probe + 1] == SubQ(src=Register('rax'), dst=Register('rsp'))
    assert not any(isinstance(i, CallInstr) and i.target == '___chkstk_ms' for i in functions[link_name('six')])
    assert not any(isinstance(i, CallInstr) and i.target == '___chkstk_ms'
                   for instructions in _functions(SYSV).values() for i in instructions)


def test_the_assembly_text_and_the_target_list():
    asm = generate_asm(analyze(_parse(SOURCE)), target=WINDOWS)
    assert "GNU-stack" not in asm and "\n    .globl main\n" in asm and "call    ___chkstk_ms" in asm
    assert 'x86_64-windows' in TARGET_NAMES and 'aarch64-windows' not in TARGET_NAMES
    with pytest.raises(ValueError, match="unknown target 'aarch64-windows'"):
        Target.parse('aarch64-windows')


def test_the_toolchain(monkeypatch):
    monkeypatch.setattr(build, 'host_target', lambda: Target('x86_64', 'linux'))
    assert build.c_compiler(WINDOWS) == ['x86_64-w64-mingw32-gcc']
    assert build.executable_name('prog', WINDOWS) == 'prog.exe' == build.executable_name('prog.exe', WINDOWS)
    assert build.executable_name('prog', Target('x86_64', 'linux')) == 'prog'
    monkeypatch.setattr(build, 'host_target', lambda: WINDOWS)
    assert build.c_compiler(WINDOWS) == ['gcc'] and build.run_prefix(WINDOWS) == []


runs_windows = pytest.mark.skipif(WINDOWS not in RUNNABLE_TARGETS, reason="needs MinGW-w64, and Wine off Windows")


def _build_and_run(tmp_path, source: str, *args, stdin: bytes = b''):
    (tmp_path / "p.ht").write_text(source)
    build.build_executable(str(tmp_path / "p.ht"), str(tmp_path / "p"), target=WINDOWS)  # no `.exe`: as asked
    return run_binary(WINDOWS, [tmp_path / "p", *args], input=stdin, capture_output=True, cwd=tmp_path)


@runs_windows
def test_a_windows_program_runs(tmp_path):
    result = _build_and_run(tmp_path, SOURCE)
    assert result.returncode == 1 + 4 + 9 + 16 + 25 + 36 + 7


@runs_windows
def test_windows_input_and_output_are_bytes(tmp_path):
    # Text mode would turn "\n" into "\r\n" going out, and drop "\r" and stop at 0x1A coming in.
    result = _build_and_run(
        tmp_path,
        "from 'os' import get_args, read_file, read_stdin, write_file, write_stdout\n"
        "from 'errors' import Error\n"
        "def int main(int argc, *byte argv):\n"
        "    []str args = get_args(argc, argv)\n"
        "    print(len(args))\n"
        "    match read_stdin() as text:\n"
        "        is str:\n"
        "            write_file(args[1], text + text)\n"
        "        is Error:\n"
        "            return 1\n"
        "    match read_file(args[1]) as back:\n"
        "        is str:\n"
        "            write_stdout(back)\n"
        "            return len(back)\n"
        "        is Error:\n"
        "            return 2\n",
        "copy.bin", stdin=b"a\r\nb\n\x1ac\n")
    assert result.stdout == b"2\n" + b"a\r\nb\n\x1ac\n" * 2
    assert result.returncode == 16 and (tmp_path / "copy.bin").read_bytes() == b"a\r\nb\n\x1ac\n" * 2


@runs_windows
def test_a_windows_panic_exits_with_code_3(tmp_path):
    (tmp_path / "p.ht").write_text("def int main():\n    []int xs = [1]\n    int i = len(xs)\n    return xs[i]\n")
    build.build_executable(str(tmp_path / "p.ht"), str(tmp_path / "p.exe"), target=WINDOWS)
    import subprocess
    raw = subprocess.run(build.run_prefix(WINDOWS) + [str(tmp_path / "p.exe")], capture_output=True)
    assert (raw.returncode, raw.stderr) == (3, b"p.ht:4:12: panic: array index out of bounds\n")


def test_unwind_tables_describe_each_prologue_and_the_epilogue_is_the_form_the_unwinder_knows():
    functions = _functions(WIN64)
    for name, instructions in functions.items():
        prologue = instructions[:next(i for i, instr in enumerate(instructions)
                                      if instr == Directive(".seh_endprologue")) + 1]
        described = [instr.text for instr in prologue if isinstance(instr, Directive)]
        pushed = [f".seh_pushreg %{instr.operand.name}" for instr in prologue if isinstance(instr, Push)]
        assert described[0] == ".seh_pushreg %rbp" and described[1] == ".seh_setframe %rbp, 0", name
        assert [d for d in described if "pushreg" in d] == pushed, name
        assert described[-2].startswith(".seh_stackalloc ") and described[-1] == ".seh_endprologue", name
        for i, instr in enumerate(instructions):
            if isinstance(instr, Ret):  # ... lea, pops, pop %rbp, ret: never `leave`
                assert instructions[i - 1] == Pop(Register('rbp')), name
    asm = generate_asm(analyze(_parse(SOURCE)), target=WINDOWS)
    assert asm.count(".seh_proc ") == asm.count(".seh_endproc") == len(functions) and "leave" not in asm
    assert ".seh_" not in generate_asm(analyze(_parse(SOURCE)), target=Target('x86_64', 'linux'))


@runs_windows
def test_the_system_can_walk_the_stack_through_hornet_frames(tmp_path):
    """CaptureStackBackTrace walks by the unwind tables: without them it stops at the first Hornet frame."""
    import subprocess
    (tmp_path / "frames.c").write_text(
        "#include <windows.h>\n"
        "long long frames_above(void) { void *frames[64]; return CaptureStackBackTrace(0, 64, frames, NULL); }\n")
    source = (
        "extern int frames_above()\n"
        "def int level(int n):\n"
        "    if n == 0:\n"
        "        return frames_above()\n"
        "    return level(n - 1) + 0\n"
        "def int big(int n):\n"              # a probed frame
        "    [1000]int table\n"
        "    table[n] = level(n)\n"
        "    return table[n]\n"
        "def int main():\n"
        "    int shallow = level(0)\n"
        "    return (level(5) - shallow) * 10 + big(3) - shallow\n")
    (tmp_path / "p.s").write_text(generate_asm(analyze(_parse(source)), target=WINDOWS))
    subprocess.run(build.c_compiler(WINDOWS) + [str(tmp_path / "p.s"), str(tmp_path / "frames.c"),
                                               str(build.runtime_object(WINDOWS)), "-o", str(tmp_path / "p.exe")],
                   check=True, capture_output=True)
    assert run_binary(WINDOWS, [tmp_path / "p.exe"]).returncode == 5 * 10 + 4
