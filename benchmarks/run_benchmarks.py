"""Benchmarks: generated-code size, register-allocation stats, and runtime for benchmarks/programs/."""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lexer import lex
from parser import Parser
from desugar import desugar_methods
from semantic import analyze
from codegen.emitter import Emitter
from build import RUNTIME_C_PATH
from ir.program_builder import build_ir_program
from optimize.optimizer import optimize
import codegen.codegen as codegen_module
import codegen.register_allocator as ra_module

PROGRAMS_DIR = Path(__file__).parent / 'programs'
BASELINE_PATH = Path(__file__).parent / 'baseline.json'
TIMING_RUNS = 7
EXECUTION_TIMEOUT = 30

HOST_IS_MACOS = sys.platform == 'darwin'
ASM_PLATFORM = 'macos' if HOST_IS_MACOS else 'linux'

STAT_KEYS = ('total_temps', 'unsafe_span_excluded', 'eligible', 'allocated', 'spilled')


def _instrumented_generate(program):
    """CodeGenerator.generate, capturing (ir, assignment) per function."""
    captured = []
    original = ra_module.allocate_registers

    def wrapper(ir):
        assignment = original(ir)
        captured.append((list(ir), dict(assignment)))
        return assignment

    codegen_module.allocate_registers = wrapper
    try:
        ir_program = build_ir_program(program)
        ir_program = optimize(ir_program)
        asm_program = codegen_module.CodeGenerator().generate(ir_program)
    finally:
        codegen_module.allocate_registers = original
    return asm_program, captured


def _allocation_stats(ir: list, assignment: dict) -> dict:
    """eligible_intervals' breakdown plus exclusion reasons."""
    blocks = ra_module.build_cfg(ir)
    live_in, live_out = ra_module.compute_liveness(blocks)
    intervals = ra_module.compute_live_intervals(blocks, live_in, live_out)
    eligible = ra_module.eligible_intervals(ir, intervals)
    unsafe_span = len(intervals) - len(eligible)
    return {
        'total_temps': len(intervals),
        'unsafe_span_excluded': unsafe_span,
        'eligible': len(eligible),
        'allocated': len(assignment),
        'spilled': len(eligible) - len(assignment),
    }


def _sum_stats(per_function: list) -> dict:
    if not per_function:
        return {key: 0 for key in STAT_KEYS}
    return {key: sum(s[key] for s in per_function) for key in STAT_KEYS}


def _instruction_count(asm_text: str) -> int:
    """Instruction line count (excludes labels, directives, comments)."""
    count = 0
    for line in asm_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or stripped.startswith('.') or stripped.endswith(':'):
            continue
        count += 1
    return count


def _time_binary(bin_path: Path) -> float:
    """Minimum wall-clock time over TIMING_RUNS runs."""
    times = []
    for _ in range(TIMING_RUNS):
        start = time.perf_counter()
        subprocess.run([str(bin_path)], capture_output=True, timeout=EXECUTION_TIMEOUT)
        times.append(time.perf_counter() - start)
    return min(times)


def run_one(ht_path: Path) -> dict:
    source = ht_path.read_text()
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / 'program.ht'
        src_path.write_text(source)
        tokens = lex(str(src_path))
        program = Parser(tokens).parse_program()
        desugar_methods(program)
        analyze(program)

        asm_program, captured = _instrumented_generate(program)
        asm_text = Emitter(platform=ASM_PLATFORM).emit(asm_program)

        asm_path = Path(tmpdir) / 'program.s'
        bin_path = Path(tmpdir) / 'program'
        runtime_o_path = Path(tmpdir) / 'runtime.o'
        asm_path.write_text(asm_text)

        runtime_cc_cmd = ['gcc']
        if HOST_IS_MACOS:
            runtime_cc_cmd += ['-arch', 'x86_64']
        runtime_cc_cmd += ['-c', str(RUNTIME_C_PATH), '-o', str(runtime_o_path)]
        runtime_result = subprocess.run(runtime_cc_cmd, capture_output=True, text=True)
        if runtime_result.returncode != 0:
            raise RuntimeError(f"gcc failed to compile runtime.c:\n{runtime_result.stderr}")

        gcc_cmd = ['gcc']
        if HOST_IS_MACOS:
            gcc_cmd += ['-arch', 'x86_64']
        gcc_cmd += [str(asm_path), str(runtime_o_path), '-o', str(bin_path)]
        result = subprocess.run(gcc_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"gcc failed to assemble/link {ht_path.name}:\n{result.stderr}")

        per_function_stats = [
            _allocation_stats(ir, assignment) for ir, assignment in captured
        ]
        elapsed = _time_binary(bin_path)

    return {
        'name': ht_path.stem,
        'instruction_count': _instruction_count(asm_text),
        'runtime_seconds': elapsed,
        'allocation': _sum_stats(per_function_stats),
    }


def format_report(results: dict) -> str:
    header = (
        f"{'benchmark':<22} {'instrs':>8} {'time(ms)':>10} "
        f"{'temps':>7} {'elig':>6} {'alloc':>6} {'spill':>6} {'unsafe':>7}"
    )
    lines = [header, '-' * len(header)]
    for name, r in sorted(results.items()):
        a = r['allocation']
        lines.append(
            f"{name:<22} {r['instruction_count']:>8} {r['runtime_seconds'] * 1000:>10.1f} "
            f"{a['total_temps']:>7} {a['eligible']:>6} {a['allocated']:>6} {a['spilled']:>6} "
            f"{a['unsafe_span_excluded']:>7}"
        )
    lines.append('')
    lines.append(
        "temps: total Temps created. elig: eligible for allocation (see "
        "register_allocator.py's own docstring for the one exclusion). "
        "alloc/spill: of those eligible, how many got a register vs. fell "
        "back to a memory slot. unsafe: excluded because their live range "
        "spans an IRCall."
    )
    return '\n'.join(lines)


def format_diff(results: dict, baseline: dict) -> str:
    lines = ['', '=== vs baseline (benchmarks/baseline.json) ===']
    for name, r in sorted(results.items()):
        if name not in baseline:
            lines.append(f"{name}: NEW (no baseline entry)")
            continue
        b = baseline[name]
        instr_delta = r['instruction_count'] - b['instruction_count']
        time_delta_pct = (
            (r['runtime_seconds'] - b['runtime_seconds']) / b['runtime_seconds'] * 100
            if b['runtime_seconds'] else 0.0
        )
        alloc_delta = r['allocation']['allocated'] - b['allocation']['allocated']
        lines.append(
            f"{name}: instructions {instr_delta:+d}, time {time_delta_pct:+.1f}%, "
            f"allocated temps {alloc_delta:+d}"
        )
    return '\n'.join(lines)


def main():
    arg_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    arg_parser.add_argument(
        '--save-baseline', action='store_true',
        help="Overwrite baseline.json with this run's own results, instead of diffing against it.",
    )
    args = arg_parser.parse_args()

    if shutil.which('gcc') is None:
        print("gcc not found on PATH -- these benchmarks compile and execute real binaries.", file=sys.stderr)
        sys.exit(1)

    results = {}
    for ht_path in sorted(PROGRAMS_DIR.glob('*.ht')):
        print(f"Running {ht_path.stem}...", file=sys.stderr)
        results[ht_path.stem] = run_one(ht_path)

    print()
    print(format_report(results))

    if args.save_baseline:
        BASELINE_PATH.write_text(json.dumps(results, indent=2, sort_keys=True) + '\n')
        print(f"\nSaved baseline to {BASELINE_PATH}")
    elif BASELINE_PATH.exists():
        baseline = json.loads(BASELINE_PATH.read_text())
        print(format_diff(results, baseline))
    else:
        print("\nNo baseline.json yet -- run with --save-baseline to create one.")


if __name__ == '__main__':
    main()
