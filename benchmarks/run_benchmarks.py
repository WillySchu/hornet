"""Benchmarks: generated-code size, register-allocation stats, and runtime for benchmarks/programs/."""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from desugar import desugar_methods
from merge import merge_programs
from modules import discover_modules
from semantic import analyze
from codegen.emitter import Emitter
from build import RUNTIME_C_PATH, c_compiler
from target import default_target
from ir.program_builder import build_ir_program
from optimize.optimizer import optimize
import codegen.codegen as codegen_module
import codegen.register_allocator as ra_module
import ir.cfg as cfg

PROGRAMS_DIR = Path(__file__).parent / 'programs'
BASELINE_PATH = Path(__file__).parent / 'baseline.json'
TIMING_RUNS = 7
EXECUTION_TIMEOUT = 30

ASM_TARGET = default_target()

STAT_KEYS = ('total_temps', 'address_taken_excluded', 'eligible', 'allocated', 'spilled', 'live_across_call')


def _instrumented_generate(program):
    """CodeGenerator.generate, capturing (ir, temp_home_slots, params, assignment) per function."""
    captured = []
    original = ra_module.allocate_registers

    def wrapper(ir, temp_home_slots=None, params=()):
        assignment = original(ir, temp_home_slots, params)
        captured.append((list(ir), dict(temp_home_slots or {}), list(params), dict(assignment)))
        return assignment

    codegen_module.allocate_registers = wrapper
    try:
        ir_program = build_ir_program(program)
        ir_program = optimize(ir_program)
        asm_program = codegen_module.CodeGenerator().generate(ir_program)
    finally:
        codegen_module.allocate_registers = original
    return asm_program, captured


def _allocation_stats(ir: list, temp_home_slots: dict, params: list, assignment: dict) -> dict:
    """Allocation breakdown, using the same eligibility rules as allocate_registers."""
    blocks = cfg.build_blocks(ir)
    live_in, live_out = cfg.liveness(blocks)
    intervals = ra_module.compute_live_intervals(blocks, live_in, live_out)
    for p in params:
        if p.id in intervals:
            intervals[p.id].start = -1
    eligible = ra_module.eligible_intervals(ir, intervals, temp_home_slots)
    return {
        'total_temps': len(intervals),
        'address_taken_excluded': len(intervals) - len(eligible),
        'eligible': len(eligible),
        'allocated': len(assignment),
        'spilled': len(eligible) - len(assignment),
        'live_across_call': len(ra_module.call_crossing(ir, eligible)),
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


def _time_binary(bin_path: Path, runs: int) -> float:
    """Minimum wall-clock time over `runs` runs; 0.0 if runs is 0."""
    times = [0.0]
    if runs:
        times = []
    for _ in range(runs):
        start = time.perf_counter()
        subprocess.run([str(bin_path)], capture_output=True, timeout=EXECUTION_TIMEOUT)
        times.append(time.perf_counter() - start)
    return min(times)


def _executed_instructions(bin_path: Path) -> int:
    """Instructions executed, counted by valgrind's cachegrind (deterministic)."""
    result = subprocess.run(
        ['valgrind', '--tool=cachegrind', '--cache-sim=no', '--cachegrind-out-file=/dev/null', str(bin_path)],
        capture_output=True, text=True,
    )
    match = re.search(r'I\s+refs:\s+([\d,]+)', result.stderr)
    if match is None:
        raise RuntimeError(f"couldn't read cachegrind output:\n{result.stderr}")
    return int(match.group(1).replace(',', ''))


def run_one(ht_path: Path, runs: int = TIMING_RUNS, icount: bool = False) -> dict:
    """Compile, link, and time one benchmark; optionally count executed instructions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        entry, modules = discover_modules(str(ht_path))
        program = merge_programs(entry, modules)
        desugar_methods(program)
        analyze(program)

        asm_program, captured = _instrumented_generate(program)
        asm_text = Emitter(ASM_TARGET).emit(asm_program)

        asm_path = Path(tmpdir) / 'program.s'
        bin_path = Path(tmpdir) / 'program'
        runtime_o_path = Path(tmpdir) / 'runtime.o'
        asm_path.write_text(asm_text)

        runtime_cc_cmd = c_compiler(ASM_TARGET)
        runtime_cc_cmd += ['-c', str(RUNTIME_C_PATH), '-o', str(runtime_o_path)]
        runtime_result = subprocess.run(runtime_cc_cmd, capture_output=True, text=True)
        if runtime_result.returncode != 0:
            raise RuntimeError(f"gcc failed to compile runtime.c:\n{runtime_result.stderr}")

        gcc_cmd = c_compiler(ASM_TARGET)
        gcc_cmd += [str(asm_path), str(runtime_o_path), '-o', str(bin_path)]
        result = subprocess.run(gcc_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"gcc failed to assemble/link {ht_path.name}:\n{result.stderr}")

        per_function_stats = [_allocation_stats(*c) for c in captured]
        elapsed = _time_binary(bin_path, runs)
        executed = _executed_instructions(bin_path) if icount else None

    return {
        'name': ht_path.stem,
        'instruction_count': _instruction_count(asm_text),
        'runtime_seconds': elapsed,
        'executed_instructions': executed,
        'allocation': _sum_stats(per_function_stats),
    }


def format_report(results: dict) -> str:
    header = (
        f"{'benchmark':<22} {'instrs':>8} {'time(ms)':>10} "
        f"{'temps':>7} {'elig':>6} {'alloc':>6} {'spill':>6} {'x-call':>7} {'addr':>5} {'exec(M)':>9}"
    )
    lines = [header, '-' * len(header)]
    for name, r in sorted(results.items()):
        a = r['allocation']
        lines.append(
            f"{name:<22} {r['instruction_count']:>8} {r['runtime_seconds'] * 1000:>10.1f} "
            f"{a['total_temps']:>7} {a['eligible']:>6} {a['allocated']:>6} {a['spilled']:>6} "
            f"{a['live_across_call']:>7} {a['address_taken_excluded']:>5} "
            f"{_fmt_exec(r.get('executed_instructions')):>9}"
        )
    lines.append('')
    lines.append(
        "temps: Temps created. elig: eligible for a register. alloc/spill: eligible Temps "
        "that got a register / a frame slot. x-call: eligible Temps live across a call. addr: address taken. "
        "exec(M): millions of instructions executed (--icount; needs valgrind)."
    )
    return '\n'.join(lines)


def _fmt_exec(n) -> str:
    return '-' if n is None else f"{n / 1e6:.1f}"


def format_diff(results: dict, baseline: dict, label: str) -> str:
    lines = ['', f'=== vs {label} ===']
    for name, r in sorted(results.items()):
        if name not in baseline:
            lines.append(f"{name}: NEW (no baseline entry)")
            continue
        b = baseline[name]
        instr_delta = r['instruction_count'] - b['instruction_count']
        timed = r['runtime_seconds'] and b['runtime_seconds']
        time_delta = f"{(r['runtime_seconds'] - b['runtime_seconds']) / b['runtime_seconds'] * 100:+.1f}%" if timed else 'n/a'

        alloc_delta = r['allocation']['allocated'] - b['allocation']['allocated']
        line = (
            f"{name}: instructions {instr_delta:+d}, time {time_delta}, "
            f"allocated temps {alloc_delta:+d}"
        )
        if r.get('executed_instructions') and b.get('executed_instructions'):
            exec_pct = (r['executed_instructions'] - b['executed_instructions']) / b['executed_instructions'] * 100
            line += f", executed {exec_pct:+.2f}%"
        lines.append(line)
    return '\n'.join(lines)


def main():
    arg_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    arg_parser.add_argument(
        '--save-baseline', action='store_true',
        help="Overwrite baseline.json with this run's own results, instead of diffing against it.",
    )
    arg_parser.add_argument('--runs', type=int, default=TIMING_RUNS, help=f'Timing runs per program (default {TIMING_RUNS}; 0 skips timing)')
    arg_parser.add_argument('--json', type=Path, help='Also write results to this file')
    arg_parser.add_argument('--icount', action='store_true', help='Count executed instructions with valgrind (slow, deterministic)')
    arg_parser.add_argument('--compare', type=Path, help='Diff against this results file instead of baseline.json')
    arg_parser.add_argument('programs', nargs='*', help='Benchmark names to run (default: all)')
    args = arg_parser.parse_args()

    if shutil.which('gcc') is None:
        print("gcc not found on PATH -- these benchmarks compile and execute real binaries.", file=sys.stderr)
        sys.exit(1)
    if args.icount and shutil.which('valgrind') is None:
        print("--icount needs valgrind on PATH.", file=sys.stderr)
        sys.exit(1)

    paths = sorted(PROGRAMS_DIR.glob('*.ht'))
    if args.programs:
        paths = [p for p in paths if p.stem in args.programs]
    results = {}
    for ht_path in paths:
        print(f"Running {ht_path.stem}...", file=sys.stderr)
        results[ht_path.stem] = run_one(ht_path, args.runs, args.icount)

    print()
    print(format_report(results))

    if args.json:
        args.json.write_text(json.dumps(results, indent=2, sort_keys=True) + '\n')
    compare_path = args.compare or BASELINE_PATH
    if args.save_baseline:
        BASELINE_PATH.write_text(json.dumps(results, indent=2, sort_keys=True) + '\n')
        print(f"\nSaved baseline to {BASELINE_PATH}")
    elif compare_path.exists():
        print(format_diff(results, json.loads(compare_path.read_text()), str(compare_path)))
    else:
        print("\nNo baseline.json yet -- run with --save-baseline to create one.")


if __name__ == '__main__':
    main()
