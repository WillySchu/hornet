"""Runs every benchmark under benchmarks/programs/, reporting
generated-code size, register-allocation statistics, and wall-clock
timing -- checked in so future changes to codegen/register_allocator.py
have something concrete to compare against, rather than "how is this
actually affecting generated code" staying opaque the way it has been
until now.

Deliberately NOT a pytest file: these programs are designed to take
tens to hundreds of milliseconds each (see each program's own comment
for why), and running them repeatedly for timing is inherently slower
and noisier than the rest of the test suite. tests/test_benchmarks.py
is the fast, deterministic sibling of this file -- it just confirms
each program still compiles and produces its known-correct result,
with no timing involved, so these programs can't silently bit-rot
between the (much less frequent) times someone actually runs this
script.

Usage:
    python3 benchmarks/run_benchmarks.py                  # run and print a report
    python3 benchmarks/run_benchmarks.py --save-baseline   # also overwrite baseline.json
"""

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
from semantic import analyze
from codegen.emitter import Emitter
import codegen.codegen as codegen_module
import codegen.register_allocator as ra_module

PROGRAMS_DIR = Path(__file__).parent / 'programs'
BASELINE_PATH = Path(__file__).parent / 'baseline.json'
TIMING_RUNS = 7
EXECUTION_TIMEOUT = 30

# Same platform-detection this compiler's own test suite uses (see
# tests/test_compiler.py's ASM_PLATFORM) -- the generated assembly has
# to match whatever gcc on *this* machine will actually assemble/link.
HOST_IS_MACOS = sys.platform == 'darwin'
ASM_PLATFORM = 'macos' if HOST_IS_MACOS else 'linux'

STAT_KEYS = ('total_temps', 'named_local_excluded', 'unsafe_span_excluded', 'eligible', 'allocated', 'spilled')


def _instrumented_generate(program):
    """Runs CodeGenerator.generate, capturing the (ir, assignment)
    pair from every allocate_registers call made along the way -- one
    per function -- without touching any production code at all: this
    wraps the module-level reference gen_function actually calls,
    exactly like the ad hoc verification scripts used earlier in this
    project's own development did, just kept around properly this
    time instead of being thrown away."""
    captured = []
    original = ra_module.allocate_registers

    def wrapper(ir):
        assignment = original(ir)
        captured.append((list(ir), dict(assignment)))
        return assignment

    codegen_module.allocate_registers = wrapper
    try:
        asm_program = codegen_module.CodeGenerator().generate(program)
    finally:
        codegen_module.allocate_registers = original
    return asm_program, captured


def _allocation_stats(ir: list, assignment: dict) -> dict:
    """Re-derives the same breakdown eligible_intervals itself
    computes, plus WHY each excluded Temp was excluded -- something
    the production code has no reason to track, since it only needs
    the final yes/no, not the reason."""
    blocks = ra_module.build_cfg(ir)
    live_in, live_out = ra_module.compute_liveness(blocks)
    intervals = ra_module.compute_live_intervals(blocks, live_in, live_out)
    eligible = ra_module.eligible_intervals(ir, intervals)
    named_local = sum(1 for iv in intervals.values() if iv.temp.is_named_local)
    unsafe_span = len(intervals) - named_local - len(eligible)
    return {
        'total_temps': len(intervals),
        'named_local_excluded': named_local,
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
    """A simple, deterministic proxy for generated-code size: counts
    real instruction lines, skipping labels, directives, and blank/
    comment lines. Not a substitute for the timing measurement below
    (two versions with the same count can still run at different
    speeds -- a register-to-register move and a memory access are
    both "one instruction"), but a useful, noise-free signal on its
    own for whether a change added or removed real work."""
    count = 0
    for line in asm_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or stripped.startswith('.') or stripped.endswith(':'):
            continue
        count += 1
    return count


def _time_binary(bin_path: Path) -> float:
    """Runs the compiled binary TIMING_RUNS times and returns the
    minimum wall-clock time. The minimum, not the mean or median, is
    the least noisy signal for "how fast can this actually run":
    scheduling noise and other processes on the machine can only ever
    make a run slower, never artificially faster, so the fastest
    observed run is the closest available estimate of the program's
    own true cost."""
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
        analyze(program)

        asm_program, captured = _instrumented_generate(program)
        asm_text = Emitter(platform=ASM_PLATFORM).emit(asm_program)

        asm_path = Path(tmpdir) / 'program.s'
        bin_path = Path(tmpdir) / 'program'
        asm_path.write_text(asm_text)

        gcc_cmd = ['gcc']
        if HOST_IS_MACOS:
            gcc_cmd += ['-arch', 'x86_64']
        gcc_cmd += [str(asm_path), '-o', str(bin_path)]
        result = subprocess.run(gcc_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"gcc failed to assemble/link {ht_path.name}:\n{result.stderr}")

        per_function_stats = [_allocation_stats(ir, assignment) for ir, assignment in captured]
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
        f"{'temps':>7} {'elig':>6} {'alloc':>6} {'spill':>6} {'named':>7} {'unsafe':>7}"
    )
    lines = [header, '-' * len(header)]
    for name, r in sorted(results.items()):
        a = r['allocation']
        lines.append(
            f"{name:<22} {r['instruction_count']:>8} {r['runtime_seconds'] * 1000:>10.1f} "
            f"{a['total_temps']:>7} {a['eligible']:>6} {a['allocated']:>6} {a['spilled']:>6} "
            f"{a['named_local_excluded']:>7} {a['unsafe_span_excluded']:>7}"
        )
    lines.append('')
    lines.append(
        "temps: total Temps created. elig: eligible for allocation (see "
        "register_allocator.py's own docstring for the two exclusions). "
        "alloc/spill: of those eligible, how many got a register vs. fell "
        "back to a memory slot. named/unsafe: excluded because they back "
        "a named variable, or because their live range spans an "
        "IRRaw/IRCall, respectively."
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
