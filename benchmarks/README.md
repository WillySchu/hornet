# Benchmarks

Six `.ht` programs, checked in specifically to keep "how is a change
actually affecting generated code" from staying opaque -- which was
fine while nothing in the compiler cared about performance, but stops
being fine now that `codegen/register_allocator.py` exists.

## What's measured

For each program, `run_benchmarks.py`:

- **Instruction count** -- a simple, deterministic proxy for
  generated-code size (counts real instruction lines in the emitted
  assembly, skipping labels/directives/comments).
- **Register-allocation stats** -- for every `Temp` created while
  compiling the program: how many total, how many were eligible for
  allocation, how many actually got a register vs. spilled to memory,
  and *why* the ineligible ones were excluded (backs a named variable,
  vs. its live range spans an `IRRaw`/`IRCall` -- see
  `register_allocator.py`'s own module docstring for what those two
  exclusions mean and why they exist).
- **Wall-clock runtime** -- the minimum of several runs of the actual
  compiled-and-linked binary. Noisier than the other two, and the one
  number here that isn't perfectly reproducible run to run -- read it
  as a rough signal, not an exact figure.

## Usage

```
python3 benchmarks/run_benchmarks.py                  # run and print a report
python3 benchmarks/run_benchmarks.py --save-baseline   # also overwrite baseline.json
```

With no flag, if `baseline.json` already exists, the report includes a
diff against it (instruction count delta, runtime % change, allocated-
Temp count delta) for each program. Run with `--save-baseline` to
accept the current numbers as the new baseline -- do this deliberately,
after confirming a change's effect on these numbers is the one you
intended, not as a routine part of every commit.

This script is deliberately **not** part of the regular `pytest`
run: these programs are sized to take tens to hundreds of milliseconds
each specifically so the timing measurement means something, and
running them (several times each, for the minimum) on every test run
would make the suite noticeably slower for a signal that's only
useful when you're actually asking a performance question.

## Keeping these honest

`tests/test_benchmarks.py` *is* part of the regular suite -- it just
compiles and runs each program once, with no timing, and asserts the
already-known-correct exit code. That's what stops a benchmark program
from silently drifting into testing something else (or nothing) as the
language evolves, without needing to run the slower, noisier
measurement script just to catch it.

## The programs

Each targets a different corner of what the register allocator can
and can't currently see, on purpose -- see the design discussion this
suite came out of for the full reasoning, but briefly:

- `arithmetic_heavy.ht` -- pure scalar arithmetic, no calls, no
  composite types. The allocator's actual sweet spot today.
- `recursive_fibonacci.ht` -- call-bound rather than named-variable-
  bound; almost every `Temp`'s live range crosses an `IRCall`.
- `loop_accumulator.ht` -- dominated by named-variable reads/writes
  (the accumulator, the loop counter), which are unconditionally
  excluded from allocation today regardless of anything else (see
  `Temp.is_named_local`'s own docstring for why).
- `array_heavy.ht`, `struct_heavy.ht`, `string_heavy.ht` -- each
  exercises a feature area that's still entirely `IRRaw`-wrapped old-
  style codegen, essentially untouched by the allocator at all. These
  exist as the "how much of a real program is the allocator currently
  blind to" control group, as much as anything else.

Every program's expected result was checked against an independent,
pure-Python reference computation, not just accepted from a first
compile -- and every return value is deliberately kept within 0-255,
since process exit codes are truncated to a single byte regardless of
what the Hornet program itself computes or returns.
