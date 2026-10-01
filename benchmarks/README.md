# Benchmarks

Programs in `programs/` for tracking how compiler changes affect generated code. `run_benchmarks.py` measures them; it is not part of `pytest`.

## Measurements

Per program:

- **instrs**: instruction lines in the emitted assembly (labels, directives, and comments excluded).
- **Register allocation** (`backend/common/regalloc.py`, summed over functions):
  - `temps`: all Temps.
  - `addr`: Temps excluded because their home slot's address is taken.
  - `elig`: the rest.
  - `alloc`/`spill`: eligible Temps that got a register or a frame slot.
  - `x-call`: eligible Temps live across a call, which may use only callee-saved registers.
- **time(ms)**: minimum wall-clock time over `--runs` runs (default 7). It is noisy; read it as a rough signal.
- **exec(M)**: millions of instructions executed, from valgrind's cachegrind. It is deterministic. Only with `--icount`, which needs a natively runnable target.

## Usage

```bash
python3 benchmarks/run_benchmarks.py                      # all programs, host target
python3 benchmarks/run_benchmarks.py tokenize branchy     # selected programs
python3 benchmarks/run_benchmarks.py --runs 0             # skip timing
python3 benchmarks/run_benchmarks.py --icount             # add executed-instruction counts
python3 benchmarks/run_benchmarks.py --target aarch64-linux
python3 benchmarks/run_benchmarks.py --json out.json      # also write results
python3 benchmarks/run_benchmarks.py --compare out.json   # diff against a results file
python3 benchmarks/run_benchmarks.py --save-baseline      # overwrite baseline.json
```

Without `--compare`, results are diffed against `baseline.json` if it exists. The diff shows the instruction delta, time %, allocated-Temp delta, and executed % (when both sides have it).

`baseline.json` doesn't record the machine or target it came from. Only compare it on the same host and target. Foreign targets run under qemu-user, so their times aren't comparable.

## Tests

`tests/test_benchmarks.py` builds each program for each end-to-end target and checks its exit code. Under `pytest --full` it also runs each binary 10 times, to catch ASLR-dependent faults. It also runs the runner and checks that its stats add up, and with valgrind present it tests `--icount`. Every `.ht` file here needs an expected exit code there. Results are exit codes, so they stay in 0–255.

## Programs

| Program | Exercises |
| --- | --- |
| `arithmetic_heavy` | Scalar arithmetic, unary operators, and narrowing casts in a hot loop; no calls |
| `array_heavy` | Bubble sort, indexing into bare literals, array equality |
| `branchy` | `if`/`elif` chains, short-circuit `and`/`or`, Collatz loops |
| `calling_convention_heavy` | Array, slice, and struct arguments; composite returns used directly |
| `calls_in_loop` | Locals live across calls to a leaf function |
| `copy_heavy` | Whole-array, struct, and slice copies; `append`; slicing |
| `dict_heavy` | `int`- and `str`-keyed insert, delete, lookup, and iteration |
| `loop_accumulator` | Scalar accumulation and per-iteration zero-initialization |
| `recursive_fibonacci` | Call-bound recursion |
| `register_pressure` | Twelve values live across a loop body |
| `string_heavy` | Repeated concatenation and string equality |
| `struct_heavy` | Field arithmetic and struct equality |
| `sum_type_walk` | Recursive `match` dispatch over a sum-typed tree |
| `tokenize` | Lexer-style byte scan over a 270 KB `str` |
