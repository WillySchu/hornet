Let me ground this in the actual current state of the code rather than working from memory, given how much subtlety this project has already surfaced.Let me lay this out properly, grounded in what I just verified rather than what I remember.

## What "clean boundary" actually means here

A few concrete properties, not just a slogan:

1. **No escape hatch.** As long as `IRRaw` exists as a general "wrap arbitrary already-emitted instructions" mechanism, the boundary isn't closed — it doesn't matter how much of the *common* path goes through real IR ops, because anything can still smuggle raw instructions through unexamined.
2. **IR construction and IR consumption are genuinely separate phases**, not one interleaved pass. We already have the *good half* of this — `gen_function` builds a function's whole IR list before allocation or lowering ever touches it. What we don't have is the *other* half: building that list currently means calling into instruction-selection-flavored code (`gen_expr_into`, `gen_statement`) mid-construction, every time something isn't yet migrated.
3. **Every source-level construct has an IR shape.** Right now that's true for scalars and false for arrays, structs, slices, and strings.
4. **Optimization passes can operate on IR alone**, with no knowledge of the source AST or the target architecture. This is really a *consequence* of properties 1–3, not a separate one — you can't write a pass over data that's opaque half the time.

## Where things actually stand (verified just now, not from memory)**Real IR today**: scalar arithmetic/comparisons, short-circuit and/or, named-local scalar variables, scalar function calls (including array/struct arguments as address-`Temp`s), scalar returns, if/while heads, and single-scalar array/struct element access through an address captured as a `Temp`. Ten `_ir_*` builder methods total, all in `dispatch.py`, `scalars.py`, `statements.py`.

**Entirely untouched by IR, still pure old-style codegen wrapped in `IRRaw`**: arrays (literals, bulk copy, bounds-checked indexing beyond one scalar), structs (copy, literal construction), slices (production, arguments, returns — the whole value, not just element access), strings (concatenation, comparison, indexing), and address *computation* itself (only its *result* is captured as a `Temp` — the computation stays opaque). Escape analysis operates purely on the AST and has no IR involvement at all.

## What it would take — high level

**Eliminate `IRRaw` as a general escape hatch.** This isn't really "finish migrating everything to IR" as one lump — it splits into pieces with genuinely different character:
- *Composite value operations* (array/struct copy) — mechanical, low-risk, same address-as-`Temp` pattern already proven by `IRLoad`/`IRStore`.
- *Address computation itself* — a bigger step than it sounds: today the compiler treats "compute `&arr[i]`" as a black box and only "read/write through it" as real IR. Making the computation inspectable means modeling bounds-check branches and offset arithmetic as IR, not just capturing whatever assembly falls out.
- *Slices* — the hard one. A slice isn't reducible to one value; it's three (ptr/len/cap), which doesn't fit anywhere in the current IR shape without a genuinely new concept.
- *Strings* — untouched by any of this so far; concatenation in particular looks more like a call to an implicit runtime helper than an arithmetic op.

**Separate "build IR" from "select instructions" as actual phases**, not just an ordering convention. Right now they're interleaved — building a fragment can require calling into instruction-selection code and wrapping the result. Closing this means every `_ir_X` builder produces *only* `ir.py` nodes, never touches `Instruction` directly, for every construct — which is really just "no more `IRRaw`" restated from the construction side rather than the consumption side.

**Let escape analysis feed the allocator.** This is structurally independent of everything above — no composite-IR prerequisite at all. Named-local `Temp`s are unconditionally excluded from allocation today because old-style code can still read a variable's memory slot directly, bypassing the `Temp`. A pass answering "does this variable's address ever escape to old-style code" would let non-escaping locals — loop counters, accumulators — become eligible. This is probably the single highest-leverage, lowest-prerequisite piece of everything on this list.

**Push the mixin decomposition further.** `InstructionSelector` still depends on `host` for four leaf methods (`gen_binary_op`, `gen_unary_op`, `_gen_read_scalar_into`, `_gen_write_scalar_from`) and two shared stack-allocation attributes. Genuine package extraction — `ir.py`/`ir_lowering.py`/`register_allocator.py` moving out of `codegen/` to sit alongside `lexer.py`/`parser.py`/`semantic.py` — needs that dependency to become an explicit, narrow interface rather than "the whole `CodeGenerator`." This is orthogonal to IR *coverage* — it's about IR *consumption* being independently packageable — and there's limited urgency until something else (an optimizer) actually wants to import that package on its own.

**Optimization passes become possible, not just faster.** This was the actual payoff flagged earlier (the `arr[i]` address-CSE opportunity) — but it *requires* address computation being real, inspectable IR first. Right now a pass has nothing to look at there; the computation is opaque assembly inside `IRRaw`.

## What it would take — low level

Concretely, in rough order of how naturally each depends on what came before:

1. **Escape analysis → allocator eligibility.** `escape_analysis.py` gains a pass answering per-variable "does its address ever escape," `eligible_intervals` gains a rule lifting the named-local exclusion when it doesn't. No new IR nodes needed at all.
2. **`IRCopy(dst_address, src_address, size)`** for whole-array/whole-struct assignment between two addresses. Reuses the address-as-`Temp` pattern directly.
3. **Address computation as real IR** — something like `IRFieldAddr`/`IRIndexAddr` ops modeling offset arithmetic and bounds checks, replacing the `IRRaw`-wrapped `gen_field_address_into`/`gen_index_address_into`. This is the prerequisite for the CSE optimization, and for slices being built on the same foundation rather than their own bespoke thing.
4. **Slice representation** — a bundled three-`Temp` (or new `IRValue` variant) value shape, covering slice literals, slicing expressions, parameters, and returns. This is what would let `_ir_call` finally drop its old-path fallback entirely.
5. **String operations as IR** — independent of slices, similar difficulty; concatenation likely needs an `IRCall`-shaped op to an implicit runtime helper rather than an arithmetic-style op.
6. **Mixin decomposition, next layer** — pull `gen_binary_op`/`gen_unary_op`/`_gen_read_scalar_into`/`_gen_write_scalar_from` behind their own narrow interface so `InstructionSelector` stops depending on `host` as a whole.

## My honest read on sequencing

Item 1 (escape analysis) stands alone — no prerequisites, real performance payoff, doesn't touch any of the "is this worth it" tension composite-IR work has. I'd do that regardless of which direction you want to go next. Items 2–5 only make sense as a *deliberate* investment in closing the boundary itself, since — as we found last time — none of them pay for themselves in speed. Item 6 only becomes urgent once something wants to consume the IR layer independently of `CodeGenerator` — which an optimizer, notably, would.

Where would you like to go — escape analysis first, or straight into scoping one of the boundary-closing pieces?
