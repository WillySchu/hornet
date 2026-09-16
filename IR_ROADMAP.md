## What's left

Let me ground this in the actual current code rather than relying purely on memory, since some scope boundaries may have shifted as side effects of later work. Let me systematically check every remaining fallback point.This surfaces something important I hadn't re-flagged recently — `_ir_append_call` still excludes composite element types entirely (`[]Point`, `[][]int`), from the very original append work. Let me check `structs.py` and `statements.py` for any other distinct fallback points before compiling the full list.Let's verify one more important distinct gap — whether composite (`ArrayLiteral`/struct-literal/`Call`) function-call *arguments* still route through old-style materialization, despite the underlying real-IR machinery now existing.This confirms a real, distinct gap — `_ir_call_arguments` never got updated to try `_ir_array_address`/`_ir_struct_address` for a `Call` argument, even though those functions now handle `Call` via `_ir_materialize_composite_call`. I now have a complete, grounded picture. Let me organize this by root cause.

## What's left in the deliberately-scoped-out areas

Everything below traces back to a small number of distinct root causes, not a long list of unrelated gaps — grouping by that, rather than by call site, since several of these will likely resolve together.

**1. Slice-literal construction as real IR** (`[]int s = [1, 2, 3]`, or the inner `[1, 2]` within an array-of-slices literal) — confirmed still entirely unbuilt, anywhere in this arc. This is distinct from slice *production* (`arr[a:b]`, already real IR) and from array-literal construction (already real IR) — it's specifically "a bracketed list, resolved to `SLICE` by context, backed by a fresh allocation." This is the root cause behind several of the remaining fallback points in `_ir_write_composite_value_into`.

**2. An `ArrayLiteral`/struct-literal `Call` used directly as an addressable base** (`[1,2,3][0]`, `Point(1,2).x`, `[1,2,3][0:2]`) — distinct from the *ordinary* composite-returning-`Call`-as-base work already done. `_ir_indexable_base`/`_ir_array_address`/`_ir_struct_address` all explicitly exclude this today; it would need the same kind of materialization `_ir_materialize_composite_call` already does for an ordinary call, just triggered for a literal instead.

**3. Composite arguments still don't reuse the newer real-IR machinery at all** — I want to flag this one clearly since I hadn't re-surfaced it recently: `_ir_call_arguments`'s own `ArrayLiteral`/struct-literal/composite-returning-`Call` argument case still falls back entirely to old-style `_gen_materialize_argument_temp_into`, wrapped in `IRRaw`. This is despite `_ir_array_address`/`_ir_struct_address` now handling a `Call` operand directly (from the addressable-base work) and `_collect_argument_temps` already reserving the exact slot this would need — the argument-passing path just never got wired up to try them. This is likely the *cheapest* item on this whole list, since it's pure reuse of existing pieces, not new design.

**4. `append` on a slice of composites** (`[]Point`, `[][]int`) is still entirely old-style — `_ir_append_call` explicitly excludes any composite element type, a deliberate scope boundary from the very first append work, not something that crept in later.

**5. Array/struct equality where an operand is an `ArrayLiteral` or composite-returning `Call`** directly (not assigned to a variable first) — the boundary I deliberately kept in the equality work just now, matching the old code's own existing restriction rather than widening it as a side effect of the addressable-base work.

**6. A `Slice`-valued `ExprStmt` whose own base is out of scope for `_ir_slice_into`** — the same root cause as #1/#2, in a narrow, low-value position.

## My honest read on sizing and sequencing

\#3 is genuinely small — wiring, not design, given everything it needs already exists. #2 is a direct extension of already-proven work (`_ir_materialize_composite_call`), just triggered from a literal instead of a call — also fairly contained. #4 and #5 are narrow, self-contained extensions of existing dispatchers. #1 is the one genuinely new piece of design in this list — slice-literal construction has never been built at all, and would need its own "materialize a fresh backing allocation, write elements into it, produce the descriptor" real-IR treatment, distinct from everything else here. #6 is low-value enough that I'd bundle it into whichever of the above happens to touch that code path.

Given #3's size and the fact that closing it doesn't require any new design at all, I'd suggest starting there if you want the cheapest win, or #1 first if you'd rather clear the one piece that's genuinely new work before the others. Let me know which you'd like to tackle, or if you'd rather do a subset and leave the rest for later.

**Tier 4 — whole areas never started**

- **`print()`** — verified there's no `_ir_`-prefixed counterpart at all, anywhere. It dispatches on argument type (scalar vs. array/struct/slice each print differently) and was never touched in this entire engagement. This is a substantial, self-contained arc of its own.

## Proposed roadmap

4. **`print()`** last, as its own dedicated arc — it's unrelated to the composite-value work and large enough to warrant a fresh scoping discussion of its own rather than folding it in as a follow-up to anything else.
