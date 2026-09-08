Let me verify the exact current scope boundaries precisely before answering, especially the `append`/`len` exclusions, given the current state of `_ir_call` and `_ir_indexable_base`.Here's a full accounting of what remains, grounded in the exact scope boundaries we built and deferred throughout this arc:

## 1. Slice-typed `Field`/`Index` as an indexing or slicing base — the most natural next step

`_ir_indexable_base`'s `SLICE` branch currently requires `isinstance(expr, Variable)` exclusively (verified directly — line 417 returns `None` for anything else). So `p.values[0]` (a slice-typed *field*, then indexed) or `rows[i][0:3]` (a slice-typed *array element*, then sliced) both still fall back to old-style entirely, even though the *array*-typed branch already handles `Field`/`Index` bases natively.

This asymmetry exists because an array base just needs its own address (already generic via `_ir_array_address`), while a slice base needs its three descriptor *fields read out* from that address — architecturally different, but not the hard case. `_ir_field_address`/`_ir_index_address` already compute "the address of this `Field`/`Index` expression" generically, regardless of the result's own type. Reusing that to get a slice descriptor's address, then reading `ptr`/`len`/`cap` off it via three `IRLoad`s (the same shape the `Variable` leaf already uses, just through a computed rather than fixed address), looks like a genuinely incremental widening — not new machinery.

## 2. `Slice`/`Call` as an indexing or slicing base — the genuinely hard, always-deferred piece

Indexing or re-slicing into an *unnamed or freshly-computed* slice expression (`arr[:][0]`, a slice-returning call immediately indexed) still needs `gen_indexable_base_into`'s shared-scratch-slot materialization — compute the value, write it somewhere real, read it back. This is the piece we've explicitly deferred since the very first address-computation discussion, and it remains the one genuinely new design problem left in this arc, not a mechanical extension of what exists.

## 3. `append()` as real IR — untouched so far

`_ir_call`'s dispatch excludes any call whose return type is `ARRAY`/`SLICE`/`STRUCT`. Since `append()` always returns a slice, it's excluded unconditionally and has never been touched by any of this work — still entirely old-style, including its own realloc/growth logic. This is squarely a "slices as IR" concern that's still fully outstanding.

## 4. `len()` as real IR — also untouched, likely cheap

`_ir_call` explicitly excludes `len` by name (alongside `print`), regardless of argument type. For an array or slice argument that's `Variable`/`Field`/`Index`, this looks like the cheapest item on this list — the underlying value is either a compile-time constant or something `_ir_indexable_base` already computes.

## 5. Slice function arguments and returns — a broader, pre-existing boundary

Not slice-specific — the same boundary already exists for array/struct arguments (`_ir_call`'s fallback) and composite returns (hidden-output-pointer convention, still old-style for all three kinds). Slices just haven't been revisited any more than arrays/structs have here.

## 6. Slice comparison to `none` (`s == none`) — not yet discussed

Never came up in this arc. Almost certainly still fully old-style (`IRRaw`-wrapped) today.

## 7. No-initializer slice `VarDecl` (implicit nil) — shared, not slice-specific

Same "needs composite-aware zero-init" exclusion that's always applied uniformly to array/struct/slice alike, from the very first IR-native step.

## 8. A verification gap worth naming honestly

Slicing into a *computed sub-array* location (`matrix[i][a:b]`) should already work — it falls out of the same recursive `_ir_array_address`/`_ir_index_address` structure we verified for multi-dimensional indexing. But I haven't explicitly tested this specific shape the way we tested plain multi-dimensional indexing, so I'd call it "probably already correct" rather than "confirmed," given how this arc has gone.

If you want my read on ordering: #1 is the natural, contained next step (proven machinery, narrow widening); #3/#4 are self-contained and clearly in-scope for "slices as IR"; #2 is the one that actually needs new design thinking, similar in weight to the original address-computation conversation.
