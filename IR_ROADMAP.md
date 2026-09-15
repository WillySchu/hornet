Let me verify the current state of several claims directly against the code before writing this up, rather than relying purely on recollection.Here's a grounded status report — I checked each claim directly against the current code rather than relying on memory alone.

## What's done

Arrays, structs, and slices are real IR across: production, whole-value copying, indexing, `none`-comparison, `append`, `len`, both directions of the calling convention (arguments and returns, including nested literal construction and zero-value initialization). Strings are real IR for literals, concatenation, and comparison.

## What's left

**Tier 3 — needs real, new design work**

- **A composite-returning `Call` used directly as an addressable base** — `foo()[i]` (array or slice indexing), `foo().field` (struct field access), `foo()[a:b]` (slice production from a call's result). Verified: `_ir_indexable_base`'s own docstring still explicitly excludes Call for this reason, and the same exclusion propagates through `_ir_array_address`/`_ir_struct_address`. This is one root cause showing up in three places, not three separate problems — a composite-returning call has no address at all until the caller gives it one via the hidden-pointer convention, so using its result directly (without first assigning it to a named variable) needs the compiler to materialize it into a fresh scratch location first. Comparable in weight to the composite-calling-convention work.
- **Array/struct equality (`==`/`!=`) as real IR** — verified still explicitly deferred (`_ir_expr_binary`'s own docstring: "not migrated yet"). This needs field-by-field/element-by-element comparison with early-exit-on-mismatch branching — structurally closer to `append`'s branch-and-merge shape than to string comparison's straight line.

**Tier 4 — whole areas never started**

- **`print()`** — verified there's no `_ir_`-prefixed counterpart at all, anywhere. It dispatches on argument type (scalar vs. array/struct/slice each print differently) and was never touched in this entire engagement. This is a substantial, self-contained arc of its own.

## Proposed roadmap

3. Then a choice between **Call-as-addressable-base** and **array/struct equality** — both are genuinely separate, Tier-3-weight pieces deserving their own discussion-first scoping conversation, the way `append` and the calling convention did. I don't have a strong pull toward one over the other; Call-as-base closes a correctness/completeness gap across three related spots at once, while array/struct equality is a single, self-contained operator.
4. **`print()`** last, as its own dedicated arc — it's unrelated to the composite-value work and large enough to warrant a fresh scoping discussion of its own rather than folding it in as a follow-up to anything else.
