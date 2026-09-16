## What's left

**Tier 4 — whole areas never started**

- **`print()`** — verified there's no `_ir_`-prefixed counterpart at all, anywhere. It dispatches on argument type (scalar vs. array/struct/slice each print differently) and was never touched in this entire engagement. This is a substantial, self-contained arc of its own.

## Proposed roadmap

4. **`print()`** last, as its own dedicated arc — it's unrelated to the composite-value work and large enough to warrant a fresh scoping discussion of its own rather than folding it in as a follow-up to anything else.
