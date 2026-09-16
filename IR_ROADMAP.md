
**1. `append` on a slice of composites** (`[]Point`, `[][]int`) is still entirely old-style — `_ir_append_call` explicitly excludes any composite element type, a deliberate scope boundary from the very first append work, not something that crept in later.

**2. Array/struct equality where an operand is an `ArrayLiteral` or composite-returning `Call`** directly (not assigned to a variable first) — the boundary I deliberately kept in the equality work just now, matching the old code's own existing restriction rather than widening it as a side effect of the addressable-base work.

**3. A `Slice`-valued `ExprStmt` whose own base is out of scope for `_ir_slice_into`** — the same root cause as #1/#2, in a narrow, low-value position.

**4. — print()**

- **`print()`** — verified there's no `_ir_`-prefixed counterpart at all, anywhere. It dispatches on argument type (scalar vs. array/struct/slice each print differently) and was never touched in this entire engagement. This is a substantial, self-contained arc of its own.
