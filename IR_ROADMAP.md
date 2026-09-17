
**1. Bare VarDecl for scalars.

**2. — print()**

- **`print()`** — verified there's no `_ir_`-prefixed counterpart at all, anywhere. It dispatches on argument type (scalar vs. array/struct/slice each print differently) and was never touched in this entire engagement. This is a substantial, self-contained arc of its own.
