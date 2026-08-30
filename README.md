Hornet Lang
-----------

```
def int main():
    print('Hello World!')
    return 0
```

TODO:

Binary Ops:
- Consider replacing ^ with ~
- Exponentiation (either ** or ^)

Updates:
- Add zero value initialization for named tuple declarations.
- Add `byte` type.
- Casting.
- `type` type.
` typeof`
- `fuction` type.
- Change slice's zero type to an empty slice rather than `none`.
- Struct literals as bare statement.
- Unnamed struct field access.
- Disallow untyped array literals from anything except assignment to a typed variable / parameter / return.
- Add source location for error messages from semantic analysis.
- Reconsider heap allocated literal arrays for `gen_indexable_base_into` (apparently `len([1, 2, 3])` and `[1, 2, 3][0]` work?)
- `assert`
- `in` keyword.
- `cap` builtin?
- Compound index assign.
- Ternary.
- For loops.
- Dicts.
- Slice / index an array return directly on function calls.
- Stack based params to get around parameter limit.
- Explore per frame vs per array heap promotion.
- Slice equality?
- Escape analysis for arrays and slices.
- Dedpulicate emitted type descriptors when we call print()
- Variadic functions.
- Make `append` variadic.
- Spread operator.
- `is` keyword.
- FFI
- Fix strings to basically be byte slices.
- Sum types / pattern matching.
- Register virtualization
- int64
- float
- Pointers
- Multithreading
- Optimization / mid level IR
- Bounds Check Elimination.
- GC...
- Free memory `malloc`ed by string concatenation.
