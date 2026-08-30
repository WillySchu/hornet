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
- Struct literals as bare statement.
- Unnamed struct field access.
- Disallow untyped array literals from anything except assignment to a typed variable / parameter / return.
- Add source location for error messages from semantic analysis.
- `fuction` type.
- Reconsider heap allocated literal arrays for `gen_indexable_base_into` (apparently `len([1, 2, 3])` and `[1, 2, 3][0]` work?)
- Implicit zero value initialization for uninitialized variables / fields.
- `assert`
- `in` keyword.
- `cap` builtin?
- Compound index assign.
- Ternary.
- For loops.
- Dicts.
- Slice / index an array return directly on function calls.
- Free memory `malloc`ed by string concatenation.
- Bounds Check Elimination.
- Stack based params to get around parameter limit.
- Explore per frame vs per array heap promotion.
- Slice equality?
- Escape analysis for arrays and slices.
- `type` Type?
- Dedpulicate emitted type descriptors when we call print()
- Variadic functions.
- Make `append` variadic.
- Spread operator.
- GC...
- `is` keyword.
- FFI
- Sum types / pattern matching.
- Register virtualization
- int64
- float
- Pointers
- Multithreading
- Optimization / mid level IR
