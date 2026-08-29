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
- Struct literals as function arguments.
- Struct literals as return values.
- Struct literals as field assigne values.
- Struct literals as index assigne values.
- Struct literals as bare statement.
- Struct literals nested in other literals.
- Named field construction for struct literals.
- `==` for structs.
- Add source location for error messages from semantic analysis.
- `fuction` type.
- Methods.
- Reconsider heap allocated literal arrays for `gen_indexable_base_into` (apparently `len([1, 2, 3])` and `[1, 2, 3][0]` work?)
- `assert`
- `in` keyword.
- Pass function calls to print.
- `cap` builtin?
- Compound index assign.
- Ternary.
- For loops.
- Slice / index an array return directly on function calls.
- Free memory `malloc`ed by string concatenation.
- Bounds Check Elimination.
- Stack based params to get around parameter limit.
- Pass array literals to function calls.
- Explore per frame vs per array heap promotion.
- Array equality.
- Slice equality?
- Escape analysis for arrays and slices.
- `type` Type?
- Dedpulicate emitted type descriptors when we call print()
- Variadic functions.
- Make `append` variadic.
- Spread operator.
- GC...
- `is` keyword.
- Pointers?
- FFI
