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
- Structs.
- Fix print for collections to build str buffer and call print once.
- Escape arrays backing arrays / slices assigned via indexing, e.g. `matrix[i][j] = arr[:]`.
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
- Variadic functions.
- Make `append` variadic.
- Spread operator.
- GC...
- `is` keyword.
- Pointers?
