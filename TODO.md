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
- `IRBranch` peephole optimization to skip redundant unconditional jumps.
- CSE pass optimization for array addresses and thus indexes.
- `type` type.
- ` typeof`
- Require `type` keyword to declare new type for structs.
- Consider enabling struct literal syntax for aliases.
- `fuction` type.
- Add `byte` literal.
- User defined types built off other types.
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
- Flow sensitive escape analysis.
- Dedpulicate emitted type descriptors when we call print()
- Variadic functions.
- Make `append` variadic.
- Spread operator.
- `is` keyword.
- FFI
- Fix strings to basically be byte slices.
- Sum types / pattern matching.
- int32
- Change int to an alias.
- IRAddress
- Pointers
- float
- Imports.
- Multithreading
- Optimization / mid level IR
- Consider adding a semantic AST with type information rather than annotating the existing AST from the parser.
- Consider adding a desugar step in between the parser and the semantic analysis (semantic analysis currently handles method desugaring, which is a little awkward).
- Bounds Check Elimination.
- GC...
- Free memory `malloc`ed by string concatenation.
- Explore graph coloring algorithm for register allocation (Chaitin-Briggs).
