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

Documentation:
- Clean up test\_compiler.py, the docstring of which is quite stale.

Updates:
- Cleanup remaining IR/backend coupling.
- FFI
- Separate the runtime code from codegen.
- `IRBranch` peephole optimization to skip redundant unconditional jumps.
- IRAddress
- CSE pass optimization for array addresses and thus indexes.
- `type` type.
- `typeof`
- Consider enabling struct literal syntax for aliases.
- `fuction` type.
- Add `byte` literal.
- User defined types built off other types.
- Change slice's zero type to an empty slice rather than `none`.
- Disallow untyped array literals from anything except assignment to a typed variable / parameter / return.
- Reconsider heap allocated literal arrays for `gen_indexable_base_into` (apparently `len([1, 2, 3])` and `[1, 2, 3][0]` work?)
- Add source location for error messages from semantic analysis.
- `assert`
- `in` keyword.
- `cap` builtin?
- Compound index assign.
- Ternary.
- For loops.
- Dicts.
- Stack based params to get around parameter limit.
- Slice equality?
- Flow sensitive escape analysis.
- Deduplicate emitted type descriptors when we call print()
- Variadic functions.
- Make `append` variadic.
- Spread operator.
- `is` keyword.
- Fix strings to basically be byte slices.
- Sum types / pattern matching.
- int32
- Change int to an alias.
- Pointers
- float
- Imports.
- Multithreading
- `gen_array_copy` currently copies each leaf one by one, so for small leaf types (int8, etc.) this is calling many `movb` rather than a few `movq`.
- Consider aligning struct storage rather than packing.
- Consider adding a semantic AST with type information rather than annotating the existing AST from the parser.
- Bounds Check Elimination.
- Error handling.
- GC...
- Free memory `malloc`ed by string concatenation.
- Explore graph coloring algorithm for register allocation (Chaitin-Briggs).
- Consider renaming CodeGenerator to something like X86Backend.
- Target different architectures?
