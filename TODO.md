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
Mutable []byte ↔ str conversion.
FFI/extern string interop design.
String slicing (s[a:b]).
String indexing (s[i] → a single byte).
- is as a general, composable boolean expression — usable with and/or/not, assignable to bool, usable in while conditions or as a function argument. This is the biggest deferred item; it needs real control-flow-sensitive narrowing (what does if shape is Circle and x > 0: narrow? what survives an early return guard?), not just a special if-condition shape.
- else-branch narrowing — even the "exactly two variants, so else means the other one" case.
- Narrowing inside a while condition. Excluded by construction in v1 (only if is recognized), but worth its own tracked item since loop bodies raise questions v1 never has to answer — what does narrowing across iterations even mean once reassignment is disallowed anyway.
- Cleanup remaining IR/backend coupling.
- Pass structs to FFI calls.
- Separate the runtime code from codegen.
- `IRBranch` peephole optimization to skip redundant unconditional jumps.
- CSE pass optimization for array addresses and thus indexes.
- `type` type.
- `typeof`
- Consider enabling struct literal syntax for aliases.
- `function` type.
- Add `byte` literal.
- User defined types built off other types.
- Change slice's zero type to an empty slice rather than `none`.
- Disallow untyped array literals from anything except assignment to a typed variable / parameter / return.
- `assert`
- `in` keyword.
- `cap` builtin?
- Ternary.
- For loops.
- Hashing.
- Dicts.
- Sets.
- Imports / packages.
- Generics.
- Sorting.
- Format strings.
- Slice equality?
- Flow sensitive escape analysis.
- Deduplicate emitted type descriptors when we call print()
- Variadic functions.
- Variadic FFI calls?
- Make `append` variadic.
- Spread operator.
- Fix strings to basically be byte slices.
- int32
- Change int to an alias.
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
- Link to non libc functions for FFI?
- Call hornet code from C?
