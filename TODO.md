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
- Nothing warns when a result is ignored, such as a bare write\_stdout(...) statement.
- Weight spills by use count and loop depth.
- Module errors have a line but no column.
- `in` checking for strings.
- Sum type equality.
- Spreading one package across multiple files.
- A package-style in-file declaration decoupling the module's name from its filename.
- Struct field/method-level visibility.
- Any import-path resolution strategy beyond "relative to the importing file" (a stdlib search path, a project manifest/root, etc.).
- is as a general, composable boolean expression — usable with and/or/not, assignable to bool, usable in while conditions or as a function argument. This is the biggest deferred item; it needs real control-flow-sensitive narrowing (what does if shape is Circle and x > 0: narrow? what survives an early return guard?), not just a special if-condition shape.
- Narrowing inside a while condition. Excluded by construction in v1 (only if is recognized), but worth its own tracked item since loop bodies raise questions v1 never has to answer — what does narrowing across iterations even mean once reassignment is disallowed anyway.
- Late argument reads. The check runs when the payload's address is computed. A composite argument is copied by the callee later, so foo(u, change(p)) can still pass a payload of the wrong variant. This is the late-read bug I reported with the first fixes; fixing that closes this window.
- A pointer into a payload that outlives the variant is untouched.
- Pass structs to FFI calls.
- Separate the runtime code from codegen.
- CSE pass optimization for array addresses and thus indexes.
- `type` type.
- `typeof`
- Consider enabling struct literal syntax for aliases.
- `function` type.
- User defined types built off other types.
- Disallow untyped array literals from anything except assignment to a typed variable / parameter / return.
- `assert`
- `cap` builtin?
- Enums.
- Ternary.
- Sets.
- Generics.
- Sorting.
- Format strings.
- Slice equality?
- Dict equality?
- Import \*?
- Borrow checker for modifying dicts / slices in for ... in ... loops?
- `for in` loops on function calls.
- Deduplicate emitted type descriptors when we call print()
- Variadic functions.
- Variadic FFI calls?
- Make `append` variadic.
- Spread operator.
- float
- Multithreading
- `gen_array_copy` currently copies each leaf one by one, so for small leaf types (int8, etc.) this is calling many `movb` rather than a few `movq`.
- Consider aligning struct storage rather than packing.
- Bounds Check Elimination.
- Interfaces.
- Error handling.
- GC...
- Free memory `malloc`ed by string concatenation.
- Explore graph coloring algorithm for register allocation (Chaitin-Briggs).
- Link to non libc functions for FFI?
- Call hornet code from C?
