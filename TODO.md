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
- Consider adding a semantic AST with type information rather than annotating the existing AST from the parser.
- Printing a struct defined in another module shows its internal name: print(write\_file(bad, x)) prints errors$Error(message: …).
- Nothing warns when a result is ignored, such as a bare write\_stdout(...) statement.
- Struct fields are resolved before sum types, so a sum type used as a field type is unknown at that point.
- Weight spills by use count and loop depth.
- Module errors have a line but no column.
- Lexer line number shifts with string literal containing a raw newline.
- hornet\_panic currently writes to stdout, not stderr.
- Add symbol table to replace IR builder's use if id() as declaration keys.
- Dicts as sum types.
- `in` checking for strings.
- Sum type equality.
- Sum type support for nested sum types?
- Spreading one package across multiple files.
- A package-style in-file declaration decoupling the module's name from its filename.
- Struct field/method-level visibility.
- Any import-path resolution strategy beyond "relative to the importing file" (a stdlib search path, a project manifest/root, etc.).
- FFI/extern string interop design.
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
- User defined types built off other types.
- Change slice's zero type to an empty slice rather than `none`?
- Change dict's zero type to an empty slice rather than `none`?
- Disallow untyped array literals from anything except assignment to a typed variable / parameter / return.
- `assert`
- `cap` builtin?
- Ternary.
- Sets.
- Generics.
- Sorting.
- Format strings.
- Write through addressing for dicts (d[key].field = value)?
- Slice equality?
- Dict equality?
- Import \*?
- Borrow checker for modifying dicts / slices in for ... in ... loops?
- Flow sensitive escape analysis.
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
