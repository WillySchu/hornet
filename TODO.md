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
- Type identity for the port. Composite dict keys, or structural equality through pointers. This is the decision the checker's port depends on.
- Narrowing. Fields and elements still need an as binding, and as cannot bind in a while condition.
- Enums. Explicit member values, ordering, for ... in over an enum.
- match exhaustiveness. It does not take earlier exclusions into account, so a variant already ruled out by a guard must still have an arm or an else.
- Report function name on stack overflow.
- Literal-only expressions. int8 x = 1 + 2 is still rejected, because only a single literal adapts to a narrow type.
- No mutable globals. Worth recording, since the port will need context structs because of it.
- Stdlib for the port. Integer parsing and path helpers.
- Pointers into a sum's payload.
- &d[k] after a rehash.
- Nothing warns when a result is ignored, such as a bare write\_stdout(...) statement.
- Weight spills by use count and loop depth.
- `in` checking for strings.
- Spreading one package across multiple files.
- A package-style in-file declaration decoupling the module's name from its filename.
- Any import-path resolution strategy beyond "relative to the importing file" (a project manifest/root, etc.).
- Pass structs to FFI calls.
- CSE pass optimization for array addresses and thus indexes.
- `type` type.
- `typeof`
- Consider enabling struct literal syntax for aliases.
- `function` type.
- User defined types built off other types.
- Disallow untyped array literals from anything except assignment to a typed variable / parameter / return.
- `assert`
- `cap` builtin?
- Ternary.
- Sets.
- Generics.
- Sorting.
- Add format string literal.
- Slice equality?
- Dict equality?
- Import \*?
- Borrow checker for modifying dicts / slices in for ... in ... loops?
- `for in` loops on function calls.
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
- TLS.
- Call hornet code from C?
