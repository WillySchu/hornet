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
- Pages that become empty go straight back to the C library, so a program that keeps filling and emptying the heap allocates new pages each cycle. Keeping a few empty pages would help; worth deciding during step three’s memory-limit tests.
- Literal-only expressions. int8 x = 1 + 2 is still rejected, because only a single literal adapts to a narrow type.
- Stdlib for the port. Integer parsing and path helpers.
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
- `assert`
- `cap` builtin?
- Ternary.
- Sets.
- Generics.
- Sorting.
- Add format string literal.
- Slice equality?
- Dict equality?
- Variadic FFI calls?
- float
- Multithreading
- `gen_array_copy` currently copies each leaf one by one, so for small leaf types (int8, etc.) this is calling many `movb` rather than a few `movq`.
- Bounds Check Elimination.
- Interfaces.
- Error handling.
- Finding the stack top uses \_\_libc\_stack\_end, which is glibc only (musl would need the record-on-entry approach). On Windows it uses GetCurrentThreadStackLimits, which needs Windows 8 or later. macOS is untested.
- Explore graph coloring algorithm for register allocation (Chaitin-Briggs).
- Link to non libc functions for FFI?
- TLS.
- Call hornet code from C?
