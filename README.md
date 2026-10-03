# Hornet

Hornet is an experimental, statically typed programming language with an indentation-based syntax and a native compiler targeting x86-64 and AArch64 on Linux and macOS.

The language is intentionally small, but the implementation includes a real compiler pipeline, a standalone intermediate representation, an optimizing native backend, a small runtime, modules, dictionaries, pointers, tagged sum types, pattern matching, and a C-compatible FFI.

Hornet source files use the `.ht` extension.

## A Small Example

```hornet
def int main():
    for int i = 0; i < 10; i += 1:
        print(i)

    return 0
```

## Building

### Requirements

The compiler itself is written in Python and has no third-party Python dependencies. The tests need `pytest`; `pytest-xdist` (`pytest -n auto`) is optional.

To build a runnable native executable, you also need a C compiler/linker. The project uses `gcc` in its build and test tooling.

The native backends target x86-64 and AArch64 on Linux and macOS. On macOS the architecture is chosen with `gcc -arch`, so an Apple Silicon Mac can also build x86-64 programs and run them under Rosetta 2.

Building for a foreign Linux architecture needs a cross toolchain named `<arch>-linux-gnu-gcc`, and running the result needs qemu-user (on Debian/Ubuntu: `apt install gcc-aarch64-linux-gnu qemu-user`). The test suite uses them when present and skips what they're needed for otherwise.

### Build an Executable

The recommended way to build a runnable Hornet program is:

```bash
python3 build.py program.ht -o program
./program
```

`build.py` compiles the Hornet source, compiles the bundled native runtime, and links the two together into an executable. The compiled runtime is cached per target under `$HORNET_CACHE_DIR` (default `~/.cache/hornet`).

The target is written `arch-os`: `x86_64-linux`, `x86_64-macos`, `aarch64-linux`, or `aarch64-macos`.

```bash
python3 build.py program.ht --target x86_64-linux -o program
```

Without `--target`, the host is used.

### Generate Assembly

`compile.py` stops after assembly generation and is useful when inspecting the compiler's output:

```bash
python3 compile.py program.ht --target x86_64-linux -o program.s
```

Without `-o`, assembly is written to standard output:

```bash
python3 compile.py program.ht
```

A generated program that uses runtime functions such as `print` must also be linked with `runtime/runtime.c`. `build.py` handles this automatically.

`--dump-typed` prints the program's typed tree (see [Compiler Architecture](#compiler-architecture)) instead of assembly.

### Errors

Compile errors are reported as `file:line:col: error: message` with the source line and a caret; several semantic errors can be reported in one run. A program built into an executable must define `main`. Both tools exit with status 1 for errors in the program and 2 for internal compiler errors. `--traceback` shows the Python traceback instead.

### Testing

Run the test suite from the repository root:

```bash
pytest           # unit tests, plus end-to-end tests on the host target
pytest --quick   # unit tests only; nothing is built or run
pytest --full    # every runnable target, plus slow tests (benchmark reruns, formatter fuzzing, sanitizers)
```

The tests cover the lexer, parser, semantic analysis, modules, IR construction and verification, optimization, the native backend, escape analysis, runtime behavior, and end-to-end compiled programs, including seeded random programs checked against a Python model.

Backend-specific tests live in `tests/backend/<arch>/` and shared backend tests in `tests/backend/common/`. With `--full`, end-to-end programs are built and run for every target that can run on the machine (natively, under Rosetta 2, or under qemu-user), and every such target must produce the same output. `HORNET_E2E_TARGETS` overrides the targets in any tier:

```bash
HORNET_E2E_TARGETS=x86_64-linux,aarch64-linux pytest
```

### Formatting

`tools/hfmt` is the Hornet source formatter, itself written in Hornet:

```bash
python3 build.py tools/hfmt/main.ht -o hfmt
./hfmt file.ht other.ht        # format in place (files are written only if they change)
./hfmt --check file.ht         # list files that would change; exit 1 if any
./hfmt < file.ht               # format stdin to stdout (also `-` as a file name)
./hfmt --tokens file.ht        # dump tokens
```

It keeps line breaks and normalizes the rest: 4-space indentation (plus 4 per open bracket on continuation lines), canonical spacing around operators, commas, and colons, two spaces before an inline comment and one after its `#`s (`##header` becomes `## header`), at most one blank line in a row, and exactly one blank line around multi-line top-level definitions (consecutive one-line declarations stay together). A list whose closing bracket is on its own line gets a trailing comma; one closed on the same line loses it. Flags may appear anywhere among the arguments. The repository's own `.ht` files are kept formatted by the test suite.

### Benchmarks

```bash
python3 benchmarks/run_benchmarks.py            # time every benchmark
python3 benchmarks/run_benchmarks.py --icount   # also count executed instructions (needs valgrind)
python3 benchmarks/run_benchmarks.py --json out.json --compare benchmarks/baseline.json
```

Each run prints a table of results, then a table of changes against `benchmarks/baseline.json` (or the `--compare` file). See `benchmarks/README.md`.

---

# Language Overview

Hornet currently provides:

* Functions and recursion
* Static typing with explicit integer conversions
* Block scoping and shadowing
* `int` (64-bit), `int32`, `int8`, `uint8`, `byte`, `bool`, and `str`
* Fixed-size arrays, sized by literals or constants
* Slices
* Nominal structs, methods, and pointer receivers
* Type aliases and compile-time constants
* Single-level pointers
* Tagged sum types, including a payload-free `none` variant and recursive types
* `if`, `elif`, `else`, and `match`
* `while` loops
* C-style `for` loops
* `for ... in ...` iteration over arrays, slices, dictionaries, and strings
* `break` and `continue`
* Dictionaries with hashing, deletion, membership, and iteration; copies share one table
* Explicit integer casts
* Arithmetic, comparison, logical, bitwise, and membership operators
* Compound assignment
* Runtime bounds checking
* Built-ins such as `print`, `len`, `append`, `del`, and `bytes`, and `str(...)` conversions
* Modules and imports
* External C functions through `extern`
* A small fixed set of compiler-defined `intrinsic` functions

The language deliberately exposes useful low-level representations instead of hiding everything behind an object system. Arrays, slices, structs, pointers, dictionaries, sum types, and foreign functions all have explicit rules for layout and interoperability.

---

# Comments

Comments begin with `#` and continue to the end of the line:

```hornet
# A comment
int x = 42  # Another comment
```

---

# Types

## Integer Types

Hornet has four integer types:

```text
int     64-bit signed
int32   32-bit signed
int8    8-bit signed
uint8   8-bit unsigned
```

`byte` is a built-in alias for `uint8` and `int64` an alias for `int`, not separate types.

Arithmetic wraps at the type's width; division truncates toward zero; shift counts use the low bits of the count (6 for `int`, 5 otherwise). Dividing by zero panics (`integer division by zero`), as does dividing the `int` or `int32` minimum by `-1` (`integer overflow in division`); for `int8` and `uint8` that quotient wraps. Constant expressions reject both at compile time.

Integer operations do not perform C-style implicit promotions. Operands normally need to have matching integer types:

```hornet
int8 a = 5
int8 b = 10
int8 c = a + b
```

Convert explicitly when necessary:

```hornet
int x = 10
int8 y = int8(x)
```

Integer literals are range-checked against their destination type.

## Boolean

Boolean values are:

```hornet
true
false
```

The logical operators are `and`, `or`, and `not`. Conditions must have type `bool`; integers are not implicitly treated as truth values.

## Strings

Strings use single quotes:

```hornet
str message = 'hello'
```

Strings are byte-oriented rather than a Unicode text abstraction. String literals support the language's escape syntax, including `\n`, `\t`, `\r`, `\0`, escaped quotes, escaped backslashes, and `\xNN` byte escapes. A literal ends on the line it starts; write `\n` for a newline.

String concatenation uses `+`:

```hornet
str message = 'hello ' + 'world'
```

Indexing a string produces a `byte`, and slicing a string produces another `str` view:

```hornet
str s = 'hello world'
byte first = s[0]
str word = s[0:5]
```

`len(s)` is the length in bytes. Strings convert to and from bytes; `str(...)` and `bytes(...)` copy, so the results are independent:

```hornet
str one = str(s[0])         # a one-byte str
[]byte buf = bytes(s)       # a mutable copy
buf[0] = "H"
str back = str(buf)         # 'Hello world'; s is unchanged
```

`+` copies both operands, so building a long string by repeated `+` is quadratic; use `Builder` from `stdlib/strings.ht` instead.

## Byte Literals

Double-quoted literals represent a single byte:

```hornet
byte a = "A"
byte newline = "\n"
byte zero = "\x00"
```

A byte literal must resolve to exactly one byte in the range `0..255`.

## Arrays

Array sizes are part of the type:

```hornet
[3]int values = [1, 2, 3]
[8]byte buffer
[2][3]int matrix = [[1, 2, 3], [4, 5, 6]]
[LIMIT + 1]int table      # sizes may be constant expressions
```

## Slices

Slices are variable-length views over storage:

```text
[]int
[]str
[][3]int
[][]int
```

A slice contains a pointer, length, and capacity. Slices alias their backing storage rather than copying it.

## Structs

Structs are nominal types:

```hornet
type Point struct:
    int x
    int y
```

## Type Aliases

`type Name = T` gives an existing type another name; values of the two are interchangeable, but struct literals use the struct's own name:

```hornet
type Grid = [3][3]int
```

## Pointers

Pointers are written using `*`:

```hornet
*Point p
```

Address-of and dereference use `&` and `*`:

```hornet
Point point = Point(10, 20)
*Point p = &point
p.x = 42
print(*p)
```

Field access through a pointer dereferences automatically. Pointers are single-level; see [Current Limitations](#current-limitations).

## Sum Types

Sum types describe a value that contains exactly one of a fixed set of variants:

```hornet
type Circle struct:
    int radius

type Square struct:
    int side

type Shape is Circle | Square
```

A variant may be a struct, scalar, `str`, array, slice, dictionary, or pointer type, or `none`, a variant with no payload; it can't be another sum type. A sum's zero value is its `none` variant. A sum without one has no zero value, so a variable of it (or a struct or array containing it) needs an initializer, and named construction can't omit such a field.

Struct fields may have sum types, so types can be recursive, through a pointer, slice, or dictionary. Where a pointer to a sum is expected, `&Variant(...)` creates a new value of the sum, on the heap, holding that variant:

```hornet
type Num struct:
    int v

type Bin struct:
    str op
    *Expr left
    *Expr right

type Expr is Num | Bin | none

Expr e = Bin('+', &Num(1), &Bin('*', &Num(2), &Num(3)))
```

A type that contains itself by value has no finite size and is rejected.

The representation uses a discriminant and payload storage for the largest variant.

## Dictionaries

Dictionaries are typed hash tables:

```hornet
dict[str]int counts = dict[str]int{
    'red': 1,
    'green': 2,
}
```

The current implementation supports lookup, assignment, overwrite, deletion, `len`, membership with `in`, growth/rehashing, tombstones, and iteration.

Dictionary keys currently must be one of:

```text
int
int32
int8
uint8
bool
str
```

Values may be any otherwise-supported Hornet type.

A dictionary value refers to its table: assigning or passing a dictionary shares it, so a change through any copy is seen through all of them. A dictionary is never `none`.

---

# Variables and Scope

Variables are declared with a type followed by a name:

```hornet
int count
bool finished
str message
```

An initializer is optional:

```hornet
int count = 42
bool finished = false
str message = 'hello'
```

Without one, a variable holds its type's zero value: `0`, `false`, `''`, `none` for pointers, an empty slice, a new empty dictionary, the `none` variant for a sum that has one, and arrays and structs of zero values. A sum without a `none` variant has no zero value.

Blocks introduce lexical scopes, and shadowing is allowed in nested scopes:

```hornet
def int main():
    int x = 10

    if true:
        int x = 20
        print(x)

    print(x)
    return 0
```

A name cannot be redeclared in the same scope.

## Constants

`const` declares a top-level constant of an integer type, `bool`, or `str`:

```hornet
const int TK_IDENT = 1
const int TK_NUMBER = TK_IDENT + 1
const int LIMIT = 1 << 16
const str GREETING = 'hello ' + 'world'
const bool DEBUG = LIMIT > 1000 and not false
```

The value is computed at compile time from literals, other constants (in any order), operators, string concatenation, and integer casts, with the same wraparound as at runtime. Integer constants can size arrays. Constants are imported and qualified like other top-level names (`from 'lexer' import TK_IDENT`, `lexer.TK_IDENT`), and a leading `_` makes one private. They can't be assigned or have their address taken, and a local variable, parameter, or loop binding can't reuse the name of a constant visible in its file.

---

# Functions

Functions begin with `def`:

```hornet
def int add(int a, int b):
    return a + b
```

The return type may be omitted. An omitted return type is the language's no-value return convention, and such a function may fall off the end:

```hornet
def greet(str name):
    print('hello ' + name)
```

A trailing comma is allowed after the last parameter or argument, and after the last element or entry of an array, slice, or dictionary literal.

Recursive functions are supported:

```hornet
def int fib(int n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    return fib(n - 1) + fib(n - 2)
```

The conventional executable entry point is:

```hornet
def int main():
    return 0
```

`main` returns the exit status as `int`, `int32`, `int8`, `uint8`, or `bool`.

A command-line program can instead use the native process arguments directly:

```hornet
def int main(int argc, *byte argv):
    # argv can be converted to []str with stdlib/os.ht
    return 0
```

---

# Structs and Methods

A struct can be constructed positionally:

```hornet
Point p = Point(10, 20)
```

or with named fields:

```hornet
Point p = Point(x=10, y=20)
```

Positional construction requires all fields in declaration order. Named construction may omit fields; omitted fields receive their zero value.

Struct fields are read and written with `.`:

```hornet
int x = p.x
p.y = 42
```

Nested aggregates can be indexed and then accessed:

```hornet
points[i].x = 10
```

Methods are declared inside the struct. The first parameter is the receiver:

```hornet
type Point struct:
    int x
    int y

    def int sum(p):
        return p.x + p.y
```

Call a method with dot syntax:

```hornet
Point p = Point(10, 20)
print(p.sum())
```

Struct values use value semantics: assigning or passing one copies its value rather than implicitly creating a reference, and a method's receiver is a copy.

A method whose receiver is written `*name` receives a pointer and can modify the struct:

```hornet
type Counter struct:
    int n

    def inc(*c, int by):
        c.n += by

Counter c = Counter(0)
c.inc(5)        # passes &c
print(c.n)      # 5
```

The receiver must be addressable (a variable, field, index, or dereference) or already a pointer; calling a pointer method on a temporary such as a call result is an error. Methods of either kind can be called through a pointer.

---

# Arrays and Slices

Slices can be formed from arrays or slices:

```hornet
[5]int values = [10, 20, 30, 40, 50]

[]int all = values[:]
[]int middle = values[1:4]
[]int tail = values[2:]
```

A slice literal uses an explicit element type, which may be omitted where a slice type is expected (a declaration, assignment, argument, return value, field, or element):

```hornet
[]int values = []int[1, 2, 3]
[]int more = [4, 5]
```

A slice is never `none`. An uninitialized slice is empty, as are `[]` and `[]int[]`; test with `len(values) == 0`.

A typed literal whose element type is a pointer to a named type after two or more fixed sizes, such as `[2][1]*P[...]`, has the same tokens as an indexed literal multiplied by an indexed value, and is read as the multiplication. Give the variable that type and use an untyped literal instead.

Indexing is bounds checked at runtime.

Appending may allocate a new backing store:

```hornet
[]int values = []int[1, 2, 3]
values = append(values, 4)
```

When the backing store is full, capacity grows to 1, then doubles up to 256, then grows by a quarter; the compiled code picks the new capacity and `hornet_slice_grow` in the runtime copies the elements.

---

# Dictionaries

A dictionary literal specifies both key and value types:

```hornet
dict[str]int counts = dict[str]int{
    'red': 2,
    'green': 1,
}
```

Read and write entries with indexing:

```hornet
int n = counts['red']
counts['red'] = n + 1
```

Check membership with `in`:

```hornet
if 'green' in counts:
    print(counts['green'])
```

Remove an entry with `del`:

```hornet
del(counts, 'green')
```

`len` returns the number of live entries.

The runtime uses open addressing with linear probing, tombstones, and rehashing on growth.

---

# Control Flow

## `if`, `elif`, and `else`

```hornet
if x < 0:
    print('negative')
elif x == 0:
    print('zero')
else:
    print('positive')
```

## `while`

```hornet
int i = 0
while i < 10:
    print(i)
    i += 1
```

## C-style `for`

Hornet supports a three-clause `for` loop:

```hornet
for int i = 0; i < 10; i += 1:
    print(i)
```

The initialization clause is currently a variable declaration, and the increment clause is any assignment. A loop variable whose address may outlive an iteration gets new storage each iteration.

## `for ... in ...`

Hornet supports collection iteration:

```hornet
for value in values:
    print(value)
```

An array or slice can provide an index and value together:

```hornet
for i, value in values:
    print(i)
    print(value)
```

A string yields its bytes, optionally with their indices:

```hornet
for i, b in 'hi':
    print(i)
    print(str(b))
```

A dictionary can provide either its keys or both keys and values:

```hornet
for key in counts:
    print(key)

for key, value in counts:
    print(key)
    print(value)
```

The iterable must be an array, slice, dictionary, or string given as a variable, field, index, slice expression, or array/dictionary/string literal. Iterating over a function-call result or a dereference is not yet supported.

Each iteration has its own bindings, so taking a binding's address is allowed. Reallocating the iterated slice (e.g. by `append`) or growing the iterated dictionary inside the loop panics.

`break` and `continue` apply to the innermost enclosing loop.

---

# Operators

### Arithmetic

```text
+  -  *  /  %
```

### Comparison

```text
<  >  <=  >=  ==  !=
```

Ordering applies to integers of the same type. Equality applies to integers, `bool`, `str`, pointers, and arrays and structs of comparable types. Pointers can be compared with `none`, and so can a sum with a `none` variant (`x == none` means `x is none`); slices and dictionaries are never `none`.

### Membership

```text
in
not in
```

Membership applies to dictionary keys and array or slice elements.

### Logical

```text
and  or  not
```

`and` and `or` use short-circuit evaluation.

### Bitwise

```text
&  |  ^
<< >>
~
```

### Assignment

```text
=
+=  -=  *=  /=  %=
&=  |=  ^=
<<= >>=
```

A compound assignment works for any assignable target (a variable, field, element, dictionary entry, or `*p`), including `+=` on a `str`, which appends. It evaluates its target once, reads it, evaluates the value, and then writes.

---

# Pattern Matching

`is` can test the active variant of a sum type in an `if`/`elif` condition:

```hornet
if shape is Circle:
    print(shape.radius)
```

A narrowed binding can be introduced explicitly:

```hornet
if shape is Circle as circle:
    print(circle.radius)
```

`match` provides multiple variant arms:

```hornet
match shape as s:
    is Circle:
        print(s.radius)
    is Square:
        print(s.side)
```

The subject of `... as NAME` can be any sum-typed expression, such as a field, an element, or `*p`; `NAME` is a copy of it, so assigning to its fields doesn't change the subject. A `none` variant is tested with `is none`.

`match` must be exhaustive, or end with `else:`. A `match` whose arms all return counts as returning; an arm ending in a call such as `panic` does not, so a `return` is still needed after it.

More general flow-sensitive narrowing through arbitrary boolean expressions and control-flow paths is still future work.

---

# Built-ins

The compiler currently provides the following built-in operations:

```text
print
len
append
del
bytes
```

`str(...)` conversions are described under [Strings](#strings).

## `print`

`print` can print supported scalar and aggregate values:

```hornet
print(42)
print('hello')
print([1, 2, 3])
print(counts)
```

Arrays, slices, structs, dictionaries, and supported sum-type values are recursively formatted by the native runtime.

## `len`

`len` returns the size of an array, slice, dictionary, or string (in bytes):

```hornet
print(len(values))
print(len(view))
print(len(counts))
print(len('hello'))
```

## `append`

`append` adds an element to a slice and returns the resulting slice:

```hornet
[]int values = []int[1, 2]
values = append(values, 3)
```

## `bytes`

`bytes(s)` returns a new `[]byte` copy of the string `s`.

## `del`

`del(dict, key)` removes an existing dictionary entry:

```hornet
del(counts, 'obsolete')
```

A missing key is a runtime error.

---

# Modules and Imports

A module is currently one `.ht` source file.

Import a module and access its exported declarations through a qualifier:

```hornet
import 'utils'

int value = utils.add(2, 3)
```

Rename the module qualifier explicitly:

```hornet
import 'utilities/string_helpers' as strings
```

Specific declarations can be imported directly:

```hornet
from 'utils' import add
```

Aliases are supported:

```hornet
from 'utils' import add as plus
```

Multiple names may be imported in one declaration:

```hornet
from 'utils' import add, other as renamed
```

Module discovery follows imports transitively. Local modules are resolved relative to the importing file, and the compiler can fall back to the bundled `stdlib/` directory.

Top-level names beginning with `_` are private to their module.

Every declaration is reached this way, including `extern` functions: a module that uses another module's `extern` imports it (`from 'os' import write_fd`) or qualifies it (`os.write_fd(...)`). A local variable, parameter, or loop binding can't have the name of an import alias or a constant in scope.

Each file's names are resolved in that file's own scope. Internally every declaration of an imported module gets a program-wide unique name, while `extern` symbols keep their foreign names for the linker; errors and printed values use names as declared.

The current module system intentionally has a narrow filesystem-based model. Project-level package roots, package declarations decoupled from filenames, and distributing one package across multiple files are future work.

---

# Foreign Functions and Intrinsics

Hornet can declare external native functions with `extern`:

```hornet
extern int abs(int value)
extern int getpid()
extern free(*byte p)
```

An omitted return type means the function returns no value.

The current FFI restricts `extern` parameters and return values to scalar and pointer types. Hornet arrays, slices, structs, sum types, and strings do not currently have a general direct FFI representation.

Hornet's `int` is 64-bit; declare C `int` parameters and results as `int32` (for example `extern int32 close(int32 fd)`). `long`, `size_t`, and `ssize_t` are `int`.

Foreign calls use the same IR call and backend calling-convention machinery as ordinary Hornet function calls.

## Intrinsics

`intrinsic` declarations are reserved for a small, fixed set of compiler-recognized operations:

```hornet
intrinsic *byte _raw_ptr(str s)
intrinsic int _raw_len(str s)
intrinsic str _from_raw_parts(*byte p, int n)
```

Unlike `extern`, intrinsics do not name a separately linked function. The compiler recognizes the declared intrinsic and emits the corresponding IR directly.

The standard library uses these facilities to build C interoperability helpers.

---

# Standard Library

The bundled `stdlib/` directory is intentionally small and is beginning to provide the facilities needed by real command-line programs.

## `stdlib/c.ht`

Low-level C interoperability helpers, including:

```text
raw string access
C-string conversion
raw byte/string construction
panic()
```

## `stdlib/hash.ht`

FNV-1a hash functions for ordinary Hornet code (dictionaries hash in the runtime):

```text
hash_int64
hash_int
hash_bool
hash_byte
hash_str
```

## `stdlib/fmt.ht`

Integer formatting:

```hornet
str a = int_to_str(-1234)       # '-1234'
str b = int_to_hex(255)         # 'ff'
str c = pad_left('7', 3, "0")   # '007'
```

## `stdlib/strings.ht`

A growable buffer, and searching and splitting helpers:

```hornet
Builder b
b.write('n=')
b.write_int(42)
b.write_byte("!")
str s = b.to_str()              # 'n=42!'; also length() and reset()

[]str parts = split('a,b,,c', ',')
str joined = join(parts, '|')
bool yes = starts_with(s, 'n=')   # also ends_with
int at = index_of(s, '42')        # -1 if absent; also index_from, index_of_byte
str t = trim('  padded \n')
str r = repeat('ab', 3)
```

## `stdlib/errors.ht`

The error convention for functions that can fail: a result sum type holding either the value or an `Error`.

```hornet
type Error struct:
    str message

type StrResult is str | Error
type IntResult is int | Error
```

Handle a result with `match` or `is`, or use `must_str`/`must_int` to take the value and panic on error. `panic` is for bugs.

## `stdlib/os.ht`

Provides a small POSIX-style operating-system interface:

```hornet
[]str args = get_args(argc, argv)
StrResult contents = read_file('input.txt')
StrResult stdin_contents = read_stdin()
StrResult fd_contents = read_all_from_fd(fd)
IntResult written = write_file('output.txt', 'hello')
write_stdout('no trailing newline')
write_stderr('error\n')
exit(0)
```

Functions that can fail return a result from `stdlib/errors.ht` (`StrResult is str | Error`, `IntResult is int | Error`); handle it with `match` or `is`, or use `must_str`/`must_int` to panic on error. Directory/path APIs remain future work.

---

# Runtime

The native runtime is located in `runtime/runtime.c` and is compiled separately from the generated Hornet assembly.

It currently provides language-level services including:

* `print` and recursive value formatting
* `hornet_panic`, which flushes standard output, writes the message to standard error, and aborts
* `hornet_slice_grow`, which copies a slice into a larger backing store
* `hornet_bytes` for `bytes(s)`
* dictionary hash tables, hashed with FNV-1a (`hornet_hash_bytes`)
* runtime type descriptors (`hornet_typedesc_tags.h` is generated from `ir/typedesc.py` by `runtime/generate_typedesc_header.py`)
* argument access, output, file creation, exit, and OS error messages for `stdlib/os.ht` (`hornet_argv_get`, `hornet_write_fd`, `hornet_open_write`, `hornet_exit`, `hornet_error_message`)

Runtime checks panic with a message: array, slice, and string bounds; division by zero and `MIN / -1`; dereferencing `none`; a missing dictionary key; and reallocating a slice or dictionary being iterated with `for ... in`.

The runtime is deliberately separate from the native backends. The compiler is responsible for semantic operations such as type checking, aggregate layout, address calculation, and bounds-check generation; the runtime implements selected algorithms and services that are better expressed as ordinary native code.

The runtime currently uses the platform C library for lower-level services such as memory allocation and byte copying rather than wrapping every libc primitive in a Hornet-specific API.

---

# Compiler Architecture

The compiler is organized into distinct frontend, IR, optimization, backend, and runtime layers:

```text
Hornet source
     │
     ▼
   Lexer
     │
     ▼
   Parser ──────── one AST per module
     │
     ▼
Module discovery ── one AST per file
     │
     ▼
Semantic analysis ── resolves names per file, checks, builds the typed tree
     │
     ▼
IR construction ── from the typed tree; escape analysis, null and division checks
     │
     ▼
 IRProgram (verified)
     │
     ▼
IR optimization (re-verified)
     │
     ▼
native backend (per architecture)
     │
     ├── frame layout
     ├── register allocation
     ├── calling convention
     ├── instruction selection
     └── assembly emission
     │
     ▼
native assembly
     │
     ├───────────────┐
     ▼               ▼
Hornet runtime   external libraries
     │               │
     └───────┬───────┘
             ▼
       native executable
```

Semantic analysis never changes the parser's ASTs: it resolves each file's names in that file's scope (`scopes.py`), records what it learns (types, the declaration each name refers to, narrowing) by node number, and from that builds the typed tree (`typed_ast.py`). The typed program `semantic.analyze()` returns is the only input to later stages: a test checks that nothing in `ir/`, `optimize/`, `backend/`, or escape analysis imports the front end. In the typed tree every node has one meaning and a concrete type: names refer to symbols, the parser's overloaded forms are split (calls, struct literals, and builtins; array, slice, string, and dictionary indexing), each implicit operation is a node (widening into a sum, `&Variant(...)`, a literal becoming a slice, zero values), methods are ordinary functions, and `match` and compound assignment are nodes of their own. `compile.py --dump-typed` prints it.

The frontend constructs a complete `IRProgram` before a backend begins lowering it. The IR is independent of any target: a function's incoming arguments are an ordered list of word-sized temporaries in Hornet's own calling convention (a composite return value's destination address first, then one word per parameter, except two for `str` and three for slices, with arrays, structs, sum types, and dicts passed by address), and where each word physically arrives is decided by the backend. Lowering never modifies the IR.

Escape analysis decides which locals must live on the heap; heap storage comes from `malloc` and is never freed. It is a flow-insensitive points-to analysis per function, run on the typed tree, with per-parameter escape summaries so that passing `&x` to a function that doesn't keep the pointer leaves `x` on the stack.

IR construction compiles conditions made of `and`, `or`, and `not` straight to branches, and copies a composite value directly between places (two places of one type are the same storage or disjoint). Loads, stores, and copies address memory as a base plus a constant offset.

The IR optimizer repeats constant folding, identity simplification, constant-branch and unreachable-block removal, copy and constant propagation within blocks, copy coalescing, address folding (an address computed as a base plus a constant becomes the access's offset), and dead-code elimination until nothing changes. `ir/cfg.py` provides the shared control-flow and liveness analysis.

`backend/` holds one package per architecture, chosen by the target, plus `backend/common/` for what they share: linear-scan register allocation over the target's register lists (values live across calls get callee-saved registers), stack-frame slot layout, the magic numbers for division by constants, and jump cleanups. Each backend adds its calling convention (SysV for `backend/x86_64/`, where four of the argument registers are also allocatable, so incoming parameters and outgoing arguments move as parallel moves; AAPCS64 for `backend/aarch64/`), prologues that save only the registers a function uses, instruction selection that works directly on registers, stack slots, and immediates, a peephole pass, and assembly emission for Linux and macOS (AT&T syntax on x86-64). The AArch64 backend also rewrites accesses to stack slots beyond the reach of a load or store's offset after the frame is laid out.

Some IR operations deliberately lower to runtime calls. A runtime operation does not require a special calling mechanism; runtime functions participate in the same native call machinery as other external functions.

---

# Project Structure

```text
lexer.py           Lexical analysis
parser.py          AST construction
semantic.py        Semantic analysis: checking, and building the typed tree
typed_ast.py       The typed tree and its text dump
modules.py         Module discovery
scopes.py          Names across modules: each file's scope and the import checks
escape_analysis.py Escape analysis, on the typed tree
folding.py         Compile-time integer arithmetic (constants and IR folding)
typesys.py         Types and type layout
ops.py             Operator enums
symbols.py         Symbol table (one Symbol per declared variable)
diagnostics.py     Error types and error reporting

ir/                Intermediate representation, IR construction from the typed tree
                   (typed_builder.py), type descriptors, checks, and verification
optimize/          IR optimization passes

target.py          Compilation targets (arch-os)
backend/           Native backends: common/ shared pieces, x86_64/ and aarch64/
runtime/           Native Hornet runtime
stdlib/            Hornet standard-library modules

examples/          Example Hornet programs
tools/hfmt/        Source formatter, written in Hornet
tests/             Compiler, runtime, and end-to-end tests
benchmarks/        Benchmark programs and tooling

build.py           Build a runnable executable
compile.py         Generate native assembly
test.sh            Build, run, and delete one program: ./test.sh path/without_ext
Dockerfile, entrypoint.sh
                   Assemble and run an x86-64 `.s` file in a Linux container
TODO.md            Open language, compiler, runtime, and tooling work
```

Editor support is also included for Vim (`hornet-vim/`) and TextMate-compatible editors. `Hornet.tmbundle/` contains the TextMate grammar used by editors such as PyCharm.

---

# Example: A Real CLI Program

The repository includes `examples/wc.ht`, a small word/line/byte counting program. It demonstrates modules, command-line arguments, filesystem input, string/byte indexing, `for` loops, and the standard library.

A simplified version looks like:

```hornet
from 'os' import get_args, read_file, read_stdin, write_stderr, exit
from 'errors' import Error, StrResult

def str contents_or_exit(StrResult r):
    match r as v:
        is str:
            return v
        is Error:
            write_stderr(v.message + '\n')
            exit(1)
    return ''

def int count_lines(str s):
    int count = 0
    for int i = 0; i < len(s); i += 1:
        if s[i] == "\n":
            count += 1
    return count

def int main(int argc, *byte argv):
    []str args = get_args(argc, argv)

    if len(args) <= 1:
        print(count_lines(contents_or_exit(read_stdin())))
        return 0

    print(count_lines(contents_or_exit(read_file(args[1]))))
    return 0
```

Build it with:

```bash
python3 build.py examples/wc.ht -o wc
./wc README.md
```

The example is intentionally modest. Its purpose is to exercise the language and standard library in a real file-processing program rather than to reproduce all of Unix `wc`.

---

# Current Limitations

Hornet is still experimental. Some notable limitations are:

* Pointer-to-pointer types are parsed but rejected semantically.
* Some advanced pointer/address-taking cases remain unsupported.
* `for ... in ...` can't iterate a function-call result or a dereference; assign it to a variable first.
* `is` is limited to dedicated `if`/`elif` condition shapes rather than being a general boolean expression.
* Sum-type narrowing does not yet fully propagate through arbitrary control flow or `else` branches.
* A sum type can't be a variant of another sum type.
* A typed literal of pointers to a named type with two or more fixed sizes (`[2][1]*P[...]`) can't be written inline; see [Arrays and Slices](#arrays-and-slices).
* There are no enums; integer constants stand in for them.
* There is no no-return type, so a `return` is still needed after a call to `panic` or `exit` that ends a function.
* Sum-type equality is not implemented.
* Slice equality and dictionary equality are not implemented.
* `in` does not apply to strings.
* The FFI currently supports only scalar and pointer arguments/results.
* Passing structs by value through FFI is not yet supported.
* The standard library currently provides only a small subset of filesystem, process, path, formatting, and collection facilities.
* Generic types and generic functions are not implemented.
* First-class function types and closures are not implemented.
* Variadic functions and variadic FFI calls are not implemented.
* There is no garbage collector yet; string concatenation in particular never frees its intermediate strings.
* There are no floating-point types yet.
* Multithreading is not implemented.
* Without generics, each result type is a separate named sum type. There is no operator for propagating errors, and ignoring a result is not diagnosed.
* Panics print a message but no source location, and stack overflow is an unreported `SIGSEGV`.

---

# Self-Hosting

A longer-term goal is to rewrite the compiler itself in Hornet.

The current language already has the structural features needed by a compiler implementation: structs, arrays, slices, dictionaries, pointers, recursive sum types (so an AST can be expressed), pattern matching, modules, FFI, and native compilation. The Python compiler's typed tree, printed by `compile.py --dump-typed`, is the intended point of comparison between the two implementations.

The standard library now covers file and stream I/O, process exit, string building and searching, integer formatting, and an error convention. The formatter in `tools/hfmt` is the first substantial tool written in Hornet; it includes a Hornet lexer that is tested token-for-token against the compiler's own. The next step is porting the compiler itself, starting from that lexer.

The intended progression is roughly:

```text
filesystem and argument handling
        ↓
string and byte manipulation
        ↓
more standard-library data structures
        ↓
write useful CLI programs in Hornet
        ↓
port the lexer
        ↓
port the parser/AST
        ↓
port semantic analysis
        ↓
port IR construction and optimization
        ↓
port the native backend
        ↓
self-hosted compiler
```

Garbage collection is not a prerequisite for bootstrapping the compiler. A compiler invocation can allocate compiler data for the duration of the process and rely on process termination to reclaim that memory. A GC would become important for general long-running programs and for the eventual memory-management model of the language.

---

# Roadmap

Current and future work includes:

* Expanding the standard library, especially directory, path, and process facilities
* Error-propagation syntax for result types
* More complete pointer and address-taking support
* More precise escape analysis for iterators, aliases, and data more than one pointer away from a call argument
* More general sum-type narrowing
* Additional sum-type composition and equality support
* Generic types and functions
* First-class function types and closures
* Richer string formatting
* Regex support
* Richer FFI, including aggregate types where a stable ABI can be defined
* Potential garbage collection and a more explicit ownership model
* More IR optimizations, including common-subexpression and bounds-check elimination
* Better aggregate copying and layout decisions
* More sophisticated register allocation, including spill choices weighted by use count and loop depth
* Additional native targets

Hornet is intentionally developed incrementally: new language features are accompanied by parser, semantic-analysis, IR, backend/runtime, and end-to-end tests whenever appropriate.

