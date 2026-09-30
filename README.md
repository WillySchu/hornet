# Hornet

Hornet is an experimental, statically typed programming language with an indentation-based syntax and a native compiler targeting x86-64 Linux and macOS.

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

The compiler itself is written in Python and currently has no third-party Python dependencies.

To build a runnable native executable, you also need a C compiler/linker. The project currently uses `gcc` in its build and test tooling.

The native backend targets x86-64 Linux and macOS. On an Apple Silicon Mac, the compiler currently produces x86-64 output and the build tooling requests an x86-64 build from the system compiler.

### Build an Executable

The recommended way to build a runnable Hornet program is:

```bash
python3 build.py program.ht -o program
./program
```

`build.py` compiles the Hornet source, compiles the bundled native runtime, and links the two together into an executable.

The target platform can be selected explicitly:

```bash
python3 build.py program.ht --platform linux -o program
python3 build.py program.ht --platform macos -o program
```

When `--platform` is omitted, the host platform (`linux` or `macos`) is used for the platform-specific symbol/linking conventions.

### Generate Assembly

`compile.py` stops after assembly generation and is useful when inspecting the compiler's output:

```bash
python3 compile.py program.ht --platform linux -o program.s
```

Without `-o`, assembly is written to standard output:

```bash
python3 compile.py program.ht --platform linux
```

A generated program that uses runtime functions such as `print` must also be linked with `runtime/runtime.c`. `build.py` handles this automatically.

### Testing

Run the test suite from the repository root with:

```bash
pytest
```

The tests cover the lexer, parser, semantic analysis, module discovery and merging, IR construction and verification, optimization, the native backend, escape analysis, runtime behavior, and end-to-end compiled programs.

---

# Language Overview

Hornet currently provides:

* Functions and recursion
* Static typing with explicit integer conversions
* Block scoping and shadowing
* `int`, `int8`, `uint8`, `byte`, `int64`, `bool`, and `str`
* Fixed-size arrays
* Slices
* Nominal structs and methods
* Type aliases
* Single-level pointers
* Tagged sum types
* `if`, `elif`, `else`, and `match`
* `while` loops
* C-style `for` loops
* `for ... in ...` iteration over arrays, slices, and dictionaries
* `break` and `continue`
* Dictionaries with hashing, deletion, membership, and iteration
* Explicit integer casts
* Arithmetic, comparison, logical, bitwise, and membership operators
* Compound assignment
* Runtime bounds checking
* Built-ins such as `print`, `len`, `append`, and `del`
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
int
int8
uint8
int64
```

`byte` is a built-in alias for `uint8`, not a separate type.

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

Strings are byte-oriented rather than a Unicode text abstraction. String literals support the language's escape syntax, including `\n`, `\t`, `\r`, `\0`, escaped quotes, escaped backslashes, and `\xNN` byte escapes.

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

The current implementation intentionally restricts pointer types and some address-taking operations; see [Current Limitations](#current-limitations).

## Sum Types

Sum types describe a value that contains exactly one of a fixed set of variants:

```hornet
type Circle struct:
    int radius

type Square struct:
    int side

type Shape is Circle | Square
```

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
int8
uint8
int64
bool
str
```

Values may be any otherwise-supported Hornet type.

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

Struct values use value semantics: assigning or passing one copies its value rather than implicitly creating a reference.

---

# Arrays and Slices

Slices can be formed from arrays or slices:

```hornet
[5]int values = [10, 20, 30, 40, 50]

[]int all = values[:]
[]int middle = values[1:4]
[]int tail = values[2:]
```

A slice literal uses an explicit element type:

```hornet
[]int values = []int[1, 2, 3]
```

The nil/zero slice is written `none`:

```hornet
[]int values = none
```

Indexing is bounds checked at runtime.

Appending may allocate a new backing store:

```hornet
[]int values = []int[1, 2, 3]
values = append(values, 4)
```

The runtime owns the backing-storage growth algorithm; the compiler remains responsible for slice address calculation and bounds checking.

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

The initialization clause is currently a variable declaration, and the increment clause is currently an assignment.

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

A dictionary can provide either its keys or both keys and values:

```hornet
for key in counts:
    print(key)

for key, value in counts:
    print(key)
    print(value)
```

The iterable expression must currently be a plain variable, field, or index expression whose type is an array, slice, or dictionary. Direct iteration over a literal or a function-call result is not yet supported.

The implementation currently uses one loop binding storage location for the whole iteration. Taking the address of an iteration binding is therefore rejected.

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

### Membership

```text
in
not in
```

Membership currently applies to dictionaries only.

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

Compound assignments are desugared into ordinary assignments plus the corresponding binary operation.

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

The current implementation checks match exhaustiveness during semantic analysis.

More general flow-sensitive narrowing through arbitrary boolean expressions and control-flow paths is still future work.

---

# Built-ins

The compiler currently provides the following built-in operations:

```text
print
len
append
del
```

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

`len` returns the size of an array, slice, or dictionary:

```hornet
print(len(values))
print(len(view))
print(len(counts))
```

`len` does not currently apply to strings.

## `append`

`append` adds an element to a slice and returns the resulting slice:

```hornet
[]int values = []int[1, 2]
values = append(values, 3)
```

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

Modules are merged into a single semantic program before IR construction. Imported declarations are internally renamed to avoid collisions, while `extern` symbols retain their foreign names for the linker.

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

Hash functions used by dictionaries and available to ordinary Hornet code:

```text
hash_int64
hash_int
hash_bool
hash_byte
hash_str
```

## `stdlib/fmt.ht`

Currently provides integer-to-string conversion for non-negative `int` values:

```hornet
str s = int_to_str(1234)
```

## `stdlib/os.ht`

Provides a small POSIX-style operating-system interface:

```hornet
[]str args = get_args(argc, argv)
StrResult contents = read_file('input.txt')
StrResult stdin_contents = read_stdin()
IntResult written = write_file('output.txt', 'hello')
write_stdout('no trailing newline')
exit(0)
```

Functions that can fail return a result from `stdlib/errors.ht` (`StrResult is str | Error`, `IntResult is int | Error`); handle it with `match` or `is`, or use `must_str`/`must_int` to panic on error. Directory/path APIs remain future work.

---

# Runtime

The native runtime is located in `runtime/runtime.c` and is compiled separately from the generated Hornet assembly.

It currently provides language-level services including:

* `print` and recursive value formatting
* `hornet_panic` for runtime failures
* `hornet_slice_grow` for slice backing-storage growth
* dictionary hash-table support
* runtime type-descriptor support

The runtime is deliberately separate from the x86-64 backend. The compiler is responsible for semantic operations such as type checking, aggregate layout, address calculation, and bounds-check generation; the runtime implements selected algorithms and services that are better expressed as ordinary native code.

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
   Parser
     │
     ▼
    AST
     │
     ▼
 Desugaring
     │
     ▼
Semantic analysis
     │
     ▼
 IRProgram
     │
     ▼
IR optimization
     │
     ▼
x86-64 backend
     │
     ├── frame layout
     ├── register allocation
     ├── calling convention
     ├── instruction selection
     └── assembly emission
     │
     ▼
x86-64 assembly
     │
     ├───────────────┐
     ▼               ▼
Hornet runtime   external libraries
     │               │
     └───────┬───────┘
             ▼
       native executable
```

The frontend constructs a complete `IRProgram` before the x86-64 backend begins lowering it. The intermediate representation is independent of the x86-64 assembly representation and is the natural boundary for future optimization passes and additional native backends.

The backend owns architecture-specific concerns such as register allocation, stack-frame layout, the native calling convention, instruction selection, and assembly emission.

Some IR operations deliberately lower to runtime calls. A runtime operation does not require a special calling mechanism; runtime functions participate in the same native call machinery as other external functions.

---

# Project Structure

```text
lexer.py           Lexical analysis
parser.py          AST construction
semantic.py        Semantic analysis and type checking
desugar.py         AST desugaring
modules.py         Module discovery
merge.py           Module merging and name resolution
escape_analysis.py Escape analysis

ir/                Intermediate representation and IR construction
optimize/          IR optimization passes

codegen/           x86-64 backend and assembly representation
runtime/           Native Hornet runtime
stdlib/            Hornet standard-library modules

examples/          Example Hornet programs
tests/             Compiler, runtime, and end-to-end tests
benchmarks/        Benchmark programs and tooling

build.py           Build a runnable executable
compile.py         Generate native assembly
SELF_HOST_CHECKLIST.md
                   Self-hosting requirements and progress
TODO.md            Open language, compiler, runtime, and tooling work
```

Editor support is also included for Vim and TextMate-compatible editors. `Hornet.tmbundle/` contains the TextMate grammar used by editors such as PyCharm.

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

* The native backend currently supports x86-64 Linux and macOS only.
* Pointer-to-pointer types are parsed but rejected semantically.
* Some advanced pointer/address-taking cases remain unsupported.
* Taking the address of a `for ... in ...` binding is currently rejected because iterator-binding escape tracking is not yet precise enough.
* `for ... in ...` currently requires a variable, field, or index as its iterable expression; literals and call results must be assigned to a variable first.
* `is` is limited to dedicated `if`/`elif` condition shapes rather than being a general boolean expression.
* Sum-type narrowing does not yet fully propagate through arbitrary control flow or `else` branches.
* Nested sum-type variants are still restricted.
* Sum-type equality is not implemented.
* Slice equality and dictionary equality are not implemented.
* `in` currently supports dictionaries only.
* String membership is not implemented.
* `len` does not currently apply to strings.
* The FFI currently supports only scalar and pointer arguments/results.
* Passing structs by value through FFI is not yet supported.
* The standard library currently provides only a small subset of filesystem, process, path, formatting, and collection facilities.
* File I/O is currently read-only.
* Generic types and generic functions are not implemented.
* First-class function types and closures are not implemented.
* Variadic functions and variadic FFI calls are not implemented.
* There is no garbage collector yet.
* There are no floating-point types yet.
* Multithreading is not implemented.
* The compiler does not yet have a mature error/result abstraction for ordinary library code; several standard-library errors currently become runtime panics.

---

# Self-Hosting

A longer-term goal is to rewrite the compiler itself in Hornet.

The current language already has most of the structural features needed by a compiler implementation: structs, arrays, slices, dictionaries, pointers, sum types, pattern matching, modules, FFI, and native compilation.

The remaining work is increasingly concentrated in the standard library and operating-system interface rather than the compiler backend itself.

The repository contains `SELF_HOST_CHECKLIST.md` to track that work.

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

* Expanding the standard library, especially filesystem, path, process, and byte/string facilities
* Better error/result handling
* More complete pointer and address-taking support
* More precise escape analysis for iterators and aliases
* More general sum-type narrowing
* Additional sum-type composition and equality support
* Generic types and functions
* First-class function types and closures
* String formatting and richer text processing
* Regex support
* Richer FFI, including aggregate types where a stable ABI can be defined
* Potential garbage collection and a more explicit ownership model
* More IR optimizations, including common-subexpression and bounds-check elimination
* Better aggregate copying and layout decisions
* More sophisticated register allocation
* Additional native targets, including a future AArch64/Apple Silicon backend

Hornet is intentionally developed incrementally: new language features are accompanied by parser, semantic-analysis, IR, backend/runtime, and end-to-end tests whenever appropriate.

