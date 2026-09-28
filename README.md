# Hornet

Hornet is an experimental, statically typed programming language with an indentation-based syntax and a native compiler targeting x86-64 Linux and macOS.

Hornet is intentionally small, but it is no longer just a toy compiler. The current implementation includes a standalone intermediate representation, an optimizing x86-64 backend, arrays and slices, structs, pointers, dictionaries, tagged sum types and pattern matching, modules, a small native runtime, and a C-compatible FFI.

Hornet source files use the `.ht` extension.

## A Small Example

```hornet
def int main():
    int i = 0

    while i < 10:
        print(i)
        i += 1

    return 0
```

## Building

### Requirements

The compiler itself is written in Python and currently has no third-party Python dependencies.

To build a runnable native executable, you also need a C compiler/linker (`gcc` is used by the project's build and test tooling). The native backend currently targets x86-64 Linux and macOS.

### Build an Executable

The recommended way to build a runnable Hornet program is:

```bash
python3 build.py program.ht -o program
./program
```

`build.py` runs the Hornet compiler, compiles the bundled C runtime, and links the two compilation units into the final executable.

The target platform can be selected explicitly:

```bash
python3 build.py program.ht --platform linux -o program
python3 build.py program.ht --platform macos -o program
```

When `--platform` is omitted, the build defaults to the host platform (`linux` or `macos`). The compiler still emits x86-64 code, so the generated executable is intended for x86-64 targets.

### Generate Assembly

`compile.py` stops after native assembly generation. This is useful when inspecting the compiler's output:

```bash
python3 compile.py program.ht --platform linux -o program.s
```

Without `-o`, assembly is written to standard output:

```bash
python3 compile.py program.ht --platform linux
```

Generated assembly that calls Hornet runtime functions must be linked with `runtime/runtime.c`; `build.py` handles that automatically.

### Test the Compiler

Run the Python test suite from the repository root with:

```bash
pytest
```

The tests cover the lexer, parser, semantic analysis, module discovery and merging, IR construction and verification, optimization, the x86-64 backend, runtime behavior, and end-to-end compiled programs.

---

# Language Overview

Hornet currently provides:

* Functions with statically typed parameters and optional return types
* Block scoping and shadowing
* `int`, `int8`, `uint8`, `byte`, `int64`, `bool`, and `str`
* Fixed-size arrays
* Slices
* Nominal structs and methods
* Type aliases
* Single-level pointers
* Tagged sum types
* `if`, `elif`, `else`, and `match`
* `while` and C-style `for` loops
* `break` and `continue`
* Dictionaries
* Explicit integer casts
* Arithmetic, comparison, logical, and bitwise operators
* Compound assignment
* Runtime bounds checking
* Built-ins such as `print`, `len`, `append`, and `del`
* Modules and imports
* External C functions through `extern`
* A small set of compiler-defined `intrinsic` functions

The language deliberately does not attempt to hide all low-level representation details. Arrays, slices, structs, pointers, and foreign functions have explicit representations and constraints.

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

`byte` is a built-in alias for `uint8` rather than a separate type.

Integer operations require matching integer types rather than applying C-style implicit promotions. For example:

```hornet
int8 a = 5
int8 b = 10
int8 c = a + b
```

but mixing integer widths requires an explicit cast:

```hornet
int x = 10
int8 y = int8(x)
```

Integer literals are checked against their destination type.

## Booleans

Boolean values are:

```hornet
true
false
```

Boolean operators are `and`, `or`, and `not`.

Conditions must be boolean expressions. Integers are not implicitly treated as truth values.

## Strings

Strings use single quotes:

```hornet
str message = 'hello'
```

A `str` is represented as a pointer/length value and is treated as a byte-oriented string rather than a Unicode text abstraction. String literals support the escape sequences `\\n`, `\\t`, `\\r`, `\\0`, escaped quote characters, escaped backslashes, and `\\xNN` byte escapes.

String concatenation uses `+`:

```hornet
str message = 'hello ' + 'world'
```

Strings can be sliced, producing another string view:

```hornet
str s = 'hello world'
str part = s[0:5]
```

## Byte Literals

Double-quoted one-character literals represent a single byte:

```hornet
byte a = "A"
byte newline = "\n"
byte zero = "\x00"
```

A byte literal must resolve to exactly one byte (`0` through `255`).

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

Blocks introduce lexical scopes. Shadowing an outer variable in an inner block is allowed:

```hornet
def int main():
    int x = 10

    if true:
        int x = 20
        print(x)

    print(x)
    return 0
```

Variables cannot be redeclared with the same name in the same scope.

---

# Functions

Functions begin with `def`:

```hornet
def int add(int a, int b):
    return a + b
```

The return type may be omitted. An omitted return type means the function has no value result and may fall off the end:

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

A function with an explicit return type must return a value on every reachable path.

The conventional executable entry point is:

```hornet
def int main():
    return 0
```

---

# Arrays

Fixed-size arrays include their size in the type:

```hornet
[3]int values = [1, 2, 3]
[8]byte buffer
[2][3]int matrix = [[1, 2, 3], [4, 5, 6]]
```

Array indexing is zero-based:

```hornet
int x = values[0]
values[1] = 42
```

Array indexing is bounds checked at runtime.

An array literal can be untyped when its expected type is already known:

```hornet
[3]int values = [1, 2, 3]
```

A fully typed array literal can also be used as a general expression:

```hornet
foo([3]int[1, 2, 3])
```

---

# Slices

Slices represent views over storage and have a runtime length and capacity.

A slice type is written:

```text
[]int
[]str
[][3]int
[][]int
```

Slices can be formed from arrays or other slices:

```hornet
[5]int values = [10, 20, 30, 40, 50]

[]int all = values[:]
[]int middle = values[1:4]
[]int tail = values[2:]
```

A slice is a view rather than a copy:

```hornet
[3]int values = [1, 2, 3]
[]int view = values[:]

view[0] = 100

print(values[0])
```

A slice literal can be written with an explicit element type:

```hornet
[]int values = []int[1, 2, 3]
```

The nil slice value is `none`:

```hornet
[]int values = none
```

Appending to a slice may allocate a new backing store:

```hornet
[]int values = []int[1, 2, 3]
values = append(values, 4)
```

The backing-storage growth operation is implemented by the native Hornet runtime.

---

# Structs

Structs are nominal types declared with `type Name struct:`:

```hornet
type Point struct:
    int x
    int y
```

Struct values are constructed using function-like syntax:

```hornet
Point p = Point(10, 20)
```

Fields can also be named:

```hornet
Point p = Point(x=10, y=20)
```

Named and positional arguments cannot be mixed.

Struct values have value semantics. Assigning or passing a struct copies its value rather than implicitly creating a reference.

Fields are read and written with `.`:

```hornet
int x = p.x
p.y = 42
```

Array elements and nested aggregate expressions can be addressed as well:

```hornet
points[i].x = 10
```

## Struct Methods

Methods are declared inside the struct body. The first parameter is the receiver:

```hornet
type Point struct:
    int x
    int y

    def int sum(p):
        return p.x + p.y
```

Methods are called using dot syntax:

```hornet
Point p = Point(10, 20)
print(p.sum())
```

---

# Type Aliases

`type Name = TargetType` defines an alias rather than a new nominal type:

```hornet
type Count = int
type Bytes = []byte
```

Aliases are interchangeable with their underlying types.

---

# Pointers

Pointers are written with `*`:

```hornet
*int p
*Point point
```

The address-of operator is `&` and dereference is `*`:

```hornet
int x = 42
*int p = &x

print(*p)
*p = 99
```

Struct pointers support automatic dereference for field access and method calls:

```hornet
type Point struct:
    int x
    int y

*Point p = &point
print(p.x)
print(p.sum())
```

Pointer-to-pointer types are parsed by the grammar but are currently rejected by semantic analysis. Pointer semantics are intentionally still conservative; some more advanced address-taking and composite-dereference cases remain future work.

---

# Sum Types

A sum type represents exactly one of a fixed set of variants:

```hornet
type Shape is Circle | Square
```

Variants can be scalar types, strings, arrays, slices, pointers, or previously declared structs. Sum types cannot currently contain another sum type as a variant.

For example:

```hornet
type Circle struct:
    int radius

type Square struct:
    int side

type Shape is Circle | Square
```

A value is converted to the sum type by flowing a compatible value into a sum-typed location:

```hornet
Circle c = Circle(10)
Shape s = c
```

The runtime representation uses a discriminant and enough payload storage for the largest variant.

## `is` Narrowing

An `if` or `elif` condition can test the active variant:

```hornet
if s is Circle:
    print(s.radius)
```

A bare variable can be rebound with `as`:

```hornet
if s is Circle as c:
    print(c.radius)
```

For an expression that does not already have a variable name, an explicit binding is required:

```hornet
if shapes[0] is Circle as c:
    print(c.radius)
```

The current `is` syntax is intentionally narrower than an ordinary boolean expression. It is handled specifically in `if`/`elif` conditions rather than being a fully composable expression, and flow-sensitive narrowing across arbitrary control-flow constructs is still future work.

## `match`

`match` provides multiple variant arms and is checked for exhaustiveness:

```hornet
match s as shape:
    is Circle:
        print(shape.radius)
    is Square:
        print(shape.side)
```

An explicit `else` arm can be supplied where appropriate:

```hornet
match s as shape:
    is Circle:
        print(shape.radius)
    else:
        print('not a circle')
```

The current implementation supports struct/scalar/string variants but does not yet allow nested sum types as variants.

---

# Dictionaries

Dictionaries use `dict[key_type]value_type`:

```hornet
dict[str]int counts
```

A dictionary literal always specifies its key and value types:

```hornet
dict[str]int counts = dict[str]int{
    'red': 1,
    'green': 2,
}
```

Entries can contain arbitrary expressions:

```hornet
dict[str]int values = dict[str]int{
    'answer': 40 + 2,
}
```

Dictionary keys currently must have one of these types:

```text
int
int8
uint8
int64
bool
str
```

Values may be arbitrary supported Hornet types.

Lookup and assignment use indexing:

```hornet
int n = counts['red']
counts['red'] = n + 1
```

Missing-key lookup is a runtime error.

Membership is tested with `in`:

```hornet
if 'red' in counts:
    print(counts['red'])
```

The corresponding negated form is also supported:

```hornet
if 'blue' not in counts:
    print('missing')
```

`len` returns the number of entries, and `del` removes an existing entry:

```hornet
print(len(counts))
del(counts, 'red')
```

Deleting a missing key is a runtime error.

The implementation uses an open-addressed hash table with linear probing, growth/rehashing, and tombstones.

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

## `for`

Hornet supports a compact three-clause `for` loop:

```hornet
for int i = 0; i < 10; i += 1:
    print(i)
```

The current form requires a fresh variable declaration in the initialization clause and an assignment to an existing bare variable in the increment clause.

`for` does not yet implement a collection-oriented `for ... in ...` loop; `in` is currently a dictionary membership operator.

## `break` and `continue`

Both apply to the innermost enclosing loop:

```hornet
for int i = 0; i < 100; i += 1:
    if i % 2 == 0:
        continue

    if i > 50:
        break

    print(i)
```

---

# Operators

## Arithmetic

```text
+  -  *  /  %
```

For integer operands, both sides must have the same integer type.

For strings, `+` performs concatenation.

## Comparison

```text
<  >  <=  >=
== !=
```

Equality is supported for matching scalar/integer/boolean/string types, pointers where applicable, arrays and comparable structs, and slice/pointer comparisons with `none`. Slice equality and sum-type equality are not currently defined.

## Logical

```text
and
or
not
```

`and` and `or` use short-circuit evaluation.

## Bitwise

```text
&  |  ^
<< >>
~
```

## Assignment

```text
=
+=  -=  *=  /=  %=
&=  |=  ^=
<<= >>=
```

Compound assignments are desugared by the parser into ordinary assignments and binary expressions.

---

# Built-ins

The compiler currently reserves these built-in function names:

```text
print
len
append
del
```

## `print`

`print` can print values of supported types, including nested arrays, slices, structs, dictionaries, and sum types. Printing is implemented in the native runtime using runtime type descriptors.

```hornet
print(42)
print('hello')
print([1, 2, 3])
print(counts)
```

## `len`

`len` returns the size of arrays, slices, and dictionaries:

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

It has no return value and raises a runtime error if the key is absent.

---

# Modules and Imports

A Hornet source file can import another `.ht` file with:

```hornet
import 'utils'
```

The imported module is referenced through a qualifier derived from the file name:

```hornet
import 'utils'

int value = utils.helper()
```

An explicit qualifier can be supplied:

```hornet
import 'utilities/string_helpers' as strings
```

Specific declarations can be imported directly:

```hornet
from 'utils' import helper
```

Aliases are supported:

```hornet
from 'utils' import helper as h
```

Multiple names can be imported in one statement:

```hornet
from 'utils' import helper, other as o
```

### Module Resolution

For version 1 of the module system, one file is one module.

An import path is resolved by:

1. Looking relative to the importing file.
2. Falling back to the compiler's bundled `stdlib/` directory if no local file matches.

A local module always takes precedence over the standard library.

Imported modules are discovered transitively and merged into one semantic program before semantic analysis. Circular imports are therefore allowed by the module-discovery layer. Imported declarations receive globally unique internal names during the merge step; `extern` declarations are kept unmangled because their names must match the foreign symbol being linked.

Top-level names beginning with `_` are private to the module that defines them; other top-level declarations are importable. Module names are currently derived from filenames. A package-style declaration that decouples module identity from filename is planned for the future.

---

# Foreign Functions and Intrinsics

Hornet can declare functions implemented outside the Hornet program with `extern`:

```hornet
extern int abs(int value)
extern int getpid()
```

An omitted return type means no return value:

```hornet
extern free(*byte p)
```

`extern` declarations are represented as ordinary calls in IR and resolved by the native linker.

The current FFI deliberately restricts function signatures to scalar and pointer types. Arrays, slices, structs, sum types, and `str` are not yet permitted directly in `extern` signatures because their Hornet representations do not currently define a general C-compatible ABI.

## Intrinsics

`intrinsic` is reserved for a small, compiler-defined set of operations that need to expose internal Hornet representations:

```hornet
intrinsic *byte _raw_ptr(str s)
intrinsic int _raw_len(str s)
intrinsic str _from_raw_parts(*byte p, int n)
```

Unlike `extern`, intrinsics are not an open-ended extension mechanism. The compiler recognizes a fixed set and validates each declaration against its required signature.

The standard library uses these primitives to build higher-level C interop helpers, such as converting between Hornet strings and NUL-terminated C strings.

---

# Runtime

Compiled Hornet programs are linked with the small native runtime in `runtime/runtime.c`.

The runtime currently provides language-level services including:

* `print`
* runtime error reporting via `hornet_panic`
* slice backing-storage growth via `hornet_slice_grow`
* hash-table support used by dictionaries
* buffer/stringification helpers used by runtime printing

The runtime is intentionally separate from the x86-64 backend. The compiler and IR describe operations; the runtime implements selected algorithms that are more naturally expressed as ordinary native code.

The runtime ABI is currently represented by a combination of C declarations, type-descriptor tags, and the compiler's IR/backend calling machinery.

---

# Compiler Architecture

Hornet has a conventional multi-stage native compiler pipeline:

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

The frontend constructs a complete `IRProgram` before the x86-64 backend lowers it. IR instructions represent values, memory operations, control flow, calls, and higher-level compiler operations without embedding x86 instruction syntax.

The backend owns architecture-specific concerns such as register allocation, stack-frame layout, the x86-64 calling convention, and instruction selection.

Some IR operations intentionally lower to runtime calls. A runtime function does not need a special IR call mechanism; it participates in the same native calling machinery as another external function.

---

# Project Structure

```text
lexer.py          Lexical analysis
parser.py         Parser and AST definitions
desugar.py        AST desugaring
semantic.py       Semantic analysis and type checking
modules.py        Module discovery
merge.py          Module merging/name resolution

ir/               Intermediate representation and IR construction
optimize/         IR optimization passes

codegen/          x86-64 backend and assembly representation
runtime/          Native Hornet runtime
stdlib/           Hornet standard-library modules

tests/            Compiler, runtime, and end-to-end tests
benchmarks/       Benchmark harness and baselines

build.py          Build a runnable executable
compile.py        Generate assembly
SELF_HOST_CHECKLIST.md
                  Current path toward writing the compiler in Hornet
Hornet.tmbundle/  TextMate grammar/bundle
hornet-vim/       Vim syntax/indentation support
```

The `ir/` package is intended to sit between the language/frontend and the architecture-specific backend. The backend should consume IR rather than source AST nodes.

---

# Standard Library

The bundled `stdlib/` directory currently contains small Hornet modules that build useful functionality on top of the runtime/FFI boundary.

### `stdlib/c.ht`

Provides low-level C interoperability helpers, including conversion between Hornet `str` values and NUL-terminated C strings.

### `stdlib/hash.ht`

Provides hashing functions used by the dictionary implementation and available to Hornet code:

```hornet
hash_int64(x)
hash_int(x)
hash_bool(x)
hash_byte(x)
hash_str(s)
```

The standard library is deliberately small and is expected to grow as Hornet begins to support larger real-world programs.

---

# Complete Examples

## Fibonacci

```hornet
def int fib(int n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    return fib(n - 1) + fib(n - 2)


def int main():
    for int i = 0; i < 20; i += 1:
        print(fib(i))
    return 0
```

## Dictionaries

```hornet
def int main():
    dict[str]int counts = dict[str]int{
        'red': 2,
        'green': 1,
    }

    counts['red'] += 1

    if 'green' in counts:
        print(counts['green'])

    del(counts, 'green')
    print(counts)
    return 0
```

## Structs and Pointers

```hornet
type Point struct:
    int x
    int y

    def int sum(p):
        return p.x + p.y


def int main():
    Point point = Point(10, 20)
    *Point p = &point

    p.x = 100
    print(p.sum())

    return 0
```

## Sum Types

```hornet
type Circle struct:
    int radius

type Square struct:
    int side

type Shape is Circle | Square


def int area_like(Shape shape):
    match shape as s:
        is Circle:
            return s.radius * s.radius
        is Square:
            return s.side * s.side


def int main():
    Circle c = Circle(5)
    Shape shape = c

    print(area_like(shape))
    return 0
```

---

# Current Limitations

Hornet is still an experimental language. Some of the most significant limitations in the current implementation are:

* Pointer-to-pointer types are not yet supported semantically.
* `is` narrowing is limited to dedicated `if`/`elif` conditions rather than arbitrary boolean expressions, and flow-sensitive narrowing is intentionally incomplete.
* Sum types cannot currently contain other sum types as variants.
* Slice equality and sum-type equality are not implemented.
* `in` currently supports dictionary membership only; array, slice, and string membership are future work.
* `len` does not currently support strings.
* `extern` signatures are limited to scalar and pointer types.
* The standard library is intentionally small.
* Error handling is still based largely on explicit status values and runtime failure rather than a dedicated result/exception abstraction.
* There is no garbage collector yet; the project still relies on explicit/native allocation and compiler-managed escape behavior.
* There are no floating-point types yet.
* There are no generic types or generic functions yet.
* There are no first-class function types or closures yet.
* There is no variadic-call support yet.
* There is no multithreading model yet.
* The native backend currently targets x86-64 Linux and macOS.

---

# Self-Hosting Direction

One of the longer-term goals of the project is to rewrite the compiler itself in Hornet.

The current compiler already has many of the language features that such a compiler needs: structured data, arrays and slices, dictionaries, pointers, tagged unions, pattern matching, modules, FFI, and native compilation.

The remaining work for a practical self-hosted compiler is increasingly concentrated in the standard library and operating-system interface rather than in the core compiler backend.

The repository contains `SELF_HOST_CHECKLIST.md`, which tracks the current bootstrap requirements.

The intended progression is roughly:

```text
file and argument handling
        ↓
useful string/byte APIs
        ↓
more standard-library data structures
        ↓
write a non-trivial CLI in Hornet
        ↓
port the lexer/parser
        ↓
port semantic analysis
        ↓
port IR construction/optimization
        ↓
port the native backend
        ↓
self-hosted Hornet compiler
```

A garbage collector is desirable for long-running applications but is not required to bootstrap a compiler: a compiler can allocate compiler objects for the duration of one invocation and rely on process termination to reclaim them.

---

# Roadmap

Near- and medium-term development is focused on making Hornet useful for larger real-world programs while keeping the language relatively small.

Areas of active/future work include:

* Expanding the standard library, especially filesystem, process, path, and byte/string facilities
* Better error-handling abstractions
* More complete pointer and address-taking support
* Flow-sensitive sum-type narrowing
* More flexible sum-type composition
* Generic types and functions
* First-class function types and closures
* String formatting and richer text processing
* Regex support
* Variadic functions and richer FFI
* Passing structs across the FFI boundary
* Better memory management, potentially including a garbage collector
* Additional IR optimizations such as common-subexpression elimination and bounds-check elimination
* Improved aggregate copying and data layout
* More sophisticated register allocation
* Potential targets beyond x86-64

The language and compiler are intentionally developed together: new features are expected to have corresponding parser, semantic-analysis, IR, backend/runtime, and integration tests.

