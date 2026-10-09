# Hornet

Hornet is an experimental, statically typed programming language with an indentation-based syntax and a native compiler targeting x86-64 and AArch64 on Linux and macOS, and x86-64 on Windows.

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

The native backends target x86-64 and AArch64 on Linux and macOS, and x86-64 on Windows. On macOS the architecture is chosen with `gcc -arch`, so an Apple Silicon Mac can also build x86-64 programs and run them under Rosetta 2.

Building for a foreign Linux architecture needs a cross toolchain named `<arch>-linux-gnu-gcc`, and running the result needs qemu-user (on Debian/Ubuntu: `apt install gcc-aarch64-linux-gnu qemu-user`). The test suite uses them when present and skips what they're needed for otherwise.

Windows programs are built with MinGW-w64: its `gcc` on Windows itself, or the cross compiler `x86_64-w64-mingw32-gcc` elsewhere, where Wine runs the result (on Debian/Ubuntu: `apt install gcc-mingw-w64-x86-64 wine64`). They are linked with an 8 MB stack, the usual size elsewhere, in place of Windows's 1 MB. MSVC's toolchain is not supported.

### Build an Executable

The recommended way to build a runnable Hornet program is:

```bash
python3 build.py program.ht -o program
./program
```

`build.py` compiles the Hornet source, compiles the bundled native runtime, and links the two together into an executable. The compiled runtime is cached per target under `$HORNET_CACHE_DIR` (default `~/.cache/hornet`).

The target is written `arch-os`: `x86_64-linux`, `x86_64-macos`, `x86_64-windows`, `aarch64-linux`, or `aarch64-macos`.

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

`--dump STAGE` prints what a stage of the compiler produces instead of assembly: `tokens`, `tree` (the parser's), `typed` (the typed tree; see [Compiler Architecture](#compiler-architecture)), `ir`, or `optimized-ir`. A dump runs the compiler only as far as its stage, so `--dump tokens` works on a file that doesn't parse. `tokens` and `tree` are of the file named; the others are of the whole program (`typed` shows every function, and the IR dumps those that are compiled).

### Errors

Compile errors are reported as `file:line:col: error: message` with the source line and a caret; several semantic errors can be reported in one run. A program built into an executable must define `main`. Both tools exit with status 1 for errors in the program and 2 for internal compiler errors. `--traceback` shows the Python traceback instead.

### Testing

Run the test suite from the repository root:

```bash
pytest           # unit tests, plus end-to-end tests on the host target
pytest --quick   # unit tests only; nothing is built or run
pytest --full    # every runnable target, plus slow tests (benchmark reruns, formatter fuzzing, sanitizers)
```

The tests cover the lexer, parser, semantic analysis, modules, IR construction and verification, optimization, the native backend, escape analysis, runtime behavior, and end-to-end compiled programs, including seeded random programs checked against a Python model. The networking library is tested against a server on the same machine; nothing leaves it.

Backend-specific tests live in `tests/backend/<arch>/` and shared backend tests in `tests/backend/common/`. With `--full`, end-to-end programs are built and run for every target that can run on the machine (natively, under Rosetta 2, under qemu-user, or, for Windows, under Wine), and every such target must produce the same output. The few tests of what systems define differently, such as an exit status above 255, say which targets they apply to. `HORNET_E2E_TARGETS` overrides the targets in any tier:

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

It keeps line breaks, and a file's line endings (`\n` or `\r\n`), and normalizes the rest: 4-space indentation (plus 4 per open bracket on continuation lines), canonical spacing around operators, commas, and colons, two spaces before an inline comment and one after its `#`s (`##header` becomes `## header`), at most one blank line in a row, and exactly one blank line around multi-line top-level definitions (consecutive one-line declarations stay together). A list whose closing bracket is on its own line gets a trailing comma; one closed on the same line loses it. Flags may appear anywhere among the arguments. The repository's own `.ht` files are kept formatted by the test suite.

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
* Enums, with methods, compared, matched exhaustively, and printed by name
* `if`, `elif`, `else`, and `match`
* `while` loops
* C-style `for` loops
* `for ... in ...` iteration over arrays, slices, dictionaries, and strings
* `break` and `continue`
* Dictionaries with hashing, deletion, membership, and iteration; copies share one table
* Explicit integer casts
* Arithmetic, comparison, logical, bitwise, and membership operators
* Compound assignment
* Runtime checks (bounds, division, `none`, missing keys, and more) that panic with a source position
* Built-ins such as `print`, `len`, `append`, `del`, `bytes`, `panic`, and `format`, and `str(...)` conversions
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

An integer literal takes its type from where it is used, and must fit it: the slot it flows into, or the other operand (`fd < 0`, `a + 1`, `1 in xs`).

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

Strings are byte-oriented rather than a Unicode text abstraction. In a literal, the escapes are `\n`, `\t`, `\r`, `\0`, `\\`, `\'`, `\"`, and `\xNN` (a byte, as two hexadecimal digits); a backslash before anything else is a compile error. A literal ends on the line it starts; write `\n` for a newline.

A source file is read as bytes, so a literal holds exactly the bytes the file holds: UTF-8 text stays UTF-8, and `len('é')` is 2. Any bytes may appear in a string literal or a comment; everywhere else source is ASCII. Line and column numbers, in compile errors and in panics, count bytes. A file may not start with a byte-order mark.

String concatenation uses `+`:

```hornet
str message = 'hello ' + 'world'
```

`format` builds a string from text and values of any type (see [Built-ins](#built-ins)):

```hornet
str line = format('{} has {} items', name, len(items))
```

Strings compare with `==` and `!=`, and order with `<`, `>`, `<=`, and `>=`: byte by byte, a string coming before the longer strings it begins (`'apple' < 'apples'`, and `'Z' < 'a'`).

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

An array literal takes its type from where it is used: the declared type it flows into (a variable, an assignment, an argument, a return value, a field, or an element), or the other operand of `in`, `==`, or `!=`. Anywhere else its type is written before it:

```hornet
if kind in [Kind.Name, Kind.Number]:      # the elements take kind's type
    count += 1
if values == [1, 2, 3]:                   # ... or the other array's
    count += 1
print([3]int[1, 2, 3])                    # nothing here says, so the literal does
for n in [2]int[10, 20]:
    count += n
```

A bare literal given to a sum type needs its type too, since the sum could hold an array or a slice.

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

A variant may be a struct, scalar, enum, `str`, array, slice, dictionary, or pointer type, or `none`, a variant with no payload. A sum's zero value is its `none` variant. A sum without one has no zero value, so a variable of it (or a struct or array containing it) needs an initializer, and named construction can't omit such a field.

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

A sum type named among another's variants gives that sum its own variants in its place:

```hornet
type StrResult is str | Error
type IntResult is int | Error

type LineResult is StrResult | none      # str, Error, none
type Either is IntResult | StrResult     # int, Error, str
```

So a sum's variants are always a flat set of types that aren't sums. One that arrives twice, as `Error` does in `Either`, is one variant; a sum can't include itself. A pointer to a sum, or a slice or dictionary of them, is a variant in its own right, not a sum to be taken apart: a result that holds a tree is `type ExprResult is *Expr | ParseError`.

A sum's value flows into any wider sum, one that has every variant it has, wherever a value of the wider type is expected:

```hornet
def LineResult next_line():
    return read_some()                   # a StrResult
```

That is true of the value only: a `*StrResult` is not a `*LineResult`, nor a `[]StrResult` a `[]LineResult`. The other direction needs a check first; see [Pattern Matching](#pattern-matching).

The representation uses a discriminant and payload storage for the largest variant.

## Enums

An enum is a type with a fixed set of named members, one per line:

```hornet
type Color enum:
    Red
    Green
    Blue

Color c = Color.Green
```

A member is written `Color.Green` (`palette.Color.Green` through a module qualifier). Enums compare with `==` and `!=`, key dictionaries, and print as written (`Color.Green`); they have no ordering or arithmetic. `is` tests one, and `match` covers one exhaustively (see [Pattern Matching](#pattern-matching)). An enum's zero value is its first member.

A member's value is its position, from 0. Conversions are explicit:

```hornet
int n = int(c)            # 1
Color d = Color(n)        # panics if no member has that value
bool ok = n in Color      # whether one does
str name = str(c)         # 'Green'
int count = len(Color)    # 3
```

`Color(2)` with a literal is checked at compile time. `len(Color)` is a constant, so `[len(Color)]int` has one element per member.

An enum may have methods, written after its members as a struct's are after its fields (see [Structs and Methods](#structs-and-methods)):

```hornet
type Light enum:
    Red
    Green

    def Light other(self):
        if self is Red:
            return Light.Green
        return Light.Red

    def flip(*self):          # a pointer receiver changes the variable it is called on
        *self = self.other()

Light light = Light.Red
light.flip()
print(Light.Red.other())      # Light.Green
```

A method can't have a member's name, and one whose name starts with `_` is private to the enum's module.

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
an enum
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

Without one, a variable holds its type's zero value: `0`, `false`, `''`, `none` for pointers, an empty slice, a new empty dictionary, the `none` variant for a sum that has one, an enum's first member, and arrays and structs of zero values. A sum without a `none` variant has no zero value.

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

`const` declares a top-level constant of an integer type, `bool`, `str`, or an enum:

```hornet
const int BASE = 1
const int NEXT = BASE + 1
const int LIMIT = 1 << 16
const str GREETING = 'hello ' + 'world'
const bool DEBUG = LIMIT > 1000 and not false
const Color DEFAULT = Color.Green
```

The value is computed at compile time from literals, other constants (in any order), operators, string concatenation, integer casts, and an enum's members and conversions, with the same wraparound as at runtime. Integer constants can size arrays. Constants are imported and qualified like other top-level names (`from 'limits' import LIMIT`, `limits.LIMIT`), and a leading `_` makes one private. They can't be assigned or have their address taken, and a local variable, parameter, or loop binding can't reuse the name of a constant visible in its file.

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

A function declared `never` doesn't return: every path through it ends in a call to another `never` function (such as `panic`) or in a `while true` it doesn't break out of. A call to one ends its path, so no `return` is needed after it. An `extern` can be `never` too; if it returns after all, the program panics.

```hornet
def never fail(str what):
    panic('failed: ' + what)
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

A field or method whose name begins with `_` is private to the module that declares the struct: other modules can't read or write the field, call the method, or give the field a value in a struct literal (so they can't use a positional literal for that struct). They can still declare one, name its public fields in a literal, copy it, compare it, and print it.

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

A literal's type may be any array or slice type, pointers included: `[2][1]*P[[&p], [&q]]`. (Those tokens could also be an indexed literal times an index, `[x][0] * ys[1]`; they are read as the typed literal, and an indexed literal needs its type anyway: `[1]int[x][0] * ys[1]`.)

Indexing and slicing are bounds checked at runtime; a failed check's panic says what it compared (`index out of bounds: index 7, length 3`). A slice of a slice may extend up to its capacity.

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

Reading, updating (`counts['green'] += 1`), or deleting a key that isn't there panics, naming the key (`dict lookup: key not found: 'green'`); test with `in` first.

`len` returns the number of live entries.

The runtime uses open addressing with linear probing and tombstones. A table that fills up is rehashed: into one twice the size, or one of the same size when deletions have left it mostly tombstones.

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

Each iteration has its own bindings, so taking a binding's address is allowed. Reallocating the iterated slice (e.g. by `append`), or inserting into the iterated dictionary so that it is rehashed, panics.

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

`is` and `is not` bind like `==`; see [Pattern Matching](#pattern-matching). Ordering applies to integers of the same type, and to strings (byte by byte). Equality applies to integers, `bool`, `str`, enums, pointers, and arrays and structs of comparable types. Pointers can be compared with `none`, and so can a sum with a `none` variant (`x == none` means `x is none`); slices and dictionaries are never `none`.

### Membership

```text
in
not in
```

Membership applies to dictionary keys and array or slice elements. With an enum on the right, `n in Color` tests whether an integer is a member's value.

### Logical

```text
and  or  not
```

`and` and `or` use short-circuit evaluation. `not` binds looser than comparisons and `in`, and tighter than `and` and `or`: `not a < b` is `not (a < b)`.

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

### Evaluation Order

Operands and arguments are evaluated left to right, and each one's value is fixed when it is evaluated: in `f(s, change(&s))`, `f` receives `s` as it was before `change` ran. A plain assignment evaluates its value before its target.

---

# Pattern Matching

`x is T` tests whether a sum holds variant `T`, and `x is not T` that it doesn't. It is an ordinary `bool` expression, and on a variable it also narrows:

```hornet
if shape is Circle:
    print(shape.radius)
```

Where a check is known to be true the variable is that variant, and where it is known to be false it isn't. With one variant left, the variable has that variant's type:

```hornet
if shape is Circle:
    return shape.radius
return shape.side       # shape is a Square here
```

What is known follows `and`, `or`, and `not`; the `else` branch; the code after an `if` chain in which every branch but one always leaves (by `return`, `break`, `continue`, or a `never` call); the body of a `while` or `for`, from its condition; and the code after a `while` with no `break`. `x == none` and `x != none` narrow like `x is none`:

```hornet
if node != none and node.value > 0:     # the right side sees a Cons
    print(node.value)
while node is Cons:
    total += node.value
    node = *node.next                   # assigning ends the narrowing
```

Assigning to a narrowed variable ends its narrowing, and a loop assumes nothing about a variable its body assigns. A compound assignment (`count += 1`) is no such assignment: it gives the variable back the variant it held, so the narrowing stands. Only variables narrow: `h.shape is Circle` is a plain test. Nothing narrows to `none`: with only `none` left, the variable keeps its sum type. If a pointer to the variable (`&shape`) changes its variant meanwhile, its next use panics.

`T` may be a sum type whose variants are all the subject's: the test is then for any of them. A variable with several variants left keeps its type, but it flows into any sum that has all those it can still hold:

```hornet
LineResult line = next_line()
if line is StrResult:
    show(line)                          # show takes a StrResult
if line is none:
    return
show(line)                              # and here, none having been ruled out
```

`as NAME` binds a narrowed copy of the subject, which is how a field, an element, or a call's result is narrowed:

```hornet
if shape is Circle as circle:
    print(circle.radius)
if e is Neg as neg and *neg.operand is Num as n:
    return n.v
```

A binding belongs to an `if` or `elif` condition, as the whole condition or one of the checks joined by `and`. A whole-condition binding lasts for the `if` and its `else`, where it is narrowed the same way; one joined by `and` is made only when the checks before it hold, and lasts for the rest of the condition and the body. `NAME` is a copy, so assigning to its fields doesn't change the subject.

`match` provides multiple variant arms:

```hornet
match shape as s:
    is Circle:
        print(s.radius)
    is Square:
        print(s.side)
```

The subject of `match ... as NAME` can be any sum-typed expression, such as a field, an element, or `*p`; `NAME` is a copy of it. A `none` variant is tested with `is none`.

`match` must be exhaustive, or end with `else:`, where the subject is none of the arms' variants. A `match` whose arms all return, or end in a `never` call such as `panic`, counts as returning. An arm may test for a sum type, taking each of its variants that no arm above has; an arm with nothing left to take is an error.

`is` and `match` also apply to an enum, with its members in place of variants (`colour is Red`). Nothing is narrowed, and a `match` must cover every member or end with `else:`:

```hornet
match colour:
    is Red:
        print('stop')
    is Color.Green:
        print('go')
    else:
        print('wait')
```

---

# Built-ins

The compiler currently provides the following built-in operations:

```text
print
len
append
del
bytes
panic
format
```

`str(...)` conversions are described under [Strings](#strings) and [Enums](#enums). A builtin's name can't be declared, or given to an import, in any file; a method or a variable may share one.

## `print`

`print` can print supported scalar and aggregate values:

```hornet
print(42)
print('hello')
print([3]int[1, 2, 3])
print(counts)
```

Arrays, slices, structs, dictionaries, and supported sum-type values are recursively formatted by the native runtime. A pointer prints as its address, and as `none` when it points to nothing; `print(none)` prints `none` too.

## `len`

`len` returns the size of an array, slice, dictionary, or string (in bytes):

```hornet
print(len(values))
print(len(view))
print(len(counts))
print(len('hello'))
```

`len(Color)` is an enum's number of members.

## `append`

`append` adds an element to a slice and returns the resulting slice:

```hornet
[]int values = []int[1, 2]
values = append(values, 3)
```

## `bytes`

`bytes(s)` returns a new `[]byte` copy of the string `s`.

## `panic`

`panic(message)` prints a `str` with the call's position (`file:line:col: panic: message`) and aborts. It is a `never` call.

## `format`

`format(template, values...)` builds a `str`: the template with each `{}` replaced by the next value, shown as `print` shows it.

```hornet
str a = format('{} + {} = {}', n, 1, n + 1)          # '3 + 1 = 4'
str b = format('{}:{}: error: {}\n', line, col, message)
str c = format('{} is {}', Color.Green, values)      # 'Color.Green is [3]int[1, 2, 3]'
str d = format('a brace pair: {{}}')                 # 'a brace pair: {}'
```

A value may have any type `print` accepts. A `str` appears as its text, and a string inside another value is quoted; `quoted` in `stdlib/strings.ht` quotes one on its own. The template must be a string literal or a `str` constant, because it is checked when compiling: its `{}` must match the values in number, `{{` and `}}` are a brace each, and nothing may be written inside a placeholder yet (there are no widths or other format options). The values are evaluated once, from left to right, and the whole result is built in one buffer.

## `del`

`del(dict, key)` removes an existing dictionary entry:

```hornet
del(counts, 'obsolete')
```

A missing key panics, naming the key.

---

# Modules and Imports

A module is currently one `.ht` source file, named by that file: the name (without `.ht`) must be an identifier, and unique among the program's modules. The entry file can't be imported.

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

Top-level names beginning with `_` are private to their module, as are a struct's fields and methods (see [Structs and Methods](#structs-and-methods)) and an enum's methods.

Every declaration is reached this way, including `extern` functions: a module that uses another module's `extern` imports it (`from 'os' import write_fd`) or qualifies it (`os.write_fd(...)`). A local variable, parameter, or loop binding can't have the name of an import alias or a constant in scope.

Each file's names are resolved in that file's own scope. Internally every declaration of an imported module gets a program-wide unique name, while `extern` symbols keep their foreign names for the linker; errors and printed values use names as declared. A function in the entry file is linked as `name$` (`main` keeps its name), so it never collides with a libc or runtime symbol.

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

Hornet's `int` is 64-bit; declare C `int` parameters and results as `int32` (for example `extern int32 close(int32 fd)`). `long`, `size_t`, and `ssize_t` are `int` (except on Windows, where C's `long` is 32-bit: `int32`).

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
memory from the runtime (hornet_alloc, hornet_alloc_zeroed)
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

Integer formatting and parsing:

```hornet
str a = int_to_str(-1234)       # '-1234'
str b = int_to_hex(255)         # 'ff'
str c = pad_left('7', 3, "0")   # '007'
IntResult n = parse_int('-42')  # an Error if it isn't a decimal integer, or doesn't fit
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
int order = compare('apple', 'pear')   # negative, zero, or positive: which comes first, by bytes
str q = quoted('it\'s')                 # the str as a literal: 'it\'s', quotes and escapes included
```

`compare` gives the order `<` does, as one three-way answer. `quoted` is for showing a string in a message without ambiguity: control bytes are escaped, and UTF-8 text is left as it is.

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
LineResult line = read_line()   # the next line of standard input, without its ending; none at the end
bool at_terminal = is_terminal(0)
StrResult fd_contents = read_all_from_fd(fd)
IntResult written = write_file('output.txt', 'hello')
write_stdout('no trailing newline')
write_stderr('error\n')
exit(0)
```

Functions that can fail return a result from `stdlib/errors.ht` (`StrResult is str | Error`, `IntResult is int | Error`); handle it with `match` or `is`, or use `must_str`/`must_int` to panic on error. `read_line` returns a `LineResult is StrResult | none`, and can be mixed with `read_stdin`, which then reads what is left. Directory/path APIs remain future work.

## `stdlib/net.ht`

TCP connections:

```hornet
ConnResult opened = connect('example.com', 80, 5000)   # a name or an address, a port, a timeout in milliseconds
if opened is Conn as conn:
    conn.write('ping\n')            # an IntResult: how many bytes were sent
    StrResult reply = conn.read()   # what has arrived, up to 64 KB; '' once the other side has closed
    conn.close()
```

The timeout bounds each read and write; 0 waits as long as the system does. `connect` returns a `ConnResult is Conn | Error`.

## `stdlib/http.ht`

HTTP/1.1 requests over plain TCP:

```hornet
ResponseResult r = get('http://example.com/')
if r is Response:
    print(r.status)                   # 200
    print(r.header('content-type'))   # found whatever its case; '' if there is none
    print(r.body)
```

`post(url, content_type, body)` sends a body, and `request(method, url, headers, body, timeout_ms)` is the general form, with `headers` a `[]Header`. `parse_url` splits a URL into its host, port, and target. A response's body is read whole, whether it comes with a length, in chunks, or until the connection closes.

There is no TLS yet, so an `https://` URL is an `Error`. Redirects are the caller's to follow, and each request uses a connection of its own.

---

# Runtime

The native runtime is located in `runtime/runtime.c` and is compiled separately from the generated Hornet assembly.

It currently provides language-level services including:

* `print` and recursive value formatting
* the panic routines (`hornet_panic`, `hornet_panic_at` for `panic(...)`, and those that add the values a failed check compared), which flush standard output, write the message to standard error, and abort (`SIGABRT`; on Windows, exit code 3)
* catching a stack overflow, to report it as a panic
* TCP sockets for `stdlib/net.ht` (`hornet_tcp_connect`, `hornet_tcp_read`, `hornet_tcp_write`, `hornet_tcp_close`, and `hornet_net_error` for why one failed), since C's sockets differ between systems; on Windows the program is linked with `ws2_32`
* memory: `hornet_alloc` and `hornet_alloc_zeroed`, which every allocation goes through, the compiled code's and the runtime's own, and which panic when memory is refused
* `format`'s buffer (`hornet_format_begin`, `hornet_format_text`, `hornet_format_value`)
* `hornet_slice_grow`, which copies a slice into a larger backing store
* `hornet_bytes` for `bytes(s)`
* dictionary hash tables, hashed with FNV-1a (`hornet_hash_bytes`)
* runtime type descriptors (`hornet_typedesc_tags.h` is generated from `ir/typedesc.py` by `runtime/generate_typedesc_header.py`)
* argument access, file and stream I/O (as bytes on every system), exit, and OS error messages for `stdlib/os.ht` (`hornet_argv_get`, `hornet_open_read`, `hornet_open_write`, `hornet_read_fd`, `hornet_read_line`, `hornet_is_terminal`, `hornet_write_fd`, `hornet_close_fd`, `hornet_exit`, `hornet_error_message`)

Runtime checks panic with `file:line:col: panic: message`, the position being where the checked expression or statement starts and the file named without its directory: array, slice, and string bounds; division by zero and `MIN / -1`; dereferencing `none`; a missing dictionary key; an integer converted to an enum with no such member; reallocating a slice or dictionary being iterated with `for ... in`; and a narrowed variable whose variant was changed through a pointer. Where a value explains the failure, the message ends with it: `index out of bounds: index 7, length 3`, `slice bounds out of range: start 5, end 2`, `dict lookup: key not found: 'apple'`, `not a member of Color: 7`. The checks are in the compiled code, which calls the runtime's panic routines.

Running out of stack is a panic too: `panic: stack overflow`, with no position, since the runtime only catches the fault (with a signal handler on a stack of its own; on Windows, an exception handler). On Linux a stack with no limit (`ulimit -s unlimited`) has no end to detect.

Being refused memory is one as well: `panic: out of memory (allocating 1048576 bytes)`. A system that grants memory it doesn't have, as Linux does by default, kills the program when the memory is used instead, and that can't be caught.

The runtime is deliberately separate from the native backends. The compiler is responsible for semantic operations such as type checking, aggregate layout, address calculation, and bounds-check generation; the runtime implements selected algorithms and services that are better expressed as ordinary native code.

The runtime currently uses the platform C library for lower-level services such as byte copying, and for memory behind its own checked allocators, rather than wrapping every libc primitive in a Hornet-specific API.

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
IR construction ── of the functions main can reach; escape analysis, null and division checks
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

Semantic analysis never changes the parser's ASTs: it resolves each file's names in that file's scope (`scopes.py`), records what it learns (types, the declaration each name refers to, narrowing) by node number, and from that builds the typed tree (`typed_ast.py`). The typed program `semantic.analyze()` returns is the only input to later stages: a test checks that nothing in `ir/`, `optimize/`, `backend/`, or escape analysis imports the front end. In the typed tree every node has one meaning and a concrete type: names refer to symbols, the parser's overloaded forms are split (calls, struct literals, and builtins; array, slice, string, and dictionary indexing), each implicit operation is a node (widening into a sum, `&Variant(...)`, a literal becoming a slice, zero values), methods are ordinary functions, and `match` and compound assignment are nodes of their own. `compile.py --dump typed` prints it.

Both frontend stages are packages. The parser (`parser/`) is functions over a token stream, which holds a file's tokens and the position in them and is all the state parsing has. There is one module for each part of the grammar: declarations call statements, which call expressions and types. Semantic analysis (`semantic/`) runs in phases that `analyzer.py` sequences. Declarations are resolved first, all of them, and the constants with them; then each function's body is checked, statement by statement, the statements calling into expression checking; then the typed tree is built from what checking recorded. The names in scope, and what `is` checks have narrowed them to, are kept by one object that both statement and expression checking use.

The frontend constructs a complete `IRProgram` before a backend begins lowering it. Only the functions `main` can reach through calls are built: every function is type-checked, but one that can never run is not compiled, and neither are the strings and descriptors only it uses, so importing a module costs what is used of it. (Calls are all resolved by then, and nothing else can call a function. A program without `main` keeps every function.) The IR is independent of any target: a function's incoming arguments are an ordered list of word-sized temporaries in Hornet's own calling convention (a composite return value's destination address first, then one word per parameter, except two for `str` and three for slices, with arrays, structs, sum types, and dicts passed by address), and where each word physically arrives is decided by the backend. Lowering never modifies the IR. An enum value reaches the IR as an `int32`.

Escape analysis decides which locals must live on the heap; heap storage comes from the runtime's `hornet_alloc` and is never freed. It is a flow-insensitive points-to analysis per function, run on the typed tree, with per-parameter escape summaries so that passing `&x` to a function that doesn't keep the pointer leaves `x` on the stack.

IR construction compiles conditions made of `and`, `or`, and `not` straight to branches, and copies a composite value directly between places (two places of one type are the same storage or disjoint). Loads, stores, and copies address memory as a base plus a constant offset.

The IR optimizer repeats constant folding, identity simplification, constant-branch and unreachable-block removal, copy and constant propagation within blocks, copy coalescing, address folding (an address computed as a base plus a constant becomes the access's offset), and dead-code elimination until nothing changes. `ir/cfg.py` provides the shared control-flow and liveness analysis.

`backend/` holds one package per architecture, chosen by the target, plus `backend/common/` for what they share: linear-scan register allocation over the target's register lists (values live across calls get callee-saved registers), stack-frame slot layout, the magic numbers for division by constants, and jump cleanups. Each backend adds its calling convention (for `backend/x86_64/`, SysV or, on Windows, the x64 convention with its shadow space, stack probes, and unwind tables; argument registers are also allocatable, so incoming parameters and outgoing arguments move as parallel moves; AAPCS64 for `backend/aarch64/`), prologues that save only the registers a function uses, instruction selection that works directly on registers, stack slots, and immediates, a peephole pass, and assembly emission for each system (AT&T syntax on x86-64). The AArch64 backend also rewrites accesses to stack slots beyond the reach of a load or store's offset after the frame is laid out.

Some IR operations deliberately lower to runtime calls. A runtime operation does not require a special calling mechanism; runtime functions participate in the same native call machinery as other external functions.

---

# Project Structure

```text
lexer.py           Lexical analysis
parser/            Parsing: the tree's nodes (nodes.py), the token stream (stream.py), and the grammar
                   as functions over it (declarations.py, statements.py, expressions.py, type_exprs.py)
semantic/          Semantic analysis: declarations.py first, then statements.py and expressions.py (with
                   calls.py) for each body, over the scopes in flow.py; constants.py for constant
                   values; typed_tree_builder.py builds the typed tree; analyzer.py runs them in order
typed_ast.py       The typed tree and its text dump
dump.py            What each stage produces, as text (`compile.py --dump`)
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

examples/          Example Hornet programs (`calc/` and `json/` are multi-file ones; `fetch.ht` is an HTTP client)
tools/hfmt/        Source formatter, written in Hornet
tests/             Compiler, runtime, and end-to-end tests
benchmarks/        Benchmark programs and tooling

build.py           Build a runnable executable
compile.py         Generate native assembly
test.sh            Build, run, and delete one program: ./test.sh path/without_ext
Dockerfile, entrypoint.sh
                   Assemble and run an x86-64 `.s` file in a Linux container
find_long_lines.py, find_newline_runs.py
                   Report long lines, and runs of blank lines, in the sources
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

## A Calculator

`examples/calc/` is a larger example in eight modules: an integer calculator built the way a compiler is. A lexer and a precedence-climbing parser turn each line into a tree (a recursive sum type), which is either evaluated directly or simplified, compiled to code for a small stack machine, and run there.

```bash
python3 build.py examples/calc/main.ht -o calc
./calc                           # a prompt: each line is answered as it is entered
./calc examples/calc/demo.calc   # or a file's lines
```

```text
> x = 6 * 7
> x + 1
43
> 10 / (x - 42)
     ^ division by zero
```

Its values are 64-bit integers with `+ - * / %`, comparisons, `and`/`or`/`not`, parentheses, and variables. `--vm` evaluates through the stack machine, and `--tree`, `--simplified`, and `--code` print each line's tree, its simplified tree, and its code:

```text
$ echo 'x * 1 + 0 * 5' | ./calc --tree
(+ (* x 1) (* 0 5))
$ echo 'x * 1 + 0 * 5' | ./calc --simplified
x
```

## A JSON Formatter

`examples/json/` is a JSON reader and writer (`json.ht`) with a command-line tool around it:

```bash
python3 build.py examples/json/main.ht -o jsonfmt
./jsonfmt data.json                         # laid out, two spaces a level
./jsonfmt --compact --sort-keys < data.json
./jsonfmt --indent 4 data.json
```

A document is a recursive sum type, `Value is none | bool | Number | str | []Value | Object`. A number keeps its text, since there is no floating point to hold it, and an object keeps its members in order. A mistake is reported as `file:line:col: error: message`. The tests run it over the 318 files of JSONTestSuite (`tests/json/test_parsing/`), and compare its output with Python's for random documents.

## An HTTP Client

`examples/fetch.ht` is twenty lines over `stdlib/http.ht`: it writes the body of an `http://` URL to standard output and the status line to standard error.

```bash
python3 build.py examples/fetch.ht -o fetch
./fetch http://example.com/
```

---

# Current Limitations

Hornet is still experimental. Some notable limitations are:

* Pointer-to-pointer types are parsed but rejected semantically.
* Some advanced pointer/address-taking cases remain unsupported.
* `for ... in ...` can't iterate a function-call result or a dereference; assign it to a variable first.
* Only variables narrow; a field, element, or call result needs an `as` binding, which is for an `if` or `elif` condition and can't sit under `or` or `not`.
* Enums have no explicit member values or ordering, and `for ... in` doesn't iterate one.
* Sum-type equality is not implemented.
* Slice equality and dictionary equality are not implemented.
* `in` does not apply to strings.
* `format` has no widths, padding, or other format options, and there are no interpolated string literals.
* The FFI currently supports only scalar and pointer arguments/results.
* Passing structs by value through FFI is not yet supported.
* The standard library currently provides only a small subset of filesystem, process, path, formatting, and collection facilities.
* Generic types and generic functions are not implemented.
* First-class function types and closures are not implemented.
* Variadic functions and variadic FFI calls are not implemented.
* There is no garbage collector yet; string concatenation in particular never frees its intermediate strings.
* There are no floating-point types yet.
* There is no TLS: `stdlib/http.ht` makes plain `http://` requests only.
* Multithreading is not implemented.
* Without generics, each result type is a separate named sum type. There is no operator for propagating errors, and ignoring a result is not diagnosed.
* The panics for a stack overflow and for running out of memory have no source position.

---

# Self-Hosting

A longer-term goal is to rewrite the compiler itself in Hornet.

The current language already has the structural features needed by a compiler implementation: structs, enums, arrays, slices, dictionaries, pointers, recursive sum types (so an AST can be expressed), pattern matching, modules, FFI, and native compilation. What each stage of the Python compiler produces, printed by `compile.py --dump` (tokens, tree, typed tree, and IR), is the intended point of comparison between the two implementations.

The standard library now covers file and stream I/O, process exit, string building and searching, integer formatting and parsing, TCP connections and HTTP requests, and an error convention. The formatter in `tools/hfmt` is the first substantial tool written in Hornet; it includes a Hornet lexer that is tested token-for-token against the compiler's own. `examples/calc` is a compiler in miniature (lexer, parser, tree, simplifier, code generation, and a machine to run the code). The next step is porting the compiler itself, starting from that lexer.

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

* Expanding the standard library, especially directory, path, and process facilities, and TLS for the HTTP client
* Error-propagation syntax for result types
* More complete pointer and address-taking support
* More precise escape analysis for iterators, aliases, and data more than one pointer away from a call argument
* Narrowing of fields and elements, and `as` bindings in loop conditions
* Sum-type equality
* Generic types and functions
* First-class function types and closures
* Interpolated string literals, and format options
* Regex support
* Richer FFI, including aggregate types where a stable ABI can be defined
* Potential garbage collection and a more explicit ownership model
* More IR optimizations, including common-subexpression and bounds-check elimination
* Better aggregate copying and layout decisions
* More sophisticated register allocation, including spill choices weighted by use count and loop depth
* Additional native targets

Hornet is intentionally developed incrementally: new language features are accompanied by parser, semantic-analysis, IR, backend/runtime, and end-to-end tests whenever appropriate.

