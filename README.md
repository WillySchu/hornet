# Hornet

Hornet is a small, statically typed programming language with a Python-like indentation-based syntax and a native compiler targeting x86-64 Linux and macOS.

The language is designed around a relatively small set of language constructs, explicit types, value semantics, and a straightforward compilation model.

## Example

A complete Hornet program:

```hornet
def int main():
    int i = 0

    while i < 10:
        print(i)
        i += 1

    return 0
```

Hornet source files use the `.ht` extension.

---

# Getting Started

## Requirements

The compiler currently requires:

* Python 3
* An x86-64 assembler and C compiler/linker (`as` and `gcc`)
* Linux or macOS for the generated native code

The compiler itself is written in Python and has no third-party runtime dependency.

## Compiling a Program

From the `compiler` directory:

```bash
python3 compile.py program.ht
```

By default, this writes assembly to standard output.

To write the generated assembly to a file:

```bash
python3 compile.py program.ht -o program.s
```

The target platform can be selected explicitly:

```bash
python3 compile.py program.ht --platform linux -o program.s
```

or:

```bash
python3 compile.py program.ht --platform macos -o program.s
```

The compiler currently defaults to the Linux target when invoked through `compile.py`.

## Building the Executable

The compiler produces assembly rather than an executable directly. On Linux, the generated assembly can be assembled and linked with:

```bash
python3 compile.py program.ht --platform linux -o program.s
as -o program.o program.s
gcc -o program program.o
./program
```

The exit status of the program is available from the shell:

```bash
echo $?
```

The repository also contains `test.sh`, which demonstrates the basic assemble/link/run workflow.

---

# Language Overview

Hornet currently provides:

* Functions
* Static typing
* `int`, `int8`, `uint8`, `int64`, `bool`, and `str`
* Fixed-size arrays
* Slices
* Structs
* Type aliases
* Struct methods
* Local variables and block scoping
* `if`, `elif`, and `else`
* `while`
* `break` and `continue`
* Arithmetic and bitwise operators
* Boolean operators with short-circuit evaluation
* Comparisons
* Compound assignment
* Array and slice indexing
* Array slicing
* Struct field access
* Function calls
* Struct construction
* Explicit integer casts
* `print`, `len`, and `append`
* Runtime bounds checking

---

# Syntax

## Functions

Functions begin with `def`.

A return type may be specified before the function name:

```hornet
def int add(int a, int b):
    return a + b
```

A function without a declared return type is also allowed:

```hornet
def greet():
    print('Hello!')
```

Such a function may simply fall off the end of its body.

Parameters are declared with their type followed by their name:

```hornet
def int multiply(int a, int b):
    return a * b
```

Functions may call functions declared later in the source file, and recursive functions are supported:

```hornet
def int fib(int n):
    if n == 0:
        return 0

    if n == 1:
        return 1

    return fib(n - 1) + fib(n - 2)
```

A conventional entry point is:

```hornet
def int main():
    return 0
```

---

# Variables

Variables are declared with a type followed by a name:

```hornet
int x
bool finished
str message
```

An initializer may be provided:

```hornet
int x = 42
bool finished = false
str message = 'hello'
```

Variables must be declared before they are used.

Hornet uses block scoping. Variables declared inside an `if`, `else`, or `while` block are not visible after the block:

```hornet
def int main():
    int x = 10

    if x > 5:
        int y = 20
        print(y)

    # y is no longer in scope here

    return x
```

Shadowing an outer variable inside a nested block is permitted:

```hornet
def int main():
    int x = 10

    if true:
        int x = 20
        print(x)

    print(x)
    return 0
```

---

# Types

## Integer Types

Hornet currently provides four integer types:

```text
int
int8
uint8
int64
```

`int` is the normal integer type for most programs.

The narrower types have explicit ranges:

```text
int8   -128 .. 127
uint8     0 .. 255
```

Integer literals are checked against the destination type:

```hornet
int8 a = 100
uint8 b = 200
```

Values are not implicitly converted between integer types.

For example, this is rejected:

```hornet
int x = 10
int8 y = x
```

Use an explicit cast instead:

```hornet
int x = 10
int8 y = int8(x)
```

## Booleans

Boolean values are:

```hornet
true
false
```

Boolean expressions have type `bool`:

```hornet
bool a = x < 10
bool b = a and true
```

Booleans are distinct from integers. An integer cannot be used as a condition:

```hornet
# Invalid:
if x:
    ...
```

Instead, write an explicit comparison:

```hornet
if x != 0:
    ...
```

---

# Strings

Strings are written using single quotes:

```hornet
str greeting = 'Hello, world!'
```

String concatenation uses `+`:

```hornet
str first = 'Hello'
str second = 'world'
str message = first + ', ' + second
print(message)
```

String values can be printed directly.

---

# Arrays

Hornet supports fixed-size arrays.

The type includes the array's size:

```hornet
[3]int
[10]bool
[4]str
```

An array can be declared and initialized:

```hornet
[3]int numbers = [1, 2, 3]
```

The number of elements must match the declared size.

Arrays can be multidimensional:

```hornet
[2][3]int matrix = [[1, 2, 3], [4, 5, 6]]
```

Elements are accessed using indexing:

```hornet
int x = numbers[0]
numbers[1] = 42
```

Indexing is bounds checked at runtime.

---

# Slices

A slice represents a variable-length view over an array.

A slice type is written:

```hornet
[]int
[]str
[][3]int
[][]int
```

A slice can be created from an array:

```hornet
[5]int numbers = [1, 2, 3, 4, 5]
[]int first_two = numbers[0:2]
```

The lower and upper bounds can be omitted:

```hornet
[]int all = numbers[:]
[]int from_two = numbers[2:]
[]int through_two = numbers[:3]
```

Slices are views rather than copies. Modifying a slice modifies the underlying storage:

```hornet
[3]int numbers = [1, 2, 3]
[]int s = numbers[:]

s[0] = 100

print(numbers[0])
```

Slices can also be created directly:

```hornet
[]int numbers = []int[1, 2, 3]
```

The same syntax works with other element types:

```hornet
[]str names = []str['Alice', 'Bob', 'Carol']
```

A slice can be empty:

```hornet
[]int empty = none
```

`none` is the nil/zero value for slices.

---

# Structs

Structs are user-defined nominal types.

```hornet
struct Point:
    int x
    int y
```

A struct value can be constructed positionally:

```hornet
Point p = Point(10, 20)
```

Fields are accessed using `.`:

```hornet
print(p.x)
print(p.y)
```

Fields can be assigned:

```hornet
p.x = 100
```

Structs may contain other structs, arrays, and slices:

```hornet
struct Point:
    int x
    int y

struct Shape:
    Point origin
    [4]int values
    []str names
```

Struct values have value semantics. Passing or assigning a struct produces a value rather than an implicit reference to the original struct.

## Named Struct Construction

Struct fields can also be specified by name:

```hornet
Point p = Point(x=10, y=20)
```

Named construction can omit fields:

```hornet
Point p = Point(x=10)
```

Omitted fields receive their zero value.

Positional construction, on the other hand, must provide one argument for every field in declaration order:

```hornet
Point p = Point(10, 20)
```

---

# Struct Methods

Structs can contain methods.

```hornet
struct Point:
    int x
    int y

    def int sum(p):
        return p.x + p.y
```

The first parameter of a method is its receiver. It is written without a type because the enclosing struct determines its type.

Methods are called with normal dot syntax:

```hornet
Point p = Point(10, 20)
int result = p.sum()
```

Methods can have additional parameters:

```hornet
struct Point:
    int x
    int y

    def int distance_from(p, int x, int y):
        return (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y)
```

Methods may return structs or other composite types:

```hornet
struct Point:
    int x
    int y

    def Point doubled(p):
        return Point(p.x * 2, p.y * 2)
```

---

# Type Aliases

A type alias introduces another name for an existing type:

```hornet
type MyInt = int
```

Aliases can refer to other aliases:

```hornet
type MyInt = int
type Counter = MyInt
```

They can also refer to composite types:

```hornet
type Numbers = []int
type Matrix = [3][3]int
```

Aliases do not create distinct nominal types. They are alternate names for the underlying type.

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

Conditions must have type `bool`.

Nested blocks are supported:

```hornet
if x > 0:
    if x % 2 == 0:
        print('positive even')
    else:
        print('positive odd')
```

## `while`

```hornet
int i = 0

while i < 10:
    print(i)
    i += 1
```

`while` loops may contain `break` and `continue`:

```hornet
while i < 100:
    i += 1

    if i % 2 == 0:
        continue

    if i > 50:
        break

    print(i)
```

`break` and `continue` apply to the innermost enclosing loop.

---

# Operators

Hornet supports the following binary operators.

### Arithmetic

```text
+  -  *  /  %
```

Example:

```hornet
int result = (10 + 5) * 2 % 7
```

### Comparisons

```text
<  >  <=  >=
== !=
```

Comparisons produce `bool`:

```hornet
bool smaller = a < b
bool equal = a == b
```

### Logical operators

```text
not
and
or
```

These operate on `bool` values and use short-circuit evaluation:

```hornet
if x != 0 and y / x > 2:
    print('large')
```

### Bitwise operators

```text
&  |  ^
<< >>
~
```

For example:

```hornet
int flags = 1
flags = flags << 2
flags = flags | 1
```

`~` is the unary bitwise complement.

### Unary negation

```hornet
int x = -10
```

Unary operators can be chained:

```hornet
int x = --10
bool y = not not true
```

---

# Compound Assignment

Hornet supports compound assignment for arithmetic and bitwise operators:

```text
+=
-=
*=
/=
%=
&=
|=
^=
<<=
>>=
```

For example:

```hornet
int x = 10

x += 5
x *= 2
x >>= 1
```

Compound assignment is equivalent to assigning the corresponding binary expression:

```hornet
x += 5
```

is equivalent to:

```hornet
x = x + 5
```

---

# Function Calls

Functions are called using ordinary call syntax:

```hornet
int result = add(10, 20)
```

Arguments are evaluated normally:

```hornet
print(add(x, y))
```

Function calls can be nested:

```hornet
return multiply(add(a, b), add(c, d))
```

Functions may return arrays, slices, structs, or scalar values.

---

# Struct Literals and Named Arguments

The same call syntax is used for struct construction.

Positional:

```hornet
Point p = Point(10, 20)
```

Named:

```hornet
Point p = Point(x=10, y=20)
```

Named and positional arguments cannot be mixed:

```hornet
# Invalid:
Point p = Point(10, y=20)
```

---

# Built-in Functions

Hornet currently provides three built-in functions.

## `print`

`print` accepts any printable value:

```hornet
print(42)
print(true)
print('hello')
print([1, 2, 3])
print(s)
print(point)
```

Arrays, slices, and structs are recursively formatted for output.

For example:

```text
[1, 2, 3]
```

and a struct such as:

```hornet
struct Point:
    int x
    int y
```

may be printed as:

```text
Point(x: 10, y: 20)
```

`print` does not return a value.

## `len`

`len` returns the length of an array or slice:

```hornet
[5]int numbers = [1, 2, 3, 4, 5]

print(len(numbers))
```

For an array, the length is known at compile time. For a slice, it is determined from the slice at runtime.

`len` currently does not accept strings.

## `append`

`append` adds an element to a slice and returns the resulting slice:

```hornet
[]int numbers = []int[1, 2, 3]

numbers = append(numbers, 4)
numbers = append(numbers, 5)

print(numbers)
```

The appended value must have the same type as the slice's element type.

For example:

```hornet
[]int values = []int[1, 2]
values = append(values, 3)
```

Nested slices are supported as well:

```hornet
[][3]int rows

[3]int row = [10, 20, 30]
rows = append(rows, row)
```

---

# Explicit Casts

Hornet uses explicit casts for integer conversions.

```hornet
int x = 100
int8 y = int8(x)
```

The supported integer types can be explicitly converted:

```text
int
int8
uint8
int64
```

For example:

```hornet
int8 a = -5
uint8 b = uint8(a)
```

There are no implicit integer conversions.

---

# Comments

Comments begin with `#`:

```hornet
# This is a comment

def int main():
    int x = 10  # Comments can follow code
    return x
```

---

# Array and Slice Examples

The following example demonstrates several of Hornet's collection features together:

```hornet
def int main():
    [5]int numbers = [10, 20, 30, 40, 50]

    []int middle = numbers[1:4]

    print(middle)
    print(len(middle))

    middle[0] = 99

    print(numbers)

    return 0
```

Because `middle` is a view into `numbers`, modifying `middle[0]` also changes the corresponding element in `numbers`.

---

# Complete Example: Fibonacci

```hornet
def int fib(int n):
    if n == 0:
        return 0

    if n == 1:
        return 1

    return fib(n - 1) + fib(n - 2)


def int main():
    int i = 0

    while i < 20:
        print(fib(i))
        i += 1

    return 0
```

Compile and run:

```bash
python3 compile.py fib.ht --platform linux -o fib.s
as -o fib.o fib.s
gcc -o fib fib.o
./fib
```

---

# Complete Example: Structs

```hornet
struct Point:
    int x
    int y

    def Point doubled(p):
        return Point(p.x * 2, p.y * 2)

    def int sum(p):
        return p.x + p.y


def int main():
    Point p = Point(x=3, y=4)

    print(p)
    print(p.sum())

    Point q = p.doubled()

    print(q)

    return 0
```

---

# Complete Example: FizzBuzz

```hornet
def int main():
    int i = 0

    while i < 20:
        i += 1

        if i % 3 == 0:
            if i % 5 == 0:
                print('fizzbuzz')
                continue

            print('fizz')

        if i % 5 == 0:
            print('buzz')

    return 0
```

---

# Compiler Architecture

The compiler is organized as a series of stages:

```text
Hornet source
     │
     ▼
   Lexer
     │
     ▼
    AST
     │
     ▼
Semantic analysis
     │
     ▼
Intermediate representation
     │
     ▼
Register allocation / lowering
     │
     ▼
Assembly AST
     │
     ▼
x86-64 assembly
```

The major compiler components are:

```text
lexer.py
    Converts source text into tokens, including INDENT/DEDENT
    tokens for block structure.

parser.py
    Converts tokens into the Hornet AST.

semantic.py
    Resolves names, types, structs, aliases, methods, and
    validates expressions and statements.

codegen/
    Contains the native code generator and backend.

codegen/ir.py
    Defines the compiler's intermediate representation.

codegen/ir_lowering.py
    Lowers IR into the assembly representation.

codegen/register_allocator.py
    Performs register allocation for IR temporaries.

codegen/arrays_slices.py
    Array and slice code generation.

codegen/structs.py
    Struct code generation.

codegen/calling_convention.py
    x86-64 calling convention handling.

codegen/emitter.py
    Emits textual assembly.
```

The compiler performs semantic analysis before code generation. Invalid names and type mismatches therefore produce compiler errors rather than being discovered accidentally during assembly generation.

---

# Testing

The compiler has an extensive test suite covering the lexer, parser, semantic analyzer, code generator, escape analysis, and register allocator.

Run the tests with:

```bash
cd compiler
pytest
```

The integration tests compile Hornet programs and, where appropriate, assemble and execute the resulting native programs.

Individual feature examples are also kept under:

```text
tests/
```

and use the `.ht` extension.

---

# Current Status

Hornet is an experimental language and compiler. The implementation is functional, but the language and compiler architecture are still evolving.

Some notable features that are **not yet implemented** include:

* `for` loops
* Dictionaries
* Floating-point types
* Pointers
* FFI
* Imports/modules
* Variadic functions
* Spread operators
* Sum types and pattern matching
* Multithreading
* Garbage collection
* Bounds-check elimination
* More advanced compiler optimization

The compiler also currently targets x86-64 Linux and macOS rather than being architecture-independent.

---

# Roadmap

The longer-term direction of the project includes:

* Completing the intermediate representation and moving more code generation through it
* Improving optimization
* Bounds-check elimination
* More sophisticated register allocation
* Pointers and foreign-function interfaces
* Imports and modules
* Additional numeric types
* Sum types and pattern matching
* Concurrency
* Automatic memory management

Hornet's implementation is intentionally incremental: new language features are added with corresponding lexer, parser, semantic-analysis, code-generation, and integration tests rather than relying solely on isolated parser tests.

---

# Repository Layout

```text
compiler/
├── lexer.py
├── parser.py
├── semantic.py
├── compile.py
│
├── codegen/
│   ├── arrays_slices.py
│   ├── assembly_ast.py
│   ├── calling_convention.py
│   ├── codegen.py
│   ├── dispatch.py
│   ├── emitter.py
│   ├── escape_analysis.py
│   ├── ir.py
│   ├── ir_lowering.py
│   ├── register_allocator.py
│   ├── scalars.py
│   ├── statements.py
│   ├── strings.py
│   ├── structs.py
│   └── utils.py
│
├── tests/
│   ├── test_compiler.py
│   ├── test_lexer.py
│   ├── test_parser.py
│   ├── test_semantic.py
│   └── codegen/
│
└── hornet-vim/
```

---

# License

Hornet is currently an experimental project. See the repository for licensing information.

