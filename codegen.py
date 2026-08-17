"""Codegen

Transforms the AST produced by parser.py into x64 assembly (AT&T syntax).

This follows the classic two-stage codegen structure used by many
introductory compilers:

    AST (parser.py)  -->  Assembly AST  -->  Assembly text

Keeping the "Assembly AST" as its own intermediate representation --
rather than emitting strings directly from the source AST -- makes this
much easier to extend later. Once expressions get more complex you'll
need to allocate registers/stack slots as part of building that assembly
AST; once you add more statements you'll add more instruction types.
None of that should require touching the string-formatting code at all,
and none of the string-formatting quirks (spacing, symbol prefixes, etc.)
should leak into the code-generation logic.

Supported so far (matches what parser.py can currently produce):

    Program  -> one or more Functions
    Function -> a name and a body of statements
    Return   -> evaluate an expression into %eax, then `ret`
    Constant -> an immediate value
    Unary    -> NEGATE ('-'), COMPLEMENT ('~'), and NOT ('not'), each
                applied in place to whatever's already in the
                destination register
    Binary   -> ADD ('+'), SUBTRACT ('-'), MULTIPLY ('*'), DIVIDE ('/'),
                MODULO ('%'), the bitwise operators (& | ^ << >>), the
                six comparisons (== != < > <= >=), and the two
                short-circuiting logical operators (and, or)
    Variable -> a read of a local variable's stack slot
    VarDecl  -> `int a` or `int a = <expr>`
    Assign   -> `a = <expr>`
    ExprStmt -> a bare expression statement, evaluated and discarded
    BoolLiteral -> `true`/`false`, immediate 1/0
    If       -> `if`/`elif`/`else`, real conditional jumps (elif is
                nested If, not a separate case -- see parser.py)
    While    -> real loop control flow, condition re-checked before
                every iteration including the first
    Break    -> unconditional jump to the innermost loop's end
    Continue -> unconditional jump back to the innermost loop's
                condition check

A note on typing: as of semantic.py, this file is no longer the only
line of defense against a malformed program -- undeclared/re-declared
variables and every type error are now caught earlier, by a dedicated
semantic-analysis pass that runs between parsing and codegen. This file
still doesn't know anything about int vs bool (both are just 4-byte
values to it; see BoolLiteral's codegen), and still doesn't need to,
since semantic.py already guaranteed the program type-checks before this
ever runs. `generate_asm` deliberately stays a pure "AST -> assembly
text" function with no semantic checking baked in, so it composes
cleanly with pipelines that already validated the AST some other way;
`compile_to_asm` is the file-based convenience wrapper that actually
chains lex -> parse -> analyze -> codegen together, and is what the CLI
below uses. A caller going through `generate_asm` directly is expected
to have called semantic.analyze() itself first.

Unlike Constant, a Unary expression can't be represented as a bare
Operand (there's no such thing as "the immediate value of `-2`" as an
assembly-level concept -- the negation has to actually happen on a
register). So expression codegen works by emitting instructions that
compute the expression's value directly into a destination (see
`gen_expr_into`), rather than by returning an Operand. Nested unary
operators fall out of this for free: `~-2` recurses inward first
(computing -2 into %eax), then applies the outer operator to whatever's
now sitting in that register (`notl %eax`).

BINARY OPERATORS AND THE ONE-REGISTER PROBLEM
------------------------------------------------
Binary operators need *two* operand values alive at once (left and
right), but `gen_expr_into` only has one destination register to work
with -- there's no register allocator yet to hand out a second one. The
fix used here is the classic minimal-effort answer: spill to the real
CPU stack.

To compute `left OP right` into `dst` (%eax):
  1. Compute `left` into %eax (the usual gen_expr_into).
  2. `pushq %rax` -- save that value on the stack.
  3. Compute `right` into %eax -- safe to reuse, since `left` is on the
     stack now.
  4. `movl %eax, %ecx` -- move `right` out of the way into a scratch
     register.
  5. `popq %rax` -- restore `left` back into %eax.
  6. Emit the actual operator instruction combining %ecx into %eax.

Because this always pushes before recursing into the right-hand side and
pops after, nested binary expressions of any depth resolve correctly:
each push/pop pair is balanced within its own gen_binary_into call
regardless of what further pushes and pops happen inside step 3, so the
stack naturally behaves like an expression-evaluation stack. This is not
efficient (a real register allocator would keep far more values in
registers), but it's correct at arbitrary nesting depth, which is what
matters before you have one.

COMPARISONS
-----------
The six comparison operators (== != < > <= >=) reuse that exact scheme
-- both sides always get evaluated -- but combine with cmpl + setCC +
movzbl instead of an arithmetic instruction: cmpl sets flags from
(left - right), setCC turns the relevant flag pattern into a 0/1 byte,
movzbl zero-extends it to fill the register. This is the same trick
logical NOT already used against a fixed comparand of 0 (see
gen_unary_op); comparisons just generalize it to comparing against a
computed `right` and to all six condition codes instead of only 'e'.

AND / OR: SHORT-CIRCUITING
---------------------------
AND and OR are NOT routed through the stack-spill scheme above, because
that scheme always evaluates both operands -- exactly what short-circuit
evaluation must avoid. `a and b` must not evaluate `b` at all if `a` is
already false, and `a or b` must not evaluate `b` at all if `a` is
already true.

Instead, gen_short_circuit emits real control flow: evaluate the left
side, compare it to 0, and conditionally jump straight past the code
that would evaluate the right side. AND and OR are mirror images of each
other and share one implementation:

    AND (jump early on left == 0, early result 0, fallthrough result 1):
        <left>              ; -> %eax
        cmpl $0, %eax
        je   .Land_short_N  ; left was false -- skip right entirely
        <right>             ; -> %eax
        cmpl $0, %eax
        je   .Land_short_N  ; right was false
        movl $1, %eax
        jmp  .Land_end_N
    .Land_short_N:
        movl $0, %eax
    .Land_end_N:

    OR is the same shape with the jump condition, early value, and
    fallthrough value each flipped (jne / 1 / 0).

Because the jump genuinely skips over the right-hand side's instructions
at runtime -- they're present in the binary but never executed -- this
is real short-circuiting, not just a coincidentally-correct value. That
distinction is externally observable: `0 and (1 / 0)` returns 0 without
crashing, while `1 and (1 / 0)` does crash (SIGFPE), because only in the
second case does control flow ever reach the division.

LOCAL VARIABLES
----------------
Every local variable lives in a fixed stack slot at a constant offset
from %rbp -- the classic frame-pointer-relative layout, e.g. the first
variable declared in a function at -4(%rbp), the second at -8(%rbp), and
so on (each int/bool is 4 bytes). _collect_locals recursively pre-scans
a function's whole body -- including into every If's then_body and
else_body, and every While's body -- for VarDecls *before* generating
any instructions, so every slot's offset is already assigned, and the
total frame size is already known, by the time the prologue is emitted:

    pushq %rbp
    movq  %rsp, %rbp
    subq  $N, %rsp      ; only if the function has any locals at all

Every function gets this prologue -- and a matching `leave` right before
every `ret` -- even ones with no locals at all, rather than only doing
it conditionally. That costs a couple of extra instructions on trivial
functions (`return 2` now sets up and tears down an empty frame), but it
keeps codegen uniform now and means there's no special case to unwind
later once this needs to support more than one frame per program (e.g.
actual function calls).

TWO NAMES, TWO NUMBERS: WHY OFFSETS ARE KEYED BY NODE, NOT NAME
-------------------------------------------------------------------
Now that if/else exist, semantic.py allows two variables in sibling
branches to share a name -- `if x: int a = 1` / `else: int a = 2` are
independent scopes, so that's legitimately two different variables that
happen to be spelled the same way. A single `Dict[str, int]` allocator
can't represent that (the second VarDecl would look like a duplicate of
the first). So _collect_locals keys offsets by `id(vardecl_node)`
instead of by name -- every VarDecl anywhere in the function, no matter
how deeply nested or how many others share its name, gets its own
permanent slot. This is simple and always correct, but not
space-optimal: two variables that can never both be alive at once (like
the `a` in an if and the `a` in its else) still each get dedicated stack
space for the whole function call, rather than sharing a slot. Trading a
few bytes of stack for a much simpler allocator felt like the right
call at this stage.

A while loop's body follows the exact same rule, for a related but
distinct reason: a VarDecl inside a loop body is only ever encountered
*once* by _collect_locals (the pre-scan walks the AST, a static tree,
not a simulation of runtime iterations), so it only ever gets one slot
-- and that's correct, because every iteration reuses that same slot by
just overwriting it. There's no "which iteration's `a`" question the
way there's a "which branch's `a`" question for if/else, so nothing
extra is needed here beyond recursing into While bodies the same way
_collect_locals already recurses into If branches.

Fixed, permanent offsets alone aren't enough to make `Variable`/`Assign`
nodes resolve to the *right* slot, though -- codegen still needs to know
which of possibly-several same-named VarDecls a given reference means at
the point it's generated, which is exactly a scoping question. So
`self.scopes: List[Dict[str, int]]` mirrors semantic.py's own scope
stack (pushed/popped around an If's then/else bodies and a While's body,
name -> offset instead of name -> Type), and _local_offset walks it
innermost-to-outermost exactly like semantic.py's _lookup does. This
does mean scope resolution is implemented twice in this codebase, once
per pass -- a real duplication, and a reasonable one to revisit (e.g. by
having semantic.py annotate each Variable/Assign node with its resolved
VarDecl directly) if it ever drifts out of sync with semantic.py's own
rules.

Reading or writing a variable never touches gen_expr_into's core
contract -- every expression still always computes into a register
(%eax). A variable read is just one more expr kind: `Mov(src=Memory(...),
dst=dst)`. A variable *write* (VarDecl-with-initializer, or Assign) is
handled one level up, in `_gen_store`: evaluate the expression into %eax
exactly as always, then a single extra `movl %eax, offset(%rbp)` to
actually store it. This is deliberately not "generate straight into the
memory operand" -- that would mean threading Memory destinations through
gen_binary_into's push/pop scheme, division's %eax requirement, and the
byte-register aliasing comparisons and NOT depend on, none of which are
built to target anything but a register. Routing every store through
%eax first avoids all of that at the cost of one extra mov per store.

As of semantic.py, double declarations, undeclared references, and
every type error are caught well before this file ever runs -- the
`_local_offset` lookup failing here would now only happen if codegen
were invoked directly on an AST that skipped semantic analysis (see this
module's top for the compile_to_asm/generate_asm split), so treat it as
a defensive check rather than the primary error-reporting path.

LOOPS
------
A while loop reuses the same condition-into-%eax-then-compare-to-0
pattern used everywhere else real control flow shows up (short-circuit
AND/OR, if/else): evaluate the condition, jump past the body if it's
false. What's new here is that the condition also needs to be
*re-checked* after the body runs, which just means the body's closing
instruction is an unconditional jump back up to a label placed right
before the condition, rather than falling through to whatever comes
next -- see gen_while's docstring for the exact shape.

break/continue are both just an unconditional Jmp to one of that loop's
two labels -- break to the end (falls out of the loop entirely),
continue to the start (re-checks the condition, which is exactly what
"skip the rest of this iteration" means here, since there's no
per-iteration cleanup step distinct from the condition check). The only
real question either one has to answer is *which* loop's labels, once
loops can nest -- `self.loop_labels: List[Tuple[str, str]]` is a stack
for exactly that reason, pushed with the current loop's (start, end)
pair before its body is generated and popped once it's done, mirroring
semantic.py's loop_depth counter (see its LOOPS section) closely enough
that it's worth noting the difference: semantic.py only needs to know
*whether* a break/continue is inside some loop, so a counter suffices;
codegen needs to know *which* loop's labels to jump to, so it needs the
actual label pair, not just a depth.

STRINGS
--------
A `str` value is a plain pointer -- to a null-terminated buffer, either
static (a literal, emitted into `.data`; see gen_string_literal_into and
Emitter.emit) or malloc'd (a concatenation result; see
gen_string_concat_into). Being a pointer, it's 8 bytes on x86-64, not
the 4 bytes every value before it fit in -- which is why _collect_locals
gives *every* local a uniform 8-byte slot now regardless of type,
trading a few wasted bytes on int/bool locals for not needing a
variable-width allocator at all.

Concatenation and equality both need real work at runtime -- there's no
way to fold `a + b` or `a == b` at compile time once `a`/`b` are
variables -- so they call real C library functions (`strlen`/`malloc`/
`strcpy`/`strcat` for concatenation, `strcmp` for equality) via the
actual SysV calling convention. This is the first place in this
compiler that calls anything external at all, and it surfaced a real
bug worth understanding, not just the fix for it:

Both operations need `left`'s value to survive while `right` gets
computed. The first implementation stashed `left` in a fixed register
(%rbx) for the duration -- which works fine as long as evaluating
`right` never *also* needs %rbx. But `right` can itself be another
string concatenation or comparison (`(a + b) + c`, or a function that
does its own string work), and that nested evaluation reaches this
exact same method again, which unconditionally overwrites %rbx as its
own first step -- silently clobbering the outer call's `left` before
it's ever used. This didn't show up in earlier testing because those
tests only ever nested a string op on one side of another (`(a+b)+c`,
where `c` is a plain variable); it takes something on the scale of `(a
+ b) == (c + d)` -- a nested op on *both* sides of another -- to
actually hit it.

The fix (see gen_string_concat_into/gen_string_compare_into) is to stop
relying on %rbx surviving on its own, and instead push `left` onto the
real stack before evaluating `right` -- the exact push-before-recursing
scheme gen_binary_into already uses for ordinary int/bool operators,
which has no such fixed-identity conflict no matter how deeply it
nests, since every push has its own place on the stack regardless of
what any nested call does to any register. `right` gets captured into
%r12 immediately after its own evaluation completes, which *is* safe as
a fixed register, since nothing recursive happens between that capture
and this method's own further use of it -- only the fixed strlen/
malloc/strcpy/strcat sequence, which can't reach back into
gen_expr_into. %r13/%r14 are similarly write-once, straight from a
direct call result, with nothing recursive after.

This is a *different* concern from why every function's own prologue/
epilogue also saves and restores %rbx/%r12/%r13/%r14 unconditionally
(see gen_function and gen_return, and this section's FUNCTIONS
neighbor) -- that fix protects a value held in one of these registers
*across a call into another function*; this one protects a value held
here *across evaluating a nested expression within the same function*,
call or no call. Both are genuinely necessary; neither one covers what
the other does.

Never frees a *named* concatenation result -- deciding whether a value
in a variable is safe to free needs real escape analysis (has it been
returned? assigned somewhere else? handed to another function that
might retain it?), which semantic.py doesn't do at all, and which is a
meaningfully bigger problem than concatenation itself. What it *does*
free now: the moment a concatenation's operand is itself a fresh,
unnamed concatenation result -- a Binary(ADD, ...) sub-expression that
was never stored into a variable, returned, or passed anywhere, and so
could not possibly be referenced by anything else -- its buffer is
freed immediately after its bytes are copied out (see
_gen_free_if_fresh_concat, called from both gen_string_concat_into and
gen_string_compare_into). That's a real, narrow, decidable-from-the-AST
case: `str r = a + b + c` mallocs three buffers under the hood, but the
intermediate `a + b` result is never reachable by anything once `+ c`
has copied its bytes into the final buffer, so it's freed rather than
silently discarded. A StringLiteral (static `.data`, never
heap-allocated), a Variable (might be read again, or aliased elsewhere
-- codegen has no visibility into that), and a Call's return value
(same problem, plus it might be a static literal or a passed-through
parameter for all codegen can tell) are all deliberately left alone.
This was verified directly during development with an LD_PRELOAD
malloc/free tracer, confirming both that frees only ever target live,
previously-malloc'd pointers (never a literal, never a double-free) and
that the reduction is real: a 6-way concatenation chain drops from 5
un-freed buffers to 1.

FUNCTIONS
----------
Parameters arrive in registers per the SysV ABI (%rdi, %rsi, %rdx,
%rcx, %r8, %r9, in that order -- see _ARG_REGISTERS_64/_32) and get
moved into their own stack slots immediately in the prologue, via
_collect_params/_bind_param -- from that point on a parameter is
indistinguishable from any other local to the rest of codegen. Only the
first 6 arguments are supported, matching how far the ABI's
register-passing goes before falling back to the stack; a 7th parameter
or argument is a CodegenError, not a silent miscompile.

A call's arguments are each fully evaluated (via the ordinary
gen_expr_into, so a nested call or a string concatenation as an
argument works correctly) and immediately pushed onto the stack, one at
a time, *before* any of them get popped into the actual argument
registers -- see gen_call_into. Only once every argument is safely
stacked does popping begin, in reverse, into %rdi/%rsi/etc. This is the
same "compute now, protect on the stack, place into position later"
principle behind the STRINGS section's fix above, for the same reason:
if argument 2 happened to be a string concatenation, and argument 1's
value were sitting in some fixed register instead of on the stack while
argument 2 gets computed, argument 2's own scratch usage could corrupt
argument 1.

Every function's prologue unconditionally pushes %rbx/%r12/%r13/%r14
right after setting up %rbp, and every return pops them back before
`leave` (see gen_function/gen_return) -- regardless of whether that
particular function happens to do any string work itself. This is what
makes it safe for function A, mid-string-operation with a value
currently sitting in one of those registers, to call function B: B's
own prologue saves whatever A left there, B is free to use those
registers however it wants internally, and B's epilogue restores A's
values before control returns -- the standard meaning of "callee-saved"
applied uniformly, rather than reasoned about per call site. Note this
protects a value *across a call*; it does not, on its own, protect a
value across a *nested expression* evaluated in the same function
without a call in between -- that's what the stack-based fix in the
STRINGS section above is for. Both together are what make deeply nested
string operations, with or without function calls mixed in, safe
regardless of how deep the nesting goes.

self.functions is never built here -- semantic.py already guaranteed
every call resolves to a real function with matching argument types and
count before this file ever runs (see this module's top for the
compile_to_asm/generate_asm split). This file doesn't need its own copy
of function signatures at all, including return types -- a Call
expression's type (int/bool/str) needed for e.g. deciding whether
`foo() + bar()` means arithmetic or concatenation is read directly via
_type_of from what semantic.py already resolved, not re-looked-up here.
"""

import argparse
from dataclasses import dataclass, field
from typing import Dict, List

from lexer import lex
from parser import (
    Assign,
    Binary,
    BinaryOp,
    BoolLiteral,
    Break,
    Call,
    Constant,
    Continue,
    ExprStmt,
    Function,
    If,
    Node,
    Param,
    Parser,
    Program,
    Return,
    StringLiteral,
    Unary,
    UnaryOp,
    VarDecl,
    Variable,
    While,
)
from semantic import analyze


# ---------------------------------------------------------------------------
# Assembly AST
# ---------------------------------------------------------------------------

class Operand:
    def emit(self) -> str:
        raise NotImplementedError


@dataclass
class Imm(Operand):
    value: int

    def emit(self) -> str:
        return f"${self.value}"


@dataclass
class Register(Operand):
    name: str  # e.g. 'eax'

    def emit(self) -> str:
        return f"%{self.name}"


@dataclass
class Memory(Operand):
    """A stack-relative memory operand: `offset(%base)`, e.g. `-4(%rbp)`.
    This is how every local variable is stored -- see the module
    docstring's LOCAL VARIABLES section."""
    base: str    # e.g. 'rbp'
    offset: int  # bytes from `base`; locals live at negative offsets

    def emit(self) -> str:
        return f"{self.offset}(%{self.base})"


class Instruction:
    """Base class for assembly instructions.

    Subclasses set `mnemonic` and implement `operands()`; `emit()` is
    generic and handles column alignment (via `mnemonic.ljust(8)`) so
    every instruction lines up the same way regardless of how long its
    mnemonic is -- compare `movl` (4 chars) and `movzbl` (6 chars) in the
    examples below, both of which align their first operand to column 8.
    """

    mnemonic: str = ""

    def operands(self) -> List[str]:
        return []

    def emit(self) -> str:
        ops = self.operands()
        if not ops:
            return self.mnemonic
        return f"{self.mnemonic:<8}{', '.join(ops)}"


@dataclass
class Mov(Instruction):
    src: Operand
    dst: Operand
    mnemonic = "movl"

    def operands(self) -> List[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class Neg(Instruction):
    """Two's-complement arithmetic negation, in place: dst = -dst."""
    operand: Operand
    mnemonic = "negl"

    def operands(self) -> List[str]:
        return [self.operand.emit()]


@dataclass
class Not(Instruction):
    """Bitwise complement, in place: dst = ~dst."""
    operand: Operand
    mnemonic = "notl"

    def operands(self) -> List[str]:
        return [self.operand.emit()]


@dataclass
class Cmp(Instruction):
    """Compares src and dst by computing dst - src and setting flags
    (notably ZF) accordingly -- doesn't modify either operand."""
    src: Operand
    dst: Operand
    mnemonic = "cmpl"

    def operands(self) -> List[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class SetCC(Instruction):
    """Sets an 8-bit operand to 1 if the given condition matches the
    flags from the last Cmp, else 0. `cc` is the x86 condition-code
    suffix -- 'e' (equal), 'ne' (not equal), 'l'/'g' (signed less/greater
    than), 'le'/'ge' (signed less/greater-or-equal) -- and the mnemonic
    is built from it (`sete`, `setne`, `setl`, ...). This is the single
    instruction behind every comparison operator (== != < > <= >=) and
    also behind logical NOT, which is just "was the operand equal to
    0?" (cc='e')."""
    cc: str
    operand: Operand

    @property
    def mnemonic(self) -> str:
        return f"set{self.cc}"

    def operands(self) -> List[str]:
        return [self.operand.emit()]


@dataclass
class MovZX(Instruction):
    """Zero-extends an 8-bit src into a 32-bit dst. Needed after SetE,
    since `sete` only ever writes the low byte (e.g. %al) and leaves the
    rest of the containing 32-bit register (e.g. %eax) untouched -- so
    without this, %eax could still hold garbage in its upper 24 bits."""
    src: Operand
    dst: Operand
    mnemonic = "movzbl"

    def operands(self) -> List[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class Add(Instruction):
    """dst += src."""
    src: Operand
    dst: Operand
    mnemonic = "addl"

    def operands(self) -> List[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class AddQ(Instruction):
    """64-bit dst += src (`addq`). Used only for the length arithmetic
    in gen_string_concat_into (`len(left) + len(right) + 1`) -- string
    lengths come back from `strlen` as a full 64-bit size_t, so this
    needs to be the 64-bit add, not Add's 32-bit `addl`."""
    src: Operand
    dst: Operand
    mnemonic = "addq"

    def operands(self) -> List[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class Sub(Instruction):
    """dst -= src."""
    src: Operand
    dst: Operand
    mnemonic = "subl"

    def operands(self) -> List[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class IMul(Instruction):
    """dst *= src (signed, two-operand form)."""
    src: Operand
    dst: Operand
    mnemonic = "imull"

    def operands(self) -> List[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class Cdq(Instruction):
    """Sign-extends %eax across the %edx:%eax pair. Required immediately
    before IDiv, which always divides that 64-bit pair (not just %eax)
    by its operand -- without this, %edx could hold garbage and corrupt
    the division."""
    mnemonic = "cdq"


@dataclass
class IDiv(Instruction):
    """Divides the 64-bit %edx:%eax pair by `operand` (signed). Quotient
    ends up in %eax, remainder in %edx. `operand` must be a register or
    memory location -- x86 doesn't support an immediate divisor for
    idiv, which is why gen_binary_into always routes the right-hand side
    through the %ecx scratch register rather than leaving it as an Imm.

    This is also what MODULO reuses -- see gen_binary_op's MODULO case
    -- since idiv computes the quotient *and* remainder in one
    instruction; modulo is exactly this same Cdq+IDiv sequence, just
    reading %edx afterward instead of %eax."""
    operand: Operand
    mnemonic = "idivl"

    def operands(self) -> List[str]:
        return [self.operand.emit()]


@dataclass
class And(Instruction):
    """dst &= src (bitwise AND)."""
    src: Operand
    dst: Operand
    mnemonic = "andl"

    def operands(self) -> List[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class Or(Instruction):
    """dst |= src (bitwise OR)."""
    src: Operand
    dst: Operand
    mnemonic = "orl"

    def operands(self) -> List[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class Xor(Instruction):
    """dst ^= src (bitwise XOR)."""
    src: Operand
    dst: Operand
    mnemonic = "xorl"

    def operands(self) -> List[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class ShiftLeft(Instruction):
    """dst <<= %cl. x86 only allows an immediate or specifically %cl as
    a shift instruction's count operand -- never an arbitrary register
    -- so, unlike And/Or/Xor above, this doesn't take a general `src`
    field at all; %cl is hardcoded, since architecturally nothing else
    could ever go there. This lines up for free with how every other
    binary operator already works: gen_binary_into always evaluates the
    right-hand operand into %ecx before calling gen_binary_op, so the
    shift count is already sitting in the one register x86 requires by
    the time this instruction is emitted."""
    dst: Operand
    mnemonic = "shll"

    def operands(self) -> List[str]:
        return ['%cl', self.dst.emit()]


@dataclass
class ShiftRightArithmetic(Instruction):
    """dst >>= %cl, sign-extending (arithmetic) shift -- matches this
    language's `int` being signed, so `-8 >> 1 == -4`, not some large
    positive value from a zero-filling logical shift. See ShiftLeft's
    docstring for why %cl is hardcoded rather than a general `src`."""
    dst: Operand
    mnemonic = "sarl"

    def operands(self) -> List[str]:
        return ['%cl', self.dst.emit()]


@dataclass
class Push(Instruction):
    """Pushes a 64-bit register onto the stack. x86-64 doesn't support a
    32-bit push in long mode, so the caller is responsible for passing
    an already-64-bit register (e.g. Register('rax'), not
    Register('eax')) -- see as_qword_register for converting a 32-bit
    general-purpose register to its 64-bit alias when spilling one."""
    operand: Register
    mnemonic = "pushq"

    def operands(self) -> List[str]:
        return [self.operand.emit()]


@dataclass
class Pop(Instruction):
    """The pop counterpart to Push -- see its docstring."""
    operand: Register
    mnemonic = "popq"

    def operands(self) -> List[str]:
        return [self.operand.emit()]


@dataclass
class LeaQ(Instruction):
    """Loads the *address* of `label` into `dst`, RIP-relative (the
    `(%rip)` addressing mode). This is the standard, PIE-friendly way to
    get a static data address on x86-64 -- an absolute `movq
    $label, %reg` would work on some setups but isn't safe to rely on
    once position-independent executables are in the picture (the
    default for `gcc`-produced binaries on both Linux and macOS), so
    this is what every string literal's address gets loaded with (see
    gen_string_literal_into)."""
    label: str
    dst: Register
    mnemonic = "leaq"

    def operands(self) -> List[str]:
        return [f"{self.label}(%rip)", self.dst.emit()]


@dataclass
class CallInstr(Instruction):
    """Calls a function (either a libc routine like `strlen`, or another
    Hornet-compiled function) by symbol name, e.g. `call strlen` or
    `call add`. Named CallInstr rather than plain Call specifically to
    avoid colliding with parser.Call -- the source-level AST node for a
    function-call *expression* -- which this file also imports; the two
    are easy to conflate by name but are completely different things
    (one is assembly, the other is source syntax), and Python will
    silently let a module-level class definition shadow an import of
    the same name with no error, which is exactly what happened here
    during development before this rename.

    `target` is always the *unprefixed* C symbol name (`malloc`, not
    `_malloc`) -- Emitter is what knows whether the target platform
    needs a leading underscore (see its emit_function), the same way it
    already decides that for this program's own function labels. Emit()
    here (unprefixed) is only ever used if a CallInstr is inspected/
    rendered outside of Emitter; the real rendering path always goes
    through Emitter's own handling instead.

    Requires %rsp to be 16-byte aligned at the point this executes, per
    the SysV ABI -- see codegen.py's LIBRARY CALLS section for how that
    invariant is maintained without explicit runtime alignment checks.
    """
    target: str
    mnemonic = "call"

    def operands(self) -> List[str]:
        return [self.target]


@dataclass
class MovQ(Instruction):
    """64-bit mov (`movq`). Used for frame-pointer setup (`movq %rsp,
    %rbp`) and for anything genuinely 64-bit -- which, as of `str`, now
    includes string pointers (see codegen.py's LOCAL VARIABLES and
    STRINGS sections). int/bool still exclusively use the 32-bit Mov
    (`movl`)."""
    src: Operand
    dst: Operand
    mnemonic = "movq"

    def operands(self) -> List[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class SubQ(Instruction):
    """64-bit subtract (`subq`). Used exactly once per function, in the
    prologue, to reserve stack space for locals: `subq $N, %rsp`."""
    src: Operand
    dst: Operand
    mnemonic = "subq"

    def operands(self) -> List[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class Leave(Instruction):
    """Tears down the current stack frame: equivalent to
    `movq %rbp, %rsp; popq %rbp`. The standard epilogue counterpart to
    the prologue's `pushq %rbp; movq %rsp, %rbp`."""
    mnemonic = "leave"


@dataclass
class Label(Instruction):
    """A jump target. Not really an "instruction" (it assembles to
    nothing -- it just names the address of whatever comes next), but it
    fits the same emit()-based rendering as everything else."""
    name: str

    def emit(self) -> str:
        return f"{self.name}:"


@dataclass
class Jmp(Instruction):
    """Unconditional jump to `target` (a Label's name)."""
    target: str
    mnemonic = "jmp"

    def operands(self) -> List[str]:
        return [self.target]


@dataclass
class Je(Instruction):
    """Jump to `target` if the last Cmp found its operands equal (ZF set)."""
    target: str
    mnemonic = "je"

    def operands(self) -> List[str]:
        return [self.target]


@dataclass
class Jne(Instruction):
    """Jump to `target` if the last Cmp found its operands unequal (ZF clear)."""
    target: str
    mnemonic = "jne"

    def operands(self) -> List[str]:
        return [self.target]


@dataclass
class Ret(Instruction):
    mnemonic = "ret"


@dataclass
class AsmFunction:
    name: str
    instructions: List[Instruction] = field(default_factory=list)


@dataclass
class AsmProgram:
    functions: List[AsmFunction] = field(default_factory=list)
    # (label, content) pairs for every string literal anywhere in the
    # program, collected across all functions during generation (see
    # CodeGenerator.gen_string_literal_into). These aren't tied to any
    # one function's frame -- they're static, immutable data -- so they
    # live at the AsmProgram level and get emitted once, in a shared
    # `.data` block, by Emitter (see its emit()).
    string_literals: List[tuple] = field(default_factory=list)


# 32-bit register name -> its 8-bit low-byte alias (e.g. %eax -> %al).
# `sete` (and friends) can only target an 8-bit operand, so codegen needs
# to be able to get from "the register I'm working in" to "its byte
# alias". Only registers actually in use are listed; extend this table
# alongside whatever new registers the code generator starts using.
_BYTE_REGISTER_ALIASES = {
    'eax': 'al',
}


def as_byte_register(reg: Operand) -> Register:
    if not isinstance(reg, Register) or reg.name not in _BYTE_REGISTER_ALIASES:
        raise CodegenError(f"No 8-bit alias known for register operand: {reg!r}")
    return Register(_BYTE_REGISTER_ALIASES[reg.name])


# 32-bit register name -> its 64-bit alias (e.g. %eax -> %rax). Needed
# because Push/Pop can't operate on a 32-bit operand size in long mode.
_QWORD_REGISTER_ALIASES = {
    'eax': 'rax',
    'ecx': 'rcx',
}


def as_qword_register(reg: Operand) -> Register:
    if not isinstance(reg, Register) or reg.name not in _QWORD_REGISTER_ALIASES:
        raise CodegenError(f"No 64-bit alias known for register operand: {reg!r}")
    return Register(_QWORD_REGISTER_ALIASES[reg.name])


# BinaryOp -> the x86 condition-code suffix that implements it, given
# that Cmp(src=right, dst=left) computes (left - right) and sets flags
# accordingly. All six comparisons share one codegen path (see
# gen_binary_op) that just plugs the relevant cc into SetCC.
_COMPARISON_CONDITION_CODES = {
    BinaryOp.EQUAL: 'e',
    BinaryOp.NOT_EQUAL: 'ne',
    BinaryOp.LESS_THAN: 'l',
    BinaryOp.GREATER_THAN: 'g',
    BinaryOp.LESS_THAN_OR_EQUAL: 'le',
    BinaryOp.GREATER_THAN_OR_EQUAL: 'ge',
}


# SysV ABI integer/pointer argument registers, in order, 64-bit and
# 32-bit forms. Only the first 6 arguments of a call are supported --
# beyond that the ABI moves to stack-passed arguments, which this
# compiler doesn't implement (see gen_call_into and gen_function's
# param-count checks). The 32-bit names don't follow one consistent
# pattern: rdi/rsi/rdx/rcx are "legacy" registers with their own
# historical e-prefixed names, while r8/r9 are x86-64-only and use a
# d-suffix instead -- hence two explicit parallel lists rather than a
# derived/computed mapping.
_ARG_REGISTERS_64 = ['rdi', 'rsi', 'rdx', 'rcx', 'r8', 'r9']
_ARG_REGISTERS_32 = ['edi', 'esi', 'edx', 'ecx', 'r8d', 'r9d']

# Registers gen_string_concat_into/gen_string_compare_into use as
# scratch (see STRINGS). Now that functions can call each other,
# *every* function's prologue/epilogue saves and restores these
# unconditionally -- see gen_function and gen_return -- regardless of
# whether that particular function happens to use them, because the
# callee-saved contract has to hold for any call, not just ones this
# compiler happens to know use string operations. See the module
# docstring's FUNCTIONS section for why this became necessary.
_CALLEE_SAVED_SCRATCH_REGISTERS = ['rbx', 'r12', 'r13', 'r14']


# ---------------------------------------------------------------------------
# AST -> Assembly AST
# ---------------------------------------------------------------------------

class CodegenError(Exception):
    """Raised when the code generator encounters an AST node it doesn't
    know how to translate yet."""


class CodeGenerator:
    """Walks the source AST (Program/Function/Return/Constant/...) and
    produces an equivalent AsmProgram."""

    def __init__(self):
        self._label_count = 0
        self._var_offsets: Dict[int, int] = {}  # id(VarDecl node) -> its permanent Memory offset
        self._next_offset = 0
        self.scopes: List[Dict[str, tuple]] = []  # name -> (offset, type_str), generation-time; see LOCAL VARIABLES
        self.loop_labels: List[tuple] = []  # stack of (start_label, end_label), innermost last; see LOOPS
        self.string_literals: List[tuple] = []  # (label, content) pairs; see STRINGS
        # Lazily created, then cached and reused for the rest of this
        # compilation -- see gen_print_call_into and the module
        # docstring's BUILTINS section for why these specifically (and
        # only these) get a small dedicated cache rather than following
        # string_literals' usual "every occurrence gets its own label,
        # no dedup" policy.
        self._int_format_label = None
        self._true_str_label = None
        self._false_str_label = None

    def new_label(self, prefix: str) -> str:
        """Returns a fresh, uniquely-numbered local label like
        `.Land_short_0`. Needed because AND/OR/if codegen all emit real
        jump targets, and a program can contain any number of them --
        each one needs a name the assembler won't collide with any
        other."""
        label = f".L{prefix}_{self._label_count}"
        self._label_count += 1
        return label

    def generate(self, program: Program) -> AsmProgram:
        functions = [self.gen_function(fn) for fn in program.functions]
        return AsmProgram(functions=functions, string_literals=self.string_literals)

    def gen_function(self, fn: Function) -> AsmFunction:
        # Fresh allocator state per function -- variables don't persist
        # across functions, and offsets are relative to *this*
        # function's own %rbp.
        self._var_offsets = {}
        self._next_offset = 0
        self._collect_params(fn.params)
        self._collect_locals(fn.body)
        self.scopes = [{}]

        if len(fn.params) > 6:
            raise CodegenError(
                f"Function '{fn.name}' has {len(fn.params)} parameters; "
                f"this compiler only supports up to 6 (passed via "
                f"registers per the SysV ABI -- stack-passed parameters "
                f"aren't implemented)"
            )

        instructions: List[Instruction] = [
            Push(Register('rbp')),
            MovQ(src=Register('rsp'), dst=Register('rbp')),
        ]
        # Save every callee-saved scratch register unconditionally, not
        # just in functions that happen to do string work themselves --
        # see _CALLEE_SAVED_SCRATCH_REGISTERS and the module docstring's
        # FUNCTIONS section for why this is now required rather than
        # optional once functions can call each other.
        for reg in _CALLEE_SAVED_SCRATCH_REGISTERS:
            instructions.append(Push(Register(reg)))

        frame_size = self._frame_size()
        if frame_size:
            instructions.append(SubQ(src=Imm(frame_size), dst=Register('rsp')))

        # Parameters arrive in registers per the SysV ABI; move each
        # into its own stack slot immediately, exactly like storing any
        # other local -- from here on, a parameter is indistinguishable
        # from a VarDecl-with-initializer as far as the rest of codegen
        # is concerned.
        for i, p in enumerate(fn.params):
            offset = self._bind_param(p)
            if p.type == 'str':
                instructions.append(MovQ(src=Register(_ARG_REGISTERS_64[i]), dst=Memory('rbp', offset)))
            else:
                instructions.append(Mov(src=Register(_ARG_REGISTERS_32[i]), dst=Memory('rbp', offset)))

        for stmt in fn.body:
            instructions.extend(self.gen_statement(stmt))

        return AsmFunction(name=fn.name, instructions=instructions)

    def _collect_params(self, params: List[Param]) -> None:
        """Gives each parameter its own permanent stack slot, exactly
        like _collect_locals does for VarDecls (same node-identity
        keying, same uniform 8-byte width) -- kept as a separate method
        since Param and VarDecl are different AST node types, not
        because parameters need fundamentally different treatment."""
        for p in params:
            self._next_offset -= 8
            self._var_offsets[id(p)] = self._next_offset

    def _bind_param(self, p: Param) -> int:
        """The Param counterpart to _bind_local -- registers `p`'s name
        and declared type in the current scope, pointing at the
        permanent offset _collect_params already assigned it."""
        offset = self._var_offsets[id(p)]
        self.scopes[-1][p.name] = (offset, p.type)
        return offset

    def _collect_locals(self, statements: List[Node]) -> None:
        """Recursively walks `statements`, including into every If's
        then_body/else_body and every While's body, and gives each
        VarDecl found its own permanent stack slot, keyed by the AST
        node's identity rather than its name -- see the module
        docstring's LOCAL VARIABLES section for why that distinction
        now matters.

        Every slot is 8 bytes, even for a 4-byte int/bool, now that str
        (an 8-byte pointer) exists -- see the module docstring's
        STRINGS section for why a uniform width was chosen over
        variable-width packing."""
        for stmt in statements:
            if isinstance(stmt, VarDecl):
                self._next_offset -= 8
                self._var_offsets[id(stmt)] = self._next_offset
            elif isinstance(stmt, If):
                self._collect_locals(stmt.then_body)
                if stmt.else_body is not None:
                    self._collect_locals(stmt.else_body)
            elif isinstance(stmt, While):
                self._collect_locals(stmt.body)

    def _frame_size(self) -> int:
        # Total bytes used by locals and parameters, rounded up to a
        # 16-byte boundary. Genuinely required, not just good practice:
        # gen_string_* and gen_call_into both emit real `call`
        # instructions (to malloc/strlen/strcpy/strcat/strcmp, or to
        # another Hornet function), and the SysV ABI requires %rsp to be
        # 16-byte-aligned at the point of every one of those. (The
        # callee-saved register pushes in gen_function's prologue don't
        # themselves need accounting for here -- there are always
        # exactly 4 of them, an already-even number of 8-byte pushes, so
        # they never change whether %rsp ends up aligned or not.)
        raw = -self._next_offset
        return ((raw + 15) // 16) * 16 if raw > 0 else 0

    def _push_scope(self) -> None:
        self.scopes.append({})

    def _pop_scope(self) -> None:
        self.scopes.pop()

    def _bind_local(self, stmt: VarDecl) -> int:
        """Registers `stmt`'s name -- and its declared type, needed by
        _local_type -- in the current (innermost) generation-time
        scope, pointing at the permanent offset _collect_locals already
        assigned this exact VarDecl node, and returns that offset."""
        offset = self._var_offsets[id(stmt)]
        self.scopes[-1][stmt.name] = (offset, stmt.var_type)
        return offset

    def _local_offset(self, name: str) -> int:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name][0]
        raise CodegenError(f"Reference to undeclared variable '{name}'")

    def _local_type(self, name: str) -> str:
        """Used specifically where a Variable's *offset* is also being
        looked up right alongside it (see gen_expr_into's Variable case)
        -- both come from the same (offset, type) tuple in the same
        scope-stack entry, which codegen has to maintain regardless of
        _type_of's existence, since resolved_type has no way to encode
        *which* stack slot a name refers to. This is deliberately not
        replaced by _type_of below, even though it would give the same
        answer for a Variable node -- see _type_of's own docstring for
        why the two coexist rather than one replacing the other."""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name][1]
        raise CodegenError(f"Reference to undeclared variable '{name}'")

    def _type_of(self, expr: Node) -> str:
        """Reads the type semantic.py already resolved and annotated
        onto this exact node (expr.resolved_type -- see semantic.py's
        check_expr) rather than re-deriving it independently.

        This replaces what used to be a separate _infer_type method
        here that re-implemented, in miniature, the same "which type
        does this operator/call produce" logic semantic.py's
        check_binary/check_call already fully implement -- a second,
        parallel copy of that logic that could (and twice actually did)
        silently drift out of sync with the real one: adding `print`
        needed a Call case added here too, and adding the six new
        int-only operators (%, &, |, ^, <<, >>) needed them added to
        this method's own int-producing branch, separately from adding
        them to semantic.py's _INT_ONLY_BINARY_OPS. Neither addition
        was structurally required by anything -- both were just easy to
        forget, and both were only caught by manual testing rather than
        anything that would have failed loudly on its own. Reading the
        annotation instead removes the second copy entirely: there's no
        per-operator or per-node-type branch here left to forget
        updating, since whatever semantic.py already decided is just
        read directly, whatever it happens to be.

        Still raises a clear, defensive CodegenError (matching
        _local_offset's own posture) rather than a bare AttributeError
        if resolved_type is somehow None -- the one legitimate way that
        happens is codegen being invoked on an AST that skipped
        semantic analysis entirely (see compile_to_asm, which always
        runs analyze() first for exactly this reason).
        """
        if expr.resolved_type is None:
            raise CodegenError(
                f"{expr!r} has no resolved type -- semantic.analyze() "
                f"must run before codegen (see compile_to_asm)"
            )
        return expr.resolved_type

    def gen_statement(self, stmt: Node) -> List[Instruction]:
        if isinstance(stmt, VarDecl):
            return self.gen_var_decl(stmt)
        if isinstance(stmt, Assign):
            return self.gen_assign(stmt)
        if isinstance(stmt, Return):
            return self.gen_return(stmt)
        if isinstance(stmt, If):
            return self.gen_if(stmt)
        if isinstance(stmt, While):
            return self.gen_while(stmt)
        if isinstance(stmt, Break):
            return self.gen_break(stmt)
        if isinstance(stmt, Continue):
            return self.gen_continue(stmt)
        if isinstance(stmt, ExprStmt):
            return self.gen_expr_stmt(stmt)
        raise CodegenError(f"No codegen rule for statement: {stmt!r}")

    def gen_var_decl(self, stmt: VarDecl) -> List[Instruction]:
        # _collect_locals already reserved this VarDecl's slot (that's
        # what sizes the frame); _bind_local just needs to make its name
        # resolvable in the current scope, and return where to store the
        # initializer, if there is one. `int a` with no initializer
        # leaves the slot's contents genuinely uninitialized, matching
        # C: reading it before assigning is undefined behavior, not
        # implicitly zero.
        offset = self._bind_local(stmt)
        if stmt.init is None:
            return []
        return self._gen_store(offset, stmt.init)

    def gen_assign(self, stmt: Assign) -> List[Instruction]:
        offset = self._local_offset(stmt.name)
        return self._gen_store(offset, stmt.value)

    def _gen_store(self, offset: int, value_expr: Node) -> List[Instruction]:
        """Shared by VarDecl-with-initializer and Assign: both are just
        "compute this expression, then write the result into that
        variable's slot" -- evaluate into %eax using the ordinary
        expression codegen (still always targets a register), then a
        single extra store to actually place it in memory. Which store
        instruction depends on the value's type: a str is an 8-byte
        pointer sitting in %rax and needs `movq`, while int/bool are
        still the original 4-byte `movl %eax, ...` -- everything about
        gen_expr_into/gen_binary_into/gen_unary_op's own internals stays
        exactly as it always has, oblivious to str entirely; only this
        one call site needs to ask "which width am I storing"."""
        instructions = self.gen_expr_into(value_expr, Register('eax'))
        if self._type_of(value_expr) == 'str':
            instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', offset)))
        else:
            instructions.append(Mov(src=Register('eax'), dst=Memory('rbp', offset)))
        return instructions

    def gen_return(self, stmt: Return) -> List[Instruction]:
        dst = Register('eax')
        instructions = self.gen_expr_into(stmt.value, dst)
        # Restore the callee-saved scratch registers *before* Leave --
        # Leave resets %rsp straight to %rbp, which was captured before
        # these were pushed in the prologue, so anything pushed after
        # that point has to be popped explicitly first or it's just
        # silently discarded (never actually restored into the
        # registers) rather than popped. Popping happens in reverse of
        # the prologue's push order, the usual stack discipline. None of
        # this touches %eax/%rax, so the return value computed above is
        # unaffected regardless of what these registers held during the
        # body (e.g. if the return expression itself did string work
        # that reused them as scratch in between).
        for reg in reversed(_CALLEE_SAVED_SCRATCH_REGISTERS):
            instructions.append(Pop(Register(reg)))
        instructions.append(Leave())
        instructions.append(Ret())
        return instructions

    def gen_if(self, stmt: If) -> List[Instruction]:
        """Computes the condition into %eax and compares it to 0, exactly
        like the short-circuit AND/OR codegen already does -- then jumps
        past the `then` body when it's false:

            <condition>          ; -> %eax
            cmpl $0, %eax
            je   .Lif_else_N     ; false -> skip straight to else (or end)
            <then_body>
            jmp  .Lif_end_N      ; true -> skip over else after then runs
        .Lif_else_N:
            <else_body>          ; only emitted if else_body is present
        .Lif_end_N:

        then_body and else_body each get their own pushed/popped scope
        (see _push_scope), matching semantic.py's independent-branch
        scoping -- and since an elif is just a nested If sitting inside
        else_body (see parser.py's If docstring), gen_statement's
        ordinary recursion handles a whole elif/else chain of any
        length with no extra logic here at all.
        """
        dst = Register('eax')
        else_label = self.new_label("if_else")
        end_label = self.new_label("if_end")

        instructions = self.gen_expr_into(stmt.condition, dst)
        instructions.append(Cmp(src=Imm(0), dst=dst))
        instructions.append(Je(else_label))

        self._push_scope()
        for s in stmt.then_body:
            instructions.extend(self.gen_statement(s))
        self._pop_scope()
        instructions.append(Jmp(end_label))

        instructions.append(Label(else_label))
        if stmt.else_body is not None:
            self._push_scope()
            for s in stmt.else_body:
                instructions.extend(self.gen_statement(s))
            self._pop_scope()
        instructions.append(Label(end_label))

        return instructions

    def gen_while(self, stmt: While) -> List[Instruction]:
        """Computes the condition, re-checked before every iteration
        (including the first), with the body sitting between two labels
        that break/continue jump to:

            .Lwhile_start_N:
                <condition>          ; -> %eax
                cmpl $0, %eax
                je   .Lwhile_end_N   ; false -> exit the loop entirely
                <body>
                jmp  .Lwhile_start_N ; loop back to re-check the condition
            .Lwhile_end_N:

        Both labels get pushed onto self.loop_labels for the duration
        of generating the body, so any Break/Continue statement inside
        it -- including ones nested inside an If -- can find its way
        back here via gen_break/gen_continue without this method needing
        to know anything about where inside the body they are. Popped
        again once the body's done, so a Break/Continue *after* this
        while (or in a sibling loop) can't accidentally resolve to this
        loop's labels -- see the module docstring's LOOPS section for
        why that matters once loops nest.

        The body gets its own pushed/popped scope, same as an If's
        then/else bodies, even though it's the same physical stack slots
        being reused on every iteration (see _collect_locals) -- this is
        purely about name resolution during code generation, not
        anything that happens at runtime.
        """
        dst = Register('eax')
        start_label = self.new_label("while_start")
        end_label = self.new_label("while_end")

        instructions = [Label(start_label)]
        instructions.extend(self.gen_expr_into(stmt.condition, dst))
        instructions.append(Cmp(src=Imm(0), dst=dst))
        instructions.append(Je(end_label))

        self.loop_labels.append((start_label, end_label))
        self._push_scope()
        for s in stmt.body:
            instructions.extend(self.gen_statement(s))
        self._pop_scope()
        self.loop_labels.pop()

        instructions.append(Jmp(start_label))
        instructions.append(Label(end_label))
        return instructions

    def gen_break(self, stmt: Break) -> List[Instruction]:
        # semantic.py already guarantees this only appears inside a
        # loop; the IndexError-avoiding check here is the same defensive
        # posture as _local_offset's -- see the module docstring on
        # generate_asm/compile_to_asm for why codegen still checks for
        # itself rather than trusting semantic analysis unconditionally.
        if not self.loop_labels:
            raise CodegenError("'break' outside of a loop")
        _, end_label = self.loop_labels[-1]
        return [Jmp(end_label)]

    def gen_continue(self, stmt: Continue) -> List[Instruction]:
        if not self.loop_labels:
            raise CodegenError("'continue' outside of a loop")
        start_label, _ = self.loop_labels[-1]
        return [Jmp(start_label)]

    def gen_expr_stmt(self, stmt: ExprStmt) -> List[Instruction]:
        # Evaluated the same way as any other expression, into %eax --
        # just with nothing done with the result afterward. Still real
        # instructions that really run; see the module docstring for how
        # that's verified (a standalone `1 / 0` genuinely crashes).
        return self.gen_expr_into(stmt.expr, Register('eax'))

    def gen_expr_into(self, expr: Node, dst: Operand) -> List[Instruction]:
        """Emits the instructions needed to compute `expr` and leave its
        result sitting in `dst`.

        This (rather than "return an Operand") is the right shape for
        expression codegen once operators are involved: a Constant can
        be represented as a bare Imm operand, but "the result of negating
        something" can't -- it has to actually be computed by an
        instruction acting on a register. So every expression, constants
        included, is generated the same way: as instructions that leave
        their answer in `dst`.
        """
        if isinstance(expr, Constant):
            return [Mov(src=Imm(expr.value), dst=dst)]
        if isinstance(expr, BoolLiteral):
            # bool has the same 4-byte runtime representation as int
            # (0/1 in a register or stack slot) -- semantic.py is what
            # keeps the two from being mixed up; codegen just needs an
            # immediate.
            return [Mov(src=Imm(1 if expr.value else 0), dst=dst)]
        if isinstance(expr, StringLiteral):
            return self.gen_string_literal_into(expr, dst)
        if isinstance(expr, Variable):
            offset = self._local_offset(expr.name)
            if self._local_type(expr.name) == 'str':
                return [MovQ(src=Memory('rbp', offset), dst=as_qword_register(dst))]
            return [Mov(src=Memory('rbp', offset), dst=dst)]
        if isinstance(expr, Call):
            if expr.name == 'print':
                return self.gen_print_call_into(expr, dst)
            return self.gen_call_into(expr, dst)
        if isinstance(expr, Unary):
            # Compute the operand into dst first, then apply this node's
            # operator to whatever's now there. This is what makes chained
            # operators (`~-2`) work: the inner Unary's instructions run
            # first, then the outer operator's instructions run on top.
            instructions = self.gen_expr_into(expr.operand, dst)
            instructions.extend(self.gen_unary_op(expr.op, dst))
            return instructions
        if isinstance(expr, Binary):
            # ADD and the two equality operators are overloaded for str
            # (concatenation and strcmp-backed comparison respectively;
            # see the module docstring's STRINGS section) -- everything
            # else, and ADD/==/!= between two ints or bools, goes
            # through the original gen_binary_into completely unchanged.
            if expr.op == BinaryOp.ADD and self._type_of(expr.left) == 'str':
                return self.gen_string_concat_into(expr, dst)
            if expr.op in (BinaryOp.EQUAL, BinaryOp.NOT_EQUAL) and self._type_of(expr.left) == 'str':
                return self.gen_string_compare_into(expr, dst)
            return self.gen_binary_into(expr, dst)
        raise CodegenError(f"No codegen rule for expression: {expr!r}")

    def gen_binary_into(self, expr: Binary, dst: Operand) -> List[Instruction]:
        """Computes `expr.left OP expr.right` into `dst`.

        AND/OR are handled entirely separately (see gen_short_circuit)
        since they must not unconditionally evaluate both sides. Every
        other binary operator -- arithmetic and comparisons alike -- goes
        through the stack-spill scheme described in the module
        docstring, which always evaluates both sides. Requires `dst` to
        be a register (there's a real 32-bit register and its 64-bit
        alias pushed/popped along the way, which an Imm can't do).
        """
        if expr.op == BinaryOp.AND:
            return self.gen_short_circuit(
                expr, dst,
                short_circuit_jump=Je,   # jump early when the left side is already false
                short_circuit_value=0,   # ...and the overall result is false
                fallthrough_value=1,     # both sides were truthy -> true
                label_prefix="and",
            )
        if expr.op == BinaryOp.OR:
            return self.gen_short_circuit(
                expr, dst,
                short_circuit_jump=Jne,  # jump early when the left side is already true
                short_circuit_value=1,   # ...and the overall result is true
                fallthrough_value=0,     # both sides were falsy -> false
                label_prefix="or",
            )

        if not isinstance(dst, Register):
            raise CodegenError(f"Binary codegen requires a register destination, got: {dst!r}")

        scratch = Register('ecx')  # holds the right-hand value while combining
        instructions = self.gen_expr_into(expr.left, dst)   # dst = left
        instructions.append(Push(as_qword_register(dst)))   # save left on the stack
        instructions.extend(self.gen_expr_into(expr.right, dst))  # dst = right (left is safe)
        instructions.append(Mov(src=dst, dst=scratch))       # scratch = right
        instructions.append(Pop(as_qword_register(dst)))     # dst = left (restored)
        instructions.extend(self.gen_binary_op(expr.op, src=scratch, dst=dst))
        return instructions

    def gen_short_circuit(
        self, expr: Binary, dst: Operand, *,
        short_circuit_jump: type,
        short_circuit_value: int,
        fallthrough_value: int,
        label_prefix: str,
    ) -> List[Instruction]:
        """Shared codegen for AND and OR -- they're mirror images of each
        other: each evaluates its left side, tests it against 0, and
        jumps straight past the right side entirely (never emitting the
        instructions that would compute it as *executed* code) if that
        test already decides the answer. Only if it doesn't -- left was
        truthy for AND, falsy for OR -- does the right side actually get
        evaluated, and *that* result decides the answer instead.

          AND: jump early (to `short_circuit_value=0`) when left == 0.
          OR:  jump early (to `short_circuit_value=1`) when left != 0.

        This is what makes `0 and (1 / 0)` return 0 instead of crashing:
        the division is real code sitting in the binary, but control
        flow jumps clean over it.
        """
        if not isinstance(dst, Register):
            raise CodegenError(f"Binary codegen requires a register destination, got: {dst!r}")

        short_label = self.new_label(f"{label_prefix}_short")
        end_label = self.new_label(f"{label_prefix}_end")

        instructions = self.gen_expr_into(expr.left, dst)
        instructions.append(Cmp(src=Imm(0), dst=dst))
        instructions.append(short_circuit_jump(short_label))

        instructions.extend(self.gen_expr_into(expr.right, dst))
        instructions.append(Cmp(src=Imm(0), dst=dst))
        instructions.append(short_circuit_jump(short_label))

        instructions.append(Mov(src=Imm(fallthrough_value), dst=dst))
        instructions.append(Jmp(end_label))
        instructions.append(Label(short_label))
        instructions.append(Mov(src=Imm(short_circuit_value), dst=dst))
        instructions.append(Label(end_label))
        return instructions

    def gen_call_into(self, expr: Call, dst: Operand) -> List[Instruction]:
        """`name(arg1, arg2, ...)`: evaluates every argument -- in
        order, each via the ordinary gen_expr_into, so an argument that
        is itself a nested call, a string concatenation, or any other
        arbitrarily complex expression works correctly -- immediately
        pushing each one's result onto the stack before moving on to the
        next. Only *after* every argument has been safely computed and
        stacked does this start popping them back off, in reverse, into
        the actual SysV argument registers (see _ARG_REGISTERS_64).

        This "compute and stack everything, then pop into place" order
        is what avoids the same register-clobbering hazard that
        motivated saving %rbx/%r12/%r13/%r14 across calls in the first
        place (see the module docstring's FUNCTIONS section): if
        argument 2 happens to be a string concatenation and argument 1's
        value were sitting in a scratch register instead of safely on
        the stack while argument 2 gets computed, argument 2's own use
        of that same scratch register would corrupt argument 1.

        The result already ends up exactly where gen_expr_into's
        contract expects it (%rax/%eax, matching `dst`, which is always
        Register('eax') throughout this file), so there's nothing left
        to move once the call returns.
        """
        if len(expr.args) > 6:
            raise CodegenError(
                f"Call to '{expr.name}' has {len(expr.args)} arguments; "
                f"this compiler only supports up to 6 (passed via "
                f"registers per the SysV ABI -- stack-passed arguments "
                f"aren't implemented)"
            )
        if dst != Register('eax'):
            raise CodegenError(f"Call codegen requires dst == %eax, got: {dst!r}")

        instructions: List[Instruction] = []
        for arg in expr.args:
            instructions.extend(self.gen_expr_into(arg, Register('eax')))
            instructions.append(Push(Register('rax')))
        for i in reversed(range(len(expr.args))):
            instructions.append(Pop(Register(_ARG_REGISTERS_64[i])))
        instructions.append(CallInstr(expr.name))
        return instructions

    def gen_print_call_into(self, expr: Call, dst: Operand) -> List[Instruction]:
        """`print(x)`: dispatches on x's *compile-time* type -- known
        exactly, since Hornet is statically typed -- to one of three
        completely different instruction sequences, each calling a
        different libc function. See the module docstring's BUILTINS
        section for why each type gets its own call rather than one
        shared, format-driven path.

          str:  puts(x)                    -- puts adds its own newline
          int:  printf("%d\\n", x)         -- needs real formatting
          bool: puts(x ? "true" : "false") -- a runtime branch (the
                exact same cmp/je/jmp/label shape gen_if already uses)
                picks which string literal's address to pass, then
                falls through to the same puts call as the str case

        Every path ends with `movl $0, %eax`, overriding whatever
        puts/printf actually returned -- print's "return value" is a
        clean, predictable 0 (see semantic.py's check_print_call),
        never leaking the underlying libc call's own return convention
        into the language.

        No register-preservation concerns beyond the ones already
        established: puts/printf are libc functions, and libc is
        already a fully ABI-compliant citizen (that's the entire point
        of the ABI), so calling them from inside a Hornet function's
        body is exactly as safe as calling another Hornet function --
        both rely on the callee-saved registers being honored by
        whatever gets called, which is now true either way (see
        gen_function's prologue).
        """
        if dst != Register('eax'):
            raise CodegenError(f"Call codegen requires dst == %eax, got: {dst!r}")

        arg = expr.args[0]
        arg_type = self._type_of(arg)

        if arg_type == 'str':
            instructions = self.gen_expr_into(arg, dst)
            instructions.append(MovQ(src=as_qword_register(dst), dst=Register('rdi')))
            instructions.append(CallInstr('puts'))
            instructions.append(Mov(src=Imm(0), dst=dst))
            return instructions

        if arg_type == 'int':
            fmt_label = self._get_int_format_label()
            instructions = self.gen_expr_into(arg, dst)
            instructions.append(Mov(src=dst, dst=Register('esi')))       # esi = value (2nd printf arg)
            instructions.append(LeaQ(label=fmt_label, dst=Register('rdi')))  # rdi = &"%d\n" (1st arg)
            # AL must be 0 before calling a variadic function per the
            # SysV ABI (it tells the callee how many vector/xmm
            # registers were used for float varargs -- always 0 here,
            # since nothing in this language is ever passed as a float).
            # A plain `movl $0, %eax` both clears AL and is a completely
            # safe clobber of %eax at this point, since the value we
            # care about was already copied into %esi just above.
            instructions.append(Mov(src=Imm(0), dst=dst))
            instructions.append(CallInstr('printf'))
            instructions.append(Mov(src=Imm(0), dst=dst))
            return instructions

        if arg_type == 'bool':
            true_label = self._get_true_str_label()
            false_label = self._get_false_str_label()
            false_branch_label = self.new_label("print_bool_false")
            end_label = self.new_label("print_bool_end")

            instructions = self.gen_expr_into(arg, dst)
            instructions.append(Cmp(src=Imm(0), dst=dst))
            instructions.append(Je(false_branch_label))
            instructions.append(LeaQ(label=true_label, dst=Register('rdi')))
            instructions.append(Jmp(end_label))
            instructions.append(Label(false_branch_label))
            instructions.append(LeaQ(label=false_label, dst=Register('rdi')))
            instructions.append(Label(end_label))
            instructions.append(CallInstr('puts'))
            instructions.append(Mov(src=Imm(0), dst=dst))
            return instructions

        raise CodegenError(f"'print' has no codegen rule for type: {arg_type}")

    def _get_int_format_label(self) -> str:
        if self._int_format_label is None:
            self._int_format_label = self.new_label("fmt_int")
            self.string_literals.append((self._int_format_label, "%d\n"))
        return self._int_format_label

    def _get_true_str_label(self) -> str:
        if self._true_str_label is None:
            self._true_str_label = self.new_label("true_str")
            self.string_literals.append((self._true_str_label, "true"))
        return self._true_str_label

    def _get_false_str_label(self) -> str:
        if self._false_str_label is None:
            self._false_str_label = self.new_label("false_str")
            self.string_literals.append((self._false_str_label, "false"))
        return self._false_str_label

    def gen_string_literal_into(self, expr: StringLiteral, dst: Operand) -> List[Instruction]:
        """Registers this literal's content for later emission as static
        `.data` (see AsmProgram.string_literals / Emitter.emit), and
        loads its address into `dst`. Every occurrence gets its own
        fresh label, even if two literals happen to have identical
        content -- no deduplication, which is a bit wasteful but keeps
        this a one-line append rather than needing a content->label
        cache."""
        label = self.new_label("str")
        self.string_literals.append((label, expr.value))
        return [LeaQ(label=label, dst=as_qword_register(dst))]

    def gen_string_concat_into(self, expr: Binary, dst: Operand) -> List[Instruction]:
        """`left + right`, both str: builds a brand-new, malloc'd,
        null-terminated buffer holding left's bytes immediately
        followed by right's -- `strlen(left) + strlen(right) + 1`
        bytes, then `strcpy` then `strcat`.

        `left` is protected across evaluating `right` by pushing it onto
        the real CPU stack -- the exact same push-before-recursing
        scheme gen_binary_into already uses for ordinary int/bool
        operators -- rather than stashing it in a fixed register like
        %rbx. That distinction matters here specifically because
        `right` can itself be *another* string concatenation or
        comparison (or a call to a function that does one): if `left`
        were sitting in %rbx while `right` gets evaluated, and `right`'s
        own evaluation also needs %rbx for its own left/right dance
        (which it does, being this same method, or via a called
        function that itself calls this method), it would silently
        clobber `left` before this method ever gets to use it. The
        stack has no such fixed-identity conflict, no matter how deeply
        this nests.

        %r12 (holding `right`), by contrast, *is* safe to set as a fixed
        register immediately after `right`'s evaluation completes:
        nothing between that point and this method's own use of %r12
        recurses back into gen_expr_into, so there's no nested
        evaluation left that could still clobber it -- only the fixed
        strlen/malloc/strcpy/strcat sequence below runs, and libc is
        itself SysV-ABI-compliant, so it's required to preserve %r12 as
        a callee-saved register on its own. %r13/%r14 are similarly
        only ever written once, from a direct call result, with nothing
        recursive happening afterward.

        This is a *distinct* concern from why every function's own
        prologue/epilogue also saves/restores %rbx/%r12/%r13/%r14 (see
        gen_function and gen_return) -- that fix protects a value held
        in one of these registers *across a call into another
        function*; this one protects a value held here *across
        evaluating a nested expression within the same function*. Both
        are needed; neither replaces the other.

        MEMORY: once an operand's bytes have been fully copied out
        (strcpy for left, strcat for right), if that operand was itself
        a fresh, unnamed concatenation result -- a Binary(ADD, ...)
        sub-expression, never a named variable, a literal, or a
        function call's return value -- its buffer is immediately
        freed. See _gen_free_if_fresh_concat and the module docstring's
        STRINGS section for exactly why this specific, narrow case is
        safe to free automatically with no broader escape analysis.
        """
        if not isinstance(dst, Register):
            raise CodegenError(f"Binary codegen requires a register destination, got: {dst!r}")
        result = as_qword_register(dst)

        instructions = self.gen_expr_into(expr.left, dst)
        instructions.append(Push(result))                                # save left on the stack
        instructions.extend(self.gen_expr_into(expr.right, dst))         # left is safe regardless of what this does
        instructions.append(MovQ(src=result, dst=Register('r12')))       # r12 = right
        instructions.append(Pop(result))                                 # restore left
        instructions.append(MovQ(src=result, dst=Register('rbx')))       # rbx = left

        instructions.append(MovQ(src=Register('rbx'), dst=Register('rdi')))
        instructions.append(CallInstr('strlen'))
        instructions.append(MovQ(src=Register('rax'), dst=Register('r13')))  # r13 = len(left)

        instructions.append(MovQ(src=Register('r12'), dst=Register('rdi')))
        instructions.append(CallInstr('strlen'))                              # rax = len(right)
        instructions.append(AddQ(src=Register('r13'), dst=Register('rax')))
        instructions.append(AddQ(src=Imm(1), dst=Register('rax')))       # rax = len(left)+len(right)+1
        instructions.append(MovQ(src=Register('rax'), dst=Register('rdi')))
        instructions.append(CallInstr('malloc'))
        instructions.append(MovQ(src=Register('rax'), dst=Register('r14')))  # r14 = new buffer

        instructions.append(MovQ(src=Register('r14'), dst=Register('rdi')))
        instructions.append(MovQ(src=Register('rbx'), dst=Register('rsi')))
        instructions.append(CallInstr('strcpy'))
        # left's bytes are now fully copied into the new buffer -- if
        # left was itself a fresh concatenation result, nothing else
        # can possibly still need it.
        instructions.extend(self._gen_free_if_fresh_concat(expr.left, 'rbx'))

        instructions.append(MovQ(src=Register('r14'), dst=Register('rdi')))
        instructions.append(MovQ(src=Register('r12'), dst=Register('rsi')))
        instructions.append(CallInstr('strcat'))
        # same reasoning, now that right's bytes have been appended.
        instructions.extend(self._gen_free_if_fresh_concat(expr.right, 'r12'))

        instructions.append(MovQ(src=Register('r14'), dst=result))
        return instructions

    def _gen_free_if_fresh_concat(self, operand: Node, holding_register: str) -> List[Instruction]:
        """If `operand` is itself a Binary(ADD, ...) node -- meaning
        whatever's sitting in `holding_register` right now is a fresh
        buffer that gen_string_concat_into just malloc'd for *this*
        expression alone, and which could never have been stored into a
        variable, returned from a function, or passed as an argument
        anywhere, since it only ever existed as this expression's own
        intermediate operand -- frees it. Everything else is left
        alone: a StringLiteral points into static `.data` and was never
        heap-allocated in the first place (freeing it would corrupt the
        allocator); a Variable or a Call's return value might be
        aliased by other code we have no visibility into here (a named
        variable could be read again later, a call's return value could
        be a parameter passed straight through, etc.) -- telling those
        apart from a genuinely fresh, exclusively-owned buffer is a
        real escape-analysis problem this narrow check deliberately
        doesn't attempt to solve. See the module docstring's STRINGS
        section for the fuller reasoning and what's intentionally still
        left leaking as a result.
        """
        if isinstance(operand, Binary) and operand.op == BinaryOp.ADD:
            return [
                MovQ(src=Register(holding_register), dst=Register('rdi')),
                CallInstr('free'),
            ]
        return []

    def gen_string_compare_into(self, expr: Binary, dst: Operand) -> List[Instruction]:
        """`left == right` / `left != right`, both str: calls `strcmp`
        (0 means equal) and converts that into this language's usual
        0/1 bool representation via the exact same cmp/setCC/movzx
        pattern every other comparison already uses -- reusing
        _COMPARISON_CONDITION_CODES[op] directly, since strcmp's result
        is a plain 32-bit int that "compared to 0" behaves exactly like
        any other int comparison from here on.

        `left` is protected across evaluating `right` via the stack, for
        exactly the same reason gen_string_concat_into does -- see its
        docstring.

        MEMORY: the same fresh-concatenation-result freeing
        gen_string_concat_into does, and for the same reason -- strcmp
        has already read both operands' bytes by the time this frees
        them, so there's nothing left that could need them. The one
        thing to get right here that concatenation didn't have to worry
        about: `call free` clobbers %rax/%eax exactly like any other
        call does, and strcmp's result is *sitting* in %eax at this
        point -- so it has to be stashed in a callee-saved register
        before either free() call, and restored into `dst` afterward,
        or freeing a fresh operand would silently destroy the very
        comparison result this method exists to compute.
        """
        if not isinstance(dst, Register):
            raise CodegenError(f"Binary codegen requires a register destination, got: {dst!r}")
        result = as_qword_register(dst)

        instructions = self.gen_expr_into(expr.left, dst)
        instructions.append(Push(result))
        instructions.extend(self.gen_expr_into(expr.right, dst))
        instructions.append(MovQ(src=result, dst=Register('r12')))
        instructions.append(Pop(result))
        instructions.append(MovQ(src=result, dst=Register('rbx')))

        instructions.append(MovQ(src=Register('rbx'), dst=Register('rdi')))
        instructions.append(MovQ(src=Register('r12'), dst=Register('rsi')))
        instructions.append(CallInstr('strcmp'))

        instructions.append(MovQ(src=result, dst=Register('r13')))  # stash strcmp's result before it can be clobbered
        instructions.extend(self._gen_free_if_fresh_concat(expr.left, 'rbx'))
        instructions.extend(self._gen_free_if_fresh_concat(expr.right, 'r12'))
        instructions.append(MovQ(src=Register('r13'), dst=result))  # restore it

        byte_dst = as_byte_register(dst)
        instructions.append(Cmp(src=Imm(0), dst=dst))
        instructions.append(SetCC(cc=_COMPARISON_CONDITION_CODES[expr.op], operand=byte_dst))
        instructions.append(MovZX(src=byte_dst, dst=dst))
        return instructions

    def gen_binary_op(self, op: BinaryOp, src: Operand, dst: Operand) -> List[Instruction]:
        if op == BinaryOp.ADD:
            return [Add(src=src, dst=dst)]
        if op == BinaryOp.SUBTRACT:
            return [Sub(src=src, dst=dst)]
        if op == BinaryOp.MULTIPLY:
            return [IMul(src=src, dst=dst)]
        if op == BinaryOp.DIVIDE:
            # idivl divides %edx:%eax by its operand, so by the time this
            # runs, the dividend (`dst`, i.e. left) must be in %eax and
            # the divisor (`src`, i.e. right) must be in a register --
            # both guaranteed by how gen_binary_into calls this.
            if dst != Register('eax'):
                raise CodegenError("Division currently requires its destination to be %eax")
            return [Cdq(), IDiv(src)]
        if op == BinaryOp.MODULO:
            # Exactly the same Cdq+IDiv sequence as DIVIDE -- idivl
            # always computes both the quotient (%eax) and the remainder
            # (%edx) in one instruction -- just followed by moving the
            # remainder into dst instead of leaving the quotient there.
            if dst != Register('eax'):
                raise CodegenError("Modulo currently requires its destination to be %eax")
            return [Cdq(), IDiv(src), Mov(src=Register('edx'), dst=Register('eax'))]
        if op == BinaryOp.BITWISE_AND:
            return [And(src=src, dst=dst)]
        if op == BinaryOp.BITWISE_OR:
            return [Or(src=src, dst=dst)]
        if op == BinaryOp.BITWISE_XOR:
            return [Xor(src=src, dst=dst)]
        if op == BinaryOp.SHIFT_LEFT:
            # `src` (== %ecx, per gen_binary_into) is never referenced
            # here -- ShiftLeft hardcodes %cl as its count operand,
            # since that's the only register x86 allows there, and %ecx
            # is already where the right-hand operand ends up by the
            # time gen_binary_op is called for any binary operator.
            return [ShiftLeft(dst=dst)]
        if op == BinaryOp.SHIFT_RIGHT:
            return [ShiftRightArithmetic(dst=dst)]
        if op in _COMPARISON_CONDITION_CODES:
            # Cmp(src=right, dst=left) computes (left - right) and sets
            # flags from that; SetCC turns the relevant flag combination
            # into a 0/1 byte; MovZX zero-extends that byte back out to
            # fill the full destination register (same pattern used for
            # NOT -- see gen_unary_op -- just against a computed `right`
            # instead of the literal 0).
            byte_dst = as_byte_register(dst)
            return [
                Cmp(src=src, dst=dst),
                SetCC(cc=_COMPARISON_CONDITION_CODES[op], operand=byte_dst),
                MovZX(src=byte_dst, dst=dst),
            ]
        raise CodegenError(f"No codegen rule for binary operator: {op}")

    def gen_unary_op(self, op: UnaryOp, dst: Operand) -> List[Instruction]:
        if op == UnaryOp.NEGATE:
            return [Neg(dst)]
        if op == UnaryOp.COMPLEMENT:
            return [Not(dst)]
        if op == UnaryOp.NOT:
            # `not x` is "1 if x == 0, else 0" -- the same cmp/setCC/movzx
            # pattern used for comparisons, just always against 0 and
            # always with cc='e'.
            byte_dst = as_byte_register(dst)
            return [
                Cmp(src=Imm(0), dst=dst),
                SetCC(cc='e', operand=byte_dst),
                MovZX(src=byte_dst, dst=dst),
            ]
        raise CodegenError(f"No codegen rule for unary operator: {op}")


# ---------------------------------------------------------------------------
# Assembly AST -> text
# ---------------------------------------------------------------------------

def _escape_for_asciz(s: str) -> str:
    """Escapes `s` (an already-unescaped Hornet string value -- see
    parser.py's _unescape_string_literal) for embedding in a GAS
    `.asciz "..."` directive. Backslash has to be escaped *first*, or
    the escapes added for the other characters would themselves get
    re-escaped; double-quote needs escaping since that's the
    directive's own delimiter; the rest are the common control
    characters getting their standard short escape so the emitted
    assembly stays readable text rather than raw control bytes."""
    s = s.replace('\\', '\\\\')
    s = s.replace('"', '\\"')
    s = s.replace('\n', '\\n')
    s = s.replace('\t', '\\t')
    s = s.replace('\r', '\\r')
    return s


class Emitter:
    """Renders an AsmProgram as textual x64 AT&T-syntax assembly.

    `platform` controls the portability wrinkles that matter at this
    stage of the compiler:
      - macOS (Mach-O) requires a leading underscore on external symbols
        (e.g. `_main`, and now also library calls like `_malloc`); Linux
        (ELF) does not. This applies uniformly to this program's own
        function labels (emit_function) and to Call instruction targets
        (also emit_function, since that's where every instruction gets
        rendered) -- both go through the same symbol() method.
      - Linux toolchains generally expect a `.note.GNU-stack` section so
        the linker doesn't warn about an executable stack; macOS doesn't
        use this.
    """

    def __init__(self, platform: str = 'macos'):
        if platform not in ('macos', 'linux'):
            raise ValueError("platform must be 'macos' or 'linux'")
        self.platform = platform

    def symbol(self, name: str) -> str:
        return f"_{name}" if self.platform == 'macos' else name

    def emit(self, program: AsmProgram) -> str:
        lines: List[str] = []
        for fn in program.functions:
            lines.extend(self.emit_function(fn))
            lines.append("")  # blank line between functions
        if program.string_literals:
            # Plain `.data` rather than a stricter read-only section
            # (like ELF's `.rodata` or Mach-O's `__TEXT,__cstring`) on
            # purpose -- `.data` is the one directive that assembles
            # correctly, unchanged, on both this Linux sandbox and
            # macOS's assembler, and nothing in this language ever
            # writes back into a string literal's bytes anyway, so the
            # extra write-protection those stricter sections would give
            # isn't actually buying anything here.
            lines.append(".data")
            for label, content in program.string_literals:
                lines.append(f"{label}:")
                lines.append(f'    .asciz "{_escape_for_asciz(content)}"')
            lines.append("")
        if self.platform == 'linux':
            lines.append('.section .note.GNU-stack,"",@progbits')
        return "\n".join(lines).rstrip() + "\n"

    def emit_function(self, fn: AsmFunction) -> List[str]:
        sym = self.symbol(fn.name)
        lines = [f"    .globl {sym}", f"{sym}:"]
        for instr in fn.instructions:
            if isinstance(instr, CallInstr):
                # Call.emit() renders its target unprefixed -- platform
                # symbol naming is this Emitter's job alone, same as for
                # this program's own function labels above, so this is
                # the one instruction type emit_function special-cases
                # rather than just calling instr.emit() uniformly.
                lines.append(f"    call    {self.symbol(instr.target)}")
            else:
                lines.append(f"    {instr.emit()}")
        return lines


# ---------------------------------------------------------------------------
# Convenience entry points
# ---------------------------------------------------------------------------

def generate_asm(program: Program, platform: str = 'macos') -> str:
    asm_program = CodeGenerator().generate(program)
    return Emitter(platform=platform).emit(asm_program)


def compile_to_asm(filename: str, platform: str = 'macos') -> str:
    tokens = lex(filename)
    ast = Parser(tokens).parse_program()
    analyze(ast)  # raises SemanticError before any code is generated
    return generate_asm(ast, platform=platform)


def main():
    arg_parser = argparse.ArgumentParser(description='Assembly generator')
    arg_parser.add_argument('file', type=str, help='Source file to compile.')
    arg_parser.add_argument(
        '--platform', choices=['macos', 'linux'], default='macos',
        help="Target platform; affects symbol naming. Default: macos",
    )
    arg_parser.add_argument(
        '-o', '--output', type=str, default=None,
        help='Write assembly to this file instead of stdout.',
    )
    args = arg_parser.parse_args()

    asm = compile_to_asm(args.file, platform=args.platform)
    if args.output:
        with open(args.output, 'w') as f:
            f.write(asm)
    else:
        print(asm, end='')


if __name__ == '__main__':
    main()
