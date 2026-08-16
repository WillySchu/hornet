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
                the six comparisons (== != < > <= >=), and the two
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
    Constant,
    Continue,
    ExprStmt,
    Function,
    If,
    Node,
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
    through the %ecx scratch register rather than leaving it as an Imm."""
    operand: Operand
    mnemonic = "idivl"

    def operands(self) -> List[str]:
        return [self.operand.emit()]


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
class Call(Instruction):
    """Calls an external function by symbol name, e.g. `call strlen`.

    `target` is always the *unprefixed* C symbol name (`malloc`, not
    `_malloc`) -- Emitter is what knows whether the target platform
    needs a leading underscore (see its emit_function), the same way it
    already decides that for this program's own function labels. Emit()
    here (unprefixed) is only ever used if a Call is inspected/rendered
    outside of Emitter; the real rendering path always goes through
    Emitter's own handling instead.

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
        self._collect_locals(fn.body)
        self.scopes = [{}]

        instructions: List[Instruction] = [
            Push(Register('rbp')),
            MovQ(src=Register('rsp'), dst=Register('rbp')),
        ]
        frame_size = self._frame_size()
        if frame_size:
            instructions.append(SubQ(src=Imm(frame_size), dst=Register('rsp')))

        for stmt in fn.body:
            instructions.extend(self.gen_statement(stmt))

        return AsmFunction(name=fn.name, instructions=instructions)

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
        # Total bytes used by locals, rounded up to a 16-byte boundary.
        # Now genuinely required, not just good practice: gen_string_*
        # emits real `call` instructions (to malloc/strlen/strcpy/
        # strcat/strcmp), and the SysV ABI requires %rsp to be
        # 16-byte-aligned at the point of every one of those.
        raw = -self._next_offset
        return ((raw + 15) // 16) * 16 if raw > 0 else 0

    def _push_scope(self) -> None:
        self.scopes.append({})

    def _pop_scope(self) -> None:
        self.scopes.pop()

    def _bind_local(self, stmt: VarDecl) -> int:
        """Registers `stmt`'s name -- and its declared type, needed by
        _local_type/_infer_type -- in the current (innermost)
        generation-time scope, pointing at the permanent offset
        _collect_locals already assigned this exact VarDecl node, and
        returns that offset."""
        offset = self._var_offsets[id(stmt)]
        self.scopes[-1][stmt.name] = (offset, stmt.var_type)
        return offset

    def _local_offset(self, name: str) -> int:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name][0]
        raise CodegenError(f"Reference to undeclared variable '{name}'")

    def _local_type(self, name: str) -> str:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name][1]
        raise CodegenError(f"Reference to undeclared variable '{name}'")

    def _infer_type(self, expr: Node) -> str:
        """A lightweight, duplicate-of-semantic.py type inference --
        trusts the program has already passed semantic analysis (so
        e.g. a Binary's operands are guaranteed type-consistent), and
        only needs to answer "is this str, so a pointer, or not" for
        deciding operand width and which of a handful of overloaded
        operators (`+`, `==`, `!=`) actually apply. See the module
        docstring's STRINGS section, and its LOCAL VARIABLES section
        for the broader precedent of codegen re-deriving something
        semantic.py already computed rather than sharing state with it.
        """
        if isinstance(expr, Constant):
            return 'int'
        if isinstance(expr, BoolLiteral):
            return 'bool'
        if isinstance(expr, StringLiteral):
            return 'str'
        if isinstance(expr, Variable):
            return self._local_type(expr.name)
        if isinstance(expr, Unary):
            return 'bool' if expr.op == UnaryOp.NOT else 'int'
        if isinstance(expr, Binary):
            if expr.op == BinaryOp.ADD:
                # str+str -> str, int+int -> int; semantic.py already
                # ruled out any other combination, so the left operand's
                # type alone determines which this is.
                return self._infer_type(expr.left)
            if expr.op in (BinaryOp.SUBTRACT, BinaryOp.MULTIPLY, BinaryOp.DIVIDE):
                return 'int'
            return 'bool'  # every comparison, and/or
        raise CodegenError(f"Cannot infer a type for expression: {expr!r}")

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
        if self._infer_type(value_expr) == 'str':
            instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', offset)))
        else:
            instructions.append(Mov(src=Register('eax'), dst=Memory('rbp', offset)))
        return instructions

    def gen_return(self, stmt: Return) -> List[Instruction]:
        dst = Register('eax')
        instructions = self.gen_expr_into(stmt.value, dst)
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
            if expr.op == BinaryOp.ADD and self._infer_type(expr.left) == 'str':
                return self.gen_string_concat_into(expr, dst)
            if expr.op in (BinaryOp.EQUAL, BinaryOp.NOT_EQUAL) and self._infer_type(expr.left) == 'str':
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

        Uses %rbx/%r12/%r13/%r14 as scratch to hold left, right, and
        intermediate values *across* several `call` instructions,
        rather than the stack-spill scheme every other binary operator
        uses (see the module docstring's STRINGS section for why: those
        are callee-saved registers, so nothing this function calls can
        clobber them, which avoids having to hand-track stack offsets
        that shift with every additional push -- exactly the kind of
        bookkeeping that gets error-prone once more than one value needs
        to stay alive across more than one call). Nothing here saves or
        restores their *prior* contents, since nothing else in this
        codebase currently uses them; a real register allocator would
        need to reconsider that.

        Never frees the buffer -- see the module docstring's STRINGS
        section for why that's an accepted, explicitly-documented
        simplification for this first pass rather than an oversight.
        """
        if not isinstance(dst, Register):
            raise CodegenError(f"Binary codegen requires a register destination, got: {dst!r}")
        result = as_qword_register(dst)

        instructions = self.gen_expr_into(expr.left, dst)
        instructions.append(MovQ(src=result, dst=Register('rbx')))       # rbx = left
        instructions.extend(self.gen_expr_into(expr.right, dst))
        instructions.append(MovQ(src=result, dst=Register('r12')))       # r12 = right

        instructions.append(MovQ(src=Register('rbx'), dst=Register('rdi')))
        instructions.append(Call('strlen'))
        instructions.append(MovQ(src=Register('rax'), dst=Register('r13')))  # r13 = len(left)

        instructions.append(MovQ(src=Register('r12'), dst=Register('rdi')))
        instructions.append(Call('strlen'))                              # rax = len(right)
        instructions.append(AddQ(src=Register('r13'), dst=Register('rax')))
        instructions.append(AddQ(src=Imm(1), dst=Register('rax')))       # rax = len(left)+len(right)+1
        instructions.append(MovQ(src=Register('rax'), dst=Register('rdi')))
        instructions.append(Call('malloc'))
        instructions.append(MovQ(src=Register('rax'), dst=Register('r14')))  # r14 = new buffer

        instructions.append(MovQ(src=Register('r14'), dst=Register('rdi')))
        instructions.append(MovQ(src=Register('rbx'), dst=Register('rsi')))
        instructions.append(Call('strcpy'))

        instructions.append(MovQ(src=Register('r14'), dst=Register('rdi')))
        instructions.append(MovQ(src=Register('r12'), dst=Register('rsi')))
        instructions.append(Call('strcat'))

        instructions.append(MovQ(src=Register('r14'), dst=result))
        return instructions

    def gen_string_compare_into(self, expr: Binary, dst: Operand) -> List[Instruction]:
        """`left == right` / `left != right`, both str: calls `strcmp`
        (0 means equal) and converts that into this language's usual
        0/1 bool representation via the exact same cmp/setCC/movzx
        pattern every other comparison already uses -- reusing
        _COMPARISON_CONDITION_CODES[op] directly, since strcmp's result
        is a plain 32-bit int that "compared to 0" behaves exactly like
        any other int comparison from here on."""
        if not isinstance(dst, Register):
            raise CodegenError(f"Binary codegen requires a register destination, got: {dst!r}")
        result = as_qword_register(dst)

        instructions = self.gen_expr_into(expr.left, dst)
        instructions.append(MovQ(src=result, dst=Register('rbx')))
        instructions.extend(self.gen_expr_into(expr.right, dst))
        instructions.append(MovQ(src=result, dst=Register('r12')))

        instructions.append(MovQ(src=Register('rbx'), dst=Register('rdi')))
        instructions.append(MovQ(src=Register('r12'), dst=Register('rsi')))
        instructions.append(Call('strcmp'))

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
            if isinstance(instr, Call):
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
