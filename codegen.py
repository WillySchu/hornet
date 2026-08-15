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
    Unary    -> NEGATE ('-'), COMPLEMENT ('~'), and NOT ('!'), each
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
so on (each int is 4 bytes). gen_function pre-scans a function's whole
body for VarDecls *before* generating any instructions, so every slot's
offset is already assigned, and the total frame size is already known,
by the time the prologue is emitted:

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

Two related simplifications worth knowing about, both consequences of
there being no semantic-analysis pass yet: declaring the same variable
twice in one function, and referencing a variable that was never
declared, are both only caught here in codegen (see _declare_local and
_local_offset) -- not in the parser. And because the whole function body
is pre-scanned for VarDecls up front rather than incrementally as
statements are generated, declare-before-use in textual order isn't
enforced either; a variable assigned above its own declaration would
"work" instead of being rejected. Both are fine for now and worth
revisiting once real error-checking matters more than getting variables
working at all.
"""

import argparse
from dataclasses import dataclass, field
from typing import List

from lexer import lex
from parser import (
    Assign,
    Binary,
    BinaryOp,
    BoolLiteral,
    Constant,
    ExprStmt,
    Function,
    Node,
    Parser,
    Program,
    Return,
    Unary,
    UnaryOp,
    VarDecl,
    Variable,
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
class MovQ(Instruction):
    """64-bit mov (`movq`). Distinct from Mov (`movl`, 32-bit) -- used
    for frame-pointer setup (`movq %rsp, %rbp`), never for general int
    computation, since every value in this language is a 32-bit int."""
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
        self.locals = {}  # name -> Memory offset; reset per function, see gen_function

    def new_label(self, prefix: str) -> str:
        """Returns a fresh, uniquely-numbered local label like
        `.Land_short_0`. Needed because AND/OR codegen emits real jump
        targets, and a program can contain any number of them -- each
        one needs a name the assembler won't collide with any other."""
        label = f".L{prefix}_{self._label_count}"
        self._label_count += 1
        return label

    def generate(self, program: Program) -> AsmProgram:
        return AsmProgram(functions=[self.gen_function(fn) for fn in program.functions])

    def gen_function(self, fn: Function) -> AsmFunction:
        # Fresh locals table per function -- variables don't persist
        # across functions, and offsets are relative to *this* function's
        # own %rbp.
        self.locals = {}
        for stmt in fn.body:
            if isinstance(stmt, VarDecl):
                self._declare_local(stmt.name)

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

    def _declare_local(self, name: str) -> None:
        if name in self.locals:
            raise CodegenError(f"Variable '{name}' is already declared in this function")
        # Every local is a 4-byte int; slots grow downward from %rbp, so
        # the first declared variable ends up closest to it (-4(%rbp)).
        offset = -4 * (len(self.locals) + 1)
        self.locals[name] = offset

    def _local_offset(self, name: str) -> int:
        if name not in self.locals:
            raise CodegenError(f"Reference to undeclared variable '{name}'")
        return self.locals[name]

    def _frame_size(self) -> int:
        # Total bytes used by locals, rounded up to a 16-byte boundary.
        # Not strictly required yet (nothing here makes a `call`, so the
        # SysV ABI's 16-byte-alignment-before-call rule doesn't bind),
        # but keeping frames aligned now avoids having to retrofit it
        # once function calls exist.
        raw = 4 * len(self.locals)
        return ((raw + 15) // 16) * 16 if raw else 0

    def gen_statement(self, stmt: Node) -> List[Instruction]:
        if isinstance(stmt, VarDecl):
            return self.gen_var_decl(stmt)
        if isinstance(stmt, Assign):
            return self.gen_assign(stmt)
        if isinstance(stmt, Return):
            return self.gen_return(stmt)
        if isinstance(stmt, ExprStmt):
            return self.gen_expr_stmt(stmt)
        raise CodegenError(f"No codegen rule for statement: {stmt!r}")

    def gen_var_decl(self, stmt: VarDecl) -> List[Instruction]:
        # `_declare_local` already ran for every VarDecl in gen_function
        # (that's what sizes the frame), so the slot already exists here
        # -- this only needs to emit the store, and only if there's an
        # initializer. `int a` with no initializer leaves the slot's
        # contents genuinely uninitialized, matching C: reading it before
        # assigning is undefined behavior, not implicitly zero.
        if stmt.init is None:
            return []
        return self._gen_store(stmt.name, stmt.init)

    def gen_assign(self, stmt: Assign) -> List[Instruction]:
        return self._gen_store(stmt.name, stmt.value)

    def _gen_store(self, name: str, value_expr: Node) -> List[Instruction]:
        """Shared by VarDecl-with-initializer and Assign: both are just
        "compute this expression, then write the result into that
        variable's slot" -- evaluate into %eax using the ordinary
        expression codegen (unchanged, still always targets a register),
        then a single extra `movl %eax, offset(%rbp)` to store it. This
        way none of gen_expr_into/gen_binary_into/gen_unary_op need to
        know memory operands exist at all -- they just always work with
        %eax, same as before variables existed."""
        offset = self._local_offset(name)
        instructions = self.gen_expr_into(value_expr, Register('eax'))
        instructions.append(Mov(src=Register('eax'), dst=Memory('rbp', offset)))
        return instructions

    def gen_return(self, stmt: Return) -> List[Instruction]:
        dst = Register('eax')
        instructions = self.gen_expr_into(stmt.value, dst)
        instructions.append(Leave())
        instructions.append(Ret())
        return instructions

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
        if isinstance(expr, Variable):
            offset = self._local_offset(expr.name)
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
            # `!x` is "1 if x == 0, else 0" -- the same cmp/setCC/movzx
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

class Emitter:
    """Renders an AsmProgram as textual x64 AT&T-syntax assembly.

    `platform` controls the two portability wrinkles that matter at this
    stage of the compiler:
      - macOS (Mach-O) requires a leading underscore on external symbols
        (e.g. `_main`); Linux (ELF) does not.
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
        if self.platform == 'linux':
            lines.append('.section .note.GNU-stack,"",@progbits')
        return "\n".join(lines).rstrip() + "\n"

    def emit_function(self, fn: AsmFunction) -> List[str]:
        sym = self.symbol(fn.name)
        lines = [f"    .globl {sym}", f"{sym}:"]
        for instr in fn.instructions:
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
