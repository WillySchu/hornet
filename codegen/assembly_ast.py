"""Definitions for assembly AST operands."""

from dataclasses import dataclass, field


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
    """A memory operand: `offset(%base)`, e.g. `-4(%rbp)`. This is how
    every local variable is stored -- see the module docstring's LOCAL
    VARIABLES section -- with `base` almost always 'rbp'. It's also
    reused, with a DIFFERENT base, for reading/writing through a
    computed address held in some other register (e.g. Memory('rbx',
    0) for the address an array index computed) -- see the ARRAYS
    section for why array copying/addressing needed this generality
    that scalar locals never did."""
    base: str    # e.g. 'rbp', or another register holding a computed address
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

    def operands(self) -> list[str]:
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

    def operands(self) -> list[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class Neg(Instruction):
    """Two's-complement arithmetic negation, in place: dst = -dst."""
    operand: Operand
    mnemonic = "negl"

    def operands(self) -> list[str]:
        return [self.operand.emit()]


@dataclass
class Not(Instruction):
    """Bitwise complement, in place: dst = ~dst."""
    operand: Operand
    mnemonic = "notl"

    def operands(self) -> list[str]:
        return [self.operand.emit()]


@dataclass
class Cmp(Instruction):
    """Compares src and dst by computing dst - src and setting flags
    (notably ZF) accordingly -- doesn't modify either operand."""
    src: Operand
    dst: Operand
    mnemonic = "cmpl"

    def operands(self) -> list[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class CmpQ(Instruction):
    """64-bit compare (`cmpq`) -- the CmpQ counterpart to Cmp (`cmpl`,
    32-bit), for the one case that needs it: checking a slice
    descriptor's own 64-bit `ptr` field against 0 (see
    gen_slice_none_comparison_into). Every OTHER comparison in this
    language compares 32-bit int/bool values, for which Cmp's cmpl is
    exactly right -- but a pointer is a full 64-bit value, and
    comparing only its low 32 bits against zero could, in principle
    (however unlikely for any real address in practice), miss a real,
    non-null pointer whose low 32 bits happen to be zero."""
    src: Operand
    dst: Operand
    mnemonic = "cmpq"

    def operands(self) -> list[str]:
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

    def operands(self) -> list[str]:
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

    def operands(self) -> list[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class Add(Instruction):
    """dst += src."""
    src: Operand
    dst: Operand
    mnemonic = "addl"

    def operands(self) -> list[str]:
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

    def operands(self) -> list[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class Sub(Instruction):
    """dst -= src."""
    src: Operand
    dst: Operand
    mnemonic = "subl"

    def operands(self) -> list[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class IMul(Instruction):
    """dst *= src (signed, two-operand form)."""
    src: Operand
    dst: Operand
    mnemonic = "imull"

    def operands(self) -> list[str]:
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

    def operands(self) -> list[str]:
        return [self.operand.emit()]


@dataclass
class Div(Instruction):
    """Divides the 64-bit %edx:%eax pair by `operand` (UNSIGNED, unlike
    IDiv). Quotient in %eax, remainder in %edx, same as IDiv -- the
    only difference is the interpretation of the bits, so %edx must be
    explicitly zeroed first (`movl $0, %edx`), never sign-extended via
    Cdq, which would inject a sign bit into a value this instruction is
    about to treat as having none.

    Exists specifically for converting an int's own MAGNITUDE to
    decimal digits (see the print machinery's own int-to-string
    conversion) without ever risking a signed-overflow trap: negating
    INT_MIN in ordinary 32-bit two's complement doesn't actually change
    its bit pattern at all (there's no positive counterpart to negate
    to), but that SAME bit pattern, read as unsigned rather than
    signed, correctly represents INT_MIN's own magnitude
    (2147483648) -- a value that doesn't fit in a signed 32-bit int at
    all, but fits an unsigned one perfectly. Dividing that magnitude
    with Div rather than IDiv is what lets the digit-extraction loop
    stay in ordinary 32-bit arithmetic throughout, with no need for a
    64-bit widening step anywhere, while still handling every int
    value -- including this one specific edge case -- correctly."""
    operand: Operand
    mnemonic = "divl"

    def operands(self) -> list[str]:
        return [self.operand.emit()]


@dataclass
class And(Instruction):
    """dst &= src (bitwise AND)."""
    src: Operand
    dst: Operand
    mnemonic = "andl"

    def operands(self) -> list[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class Or(Instruction):
    """dst |= src (bitwise OR)."""
    src: Operand
    dst: Operand
    mnemonic = "orl"

    def operands(self) -> list[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class Xor(Instruction):
    """dst ^= src (bitwise XOR)."""
    src: Operand
    dst: Operand
    mnemonic = "xorl"

    def operands(self) -> list[str]:
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

    def operands(self) -> list[str]:
        return ['%cl', self.dst.emit()]


@dataclass
class ShiftRightArithmetic(Instruction):
    """dst >>= %cl, sign-extending (arithmetic) shift -- matches this
    language's `int` being signed, so `-8 >> 1 == -4`, not some large
    positive value from a zero-filling logical shift. See ShiftLeft's
    docstring for why %cl is hardcoded rather than a general `src`."""
    dst: Operand
    mnemonic = "sarl"

    def operands(self) -> list[str]:
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

    def operands(self) -> list[str]:
        return [self.operand.emit()]


@dataclass
class Pop(Instruction):
    """The pop counterpart to Push -- see its docstring."""
    operand: Register
    mnemonic = "popq"

    def operands(self) -> list[str]:
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

    def operands(self) -> list[str]:
        return [f"{self.label}(%rip)", self.dst.emit()]


@dataclass
class LeaQFrame(Instruction):
    """Loads the *address* of a %rbp-relative stack location into
    `dst` -- `leaq offset(%rbp), dst`. Distinct from LeaQ (which is
    RIP-relative, for static data like string literals): this is
    relative to the CURRENT function's own frame, and is how an
    array-typed local's address is obtained -- see the ARRAYS section
    for why arrays need their own address computed at all, unlike a
    scalar local, which is always read/written directly by offset."""
    offset: int
    dst: Register
    mnemonic = "leaq"

    def operands(self) -> list[str]:
        return [f"{self.offset}(%rbp)", self.dst.emit()]


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

    def operands(self) -> list[str]:
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

    def operands(self) -> list[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class MovB(Instruction):
    """8-bit mov (`movb`). Needed the first time this compiler ever
    copies a single BYTE to or from memory, as opposed to a 4-byte
    int/bool or an 8-byte pointer/qword -- see the print-buffer growth
    machinery this exists for. Both operands must already be 8-bit
    themselves (an 8-bit register alias, e.g. Register('al') via
    as_byte_register, an Imm, or a byte-addressed Memory location) --
    unlike Mov/MovQ, there's no separate 8-bit General-purpose register
    name at all (%al IS %eax's own low byte, not a distinct register),
    so passing a 32-bit Register here would silently assemble as
    something else entirely rather than raising a clear error; callers
    are responsible for using as_byte_register first."""
    src: Operand
    dst: Operand
    mnemonic = "movb"

    def operands(self) -> list[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class SubQ(Instruction):
    """64-bit subtract (`subq`). Used exactly once per function, in the
    prologue, to reserve stack space for locals: `subq $N, %rsp`."""
    src: Operand
    dst: Operand
    mnemonic = "subq"

    def operands(self) -> list[str]:
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

    def operands(self) -> list[str]:
        return [self.target]


@dataclass
class Je(Instruction):
    """Jump to `target` if the last Cmp found its operands equal (ZF set)."""
    target: str
    mnemonic = "je"

    def operands(self) -> list[str]:
        return [self.target]


@dataclass
class Jne(Instruction):
    """Jump to `target` if the last Cmp found its operands unequal (ZF clear)."""
    target: str
    mnemonic = "jne"

    def operands(self) -> list[str]:
        return [self.target]


@dataclass
class Jae(Instruction):
    """Jump to `target` if the last Cmp found dst >= src, using an
    UNSIGNED interpretation of the compared values -- unlike Je/Jne,
    which only look at the zero flag (equal or not, meaningless
    whether signed or unsigned). This is what makes array bounds
    checking a single comparison: `cmpl $size, %index; jae fail_label`
    correctly catches BOTH index >= size and index < 0 at once, since
    a negative int, reinterpreted unsigned, becomes a huge positive
    number -- see gen_index_address_into."""
    target: str
    mnemonic = "jae"

    def operands(self) -> list[str]:
        return [self.target]


@dataclass
class Ja(Instruction):
    """Jump to `target` if the last Cmp found dst > src (STRICTLY
    greater), using an UNSIGNED interpretation -- the strict-
    inequality counterpart to Jae, needed for slice bounds checking
    specifically (see gen_slice_into): `low == length` and
    `high == length` are both VALID slice bounds (`arr[5:5]` on a
    5-element array is a valid, empty-slice-producing expression),
    unlike ordinary indexing, where an index equal to the array's own
    size is already out of bounds -- so the boundary condition itself
    genuinely differs here, not just the label it jumps to. Still
    catches a negative value via the same unsigned-reinterpretation
    trick Jae relies on: a negative int, reinterpreted unsigned,
    becomes huge, and so is "above" any non-negative length."""
    target: str
    mnemonic = "ja"

    def operands(self) -> list[str]:
        return [self.target]


@dataclass
class Jle(Instruction):
    """Jump to `target` if the last Cmp found dst <= src, using a
    SIGNED interpretation -- unlike Jae/Ja, which are unsigned
    (array/slice lengths and indices, where a negative value needs to
    be caught by reinterpreting it as huge). Needed for the print
    buffer's own bulk-append growth check (see gen_buffer_append_
    bytes_into): comparing `needed` against `cap`, both ordinary,
    already-validated non-negative ints where a signed comparison is
    the natural, and here equivalent, choice -- matching how every
    ordinary int comparison elsewhere in this file (_COMPARISON_
    CONDITION_CODES) is already signed by default."""
    target: str
    mnemonic = "jle"

    def operands(self) -> list[str]:
        return [self.target]


@dataclass
class Jg(Instruction):
    """Jump to `target` if the last Cmp found dst > src, using a
    SIGNED interpretation -- the strict-inequality counterpart to
    Jle, for the identical reason and the identical use site."""
    target: str
    mnemonic = "jg"

    def operands(self) -> list[str]:
        return [self.target]


@dataclass
class Ret(Instruction):
    mnemonic = "ret"


@dataclass
class AsmFunction:
    name: str
    instructions: list[Instruction] = field(default_factory=list)


@dataclass
class AsmProgram:
    functions: list[AsmFunction] = field(default_factory=list)
    # (label, content) pairs for every string literal anywhere in the
    # program, collected across all functions during generation (see
    # CodeGenerator.gen_string_literal_into). These aren't tied to any
    # one function's frame -- they're static, immutable data -- so they
    # live at the AsmProgram level and get emitted once, in a shared
    # `.data` block, by Emitter (see its emit()).
    string_literals: list[tuple] = field(default_factory=list)
    # (label, fields) pairs for every runtime type descriptor built for
    # the print machinery (see CodeGenerator._get_or_build_type_
    # descriptor) -- the first place this compiler has ever needed any
    # runtime type information at all, since every other type-driven
    # decision anywhere else happens entirely at compile time. Each
    # `fields` entry is a flat list of ints (emitted as a literal
    # `.quad N`) and label-name strings (emitted as `.quad label`, a
    # perfectly ordinary assembler/linker relocation -- the same
    # mechanism behind a vtable or jump table in any real compiled
    # language, nothing new at the ASSEMBLY level, just new on this
    # compiler's own emission side). Structured identically to string_
    # literals for the identical reason: static, immutable, not tied to
    # any one function's own frame.
    type_descriptors: list[tuple] = field(default_factory=list)
