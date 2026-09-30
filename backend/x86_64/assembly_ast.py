"""Assembly AST: operands, instructions, and program containers (AT&T syntax)."""

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
    name: str

    def emit(self) -> str:
        return f"%{self.name}"


@dataclass
class Memory(Operand):
    """`offset(%base)`."""
    base: str
    offset: int

    def emit(self) -> str:
        return f"{self.offset}(%{self.base})"


@dataclass
class FrameSlot(Operand):
    """Unresolved frame slot; replaced with Memory after frame layout. Never emitted."""
    slot: int
    extra_offset: int = 0  # offset within the slot

    def emit(self) -> str:
        raise NotImplementedError(
            f"FrameSlot(slot={self.slot}) reached emit() unresolved -- "
            f"_patch_frame_slots should have replaced every one of "
            f"these with a concrete Memory operand first"
        )


class Instruction:
    """Instruction base: subclasses set `mnemonic` and `operands()`."""

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
    """dst = -dst."""
    operand: Operand
    mnemonic = "negl"

    def operands(self) -> list[str]:
        return [self.operand.emit()]


@dataclass
class NegQ(Instruction):
    """64-bit Neg."""
    operand: Operand
    mnemonic = "negq"

    def operands(self) -> list[str]:
        return [self.operand.emit()]


@dataclass
class Not(Instruction):
    """dst = ~dst."""
    operand: Operand
    mnemonic = "notl"

    def operands(self) -> list[str]:
        return [self.operand.emit()]


@dataclass
class NotQ(Instruction):
    """64-bit Not."""
    operand: Operand
    mnemonic = "notq"

    def operands(self) -> list[str]:
        return [self.operand.emit()]


@dataclass
class Cmp(Instruction):
    """Set flags from dst - src."""
    src: Operand
    dst: Operand
    mnemonic = "cmpl"

    def operands(self) -> list[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class CmpQ(Instruction):
    """64-bit Cmp."""
    src: Operand
    dst: Operand
    mnemonic = "cmpq"

    def operands(self) -> list[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class SetCC(Instruction):
    """Set an 8-bit operand to condition `cc` (0/1)."""
    cc: str
    operand: Operand

    @property
    def mnemonic(self) -> str:
        return f"set{self.cc}"

    def operands(self) -> list[str]:
        return [self.operand.emit()]


@dataclass
class MovZX(Instruction):
    """Zero-extend 8 -> 32 bits."""
    src: Operand
    dst: Operand
    mnemonic = "movzbl"

    def operands(self) -> list[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class MovSX(Instruction):
    """Sign-extend 8 -> 32 bits."""
    src: Operand
    dst: Operand
    mnemonic = "movsbl"

    def operands(self) -> list[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class MovSXD(Instruction):
    """Sign-extend 32 -> 64 bits."""
    src: Operand
    dst: Operand
    mnemonic = "movslq"

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
    """64-bit Add."""
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
    """dst *= src (signed)."""
    src: Operand
    dst: Operand
    mnemonic = "imull"

    def operands(self) -> list[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class IMulQ(Instruction):
    """64-bit IMul."""
    src: Operand
    dst: Operand
    mnemonic = "imulq"

    def operands(self) -> list[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class IMulWide(Instruction):
    """Signed %rdx:%rax = %rax * operand (one-operand imulq)."""
    operand: Operand
    mnemonic = "imulq"

    def operands(self) -> list[str]:
        return [self.operand.emit()]


@dataclass
class ShiftImmQ(Instruction):
    """64-bit shift of dst by an immediate count; kind is 'sar', 'shr', or 'shl'."""
    kind: str
    count: int
    dst: Operand

    @property
    def mnemonic(self) -> str:
        return f"{self.kind}q"

    def operands(self) -> list[str]:
        return [f"${self.count}", self.dst.emit()]


@dataclass
class Cdq(Instruction):
    """Sign-extend %eax into %edx:%eax (before IDiv)."""
    mnemonic = "cdq"


@dataclass
class Cqto(Instruction):
    """Sign-extend %rax into %rdx:%rax (before IDivQ)."""
    mnemonic = "cqto"


@dataclass
class IDiv(Instruction):
    """Signed %edx:%eax / operand: quotient %eax, remainder %edx. Operand must be a register or memory."""
    operand: Operand
    mnemonic = "idivl"

    def operands(self) -> list[str]:
        return [self.operand.emit()]


@dataclass
class IDivQ(Instruction):
    """64-bit IDiv."""
    operand: Operand
    mnemonic = "idivq"

    def operands(self) -> list[str]:
        return [self.operand.emit()]


@dataclass
class Div(Instruction):
    """Unsigned %edx:%eax / operand."""
    operand: Operand
    mnemonic = "divl"

    def operands(self) -> list[str]:
        return [self.operand.emit()]


@dataclass
class DivQ(Instruction):
    """64-bit Div."""
    operand: Operand
    mnemonic = "divq"

    def operands(self) -> list[str]:
        return [self.operand.emit()]


@dataclass
class And(Instruction):
    """dst &= src."""
    src: Operand
    dst: Operand
    mnemonic = "andl"

    def operands(self) -> list[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class AndQ(Instruction):
    """64-bit And."""
    src: Operand
    dst: Operand
    mnemonic = "andq"

    def operands(self) -> list[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class Or(Instruction):
    """dst |= src."""
    src: Operand
    dst: Operand
    mnemonic = "orl"

    def operands(self) -> list[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class OrQ(Instruction):
    """64-bit Or."""
    src: Operand
    dst: Operand
    mnemonic = "orq"

    def operands(self) -> list[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class Xor(Instruction):
    """dst ^= src."""
    src: Operand
    dst: Operand
    mnemonic = "xorl"

    def operands(self) -> list[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class XorQ(Instruction):
    """64-bit Xor."""
    src: Operand
    dst: Operand
    mnemonic = "xorq"

    def operands(self) -> list[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class ShiftLeft(Instruction):
    """dst <<= %cl."""
    dst: Operand
    mnemonic = "shll"

    def operands(self) -> list[str]:
        return ['%cl', self.dst.emit()]


@dataclass
class ShiftLeftQ(Instruction):
    """64-bit ShiftLeft."""
    dst: Operand
    mnemonic = "shlq"

    def operands(self) -> list[str]:
        return ['%cl', self.dst.emit()]


@dataclass
class ShiftRightArithmetic(Instruction):
    """dst >>= %cl, arithmetic."""
    dst: Operand
    mnemonic = "sarl"

    def operands(self) -> list[str]:
        return ['%cl', self.dst.emit()]


@dataclass
class ShiftRightArithmeticQ(Instruction):
    """64-bit ShiftRightArithmetic."""
    dst: Operand
    mnemonic = "sarq"

    def operands(self) -> list[str]:
        return ['%cl', self.dst.emit()]


@dataclass
class Push(Instruction):
    """Push a 64-bit register."""
    operand: Register
    mnemonic = "pushq"

    def operands(self) -> list[str]:
        return [self.operand.emit()]


@dataclass
class Pop(Instruction):
    """Pop a 64-bit register."""
    operand: Register
    mnemonic = "popq"

    def operands(self) -> list[str]:
        return [self.operand.emit()]


@dataclass
class LeaQ(Instruction):
    """RIP-relative address of `label`."""
    label: str
    dst: Register
    mnemonic = "leaq"

    def operands(self) -> list[str]:
        return [f"{self.label}(%rip)", self.dst.emit()]


@dataclass
class LeaQFrame(Instruction):
    """`leaq offset(%rbp), dst`."""
    offset: int
    dst: Register
    mnemonic = "leaq"

    def operands(self) -> list[str]:
        return [f"{self.offset}(%rbp)", self.dst.emit()]


@dataclass
class LeaQFrameSlot(Instruction):
    """Unresolved LeaQFrame; patched after frame layout."""
    slot: int
    dst: Register

    def emit(self) -> str:
        raise NotImplementedError(
            f"LeaQFrameSlot(slot={self.slot}) reached emit() unresolved -- "
            f"_patch_frame_slots should have replaced this whole "
            f"instruction with a concrete LeaQFrame first"
        )


@dataclass
class CallInstr(Instruction):
    """Call a symbol; the Emitter applies the target's symbol naming."""
    target: str
    mnemonic = "call"

    def operands(self) -> list[str]:
        return [self.target]


@dataclass
class MovQ(Instruction):
    """64-bit mov."""
    src: Operand
    dst: Operand
    mnemonic = "movq"

    def operands(self) -> list[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class MovB(Instruction):
    """8-bit mov."""
    src: Operand
    dst: Operand
    mnemonic = "movb"

    def operands(self) -> list[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class SubQ(Instruction):
    """64-bit subtract."""
    src: Operand
    dst: Operand
    mnemonic = "subq"

    def operands(self) -> list[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class Leave(Instruction):
    """`leave`: movq %rbp, %rsp; popq %rbp."""
    mnemonic = "leave"


@dataclass
class Label(Instruction):
    """Jump target."""
    name: str

    def emit(self) -> str:
        return f"{self.name}:"


@dataclass
class Jmp(Instruction):
    """Unconditional jump."""
    target: str
    mnemonic = "jmp"

    def operands(self) -> list[str]:
        return [self.target]


@dataclass
class JCC(Instruction):
    """Jump if condition `cc` (a setcc suffix, e.g. 'l', 'ne') holds."""
    cc: str
    target: str

    @property
    def mnemonic(self) -> str:
        return f"j{self.cc}"

    def operands(self) -> list[str]:
        return [self.target]


@dataclass
class Je(Instruction):
    """Jump if equal."""
    target: str
    mnemonic = "je"

    def operands(self) -> list[str]:
        return [self.target]


@dataclass
class Jne(Instruction):
    """Jump if not equal."""
    target: str
    mnemonic = "jne"

    def operands(self) -> list[str]:
        return [self.target]


@dataclass
class Jae(Instruction):
    """Jump if dst >= src (unsigned)."""
    target: str
    mnemonic = "jae"

    def operands(self) -> list[str]:
        return [self.target]


@dataclass
class Ja(Instruction):
    """Jump if dst > src (unsigned)."""
    target: str
    mnemonic = "ja"

    def operands(self) -> list[str]:
        return [self.target]


@dataclass
class Jle(Instruction):
    """Jump if dst <= src (signed)."""
    target: str
    mnemonic = "jle"

    def operands(self) -> list[str]:
        return [self.target]


@dataclass
class Jg(Instruction):
    """Jump if dst > src (signed)."""
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
    # (label, content)
    string_literals: list[tuple] = field(default_factory=list)
    # (label, fields)
    type_descriptors: list[tuple] = field(default_factory=list)
