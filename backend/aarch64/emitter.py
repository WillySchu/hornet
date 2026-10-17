"""AsmProgram -> GNU assembler text for AArch64 Linux (ELF) or macOS (Mach-O)."""

from backend.aarch64.assembly import (
    AsmFunction, AsmProgram, Call, Cond, GlobalPage, Imm, Instr, LabelDef, LabelRef, Mem, Reg, Shift, SymMem, SymPage,
    SymPageOffset,
)
from backend.common.function_table import end_label, table_lines
from backend.common.text import escape_for_asciz
from target import Target


class Emitter:
    def __init__(self, target: Target):
        self.os = target.os

    def symbol(self, name: str) -> str:
        return f"_{name}" if self.os == 'macos' else name

    def operand(self, op) -> str:
        if isinstance(op, Reg):
            return op.name
        if isinstance(op, Imm):
            return f"#{op.value}"
        if isinstance(op, Mem):
            if op.mode == 'pre':
                return f"[{op.base.name}, #{op.offset}]!"
            if op.mode == 'post':
                return f"[{op.base.name}], #{op.offset}"
            return f"[{op.base.name}, #{op.offset}]" if op.offset else f"[{op.base.name}]"
        if isinstance(op, LabelRef):
            return op.name
        if isinstance(op, SymPage):
            return f"{op.name}@PAGE" if self.os == 'macos' else op.name
        if isinstance(op, SymPageOffset):
            return f"{op.name}@PAGEOFF" if self.os == 'macos' else f":lo12:{op.name}"
        if isinstance(op, GlobalPage):
            return f"{self.symbol(op.name)}@PAGE" if self.os == 'macos' else op.name
        if isinstance(op, SymMem):
            if self.os == 'macos':
                return f"[{op.base.name}, {self.symbol(op.name)}@PAGEOFF]"
            return f"[{op.base.name}, :lo12:{op.name}]"
        if isinstance(op, Cond):
            return op.name
        if isinstance(op, Shift):
            return f"{op.kind} #{op.amount}"
        raise TypeError(f"unresolved operand {op!r}")

    def instruction(self, instr) -> str:
        if isinstance(instr, LabelDef):
            return f"{instr.name}:"
        if isinstance(instr, Call):
            return f"    bl      {self.symbol(instr.target)}"
        assert isinstance(instr, Instr), instr
        if not instr.operands:
            return f"    {instr.mnemonic}"
        return f"    {instr.mnemonic:<8}{', '.join(self.operand(o) for o in instr.operands)}"

    def emit(self, program: AsmProgram) -> str:
        lines = ["    .text"]
        for index, fn in enumerate(program.functions):
            lines.extend(self.emit_function(fn, index))
            lines.append("")
        if program.string_literals or program.type_descriptors:
            lines.append("    .data")
            for label, content in program.string_literals:
                lines.append(f"{label}:")
                lines.append(f'    .asciz "{escape_for_asciz(content)}"')
            for label, fields in program.type_descriptors:
                lines.append("    .p2align 3")
                lines.append(f"{label}:")
                for f in fields:
                    lines.append(f"    .quad {f}")
            lines.append("")
        lines += table_lines([fn.name for fn in program.functions], self.symbol) + [""]
        if self.os == 'linux':
            lines.append('    .section .note.GNU-stack,"",%progbits')
        return "\n".join(lines).rstrip() + "\n"

    def emit_function(self, fn: AsmFunction, index: int) -> list:
        """`fn`'s text, the function at `index` in the program: its end is labelled for the table of
        functions the runtime reads."""
        sym = self.symbol(fn.name)
        return [f"    .globl {sym}", "    .p2align 2", f"{sym}:"] + [self.instruction(i) for i in fn.instructions] + [
            f"{end_label(index)}:"]
