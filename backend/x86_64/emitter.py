"""AsmProgram -> AT&T assembly text."""

from backend.x86_64.assembly_ast import AsmProgram, AsmFunction, CallInstr, SafepointCheck
from backend.common.function_table import end_label, table_lines
from backend.common.text import escape_for_asciz
from target import Target


class Emitter:
    """The target's OS controls symbol prefixes and section names (Windows, through MinGW, takes the
    same text as Linux without the ELF stack note)."""

    def __init__(self, target: Target):
        self.os = target.os

    def symbol(self, name: str) -> str:
        return f"_{name}" if self.os == 'macos' else name

    def emit(self, program: AsmProgram) -> str:
        lines: list[str] = []
        for index, fn in enumerate(program.functions):
            lines.extend(self.emit_function(fn, index))
            lines.append("")
        if program.string_literals or program.type_descriptors:
            # writable .data: portable across ELF and Mach-O
            lines.append(".data")
            for label, content in program.string_literals:
                lines.append(f"{label}:")
                lines.append(f'    .asciz "{escape_for_asciz(content)}"')
            for label, fields in program.type_descriptors:
                # int fields emit as .quad literals; strings as label addresses
                lines.append(f"{label}:")
                for f in fields:
                    lines.append(f"    .quad {f}")
            lines.append("")
        lines += table_lines([fn.name for fn in program.functions], self.symbol) + [""]
        if self.os == 'linux':
            lines.append('.section .note.GNU-stack,"",@progbits')
        return "\n".join(lines).rstrip() + "\n"

    def emit_function(self, fn: AsmFunction, index: int) -> list[str]:
        """`fn`'s text, the function at `index` in the program: its end is labelled for the table of
        functions the runtime reads."""
        sym = self.symbol(fn.name)
        unwind = self.os == 'windows'  # the function's unwind table: its directives sit between these two
        lines = [f"    .globl {sym}"] + ([f"    .seh_proc {sym}"] if unwind else []) + [f"{sym}:"]
        for instr in fn.instructions:
            if isinstance(instr, CallInstr):
                lines.append(f"    call    {self.symbol(instr.target)}")
            elif isinstance(instr, SafepointCheck):
                lines.append(f"    cmpb    $0, {self.symbol(instr.flag)}(%rip)")
            else:
                lines.append(f"    {instr.emit()}")
        return lines + [f"{end_label(index)}:"] + (["    .seh_endproc"] if unwind else [])
