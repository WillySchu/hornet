"""AsmProgram -> AT&T assembly text."""

from codegen.assembly_ast import AsmProgram, AsmFunction, CallInstr
from codegen.utils import escape_for_asciz


class Emitter:
    """`platform` controls symbol prefixes and section names."""

    def __init__(self, platform: str = 'macos'):
        if platform not in ('macos', 'linux'):
            raise ValueError("platform must be 'macos' or 'linux'")
        self.platform = platform

    def symbol(self, name: str) -> str:
        return f"_{name}" if self.platform == 'macos' else name

    def emit(self, program: AsmProgram) -> str:
        lines: list[str] = []
        for fn in program.functions:
            lines.extend(self.emit_function(fn))
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
        if self.platform == 'linux':
            lines.append('.section .note.GNU-stack,"",@progbits')
        return "\n".join(lines).rstrip() + "\n"

    def emit_function(self, fn: AsmFunction) -> list[str]:
        sym = self.symbol(fn.name)
        lines = [f"    .globl {sym}", f"{sym}:"]
        for instr in fn.instructions:
            if isinstance(instr, CallInstr):
                lines.append(f"    call    {self.symbol(instr.target)}")
            else:
                lines.append(f"    {instr.emit()}")
        return lines
