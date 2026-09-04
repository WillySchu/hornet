"""Emitter, which takes an AsmProgram and emits its assembly code."""

from codegen.assembly_ast import AsmProgram, AsmFunction, CallInstr
from codegen.utils import escape_for_asciz


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
        lines: list[str] = []
        for fn in program.functions:
            lines.extend(self.emit_function(fn))
            lines.append("")  # blank line between functions
        if program.string_literals or program.type_descriptors:
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
                lines.append(f'    .asciz "{escape_for_asciz(content)}"')
            for label, fields in program.type_descriptors:
                # Each field is either a plain int (a kind tag, a
                # count, a byte offset -- emitted as a literal .quad)
                # or a label name string (a pointer to another type
                # descriptor, or to a string literal holding a field's
                # own name -- emitted as .quad <label>, an ordinary
                # assembler/linker relocation that resolves correctly
                # regardless of whether that label appears earlier or
                # LATER in this same .data block, which is exactly
                # what makes a self-referential struct's own
                # descriptor -- pointing at its own, not-yet-fully-
                # emitted label -- work at all).
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
                # Call.emit() renders its target unprefixed -- platform
                # symbol naming is this Emitter's job alone, same as for
                # this program's own function labels above, so this is
                # the one instruction type emit_function special-cases
                # rather than just calling instr.emit() uniformly.
                lines.append(f"    call    {self.symbol(instr.target)}")
            else:
                lines.append(f"    {instr.emit()}")
        return lines
