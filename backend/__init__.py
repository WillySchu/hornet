"""Backends, one package per architecture, each exposing lower_to_asm(ir_program, target) -> str."""

from diagnostics import CompileError
from target import IMPLEMENTED_ARCHES, Target


class TargetError(CompileError):
    """A target the compiler has no backend for."""


def lower_to_asm(ir_program, target: Target) -> str:
    """Hand an optimized IRProgram to the target architecture's backend."""
    if target.arch == 'x86_64':
        from backend.x86_64.codegen import lower_to_asm as lower
        return lower(ir_program, target)
    raise TargetError(f"no backend for {target.arch} yet (implemented: {', '.join(IMPLEMENTED_ARCHES)})")
