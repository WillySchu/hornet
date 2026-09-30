"""Assembler text shared by every GNU-syntax target."""


def escape_for_asciz(s: str) -> str:
    """Escape `s` for `.asciz`."""
    s = s.replace('\\', '\\\\')
    s = s.replace('"', '\\"')
    s = s.replace('\n', '\\n')
    s = s.replace('\t', '\\t')
    s = s.replace('\r', '\\r')
    s = s.replace('\0', '\\000')
    return s
