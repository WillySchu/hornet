"""Escapes: what a string or byte literal's text stands for."""

# Keyed by the character after the backslash; \xNN is handled separately.
_ESCAPE_SEQUENCES = {
    'n': '\n',
    't': '\t',
    'r': '\r',
    '0': '\0',
    "'": "'",
    '"': '"',
    '\\': '\\',
}

_HEX_DIGITS = '0123456789abcdefABCDEF'


class BadEscape(Exception):
    """An escape that isn't one, `offset` characters into the literal's token."""

    def __init__(self, message: str, offset: int):
        super().__init__(message)
        self.message, self.offset = message, offset


def unescape_quoted_literal(raw: str) -> str:
    """Strip quotes and resolve escape sequences. BadEscape for a backslash that starts none: a
    mistyped one would otherwise quietly become some other text."""
    inner = raw[1:-1]
    chars = []
    i = 0
    while i < len(inner):
        ch = inner[i]
        if ch == '\\' and i + 1 < len(inner):
            nxt = inner[i + 1]
            if nxt == 'x':
                digits = inner[i + 2:i + 4]
                if len(digits) != 2 or digits[0] not in _HEX_DIGITS or digits[1] not in _HEX_DIGITS:
                    raise BadEscape("'\\x' must be followed by two hexadecimal digits, as in '\\x41'", i + 1)
                chars.append(chr(int(digits, 16)))
                i += 4
            elif nxt in _ESCAPE_SEQUENCES:
                chars.append(_ESCAPE_SEQUENCES[nxt])
                i += 2
            else:
                shown = nxt if ' ' <= nxt <= '~' else f"\\x{ord(nxt):02x}"
                raise BadEscape(
                    f"Unknown escape '\\{shown}' -- the escapes are \\n, \\t, \\r, \\0, \\\\, \\', \\\", and \\xNN "
                    f"(a byte, as two hexadecimal digits)", i + 1)
        else:
            chars.append(ch)
            i += 1
    return ''.join(chars)
