"""Lexer: source -> tokens, with Python-style INDENT/DEDENT."""

import argparse
import re
from enum import auto, Enum
from typing import Optional

from diagnostics import CompileError, InternalCompilerError, file_error


class TokenType(Enum):
    NUMBER = auto()
    IDENTIFIER = auto()
    STRING = auto()  # string literal; STR is the type keyword
    BYTE = auto()  # byte literal, double-quoted

    OPEN_PAREN = auto()
    CLOSE_PAREN = auto()
    OPEN_BRACKET = auto()
    CLOSE_BRACKET = auto()
    OPEN_BRACE = auto()
    CLOSE_BRACE = auto()
    COLON = auto()
    SEMICOLON = auto()
    COMMA = auto()
    DOT = auto()

    ASSIGN = auto()
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    TILDE = auto()
    AMPERSAND = auto()
    PIPE = auto()
    CARET = auto()
    SHIFT_LEFT = auto()
    SHIFT_RIGHT = auto()
    EQUAL = auto()
    NOT_EQUAL = auto()
    LESS_THAN = auto()
    GREATER_THAN = auto()
    LESS_THAN_OR_EQUAL = auto()
    GREATER_THAN_OR_EQUAL = auto()

    PLUS_ASSIGN = auto()
    MINUS_ASSIGN = auto()
    STAR_ASSIGN = auto()
    SLASH_ASSIGN = auto()
    PERCENT_ASSIGN = auto()
    AMPERSAND_ASSIGN = auto()
    PIPE_ASSIGN = auto()
    CARET_ASSIGN = auto()
    SHIFT_LEFT_ASSIGN = auto()
    SHIFT_RIGHT_ASSIGN = auto()

    DEF = auto()
    INT = auto()
    INT8 = auto()
    UINT8 = auto()
    INT64 = auto()
    INT32 = auto()
    STR = auto()
    RETURN = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    BOOL = auto()
    TRUE = auto()
    FALSE = auto()
    IF = auto()
    ELSE = auto()
    ELIF = auto()
    FOR = auto()
    WHILE = auto()
    BREAK = auto()
    CONTINUE = auto()
    DEFER = auto()
    NONE = auto()
    STRUCT = auto()
    ENUM = auto()
    TYPE = auto()
    IS = auto()
    IN = auto()
    MATCH = auto()
    AS = auto()
    IMPORT = auto()
    FROM = auto()
    EXTERN = auto()
    INTRINSIC = auto()
    DICT = auto()
    CONST = auto()
    NEVER = auto()

    NEWLINE = auto()
    INDENT = auto()
    DEDENT = auto()
    MISMATCH = auto()
    EOF = auto()


class LexError(CompileError):
    """Invalid character or indentation."""


class Token:
    def __init__(self, t: TokenType, val: str, line: int, col: int, file: Optional[str] = None):
        self.type = t
        self.val = val
        self.line = line
        self.col = col
        self.file = file

    def __eq__(self, other) -> bool:
        if not isinstance(other, Token):
            return False
        return (self.type, self.val, self.line, self.col) == (other.type, other.val, other.line, other.col)

    def __str__(self) -> str:
        return f'Token(type={self.type}, val={self.val})'

    def __repr__(self) -> str:
        return self.__str__()


class Lexer:
    def __init__(self, source: str, file: Optional[str] = None):
        self.source = source
        self.file = file
        self.tokens = []
        self.line = 1
        self.line_start = 0

        self.indent_stack = [0]
        self.at_line_start = True

        self.bracket_depth = 0

        self.keywords = {
            'def': TokenType.DEF,
            'int': TokenType.INT,
            'int8': TokenType.INT8,
            'uint8': TokenType.UINT8,
            'int64': TokenType.INT64,
            'int32': TokenType.INT32,
            # alias for uint8, not a distinct token
            'byte': TokenType.UINT8,
            'str': TokenType.STR,
            'return': TokenType.RETURN,
            'and': TokenType.AND,
            'or': TokenType.OR,
            'not': TokenType.NOT,
            'bool': TokenType.BOOL,
            'true': TokenType.TRUE,
            'false': TokenType.FALSE,
            'if': TokenType.IF,
            'else': TokenType.ELSE,
            'elif': TokenType.ELIF,
            'for': TokenType.FOR,
            'while': TokenType.WHILE,
            'break': TokenType.BREAK,
            'continue': TokenType.CONTINUE,
            'defer': TokenType.DEFER,
            'none': TokenType.NONE,
            'struct': TokenType.STRUCT,
            'enum': TokenType.ENUM,
            'type': TokenType.TYPE,
            'is': TokenType.IS,
            'in': TokenType.IN,
            'match': TokenType.MATCH,
            'as': TokenType.AS,
            'import': TokenType.IMPORT,
            'from': TokenType.FROM,
            'extern': TokenType.EXTERN,
            'intrinsic': TokenType.INTRINSIC,
            'dict': TokenType.DICT,
            'const': TokenType.CONST,
            'never': TokenType.NEVER,
        }

        self.rules = [
            ('NUMBER',      r'\d+(\.\d+)?'),
            ('IDENTIFIER',  r'[a-zA-Z_]\w*'),
            # Literals end on their own line: a raw newline inside one is an error (UNCLOSED).
            ('STRING',      r"'([^'\\\n]|\\.)*'"),
            ('BYTE',        r'"([^"\\\n]|\\.)*"'),

            # Longer operators must precede their prefixes (greedy alternation).
            ('SHIFT_LEFT_ASSIGN',  r'<<='),
            ('SHIFT_RIGHT_ASSIGN', r'>>='),

            ('EQUAL',                 r'=='),
            ('NOT_EQUAL',             r'!='),
            ('GREATER_THAN_OR_EQUAL', r'>='),
            ('LESS_THAN_OR_EQUAL',    r'<='),
            ('SHIFT_LEFT',            r'<<'),
            ('SHIFT_RIGHT',           r'>>'),
            ('PLUS_ASSIGN',      r'\+='),
            ('MINUS_ASSIGN',     r'\-='),
            ('STAR_ASSIGN',      r'\*='),
            ('SLASH_ASSIGN',     r'/='),
            ('PERCENT_ASSIGN',   r'%='),
            ('AMPERSAND_ASSIGN', r'&='),
            ('PIPE_ASSIGN',      r'\|='),
            ('CARET_ASSIGN',     r'\^='),

            ('NEWLINE',       r'\n'),
            ('OPEN_PAREN',    r'\('),
            ('CLOSE_PAREN',   r'\)'),
            ('OPEN_BRACKET',  r'\['),
            ('CLOSE_BRACKET', r'\]'),
            ('OPEN_BRACE',    r'\{'),
            ('CLOSE_BRACE',   r'\}'),
            ('GREATER_THAN',  r'>'),
            ('LESS_THAN',     r'<'),
            ('COLON',         r':'),
            ('SEMICOLON',     r';'),
            ('COMMA',         r','),
            ('ASSIGN',        r'='),
            ('PLUS',          r'\+'),
            ('MINUS',         r'\-'),
            ('STAR',          r'\*'),
            ('SLASH',         r'/'),
            ('PERCENT',       r'%'),
            ('TILDE',         r'\~'),
            ('AMPERSAND',     r'&'),
            ('PIPE',          r'\|'),
            ('CARET',         r'\^'),
            ('DOT',           r'\.'),

            ('COMMENT',       r'#[^\n]*'),  # Stops before '\n' so NEWLINE is still emitted.
            ('SKIP',          r'[ \t\r]+'),
            ('UNCLOSED',      r"'([^'\\\n]|\\.)*|\"([^\"\\\n]|\\.)*"),  # a literal cut off by a newline or the end
            ('MISMATCH',      r'.'),
        ]

        # ASCII: a name is letters, digits, and underscores, and a source file's other bytes (it is
        # read one byte to a character) belong only in strings and comments.
        self.regex = re.compile('|'.join(f'(?P<{name}>{pattern})' for name, pattern in self.rules), re.ASCII)

    def tokenize(self):
        """Tokenize, emitting INDENT/DEDENT from the indent stack. Blank lines are skipped; tabs count as one column."""
        for match in self.regex.finditer(self.source):
            kind = match.lastgroup
            value = match.group(kind)
            column = match.start() - self.line_start + 1

            if self.at_line_start and kind not in ('NEWLINE', 'SKIP', 'COMMENT'):
                self._handle_indentation(column - 1)
                self.at_line_start = False

            if kind == 'NUMBER':
                self.tokens.append(Token(TokenType.NUMBER, value, self.line, column))
            elif kind == 'IDENTIFIER':
                token_type = self.keywords.get(value, TokenType.IDENTIFIER)
                self.tokens.append(Token(token_type, value, self.line, column))
            elif kind == 'STRING':
                self.tokens.append(Token(TokenType.STRING, value, self.line, column))
            elif kind == 'BYTE':
                self.tokens.append(Token(TokenType.BYTE, value, self.line, column))
            elif kind == 'ASSIGN':
                self.tokens.append(Token(TokenType.ASSIGN, value, self.line, column))
            elif kind == 'NEWLINE':
                self.line += 1
                self.line_start = match.end()
                if self.bracket_depth == 0:
                    self.tokens.append(Token(TokenType.NEWLINE, value, self.line - 1, column))
                    self.at_line_start = True
                # Inside brackets a newline is a continuation: no NEWLINE, no indent check.
            elif kind == 'OPEN_PAREN':
                self.tokens.append(Token(TokenType.OPEN_PAREN, value, self.line, column))
                self.bracket_depth += 1
            elif kind == 'CLOSE_PAREN':
                self.tokens.append(Token(TokenType.CLOSE_PAREN, value, self.line, column))
                self.bracket_depth -= 1
            elif kind == 'OPEN_BRACKET':
                self.tokens.append(Token(TokenType.OPEN_BRACKET, value, self.line, column))
                self.bracket_depth += 1
            elif kind == 'CLOSE_BRACKET':
                self.tokens.append(Token(TokenType.CLOSE_BRACKET, value, self.line, column))
                self.bracket_depth -= 1
            elif kind == 'OPEN_BRACE':
                self.tokens.append(Token(TokenType.OPEN_BRACE, value, self.line, column))
                self.bracket_depth += 1
            elif kind == 'CLOSE_BRACE':
                self.tokens.append(Token(TokenType.CLOSE_BRACE, value, self.line, column))
                self.bracket_depth -= 1
            elif kind == 'COLON':
                self.tokens.append(Token(TokenType.COLON, value, self.line, column))
            elif kind == 'SEMICOLON':
                self.tokens.append(Token(TokenType.SEMICOLON, value, self.line, column))
            elif kind == 'COMMA':
                self.tokens.append(Token(TokenType.COMMA, value, self.line, column))
            elif kind == 'PLUS':
                self.tokens.append(Token(TokenType.PLUS, value, self.line, column))
            elif kind == 'MINUS':
                self.tokens.append(Token(TokenType.MINUS, value, self.line, column))
            elif kind == 'STAR':
                self.tokens.append(Token(TokenType.STAR, value, self.line, column))
            elif kind == 'SLASH':
                self.tokens.append(Token(TokenType.SLASH, value, self.line, column))
            elif kind == 'PERCENT':
                self.tokens.append(Token(TokenType.PERCENT, value, self.line, column))
            elif kind == 'TILDE':
                self.tokens.append(Token(TokenType.TILDE, value, self.line, column))
            elif kind == 'AMPERSAND':
                self.tokens.append(Token(TokenType.AMPERSAND, value, self.line, column))
            elif kind == 'PIPE':
                self.tokens.append(Token(TokenType.PIPE, value, self.line, column))
            elif kind == 'CARET':
                self.tokens.append(Token(TokenType.CARET, value, self.line, column))
            elif kind == 'SHIFT_LEFT':
                self.tokens.append(Token(TokenType.SHIFT_LEFT, value, self.line, column))
            elif kind == 'SHIFT_RIGHT':
                self.tokens.append(Token(TokenType.SHIFT_RIGHT, value, self.line, column))
            elif kind == 'EQUAL':
                self.tokens.append(Token(TokenType.EQUAL, value, self.line, column))
            elif kind == 'NOT_EQUAL':
                self.tokens.append(Token(TokenType.NOT_EQUAL, value, self.line, column))
            elif kind == 'GREATER_THAN':
                self.tokens.append(Token(TokenType.GREATER_THAN, value, self.line, column))
            elif kind == 'LESS_THAN':
                self.tokens.append(Token(TokenType.LESS_THAN, value, self.line, column))
            elif kind == 'GREATER_THAN_OR_EQUAL':
                self.tokens.append(Token(TokenType.GREATER_THAN_OR_EQUAL, value, self.line, column))
            elif kind == 'LESS_THAN_OR_EQUAL':
                self.tokens.append(Token(TokenType.LESS_THAN_OR_EQUAL, value, self.line, column))
            elif kind == 'PLUS_ASSIGN':
                self.tokens.append(Token(TokenType.PLUS_ASSIGN, value, self.line, column))
            elif kind == 'MINUS_ASSIGN':
                self.tokens.append(Token(TokenType.MINUS_ASSIGN, value, self.line, column))
            elif kind == 'STAR_ASSIGN':
                self.tokens.append(Token(TokenType.STAR_ASSIGN, value, self.line, column))
            elif kind == 'SLASH_ASSIGN':
                self.tokens.append(Token(TokenType.SLASH_ASSIGN, value, self.line, column))
            elif kind == 'PERCENT_ASSIGN':
                self.tokens.append(Token(TokenType.PERCENT_ASSIGN, value, self.line, column))
            elif kind == 'AMPERSAND_ASSIGN':
                self.tokens.append(Token(TokenType.AMPERSAND_ASSIGN, value, self.line, column))
            elif kind == 'PIPE_ASSIGN':
                self.tokens.append(Token(TokenType.PIPE_ASSIGN, value, self.line, column))
            elif kind == 'CARET_ASSIGN':
                self.tokens.append(Token(TokenType.CARET_ASSIGN, value, self.line, column))
            elif kind == 'SHIFT_LEFT_ASSIGN':
                self.tokens.append(Token(TokenType.SHIFT_LEFT_ASSIGN, value, self.line, column))
            elif kind == 'SHIFT_RIGHT_ASSIGN':
                self.tokens.append(Token(TokenType.SHIFT_RIGHT_ASSIGN, value, self.line, column))
            elif kind == 'DOT':
                self.tokens.append(Token(TokenType.DOT, value, self.line, column))
            elif kind == 'COMMENT':
                continue
            elif kind == 'SKIP':
                continue
            elif kind == 'UNCLOSED':
                what = 'String' if value[0] == "'" else 'Byte'
                if match.end() < len(self.source):
                    raise LexError(f"Newline in {what.lower()} literal -- a literal ends on its own line; "
                                   f"write \\n for a newline", self.file, self.line, column)
                raise LexError(f"Unterminated {what.lower()} literal", self.file, self.line, column)
            elif kind == 'MISMATCH':
                if ord(value) > 127:  # part of a character that isn't ASCII: there is no one byte to show
                    raise LexError(f"Unexpected byte 0x{ord(value):02X} -- outside strings and comments, source "
                                   f"is ASCII", self.file, self.line, column)
                raise LexError(f"Unexpected character '{value}'", self.file, self.line, column)
            else:
                raise InternalCompilerError(f'Unhandled token kind {kind!r} at line {self.line}, column {column}')

        # Synthesize a final NEWLINE before closing DEDENTs.
        if self.tokens and self.tokens[-1].type != TokenType.NEWLINE:
            col = len(self.source) - self.line_start + 1
            self.tokens.append(Token(TokenType.NEWLINE, '', self.line, col))

        while len(self.indent_stack) > 1:
            self.indent_stack.pop()
            self.tokens.append(Token(TokenType.DEDENT, '', self.line, 1))

        self.tokens.append(Token(TokenType.EOF, "", self.line, len(self.source) - self.line_start + 1))
        for tok in self.tokens:
            tok.file = self.file
        return self.tokens

    def _handle_indentation(self, width: int) -> None:
        """Emit INDENT/DEDENT to reach indentation `width`."""
        top = self.indent_stack[-1]
        if width > top:
            self.indent_stack.append(width)
            self.tokens.append(Token(TokenType.INDENT, '', self.line, 1))
        elif width < top:
            while width < self.indent_stack[-1]:
                self.indent_stack.pop()
                self.tokens.append(Token(TokenType.DEDENT, '', self.line, 1))
            if width != self.indent_stack[-1]:
                msg = "Unindent does not match any outer indentation level"
                raise LexError(msg, self.file, self.line, 0, legacy=f"{msg} at line {self.line}")


def main():
    parser = argparse.ArgumentParser(description='Lexer')
    parser.add_argument('file', type=str, help='File to lex.')
    args = parser.parse_args()
    print(lex(args.file))


_TOKEN_NAMES = {
    TokenType.IDENTIFIER: 'identifier', TokenType.NUMBER: 'number',
    TokenType.STRING: 'string literal', TokenType.BYTE: 'byte literal',
    TokenType.NEWLINE: 'end of line', TokenType.INDENT: 'indentation',
    TokenType.DEDENT: 'end of block', TokenType.EOF: 'end of input',
}


def _build_spellings() -> dict:
    lx = Lexer('')
    out = {t: f"'{word}'" for word, t in lx.keywords.items() if word != 'byte'}
    for name, pattern in lx.rules:
        t = getattr(TokenType, name, None)
        if t is not None and t not in out and t not in _TOKEN_NAMES:
            out[t] = "'" + re.sub(r'\\(.)', r'\1', pattern) + "'"
    out.update(_TOKEN_NAMES)
    return out


_SPELLINGS: dict = {}


def describe_token_type(t: TokenType) -> str:
    """Human-readable name of a token type, e.g. "':'" or "end of input"."""
    if not _SPELLINGS:
        _SPELLINGS.update(_build_spellings())
    return _SPELLINGS.get(t, t.name.lower())


def describe_token(tok: Token) -> str:
    """Human-readable token, e.g. "identifier 'x'" or "':'". A literal is shown as written, in its own quotes."""
    if tok.type in (TokenType.STRING, TokenType.BYTE):
        return f"{_TOKEN_NAMES[tok.type]} {tok.val}"
    if tok.type in (TokenType.IDENTIFIER, TokenType.NUMBER):
        return f"{_TOKEN_NAMES[tok.type]} {tok.val!r}"
    return describe_token_type(tok.type)


# A UTF-8 byte-order mark, as its three bytes read.
_BYTE_ORDER_MARK = '\xef\xbb\xbf'


def lex(filename: str) -> list:
    """The tokens of a source file. It is read as bytes, one to a character (which is what latin-1
    does), so that a string literal holds what the file holds whatever its encoding."""
    try:
        with open(filename, 'r', encoding='latin-1') as f:
            lines = f.readlines()
    except OSError as problem:
        raise file_error("read", filename, problem) from None
    if lines and lines[0].startswith(_BYTE_ORDER_MARK):
        raise LexError("This file starts with a UTF-8 byte-order mark, which Hornet source doesn't use -- save it "
                       "without one", filename, 1, 1)
    lexer = Lexer(''.join(lines), filename)
    tokens = lexer.tokenize()
    return tokens


if __name__ == '__main__':
    main()
