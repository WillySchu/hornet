"""Lexer"""

import argparse
import re

from enum import auto, Enum


class TokenType(Enum):
    # Literals
    NUMBER = auto()
    IDENTIFIER = auto()

    # Punctuation
    OPEN_PAREN = auto()
    CLOSE_PAREN = auto()
    COLON = auto()

    # Operators
    ASSIGN = auto()
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    TILDE = auto()
    BANG = auto()
    AND = auto()
    OR = auto()
    EQUAL = auto()
    NOT_EQUAL = auto()
    LESS_THAN = auto()
    GREATER_THAN = auto()
    LESS_THAN_OR_EQUAL = auto()
    GREATER_THAN_OR_EQUAL = auto()

    # Keywords
    DEF = auto()
    INT = auto()
    RETURN = auto()
    NOT = auto()
    BOOL = auto()
    TRUE = auto()
    FALSE = auto()
    IF = auto()
    ELSE = auto()
    ELIF = auto()

    # Special
    NEWLINE = auto()
    INDENT = auto()
    DEDENT = auto()
    MISMATCH = auto()
    EOF = auto()


class Token():
    def __init__(self, t: TokenType, val: str, line: int, col: int):
        self.type = t
        self.val = val
        self.line = line
        self.col = col

    def __eq__(self, other) -> bool:
        if not isinstance(other, Token):
            return False
        return other.__dict__ == self.__dict__

    def __str__(self) -> str:
        return f'Token(type={self.type}, val={self.val})'

    def __repr__(self) -> str:
        return self.__str__()


class Lexer():
    """Tokenizes Hornet source, including synthesizing INDENT/DEDENT
    tokens for block structure (see tokenize()'s docstring).
    """

    def __init__(self, source: str):
        self.source = source
        self.tokens = []
        self.line = 1
        self.line_start = 0

        # Indentation tracking -- see tokenize() for how these are used.
        # indent_stack always starts at [0] (top-level code is
        # unindented); at_line_start tracks whether the next real token
        # we see will be the first one on its logical line, which is
        # the only time indentation actually gets measured.
        self.indent_stack = [0]
        self.at_line_start = True

        # Define keywords mapping
        self.keywords = {
            'def': TokenType.DEF,
            'int': TokenType.INT,
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
        }

        # Compile master regex pattern
        self.rules = [
            # Multi character.
            ('NUMBER',      r'\d+(\.\d+)?'),     # Integer or decimal
            ('IDENTIFIER',  r'[a-zA-Z_]\w*'),    # Variable names/keywords

            # Double character
            ('EQUAL',                 r'=='),    # Equal
            ('NOT_EQUAL',             r'!='),    # Not equal
            ('GREATER_THAN_OR_EQUAL', r'>='),    # Greater than or equal
            ('LESS_THAN_OR_EQUAL',    r'<='),

            # Single character
            ('NEWLINE',      r'\n'),              # Line breaks
            ('OPEN_PAREN',   r'\('),              # Open paren
            ('CLOSE_PAREN',  r'\)'),              # Close paren
            ('GREATER_THAN', r'>'),
            ('LESS_THAN',    r'<'),
            ('COLON',        r':'),               # Colon
            ('ASSIGN',       r'='),                # Assignment operator
            ('PLUS',         r'\+'),              # Add
            ('MINUS',        r'\-'),              # Subtract
            ('STAR',         r'\*'),              # Multiply
            ('SLASH',        r'/'),               # Divide
            ('TILDE',        r'\~'),              # Tilde
            ('BANG',         r'\!'),              # Bang
            ('SKIP',         r'[ \t\r]+'),        # Spaces and tabs
            ('MISMATCH',     r'.'),               # Any other character (error)
        ]

        # Combine rules into a single regex string
        self.regex = re.compile('|'.join(f'(?P<{name}>{pattern})' for name, pattern in self.rules))

    def tokenize(self):
        """Tokenizes the source, including block structure.

        Previously this language had no INDENT/DEDENT tokens at all --
        the SKIP rule below just swallowed all whitespace, including
        leading indentation, uniformly. That was fine as long as every
        block was flat (a function body with no nested if/while), since
        the parser could get away with "a block ends at the next 'def'
        or EOF". It cannot work for nested blocks: once an `if` can
        appear inside a function body, "the next def or EOF" no longer
        tells you where the if's own body ends versus where an `elif`/
        `else` begins, or where control returns to the enclosing block.

        The fix is the classic Python-style approach: track an
        indentation stack, and synthesize INDENT/DEDENT tokens whenever
        a new logical line's leading whitespace goes deeper or
        shallower than what's currently open. The parser then treats a
        block as `INDENT statement+ DEDENT` -- a signal that
        generalizes to any nesting depth, unlike scanning for `def`.

        The key trick for finding where a logical line's real content
        starts: rather than specifically inspecting the SKIP match at
        the front of each line, this waits for the first match that
        ISN'T itself SKIP or NEWLINE while `at_line_start` is true, and
        computes that token's column directly. That sidesteps having to
        special-case blank lines (lines that are only whitespace, or
        entirely empty) -- a blank line never produces such a match, so
        `at_line_start` just stays true across it, and the next genuinely
        content-bearing line is what actually gets measured. Comparing
        tabs and spaces isn't handled specially; each whitespace
        character just counts as one column of indentation, which is a
        simplification worth knowing about if you ever mix the two.
        """
        for match in self.regex.finditer(self.source):
            kind = match.lastgroup
            value = match.group(kind)
            column = match.start() - self.line_start + 1

            if self.at_line_start and kind not in ('NEWLINE', 'SKIP'):
                self._handle_indentation(column - 1)
                self.at_line_start = False

            if kind == 'NUMBER':
                self.tokens.append(Token(TokenType.NUMBER, value, self.line, column))
            elif kind == 'IDENTIFIER':
                # Check if the identifier is actually a reserved keyword
                token_type = self.keywords.get(value, TokenType.IDENTIFIER)
                self.tokens.append(Token(token_type, value, self.line, column))
            elif kind == 'ASSIGN':
                self.tokens.append(Token(TokenType.ASSIGN, value, self.line, column))
            elif kind == 'NEWLINE':
                self.tokens.append(Token(TokenType.NEWLINE, value, self.line, column))
                self.line += 1
                self.line_start = match.end()
                self.at_line_start = True
            elif kind == 'OPEN_PAREN':
                self.tokens.append(Token(TokenType.OPEN_PAREN, value, self.line, column))
            elif kind == 'CLOSE_PAREN':
                self.tokens.append(Token(TokenType.CLOSE_PAREN, value, self.line, column))
            elif kind == 'COLON':
                self.tokens.append(Token(TokenType.COLON, value, self.line, column))
            elif kind == 'PLUS':
                self.tokens.append(Token(TokenType.PLUS, value, self.line, column))
            elif kind == 'MINUS':
                self.tokens.append(Token(TokenType.MINUS, value, self.line, column))
            elif kind == 'STAR':
                self.tokens.append(Token(TokenType.STAR, value, self.line, column))
            elif kind == 'SLASH':
                self.tokens.append(Token(TokenType.SLASH, value, self.line, column))
            elif kind == 'TILDE':
                self.tokens.append(Token(TokenType.TILDE, value, self.line, column))
            elif kind == 'BANG':
                self.tokens.append(Token(TokenType.BANG, value, self.line, column))
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
            elif kind == 'SKIP':
                continue
            elif kind == 'MISMATCH':
                raise SyntaxError(
                    f"Unexpected character '{value}' at line {self.line}, column {column}")
            else:
                raise RuntimeError('This should be impossible.')

        # If the source didn't end with a newline, synthesize one before
        # closing out indentation -- keeps the final logical line's
        # shape consistent with every other line, whose NEWLINE arrives
        # before any DEDENTs that follow it.
        if self.tokens and self.tokens[-1].type != TokenType.NEWLINE:
            col = len(self.source) - self.line_start + 1
            self.tokens.append(Token(TokenType.NEWLINE, '', self.line, col))

        # Unwind any indentation still open at EOF (e.g. a file that
        # ends inside an if-block, with no trailing dedent to close it).
        while len(self.indent_stack) > 1:
            self.indent_stack.pop()
            self.tokens.append(Token(TokenType.DEDENT, '', self.line, 1))

        # Append End-Of-File token
        self.tokens.append(Token(TokenType.EOF, "", self.line, len(self.source) - self.line_start + 1))
        return self.tokens

    def _handle_indentation(self, width: int) -> None:
        """Compares `width` (the new logical line's indentation, in
        characters) against the current indentation stack, emitting
        INDENT/DEDENT tokens to reconcile the difference."""
        top = self.indent_stack[-1]
        if width > top:
            self.indent_stack.append(width)
            self.tokens.append(Token(TokenType.INDENT, '', self.line, 1))
        elif width < top:
            while width < self.indent_stack[-1]:
                self.indent_stack.pop()
                self.tokens.append(Token(TokenType.DEDENT, '', self.line, 1))
            if width != self.indent_stack[-1]:
                raise SyntaxError(
                    f"Unindent does not match any outer indentation "
                    f"level at line {self.line}"
                )


def main():
    parser = argparse.ArgumentParser(description='Lexer')
    parser.add_argument('file', type=str, help='File to lex.')
    args = parser.parse_args()
    print(lex(args.file))


def lex(filename: str) -> list:
    with open(filename, 'r') as f:
        lines = f.readlines()
    lexer = Lexer(''.join(lines))
    tokens = lexer.tokenize()
    return tokens


if __name__ == '__main__':
    main()
