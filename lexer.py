"""Lexer"""

import argparse
import re

from enum import auto, Enum


class TokenType(Enum):
    # Literals
    NUMBER = auto()
    IDENTIFIER = auto()
    STRING = auto()  # a string *literal*, e.g. 'hello' -- distinct from
                      # STR below (the 'str' *type keyword*). Reusing one
                      # token type for both would make it impossible for
                      # the parser to tell "the word str" apart from "an
                      # actual string value" by type alone.

    # Punctuation
    OPEN_PAREN = auto()
    CLOSE_PAREN = auto()
    OPEN_BRACKET = auto()
    CLOSE_BRACKET = auto()
    COLON = auto()
    COMMA = auto()
    DOT = auto()

    # Operators
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

    # Compound assignment
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

    # Keywords
    DEF = auto()
    INT = auto()
    INT8 = auto()
    UINT8 = auto()
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
    # FOR = auto()
    WHILE = auto()
    BREAK = auto()
    CONTINUE = auto()
    NONE = auto()
    STRUCT = auto()
    TYPE = auto()

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
            'int8': TokenType.INT8,
            'uint8': TokenType.UINT8,
            # 'byte' is deliberately mapped to the SAME TokenType as
            # 'uint8', not a new TokenType.BYTE.
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
            # 'for': TokenType.FOR,
            'while': TokenType.WHILE,
            'break': TokenType.BREAK,
            'continue': TokenType.CONTINUE,
            'none': TokenType.NONE,
            'struct': TokenType.STRUCT,
            'type': TokenType.TYPE,
        }

        # Compile master regex pattern
        self.rules = [
            # Multi character.
            ('NUMBER',      r'\d+(\.\d+)?'),     # Integer or decimal
            ('IDENTIFIER',  r'[a-zA-Z_]\w*'),    # Variable names/keywords
            ('STRING',      r"'([^'\\]|\\.)*'"), # String literals

            # Triple character -- must come before the double- and
            # single-character '<'/'>' rules below (SHIFT_LEFT,
            # SHIFT_RIGHT, LESS_THAN, GREATER_THAN, LESS_THAN_OR_EQUAL,
            # GREATER_THAN_OR_EQUAL), or those would each greedily
            # consume a prefix of '<<=' / '>>=' one or two characters at
            # a time and never let the full three-character token match.
            ('SHIFT_LEFT_ASSIGN',  r'<<='),
            ('SHIFT_RIGHT_ASSIGN', r'>>='),

            # Double character
            ('EQUAL',                 r'=='),    # Equal
            ('NOT_EQUAL',             r'!='),    # Not equal
            ('GREATER_THAN_OR_EQUAL', r'>='),    # Greater than or equal
            ('LESS_THAN_OR_EQUAL',    r'<='),
            ('SHIFT_LEFT',            r'<<'),    # Bitwise shift left -- must come before
            ('SHIFT_RIGHT',           r'>>'),    # LESS_THAN/GREATER_THAN below, or those
                                                  # single-char rules would consume one '<'/'>'
                                                  # at a time and never let this match.
            ('PLUS_ASSIGN',      r'\+='),        # Compound assignment -- each of these must
            ('MINUS_ASSIGN',     r'\-='),        # come before its corresponding single-character
            ('STAR_ASSIGN',      r'\*='),        # operator rule below, for the same greedy-
            ('SLASH_ASSIGN',     r'/='),         # single-char-match-first reason as SHIFT_LEFT/
            ('PERCENT_ASSIGN',   r'%='),         # SHIFT_RIGHT above.
            ('AMPERSAND_ASSIGN', r'&='),
            ('PIPE_ASSIGN',      r'\|='),
            ('CARET_ASSIGN',     r'\^='),

            # Single character
            ('NEWLINE',       r'\n'),              # Line breaks
            ('OPEN_PAREN',    r'\('),              # Open paren
            ('CLOSE_PAREN',   r'\)'),              # Close paren
            ('OPEN_BRACKET',  r'\['),              # Array type/literal/index open
            ('CLOSE_BRACKET', r'\]'),              # Array type/literal/index close
            ('GREATER_THAN',  r'>'),
            ('LESS_THAN',     r'<'),
            ('COLON',         r':'),               # Colon
            ('COMMA',         r','),
            ('ASSIGN',        r'='),               # Assignment operator
            ('PLUS',          r'\+'),              # Add
            ('MINUS',         r'\-'),              # Subtract
            ('STAR',          r'\*'),              # Multiply
            ('SLASH',         r'/'),               # Divide
            ('PERCENT',       r'%'),               # Modulo
            ('TILDE',         r'\~'),              # Tilde
            ('AMPERSAND',     r'&'),               # Bitwise AND
            ('PIPE',          r'\|'),              # Bitwise OR
            ('CARET',         r'\^'),              # Bitwise XOR
            ('DOT',           r'\.'),

            ('COMMENT',       r'#[^\n]*'),         # Single-line comment -- from '#' to
                                                    # end of line, NOT including the
                                                    # newline itself, so the newline
                                                    # still gets tokenized normally right
                                                    # after and statement-termination
                                                    # logic doesn't need to know comments
                                                    # exist at all. Placed here, next to
                                                    # SKIP, since both are discarded
                                                    # rather than producing a real token
                                                    # -- see tokenize()'s own handling of
                                                    # both, and _handle_indentation's
                                                    # exclusion of both from what counts
                                                    # as a line's first real content.
            ('SKIP',          r'[ \t\r]+'),        # Spaces and tabs
            ('MISMATCH',      r'.'),               # Any other character (error)
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

        NOTE: a STRING literal that spans a genuine embedded newline
        (an actual newline character typed between the quotes, not the
        two-character `\\n` escape) is matched as a single token here,
        since the STRING rule is tried -- and consumes as far as it
        matches -- before NEWLINE gets a chance to. That means such a
        newline never increments self.line, so line numbers reported in
        errors after a multi-line string literal can drift. This is a
        known, narrow edge case, not something worth the extra
        bookkeeping to fix right now.
        """
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
                # Check if the identifier is actually a reserved keyword
                token_type = self.keywords.get(value, TokenType.IDENTIFIER)
                self.tokens.append(Token(token_type, value, self.line, column))
            elif kind == 'STRING':
                self.tokens.append(Token(TokenType.STRING, value, self.line, column))
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
            elif kind == 'OPEN_BRACKET':
                self.tokens.append(Token(TokenType.OPEN_BRACKET, value, self.line, column))
            elif kind == 'CLOSE_BRACKET':
                self.tokens.append(Token(TokenType.CLOSE_BRACKET, value, self.line, column))
            elif kind == 'COLON':
                self.tokens.append(Token(TokenType.COLON, value, self.line, column))
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
            elif kind == 'MISMATCH':
                raise SyntaxError(
                    f"Unexpected character '{value}' at line {self.line}, column {column}")
            else:
                raise RuntimeError(f'Unhandled character "{value}" at line {self.line}, column {column}')

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
