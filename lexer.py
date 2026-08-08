"""Lexer"""

import argparse
import re

from enum import auto, Enum


class TokenType(Enum):
    # Literals
    NUMBER = auto()
    IDENTIFIER = auto()

    # TODO(will): What's a good categorie name here?
    OPEN_PAREN = auto()
    CLOSE_PAREN = auto()
    NEWLINE = auto()
    COLON = auto()
    EOF = auto()

    # Operators
    MINUS = auto()
    TILDE = auto()
    BANG = auto()

    # Keywords
    DEF = auto()
    INT = auto()
    RETURN = auto()

    # Special
    MISMATCH = auto()

    
class Token():
    def __init__(self, t: TokenType, val: str, line: int, col: int):
        self.type = t
        self.val = val
        self.line = line
        self.col = col

    def __str__(self) -> str:
        return f'Token(type={self.type}, val={self.val})'

    def __repr__(self) -> str:
        return self.__str__()


class Lexer():
    def __init__(self, source: str):
        self.source = source
        self.tokens = []
        self.line = 1
        self.line_start = 0

        # Define keywords mapping
        self.keywords = {
            'def': TokenType.DEF,
            'int': TokenType.INT,
            'return': TokenType.RETURN,
        }

        # Compile master regex pattern
        self.rules = [
            ('NUMBER',      r'\d+(\.\d+)?'),     # Integer or decimal
            ('IDENTIFIER',  r'[a-zA-Z_]\w*'),    # Variable names/keywords
            ('NEWLINE',     r'\n'),              # Line breaks
            ('OPEN_PAREN',  r'\('),              # Open paren
            ('CLOSE_PAREN', r'\)'),              # Close paren
            ('COLON',       r':'),               # Colon
            #('ASSIGN',     r'='),               # Assignment operator
            #('PLUS',       r'\+'),              # Add
            ('MINUS',        r'\-'),             # Subtract
            ('TILDE',       r'\~'),              # Tilde
            ('BANG',        r'\!'),              # Bang
            ('SKIP',        r'[ \t\r]+'),        # Spaces and tabs
            ('MISMATCH',    r'.'),               # Any other character (error)
        ]

        # Combine rules into a single regex string
        self.regex = re.compile('|'.join(f'(?P<{name}>{pattern})' for name, pattern in self.rules))

    def tokenize(self):
        for match in self.regex.finditer(self.source):
            kind = match.lastgroup
            value = match.group(kind)
            column = match.start() - self.line_start + 1

            if kind == 'NUMBER':
                self.tokens.append(Token(TokenType.NUMBER, value, self.line, column))
            elif kind == 'IDENTIFIER':
                # Check if the identifier is actually a reserved keyword
                token_type = self.keywords.get(value, TokenType.IDENTIFIER)
                self.tokens.append(Token(token_type, value, self.line, column))
            #elif kind == 'ASSIGN':
            #    self.tokens.append(Token(TokenType.ASSIGN, value, self.line, column))
            #elif kind == 'PLUS':
            #    self.tokens.append(Token(TokenType.PLUS, value, self.line, column))
            #elif kind == 'MINUS':
            #    self.tokens.append(Token(TokenType.MINUS, value, self.line, column))
            elif kind == 'NEWLINE':
                self.tokens.append(Token(TokenType.NEWLINE, value, self.line, column))
                self.line += 1
                self.line_start = match.end()
            elif kind == 'OPEN_PAREN':
                self.tokens.append(Token(TokenType.OPEN_PAREN, value, self.line, column))
            elif kind == 'CLOSE_PAREN':
                self.tokens.append(Token(TokenType.CLOSE_PAREN, value, self.line, column))
            elif kind == 'COLON':
                self.tokens.append(Token(TokenType.COLON, value, self.line, column))
            elif kind == 'MINUS':
                self.tokens.append(Token(TokenType.MINUS, value, self.line, column))
            elif kind == 'TILDE':
                self.tokens.append(Token(TokenType.TILDE, value, self.line, column))
            elif kind == 'BANG':
                self.tokens.append(Token(TokenType.BANG, value, self.line, column))
            elif kind == 'SKIP':
                continue
            elif kind == 'MISMATCH':
                raise SyntaxError(
                    f"Unexpected character '{value}' at line {self.line}, column {column}")
            else:
                raise RuntimeError('This should be impossible.')

        # Append End-Of-File token
        self.tokens.append(Token(TokenType.EOF, "", self.line, len(self.source) - self.line_start + 1))
        return self.tokens


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
