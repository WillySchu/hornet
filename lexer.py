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

    # Special
    NEWLINE = auto()
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
            'and': TokenType.AND,
            'or': TokenType.OR,
            'not': TokenType.NOT,
            'bool': TokenType.BOOL,
            'true': TokenType.TRUE,
            'false': TokenType.FALSE,
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
            elif kind == 'ASSIGN':
                self.tokens.append(Token(TokenType.ASSIGN, value, self.line, column))
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
