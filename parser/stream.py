"""The token stream: a file's tokens, and where the parser is in them.

It is all the state parsing has. The grammar's functions each take one and move it along; nothing
else is carried between them."""

from dataclasses import dataclass, field
from typing import List, Optional

from lexer import Token, TokenType, describe_token, describe_token_type
from parser.errors import ParseError
from parser.escapes import BadEscape, unescape_quoted_literal


@dataclass
class TokenStream:
    tokens: List[Token]
    pos: int = 0
    # Token positions where a typed literal's tokens could also be an indexed literal times an index
    # (expressions: _looks_like_typed_literal). Kept here because it is keyed by position.
    could_be_multiplication: set = field(default_factory=set)

    def __post_init__(self):
        if len(self.tokens) == 0:
            raise ValueError('tokens must have non zero length')
        if self.tokens[-1].type != TokenType.EOF:
            raise ValueError('tokens must be terminated by an EOF')

    # -- looking

    def peek(self, offset: int = 0) -> Token:
        idx = min(self.pos + offset, len(self.tokens) - 1)
        return self.tokens[idx]

    def current(self) -> Token:
        return self.peek()

    def previous(self) -> Optional[Token]:
        """The token just passed; None at the start."""
        return self.tokens[self.pos - 1] if self.pos else None

    def at_end(self) -> bool:
        return self.current().type == TokenType.EOF

    def check(self, *types: TokenType) -> bool:
        return not self.at_end() and self.current().type in types

    # -- moving

    def advance(self) -> Token:
        tok = self.current()
        if not self.at_end():
            self.pos += 1
        return tok

    def match(self, *types: TokenType) -> bool:
        if self.check(*types):
            self.advance()
            return True
        return False

    def expect(self, type_: TokenType, message: str = None) -> Token:
        if self.check(type_):
            return self.advance()
        tok = self.current()
        msg = message or f"Expected {describe_token_type(type_)}, got {describe_token(tok)}"
        raise self.error(msg, tok)

    def skip_newlines(self) -> None:
        while self.match(TokenType.NEWLINE):
            pass

    def expect_statement_end(self) -> None:
        """A statement ends its line: a block statement has consumed its block (DEDENT), a simple
        one must be followed by a newline, the block's end, or the end of input."""
        before = self.previous()
        if (before is not None and before.type in (TokenType.DEDENT, TokenType.NEWLINE)) or self.at_end():
            return
        if not self.check(TokenType.NEWLINE, TokenType.DEDENT):
            tok = self.current()
            raise self.error(f"Expected the end of the line after this statement, got {describe_token(tok)}", tok)

    # -- trying something and taking it back

    def mark(self) -> int:
        """Where the stream is, for reset()."""
        return self.pos

    def reset(self, mark: int) -> None:
        """Go back to `mark`: what was parsed since is to be read some other way. The position is all
        there is to restore. could_be_multiplication is keyed by position, so what a first reading
        noted is still true of the same tokens; if the stream ever holds anything a reading can
        change that isn't, it must be restored here."""
        self.pos = mark

    # -- tokens to text, and errors at them

    def error(self, message: str, tok: Token) -> ParseError:
        return ParseError(message, tok.file, tok.line, tok.col)

    def literal_text(self, tok: Token) -> str:
        """What a string or byte literal's token stands for, its escapes resolved."""
        try:
            return unescape_quoted_literal(tok.val)
        except BadEscape as bad:
            raise ParseError(bad.message, tok.file, tok.line, tok.col + bad.offset) from None
