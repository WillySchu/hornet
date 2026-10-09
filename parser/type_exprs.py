"""Types as they are written: `int`, `[3]T`, `[]T`, `*T`, `dict[K]V`, a name, `alias.Name`.

An array's size may be an expression, and an expression may hold a type (a typed literal, an `is`
check), so this module and expressions.py need each other: each imports the other whole and calls
through it. Importing each other's functions by name would fail, whichever was loaded first."""

from typing import Union

from lexer import TokenType, describe_token
from parser import expressions
from parser.errors import ParseError
from parser.nodes import (
    ArrayTypeExpr, DictTypeExpr, PointerTypeExpr, QualifiedTypeExpr, SliceTypeExpr,
)
from parser.stream import TokenStream


def check_starts_with_return_type(p: TokenStream) -> bool:
    """Whether a return type precedes the def's name."""
    return p.check(
        TokenType.INT,
        TokenType.INT8,
        TokenType.UINT8,
        TokenType.INT64,
        TokenType.INT32,
        TokenType.BOOL,
        TokenType.STR,
        TokenType.OPEN_BRACKET,
        TokenType.STAR,
        TokenType.DICT
    ) or (
            p.check(TokenType.IDENTIFIER) and p.peek(1).type == TokenType.IDENTIFIER
    ) or (
            p.check(TokenType.IDENTIFIER) and p.peek(1).type == TokenType.DOT
            and p.peek(2).type == TokenType.IDENTIFIER and p.peek(3).type == TokenType.IDENTIFIER
    )


def parse_return_type(p: TokenStream):
    """What precedes a def's or extern's name: a type, `never` (it doesn't return), or nothing."""
    if p.match(TokenType.NEVER):
        return 'never'
    return parse_type(p) if check_starts_with_return_type(p) else None


def parse_type(p: TokenStream) -> Union[str, ArrayTypeExpr, SliceTypeExpr, PointerTypeExpr, DictTypeExpr]:
    if p.check(TokenType.DICT):
        dict_tok = p.advance()
        p.expect(TokenType.OPEN_BRACKET, "Expected '[' after 'dict'")
        key_type = parse_type(p)
        p.expect(TokenType.CLOSE_BRACKET, "Expected ']' after a dict's own key type")
        value_type = parse_type(p)
        return DictTypeExpr(key_type=key_type, value_type=value_type, line=dict_tok.line, col=dict_tok.col)
    if p.check(TokenType.STAR):
        star_tok = p.advance()
        pointee_type = parse_type(p)
        return PointerTypeExpr(pointee_type=pointee_type, line=star_tok.line, col=star_tok.col)
    if p.check(TokenType.OPEN_BRACKET):
        open_tok = p.advance()
        if p.check(TokenType.CLOSE_BRACKET):
            p.advance()
            element_type = parse_type(p)
            return SliceTypeExpr(element_type=element_type, line=open_tok.line, col=open_tok.col)
        if p.check(TokenType.NUMBER) and p.peek(1).type == TokenType.CLOSE_BRACKET:
            size_tok = p.advance()
            if '.' in size_tok.val:
                raise p.error(f"Array size must be a whole number, got '{size_tok.val}'", size_tok)
            size = int(size_tok.val)
            if size <= 0:
                raise p.error(f"Array size must be positive, got {size}", size_tok)
        else:
            # A constant expression, resolved during semantic analysis.
            start = p.pos
            try:
                size = expressions.parse_expression(p)
            except ParseError:
                if p.pos != start:
                    raise
                raise p.error("Expected an array size (a positive integer or constant expression), "
                                  "or ']' for a slice type", p.current())
        p.expect(TokenType.CLOSE_BRACKET, "Expected ']' after array size")
        element_type = parse_type(p)
        return ArrayTypeExpr(size=size, element_type=element_type, line=open_tok.line, col=open_tok.col)
    if p.check(
            TokenType.INT,
            TokenType.INT8,
            TokenType.UINT8,
            TokenType.INT64,
            TokenType.INT32,
            TokenType.BOOL,
            TokenType.STR
    ):
        return p.advance().val
    if p.check(TokenType.IDENTIFIER):
        # Unvalidated here; semantic analysis resolves type names.
        name_tok = p.advance()
        if p.check(TokenType.DOT):
            p.advance()
            qualified_name_tok = p.expect(TokenType.IDENTIFIER, "Expected a type name after '.'")
            return QualifiedTypeExpr(
                module=name_tok.val, name=qualified_name_tok.val, line=name_tok.line, col=name_tok.col)
        return name_tok.val
    tok = p.current()
    raise p.error(
        f"Expected a type ('int', 'int8', 'uint8', 'int64', 'bool', 'str', a "
        f"struct name, '[size]type', or '[]type'), got {describe_token(tok)}",
        tok,
    )


def parse_qualifiable_type_name(
        p: TokenStream,
        expected_message: str) -> Union[str, QualifiedTypeExpr, ArrayTypeExpr, SliceTypeExpr, PointerTypeExpr]:
    """A possibly qualified type name, builtin type keyword, `none`, or array/slice/pointer/dict type."""
    if p.check(TokenType.STAR, TokenType.OPEN_BRACKET, TokenType.DICT):
        return parse_type(p)
    if p.check(
            TokenType.INT,
            TokenType.INT8,
            TokenType.UINT8,
            TokenType.INT64,
            TokenType.INT32,
            TokenType.BOOL,
            TokenType.STR,
            TokenType.NONE
    ):  # `none`: a sum type's payload-free variant
        return p.advance().val
    name_tok = p.expect(TokenType.IDENTIFIER, f"Expected {expected_message}")
    if p.check(TokenType.DOT):
        p.advance()
        qualified_name_tok = p.expect(TokenType.IDENTIFIER, "Expected a type name after '.'")
        return QualifiedTypeExpr(
            module=name_tok.val, name=qualified_name_tok.val, line=name_tok.line, col=name_tok.col)
    return name_tok.val
