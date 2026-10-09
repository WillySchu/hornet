"""Expressions: operators by precedence climbing, then what binds tighter (unary, postfix, primary),
with literals, calls, and casts.

(This module and type_exprs.py need each other, and each imports the other whole: see there.)"""

from dataclasses import dataclass
from enum import auto, Enum
from typing import List, Optional, Tuple, Union

from diagnostics import quoted_text
from lexer import TokenType, describe_token
from ops import BinaryOp, UnaryOp
from parser import type_exprs
from parser.errors import ParseError
from parser.nodes import (
    ArrayLiteral, ArrayTypeExpr, Binary, BoolLiteral, ByteLiteral, Call, Cast, Constant, DictLiteral, DictTypeExpr,
    Field, Index, IsCheck, Node, NoneLiteral, Slice, SliceLiteral, SliceTypeExpr, StringLiteral, Unary, Variable,
)
from parser.stream import TokenStream

# Added to an error in the type of a literal that ArrayLiteral.could_be_multiplication marks.
READ_AS_A_TYPED_LITERAL = (
    " -- this is read as a typed array literal, like `[2][1]*P[...]`; to multiply an indexed literal by a "
    "value, write that literal's type, as in `[1]int[x][0] * ys[1]`")


# Prefix operators that bind tighter than every binary operator; `not` is handled in parse_unary.
_UNARY_OPS = {
    TokenType.MINUS: UnaryOp.NEGATE,
    TokenType.TILDE: UnaryOp.COMPLEMENT,
    TokenType.AMPERSAND: UnaryOp.ADDRESS_OF,
    TokenType.STAR: UnaryOp.DEREFERENCE,
}


class Associativity(Enum):
    LEFT = auto()
    RIGHT = auto()


@dataclass(frozen=True)
class OperatorInfo:
    op: BinaryOp
    precedence: int
    associativity: Associativity


# TokenType -> (BinaryOp, precedence, associativity).
_BINARY_OPS = {
    TokenType.STAR:    OperatorInfo(BinaryOp.MULTIPLY, precedence=10, associativity=Associativity.LEFT),
    TokenType.SLASH:   OperatorInfo(BinaryOp.DIVIDE,   precedence=10, associativity=Associativity.LEFT),
    TokenType.PERCENT: OperatorInfo(BinaryOp.MODULO,   precedence=10, associativity=Associativity.LEFT),

    TokenType.PLUS:  OperatorInfo(BinaryOp.ADD,      precedence=9, associativity=Associativity.LEFT),
    TokenType.MINUS: OperatorInfo(BinaryOp.SUBTRACT, precedence=9, associativity=Associativity.LEFT),

    TokenType.SHIFT_LEFT:  OperatorInfo(BinaryOp.SHIFT_LEFT,  precedence=8, associativity=Associativity.LEFT),
    TokenType.SHIFT_RIGHT: OperatorInfo(BinaryOp.SHIFT_RIGHT, precedence=8, associativity=Associativity.LEFT),

    TokenType.LESS_THAN:             OperatorInfo(
        BinaryOp.LESS_THAN,             precedence=7, associativity=Associativity.LEFT),
    TokenType.GREATER_THAN:          OperatorInfo(
        BinaryOp.GREATER_THAN,          precedence=7, associativity=Associativity.LEFT),
    TokenType.LESS_THAN_OR_EQUAL:    OperatorInfo(
        BinaryOp.LESS_THAN_OR_EQUAL,    precedence=7, associativity=Associativity.LEFT),
    TokenType.GREATER_THAN_OR_EQUAL: OperatorInfo(
        BinaryOp.GREATER_THAN_OR_EQUAL, precedence=7, associativity=Associativity.LEFT),

    TokenType.EQUAL:     OperatorInfo(BinaryOp.EQUAL,     precedence=6, associativity=Associativity.LEFT),
    TokenType.NOT_EQUAL: OperatorInfo(BinaryOp.NOT_EQUAL, precedence=6, associativity=Associativity.LEFT),
    TokenType.IN:         OperatorInfo(BinaryOp.IN,        precedence=6, associativity=Associativity.LEFT),

    TokenType.AMPERSAND: OperatorInfo(BinaryOp.BITWISE_AND, precedence=5, associativity=Associativity.LEFT),
    TokenType.CARET:     OperatorInfo(BinaryOp.BITWISE_XOR, precedence=4, associativity=Associativity.LEFT),
    TokenType.PIPE:      OperatorInfo(BinaryOp.BITWISE_OR,  precedence=3, associativity=Associativity.LEFT),

    TokenType.AND: OperatorInfo(BinaryOp.AND, precedence=2, associativity=Associativity.LEFT),

    TokenType.OR: OperatorInfo(BinaryOp.OR, precedence=1, associativity=Associativity.LEFT),
}


# `is` binds like `==` and `in`.
_IS_PRECEDENCE = _BINARY_OPS[TokenType.EQUAL].precedence


# `not` binds looser than every binary operator except `and` and `or`: `not a < b` is `not (a < b)`.
_NOT_OPERAND_PRECEDENCE = _BINARY_OPS[TokenType.AND].precedence + 1


_TYPE_START_TOKENS = (
    TokenType.INT,
    TokenType.INT8,
    TokenType.UINT8,
    TokenType.INT64,
    TokenType.INT32,
    TokenType.BOOL,
    TokenType.STR,
    TokenType.IDENTIFIER,
    TokenType.STAR,
    TokenType.DICT,
)


def parse_expression(p: TokenStream) -> Node:
    return parse_binary(p)


def parse_binary(p: TokenStream, min_prec: int = 0) -> Node:
    """Precedence climbing over _BINARY_OPS."""
    left = parse_unary(p)
    while True:
        if p.check(TokenType.NOT) and p.peek(1).type == TokenType.IN:
            in_info = _BINARY_OPS[TokenType.IN]
            if in_info.precedence < min_prec:
                break
            not_tok = p.advance()
            p.advance()
            right = parse_binary(p, in_info.precedence + 1)
            membership = Binary(op=BinaryOp.IN, left=left, right=right, line=left.line, col=left.col)
            left = Unary(op=UnaryOp.NOT, operand=membership, line=not_tok.line, col=not_tok.col)
            continue
        if p.check(TokenType.IS):  # `left is T`, `left is not T`: T is a type, not an expression
            if _IS_PRECEDENCE < min_prec:
                break
            is_tok = p.advance()
            negated = p.match(TokenType.NOT)
            type_name = type_exprs.parse_qualifiable_type_name(p, "a type name after 'is'")
            at = left if isinstance(left, Variable) else is_tok
            if p.check(TokenType.AS):  # `left is T as NAME`: NAME is a copy of left
                if negated:
                    raise p.error("'as NAME' can't follow 'is not': there would be nothing of that type "
                                      "to bind", p.current())
                p.advance()
                binding_tok = p.expect(TokenType.IDENTIFIER, "Expected a binding name after 'as'")
                left = IsCheck(variable_name=binding_tok.val, type_name=type_name, subject=left,
                               line=at.line, col=at.col)
            elif isinstance(left, Variable):
                left = IsCheck(variable_name=left.name, type_name=type_name, line=at.line, col=at.col)
            else:
                left = IsCheck(variable_name=None, type_name=type_name, subject=left, line=at.line, col=at.col)
            if negated:
                left = Unary(op=UnaryOp.NOT, operand=left, line=left.line, col=left.col)
            continue
        op_info = _BINARY_OPS.get(p.current().type)
        if op_info is None or op_info.precedence < min_prec:
            break
        p.advance()
        next_min_prec = (
            op_info.precedence + 1
            if op_info.associativity == Associativity.LEFT
            else op_info.precedence
        )
        right = parse_binary(p, next_min_prec)
        left = Binary(op=op_info.op, left=left, right=right, line=left.line, col=left.col)
    return left


def parse_unary(p: TokenStream) -> Node:
    if p.check(TokenType.NOT):
        op_tok = p.advance()
        operand = parse_binary(p, _NOT_OPERAND_PRECEDENCE)
        return Unary(op=UnaryOp.NOT, operand=operand, line=op_tok.line, col=op_tok.col)
    if p.check(*_UNARY_OPS):
        op_tok = p.advance()
        # Recurse on parse_unary so prefix operators chain.
        operand = parse_unary(p)
        return Unary(op=_UNARY_OPS[op_tok.type], operand=operand, line=op_tok.line, col=op_tok.col)
    return parse_postfix(p)


def parse_postfix(p: TokenStream) -> Node:
    """Apply `[...]`, `.name`, `.name(...)` suffixes."""
    expr = parse_primary(p)
    while p.check(TokenType.OPEN_BRACKET, TokenType.DOT):
        if p.match(TokenType.DOT):
            name_tok = p.expect(TokenType.IDENTIFIER, "Expected a field name after '.'")
            if p.match(TokenType.OPEN_PAREN):
                args, kwargs = parse_receiver_call_args(p)
                p.expect(TokenType.CLOSE_PAREN, "Expected ')' after call arguments")
                expr = Call(
                    name=name_tok.val, args=args, kwargs=kwargs, receiver=expr, line=expr.line, col=expr.col)
            else:
                expr = Field(base=expr, name=name_tok.val, line=expr.line, col=expr.col)
        else:
            p.advance()
            expr = parse_index_or_slice(p, expr)
    return expr


def parse_receiver_call_args(p: TokenStream) -> Tuple[List[Node], Optional[List[Tuple[str, Node]]]]:
    """Positional or named args after `.name(`."""
    args: List[Node] = []
    kwargs: Optional[List[Tuple[str, Node]]] = None
    if p.check(TokenType.CLOSE_PAREN):
        return args, kwargs
    while True:
        start_tok = p.current()
        if p.check(TokenType.IDENTIFIER) and p.peek(1).type == TokenType.ASSIGN:
            field_name = p.advance().val
            p.advance()
            value = parse_expression(p)
            if args:
                raise p.error(
                    f"Cannot mix positional and named arguments in a call -- "
                    f"'{field_name}=...' follows a positional argument",
                    start_tok,
                )
            if kwargs is None:
                kwargs = []
            kwargs.append((field_name, value))
        else:
            value = parse_expression(p)
            if kwargs is not None:
                raise p.error(
                    f"Cannot mix positional and named arguments in a call -- a positional "
                    f"argument follows a named one",
                    start_tok,
                )
            args.append(value)
        if not p.match(TokenType.COMMA) or p.check(TokenType.CLOSE_PAREN):  # trailing comma
            break
    return args, kwargs


def parse_index_or_slice(p: TokenStream, array_expr: Node) -> Node:
    """Index or Slice after '['."""
    if p.check(TokenType.COLON):
        p.advance()
        high = None if p.check(TokenType.CLOSE_BRACKET) else parse_expression(p)
        p.expect(TokenType.CLOSE_BRACKET, "Expected ']' to close a slice expression")
        return Slice(array=array_expr, low=None, high=high, line=array_expr.line, col=array_expr.col)

    first = parse_expression(p)

    if p.match(TokenType.COLON):
        high = None if p.check(TokenType.CLOSE_BRACKET) else parse_expression(p)
        p.expect(TokenType.CLOSE_BRACKET, "Expected ']' to close a slice expression")
        return Slice(array=array_expr, low=first, high=high, line=array_expr.line, col=array_expr.col)

    p.expect(TokenType.CLOSE_BRACKET, "Expected ']' after array index")
    return Index(array=array_expr, index=first, line=array_expr.line, col=array_expr.col)


def parse_primary(p: TokenStream) -> Node:
    if p.check(TokenType.NUMBER):
        tok = p.advance()
        if '.' in tok.val:  # (the lexer takes `1.5` as one number, so that this can say so)
            raise p.error(
                f"A number can't have a fractional part ('{tok.val}') -- there are no floating-point types "
                f"yet, only integers", tok)
        return Constant(value=int(tok.val), line=tok.line, col=tok.col)
    if p.check(TokenType.TRUE, TokenType.FALSE):
        tok = p.advance()
        return BoolLiteral(value=(tok.type == TokenType.TRUE), line=tok.line, col=tok.col)
    if p.check(TokenType.NONE):
        tok = p.advance()
        return NoneLiteral(line=tok.line, col=tok.col)
    if p.check(TokenType.STRING):
        tok = p.advance()
        return StringLiteral(value=p.literal_text(tok), line=tok.line, col=tok.col)
    if p.check(TokenType.BYTE):
        tok = p.advance()
        resolved = p.literal_text(tok)
        if len(resolved) != 1 or ord(resolved) > 255:
            raise p.error(
                f"A byte literal must resolve to exactly one byte (0-255), got "
                f"{quoted_text(resolved)}",
                tok,
            )
        return ByteLiteral(value=ord(resolved), line=tok.line, col=tok.col)
    if (
            p.check(
                TokenType.INT,
                TokenType.INT8,
                TokenType.UINT8,
                TokenType.INT64,
                TokenType.INT32,
                TokenType.BOOL,
                TokenType.STR
            )
            and p.peek(1).type == TokenType.OPEN_PAREN
    ):
        return parse_cast(p)
    if p.check(TokenType.DICT):
        parsed_type = type_exprs.parse_type(p)
        return parse_dict_literal(p, parsed_type)
    if _looks_like_typed_literal(p):
        could_be_multiplication = p.pos in p.could_be_multiplication
        try:
            parsed_type = type_exprs.parse_type(p)
        except ParseError as problem:
            if not could_be_multiplication:
                raise
            raise ParseError(problem.message + READ_AS_A_TYPED_LITERAL, problem.file, problem.line,
                             problem.col) from None
        literal = parse_bracketed_literal(p, parsed_type)
        if could_be_multiplication and isinstance(literal, ArrayLiteral):
            literal.could_be_multiplication = True
        return literal
    if p.check(TokenType.OPEN_BRACKET):
        return parse_array_literal(p)
    if p.check(TokenType.IDENTIFIER):
        if p.peek(1).type == TokenType.OPEN_PAREN:
            return parse_call(p)
        tok = p.advance()
        return Variable(name=tok.val, line=tok.line, col=tok.col)
    if p.match(TokenType.OPEN_PAREN):
        expr = parse_expression(p)
        p.expect(TokenType.CLOSE_PAREN, "Expected ')' to close grouped expression")
        return expr
    tok = p.current()
    raise p.error(f"Expected an expression, got {describe_token(tok)}", tok)


def _looks_like_typed_literal(p: TokenStream) -> bool:
    """Whether '[' starts a typed literal (`[3]int[...]`, `[N][]T[...]`, `[]dict[K]V[...]`) rather
    than an untyped one: one or more bracket groups followed by the start of a type."""
    if not p.check(TokenType.OPEN_BRACKET):
        return False
    k = 0
    groups = 0
    empty_group = False
    while p.peek(k).type == TokenType.OPEN_BRACKET:
        groups += 1
        empty_group = empty_group or p.peek(k + 1).type == TokenType.CLOSE_BRACKET
        depth = 0
        while True:
            t = p.peek(k).type
            if t == TokenType.EOF:
                return False
            depth += (t == TokenType.OPEN_BRACKET) - (t == TokenType.CLOSE_BRACKET)
            k += 1
            if depth == 0:
                break
    if p.peek(k).type != TokenType.STAR:
        return p.peek(k).type in _TYPE_START_TOKENS
    # `*` might instead be a multiplication (`[x][0] * y`): a pointer element type must run on to
    # the elements' '['.
    while p.peek(k).type == TokenType.STAR:
        k += 1
    if p.peek(k).type in (TokenType.OPEN_BRACKET, TokenType.DICT):
        return True
    if p.peek(k).type == TokenType.IDENTIFIER:
        if groups >= 2 and not empty_group:
            # `[N][M]*P[...]` has the tokens of `[x][0] * ys[1]`, an indexed literal times an
            # index. It is the typed literal: a literal without a type can't be indexed.
            p.could_be_multiplication.add(p.pos)
        k += 1
        if p.peek(k).type == TokenType.DOT and p.peek(k + 1).type == TokenType.IDENTIFIER:
            k += 2
    elif p.peek(k).type in _TYPE_START_TOKENS:
        k += 1
    else:
        return False
    return p.peek(k).type == TokenType.OPEN_BRACKET


def parse_bracketed_literal(p: TokenStream, parsed_type: Union[str, 'ArrayTypeExpr', 'SliceTypeExpr']) -> Node:
    """Bracketed elements after a pre-parsed literal type."""
    if isinstance(parsed_type, SliceTypeExpr):
        elements = parse_array_literal(p).elements
        return SliceLiteral(element_type=parsed_type.element_type, elements=elements,
                            line=parsed_type.line, col=parsed_type.col)
    return parse_array_literal(p, type_expr=parsed_type)


def parse_array_literal(p: TokenStream, type_expr: Optional['ArrayTypeExpr'] = None) -> ArrayLiteral:
    """`[e1, ...]`, optionally typed."""
    open_tok = p.expect(TokenType.OPEN_BRACKET)
    elements = []
    if not p.check(TokenType.CLOSE_BRACKET):
        elements.append(parse_expression(p))
        while p.match(TokenType.COMMA) and not p.check(TokenType.CLOSE_BRACKET):  # trailing comma
            elements.append(parse_expression(p))
    p.expect(TokenType.CLOSE_BRACKET, "Expected ']' to close array literal")
    return ArrayLiteral(elements=elements, type_expr=type_expr, line=open_tok.line, col=open_tok.col)


def parse_dict_literal(p: TokenStream, dict_type: DictTypeExpr) -> DictLiteral:
    """`{k: v, ...}` after a parsed dict type."""
    open_tok = p.expect(TokenType.OPEN_BRACE)
    p.skip_newlines()
    entries = []
    if not p.check(TokenType.CLOSE_BRACE):
        entries.append(_parse_dict_entry(p))
        p.skip_newlines()
        while p.match(TokenType.COMMA):
            p.skip_newlines()
            if p.check(TokenType.CLOSE_BRACE):
                break
            entries.append(_parse_dict_entry(p))
            p.skip_newlines()
    p.expect(TokenType.CLOSE_BRACE, "Expected '}' to close dict literal")
    return DictLiteral(
        key_type=dict_type.key_type, value_type=dict_type.value_type,
        entries=entries, line=open_tok.line, col=open_tok.col,
    )


def _parse_dict_entry(p: TokenStream) -> Tuple[Node, Node]:
    """`key: value`."""
    key = parse_expression(p)
    p.expect(TokenType.COLON, "Expected ':' between a dict entry's key and value")
    value = parse_expression(p)
    return key, value


def parse_call(p: TokenStream) -> Call:
    """`name(args)` or `name(f=v, ...)`; no mixing."""
    name_tok = p.expect(TokenType.IDENTIFIER)
    p.expect(TokenType.OPEN_PAREN, "Expected '(' to start a call's argument list")
    args: List[Node] = []
    kwargs: Optional[List[Tuple[str, Node]]] = None
    if not p.check(TokenType.CLOSE_PAREN):
        while True:
            start_tok = p.current()
            if p.check(TokenType.IDENTIFIER) and p.peek(1).type == TokenType.ASSIGN:
                field_name = p.advance().val
                p.advance()
                value = parse_expression(p)
                if args:
                    raise p.error(
                        f"Cannot mix positional and named arguments in a call -- "
                        f"'{field_name}=...' follows a positional argument",
                        start_tok,
                    )
                if kwargs is None:
                    kwargs = []
                kwargs.append((field_name, value))
            else:
                value = parse_expression(p)
                if kwargs is not None:
                    raise p.error(
                        f"Cannot mix positional and named arguments in a call -- a positional "
                        f"argument follows a named one",
                        start_tok,
                    )
                args.append(value)
            if not p.match(TokenType.COMMA) or p.check(TokenType.CLOSE_PAREN):  # trailing comma
                break
    p.expect(TokenType.CLOSE_PAREN, "Expected ')' to close a call's argument list")
    return Call(name=name_tok.val, args=args, kwargs=kwargs, line=name_tok.line, col=name_tok.col)


def parse_cast(p: TokenStream) -> Cast:
    """`T(expr)`."""
    type_tok = p.advance()
    p.expect(TokenType.OPEN_PAREN, "Expected '(' to start a cast's argument")
    expr = parse_expression(p)
    p.expect(TokenType.CLOSE_PAREN, "Expected ')' to close a cast's argument")
    return Cast(target_type=type_tok.val, expr=expr, line=type_tok.line, col=type_tok.col)
