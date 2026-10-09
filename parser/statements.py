"""Statements: a block, and each kind of statement in one.

A statement is told from its first tokens: a keyword (`if`, `while`, `for`, `match`, `return`,
`break`, `continue`), a type that starts a declaration, or else an expression, which may turn out
to be an assignment's target."""

from typing import List, Optional, Tuple, Union

from lexer import Token, TokenType
from ops import BinaryOp, UnaryOp
from parser import expressions, type_exprs
from parser.errors import ParseError
from parser.nodes import (
    ArrayTypeExpr, Assign, Break, Continue, ExprStmt, Field, For, ForIn, If, Index, IsCheck, Match, Node,
    QualifiedTypeExpr, Return, SliceTypeExpr, Unary, VarDecl, Variable, While,
)
from parser.stream import TokenStream

# TokenType -> BinaryOp for compound assignment.
_COMPOUND_ASSIGN_OPS = {
    TokenType.PLUS_ASSIGN:      BinaryOp.ADD,
    TokenType.MINUS_ASSIGN:     BinaryOp.SUBTRACT,
    TokenType.STAR_ASSIGN:      BinaryOp.MULTIPLY,
    TokenType.SLASH_ASSIGN:     BinaryOp.DIVIDE,
    TokenType.PERCENT_ASSIGN:   BinaryOp.MODULO,
    TokenType.AMPERSAND_ASSIGN: BinaryOp.BITWISE_AND,
    TokenType.PIPE_ASSIGN:      BinaryOp.BITWISE_OR,
    TokenType.CARET_ASSIGN:     BinaryOp.BITWISE_XOR,
    TokenType.SHIFT_LEFT_ASSIGN:  BinaryOp.SHIFT_LEFT,
    TokenType.SHIFT_RIGHT_ASSIGN: BinaryOp.SHIFT_RIGHT,
}


def parse_block(p: TokenStream) -> List[Node]:
    """INDENT statement+ DEDENT."""
    p.skip_newlines()
    p.expect(TokenType.INDENT, "Expected an indented block")
    p.skip_newlines()
    statements = []
    while not p.check(TokenType.DEDENT) and not p.at_end():
        statements.append(parse_statement(p))
        p.expect_statement_end()
        p.skip_newlines()
    p.expect(TokenType.DEDENT, "Expected the end of an indented block")
    if not statements:
        raise p.error("Expected at least one statement in this block", p.current())
    return statements


def parse_statement(p: TokenStream) -> Node:
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
        # Scalar keyword + '(' is a cast statement, not a VarDecl.
        return parse_expr_stmt_or_assign(p)
    if p.check(TokenType.STAR):
        # '*' may start a pointer-typed VarDecl or a deref expression; try the type first and backtrack.
        mark = p.mark()
        start_tok = p.current()
        try:
            parsed_type = type_exprs.parse_type(p)
        except ParseError:
            parsed_type = None
        if parsed_type is not None and p.check(TokenType.IDENTIFIER):
            return parse_var_decl(p, var_type=parsed_type, start_tok=start_tok)
        p.reset(mark)
    if p.check(
            TokenType.INT,
            TokenType.INT8,
            TokenType.UINT8,
            TokenType.INT64,
            TokenType.INT32,
            TokenType.BOOL,
            TokenType.STR,
            TokenType.OPEN_BRACKET,
            TokenType.DICT
    ):
        # Type may start a VarDecl or a typed literal statement; parse it once, then decide.
        start_tok = p.current()
        parsed_type = type_exprs.parse_type(p)
        if isinstance(parsed_type, (ArrayTypeExpr, SliceTypeExpr)) and p.check(TokenType.OPEN_BRACKET):
            literal = expressions.parse_bracketed_literal(p, parsed_type)
            return ExprStmt(expr=literal, line=start_tok.line, col=start_tok.col)
        return parse_var_decl(p, var_type=parsed_type, start_tok=start_tok)
    if p.check(TokenType.RETURN):
        return parse_return(p)
    if p.check(TokenType.IF):
        return parse_if(p)
    if p.check(TokenType.MATCH):
        return parse_match(p)
    if p.check(TokenType.WHILE):
        return parse_while(p)
    if p.check(TokenType.FOR):
        return parse_for(p)
    if p.check(TokenType.BREAK):
        return parse_break(p)
    if p.check(TokenType.CONTINUE):
        return parse_continue(p)
    if p.check(TokenType.IDENTIFIER) and p.peek(1).type == TokenType.IDENTIFIER:
        # IDENTIFIER IDENTIFIER is a struct-typed VarDecl.
        start_tok = p.current()
        parsed_type = type_exprs.parse_type(p)
        return parse_var_decl(p, var_type=parsed_type, start_tok=start_tok)
    if (
            p.check(TokenType.IDENTIFIER)
            and p.peek(1).type == TokenType.DOT
            and p.peek(2).type == TokenType.IDENTIFIER
            and p.peek(3).type == TokenType.IDENTIFIER
    ):
        # IDENTIFIER . IDENTIFIER IDENTIFIER is a qualified struct-typed VarDecl.
        start_tok = p.current()
        parsed_type = type_exprs.parse_type(p)
        return parse_var_decl(p, var_type=parsed_type, start_tok=start_tok)
    return parse_expr_stmt_or_assign(p)


def parse_while(p: TokenStream) -> While:
    start_tok = p.expect(TokenType.WHILE, "Expected 'while'")
    condition = expressions.parse_expression(p)
    p.expect(TokenType.COLON, "Expected ':' to start the while body")
    p.expect(TokenType.NEWLINE, "Expected a newline after ':'")
    body = parse_block(p)
    return While(condition=condition, body=body, line=start_tok.line, col=start_tok.col)


def parse_for(p: TokenStream) -> Node:
    """Dispatch between three-clause and for-in forms."""
    if p.peek(1).type == TokenType.IDENTIFIER and p.peek(2).type in (TokenType.COMMA, TokenType.IN):
        return parse_for_in(p)
    return _parse_for_three_clause(p)


def _parse_for_three_clause(p: TokenStream) -> For:
    """`for init; cond; increment:`."""
    start_tok = p.expect(TokenType.FOR, "Expected 'for'")
    init = _parse_for_init_clause(p)
    p.expect(TokenType.SEMICOLON, "Expected ';' after the for-loop's own init clause")
    condition = expressions.parse_expression(p)
    p.expect(TokenType.SEMICOLON, "Expected ';' after the for-loop's own condition")
    increment = _parse_for_increment_clause(p)
    p.expect(TokenType.COLON, "Expected ':' to start the for body")
    p.expect(TokenType.NEWLINE, "Expected a newline after ':'")
    body = parse_block(p)
    return For(
        init=init, condition=condition, increment=increment, body=body, line=start_tok.line, col=start_tok.col)


def parse_for_in(p: TokenStream) -> ForIn:
    """`for a[, b] in iterable:`."""
    start_tok = p.expect(TokenType.FOR, "Expected 'for'")
    first_name_tok = p.expect(TokenType.IDENTIFIER, "Expected a loop variable name after 'for'")
    binding_names = [first_name_tok.val]
    if p.match(TokenType.COMMA):
        second_name_tok = p.expect(TokenType.IDENTIFIER, "Expected a second loop variable name after ','")
        binding_names.append(second_name_tok.val)
    p.expect(TokenType.IN, "Expected 'in' after the for-loop's own binding name(s)")
    iterable = expressions.parse_expression(p)
    p.expect(TokenType.COLON, "Expected ':' to start the for body")
    p.expect(TokenType.NEWLINE, "Expected a newline after ':'")
    body = parse_block(p)
    return ForIn(binding_names=binding_names, iterable=iterable, body=body, line=start_tok.line, col=start_tok.col)


def _parse_for_init_clause(p: TokenStream) -> Node:
    """Init clause: always a VarDecl."""
    start_tok = p.current()
    if p.check(
            TokenType.INT,
            TokenType.INT8,
            TokenType.UINT8,
            TokenType.INT64,
            TokenType.INT32,
            TokenType.BOOL,
            TokenType.STR,
            TokenType.OPEN_BRACKET,
            TokenType.DICT
    ):
        parsed_type = type_exprs.parse_type(p)
        return parse_var_decl(p, var_type=parsed_type, start_tok=start_tok)
    if p.check(TokenType.IDENTIFIER) and p.peek(1).type == TokenType.IDENTIFIER:
        parsed_type = type_exprs.parse_type(p)
        return parse_var_decl(p, var_type=parsed_type, start_tok=start_tok)
    raise p.error(
        f"Expected a variable declaration (e.g. `int i = 0`) as the for-loop's "
        f"own init clause",
        start_tok,
    )


def _parse_for_increment_clause(p: TokenStream) -> Node:
    """Increment clause: an assignment."""
    start_tok = p.current()
    statement = parse_expr_stmt_or_assign(p)
    if isinstance(statement, Assign):
        return statement
    raise p.error(
        f"Expected an assignment (e.g. `i += 1`) as the for-loop's own increment "
        f"clause",
        start_tok,
    )


def parse_break(p: TokenStream) -> Break:
    tok = p.expect(TokenType.BREAK, "Expected 'break'")
    return Break(line=tok.line, col=tok.col)


def parse_continue(p: TokenStream) -> Continue:
    tok = p.expect(TokenType.CONTINUE, "Expected 'continue'")
    return Continue(line=tok.line, col=tok.col)


def parse_if(p: TokenStream) -> If:
    start_tok = p.expect(TokenType.IF, "Expected 'if'")
    return _parse_if_body(p, start_tok)


def parse_elif_as_if(p: TokenStream) -> If:
    start_tok = p.expect(TokenType.ELIF, "Expected 'elif'")
    return _parse_if_body(p, start_tok)


def _parse_if_body(p: TokenStream, start_tok: Token) -> If:
    """`KEYWORD condition : block`; shared by if/elif."""
    condition = _parse_if_condition(p)
    p.expect(TokenType.COLON, "Expected ':' to start the if body")
    p.expect(TokenType.NEWLINE, "Expected a newline after ':'")
    then_body = parse_block(p)

    else_body = None
    p.skip_newlines()
    if p.check(TokenType.ELIF):
        else_body = [parse_elif_as_if(p)]
    elif p.match(TokenType.ELSE):
        p.expect(TokenType.COLON, "Expected ':' to start the else body")
        p.expect(TokenType.NEWLINE, "Expected a newline after ':'")
        else_body = parse_block(p)

    return If(condition=condition, then_body=then_body, else_body=else_body, line=start_tok.line, col=start_tok.col)


def _parse_if_condition(p: TokenStream) -> Node:
    """A boolean expression (an `is` check in it may bind its subject with `as NAME`)."""
    expr = expressions.parse_expression(p)
    if p.check(TokenType.AS):
        raise p.error("'as NAME' follows an 'is' check (`EXPR is T as NAME`)", p.current())
    return expr


def parse_match(p: TokenStream) -> 'Match':
    """`match NAME:` / `match EXPR as NAME:` with `is T:` arms and optional `else:`."""
    start_tok = p.expect(TokenType.MATCH, "Expected 'match'")
    if p.check(TokenType.IDENTIFIER) and p.peek(1).type == TokenType.COLON:
        name_tok = p.advance()
        binding_name = name_tok.val
        subject = None
    else:
        subject = expressions.parse_expression(p)
        p.expect(
            TokenType.AS,
            "Expected 'as NAME' after the match subject -- a non-bare-"
            "variable (or explicitly renamed) subject needs an explicit "
            "binding name to narrow, since it has no existing name of "
            "its own",
        )
        binding_tok = p.expect(TokenType.IDENTIFIER, "Expected a binding name after 'as'")
        binding_name = binding_tok.val
    p.expect(TokenType.COLON, "Expected ':' to start the match body")
    p.expect(TokenType.NEWLINE, "Expected a newline after ':'")
    p.skip_newlines()
    p.expect(TokenType.INDENT, "Expected an indented block")
    p.skip_newlines()

    arms: List[Tuple[Token, Union[str, QualifiedTypeExpr], List[Node]]] = []
    else_body: Optional[List[Node]] = None
    while not p.check(TokenType.DEDENT) and not p.at_end():
        if p.match(TokenType.ELSE):
            p.expect(TokenType.COLON, "Expected ':' to start the else body")
            p.expect(TokenType.NEWLINE, "Expected a newline after ':'")
            else_body = parse_block(p)
            p.skip_newlines()
            break
        arm_tok = p.expect(TokenType.IS, "Expected 'is' (a match arm) or 'else'")
        type_name = type_exprs.parse_qualifiable_type_name(p, "a type name after 'is'")
        p.expect(TokenType.COLON, "Expected ':' to start this arm's body")
        p.expect(TokenType.NEWLINE, "Expected a newline after ':'")
        arm_body = parse_block(p)
        arms.append((arm_tok, type_name, arm_body))
        p.skip_newlines()

    p.expect(TokenType.DEDENT, "Expected the match body to end")

    if not arms:
        raise p.error(f"Expected at least one 'is' arm in this match", start_tok)

    checks = [(IsCheck(variable_name=binding_name, type_name=type_name, subject=subject if i == 0 else None,
                       line=arm_tok.line, col=arm_tok.col), arm_body)
              for i, (arm_tok, type_name, arm_body) in enumerate(arms)]
    return Match(variable_name=binding_name, subject=subject, arms=checks, else_body=else_body,
                 line=arms[0][0].line, col=arms[0][0].col)


def parse_var_decl(p: TokenStream, var_type: Optional[Union[str, 'ArrayTypeExpr', 'SliceTypeExpr']] = None,
        start_tok: Optional[Token] = None,
) -> VarDecl:
    """`T name [= expr]`; var_type may be pre-parsed."""
    if start_tok is None:
        start_tok = p.current()
    if var_type is None:
        var_type = type_exprs.parse_type(p)
    name_tok = p.expect(TokenType.IDENTIFIER, "Expected a variable name")
    init = None
    if p.match(TokenType.ASSIGN):
        init = expressions.parse_expression(p)
    return VarDecl(name=name_tok.val, var_type=var_type, init=init, line=start_tok.line, col=start_tok.col)


def parse_return(p: TokenStream) -> Return:
    """`return [expr]`."""
    start_tok = p.expect(TokenType.RETURN)
    if p.check(TokenType.NEWLINE):
        return Return(value=None, line=start_tok.line, col=start_tok.col)
    value = expressions.parse_expression(p)
    return Return(value=value, line=start_tok.line, col=start_tok.col)


def parse_expr_stmt_or_assign(p: TokenStream) -> Node:
    """Expression statement, or assignment to a name, field, index, or dereference (parse the
    left side, then check for '=' or a compound operator)."""
    expr = expressions.parse_expression(p)
    op_tok = p.current()
    compound_op = _COMPOUND_ASSIGN_OPS.get(op_tok.type)
    if op_tok.type == TokenType.ASSIGN or compound_op is not None:
        assignable = isinstance(expr, (Variable, Index, Field)) or (
                isinstance(expr, Unary) and expr.op == UnaryOp.DEREFERENCE)
        if not assignable:
            raise p.error(f"Left-hand side of '{op_tok.val}' is not assignable", op_tok)
        p.advance()
        value = expressions.parse_expression(p)
        return Assign(target=expr, value=value, op=compound_op, line=expr.line, col=expr.col)
    return ExprStmt(expr=expr, line=expr.line, col=expr.col)
