"""Recursive-descent parser: tokens -> AST. Precedence climbing for binary operators."""

from typing import List, Optional, Tuple, Union

from lexer import Token, TokenType
from ops import BinaryOp, UnaryOp
from parser import expressions, type_exprs
from parser.errors import ParseError
from parser.nodes import (
    ArrayTypeExpr, Assign, Break, ConstDecl, Continue, EnumDef, EnumMember, ExprStmt, ExternFunctionDecl, Field, For,
    ForIn, FromImportDecl, Function, If, ImportDecl, Index, IntrinsicDecl, IsCheck, Match, MethodDef, Node, Param,
    Program, QualifiedTypeExpr, Return, SliceTypeExpr, StructDef, StructField, SumTypeDef, TypeAlias, Unary, VarDecl,
    Variable, While, stamp_file,
)
from parser.stream import TokenStream




def _default_import_qualifier(path: str) -> str:
    """Last path component, minus '.ht'."""
    last_component = path.rsplit('/', 1)[-1]
    if last_component.endswith('.ht'):
        last_component = last_component[:-len('.ht')]
    return last_component













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




class Parser:
    def __init__(self, tokens: List[Token]):
        self.stream = TokenStream(tokens)

    # The cursor is the stream's. These stand in for it while the grammar below is still this
    # class's methods; each part of it that becomes functions takes the stream itself.

    @property
    def tokens(self) -> List[Token]:
        return self.stream.tokens

    @property
    def pos(self) -> int:
        return self.stream.pos

    @pos.setter
    def pos(self, pos: int) -> None:
        self.stream.pos = pos

    @property
    def _could_be_multiplication(self) -> set:
        return self.stream.could_be_multiplication

    def _error(self, message: str, tok: Token) -> ParseError:
        return self.stream.error(message, tok)

    def _literal_text(self, tok: Token) -> str:
        return self.stream.literal_text(tok)

    def peek(self, offset: int = 0) -> Token:
        return self.stream.peek(offset)

    def current(self) -> Token:
        return self.stream.current()

    def at_end(self) -> bool:
        return self.stream.at_end()

    def check(self, *types: TokenType) -> bool:
        return self.stream.check(*types)

    def advance(self) -> Token:
        return self.stream.advance()

    def match(self, *types: TokenType) -> bool:
        return self.stream.match(*types)

    def expect(self, type_: TokenType, message: str = None) -> Token:
        return self.stream.expect(type_, message)

    def skip_newlines(self):
        self.stream.skip_newlines()

    def parse_program(self) -> Program:
        start_tok = self.current()
        functions = []
        structs = []
        type_aliases = []
        sum_types = []
        extern_functions = []
        imports = []
        from_imports = []
        intrinsics = []
        consts = []
        enums = []
        self.skip_newlines()
        while not self.at_end():
            if self.check(TokenType.STRUCT):
                tok = self.current()
                raise self._error(
                    f"Bare 'struct Name:' is no longer supported -- write 'type Name "
                    f"struct:' instead",
                    tok,
                )
            elif self.check(TokenType.TYPE):
                declaration = self.parse_type_declaration()
                if isinstance(declaration, StructDef):
                    structs.append(declaration)
                elif isinstance(declaration, EnumDef):
                    enums.append(declaration)
                elif isinstance(declaration, SumTypeDef):
                    sum_types.append(declaration)
                else:
                    type_aliases.append(declaration)
            elif self.check(TokenType.EXTERN):
                extern_functions.append(self.parse_extern_function())
            elif self.check(TokenType.INTRINSIC):
                intrinsics.append(self.parse_intrinsic())
            elif self.check(TokenType.CONST):
                consts.append(self.parse_const())
            elif self.check(TokenType.IMPORT):
                imports.append(self.parse_import())
            elif self.check(TokenType.FROM):
                from_imports.append(self.parse_from_import())
            else:
                functions.append(self.parse_function())
            self.skip_newlines()
        program = Program(
            functions=functions, structs=structs, type_aliases=type_aliases, sum_types=sum_types,
            extern_functions=extern_functions, imports=imports, from_imports=from_imports,
            intrinsics=intrinsics, consts=consts, enums=enums,
            line=start_tok.line, col=start_tok.col,
        )
        if start_tok.file:
            stamp_file(program, start_tok.file)
        return program

    def parse_intrinsic(self) -> IntrinsicDecl:
        """`intrinsic T name(params)`."""
        start_tok = self.expect(TokenType.INTRINSIC, "Expected 'intrinsic'")
        return_type = None
        if type_exprs.check_starts_with_return_type(self.stream):
            return_type = type_exprs.parse_type(self.stream)
        name_tok = self.expect(TokenType.IDENTIFIER, "Expected a function name")
        self.expect(TokenType.OPEN_PAREN, "Expected '(' after function name")
        params = self.parse_params()
        self.expect(TokenType.CLOSE_PAREN, "Expected ')' after parameter list")
        return IntrinsicDecl(
            name=name_tok.val, original_name=name_tok.val, return_type=return_type, params=params,
            line=start_tok.line, col=start_tok.col,
        )

    def parse_import(self) -> ImportDecl:
        """`import 'path' [as name]`."""
        start_tok = self.expect(TokenType.IMPORT, "Expected 'import'")
        path_tok = self.expect(TokenType.STRING, "Expected a quoted path after 'import'")
        path = self._literal_text(path_tok)
        if self.match(TokenType.AS):
            alias_tok = self.expect(TokenType.IDENTIFIER, "Expected a module name after 'as'")
            qualifier = alias_tok.val
        else:
            qualifier = _default_import_qualifier(path)  # an identifier, or modules.py rejects the file
        return ImportDecl(path=path, qualifier=qualifier, line=start_tok.line, col=start_tok.col)

    def parse_from_import(self) -> FromImportDecl:
        """`from 'path' import a [as b], ...`."""
        start_tok = self.expect(TokenType.FROM, "Expected 'from'")
        path_tok = self.expect(TokenType.STRING, "Expected a quoted path after 'from'")
        path = self._literal_text(path_tok)
        self.expect(TokenType.IMPORT, "Expected 'import' after the path")
        names: List[Tuple[str, str]] = []
        while True:
            name_tok = self.expect(TokenType.IDENTIFIER, "Expected a name to import")
            if self.match(TokenType.AS):
                alias_tok = self.expect(TokenType.IDENTIFIER, "Expected an alias name after 'as'")
                names.append((name_tok.val, alias_tok.val))
            else:
                names.append((name_tok.val, name_tok.val))
            if not self.match(TokenType.COMMA):
                break
        return FromImportDecl(path=path, names=names, line=start_tok.line, col=start_tok.col)

    def parse_type_declaration(self) -> Union[TypeAlias, StructDef, SumTypeDef, EnumDef]:
        """`type Name = T`, `type Name struct:`, `type Name enum:`, or `type Name is A | B`."""
        start_tok = self.expect(TokenType.TYPE, "Expected 'type' to start a type declaration")
        name_tok = self.expect(TokenType.IDENTIFIER, "Expected a name for this type declaration")
        if self.check(TokenType.STRUCT):
            self.advance()
            return self._parse_struct_body(start_tok, name_tok)
        if self.check(TokenType.ENUM):
            self.advance()
            return self._parse_enum_body(start_tok, name_tok)
        if self.check(TokenType.IS):
            self.advance()
            return self._parse_sum_type_body(start_tok, name_tok)
        self.expect(TokenType.ASSIGN, "Expected '=' (for a type alias), 'struct' (for a struct declaration), "
                                      "'enum' (for an enum), or 'is' (for a sum type)")
        target_type = type_exprs.parse_type(self.stream)
        self.expect(TokenType.NEWLINE, "Expected a newline after a type alias declaration")
        return TypeAlias(name=name_tok.val, target_type=target_type, line=start_tok.line, col=start_tok.col)

    def _parse_sum_type_body(self, start_tok: Token, name_tok: Token) -> SumTypeDef:
        """`A | B ...` after `type Name is`; at least two variants."""
        first_line_tok = self.current()
        first_variant = type_exprs.parse_qualifiable_type_name(self.stream, "a variant name")
        variants = [first_variant]
        while self.match(TokenType.PIPE):
            variant = type_exprs.parse_qualifiable_type_name(self.stream, "a variant name after '|'")
            variants.append(variant)
        if len(variants) < 2:
            raise self._error(
                f"Expected at least one '|' and a second variant in sum type "
                f"'{name_tok.val}' -- a sum type needs at least two variants",
                first_line_tok,
            )
        self.expect(TokenType.NEWLINE, "Expected a newline after a sum type declaration")
        return SumTypeDef(name=name_tok.val, variants=variants, line=start_tok.line, col=start_tok.col)

    def _parse_enum_body(self, start_tok: Token, name_tok: Token) -> EnumDef:
        """Enum body after `type Name enum`: one member name per line, then any methods."""
        self.expect(TokenType.COLON, "Expected ':' to start the enum body")
        self.expect(TokenType.NEWLINE, "Expected a newline after ':'")
        self.skip_newlines()
        self.expect(TokenType.INDENT, "Expected an indented enum body")
        self.skip_newlines()
        members: List[EnumMember] = []
        methods: List[MethodDef] = []
        while not self.check(TokenType.DEDENT) and not self.at_end():
            if self.check(TokenType.DEF):
                methods.append(self.parse_method_def())
            elif methods:
                raise self._error("An enum's members come before its methods", self.current())
            else:
                member_tok = self.expect(TokenType.IDENTIFIER, "Expected a member name")
                self.expect(
                    TokenType.NEWLINE, "Expected a newline after a member name -- an enum lists one member per line")
                members.append(EnumMember(name=member_tok.val, line=member_tok.line, col=member_tok.col))
            self.skip_newlines()
        self.expect(TokenType.DEDENT, "Expected a dedent to end the enum body")
        return EnumDef(
            name=name_tok.val, members=members, methods=methods, line=start_tok.line, col=start_tok.col)

    def _parse_struct_body(self, start_tok: Token, name_tok: Token) -> StructDef:
        """Struct body after `type Name struct`."""
        self.expect(TokenType.COLON, "Expected ':' to start the struct body")
        self.expect(TokenType.NEWLINE, "Expected a newline after ':'")
        self.skip_newlines()
        self.expect(TokenType.INDENT, "Expected an indented struct body")
        self.skip_newlines()
        fields: List[StructField] = []
        methods: List[MethodDef] = []
        while not self.check(TokenType.DEDENT) and not self.at_end():
            if self.check(TokenType.DEF):
                methods.append(self.parse_method_def())
            else:
                field_start_tok = self.current()
                field_type = type_exprs.parse_type(self.stream)
                field_name_tok = self.expect(TokenType.IDENTIFIER, "Expected a field name")
                self.expect(TokenType.NEWLINE, "Expected a newline after a field declaration")
                fields.append(StructField(
                    name=field_name_tok.val, field_type=field_type,
                    line=field_start_tok.line, col=field_start_tok.col,
                ))
            self.skip_newlines()
        self.expect(TokenType.DEDENT, "Expected a dedent to end the struct body")
        if not fields:
            raise self._error(f"Expected at least one field in struct '{name_tok.val}'", self.current())
        return StructDef(name=name_tok.val, fields=fields, methods=methods, line=start_tok.line, col=start_tok.col)



    def parse_function(self) -> Function:
        start_tok = self.expect(TokenType.DEF, "Expected 'def' to start a function definition")
        return_type = type_exprs.parse_return_type(self.stream)
        name_tok = self.expect(TokenType.IDENTIFIER, "Expected a function name")
        self.expect(TokenType.OPEN_PAREN, "Expected '(' after function name")
        params = self.parse_params()
        self.expect(TokenType.CLOSE_PAREN, "Expected ')' after parameter list")
        self.expect(TokenType.COLON, "Expected ':' to start the function body")
        self.expect(TokenType.NEWLINE, "Expected a newline after ':'")

        body = self.parse_block()
        return Function(
            name=name_tok.val, return_type=return_type, params=params, body=body,
            line=start_tok.line, col=start_tok.col,
        )

    def parse_extern_function(self) -> ExternFunctionDecl:
        """`extern [T] name(params)`."""
        start_tok = self.expect(TokenType.EXTERN, "Expected 'extern' to start an external function declaration")
        return_type = type_exprs.parse_return_type(self.stream)
        name_tok = self.expect(TokenType.IDENTIFIER, "Expected a function name")
        self.expect(TokenType.OPEN_PAREN, "Expected '(' after function name")
        params = self.parse_params()
        self.expect(TokenType.CLOSE_PAREN, "Expected ')' after parameter list")
        return ExternFunctionDecl(
            name=name_tok.val, return_type=return_type, params=params,
            line=start_tok.line, col=start_tok.col,
        )

    def parse_method_def(self) -> MethodDef:
        """`def [T] name(receiver, ...):` or `def [T] name(*receiver, ...):`."""
        start_tok = self.expect(TokenType.DEF, "Expected 'def' to start a method definition")
        return_type = type_exprs.parse_return_type(self.stream)
        name_tok = self.expect(TokenType.IDENTIFIER, "Expected a method name")
        self.expect(TokenType.OPEN_PAREN, "Expected '(' after method name")
        receiver_is_pointer = self.match(TokenType.STAR)
        receiver_tok = self.expect(TokenType.IDENTIFIER, "Expected a receiver name as a method's first parameter")
        params: List[Param] = []
        while self.match(TokenType.COMMA) and not self.check(TokenType.CLOSE_PAREN):  # trailing comma
            params.append(self.parse_param())
        self.expect(TokenType.CLOSE_PAREN, "Expected ')' after parameter list")
        self.expect(TokenType.COLON, "Expected ':' to start the method body")
        self.expect(TokenType.NEWLINE, "Expected a newline after ':'")

        body = self.parse_block()
        return MethodDef(
            receiver_name=receiver_tok.val, name=name_tok.val, return_type=return_type, params=params, body=body,
            receiver_is_pointer=receiver_is_pointer, line=start_tok.line, col=start_tok.col,
        )

    def parse_params(self) -> List[Param]:
        """`T name, ...` up to, not including, ')'; a trailing comma is allowed."""
        params = []
        if self.check(TokenType.CLOSE_PAREN):
            return params
        params.append(self.parse_param())
        while self.match(TokenType.COMMA) and not self.check(TokenType.CLOSE_PAREN):
            params.append(self.parse_param())
        return params

    def parse_const(self) -> ConstDecl:
        """`const T NAME = expr`."""
        start_tok = self.expect(TokenType.CONST, "Expected 'const'")
        const_type = type_exprs.parse_type(self.stream)
        name_tok = self.expect(TokenType.IDENTIFIER, "Expected a constant name")
        self.expect(TokenType.ASSIGN, "Expected '=' after a constant's name")
        value = expressions.parse_expression(self.stream)
        self.expect(TokenType.NEWLINE, "Expected a newline after a constant declaration")
        return ConstDecl(name=name_tok.val, const_type=const_type, value=value, line=start_tok.line, col=start_tok.col)

    def parse_param(self) -> Param:
        start_tok = self.current()
        param_type = type_exprs.parse_type(self.stream)
        name_tok = self.expect(TokenType.IDENTIFIER, "Expected a parameter name")
        return Param(name=name_tok.val, type=param_type, line=start_tok.line, col=start_tok.col)


    def parse_block(self) -> List[Node]:
        """INDENT statement+ DEDENT."""
        self.skip_newlines()
        self.expect(TokenType.INDENT, "Expected an indented block")
        self.skip_newlines()
        statements = []
        while not self.check(TokenType.DEDENT) and not self.at_end():
            statements.append(self.parse_statement())
            self._expect_statement_end()
            self.skip_newlines()
        self.expect(TokenType.DEDENT, "Expected the end of an indented block")
        if not statements:
            raise self._error("Expected at least one statement in this block", self.current())
        return statements

    def _expect_statement_end(self) -> None:
        self.stream.expect_statement_end()

    def parse_statement(self) -> Node:
        if (
                self.check(
                    TokenType.INT,
                    TokenType.INT8,
                    TokenType.UINT8,
                    TokenType.INT64,
                    TokenType.INT32,
                    TokenType.BOOL,
                    TokenType.STR
                )
                and self.peek(1).type == TokenType.OPEN_PAREN
        ):
            # Scalar keyword + '(' is a cast statement, not a VarDecl.
            return self.parse_expr_stmt_or_assign()
        if self.check(TokenType.STAR):
            # '*' may start a pointer-typed VarDecl or a deref expression; try the type first and backtrack.
            mark = self.stream.mark()
            start_tok = self.current()
            try:
                parsed_type = type_exprs.parse_type(self.stream)
            except ParseError:
                parsed_type = None
            if parsed_type is not None and self.check(TokenType.IDENTIFIER):
                return self.parse_var_decl(var_type=parsed_type, start_tok=start_tok)
            self.stream.reset(mark)
        if self.check(
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
            start_tok = self.current()
            parsed_type = type_exprs.parse_type(self.stream)
            if isinstance(parsed_type, (ArrayTypeExpr, SliceTypeExpr)) and self.check(TokenType.OPEN_BRACKET):
                literal = expressions.parse_bracketed_literal(self.stream, parsed_type)
                return ExprStmt(expr=literal, line=start_tok.line, col=start_tok.col)
            return self.parse_var_decl(var_type=parsed_type, start_tok=start_tok)
        if self.check(TokenType.RETURN):
            return self.parse_return()
        if self.check(TokenType.IF):
            return self.parse_if()
        if self.check(TokenType.MATCH):
            return self.parse_match()
        if self.check(TokenType.WHILE):
            return self.parse_while()
        if self.check(TokenType.FOR):
            return self.parse_for()
        if self.check(TokenType.BREAK):
            return self.parse_break()
        if self.check(TokenType.CONTINUE):
            return self.parse_continue()
        if self.check(TokenType.IDENTIFIER) and self.peek(1).type == TokenType.IDENTIFIER:
            # IDENTIFIER IDENTIFIER is a struct-typed VarDecl.
            start_tok = self.current()
            parsed_type = type_exprs.parse_type(self.stream)
            return self.parse_var_decl(var_type=parsed_type, start_tok=start_tok)
        if (
                self.check(TokenType.IDENTIFIER)
                and self.peek(1).type == TokenType.DOT
                and self.peek(2).type == TokenType.IDENTIFIER
                and self.peek(3).type == TokenType.IDENTIFIER
        ):
            # IDENTIFIER . IDENTIFIER IDENTIFIER is a qualified struct-typed VarDecl.
            start_tok = self.current()
            parsed_type = type_exprs.parse_type(self.stream)
            return self.parse_var_decl(var_type=parsed_type, start_tok=start_tok)
        return self.parse_expr_stmt_or_assign()

    def parse_while(self) -> While:
        start_tok = self.expect(TokenType.WHILE, "Expected 'while'")
        condition = expressions.parse_expression(self.stream)
        self.expect(TokenType.COLON, "Expected ':' to start the while body")
        self.expect(TokenType.NEWLINE, "Expected a newline after ':'")
        body = self.parse_block()
        return While(condition=condition, body=body, line=start_tok.line, col=start_tok.col)

    def parse_for(self) -> Node:
        """Dispatch between three-clause and for-in forms."""
        if self.peek(1).type == TokenType.IDENTIFIER and self.peek(2).type in (TokenType.COMMA, TokenType.IN):
            return self.parse_for_in()
        return self._parse_for_three_clause()

    def _parse_for_three_clause(self) -> For:
        """`for init; cond; increment:`."""
        start_tok = self.expect(TokenType.FOR, "Expected 'for'")
        init = self._parse_for_init_clause()
        self.expect(TokenType.SEMICOLON, "Expected ';' after the for-loop's own init clause")
        condition = expressions.parse_expression(self.stream)
        self.expect(TokenType.SEMICOLON, "Expected ';' after the for-loop's own condition")
        increment = self._parse_for_increment_clause()
        self.expect(TokenType.COLON, "Expected ':' to start the for body")
        self.expect(TokenType.NEWLINE, "Expected a newline after ':'")
        body = self.parse_block()
        return For(
            init=init, condition=condition, increment=increment, body=body, line=start_tok.line, col=start_tok.col)

    def parse_for_in(self) -> ForIn:
        """`for a[, b] in iterable:`."""
        start_tok = self.expect(TokenType.FOR, "Expected 'for'")
        first_name_tok = self.expect(TokenType.IDENTIFIER, "Expected a loop variable name after 'for'")
        binding_names = [first_name_tok.val]
        if self.match(TokenType.COMMA):
            second_name_tok = self.expect(TokenType.IDENTIFIER, "Expected a second loop variable name after ','")
            binding_names.append(second_name_tok.val)
        self.expect(TokenType.IN, "Expected 'in' after the for-loop's own binding name(s)")
        iterable = expressions.parse_expression(self.stream)
        self.expect(TokenType.COLON, "Expected ':' to start the for body")
        self.expect(TokenType.NEWLINE, "Expected a newline after ':'")
        body = self.parse_block()
        return ForIn(binding_names=binding_names, iterable=iterable, body=body, line=start_tok.line, col=start_tok.col)

    def _parse_for_init_clause(self) -> Node:
        """Init clause: always a VarDecl."""
        start_tok = self.current()
        if self.check(
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
            parsed_type = type_exprs.parse_type(self.stream)
            return self.parse_var_decl(var_type=parsed_type, start_tok=start_tok)
        if self.check(TokenType.IDENTIFIER) and self.peek(1).type == TokenType.IDENTIFIER:
            parsed_type = type_exprs.parse_type(self.stream)
            return self.parse_var_decl(var_type=parsed_type, start_tok=start_tok)
        raise self._error(
            f"Expected a variable declaration (e.g. `int i = 0`) as the for-loop's "
            f"own init clause",
            start_tok,
        )

    def _parse_for_increment_clause(self) -> Node:
        """Increment clause: an assignment."""
        start_tok = self.current()
        statement = self.parse_expr_stmt_or_assign()
        if isinstance(statement, Assign):
            return statement
        raise self._error(
            f"Expected an assignment (e.g. `i += 1`) as the for-loop's own increment "
            f"clause",
            start_tok,
        )

    def parse_break(self) -> Break:
        tok = self.expect(TokenType.BREAK, "Expected 'break'")
        return Break(line=tok.line, col=tok.col)

    def parse_continue(self) -> Continue:
        tok = self.expect(TokenType.CONTINUE, "Expected 'continue'")
        return Continue(line=tok.line, col=tok.col)

    def parse_if(self) -> If:
        start_tok = self.expect(TokenType.IF, "Expected 'if'")
        return self._parse_if_body(start_tok)

    def parse_elif_as_if(self) -> If:
        start_tok = self.expect(TokenType.ELIF, "Expected 'elif'")
        return self._parse_if_body(start_tok)

    def _parse_if_body(self, start_tok: Token) -> If:
        """`KEYWORD condition : block`; shared by if/elif."""
        condition = self._parse_if_condition()
        self.expect(TokenType.COLON, "Expected ':' to start the if body")
        self.expect(TokenType.NEWLINE, "Expected a newline after ':'")
        then_body = self.parse_block()

        else_body = None
        self.skip_newlines()
        if self.check(TokenType.ELIF):
            else_body = [self.parse_elif_as_if()]
        elif self.match(TokenType.ELSE):
            self.expect(TokenType.COLON, "Expected ':' to start the else body")
            self.expect(TokenType.NEWLINE, "Expected a newline after ':'")
            else_body = self.parse_block()

        return If(condition=condition, then_body=then_body, else_body=else_body, line=start_tok.line, col=start_tok.col)

    def _parse_if_condition(self) -> Node:
        """A boolean expression (an `is` check in it may bind its subject with `as NAME`)."""
        expr = expressions.parse_expression(self.stream)
        if self.check(TokenType.AS):
            raise self._error("'as NAME' follows an 'is' check (`EXPR is T as NAME`)", self.current())
        return expr


    def parse_match(self) -> 'Match':
        """`match NAME:` / `match EXPR as NAME:` with `is T:` arms and optional `else:`."""
        start_tok = self.expect(TokenType.MATCH, "Expected 'match'")
        if self.check(TokenType.IDENTIFIER) and self.peek(1).type == TokenType.COLON:
            name_tok = self.advance()
            binding_name = name_tok.val
            subject = None
        else:
            subject = expressions.parse_expression(self.stream)
            self.expect(
                TokenType.AS,
                "Expected 'as NAME' after the match subject -- a non-bare-"
                "variable (or explicitly renamed) subject needs an explicit "
                "binding name to narrow, since it has no existing name of "
                "its own",
            )
            binding_tok = self.expect(TokenType.IDENTIFIER, "Expected a binding name after 'as'")
            binding_name = binding_tok.val
        self.expect(TokenType.COLON, "Expected ':' to start the match body")
        self.expect(TokenType.NEWLINE, "Expected a newline after ':'")
        self.skip_newlines()
        self.expect(TokenType.INDENT, "Expected an indented block")
        self.skip_newlines()

        arms: List[Tuple[Token, Union[str, QualifiedTypeExpr], List[Node]]] = []
        else_body: Optional[List[Node]] = None
        while not self.check(TokenType.DEDENT) and not self.at_end():
            if self.match(TokenType.ELSE):
                self.expect(TokenType.COLON, "Expected ':' to start the else body")
                self.expect(TokenType.NEWLINE, "Expected a newline after ':'")
                else_body = self.parse_block()
                self.skip_newlines()
                break
            arm_tok = self.expect(TokenType.IS, "Expected 'is' (a match arm) or 'else'")
            type_name = type_exprs.parse_qualifiable_type_name(self.stream, "a type name after 'is'")
            self.expect(TokenType.COLON, "Expected ':' to start this arm's body")
            self.expect(TokenType.NEWLINE, "Expected a newline after ':'")
            arm_body = self.parse_block()
            arms.append((arm_tok, type_name, arm_body))
            self.skip_newlines()

        self.expect(TokenType.DEDENT, "Expected the match body to end")

        if not arms:
            raise self._error(f"Expected at least one 'is' arm in this match", start_tok)

        checks = [(IsCheck(variable_name=binding_name, type_name=type_name, subject=subject if i == 0 else None,
                           line=arm_tok.line, col=arm_tok.col), arm_body)
                  for i, (arm_tok, type_name, arm_body) in enumerate(arms)]
        return Match(variable_name=binding_name, subject=subject, arms=checks, else_body=else_body,
                     line=arms[0][0].line, col=arms[0][0].col)

    def parse_var_decl(
            self,
            var_type: Optional[Union[str, 'ArrayTypeExpr', 'SliceTypeExpr']] = None,
            start_tok: Optional[Token] = None,
    ) -> VarDecl:
        """`T name [= expr]`; var_type may be pre-parsed."""
        if start_tok is None:
            start_tok = self.current()
        if var_type is None:
            var_type = type_exprs.parse_type(self.stream)
        name_tok = self.expect(TokenType.IDENTIFIER, "Expected a variable name")
        init = None
        if self.match(TokenType.ASSIGN):
            init = expressions.parse_expression(self.stream)
        return VarDecl(name=name_tok.val, var_type=var_type, init=init, line=start_tok.line, col=start_tok.col)

    def parse_return(self) -> Return:
        """`return [expr]`."""
        start_tok = self.expect(TokenType.RETURN)
        if self.check(TokenType.NEWLINE):
            return Return(value=None, line=start_tok.line, col=start_tok.col)
        value = expressions.parse_expression(self.stream)
        return Return(value=value, line=start_tok.line, col=start_tok.col)

    def parse_expr_stmt_or_assign(self) -> Node:
        """Expression statement, or assignment to a name, field, index, or dereference (parse the
        left side, then check for '=' or a compound operator)."""
        expr = expressions.parse_expression(self.stream)
        op_tok = self.current()
        compound_op = _COMPOUND_ASSIGN_OPS.get(op_tok.type)
        if op_tok.type == TokenType.ASSIGN or compound_op is not None:
            assignable = isinstance(expr, (Variable, Index, Field)) or (
                    isinstance(expr, Unary) and expr.op == UnaryOp.DEREFERENCE)
            if not assignable:
                raise self._error(f"Left-hand side of '{op_tok.val}' is not assignable", op_tok)
            self.advance()
            value = expressions.parse_expression(self.stream)
            return Assign(target=expr, value=value, op=compound_op, line=expr.line, col=expr.col)
        return ExprStmt(expr=expr, line=expr.line, col=expr.col)














