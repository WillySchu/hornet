"""Recursive-descent parser: tokens -> AST. Precedence climbing for binary operators."""

from typing import List, Tuple, Union

from lexer import Token, TokenType
from parser import expressions, statements, type_exprs
from parser.errors import ParseError
from parser.nodes import (
    ConstDecl, EnumDef, EnumMember, ExternFunctionDecl, FromImportDecl, Function, ImportDecl, IntrinsicDecl,
    MethodDef, Param, Program, StructDef, StructField, SumTypeDef, TypeAlias, stamp_file,
)
from parser.stream import TokenStream




def _default_import_qualifier(path: str) -> str:
    """Last path component, minus '.ht'."""
    last_component = path.rsplit('/', 1)[-1]
    if last_component.endswith('.ht'):
        last_component = last_component[:-len('.ht')]
    return last_component

















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

        body = statements.parse_block(self.stream)
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

        body = statements.parse_block(self.stream)
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



    def _expect_statement_end(self) -> None:
        self.stream.expect_statement_end()
































