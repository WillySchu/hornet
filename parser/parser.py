"""Recursive-descent parser: tokens -> AST. Precedence climbing for binary operators."""

from dataclasses import dataclass
from enum import auto, Enum
from typing import List, Optional, Tuple, Union

from diagnostics import quoted_text
from lexer import Token, TokenType, describe_token
from ops import BinaryOp, UnaryOp
from parser.errors import ParseError
from parser.nodes import (
    ArrayLiteral, ArrayTypeExpr, Assign, Binary, BoolLiteral, Break, ByteLiteral, Call, Cast, ConstDecl, Constant,
    Continue, DictLiteral, DictTypeExpr, EnumDef, EnumMember, ExprStmt, ExternFunctionDecl, Field, For, ForIn,
    FromImportDecl, Function, If, ImportDecl, Index, IntrinsicDecl, IsCheck, Match, MethodDef, Node, NoneLiteral,
    Param, PointerTypeExpr, Program, QualifiedTypeExpr, Return, Slice, SliceLiteral, SliceTypeExpr, StringLiteral,
    StructDef, StructField, SumTypeDef, TypeAlias, Unary, VarDecl, Variable, While, stamp_file,
)
from parser.stream import TokenStream


# Added to an error in the type of a literal that ArrayLiteral.could_be_multiplication marks.
READ_AS_A_TYPED_LITERAL = (
    " -- this is read as a typed array literal, like `[2][1]*P[...]`; to multiply an indexed literal by a "
    "value, write that literal's type, as in `[1]int[x][0] * ys[1]`")


def _default_import_qualifier(path: str) -> str:
    """Last path component, minus '.ht'."""
    last_component = path.rsplit('/', 1)[-1]
    if last_component.endswith('.ht'):
        last_component = last_component[:-len('.ht')]
    return last_component


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
        return_type = self.parse_type() if self._check_starts_with_return_type() else None
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
        target_type = self.parse_type()
        self.expect(TokenType.NEWLINE, "Expected a newline after a type alias declaration")
        return TypeAlias(name=name_tok.val, target_type=target_type, line=start_tok.line, col=start_tok.col)

    def _parse_sum_type_body(self, start_tok: Token, name_tok: Token) -> SumTypeDef:
        """`A | B ...` after `type Name is`; at least two variants."""
        first_line_tok = self.current()
        first_variant = self._parse_qualifiable_type_name("a variant name")
        variants = [first_variant]
        while self.match(TokenType.PIPE):
            variant = self._parse_qualifiable_type_name("a variant name after '|'")
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
                field_type = self.parse_type()
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

    def _check_starts_with_return_type(self) -> bool:
        """Whether a return type precedes the def's name."""
        return self.check(
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
                self.check(TokenType.IDENTIFIER) and self.peek(1).type == TokenType.IDENTIFIER
        ) or (
                self.check(TokenType.IDENTIFIER) and self.peek(1).type == TokenType.DOT
                and self.peek(2).type == TokenType.IDENTIFIER and self.peek(3).type == TokenType.IDENTIFIER
        )

    def _parse_return_type(self):
        """What precedes a def's or extern's name: a type, `never` (it doesn't return), or nothing."""
        if self.match(TokenType.NEVER):
            return 'never'
        return self.parse_type() if self._check_starts_with_return_type() else None

    def parse_function(self) -> Function:
        start_tok = self.expect(TokenType.DEF, "Expected 'def' to start a function definition")
        return_type = self._parse_return_type()
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
        return_type = self._parse_return_type()
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
        return_type = self._parse_return_type()
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
        const_type = self.parse_type()
        name_tok = self.expect(TokenType.IDENTIFIER, "Expected a constant name")
        self.expect(TokenType.ASSIGN, "Expected '=' after a constant's name")
        value = self.parse_expression()
        self.expect(TokenType.NEWLINE, "Expected a newline after a constant declaration")
        return ConstDecl(name=name_tok.val, const_type=const_type, value=value, line=start_tok.line, col=start_tok.col)

    def parse_param(self) -> Param:
        start_tok = self.current()
        param_type = self.parse_type()
        name_tok = self.expect(TokenType.IDENTIFIER, "Expected a parameter name")
        return Param(name=name_tok.val, type=param_type, line=start_tok.line, col=start_tok.col)

    def parse_type(self) -> Union[str, ArrayTypeExpr, SliceTypeExpr, PointerTypeExpr, DictTypeExpr]:
        if self.check(TokenType.DICT):
            dict_tok = self.advance()
            self.expect(TokenType.OPEN_BRACKET, "Expected '[' after 'dict'")
            key_type = self.parse_type()
            self.expect(TokenType.CLOSE_BRACKET, "Expected ']' after a dict's own key type")
            value_type = self.parse_type()
            return DictTypeExpr(key_type=key_type, value_type=value_type, line=dict_tok.line, col=dict_tok.col)
        if self.check(TokenType.STAR):
            star_tok = self.advance()
            pointee_type = self.parse_type()
            return PointerTypeExpr(pointee_type=pointee_type, line=star_tok.line, col=star_tok.col)
        if self.check(TokenType.OPEN_BRACKET):
            open_tok = self.advance()
            if self.check(TokenType.CLOSE_BRACKET):
                self.advance()
                element_type = self.parse_type()
                return SliceTypeExpr(element_type=element_type, line=open_tok.line, col=open_tok.col)
            if self.check(TokenType.NUMBER) and self.peek(1).type == TokenType.CLOSE_BRACKET:
                size_tok = self.advance()
                if '.' in size_tok.val:
                    raise self._error(f"Array size must be a whole number, got '{size_tok.val}'", size_tok)
                size = int(size_tok.val)
                if size <= 0:
                    raise self._error(f"Array size must be positive, got {size}", size_tok)
            else:
                # A constant expression, resolved during semantic analysis.
                start = self.pos
                try:
                    size = self.parse_expression()
                except ParseError:
                    if self.pos != start:
                        raise
                    raise self._error("Expected an array size (a positive integer or constant expression), "
                                      "or ']' for a slice type", self.current())
            self.expect(TokenType.CLOSE_BRACKET, "Expected ']' after array size")
            element_type = self.parse_type()
            return ArrayTypeExpr(size=size, element_type=element_type, line=open_tok.line, col=open_tok.col)
        if self.check(
                TokenType.INT,
                TokenType.INT8,
                TokenType.UINT8,
                TokenType.INT64,
                TokenType.INT32,
                TokenType.BOOL,
                TokenType.STR
        ):
            return self.advance().val
        if self.check(TokenType.IDENTIFIER):
            # Unvalidated here; semantic analysis resolves type names.
            name_tok = self.advance()
            if self.check(TokenType.DOT):
                self.advance()
                qualified_name_tok = self.expect(TokenType.IDENTIFIER, "Expected a type name after '.'")
                return QualifiedTypeExpr(
                    module=name_tok.val, name=qualified_name_tok.val, line=name_tok.line, col=name_tok.col)
            return name_tok.val
        tok = self.current()
        raise self._error(
            f"Expected a type ('int', 'int8', 'uint8', 'int64', 'bool', 'str', a "
            f"struct name, '[size]type', or '[]type'), got {describe_token(tok)}",
            tok,
        )

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
                parsed_type = self.parse_type()
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
            parsed_type = self.parse_type()
            if isinstance(parsed_type, (ArrayTypeExpr, SliceTypeExpr)) and self.check(TokenType.OPEN_BRACKET):
                literal = self._parse_bracketed_literal(parsed_type)
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
            parsed_type = self.parse_type()
            return self.parse_var_decl(var_type=parsed_type, start_tok=start_tok)
        if (
                self.check(TokenType.IDENTIFIER)
                and self.peek(1).type == TokenType.DOT
                and self.peek(2).type == TokenType.IDENTIFIER
                and self.peek(3).type == TokenType.IDENTIFIER
        ):
            # IDENTIFIER . IDENTIFIER IDENTIFIER is a qualified struct-typed VarDecl.
            start_tok = self.current()
            parsed_type = self.parse_type()
            return self.parse_var_decl(var_type=parsed_type, start_tok=start_tok)
        return self.parse_expr_stmt_or_assign()

    def parse_while(self) -> While:
        start_tok = self.expect(TokenType.WHILE, "Expected 'while'")
        condition = self.parse_expression()
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
        condition = self.parse_expression()
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
        iterable = self.parse_expression()
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
            parsed_type = self.parse_type()
            return self.parse_var_decl(var_type=parsed_type, start_tok=start_tok)
        if self.check(TokenType.IDENTIFIER) and self.peek(1).type == TokenType.IDENTIFIER:
            parsed_type = self.parse_type()
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
        expr = self.parse_expression()
        if self.check(TokenType.AS):
            raise self._error("'as NAME' follows an 'is' check (`EXPR is T as NAME`)", self.current())
        return expr

    def _parse_qualifiable_type_name(
            self,
            expected_message: str) -> Union[str, QualifiedTypeExpr, ArrayTypeExpr, SliceTypeExpr, PointerTypeExpr]:
        """A possibly qualified type name, builtin type keyword, `none`, or array/slice/pointer/dict type."""
        if self.check(TokenType.STAR, TokenType.OPEN_BRACKET, TokenType.DICT):
            return self.parse_type()
        if self.check(
                TokenType.INT,
                TokenType.INT8,
                TokenType.UINT8,
                TokenType.INT64,
                TokenType.INT32,
                TokenType.BOOL,
                TokenType.STR,
                TokenType.NONE
        ):  # `none`: a sum type's payload-free variant
            return self.advance().val
        name_tok = self.expect(TokenType.IDENTIFIER, f"Expected {expected_message}")
        if self.check(TokenType.DOT):
            self.advance()
            qualified_name_tok = self.expect(TokenType.IDENTIFIER, "Expected a type name after '.'")
            return QualifiedTypeExpr(
                module=name_tok.val, name=qualified_name_tok.val, line=name_tok.line, col=name_tok.col)
        return name_tok.val

    def parse_match(self) -> 'Match':
        """`match NAME:` / `match EXPR as NAME:` with `is T:` arms and optional `else:`."""
        start_tok = self.expect(TokenType.MATCH, "Expected 'match'")
        if self.check(TokenType.IDENTIFIER) and self.peek(1).type == TokenType.COLON:
            name_tok = self.advance()
            binding_name = name_tok.val
            subject = None
        else:
            subject = self.parse_expression()
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
            type_name = self._parse_qualifiable_type_name("a type name after 'is'")
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
            var_type = self.parse_type()
        name_tok = self.expect(TokenType.IDENTIFIER, "Expected a variable name")
        init = None
        if self.match(TokenType.ASSIGN):
            init = self.parse_expression()
        return VarDecl(name=name_tok.val, var_type=var_type, init=init, line=start_tok.line, col=start_tok.col)

    def parse_return(self) -> Return:
        """`return [expr]`."""
        start_tok = self.expect(TokenType.RETURN)
        if self.check(TokenType.NEWLINE):
            return Return(value=None, line=start_tok.line, col=start_tok.col)
        value = self.parse_expression()
        return Return(value=value, line=start_tok.line, col=start_tok.col)

    def parse_expr_stmt_or_assign(self) -> Node:
        """Expression statement, or assignment to a name, field, index, or dereference (parse the
        left side, then check for '=' or a compound operator)."""
        expr = self.parse_expression()
        op_tok = self.current()
        compound_op = _COMPOUND_ASSIGN_OPS.get(op_tok.type)
        if op_tok.type == TokenType.ASSIGN or compound_op is not None:
            assignable = isinstance(expr, (Variable, Index, Field)) or (
                    isinstance(expr, Unary) and expr.op == UnaryOp.DEREFERENCE)
            if not assignable:
                raise self._error(f"Left-hand side of '{op_tok.val}' is not assignable", op_tok)
            self.advance()
            value = self.parse_expression()
            return Assign(target=expr, value=value, op=compound_op, line=expr.line, col=expr.col)
        return ExprStmt(expr=expr, line=expr.line, col=expr.col)

    def parse_expression(self) -> Node:
        return self.parse_binary()

    def parse_binary(self, min_prec: int = 0) -> Node:
        """Precedence climbing over _BINARY_OPS."""
        left = self.parse_unary()
        while True:
            if self.check(TokenType.NOT) and self.peek(1).type == TokenType.IN:
                in_info = _BINARY_OPS[TokenType.IN]
                if in_info.precedence < min_prec:
                    break
                not_tok = self.advance()
                self.advance()
                right = self.parse_binary(in_info.precedence + 1)
                membership = Binary(op=BinaryOp.IN, left=left, right=right, line=left.line, col=left.col)
                left = Unary(op=UnaryOp.NOT, operand=membership, line=not_tok.line, col=not_tok.col)
                continue
            if self.check(TokenType.IS):  # `left is T`, `left is not T`: T is a type, not an expression
                if _IS_PRECEDENCE < min_prec:
                    break
                is_tok = self.advance()
                negated = self.match(TokenType.NOT)
                type_name = self._parse_qualifiable_type_name("a type name after 'is'")
                at = left if isinstance(left, Variable) else is_tok
                if self.check(TokenType.AS):  # `left is T as NAME`: NAME is a copy of left
                    if negated:
                        raise self._error("'as NAME' can't follow 'is not': there would be nothing of that type "
                                          "to bind", self.current())
                    self.advance()
                    binding_tok = self.expect(TokenType.IDENTIFIER, "Expected a binding name after 'as'")
                    left = IsCheck(variable_name=binding_tok.val, type_name=type_name, subject=left,
                                   line=at.line, col=at.col)
                elif isinstance(left, Variable):
                    left = IsCheck(variable_name=left.name, type_name=type_name, line=at.line, col=at.col)
                else:
                    left = IsCheck(variable_name=None, type_name=type_name, subject=left, line=at.line, col=at.col)
                if negated:
                    left = Unary(op=UnaryOp.NOT, operand=left, line=left.line, col=left.col)
                continue
            op_info = _BINARY_OPS.get(self.current().type)
            if op_info is None or op_info.precedence < min_prec:
                break
            self.advance()
            next_min_prec = (
                op_info.precedence + 1
                if op_info.associativity == Associativity.LEFT
                else op_info.precedence
            )
            right = self.parse_binary(next_min_prec)
            left = Binary(op=op_info.op, left=left, right=right, line=left.line, col=left.col)
        return left

    def parse_unary(self) -> Node:
        if self.check(TokenType.NOT):
            op_tok = self.advance()
            operand = self.parse_binary(_NOT_OPERAND_PRECEDENCE)
            return Unary(op=UnaryOp.NOT, operand=operand, line=op_tok.line, col=op_tok.col)
        if self.check(*_UNARY_OPS):
            op_tok = self.advance()
            # Recurse on parse_unary so prefix operators chain.
            operand = self.parse_unary()
            return Unary(op=_UNARY_OPS[op_tok.type], operand=operand, line=op_tok.line, col=op_tok.col)
        return self.parse_postfix()

    def parse_postfix(self) -> Node:
        """Apply `[...]`, `.name`, `.name(...)` suffixes."""
        expr = self.parse_primary()
        while self.check(TokenType.OPEN_BRACKET, TokenType.DOT):
            if self.match(TokenType.DOT):
                name_tok = self.expect(TokenType.IDENTIFIER, "Expected a field name after '.'")
                if self.match(TokenType.OPEN_PAREN):
                    args, kwargs = self.parse_receiver_call_args()
                    self.expect(TokenType.CLOSE_PAREN, "Expected ')' after call arguments")
                    expr = Call(
                        name=name_tok.val, args=args, kwargs=kwargs, receiver=expr, line=expr.line, col=expr.col)
                else:
                    expr = Field(base=expr, name=name_tok.val, line=expr.line, col=expr.col)
            else:
                self.advance()
                expr = self.parse_index_or_slice(expr)
        return expr

    def parse_receiver_call_args(self) -> Tuple[List[Node], Optional[List[Tuple[str, Node]]]]:
        """Positional or named args after `.name(`."""
        args: List[Node] = []
        kwargs: Optional[List[Tuple[str, Node]]] = None
        if self.check(TokenType.CLOSE_PAREN):
            return args, kwargs
        while True:
            start_tok = self.current()
            if self.check(TokenType.IDENTIFIER) and self.peek(1).type == TokenType.ASSIGN:
                field_name = self.advance().val
                self.advance()
                value = self.parse_expression()
                if args:
                    raise self._error(
                        f"Cannot mix positional and named arguments in a call -- "
                        f"'{field_name}=...' follows a positional argument",
                        start_tok,
                    )
                if kwargs is None:
                    kwargs = []
                kwargs.append((field_name, value))
            else:
                value = self.parse_expression()
                if kwargs is not None:
                    raise self._error(
                        f"Cannot mix positional and named arguments in a call -- a positional "
                        f"argument follows a named one",
                        start_tok,
                    )
                args.append(value)
            if not self.match(TokenType.COMMA) or self.check(TokenType.CLOSE_PAREN):  # trailing comma
                break
        return args, kwargs

    def parse_index_or_slice(self, array_expr: Node) -> Node:
        """Index or Slice after '['."""
        if self.check(TokenType.COLON):
            self.advance()
            high = None if self.check(TokenType.CLOSE_BRACKET) else self.parse_expression()
            self.expect(TokenType.CLOSE_BRACKET, "Expected ']' to close a slice expression")
            return Slice(array=array_expr, low=None, high=high, line=array_expr.line, col=array_expr.col)

        first = self.parse_expression()

        if self.match(TokenType.COLON):
            high = None if self.check(TokenType.CLOSE_BRACKET) else self.parse_expression()
            self.expect(TokenType.CLOSE_BRACKET, "Expected ']' to close a slice expression")
            return Slice(array=array_expr, low=first, high=high, line=array_expr.line, col=array_expr.col)

        self.expect(TokenType.CLOSE_BRACKET, "Expected ']' after array index")
        return Index(array=array_expr, index=first, line=array_expr.line, col=array_expr.col)

    def parse_primary(self) -> Node:
        if self.check(TokenType.NUMBER):
            tok = self.advance()
            if '.' in tok.val:  # (the lexer takes `1.5` as one number, so that this can say so)
                raise self._error(
                    f"A number can't have a fractional part ('{tok.val}') -- there are no floating-point types "
                    f"yet, only integers", tok)
            return Constant(value=int(tok.val), line=tok.line, col=tok.col)
        if self.check(TokenType.TRUE, TokenType.FALSE):
            tok = self.advance()
            return BoolLiteral(value=(tok.type == TokenType.TRUE), line=tok.line, col=tok.col)
        if self.check(TokenType.NONE):
            tok = self.advance()
            return NoneLiteral(line=tok.line, col=tok.col)
        if self.check(TokenType.STRING):
            tok = self.advance()
            return StringLiteral(value=self._literal_text(tok), line=tok.line, col=tok.col)
        if self.check(TokenType.BYTE):
            tok = self.advance()
            resolved = self._literal_text(tok)
            if len(resolved) != 1 or ord(resolved) > 255:
                raise self._error(
                    f"A byte literal must resolve to exactly one byte (0-255), got "
                    f"{quoted_text(resolved)}",
                    tok,
                )
            return ByteLiteral(value=ord(resolved), line=tok.line, col=tok.col)
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
            return self.parse_cast()
        if self.check(TokenType.DICT):
            parsed_type = self.parse_type()
            return self.parse_dict_literal(parsed_type)
        if self._looks_like_typed_literal():
            could_be_multiplication = self.pos in self._could_be_multiplication
            try:
                parsed_type = self.parse_type()
            except ParseError as problem:
                if not could_be_multiplication:
                    raise
                raise ParseError(problem.message + READ_AS_A_TYPED_LITERAL, problem.file, problem.line,
                                 problem.col) from None
            literal = self._parse_bracketed_literal(parsed_type)
            if could_be_multiplication and isinstance(literal, ArrayLiteral):
                literal.could_be_multiplication = True
            return literal
        if self.check(TokenType.OPEN_BRACKET):
            return self.parse_array_literal()
        if self.check(TokenType.IDENTIFIER):
            if self.peek(1).type == TokenType.OPEN_PAREN:
                return self.parse_call()
            tok = self.advance()
            return Variable(name=tok.val, line=tok.line, col=tok.col)
        if self.match(TokenType.OPEN_PAREN):
            expr = self.parse_expression()
            self.expect(TokenType.CLOSE_PAREN, "Expected ')' to close grouped expression")
            return expr
        tok = self.current()
        raise self._error(f"Expected an expression, got {describe_token(tok)}", tok)

    def _looks_like_typed_literal(self) -> bool:
        """Whether '[' starts a typed literal (`[3]int[...]`, `[N][]T[...]`, `[]dict[K]V[...]`) rather
        than an untyped one: one or more bracket groups followed by the start of a type."""
        if not self.check(TokenType.OPEN_BRACKET):
            return False
        k = 0
        groups = 0
        empty_group = False
        while self.peek(k).type == TokenType.OPEN_BRACKET:
            groups += 1
            empty_group = empty_group or self.peek(k + 1).type == TokenType.CLOSE_BRACKET
            depth = 0
            while True:
                t = self.peek(k).type
                if t == TokenType.EOF:
                    return False
                depth += (t == TokenType.OPEN_BRACKET) - (t == TokenType.CLOSE_BRACKET)
                k += 1
                if depth == 0:
                    break
        if self.peek(k).type != TokenType.STAR:
            return self.peek(k).type in _TYPE_START_TOKENS
        # `*` might instead be a multiplication (`[x][0] * y`): a pointer element type must run on to
        # the elements' '['.
        while self.peek(k).type == TokenType.STAR:
            k += 1
        if self.peek(k).type in (TokenType.OPEN_BRACKET, TokenType.DICT):
            return True
        if self.peek(k).type == TokenType.IDENTIFIER:
            if groups >= 2 and not empty_group:
                # `[N][M]*P[...]` has the tokens of `[x][0] * ys[1]`, an indexed literal times an
                # index. It is the typed literal: a literal without a type can't be indexed.
                self._could_be_multiplication.add(self.pos)
            k += 1
            if self.peek(k).type == TokenType.DOT and self.peek(k + 1).type == TokenType.IDENTIFIER:
                k += 2
        elif self.peek(k).type in _TYPE_START_TOKENS:
            k += 1
        else:
            return False
        return self.peek(k).type == TokenType.OPEN_BRACKET

    def _parse_bracketed_literal(self, parsed_type: Union[str, 'ArrayTypeExpr', 'SliceTypeExpr']) -> Node:
        """Bracketed elements after a pre-parsed literal type."""
        if isinstance(parsed_type, SliceTypeExpr):
            elements = self.parse_array_literal().elements
            return SliceLiteral(element_type=parsed_type.element_type, elements=elements,
                                line=parsed_type.line, col=parsed_type.col)
        return self.parse_array_literal(type_expr=parsed_type)

    def parse_array_literal(self, type_expr: Optional['ArrayTypeExpr'] = None) -> ArrayLiteral:
        """`[e1, ...]`, optionally typed."""
        open_tok = self.expect(TokenType.OPEN_BRACKET)
        elements = []
        if not self.check(TokenType.CLOSE_BRACKET):
            elements.append(self.parse_expression())
            while self.match(TokenType.COMMA) and not self.check(TokenType.CLOSE_BRACKET):  # trailing comma
                elements.append(self.parse_expression())
        self.expect(TokenType.CLOSE_BRACKET, "Expected ']' to close array literal")
        return ArrayLiteral(elements=elements, type_expr=type_expr, line=open_tok.line, col=open_tok.col)

    def parse_dict_literal(self, dict_type: DictTypeExpr) -> DictLiteral:
        """`{k: v, ...}` after a parsed dict type."""
        open_tok = self.expect(TokenType.OPEN_BRACE)
        self.skip_newlines()
        entries = []
        if not self.check(TokenType.CLOSE_BRACE):
            entries.append(self._parse_dict_entry())
            self.skip_newlines()
            while self.match(TokenType.COMMA):
                self.skip_newlines()
                if self.check(TokenType.CLOSE_BRACE):
                    break
                entries.append(self._parse_dict_entry())
                self.skip_newlines()
        self.expect(TokenType.CLOSE_BRACE, "Expected '}' to close dict literal")
        return DictLiteral(
            key_type=dict_type.key_type, value_type=dict_type.value_type,
            entries=entries, line=open_tok.line, col=open_tok.col,
        )

    def _parse_dict_entry(self) -> Tuple[Node, Node]:
        """`key: value`."""
        key = self.parse_expression()
        self.expect(TokenType.COLON, "Expected ':' between a dict entry's key and value")
        value = self.parse_expression()
        return key, value

    def parse_call(self) -> Call:
        """`name(args)` or `name(f=v, ...)`; no mixing."""
        name_tok = self.expect(TokenType.IDENTIFIER)
        self.expect(TokenType.OPEN_PAREN, "Expected '(' to start a call's argument list")
        args: List[Node] = []
        kwargs: Optional[List[Tuple[str, Node]]] = None
        if not self.check(TokenType.CLOSE_PAREN):
            while True:
                start_tok = self.current()
                if self.check(TokenType.IDENTIFIER) and self.peek(1).type == TokenType.ASSIGN:
                    field_name = self.advance().val
                    self.advance()
                    value = self.parse_expression()
                    if args:
                        raise self._error(
                            f"Cannot mix positional and named arguments in a call -- "
                            f"'{field_name}=...' follows a positional argument",
                            start_tok,
                        )
                    if kwargs is None:
                        kwargs = []
                    kwargs.append((field_name, value))
                else:
                    value = self.parse_expression()
                    if kwargs is not None:
                        raise self._error(
                            f"Cannot mix positional and named arguments in a call -- a positional "
                            f"argument follows a named one",
                            start_tok,
                        )
                    args.append(value)
                if not self.match(TokenType.COMMA) or self.check(TokenType.CLOSE_PAREN):  # trailing comma
                    break
        self.expect(TokenType.CLOSE_PAREN, "Expected ')' to close a call's argument list")
        return Call(name=name_tok.val, args=args, kwargs=kwargs, line=name_tok.line, col=name_tok.col)

    def parse_cast(self) -> Cast:
        """`T(expr)`."""
        type_tok = self.advance()
        self.expect(TokenType.OPEN_PAREN, "Expected '(' to start a cast's argument")
        expr = self.parse_expression()
        self.expect(TokenType.CLOSE_PAREN, "Expected ')' to close a cast's argument")
        return Cast(target_type=type_tok.val, expr=expr, line=type_tok.line, col=type_tok.col)
