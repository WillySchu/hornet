"""Declarations: a file from its top, and each thing declared there: imports, constants, types
(structs, enums, sum types, aliases), functions and methods, externs and intrinsics."""

from typing import List, Tuple, Union

from lexer import Token, TokenType
from parser import expressions, statements, type_exprs
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


def parse_program(p: TokenStream) -> Program:
    start_tok = p.current()
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
    p.skip_newlines()
    while not p.at_end():
        if p.check(TokenType.STRUCT):
            tok = p.current()
            raise p.error(
                f"Bare 'struct Name:' is no longer supported -- write 'type Name "
                f"struct:' instead",
                tok,
            )
        elif p.check(TokenType.TYPE):
            declaration = parse_type_declaration(p)
            if isinstance(declaration, StructDef):
                structs.append(declaration)
            elif isinstance(declaration, EnumDef):
                enums.append(declaration)
            elif isinstance(declaration, SumTypeDef):
                sum_types.append(declaration)
            else:
                type_aliases.append(declaration)
        elif p.check(TokenType.EXTERN):
            extern_functions.append(parse_extern_function(p))
        elif p.check(TokenType.INTRINSIC):
            intrinsics.append(parse_intrinsic(p))
        elif p.check(TokenType.CONST):
            consts.append(parse_const(p))
        elif p.check(TokenType.IMPORT):
            imports.append(parse_import(p))
        elif p.check(TokenType.FROM):
            from_imports.append(parse_from_import(p))
        else:
            functions.append(parse_function(p))
        p.skip_newlines()
    program = Program(
        functions=functions, structs=structs, type_aliases=type_aliases, sum_types=sum_types,
        extern_functions=extern_functions, imports=imports, from_imports=from_imports,
        intrinsics=intrinsics, consts=consts, enums=enums,
        line=start_tok.line, col=start_tok.col,
    )
    if start_tok.file:
        stamp_file(program, start_tok.file)
    return program


def parse_intrinsic(p: TokenStream) -> IntrinsicDecl:
    """`intrinsic T name(params)`."""
    start_tok = p.expect(TokenType.INTRINSIC, "Expected 'intrinsic'")
    return_type = None
    if type_exprs.check_starts_with_return_type(p):
        return_type = type_exprs.parse_type(p)
    name_tok = p.expect(TokenType.IDENTIFIER, "Expected a function name")
    p.expect(TokenType.OPEN_PAREN, "Expected '(' after function name")
    params = parse_params(p)
    p.expect(TokenType.CLOSE_PAREN, "Expected ')' after parameter list")
    return IntrinsicDecl(
        name=name_tok.val, original_name=name_tok.val, return_type=return_type, params=params,
        line=start_tok.line, col=start_tok.col,
    )


def parse_import(p: TokenStream) -> ImportDecl:
    """`import 'path' [as name]`."""
    start_tok = p.expect(TokenType.IMPORT, "Expected 'import'")
    path_tok = p.expect(TokenType.STRING, "Expected a quoted path after 'import'")
    path = p.literal_text(path_tok)
    if p.match(TokenType.AS):
        alias_tok = p.expect(TokenType.IDENTIFIER, "Expected a module name after 'as'")
        qualifier = alias_tok.val
    else:
        qualifier = _default_import_qualifier(path)  # an identifier, or modules.py rejects the file
    return ImportDecl(path=path, qualifier=qualifier, line=start_tok.line, col=start_tok.col)


def parse_from_import(p: TokenStream) -> FromImportDecl:
    """`from 'path' import a [as b], ...`."""
    start_tok = p.expect(TokenType.FROM, "Expected 'from'")
    path_tok = p.expect(TokenType.STRING, "Expected a quoted path after 'from'")
    path = p.literal_text(path_tok)
    p.expect(TokenType.IMPORT, "Expected 'import' after the path")
    names: List[Tuple[str, str]] = []
    while True:
        name_tok = p.expect(TokenType.IDENTIFIER, "Expected a name to import")
        if p.match(TokenType.AS):
            alias_tok = p.expect(TokenType.IDENTIFIER, "Expected an alias name after 'as'")
            names.append((name_tok.val, alias_tok.val))
        else:
            names.append((name_tok.val, name_tok.val))
        if not p.match(TokenType.COMMA):
            break
    return FromImportDecl(path=path, names=names, line=start_tok.line, col=start_tok.col)


def parse_type_declaration(p: TokenStream) -> Union[TypeAlias, StructDef, SumTypeDef, EnumDef]:
    """`type Name = T`, `type Name struct:`, `type Name enum:`, or `type Name is A | B`."""
    start_tok = p.expect(TokenType.TYPE, "Expected 'type' to start a type declaration")
    name_tok = p.expect(TokenType.IDENTIFIER, "Expected a name for this type declaration")
    if p.check(TokenType.STRUCT):
        p.advance()
        return _parse_struct_body(p, start_tok, name_tok)
    if p.check(TokenType.ENUM):
        p.advance()
        return _parse_enum_body(p, start_tok, name_tok)
    if p.check(TokenType.IS):
        p.advance()
        return _parse_sum_type_body(p, start_tok, name_tok)
    p.expect(TokenType.ASSIGN, "Expected '=' (for a type alias), 'struct' (for a struct declaration), "
                                  "'enum' (for an enum), or 'is' (for a sum type)")
    target_type = type_exprs.parse_type(p)
    p.expect(TokenType.NEWLINE, "Expected a newline after a type alias declaration")
    return TypeAlias(name=name_tok.val, target_type=target_type, line=start_tok.line, col=start_tok.col)


def _parse_sum_type_body(p: TokenStream, start_tok: Token, name_tok: Token) -> SumTypeDef:
    """`A | B ...` after `type Name is`; at least two variants."""
    first_line_tok = p.current()
    first_variant = type_exprs.parse_qualifiable_type_name(p, "a variant name")
    variants = [first_variant]
    while p.match(TokenType.PIPE):
        variant = type_exprs.parse_qualifiable_type_name(p, "a variant name after '|'")
        variants.append(variant)
    if len(variants) < 2:
        raise p.error(
            f"Expected at least one '|' and a second variant in sum type "
            f"'{name_tok.val}' -- a sum type needs at least two variants",
            first_line_tok,
        )
    p.expect(TokenType.NEWLINE, "Expected a newline after a sum type declaration")
    return SumTypeDef(name=name_tok.val, variants=variants, line=start_tok.line, col=start_tok.col)


def _parse_enum_body(p: TokenStream, start_tok: Token, name_tok: Token) -> EnumDef:
    """Enum body after `type Name enum`: one member name per line, then any methods."""
    p.expect(TokenType.COLON, "Expected ':' to start the enum body")
    p.expect(TokenType.NEWLINE, "Expected a newline after ':'")
    p.skip_newlines()
    p.expect(TokenType.INDENT, "Expected an indented enum body")
    p.skip_newlines()
    members: List[EnumMember] = []
    methods: List[MethodDef] = []
    while not p.check(TokenType.DEDENT) and not p.at_end():
        if p.check(TokenType.DEF):
            methods.append(parse_method_def(p))
        elif methods:
            raise p.error("An enum's members come before its methods", p.current())
        else:
            member_tok = p.expect(TokenType.IDENTIFIER, "Expected a member name")
            p.expect(
                TokenType.NEWLINE, "Expected a newline after a member name -- an enum lists one member per line")
            members.append(EnumMember(name=member_tok.val, line=member_tok.line, col=member_tok.col))
        p.skip_newlines()
    p.expect(TokenType.DEDENT, "Expected a dedent to end the enum body")
    return EnumDef(
        name=name_tok.val, members=members, methods=methods, line=start_tok.line, col=start_tok.col)


def _parse_struct_body(p: TokenStream, start_tok: Token, name_tok: Token) -> StructDef:
    """Struct body after `type Name struct`."""
    p.expect(TokenType.COLON, "Expected ':' to start the struct body")
    p.expect(TokenType.NEWLINE, "Expected a newline after ':'")
    p.skip_newlines()
    p.expect(TokenType.INDENT, "Expected an indented struct body")
    p.skip_newlines()
    fields: List[StructField] = []
    methods: List[MethodDef] = []
    while not p.check(TokenType.DEDENT) and not p.at_end():
        if p.check(TokenType.DEF):
            methods.append(parse_method_def(p))
        else:
            field_start_tok = p.current()
            field_type = type_exprs.parse_type(p)
            field_name_tok = p.expect(TokenType.IDENTIFIER, "Expected a field name")
            p.expect(TokenType.NEWLINE, "Expected a newline after a field declaration")
            fields.append(StructField(
                name=field_name_tok.val, field_type=field_type,
                line=field_start_tok.line, col=field_start_tok.col,
            ))
        p.skip_newlines()
    p.expect(TokenType.DEDENT, "Expected a dedent to end the struct body")
    if not fields:
        raise p.error(f"Expected at least one field in struct '{name_tok.val}'", p.current())
    return StructDef(name=name_tok.val, fields=fields, methods=methods, line=start_tok.line, col=start_tok.col)


def parse_function(p: TokenStream) -> Function:
    start_tok = p.expect(TokenType.DEF, "Expected 'def' to start a function definition")
    return_type = type_exprs.parse_return_type(p)
    name_tok = p.expect(TokenType.IDENTIFIER, "Expected a function name")
    p.expect(TokenType.OPEN_PAREN, "Expected '(' after function name")
    params = parse_params(p)
    p.expect(TokenType.CLOSE_PAREN, "Expected ')' after parameter list")
    p.expect(TokenType.COLON, "Expected ':' to start the function body")
    p.expect(TokenType.NEWLINE, "Expected a newline after ':'")

    body = statements.parse_block(p)
    return Function(
        name=name_tok.val, return_type=return_type, params=params, body=body,
        line=start_tok.line, col=start_tok.col,
    )


def parse_extern_function(p: TokenStream) -> ExternFunctionDecl:
    """`extern [T] name(params)`."""
    start_tok = p.expect(TokenType.EXTERN, "Expected 'extern' to start an external function declaration")
    return_type = type_exprs.parse_return_type(p)
    name_tok = p.expect(TokenType.IDENTIFIER, "Expected a function name")
    p.expect(TokenType.OPEN_PAREN, "Expected '(' after function name")
    params = parse_params(p)
    p.expect(TokenType.CLOSE_PAREN, "Expected ')' after parameter list")
    return ExternFunctionDecl(
        name=name_tok.val, return_type=return_type, params=params,
        line=start_tok.line, col=start_tok.col,
    )


def parse_method_def(p: TokenStream) -> MethodDef:
    """`def [T] name(receiver, ...):` or `def [T] name(*receiver, ...):`."""
    start_tok = p.expect(TokenType.DEF, "Expected 'def' to start a method definition")
    return_type = type_exprs.parse_return_type(p)
    name_tok = p.expect(TokenType.IDENTIFIER, "Expected a method name")
    p.expect(TokenType.OPEN_PAREN, "Expected '(' after method name")
    receiver_is_pointer = p.match(TokenType.STAR)
    receiver_tok = p.expect(TokenType.IDENTIFIER, "Expected a receiver name as a method's first parameter")
    params: List[Param] = []
    while p.match(TokenType.COMMA) and not p.check(TokenType.CLOSE_PAREN):  # trailing comma
        params.append(parse_param(p))
    p.expect(TokenType.CLOSE_PAREN, "Expected ')' after parameter list")
    p.expect(TokenType.COLON, "Expected ':' to start the method body")
    p.expect(TokenType.NEWLINE, "Expected a newline after ':'")

    body = statements.parse_block(p)
    return MethodDef(
        receiver_name=receiver_tok.val, name=name_tok.val, return_type=return_type, params=params, body=body,
        receiver_is_pointer=receiver_is_pointer, line=start_tok.line, col=start_tok.col,
    )


def parse_params(p: TokenStream) -> List[Param]:
    """`T name, ...` up to, not including, ')'; a trailing comma is allowed."""
    params = []
    if p.check(TokenType.CLOSE_PAREN):
        return params
    params.append(parse_param(p))
    while p.match(TokenType.COMMA) and not p.check(TokenType.CLOSE_PAREN):
        params.append(parse_param(p))
    return params


def parse_const(p: TokenStream) -> ConstDecl:
    """`const T NAME = expr`."""
    start_tok = p.expect(TokenType.CONST, "Expected 'const'")
    const_type = type_exprs.parse_type(p)
    name_tok = p.expect(TokenType.IDENTIFIER, "Expected a constant name")
    p.expect(TokenType.ASSIGN, "Expected '=' after a constant's name")
    value = expressions.parse_expression(p)
    p.expect(TokenType.NEWLINE, "Expected a newline after a constant declaration")
    return ConstDecl(name=name_tok.val, const_type=const_type, value=value, line=start_tok.line, col=start_tok.col)


def parse_param(p: TokenStream) -> Param:
    start_tok = p.current()
    param_type = type_exprs.parse_type(p)
    name_tok = p.expect(TokenType.IDENTIFIER, "Expected a parameter name")
    return Param(name=name_tok.val, type=param_type, line=start_tok.line, col=start_tok.col)
