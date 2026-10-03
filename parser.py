"""Recursive-descent parser: tokens -> AST. Precedence climbing for binary operators."""

import argparse
import itertools
from dataclasses import dataclass, field, fields
from enum import auto, Enum
from typing import Any, List, Optional, Tuple, Union

from lexer import Token, TokenType, describe_token, describe_token_type, lex
from ops import BinaryOp, UnaryOp
from diagnostics import CompileError


# AST nodes

_PRETTY_MAX_WIDTH = 88
_HIDDEN_FIELDS = {'line', 'col', 'file', 'nid'}
_PRETTY_INDENT = "    "


def _pretty_scalar(value: Any) -> str:
    """Render a non-Node field; operators render as their symbol."""
    if isinstance(value, (UnaryOp, BinaryOp)):
        value = value.symbol()
    return repr(value)


def _pretty_value(value: Any, indent: int) -> str:
    """Render a Node, list, or scalar at depth `indent`; first line unindented."""
    if isinstance(value, Node):
        return _pretty_node(value, indent)
    if isinstance(value, list):
        return _pretty_list(value, indent)
    return _pretty_scalar(value)


def _pretty_list(items: list, indent: int) -> str:
    """Render a list on one line if it fits, else one item per line."""
    if not items:
        return "[]"
    rendered = [_pretty_value(item, indent + 1) for item in items]
    if not any('\n' in r for r in rendered):
        candidate = f"[{', '.join(rendered)}]"
        if indent * len(_PRETTY_INDENT) + len(candidate) <= _PRETTY_MAX_WIDTH:
            return candidate
    inner = ",\n".join(f"{_PRETTY_INDENT * (indent + 1)}{r}" for r in rendered)
    return f"[\n{inner},\n{_PRETTY_INDENT * indent}]"


def _pretty_node(node: 'Node', indent: int) -> str:
    """Render `ClassName(field=value, ...)` via dataclasses.fields."""
    class_name = type(node).__name__
    field_names = [f.name for f in fields(node) if f.name not in _HIDDEN_FIELDS]
    if not field_names:
        return f"{class_name}()"

    rendered = {name: _pretty_value(getattr(node, name), indent + 1) for name in field_names}
    if not any('\n' in r for r in rendered.values()):
        candidate = f"{class_name}({', '.join(f'{n}={rendered[n]}' for n in field_names)})"
        if indent * len(_PRETTY_INDENT) + len(candidate) <= _PRETTY_MAX_WIDTH:
            return candidate

    inner = ",\n".join(f"{_PRETTY_INDENT * (indent + 1)}{n}={rendered[n]}" for n in field_names)
    return f"{class_name}(\n{inner},\n{_PRETTY_INDENT * indent})"


_node_numbers = itertools.count()


@dataclass
class Node:
    """AST base. pretty() renders any node generically. Every node gets a number (`nid`) when it is
    created: passes key per-node facts on it, never on object identity."""
    line: int = field(default=0, kw_only=True, compare=False, repr=False)
    col: int = field(default=0, kw_only=True, compare=False, repr=False)
    file: Optional[str] = field(default=None, kw_only=True, compare=False, repr=False)
    nid: int = field(default=-1, kw_only=True, compare=False, repr=False)

    def __post_init__(self):
        self.nid = next(_node_numbers)

    def pretty(self) -> str:
        return _pretty_node(self, indent=0)


@dataclass
class Constant(Node):
    """A number literal. One with a fraction (`1.5`) parses and is rejected by semantic analysis:
    there is no floating-point type."""
    value: Union[int, float]


@dataclass
class BoolLiteral(Node):
    """`true` / `false`."""
    value: bool


@dataclass
class NoneLiteral(Node):
    """`none`: a pointer to nothing, or a sum type's `none` variant. Typed Type.NONE."""


@dataclass
class StringLiteral(Node):
    """`'...'`; `value` is unescaped. Strings are bytes, not Unicode."""
    value: str


@dataclass
class ByteLiteral(Node):
    """`"a"`: a single uint8."""
    value: int


@dataclass
class Variable(Node):
    """Variable reference."""
    name: str


@dataclass
class ArrayLiteral(Node):
    """`[e1, ...]` or typed `[N]T[...]` / `[]T[...]`."""
    elements: List[Node] = field(default_factory=list)
    type_expr: Optional['ArrayTypeExpr'] = None


@dataclass
class DictLiteral(Node):
    """`dict[K]V{k: v, ...}`."""
    key_type: Union[str, 'ArrayTypeExpr', 'SliceTypeExpr', 'PointerTypeExpr', 'DictTypeExpr']
    value_type: Union[str, 'ArrayTypeExpr', 'SliceTypeExpr', 'PointerTypeExpr', 'DictTypeExpr']
    entries: List[Tuple[Node, Node]] = field(default_factory=list)


@dataclass
class Index(Node):
    """`array[index]`; multi-dimensional indexing nests."""
    array: Node
    index: Node


@dataclass
class SliceLiteral(Node):
    """`[]T[e1, ...]`: a new slice holding the elements."""
    element_type: Any
    elements: List[Node]


@dataclass
class Slice(Node):
    """`array[low:high]`: a view over [low, high); either bound optional."""
    array: Node
    low: Optional[Node] = None
    high: Optional[Node] = None


@dataclass
class Call(Node):
    """`name(args)` or `name(f=v, ...)`; with a `receiver`, a method call or a module-qualified one.
    Struct literals share this shape."""
    name: str
    args: List[Node] = field(default_factory=list)
    kwargs: Optional[List[Tuple[str, Node]]] = None
    receiver: Optional[Node] = None


@dataclass
class Unary(Node):
    op: UnaryOp
    operand: Node


@dataclass
class Cast(Node):
    """`T(expr)`: explicit scalar cast."""
    target_type: str
    expr: Node


@dataclass
class Binary(Node):
    op: BinaryOp
    left: Node
    right: Node


@dataclass
class Return(Node):
    """`return [expr]`."""
    value: Optional[Node] = None


@dataclass
class ArrayTypeExpr(Node):
    """`[N]T`; nests row-major. `size` is an int, or a constant expression that semantic analysis
    evaluates."""
    size: Union[int, 'Node']
    element_type: Union[str, 'ArrayTypeExpr', 'SliceTypeExpr']


@dataclass
class SliceTypeExpr(Node):
    """`[]T`."""
    element_type: Union[str, ArrayTypeExpr, 'SliceTypeExpr']


@dataclass
class PointerTypeExpr(Node):
    """`*T`."""
    pointee_type: Union[str, ArrayTypeExpr, SliceTypeExpr, 'PointerTypeExpr']


@dataclass
class DictTypeExpr(Node):
    """`dict[K]V`."""
    key_type: Union[str, ArrayTypeExpr, SliceTypeExpr, PointerTypeExpr, 'DictTypeExpr']
    value_type: Union[str, ArrayTypeExpr, SliceTypeExpr, PointerTypeExpr, 'DictTypeExpr']


@dataclass
class QualifiedTypeExpr(Node):
    """`module.Name`; semantic analysis resolves it to the declaration's key (scopes.py)."""
    module: str
    name: str


@dataclass
class VarDecl(Node):
    """`T name [= init]`."""
    name: str
    var_type: Union[str, ArrayTypeExpr, SliceTypeExpr]
    init: Optional[Node] = None


@dataclass
class Assign(Node):
    """`target = value`, or `target op= value` (`op` the operator); the target is a name, field,
    index, or dereference, as written."""
    target: Node
    value: Node
    op: Optional[BinaryOp] = None


@dataclass
class Field(Node):
    """`base.name`; pointers auto-deref."""
    base: Node
    name: str


@dataclass
class StructField(Node):
    """Struct field `T name`; zero-initialized."""
    name: str
    field_type: Union[str, 'ArrayTypeExpr', 'SliceTypeExpr']


@dataclass
class MethodDef(Node):
    """Struct method; first param is the untyped receiver (`*name` for a pointer receiver)."""
    receiver_name: str
    name: str
    return_type: Optional[Union[str, ArrayTypeExpr, SliceTypeExpr]]
    params: List['Param'] = field(default_factory=list)
    body: List[Node] = field(default_factory=list)
    receiver_is_pointer: bool = False


@dataclass
class StructDef(Node):
    """`type Name struct:`; field order fixes layout."""
    name: str
    fields: List[StructField] = field(default_factory=list)
    methods: List[MethodDef] = field(default_factory=list)


@dataclass
class ExprStmt(Node):
    """Expression statement."""
    expr: Node


@dataclass
class IsCheck(Node):
    """If-condition `NAME is T` or `EXPR is T as NAME`. Not a general expression."""
    variable_name: str
    type_name: Union[str, QualifiedTypeExpr, ArrayTypeExpr, SliceTypeExpr, PointerTypeExpr]
    subject: Optional[Node] = None


@dataclass
class If(Node):
    """`if`; `elif` nests in else_body."""
    condition: Node
    then_body: List[Node]
    else_body: Optional[List[Node]] = None


@dataclass
class Match(Node):
    """`match NAME:` / `match EXPR as NAME:`. Each arm is (`NAME is T` check, body), tried in order;
    the first arm's check carries the subject expression, if any. Positioned at the first arm."""
    variable_name: str
    subject: Optional[Node]
    arms: List[Tuple[Node, List[Node]]]
    else_body: Optional[List[Node]] = None


@dataclass
class While(Node):
    """`while cond:`."""
    condition: Node
    body: List[Node]


@dataclass
class For(Node):
    """`for init; cond; increment:`."""
    init: Node
    condition: Node
    increment: Node
    body: List[Node]


@dataclass
class ForIn(Node):
    """`for a[, b] in iterable:`."""
    binding_names: List[str]
    iterable: Node
    body: List[Node]


@dataclass
class Break(Node):
    """Exits the innermost loop."""


@dataclass
class Continue(Node):
    """Continues the innermost loop."""


@dataclass
class Param(Node):
    """Parameter `T name`."""
    name: str
    type: Union[str, ArrayTypeExpr, SliceTypeExpr]


@dataclass
class Function(Node):
    """`def [T] name(params):`; return_type None means no value, 'never' that it doesn't return."""
    name: str
    return_type: Optional[Union[str, ArrayTypeExpr, SliceTypeExpr]]
    params: List[Param] = field(default_factory=list)
    body: List[Node] = field(default_factory=list)


@dataclass
class ExternFunctionDecl(Node):
    """`extern [T] name(params)`: a linker symbol, never mangled."""
    name: str
    return_type: Optional[Union[str, ArrayTypeExpr, SliceTypeExpr]]
    params: List[Param] = field(default_factory=list)


@dataclass
class IntrinsicDecl(Node):
    """`intrinsic T name(params)`: compiler-provided operation. original_name survives mangling."""
    name: str
    original_name: str
    return_type: Optional[Union[str, ArrayTypeExpr, SliceTypeExpr, PointerTypeExpr]]
    params: List[Param] = field(default_factory=list)


@dataclass
class ConstDecl(Node):
    """Top-level `const T NAME = expr`; expr must be a compile-time constant."""
    name: str
    const_type: Union[str, ArrayTypeExpr, SliceTypeExpr]
    value: Node


@dataclass
class TypeAlias(Node):
    """`type Name = T`: an interchangeable alias."""
    name: str
    target_type: Union[str, ArrayTypeExpr, SliceTypeExpr]


@dataclass
class SumTypeDef(Node):
    """`type Name is A | B`: tagged sum type. Variant order fixes discriminants."""
    name: str
    variants: List[Union[str, QualifiedTypeExpr, ArrayTypeExpr, SliceTypeExpr, PointerTypeExpr]] = field(default_factory=list)


@dataclass
class ImportDecl(Node):
    """`import 'path' [as name]`."""
    path: str
    qualifier: str


@dataclass
class FromImportDecl(Node):
    """`from 'path' import a [as b], ...`."""
    path: str
    names: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class Program(Node):
    functions: List[Function] = field(default_factory=list)
    structs: List[StructDef] = field(default_factory=list)
    type_aliases: List[TypeAlias] = field(default_factory=list)
    sum_types: List[SumTypeDef] = field(default_factory=list)
    extern_functions: List[ExternFunctionDecl] = field(default_factory=list)
    imports: List[ImportDecl] = field(default_factory=list)
    from_imports: List[FromImportDecl] = field(default_factory=list)
    intrinsics: List[IntrinsicDecl] = field(default_factory=list)
    consts: List[ConstDecl] = field(default_factory=list)

    def __repr__(self) -> str:
        return self.pretty()


# Parser

def stamp_file(node, file: str) -> None:
    """Set `file` on `node` and every descendant that lacks one."""
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, Node):
            if n.file is None:
                n.file = file
            stack.extend(getattr(n, f.name) for f in fields(n))
        elif isinstance(n, (list, tuple)):
            stack.extend(n)


class ParseError(CompileError):
    """Malformed input."""


# Keyed by the character after the backslash; \xNN is handled separately.
_ESCAPE_SEQUENCES = {
    'n': '\n',
    't': '\t',
    'r': '\r',
    '0': '\0',
    "'": "'",
    '"': '"',
    '\\': '\\',
}

_HEX_DIGITS = '0123456789abcdefABCDEF'


def _unescape_quoted_literal(raw: str) -> str:
    """Strip quotes and resolve escape sequences."""
    inner = raw[1:-1]
    chars = []
    i = 0
    while i < len(inner):
        ch = inner[i]
        if ch == '\\' and i + 1 < len(inner):
            nxt = inner[i + 1]
            if nxt == 'x' and i + 3 < len(inner) and inner[i + 2] in _HEX_DIGITS and inner[i + 3] in _HEX_DIGITS:
                chars.append(chr(int(inner[i + 2:i + 4], 16)))
                i += 4
            else:
                chars.append(_ESCAPE_SEQUENCES.get(nxt, nxt))
                i += 2
        else:
            chars.append(ch)
            i += 1
    return ''.join(chars)


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

    TokenType.LESS_THAN:             OperatorInfo(BinaryOp.LESS_THAN,             precedence=7, associativity=Associativity.LEFT),
    TokenType.GREATER_THAN:          OperatorInfo(BinaryOp.GREATER_THAN,          precedence=7, associativity=Associativity.LEFT),
    TokenType.LESS_THAN_OR_EQUAL:    OperatorInfo(BinaryOp.LESS_THAN_OR_EQUAL,    precedence=7, associativity=Associativity.LEFT),
    TokenType.GREATER_THAN_OR_EQUAL: OperatorInfo(BinaryOp.GREATER_THAN_OR_EQUAL, precedence=7, associativity=Associativity.LEFT),

    TokenType.EQUAL:     OperatorInfo(BinaryOp.EQUAL,     precedence=6, associativity=Associativity.LEFT),
    TokenType.NOT_EQUAL: OperatorInfo(BinaryOp.NOT_EQUAL, precedence=6, associativity=Associativity.LEFT),
    TokenType.IN:         OperatorInfo(BinaryOp.IN,        precedence=6, associativity=Associativity.LEFT),

    TokenType.AMPERSAND: OperatorInfo(BinaryOp.BITWISE_AND, precedence=5, associativity=Associativity.LEFT),
    TokenType.CARET:     OperatorInfo(BinaryOp.BITWISE_XOR, precedence=4, associativity=Associativity.LEFT),
    TokenType.PIPE:      OperatorInfo(BinaryOp.BITWISE_OR,  precedence=3, associativity=Associativity.LEFT),

    TokenType.AND: OperatorInfo(BinaryOp.AND, precedence=2, associativity=Associativity.LEFT),

    TokenType.OR: OperatorInfo(BinaryOp.OR, precedence=1, associativity=Associativity.LEFT),
}


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



_TYPE_START_TOKENS = (TokenType.INT, TokenType.INT8, TokenType.UINT8, TokenType.INT64, TokenType.INT32, TokenType.BOOL,
                      TokenType.STR, TokenType.IDENTIFIER, TokenType.STAR, TokenType.DICT)


class Parser:
    def __init__(self, tokens: List[Token]):
        if len(tokens) == 0:
            raise ValueError('tokens must have non zero length')
        if tokens[-1].type != TokenType.EOF:
            raise ValueError('tokens must be terminated by an EOF')
        self.tokens = tokens
        self.pos = 0
        self._ambiguous_element_brackets: set = set()  # token positions; see _looks_like_typed_literal

    def _error(self, message: str, tok: Token) -> 'ParseError':
        return ParseError(message, tok.file, tok.line, tok.col)


    def peek(self, offset: int = 0) -> Token:
        idx = min(self.pos + offset, len(self.tokens) - 1)
        return self.tokens[idx]

    def current(self) -> Token:
        return self.peek()

    def at_end(self) -> bool:
        return self.current().type == TokenType.EOF

    def check(self, *types: TokenType) -> bool:
        return not self.at_end() and self.current().type in types

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
        raise self._error(msg, tok)

    def skip_newlines(self):
        while self.match(TokenType.NEWLINE):
            pass


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
            intrinsics=intrinsics, consts=consts,
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
        path = _unescape_quoted_literal(path_tok.val)
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
        path = _unescape_quoted_literal(path_tok.val)
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

    def parse_type_declaration(self) -> Union[TypeAlias, StructDef, SumTypeDef]:
        """`type Name = T`, `type Name struct:`, or `type Name is A | B`."""
        start_tok = self.expect(TokenType.TYPE, "Expected 'type' to start a type declaration")
        name_tok = self.expect(TokenType.IDENTIFIER, "Expected a name for this type declaration")
        if self.check(TokenType.STRUCT):
            self.advance()
            return self._parse_struct_body(start_tok, name_tok)
        if self.check(TokenType.IS):
            self.advance()
            return self._parse_sum_type_body(start_tok, name_tok)
        self.expect(TokenType.ASSIGN, "Expected '=' (for a type alias), 'struct' (for a struct declaration), or 'is' (for a sum type)")
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
        return self.check(TokenType.INT, TokenType.INT8, TokenType.UINT8, TokenType.INT64, TokenType.INT32, TokenType.BOOL, TokenType.STR, TokenType.OPEN_BRACKET, TokenType.STAR, TokenType.DICT) or (
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
        if self.check(TokenType.INT, TokenType.INT8, TokenType.UINT8, TokenType.INT64, TokenType.INT32, TokenType.BOOL, TokenType.STR):
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
        """A statement ends its line: a block statement has consumed its block (DEDENT), a simple
        one must be followed by a newline, the block's end, or the end of input."""
        previous = self.tokens[self.pos - 1].type if self.pos else None
        if previous in (TokenType.DEDENT, TokenType.NEWLINE) or self.at_end():
            return
        if not self.check(TokenType.NEWLINE, TokenType.DEDENT):
            tok = self.current()
            raise self._error(f"Expected the end of the line after this statement, got {describe_token(tok)}", tok)

    def parse_statement(self) -> Node:
        if self.check(TokenType.INT, TokenType.INT8, TokenType.UINT8, TokenType.INT64, TokenType.INT32, TokenType.BOOL, TokenType.STR) and self.peek(1).type == TokenType.OPEN_PAREN:
            # Scalar keyword + '(' is a cast statement, not a VarDecl.
            return self.parse_expr_stmt_or_assign()
        if self.check(TokenType.STAR):
            # '*' may start a pointer-typed VarDecl or a deref expression; try the type first and backtrack.
            saved_pos = self.pos
            start_tok = self.current()
            try:
                parsed_type = self.parse_type()
            except ParseError:
                parsed_type = None
            if parsed_type is not None and self.check(TokenType.IDENTIFIER):
                return self.parse_var_decl(var_type=parsed_type, start_tok=start_tok)
            self.pos = saved_pos
        if self.check(TokenType.INT, TokenType.INT8, TokenType.UINT8, TokenType.INT64, TokenType.INT32, TokenType.BOOL, TokenType.STR, TokenType.OPEN_BRACKET, TokenType.DICT):
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
        return For(init=init, condition=condition, increment=increment, body=body, line=start_tok.line, col=start_tok.col)

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
        if self.check(TokenType.INT, TokenType.INT8, TokenType.UINT8, TokenType.INT64, TokenType.INT32, TokenType.BOOL, TokenType.STR,
                       TokenType.OPEN_BRACKET, TokenType.DICT):
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
        """IsCheck form, or an ordinary boolean expression."""
        if self.check(TokenType.IDENTIFIER) and self.peek(1).type == TokenType.IS:
            name_tok = self.advance()
            self.advance()
            type_name = self._parse_qualifiable_type_name("a type name after 'is'")
            if self.check(TokenType.AS):
                self.advance()
                binding_tok = self.expect(TokenType.IDENTIFIER, "Expected a binding name after 'as'")
                subject = Variable(name=name_tok.val, line=name_tok.line, col=name_tok.col)
                return IsCheck(variable_name=binding_tok.val, type_name=type_name, subject=subject, line=name_tok.line, col=name_tok.col)
            return IsCheck(variable_name=name_tok.val, type_name=type_name, line=name_tok.line, col=name_tok.col)
        expr = self.parse_expression()
        if self.check(TokenType.IS):
            is_tok = self.advance()
            type_name = self._parse_qualifiable_type_name("a type name after 'is'")
            self.expect(
                TokenType.AS,
                "Expected 'as NAME' after the type name -- a non-bare-variable "
                "subject (an Index, a Call, ...) needs an explicit binding name "
                "to narrow, since it has no existing name of its own",
            )
            binding_tok = self.expect(TokenType.IDENTIFIER, "Expected a binding name after 'as'")
            return IsCheck(variable_name=binding_tok.val, type_name=type_name, subject=expr, line=is_tok.line, col=is_tok.col)
        return expr

    def _parse_qualifiable_type_name(self, expected_message: str) -> Union[str, QualifiedTypeExpr, ArrayTypeExpr, SliceTypeExpr, PointerTypeExpr]:
        """A possibly qualified type name, builtin type keyword, `none`, or array/slice/pointer/dict type."""
        if self.check(TokenType.STAR, TokenType.OPEN_BRACKET, TokenType.DICT):
            return self.parse_type()
        if self.check(TokenType.INT, TokenType.INT8, TokenType.UINT8, TokenType.INT64, TokenType.INT32, TokenType.BOOL, TokenType.STR,
                      TokenType.NONE):  # `none`: a sum type's payload-free variant
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
                    expr = Call(name=name_tok.val, args=args, kwargs=kwargs, receiver=expr, line=expr.line, col=expr.col)
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
        open_pos = self.pos - 1
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

        if self.check(TokenType.COMMA) and open_pos in self._ambiguous_element_brackets:
            raise self._error(
                "Expected ']' after array index, got ',' -- a typed literal of pointers with only fixed sizes, "
                "like `[2][1]*P[a, b]`, reads as a multiplication: give the variable (or parameter) the type and "
                "use an untyped literal, e.g. `[2][1]*P g = [a, b]`", self.current())
        self.expect(TokenType.CLOSE_BRACKET, "Expected ']' after array index")
        return Index(array=array_expr, index=first, line=array_expr.line, col=array_expr.col)

    def parse_primary(self) -> Node:
        if self.check(TokenType.NUMBER):
            tok = self.advance()
            value = float(tok.val) if '.' in tok.val else int(tok.val)
            return Constant(value=value, line=tok.line, col=tok.col)
        if self.check(TokenType.TRUE, TokenType.FALSE):
            tok = self.advance()
            return BoolLiteral(value=(tok.type == TokenType.TRUE), line=tok.line, col=tok.col)
        if self.check(TokenType.NONE):
            tok = self.advance()
            return NoneLiteral(line=tok.line, col=tok.col)
        if self.check(TokenType.STRING):
            tok = self.advance()
            return StringLiteral(value=_unescape_quoted_literal(tok.val), line=tok.line, col=tok.col)
        if self.check(TokenType.BYTE):
            tok = self.advance()
            resolved = _unescape_quoted_literal(tok.val)
            if len(resolved) != 1 or ord(resolved) > 255:
                raise self._error(
                    f"A byte literal must resolve to exactly one byte (0-255), got "
                    f"{resolved!r}",
                    tok,
                )
            return ByteLiteral(value=ord(resolved), line=tok.line, col=tok.col)
        if self.check(TokenType.INT, TokenType.INT8, TokenType.UINT8, TokenType.INT64, TokenType.INT32, TokenType.BOOL, TokenType.STR) and self.peek(1).type == TokenType.OPEN_PAREN:
            return self.parse_cast()
        if self.check(TokenType.DICT):
            parsed_type = self.parse_type()
            return self.parse_dict_literal(parsed_type)
        if self._looks_like_typed_literal():
            parsed_type = self.parse_type()
            return self._parse_bracketed_literal(parsed_type)
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
                # `[x][0] * ys[1]` and `[N][M]*P[...]` are the same tokens; whether `ys`/`P` is a type
                # is only known later, so this shape is always the multiplication (see check_binary).
                j = k + 1
                if self.peek(j).type == TokenType.DOT and self.peek(j + 1).type == TokenType.IDENTIFIER:
                    j += 2
                self._ambiguous_element_brackets.add(self.pos + j)
                return False
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


# Entry points

def parse_tokens(tokens: List[Token]) -> Program:
    return Parser(tokens).parse_program()


def parse(filename: str) -> Program:
    tokens = lex(filename)
    return parse_tokens(tokens)


def main():
    arg_parser = argparse.ArgumentParser(description='Parser')
    arg_parser.add_argument('file', type=str, help='File to parse.')
    args = arg_parser.parse_args()
    ast = parse(args.file)
    print(ast.pretty())


if __name__ == '__main__':
    main()
