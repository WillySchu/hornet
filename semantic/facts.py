import dataclasses


@dataclasses.dataclass
class Facts:
    """What checking learned, keyed by parser-node number (Node.nid); the parser's nodes are never
    changed. TypedTreeBuilder builds the typed tree from these."""
    types: dict = dataclasses.field(default_factory=dict)  # expression, VarDecl, Param -> Type
    decls: dict = dataclasses.field(default_factory=dict)  # Variable, Assign, IsCheck -> Symbol.id (None: a constant)
    symbols: dict = dataclasses.field(default_factory=dict)  # VarDecl, Param -> Symbol
    for_symbols: dict = dataclasses.field(default_factory=dict)  # ForIn -> [Symbol] per binding
    bindings: dict = dataclasses.field(default_factory=dict)  # IsCheck with `as NAME` -> synthetic VarDecl for NAME
    member_loops: dict = dataclasses.field(default_factory=dict)  # ForIn over an enum's members -> the enum's type
    narrowed: dict = dataclasses.field(default_factory=dict)  # IsCheck -> the variant (or sum) it tests for
    sum_equalities: dict = dataclasses.field(default_factory=dict)  # `==`/`!=` Binary on sums -> the sum compared as
    # A Variable read flowing into a sum narrower than its own, which what is known of it there fits -> that sum.
    narrowed_sums: dict = dataclasses.field(default_factory=dict)
    boxed: dict = dataclasses.field(default_factory=dict)  # Unary `&Variant(...)` -> the sum it boxes into
    returns: dict = dataclasses.field(default_factory=dict)  # Function -> return Type
    calls: dict = dataclasses.field(default_factory=dict)  # Call -> (callee's key, args) unless name(args) as written
    const_refs: dict = dataclasses.field(default_factory=dict)  # Variable, or `alias.NAME` Field -> constant's key
    enum_members: dict = dataclasses.field(default_factory=dict)  # `Enum.Member` Field -> (enum's key, index)
    enum_checks: dict = dataclasses.field(default_factory=dict)  # IsCheck on an enum -> (member's index, the enum)
    enum_lens: dict = dataclasses.field(default_factory=dict)  # `len(Enum)` Call -> the number of members
    formats: dict = dataclasses.field(default_factory=dict)  # `format(...)` Call -> its template's text
    enum_ins: dict = dataclasses.field(default_factory=dict)  # `n in Enum` Binary -> the enum's key
    array_sizes: dict = dataclasses.field(default_factory=dict)  # `[EXPR]T` ArrayTypeExpr -> its size, EXPR's value
