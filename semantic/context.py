"""Where checking is: what changes as the checker moves through a program.

One object, shared by everything that checks: whoever moves on (to another function, into another
file's declaration, into a constant expression) changes it here, and the rest see it."""

import dataclasses


@dataclasses.dataclass
class Context:
    scope: object = None  # the scope of the file being checked (scopes.py): what its top-level names refer to
    scopes: object = None  # the names in scope in the function being checked, as narrowed there (flow.py's Scopes)
    # `EXPR is T as NAME` checks that may bind NAME where they are (an `if`'s or a `match`'s); those that have.
    bindable: set = dataclasses.field(default_factory=set)
    bound: set = dataclasses.field(default_factory=set)
