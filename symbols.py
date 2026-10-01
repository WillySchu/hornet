"""The symbol table: one Symbol per declared variable (local, parameter, loop binding, narrowing
binding), created by semantic analysis. Later passes key per-variable facts on Symbol.id, and
name uses refer to their declaration through it (Variable.decl_id)."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(eq=False)
class Symbol:
    id: int
    name: str
    kind: str  # 'local', 'param', 'binding' (for-in), or 'narrowing' (`is`/`match ... as NAME`)
    type: Any
    line: int = 0
    col: int = 0
    file: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.name}#{self.id}"


@dataclass
class SymbolTable:
    symbols: list = field(default_factory=list)  # indexed by Symbol.id

    def new(self, name: str, kind: str, type_, node=None) -> Symbol:
        sym = Symbol(len(self.symbols), name, kind, type_, getattr(node, 'line', 0), getattr(node, 'col', 0),
                     getattr(node, 'file', None))
        self.symbols.append(sym)
        return sym

    def __getitem__(self, symbol_id: int) -> Symbol:
        return self.symbols[symbol_id]

    def __len__(self) -> int:
        return len(self.symbols)
