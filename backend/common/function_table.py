"""The table of a program's functions that the runtime reads (runtime.c's hornet_functions): where
each starts and ends, and its name. With it an address can be put to a name, which is how a stack
overflow says which function it was in."""

from typing import Callable, List

SYMBOL = "hornet_functions"


def end_label(index: int) -> str:
    """The label an emitter puts after the last instruction of the function at `index`."""
    return f".Lhornet_function_end_{index}"


def shown_name(link_name: str) -> str:
    """A function's name as it is declared: without the module prefix of its key, or the `$` that an
    entry file's function is linked under."""
    return link_name.rstrip('$').rsplit('$', 1)[-1]


def table_lines(link_names: List[str], symbol: Callable[[str], str]) -> List[str]:
    """The table's assembly, for the functions emitted in this order with those end labels: three
    addresses for each, then three zeros; and the names' text. `symbol` spells a global name as the
    target's assembler wants it."""
    lines = ["    .data", f"    .globl {symbol(SYMBOL)}", "    .p2align 3", f"{symbol(SYMBOL)}:"]
    for index, name in enumerate(link_names):
        lines.append(f"    .quad {symbol(name)}, {end_label(index)}, .Lhornet_function_name_{index}")
    lines.append("    .quad 0, 0, 0")
    for index, name in enumerate(link_names):
        lines += [f".Lhornet_function_name_{index}:", f'    .asciz "{shown_name(name)}"']
    return lines
