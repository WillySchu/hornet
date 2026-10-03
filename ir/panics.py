"""Panic blocks: a labelled call to hornet_panic with a fixed message, one per message per function.
A message carries the source position of what failed (`located`), so each failing site has its own."""

from typing import Optional

from ir.ir import IRCall, IRJump, IRLabel, IRStaticDataAddress
from typesys import Type


def located(message: str, where: Optional[str]) -> str:
    """`file:line:col: panic: message`; the message alone where there is no position."""
    return f"{where}: panic: {message}" if where else message


def message_label(ir_program, message: str) -> str:
    """The static string holding `message`, one per program."""
    labels = ir_program.__dict__.setdefault('_panic_message_labels', {})
    if message not in labels:
        labels[message] = ir_program.ids.new_label("panic_msg")
        ir_program.string_literals.append((labels[message], message))
    return labels[message]


class PanicBlocks:
    """One function's panic blocks: label(message) to branch to, blocks() to append at its end."""

    def __init__(self, ir_program, label_prefix: str):
        self.ir_program = ir_program
        self.prefix = label_prefix
        self.labels: dict = {}  # message -> label

    def label(self, message: str) -> str:
        if message not in self.labels:
            self.labels[message] = self.ir_program.ids.new_label(self.prefix)
        return self.labels[message]

    def blocks(self) -> list:
        out = []
        for message, label in self.labels.items():
            msg = self.ir_program.ids.new_temp(Type.INT64)
            out += [IRLabel(label),
                    IRStaticDataAddress(dst=msg, label=message_label(self.ir_program, message)),
                    IRCall(dst=None, name='hornet_panic', args=[msg]),
                    IRJump(label)]  # not reached: hornet_panic aborts
        return out
