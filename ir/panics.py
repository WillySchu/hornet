"""Panic blocks: a labelled call to hornet_panic with a fixed message, one per message per function.
A message carries the source position of what failed (`located`), so each failing site has its own.
A missing dict key's block calls hornet_panic_missing_key instead, which adds the key."""

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
        self.labels: dict = {}  # (message, key descriptor) -> label

    def label(self, message: str, key_descriptor: Optional[str] = None) -> str:
        """The block to branch to. With `key_descriptor` (the label of a dict's key type's
        descriptor), the panic is for a key that wasn't found, and the runtime prints the key too."""
        if (message, key_descriptor) not in self.labels:
            self.labels[message, key_descriptor] = self.ir_program.ids.new_label(self.prefix)
        return self.labels[message, key_descriptor]

    def blocks(self) -> list:
        out = []
        for (message, key_descriptor), label in self.labels.items():
            msg = self.ir_program.ids.new_temp(Type.INT64)
            out += [IRLabel(label), IRStaticDataAddress(dst=msg, label=message_label(self.ir_program, message))]
            if key_descriptor is None:
                out.append(IRCall(dst=None, name='hornet_panic', args=[msg]))
            else:
                descriptor = self.ir_program.ids.new_temp(Type.INT64)
                out += [IRStaticDataAddress(dst=descriptor, label=key_descriptor),
                        IRCall(dst=None, name='hornet_panic_missing_key', args=[msg, descriptor])]
            out.append(IRJump(label))  # not reached: the panic aborts
        return out
