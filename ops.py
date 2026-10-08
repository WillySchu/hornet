"""Operator enums shared by the frontend, IR, and backend."""

from enum import auto, Enum


class UnaryOp(Enum):
    NEGATE = auto()
    COMPLEMENT = auto()
    NOT = auto()
    ADDRESS_OF = auto()
    DEREFERENCE = auto()  # shares STAR with pointer types and multiply

    def symbol(self) -> str:
        return {
            UnaryOp.NEGATE: '-',
            UnaryOp.COMPLEMENT: '~',
            UnaryOp.NOT: 'not',
            UnaryOp.ADDRESS_OF: '&',
            UnaryOp.DEREFERENCE: '*',
        }[self]


class BinaryOp(Enum):
    ADD = auto()
    SUBTRACT = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    MODULO = auto()

    SHIFT_LEFT = auto()
    SHIFT_RIGHT = auto()

    LESS_THAN = auto()
    GREATER_THAN = auto()
    LESS_THAN_OR_EQUAL = auto()
    GREATER_THAN_OR_EQUAL = auto()

    EQUAL = auto()
    NOT_EQUAL = auto()

    IN = auto()

    BITWISE_AND = auto()
    BITWISE_XOR = auto()
    BITWISE_OR = auto()

    AND = auto()
    OR = auto()

    def symbol(self) -> str:
        return {
            BinaryOp.ADD: '+',
            BinaryOp.SUBTRACT: '-',
            BinaryOp.MULTIPLY: '*',
            BinaryOp.DIVIDE: '/',
            BinaryOp.MODULO: '%',
            BinaryOp.SHIFT_LEFT: '<<',
            BinaryOp.SHIFT_RIGHT: '>>',
            BinaryOp.LESS_THAN: '<',
            BinaryOp.GREATER_THAN: '>',
            BinaryOp.LESS_THAN_OR_EQUAL: '<=',
            BinaryOp.GREATER_THAN_OR_EQUAL: '>=',
            BinaryOp.EQUAL: '==',
            BinaryOp.NOT_EQUAL: '!=',
            BinaryOp.IN: 'in',
            BinaryOp.BITWISE_AND: '&',
            BinaryOp.BITWISE_XOR: '^',
            BinaryOp.BITWISE_OR: '|',
            BinaryOp.AND: 'and',
            BinaryOp.OR: 'or',
        }[self]


ORDERING_OPS = {BinaryOp.LESS_THAN, BinaryOp.GREATER_THAN, BinaryOp.LESS_THAN_OR_EQUAL, BinaryOp.GREATER_THAN_OR_EQUAL}
EQUALITY_OPS = {BinaryOp.EQUAL, BinaryOp.NOT_EQUAL}
LOGICAL_OPS = {BinaryOp.AND, BinaryOp.OR}
