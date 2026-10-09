"""Calls: everything written `name(...)` or `value.name(...)`.

That is a call to a function, an extern, or a method; a builtin (`print`, `len`, `append`, `del`,
`bytes`, `panic`, `format`); the conversion of an integer to an enum; or a struct literal, which is
told from a call only by what its name refers to.

A call's arguments are expressions, so CallChecker is one part of the expression checker and calls
back into the rest of it (Checker, below) for them. It keeps no state of its own."""

from typing import Optional, Protocol

from ops import UnaryOp
from parser import Call, Field, Index, Node, Unary, Variable
from scopes import display_name as shown
from semantic.errors import SemanticError
from typesys import BYTE_SLICE, INTEGER_TYPES, StructInfo, Type, TypeKind
import typed_ast as typed


class Checker(Protocol):
    """What CallChecker needs of the expression checker it is part of."""
    scope: object  # the scope of the file being checked (scopes.py)

    def check_expr(self, expr: Node) -> Type: ...

    def check_expr_allowing_struct_literal(self, expr: Node) -> Type: ...

    def check_value_flowing_into(self, expr: Node, target_type: Type) -> Type: ...

    def check_value_flowing_into_allowing_struct_literal(self, expr: Node, target_type: Type) -> Type: ...

    def types_compatible(self, value_type: Type, target_type: Type) -> bool: ...

    def sum_gap(self, value_type: Type, target_type: Type, expr: Optional[Node] = None) -> str: ...

    def hidden(self, struct: str, name: str) -> bool: ...

    def enum_named_by(self, expr: Node) -> Optional[str]: ...

    def as_folded_int_literal(self, expr: Node) -> Optional[int]: ...

    def without_zero_value(self, t: Type, seen: Optional[set] = None) -> Optional[Type]: ...


class CallChecker:
    """Checks calls and struct literals, for `checker`."""

    def __init__(self, checker: Checker, decls, facts, constants, module_set):
        self.checker = checker
        self.decls = decls
        self.facts = facts
        self.constants = constants
        self.module_set = module_set

    def _callee(self, expr: Call) -> Optional[str]:
        """The key a call's name refers to (a function, struct, extern, or intrinsic), resolving
        `alias.name(...)`; the name as written if it names nothing at top level (a builtin, or
        undeclared); None for a method call."""
        if expr.nid in self.module_set.qualified:
            return self.module_set.qualified[expr.nid]
        if expr.receiver is not None:
            return None
        return self.checker.scope.resolve(expr.name) or expr.name

    def struct_literal(self, expr: Node) -> Optional[str]:
        """The struct's key if `expr` is a struct literal."""
        if isinstance(expr, Call):
            name = self._callee(expr)
            if name in self.decls.structs:
                return name
        return None

    def _record_call(self, expr: Call, name: str) -> None:
        """Note the key a call refers to, unless it's the name as written."""
        if name != expr.name or expr.receiver is not None:
            self.facts.calls[expr.nid] = (name, list(expr.args))

    def check_call(self, expr: Call) -> Type:
        name = self._callee(expr)
        if name is None:
            # Receiver present (and not a module) means method call, checked first.
            return self._check_method_call(expr)
        if name in self.decls.structs:
            raise SemanticError(
                f"'{expr.name}(...)' is a struct literal, which is only "
                f"allowed as a variable's initializer, a plain "
                f"assignment's value, a direct function-call or "
                f"method-call argument, a method-call receiver, a "
                f"direct return value, an array literal's own "
                f"element, an assigned element or field, an assigned "
                f"field's base, a field-access "
                f"base, a binary operand, or a bare statement -- not "
                f"most other kinds of expressions (an Index/Slice "
                f"base, a Cast's own expression, ...); assign it to a "
                f"variable first if you need it in one of those "
                f"positions",
                expr,
            )
        if expr.kwargs is not None:
            raise SemanticError(
                f"'{expr.name}(...)' uses named arguments, which are "
                f"only supported for struct literals, not function calls",
                expr,
            )
        if name == 'print':
            return self.check_print_call(expr)
        if name == 'len':
            return self.check_len_call(expr)
        if name == 'append':
            return self.check_append_call(expr)
        if name == 'del':
            return self.check_del_call(expr)
        if name == 'bytes':
            return self.check_bytes_call(expr)
        if name == 'panic':
            return self.check_panic_call(expr)
        if name == 'format':
            return self.check_format_call(expr)
        if name in self.decls.enums:
            return self.check_enum_conversion(expr, name)
        visible = expr.nid in self.module_set.qualified or self.checker.scope.resolve(expr.name) is not None
        if name not in self.decls.functions or not visible:  # another module's extern needs an import too
            raise SemanticError(f"Call to undeclared function '{shown(name)}'", expr)
        param_types, return_type = self.decls.functions[name]
        self._record_call(expr, name)

        if len(expr.args) != len(param_types):
            raise SemanticError(
                f"Function '{shown(name)}' expects {len(param_types)} "
                f"argument(s), got {len(expr.args)}",
                expr,
            )
        for i, (arg, expected_type) in enumerate(zip(expr.args, param_types), start=1):
            actual_type = self.checker.check_value_flowing_into_allowing_struct_literal(arg, expected_type)
            if not self.checker.types_compatible(actual_type, expected_type):
                raise SemanticError(
                    f"Argument {i} to '{shown(name)}' should be "
                    f"{expected_type}, got {actual_type}" + self.checker.sum_gap(actual_type, expected_type, arg),
                    arg,
                )
        return return_type

    def _check_method_call(self, expr: Call) -> Type:
        """Resolve `receiver.name(args)` to its method: a call of the mangled function with the receiver
        (or its address, for a pointer receiver) first."""
        if expr.kwargs is not None:
            raise SemanticError(
                f"'{expr.name}(...)' uses named arguments, which are not "
                f"supported for method calls",
                expr,
            )
        receiver_type = self.checker.check_expr_allowing_struct_literal(expr.receiver)
        has_methods = (TypeKind.STRUCT, TypeKind.ENUM)
        receiver_is_pointer = (receiver_type.kind == TypeKind.POINTER
                               and receiver_type.element_type.kind in has_methods)
        if receiver_is_pointer:
            # auto-deref
            receiver_type = receiver_type.element_type
        if receiver_type.kind not in has_methods:
            raise SemanticError(
                f"Cannot call method '{expr.name}' on a value of type "
                f"{receiver_type} -- methods are only defined on structs and enums",
                expr.receiver,
            )
        is_enum = receiver_type.kind == TypeKind.ENUM
        owner = receiver_type.enum_name if is_enum else receiver_type.struct_name
        key = (owner, expr.name)
        if key not in self.decls.methods:
            raise SemanticError(
                f"{'Enum' if is_enum else 'Struct'} '{shown(owner)}' has no method "
                f"'{expr.name}'",
                expr,
            )
        if self.checker.hidden(owner, expr.name):
            raise SemanticError(
                f"Method '{expr.name}' of '{shown(owner)}' is not visible outside the module "
                f"that defines the {'enum' if is_enum else 'struct'} -- names starting with '_' are private to "
                f"their own module", expr)
        param_types, return_type, mangled_name = self.decls.methods[key]
        receiver = expr.receiver
        if key in self.decls.pointer_receivers and not receiver_is_pointer:
            # Pointer receiver: pass the receiver's address.
            is_place = isinstance(expr.receiver, (Variable, Field, Index)) or (
                    isinstance(expr.receiver, Unary) and expr.receiver.op == UnaryOp.DEREFERENCE)
            if not is_place or expr.receiver.nid in self.facts.enum_members:  # (`Color.Red` is a value)
                raise SemanticError(
                    f"Method '{expr.name}' on '{shown(owner)}' has a pointer receiver, so it "
                    f"needs an addressable receiver (a variable, field, index, or dereference), not a temporary",
                    expr.receiver,
                )
            if isinstance(receiver, Unary):
                receiver = receiver.operand  # &(*p) is p
            else:
                receiver = Unary(op=UnaryOp.ADDRESS_OF, operand=receiver,
                                 line=receiver.line, col=receiver.col, file=receiver.file)
                self.checker.check_expr(receiver)
        if len(expr.args) != len(param_types):
            raise SemanticError(
                f"Method '{expr.name}' on '{shown(owner)}' "
                f"expects {len(param_types)} argument(s), got "
                f"{len(expr.args)}",
                expr,
            )
        for i, (arg, expected_type) in enumerate(zip(expr.args, param_types), start=1):
            actual_type = self.checker.check_value_flowing_into_allowing_struct_literal(arg, expected_type)
            if not self.checker.types_compatible(actual_type, expected_type):
                raise SemanticError(
                    f"Argument {i} to method '{expr.name}' on "
                    f"'{shown(owner)}' should be "
                    f"{expected_type}, got {actual_type}" + self.checker.sum_gap(actual_type, expected_type, arg),
                    arg,
                )
        self.facts.calls[expr.nid] = (mangled_name, [receiver] + list(expr.args))
        self.facts.types[expr.nid] = return_type
        return return_type

    def check_print_call(self, expr: Call) -> Type:
        """`print(x)` for any non-void type."""
        if len(expr.args) != 1:
            raise SemanticError(
                f"'print' expects exactly 1 argument, got {len(expr.args)}",
                expr,
            )
        arg_type = self.checker.check_expr_allowing_struct_literal(expr.args[0])
        if arg_type in (Type.VOID, Type.NEVER):
            raise SemanticError(
                "'print' cannot be called with the result of a function "
                "that has no declared return type -- there's no value there to print",
                expr.args[0],
            )
        return Type.VOID

    def check_panic_call(self, expr: Call) -> Type:
        """`panic(message)`: report a str with the call's position and abort."""
        if len(expr.args) != 1:
            raise SemanticError(f"'panic' expects exactly 1 argument, got {len(expr.args)}", expr)
        arg_type = self.checker.check_expr(expr.args[0])
        if arg_type != Type.STR:
            raise SemanticError(f"'panic' expects a str, got {arg_type}", expr.args[0])
        return Type.NEVER

    def check_format_call(self, expr: Call) -> Type:
        """`format(template, args...)`: a str, the template with each `{}` replaced by the next
        argument as print shows it. The template is known here, so it is checked against them."""
        if not expr.args:
            raise SemanticError("'format' expects a template, then a value for each '{}' in it", expr)
        template = expr.args[0]
        if self.checker.check_expr(template) != Type.STR:
            raise SemanticError(
                f"'format' expects a str template first, got {self.facts.types[template.nid]}", template)
        try:
            text = self.constants.evaluate(template)
        except SemanticError:
            raise SemanticError(
                "'format' needs its template as a string literal or a constant, so that it can be checked "
                "against the values", template) from None
        try:
            holes = len(typed.format_pieces(text)) - 1
        except ValueError as problem:
            raise SemanticError(f"In this 'format' template: {problem}", template) from None
        values = expr.args[1:]
        if holes != len(values):
            raise SemanticError(
                f"This 'format' template has {holes} '{{}}' placeholder{'' if holes == 1 else 's'}, "
                f"but {len(values)} value{' was' if len(values) == 1 else 's were'} given", expr)
        for value in values:
            if self.checker.check_expr_allowing_struct_literal(value) in (Type.VOID, Type.NEVER):
                raise SemanticError(
                    "'format' cannot show the result of a function that has no declared return type -- there's "
                    "no value there", value)
        self.facts.formats[expr.nid] = text
        return Type.STR

    def check_enum_conversion(self, expr: Call, enum: str) -> Type:
        """`Enum(n)`: the member whose value is the integer `n`. Checked when it runs (a panic if there
        is none), or here when `n` is a literal."""
        members = self.decls.enums[enum].members
        if len(expr.args) != 1:
            raise SemanticError(
                f"'{expr.name}(...)' converts one integer to the enum {shown(enum)}, got {len(expr.args)} arguments",
                expr)
        arg_type = self.checker.check_expr(expr.args[0])
        if arg_type not in INTEGER_TYPES:
            raise SemanticError(
                f"'{expr.name}(...)' converts an integer to the enum {shown(enum)}, got {arg_type}", expr.args[0])
        literal = self.checker.as_folded_int_literal(expr.args[0])
        if literal is not None and not 0 <= literal < len(members):
            raise SemanticError(
                f"{literal} is not a member of {shown(enum)} (its members' values are 0 to {len(members) - 1})",
                expr.args[0]
            )
        self._record_call(expr, enum)
        return Type(TypeKind.ENUM, enum_name=enum)

    def check_len_call(self, expr: Call) -> Type:
        """`len(x)` for arrays, slices, str, and dicts; `len(Enum)` is an enum's number of members."""
        if len(expr.args) != 1:
            raise SemanticError(
                f"'len' expects exactly 1 argument, got {len(expr.args)}",
                expr,
            )
        enum = self.checker.enum_named_by(expr.args[0])
        if enum is not None:
            self.facts.enum_lens[expr.nid] = len(self.decls.enums[enum].members)
            return Type.INT
        arg_type = self.checker.check_expr(expr.args[0])
        if arg_type.kind not in (TypeKind.ARRAY, TypeKind.SLICE, TypeKind.STR, TypeKind.DICT):
            raise SemanticError(
                f"'len' requires an array, slice, str, or dict argument, got {arg_type}",
                expr.args[0],
            )
        return Type.INT

    def check_append_call(self, expr: Call) -> Type:
        """`append(s, v)` returns a new slice."""
        if len(expr.args) != 2:
            raise SemanticError(
                f"'append' expects exactly 2 arguments, got {len(expr.args)}",
                expr,
            )
        slice_arg, value_arg = expr.args
        slice_type = self.checker.check_expr(slice_arg)
        if slice_type.kind != TypeKind.SLICE:
            raise SemanticError(
                f"'append' requires a slice as its first argument, "
                f"got {slice_type}",
                slice_arg,
            )
        value_type = self.checker.check_value_flowing_into_allowing_struct_literal(value_arg, slice_type.element_type)
        if not self.checker.types_compatible(value_type, slice_type.element_type):
            raise SemanticError(
                f"'append' cannot append a value of type {value_type} "
                f"to a {slice_type} (element type "
                f"{slice_type.element_type})",
                value_arg,
            )
        return slice_type

    def check_bytes_call(self, expr: Call) -> Type:
        """`bytes(s)`: a new []byte copy of str s."""
        if len(expr.args) != 1 or expr.kwargs:
            raise SemanticError(f"bytes() takes exactly one argument, got {len(expr.args)}", expr)
        arg_type = self.checker.check_expr(expr.args[0])
        if arg_type != Type.STR:
            raise SemanticError(f"bytes() takes a str, got {arg_type}", expr)
        return BYTE_SLICE

    def check_del_call(self, expr: Call) -> Type:
        """`del(d, key)` mutates d in place."""
        if len(expr.args) != 2:
            raise SemanticError(
                f"'del' expects exactly 2 arguments, got {len(expr.args)}",
                expr,
            )
        dict_arg, key_arg = expr.args
        dict_type = self.checker.check_expr(dict_arg)
        if dict_type.kind != TypeKind.DICT:
            raise SemanticError(
                f"'del' requires a dict as its first argument, got {dict_type}",
                dict_arg,
            )
        key_type = self.checker.check_value_flowing_into(key_arg, dict_type.key_type)
        if not self.checker.types_compatible(key_type, dict_type.key_type):
            raise SemanticError(
                f"'del' cannot look up a key of type {key_type} in a "
                f"{dict_type} (key type {dict_type.key_type})",
                key_arg,
            )
        return Type.VOID

    def check_struct_literal(self, expr: Call) -> Type:
        """`Name(args)`: positional struct literal; must be exhaustive."""
        name = self.struct_literal(expr)
        self._record_call(expr, name)
        struct_info = self.decls.structs[name]
        field_items = list(struct_info.fields.items())
        if expr.kwargs is not None:
            return self._check_named_struct_literal(expr, struct_info, field_items)
        private = [field for field, _ in field_items if self.checker.hidden(name, field)]
        if private:  # it would have to give them values
            raise SemanticError(
                f"'{shown(name)}(...)' gives every field by position, but {', '.join(private)} "
                f"{'is' if len(private) == 1 else 'are'} private to the module that defines '{shown(name)}' -- "
                f"name the public fields instead (`{shown(name)}(field=value)`); private ones start as zero", expr)
        if len(expr.args) != len(field_items):
            field_names = ', '.join(name for name, _ in field_items)
            raise SemanticError(
                f"Struct literal for '{shown(name)}' expects "
                f"{len(field_items)} argument(s) (one per field, in "
                f"declaration order: {field_names}), got {len(expr.args)}",
                expr,
            )
        for i, (arg, (field_name, field_type)) in enumerate(zip(expr.args, field_items), start=1):
            arg_type = self.checker.check_value_flowing_into_allowing_struct_literal(arg, field_type)
            if not self.checker.types_compatible(arg_type, field_type):
                raise SemanticError(
                    f"Argument {i} to struct literal '{shown(name)}' "
                    f"(field '{field_name}') should be {field_type}, "
                    f"got {arg_type}",
                    arg,
                )
        result = Type(TypeKind.STRUCT, struct_name=name)
        self.facts.types[expr.nid] = result
        return result

    def _check_named_struct_literal(self, expr: Call, struct_info: StructInfo, field_items: list) -> Type:
        """`Name(f=v, ...)`: named struct literal; omitted fields are zero."""
        name = struct_info.name
        field_types = struct_info.fields
        valid_names = ', '.join(name for name, _ in field_items)
        seen = set()
        for field_name, value in expr.kwargs:
            if field_name not in field_types:
                raise SemanticError(
                    f"Struct literal for '{shown(name)}' has no field "
                    f"'{field_name}' -- valid fields are: {valid_names}",
                    expr,
                )
            if self.checker.hidden(name, field_name):
                raise SemanticError(
                    f"Field '{field_name}' of '{shown(name)}' is not visible outside the module that defines the "
                    f"struct -- names starting with '_' are private to their own module", expr)
            if field_name in seen:
                raise SemanticError(
                    f"Field '{field_name}' specified more than once in "
                    f"struct literal for '{shown(name)}'",
                    expr,
                )
            seen.add(field_name)
            value_type = self.checker.check_value_flowing_into_allowing_struct_literal(value, field_types[field_name])
            expected_type = field_types[field_name]
            if not self.checker.types_compatible(value_type, expected_type):
                raise SemanticError(
                    f"Field '{field_name}' of struct literal '{shown(name)}' "
                    f"should be {expected_type}, got {value_type}",
                    value,
                )
        for field_name, field_type in field_types.items():
            if field_name not in seen:
                missing = self.checker.without_zero_value(field_type)
                if missing is not None:
                    raise SemanticError(
                        f"Struct literal for '{shown(name)}' omits field '{field_name}', but {missing} has no "
                        f"zero value (only a sum type with a `none` variant does) -- give the field a value",
                        expr)
        result = Type(TypeKind.STRUCT, struct_name=name)
        self.facts.types[expr.nid] = result
        return result
