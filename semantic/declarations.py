"""Declarations: what a program declares at top level, resolved before any function body is checked.

DeclarationResolver fills a Declarations, which the body checker then only reads. The order of its
steps matters and is spelled out in resolve(). Two things it must ask of the checker, both about
constant expressions: the value of an array size, and (through ConstEvaluator) the check of a
constant's declaration. While it runs, the checker therefore sees a Declarations that is still
filling; such expressions can involve only enums and constants, which come first."""

import dataclasses
import os
from typing import Callable, Dict, List, Optional, Set, Tuple

from diagnostics import path_text
from parser import (
    READ_AS_A_TYPED_LITERAL, ArrayLiteral, ArrayTypeExpr, EnumDef, ExternFunctionDecl, Function, IntrinsicDecl,
    MethodDef, Node, Param, PointerTypeExpr, Program, QualifiedTypeExpr, SliceTypeExpr, StructDef, SumTypeDef,
    TypeAlias,
)
from scopes import BUILTIN_FUNCTION_NAMES, display_name as shown
from semantic.constants import ConstEvaluator
from semantic.errors import SemanticError
from semantic.facts import Facts
from semantic.type_resolution import TYPE_NAMES, TypeResolver, type_from_name
from typesys import EnumInfo, StructInfo, SumTypeInfo, Type, TypeKind


@dataclasses.dataclass
class Declarations:
    """A program's top-level declarations, each registry by key (see scopes.py)."""
    enums: Dict[str, EnumInfo] = dataclasses.field(default_factory=dict)
    type_aliases: Dict[str, Type] = dataclasses.field(default_factory=dict)
    structs: Dict[str, StructInfo] = dataclasses.field(default_factory=dict)
    sum_types: Dict[str, SumTypeInfo] = dataclasses.field(default_factory=dict)
    # (struct or enum, method) -> (param types, return type, mangled name)
    methods: Dict[Tuple[str, str], Tuple[List[Type], Type, str]] = dataclasses.field(default_factory=dict)
    pointer_receivers: set = dataclasses.field(default_factory=set)  # (struct or enum, method) with a `*receiver`
    functions: Dict[str, tuple] = dataclasses.field(default_factory=dict)  # name -> (param types, return type)
    extern_names: set = dataclasses.field(default_factory=set)
    intrinsic_original_names: dict = dataclasses.field(default_factory=dict)  # mangled name -> the intrinsic's own
    # Every function to check and compile: the program's, and each method as one (see _method_function).
    all_functions: list = dataclasses.field(default_factory=list)


def _file_and_line(node: Node) -> str:
    """`file (line N)` of a declaration, for an error reported somewhere else."""
    return f"{path_text(os.path.basename(node.file)) if node.file else '<input>'} (line {node.line})"


def mangle_method_name(struct_name: str, method_name: str) -> str:
    """`Struct.method` (or `Enum.method`); '.' can't appear in identifiers, so no collisions."""
    return f"{struct_name}.{method_name}"


def _method_function(sd, md: MethodDef) -> Function:
    """A method of the struct or enum `sd` as the function it compiles to: its mangled name, and the
    receiver as the first parameter (`*S` for a pointer receiver). New nodes; the program's own are
    left as they are."""
    receiver_type = sd.name
    if md.receiver_is_pointer:
        receiver_type = PointerTypeExpr(pointee_type=sd.name, line=md.line, col=md.col, file=md.file)
    receiver = Param(name=md.receiver_name, type=receiver_type, line=md.line, col=md.col, file=md.file)
    return Function(name=mangle_method_name(sd.name, md.name), return_type=md.return_type,
                    params=[receiver] + md.params, body=md.body, line=md.line, col=md.col, file=md.file)


class DeclarationResolver:
    """Resolves a program's declarations into `decls`."""

    def __init__(self, program, decls: Declarations, types: TypeResolver, facts: Facts, constants: ConstEvaluator,
                 array_size: Callable):
        """`array_size(expr, scope)` is the checker's: the value of an array-size expression written in
        the file whose scope is `scope`."""
        self.program = program
        self.decls = decls
        self.types = types
        self.facts = facts
        self.constants = constants
        self._array_size = array_size
        self.scope = program.files[0][1]  # of the file whose declaration is being resolved
        self._extern_decls: dict = {}  # name -> its declaration, for the duplicate's error

    def resolve(self) -> None:
        program, decls = self.program, self.decls
        # Each method is checked and compiled as a function of its own (see _method_function).
        methods = [(_method_function(sd, md), sd) for sd in list(program.structs) + list(program.enums)
                   for md in sd.methods]
        for fn, sd in methods:
            program.scope_of[fn.nid] = program.scope_of[sd.nid]
        decls.all_functions = list(program.functions) + [fn for fn, _ in methods]
        # Order matters: 0. enums depend on nothing, and constant array sizes are evaluated before any
        # type is resolved. (In place: the constant evaluator already holds this registry.)
        decls.enums.update(self._collect_enums(program.enums))
        self.constants.declare(program.consts)
        self._resolve_array_sizes(program)

        # 1. Reserve struct names so aliases can target them.
        struct_registry = self._reserve_struct_names(program.structs)

        # 2. Resolve aliases before struct fields, which may use them.
        decls.type_aliases = self._collect_type_aliases(program.type_aliases, struct_registry)

        # 3. Resolve struct fields; sum-type names are reserved first so fields can name them.
        reserved_sums = {std.name: None for std in program.sum_types}
        decls.structs = self._resolve_struct_fields(program.structs, struct_registry, reserved_sums)

        # 3.5. Sum types need resolved structs. Then no type may contain itself by value.
        decls.sum_types = self._resolve_sum_types(program.sum_types, decls.structs)
        self._check_value_containment(program)

        # 3.6. Methods, after struct resolution.
        decls.methods = self._collect_methods(program)

        # 4. All signatures before any body, so order doesn't matter.
        self._collect_functions()

        # 4.5. Externs share the function registry.
        decls.extern_names = {ext.name for ext in program.extern_functions}
        for ext in program.extern_functions:
            self._enter(ext)
            self.check_extern_function_decl(ext)

        # 4.6. Intrinsics share the function registry.
        for ic in program.intrinsics:
            self._enter(ic)
            self.check_intrinsic_decl(ic)

        self._check_enum_name_collisions(program)

        # 4.7. Constant values, in dependency order.
        self.constants.evaluate_all()
        self._check_const_name_collisions(program)

    # -- its place in the program, and types as written there

    def _enter(self, decl: Node) -> None:
        """Resolve `decl` (a top-level declaration, under its key) in its own file's scope."""
        self.scope = self.program.scope_of[decl.nid]

    def _resolve_type_name(self, name):
        return self.types.key(name, self.scope)

    def _type(self, type_expr, node: Node, sums: bool = True) -> Type:
        return self.types.resolve(type_expr, node, self.scope, sums)

    def _return_type(self, decl) -> Type:
        return self.types.return_type(decl, self.scope)

    # -- the steps

    def _collect_functions(self) -> None:
        for fn in self.decls.all_functions:
            self._enter(fn)
            if shown(fn.name) in BUILTIN_FUNCTION_NAMES:
                raise SemanticError(
                    f"'{shown(fn.name)}' is a builtin and can't be redefined as "
                    f"a function",
                    fn,
                )
            if fn.name in self.decls.structs:
                raise SemanticError(
                    f"Function '{shown(fn.name)}' collides with a struct of the "
                    f"same name -- struct and function names share one "
                    f"namespace and can never be the same, since "
                    f"'{shown(fn.name)}(...)' would otherwise be ambiguous "
                    f"between a call and a struct literal",
                    fn,
                )
            if fn.name in self.decls.type_aliases:
                raise SemanticError(
                    f"Function '{shown(fn.name)}' collides with a type alias "
                    f"of the same name -- function and type-alias "
                    f"names share one namespace and can never be the "
                    f"same",
                    fn,
                )
            if fn.name in self.decls.sum_types:
                raise SemanticError(
                    f"Function '{shown(fn.name)}' collides with a sum type "
                    f"of the same name -- function and sum-type names "
                    f"share one namespace and can never be the same",
                    fn,
                )
            if fn.name in self.decls.functions:
                raise SemanticError(f"Function '{shown(fn.name)}' is already declared", fn)
            param_types = [self._type(p.type, p) for p in fn.params]
            self.decls.functions[fn.name] = (param_types, self._return_type(fn))

    def _collect_enums(self, enum_defs: List[EnumDef]) -> Dict[str, EnumInfo]:
        """The enum registry: each enum's members, distinct and at least one."""
        registry: Dict[str, EnumInfo] = {}
        for ed in enum_defs:
            self._enter(ed)
            if shown(ed.name) in BUILTIN_FUNCTION_NAMES:
                raise SemanticError(f"'{shown(ed.name)}' is a builtin and can't be used as an enum name", ed)
            if ed.name in registry:
                raise SemanticError(f"Enum '{shown(ed.name)}' is already declared", ed)
            names: List[str] = []
            for member in ed.members:
                if member.name in names:
                    raise SemanticError(
                        f"Member '{member.name}' is already declared in enum '{shown(ed.name)}'", member)
                names.append(member.name)
            registry[ed.name] = EnumInfo(name=ed.name, members=names)
        return registry

    def _resolve_array_sizes(self, program: Program) -> None:
        """Record each `[EXPR]T` size's value (in Facts.array_sizes): a positive integer computed from
        literals and constants only (whose types must therefore be builtin)."""
        for file_program, scope in program.files:
            self.scope = scope
            self._resolve_array_sizes_in(file_program)
        self.scope = program.files[0][1]

    def _resolve_array_sizes_in(self, program: Program) -> None:
        seen = set()
        stack = [program]
        while stack:
            node = stack.pop()
            if node.nid in seen:
                continue
            seen.add(node.nid)
            if isinstance(node, ArrayTypeExpr) and not isinstance(node.size, int):
                self.facts.array_sizes[node.nid] = self._array_size(node.size, self.scope)
            if isinstance(node, ArrayLiteral) and node.could_be_multiplication:
                try:  # its sizes now, to say how it was read if one is no constant
                    self._resolve_array_sizes_in(node.type_expr)
                except SemanticError as problem:
                    raise SemanticError(problem.message + READ_AS_A_TYPED_LITERAL, node) from None
            if dataclasses.is_dataclass(node) and not isinstance(node, type):
                for f in dataclasses.fields(node):
                    value = getattr(node, f.name)
                    for v in value if isinstance(value, (list, tuple)) else [value]:
                        if isinstance(v, (Node, Program)):
                            stack.append(v)
                        elif isinstance(v, tuple):
                            stack.extend(x for x in v if isinstance(x, Node))

    def _reserve_struct_names(self, struct_defs: List[StructDef]) -> Dict[str, StructInfo]:
        """Reserve struct names (None placeholders) so fields can forward-reference."""
        registry: Dict[str, StructInfo] = {}
        for sd in struct_defs:
            if shown(sd.name) in BUILTIN_FUNCTION_NAMES:
                raise SemanticError(
                    f"'{shown(sd.name)}' is a builtin and can't be used as a "
                    f"struct name",
                    sd,
                )
            if sd.name in registry:
                raise SemanticError(f"Struct '{shown(sd.name)}' is already declared", sd)
            registry[sd.name] = None
        return registry

    def _collect_type_aliases(self, alias_defs: List[TypeAlias], structs: Dict[str, StructInfo]) -> Dict[str, Type]:
        """Resolve every alias fully, detecting cycles."""
        # Pass 1: reserve names, rejecting duplicates and collisions.
        seen: Dict[str, TypeAlias] = {}
        for ad in alias_defs:
            if shown(ad.name) in BUILTIN_FUNCTION_NAMES:
                raise SemanticError(
                    f"'{shown(ad.name)}' is a builtin and can't be used as a "
                    f"type alias name",
                    ad,
                )
            if ad.name in structs:
                raise SemanticError(
                    f"Type alias '{shown(ad.name)}' collides with a struct of "
                    f"the same name -- struct and type-alias names "
                    f"share one namespace and can never be the same",
                    ad,
                )
            if ad.name in seen:
                raise SemanticError(f"Type alias '{shown(ad.name)}' is already declared", ad)
            seen[ad.name] = ad

        resolved: Dict[str, Type] = {}
        resolving: Set[str] = set()

        def resolve(name: str) -> Type:
            if name in resolved:
                return resolved[name]
            if name in resolving:
                raise SemanticError(
                    f"Type alias '{shown(name)}' is defined in terms of "
                    f"itself (a cycle)",
                    seen[name],
                )
            resolving.add(name)
            result = resolve_target(seen[name].target_type, seen[name])
            resolving.discard(name)
            resolved[name] = result
            return result

        def resolve_target(target, alias_node: TypeAlias) -> Type:
            """`alias_node` is for error positions."""
            if isinstance(target, ArrayTypeExpr):
                size = target.size if isinstance(target.size, int) else self.facts.array_sizes[target.nid]
                return Type(TypeKind.ARRAY, element_type=resolve_target(target.element_type, alias_node), size=size)
            if isinstance(target, SliceTypeExpr):
                return Type(TypeKind.SLICE, element_type=resolve_target(target.element_type, alias_node))
            if isinstance(target, str) and target in TYPE_NAMES:
                return TYPE_NAMES[target]
            saved_scope = self.scope
            self._enter(alias_node)
            target = self._resolve_type_name(target)
            self.scope = saved_scope
            if target in seen:
                return resolve(target)
            if target in structs:
                return Type(TypeKind.STRUCT, struct_name=target)
            if target in self.decls.enums:
                return Type(TypeKind.ENUM, enum_name=target)
            raise SemanticError(
                f"Unknown type '{target}' in a type alias's own target "
                f"-- expected int, bool, str, a struct or enum name, or "
                f"another type alias",
                alias_node,
            )

        for ad in alias_defs:
            resolve(ad.name)
        return resolved

    def _resolve_struct_fields(self, struct_defs: List[StructDef], registry: Dict[str, StructInfo],
                               sum_names: Dict[str, None]) -> Dict[str, StructInfo]:
        """Resolve field types (which may name structs and sum types declared anywhere)."""
        for sd in struct_defs:
            self._enter(sd)
            fields: Dict[str, Type] = {}
            for f in sd.fields:
                if f.name in fields:
                    raise SemanticError(
                        f"Field '{f.name}' is already declared in struct '{shown(sd.name)}'",
                        f,
                    )
                fields[f.name] = type_from_name(
                    f.field_type, registry, self.decls.type_aliases, f, sum_names, resolve=self._resolve_type_name,
                    enums=self.decls.enums, array_sizes=self.facts.array_sizes)
            registry[sd.name] = StructInfo(name=sd.name, fields=fields)
        return registry

    def _resolve_sum_types(
            self, sum_type_defs: List[SumTypeDef], structs: Dict[str, StructInfo]) -> Dict[str, SumTypeInfo]:
        """Resolve sum type variants and check name collisions. A sum named as a variant gives its
        own variants in its place (and so on down), so every sum's variants are a flat list of types
        that aren't sums, each once: what arrives twice through different sums is one variant."""
        registry: Dict[str, SumTypeInfo] = {}
        sum_names = {std.name: None for std in sum_type_defs}
        by_name: Dict[str, SumTypeDef] = {}
        for std in sum_type_defs:
            by_name.setdefault(std.name, std)
        resolving: List[str] = []  # the sums whose variants are being worked out, outermost first

        def variants_of(std: SumTypeDef) -> List[Type]:
            if std.name in registry:
                return registry[std.name].variants
            if std.name in resolving:
                through = [f"'{shown(name)}'" for name in resolving[resolving.index(std.name) + 1:]]
                raise SemanticError(
                    f"Sum type '{shown(std.name)}' includes itself as a variant"
                    + (f" (through {', then '.join(through)})" if through else "")
                    + " -- a sum type's variants are those of the sums it names, so one can't name itself",
                    by_name[resolving[-1]])
            resolving.append(std.name)
            scope = self.scope
            try:
                variants = self._flat_variants(std, structs, sum_names, by_name, variants_of)
            finally:
                self.scope = scope
                resolving.pop()
            registry[std.name] = SumTypeInfo(name=std.name, variants=variants)
            return variants

        declared = set()
        for std in sum_type_defs:
            if shown(std.name) in BUILTIN_FUNCTION_NAMES:
                raise SemanticError(
                    f"'{shown(std.name)}' is a builtin and can't be used as a "
                    f"sum type name",
                    std,
                )
            if std.name in structs:
                raise SemanticError(
                    f"Sum type '{shown(std.name)}' collides with a struct of "
                    f"the same name -- struct and sum-type names share "
                    f"one namespace and can never be the same",
                    std,
                )
            if std.name in self.decls.type_aliases:
                raise SemanticError(
                    f"Sum type '{shown(std.name)}' collides with a type alias "
                    f"of the same name -- type-alias and sum-type names "
                    f"share one namespace and can never be the same",
                    std,
                )
            if std.name in declared:
                raise SemanticError(f"Sum type '{shown(std.name)}' is already declared", std)
            declared.add(std.name)
            variants_of(std)
        return registry

    def _flat_variants(self, std: SumTypeDef, structs: Dict[str, StructInfo], sum_names: dict,
                       by_name: Dict[str, SumTypeDef], variants_of: Callable) -> List[Type]:
        """The variants of `std`, in the order written, each sum it names replaced by that sum's."""
        self._enter(std)
        variants: List[Type] = []
        written: list = []  # what std itself lists: a Type, or the name of a sum
        for variant_name in std.variants:
            named_sum = None
            if isinstance(variant_name, (str, QualifiedTypeExpr)):
                named_sum = by_name.get(self._resolve_type_name(variant_name))
            if named_sum is not None:
                if named_sum.name in written:
                    raise SemanticError(
                        f"Sum type '{shown(std.name)}' lists '{variant_name}' as a variant more than once", std)
                written.append(named_sum.name)
                included = variants_of(named_sum)
                self._enter(std)  # (working those out was done in that sum's file)
                variants.extend(v for v in included if v not in variants)
                continue
            try:
                # A pointer to a sum, or a slice or dict of them, is a variant in its own right:
                # the names of the sums are enough to resolve those.
                variant_type = type_from_name(
                    variant_name, structs, self.decls.type_aliases, std, sum_names, resolve=self._resolve_type_name,
                    enums=self.decls.enums, array_sizes=self.facts.array_sizes)
            except SemanticError:
                # Name what's allowed for a simple typo.
                if not isinstance(variant_name, str):
                    raise
                raise SemanticError(
                    f"Sum type '{shown(std.name)}' names '{variant_name}' as "
                    f"a variant, but '{variant_name}' isn't a declared "
                    f"struct, sum type, `none`, or a valid scalar/str/array/slice/pointer/dict "
                    f"type",
                    std,
                )
            if variant_type in written:
                raise SemanticError(
                    f"Sum type '{shown(std.name)}' lists '{variant_type}' "
                    f"as a variant more than once",
                    std,
                )
            written.append(variant_type)
            if variant_type not in variants:
                variants.append(variant_type)
        return variants

    def _check_value_containment(self, program: Program) -> None:
        """No struct or sum type may contain itself by value (directly, through arrays, or through
        another struct or sum): it would have no finite size. Pointers, slices and dicts are
        indirections, so recursion through them is fine."""
        decl = {sd.name: sd for sd in program.structs}
        decl.update({std.name: std for std in program.sum_types})

        def embedded(t: Type) -> Optional[str]:
            while t.kind == TypeKind.ARRAY:
                t = t.element_type
            if t.kind == TypeKind.STRUCT:
                return t.struct_name
            if t.kind == TypeKind.SUM:
                return t.sum_type_name
            return None

        def contents(name: str) -> List[Tuple[str, str]]:
            """(member description, embedded type name) for each by-value member."""
            if name in self.decls.structs:
                fields = self.decls.structs[name].fields
                pairs = [(f"{shown(name)}.{field}", embedded(t)) for field, t in fields.items()]
            else:
                pairs = [(f"{shown(name)}'s {t} variant", embedded(t)) for t in self.decls.sum_types[name].variants]
            return [(desc, inner) for desc, inner in pairs if inner is not None]

        def visit(name: str, path: List[Tuple[str, str]]) -> None:
            for desc, inner in contents(name):
                if any(n == inner for n, _ in path) or inner == path[0][0]:
                    chain = [d for _, d in path[1:]] + [desc]
                    raise SemanticError(
                        f"'{inner}' contains itself by value ({' -> '.join(chain)}), so it would have no "
                        f"finite size -- hold it through a pointer (*{inner}), slice, or dict instead",
                        decl[inner])
                visit(inner, path + [(inner, desc)])

        for name in decl:
            visit(name, [(name, name)])

    def _collect_methods(self, program: Program) -> Dict[Tuple[str, str], Tuple[List[Type], Type, str]]:
        """Reject duplicate method names per struct or enum; return the method registry."""
        methods: Dict[Tuple[str, str], Tuple[List[Type], Type, str]] = {}
        for sd in list(program.structs) + list(program.enums):
            self._enter(sd)
            kind = 'enum' if isinstance(sd, EnumDef) else 'struct'
            members = {member.name for member in sd.members} if isinstance(sd, EnumDef) else set()
            seen_names: Set[str] = set()
            for md in sd.methods:
                if md.name in seen_names:
                    raise SemanticError(
                        f"Method '{shown(md.name)}' is already declared on "
                        f"{kind} '{shown(sd.name)}'",
                        md,
                    )
                if md.name in members:  # `Color.Red` and `c.Red()` would read as the same thing
                    raise SemanticError(
                        f"Method '{md.name}' has the same name as a member of enum '{shown(sd.name)}'", md)
                seen_names.add(md.name)
                param_types = [
                    self._type(p.type, p) for p in md.params
                ]
                return_type = self._return_type(md)
                methods[(sd.name, md.name)] = (param_types, return_type, mangle_method_name(sd.name, md.name))
                if md.receiver_is_pointer:
                    self.decls.pointer_receivers.add((sd.name, md.name))
        return methods

    def check_extern_function_decl(self, ext: ExternFunctionDecl) -> None:
        """Validate an extern signature (scalars and pointers only) and register it."""
        if shown(ext.name) in BUILTIN_FUNCTION_NAMES:
            raise SemanticError(
                f"'{shown(ext.name)}' is a builtin and can't be redefined as "
                f"an extern function",
                ext,
            )
        if ext.name in self.decls.structs:
            raise SemanticError(
                f"Extern function '{shown(ext.name)}' collides with a struct "
                f"of the same name -- struct and function names share "
                f"one namespace and can never be the same, since "
                f"'{shown(ext.name)}(...)' would otherwise be ambiguous "
                f"between a call and a struct literal",
                ext,
            )
        if ext.name in self.decls.type_aliases:
            raise SemanticError(
                f"Extern function '{shown(ext.name)}' collides with a type "
                f"alias of the same name -- function and type-alias "
                f"names share one namespace and can never be the same",
                ext,
            )
        if ext.name in self.decls.sum_types:
            raise SemanticError(
                f"Extern function '{shown(ext.name)}' collides with a sum "
                f"type of the same name -- function and sum-type "
                f"names share one namespace and can never be the same",
                ext,
            )
        if ext.name in self._extern_decls:
            raise SemanticError(
                f"Extern '{shown(ext.name)}' is already declared in {_file_and_line(self._extern_decls[ext.name])} "
                f"-- declare an extern once and import it where else it is needed", ext)
        if ext.name in self.decls.functions:
            # A function of the entry file: only there is a function's key its bare name, like an extern's.
            function = next((fn for fn in self.decls.all_functions if fn.name == ext.name), None)
            raise SemanticError(
                f"Function '{shown(ext.name)}' has the same name as an extern declared in {_file_and_line(ext)} "
                f"-- rename the function", function if function is not None else ext)
        self._extern_decls[ext.name] = ext

        param_types = [self._type(p.type, p) for p in ext.params]
        return_type = self._return_type(ext)

        for p, p_type in zip(ext.params, param_types):
            if p_type.kind in (TypeKind.ARRAY, TypeKind.SLICE, TypeKind.STRUCT, TypeKind.SUM, TypeKind.STR):
                raise SemanticError(
                    f"Extern function '{shown(ext.name)}''s parameter '{p.name}' has "
                    f"type {p_type} -- only scalar and pointer types are "
                    f"supported in an extern function's signature for now "
                    f"(array/slice/struct/sum/str-typed parameters aren't yet)",
                    p,
                )
        if return_type.kind in (TypeKind.ARRAY, TypeKind.SLICE, TypeKind.STRUCT, TypeKind.SUM, TypeKind.STR):
            raise SemanticError(
                f"Extern function '{shown(ext.name)}' returns {return_type} -- "
                f"only scalar and pointer types are supported as an "
                f"extern function's own return type for now "
                f"(array/slice/struct/sum/str aren't yet)",
                ext,
            )

        self.decls.functions[ext.name] = (param_types, return_type)

    def check_intrinsic_decl(self, ic: IntrinsicDecl) -> None:
        """Validate an intrinsic signature and register it."""
        if shown(ic.name) in BUILTIN_FUNCTION_NAMES:
            raise SemanticError(
                f"'{shown(ic.name)}' is a builtin and can't be redefined as "
                f"an intrinsic",
                ic,
            )
        if ic.name in self.decls.structs:
            raise SemanticError(
                f"Intrinsic '{shown(ic.name)}' collides with a struct "
                f"of the same name -- struct and function names share "
                f"one namespace and can never be the same, since "
                f"'{shown(ic.name)}(...)' would otherwise be ambiguous "
                f"between a call and a struct literal",
                ic,
            )
        if ic.name in self.decls.type_aliases:
            raise SemanticError(
                f"Intrinsic '{shown(ic.name)}' collides with a type "
                f"alias of the same name -- function and type-alias "
                f"names share one namespace and can never be the same",
                ic,
            )
        if ic.name in self.decls.sum_types:
            raise SemanticError(
                f"Intrinsic '{shown(ic.name)}' collides with a sum "
                f"type of the same name -- function and sum-type "
                f"names share one namespace and can never be the same",
                ic,
            )
        if ic.name in self.decls.functions:
            raise SemanticError(f"Function '{shown(ic.name)}' is already declared", ic)

        param_types = [self._type(p.type, p) for p in ic.params]
        return_type = Type.VOID if ic.return_type is None else self._type(ic.return_type, ic)

        self.decls.functions[ic.name] = (param_types, return_type)
        self.decls.intrinsic_original_names[ic.name] = ic.original_name

    def _check_enum_name_collisions(self, program: Program) -> None:
        for ed in program.enums:
            for kind, table in (('function', self.decls.functions), ('struct', self.decls.structs),
                                ('constant', self.constants.decls), ('type alias', self.decls.type_aliases),
                                ('sum type', self.decls.sum_types)):
                if ed.name in table:
                    self._enter(ed)
                    raise SemanticError(f"Enum '{shown(ed.name)}' collides with a {kind} of the same name", ed)

    def _check_const_name_collisions(self, program: Program) -> None:
        for name, cd in self.constants.decls.items():
            for kind, table in (('function', self.decls.functions), ('struct', self.decls.structs),
                                ('type alias', self.decls.type_aliases), ('sum type', self.decls.sum_types)):
                if name in table:
                    raise SemanticError(f"Constant '{shown(name)}' collides with a {kind} of the same name", cd)
            if shown(name) in BUILTIN_FUNCTION_NAMES:
                raise SemanticError(f"'{shown(name)}' is a builtin and can't be used as a constant name", cd)
