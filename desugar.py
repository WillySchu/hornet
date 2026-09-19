"""Method desugaring: lowers every struct method declaration into an
ordinary, mangled-name Function, appended to program.functions --
called once, right after parsing and before semantic.analyze() ever
runs (see compile_to_asm/compile.py). Has to run BEFORE analyze(),
not after or interleaved partway through it: analyze() type-checks
every function's own body, synthesized method-functions included, by
iterating program.functions near the very end of its own pipeline --
a method's own body is checked at all only because it's already been
turned into an ordinary Function by the time that step runs, no
different from one the programmer wrote directly. Desugaring after
analyze() completes would mean that step never sees a method's own
body in the first place, silently skipping it entirely.

Purely structural, and never raises: uses each method's own declared
parameter/return type NAMES directly (Param's own `type` field, a
plain string) rather than a resolved Type object, since there isn't
one yet at this point in the pipeline -- struct/type-alias resolution
hasn't run at all yet. Validating an actual call site against a
method's own signature (receiver is actually a struct, argument
count/types match, the method exists at all) stays entirely in
semantic.py's own _check_method_call, which needs real, resolved types
to do any of that; likewise the duplicate-method-name-on-one-struct
check semantic.py's own _collect_methods still performs, even though
it's a check this module could technically make without any type
information -- keeping ALL validation in the validation step, with
nothing here able to reject a program, not just the checks that
happen to need type information to run.

semantic.py's own _collect_methods still independently re-derives the
resolved parameter/return types and the identical mangled name (via
mangle_method_name, shared here so the two can never disagree) once
the struct registry it needs actually exists, to build the lookup
table _check_method_call resolves a call site's own method through."""

from parser import Function, Param, Program


def mangle_method_name(struct_name: str, method_name: str) -> str:
    """`StructName.methodName` -- '.' can't appear in a Hornet
    IDENTIFIER, so this structurally can't collide with any free
    function, another struct's method, or a builtin; no explicit
    collision check needed against anything but another method of the
    same name on the SAME struct (see semantic.py's own _collect_
    methods for that one)."""
    return f"{struct_name}.{method_name}"


def desugar_methods(program: Program) -> None:
    """For every struct's own methods: synthesizes an ordinary
    Function (the receiver becomes a typed first Param, ahead of the
    method's own declared ones) and appends it to program.functions in
    place -- see this module's own docstring for why this has to
    happen here, before semantic analysis, rather than during or after
    it."""
    for sd in program.structs:
        for md in sd.methods:
            # Neither the synthesized receiver Param nor the Function
            # itself has a token of its own -- both take the MethodDef's
            # own position, so a semantic error blamed on either (e.g. a
            # duplicate-method-name check) still points somewhere real
            # in the source; every statement in body already carries its
            # own real position from parsing, untouched here.
            receiver_param = Param(name=md.receiver_name, type=sd.name, line=md.line, col=md.col)
            program.functions.append(Function(
                name=mangle_method_name(sd.name, md.name),
                return_type=md.return_type,
                params=[receiver_param] + md.params,
                body=md.body,
                line=md.line, col=md.col,
            ))
