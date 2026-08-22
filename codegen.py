"""Codegen

Transforms the AST produced by parser.py into x64 assembly (AT&T syntax).

This follows the classic two-stage codegen structure used by many
introductory compilers:

    AST (parser.py)  -->  Assembly AST  -->  Assembly text

Keeping the "Assembly AST" as its own intermediate representation --
rather than emitting strings directly from the source AST -- makes this
much easier to extend later. Once expressions get more complex you'll
need to allocate registers/stack slots as part of building that assembly
AST; once you add more statements you'll add more instruction types.
None of that should require touching the string-formatting code at all,
and none of the string-formatting quirks (spacing, symbol prefixes, etc.)
should leak into the code-generation logic.

Supported so far (matches what parser.py can currently produce):

    Program  -> one or more Functions
    Function -> a name and a body of statements
    Return   -> evaluate an expression into %eax, then `ret`
    Constant -> an immediate value
    Unary    -> NEGATE ('-'), COMPLEMENT ('~'), and NOT ('not'), each
                applied in place to whatever's already in the
                destination register
    Binary   -> ADD ('+'), SUBTRACT ('-'), MULTIPLY ('*'), DIVIDE ('/'),
                MODULO ('%'), the bitwise operators (& | ^ << >>), the
                six comparisons (== != < > <= >=), and the two
                short-circuiting logical operators (and, or)
    Variable -> a read of a local variable's stack slot
    VarDecl  -> `int a` or `int a = <expr>`
    Assign   -> `a = <expr>`
    ExprStmt -> a bare expression statement, evaluated and discarded
    BoolLiteral -> `true`/`false`, immediate 1/0
    If       -> `if`/`elif`/`else`, real conditional jumps (elif is
                nested If, not a separate case -- see parser.py)
    While    -> real loop control flow, condition re-checked before
                every iteration including the first
    Break    -> unconditional jump to the innermost loop's end
    Continue -> unconditional jump back to the innermost loop's
                condition check

A note on typing: as of semantic.py, this file is no longer the only
line of defense against a malformed program -- undeclared/re-declared
variables and every type error are now caught earlier, by a dedicated
semantic-analysis pass that runs between parsing and codegen. This file
still doesn't know anything about int vs bool (both are just 4-byte
values to it; see BoolLiteral's codegen), and still doesn't need to,
since semantic.py already guaranteed the program type-checks before this
ever runs. `generate_asm` deliberately stays a pure "AST -> assembly
text" function with no semantic checking baked in, so it composes
cleanly with pipelines that already validated the AST some other way;
`compile_to_asm` is the file-based convenience wrapper that actually
chains lex -> parse -> analyze -> codegen together, and is what the CLI
below uses. A caller going through `generate_asm` directly is expected
to have called semantic.analyze() itself first.

Unlike Constant, a Unary expression can't be represented as a bare
Operand (there's no such thing as "the immediate value of `-2`" as an
assembly-level concept -- the negation has to actually happen on a
register). So expression codegen works by emitting instructions that
compute the expression's value directly into a destination (see
`gen_expr_into`), rather than by returning an Operand. Nested unary
operators fall out of this for free: `~-2` recurses inward first
(computing -2 into %eax), then applies the outer operator to whatever's
now sitting in that register (`notl %eax`).

BINARY OPERATORS AND THE ONE-REGISTER PROBLEM
------------------------------------------------
Binary operators need *two* operand values alive at once (left and
right), but `gen_expr_into` only has one destination register to work
with -- there's no register allocator yet to hand out a second one. The
fix used here is the classic minimal-effort answer: spill to the real
CPU stack.

To compute `left OP right` into `dst` (%eax):
  1. Compute `left` into %eax (the usual gen_expr_into).
  2. `pushq %rax` -- save that value on the stack.
  3. Compute `right` into %eax -- safe to reuse, since `left` is on the
     stack now.
  4. `movl %eax, %ecx` -- move `right` out of the way into a scratch
     register.
  5. `popq %rax` -- restore `left` back into %eax.
  6. Emit the actual operator instruction combining %ecx into %eax.

Because this always pushes before recursing into the right-hand side and
pops after, nested binary expressions of any depth resolve correctly:
each push/pop pair is balanced within its own gen_binary_into call
regardless of what further pushes and pops happen inside step 3, so the
stack naturally behaves like an expression-evaluation stack. This is not
efficient (a real register allocator would keep far more values in
registers), but it's correct at arbitrary nesting depth, which is what
matters before you have one.

COMPARISONS
-----------
The six comparison operators (== != < > <= >=) reuse that exact scheme
-- both sides always get evaluated -- but combine with cmpl + setCC +
movzbl instead of an arithmetic instruction: cmpl sets flags from
(left - right), setCC turns the relevant flag pattern into a 0/1 byte,
movzbl zero-extends it to fill the register. This is the same trick
logical NOT already used against a fixed comparand of 0 (see
gen_unary_op); comparisons just generalize it to comparing against a
computed `right` and to all six condition codes instead of only 'e'.

AND / OR: SHORT-CIRCUITING
---------------------------
AND and OR are NOT routed through the stack-spill scheme above, because
that scheme always evaluates both operands -- exactly what short-circuit
evaluation must avoid. `a and b` must not evaluate `b` at all if `a` is
already false, and `a or b` must not evaluate `b` at all if `a` is
already true.

Instead, gen_short_circuit emits real control flow: evaluate the left
side, compare it to 0, and conditionally jump straight past the code
that would evaluate the right side. AND and OR are mirror images of each
other and share one implementation:

    AND (jump early on left == 0, early result 0, fallthrough result 1):
        <left>              ; -> %eax
        cmpl $0, %eax
        je   .Land_short_N  ; left was false -- skip right entirely
        <right>             ; -> %eax
        cmpl $0, %eax
        je   .Land_short_N  ; right was false
        movl $1, %eax
        jmp  .Land_end_N
    .Land_short_N:
        movl $0, %eax
    .Land_end_N:

    OR is the same shape with the jump condition, early value, and
    fallthrough value each flipped (jne / 1 / 0).

Because the jump genuinely skips over the right-hand side's instructions
at runtime -- they're present in the binary but never executed -- this
is real short-circuiting, not just a coincidentally-correct value. That
distinction is externally observable: `0 and (1 / 0)` returns 0 without
crashing, while `1 and (1 / 0)` does crash (SIGFPE), because only in the
second case does control flow ever reach the division.

LOCAL VARIABLES
----------------
Every local variable lives in a fixed stack slot at a constant offset
from %rbp -- the classic frame-pointer-relative layout, e.g. the first
variable declared in a function at -4(%rbp), the second at -8(%rbp), and
so on (each int/bool is 4 bytes). _collect_locals recursively pre-scans
a function's whole body -- including into every If's then_body and
else_body, and every While's body -- for VarDecls *before* generating
any instructions, so every slot's offset is already assigned, and the
total frame size is already known, by the time the prologue is emitted:

    pushq %rbp
    movq  %rsp, %rbp
    subq  $N, %rsp      ; only if the function has any locals at all

Every function gets this prologue -- and a matching `leave` right before
every `ret` -- even ones with no locals at all, rather than only doing
it conditionally. That costs a couple of extra instructions on trivial
functions (`return 2` now sets up and tears down an empty frame), but it
keeps codegen uniform now and means there's no special case to unwind
later once this needs to support more than one frame per program (e.g.
actual function calls).

TWO NAMES, TWO NUMBERS: WHY OFFSETS ARE KEYED BY NODE, NOT NAME
-------------------------------------------------------------------
Now that if/else exist, semantic.py allows two variables in sibling
branches to share a name -- `if x: int a = 1` / `else: int a = 2` are
independent scopes, so that's legitimately two different variables that
happen to be spelled the same way. A single `Dict[str, int]` allocator
can't represent that (the second VarDecl would look like a duplicate of
the first). So _collect_locals keys offsets by `id(vardecl_node)`
instead of by name -- every VarDecl anywhere in the function, no matter
how deeply nested or how many others share its name, gets its own
permanent slot. This is simple and always correct, but not
space-optimal: two variables that can never both be alive at once (like
the `a` in an if and the `a` in its else) still each get dedicated stack
space for the whole function call, rather than sharing a slot. Trading a
few bytes of stack for a much simpler allocator felt like the right
call at this stage.

A while loop's body follows the exact same rule, for a related but
distinct reason: a VarDecl inside a loop body is only ever encountered
*once* by _collect_locals (the pre-scan walks the AST, a static tree,
not a simulation of runtime iterations), so it only ever gets one slot
-- and that's correct, because every iteration reuses that same slot by
just overwriting it. There's no "which iteration's `a`" question the
way there's a "which branch's `a`" question for if/else, so nothing
extra is needed here beyond recursing into While bodies the same way
_collect_locals already recurses into If branches.

Fixed, permanent offsets alone aren't enough to make `Variable`/`Assign`
nodes resolve to the *right* slot, though -- codegen still needs to know
which of possibly-several same-named VarDecls a given reference means at
the point it's generated, which is exactly a scoping question. So
`self.scopes: List[Dict[str, int]]` mirrors semantic.py's own scope
stack (pushed/popped around an If's then/else bodies and a While's body,
name -> offset instead of name -> Type), and _local_offset walks it
innermost-to-outermost exactly like semantic.py's _lookup does. This
does mean scope resolution is implemented twice in this codebase, once
per pass -- a real duplication, and a reasonable one to revisit (e.g. by
having semantic.py annotate each Variable/Assign node with its resolved
VarDecl directly) if it ever drifts out of sync with semantic.py's own
rules.

Reading or writing a variable never touches gen_expr_into's core
contract -- every expression still always computes into a register
(%eax). A variable read is just one more expr kind: `Mov(src=Memory(...),
dst=dst)`. A variable *write* (VarDecl-with-initializer, or Assign) is
handled one level up, in `_gen_store`: evaluate the expression into %eax
exactly as always, then a single extra `movl %eax, offset(%rbp)` to
actually store it. This is deliberately not "generate straight into the
memory operand" -- that would mean threading Memory destinations through
gen_binary_into's push/pop scheme, division's %eax requirement, and the
byte-register aliasing comparisons and NOT depend on, none of which are
built to target anything but a register. Routing every store through
%eax first avoids all of that at the cost of one extra mov per store.

As of semantic.py, double declarations, undeclared references, and
every type error are caught well before this file ever runs -- the
`_local_offset` lookup failing here would now only happen if codegen
were invoked directly on an AST that skipped semantic analysis (see this
module's top for the compile_to_asm/generate_asm split), so treat it as
a defensive check rather than the primary error-reporting path.

LOOPS
------
A while loop reuses the same condition-into-%eax-then-compare-to-0
pattern used everywhere else real control flow shows up (short-circuit
AND/OR, if/else): evaluate the condition, jump past the body if it's
false. What's new here is that the condition also needs to be
*re-checked* after the body runs, which just means the body's closing
instruction is an unconditional jump back up to a label placed right
before the condition, rather than falling through to whatever comes
next -- see gen_while's docstring for the exact shape.

break/continue are both just an unconditional Jmp to one of that loop's
two labels -- break to the end (falls out of the loop entirely),
continue to the start (re-checks the condition, which is exactly what
"skip the rest of this iteration" means here, since there's no
per-iteration cleanup step distinct from the condition check). The only
real question either one has to answer is *which* loop's labels, once
loops can nest -- `self.loop_labels: List[Tuple[str, str]]` is a stack
for exactly that reason, pushed with the current loop's (start, end)
pair before its body is generated and popped once it's done, mirroring
semantic.py's loop_depth counter (see its LOOPS section) closely enough
that it's worth noting the difference: semantic.py only needs to know
*whether* a break/continue is inside some loop, so a counter suffices;
codegen needs to know *which* loop's labels to jump to, so it needs the
actual label pair, not just a depth.

STRINGS
--------
A `str` value is a plain pointer -- to a null-terminated buffer, either
static (a literal, emitted into `.data`; see gen_string_literal_into and
Emitter.emit) or malloc'd (a concatenation result; see
gen_string_concat_into). Being a pointer, it's 8 bytes on x86-64, not
the 4 bytes every value before it fit in -- which is why _collect_locals
gives *every* local a uniform 8-byte slot now regardless of type,
trading a few wasted bytes on int/bool locals for not needing a
variable-width allocator at all.

Concatenation and equality both need real work at runtime -- there's no
way to fold `a + b` or `a == b` at compile time once `a`/`b` are
variables -- so they call real C library functions (`strlen`/`malloc`/
`strcpy`/`strcat` for concatenation, `strcmp` for equality) via the
actual SysV calling convention. This is the first place in this
compiler that calls anything external at all, and it surfaced a real
bug worth understanding, not just the fix for it:

Both operations need `left`'s value to survive while `right` gets
computed. The first implementation stashed `left` in a fixed register
(%rbx) for the duration -- which works fine as long as evaluating
`right` never *also* needs %rbx. But `right` can itself be another
string concatenation or comparison (`(a + b) + c`, or a function that
does its own string work), and that nested evaluation reaches this
exact same method again, which unconditionally overwrites %rbx as its
own first step -- silently clobbering the outer call's `left` before
it's ever used. This didn't show up in earlier testing because those
tests only ever nested a string op on one side of another (`(a+b)+c`,
where `c` is a plain variable); it takes something on the scale of `(a
+ b) == (c + d)` -- a nested op on *both* sides of another -- to
actually hit it.

The fix (see gen_string_concat_into/gen_string_compare_into) is to stop
relying on %rbx surviving on its own, and instead push `left` onto the
real stack before evaluating `right` -- the exact push-before-recursing
scheme gen_binary_into already uses for ordinary int/bool operators,
which has no such fixed-identity conflict no matter how deeply it
nests, since every push has its own place on the stack regardless of
what any nested call does to any register. `right` gets captured into
%r12 immediately after its own evaluation completes, which *is* safe as
a fixed register, since nothing recursive happens between that capture
and this method's own further use of it -- only the fixed strlen/
malloc/strcpy/strcat sequence, which can't reach back into
gen_expr_into. %r13/%r14 are similarly write-once, straight from a
direct call result, with nothing recursive after.

This is a *different* concern from why every function's own prologue/
epilogue also saves and restores %rbx/%r12/%r13/%r14 unconditionally
(see gen_function and gen_return, and this section's FUNCTIONS
neighbor) -- that fix protects a value held in one of these registers
*across a call into another function*; this one protects a value held
here *across evaluating a nested expression within the same function*,
call or no call. Both are genuinely necessary; neither one covers what
the other does.

Never frees a *named* concatenation result -- deciding whether a value
in a variable is safe to free needs real escape analysis (has it been
returned? assigned somewhere else? handed to another function that
might retain it?), which semantic.py doesn't do at all, and which is a
meaningfully bigger problem than concatenation itself. What it *does*
free now: the moment a concatenation's operand is itself a fresh,
unnamed concatenation result -- a Binary(ADD, ...) sub-expression that
was never stored into a variable, returned, or passed anywhere, and so
could not possibly be referenced by anything else -- its buffer is
freed immediately after its bytes are copied out (see
_gen_free_if_fresh_concat, called from both gen_string_concat_into and
gen_string_compare_into). That's a real, narrow, decidable-from-the-AST
case: `str r = a + b + c` mallocs three buffers under the hood, but the
intermediate `a + b` result is never reachable by anything once `+ c`
has copied its bytes into the final buffer, so it's freed rather than
silently discarded. A StringLiteral (static `.data`, never
heap-allocated), a Variable (might be read again, or aliased elsewhere
-- codegen has no visibility into that), and a Call's return value
(same problem, plus it might be a static literal or a passed-through
parameter for all codegen can tell) are all deliberately left alone.
This was verified directly during development with an LD_PRELOAD
malloc/free tracer, confirming both that frees only ever target live,
previously-malloc'd pointers (never a literal, never a double-free) and
that the reduction is real: a 6-way concatenation chain drops from 5
un-freed buffers to 1.

FUNCTIONS
----------
Parameters arrive in registers per the SysV ABI (%rdi, %rsi, %rdx,
%rcx, %r8, %r9, in that order -- see _ARG_REGISTERS_64/_32) and get
moved into their own stack slots immediately in the prologue, via
_collect_params/_bind_param -- from that point on a parameter is
indistinguishable from any other local to the rest of codegen. Only the
first 6 arguments are supported, matching how far the ABI's
register-passing goes before falling back to the stack; a 7th parameter
or argument is a CodegenError, not a silent miscompile.

A call's arguments are each fully evaluated (via the ordinary
gen_expr_into, so a nested call or a string concatenation as an
argument works correctly) and immediately pushed onto the stack, one at
a time, *before* any of them get popped into the actual argument
registers -- see gen_call_into. Only once every argument is safely
stacked does popping begin, in reverse, into %rdi/%rsi/etc. This is the
same "compute now, protect on the stack, place into position later"
principle behind the STRINGS section's fix above, for the same reason:
if argument 2 happened to be a string concatenation, and argument 1's
value were sitting in some fixed register instead of on the stack while
argument 2 gets computed, argument 2's own scratch usage could corrupt
argument 1.

Every function's prologue unconditionally pushes %rbx/%r12/%r13/%r14
right after setting up %rbp, and every return pops them back before
`leave` (see gen_function/gen_return) -- regardless of whether that
particular function happens to do any string work itself. This is what
makes it safe for function A, mid-string-operation with a value
currently sitting in one of those registers, to call function B: B's
own prologue saves whatever A left there, B is free to use those
registers however it wants internally, and B's epilogue restores A's
values before control returns -- the standard meaning of "callee-saved"
applied uniformly, rather than reasoned about per call site. Note this
protects a value *across a call*; it does not, on its own, protect a
value across a *nested expression* evaluated in the same function
without a call in between -- that's what the stack-based fix in the
STRINGS section above is for. Both together are what make deeply nested
string operations, with or without function calls mixed in, safe
regardless of how deep the nesting goes.

self.functions is never built here -- semantic.py already guaranteed
every call resolves to a real function with matching argument types and
count before this file ever runs (see this module's top for the
compile_to_asm/generate_asm split). This file doesn't need its own copy
of function signatures at all, including return types -- a Call
expression's type (int/bool/str) needed for e.g. deciding whether
`foo() + bar()` means arithmetic or concatenation is read directly via
_type_of from what semantic.py already resolved, not re-looked-up here.

FUNCTIONS WITH NO DECLARED RETURN TYPE
-------------------------------------------
fn.return_type is None (see Function's own docstring in parser.py) for
`def NAME(params):` -- a function with no declared return type at all.
Mapped to Type.VOID, the same internal-only sentinel semantic.py's own
analyze_function already uses, right at the top of gen_function --
kept as one shared representation across both files rather than a
second, independent "no return type" encoding invented here.

Two things follow from this, one at each end of the function:

Return values: gen_return's stmt.value can be None (a bare `return`,
valid exactly when semantic.py already confirmed this function has no
declared return type -- see analyze_return) -- nothing needs computing
at all in that case, just the ordinary epilogue (see _gen_epilogue).

Falling off the end: every OTHER function relies on semantic.py's
always_returns already having guaranteed an explicit `return` on every
path (see the module docstring's ALL PATHS RETURN section in
semantic.py), which is what lets gen_function get away with never
emitting its own trailing epilogue -- some gen_return-emitted one is
always guaranteed to execute first, making a trailing one permanently
unreachable dead code. A function with no declared return type
deliberately skips that guarantee, since falling off the end IS how
such a function is expected to exit when it doesn't return early -- so
gen_function appends a trailing epilogue unconditionally in this one
case, or execution would fall straight through into whatever comes
next in the generated assembly instead (the bounds-check panic block,
or the next function's own prologue) -- a real, silent crash, not a
hypothetical one. Appended even when this particular body happens to
already return explicitly on every path (e.g. via an if/else where
both branches return): there's no cheap way to know that without
effectively re-running always_returns here too, and an extra,
unreachable epilogue costs nothing but a few bytes.

_gen_epilogue itself (restore every callee-saved scratch register,
then leave/ret) is shared, unchanged code between gen_return's bare-
return case and gen_function's trailing one -- both are exactly the
same "there's no value to compute, just exit the function cleanly"
situation, just reached from a different place.

No changes were needed anywhere calls themselves are generated
(gen_call_into, gen_expr_into's Call dispatch): calling a void function
works exactly like calling any other one, and %eax simply ends up
holding whatever it held before the call (never written to, since
there's no return value to compute) -- harmless, since semantic.py
already guarantees the only context a void call's result can ever
appear in is a bare ExprStmt, which discards whatever's in %eax
unconditionally regardless of what put it there.

ARRAYS
-------
Fixed-size and stack-allocated: an array's size is a compile-time
constant (see parser.py's ArrayTypeExpr), so the whole thing -- for a
[2][3]int, all 24 bytes of it -- lives directly in the local's own
stack slot, contiguous, row-major (the outermost dimension's elements
each occupy a whole inner-array's worth of space, one after another),
exactly the same layout C uses for a fixed-size multi-dimensional
array. This is why _collect_locals/_collect_params stopped using a
uniform 8-byte slot for every local once arrays existed (see LOCAL
VARIABLES above) -- type_byte_width computes a type's real, possibly-
large-and-array-shaped footprint, and every local gets exactly that
much space now, not a fixed amount.

VALUE SEMANTICS, AND WHY THAT'S WHAT MAKES A COPY A COPY
-------------------------------------------------------------
Arrays are values, not references: `b = a` (or `[3]int b = a` as an
initializer) copies every element of `a` into `b`'s own, completely
independent storage -- mutating `b` afterward never affects `a`. This
falls out of gen_array_value_into's Variable case doing a flat,
element-by-element copy (gen_array_copy) between the two variables'
OWN fixed stack offsets, rather than ever copying a pointer the way a
str assignment does. That's also what keeps arrays out of the
aliasing/lifetime problems str's heap-backed, pointer-copied values
have (see STRINGS above) -- an array's storage is exactly as long-lived
as the local variable holding it, no more and no less, so there's
nothing to leak and nothing to free.

ADDRESS COMPUTATION AND BOUNDS CHECKING
-------------------------------------------
gen_index_address_into computes the address of `array[index]` --
element_stride (the size, in bytes, of ONE element -- for a
multi-dimensional array's outer index, that's a whole inner array's
width, e.g. 12 bytes for a [3]int row, not a power of two x86's native
scaled-addressing mode (`offset(base,index,scale)`, which only accepts
scale factors of 1/2/4/8) could always handle) is multiplied by the
index explicitly via imull, rather than relying on that native mode at
all. This is deliberately the SAME approach regardless of whether the
stride happens to be a "nice" value like 4 or 8 or an "awkward" one
like 12 -- one uniform code path, not two, consistent with this
compiler's general preference for a single simple mechanism over a
faster one that only sometimes applies.

Every access is bounds-checked at runtime: one unsigned comparison
(`cmpl $size, %index; jae fail_label`) catches both index >= size and
index < 0 at once, since a negative int, reinterpreted unsigned,
becomes a huge positive number. This isn't just a correctness nicety --
an array's storage sits in the same stack frame as the saved return
address and the callee-saved registers every function call already
depends on (see FUNCTIONS above), so an unchecked out-of-bounds WRITE
could silently corrupt exactly the state that keeps `call`/`ret`
working correctly, not just produce a wrong value. On failure, the
program prints a message and calls abort() (SIGABRT -- a genuine
program bug, not a normal termination, the same character division by
zero's hardware-trapped SIGFPE already has) -- with an explicit
fflush(NULL) between the two, found necessary by testing rather than
assumed: abort() bypasses the normal exit() path that would otherwise
flush libc's buffered stdio, so without the explicit flush the message
was reliably printed to an interactive terminal but silently lost
whenever output was piped or redirected, which is the common case for
a program run non-interactively. The fail-label itself is a single,
per-function jump target shared by every bounds check in that function
(see _get_bounds_check_fail_label), reset fresh per function
(gen_function) -- not duplicated at every individual check site, and
not shared ACROSS functions, since it's a local jump target.

FUNCTION PARAMETERS AND RETURN VALUES
-------------------------------------------
Both are fully supported, via the standard calling-convention
extensions this needs:

RETURN VALUES use a "hidden output pointer": when a function's return
type is an array, the caller passes an extra, FIRST argument -- the
address of where the result should be written -- shifting every real
argument's register one position later (the first real argument goes
in %rsi instead of %rdi, and so on; see gen_array_call_into and
gen_function's own receiving side). Rather than dedicate a register to
holding this pointer for the whole function (which would need its own
save/restore discipline, and -- worse -- would break the callee-saved-
register prologue's even-push-count 16-byte alignment invariant if
added on top of the existing four scratch registers), it gets its own
ordinary stack slot instead, filled by exactly the same "reserve a
slot, then store the incoming register into it" mechanism every real
parameter already uses (see gen_function). gen_return then loads it
back out and hands it to gen_array_value_into as an ordinary Memory
destination -- which is also what makes forwarding one array-returning
call's result straight out of another free (`return inner()`, where
inner also returns an array): the SAME address just gets passed one
level deeper, with no intermediate copy ever materialized.

PARAMETERS: the caller computes the address of its own argument (see
gen_array_arg_address_into -- only a Variable or an Index yielding a
sub-array is supported directly; an ArrayLiteral or a call returning
an array used DIRECTLY as an argument has no address of its own to
point at, and needs to be assigned to a named variable first) and
passes that address as a pointer; the callee copies from it into its
own, already-reserved local slot on entry (gen_function's parameter
loop). After that copy, the parameter is indistinguishable from an
ordinary local array variable for the rest of the function -- this is
also what preserves value semantics across a call: the callee's copy
is independent, so mutating a parameter inside the callee never
affects the caller's own array, the same guarantee `arr2 = arr1`
already gives within a single function.

A function returning an array supports at most 5 real parameters, not
6 -- the hidden pointer itself occupies the first argument register.

A REAL BUG, FOUND THREE TIMES, ONE MECHANISM
-----------------------------------------------
Getting return values working surfaced the same mistake in three
different call paths, each found by testing a different shape of
return value rather than assuming one passing test meant the
underlying mechanism was sound:

  - gen_array_copy unconditionally used %rax as scratch for shuttling
    each element's value -- fine as long as the destination was always
    a fixed %rbp-relative local slot, but broken the moment the
    destination itself became a computed address held in %rax (which
    is exactly what returning an array-typed local variable looks
    like): the very first element copied silently overwrote the
    destination address before any subsequent element could be
    written through it. Fixed by picking a scratch register
    dynamically, guaranteed distinct from both the source's and
    destination's own base register.
  - gen_array_literal_into had the identical problem for a literal
    returned directly (`return [1,2,3]`): evaluating each element's
    value always goes through gen_expr_into, which always computes
    into %eax/%rax, silently destroying a hidden pointer sitting
    there before a single element was ever written. Fixed with a
    push/pop protecting the destination's base register across each
    element's value computation, whenever that base isn't 'rbp' --
    which is never clobbered by anything in this file, and so never
    needs protecting.
  - One layer deeper: returning a sub-array (`return matrix[i]`)
    computes the SOURCE address first -- bounds-checking and index
    arithmetic that freely use %rax/%rcx internally -- before the
    copy ever runs, clobbering the destination the same way. Fixed
    with a small, reusable _gen_protecting_dst_across helper.

All three were genuine segfaults, not wrong answers -- a write
through a destination address that's just been silently overwritten
with unrelated data lands wherever that data happens to point, which
is usually nowhere valid. Worth remembering as a general lesson for
this kind of register-shuffling code: a Memory operand whose base is
a general-purpose register (not 'rbp') has to be treated as fragile
across ANY subsequent code that might use that same register as
scratch, not just across the one method that finally reads or writes
through it.

SIZE-BASED STACK SAFETY
---------------------------
An array over _STACK_ARRAY_LIMIT_BYTES (16KB, hardcoded -- see
is_heap_allocated) is heap-allocated instead of living inline in its
own stack slot: the slot holds an 8-byte pointer to a malloc'd block
rather than the array's data directly. This closes the one concrete
danger fixed-size arrays already had before any of this existed --
nothing stopped a single huge array from silently blowing the stack,
exactly the way it wouldn't in C -- without touching value semantics
at all: `arr2 = arr1` still copies elements rather than aliasing a
pointer regardless of which allocator is backing either side (see
gen_array_copy/gen_array_value_into, which already work with an
arbitrary Memory source/destination and needed no new logic here,
only a different way of getting the address in the first place).

This is deliberately a PER-ARRAY check, not a per-frame budget: it
catches the case that actually matters most (one array declared far
too large for the stack) but not, say, five moderately-sized arrays in
the same function each individually under the limit, or a moderate
array under deep recursion. Both are known, accepted gaps -- closing
them would mean summing every local's width per frame or per call
chain, real additional complexity for a problem this simple check
already solves for the common case.

Every place that used to assume an array-typed variable's own slot
holds its data directly needed to branch on is_heap_allocated instead:
gen_array_address_into's Variable case (load the stored pointer vs.
compute the slot's own address), gen_array_value_into's Variable
(source) case (load the source's pointer before copying, if it's heap-
backed), gen_var_decl (malloc a fresh block, once, at declaration time
-- reused by every later assignment, never reallocated, since a fixed-
size array's footprint never changes across its lifetime), gen_assign
(load the existing pointer and write through it, no malloc), and
gen_function's own parameter loop (a heap-allocated parameter needs its
own independent copy via malloc, exactly like the stack-allocated case
already gets via gen_array_copy, to preserve value semantics across the
call). Everything else -- gen_index_address_into, gen_index_assign, an
Index read in gen_expr_into, gen_array_arg_address_into -- needed no
direct changes at all, since each already delegates through one of the
methods above rather than assuming a slot's shape itself.

Array RETURN values need no changes whatsoever: an array-typed return
already writes directly through the caller-provided hidden pointer
(see gen_return), never allocating any storage of its own regardless
of the array's size -- that path was already safe before this feature
existed.

A REAL HAZARD FOUND WHILE BUILDING THIS: PARAMETER STASHING
-----------------------------------------------------------------
A heap-allocated parameter's malloc call is a real function call, and
like any call, it can clobber every caller-saved register -- including
OTHER, not-yet-processed parameters' own incoming values still sitting
in their argument registers. Naively processing parameters directly
out of their registers, one at a time, breaks the moment any parameter
needs malloc: an earlier one's malloc call can destroy a later one's
still-unread value.

The first fix attempted -- protecting each parameter's register with
an ordinary push, popping it back immediately before that parameter is
processed -- turned out to be wrong in a subtler way: popping one
value at a time leaves a DIFFERENT number of not-yet-popped values on
the stack ahead of each parameter's own malloc call, misaligning %rsp
(a SysV ABI violation) for roughly half of them, depending on the
parameter's position. The actual fix: every incoming argument register
is stashed into its own dedicated, permanently-reserved temporary slot
via a plain %rbp-relative store, in one pass, before any parameter is
processed at all. Plain stores never touch %rsp, so there's no
alignment question to get right -- the two-pass structure (stash
everything, then process everything) exists specifically to sidestep
this class of bug rather than to work around it case by case.

SCOPE: WHAT THIS STILL DOESN'T COVER
------------------------------------------
An ArrayLiteral or a Call returning an array used DIRECTLY as a
function-call argument (`foo([1,2,3])` or `foo(bar())`) isn't
supported -- gen_array_arg_address_into raises a clear error pointing
at the workaround: assign it to a named variable first
(`[3]int t = [1,2,3]; foo(t)`), which already works today. Whole-array
equality (`==`/`!=`) is rejected at the semantic level for a related
reason -- a real, well-defined feature to consider later, just not
implemented yet (see semantic.py's check_binary).

SLICES
-------
A view-only, Go-style slice: a fixed 24-byte {pointer, length,
capacity} descriptor (type_byte_width returns 24 for TypeKind.SLICE
regardless of element type -- two slices of different element types
are still both 24 bytes, unlike two arrays), NOT a copy of whatever
it's a slice of. `base[low:high]` -- both bounds optional,
independently -- reads either an existing array's or an existing
slice's own backing storage directly. This is what makes a slice write
visible through the array (or other slice) it came from: unlike `arr2
= arr1`, which copies elements, a slice's whole point is to alias, not
copy.

cap exists specifically to support `append` -- knowing how much spare
room a slice's own backing array still has (beyond its current len) is
what lets append sometimes write into that EXISTING array instead of
always allocating a fresh one, the mechanism that makes appending in a
loop amortized O(n) rather than O(n^2). At a slice LITERAL or `none`,
cap is set equal to len -- a freshly-created backing array is sized to
exactly fit its own elements, with no spare room to grow into yet.
Re-slicing is different: cap = base_cap - low, inheriting the base's
own remaining capacity from the new starting point (Go's actual
re-slicing rule -- see gen_slice_into's own docstring), which is
exactly what lets a re-sliced view grow into room a PRIOR append, or
the base's own construction, already reserved -- see the APPEND
BUILTIN section below for the full story this makes possible. This
descriptor used to be 16 bytes, {pointer, length} only, before cap was
added -- see the SLICE PARAMETERS AND RETURNS section below for the
real, cascading consequence that had on how a slice crosses a function
boundary.

SAFETY: WHY A SLICED ARRAY MIGHT NEED TO OUTLIVE ITS OWN FRAME
-------------------------------------------------------------------
Slicing something that lives on the stack creates a real dangling-
pointer risk the moment the function that declared it returns -- a
problem plain arrays never had, since every array-typed value is
always either copied (assignment, parameter passing) or written
through a caller-provided pointer before the frame that held it tears
down (return values). A slice is different in kind: it CAN be designed
to outlive the exact call that created it, aliasing storage that has
to still be there afterward -- but only if it actually escapes; a
slice used purely within the function that made it never needs its
backing array to survive past that function's own return at all.

The original plan here was to sidestep deciding that case by case:
heap-allocate ANY array that's ever sliced, unconditionally, reusing
the exact machinery size-based stack safety already had, rather than
building a real escape-analysis pass. That plan was documented in this
exact section, repeatedly, as though it had actually been built --
it never was. is_heap_allocated only ever checked size; there was no
second trigger anywhere in the code. The result was a real, live
memory-safety bug: a small array, sliced, with the slice returned,
kept its stack-allocated inline slot regardless of any of that,
leaving the returned slice's own pointer field dangling into a torn-
down stack frame the moment anything else got called before reading
it again. Found by compiling and running an actual program that did
exactly this -- see analyze_array_escapes's own docstring, and its
module-level test note in test_compiler.py, for the full account.

What actually exists now is a real escape analysis (analyze_array_
escapes, run once per function in gen_function, before _collect_
params/_collect_locals need its answer) -- intraprocedural and flow-
insensitive, not the simpler "any sliced array escapes" rule this
section used to (inaccurately) describe: it tracks, for every slice-
typed variable, which array-typed declaration(s) it might be backed by
(through direct slicing, re-slicing, plain slice-to-slice copies, and
`append`), and only promotes an array if something backed by it
actually flows into a `return` or gets passed to a user-defined
function call (treated as escaping unconditionally, without tracing
into the callee -- see analyze_array_escapes's own docstring for why
that, and its other explicitly-scoped limitation -- array-of-slices
elements aren't tracked at all -- are each real, deliberate boundaries
rather than gaps found by accident). CodeGenerator._is_array_heap_
allocated is where this result combines with is_heap_allocated's own,
independent size check -- either reason alone is sufficient -- and is
what every caller that used to call is_heap_allocated directly for a
SPECIFIC, named variable now goes through instead.

This needed no new addressing logic of its own: gen_array_address_into
and gen_indexable_base_into below already handled a heap- vs. stack-
allocated array identically (the whole point of routing every such
decision through one check in the first place), so an array found to
escape simply flows through the exact same paths every size-promoted
array already did -- only the DECISION of which arrays qualify grew
more precise, not the mechanism that acts on it.

ADDRESS AND LENGTH: gen_indexable_base_into
-----------------------------------------------
Indexing into a slice (`s[i]`) and slicing one (`s[low:high]`) both
need exactly the same information about whatever's on the left of the
`[...]`: an address to compute from, and a length to bounds-check
against. gen_indexable_base_into returns both uniformly, regardless of
which kind of base it's given -- an Imm (compile-time constant) for an
array base, or a register loaded from the slice's own descriptor (a
genuine runtime value) for a slice base -- so gen_index_address_into's
own bounds-check comparison, and gen_slice_into's three-way one,
each just work with whichever Operand comes back, uniformly, with no
separate code path needed per base kind.

The base can now be a Variable (a named slice, loaded directly out of
its own slot), a Slice expression itself (`arr[:][0]`, or
`matrix[:][0][0]` -- materialized into a dedicated scratch slot first;
see the INDEXING INTO UNNAMED SLICES section below for the full
design), a Call to a slice-returning function (materialized into that
exact same scratch slot -- it used to arrive already sitting in %rax:
%rdx, needing no materialization at all, back when a slice's own
descriptor still fit two registers; see the SLICE PARAMETERS AND
RETURNS section below for why that's no longer true), or an Index
yielding a slice (`rows[0][1]`, indexing into an array OF slices,
reusing that same scratch slot too -- this used to be unreachable,
before array-of-slices construction existed at all).

BOUNDS CHECKING: A DIFFERENT BOUNDARY THAN INDEXING'S OWN
-----------------------------------------------------------------
Slice bounds needed a genuinely different comparison from ordinary
indexing's, not just a reused check with a different message: an
ordinary index equal to the array's own size is already invalid
(`Jae`, jump if >=), but a slice's low and high are BOTH allowed to
equal the base's own length (`arr[5:5]` on a 5-element array is a
valid, empty-slice-producing expression) -- so this needed a new,
strict comparison (Ja, jump if strictly >), not Jae reused wholesale.
Both still catch a negative bound via the same unsigned-
reinterpretation trick Jae already relies on.

Slice-bounds failures get their own panic message ("slice bounds out
of range", distinct from ordinary indexing's "array index out of
bounds") -- which is what actually forced the bounds-check panic
infrastructure to generalize from a single hardcoded message to a
dict keyed by message text, each with its own per-function fail label
and program-wide-cached message label, rather than everything sharing
the one panic block indexing already had.

gen_slice_into's OWN REGISTER DISCIPLINE
--------------------------------------------
Every intermediate value (the base's address, its length, high, low)
is protected on the real CPU stack across evaluating whichever of
low/high are actually present -- each could be an arbitrarily complex
expression, including a function call -- rather than assumed to
survive in whatever register initially held it. Pushed in a specific
order (address, then length if it's a runtime value, then high, then
low) and popped in exact reverse, so nothing ever needs to be read out
of the middle of the stack, with one deliberate exception: defaulting
`high` to the base's own RUNTIME length reads it via a plain peek at
the top of the stack (`(%rsp)`, no pop) at the one point where doing so
is safe -- nothing has been pushed since the length was, and peeking
rather than popping keeps it protected for the bounds check that
comes later, without needing a separate temporary slot just to hold a
value that's already sitting exactly where it needs to be.

PRINTING ARRAYS AND SLICES
-------------------------------
`print` on an array or slice formats as `TYPE[elem, elem, ...]` --
e.g. `[3]int[1, 2, 3]` or `[]int[1, 2, 3]` -- the type prefix
(str(arg_type), matching semantic.Type.__str__ exactly, so no new
formatting logic was needed for it) appearing exactly once, at the
outermost level, never repeated for a nested row (a [2][3]int prints
as `[2][3]int[[1, 2, 3], [4, 5, 6]]`, not with "[3]int" repeated on
each inner row). A str element is quoted inside a collection
(`'alice'`) even though a bare str argument to print still prints
unquoted -- two different, both intentional, conventions, matching
how most languages format a string differently in a collection/repr
context than when printed bare.

Built as a sequence of direct printf calls -- one piece at a time
(the type prefix, each bracket, each separator, each element) -- via
_gen_print_static/_gen_print_quoted_str/_gen_print_int_value/
_gen_print_bool_value, rather than materializing one big string via
malloc and printing it in one shot. That alternative would need a new
int-to-string conversion step this language has no other reason to
have: every existing int print already goes straight to printf's own
%d formatting, never through an intermediate string buffer -- adding
one just for this would be real, separable work (buffer sizing, how
it interacts with str's existing memory-leak policy) for a feature
that doesn't otherwise need it. The trade-off is more instructions per
print call on a collection than a scalar -- a reasonable one for a
teaching compiler, not an accident.

gen_print_collection: ONE LOOP, NOT UNROLLED-VS-LOOPED
------------------------------------------------------------
An array's length is known at compile time; a slice's is only known
at runtime. Rather than unroll an array's printing at compile time
(fewer instructions, but a second code path to maintain) and loop only
for a slice, _gen_print_collection uses ONE uniform runtime loop for
both, reusing gen_indexable_base_into's own "address plus length,
either an Imm or a runtime register" abstraction directly -- the
comparison that ends the loop just works with whichever Operand comes
back, uniformly, exactly like gen_index_address_into's own bounds
check already does.

%rbx (the base address), %r12 (the length, when it's a runtime value),
and %r13 (the loop counter) are all CALLEE-SAVED, not the caller-saved
scratch (rax, rcx, rdx, ...) most of this file's transient
computations already use -- because all three have to survive across
every printf/puts call the loop body makes, at least one per element,
and a well-behaved libc call is obligated to preserve a callee-saved
register the same way another Hornet function already has to. A
nested array element (this method's own recursive case, for a multi-
dimensional array's rows) protects all three on the stack across the
RECURSIVE call specifically, since that call reuses these same three
physical registers for its own, independent address/length/counter --
the same push-before-recursing discipline used everywhere else in this
file a value needs to survive evaluating something else, just applied
to a whole recursive call instead of a single sub-expression.

print's own argument, when array- or slice-typed, is restricted to a
Variable or Index -- the same restriction gen_array_arg_address_into
already imposes on array-typed call arguments, for the same reason: a
bare ArrayLiteral, Slice, or array/slice-returning Call has no address
of its own to print through. Assign it to a named variable first.

LEN BUILTIN
----------------
`len(x)`, Hornet's second builtin -- gen_len_call_into. Unlike print's
own restriction just above, len's argument is NOT restricted to a
Variable or Index: it reuses gen_indexable_base_into directly, so
whatever that method currently accepts as a base (a Variable, an
Index, a Slice expression, a slice-returning Call, or an ArrayLiteral)
is automatically valid for len too, with nothing to keep in sync if
that set ever grows. Deliberately not narrowed to match print's own,
older restriction, which predates most of those cases existing at all.

For an ARRAY argument, the returned length is a compile-time Imm --
the array's own declared size -- never actually read out of the
argument's runtime value. For a SLICE argument, it's a genuine runtime
read out of the descriptor's own len field, narrowed through its own
32-bit register alias the same way every other reader of a slice's
length already does (Hornet's int is always 32 bits, even though the
descriptor's own len field occupies a full 8-byte slot).

The argument is still FULLY evaluated in either case, including
whatever real work is buried inside it (a bounds check, a heap
allocation, a side-effecting function call), regardless of whether the
resulting length ends up depending on that work's outcome at all --
gen_indexable_base_into's own address computation for the ARRAY case
is simply discarded afterward, unused, rather than being skipped as
an optimization. This is a deliberate consistency choice, not an
oversight: `len(arr[i])` still aborts on an out-of-range i even though
a sub-array's own length never actually depends on i's value, and
`len([]int[1, 2, 3])` still performs a real, if wasted, heap
allocation for a length that was already fully known from the
literal's own shape before a single instruction ran. The alternative
-- skip evaluation whenever the length happens to be compile-time-
derivable -- would make len's argument evaluation behave differently
depending on what shape the argument happens to take, which is a
worse inconsistency than one rare, low-value wasted allocation.

APPEND BUILTIN
-------------------
`append(s, value)`, Hornet's third builtin -- gen_append_call_into --
Go-style: returns a NEW {ptr, len, cap} descriptor rather than
mutating s in place. s itself is never touched; the result is written
to wherever the call's own value flows (a VarDecl, an Assign, ...) via
gen_slice_value_into's own dispatch, exactly like any other slice-
producing expression.

s can be ANY slice-typed expression, not just a bare Variable or
`none` -- a re-slice, an Index, a whole slice literal, another
append call, a slice-returning function call, ... A bare Variable or
`none` is handled inline (no extra work needed: `none`'s own zero
descriptor is just three immediate zeros, and a Variable's own
{ptr, len, cap} already lives in a known stack slot); anything else
is first materialized into the same shared, per-function unnamed-
slice scratch slot gen_indexable_base_into's own Slice-base case
already uses, via gen_slice_value_into, then read back out exactly
like a Variable's own slot would be. This used to be restricted to
just a Variable or `none`, on the theory that append exists
specifically to feed a reassignment (`x = append(x, v)`) and the
extra materialization step wasn't worth it for what looked like a
rare shape -- lifted once `append([]int[], 1)`, building a slice from
scratch in a single expression, turned out to be exactly the shape
someone actually reached for.

REUSE VS. REALLOCATE: THE GROWTH POLICY
-----------------------------------------------
s's own ptr/len/cap are loaded into CALLEE-SAVED registers (%rbx/%r12/
%r13) up front, not caller-saved ones -- specifically because the
reallocating path below calls malloc, which (like any real, ABI-
conforming function) is free to clobber any caller-saved register but
is OBLIGATED to preserve callee-saved ones, the exact same guarantee
gen_array_literal_heap_alloc_into and gen_function's own heap-
allocated-parameter handling already rely on.

The decision itself is a single comparison: len >= cap means no spare
room -- the only way that can happen, given the invariant len <= cap
always holds, is len == cap exactly -- so `jae` is both correct and
sufficient, no separate "is it exactly equal" check needed.

REUSE (len < cap): the new element is written directly into the
EXISTING backing array, at ptr + len*element_width -- s's own array,
fully intact and unaffected, since s's own len field is never
touched. This is the observable aliasing the whole growth policy
exists to make possible: a slice produced by re-slicing with spare
capacity (see the SLICES section's own note on cap = base_cap - low),
or by a PRIOR append that over-allocated, can have a later append
write into storage some OTHER slice still watches -- not a bug this
design works around, but the mechanism that makes appending in a loop
amortized O(n) rather than O(n^2) in the first place.

REALLOCATE (len == cap): new_cap is computed from cap alone (see the
growth-policy arithmetic below), a fresh block of new_cap*element_
width bytes is malloc'd, the existing len elements are copied into it
via a genuine RUNTIME loop -- len is a runtime value here, unlike
every other array copy in this file (gen_array_copy's own flat-copy
loop), which always moves a compile-time-known total width -- and only
then is the new element written into the new array. Each loop
iteration reuses gen_array_copy anyway, for exactly ONE element's
worth of data: that method's own logic (copy type_byte_width(T) bytes,
dispatching on leaf_type(T) for the per-chunk width) already
generalizes correctly to a single, arbitrary-type value, not just a
whole array, so no separate "copy one value" helper was needed just
for this loop body. The OLD backing array is simply never freed,
matching this compiler's existing no-`free`-anywhere memory model
everywhere else.

GROWTH POLICY: new_cap = cap*2 if cap < 256, else cap + cap//4, with a
cap==0 floor of 1. This is the general max(needed, doubled-or-
quartered) formula simplified, not a different rule: reallocation only
ever happens when len == cap exactly, so `needed` (len+1) is always
cap+1, and doubling already exceeds cap+1 for any cap >= 1 (quartering
trivially does too, for cap >= 256) -- the max only actually matters,
and only ever resolves in needed's favor, at cap == 0, which is
exactly the explicit floor case here. cap/4 is computed via a right
shift by 2 (arithmetic, though cap's own non-negativity means a
logical shift would give the identical result) rather than idiv --
idiv can't take an immediate divisor at all on x86 (see IDiv's own
docstring), and a shift is simpler besides.

WRITING THE NEW ELEMENT: _gen_write_value_at_address_into
-------------------------------------------------------------------
Both the reuse and reallocate paths need to write the newly-appended
element at a COMPUTED address (not a fixed offset), with the
element's own type possibly being scalar, array, or slice -- shared
between them via one helper rather than duplicated. For an array or
slice element type, this just hands the computed address straight to
gen_array_value_into/gen_slice_value_into as an ordinary Memory
destination -- both already protect an arbitrary base internally (see
their own docstrings). For a scalar, the address is protected
manually, matching gen_array_literal_into's own scalar-element pattern
exactly: push it, compute the value (which could itself involve a
function call that clobbers the address register, if the value
expression is arbitrarily complex), stash the computed value in %r8/
%r8d, pop the address back, then write from %r8/%r8d -- never straight
from %eax/%rax, which popping the address back into would otherwise
have to clobber.

NONE: THE SLICE ZERO VALUE
-------------------------------
`none` (see NoneLiteral's own docstring in parser.py, and semantic.py's
own NONE section) becomes a {ptr: 0, len: 0, cap: 0} slice descriptor
at the machine level -- the same shape Go's own nil slice has. Every
existing slice operation (indexing, printing, re-slicing) already
handles a zero-length slice correctly -- see TestSliceBoundsChecking's
own `arr[5:5]` positive control in test_compiler.py -- so a
none-valued slice needed no new mechanism for any of those; only two
genuinely new pieces were needed: producing the {0, 0, 0} descriptor
in the first place (gen_none_into), and comparing a slice against
`none` directly (gen_slice_none_comparison_into).

gen_none_into is called directly from gen_var_decl/gen_assign's own
NoneLiteral short-circuit, rather than folded into
gen_slice_value_into's existing dispatch -- unlike every OTHER kind of
slice-producing expression there (a Slice expression, a Variable
holding one), a NoneLiteral's own resolved type (Type.NONE) never
equals the slice type it's being stored into, so the caller has to
already know and pass the TARGET type explicitly. gen_slice_value_into
itself never needed a target-type parameter before this, since every
other case's own resolved type already matched what needed to be
stored -- this is the one place that invariant doesn't hold, so it's
handled one level up instead of restructuring that method's signature
for every existing caller.

gen_slice_none_comparison_into checks specifically the slice
descriptor's own `ptr` field against 0 -- not its length -- matching
Go's own well-known nil-vs-empty-slice distinction: a real, zero-length
slice sliced from a real array (e.g. `arr[5:5]`) has a non-null
pointer and is NOT `== none`, even though it's equally safe and
equally zero-length as a genuinely nil slice for every other purpose.
This needed a new CmpQ instruction (64-bit compare) alongside the
existing, 32-bit-only Cmp: every OTHER comparison in this language
compares int/bool values, for which cmpl is exactly right, but a
pointer is a full 64-bit value, and comparing only its low 32 bits
against zero could, in principle, miss a real, non-null pointer whose
low 32 bits happen to be zero. gen_binary_into dispatches EQUAL/
NOT_EQUAL here whenever either operand is slice-typed, before ever
reaching the ordinary single-register stack-spill scheme below it --
semantic.py's check_binary already guarantees, by the time this is
reached, that exactly one side is slice-typed and the other is
none-typed (a real slice compared to another real slice, or none
compared to none, are both rejected earlier), so this doesn't need to
re-derive or defensively check which side is which beyond that.

`none` used as a function argument or return value for a slice-typed
parameter/return works correctly now -- see the SLICE PARAMETERS AND
RETURNS section just below for the calling convention itself, and
gen_slice_arg_into/gen_return's own NoneLiteral case for how `none`
specifically flows through it (a {0, 0, 0} triple pushed like any
other slice argument, or written through the hidden return pointer,
exactly like any other slice-typed return value).

SLICE PARAMETERS AND RETURNS
-----------------------------------
A slice's own descriptor is {ptr, len, cap} -- three 8-byte fields, 24
bytes total (see the SLICES section for why cap was added). As a
PARAMETER, this crosses a function boundary via THREE consecutive
integer argument registers directly -- matching exactly what a real C
compiler does for an equivalent `struct{void*,long,long}` passed by
value under the SysV ABI -- and is never copied on entry the way an
array parameter is: a slice parameter is just an alias, exactly like
any other slice variable. This is safe by construction, not by luck:
analyze_array_escapes treats passing a slice as an argument to any
user-defined function call as escaping (see its own docstring) --
exactly this situation -- so whatever array backs it in the CALLER is
already guaranteed to be heap-allocated by the time the call happens,
regardless of what this function goes on to do with its own copy of
that alias.

As a RETURN VALUE, a slice now uses the exact same hidden-output-
pointer convention arrays already established (see the ARRAYS section
above) -- gen_return's own Slice case is structurally identical to its
Array one, and gen_slice_call_into mirrors gen_array_call_into exactly.
This used to be different: a slice's descriptor was small enough (16
bytes, before cap existed) to return directly in %rax:%rdx, the SysV
ABI's own convention for a small, all-integer struct return -- no
hidden pointer needed at all, genuinely simpler than the array case
rather than an extension of it. Adding cap grew the descriptor past
what any two- or three-register return shape this compiler has
precedent for could hold, so rather than invent a new one, slice
returns converged onto the mechanism arrays already had. One real
consequence: gen_return's own NoneLiteral case (`return none`) also
moved to writing through the hidden pointer, instead of zeroing %rax/
%rdx directly.

REGISTER-SLOT ACCOUNTING: NO LONGER 1:1
-----------------------------------------------
Since a slice now costs 3 of the 6 available argument-register slots
instead of 1, the mapping from argument/parameter INDEX to register
INDEX stopped being the simple 1:1 one every OTHER type still uses.
Both sides track a running slot count instead:
  - CALLER side: _gen_call_arguments_into, shared by gen_call_into,
    gen_array_call_into, and gen_slice_call_into, pushes each
    argument's value(s) in left-to-right order (a slice contributing
    its ptr, then its len, then its cap, as three separate pushes) and
    pops everything back off in exact reverse into the correct
    register -- the same push-then-pop-in-reverse discipline this file
    already used for ordinary scalar arguments, just with a variable
    number of slots per argument instead of always one.
  - CALLEE side: gen_function's own parameter-stashing loop advances
    its register-index counter by 3 for a slice parameter (stashing
    THREE consecutive incoming registers into that parameter's own
    24-byte temp slot) instead of 1.
_total_arg_slots (caller) and the equivalent inline sum in gen_
function (callee) both compute "how many slots will this collection
of arguments/parameters actually need" for their own "too many
arguments/parameters" check -- no longer just len(args) > 6, since a
single slice-typed argument or parameter can by itself use half the
budget.

INDEXING INTO UNNAMED SLICES
-----------------------------------
`arr[:][0]`, or `matrix[:][0][0]`, or `someSliceFn()[0]` -- indexing
(or re-slicing) directly into a Slice expression's or slice-returning
Call's own result, without first assigning it to a named variable.
gen_indexable_base_into used to require a Variable here, since neither
kind of result has a pre-existing address to take (a freshly computed
descriptor, or one now received through a hidden pointer this function
itself doesn't own past the call); this closes that gap by giving
either one somewhere to put its result: a dedicated, per-function
scratch slot (_unnamed_slice_temp_offset, 24 bytes, reserved
unconditionally in gen_function for every function, not just ones that
happen to use it) that a Slice base (via gen_slice_into), a slice-
returning Call base (via gen_slice_call_into, now that it takes a
Memory destination directly), or a slice-typed Index base (via gen_
slice_value_into's own Index case) all get materialized into, then
immediately read back out of into addr_dst/len_dst -- the exact same
"compute into a Memory destination" contract each of those already had
for every other caller, just with a throwaway destination instead of a
real variable's own slot.

WHY ONE SHARED SCRATCH SLOT IS SAFE UNDER ARBITRARY NESTING
-------------------------------------------------------------------
Reusing a SINGLE shared slot for every Slice materialization -- rather
than a fresh one per nesting level -- relies on how gen_slice_into and
gen_index_address_into are both already structured: each computes its
OWN base's address/length FIRST, immediately consumes it (into
registers, then protects those on the real CPU stack before evaluating
anything else -- e.g. a slice's own low/high bounds, or an index
expression, either of which could itself trigger another
materialization), and only ever WRITES its own result into a
destination as the very LAST step. That means a deeper level's own
write to the shared slot always happens -- and is always fully drained
back into registers -- strictly BEFORE the shallower level that
triggered it writes ITS OWN result there. This is the same strictly-
nested lifetime discipline that makes reusing one call stack safe for
recursion of any depth, just applied to one scratch memory slot instead
of the stack itself. Verified directly, not just reasoned about: a
binary expression whose two operands each need their own, independent
materialization (`arr1[:][0] + arr2[:][0]`), and a slice expression
whose own low bound itself requires materializing a DIFFERENT unnamed
slice (`arr[idx_holder[:][0]:5]`) both compute correctly -- see
test_compiler.py's TestIndexingUnnamedSlices for both, and its own
module-level note for why these two specifically, not just the two
headline examples, are what actually stress-test this reasoning.

Kept entirely SEPARATE from _slice_return_temp_offset above, rather
than reusing it for both purposes: the same reasoning would, in fact,
make sharing that one safe too, but two small, obviously-correct-by-
construction slots are worth more than the 16 bytes saved by relying
on a shared one being safe across two conceptually distinct purposes.

Deliberately still out of scope: an unnamed slice used directly as a
function-call argument (`foo(arr[:])`) or as print's own argument
(`print(arr[:])`) -- both keep their existing, separate "must be a
named Variable" restrictions (gen_slice_arg_into, gen_print_call_into),
unrelated to and unaffected by this fix, which is specifically about
gen_indexable_base_into's own base-of-`[...]` restriction.

TYPED ARRAY LITERALS
-------------------------
`[3]int[1, 2, 3]` -- a fully-typed array literal, self-describing
enough to work as a genuine, general expression rather than only ever
appearing as a VarDecl's own initializer the way the plain `[1, 2, 3]`
form still does (see ArrayLiteral's own docstring in parser.py for
why that restriction was never really about semantics at all: even
the untyped form already infers its own type entirely from its
elements, with check_array_literal needing no externally-supplied
"expected type" to do it -- it's purely that codegen never had
anywhere else to WRITE the resulting value).

This needed essentially NO changes here, which is itself worth stating
plainly: gen_array_literal_into and gen_array_value_into already work
purely off an externally-supplied array_type parameter, never reading
ArrayLiteral.type_expr at all -- the typed and untyped forms produce
byte-for-byte identical instructions once semantic.py has resolved
either one down to a real Type. The one genuinely new piece needed was
for a BARE literal statement specifically (`[3]int[1, 2, 3]` alone,
with no assignment) -- see gen_expr_stmt's own dispatch and gen_array_
literal_side_effects_only: since nothing ever reads such a statement's
value, and an array literal has no natural upper bound the way a
slice's fixed 24-byte descriptor does (so there's no single scratch
slot size that would always be enough to reserve one for), this just
evaluates each of the literal's own, directly-written elements for
whatever side effects they might have, discarding every result,
without ever materializing a real array in memory at all. An element
that's itself some OTHER array-typed expression (a bare Variable, an
indexed sub-array, an array-returning Call) inside a bare-statement
literal is a deliberate, explicit gap: correctly distinguishing "no
side effect worth preserving" (a Variable) from "might have one" (a
Call) -- or materializing either one just to immediately discard it --
isn't implemented; this raises a clear error instead of guessing.

SLICE LITERALS
-------------------
`[]int[1, 2, 3]` -- automatically creates a heap-allocated backing
array (sized to the literal's own element count) AND the {ptr, len,
cap} descriptor pointing at it, in one expression. Also `[]int s =
[1, 2, 3]`: an untyped bracket list flowing directly into a
slice-typed VarDecl/Assign is treated the same way, just with the
element type inferred from the DECLARED slice type instead of
restated in the literal (see semantic.py's own _check_value_flowing_
into).

Implemented as parser sugar, not a new AST node or a new top-level
codegen entry point: `[]int[1, 2, 3]` parses directly into
`Slice(array=ArrayLiteral(...), low=None, high=None)` -- an implicit
"the whole thing" slice of a freshly-parsed array literal (see
ArrayLiteral's own docstring in parser.py) -- which is what lets
check_slice, gen_slice_into, and gen_indexable_base_into handle the
general, TYPED form almost entirely via machinery that already existed
for slicing a NAMED array. The one genuinely new piece needed is
gen_indexable_base_into's own ArrayLiteral case (inside its existing
ARRAY branch, not the SLICE one -- an ArrayLiteral's own type is
ARRAY-kind; the slice-ness comes entirely from the outer Slice
wrapping, not anything intrinsic to the literal node itself): where
the ordinary Variable/Index cases compute the address of something
that ALREADY exists, this one calls gen_array_literal_heap_alloc_into
to create something new -- a fresh, heap-backed allocation, written
with the literal's own elements, whose resulting pointer becomes the
"address" gen_slice_into's own low/high-defaulting logic then slices
in the ordinary way (trivially, the whole thing, since low/high are
both None).

The UNTYPED form (`[]int s = [1, 2, 3]`) can't be handled by parser
sugar at all -- the parser has no way to know, at parse time, that a
plain `[1, 2, 3]` is meant for a slice rather than an array, since
that depends entirely on the SURROUNDING declared/target type. This is
instead a small, explicit ArrayLiteral-as-slice-value short-circuit
directly in gen_var_decl and gen_assign, reusing the exact same
gen_array_literal_heap_alloc_into helper the general, typed form's
gen_indexable_base_into case does -- both ultimately need identical
work (malloc a backing array, write the literal's elements into it,
record the resulting pointer and length), just reached from two
different call sites for two different syntactic shapes.

gen_array_literal_heap_alloc_into itself always allocates AT LEAST 1
BYTE, even for a completely empty literal (`[]int[]`) -- deliberately
not relying on libc's own malloc(0) behavior, which POSIX leaves
implementation-defined (either a null return or a valid, unique
pointer are both conforming). This is what makes `s == none` correctly
FALSE for an intentionally empty slice literal: `[]int[]` is a real,
live, zero-length slice with a genuine (if trivial) backing
allocation, not a nil one -- the exact same nil-vs-empty distinction
`arr[5:5]` already has (see gen_slice_none_comparison_into), extended
here to a literal that was never sliced from anything at all. Every
slice literal's own backing array is heap-allocated UNCONDITIONALLY
here, regardless of size, unlike an ordinary array variable (which
only heap-promotes past the existing 16KB stack-size threshold, see
is_heap_allocated) -- not a size-based decision at all: a slice
literal's backing array has to outlive the statement that creates it,
the same "can safely cross frame boundaries" guarantee every OTHER
sliced array already gets, unconditionally, for the same reason.

A bare Slice-expression statement (`[]int[se(), 2, 3]` alone, with no
assignment -- desugaring, per the above, into a bare Slice statement
just like any other) needed its own new codegen too: gen_expr_stmt
previously had no Slice case at all, so this fell through to the
ordinary gen_expr_into dispatch and hit its ArrayLiteral/Slice
defensive rejection (neither fits in a single register). Unlike the
analogous ArrayLiteral case just above, this doesn't need its own
narrower, side-effects-only path: gen_slice_into already computes
fully correctly into any Memory destination, including a genuine
runtime bounds check on low/high, so gen_expr_stmt's own new Slice
case just reuses the same per-function scratch slot gen_indexable_
base_into's own Slice-base case already needed (_unnamed_slice_temp_
offset) and discards the result -- an out-of-range bound still aborts
here, matching how any other bare expression statement's real
instructions genuinely run. This same fix, found specifically by
testing the slice-literal case, also closed a genuinely pre-existing,
unrelated gap: a bare statement slicing an ordinary, already-existing
array (`arr[:]` alone) was already broken before slice literals
existed at all.

NESTED SLICES
------------------
A slice (or array) whose own ELEMENT type is itself a slice --
`[][]int`, `[2][]int`, arbitrarily deep. This used to be the one thing
explicitly scoped out as separable follow-up work -- gen_array_copy
and gen_array_literal_into both used to raise a clear error rather
than silently truncate a slice's own descriptor down to a 4-or-8-byte
scalar move, which is all their existing flat-copy logic knew how to
do.

Closing that gap turned out to need one thing beyond just teaching
those two methods a new leaf width: gen_slice_value_into (the shared
"produce a slice-typed value into a Memory destination" dispatch --
Slice, Variable, Call, Index, ArrayLiteral) had to be made safe to call
with an ARBITRARY destination, not just an ordinary local slot, since
writing a slice-typed ARRAY ELEMENT means dst_mem.base is the outer
array's own base -- 'rbp' if it's stack-allocated, or 'rax' if it's
heap-allocated (every slice literal's own backing array always is).
That in turn meant gen_slice_into itself needed the same treatment --
it used to explicitly assert dst_mem.base == 'rbp' and refuse anything
else, since nothing before this ever needed to write a slice value
anywhere but an ordinary local. See gen_slice_value_into's own
docstring for the full case-by-case account of what each of its five
cases needed (a genuinely new register-clobbering risk in three of
them, once dst_mem.base could be something other than 'rbp' -- the
same class of bug this file has hit before in this exact area, not a
hypothetical one).

Two closely-related pieces fell out of that same generalization,
neither separately requested but both a small, natural extension of
it: gen_indexable_base_into gained an Index-as-base case (`rows[0][1]`,
chaining directly into a slice-typed indexing result with no
intermediate named variable -- reusing gen_slice_value_into's own,
new Index case through the same shared scratch slot the existing
Slice-as-base case already used), and gen_index_assign gained a
SLICE-element case (`rows[i] = someSlice`), deriving element_type from
the INDEXED ARRAY's own type rather than the VALUE's -- the value's
own resolved type is wrong for exactly the same reason it was wrong in
gen_var_decl/gen_assign before those were fixed (see gen_index_assign's
own docstring): an untyped literal flowing into a slice-typed element
has its own resolved type set to the ARRAY it builds, not the slice
it's being treated as.

Copying an array of slices (gen_array_copy's own 24-byte leaf
case) is a SHALLOW copy of each element's own {ptr, len, cap} descriptor --
matching how copying a bare slice variable (`s2 = s1`) already works,
not a deep, recursive re-allocation. A real, deliberate consistency
choice: the copy's own slice elements end up pointing at the exact
same backing data the original's do.

On the semantic.py side, the underlying bug that made all of this
worth discovering in the first place wasn't a scope decision at all:
check_array_literal's own element-checking loop called check_expr
directly on each element rather than routing through _check_value_
flowing_into, so an untyped INNER literal never got the "this
constructs a nested slice" treatment -- only a literal's own, TOP-
level value ever did. `[][2]int` (an array-typed inner element)
happened to keep working by coincidence, since plain type equality was
all THAT case ever needed -- which is exactly what masked the gap
until a genuinely nested SLICE was tried. The identical bug-class
turned out to exist in one more place doing the same "value flows into
an already-typed slot" check: analyze_index_assign.
"""

import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

from lexer import lex
from parser import (
    ArrayLiteral,
    Assign,
    Binary,
    BinaryOp,
    BoolLiteral,
    Break,
    Call,
    Constant,
    Continue,
    ExprStmt,
    Function,
    If,
    Index,
    IndexAssign,
    Node,
    NoneLiteral,
    Param,
    Parser,
    Program,
    Return,
    Slice,
    StringLiteral,
    Unary,
    UnaryOp,
    VarDecl,
    Variable,
    While,
)
from semantic import analyze, type_from_name, Type, TypeKind


# ---------------------------------------------------------------------------
# Assembly AST
# ---------------------------------------------------------------------------

class Operand:
    def emit(self) -> str:
        raise NotImplementedError


@dataclass
class Imm(Operand):
    value: int

    def emit(self) -> str:
        return f"${self.value}"


@dataclass
class Register(Operand):
    name: str  # e.g. 'eax'

    def emit(self) -> str:
        return f"%{self.name}"


@dataclass
class Memory(Operand):
    """A memory operand: `offset(%base)`, e.g. `-4(%rbp)`. This is how
    every local variable is stored -- see the module docstring's LOCAL
    VARIABLES section -- with `base` almost always 'rbp'. It's also
    reused, with a DIFFERENT base, for reading/writing through a
    computed address held in some other register (e.g. Memory('rbx',
    0) for the address an array index computed) -- see the ARRAYS
    section for why array copying/addressing needed this generality
    that scalar locals never did."""
    base: str    # e.g. 'rbp', or another register holding a computed address
    offset: int  # bytes from `base`; locals live at negative offsets

    def emit(self) -> str:
        return f"{self.offset}(%{self.base})"


class Instruction:
    """Base class for assembly instructions.

    Subclasses set `mnemonic` and implement `operands()`; `emit()` is
    generic and handles column alignment (via `mnemonic.ljust(8)`) so
    every instruction lines up the same way regardless of how long its
    mnemonic is -- compare `movl` (4 chars) and `movzbl` (6 chars) in the
    examples below, both of which align their first operand to column 8.
    """

    mnemonic: str = ""

    def operands(self) -> List[str]:
        return []

    def emit(self) -> str:
        ops = self.operands()
        if not ops:
            return self.mnemonic
        return f"{self.mnemonic:<8}{', '.join(ops)}"


@dataclass
class Mov(Instruction):
    src: Operand
    dst: Operand
    mnemonic = "movl"

    def operands(self) -> List[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class Neg(Instruction):
    """Two's-complement arithmetic negation, in place: dst = -dst."""
    operand: Operand
    mnemonic = "negl"

    def operands(self) -> List[str]:
        return [self.operand.emit()]


@dataclass
class Not(Instruction):
    """Bitwise complement, in place: dst = ~dst."""
    operand: Operand
    mnemonic = "notl"

    def operands(self) -> List[str]:
        return [self.operand.emit()]


@dataclass
class Cmp(Instruction):
    """Compares src and dst by computing dst - src and setting flags
    (notably ZF) accordingly -- doesn't modify either operand."""
    src: Operand
    dst: Operand
    mnemonic = "cmpl"

    def operands(self) -> List[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class CmpQ(Instruction):
    """64-bit compare (`cmpq`) -- the CmpQ counterpart to Cmp (`cmpl`,
    32-bit), for the one case that needs it: checking a slice
    descriptor's own 64-bit `ptr` field against 0 (see
    gen_slice_none_comparison_into). Every OTHER comparison in this
    language compares 32-bit int/bool values, for which Cmp's cmpl is
    exactly right -- but a pointer is a full 64-bit value, and
    comparing only its low 32 bits against zero could, in principle
    (however unlikely for any real address in practice), miss a real,
    non-null pointer whose low 32 bits happen to be zero."""
    src: Operand
    dst: Operand
    mnemonic = "cmpq"

    def operands(self) -> List[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class SetCC(Instruction):
    """Sets an 8-bit operand to 1 if the given condition matches the
    flags from the last Cmp, else 0. `cc` is the x86 condition-code
    suffix -- 'e' (equal), 'ne' (not equal), 'l'/'g' (signed less/greater
    than), 'le'/'ge' (signed less/greater-or-equal) -- and the mnemonic
    is built from it (`sete`, `setne`, `setl`, ...). This is the single
    instruction behind every comparison operator (== != < > <= >=) and
    also behind logical NOT, which is just "was the operand equal to
    0?" (cc='e')."""
    cc: str
    operand: Operand

    @property
    def mnemonic(self) -> str:
        return f"set{self.cc}"

    def operands(self) -> List[str]:
        return [self.operand.emit()]


@dataclass
class MovZX(Instruction):
    """Zero-extends an 8-bit src into a 32-bit dst. Needed after SetE,
    since `sete` only ever writes the low byte (e.g. %al) and leaves the
    rest of the containing 32-bit register (e.g. %eax) untouched -- so
    without this, %eax could still hold garbage in its upper 24 bits."""
    src: Operand
    dst: Operand
    mnemonic = "movzbl"

    def operands(self) -> List[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class Add(Instruction):
    """dst += src."""
    src: Operand
    dst: Operand
    mnemonic = "addl"

    def operands(self) -> List[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class AddQ(Instruction):
    """64-bit dst += src (`addq`). Used only for the length arithmetic
    in gen_string_concat_into (`len(left) + len(right) + 1`) -- string
    lengths come back from `strlen` as a full 64-bit size_t, so this
    needs to be the 64-bit add, not Add's 32-bit `addl`."""
    src: Operand
    dst: Operand
    mnemonic = "addq"

    def operands(self) -> List[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class Sub(Instruction):
    """dst -= src."""
    src: Operand
    dst: Operand
    mnemonic = "subl"

    def operands(self) -> List[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class IMul(Instruction):
    """dst *= src (signed, two-operand form)."""
    src: Operand
    dst: Operand
    mnemonic = "imull"

    def operands(self) -> List[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class Cdq(Instruction):
    """Sign-extends %eax across the %edx:%eax pair. Required immediately
    before IDiv, which always divides that 64-bit pair (not just %eax)
    by its operand -- without this, %edx could hold garbage and corrupt
    the division."""
    mnemonic = "cdq"


@dataclass
class IDiv(Instruction):
    """Divides the 64-bit %edx:%eax pair by `operand` (signed). Quotient
    ends up in %eax, remainder in %edx. `operand` must be a register or
    memory location -- x86 doesn't support an immediate divisor for
    idiv, which is why gen_binary_into always routes the right-hand side
    through the %ecx scratch register rather than leaving it as an Imm.

    This is also what MODULO reuses -- see gen_binary_op's MODULO case
    -- since idiv computes the quotient *and* remainder in one
    instruction; modulo is exactly this same Cdq+IDiv sequence, just
    reading %edx afterward instead of %eax."""
    operand: Operand
    mnemonic = "idivl"

    def operands(self) -> List[str]:
        return [self.operand.emit()]


@dataclass
class And(Instruction):
    """dst &= src (bitwise AND)."""
    src: Operand
    dst: Operand
    mnemonic = "andl"

    def operands(self) -> List[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class Or(Instruction):
    """dst |= src (bitwise OR)."""
    src: Operand
    dst: Operand
    mnemonic = "orl"

    def operands(self) -> List[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class Xor(Instruction):
    """dst ^= src (bitwise XOR)."""
    src: Operand
    dst: Operand
    mnemonic = "xorl"

    def operands(self) -> List[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class ShiftLeft(Instruction):
    """dst <<= %cl. x86 only allows an immediate or specifically %cl as
    a shift instruction's count operand -- never an arbitrary register
    -- so, unlike And/Or/Xor above, this doesn't take a general `src`
    field at all; %cl is hardcoded, since architecturally nothing else
    could ever go there. This lines up for free with how every other
    binary operator already works: gen_binary_into always evaluates the
    right-hand operand into %ecx before calling gen_binary_op, so the
    shift count is already sitting in the one register x86 requires by
    the time this instruction is emitted."""
    dst: Operand
    mnemonic = "shll"

    def operands(self) -> List[str]:
        return ['%cl', self.dst.emit()]


@dataclass
class ShiftRightArithmetic(Instruction):
    """dst >>= %cl, sign-extending (arithmetic) shift -- matches this
    language's `int` being signed, so `-8 >> 1 == -4`, not some large
    positive value from a zero-filling logical shift. See ShiftLeft's
    docstring for why %cl is hardcoded rather than a general `src`."""
    dst: Operand
    mnemonic = "sarl"

    def operands(self) -> List[str]:
        return ['%cl', self.dst.emit()]


@dataclass
class Push(Instruction):
    """Pushes a 64-bit register onto the stack. x86-64 doesn't support a
    32-bit push in long mode, so the caller is responsible for passing
    an already-64-bit register (e.g. Register('rax'), not
    Register('eax')) -- see as_qword_register for converting a 32-bit
    general-purpose register to its 64-bit alias when spilling one."""
    operand: Register
    mnemonic = "pushq"

    def operands(self) -> List[str]:
        return [self.operand.emit()]


@dataclass
class Pop(Instruction):
    """The pop counterpart to Push -- see its docstring."""
    operand: Register
    mnemonic = "popq"

    def operands(self) -> List[str]:
        return [self.operand.emit()]


@dataclass
class LeaQ(Instruction):
    """Loads the *address* of `label` into `dst`, RIP-relative (the
    `(%rip)` addressing mode). This is the standard, PIE-friendly way to
    get a static data address on x86-64 -- an absolute `movq
    $label, %reg` would work on some setups but isn't safe to rely on
    once position-independent executables are in the picture (the
    default for `gcc`-produced binaries on both Linux and macOS), so
    this is what every string literal's address gets loaded with (see
    gen_string_literal_into)."""
    label: str
    dst: Register
    mnemonic = "leaq"

    def operands(self) -> List[str]:
        return [f"{self.label}(%rip)", self.dst.emit()]


@dataclass
class LeaQFrame(Instruction):
    """Loads the *address* of a %rbp-relative stack location into
    `dst` -- `leaq offset(%rbp), dst`. Distinct from LeaQ (which is
    RIP-relative, for static data like string literals): this is
    relative to the CURRENT function's own frame, and is how an
    array-typed local's address is obtained -- see the ARRAYS section
    for why arrays need their own address computed at all, unlike a
    scalar local, which is always read/written directly by offset."""
    offset: int
    dst: Register
    mnemonic = "leaq"

    def operands(self) -> List[str]:
        return [f"{self.offset}(%rbp)", self.dst.emit()]


@dataclass
class CallInstr(Instruction):
    """Calls a function (either a libc routine like `strlen`, or another
    Hornet-compiled function) by symbol name, e.g. `call strlen` or
    `call add`. Named CallInstr rather than plain Call specifically to
    avoid colliding with parser.Call -- the source-level AST node for a
    function-call *expression* -- which this file also imports; the two
    are easy to conflate by name but are completely different things
    (one is assembly, the other is source syntax), and Python will
    silently let a module-level class definition shadow an import of
    the same name with no error, which is exactly what happened here
    during development before this rename.

    `target` is always the *unprefixed* C symbol name (`malloc`, not
    `_malloc`) -- Emitter is what knows whether the target platform
    needs a leading underscore (see its emit_function), the same way it
    already decides that for this program's own function labels. Emit()
    here (unprefixed) is only ever used if a CallInstr is inspected/
    rendered outside of Emitter; the real rendering path always goes
    through Emitter's own handling instead.

    Requires %rsp to be 16-byte aligned at the point this executes, per
    the SysV ABI -- see codegen.py's LIBRARY CALLS section for how that
    invariant is maintained without explicit runtime alignment checks.
    """
    target: str
    mnemonic = "call"

    def operands(self) -> List[str]:
        return [self.target]


@dataclass
class MovQ(Instruction):
    """64-bit mov (`movq`). Used for frame-pointer setup (`movq %rsp,
    %rbp`) and for anything genuinely 64-bit -- which, as of `str`, now
    includes string pointers (see codegen.py's LOCAL VARIABLES and
    STRINGS sections). int/bool still exclusively use the 32-bit Mov
    (`movl`)."""
    src: Operand
    dst: Operand
    mnemonic = "movq"

    def operands(self) -> List[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class SubQ(Instruction):
    """64-bit subtract (`subq`). Used exactly once per function, in the
    prologue, to reserve stack space for locals: `subq $N, %rsp`."""
    src: Operand
    dst: Operand
    mnemonic = "subq"

    def operands(self) -> List[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class Leave(Instruction):
    """Tears down the current stack frame: equivalent to
    `movq %rbp, %rsp; popq %rbp`. The standard epilogue counterpart to
    the prologue's `pushq %rbp; movq %rsp, %rbp`."""
    mnemonic = "leave"


@dataclass
class Label(Instruction):
    """A jump target. Not really an "instruction" (it assembles to
    nothing -- it just names the address of whatever comes next), but it
    fits the same emit()-based rendering as everything else."""
    name: str

    def emit(self) -> str:
        return f"{self.name}:"


@dataclass
class Jmp(Instruction):
    """Unconditional jump to `target` (a Label's name)."""
    target: str
    mnemonic = "jmp"

    def operands(self) -> List[str]:
        return [self.target]


@dataclass
class Je(Instruction):
    """Jump to `target` if the last Cmp found its operands equal (ZF set)."""
    target: str
    mnemonic = "je"

    def operands(self) -> List[str]:
        return [self.target]


@dataclass
class Jne(Instruction):
    """Jump to `target` if the last Cmp found its operands unequal (ZF clear)."""
    target: str
    mnemonic = "jne"

    def operands(self) -> List[str]:
        return [self.target]


@dataclass
class Jae(Instruction):
    """Jump to `target` if the last Cmp found dst >= src, using an
    UNSIGNED interpretation of the compared values -- unlike Je/Jne,
    which only look at the zero flag (equal or not, meaningless
    whether signed or unsigned). This is what makes array bounds
    checking a single comparison: `cmpl $size, %index; jae fail_label`
    correctly catches BOTH index >= size and index < 0 at once, since
    a negative int, reinterpreted unsigned, becomes a huge positive
    number -- see gen_index_address_into."""
    target: str
    mnemonic = "jae"

    def operands(self) -> List[str]:
        return [self.target]


@dataclass
class Ja(Instruction):
    """Jump to `target` if the last Cmp found dst > src (STRICTLY
    greater), using an UNSIGNED interpretation -- the strict-
    inequality counterpart to Jae, needed for slice bounds checking
    specifically (see gen_slice_into): `low == length` and
    `high == length` are both VALID slice bounds (`arr[5:5]` on a
    5-element array is a valid, empty-slice-producing expression),
    unlike ordinary indexing, where an index equal to the array's own
    size is already out of bounds -- so the boundary condition itself
    genuinely differs here, not just the label it jumps to. Still
    catches a negative value via the same unsigned-reinterpretation
    trick Jae relies on: a negative int, reinterpreted unsigned,
    becomes huge, and so is "above" any non-negative length."""
    target: str
    mnemonic = "ja"

    def operands(self) -> List[str]:
        return [self.target]


@dataclass
class Ret(Instruction):
    mnemonic = "ret"


@dataclass
class AsmFunction:
    name: str
    instructions: List[Instruction] = field(default_factory=list)


@dataclass
class AsmProgram:
    functions: List[AsmFunction] = field(default_factory=list)
    # (label, content) pairs for every string literal anywhere in the
    # program, collected across all functions during generation (see
    # CodeGenerator.gen_string_literal_into). These aren't tied to any
    # one function's frame -- they're static, immutable data -- so they
    # live at the AsmProgram level and get emitted once, in a shared
    # `.data` block, by Emitter (see its emit()).
    string_literals: List[tuple] = field(default_factory=list)


# 32-bit register name -> its 8-bit low-byte alias (e.g. %eax -> %al).
# `sete` (and friends) can only target an 8-bit operand, so codegen needs
# to be able to get from "the register I'm working in" to "its byte
# alias". Only registers actually in use are listed; extend this table
# alongside whatever new registers the code generator starts using.
_BYTE_REGISTER_ALIASES = {
    'eax': 'al',
}


def as_byte_register(reg: Operand) -> Register:
    if not isinstance(reg, Register) or reg.name not in _BYTE_REGISTER_ALIASES:
        raise CodegenError(f"No 8-bit alias known for register operand: {reg!r}")
    return Register(_BYTE_REGISTER_ALIASES[reg.name])


# 32-bit register name -> its 64-bit alias (e.g. %eax -> %rax). Needed
# because Push/Pop can't operate on a 32-bit operand size in long mode.
_QWORD_REGISTER_ALIASES = {
    'eax': 'rax',
    'ecx': 'rcx',
}


def as_qword_register(reg: Operand) -> Register:
    if not isinstance(reg, Register) or reg.name not in _QWORD_REGISTER_ALIASES:
        raise CodegenError(f"No 64-bit alias known for register operand: {reg!r}")
    return Register(_QWORD_REGISTER_ALIASES[reg.name])


# BinaryOp -> the x86 condition-code suffix that implements it, given
# that Cmp(src=right, dst=left) computes (left - right) and sets flags
# accordingly. All six comparisons share one codegen path (see
# gen_binary_op) that just plugs the relevant cc into SetCC.
_COMPARISON_CONDITION_CODES = {
    BinaryOp.EQUAL: 'e',
    BinaryOp.NOT_EQUAL: 'ne',
    BinaryOp.LESS_THAN: 'l',
    BinaryOp.GREATER_THAN: 'g',
    BinaryOp.LESS_THAN_OR_EQUAL: 'le',
    BinaryOp.GREATER_THAN_OR_EQUAL: 'ge',
}


# SysV ABI integer/pointer argument registers, in order, 64-bit and
# 32-bit forms. Only the first 6 arguments of a call are supported --
# beyond that the ABI moves to stack-passed arguments, which this
# compiler doesn't implement (see gen_call_into and gen_function's
# param-count checks). The 32-bit names don't follow one consistent
# pattern: rdi/rsi/rdx/rcx are "legacy" registers with their own
# historical e-prefixed names, while r8/r9 are x86-64-only and use a
# d-suffix instead -- hence two explicit parallel lists rather than a
# derived/computed mapping.
_ARG_REGISTERS_64 = ['rdi', 'rsi', 'rdx', 'rcx', 'r8', 'r9']
_ARG_REGISTERS_32 = ['edi', 'esi', 'edx', 'ecx', 'r8d', 'r9d']

# Registers gen_string_concat_into/gen_string_compare_into use as
# scratch (see STRINGS). Now that functions can call each other,
# *every* function's prologue/epilogue saves and restores these
# unconditionally -- see gen_function and gen_return -- regardless of
# whether that particular function happens to use them, because the
# callee-saved contract has to hold for any call, not just ones this
# compiler happens to know use string operations. See the module
# docstring's FUNCTIONS section for why this became necessary.
_CALLEE_SAVED_SCRATCH_REGISTERS = ['rbx', 'r12', 'r13', 'r14']


def type_byte_width(t: Type) -> int:
    """Total bytes needed to store a value of type `t`: 4 for int/bool,
    8 for str (a pointer), 24 for a slice (its own fixed-size
    descriptor -- {ptr, len, cap}, 8 bytes each, in that order,
    matching Go's own slice header layout -- see the SLICES section --
    regardless of what it's a slice OF: two slices of different
    element types are still both 24 bytes, unlike two arrays of
    different element types or sizes), and recursively `size *
    type_byte_width(element_type)` for an array -- its full, flattened
    stack footprint, matching how it's laid out contiguously in
    row-major order regardless of how many dimensions it has (see the
    ARRAYS section). This is the one place that recursion lives; every
    caller that needs an array's total size (stack allocation,
    whole-array copies) or the shift-per-index (address computation)
    goes through this or leaf_type below rather than re-deriving
    either."""
    if t.kind == TypeKind.ARRAY:
        return t.size * type_byte_width(t.element_type)
    if t.kind == TypeKind.SLICE:
        return 24
    if t.kind == TypeKind.STR:
        return 8
    return 4  # INT, BOOL


def leaf_type(t: Type) -> Type:
    """Recursively unwraps array types to find the innermost, non-array
    element type -- e.g. for [2][3]int, the leaf type is int. Used
    wherever codegen needs to know the actual SCALAR type stored at
    the bottom of a (possibly multi-dimensional) array, e.g. to decide
    whether a flat element-by-element copy should move 4 or 8 bytes at
    a time -- a multi-dimensional array is just one contiguous block
    of leaf values for copying purposes, with no per-dimension logic
    needed once this is known.

    Stops at a SLICE the same way it already stops at STR -- neither
    is unwrapped further, since both are copied as one fixed-size
    unit (a pointer, or a {pointer, length} pair) rather than
    recursed into element by element. An array whose ELEMENTS are
    themselves slices (`[3][]int`) is a real gap this leaves --
    gen_array_copy's own flat-copy loop only knows how to move 4 or 8
    bytes at a time, not a slice descriptor's 16 -- see its own
    docstring for the explicit, deliberate rejection this leads to,
    rather than a silent miscompile."""
    while t.kind == TypeKind.ARRAY:
        t = t.element_type
    return t


# Fixed, hardcoded threshold for size-based stack safety (see the
# module docstring's ARRAYS section). Any array-typed local or
# parameter whose total flattened footprint (type_byte_width) exceeds
# this many bytes is heap-allocated instead of living inline in its
# own stack slot -- a deliberately simple, PER-ARRAY check, not a
# per-frame budget: it catches the single-array case (one huge local
# or parameter that would blow the stack on its own) but not, say,
# five moderately-sized arrays in the same function each individually
# under the limit, or a moderate array in a deeply recursive call
# chain. Both are known, accepted gaps, not oversights -- closing them
# would mean summing every local's width per frame (or per call
# chain), which is real additional complexity for a problem this
# simple, per-array check already solves for the common case that
# actually matters: one array declared far too large for the stack.
#
# 16KB, not the more "principled" 4KB single-page size: generous
# enough that ordinary matrix-shaped code (e.g. a [50][50]int, exactly
# 10000 bytes) doesn't get quietly promoted to the heap by surprise,
# while still leaving a wide safety margin against the default ~8MB
# stack budget even under moderate recursion (8MB / 16KB = 512
# same-sized frames before exhaustion).
_STACK_ARRAY_LIMIT_BYTES = 16384


def is_heap_allocated(t: Type) -> bool:
    """Whether a value of type `t` is heap-allocated rather than stored
    inline in its own stack slot purely because of its OWN size -- true
    for an array type whose total footprint (type_byte_width) exceeds
    _STACK_ARRAY_LIMIT_BYTES, false for every scalar type and every
    array under the limit. Purely a function of the type itself, not of
    any per-variable state, so it's never stored anywhere -- anywhere
    codegen already has the Type (via _local_type or _type_of), it can
    just call this directly.

    This is NOT the only reason a particular array ends up heap-
    allocated any more -- see analyze_array_escapes below for the
    other, independent trigger (a small array that backs a slice which
    escapes the function it's declared in still needs to survive past
    that function's own return, regardless of its size) -- so a
    caller deciding whether a SPECIFIC, NAMED variable needs heap
    allocation should go through CodeGenerator._is_array_heap_allocated
    instead, which combines this size check with that escape-analysis
    result; this function alone only ever answers the size half of
    that question."""
    return t.kind == TypeKind.ARRAY and type_byte_width(t) > _STACK_ARRAY_LIMIT_BYTES


def analyze_array_escapes(fn: Function, param_types: List[Type]) -> Set[int]:
    """Returns the set of id()s -- of this function's own VarDecl or
    Param nodes -- for array-typed declarations that need to be heap-
    allocated because a slice backed by them might outlive this
    function's own call, REGARDLESS of their own size. This is what
    closes the real memory-safety gap size-based heap promotion alone
    left open: a small, stack-allocated array, sliced and returned
    (directly, or via a named slice variable, or after an `append`
    that happened to reuse its backing storage), leaves the returned
    slice's own pointer field dangling into a stack frame that's
    already been torn down by the time anything reads it again.

    An intraprocedural, FLOW-INSENSITIVE analysis: it doesn't reason
    about the order statements execute in, or which branch of an
    if/while actually runs -- every assignment to a given slice
    variable ANYWHERE in the function is unioned together into one
    combined "what might this be backed by" answer, used uniformly
    wherever that variable is read. This is a real, deliberate
    simplification (a variable reused for two logically-unrelated
    slices at different points in the same function gets treated as
    if it could be either one everywhere), not an oversight -- it
    avoids needing a genuine fixed-point dataflow pass over branches
    and loops, which true flow sensitivity would require even before
    any function calls enter the picture, for a level of precision
    ordinary code doesn't often need. The SAME flow-insensitive
    treatment now also covers a slice stored as an element of an
    AGGREGATE -- today, specifically an array- or slice-of-slices
    (`rows[i] = arr[:]`) -- see AGGREGATES AND SLOTS below for what
    that word is doing here and why it's phrased more generally than
    "array-of-slices" alone. Three further, real limitations, each
    deliberately out of scope for now rather than silently mishandled:
      - Purely INTRAprocedural: a slice passed as an argument to any
        user-defined function call is conservatively treated as
        escaping unconditionally, without looking at what the callee
        actually does with it (does it store it somewhere that
        outlives the call, or just read it and let it go). A real
        interprocedural version would need a per-function escape
        SUMMARY, computed for every function in dependency order --
        and since Hornet allows recursion, computing those soundly
        needs a genuine fixed-point iteration over the call graph, not
        a single pass. That's a substantially larger undertaking than
        this, and left for its own, separate follow-up.
      - Only ONE level of aggregate nesting is tracked: `rows[i] =
        arr[:]` where `rows` is a bare Variable is handled, but
        `matrix[i][j] = arr[:]` (an aggregate reached through a
        further Index, not a bare Variable) is not -- IndexAssign's
        own target has to resolve directly to a declared aggregate,
        matching the single-hop treatment elsewhere in this analysis.
      - A slice stored as an element of something that ISN'T itself a
        declared local or parameter -- e.g. through a pointer-like
        indirection this language doesn't actually have -- was never
        in scope to begin with and remains so.

    The algorithm itself has two phases:
      1. Walk the function body once (recursing into every If's own
         then_body/else_body and every While's own body, maintaining a
         scope stack so a name resolves to the SPECIFIC declaration it
         actually refers to at that point -- Hornet allows shadowing a
         name in a nested block, so a plain name-based lookup would
         risk conflating two entirely different variables), building:
           - direct_backing: for each trackable node (a slice-typed
             declaration OR an aggregate's own slot -- see AGGREGATES
             AND SLOTS below), which array-typed declaration(s) it's
             ever directly sliced from (`s = arr[low:high]`, or
             `s = matrix[i][low:high]` -- indexing into a multi-
             dimensional array's own row is still a view into the SAME
             backing storage, not a copy, so this unwraps nested Index
             nodes down to their root Variable rather than requiring a
             bare one).
           - slice_deps: for each trackable node, which OTHER trackable
             node(s) it might in turn be derived from (re-slicing a
             slice, a plain slice-to-slice copy, `append` -- which
             might reuse its own first argument's backing array -- or
             reading an element back out of an aggregate).
           - escaping_slices / escaping_arrays: nodes directly marked
             as escaping, from a `return` statement's own value or a
             slice/array-typed argument passed to a user-defined call
             (found via a full recursive scan of every expression for
             a nested Call, not just ones at a statement's own top
             level -- `return foo(bar(s))` still needs `s` to be
             checked as bar's own argument even though the RETURN
             itself is really about foo's result, not s directly).

         AGGREGATES AND SLOTS: this is the piece that exists purely to
         make the NEXT thing built on top of this analysis (struct
         support) a small addition rather than a third near-copy of
         very similar logic -- see the module's own note on why this
         was worth doing as its own, standalone step before struct
         work starts. An "aggregate" is any declaration that can hold
         MULTIPLE independently-accessed values, at least one of which
         might be slice-typed -- today, that's exactly the array- and
         slice-of-slices case, but a struct is exactly this too, just
         with named fields instead of indices. A "slot" identifies
         WHICH part of the aggregate is being accessed; slot_node_id
         gives every distinct (aggregate declaration, slot) pair its
         own stable, synthesized node id (a small negative integer,
         guaranteed distinct from every real id() -- id() is always a
         positive address in CPython -- and from every other slot's own
         id, generated once per pair and memoized in aggregate_slot_ids
         for every later request), then registers THAT id in the very
         SAME slice_decls/direct_backing/slice_deps structures a bare
         slice-typed declaration already uses. Nothing downstream --
         not the transitive closure walk, not contribution()'s own
         callers -- needs to know or care whether a node id it's
         holding came from id()-ing a real declaration or from
         slot_node_id: they're both just integers in the same graph.

         Only one KIND of slot exists right now: indexed_slot_of
         recognizes `rows[i]` (an Index) or `rows[i] = ...`
         (IndexAssign's own target) where `rows` is a bare Variable
         declared with an array or slice element type that's ITSELF
         slice-typed, and maps it to the slot key _INDEXED_ELEMENTS_
         SLOT -- a single, SHARED slot for the whole declaration,
         regardless of which actual index `i` evaluates to at runtime,
         since indices are dynamic values this analysis can't
         distinguish without real per-index tracking (a different,
         larger undertaking, and not attempted here -- see the
         limitations list above). This is exactly the "one combined
         blob per declaration" flow-insensitive treatment a bare slice
         variable already gets, just extended to an aggregate's
         elements collectively.

         WHOLE-VALUE READS OF AN AGGREGATE ALSO GO THROUGH THE SAME
         SLOT, not a separate one: `return rows`, `rows2 = rows`, and
         `rows` passed as a call argument all resolve to indexed_slot_
         of's own node (via whole_value_node_of, contribution()'s and
         the VarDecl/Assign target-resolution's shared entry point for
         "what node tracks whatever this name's value is"), exactly
         like reading `rows[i]` does -- not a second, disconnected node
         that happens to sit next to it. This matters concretely, not
         just tidily: `rows` and `rows2` in `rows2 = rows` alias the
         SAME backing storage (copying a slice descriptor is a shallow,
         alias-preserving copy, the same as anywhere else in this
         language), so a later `rows[0] = arr[:]` has to still be
         visible through `rows2[0]` -- and, just as much, through a
         bare `return rows2` -- for this analysis to stay sound. This
         wasn't a hypothetical worth hardening pre-emptively: an
         earlier version of this exact refactor gave whole-aggregate
         reads their OWN separate node instead of sharing indexed_slot_
         of's, and `return rows` (with nothing ever indexing into it
         at all) silently stopped resolving to anything -- caught by
         testing the refactor against the very scenarios it was
         supposed to preserve, not found by inspection.

         When struct support lands, a FIELD access (`s.my_ints`) would
         get its own analogous function -- say, field_slot_of -- doing
         the same shape of recognition (a bare Variable declared with a
         struct type, whose SPECIFIC named field is slice-typed) but
         computing a genuinely PRECISE slot key from the field's own
         name rather than one shared sentinel: unlike a dynamic array
         index, a field name is known statically, so `s.a` and `s.b`
         can -- and should -- get their OWN separate slots rather than
         being lumped together the way `rows[i]` and `rows[j]` have to
         be. slot_node_id already supports this without any change:
         it's already keyed on an arbitrary slot value, not hardcoded
         to the one sentinel indexed_slot_of happens to use today. And
         whole_value_node_of would need the analogous extension too --
         a bare `Variable` referring to a struct falls back to it
         exactly like an aggregate-of-slices does, so `return s` has to
         resolve to (the union of) that struct's own field slots the
         same deliberate way `return rows` resolves to indexed_slot_
         of's.

         An array-typed aggregate (`[N][]int`) is registered in BOTH
         array_decls (for its own, unrelated existing role -- e.g.
         `rows[0:2]`, slicing the aggregate itself, still works via the
         existing array_decls/root_variable_name path) and, via its
         slot(s), the unified slice-tracking structure (for its role as
         a holder of slice elements) -- these are two independent
         things a single declaration can be, not a conflict between
         them.
      2. Compute the transitive closure from escaping_slices, following
         slice_deps edges (an ordinary graph reachability walk -- BFS
         via an explicit stack, not recursion, so it can't stack-
         overflow on a pathologically long dependency chain), unioning
         in direct_backing at every node reached along the way, plus
         escaping_arrays found directly. The result is exactly the set
         of array declarations that need to survive past this
         function's own return.
    """
    array_decls: Set[int] = set()
    slice_decls: Set[int] = set()
    decl_types: Dict[int, Type] = {}
    direct_backing: Dict[int, Set[int]] = {}
    slice_deps: Dict[int, Set[int]] = {}
    escaping_slices: Set[int] = set()
    escaping_arrays: Set[int] = set()
    aggregate_slot_ids: Dict[Tuple[int, str], int] = {}

    scopes: List[Dict[str, int]] = [{}]

    def resolve(name: str) -> Optional[int]:
        for scope in reversed(scopes):
            if name in scope:
                return scope[name]
        return None

    def declare(name: str, decl_id: int, decl_type: Type) -> None:
        scopes[-1][name] = decl_id
        decl_types[decl_id] = decl_type
        if decl_type.kind == TypeKind.ARRAY:
            array_decls.add(decl_id)
        if decl_type.kind == TypeKind.SLICE:
            slice_decls.add(decl_id)
            direct_backing.setdefault(decl_id, set())
            slice_deps.setdefault(decl_id, set())

    for p, p_type in zip(fn.params, param_types):
        declare(p.name, id(p), p_type)

    def slot_node_id(container_id: int, slot: str) -> int:
        """The one piece of plumbing every aggregate kind shares --
        see AGGREGATES AND SLOTS above. Deliberately agnostic to what
        `slot` actually means (an index-sentinel today, a field name
        once structs exist): callers decide what a slot IS; this just
        gives each distinct (container_id, slot) pair a stable node id
        in the shared slice-tracking graph, synthesizing one the first
        time that exact pair is seen and returning the same one every
        time after."""
        key = (container_id, slot)
        if key not in aggregate_slot_ids:
            node_id = -(len(aggregate_slot_ids) + 1)  # always negative;
            # id() is always a positive address in CPython, so this can
            # never collide with a real declaration's own node id.
            aggregate_slot_ids[key] = node_id
            slice_decls.add(node_id)
            direct_backing.setdefault(node_id, set())
            slice_deps.setdefault(node_id, set())
        return aggregate_slot_ids[key]

    _INDEXED_ELEMENTS_SLOT = '[]'  # the one shared slot indexed_slot_of
    # uses for a whole array-/slice-of-slices declaration, regardless of
    # which actual index is involved (see AGGREGATES AND SLOTS above for
    # why) -- chosen because '[' and ']' can never appear in a Hornet
    # identifier, so this can never collide with a future field-name slot.

    def indexed_slot_of(base_expr: Node) -> Optional[int]:
        """Recognizes `base_expr` as a bare Variable, declared with an
        array- or slice-of-slices type, that Index/IndexAssign is
        accessing -- e.g. the `rows` in `rows[i]` or `rows[i] = ...` --
        and returns that declaration's own shared indexed-elements slot
        id (see slot_node_id), or None if base_expr doesn't resolve to
        one at all (a different kind of base entirely, an aggregate
        whose element type isn't itself slice-typed, or an expression
        more complex than a bare Variable -- see this function's own
        single-hop limitation, documented in this analysis's own
        docstring). Just a Variable-shaped wrapper around whole_value_
        node_of, which does the actual resolution and is what makes
        indexed access and whole-value access of the same aggregate
        share one node rather than getting two disconnected ones --
        see WHOLE-VALUE READS OF AN AGGREGATE ALSO GO THROUGH THE SAME
        SLOT above for why that sharing is load-bearing, not cosmetic."""
        if not isinstance(base_expr, Variable):
            return None
        return whole_value_node_of(base_expr.name)

    def whole_value_node_of(name: str) -> Optional[int]:
        """Resolves `name` to the node id this analysis's graph uses
        to track EVERYTHING relevant about the value it holds --
        deliberately shared with indexed_slot_of's own notion of "the
        declaration's indexed-elements slot" when `name` is an
        aggregate-of-slices, since a whole-aggregate read (`return
        rows`, `rows2 = rows`, `rows` passed to a call) has to be
        treated as being just as capable of exposing ANY element's own
        backing as reading one element out directly is -- they're the
        SAME underlying storage, so they get the SAME node, not two
        disconnected ones that would silently stop propagating into
        each other. Falls back to `name`'s own bare declaration id for
        an ORDINARY slice-typed declaration (not an aggregate at all),
        exactly like before this function existed. Returns None if
        `name` doesn't resolve to anything this analysis tracks."""
        decl_id = resolve(name)
        if decl_id is None:
            return None
        decl_type = decl_types.get(decl_id)
        if decl_type is not None and decl_type.kind in (TypeKind.ARRAY, TypeKind.SLICE):
            element_type = decl_type.element_type
            if element_type is not None and element_type.kind == TypeKind.SLICE:
                return slot_node_id(decl_id, _INDEXED_ELEMENTS_SLOT)
        if decl_id in slice_decls:
            return decl_id
        return None

    def root_variable_name(expr: Node) -> Optional[str]:
        while isinstance(expr, Index):
            expr = expr.array
        return expr.name if isinstance(expr, Variable) else None

    def contribution(value_expr: Node) -> Tuple[Optional[int], Optional[int]]:
        """Returns (array_decl_id, slice_decl_id) -- whichever ONE of
        the two value_expr's own aliasing actually resolves to (never
        both), or (None, None) if it isn't backed by any of this
        function's own declarations at all (a fresh literal, `none`,
        an ordinary user-function call's own return value, ...). An
        aggregate's own slot id (see AGGREGATES AND SLOTS above) is
        returned as the second element here exactly like a bare slice
        Variable's own id would be -- callers don't need to know or
        care that it came from indexing into an aggregate rather than
        reading a plain slice variable directly."""
        if isinstance(value_expr, Slice):
            base_name = root_variable_name(value_expr.array)
            if base_name is not None:
                base_id = resolve(base_name)
                if base_id in array_decls:
                    return base_id, None
                if base_id in slice_decls:
                    return None, base_id
        elif isinstance(value_expr, Variable):
            node_id = whole_value_node_of(value_expr.name)
            if node_id is not None:
                return None, node_id
        elif isinstance(value_expr, Index):
            # Reading an element back out of a declared aggregate --
            # e.g. `rows[i]` -- resolves to that aggregate's own
            # indexed-elements slot.
            slot_id = indexed_slot_of(value_expr.array)
            if slot_id is not None:
                return None, slot_id
        elif isinstance(value_expr, Call) and value_expr.name == 'append':
            # append's own first argument might reuse ITS OWN backing
            # storage (the reuse path -- see gen_append_call_into), so
            # whatever that argument itself resolves to is exactly
            # this call's own contribution too. Recursing into
            # contribution() here (rather than only handling a bare
            # Variable) means append's first argument gets the SAME
            # treatment any other slice-valued expression already
            # does -- an aggregate element (`append(rows[0], v)`), an
            # unnamed slice expression (`append(arr[0:2], v)`), or
            # even another append call's own result -- not just a
            # named slice variable.
            return contribution(value_expr.args[0])
        return None, None


    def scan_expr_for_escaping_calls(expr: Node) -> None:
        if isinstance(expr, Call):
            if expr.name not in ('print', 'len', 'append'):
                for arg in expr.args:
                    array_id, slice_id = contribution(arg)
                    if array_id is not None:
                        escaping_arrays.add(array_id)
                    if slice_id is not None:
                        escaping_slices.add(slice_id)
            for arg in expr.args:
                scan_expr_for_escaping_calls(arg)
        elif isinstance(expr, Binary):
            scan_expr_for_escaping_calls(expr.left)
            scan_expr_for_escaping_calls(expr.right)
        elif isinstance(expr, Unary):
            scan_expr_for_escaping_calls(expr.operand)
        elif isinstance(expr, Index):
            scan_expr_for_escaping_calls(expr.array)
            scan_expr_for_escaping_calls(expr.index)
        elif isinstance(expr, Slice):
            scan_expr_for_escaping_calls(expr.array)
            if expr.low is not None:
                scan_expr_for_escaping_calls(expr.low)
            if expr.high is not None:
                scan_expr_for_escaping_calls(expr.high)
        elif isinstance(expr, ArrayLiteral):
            for element in expr.elements:
                scan_expr_for_escaping_calls(element)
        # Variable, Constant, BoolLiteral, StringLiteral, NoneLiteral:
        # leaves, nothing further to recurse into.

    def walk_statements(statements: List[Node]) -> None:
        for stmt in statements:
            if isinstance(stmt, VarDecl):
                var_type = type_from_name(stmt.var_type)
                declare(stmt.name, id(stmt), var_type)
                if stmt.init is not None:
                    target_node = whole_value_node_of(stmt.name)
                    if target_node is not None:
                        array_id, slice_id = contribution(stmt.init)
                        if array_id is not None:
                            direct_backing[target_node].add(array_id)
                        if slice_id is not None:
                            slice_deps[target_node].add(slice_id)
                    scan_expr_for_escaping_calls(stmt.init)
            elif isinstance(stmt, Assign):
                target_node = whole_value_node_of(stmt.name)
                if target_node is not None:
                    array_id, slice_id = contribution(stmt.value)
                    if array_id is not None:
                        direct_backing[target_node].add(array_id)
                    if slice_id is not None:
                        slice_deps[target_node].add(slice_id)
                scan_expr_for_escaping_calls(stmt.value)
            elif isinstance(stmt, IndexAssign):
                slot_id = indexed_slot_of(stmt.array)
                if slot_id is not None:
                    array_id, slice_id = contribution(stmt.value)
                    if array_id is not None:
                        direct_backing[slot_id].add(array_id)
                    if slice_id is not None:
                        slice_deps[slot_id].add(slice_id)
                scan_expr_for_escaping_calls(stmt.array)
                scan_expr_for_escaping_calls(stmt.index)
                scan_expr_for_escaping_calls(stmt.value)
            elif isinstance(stmt, Return):
                if stmt.value is not None:
                    array_id, slice_id = contribution(stmt.value)
                    if array_id is not None:
                        escaping_arrays.add(array_id)
                    if slice_id is not None:
                        escaping_slices.add(slice_id)
                    scan_expr_for_escaping_calls(stmt.value)
            elif isinstance(stmt, ExprStmt):
                scan_expr_for_escaping_calls(stmt.expr)
            elif isinstance(stmt, If):
                scan_expr_for_escaping_calls(stmt.condition)
                scopes.append({})
                walk_statements(stmt.then_body)
                scopes.pop()
                if stmt.else_body is not None:
                    scopes.append({})
                    walk_statements(stmt.else_body)
                    scopes.pop()
            elif isinstance(stmt, While):
                scan_expr_for_escaping_calls(stmt.condition)
                scopes.append({})
                walk_statements(stmt.body)
                scopes.pop()
            # Break, Continue: nothing to do.

    walk_statements(fn.body)

    result: Set[int] = set(escaping_arrays)
    visited: Set[int] = set()
    stack: List[int] = list(escaping_slices)
    while stack:
        slice_id = stack.pop()
        if slice_id in visited:
            continue
        visited.add(slice_id)
        result |= direct_backing.get(slice_id, set())
        for dep in slice_deps.get(slice_id, set()):
            if dep not in visited:
                stack.append(dep)
    return result


# ---------------------------------------------------------------------------
# AST -> Assembly AST
# ---------------------------------------------------------------------------

class CodegenError(Exception):
    """Raised when the code generator encounters an AST node it doesn't
    know how to translate yet."""


class CodeGenerator:
    """Walks the source AST (Program/Function/Return/Constant/...) and
    produces an equivalent AsmProgram."""

    def __init__(self):
        self._label_count = 0
        self._var_offsets: Dict[int, int] = {}  # id(VarDecl node) -> its permanent Memory offset
        self._next_offset = 0
        self.scopes: List[Dict[str, tuple]] = []  # name -> (offset, Type), generation-time; see LOCAL VARIABLES
        self.loop_labels: List[tuple] = []  # stack of (start_label, end_label), innermost last; see LOOPS
        self.string_literals: List[tuple] = []  # (label, content) pairs; see STRINGS
        # Lazily created, then cached and reused for the rest of this
        # compilation -- see gen_print_call_into and the module
        # docstring's BUILTINS section for why these specifically (and
        # only these) get a small dedicated cache rather than following
        # string_literals' usual "every occurrence gets its own label,
        # no dedup" policy.
        self._int_format_label = None
        self._true_str_label = None
        self._false_str_label = None
        # Lazily created, but with different lifetimes from each other
        # -- see _get_bounds_check_fail_label/_get_bounds_check_message_
        # label's own docstrings. The fail labels are reset per function
        # (gen_function); the message labels, like the print-related
        # ones above, are cached for the whole compilation. Both are
        # dicts keyed by message text, since a function can trigger more
        # than one distinct bounds-check message (e.g. plain indexing
        # vs. a slice expression's own bounds).
        self._bounds_check_fail_labels = {}
        self._bounds_check_message_labels = {}
        # Lazily created and cached for the whole compilation, keyed by
        # content -- a general-purpose version of the print-related
        # caches above, for the punctuation/prefix pieces printing an
        # array or slice needs (brackets, separators, a newline, each
        # type's own "[N]int"/"[]int"-style prefix string, and the
        # "'%s'" format used to quote a str element inside a
        # collection). See _get_static_string_label.
        self._static_string_labels = {}
        # Set fresh at the start of every gen_function call -- see its
        # own comments -- to either None (this function's own return
        # type isn't an array) or the %rbp offset of the stack slot
        # holding the hidden output pointer the caller passed in.
        # Declared here too, defensively, so referencing it before any
        # function has been generated fails with a clear AttributeError
        # rather than silently reading a stale value from a previous
        # instance of this class (there shouldn't be one, but this
        # costs nothing to be explicit about).
        self._hidden_return_ptr_offset = None

    def new_label(self, prefix: str) -> str:
        """Returns a fresh, uniquely-numbered local label like
        `.Land_short_0`. Needed because AND/OR/if codegen all emit real
        jump targets, and a program can contain any number of them --
        each one needs a name the assembler won't collide with any
        other."""
        label = f".L{prefix}_{self._label_count}"
        self._label_count += 1
        return label

    def generate(self, program: Program) -> AsmProgram:
        functions = [self.gen_function(fn) for fn in program.functions]
        return AsmProgram(functions=functions, string_literals=self.string_literals)

    def gen_function(self, fn: Function) -> AsmFunction:
        # Fresh allocator state per function -- variables don't persist
        # across functions, and offsets are relative to *this*
        # function's own %rbp.
        self._var_offsets = {}
        self._next_offset = 0
        # No declared return type (fn.return_type is None -- see
        # Function's own docstring in parser.py) means Type.VOID, the
        # same internal-only sentinel semantic.py's own analyze_function
        # already uses -- kept consistent here rather than reinventing
        # a second "no return type" representation in this file.
        return_type = Type.VOID if fn.return_type is None else type_from_name(fn.return_type)
        param_types = [type_from_name(p.type) for p in fn.params]

        # Which of THIS function's own array declarations need to be
        # heap-allocated because a slice backed by them might outlive
        # this function's own return, regardless of their own size --
        # see analyze_array_escapes's own docstring for the full
        # algorithm. Computed once, up front, since _collect_params/
        # _collect_locals (just below) already need to know this to
        # decide how much stack space each declaration's own slot
        # takes (8 bytes for a heap pointer vs. the array's own full
        # width) -- this has to exist before either of them run.
        self._escaping_array_ids = analyze_array_escapes(fn, param_types)

        # An array- OR slice-typed return needs a hidden pointer -- the
        # caller passes the address to write the result into, as an
        # extra, FIRST argument (see gen_array_call_into/gen_slice_
        # call_into), shifting every real parameter one register
        # position later. Rather than dedicate a register to holding it
        # for the whole function (which would need its own save/restore
        # discipline, and -- worse -- would break the callee-saved-
        # register prologue's even-push-count alignment invariant if
        # added on top of the existing four), it just gets its own
        # ordinary stack slot, handled by exactly the same "reserve a
        # slot, then store the incoming register into it" mechanism
        # every real parameter already uses. See the module docstring's
        # ARRAYS section.
        #
        # A slice-typed return uses this SAME mechanism now, not a
        # separate one -- a slice's own {ptr, len, cap} descriptor is
        # 24 bytes, too wide for any two- or three-register return
        # shape this compiler has precedent for, so gen_return's own
        # Slice case is now structurally identical to its Array one:
        # load the received pointer, hand it to gen_slice_value_into as
        # an ordinary Memory destination. This is also what makes
        # forwarding one slice-returning call's result straight out of
        # another free (`return otherFn()`), exactly like it already
        # was for arrays -- the SAME address just gets passed one level
        # deeper, with no intermediate copy ever materialized.
        self._hidden_return_ptr_offset = None
        arg_shift = 0
        if return_type.kind in (TypeKind.ARRAY, TypeKind.SLICE):
            self._next_offset -= 8
            self._hidden_return_ptr_offset = self._next_offset
            arg_shift = 1

        # A second, 24-byte slot -- reserved unconditionally for EVERY
        # function, not just ones that return or otherwise produce a
        # slice -- used by gen_indexable_base_into to materialize an
        # unnamed Slice or slice-returning Call expression's descriptor
        # when it's used directly as the base of a `[...]` chain (e.g.
        # `arr[:][0]`, or `matrix[:][0][0]`, or `someSliceFn()[0]`),
        # rather than requiring it be assigned to a named variable
        # first, and by gen_expr_stmt for a bare Slice-expression
        # statement. See gen_indexable_base_into's own docstring for
        # why reusing a single shared slot is safe even under
        # arbitrarily deep nesting (each materialization is fully
        # consumed -- read out into registers -- before any subsequent
        # one can write to it again, the same way a call stack's own
        # frames nest).
        self._next_offset -= 24
        self._unnamed_slice_temp_offset = self._next_offset

        # One extra, purely internal temp slot per parameter, used to
        # stash its incoming register value(s) immediately, before any
        # parameter is actually processed -- see the loop below for
        # why this has to happen up front rather than processing each
        # parameter directly out of its own argument register in turn.
        # 24 bytes for a slice parameter (its own ptr, len, AND cap
        # each need stashing), 8 for everything else.
        param_temp_offsets = []
        for p_type in param_types:
            width = 24 if p_type.kind == TypeKind.SLICE else 8
            self._next_offset -= width
            param_temp_offsets.append(self._next_offset)

        self._collect_params(fn.params)
        self._collect_locals(fn.body)
        self.scopes = [{}]

        # A slice parameter needs THREE consecutive argument-register
        # slots (its own ptr, len, then cap), not one -- matching
        # exactly how a real C compiler would pass a `struct{void*,
        # long,long}` parameter, and the same "running slot count, not
        # a straight per-parameter count" accounting _gen_call_
        # arguments_into already needs on the CALLER side for the same
        # reason.
        param_slots = sum(3 if pt.kind == TypeKind.SLICE else 1 for pt in param_types)
        total_slots = arg_shift + param_slots
        if total_slots > 6:
            raise CodegenError(
                f"Function '{fn.name}' needs {total_slots} argument "
                f"register(s) for its parameters (a slice-typed "
                f"parameter needs 3)"
                + (" plus the hidden array/slice-return pointer" if arg_shift else "")
                + " -- this compiler only supports up to 6 (passed via "
                "registers per the SysV ABI -- stack-passed parameters "
                "aren't implemented)"
            )

        instructions: List[Instruction] = [
            Push(Register('rbp')),
            MovQ(src=Register('rsp'), dst=Register('rbp')),
        ]
        # Save every callee-saved scratch register unconditionally, not
        # just in functions that happen to do string work themselves --
        # see _CALLEE_SAVED_SCRATCH_REGISTERS and the module docstring's
        # FUNCTIONS section for why this is now required rather than
        # optional once functions can call each other.
        for reg in _CALLEE_SAVED_SCRATCH_REGISTERS:
            instructions.append(Push(Register(reg)))

        frame_size = self._frame_size()
        if frame_size:
            instructions.append(SubQ(src=Imm(frame_size), dst=Register('rsp')))

        if self._hidden_return_ptr_offset is not None:
            instructions.append(MovQ(src=Register('rdi'), dst=Memory('rbp', self._hidden_return_ptr_offset)))

        # Parameters arrive in registers per the SysV ABI (shifted one
        # position later than usual if this function itself returns an
        # array or slice -- see arg_shift above). Handled in two passes
        # rather than reading each one directly out of its own argument
        # register in turn:
        #
        # FIRST, every incoming register is stashed into its own
        # temporary slot (param_temp_offsets, reserved above) via a
        # plain %rbp-relative store -- these never touch %rsp, so
        # there's no stack-alignment concern regardless of how many
        # parameters there are or which ones turn out to need malloc.
        # A slice parameter stashes THREE consecutive registers (its
        # own ptr, len, then cap) into its own 24-byte temp slot,
        # advancing the running register-index counter by 3 instead of
        # 1 -- the same accounting _gen_call_arguments_into already
        # needs on the CALLER side, for the same underlying reason (a
        # slice needs three argument-register slots, not one).
        #
        # SECOND, each parameter is processed using its safely-stashed
        # value(s) rather than its original argument register(s). This
        # two-pass structure exists specifically because a heap-
        # allocated array parameter (see is_heap_allocated) needs its
        # own malloc call to build an independent copy -- and malloc,
        # like any real call, can clobber every caller-saved register,
        # including OTHER, not-yet-processed parameters' own incoming
        # values still sitting in their argument registers. Stashing
        # everything first, before any malloc call can possibly run,
        # avoids that regardless of which parameters (if any) end up
        # needing one. (An earlier version of this tried protecting
        # registers with ordinary push/pop instead -- which works for
        # a single value, but breaks down here: popping one parameter's
        # value at a time, immediately before processing it, leaves a
        # DIFFERENT number of not-yet-popped values on the stack ahead
        # of each parameter's own malloc call, which misaligns %rsp
        # for roughly half of them. Plain %rbp-relative stores sidestep
        # that failure mode entirely, since they never move %rsp.)
        reg_index = arg_shift
        for i, p_type in enumerate(param_types):
            if p_type.kind == TypeKind.SLICE:
                instructions.append(MovQ(src=Register(_ARG_REGISTERS_64[reg_index]), dst=Memory('rbp', param_temp_offsets[i])))
                instructions.append(MovQ(src=Register(_ARG_REGISTERS_64[reg_index + 1]), dst=Memory('rbp', param_temp_offsets[i] + 8)))
                instructions.append(MovQ(src=Register(_ARG_REGISTERS_64[reg_index + 2]), dst=Memory('rbp', param_temp_offsets[i] + 16)))
                reg_index += 3
            else:
                instructions.append(MovQ(src=Register(_ARG_REGISTERS_64[reg_index]), dst=Memory('rbp', param_temp_offsets[i])))
                reg_index += 1

        for i, p in enumerate(fn.params):
            offset = self._bind_param(p)
            p_type = param_types[i]
            temp_offset = param_temp_offsets[i]
            if p_type.kind == TypeKind.ARRAY:
                if self._is_array_heap_allocated(id(p), p_type):
                    # Needs its own, independent heap copy -- exactly
                    # like the stack-allocated case below, just backed
                    # by malloc'd memory instead of an inline slot --
                    # to preserve value semantics across the call:
                    # mutating this parameter must never affect the
                    # caller's own array. %rbx holds the caller's
                    # pointer across the malloc call itself: it's
                    # callee-saved, so malloc (a well-behaved, ABI-
                    # conforming function) is obligated to preserve it,
                    # the same guarantee gen_string_concat_into's own
                    # malloc/strlen/strcpy calls already rely on.
                    instructions.append(MovQ(src=Memory('rbp', temp_offset), dst=Register('rbx')))
                    instructions.extend(self._gen_malloc_array(p_type))
                    instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', offset)))
                    instructions.append(MovQ(src=Register('rax'), dst=Register('r10')))
                    instructions.extend(self.gen_array_copy(Memory('r10', 0), Memory('rbx', 0), p_type))
                else:
                    instructions.append(MovQ(src=Memory('rbp', temp_offset), dst=Register('rbx')))
                    instructions.extend(self.gen_array_copy(Memory('rbp', offset), Memory('rbx', 0), p_type))
            elif p_type.kind == TypeKind.SLICE:
                # A slice parameter is never heap-promoted or copied
                # the way an array is -- it's just an alias, exactly
                # like any other slice variable, so this only needs to
                # copy the three already-stashed values (ptr, len, cap)
                # into its own permanent slot; no malloc, no is_heap_
                # allocated check. The underlying array it points to
                # (if any) is already guaranteed to outlive this call
                # regardless: analyze_array_escapes treats passing a
                # slice as an argument to a user-defined call (which is
                # exactly how THIS parameter got here) as escaping, so
                # whatever array backs it in the CALLER is already
                # heap-allocated by the time this function even starts.
                instructions.append(MovQ(src=Memory('rbp', temp_offset), dst=Register('rax')))
                instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', offset)))
                instructions.append(MovQ(src=Memory('rbp', temp_offset + 8), dst=Register('rax')))
                instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', offset + 8)))
                instructions.append(MovQ(src=Memory('rbp', temp_offset + 16), dst=Register('rax')))
                instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', offset + 16)))
            elif p_type == Type.STR:
                instructions.append(MovQ(src=Memory('rbp', temp_offset), dst=Register('rax')))
                instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', offset)))
            else:
                instructions.append(Mov(src=Memory('rbp', temp_offset), dst=Register('eax')))
                instructions.append(Mov(src=Register('eax'), dst=Memory('rbp', offset)))

        self._bounds_check_fail_labels = {}  # fresh, per-function jump targets; see their own docstring
        for stmt in fn.body:
            instructions.extend(self.gen_statement(stmt))
        if return_type == Type.VOID:
            # A function with no declared return type never has to
            # guarantee every path returns explicitly (see
            # analyze_function's own always_returns skip for this case)
            # -- its body can legitimately fall off the end, relying on
            # THIS trailing epilogue rather than a gen_return-emitted
            # one on every path. Every OTHER function never needed
            # this: always_returns already guarantees some gen_return-
            # emitted epilogue executes on every path, making a
            # trailing one here permanently unreachable dead code -- so
            # it's only ever added for the one case that genuinely
            # needs it. Without this, a void function that fell off the
            # end would fall straight through into whatever comes next
            # in the generated assembly instead -- the bounds-check
            # panic block below, or the next function's own prologue --
            # a real, silent crash, not a hypothetical one.
            #
            # Appended unconditionally, even when this particular body
            # happens to already return explicitly on every path (e.g.
            # via an if/else where both branches return): there's no
            # cheap way to know that without effectively re-running
            # always_returns, and an extra, unreachable epilogue costs
            # nothing but a few bytes.
            instructions.extend(self._gen_epilogue())
        instructions.extend(self._gen_bounds_check_panic_block())

        return AsmFunction(name=fn.name, instructions=instructions)

    def _collect_params(self, params: List[Param]) -> None:
        """Gives each parameter its own permanent stack slot, exactly
        like _collect_locals does for VarDecls (same node-identity
        keying) -- kept as a separate method since Param and VarDecl
        are different AST node types, not because parameters need
        fundamentally different treatment. Each slot's width is the
        parameter's own actual type width (see type_byte_width) -- 4
        bytes for int/bool, 8 for str, and an array's own full,
        flattened footprint for a stack-allocated array parameter --
        except for an array parameter over _STACK_ARRAY_LIMIT_BYTES
        (see is_heap_allocated), which only needs 8 bytes here: its
        slot holds a pointer to a heap block gen_function's own
        parameter loop allocates, not the array's data directly. See
        the module docstring's ARRAYS section. Called after
        gen_function has already reserved the hidden-return-pointer
        slot, if this function needs one -- _next_offset just keeps
        counting down from wherever it already is, agnostic to what it
        was decremented for so far."""
        for p in params:
            p_type = type_from_name(p.type)
            width = 8 if self._is_array_heap_allocated(id(p), p_type) else type_byte_width(p_type)
            self._next_offset -= width
            self._var_offsets[id(p)] = self._next_offset

    def _bind_param(self, p: Param) -> int:
        """The Param counterpart to _bind_local -- registers `p`'s name
        and declared type (as a real semantic.Type, via type_from_name,
        not the raw parser-level string/ArrayTypeExpr -- see
        _local_type's own docstring for why), plus id(p) itself (see
        _local_decl_id's own docstring for why that's needed
        alongside the type and offset), in the current scope, pointing
        at the permanent offset _collect_params already assigned it."""
        offset = self._var_offsets[id(p)]
        self.scopes[-1][p.name] = (offset, type_from_name(p.type), id(p))
        return offset

    def _collect_locals(self, statements: List[Node]) -> None:
        """Recursively walks `statements`, including into every If's
        then_body/else_body and every While's body, and gives each
        VarDecl found its own permanent stack slot, keyed by the AST
        node's identity rather than its name -- see the module
        docstring's LOCAL VARIABLES section for why that distinction
        now matters.

        Each slot's width is the variable's own actual type width (see
        type_byte_width) -- 4 bytes for int/bool, 8 for str, and an
        array's own full, flattened footprint (e.g. 24 bytes for
        [2][3]int) for a stack-allocated array local. Uniform 8-byte
        slots were a deliberate simplification back when str was the
        only thing wider than 4 bytes; a fixed-size array can be
        arbitrarily larger than 8 bytes, so that simplification stops
        making sense once arrays exist -- see the module docstring's
        ARRAYS section. An array whose own footprint exceeds
        _STACK_ARRAY_LIMIT_BYTES only needs 8 bytes here regardless of
        its real size (see is_heap_allocated): its slot holds a
        pointer to a heap block, allocated by gen_var_decl, not the
        array's data directly -- this is the one and only place that
        decision actually changes how much stack space gets reserved.
        No alignment padding is added between slots: x86-64 doesn't
        require aligned access the way some architectures do, and
        %rsp's OWN 16-byte alignment requirement is still satisfied
        purely by _frame_size rounding the TOTAL frame size up at the
        end, regardless of how the space within it is subdivided."""
        for stmt in statements:
            if isinstance(stmt, VarDecl):
                var_type = type_from_name(stmt.var_type)
                width = 8 if self._is_array_heap_allocated(id(stmt), var_type) else type_byte_width(var_type)
                self._next_offset -= width
                self._var_offsets[id(stmt)] = self._next_offset
            elif isinstance(stmt, If):
                self._collect_locals(stmt.then_body)
                if stmt.else_body is not None:
                    self._collect_locals(stmt.else_body)
            elif isinstance(stmt, While):
                self._collect_locals(stmt.body)

    def _frame_size(self) -> int:
        # Total bytes used by locals and parameters, rounded up to a
        # 16-byte boundary. Genuinely required, not just good practice:
        # gen_string_* and gen_call_into both emit real `call`
        # instructions (to malloc/strlen/strcpy/strcat/strcmp, or to
        # another Hornet function), and the SysV ABI requires %rsp to be
        # 16-byte-aligned at the point of every one of those. (The
        # callee-saved register pushes in gen_function's prologue don't
        # themselves need accounting for here -- there are always
        # exactly 4 of them, an already-even number of 8-byte pushes, so
        # they never change whether %rsp ends up aligned or not.)
        raw = -self._next_offset
        return ((raw + 15) // 16) * 16 if raw > 0 else 0

    def _push_scope(self) -> None:
        self.scopes.append({})

    def _pop_scope(self) -> None:
        self.scopes.pop()

    def _bind_local(self, stmt: VarDecl) -> int:
        """Registers `stmt`'s name -- its declared type, needed by
        _local_type, and id(stmt) itself, needed by _local_decl_id --
        in the current (innermost) generation-time scope, pointing at
        the permanent offset _collect_locals already assigned this
        exact VarDecl node, and returns that offset."""
        offset = self._var_offsets[id(stmt)]
        self.scopes[-1][stmt.name] = (offset, type_from_name(stmt.var_type), id(stmt))
        return offset

    def _local_offset(self, name: str) -> int:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name][0]
        raise CodegenError(f"Reference to undeclared variable '{name}'")

    def _local_type(self, name: str) -> Type:
        """Used specifically where a Variable's *offset* is also being
        looked up right alongside it (see gen_expr_into's Variable case,
        and gen_array_address_into) -- both come from the same
        (offset, Type, decl_id) tuple in the same scope-stack entry,
        which codegen has to maintain regardless of _type_of's
        existence, since resolved_type has no way to encode *which*
        stack slot a name refers to. This is deliberately not replaced
        by _type_of below, even though it would give the same answer
        for a Variable node -- see _type_of's own docstring for why the
        two coexist rather than one replacing the other.

        Returns a real semantic.Type (via type_from_name, called once
        up front in _bind_local/_bind_param, not re-derived here) --
        not the raw parser-level string/ArrayTypeExpr -- so callers can
        uniformly inspect .kind/.element_type/.size exactly like they
        already can on whatever _type_of returns."""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name][1]
        raise CodegenError(f"Reference to undeclared variable '{name}'")

    def _local_decl_id(self, name: str) -> int:
        """Returns id(the VarDecl or Param node) that `name` currently
        resolves to -- the third element of the same (offset, Type,
        decl_id) tuple _local_offset/_local_type already read the
        first two of, kept in the SAME scope-stack lookup (rather than
        a separate, parallel name-to-id table) specifically so this
        respects shadowing exactly like they do: Hornet allows
        re-declaring a name in a nested if/while block, so a plain
        name alone doesn't uniquely identify a declaration the way
        id() of the actual AST node does. Used by _is_array_heap_
        allocated to look up whether THIS SPECIFIC declaration (not
        just any variable that happens to share its name) was found to
        escape by analyze_array_escapes."""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name][2]
        raise CodegenError(f"Reference to undeclared variable '{name}'")

    def _is_array_heap_allocated(self, decl_id: int, t: Type) -> bool:
        """Whether the SPECIFIC array-typed declaration identified by
        decl_id (id() of its own VarDecl or Param node) needs to be
        heap-allocated -- combining is_heap_allocated's own, pure
        size check with analyze_array_escapes's own, independent
        result (computed once per function, in gen_function, and
        cached in self._escaping_array_ids): either reason alone is
        sufficient. This is the actual decision point every one of
        this file's 8 call sites that used to call is_heap_allocated
        directly now goes through instead, each passing whichever
        decl_id it has on hand -- id(a VarDecl or Param) directly, or
        self._local_decl_id(name) wherever only a Variable's own name
        is available at that point."""
        return is_heap_allocated(t) or decl_id in self._escaping_array_ids

    def _type_of(self, expr: Node) -> Type:
        """Reads the type semantic.py already resolved and annotated
        onto this exact node (expr.resolved_type -- see semantic.py's
        check_expr) rather than re-deriving it independently.

        This replaces what used to be a separate _infer_type method
        here that re-implemented, in miniature, the same "which type
        does this operator/call produce" logic semantic.py's
        check_binary/check_call already fully implement -- a second,
        parallel copy of that logic that could (and twice actually did)
        silently drift out of sync with the real one: adding `print`
        needed a Call case added here too, and adding the six new
        int-only operators (%, &, |, ^, <<, >>) needed them added to
        this method's own int-producing branch, separately from adding
        them to semantic.py's _INT_ONLY_BINARY_OPS. Neither addition
        was structurally required by anything -- both were just easy to
        forget, and both were only caught by manual testing rather than
        anything that would have failed loudly on its own. Reading the
        annotation instead removes the second copy entirely: there's no
        per-operator or per-node-type branch here left to forget
        updating, since whatever semantic.py already decided is just
        read directly, whatever it happens to be.

        Still raises a clear, defensive CodegenError (matching
        _local_offset's own posture) rather than a bare AttributeError
        if resolved_type is somehow None -- the one legitimate way that
        happens is codegen being invoked on an AST that skipped
        semantic analysis entirely (see compile_to_asm, which always
        runs analyze() first for exactly this reason).

        Returns a full semantic.Type object (not a string -- that
        changed when array types were added, since a bare name can't
        represent an element type and size). Callers can compare
        against Type.INT/Type.BOOL/Type.STR directly, or inspect
        .kind/.element_type/.size for an array.
        """
        if expr.resolved_type is None:
            raise CodegenError(
                f"{expr!r} has no resolved type -- semantic.analyze() "
                f"must run before codegen (see compile_to_asm)"
            )
        return expr.resolved_type

    # -- arrays -----------------------------------------------------------
    # See the module docstring's ARRAYS section for the full design.
    # Scope note: this covers LOCAL arrays completely -- declaration
    # (literal or copy-initialized), reading/writing an element at any
    # nesting depth, and whole-array copy via plain assignment. Array
    # function PARAMETERS and RETURN VALUES are a deliberately separate,
    # not-yet-built piece of work (a real calling-convention extension,
    # not a small addition) -- see gen_array_value_into's Call case and
    # gen_function's existing parameter loop, both of which raise a
    # clear CodegenError rather than silently mishandling one.

    def gen_array_literal_heap_alloc_into(self, expr: ArrayLiteral) -> List[Instruction]:
        """Mallocs a NEW, heap-allocated array sized to fit expr's own
        elements, writes them in (via the ordinary gen_array_literal_
        into, whose own dst_mem-protection logic already handles a
        non-'rbp' base correctly -- Memory('rax', 0) here is exactly
        that case, and by the time gen_array_literal_into returns,
        %rax is guaranteed to still hold the original malloc'd address,
        the same guarantee every other caller of it already relies
        on), and leaves the resulting pointer in %rax.

        Shared by both ways a slice literal's backing array gets
        created: the general, typed expression form (`[]int[1, 2, 3]`,
        parsed as an implicit whole-array Slice -- see parser.py's own
        _parse_bracketed_literal) via gen_indexable_base_into's own
        ArrayLiteral case, and the untyped form used directly as a
        slice-typed VarDecl/Assign value (`[]int s = [1, 2, 3]`) via
        gen_var_decl/gen_assign's own ArrayLiteral-as-slice-value
        short-circuit.

        Always allocates at least 1 byte, even for an empty literal
        (`[]int[]`) -- guaranteeing a genuine, non-null, unique pointer
        regardless of libc's own malloc(0) behavior (implementation-
        defined by POSIX; this doesn't rely on it), which is what makes
        `s == none` correctly false for an intentionally empty slice
        literal, matching the same nil-vs-empty distinction a real,
        non-empty slice already has (see gen_slice_none_comparison_
        into) -- `[]int[]` is a real, live, zero-length slice, not a
        nil one, the same way `arr[5:5]` already is.

        Every slice literal's own backing array is heap-allocated here
        UNCONDITIONALLY, regardless of size -- unlike an ordinary array
        variable, which only heap-promotes past the 16KB stack-size
        threshold (see is_heap_allocated). This isn't a size-based
        decision at all: a slice literal's backing array has to outlive
        the statement that creates it (the whole POINT of a slice is to
        be usable after the expression that produced it), so it needs
        the SAME "can safely cross frame boundaries" guarantee every
        OTHER sliced array already gets, unconditionally, for exactly
        the same reason.
        """
        array_type = self._type_of(expr)
        width = max(1, type_byte_width(array_type))
        instructions = [
            Mov(src=Imm(width), dst=Register('edi')),
            CallInstr('malloc'),
        ]
        instructions.extend(self.gen_array_literal_into(Memory('rax', 0), expr, array_type))
        return instructions

    def gen_indexable_base_into(self, expr: Node, addr_dst: Register, len_dst: Register, cap_dst: Register) -> Tuple[List[Instruction], Union[Imm, Register], Union[Imm, Register]]:
        """Computes the address of `expr`'s own data into `addr_dst`,
        and returns (instructions, length_operand, cap_operand): each
        operand is an Imm (a compile-time constant, equal to the
        array's own declared size for BOTH len and cap -- an array has
        no separate capacity concept of its own) when `expr` is
        array-typed, or `len_dst`/`cap_dst` themselves (populated with
        a runtime value read out of a slice's own descriptor) when
        `expr` is slice-typed. cap_dst is computed and populated
        unconditionally, even by callers (gen_index_address_into) that
        never read it back out afterward -- cheap enough (one extra
        Imm, or one extra runtime read alongside the len one already
        happening) that a single, uniform three-value contract beats
        making it optional.

        Shared by gen_index_address_into (indexing, `base[i]`, which
        never needs cap: an index equal to len is already out of
        bounds regardless of any spare room past it) and gen_slice_
        into (slicing, `base[low:high]`, which needs cap for both its
        own bounds check and the result's own capacity -- see its own
        docstring), and now gen_append_call_into too (which needs all
        three fields as genuine input values, not just for a bounds
        check) -- all three need exactly this same "address plus
        length (plus, now, capacity), however each is represented"
        information about whatever's on the left of a `[...]`
        expression or append's own first argument, and each already
        has to branch on which kind of Operand comes back for its own
        use of length.

        A slice-typed `expr` can be a Variable (a named slice, loaded
        directly out of its own %rbp-relative slot), a Slice (an
        UNNAMED slice expression used directly as a base -- e.g.
        `arr[:][0]`, or `matrix[:][0][0]` -- materialized into a
        dedicated, per-function scratch slot, _unnamed_slice_temp_
        offset, via gen_slice_into, then immediately read back out
        into addr_dst/len_dst/cap_dst), a Call to a function that
        itself returns a slice (materialized into that exact same
        scratch slot, via gen_slice_call_into, then read back out the
        same way -- it used to arrive already sitting in %rax/%rdx by
        a dedicated two-register return convention, needing no scratch
        slot at all, back when a slice's own descriptor still fit two
        registers; see the module docstring's SLICE PARAMETERS AND
        RETURNS section for why that's no longer true), or an Index
        yielding a slice (`rows[0][1]`, one element of an array OF
        slices used directly as a further base -- materialized into
        that same scratch slot too, via gen_slice_value_into's own
        Index case).

        An ARRAY-typed `expr` can ALSO be an ArrayLiteral directly --
        not an existing Variable/Index at all, but a freshly-created
        one -- for a slice LITERAL's own backing array (`[]int[1, 2,
        3]`, parsed as an implicit whole-array Slice wrapping an
        ArrayLiteral -- see parser.py's own _parse_bracketed_literal).
        See gen_array_literal_heap_alloc_into's own docstring for why
        this is a genuinely different kind of "address" than the
        ordinary Variable/Index cases below: it mallocs a brand new
        allocation and writes the literal's own elements into it,
        rather than computing the address of something that already
        exists.

        Reusing ONE shared scratch slot for every Slice materialization
        -- rather than a fresh one per nesting level -- is safe under
        arbitrarily deep chaining (`arr[:][0:2][0]`, `rows[0][1]`, and
        so on) specifically because of how gen_slice_into and gen_
        index_address_into are both already structured: each computes
        its OWN base's address/length FIRST, immediately consumes it
        (into addr_dst/len_dst, then protects those on the real CPU
        stack before evaluating anything else), and only ever WRITES
        its own result into a destination as the very LAST step. That
        means a deeper level's own write to the shared slot always
        happens (and is always fully drained back into registers)
        strictly BEFORE the shallower level that triggered it writes
        its own result there -- the same strictly-nested lifetime
        discipline that makes reusing one call stack safe for
        recursion of any depth, just applied to one scratch memory
        slot instead of the stack. An Index base (e.g. `rows[0][1]`,
        indexing into an array/slice OF slices) reuses this exact same
        scratch slot too, via gen_slice_value_into's own Index case --
        neither is an array-returning Call as a SLICE base, since it
        can never actually BE slice-typed.

        `expr` being anything else (a Call returning something other
        than an array/slice can't reach here at all, being neither
        array- nor slice-typed) falls through to the final
        CodegenError below.
        """
        base_type = self._type_of(expr)
        if base_type.kind == TypeKind.ARRAY:
            if isinstance(expr, ArrayLiteral):
                instructions = self.gen_array_literal_heap_alloc_into(expr)
                instructions.append(MovQ(src=Register('rax'), dst=addr_dst))
                return instructions, Imm(base_type.size), Imm(base_type.size)
            instructions = self.gen_array_address_into(expr, addr_dst)
            return instructions, Imm(base_type.size), Imm(base_type.size)
        if base_type.kind == TypeKind.SLICE:
            if isinstance(expr, Variable):
                offset = self._local_offset(expr.name)
                instructions = [
                    MovQ(src=Memory('rbp', offset + 8), dst=len_dst),
                    MovQ(src=Memory('rbp', offset + 16), dst=cap_dst),
                    MovQ(src=Memory('rbp', offset), dst=addr_dst),
                ]
                return instructions, len_dst, cap_dst
            if isinstance(expr, Slice):
                temp = self._unnamed_slice_temp_offset
                instructions = self.gen_slice_into(expr, Memory('rbp', temp))
                instructions.append(MovQ(src=Memory('rbp', temp + 8), dst=len_dst))
                instructions.append(MovQ(src=Memory('rbp', temp + 16), dst=cap_dst))
                instructions.append(MovQ(src=Memory('rbp', temp), dst=addr_dst))
                return instructions, len_dst, cap_dst
            if isinstance(expr, Index):
                # A slice-typed Index result (e.g. `rows[0]`, one
                # element of an array OF slices, used directly as the
                # base of a further `[...]`) -- materialized through
                # the exact same scratch slot the Slice case just
                # above uses, via gen_slice_value_into's own Index
                # case, then immediately read back out the same way.
                temp = self._unnamed_slice_temp_offset
                instructions = self.gen_slice_value_into(expr, Memory('rbp', temp))
                instructions.append(MovQ(src=Memory('rbp', temp + 8), dst=len_dst))
                instructions.append(MovQ(src=Memory('rbp', temp + 16), dst=cap_dst))
                instructions.append(MovQ(src=Memory('rbp', temp), dst=addr_dst))
                return instructions, len_dst, cap_dst
            if isinstance(expr, Call):
                # A slice-returning Call now writes through the hidden-
                # pointer convention (see gen_slice_call_into), just
                # like an array-returning one -- materialized through
                # the exact same shared scratch slot the Slice/Index
                # cases just above use, then immediately read back out
                # the same way. Used to leave its result directly in
                # %rax/%rdx instead, back when a slice's own descriptor
                # still fit two registers.
                temp = self._unnamed_slice_temp_offset
                instructions = self.gen_slice_call_into(Memory('rbp', temp), expr)
                instructions.append(MovQ(src=Memory('rbp', temp + 8), dst=len_dst))
                instructions.append(MovQ(src=Memory('rbp', temp + 16), dst=cap_dst))
                instructions.append(MovQ(src=Memory('rbp', temp), dst=addr_dst))
                return instructions, len_dst, cap_dst
            raise CodegenError(
                f"Cannot use a {type(expr).__name__} directly as the "
                f"base of an index or slice expression when it's "
                f"slice-typed -- assign it to a named variable first"
            )
        raise CodegenError(f"Cannot index or slice a value of type {base_type}")


    def gen_array_address_into(self, expr: Node, dst: Register) -> List[Instruction]:
        """Computes the ADDRESS of an array-typed expression -- a
        Variable referring to an array-typed local, or an Index node
        that itself resolves to a sub-array (the outer dimensions of a
        multi-dimensional access) -- into the 64-bit register `dst`.
        `dst` must already be a 64-bit register (e.g. Register('rax'),
        not Register('eax')) -- addresses are always 64-bit values,
        regardless of how wide the array's own elements are.

        A heap-allocated Variable (see is_heap_allocated) needs a
        genuinely different instruction here, not just a different
        offset: its own slot holds a POINTER to the array's actual
        data, not the data itself, so getting the array's address
        means LOADING that pointer (movq) rather than computing the
        slot's own address (leaq) the way a stack-allocated array's
        does. Every other array-address computation in this file --
        gen_index_address_into's own recursive base case, and
        everything that calls through it (gen_index_assign, an Index
        read in gen_expr_into, gen_array_arg_address_into) -- goes
        through this one method for a bare Variable, so this is the
        only place that distinction needs to be made at all.
        """
        if isinstance(expr, Variable):
            offset = self._local_offset(expr.name)
            array_type = self._local_type(expr.name)
            if self._is_array_heap_allocated(self._local_decl_id(expr.name), array_type):
                return [MovQ(src=Memory('rbp', offset), dst=dst)]
            return [LeaQFrame(offset=offset, dst=dst)]
        if isinstance(expr, Index):
            return self.gen_index_address_into(expr, dst)
        raise CodegenError(f"Cannot compute an array address for: {expr!r}")

    def gen_index_address_into(self, expr: Index, dst: Register) -> List[Instruction]:
        """Computes the address of `expr.array[expr.index]` into `dst`
        (a 64-bit register) -- the shared foundation for reading an
        element (gen_expr_into's Index case), writing one
        (gen_index_assign), and reading a whole SUB-array for
        multi-dimensional access (this method's own recursive base
        case, via gen_array_address_into above, when `expr.array` is
        itself an Index).

        `expr.array` can now be array- OR slice-typed (indexing into a
        slice, `s[i]`, uses this exact same method) -- see
        gen_indexable_base_into for how the base's address and length
        are computed either way. For an array base, the length is an
        Imm known at compile time; for a slice base, it's a runtime
        value read out of the slice's own descriptor, kept alive in
        `len_reg` (picked dynamically, distinct from `dst`, the same
        way gen_array_copy's own scratch register is) across
        evaluating the index expression.

        Includes a runtime bounds check: an out-of-range index prints a
        message and calls abort() (see _gen_bounds_check_panic_block)
        rather than silently reading or writing adjacent stack memory
        -- which, given arrays live in the same frame as the saved
        return address and the callee-saved registers every function
        call already depends on, could otherwise corrupt exactly the
        state that keeps `call`/`ret` working correctly, not just
        return a wrong value.

        `expr.array`'s own address (and, for a slice base, its length
        too) is computed first and protected on the real CPU stack (not
        a fixed register) while the index expression -- which could be
        arbitrarily complex, including another indexing operation or a
        function call -- is evaluated, the same push-before-recursing
        pattern used everywhere else in this file a value needs to
        survive evaluating something else. This works out correctly no
        matter what register `dst` itself is (including if it happens
        to coincide with the %rax/%rcx this method uses internally): every
        value that needs to survive is protected on the stack, and the
        final address is only ever written into `dst` as the very last
        step.
        """
        array_type = self._type_of(expr.array)
        element_stride = type_byte_width(array_type.element_type)

        # len_reg only matters for a slice base (a runtime length);
        # picked dynamically, distinct from dst, since dst could in
        # principle be any register a caller passes. cap_reg is never
        # actually read afterward -- ordinary indexing never needs
        # capacity, only length -- but gen_indexable_base_into's own
        # contract always populates it, so this still needs a real,
        # distinct register to receive it into, picked the same way.
        len_reg = Register('rdx' if dst.name != 'rdx' else 'r10')
        cap_reg = next(Register(r) for r in ('rdx', 'r10', 'r11') if r not in (dst.name, len_reg.name))

        instructions, length_operand, _ = self.gen_indexable_base_into(expr.array, dst, len_reg, cap_reg)
        instructions.append(Push(dst))
        is_runtime_length = isinstance(length_operand, Register)
        if is_runtime_length:
            instructions.append(Push(len_reg))

        instructions.extend(self.gen_expr_into(expr.index, Register('eax')))

        # Unsigned comparison: catches index >= length AND index < 0 in
        # one check, since a negative int, reinterpreted unsigned,
        # becomes a huge positive number.
        if is_runtime_length:
            instructions.append(Pop(len_reg))
            len_reg_32 = Register({'rdx': 'edx', 'r10': 'r10d'}[len_reg.name])
            instructions.append(Cmp(src=len_reg_32, dst=Register('eax')))
        else:
            instructions.append(Cmp(src=length_operand, dst=Register('eax')))
        instructions.append(Jae(self._get_bounds_check_fail_label("array index out of bounds")))
        # A plain 32-bit imul is safe here: the bounds check above
        # already guarantees the index is small and non-negative, and
        # a 32-bit write zero-extends into the full 64-bit rax.
        instructions.append(IMul(src=Imm(element_stride), dst=Register('eax')))
        instructions.append(Pop(Register('rcx')))  # restore expr.array's base address
        instructions.append(AddQ(src=Register('rax'), dst=Register('rcx')))
        instructions.append(MovQ(src=Register('rcx'), dst=dst))
        return instructions

    def gen_slice_into(self, expr: Slice, dst_mem: Memory) -> List[Instruction]:
        """Generates `expr.array[expr.low:expr.high]`'s resulting
        {ptr, len, cap} descriptor directly into dst_mem (ptr at
        offset+0, len at offset+8, cap at offset+16) -- the slice
        counterpart to gen_array_value_into, dispatched from gen_slice_
        value_into wherever a slice-typed value needs to be produced.

        The base's own address and length are computed first (see
        gen_indexable_base_into -- an Imm for an array base, a
        runtime value read out of the descriptor for a slice base),
        then `low` and `high` are resolved -- each defaulting to 0 /
        the base's own length respectively when omitted (see Slice's
        own docstring in parser.py for why both stay None at parse
        time rather than one being defaulted earlier) -- and finally
        bounds-checked against each other and the base's length
        before the resulting ptr/len are computed and written.

        Every intermediate value (the base's address, its length,
        high, low) is protected on the real CPU stack across
        evaluating whichever of expr.low/expr.high are present --
        each of which could be an arbitrarily complex expression,
        including a function call -- rather than assumed to survive
        in whatever register initially held it. Pushed in a specific
        order (address, then length if runtime, then high, then low)
        and popped in exact reverse, so nothing ever needs to be read
        out of the middle of the stack -- except for one case:
        defaulting `high` to the base's own RUNTIME length (a slice
        base with no explicit high bound) reads it via a plain peek at
        the top of the stack (`(%rsp)`, no pop), since at that exact
        point nothing else has been pushed since the length was, and
        reading it without popping keeps it protected for the bounds
        check that comes later.

        Bounds checks use `ja` (strictly "above", unsigned), not `jae`
        -- unlike ordinary indexing (see gen_index_address_into),
        where an index equal to the array's own size is already out
        of bounds, `low` and `high` are both allowed to equal the
        base's own CAP (`arr[5:5]` on a 5-element array, or `s[5:5]`
        on a slice whose own cap is 5 even if its len is smaller, is a
        valid, empty-slice-producing expression) -- so the boundary
        itself genuinely differs here, not just the label it jumps to.

        Checked against CAP, not len -- `high` may reach all the way
        to the base's own remaining CAPACITY, matching Go's actual
        re-slicing rule, not just its current length. This is what
        lets a re-sliced view grow into room a PRIOR append (or the
        base's own construction) already reserved: cap is computed as
        base_cap - low below, inheriting the base's own remaining
        capacity from the new starting point, rather than simply
        matching the newly-computed len (high - low) the way it used
        to before cap-aware re-slicing existed -- see the module
        docstring's APPEND BUILTIN section for why this and `append`
        itself landed together rather than as two separate,
        sequential pieces of work: with every other slice-producing
        site setting cap equal to len, there was no way to observe or
        test a genuinely cap-aware re-slice until append existed to
        first produce a slice whose cap differs from its len at all.

        dst_mem.base is protected on the stack too, whenever it isn't
        'rbp', across ALL of the above -- pushed before even
        gen_indexable_base_into runs (the earliest point it could be
        clobbered: an ArrayLiteral base mallocs, via
        gen_array_literal_heap_alloc_into, and any evaluated low/high
        expression always targets %eax/%rax like every other
        expression in this file does) and popped back right before the
        final two writes, nested as the OUTERMOST push/pop pair around
        this method's own existing stack discipline -- everything else
        this method already pushes and pops happens strictly between
        the two, so nothing about their own relative ordering changes.
        Needed once a slice-typed value can be produced somewhere other
        than an ordinary local slot: an array literal whose OWN
        elements are themselves slices (`[][]int rows = [][]int[[1,
        2], [3, 4]]`) writes each element by calling this method with
        dst_mem.base equal to the OUTER array's own base -- 'rbp' if
        it's stack-allocated, or 'rax' if it's heap-allocated (every
        slice literal's own backing array always is -- see gen_array_
        literal_heap_alloc_into). Found necessary by the same class of
        real bug this file has hit before in this exact area (see
        gen_array_literal_into's own docstring), not assumed
        defensively -- this method used to just assert dst_mem.base ==
        'rbp' and refuse anything else, rather than risk it silently.
        """
        protect_dst = dst_mem.base != 'rbp'
        instructions = []
        if protect_dst:
            instructions.append(Push(Register(dst_mem.base)))

        base_type = self._type_of(expr.array)
        element_stride = type_byte_width(base_type.element_type)

        addr_reg = Register('rbx')
        len_reg = Register('r11')
        cap_reg = Register('r14')
        base_instructions, length_operand, cap_operand = self.gen_indexable_base_into(expr.array, addr_reg, len_reg, cap_reg)
        instructions.extend(base_instructions)
        is_runtime_length = isinstance(length_operand, Register)

        instructions.append(Push(addr_reg))
        if is_runtime_length:
            # Pushed cap BEFORE len (the reverse of the field order in
            # the descriptor itself) specifically so len ends up on
            # TOP of the stack -- preserving, unchanged, the existing
            # "peek at (%rsp) for len's own default" logic just below,
            # which predates cap existing at all.
            instructions.append(Push(cap_reg))
            instructions.append(Push(len_reg))

        # Resolve `high` before `low`, so that defaulting it (when the
        # base's length is a runtime value) can safely peek the top of
        # the stack -- nothing else has been pushed since the length
        # was, right above. high still defaults to the base's own LEN
        # here, not its cap -- `arr[3:]` means "from 3 to the current
        # end", exactly as before; only the UPPER BOUND high is
        # allowed to reach when explicitly given (checked below) has
        # changed, not what an omitted one defaults to.
        if expr.high is not None:
            instructions.extend(self.gen_expr_into(expr.high, Register('eax')))
        elif is_runtime_length:
            instructions.append(Mov(src=Memory('rsp', 0), dst=Register('eax')))
        else:
            instructions.append(Mov(src=Imm(base_type.size), dst=Register('eax')))
        instructions.append(Push(Register('rax')))

        if expr.low is not None:
            instructions.extend(self.gen_expr_into(expr.low, Register('eax')))
        else:
            instructions.append(Mov(src=Imm(0), dst=Register('eax')))
        instructions.append(Push(Register('rax')))

        low_reg = Register('r10')
        high_reg = Register('r9')
        low_32 = Register('r10d')
        high_32 = Register('r9d')

        instructions.append(Pop(low_reg))
        instructions.append(Pop(high_reg))
        if is_runtime_length:
            instructions.append(Pop(len_reg))
            instructions.append(Pop(cap_reg))
        instructions.append(Pop(addr_reg))

        # Bounds check: 0 <= low <= high <= CAP -- not len. This is
        # the real, deliberate change from before cap existed: Go's
        # own re-slicing rule allows high to reach all the way to the
        # base's remaining CAPACITY, not just its current length,
        # which is exactly what lets a re-slice grow into room a
        # PRIOR append (or the base's own construction) already
        # reserved. `low` is bounded by cap too, for the same reason
        # (low can equal cap, producing a valid, empty, zero-capacity
        # slice at the very end -- the base case a chain of further
        # re-slices or appends would still handle correctly).
        fail_label = self._get_bounds_check_fail_label("slice bounds out of range")
        cap_op = Register('r14d') if is_runtime_length else cap_operand
        instructions.append(Cmp(src=cap_op, dst=low_32))
        instructions.append(Ja(fail_label))
        instructions.append(Cmp(src=cap_op, dst=high_32))
        instructions.append(Ja(fail_label))
        instructions.append(Cmp(src=high_32, dst=low_32))
        instructions.append(Ja(fail_label))

        # new_cap = cap - low -- the base's own remaining capacity
        # from the new starting point, matching Go's actual re-slicing
        # rule (see this method's own docstring for the growth-policy
        # motivation: this is what lets `append` sometimes grow a
        # re-sliced view into its parent's own backing array instead
        # of always allocating fresh). Computed into its own register,
        # BEFORE low is scaled below, for the same reason new_len
        # (high - low, just after) already has to be: scaling would
        # destroy the unscaled value both of these still need. Mov
        # (not MovQ) here since cap_op may be a 32-bit Imm (an array
        # base) as easily as a 32-bit register (a slice base) -- both
        # are valid Mov sources into a 32-bit destination alike.
        new_cap_32 = Register('r13d')
        instructions.append(Mov(src=cap_op, dst=new_cap_32))
        instructions.append(Sub(src=low_32, dst=new_cap_32))

        # len = high - low, computed BEFORE low is scaled below --
        # scaling would destroy the unscaled value this still needs.
        instructions.append(Sub(src=low_32, dst=high_32))
        # ptr = addr + low * element_stride. A plain 32-bit imul is
        # safe here: the bounds check above already guarantees low is
        # small and non-negative, and a 32-bit write zero-extends into
        # the full 64-bit low_reg.
        instructions.append(IMul(src=Imm(element_stride), dst=low_32))
        instructions.append(AddQ(src=low_reg, dst=addr_reg))

        # dst_mem.base is only needed again now, for these final
        # writes -- restored here, after every other computation above
        # (including the bounds check, which never falls through to
        # here on failure at all -- it aborts) has already finished.
        if protect_dst:
            instructions.append(Pop(Register(dst_mem.base)))
        instructions.append(MovQ(src=addr_reg, dst=Memory(dst_mem.base, dst_mem.offset)))
        instructions.append(MovQ(src=high_reg, dst=Memory(dst_mem.base, dst_mem.offset + 8)))
        instructions.append(MovQ(src=Register('r13'), dst=Memory(dst_mem.base, dst_mem.offset + 16)))
        return instructions

    def gen_none_into(self, dst_mem: Memory, target_type: Type) -> List[Instruction]:
        """Writes `none`'s own zero-value representation into dst_mem,
        for whichever nilable type target_type actually is. Only
        slices are nilable so far (see NoneLiteral's own docstring in
        parser.py) -- a {ptr: 0, len: 0, cap: 0} descriptor, the same
        shape Go's own nil slice has: a valid, safely-indexable-into-
        nothing slice with no backing array, not a special, separately-
        tracked null flag. Every existing slice operation (indexing,
        printing, re-slicing) already handles a zero-length slice
        correctly -- see TestSliceBoundsChecking's own positive
        control for `arr[5:5]` -- so this is the ONLY new codegen a
        none-valued slice needs on the producing side; comparing one
        against `none` again (see gen_slice_none_comparison_into) is
        the only other.

        Called directly from gen_var_decl/gen_assign's own NoneLiteral
        short-circuit, rather than being folded into
        gen_slice_value_into's own dispatch -- unlike every OTHER kind
        of slice-producing expression there (a Slice expression, a
        Variable holding one), a NoneLiteral's own resolved type
        (Type.NONE) never equals the slice type it's being stored
        into, so the caller has to already know and pass the TARGET
        type; gen_slice_value_into's whole existing dispatch, by
        contrast, only ever needs the expression itself, since every
        other case's own type already matches what needs to be stored.

        Defensively re-checks target_type.kind here even though
        semantic.py's own _types_compatible already guarantees `none`
        was only ever allowed through for a slice target -- the same
        "codegen doesn't blindly trust its input" posture
        gen_array_copy's own array-of-slices handling already takes.
        """
        if target_type.kind != TypeKind.SLICE:
            raise CodegenError(
                f"'none' is only supported as a slice's zero value "
                f"right now, not {target_type}"
            )
        return [
            MovQ(src=Imm(0), dst=Memory(dst_mem.base, dst_mem.offset)),
            MovQ(src=Imm(0), dst=Memory(dst_mem.base, dst_mem.offset + 8)),
            MovQ(src=Imm(0), dst=Memory(dst_mem.base, dst_mem.offset + 16)),
        ]


    def gen_slice_value_into(self, expr: Node, dst_mem: Memory) -> List[Instruction]:
        """Stores a slice-typed expression's VALUE (its {ptr, len,
        cap} descriptor) into dst_mem -- an arbitrary Memory operand,
        not just an ordinary local slot: dst_mem.base is 'rax'
        whenever this is writing a SLICE-typed element into an array
        literal whose OWN backing storage is heap-allocated (which
        every slice literal's own backing array always is -- see
        gen_array_literal_heap_alloc_into), the case that first made
        every one of the cases below need real protection rather than
        assuming 'rbp'. Dispatched on what kind of expression is
        producing the value:
          - Slice (e.g. `arr[1:3]`, or a slice LITERAL, `[]int[1, 2,
            3]`, parsed as one -- see parser.py's own _parse_
            bracketed_literal): computed directly, protecting
            dst_mem.base internally across its own, considerably more
            involved computation (see gen_slice_into's own docstring).
          - Variable (e.g. `s2 = s1`): a flat 24-byte copy of an
            existing slice's own descriptor. Deliberately NOT routed
            through gen_array_copy, even though that method's own
            flat-copy loop could technically move 24 bytes just as
            well as any other width: gen_array_copy's own handling of
            a slice LEAF type is specifically about an ARRAY whose
            ELEMENTS are slices, not about copying a bare slice
            descriptor itself, which is exactly what this case is --
            and a slice's descriptor is always exactly 24 bytes
            regardless of element type, so a fixed three-field copy
            (no loop needed at all) is both simpler and avoids that
            mismatch entirely. Uses %r8/%r9/%r10 as scratch, not %rax
            -- dst_mem.base is never %r8, %r9, or %r10 anywhere in
            this file (its only two established values are 'rbp' and
            'rax'), so this case needs no push/pop protection at all,
            unlike the others: an EARLIER version of this case used
            %rax as scratch for both (then just two) fields, which was
            silently wrong whenever dst_mem.base happened to BE 'rax'
            -- a later field's write would have used the just-loaded
            VALUE as the base address instead of the real one, since
            %rax no longer held it by then.
          - Call (e.g. `[]int s = otherFn()`, where otherFn also
            returns a slice): calls through the hidden-output-pointer
            convention, writing directly into dst_mem -- see gen_
            slice_call_into. Structurally identical to the ARRAY
            counterpart of this same case (gen_array_value_into's own
            Call case), now that slice returns use the exact same
            mechanism arrays already did.
          - Index (e.g. `[]int r = rows[0]`, reading one slice-typed
            element out of an array OF slices): the element's own
            address is computed first (gen_index_address_into), then
            its 24-byte descriptor is read through it -- structurally
            the same flat copy the Variable case does, just from a
            computed address rather than a fixed local offset.
          - ArrayLiteral (e.g. `[]int s = [1, 2, 3]`, an UNTYPED
            literal flowing directly into a slice-typed target -- see
            semantic.py's _check_value_flowing_into and check_array_
            literal's own expected_element_type parameter for how this
            gets recognized during type-checking; note the general,
            TYPED form, `[]int[1, 2, 3]`, never reaches this case at
            all, since it parses as a Slice wrapping an ArrayLiteral,
            not a bare one -- see the Slice case above): mallocs a
            fresh backing array and writes the literal's own elements
            into it (gen_array_literal_heap_alloc_into), exactly like
            gen_indexable_base_into's own, separate ArrayLiteral case
            does for the general, typed form -- both ultimately need
            identical work, just reached from different call sites.
            cap is set equal to len here -- a fresh literal's backing
            array is sized to exactly fit its own elements, with no
            spare room to grow into yet.

        Every case that does real work between "start" and "write the
        result into dst_mem" -- every one except Variable and Call, per
        their own notes above -- protects dst_mem.base on the stack
        across that work whenever it isn't 'rbp', computing the result
        into scratch registers (or gen_slice_into's own internal ones)
        first and restoring dst_mem.base only immediately before the
        final writes use it. This generalizes what gen_array_literal_
        into's own scalar-element case already established for a
        single value; see its docstring for the real bug (not a
        hypothetical one) that made it necessary there -- the exact
        same class of bug applies here, just for a wider value produced
        across more instructions instead of a 4-or-8-byte one produced
        by a single gen_expr_into call.

        NoneLiteral is NOT handled here -- its own resolved type
        (Type.NONE) never matches SLICE, so it can't even reach this
        method through _gen_store's ordinary dispatch; see
        gen_none_into and gen_var_decl/gen_assign's own NoneLiteral
        short-circuit for why that's handled one level up instead.
        """
        protect_dst = dst_mem.base != 'rbp'

        if isinstance(expr, Slice):
            return self.gen_slice_into(expr, dst_mem)

        if isinstance(expr, Variable):
            src_offset = self._local_offset(expr.name)
            return [
                MovQ(src=Memory('rbp', src_offset), dst=Register('r8')),
                MovQ(src=Register('r8'), dst=Memory(dst_mem.base, dst_mem.offset)),
                MovQ(src=Memory('rbp', src_offset + 8), dst=Register('r9')),
                MovQ(src=Register('r9'), dst=Memory(dst_mem.base, dst_mem.offset + 8)),
                MovQ(src=Memory('rbp', src_offset + 16), dst=Register('r10')),
                MovQ(src=Register('r10'), dst=Memory(dst_mem.base, dst_mem.offset + 16)),
            ]

        if isinstance(expr, Call):
            if expr.name == 'append':
                return self.gen_append_call_into(expr, dst_mem)
            return self.gen_slice_call_into(dst_mem, expr)

        if isinstance(expr, Index):
            instructions = []
            if protect_dst:
                instructions.append(Push(Register(dst_mem.base)))
            addr_reg = Register('r11')
            instructions.extend(self.gen_index_address_into(expr, addr_reg))
            instructions.append(MovQ(src=Memory(addr_reg.name, 0), dst=Register('r8')))
            instructions.append(MovQ(src=Memory(addr_reg.name, 8), dst=Register('r9')))
            instructions.append(MovQ(src=Memory(addr_reg.name, 16), dst=Register('r10')))
            if protect_dst:
                instructions.append(Pop(Register(dst_mem.base)))
            instructions.append(MovQ(src=Register('r8'), dst=Memory(dst_mem.base, dst_mem.offset)))
            instructions.append(MovQ(src=Register('r9'), dst=Memory(dst_mem.base, dst_mem.offset + 8)))
            instructions.append(MovQ(src=Register('r10'), dst=Memory(dst_mem.base, dst_mem.offset + 16)))
            return instructions

        if isinstance(expr, ArrayLiteral):
            instructions = []
            if protect_dst:
                instructions.append(Push(Register(dst_mem.base)))
            instructions.extend(self.gen_array_literal_heap_alloc_into(expr))
            instructions.append(MovQ(src=Register('rax'), dst=Register('r8')))
            element_count = len(expr.elements)
            if protect_dst:
                instructions.append(Pop(Register(dst_mem.base)))
            instructions.append(MovQ(src=Register('r8'), dst=Memory(dst_mem.base, dst_mem.offset)))
            instructions.append(MovQ(src=Imm(element_count), dst=Memory(dst_mem.base, dst_mem.offset + 8)))
            instructions.append(MovQ(src=Imm(element_count), dst=Memory(dst_mem.base, dst_mem.offset + 16)))
            return instructions

        raise CodegenError(f"No codegen rule for a slice-typed value: {expr!r}")

    def _gen_write_value_at_address_into(self, value_expr: Node, element_type: Type, addr_reg: Register) -> List[Instruction]:
        """Writes value_expr (evaluated as element_type) into
        Memory(addr_reg, 0) -- shared by gen_append_call_into's own
        reuse and reallocate paths, both of which need to write the
        newly-appended element at a computed (not fixed-offset)
        address, with the element's own type possibly being scalar,
        array, or slice.

        For an ARRAY or SLICE element type, this just hands addr_reg
        straight to gen_array_value_into/gen_slice_value_into as an
        ordinary Memory destination -- both already protect an
        arbitrary base internally (see their own docstrings), so
        there's nothing extra to do here. For a scalar (int/bool/str),
        addr_reg is protected manually, matching gen_array_literal_
        into's own scalar-element pattern exactly: push addr_reg,
        compute the value (which could itself involve a function call
        that clobbers addr_reg, if value_expr is arbitrarily complex),
        stash the computed value in %r8/%r8d (a register distinct from
        addr_reg in every actual call site), pop addr_reg back, then
        write from %r8/%r8d -- never straight from %eax/%rax, which
        popping addr_reg back into would otherwise have to clobber."""
        if element_type.kind == TypeKind.SLICE:
            return self.gen_slice_value_into(value_expr, Memory(addr_reg.name, 0))
        if element_type.kind == TypeKind.ARRAY:
            return self.gen_array_value_into(value_expr, Memory(addr_reg.name, 0), element_type)
        instructions = [Push(addr_reg)]
        instructions.extend(self.gen_expr_into(value_expr, Register('eax')))
        if element_type == Type.STR:
            instructions.append(MovQ(src=Register('rax'), dst=Register('r8')))
            instructions.append(Pop(addr_reg))
            instructions.append(MovQ(src=Register('r8'), dst=Memory(addr_reg.name, 0)))
        else:
            instructions.append(Mov(src=Register('eax'), dst=Register('r8d')))
            instructions.append(Pop(addr_reg))
            instructions.append(Mov(src=Register('r8d'), dst=Memory(addr_reg.name, 0)))
        return instructions

    def gen_append_call_into(self, expr: Call, dst_mem: Memory) -> List[Instruction]:
        """`append(s, value)` -- Go-style: writes a NEW {ptr, len, cap}
        descriptor into dst_mem, never mutating s's own three fields
        (s keeps pointing at exactly what it always did, with exactly
        its own original len and cap) -- see the module docstring's
        APPEND BUILTIN section for the full growth-and-aliasing story
        this is built around.

        s (expr.args[0]) can be any slice-typed expression -- a bare
        Variable or `none` (materialized inline, no scratch slot
        needed), or anything else (a slice literal, a re-slice, an
        Index, a slice-returning Call, ...), which gets materialized
        into the shared per-function scratch slot (_unnamed_slice_
        temp_offset -- see gen_indexable_base_into's own Slice-base
        case for the same pattern already used there) via gen_slice_
        value_into, then read back out exactly like a Variable's own
        slot would be. This used to be restricted to just a Variable
        or NoneLiteral, on the theory that append exists specifically
        to feed a reassignment (`x = append(x, v)`) and a bare slice
        expression as its own first argument would be rare enough not
        to justify the extra materialization step -- lifted once that
        turned out to matter in practice (`append([]int[], 1)`, for
        instance, needs exactly this to build a slice from scratch in
        a single expression). The materialization itself needs no
        special handling for what it's protecting: gen_slice_value_
        into already protects an arbitrary destination base
        internally, and the scratch slot here is always 'rbp'-based,
        so there's nothing dst_mem-shaped for it to clobber.

        s's own three fields are loaded into CALLEE-SAVED registers
        (%rbx/%r12/%r13 for ptr/len/cap) -- not caller-saved ones --
        specifically because the REALLOCATE path below calls malloc,
        which (like any real, ABI-conforming function) is free to
        clobber any caller-saved register but is OBLIGATED to preserve
        callee-saved ones; the exact same guarantee gen_array_literal_
        heap_alloc_into and gen_function's own heap-allocated-parameter
        handling already rely on.

        The reuse-vs-reallocate decision is a single comparison: len
        >= cap means no spare room (the only way that can happen,
        given the invariant len <= cap always holds, is len == cap
        exactly), so `jae` -- not `jne` or `jg` -- is both correct and
        sufficient.

        REUSE PATH (len < cap): the new element is written directly
        into the EXISTING backing array, at ptr + len*element_width --
        s's own array, still fully intact and unaffected, since s's
        own len field is never touched. This is the observable
        aliasing this whole growth policy exists to make possible: a
        LATER append on some other slice that shares this same backing
        array (e.g. one produced by an earlier append that over-
        allocated) can see this write, and vice versa.

        REALLOCATE PATH (len == cap): new_cap is computed from cap
        alone (the growth policy -- see below), a fresh block of
        new_cap*element_width bytes is malloc'd, the existing len
        elements are copied into it via a genuine RUNTIME loop (len is
        a runtime value here, unlike every other array copy in this
        file, which always moves a compile-time-known number of
        bytes -- see the loop's own comments), and only then is the
        new element written into the new array. The OLD backing array
        is simply never freed, matching this compiler's existing
        no-`free`-anywhere memory model everywhere else.

        GROWTH POLICY: new_cap = cap*2 if cap < 256, else cap +
        cap//4, with a cap==0 floor of 1. This is the general max(len+1,
        doubled-or-quartered) formula simplified: reallocation only
        ever happens when len == cap exactly, so `needed` (len+1) is
        always cap+1, and doubling already exceeds cap+1 for any cap
        >= 1 (quartering trivially does too, for cap >= 256) -- the
        max only actually matters, and only ever resolves in needed's
        favor, at cap == 0, which is exactly the explicit floor case
        here. cap/4 is computed via a right shift by 2 (arithmetic,
        though cap's own non-negativity means a logical shift would
        give the identical result) rather than idiv -- idiv can't take
        an immediate divisor at all on x86, and a shift is simpler
        besides.

        Both paths write value into its final resting place via
        _gen_write_value_at_address_into, sharing one implementation
        for a scalar, array, or slice element type alike, and both
        protect dst_mem.base (whenever it isn't 'rbp') across their own
        entire computation the same way every other slice-producing
        case in this file does -- popped back only immediately before
        the final three-field write actually needs it.
        """
        slice_arg, value_arg = expr.args
        slice_type = self._type_of(slice_arg)
        element_type = slice_type.element_type
        element_width = type_byte_width(element_type)

        protect_dst = dst_mem.base != 'rbp'
        instructions = []
        if protect_dst:
            instructions.append(Push(Register(dst_mem.base)))

        r_ptr = Register('rbx')
        r_len = Register('r12')
        r_cap = Register('r13')
        r_len_32 = Register('r12d')
        r_cap_32 = Register('r13d')

        if isinstance(slice_arg, NoneLiteral):
            instructions.append(MovQ(src=Imm(0), dst=r_ptr))
            instructions.append(MovQ(src=Imm(0), dst=r_len))
            instructions.append(MovQ(src=Imm(0), dst=r_cap))
        elif isinstance(slice_arg, Variable):
            offset = self._local_offset(slice_arg.name)
            instructions.append(MovQ(src=Memory('rbp', offset), dst=r_ptr))
            instructions.append(MovQ(src=Memory('rbp', offset + 8), dst=r_len))
            instructions.append(MovQ(src=Memory('rbp', offset + 16), dst=r_cap))
        else:
            # Any other slice-typed expression (a slice literal, a
            # re-slice, an Index, a slice-returning Call, ...): build
            # its own {ptr, len, cap} descriptor into the shared,
            # per-function unnamed-slice scratch slot first, then read
            # it back out exactly like a Variable's own slot would be.
            # This scratch slot is always 'rbp'-based, so there's
            # nothing here for gen_slice_value_into's own dst_mem
            # protection to need to guard against beyond what it
            # already does internally.
            scratch = self._unnamed_slice_temp_offset
            instructions.extend(self.gen_slice_value_into(slice_arg, Memory('rbp', scratch)))
            instructions.append(MovQ(src=Memory('rbp', scratch), dst=r_ptr))
            instructions.append(MovQ(src=Memory('rbp', scratch + 8), dst=r_len))
            instructions.append(MovQ(src=Memory('rbp', scratch + 16), dst=r_cap))

        realloc_label = self.new_label("append_realloc")
        end_label = self.new_label("append_end")

        # len >= cap (equivalently, given len <= cap always, len ==
        # cap exactly) means no spare room -- must reallocate.
        instructions.append(Cmp(src=r_cap_32, dst=r_len_32))
        instructions.append(Jae(realloc_label))

        # REUSE PATH: len < cap. target = ptr + len*element_width.
        target_addr = Register('r10')
        instructions.append(MovQ(src=r_ptr, dst=target_addr))
        instructions.append(Mov(src=r_len_32, dst=Register('r11d')))
        instructions.append(IMul(src=Imm(element_width), dst=Register('r11d')))
        instructions.append(AddQ(src=Register('r11'), dst=target_addr))
        instructions.extend(self._gen_write_value_at_address_into(value_arg, element_type, target_addr))
        instructions.append(Add(src=Imm(1), dst=r_len_32))
        if protect_dst:
            instructions.append(Pop(Register(dst_mem.base)))
        instructions.append(MovQ(src=r_ptr, dst=Memory(dst_mem.base, dst_mem.offset)))
        instructions.append(MovQ(src=r_len, dst=Memory(dst_mem.base, dst_mem.offset + 8)))
        instructions.append(MovQ(src=r_cap, dst=Memory(dst_mem.base, dst_mem.offset + 16)))
        instructions.append(Jmp(end_label))

        # REALLOCATE PATH: len == cap. Compute new_cap from cap alone
        # (see this method's own docstring for the growth-policy
        # arithmetic), in place -- the old cap value is never needed
        # again once this decides new_cap, so overwriting r_cap_32
        # here is safe.
        instructions.append(Label(realloc_label))
        zero_label = self.new_label("append_cap_zero")
        quarter_label = self.new_label("append_cap_quarter")
        growth_done_label = self.new_label("append_growth_done")

        instructions.append(Cmp(src=Imm(0), dst=r_cap_32))
        instructions.append(Je(zero_label))
        instructions.append(Cmp(src=Imm(256), dst=r_cap_32))
        instructions.append(Jae(quarter_label))
        instructions.append(IMul(src=Imm(2), dst=r_cap_32))
        instructions.append(Jmp(growth_done_label))
        instructions.append(Label(zero_label))
        instructions.append(Mov(src=Imm(1), dst=r_cap_32))
        instructions.append(Jmp(growth_done_label))
        instructions.append(Label(quarter_label))
        instructions.append(Mov(src=r_cap_32, dst=Register('eax')))
        instructions.append(Mov(src=Imm(2), dst=Register('ecx')))
        instructions.append(ShiftRightArithmetic(dst=r_cap_32))
        instructions.append(Add(src=Register('eax'), dst=r_cap_32))
        instructions.append(Label(growth_done_label))
        # r_cap_32 (and, via the zero-extension a 32-bit write always
        # gives its own 64-bit register, r_cap itself) now holds
        # new_cap.

        instructions.append(Mov(src=r_cap_32, dst=Register('edi')))
        instructions.append(IMul(src=Imm(element_width), dst=Register('edi')))
        instructions.append(CallInstr('malloc'))
        r_new_ptr = Register('r14')
        instructions.append(MovQ(src=Register('rax'), dst=r_new_ptr))

        # Copy the existing len elements from the OLD array (r_ptr)
        # into the NEW one (r_new_ptr), element_width bytes each -- a
        # genuine RUNTIME loop, since len is a runtime value here,
        # unlike every other array copy in this file (gen_array_copy's
        # own flat-copy loop), which always moves a compile-time-known
        # total width. Each iteration reuses gen_array_copy anyway, for
        # exactly ONE element's worth of data (type_byte_width(element_
        # type) bytes, dispatching on leaf_type(element_type) for the
        # per-chunk width) -- that method's own logic already
        # generalizes correctly to a single, arbitrary-type value, not
        # just a whole array, so no separate "copy one value" helper
        # was needed just for this loop body.
        loop_start_label = self.new_label("append_copy_loop")
        loop_done_label = self.new_label("append_copy_done")
        i_32 = Register('r9d')
        instructions.append(Mov(src=Imm(0), dst=i_32))
        instructions.append(Label(loop_start_label))
        instructions.append(Cmp(src=r_len_32, dst=i_32))
        instructions.append(Jae(loop_done_label))
        instructions.append(Mov(src=i_32, dst=Register('r11d')))
        instructions.append(IMul(src=Imm(element_width), dst=Register('r11d')))
        instructions.append(MovQ(src=r_ptr, dst=Register('r10')))
        instructions.append(AddQ(src=Register('r11'), dst=Register('r10')))
        instructions.append(MovQ(src=r_new_ptr, dst=Register('r8')))
        instructions.append(AddQ(src=Register('r11'), dst=Register('r8')))
        instructions.extend(self.gen_array_copy(Memory('r8', 0), Memory('r10', 0), element_type))
        instructions.append(Add(src=Imm(1), dst=i_32))
        instructions.append(Jmp(loop_start_label))
        instructions.append(Label(loop_done_label))

        # Write the new element at new_ptr + len*element_width -- the
        # one slot the copy loop above deliberately left untouched.
        target_addr2 = Register('r10')
        instructions.append(MovQ(src=r_new_ptr, dst=target_addr2))
        instructions.append(Mov(src=r_len_32, dst=Register('r11d')))
        instructions.append(IMul(src=Imm(element_width), dst=Register('r11d')))
        instructions.append(AddQ(src=Register('r11'), dst=target_addr2))
        instructions.extend(self._gen_write_value_at_address_into(value_arg, element_type, target_addr2))

        instructions.append(Add(src=Imm(1), dst=r_len_32))
        if protect_dst:
            instructions.append(Pop(Register(dst_mem.base)))
        instructions.append(MovQ(src=r_new_ptr, dst=Memory(dst_mem.base, dst_mem.offset)))
        instructions.append(MovQ(src=r_len, dst=Memory(dst_mem.base, dst_mem.offset + 8)))
        instructions.append(MovQ(src=r_cap, dst=Memory(dst_mem.base, dst_mem.offset + 16)))

        instructions.append(Label(end_label))
        return instructions

    def gen_array_copy(self, dst_mem: Memory, src_mem: Memory, array_type: Type) -> List[Instruction]:
        """Copies array_type's worth of data from src_mem to dst_mem --
        both arbitrary Memory operands (e.g. Memory('rbp', -24) for a
        fixed local's own slot, or Memory('rbx', 0) for a computed
        address held in %rbx) -- via a flat sequence of movl/movq
        instructions, one per leaf-typed element. A multi-dimensional
        array is just one contiguous block of leaf values in row-major
        order for copying purposes, so no per-dimension logic is
        needed here at all, just the total byte width and the leaf
        element's own width (see type_byte_width/leaf_type).

        An array whose ELEMENTS are themselves slices (`[3][]int`) is
        handled the exact same way as any other leaf type, just with a
        24-byte width (three sequential 8-byte movqs -- the pointer,
        then the length, then the cap -- through the same scratch
        register, rather than the single movl/movq every other leaf
        width uses): this is a SHALLOW copy of each element's own
        {ptr, len, cap} descriptor, matching how copying an ordinary,
        bare slice variable (`s2 = s1`, see gen_slice_value_into's own
        Variable case) already works -- the copy's own slice elements
        end up pointing at the exact same backing data the original's
        do, not independently, recursively re-allocated ones. This is
        deliberate, not a shortcut: it's the array counterpart of the
        very same, already-established slice value semantics, not a
        new rule invented for this case.

        The scratch register shuttling each element's value between
        src and dst is picked dynamically to differ from BOTH src_mem's
        and dst_mem's own base register -- otherwise loading a value
        into it would destroy the very address a later iteration still
        needs to read from or write to. Found as a real bug during
        development, not a hypothetical one: gen_return passes
        Memory('rax', 0) as the destination when writing an array
        directly through a received hidden return pointer, and
        unconditionally using %eax/%rax as scratch (the very reasonable
        choice everywhere else in this file, since gen_expr_into always
        targets it) destroyed that address the moment the first
        element's value was loaded, before it could even be written
        anywhere. rcx and rdx are never used as a Memory base anywhere
        else in this file, so picking whichever of rax/rcx/rdx isn't
        already one of the two bases here stays correct even if that
        ever changes -- unaffected by the 24-byte case just above,
        which reuses this exact same scratch register for its own
        three, sequential 8-byte moves, rather than needing a second
        one."""
        leaf = leaf_type(array_type)
        used_bases = {src_mem.base, dst_mem.base}
        scratch_64, scratch_32 = next(
            (r64, r32) for r64, r32 in [('rax', 'eax'), ('rcx', 'ecx'), ('rdx', 'edx')]
            if r64 not in used_bases
        )
        width = type_byte_width(leaf)
        total = type_byte_width(array_type)
        instructions = []
        off = 0
        while off < total:
            src = Memory(src_mem.base, src_mem.offset + off)
            dst = Memory(dst_mem.base, dst_mem.offset + off)
            if width == 24:
                for field_offset in (0, 8, 16):
                    field_src = Memory(src_mem.base, src_mem.offset + off + field_offset)
                    field_dst = Memory(dst_mem.base, dst_mem.offset + off + field_offset)
                    instructions.append(MovQ(src=field_src, dst=Register(scratch_64)))
                    instructions.append(MovQ(src=Register(scratch_64), dst=field_dst))
            elif width == 8:
                instructions.append(MovQ(src=src, dst=Register(scratch_64)))
                instructions.append(MovQ(src=Register(scratch_64), dst=dst))
            else:
                instructions.append(Mov(src=src, dst=Register(scratch_32)))
                instructions.append(Mov(src=Register(scratch_32), dst=dst))
            off += width
        return instructions

    def _gen_address_of_memory_into(self, mem: Memory, dst: Register) -> List[Instruction]:
        """Computes the ADDRESS a Memory operand refers to, into `dst`
        (a 64-bit register). Memory('rbp', offset) needs a real leaq --
        the address is offset-from-frame-pointer, not stored anywhere
        as a value in its own right; Memory(some_reg, 0) already IS an
        address, sitting directly in some_reg (see gen_array_copy's own
        docstring for how that shape arises elsewhere in this file), so
        this just copies it. Used specifically for passing a Memory
        destination on as a POINTER argument -- the hidden output
        pointer for an array-returning call (gen_array_call_into) or an
        array-typed argument's own address (gen_array_arg_address_into)
        -- everywhere else, a Memory operand is read from or written to
        directly rather than having its own address taken."""
        if mem.base == 'rbp':
            return [LeaQFrame(offset=mem.offset, dst=dst)]
        return [MovQ(src=Register(mem.base), dst=dst)]

    def gen_array_arg_address_into(self, expr: Node, dst: Register) -> List[Instruction]:
        """Computes the address to pass for an array-typed function-call
        argument, into the 64-bit register `dst`. Only a Variable or an
        Index yielding a sub-array is supported -- both already have a
        real, existing address (see gen_array_address_into) -- an
        ArrayLiteral or a call returning an array used DIRECTLY as an
        argument (e.g. `foo([1,2,3])` or `foo(bar())`) has no home of
        its own to point at, and isn't supported: assign it to a named
        variable first (`[3]int t = [1,2,3]; foo(t)`), which already
        works today.

        The callee copies from this address into its own local slot on
        entry (see gen_function's parameter loop) -- so what's passed
        here only needs to stay valid for the duration of that one
        copy, not any longer, and the caller's own array is never
        itself mutated through it: the callee's copy is independent,
        preserving value semantics across the call the same way an
        ordinary `arr2 = arr1` does within a single function (see the
        module docstring's ARRAYS section)."""
        if isinstance(expr, (Variable, Index)):
            return self.gen_array_address_into(expr, dst)
        raise CodegenError(
            f"Array-typed call arguments must be a variable or an "
            f"indexing expression, not {type(expr).__name__} -- assign "
            f"it to a variable first"
        )

    def _total_arg_slots(self, args: List[Node]) -> int:
        """How many argument registers `args` will collectively need
        -- 3 for each slice-typed (or `none`) argument, 1 for
        everything else. `none`'s own resolved type (Type.NONE) never
        equals SLICE, so it's checked for separately here -- the same
        "check isinstance(expr, NoneLiteral) directly rather than
        relying on its own resolved type" pattern gen_var_decl/
        gen_assign/gen_return already use, safe for the same reason:
        semantic.py's _types_compatible already guarantees `none` is
        only ever valid where a slice is expected, since slices are
        the only nilable type that exists. Shared by every call-
        codegen entry point's own "too many arguments" check."""
        return sum(
            3 if self._type_of(a).kind == TypeKind.SLICE or isinstance(a, NoneLiteral) else 1
            for a in args
        )

    def gen_slice_arg_into(self, expr: Node, ptr_dst: Register, len_dst: Register, cap_dst: Register) -> List[Instruction]:
        """Computes a slice-typed call ARGUMENT's own ptr/len/cap
        directly into ptr_dst/len_dst/cap_dst. Restricted to a
        Variable (a named slice) or NoneLiteral (`none`) -- the same
        restriction slice bases have everywhere else in this file (see
        gen_indexable_base_into's own docstring): a bare Slice
        expression (`foo(arr[1:3])`) has no pre-existing descriptor to
        read, and would need the same temporary-materialization
        mechanism this codebase has consistently deferred elsewhere --
        assign it to a named variable first."""
        if isinstance(expr, NoneLiteral):
            return [
                MovQ(src=Imm(0), dst=ptr_dst),
                MovQ(src=Imm(0), dst=len_dst),
                MovQ(src=Imm(0), dst=cap_dst),
            ]
        if isinstance(expr, Variable):
            offset = self._local_offset(expr.name)
            return [
                MovQ(src=Memory('rbp', offset), dst=ptr_dst),
                MovQ(src=Memory('rbp', offset + 8), dst=len_dst),
                MovQ(src=Memory('rbp', offset + 16), dst=cap_dst),
            ]
        raise CodegenError(
            f"A slice-typed call argument must be a variable or "
            f"'none', not {type(expr).__name__} -- assign it to a "
            f"variable first"
        )

    def _gen_call_arguments_into(self, args: List[Node], reg_shift: int = 0) -> List[Instruction]:
        """Shared by gen_call_into, gen_array_call_into, and
        gen_slice_call_into: evaluates every argument -- in order,
        each via the ordinary gen_expr_into (a scalar), gen_array_arg_
        address_into (an array), or gen_slice_arg_into (a slice or
        none, which needs THREE consecutive register slots, not one)
        -- immediately pushing each one's resulting value(s) onto the
        stack before moving on to the next. Only *after* every
        argument has been safely computed and stacked does this start
        popping everything back off, in reverse, into the actual SysV
        argument registers (see _ARG_REGISTERS_64).

        This "compute and stack everything, then pop into place" order
        is what avoids the same register-clobbering hazard that
        motivated saving %rbx/%r12/%r13/%r14 across calls in the first
        place (see the module docstring's FUNCTIONS section): if
        argument 2 happens to be a nested call and argument 1's value
        were sitting in a scratch register instead of safely on the
        stack while argument 2 gets computed, argument 2's own use of
        that same scratch register would corrupt argument 1.

        `reg_shift` shifts every argument's own register slot later by
        that many positions -- 1 for a call to a function that returns
        an array or slice (whose hidden output pointer already
        occupies the first slot, pushed/popped separately by gen_
        array_call_into/gen_slice_call_into themselves, outside this
        method entirely), 0 otherwise.

        Because a slice argument needs three consecutive slots, the
        mapping from argument index to register index isn't the
        simple 1:1 one it used to be -- this tracks a running slot
        count instead, matching exactly how a real C compiler would
        place `struct{void*,long,long}` arguments among ordinary
        scalar ones: pushed in the same left-to-right order as written
        (a slice contributing its ptr, then its len, then its cap),
        and popped in exact reverse, so each slot lands in its own
        correct register regardless of whether it came from a whole
        scalar argument or a third of a slice one.
        """
        instructions: List[Instruction] = []
        arg_slot_counts = []
        for arg in args:
            arg_type = self._type_of(arg)
            if arg_type.kind == TypeKind.SLICE or isinstance(arg, NoneLiteral):
                instructions.extend(self.gen_slice_arg_into(arg, Register('rax'), Register('rdx'), Register('rcx')))
                instructions.append(Push(Register('rax')))
                instructions.append(Push(Register('rdx')))
                instructions.append(Push(Register('rcx')))
                arg_slot_counts.append(3)
            elif arg_type.kind == TypeKind.ARRAY:
                instructions.extend(self.gen_array_arg_address_into(arg, Register('rax')))
                instructions.append(Push(Register('rax')))
                arg_slot_counts.append(1)
            else:
                instructions.extend(self.gen_expr_into(arg, Register('eax')))
                instructions.append(Push(Register('rax')))
                arg_slot_counts.append(1)

        slot = sum(arg_slot_counts) - 1 + reg_shift
        for count in reversed(arg_slot_counts):
            if count == 3:
                instructions.append(Pop(Register(_ARG_REGISTERS_64[slot])))
                instructions.append(Pop(Register(_ARG_REGISTERS_64[slot - 1])))
                instructions.append(Pop(Register(_ARG_REGISTERS_64[slot - 2])))
            else:
                instructions.append(Pop(Register(_ARG_REGISTERS_64[slot])))
            slot -= count
        return instructions

    def gen_array_call_into(self, dst_mem: Memory, expr: Call, array_type: Type) -> List[Instruction]:
        """Calls a function that returns an array, writing its result
        directly into dst_mem via the hidden-pointer convention: the
        callee receives a pointer to where its result should go as an
        extra, FIRST argument (in %rdi), with every genuine argument
        shifted one register position later (see gen_function's own
        handling on the receiving side). The callee writes its return
        value directly through that pointer (see gen_return's array
        case) -- there's nothing for the CALLER to copy afterward,
        unlike an ordinary array-typed expression. This is also what
        makes forwarding one array-returning call's result straight out
        of another free (`return bar()`, where bar also returns an
        array): the SAME destination address just gets passed one
        level deeper, with no intermediate copy -- see gen_return's own
        docstring.

        dst_mem's own address is computed and pushed onto the stack
        FIRST, before any argument is evaluated, so it survives
        regardless of what an argument expression does internally (a
        nested call, string concatenation, another indexing operation
        -- anything that might otherwise clobber a register holding it)
        -- the same push-before-evaluating-something-else discipline
        used everywhere else in this file a value needs to survive past
        a sub-expression. Every other argument is handled by
        _gen_call_arguments_into (reg_shift=1, since the hidden pointer
        already occupies the first register slot) -- see its own
        docstring for how a slice-typed argument's own three slots are
        placed correctly among any ordinary scalar/array ones.
        """
        total_slots = 1 + self._total_arg_slots(expr.args)  # +1: the hidden pointer itself
        if total_slots > 6:
            raise CodegenError(
                f"Call to '{expr.name}' needs {total_slots} argument "
                f"register(s) (the hidden output pointer uses one, "
                f"and a slice-typed argument needs 3) -- this compiler "
                f"only supports up to 6"
            )
        instructions = self._gen_address_of_memory_into(dst_mem, Register('rax'))
        instructions.append(Push(Register('rax')))
        instructions.extend(self._gen_call_arguments_into(expr.args, reg_shift=1))
        instructions.append(Pop(Register('rdi')))
        instructions.append(CallInstr(expr.name))
        return instructions

    def gen_slice_call_into(self, dst_mem: Memory, expr: Call) -> List[Instruction]:
        """Calls a function that returns a slice, writing its result
        directly into dst_mem via the exact same hidden-pointer
        convention gen_array_call_into already uses -- see its own
        docstring for the full reasoning, unchanged here in every
        respect except that this writes 24 bytes (a slice's own {ptr,
        len, cap} descriptor -- see gen_return's own Slice case on the
        receiving side) rather than an array's own, type-dependent
        width. Slices used to return via a dedicated %rax:%rdx two-
        register convention instead; that stopped fitting once a
        slice's own descriptor grew a third field (cap), with no
        established three-register return shape to grow into -- so
        slice returns now share the exact same mechanism arrays
        already had, rather than inventing a new one. This is also
        what makes forwarding one slice-returning call's result
        straight out of another free (`return otherFn()`), exactly
        like it already was for arrays: the SAME destination address
        just gets passed one level deeper, with no intermediate copy.
        """
        total_slots = 1 + self._total_arg_slots(expr.args)  # +1: the hidden pointer itself
        if total_slots > 6:
            raise CodegenError(
                f"Call to '{expr.name}' needs {total_slots} argument "
                f"register(s) (the hidden output pointer uses one, "
                f"and a slice-typed argument needs 3) -- this compiler "
                f"only supports up to 6"
            )
        instructions = self._gen_address_of_memory_into(dst_mem, Register('rax'))
        instructions.append(Push(Register('rax')))
        instructions.extend(self._gen_call_arguments_into(expr.args, reg_shift=1))
        instructions.append(Pop(Register('rdi')))
        instructions.append(CallInstr(expr.name))
        return instructions

    def gen_array_literal_into(self, dst_mem: Memory, expr: ArrayLiteral, array_type: Type) -> List[Instruction]:
        """Stores an array literal's elements directly into consecutive
        memory locations starting at dst_mem -- almost always a fixed
        local slot (Memory('rbp', offset)), but see gen_array_value_into
        for why this takes a general Memory operand rather than a bare
        offset. Each element is evaluated via the ordinary
        gen_expr_into (so an element can be any expression, not just a
        constant), except when the element type is ITSELF an array (a
        multi-dimensional literal's "elements" are themselves
        ArrayLiterals, handled by recursing through gen_array_value_into,
        which dispatches straight back here) or a SLICE (an array whose
        elements are slices -- `[N][]int` -- e.g. the synthesized outer
        literal a slice-of-slices literal always is, `[][]int[[1, 2],
        [3, 4]]` -- handled by gen_slice_value_into, exactly like any
        other slice-producing expression; each element there might be
        an untyped ArrayLiteral needing a fresh backing allocation of
        its own, a named slice Variable, another Slice expression, or
        anything else that method already covers).

        dst_mem's own base register is protected on the stack across
        each element's value computation whenever it isn't 'rbp' --
        found necessary by a real bug during development, not assumed:
        'rbp' (the frame pointer, used for every ordinary local slot)
        is never clobbered by gen_expr_into, so no protection is needed
        there, but a computed or received address held in a general-
        purpose register (e.g. Memory('rax', 0), the hidden return
        pointer for a literal returned directly -- `return [1,2,3]`,
        or a slice literal's own freshly-mallocd backing array) is
        exactly the kind of register gen_expr_into's own value
        computation, which always targets %eax/%rax, can and did
        clobber -- silently overwriting the destination address before
        a single element was ever actually written through it. The
        SLICE case needs no protection of its own here, unlike the
        scalar case just below: gen_slice_value_into already protects
        dst_mem.base internally across whatever real work producing a
        slice value takes (see its own docstring), so by the time it
        returns, dst_mem.base is guaranteed correct again -- this loop
        can just call it directly and move on to the next element."""
        element_type = array_type.element_type
        element_width = type_byte_width(element_type)
        protect_dst = dst_mem.base != 'rbp'
        instructions = []
        for i, elem_expr in enumerate(expr.elements):
            elem_mem = Memory(dst_mem.base, dst_mem.offset + i * element_width)
            if element_type.kind == TypeKind.ARRAY:
                instructions.extend(self.gen_array_value_into(elem_expr, elem_mem, element_type))
                continue
            if element_type.kind == TypeKind.SLICE:
                instructions.extend(self.gen_slice_value_into(elem_expr, elem_mem))
                continue
            if protect_dst:
                instructions.append(Push(Register(dst_mem.base)))
            instructions.extend(self.gen_expr_into(elem_expr, Register('eax')))
            if element_type == Type.STR:
                if protect_dst:
                    instructions.append(MovQ(src=Register('rax'), dst=Register('r8')))
                    instructions.append(Pop(Register(dst_mem.base)))
                    instructions.append(MovQ(src=Register('r8'), dst=elem_mem))
                else:
                    instructions.append(MovQ(src=Register('rax'), dst=elem_mem))
            else:
                if protect_dst:
                    instructions.append(Mov(src=Register('eax'), dst=Register('r8d')))
                    instructions.append(Pop(Register(dst_mem.base)))
                    instructions.append(Mov(src=Register('r8d'), dst=elem_mem))
                else:
                    instructions.append(Mov(src=Register('eax'), dst=elem_mem))
        return instructions

    def _gen_protecting_dst_across(self, dst_mem: Memory, inner: List[Instruction]) -> List[Instruction]:
        """Wraps `inner` with a push/pop protecting dst_mem's own base
        register across it, but only when that base isn't 'rbp' -- the
        frame pointer, never clobbered by anything in this file, so
        wrapping would just be wasted instructions. Used wherever code
        that might use dst_mem.base as scratch internally (bounds-
        checking, evaluating an arbitrary expression, computing another
        address entirely) has to run before dst_mem is finally read
        from or written to -- e.g. gen_array_value_into's Index case
        below, where gen_array_address_into's own bounds-checking and
        index arithmetic freely uses %rax/%rcx, which would otherwise
        silently destroy a hidden return pointer received in %rax
        before it was ever used. Found necessary by a real bug during
        development (a segfault on `return matrix[i]` from a function
        returning an array), not assumed defensively."""
        if dst_mem.base == 'rbp':
            return inner
        return [Push(Register(dst_mem.base))] + inner + [Pop(Register(dst_mem.base))]

    def gen_array_value_into(self, expr: Node, dst_mem: Memory, array_type: Type) -> List[Instruction]:
        """Stores an array-typed expression's VALUE into dst_mem,
        matching array_type's shape. This is the array counterpart to
        _gen_store's scalar path -- an array can't fit into a single
        register the way an int/bool/str value can, so it needs its
        own dedicated "store" logic entirely, dispatched on what kind
        of expression is producing the value:
          - ArrayLiteral: each element stored directly (see
            gen_array_literal_into).
          - Variable: a copy from wherever the source's data actually
            lives (see gen_array_copy) into dst_mem. For a stack-
            allocated source, that's a flat offset-to-offset copy --
            no address computation needed at all, since the variable's
            own slot offset is already known at compile time. For a
            heap-allocated source (see is_heap_allocated), the slot
            holds a POINTER rather than the data itself, so that
            pointer is loaded first (protecting dst_mem's own base
            register across the load, via _gen_protecting_dst_across,
            in case they happen to coincide), and the copy reads
            through it instead. Either way, this is what makes
            `arr2 = arr1` a real, independent copy rather than a
            pointer alias -- see the module docstring's ARRAYS section
            on value semantics: heap-backed storage doesn't change
            that guarantee, only where the bytes being copied live.
          - Index (a sub-array, e.g. `[3]int row = matrix[i]`): its
            SOURCE address has to be computed first
            (gen_array_address_into), since it depends on a runtime
            index, then copied from that computed address. dst_mem is
            protected (see _gen_protecting_dst_across) across that
            computation, since it isn't just a simple move -- it
            includes bounds-checking and index arithmetic that freely
            uses %rax/%rcx internally.
          - Call (a function returning an array): calls through the
            hidden-output-pointer convention, writing directly into
            dst_mem -- see gen_array_call_into.

        dst_mem is a general Memory operand, not always a fixed local
        slot: it's Memory('rbp', offset) for an ordinary local variable
        or literal-initialized declaration, but Memory(some_reg, 0) when
        the destination is itself a computed or received address --
        e.g. gen_return uses this to write an array-typed return value
        directly through the hidden pointer it received, without ever
        materializing an intermediate local copy first. gen_array_copy
        and gen_array_literal_into both already work with an arbitrary
        Memory destination for exactly this reason -- nothing about the
        recursive structure here needed to change to support returns,
        only the type of the destination each caller happens to pass.
        """
        if isinstance(expr, ArrayLiteral):
            return self.gen_array_literal_into(dst_mem, expr, array_type)
        if isinstance(expr, Variable):
            src_offset = self._local_offset(expr.name)
            src_type = self._local_type(expr.name)
            if self._is_array_heap_allocated(self._local_decl_id(expr.name), src_type):
                load_ptr = self._gen_protecting_dst_across(
                    dst_mem, [MovQ(src=Memory('rbp', src_offset), dst=Register('rbx'))]
                )
                return load_ptr + self.gen_array_copy(dst_mem, Memory('rbx', 0), array_type)
            return self.gen_array_copy(dst_mem, Memory('rbp', src_offset), array_type)
        if isinstance(expr, Index):
            addr_instructions = self._gen_protecting_dst_across(
                dst_mem, self.gen_array_address_into(expr, Register('rbx'))
            )
            return addr_instructions + self.gen_array_copy(dst_mem, Memory('rbx', 0), array_type)
        if isinstance(expr, Call):
            return self.gen_array_call_into(dst_mem, expr, array_type)
        raise CodegenError(f"No codegen rule for an array-typed value: {expr!r}")

    def _get_bounds_check_fail_label(self, message: str) -> str:
        """Lazily creates a per-function, per-message label that every
        bounds check using this exact `message` jumps to on failure --
        reused across however many checks in this function share the
        same message, rather than duplicating the panic sequence (see
        _gen_bounds_check_panic_block) at every individual check site.
        A single function can use more than one message (e.g. "array
        index out of bounds" for ordinary indexing, "slice bounds out
        of range" for a slice expression's own low/high check) --
        each gets its own fail label, all reset together at the start
        of every function (see gen_function) -- unlike the message
        labels below, these are purely LOCAL jump targets, meaningless
        outside the function they're generated for."""
        if message not in self._bounds_check_fail_labels:
            self._bounds_check_fail_labels[message] = self.new_label("bounds_check_fail")
        return self._bounds_check_fail_labels[message]

    def _get_bounds_check_message_label(self, message: str) -> str:
        """Lazily creates and caches (for the rest of the WHOLE
        compilation, unlike the per-function fail labels above -- each
        is just a static string, safely shared by every function that
        needs it, matching the same lazy-cache pattern print's own
        format-string/true/false labels already use) a label for this
        exact `message` string."""
        if message not in self._bounds_check_message_labels:
            label = self.new_label("bounds_msg")
            self._bounds_check_message_labels[message] = label
            self.string_literals.append((label, message))
        return self._bounds_check_message_labels[message]

    def _gen_bounds_check_panic_block(self) -> List[Instruction]:
        """Appended once at the end of a function's own instructions
        (see gen_function) for every distinct message that function's
        own bounds checks actually used (see
        _get_bounds_check_fail_label) -- none at all if it never
        triggered any. Each block prints its own clear message, then
        calls abort() (SIGABRT) rather than a plain exit() -- an out-
        of-bounds access is a genuine program bug, not a normal
        termination condition, the same "abnormal termination"
        character division by zero's hardware-trapped SIGFPE already
        has, just deliberately raised by this compiler's own generated
        code instead of by the CPU. Never reached via ordinary fall-
        through from the function's own body -- every return already
        leaves via `leave; ret` before control could reach this point,
        and abort() itself never returns -- so appending these at the
        very end of the function is always safe.

        Explicitly calls fflush(NULL) between puts() and abort() --
        found necessary by testing, not assumed: abort() terminates
        the process via a raw signal, bypassing the normal exit() path
        that would otherwise flush libc's buffered stdio streams. Without
        this, the message is reliably printed when stdout happens to be
        line-buffered (an interactive terminal) but silently LOST
        whenever stdout is redirected or piped -- exactly the case for
        any program run non-interactively, which is most of them. A
        NULL argument tells fflush to flush every open output stream,
        so this doesn't need to reference libc's `stdout` symbol
        directly (a global variable, not a function -- meaningfully
        more awkward to reference correctly from hand-written assembly
        than another ordinary `call`).
        """
        instructions = []
        for message, fail_label in self._bounds_check_fail_labels.items():
            msg_label = self._get_bounds_check_message_label(message)
            instructions.extend([
                Label(fail_label),
                LeaQ(label=msg_label, dst=Register('rdi')),
                CallInstr('puts'),
                Mov(src=Imm(0), dst=Register('edi')),
                CallInstr('fflush'),
                CallInstr('abort'),
            ])
        return instructions

    def gen_statement(self, stmt: Node) -> List[Instruction]:
        if isinstance(stmt, VarDecl):
            return self.gen_var_decl(stmt)
        if isinstance(stmt, Assign):
            return self.gen_assign(stmt)
        if isinstance(stmt, IndexAssign):
            return self.gen_index_assign(stmt)
        if isinstance(stmt, Return):
            return self.gen_return(stmt)
        if isinstance(stmt, If):
            return self.gen_if(stmt)
        if isinstance(stmt, While):
            return self.gen_while(stmt)
        if isinstance(stmt, Break):
            return self.gen_break(stmt)
        if isinstance(stmt, Continue):
            return self.gen_continue(stmt)
        if isinstance(stmt, ExprStmt):
            return self.gen_expr_stmt(stmt)
        raise CodegenError(f"No codegen rule for statement: {stmt!r}")

    def _gen_malloc_array(self, array_type: Type) -> List[Instruction]:
        """Calls malloc for array_type's own total footprint
        (type_byte_width), a compile-time-known constant, leaving the
        returned pointer in %rax (the ordinary SysV return-value
        register, not chosen specially here). Used wherever a heap-
        allocated array (see is_heap_allocated) needs its own, fresh
        backing allocation: a VarDecl declaring one (gen_var_decl) or a
        parameter receiving one (gen_function's own parameter loop,
        which needs its own independent copy of the caller's data to
        preserve value semantics across the call -- exactly like a
        stack-allocated parameter already gets via gen_array_copy, just
        backed by malloc'd memory instead of an inline slot)."""
        size = type_byte_width(array_type)
        return [Mov(src=Imm(size), dst=Register('edi')), CallInstr('malloc')]

    def gen_var_decl(self, stmt: VarDecl) -> List[Instruction]:
        # _collect_locals already reserved this VarDecl's slot (that's
        # what sizes the frame); _bind_local just needs to make its name
        # resolvable in the current scope, and return where to store the
        # initializer, if there is one. `int a` with no initializer
        # leaves the slot's contents genuinely uninitialized, matching
        # C: reading it before assigning is undefined behavior, not
        # implicitly zero -- and the same holds for a heap-allocated
        # array's own malloc'd memory below: allocated, but left
        # unwritten, if there's no initializer to write through it.
        offset = self._bind_local(stmt)
        var_type = self._local_type(stmt.name)
        if self._is_array_heap_allocated(id(stmt), var_type):
            # A fresh backing allocation, made exactly once here at
            # declaration time -- see gen_assign's own array case for
            # why a later assignment reuses this same allocation
            # rather than mallocing again. %rax still holds the
            # pointer right after storing it into the slot (that store
            # only READS %rax, it doesn't touch it), so it's safe to
            # use directly as the destination for the initializer, if
            # there is one -- the same "destination is a register-held
            # address" shape gen_return already established for the
            # hidden output pointer, and gen_array_value_into/
            # gen_array_literal_into already handle generically.
            instructions = self._gen_malloc_array(var_type)
            instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', offset)))
            if stmt.init is not None:
                instructions.extend(self.gen_array_value_into(stmt.init, Memory('rax', 0), var_type))
            return instructions
        if stmt.init is None:
            return []
        if isinstance(stmt.init, NoneLiteral):
            # none's own resolved type (Type.NONE) never equals
            # var_type -- semantic.py's _types_compatible is what lets
            # this declaration through despite that (see its own
            # docstring) -- so this needs var_type, the TARGET type,
            # passed explicitly, rather than going through _gen_store's
            # ordinary dispatch, which only ever needs the value
            # expression itself since every OTHER kind of value's own
            # resolved type already matches what needs to be stored.
            return self.gen_none_into(Memory('rbp', offset), var_type)
        if isinstance(stmt.init, ArrayLiteral) and var_type.kind == TypeKind.SLICE:
            # `[]int s = [1, 2, 3]` -- an UNTYPED array literal used
            # directly as a slice's own initializer, treated exactly
            # like the general, explicitly-typed form (`[]int s =
            # []int[1, 2, 3]`, a Slice wrapping an ArrayLiteral --
            # see gen_indexable_base_into's own ArrayLiteral case):
            # construct a new, heap-allocated backing array and
            # produce a descriptor for the whole thing. Needed here,
            # separately, specifically because stmt.init's own
            # resolved type (Type(ARRAY, ...) -- see semantic.py's
            # _check_value_flowing_into) never equals var_type
            # (Type(SLICE, ...)), so _gen_store's ordinary dispatch,
            # which trusts the value's own resolved type completely,
            # would never route this to slice-producing codegen at all
            # on its own.
            instructions = self.gen_array_literal_heap_alloc_into(stmt.init)
            instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', offset)))
            instructions.append(MovQ(src=Imm(len(stmt.init.elements)), dst=Memory('rbp', offset + 8)))
            return instructions
        return self._gen_store(offset, stmt.init)

    def gen_assign(self, stmt: Assign) -> List[Instruction]:
        offset = self._local_offset(stmt.name)
        if isinstance(stmt.value, NoneLiteral):
            # See gen_var_decl's own identical case just above for why
            # this needs the TARGET type (the variable's own declared
            # type), not stmt.value's own resolved type (Type.NONE).
            var_type = self._local_type(stmt.name)
            return self.gen_none_into(Memory('rbp', offset), var_type)
        var_type = self._local_type(stmt.name)
        if isinstance(stmt.value, ArrayLiteral) and var_type.kind == TypeKind.SLICE:
            # See gen_var_decl's own identical case just above for the
            # full reasoning -- unlike an array's own Assign (just
            # below), this always mallocs a FRESH allocation rather
            # than reusing an existing one: an assigned-to slice
            # variable might currently be pointing at a DIFFERENT
            # array (or none at all) of a completely different size,
            # so there's no existing allocation here that could
            # possibly be safe to reuse in place.
            instructions = self.gen_array_literal_heap_alloc_into(stmt.value)
            instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', offset)))
            instructions.append(MovQ(src=Imm(len(stmt.value.elements)), dst=Memory('rbp', offset + 8)))
            return instructions
        value_type = self._type_of(stmt.value)
        if value_type.kind == TypeKind.ARRAY and self._is_array_heap_allocated(self._local_decl_id(stmt.name), value_type):
            # Reuses the EXISTING allocation from this variable's own
            # declaration -- a fixed-size array's footprint never
            # changes across its lifetime, so there's nothing to
            # reallocate here, only to load the existing pointer and
            # write the new value through it, exactly like gen_return
            # does for the hidden pointer it receives.
            instructions = [MovQ(src=Memory('rbp', offset), dst=Register('rax'))]
            instructions.extend(self.gen_array_value_into(stmt.value, Memory('rax', 0), value_type))
            return instructions
        return self._gen_store(offset, stmt.value)

    def gen_index_assign(self, stmt: IndexAssign) -> List[Instruction]:
        """`array[index] = value` -- computes the target element's
        address (via gen_index_address_into, which includes the
        runtime bounds check), protects it on the stack while the
        value expression is evaluated (the same push-before-recursing
        pattern used throughout this file), then writes through it.
        The element's own DECLARED type -- derived from stmt.array's
        own type, not stmt.value's -- decides the store width exactly
        like _gen_store does for an ordinary variable -- str needs
        `movq`, everything else `movl`, and a SLICE element (`rows[i]
        = someSlice`, one element of an array OF slices) needs its own
        24-byte descriptor write, via gen_slice_value_into -- which
        already protects an arbitrary dst_mem.base internally (see its
        own docstring), so this can just hand it Memory('rax', 0)
        directly rather than needing its own, separate push/pop
        dance the way the scalar path below still does.

        Deliberately NOT stmt.value's own resolved type (self._type_of
        (stmt.value)), the way this used to be computed: an UNTYPED
        array literal flowing into a SLICE-typed element (`rows[0] =
        [9, 9, 9]`) has its own resolved type set to the ARRAY it
        actually builds (see semantic.py's _check_value_flowing_into),
        not the slice it's being treated as -- so dispatching on the
        VALUE's own type would miss this case entirely and fall
        through to the scalar path below, the same bug-class already
        fixed in gen_var_decl/gen_assign (see their own docstrings)
        and analyze_index_assign, just at a third call site.

        An ARRAY-typed element (writing a whole sub-array via
        `matrix[i] = other_row`) isn't reachable here at all:
        IndexAssign's own grammar only ever produces a single
        leaf-level element write; a whole-row assignment would need
        `matrix[i]` to appear as an ordinary Assign target, which
        parser.py doesn't produce (see IndexAssign's own docstring).
        """
        base_type = self._type_of(stmt.array)
        element_type = base_type.element_type
        addr_reg = Register('rax')
        instructions = self.gen_index_address_into(Index(array=stmt.array, index=stmt.index), addr_reg)
        if element_type.kind == TypeKind.SLICE:
            instructions.extend(self.gen_slice_value_into(stmt.value, Memory('rax', 0)))
            return instructions
        instructions.append(Push(addr_reg))
        if element_type == Type.STR:
            instructions.extend(self.gen_expr_into(stmt.value, Register('eax')))
            instructions.append(MovQ(src=Register('rax'), dst=Register('r8')))  # value survives the pop below
            instructions.append(Pop(addr_reg))
            instructions.append(MovQ(src=Register('r8'), dst=Memory('rax', 0)))
        else:
            instructions.extend(self.gen_expr_into(stmt.value, Register('eax')))
            instructions.append(Mov(src=Register('eax'), dst=Register('r8d')))
            instructions.append(Pop(addr_reg))
            instructions.append(Mov(src=Register('r8d'), dst=Memory('rax', 0)))
        return instructions

    def _gen_store(self, offset: int, value_expr: Node) -> List[Instruction]:
        """Shared by VarDecl-with-initializer and Assign: both are just
        "compute this expression, then write the result into that
        variable's slot". Which store instruction depends on the
        value's type: an array can't fit into a single register at
        all, so it's dispatched to gen_array_value_into entirely
        separately (see its own docstring); a slice is a fixed-size
        24-byte descriptor, dispatched to gen_slice_value_into (see
        its own docstring) the same way; a str is an 8-byte pointer
        sitting in %rax and needs `movq`; int/bool are still the
        original 4-byte `movl %eax, ...` -- everything about
        gen_expr_into/gen_binary_into/gen_unary_op's own internals
        stays exactly as it always has, oblivious to str (or arrays,
        or slices) entirely; only this one call site needs to ask
        "which width, or which entirely different mechanism, am I
        storing"."""
        value_type = self._type_of(value_expr)
        if value_type.kind == TypeKind.ARRAY:
            return self.gen_array_value_into(value_expr, Memory('rbp', offset), value_type)
        if value_type.kind == TypeKind.SLICE:
            return self.gen_slice_value_into(value_expr, Memory('rbp', offset))
        instructions = self.gen_expr_into(value_expr, Register('eax'))
        if value_type == Type.STR:
            instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', offset)))
        else:
            instructions.append(Mov(src=Register('eax'), dst=Memory('rbp', offset)))
        return instructions

    def _gen_epilogue(self) -> List[Instruction]:
        """The ordinary function epilogue: restore every callee-saved
        scratch register (in reverse of the prologue's own push
        order), then leave/ret. Shared by gen_return's own bare-return
        case (no value to compute at all -- see Return's own docstring
        in parser.py) and gen_function's own trailing fall-through case
        (see its own comment): both are "there's no value to compute,
        just exit the function cleanly" situations. Leave resets %rsp
        straight to %rbp, which was captured before the callee-saved
        registers were pushed in the prologue, so anything pushed after
        that point has to be popped explicitly first or it's just
        silently discarded (never actually restored into the
        registers) rather than popped."""
        instructions = []
        for reg in reversed(_CALLEE_SAVED_SCRATCH_REGISTERS):
            instructions.append(Pop(Register(reg)))
        instructions.append(Leave())
        instructions.append(Ret())
        return instructions

    def gen_return(self, stmt: Return) -> List[Instruction]:
        # A bare `return` (no value at all -- valid exactly when this
        # function has no declared return type, see Return's own
        # docstring in parser.py and analyze_return's own check) needs
        # nothing computed at all, just the ordinary epilogue.
        if stmt.value is None:
            return self._gen_epilogue()

        if isinstance(stmt.value, NoneLiteral):
            # none's own resolved type (Type.NONE) never equals SLICE
            # -- semantic.py's _types_compatible is what lets `return
            # none` through despite that (see its own docstring), and
            # already guarantees it's only ever valid when THIS
            # function's own declared return type IS a slice, since
            # slices are the only nilable type that exists. Written
            # directly (not via gen_none_into, which needs a real
            # target_type to defensively check against -- not readily
            # available here, and not worth threading through just for
            # this) through the hidden return pointer, exactly like
            # every other slice-typed return value now (see below) --
            # this used to write straight into %rax/%rdx, back when a
            # slice's own descriptor still fit two registers.
            ptr_reg = Register('rax')
            instructions = [MovQ(src=Memory('rbp', self._hidden_return_ptr_offset), dst=ptr_reg)]
            instructions.append(MovQ(src=Imm(0), dst=Memory('rax', 0)))
            instructions.append(MovQ(src=Imm(0), dst=Memory('rax', 8)))
            instructions.append(MovQ(src=Imm(0), dst=Memory('rax', 16)))
            instructions.extend(self._gen_epilogue())
            return instructions

        # An array- OR slice-typed return writes directly through the
        # hidden pointer this function received (see gen_function's
        # own prologue handling and the module docstring's ARRAYS
        # section) instead of ever putting anything in %eax/%rax --
        # nothing reads a return value that way for an array- or
        # slice-returning call (see gen_array_value_into/gen_slice_
        # value_into's own Call cases, the only way such a call's
        # result is ever consumed). Loading the pointer back out of
        # its slot and handing it to gen_array_value_into/gen_slice_
        # value_into as an ordinary Memory destination is also what
        # makes `return bar()` (forwarding another array- or slice-
        # returning call's result straight out) free: the Call case
        # just passes that SAME address one level deeper via gen_
        # array_call_into/gen_slice_call_into, with no intermediate
        # copy ever materialized.
        value_type = self._type_of(stmt.value)
        if value_type.kind == TypeKind.ARRAY:
            ptr_reg = Register('rax')
            instructions = [MovQ(src=Memory('rbp', self._hidden_return_ptr_offset), dst=ptr_reg)]
            instructions.extend(self.gen_array_value_into(stmt.value, Memory('rax', 0), value_type))
        elif value_type.kind == TypeKind.SLICE:
            ptr_reg = Register('rax')
            instructions = [MovQ(src=Memory('rbp', self._hidden_return_ptr_offset), dst=ptr_reg)]
            instructions.extend(self.gen_slice_value_into(stmt.value, Memory('rax', 0)))
        else:
            dst = Register('eax')
            instructions = self.gen_expr_into(stmt.value, dst)
        # None of the epilogue touches %eax/%rax/%rdx, so a scalar
        # return value computed above is unaffected regardless of what
        # these registers held during the body (e.g. if the return
        # expression itself did string work that reused them as
        # scratch in between).
        instructions.extend(self._gen_epilogue())
        return instructions

    def gen_if(self, stmt: If) -> List[Instruction]:
        """Computes the condition into %eax and compares it to 0, exactly
        like the short-circuit AND/OR codegen already does -- then jumps
        past the `then` body when it's false:

            <condition>          ; -> %eax
            cmpl $0, %eax
            je   .Lif_else_N     ; false -> skip straight to else (or end)
            <then_body>
            jmp  .Lif_end_N      ; true -> skip over else after then runs
        .Lif_else_N:
            <else_body>          ; only emitted if else_body is present
        .Lif_end_N:

        then_body and else_body each get their own pushed/popped scope
        (see _push_scope), matching semantic.py's independent-branch
        scoping -- and since an elif is just a nested If sitting inside
        else_body (see parser.py's If docstring), gen_statement's
        ordinary recursion handles a whole elif/else chain of any
        length with no extra logic here at all.
        """
        dst = Register('eax')
        else_label = self.new_label("if_else")
        end_label = self.new_label("if_end")

        instructions = self.gen_expr_into(stmt.condition, dst)
        instructions.append(Cmp(src=Imm(0), dst=dst))
        instructions.append(Je(else_label))

        self._push_scope()
        for s in stmt.then_body:
            instructions.extend(self.gen_statement(s))
        self._pop_scope()
        instructions.append(Jmp(end_label))

        instructions.append(Label(else_label))
        if stmt.else_body is not None:
            self._push_scope()
            for s in stmt.else_body:
                instructions.extend(self.gen_statement(s))
            self._pop_scope()
        instructions.append(Label(end_label))

        return instructions

    def gen_while(self, stmt: While) -> List[Instruction]:
        """Computes the condition, re-checked before every iteration
        (including the first), with the body sitting between two labels
        that break/continue jump to:

            .Lwhile_start_N:
                <condition>          ; -> %eax
                cmpl $0, %eax
                je   .Lwhile_end_N   ; false -> exit the loop entirely
                <body>
                jmp  .Lwhile_start_N ; loop back to re-check the condition
            .Lwhile_end_N:

        Both labels get pushed onto self.loop_labels for the duration
        of generating the body, so any Break/Continue statement inside
        it -- including ones nested inside an If -- can find its way
        back here via gen_break/gen_continue without this method needing
        to know anything about where inside the body they are. Popped
        again once the body's done, so a Break/Continue *after* this
        while (or in a sibling loop) can't accidentally resolve to this
        loop's labels -- see the module docstring's LOOPS section for
        why that matters once loops nest.

        The body gets its own pushed/popped scope, same as an If's
        then/else bodies, even though it's the same physical stack slots
        being reused on every iteration (see _collect_locals) -- this is
        purely about name resolution during code generation, not
        anything that happens at runtime.
        """
        dst = Register('eax')
        start_label = self.new_label("while_start")
        end_label = self.new_label("while_end")

        instructions = [Label(start_label)]
        instructions.extend(self.gen_expr_into(stmt.condition, dst))
        instructions.append(Cmp(src=Imm(0), dst=dst))
        instructions.append(Je(end_label))

        self.loop_labels.append((start_label, end_label))
        self._push_scope()
        for s in stmt.body:
            instructions.extend(self.gen_statement(s))
        self._pop_scope()
        self.loop_labels.pop()

        instructions.append(Jmp(start_label))
        instructions.append(Label(end_label))
        return instructions

    def gen_break(self, stmt: Break) -> List[Instruction]:
        # semantic.py already guarantees this only appears inside a
        # loop; the IndexError-avoiding check here is the same defensive
        # posture as _local_offset's -- see the module docstring on
        # generate_asm/compile_to_asm for why codegen still checks for
        # itself rather than trusting semantic analysis unconditionally.
        if not self.loop_labels:
            raise CodegenError("'break' outside of a loop")
        _, end_label = self.loop_labels[-1]
        return [Jmp(end_label)]

    def gen_continue(self, stmt: Continue) -> List[Instruction]:
        if not self.loop_labels:
            raise CodegenError("'continue' outside of a loop")
        start_label, _ = self.loop_labels[-1]
        return [Jmp(start_label)]

    def gen_expr_stmt(self, stmt: ExprStmt) -> List[Instruction]:
        # Evaluated the same way as any other expression, into %eax --
        # just with nothing done with the result afterward. Still real
        # instructions that really run; see the module docstring for
        # how that's verified (a standalone `1 / 0` genuinely crashes).
        #
        # An ArrayLiteral is the one exception: it can't be computed
        # via gen_expr_into at all (an array doesn't fit in a single
        # register), and unlike a VarDecl/Assign's own use of one, a
        # bare literal statement has no destination to write the
        # resulting array into -- but it doesn't need one, since
        # nothing ever reads the array as a whole. See gen_array_
        # literal_side_effects_only's own docstring for the resulting,
        # narrower approach: evaluate each element for whatever side
        # effects it might have, without ever materializing a real
        # array in memory at all.
        if isinstance(stmt.expr, ArrayLiteral):
            return self.gen_array_literal_side_effects_only(stmt.expr)
        # A Slice expression is the analogous exception for slices --
        # a 24-byte descriptor doesn't fit in a single register either
        # -- but unlike ArrayLiteral, this doesn't need its own
        # narrower, side-effects-only path: gen_slice_into already
        # computes fully correctly into any Memory destination,
        # including a genuine runtime bounds check on low/high (an
        # out-of-range bound still aborts here, matching how any other
        # bare expression statement's real instructions genuinely run
        # -- see this method's own opening comment), so this just
        # reuses the same per-function scratch slot gen_indexable_
        # base_into's own Slice-base case already uses (_unnamed_
        # slice_temp_offset) and discards the result -- nothing ever
        # reads it. Covers both a bare slice LITERAL statement
        # (`[]int[se(), 2, 3]`, parsed as a Slice wrapping an
        # ArrayLiteral -- see parser.py's own _parse_bracketed_
        # literal) and an ordinary bare slice of an EXISTING array or
        # slice (`arr[:]` alone, pointless but not an error) with the
        # exact same code path.
        if isinstance(stmt.expr, Slice):
            return self.gen_slice_into(stmt.expr, Memory('rbp', self._unnamed_slice_temp_offset))
        return self.gen_expr_into(stmt.expr, Register('eax'))

    def gen_array_literal_side_effects_only(self, expr: ArrayLiteral) -> List[Instruction]:
        """A bare array-literal statement (`[3]int[1, 2, 3]` alone,
        with no assignment) never needs its VALUE materialized
        anywhere at all -- nothing ever reads it as a coherent array
        -- so rather than reserving a scratch slot sized to fit it
        (which, unlike a slice's fixed 24-byte descriptor, an array
        literal has no natural upper bound for), this just evaluates
        each of the literal's own, directly-written elements for
        whatever side effects it might have (e.g. a function call),
        discarding every result -- exactly like any other bare
        expression statement already does (see gen_expr_stmt).

        Recurses for a nested ArrayLiteral element (a multi-dimensional
        literal used bare), the same way check_array_literal's own
        type-checking already does. An element that's ITSELF some
        other, non-literal array- or slice-typed expression (a
        Variable, an indexed sub-array, or an array-returning Call
        used as an element) is a real, deliberately out-of-scope gap:
        reading a bare array-typed Variable has no side effect worth
        preserving, but an array-returning Call might, and correctly
        distinguishing the two -- or materializing either one just to
        discard it -- isn't implemented here. Raises a clear error
        rather than silently skipping (which could drop a real side
        effect) or guessing.
        """
        instructions = []
        for element in expr.elements:
            if isinstance(element, ArrayLiteral):
                instructions.extend(self.gen_array_literal_side_effects_only(element))
                continue
            element_type = self._type_of(element)
            if element_type.kind in (TypeKind.ARRAY, TypeKind.SLICE):
                raise CodegenError(
                    f"A bare array-literal statement can't have a "
                    f"{type(element).__name__} element of type "
                    f"{element_type} -- assign the literal to a "
                    f"variable first if you need this element's value "
                    f"or side effect evaluated"
                )
            instructions.extend(self.gen_expr_into(element, Register('eax')))
        return instructions

    def gen_expr_into(self, expr: Node, dst: Operand) -> List[Instruction]:
        """Emits the instructions needed to compute `expr` and leave its
        result sitting in `dst`.

        This (rather than "return an Operand") is the right shape for
        expression codegen once operators are involved: a Constant can
        be represented as a bare Imm operand, but "the result of negating
        something" can't -- it has to actually be computed by an
        instruction acting on a register. So every expression, constants
        included, is generated the same way: as instructions that leave
        their answer in `dst`.
        """
        if isinstance(expr, Constant):
            return [Mov(src=Imm(expr.value), dst=dst)]
        if isinstance(expr, BoolLiteral):
            # bool has the same 4-byte runtime representation as int
            # (0/1 in a register or stack slot) -- semantic.py is what
            # keeps the two from being mixed up; codegen just needs an
            # immediate.
            return [Mov(src=Imm(1 if expr.value else 0), dst=dst)]
        if isinstance(expr, StringLiteral):
            return self.gen_string_literal_into(expr, dst)
        if isinstance(expr, ArrayLiteral):
            # Never reachable in correct codegen -- an array literal's
            # value can't fit in a single register, so every producer
            # of one (VarDecl init, Assign, a nested literal element)
            # routes through gen_array_value_into/gen_array_literal_into
            # instead of ever calling gen_expr_into on it directly. A
            # clear error here catches a codegen bug immediately rather
            # than silently truncating an array down to whatever
            # happens to fit in %eax.
            raise CodegenError(
                "Cannot compute an array literal via gen_expr_into -- "
                "arrays don't fit in a single register; use "
                "gen_array_value_into instead"
            )
        if isinstance(expr, Slice):
            # Never reachable in correct codegen -- a slice's value is
            # a 24-byte {ptr, len, cap} descriptor, which can't fit in a
            # single register either. Every producer of one (VarDecl
            # init, Assign) routes through gen_slice_value_into/
            # gen_slice_into instead of ever calling gen_expr_into on
            # it directly -- see ArrayLiteral's own case just above
            # for the identical reasoning.
            raise CodegenError(
                "Cannot compute a slice expression via gen_expr_into -- "
                "slices don't fit in a single register; use "
                "gen_slice_value_into instead"
            )
        if isinstance(expr, NoneLiteral):
            # Never reachable in correct codegen either, for a
            # different reason than ArrayLiteral/Slice above: it's not
            # a SIZE problem here at all (rejected for the same size
            # reason Slice is, now that none's own {0, 0, 0} descriptor
            # is exactly as wide as any other slice's) -- it's that
            # none has no ONE fixed target type of its own to compute
            # INTO -- see gen_none_into's own docstring for why its
            # callers (gen_var_decl/gen_assign's own NoneLiteral
            # short-circuit) have to already know and pass the target
            # type explicitly, something gen_expr_into's own signature
            # has no way to supply. A slice-vs-none comparison (`s ==
            # none`) is handled entirely separately too, via
            # gen_slice_none_comparison_into, dispatched from
            # gen_binary_into before it would ever reach here.
            raise CodegenError(
                "Cannot compute 'none' via gen_expr_into -- it's only "
                "supported as a slice's zero value (see gen_none_into) "
                "or as one side of a slice comparison (see "
                "gen_slice_none_comparison_into), never as a general-"
                "purpose expression value"
            )
        if isinstance(expr, Variable):
            offset = self._local_offset(expr.name)
            var_type = self._local_type(expr.name)
            if var_type.kind == TypeKind.ARRAY:
                raise CodegenError(
                    f"Cannot read array-typed variable '{expr.name}' via "
                    f"gen_expr_into -- arrays don't fit in a single "
                    f"register; use gen_array_value_into or "
                    f"gen_array_address_into instead"
                )
            if var_type.kind == TypeKind.SLICE:
                raise CodegenError(
                    f"Cannot read slice-typed variable '{expr.name}' via "
                    f"gen_expr_into -- slices don't fit in a single "
                    f"register; use gen_slice_value_into instead"
                )
            if var_type == Type.STR:
                return [MovQ(src=Memory('rbp', offset), dst=as_qword_register(dst))]
            return [Mov(src=Memory('rbp', offset), dst=dst)]
        if isinstance(expr, Index):
            element_type = self._type_of(expr)
            if element_type.kind == TypeKind.ARRAY:
                # Reading a sub-array (e.g. `matrix[i]` alone, not yet
                # fully indexed down to a scalar) has the same "doesn't
                # fit in a register" problem as an array literal --
                # `[3]int row = matrix[i]` is handled via
                # gen_array_value_into instead, which calls
                # gen_array_address_into directly rather than ever
                # reaching this method for the sub-array's VALUE.
                raise CodegenError(
                    "Cannot read a sub-array via gen_expr_into -- arrays "
                    "don't fit in a single register; use "
                    "gen_array_value_into or gen_array_address_into instead"
                )
            addr_reg = as_qword_register(dst)
            instructions = self.gen_index_address_into(expr, addr_reg)
            if element_type == Type.STR:
                instructions.append(MovQ(src=Memory(addr_reg.name, 0), dst=addr_reg))
            else:
                instructions.append(Mov(src=Memory(addr_reg.name, 0), dst=dst))
            return instructions
        if isinstance(expr, Call):
            if self._type_of(expr).kind == TypeKind.ARRAY:
                # Never reachable in correct codegen -- see ArrayLiteral
                # and the Index sub-array case just above for the same
                # reasoning. An array-returning call's result is only
                # ever consumed via gen_array_value_into's own Call case
                # (which writes it, through the hidden-pointer
                # convention, straight into a given destination), never
                # by trying to land it in a single register here.
                raise CodegenError(
                    f"Cannot call '{expr.name}' (which returns an array) "
                    f"via gen_expr_into -- arrays don't fit in a single "
                    f"register; use gen_array_value_into instead"
                )
            if self._type_of(expr).kind == TypeKind.SLICE:
                # Same reasoning, for the same underlying cause: a
                # slice-returning call now writes its result through
                # the hidden-pointer convention too (see gen_slice_
                # call_into), just like an array-returning one, never
                # by trying to land it in a single register here. Only
                # ever reached via gen_slice_value_into's own Call case
                # (VarDecl/Assign) or gen_return's own forwarding case.
                raise CodegenError(
                    f"Cannot call '{expr.name}' (which returns a slice) "
                    f"via gen_expr_into -- a slice descriptor doesn't "
                    f"fit in a single register; use gen_slice_call_into "
                    f"instead"
                )
            if expr.name == 'print':
                return self.gen_print_call_into(expr, dst)
            if expr.name == 'len':
                return self.gen_len_call_into(expr, dst)
            return self.gen_call_into(expr, dst)
        if isinstance(expr, Unary):
            # Compute the operand into dst first, then apply this node's
            # operator to whatever's now there. This is what makes chained
            # operators (`~-2`) work: the inner Unary's instructions run
            # first, then the outer operator's instructions run on top.
            instructions = self.gen_expr_into(expr.operand, dst)
            instructions.extend(self.gen_unary_op(expr.op, dst))
            return instructions
        if isinstance(expr, Binary):
            # ADD and the two equality operators are overloaded for str
            # (concatenation and strcmp-backed comparison respectively;
            # see the module docstring's STRINGS section) -- everything
            # else, and ADD/==/!= between two ints or bools, goes
            # through the original gen_binary_into completely unchanged.
            if expr.op == BinaryOp.ADD and self._type_of(expr.left) == Type.STR:
                return self.gen_string_concat_into(expr, dst)
            if expr.op in (BinaryOp.EQUAL, BinaryOp.NOT_EQUAL) and self._type_of(expr.left) == Type.STR:
                return self.gen_string_compare_into(expr, dst)
            return self.gen_binary_into(expr, dst)
        raise CodegenError(f"No codegen rule for expression: {expr!r}")

    def gen_binary_into(self, expr: Binary, dst: Operand) -> List[Instruction]:
        """Computes `expr.left OP expr.right` into `dst`.

        AND/OR are handled entirely separately (see gen_short_circuit)
        since they must not unconditionally evaluate both sides. Every
        other binary operator -- arithmetic and comparisons alike -- goes
        through the stack-spill scheme described in the module
        docstring, which always evaluates both sides. Requires `dst` to
        be a register (there's a real 32-bit register and its 64-bit
        alias pushed/popped along the way, which an Imm can't do).
        """
        if expr.op == BinaryOp.AND:
            return self.gen_short_circuit(
                expr, dst,
                short_circuit_jump=Je,   # jump early when the left side is already false
                short_circuit_value=0,   # ...and the overall result is false
                fallthrough_value=1,     # both sides were truthy -> true
                label_prefix="and",
            )
        if expr.op == BinaryOp.OR:
            return self.gen_short_circuit(
                expr, dst,
                short_circuit_jump=Jne,  # jump early when the left side is already true
                short_circuit_value=1,   # ...and the overall result is true
                fallthrough_value=0,     # both sides were falsy -> false
                label_prefix="or",
            )

        if not isinstance(dst, Register):
            raise CodegenError(f"Binary codegen requires a register destination, got: {dst!r}")

        # A slice compared to `none` (in either order -- `s == none`
        # and `none == s` both reach here) needs its own dedicated
        # codegen too, for a related but distinct reason from AND/OR
        # above: a slice's "value" is a 24-byte descriptor, which
        # can't flow through the ordinary single-register stack-spill
        # scheme below the way an int/bool/str value can. semantic.py's
        # check_binary already guarantees, by the time this is reached,
        # that exactly one side is slice-typed and the other is none-
        # typed -- a real slice compared to another real slice
        # (`s1 == s2`), or none compared to none, is rejected earlier
        # -- so this doesn't need to re-derive or defensively check
        # which side is which beyond that.
        if expr.op in (BinaryOp.EQUAL, BinaryOp.NOT_EQUAL):
            if self._type_of(expr.left).kind == TypeKind.SLICE or self._type_of(expr.right).kind == TypeKind.SLICE:
                return self.gen_slice_none_comparison_into(expr, dst)

        scratch = Register('ecx')  # holds the right-hand value while combining
        instructions = self.gen_expr_into(expr.left, dst)   # dst = left
        instructions.append(Push(as_qword_register(dst)))   # save left on the stack
        instructions.extend(self.gen_expr_into(expr.right, dst))  # dst = right (left is safe)
        instructions.append(Mov(src=dst, dst=scratch))       # scratch = right
        instructions.append(Pop(as_qword_register(dst)))     # dst = left (restored)
        instructions.extend(self.gen_binary_op(expr.op, src=scratch, dst=dst))
        return instructions

    def gen_slice_none_comparison_into(self, expr: Binary, dst: Register) -> List[Instruction]:
        """Computes `slice_expr == none` or `slice_expr != none` (in
        either operand order) into dst -- checking specifically
        whether the slice's own `ptr` field is null, matching Go's
        own nil-vs-empty-slice distinction (see NoneLiteral's own
        docstring in parser.py): a real, zero-length slice sliced
        from a real array (e.g. `arr[5:5]`) has a non-null pointer
        and is NOT `== none`, even though both are equally safe,
        equally zero-length slices for every other purpose (indexing,
        printing, re-slicing).

        semantic.py's check_binary already guarantees, by the time
        this is reached, that exactly one operand is slice-typed and
        the other is none-typed -- so this doesn't need to re-derive
        or defensively check which side is which beyond picking out
        whichever one IS slice-typed.

        Reuses gen_indexable_base_into for the slice's own address
        (see its own docstring for why the base must be a bare
        Variable when it's slice-typed) even though only the address,
        not the length or capacity it also computes, is actually
        needed here -- two harmless, unused extra loads rather than a
        second, narrower helper that would duplicate its Variable-vs-
        Index and array-vs-slice handling for a single call site.

        Uses CmpQ (64-bit), not the ordinary 32-bit Cmp every other
        comparison in this file uses -- a pointer is a full 64-bit
        value, and checking only its low 32 bits against zero could,
        in principle, miss a real, non-null pointer whose low 32 bits
        happen to be zero.
        """
        slice_expr = expr.left if self._type_of(expr.left).kind == TypeKind.SLICE else expr.right
        addr_reg = Register('rbx')
        len_reg = Register('r12')  # unused here; gen_indexable_base_into always computes it
        cap_reg = Register('r13')  # unused here too
        instructions, _, _ = self.gen_indexable_base_into(slice_expr, addr_reg, len_reg, cap_reg)
        instructions.append(CmpQ(src=Imm(0), dst=addr_reg))
        cc = 'e' if expr.op == BinaryOp.EQUAL else 'ne'
        byte_dst = as_byte_register(dst)
        instructions.append(SetCC(cc=cc, operand=byte_dst))
        instructions.append(MovZX(src=byte_dst, dst=dst))
        return instructions

    def gen_short_circuit(
        self, expr: Binary, dst: Operand, *,
        short_circuit_jump: type,
        short_circuit_value: int,
        fallthrough_value: int,
        label_prefix: str,
    ) -> List[Instruction]:
        """Shared codegen for AND and OR -- they're mirror images of each
        other: each evaluates its left side, tests it against 0, and
        jumps straight past the right side entirely (never emitting the
        instructions that would compute it as *executed* code) if that
        test already decides the answer. Only if it doesn't -- left was
        truthy for AND, falsy for OR -- does the right side actually get
        evaluated, and *that* result decides the answer instead.

          AND: jump early (to `short_circuit_value=0`) when left == 0.
          OR:  jump early (to `short_circuit_value=1`) when left != 0.

        This is what makes `0 and (1 / 0)` return 0 instead of crashing:
        the division is real code sitting in the binary, but control
        flow jumps clean over it.
        """
        if not isinstance(dst, Register):
            raise CodegenError(f"Binary codegen requires a register destination, got: {dst!r}")

        short_label = self.new_label(f"{label_prefix}_short")
        end_label = self.new_label(f"{label_prefix}_end")

        instructions = self.gen_expr_into(expr.left, dst)
        instructions.append(Cmp(src=Imm(0), dst=dst))
        instructions.append(short_circuit_jump(short_label))

        instructions.extend(self.gen_expr_into(expr.right, dst))
        instructions.append(Cmp(src=Imm(0), dst=dst))
        instructions.append(short_circuit_jump(short_label))

        instructions.append(Mov(src=Imm(fallthrough_value), dst=dst))
        instructions.append(Jmp(end_label))
        instructions.append(Label(short_label))
        instructions.append(Mov(src=Imm(short_circuit_value), dst=dst))
        instructions.append(Label(end_label))
        return instructions

    def gen_call_into(self, expr: Call, dst: Operand) -> List[Instruction]:
        """`name(arg1, arg2, ...)`: evaluates and passes every
        argument via the shared _gen_call_arguments_into (see its own
        docstring for the full push-then-pop-in-reverse discipline,
        and how a slice-typed argument's own three register slots are
        placed correctly among any ordinary scalar/array ones), then
        calls the function.

        The result already ends up exactly where gen_expr_into's
        contract expects it (%rax/%eax, matching `dst`, which is always
        Register('eax') throughout this file), so there's nothing left
        to move once the call returns. This method is never reached at
        all for a callee that returns an array or a slice -- see
        gen_array_call_into and gen_slice_call_into, which now share
        the exact same hidden-pointer return-value convention (see the
        module docstring's SLICE PARAMETERS AND RETURNS section), and
        that convention doesn't fit a single generic `dst` the way an
        ordinary scalar return does.
        """
        total_slots = self._total_arg_slots(expr.args)
        if total_slots > 6:
            raise CodegenError(
                f"Call to '{expr.name}' needs {total_slots} argument "
                f"register(s) (a slice-typed argument needs 3); this "
                f"compiler only supports up to 6 (passed via registers "
                f"per the SysV ABI -- stack-passed arguments aren't "
                f"implemented)"
            )
        if dst != Register('eax'):
            raise CodegenError(f"Call codegen requires dst == %eax, got: {dst!r}")

        instructions = self._gen_call_arguments_into(expr.args)
        instructions.append(CallInstr(expr.name))
        return instructions

    def gen_print_call_into(self, expr: Call, dst: Operand) -> List[Instruction]:
        """`print(x)`: dispatches on x's *compile-time* type -- known
        exactly, since Hornet is statically typed.

          str:  puts(x)                    -- puts adds its own newline
          int:  printf("%d\\n", x)         -- needs real formatting
          bool: puts(x ? "true" : "false") -- a runtime branch (the
                exact same cmp/je/jmp/label shape gen_if already uses)
                picks which string literal's address to pass, then
                falls through to the same puts call as the str case
          array/slice: `TYPE[elem, elem, ...]\\n` -- the type prefix
                (str(arg_type), e.g. "[3]int" or "[]int") printed once,
                then gen_indexable_base_into's own address+length,
                handed to _gen_print_collection for the recursive,
                piece-by-piece body -- see the module docstring's
                PRINTING ARRAYS AND SLICES section for the full design.
                Restricted to a Variable or Index argument, matching
                gen_array_arg_address_into's own restriction elsewhere:
                a bare ArrayLiteral, Slice, or array/slice-returning
                Call has no address of its own to print through --
                assign it to a named variable first.

        Every path still ends with `movl $0, %eax`, a harmless leftover
        from before print had anywhere real to return to (it used to be
        Type.INT, "returning" a clean, predictable 0 -- see semantic.py's
        check_print_call). print is Type.VOID now, so nothing reads
        %eax after this call anymore, the same as any other void call's
        leftover register value -- but leaving this in costs nothing
        and avoids restructuring dst handling here just because the
        value it computes is no longer semantically meaningful.

        No register-preservation concerns beyond the ones already
        established: puts/printf are libc functions, and libc is
        already a fully ABI-compliant citizen (that's the entire point
        of the ABI), so calling them from inside a Hornet function's
        body is exactly as safe as calling another Hornet function --
        both rely on the callee-saved registers being honored by
        whatever gets called, which is now true either way (see
        gen_function's prologue).
        """
        if dst != Register('eax'):
            raise CodegenError(f"Call codegen requires dst == %eax, got: {dst!r}")

        arg = expr.args[0]
        arg_type = self._type_of(arg)

        if arg_type.kind in (TypeKind.ARRAY, TypeKind.SLICE):
            if not isinstance(arg, (Variable, Index)):
                raise CodegenError(
                    f"print's argument must be a variable or an "
                    f"indexing expression when it's array- or slice-"
                    f"typed, not {type(arg).__name__} -- assign it to "
                    f"a variable first"
                )
            instructions, length_operand, _ = self.gen_indexable_base_into(
                arg, Register('rbx'), Register('r12'), Register('r13')
            )
            instructions.extend(self._gen_print_static(str(arg_type)))
            instructions.extend(self._gen_print_collection(length_operand, arg_type))
            instructions.extend(self._gen_print_static("\n"))
            instructions.append(Mov(src=Imm(0), dst=dst))
            return instructions

        if arg_type == Type.STR:
            instructions = self.gen_expr_into(arg, dst)
            instructions.append(MovQ(src=as_qword_register(dst), dst=Register('rdi')))
            instructions.append(CallInstr('puts'))
            instructions.append(Mov(src=Imm(0), dst=dst))
            return instructions

        if arg_type == Type.INT:
            fmt_label = self._get_int_format_label()
            instructions = self.gen_expr_into(arg, dst)
            instructions.append(Mov(src=dst, dst=Register('esi')))       # esi = value (2nd printf arg)
            instructions.append(LeaQ(label=fmt_label, dst=Register('rdi')))  # rdi = &"%d\n" (1st arg)
            # AL must be 0 before calling a variadic function per the
            # SysV ABI (it tells the callee how many vector/xmm
            # registers were used for float varargs -- always 0 here,
            # since nothing in this language is ever passed as a float).
            # A plain `movl $0, %eax` both clears AL and is a completely
            # safe clobber of %eax at this point, since the value we
            # care about was already copied into %esi just above.
            instructions.append(Mov(src=Imm(0), dst=dst))
            instructions.append(CallInstr('printf'))
            instructions.append(Mov(src=Imm(0), dst=dst))
            return instructions

        if arg_type == Type.BOOL:
            true_label = self._get_true_str_label()
            false_label = self._get_false_str_label()
            false_branch_label = self.new_label("print_bool_false")
            end_label = self.new_label("print_bool_end")

            instructions = self.gen_expr_into(arg, dst)
            instructions.append(Cmp(src=Imm(0), dst=dst))
            instructions.append(Je(false_branch_label))
            instructions.append(LeaQ(label=true_label, dst=Register('rdi')))
            instructions.append(Jmp(end_label))
            instructions.append(Label(false_branch_label))
            instructions.append(LeaQ(label=false_label, dst=Register('rdi')))
            instructions.append(Label(end_label))
            instructions.append(CallInstr('puts'))
            instructions.append(Mov(src=Imm(0), dst=dst))
            return instructions

        raise CodegenError(f"'print' has no codegen rule for type: {arg_type}")

    def gen_len_call_into(self, expr: Call, dst: Operand) -> List[Instruction]:
        """`len(x)`: reuses gen_indexable_base_into directly -- the
        exact same "address plus length, however each is represented"
        abstraction indexing and slicing already share -- rather than
        a narrower restriction of its own like print's Variable-or-
        Index one (see gen_print_call_into's own docstring): whatever
        gen_indexable_base_into currently accepts as a base (a
        Variable, an Index, a Slice expression, a slice-returning
        Call, or an ArrayLiteral) is automatically valid here too,
        with nothing to keep in sync if that set ever grows.

        x's own address is computed and then simply discarded -- len
        only ever needs the LENGTH half of gen_indexable_base_into's
        own return value -- but computing it is not wasted: x is still
        fully evaluated regardless (any bounds-check or side effect
        buried in it genuinely runs), matching how any other function
        argument's evaluation works, whether or not the computed
        address ends up used for anything afterward. This does mean
        `len(arr[i])` still aborts if i is out of range, and
        `len([]int[1, 2, 3])` still performs a real, if wasted, heap
        allocation -- both deliberate, not something a narrower
        special case tries to avoid (see the module docstring's LEN
        BUILTIN section).

        For an ARRAY base, length_operand comes back as an Imm (a
        compile-time constant -- the array's own declared size, never
        actually read out of x at runtime at all); for a SLICE base,
        as the 64-bit len_dst register holding a runtime value read
        out of the slice's own descriptor -- moved through its own
        32-bit alias here, matching how every other reader of a
        slice's length field already narrows it the same way, since
        Hornet's int is always 32 bits even though the descriptor's
        own len field is stored in a full 8-byte slot."""
        if dst != Register('eax'):
            raise CodegenError(f"Call codegen requires dst == %eax, got: {dst!r}")
        arg = expr.args[0]
        len_reg = Register('r12')
        cap_reg = Register('r13')
        instructions, length_operand, _ = self.gen_indexable_base_into(
            arg, Register('rbx'), len_reg, cap_reg
        )
        if isinstance(length_operand, Register):
            instructions.append(Mov(src=Register('r12d'), dst=dst))
        else:
            instructions.append(Mov(src=length_operand, dst=dst))
        return instructions

    def _get_int_format_label(self) -> str:
        if self._int_format_label is None:
            self._int_format_label = self.new_label("fmt_int")
            self.string_literals.append((self._int_format_label, "%d\n"))
        return self._int_format_label

    def _get_true_str_label(self) -> str:
        if self._true_str_label is None:
            self._true_str_label = self.new_label("true_str")
            self.string_literals.append((self._true_str_label, "true"))
        return self._true_str_label

    def _get_false_str_label(self) -> str:
        if self._false_str_label is None:
            self._false_str_label = self.new_label("false_str")
            self.string_literals.append((self._false_str_label, "false"))
        return self._false_str_label

    # -- printing arrays and slices ----------------------------------------
    # See the module docstring's PRINTING ARRAYS AND SLICES section for
    # the full design. Format: `TYPE[elem, elem, ...]` -- e.g.
    # `[3]int[1, 2, 3]` or `[]int[1, 2, 3]` -- the type prefix appearing
    # exactly once, at the outermost level, never repeated for a nested
    # row. Built as a sequence of direct printf calls, one piece at a
    # time (the type prefix, each bracket, each separator, each
    # element), rather than materializing one big string via malloc and
    # printing it in one shot -- which would need a new int-to-string
    # conversion step this language has no other reason to have; every
    # existing int print already goes straight to printf's own %d
    # formatting, never through an intermediate string buffer.

    def _get_static_string_label(self, text: str) -> str:
        """Lazily creates and caches (for the whole compilation, like
        print's own %d-format/true/false labels above) a label for
        this exact string, deduped by content -- e.g. every
        "[3]int"-typed print call anywhere in the program shares one
        cached label rather than emitting a fresh string literal per
        call site, the same "small dedicated cache" policy those
        labels already use, just generalized to arbitrary content
        instead of three fixed strings."""
        if text not in self._static_string_labels:
            label = self.new_label("str_lit")
            self._static_string_labels[text] = label
            self.string_literals.append((label, text))
        return self._static_string_labels[text]

    def _gen_print_static(self, text: str) -> List[Instruction]:
        """Prints a compile-time-known string with NO trailing newline
        (unlike puts, which always appends one) -- used for every
        punctuation/prefix piece of array/slice printing (the type
        prefix, brackets, separators, and the single final newline
        gen_print_call_into appends once at the very end), each of
        which needs to NOT have its own newline so the whole
        collection prints as one line.

        Passes `text` directly as printf's own format string, rather
        than going through a separate "%s" argument -- safe because
        every string this is ever called with is either a hardcoded
        punctuation piece or a type's own str() (see
        semantic.Type.__str__), neither of which can ever contain a
        literal '%' character."""
        label = self._get_static_string_label(text)
        return [
            LeaQ(label=label, dst=Register('rdi')),
            Mov(src=Imm(0), dst=Register('eax')),
            CallInstr('printf'),
        ]

    def _gen_print_quoted_str(self, value_reg: Register) -> List[Instruction]:
        """Prints a str VALUE (already in value_reg, a 64-bit pointer)
        wrapped in single quotes, via printf("'%s'", value) -- used
        for str elements WITHIN an array/slice being printed.
        Distinct from print's own top-level str handling (unquoted,
        via puts): quoting only applies inside a collection, matching
        how most languages format a string differently in a
        collection/repr context than when printed bare."""
        fmt_label = self._get_static_string_label("'%s'")
        return [
            MovQ(src=value_reg, dst=Register('rsi')),
            LeaQ(label=fmt_label, dst=Register('rdi')),
            Mov(src=Imm(0), dst=Register('eax')),
            CallInstr('printf'),
        ]

    def _gen_print_int_value(self, value_reg: Register) -> List[Instruction]:
        """Prints an int VALUE (already in a 32-bit register) via
        printf("%d", value), with NO trailing newline -- used for int
        elements within an array/slice. Distinct from print's own
        top-level int handling (_get_int_format_label's "%d\\n"),
        which does include one."""
        fmt_label = self._get_static_string_label("%d")
        return [
            Mov(src=value_reg, dst=Register('esi')),
            LeaQ(label=fmt_label, dst=Register('rdi')),
            Mov(src=Imm(0), dst=Register('eax')),
            CallInstr('printf'),
        ]

    def _gen_print_bool_value(self, value_reg: Register) -> List[Instruction]:
        """Prints a bool VALUE (already in a 32-bit register, 0 or 1)
        as "true"/"false", with NO trailing newline -- reuses the same
        cached true/false string labels print's own top-level bool
        handling already caches (they were never stored WITH a
        newline in the first place -- puts is what adds one there,
        not the string itself), just calling printf directly on
        whichever one applies instead of going through puts."""
        true_label = self._get_true_str_label()
        false_label = self._get_false_str_label()
        false_branch = self.new_label("print_elem_bool_false")
        end_label = self.new_label("print_elem_bool_end")
        return [
            Cmp(src=Imm(0), dst=value_reg),
            Je(false_branch),
            LeaQ(label=true_label, dst=Register('rdi')),
            Jmp(end_label),
            Label(false_branch),
            LeaQ(label=false_label, dst=Register('rdi')),
            Label(end_label),
            Mov(src=Imm(0), dst=Register('eax')),
            CallInstr('printf'),
        ]

    def _gen_print_collection(self, length_operand: Union[Imm, Register], collection_type: Type) -> List[Instruction]:
        """Prints `[elem, elem, ...]` for a collection (array- or
        slice-typed) whose base ADDRESS the caller has already placed
        in %rbx, and whose LENGTH is `length_operand` -- an Imm for a
        compile-time-known array length, or Register('r12')
        (populated by the caller with a runtime value) for a slice's
        own length. No leading type prefix, no trailing newline -- see
        gen_print_call_into for those, which only ever happen once, at
        the very outermost level.

        Uses a genuine runtime loop, even when length_operand is a
        compile-time Imm (an array base) -- rather than unrolling at
        compile time, one uniform code path handles both an array's
        and a slice's own length identically, the same "however it's
        represented" uniformity gen_indexable_base_into's own other
        callers already rely on.

        %rbx (the address), %r12 (the length, when it's a runtime
        value), and %r13 (the loop counter) all have to survive across
        every printf/puts call this loop makes -- at least one per
        element -- so all three are CALLEE-SAVED registers, which a
        well-behaved libc call is obligated to preserve, rather than
        the caller-saved scratch (rax, rcx, rdx, ...) most of this
        file's transient computations already use. A nested array
        element (this method's own recursive case, for a multi-
        dimensional array's rows) needs all three protected on the
        stack across the RECURSIVE call specifically, since that call
        reuses these same three physical registers for its own,
        independent address/length/counter -- exactly the same push-
        before-recursing discipline used everywhere else in this file
        a value needs to survive evaluating something else, just
        applied to a whole recursive call instead of a single sub-
        expression.
        """
        element_type = collection_type.element_type
        element_stride = type_byte_width(element_type)
        is_runtime_length = isinstance(length_operand, Register)

        ADDR = Register('rbx')
        LEN = Register('r12')
        COUNTER = Register('r13')

        instructions = self._gen_print_static("[")
        instructions.append(Mov(src=Imm(0), dst=Register('r13d')))

        loop_start = self.new_label("print_loop_start")
        loop_end = self.new_label("print_loop_end")
        skip_sep = self.new_label("print_skip_sep")

        instructions.append(Label(loop_start))
        length_op_32 = Register('r12d') if is_runtime_length else length_operand
        instructions.append(Cmp(src=length_op_32, dst=Register('r13d')))
        instructions.append(Jae(loop_end))

        instructions.append(Cmp(src=Imm(0), dst=Register('r13d')))
        instructions.append(Je(skip_sep))
        instructions.extend(self._gen_print_static(", "))
        instructions.append(Label(skip_sep))

        # element address = ADDR + COUNTER * element_stride, into %rax.
        # A plain 32-bit imul is safe: COUNTER is always small and
        # non-negative (it's this loop's own counter), and a 32-bit
        # write zero-extends into the full 64-bit rax.
        instructions.append(Mov(src=Register('r13d'), dst=Register('eax')))
        instructions.append(IMul(src=Imm(element_stride), dst=Register('eax')))
        instructions.append(AddQ(src=ADDR, dst=Register('rax')))
        # %rax now holds the element's own address.

        if element_type.kind == TypeKind.ARRAY:
            instructions.append(Push(ADDR))
            if is_runtime_length:
                instructions.append(Push(LEN))
            instructions.append(Push(COUNTER))
            instructions.append(MovQ(src=Register('rax'), dst=ADDR))
            instructions.extend(self._gen_print_collection(Imm(element_type.size), element_type))
            instructions.append(Pop(COUNTER))
            if is_runtime_length:
                instructions.append(Pop(LEN))
            instructions.append(Pop(ADDR))
        elif element_type.kind == TypeKind.SLICE:
            # Not reachable via any currently-constructible program
            # (an array or slice of slices can't be initialized --
            # see gen_array_copy's own rejection), but handled
            # correctly anyway rather than left to do something
            # arbitrary: reads the nested slice's own descriptor
            # (ptr at +0, len at +8) from the element address just
            # computed, exactly like the top-level slice case does.
            instructions.append(Push(ADDR))
            if is_runtime_length:
                instructions.append(Push(LEN))
            instructions.append(Push(COUNTER))
            instructions.append(MovQ(src=Memory('rax', 8), dst=LEN))
            instructions.append(MovQ(src=Memory('rax', 0), dst=ADDR))
            instructions.extend(self._gen_print_collection(LEN, element_type))
            instructions.append(Pop(COUNTER))
            if is_runtime_length:
                instructions.append(Pop(LEN))
            instructions.append(Pop(ADDR))
        elif element_type == Type.STR:
            instructions.append(MovQ(src=Memory('rax', 0), dst=Register('rax')))
            instructions.extend(self._gen_print_quoted_str(Register('rax')))
        elif element_type == Type.BOOL:
            instructions.append(Mov(src=Memory('rax', 0), dst=Register('eax')))
            instructions.extend(self._gen_print_bool_value(Register('eax')))
        else:  # INT
            instructions.append(Mov(src=Memory('rax', 0), dst=Register('esi')))
            instructions.extend(self._gen_print_int_value(Register('esi')))

        instructions.append(Add(src=Imm(1), dst=Register('r13d')))
        instructions.append(Jmp(loop_start))
        instructions.append(Label(loop_end))
        instructions.extend(self._gen_print_static("]"))
        return instructions

    def gen_string_literal_into(self, expr: StringLiteral, dst: Operand) -> List[Instruction]:
        """Registers this literal's content for later emission as static
        `.data` (see AsmProgram.string_literals / Emitter.emit), and
        loads its address into `dst`. Every occurrence gets its own
        fresh label, even if two literals happen to have identical
        content -- no deduplication, which is a bit wasteful but keeps
        this a one-line append rather than needing a content->label
        cache."""
        label = self.new_label("str")
        self.string_literals.append((label, expr.value))
        return [LeaQ(label=label, dst=as_qword_register(dst))]

    def gen_string_concat_into(self, expr: Binary, dst: Operand) -> List[Instruction]:
        """`left + right`, both str: builds a brand-new, malloc'd,
        null-terminated buffer holding left's bytes immediately
        followed by right's -- `strlen(left) + strlen(right) + 1`
        bytes, then `strcpy` then `strcat`.

        `left` is protected across evaluating `right` by pushing it onto
        the real CPU stack -- the exact same push-before-recursing
        scheme gen_binary_into already uses for ordinary int/bool
        operators -- rather than stashing it in a fixed register like
        %rbx. That distinction matters here specifically because
        `right` can itself be *another* string concatenation or
        comparison (or a call to a function that does one): if `left`
        were sitting in %rbx while `right` gets evaluated, and `right`'s
        own evaluation also needs %rbx for its own left/right dance
        (which it does, being this same method, or via a called
        function that itself calls this method), it would silently
        clobber `left` before this method ever gets to use it. The
        stack has no such fixed-identity conflict, no matter how deeply
        this nests.

        %r12 (holding `right`), by contrast, *is* safe to set as a fixed
        register immediately after `right`'s evaluation completes:
        nothing between that point and this method's own use of %r12
        recurses back into gen_expr_into, so there's no nested
        evaluation left that could still clobber it -- only the fixed
        strlen/malloc/strcpy/strcat sequence below runs, and libc is
        itself SysV-ABI-compliant, so it's required to preserve %r12 as
        a callee-saved register on its own. %r13/%r14 are similarly
        only ever written once, from a direct call result, with nothing
        recursive happening afterward.

        This is a *distinct* concern from why every function's own
        prologue/epilogue also saves/restores %rbx/%r12/%r13/%r14 (see
        gen_function and gen_return) -- that fix protects a value held
        in one of these registers *across a call into another
        function*; this one protects a value held here *across
        evaluating a nested expression within the same function*. Both
        are needed; neither replaces the other.

        MEMORY: once an operand's bytes have been fully copied out
        (strcpy for left, strcat for right), if that operand was itself
        a fresh, unnamed concatenation result -- a Binary(ADD, ...)
        sub-expression, never a named variable, a literal, or a
        function call's return value -- its buffer is immediately
        freed. See _gen_free_if_fresh_concat and the module docstring's
        STRINGS section for exactly why this specific, narrow case is
        safe to free automatically with no broader escape analysis.
        """
        if not isinstance(dst, Register):
            raise CodegenError(f"Binary codegen requires a register destination, got: {dst!r}")
        result = as_qword_register(dst)

        instructions = self.gen_expr_into(expr.left, dst)
        instructions.append(Push(result))                                # save left on the stack
        instructions.extend(self.gen_expr_into(expr.right, dst))         # left is safe regardless of what this does
        instructions.append(MovQ(src=result, dst=Register('r12')))       # r12 = right
        instructions.append(Pop(result))                                 # restore left
        instructions.append(MovQ(src=result, dst=Register('rbx')))       # rbx = left

        instructions.append(MovQ(src=Register('rbx'), dst=Register('rdi')))
        instructions.append(CallInstr('strlen'))
        instructions.append(MovQ(src=Register('rax'), dst=Register('r13')))  # r13 = len(left)

        instructions.append(MovQ(src=Register('r12'), dst=Register('rdi')))
        instructions.append(CallInstr('strlen'))                              # rax = len(right)
        instructions.append(AddQ(src=Register('r13'), dst=Register('rax')))
        instructions.append(AddQ(src=Imm(1), dst=Register('rax')))       # rax = len(left)+len(right)+1
        instructions.append(MovQ(src=Register('rax'), dst=Register('rdi')))
        instructions.append(CallInstr('malloc'))
        instructions.append(MovQ(src=Register('rax'), dst=Register('r14')))  # r14 = new buffer

        instructions.append(MovQ(src=Register('r14'), dst=Register('rdi')))
        instructions.append(MovQ(src=Register('rbx'), dst=Register('rsi')))
        instructions.append(CallInstr('strcpy'))
        # left's bytes are now fully copied into the new buffer -- if
        # left was itself a fresh concatenation result, nothing else
        # can possibly still need it.
        instructions.extend(self._gen_free_if_fresh_concat(expr.left, 'rbx'))

        instructions.append(MovQ(src=Register('r14'), dst=Register('rdi')))
        instructions.append(MovQ(src=Register('r12'), dst=Register('rsi')))
        instructions.append(CallInstr('strcat'))
        # same reasoning, now that right's bytes have been appended.
        instructions.extend(self._gen_free_if_fresh_concat(expr.right, 'r12'))

        instructions.append(MovQ(src=Register('r14'), dst=result))
        return instructions

    def _gen_free_if_fresh_concat(self, operand: Node, holding_register: str) -> List[Instruction]:
        """If `operand` is itself a Binary(ADD, ...) node -- meaning
        whatever's sitting in `holding_register` right now is a fresh
        buffer that gen_string_concat_into just malloc'd for *this*
        expression alone, and which could never have been stored into a
        variable, returned from a function, or passed as an argument
        anywhere, since it only ever existed as this expression's own
        intermediate operand -- frees it. Everything else is left
        alone: a StringLiteral points into static `.data` and was never
        heap-allocated in the first place (freeing it would corrupt the
        allocator); a Variable or a Call's return value might be
        aliased by other code we have no visibility into here (a named
        variable could be read again later, a call's return value could
        be a parameter passed straight through, etc.) -- telling those
        apart from a genuinely fresh, exclusively-owned buffer is a
        real escape-analysis problem this narrow check deliberately
        doesn't attempt to solve. See the module docstring's STRINGS
        section for the fuller reasoning and what's intentionally still
        left leaking as a result.
        """
        if isinstance(operand, Binary) and operand.op == BinaryOp.ADD:
            return [
                MovQ(src=Register(holding_register), dst=Register('rdi')),
                CallInstr('free'),
            ]
        return []

    def gen_string_compare_into(self, expr: Binary, dst: Operand) -> List[Instruction]:
        """`left == right` / `left != right`, both str: calls `strcmp`
        (0 means equal) and converts that into this language's usual
        0/1 bool representation via the exact same cmp/setCC/movzx
        pattern every other comparison already uses -- reusing
        _COMPARISON_CONDITION_CODES[op] directly, since strcmp's result
        is a plain 32-bit int that "compared to 0" behaves exactly like
        any other int comparison from here on.

        `left` is protected across evaluating `right` via the stack, for
        exactly the same reason gen_string_concat_into does -- see its
        docstring.

        MEMORY: the same fresh-concatenation-result freeing
        gen_string_concat_into does, and for the same reason -- strcmp
        has already read both operands' bytes by the time this frees
        them, so there's nothing left that could need them. The one
        thing to get right here that concatenation didn't have to worry
        about: `call free` clobbers %rax/%eax exactly like any other
        call does, and strcmp's result is *sitting* in %eax at this
        point -- so it has to be stashed in a callee-saved register
        before either free() call, and restored into `dst` afterward,
        or freeing a fresh operand would silently destroy the very
        comparison result this method exists to compute.
        """
        if not isinstance(dst, Register):
            raise CodegenError(f"Binary codegen requires a register destination, got: {dst!r}")
        result = as_qword_register(dst)

        instructions = self.gen_expr_into(expr.left, dst)
        instructions.append(Push(result))
        instructions.extend(self.gen_expr_into(expr.right, dst))
        instructions.append(MovQ(src=result, dst=Register('r12')))
        instructions.append(Pop(result))
        instructions.append(MovQ(src=result, dst=Register('rbx')))

        instructions.append(MovQ(src=Register('rbx'), dst=Register('rdi')))
        instructions.append(MovQ(src=Register('r12'), dst=Register('rsi')))
        instructions.append(CallInstr('strcmp'))

        instructions.append(MovQ(src=result, dst=Register('r13')))  # stash strcmp's result before it can be clobbered
        instructions.extend(self._gen_free_if_fresh_concat(expr.left, 'rbx'))
        instructions.extend(self._gen_free_if_fresh_concat(expr.right, 'r12'))
        instructions.append(MovQ(src=Register('r13'), dst=result))  # restore it

        byte_dst = as_byte_register(dst)
        instructions.append(Cmp(src=Imm(0), dst=dst))
        instructions.append(SetCC(cc=_COMPARISON_CONDITION_CODES[expr.op], operand=byte_dst))
        instructions.append(MovZX(src=byte_dst, dst=dst))
        return instructions

    def gen_binary_op(self, op: BinaryOp, src: Operand, dst: Operand) -> List[Instruction]:
        if op == BinaryOp.ADD:
            return [Add(src=src, dst=dst)]
        if op == BinaryOp.SUBTRACT:
            return [Sub(src=src, dst=dst)]
        if op == BinaryOp.MULTIPLY:
            return [IMul(src=src, dst=dst)]
        if op == BinaryOp.DIVIDE:
            # idivl divides %edx:%eax by its operand, so by the time this
            # runs, the dividend (`dst`, i.e. left) must be in %eax and
            # the divisor (`src`, i.e. right) must be in a register --
            # both guaranteed by how gen_binary_into calls this.
            if dst != Register('eax'):
                raise CodegenError("Division currently requires its destination to be %eax")
            return [Cdq(), IDiv(src)]
        if op == BinaryOp.MODULO:
            # Exactly the same Cdq+IDiv sequence as DIVIDE -- idivl
            # always computes both the quotient (%eax) and the remainder
            # (%edx) in one instruction -- just followed by moving the
            # remainder into dst instead of leaving the quotient there.
            if dst != Register('eax'):
                raise CodegenError("Modulo currently requires its destination to be %eax")
            return [Cdq(), IDiv(src), Mov(src=Register('edx'), dst=Register('eax'))]
        if op == BinaryOp.BITWISE_AND:
            return [And(src=src, dst=dst)]
        if op == BinaryOp.BITWISE_OR:
            return [Or(src=src, dst=dst)]
        if op == BinaryOp.BITWISE_XOR:
            return [Xor(src=src, dst=dst)]
        if op == BinaryOp.SHIFT_LEFT:
            # `src` (== %ecx, per gen_binary_into) is never referenced
            # here -- ShiftLeft hardcodes %cl as its count operand,
            # since that's the only register x86 allows there, and %ecx
            # is already where the right-hand operand ends up by the
            # time gen_binary_op is called for any binary operator.
            return [ShiftLeft(dst=dst)]
        if op == BinaryOp.SHIFT_RIGHT:
            return [ShiftRightArithmetic(dst=dst)]
        if op in _COMPARISON_CONDITION_CODES:
            # Cmp(src=right, dst=left) computes (left - right) and sets
            # flags from that; SetCC turns the relevant flag combination
            # into a 0/1 byte; MovZX zero-extends that byte back out to
            # fill the full destination register (same pattern used for
            # NOT -- see gen_unary_op -- just against a computed `right`
            # instead of the literal 0).
            byte_dst = as_byte_register(dst)
            return [
                Cmp(src=src, dst=dst),
                SetCC(cc=_COMPARISON_CONDITION_CODES[op], operand=byte_dst),
                MovZX(src=byte_dst, dst=dst),
            ]
        raise CodegenError(f"No codegen rule for binary operator: {op}")

    def gen_unary_op(self, op: UnaryOp, dst: Operand) -> List[Instruction]:
        if op == UnaryOp.NEGATE:
            return [Neg(dst)]
        if op == UnaryOp.COMPLEMENT:
            return [Not(dst)]
        if op == UnaryOp.NOT:
            # `not x` is "1 if x == 0, else 0" -- the same cmp/setCC/movzx
            # pattern used for comparisons, just always against 0 and
            # always with cc='e'.
            byte_dst = as_byte_register(dst)
            return [
                Cmp(src=Imm(0), dst=dst),
                SetCC(cc='e', operand=byte_dst),
                MovZX(src=byte_dst, dst=dst),
            ]
        raise CodegenError(f"No codegen rule for unary operator: {op}")


# ---------------------------------------------------------------------------
# Assembly AST -> text
# ---------------------------------------------------------------------------

def _escape_for_asciz(s: str) -> str:
    """Escapes `s` (an already-unescaped Hornet string value -- see
    parser.py's _unescape_string_literal) for embedding in a GAS
    `.asciz "..."` directive. Backslash has to be escaped *first*, or
    the escapes added for the other characters would themselves get
    re-escaped; double-quote needs escaping since that's the
    directive's own delimiter; the rest are the common control
    characters getting their standard short escape so the emitted
    assembly stays readable text rather than raw control bytes."""
    s = s.replace('\\', '\\\\')
    s = s.replace('"', '\\"')
    s = s.replace('\n', '\\n')
    s = s.replace('\t', '\\t')
    s = s.replace('\r', '\\r')
    return s


class Emitter:
    """Renders an AsmProgram as textual x64 AT&T-syntax assembly.

    `platform` controls the portability wrinkles that matter at this
    stage of the compiler:
      - macOS (Mach-O) requires a leading underscore on external symbols
        (e.g. `_main`, and now also library calls like `_malloc`); Linux
        (ELF) does not. This applies uniformly to this program's own
        function labels (emit_function) and to Call instruction targets
        (also emit_function, since that's where every instruction gets
        rendered) -- both go through the same symbol() method.
      - Linux toolchains generally expect a `.note.GNU-stack` section so
        the linker doesn't warn about an executable stack; macOS doesn't
        use this.
    """

    def __init__(self, platform: str = 'macos'):
        if platform not in ('macos', 'linux'):
            raise ValueError("platform must be 'macos' or 'linux'")
        self.platform = platform

    def symbol(self, name: str) -> str:
        return f"_{name}" if self.platform == 'macos' else name

    def emit(self, program: AsmProgram) -> str:
        lines: List[str] = []
        for fn in program.functions:
            lines.extend(self.emit_function(fn))
            lines.append("")  # blank line between functions
        if program.string_literals:
            # Plain `.data` rather than a stricter read-only section
            # (like ELF's `.rodata` or Mach-O's `__TEXT,__cstring`) on
            # purpose -- `.data` is the one directive that assembles
            # correctly, unchanged, on both this Linux sandbox and
            # macOS's assembler, and nothing in this language ever
            # writes back into a string literal's bytes anyway, so the
            # extra write-protection those stricter sections would give
            # isn't actually buying anything here.
            lines.append(".data")
            for label, content in program.string_literals:
                lines.append(f"{label}:")
                lines.append(f'    .asciz "{_escape_for_asciz(content)}"')
            lines.append("")
        if self.platform == 'linux':
            lines.append('.section .note.GNU-stack,"",@progbits')
        return "\n".join(lines).rstrip() + "\n"

    def emit_function(self, fn: AsmFunction) -> List[str]:
        sym = self.symbol(fn.name)
        lines = [f"    .globl {sym}", f"{sym}:"]
        for instr in fn.instructions:
            if isinstance(instr, CallInstr):
                # Call.emit() renders its target unprefixed -- platform
                # symbol naming is this Emitter's job alone, same as for
                # this program's own function labels above, so this is
                # the one instruction type emit_function special-cases
                # rather than just calling instr.emit() uniformly.
                lines.append(f"    call    {self.symbol(instr.target)}")
            else:
                lines.append(f"    {instr.emit()}")
        return lines


# ---------------------------------------------------------------------------
# Convenience entry points
# ---------------------------------------------------------------------------

def generate_asm(program: Program, platform: str = 'macos') -> str:
    asm_program = CodeGenerator().generate(program)
    return Emitter(platform=platform).emit(asm_program)


def compile_to_asm(filename: str, platform: str = 'macos') -> str:
    tokens = lex(filename)
    ast = Parser(tokens).parse_program()
    analyze(ast)  # raises SemanticError before any code is generated
    return generate_asm(ast, platform=platform)


def main():
    arg_parser = argparse.ArgumentParser(description='Assembly generator')
    arg_parser.add_argument('file', type=str, help='Source file to compile.')
    arg_parser.add_argument(
        '--platform', choices=['macos', 'linux'], default='macos',
        help="Target platform; affects symbol naming. Default: macos",
    )
    arg_parser.add_argument(
        '-o', '--output', type=str, default=None,
        help='Write assembly to this file instead of stdout.',
    )
    args = arg_parser.parse_args()

    asm = compile_to_asm(args.file, platform=args.platform)
    if args.output:
        with open(args.output, 'w') as f:
            f.write(asm)
    else:
        print(asm, end='')


if __name__ == '__main__':
    main()
