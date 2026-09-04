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
type_of from what semantic.py already resolved, not re-looked-up here.

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

PRINTING
-------------
`print(x)` works for every type -- int, bool, str, array, slice, and
struct -- through a single, uniform pipeline (gen_print_call_into):
allocate a small growable buffer, compute the address of x's own
value, call hornet_stringify(value_addr, type_desc, quote_strings=0,
&buf_state) to append x's own textual representation onto that
buffer, append a trailing newline, write() the result to stdout, then
free() the buffer. See gen_print_call_into's own docstring for the
full step-by-step and exactly why step order matters (the buffer is
allocated BEFORE value_addr is computed, not after, to avoid a value
needing to survive an internal malloc call in a register -- the exact
bug class found and fixed in gen_buffer_append_bytes_into, see its own
docstring).

hornet_stringify itself (build_stringify_function) is a single,
hand-built AsmFunction -- not derived from any Hornet source, and not
duplicated per print() call site -- that recursively converts ANY
value into bytes appended onto a shared buffer, dispatching at
runtime on a small integer KIND tag read from that value's own type
descriptor (_get_or_build_type_descriptor lazily builds one static
descriptor per distinct Type, memoized by structural identity so a
self-referential struct's own descriptor -- e.g. `struct Node: []Node
children` -- terminates correctly: the descriptor's own label is
reserved before recursing into its element type, so the recursive
reference resolves to an already-known label rather than looping
forever). It's added to the program's own function list only if
print() is actually used anywhere (see generate()'s own _print_used
check) -- a program that never prints shouldn't pay for it.

Formatting, by kind:
  int/bool/str: the same as printing one bare -- digits, "true"/
    "false", or (quoted with single quotes, UNLESS this is the
    outermost value of the whole print() call -- see quote_strings)
    the string's own bytes.
  array/slice: `NAME[elem, elem, ...]` -- e.g. `[3]int[1, 2, 3]` or
    `[]int[1, 2, 3]` -- NAME is the type's own name (matching
    semantic.Type.__str__ exactly: "[3]int", "[]int", ...), read out
    of the type descriptor's own second field (see _get_or_build_
    type_descriptor) and printed at EVERY level a value of this kind
    appears, not just the outermost one a print() call names
    directly -- a nested row of a [2][3]int shows its own "[3]int"
    name too (`[2][3]int[[3]int[1, 2, 3], [3]int[4, 5, 6]]`), rather
    than suppressing it just because it's nested.
  struct: `NAME(field: value, field: value, ...)` -- NAME is the
    struct's own declared name (e.g. "Point"), read out of the type
    descriptor the same way, then each field's own declared name,
    ": ", then its value, in declaration order, read at runtime out
    of the struct's own type descriptor (a list of (name, type_desc,
    byte_offset) triples), not unrolled per field at compile time --
    the same "one loop, not one code path per shape" choice
    arrays/slices already made over compile-time unrolling, extended
    one level further. Like array/slice, this name is printed at
    every level, so a struct field that's itself a struct shows its
    own name too (`Outer(inner: Inner(v: 99))`).

Every element or field VALUE nested inside a collection or struct is
printed with quote_strings=1 hardcoded at that specific recursive call
site -- a str nested this way is always quoted, unambiguous next to
its own neighbors, regardless of whether the OUTERMOST print() call's
own argument was quoted (it never is: quote_strings=0 always, at the
one, single call site gen_print_call_into itself makes).

print's own argument, when array-, slice-, or struct-typed, is
restricted to a Variable, Field, or Index -- the same restriction
gen_array_arg_address_into already imposes on array-typed call
arguments, for the same reason: a bare literal, or a call returning
one of these types, has no address of its own to print through, and
(unlike a scalar int/bool/str) these can be arbitrarily large, so
there's no fixed-size scratch slot that could safely hold an
arbitrary one anyway. Assign it to a named variable first.

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

STRUCTS
--------
A struct declares a new, NOMINAL type -- `struct Point: int x; int y`
-- with its own named, ordered, heterogeneous fields, read and written
via `.` (`p.x`, `p.x = 1`). Value semantics throughout, exactly like an
array: copied on VarDecl initialization, plain Assign, parameter
passing, and return, never aliased -- so `q = p` (both Point) makes an
independent copy, and mutating one afterward never affects the other.
This isn't a separate rule invented for struct; it's the SAME rule
arrays already established, just for a different-shaped value.

NOMINAL TYPING, FOR FREE: two structs with identical field lists but
different declared names are different types (`struct A: int v` and
`struct B: int v` don't type-check interchangeably) -- and this falls
out of Type's own existing structural-equality dataclass machinery
with no new mechanism at all. Type gained one new field, struct_name;
two Type(STRUCT, struct_name='Point') instances already compare equal
via ordinary dataclass equality (same name -> same type), and a
different name already compares unequal, without ever needing to
compare the two structs' own field lists against each other. See
semantic.py's own Type docstring for the fuller version of this
argument.

FIELD LAYOUT: sequential byte offsets in declaration order, no padding
or alignment ever inserted between fields (x86-64 doesn't require
aligned access the way some architectures do, the same reasoning
_frame_size's own docstring already gives for why a stack frame's own
locals need none either) -- see _field_offset, which is exactly the
same "sum of what came before" computation type_byte_width itself
already does for a struct's own TOTAL width, just stopping partway
through. A struct can contain another struct as a field (nested
structs), or an array of structs (`[3]Point`) -- both handled by
type_byte_width's own new STRUCT case (sum of type_byte_width over
each field, recursively) with no special-casing needed beyond that one
addition.

THE REGISTRY: a struct's own field list lives in a StructInfo
(semantic.py), keyed by name in a registry dict built once, before
even function signatures are resolved, by SemanticAnalyzer's own
struct-collection pass (see its own docstring for why THAT needs two
internal sub-passes: reserving every name up front is what makes a
forward reference -- struct A, declared first, referencing struct B,
declared later in the same file -- resolve correctly). That registry
is stashed onto Program.struct_registry once analysis finishes, which
is how CodeGenerator gets its own copy (self.struct_registry, read
once at the very start of generate() -- see its own defensive check
for what happens if a caller skips semantic.analyze() entirely) --
the same "resolve once, thread the result through everywhere it's
needed" shape type_from_name's own structs parameter already
established, extended here to codegen.py's own side of that same
boundary. Cycle detection (a struct can never contain itself, directly
or transitively) lives entirely in semantic.py, since it only needs to
reason about which struct names exist and how their fields reference
each other, never about layout or codegen at all -- see _check_struct_
contains's own docstring for exactly which field shapes count as
"containment" (a direct or array-embedded struct field does; a SLICE-
typed one deliberately doesn't, since a slice's own backing storage is
a separate runtime allocation, not embedded inline -- see SLICE-TYPED
FIELDS below for why that distinction matters beyond just cycle
detection).

THE COPY MECHANISM NEEDED NO NEW MACHINERY AT ALL: gen_array_copy's
own flat-byte-chunking loop, generalized to handle ANY leaf width (not
just the three -- 4, 8, 24 -- it used to hardcode) rather than
recursing field by field, is already exactly correct for copying a
struct. This isn't a coincidence or a shortcut: this language has no
reference counting, no copy constructor, and no write barrier
anywhere, so a flat, raw copy of every byte a value occupies is ALWAYS
semantically identical to copying it "as" whatever fields or elements
those bytes represent -- which is exactly why an array-of-slices
element (24 bytes: pointer, then length, then cap, copied as three
sequential movqs) already worked before struct existed at all: that
IS a flat byte copy, already producing the correct shallow, alias-
preserving semantics slice values need everywhere else. A struct
containing a nested array, slice, or another struct needs nothing
more than this same flat copy, for the identical reason.

FIELD ADDRESS COMPUTATION mirrors index address computation one level
over: gen_struct_address_into (a Variable, a Field for a nested chain,
or an Index for a struct-typed array element -- deliberately NOT a
struct-returning Call, matching this file's established restriction on
other unnamed-expression bases; assign it to a named variable first)
and gen_field_address_into (adds the field's own byte offset on top).
gen_array_address_into also gained a Field case of its own, for the
`b.data[0]` shape -- an array-typed FIELD, indexed further -- since
Field can now appear anywhere Variable or Index already could as the
base of an array address computation.

CALLING CONVENTION: identical to an array's in every respect -- a
struct-typed return uses the same hidden-output-pointer convention
(gen_struct_call_into is a thin, separately-named wrapper around gen_
array_call_into, since that method's own body never actually reads its
array_type argument at all -- the callee is the one that knows its own
return type's width and writes exactly that many bytes through the
pointer it receives, regardless of what produced that pointer), and a
struct-typed parameter is copied on entry exactly like an array one is
(including the heap-vs-stack decision below), via the same gen_array_
copy this section already covered. A struct-typed call ARGUMENT passes
its address the same way an array argument does, with one real
addition: it can be a Field (`foo(s.inner)`), which arrays never
needed, since a struct field is a new kind of "named location" arrays
don't have.

HEAP PROMOTION: is_heap_allocated's own size check now covers STRUCT
alongside ARRAY -- a large struct (over _STACK_ARRAY_LIMIT_BYTES) gets
promoted to the heap exactly like a large array would, for the
identical reason (one huge local or parameter blowing the stack on its
own). This is entirely independent of the ESCAPE-based promotion a
slice field's own backing array might separately need -- see SLICE-
TYPED FIELDS below -- the two triggers are combined by CodeGenerator's
own _is_heap_allocated, exactly the way they already are for arrays.

SLICE-TYPED FIELDS (`struct Row: []int values`) are fully supported,
including escape analysis: if a struct value escapes a function, the
array backing any of its slice fields' own values escapes with it,
via analyze_array_escapes's own field_slot_of, added alongside
indexed_slot_of (see AGGREGATES AND SLOTS in its own docstring) --
resolving a Field access the same way indexed_slot_of already resolves
an Index one, down to whatever ROOT Variable underlies the whole
access chain (root_variable_name, now unwrapping Field alongside Index
and Slice), then to that root's own shared aggregate-elements slot.
This is also exactly why cycle detection's own array-counts/slice-
doesn't distinction (see THE REGISTRY above) mattered even while slice
fields were still rejected outright: `struct Node: []Node children` --
a self-referential struct through a slice, a real tree or linked
structure -- is now the genuinely supported pattern that exclusion was
always meant to enable, not something cycle detection would ever have
needed a later carve-out for.
Deliberately ONE combined slot per struct declaration, not a separate
one per distinct field (`p.a` and `p.b` share the same slot, even
though a field name -- unlike a dynamic array index -- is known
statically and so could in principle get its own precise one): true
per-field precision would mean a struct-to-struct copy of just PART of
a struct (`i = outer.inner`) needs to precisely propagate only the
sub-struct's own field slots to `i`'s own, a genuinely larger
mechanism than this analysis's existing "resolve to exactly one node"
shape supports without a much bigger refactor. Lumping every field of
a given declaration into one shared slot instead keeps this an
incremental extension of the exact same machinery indexed_slot_of
already established, at the cost of the identical kind of precision
loss already accepted for array elements (`rows[0]` and `rows[1]`
already share one slot too) -- sound, just coarser than necessary for
two logically-independent slice fields on the same struct. Building
this closed semantic.py's own explicit rejection of slice-typed
fields, which existed for exactly as long as this analysis didn't:
without it, writing a slice into a field, or copying the whole struct
via the ordinary flat byte copy this section already covered, would
have silently compiled with no error at all, while the array backing
that slice could still have been left stack-allocated and outlived
the frame it came from.

Building this surfaced two real, separately-rooted bugs in the
ESCAPE analysis's own pre-existing machinery, both found by testing
(compiling, linking, and running the resulting binary with a large
intervening stack write specifically designed to clobber a wrongly-
stack-allocated array), not by inspection -- neither is specific to
struct fields at all, even though building fields is what surfaced
them:
  - contribution()'s own Slice case used to resolve the thing being
    re-sliced (`rows[0][0:2]`, or the new `p.values[0:2]`) via root_
    variable_name straight to its raw declaration and check plain
    array/slice-declaration-set membership -- correct when the thing
    being re-sliced is a bare Variable, but wrong when it's itself
    reading a slice out of an aggregate: `rows[0][0:2]` resolved to
    `rows` itself (an array declaration, so the membership check
    matched), not to whatever `rows[0]`'s own slice descriptor
    actually points at. Fixed by a new _unwrap_slices helper (unwraps
    just the re-slicing chain, stopping at an Index or Field rather
    than continuing through it the way root_variable_name does) that
    lets contribution() try indexed_slot_of/field_slot_of FIRST, with
    the original root-based check surviving as a fallback for the
    genuinely different case of slicing a plain sub-array row out of
    a multi-dimensional array with no slices involved at all
    (`matrix[1][0:2]`) -- the first version of this fix broke exactly
    that case, caught immediately by the existing test suite.
  - field_slot_of needs an actual Field node (base and name together)
    to do its own resolution, and FieldAssign was initially passed
    directly on the theory that duck-typing (it only ever reads
    .base/.name, which FieldAssign has exactly like Field does) would
    work the same way it already does at other call sites in this
    file. It doesn't here: root_variable_name's own isinstance check,
    which field_slot_of calls into, only recognizes Index/Slice/Field
    -- not FieldAssign -- so passing stmt directly silently failed to
    unwrap anything at all, always returning None. Every write to a
    slice-typed field was consequently untracked. Fixed by
    constructing an actual Field(base=stmt.base, name=stmt.name) at
    the walk_statements call site instead.

A third, unrelated gap surfaced alongside these: gen_field_assign
never had a NoneLiteral short-circuit before dispatching to gen_
slice_value_into, unlike gen_var_decl and gen_assign, which both
already needed one (none's own resolved type, Type.NONE, never equals
the slice type it's flowing into, so gen_slice_value_into's own
dispatch -- which only ever needs the expression itself, since every
OTHER kind of value's resolved type already matches what needs to be
stored -- has no case for it). This was never reachable before slice-
typed fields existed at all, so it was never exercised until now;
fixed the same way the other two call sites already handle it, via
gen_none_into.

STRUCT LITERALS
----------------
`Name(arg1, arg2, ...)` -- e.g. `Point p = Point(3, 4)` for `struct
Point: int x; int y`. Resolved the positional, call-like way this
section used to flag as an open design question, not the brace-
delimited alternative: `Point(3, 4)` needed no new lexer tokens or
parser syntax at all, since it already parses as an ordinary Call node
(see parser.py) -- the ambiguity with an ordinary function call is
resolved entirely at the semantic layer instead, by registry
membership (see semantic.py's own check_struct_literal/check_call
split), which is exactly why this compiler's own lack of a parse-time
symbol table -- the original objection to this syntax -- never actually
mattered: nothing here needs to disambiguate "MyStruct(...)" from an
ordinary call until semantic analysis, by which point the struct
registry already exists. semantic.py's own analyze() guarantees a
struct name and a function name can never collide, so that dispatch is
never genuinely ambiguous.

Positional and exhaustive: exactly one argument per field, in
declaration order -- no named arguments, no partial construction with
an implicit zero value for an omitted field. Deliberately scoped
narrower than an ordinary function call is allowed to appear: valid
only as a VarDecl's own initializer or a plain Assign's own value, not
as a function argument, a return value, an IndexAssign/FieldAssign
value, nested inside another expression (including as an argument to
ANOTHER struct literal -- `Outer(Inner(1, 2), 3)` is rejected; build
the inner value in its own variable first), or a bare statement --
enforced entirely in semantic.py (see check_struct_literal's own
docstring for the exact mechanism), so by the time an AST reaches this
file, a struct-literal Call can only ever appear in one of those two
positions. gen_struct_value_into checks for this shape first (Call
where expr.name names a struct, not a function) and routes it to gen_
struct_literal_into, which writes each argument directly into its own
field's offset -- the same per-field, address-plus-offset writing
gen_array_literal_into already established for an array literal's own
elements, one level over, dispatching to gen_array_value_into/gen_
slice_value_into/gen_struct_value_into for a composite field (each of
which already protects an arbitrary destination base internally) or a
plain gen_expr_into-then-store for a scalar one, protected the same
way gen_array_literal_into's own scalar case already is.

Still out of scope, unrelated to the syntax question above:
  - `==` ON STRUCTS. Two structs' equality isn't checked or generated
    at all yet -- deferred rather than building field-by-field
    structural comparison for a phase that doesn't need it yet.

(`print` on a struct was ALSO deliberately deferred at this point in
the project's own history -- printing then still worked by building a
fixed format string per call site, which didn't extend cleanly to a
struct's own arbitrarily-nested field structure. That gap has since
been closed: see the module docstring's own PRINTING section above
for the real, growable-buffer-based mechanism -- hornet_stringify --
that now backs every type's own print output, struct included.)
"""


import argparse
from typing import Dict, List, Optional, Tuple, Union

from codegen.assembly_ast import (
    Add,
    AddQ,
    And,
    AndQ,
    AsmFunction,
    AsmProgram,
    CallInstr,
    Cdq,
    Cmp,
    CmpQ,
    Cqto,
    IDiv,
    IDivQ,
    Imm,
    IMul,
    IMulQ,
    Instruction,
    Ja,
    Jae,
    Je,
    Jle,
    Jmp,
    Jne,
    Label,
    LeaQ,
    LeaQFrame,
    Leave,
    Memory,
    Mov,
    MovB,
    MovQ,
    MovZX,
    MovSX,
    MovSXD,
    Neg,
    NegQ,
    Not,
    NotQ,
    Operand,
    Or,
    OrQ,
    Pop,
    Push,
    Register,
    Ret,
    SetCC,
    ShiftLeft,
    ShiftLeftQ,
    ShiftRightArithmetic,
    ShiftRightArithmeticQ,
    Sub,
    SubQ,
    Xor,
    XorQ,
)
from codegen.calling_convention import CALLEE_SAVED_SCRATCH_REGISTERS, CallingConventionMixin
from codegen.emitter import Emitter
from codegen.errors import CodegenError
from codegen.escape_analysis import analyze_array_escapes, is_heap_allocated
from codegen.statements import StatementsMixin
from codegen.strings import StringsMixin
from codegen.structs import StructsMixin
from codegen.utils import as_byte_register, as_qword_register, leaf_type, type_byte_width, type_of, \
    ARG_REGISTERS_64, COMPARISON_CONDITION_CODES
from lexer import lex
from parser import (
    ArrayLiteral,
    Assign,
    Binary,
    BinaryOp,
    BoolLiteral,
    Break,
    Call,
    Cast,
    Constant,
    Continue,
    ExprStmt,
    Field,
    FieldAssign,
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
from semantic import analyze, type_from_name, Type, TypeKind, StructInfo


# ---------------------------------------------------------------------------
# AST -> Assembly AST
# ---------------------------------------------------------------------------

class CodeGenerator(CallingConventionMixin, StatementsMixin, StringsMixin, StructsMixin):
    """Walks the source AST (Program/Function/Return/Constant/...) and
    produces an equivalent AsmProgram."""

    def __init__(self):
        self._label_count = 0
        self._var_offsets: Dict[int, int] = {}  # id(VarDecl node) -> its permanent Memory offset
        self._next_offset = 0
        self.scopes: List[Dict[str, tuple]] = []  # name -> (offset, Type), generation-time; see LOCAL VARIABLES
        self.loop_labels: List[tuple] = []  # stack of (start_label, end_label), innermost last; see LOOPS
        self.string_literals: List[tuple] = []  # (label, content) pairs; see STRINGS
        self.type_descriptors: List[tuple] = []  # (label, fields) pairs; see PRINTING and AsmProgram's own docstring
        # Set once, at the very start of generate(), from Program.
        # struct_registry (itself stashed there by semantic.analyze --
        # see SemanticAnalyzer.analyze's own struct-collection pass).
        # Declared here defensively (an empty dict, not left unset) so
        # a bug that somehow calls a method needing this before
        # generate() itself runs fails with a clear "unknown struct"
        # or "no such field" error rather than an AttributeError from
        # nowhere.
        self.struct_registry: Dict[str, StructInfo] = {}
        # Set the same way, from Program.type_alias_registry (stashed
        # there by semantic.analyze -- see SemanticAnalyzer.analyze's
        # own type-alias-collection pass, which runs before struct
        # collection even starts). Every entry is already a fully-
        # resolved Type by this point -- type_from_name just does a
        # single dict lookup with this, never a recursive re-resolution
        # of an alias's own target.
        self.type_alias_registry: Dict[str, Type] = {}
        # Lazily created, then cached and reused for the rest of this
        # compilation -- see gen_print_call_into and the module
        # docstring's BUILTINS section for why these specifically (and
        # only these) get a small dedicated cache rather than following
        # string_literals' usual "every occurrence gets its own label,
        # no dedup" policy.
        self._true_str_label = None
        self._false_str_label = None
        self._comma_space_label = None  # ", " -- the print machinery's own element/field separator
        self._colon_space_label = None  # ": " -- between a struct field's own name and its value
        self._empty_str_label = None  # "" -- str's own zero value; see _get_empty_str_label
        # Set the first time gen_print_call_into actually runs; checked
        # in generate() to decide whether hornet_stringify itself needs
        # to be added to the program's own function list at all --
        # a program that never calls print() shouldn't pay for it.
        self._print_used = False
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
        # getattr, not direct attribute access: Program.struct_registry
        # is stamped on by semantic.analyze() (see SemanticAnalyzer.
        # analyze's own struct-collection pass), not a field the
        # dataclass itself declares -- an AST that skipped analyze()
        # entirely simply won't have it at all. Matching type_of's own
        # "has no resolved type" defensive check one level up: fail
        # with a clear, actionable CodegenError right here, at the
        # very first thing generate() does, rather than a bare
        # AttributeError from whatever the first struct-registry
        # lookup happens to be further down.
        if not hasattr(program, 'struct_registry'):
            raise CodegenError(
                "Program has no struct registry -- semantic.analyze() "
                "must run before codegen (see compile_to_asm)"
            )
        self.struct_registry = program.struct_registry
        # Same defensive check, same reason, one registry over -- see
        # this attribute's own docstring at its declaration.
        if not hasattr(program, 'type_alias_registry'):
            raise CodegenError(
                "Program has no type alias registry -- semantic.analyze() "
                "must run before codegen (see compile_to_asm)"
            )
        self.type_alias_registry = program.type_alias_registry
        functions = [self.gen_function(fn) for fn in program.functions]
        if self._print_used:
            functions.append(self.build_stringify_function())
        return AsmProgram(functions=functions, string_literals=self.string_literals, type_descriptors=self.type_descriptors)

    def gen_function(self, fn: Function) -> AsmFunction:
        # Fresh allocator state per function -- variables don't persist
        # across functions, and offsets are relative to *this*
        # function's own %rbp.
        self._var_offsets = {}
        self._argument_temp_offsets = {}  # id(ArrayLiteral or Call) -> its permanent slot; see _collect_argument_temps
        self._next_offset = 0
        # No declared return type (fn.return_type is None -- see
        # Function's own docstring in parser.py) means Type.VOID, the
        # same internal-only sentinel semantic.py's own analyze_function
        # already uses -- kept consistent here rather than reinventing
        # a second "no return type" representation in this file.
        return_type = Type.VOID if fn.return_type is None else type_from_name(fn.return_type, self.struct_registry, self.type_alias_registry)
        param_types = [type_from_name(p.type, self.struct_registry, self.type_alias_registry) for p in fn.params]

        # Which of THIS function's own array declarations need to be
        # heap-allocated because a slice backed by them might outlive
        # this function's own return, regardless of their own size --
        # see analyze_array_escapes's own docstring for the full
        # algorithm. Computed once, up front, since _collect_params/
        # _collect_locals (just below) already need to know this to
        # decide how much stack space each declaration's own slot
        # takes (8 bytes for a heap pointer vs. the array's own full
        # width) -- this has to exist before either of them run.
        self._escaping_array_ids = analyze_array_escapes(fn, param_types, self.struct_registry, self.type_alias_registry)

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
        if return_type.kind in (TypeKind.ARRAY, TypeKind.SLICE, TypeKind.STRUCT):
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

        # A third, small (8-byte) scratch slot -- also reserved
        # unconditionally for every function -- used by gen_print_
        # call_into to materialize a non-Variable int/bool/str
        # argument (e.g. `print(x + 1)`, `print(a.name)`) so hornet_
        # stringify always has a real address to read from, exactly
        # the same "one shared slot, safe because each use is fully
        # consumed before the next one can start" reasoning as the
        # unnamed-slice slot just above -- see gen_print_call_into's
        # own docstring for why array/slice/struct arguments don't
        # need (and can't safely share) a slot like this one: those
        # can be arbitrarily large, so print requires a Variable or
        # Index for them instead of ever materializing a copy.
        self._next_offset -= 8
        self._print_scalar_temp_offset = self._next_offset

        # A fourth scratch slot (24 bytes) -- the {ptr, len, cap}
        # triple gen_print_call_into's own growable buffer lives in.
        # Reserved unconditionally alongside the others above, for the
        # same reason: a print() call's own buffer setup needs
        # somewhere real to live for the duration of one print() call,
        # and reusing a single shared slot is safe by the same
        # non-overlapping-lifetime argument -- one print() call's own
        # buffer is fully written out and freed before the next one
        # (or any nested print inside a printed expression's own
        # evaluation, though print itself never returns a value that
        # could be printed again) could possibly start using this slot.
        self._next_offset -= 24
        self._print_buf_state_temp_offset = self._next_offset

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
        # A THIRD pre-pass, alongside the two above: finds every array-
        # or struct-typed function-call argument that has no address
        # of its own -- an ArrayLiteral, a struct literal, or an
        # ordinary array/struct-returning Call used directly as an
        # argument (`foo([1,2,3])`, `foo(A(1,2))`, `foo(bar())`) --
        # anywhere in this function's body, however deeply nested
        # inside other expressions, and reserves each one its own
        # permanent stack slot up front, sized to fit. See _collect_
        # argument_temps's own docstring for why this can't reuse the
        # single-shared-slot trick _unnamed_slice_temp_offset above
        # relies on, and gen_array_arg_address_into/_gen_call_
        # arguments_into's own STRUCT branch for where these slots
        # actually get read back out.
        self._collect_argument_temps(fn.body)
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
        for reg in CALLEE_SAVED_SCRATCH_REGISTERS:
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
                instructions.append(MovQ(src=Register(ARG_REGISTERS_64[reg_index]), dst=Memory('rbp', param_temp_offsets[i])))
                instructions.append(MovQ(src=Register(ARG_REGISTERS_64[reg_index + 1]), dst=Memory('rbp', param_temp_offsets[i] + 8)))
                instructions.append(MovQ(src=Register(ARG_REGISTERS_64[reg_index + 2]), dst=Memory('rbp', param_temp_offsets[i] + 16)))
                reg_index += 3
            else:
                instructions.append(MovQ(src=Register(ARG_REGISTERS_64[reg_index]), dst=Memory('rbp', param_temp_offsets[i])))
                reg_index += 1

        for i, p in enumerate(fn.params):
            offset = self._bind_param(p)
            p_type = param_types[i]
            temp_offset = param_temp_offsets[i]
            if p_type.kind in (TypeKind.ARRAY, TypeKind.STRUCT):
                if self._is_heap_allocated(id(p), p_type):
                    # Needs its own, independent heap copy -- exactly
                    # like the stack-allocated case below, just backed
                    # by malloc'd memory instead of an inline slot --
                    # to preserve value semantics across the call:
                    # mutating this parameter must never affect the
                    # caller's own array or struct. %rbx holds the
                    # caller's pointer across the malloc call itself:
                    # it's callee-saved, so malloc (a well-behaved,
                    # ABI-conforming function) is obligated to preserve
                    # it, the same guarantee gen_string_concat_into's
                    # own malloc/strlen/strcpy calls already rely on.
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
                instructions.extend(self._gen_read_scalar_into(Memory('rbp', temp_offset), p_type, Register('eax')))
                instructions.extend(self._gen_write_scalar_from(Register('eax'), p_type, Memory('rbp', offset)))

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
        parameter's own actual type width (see type_byte_width) -- 1
        byte for int8/uint8, 4 for int/bool, 8 for str, and an array's
        own full, flattened footprint for a stack-allocated array
        parameter --
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
            p_type = type_from_name(p.type, self.struct_registry, self.type_alias_registry)
            width = 8 if self._is_heap_allocated(id(p), p_type) else type_byte_width(p_type, self.struct_registry)
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
        self.scopes[-1][p.name] = (offset, type_from_name(p.type, self.struct_registry, self.type_alias_registry), id(p))
        return offset

    def _collect_locals(self, statements: List[Node]) -> None:
        """Recursively walks `statements`, including into every If's
        then_body/else_body and every While's body, and gives each
        VarDecl found its own permanent stack slot, keyed by the AST
        node's identity rather than its name -- see the module
        docstring's LOCAL VARIABLES section for why that distinction
        now matters.

        Each slot's width is the variable's own actual type width (see
        type_byte_width) -- 1 byte for int8/uint8, 4 for int/bool, 8
        for str, and an array's own full, flattened footprint (e.g. 24
        bytes for [2][3]int) for a stack-allocated array local. Uniform 8-byte
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
                var_type = type_from_name(stmt.var_type, self.struct_registry, self.type_alias_registry)
                width = 8 if self._is_heap_allocated(id(stmt), var_type) else type_byte_width(var_type, self.struct_registry)
                self._next_offset -= width
                self._var_offsets[id(stmt)] = self._next_offset
            elif isinstance(stmt, If):
                self._collect_locals(stmt.then_body)
                if stmt.else_body is not None:
                    self._collect_locals(stmt.else_body)
            elif isinstance(stmt, While):
                self._collect_locals(stmt.body)

    def _collect_argument_temps(self, statements: List[Node]) -> None:
        """Recursively walks `statements` -- including into every If's
        then_body/else_body and every While's body, exactly like
        _collect_locals above -- looking for a function-call argument
        that is array- or struct-typed but has no address of its own:
        an ArrayLiteral, a struct literal, or an ordinary array- or
        struct-returning Call used DIRECTLY as an argument (`foo([1,
        2,3])`, `foo(A(1,2))`, `foo(bar())`) -- as opposed to a
        Variable, Index, or Field, each of which already has a real
        address via gen_array_address_into/gen_struct_address_into and
        so never needs one of these.

        Not just ORDINARY function-call arguments, despite the name:
        _collect_argument_temps_in_expr's own walk finds a qualifying
        argument inside ANY Call node, with no check on `expr.name` at
        all -- print, len, and append are all ordinary Call nodes as
        far as this pass is concerned, so an array-typed literal or
        returning-call passed to print() already gets a slot reserved
        here, unbeknownst to this pass itself, long before print's own
        codegen (gen_print_call_into) actually learned to make use of
        one. (len's and append's own array/struct-typed arguments, if
        they're ever a literal or returning-call, ALSO get a slot
        reserved here that neither currently reads back out --
        harmless, just a few unused bytes of frame space, since
        neither has a reason to use this mechanism: gen_indexable_
        base_into already heap-allocates an ArrayLiteral base
        unconditionally for len, and gen_array_value_into/gen_struct_
        value_into already handle append's own value argument directly
        without needing an address-yielding temp at all.)

        WHY THIS CAN'T REUSE THE SHARED-SLOT TRICK
        -------------------------------------------------
        _unnamed_slice_temp_offset above gets away with ONE shared,
        per-function scratch slot reused for every unnamed slice,
        because a slice's own 24-byte descriptor is written into it and
        then immediately drained into registers -- by the time
        anything else could reuse the slot, nothing still needs to
        read from it (see gen_indexable_base_into's own docstring for
        the full argument). An array or struct argument is different
        in kind: it's passed BY ADDRESS, and that address has to keep
        pointing at valid data right up until the moment the actual
        `call` instruction executes, since the callee only reads
        through it after control transfer, inside its own prologue.
        That rules out a single shared slot the same way slices use
        one: a single call can have MORE THAN ONE such argument at
        once (`foo([1,2], [3,4])`, or `foo(A(1,2), B(3,4))`), and both
        need to be alive simultaneously, all the way through the call
        -- a shared slot would let the second one's write clobber the
        first's before `call` ever runs. So each occurrence needs its
        OWN, genuinely distinct backing storage, discovered ahead of
        time (this pass) the same way every named local already is.

        SIZE THRESHOLD, MATCHING EVERY OTHER ARRAY/STRUCT VALUE
        -------------------------------------------------------------
        Not every occurrence found here actually gets a stack slot:
        _reserve_argument_temp applies the exact same is_heap_
        allocated size check every named local/parameter already goes
        through. A small literal or returning-call's result gets a
        real, permanent slot, reserved here and read back out by
        _gen_materialize_argument_temp_into. A large one gets NO slot
        at all -- it's heap-allocated fresh at the point of the call
        instead (same method), which needs no space reserved in this
        function's own frame, unlike a large NAMED local's heap
        pointer (which still needs 8 bytes to remember the pointer for
        as long as the variable stays in scope): an argument-temp's
        pointer is read exactly once, by the callee's own entry-time
        copy, and never again afterward, so there's nothing here that
        needs to survive past the `call` itself.

        WHY THIS WALKS EXPRESSIONS, NOT JUST STATEMENTS
        -------------------------------------------------------
        Unlike _collect_locals, which only ever needs to look at
        statement-level constructs (a VarDecl can't hide inside an
        expression), a literal-or-returning-call-as-argument can be
        buried arbitrarily deep inside another expression entirely
        unrelated to the call itself -- `int x = foo(1) + bar([1, 2,
        3])`, or `[foo([1, 2, 3]), 5, 6]` (an array literal whose own
        element happens to be a call that itself takes one) -- so this
        needs a real, general expression walk (_collect_argument_
        temps_in_expr) rather than only inspecting a statement's own
        top-level shape."""
        for stmt in statements:
            if isinstance(stmt, VarDecl):
                if stmt.init is not None:
                    self._collect_argument_temps_in_expr(stmt.init)
            elif isinstance(stmt, Assign):
                self._collect_argument_temps_in_expr(stmt.value)
            elif isinstance(stmt, IndexAssign):
                self._collect_argument_temps_in_expr(stmt.array)
                self._collect_argument_temps_in_expr(stmt.index)
                self._collect_argument_temps_in_expr(stmt.value)
            elif isinstance(stmt, FieldAssign):
                self._collect_argument_temps_in_expr(stmt.base)
                self._collect_argument_temps_in_expr(stmt.value)
            elif isinstance(stmt, Return):
                if stmt.value is not None:
                    self._collect_argument_temps_in_expr(stmt.value)
            elif isinstance(stmt, If):
                self._collect_argument_temps_in_expr(stmt.condition)
                self._collect_argument_temps(stmt.then_body)
                if stmt.else_body is not None:
                    self._collect_argument_temps(stmt.else_body)
            elif isinstance(stmt, While):
                self._collect_argument_temps_in_expr(stmt.condition)
                self._collect_argument_temps(stmt.body)
            elif isinstance(stmt, ExprStmt):
                self._collect_argument_temps_in_expr(stmt.expr)
            # Break/Continue carry no expressions at all.

    def _collect_argument_temps_in_expr(self, expr: Optional[Node]) -> None:
        """The general expression-tree walk _collect_argument_temps
        needs but _collect_locals never did -- recurses into every
        expression node that can contain another expression (Binary,
        Unary, Index, Field, Slice, ArrayLiteral's own elements, and a
        Call's own arguments), with a leaf case for everything else
        (Constant/BoolLiteral/StringLiteral/NoneLiteral/Variable, none
        of which can contain a nested Call at all).

        The actual DECISION -- does this specific Call argument need
        its own reserved slot -- is made only at a Call node: after
        recursing into each of ITS OWN arguments first (so a call
        nested inside another call's argument, `foo(bar([1,2,3]))`, is
        discovered on the way back up, innermost first, though nothing
        about reservation order actually depends on that), any
        argument that's array- or struct-typed and isn't a Variable,
        Index, or Field gets handed to _reserve_argument_temp. A
        Variable/Index/Field argument is skipped entirely here -- it
        already has a real address of its own (see gen_array_address_
        into/gen_struct_address_into), so it was never a candidate for
        one of these slots in the first place."""
        if expr is None:
            return
        if isinstance(expr, Call):
            for arg in expr.args:
                self._collect_argument_temps_in_expr(arg)
                arg_type = type_of(arg)
                if arg_type.kind in (TypeKind.ARRAY, TypeKind.STRUCT) and not isinstance(arg, (Variable, Index, Field)):
                    self._reserve_argument_temp(arg, arg_type)
        elif isinstance(expr, Binary):
            self._collect_argument_temps_in_expr(expr.left)
            self._collect_argument_temps_in_expr(expr.right)
        elif isinstance(expr, Unary):
            self._collect_argument_temps_in_expr(expr.operand)
        elif isinstance(expr, Index):
            self._collect_argument_temps_in_expr(expr.array)
            self._collect_argument_temps_in_expr(expr.index)
        elif isinstance(expr, Field):
            self._collect_argument_temps_in_expr(expr.base)
        elif isinstance(expr, Slice):
            self._collect_argument_temps_in_expr(expr.array)
            self._collect_argument_temps_in_expr(expr.low)
            self._collect_argument_temps_in_expr(expr.high)
        elif isinstance(expr, ArrayLiteral):
            for element in expr.elements:
                self._collect_argument_temps_in_expr(element)
        # Constant/BoolLiteral/StringLiteral/NoneLiteral/Variable: leaves,
        # nothing further to recurse into.

    def _reserve_argument_temp(self, expr: Node, t: Type) -> None:
        """Reserves a permanent stack slot for `expr` -- an ArrayLiteral,
        a struct literal, or an ordinary array/struct-returning Call
        used directly as a function-call argument -- keyed by id(expr)
        exactly like _var_offsets keys a VarDecl/Param, just for a
        synthetic, unnamed "declaration" that corresponds to no actual
        source-level variable.

        Skips reservation entirely when `t` is over the same is_heap_
        allocated size threshold every named local/parameter already
        uses: a large value gets heap-allocated fresh at the point of
        the call instead (see _gen_materialize_argument_temp_into),
        needing no space in this function's own frame at all -- unlike
        a large NAMED local, whose heap pointer still needs a permanent
        8-byte slot to survive for as long as the variable stays in
        scope, an argument-temp's pointer is read exactly once, by the
        callee's own entry-time copy, and never again -- there's
        nothing here that needs to outlive the call itself.

        Deliberately not routed through _is_heap_allocated (which also
        consults self._escaping_array_ids, keyed by a VarDecl/Param's
        own id): an argument-temp is never a candidate for escape-
        driven promotion at all -- it's never sliced by the caller (it
        flows into the callee as a whole value, copied on entry, per
        the module docstring's ARRAYS/STRUCTS sections), so only the
        plain size check ever applies to it, via is_heap_allocated
        directly."""
        if is_heap_allocated(t, self.struct_registry):
            return
        width = type_byte_width(t, self.struct_registry)
        self._next_offset -= width
        self._argument_temp_offsets[id(expr)] = self._next_offset

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
        self.scopes[-1][stmt.name] = (offset, type_from_name(stmt.var_type, self.struct_registry, self.type_alias_registry), id(stmt))
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
        which codegen has to maintain regardless of type_of's
        existence, since resolved_type has no way to encode *which*
        stack slot a name refers to. This is deliberately not replaced
        by type_of, even though it would give the same answer
        for a Variable node -- see type_of's own docstring for why the
        two coexist rather than one replacing the other.

        Returns a real semantic.Type (via type_from_name, called once
        up front in _bind_local/_bind_param, not re-derived here) --
        not the raw parser-level string/ArrayTypeExpr -- so callers can
        uniformly inspect .kind/.element_type/.size exactly like they
        already can on whatever type_of returns."""
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

    def _is_heap_allocated(self, decl_id: int, t: Type) -> bool:
        """Whether the SPECIFIC array- or struct-typed declaration
        identified by decl_id (id() of its own VarDecl or Param node)
        needs to be heap-allocated -- combining is_heap_allocated's
        own, pure size check (now covering both array and struct) with
        analyze_array_escapes's own, independent result (computed once
        per function, in gen_function, and cached in self._escaping_
        array_ids -- an array-specific trigger only, since the actual,
        terminal backing storage a slice descriptor ever points at is
        always a real array, never a struct directly, regardless of
        whether the slice was reached through an array-of-slices
        container or a struct's own slice-typed field: either kind of
        container might itself need promoting for size (is_heap_
        allocated's own check), but never merely because a slice
        somewhere within it escapes -- see analyze_array_escapes's own
        AGGREGATES AND SLOTS section for the full reasoning): either
        reason alone is sufficient. This is the actual decision point
        every one of this file's call sites that used to call is_heap_
        allocated directly now goes through instead, each passing
        whichever decl_id it has on hand -- id(a VarDecl or Param)
        directly, or self._local_decl_id(name) wherever only a
        Variable's own name is available at that point."""
        return is_heap_allocated(t, self.struct_registry) or decl_id in self._escaping_array_ids

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
        array_type = type_of(expr)
        width = max(1, type_byte_width(array_type, self.struct_registry))
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
        RETURNS section for why that's no longer true), an Index
        yielding a slice (`rows[0][1]`, one element of an array OF
        slices used directly as a further base -- materialized into
        that same scratch slot too, via gen_slice_value_into's own
        Index case), or a Field yielding a slice (`p.values[0]`, a
        struct's own slice-typed field used directly as a further
        base -- materialized into that same scratch slot, via gen_
        slice_value_into's own Field case, the identical mechanism
        one level over).

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
        base_type = type_of(expr)
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
            if isinstance(expr, Field):
                # A slice-typed Field result (e.g. `p.values`, a
                # struct's own slice-typed field, used directly as the
                # base of a further `[...]`) -- structurally identical
                # to the Index case just above, one level over:
                # materialized through the exact same shared scratch
                # slot, via gen_slice_value_into's own Field case, then
                # immediately read back out the same way.
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
        Variable referring to an array-typed local, an Index node
        that itself resolves to a sub-array (the outer dimensions of a
        multi-dimensional access), or a Field node that resolves to an
        array-typed struct field (`b.data`, then indexed further as
        `b.data[0]`) -- into the 64-bit register `dst`. `dst` must
        already be a 64-bit register (e.g. Register('rax'), not
        Register('eax')) -- addresses are always 64-bit values,
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
            if self._is_heap_allocated(self._local_decl_id(expr.name), array_type):
                return [MovQ(src=Memory('rbp', offset), dst=dst)]
            return [LeaQFrame(offset=offset, dst=dst)]
        if isinstance(expr, Index):
            return self.gen_index_address_into(expr, dst)
        if isinstance(expr, Field):
            return self.gen_field_address_into(expr, dst)
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
        array_type = type_of(expr.array)
        element_stride = type_byte_width(array_type.element_type, self.struct_registry)

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

        base_type = type_of(expr.array)
        element_stride = type_byte_width(base_type.element_type, self.struct_registry)

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
          - Field (e.g. `[]int r = p.values`, reading a slice-typed
            STRUCT FIELD): structurally identical to the Index case
            just above, one level over -- the field's own address is
            computed first (gen_field_address_into), then its 24-byte
            descriptor is read through it the same way.
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

        if isinstance(expr, Field):
            instructions = []
            if protect_dst:
                instructions.append(Push(Register(dst_mem.base)))
            addr_reg = Register('r11')
            instructions.extend(self.gen_field_address_into(expr, addr_reg))
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
        array, slice, or struct.

        For an ARRAY, SLICE, or STRUCT element type, this just hands
        addr_reg straight to gen_array_value_into/gen_slice_value_
        into/gen_struct_value_into as an ordinary Memory destination --
        all three already protect an arbitrary base internally (see
        their own docstrings), so there's nothing extra to do here;
        mirrors gen_array_literal_into's own identical three-way
        dispatch for a literal's per-element writing, one level over
        (append's own newly-appended element, rather than a literal's
        directly-written one). For a scalar (int/bool/str), addr_reg
        is protected manually, matching gen_array_literal_into's own
        scalar-element pattern exactly: push addr_reg, compute the
        value (which could itself involve a function call that
        clobbers addr_reg, if value_expr is arbitrarily complex),
        stash the computed value in %r8/%r8d (a register distinct from
        addr_reg in every actual call site), pop addr_reg back, then
        write from %r8/%r8d -- never straight from %eax/%rax, which
        popping addr_reg back into would otherwise have to clobber."""
        if element_type.kind == TypeKind.SLICE:
            return self.gen_slice_value_into(value_expr, Memory(addr_reg.name, 0))
        if element_type.kind == TypeKind.ARRAY:
            return self.gen_array_value_into(value_expr, Memory(addr_reg.name, 0), element_type)
        if element_type.kind == TypeKind.STRUCT:
            return self.gen_struct_value_into(value_expr, Memory(addr_reg.name, 0), element_type)
        instructions = [Push(addr_reg)]
        instructions.extend(self.gen_expr_into(value_expr, Register('eax')))
        if element_type == Type.STR:
            instructions.append(MovQ(src=Register('rax'), dst=Register('r8')))
            instructions.append(Pop(addr_reg))
            instructions.append(MovQ(src=Register('r8'), dst=Memory(addr_reg.name, 0)))
        else:
            # A full 64-bit shuttle for int64 -- an ordinary 32-bit Mov
            # would discard its own high 32 bits before ever reaching
            # the final write below. The final write itself goes
            # through _gen_write_scalar_from (not a bare Mov, which
            # this used to be) so int8/uint8 get their own correct,
            # narrow, 1-byte write too -- a bare 4-byte Mov here was a
            # latent, if not directly observed, bug for them as well:
            # writing 4 bytes at a 1-byte element's own offset can
            # write past a freshly-grown backing array's own allocated
            # capacity, not just harmlessly into extra headroom.
            if element_type == Type.INT64:
                instructions.append(MovQ(src=Register('rax'), dst=Register('r8')))
            else:
                instructions.append(Mov(src=Register('eax'), dst=Register('r8d')))
            instructions.append(Pop(addr_reg))
            instructions.extend(self._gen_write_scalar_from(Register('r8d'), element_type, Memory(addr_reg.name, 0)))
        return instructions

    def _gen_grow_and_append_one_into(
        self,
        r_ptr: Register, r_len: Register, r_len_32: Register,
        r_cap: Register, r_cap_32: Register,
        element_width: int,
        copy_one_element,
        write_new_value_at,
    ) -> List[Instruction]:
        """The reuse-vs-reallocate growth core shared by gen_append_
        call_into and (see gen_buffer_append_byte_into) the print
        buffer's own single-byte append -- factored out of what used
        to be gen_append_call_into's own, sole implementation of this
        exact policy, specifically so a second, parallel copy of the
        identical growth arithmetic never has to exist. See gen_
        append_call_into's own docstring for the growth-policy
        arithmetic itself (new_cap = cap*2 if cap < 256, else cap +
        cap//4, with a cap==0 floor of 1) and the reuse-vs-reallocate
        reasoning behind it -- unchanged here, just no longer
        duplicated.

        Operates entirely on registers the caller has already loaded
        with a live {ptr, len, cap} triple (both the 64-bit register
        and its own 32-bit view, passed explicitly as a pair for each
        of len/cap, matching how every other place in this file that
        needs both already keeps them as two separate Register values
        rather than deriving one from the other) -- never on a Memory
        location directly. Loading the initial triple from wherever it
        actually lives, and writing the final one back out afterward
        (or, for the print buffer's own case, simply leaving it live in
        registers across many further appends without ever touching
        Memory at all), is entirely the caller's own responsibility.

        Internally fixed scratch, matching gen_append_call_into's own
        original, unfactored version exactly: %r8/%r8d, %r9d, %r10/
        %r10d, %r11/%r11d, %eax/%ecx, and %r14. Callers must choose
        their own ptr/len/cap registers to avoid these.

        `copy_one_element(dst_addr, src_addr)` is called once per
        existing element, ONLY on the reallocate path, to move it from
        the old backing array into the new one -- gen_append_call_into
        passes one that calls gen_array_copy for element_type; the
        print buffer passes one that just moves a single byte.

        `write_new_value_at(target_addr)` is called exactly once, at
        the correct final address for the newly appended element (in
        whichever backing array ends up live -- the original, on the
        reuse path, or the freshly malloc'd one, on the reallocate
        path) -- gen_append_call_into passes one that evaluates an
        arbitrary Hornet expression via _gen_write_value_at_address_
        into; the print buffer passes one that writes an already-
        computed raw byte operand directly. Neither callback needs to
        know how the other one works, or how growth itself is decided.

        On return, r_ptr/r_len/r_cap (and their 32-bit views) hold the
        FINAL triple: len always incremented by exactly one, ptr
        repointed to a fresh block if reallocation happened,
        unchanged otherwise."""
        instructions = []
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
        instructions.extend(write_new_value_at(target_addr))
        instructions.append(Add(src=Imm(1), dst=r_len_32))
        instructions.append(Jmp(end_label))

        # REALLOCATE PATH: len == cap. new_cap computed from cap alone,
        # in place -- the old cap value is never needed again once
        # this decides new_cap, so overwriting r_cap_32 here is safe.
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
        # into the NEW one (r_new_ptr) via copy_one_element, a genuine
        # RUNTIME loop since len is a runtime value here.
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
        instructions.extend(copy_one_element(Register('r8'), Register('r10')))
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
        instructions.extend(write_new_value_at(target_addr2))

        instructions.append(Add(src=Imm(1), dst=r_len_32))
        instructions.append(MovQ(src=r_new_ptr, dst=r_ptr))

        instructions.append(Label(end_label))
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
        specifically because the REALLOCATE path (inside _gen_grow_
        and_append_one_into) calls malloc, which (like any real, ABI-
        conforming function) is free to clobber any caller-saved
        register but is OBLIGATED to preserve callee-saved ones -- the
        exact same guarantee gen_array_literal_heap_alloc_into and
        gen_function's own heap-allocated-parameter handling already
        rely on.

        The actual growth policy -- the reuse-vs-reallocate decision,
        the growth-policy arithmetic itself, the copy-existing-
        elements loop -- lives entirely in _gen_grow_and_append_one_
        into now, shared with the print buffer's own single-byte
        append; this method's own remaining job is just materializing
        s into registers beforehand, and writing the final triple to
        dst_mem afterward, protecting dst_mem.base (whenever it isn't
        'rbp') across the whole thing the same way every other slice-
        producing case in this file does -- popped back only
        immediately before the final three-field write actually
        needs it.
        """
        slice_arg, value_arg = expr.args
        slice_type = type_of(slice_arg)
        element_type = slice_type.element_type
        element_width = type_byte_width(element_type, self.struct_registry)

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

        instructions.extend(self._gen_grow_and_append_one_into(
            r_ptr, r_len, r_len_32, r_cap, r_cap_32, element_width,
            copy_one_element=lambda dst, src: self.gen_array_copy(
                Memory(dst.name, 0), Memory(src.name, 0), element_type
            ),
            write_new_value_at=lambda target: self._gen_write_value_at_address_into(
                value_arg, element_type, target
            ),
        ))

        if protect_dst:
            instructions.append(Pop(Register(dst_mem.base)))
        instructions.append(MovQ(src=r_ptr, dst=Memory(dst_mem.base, dst_mem.offset)))
        instructions.append(MovQ(src=r_len, dst=Memory(dst_mem.base, dst_mem.offset + 8)))
        instructions.append(MovQ(src=r_cap, dst=Memory(dst_mem.base, dst_mem.offset + 16)))
        return instructions

    def gen_buffer_append_byte_into(
        self,
        r_ptr: Register, r_len: Register, r_len_32: Register,
        r_cap: Register, r_cap_32: Register,
        byte_value: Operand,
    ) -> List[Instruction]:
        """Appends exactly ONE byte to a growable byte buffer already
        held in r_ptr/r_len/r_cap (and their own 32-bit views) -- the
        print machinery's own single-character append, sharing the
        identical growth policy `append()` itself uses (see _gen_
        grow_and_append_one_into) with element_width=1 and a byte-
        sized write/copy in place of a general Hornet element type's
        own gen_array_copy/_gen_write_value_at_address_into.

        `byte_value` is whatever operand already holds the byte to
        append -- typically an Imm (a literal ASCII byte, e.g. the
        '[' opening a collection, or a decimal digit already reduced
        to a compile-time or runtime-computed Imm) or an 8-bit
        register alias (via as_byte_register) if the byte was computed
        into a register first. Written via MovB -- the first place
        this compiler has ever needed a genuine single-byte memory
        write, as opposed to a 4-byte int/bool or an 8-byte pointer.

        Unlike gen_append_call_into, this never touches a Memory
        destination at all: r_ptr/r_len/r_cap are expected to stay
        live in registers across many further appends while a single
        value's own representation is being built up (see the
        recursive stringify machinery this exists for), not written
        back out after every single byte -- that would be needless
        memory traffic for something that might get appended to
        hundreds of times while building one struct's own printed
        form. Callers that DO need the current triple durably
        persisted (spanning a call into another function, for
        instance) are responsible for spilling it themselves."""
        def copy_one_byte(dst_addr: Register, src_addr: Register) -> List[Instruction]:
            scratch = as_byte_register(Register('eax'))
            return [
                MovB(src=Memory(src_addr.name, 0), dst=scratch),
                MovB(src=scratch, dst=Memory(dst_addr.name, 0)),
            ]

        def write_byte_at(target_addr: Register) -> List[Instruction]:
            return [MovB(src=byte_value, dst=Memory(target_addr.name, 0))]

        return self._gen_grow_and_append_one_into(
            r_ptr, r_len, r_len_32, r_cap, r_cap_32,
            element_width=1,
            copy_one_element=copy_one_byte,
            write_new_value_at=write_byte_at,
        )

    def gen_buffer_append_bytes_into(
        self,
        r_ptr: Register, r_len: Register, r_len_32: Register,
        r_cap: Register, r_cap_32: Register,
        source_addr: Register, count: Operand,
    ) -> List[Instruction]:
        """Appends `count` bytes in one bulk operation, copied from
        source_addr -- the print machinery's own multi-byte append
        (a whole literal fragment like '[', a run of decimal digits
        just converted, another already-built piece), as opposed to
        gen_buffer_append_byte_into's single-character one. This is a
        genuinely DIFFERENT growth calculation, not just a loop calling
        the single-byte version count times: that method's own growth
        formula is only correct because it's derived under the
        assumption reallocation happens exactly when len == cap, one
        element at a time (see gen_append_call_into's own docstring)
        -- appending a 40-byte chunk when only, say, 4 bytes of spare
        capacity remain needs `needed` (len + count) to enter the
        decision directly, which the single-element formula's own
        already-simplified arithmetic has no way to do.

        GROWTH: needed = len + count. If needed <= cap, there's
        already enough spare room -- no reallocation at all, just copy
        directly into the existing backing array at ptr + len. Only
        when needed > cap does this reallocate, to new_cap = max(needed,
        cap*2 if cap < 256 else cap + cap//4) -- the FULL, general
        formula gen_append_call_into's own docstring describes and
        then simplifies away (since ITS OWN needed is always exactly
        cap+1, small enough that doubling always already exceeds it).
        Here needed can be arbitrarily larger than a single doubling
        would produce, so the max has to be computed for real, not
        assumed away -- and this also means the single-element
        formula's own explicit cap==0 floor of 1 isn't needed here
        either: when cap is 0, the doubled-or-quartered side of the
        max is just 0, and max(needed, 0) already correctly resolves
        to needed on its own, since needed (len + count, with count
        always at least 1 for any real call) is always positive.

        Both the reallocate path's own copy-existing-bytes step and
        the final copy-the-new-bytes-in step move `len` (or `count`)
        bytes one at a time via MovB, in a genuine runtime loop --
        there's no bulk memory-move instruction in this file's own
        Instruction vocabulary yet (a real `rep movsb`, or SSE-based
        copy, would be the natural next step if this ever needs to be
        fast; correctness came first here, matching this compiler's
        existing posture everywhere else).

        Internally fixed scratch, distinct from gen_grow_and_append_
        one_into's own set so this can be called independently:
        %rax/%eax, %rcx/%ecx, %rdx/%edx, %rdi, %r10, %r11, %r14, %r15
        (the latter two hold protected copies of source_addr/count --
        see the comment where they're introduced below). Callers must
        choose r_ptr/r_len/r_cap, and whatever register source_addr
        itself lives in, to avoid all of these, exactly like
        gen_buffer_append_byte_into's own callers already must."""
        instructions = []
        no_grow_label = self.new_label("bulk_append_no_grow")
        copy_new_label = self.new_label("bulk_append_copy_new")

        # Move source_addr (always a register in practice) -- and
        # count, if it's a register rather than a compile-time Imm --
        # into callee-saved registers (%r14/%r15) BEFORE any of the
        # growth/malloc logic below runs. The reallocate path calls
        # malloc internally, which -- like any real, ABI-conforming
        # function -- is free to clobber whatever CALLER-saved
        # register the caller happened to pass in for these (e.g.
        # %r8/%r9, as _gen_stringify_bulk_append's own callers do),
        # corrupting them by the time the copy-new loop below needs
        # them afterward. This exact class of bug already has one
        # instance fixed below for new_cap (%ecx -> r_cap_32) -- the
        # same protection was missing here, and manifested identically:
        # correct on the no-grow path (no malloc call to clobber
        # anything), silently wrong -- reading and copying garbage as
        # the new bytes' own source -- on the reallocate path
        # specifically, and only there, which is exactly why it first
        # surfaced as heap corruption several appends downstream
        # rather than as an obviously-wrong value at the call site
        # itself. An Imm count needs no such protection: it's baked
        # directly into the instructions that use it, never stored in
        # a register malloc could touch.
        instructions.append(MovQ(src=source_addr, dst=Register('r14')))
        source_addr = Register('r14')
        if not isinstance(count, Imm):
            instructions.append(Mov(src=count, dst=Register('r15d')))
            count = Register('r15d')

        # needed = len + count, in %eax.
        if isinstance(count, Imm):
            instructions.append(Mov(src=Imm(count.value), dst=Register('eax')))
        else:
            instructions.append(Mov(src=count, dst=Register('eax')))
        instructions.append(Add(src=r_len_32, dst=Register('eax')))

        instructions.append(Cmp(src=r_cap_32, dst=Register('eax')))
        instructions.append(Jle(no_grow_label))

        # REALLOCATE: new_cap = max(needed, doubled-or-quartered(cap)).
        # %eax already holds `needed`; %ecx becomes the doubled-or-
        # quartered candidate, then the larger of the two wins.
        quarter_label = self.new_label("bulk_append_quarter")
        candidate_done_label = self.new_label("bulk_append_candidate_done")
        instructions.append(Mov(src=r_cap_32, dst=Register('ecx')))
        instructions.append(Cmp(src=Imm(256), dst=Register('ecx')))
        instructions.append(Jae(quarter_label))
        instructions.append(IMul(src=Imm(2), dst=Register('ecx')))
        instructions.append(Jmp(candidate_done_label))
        instructions.append(Label(quarter_label))
        instructions.append(Mov(src=r_cap_32, dst=Register('edx')))
        instructions.append(Mov(src=Imm(2), dst=Register('ecx')))
        # Shift the CANDIDATE (starting from cap, held in %edx) right
        # by 2, then add cap back -- ShiftRightArithmetic's own fixed
        # %cl-sourced count means %ecx has to hold the shift amount
        # (2) here, not the candidate itself, unlike the plain-doubling
        # branch just above where %ecx directly holds the result.
        instructions.append(ShiftRightArithmetic(dst=Register('edx')))
        instructions.append(Mov(src=r_cap_32, dst=Register('ecx')))
        instructions.append(Add(src=Register('edx'), dst=Register('ecx')))
        instructions.append(Label(candidate_done_label))
        # %eax = needed, %ecx = candidate. new_cap = max of the two,
        # left in %ecx (needed's own value in %eax is still required
        # below, for how many NEW bytes to copy in, so %eax itself is
        # never overwritten by this comparison).
        instructions.append(Cmp(src=Register('ecx'), dst=Register('eax')))
        max_done_label = self.new_label("bulk_append_max_done")
        instructions.append(Jle(max_done_label))
        instructions.append(Mov(src=Register('eax'), dst=Register('ecx')))
        instructions.append(Label(max_done_label))

        # new_cap (in %ecx, caller-saved) MUST move into r_cap_32
        # (callee-saved) before calling malloc, not after: malloc, like
        # any real ABI-conforming function, is free to clobber %ecx
        # during its own execution, and is only OBLIGATED to preserve
        # callee-saved registers -- exactly the guarantee gen_append_
        # call_into's own %rbx/%r12/%r13 already rely on. Keeping the
        # computed value in %ecx across the call and reading it back
        # afterward (an earlier version of this method did exactly
        # that) is a real, silent bug: %ecx isn't guaranteed to still
        # hold what was put there once malloc returns, and glibc's own
        # malloc does in practice clobber it -- found only by directly
        # checking the resulting cap against a hand-worked-out
        # expected value, since the visible SYMPTOM (a buffer sized
        # smaller than what was actually written into it) doesn't
        # reliably crash for a small overrun like this one.
        instructions.append(Mov(src=Register('ecx'), dst=r_cap_32))

        instructions.append(Mov(src=r_cap_32, dst=Register('edi')))
        instructions.append(CallInstr('malloc'))
        r_new_ptr = Register('r10')
        instructions.append(MovQ(src=Register('rax'), dst=r_new_ptr))

        # Copy the existing len bytes from the OLD array into the NEW
        # one, one byte at a time -- a genuine runtime loop, since len
        # is a runtime value.
        copy_old_loop = self.new_label("bulk_append_copy_old_loop")
        copy_old_done = self.new_label("bulk_append_copy_old_done")
        i_reg = Register('edx')
        i_reg_64 = Register('rdx')  # same physical register, 64-bit view for AddQ's own address arithmetic below
        instructions.append(Mov(src=Imm(0), dst=i_reg))
        instructions.append(Label(copy_old_loop))
        instructions.append(Cmp(src=r_len_32, dst=i_reg))
        instructions.append(Jae(copy_old_done))
        # %eax/%ecx are both already free by this point in the
        # reallocate path -- needed's own value in %eax was last
        # needed to compute new_cap, already consumed into %ecx and
        # then malloc'd (whose OWN return value, briefly also in
        # %eax, has already been copied out into r_new_ptr) -- so
        # nothing here needs preserving across a single byte move,
        # unlike the two Push/Pop pairs an earlier draft of this loop
        # had, which protected against nothing actually still live.
        old_byte = as_byte_register(Register('eax'))
        instructions.append(MovQ(src=r_ptr, dst=Register('r11')))
        instructions.append(AddQ(src=i_reg_64, dst=Register('r11')))
        instructions.append(MovB(src=Memory('r11', 0), dst=old_byte))
        instructions.append(MovQ(src=r_new_ptr, dst=Register('r11')))
        instructions.append(AddQ(src=i_reg_64, dst=Register('r11')))
        instructions.append(MovB(src=old_byte, dst=Memory('r11', 0)))
        instructions.append(Add(src=Imm(1), dst=i_reg))
        instructions.append(Jmp(copy_old_loop))
        instructions.append(Label(copy_old_done))

        instructions.append(MovQ(src=r_new_ptr, dst=r_ptr))
        # r_cap_32 already holds new_cap -- written BEFORE the malloc
        # call above, precisely so it survives that call correctly
        # (see this method's own comment there); nothing further to do
        # with %ecx here, which may no longer even hold that value.
        instructions.append(Jmp(copy_new_label))

        # NO-GROW: needed <= cap, existing backing array already has
        # enough spare room.
        instructions.append(Label(no_grow_label))

        # COPY-NEW: copy `count` bytes from source_addr into ptr+len,
        # one byte at a time, then len += count. By this point r_ptr
        # is already correct either way (untouched on the no-grow
        # path, repointed to the fresh block on the reallocate path).
        instructions.append(Label(copy_new_label))
        if isinstance(count, Imm):
            count_reg = Register('ecx')
            instructions.append(Mov(src=Imm(count.value), dst=count_reg))
        else:
            count_reg = Register('ecx')
            instructions.append(Mov(src=count, dst=count_reg))
        copy_new_loop = self.new_label("bulk_append_copy_new_loop")
        copy_new_done = self.new_label("bulk_append_copy_new_done")
        j_reg = Register('edx')
        j_reg_64 = Register('rdx')  # same physical register, 64-bit view for AddQ's own address arithmetic below
        instructions.append(Mov(src=Imm(0), dst=j_reg))
        instructions.append(Label(copy_new_loop))
        instructions.append(Cmp(src=count_reg, dst=j_reg))
        instructions.append(Jae(copy_new_done))
        new_byte = as_byte_register(Register('eax'))
        instructions.append(MovQ(src=source_addr, dst=Register('r11')))
        instructions.append(AddQ(src=j_reg_64, dst=Register('r11')))
        instructions.append(MovB(src=Memory('r11', 0), dst=new_byte))
        instructions.append(MovQ(src=r_ptr, dst=Register('r11')))
        instructions.append(AddQ(src=r_len, dst=Register('r11')))
        instructions.append(AddQ(src=j_reg_64, dst=Register('r11')))
        instructions.append(MovB(src=new_byte, dst=Memory('r11', 0)))
        instructions.append(Add(src=Imm(1), dst=j_reg))
        instructions.append(Jmp(copy_new_loop))
        instructions.append(Label(copy_new_done))

        instructions.append(Add(src=count_reg, dst=r_len_32))
        return instructions

    def gen_array_copy(self, dst_mem: Memory, src_mem: Memory, array_type: Type) -> List[Instruction]:
        """Copies array_type's worth of data from src_mem to dst_mem --
        both arbitrary Memory operands (e.g. Memory('rbp', -24) for a
        fixed local's own slot, or Memory('rbx', 0) for a computed
        address held in %rbx) -- via a flat sequence of movl/movq
        instructions. A multi-dimensional array is just one contiguous
        block of leaf values in row-major order for copying purposes,
        so no per-dimension logic is needed here at all, just the
        total byte width and the leaf element's own width (see
        type_byte_width/leaf_type).

        Each leaf-sized chunk is copied as a flat run of 8-byte movqs,
        then one trailing 4-byte movl if at least 4 bytes remain, then
        a trailing run of 1-byte movbs for whatever's left after
        that (0 to 3 bytes) -- correct for ANY leaf width at all, not
        just a multiple of 4 the way this used to assume (back when
        every leaf was at least 4 bytes wide: 4 for int/bool, 8 for
        str, 24 for a slice descriptor, or a struct's own width, which
        used to always be a sum of 4-and-8-byte fields and so always
        landed on a multiple of 4 itself). int8/uint8's own genuinely
        1-byte-wide storage broke that assumption two ways at once: a
        BARE int8/uint8 leaf has leaf_width 1 directly, and a STRUCT
        leaf containing an int8/uint8 field can land on any width at
        all (1 int8 + 1 int field is 5, two int8s alone is 2, ...) --
        both were a real, found bug, not a hypothetical one: the old
        two-tier version (8-byte chunks, then EXACTLY one 4-byte
        remainder or none at all) silently copied NOTHING for either
        shape, since neither loop condition (`>= 8` chunks, `== 4`
        exactly) was ever satisfied by a 1-byte or 5-byte leaf_width --
        `b = a` for a `[3]int8` array, or an array of a struct with an
        int8 field, was a complete, silent no-op, not a wrong-but-
        partial copy. That generality is exactly why a STRUCT leaf
        already worked at all before int8/uint8 existed (a struct's
        own width can be any multiple of 4: 12, 20, 28, ... depending
        on its fields), and it needed no field-by-field recursion to
        get there: a raw,
        flat copy of every byte a value occupies is ALWAYS semantically
        identical to copying it "as" whatever logical type or fields
        those bytes represent, given this language's value semantics
        throughout -- there's no reference counting, no copy-
        constructor, and no write barrier anywhere in this language
        that a flat byte copy could possibly get wrong. This is exactly
        why a slice ELEMENT (24 bytes: pointer, then length, then cap)
        already worked before struct existed at all: those three
        sequential 8-byte movqs ARE a flat byte copy of the descriptor,
        which is exactly the shallow, alias-preserving copy slice
        values already get everywhere else (`s2 = s1`, see gen_slice_
        value_into's own Variable case) -- not a special case invented
        for arrays specifically. A struct containing a nested array,
        slice, or another struct needs nothing more than this same
        flat copy, for the identical reason: whatever's nested is
        already just more contiguous bytes within the outer value's
        own footprint.

        The scratch register shuttling each chunk's value between src
        and dst is picked dynamically to differ from BOTH src_mem's and
        dst_mem's own base register -- otherwise loading a value into
        it would destroy the very address a later iteration still needs
        to read from or write to. Found as a real bug during
        development, not a hypothetical one: gen_return passes
        Memory('rax', 0) as the destination when writing an array
        directly through a received hidden return pointer, and
        unconditionally using %eax/%rax as scratch (the very reasonable
        choice everywhere else in this file, since gen_expr_into always
        targets it) destroyed that address the moment the first
        element's value was loaded, before it could even be written
        anywhere. rcx and rdx are never used as a Memory base anywhere
        else in this file, so picking whichever of rax/rcx/rdx isn't
        already one of the two bases here stays correct regardless of
        how many 8- or 4-byte chunks a single leaf's own copy needs."""
        leaf = leaf_type(array_type)
        used_bases = {src_mem.base, dst_mem.base}
        scratch_64, scratch_32 = next(
            (r64, r32) for r64, r32 in [('rax', 'eax'), ('rcx', 'ecx'), ('rdx', 'edx')]
            if r64 not in used_bases
        )
        leaf_width = type_byte_width(leaf, self.struct_registry)
        total = type_byte_width(array_type, self.struct_registry)
        instructions = []
        off = 0
        while off < total:
            # Copy exactly leaf_width bytes starting at offset `off`:
            # as many 8-byte movq chunks as fit, then one 4-byte movl
            # if at least 4 bytes remain after that, then a trailing
            # run of 1-byte movbs (via as_byte_register on the same
            # scratch_32 register the 4-byte case already uses) for
            # whatever's left after THAT -- always 0 to 3 bytes, so at
            # most three movb pairs, never a real loop of its own. See
            # this method's own docstring for why all three tiers are
            # necessary now, not just the first two.
            chunk_off = 0
            while leaf_width - chunk_off >= 8:
                field_src = Memory(src_mem.base, src_mem.offset + off + chunk_off)
                field_dst = Memory(dst_mem.base, dst_mem.offset + off + chunk_off)
                instructions.append(MovQ(src=field_src, dst=Register(scratch_64)))
                instructions.append(MovQ(src=Register(scratch_64), dst=field_dst))
                chunk_off += 8
            if leaf_width - chunk_off >= 4:
                field_src = Memory(src_mem.base, src_mem.offset + off + chunk_off)
                field_dst = Memory(dst_mem.base, dst_mem.offset + off + chunk_off)
                instructions.append(Mov(src=field_src, dst=Register(scratch_32)))
                instructions.append(Mov(src=Register(scratch_32), dst=field_dst))
                chunk_off += 4
            scratch_8 = None
            while leaf_width - chunk_off >= 1:
                if scratch_8 is None:
                    scratch_8 = as_byte_register(Register(scratch_32))
                field_src = Memory(src_mem.base, src_mem.offset + off + chunk_off)
                field_dst = Memory(dst_mem.base, dst_mem.offset + off + chunk_off)
                instructions.append(MovB(src=field_src, dst=scratch_8))
                instructions.append(MovB(src=scratch_8, dst=field_dst))
                chunk_off += 1
            off += leaf_width
        return instructions

    def _gen_address_of_memory_into(self, mem: Memory, dst: Register) -> List[Instruction]:
        """Computes the ADDRESS a Memory operand refers to, into `dst`
        (a 64-bit register). Memory('rbp', offset) needs a real leaq --
        the address is offset-from-frame-pointer, not stored anywhere
        as a value in its own right; Memory(some_reg, offset) already
        HAS its address sitting directly in some_reg, with `offset`
        (if non-zero) added on top via a single AddQ -- see gen_array_
        copy's own docstring for how the some_reg shape arises
        elsewhere in this file. Used specifically for passing a Memory
        destination on as a POINTER argument -- the hidden output
        pointer for an array-returning call (gen_array_call_into) or a
        struct-returning one (gen_struct_call_into, which is really
        gen_array_call_into under a different name -- see its own
        docstring) -- everywhere else, a Memory operand is read from
        or written to directly rather than having its own address
        taken.

        The offset(some_reg) case used to assume offset was always 0
        whenever base wasn't 'rbp' -- true at the time, since nothing
        computed a destination this way for anything but the WHOLE of
        a Memory destination, offset already folded in or genuinely
        zero. That stopped being true once a struct literal's own
        array-typed FIELD could be populated directly by an array-
        returning call (`Big(1, makeArr())`, where `data` -- an array
        field -- sits at some non-zero offset on a heap-allocated
        Big): gen_struct_literal_into's own field_mem for that field
        is Memory('rax', 4), say, and the OLD version of this method
        silently discarded that +4, handing makeArr() the STRUCT's own
        base address as its hidden return pointer instead of the
        field's -- a real, silent miscompile (verified directly: it
        corrupted the PRECEDING field along with the start of the
        array itself), not a hypothetical one. Adding the AddQ here is
        safe for every EXISTING caller too: each one already only ever
        passed offset=0 for a non-'rbp' base, so this is a pure
        generalization, not a behavior change for anything already
        working."""
        if mem.base == 'rbp':
            return [LeaQFrame(offset=mem.offset, dst=dst)]
        instructions = [MovQ(src=Register(mem.base), dst=dst)]
        if mem.offset:
            instructions.append(AddQ(src=Imm(mem.offset), dst=dst))
        return instructions

    def gen_array_arg_address_into(self, expr: Node, dst: Register) -> List[Instruction]:
        """Computes the address to pass for an array-typed function-call
        argument, into the 64-bit register `dst`. A Variable or an
        Index yielding a sub-array already has a real, existing address
        (see gen_array_address_into); an ArrayLiteral or a call
        returning an array used DIRECTLY as an argument (`foo([1,2,3])`
        or `foo(bar())`) has no home of its own, so it's materialized
        first -- see _gen_materialize_argument_temp_into.

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
        if isinstance(expr, (ArrayLiteral, Call)):
            return self._gen_materialize_argument_temp_into(expr, type_of(expr), dst)
        raise CodegenError(
            f"Array-typed call arguments must be a variable, an "
            f"indexing expression, an array literal, or a call, not "
            f"{type(expr).__name__}"
        )

    def gen_slice_arg_into(self, expr: Node, ptr_dst: Register, len_dst: Register, cap_dst: Register) -> List[Instruction]:
        """Computes a slice-typed call ARGUMENT's own ptr/len/cap
        directly into ptr_dst/len_dst/cap_dst. A Variable (a named
        slice) or NoneLiteral (`none`) already has its own descriptor
        sitting somewhere real (a local slot, or nowhere at all --
        `none` is just three immediate zeros); anything else -- a
        slice literal or re-slice (`foo([]int[1,2,3])`, `foo(arr[1:
        3])`), an ordinary slice-returning Call (`foo(makeSlice())`),
        or a slice-typed Field/Index (`foo(s.values)`, `foo(rows[0])`)
        -- has no pre-existing descriptor of its own to read, and is
        materialized first via gen_slice_value_into (which already
        handles every one of these shapes) into the SAME shared, per-
        function scratch slot gen_indexable_base_into's own analogous
        cases already use (_unnamed_slice_temp_offset), then read
        straight back out of it.

        Reusing that ONE shared slot here -- rather than needing its
        own per-occurrence storage the way an array/struct-typed
        argument does (see _collect_argument_temps's own docstring for
        why THOSE need one) -- is safe for a genuinely different
        reason than gen_indexable_base_into's own "fully drained
        before anything else can reuse it" argument: a slice argument
        is passed BY VALUE -- three register values, immediately
        pushed onto the real stack right after this method returns
        (see _gen_call_arguments_into) -- not by address the way an
        array/struct argument is, so nothing about the call itself
        ever needs this scratch slot's own contents to still be valid
        afterward; only the pushed register VALUES do, and those
        already live on the real stack by then. That's also what makes
        two slice-typed arguments to the SAME call safe with only one
        shared slot (`foo([]int[1,2], []int[3,4])`): _gen_call_
        arguments_into evaluates arguments strictly one at a time, so
        the first argument's own descriptor is fully read out of the
        slot and pushed onto the stack before the second argument's
        own materialization ever touches the slot again -- and the
        identical strictly-nested reasoning covers a slice-returning
        call whose OWN argument is itself another unnamed slice
        (`foo(makeOuter(makeInner()))`): makeInner()'s own result is
        fully drained out of the slot and pushed as makeOuter's own
        argument, and `call makeOuter` -- the only thing that will
        eventually write THIS level's own result into the slot, via
        whatever hidden pointer it receives -- doesn't even execute
        until after that inner materialization has already completed."""
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
        temp = self._unnamed_slice_temp_offset
        instructions = self.gen_slice_value_into(expr, Memory('rbp', temp))
        instructions.append(MovQ(src=Memory('rbp', temp), dst=ptr_dst))
        instructions.append(MovQ(src=Memory('rbp', temp + 8), dst=len_dst))
        instructions.append(MovQ(src=Memory('rbp', temp + 16), dst=cap_dst))
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
        which dispatches straight back here), a SLICE (an array whose
        elements are slices -- `[N][]int` -- e.g. the synthesized outer
        literal a slice-of-slices literal always is, `[][]int[[1, 2],
        [3, 4]]` -- handled by gen_slice_value_into, exactly like any
        other slice-producing expression; each element there might be
        an untyped ArrayLiteral needing a fresh backing allocation of
        its own, a named slice Variable, another Slice expression, or
        anything else that method already covers), or a STRUCT (an
        array of structs, `[N]Point` -- handled by gen_struct_value_
        into, which already covers every shape a struct-typed element
        can take: an ordinary Variable/Field/Index, a struct-returning
        Call, or -- as of the same fix that added struct literals as
        array elements at the semantic layer -- a struct literal
        directly, `[Point(1,2), Point(3,4)]`, via that method's own
        Call-is-a-struct-name dispatch).

        This STRUCT case used to not exist at all: an array-of-structs
        literal (or a struct-typed element written through _gen_write_
        value_at_address_into, append's own counterpart to this method
        -- see its own identical fix) would fall through to the
        scalar path below, whose gen_expr_into call flatly rejects any
        struct-typed read regardless of what expression produced it --
        this failed even for the simplest possible case, `[p1, p2]`
        with p1/p2 ordinary, already-declared struct variables, with
        no literal construction involved at all. Every OTHER operation
        on an array of structs (declaring one, indexing into it and
        reading/writing a FIELD of one element via Index-then-Field,
        whole-array copy via plain assignment, passing one as a
        parameter, returning one) already worked before this fix,
        since each of those routes through gen_array_copy (which
        already handles ANY leaf width as a flat byte copy -- struct
        included, see its own docstring) or gen_index_assign/gen_
        field_address_into (which already had their own STRUCT cases)
        rather than through this method's own per-element construction
        path -- literal construction specifically was the one gap.

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
        a single element was ever actually written through it. Neither
        the SLICE nor the STRUCT case needs protection of its own
        here, unlike the scalar case just below: gen_slice_value_into/
        gen_struct_value_into both already protect dst_mem.base
        internally across whatever real work producing their own value
        takes (see their own docstrings), so by the time either
        returns, dst_mem.base is guaranteed correct again -- this loop
        can just call either directly and move on to the next
        element."""
        element_type = array_type.element_type
        element_width = type_byte_width(element_type, self.struct_registry)
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
            if element_type.kind == TypeKind.STRUCT:
                instructions.extend(self.gen_struct_value_into(elem_expr, elem_mem, element_type))
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
                    if element_type == Type.INT64:
                        # A full 64-bit shuttle -- an ordinary 32-bit
                        # Mov here would discard int64's own high 32
                        # bits before _gen_write_scalar_from ever gets
                        # a chance to correctly write them, the exact
                        # same bug already found and fixed in gen_
                        # function's own parameter-binding logic and
                        # gen_print_call_into's own scratch-slot write.
                        instructions.append(MovQ(src=Register('rax'), dst=Register('r8')))
                    else:
                        instructions.append(Mov(src=Register('eax'), dst=Register('r8d')))
                    instructions.append(Pop(Register(dst_mem.base)))
                    instructions.extend(self._gen_write_scalar_from(Register('r8d'), element_type, elem_mem))
                else:
                    instructions.extend(self._gen_write_scalar_from(Register('eax'), element_type, elem_mem))
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
            if self._is_heap_allocated(self._local_decl_id(expr.name), src_type):
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
        size = type_byte_width(array_type, self.struct_registry)
        return [Mov(src=Imm(size), dst=Register('edi')), CallInstr('malloc')]

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
        for reg in reversed(CALLEE_SAVED_SCRATCH_REGISTERS):
            instructions.append(Pop(Register(reg)))
        instructions.append(Leave())
        instructions.append(Ret())
        return instructions

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
        other, non-literal array-, slice-, or struct-typed expression
        (a Variable, an indexed sub-array, an array/struct-returning
        Call, ...) is a real, deliberately out-of-scope gap: reading a
        bare array-typed Variable has no side effect worth preserving,
        but an array-returning Call might, and correctly distinguishing
        the two -- or materializing either one just to discard it --
        isn't implemented here. Raises a clear error rather than
        silently skipping (which could drop a real side effect) or
        guessing. (A struct-typed element specifically would already
        raise via gen_expr_into's own defensive rejection even without
        being listed here explicitly, since this method falls through
        to it for anything not caught above -- STRUCT is included in
        the tuple below anyway, for the same clearer, statement-
        specific message every other composite kind gets here, rather
        than relying on gen_expr_into's own more generic one.)
        """
        instructions = []
        for element in expr.elements:
            if isinstance(element, ArrayLiteral):
                instructions.extend(self.gen_array_literal_side_effects_only(element))
                continue
            element_type = type_of(element)
            if element_type.kind in (TypeKind.ARRAY, TypeKind.SLICE, TypeKind.STRUCT):
                raise CodegenError(
                    f"A bare array-literal statement can't have a "
                    f"{type(element).__name__} element of type "
                    f"{element_type} -- assign the literal to a "
                    f"variable first if you need this element's value "
                    f"or side effect evaluated"
                )
            instructions.extend(self.gen_expr_into(element, Register('eax')))
        return instructions

    def _gen_read_scalar_into(self, mem: Memory, t: Type, dst: Register) -> List[Instruction]:
        """Reads a scalar value of type `t` (int, int8, uint8, int64,
        or bool) from `mem` into `dst` -- the one choke point every
        scalar READ site in this file goes through, so int8/uint8's
        own genuinely narrow (1-byte) storage AND int64's own genuinely
        wide (8-byte) storage (see type_byte_width) only ever needed
        teaching to ONE place, not rediscovering at every Variable/
        Field/Index read site individually.

        int8 needs a SIGN-extending read (MovSX) and uint8 a ZERO-
        extending one (MovZX, already built for SetE's own unrelated
        need) rather than an ordinary 4-byte Mov, which would read
        three bytes of adjacent memory that were never part of this
        value at all -- and for int8 specifically, would also silently
        misinterpret a negative value as a large positive one (int8(-1)
        == 0xFF read as a raw 4-byte int would become 0x000000FF ==
        255, not -1) even if the adjacent bytes happened to be zero.
        Every later arithmetic/comparison instruction in this file
        already assumes it's operating on a genuinely correct 32-bit
        value, so getting the WIDENING right here, once, is what lets
        everything downstream stay completely unaware int8/uint8 are
        narrower than int at all.

        int64 needs a full 8-byte read (MovQ) into `dst`'s own 64-bit
        VIEW (as_qword_register(dst), e.g. %eax -> %rax) -- `dst`
        itself is always passed as a 32-bit-named register by every
        caller in this file (the same convention str's own handling
        already established elsewhere in gen_expr_into), with THIS
        method responsible for deciding which actual view to read
        into, exactly the way int8/uint8's own narrowing decision is
        made here rather than pushed onto every caller. An ordinary
        4-byte Mov here would silently drop int64's own high 32 bits
        entirely, not just read a stale/incorrect value -- reading is
        the one direction narrower-than-needed storage access can
        outright discard real, distinct bits of a value rather than
        merely mis-INTERPRET already-present ones the way int8/uint8's
        own narrow case can.

        int and bool are untouched -- an ordinary 4-byte Mov, exactly
        as before this method existed."""
        if t == Type.INT8:
            return [MovSX(src=mem, dst=dst)]
        if t == Type.UINT8:
            return [MovZX(src=mem, dst=dst)]
        if t == Type.INT64:
            return [MovQ(src=mem, dst=as_qword_register(dst))]
        return [Mov(src=mem, dst=dst)]

    def _gen_write_scalar_from(self, src: Register, t: Type, dst_mem: Memory) -> List[Instruction]:
        """Writes a scalar value of type `t`, already computed into
        `src`, into `dst_mem` -- the WRITE-side counterpart to _gen_
        read_scalar_into, and the other half of the same one-choke-
        point principle: every scalar WRITE site in this file goes
        through this, rather than each one separately remembering that
        int8/uint8 need a narrower store or int64 a wider one.

        int8/uint8 need a 1-byte, TRUNCATING store (MovB, of src's own
        low-byte alias -- see as_byte_register) rather than an ordinary
        4-byte Mov, which would clobber whatever adjacent memory
        happens to immediately follow this value (an adjacent struct
        field, the next array element, ...) -- exactly the kind of
        silent, hard-to-diagnose corruption a narrow type's own
        storage existing at all is supposed to make possible to write
        correctly, not introduce a new way to get wrong.

        int64 needs a full 8-byte store (MovQ, of src's own 64-bit VIEW
        -- as_qword_register(src)) -- CALLERS are responsible for
        having already computed the value into that same 64-bit view
        before reaching this method (every gen_expr_into case that can
        produce an int64 result does exactly this -- see its own
        Constant/Variable/Binary/Unary/Cast cases), not just src's own
        low 32 bits: an ordinary 4-byte Mov here would write only the
        low half of a value whose own high half might be meaningful,
        and reading src's own 64-bit view when only the low 32 bits
        were ever actually computed would write whatever stale garbage
        happened to occupy that register's own high bits, silently
        corrupting the stored value in a way that could be very hard
        to trace back to its actual cause.

        int and bool are untouched -- an ordinary 4-byte Mov, exactly
        as before this method existed."""
        if t == Type.INT8 or t == Type.UINT8:
            return [MovB(src=as_byte_register(src), dst=dst_mem)]
        if t == Type.INT64:
            return [MovQ(src=as_qword_register(src), dst=dst_mem)]
        return [Mov(src=src, dst=dst_mem)]

    def gen_cast_narrowing_into(self, target_type: Type, dst: Register) -> List[Instruction]:
        """The actual work behind an explicit `TYPE(expr)` cast (see
        gen_expr_into's own Cast case): re-narrows `dst`'s own value
        to genuinely, correctly represent target_type, given that
        gen_expr_into has already computed the SOURCE expression into
        it (already correctly widened, if the source happened to be
        int8/uint8-typed itself -- see _gen_read_scalar_into).

        A target of int needs NOTHING further: the source's own
        already-widened 32-bit value already IS a valid int, whatever
        the source type actually was (an int8/uint8 source is already
        sign/zero-extended; an int source needs no widening at all).

        A target of int8 or uint8 needs exactly ONE more instruction:
        MovSX (int8) or MovZX (uint8) applied to dst's OWN low-byte
        alias, writing the result back into dst itself -- a single,
        purely register-to-register re-widening (MovSX/MovZX's own
        src operand doesn't have to be memory; see MovZX's own
        docstring), no memory round-trip needed at all. This is
        DELIBERATE, not just an optimization: a cast's own RESULT has
        to be a genuinely, correctly narrowed value immediately, not
        merely "correct once eventually written to int8/uint8-typed
        storage" the way _gen_write_scalar_from's own truncation is --
        `int8(300) + int8(5)` needs 300 already wrapped to 44 BEFORE
        the addition happens, or the arithmetic itself would silently
        be wrong, since every later int8/uint8 operation assumes its
        own operands already correctly represent a narrow value, never
        re-validating that itself.

        Correct regardless of what the source type actually was, not
        just for a narrowing int-to-int8/uint8 cast: re-extending
        whatever's already sitting in the low byte is exactly as
        correct for a same-width REINTERPRETATION (int8-to-uint8 or
        back) as it is for genuine narrowing, since both are really
        the same operation -- "take the low byte, reinterpret it under
        a new sign convention" -- differing only in whether the high
        bytes being discarded happened to already be a trivial (int8/
        uint8 source) or a real (int source) sign/zero-extension of
        that byte. Verified against concrete cases during design, not
        just asserted: int(300) as int8 gives 44 (300's own low byte,
        0x2C, has its high bit clear, so sign-extension leaves it
        positive); int(200) as int8 gives -56 (200's own low byte,
        0xC8, has its high bit set, so sign-extension correctly
        produces the negative two's-complement reinterpretation) --
        both match a real 8-bit truncate-then-reinterpret exactly.

        A target of int64 needs exactly one instruction too, in the
        OPPOSITE direction: MovSXD, sign-extending dst's own 32-bit
        view up into its 64-bit one -- correct regardless of whether
        the source was int, int8, or uint8, since all three are
        already read into a genuinely correct, ordinary 32-bit value
        by the time this runs (see _gen_read_scalar_into), and a non-
        negative 32-bit value's own sign bit is already clear, so
        sign-extending it produces the identical result zero-extending
        it would have. NARROWING out of int64 (int64(x) targeting int,
        int8, or uint8) needs NO new instruction here at all: dst's own
        32-bit view is always simply the low half of whatever's in its
        64-bit view, so falling through to the existing int/int8/uint8
        branches above -- which already operate on dst's own 32-bit
        view or its own low-byte alias -- is already exactly correct,
        the same way it already was before int64 existed."""
        if target_type == Type.INT8:
            return [MovSX(src=as_byte_register(dst), dst=dst)]
        if target_type == Type.UINT8:
            return [MovZX(src=as_byte_register(dst), dst=dst)]
        if target_type == Type.INT64:
            return [MovSXD(src=dst, dst=as_qword_register(dst))]
        return []

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
            if expr.resolved_type == Type.INT64:
                # A full 64-bit immediate move (`movq $9000000000,
                # %rax`) -- GNU as accepts an immediate this wide
                # specifically for movq (silently using the `movabs`
                # encoding under the hood), the one exception to
                # ordinary x86-64 instructions being limited to a
                # 32-bit immediate operand. An ordinary 32-bit Mov
                # here would either truncate the value or simply fail
                # to assemble for anything outside int32's own range.
                return [MovQ(src=Imm(expr.value), dst=as_qword_register(dst))]
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
            if var_type.kind == TypeKind.STRUCT:
                raise CodegenError(
                    f"Cannot read struct-typed variable '{expr.name}' via "
                    f"gen_expr_into -- a struct doesn't fit in a single "
                    f"register; use gen_struct_value_into or "
                    f"gen_struct_address_into instead"
                )
            if var_type == Type.STR:
                return [MovQ(src=Memory('rbp', offset), dst=as_qword_register(dst))]
            return self._gen_read_scalar_into(Memory('rbp', offset), var_type, dst)
        if isinstance(expr, Index):
            element_type = type_of(expr)
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
            if element_type.kind == TypeKind.STRUCT:
                # Same reasoning, for a struct-typed array element
                # (`rows[i]` where rows is an array of structs) --
                # `Point p = rows[i]` is handled via gen_struct_value_
                # into instead.
                raise CodegenError(
                    "Cannot read a struct-typed array element via "
                    "gen_expr_into -- a struct doesn't fit in a single "
                    "register; use gen_struct_value_into or "
                    "gen_struct_address_into instead"
                )
            addr_reg = as_qword_register(dst)
            instructions = self.gen_index_address_into(expr, addr_reg)
            if element_type == Type.STR:
                instructions.append(MovQ(src=Memory(addr_reg.name, 0), dst=addr_reg))
            else:
                instructions.extend(self._gen_read_scalar_into(Memory(addr_reg.name, 0), element_type, dst))
            return instructions
        if isinstance(expr, Field):
            field_type = type_of(expr)
            if field_type.kind == TypeKind.ARRAY:
                raise CodegenError(
                    "Cannot read an array-typed field via gen_expr_into "
                    "-- arrays don't fit in a single register; use "
                    "gen_array_value_into or gen_array_address_into instead"
                )
            if field_type.kind == TypeKind.SLICE:
                raise CodegenError(
                    "Cannot read a slice-typed field via gen_expr_into -- "
                    "slices don't fit in a single register; use "
                    "gen_slice_value_into instead"
                )
            if field_type.kind == TypeKind.STRUCT:
                raise CodegenError(
                    "Cannot read a struct-typed field via gen_expr_into -- "
                    "a struct doesn't fit in a single register; use "
                    "gen_struct_value_into or gen_struct_address_into "
                    "instead"
                )
            addr_reg = as_qword_register(dst)
            instructions = self.gen_field_address_into(expr, addr_reg)
            if field_type == Type.STR:
                instructions.append(MovQ(src=Memory(addr_reg.name, 0), dst=addr_reg))
            else:
                instructions.extend(self._gen_read_scalar_into(Memory(addr_reg.name, 0), field_type, dst))
            return instructions
        if isinstance(expr, Call):
            if type_of(expr).kind == TypeKind.ARRAY:
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
            if type_of(expr).kind == TypeKind.SLICE:
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
            if type_of(expr).kind == TypeKind.STRUCT:
                # Same reasoning again, for a struct-returning call --
                # only ever reached via gen_struct_value_into's own
                # Call case or gen_return's own forwarding case.
                raise CodegenError(
                    f"Cannot call '{expr.name}' (which returns a struct) "
                    f"via gen_expr_into -- a struct doesn't fit in a "
                    f"single register; use gen_struct_call_into instead"
                )
            if expr.name == 'print':
                return self.gen_print_call_into(expr, dst)
            if expr.name == 'len':
                return self.gen_len_call_into(expr, dst)
            return self.gen_call_into(expr, dst)
        if isinstance(expr, Cast):
            # Compute the source expression into dst first (already
            # correctly widened if it was itself int8/uint8-typed --
            # see _gen_read_scalar_into), then re-narrow dst's own
            # LOW BYTE if the target is int8/uint8 -- see gen_cast_
            # narrowing_into's own docstring for why this single,
            # register-to-register instruction is enough regardless
            # of what the SOURCE type actually was. A target of int
            # needs nothing further at all: the source's own already-
            # widened value already IS a valid int.
            instructions = self.gen_expr_into(expr.expr, dst)
            instructions.extend(self.gen_cast_narrowing_into(expr.resolved_type, dst))
            return instructions
        if isinstance(expr, Unary):
            # Compute the operand into dst first, then apply this node's
            # operator to whatever's now there. This is what makes chained
            # operators (`~-2`) work: the inner Unary's instructions run
            # first, then the outer operator's instructions run on top.
            #
            # operand_type reads type_of(expr) -- this OUTER node's own
            # resolved_type -- not type_of(expr.operand). The two are
            # ordinarily identical (check_unary's own rule is "stays the
            # operand's own type"), EXCEPT for a widened literal: `int64
            # x = -5` sets resolved_type to int64 on the OUTER Unary node
            # (see _check_value_flowing_into's own case 3), but the INNER
            # Constant(5) node's own resolved_type is still whatever
            # check_expr's earlier, ordinary recursive pass already set
            # it to (Type.INT) -- never updated, since the widening logic
            # only ever touches the outermost node of a literal
            # expression. Reading the inner one here was a real, found
            # bug: it silently fed the wrong operand_type into gen_
            # unary_op, using 32-bit Neg instead of NegQ for `-5` widened
            # to int64 -- invisible for int8/uint8 only because THEIR
            # own unary dispatch never branched on operand_type at all
            # before int64 existed, so any operand_type value produced
            # the identical, correct 32-bit instruction either way.
            instructions = self.gen_expr_into(expr.operand, dst)
            instructions.extend(self.gen_unary_op(expr.op, dst, operand_type=type_of(expr)))
            return instructions
        if isinstance(expr, Binary):
            # ADD and the two equality operators are overloaded for str
            # (concatenation and strcmp-backed comparison respectively;
            # see the module docstring's STRINGS section) -- everything
            # else, and ADD/==/!= between two ints or bools, goes
            # through the original gen_binary_into completely unchanged.
            if expr.op == BinaryOp.ADD and type_of(expr.left) == Type.STR:
                return self.gen_string_concat_into(expr, dst)
            if expr.op in (BinaryOp.EQUAL, BinaryOp.NOT_EQUAL) and type_of(expr.left) == Type.STR:
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
        #
        # ARRAY and STRUCT equality are dispatched the same way, for
        # the same root reason: neither one's own value fits through a
        # single register the way an int/bool/str value does.
        # semantic.py's own check_binary already guarantees, by the
        # time this is reached, that both sides are the exact same
        # array or struct type, and that the type is actually
        # comparable (see _is_comparable_type) -- no slice anywhere
        # inside it, directly or nested through a struct field or an
        # array element -- see gen_array_equality_into/gen_struct_
        # equality_into's own docstrings for how each dispatches
        # internally.
        if expr.op in (BinaryOp.EQUAL, BinaryOp.NOT_EQUAL):
            if type_of(expr.left).kind == TypeKind.SLICE or type_of(expr.right).kind == TypeKind.SLICE:
                return self.gen_slice_none_comparison_into(expr, dst)
            if type_of(expr.left).kind == TypeKind.ARRAY:
                return self.gen_array_equality_into(expr, dst)
            if type_of(expr.left).kind == TypeKind.STRUCT:
                return self.gen_struct_equality_into(expr, dst)

        scratch = Register('ecx')  # holds the right-hand value while combining
        operand_type = type_of(expr.left)  # left and right are guaranteed the same type by semantic.py's own check_binary
        instructions = self.gen_expr_into(expr.left, dst)   # dst = left
        instructions.append(Push(as_qword_register(dst)))   # save left on the stack
        instructions.extend(self.gen_expr_into(expr.right, dst))  # dst = right (left is safe)
        if operand_type == Type.INT64:
            # An ordinary 32-bit Mov here would silently drop the
            # right-hand value's own high 32 bits -- scratch has to
            # receive the SAME 64-bit view dst's own value was just
            # computed into (see gen_expr_into's own Constant/Variable/
            # Binary/Unary/Cast cases, all of which compute an int64
            # result into dst's 64-bit view specifically for this
            # reason), not just its low half.
            instructions.append(MovQ(src=as_qword_register(dst), dst=as_qword_register(scratch)))
        else:
            instructions.append(Mov(src=dst, dst=scratch))       # scratch = right
        instructions.append(Pop(as_qword_register(dst)))     # dst = left (restored)
        instructions.extend(self.gen_binary_op(expr.op, src=scratch, dst=dst, operand_type=operand_type))
        return instructions

    def gen_array_equality_into(self, expr: Binary, dst: Register) -> List[Instruction]:
        """`left == right` / `left != right`, both the exact same
        array type (already guaranteed by semantic.py's own check_
        binary, including that the array is actually comparable --
        see _is_comparable_type -- meaning its own LEAF type -- see
        leaf_type -- is int, bool, str, or a comparable struct; never
        a slice, or a struct with a slice buried in it somewhere,
        neither of which has '==' defined for it at all yet).

        `left`/`right` must each already have a real address (a
        Variable, Index, or Field -- whatever gen_array_address_into
        already accepts); an array literal or an array-returning call
        used directly as an equality operand isn't supported, matching
        this file's established "assign it to a variable first"
        restriction on unnamed array values elsewhere (e.g. gen_array_
        arg_address_into before argument materialization existed).

        Dispatches on the array's own leaf type into one of three
        genuinely different comparison strategies, each factored into
        its own small loop helper:

          - int/bool leaf: _gen_array_flat_byte_equality_loop. Neither
            type is a pointer, so the WHOLE array -- however many
            elements, however deeply nested (`[2][3]int` is just one
            contiguous 24-byte block) -- can be compared as one flat
            run of bytes, exactly the same "treat a nested array as
            one flat block" trick gen_array_copy already relies on for
            copying (via leaf_type/type_byte_width), just applied to
            comparison instead of copying.
          - str leaf: _gen_array_str_equality_loop. A str element IS a
            pointer (see the module docstring's STRINGS section), so
            raw byte-for-byte equality of the pointers themselves
            would be wrong -- two equal strings can easily live at two
            different addresses. This calls strcmp on each
            corresponding pair of elements instead, exactly like
            gen_string_compare_into's own ordinary str == str
            comparison does for a single pair, just without that
            method's own concatenation-freeing logic: array elements
            are always fixed, already-allocated storage, never a
            fresh concatenation result of their own.
          - struct leaf: _gen_array_struct_equality_loop. Neither of
            the two loops above applies: a struct's own fields can be
            a MIX of types, so there's no single flat-byte-or-strcmp
            strategy that covers a whole struct-typed element the way
            there is for a uniformly-int/bool or uniformly-str one --
            this reuses _gen_struct_fields_equality_at_addresses (the
            same field-by-field comparison gen_struct_equality_into's
            own bare struct-vs-struct case uses) once per element.

        All three loops jump to a shared `mismatch_label` the moment
        any element differs (or, for the flat-byte path, any 4-byte
        chunk differs); falling all the way through any of them means
        every element matched. From there, the final result is just
        two immediate moves -- 1/0 for EQUAL, or 0/1 for NOT_EQUAL,
        whichever bytes-matched actually means for this specific
        operator -- exactly the same "compute the boolean the long
        way, then pick the right immediate for this operator" shape
        gen_short_circuit already uses for AND/OR, one level over."""
        array_type = type_of(expr.left)
        leaf = leaf_type(array_type)
        total_width = type_byte_width(array_type, self.struct_registry)

        left_addr = Register('r10')
        right_addr = Register('r11')
        instructions = self.gen_array_address_into(expr.left, left_addr)
        instructions.append(Push(left_addr))
        instructions.extend(self.gen_array_address_into(expr.right, right_addr))
        instructions.append(Pop(left_addr))

        mismatch_label = self.new_label("array_eq_mismatch")
        done_label = self.new_label("array_eq_done")

        if leaf == Type.STR:
            instructions.extend(self._gen_array_str_equality_loop(
                left_addr, right_addr, total_width // 8, mismatch_label
            ))
        elif leaf.kind == TypeKind.STRUCT:
            struct_width = type_byte_width(leaf, self.struct_registry)
            instructions.extend(self._gen_array_struct_equality_loop(
                left_addr, right_addr, leaf.struct_name, total_width // struct_width, mismatch_label
            ))
        else:
            # int8/uint8 need a 1-byte step (see this loop's own
            # docstring for why 4 bytes at a time would be a real,
            # out-of-bounds bug for either); int/bool/slice all stay
            # the existing 4-byte step, since type_byte_width already
            # guarantees their own total_width is a multiple of 4
            # regardless of nesting depth.
            step = 1 if leaf in (Type.INT8, Type.UINT8) else 4
            instructions.extend(self._gen_array_flat_byte_equality_loop(
                left_addr, right_addr, total_width, mismatch_label, step=step
            ))

        # Fell all the way through: every element matched.
        instructions.append(Mov(src=Imm(1 if expr.op == BinaryOp.EQUAL else 0), dst=dst))
        instructions.append(Jmp(done_label))
        instructions.append(Label(mismatch_label))
        instructions.append(Mov(src=Imm(0 if expr.op == BinaryOp.EQUAL else 1), dst=dst))
        instructions.append(Label(done_label))
        return instructions

    def _gen_array_flat_byte_equality_loop(self, left_addr: Register, right_addr: Register, total_width: int, mismatch_label: str, step: int = 4) -> List[Instruction]:
        """Compares `total_width` bytes at left_addr/right_addr, `step`
        bytes at a time -- 4 for an int/bool/slice leaf (type_byte_
        width guarantees total_width is always a multiple of 4 for any
        of those, however deeply nested), or 1 for an int8/uint8 leaf.

        The 1-byte case is a real, found bug's fix, not a defensive
        addition: this loop used to ALWAYS step 4 bytes at a time,
        which was fine as long as every leaf this compiler had was 4
        (or a multiple of 4, for a slice leaf's own 24) bytes wide --
        but int8/uint8's own genuinely 1-byte-wide storage means
        total_width isn't generally a multiple of 4 at all (a [3]int8
        array is 3 bytes total). Stepping 4 bytes at a time regardless
        read one byte past the end of the array on every comparison,
        silently comparing whatever adjacent stack memory happened to
        follow it instead of correctly reporting equality.

        The 1-byte step reads each side via MovZX (zero-extending into
        a 32-bit register) rather than a plain 4-byte Mov, needing a
        second register (%edx, otherwise unused in this loop) to hold
        the right side's own zero-extended value before comparing the
        two directly -- correct regardless of whether the ACTUAL leaf
        is signed (int8) or unsigned (uint8): byte-for-byte equality
        never depends on how those bits are INTERPRETED, only on
        whether they're identical, and zero-extension is a
        deterministic, injective mapping from one byte to 32 bits, so
        two bytes are equal if and only if their zero-extended 32-bit
        versions are.

        Jumps to mismatch_label the moment any chunk differs, or
        simply falls through once every chunk has matched. No calls
        happen anywhere in this loop, so nothing here needs a callee-
        saved register the way the str-leaf loop below does; every
        register used is ordinary caller-saved scratch, freely
        reusable by whatever runs after this method returns."""
        index_32 = Register('ecx')
        index_64 = Register('rcx')
        loop_start = self.new_label("array_eq_flat_loop")
        loop_done = self.new_label("array_eq_flat_done")
        left_word_addr = Register('r8')
        right_word_addr = Register('r9')
        instructions = [
            Mov(src=Imm(0), dst=index_32),
            Label(loop_start),
            Cmp(src=Imm(total_width), dst=index_32),
            Jae(loop_done),
            MovQ(src=left_addr, dst=left_word_addr),
            AddQ(src=index_64, dst=left_word_addr),
            MovQ(src=right_addr, dst=right_word_addr),
            AddQ(src=index_64, dst=right_word_addr),
        ]
        if step == 1:
            instructions.append(MovZX(src=Memory(left_word_addr.name, 0), dst=Register('eax')))
            instructions.append(MovZX(src=Memory(right_word_addr.name, 0), dst=Register('edx')))
            instructions.append(Cmp(src=Register('edx'), dst=Register('eax')))
        else:
            instructions.append(Mov(src=Memory(left_word_addr.name, 0), dst=Register('eax')))
            instructions.append(Cmp(src=Memory(right_word_addr.name, 0), dst=Register('eax')))
        instructions.append(Jne(mismatch_label))
        instructions.append(Add(src=Imm(step), dst=index_32))
        instructions.append(Jmp(loop_start))
        instructions.append(Label(loop_done))
        return instructions

    def _gen_array_str_equality_loop(self, left_addr: Register, right_addr: Register, element_count: int, mismatch_label: str) -> List[Instruction]:
        """The str-leaf counterpart to _gen_array_flat_byte_equality_
        loop just above: each of the array's `element_count` str
        elements is a POINTER, so this calls strcmp on each
        corresponding pair rather than comparing raw pointer bytes.

        strcmp is a real external call, free to clobber any CALLER-
        saved register -- so unlike the flat-byte loop, the two base
        addresses and the loop index all have to live in CALLEE-saved
        registers (%rbx/%r12/%r13) to survive it, the exact same
        discipline gen_append_call_into's own malloc-crossing
        registers already follow, for the identical reason: every
        function's own prologue/epilogue already saves and restores
        these four unconditionally (see _CALLEE_SAVED_SCRATCH_
        REGISTERS), so using them as scratch across a call, in ANY
        function, is always safe."""
        left_base = Register('rbx')
        right_base = Register('r12')
        index_32 = Register('r13d')
        index_64 = Register('r13')
        offset_32 = Register('r14d')
        offset_64 = Register('r14')

        loop_start = self.new_label("array_eq_str_loop")
        loop_done = self.new_label("array_eq_str_done")

        instructions = [
            MovQ(src=left_addr, dst=left_base),
            MovQ(src=right_addr, dst=right_base),
            Mov(src=Imm(0), dst=index_32),
            Label(loop_start),
            Cmp(src=Imm(element_count), dst=index_32),
            Jae(loop_done),
        ]
        # byte offset = index * 8 (each str element is one 8-byte
        # pointer) -- computed fresh each iteration, before the call,
        # so it never needs to survive one itself.
        instructions.append(Mov(src=index_32, dst=offset_32))
        instructions.append(IMul(src=Imm(8), dst=offset_32))
        left_elem_addr = Register('r8')
        right_elem_addr = Register('r9')
        instructions.append(MovQ(src=left_base, dst=left_elem_addr))
        instructions.append(AddQ(src=offset_64, dst=left_elem_addr))
        instructions.append(MovQ(src=right_base, dst=right_elem_addr))
        instructions.append(AddQ(src=offset_64, dst=right_elem_addr))
        # Load the actual string POINTERS stored at these element
        # addresses -- straight into the argument registers strcmp
        # itself expects, since nothing else needs them first.
        instructions.append(MovQ(src=Memory(left_elem_addr.name, 0), dst=Register('rdi')))
        instructions.append(MovQ(src=Memory(right_elem_addr.name, 0), dst=Register('rsi')))
        instructions.append(CallInstr('strcmp'))
        instructions.append(Cmp(src=Imm(0), dst=Register('eax')))
        instructions.append(Jne(mismatch_label))
        instructions.append(Add(src=Imm(1), dst=index_32))
        instructions.append(Jmp(loop_start))
        instructions.append(Label(loop_done))
        return instructions

    def _gen_array_struct_equality_loop(self, left_addr: Register, right_addr: Register, struct_name: str, element_count: int, mismatch_label: str) -> List[Instruction]:
        """The struct-leaf counterpart to the two array-equality loops
        above: unlike int/bool (one flat byte comparison) or str (one
        strcmp per element), a struct element's own fields can be a
        MIX of types, so each element is compared via _gen_struct_
        fields_equality_at_addresses -- the same field-by-field
        machinery gen_struct_equality_into's own bare struct-vs-struct
        case uses -- rather than a single uniform per-element
        operation.

        Uses the identical CALLEE-saved register discipline _gen_
        array_str_equality_loop already established, for the same
        reason (a struct element being compared might itself contain
        a str field, which needs a real strcmp CALL somewhere inside
        _gen_struct_fields_equality_at_addresses).

        Beyond that, this loop's own left_base/right_base/index
        (%rbx/%r12/%r13) need one MORE layer of protection that
        neither sibling loop does: _gen_struct_fields_equality_at_
        addresses can itself recurse back into ONE of these same
        three array-equality loops, for a struct field that's itself
        an array (including, recursively, another array of structs --
        e.g. comparing `[M]Outer` where Outer has a `[N]Inner rows`
        field, and Inner is itself a comparable struct). Any such
        nested loop reuses the EXACT SAME fixed register names this
        one does (%rbx/%r12/%r13/%r14, since there's no way to
        allocate a fresh, distinct set per nesting depth at codegen
        time) -- so without explicitly saving THIS loop's own
        %rbx/%r12/%r13 across the per-element comparison call, a
        struct field found to need one of those nested loops would
        silently corrupt this OUTER loop's own base addresses and
        index. Protecting them via an ordinary push/pop pair around
        that one call is what makes this correct at ANY nesting depth,
        by the same induction _gen_struct_fields_equality_at_
        addresses's own per-field protection already relies on: at
        every level, whatever's ABOUT to run might reuse these
        registers for its own purposes, so whatever's ALREADY relying
        on them saves its own values first and restores them
        afterward, regardless of what happened in between. (%r14,
        the per-iteration byte offset, needs no such protection: it's
        always freshly recomputed at the START of each iteration,
        before being used to compute this iteration's own element
        addresses, and never read again afterward.)"""
        struct_width = type_byte_width(Type(TypeKind.STRUCT, struct_name=struct_name), self.struct_registry)
        left_base = Register('rbx')
        right_base = Register('r12')
        index_32 = Register('r13d')
        index_64 = Register('r13')
        offset_32 = Register('r14d')
        offset_64 = Register('r14')

        loop_start = self.new_label("array_eq_struct_loop")
        loop_done = self.new_label("array_eq_struct_done")

        instructions = [
            MovQ(src=left_addr, dst=left_base),
            MovQ(src=right_addr, dst=right_base),
            Mov(src=Imm(0), dst=index_32),
            Label(loop_start),
            Cmp(src=Imm(element_count), dst=index_32),
            Jae(loop_done),
        ]
        instructions.append(Mov(src=index_32, dst=offset_32))
        instructions.append(IMul(src=Imm(struct_width), dst=offset_32))
        left_elem_addr = Register('r8')
        right_elem_addr = Register('r9')
        instructions.append(MovQ(src=left_base, dst=left_elem_addr))
        instructions.append(AddQ(src=offset_64, dst=left_elem_addr))
        instructions.append(MovQ(src=right_base, dst=right_elem_addr))
        instructions.append(AddQ(src=offset_64, dst=right_elem_addr))

        instructions.append(Push(left_base))
        instructions.append(Push(right_base))
        instructions.append(Push(index_64))
        instructions.extend(self._gen_struct_fields_equality_at_addresses(
            struct_name, left_elem_addr, right_elem_addr, mismatch_label
        ))
        instructions.append(Pop(index_64))
        instructions.append(Pop(right_base))
        instructions.append(Pop(left_base))

        instructions.append(Add(src=Imm(1), dst=index_32))
        instructions.append(Jmp(loop_start))
        instructions.append(Label(loop_done))
        return instructions

    def _gen_zero_value_into(self, t: Type, dst_mem: Memory) -> List[Instruction]:
        """Writes t's own implicit zero value into dst_mem -- what a
        `T x` VarDecl with no initializer now gets, instead of the
        genuinely uninitialized memory it used to leave behind (see
        gen_var_decl's own note on the earlier, now-superseded
        behavior). Dispatches by kind:
          - int/bool/int8/uint8: an ordinary 0 -- a plain 4-byte write
            for int/bool, a 1-byte one (MovB) for int8/uint8, matching
            each type's own genuine storage width (see type_byte_
            width).
          - str: the address of a single shared, static empty-string
            constant (_get_empty_str_label) -- NEVER a null pointer;
            see that method's own docstring for why a null zero value
            would be an active hazard, not just an unusual choice.
          - slice: none's own {ptr: 0, len: 0, cap: 0} descriptor,
            reusing gen_none_into exactly as-is -- this needed no new
            code at all, since a zero-value slice and a none-valued
            one are, by design, the identical representation.
          - array: delegated to _gen_zero_array_into, which further
            dispatches on the array's own LEAF type -- see its own
            docstring for why int/bool/slice, str, and struct leaves
            each need a genuinely different strategy.
          - struct: every one of the struct's own fields, flattened
            via _flatten_struct_fields exactly the way struct equality
            already flattens them for comparison -- recursing back
            into THIS method for each field's own (never struct-kind,
            since flattening already unwrapped any nested struct away)
            type.

        dst_mem.base is protected via push/pop across EVERY field's own
        zero-fill, when it isn't 'rbp' (rbp itself is never at risk --
        nothing in this file ever treats it as scratch, so pushing and
        popping it would be actively wrong, not merely unnecessary; see
        gen_struct_literal_into's own identical `!= 'rbp'` guard for
        the same reasoning applied to a different write). Needed
        because the array case below computes a fresh address by
        calling _gen_address_of_memory_into with dst_mem.base itself as
        the destination register in some call shapes -- which, unlike
        every scalar/slice write here, can OVERWRITE dst_mem.base's own
        physical register in place. Without protecting it, a struct
        with an array-typed field followed by ANY other field would
        silently compute that later field's own address from garbage
        (whatever the array zero-loop happened to leave behind) instead
        of the struct's real base -- the exact register-collision
        failure mode _gen_struct_fields_equality_at_addresses's own
        docstring already documents at length for the identical reason,
        one construct over. Applied unconditionally, even for the
        scalar/slice cases that don't strictly need it, matching that
        same method's own "protect everything, don't try to be clever
        about which fields actually need it" posture."""
        if t.kind == TypeKind.STRUCT:
            protect_dst = dst_mem.base != 'rbp'
            instructions = []
            for field_type, offset in self._flatten_struct_fields(t.struct_name):
                field_mem = Memory(dst_mem.base, dst_mem.offset + offset)
                if protect_dst:
                    instructions.append(Push(Register(dst_mem.base)))
                instructions.extend(self._gen_zero_value_into(field_type, field_mem))
                if protect_dst:
                    instructions.append(Pop(Register(dst_mem.base)))
            return instructions
        if t.kind == TypeKind.ARRAY:
            return self._gen_zero_array_into(t, dst_mem)
        if t.kind == TypeKind.SLICE:
            return self.gen_none_into(dst_mem, t)
        if t == Type.STR:
            # Whichever of rax/rcx isn't dst_mem's own base -- a single
            # scratch register is all this needs, computed and consumed
            # in the same two instructions, with nothing relying on it
            # afterward.
            scratch = Register('rax') if dst_mem.base != 'rax' else Register('rcx')
            return [
                LeaQ(label=self._get_empty_str_label(), dst=scratch),
                MovQ(src=scratch, dst=dst_mem),
            ]
        if t == Type.INT8 or t == Type.UINT8:
            return [MovB(src=Imm(0), dst=dst_mem)]
        return [Mov(src=Imm(0), dst=dst_mem)]  # int or bool

    def _gen_zero_array_into(self, array_type: Type, dst_mem: Memory) -> List[Instruction]:
        """Zeroes a whole array -- dispatching on the array's own LEAF
        type (see leaf_type) into one of three genuinely different
        strategies, mirroring array equality's own identical three-way
        split (_gen_array_flat_byte_equality_loop/_gen_array_str_
        equality_loop/_gen_array_struct_equality_loop) for the same
        underlying reason: a leaf's own zero-value representation
        determines whether the whole array can be zeroed as one flat
        run of raw bytes, or needs a real per-element write.

          - int8/uint8, int, bool, OR SLICE leaf: _gen_array_flat_zero_
            loop. All four types' own zero value is ALL RAW ZERO BYTES
            with no pointer or other special representation (a slice's
            own none-shaped {0, 0, 0} descriptor -- see gen_none_
            into -- IS 24 zero bytes, nothing more), so the WHOLE
            array, however many elements and however deeply nested, is
            zeroed as one flat run -- the same "treat a nested array as
            one contiguous block" trick gen_array_copy/array equality's
            own flat-byte loop already rely on. Array equality couldn't
            offer slice this same treatment (a slice-typed array
            element isn't COMPARABLE at all yet, so it never reached
            that dispatch), but zeroing has no such restriction: there
            being nothing to compare, only a zero value to write, is
            exactly what makes slice fit here for free. int8/uint8 need
            a 1-byte STEP through that same flat run rather than
            int/bool/slice's own 4-byte one -- see _gen_array_flat_
            zero_loop's own docstring for why (the identical "total_
            width isn't generally a multiple of 4 for a genuinely
            1-byte-wide leaf" reasoning array equality's own flat loop
            already needed fixing for).
          - str leaf: _gen_array_str_zero_loop -- a str's own zero
            value is a POINTER (see _gen_zero_value_into's own STR
            case), so each element needs that same address written
            individually, not raw zero bytes.
          - struct leaf: _gen_array_struct_zero_loop -- a struct's own
            fields can be a MIX of types, so there's no single flat-
            bytes-or-repeated-pointer strategy that covers a whole
            struct-typed element; each one is zeroed via a recursive
            call back into _gen_zero_value_into itself."""
        leaf = leaf_type(array_type)
        total_width = type_byte_width(array_type, self.struct_registry)
        if leaf == Type.STR:
            return self._gen_array_str_zero_loop(dst_mem, total_width // 8)
        if leaf.kind == TypeKind.STRUCT:
            struct_width = type_byte_width(leaf, self.struct_registry)
            return self._gen_array_struct_zero_loop(dst_mem, leaf.struct_name, total_width // struct_width)
        step = 1 if leaf in (Type.INT8, Type.UINT8) else 4
        return self._gen_array_flat_zero_loop(dst_mem, total_width, step=step)

    def _gen_array_flat_zero_loop(self, dst_mem: Memory, total_width: int, step: int = 4) -> List[Instruction]:
        """Zeroes `total_width` bytes at dst_mem, `step` bytes at a
        time -- see _gen_zero_array_into's own docstring for why 4 is
        correct for an int, bool, or slice leaf at any nesting depth,
        via a plain 4-byte Mov of Imm(0).

        1 is correct instead for an int8/uint8 leaf (via a 1-byte
        MovB, rather than a 4-byte Mov, of that same Imm(0)) for the
        identical reason _gen_array_flat_byte_equality_loop's own step
        parameter exists: type_byte_width no longer guarantees total_
        width is a multiple of 4 once a genuinely 1-byte-wide leaf
        exists (a [3]int8 array is 3 bytes total) -- stepping 4 bytes
        at a time regardless would write one byte past the end of the
        array, corrupting whatever stack memory happens to follow it.

        No calls happen anywhere in this loop, so every register here
        is ordinary caller-saved scratch, freely reusable by whatever
        runs after this method returns -- the same posture _gen_array_
        flat_byte_equality_loop's own identical loop shape already
        takes, for the identical reason.

        Computes dst_mem's own starting address into a FIXED register
        (%r10) via _gen_address_of_memory_into, rather than assuming
        dst_mem.base itself remains valid to keep reading from
        directly -- correct even when dst_mem.base happens to BE %r10
        already (that method's own self-copy-then-add shape handles
        that case safely), but this method never relies on dst_mem's
        own base surviving its own execution either way; any caller
        that needs dst_mem.base to still be valid AFTERWARD (see _gen_
        zero_value_into's own struct case) is responsible for
        protecting it externally, via push/pop, before calling this."""
        base_reg = Register('r10')
        instructions = self._gen_address_of_memory_into(dst_mem, base_reg)
        index_32 = Register('ecx')
        index_64 = Register('rcx')
        write_addr = Register('r11')
        loop_start = self.new_label("array_zero_flat_loop")
        loop_done = self.new_label("array_zero_flat_done")
        instructions.append(Mov(src=Imm(0), dst=index_32))
        instructions.append(Label(loop_start))
        instructions.append(Cmp(src=Imm(total_width), dst=index_32))
        instructions.append(Jae(loop_done))
        instructions.append(MovQ(src=base_reg, dst=write_addr))
        instructions.append(AddQ(src=index_64, dst=write_addr))
        if step == 1:
            instructions.append(MovB(src=Imm(0), dst=Memory(write_addr.name, 0)))
        else:
            instructions.append(Mov(src=Imm(0), dst=Memory(write_addr.name, 0)))
        instructions.append(Add(src=Imm(step), dst=index_32))
        instructions.append(Jmp(loop_start))
        instructions.append(Label(loop_done))
        return instructions

    def _gen_array_str_zero_loop(self, dst_mem: Memory, element_count: int) -> List[Instruction]:
        """The str-leaf counterpart to _gen_array_flat_zero_loop just
        above: computes the shared empty-string address ONCE, then
        writes that same 8-byte value into each of `element_count`
        consecutive element slots. No calls happen here either (LeaQ
        computes a RIP-relative address, it doesn't call anything), so
        -- like the flat-zero loop -- every register is ordinary
        caller-saved scratch, with nothing here relying on it surviving
        past this method's own return."""
        base_reg = Register('r10')
        instructions = self._gen_address_of_memory_into(dst_mem, base_reg)
        empty_str_reg = Register('r11')
        instructions.append(LeaQ(label=self._get_empty_str_label(), dst=empty_str_reg))
        total_width = element_count * 8
        offset_32 = Register('ecx')
        offset_64 = Register('rcx')
        write_addr = Register('r8')
        loop_start = self.new_label("array_zero_str_loop")
        loop_done = self.new_label("array_zero_str_done")
        instructions.append(Mov(src=Imm(0), dst=offset_32))
        instructions.append(Label(loop_start))
        instructions.append(Cmp(src=Imm(total_width), dst=offset_32))
        instructions.append(Jae(loop_done))
        instructions.append(MovQ(src=base_reg, dst=write_addr))
        instructions.append(AddQ(src=offset_64, dst=write_addr))
        instructions.append(MovQ(src=empty_str_reg, dst=Memory(write_addr.name, 0)))
        instructions.append(Add(src=Imm(8), dst=offset_32))
        instructions.append(Jmp(loop_start))
        instructions.append(Label(loop_done))
        return instructions

    def _gen_array_struct_zero_loop(self, dst_mem: Memory, struct_name: str, element_count: int) -> List[Instruction]:
        """The struct-leaf counterpart to the two loops above: each
        element is zeroed via a recursive call back into _gen_zero_
        value_into, since a struct's own fields can be a mix of types
        with no single flat-bytes-or-repeated-value strategy.

        Unlike its two siblings, this loop's own base address (%r12)
        and index (%r13) DO need protecting across that recursive
        call: the struct being zeroed could itself have an array-typed
        field, which would dispatch back into one of THESE SAME THREE
        loops (reusing the identical fixed register names, since
        there's no way to hand out a fresh, distinct set per nesting
        depth at codegen time) -- silently corrupting this outer
        loop's own %r12/%r13 if they weren't saved first. Protected
        via an ordinary push/pop pair around the one recursive call,
        exactly like _gen_array_struct_equality_loop's own identical
        situation (see its own docstring for the fuller explanation of
        why this is correct at any nesting depth, by induction: every
        level protects only what IT locally needs to survive, trusting
        nothing else about what runs in between). %r14 (the per-
        iteration byte offset) needs no such protection, for the same
        reason it doesn't there either: always freshly recomputed at
        the start of an iteration, never read again after computing
        this iteration's own element address."""
        struct_width = type_byte_width(Type(TypeKind.STRUCT, struct_name=struct_name), self.struct_registry)
        base_reg = Register('r12')
        instructions = self._gen_address_of_memory_into(dst_mem, base_reg)
        index_32 = Register('r13d')
        index_64 = Register('r13')
        offset_32 = Register('r14d')
        offset_64 = Register('r14')
        elem_addr = Register('r10')
        loop_start = self.new_label("array_zero_struct_loop")
        loop_done = self.new_label("array_zero_struct_done")
        instructions.append(Mov(src=Imm(0), dst=index_32))
        instructions.append(Label(loop_start))
        instructions.append(Cmp(src=Imm(element_count), dst=index_32))
        instructions.append(Jae(loop_done))
        instructions.append(Mov(src=index_32, dst=offset_32))
        instructions.append(IMul(src=Imm(struct_width), dst=offset_32))
        instructions.append(MovQ(src=base_reg, dst=elem_addr))
        instructions.append(AddQ(src=offset_64, dst=elem_addr))
        instructions.append(Push(base_reg))
        instructions.append(Push(index_64))
        instructions.extend(self._gen_zero_value_into(
            Type(TypeKind.STRUCT, struct_name=struct_name), Memory(elem_addr.name, 0)
        ))
        instructions.append(Pop(index_64))
        instructions.append(Pop(base_reg))
        instructions.append(Add(src=Imm(1), dst=index_32))
        instructions.append(Jmp(loop_start))
        instructions.append(Label(loop_done))
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
        slice_expr = expr.left if type_of(expr.left).kind == TypeKind.SLICE else expr.right
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

    def gen_binary_op(self, op: BinaryOp, src: Operand, dst: Operand, operand_type: Type = Type.INT) -> List[Instruction]:
        """Emits the actual operator instruction(s) for `op`, given
        that `src`/`dst` already hold the right-hand/left-hand values
        (see gen_binary_into's own orchestration for how they get
        there). `src`/`dst` are always passed as their ordinary
        32-bit-named register (e.g. 'eax'/'ecx'), the SAME convention
        every other caller in this file follows -- this method itself
        decides internally, via `operand_type`, whether to actually
        operate on that register's own 64-bit VIEW (as_qword_register)
        for int64, exactly the same "caller always passes the 32-bit
        name, the callee decides which view to use" pattern _gen_read_
        scalar_into/_gen_write_scalar_from already established for
        int8/uint8/int64's own storage access -- rather than pushing
        that decision out onto gen_binary_into or every other call
        site.

        A COMPARISON's own RESULT, though, is always an ordinary
        32-bit bool regardless of operand_type: SetCC/MovZX always
        target dst's own 32-bit view even when the comparison itself
        (Cmp vs CmpQ) operated on its 64-bit one, since a bool value
        is never itself wider than 4 bytes no matter how wide the two
        values being compared were -- this is why the comparison
        branch converts to a 64-bit view locally, just for the Cmp/
        CmpQ instruction itself, rather than reusing a single dst64
        variable the way every arithmetic branch above it does."""
        is_64bit = operand_type == Type.INT64
        if is_64bit and op not in COMPARISON_CONDITION_CODES:
            src64 = as_qword_register(src)
            dst64 = as_qword_register(dst)
            if op == BinaryOp.ADD:
                return [AddQ(src=src64, dst=dst64)]
            if op == BinaryOp.SUBTRACT:
                return [SubQ(src=src64, dst=dst64)]
            if op == BinaryOp.MULTIPLY:
                return [IMulQ(src=src64, dst=dst64)]
            if op == BinaryOp.DIVIDE:
                # idivq divides %rdx:%rax by its operand, so by the time
                # this runs, the dividend (dst64, i.e. left) must be in
                # %rax and the divisor (src64, i.e. right) must be in a
                # register -- both guaranteed by how gen_binary_into
                # calls this, exactly like the 32-bit case below.
                if dst64 != Register('rax'):
                    raise CodegenError("Division currently requires its destination to be %rax")
                return [Cqto(), IDivQ(src64)]
            if op == BinaryOp.MODULO:
                # Exactly the same Cqto+IDivQ sequence as DIVIDE --
                # idivq always computes both the quotient (%rax) and the
                # remainder (%rdx) in one instruction -- just followed
                # by moving the remainder into dst64 instead of leaving
                # the quotient there.
                if dst64 != Register('rax'):
                    raise CodegenError("Modulo currently requires its destination to be %rax")
                return [Cqto(), IDivQ(src64), MovQ(src=Register('rdx'), dst=Register('rax'))]
            if op == BinaryOp.BITWISE_AND:
                return [AndQ(src=src64, dst=dst64)]
            if op == BinaryOp.BITWISE_OR:
                return [OrQ(src=src64, dst=dst64)]
            if op == BinaryOp.BITWISE_XOR:
                return [XorQ(src=src64, dst=dst64)]
            if op == BinaryOp.SHIFT_LEFT:
                # `src64` is never referenced here -- ShiftLeftQ
                # hardcodes %cl as its count operand, the identical
                # reason ShiftLeft's own docstring already explains one
                # register-width down; the count itself is never wider
                # than a byte regardless of the value being shifted.
                return [ShiftLeftQ(dst=dst64)]
            if op == BinaryOp.SHIFT_RIGHT:
                return [ShiftRightArithmeticQ(dst=dst64)]
            raise CodegenError(f"No codegen rule for binary operator: {op}")

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
        if op in COMPARISON_CONDITION_CODES:
            # Cmp(src=right, dst=left) computes (left - right) and sets
            # flags from that; SetCC turns the relevant flag combination
            # into a 0/1 byte; MovZX zero-extends that byte back out to
            # fill the full destination register (same pattern used for
            # NOT -- see gen_unary_op -- just against a computed `right`
            # instead of the literal 0). For a 64-bit operand_type, the
            # comparison ITSELF (CmpQ, against the 64-bit views) needs
            # the full value to compare correctly -- comparing only the
            # low 32 bits could, e.g., call two large int64 values equal
            # when they aren't -- but the RESULT byte/register (byte_dst,
            # dst) stays exactly as it already was: a bool is always
            # 32-bit-or-narrower regardless of what was being compared.
            byte_dst = as_byte_register(dst)
            cmp_instr = CmpQ(src=as_qword_register(src), dst=as_qword_register(dst)) if is_64bit else Cmp(src=src, dst=dst)
            return [
                cmp_instr,
                SetCC(cc=COMPARISON_CONDITION_CODES[op], operand=byte_dst),
                MovZX(src=byte_dst, dst=dst),
            ]
        raise CodegenError(f"No codegen rule for binary operator: {op}")

    def gen_unary_op(self, op: UnaryOp, dst: Operand, operand_type: Type = Type.INT) -> List[Instruction]:
        """`operand_type` follows the identical convention gen_binary_
        op's own new parameter does -- `dst` is always passed as its
        ordinary 32-bit-named register, and this method decides
        internally whether to operate on its 64-bit view for int64.
        UnaryOp.NOT never reaches the int64 branch at all: `not`
        requires a bool operand (see check_unary), which int64 can
        never be, so its own Cmp-against-0/SetCC/MovZX sequence stays
        exactly as it always has, unconditionally 32-bit."""
        if op == UnaryOp.NEGATE:
            if operand_type == Type.INT64:
                return [NegQ(as_qword_register(dst))]
            return [Neg(dst)]
        if op == UnaryOp.COMPLEMENT:
            if operand_type == Type.INT64:
                return [NotQ(as_qword_register(dst))]
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
