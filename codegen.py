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
"""

import argparse
from dataclasses import dataclass, field
from typing import Dict, List

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
    Param,
    Parser,
    Program,
    Return,
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
    8 for str (a pointer), and recursively `size *
    type_byte_width(element_type)` for an array -- its full, flattened
    stack footprint, matching how it's laid out contiguously in
    row-major order regardless of how many dimensions it has (see the
    ARRAYS section). This is the one place that recursion lives; every
    caller that needs an array's total size (stack allocation, whole-
    array copies) or the shift-per-index (address computation) goes
    through this or leaf_type below rather than re-deriving either."""
    if t.kind == TypeKind.ARRAY:
        return t.size * type_byte_width(t.element_type)
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
    needed once this is known."""
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
    inline in its own stack slot -- true for an array type whose total
    footprint (type_byte_width) exceeds _STACK_ARRAY_LIMIT_BYTES, false
    for every scalar type and every array under the limit. This is the
    one place that decision is made; every caller that needs to know
    -- stack allocation width, how to compute a variable's address,
    how to read or write its value -- goes through this rather than
    re-deriving it, so the threshold only ever needs to change in one
    place. Purely a function of the type itself, not of any per-
    variable state, so it's never stored anywhere -- anywhere codegen
    already has the Type (via _local_type or _type_of), it can just
    call this directly."""
    return t.kind == TypeKind.ARRAY and type_byte_width(t) > _STACK_ARRAY_LIMIT_BYTES


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
        # label's own docstrings. The fail label is reset per function
        # (gen_function); the message label, like the print-related
        # ones above, is cached for the whole compilation.
        self._bounds_check_fail_label = None
        self._bounds_check_message_label = None
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
        return_type = type_from_name(fn.return_type)

        # An array-typed return needs a hidden pointer -- the caller
        # passes the address to write the result into, as an extra,
        # FIRST argument (see gen_array_call_into), shifting every real
        # parameter one register position later. Rather than dedicate
        # a register to holding it for the whole function (which would
        # need its own save/restore discipline, and -- worse -- would
        # break the callee-saved-register prologue's even-push-count
        # alignment invariant if added on top of the existing four),
        # it just gets its own ordinary stack slot, handled by exactly
        # the same "reserve a slot, then store the incoming register
        # into it" mechanism every real parameter already uses. See
        # the module docstring's ARRAYS section.
        self._hidden_return_ptr_offset = None
        arg_shift = 0
        if return_type.kind == TypeKind.ARRAY:
            self._next_offset -= 8
            self._hidden_return_ptr_offset = self._next_offset
            arg_shift = 1

        # One extra, purely internal 8-byte slot per parameter, used to
        # stash its incoming register value immediately, before any
        # parameter is actually processed -- see the loop below for
        # why this has to happen up front rather than processing each
        # parameter directly out of its own argument register in turn.
        param_temp_offsets = []
        for _ in fn.params:
            self._next_offset -= 8
            param_temp_offsets.append(self._next_offset)

        self._collect_params(fn.params)
        self._collect_locals(fn.body)
        self.scopes = [{}]

        max_params = 6 - arg_shift
        if len(fn.params) > max_params:
            reason = (
                "the hidden output pointer itself uses the first argument "
                "register, leaving 5"
                if arg_shift else
                "this compiler only supports up to 6 (passed via registers "
                "per the SysV ABI -- stack-passed parameters aren't "
                "implemented)"
            )
            raise CodegenError(
                f"Function '{fn.name}' has {len(fn.params)} parameters; "
                f"{reason}"
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
        # array -- see arg_shift above). Handled in two passes rather
        # than reading each one directly out of its own argument
        # register in turn:
        #
        # FIRST, every incoming register is stashed into its own
        # temporary slot (param_temp_offsets, reserved above) via a
        # plain %rbp-relative store -- these never touch %rsp, so
        # there's no stack-alignment concern regardless of how many
        # parameters there are or which ones turn out to need malloc.
        #
        # SECOND, each parameter is processed using its safely-stashed
        # value rather than its original argument register. This
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
        for i in range(len(fn.params)):
            reg_index = i + arg_shift
            instructions.append(MovQ(src=Register(_ARG_REGISTERS_64[reg_index]), dst=Memory('rbp', param_temp_offsets[i])))

        for i, p in enumerate(fn.params):
            offset = self._bind_param(p)
            p_type = type_from_name(p.type)
            temp_offset = param_temp_offsets[i]
            if p_type.kind == TypeKind.ARRAY:
                if is_heap_allocated(p_type):
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
            elif p_type == Type.STR:
                instructions.append(MovQ(src=Memory('rbp', temp_offset), dst=Register('rax')))
                instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', offset)))
            else:
                instructions.append(Mov(src=Memory('rbp', temp_offset), dst=Register('eax')))
                instructions.append(Mov(src=Register('eax'), dst=Memory('rbp', offset)))

        self._bounds_check_fail_label = None  # fresh, per-function jump target; see its own docstring
        for stmt in fn.body:
            instructions.extend(self.gen_statement(stmt))
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
            width = 8 if is_heap_allocated(p_type) else type_byte_width(p_type)
            self._next_offset -= width
            self._var_offsets[id(p)] = self._next_offset

    def _bind_param(self, p: Param) -> int:
        """The Param counterpart to _bind_local -- registers `p`'s name
        and declared type (as a real semantic.Type, via type_from_name,
        not the raw parser-level string/ArrayTypeExpr -- see
        _local_type's own docstring for why) in the current scope,
        pointing at the permanent offset _collect_params already
        assigned it."""
        offset = self._var_offsets[id(p)]
        self.scopes[-1][p.name] = (offset, type_from_name(p.type))
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
                width = 8 if is_heap_allocated(var_type) else type_byte_width(var_type)
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
        """Registers `stmt`'s name -- and its declared type, needed by
        _local_type -- in the current (innermost) generation-time
        scope, pointing at the permanent offset _collect_locals already
        assigned this exact VarDecl node, and returns that offset."""
        offset = self._var_offsets[id(stmt)]
        self.scopes[-1][stmt.name] = (offset, type_from_name(stmt.var_type))
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
        (offset, Type) tuple in the same scope-stack entry, which
        codegen has to maintain regardless of _type_of's existence,
        since resolved_type has no way to encode *which* stack slot a
        name refers to. This is deliberately not replaced by _type_of
        below, even though it would give the same answer for a
        Variable node -- see _type_of's own docstring for why the two
        coexist rather than one replacing the other.

        Returns a real semantic.Type (via type_from_name, called once
        up front in _bind_local/_bind_param, not re-derived here) --
        not the raw parser-level string/ArrayTypeExpr -- so callers can
        uniformly inspect .kind/.element_type/.size exactly like they
        already can on whatever _type_of returns."""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name][1]
        raise CodegenError(f"Reference to undeclared variable '{name}'")

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
            if is_heap_allocated(array_type):
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

        Includes a runtime bounds check: an out-of-range index prints a
        message and calls abort() (see _gen_bounds_check_panic_block)
        rather than silently reading or writing adjacent stack memory
        -- which, given arrays live in the same frame as the saved
        return address and the callee-saved registers every function
        call already depends on, could otherwise corrupt exactly the
        state that keeps `call`/`ret` working correctly, not just
        return a wrong value.

        `expr.array`'s own address is computed first and protected on
        the real CPU stack (not a fixed register) while the index
        expression -- which could be arbitrarily complex, including
        another indexing operation or a function call -- is evaluated,
        the same push-before-recursing pattern used everywhere else in
        this file a value needs to survive evaluating something else.
        This works out correctly no matter what register `dst` itself
        is (including if it happens to coincide with the %rax/%rcx
        this method uses internally): the base address is safely on
        the stack while %rax/%rcx are used for the index/offset
        arithmetic, and the final address is only ever written into
        `dst` as the very last step.
        """
        array_type = self._type_of(expr.array)
        element_stride = type_byte_width(array_type.element_type)
        size = array_type.size

        instructions = self.gen_array_address_into(expr.array, dst)
        instructions.append(Push(dst))
        instructions.extend(self.gen_expr_into(expr.index, Register('eax')))
        # Unsigned comparison: catches index >= size AND index < 0 in
        # one check, since a negative int, reinterpreted unsigned,
        # becomes a huge positive number.
        instructions.append(Cmp(src=Imm(size), dst=Register('eax')))
        instructions.append(Jae(self._get_bounds_check_fail_label()))
        # A plain 32-bit imul is safe here: the bounds check above
        # already guarantees the index is small and non-negative, and
        # a 32-bit write zero-extends into the full 64-bit rax.
        instructions.append(IMul(src=Imm(element_stride), dst=Register('eax')))
        instructions.append(Pop(Register('rcx')))  # restore expr.array's base address
        instructions.append(AddQ(src=Register('rax'), dst=Register('rcx')))
        instructions.append(MovQ(src=Register('rcx'), dst=dst))
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
        ever changes."""
        used_bases = {src_mem.base, dst_mem.base}
        scratch_64, scratch_32 = next(
            (r64, r32) for r64, r32 in [('rax', 'eax'), ('rcx', 'ecx'), ('rdx', 'edx')]
            if r64 not in used_bases
        )
        width = type_byte_width(leaf_type(array_type))
        total = type_byte_width(array_type)
        instructions = []
        off = 0
        while off < total:
            src = Memory(src_mem.base, src_mem.offset + off)
            dst = Memory(dst_mem.base, dst_mem.offset + off)
            if width == 8:
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
        a sub-expression. Each argument is then pushed in turn (an
        array-typed one as an address via gen_array_arg_address_into,
        anything else as a value via the ordinary gen_expr_into) and
        popped back off in reverse, into registers shifted one position
        past the hidden pointer -- exactly gen_call_into's own
        push-then-pop-in-reverse pattern, just with that one-position
        shift threaded through.
        """
        if len(expr.args) > 5:
            raise CodegenError(
                f"Call to '{expr.name}' has {len(expr.args)} arguments; "
                f"a call to a function that returns an array supports "
                f"at most 5 (the hidden output pointer itself uses the "
                f"first argument register)"
            )
        instructions = self._gen_address_of_memory_into(dst_mem, Register('rax'))
        instructions.append(Push(Register('rax')))
        for arg in expr.args:
            arg_type = self._type_of(arg)
            if arg_type.kind == TypeKind.ARRAY:
                instructions.extend(self.gen_array_arg_address_into(arg, Register('rax')))
            else:
                instructions.extend(self.gen_expr_into(arg, Register('eax')))
            instructions.append(Push(Register('rax')))
        for i in reversed(range(len(expr.args))):
            instructions.append(Pop(Register(_ARG_REGISTERS_64[i + 1])))  # +1: shifted past the hidden pointer
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
        constant), except when the element type is ITSELF an array -- a
        multi-dimensional literal's "elements" are themselves
        ArrayLiterals, handled by recursing through gen_array_value_into
        (which dispatches straight back here).

        dst_mem's own base register is protected on the stack across
        each element's value computation whenever it isn't 'rbp' --
        found necessary by a real bug during development, not assumed:
        'rbp' (the frame pointer, used for every ordinary local slot)
        is never clobbered by gen_expr_into, so no protection is needed
        there, but a computed or received address held in a general-
        purpose register (e.g. Memory('rax', 0), the hidden return
        pointer for a literal returned directly -- `return [1,2,3]`)
        is exactly the kind of register gen_expr_into's own value
        computation, which always targets %eax/%rax, can and did
        clobber -- silently overwriting the destination address before
        a single element was ever actually written through it."""
        element_type = array_type.element_type
        element_width = type_byte_width(element_type)
        protect_dst = dst_mem.base != 'rbp'
        instructions = []
        for i, elem_expr in enumerate(expr.elements):
            elem_mem = Memory(dst_mem.base, dst_mem.offset + i * element_width)
            if element_type.kind == TypeKind.ARRAY:
                instructions.extend(self.gen_array_value_into(elem_expr, elem_mem, element_type))
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
            if is_heap_allocated(src_type):
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

    def _get_bounds_check_fail_label(self) -> str:
        """Lazily creates a single, per-function label that every
        bounds check within this function jumps to on failure --
        reused across however many indexing operations this function
        has, rather than duplicating the panic sequence (see
        _gen_bounds_check_panic_block) at every individual check site.
        Reset to None at the start of every function (see
        gen_function) -- unlike the message label below, this one is a
        purely LOCAL jump target, meaningless outside the function
        it's generated for."""
        if self._bounds_check_fail_label is None:
            self._bounds_check_fail_label = self.new_label("bounds_check_fail")
        return self._bounds_check_fail_label

    def _get_bounds_check_message_label(self) -> str:
        """Lazily creates and caches (for the rest of the WHOLE
        compilation, unlike the per-function label above -- this is
        just a static string, safely shared by every function that
        needs it, matching the same lazy-cache pattern print's own
        format-string/true/false labels already use) the "array index
        out of bounds" message string."""
        if self._bounds_check_message_label is None:
            self._bounds_check_message_label = self.new_label("bounds_msg")
            self.string_literals.append((self._bounds_check_message_label, "array index out of bounds"))
        return self._bounds_check_message_label

    def _gen_bounds_check_panic_block(self) -> List[Instruction]:
        """Appended once at the end of a function's own instructions
        (see gen_function) if -- and only if -- that function's own
        bounds checks ever actually used _get_bounds_check_fail_label.
        Prints a clear message, then calls abort() (SIGABRT) rather
        than a plain exit() -- an out-of-bounds access is a genuine
        program bug, not a normal termination condition, the same
        "abnormal termination" character division by zero's hardware-
        trapped SIGFPE already has, just deliberately raised by this
        compiler's own generated code instead of by the CPU. Never
        reached via ordinary fall-through from the function's own body
        -- every return already leaves via `leave; ret` before control
        could reach this point, and abort() itself never returns -- so
        appending it at the very end of the function is always safe.

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
        if self._bounds_check_fail_label is None:
            return []
        msg_label = self._get_bounds_check_message_label()
        return [
            Label(self._bounds_check_fail_label),
            LeaQ(label=msg_label, dst=Register('rdi')),
            CallInstr('puts'),
            Mov(src=Imm(0), dst=Register('edi')),
            CallInstr('fflush'),
            CallInstr('abort'),
        ]

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
        if is_heap_allocated(var_type):
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
        return self._gen_store(offset, stmt.init)

    def gen_assign(self, stmt: Assign) -> List[Instruction]:
        offset = self._local_offset(stmt.name)
        value_type = self._type_of(stmt.value)
        if value_type.kind == TypeKind.ARRAY and is_heap_allocated(value_type):
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
        The element's own type decides the store width exactly like
        _gen_store does for an ordinary variable -- str needs `movq`,
        everything else `movl`. An array-typed element (writing a
        whole sub-array via `matrix[i] = other_row`) isn't reachable
        here at all: IndexAssign's own grammar only ever produces a
        single leaf-level element write; a whole-row assignment would
        need `matrix[i]` to appear as an ordinary Assign target, which
        parser.py doesn't produce (see IndexAssign's own docstring).
        """
        element_type = self._type_of(stmt.value)
        addr_reg = Register('rax')
        instructions = self.gen_index_address_into(Index(array=stmt.array, index=stmt.index), addr_reg)
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
        separately (see its own docstring); a str is an 8-byte pointer
        sitting in %rax and needs `movq`; int/bool are still the
        original 4-byte `movl %eax, ...` -- everything about
        gen_expr_into/gen_binary_into/gen_unary_op's own internals
        stays exactly as it always has, oblivious to str (or arrays)
        entirely; only this one call site needs to ask "which width,
        or which entirely different mechanism, am I storing"."""
        value_type = self._type_of(value_expr)
        if value_type.kind == TypeKind.ARRAY:
            return self.gen_array_value_into(value_expr, Memory('rbp', offset), value_type)
        instructions = self.gen_expr_into(value_expr, Register('eax'))
        if value_type == Type.STR:
            instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', offset)))
        else:
            instructions.append(Mov(src=Register('eax'), dst=Memory('rbp', offset)))
        return instructions

    def gen_return(self, stmt: Return) -> List[Instruction]:
        # An array-typed return writes directly through the hidden
        # pointer this function received (see gen_function's own
        # prologue handling and the module docstring's ARRAYS section)
        # instead of ever putting anything in %eax/%rax -- nothing
        # reads a return value that way for an array-returning call
        # (see gen_array_value_into's Call case, which is the only way
        # such a call's result is ever consumed). Loading the pointer
        # back out of its slot and handing it to gen_array_value_into
        # as an ordinary Memory destination is also what makes
        # `return bar()` (forwarding another array-returning call's
        # result straight out) free: gen_array_value_into's own Call
        # case just passes that SAME address one level deeper via
        # gen_array_call_into, with no intermediate copy ever
        # materialized.
        value_type = self._type_of(stmt.value)
        if value_type.kind == TypeKind.ARRAY:
            ptr_reg = Register('rax')
            instructions = [MovQ(src=Memory('rbp', self._hidden_return_ptr_offset), dst=ptr_reg)]
            instructions.extend(self.gen_array_value_into(stmt.value, Memory('rax', 0), value_type))
        else:
            dst = Register('eax')
            instructions = self.gen_expr_into(stmt.value, dst)
        # Restore the callee-saved scratch registers *before* Leave --
        # Leave resets %rsp straight to %rbp, which was captured before
        # these were pushed in the prologue, so anything pushed after
        # that point has to be popped explicitly first or it's just
        # silently discarded (never actually restored into the
        # registers) rather than popped. Popping happens in reverse of
        # the prologue's push order, the usual stack discipline. None of
        # this touches %eax/%rax, so a scalar return value computed
        # above is unaffected regardless of what these registers held
        # during the body (e.g. if the return expression itself did
        # string work that reused them as scratch in between).
        for reg in reversed(_CALLEE_SAVED_SCRATCH_REGISTERS):
            instructions.append(Pop(Register(reg)))
        instructions.append(Leave())
        instructions.append(Ret())
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
        # instructions that really run; see the module docstring for how
        # that's verified (a standalone `1 / 0` genuinely crashes).
        return self.gen_expr_into(stmt.expr, Register('eax'))

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
            if expr.name == 'print':
                return self.gen_print_call_into(expr, dst)
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

        scratch = Register('ecx')  # holds the right-hand value while combining
        instructions = self.gen_expr_into(expr.left, dst)   # dst = left
        instructions.append(Push(as_qword_register(dst)))   # save left on the stack
        instructions.extend(self.gen_expr_into(expr.right, dst))  # dst = right (left is safe)
        instructions.append(Mov(src=dst, dst=scratch))       # scratch = right
        instructions.append(Pop(as_qword_register(dst)))     # dst = left (restored)
        instructions.extend(self.gen_binary_op(expr.op, src=scratch, dst=dst))
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
        """`name(arg1, arg2, ...)`: evaluates every argument -- in
        order, each via the ordinary gen_expr_into, so an argument that
        is itself a nested call, a string concatenation, or any other
        arbitrarily complex expression works correctly -- immediately
        pushing each one's result onto the stack before moving on to the
        next. Only *after* every argument has been safely computed and
        stacked does this start popping them back off, in reverse, into
        the actual SysV argument registers (see _ARG_REGISTERS_64).

        This "compute and stack everything, then pop into place" order
        is what avoids the same register-clobbering hazard that
        motivated saving %rbx/%r12/%r13/%r14 across calls in the first
        place (see the module docstring's FUNCTIONS section): if
        argument 2 happens to be a string concatenation and argument 1's
        value were sitting in a scratch register instead of safely on
        the stack while argument 2 gets computed, argument 2's own use
        of that same scratch register would corrupt argument 1.

        The result already ends up exactly where gen_expr_into's
        contract expects it (%rax/%eax, matching `dst`, which is always
        Register('eax') throughout this file), so there's nothing left
        to move once the call returns.
        """
        if len(expr.args) > 6:
            raise CodegenError(
                f"Call to '{expr.name}' has {len(expr.args)} arguments; "
                f"this compiler only supports up to 6 (passed via "
                f"registers per the SysV ABI -- stack-passed arguments "
                f"aren't implemented)"
            )
        if dst != Register('eax'):
            raise CodegenError(f"Call codegen requires dst == %eax, got: {dst!r}")

        instructions: List[Instruction] = []
        for arg in expr.args:
            arg_type = self._type_of(arg)
            if arg_type.kind == TypeKind.ARRAY:
                # This callee doesn't itself return an array (that's
                # gen_array_call_into's job), but it can still ACCEPT
                # one as a parameter -- e.g. `def int sum_array([3]int
                # arr): ...` -- passed the same way either way: the
                # address of an existing variable or sub-array, per
                # gen_array_arg_address_into's own restriction, since
                # the callee copies from it into its own local slot on
                # entry regardless of which kind of call brought it in.
                instructions.extend(self.gen_array_arg_address_into(arg, Register('rax')))
            else:
                instructions.extend(self.gen_expr_into(arg, Register('eax')))
            instructions.append(Push(Register('rax')))
        for i in reversed(range(len(expr.args))):
            instructions.append(Pop(Register(_ARG_REGISTERS_64[i])))
        instructions.append(CallInstr(expr.name))
        return instructions

    def gen_print_call_into(self, expr: Call, dst: Operand) -> List[Instruction]:
        """`print(x)`: dispatches on x's *compile-time* type -- known
        exactly, since Hornet is statically typed -- to one of three
        completely different instruction sequences, each calling a
        different libc function. See the module docstring's BUILTINS
        section for why each type gets its own call rather than one
        shared, format-driven path.

          str:  puts(x)                    -- puts adds its own newline
          int:  printf("%d\\n", x)         -- needs real formatting
          bool: puts(x ? "true" : "false") -- a runtime branch (the
                exact same cmp/je/jmp/label shape gen_if already uses)
                picks which string literal's address to pass, then
                falls through to the same puts call as the str case

        Every path ends with `movl $0, %eax`, overriding whatever
        puts/printf actually returned -- print's "return value" is a
        clean, predictable 0 (see semantic.py's check_print_call),
        never leaking the underlying libc call's own return convention
        into the language.

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
