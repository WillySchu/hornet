"""
test_compiler.py

Consolidated pytest suite for the lexer -> parser -> semantic -> codegen
pipeline.

Every execution-based test here compiles a small program through the
real pipeline, assembles and links it with gcc, actually *runs* the
resulting binary, and checks its exit code (or, for the short-circuit
"control" tests, that it crashes with a specific signal). This is
deliberately not just inspecting the generated assembly text -- actually
executing the binary is the strongest available check that codegen is
correct, not merely plausible-looking. Semantic-error tests are
different in kind: they assert that bad programs are *rejected* before
codegen ever runs, so they don't need gcc at all and aren't skipped when
it's unavailable (see GCC_SKIP below).

Requires `gcc` (and `as`/`ld`, which it drives) on PATH for the
execution-based classes. If it's not found, those are skipped with a
clear reason rather than failing with a wall of "file not found" errors.

Run with:
    pytest test_compiler.py -v

Organization:
    TestUnaryOperators                 ( 7 tests)
    TestBinaryArithmetic                (18 tests)
    TestBitwiseAndModuloOperators       (18 tests)
    TestComparisons                     (12 tests)
    TestPrecedenceAndLogicalOperators   ( 8 tests)
    TestShortCircuitEvaluation          ( 4 tests)
    TestVariablesAndStatements          (12 tests)
    TestCompoundAssignment              (16 tests)
    TestIfStatements                    (18 tests)
    TestWhileLoops                      (10 tests)
    TestStrings                         (13 tests)
    TestStringMemory                    (12 tests)
    TestFunctions                       (12 tests)
    TestFunctionsWithNoDeclaredReturnType (12 tests)
    TestTypeAnnotation                  ( 4 tests)
    TestPrint                           (12 tests)
    TestAllPathsReturn                  (18 tests)
    TestArrays                          (26 tests)
    TestBoundsChecking                  ( 4 tests)
    TestHeapAllocatedArrays             (12 tests)
    TestSlices                          (10 tests)
    TestSliceBoundsChecking             ( 6 tests)
    TestSliceParametersAndReturns       (15 tests)
    TestPrintArraysAndSlices            (11 tests)
    TestNone                            (17 tests)
    TestSemanticErrors                  (75 tests)
                                        ----------
                                        382 tests total

A NOTE ON ARRAYS
-----------------------------------------------------------------
Fixed-size, stack-allocated, value-typed (see codegen.py's ARRAYS
section for the full design). TestArrays' value-semantics tests are
the ones that actually prove the headline design decision holds at the
machine-code level, not just conceptually: `b = a; b[0] = 99` must
leave `a[0]` completely untouched, for both 1D and 2D arrays, and for
a sub-array extracted via `[3]int row = matrix[i]` too -- and, once
arrays could cross a function boundary, for a parameter mutated inside
a callee and a value returned from separate calls to the same function
as well.

TestBoundsChecking exists because every array access is runtime-
checked -- one unsigned comparison catches both an over-large index
and a negative one at once. test_panic_message_survives_piped_output
is a genuine regression test, not a hypothetical one: during
development, the "array index out of bounds" message was reliably
printed to an interactive terminal but silently LOST whenever output
was piped or redirected (the common case for a program run non-
interactively), because abort() bypasses the normal exit() path that
would otherwise flush libc's buffered stdio. Caught only by explicitly
capturing and checking stdout, not by eyeballing an interactive run --
worth remembering as a reason to prefer capture_output-based
assertions over manual spot-checks for anything involving abort()/
exit() specifically.

Array function parameters and return values are fully supported, via
a hidden-output-pointer convention for returns and copy-on-entry for
parameters (see codegen.py's ARRAYS section). Getting there surfaced
the same register-clobbering mistake in three separate call paths --
test_array_return_direct_literal and test_array_return_via_sub_array_
index are both genuine regression tests for segfaults found during
development, not hypothetical edge cases: a destination address held
in a general-purpose register (the hidden pointer, sitting in %rax)
was silently overwritten by code that assumed it was free to use that
same register as scratch, in three different places, each only found
by testing a different shape of return value rather than trusting one
passing test to mean the whole mechanism was sound. What's still a
deliberate, explicit gap -- not silently missing -- is an ArrayLiteral
or a call returning an array used DIRECTLY as a function-call argument
(`foo([1,2,3])`); test_array_literal_as_direct_call_argument_not_
supported confirms this fails with a clear error pointing at the
workaround (assign it to a variable first) rather than a confusing
crash.

A NOTE ON TestHeapAllocatedArrays
-----------------------------------------------------------------
An array over 16KB (codegen.py's _STACK_ARRAY_LIMIT_BYTES, hardcoded --
see is_heap_allocated) is heap-allocated instead of living inline on
the stack, closing off the one concrete danger fixed-size arrays
already had before any of this existed: nothing stopped a single huge
array from silently blowing the stack. test_exactly_at_threshold_
stays_on_stack and test_just_over_threshold_is_heap_allocated both
inspect the generated assembly directly for the presence or complete
absence of a malloc call, rather than only checking an exit code --
proof the boundary itself is exactly right, not just that some array
somewhere behaves plausibly.

Heap-promoting an array turned out to touch nearly every piece of
array codegen as a genuinely separate code path, not a transparent
allocator swap -- test_heap_allocated_local_with_literal_initializer
specifically exercises gen_var_decl's malloc-then-store-initializer
path, which is fully distinct code from its malloc-then-copy path for
a Variable/Index/Call initializer, and
test_multiple_heap_allocated_parameters stress-tests gen_function's
two-pass parameter handling: every incoming argument register gets
stashed into its own slot before any parameter is processed, since a
heap-allocated parameter's malloc call -- like any real call -- can
clobber other, not-yet-processed parameters' own incoming values. An
earlier, simpler attempt at protecting those registers (an ordinary
push, popped immediately before each parameter was processed) turned
out to misalign %rsp for roughly half of them; see codegen.py's own
ARRAYS section for why. test_heap_allocated_array_as_return_type is
the positive control proving array returns needed no changes at all --
they already write through a caller-provided pointer regardless of
size.

A NOTE ON TestSlices AND TestSliceBoundsChecking
-----------------------------------------------------------------
A slice is a Go-style VIEW into an existing array or slice's own
backing storage -- a fixed {pointer, length} descriptor (16 bytes),
not a copy the way plain array assignment already is (see codegen.py's
SLICES section for the full design). test_slice_write_mutates_
underlying_array and test_overlapping_slices_alias_each_others_writes
are the two tests that actually prove that holds at the machine-code
level, not just conceptually -- writing through a slice has to be
visible through the array it came from, and through any OTHER slice
that overlaps it, or slices would just be a more awkward way to copy
an array.

test_slicing_a_slice is the hardest case exercised directly: the base
being sliced is itself a slice, so its own length is a runtime value
read out of its descriptor rather than a compile-time constant the
way an array base's is -- this is what actually exercises
gen_indexable_base_into's two different code paths, not just the
array one.

TestSliceBoundsChecking exists because slice bounds needed a genuinely
different comparison from ordinary indexing's, not just a reused
check with a different message: `low == length` and `high == length`
are both VALID slice bounds (`arr[5:5]` is a valid, empty-slice-
producing expression), unlike an ordinary index, where being equal to
the array's own size is already invalid. test_low_equals_high_equals_
length_is_valid is the positive control proving that boundary is
exactly right (a strict `ja`, not `jae`) -- getting this wrong in
either direction would either reject valid empty slices or silently
accept a genuinely out-of-range one.

test_slice_parameter and test_slice_return, in this class, cover the
basic shape of crossing a function boundary at all -- they replaced
what used to be test_slice_parameter_not_supported_yet and test_slice_
return_not_supported_yet, back when neither was implemented. Both gaps
were caught by deliberately compiling exactly that case and checking
the generated assembly, before any test asserted anything about it:
neither was rejected at first, and a slice's own 16-byte descriptor
was silently truncated down to whatever fit in one 32-bit register
instead. See TestSliceParametersAndReturns for the full calling-
convention implementation and its own, much more thorough coverage
(register-slot interleaving, the exact 6-vs-7-slot boundary, aliasing
across a call, forwarding a returned slice for free, and more).

A NOTE ON TestSliceParametersAndReturns
-----------------------------------------------------------------
A slice crosses a function boundary via TWO registers directly -- its
own ptr, then len -- matching exactly what a real C compiler does for
an equivalent `struct{void*,long}` passed or returned by value under
the SysV ABI: as a parameter, it consumes two consecutive integer
argument registers; as a return value, it comes back in %rax:%rdx.
Neither a hidden pointer (arrays' own return convention) nor a copy
(arrays' own parameter convention) is involved at all -- a slice
parameter is just an alias crossing the boundary, exactly like any
other slice variable.

test_slice_interleaved_with_scalar_parameters and test_two_slices_
and_two_scalars_are_exactly_six_slots are the tests that actually
prove the trickiest part of this feature holds, not just that a slice
CAN be a parameter: since a slice now costs 2 of the 6 available
argument-register slots instead of 1, the mapping from argument/
parameter INDEX to register INDEX stopped being 1:1 on both the
caller side (_gen_call_arguments_into) and the callee side
(gen_function's own parameter loop) -- both now track a running slot
count instead, and these tests confirm a slice's own two slots land
correctly among ordinary scalar ones regardless of position, not just
when a slice happens to be the only or the last parameter.

test_exactly_six_slots_from_three_slice_parameters and test_seven_
slots_from_three_slices_and_a_scalar_is_rejected are the positive/
negative pair proving the boundary itself is exactly right: 6 slots
accepted, 7 cleanly rejected with a clear message -- not silently
truncated or off by one in either direction, which an easy mis-count
in the running-slot arithmetic could otherwise produce without any
test noticing.

test_writing_through_a_slice_parameter_mutates_callers_array is the
test that actually proves a slice parameter is a genuine alias
crossing the function boundary, not a copy -- the same aliasing
guarantee slices already have within a single function, now verified
to survive a call. test_forwarding_a_slice_returning_calls_result is
the free case the %rax:%rdx convention was specifically chosen for:
gen_slice_call_into already leaves a callee's own result exactly
where a caller needs to leave its own, so `return bar()` (bar also
returning a slice) costs nothing beyond the call itself -- no
intermediate copy, unlike an array-returning function's own hidden-
pointer forwarding (which avoids a copy too, but still has to thread
the SAME address one level deeper explicitly).

test_slice_parameter_with_heap_allocated_array_parameter confirms a
slice parameter's own register-based passing and an array parameter's
own copy-on-entry mechanism (heap-backed, in that test, since the
array involved exceeds the stack-array threshold) coexist correctly
in the same call, each going through its own, independent, unrelated
code path. test_slice_argument_must_be_a_variable_or_none is the same
deliberate restriction slice bases have everywhere else in this
codebase (indexing, print, re-slicing) applied here too, for the same
reason: a bare Slice expression has no pre-existing descriptor to
read at a call site.

A NOTE ON TestPrintArraysAndSlices
-----------------------------------------------------------------
`print` on an array or slice formats as `TYPE[elem, elem, ...]` --
e.g. `[3]int[1, 2, 3]` or `[]int[1, 2, 3]` -- built as a sequence of
direct printf calls, one piece at a time, rather than materializing
one big string via malloc first (see codegen.py's PRINTING ARRAYS AND
SLICES section for why: doing that would need a new int-to-string
conversion step this language has no other reason to have).

test_nested_2d_array_prefix_appears_once_not_per_row is the test that
actually proves the headline formatting decision holds, not just the
one-dimensional case the original design examples showed: the type
prefix appears exactly once, at the outermost level -- a [2][3]int
prints as `[2][3]int[[1, 2, 3], [4, 5, 6]]`, with no "[3]int" repeated
on each inner row. test_str_elements_are_quoted is the other
deliberate asymmetry worth calling out: a str element inside a
collection is quoted (`'alice'`) even though a bare str argument to
print still prints unquoted -- two different, both intentional,
conventions.

Since an array's length is known at compile time but a slice's is
only known at runtime, printing uses ONE uniform runtime loop for
both rather than maintaining two separate code paths (unrolled vs.
looped) -- test_printing_a_slice_of_a_slice exercises the harder,
runtime-length path directly. test_empty_slice_prints_with_no_
trailing_comma is the positive control for `arr[5:5]` (see
TestSliceBoundsChecking's own boundary test) actually printing
cleanly, not just type-checking. test_array_literal_as_direct_print_
argument_not_supported and its Slice-expression counterpart are the
same deliberate restriction gen_array_arg_address_into already
imposes on array-typed call arguments, applied here for the same
reason: neither has an address of its own to print through.

A NOTE ON TestNone
-----------------------
`none` is Hornet's nil-style zero value, analogous to Go's own `nil`
-- only slices are nilable so far. Internally it's given one single,
fixed type (Type.NONE, see semantic.py), checked for COMPATIBILITY
(not equality) at the handful of sites a value flows into a slice-
typed context, rather than a fully general untyped-constant mechanism
the way Go's own nil actually works -- see NoneLiteral's own docstring
in parser.py for why that's a deliberately narrower, but from-the-
outside equivalent, mechanism for what's needed right now.

test_real_empty_slice_is_not_equal_to_none and test_real_nonempty_
slice_is_not_equal_to_none together are what actually prove the
subtlest, easiest-to-get-wrong part of this feature: `s == none`
checks specifically the slice descriptor's own `ptr` field, not its
length, matching Go's own well-known nil-vs-empty-slice distinction --
`arr[5:5]` is a real, zero-length slice with a non-null pointer, and
is NOT `== none`, even though it's equally safe and equally
zero-length as a genuinely nil slice for every other purpose. Checking
length instead of (or in addition to) the pointer would have silently
conflated two states this test deliberately keeps apart.

test_indexing_a_none_valued_slice_aborts, test_printing_a_none_valued_
slice, and test_reslicing_a_none_valued_slice_at_zero_zero together
confirm the other half of the design: a none-valued slice's {0, 0}
descriptor needed no new mechanism at all for indexing, printing, or
re-slicing, since every one of those already handles an ordinary
zero-length slice correctly (see TestSliceBoundsChecking's own
`arr[5:5]` positive control) -- gen_none_into only had to produce that
descriptor once, not teach any existing slice operation a new case.

test_comparing_none_to_none_is_rejected exists for the same underlying
reason test_comparing_two_void_call_results_is_rejected does in
TestFunctionsWithNoDeclaredReturnType: `Type.NONE == Type.NONE` would
otherwise trivially type-check by ordinary structural equality alone,
so it needed its own explicit exclusion in check_binary, not just the
slice-vs-none exception. test_slice_parameter_with_none_argument_hits_
existing_restriction is the reminder that `none` doesn't need its own
codegen-level rejection for slice parameters/returns -- those aren't
supported in codegen at all yet (see TestSlices), so any program using
them hits that existing, unrelated error regardless of what's passed.

A NOTE ON TestFunctionsWithNoDeclaredReturnType
-----------------------------------------------------------------
`def NAME(params):` -- the type before the name omitted entirely,
not a `void`/`none` keyword (there is no such keyword) -- means this
function has no declared return type. Such a function may fall off
the end of its body with no explicit return at all, or exit early via
a bare `return`.

test_falls_off_the_end_with_no_explicit_return_at_all is the test
that actually proves the core mechanism holds, not just that the
syntax parses: every OTHER function relies on always_returns
guaranteeing an explicit return on some path, which is what lets
gen_function skip ever emitting its own trailing epilogue (some
gen_return-emitted one is always guaranteed to run first). A function
with no declared return type deliberately skips that guarantee, so
gen_function has to append a trailing epilogue unconditionally --
without it, this exact test would fall through into whatever comes
next in the generated assembly instead of returning to its caller, a
real, silent crash, not a hypothetical one.
test_while_loop_inside_a_void_function is the same proof for a
different shape of fall-through: reachable after a loop completes,
not just after a straight-line sequence of statements.
test_mixed_early_return_and_fall_through_paths stresses that the
trailing epilogue and gen_return's own, ordinary per-path epilogues
are both genuinely reachable in the same function, not just one or
the other.

test_comparing_two_void_call_results_is_rejected exists because
`Type.VOID == Type.VOID` is trivially true by structural equality
alone, the same way any type equals itself -- every OTHER "void used
as a value" case (a VarDecl initializer, an Assign, a binary operand,
a function argument) is already rejected for free, just by never
matching the real, user-declared type each of those checks compares
against; equality between two void results specifically needed its
own explicit rejection in check_binary, since two "nothing"s would
otherwise match each other instead. test_non_void_function_still_
requires_explicit_returns_on_every_path is the regression check
proving the always_returns skip in analyze_function is specific to
Type.VOID, not a blanket relaxation for every function.

test_print_result_not_usable_as_a_value in TestPrint is the other
half of this feature worth knowing about here: print's own docstring
always said it returned a hardcoded, meaningless 0 specifically
because there was no real void type to give it -- print became
Type.VOID's first real user the moment one existed, and that test
confirms the old workaround is gone, not still lingering alongside
the new mechanism.

A NOTE ON TestAllPathsReturn
-----------------------------------------------------------------
semantic.py now rejects any function where some execution path could
fall off the end of its body without hitting a `return` -- see
always_returns/contains_reachable_break there. This isn't just a
correctness nicety: once functions could call each other (see
codegen.py's FUNCTIONS section), a function falling through with no
`ret` executed corrupts the *calling* function's own stack, not just
the callee's exit code, since there's a real return address on the
stack with nothing left to pop and jump to it.

The genuinely subtle case, and the one the CRITICAL-labeled tests in
TestAllPathsReturn specifically target, is `while true` with a `break`
somewhere inside it. A bare `while true: ...; return x` is fine on its
own -- the loop never falls through, it either returns from inside or
runs forever -- but the instant a `break` exists anywhere in that
loop's body, even buried inside a nested if/elif chain, the loop can
fall through to whatever comes after it, so it stops counting as
guaranteeing a return on its own. Getting this exactly right (finding a
break nested arbitrarily deep in if/elif/else, while correctly *not*
letting a break that belongs to a nested loop count toward the outer
one) is most of what makes this check nontrivial rather than a simple
"does every function end in a return statement" pattern match.

A NOTE ON TestTypeAnnotation
-----------------------------------------------------------------
semantic.py's check_expr now annotates every expression node with its
resolved type (expr.resolved_type), and codegen.py's _type_of reads
that directly instead of re-deriving a type independently the way its
old _infer_type method used to. That old method wasn't just a
theoretical duplication risk -- it silently caused two real bugs, once
each when `print` and the six int-only operators (% & | ^ << >>) were
added, since each addition needed a matching update to semantic.py's
real type-checking logic *and* a separate, easy-to-forget update to
_infer_type's own parallel copy of that logic. TestTypeAnnotation
exists specifically to regression-test those two exact bug shapes
directly (a Call result and a modulo result each used straight as an
operand of `+`, with no intermediate variable), plus the new
CodegenError _type_of raises if codegen somehow runs before semantic
analysis. See semantic.py's TYPES section and codegen.py's own comments
on _type_of for the full reasoning.

A NOTE ON COMPOUND ASSIGNMENT BEING PURE DESUGARING
-----------------------------------------------------------------
+= -= *= /= %= &= |= ^= <<= >>= are parsed directly into the same AST a
hand-written `a = a + b` would already produce (see parser.py's
parse_assign and its COMPOUND ASSIGNMENT docstring section) -- not a
dedicated CompoundAssign node. That means semantic.py and codegen.py
needed zero changes for any of these ten operators; most of
TestCompoundAssignment is really confirming the desugaring round-trips
correctly through already-tested machinery, not exercising new code
paths. test_string_concat_via_plus_equals is the one genuinely new
runtime path (reaching string concatenation through `+=` rather than an
explicit `s = s + ...`), and the type-mismatch/undeclared-variable
tests confirm the desugared form still gets full checking rather than
some kind of bypass.

test_compound_assignment_in_a_loop's docstring is worth reading even
though the test itself only uses int: it connects to
TestStringMemory's existing leak tests -- `result += 'x'` in a loop is
now a much more natural, easy-to-write-by-accident way to reach the
same "named variable buffers are never automatically freed" limitation
that was already true and already documented before this feature
existed. Compound assignment doesn't introduce a new leak; it just
makes the existing one easier to hit.

A NOTE ON THE BITWISE OPERATORS' PRECEDENCE
-----------------------------------------------------------------
% & | ^ << >> follow the classic C precedence ladder (see parser.py's
_BINARY_OPS comment), adopted deliberately rather than invented fresh.
That choice reproduces a well-known C surprise on purpose: `a & b == c`
parses as `a & (b == c)`, not `(a & b) == c`, since == binds tighter
than &. In C that silently compiles into something almost nobody
intends. Here it can't -- `b == c` is bool, & requires int, so it's a
compile-time type error instead of a silent footgun.
test_bitwise_and_equality_precedence_is_a_type_error in
TestSemanticErrors is the test that actually proves this, paired with
test_bitwise_and_equality_with_explicit_parens_is_valid as the positive
control showing the fix (adding the parens) works.

Also worth knowing: modulo shares codegen with division (idivl computes
both the quotient and the remainder in one instruction), so it inherits
the exact same division-by-zero SIGFPE crash -- see
test_modulo_by_zero_crashes_with_sigfpe. And
test_modulo_result_used_as_operand_of_plus exists specifically because
adding these operators surfaced a real bug in codegen.py's _infer_type:
it needed these six new operators added to its int-producing branch, or
an expression like `5 % 2 + 3` would have misidentified `5 % 2` as
bool-typed and routed the outer `+` to string concatenation codegen
instead of ordinary integer addition.

A NOTE ON print AND WHY assert_stdout EXISTS
-----------------------------------------------------------------
`print` is the first genuinely observable I/O this language has -- every
prior feature was only ever checkable through a program's exit code
(the low byte of whatever `main` returns). print's entire purpose is a
side effect (what it writes to stdout), so testing it via exit codes
alone would be a real step down in rigor -- an exit-code check can't
tell "printed the right text" apart from "printed nothing at all, but
happened to exit 0 anyway". assert_stdout/assert_program_stdout check
the compiled program's actual captured stdout content instead (see
compile_and_run, which now passes capture_output=True for exactly this
reason), and TestPrint uses them throughout rather than falling back to
exit-code checks out of habit.

Each of print's three argument types (int, bool, str) goes through a
completely different instruction sequence in codegen.py's
gen_print_call_into -- calling libc's printf, or puts, or a runtime
branch into puts -- so TestPrint deliberately exercises all three
individually rather than assuming "int works, so the others probably do
too". print's return value (always a predictable int 0, never
whatever puts/printf themselves returned) gets its own test via the
ordinary exit-code path, since that's a case where the exit code is
actually the relevant observable, not the printed text.

A NOTE ON FUNCTION CALLS AND THE SECOND REGISTER-PRESERVATION FIX
-----------------------------------------------------------------
Function calls exposed a real bug in how `str` concatenation/comparison
were implemented: they use %rbx/%r12/%r13/%r14 as scratch, on the
reasoning (accurate at the time) that "nothing else uses them". That
stopped being true the moment one Hornet function could call another --
if function A is mid-concatenation (holding a value in %rbx) and calls
function B, and B also does string work, B would silently clobber A's
%rbx with no compiler warning and no crash, just a wrong answer. Every
function's prologue/epilogue now unconditionally saves/restores these
four registers (see codegen.py's FUNCTIONS section), regardless of
whether that particular function happens to use them itself, which is
what a callee-saved contract actually requires.

TestFunctions' register-preservation and recursive-string-concatenation
tests exist specifically to prove that fix, not just that calls work in
general -- the former nests one string-using call inside another,
the latter stress-tests the same fix under genuine recursion, where
each level's saved registers live on a distinct stack frame rather than
just one level of nesting. TestFunctions' mutual-recursion test proves
the *other* half of what functions needed: semantic.py collects every
function's signature in a first pass over the whole program before
checking any function's body, so call order doesn't matter and forward
references / recursion just work, rather than requiring functions to be
defined before they're used.

A NOTE ON str AND WHY THIS TOUCHED FAR MORE THAN THE TYPE CHECKER
-----------------------------------------------------------------
Every type added before `str` (bool) fit in the same 4 bytes as `int`,
so codegen never had to think about width -- everything was always
%eax, always `movl`. A string is an 8-byte pointer, and concatenation
needs to call real C library functions (malloc/strlen/strcpy/strcat)
via the actual SysV calling convention, which is the first time this
compiler has ever called anything external at all. See codegen.py's
STRINGS and LOCAL VARIABLES sections for the mechanism; the short
version is every local now gets a uniform 8-byte stack slot regardless
of type (simpler than variable-width packing, at the cost of a few
wasted bytes on int/bool locals), and codegen re-derives just enough
type information on its own (_infer_type) to know when a value is a
pointer rather than duplicating semantic.py's full type checker.

TestStrings' chained-concatenation and reused-result tests exist
specifically to prove the malloc/strlen/strcpy/strcat sequence can run
more than once within one expression, and that a concatenation's result
survives being used as an operand in a *later* concatenation, without
the scratch registers (%rbx/%r12/%r13/%r14) from one call clobbering
values a still-in-progress outer expression depends on.

A NOTE ON LOOPS AND WHY THE EXECUTION HELPER GAINED A TIMEOUT
-----------------------------------------------------------------
`while`/`break`/`continue` are the first feature in this language where
a codegen bug can produce a compiled program that genuinely never
terminates -- every previous feature, however buggy, still always ran
to completion (or crashed) in bounded time. compile_and_run now passes
`timeout=EXECUTION_TIMEOUT` to the actual process execution and fails
with a clear "likely an infinite loop" message on expiry, rather than
hanging the whole test run. TestWhileLoops' two nested-loop tests are
the ones this matters most for: break/continue are resolved via
codegen's loop_labels *stack* (see codegen.py's LOOPS section)
specifically so they target the innermost enclosing loop once loops
nest -- a bug there (e.g. accidentally using the outer loop's labels)
would very plausibly manifest as an infinite loop rather than a wrong
answer, which is exactly the failure mode the timeout exists to catch
cleanly instead of silently hanging.

A NOTE ON BLOCK SCOPING AND WHY CODEGEN'S ALLOCATOR CHANGED SHAPE
-----------------------------------------------------------------
`if`/`elif`/`else` introduced real nested scopes (see semantic.py's
SCOPING section), which broke an assumption codegen's local-variable
allocator used to rely on: that a variable name always maps to exactly
one stack slot per function. Once sibling branches can each declare a
variable with the same name (`if x: int a = 1` / `else: int a = 2`,
now legitimately two different variables), stack slots have to be
keyed by which specific declaration a name resolves to at a given
point in the program, not by the name alone -- see codegen.py's LOCAL
VARIABLES section for the actual mechanism. TestIfStatements' two
same-name-in-both-branches tests exist specifically to prove that
allocator change is correct, not just that if/else branch correctly.
`while` bodies reuse the same node-identity-keyed allocation, though
for a simpler reason -- see codegen.py's LOCAL VARIABLES section for
why a loop body's own variables don't need anything extra beyond that.

A NOTE ON THE TYPE SYSTEM AND WHY SEVERAL TESTS CHANGED SHAPE
-----------------------------------------------------------------
This language now has a strong static type system (see semantic.py):
`int` and `bool` are distinct types with *no* implicit conversion
between them in either direction. That's a real behavior change, not
just an addition, and it broke several tests that predate it:
  - `not 0`, `not 5`, `not -0` used to work (NOT applied to a raw int,
    back when NOT was spelled `!` -- see the note below). Under strict
    typing, `not` requires a genuine `bool` operand -- `not 0` is now a
    type error, not "not true". These are now written with real bool
    values (`not true`, `not (x == y)`, etc.) instead.
  - Every test that did `return <comparison-or-logical-expr>` from a
    `def int main()` now needs `def bool main()` instead, since
    comparisons and `and`/`or` produce `bool`, and a strict return type
    has to match exactly.
  - `0 and (1 / 0)`-style short-circuit tests used raw int operands for
    `and`/`or`, which now requires bool operands on both sides -- these
    were rewritten with `true`/`false` and a comparison
    (`(1 / 0) == 1`) standing in for "an int expression that crashes if
    evaluated, but produces a bool so `and`/`or` will accept it".

A NOTE ON '!' -> 'not' (RESOLVED)
-----------------------------------
Logical NOT used to be spelled `!` (the BANG token). The lexer has
since dropped BANG entirely -- a bare `!` is now a genuine lexer error,
not just unused -- and NOT is spelled with the `not` keyword instead
(see parser.py's own note on this). `!=` (NOT_EQUAL) was never part of
this rename and is completely unaffected; it's a separate two-character
token that never depended on BANG existing.
"""
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from codegen import CodegenError, generate_asm
from lexer import lex
from parser import Parser
from semantic import SemanticError, analyze


GCC_AVAILABLE = shutil.which("gcc") is not None
GCC_SKIP = pytest.mark.skipif(
    not GCC_AVAILABLE,
    reason="gcc not found on PATH; these tests compile and execute real binaries",
)
# Applied per-class (not module-wide): TestSemanticErrors never reaches
# codegen, let alone gcc, so it shouldn't be skipped just because gcc
# happens to be missing.

# These tests actually assemble and run the generated code, so the
# assembly's platform has to match whatever `gcc` on *this* machine will
# actually produce/link -- not be hardcoded to one platform. On macOS
# that also means explicitly targeting x86_64: this compiler only ever
# generates x86-64, but Apple Silicon Macs default to arm64, and Xcode's
# toolchain needs to be told `-arch x86_64` to assemble/link x86-64 input
# instead of rejecting it outright. The resulting binary then runs under
# Rosetta 2 (installed automatically the first time it's needed, or via
# `softwareupdate --install-rosetta`).
HOST_IS_MACOS = sys.platform == "darwin"
ASM_PLATFORM = "macos" if HOST_IS_MACOS else "linux"

# How long a compiled program gets to run before it's treated as hung.
# Every test in this file finishes in well under a second normally; a
# few seconds of headroom absorbs slow CI machines without making a
# genuinely infinite loop (e.g. from a break/continue codegen bug) wait
# long to fail.
EXECUTION_TIMEOUT = 5


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _parse(source: str):
    """lex + parse only, no semantic analysis -- used internally by both
    compile_and_run (which analyzes explicitly, see below) and
    assert_semantic_error (which asserts analysis itself fails)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "program.lang"
        src_path.write_text(source)
        tokens = lex(str(src_path))
        return Parser(tokens).parse_program()


def compile_and_run(source: str) -> subprocess.CompletedProcess:
    """Runs `source` through the real lex -> parse -> analyze -> codegen
    pipeline, assembles and links it with gcc (using ASM_PLATFORM, the
    platform this test process is actually running on -- see above),
    and runs the resulting binary, subject to EXECUTION_TIMEOUT below.

    Returns the CompletedProcess so callers can inspect `.returncode`:
    0-255 for a normal exit, or -N if the process was killed by signal N
    (that's how Python's subprocess reports signal termination when not
    going through a shell -- see assert_crashes_with_sigfpe below).

    Uses a plain tempfile.TemporaryDirectory rather than pytest's
    tmp_path fixture so the many parametrized one-liner tests below
    don't each need to declare and thread a fixture through just to
    call this helper.
    """
    ast = _parse(source)
    analyze(ast)  # every program reaching codegen in this file is expected to be well-typed

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        asm_path = tmp / "program.s"
        bin_path = tmp / "program"

        asm = generate_asm(ast, platform=ASM_PLATFORM)
        asm_path.write_text(asm)

        gcc_cmd = ["gcc"]
        if HOST_IS_MACOS:
            gcc_cmd += ["-arch", "x86_64"]
        gcc_cmd += [str(asm_path), "-o", str(bin_path)]

        result = subprocess.run(gcc_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Don't just let CalledProcessError's bare "exit status 1"
            # through -- that hides the one thing that actually explains
            # a compile failure. Show the real diagnostic, the command,
            # and the generated assembly so a failure here is
            # self-diagnosing instead of needing a follow-up round trip.
            pytest.fail(
                "gcc failed to assemble/link the generated program.\n"
                f"command: {' '.join(gcc_cmd)}\n"
                f"--- gcc stdout ---\n{result.stdout}\n"
                f"--- gcc stderr ---\n{result.stderr}\n"
                f"--- generated assembly ---\n{asm}"
            )

        try:
            # capture_output=True so callers can also inspect .stdout --
            # needed now that print exists and genuinely produces
            # observable output beyond just an exit code (see
            # assert_program_stdout below). Every prior helper here only
            # ever looked at .returncode, so capturing stdout/stderr as
            # well doesn't change anything about their behavior.
            return subprocess.run(
                [str(bin_path)], timeout=EXECUTION_TIMEOUT,
                capture_output=True, text=True,
            )
        except subprocess.TimeoutExpired:
            # Now that while loops exist, a genuine codegen bug (e.g. a
            # break/continue that jumps to the wrong label) could produce
            # a real infinite loop -- without this, that would just hang
            # the test suite forever instead of failing with a message
            # that points at what's actually wrong.
            pytest.fail(
                f"Compiled program did not exit within {EXECUTION_TIMEOUT}s "
                "-- likely an infinite loop.\n"
                f"--- generated assembly ---\n{asm}"
            )
        except OSError as e:
            if HOST_IS_MACOS:
                pytest.fail(
                    f"Failed to execute the compiled x86_64 binary ({e}). "
                    "On Apple Silicon this usually means Rosetta 2 isn't "
                    "installed -- try `softwareupdate --install-rosetta`."
                )
            raise



def assert_exit_code(body: str, expected: int, return_type: str = "int") -> None:
    """Wraps `body` (the statements, one per line, each pre-indented) in
    `def {return_type} main():` and asserts the compiled-and-executed
    program exits with `expected`."""
    source = f"def {return_type} main():\n{body}\n"
    result = compile_and_run(source)
    assert result.returncode == expected, (
        f"body:\n{body}\nexpected exit {expected}, got {result.returncode}"
    )


def assert_crashes_with_sigfpe(body: str, return_type: str = "int") -> None:
    """Same as assert_exit_code, but asserts the program is killed by
    SIGFPE (a real divide-by-zero trap) rather than exiting normally."""
    source = f"def {return_type} main():\n{body}\n"
    result = compile_and_run(source)
    assert result.returncode == -signal.SIGFPE, (
        f"body:\n{body}\nexpected SIGFPE crash, got exit {result.returncode}"
    )


def assert_crashes_with_sigabrt(body: str, return_type: str = "int") -> None:
    """Same as assert_crashes_with_sigfpe, but for SIGABRT -- what an
    out-of-bounds array access deliberately triggers (see codegen.py's
    _gen_bounds_check_panic_block), rather than a hardware-trapped
    SIGFPE."""
    source = f"def {return_type} main():\n{body}\n"
    result = compile_and_run(source)
    assert result.returncode == -signal.SIGABRT, (
        f"body:\n{body}\nexpected SIGABRT crash, got exit {result.returncode}"
    )


def assert_semantic_error(body: str, return_type: str = "int", match: str = None) -> None:
    """Asserts that `body` (wrapped in `def {return_type} main():`) is
    rejected by semantic analysis. Never reaches codegen or gcc, so
    these run regardless of GCC_AVAILABLE."""
    source = f"def {return_type} main():\n{body}\n"
    ast = _parse(source)
    with pytest.raises(SemanticError, match=match):
        analyze(ast)


def assert_program_exit_code(source: str, expected: int) -> None:
    """Like assert_exit_code, but takes a complete, ready-to-run program
    (possibly multiple functions) rather than wrapping a body in a
    single `main` -- needed for function-call tests, which by their
    nature involve more than one function definition."""
    result = compile_and_run(source)
    assert result.returncode == expected, (
        f"program:\n{source}\nexpected exit {expected}, got {result.returncode}"
    )


def assert_program_semantic_error(source: str, match: str = None) -> None:
    """The assert_semantic_error counterpart to assert_program_exit_code
    -- takes a complete program rather than wrapping a single-function
    body."""
    ast = _parse(source)
    with pytest.raises(SemanticError, match=match):
        analyze(ast)


def assert_stdout(body: str, expected_stdout: str, return_type: str = "int") -> None:
    """Like assert_exit_code, but checks the program's actual printed
    output instead of its exit code -- the only way to meaningfully
    test `print`, whose entire observable effect (from a Hornet
    program's point of view) is what it writes to stdout, not its exit
    code. `body` still needs its own `return`, exactly like every other
    body-based helper here -- this doesn't check the exit code, but
    doesn't preclude a caller checking it too if they care."""
    source = f"def {return_type} main():\n{body}\n"
    result = compile_and_run(source)
    assert result.stdout == expected_stdout, (
        f"body:\n{body}\nexpected stdout {expected_stdout!r}, got {result.stdout!r}"
    )


def assert_program_stdout(source: str, expected_stdout: str) -> None:
    """The assert_stdout counterpart to assert_program_exit_code -- takes
    a complete, ready-to-run program rather than wrapping a body in a
    single `main`."""
    result = compile_and_run(source)
    assert result.stdout == expected_stdout, (
        f"program:\n{source}\nexpected stdout {expected_stdout!r}, got {result.stdout!r}"
    )


# ---------------------------------------------------------------------------
# Unary operators: -, ~, not  (including chaining, e.g. ~-2)
# ---------------------------------------------------------------------------

class TestUnaryOperators:
    pytestmark = GCC_SKIP

    @pytest.mark.parametrize("expr,expected", [
        ("-2", 254),   # two's-complement wraparound: -2 -> 254
        ("~2", 253),   # ~2 == -3 == 253
        ("~-2", 1),    # ~(-2) == 1
        ("--2", 2),    # -(-2) == 2
    ])
    def test_int_unary_operator(self, expr, expected):
        assert_exit_code(f"    return {expr}", expected, return_type="int")

    @pytest.mark.parametrize("expr,expected", [
        ("not true", 0),
        ("not false", 1),
        ("not not true", 1),   # chained NOT, still requires (and produces) bool at each step
    ])
    def test_bool_not(self, expr, expected):
        assert_exit_code(f"    return {expr}", expected, return_type="bool")


# ---------------------------------------------------------------------------
# Binary arithmetic: + - * / , precedence, associativity, grouping
# ---------------------------------------------------------------------------

class TestBinaryArithmetic:
    pytestmark = GCC_SKIP

    @pytest.mark.parametrize("expr,expected", [
        ("1 + 2", 3),
        ("5 - 8", 253),          # -3 -> 253
        ("3 * 4", 12),
        ("20 / 4", 5),
        ("7 / 2", 3),             # integer division truncates
        ("1 + 2 * 3", 7),         # * binds tighter than +
        ("(1 + 2) * 3", 9),       # grouping overrides precedence
        ("10 - 2 - 3", 5),        # left-associative: (10-2)-3
        ("10 - (2 - 3)", 11),     # grouping overrides associativity
        ("20 / 5 / 2", 2),        # left-associative: (20/5)/2
        ("20 / (5 / 2)", 10),     # grouping overrides associativity
        ("2 + 3 * 4 - 5", 9),
        ("-2 + 3", 1),            # unary and binary minus disambiguation
        ("2 + -3", 255),          # 2 + (-3) == -1 -> 255
        ("-(2 + 3)", 251),        # -5 -> 251
        ("~2 + 1", 254),          # ~2 == -3; -3+1 == -2 -> 254
        ("~0 + ~0", 254),         # ~0 == -1; -1 + -1 == -2 -> 254
        ("2 * (3 + 4) - 5", 9),
    ])
    def test_binary_arithmetic(self, expr, expected):
        assert_exit_code(f"    return {expr}", expected, return_type="int")


# ---------------------------------------------------------------------------
# Modulo and the bitwise operators (% & | ^ << >>) -- the last of the
# operators from the README's original TODO list. Precedence follows
# the classic C ladder (see parser.py's _BINARY_OPS comment): % sits
# with * /; << >> sit between +- and the relational operators; & ^ |
# sit between == != and and/or, in that tightness order (& tightest,
# | loosest).
#
# That specific placement reproduces a well-known C surprise on
# purpose: `a & b == c` parses as `a & (b == c)`, not `(a & b) == c`,
# since == binds tighter than &. TestSemanticErrors has the positive
# proof that this language turns that into a real type error rather
# than silently accepting the "wrong" grouping the way C does.
# ---------------------------------------------------------------------------

class TestBitwiseAndModuloOperators:
    pytestmark = GCC_SKIP

    @pytest.mark.parametrize("expr,expected", [
        ("7 % 3", 1),
        ("17 % 5", 2),
        ("-7 % 3", 255),          # C-style truncating modulo: -7 % 3 == -1 -> 255
        ("12 & 10", 8),
        ("12 | 10", 14),
        ("12 ^ 10", 6),
        ("1 << 4", 16),
        ("256 >> 4", 16),
        ("-8 >> 1", 252),         # arithmetic (sign-preserving) shift: -8 >> 1 == -4 -> 252
        ("10 - 6 % 4", 8),        # % binds as tight as * / -- tighter than -
        ("1 + 1 << 2", 8),        # << is looser than + -- (1+1)<<2, not 1+(1<<2)
        ("5 & 3 | 8", 9),         # & binds tighter than |
        ("1 | 2 ^ 3 & 3", 1),     # & tightest of these three, then ^, then |
    ])
    def test_bitwise_and_modulo(self, expr, expected):
        assert_exit_code(f"    return {expr}", expected, return_type="int")

    def test_modulo_with_variables(self):
        assert_exit_code(
            "    int a = 17\n"
            "    int b = 5\n"
            "    return a % b",
            2,
        )

    def test_modulo_by_zero_crashes_with_sigfpe(self):
        """Modulo reuses idivl (see codegen.py's gen_binary_op MODULO
        case -- it's the exact same Cdq+IDiv sequence as division, just
        reading %edx instead of %eax afterward), so it inherits the
        same division-by-zero hardware trap DIVIDE already has."""
        assert_crashes_with_sigfpe("    int a = 5\n    int b = 0\n    return a % b")

    def test_modulo_result_used_as_operand_of_plus(self):
        """Specifically exercises _infer_type's handling of these new
        operators (see codegen.py) -- if MODULO weren't included in its
        int-producing branch, this expression's `5 % 2` would be
        misidentified as bool-typed, and the outer `+` would be wrongly
        routed to string concatenation codegen instead of ordinary
        integer addition."""
        assert_exit_code(
            "    int x = 5 % 2 + 3\n"
            "    return x",
            4,
        )

    def test_modulo_in_a_loop_condition(self):
        assert_exit_code(
            "    int x = 0\n"
            "    int i = 0\n"
            "    while i < 10:\n"
            "        if i % 2 == 0:\n"
            "            x = x + 1\n"
            "        i = i + 1\n"
            "    return x",
            5,
        )

    def test_bitwise_and_combined_with_logical_and(self):
        assert_exit_code(
            "    int flags = 6\n"
            "    return (flags & 2) == 2 and (flags & 1) == 0",
            1,
            return_type="bool",
        )


# ---------------------------------------------------------------------------
# Comparisons: == != < > <= >=
#
# All of these produce `bool` now, so every one of these functions is
# declared `def bool main()`, not `def int main()` -- the exit-code
# mechanics work identically either way (the OS only ever sees the
# 0/1 that ends up in %eax), but the *type* the language sees matters
# to the analyzer.
# ---------------------------------------------------------------------------

class TestComparisons:
    pytestmark = GCC_SKIP

    @pytest.mark.parametrize("expr,expected", [
        ("3 < 5", 1),
        ("5 < 3", 0),
        ("3 <= 3", 1),
        ("4 <= 3", 0),
        ("5 > 3", 1),
        ("3 > 5", 0),
        ("5 >= 5", 1),
        ("3 >= 5", 0),
        ("3 == 3", 1),
        ("3 == 4", 0),
        ("3 != 4", 1),
        ("3 != 3", 0),
    ])
    def test_comparison(self, expr, expected):
        assert_exit_code(f"    return {expr}", expected, return_type="bool")


# ---------------------------------------------------------------------------
# Precedence across every level, and 'and'/'or' chaining
# ---------------------------------------------------------------------------

class TestPrecedenceAndLogicalOperators:
    pytestmark = GCC_SKIP

    @pytest.mark.parametrize("expr,expected", [
        ("1 + 2 * 3 == 7 and 4 < 5", 1),
        ("1 + 2 * 3 == 8 and 4 < 5", 0),
        ("1 < 2 and 3 > 4 or 5 == 5", 1),
        ("1 + 1 == 2 and 3 > 2", 1),   # precedence: == and > both bind tighter than 'and'
        ("false or false or true", 1),
        ("false or false or false", 0),
        ("true and true and false", 0),
        ("true and true and true", 1),
    ])
    def test_precedence(self, expr, expected):
        assert_exit_code(f"    return {expr}", expected, return_type="bool")


# ---------------------------------------------------------------------------
# Short-circuit evaluation of 'and' / 'or'
#
# A correct boolean *result* alone doesn't prove short-circuiting
# happened, since this language has no other observable side effects
# yet. The definitive check is a division-by-zero trap on the side that
# must NOT run: if it's genuinely skipped, the program returns cleanly;
# if it's evaluated anyway, the program crashes with a real SIGFPE.
#
# 'and'/'or' require bool operands, so "an int expression that crashes
# if evaluated" has to be wrapped as one: `(1 / 0) == 1` is bool-typed
# (int == int), but still crashes at runtime the moment `1 / 0` actually
# executes.
# ---------------------------------------------------------------------------

class TestShortCircuitEvaluation:
    pytestmark = GCC_SKIP

    @pytest.mark.parametrize("expr,expected", [
        ("false and ((1 / 0) == 1)", 0),   # left is false -> right must be skipped
        ("true or ((1 / 0) == 1)", 1),     # left is true  -> right must be skipped
    ])
    def test_short_circuit_skips_crashing_side(self, expr, expected):
        assert_exit_code(f"    return {expr}", expected, return_type="bool")

    @pytest.mark.parametrize("expr", [
        "true and ((1 / 0) == 1)",   # left doesn't decide -> right MUST run
        "false or ((1 / 0) == 1)",   # left doesn't decide -> right MUST run
    ])
    def test_short_circuit_control_evaluates_when_needed(self, expr):
        """The control for the pair above: proves short-circuiting is
        genuinely conditional, not just "always skip the right side" by
        coincidence. When the left side does NOT already decide the
        result, the right side must actually execute -- so this SHOULD
        crash."""
        assert_crashes_with_sigfpe(f"    return {expr}", return_type="bool")


# ---------------------------------------------------------------------------
# Local variables: declaration, assignment, and standalone expression
# statements
# ---------------------------------------------------------------------------

class TestVariablesAndStatements:
    pytestmark = GCC_SKIP

    def test_decl_and_assign_same_line(self):
        assert_exit_code(
            "    int a = 1\n"
            "    a = a + 1\n"
            "    return a",
            2,
        )

    def test_decl_and_assign_split_across_lines(self):
        assert_exit_code(
            "    int a\n"
            "    a = 1\n"
            "    a = a + 1\n"
            "    return a",
            2,
        )

    def test_standalone_expression_statement(self):
        assert_exit_code(
            "    2 + 2\n"
            "    return 0",
            0,
        )

    def test_two_variables(self):
        assert_exit_code(
            "    int a = 3\n"
            "    int b = 4\n"
            "    return a + b",
            7,
        )

    def test_variables_in_complex_expression(self):
        assert_exit_code(
            "    int a = 2\n"
            "    int b = 3\n"
            "    return a * b + 1",
            7,
        )

    def test_reassignment_chain(self):
        assert_exit_code(
            "    int a = 1\n"
            "    a = a + 1\n"
            "    a = a + 1\n"
            "    a = a * 10\n"
            "    return a",
            30,
        )

    def test_declare_without_initializer_then_assign(self):
        assert_exit_code(
            "    int a\n"
            "    a = 5\n"
            "    return a",
            5,
        )

    def test_five_variables_frame_size(self):
        assert_exit_code(
            "    int a = 1\n"
            "    int b = 2\n"
            "    int c = 3\n"
            "    int d = 4\n"
            "    int e = 5\n"
            "    return a + b + c + d + e",
            15,
        )

    def test_variable_in_comparison(self):
        assert_exit_code(
            "    int a = 5\n"
            "    int b = 3\n"
            "    return a > b",
            1,
            return_type="bool",
        )

    def test_variable_in_short_circuit(self):
        assert_exit_code(
            "    bool a = false\n"
            "    return a and ((1 / 0) == 1)",
            0,
            return_type="bool",
        )

    def test_standalone_expression_statement_actually_executes(self):
        """Proof that expression statements aren't silently dropped:
        a division by zero as a bare statement, with its result
        discarded, must still crash the program."""
        assert_crashes_with_sigfpe(
            "    1 / 0\n"
            "    return 0"
        )

    def test_standalone_safe_expression_does_not_crash(self):
        assert_exit_code(
            "    1 + 1\n"
            "    return 42",
            42,
        )


# ---------------------------------------------------------------------------
# Compound assignment (+= -= *= /= %= &= |= ^= <<= >>=).
#
# These are parsed as pure syntactic sugar -- parser.py's parse_assign
# desugars `a += b` directly into the same AST a hand-written `a = a +
# b` would produce (Assign wrapping a Binary), so semantic.py and
# codegen.py needed zero changes to support any of these ten operators.
# That means most of these tests are really testing the desugaring
# itself and confirming nothing was lost by reusing existing machinery,
# not exercising new codegen. test_string_concat_via_plus_equals is the
# one genuinely new runtime path (compound assignment reaching the
# string-concatenation/malloc machinery through the new syntax), and
# test_compound_assignment_type_mismatch_is_rejected +
# test_compound_assignment_to_undeclared_variable_is_rejected confirm
# the desugared form still gets full type- and scope-checking, not a
# bypass around it.
# ---------------------------------------------------------------------------

class TestCompoundAssignment:
    pytestmark = GCC_SKIP

    @pytest.mark.parametrize("body,expected", [
        ("    int x = 5\n    x += 3\n    return x", 8),
        ("    int x = 5\n    x -= 3\n    return x", 2),
        ("    int x = 5\n    x *= 3\n    return x", 15),
        ("    int x = 20\n    x /= 4\n    return x", 5),
        ("    int x = 17\n    x %= 5\n    return x", 2),
        ("    int x = 12\n    x &= 10\n    return x", 8),
        ("    int x = 12\n    x |= 10\n    return x", 14),
        ("    int x = 12\n    x ^= 10\n    return x", 6),
        ("    int x = 1\n    x <<= 4\n    return x", 16),
        ("    int x = 256\n    x >>= 4\n    return x", 16),
    ])
    def test_each_compound_operator(self, body, expected):
        assert_exit_code(body, expected)

    def test_chained_compound_assignments(self):
        """Every operator applied in sequence to the same variable --
        proof the desugared reads/writes compose correctly across
        multiple statements, not just in isolation."""
        assert_exit_code(
            "    int x = 5\n"
            "    x += 3\n"   # 8
            "    x -= 1\n"   # 7
            "    x *= 2\n"   # 14
            "    x /= 2\n"   # 7
            "    x %= 5\n"   # 2
            "    x &= 3\n"   # 2
            "    x |= 4\n"   # 6
            "    x ^= 1\n"   # 7
            "    x <<= 2\n"  # 28
            "    x >>= 1\n"  # 14
            "    return x",
            14,
        )

    def test_string_concat_via_plus_equals(self):
        """The one genuinely new runtime path here: compound assignment
        reaching string concatenation's malloc/strlen/strcpy/strcat
        codegen through the new `+=` syntax rather than an explicit
        `s = s + ...`."""
        assert_exit_code(
            "    str s = 'hello'\n"
            "    s += ' world'\n"
            "    return s == 'hello world'",
            1,
            return_type="bool",
        )

    def test_compound_assignment_in_a_loop(self):
        """The natural, idiomatic use case for compound assignment --
        an accumulator. Also the case worth knowing accumulates
        garbage: each `total += i` here still only ever costs a few
        bytes of stack, but the equivalent `result += 'x'` pattern for
        str would leak one buffer per iteration, for the exact same
        underlying reason `test_concatenation_with_fresh_intermediate_
        inside_a_loop` in TestStringMemory already does -- `result` is
        a named variable, not a fresh Binary(ADD, ...) result, so the
        memory-freeing optimization correctly (if unfortunately) leaves
        it alone every time. Compound assignment doesn't introduce a
        new leak here; it just makes the existing one much easier to
        write by accident.
        """
        assert_exit_code(
            "    int total = 0\n"
            "    int i = 1\n"
            "    while i <= 5:\n"
            "        total += i\n"
            "        i += 1\n"
            "    return total",
            15,
        )

    def test_compound_assignment_type_mismatch_is_rejected(self):
        """Confirms the desugared form still gets full type-checking --
        `b += 1` desugars to `b = b + 1`, and `+` on bool and int is
        exactly as invalid as it would be written out longhand."""
        assert_semantic_error(
            "    bool b = true\n"
            "    b += 1\n"
            "    return 0",
            match="requires two int operands or two str operands",
        )

    def test_compound_assignment_to_undeclared_variable_is_rejected(self):
        assert_semantic_error(
            "    undeclared_var += 1\n"
            "    return 0",
            match="undeclared variable",
        )

    def test_modulo_assign_by_zero_still_crashes_with_sigfpe(self):
        """%= reuses ordinary modulo codegen via desugaring, so it
        inherits the same division-by-zero hardware trap DIVIDE and
        MODULO already have."""
        assert_crashes_with_sigfpe(
            "    int a = 5\n"
            "    int zero = 0\n"
            "    a %= zero\n"
            "    return a"
        )


# ---------------------------------------------------------------------------
# if / elif / else: branching, elif chains, nesting, and block scoping.
#
# The scoping tests here matter more than they might look -- they're not
# just "does the value come out right", they're proof that codegen's
# stack-slot allocator correctly gives two *different* variables their
# own storage even when they share a name across sibling branches (see
# codegen.py's LOCAL VARIABLES section). Before that allocator was
# rewritten, a program declaring `a` in both an if and its else would
# have been rejected as a duplicate declaration at the codegen layer,
# even though semantic.py correctly allows it.
# ---------------------------------------------------------------------------

class TestIfStatements:
    pytestmark = GCC_SKIP

    def test_if_true_branch_taken(self):
        assert_exit_code(
            "    int a = 1\n"
            "    if a == 1:\n"
            "        return true\n"
            "    else:\n"
            "        return false",
            1,
            return_type="bool",
        )

    def test_if_false_branch_taken(self):
        assert_exit_code(
            "    int a = 2\n"
            "    if a == 1:\n"
            "        return true\n"
            "    else:\n"
            "        return false",
            0,
            return_type="bool",
        )

    def test_if_no_else_condition_false_falls_through(self):
        assert_exit_code(
            "    int a = 0\n"
            "    if a == 1:\n"
            "        a = 99\n"
            "    return a",
            0,
        )

    def test_if_no_else_condition_true_body_runs(self):
        assert_exit_code(
            "    int a = 1\n"
            "    if a == 1:\n"
            "        a = 99\n"
            "    return a",
            99,
        )

    @pytest.mark.parametrize("a,expected", [
        (1, 10),   # first branch matches
        (5, 20),   # second branch matches
        (9, 30),   # third branch matches
        (100, 40),  # falls through to else
    ])
    def test_elif_chain(self, a, expected):
        assert_exit_code(
            f"    int a = {a}\n"
            "    if a == 1:\n"
            "        return 10\n"
            "    elif a == 5:\n"
            "        return 20\n"
            "    elif a == 9:\n"
            "        return 30\n"
            "    else:\n"
            "        return 40",
            expected,
        )

    def test_nested_if_in_if_inner_false(self):
        assert_exit_code(
            "    if true:\n"
            "        if false:\n"
            "            return 1\n"
            "        return 2\n"
            "    return 3",
            2,
        )

    def test_nested_if_in_if_outer_false(self):
        assert_exit_code(
            "    if false:\n"
            "        if true:\n"
            "            return 1\n"
            "        return 2\n"
            "    return 3",
            3,
        )

    def test_same_name_in_both_branches_then(self):
        """The key scoping case: `a` in the then-branch and `a` in the
        else-branch are independent variables that happen to share a
        name -- semantic.py allows this (separate scopes), and codegen
        must give them genuinely separate stack slots for this to come
        out right."""
        assert_exit_code(
            "    if true:\n"
            "        int a = 1\n"
            "        return a\n"
            "    else:\n"
            "        int a = 2\n"
            "        return a",
            1,
        )

    def test_same_name_in_both_branches_else(self):
        assert_exit_code(
            "    if false:\n"
            "        int a = 1\n"
            "        return a\n"
            "    else:\n"
            "        int a = 2\n"
            "        return a",
            2,
        )

    def test_shadowing_outer_variable_returns_inner_value(self):
        assert_exit_code(
            "    int a = 100\n"
            "    if true:\n"
            "        int a = 5\n"
            "        return a\n"
            "    return a",
            5,
        )

    def test_outer_variable_unaffected_after_shadowing_block_ends(self):
        assert_exit_code(
            "    int a = 100\n"
            "    if true:\n"
            "        int a = 5\n"
            "    return a",
            100,
        )

    def test_assignment_to_outer_variable_from_inside_if(self):
        assert_exit_code(
            "    int a = 1\n"
            "    if true:\n"
            "        a = 2\n"
            "    return a",
            2,
        )

    def test_early_return_from_then_branch(self):
        assert_exit_code(
            "    if true:\n"
            "        return 1\n"
            "    return 2",
            1,
        )

    def test_if_condition_using_and_and_comparisons(self):
        assert_exit_code(
            "    int a = 5\n"
            "    int b = 3\n"
            "    if a > b and b > 0:\n"
            "        return 1\n"
            "    return 0",
            1,
        )

    def test_two_separate_if_blocks_each_declare_their_own_variable(self):
        """Two *non-overlapping* if-blocks (not sibling branches of the
        same if) each declaring a variable called `y` -- distinct from
        the sibling-branch case above, but exercising the same
        node-identity-keyed allocation."""
        assert_exit_code(
            "    int x = 1\n"
            "    if true:\n"
            "        int y = 2\n"
            "        x = x + y\n"
            "    if true:\n"
            "        int y = 10\n"
            "        x = x + y\n"
            "    return x",
            13,
        )


# ---------------------------------------------------------------------------
# while / break / continue.
#
# The nested-loop tests here matter more than they might look, same as
# the sibling-branch tests in TestIfStatements matter more than they
# look -- they're not just "does break/continue work", they're proof
# that codegen's loop_labels stack (see codegen.py's LOOPS section)
# correctly resolves break/continue to the *innermost* enclosing loop
# rather than some outer one, which a naive single-pair implementation
# (rather than a stack) would get wrong the moment loops nest.
# ---------------------------------------------------------------------------

class TestWhileLoops:
    pytestmark = GCC_SKIP

    def test_counts_to_five(self):
        assert_exit_code(
            "    int i = 0\n"
            "    while i < 5:\n"
            "        i = i + 1\n"
            "    return i",
            5,
        )

    def test_condition_false_immediately_zero_iterations(self):
        assert_exit_code(
            "    int i = 10\n"
            "    while i < 5:\n"
            "        i = i + 1\n"
            "    return i",
            10,
        )

    def test_break_exits_immediately(self):
        assert_exit_code(
            "    int i = 0\n"
            "    while true:\n"
            "        i = i + 1\n"
            "        if i == 3:\n"
            "            break\n"
            "    return i",
            3,
        )

    def test_continue_skips_specific_iterations(self):
        """Sums 1..5 but skips adding when i is 2 or 4, via continue --
        1 + 3 + 5 = 9. Proves continue skips only the rest of *that*
        iteration's body (the `sum = sum + i` line), not the increment
        that already happened above it, and not the loop entirely."""
        assert_exit_code(
            "    int i = 0\n"
            "    int sum = 0\n"
            "    while i < 5:\n"
            "        i = i + 1\n"
            "        if i == 2 or i == 4:\n"
            "            continue\n"
            "        sum = sum + i\n"
            "    return sum",
            9,
        )

    def test_nested_loops_break_only_exits_innermost(self):
        """Inner loop always breaks on its second check (j==1), so it
        contributes exactly one `count = count + 1` per outer iteration
        -- if break incorrectly exited *both* loops, count would only
        ever reach 1, not 3."""
        assert_exit_code(
            "    int count = 0\n"
            "    int i = 0\n"
            "    while i < 3:\n"
            "        int j = 0\n"
            "        while j < 3:\n"
            "            if j == 1:\n"
            "                break\n"
            "            count = count + 1\n"
            "            j = j + 1\n"
            "        i = i + 1\n"
            "    return count",
            3,
        )

    def test_nested_loops_continue_only_affects_innermost(self):
        """Inner loop runs 3 times per outer iteration, skipping one via
        continue, so 2 increments per outer iteration -- 3 outer
        iterations x 2 = 6. If continue incorrectly targeted the outer
        loop's condition instead, this would come out very differently
        (and likely loop far more than 3 outer times)."""
        assert_exit_code(
            "    int total = 0\n"
            "    int i = 0\n"
            "    while i < 3:\n"
            "        int j = 0\n"
            "        while j < 3:\n"
            "            j = j + 1\n"
            "            if j == 2:\n"
            "                continue\n"
            "            total = total + 1\n"
            "        i = i + 1\n"
            "    return total",
            6,
        )

    def test_variable_declared_inside_loop_body_reused_each_iteration(self):
        assert_exit_code(
            "    int total = 0\n"
            "    int i = 0\n"
            "    while i < 4:\n"
            "        int doubled = i * 2\n"
            "        total = total + doubled\n"
            "        i = i + 1\n"
            "    return total",
            12,  # 0 + 2 + 4 + 6
        )

    def test_while_condition_using_and(self):
        assert_exit_code(
            "    int i = 0\n"
            "    int j = 10\n"
            "    while i < 5 and j > 0:\n"
            "        i = i + 1\n"
            "        j = j - 1\n"
            "    return i",
            5,
        )

    def test_early_return_from_inside_while(self):
        assert_exit_code(
            "    int i = 0\n"
            "    while true:\n"
            "        i = i + 1\n"
            "        if i == 7:\n"
            "            return i\n"
            "    return 0",
            7,
        )

    def test_if_followed_by_while_mixed_control_flow(self):
        assert_exit_code(
            "    int a = 5\n"
            "    if a > 0:\n"
            "        a = a + 1\n"
            "    int i = 0\n"
            "    while i < a:\n"
            "        i = i + 1\n"
            "    return i",
            6,
        )


# ---------------------------------------------------------------------------
# str: literals, equality/inequality, and concatenation.
#
# Every test here that touches equality or concatenation is a real,
# end-to-end proof of the runtime mechanism, not just a type-checking
# formality: concatenation genuinely calls malloc/strlen/strcpy/strcat
# via the SysV ABI (see codegen.py's STRINGS section), and equality
# genuinely calls strcmp -- there's no shortcut where the compiler
# "knows" two literals are equal at compile time and folds the
# comparison away. The chained-concatenation and reused-result tests in
# particular are what prove multiple concatenations in sequence don't
# corrupt each other's scratch registers.
# ---------------------------------------------------------------------------

class TestStrings:
    pytestmark = GCC_SKIP

    def test_equal_string_literals_compare_equal(self):
        assert_exit_code(
            "    str a = 'hello'\n"
            "    str b = 'hello'\n"
            "    return a == b",
            1,
            return_type="bool",
        )

    def test_different_string_literals_compare_unequal(self):
        assert_exit_code(
            "    str a = 'hello'\n"
            "    str b = 'world'\n"
            "    return a == b",
            0,
            return_type="bool",
        )

    def test_not_equal_on_different_strings(self):
        assert_exit_code(
            "    str a = 'hello'\n"
            "    str b = 'world'\n"
            "    return a != b",
            1,
            return_type="bool",
        )

    def test_basic_concatenation(self):
        assert_exit_code(
            "    str a = 'foo'\n"
            "    str b = 'bar'\n"
            "    return (a + b) == 'foobar'",
            1,
            return_type="bool",
        )

    def test_concatenation_of_two_literals_directly(self):
        assert_exit_code(
            "    return ('foo' + 'bar') == 'foobar'",
            1,
            return_type="bool",
        )

    def test_chained_concatenation_of_three_strings(self):
        """Proves the malloc/strlen/strcpy/strcat sequence in
        gen_string_concat_into can run more than once in a row within
        one expression without the second call clobbering scratch state
        the first call's result still depends on."""
        assert_exit_code(
            "    str a = 'a'\n"
            "    str b = 'b'\n"
            "    str c = 'c'\n"
            "    return (a + b + c) == 'abc'",
            1,
            return_type="bool",
        )

    def test_concatenation_result_reused_in_later_concatenation(self):
        assert_exit_code(
            "    str a = 'hello'\n"
            "    str greeting = a + ', world'\n"
            "    str full = greeting + '!'\n"
            "    return full == 'hello, world!'",
            1,
            return_type="bool",
        )

    def test_escape_sequence_matches_manual_construction(self):
        assert_exit_code(
            "    str a = 'line1\\nline2'\n"
            "    str b = 'line1' + '\\n' + 'line2'\n"
            "    return a == b",
            1,
            return_type="bool",
        )

    def test_escaped_quote_in_string_literal(self):
        assert_exit_code(
            "    str a = 'it\\'s here'\n"
            "    return a == 'it\\'s here'",
            1,
            return_type="bool",
        )

    def test_string_equality_as_if_condition(self):
        assert_exit_code(
            "    str a = 'hello'\n"
            "    if a == 'hello':\n"
            "        return 1\n"
            "    return 0",
            1,
        )

    def test_string_equality_driving_a_while_loop(self):
        assert_exit_code(
            "    str target = 'stop'\n"
            "    str current = 'go'\n"
            "    int count = 0\n"
            "    while current != target:\n"
            "        count = count + 1\n"
            "        if count == 3:\n"
            "            current = 'stop'\n"
            "        else:\n"
            "            current = current + 'x'\n"
            "    return count",
            3,
        )

    def test_reassigning_a_str_variable(self):
        assert_exit_code(
            "    str a = 'first'\n"
            "    a = 'second'\n"
            "    return a == 'second'",
            1,
            return_type="bool",
        )

    def test_str_int_bool_locals_coexisting(self):
        """Exercises the uniform 8-byte stack-slot allocation (see
        codegen.py's LOCAL VARIABLES section) with all three types
        present in the same frame at once."""
        assert_exit_code(
            "    int x = 5\n"
            "    str s = 'test'\n"
            "    bool b = true\n"
            "    if s == 'test' and b and x == 5:\n"
            "        return 42\n"
            "    return 0",
            42,
        )


# ---------------------------------------------------------------------------
# String memory management: freeing an intermediate concatenation result
# the moment it's no longer needed (see codegen.py's
# _gen_free_if_fresh_concat and its STRINGS section).
#
# None of these tests can observe the freeing itself from inside the
# Hornet language -- there's no way to inspect the heap from here, so
# what they actually verify is that the optimization *doesn't break
# anything*: every one of them would still produce the exact same
# result if this feature didn't exist at all. The CRITICAL-labeled
# tests are the ones that would actually catch a mistake in this
# feature specifically -- reusing a named variable, a literal, or a
# function call's return value after it was incorrectly freed would
# show up here as wrong output or a crash, not just a leak.
#
# The freeing itself -- that it actually happens, targets the right
# pointer, and never double-frees or frees a non-heap address -- was
# verified directly with an LD_PRELOAD malloc/free tracer during
# development (not part of this suite, since it needs a compiled .so
# shim well outside what a portable pytest file should depend on): a
# 3-way chain showed exactly 1 free (the intermediate result, not the
# final one); a 6-way chain (5 concatenations) showed exactly 4 frees
# with 1 correctly left un-freed; reusing a named variable or two
# literals showed zero frees; and zero invalid frees appeared in any
# case tested.
# ---------------------------------------------------------------------------

class TestStringMemory:
    pytestmark = GCC_SKIP

    def test_basic_concat_no_fresh_operands(self):
        assert_exit_code(
            "    str a = 'foo'\n"
            "    str b = 'bar'\n"
            "    str c = a + b\n"
            "    return c == 'foobar'",
            1,
            return_type="bool",
        )

    def test_three_way_chain_intermediate_result_freed(self):
        assert_exit_code(
            "    str a = 'x'\n"
            "    str b = 'y'\n"
            "    str c = 'z'\n"
            "    str r = a + b + c\n"
            "    return r == 'xyz'",
            1,
            return_type="bool",
        )

    def test_deep_six_way_chain(self):
        assert_exit_code(
            "    str a = '1'\n"
            "    str b = '2'\n"
            "    str c = '3'\n"
            "    str d = '4'\n"
            "    str e = '5'\n"
            "    str f = '6'\n"
            "    str r = a + b + c + d + e + f\n"
            "    return r == '123456'",
            1,
            return_type="bool",
        )

    def test_critical_named_variable_reused_across_two_concats(self):
        """A named variable used as an operand in one concatenation must
        still be fully intact and usable in a *second*, later
        concatenation -- if the freeing check incorrectly matched
        Variable nodes instead of only Binary(ADD, ...) nodes, this
        would read freed memory the second time and almost certainly
        produce garbage or crash."""
        assert_exit_code(
            "    str a = 'shared'\n"
            "    str b = a + '_first'\n"
            "    str c = a + '_second'\n"
            "    return b == 'shared_first' and c == 'shared_second'",
            1,
            return_type="bool",
        )

    def test_critical_named_variable_reused_four_times(self):
        assert_exit_code(
            "    str base = 'X'\n"
            "    str r1 = base + '1'\n"
            "    str r2 = base + '2'\n"
            "    str r3 = base + '3'\n"
            "    str r4 = base + '4'\n"
            "    return r1 == 'X1' and r2 == 'X2' and r3 == 'X3' and r4 == 'X4'",
            1,
            return_type="bool",
        )

    def test_critical_two_literal_operands_never_freed(self):
        """Both operands here point into static `.data`, never the
        heap -- calling free() on either would be undefined behavior
        (most likely heap corruption or an immediate crash), so this is
        the test that would catch the freeing check failing to exclude
        StringLiteral operands."""
        assert_exit_code(
            "    str r = 'lit1' + 'lit2'\n"
            "    return r == 'lit1lit2'",
            1,
            return_type="bool",
        )

    def test_critical_function_call_results_never_freed(self):
        """A function's return value might be a static literal, a fresh
        heap buffer, or a parameter passed straight through -- codegen
        has no visibility into which, so Call results are always
        excluded from freeing. Using both results again afterward (via
        the equality check) would surface a use-after-free if this
        exclusion were missing."""
        assert_program_exit_code(
            "def str make_a():\n"
            "    return 'aaa'\n"
            "\n"
            "def str make_b():\n"
            "    return 'bbb'\n"
            "\n"
            "def bool main():\n"
            "    str r = make_a() + make_b()\n"
            "    return r == 'aaabbb'\n",
            1,
        )

    def test_fresh_concat_compared_directly(self):
        """A fresh concatenation result used immediately as an operand
        of == also gets freed (in gen_string_compare_into, after
        strcmp) -- this is the case that specifically exercises the
        stash-before-free/restore-after-free dance needed there, since
        `call free` clobbers %eax exactly where strcmp's own result
        briefly lives."""
        assert_exit_code(
            "    str a = 'foo'\n"
            "    str b = 'bar'\n"
            "    return (a + b) == 'foobar'",
            1,
            return_type="bool",
        )

    def test_fresh_concat_on_both_sides_of_comparison(self):
        assert_exit_code(
            "    str a = 'x'\n"
            "    str b = 'y'\n"
            "    return (a + b) == ('x' + 'y')",
            1,
            return_type="bool",
        )

    def test_one_fresh_one_named_operand_in_same_concat(self):
        assert_exit_code(
            "    str a = 'p'\n"
            "    str b = 'q'\n"
            "    str c = 'r'\n"
            "    str result = (a + b) + c\n"
            "    return result == 'pqr'",
            1,
            return_type="bool",
        )

    def test_critical_concat_result_stored_in_variable_reused_twice(self):
        """The subtlest case: `combined`'s value originally came from a
        concatenation, but once it's stored in a named variable, later
        *references* to it are Variable nodes, not Binary nodes -- the
        freeing check has to look at the AST shape at each use site,
        not "was this value ever produced by a concatenation
        somewhere". Verified directly with the malloc/free tracer
        during development: this program allocates 3 buffers and frees
        none of them, confirming `combined` is correctly never freed on
        either reuse."""
        assert_exit_code(
            "    str a = 'hello'\n"
            "    str b = 'world'\n"
            "    str combined = a + b\n"
            "    str r1 = combined + '!'\n"
            "    str r2 = combined + '?'\n"
            "    return r1 == 'helloworld!' and r2 == 'helloworld?'",
            1,
            return_type="bool",
        )

    def test_concatenation_with_fresh_intermediate_inside_a_loop(self):
        assert_exit_code(
            "    str result = ''\n"
            "    int i = 0\n"
            "    while i < 5:\n"
            "        result = result + 'a' + 'b'\n"
            "        i = i + 1\n"
            "    return result == 'ababababab'",
            1,
            return_type="bool",
        )


# ---------------------------------------------------------------------------
# Function calls: parameters, arguments, recursion, and the two distinct
# register-preservation fixes that make string operations safe across
# both nested expressions and nested calls (see codegen.py's STRINGS and
# FUNCTIONS docstring sections). The mutual-recursion and register-
# preservation tests here are the ones that actually prove something
# subtle is correct, not just that a call compiles and runs.
# ---------------------------------------------------------------------------

class TestFunctions:
    pytestmark = GCC_SKIP

    def test_no_arg_function_call(self):
        assert_program_exit_code(
            "def int five():\n"
            "    return 5\n"
            "\n"
            "def int main():\n"
            "    return five()\n",
            5,
        )

    def test_two_arg_function_call(self):
        assert_program_exit_code(
            "def int add(int a, int b):\n"
            "    return a + b\n"
            "\n"
            "def int main():\n"
            "    return add(2, 3)\n",
            5,
        )

    def test_nested_calls(self):
        assert_program_exit_code(
            "def int inc(int x):\n"
            "    return x + 1\n"
            "\n"
            "def int main():\n"
            "    return inc(inc(5))\n",
            7,
        )

    def test_recursive_factorial(self):
        assert_program_exit_code(
            "def int fact(int n):\n"
            "    if n == 0:\n"
            "        return 1\n"
            "    return n * fact(n - 1)\n"
            "\n"
            "def int main():\n"
            "    return fact(5)\n",
            120,
        )

    def test_recursive_fibonacci(self):
        assert_program_exit_code(
            "def int fib(int n):\n"
            "    if n < 2:\n"
            "        return n\n"
            "    return fib(n - 1) + fib(n - 2)\n"
            "\n"
            "def int main():\n"
            "    return fib(10)\n",
            55,
        )

    def test_mutual_recursion_with_forward_reference(self):
        """is_even is defined *before* is_odd but calls it -- proves
        semantic.py's two-pass signature collection (and codegen's own
        function_return_types pre-scan) make call order not matter."""
        assert_program_exit_code(
            "def bool is_even(int n):\n"
            "    if n == 0:\n"
            "        return true\n"
            "    return is_odd(n - 1)\n"
            "\n"
            "def bool is_odd(int n):\n"
            "    if n == 0:\n"
            "        return false\n"
            "    return is_even(n - 1)\n"
            "\n"
            "def int main():\n"
            "    if is_even(10):\n"
            "        return 1\n"
            "    return 0\n",
            1,
        )

    def test_call_result_used_as_argument_to_another_call(self):
        assert_program_exit_code(
            "def int add(int a, int b):\n"
            "    return a + b\n"
            "\n"
            "def int main():\n"
            "    return add(add(1, 2), add(3, 4))\n",
            10,
        )

    def test_str_parameter_and_str_return_type(self):
        assert_program_exit_code(
            "def str greet(str name):\n"
            "    return 'hello, ' + name\n"
            "\n"
            "def bool main():\n"
            "    str result = greet('world')\n"
            "    return result == 'hello, world'\n",
            1,
        )

    def test_function_call_as_bare_statement(self):
        """A call's result can be discarded entirely, via the ordinary
        expr_stmt grammar rule -- no separate "call statement" concept
        needed (see parser.py's Call docstring)."""
        assert_program_exit_code(
            "def int side_effect():\n"
            "    return 99\n"
            "\n"
            "def int main():\n"
            "    side_effect()\n"
            "    return 42\n",
            42,
        )

    def test_register_preservation_across_nested_string_using_call(self):
        """The critical test for codegen.py's FUNCTIONS section: `outer`
        is mid-concatenation (holding a value that needs to survive)
        when it calls `inner_concat`, which does its *own* string
        concatenation internally. If the callee-saved-register fix in
        every function's prologue/epilogue weren't there, this would
        silently compute a wrong answer rather than fail loudly."""
        assert_program_exit_code(
            "def str inner_concat(str a, str b):\n"
            "    return a + b\n"
            "\n"
            "def bool outer(str x, str y, str z):\n"
            "    str first = x + y\n"
            "    str second = inner_concat(y, z)\n"
            "    return (first + second) == (x + y + y + z)\n"
            "\n"
            "def bool main():\n"
            "    return outer('a', 'b', 'c')\n",
            1,
        )

    def test_recursive_string_concatenation(self):
        """A more aggressive version of the register-preservation test:
        `repeat` calls *itself*, each level doing its own concatenation,
        stress-testing that the callee-saved fix holds up under actual
        recursion (each level's saved registers living on a genuinely
        different stack frame), not just a single level of nesting."""
        assert_program_exit_code(
            "def str repeat(str s, int n):\n"
            "    if n == 0:\n"
            "        return ''\n"
            "    return s + repeat(s, n - 1)\n"
            "\n"
            "def bool main():\n"
            "    str result = repeat('x', 4)\n"
            "    return result == 'xxxx'\n",
            1,
        )

    def test_more_than_six_parameters_is_a_clean_codegen_error(self):
        """Only up to 6 parameters/arguments are supported (register-
        passed per the SysV ABI; stack-passed ones aren't implemented)
        -- a 7th should fail loudly and clearly, not silently miscompile
        or crash at runtime."""
        source = (
            "def int seven(int a, int b, int c, int d, int e, int f, int g):\n"
            "    return a\n"
            "\n"
            "def int main():\n"
            "    return seven(1, 2, 3, 4, 5, 6, 7)\n"
        )
        ast = _parse(source)
        analyze(ast)  # semantically fine -- the limit is a codegen-level one
        with pytest.raises(CodegenError, match="only supports up to 6"):
            generate_asm(ast, platform=ASM_PLATFORM)


# ---------------------------------------------------------------------------
# Functions with no declared return type: `def NAME(params):`, the type
# before the name omitted entirely (Function.return_type=None) rather
# than a `void`/`none` keyword -- there is no such keyword. Such a
# function may fall off the end of its body without an explicit return
# at all, or exit early via a bare `return` (Return.value=None) -- see
# their own docstrings in parser.py. Internally, semantic.py gives this
# a real (if purely internal, never user-writable) Type.VOID rather than
# reusing Python's own None for it, specifically to keep it distinct
# from resolved_type's OWN None, which already means "not yet type-
# checked" everywhere else -- conflating the two would make a
# legitimately void expression indistinguishable from one semantic
# analysis simply hadn't reached yet.
#
# test_falls_off_the_end_with_no_explicit_return_at_all is the test
# that actually proves the core mechanism holds, not just that the
# feature parses: every OTHER function relies on always_returns
# guaranteeing an explicit return on some path, which is what lets
# gen_function skip ever emitting its own trailing epilogue (some
# gen_return-emitted one is always guaranteed to run first). A function
# with no declared return type deliberately skips that guarantee, so
# gen_function has to append a trailing epilogue unconditionally --
# without it, this exact test would fall through into whatever comes
# next in the generated assembly (the bounds-check panic block, or the
# next function's own prologue) instead of returning to its caller, a
# real, silent crash, not a hypothetical one.
#
# print itself is Type.VOID now -- it was always documented as
# returning a hardcoded, meaningless 0 specifically as a workaround for
# there being no real void type at all; test_print_result_not_usable_
# as_a_value in TestPrint is the test confirming that workaround is
# gone.
# ---------------------------------------------------------------------------

class TestFunctionsWithNoDeclaredReturnType:
    pytestmark = GCC_SKIP

    def test_falls_off_the_end_with_no_explicit_return_at_all(self):
        assert_program_stdout(
            "def log(str msg):\n"
            "    print(msg)\n"
            "\n"
            "def int main():\n"
            "    log('hello')\n"
            "    return 0\n",
            "hello\n",
        )

    def test_bare_return_exits_early(self):
        assert_program_stdout(
            "def log(int x):\n"
            "    if x < 0:\n"
            "        return\n"
            "    print(x)\n"
            "\n"
            "def int main():\n"
            "    log(-5)\n"
            "    log(42)\n"
            "    return 0\n",
            "42\n",
        )

    def test_no_parameters_and_no_return_type(self):
        assert_program_stdout(
            "def greet():\n"
            "    print('hi')\n"
            "\n"
            "def int main():\n"
            "    greet()\n"
            "    return 0\n",
            "hi\n",
        )

    def test_mixed_early_return_and_fall_through_paths(self):
        """Several if-guarded early returns followed by a final fall-
        through case, all in the same function -- stresses that the
        trailing epilogue gen_function appends is genuinely reachable
        (the fall-through case) alongside gen_return's own, ordinary
        per-path epilogues (the early-return cases), not just one or
        the other."""
        assert_program_stdout(
            "def classify(int x):\n"
            "    if x < 0:\n"
            "        print('negative')\n"
            "        return\n"
            "    if x == 0:\n"
            "        print('zero')\n"
            "        return\n"
            "    print('positive')\n"
            "\n"
            "def int main():\n"
            "    classify(-1)\n"
            "    classify(0)\n"
            "    classify(5)\n"
            "    return 0\n",
            "negative\nzero\npositive\n",
        )

    def test_void_function_calling_another_void_function(self):
        assert_program_stdout(
            "def inner():\n"
            "    print('inner')\n"
            "\n"
            "def outer():\n"
            "    print('outer')\n"
            "    inner()\n"
            "\n"
            "def int main():\n"
            "    outer()\n"
            "    return 0\n",
            "outer\ninner\n",
        )

    def test_recursive_void_function(self):
        assert_program_stdout(
            "def countdown(int n):\n"
            "    if n <= 0:\n"
            "        return\n"
            "    print(n)\n"
            "    countdown(n - 1)\n"
            "\n"
            "def int main():\n"
            "    countdown(3)\n"
            "    return 0\n",
            "3\n2\n1\n",
        )

    def test_while_loop_inside_a_void_function(self):
        """A different shape of "falls off the end" than the plain,
        straight-line case above: the trailing epilogue has to be
        reachable AFTER a loop completes, not just after a sequence of
        ordinary statements."""
        assert_program_stdout(
            "def count_up(int n):\n"
            "    int i = 0\n"
            "    while i < n:\n"
            "        print(i)\n"
            "        i = i + 1\n"
            "\n"
            "def int main():\n"
            "    count_up(3)\n"
            "    return 0\n",
            "0\n1\n2\n",
        )

    def test_returning_a_value_from_a_void_function_is_rejected(self):
        assert_program_semantic_error(
            "def log(int x):\n"
            "    return x\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            match="cannot return a value",
        )

    def test_bare_return_inside_a_non_void_function_is_rejected(self):
        assert_semantic_error(
            "    return",
            match="bare 'return' returns nothing",
        )

    def test_using_a_void_call_result_as_a_value_is_rejected(self):
        source = (
            "def log(int x):\n"
            "    print(x)\n"
            "\n"
            "def int main():\n"
            "    int y = log(5)\n"
            "    return y\n"
        )
        ast = _parse(source)
        with pytest.raises(SemanticError, match="Cannot initialize"):
            analyze(ast)

    def test_non_void_function_still_requires_explicit_returns_on_every_path(self):
        """The regression check: a function with a REAL declared
        return type still goes through always_returns exactly as
        before -- the skip in analyze_function is specific to
        Type.VOID, not a blanket relaxation."""
        source = (
            "def int classify(int x):\n"
            "    if x < 0:\n"
            "        return -1\n"
            "    print(x)\n"
            "\n"
            "def int main():\n"
            "    return classify(5)\n"
        )
        ast = _parse(source)
        with pytest.raises(SemanticError, match="does not return a value on all code paths"):
            analyze(ast)

    def test_comparing_two_void_call_results_is_rejected(self):
        """`Type.VOID == Type.VOID` is trivially true by structural
        equality alone -- the same way any type equals itself -- so
        this needs its own explicit rejection in check_binary rather
        than falling out for free the way every OTHER "void used as a
        value" case already does (see check_binary's own comment)."""
        source = (
            "def log(int x):\n"
            "    print(x)\n"
            "\n"
            "def bool main():\n"
            "    return log(1) == log(2)\n"
        )
        ast = _parse(source)
        with pytest.raises(SemanticError, match="does not support array, slice, void, or none operands"):
            analyze(ast)


# ---------------------------------------------------------------------------
# Type annotation: semantic.py's check_expr annotates every expression
# node with its resolved type (expr.resolved_type), which codegen.py's
# _type_of reads directly instead of re-deriving a type independently.
#
# This replaced a previous codegen.py-internal method, _infer_type, that
# duplicated -- in miniature, via its own per-operator/per-node-type
# branches -- the same "what type does this produce" logic semantic.py's
# check_binary/check_call already fully implement. That duplication
# wasn't hypothetical risk: it silently caused two real bugs. Adding
# `print` needed a Call case added to _infer_type separately from
# semantic.py's own check_call; adding the six int-only operators (%  &
# | ^ << >>) needed them added to _infer_type's int-producing branch
# separately from semantic.py's _INT_ONLY_BINARY_OPS. Neither omission
# caused an immediate, loud failure -- both were only caught by manual
# testing during those turns, which is exactly the failure mode worth
# structurally preventing rather than just fixing twice.
#
# These tests specifically target the annotation mechanism and the two
# bug patterns above -- most of the ordinary coverage that this
# mechanism also has to get right already exists throughout the rest of
# this file (every test that computes a nontrivial expression exercises
# it, whether or not that test was written with this in mind).
# ---------------------------------------------------------------------------

class TestTypeAnnotation:
    pytestmark = GCC_SKIP

    def test_call_result_directly_as_operand_of_plus(self):
        """The exact shape of the first bug this refactor prevents:
        codegen needs to know a Call expression's type to decide
        whether the outer `+` means concatenation or arithmetic, with
        the call's result never stored in an intermediate variable
        first -- every existing function-call test always assigns a
        call's result to a variable before using it further, so this
        specific shape wasn't previously covered anywhere."""
        assert_program_exit_code(
            "def int five():\n"
            "    return 5\n"
            "\n"
            "def int main():\n"
            "    int x = five() + 3\n"
            "    return x\n",
            8,
        )

    def test_modulo_result_directly_as_operand_of_plus(self):
        """The exact shape of the second bug this refactor prevents.
        Already covered by test_modulo_result_used_as_operand_of_plus
        in TestBitwiseAndModuloOperators (added at the time that bug
        was found); repeated here as a direct regression test scoped
        to the annotation mechanism itself, so this file's own
        organization doesn't obscure that the two bugs share one root
        cause and one fix."""
        assert_exit_code(
            "    int x = 5 % 2 + 3\n"
            "    return x",
            4,
        )

    def test_deeply_nested_mixed_expression_annotates_and_executes_correctly(self):
        """A kitchen-sink expression touching every expression node
        type and several operator categories at once -- a call, int
        arithmetic, modulo, a comparison, a bitwise AND, equality,
        unary not, and logical and -- nested three levels deep. Proof
        the annotation mechanism correctly threads a resolved type
        through arbitrary nesting, not just each category checked in
        isolation. Also, not incidentally, another instance of the
        bitwise/equality-precedence type error from
        TestSemanticErrors' test_bitwise_and_equality_precedence_is_a_
        type_error -- `(5 & 2) == 2` needs its explicit parens for
        exactly the same reason `(1 & 2) == 2` does there."""
        assert_program_exit_code(
            "def int add(int a, int b):\n"
            "    return a + b\n"
            "\n"
            "def bool main():\n"
            "    int n = add(1, 2) + (3 % 2)\n"
            "    bool b = (n > 0) and not ((5 & 2) == 2)\n"
            "    return b\n",
            1,
        )

    def test_codegen_without_semantic_analysis_raises_clear_error(self):
        """_type_of's defensive check: codegen invoked on an AST that
        skipped semantic.analyze() (so no node has a resolved_type)
        must fail with a clear, actionable CodegenError -- matching
        _local_offset's own established posture -- rather than a bare
        AttributeError or, worse, silently wrong codegen."""
        ast = _parse("def int main():\n    return 1 + 2\n")
        # Deliberately not calling analyze(ast) here.
        with pytest.raises(CodegenError, match="has no resolved type"):
            generate_asm(ast, platform=ASM_PLATFORM)


# ---------------------------------------------------------------------------
# print: the first builtin. Every test here checks actual stdout content
# via assert_stdout/assert_program_stdout, not just an exit code -- exit
# codes can't tell "printed the right thing" from "printed nothing at
# all", which is exactly the distinction that matters for a function
# whose entire purpose is its side effect.
# ---------------------------------------------------------------------------

class TestPrint:
    pytestmark = GCC_SKIP

    def test_print_int(self):
        assert_stdout(
            "    print(5)\n"
            "    return 0",
            "5\n",
        )

    def test_print_negative_int(self):
        assert_stdout(
            "    print(-42)\n"
            "    return 0",
            "-42\n",
        )

    def test_print_str(self):
        assert_stdout(
            "    print('hello')\n"
            "    return 0",
            "hello\n",
        )

    def test_print_bool_true(self):
        assert_stdout(
            "    print(true)\n"
            "    return 0",
            "true\n",
        )

    def test_print_bool_false(self):
        assert_stdout(
            "    print(false)\n"
            "    return 0",
            "false\n",
        )

    def test_print_expression_results_not_just_literals(self):
        """print's argument can be any expression, not just a bare
        literal -- proves the argument is genuinely evaluated first,
        not special-cased to only accept literal syntax."""
        assert_stdout(
            "    int a = 3\n"
            "    int b = 4\n"
            "    print(a + b)\n"
            "    print(a > b)\n"
            "    print('re' + 'sult')\n"
            "    return 0",
            "7\nfalse\nresult\n",
        )

    def test_print_multiple_calls_in_sequence(self):
        assert_stdout(
            "    print(1)\n"
            "    print(2)\n"
            "    print(3)\n"
            "    return 0",
            "1\n2\n3\n",
        )

    def test_print_inside_a_loop(self):
        assert_stdout(
            "    int i = 0\n"
            "    while i < 3:\n"
            "        print(i)\n"
            "        i = i + 1\n"
            "    return 0",
            "0\n1\n2\n",
        )

    def test_print_inside_if_branches(self):
        assert_stdout(
            "    bool flag = true\n"
            "    if flag:\n"
            "        print('yes')\n"
            "    else:\n"
            "        print('no')\n"
            "    return 0",
            "yes\n",
        )

    def test_print_result_not_usable_as_a_value(self):
        """print is Type.VOID (see semantic.py's check_print_call) --
        this used to be a positive test for print "returning" a usable
        int 0, back when there was no real void type to give it and
        that was the documented workaround. Now that a function with no
        declared return type exists, print became its first real user,
        and using its result as a value is a genuine semantic error,
        the same as calling any other void function that way -- caught
        here by the same type-mismatch check '+' already had for any
        other non-int/non-str operand, with no void-specific code
        needed at this particular call site."""
        assert_semantic_error(
            "    int x = print(5) + 41\n"
            "    return x",
            match="requires two int operands or two str operands, got void",
        )

    def test_print_inside_a_user_defined_function(self):
        assert_program_stdout(
            "def int announce(int x):\n"
            "    print(x)\n"
            "    return x * 2\n"
            "\n"
            "def int main():\n"
            "    int result = announce(21)\n"
            "    print(result)\n"
            "    return 0\n",
            "21\n42\n",
        )

    def test_print_repeated_int_calls_reuse_cached_format_string(self):
        """Not observable from stdout content alone, but this exercises
        the lazy-cached %d format-string label (see codegen.py's
        _get_int_format_label) across multiple print(int) calls in the
        same program -- if the caching logic were broken (e.g. reusing
        a stale label across functions), this would show up as garbled
        or missing output rather than a clean failure."""
        assert_stdout(
            "    print(1)\n"
            "    print(22)\n"
            "    print(333)\n"
            "    return 0",
            "1\n22\n333\n",
        )


# ---------------------------------------------------------------------------
# All-paths-return checking (semantic.py's always_returns /
# contains_reachable_break). Every function needs this regardless of
# return type, since this language has no void -- but it became a real
# safety issue, not just a correctness nicety, once functions could call
# each other: control falling off the end of a function's generated
# code with no `ret` executed corrupts the *calling* function's own
# stack, not just the callee's exit code.
#
# The genuinely subtle case here is `while true` with a `break` inside
# it -- a bare `while true: ...; return x` is fine on its own (the loop
# never falls through: it either returns from inside or runs forever),
# but the moment a `break` exists anywhere in that loop's body (even
# buried inside a nested if/elif chain), the loop CAN fall through to
# whatever comes after it, so it stops counting as guaranteeing a
# return and something has to catch that path explicitly. The
# CRITICAL-labeled tests are the ones that would actually catch a
# mistake in this specific piece of the algorithm, not just prove the
# ordinary if/else and trailing-return cases work.
# ---------------------------------------------------------------------------

class TestAllPathsReturn:

    # -- accepted (analyze() must NOT raise) -------------------------------

    def test_simple_trailing_return(self):
        ast = _parse("def int f():\n    return 1\n")
        analyze(ast)  # should not raise

    def test_if_else_both_branches_return(self):
        ast = _parse(
            "def int f(int x):\n"
            "    if x > 0:\n"
            "        return 1\n"
            "    else:\n"
            "        return 2\n"
        )
        analyze(ast)  # should not raise

    def test_if_without_else_followed_by_trailing_return(self):
        """An if with no else can never guarantee a return by itself --
        it's the return statement *after* it that makes this valid."""
        ast = _parse(
            "def int f(int x):\n"
            "    if x > 0:\n"
            "        return 1\n"
            "    return 2\n"
        )
        analyze(ast)  # should not raise

    def test_if_elif_else_chain_all_branches_return(self):
        """elif desugars into a nested If in else_body (see parser.py),
        so this also proves always_returns recurses correctly through
        an elif chain of arbitrary length, not just a single if/else."""
        ast = _parse(
            "def int f(int x):\n"
            "    if x > 0:\n"
            "        return 1\n"
            "    elif x < 0:\n"
            "        return 2\n"
            "    else:\n"
            "        return 3\n"
        )
        analyze(ast)  # should not raise

    def test_if_elif_without_final_else_followed_by_trailing_return(self):
        """The elif chain itself isn't exhaustive (no final else), but
        the trailing return after it catches every path that falls
        through the chain without returning."""
        ast = _parse(
            "def int f(int x):\n"
            "    if x > 0:\n"
            "        return 1\n"
            "    elif x < 0:\n"
            "        return 2\n"
            "    return 99\n"
        )
        analyze(ast)  # should not raise

    def test_while_true_with_no_break_needs_no_trailing_return(self):
        """A genuine `while true` with nothing that can break out of it
        never falls through to whatever comes after it -- it either
        loops forever or returns from inside -- so this is valid even
        though nothing follows the loop and the loop body itself has no
        return in it. See codegen.py's own gaps around genuinely
        infinite loops for the flip side of this: this is a legitimate,
        if unusual, thing to write."""
        ast = _parse(
            "def int f():\n"
            "    while true:\n"
            "        int x = 1\n"
        )
        analyze(ast)  # should not raise

    def test_critical_while_true_with_break_and_trailing_return(self):
        """The positive control for the critical case: once a `while
        true` loop has a `break`, the loop alone can no longer
        guarantee a return -- but an explicit return placed after the
        loop correctly catches the break-exit path."""
        ast = _parse(
            "def int f(bool x):\n"
            "    while true:\n"
            "        if x:\n"
            "            break\n"
            "        int y = 1\n"
            "    return 99\n"
        )
        analyze(ast)  # should not raise

    def test_critical_nested_while_true_inner_break_does_not_satisfy_outer(self):
        """A break inside a nested while loop belongs to that inner
        loop, not the outer one (the exact same scoping break already
        has for its own semantic validity -- see analyze_break/
        loop_depth -- and at the codegen level -- see codegen.py's
        loop_labels stack). So the outer while here is correctly still
        recognized as unbreakable-except-by-return, purely because of
        its own trailing `return 1`, with the inner loop's break having
        no bearing on that."""
        ast = _parse(
            "def int f():\n"
            "    while true:\n"
            "        while true:\n"
            "            break\n"
            "        return 1\n"
        )
        analyze(ast)  # should not raise

    def test_finite_while_loop_followed_by_trailing_return(self):
        """The most common real shape: an ordinary, condition-bounded
        loop (not `while true`) can never itself guarantee a return --
        its condition might be false immediately -- so it's the return
        after the loop that makes this valid, exactly like an if
        without an else."""
        ast = _parse(
            "def int f():\n"
            "    int i = 0\n"
            "    while i < 10:\n"
            "        i = i + 1\n"
            "    return i\n"
        )
        analyze(ast)  # should not raise

    def test_str_returning_function_with_trailing_return(self):
        """The check applies uniformly regardless of the function's
        declared return type -- this isn't an int/bool-specific rule."""
        ast = _parse(
            "def str f():\n"
            "    str s = 'hello'\n"
            "    return s\n"
        )
        analyze(ast)  # should not raise

    def test_critical_break_inside_elif_chain_inside_while_true_with_trailing_return(self):
        """A break buried three levels deep inside an elif chain,
        itself inside a while-true loop, must still be found by
        contains_reachable_break (which has to recurse through If's
        then_body/else_body, including the nested-If shape an elif
        chain desugars into) -- and the trailing return after the loop
        must still correctly catch the resulting break-exit path."""
        ast = _parse(
            "def int f(int x):\n"
            "    while true:\n"
            "        if x == 1:\n"
            "            int y = 1\n"
            "        elif x == 2:\n"
            "            break\n"
            "        else:\n"
            "            int z = 1\n"
            "        return 1\n"
            "    return 99\n"
        )
        analyze(ast)  # should not raise

    # -- rejected (analyze() must raise SemanticError) ---------------------

    def test_no_return_at_all(self):
        assert_semantic_error(
            "    int x = 1",
            match="does not return a value on all code paths",
        )

    def test_print_only_function_with_no_return(self):
        """print's own presence has no bearing on this check -- it's
        just an ordinary expression statement as far as always_returns
        is concerned."""
        assert_semantic_error(
            "    print(5)",
            match="does not return a value on all code paths",
        )

    def test_if_without_else_and_nothing_after_it(self):
        assert_semantic_error(
            "    bool x = true\n"
            "    if x:\n"
            "        return 1",
            match="does not return a value on all code paths",
        )

    def test_if_elif_without_final_else_and_nothing_after_it(self):
        assert_semantic_error(
            "    int x = 0\n"
            "    if x == 1:\n"
            "        return 1\n"
            "    elif x == 2:\n"
            "        return 2",
            match="does not return a value on all code paths",
        )

    def test_critical_while_true_with_break_and_no_trailing_return(self):
        """The critical negative case: exactly the shape of program
        that motivated this whole feature -- a `while true` loop that
        looks like it always returns at a glance (it has a return
        inside it), but can actually fall through to the end of the
        function whenever `x` is true and the break fires."""
        assert_semantic_error(
            "    bool x = true\n"
            "    while true:\n"
            "        if x:\n"
            "            break\n"
            "        return 1",
            match="does not return a value on all code paths",
        )

    def test_finite_while_loop_with_nothing_after_it(self):
        """A condition-bounded while loop's body might never execute
        (the condition could be false from the start), so even a
        return unconditionally reached *inside* the loop body doesn't
        help if there's nothing after the loop to catch the
        zero-iterations case."""
        assert_semantic_error(
            "    bool x = true\n"
            "    int i = 0\n"
            "    while i < 10:\n"
            "        if x:\n"
            "            return 1\n"
            "        i = i + 1",
            match="does not return a value on all code paths",
        )

    def test_str_returning_function_with_no_return(self):
        assert_semantic_error(
            "    str s = 'hello'",
            match="does not return a value on all code paths",
            return_type="str",
        )


# ---------------------------------------------------------------------------
# Arrays: fixed-size, stack-allocated, value-typed (see codegen.py's
# ARRAYS section for the full design). Scope note, repeated from there:
# this covers LOCAL arrays completely -- declaration (literal- or
# copy-initialized), reading/writing an element at any nesting depth,
# and whole-array copy via plain assignment -- but array function
# PARAMETERS and RETURN VALUES are a deliberately separate, not-yet-
# built piece of work (a real calling-convention extension), covered
# here only by the tests proving that gap fails with a clear,
# actionable error rather than silently miscompiling.
#
# test_value_semantics_1d/2d and test_sub_array_extraction_is_independent
# are the tests that actually prove the headline design decision --
# arrays are values, not references -- holds at the machine-code level:
# mutating a copy must never affect the original. The bounds-checking
# tests are the other centerpiece: TestBoundsChecking proves the single
# unsigned comparison genuinely catches both an over-large index and a
# negative one, correctly leaves a valid boundary index alone, and --
# found only by testing, not assumed -- that the panic message actually
# reaches the user rather than being silently lost in an unflushed
# stdio buffer when abort() bypasses the normal exit() path.
# ---------------------------------------------------------------------------

class TestArrays:
    pytestmark = GCC_SKIP

    def test_basic_1d_array_read(self):
        assert_exit_code(
            "    [3]int arr = [10, 20, 30]\n"
            "    return arr[0] + arr[1] + arr[2]",
            60,
        )

    def test_index_write(self):
        assert_exit_code(
            "    [3]int arr = [1, 2, 3]\n"
            "    arr[1] = 99\n"
            "    return arr[1]",
            99,
        )

    def test_value_semantics_1d(self):
        """The headline design property: assigning one array to
        another copies its elements -- it does not alias them.
        Mutating the copy must leave the original untouched."""
        assert_exit_code(
            "    [3]int a = [1, 2, 3]\n"
            "    [3]int b = [0, 0, 0]\n"
            "    b = a\n"
            "    b[0] = 99\n"
            "    return a[0] == 1 and b[0] == 99",
            1,
            return_type="bool",
        )

    def test_2d_array_read(self):
        assert_exit_code(
            "    [2][3]int matrix = [[1, 2, 3], [4, 5, 6]]\n"
            "    return matrix[0][1] + matrix[1][2]",
            8,
        )

    def test_2d_array_index_write(self):
        assert_exit_code(
            "    [2][3]int matrix = [[1, 2, 3], [4, 5, 6]]\n"
            "    matrix[1][0] = 99\n"
            "    return matrix[1][0]",
            99,
        )

    def test_value_semantics_2d(self):
        assert_exit_code(
            "    [2][2]int a = [[1, 2], [3, 4]]\n"
            "    [2][2]int b = [[0, 0], [0, 0]]\n"
            "    b = a\n"
            "    b[0][0] = 99\n"
            "    return a[0][0] == 1 and b[0][0] == 99",
            1,
            return_type="bool",
        )

    def test_sub_array_extraction_is_independent(self):
        """`[3]int row = matrix[1]` extracts a whole row -- and per the
        same value semantics, `row` is its own independent copy, not an
        alias into `matrix`'s own storage."""
        assert_exit_code(
            "    [2][3]int matrix = [[1, 2, 3], [4, 5, 6]]\n"
            "    [3]int row = matrix[1]\n"
            "    row[0] = 99\n"
            "    return matrix[1][0] == 4 and row[0] == 99",
            1,
            return_type="bool",
        )

    def test_array_of_str_elements(self):
        assert_exit_code(
            "    [3]str names = ['alice', 'bob', 'carol']\n"
            "    return names[0] == 'alice' and names[2] == 'carol'",
            1,
            return_type="bool",
        )

    def test_array_element_in_larger_expression(self):
        assert_exit_code(
            "    [4]int arr = [10, 20, 30, 40]\n"
            "    int i = 2\n"
            "    return arr[i] * 2 + arr[0]",
            70,
        )

    def test_array_element_as_function_argument(self):
        assert_program_exit_code(
            "def int double(int x):\n"
            "    return x * 2\n"
            "\n"
            "def int main():\n"
            "    [3]int arr = [5, 10, 15]\n"
            "    return double(arr[1])\n",
            20,
        )

    def test_array_iteration_via_while_loop(self):
        assert_exit_code(
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    int sum = 0\n"
            "    int i = 0\n"
            "    while i < 5:\n"
            "        sum = sum + arr[i]\n"
            "        i = i + 1\n"
            "    return sum",
            15,
        )

    def test_index_assignment_with_computed_index(self):
        assert_exit_code(
            "    [5]int arr = [0, 0, 0, 0, 0]\n"
            "    int i = 1\n"
            "    arr[i + 1] = 42\n"
            "    return arr[2]",
            42,
        )

    def test_array_parameter_basic(self):
        assert_program_exit_code(
            "def int sum_array([3]int arr):\n"
            "    return arr[0] + arr[1] + arr[2]\n"
            "\n"
            "def int main():\n"
            "    [3]int a = [1, 2, 3]\n"
            "    return sum_array(a)\n",
            6,
        )

    def test_array_parameter_value_semantics(self):
        """The parameter-passing counterpart to test_value_semantics_1d:
        mutating an array PARAMETER inside the callee must never affect
        the caller's own array -- the callee receives a pointer to a
        copy the caller made just for this call (see
        gen_array_arg_address_into), not a reference to the original."""
        assert_program_exit_code(
            "def int mutate([3]int arr):\n"
            "    arr[0] = 999\n"
            "    return arr[0]\n"
            "\n"
            "def bool main():\n"
            "    [3]int a = [1, 2, 3]\n"
            "    int result = mutate(a)\n"
            "    return result == 999 and a[0] == 1\n",
            1,
        )

    def test_array_return_basic(self):
        assert_program_exit_code(
            "def [3]int make():\n"
            "    [3]int r = [10, 20, 30]\n"
            "    return r\n"
            "\n"
            "def int main():\n"
            "    [3]int x = make()\n"
            "    return x[0] + x[1] + x[2]\n",
            60,
        )

    def test_array_return_direct_literal(self):
        """Regression test for a real bug found during development:
        `return [1,2,3]` writes each element straight through the
        hidden return pointer without ever materializing an
        intermediate local. When that pointer happens to be sitting in
        %rax (the same register gen_expr_into always computes an
        element's value into), evaluating the first element used to
        silently destroy the pointer before anything was ever written
        through it -- a segfault, not a wrong answer, since the write
        landed at whatever address the corrupted "pointer" happened to
        be. See gen_array_literal_into's own docstring for the fix."""
        assert_program_exit_code(
            "def [3]int make():\n"
            "    return [10, 20, 30]\n"
            "\n"
            "def int main():\n"
            "    [3]int x = make()\n"
            "    return x[0] + x[1] + x[2]\n",
            60,
        )

    def test_array_return_via_sub_array_index(self):
        """Another real bug found during development, the same class as
        the literal-return one above but one layer deeper: `return
        matrix[i]` computes the sub-array's SOURCE address (bounds-
        checking and index arithmetic that freely use %rax/%rcx
        internally) before ever touching the destination -- which,
        again, could be sitting in %rax. See gen_array_value_into's
        _gen_protecting_dst_across for the fix."""
        assert_program_exit_code(
            "def [3]int get_row([2][3]int matrix, int i):\n"
            "    return matrix[i]\n"
            "\n"
            "def int main():\n"
            "    [2][3]int m = [[1, 2, 3], [4, 5, 6]]\n"
            "    [3]int row = get_row(m, 1)\n"
            "    return row[0] + row[1] + row[2]\n",
            15,
        )

    def test_nested_array_returning_call_forwarding(self):
        """`return inner()`, where inner ALSO returns an array, forwards
        the same hidden pointer one level deeper with no intermediate
        copy ever materialized -- see gen_array_call_into's own
        docstring."""
        assert_program_exit_code(
            "def [3]int inner():\n"
            "    return [7, 8, 9]\n"
            "\n"
            "def [3]int outer():\n"
            "    return inner()\n"
            "\n"
            "def int main():\n"
            "    [3]int x = outer()\n"
            "    return x[0] + x[1] + x[2]\n",
            24,
        )

    def test_2d_array_as_parameter_and_return_type(self):
        assert_program_exit_code(
            "def [2][2]int double_all([2][2]int m):\n"
            "    [2][2]int result = [[0, 0], [0, 0]]\n"
            "    int i = 0\n"
            "    while i < 2:\n"
            "        int j = 0\n"
            "        while j < 2:\n"
            "            result[i][j] = m[i][j] * 2\n"
            "            j = j + 1\n"
            "        i = i + 1\n"
            "    return result\n"
            "\n"
            "def int main():\n"
            "    [2][2]int a = [[1, 2], [3, 4]]\n"
            "    [2][2]int b = double_all(a)\n"
            "    return b[0][0] + b[0][1] + b[1][0] + b[1][1]\n",
            20,
        )

    def test_multiple_array_parameters(self):
        assert_program_exit_code(
            "def int dot_product([3]int a, [3]int b):\n"
            "    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]\n"
            "\n"
            "def int main():\n"
            "    [3]int x = [1, 2, 3]\n"
            "    [3]int y = [4, 5, 6]\n"
            "    return dot_product(x, y)\n",
            32,
        )

    def test_returned_array_independent_across_separate_calls(self):
        """Value semantics across the return boundary: two separate
        calls to the same array-returning function must produce two
        completely independent results, even though the function
        builds its result in the exact same local slot both times."""
        assert_program_exit_code(
            "def [3]int make_and_mutate():\n"
            "    [3]int local = [1, 2, 3]\n"
            "    local[0] = 100\n"
            "    return local\n"
            "\n"
            "def bool main():\n"
            "    [3]int a = make_and_mutate()\n"
            "    [3]int b = make_and_mutate()\n"
            "    a[1] = 999\n"
            "    return a[0] == 100 and a[1] == 999 and b[0] == 100 and b[1] == 2\n",
            1,
        )

    def test_array_argument_as_index_expression(self):
        """An array-typed argument doesn't have to be a bare variable --
        gen_array_arg_address_into also accepts an Index yielding a
        sub-array, e.g. passing one row of a matrix straight through."""
        assert_program_exit_code(
            "def int sum3([3]int arr):\n"
            "    return arr[0] + arr[1] + arr[2]\n"
            "\n"
            "def int main():\n"
            "    [2][3]int matrix = [[1, 2, 3], [4, 5, 6]]\n"
            "    return sum3(matrix[1])\n",
            15,
        )

    def test_str_element_array_as_parameter_and_return_type(self):
        assert_program_exit_code(
            "def bool first_is([3]str names, str target):\n"
            "    return names[0] == target\n"
            "\n"
            "def [3]str make_names():\n"
            "    return ['x', 'y', 'z']\n"
            "\n"
            "def bool main():\n"
            "    [3]str n = make_names()\n"
            "    return first_is(n, 'x') and n[2] == 'z'\n",
            1,
        )

    def test_five_real_params_on_array_returning_function(self):
        """The boundary case: an array-returning function supports at
        most 5 REAL parameters, one fewer than the usual 6, since the
        hidden output pointer itself occupies the first argument
        register."""
        assert_program_exit_code(
            "def [2]int make5(int a, int b, int c, int d, int e):\n"
            "    return [a + b, c + d + e]\n"
            "\n"
            "def int main():\n"
            "    [2]int r = make5(1, 2, 3, 4, 5)\n"
            "    return r[0] + r[1]\n",
            15,
        )

    def test_six_real_params_on_array_returning_function_is_rejected(self):
        source = (
            "def [2]int make6(int a, int b, int c, int d, int e, int f):\n"
            "    return [a, b]\n"
            "\n"
            "def int main():\n"
            "    [2]int r = make6(1, 2, 3, 4, 5, 6)\n"
            "    return r[0]\n"
        )
        ast = _parse(source)
        analyze(ast)  # semantically fine -- the limit is codegen-level only
        with pytest.raises(CodegenError, match="needs 7 argument register"):
            generate_asm(ast, platform=ASM_PLATFORM)

    def test_array_literal_as_direct_call_argument_not_supported(self):
        """A real, deliberate gap, not silent mishandling: an
        ArrayLiteral (or a call returning an array) has no address of
        its own to pass as an argument -- see
        gen_array_arg_address_into's own docstring for the workaround
        (assign it to a named variable first, which already works)."""
        source = (
            "def int sum3([3]int arr):\n"
            "    return arr[0] + arr[1] + arr[2]\n"
            "\n"
            "def int main():\n"
            "    return sum3([1, 2, 3])\n"
        )
        ast = _parse(source)
        analyze(ast)  # semantically fine -- the gap is codegen-level only
        with pytest.raises(CodegenError, match="assign it to a variable first"):
            generate_asm(ast, platform=ASM_PLATFORM)


class TestBoundsChecking:
    """Every array access is runtime-checked (see codegen.py's
    gen_index_address_into): a single unsigned comparison against the
    array's size catches an over-large index and a negative one alike,
    since a negative int reinterpreted unsigned becomes a huge positive
    number. test_panic_message_survives_piped_output specifically
    guards against a real bug found during development, not a
    hypothetical one: abort() terminates via a raw signal, bypassing
    the normal exit() path that would otherwise flush libc's buffered
    stdio -- without an explicit fflush(NULL) before the abort() call,
    the panic message was reliably printed to an interactive terminal
    but silently lost whenever output was piped or redirected, which is
    the common case for a program run non-interactively.
    """
    pytestmark = GCC_SKIP

    def test_index_too_large_aborts(self):
        assert_crashes_with_sigabrt(
            "    [3]int arr = [1, 2, 3]\n"
            "    int i = 5\n"
            "    return arr[i]"
        )

    def test_negative_index_aborts(self):
        assert_crashes_with_sigabrt(
            "    [3]int arr = [1, 2, 3]\n"
            "    int i = 0 - 1\n"
            "    return arr[i]"
        )

    def test_valid_boundary_index_does_not_abort(self):
        """The positive control: the LAST valid index (size - 1) must
        not trip the bounds check -- proof the comparison's boundary
        condition (unsigned >=, not >) is exactly right, not
        off-by-one in either direction."""
        assert_exit_code(
            "    [3]int arr = [10, 20, 30]\n"
            "    int i = 2\n"
            "    return arr[i]",
            30,
        )

    def test_panic_message_survives_piped_output(self):
        """Regression test for a real bug found during development:
        the "array index out of bounds" message must actually reach
        stdout, not be silently discarded in an unflushed buffer when
        abort() bypasses the normal exit() path. Captures output
        directly (compile_and_run's capture_output=True) rather than
        relying on an interactive terminal's line-buffering to mask
        the bug the way a casual manual test would."""
        result = compile_and_run(
            "def int main():\n"
            "    [3]int arr = [1, 2, 3]\n"
            "    int i = 5\n"
            "    return arr[i]\n"
        )
        assert result.returncode == -signal.SIGABRT
        assert "array index out of bounds" in result.stdout


# ---------------------------------------------------------------------------
# Size-based stack safety: an array over _STACK_ARRAY_LIMIT_BYTES (16KB,
# hardcoded -- see codegen.py's is_heap_allocated) is heap-allocated
# instead of living inline in its own stack slot, closing off the one
# concrete danger fixed-size arrays already had before any of this
# existed: nothing stopped a single huge array from silently blowing
# the stack, the exact same way it wouldn't in C. This is deliberately
# a PER-ARRAY check, not a per-frame budget -- see is_heap_allocated's
# own docstring for the accepted gaps that leaves (several moderate
# arrays in one function, or a moderate array under deep recursion,
# can still exhaust the stack even though no single array ever trips
# this check).
#
# test_exactly_at_threshold_stays_on_stack and
# test_just_over_threshold_is_heap_allocated both inspect the generated
# assembly directly (via generate_asm) rather than only checking exit
# codes -- proof the boundary itself is exactly right (>, not >=),
# not just that some array somewhere behaves plausibly.
#
# The other tests exist because heap-promoting an array is a real,
# separate code path through nearly every piece of array codegen, not
# a transparent swap of one allocator for another: gen_var_decl's
# malloc-then-store-initializer path is fully distinct from its
# malloc-then-copy path (an ArrayLiteral initializer vs. a Variable/
# Index/Call one), gen_assign has to reuse the existing allocation
# rather than mallocing again, and gen_function's parameter loop needs
# its own independent copy of a heap-allocated argument to preserve
# value semantics across a call, exactly like the stack-allocated case
# already had. Every one of these is exercised directly below, not
# just assumed to follow from the allocation-site change alone.
# ---------------------------------------------------------------------------

class TestHeapAllocatedArrays:
    pytestmark = GCC_SKIP

    def test_exactly_at_threshold_stays_on_stack(self):
        """The boundary case: exactly _STACK_ARRAY_LIMIT_BYTES (16384)
        must NOT be heap-allocated -- checked by inspecting the
        generated assembly for the complete absence of a malloc call,
        not just by trusting a plausible-looking exit code."""
        n = 4096  # 4096 * 4 = 16384, exactly the threshold
        source = (
            f"def int main():\n"
            f"    [{n}]int arr\n"
            f"    arr[0] = 1\n"
            f"    return arr[0]\n"
        )
        ast = _parse(source)
        analyze(ast)
        asm = generate_asm(ast, platform=ASM_PLATFORM)
        assert "malloc" not in asm

    def test_just_over_threshold_is_heap_allocated(self):
        """The other side of the same boundary: even one byte over the
        threshold must be heap-allocated -- checked by confirming a
        malloc call is present, sized to the array's own exact
        footprint, not merely "big enough"."""
        n = 4097  # 4097 * 4 = 16388, one int over the threshold
        source = (
            f"def int main():\n"
            f"    [{n}]int arr\n"
            f"    arr[0] = 1\n"
            f"    return arr[0]\n"
        )
        ast = _parse(source)
        analyze(ast)
        asm = generate_asm(ast, platform=ASM_PLATFORM)
        assert "malloc" in asm
        assert "$16388" in asm

    def test_heap_allocated_local_basic_read_write(self):
        assert_exit_code(
            "    [10000]int big\n"
            "    big[0] = 42\n"
            "    big[9999] = 99\n"
            "    return big[0] + big[9999]",
            141,
        )

    def test_heap_allocated_local_with_literal_initializer(self):
        """gen_var_decl's malloc-then-store-initializer path is
        genuinely distinct code from its malloc-then-copy path (see
        this class's own module-level comment) -- exercised directly
        with a literal large enough to force heap promotion, not
        inferred from the Variable-initializer case below."""
        n = 4200  # 4200 * 4 = 16800 bytes, over the threshold
        elems = ', '.join(str(i % 10) for i in range(n))
        assert_program_exit_code(
            f"def int main():\n"
            f"    [{n}]int arr = [{elems}]\n"
            f"    return arr[0] + arr[1] + arr[9] + arr[{n - 1}]\n",
            19,  # 0 + 1 + 9 + (4199 % 10 == 9)
        )

    def test_heap_allocated_value_semantics(self):
        """The headline property, for heap-backed storage specifically:
        assigning one heap-allocated array to another still copies
        elements rather than aliasing the pointer -- reusing the
        destination's existing allocation (see gen_assign's own array
        case) rather than mallocing again, but still a real,
        independent copy."""
        assert_exit_code(
            "    [10000]int a\n"
            "    [10000]int b\n"
            "    a[0] = 1\n"
            "    b[0] = 0\n"
            "    b = a\n"
            "    b[0] = 999\n"
            "    return a[0] == 1 and b[0] == 999",
            1,
            return_type="bool",
        )

    def test_heap_allocated_2d_array_value_semantics(self):
        assert_exit_code(
            "    [100][100]int grid\n"
            "    grid[0][0] = 1\n"
            "    grid[99][99] = 2\n"
            "    [100][100]int copy = grid\n"
            "    copy[0][0] = 999\n"
            "    return grid[0][0] == 1 and copy[0][0] == 999 and copy[99][99] == 2",
            1,
            return_type="bool",
        )

    def test_heap_allocated_array_as_function_parameter(self):
        assert_program_exit_code(
            "def int sum_first_and_last([10000]int arr):\n"
            "    return arr[0] + arr[9999]\n"
            "\n"
            "def int main():\n"
            "    [10000]int big\n"
            "    big[0] = 5\n"
            "    big[9999] = 7\n"
            "    return sum_first_and_last(big)\n",
            12,
        )

    def test_heap_allocated_parameter_value_semantics(self):
        """The parameter-passing counterpart to
        test_heap_allocated_value_semantics: a heap-allocated array
        parameter still gets its own independent copy on entry (see
        gen_function's own parameter loop) -- mutating it inside the
        callee must never affect the caller's original."""
        assert_program_exit_code(
            "def int mutate([10000]int arr):\n"
            "    arr[0] = 999\n"
            "    return arr[0]\n"
            "\n"
            "def bool main():\n"
            "    [10000]int big\n"
            "    big[0] = 1\n"
            "    int result = mutate(big)\n"
            "    return result == 999 and big[0] == 1\n",
            1,
        )

    def test_heap_allocated_array_as_return_type(self):
        """Array returns need no heap-allocation logic of their own at
        all -- see is_heap_allocated's own scope note -- since an
        array-typed return already writes directly through the
        caller-provided hidden pointer (see gen_return), regardless of
        the array's size. This just confirms that still holds once the
        array involved happens to be heap-allocated."""
        assert_program_exit_code(
            "def [10000]int make():\n"
            "    [10000]int r\n"
            "    r[0] = 42\n"
            "    r[9999] = 84\n"
            "    return r\n"
            "\n"
            "def int main():\n"
            "    [10000]int x = make()\n"
            "    return x[0] + x[9999]\n",
            126,
        )

    def test_heap_allocated_parameter_and_return_combined(self):
        assert_program_exit_code(
            "def [10000]int double_first([10000]int arr):\n"
            "    [10000]int result\n"
            "    result[0] = arr[0] * 2\n"
            "    return result\n"
            "\n"
            "def int main():\n"
            "    [10000]int a\n"
            "    a[0] = 21\n"
            "    [10000]int b = double_first(a)\n"
            "    return b[0]\n",
            42,
        )

    def test_multiple_heap_allocated_parameters(self):
        """Stress-tests gen_function's two-pass parameter handling:
        every incoming argument register is stashed into its own
        temporary slot before any parameter is processed, specifically
        because a heap-allocated parameter's malloc call can clobber
        ANY caller-saved register, including other, not-yet-processed
        parameters' own incoming values. Three heap-allocated
        parameters in one function is exactly the scenario that would
        expose a mistake in that stashing."""
        assert_program_exit_code(
            "def int sum_firsts([10000]int a, [10000]int b, [10000]int c):\n"
            "    return a[0] + b[0] + c[0]\n"
            "\n"
            "def int main():\n"
            "    [10000]int x\n"
            "    [10000]int y\n"
            "    [10000]int z\n"
            "    x[0] = 1\n"
            "    y[0] = 2\n"
            "    z[0] = 3\n"
            "    return sum_firsts(x, y, z)\n",
            6,
        )

    def test_mixed_parameter_types_with_heap_allocated_array(self):
        """A heap-allocated array parameter alongside ordinary scalar,
        str, and small (stack-allocated) array parameters in the same
        call -- confirms the two-pass parameter stashing handles a mix
        of types correctly, not just a function whose parameters are
        uniformly heap-allocated."""
        assert_program_exit_code(
            "def int mix(int a, str s, [3]int small, [10000]int big):\n"
            "    int slen_check = 0\n"
            "    if s == 'hi':\n"
            "        slen_check = 1\n"
            "    return a + slen_check + small[0] + big[0]\n"
            "\n"
            "def int main():\n"
            "    [10000]int huge\n"
            "    huge[0] = 100\n"
            "    [3]int small = [2, 0, 0]\n"
            "    return mix(1, 'hi', small, huge)\n",
            104,
        )


# ---------------------------------------------------------------------------
# Slices: Go-style views into an existing array or slice's own backing
# storage -- a fixed {pointer, length} descriptor (16 bytes), NOT a
# copy the way plain array assignment already is. This first pass is
# deliberately view-only: no append, no growth, no capacity -- just
# `base[low:high]`, both bounds optionally omitted, backed by whatever
# array or slice `base` already is.
#
# test_slice_write_mutates_underlying_array and
# test_overlapping_slices_alias_each_others_writes are the two tests
# that actually prove the entire point of this feature holds at the
# machine-code level, not just conceptually: a slice is a genuine
# alias, so writing through one must be visible through the array it
# came from, and through any OTHER slice that overlaps it -- if either
# of these failed, slices would just be a more awkward way to copy an
# array, not a real view.
#
# Any array that's ever sliced is unconditionally heap-allocated (see
# codegen.py's is_heap_allocated and its own ARRAYS section) --
# reusing the size-based promotion machinery with a second trigger --
# specifically so a slice can never outlive the stack frame its
# backing array would otherwise have lived in. That promotion isn't
# exercised directly in this class; it's implicit in every test here
# that slices anything, since the alternative (a dangling view into a
# torn-down stack frame) is exactly the memory-safety hole this
# feature was scoped to close from day one.
# ---------------------------------------------------------------------------

class TestSlices:
    pytestmark = GCC_SKIP

    def test_basic_slice_declare_and_index_read(self):
        assert_exit_code(
            "    [5]int arr = [10, 20, 30, 40, 50]\n"
            "    []int s = arr[1:4]\n"
            "    return s[0] + s[1] + s[2]",
            90,
        )

    def test_omitted_bounds(self):
        """All three omitted-bound forms together: `arr[:]` (both),
        `arr[2:]` (high only), `arr[:3]` (low only)."""
        assert_exit_code(
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    []int a = arr[:]\n"
            "    []int b = arr[2:]\n"
            "    []int c = arr[:3]\n"
            "    return a[4] + b[0] + c[2]",
            11,
        )

    def test_slicing_a_slice(self):
        """The trickiest case for gen_indexable_base_into/gen_slice_
        into: the BASE being sliced is itself a slice, so its own
        length is a runtime value read out of its descriptor, not a
        compile-time constant the way an array base's is."""
        assert_exit_code(
            "    [6]int arr = [1, 2, 3, 4, 5, 6]\n"
            "    []int s = arr[1:5]\n"
            "    []int s2 = s[1:3]\n"
            "    return s2[0] + s2[1]",
            7,
        )

    def test_indexing_slice_with_variable_index(self):
        assert_exit_code(
            "    [5]int arr = [10, 20, 30, 40, 50]\n"
            "    []int s = arr[1:4]\n"
            "    int i = 1\n"
            "    return s[i]",
            30,
        )

    def test_slicing_outer_dimension_of_2d_array(self):
        """`matrix[0:2]` yields a slice of ROWS ([][3]int), not a
        slice of ints -- confirmed by indexing two levels deep into
        the result."""
        assert_exit_code(
            "    [3][3]int matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\n"
            "    [][3]int rows = matrix[0:2]\n"
            "    return rows[0][0] + rows[1][2]",
            7,
        )

    def test_whole_slice_assignment(self):
        """`s2 = s1` copies s1's own {ptr, len} DESCRIPTOR into s2's
        slot -- after which s2 aliases whatever s1 aliased, not
        whatever s2 originally pointed at."""
        assert_exit_code(
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    []int s1 = arr[0:3]\n"
            "    []int s2 = arr[2:5]\n"
            "    s2 = s1\n"
            "    return s2[0] + s2[1] + s2[2]",
            6,
        )

    def test_slice_write_mutates_underlying_array(self):
        """The headline property: a slice is a genuine ALIAS into its
        base's own storage, not a copy -- writing through a slice
        index must be visible through the array it came from too."""
        assert_exit_code(
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    []int s = arr[1:4]\n"
            "    s[0] = 999\n"
            "    return arr[1] == 999 and s[0] == 999",
            1,
            return_type="bool",
        )

    def test_overlapping_slices_alias_each_others_writes(self):
        assert_exit_code(
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    []int s1 = arr[0:3]\n"
            "    []int s2 = arr[1:4]\n"
            "    s1[1] = 777\n"
            "    return s2[0] == 777",
            1,
            return_type="bool",
        )

    def test_slice_parameter(self):
        """A slice parameter is passed via two registers (its own
        ptr, then len) directly, per the SysV ABI's own rule for a
        16-byte, two-integer-eightbyte struct -- not through a stack
        slot the way an ordinary scalar parameter's copy-on-entry
        works, and not copied the way an array parameter is (a slice
        parameter is just an alias, exactly like any other slice
        variable)."""
        assert_program_exit_code(
            "def int first([]int s):\n"
            "    return s[0]\n"
            "\n"
            "def int main():\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    []int s = arr[1:4]\n"
            "    return first(s)\n",
            2,
        )

    def test_slice_return(self):
        """A slice return value comes back in %rax:%rdx directly --
        the SysV ABI's own convention for a small, all-integer-
        eightbyte struct return -- not through the hidden-pointer
        mechanism arrays use."""
        assert_program_exit_code(
            "def []int make():\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    return arr[1:4]\n"
            "\n"
            "def int main():\n"
            "    []int s = make()\n"
            "    return s[0]\n",
            2,
        )


class TestSliceBoundsChecking:
    """Slice bounds get their own message ("slice bounds out of
    range", distinct from ordinary indexing's "array index out of
    bounds") and their own comparison: `ja` (strictly "above"), not
    `jae` -- unlike an ordinary index, where being equal to the
    array's own size is already invalid, a slice's low/high are both
    allowed to equal the base's length (`arr[5:5]` is a valid, empty-
    slice-producing expression). test_low_equals_high_equals_length_
    is_valid is the positive control proving that boundary is exactly
    right, not off by one in either direction.
    """
    pytestmark = GCC_SKIP

    def test_index_into_slice_out_of_bounds_aborts(self):
        assert_crashes_with_sigabrt(
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    []int s = arr[1:4]\n"
            "    int i = 10\n"
            "    return s[i]"
        )

    def test_low_greater_than_high_aborts(self):
        assert_crashes_with_sigabrt(
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    int lo = 3\n"
            "    int hi = 1\n"
            "    []int s = arr[lo:hi]\n"
            "    return s[0]"
        )

    def test_high_greater_than_length_aborts(self):
        assert_crashes_with_sigabrt(
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    int hi = 10\n"
            "    []int s = arr[0:hi]\n"
            "    return s[0]"
        )

    def test_negative_low_aborts(self):
        assert_crashes_with_sigabrt(
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    int lo = 0 - 1\n"
            "    []int s = arr[lo:3]\n"
            "    return s[0]"
        )

    def test_low_equals_high_equals_length_is_valid(self):
        """The positive control: `arr[5:5]` on a 5-element array must
        NOT abort -- it's a valid expression producing an empty
        slice -- proof the `ja` (not `jae`) choice is exactly right."""
        assert_exit_code(
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    []int s = arr[5:5]\n"
            "    return 42",
            42,
        )

    def test_slice_bounds_panic_message(self):
        """Regression-style check that the slice-specific message is
        actually the one printed, not the ordinary indexing one --
        confirming the bounds-check panic infrastructure's
        generalization to multiple, distinct messages (see
        codegen.py's _get_bounds_check_fail_label) works correctly."""
        result = compile_and_run(
            "def int main():\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    int hi = 10\n"
            "    []int s = arr[0:hi]\n"
            "    return s[0]\n"
        )
        assert result.returncode == -signal.SIGABRT
        assert "slice bounds out of range" in result.stdout


# ---------------------------------------------------------------------------
# Slice parameters and return values: a slice crosses a function
# boundary via TWO registers directly -- its own ptr, then len -- not
# through a hidden pointer the way an array's return does, and not
# copied on entry the way an array parameter is. This matches exactly
# what a real C compiler does for an equivalent `struct{void*,long}`
# passed or returned by value under the SysV ABI: as a PARAMETER, it
# consumes two consecutive integer argument registers; as a RETURN
# VALUE, it comes back in %rax:%rdx (first eightbyte in %rax, second
# in %rdx).
#
# test_slice_interleaved_with_scalar_parameters and
# test_two_slices_and_two_scalars_are_exactly_six_slots are the tests
# that actually prove the trickiest part of this feature holds: since
# a slice now costs 2 of the 6 available argument-register slots
# instead of 1, the mapping from argument/parameter INDEX to register
# INDEX is no longer 1:1 -- both the caller side
# (_gen_call_arguments_into) and the callee side (gen_function's own
# parameter loop) track a running slot count instead, and these tests
# are what confirm a slice's own two slots land correctly among
# ordinary scalar ones on both sides, not just when a slice happens to
# be the only or the last parameter.
#
# test_exactly_six_slots_from_three_slice_parameters and
# test_seven_slots_from_three_slices_and_a_scalar_is_rejected are the
# positive/negative pair proving the boundary itself is exactly
# right: 6 slots must be accepted, 7 must be cleanly rejected, not
# silently truncated or off by one in either direction.
#
# test_writing_through_a_slice_parameter_mutates_callers_array is the
# test that actually proves a slice parameter is a genuine alias
# crossing the function boundary, not a copy -- matching the same
# aliasing guarantee slices already have within a single function.
# test_forwarding_a_slice_returning_calls_result is the free case the
# %rax:%rdx convention was specifically chosen for: gen_slice_call_
# into already leaves a callee's own result exactly where a caller
# needs to leave its own, so `return bar()` (bar also returning a
# slice) costs nothing beyond the call itself.
# ---------------------------------------------------------------------------

class TestSliceParametersAndReturns:
    pytestmark = GCC_SKIP

    def test_multiple_slice_parameters(self):
        assert_program_exit_code(
            "def int sum_two([]int a, []int b):\n"
            "    return a[0] + b[0]\n"
            "\n"
            "def int main():\n"
            "    [3]int x = [10, 20, 30]\n"
            "    [3]int y = [1, 2, 3]\n"
            "    []int sx = x[0:3]\n"
            "    []int sy = y[0:3]\n"
            "    return sum_two(sx, sy)\n",
            11,
        )

    def test_slice_interleaved_with_scalar_parameters(self):
        """The test that actually proves the register-slot accounting
        holds, not just that a slice CAN be a parameter: a slice
        between two scalars needs its own two slots to land in the
        right registers without disturbing either scalar's own slot."""
        assert_program_exit_code(
            "def int f(int a, []int s, int b):\n"
            "    return a + s[0] + s[1] + b\n"
            "\n"
            "def int main():\n"
            "    [3]int arr = [10, 20, 30]\n"
            "    []int s = arr[0:3]\n"
            "    return f(1, s, 2)\n",
            33,
        )

    def test_writing_through_a_slice_parameter_mutates_callers_array(self):
        """Proves a slice parameter is a genuine alias crossing the
        function boundary, not a copy -- the same aliasing guarantee
        slices already have within a single function, now verified to
        survive a call."""
        assert_program_exit_code(
            "def mutate([]int s):\n"
            "    s[0] = 42\n"
            "\n"
            "def int main():\n"
            "    [3]int arr = [1, 2, 3]\n"
            "    []int s = arr[0:3]\n"
            "    mutate(s)\n"
            "    return arr[0]\n",
            42,
        )

    def test_recursive_function_with_a_slice_parameter(self):
        assert_program_exit_code(
            "def int sum_slice([]int s, int i):\n"
            "    if i >= 5:\n"
            "        return 0\n"
            "    return s[i] + sum_slice(s, i + 1)\n"
            "\n"
            "def int main():\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    []int s = arr[0:5]\n"
            "    return sum_slice(s, 0)\n",
            15,
        )

    def test_forwarding_a_slice_returning_calls_result(self):
        """The free case the %rax:%rdx convention was specifically
        chosen for: gen_slice_call_into already leaves a callee's own
        result exactly where a caller needs to leave its own, so this
        costs nothing beyond the call itself -- no intermediate copy,
        unlike an array-returning function's own hidden-pointer
        forwarding, which still needs the SAME address threaded one
        level deeper (though also without a copy)."""
        assert_program_exit_code(
            "def []int inner():\n"
            "    [3]int arr = [7, 8, 9]\n"
            "    return arr[0:3]\n"
            "\n"
            "def []int outer():\n"
            "    return inner()\n"
            "\n"
            "def int main():\n"
            "    []int s = outer()\n"
            "    return s[0] + s[1] + s[2]\n",
            24,
        )

    def test_printing_a_slice_parameter(self):
        assert_program_stdout(
            "def show([]int s):\n"
            "    print(s)\n"
            "\n"
            "def int main():\n"
            "    [3]int arr = [4, 5, 6]\n"
            "    []int s = arr[0:3]\n"
            "    show(s)\n"
            "    return 0\n",
            "[]int[4, 5, 6]\n",
        )

    def test_reslicing_a_received_slice_parameter(self):
        assert_program_exit_code(
            "def int second_half([]int s):\n"
            "    []int half = s[2:4]\n"
            "    return half[0] + half[1]\n"
            "\n"
            "def int main():\n"
            "    [4]int arr = [10, 20, 30, 40]\n"
            "    []int s = arr[0:4]\n"
            "    return second_half(s)\n",
            70,
        )

    def test_returning_none_from_a_slice_returning_function(self):
        assert_program_exit_code(
            "def []int maybe(bool give):\n"
            "    if give:\n"
            "        [2]int arr = [1, 2]\n"
            "        return arr[0:2]\n"
            "    return none\n"
            "\n"
            "def bool main():\n"
            "    []int s = maybe(false)\n"
            "    return s == none\n",
            1,
        )

    def test_array_parameter_and_slice_parameter_together(self):
        assert_program_exit_code(
            "def int combo([3]int arr, []int s):\n"
            "    return arr[0] + s[0]\n"
            "\n"
            "def int main():\n"
            "    [3]int a = [100, 200, 300]\n"
            "    [2]int b = [5, 6]\n"
            "    []int s = b[0:2]\n"
            "    return combo(a, s)\n",
            105,
        )

    def test_slice_parameter_with_heap_allocated_array_parameter(self):
        """A slice parameter's own register-based passing has nothing
        to do with an array parameter's own copy-on-entry mechanism
        (heap-backed here, since 5000 ints exceeds the stack-array
        threshold) -- this confirms the two coexist correctly in the
        same call, each going through its own, independent path."""
        assert_program_exit_code(
            "def int combo([5000]int big, []int s):\n"
            "    return big[0] + big[4999] + s[0]\n"
            "\n"
            "def int main():\n"
            "    [5000]int huge\n"
            "    huge[0] = 1\n"
            "    huge[4999] = 2\n"
            "    [1]int small = [100]\n"
            "    []int s = small[0:1]\n"
            "    return combo(huge, s)\n",
            103,
        )

    def test_mix_of_real_slice_and_none_arguments(self):
        assert_program_exit_code(
            "def bool f([]int a, []int b):\n"
            "    return a != none and b == none\n"
            "\n"
            "def bool main():\n"
            "    [3]int arr = [1, 2, 3]\n"
            "    []int s = arr[0:3]\n"
            "    return f(s, none)\n",
            1,
        )

    def test_exactly_six_slots_from_three_slice_parameters(self):
        """The positive half of the boundary pair: three slice
        parameters alone need exactly 6 register slots -- the limit
        itself -- and must be accepted, not rejected off by one."""
        assert_program_exit_code(
            "def int f([]int a, []int b, []int c):\n"
            "    return a[0] + b[0] + c[0]\n"
            "\n"
            "def int main():\n"
            "    [1]int x = [1]\n"
            "    [1]int y = [2]\n"
            "    [1]int z = [3]\n"
            "    []int sx = x[0:1]\n"
            "    []int sy = y[0:1]\n"
            "    []int sz = z[0:1]\n"
            "    return f(sx, sy, sz)\n",
            6,
        )

    def test_seven_slots_from_three_slices_and_a_scalar_is_rejected(self):
        """The negative half of the boundary pair: one more scalar
        parameter pushes the same three slices over the 6-slot limit,
        and must be cleanly rejected -- not silently truncated."""
        source = (
            "def int f([]int a, []int b, []int c, int d):\n"
            "    return a[0] + b[0] + c[0] + d\n"
            "\n"
            "def int main():\n"
            "    [1]int x = [1]\n"
            "    [1]int y = [2]\n"
            "    [1]int z = [3]\n"
            "    []int sx = x[0:1]\n"
            "    []int sy = y[0:1]\n"
            "    []int sz = z[0:1]\n"
            "    return f(sx, sy, sz, 4)\n"
        )
        ast = _parse(source)
        analyze(ast)  # semantically fine -- the limit is codegen-level only
        with pytest.raises(CodegenError, match="needs 7 argument register"):
            generate_asm(ast, platform=ASM_PLATFORM)

    def test_two_slices_and_two_scalars_are_exactly_six_slots(self):
        assert_program_exit_code(
            "def int f(int a, []int s1, int b, []int s2):\n"
            "    return a + s1[0] + b + s2[0]\n"
            "\n"
            "def int main():\n"
            "    [1]int x = [10]\n"
            "    [1]int y = [20]\n"
            "    []int sx = x[0:1]\n"
            "    []int sy = y[0:1]\n"
            "    return f(1, sx, 2, sy)\n",
            33,
        )

    def test_slice_argument_must_be_a_variable_or_none(self):
        """The same restriction slice bases have everywhere else in
        this codebase (indexing, print, re-slicing): a bare Slice
        expression has no pre-existing descriptor to read at a call
        site -- assign it to a named variable first."""
        source = (
            "def int f([]int s):\n"
            "    return s[0]\n"
            "\n"
            "def int main():\n"
            "    [3]int arr = [1, 2, 3]\n"
            "    return f(arr[0:3])\n"
        )
        ast = _parse(source)
        analyze(ast)
        with pytest.raises(CodegenError, match="assign it to a variable first"):
            generate_asm(ast, platform=ASM_PLATFORM)


# ---------------------------------------------------------------------------
# Printing arrays and slices: `TYPE[elem, elem, ...]` -- e.g.
# `[3]int[1, 2, 3]` or `[]int[1, 2, 3]` -- the type prefix (matching
# semantic.Type.__str__ exactly, so no new formatting logic was needed
# for it) appearing exactly once, at the outermost level, never
# repeated for a nested row. A str element is quoted inside a
# collection (`'alice'`) even though a bare str argument to print
# still prints unquoted -- the two behave differently on purpose, not
# by oversight.
#
# test_nested_2d_array_prefix_appears_once_not_per_row is the test
# that actually proves the headline formatting decision holds, not
# just the one-dimensional case both of the original examples showed:
# a [2][3]int prints as `[2][3]int[[1, 2, 3], [4, 5, 6]]`, not with
# "[3]int" repeated on each inner row.
#
# Built as a sequence of direct printf calls -- one piece at a time
# (the type prefix, each bracket, each separator, each element) --
# rather than materializing one big string via malloc and printing it
# in one shot, which would have needed a new int-to-string conversion
# step this language has no other reason to have (see codegen.py's
# PRINTING ARRAYS AND SLICES section). Since an array's length is
# known at compile time but a slice's is only known at runtime, this
# uses ONE uniform runtime loop for both rather than unrolling arrays
# separately -- test_printing_a_slice_of_a_slice exercises the harder,
# runtime-length path directly, and
# test_multiple_prints_each_get_exactly_one_newline confirms the loop
# never leaks an extra or missing newline across separate print calls.
# ---------------------------------------------------------------------------

class TestPrintArraysAndSlices:
    pytestmark = GCC_SKIP

    def test_print_array_of_int(self):
        assert_stdout(
            "    [3]int arr = [1, 2, 3]\n"
            "    print(arr)\n"
            "    return 0",
            "[3]int[1, 2, 3]\n",
        )

    def test_print_slice_of_int(self):
        assert_stdout(
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    []int s = arr[1:4]\n"
            "    print(s)\n"
            "    return 0",
            "[]int[2, 3, 4]\n",
        )

    def test_nested_2d_array_prefix_appears_once_not_per_row(self):
        """The test that actually proves the headline formatting
        decision: the type prefix appears exactly once, at the
        outermost level -- NOT repeated as "[3]int" on each inner row."""
        assert_stdout(
            "    [2][3]int matrix = [[1, 2, 3], [4, 5, 6]]\n"
            "    print(matrix)\n"
            "    return 0",
            "[2][3]int[[1, 2, 3], [4, 5, 6]]\n",
        )

    def test_str_elements_are_quoted(self):
        """A str element is quoted inside a collection even though a
        bare str argument to print prints unquoted -- the two are
        deliberately different conventions, not an inconsistency."""
        assert_stdout(
            "    [3]str names = ['alice', 'bob', 'carol']\n"
            "    print(names)\n"
            "    return 0",
            "[3]str['alice', 'bob', 'carol']\n",
        )

    def test_bool_elements(self):
        assert_stdout(
            "    [3]bool flags = [true, false, true]\n"
            "    print(flags)\n"
            "    return 0",
            "[3]bool[true, false, true]\n",
        )

    def test_empty_slice_prints_with_no_trailing_comma(self):
        """`arr[5:5]` is a valid, empty-slice-producing expression
        (see TestSliceBoundsChecking's own positive control) -- this
        confirms printing one produces `[]` cleanly, not a trailing
        comma or an error."""
        assert_stdout(
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    []int s = arr[5:5]\n"
            "    print(s)\n"
            "    return 0",
            "[]int[]\n",
        )

    def test_slice_of_2d_array_outer_dimension(self):
        assert_stdout(
            "    [3][3]int matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\n"
            "    [][3]int rows = matrix[0:2]\n"
            "    print(rows)\n"
            "    return 0",
            "[][3]int[[1, 2, 3], [4, 5, 6]]\n",
        )

    def test_printing_a_slice_of_a_slice(self):
        """Exercises the harder of gen_indexable_base_into's two code
        paths directly: the base's own length is a RUNTIME value read
        out of its descriptor, not a compile-time constant the way an
        array base's is."""
        assert_stdout(
            "    [6]int arr = [1, 2, 3, 4, 5, 6]\n"
            "    []int s = arr[1:5]\n"
            "    []int s2 = s[1:3]\n"
            "    print(s2)\n"
            "    return 0",
            "[]int[3, 4]\n",
        )

    def test_multiple_prints_each_get_exactly_one_newline(self):
        assert_stdout(
            "    [2]int a = [1, 2]\n"
            "    [2]int b = [3, 4]\n"
            "    print(a)\n"
            "    print(b)\n"
            "    return 0",
            "[2]int[1, 2]\n[2]int[3, 4]\n",
        )

    def test_array_literal_as_direct_print_argument_not_supported(self):
        """A real, deliberate gap, matching the same restriction
        gen_array_arg_address_into already imposes on array-typed call
        arguments: a bare ArrayLiteral has no address of its own to
        print through. Assign it to a named variable first."""
        source = (
            "def int main():\n"
            "    print([1, 2, 3])\n"
            "    return 0\n"
        )
        ast = _parse(source)
        analyze(ast)  # semantically fine -- the gap is codegen-level only
        with pytest.raises(CodegenError, match="assign it to a variable first"):
            generate_asm(ast, platform=ASM_PLATFORM)

    def test_slice_expression_as_direct_print_argument_not_supported(self):
        source = (
            "def int main():\n"
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    print(arr[1:3])\n"
            "    return 0\n"
        )
        ast = _parse(source)
        analyze(ast)
        with pytest.raises(CodegenError, match="assign it to a variable first"):
            generate_asm(ast, platform=ASM_PLATFORM)


# ---------------------------------------------------------------------------
# `none`: Hornet's nil-style zero value, analogous to Go's own `nil`, but
# deliberately narrower internally -- see NoneLiteral's own docstring in
# parser.py for why this doesn't need a general untyped-constant
# mechanism (which this language has no other reason to have) to work
# for everything usable today. Only slices are nilable so far.
#
# test_real_empty_slice_is_not_equal_to_none is the test that actually
# proves the subtlest, easiest-to-get-wrong part of this feature: a
# real, zero-length slice sliced from a real array (`arr[5:5]`) is NOT
# `== none`, even though it's equally safe and equally zero-length for
# every other purpose (indexing, printing, re-slicing) as a genuinely
# nil one -- matching Go's own well-known nil-vs-empty-slice
# distinction. Getting this wrong in either direction (checking length
# instead of the pointer, or checking both) would silently conflate two
# states Go -- and this test -- deliberately keeps apart.
#
# test_indexing_a_none_valued_slice_aborts and
# test_printing_a_none_valued_slice confirm the OTHER half of the
# design: a none-valued slice's {0, 0} descriptor needs no new
# mechanism at all for indexing or printing, since both already handle
# an ordinary zero-length slice correctly (see TestSliceBoundsChecking's
# own `arr[5:5]` positive control) -- gen_none_into only had to produce
# that descriptor, not teach every existing slice operation a new case.
# ---------------------------------------------------------------------------

class TestNone:
    pytestmark = GCC_SKIP

    def test_slice_vardecl_with_none(self):
        assert_stdout(
            "    []int s = none\n"
            "    print(s)\n"
            "    return 0",
            "[]int[]\n",
        )

    def test_slice_assign_with_none(self):
        assert_stdout(
            "    [3]int arr = [1, 2, 3]\n"
            "    []int s = arr[0:3]\n"
            "    s = none\n"
            "    print(s)\n"
            "    return 0",
            "[]int[]\n",
        )

    def test_none_valued_slice_equals_none(self):
        assert_exit_code(
            "    []int s = none\n"
            "    return s == none",
            1,
            return_type="bool",
        )

    def test_real_empty_slice_is_not_equal_to_none(self):
        """The test that actually proves the subtlest part of this
        feature holds: `arr[5:5]` is a real, zero-length slice with a
        non-null pointer -- equally safe and equally zero-length as a
        genuinely nil slice for every other purpose, but NOT `==
        none`, matching Go's own nil-vs-empty-slice distinction."""
        assert_exit_code(
            "    [5]int arr = [1, 2, 3, 4, 5]\n"
            "    []int s = arr[5:5]\n"
            "    return s == none",
            0,
            return_type="bool",
        )

    def test_real_nonempty_slice_is_not_equal_to_none(self):
        assert_exit_code(
            "    [3]int arr = [1, 2, 3]\n"
            "    []int s = arr[0:3]\n"
            "    return s == none",
            0,
            return_type="bool",
        )

    def test_none_on_the_left_side(self):
        assert_exit_code(
            "    []int s = none\n"
            "    return none == s",
            1,
            return_type="bool",
        )

    def test_not_equal_with_none(self):
        assert_exit_code(
            "    [3]int arr = [1, 2, 3]\n"
            "    []int s = arr[0:3]\n"
            "    return s != none",
            1,
            return_type="bool",
        )

    def test_indexing_a_none_valued_slice_aborts(self):
        """A none-valued slice's length is 0, so this hits the exact
        same bounds check (and the exact same "array index out of
        bounds" message) as indexing into any other empty slice --
        no none-specific codegen needed for this at all."""
        assert_crashes_with_sigabrt(
            "    []int s = none\n"
            "    return s[0]"
        )

    def test_printing_a_none_valued_slice(self):
        assert_stdout(
            "    []int s = none\n"
            "    print(s)\n"
            "    return 0",
            "[]int[]\n",
        )

    def test_reslicing_a_none_valued_slice_at_zero_zero(self):
        """`s[0:0]` on a none-valued slice produces ANOTHER
        none-equal slice ({0,0} + 0*stride = {0,0}) -- confirming
        gen_slice_into's existing machinery handles a none-valued base
        correctly with no special-casing, the same way indexing and
        printing already do."""
        assert_exit_code(
            "    []int s = none\n"
            "    []int s2 = s[0:0]\n"
            "    return s2 == none",
            1,
            return_type="bool",
        )

    def test_int_vardecl_with_none_is_rejected(self):
        assert_semantic_error(
            "    int x = none\n"
            "    return 0",
            match="Cannot initialize",
        )

    def test_str_vardecl_with_none_is_rejected(self):
        assert_semantic_error(
            "    str x = none\n"
            "    return 0",
            match="Cannot initialize",
        )

    def test_array_vardecl_with_none_is_rejected(self):
        assert_semantic_error(
            "    [3]int arr = none\n"
            "    return 0",
            match="Cannot initialize",
        )

    def test_print_bare_none_is_rejected(self):
        assert_semantic_error(
            "    print(none)\n"
            "    return 0",
            match="cannot be called with a bare 'none'",
        )

    def test_comparing_none_to_none_is_rejected(self):
        """`none == none` would otherwise trivially type-check --
        Type.NONE equals itself the same way any type does -- so this
        needed its own explicit exclusion, not just the slice-vs-none
        exception (see check_binary's own comment)."""
        assert_semantic_error(
            "    return none == none",
            match="does not support array, slice, void, or none operands",
            return_type="bool",
        )

    def test_comparing_int_to_none_is_rejected(self):
        assert_semantic_error(
            "    return 5 == none",
            match="does not support array, slice, void, or none operands",
            return_type="bool",
        )

    def test_none_as_a_slice_argument(self):
        """Now that slice parameters are supported, `none` passed as
        a slice-typed argument works correctly -- the callee receives
        a genuinely nil slice ({ptr: 0, len: 0}), so indexing into it
        aborts exactly like indexing into any other zero-length slice
        would (see TestSliceBoundsChecking)."""
        result = compile_and_run(
            "def int first([]int s):\n"
            "    return s[0]\n"
            "\n"
            "def int main():\n"
            "    return first(none)\n"
        )
        assert result.returncode == -signal.SIGABRT


# ---------------------------------------------------------------------------
# Semantic analysis: scope/declaration checking and the strict int/bool
# type system. These never reach codegen -- each one asserts that
# analyze() itself raises SemanticError -- so they don't need gcc and
# aren't skipped when it's missing.
# ---------------------------------------------------------------------------

class TestSemanticErrors:

    # -- scope / declaration errors --------------------------------------

    def test_reference_to_undeclared_variable(self):
        assert_semantic_error(
            "    return a",
            match="undeclared variable",
        )

    def test_assignment_to_undeclared_variable(self):
        assert_semantic_error(
            "    a = 1\n"
            "    return 0",
            match="undeclared variable",
        )

    def test_double_declaration(self):
        assert_semantic_error(
            "    int a = 1\n"
            "    int a = 2\n"
            "    return a",
            match="already declared",
        )

    def test_declare_before_use_is_enforced_in_textual_order(self):
        """A variable assigned above its own declaration is, from the
        analyzer's point of view, simply not in scope yet -- see
        semantic.py's module docstring for why that falls out of
        walking statements in program order."""
        assert_semantic_error(
            "    a = 1\n"
            "    int a\n"
            "    return a",
            match="undeclared variable",
        )

    def test_self_referential_initializer(self):
        """`int a = a` -- the right-hand `a` is checked before the new
        `a` is added to scope, so this is indistinguishable from any
        other undeclared reference."""
        assert_semantic_error(
            "    int a = a\n"
            "    return a",
            match="undeclared variable",
        )

    def test_valid_program_does_not_raise(self):
        """Sanity check on the harness itself: a well-formed program
        must NOT raise, so the tests above are actually testing
        something specific rather than everything just failing."""
        ast = _parse("def int main():\n    int a = 1\n    return a\n")
        analyze(ast)  # should not raise

    # -- initialization / assignment / return type mismatches -----------

    def test_initializer_type_mismatch(self):
        assert_semantic_error(
            "    int a = true\n"
            "    return a",
            match="Cannot initialize",
        )

    def test_assignment_type_mismatch(self):
        assert_semantic_error(
            "    bool a = true\n"
            "    a = 1\n"
            "    return a",
            return_type="bool",
            match="Cannot assign",
        )

    def test_return_type_mismatch_bool_where_int_expected(self):
        assert_semantic_error(
            "    return 3 < 5",
            return_type="int",
            match="declared to return",
        )

    def test_return_type_mismatch_int_where_bool_expected(self):
        assert_semantic_error(
            "    return 1",
            return_type="bool",
            match="declared to return",
        )

    # -- operator operand-type errors ------------------------------------

    def test_not_requires_bool_not_int(self):
        assert_semantic_error(
            "    return not 0",
            return_type="bool",
            match="requires a bool operand",
        )

    def test_negate_requires_int_not_bool(self):
        assert_semantic_error(
            "    return -true",
            match="requires an int operand",
        )

    def test_complement_requires_int_not_bool(self):
        assert_semantic_error(
            "    return ~true",
            match="requires an int operand",
        )

    def test_arithmetic_requires_int_operands(self):
        """- * / are int-only, unaffected by ADD's overload -- '+' gets
        its own dedicated test below, since mixing bool into it hits a
        different, ADD-specific error message."""
        assert_semantic_error(
            "    return true - false",
            match="requires int operands",
        )

    def test_add_requires_two_int_or_two_str_operands(self):
        """'+' is overloaded (int+int is arithmetic, str+str is
        concatenation -- see semantic.py's check_binary), so it doesn't
        go through the generic _require_type path the other arithmetic
        operators use, and gets its own distinct error message."""
        assert_semantic_error(
            "    return true + false",
            match="requires two int operands or two str operands",
        )

    def test_ordering_comparison_requires_int_operands(self):
        """< > <= >= don't accept bool -- there's no inherent ordering
        on bool in this language."""
        assert_semantic_error(
            "    return true < false",
            return_type="bool",
            match="requires int operands",
        )

    def test_chained_ordering_comparison_is_rejected(self):
        """`1 < 2 < 3` parses left-associatively as `(1 < 2) < 3` (see
        parser.py) -- and `1 < 2` produces bool, which `<` doesn't
        accept. Under implicit-conversion languages this silently means
        something most people don't intend (`(1<2)<3` happens to
        evaluate `1<3`); strict typing turns it into a hard error
        instead of a footgun."""
        assert_semantic_error(
            "    return 1 < 2 < 3",
            return_type="bool",
            match="requires int operands",
        )

    def test_logical_and_requires_bool_operands(self):
        assert_semantic_error(
            "    return 1 and 0",
            return_type="bool",
            match="requires bool operands",
        )

    def test_logical_or_requires_bool_operands(self):
        assert_semantic_error(
            "    return 1 or 0",
            return_type="bool",
            match="requires bool operands",
        )

    def test_equality_cannot_compare_int_to_bool(self):
        assert_semantic_error(
            "    return 1 == true",
            return_type="bool",
            match="Cannot compare",
        )

    def test_equality_same_type_is_valid(self):
        """The positive control for the test above: comparing two
        values of the *same* type must NOT raise."""
        ast = _parse("def bool main():\n    return 1 == 1\n")
        analyze(ast)  # should not raise
        ast = _parse("def bool main():\n    return true == false\n")
        analyze(ast)  # should not raise

    # -- literal type errors ---------------------------------------------

    def test_float_literal_is_rejected(self):
        """There's no float type yet -- only int and bool -- even
        though the lexer's NUMBER rule matches decimals."""
        assert_semantic_error(
            "    return 2.5",
            match="not a whole number",
        )

    # -- if/elif/else: condition typing and block scoping ----------------

    def test_if_condition_must_be_bool(self):
        """No int-as-truthy shortcut -- same rule as everywhere else in
        this type system."""
        assert_semantic_error(
            "    if 1:\n"
            "        return 1\n"
            "    return 0",
            match="'if' condition must be bool",
        )

    def test_elif_condition_must_be_bool(self):
        """The condition check applies to every elif in a chain, not
        just the first `if` -- since an elif is just a nested If (see
        parser.py's If docstring), this falls out of analyze_if being
        called recursively rather than needing separate elif logic."""
        assert_semantic_error(
            "    if false:\n"
            "        return 1\n"
            "    elif 1:\n"
            "        return 2\n"
            "    return 0",
            match="'if' condition must be bool",
        )

    def test_variable_declared_in_if_does_not_leak_outside(self):
        """A variable declared inside an if-block goes out of scope
        once the block ends -- referencing it afterward is exactly as
        invalid as referencing any other undeclared name."""
        assert_semantic_error(
            "    if true:\n"
            "        int a = 1\n"
            "    return a",
            match="undeclared variable",
        )

    def test_variable_declared_in_then_not_visible_in_else(self):
        """then and else get independent scopes -- a name from one
        branch isn't visible in the other, since they're mutually
        exclusive at runtime."""
        assert_semantic_error(
            "    if true:\n"
            "        int a = 1\n"
            "    else:\n"
            "        return a",
            match="undeclared variable",
        )

    def test_same_name_in_sibling_branches_is_allowed(self):
        """The positive control for the two tests above: declaring `a`
        independently in both an if and its else must NOT raise, since
        they're separate scopes -- this is exactly the scenario that
        motivated rewriting codegen's allocator (see codegen.py's LOCAL
        VARIABLES section)."""
        ast = _parse(
            "def int main():\n"
            "    if true:\n"
            "        int a = 1\n"
            "        return a\n"
            "    else:\n"
            "        int a = 2\n"
            "        return a\n"
        )
        analyze(ast)  # should not raise

    def test_shadowing_outer_variable_in_if_is_allowed(self):
        """A block-local declaration is allowed to shadow a
        same-named variable from an enclosing scope."""
        ast = _parse(
            "def int main():\n"
            "    int a = 1\n"
            "    if true:\n"
            "        int a = 2\n"
            "        return a\n"
            "    return a\n"
        )
        analyze(ast)  # should not raise

    def test_double_declaration_within_same_if_branch_is_rejected(self):
        """Shadowing an *enclosing* scope is fine, but the ordinary
        double-declaration rule still applies *within* a single
        branch's own scope."""
        assert_semantic_error(
            "    if true:\n"
            "        int a = 1\n"
            "        int a = 2\n"
            "        return a\n"
            "    return 0",
            match="already declared",
        )

    # -- while/break/continue ---------------------------------------------

    def test_while_condition_must_be_bool(self):
        assert_semantic_error(
            "    while 1:\n"
            "        return 1\n"
            "    return 0",
            match="'while' condition must be bool",
        )

    def test_break_outside_loop_is_rejected(self):
        assert_semantic_error(
            "    break\n"
            "    return 0",
            match="'break' outside of a loop",
        )

    def test_continue_outside_loop_is_rejected(self):
        assert_semantic_error(
            "    continue\n"
            "    return 0",
            match="'continue' outside of a loop",
        )

    def test_break_inside_if_inside_while_is_allowed(self):
        """loop_depth (see semantic.py's LOOPS section) has to survive
        being nested inside a non-loop block -- an `if` between the
        `break` and its enclosing `while` shouldn't matter."""
        ast = _parse(
            "def int main():\n"
            "    while true:\n"
            "        if true:\n"
            "            break\n"
            "    return 0\n"
        )
        analyze(ast)  # should not raise

    def test_break_inside_if_not_inside_while_is_rejected(self):
        """The negative control for the test above: an `if` on its own,
        with no enclosing `while` at all, still correctly rejects a
        `break` inside it."""
        assert_semantic_error(
            "    if true:\n"
            "        break\n"
            "    return 0",
            match="'break' outside of a loop",
        )

    def test_break_after_loop_ends_is_rejected(self):
        """Proves loop_depth is correctly *decremented* once a while's
        body finishes being analyzed -- a break textually after the
        loop, at the same level, must not be treated as still being
        inside it."""
        assert_semantic_error(
            "    while true:\n"
            "        return 1\n"
            "    break\n"
            "    return 0",
            match="'break' outside of a loop",
        )

    def test_variable_declared_in_while_does_not_leak_outside(self):
        assert_semantic_error(
            "    while true:\n"
            "        int a = 1\n"
            "    return a",
            match="undeclared variable",
        )

    def test_break_in_outer_loop_after_inner_loop_ends_is_allowed(self):
        """The positive control matching the nested-loop codegen tests
        in TestWhileLoops: a break in the *outer* loop, positioned
        after an inner loop's body has already been fully analyzed and
        its scope popped, must still correctly resolve as being inside
        the outer loop (loop_depth is a counter, not reset to 0 by the
        inner loop's own pop)."""
        ast = _parse(
            "def int main():\n"
            "    while true:\n"
            "        int j = 0\n"
            "        while j < 3:\n"
            "            j = j + 1\n"
            "        break\n"
            "    return 0\n"
        )
        analyze(ast)  # should not raise

    # -- str -----------------------------------------------------------

    def test_add_rejects_mixed_int_and_str(self):
        assert_semantic_error(
            "    str a = 'hello'\n"
            "    int b = 5\n"
            "    return a + b",
            return_type="str",
            match="requires two int operands or two str operands",
        )

    def test_subtract_rejects_str_operands(self):
        """Only '+' is overloaded for str -- every other arithmetic
        operator stays strictly int-only."""
        assert_semantic_error(
            "    str a = 'hello'\n"
            "    str b = 'world'\n"
            "    return a - b",
            return_type="str",
            match="requires int operands",
        )

    def test_ordering_comparison_rejects_str_operands(self):
        """No inherent ordering on str in this language, same as bool
        -- only == and != are defined for it."""
        assert_semantic_error(
            "    str a = 'hello'\n"
            "    str b = 'world'\n"
            "    return a < b",
            return_type="bool",
            match="requires int operands",
        )

    def test_equality_rejects_str_compared_to_int(self):
        assert_semantic_error(
            "    str a = 'hello'\n"
            "    return a == 5",
            return_type="bool",
            match="Cannot compare",
        )

    def test_str_equality_same_type_is_valid(self):
        """The positive control: comparing two str values with == must
        NOT raise -- this is what makes string equality actually usable
        at all."""
        ast = _parse(
            "def bool main():\n"
            "    str a = 'hello'\n"
            "    str b = 'hello'\n"
            "    return a == b\n"
        )
        analyze(ast)  # should not raise

    def test_initializer_type_mismatch_int_into_str(self):
        assert_semantic_error(
            "    str a = 5\n"
            "    return 0",
            match="Cannot initialize",
        )

    def test_assignment_type_mismatch_str_into_int(self):
        assert_semantic_error(
            "    int a = 5\n"
            "    a = 'hello'\n"
            "    return a",
            match="Cannot assign",
        )

    def test_return_type_mismatch_str_where_int_expected(self):
        assert_semantic_error(
            "    return 'hello'",
            return_type="int",
            match="declared to return",
        )

    def test_concatenation_type_checks_as_valid_str(self):
        """The positive control for concatenation: `+` on two str
        operands must produce a value assignable to a str variable,
        with no error anywhere along the way."""
        ast = _parse(
            "def str main():\n"
            "    str a = 'hello'\n"
            "    str b = ' world'\n"
            "    str c = a + b\n"
            "    return c\n"
        )
        analyze(ast)  # should not raise

    # -- functions -------------------------------------------------------

    def test_call_to_undeclared_function(self):
        assert_semantic_error(
            "    return foo(1)",
            match="undeclared function",
        )

    def test_duplicate_function_name(self):
        assert_program_semantic_error(
            "def int foo():\n"
            "    return 1\n"
            "\n"
            "def int foo():\n"
            "    return 2\n",
            match="already declared",
        )

    def test_call_wrong_argument_count(self):
        assert_program_semantic_error(
            "def int add(int a, int b):\n"
            "    return a + b\n"
            "\n"
            "def int main():\n"
            "    return add(1)\n",
            match="expects 2 argument",
        )

    def test_call_wrong_argument_type(self):
        assert_program_semantic_error(
            "def int add(int a, int b):\n"
            "    return a + b\n"
            "\n"
            "def int main():\n"
            "    return add(1, 'hello')\n",
            match="should be int, got str",
        )

    def test_duplicate_parameter_name(self):
        """A parameter is declared into the function's own scope exactly
        like a local, so this is caught by the ordinary double-
        declaration check, not a function-specific one."""
        assert_program_semantic_error(
            "def int add(int a, int a):\n"
            "    return a\n",
            match="already declared",
        )

    def test_recursive_call_is_allowed(self):
        """The positive control: a function calling itself must NOT
        raise, since self.functions already has this function's own
        signature by the time its body is checked (see semantic.py's
        FUNCTIONS section)."""
        ast = _parse(
            "def int fact(int n):\n"
            "    if n == 0:\n"
            "        return 1\n"
            "    return n * fact(n - 1)\n"
        )
        analyze(ast)  # should not raise

    def test_mutual_recursion_with_forward_reference_is_allowed(self):
        """The positive control for forward references specifically:
        is_even calls is_odd, which is defined *after* it in the file --
        must not raise, since every signature is collected before any
        body is checked."""
        ast = _parse(
            "def bool is_even(int n):\n"
            "    if n == 0:\n"
            "        return true\n"
            "    return is_odd(n - 1)\n"
            "\n"
            "def bool is_odd(int n):\n"
            "    if n == 0:\n"
            "        return false\n"
            "    return is_even(n - 1)\n"
        )
        analyze(ast)  # should not raise

    # -- print (the first builtin) ----------------------------------------

    def test_print_wrong_argument_count_zero(self):
        assert_semantic_error(
            "    print()\n"
            "    return 0",
            match="expects exactly 1 argument",
        )

    def test_print_wrong_argument_count_multiple(self):
        assert_semantic_error(
            "    print(1, 2)\n"
            "    return 0",
            match="expects exactly 1 argument",
        )

    def test_print_argument_must_still_be_well_typed(self):
        """print accepts any *valid* type, but its argument still has to
        actually type-check on its own -- an undeclared variable inside
        it is still an error, same as anywhere else."""
        assert_semantic_error(
            "    print(undeclared_variable)\n"
            "    return 0",
            match="undeclared variable",
        )

    def test_redefining_print_is_rejected(self):
        assert_program_semantic_error(
            "def int print(int x):\n"
            "    return x\n"
            "\n"
            "def int main():\n"
            "    return 0\n",
            match="builtin",
        )

    def test_print_accepts_int_bool_and_str_without_raising(self):
        """The positive control: all three types must be individually
        acceptable to print, with no error for any of them."""
        for arg in ("5", "true", "'hello'"):
            ast = _parse(f"def int main():\n    print({arg})\n    return 0\n")
            analyze(ast)  # should not raise

    # -- modulo and the bitwise operators (% & | ^ << >>) -----------------

    def test_modulo_requires_int_operands(self):
        assert_semantic_error(
            "    return true % 2",
            match="requires int operands",
        )

    def test_bitwise_and_requires_int_operands(self):
        assert_semantic_error(
            "    return true & false",
            match="requires int operands",
        )

    def test_bitwise_or_requires_int_operands(self):
        assert_semantic_error(
            "    return 'x' | 1",
            match="requires int operands",
        )

    def test_bitwise_xor_requires_int_operands(self):
        assert_semantic_error(
            "    return true ^ true",
            match="requires int operands",
        )

    def test_shift_left_requires_int_operands(self):
        assert_semantic_error(
            "    return 'x' << 1",
            match="requires int operands",
        )

    def test_shift_right_requires_int_operands(self):
        assert_semantic_error(
            "    return true >> 1",
            match="requires int operands",
        )

    def test_bitwise_and_equality_precedence_is_a_type_error(self):
        """The C footgun, made real: `1 & 2 == 2` parses as
        `1 & (2 == 2)` (== binds tighter than &, per parser.py's
        _BINARY_OPS), so the right-hand side of & is bool, not int.
        In C this silently compiles into something almost nobody
        intends; here, strong typing turns it into a compile error
        instead."""
        assert_semantic_error(
            "    return 1 & 2 == 2",
            match="requires int operands",
        )

    def test_bitwise_and_equality_with_explicit_parens_is_valid(self):
        """The positive control for the test above: adding the
        parentheses C programmers usually need to remember here
        (`(1 & 2) == 2`) makes the grouping explicit and the program
        valid."""
        ast = _parse("def bool main():\n    return (1 & 2) == 2\n")
        analyze(ast)  # should not raise

    # -- arrays ------------------------------------------------------------

    def test_array_literal_size_mismatch(self):
        assert_semantic_error(
            "    [3]int arr = [1, 2]\n"
            "    return arr[0]",
            match="Cannot initialize",
        )

    def test_ragged_2d_array_literal_is_rejected(self):
        """A ragged literal like `[[1,2,3],[4,5]]` is rejected with no
        special-cased "ragged" logic at all -- once Type is
        structurally comparable, the two rows are just genuinely
        different types ([3]int vs [2]int), caught by the exact same
        check that rejects [3]int vs [3]bool."""
        assert_semantic_error(
            "    [2][3]int matrix = [[1, 2, 3], [4, 5]]\n"
            "    return matrix[0][0]",
            match="Array literal elements must all be the same type",
        )

    def test_heterogeneous_array_literal_is_rejected(self):
        assert_semantic_error(
            "    [3]int arr = [1, true, 3]\n"
            "    return arr[0]",
            match="Array literal elements must all be the same type",
        )

    def test_non_int_array_index_is_rejected(self):
        assert_semantic_error(
            "    [3]int arr = [1, 2, 3]\n"
            "    return arr[true]",
            match="Index must be int",
        )

    def test_indexing_a_non_array_value_is_rejected(self):
        assert_semantic_error(
            "    int x = 5\n"
            "    return x[0]",
            match="only arrays and slices support indexing",
        )

    def test_indexing_past_available_dimensions_is_rejected(self):
        assert_semantic_error(
            "    [2][3]int matrix = [[1, 2, 3], [4, 5, 6]]\n"
            "    return matrix[0][0][0]",
            match="only arrays and slices support indexing",
        )

    def test_wrong_element_type_in_index_assignment_is_rejected(self):
        assert_semantic_error(
            "    [3]int arr = [1, 2, 3]\n"
            "    arr[0] = true\n"
            "    return arr[0]",
            match="Cannot assign a value of type bool to an array element of type int",
        )

    def test_array_equality_comparison_is_rejected(self):
        """Not yet implemented, not a real type error -- explicitly
        excluded (see check_binary's equality handling) since
        codegen.py has no element-wise array-comparison logic, unlike
        str's real strcmp-backed one. Rejected even when both arrays
        are the exact same type, to avoid type-checking fine and then
        hitting an unhandled case in codegen."""
        assert_semantic_error(
            "    [3]int a = [1, 2, 3]\n"
            "    [3]int b = [1, 2, 3]\n"
            "    return a == b",
            match="does not support array, slice, void, or none operands",
            return_type="bool",
        )

    def test_array_as_function_param_and_return_type_checks_correctly(self):
        """The positive control: semantic.py fully accepts arrays as
        parameter and return types -- type_from_name and Type's
        structural equality handle this with no special-casing needed
        anywhere in this file. The gap is codegen-only (see
        TestArrays' test_array_parameter_not_supported_yet and
        test_array_return_not_supported_yet in the codegen-level
        suite) -- semantic analysis alone has no reason to reject
        this program."""
        ast = _parse(
            "def [3]int make_array(int a, int b, int c):\n"
            "    [3]int result = [a, b, c]\n"
            "    return result\n"
            "\n"
            "def int main():\n"
            "    [3]int r = make_array(1, 2, 3)\n"
            "    return r[0]\n"
        )
        analyze(ast)  # should not raise
