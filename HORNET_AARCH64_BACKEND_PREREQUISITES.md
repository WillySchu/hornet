# AArch64 Backend Prerequisites for Hornet

This checklist describes the architectural and tooling work I would complete **before implementing the AArch64/Apple Silicon backend**.

The goal is to make AArch64 a second lowering of the existing IR rather than another compiler-wide architectural effort.

---

## 1. Target Model

- [ ] Replace the current `platform = "linux" | "macos"` concept with an explicit target containing at least:
  - [ ] architecture (`x86_64`, `aarch64`)
  - [ ] OS (`linux`, `macos`)
  - [ ] ABI/calling convention
  - [ ] object format/linking conventions
- [ ] Make `x86_64-macos`, `x86_64-linux`, and future `aarch64-macos` distinct targets.
- [ ] Update `compile.py` and `build.py` to select a target rather than just a platform.
- [ ] Keep OS-specific behavior separate from architecture-specific behavior.
- [ ] Define how cross-compilation is represented, even if the first AArch64 target is only native Apple Silicon.

---

## 2. Remove Remaining ABI Assumptions from IR

This is the most important prerequisite.

- [ ] Replace `IRReadArgument(index)` with an operation representing a **source-level function parameter**, not a physical ABI argument slot.
- [ ] Remove the assumption that a hidden composite return pointer is "argument 0" in IR.
- [ ] Represent an indirect/composite return destination explicitly at the IR/function level.
- [ ] Define logical parameter classification independently of physical registers.
- [ ] Keep slice/string multi-word representations in IR as logical values rather than ABI register slots.
- [ ] Move decisions such as:
  - [ ] which registers carry arguments
  - [ ] where excess arguments go on the stack
  - [ ] where an indirect result pointer goes
  - [ ] how multiword values are passed
  from IR into the target ABI layer.

The current IR still describes `IRReadArgument` in terms of SysV ordering, while the builder reserves argument slot 0 for composite returns. Those assumptions should disappear before the IR is expected to cleanly support multiple targets.

---

## 3. Define a Target ABI Interface

Create one explicit abstraction for the native calling convention.

- [ ] Define integer/pointer argument registers.
- [ ] Define return registers.
- [ ] Define caller-saved registers.
- [ ] Define callee-saved registers.
- [ ] Define scratch registers.
- [ ] Define stack argument placement.
- [ ] Define required stack alignment.
- [ ] Define frame-pointer convention.
- [ ] Define indirect-result convention.
- [ ] Define treatment of multi-register values.
- [ ] Define the ABI used by `extern` functions.
- [ ] Define the ABI used by the Hornet runtime.
- [ ] Define the ABI for `main`.

Ideally, the same calling-convention abstraction should describe:

```text
Hornet → Hornet
Hornet → runtime
Hornet → FFI
```

with only the callee's linkage differing.

---

## 4. Parameterize Register Allocation

The linear-scan allocator itself should remain target-independent.

- [ ] Remove hard-coded x86 register pools from the allocator.
- [ ] Introduce a target-provided register set.
- [ ] Describe each register as:
  - [ ] allocatable/non-allocatable
  - [ ] caller-saved/callee-saved
  - [ ] usable as scratch
  - [ ] usable for particular instruction classes, if necessary
- [ ] Make "live across call" allocation use the target's callee-saved set.
- [ ] Ensure register naming is no longer interpreted by generic allocator code.
- [ ] Preserve the existing allocator algorithm; do not replace it just for AArch64.

A good endpoint would look conceptually like:

```python
allocator.allocate(
    intervals,
    register_set=target.registers,
    calling_convention=target.calling_convention,
)
```

rather than the allocator knowing that `r10d`, `r11d`, `r12d`, etc. exist.

---

## 5. Separate Machine-Specific Register Operations

The current backend has helpers for things such as 32-bit/64-bit/8-bit x86 register aliases. Those are inherently x86 concepts.

- [ ] Move register-width aliasing into the x86-64 backend.
- [ ] Remove generic dependencies on `%rax`, `%eax`, `%cl`, etc.
- [ ] Define the AArch64 `wN`/`xN` relationship in the ARM backend.
- [ ] Remove generic lowering assumptions about a dedicated shift-count register.
- [ ] Remove generic lowering assumptions about implicit multiply/divide registers.

---

## 6. Separate Frame Layout from x86 Implementation

The logical slot system is already a good abstraction; preserve it.

- [ ] Keep logical IR frame slots architecture-independent.
- [ ] Remove `%rbp` offsets from anything above the backend.
- [ ] Define target-specific frame layout policy.
- [ ] Make stack-frame alignment target/ABI-specific.
- [ ] Make outgoing argument areas target-specific.
- [ ] Move prologue/epilogue construction entirely into the target backend.
- [ ] Make saved-register selection target-specific.
- [ ] Do not encode `push`/`pop`/`leave` assumptions into shared code.

The ARM backend should be free to implement frame setup, register saves, local slots, outgoing arguments, and frame teardown without changing the IR representation.

---

## 7. Make Assembly AST Explicitly Target-Specific

Do not turn the current x86 assembly AST into a universal assembly language.

- [ ] Keep the existing assembly AST as x86-64-specific.
- [ ] Rename/rehome it as an x86-64 backend representation.
- [ ] Create a separate AArch64 assembly AST.
- [ ] Create a separate AArch64 emitter.
- [ ] Keep AT&T syntax assumptions confined to x86-64.
- [ ] Keep AArch64 register and addressing syntax confined to AArch64.
- [ ] Keep instruction legality/operand restrictions in the target backend.

A good target layout is:

```text
backend/
    x86_64/
        assembly_ast.py
        lowering.py
        emitter.py

    aarch64/
        assembly_ast.py
        lowering.py
        emitter.py
```

---

## 8. Keep IR Operations Genuinely Target-Neutral

Before starting ARM codegen, audit every IR instruction and ask:

> Does this describe a language/compiler operation, or does it accidentally describe x86 implementation?

- [ ] `IRMove`
- [ ] `IRBinOp`
- [ ] `IRUnOp`
- [ ] `IRCast`
- [ ] `IRCall`
- [ ] `IRLoad`
- [ ] `IRStore`
- [ ] `IRCopy`
- [ ] `IRBoundsCheck`
- [ ] `IRSliceBoundsCheck`
- [ ] `IRLocalAddress`
- [ ] `IRStaticDataAddress`
- [ ] `IRBranch`
- [ ] `IRJump`
- [ ] `IRReturn`
- [ ] `IRSliceGrow`

Verify that none require knowledge of physical registers or x86 instruction constraints.

The notable exception today is argument handling via `IRReadArgument`; that should be fixed before the port.

---

## 9. Isolate x86-Only Instruction Lowering

Audit shared/backend code for assumptions about:

- [ ] `rax`
- [ ] `eax`
- [ ] `edx`
- [ ] `ecx`
- [ ] `cl`
- [ ] `rbp`
- [ ] `rsp`
- [ ] `rdi` through `r9`
- [ ] `idiv`
- [ ] `cqto`
- [ ] `cdq`
- [ ] `setcc`
- [ ] x86 memory operands
- [ ] x86 `lea`
- [ ] `push`/`pop`

The current scalar lowering has deliberate x86 mechanisms around division/modulo, comparisons, and register constraints. Those should become target-local.

---

## 10. Define Target-Independent Data-Layout Invariants

Most current layout is compatible with AArch64 because both targets are 64-bit, but the assumptions should be explicit.

- [ ] Confirm pointer width is defined as 8 bytes.
- [ ] Confirm `int`/`int64` semantics are explicitly 64-bit.
- [ ] Confirm `int32` is exactly 32-bit.
- [ ] Confirm `int8`/`uint8`/`byte` are exactly 8-bit.
- [ ] Confirm `bool` storage width is intentional.
- [ ] Confirm slice layout is explicitly defined.
- [ ] Confirm string layout is explicitly defined.
- [ ] Confirm dictionary layout is explicitly defined.
- [ ] Confirm sum-type layout is explicitly defined.
- [ ] Confirm struct layout is independent of backend register conventions.
- [ ] Add tests that assert those layouts rather than relying on the x86 backend.

This is especially important because the same runtime representations must be consumed correctly by C on AArch64.

---

## 11. Separate Symbol Naming from Target Selection

- [ ] Move symbol naming into the target/linkage abstraction.
- [ ] Preserve Darwin symbol conventions independently of architecture.
- [ ] Preserve ELF conventions independently of architecture.
- [ ] Ensure `aarch64-macos` does not accidentally inherit x86-64 assumptions simply because it is "macOS."

---

## 12. Make Static-Data Lowering Target-Neutral

The IR should continue to say:

```text
IRStaticDataAddress("foo")
```

and nothing more.

- [ ] Keep labels in IR.
- [ ] Keep string/type-descriptor contents in IR.
- [ ] Move section/directive syntax into the target emitter.
- [ ] Verify `.asciz`, `.quad`, global symbols, and data-section conventions for AArch64 Mach-O.
- [ ] Keep linker/platform decisions out of IR.

---

## 13. Define the Runtime ABI Independently of the Backend

The C runtime should compile natively on both architectures.

- [ ] Document the ABI for `hornet_print`.
- [ ] Document the ABI for `hornet_panic`.
- [ ] Document the ABI for `hornet_slice_grow`.
- [ ] Document type-descriptor layout.
- [ ] Document slice layout.
- [ ] Document string layout.
- [ ] Ensure runtime structs/types do not depend on accidental C packing/alignment behavior.
- [ ] Compile and test `runtime/runtime.c` natively on arm64.
- [ ] Keep FFI calls subject to the same target ABI description as ordinary external calls.

---

## 14. Add Target-Independent Backend Conformance Tests

Before ARM exists, establish tests against the IR.

For every IR instruction, have at least one test that verifies its semantic result.

- [ ] Move
- [ ] arithmetic
- [ ] unary operations
- [ ] casts
- [ ] loads/stores
- [ ] branches
- [ ] calls
- [ ] returns
- [ ] addresses
- [ ] aggregate copies
- [ ] bounds checks
- [ ] slice growth
- [ ] composite returns
- [ ] multiword values

Then the eventual AArch64 backend can be tested against the same semantic expectations.

---

## 15. Add a Target Test Matrix

Once the target abstraction exists:

- [ ] Run existing compiler tests against x86-64 Linux.
- [ ] Run existing compiler tests against x86-64 macOS.
- [ ] Add AArch64 macOS to the matrix.
- [ ] Mark tests that require a native target appropriately.
- [ ] Add a native arm64 smoke test.
- [ ] Add runtime-only arm64 tests.
- [ ] Add FFI smoke tests on arm64.
- [ ] Add aggregate/pointer/slice smoke tests on arm64.
- [ ] Add randomized compiler tests to at least one native arm64 CI environment.

The key goal is:

```text
same Hornet source
        ↓
same IR
        ↓
different backend
        ↓
same observable result
```

---

## 16. Clean Up Remaining IR/Backend Dependency Direction

Before starting the second backend:

- [ ] Move `IdAllocator` fully into IR/infrastructure.
- [ ] Move escape analysis out of `codegen`.
- [ ] Keep ABI-specific helpers in backend/ABI code.
- [ ] Ensure `ir/` does not import the x86 backend.
- [ ] Ideally make `ir/` importable without importing any assembly implementation.

This is not strictly required to make AArch64 work, but it will make the architecture much easier to reason about.

---

## 17. Refactor Current Backend Naming

Before creating a sibling backend:

- [ ] Rename `CodeGenerator` to something like `X86Backend`.
- [ ] Rename/rehome x86-specific lowering modules.
- [ ] Make `lower_to_asm()` explicitly target the x86 backend.
- [ ] Remove generic-sounding names from x86-only utilities.
- [ ] Make the backend interface obvious enough that adding AArch64 does not require another rename pass.

---

## 18. Define the Backend Interface Before Implementing AArch64

Aim for a small shared interface along these lines:

```python
class Backend:
    architecture: str
    abi: ...
    registers: ...
    instruction_set: ...

    def lower(program: IRProgram) -> NativeProgram:
        ...
```

with:

```text
X86_64Backend
AArch64Backend
```

implementing the same conceptual interface.

Do not over-generalize the interface. Its purpose is simply to make the IR → native-code boundary explicit.

---

# What Does Not Need to Block AArch64

The following can happen later:

- [ ] Garbage collection
- [ ] Generic types
- [ ] Function values/closures
- [ ] Regex
- [ ] A larger standard library
- [ ] Self-hosting
- [ ] A new optimization framework
- [ ] A new register-allocation algorithm
- [ ] Moving more operations into the runtime
- [ ] Rewriting the IR

---

# Ready-to-Start-AArch64 Bar

I would consider the project ready to begin implementing AArch64 when all of these are true:

- [ ] Target = architecture + OS + ABI, not just platform.
- [ ] IR has no physical-register/ABI argument-slot semantics.
- [ ] Hidden composite returns are represented abstractly.
- [ ] Calling conventions are backend-owned.
- [ ] Register allocation accepts a target register model.
- [ ] Frame layout is backend-owned.
- [ ] x86 register-width helpers are x86-local.
- [ ] x86 assembly AST is explicitly x86-specific.
- [ ] Static-data emission is target-specific.
- [ ] Runtime ABI is documented.
- [ ] FFI uses the same external-call abstraction as runtime calls.
- [ ] IR can be built and optimized without importing x86 assembly code.
- [ ] Existing x86 tests continue to pass after the refactoring.
- [ ] The build system can explicitly name/select `x86_64-macos`.

## Highest-Priority Item

If only one prerequisite is addressed first, make it:

```text
IRReadArgument(index)
        ↓
IRReadParameter(parameter_index)
```

and move physical parameter placement into the backend ABI.

The current representation explicitly embeds SysV/x86-64 argument ordering and reserves the first slot for a hidden composite-return pointer. Removing that assumption is the clearest way to make the IR genuinely reusable across x86-64 and AArch64.
