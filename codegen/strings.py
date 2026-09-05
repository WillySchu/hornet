"""A str value is a plain pointer -- to a literal's static address, or
a concatenation's malloc'd buffer -- never copied for its bytes the
way an array or struct is. Also home to hornet_stringify, the hand-
built, recursive runtime function that turns any typed value into
printable bytes via a shared growable-buffer append primitive, driven
by a runtime type-descriptor tree built once per distinct type."""

from codegen.assembly_ast import (
    Add,
    AddQ,
    AsmFunction,
    CallInstr,
    Cmp,
    CmpQ,
    Div,
    DivQ,
    Jae,
    Je,
    Jg,
    Jmp,
    Jne,
    Imm,
    IMul,
    Instruction,
    Label,
    LeaQ,
    LeaQFrame,
    Leave,
    Memory,
    Mov,
    MovB,
    MovQ,
    MovSX,
    MovZX,
    Neg,
    NegQ,
    Operand,
    Pop,
    Push,
    Register,
    Ret,
    SubQ,
    SetCC,
)
from codegen.errors import CodegenError
from codegen.utils import as_byte_register, type_byte_width, type_of, as_qword_register, COMPARISON_CONDITION_CODES
from parser import Call, Variable, Field, Index, StringLiteral, Binary, Node, BinaryOp
from semantic import Type, TypeKind


# Kind tags for print's runtime type descriptors (see
# _get_or_build_type_descriptor and AsmProgram.type_descriptors).
# Plain small ints, embedded as a descriptor's first .quad field;
# hornet_stringify switches on this to interpret the rest.
_TYPEDESC_INT = 0
_TYPEDESC_BOOL = 1
_TYPEDESC_STR = 2
_TYPEDESC_ARRAY = 3
_TYPEDESC_SLICE = 4
_TYPEDESC_STRUCT = 5
_TYPEDESC_INT8 = 6
_TYPEDESC_UINT8 = 7
_TYPEDESC_INT64 = 8


# Fixed %rbp-relative local-slot layout for the hand-built
# hornet_stringify runtime function (see build_stringify_function).
# Every local gets its own permanent slot, never reused across
# branches even though ARRAY/SLICE and STRUCT never run concurrently
# within one call -- stack space is cheap. All are spilled to their
# slot on entry and reloaded as needed, never kept live in a register
# across this function's own recursive calls or its calls to malloc:
# registers don't survive a nested call, and getting that wrong cost
# real time earlier (see gen_buffer_append_bytes_into's own
# %ecx-across-malloc bug).
_STRINGIFY_VALUE_ADDR = -8
_STRINGIFY_TYPE_DESC = -16
_STRINGIFY_QUOTE_STRINGS = -24
_STRINGIFY_BUF_STATE_ADDR = -32
_STRINGIFY_ITOA_SCRATCH = -48       # 16 bytes: -48 .. -33
_STRINGIFY_LOOP_INDEX = -56
_STRINGIFY_ELEM_ADDR = -64
_STRINGIFY_ELEM_TYPE_DESC = -72
_STRINGIFY_COLLECTION_LENGTH = -80
_STRINGIFY_ELEM_WIDTH = -88
_STRINGIFY_FIELD_COUNT = -96
_STRINGIFY_FIELD_NAME_ADDR = -104
_STRINGIFY_STR_PTR_SCRATCH = -112
_STRINGIFY_STR_LEN_SCRATCH = -120
_STRINGIFY_SLICE_BASE_PTR = -128    # a slice's backing-array pointer,
# read from value_addr's {ptr,len,cap} descriptor -- unlike ARRAY,
# whose value_addr points directly at the data.
_STRINGIFY_FIELD_TYPE_DESC = -136
_STRINGIFY_FIELD_OFFSET = -144
_STRINGIFY_ITOA_SCRATCH_64 = -176   # 32 bytes: -176 .. -145. A
# separate slot from _STRINGIFY_ITOA_SCRATCH above, not a resize of
# it: int64's max magnitude needs 20 characters (incl. sign) vs. the
# 16-byte buffer's 11.
_STRINGIFY_FRAME_SIZE = 176  # already a multiple of 16


class StringsMixin:
    def gen_int_to_decimal_into(self, value: Operand, scratch_offset: int, scratch_size: int) -> list[Instruction]:
        """Converts a 32-bit signed int into its decimal ASCII
        representation, written into a caller-provided scratch buffer
        at Memory('rbp', scratch_offset) .. Memory('rbp',
        scratch_offset + scratch_size - 1) -- scratch_size must be at
        least 11 (any 32-bit value's digits plus a leading '-'); 16
        gives comfortable margin.

        Builds digits into the buffer from the END backward (least-
        significant digit first, the order repeated division by 10
        produces them in) -- one pass, no separate reversal step.

        Handles INT_MIN correctly without widening to 64 bits:
        negating INT_MIN in 32-bit two's complement leaves its bit
        pattern unchanged (there's no positive counterpart to negate
        to), but that same bit pattern, reinterpreted as UNSIGNED, is
        exactly INT_MIN's correct magnitude (2147483648). So after
        peeling off the sign (checked before negation) and negating,
        the digit-extraction loop divides with Div, not IDiv --
        treating the magnitude as unsigned throughout makes INT_MIN
        correct with no special-casing.

        On return: %r8 holds the address of the first character
        (which may be partway into the buffer, since digits are
        written backward from the end), %r9d holds the character
        count including a leading '-' if negative -- ready to hand
        directly to gen_buffer_append_bytes_into as (source_addr,
        count). Fixed internal scratch beyond %r8/%r9d: %eax, %edx,
        %ecx (holds 10), %r10 (write-position pointer). Callers must
        avoid all of these while this runs."""
        instructions = []
        zero_label = self.new_label("itoa_zero")
        positive_label = self.new_label("itoa_positive")
        digits_label = self.new_label("itoa_digits")
        loop_label = self.new_label("itoa_loop")
        skip_sign_label = self.new_label("itoa_skip_sign")
        done_label = self.new_label("itoa_done")

        is_negative = Register('r9d')  # 0 or 1, set below; read again after the digit loop
        write_pos = Register('r10')

        instructions.append(Mov(src=value, dst=Register('eax')))
        instructions.append(Cmp(src=Imm(0), dst=Register('eax')))
        instructions.append(Je(zero_label))
        instructions.append(Jg(positive_label))

        # negative: record it, then negate to get the magnitude --
        # correct even for INT_MIN, per this method's own docstring.
        instructions.append(Mov(src=Imm(1), dst=is_negative))
        instructions.append(Neg(Register('eax')))
        instructions.append(Jmp(digits_label))

        instructions.append(Label(positive_label))
        instructions.append(Mov(src=Imm(0), dst=is_negative))

        instructions.append(Label(digits_label))
        # %eax now holds a non-negative magnitude either way. write_pos
        # starts one past the buffer's own last byte, since the loop
        # always decrements BEFORE writing.
        instructions.append(LeaQFrame(offset=scratch_offset + scratch_size, dst=write_pos))
        instructions.append(Mov(src=Imm(10), dst=Register('ecx')))
        instructions.append(Label(loop_label))
        instructions.append(Mov(src=Imm(0), dst=Register('edx')))  # zero-extend: this is an UNSIGNED divide
        instructions.append(Div(Register('ecx')))
        instructions.append(Add(src=Imm(ord('0')), dst=Register('edx')))
        instructions.append(SubQ(src=Imm(1), dst=write_pos))
        instructions.append(MovB(src=as_byte_register(Register('edx')), dst=Memory(write_pos.name, 0)))
        instructions.append(Cmp(src=Imm(0), dst=Register('eax')))
        instructions.append(Jne(loop_label))

        instructions.append(Cmp(src=Imm(0), dst=is_negative))
        instructions.append(Je(skip_sign_label))
        instructions.append(SubQ(src=Imm(1), dst=write_pos))
        instructions.append(MovB(src=Imm(ord('-')), dst=Memory(write_pos.name, 0)))
        instructions.append(Label(skip_sign_label))
        instructions.append(Jmp(done_label))

        instructions.append(Label(zero_label))
        instructions.append(LeaQFrame(offset=scratch_offset + scratch_size - 1, dst=write_pos))
        instructions.append(MovB(src=Imm(ord('0')), dst=Memory(write_pos.name, 0)))

        instructions.append(Label(done_label))
        # count = (one-past-the-end address) - (final write_pos), a
        # 64-bit address subtraction whose result is always small
        # enough that %r9d (the low 32 bits, automatically correct via
        # ordinary x86-64 zero-extension) already holds it directly.
        instructions.append(LeaQFrame(offset=scratch_offset + scratch_size, dst=Register('r9')))
        instructions.append(SubQ(src=write_pos, dst=Register('r9')))
        instructions.append(MovQ(src=write_pos, dst=Register('r8')))
        return instructions

    def gen_int64_to_decimal_into(self, value: Operand, scratch_offset: int, scratch_size: int) -> list[Instruction]:
        """The int64 counterpart to gen_int_to_decimal_into -- same
        algorithm, one register width up. `value` must already be a
        64-bit operand (e.g. Register('rax'), not Register('eax')).

        scratch_size must be at least 21: INT64_MIN's magnitude is 19
        digits, needing 20 characters plus a leading '-'. Uses the
        dedicated _STRINGIFY_ITOA_SCRATCH_64 buffer, a separate slot
        from the 32-bit version's 16-byte one.

        Operates on the 64-bit view of the same registers
        (%rax/%rdx/%rcx in place of %eax/%edx/%ecx; %r10/%r8/%r9d
        unchanged, since the first two already hold addresses and the
        digit count never exceeds two digits) via DivQ/NegQ/CmpQ/MovQ
        -- handling INT64_MIN the same way the 32-bit version handles
        INT_MIN (negate, then divide as unsigned).

        On return: %r8/%r9d hold (address, count), the identical
        contract gen_int_to_decimal_into specifies."""
        instructions = []
        zero_label = self.new_label("itoa64_zero")
        positive_label = self.new_label("itoa64_positive")
        digits_label = self.new_label("itoa64_digits")
        loop_label = self.new_label("itoa64_loop")
        skip_sign_label = self.new_label("itoa64_skip_sign")
        done_label = self.new_label("itoa64_done")

        is_negative = Register('r9d')  # 0 or 1, set below; read again after the digit loop
        write_pos = Register('r10')

        instructions.append(MovQ(src=value, dst=Register('rax')))
        instructions.append(CmpQ(src=Imm(0), dst=Register('rax')))
        instructions.append(Je(zero_label))
        instructions.append(Jg(positive_label))

        # negative: record it, then negate to get the magnitude --
        # correct even for INT64_MIN, per this method's own docstring.
        instructions.append(Mov(src=Imm(1), dst=is_negative))
        instructions.append(NegQ(Register('rax')))
        instructions.append(Jmp(digits_label))

        instructions.append(Label(positive_label))
        instructions.append(Mov(src=Imm(0), dst=is_negative))

        instructions.append(Label(digits_label))
        # %rax now holds a non-negative magnitude either way. write_pos
        # starts one past the buffer's own last byte, since the loop
        # always decrements BEFORE writing.
        instructions.append(LeaQFrame(offset=scratch_offset + scratch_size, dst=write_pos))
        instructions.append(MovQ(src=Imm(10), dst=Register('rcx')))
        instructions.append(Label(loop_label))
        instructions.append(MovQ(src=Imm(0), dst=Register('rdx')))  # zero-extend: this is an UNSIGNED divide
        instructions.append(DivQ(Register('rcx')))
        instructions.append(Add(src=Imm(ord('0')), dst=Register('edx')))  # the remainder is always a single digit (< 10), so this stays 32-bit
        instructions.append(SubQ(src=Imm(1), dst=write_pos))
        instructions.append(MovB(src=as_byte_register(Register('edx')), dst=Memory(write_pos.name, 0)))
        instructions.append(CmpQ(src=Imm(0), dst=Register('rax')))
        instructions.append(Jne(loop_label))

        instructions.append(Cmp(src=Imm(0), dst=is_negative))
        instructions.append(Je(skip_sign_label))
        instructions.append(SubQ(src=Imm(1), dst=write_pos))
        instructions.append(MovB(src=Imm(ord('-')), dst=Memory(write_pos.name, 0)))
        instructions.append(Label(skip_sign_label))
        instructions.append(Jmp(done_label))

        instructions.append(Label(zero_label))
        instructions.append(LeaQFrame(offset=scratch_offset + scratch_size - 1, dst=write_pos))
        instructions.append(MovB(src=Imm(ord('0')), dst=Memory(write_pos.name, 0)))

        instructions.append(Label(done_label))
        instructions.append(LeaQFrame(offset=scratch_offset + scratch_size, dst=Register('r9')))
        instructions.append(SubQ(src=write_pos, dst=Register('r9')))
        instructions.append(MovQ(src=write_pos, dst=Register('r8')))
        return instructions

    def _get_or_build_type_descriptor(self, t: Type, in_progress: dict[Type, str]) -> str:
        """Returns the label of t's runtime type descriptor, building
        and registering it into self.type_descriptors if it hasn't
        already been built WITHIN THIS ONE CALL (in_progress, keyed on
        t -- Type is frozen, so it's already a safe, correct dict key).

        `in_progress` is scoped to a single top-level call (one fresh
        dict per print() call site that needs a descriptor tree), not
        a whole-program cache -- two print() calls on the same struct
        type each build their own tree from scratch, matching
        gen_string_literal_into's own "no cross-occurrence dedup, keep
        it simple" choice.

        Reuse WITHIN one call isn't optional, though: it's the only
        way a self-referential struct (`struct Node: int value; []Node
        children`) can be represented as a finite amount of static
        data at all. Reserving this type's label BEFORE recursing into
        anything it contains is what breaks that cycle -- a nested
        reference back to the same type finds its label already in
        in_progress and reuses it.

        Every non-leaf kind (ARRAY, SLICE, STRUCT) carries its own
        type-name string (e.g. "[3]int", "Point") as a second field
        right after the kind tag; hornet_stringify prints this
        immediately before that kind's opening bracket/brace at EVERY
        level it appears, not just the outermost value a print() call
        names directly. INT/INT8/UINT8/BOOL/STR carry no name field at
        all -- a genuine per-kind layout difference, not a uniform
        field every descriptor has.
        """
        if t in in_progress:
            return in_progress[t]
        label = self.new_label("typedesc")
        in_progress[t] = label

        if t.kind == TypeKind.INT:
            self.type_descriptors.append((label, [_TYPEDESC_INT]))
        elif t == Type.INT8:
            self.type_descriptors.append((label, [_TYPEDESC_INT8]))
        elif t == Type.UINT8:
            self.type_descriptors.append((label, [_TYPEDESC_UINT8]))
        elif t == Type.INT64:
            self.type_descriptors.append((label, [_TYPEDESC_INT64]))
        elif t.kind == TypeKind.BOOL:
            self.type_descriptors.append((label, [_TYPEDESC_BOOL]))
        elif t.kind == TypeKind.STR:
            self.type_descriptors.append((label, [_TYPEDESC_STR]))
        elif t.kind == TypeKind.ARRAY:
            name_label = self.new_label("typedesc_name")
            self.string_literals.append((name_label, str(t)))
            elem_label = self._get_or_build_type_descriptor(t.element_type, in_progress)
            elem_width = type_byte_width(t.element_type, self.struct_registry)
            self.type_descriptors.append((label, [_TYPEDESC_ARRAY, name_label, elem_label, t.size, elem_width]))
        elif t.kind == TypeKind.SLICE:
            name_label = self.new_label("typedesc_name")
            self.string_literals.append((name_label, str(t)))
            elem_label = self._get_or_build_type_descriptor(t.element_type, in_progress)
            elem_width = type_byte_width(t.element_type, self.struct_registry)
            self.type_descriptors.append((label, [_TYPEDESC_SLICE, name_label, elem_label, elem_width]))
        elif t.kind == TypeKind.STRUCT:
            name_label = self.new_label("typedesc_name")
            self.string_literals.append((name_label, str(t)))
            struct_info = self.struct_registry[t.struct_name]
            field_fields: list = []
            field_count = 0
            for field_name, field_type in struct_info.fields.items():
                field_name_label = self.new_label("typedesc_fname")
                self.string_literals.append((field_name_label, field_name))
                field_type_label = self._get_or_build_type_descriptor(field_type, in_progress)
                field_offset = self._field_offset(t.struct_name, field_name)
                field_fields.extend([field_name_label, field_type_label, field_offset])
                field_count += 1
            self.type_descriptors.append((label, [_TYPEDESC_STRUCT, name_label, field_count] + field_fields))
        else:
            raise CodegenError(f"No type descriptor rule for: {t}")

        return label

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

    def _get_comma_space_label(self) -> str:
        if self._comma_space_label is None:
            self._comma_space_label = self.new_label("comma_space_str")
            self.string_literals.append((self._comma_space_label, ", "))
        return self._comma_space_label

    def _get_colon_space_label(self) -> str:
        if self._colon_space_label is None:
            self._colon_space_label = self.new_label("colon_space_str")
            self.string_literals.append((self._colon_space_label, ": "))
        return self._colon_space_label

    def _get_empty_str_label(self) -> str:
        """A shared, static, empty ("") string constant -- str's zero
        value. Deliberately NOT a null pointer: every string operation
        this file generates dereferences a str value with no null
        check, so a zero-initialized str has to point at a real, valid
        (if empty) C string -- a null pointer would segfault the
        instant anything touched it."""
        if self._empty_str_label is None:
            self._empty_str_label = self.new_label("empty_str")
            self.string_literals.append((self._empty_str_label, ""))
        return self._empty_str_label

    def _gen_stringify_bulk_append(self, source_addr: Register, count: Operand) -> list[Instruction]:
        """Loads the print buffer's {ptr, len, cap} triple from
        _STRINGIFY_BUF_STATE_ADDR, bulk-appends `count` bytes from
        `source_addr` via gen_buffer_append_bytes_into, then writes
        the triple back out -- the load-append-store sequence every
        multi-byte append inside hornet_stringify needs, factored out
        once rather than re-derived at each call site. Only valid to
        call from within hornet_stringify's own body -- it hardcodes
        that function's fixed frame layout.

        `source_addr` and `count` must not be %rbx/%r12/%r13/%r11
        (clobbered loading/storing the buffer state here) or any of
        gen_buffer_append_bytes_into's own internal scratch (%rax/
        %rcx/%rdx/%rdi/%r10/%r11). %r8/%r9 are safe against both,
        which is why gen_int_to_decimal_into's contract (and this
        function's own BOOL case) leave their result there."""
        instructions = []
        instructions.append(MovQ(src=Memory('rbp', _STRINGIFY_BUF_STATE_ADDR), dst=Register('r11')))
        instructions.append(MovQ(src=Memory('r11', 0), dst=Register('rbx')))
        instructions.append(MovQ(src=Memory('r11', 8), dst=Register('r12')))
        instructions.append(MovQ(src=Memory('r11', 16), dst=Register('r13')))
        instructions.extend(self.gen_buffer_append_bytes_into(
            Register('rbx'), Register('r12'), Register('r12d'),
            Register('r13'), Register('r13d'),
            source_addr, count,
        ))
        instructions.append(MovQ(src=Memory('rbp', _STRINGIFY_BUF_STATE_ADDR), dst=Register('r11')))
        instructions.append(MovQ(src=Register('rbx'), dst=Memory('r11', 0)))
        instructions.append(MovQ(src=Register('r12'), dst=Memory('r11', 8)))
        instructions.append(MovQ(src=Register('r13'), dst=Memory('r11', 16)))
        return instructions

    def _gen_stringify_byte_append(self, byte_value: Operand) -> list[Instruction]:
        """The single-byte counterpart to _gen_stringify_bulk_append,
        via gen_buffer_append_byte_into -- used for a bracket, paren,
        or quote mark. Same fixed-frame and register-safety
        constraints as that method."""
        instructions = []
        instructions.append(MovQ(src=Memory('rbp', _STRINGIFY_BUF_STATE_ADDR), dst=Register('r11')))
        instructions.append(MovQ(src=Memory('r11', 0), dst=Register('rbx')))
        instructions.append(MovQ(src=Memory('r11', 8), dst=Register('r12')))
        instructions.append(MovQ(src=Memory('r11', 16), dst=Register('r13')))
        instructions.extend(self.gen_buffer_append_byte_into(
            Register('rbx'), Register('r12'), Register('r12d'),
            Register('r13'), Register('r13d'),
            byte_value,
        ))
        instructions.append(MovQ(src=Memory('rbp', _STRINGIFY_BUF_STATE_ADDR), dst=Register('r11')))
        instructions.append(MovQ(src=Register('rbx'), dst=Memory('r11', 0)))
        instructions.append(MovQ(src=Register('r12'), dst=Memory('r11', 8)))
        instructions.append(MovQ(src=Register('r13'), dst=Memory('r11', 16)))
        return instructions

    def _gen_stringify_append_c_string_at(self, name_addr_offset: int) -> list[Instruction]:
        """strlen()s the null-terminated string whose address is
        already stored at Memory('rbp', name_addr_offset), then bulk-
        appends that many bytes onto the buffer -- shared by every
        place hornet_stringify prints a compile-time-known NAME (a
        type's name, or a struct field's name) rather than a runtime
        VALUE: none of these carry their own length in a type
        descriptor's static data, so strlen supplies it.

        Reads the address from `name_addr_offset` (Memory, not a
        register) both before AND after the intervening `call strlen`,
        since a real function call is free to clobber any caller-saved
        register but never touches this function's own stack frame --
        the "spill across a call, never trust a register to survive
        one" discipline used throughout this file."""
        instructions = []
        instructions.append(MovQ(src=Memory('rbp', name_addr_offset), dst=Register('rdi')))
        instructions.append(CallInstr('strlen'))
        instructions.append(MovQ(src=Memory('rbp', name_addr_offset), dst=Register('r8')))
        instructions.append(Mov(src=Register('eax'), dst=Register('r9d')))
        instructions.extend(self._gen_stringify_bulk_append(Register('r8'), Register('r9d')))
        return instructions

    def _gen_stringify_collection_loop_body(self, base_addr_slot: int) -> list[Instruction]:
        """Shared by the ARRAY and SLICE dispatch cases in
        build_stringify_function: given _STRINGIFY_COLLECTION_LENGTH/
        _STRINGIFY_ELEM_WIDTH/_STRINGIFY_ELEM_TYPE_DESC already
        populated by the caller, and `base_addr_slot` naming whichever
        local slot holds element 0's address (ARRAY's own
        _STRINGIFY_VALUE_ADDR directly; SLICE's own
        _STRINGIFY_SLICE_BASE_PTR, already unwrapped one level of
        indirection from value_addr) -- emits `[elem, elem, ...]`'s
        INSIDE (the caller appends the brackets themselves): a ", "
        separator before every element but the first, then a recursive
        call into hornet_stringify for each element's value, with
        quote_strings hardcoded to 1 (an array/slice element is never
        the outermost value of a print() call).

        Element addresses use the same 32-bit-multiply-then-zero-
        extend idiom gen_index_address_into uses for ordinary
        indexing, safe for the same reason: no index this language can
        construct is large enough to make the implicit zero-extension
        incorrect."""
        instructions = []
        loop_start = self.new_label("stringify_collection_loop")
        loop_done = self.new_label("stringify_collection_loop_done")
        skip_sep = self.new_label("stringify_collection_skip_sep")

        instructions.append(MovQ(src=Imm(0), dst=Memory('rbp', _STRINGIFY_LOOP_INDEX)))
        instructions.append(Label(loop_start))
        instructions.append(Mov(src=Memory('rbp', _STRINGIFY_LOOP_INDEX), dst=Register('eax')))
        instructions.append(Mov(src=Memory('rbp', _STRINGIFY_COLLECTION_LENGTH), dst=Register('ecx')))
        instructions.append(Cmp(src=Register('ecx'), dst=Register('eax')))
        instructions.append(Jae(loop_done))

        instructions.append(Cmp(src=Imm(0), dst=Memory('rbp', _STRINGIFY_LOOP_INDEX)))
        instructions.append(Je(skip_sep))
        instructions.append(LeaQ(label=self._get_comma_space_label(), dst=Register('r8')))
        instructions.append(Mov(src=Imm(2), dst=Register('r9d')))
        instructions.extend(self._gen_stringify_bulk_append(Register('r8'), Register('r9d')))
        instructions.append(Label(skip_sep))

        instructions.append(Mov(src=Memory('rbp', _STRINGIFY_LOOP_INDEX), dst=Register('eax')))
        instructions.append(Mov(src=Memory('rbp', _STRINGIFY_ELEM_WIDTH), dst=Register('ecx')))
        instructions.append(IMul(src=Register('ecx'), dst=Register('eax')))
        instructions.append(MovQ(src=Memory('rbp', base_addr_slot), dst=Register('r10')))
        instructions.append(AddQ(src=Register('rax'), dst=Register('r10')))
        instructions.append(MovQ(src=Register('r10'), dst=Memory('rbp', _STRINGIFY_ELEM_ADDR)))

        instructions.append(MovQ(src=Memory('rbp', _STRINGIFY_ELEM_ADDR), dst=Register('rdi')))
        instructions.append(MovQ(src=Memory('rbp', _STRINGIFY_ELEM_TYPE_DESC), dst=Register('rsi')))
        instructions.append(Mov(src=Imm(1), dst=Register('edx')))
        instructions.append(MovQ(src=Memory('rbp', _STRINGIFY_BUF_STATE_ADDR), dst=Register('rcx')))
        instructions.append(CallInstr('hornet_stringify'))

        instructions.append(Mov(src=Memory('rbp', _STRINGIFY_LOOP_INDEX), dst=Register('eax')))
        instructions.append(Add(src=Imm(1), dst=Register('eax')))
        instructions.append(Mov(src=Register('eax'), dst=Memory('rbp', _STRINGIFY_LOOP_INDEX)))
        instructions.append(Jmp(loop_start))
        instructions.append(Label(loop_done))
        return instructions

    def build_stringify_function(self) -> AsmFunction:
        """Builds `hornet_stringify`, the print machinery's hand-built
        runtime function -- built ONCE per program (not derived from
        any Hornet AST Function, not tied to any one print() call
        site), added to AsmProgram.functions alongside every ordinary
        Hornet-compiled function, and called via an ordinary `call
        hornet_stringify` from gen_print_call_into. This is the first
        hand-built function in this compiler needing genuine, unbounded
        recursion (calling itself once per array/slice element or
        struct field) rather than instructions inlined at each call
        site -- the only way a self-referential struct can ever be
        printed, since the recursion depth depends on a value's
        runtime shape, not anything compile-time code generation could
        unroll for.

        SIGNATURE (an ordinary SysV integer-argument call, as if
        declared `void hornet_stringify(void* value_addr, void*
        type_desc, long quote_strings, void* buf_state_addr)`):
          %rdi = value_addr     -- address of the value being printed
          %rsi = type_desc      -- pointer to its own type descriptor
          %rdx = quote_strings  -- 0 or 1; whether a STR value here
                                    should be wrapped in single quotes
                                    (never true for the OUTERMOST call
                                    from gen_print_call_into, always
                                    true for every recursive call this
                                    function makes to itself, so a str
                                    nested inside a collection or
                                    struct stays unambiguous next to
                                    its neighbors)
          %rcx = buf_state_addr -- pointer to a 24-byte {ptr, len, cap}
                                    block, OWNED by gen_print_call_into
                                    (not this function), read from and
                                    written back to on every append, so
                                    growth or reallocation anywhere in
                                    a deeply nested value is visible
                                    everywhere else building the same
                                    print's output

        Every incoming argument is spilled into its own fixed local
        slot immediately on entry (see the _STRINGIFY_* offset
        constants above) and reloaded from there whenever needed,
        never kept live in a register across this function's own
        recursive calls or its calls to malloc -- registers don't
        survive a nested call the way a fixed stack slot does.

        Dispatches on type_desc's first field (the kind tag) via an
        ordinary chain of comparisons -- nine kinds as of int64, still
        not enough to justify a jump table. The tag is stored as a
        full .quad but compared via its 32-bit view, since every kind
        value is small and non-negative and Cmp is fixed at 32-bit.
        """
        instructions: list[Instruction] = []
        instructions.append(Push(Register('rbp')))
        instructions.append(MovQ(src=Register('rsp'), dst=Register('rbp')))
        instructions.append(SubQ(src=Imm(_STRINGIFY_FRAME_SIZE), dst=Register('rsp')))
        instructions.append(MovQ(src=Register('rdi'), dst=Memory('rbp', _STRINGIFY_VALUE_ADDR)))
        instructions.append(MovQ(src=Register('rsi'), dst=Memory('rbp', _STRINGIFY_TYPE_DESC)))
        instructions.append(MovQ(src=Register('rdx'), dst=Memory('rbp', _STRINGIFY_QUOTE_STRINGS)))
        instructions.append(MovQ(src=Register('rcx'), dst=Memory('rbp', _STRINGIFY_BUF_STATE_ADDR)))

        int_label = self.new_label("stringify_int")
        int8_label = self.new_label("stringify_int8")
        uint8_label = self.new_label("stringify_uint8")
        int64_label = self.new_label("stringify_int64")
        bool_label = self.new_label("stringify_bool")
        str_label = self.new_label("stringify_str")
        array_label = self.new_label("stringify_array")
        slice_label = self.new_label("stringify_slice")
        struct_label = self.new_label("stringify_struct")
        done_label = self.new_label("stringify_done")

        instructions.append(MovQ(src=Memory('rbp', _STRINGIFY_TYPE_DESC), dst=Register('r10')))
        instructions.append(MovQ(src=Memory('r10', 0), dst=Register('r10')))
        instructions.append(Cmp(src=Imm(_TYPEDESC_INT), dst=Register('r10d')))
        instructions.append(Je(int_label))
        instructions.append(Cmp(src=Imm(_TYPEDESC_INT8), dst=Register('r10d')))
        instructions.append(Je(int8_label))
        instructions.append(Cmp(src=Imm(_TYPEDESC_UINT8), dst=Register('r10d')))
        instructions.append(Je(uint8_label))
        instructions.append(Cmp(src=Imm(_TYPEDESC_INT64), dst=Register('r10d')))
        instructions.append(Je(int64_label))
        instructions.append(Cmp(src=Imm(_TYPEDESC_BOOL), dst=Register('r10d')))
        instructions.append(Je(bool_label))
        instructions.append(Cmp(src=Imm(_TYPEDESC_STR), dst=Register('r10d')))
        instructions.append(Je(str_label))
        instructions.append(Cmp(src=Imm(_TYPEDESC_ARRAY), dst=Register('r10d')))
        instructions.append(Je(array_label))
        instructions.append(Cmp(src=Imm(_TYPEDESC_SLICE), dst=Register('r10d')))
        instructions.append(Je(slice_label))
        instructions.append(Cmp(src=Imm(_TYPEDESC_STRUCT), dst=Register('r10d')))
        instructions.append(Je(struct_label))
        instructions.append(Jmp(done_label))  # unreachable for any type this compiler ever hands here

        # -- INT ----------------------------------------------------------
        instructions.append(Label(int_label))
        instructions.append(MovQ(src=Memory('rbp', _STRINGIFY_VALUE_ADDR), dst=Register('r10')))
        instructions.append(Mov(src=Memory('r10', 0), dst=Register('eax')))
        instructions.extend(self.gen_int_to_decimal_into(
            Register('eax'), _STRINGIFY_ITOA_SCRATCH, 16,
        ))
        # %r8 = digits start address, %r9d = digit count
        # (gen_int_to_decimal_into's contract) -- both survive
        # untouched into _gen_stringify_bulk_append, which never uses
        # %r8/%r9.
        instructions.extend(self._gen_stringify_bulk_append(Register('r8'), Register('r9d')))
        instructions.append(Jmp(done_label))

        # -- INT8/UINT8 -----------------------------------------------------
        # Identical to INT above except for one instruction: value_addr
        # points at a genuinely 1-byte-wide value, so reading it needs a
        # WIDENING load -- MovSX (sign-extend) for int8, MovZX
        # (zero-extend) for uint8 -- rather than INT's plain 4-byte Mov,
        # which would read adjacent garbage bytes and, for int8, could
        # misread a negative value as large and positive. Once widened
        # into %eax, the same gen_int_to_decimal_into/bulk-append
        # sequence INT uses works for either.
        instructions.append(Label(int8_label))
        instructions.append(MovQ(src=Memory('rbp', _STRINGIFY_VALUE_ADDR), dst=Register('r10')))
        instructions.append(MovSX(src=Memory('r10', 0), dst=Register('eax')))
        instructions.extend(self.gen_int_to_decimal_into(
            Register('eax'), _STRINGIFY_ITOA_SCRATCH, 16,
        ))
        instructions.extend(self._gen_stringify_bulk_append(Register('r8'), Register('r9d')))
        instructions.append(Jmp(done_label))

        instructions.append(Label(uint8_label))
        instructions.append(MovQ(src=Memory('rbp', _STRINGIFY_VALUE_ADDR), dst=Register('r10')))
        instructions.append(MovZX(src=Memory('r10', 0), dst=Register('eax')))
        instructions.extend(self.gen_int_to_decimal_into(
            Register('eax'), _STRINGIFY_ITOA_SCRATCH, 16,
        ))
        instructions.extend(self._gen_stringify_bulk_append(Register('r8'), Register('r9d')))
        instructions.append(Jmp(done_label))

        # -- INT64 ----------------------------------------------------------
        # int64's value_addr points at a genuinely 8-byte-wide value,
        # needing a full-width MovQ into %rax rather than INT's 4-byte
        # Mov, which would silently drop the high 32 bits. Unlike
        # int8/uint8's narrowing case, this can't reuse gen_int_to_
        # decimal_into (written entirely in 32-bit instructions), so it
        # calls the dedicated gen_int64_to_decimal_into instead, into
        # its own _STRINGIFY_ITOA_SCRATCH_64 buffer.
        instructions.append(Label(int64_label))
        instructions.append(MovQ(src=Memory('rbp', _STRINGIFY_VALUE_ADDR), dst=Register('r10')))
        instructions.append(MovQ(src=Memory('r10', 0), dst=Register('rax')))
        instructions.extend(self.gen_int64_to_decimal_into(
            Register('rax'), _STRINGIFY_ITOA_SCRATCH_64, 32,
        ))
        instructions.extend(self._gen_stringify_bulk_append(Register('r8'), Register('r9d')))
        instructions.append(Jmp(done_label))

        # -- BOOL ---------------------------------------------------------
        instructions.append(Label(bool_label))
        instructions.append(MovQ(src=Memory('rbp', _STRINGIFY_VALUE_ADDR), dst=Register('r10')))
        instructions.append(Mov(src=Memory('r10', 0), dst=Register('eax')))
        bool_false_label = self.new_label("stringify_bool_false")
        bool_append_label = self.new_label("stringify_bool_append")
        instructions.append(Cmp(src=Imm(0), dst=Register('eax')))
        instructions.append(Je(bool_false_label))
        instructions.append(LeaQ(label=self._get_true_str_label(), dst=Register('r8')))
        instructions.append(Mov(src=Imm(4), dst=Register('r9d')))
        instructions.append(Jmp(bool_append_label))
        instructions.append(Label(bool_false_label))
        instructions.append(LeaQ(label=self._get_false_str_label(), dst=Register('r8')))
        instructions.append(Mov(src=Imm(5), dst=Register('r9d')))
        instructions.append(Label(bool_append_label))
        instructions.extend(self._gen_stringify_bulk_append(Register('r8'), Register('r9d')))
        instructions.append(Jmp(done_label))

        # -- STR ------------------------------------------------------------
        # A str VALUE is itself a pointer, so unlike INT/BOOL, value_addr
        # here holds the address of a POINTER, which (once dereferenced)
        # is what needs stringifying. quote_strings decides whether it's
        # wrapped in single quotes (nested inside a collection or
        # struct) or left bare (the outermost value of a print() call).
        instructions.append(Label(str_label))
        instructions.append(MovQ(src=Memory('rbp', _STRINGIFY_VALUE_ADDR), dst=Register('r10')))
        instructions.append(MovQ(src=Memory('r10', 0), dst=Register('r10')))
        instructions.append(MovQ(src=Register('r10'), dst=Memory('rbp', _STRINGIFY_STR_PTR_SCRATCH)))
        instructions.append(MovQ(src=Register('r10'), dst=Register('rdi')))
        instructions.append(CallInstr('strlen'))
        instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', _STRINGIFY_STR_LEN_SCRATCH)))

        str_unquoted_label = self.new_label("stringify_str_unquoted")
        instructions.append(Cmp(src=Imm(0), dst=Memory('rbp', _STRINGIFY_QUOTE_STRINGS)))
        instructions.append(Je(str_unquoted_label))

        instructions.extend(self._gen_stringify_byte_append(Imm(ord("'"))))
        instructions.append(MovQ(src=Memory('rbp', _STRINGIFY_STR_PTR_SCRATCH), dst=Register('r8')))
        instructions.append(Mov(src=Memory('rbp', _STRINGIFY_STR_LEN_SCRATCH), dst=Register('r9d')))
        instructions.extend(self._gen_stringify_bulk_append(Register('r8'), Register('r9d')))
        instructions.extend(self._gen_stringify_byte_append(Imm(ord("'"))))
        instructions.append(Jmp(done_label))

        instructions.append(Label(str_unquoted_label))
        instructions.append(MovQ(src=Memory('rbp', _STRINGIFY_STR_PTR_SCRATCH), dst=Register('r8')))
        instructions.append(Mov(src=Memory('rbp', _STRINGIFY_STR_LEN_SCRATCH), dst=Register('r9d')))
        instructions.extend(self._gen_stringify_bulk_append(Register('r8'), Register('r9d')))
        instructions.append(Jmp(done_label))

        # -- ARRAY ----------------------------------------------------------
        # value_addr points directly at the array's inline data (element
        # 0 at value_addr, element 1 at value_addr+elem_width, ...) --
        # no indirection to unwrap, since an array's bytes ARE its
        # storage. count comes from the type descriptor (a compile-
        # time-known array size).
        #
        # The type's name (e.g. "[3]int", read from the type
        # descriptor) is printed before the opening bracket at EVERY
        # level this runs, not just the outermost value, so a nested
        # array field or row shows its own type too.
        instructions.append(Label(array_label))
        instructions.append(MovQ(src=Memory('rbp', _STRINGIFY_TYPE_DESC), dst=Register('r10')))
        instructions.append(MovQ(src=Memory('r10', 8), dst=Register('rax')))
        instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', _STRINGIFY_FIELD_NAME_ADDR)))
        instructions.extend(self._gen_stringify_append_c_string_at(_STRINGIFY_FIELD_NAME_ADDR))
        instructions.extend(self._gen_stringify_byte_append(Imm(ord('['))))
        instructions.append(MovQ(src=Memory('rbp', _STRINGIFY_TYPE_DESC), dst=Register('r10')))
        instructions.append(MovQ(src=Memory('r10', 16), dst=Register('rax')))
        instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', _STRINGIFY_ELEM_TYPE_DESC)))
        instructions.append(MovQ(src=Memory('r10', 24), dst=Register('rax')))
        instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', _STRINGIFY_COLLECTION_LENGTH)))
        instructions.append(MovQ(src=Memory('r10', 32), dst=Register('rax')))
        instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', _STRINGIFY_ELEM_WIDTH)))
        instructions.extend(self._gen_stringify_collection_loop_body(
            base_addr_slot=_STRINGIFY_VALUE_ADDR,
        ))
        instructions.extend(self._gen_stringify_byte_append(Imm(ord(']'))))
        instructions.append(Jmp(done_label))

        # -- SLICE ------------------------------------------------------
        # Unlike ARRAY, value_addr here points at a runtime {ptr, len,
        # cap} DESCRIPTOR, not the backing data directly -- a slice is
        # an alias, one level of indirection from wherever its backing
        # array actually lives. So the base address for element
        # addressing is read out of that descriptor first
        # (_STRINGIFY_SLICE_BASE_PTR), and the length comes from the
        # descriptor's runtime len field, not the type descriptor (a
        # slice's length is a runtime property of the value, not its
        # static type) -- but the slice's NAME (e.g. "[]int") still
        # comes from the type descriptor, same as ARRAY, for the same
        # reason: printed at every level, not just once.
        instructions.append(Label(slice_label))
        instructions.append(MovQ(src=Memory('rbp', _STRINGIFY_TYPE_DESC), dst=Register('r10')))
        instructions.append(MovQ(src=Memory('r10', 8), dst=Register('rax')))
        instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', _STRINGIFY_FIELD_NAME_ADDR)))
        instructions.extend(self._gen_stringify_append_c_string_at(_STRINGIFY_FIELD_NAME_ADDR))
        instructions.extend(self._gen_stringify_byte_append(Imm(ord('['))))
        instructions.append(MovQ(src=Memory('rbp', _STRINGIFY_TYPE_DESC), dst=Register('r10')))
        instructions.append(MovQ(src=Memory('r10', 16), dst=Register('rax')))
        instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', _STRINGIFY_ELEM_TYPE_DESC)))
        instructions.append(MovQ(src=Memory('r10', 24), dst=Register('rax')))
        instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', _STRINGIFY_ELEM_WIDTH)))
        instructions.append(MovQ(src=Memory('rbp', _STRINGIFY_VALUE_ADDR), dst=Register('r10')))
        instructions.append(MovQ(src=Memory('r10', 0), dst=Register('rax')))
        instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', _STRINGIFY_SLICE_BASE_PTR)))
        instructions.append(MovQ(src=Memory('r10', 8), dst=Register('rax')))
        instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', _STRINGIFY_COLLECTION_LENGTH)))
        instructions.extend(self._gen_stringify_collection_loop_body(
            base_addr_slot=_STRINGIFY_SLICE_BASE_PTR,
        ))
        instructions.extend(self._gen_stringify_byte_append(Imm(ord(']'))))
        instructions.append(Jmp(done_label))

        # -- STRUCT -----------------------------------------------------
        # Format: `Point(x: 1, y: 2)` -- parentheses, not the square
        # brackets array/slice use, matching struct-literal syntax.
        # STRUCT's descriptor layout (see _get_or_build_type_
        # descriptor): [kind, name, field_count, then field_count
        # triples of (field_name_str_ptr, field_type_desc_ptr,
        # field_byte_offset), 24 bytes per triple] -- so field i's
        # triple sits at type_desc + 24 + i*24, read fresh from that
        # static data each loop iteration (no reason to cache it the
        # way ARRAY/SLICE cache length/width, since each iteration
        # needs a different triple anyway).
        #
        # The struct's own name (e.g. "Point") is printed first, before
        # the opening paren -- at every level this runs, so a struct
        # field that's itself a struct shows its own name too.
        instructions.append(Label(struct_label))
        instructions.append(MovQ(src=Memory('rbp', _STRINGIFY_TYPE_DESC), dst=Register('r10')))
        instructions.append(MovQ(src=Memory('r10', 8), dst=Register('rax')))
        instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', _STRINGIFY_FIELD_NAME_ADDR)))
        instructions.extend(self._gen_stringify_append_c_string_at(_STRINGIFY_FIELD_NAME_ADDR))
        instructions.extend(self._gen_stringify_byte_append(Imm(ord('('))))
        instructions.append(MovQ(src=Memory('rbp', _STRINGIFY_TYPE_DESC), dst=Register('r10')))
        instructions.append(MovQ(src=Memory('r10', 16), dst=Register('rax')))
        instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', _STRINGIFY_FIELD_COUNT)))

        struct_loop_start = self.new_label("stringify_struct_loop")
        struct_loop_done = self.new_label("stringify_struct_loop_done")
        struct_skip_sep = self.new_label("stringify_struct_skip_sep")

        instructions.append(MovQ(src=Imm(0), dst=Memory('rbp', _STRINGIFY_LOOP_INDEX)))
        instructions.append(Label(struct_loop_start))
        instructions.append(Mov(src=Memory('rbp', _STRINGIFY_LOOP_INDEX), dst=Register('eax')))
        instructions.append(Mov(src=Memory('rbp', _STRINGIFY_FIELD_COUNT), dst=Register('ecx')))
        instructions.append(Cmp(src=Register('ecx'), dst=Register('eax')))
        instructions.append(Jae(struct_loop_done))

        instructions.append(Cmp(src=Imm(0), dst=Memory('rbp', _STRINGIFY_LOOP_INDEX)))
        instructions.append(Je(struct_skip_sep))
        instructions.append(LeaQ(label=self._get_comma_space_label(), dst=Register('r8')))
        instructions.append(Mov(src=Imm(2), dst=Register('r9d')))
        instructions.extend(self._gen_stringify_bulk_append(Register('r8'), Register('r9d')))
        instructions.append(Label(struct_skip_sep))

        # field entry address = type_desc + 24 + loop_index*24
        instructions.append(Mov(src=Memory('rbp', _STRINGIFY_LOOP_INDEX), dst=Register('eax')))
        instructions.append(IMul(src=Imm(24), dst=Register('eax')))
        instructions.append(Add(src=Imm(24), dst=Register('eax')))
        instructions.append(MovQ(src=Memory('rbp', _STRINGIFY_TYPE_DESC), dst=Register('r10')))
        instructions.append(AddQ(src=Register('rax'), dst=Register('r10')))
        instructions.append(MovQ(src=Memory('r10', 0), dst=Register('rax')))
        instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', _STRINGIFY_FIELD_NAME_ADDR)))
        instructions.append(MovQ(src=Memory('r10', 8), dst=Register('rax')))
        instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', _STRINGIFY_FIELD_TYPE_DESC)))
        instructions.append(MovQ(src=Memory('r10', 16), dst=Register('rax')))
        instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', _STRINGIFY_FIELD_OFFSET)))

        # append the field's name -- length via strlen, like the
        # type-name appends above; field names are ordinary null-
        # terminated string literals rather than also carrying a
        # length field, so this costs one extra runtime call.
        instructions.extend(self._gen_stringify_append_c_string_at(_STRINGIFY_FIELD_NAME_ADDR))

        instructions.append(LeaQ(label=self._get_colon_space_label(), dst=Register('r8')))
        instructions.append(Mov(src=Imm(2), dst=Register('r9d')))
        instructions.extend(self._gen_stringify_bulk_append(Register('r8'), Register('r9d')))

        # field value address = value_addr + field_offset; ELEM_ADDR is
        # reused here rather than a dedicated slot, since it means the
        # same thing ARRAY/SLICE use it for ("address of a nested value
        # about to be recursed into"), and STRUCT never runs
        # concurrently with either.
        instructions.append(MovQ(src=Memory('rbp', _STRINGIFY_VALUE_ADDR), dst=Register('r10')))
        instructions.append(MovQ(src=Memory('rbp', _STRINGIFY_FIELD_OFFSET), dst=Register('rax')))
        instructions.append(AddQ(src=Register('rax'), dst=Register('r10')))
        instructions.append(MovQ(src=Register('r10'), dst=Memory('rbp', _STRINGIFY_ELEM_ADDR)))

        instructions.append(MovQ(src=Memory('rbp', _STRINGIFY_ELEM_ADDR), dst=Register('rdi')))
        instructions.append(MovQ(src=Memory('rbp', _STRINGIFY_FIELD_TYPE_DESC), dst=Register('rsi')))
        instructions.append(Mov(src=Imm(1), dst=Register('edx')))
        instructions.append(MovQ(src=Memory('rbp', _STRINGIFY_BUF_STATE_ADDR), dst=Register('rcx')))
        instructions.append(CallInstr('hornet_stringify'))

        instructions.append(Mov(src=Memory('rbp', _STRINGIFY_LOOP_INDEX), dst=Register('eax')))
        instructions.append(Add(src=Imm(1), dst=Register('eax')))
        instructions.append(Mov(src=Register('eax'), dst=Memory('rbp', _STRINGIFY_LOOP_INDEX)))
        instructions.append(Jmp(struct_loop_start))
        instructions.append(Label(struct_loop_done))

        instructions.extend(self._gen_stringify_byte_append(Imm(ord(')'))))
        instructions.append(Jmp(done_label))

        instructions.append(Label(done_label))
        instructions.append(Leave())
        instructions.append(Ret())
        return AsmFunction(name='hornet_stringify', instructions=instructions)

    def gen_print_call_into(self, expr: Call, dst: Operand) -> list[Instruction]:
        """`print(x)`: dispatches on x's *compile-time* type -- known
        exactly, since Hornet is statically typed. Every type
        (including struct) goes through the same uniform pipeline:

          1. Allocate a small (16-byte) growable buffer and set up its
             {ptr, len, cap} state.
          2. Compute value_addr: the address of x's value.
          3. Look up (or lazily build) x's type's runtime descriptor
             (see _get_or_build_type_descriptor).
          4. Call hornet_stringify(value_addr, type_desc,
             quote_strings=0, &buf_state) -- the single, recursive
             function that turns any value, of any type, into bytes
             appended onto that buffer. quote_strings=0 here
             specifically because x is the OUTERMOST value of this
             print() call: a bare str prints unquoted ("hello", not
             'hello'); only a str reached by recursing into a
             collection or struct gets quoted.
          5. Append a trailing newline.
          6. write() the buffer's content to stdout, then free() it.

        Step 1 deliberately runs BEFORE step 2: malloc is free to
        clobber any caller-saved register, so computing value_addr
        first and keeping it alive across this malloc call would
        repeat the class of bug already found in
        gen_buffer_append_bytes_into -- computing it AFTER means
        nothing here needs to survive a call in a register at all.

        value_addr itself, depending on x's type:
          - A bare Variable of ANY type: that variable's existing
            storage, addressed directly -- no copy made.
          - A non-Variable int/bool/str/int64 expression (`print(a +
            1)`): evaluated into a small, shared scratch slot
            (_print_scalar_temp_offset, reserved unconditionally per
            function), then that slot's address is used. Safe as a
            shared slot because one print() call's argument is fully
            consumed before another print() call could reuse it.
          - An ARRAY expression that isn't a Variable/Index/Field (an
            ArrayLiteral, or an array-returning Call): materialized
            via _gen_materialize_argument_temp_into, the same
            mechanism already built for an array literal/call used as
            an ordinary function-call argument.
          - A SLICE expression that isn't a Variable/Index/Field:
            materialized via gen_slice_value_into into the same shared
            _unnamed_slice_temp_offset scratch slot gen_indexable_
            base_into's analogous cases use.
          - A non-Variable STRUCT expression: NOT supported --
            restricted to a Variable, Field, or Index. Assign it to a
            named variable first.
        """
        if dst != Register('eax'):
            raise CodegenError(f"Call codegen requires dst == %eax, got: {dst!r}")

        self._print_used = True
        arg = expr.args[0]
        arg_type = type_of(arg)
        buf_state_offset = self._print_buf_state_temp_offset

        instructions: list[Instruction] = []

        # Step 1: allocate the initial buffer and set up buf_state --
        # see this method's own docstring for why this runs FIRST.
        instructions.append(Mov(src=Imm(16), dst=Register('edi')))
        instructions.append(CallInstr('malloc'))
        instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', buf_state_offset)))
        instructions.append(MovQ(src=Imm(0), dst=Memory('rbp', buf_state_offset + 8)))
        instructions.append(MovQ(src=Imm(16), dst=Memory('rbp', buf_state_offset + 16)))

        # Step 2: compute value_addr into %r10 (caller-saved -- it's
        # consumed immediately below into %rdi, never surviving a call
        # of its own, so it doesn't need to be callee-saved).
        value_addr_reg = Register('r10')
        if arg_type.kind == TypeKind.STRUCT:
            if not isinstance(arg, (Variable, Field, Index)):
                raise CodegenError(
                    "print's argument must be a variable, field access, "
                    "or indexing expression when it's struct-typed -- "
                    "assign it to a variable first"
                )
            instructions.extend(self.gen_struct_address_into(arg, value_addr_reg))
        elif arg_type.kind == TypeKind.ARRAY:
            if isinstance(arg, (Variable, Field, Index)):
                instructions.extend(self.gen_array_address_into(arg, value_addr_reg))
            else:
                instructions.extend(self._gen_materialize_argument_temp_into(arg, arg_type, value_addr_reg))
        elif arg_type.kind == TypeKind.SLICE:
            if isinstance(arg, Variable):
                instructions.append(LeaQFrame(offset=self._local_offset(arg.name), dst=value_addr_reg))
            elif isinstance(arg, Field):
                instructions.extend(self.gen_field_address_into(arg, value_addr_reg))
            elif isinstance(arg, Index):
                instructions.extend(self.gen_index_address_into(arg, value_addr_reg))
            else:
                # A Slice expression or slice-returning Call has no
                # existing descriptor to address directly, so
                # materialize one via gen_slice_value_into into the
                # same shared _unnamed_slice_temp_offset scratch slot
                # gen_indexable_base_into's analogous cases use, then
                # take that slot's address. No new storage-reservation
                # needed: a slice descriptor is always exactly 24
                # bytes, so the existing shared slot already fits it.
                temp_offset = self._unnamed_slice_temp_offset
                instructions.extend(self.gen_slice_value_into(arg, Memory('rbp', temp_offset)))
                instructions.append(LeaQFrame(offset=temp_offset, dst=value_addr_reg))
        elif isinstance(arg, Variable):
            instructions.append(LeaQFrame(offset=self._local_offset(arg.name), dst=value_addr_reg))
        else:
            # int/int64/bool/str, not a bare Variable -- materialize
            # into the shared scalar scratch slot (used at its full
            # 8-byte width regardless of the value's actual width, so
            # the same slot serves int/bool -- 4 bytes -- and
            # str/int64 -- 8 -- uniformly), then take that slot's
            # address. int64 needs the same full 8-byte MovQ str
            # already gets here; an ordinary 4-byte Mov would discard
            # its high 32 bits, leaving stale data in the slot's upper
            # half.
            scratch_offset = self._print_scalar_temp_offset
            instructions.extend(self.gen_expr_into(arg, Register('eax')))
            if arg_type == Type.STR or arg_type == Type.INT64:
                instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', scratch_offset)))
            else:
                instructions.append(Mov(src=Register('eax'), dst=Memory('rbp', scratch_offset)))
            instructions.append(LeaQFrame(offset=scratch_offset, dst=value_addr_reg))

        # Step 3: look up (or lazily build) x's type descriptor -- a
        # pure compile-time operation, emits no instructions of its
        # own, just returns a label name.
        desc_label = self._get_or_build_type_descriptor(arg_type, {})

        # Step 4: call hornet_stringify(value_addr, type_desc,
        # quote_strings=0, &buf_state).
        instructions.append(MovQ(src=value_addr_reg, dst=Register('rdi')))
        instructions.append(LeaQ(label=desc_label, dst=Register('rsi')))
        instructions.append(Mov(src=Imm(0), dst=Register('edx')))
        instructions.append(LeaQFrame(offset=buf_state_offset, dst=Register('rcx')))
        instructions.append(CallInstr('hornet_stringify'))

        # Step 5: append a trailing newline -- load buf_state fresh
        # (hornet_stringify's own recursive calls may have grown and
        # relocated it any number of times by now), append, write back.
        instructions.append(MovQ(src=Memory('rbp', buf_state_offset), dst=Register('rbx')))
        instructions.append(Mov(src=Memory('rbp', buf_state_offset + 8), dst=Register('r12d')))
        instructions.append(Mov(src=Memory('rbp', buf_state_offset + 16), dst=Register('r13d')))
        instructions.extend(self.gen_buffer_append_byte_into(
            Register('rbx'), Register('r12'), Register('r12d'),
            Register('r13'), Register('r13d'), Imm(10),  # '\n'
        ))
        instructions.append(MovQ(src=Register('rbx'), dst=Memory('rbp', buf_state_offset)))
        instructions.append(Mov(src=Register('r12d'), dst=Memory('rbp', buf_state_offset + 8)))
        instructions.append(Mov(src=Register('r13d'), dst=Memory('rbp', buf_state_offset + 16)))

        # Step 6: write(1, buf.ptr, buf.len) to stdout -- an explicit
        # length, not a null-terminated C string, since the buffer's
        # content can legitimately contain embedded NUL bytes (e.g. a
        # str field inside a printed struct), which would truncate a
        # puts() call but doesn't affect write().
        instructions.append(MovQ(src=Memory('rbp', buf_state_offset), dst=Register('rsi')))
        instructions.append(Mov(src=Memory('rbp', buf_state_offset + 8), dst=Register('edx')))
        instructions.append(Mov(src=Imm(1), dst=Register('edi')))
        instructions.append(CallInstr('write'))

        # Step 7: free the buffer, exactly once, regardless of how many
        # times hornet_stringify relocated it in between -- buf_state's
        # ptr field always holds whichever block is currently live.
        instructions.append(MovQ(src=Memory('rbp', buf_state_offset), dst=Register('rdi')))
        instructions.append(CallInstr('free'))

        instructions.append(Mov(src=Imm(0), dst=dst))
        return instructions

    def gen_string_literal_into(self, expr: StringLiteral, dst: Operand) -> list[Instruction]:
        """Registers this literal's content for later emission as
        static `.data`, and loads its address into `dst`. Every
        occurrence gets its own fresh label, even for identical
        content -- no deduplication, a bit wasteful but keeps this a
        one-line append rather than needing a content->label cache."""
        label = self.new_label("str")
        self.string_literals.append((label, expr.value))
        return [LeaQ(label=label, dst=as_qword_register(dst))]

    def gen_string_concat_into(self, expr: Binary, dst: Operand) -> list[Instruction]:
        """`left + right`, both str: builds a brand-new, malloc'd,
        null-terminated buffer holding left's bytes immediately
        followed by right's -- `strlen(left) + strlen(right) + 1`
        bytes, then `strcpy` then `strcat`.

        `left` is protected across evaluating `right` by pushing it
        onto the stack -- the same push-before-recursing scheme
        gen_binary_into uses -- rather than stashing it in a fixed
        register like %rbx. That distinction matters here specifically
        because `right` can itself be another string concatenation or
        comparison: if `left` were sitting in %rbx while `right` gets
        evaluated, and `right`'s own evaluation needs %rbx for its own
        left/right dance too, it would silently clobber `left`. The
        stack has no such fixed-identity conflict, however deep the
        nesting.

        %r12 (holding `right`), by contrast, *is* safe to set as a
        fixed register once `right`'s evaluation completes: nothing
        after that point recurses back into gen_expr_into, so there's
        no nested evaluation left to clobber it -- only the fixed
        strlen/malloc/strcpy/strcat sequence below runs, and libc is
        itself SysV-ABI-compliant, so it preserves %r12 on its own.
        %r13/%r14 are similarly only ever written once, from a direct
        call result, with nothing recursive happening afterward.

        This is a *distinct* concern from why every function's own
        prologue/epilogue also saves/restores %rbx/%r12/%r13/%r14 (see
        gen_function and gen_return): that fix protects a value held
        in one of these registers *across a call into another
        function*; this one protects a value held here *across
        evaluating a nested expression within the same function*. Both
        are needed; neither replaces the other.

        MEMORY: once an operand's bytes have been fully copied out
        (strcpy for left, strcat for right), if that operand was
        itself a fresh, unnamed concatenation result -- a Binary(ADD,
        ...) sub-expression, never a named variable, literal, or
        function call's return value -- its buffer is immediately
        freed. See _gen_free_if_fresh_concat for why this narrow case
        is safe to free automatically with no broader escape analysis.
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

    def _gen_free_if_fresh_concat(self, operand: Node, holding_register: str) -> list[Instruction]:
        """If `operand` is itself a Binary(ADD, ...) node -- meaning
        whatever's in `holding_register` is a fresh buffer
        gen_string_concat_into just malloc'd for *this* expression
        alone, which could never have been stored into a variable,
        returned, or passed as an argument -- frees it. Everything
        else is left alone: a StringLiteral points into static `.data`
        and was never heap-allocated (freeing it would corrupt the
        allocator); a Variable or a Call's return value might be
        aliased by code we have no visibility into here -- telling
        those apart from a genuinely fresh, exclusively-owned buffer
        is a real escape-analysis problem this narrow check
        deliberately doesn't attempt to solve."""
        if isinstance(operand, Binary) and operand.op == BinaryOp.ADD:
            return [
                MovQ(src=Register(holding_register), dst=Register('rdi')),
                CallInstr('free'),
            ]
        return []

    def gen_string_compare_into(self, expr: Binary, dst: Operand) -> list[Instruction]:
        """`left == right` / `left != right`, both str: calls `strcmp`
        (0 means equal) and converts that into this language's usual
        0/1 bool representation via the same cmp/setCC/movzx pattern
        every other comparison uses, reusing
        _COMPARISON_CONDITION_CODES[op] directly.

        `left` is protected across evaluating `right` via the stack,
        for the same reason gen_string_concat_into does.

        MEMORY: the same fresh-concatenation-result freeing
        gen_string_concat_into does, and for the same reason -- strcmp
        has already read both operands' bytes by the time this frees
        them. The one thing to get right here that concatenation
        didn't: `call free` clobbers %eax exactly like any other call,
        and strcmp's result is *sitting* in %eax at this point -- so it
        has to be stashed in a callee-saved register before either
        free() call, and restored into `dst` afterward, or freeing a
        fresh operand would silently destroy the comparison result
        this method exists to compute."""
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
        instructions.append(SetCC(cc=COMPARISON_CONDITION_CODES[expr.op], operand=byte_dst))
        instructions.append(MovZX(src=byte_dst, dst=dst))
        return instructions
