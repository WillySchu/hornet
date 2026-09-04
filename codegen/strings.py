"""TODO"""

from codegen.assembly_ast import (
    Operand,
    Instruction,
    Register,
    Mov,
    Cmp,
    Imm,
    Je,
    Jg,
    Neg,
    Jmp,
    Label,
    LeaQFrame,
    Div,
    Add,
    SubQ,
    MovB,
    Memory,
    Jne, MovQ,
    CmpQ,
    NegQ,
    DivQ,
    AsmFunction,
    Ret,
    Leave,
    CallInstr,
    AddQ,
    LeaQ,
    IMul,
    Jae,
    MovSX,
    MovZX,
    Push,
    SetCC,
    Pop,
)
from codegen.errors import CodegenError
from codegen.utils import as_byte_register, type_byte_width, type_of, as_qword_register, COMPARISON_CONDITION_CODES
from parser import Call, Variable, Field, Index, StringLiteral, Binary, Node, BinaryOp
from semantic import Type, TypeKind


# Kind tags for the print machinery's own runtime type descriptors --
# see CodeGenerator._get_or_build_type_descriptor and AsmProgram's own
# type_descriptors field. Plain small ints, embedded directly as a
# descriptor's own first .quad field; the hand-built stringify runtime
# function (see PRINTING) switches on this to decide how to interpret
# everything that follows it in that same descriptor.
_TYPEDESC_INT = 0
_TYPEDESC_BOOL = 1
_TYPEDESC_STR = 2
_TYPEDESC_ARRAY = 3
_TYPEDESC_SLICE = 4
_TYPEDESC_STRUCT = 5
_TYPEDESC_INT8 = 6
_TYPEDESC_UINT8 = 7
_TYPEDESC_INT64 = 8


# Fixed %rbp-relative local-slot layout for the hand-built hornet_
# stringify runtime function (see CodeGenerator.build_stringify_
# function) -- every local this function ever needs, across every
# branch, gets its own distinct 8-byte slot (16 for the itoa scratch),
# never reused between branches even though ARRAY/SLICE and STRUCT are
# mutually exclusive within one call: stack space is cheap, and giving
# every value an unambiguous, permanent name is worth far more than
# the handful of bytes saved by overlapping slots across branches that
# never actually run together. All are spilled-to-immediately-on-entry
# and reloaded-as-needed throughout, rather than kept live in
# registers across this function's own recursive calls to itself --
# registers don't survive a nested call the way a value in a fixed
# %rbp-relative slot does, and getting that distinction wrong (keeping
# something "cached" in a register across a call that might itself
# need that same register) is exactly the shape of bug that cost real
# time earlier in this same effort (see gen_buffer_append_bytes_into's
# own %ecx-across-malloc story) -- so here, deliberately, NOTHING
# stays in a register across a call unless that call's own contract
# explicitly guarantees it (a callee-saved register, used the same way
# gen_append_call_into already relies on malloc preserving %rbx/%r12/
# %r13).
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
_STRINGIFY_SLICE_BASE_PTR = -128    # a slice's own backing-array pointer, read from value_addr's
# own runtime {ptr, len, cap} descriptor -- ARRAY's own
# value_addr already points AT the data directly, but SLICE's
# own value_addr points at a descriptor one level of
# indirection away, so this needs its own slot
_STRINGIFY_FIELD_TYPE_DESC = -136
_STRINGIFY_FIELD_OFFSET = -144
_STRINGIFY_ITOA_SCRATCH_64 = -176   # 32 bytes: -176 .. -145 -- a SEPARATE,
# wider buffer for int64's own decimal
# conversion, not a resize of _STRINGIFY_
# ITOA_SCRATCH above: int64's own max
# magnitude (9223372036854775808) is 19
# digits, needing 20 characters with a
# leading '-', comfortably more than the
# 16-byte buffer int/int8/uint8 already
# share (max 11 characters, for a 32-bit
# value) -- kept as its own, separate
# slot rather than widening the shared
# one in place, so every existing offset
# below it stays untouched.
_STRINGIFY_FRAME_SIZE = 176  # already a multiple of 16; see build_stringify_function's own alignment note


class StringsMixin:
    def gen_int_to_decimal_into(self, value: Operand, scratch_offset: int, scratch_size: int) -> list[Instruction]:
        """Converts a 32-bit signed int into its own decimal ASCII
        representation, written into a caller-provided scratch buffer
        at a fixed %rbp-relative offset (Memory('rbp', scratch_offset)
        .. Memory('rbp', scratch_offset + scratch_size - 1) --
        scratch_size must be at least 11, enough for any 32-bit
        value's digits plus a leading '-'; 16 gives comfortable
        margin) -- the print machinery's own first real numeric-to-
        text conversion, needed because everything now funnels through
        one buffer-building mechanism rather than reaching for printf's
        own '%d' formatting the way every int print used to (see the
        module docstring's PRINTING section for the full story of why
        that shortcut no longer applies).

        Builds digits into the scratch buffer from the END backward
        (least-significant digit first, naturally, since that's the
        order repeated division by 10 produces them in) rather than
        forward-then-reversed -- one pass, no separate reversal step.

        Handles the INT_MIN edge case correctly without ever widening
        to 64 bits: negating INT_MIN in ordinary 32-bit two's
        complement leaves its bit pattern completely unchanged (there's
        no positive counterpart to negate to, so the negation
        "overflows" right back to the same bits) -- but that SAME bit
        pattern, reinterpreted as UNSIGNED rather than signed, is
        exactly INT_MIN's own correct magnitude (2147483648), a value
        that doesn't fit in a signed 32-bit int at all but fits an
        unsigned one perfectly. So after peeling off the sign (checked
        BEFORE any negation happens, via the same Cmp-against-0 used
        for the zero/positive/negative three-way split below) and
        negating, the digit-extraction loop divides that magnitude
        with Div (see its own docstring), not IDiv -- treating it as
        unsigned throughout is what makes the INT_MIN case correct
        with no special-casing beyond the ordinary negate-and-loop
        every other negative value already needs.

        On return: %r8 holds the address of the first character
        (which may be partway into the scratch buffer, not
        necessarily its start, since digits are written backward from
        the end), and %r9d holds the character count, including a
        leading '-' if the value was negative -- ready to hand
        directly to gen_buffer_append_bytes_into as (source_addr,
        count). Fixed internal scratch beyond %r8/%r9d: %eax (the
        value being divided, then the quotient), %edx (the remainder),
        %ecx (holds the constant 10), %r10 (the write-position
        pointer, decremented as each digit is written, and the running
        flag for whether a '-' still needs writing). Callers must
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
        """The int64 counterpart to gen_int_to_decimal_into just above
        -- see its own docstring for the full algorithm this mirrors
        exactly, one register width up, rather than repeating that
        explanation here. `value` is expected to already be a 64-bit
        operand (a Memory location or a Register already holding the
        full 64-bit value -- e.g. Register('rax'), not Register('eax')
        -- unlike the 32-bit version's own `value` parameter).

        scratch_size must be at least 21: int64's own max magnitude
        (9223372036854775808, INT64_MIN's own magnitude) is 19 digits,
        needing 20 characters plus a leading '-'. The dedicated
        _STRINGIFY_ITOA_SCRATCH_64 buffer (32 bytes, a SEPARATE slot
        from the 32-bit version's own 16-byte one -- see its own
        definition for why) gives comfortable margin, the same
        "more than strictly needed" choice the 32-bit buffer already
        makes.

        Every instruction here operates on the 64-bit VIEW of the
        exact same registers gen_int_to_decimal_into uses (%rax not
        %eax, %rdx not %edx, %rcx not %ecx; %r10/%r8/%r9d are
        unchanged, since %r10/%r8 already hold addresses -- always
        64-bit regardless -- and %r9d, the digit count, never needs to
        exceed a small two-digit number even for int64's own longest
        possible string), via DivQ/NegQ/CmpQ/MovQ rather than
        Div/Neg/Cmp/Mov -- correctly handling INT64_MIN the identical
        way the 32-bit version handles INT_MIN (see Div's own
        docstring, and DivQ's own): its negation leaves the bit
        pattern unchanged, but that pattern, read as unsigned via
        DivQ, correctly represents its own magnitude, a value that
        doesn't fit in a signed int64 at all but fits an unsigned
        64-bit divide perfectly.

        On return: %r8 holds the address of the first character, %r9d
        holds the character count -- the identical contract gen_int_
        to_decimal_into's own docstring specifies, ready to hand
        directly to gen_buffer_append_bytes_into the same way."""
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
        """Returns the label of t's own runtime type descriptor,
        building and registering it into self.type_descriptors if it
        hasn't already been built WITHIN THIS ONE CALL (in_progress,
        keyed on t itself -- Type is frozen, giving it correct
        structural/nominal equality and hashing for free, so it's
        already a safe, correct dict key with no extra work).

        `in_progress` is deliberately scoped to a single top-level
        call into this method (one fresh, empty dict per print() call
        site that needs a descriptor tree at all -- see gen_print_
        call_into), NOT a whole-program cache: two separate print()
        calls on the same struct type each build their OWN, separately
        allocated, complete descriptor tree from scratch, mirroring
        gen_string_literal_into's own explicit "no cross-occurrence
        deduplication, accept the waste, keep this simple" choice (see
        its own docstring) for the identical reason.

        Reuse WITHIN one call, though, is not optional -- it's the
        only way a self-referential struct (`struct Node: int value;
        []Node children`, a real, legal pattern once slice-typed
        fields were supported -- see the STRUCTS section) can even be
        represented as a finite amount of static data at all: without
        it, building Node's own descriptor would need Node's own
        descriptor, forever. Reserving this type's own label BEFORE
        recursing into anything it contains (fields, element types) is
        what breaks that cycle -- a nested reference back to the same
        type finds its own label already in in_progress and reuses it
        directly, rather than trying to build it a second time.

        Every non-leaf kind (ARRAY, SLICE, STRUCT) carries its own
        "type name" (e.g. "[3]int", "[]int", or a struct's own
        declared name like "Point") as a second field, right after the
        KIND tag -- a plain string literal, computed once here at
        build time via str(t) for ARRAY/SLICE or t.struct_name for
        STRUCT, registered exactly like a struct field's own name is
        just below. hornet_stringify prints this name immediately
        before that kind's own opening bracket/brace at EVERY level it
        appears, not just the outermost one a print() call names
        directly -- so a struct field that's itself an array, or an
        array element that's itself a struct, shows its own type too,
        e.g. `(row: [3]int[1, 2, 3])` or `[Point(x: 1, y: 2), Point(x:
        3, y: 4)]`. INT/INT8/UINT8/BOOL/STR carry no name field at all
        (there's nothing useful to prefix a bare int or string with),
        so this is a genuine per-kind layout difference reflected
        directly in each branch below, not a uniform field every
        descriptor has.
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
        """A shared, static, empty ("") string constant -- str's own
        zero value (see _gen_zero_value_into). Deliberately NOT a null
        pointer: every string operation this file generates (print,
        concatenation, gen_string_compare_into's own strcmp call)
        dereferences a str value with no null check at all, so a
        zero-initialized str field or variable has to point at a real,
        valid (if empty) C string -- pointing at nothing would segfault
        the instant anything touched it, the first time any Hornet
        program actually exercised implicit zero-initialization for a
        str."""
        if self._empty_str_label is None:
            self._empty_str_label = self.new_label("empty_str")
            self.string_literals.append((self._empty_str_label, ""))
        return self._empty_str_label

    def _gen_stringify_bulk_append(self, source_addr: Register, count: Operand) -> list[Instruction]:
        """Loads the print buffer's own {ptr, len, cap} triple from
        _STRINGIFY_BUF_STATE_ADDR, bulk-appends `count` bytes from
        `source_addr` via gen_buffer_append_bytes_into, then writes the
        resulting triple back out -- the load-append-store sequence
        every single multi-byte append inside hornet_stringify's own
        body needs (a whole string's own content, converted decimal
        digits, a two-byte separator like ", " or ": "), factored out
        once, rather than re-derived by hand at each of the many call
        sites that need it, exactly the kind of duplication that's
        cheap to get subtly wrong the third or fourth time it's
        transcribed rather than reused. Only ever valid to call from
        within hornet_stringify's own body -- it hardcodes that
        function's fixed frame layout (_STRINGIFY_BUF_STATE_ADDR).

        `source_addr` and `count` must not be %rbx/%r12/%r13/%r11
        (clobbered loading/storing the buffer state itself here) or any
        of gen_buffer_append_bytes_into's own internal scratch (%rax/
        %rcx/%rdx/%rdi/%r10/%r11 -- see its own docstring). %r8/%r9 are
        always safe against both, which is exactly why gen_int_to_
        decimal_into's own contract (and this function's own BOOL case)
        both deliberately leave their (address, count) result there."""
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
        constraints as its own sibling; see its docstring."""
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
        """strlen()s the null-terminated string whose own address has
        already been stored at Memory('rbp', name_addr_offset), then
        bulk-appends exactly that many bytes onto the buffer -- shared
        by every place hornet_stringify prints a compile-time-known
        NAME (a type's own name, e.g. "[3]int" or a struct's own
        declared name, or a struct field's own name) rather than a
        runtime VALUE: none of these carry their own length anywhere
        in a type descriptor's own static data (unlike the fixed
        2-byte ", "/": " separators, which use a plain Imm(2) instead
        of this), so strlen is what supplies it.

        Reads the address from `name_addr_offset` (Memory, not a
        register) both before AND after the intervening `call
        strlen`, since a real function call is free to clobber any
        caller-saved register but never touches this function's own
        stack frame -- the same "spill across a call, never trust a
        register to survive one" discipline established throughout
        this file (see e.g. gen_buffer_append_bytes_into's own
        docstring for the bug this exact pattern was written to
        avoid)."""
        instructions = []
        instructions.append(MovQ(src=Memory('rbp', name_addr_offset), dst=Register('rdi')))
        instructions.append(CallInstr('strlen'))
        instructions.append(MovQ(src=Memory('rbp', name_addr_offset), dst=Register('r8')))
        instructions.append(Mov(src=Register('eax'), dst=Register('r9d')))
        instructions.extend(self._gen_stringify_bulk_append(Register('r8'), Register('r9d')))
        return instructions

    def _gen_stringify_collection_loop_body(self, base_addr_slot: int) -> list[Instruction]:
        """Shared by the ARRAY and SLICE dispatch cases in build_
        stringify_function: given _STRINGIFY_COLLECTION_LENGTH/_
        STRINGIFY_ELEM_WIDTH/_STRINGIFY_ELEM_TYPE_DESC already
        populated by the caller, and `base_addr_slot` naming whichever
        local slot holds the address of ELEMENT 0 (ARRAY's own
        _STRINGIFY_VALUE_ADDR directly, since an array's own bytes ARE
        its own storage; SLICE's own _STRINGIFY_SLICE_BASE_PTR, the
        backing pointer already unwrapped one level of indirection
        away from value_addr's own runtime descriptor) -- emits the
        loop that produces `[elem, elem, ...]`'s own INSIDE (the
        caller appends the brackets themselves, before and after
        calling this): a ", " separator before every element but the
        first, then a recursive call into hornet_stringify itself for
        each element's own value, with quote_strings hardcoded to 1 --
        an array/slice element is, by definition, never the outermost
        value of an entire print() call.

        Element addresses are computed via the identical 32-bit-
        multiply-then-zero-extend idiom gen_index_address_into already
        establishes for ordinary array/slice indexing (see its own
        docstring's note on exactly this) -- safe here for the
        identical reason: an element count or index this language
        could ever actually construct is always far too small to make
        a 32-bit multiply's own implicit zero-extension incorrect."""
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
        """Builds `hornet_stringify`, the print machinery's own hand-
        built runtime function -- built ONCE per program (not derived
        from any Hornet AST Function at all, and not tied to any one
        print() call site), added directly to AsmProgram.functions
        alongside every ordinary Hornet-compiled function, and called
        via a completely ordinary `call hornet_stringify` from gen_
        print_call_into. This is the first hand-built function in this
        compiler that needs GENUINE, unbounded recursion (calling
        itself, via a real `call`, once per array/slice element or
        struct field) rather than instructions inlined at each call
        site -- the only way a self-referential struct (`struct Node:
        int value; []Node children`) can ever be printed at all, since
        the recursion depth needed depends on a VALUE's own runtime
        shape (how deep the tree actually is), which no amount of
        compile-time code generation could ever unroll for -- see the
        module docstring's PRINTING section for the full argument.

        SIGNATURE (an ordinary SysV integer-argument call, exactly as
        if this were declared `void hornet_stringify(void* value_addr,
        void* type_desc, long quote_strings, void* buf_state_addr)`):
          %rdi = value_addr     -- address of the value being printed
          %rsi = type_desc      -- pointer to its own type descriptor
          %rdx = quote_strings  -- 0 or 1; whether a STR value here
                                    should be wrapped in single quotes
                                    (never true for the OUTERMOST call
                                    from gen_print_call_into, always
                                    true for every recursive call this
                                    function makes to itself -- see the
                                    module docstring's own note on
                                    quoting a str inside a collection)
          %rcx = buf_state_addr -- pointer to a 24-byte {ptr, len, cap}
                                    block, OWNED by gen_print_call_into
                                    (not this function), read from and
                                    written back to on every append,
                                    and by every recursive call this
                                    function makes to itself, so growth
                                    or reallocation anywhere in a deeply
                                    nested value is visible everywhere
                                    else building the SAME print's own
                                    output

        Every incoming argument is spilled into its own fixed local
        slot immediately on entry (see the _STRINGIFY_* offset
        constants just above) and reloaded from there whenever needed,
        rather than kept live in a register across this function's own
        recursive calls to itself or its calls to malloc (via the
        buffer-append primitives) -- registers don't survive a nested
        call the way a value in a fixed stack slot does; see those
        constants' own shared docstring for why this discipline is
        non-negotiable here specifically.

        Dispatches on type_desc's own first field (the kind tag) via
        an ordinary chain of comparisons -- nine kinds as of int64
        (originally six, then eight with int8/uint8), still not enough
        to justify a jump table over a plain comparison chain. The kind
        tag is stored as a full .quad but
        compared via its own 32-bit view (e.g. %r10d): every kind
        value is small and non-negative, so the upper 32 bits are
        always zero, and Cmp is fixed at 32-bit (`cmpl`) -- matching
        it rather than introducing a new 64-bit compare instruction
        just for this.

        THIS METHOD IS BUILT INCREMENTALLY, one kind at a time, each
        verified independently before the next is added -- see the
        surrounding commit history/conversation for exactly which
        kinds are implemented as of any given point; an unimplemented
        kind's own dispatch target does not yet exist, and building
        this function while any dispatch case is missing is expected
        to be a WORK-IN-PROGRESS state, not a finished one.
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
        # %r8 = digits start address, %r9d = digit count (gen_int_to_
        # decimal_into's own contract) -- both survive untouched into
        # _gen_stringify_bulk_append, which never uses %r8/%r9 (see its
        # own docstring for exactly why that's guaranteed, not assumed).
        instructions.extend(self._gen_stringify_bulk_append(Register('r8'), Register('r9d')))
        instructions.append(Jmp(done_label))

        # -- INT8/UINT8 -----------------------------------------------------
        # Identical to INT just above except for ONE instruction: value_
        # addr points at a genuinely 1-byte-wide value (see type_byte_
        # width), so reading it needs a WIDENING load -- MovSX (sign-
        # extend) for int8, MovZX (zero-extend) for uint8 -- rather than
        # INT's own ordinary 4-byte Mov, which would read three bytes of
        # adjacent memory that were never part of this value at all, and
        # for int8 specifically would also silently misinterpret a
        # negative value as a large positive one. Once correctly widened
        # into %eax, the exact same gen_int_to_decimal_into/bulk-append
        # sequence INT already uses produces the correct decimal string
        # for either -- there is no separate "narrow int to decimal"
        # algorithm anywhere in this file, because there doesn't need to
        # be one: a correctly sign- or zero-extended 32-bit value's own
        # decimal representation is computed identically regardless of
        # how many bits it started out occupying in memory.
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
        # int64's own value_addr points at a genuinely 8-byte-wide value
        # (see type_byte_width), needing a full-width MovQ read into
        # %rax rather than INT's own 4-byte Mov into %eax, which would
        # silently drop the value's own high 32 bits entirely -- unlike
        # int8/uint8's own narrowing case, this can't reuse gen_int_to_
        # decimal_into at all (that routine is written entirely in
        # 32-bit instructions throughout, not just at the read), so this
        # calls the dedicated gen_int64_to_decimal_into instead, into
        # its own, separately-sized _STRINGIFY_ITOA_SCRATCH_64 buffer
        # (see that constant's own docstring for why it's a separate
        # slot, not a resize of the shared 16-byte one).
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
        # A str VALUE is itself a pointer (see the module docstring's
        # STRINGS section) -- so, unlike INT/BOOL, value_addr here holds
        # the address of a POINTER, and that pointer itself (once
        # dereferenced) is what needs stringifying, not value_addr's own
        # bytes directly. quote_strings decides whether it's wrapped in
        # single quotes (nested inside a collection or struct) or left
        # bare (the outermost value of an entire print() call) -- see
        # the module docstring's own note on this being a real,
        # deliberate, CONTEXT-dependent distinction, not an oversight.
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
        # value_addr already points DIRECTLY at the array's own inline
        # data (element 0 immediately at value_addr, element 1 at
        # value_addr+elem_width, ...) -- unlike SLICE below, there's no
        # extra indirection to unwrap, since an array's own bytes ARE
        # its own storage, matching how ordinary array indexing already
        # works everywhere else in this compiler. count comes straight
        # from the type descriptor (a compile-time-known array size,
        # just stored as ordinary runtime data here like everything
        # else in a descriptor).
        #
        # The type's own name (e.g. "[3]int", read fresh from the type
        # descriptor's own second field) is printed first, before the
        # opening bracket -- at EVERY level this ever runs, nested or
        # not, not just when this is the outermost value a print() call
        # names directly, so an array field inside a struct, or an
        # array-of-arrays' own inner row, shows its own type too.
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
        # cap} DESCRIPTOR, not at the backing data directly -- a slice
        # is an alias, and value_addr is the address of the slice
        # VARIABLE's own three fields, one level of indirection away
        # from wherever its own backing array actually lives (see the
        # module docstring's STRINGS -- no, SLICES section for the full
        # {ptr,len,cap} design). So the base address for element
        # addressing has to be READ out of that descriptor first
        # (_STRINGIFY_SLICE_BASE_PTR), and the length comes from the
        # descriptor's own runtime len field, not from the type
        # descriptor at all (a slice's length is a runtime property of
        # the VALUE, never part of its own static type) -- but the
        # slice's own NAME (e.g. "[]int") still comes from the type
        # descriptor, same as ARRAY just above, and for the identical
        # reason: printed at every level this runs, not just once.
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
        # Format: `Point(x: 1, y: 2)` -- see the module docstring's own
        # note on why parentheses, not the square brackets array/slice
        # already use (matching the eventual struct-literal syntax,
        # not just a printing-only convention invented in isolation).
        # STRUCT's own descriptor layout (see _get_or_build_type_
        # descriptor): [kind, name, field_count, then field_count
        # triples of (field_name_str_ptr, field_type_desc_ptr,
        # field_byte_offset), 24 bytes per triple] -- so field i's own
        # triple sits at type_desc + 24 + i*24, read fresh out of that
        # STATIC data on every loop iteration (there's no reason to
        # cache it across iterations the way collection length/element
        # width are cached once for ARRAY/SLICE, since each iteration
        # needs a DIFFERENT triple anyway).
        #
        # The struct's own name (its declared name, e.g. "Point") is
        # printed first, before the opening paren -- at EVERY level
        # this ever runs, nested or not, so a struct field that's
        # itself a struct shows its own name too, not just the
        # outermost value a print() call names directly.
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

        # append the field's own name -- length via strlen, exactly
        # like the type-name appends above do, and via the same
        # shared helper; field names are statically known but kept as
        # ordinary null-terminated string literals rather than also
        # carrying a separate, redundant length field in the
        # descriptor, so this is one of the places a name costs one
        # extra runtime call to print.
        instructions.extend(self._gen_stringify_append_c_string_at(_STRINGIFY_FIELD_NAME_ADDR))

        instructions.append(LeaQ(label=self._get_colon_space_label(), dst=Register('r8')))
        instructions.append(Mov(src=Imm(2), dst=Register('r9d')))
        instructions.extend(self._gen_stringify_bulk_append(Register('r8'), Register('r9d')))

        # field value address = value_addr + field_offset; ELEM_ADDR is
        # reused here (not a dedicated FIELD_VALUE_ADDR slot) since it
        # means exactly the same thing ARRAY/SLICE already use it for
        # -- "the address of a nested value about to be recursed into"
        # -- and STRUCT never runs concurrently with either of them
        # within one call.
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
        exactly, since Hornet is statically typed -- but, unlike the
        old printf-piece-by-piece version this replaces, every type
        (including struct, not previously supported at all) now goes
        through exactly the same, uniform pipeline:

          1. Allocate a small (16-byte) growable buffer and set up its
             own {ptr, len, cap} state -- see the module docstring's
             PRINTING section for the buffer-append machinery this
             reuses.
          2. Compute value_addr: the address of x's own value.
          3. Look up (or lazily build) x's own compile-time type's
             runtime descriptor (see _get_or_build_type_descriptor).
          4. Call hornet_stringify(value_addr, type_desc,
             quote_strings=0, &buf_state) -- the single, recursive
             function that knows how to turn ANY value, of any type,
             into bytes appended onto that buffer. quote_strings=0
             here specifically because x is the OUTERMOST value of
             this print() call: a bare str argument prints unquoted
             ("hello", not 'hello'), matching how puts(x) used to
             behave -- only a str reached by recursing INTO a
             collection or struct (quote_strings=1, hardcoded at every
             such call site inside hornet_stringify itself) gets
             quoted, to stay unambiguous next to its own neighbors.
          5. Append a trailing newline.
          6. write() the buffer's own content to stdout, then free() it.

        Step 1 deliberately runs BEFORE step 2: malloc, like any real
        function, is free to clobber any caller-saved register, so
        computing value_addr first and trying to keep it alive across
        this malloc call would repeat the exact class of bug already
        found and fixed in gen_buffer_append_bytes_into (see its own
        docstring) -- computing it AFTER instead means nothing here
        ever needs to survive a call in a register at all.

        value_addr itself, depending on x's own type:
          - A bare Variable of ANY type: that variable's own existing
            storage, addressed directly (LeaQFrame, or a movq load for
            a heap-allocated array/struct -- see gen_array_address_into/
            gen_struct_address_into) -- no copy ever made.
          - A non-Variable int/bool/str expression (`print(a + 1)`):
            evaluated into a small, shared scratch slot (_print_
            scalar_temp_offset, reserved unconditionally per function
            alongside the other print-related temps -- see gen_
            function), then that slot's own address is used. Safe as a
            SHARED slot for the same reason _unnamed_slice_temp_offset
            already is: one print() call's own argument is fully
            consumed (copied into value_addr's own use, then handed to
            hornet_stringify) before another print() call, or anything
            else, could reuse this same slot.
          - An ARRAY expression that isn't a Variable/Index/Field (an
            ArrayLiteral, or an array-returning Call, e.g. `print([3]
            int[1, 2, 3])` or `print(makeArr())`): materialized via
            _gen_materialize_argument_temp_into, the exact same
            mechanism already built for an array literal or returning-
            call used as an ordinary function-call argument. This
            needed no new storage-reservation logic at all:
            _collect_argument_temps_in_expr's own walk finds a
            qualifying argument inside ANY Call node, regardless of
            whether that call turns out to be print, len, append, or
            an ordinary function -- it was already reserving a slot
            for an expression like this one long before print's own
            codegen knew how to use it. This method just needed to
            stop rejecting the case and start reading it back out.
          - A SLICE expression that isn't a Variable/Index/Field (a
            Slice re-slice or slice literal, or a slice-returning
            Call, e.g. `print([]int[1, 2, 3])` or `print(makeSlice())`):
            materialized via gen_slice_value_into into the SAME shared
            _unnamed_slice_temp_offset scratch slot gen_indexable_
            base_into's own analogous cases already use -- unlike the
            array case just above, this needs no new storage-
            reservation pass at all, since a slice's own descriptor is
            always exactly 24 bytes regardless of what it points to,
            so the existing, unconditionally-reserved shared slot
            already fits it, the same way it already fits every other
            unnamed slice this file materializes.
          - A non-Variable STRUCT expression: still NOT supported --
            deliberately narrower scope than the array and slice cases
            above, matching what was actually asked for (see this
            method's own note on gen_struct_address_into's established
            restriction elsewhere for the struct case). Restricted to
            a Variable, Field, or Index (an already-addressable,
            existing piece of storage) instead. Assign it to a named
            variable first.

        Every path still ends with `movl $0, %eax`, a harmless leftover
        from before print had anywhere real to return to -- print is
        Type.VOID, so nothing reads %eax after this call anymore, but
        leaving this in costs nothing and avoids restructuring dst
        handling here just because the value it computes is no longer
        semantically meaningful.
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

        # Step 2: compute value_addr into %r10 (caller-saved, chosen
        # deliberately -- it's consumed immediately below, into %rdi
        # for the hornet_stringify call, never needing to survive any
        # call of its own, so it doesn't need to be one of this file's
        # established callee-saved scratch registers).
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
                # A Slice expression (a re-slice or a slice literal) or
                # an ordinary slice-returning Call -- neither has an
                # existing descriptor to take the address of directly,
                # so materialize one into the SAME shared, per-function
                # scratch slot gen_indexable_base_into's own analogous
                # cases already use (_unnamed_slice_temp_offset), via
                # gen_slice_value_into (which already handles every
                # shape a slice-typed expression can take), then take
                # that slot's own address. Unlike the ARRAY case just
                # above, this needs no new storage-reservation pass at
                # all: _unnamed_slice_temp_offset is already reserved
                # unconditionally for every function, since a slice's
                # own descriptor is always exactly 24 bytes regardless
                # of what it points to -- there's no "how big could
                # this possibly be" question the way an array's
                # unbounded literal size raises.
                temp_offset = self._unnamed_slice_temp_offset
                instructions.extend(self.gen_slice_value_into(arg, Memory('rbp', temp_offset)))
                instructions.append(LeaQFrame(offset=temp_offset, dst=value_addr_reg))
        elif isinstance(arg, Variable):
            instructions.append(LeaQFrame(offset=self._local_offset(arg.name), dst=value_addr_reg))
        else:
            # int/int64/bool/str, not a bare Variable -- materialize
            # into the shared scalar scratch slot (always used at its
            # own full 8-byte width, regardless of the value's own
            # actual width, so the SAME slot serves int/bool -- 4
            # bytes -- and str/int64 -- 8 -- uniformly), then take that
            # slot's address. int64 needs the SAME full 8-byte MovQ str
            # already gets here -- an ordinary 4-byte Mov would discard
            # its own high 32 bits entirely, leaving whatever stale
            # value previously occupied this shared slot's own upper
            # half, not merely a narrower-than-ideal but still-correct
            # write the way it would be for int/bool (which are never
            # wider than 4 bytes to begin with).
            scratch_offset = self._print_scalar_temp_offset
            instructions.extend(self.gen_expr_into(arg, Register('eax')))
            if arg_type == Type.STR or arg_type == Type.INT64:
                instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', scratch_offset)))
            else:
                instructions.append(Mov(src=Register('eax'), dst=Memory('rbp', scratch_offset)))
            instructions.append(LeaQFrame(offset=scratch_offset, dst=value_addr_reg))

        # Step 3: look up (or lazily build) x's own type descriptor --
        # a pure compile-time/bookkeeping operation (see _get_or_build_
        # type_descriptor's own docstring), so its placement here,
        # after value_addr's own computation, is not itself subject to
        # any register-survival concern -- it emits no instructions of
        # its own at all, just returns a label name.
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
        # length, not a null-terminated C string, so write() (not
        # puts/printf) is the right primitive: the buffer's own
        # content can itself legitimately contain embedded NUL bytes
        # transitively (e.g. a str field inside a printed struct that
        # happens to hold one), which would truncate a puts() call
        # early but doesn't affect write(), since it's told exactly
        # how many bytes to send regardless of their content.
        instructions.append(MovQ(src=Memory('rbp', buf_state_offset), dst=Register('rsi')))
        instructions.append(Mov(src=Memory('rbp', buf_state_offset + 8), dst=Register('edx')))
        instructions.append(Mov(src=Imm(1), dst=Register('edi')))
        instructions.append(CallInstr('write'))

        # Step 7: free the buffer -- this print() call's own, exactly
        # once, regardless of how many times hornet_stringify itself
        # relocated it in between; buf_state's own ptr field always
        # holds whichever block is CURRENTLY live by the time we get
        # here, which is exactly the one (and only) block genuinely
        # owned by this call that needs freeing.
        instructions.append(MovQ(src=Memory('rbp', buf_state_offset), dst=Register('rdi')))
        instructions.append(CallInstr('free'))

        instructions.append(Mov(src=Imm(0), dst=dst))
        return instructions

    def gen_string_literal_into(self, expr: StringLiteral, dst: Operand) -> list[Instruction]:
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

    def gen_string_concat_into(self, expr: Binary, dst: Operand) -> list[Instruction]:
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

    def _gen_free_if_fresh_concat(self, operand: Node, holding_register: str) -> list[Instruction]:
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

    def gen_string_compare_into(self, expr: Binary, dst: Operand) -> list[Instruction]:
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
        instructions.append(SetCC(cc=COMPARISON_CONDITION_CODES[expr.op], operand=byte_dst))
        instructions.append(MovZX(src=byte_dst, dst=dst))
        return instructions
