"""IdAllocator: the one thing that survives the ENTIRE compilation,
start to finish -- every fresh Temp, label, and logical frame slot
this compiler will ever hand out, across every function's own build,
across lowering, and (once they exist) across whatever optimization
passes run in between, all draw from these same three counters.

Not scoped to any one function, and deliberately not owned by
whichever object is building or lowering IR at a given moment: a
function's own build finishes and its own IRFunctionBuilder goes away,
but ir_lowering.py's own _temp_mem still needs to mint MORE logical
slots afterward, for Temps register_allocator.py didn't promote to a
register -- using this SAME counter, never a fresh one, since two
different objects each handing out their own "slot 0" would collide
the moment both slots landed in the same function's own frame. The
identical reasoning applies to new_label: a bounds-check fail label,
minted during lowering, must never collide with an ordinary control-
flow label minted earlier, during that same function's own build.

The only property this needs to guarantee is uniqueness, never
density -- nothing anywhere in this compiler assumes ids form a
contiguous range (checked directly: every Temp/slot id is used purely
as a dict key or a set member, never as an array index), so a gap left
by a future optimization pass deleting a Temp is invisible everywhere
it matters. Ids are never reused and never renumbered."""

from typing import Dict

from codegen.ir import IRFunction, Temp
from semantic import Type


class IdAllocator:
    def __init__(self):
        self._temp_count = 0
        self._label_count = 0
        self._slot_count = 0
        # IR temps (see ir.py): _temp_offsets maps a NAMED-local Temp's
        # id to its own logical slot (set by temp_at_offset, when the
        # Temp itself is created -- not a resolved physical offset at
        # all; see its own docstring). _temp_slots maps an ANONYMOUS
        # Temp's id to its own logical slot instead -- populated
        # lazily by ir_lowering.py's _temp_mem, the first time a Temp
        # is actually referenced during lowering, not when it's
        # created; see new_temp's own docstring for why that split
        # matters, and _temp_mem's own docstring for why an anonymous
        # Temp's slot needs a placeholder (FrameSlot) rather than an
        # immediately-resolved offset the way every other slot in this
        # compiler gets one. Both live here, not on whatever object
        # created the Temp, for the identical reason the counters
        # above do: a named-local Temp is created during one
        # function's own build, but read back out during that same
        # function's own, separately-scoped lowering.
        self._temp_offsets: Dict[int, int] = {}
        self._temp_slots: Dict[int, int] = {}

    def new_label(self, prefix: str) -> str:
        """Returns a fresh, uniquely-numbered local label like
        `.Land_short_0`. Needed because AND/OR/if codegen all emit real
        jump targets, and a program can contain any number of them --
        each one needs a name the assembler won't collide with any
        other."""
        label = f".L{prefix}_{self._label_count}"
        self._label_count += 1
        return label

    def new_temp(self, t: Type) -> Temp:
        """Allocates a fresh virtual register (see ir.py) of type `t`
        -- id only, no storage decision yet. Where a Temp actually
        lives is v1's own lowering policy, decided lazily on first
        reference by ir_lowering.py's _temp_mem, not here -- this
        method's only job is handing out an identity, the same way
        new_label's only job is handing out a name."""
        temp_id = self._temp_count
        self._temp_count += 1
        return Temp(id=temp_id, type=t)

    def temp_at_offset(self, t: Type, slot: int) -> Temp:
        """Allocates a fresh virtual register that already has a
        known home -- `slot`, not a freshly-carved one -- registered
        immediately rather than left for _temp_mem to decide lazily.
        Used exactly once per named scalar local/parameter (see
        _bind_local/_bind_param): a variable's own slot is already
        fixed by _collect_locals/_collect_params before this ever
        runs, so there's no lazy decision left to make, and reusing
        that slot -- rather than allocating a second, redundant one --
        is what lets every read and write of that variable, for the
        rest of the function, share one Temp identity. is_named_local
        marks it as such -- see Temp's own docstring for why.

        Stores the SLOT here, not a resolved physical offset -- unlike
        an eager version of this method used to. Every slot in this
        compiler, named-local or otherwise, gets its own final offset
        decided the identical way now: once, by _resolve_frame_layout,
        after every slot a function will ever need (including whatever
        _temp_mem discovers lazily, mid-lowering) is known. _temp_mem's
        own "named-local" branch is what reads self._temp_offsets back
        out, returning a FrameSlot placeholder for _patch_frame_slots
        to resolve later, exactly like any other slot."""
        temp_id = self._temp_count
        self._temp_count += 1
        self._temp_offsets[temp_id] = slot
        return Temp(id=temp_id, type=t, is_named_local=True)

    def new_slot(self, width: int, label: str, ir_fn: IRFunction) -> int:
        """Allocates a fresh, logical frame-slot identifier -- the
        new_temp of frame slots, but for IRLocalAddress's own `slot`
        field rather than a virtual register. Carries no physical
        offset of its own at all yet: every caller here used to
        compute `self._next_offset -= width` and use the result
        directly (as an offset, immediately) -- now it records the
        slot's own width and a human-readable label (for debugging;
        never read by anything else) onto `ir_fn` itself, not here
        (see IRFunction's own docstring for why), and returns an
        opaque id instead, deferring the actual offset decision to
        _resolve_frame_layout.

        Order matters here: _resolve_frame_layout assigns offsets in
        the exact order slots were created in (ir_fn.slot_widths is a
        plain dict, so insertion order is preserved), reproducing
        today's own running-counter layout exactly. Every caller below
        that used to reserve a slot with `self._next_offset -= width`
        directly now calls this instead, in the identical order those
        subtractions used to happen in."""
        slot_id = self._slot_count
        self._slot_count += 1
        ir_fn.slot_widths[slot_id] = width
        ir_fn.slot_labels[slot_id] = label
        return slot_id
