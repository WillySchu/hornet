#!/bin/bash
#
# trace_memory.sh -- compile a Hornet (.ht) program through the real
# pipeline and run it under malloc_trace.c, reporting both the
# program's own output/exit code and a memory-safety summary.
#
# Usage:
#     ./trace_memory.sh path/to/program.ht
#
# Requires: python3, gcc, and this script sitting next to lexer.py,
# parser.py, semantic.py, codegen.py, and malloc_trace.c.
#
# What to actually look at in the output:
#   - "invalid frees" in the summary should ALWAYS be 0. Anything else
#     is a real bug (a double-free, or a free() on a pointer that was
#     never malloc'd -- most likely a string literal). This is the
#     number that matters.
#   - "still-leaked" being nonzero is expected, not necessarily a bug --
#     see malloc_trace.c's own header comment for why. Compare it
#     against what you'd predict by hand for the specific program
#     you're testing (e.g. "this declares 3 concatenation results and
#     stores each in a variable, so I'd expect 3 leaked, 0 freed").
#
# See malloc_trace.c's own comments for what this actually checks, its
# limitations, and an honest note that the macOS path here is written
# from documentation, not verified against a real Mac.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "$1" ]; then
    echo "Usage: $0 path/to/program.ht"
    exit 1
fi
HT_FILE="$1"
if [ ! -f "$HT_FILE" ]; then
    echo "No such file: $HT_FILE"
    exit 1
fi

UNAME="$(uname -s)"
if [ "$UNAME" = "Darwin" ]; then
    PLATFORM="macos"
    SHIM="$SCRIPT_DIR/malloc_trace.dylib"
else
    PLATFORM="linux"
    SHIM="$SCRIPT_DIR/malloc_trace.so"
fi
SHIM_SRC="$SCRIPT_DIR/malloc_trace.c"

# Build (or rebuild, if the source changed) the shim.
if [ ! -f "$SHIM" ] || [ "$SHIM_SRC" -nt "$SHIM" ]; then
    echo "Building $SHIM..."
    if [ "$PLATFORM" = "macos" ]; then
        gcc -shared -fPIC -o "$SHIM" "$SHIM_SRC"
    else
        gcc -shared -fPIC -o "$SHIM" "$SHIM_SRC" -ldl
    fi
fi

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT
ASM="$TMPDIR/program.s"
BIN="$TMPDIR/program"
TRACE_LOG="$TMPDIR/trace.log"

echo "--- compiling $HT_FILE (platform: $PLATFORM) ---"
# codegen.py's own CLI runs the full lex -> parse -> analyze -> codegen
# pipeline (compile_to_asm), so a semantic error here fails loudly and
# clearly before this script ever tries to assemble anything.
python3 "$SCRIPT_DIR/codegen.py" "$HT_FILE" --platform "$PLATFORM" -o "$ASM"

echo "--- assembling/linking ---"
if [ "$PLATFORM" = "macos" ]; then
    gcc -arch x86_64 "$ASM" -o "$BIN"
else
    gcc "$ASM" -o "$BIN"
fi

echo "--- running under malloc_trace ---"
echo "(program's own stdout/stderr below, if any)"
echo "----------------------------------------"
set +e
if [ "$PLATFORM" = "macos" ]; then
    DYLD_INSERT_LIBRARIES="$SHIM" DYLD_FORCE_FLAT_NAMESPACE=1 "$BIN" 2>"$TRACE_LOG"
else
    LD_PRELOAD="$SHIM" "$BIN" 2>"$TRACE_LOG"
fi
EXIT_CODE=$?
set -e
echo "----------------------------------------"
echo "program exit code: $EXIT_CODE"
echo

echo "--- memory trace ---"
cat "$TRACE_LOG"

# Pull the invalid-frees count back out of the trace log for a clear,
# unmissable pass/fail line at the very end. sed -E (not grep -P) is
# used here specifically because it's portable to both GNU sed (Linux)
# and BSD sed (macOS) -- grep -P is a GNU/PCRE extension BSD grep
# doesn't have.
INVALID="$(grep 'invalid frees:' "$TRACE_LOG" | sed -E 's/.*invalid frees: *([0-9]+).*/\1/')"

echo
if [ "$INVALID" = "0" ]; then
    echo "RESULT: OK -- no invalid frees detected."
elif [ -z "$INVALID" ]; then
    echo "RESULT: UNKNOWN -- couldn't find a trace summary. Did the shim actually load?"
    echo "(On Linux, check that LD_PRELOAD pointed at a real .so. On macOS, DYLD_INSERT_LIBRARIES"
    echo " is sometimes blocked by System Integrity Protection for certain binaries.)"
else
    echo "RESULT: *** FAILED -- $INVALID invalid free(s) detected. This is a real bug. ***"
fi
