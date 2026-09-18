"""Confirms runtime/hornet_typedesc_tags.h, as checked in, is exactly
what generate_typedesc_header.py would produce right now from
ir/strings.py's own live _TYPEDESC_* constants.

This is the actual anti-drift safeguard for the tag-value half of the
type-descriptor ABI (see generate_typedesc_header.py's own module
docstring for what this does and doesn't cover): if someone changes a
_TYPEDESC_* value in ir/strings.py and forgets to re-run the
generator and check in the result, this test fails, rather than the
mismatch surviving silently until it corrupts something a runtime.c
that trusted the stale header actually reads. There is not yet a build
step that regenerates this file automatically -- this test is what
catches that gap until one exists.
"""
from pathlib import Path

from runtime.generate_typedesc_header import (
    OUTPUT_PATH,
    _collect_typedesc_constants,
    _render_header,
)


def test_generated_header_matches_checked_in_file():
    checked_in = Path(OUTPUT_PATH).read_text()
    freshly_generated = _render_header(_collect_typedesc_constants())
    assert checked_in == freshly_generated, (
        "runtime/hornet_typedesc_tags.h is out of date relative to "
        "ir/strings.py's own _TYPEDESC_* constants -- re-run "
        "`python3 runtime/generate_typedesc_header.py` and check in "
        "the result."
    )
