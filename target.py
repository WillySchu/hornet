"""Compilation targets: an architecture and an operating system, written `arch-os`."""

import platform
import sys
from dataclasses import dataclass

ARCHES = ('x86_64', 'aarch64')
OSES = ('linux', 'macos')
TARGET_NAMES = tuple(f"{a}-{o}" for a in ARCHES for o in OSES)

# Architectures with a complete backend: the default target and end-to-end tests use these.
IMPLEMENTED_ARCHES = ('x86_64', 'aarch64')
# Architectures whose backend is being built: selectable with --target, not yet the default.
IN_PROGRESS_ARCHES = ()


@dataclass(frozen=True)
class Target:
    arch: str
    os: str

    def __str__(self) -> str:
        return f"{self.arch}-{self.os}"

    @staticmethod
    def parse(text: str) -> 'Target':
        arch, _, os_name = text.partition('-')
        if arch not in ARCHES or os_name not in OSES:
            raise ValueError(f"unknown target '{text}' (expected one of: {', '.join(TARGET_NAMES)})")
        return Target(arch, os_name)


def host_target() -> Target:
    machine = platform.machine().lower()
    arch = 'aarch64' if machine in ('arm64', 'aarch64') else 'x86_64'
    return Target(arch, 'macos' if sys.platform == 'darwin' else 'linux')


def default_target() -> Target:
    """The host, or x86-64 on the host's OS while the host's architecture has no backend."""
    host = host_target()
    return host if host.arch in IMPLEMENTED_ARCHES else Target('x86_64', host.os)


def as_target(value) -> Target:
    """A Target from a Target, an `arch-os` string, or None (the default)."""
    if value is None:
        return default_target()
    if isinstance(value, Target):
        return value
    return Target.parse(value)
