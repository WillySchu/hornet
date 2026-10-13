"""Tests for build.py's own build_executable: the one entry point
that compiles a Hornet source file AND links it against a freshly-
built runtime into a single, runnable executable -- the piece of
the toolchain that didn't exist at all before the runtime did (see
build.py's own module docstring).
"""

import shutil
import subprocess
import sys
import tempfile

import pytest

import build
from build import REPO_ROOT

GCC_AVAILABLE = shutil.which("gcc") is not None
pytestmark = pytest.mark.skipif(not GCC_AVAILABLE, reason="gcc not available")


def _write(tmpdir, name, content):
    path = f"{tmpdir}/{name}"
    with open(path, "w") as f:
        f.write(content)
    return path


def test_build_executable_compiles_and_runs_correctly():
    with tempfile.TemporaryDirectory() as tmpdir:
        source = _write(tmpdir, "program.ht", "def int main():\n    int x = 5\n    int y = 10\n    return x + y\n")
        binary = f"{tmpdir}/program"
        build.build_executable(source, binary)
        result = subprocess.run([binary], capture_output=True)
        assert result.returncode == 15


def test_build_executable_propagates_semantic_errors_directly():
    """A SemanticError (or SyntaxError) from the ordinary Hornet
    pipeline is a lexer/parser/semantic-analysis failure, not a build
    failure -- it should propagate as itself, not get wrapped into a
    BuildError alongside actual compiling/linking failures."""
    with tempfile.TemporaryDirectory() as tmpdir:
        source = _write(tmpdir, "program.ht", "def int main():\n    return undeclaredVariable\n")
        binary = f"{tmpdir}/program"
        with pytest.raises(Exception) as exc_info:
            build.build_executable(source, binary)
        assert not isinstance(exc_info.value, build.BuildError)


def _runtime_copy(tmp_path):
    """A copy of the runtime's sources, for a test to change."""
    import shutil
    copy = tmp_path / "runtime"
    shutil.copytree(build.RUNTIME_DIR, copy, ignore=shutil.ignore_patterns("__pycache__"))
    return copy


def test_build_executable_raises_build_error_with_actionable_message_on_bad_runtime_c(monkeypatch, tmp_path):
    runtime = _runtime_copy(tmp_path)
    (runtime / "memory.c").write_text("this is not valid C at all !!!\n")
    monkeypatch.setattr(build, "RUNTIME_DIR", runtime)
    monkeypatch.setenv("HORNET_CACHE_DIR", str(tmp_path / "cache"))

    source = tmp_path / "program.ht"
    source.write_text("def int main():\n    return 0\n")
    binary = tmp_path / "program"

    with pytest.raises(build.BuildError) as exc_info:
        build.build_executable(str(source), str(binary))
    message = str(exc_info.value)
    assert "compiling runtime/memory.c" in message
    assert "gcc" in message


def test_c_compiler_selects_the_architecture_on_macos():
    """Apple's gcc defaults to the host architecture; -arch picks the target's."""
    from target import Target
    assert build.c_compiler(Target('x86_64', 'macos')) == ["gcc", "-arch", "x86_64"]
    assert build.c_compiler(Target('aarch64', 'macos')) == ["gcc", "-arch", "arm64"]


def test_targets_parse_and_default_to_an_implemented_architecture():
    from target import IMPLEMENTED_ARCHES, Target, as_target, default_target
    assert Target.parse('aarch64-macos') == Target('aarch64', 'macos')
    assert str(as_target('x86_64-linux')) == 'x86_64-linux'
    assert default_target().arch in IMPLEMENTED_ARCHES
    with pytest.raises(ValueError, match="unknown target 'arm-linux'"):
        Target.parse('arm-linux')


def test_every_target_compiles_from_the_command_line(tmp_path):
    src = tmp_path / 'p.ht'
    src.write_text("def int main():\n    return 0\n")
    for name, label in [('x86_64-macos', '_main:'), ('aarch64-linux', '\nmain:'), ('aarch64-macos', '_main:')]:
        r = subprocess.run([sys.executable, str(REPO_ROOT / 'compile.py'), str(src), '--target', name],
                           capture_output=True, text=True)
        assert r.returncode == 0 and label in r.stdout, (name, r.stderr)


def test_architecture_without_a_backend_is_a_clean_error():
    from backend import TargetError, lower_to_asm
    from target import Target
    with pytest.raises(TargetError, match="no backend for riscv64 yet"):
        lower_to_asm(None, Target('riscv64', 'linux'))


def test_cross_toolchain_and_run_prefix_selection(monkeypatch):
    from target import Target
    x86_linux, arm_linux = Target('x86_64', 'linux'), Target('aarch64', 'linux')
    x86_mac, arm_mac = Target('x86_64', 'macos'), Target('aarch64', 'macos')
    monkeypatch.setattr(build, 'host_target', lambda: x86_linux)
    assert build.c_compiler(x86_linux) == ["gcc"]
    assert build.c_compiler(arm_linux) == ["aarch64-linux-gnu-gcc"]
    assert build.run_prefix(x86_linux) == []
    assert build.run_prefix(arm_linux) == ["qemu-aarch64", "-L", "/usr/aarch64-linux-gnu"]
    assert build.run_prefix(x86_mac) is None and not build.can_build(x86_mac)
    monkeypatch.setattr(build, 'host_target', lambda: arm_mac)
    assert build.run_prefix(x86_mac) == []  # Rosetta 2
    assert build.run_prefix(arm_mac) == []
    assert build.run_prefix(x86_linux) is None
    monkeypatch.setattr(build, 'host_target', lambda: x86_mac)
    assert build.run_prefix(arm_mac) is None


def test_missing_toolchain_is_a_build_error():
    with pytest.raises(build.BuildError, match="'no-such-cc' not found"):
        build._run(["no-such-cc", "-c", "x.c"], "compiling runtime/x.c")


def test_default_target_is_the_host_now_that_both_architectures_have_backends(monkeypatch):
    import target
    from target import Target
    for host in (Target('aarch64', 'macos'), Target('x86_64', 'linux'), Target('aarch64', 'linux')):
        monkeypatch.setattr(target, 'host_target', lambda host=host: host)
        assert target.default_target() == host


def test_the_runtime_is_built_once_and_rebuilt_when_it_changes(monkeypatch, tmp_path):
    from target import default_target
    monkeypatch.setenv("HORNET_CACHE_DIR", str(tmp_path / "cache"))
    calls = []
    real_run = build._run
    monkeypatch.setattr(build, "_run", lambda args, step: (calls.append(step), real_run(args, step)))
    first = build.runtime_objects(default_target())
    built = len(calls)
    assert built == len(build.runtime_sources(default_target())) + 1  # each file, then the archive
    assert build.runtime_objects(default_target()) == first and len(calls) == built
    assert [path.parent for path in first] == [tmp_path / "cache"] * 2 and all(path.exists() for path in first)
    runtime = _runtime_copy(tmp_path)
    monkeypatch.setattr(build, "RUNTIME_DIR", runtime)
    assert build.runtime_objects(default_target()) == first  # (the same sources)
    for name in ("dict.c", "internal.h"):  # a header changes it too
        (runtime / name).write_text((runtime / name).read_text() + "\n/* changed */\n")
        assert build.runtime_objects(default_target()) not in (first, [])
    assert len(calls) == 3 * built


def test_each_target_gets_its_own_platform_s_files():
    from target import Target
    names = lambda t: [path.name for path in build.runtime_sources(Target.parse(t))]
    assert "stack_windows.c" in names("x86_64-windows") and "stack_posix.c" not in names("x86_64-windows")
    for t in ("x86_64-linux", "aarch64-linux", "aarch64-macos"):
        assert "stack_posix.c" in names(t) and "stack_windows.c" not in names(t)
    assert all(not name.startswith("test_") for name in names("x86_64-linux"))
    assert build.RUNTIME_START in names("x86_64-windows") and build.RUNTIME_START in names("x86_64-linux")


def test_every_runtime_file_is_listed():
    """Each C file in runtime/ is built into the runtime, or is a test; one left out of
    build.RUNTIME_FILES, or left behind after it was taken out, is noticed here."""
    found = {path.name for path in build.RUNTIME_DIR.glob("*.c") if not path.name.startswith("test_")}
    assert found == set(build.RUNTIME_FILES), (
        f"in runtime/ but not build.RUNTIME_FILES: {sorted(found - set(build.RUNTIME_FILES))}; "
        f"listed but not there: {sorted(set(build.RUNTIME_FILES) - found)}")


def test_the_archiver_goes_with_the_compiler():
    from target import Target
    assert build.archiver(Target('x86_64', 'macos')) == "ar"
    assert build.archiver(Target('aarch64', 'linux')) in ("ar", "aarch64-linux-gnu-ar")
    assert build.archiver(Target('x86_64', 'windows')) in ("ar", "x86_64-w64-mingw32-ar")
