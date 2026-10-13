"""stdlib/process.ht (running other programs) and stdlib/os.ht's environment and file-system calls."""

from target import default_target
from tests.targets import build_and_run, on_every_target, posix, windows
from tests.test_compiler import GCC_SKIP

SHOW = (
    "from 'std:process' import run, Finished, RunResult\n"
    "from 'std:errors' import Error\n"
    "def show(RunResult r):\n"
    "    match r as v:\n"
    "        is Error:\n"
    "            print('error')\n"
    "        is Finished:\n"
    "            print(format('code {} signal {} ok {} out [{}] err [{}]', v.code, v.signal, v.ok(), v.out, v.err))\n"
)


def _run(calls: str, only=None):
    source = SHOW + "def int main():\n" + calls + "    return 0\n"
    return on_every_target(lambda target: build_and_run(source, target), only=only)


@GCC_SKIP
def test_a_posix_program_runs_and_what_it_writes_is_kept():
    result = _run(
        "    show(run([]str['sh', '-c', 'printf out; printf err >&2; exit 3']))\n"
        "    show(run([]str['sh', '-c', 'printf \"%s|\" \"$@\"', 'sh', 'a b', 'c\"d', '', '*']))\n"   # no shell between
        "    show(run([]str['sh', '-c', 'kill -TERM $$']))\n"
        "    RunResult big = run([]str['sh', '-c', 'head -c 300000 /dev/zero | tr \"\\\\0\" a'])\n"
        "    if big is Finished:\n"                       # more than a pipe holds, from both outputs at once
        "        print(len(big.out))\n", only=posix)
    if result is not None:
        assert result.stdout == (
            "code 3 signal 0 ok false out [out] err [err]\n"
            "code 0 signal 0 ok true out [a b|c\"d||*|] err []\n"
            "code 0 signal 15 ok false out [] err []\n"
            "300000\n")


@GCC_SKIP
def test_a_windows_program_runs_and_what_it_writes_is_kept():
    result = _run("    show(run([]str['cmd', '/c', 'echo out& exit /b 3']))\n", only=windows)
    if result is not None:
        assert result.stdout == "code 3 signal 0 ok false out [out\n] err []\n"  # (cmd's "\r\n", read as text)


@GCC_SKIP
def test_what_cannot_be_run_is_an_error():
    result = _run(
        "    show(run([]str['no-such-program-hornet']))\n"
        "    show(run([]str[]))\n"
        "    show(run([]str['a\\0b']))\n")
    assert result.stdout == "error\nerror\nerror\n"


@GCC_SKIP
def test_arguments_arrive_as_given():
    # The program runs itself: the child prints its arguments. On Windows they pass through a command
    # line, quoted, and are split again by the child's C runtime.
    source = (
        SHOW +
        "from 'std:os' import get_args\n"
        "from 'std:strings' import join\n"
        "def int main(int argc, *byte argv):\n"
        "    []str args = get_args(argc, argv)\n"
        "    if len(args) > 1 and args[1] == 'child':\n"
        "        print(join(args[2:], '|'))\n"
        "        return 4\n"
        "    show(run([]str[args[0], 'child', 'a b', 'c\"d', '', 'e\\\\', 'f\\\\\"g', 'h\\\\\\\\', ' ', '\\t']))\n"
        "    return 0\n")
    # (qemu-user can't start a program for another architecture, so only this machine's own and Windows.)
    result = on_every_target(lambda target: build_and_run(source, target),
                             only=lambda target: target == default_target() or windows(target))
    assert result.stdout == "code 4 signal 0 ok false out [a b|c\"d||e\\|f\\\"g|h\\\\| |\t\n] err []\n"


@GCC_SKIP
def test_the_environment_and_the_file_system():
    result = on_every_target(lambda target: build_and_run(
        "from 'std:os' import get_env, process_id, make_temp_dir, make_dirs, exists, rename, remove, write_file\n"
        "from 'std:os' import read_file\n"
        "from 'std:errors' import Error, must_str\n"
        "def int main():\n"
        "    print(get_env('PATH') is str)\n"
        "    print(get_env('HORNET_SURELY_UNSET') is none)\n"
        "    print(process_id() > 0)\n"
        "    str dir = must_str(make_temp_dir())\n"
        "    print(exists(dir))\n"
        "    print(make_dirs(dir + '/a/b/c') is none)\n"
        "    print(make_dirs(dir + '/a/b') is none)\n"                 # what exists is left as it is
        "    print(exists(dir + '/a/b/c'))\n"
        "    _ = write_file(dir + '/a/f', 'new')\n"
        "    _ = write_file(dir + '/a/b/g', 'old')\n"
        "    print(make_dirs(dir + '/a/f/h') is Error)\n"             # a file in the way
        "    print(rename(dir + '/a/f', dir + '/a/b/g') is none)\n"   # replacing what is there
        "    print(must_str(read_file(dir + '/a/b/g')))\n"
        "    print(exists(dir + '/a/f'))\n"
        "    print(remove(dir + '/a/b/c') is none)\n"
        "    print(remove(dir + '/a/b') is Error)\n"                  # not empty
        "    print(rename(dir + '/nothing', dir + '/x') is Error)\n"
        "    _ = remove(dir + '/a/b/g')\n"
        "    _ = remove(dir + '/a/b')\n"
        "    _ = remove(dir + '/a')\n"
        "    _ = remove(dir)\n"
        "    print(exists(dir))\n"
        "    return 0\n", target))
    assert result.stdout.split() == ['true'] * 9 + ['new', 'false', 'true', 'true', 'true', 'false']
