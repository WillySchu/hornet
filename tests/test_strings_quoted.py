"""stdlib/strings.ht's quoted: a str as Hornet writes a string literal."""

from tests.targets import build_and_run, on_every_target
from tests.test_compiler import GCC_SKIP


@GCC_SKIP
def test_quoted():
    source = (
        "from 'strings' import quoted\n"
        "def int main():\n"
        "    print(quoted('plain'))\n"
        "    print(quoted(''))\n"
        "    print(quoted('it\\'s a \\\\ and\\ttab\\nline\\r'))\n"
        "    print(quoted(str(byte(1)) + str(byte(31)) + str(byte(127)) + str(byte(0))))\n"
        "    print(quoted('caf' + str(byte(195)) + str(byte(169))))\n"      # UTF-8 is left as it is
        "    print(format('unknown name {}', quoted('x y')))\n"
        "    return 0\n")
    result = on_every_target(lambda target: build_and_run(source, target))
    assert result.stdout.split("\n")[:4] == [
        "'plain'", "''", "'it\\'s a \\\\ and\\ttab\\nline\\r'", "'\\x01\\x1f\\x7f\\x00'"]
    assert result.stdout.split("\n")[5] == "unknown name 'x y'"
