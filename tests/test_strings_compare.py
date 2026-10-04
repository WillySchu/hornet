"""stdlib/strings.ht's compare: which of two strings comes first, byte by byte."""

from tests.targets import build_and_run, on_every_target
from tests.test_compiler import GCC_SKIP

PAIRS = [("", ""), ("a", "a"), ("a", "b"), ("b", "a"), ("ab", "a"), ("a", "ab"), ("", "a"), ("Z", "a"),
         ("abc", "abd"), ("key10", "key9"), ("caf\\xc3\\xa9", "cafz")]


@GCC_SKIP
def test_compare_orders_like_bytes():
    calls = "".join(f"    print(sign(compare('{a}', '{b}')))\n" for a, b in PAIRS)
    source = (
        "from 'strings' import compare\n"
        "def int sign(int n):\n"
        "    if n < 0:\n"
        "        return -1\n"
        "    if n > 0:\n"
        "        return 1\n"
        "    return 0\n"
        "def int main():\n" + calls + "    return 0\n")
    result = on_every_target(lambda target: build_and_run(source, target))

    def as_bytes(text: str) -> bytes:
        return text.encode().decode('unicode_escape').encode('latin-1')
    expected = [(as_bytes(a) > as_bytes(b)) - (as_bytes(a) < as_bytes(b)) for a, b in PAIRS]
    assert result.stdout.split() == [str(sign) for sign in expected]
