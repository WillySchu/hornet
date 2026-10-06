"""A failed bounds check says what it compared: `index out of bounds: index 7, length 3`, and for
`a[start:end]` which bound was past the length (a slice's capacity) or that they were out of order.
The values are read from wherever they live when the check fails."""

import pytest

from tests.test_compiler import GCC_SKIP, assert_program_panics

SETUP = (
    "def int main():\n"
    "    [3]int arr = [1, 2, 3]\n"
    "    []int xs = [1, 2, 3, 4, 5]\n"
    "    []int part = xs[1:3]\n"          # length 2, capacity 4
    "    str s = 'hello'\n"
    "    int two = len(part)\n"
)


@GCC_SKIP
@pytest.mark.parametrize("expression,message", [
    ("arr[two + 5]", "index out of bounds: index 7, length 3"),
    ("xs[two + 3]", "index out of bounds: index 5, length 5"),
    ("part[two]", "index out of bounds: index 2, length 2"),          # its length, not its capacity
    ("int(s[two * 4])", "index out of bounds: index 8, length 5"),
    ("arr[two - 3]", "index out of bounds: index -1, length 3"),      # a negative index, as written
    ("xs[two - 9000000000]", "index out of bounds: index -8999999998, length 5"),
    ("len(arr[two + 2:])", "slice bounds out of range: start 4, length 3"),
    ("len(arr[:two + 2])", "slice bounds out of range: end 4, length 3"),
    ("len(arr[two:two - 1])", "slice bounds out of range: start 2, end 1"),
    ("len(s[two:two * 4])", "slice bounds out of range: end 8, length 5"),
    ("len(part[two + 3:])", "slice bounds out of range: start 5, capacity 4"),   # a slice may grow to its capacity
    ("len(part[:two + 3])", "slice bounds out of range: end 5, capacity 4"),
    ("len(part[two + 1:two])", "slice bounds out of range: start 3, end 2"),
    ("len(part[two - 3:])", "slice bounds out of range: start -1, capacity 4"),
])
def test_a_failed_check_says_what_it_compared(expression, message):
    assert_program_panics(SETUP + f"    return {expression}\n", message)


# The index and the length arrive in different registers, or on the stack, as the parameters move.
SIGNATURES = [
    ("int i, []int xs", "9, xs"),
    ("[]int xs, int i", "xs, 9"),
    ("int a, int i, []int xs", "0, 9, xs"),
    ("int a, int b, []int xs, int i", "0, 0, xs, 9"),
    ("[]int xs, int a, int b, int c, int d, int i", "xs, 0, 0, 0, 0, 9"),
    ("int a, int b, int c, int d, int e, int f, int i, []int xs", "0, 0, 0, 0, 0, 0, 9, xs"),
]


@GCC_SKIP
@pytest.mark.parametrize("parameters,arguments", SIGNATURES)
@pytest.mark.parametrize("body,printed", [
    ("    return xs[i]\n", ""),
    # ... and are held across calls first.
    ("    int keep = i * 2\n    print('x')\n    print(len(xs))\n    return xs[i] + keep\n", "x\n3\n"),
])
def test_the_values_are_found_wherever_they_live(parameters, arguments, body, printed):
    assert_program_panics(
        f"def int pick({parameters}):\n" + body +
        "def int main():\n"
        "    []int xs = [1, 2, 3]\n"
        f"    return pick({arguments})\n",
        "index out of bounds: index 9, length 3",
        expected_stdout=printed,
    )
