"""`x[low:high]` evaluates `low` before `high`."""

from tests.test_compiler import GCC_SKIP, assert_program_stdout


@GCC_SKIP
def test_slice_bounds_are_evaluated_left_to_right():
    assert_program_stdout(
        "def int low():\n"
        "    print('low')\n"
        "    return 1\n"
        "def int high():\n"
        "    print('high')\n"
        "    return 3\n"
        "def int main():\n"
        "    [5]int a = [1, 2, 3, 4, 5]\n"
        "    []int s = a[:]\n"
        "    str t = 'hello'\n"
        "    print(a[low():high()])\n"
        "    print(s[low():high()])\n"
        "    print(t[low():high()])\n"
        "    return 0\n",
        "low\nhigh\n[]int[2, 3]\nlow\nhigh\n[]int[2, 3]\nlow\nhigh\nel\n",
    )
