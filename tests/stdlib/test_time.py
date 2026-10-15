"""stdlib/time.ht: the clocks, and dates in UTC (checked against Python's datetime)."""

import datetime
import email.utils
import random
import time

from tests.targets import build_and_run, on_every_target
from tests.test_compiler import GCC_SKIP

MOMENTS = [0, -1, 1, 86399999, -86400001, 951782400000, 951868799999, 4107542400000, 253402300799999,
           -62135596800000, -12219292800001, 1700000000123] + random.Random(7).sample(range(-62135596800000,
                                                                                       253402300800000), 40)


def _expected(ms: int) -> str:
    d = datetime.datetime(1970, 1, 1) + datetime.timedelta(milliseconds=ms)
    return (f"DateTime(year: {d.year}, month: {d.month}, day: {d.day}, hour: {d.hour}, minute: {d.minute}, "
            f"second: {d.second}, millisecond: {d.microsecond // 1000}, weekday: {d.isoweekday() % 7})")


@GCC_SKIP
def test_utc_and_http_date():
    calls = "".join(f"    print(utc({ms}))\n" for ms in MOMENTS)
    source = ("from 'std:time' import utc\nfrom 'std:http' import http_date\ndef int main():\n" + calls +
              "    print(http_date(784111777000))\n    print(http_date(1792000000000))\n    return 0\n")
    r = on_every_target(lambda target: build_and_run(source, target))
    assert (r.returncode, r.stderr) == (0, "")
    assert r.stdout == "".join(_expected(ms) + "\n" for ms in MOMENTS) + "Sun, 06 Nov 1994 08:49:37 GMT\n" + \
        email.utils.formatdate(1792000000, usegmt=True) + "\n"


@GCC_SKIP
def test_the_clocks():
    source = ("from 'std:time' import now_ms, monotonic_ms\ndef int main():\n"
              "    int start = monotonic_ms()\n    int now = now_ms()\n    int later = monotonic_ms()\n"
              "    print(now)\n    print(later >= start)\n    return 0\n")

    def run(target):
        before = time.time() * 1000
        r = build_and_run(source, target)
        now, ordered = r.stdout.split()
        assert before - 1000 <= int(now) <= time.time() * 1000 + 1000 and ordered == "true", target
        return r
    r = on_every_target(run, agree=False)
    assert (r.returncode, r.stderr) == (0, "")
