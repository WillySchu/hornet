"""stdlib/http.ht's server and stdlib/net.ht's listener, against clients on this machine (nothing leaves
it), and examples/serve.ht."""

import email.utils
import http.client
import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest

from build import build_executable, executable_name
from target import host_target
from tests.targets import E2E_TARGETS, program_prefix, run_binary
from tests.test_compiler import GCC_SKIP

pytestmark = GCC_SKIP

EXAMPLE = Path(__file__).parent.parent / "examples" / "serve.ht"

# MAX_BODY TIMEOUT_MS: print the port, then answer until /stop. What respond refuses is printed.
SERVER = """\
from 'std:errors' import Error, NoneResult, must_int
from 'std:fmt' import parse_int
from 'std:http' import listen, parse_query, Header, Request, RequestResult, Response, ServerResult
from 'std:os' import get_args

def Response answer(Request r):
    if r.path == '/hello':
        return Response(200, '', [Header('Content-Type', 'text/plain')], 'hello\\n')
    if r.path == '/bad-header':
        return Response(200, '', [Header('X-Bad', 'a\\r\\nb')], '')
    if r.path == '/interim':
        return Response(100, '', [], '')
    if r.path == '/custom':
        return Response(299, 'Fine', [Header('Content-Length', '99'), Header('X-Custom', 'yes')], 'body')
    if r.path == '/status':
        return Response(must_int(parse_int(r.query)), '', [], 'status\\n')
    str text = r.method + ' ' + r.target + '\\npath ' + r.path + '\\n'
    for name, value in parse_query(r.query):
        text += 'query ' + name + '=' + value + '\\n'
    text += 'x-test ' + r.header('X-TEST') + '\\nbody ' + r.body
    return Response(200, '', [Header('Content-Type', 'text/plain')], text)

def int main(int argc, *byte argv):
    []str args = get_args(argc, argv)
    ServerResult s = listen('127.0.0.1', 0)
    if s is Error:
        print(s.message)
        return 1
    s.max_body = must_int(parse_int(args[1]))
    s.timeout_ms = must_int(parse_int(args[2]))
    print(s.port())
    while true:
        RequestResult r = s.next()
        if r is Error:
            print(r.message)
            return 1
        _ = r.body
        if r.path == '/stop':
            _ = s.respond(r, Response(200, '', [], 'bye\\n'))
            s.close()
            return 0
        NoneResult sent = s.respond(r, answer(r))
        if sent is Error:
            print('respond: ' + sent.message)
"""

# PORT: what get and post of std:http make of the server's answers.
CLIENT = """\
from 'std:errors' import Error
from 'std:http' import get, post, ResponseResult
from 'std:os' import get_args

def int main(int argc, *byte argv):
    str base = 'http://127.0.0.1:' + get_args(argc, argv)[1]
    ResponseResult got = get(base + '/hello')
    ResponseResult posted = post(base + '/echo?a=1', 'text/plain', 'sent')
    if got is Error or posted is Error:
        return 1
    print(got.status)
    print(got.body + posted.body)
    return 0
"""

# HOST PORT: whether std:net can listen there.
LISTEN = """\
from 'std:errors' import Error, must_int
from 'std:fmt' import parse_int
from 'std:net' import listen, ListenerResult
from 'std:os' import get_args

def int main(int argc, *byte argv):
    []str args = get_args(argc, argv)
    ListenerResult l = listen(args[1], must_int(parse_int(args[2])))
    if l is Error:
        print(l.message)
        return 1
    print(l.port() > 0)
    l.close()
    return 0
"""


def _built(tmp_path_factory, source: str, target) -> Path:
    directory = tmp_path_factory.mktemp("server")
    (directory / "p.ht").write_text(source)
    exe = directory / executable_name("p", target)
    build_executable(str(directory / "p.ht"), str(exe), target=target)
    return exe


class Running:
    """A server program, started; `port` is the first line it prints."""

    def __init__(self, target, argv: list):
        self.process = subprocess.Popen(program_prefix(target) + [str(a) for a in argv], stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
        line = self.process.stdout.readline()
        assert line, self.process.stderr.read()
        self.first_line = line.decode().strip()
        self.port = int(self.first_line.rsplit(":", 1)[-1].strip("/"))

    def stop(self) -> tuple:
        """Ask it to stop (GET /stop); its exit status, and what it printed after the port."""
        _request(self.port, b"GET /stop HTTP/1.1\r\nHost: x\r\n\r\n")
        out, err = self.process.communicate(timeout=30)
        return self.process.returncode, out.decode(), err.decode()

    def kill(self) -> str:
        self.process.kill()
        out, _ = self.process.communicate(timeout=30)
        return out.decode()


def _request(port: int, raw: bytes, pause: float = 0) -> bytes:
    """Send `raw` (pausing before the last byte, if asked); everything sent back until the server closes."""
    with socket.create_connection(("127.0.0.1", port), timeout=10) as s:
        if pause:
            s.sendall(raw[:-1])
            time.sleep(pause)
            raw = raw[-1:]
        s.sendall(raw)
        answer = b""
        while True:
            more = s.recv(65536)
            if not more:
                return answer
            answer += more


def _status(answer: bytes) -> int:
    return int(answer.split(b" ", 2)[1])


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """A SERVER on the host, taking bodies of up to 1000 bytes, and quiet clients for 500 ms."""
    running = Running(host_target(), [_built(tmp_path_factory, SERVER, host_target()), 1000, 500])
    yield running
    assert running.stop()[0] == 0


def _get(server, path: str, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    conn.request("GET", path, headers=headers or {})
    response = conn.getresponse()
    body = response.read()
    conn.close()
    return response, body


def test_a_response_has_its_status_headers_and_body(server):
    response, body = _get(server, "/hello")
    assert (response.status, response.reason, body) == (200, "OK", b"hello\n")
    assert [name for name, _ in response.getheaders()] == ["Date", "Connection", "Content-Type", "Content-Length"]
    assert (response.getheader("Connection"), response.getheader("Content-Length")) == ("close", "6")
    sent = email.utils.parsedate_to_datetime(response.getheader("Date")).timestamp()
    assert abs(sent - time.time()) < 60


def test_the_target_path_and_query(server):
    _, body = _get(server, "/echo/a%20b/%C3%A9?x=1&y=two+words&z=%41&x=2&flag&bad=%zz", {"X-Test": "seen"})
    lines = body.decode().splitlines()
    assert lines[:2] + lines[-2:] == [
        "GET /echo/a%20b/%C3%A9?x=1&y=two+words&z=%41&x=2&flag&bad=%zz", "path /echo/a b/é", "x-test seen", "body "]
    assert sorted(lines[2:-2]) == [  # (the first x is kept)
        "query bad=%zz", "query flag=", "query x=1", "query y=two words", "query z=A"]
    # The absolute form, as to a proxy, is its path and query.
    answer = _request(server.port, b"GET http://example.com:8080/echo?q=1 HTTP/1.1\r\nHost: example.com\r\n\r\n")
    assert b"\r\n\r\nGET /echo?q=1\npath /echo\nquery q=1\n" in answer


def test_a_body_by_length_by_chunks_and_after_100_continue(server):
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    conn.request("POST", "/echo", body=b"a body\r\n")
    assert conn.getresponse().read().endswith(b"\nbody a body\r\n")
    conn.close()
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    conn.request("PUT", "/echo", body=iter([b"chunk", b"ed"]), encode_chunked=True)
    assert conn.getresponse().read().endswith(b"\nbody chunked")
    conn.close()
    with socket.create_connection(("127.0.0.1", server.port), timeout=10) as s:
        s.sendall(b"POST /echo HTTP/1.1\r\nHost: x\r\nContent-Length: 4\r\nExpect: 100-continue\r\n\r\n")
        assert s.recv(65536) == b"HTTP/1.1 100 Continue\r\n\r\n"  # (and only then the body)
        s.sendall(b"late")
        answer = b""
        while more := s.recv(65536):
            answer += more
    assert _status(answer) == 200 and answer.endswith(b"\nbody late")


def test_head_and_statuses(server):
    head = _request(server.port, b"HEAD /hello HTTP/1.1\r\nHost: x\r\n\r\n")
    assert head.startswith(b"HTTP/1.1 200 OK\r\n") and b"\r\nContent-Length: 6\r\n" in head
    assert head.endswith(b"\r\n\r\n")  # no body
    response, body = _get(server, "/status?404")
    assert (response.status, response.reason, body) == (404, "Not Found", b"status\n")
    for status in (204, 304):
        answer = _request(server.port, b"GET /status?%d HTTP/1.1\r\n\r\n" % status)
        assert b"Content-Length" not in answer and answer.endswith(b"\r\n\r\n")
    # Its own reason; Content-Length written here, whatever the program said.
    answer = _request(server.port, b"GET /custom HTTP/1.1\r\n\r\n")
    assert answer.startswith(b"HTTP/1.1 299 Fine\r\n") and answer.endswith(b"\r\nContent-Length: 4\r\n\r\nbody")
    assert b"X-Custom: yes\r\n" in answer and b"Content-Length: 99" not in answer


@pytest.mark.parametrize("raw, status", [
    (b"GARBAGE\r\n\r\n", 400),
    (b"GET / HTTP/2.0\r\n\r\n", 505),
    (b"GET no-slash HTTP/1.1\r\n\r\n", 400),
    (b"GET /%zz HTTP/1.1\r\n\r\n", 400),
    (b"GET /%00 HTTP/1.1\r\n\r\n", 400),
    (b"GET / HTTP/1.1\r\nX-Long: one\r\n two\r\n\r\n", 400),
    (b"GET / HTTP/1.1\r\nBad Name: x\r\n\r\n", 400),
    (b"POST / HTTP/1.1\r\nContent-Length: 5\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n", 400),
    (b"POST / HTTP/1.1\r\nContent-Length: abc\r\n\r\n", 400),
    (b"POST / HTTP/1.1\r\nContent-Length: +5\r\n\r\nhello", 400),
    (b"POST / HTTP/1.1\r\nContent-Length: 1\r\nContent-Length: 2\r\n\r\nab", 400),
    (b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\nzz\r\n", 400),
    (b"POST / HTTP/1.1\r\nTransfer-Encoding: gzip\r\n\r\n", 501),
    (b"GET / HTTP/1.1\r\nExpect: something\r\n\r\n", 417),
    (b"GET /" + b"a" * 70000 + b" HTTP/1.1\r\n\r\n", 414),
    (b"GET / HTTP/1.1\r\nX-Big: " + b"a" * 70000 + b"\r\n\r\n", 431),
    (b"GET / HTTP/1.1\r\n" + b"X-Many: a\r\n" * 7000 + b"\r\n", 431),
    (b"POST / HTTP/1.1\r\nContent-Length: 2000\r\n\r\n" + b"x" * 2000, 413),  # (sent whole: still answered)
    (b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n" + b"200\r\n" + b"x" * 512 + b"\r\n" * 2 * 1, 413),
])
def test_a_malformed_or_oversized_request_is_refused(server, raw, status):
    if raw.endswith(b"x" * 512 + b"\r\n\r\n"):
        raw = raw[:-2] + b"200\r\n" + b"x" * 512 + b"\r\n0\r\n\r\n"  # 1024 bytes in chunks
    answer = _request(server.port, raw)
    assert _status(answer) == status, answer[:200]
    assert b"\r\nConnection: close\r\n" in answer
    assert _get(server, "/hello")[1] == b"hello\n"  # and it serves on


def test_a_client_that_goes_quiet_or_closes_is_passed_over(server):
    quiet = socket.create_connection(("127.0.0.1", server.port), timeout=10)
    quiet.sendall(b"GET /hello HT")  # and no more: the server waits 500 ms for it
    closed = socket.create_connection(("127.0.0.1", server.port), timeout=10)
    closed.close()
    started = time.monotonic()
    assert _get(server, "/hello")[1] == b"hello\n"
    assert time.monotonic() - started < 5
    assert quiet.recv(65536) == b""  # closed, unanswered
    quiet.close()
    # One that sends slowly, but within each read's bound, is waited for.
    assert _status(_request(server.port, b"GET /hello HTTP/1.1\r\n\r\n", pause=0.3)) == 200


def test_what_respond_refuses_is_an_error_and_nothing_is_sent(tmp_path_factory):
    # (A server of its own: what it prints is checked, and pytest -n may run tests of one module apart.)
    running = Running(host_target(), [_built(tmp_path_factory, SERVER, host_target()), 1000, 500])
    try:
        assert _request(running.port, b"GET /bad-header HTTP/1.1\r\n\r\n") == b""
        assert _request(running.port, b"GET /interim HTTP/1.1\r\n\r\n") == b""
        assert running.stop() == (0, "respond: malformed header 'X-Bad'\nrespond: not a final HTTP status: 100\n", "")
    finally:
        if running.process.poll() is None:
            running.kill()


def test_many_requests_one_after_another(server):
    def post(i):
        conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
        conn.request("POST", f"/echo?i={i}", body=b"%d" % i)
        body = conn.getresponse().read()
        conn.close()
        return body
    assert all(post(i).endswith(b"\nquery i=%d\nx-test \nbody %d" % (i, i)) for i in range(300))
    # Clients at once are answered one after another.
    results = []
    threads = [threading.Thread(target=lambda i=i: results.append(post(i))) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 20


def test_the_server_on_every_target(tmp_path_factory):
    for target in E2E_TARGETS:
        running = Running(target, [_built(tmp_path_factory, SERVER, target), 1000, 5000])
        try:
            client = _built(tmp_path_factory, CLIENT, host_target())
            r = run_binary(host_target(), [client, running.port], capture_output=True, timeout=60)
            assert (r.returncode, r.stderr) == (0, b""), target
            assert r.stdout == (b"200\nhello\nPOST /echo?a=1\npath /echo\nquery a=1\nx-test \nbody sent\n"), target
            # Refused while the body is still arriving: the answer isn't lost to a reset connection.
            assert _status(_request(running.port, b"POST / HTTP/1.1\r\nContent-Length: 200000\r\n\r\n" +
                                    b"x" * 200000)) == 413, target
            assert _status(_request(running.port, b"GARBAGE\r\n\r\n")) == 400, target
            assert running.stop() == (0, "", ""), target
        finally:
            if running.process.poll() is None:
                running.kill()


@pytest.mark.skipif(not Path("/proc/self/status").exists(), reason="reads the server's peak memory from /proc")
def test_a_server_s_memory_stays_bounded(tmp_path_factory):
    running = Running(host_target(), [_built(tmp_path_factory, SERVER, host_target()), 1 << 20, 5000])
    try:
        for i in range(300):  # 150 MB of bodies, each echoed
            conn = http.client.HTTPConnection("127.0.0.1", running.port, timeout=10)
            conn.request("POST", "/echo", body=b"%07d" % i * 70000)
            assert len(conn.getresponse().read()) > 490000
            conn.close()
        status = Path(f"/proc/{running.process.pid}/status").read_text()
        peak = int(next(line for line in status.splitlines() if line.startswith("VmHWM:")).split()[1])
        assert peak < 64 * 1024, f"{peak} KB"
        assert running.stop()[0] == 0
    finally:
        if running.process.poll() is None:
            running.kill()


def test_listening(tmp_path_factory):
    exe = _built(tmp_path_factory, LISTEN, host_target())

    def listen(host, port):
        r = run_binary(host_target(), [exe, host, str(port)], capture_output=True, text=True, timeout=30)
        return r.returncode, r.stdout
    assert listen("127.0.0.1", 0) == (0, "true\n")
    assert listen("", 0) == (0, "true\n")
    assert listen("127.0.0.1", 70000) == (1, "could not listen on 127.0.0.1:70000: a port is 0 to 65535\n")
    taken = socket.socket()
    taken.bind(("127.0.0.1", 0))
    taken.listen(1)
    port = taken.getsockname()[1]
    status, out = listen("127.0.0.1", port)
    taken.close()
    assert status == 1 and out.startswith(f"could not listen on 127.0.0.1:{port}: ")  # (the reason is the system's)
    status, out = listen("no-such-host.invalid", 0)
    assert status == 1 and out.startswith("could not listen on no-such-host.invalid:0: ")


def test_the_serve_example(tmp_path):
    (tmp_path / "www" / "sub").mkdir(parents=True)
    (tmp_path / "www" / "index.html").write_text("<h1>hi</h1>\n")
    (tmp_path / "www" / "sub" / "index.html").write_text("sub\n")
    (tmp_path / "www" / "style.css").write_text("body {}\n")
    (tmp_path / "secret.txt").write_text("secret\n")
    exe = tmp_path / executable_name("serve", host_target())
    build_executable(str(EXAMPLE), str(exe))
    running = Running(host_target(), [exe, tmp_path / "www", 0])
    try:
        assert running.first_line == f"serving {tmp_path / 'www'} on http://127.0.0.1:{running.port}/"

        def get(path, method=b"GET"):
            answer = _request(running.port, method + b" " + path + b" HTTP/1.1\r\nHost: x\r\n\r\n")
            head, _, body = answer.partition(b"\r\n\r\n")
            headers = dict(line.split(b": ", 1) for line in head.split(b"\r\n")[1:])
            return _status(answer), headers.get(b"Content-Type"), headers.get(b"Location"), body
        assert get(b"/") == (200, b"text/html; charset=utf-8", None, b"<h1>hi</h1>\n")
        assert get(b"/style.css") == (200, b"text/css; charset=utf-8", None, b"body {}\n")
        assert get(b"/sub") == (301, b"text/plain; charset=utf-8", b"/sub/", b"Moved Permanently\n")
        assert get(b"/sub/")[3] == b"sub\n"
        assert get(b"/missing")[0] == 404
        assert get(b"/../secret.txt")[0] == 404
        assert get(b"/sub/%2e%2e/%2e%2e/secret.txt")[0] == 404
        assert get(b"/", b"DELETE")[0] == 405
    finally:
        out = running.kill()
    assert out.splitlines() == [
        "GET / 200", "GET /style.css 200", "GET /sub 301", "GET /sub/ 200", "GET /missing 404",
        "GET /../secret.txt 404", "GET /sub/%2e%2e/%2e%2e/secret.txt 404", "DELETE / 405"]
