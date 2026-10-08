"""stdlib/net.ht and stdlib/http.ht, against a server on this machine (nothing leaves it)."""

import socket
import threading
import time
from pathlib import Path

import pytest

from build import build_executable, executable_name
from target import host_target
from tests.targets import on_every_target, run_binary
from tests.test_compiler import GCC_SKIP

EXAMPLE = Path(__file__).parent.parent / "examples" / "fetch.ht"
EVERY_BYTE = bytes(range(256)) * 4096  # 1 MB

# METHOD URL HEADER TIMEOUT_MS [BODY]: the status line, the header's value, then the body. Or: url URL.
CLIENT = """\
from 'errors' import Error, must_int
from 'fmt' import int_to_str, parse_int
from 'http' import request, parse_url, Header, ResponseResult, UrlResult
from 'os' import get_args, write_stdout

def int main(int argc, *byte argv):
    []str args = get_args(argc, argv)
    if args[1] == 'url':
        UrlResult u = parse_url(args[2])
        if u is Error:
            print('error: ' + u.message)
            return 1
        print(u)
        return 0
    []Header headers
    str body = ''
    if len(args) > 5:
        body = args[5]
        headers = append(headers, Header('Content-Type', 'text/plain'))
    ResponseResult r = request(args[1], args[2], headers, body, must_int(parse_int(args[4])))
    if r is Error:
        print('error: ' + r.message)
        return 1
    write_stdout(int_to_str(r.status) + ' ' + r.reason + '\\n' + r.header(args[3]) + '\\n' + r.body)
    return 0
"""

GET_AND_POST = """\
from 'errors' import Error
from 'http' import get, post, ResponseResult
from 'os' import get_args

def int main(int argc, *byte argv):
    []str args = get_args(argc, argv)
    ResponseResult got = get(args[1] + '/hello')
    ResponseResult posted = post(args[1] + '/echo', 'text/x-test', 'sent')
    if got is Error or posted is Error:
        return 1
    print(got.status)
    print(got.body + posted.body)
    return 0
"""


def _ok(body: bytes, headers: bytes = b"") -> bytes:
    return b"HTTP/1.1 200 OK\r\n" + headers + b"Content-Length: %d\r\n\r\n" % len(body) + body


def _echo(request: bytes) -> bytes:
    head, _, body = request.partition(b"\r\n\r\n")
    lines = head.split(b"\r\n")
    wanted = (b"host:", b"connection:", b"content-length:", b"content-type:")
    return _ok(b"\n".join([lines[0]] + [l for l in lines[1:] if l.lower().startswith(wanted)] + [body]))


RESPONSES = {
    b"/hello": _ok(b"hello\n", b"Content-Type: text/plain\r\nX-Mixed-Case: Yes\r\n"),
    b"/chunked": b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
                 b"5\r\nhello\r\n1;note=1\r\n \r\nA\r\n0123456789\r\n0\r\nTrailer: x\r\n\r\n",
    b"/until-close": b"HTTP/1.0 200 OK\r\n\r\nuntil the end",
    b"/big": _ok(EVERY_BYTE),
    b"/missing": b"HTTP/1.1 404 Not Found\r\nContent-Length: 5\r\n\r\ngone\n",
    b"/empty": b"HTTP/1.1 204 No Content\r\n\r\n",
    b"/moved": b"HTTP/1.1 302 Found\r\nLocation: /hello\r\nContent-Length: 0\r\n\r\n",
    b"/early": b"HTTP/1.1 103 Early Hints\r\nLink: </x>\r\n\r\n" + _ok(b"ok"),
    b"/folded": _ok(b"ok", b"X-Long: one\r\n  two\r\n"),
    b"/bare-lf": b"HTTP/1.1 200 OK\nContent-Length: 2\n\nok",
    b"/head": b"HTTP/1.1 200 OK\r\nContent-Length: 6\r\n\r\n",
    b"/short": b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\nshort",
    b"/bad-chunk": b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\nzz\r\n",
    b"/cut-chunk": b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhel",
    b"/not-http": b"SSH-2.0-OpenSSH_9\r\n",
    b"/bad-status": b"HTTP/1.1 -12 Odd\r\n\r\n",
    b"/bad-length": b"HTTP/1.1 200 OK\r\nContent-Length: lots\r\n\r\n",
    b"/nothing": b"",
}


def _answer(conn: socket.socket) -> None:
    with conn:
        request = b""
        while b"\r\n\r\n" not in request:
            more = conn.recv(65536)
            if not more:
                return
            request += more
        head, _, body = request.partition(b"\r\n\r\n")
        length = [int(l.split(b":")[1]) for l in head.split(b"\r\n") if l.lower().startswith(b"content-length:")]
        while length and len(body) < length[0]:
            body += conn.recv(65536)
        path = head.split(b" ")[1].split(b"?")[0]
        if path == b"/stall":
            time.sleep(3)
        elif path == b"/echo":
            conn.sendall(_echo(head + b"\r\n\r\n" + body))
        else:
            conn.sendall(RESPONSES[path])


@pytest.fixture(scope="module")
def server():
    """The address (`http://127.0.0.1:PORT`) of a server answering each request from RESPONSES."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(16)

    def serve():
        while True:
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            threading.Thread(target=_answer, args=(conn,), daemon=True).start()
    threading.Thread(target=serve, daemon=True).start()
    yield f"http://127.0.0.1:{listener.getsockname()[1]}"
    listener.close()


def _built(tmp_path_factory, source: str, target=None) -> Path:
    target = target or host_target()
    directory = tmp_path_factory.mktemp("http")
    (directory / "p.ht").write_text(source)
    exe = directory / executable_name("p", target)
    build_executable(str(directory / "p.ht"), str(exe), target=target)
    return exe


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Run CLIENT with these arguments; (exit status, stdout)."""
    exe = _built(tmp_path_factory, CLIENT)

    def run(*args):
        r = run_binary(host_target(), [exe, *args], capture_output=True, timeout=30)
        assert r.stderr == b""
        return r.returncode, r.stdout
    return run


def _get(client, url: str, header: str = "-", timeout_ms: int = 5000):
    return client("GET", url, header, str(timeout_ms))


@GCC_SKIP
@pytest.mark.parametrize("path, expected", [
    ("/hello", b"200 OK\nYes\nhello\n"),  # a header is found whatever its case
    ("/chunked", b"200 OK\n\nhello 0123456789"),
    ("/until-close", b"200 OK\n\nuntil the end"),
    ("/missing", b"404 Not Found\n\ngone\n"),
    ("/empty", b"204 No Content\n\n"),
    ("/moved", b"302 Found\n\n"),  # a redirect is the caller's to follow
    ("/early", b"200 OK\n\nok"),  # an interim response is passed over
    ("/bare-lf", b"200 OK\n\nok"),
])
def test_a_response_is_read_as_its_headers_say(server, client, path, expected):
    assert _get(client, server + path, "x-mixed-case") == (0, expected)


@GCC_SKIP
def test_a_body_keeps_every_byte(server, client):
    assert _get(client, server + "/big") == (0, b"200 OK\n\n" + EVERY_BYTE)


@GCC_SKIP
def test_a_host_name_is_looked_up(server, client):
    # (Whichever of the name's addresses the server isn't listening on is refused, and the next tried.)
    assert _get(client, server.replace("127.0.0.1", "localhost") + "/hello") == (0, b"200 OK\n\nhello\n")
    status, out = _get(client, "http://no-such-host.invalid/")
    assert status == 1 and out.startswith(b"error: could not connect to no-such-host.invalid: ")


@GCC_SKIP
def test_headers(server, client):
    assert _get(client, server + "/moved", "LOCATION") == (0, b"302 Found\n/hello\n")
    assert _get(client, server + "/folded", "X-Long") == (0, b"200 OK\none two\nok")


@GCC_SKIP
def test_what_is_sent(server, client):
    host = server[len("http://"):].encode()
    assert _get(client, server + "/echo?a=1&b=2#part") == (
        0, b"200 OK\n\nGET /echo?a=1&b=2 HTTP/1.1\nHost: " + host + b"\nConnection: close\n")
    assert client("POST", server + "/echo", "-", "5000", "a body\r\n") == (
        0, b"200 OK\n\nPOST /echo HTTP/1.1\nHost: " + host +
        b"\nConnection: close\nContent-Type: text/plain\nContent-Length: 8\na body\r\n")
    assert client("HEAD", server + "/head", "content-length", "5000") == (0, b"200 OK\n6\n")  # no body follows


@GCC_SKIP
@pytest.mark.parametrize("path, message", [
    ("/short", b"the connection closed before the whole body arrived"),
    ("/cut-chunk", b"the connection closed before the whole body arrived"),
    ("/bad-chunk", b"malformed chunk size 'zz'"),
    ("/not-http", b"not an HTTP response: it starts 'SSH-2.0-OpenSSH_9'"),
    ("/bad-status", b"not an HTTP response: it starts 'HTTP/1.1 -12 Odd'"),
    ("/bad-length", b"malformed Content-Length 'lots'"),
    ("/nothing", b"the connection closed in the middle of the response"),
])
def test_a_broken_response_is_an_error(server, client, path, message):
    assert _get(client, server + path) == (1, b"error: " + message + b"\n")


@GCC_SKIP
def test_a_server_that_does_not_answer(server, client):
    started = time.monotonic()
    assert _get(client, server + "/stall", timeout_ms=300) == (1, b"error: read failed: timed out\n")
    assert time.monotonic() - started < 2.5
    unused = socket.socket()
    unused.bind(("127.0.0.1", 0))
    port = unused.getsockname()[1]
    unused.close()
    status, out = _get(client, f"http://127.0.0.1:{port}/")
    assert status == 1 and out.startswith(b"error: could not connect to 127.0.0.1: ")  # (the reason is the system's)


@GCC_SKIP
@pytest.mark.parametrize("url, expected", [
    ("http://example.com", "Url(host: 'example.com', port: 80, target: '/')"),
    ("HTTP://example.com:8080/a/b?q=1#frag", "Url(host: 'example.com', port: 8080, target: '/a/b?q=1')"),
    ("http://example.com?q=1", "Url(host: 'example.com', port: 80, target: '/?q=1')"),
    ("http://[::1]:8080/x", "Url(host: '::1', port: 8080, target: '/x')"),
    ("http://[::1]", "Url(host: '::1', port: 80, target: '/')"),
    ("https://example.com/", "error: https is not supported: there is no TLS yet"),
    ("ftp://example.com/", "error: unsupported URL scheme 'ftp'"),
    ("example.com/x", "error: not a URL (it has no scheme): 'example.com/x'"),
    ("http://example.com:http/", "error: malformed port in URL: 'example.com:http'"),
    ("http://example.com:70000/", "error: malformed port in URL: 'example.com:70000'"),
    ("http://[::1/", "error: malformed address in URL: '[::1'"),
    ("http://user@example.com/", "error: a URL with a user name is not supported"),
    ("http:///path", "error: a URL needs a host: 'http:///path'"),
    ("http://example.com/a b", "error: a URL can't contain spaces or control characters"),
])
def test_parse_url(client, url, expected):
    status, out = client("url", url)
    assert (status, out.decode()) == (1 if expected.startswith("error") else 0, expected + "\n")


@GCC_SKIP
def test_a_request_that_cannot_be_sent(server, client):
    assert client("GE T", server + "/hello", "-", "5000") == (1, b"error: not an HTTP method: 'GE T'\n")


@GCC_SKIP
def test_get_and_post_on_every_target(server, tmp_path_factory):
    def run(target):
        exe = _built(tmp_path_factory, GET_AND_POST, target)
        return run_binary(target, [exe, server], capture_output=True, timeout=60)
    host = server[len("http://"):].encode()
    r = on_every_target(run)
    assert (r.returncode, r.stderr) == (0, b"")
    assert r.stdout == (b"200\nhello\nPOST /echo HTTP/1.1\nHost: " + host +
                        b"\nConnection: close\nContent-Type: text/x-test\nContent-Length: 4\nsent\n")


@GCC_SKIP
def test_the_fetch_example(server, tmp_path):
    exe = tmp_path / executable_name("fetch", host_target())
    build_executable(str(EXAMPLE), str(exe))
    r = run_binary(host_target(), [exe, server + "/chunked"], capture_output=True, timeout=30)
    assert (r.returncode, r.stdout, r.stderr) == (0, b"hello 0123456789", b"200 OK\n")
    r = run_binary(host_target(), [exe, server + "/missing"], capture_output=True, timeout=30)
    assert (r.returncode, r.stdout, r.stderr) == (1, b"gone\n", b"404 Not Found\n")
    r = run_binary(host_target(), [exe, "https://example.com/"], capture_output=True, timeout=30)
    assert (r.returncode, r.stdout) == (1, b"") and r.stderr == b"fetch: https is not supported: there is no TLS yet\n"
