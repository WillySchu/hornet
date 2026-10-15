// TCP connections, for stdlib/net.ht.
//
// A socket is an int on POSIX and a SOCKET on Windows, where it isn't a file descriptor either:
// reading, writing, and closing one have calls of their own. Hornet sees an int64 everywhere.
#include "internal.h"
#ifndef _WIN32
#include <netinet/in.h>
#endif

#ifdef _WIN32
typedef SOCKET hornet_socket;
#define HORNET_NO_SOCKET INVALID_SOCKET
#define hornet_socket_close closesocket
#else
typedef int hornet_socket;
#define HORNET_NO_SOCKET (-1)
#define hornet_socket_close close
#endif

// Sending to a connection the other side has closed raises SIGPIPE, which would end the program
// without a word; an error from the call is wanted instead. Linux asks per send, macOS per socket
// (SO_NOSIGPIPE, below), and Windows has no such signal.
#ifdef MSG_NOSIGNAL
#define HORNET_SEND_FLAGS MSG_NOSIGNAL
#else
#define HORNET_SEND_FLAGS 0
#endif

static char hornet_net_error_text[160] = "";

// Why the most recent hornet_tcp_* call failed.
const char *hornet_net_error(void) {
    return hornet_net_error_text;
}

static void hornet_net_set_error(const char *text) {
    snprintf(hornet_net_error_text, sizeof hornet_net_error_text, "%s", text);
}

// Record the system's reason for the socket call that just failed.
static void hornet_net_failed(void) {
#ifdef _WIN32
    int code = WSAGetLastError();
    if (code == WSAETIMEDOUT) {
        hornet_net_set_error("timed out");
        return;
    }
    char text[128];
    DWORD n = FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, (DWORD)code, 0,
                             text, sizeof text, NULL);
    while (n > 0 && (text[n - 1] == '\n' || text[n - 1] == '\r' || text[n - 1] == '.' || text[n - 1] == ' ')) {
        text[--n] = '\0';
    }
    if (n == 0) {
        snprintf(text, sizeof text, "Winsock error %d", code);
    }
    hornet_net_set_error(text);
#else
    // A socket with a timeout reports one as "try again".
    hornet_net_set_error(errno == EAGAIN || errno == EWOULDBLOCK ? "timed out" : strerror(errno));
#endif
}

// Winsock, started once before the first socket. 0 if it isn't available.
static int hornet_net_start(void) {
#ifdef _WIN32
    static int started = 0;
    WSADATA data;
    if (!started && WSAStartup(MAKEWORD(2, 2), &data) != 0) {
        hornet_net_set_error("Winsock is not available");
        return 0;
    }
    started = 1;
#endif
    return 1;
}

// The addresses of `host` (NULL: every one of this machine's, to listen on) and `port`; NULL if
// there are none.
static struct addrinfo *hornet_net_addresses(const char *host, int64_t port) {
    char service[16];
    snprintf(service, sizeof service, "%d", (int)port);
    struct addrinfo hints, *found = NULL;
    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = host == NULL ? AI_PASSIVE : 0;
    int rc = getaddrinfo(host, service, &hints, &found);
    if (rc != 0) {
#ifdef _WIN32
        WSASetLastError(rc);
        hornet_net_failed();
#else
        hornet_net_set_error(rc == EAI_SYSTEM ? strerror(errno) : gai_strerror(rc));
#endif
        return NULL;
    }
    return found;
}

// A positive `timeout_ms` bounds each read and write on `s`; and sending to a closed connection is
// an error, not SIGPIPE.
static void hornet_net_configure(hornet_socket s, int64_t timeout_ms) {
    if (timeout_ms > 0) {
#ifdef _WIN32
        DWORD timeout = (DWORD)timeout_ms;
#else
        struct timeval timeout = {(time_t)(timeout_ms / 1000), (suseconds_t)(timeout_ms % 1000 * 1000)};
#endif
        setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char *)&timeout, sizeof timeout);
        setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, (const char *)&timeout, sizeof timeout);
    }
#ifdef SO_NOSIGPIPE
    int on = 1;
    setsockopt(s, SOL_SOCKET, SO_NOSIGPIPE, &on, sizeof on);
#endif
}

// Connect to `host` (a name or an address) on `port`, trying each address the name has. A positive
// `timeout_ms` bounds each later read and write (and, on Linux, the connecting itself). The socket,
// or -1.
int64_t hornet_tcp_connect(const char *host, int64_t port, int64_t timeout_ms) {
    if (!hornet_net_start()) {
        return -1;
    }
    struct addrinfo *found = hornet_net_addresses(host, port);
    if (found == NULL) {
        return -1;
    }
    hornet_socket s = HORNET_NO_SOCKET;
    hornet_net_set_error("no address");
    for (struct addrinfo *a = found; a != NULL; a = a->ai_next) {
        s = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (s == HORNET_NO_SOCKET) {
            hornet_net_failed();
            continue;
        }
        hornet_net_configure(s, timeout_ms);
        if (connect(s, a->ai_addr, (socklen_t)a->ai_addrlen) == 0) {
            break;
        }
#ifndef _WIN32
        if (errno == EINPROGRESS) {  // what a connect that outlasts SO_SNDTIMEO reports
            errno = EAGAIN;
        }
#endif
        hornet_net_failed();
        hornet_socket_close(s);
        s = HORNET_NO_SOCKET;
    }
    freeaddrinfo(found);
    return s == HORNET_NO_SOCKET ? -1 : (int64_t)s;
}

// Up to `len` bytes from a connection: how many arrived, 0 once the other side has closed, -1 on
// an error.
int64_t hornet_tcp_read(int64_t sock, char *ptr, int64_t len) {
    for (;;) {
        int64_t n = recv((hornet_socket)sock, ptr, len > 1 << 30 ? 1 << 30 : (int)len, 0);
        if (n >= 0) {
            return n;
        }
#ifndef _WIN32
        if (errno == EINTR) {
            continue;
        }
#endif
        hornet_net_failed();
        return -1;
    }
}

// Send all `len` bytes to a connection; -1 on an error.
int64_t hornet_tcp_write(int64_t sock, const char *ptr, int64_t len) {
    int64_t done = 0;
    while (done < len) {
        int64_t left = len - done;
        int64_t n = send((hornet_socket)sock, ptr + done, left > 1 << 30 ? 1 << 30 : (int)left, HORNET_SEND_FLAGS);
        if (n < 0) {
#ifndef _WIN32
            if (errno == EINTR) {
                continue;
            }
#endif
            hornet_net_failed();
            return -1;
        }
        done += n;
    }
    return done;
}

int64_t hornet_tcp_close(int64_t sock) {
    return hornet_socket_close((hornet_socket)sock);
}

// Listen on `port` of `host` (an address or a name; "" for all of this machine's), on the first of
// its addresses that can be had. Port 0 asks the system for a free one (hornet_tcp_local_port). The
// listening socket, or -1.
int64_t hornet_tcp_listen(const char *host, int64_t port) {
    if (!hornet_net_start()) {
        return -1;
    }
    struct addrinfo *found = hornet_net_addresses(host[0] == '\0' ? NULL : host, port);
    if (found == NULL) {
        return -1;
    }
    hornet_socket s = HORNET_NO_SOCKET;
    hornet_net_set_error("no address");
    for (struct addrinfo *a = found; a != NULL; a = a->ai_next) {
        s = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (s == HORNET_NO_SOCKET) {
            hornet_net_failed();
            continue;
        }
#ifndef _WIN32
        // A port a previous run's connections still linger on can be listened on at once. (Windows'
        // option of that name lets another program take a port in use, so it isn't asked for there.)
        int on = 1;
        setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &on, sizeof on);
#endif
        if (bind(s, a->ai_addr, (socklen_t)a->ai_addrlen) == 0 && listen(s, 128) == 0) {
            break;
        }
        hornet_net_failed();
        hornet_socket_close(s);
        s = HORNET_NO_SOCKET;
    }
    freeaddrinfo(found);
    return s == HORNET_NO_SOCKET ? -1 : (int64_t)s;
}

// The next connection made to a listening socket, waiting for one. A positive `timeout_ms` bounds
// each read and write on it. The connection's socket, or -1.
int64_t hornet_tcp_accept(int64_t listener, int64_t timeout_ms) {
    for (;;) {
        hornet_socket s = accept((hornet_socket)listener, NULL, NULL);
        if (s != HORNET_NO_SOCKET) {
            hornet_net_configure(s, timeout_ms);
            return (int64_t)s;
        }
#ifndef _WIN32
        if (errno == EINTR || errno == ECONNABORTED) {  // (one that closed before it was taken)
            continue;
        }
#endif
        hornet_net_failed();
        return -1;
    }
}

// The port a socket is on, or -1.
int64_t hornet_tcp_local_port(int64_t sock) {
    struct sockaddr_storage address;
    socklen_t size = sizeof address;
    if (getsockname((hornet_socket)sock, (struct sockaddr *)&address, &size) != 0) {
        hornet_net_failed();
        return -1;
    }
    if (address.ss_family == AF_INET6) {
        return ntohs(((struct sockaddr_in6 *)&address)->sin6_port);
    }
    return ntohs(((struct sockaddr_in *)&address)->sin_port);
}

// Close a connection once what was written has gone: say nothing more will be, and read what the
// other side still sends (for up to a second, and a megabyte) until it closes too. Closing with
// what it sent unread would reset the connection, which can lose what was written before it arrives.
int64_t hornet_tcp_finish(int64_t sock) {
    hornet_socket s = (hornet_socket)sock;
#ifdef _WIN32
    shutdown(s, SD_SEND);
    DWORD wait = 1000;
#else
    shutdown(s, SHUT_WR);
    struct timeval wait = {1, 0};
#endif
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char *)&wait, sizeof wait);
    char unread[4096];
    for (int64_t drained = 0; drained < (1 << 20);) {
        int n = (int)recv(s, unread, sizeof unread, 0);
        if (n <= 0) {
            break;
        }
        drained += n;
    }
    return hornet_socket_close(s);
}
