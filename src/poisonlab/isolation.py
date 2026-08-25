from __future__ import annotations

import ipaddress
import os
import socket
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "WANDB_MODE": "offline",
    "WANDB_DISABLED": "true",
    "TOKENIZERS_PARALLELISM": "false",
    "NO_PROXY": "*",
    "no_proxy": "*",
}

LOOPBACK_NAMES = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
        "loopback",
    }
)

PROBE_HOSTS = (
    ("poisonlab-isolation-probe.invalid", 443),
    ("169.254.169.254", 80),
)


class NetworkIsolationError(RuntimeError):
    pass


def _hostname(address: Any) -> str:
    host = address[0] if isinstance(address, (tuple, list)) and address else address
    if isinstance(host, (bytes, bytearray)):
        host = host.decode("utf-8", "replace")
    text = str(host).strip().lower()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if "%" in text:
        text = text.split("%", 1)[0]
    return text


def is_loopback(address: Any) -> bool:
    host = address[0] if isinstance(address, (tuple, list)) and address else address
    if host is None:
        return True
    text = _hostname(address)
    if not text:
        return True
    if text in LOOPBACK_NAMES:
        return True
    try:
        parsed = ipaddress.ip_address(text)
    except ValueError:
        return False
    return bool(parsed.is_loopback or parsed.is_unspecified)


_is_loopback = is_loopback


class _Registry:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.originals: Dict[str, Any] = {}
        self.active: List["NetworkIsolation"] = []
        self.environment: Dict[str, Optional[str]] = {}

    def policy(self) -> Tuple[bool, bool]:
        active = list(self.active)
        allow_loopback = all(guard.allow_loopback for guard in active)
        strict = any(guard.strict for guard in active)
        return allow_loopback, strict

    def record(self, kind: str, target: Any) -> None:
        for guard in list(self.active):
            guard.violations.append((kind, str(target)))

    def enter(self, guard: "NetworkIsolation") -> None:
        with self.lock:
            if not self.active:
                self._install()
            self.active.append(guard)

    def leave(self, guard: "NetworkIsolation") -> None:
        with self.lock:
            if guard in self.active:
                self.active.remove(guard)
            if not self.active:
                self._restore()

    def installed(self) -> bool:
        return bool(self.active)

    def _install(self) -> None:
        self.originals = {
            "connect": socket.socket.connect,
            "connect_ex": socket.socket.connect_ex,
            "sendto": socket.socket.sendto,
            "create_connection": socket.create_connection,
            "getaddrinfo": socket.getaddrinfo,
            "gethostbyname": socket.gethostbyname,
            "gethostbyname_ex": socket.gethostbyname_ex,
        }
        registry = self

        def guarded_connect(sock, address, *args, **kwargs):
            allow, _ = registry.policy()
            if allow and is_loopback(address):
                return registry.originals["connect"](sock, address, *args, **kwargs)
            registry.record("connect", address)
            raise NetworkIsolationError("sandbox blocked an outbound connect to %s" % (address,))

        def guarded_connect_ex(sock, address, *args, **kwargs):
            allow, _ = registry.policy()
            if allow and is_loopback(address):
                return registry.originals["connect_ex"](sock, address, *args, **kwargs)
            registry.record("connect_ex", address)
            return 1

        def guarded_sendto(sock, data, *args, **kwargs):
            address = args[-1] if args else kwargs.get("address")
            allow, _ = registry.policy()
            if allow and is_loopback(address):
                return registry.originals["sendto"](sock, data, *args, **kwargs)
            registry.record("sendto", address)
            raise NetworkIsolationError("sandbox blocked an outbound datagram to %s" % (address,))

        def guarded_create_connection(address, *args, **kwargs):
            allow, _ = registry.policy()
            if allow and is_loopback(address):
                return registry.originals["create_connection"](address, *args, **kwargs)
            registry.record("create_connection", address)
            raise NetworkIsolationError("sandbox blocked an outbound connection to %s" % (address,))

        def guarded_getaddrinfo(host, *args, **kwargs):
            allow, _ = registry.policy()
            if allow and is_loopback(host):
                return registry.originals["getaddrinfo"](host, *args, **kwargs)
            registry.record("dns", host)
            raise NetworkIsolationError("sandbox blocked a dns lookup for %s" % (host,))

        def guarded_gethostbyname(host, *args, **kwargs):
            allow, _ = registry.policy()
            if allow and is_loopback(host):
                return registry.originals["gethostbyname"](host, *args, **kwargs)
            registry.record("dns", host)
            raise NetworkIsolationError("sandbox blocked a dns lookup for %s" % (host,))

        def guarded_gethostbyname_ex(host, *args, **kwargs):
            allow, _ = registry.policy()
            if allow and is_loopback(host):
                return registry.originals["gethostbyname_ex"](host, *args, **kwargs)
            registry.record("dns", host)
            raise NetworkIsolationError("sandbox blocked a dns lookup for %s" % (host,))

        socket.socket.connect = guarded_connect
        socket.socket.connect_ex = guarded_connect_ex
        socket.socket.sendto = guarded_sendto
        socket.create_connection = guarded_create_connection
        socket.getaddrinfo = guarded_getaddrinfo
        socket.gethostbyname = guarded_gethostbyname
        socket.gethostbyname_ex = guarded_gethostbyname_ex
        self.environment = {key: os.environ.get(key) for key in OFFLINE_ENVIRONMENT}
        os.environ.update(OFFLINE_ENVIRONMENT)

    def _restore(self) -> None:
        if not self.originals:
            return
        socket.socket.connect = self.originals["connect"]
        socket.socket.connect_ex = self.originals["connect_ex"]
        socket.socket.sendto = self.originals["sendto"]
        socket.create_connection = self.originals["create_connection"]
        socket.getaddrinfo = self.originals["getaddrinfo"]
        socket.gethostbyname = self.originals["gethostbyname"]
        socket.gethostbyname_ex = self.originals["gethostbyname_ex"]
        for key, value in self.environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.originals = {}
        self.environment = {}


_REGISTRY = _Registry()


@dataclass
class NetworkIsolation:
    allow_loopback: bool = True
    strict: bool = True
    violations: List[Tuple[str, str]] = field(default_factory=list)
    verified: Optional[bool] = None
    engaged: bool = False
    _entered: bool = False

    def __enter__(self) -> "NetworkIsolation":
        if self._entered:
            return self
        _REGISTRY.enter(self)
        self._entered = True
        self.engaged = True
        self.verified = self._probe()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not self._entered:
            return
        self._entered = False
        _REGISTRY.leave(self)

    def _probe(self) -> bool:
        keep = len(self.violations)
        blocked = 0
        for host, port in PROBE_HOSTS:
            try:
                socket.getaddrinfo(host, port)
            except NetworkIsolationError:
                blocked += 1
            except OSError:
                blocked += 1
        del self.violations[keep:]
        return blocked == len(PROBE_HOSTS)

    def report(self) -> Dict[str, Any]:
        return {
            "enforced": bool(self.engaged),
            "active": bool(self._entered and _REGISTRY.installed()),
            "verified": self.verified,
            "allow_loopback": self.allow_loopback,
            "strict": self.strict,
            "violations": [
                {"kind": kind, "target": target} for kind, target in self.violations
            ],
        }


def self_test(strict: bool = False) -> Dict[str, Any]:
    guard = NetworkIsolation(strict=strict)
    blocked = False
    bypass = False
    with guard:
        try:
            socket.create_connection(("example.com", 80), timeout=0.2)
        except Exception:
            blocked = True
        for host in ("localhost.poisonlab-probe.invalid", "127.0.0.1.poisonlab-probe.invalid"):
            try:
                socket.getaddrinfo(host, 80)
                bypass = True
            except NetworkIsolationError:
                pass
            except OSError:
                bypass = True
        report = guard.report()
    return {
        "blocked": blocked,
        "verified": report["verified"],
        "allowlist_bypass": bypass,
        "violations": report["violations"],
    }
