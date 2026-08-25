from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from typing import List, Optional, Tuple

from ..safety import UnsafeInput, is_world_writable, owned_by_current_user

SOURCE_NAME = "poisonscan.c"
DIGEST_SUFFIX = ".sha256"


def source_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_c")


def source_path() -> str:
    return os.path.join(source_dir(), SOURCE_NAME)


def library_suffix() -> str:
    if sys.platform.startswith("win"):
        return ".dll"
    if sys.platform == "darwin":
        return ".dylib"
    return ".so"


def source_fingerprint() -> str:
    with open(source_path(), "rb") as handle:
        payload = handle.read()
    digest = hashlib.sha256(payload)
    digest.update(sysconfig.get_platform().encode("utf-8"))
    return digest.hexdigest()[:12]


def user_cache_root() -> str:
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return os.path.join(base, "poisonlab", "accel")
    if sys.platform == "darwin":
        return os.path.join(os.path.expanduser("~"), "Library", "Caches", "poisonlab", "accel")
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "poisonlab", "accel")


def _usable(directory: str, create: bool = True) -> bool:
    try:
        if create:
            os.makedirs(directory, exist_ok=True)
            if not sys.platform.startswith("win"):
                os.chmod(directory, 0o700)
        elif not os.path.isdir(directory):
            return False
    except OSError:
        return False
    if os.path.islink(directory):
        return False
    if is_world_writable(directory):
        return False
    if not owned_by_current_user(directory):
        return False
    probe = os.path.join(directory, ".writable")
    try:
        with open(probe, "w", encoding="utf-8") as handle:
            handle.write("ok")
        os.remove(probe)
    except OSError:
        return False
    return True


def cache_dir() -> str:
    override = os.environ.get("POISONLAB_ACCEL_DIR")
    if override:
        resolved = os.path.abspath(override)
        if not _usable(resolved):
            raise UnsafeInput(
                "POISONLAB_ACCEL_DIR %s is a symlink, world writable, or not owned by this user"
                % resolved
            )
        return resolved
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_bin")
    if _usable(local):
        return local
    private = user_cache_root()
    if _usable(private):
        return private
    marker = os.getuid() if hasattr(os, "getuid") else os.getpid()
    fallback = os.path.join(tempfile.gettempdir(), "poisonlab-accel-%d" % marker)
    _usable(fallback)
    return fallback


def library_path() -> str:
    name = "poisonscan-%s%s" % (source_fingerprint(), library_suffix())
    return os.path.join(cache_dir(), name)


def digest_path(target: str) -> str:
    return target + DIGEST_SUFFIX


def file_digest(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_digest(target: str) -> str:
    value = file_digest(target)
    with open(digest_path(target), "w", encoding="utf-8") as handle:
        handle.write(value)
    return value


def verify_digest(target: str) -> bool:
    sidecar = digest_path(target)
    if not os.path.exists(sidecar):
        return False
    try:
        with open(sidecar, "r", encoding="utf-8") as handle:
            expected = handle.read().strip()
        return bool(expected) and expected == file_digest(target)
    except OSError:
        return False


def find_compiler() -> Optional[str]:
    candidates: List[str] = []
    env = os.environ.get("CC")
    if env:
        candidates.append(env)
    candidates.extend(["cc", "gcc", "clang"])
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def compile_library(verbose: bool = False) -> Tuple[bool, str]:
    compiler = find_compiler()
    if compiler is None:
        return False, "no C compiler found (looked for CC, cc, gcc, clang)"
    target = library_path()
    os.makedirs(os.path.dirname(target), exist_ok=True)
    command = [
        compiler,
        "-O3",
        "-std=c11",
        "-fPIC",
        "-shared",
        "-fvisibility=hidden",
        "-o",
        target,
        source_path(),
    ]
    if sys.platform.startswith("win"):
        command.remove("-fPIC")
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=180,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return False, "compiler invocation failed: %s" % error
    output = result.stdout.decode("utf-8", "replace").strip()
    if verbose and output:
        sys.stderr.write(output + "\n")
    if result.returncode != 0:
        return False, "compilation failed: %s" % (output or result.returncode)
    record_digest(target)
    return True, target


def ensure_library(auto_build: bool = True, verbose: bool = False) -> Tuple[Optional[str], str]:
    try:
        target = library_path()
    except UnsafeInput as error:
        return None, str(error)
    if os.path.exists(target):
        if os.path.islink(target):
            return None, "refusing to load a symlinked accelerator at %s" % target
        if verify_digest(target):
            return target, "cached"
        if not auto_build:
            return None, "cached accelerator failed its integrity check"
        try:
            os.remove(target)
        except OSError:
            return None, "cached accelerator failed its integrity check"
    if not auto_build:
        return None, "not built"
    ok, detail = compile_library(verbose=verbose)
    if ok:
        return detail, "compiled"
    return None, detail
