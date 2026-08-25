from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from typing import List, Optional, Tuple

from ..safety import UnsafeInput, directory_refusal, inspect_directory

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


def _create_private(directory: str) -> Optional[str]:
    try:
        os.makedirs(directory, mode=0o700, exist_ok=True)
        if os.name != "nt":
            os.chmod(directory, 0o700)
    except OSError as error:
        return "could not be created (%s)" % error
    return None


def _refuse(directory: str, create: bool = True) -> Optional[str]:
    facts = inspect_directory(directory)
    if not facts.exists and not facts.is_symlink:
        if not create:
            return "does not exist"
        failure = _create_private(directory)
        if failure:
            return failure
        facts = inspect_directory(directory)
    reason = directory_refusal(facts)
    if reason:
        return reason
    probe = os.path.join(directory, ".writable")
    try:
        with open(probe, "w", encoding="utf-8") as handle:
            handle.write("ok")
        os.remove(probe)
    except OSError as error:
        return "is not writable (%s)" % error
    return None


def _usable(directory: str, create: bool = True) -> bool:
    return _refuse(directory, create) is None


def cache_dir() -> str:
    override = os.environ.get("POISONLAB_ACCEL_DIR")
    if override:
        resolved = os.path.abspath(override)
        reason = _refuse(resolved)
        if reason:
            raise UnsafeInput("POISONLAB_ACCEL_DIR %s %s" % (resolved, reason))
        return resolved
    for candidate in _candidate_dirs():
        if _refuse(candidate) is None:
            return candidate
    raise UnsafeInput(
        "no private directory is available for the compiled accelerator, "
        "set POISONLAB_ACCEL_DIR to one you own or run with POISONLAB_ACCEL=off"
    )


def _candidate_dirs() -> List[str]:
    marker = os.geteuid() if hasattr(os, "geteuid") else os.getpid()
    return [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "_bin"),
        user_cache_root(),
        os.path.join(tempfile.gettempdir(), "poisonlab-accel-%d" % marker),
    ]


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
    try:
        target = library_path()
    except UnsafeInput as error:
        return False, str(error)
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
