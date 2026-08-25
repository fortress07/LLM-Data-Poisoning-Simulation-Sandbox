from __future__ import annotations

import os
import sys
import unittest
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

POSIX = os.name != "nt"


def collect() -> List[Tuple[str, bool, str]]:
    loader = unittest.TestLoader()
    suite = loader.discover(os.path.join(ROOT, "tests"), top_level_dir=ROOT)
    found: List[Tuple[str, bool, str]] = []
    stack = [suite]
    while stack:
        item = stack.pop()
        if isinstance(item, unittest.TestSuite):
            stack.extend(list(item))
            continue
        method = getattr(item, item._testMethodName, None)
        skipped = bool(
            getattr(method, "__unittest_skip__", False)
            or getattr(item.__class__, "__unittest_skip__", False)
        )
        reason = str(
            getattr(method, "__unittest_skip_why__", "")
            or getattr(item.__class__, "__unittest_skip_why__", "")
        )
        found.append((item.id(), skipped, reason))
    return sorted(found)


def main() -> int:
    tests = collect()
    platform_gated = [
        (name, skipped, reason)
        for name, skipped, reason in tests
        if "posix" in reason.lower() or "windows" in reason.lower()
    ]
    print("%d tests discovered, %d gated on the platform" % (len(tests), len(platform_gated)))
    for name, skipped, reason in platform_gated:
        print("  %-6s %s (%s)" % ("skip" if skipped else "run", name, reason))

    failures: List[str] = []
    if POSIX:
        for name, skipped, reason in platform_gated:
            if skipped and "posix" in reason.lower():
                failures.append("%s is gated on posix but skipped while running on posix" % name)
    else:
        running = [name for name, skipped, _ in platform_gated if not skipped]
        if not platform_gated:
            failures.append("no platform gated tests found, the guard has nothing to protect")
        print("  note: %d platform gated tests run on this platform" % len(running))

    unreachable = [name for name, skipped, reason in platform_gated if skipped and not POSIX]
    if unreachable:
        print()
        print(
            "the following tests never execute on this platform, so their logic must also be "
            "covered by a platform independent test:"
        )
        for name in unreachable:
            print("  %s" % name)

    if failures:
        print()
        for line in failures:
            sys.stderr.write("%s\n" % line)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
