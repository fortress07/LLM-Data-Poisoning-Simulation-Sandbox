from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, "src")
if SOURCE not in sys.path:
    sys.path.insert(0, SOURCE)
