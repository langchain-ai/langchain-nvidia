"""Test configuration for the Switchyard LangChain package."""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
SOURCE_ROOT = PACKAGE_ROOT

sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT))
