"""Smoke import checks for ``langchain_nvidia_openshell``."""

from __future__ import annotations

import langchain_nvidia_openshell
from langchain_nvidia_openshell import OpenShellSandbox
from langchain_nvidia_openshell.sandbox import OpenShellSandbox as DirectImport


def test_package_imports() -> None:
    assert langchain_nvidia_openshell is not None


def test_openshell_sandbox_exported_from_top_level() -> None:
    assert OpenShellSandbox is DirectImport
    assert "OpenShellSandbox" in langchain_nvidia_openshell.__all__


def test_version_is_present() -> None:
    assert isinstance(langchain_nvidia_openshell.__version__, str)
    assert langchain_nvidia_openshell.__version__
