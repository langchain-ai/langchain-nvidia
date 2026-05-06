"""LangChain NVIDIA OpenShell sandbox integration.

The :class:`~langchain_nvidia_openshell.sandbox.OpenShellSandbox` adapter
exposes an `NVIDIA OpenShell <https://github.com/NVIDIA/OpenShell>`_ sandbox
to LangChain's :class:`~deepagents.backends.sandbox.BaseSandbox` interface.

This implements the **sandbox-as-tool** pattern: the agent runs on the host,
and tool calls (shell commands, file uploads, file downloads) are dispatched
into a policy-enforced sandbox over OpenShell's gRPC API.

Quickstart::

    import openshell
    from langchain_nvidia_openshell import OpenShellSandbox

    with openshell.Sandbox() as sb:
        backend = OpenShellSandbox(sandbox=sb)
        result = backend.execute("echo hello")
        print(result.output)
"""

from langchain_nvidia_openshell._version import __version__
from langchain_nvidia_openshell.sandbox import OpenShellSandbox

__all__ = [
    "OpenShellSandbox",
    "__version__",
]
