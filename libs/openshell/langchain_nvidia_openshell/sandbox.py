"""NVIDIA OpenShell sandbox backend implementation.

`OpenShellSandbox` adapts an
`openshell <https://github.com/NVIDIA/OpenShell>`_ sandbox to the LangChain
:class:`deepagents.backends.sandbox.BaseSandbox` contract, implementing the
"sandbox-as-tool" pattern documented at
https://docs.langchain.com/oss/python/deepagents/sandboxes.

Implementation notes
--------------------

* OpenShell's ``ExecSandbox`` RPC takes an argv ``Sequence[str]``, not a shell
  string. To honour ``BaseSandbox.execute(command: str)`` we wrap normal calls
  as ``[*shell, command]`` (default ``("bash", "-c")``). Multiline shell
  payloads are sent over stdin to ``bash -s`` / ``sh -s`` because tested
  OpenShell releases reject newline-bearing ``bash -c`` argv payloads at the
  RPC layer.
* OpenShell does not expose a typed file upload / download SDK call. We
  implement those over ``exec`` itself: stdin-fed base64 for uploads and
  stdout-captured base64 for downloads. Two tiny inline ``python3 -c``
  bootstraps run inside the sandbox.
* This wrapper is **lifecycle-agnostic**. The user passes in an already
  constructed (and entered) ``openshell.Sandbox`` or ``SandboxSession`` and is
  responsible for tearing it down. This matches the convention of the Daytona,
  Modal, and Runloop partner integrations.
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileOperationError,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


# Inline bootstraps run inside the sandbox via ``python3 -c <bootstrap>``.
# Both are kept as compact one-liners (statements separated by ``;``) so the
# argv stays a single small token, well under the gRPC message-size budget.
#
# Upload: read stdin (base64-encoded payload), decode, write to the path argv.
# Mode is ``wb`` by default (truncate + write); large uploads pass ``ab`` on
# subsequent chunks. ``os.makedirs`` ensures the parent exists when the policy
# permits.
_UPLOAD_BOOTSTRAP = (
    "import base64,os,sys;"
    "data=base64.b64decode(sys.stdin.buffer.read());"
    "p=sys.argv[1];"
    "m=sys.argv[2];"
    "d=os.path.dirname(p);"
    "(os.makedirs(d,exist_ok=True) if d else None);"
    "open(p,m).write(data)"
)

# Download: read the path argv and write base64 of its bytes to stdout.
# Distinguishes "is a directory" because that has its own FileOperationError
# code on the LangChain side.
_DOWNLOAD_BOOTSTRAP = (
    "import base64,os,sys;"
    "p=sys.argv[1];"
    "os.path.isdir(p) and (sys.stderr.write('error: is a directory\\n'),"
    "sys.exit(2));"
    "sys.stdout.buffer.write(base64.b64encode(open(p,'rb').read()))"
)


@runtime_checkable
class _OpenShellExec(Protocol):
    """Structural type for the slice of the OpenShell SDK we depend on.

    Matches both ``openshell.Sandbox`` (post ``__enter__``) and
    ``openshell.SandboxSession``. Declaring this here means our unit tests do
    not need to import ``openshell`` to construct a fake.
    """

    @property
    def id(self) -> str:
        ...

    def exec(  # noqa: A003 - matches OpenShell SDK method name
        self,
        command: Sequence[str],
        *,
        stream_output: bool = ...,
        workdir: str | None = ...,
        env: Mapping[str, str] | None = ...,
        stdin: bytes | None = ...,
        timeout_seconds: int | None = ...,
    ) -> Any:
        ...


# Default per-call timeout (seconds). 30 minutes mirrors the Daytona partner.
_DEFAULT_TIMEOUT_SECONDS = 30 * 60
# Maximum combined stdout+stderr returned to the LLM. 1 MiB keeps the agent
# context bounded against pathological commands like ``cat /var/log/...``.
_DEFAULT_MAX_OUTPUT_BYTES = 1 << 20
# Tested OpenShell VM drivers reject stdin payloads before gRPC's nominal
# 4 MiB default. Uploads are base64-encoded first, so keep raw chunks at a
# measured-safe size.
_DEFAULT_MAX_UPLOAD_CHUNK_BYTES = 512 * 1024
# Exit code we synthesize for our own preconditions / timeouts (matches the
# coreutils convention used by Daytona's wrapper).
_TIMEOUT_EXIT_CODE = 124
_PRECONDITION_EXIT_CODE = 2


class OpenShellSandbox(BaseSandbox):
    """LangChain :class:`BaseSandbox` backed by an NVIDIA OpenShell sandbox.

    Args:
        sandbox: An already-entered ``openshell.Sandbox`` (or a
            ``openshell.SandboxSession``) ready to accept ``exec`` calls. The
            caller owns the lifecycle - this wrapper does not start, stop, or
            destroy the sandbox.
        timeout: Default per-call timeout in seconds for :meth:`execute` when
            the caller does not pass an explicit ``timeout``. Values ``<= 0``
            are interpreted as "no timeout".
        shell: Argv prefix used to translate the ``command: str`` LangChain
            contract into the ``Sequence[str]`` OpenShell expects. Defaults to
            ``("bash", "-c")``. Override (e.g. ``("sh", "-c")``) for sandbox
            images without bash.
        max_output_bytes: Hard cap on combined stdout+stderr returned in
            :class:`~deepagents.backends.protocol.ExecuteResponse`. Anything
            longer is truncated and ``truncated=True`` is set.
        max_upload_chunk_bytes: Maximum payload size per ``upload_files``
            ``exec`` call before we split into multiple appends.
    """

    def __init__(
        self,
        *,
        sandbox: _OpenShellExec,
        timeout: int = _DEFAULT_TIMEOUT_SECONDS,
        shell: tuple[str, ...] = ("bash", "-c"),
        max_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES,
        max_upload_chunk_bytes: int = _DEFAULT_MAX_UPLOAD_CHUNK_BYTES,
    ) -> None:
        if not shell:
            raise ValueError("`shell` must contain at least one element")
        if max_output_bytes <= 0:
            raise ValueError("`max_output_bytes` must be positive")
        if max_upload_chunk_bytes <= 0:
            raise ValueError("`max_upload_chunk_bytes` must be positive")
        self._sandbox = sandbox
        self._default_timeout = timeout
        self._shell = tuple(shell)
        self._max_output_bytes = max_output_bytes
        self._max_upload_chunk_bytes = max_upload_chunk_bytes

    # ------------------------------------------------------------------
    # SandboxBackendProtocol surface
    # ------------------------------------------------------------------

    @property
    def id(self) -> str:
        """Return the underlying OpenShell sandbox id."""
        return self._sandbox.id

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Run ``command`` as a shell string inside the sandbox.

        Args:
            command: Shell command. Wrapped as ``[*shell, command]`` and sent
                to OpenShell as a single argv RPC.
            timeout: Maximum seconds to wait. ``None`` falls back to the
                instance default. Values ``<= 0`` disable the timeout.

        Returns:
            :class:`ExecuteResponse` whose ``output`` is the combined
            stdout + stderr (separated by a newline when both are non-empty),
            ``exit_code`` is the underlying process exit code, and
            ``truncated`` indicates whether the output was clipped to
            ``max_output_bytes``.
        """
        if not command:
            return ExecuteResponse(
                output="error: command must not be empty",
                exit_code=_PRECONDITION_EXIT_CODE,
                truncated=False,
            )

        effective_timeout = self._resolve_timeout(timeout)
        argv, stdin = self._build_shell_invocation(command)
        try:
            result = self._sandbox.exec(
                argv,
                stdin=stdin,
                timeout_seconds=effective_timeout,
            )
        except Exception as exc:  # noqa: BLE001 - SDK wraps a wide set of grpc errors
            return _exception_to_response(exc)

        return self._build_execute_response(
            stdout=getattr(result, "stdout", "") or "",
            stderr=getattr(result, "stderr", "") or "",
            exit_code=getattr(result, "exit_code", None),
        )

    def _build_shell_invocation(self, command: str) -> tuple[list[str], bytes | None]:
        """Build the OpenShell argv/stdin pair for a shell command string."""
        if "\n" not in command:
            return list(self._shell) + [command], None

        if len(self._shell) >= 2 and self._shell[1] == "-c":
            return [self._shell[0], "-s", *self._shell[2:]], command.encode("utf-8")

        return list(self._shell) + [command], None

    def upload_files(
        self,
        files: list[tuple[str, bytes]],
    ) -> list[FileUploadResponse]:
        """Upload one or more files into the sandbox.

        Per LangChain's contract, partial success is supported: each file's
        outcome is reported in its own :class:`FileUploadResponse`; we never
        raise out of this call.
        """
        responses: list[FileUploadResponse] = []
        for path, content in files:
            responses.append(self._upload_one(path, content))
        return responses

    def download_files(
        self,
        paths: list[str],
    ) -> list[FileDownloadResponse]:
        """Download one or more files from the sandbox."""
        responses: list[FileDownloadResponse] = []
        for path in paths:
            responses.append(self._download_one(path))
        return responses

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_timeout(self, timeout: int | None) -> int | None:
        """Compute the ``timeout_seconds`` argument for an ``exec`` call.

        Returns ``None`` when callers explicitly want "no timeout" so that the
        OpenShell SDK can in turn pass ``0`` (its no-timeout sentinel).
        """
        effective = timeout if timeout is not None else self._default_timeout
        if effective is None or effective <= 0:
            return None
        return effective

    def _build_execute_response(
        self,
        *,
        stdout: str,
        stderr: str,
        exit_code: int | None,
    ) -> ExecuteResponse:
        """Combine stdout / stderr and apply the truncation cap."""
        if stdout and stderr:
            joiner = "" if stdout.endswith("\n") else "\n"
            output = stdout + joiner + stderr
        else:
            output = stdout or stderr
        truncated = False
        encoded = output.encode("utf-8")
        if len(encoded) > self._max_output_bytes:
            output = encoded[: self._max_output_bytes].decode("utf-8", errors="ignore")
            truncated = True
        return ExecuteResponse(
            output=output,
            exit_code=exit_code,
            truncated=truncated,
        )

    def _upload_one(self, path: str, content: bytes) -> FileUploadResponse:
        if not path.startswith("/"):
            return FileUploadResponse(path=path, error="invalid_path")

        if not content:
            # Empty file: still issue a single ``wb`` exec so the file is
            # created (or truncated) even when there is no payload.
            return self._upload_chunk(path, b"", append=False)

        # Single-shot when payload fits one chunk; otherwise truncate-then-append
        # so the on-disk file matches the source bytes exactly.
        if len(content) <= self._max_upload_chunk_bytes:
            return self._upload_chunk(path, content, append=False)

        chunks = [
            content[i : i + self._max_upload_chunk_bytes]
            for i in range(0, len(content), self._max_upload_chunk_bytes)
        ]
        first = self._upload_chunk(path, chunks[0], append=False)
        if first.error is not None:
            return first
        for chunk in chunks[1:]:
            response = self._upload_chunk(path, chunk, append=True)
            if response.error is not None:
                return response
        return FileUploadResponse(path=path, error=None)

    def _upload_chunk(
        self,
        path: str,
        content: bytes,
        *,
        append: bool,
    ) -> FileUploadResponse:
        stdin = base64.b64encode(content)
        mode = "ab" if append else "wb"
        try:
            result = self._sandbox.exec(
                ["python3", "-c", _UPLOAD_BOOTSTRAP, path, mode],
                stdin=stdin,
                timeout_seconds=self._resolve_timeout(None),
            )
        except Exception as exc:  # noqa: BLE001
            return FileUploadResponse(
                path=path,
                error=_classify_fs_error(_format_exception(exc)),
            )

        exit_code = getattr(result, "exit_code", None)
        if exit_code == 0:
            return FileUploadResponse(path=path, error=None)
        return FileUploadResponse(
            path=path,
            error=_classify_fs_error(getattr(result, "stderr", "") or ""),
        )

    def _download_one(self, path: str) -> FileDownloadResponse:
        if not path.startswith("/"):
            return FileDownloadResponse(
                path=path,
                content=None,
                error="invalid_path",
            )

        try:
            result = self._sandbox.exec(
                ["python3", "-c", _DOWNLOAD_BOOTSTRAP, path],
                timeout_seconds=self._resolve_timeout(None),
            )
        except Exception as exc:  # noqa: BLE001
            return FileDownloadResponse(
                path=path,
                content=None,
                error=_classify_fs_error(_format_exception(exc)),
            )

        exit_code = getattr(result, "exit_code", None)
        stdout = getattr(result, "stdout", "") or ""
        stderr = getattr(result, "stderr", "") or ""
        if exit_code == 0:
            try:
                content = base64.b64decode(
                    stdout.strip().encode("ascii"),
                    validate=True,
                )
            except (ValueError, base64.binascii.Error) as exc:  # type: ignore[attr-defined]
                return FileDownloadResponse(
                    path=path,
                    content=None,
                    error=_classify_fs_error(f"decode_failed: {exc}"),
                )
            return FileDownloadResponse(path=path, content=content, error=None)
        return FileDownloadResponse(
            path=path,
            content=None,
            error=_classify_fs_error(stderr),
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _format_exception(exc: BaseException) -> str:
    """Render a brief, single-line summary of ``exc``."""
    text = f"{type(exc).__name__}: {exc}".strip()
    return text.splitlines()[0] if text else type(exc).__name__


def _exception_to_response(exc: BaseException) -> ExecuteResponse:
    """Map an SDK exception thrown out of ``exec`` to an :class:`ExecuteResponse`."""
    message = _format_exception(exc)
    lowered = message.lower()
    if "timeout" in lowered or "timed out" in lowered or "deadline" in lowered:
        return ExecuteResponse(
            output=f"error: {message}",
            exit_code=_TIMEOUT_EXIT_CODE,
            truncated=False,
        )
    return ExecuteResponse(
        output=f"error: {message}",
        exit_code=1,
        truncated=False,
    )


def _classify_fs_error(text: str) -> FileOperationError:
    """Map sandbox-side stderr / SDK exception text to a ``FileOperationError``.

    The LangChain :class:`~deepagents.backends.protocol.FileOperationError`
    literal has four members. Anything we cannot positively identify falls
    back to ``"permission_denied"`` because OpenShell sandboxes are
    Landlock-enforced and unclassified failures are most commonly policy
    rejections. Returning a literal (rather than a free-form stderr line)
    keeps the response type-safe and easy for an LLM to switch on.
    """
    lowered = text.lower()
    if "no such file" in lowered or "file not found" in lowered:
        return "file_not_found"
    if "is a directory" in lowered:
        return "is_directory"
    if "invalid" in lowered and "path" in lowered:
        return "invalid_path"
    return "permission_denied"
