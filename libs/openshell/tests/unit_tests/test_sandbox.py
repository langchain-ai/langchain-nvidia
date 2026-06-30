"""Unit tests for :class:`OpenShellSandbox`.

Covers the SP-2 (execute), SP-3 (upload_files), and SP-4 (download_files)
acceptance criteria from the spec, plus the construction/parameter contract.

The :class:`FakeSandbox` fixture (see ``conftest.py``) simulates the slice of
the OpenShell SDK we depend on, so these tests do not require the
``openshell`` package or a running OpenShell gateway.
"""

from __future__ import annotations

import asyncio
import base64

import pytest

from langchain_nvidia_openshell import OpenShellSandbox
from langchain_nvidia_openshell.sandbox import (
    _DOWNLOAD_BOOTSTRAP,
    _UPLOAD_BOOTSTRAP,
)

from .conftest import (
    FakeExecResult,
    FakeSandbox,
    make_filesystem_handler,
)

# ---------------------------------------------------------------------------
# Construction / argument validation
# ---------------------------------------------------------------------------


def test_inline_bootstraps_compile() -> None:
    compile(_UPLOAD_BOOTSTRAP, "<openshell-upload-bootstrap>", "exec")
    compile(_DOWNLOAD_BOOTSTRAP, "<openshell-download-bootstrap>", "exec")


def test_id_proxies_underlying_sandbox(fake_sandbox: FakeSandbox) -> None:
    sb = OpenShellSandbox(sandbox=fake_sandbox)
    assert sb.id == fake_sandbox.id == "openshell-fake-1"


def test_rejects_empty_shell(fake_sandbox: FakeSandbox) -> None:
    with pytest.raises(ValueError, match="shell"):
        OpenShellSandbox(sandbox=fake_sandbox, shell=())


def test_rejects_nonpositive_max_output_bytes(fake_sandbox: FakeSandbox) -> None:
    with pytest.raises(ValueError, match="max_output_bytes"):
        OpenShellSandbox(sandbox=fake_sandbox, max_output_bytes=0)


def test_rejects_nonpositive_max_upload_chunk_bytes(fake_sandbox: FakeSandbox) -> None:
    with pytest.raises(ValueError, match="max_upload_chunk_bytes"):
        OpenShellSandbox(sandbox=fake_sandbox, max_upload_chunk_bytes=-1)


# ---------------------------------------------------------------------------
# execute()
# ---------------------------------------------------------------------------


def test_execute_returns_stdout_and_zero_exit(fake_sandbox: FakeSandbox) -> None:
    fake_sandbox.queue(FakeExecResult(exit_code=0, stdout="hi\n"))
    sb = OpenShellSandbox(sandbox=fake_sandbox)

    result = sb.execute("echo hi")

    assert result.output == "hi\n"
    assert result.exit_code == 0
    assert result.truncated is False
    # Argv was wrapped with the default shell prefix
    assert fake_sandbox.calls[0].command == ["bash", "-c", "echo hi"]


def test_execute_sends_multiline_default_shell_script_over_stdin(
    fake_sandbox: FakeSandbox,
) -> None:
    fake_sandbox.queue(FakeExecResult(exit_code=0, stdout="ok\n"))
    sb = OpenShellSandbox(sandbox=fake_sandbox)

    result = sb.execute("python3 - <<'PY'\nprint('ok')\nPY")

    assert result.exit_code == 0
    assert fake_sandbox.calls[0].command == ["bash", "-s"]
    assert fake_sandbox.calls[0].stdin == b"python3 - <<'PY'\nprint('ok')\nPY"


def test_execute_sends_multiline_custom_shell_script_over_stdin(
    fake_sandbox: FakeSandbox,
) -> None:
    fake_sandbox.queue(FakeExecResult())
    sb = OpenShellSandbox(sandbox=fake_sandbox, shell=("sh", "-c"))

    sb.execute("printf '%s\\n' one\nprintf '%s\\n' two")

    assert fake_sandbox.calls[0].command == ["sh", "-s"]


def test_execute_propagates_nonzero_exit(fake_sandbox: FakeSandbox) -> None:
    fake_sandbox.queue(FakeExecResult(exit_code=1, stderr="bad command\n"))
    sb = OpenShellSandbox(sandbox=fake_sandbox)

    result = sb.execute("false")

    assert result.exit_code == 1
    assert "bad command" in result.output


def test_execute_rejects_empty_command(fake_sandbox: FakeSandbox) -> None:
    sb = OpenShellSandbox(sandbox=fake_sandbox)

    result = sb.execute("")

    assert result.exit_code == 2
    assert result.output.startswith("error:")
    # No exec call was issued
    assert fake_sandbox.calls == []


def test_execute_combines_stdout_and_stderr(fake_sandbox: FakeSandbox) -> None:
    fake_sandbox.queue(
        FakeExecResult(exit_code=0, stdout="OUT\n", stderr="ERR\n"),
    )
    sb = OpenShellSandbox(sandbox=fake_sandbox)

    result = sb.execute("noisy")

    assert result.output == "OUT\nERR\n"


def test_execute_truncates_when_output_exceeds_cap(fake_sandbox: FakeSandbox) -> None:
    payload = "x" * 200
    fake_sandbox.queue(FakeExecResult(exit_code=0, stdout=payload))
    sb = OpenShellSandbox(sandbox=fake_sandbox, max_output_bytes=100)

    result = sb.execute("dump")

    assert result.truncated is True
    assert len(result.output.encode("utf-8")) <= 100


def test_execute_truncation_preserves_byte_cap_for_multibyte_text(
    fake_sandbox: FakeSandbox,
) -> None:
    fake_sandbox.queue(FakeExecResult(exit_code=0, stdout="x" * 99 + "€"))
    sb = OpenShellSandbox(sandbox=fake_sandbox, max_output_bytes=100)

    result = sb.execute("dump")

    assert result.truncated is True
    assert result.output == "x" * 99
    assert len(result.output.encode("utf-8")) <= 100


def test_execute_forwards_default_timeout(fake_sandbox: FakeSandbox) -> None:
    fake_sandbox.queue(FakeExecResult())
    sb = OpenShellSandbox(sandbox=fake_sandbox, timeout=42)

    sb.execute("noop")

    assert fake_sandbox.calls[0].timeout_seconds == 42


def test_execute_overrides_timeout_per_call(fake_sandbox: FakeSandbox) -> None:
    fake_sandbox.queue(FakeExecResult())
    sb = OpenShellSandbox(sandbox=fake_sandbox, timeout=42)

    sb.execute("noop", timeout=7)

    assert fake_sandbox.calls[0].timeout_seconds == 7


def test_execute_zero_timeout_means_no_timeout(fake_sandbox: FakeSandbox) -> None:
    fake_sandbox.queue(FakeExecResult())
    sb = OpenShellSandbox(sandbox=fake_sandbox, timeout=42)

    sb.execute("noop", timeout=0)

    assert fake_sandbox.calls[0].timeout_seconds is None


def test_execute_maps_timeout_exception_to_124(fake_sandbox: FakeSandbox) -> None:
    fake_sandbox.raise_next(RuntimeError("deadline exceeded"))
    sb = OpenShellSandbox(sandbox=fake_sandbox)

    result = sb.execute("sleep 999", timeout=1)

    assert result.exit_code == 124
    assert "deadline" in result.output.lower()


def test_execute_maps_other_exception_to_exit_1(fake_sandbox: FakeSandbox) -> None:
    fake_sandbox.raise_next(RuntimeError("connection refused"))
    sb = OpenShellSandbox(sandbox=fake_sandbox)

    result = sb.execute("noop")

    assert result.exit_code == 1
    assert "connection refused" in result.output.lower()


def test_execute_honours_custom_shell(fake_sandbox: FakeSandbox) -> None:
    fake_sandbox.queue(FakeExecResult())
    sb = OpenShellSandbox(sandbox=fake_sandbox, shell=("sh", "-c"))

    sb.execute("echo hi")

    assert fake_sandbox.calls[0].command == ["sh", "-c", "echo hi"]


# ---------------------------------------------------------------------------
# upload_files()
# ---------------------------------------------------------------------------


def test_upload_files_round_trips_via_storage() -> None:
    sandbox = FakeSandbox()
    storage, handler = make_filesystem_handler()
    sandbox.set_handler(handler)
    sb = OpenShellSandbox(sandbox=sandbox)

    responses = sb.upload_files([("/sandbox/a.txt", b"hello")])

    assert responses == [type(responses[0])(path="/sandbox/a.txt", error=None)]
    assert storage["/sandbox/a.txt"] == b"hello"
    # The bootstrap is invoked with path/mode argv and a base64 stdin payload.
    call = sandbox.calls[0]
    assert call.command[3:] == ["/sandbox/a.txt", "wb"]
    assert call.env == {}
    assert base64.b64decode(call.stdin or b"") == b"hello"


def test_upload_files_round_trips_binary_content() -> None:
    sandbox = FakeSandbox()
    storage, handler = make_filesystem_handler()
    sandbox.set_handler(handler)
    sb = OpenShellSandbox(sandbox=sandbox)

    payload = bytes(range(256)) * 4
    responses = sb.upload_files([("/sandbox/blob.bin", payload)])

    assert responses[0].error is None
    assert storage["/sandbox/blob.bin"] == payload


def test_upload_files_rejects_relative_path(fake_sandbox: FakeSandbox) -> None:
    sb = OpenShellSandbox(sandbox=fake_sandbox)

    responses = sb.upload_files([("relative.txt", b"hi")])

    assert responses[0].error == "invalid_path"
    # No exec was issued
    assert fake_sandbox.calls == []


def test_upload_files_partial_success(fake_sandbox: FakeSandbox) -> None:
    """Bad path should be reported per-file, others must still be attempted."""
    storage, handler = make_filesystem_handler()
    fake_sandbox.set_handler(handler)
    sb = OpenShellSandbox(sandbox=fake_sandbox)

    responses = sb.upload_files(
        [
            ("bad", b"x"),
            ("/sandbox/good.txt", b"good"),
        ],
    )

    assert responses[0].error == "invalid_path"
    assert responses[1].error is None
    assert storage["/sandbox/good.txt"] == b"good"


def test_upload_files_chunks_large_payloads() -> None:
    sandbox = FakeSandbox()
    storage, handler = make_filesystem_handler()
    sandbox.set_handler(handler)
    # Use a small chunk size so we can prove the chunking path with a tiny payload
    sb = OpenShellSandbox(sandbox=sandbox, max_upload_chunk_bytes=4)
    payload = b"abcdefghij"  # 10 bytes -> 3 chunks of 4,4,2

    responses = sb.upload_files([("/sandbox/big.bin", payload)])

    assert responses[0].error is None
    assert storage["/sandbox/big.bin"] == payload
    upload_calls = [c for c in sandbox.calls if c.command[:2] == ["python3", "-c"]]
    assert len(upload_calls) == 3
    assert upload_calls[0].command[3:] == ["/sandbox/big.bin", "wb"]
    assert upload_calls[1].command[3:] == ["/sandbox/big.bin", "ab"]
    assert upload_calls[2].command[3:] == ["/sandbox/big.bin", "ab"]


def test_upload_files_maps_permission_error(fake_sandbox: FakeSandbox) -> None:
    fake_sandbox.queue(
        FakeExecResult(
            exit_code=1,
            stderr="PermissionError: [Errno 13] Permission denied: '/etc/x'\n",
        ),
    )
    sb = OpenShellSandbox(sandbox=fake_sandbox)

    responses = sb.upload_files([("/etc/x", b"hi")])

    assert responses[0].error == "permission_denied"


def test_upload_files_creates_empty_file() -> None:
    sandbox = FakeSandbox()
    storage, handler = make_filesystem_handler()
    sandbox.set_handler(handler)
    sb = OpenShellSandbox(sandbox=sandbox)

    sb.upload_files([("/sandbox/empty", b"")])

    assert storage["/sandbox/empty"] == b""


# ---------------------------------------------------------------------------
# download_files()
# ---------------------------------------------------------------------------


def test_download_files_returns_exact_bytes() -> None:
    sandbox = FakeSandbox()
    storage, handler = make_filesystem_handler({"/sandbox/x.bin": b"\x00\x01\x02"})
    sandbox.set_handler(handler)
    sb = OpenShellSandbox(sandbox=sandbox)

    responses = sb.download_files(["/sandbox/x.bin"])

    assert responses[0].content == b"\x00\x01\x02"
    assert responses[0].error is None


def test_download_files_round_trip_with_upload() -> None:
    sandbox = FakeSandbox()
    storage, handler = make_filesystem_handler()
    sandbox.set_handler(handler)
    sb = OpenShellSandbox(sandbox=sandbox)

    sb.upload_files([("/sandbox/payload.bin", bytes(range(64)))])
    [resp] = sb.download_files(["/sandbox/payload.bin"])

    assert resp.error is None
    assert resp.content == bytes(range(64))


def test_download_files_missing_returns_file_not_found() -> None:
    sandbox = FakeSandbox()
    storage, handler = make_filesystem_handler()
    sandbox.set_handler(handler)
    sb = OpenShellSandbox(sandbox=sandbox)

    [resp] = sb.download_files(["/sandbox/does_not_exist.txt"])

    assert resp.content is None
    assert resp.error == "file_not_found"


def test_download_files_rejects_relative_path(fake_sandbox: FakeSandbox) -> None:
    sb = OpenShellSandbox(sandbox=fake_sandbox)

    [resp] = sb.download_files(["relative.txt"])

    assert resp.error == "invalid_path"
    assert resp.content is None
    assert fake_sandbox.calls == []


def test_download_files_maps_is_directory(fake_sandbox: FakeSandbox) -> None:
    fake_sandbox.queue(
        FakeExecResult(exit_code=2, stderr="error: is a directory\n"),
    )
    sb = OpenShellSandbox(sandbox=fake_sandbox)

    [resp] = sb.download_files(["/sandbox/somedir"])

    assert resp.error == "is_directory"


def test_download_files_rejects_non_base64_output(fake_sandbox: FakeSandbox) -> None:
    payload = base64.b64encode(b"hello").decode("ascii")
    fake_sandbox.queue(FakeExecResult(exit_code=0, stdout=f"{payload} unexpected"))
    sb = OpenShellSandbox(sandbox=fake_sandbox)

    [resp] = sb.download_files(["/sandbox/file.txt"])

    assert resp.content is None
    assert resp.error is not None


# ---------------------------------------------------------------------------
# Async wrapper (inherited from BaseSandbox; sanity check the path)
# ---------------------------------------------------------------------------


def test_aexecute_proxies_to_execute(fake_sandbox: FakeSandbox) -> None:
    fake_sandbox.queue(FakeExecResult(exit_code=0, stdout="async\n"))
    sb = OpenShellSandbox(sandbox=fake_sandbox)

    result = asyncio.run(sb.aexecute("noop"))

    assert result.output == "async\n"
    assert result.exit_code == 0
