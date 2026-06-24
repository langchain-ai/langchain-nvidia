"""Tests for the OpenShell notebook setup shell script."""

from __future__ import annotations

import os
import socket
import subprocess
import tempfile
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "sandboxes"
    / "setup_openshell.sh"
)


def _run_check(
    *,
    os_release: str | None = None,
    env: dict[str, str] | None = None,
    args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    test_env = os.environ.copy()
    test_env.update(env or {})
    test_env["OPENSHELL_SETUP_TESTING"] = "0"
    if os_release is not None:
        path = Path(os_release)
        test_env["OPENSHELL_TEST_OS_RELEASE"] = str(path)
    return subprocess.run(
        args or ["bash", str(SCRIPT), "--check-only", "--skip-system-install"],
        capture_output=True,
        text=True,
        env=test_env,
        timeout=90,
        check=False,
    )


def _ubuntu_release(tmp_path: Path, version: str) -> str:
    p = tmp_path / f"ubuntu-{version}.os-release"
    p.write_text(
        "\n".join(
            [
                "ID=ubuntu",
                f'VERSION_ID="{version}"',
                f'PRETTY_NAME="Ubuntu {version}"',
            ]
        )
        + "\n"
    )
    return str(p)


def test_ubuntu_18_is_rejected_before_setup(tmp_path: Path) -> None:
    result = _run_check(
        os_release=_ubuntu_release(tmp_path, "18.04"),
        env={
            "OPENSHELL_TEST_UNAME_S": "Linux",
            "OPENSHELL_TEST_UNAME_M": "x86_64",
            "OPENSHELL_TEST_GLIBC_VERSION": "2.27",
        },
    )

    assert result.returncode == 1
    assert "UNSUPPORTED: glibc 2.27 is below gateway floor 2.28" in result.stdout
    assert "UNSUPPORTED: Ubuntu 18.04 is unsupported" in result.stdout
    assert (
        "Stopping before environment changes because prerequisite validation failed"
        in result.stdout
    )


def test_ubuntu_19_is_rejected_even_when_glibc_floor_passes(tmp_path: Path) -> None:
    result = _run_check(
        os_release=_ubuntu_release(tmp_path, "19.10"),
        env={
            "OPENSHELL_TEST_UNAME_S": "Linux",
            "OPENSHELL_TEST_UNAME_M": "x86_64",
            "OPENSHELL_TEST_GLIBC_VERSION": "2.30",
        },
    )

    assert result.returncode == 1
    assert "PASS: glibc 2.30 satisfies gateway floor 2.28" in result.stdout
    assert "UNSUPPORTED: Ubuntu 19.10 is unsupported" in result.stdout


def test_ubuntu_20_reports_gateway_compatible_but_sdk_incompatible(
    tmp_path: Path,
) -> None:
    result = _run_check(
        os_release=_ubuntu_release(tmp_path, "20.04"),
        env={
            "OPENSHELL_TEST_UNAME_S": "Linux",
            "OPENSHELL_TEST_UNAME_M": "x86_64",
            "OPENSHELL_TEST_GLIBC_VERSION": "2.31",
        },
    )

    assert result.returncode == 1
    assert "PASS: glibc 2.31 satisfies gateway floor 2.28" in result.stdout
    assert "SDK wheels are manylinux_2_39; detected glibc 2.31" in result.stdout


def test_ubuntu_21_reports_gateway_compatible_but_sdk_incompatible(
    tmp_path: Path,
) -> None:
    result = _run_check(
        os_release=_ubuntu_release(tmp_path, "21.10"),
        env={
            "OPENSHELL_TEST_UNAME_S": "Linux",
            "OPENSHELL_TEST_UNAME_M": "x86_64",
            "OPENSHELL_TEST_GLIBC_VERSION": "2.34",
        },
    )

    assert result.returncode == 1
    assert "PASS: glibc 2.34 satisfies gateway floor 2.28" in result.stdout
    assert "SDK wheels are manylinux_2_39; detected glibc 2.34" in result.stdout


def test_ubuntu_22_reports_gateway_compatible_but_sdk_risky(tmp_path: Path) -> None:
    result = _run_check(
        os_release=_ubuntu_release(tmp_path, "22.04"),
        env={
            "OPENSHELL_TEST_UNAME_S": "Linux",
            "OPENSHELL_TEST_UNAME_M": "x86_64",
            "OPENSHELL_TEST_GLIBC_VERSION": "2.35",
        },
    )

    assert result.returncode == 1
    assert "PASS: glibc 2.35 satisfies gateway floor 2.28" in result.stdout
    assert "manylinux_2_39" in result.stdout
    assert "can install missing Python/Poetry prerequisites" not in result.stdout


def test_ubuntu_24_reaches_driver_validation(tmp_path: Path) -> None:
    result = _run_check(
        os_release=_ubuntu_release(tmp_path, "24.04"),
        env={
            "OPENSHELL_TEST_UNAME_S": "Linux",
            "OPENSHELL_TEST_UNAME_M": "x86_64",
            "OPENSHELL_TEST_GLIBC_VERSION": "2.39",
            "OPENSHELL_TEST_NO_DOCKER": "1",
        },
    )

    assert result.returncode == 1
    assert (
        "PASS: glibc 2.39 satisfies OpenShell SDK wheel tag floor 2.39"
        in result.stdout
    )
    assert (
        "PASS: pip can resolve openshell==0.0.68 for manylinux_2_39_x86_64"
        in result.stdout
    )
    assert (
        "FAIL: Docker is required for the notebook's local demo image path"
        in result.stdout
    )


def test_ubuntu_24_check_only_reports_installable_prereqs(tmp_path: Path) -> None:
    result = _run_check(
        os_release=_ubuntu_release(tmp_path, "24.04"),
        env={
            "OPENSHELL_TEST_UNAME_S": "Linux",
            "OPENSHELL_TEST_UNAME_M": "x86_64",
            "OPENSHELL_TEST_GLIBC_VERSION": "2.39",
            "OPENSHELL_TEST_NO_PYTHON": "1",
            "OPENSHELL_TEST_NO_POETRY": "1",
            "OPENSHELL_TEST_NO_DOCKER": "1",
        },
    )

    assert result.returncode == 1
    assert (
        "CHECK_ONLY: compatible Ubuntu host can install missing Python/Poetry "
        "prerequisites with apt and pipx"
    ) in result.stdout
    assert (
        "CHECK_ONLY: compatible Ubuntu host can install Docker Engine from Docker's "
        "official apt repository"
    ) in result.stdout
    assert "Installing Ubuntu Python" not in result.stdout


def test_skip_prereq_install_preserves_report_only_behavior(tmp_path: Path) -> None:
    result = _run_check(
        os_release=_ubuntu_release(tmp_path, "24.04"),
        env={
            "OPENSHELL_TEST_UNAME_S": "Linux",
            "OPENSHELL_TEST_UNAME_M": "x86_64",
            "OPENSHELL_TEST_GLIBC_VERSION": "2.39",
            "OPENSHELL_TEST_NO_PYTHON": "1",
            "OPENSHELL_TEST_NO_POETRY": "1",
            "OPENSHELL_TEST_NO_DOCKER": "1",
        },
        args=[
            "bash",
            str(SCRIPT),
            "--skip-prereq-install",
            "--skip-system-install",
        ],
    )

    assert result.returncode == 1
    assert "Skipping Ubuntu Python/Poetry prerequisite install" in result.stdout
    assert "Skipping Ubuntu Docker prerequisite install" in result.stdout
    assert "Install missing Ubuntu Python/Poetry prerequisites" not in result.stdout


def test_macos_intel_is_rejected() -> None:
    result = _run_check(
        env={
            "OPENSHELL_TEST_UNAME_S": "Darwin",
            "OPENSHELL_TEST_UNAME_M": "x86_64",
            "OPENSHELL_TEST_MACOS_VERSION": "14.5",
        },
    )

    assert result.returncode == 1
    assert "UNSUPPORTED: Intel macOS is not supported" in result.stdout


def test_macos_12_arm64_is_rejected_for_sdk_wheel() -> None:
    result = _run_check(
        env={
            "OPENSHELL_TEST_UNAME_S": "Darwin",
            "OPENSHELL_TEST_UNAME_M": "arm64",
            "OPENSHELL_TEST_MACOS_VERSION": "12.7",
        },
    )

    assert result.returncode == 1
    assert "below OpenShell SDK wheel tag macosx_13_0_arm64" in result.stdout


def test_macos_arm64_missing_docker_offers_colima() -> None:
    result = _run_check(
        env={
            "OPENSHELL_TEST_UNAME_S": "Darwin",
            "OPENSHELL_TEST_UNAME_M": "arm64",
            "OPENSHELL_TEST_MACOS_VERSION": "14.5",
            "OPENSHELL_TEST_NO_DOCKER": "1",
        },
    )

    assert result.returncode == 1
    assert (
        "CHECK_ONLY: compatible macOS host can install/start Colima with Homebrew"
        in result.stdout
    )


def test_colima_socket_detection_prefers_openshell_profile() -> None:
    with tempfile.TemporaryDirectory(prefix="os-", dir="/tmp") as home:
        home_path = Path(home)
        openshell_socket = home_path / ".colima" / "openshell" / "docker.sock"
        default_socket = home_path / ".colima" / "default" / "docker.sock"
        openshell_socket.parent.mkdir(parents=True)
        default_socket.parent.mkdir(parents=True)

        sockets: list[socket.socket] = []
        for path in (openshell_socket, default_socket):
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.bind(str(path))
            sockets.append(sock)
        try:
            result = _run_check(
                env={
                    "HOME": str(home_path),
                    "OPENSHELL_TEST_UNAME_S": "Darwin",
                    "OPENSHELL_TEST_UNAME_M": "arm64",
                    "OPENSHELL_TEST_MACOS_VERSION": "14.5",
                    "OPENSHELL_TEST_NO_DOCKER": "1",
                },
            )
        finally:
            for sock in sockets:
                sock.close()

        assert result.returncode == 1
        assert f"Using Colima Docker socket at {openshell_socket}" in result.stdout


def test_docker_group_membership_rerun_guidance_is_present() -> None:
    text = SCRIPT.read_text()
    assert "newgrp docker" in text
    assert "refreshed group membership" in text


def test_docker_client_without_daemon_fails_validation(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker = fake_bin / "docker"
    docker.write_text(
        """#!/usr/bin/env bash
if [[ "$*" == *".Client.Version"* ]]; then
  echo "29.6.0"
  exit 0
fi
if [[ "$*" == *".Server.Version"* ]]; then
  exit 1
fi
echo "Docker version 29.6.0"
"""
    )
    docker.chmod(0o755)

    result = _run_check(
        os_release=_ubuntu_release(tmp_path, "24.04"),
        env={
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "OPENSHELL_TEST_UNAME_S": "Linux",
            "OPENSHELL_TEST_UNAME_M": "x86_64",
            "OPENSHELL_TEST_GLIBC_VERSION": "2.39",
        },
    )

    assert result.returncode == 1
    assert "Docker CLI 29.6.0 exists but daemon is not reachable" in result.stdout


def test_script_mentions_grpc_runtime_floor() -> None:
    text = SCRIPT.read_text()
    assert "MIN_GRPCIO_VERSION=\"1.78.0\"" in text
    assert "grpcio" in text
    assert "1.78.0" in text
