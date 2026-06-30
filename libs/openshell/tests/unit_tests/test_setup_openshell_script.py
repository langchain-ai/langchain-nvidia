"""Tests for the minimal OpenShell notebook setup script."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2] / "docs" / "sandboxes" / "setup_openshell.sh"
)


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def _ubuntu_release(tmp_path: Path, version: str) -> Path:
    path = tmp_path / "os-release"
    path.write_text(
        "\n".join(
            [
                "ID=ubuntu",
                f'VERSION_ID="{version}"',
                f'PRETTY_NAME="Ubuntu {version}"',
            ]
        )
        + "\n"
    )
    return path


def _fake_tools(tmp_path: Path, *, docker_version: str | None = "29.6.0") -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker_output = f'echo "{docker_version}"' if docker_version else "exit 1"
    _write_executable(
        bin_dir / "docker",
        f"""#!/usr/bin/env bash
if [[ "$*" == *".Server.Version"* ]]; then
  {docker_output}
  exit $?
fi
exit 0
""",
    )
    _write_executable(
        bin_dir / "poetry",
        """#!/usr/bin/env bash
echo "Poetry (version 2.1.3)"
""",
    )
    return bin_dir


def _run_check(
    tmp_path: Path,
    *,
    ubuntu_version: str = "24.04",
    openshell_version: str = "0.0.72",
    docker_version: str | None = "29.6.0",
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    bin_dir = _fake_tools(tmp_path, docker_version=docker_version)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "OPENSHELL_TEST_UNAME_S": "Linux",
            "OPENSHELL_TEST_UNAME_M": "x86_64",
            "OPENSHELL_TEST_GLIBC_VERSION": "2.39",
            "OPENSHELL_TEST_OS_RELEASE": str(_ubuntu_release(tmp_path, ubuntu_version)),
        }
    )
    env.update(extra_env or {})
    return subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--check-only",
            "--openshell-version",
            openshell_version,
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=False,
    )


def test_help_describes_the_small_public_interface() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--openshell-version VERSION" in result.stdout
    assert "--check-only" in result.stdout
    assert "--skip-system-install" in result.stdout
    assert "--driver" not in result.stdout
    assert "--skip-prereq-install" not in result.stdout


def test_version_comparison_handles_patch_versions() -> None:
    command = f"""
OPENSHELL_SETUP_TESTING=1 source {SCRIPT!s}
version_ge 24.04.4 24.04
! version_ge 2.38 2.39
version_ge 0.0.72 0.0.72
"""
    result = subprocess.run(
        ["bash", "-c", command], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr


def test_ubuntu_22_is_rejected_at_the_sdk_wheel_boundary(tmp_path: Path) -> None:
    result = _run_check(tmp_path, ubuntu_version="22.04")

    assert result.returncode == 1
    assert "Ubuntu 22.04 is below the notebook SDK floor of 24.04" in result.stdout


def test_ubuntu_24_prerequisites_pass_without_mutation(tmp_path: Path) -> None:
    marker = tmp_path / "sudo-called"
    bin_dir = _fake_tools(tmp_path)
    _write_executable(
        bin_dir / "sudo",
        f"""#!/usr/bin/env bash
touch {marker}
exit 1
""",
    )
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "OPENSHELL_TEST_UNAME_S": "Linux",
            "OPENSHELL_TEST_UNAME_M": "x86_64",
            "OPENSHELL_TEST_GLIBC_VERSION": "2.39",
            "OPENSHELL_TEST_OS_RELEASE": str(_ubuntu_release(tmp_path, "24.04.4")),
        }
    )

    result = subprocess.run(
        ["bash", str(SCRIPT), "--check-only"],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: Ubuntu 24.04.4 x86_64 with glibc 2.39" in result.stdout
    assert "PASS: Docker 29.6.0 is reachable without sudo" in result.stdout
    assert "SUCCESS: prerequisites are ready; no changes were made" in result.stdout
    assert not marker.exists()


def test_release_before_pr_2029_fix_is_rejected(
    tmp_path: Path,
) -> None:
    result = _run_check(tmp_path, openshell_version="0.0.71")

    assert result.returncode == 1
    assert "predates the fixed Linux wheels in 0.0.72" in result.stdout


def test_first_fixed_release_reaches_prerequisite_checks(tmp_path: Path) -> None:
    result = _run_check(tmp_path, openshell_version="0.0.72")

    assert result.returncode == 0


def test_missing_docker_fails_with_a_prerequisite_message(tmp_path: Path) -> None:
    result = _run_check(
        tmp_path,
        extra_env={"OPENSHELL_TEST_NO_DOCKER": "1"},
    )

    assert result.returncode == 1
    assert "docker is required" in result.stdout


def test_inaccessible_docker_reports_group_membership_when_sudo_works(
    tmp_path: Path,
) -> None:
    bin_dir = _fake_tools(tmp_path, docker_version=None)
    _write_executable(
        bin_dir / "sudo",
        """#!/usr/bin/env bash
if [[ "$*" == *"docker version"* ]]; then
  echo "29.6.0"
  exit 0
fi
exit 1
""",
    )
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "OPENSHELL_TEST_UNAME_S": "Linux",
            "OPENSHELL_TEST_UNAME_M": "x86_64",
            "OPENSHELL_TEST_GLIBC_VERSION": "2.39",
            "OPENSHELL_TEST_OS_RELEASE": str(_ubuntu_release(tmp_path, "24.04")),
        }
    )

    result = subprocess.run(
        ["bash", str(SCRIPT), "--check-only"],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=False,
    )

    assert result.returncode == 1
    assert "Docker works only through sudo" in result.stdout
    assert "Log out and back in" in result.stdout


def test_macos_intel_is_rejected() -> None:
    env = os.environ.copy()
    env.update(
        {
            "OPENSHELL_TEST_UNAME_S": "Darwin",
            "OPENSHELL_TEST_UNAME_M": "x86_64",
            "OPENSHELL_TEST_MACOS_VERSION": "14.5",
        }
    )
    result = subprocess.run(
        ["bash", str(SCRIPT), "--check-only"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 1
    assert "supports Apple Silicon macOS" in result.stdout


def test_script_delegates_gateway_ownership_to_upstream() -> None:
    text = SCRIPT.read_text()

    assert "raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh" in text
    assert "loginctl enable-linger" in text
    assert "systemctl --user show-environment" in text
    assert "gateway add" not in text
    assert "nohup openshell-gateway" not in text
    assert "apt-get" not in text
    assert "docker.sources" not in text


def test_install_confirmation_precedes_user_systemd_mutation() -> None:
    text = SCRIPT.read_text()

    assert 'MIN_DOCKER_VERSION="28.0.4"' in text
    assert (
        "confirm_install\n  prepare_linux_user_systemd\n  "
        "configure_gateway_environment\n  install_openshell_system" in text
    )


def test_legacy_wheel_repair_has_been_removed() -> None:
    text = SCRIPT.read_text()

    assert 'DEFAULT_OPENSHELL_VERSION="0.0.72"' in text
    assert "repair_legacy_linux_wheel" not in text
    assert "macosx_13_0_arm64" not in text
    assert "zipfile" not in text


def test_named_colima_socket_is_passed_to_the_gateway_service() -> None:
    text = SCRIPT.read_text()

    assert '"${HOME}/.colima/openshell/docker.sock"' in text
    assert "select_docker_host" in text
    assert "configure_gateway_environment" in text
    assert "OPENSHELL_DRIVERS=docker" in text
    assert "DOCKER_HOST=%q" in text


def test_setup_preserves_locked_versions_and_hides_stale_sandboxes() -> None:
    text = SCRIPT.read_text()

    assert '"$python_bin" -m pip install --upgrade' not in text
    assert '"$python_bin" -m pip install --disable-pip-version-check --quiet' in text
    assert '"$cli" sandbox list >/dev/null' in text
    assert "existing sandbox states are unchanged" in text


def test_shell_script_has_valid_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
