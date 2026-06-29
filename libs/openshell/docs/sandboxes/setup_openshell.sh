#!/usr/bin/env bash
set -euo pipefail

DEFAULT_OPENSHELL_VERSION="0.0.72"
MIN_PYTHON_VERSION="3.12"
MIN_GRPCIO_VERSION="1.78.0"
MIN_DOCKER_VERSION="28.0.4"
KERNEL_NAME="langchain-nvidia-openshell"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPORT_FILE="${SCRIPT_DIR}/openshell_setup_report.txt"

OPENSHELL_VERSION="$DEFAULT_OPENSHELL_VERSION"
CHECK_ONLY=0
SKIP_SYSTEM_INSTALL=0
ASSUME_YES=0
SYSTEM_NAME=""

usage() {
  cat <<EOF
Usage: ./setup_openshell.sh [options]

Set up the OpenShell environment used by nvidia_openshell_sandbox.ipynb.

Options:
  --openshell-version VERSION  OpenShell CLI and SDK version (default: ${DEFAULT_OPENSHELL_VERSION})
  --check-only                 Validate prerequisites without changing the host
  --skip-system-install        Keep the existing OpenShell CLI and gateway package
  --yes, -y                    Run installation without confirmation
  --help, -h                   Show this help
EOF
}

log() {
  printf '%s\n' "$*"
}

pass() {
  log "PASS: $*"
}

warn() {
  log "WARN: $*"
}

die() {
  log "FAIL: $*"
  log "RESULT: setup failed. See ${REPORT_FILE}"
  exit 1
}

have_cmd() {
  case "$1" in
    docker) [ "${OPENSHELL_TEST_NO_DOCKER:-0}" != "1" ] || return 1 ;;
    poetry) [ "${OPENSHELL_TEST_NO_POETRY:-0}" != "1" ] || return 1 ;;
    python3) [ "${OPENSHELL_TEST_NO_PYTHON:-0}" != "1" ] || return 1 ;;
  esac
  command -v "$1" >/dev/null 2>&1
}

normalize_version() {
  printf '%s\n' "${1#v}" | sed 's/[^0-9.].*$//'
}

version_ge() {
  local version minimum
  version="$(normalize_version "$1")"
  minimum="$(normalize_version "$2")"
  awk -v version="$version" -v minimum="$minimum" '
    BEGIN {
      split(version, v, ".")
      split(minimum, m, ".")
      for (i = 1; i <= 4; i++) {
        vi = (v[i] == "" ? 0 : v[i]) + 0
        mi = (m[i] == "" ? 0 : m[i]) + 0
        if (vi > mi) exit 0
        if (vi < mi) exit 1
      }
      exit 0
    }'
}

parse_args() {
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --openshell-version)
        [ "$#" -ge 2 ] || die "--openshell-version requires a value"
        OPENSHELL_VERSION="$2"
        shift 2
        ;;
      --check-only)
        CHECK_ONLY=1
        shift
        ;;
      --skip-system-install)
        SKIP_SYSTEM_INSTALL=1
        shift
        ;;
      --yes|-y)
        ASSUME_YES=1
        shift
        ;;
      --help|-h)
        usage
        exit 0
        ;;
      *)
        usage >&2
        die "unknown option: $1"
        ;;
    esac
  done

  [[ "$OPENSHELL_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] \
    || die "--openshell-version must be a stable numeric release such as 0.0.72"
  version_ge "$OPENSHELL_VERSION" "$DEFAULT_OPENSHELL_VERSION" \
    || die "OpenShell ${OPENSHELL_VERSION} predates the fixed Linux wheels in ${DEFAULT_OPENSHELL_VERSION}"
}

detect_glibc() {
  if [ -n "${OPENSHELL_TEST_GLIBC_VERSION:-}" ]; then
    printf '%s\n' "$OPENSHELL_TEST_GLIBC_VERSION"
  elif have_cmd getconf; then
    getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $NF}'
  fi
}

validate_platform() {
  local arch os_release distro version glibc
  SYSTEM_NAME="${OPENSHELL_TEST_UNAME_S:-$(uname -s)}"
  arch="${OPENSHELL_TEST_UNAME_M:-$(uname -m)}"

  case "$SYSTEM_NAME" in
    Linux)
      os_release="${OPENSHELL_TEST_OS_RELEASE:-/etc/os-release}"
      [ -r "$os_release" ] || die "cannot read ${os_release}"
      # shellcheck disable=SC1090
      . "$os_release"
      distro="${ID:-}"
      version="${VERSION_ID:-}"
      [ "$distro" = "ubuntu" ] || die "this notebook setup supports Ubuntu 24.04+; detected ${distro:-unknown}"
      version_ge "$version" "24.04" || die "Ubuntu ${version:-unknown} is below the notebook SDK floor of 24.04"
      case "$arch" in
        x86_64|amd64|aarch64|arm64) ;;
        *) die "unsupported Linux architecture: ${arch}" ;;
      esac
      glibc="$(detect_glibc)"
      [ -n "$glibc" ] || die "could not detect glibc"
      version_ge "$glibc" "2.39" || die "glibc ${glibc} cannot install OpenShell's manylinux_2_39 SDK wheel"
      pass "Ubuntu ${version} ${arch} with glibc ${glibc}"
      ;;
    Darwin)
      version="${OPENSHELL_TEST_MACOS_VERSION:-$(sw_vers -productVersion 2>/dev/null || true)}"
      [ "$arch" = "arm64" ] || die "OpenShell supports Apple Silicon macOS; detected ${arch}"
      version_ge "$version" "13" || die "macOS ${version:-unknown} is below the SDK wheel floor of 13"
      pass "macOS ${version} ${arch}"
      ;;
    *)
      die "unsupported operating system: ${SYSTEM_NAME}"
      ;;
  esac
}

require_cmd() {
  have_cmd "$1" || die "$1 is required; install it and rerun this script"
}

select_docker_host() {
  local context_host socket

  context_host="$(docker context inspect --format '{{.Endpoints.docker.Host}}' 2>/dev/null || true)"
  case "$context_host" in
    unix://*) export DOCKER_HOST="$context_host" ;;
  esac
  if docker version --format '{{.Server.Version}}' >/dev/null 2>&1; then
    return 0
  fi

  unset DOCKER_HOST
  for socket in \
    "${HOME}/.colima/openshell/docker.sock" \
    "${HOME}/.colima/default/docker.sock" \
    "${HOME}/.docker/run/docker.sock" \
    "/var/run/docker.sock"; do
    [ -S "$socket" ] || continue
    export DOCKER_HOST="unix://${socket}"
    if docker version --format '{{.Server.Version}}' >/dev/null 2>&1; then
      pass "Using Docker socket ${socket}"
      return 0
    fi
  done
  unset DOCKER_HOST
}

validate_prerequisites() {
  local python_version docker_version
  require_cmd curl
  require_cmd python3
  require_cmd poetry
  require_cmd docker

  python_version="$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
  version_ge "$python_version" "$MIN_PYTHON_VERSION" || die "Python ${python_version} is below ${MIN_PYTHON_VERSION}"
  pass "Python ${python_version} and Poetry are available"

  select_docker_host
  docker_version="$(docker version --format '{{.Server.Version}}' 2>/dev/null || true)"
  if [ -z "$docker_version" ]; then
    if have_cmd sudo && sudo -n docker version --format '{{.Server.Version}}' >/dev/null 2>&1; then
      die "Docker works only through sudo. Log out and back in after joining the docker group, or run 'newgrp docker', then rerun this script"
    fi
    die "Docker is installed but its daemon is not reachable by ${USER:-the current user}"
  fi
  version_ge "$docker_version" "$MIN_DOCKER_VERSION" || die "Docker ${docker_version} is below ${MIN_DOCKER_VERSION}"
  pass "Docker ${docker_version} is reachable without sudo"
}

configure_gateway_environment() {
  [ -n "${DOCKER_HOST:-}" ] || return 0
  local config_dir env_file
  config_dir="${XDG_CONFIG_HOME:-${HOME}/.config}/openshell"
  env_file="${config_dir}/gateway.env"
  mkdir -p "$config_dir"
  {
    printf '%s\n' "# Managed by langchain-nvidia OpenShell setup"
    printf '%s\n' "OPENSHELL_DRIVERS=docker"
    printf 'DOCKER_HOST=%q\n' "$DOCKER_HOST"
  } > "$env_file"
  pass "OpenShell gateway Docker environment written to ${env_file}"
}

user_systemd_available() {
  [ "$SYSTEM_NAME" = "Linux" ] || return 0
  have_cmd systemctl || return 1
  systemctl --user show-environment >/dev/null 2>&1
}

prepare_linux_user_systemd() {
  [ "$SYSTEM_NAME" = "Linux" ] || return 0

  local uid user runtime_dir
  uid="$(id -u)"
  user="$(id -un)"
  runtime_dir="${XDG_RUNTIME_DIR:-/run/user/${uid}}"
  export XDG_RUNTIME_DIR="$runtime_dir"
  export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=${runtime_dir}/bus}"

  if user_systemd_available; then
    pass "systemd user manager is reachable"
    return 0
  fi
  if [ "$CHECK_ONLY" -eq 1 ]; then
    warn "systemd user manager is not running; setup will enable linger and start user@${uid}.service"
    return 0
  fi

  require_cmd sudo
  require_cmd loginctl
  require_cmd systemctl
  log "Starting the systemd user manager required by the OpenShell gateway"
  sudo loginctl enable-linger "$user"
  sudo systemctl start "user@${uid}.service"

  for _ in 1 2 3 4 5; do
    if user_systemd_available; then
      pass "systemd user manager is reachable with linger enabled"
      return 0
    fi
    sleep 1
  done
  die "could not start the systemd user manager at ${DBUS_SESSION_BUS_ADDRESS}"
}

confirm_install() {
  [ "$ASSUME_YES" -eq 1 ] && return 0
  [ -t 0 ] || die "installation requires confirmation; rerun with --yes"
  local answer
  printf 'Install OpenShell %s and configure the notebook environment? [y/N] ' "$OPENSHELL_VERSION"
  read -r answer
  case "$answer" in
    y|Y|yes|YES) ;;
    *) die "installation cancelled" ;;
  esac
}

install_openshell_system() {
  [ "$SKIP_SYSTEM_INSTALL" -eq 0 ] || {
    log "Skipping the OpenShell system package due to --skip-system-install"
    return 0
  }

  log "Installing OpenShell v${OPENSHELL_VERSION} with NVIDIA's installer"
  if curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh \
    | OPENSHELL_VERSION="v${OPENSHELL_VERSION}" sh; then
    pass "NVIDIA OpenShell installer completed"
  else
    die "NVIDIA OpenShell installer failed for v${OPENSHELL_VERSION}"
  fi
}

configure_python_environment() {
  local env_path python_bin
  log "Installing the Poetry environment and Jupyter kernel"
  (
    cd "$PROJECT_DIR"
    poetry config virtualenvs.in-project true --local
    poetry install --with test
  )
  env_path="$(cd "$PROJECT_DIR" && poetry env info --path)"
  python_bin="${env_path}/bin/python"
  [ -x "$python_bin" ] || die "Poetry environment has no Python executable at ${python_bin}"

  "$python_bin" -m pip --version >/dev/null 2>&1 || "$python_bin" -m ensurepip --upgrade
  "$python_bin" -m pip install --disable-pip-version-check --quiet \
    "openshell==${OPENSHELL_VERSION}" \
    "grpcio>=${MIN_GRPCIO_VERSION},<2" \
    ipykernel
  "$python_bin" -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_NAME"
  pass "Poetry environment and ${KERNEL_NAME} kernel are installed"
}

dump_gateway_diagnostics() {
  [ "$SYSTEM_NAME" = "Linux" ] || return 0
  log "OpenShell gateway diagnostics:"
  systemctl --user status openshell-gateway --no-pager 2>&1 || true
  journalctl --user -u openshell-gateway --no-pager -n 80 2>&1 || true
}

validate_installation() {
  local env_path python_bin cli cli_version
  env_path="$(cd "$PROJECT_DIR" && poetry env info --path)"
  python_bin="${env_path}/bin/python"
  cli="${env_path}/bin/openshell"
  [ -x "$cli" ] || die "OpenShell CLI is missing from ${env_path}/bin"

  "$python_bin" - <<'PY'
import warnings

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

warnings.filterwarnings(
    "ignore",
    message="The default value of `allowed_objects` will change.*",
    category=LangChainPendingDeprecationWarning,
)

import grpc
import google.protobuf
import openshell
import openshell.sandbox
import langchain_nvidia_openshell

parts = tuple(int(part) for part in grpc.__version__.split(".")[:2])
assert parts >= (1, 78), grpc.__version__
print(f"OpenShell Python imports OK; grpcio {grpc.__version__}")
PY
  pass "OpenShell SDK and langchain_nvidia_openshell imports are healthy"

  cli_version="$("$cli" --version 2>&1)"
  case "$cli_version" in
    *"${OPENSHELL_VERSION}"*) pass "OpenShell CLI version matches ${OPENSHELL_VERSION}" ;;
    *) die "OpenShell CLI version mismatch: expected ${OPENSHELL_VERSION}, got ${cli_version:-unknown}" ;;
  esac
  if ! "$cli" status; then
    dump_gateway_diagnostics
    die "OpenShell CLI cannot reach the gateway"
  fi
  if ! "$cli" sandbox list >/dev/null; then
    dump_gateway_diagnostics
    die "OpenShell gateway cannot list sandboxes"
  fi
  pass "OpenShell gateway is connected and the sandbox API is reachable; existing sandbox states are unchanged"
}

main() {
  parse_args "$@"

  log "OpenShell notebook setup"
  log "Target version: ${OPENSHELL_VERSION}"
  log "Report: ${REPORT_FILE}"

  [ "$(id -u)" -ne 0 ] || die "run this script as your normal user, not with sudo"
  validate_platform
  validate_prerequisites

  if [ "$CHECK_ONLY" -eq 1 ]; then
    prepare_linux_user_systemd
    log "SUCCESS: prerequisites are ready; no changes were made."
    return 0
  fi

  confirm_install
  prepare_linux_user_systemd
  configure_gateway_environment
  install_openshell_system
  configure_python_environment
  validate_installation
  log "SUCCESS: OpenShell notebook environment is ready."
}

if [ "${OPENSHELL_SETUP_TESTING:-0}" != "1" ]; then
  : > "$REPORT_FILE"
  main "$@" 2>&1 | tee -a "$REPORT_FILE"
fi
