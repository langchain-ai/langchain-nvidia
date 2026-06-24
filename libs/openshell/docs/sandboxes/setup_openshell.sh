#!/usr/bin/env bash
set -euo pipefail

DEFAULT_OPENSHELL_VERSION="0.0.68"
MIN_PYTHON_VERSION="3.12"
MIN_GRPCIO_VERSION="1.78.0"
MIN_DOCKER_VERSION="28.04"
MIN_GLIBC_GATEWAY_VERSION="2.28"
MIN_GLIBC_SDK_VERSION="2.39"
MIN_MACOS_SDK_MAJOR="13"
KERNEL_NAME="langchain-nvidia-openshell"
DISPLAY_NAME="langchain-nvidia-openshell"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPEN_SHELL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPORT_FILE="${SCRIPT_DIR}/openshell_setup_report.txt"

OPENSHELL_VERSION="${DEFAULT_OPENSHELL_VERSION}"
CHECK_ONLY=0
SKIP_SYSTEM_INSTALL=0
SKIP_PREREQ_INSTALL=0
DRIVER="docker"
ASSUME_YES=0
VERBOSE=0
FAILED=0
WARNED=0
UNSUPPORTED=0
SYSTEM_NAME=""
ARCH_NAME=""
DISTRO_ID=""
DISTRO_VERSION_ID=""
OS_PRETTY=""
GLIBC_VERSION=""
MACOS_VERSION=""
PYTHON_BIN=""
OPENSHELL_BIN=""
GATEWAY_DRIVER=""
WSL_ENVIRONMENT=0
PREREQ_RERUN_REQUIRED=0

usage() {
  cat <<EOF
Usage: ./setup_openshell.sh [options]

Set up and validate the OpenShell environment for nvidia_openshell_sandbox.ipynb.

Options:
  --openshell-version VERSION   OpenShell version to install and validate (default: ${DEFAULT_OPENSHELL_VERSION})
  --check-only                  Validate the host and current tools, but do not install anything
  --skip-prereq-install         Do not install Ubuntu/macOS prerequisites
  --skip-system-install         Do not run NVIDIA's OpenShell installer
  --driver DRIVER               docker, podman, kubernetes, vm, or auto (default: docker)
  --yes                         Do not prompt before system install
  --verbose                     Print commands as they run
  --help                        Show this help
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --openshell-version)
      [ "$#" -ge 2 ] || { echo "--openshell-version requires a value" >&2; exit 2; }
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
    --skip-prereq-install)
      SKIP_PREREQ_INSTALL=1
      shift
      ;;
    --driver)
      [ "$#" -ge 2 ] || { echo "--driver requires a value" >&2; exit 2; }
      DRIVER="$2"
      shift 2
      ;;
    --yes|-y)
      ASSUME_YES=1
      shift
      ;;
    --verbose)
      VERBOSE=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$DRIVER" in
  docker|podman|kubernetes|vm|auto) ;;
  *)
    echo "--driver must be docker, podman, kubernetes, vm, or auto" >&2
    exit 2
    ;;
esac

if [ "$VERBOSE" -eq 1 ]; then
  set -x
fi

: > "$REPORT_FILE"

report() {
  printf '%s\n' "$*" | tee -a "$REPORT_FILE"
}

pass() {
  report "PASS: $*"
}

warn() {
  WARNED=1
  report "WARN: $*"
}

fail() {
  FAILED=1
  report "FAIL: $*"
}

unsupported() {
  UNSUPPORTED=1
  FAILED=1
  report "UNSUPPORTED: $*"
}

have_cmd() {
  case "$1" in
    python|python3)
      [ "${OPENSHELL_TEST_NO_PYTHON:-0}" = "1" ] && return 1
      ;;
    poetry)
      [ "${OPENSHELL_TEST_NO_POETRY:-0}" = "1" ] && return 1
      ;;
    docker)
      [ "${OPENSHELL_TEST_NO_DOCKER:-0}" = "1" ] && return 1
      ;;
    brew)
      [ "${OPENSHELL_TEST_NO_BREW:-0}" = "1" ] && return 1
      ;;
    colima)
      [ "${OPENSHELL_TEST_NO_COLIMA:-0}" = "1" ] && return 1
      ;;
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
        if (vi > mi) { exit 0 }
        if (vi < mi) { exit 1 }
      }
      exit 0
    }'
}

version_major() {
  local version
  version="$(normalize_version "$1")"
  printf '%s\n' "${version%%.*}"
}

detect_os() {
  SYSTEM_NAME="${OPENSHELL_TEST_UNAME_S:-$(uname -s)}"
  ARCH_NAME="${OPENSHELL_TEST_UNAME_M:-$(uname -m)}"
  OS_PRETTY="$SYSTEM_NAME"

  if [ "$SYSTEM_NAME" = "Linux" ]; then
    local os_release
    os_release="${OPENSHELL_TEST_OS_RELEASE:-/etc/os-release}"
    if [ -f "$os_release" ]; then
      # shellcheck disable=SC1090
      . "$os_release"
      DISTRO_ID="${ID:-}"
      DISTRO_VERSION_ID="${VERSION_ID:-}"
      OS_PRETTY="${PRETTY_NAME:-${DISTRO_ID} ${DISTRO_VERSION_ID}}"
    fi
    if [ -n "${OPENSHELL_TEST_GLIBC_VERSION:-}" ]; then
      GLIBC_VERSION="$OPENSHELL_TEST_GLIBC_VERSION"
    elif have_cmd getconf && getconf GNU_LIBC_VERSION >/dev/null 2>&1; then
      GLIBC_VERSION="$(getconf GNU_LIBC_VERSION | awk '{print $NF}')"
    elif have_cmd ldd; then
      GLIBC_VERSION="$(ldd --version 2>&1 | awk 'NR == 1 { for (i = 1; i <= NF; i++) if ($i ~ /^[0-9]+\.[0-9]+/) { print $i; exit } }')"
    fi
    if [ "${OPENSHELL_TEST_WSL:-0}" = "1" ]; then
      WSL_ENVIRONMENT=1
    elif [ -r /proc/sys/kernel/osrelease ] && grep -qi microsoft /proc/sys/kernel/osrelease; then
      WSL_ENVIRONMENT=1
    fi
  elif [ "$SYSTEM_NAME" = "Darwin" ]; then
    MACOS_VERSION="${OPENSHELL_TEST_MACOS_VERSION:-$(sw_vers -productVersion 2>/dev/null || true)}"
    OS_PRETTY="macOS ${MACOS_VERSION}"
  fi
}

validate_platform() {
  report "Detected OS: ${OS_PRETTY}"
  report "Detected arch: ${ARCH_NAME}"

  case "$SYSTEM_NAME" in
    Linux)
      if [ "$WSL_ENVIRONMENT" -eq 1 ]; then
        warn "WSL2 is experimental upstream and is not automated by this notebook setup"
      fi
      case "$ARCH_NAME" in
        x86_64|amd64|aarch64|arm64) pass "Linux architecture ${ARCH_NAME} is supported by OpenShell release assets" ;;
        *) unsupported "Linux architecture ${ARCH_NAME} is not supported; expected amd64/x86_64 or arm64/aarch64" ;;
      esac

      if [ -z "$GLIBC_VERSION" ]; then
        unsupported "Could not detect glibc; Alpine/musl and unknown libc environments are unsupported for the gateway package"
      elif version_ge "$GLIBC_VERSION" "$MIN_GLIBC_GATEWAY_VERSION"; then
        pass "glibc ${GLIBC_VERSION} satisfies gateway floor ${MIN_GLIBC_GATEWAY_VERSION}"
      else
        unsupported "glibc ${GLIBC_VERSION} is below gateway floor ${MIN_GLIBC_GATEWAY_VERSION}"
      fi

      if [ "$DISTRO_ID" = "ubuntu" ]; then
        local ubuntu_major
        ubuntu_major="$(version_major "$DISTRO_VERSION_ID")"
        if [ "$ubuntu_major" -lt 20 ]; then
          unsupported "Ubuntu ${DISTRO_VERSION_ID} is unsupported for this notebook setup"
        elif [ "$ubuntu_major" -lt 24 ]; then
          warn "Ubuntu ${DISTRO_VERSION_ID} can satisfy gateway docs, but current OpenShell SDK wheels require glibc ${MIN_GLIBC_SDK_VERSION}+ / manylinux_2_39"
        else
          pass "Ubuntu ${DISTRO_VERSION_ID} is in the expected notebook-compatible family when glibc and pip checks pass"
        fi
      fi

      if [ -n "$GLIBC_VERSION" ] && version_ge "$GLIBC_VERSION" "$MIN_GLIBC_SDK_VERSION"; then
        pass "glibc ${GLIBC_VERSION} satisfies OpenShell SDK wheel tag floor ${MIN_GLIBC_SDK_VERSION}"
      else
        warn "Current OpenShell SDK wheels are manylinux_2_39; this host may be gateway-compatible but notebook-SDK-incompatible"
      fi
      ;;
    Darwin)
      if [ "$ARCH_NAME" != "arm64" ]; then
        unsupported "Intel macOS is not supported by OpenShell release assets for this notebook"
      fi
      if [ -z "$MACOS_VERSION" ]; then
        unsupported "Could not detect macOS version"
      else
        local mac_major
        mac_major="$(version_major "$MACOS_VERSION")"
        if [ "$mac_major" -lt "$MIN_MACOS_SDK_MAJOR" ]; then
          unsupported "macOS ${MACOS_VERSION} is below OpenShell SDK wheel tag macosx_13_0_arm64"
        else
          pass "macOS ${MACOS_VERSION} arm64 satisfies OpenShell SDK wheel tag macosx_13_0_arm64"
        fi
      fi
      ;;
    MINGW*|MSYS*|CYGWIN*)
      unsupported "Windows is not automated by this notebook setup; WSL2 + Docker Desktop is experimental upstream"
      ;;
    *)
      unsupported "Unsupported OS: ${SYSTEM_NAME}"
      ;;
  esac
}

find_python() {
  if [ -n "${OPENSHELL_PYTHON:-}" ]; then
    PYTHON_BIN="$OPENSHELL_PYTHON"
  elif have_cmd python3; then
    PYTHON_BIN="$(command -v python3)"
  elif have_cmd python; then
    PYTHON_BIN="$(command -v python)"
  else
    fail "Python ${MIN_PYTHON_VERSION}+ is required"
    return
  fi

  local py_version
  py_version="$("$PYTHON_BIN" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
  if version_ge "$py_version" "$MIN_PYTHON_VERSION"; then
    pass "Python ${py_version} at ${PYTHON_BIN}"
  else
    fail "Python ${py_version} is below required ${MIN_PYTHON_VERSION}"
  fi

  if "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
    pass "pip is available for ${PYTHON_BIN}"
  else
    fail "pip is not available for ${PYTHON_BIN}"
  fi
}

pip_platform_args() {
  local arch platform
  arch="$ARCH_NAME"
  case "$SYSTEM_NAME:$arch" in
    Darwin:arm64)
      platform="macosx_13_0_arm64"
      ;;
    Linux:x86_64|Linux:amd64)
      platform="manylinux_2_39_x86_64"
      ;;
    Linux:aarch64|Linux:arm64)
      platform="manylinux_2_39_aarch64"
      ;;
    *)
      platform=""
      ;;
  esac
  printf '%s\n' "$platform"
}

validate_openshell_wheel_resolution() {
  [ -n "$PYTHON_BIN" ] || return 0
  local platform tmpdir log_file
  platform="$(pip_platform_args)"
  if [ -z "$platform" ]; then
    warn "Skipping OpenShell wheel resolution; no known wheel platform for ${SYSTEM_NAME}/${ARCH_NAME}"
    return
  fi

  if [ "$SYSTEM_NAME" = "Linux" ]; then
    if [ -z "$GLIBC_VERSION" ] || ! version_ge "$GLIBC_VERSION" "$MIN_GLIBC_SDK_VERSION"; then
      fail "openshell==${OPENSHELL_VERSION} SDK wheels are manylinux_2_39; detected glibc ${GLIBC_VERSION:-unknown}"
      return
    fi
  fi

  tmpdir="$(mktemp -d)"
  log_file="${tmpdir}/pip-download.log"
  if "$PYTHON_BIN" -m pip download \
    --disable-pip-version-check \
    --no-deps \
    --only-binary=:all: \
    --python-version 312 \
    --implementation py \
    --abi none \
    --platform "$platform" \
    "openshell==${OPENSHELL_VERSION}" \
    -d "$tmpdir" >"$log_file" 2>&1; then
    pass "pip can resolve openshell==${OPENSHELL_VERSION} for ${platform}"
  else
    fail "pip cannot resolve openshell==${OPENSHELL_VERSION} for ${platform}; see ${log_file}"
    sed 's/^/  /' "$log_file" | tee -a "$REPORT_FILE"
  fi
  rm -rf "$tmpdir"
}

validate_poetry_and_jupyter() {
  if have_cmd poetry; then
    pass "Poetry found: $(command -v poetry)"
  else
    fail "Poetry is required; install it before running notebook setup"
  fi

  if have_cmd jupyter; then
    pass "Jupyter found: $(command -v jupyter)"
  else
    warn "jupyter command not found on PATH; kernel registration can still work, but kernel listing may not"
  fi
}

extract_openshell_version() {
  awk '{ for (i = 1; i <= NF; i++) if ($i ~ /^v?[0-9]+\.[0-9]+\.[0-9]+/) { gsub(/^v/, "", $i); print $i; exit } }'
}

confirm_action() {
  local prompt answer
  prompt="$1"
  if [ "$ASSUME_YES" -eq 1 ]; then
    return 0
  fi
  if [ ! -t 0 ]; then
    fail "${prompt} requires confirmation; rerun with --yes to allow this install"
    return 1
  fi
  printf '%s [y/N] ' "$prompt"
  read -r answer
  case "$answer" in
    y|Y|yes|YES) return 0 ;;
    *) fail "User declined: ${prompt}"; return 1 ;;
  esac
}

run_sudo() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  elif have_cmd sudo; then
    sudo "$@"
  else
    fail "sudo is required to install system prerequisites"
    return 1
  fi
}

ubuntu_major_version() {
  version_major "$DISTRO_VERSION_ID"
}

ubuntu_prereq_install_supported() {
  [ "$SYSTEM_NAME" = "Linux" ] || return 1
  [ "$DISTRO_ID" = "ubuntu" ] || return 1
  [ "$WSL_ENVIRONMENT" -eq 0 ] || return 1
  case "$ARCH_NAME" in
    x86_64|amd64|aarch64|arm64) ;;
    *) return 1 ;;
  esac
  [ -n "$GLIBC_VERSION" ] && version_ge "$GLIBC_VERSION" "$MIN_GLIBC_SDK_VERSION" || return 1
  [ "$(ubuntu_major_version)" -ge 24 ] || return 1
}

macos_colima_install_supported() {
  [ "$SYSTEM_NAME" = "Darwin" ] || return 1
  [ "$ARCH_NAME" = "arm64" ] || return 1
  [ -n "$MACOS_VERSION" ] || return 1
  [ "$(version_major "$MACOS_VERSION")" -ge "$MIN_MACOS_SDK_MAJOR" ] || return 1
}

python_prereqs_healthy() {
  local candidate version
  candidate=""
  if have_cmd python3; then
    candidate="$(command -v python3)"
  elif have_cmd python; then
    candidate="$(command -v python)"
  else
    return 1
  fi
  version="$("$candidate" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null || true)"
  [ -n "$version" ] && version_ge "$version" "$MIN_PYTHON_VERSION" || return 1
  "$candidate" -m pip --version >/dev/null 2>&1 || return 1
  local ok
  if "$candidate" -m venv /tmp/openshell-venv-check.$$ >/tmp/openshell-venv-check.txt 2>&1; then
    ok=0
  else
    ok=1
  fi
  rm -rf /tmp/openshell-venv-check.$$ /tmp/openshell-venv-check.txt
  return "$ok"
}

docker_daemon_accessible() {
  have_cmd docker || return 1
  docker version --format '{{.Server.Version}}' >/dev/null 2>&1
}

docker_server_version() {
  have_cmd docker || return 0
  docker version --format '{{.Server.Version}}' 2>/dev/null || true
}

docker_client_version() {
  have_cmd docker || return 0
  docker version --format '{{.Client.Version}}' 2>/dev/null || true
}

docker_needs_linux_group_refresh() {
  [ "$SYSTEM_NAME" = "Linux" ] || return 1
  [ "$(id -u)" -ne 0 ] || return 1
  have_cmd docker || return 1
  docker version --format '{{.Server.Version}}' >/dev/null 2>&1 && return 1
  have_cmd sudo || return 1
  sudo docker version --format '{{.Server.Version}}' >/dev/null 2>&1
}

install_ubuntu_python_poetry_prereqs() {
  if python_prereqs_healthy && have_cmd poetry; then
    return 0
  fi

  if [ "$CHECK_ONLY" -eq 1 ]; then
    report "CHECK_ONLY: compatible Ubuntu host can install missing Python/Poetry prerequisites with apt and pipx"
    return 0
  fi
  if [ "$SKIP_PREREQ_INSTALL" -eq 1 ]; then
    report "Skipping Ubuntu Python/Poetry prerequisite install due to --skip-prereq-install"
    return 0
  fi
  confirm_action "Install missing Ubuntu Python/Poetry prerequisites?" || return

  report "Installing Ubuntu Python and Poetry prerequisites"
  export DEBIAN_FRONTEND="${DEBIAN_FRONTEND:-noninteractive}"
  run_sudo apt-get update
  run_sudo apt-get install -y ca-certificates curl python3-full python3-pip pipx
  export PATH="${HOME}/.local/bin:${PATH}"
  if ! have_cmd poetry; then
    pipx install poetry
  fi
  if python_prereqs_healthy && have_cmd poetry; then
    pass "Ubuntu Python/Poetry prerequisites are installed"
  else
    fail "Ubuntu Python/Poetry prerequisite install did not produce a healthy Python/Poetry environment"
  fi
}

install_ubuntu_docker_prereqs() {
  case "$DRIVER" in
    docker|auto) ;;
    *) return 0 ;;
  esac

  local version
  version="$(docker_server_version)"
  if [ -n "$version" ] && version_ge "$version" "$MIN_DOCKER_VERSION"; then
    return 0
  fi

  if [ "$CHECK_ONLY" -eq 1 ]; then
    report "CHECK_ONLY: compatible Ubuntu host can install Docker Engine from Docker's official apt repository"
    return 0
  fi
  if [ "$SKIP_PREREQ_INSTALL" -eq 1 ]; then
    report "Skipping Ubuntu Docker prerequisite install due to --skip-prereq-install"
    return 0
  fi
  confirm_action "Install or repair Docker Engine from Docker's official Ubuntu apt repository?" || return

  report "Installing Docker Engine prerequisites from Docker's official apt repository"
  export DEBIAN_FRONTEND="${DEBIAN_FRONTEND:-noninteractive}"
  run_sudo apt-get update
  run_sudo apt-get install -y ca-certificates curl
  run_sudo install -m 0755 -d /etc/apt/keyrings
  run_sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  run_sudo chmod a+r /etc/apt/keyrings/docker.asc
  local codename arch
  codename="$(
    # shellcheck disable=SC1091
    . /etc/os-release && printf '%s\n' "${UBUNTU_CODENAME:-$VERSION_CODENAME}"
  )"
  arch="$(dpkg --print-architecture)"
  {
    printf '%s\n' "Types: deb"
    printf '%s\n' "URIs: https://download.docker.com/linux/ubuntu"
    printf '%s\n' "Suites: ${codename}"
    printf '%s\n' "Components: stable"
    printf '%s\n' "Architectures: ${arch}"
    printf '%s\n' "Signed-By: /etc/apt/keyrings/docker.asc"
  } | run_sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null
  run_sudo apt-get update
  run_sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

  if have_cmd systemctl; then
    run_sudo systemctl enable --now docker.service || run_sudo systemctl start docker.service || true
    run_sudo systemctl enable --now containerd.service || true
  elif have_cmd service; then
    run_sudo service docker start || true
  fi

  if docker_needs_linux_group_refresh; then
    run_sudo groupadd -f docker
    run_sudo usermod -aG docker "$USER"
    PREREQ_RERUN_REQUIRED=1
    fail "Docker access requires refreshed group membership; log out and back in or run 'newgrp docker', then rerun this script"
    return
  fi

  version="$(docker_server_version)"
  if [ -n "$version" ] && version_ge "$version" "$MIN_DOCKER_VERSION"; then
    pass "Docker Engine ${version} is installed and reachable"
  else
    fail "Docker Engine install completed, but the Docker daemon is not reachable"
  fi
}

detect_colima_socket() {
  local socket
  if [ -n "${DOCKER_HOST:-}" ] && [ "${DOCKER_HOST#unix://}" != "$DOCKER_HOST" ]; then
    socket="${DOCKER_HOST#unix://}"
    [ -S "$socket" ] && { printf '%s\n' "$socket"; return 0; }
  fi
  for socket in \
    "${HOME}/.colima/openshell/docker.sock" \
    "${HOME}/.colima/default/docker.sock" \
    "${HOME}/.colima/docker.sock" \
    "${HOME}/.docker/run/docker.sock"; do
    [ -S "$socket" ] && { printf '%s\n' "$socket"; return 0; }
  done
  return 1
}

install_macos_colima_prereqs() {
  case "$DRIVER" in
    docker|auto) ;;
    *) return 0 ;;
  esac

  if docker_daemon_accessible; then
    return 0
  fi

  if [ "$CHECK_ONLY" -eq 1 ]; then
    report "CHECK_ONLY: compatible macOS host can install/start Colima with Homebrew for Docker-backed OpenShell"
    return 0
  fi
  if [ "$SKIP_PREREQ_INSTALL" -eq 1 ]; then
    report "Skipping macOS Colima prerequisite install due to --skip-prereq-install"
    return 0
  fi
  confirm_action "Install/start Colima and Docker CLI for macOS OpenShell?" || return

  if ! have_cmd brew; then
    fail "Homebrew is required to install Colima and Docker CLI on macOS"
    return
  fi
  if ! have_cmd colima; then
    brew install colima
  fi
  if ! have_cmd docker; then
    brew install docker
  fi

  if ! colima status openshell >/dev/null 2>&1; then
    colima start openshell --runtime docker
  elif ! docker_daemon_accessible; then
    colima start openshell --runtime docker
  fi

  local socket
  socket="$(detect_colima_socket || true)"
  if [ -n "$socket" ]; then
    export DOCKER_HOST="unix://${socket}"
    pass "Using Colima Docker socket at ${socket}"
  else
    fail "Colima started, but no Docker socket was detected"
  fi
}

install_prerequisites_if_supported() {
  if [ "$UNSUPPORTED" -eq 1 ]; then
    return
  fi
  if ubuntu_prereq_install_supported; then
    install_ubuntu_python_poetry_prereqs
    install_ubuntu_docker_prereqs
  elif [ "$SYSTEM_NAME" = "Linux" ] && [ "$DISTRO_ID" = "ubuntu" ] && [ "$CHECK_ONLY" -eq 1 ]; then
    report "CHECK_ONLY: Ubuntu ${DISTRO_VERSION_ID:-unknown} is not eligible for automated prerequisite install for this notebook setup"
  fi

  if macos_colima_install_supported; then
    install_macos_colima_prereqs
  fi
}

validate_openshell_cli() {
  local candidate cli_output cli_version path_openshell
  path_openshell="$(command -v openshell 2>/dev/null || true)"
  for candidate in \
    "$path_openshell" \
    "${OPEN_SHELL_ROOT}/.venv/bin/openshell" \
    "/opt/homebrew/bin/openshell" \
    "/usr/local/bin/openshell" \
    "/usr/bin/openshell"; do
    [ -n "$candidate" ] || continue
    [ -x "$candidate" ] || continue
    cli_output="$("$candidate" --version 2>/dev/null || true)"
    cli_version="$(printf '%s\n' "$cli_output" | extract_openshell_version)"
    if [ "$cli_version" = "$OPENSHELL_VERSION" ]; then
      OPENSHELL_BIN="$candidate"
      pass "openshell CLI version ${cli_version} at ${candidate}"
      if [ -n "$path_openshell" ] && [ "$path_openshell" != "$candidate" ]; then
        warn "PATH openshell is ${path_openshell}; using ${candidate} for validation because it matches ${OPENSHELL_VERSION}"
      fi
      return
    fi
  done

  if [ -z "$path_openshell" ]; then
    fail "openshell CLI is not on PATH and no matching fallback was found"
    return
  fi
  cli_output="$("$path_openshell" --version 2>/dev/null || true)"
  cli_version="$(printf '%s\n' "$cli_output" | extract_openshell_version)"
  fail "openshell CLI version mismatch: expected ${OPENSHELL_VERSION}, got ${cli_output:-unknown} at ${path_openshell}"
}

install_openshell_system() {
  if [ "$CHECK_ONLY" -eq 1 ] || [ "$SKIP_SYSTEM_INSTALL" -eq 1 ]; then
    report "Skipping OpenShell system install"
    return
  fi

  if [ "$ASSUME_YES" -ne 1 ]; then
    printf 'Install/update OpenShell CLI and gateway to v%s with NVIDIA installer? [y/N] ' "$OPENSHELL_VERSION"
    local answer
    read -r answer
    case "$answer" in
      y|Y|yes|YES) ;;
      *)
        fail "User declined OpenShell system install"
        return
        ;;
    esac
  fi

  report "Installing OpenShell v${OPENSHELL_VERSION} with NVIDIA installer"
  if curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh | OPENSHELL_VERSION="v${OPENSHELL_VERSION}" sh; then
    pass "OpenShell installer completed"
  else
    fail "OpenShell installer failed for v${OPENSHELL_VERSION}"
  fi
}

validate_driver() {
  case "$DRIVER" in
    docker|auto)
      if [ -z "${DOCKER_HOST:-}" ] && [ "$SYSTEM_NAME" = "Darwin" ]; then
        local colima_socket
        colima_socket="$(detect_colima_socket || true)"
        if [ -S "$colima_socket" ]; then
          export DOCKER_HOST="unix://${colima_socket}"
          pass "Using Colima Docker socket at ${colima_socket}"
        fi
      fi
      if [ "${OPENSHELL_TEST_NO_DOCKER:-0}" = "1" ]; then
        if [ "$DRIVER" = "docker" ]; then
          fail "Docker is required for the notebook's local demo image path"
        else
          warn "Docker not found during auto driver validation"
        fi
      elif have_cmd docker; then
        local docker_client docker_server
        docker_client="$(docker version --format '{{.Client.Version}}' 2>/dev/null || true)"
        docker_server="$(docker version --format '{{.Server.Version}}' 2>/dev/null || true)"
        if [ -n "$docker_server" ] && version_ge "$docker_server" "$MIN_DOCKER_VERSION"; then
          pass "Docker ${docker_server} satisfies ${MIN_DOCKER_VERSION}+"
          GATEWAY_DRIVER="docker"
        elif [ -n "$docker_server" ]; then
          fail "Docker ${docker_server} is below required ${MIN_DOCKER_VERSION}"
        elif [ -n "$docker_client" ]; then
          fail "Docker CLI ${docker_client} exists but daemon is not reachable"
        else
          fail "Docker CLI exists but daemon is not reachable"
        fi
      elif [ "$DRIVER" = "docker" ]; then
        fail "Docker is required for the notebook's local demo image path"
      else
        warn "Docker not found during auto driver validation"
      fi
      ;;
  esac

  case "$DRIVER" in
    podman|auto)
      if have_cmd podman; then
        local podman_version
        podman_version="$(podman --version 2>/dev/null | awk '{print $NF}' || true)"
        if version_ge "$podman_version" "5.0"; then
          pass "Podman ${podman_version} satisfies 5.x requirement"
        else
          fail "Podman ${podman_version:-unknown} is below required 5.x"
        fi
      elif [ "$DRIVER" = "podman" ]; then
        fail "Podman is required for --driver podman"
      fi
      ;;
  esac

  case "$DRIVER" in
    kubernetes|auto)
      if have_cmd kubectl; then
        pass "kubectl found: $(command -v kubectl)"
      elif [ "$DRIVER" = "kubernetes" ]; then
        fail "kubectl is required for --driver kubernetes"
      fi
      if have_cmd helm; then
        pass "helm found: $(command -v helm)"
      elif [ "$DRIVER" = "kubernetes" ]; then
        fail "helm is required for --driver kubernetes"
      fi
      ;;
  esac

  case "$DRIVER" in
    vm)
      if [ "$SYSTEM_NAME" = "Darwin" ] && [ "$ARCH_NAME" = "arm64" ]; then
        pass "VM driver can use Hypervisor.framework on Apple Silicon"
      elif [ "$SYSTEM_NAME" = "Linux" ] && [ -e /dev/kvm ]; then
        pass "VM driver can use /dev/kvm"
      else
        fail "VM driver requires Hypervisor.framework on macOS arm64 or /dev/kvm on Linux"
      fi
      ;;
  esac
}

write_managed_env_block() {
  local env_file managed_driver managed_docker_host tmp_file
  env_file="$1"
  managed_driver="$2"
  managed_docker_host="${3:-}"
  tmp_file="$(mktemp)"

  mkdir -p "$(dirname "$env_file")"
  if [ -f "$env_file" ]; then
    awk '
      /^# BEGIN langchain-nvidia OpenShell setup$/ { skip = 1; next }
      /^# END langchain-nvidia OpenShell setup$/ { skip = 0; next }
      skip == 0 { print }
    ' "$env_file" > "$tmp_file"
  fi

  {
    printf '%s\n' '# BEGIN langchain-nvidia OpenShell setup'
    printf 'OPENSHELL_DRIVERS=%s\n' "$managed_driver"
    if [ -n "$managed_docker_host" ]; then
      printf 'DOCKER_HOST=%s\n' "$managed_docker_host"
    fi
    printf '%s\n' '# END langchain-nvidia OpenShell setup'
  } >> "$tmp_file"

  mv "$tmp_file" "$env_file"
}

configure_gateway_launch_environment() {
  if [ "$CHECK_ONLY" -eq 1 ]; then
    report "Skipping gateway launch environment changes in --check-only mode"
    return
  fi

  local selected_driver env_file
  selected_driver="${GATEWAY_DRIVER:-$DRIVER}"
  if [ "$selected_driver" = "auto" ]; then
    selected_driver="docker"
  fi

  case "$selected_driver" in
    docker|podman|kubernetes|vm) ;;
    *)
      warn "Skipping gateway launch environment; unresolved driver ${selected_driver}"
      return
      ;;
  esac

  env_file="${XDG_CONFIG_HOME:-${HOME}/.config}/openshell/gateway.env"
  case "$selected_driver" in
    docker)
      write_managed_env_block "$env_file" "$selected_driver" "${DOCKER_HOST:-}"
      ;;
    *)
      write_managed_env_block "$env_file" "$selected_driver" ""
      ;;
  esac
  pass "Gateway launch environment configured at ${env_file}"
}

restart_gateway_service() {
  if [ "$CHECK_ONLY" -eq 1 ]; then
    report "Skipping gateway restart in --check-only mode"
    return
  fi

  case "$SYSTEM_NAME" in
    Darwin)
      if have_cmd brew; then
        if brew services restart openshell >/tmp/openshell-service-restart.txt 2>&1; then
          pass "Homebrew OpenShell gateway service restarted"
        else
          fail "Homebrew OpenShell gateway service restart failed"
          sed 's/^/  /' /tmp/openshell-service-restart.txt | tee -a "$REPORT_FILE"
        fi
      else
        warn "Homebrew is unavailable; cannot restart the macOS OpenShell service"
      fi
      ;;
    Linux)
      if have_cmd systemctl; then
        if systemctl --user restart openshell-gateway >/tmp/openshell-service-restart.txt 2>&1; then
          pass "systemd user OpenShell gateway service restarted"
        else
          warn "systemd user OpenShell gateway restart failed; continuing to validation"
          sed 's/^/  /' /tmp/openshell-service-restart.txt | tee -a "$REPORT_FILE"
        fi
      else
        warn "systemctl is unavailable; cannot restart the Linux OpenShell service"
      fi
      ;;
  esac
  rm -f /tmp/openshell-service-restart.txt
}

configure_poetry_environment() {
  if [ "$CHECK_ONLY" -eq 1 ]; then
    report "Skipping Poetry environment changes in --check-only mode"
    return
  fi
  if ! have_cmd poetry; then
    fail "Cannot configure environment without Poetry"
    return
  fi

  (
    cd "$OPEN_SHELL_ROOT"
    poetry config virtualenvs.in-project true --local
    poetry install --with test
    env_path="$(poetry env info --path)"
    env_python="${env_path}/bin/python"
    if ! "$env_python" -m pip --version >/dev/null 2>&1; then
      "$env_python" -m ensurepip --upgrade
    fi
    "$env_python" -m pip install --upgrade ipykernel
    "$env_python" -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$DISPLAY_NAME"
  )
  pass "Poetry environment and Jupyter kernel are installed"
}

validate_python_imports() {
  local python_for_env
  python_for_env="$PYTHON_BIN"
  if [ -x "${OPEN_SHELL_ROOT}/.venv/bin/python" ]; then
    python_for_env="${OPEN_SHELL_ROOT}/.venv/bin/python"
  fi
  [ -n "$python_for_env" ] || return

  if "$python_for_env" - <<'PY' >/tmp/openshell-import-check.txt 2>&1
import grpc

def version_tuple(value):
    head = value.split("-", 1)[0].split("+", 1)[0]
    parts = []
    for item in head.split("."):
        try:
            parts.append(int(item))
        except ValueError:
            parts.append(0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])

if version_tuple(grpc.__version__) < (1, 78, 0):
    raise SystemExit(f"grpcio {grpc.__version__} is below required 1.78.0")

import google.protobuf  # noqa: F401
import openshell  # noqa: F401
import langchain_nvidia_openshell  # noqa: F401

print("imports OK")
PY
  then
    pass "OpenShell SDK, grpcio, protobuf, and langchain_nvidia_openshell imports are healthy"
  else
    fail "Python import validation failed"
    sed 's/^/  /' /tmp/openshell-import-check.txt | tee -a "$REPORT_FILE"
  fi
  rm -f /tmp/openshell-import-check.txt
}

validate_kernel() {
  if have_cmd jupyter; then
    if jupyter kernelspec list 2>/dev/null | grep -q "$KERNEL_NAME"; then
      pass "Jupyter kernel ${KERNEL_NAME} is registered"
    else
      fail "Jupyter kernel ${KERNEL_NAME} is not registered"
    fi
  else
    warn "Cannot list kernels because jupyter is not on PATH"
  fi
}

validate_gateway() {
  local cli
  cli="${OPENSHELL_BIN:-$(command -v openshell 2>/dev/null || true)}"
  if [ -n "$cli" ]; then
    if "$cli" status >/tmp/openshell-status-check.txt 2>&1; then
      pass "openshell status succeeded"
    else
      fail "openshell status failed"
      sed 's/^/  /' /tmp/openshell-status-check.txt | tee -a "$REPORT_FILE"
    fi
    if "$cli" sandbox list >/tmp/openshell-sandbox-list-check.txt 2>&1; then
      pass "openshell sandbox list succeeded"
    else
      fail "openshell sandbox list failed"
      sed 's/^/  /' /tmp/openshell-sandbox-list-check.txt | tee -a "$REPORT_FILE"
    fi
  fi
  rm -f /tmp/openshell-status-check.txt /tmp/openshell-sandbox-list-check.txt
}

main() {
  report "OpenShell notebook setup report"
  report "Script: ${BASH_SOURCE[0]}"
  report "OpenShell target version: ${OPENSHELL_VERSION}"
  report "grpcio runtime floor: ${MIN_GRPCIO_VERSION}"
  report "Driver target: ${DRIVER}"
  report "Report file: ${REPORT_FILE}"

  detect_os
  validate_platform
  install_prerequisites_if_supported
  if [ "$PREREQ_RERUN_REQUIRED" -eq 1 ]; then
    report "RESULT: prerequisite changes require a refreshed shell session before OpenShell setup can continue."
    exit 1
  fi
  find_python
  validate_poetry_and_jupyter
  validate_openshell_wheel_resolution
  validate_driver

  if [ "$FAILED" -ne 0 ]; then
    report "Stopping before environment changes because prerequisite validation failed"
  else
    install_openshell_system
    configure_gateway_launch_environment
    restart_gateway_service
    validate_openshell_cli
    configure_poetry_environment
    validate_python_imports
    validate_kernel
    validate_gateway
  fi

  if [ "$FAILED" -eq 0 ]; then
    if [ "$WARNED" -eq 0 ]; then
      report "SUCCESS: OpenShell notebook environment is ready."
    else
      report "SUCCESS_WITH_WARNINGS: OpenShell notebook environment is ready; review warnings above."
    fi
    exit 0
  fi

  if [ "$UNSUPPORTED" -eq 1 ]; then
    report "RESULT: unsupported environment for this notebook setup."
  else
    report "RESULT: setup validation failed."
  fi
  exit 1
}

if [ "${OPENSHELL_SETUP_TESTING:-0}" != "1" ]; then
  main "$@"
fi
