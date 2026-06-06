# OpenShell 0.0.57 Validation Report

Date: 2026-06-06

This report records the OpenShell upgrade, live validation results, deployment
findings, and AI-Q adoption path for `langchain-nvidia-openshell`.

## Executive Summary

- Upgraded `langchain-nvidia-openshell` to `openshell>=0.0.57,<0.1`.
- Added an explicit `grpcio>=1.78,<2` floor because the OpenShell `0.0.57`
  generated stubs require newer gRPC APIs at import time.
- Verified the OpenShell SDK execution contract still accepts
  `command: Sequence[str]`, `env`, `stdin`, `workdir`, and `timeout_seconds`.
- Fixed two issues found only during live OpenShell VM-driver testing:
  multiline shell commands now run through shell stdin, and uploads now chunk
  raw payloads at 512 KiB.
- Added a credentials-free LangChain Deep Agents integration regression that
  proves Deep Agents' built-in `execute` tool calls the OpenShell backend.
- Added an OpenShell `0.0.57` validation plan covering supported operating
  systems, compute drivers, policy coverage, and release-upgrade gates.
- Added AI-Q deep researcher OpenShell provider support and an AI-Q adoption
  test plan in the AI-Q repository.

## Upstream Version and Support Evidence

Latest package check on 2026-06-06:

```text
openshell (0.0.57)
LATEST: 0.0.57
```

Local SDK inspection:

```text
openshell 0.0.57
grpc 1.80.0
Sandbox.exec(command: Sequence[str], stream_output, workdir, env, stdin, timeout_seconds)
SandboxSession.exec(command: Sequence[str], stream_output, workdir, env, stdin, timeout_seconds)
```

Primary upstream references:

- GitHub releases: https://github.com/NVIDIA/OpenShell/releases
- Support matrix: https://docs.nvidia.com/openshell/reference/support-matrix
- Compute drivers: https://docs.nvidia.com/openshell/reference/sandbox-compute-drivers
- Supported agents/default policy coverage:
  https://docs.nvidia.com/openshell/latest/about/supported-agents.html
- Installation:
  https://docs.nvidia.com/openshell/latest/about/installation

## Supported Test Matrix

OpenShell `0.0.57` should be validated on the following host and driver matrix
before release qualification:

| Platform | Architecture | Status | Drivers to test |
|---|---|---|---|
| Linux Debian/Ubuntu | amd64 | Supported | Docker, Podman, Kubernetes, MicroVM/KVM |
| Linux Debian/Ubuntu | arm64 | Supported | Docker, Podman, Kubernetes, MicroVM/KVM |
| macOS Apple Silicon | arm64 | Supported | Docker Desktop, MicroVM/Hypervisor.framework |
| Windows WSL 2 | amd64 | Experimental | Docker Desktop through WSL 2 |

Minimum driver/runtime versions from upstream docs:

| Driver | Minimum runtime | Notes |
|---|---|---|
| Docker | Docker Desktop or Docker Engine 28.04 | Local development and single-machine gateways |
| Podman | Podman 5.x | Rootless workstation workflows require socket/rootless networking checks |
| Kubernetes | Kubernetes 1.29, Helm 3.x | Use OpenShell Helm chart and gateway image |
| MicroVM | Host virtualization | Uses Hypervisor.framework on macOS and KVM on Linux |

OpenShell gateway images are published for `linux/amd64` and `linux/arm64`.
Sandbox images are maintained separately in NVIDIA OpenShell Community.

## Deployment Findings

### Gateway Driver Selection

When Docker and Podman are unavailable or unhealthy, OpenShell may fail gateway
startup with no compute driver selected. The VM driver is not auto-detected by
the gateway; start the gateway with:

```bash
OPENSHELL_DRIVERS=vm openshell-gateway
```

or configure the equivalent gateway TOML setting.

### macOS MicroVM Prerequisite

Live macOS Apple Silicon VM-driver validation found that OpenShell `0.0.57`
needs `mkfs.ext4` from `e2fsprogs` to format MicroVM root filesystems. With
Homebrew, install `e2fsprogs` and ensure `mkfs.ext4` is visible to the gateway
service under the prefix it uses.

### Policy Coverage

The default policy is useful but not sufficient for all agents. Upstream
supported-agent docs currently list:

- Claude Code: full default-policy coverage.
- OpenCode: partial default-policy coverage.
- Codex: no default-policy coverage; requires a custom policy.

For LangChain Deep Agents, use explicit policy files for the commands,
binaries, package endpoints, and external APIs needed by the demo or AI-Q
deployment.

## Issues Found and Fixed

### Multiline Shell Commands

Deep Agents' inherited file helpers generate multiline shell snippets such as
`python3 -c "..." 2>&1`. OpenShell `0.0.57` rejected newline-bearing `bash -c`
argv payloads at the RPC layer.

Fix:

- Single-line commands still use `["bash", "-c", command]`.
- Multiline commands use `["bash", "-s"]` with the script sent on stdin.

### Upload Payload Size

The original 4 MiB raw upload chunk size became larger after base64 expansion
and exceeded the live VM-driver stdin transport limit. Live probing accepted
512 KiB raw chunks and rejected 768 KiB raw chunks.

Fix:

- Default `max_upload_chunk_bytes` is now 512 KiB.
- Large uploads still append chunks to preserve exact bytes.

### Stale Gateway Detection

Integration tests originally skipped only by checking OpenShell config files,
which allowed stale active gateway pointers to produce setup errors.

Fix:

- Integration tests now perform a non-mutating `SandboxClient.list(limit=1)`
  readiness probe and skip with a useful connection error when the gateway is
  not reachable.

## Validation Results

Live validation was run against OpenShell `0.0.57` with the VM compute driver
on macOS Apple Silicon after installing the required MicroVM prerequisites.

`langchain-nvidia` / `libs/openshell`:

```text
poetry run python -m pytest tests/unit_tests -q
44 passed, 1 warning

poetry run python -m pytest tests/integration_tests -q -rs
92 passed, 1 warning

poetry run ruff check langchain_nvidia_openshell tests
passed

poetry run ruff format langchain_nvidia_openshell tests --diff
clean

poetry run python -m compileall -q langchain_nvidia_openshell tests
passed
```

The integration suite includes:

- direct `execute` smoke;
- upload/download round trip;
- Deep Agents notebook tool-pattern smoke;
- credentials-free Deep Agents built-in `execute` tool smoke;
- LangChain `SandboxIntegrationTests` conformance suite.

AI-Q deep researcher:

```text
uv run pytest tests/aiq_agent/agents/deep_researcher/test_deepagents_runtime.py \
  tests/aiq_agent/agents/deep_researcher/test_agent.py -q
46 passed, 1 warning

uv run pytest tests/aiq_agent/jobs/test_runner.py -q
80 passed, 1 warning

uv run ruff check \
  src/aiq_agent/agents/deep_researcher/deepagents_runtime.py \
  tests/aiq_agent/agents/deep_researcher/test_deepagents_runtime.py \
  docs/source/architecture/agents \
  docs/source/examples/skills-sandbox/index.md
passed
```

AI-Q live provider smoke:

```text
SandboxConfig(provider="openshell")
execute -> exit 0, output "aiq-openshell-smoke"
upload /tmp/aiq-openshell-smoke.txt -> success
download /tmp/aiq-openshell-smoke.txt -> b"hello-aiq"
backend.close() -> closed sandbox context
```

## Documentation Updates

Updated or added:

- `README.md`: OpenShell `0.0.57`, support matrix, deployment notes, API
  defaults, and VM-driver prerequisites.
- `nvidia_openshell_sandbox.ipynb`: version pins and demo image tag.
- This validation report.
- AI-Q `sandbox.md`: Modal and OpenShell provider behavior.
- AI-Q `skills-sandbox` example: OpenShell provider config.

## AI-Q Adoption Path

AI-Q now supports:

```yaml
sandbox:
  provider: openshell
  cluster: null
  sandbox_name_prefix: aiq-deep-research
  ready_timeout_seconds: 300
  delete_on_exit: true
  shell:
    - bash
    - -c
```

Implementation notes:

- The OpenShell backend is lazy and creates the sandbox on first operation.
- `execute`, `upload_files`, and `download_files` delegate to
  `OpenShellSandbox`.
- The backend retries once if a stale/not-found sandbox error occurs.
- `close()` is available for future async-job cleanup hooks.
- The OpenShell Python SDK does not currently expose a CLI-equivalent named
  sandbox creation path, so SDK-created gateway sandboxes are anonymous. AI-Q
  derives a stable local backend identity from `sandbox_name_prefix` and job ID
  until upstream exposes named creation in the SDK.
- `langchain-nvidia-openshell` is not yet published on PyPI. AI-Q uses lazy
  imports and a clear runtime error instead of adding it to the base dependency
  set. This also avoids breaking AI-Q's Python 3.11 support, because OpenShell
  requires Python 3.12+.

## Remaining Cross-Platform Work

The local VM-driver validation is green, but release qualification should still
run the full battery on:

- Linux amd64 with Docker.
- Linux arm64 with Docker or Podman.
- macOS Apple Silicon with Docker Desktop.
- Windows WSL 2 with Docker Desktop, marked experimental.
- Kubernetes, when targeting shared clusters.
- MicroVM on Linux/KVM if VM-backed isolation is required there.

For each platform, record:

- OpenShell CLI and SDK version.
- Gateway driver and configuration.
- Sandbox image reference.
- Policy file hash.
- Test command output.
- Sandbox cleanup evidence.

## Conclusion

OpenShell `0.0.57` is integrated and live-tested for the LangChain Deep Agents
adapter and is now available as an AI-Q deep researcher sandbox provider. The
local macOS VM-driver path is green after documented prerequisites. The path to
broader confidence is cross-platform execution of the validation plan rather
than additional local code changes.
