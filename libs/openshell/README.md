# langchain-nvidia-openshell

[NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell) sandbox backend for
[LangChain Deep Agents](https://docs.langchain.com/oss/python/deepagents/sandboxes).

> ⚠️ **Alpha**. NVIDIA OpenShell itself is alpha software ("single-player mode")
> and this integration tracks its API closely. Expect breakage.

`OpenShellSandbox` adapts NVIDIA's OpenShell — a policy-enforced, sandboxed
runtime for autonomous agents — to the LangChain `BaseSandbox` interface, so
your `ChatNVIDIA`-driven agent (running on the host) can dispatch shell
commands, file uploads, and file downloads into a hardened sandbox without
ever placing the agent itself inside.

This is the **sandbox-as-tool** pattern: the agent stays out, the tools go in.

```text
                  ┌──────────────────────────────┐
                  │   Host (your agent)          │
                  │                              │
                  │   ChatNVIDIA(...)            │
                  │       │                      │
                  │       ▼                      │
                  │   create_deep_agent(         │
                  │     model=...,               │
                  │     backend=OpenShellSandbox │
                  │   )                          │
                  └──────────────┬───────────────┘
                                 │  gRPC (mTLS)
                                 ▼
                  ┌──────────────────────────────┐
                  │   OpenShell Gateway          │
                  └──────────────┬───────────────┘
                                 │  ExecSandbox stream
                                 ▼
                  ┌──────────────────────────────┐
                  │   Sandbox compute driver     │
                  │   Docker/Podman/K8s/MicroVM  │
                  │   policy-enforced execution  │
                  └──────────────────────────────┘
```

## Install

```bash
pip install langchain-nvidia-openshell
```

You also need [the OpenShell CLI / gateway](https://docs.nvidia.com/openshell/latest/get-started/quickstart/), pinned to match the SDK:

```bash
curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh \
  | OPENSHELL_VERSION=v0.0.72 sh
```

Confirm with `openshell --version`; this integration is tested against
OpenShell `0.0.72+`.

## Quickstart (local)

There are three moving parts:

1. Start or connect to an OpenShell gateway.
2. Create an OpenShell sandbox.
3. Attach a Deep Agent to that sandbox through `OpenShellSandbox`.

First, make sure the gateway is reachable:

```bash
openshell status
openshell sandbox list
```

Then choose how the sandbox lifecycle should work.

### Option A: Python-owned sandbox

Use this when each Python process should create its own short-lived sandbox.
`openshell.Sandbox(...)` uses the active OpenShell gateway configuration and
the OpenShell SDK defaults unless you pass explicit SDK options. The adapter
does not choose an image, policy, driver, or provider profile.

```python
import openshell
from langchain_nvidia_openshell import OpenShellSandbox

with openshell.Sandbox(delete_on_exit=True) as sb:
    backend = OpenShellSandbox(sandbox=sb)

    print(backend.execute("uname -a").output)
    backend.upload_files([("/sandbox/hello.py", b"print('hello from openshell')\n")])
    print(backend.execute("python3 /sandbox/hello.py").output)
    print(backend.download_files(["/sandbox/hello.py"])[0].content)
```

### Option B: CLI-created sandbox by name

Use this when you want an explicit policy, image, or long-lived sandbox that
can be reused across agent runs. Create it once with the OpenShell CLI:

```bash
openshell sandbox create \
  --name langchain-demo \
  --policy ./example-policy.yaml \
  --keep --no-tty -- bash
```

Then attach Python to the named sandbox:

```python
import openshell
from langchain_nvidia_openshell import OpenShellSandbox

client = openshell.SandboxClient.from_active_cluster()
try:
    backend = OpenShellSandbox(sandbox=client.get_session("langchain-demo"))
    print(backend.execute("pwd").output)
finally:
    client.close()
```

A named sandbox keeps running until you delete it with
`openshell sandbox delete langchain-demo`, so it can be reused by later Python
processes. That is convenient for demos and debugging; for production agent
runs, prefer a clear cleanup policy so stale state does not leak between runs.

## Policies & security model

OpenShell policy controls the sandbox process, filesystem, and network
surface. `OpenShellSandbox` is policy-agnostic: the sandbox you pass in
already carries its policy, and this adapter only forwards Deep Agent tool
calls into that sandbox.

| Layer | What it controls | Enforced by | Set at create | Hot-reloadable |
|---|---|---|---|---|
| **Filesystem** | RO/RW mounts inside the sandbox | Linux Landlock LSM | ✅ | ❌ locked once running |
| **Network** | Outbound hosts, ports, HTTP methods/paths | In-sandbox HTTP CONNECT proxy + OPA/Rego | ✅ | ✅ via `openshell policy set` |
| **Process** | Run-as user/group, syscall filter | seccomp BPF + dropped privileges | ✅ | ❌ locked once running |

The stock community `base` image ships a default policy, but agent coverage is
not universal: NVIDIA's current default policy reference lists Claude Code as
fully covered, OpenCode as partially covered, and Codex as requiring a custom
policy. For LangChain Deep Agents, prefer an explicit policy that permits the
runtime tools you need (`bash`, `python3`, `curl`, package managers, and exact
network destinations) rather than depending on an agent preset.

Inference routing is adjacent gateway configuration, not network-policy CRUD.
If code inside the sandbox calls `https://inference.local`, policy can allow or
deny that hostname, but the provider credentials and model route are set with
OpenShell provider and inference commands.

### Policy and inference workflows

**1. Apply policy at create time.** This is the only path that can change
locked Filesystem and Process settings:

```bash
openshell sandbox create \
  --policy ./example-policy.yaml \
  --keep --no-tty -- bash
```

**2. Hot-reload network policy.** This works for Network rules without
recreating the sandbox:

```bash
openshell policy set <sandbox-name> --policy ./tighten.yaml --wait
```

**3. Configure managed inference separately.** Create a provider record, then
point `inference.local` at the model you want:

```bash
openshell provider create --name nvidia-prod --type nvidia --from-existing
openshell inference set \
  --provider nvidia-prod \
  --model nvidia/nemotron-3-nano-30b-a3b
openshell inference get
```

`--from-existing` reads `NVIDIA_API_KEY` from the host environment. The key
stays with the gateway; the sandbox reaches it through `inference.local` only
when its network policy allows that route.

A complete example policy ships in
[`docs/sandboxes/example-policy.yaml`](docs/sandboxes/example-policy.yaml).
Copy it, edit it, and pass it to `openshell sandbox create --policy ...`.
To change locked Filesystem or Process defaults, recreate the sandbox with a
new policy or bake `/etc/openshell/policy.yaml` into a custom image.

### Further reading

- [OpenShell docs](https://docs.nvidia.com/openshell/latest/) — top-level.
- [Security controls & defaults](https://docs.nvidia.com/openshell/latest/security) — what each setting protects against.
- [Policy schema reference](https://docs.nvidia.com/openshell/latest/reference/policy-schema) — every YAML field.
- [`security-policy.md`](https://github.com/NVIDIA/OpenShell/blob/main/architecture/security-policy.md) — design doc with locked-vs-hot-reloadable rationale.
- [`sandbox-policy-quickstart`](https://github.com/NVIDIA/OpenShell/tree/main/examples/sandbox-policy-quickstart) — runnable demo of the create-then-tighten workflow.

## Use with Deep Agents + ChatNVIDIA

```python
import openshell
from deepagents import create_deep_agent
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_nvidia_openshell import OpenShellSandbox

with openshell.Sandbox() as sb:
    backend = OpenShellSandbox(sandbox=sb)
    agent = create_deep_agent(
        model=ChatNVIDIA(model="nvidia/nemotron-3-nano-30b-a3b"),
        system_prompt="You are a careful coding agent.",
        backend=backend,
    )
    print(agent.invoke({"messages": [{"role": "user",
        "content": "Compute the sha256 of '/etc/hostname' inside the sandbox."}]}))
```

## Deployment support matrix

OpenShell `0.0.72` has two relevant compatibility layers: the gateway/CLI host
matrix from NVIDIA, and the Python SDK wheels required by this notebook.

| Target | Gateway status | Notebook SDK status | Required runtime |
|---|---|---|---|
| Linux Debian/Ubuntu amd64 | Supported | Requires a host that can install `manylinux_2_39_x86_64` wheels; Ubuntu 24.04+ is the expected floor | Docker Engine/Desktop 28.04+, Podman 5.x, Kubernetes 1.29+, or MicroVM host virtualization |
| Linux Debian/Ubuntu arm64 | Supported | Requires a host that can install `manylinux_2_39_aarch64` wheels; Ubuntu 24.04+ is the expected floor | Docker Engine/Desktop 28.04+, Podman 5.x, Kubernetes 1.29+, or MicroVM host virtualization |
| macOS Apple Silicon | Supported | Requires macOS 13+ for the `macosx_13_0_arm64` OpenShell SDK wheel | Docker Desktop for container-backed sandboxes or Hypervisor.framework for MicroVM |
| Windows WSL 2 amd64 | Experimental | Not automated by this notebook setup | Docker Desktop through WSL 2 |

Compute drivers are selected at the gateway layer: `docker`, `podman`,
`kubernetes`, or `vm` for MicroVM-backed sandboxes. The OpenShell gateway image
is published for `linux/amd64` and `linux/arm64`; sandbox images are maintained
separately in the NVIDIA OpenShell Community repository.

On macOS with the `vm` driver, the gateway may need explicit driver selection
when Docker/Podman is unavailable:

```bash
OPENSHELL_DRIVERS=vm openshell-gateway
```

OpenShell `0.0.72` also expects `mkfs.ext4` from `e2fsprogs` while preparing
MicroVM root filesystems. With Homebrew, install `e2fsprogs` and make sure
`mkfs.ext4` is visible under the prefix used by the gateway service.

For the notebook, run the setup script from the sandbox docs directory. It
validates the OS, OpenShell release, Python SDK wheel tag, `grpcio` runtime,
driver readiness, Poetry environment, and Jupyter kernel registration:

```bash
cd libs/openshell/docs/sandboxes
./setup_openshell.sh --openshell-version 0.0.72
```

## API

```python
class OpenShellSandbox(BaseSandbox):
    def __init__(
        self,
        *,
        sandbox: openshell.Sandbox | openshell.SandboxSession,
        timeout: int = 30 * 60,
        shell: tuple[str, ...] = ("bash", "-c"),
        max_output_bytes: int = 1 << 20,           # 1 MiB
        max_upload_chunk_bytes: int = 512 * 1024,  # 512 KiB
    ) -> None: ...

    @property
    def id(self) -> str: ...

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse: ...
    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]: ...
    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]: ...
    # `aexecute`, `aupload_files`, `adownload_files`, and all other file ops
    # (`ls`, `read`, `write`, `edit`, `glob`, `grep`) are inherited from
    # `BaseSandbox`.
```

`OpenShellSandbox` does **not** own the lifecycle of the underlying OpenShell
sandbox. It accepts either an SDK-created sandbox/session or a session returned
from `SandboxClient.get_session("<name>")`.

Use a Python context manager when you want per-run isolation and automatic
cleanup. Use a named CLI-created sandbox when you need to set a specific image
or policy before Python attaches, or when you want to reuse the same sandbox
across multiple Deep Agent runs. Reused sandboxes keep filesystem and process
state until deleted, so they are best for demos, debugging, and workflows that
intentionally preserve state.

For OpenShell `0.0.72`, multiline shell commands are sent through shell stdin
instead of `bash -c` argv because the VM driver rejects newline-bearing command
argv payloads. File uploads are chunked to 512 KiB raw payloads to stay within
the observed stdin transport limit after base64 encoding.

## Notebook walkthrough

See [`docs/sandboxes/nvidia_openshell_sandbox.ipynb`](docs/sandboxes/nvidia_openshell_sandbox.ipynb)
for a step-by-step tutorial including local-only setup, `ChatNVIDIA` agent
integration, file transfer, and policy-enforced behavior demos.

## Compatibility

| Layer | Version |
|---|---|
| Python | 3.12 – 3.14 (the OpenShell SDK requires 3.12+) |
| `deepagents` | `>=0.5.0,<0.6.0` |
| `openshell` (NVIDIA SDK + CLI) | `>=0.0.68,<0.1` — both ship from the same `openshell` package; verify with `openshell --version` |
| `grpcio` | `>=1.78,<2`; OpenShell's wheel metadata currently declares `>=1.60`, but generated gRPC stubs require `>=1.78.0` at import time |
| LangChain Deep Agents `BaseSandbox` | follows `deepagents` |

## License

MIT.
