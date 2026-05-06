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
                  │     backend=OpenShellSandbox│
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
                  │   Sandbox (Docker/VM/K8s)    │
                  │   policy-enforced execution  │
                  └──────────────────────────────┘
```

## Install

```bash
pip install langchain-nvidia-openshell
```

You also need [the OpenShell CLI / gateway](https://docs.nvidia.com/openshell/latest/get-started/quickstart/):

```bash
uv tool install -U openshell
# or
curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh | sh
```

## Quickstart (local)

Bring up a local sandbox first (this also boots a local gateway on `127.0.0.1:8080`):

```bash
openshell sandbox create --keep --no-tty -- bash
```

Then drive it from Python:

```python
import openshell
from langchain_nvidia_openshell import OpenShellSandbox

with openshell.Sandbox() as sb:
    backend = OpenShellSandbox(sandbox=sb)

    print(backend.execute("uname -a").output)
    backend.upload_files([("/sandbox/hello.py", b"print('hello from openshell')\n")])
    print(backend.execute("python3 /sandbox/hello.py").output)
    print(backend.download_files(["/sandbox/hello.py"])[0].content)
```

## Policies & security model

OpenShell enforces isolation across **four declarative policy layers**.
`OpenShellSandbox` is policy-agnostic — the sandbox you pass in already
carries its policy. Authoring and applying policy stays entirely in the
OpenShell CLI (no Python protobufs).

| Layer | What it controls | Enforced by | Set at create | Hot-reloadable |
|---|---|---|---|---|
| **Filesystem** | RO/RW mounts inside the sandbox | Linux Landlock LSM | ✅ | ❌ locked once running |
| **Network** | Outbound hosts, ports, HTTP methods/paths | In-sandbox HTTP CONNECT proxy + OPA/Rego | ✅ | ✅ via `openshell policy set` |
| **Process** | Run-as user/group, syscall filter | seccomp BPF + dropped privileges | ✅ | ❌ locked once running |
| **Inference** | Which LLM endpoints `inference.local` proxies to | Sandbox-local inference router | ✅ | ✅ via `openshell policy set` |

The stock `base` image ships a sensible default: RO `/usr /lib /etc
/app /var/log /proc/self /dev/urandom`, RW `/sandbox /tmp /dev/null`,
**default-deny outbound network**, restricted `sandbox` user, local
inference router. Run `openshell sandbox create -- bash` and you get all of
the above.

### Two CLI workflows for picking your own policy

**1. Set everything at create time** — the only path that can change the
locked Filesystem and Process layers:

```bash
openshell sandbox create \
  --policy ./example-policy.yaml \
  --keep --no-tty -- bash
```

**2. Hot-reload the running sandbox** — works for Network and Inference,
no recreate required:

```bash
openshell policy set <sandbox-name> --policy ./tighten.yaml --wait
```

A complete example policy covering all four layers ships in
[`docs/example-policy.yaml`](docs/example-policy.yaml). Copy, edit, point
the CLI at it. To change the locked Filesystem or Process defaults
permanently for a fleet, bake a custom `/etc/openshell/policy.yaml` into a
container image (see the OpenShell
[BYOC example](https://github.com/NVIDIA/OpenShell/tree/main/examples/bring-your-own-container)).

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
        model=ChatNVIDIA(model="meta/llama-3.3-70b-instruct"),
        system_prompt="You are a careful coding agent.",
        backend=backend,
    )
    print(agent.invoke({"messages": [{"role": "user",
        "content": "Compute the sha256 of '/etc/hostname' inside the sandbox."}]}))
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
        max_upload_chunk_bytes: int = 4 * (1 << 20),  # 4 MiB
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
sandbox. The user creates and disposes of it (via `with openshell.Sandbox()`,
or by calling `sandbox.delete()` directly). This matches the convention of
the Daytona, Modal, and Runloop integrations.

## Notebook walkthrough

See [`docs/nvidia_openshell_sandbox.ipynb`](docs/nvidia_openshell_sandbox.ipynb)
for a step-by-step tutorial including local-only setup, `ChatNVIDIA` agent
integration, file transfer, and policy-enforced behavior demos.

## Compatibility

| Layer | Version |
|---|---|
| Python | 3.12 – 3.13 (the OpenShell SDK requires 3.12+) |
| `deepagents` | `>=0.5.0,<0.6.0` |
| `openshell` (NVIDIA SDK) | `>=0.0.36,<0.1` (alpha; pinned tightly) |
| LangChain Deep Agents `BaseSandbox` | follows `deepagents` |

## License

MIT.
