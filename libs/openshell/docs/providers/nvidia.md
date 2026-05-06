# NVIDIA OpenShell

The `langchain-nvidia-openshell` package adapts
[NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell) — a policy-enforced,
sandboxed runtime for autonomous AI agents — to LangChain's
[`BaseSandbox`](https://docs.langchain.com/oss/python/deepagents/sandboxes)
contract. Your `ChatNVIDIA`-driven Deep Agent runs on the host while shell
commands, file uploads, and file downloads are dispatched into a hardened
OpenShell sandbox over gRPC. This is the **sandbox-as-tool** pattern.

OpenShell isolates each sandbox across four declarative policy layers
(Filesystem · Network · Process · Inference), enforced by Linux Landlock,
seccomp BPF, an in-sandbox HTTP CONNECT proxy backed by OPA/Rego, and a
local inference router. Authoring policy and managing sandbox lifecycle
both happen in the OpenShell CLI.

> ⚠️ **Alpha**. NVIDIA OpenShell is alpha software ("single-player mode"). The
> `openshell` SDK is pinned tightly (`>=0.0.36,<0.1`); expect breakage.



## Install the package

```python
pip install -U --quiet langchain-nvidia-openshell
```

You also need the [OpenShell CLI / gateway](https://docs.nvidia.com/openshell/latest/get-started/quickstart/),
which is not pip-installable:

```bash
uv tool install -U openshell
# or
curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh | sh
```

The first `openshell sandbox create` call boots a local gateway on
`127.0.0.1:8080` and writes mTLS material to `~/.config/openshell/`.



## Access the NVIDIA API Catalog

Your host-side agent uses `ChatNVIDIA`, so you need a
[NVIDIA API Catalog](https://build.nvidia.com/) key:

1. Create a free account on the [NVIDIA API Catalog](https://build.nvidia.com/) and log in.
2. Click your profile icon, and then click **API Keys**. The **API Keys** page appears.
3. Click **Generate API Key**. The **Generate API Key** window appears.
4. Click **Generate Key**. You should see **API Key Granted**, and your key appears.
5. Copy and save the key as `NVIDIA_API_KEY`.
6. To verify your key, use the following code.

   ```python
   import getpass
   import os

   if os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
       print("Valid NVIDIA_API_KEY already in environment. Delete to reset")
   else:
       nvapi_key = getpass.getpass("NVAPI Key (starts with nvapi-): ")
       assert nvapi_key.startswith(
           "nvapi-"
       ), f"{nvapi_key[:5]}... is not a valid key"
       os.environ["NVIDIA_API_KEY"] = nvapi_key
   ```



## Quickstart

Bring up a local sandbox first (the CLI also starts a local gateway):

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

`OpenShellSandbox` is **lifecycle-agnostic**: you bring an already-entered
`openshell.Sandbox` (or a `SandboxSession`) and the wrapper composes file ops
on top via `BaseSandbox`.



## Use with Deep Agents + ChatNVIDIA

```python
import openshell
from deepagents import create_deep_agent
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_nvidia_openshell import OpenShellSandbox

client = openshell.SandboxClient.from_active_cluster()
backend = OpenShellSandbox(sandbox=client.get_session("openshell-demo"))

agent = create_deep_agent(
    model=ChatNVIDIA(model="meta/llama-3.3-70b-instruct"),
    system_prompt="You are a careful coding agent.",
    backend=backend,
)
print(agent.invoke({"messages": [{"role": "user",
    "content": "Compute the sha256 of '/etc/hostname' inside the sandbox."}]}))
client.close()
```



## Policies in one paragraph

OpenShell sandboxes are governed by declarative YAML policies enforcing four
domains. Filesystem and Process layers are **locked at create time**;
Network and Inference are **hot-reloadable**:

```bash
openshell sandbox create --policy ./policy.yaml --keep --no-tty -- bash
openshell policy set <sandbox-name> --policy ./tighten.yaml --wait
```

A complete annotated policy ships at
[`sandboxes/example-policy.yaml`](../sandboxes/example-policy.yaml). For the
full schema see the
[OpenShell policy schema reference](https://docs.nvidia.com/openshell/latest/reference/policy-schema).



## Use the sandbox integration

The OpenShell sandbox notebook in this package walks through a complete
deny→allow demo with a `ChatNVIDIA`-driven Deep Agent:

- [sandboxes/nvidia_openshell_sandbox.ipynb](https://github.com/langchain-ai/langchain-nvidia/blob/main/libs/openshell/docs/sandboxes/nvidia_openshell_sandbox.ipynb)



## Related Topics

- [LangChain Deep Agents — sandboxes](https://docs.langchain.com/oss/python/deepagents/sandboxes) — the sandbox-as-tool pattern.
- [LangChain — Implementing a sandbox integration](https://docs.langchain.com/oss/python/contributing/implement-langchain) — `BaseSandbox` contract.
- [NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell) — runtime source.
- [OpenShell docs](https://docs.nvidia.com/openshell/latest/) — install, quickstart, security controls, observability.
- [OpenShell `sandbox-policy-quickstart`](https://github.com/NVIDIA/OpenShell/tree/main/examples/sandbox-policy-quickstart) — the upstream demo this package's notebook mirrors.
- [`langchain-nvidia-openshell` README](https://github.com/langchain-ai/langchain-nvidia/blob/main/libs/openshell/README.md) — full API surface and the four-layer policy summary.
