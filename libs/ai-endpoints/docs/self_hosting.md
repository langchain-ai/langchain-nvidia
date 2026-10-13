# Self-host NVIDIA NIM for LangChain

This guide takes an example that runs against the hosted
[NVIDIA API Catalog](https://build.nvidia.com) and runs it against NVIDIA NIM
microservices on your own GPU machine. The LangChain code stays the same: you
add `base_url=` (or set `NVIDIA_BASE_URL`) and keep using `ChatNVIDIA`,
`NVIDIAEmbeddings`, and `NVIDIARerank`.

| Role | Model ID | NIM container | Local `base_url` |
| --- | --- | --- | --- |
| Chat (single GPU) | `nvidia/nemotron-3-nano-30b-a3b` | `nvcr.io/nim/nvidia/nemotron-3-nano:2.1.4` | `http://localhost:8000/v1` |
| Chat (multi-GPU) | `nvidia/nemotron-3-super-120b-a12b` | `nvcr.io/nim/nvidia/nemotron-3-super-120b-a12b:2.1.4` | `http://localhost:8000/v1` |
| Embedding | `nvidia/llama-nemotron-embed-1b-v2` | `nvcr.io/nim/nvidia/llama-nemotron-embed-1b-v2:1.13.0` | `http://localhost:8080/v1` |
| Rerank | `nvidia/llama-nemotron-rerank-1b-v2` | `nvcr.io/nim/nvidia/llama-nemotron-rerank-1b-v2:1.10.0` | `http://localhost:2016/v1` |

Every NIM listens on port 8000 inside its container. The commands below
publish them on host ports 8000, 8080, and 2016, the same ports used in the
[README](../README.md). Tags are the
versions available on NGC when this guide was written. Check the NGC pages
linked below for newer tags.

## Prerequisites

- A Linux machine with NVIDIA GPUs that the NIM supports. See
  [Choose a chat NIM](#choose-a-chat-nim) and the support matrices below.
- An NVIDIA GPU driver supported by your NIM version. See the
  [NIM for LLMs documentation](https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html).
- [Docker](https://docs.docker.com/engine/install/) with the Compose plugin, and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  configured for Docker. Check that containers can see your GPUs with
  `docker run --rm --gpus all ubuntu nvidia-smi`.
- Enough disk space for model weights, which are cached under `LOCAL_NIM_CACHE`.
- `pip install -U langchain-nvidia-ai-endpoints` in the Python environment
  that runs your application.

## 1. Set up NGC access

NIM containers and model weights come from NGC. Create an API key in
[NGC API key setup](https://org.ngc.nvidia.com/account/api-keys), then put it in
your environment. Don't paste the key into commands that end up in your shell
history or in source code.

```bash
export NGC_API_KEY="<your NGC API key>"
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

The username is the literal string `$oauthtoken`. The single quotes stop the
shell from expanding it.

Pick a writable cache directory. All three NIMs share it, so weights are
downloaded only once per model and are reused across restarts:

```bash
export LOCAL_NIM_CACHE=~/.cache/nim
mkdir -p "$LOCAL_NIM_CACHE"
```

## Choose a chat NIM

`nvidia/nemotron-3-super-120b-a12b` is the recommended hosted chat model, but
its NIM does not fit on one GPU. Its
[NGC model card](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/models/nemotron-3-super-120b-a12b)
lists multi-GPU minimums, and NVIDIA's
[RAG Blueprint support matrix](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/docs/support-matrix.md)
runs its FP8 profile on two GPUs.

`nvidia/nemotron-3-nano-30b-a3b` (container `nemotron-3-nano`) is a smaller
current Nemotron chat model. Its
[NGC model card](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/models/nemotron-3-nano)
shows single-GPU (tensor parallel 1) serving and lists 80 GB A100 and H100
GPUs among its test hardware. It is the default in this guide.

The tradeoff: Nano uses 3.5B active parameters out of 30B, so it is cheaper
and faster to serve. Super uses 12B active parameters out of 120B and is the
stronger model for complex agentic and reasoning work. Both use the same
`ChatNVIDIA` code, so you can start on Nano and switch to Super by changing
the image, the GPUs, and the `model=` value.

Don't size GPUs from this page. Use the NIM documentation for the image tag
you deploy:

- [NIM for LLMs: supported models and GPUs](https://docs.nvidia.com/nim/large-language-models/latest/supported-models.html)
- [Text embedding NIM support matrix](https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/support-matrix.html)
- [Text reranking NIM support matrix](https://docs.nvidia.com/nim/nemo-retriever/text-reranking/latest/support-matrix.html)
- [NVIDIA NIM documentation](https://docs.nvidia.com/nim/)

## 2. Start the NIMs with `docker run`

Each command runs one NIM in the foreground. Use a separate terminal for each,
or replace `-it --rm` with `-d` to run it in the background. The `--gpus`
value pins each NIM to a GPU. The examples use GPU 0 for chat and GPU 1 for the
two small retrieval NIMs. Change the device numbers to match your machine.

Chat, Nemotron 3 Nano, on host port 8000
([NGC page](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/nemotron-3-nano)):

```bash
docker run -it --rm --name nim-chat \
  --gpus '"device=0"' \
  --shm-size=16GB \
  -e NGC_API_KEY \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -u "$(id -u)" \
  -p 8000:8000 \
  nvcr.io/nim/nvidia/nemotron-3-nano:2.1.4
```

Chat, Nemotron 3 Super, on host port 8000, if you have the GPUs for it
([NGC page](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/nemotron-3-super-120b-a12b)).
Run it instead of Nano, not alongside it, because both use port 8000:

```bash
docker run -it --rm --name nim-chat \
  --gpus all \
  --shm-size=16GB \
  -e NGC_API_KEY \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -u "$(id -u)" \
  -p 8000:8000 \
  nvcr.io/nim/nvidia/nemotron-3-super-120b-a12b:2.1.4
```

Embedding, on host port 8080
([NGC page](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/llama-nemotron-embed-1b-v2)):

```bash
docker run -it --rm --name nim-embed \
  --gpus '"device=1"' \
  --shm-size=16GB \
  -e NGC_API_KEY \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -u "$(id -u)" \
  -p 8080:8000 \
  nvcr.io/nim/nvidia/llama-nemotron-embed-1b-v2:1.13.0
```

Rerank, on host port 2016
([NGC page](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/llama-nemotron-rerank-1b-v2)):

```bash
docker run -it --rm --name nim-rerank \
  --gpus '"device=1"' \
  --shm-size=16GB \
  -e NGC_API_KEY \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -u "$(id -u)" \
  -p 2016:8000 \
  nvcr.io/nim/nvidia/llama-nemotron-rerank-1b-v2:1.10.0
```

The first start downloads weights into `LOCAL_NIM_CACHE` and can take a long
time. Later starts reuse the cache.

## 2 (alternative). Start all three with Docker Compose

[`self_hosting/docker-compose.yml`](self_hosting/docker-compose.yml) runs the
same chat, embedding, and rerank trio on ports 8000, 8080, and 2016. It
includes GPU reservations, the shared cache, `NGC_API_KEY` from your
environment, and health checks.

```bash
cd libs/ai-endpoints/docs/self_hosting
export USERID=$(id -u)
docker compose up -d
docker compose ps    # wait until all three show (healthy)
```

By default, chat uses GPU 0 and embedding and rerank share GPU 1. To change
that, set `CHAT_GPU_ID`, `EMBED_GPU_ID`, and `RERANK_GPU_ID`. To run a
different chat NIM, set `CHAT_IMAGE`. Nemotron 3 Super needs several GPUs, so
for Super also replace the `nim-chat` `device_ids` entry with your GPU list, or
with `count: all`.

Stop the stack with `docker compose down`. The cache directory is kept.

## 3. Check that each NIM is ready

```bash
curl -fsS localhost:8000/v1/health/ready   # chat
curl -fsS localhost:8080/v1/health/ready   # embedding
curl -fsS localhost:2016/v1/health/ready   # rerank

curl -fsS localhost:8000/v1/models         # model IDs served by the chat NIM
curl -fsS localhost:8080/v1/models
curl -fsS localhost:2016/v1/models
```

A NIM is ready when `/v1/health/ready` returns HTTP 200. Use the `id` values
from `/v1/models` as `model=` in LangChain.

You can also run Doctor against each NIM. It makes read-only `GET /v1/models`
requests and sends no inference request:

```bash
python -m langchain_nvidia_ai_endpoints doctor --base-url http://localhost:8000/v1 --model nvidia/nemotron-3-nano-30b-a3b --capability chat
python -m langchain_nvidia_ai_endpoints doctor --base-url http://localhost:8080/v1 --model nvidia/llama-nemotron-embed-1b-v2 --capability embeddings
python -m langchain_nvidia_ai_endpoints doctor --base-url http://localhost:2016/v1 --model nvidia/llama-nemotron-rerank-1b-v2 --capability ranking
```

## 4. Point LangChain at your NIMs

Hosted code:

```python
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank

llm = ChatNVIDIA(model="nvidia/nemotron-3-super-120b-a12b")
embedder = NVIDIAEmbeddings(model="nvidia/llama-nemotron-embed-1b-v2")
ranker = NVIDIARerank(model="nvidia/llama-nemotron-rerank-1b-v2")
```

The same code against your NIMs. Only `base_url` changes, plus the chat model
ID if you run Nano instead of Super:

```python
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank

llm = ChatNVIDIA(
    base_url="http://localhost:8000/v1",
    model="nvidia/nemotron-3-nano-30b-a3b",  # or nvidia/nemotron-3-super-120b-a12b
)
embedder = NVIDIAEmbeddings(
    base_url="http://localhost:8080/v1",
    model="nvidia/llama-nemotron-embed-1b-v2",
)
ranker = NVIDIARerank(
    base_url="http://localhost:2016/v1",
    model="nvidia/llama-nemotron-rerank-1b-v2",
)

print(llm.invoke("Write a haiku about GPUs.").content)
print(len(embedder.embed_query("What is NVIDIA NIM?")))
docs = [
    Document(page_content="NIM packages models as GPU-accelerated microservices."),
    Document(page_content="Bananas are rich in potassium."),
]
for doc in ranker.compress_documents(query="What is NIM?", documents=docs):
    print(doc.metadata["relevance_score"], doc.page_content)
```

A local NIM doesn't need `NVIDIA_API_KEY` unless you put an authenticating
proxy in front of it. To switch an existing app without changing code, set
`NVIDIA_BASE_URL` instead. That works when all clients point at the same NIM.
For separate chat, embedding, and rerank NIMs, pass `base_url=` to each client.

## Kubernetes

For Kubernetes deployments, use the NIM Operator or the NIM Helm charts
instead of `docker run`:

- [NVIDIA NIM Operator documentation](https://docs.nvidia.com/nim-operator/latest)
  ([source](https://github.com/NVIDIA/k8s-nim-operator))
- [`nim-llm` Helm chart on NGC](https://catalog.ngc.nvidia.com/orgs/nim/helm-charts/nim-llm)
- [NIM deployment examples](https://github.com/NVIDIA/nim-deploy) for managed
  Kubernetes services and KServe

Expose each NIM as a Service and use its URL as `base_url`, for example
`http://nim-chat.my-namespace.svc.cluster.local:8000/v1`.

## Troubleshooting

Start with Doctor. It runs without inference and reports what's wrong:

```bash
python -m langchain_nvidia_ai_endpoints doctor --base-url http://localhost:8000/v1 --model nvidia/nemotron-3-nano-30b-a3b --capability chat
```

| Doctor output | Likely cause | What to do |
| --- | --- | --- |
| `FAIL Cannot connect to /models` | The container isn't running, is still starting, or uses a different host port. | Check `docker ps` and `docker logs nim-chat`, and wait for `/v1/health/ready` to return 200. Check the `-p <host>:8000` mapping. |
| `FAIL /models timed out` | The NIM is busy loading the model, or a firewall drops the traffic. | Wait for readiness, or retry with `--timeout 30`. |
| `FAIL Requested model is absent from /models` | The `model=` value doesn't match the ID served by this NIM. | Use an `id` from `curl localhost:<port>/v1/models`. |
| `FAIL Model does not support chat` (or embeddings, ranking) | `base_url` points at the wrong NIM, such as the embedding port for chat. | Use port 8000 for chat, 8080 for embedding, and 2016 for rerank. |
| `FAIL Invalid base URL` | `base_url` includes an endpoint path such as `/chat/completions`. | Use the deployment root or `/v1`. |

Container problems that Doctor can't see:

- `unauthorized` or `denied` when pulling from `nvcr.io`: run `docker login nvcr.io` again with username `$oauthtoken` and a valid NGC API key, and check that your NGC account has access to the image.
- `could not select device driver "nvidia"` or no GPUs in the container: install and configure the NVIDIA Container Toolkit, then restart Docker.
- `Permission denied` under `/opt/nim/.cache`: the container user can't write to `LOCAL_NIM_CACHE`. Run with `-u "$(id -u)"` (or `USERID=$(id -u)` for Compose) on a directory you own.
- Out-of-memory errors or no compatible profile found at startup: the GPU is too small for that model or profile. Check the support matrix, use a larger GPU or more GPUs, or switch to the smaller chat NIM.

After a successful hosted check, Doctor also prints a `HINT Self-host` line.
It shows the NIM container and local `base_url` for known models and links
back to this guide.
