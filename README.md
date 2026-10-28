# 🦜️🔗 LangChain NVIDIA

LangChain integrations for NVIDIA models, including [Nemotron](https://www.nvidia.com/en-us/ai-data-science/foundation-models/nemotron/), served from [build.nvidia.com](https://build.nvidia.com/) or your own GPUs with [NVIDIA NIM](https://www.nvidia.com/en-us/ai-data-science/products/nim-microservices/).

## Quickstart

```bash
pip install -U langchain-nvidia-ai-endpoints
export NVIDIA_API_KEY="nvapi-..."  # free key from https://build.nvidia.com/
```

```python
from langchain_nvidia_ai_endpoints import ChatNVIDIA
llm = ChatNVIDIA(model="nvidia/nemotron-3-super-120b-a12b")
print(llm.invoke("Write a haiku about GPUs.").content)
```

## Hosted or self-hosted

| | Hosted (build.nvidia.com) | Self-hosted (NIM container) |
| --- | --- | --- |
| Where it runs | NVIDIA-hosted API Catalog | Your GPU, cloud, or cluster |
| `base_url` | Omit (defaults to `https://integrate.api.nvidia.com/v1`) | `base_url="http://localhost:8000/v1"` |
| Auth | `NVIDIA_API_KEY` required | Usually none; set `NVIDIA_API_KEY` if your deployment requires it |
| Guide | [Package README](./libs/ai-endpoints/README.md) | [Self-hosting guide](./libs/ai-endpoints/docs/self_hosting.md) |

See the [adoption guide](./libs/ai-endpoints/docs/adoption_guide.md) and [advanced configuration](./libs/ai-endpoints/docs/advanced_configuration.md) for base-URL rules, TLS, and the Doctor preflight.

## Packages

This repository contains packages with NVIDIA integrations with LangChain:
- [langchain-nvidia-ai-endpoints](https://pypi.org/project/langchain-nvidia-ai-endpoints/) integrates [NVIDIA AI Foundation Models and Endpoints](https://www.nvidia.com/en-us/ai-data-science/foundation-models/), including [Nemotron](https://www.nvidia.com/en-us/ai-data-science/foundation-models/nemotron/), NVIDIA's open model family built for agentic AI.
- [langchain-nvidia-trt](https://pypi.org/project/langchain-nvidia-trt/) implements integrations of NVIDIA [TensorRT](https://developer.nvidia.com/tensorrt) models.
- [langchain-nvidia-langgraph](./libs/langgraph/) provides NVIDIA-optimized LangGraph execution with parallel and speculative execution strategies.
- [langchain-nvidia-openshell](./libs/openshell/) connects [NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell) sandboxes to LangChain Deep Agents.
- [langchain-nvidia-switchyard](./libs/switchyard/) routes LangChain and Deep Agent model calls through NVIDIA Switchyard's `libsy` algorithms.

## Cookbooks

See the notebooks in the [cookbook](./cookbook) directory for examples of using `ChatNVIDIA` and `NVIDIAEmbeddings` with LangGraph for agentic RAG and tool-calling agents.

## Studio

See the [studio](./studio) directory to test the agentic RAG workflow in LangGraph Studio.

Simply load the `studio` directory in [LangGraph Studio](https://github.com/langchain-ai/langgraph-studio?tab=readme-ov-file#download) and click the "Run" button with an input question.

This will run agentic RAG where it first reflects on the question to decide whether to use web search or vectorstore retrieval. It also grades retrieved documents as well as generated answers.

## Debugging and deployment

For developing, debugging, and deploying AI agents and LLM applications, see [LangSmith](https://docs.langchain.com/langsmith/home).
