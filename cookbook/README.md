# Cookbook

Runnable notebooks that use NVIDIA NIM microservices through
[`langchain-nvidia-ai-endpoints`](../libs/ai-endpoints). They default to the
`nvidia/nemotron-3-super-120b-a12b` chat model, `nvidia/nemotron-3-embed-1b` embeddings, and
`nvidia/llama-nemotron-rerank-vl-1b-v2` reranking.

| Notebook | Description | Colab |
| --- | --- | --- |
| [nemotron_tool_calling_agent.ipynb](nemotron_tool_calling_agent.ipynb) | Tool-calling agent with LangChain v1 `create_agent`, `ChatNVIDIA`, and `init_chat_model("nvidia:...")`. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-nvidia/blob/main/cookbook/nemotron_tool_calling_agent.ipynb) |
| [nemotron_deep_agent_nim.ipynb](nemotron_deep_agent_nim.ipynb) | Deep Agent (`deepagents`) on Nemotron that plans, uses a local filesystem, and calls a calculator. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-nvidia/blob/main/cookbook/nemotron_deep_agent_nim.ipynb) |
| [langgraph_rag_agent_nvidia_nim.ipynb](langgraph_rag_agent_nvidia_nim.ipynb) | Adaptive RAG agent in LangGraph with NVIDIA embeddings, reranking, and Nemotron grading. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-nvidia/blob/main/cookbook/langgraph_rag_agent_nvidia_nim.ipynb) |
| [nvidia_nim_agents.ipynb](nvidia_nim_agents.ipynb) | Tool-calling agent built from LangGraph primitives (`StateGraph`, `ToolNode`, `tools_condition`). | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-nvidia/blob/main/cookbook/nvidia_nim_agents.ipynb) |

The tools in these notebooks run offline, so the only key you need is an NVIDIA API key for hosted endpoints.

## Hosted or self-hosted

Each notebook has one configuration cell near the top. Every model client in the notebook uses it.

- **NVIDIA-hosted (default):** leave `NVIDIA_BASE_URL` unset and set `NVIDIA_API_KEY` (get a key from
  [build.nvidia.com](https://build.nvidia.com)). The notebook prompts for the key if it is not set.
- **Self-hosted NIM:** point `NVIDIA_BASE_URL` at your NIM's OpenAI-compatible endpoint. No API key is needed.

```bash
# Hosted
export NVIDIA_API_KEY=nvapi-...

# Self-hosted
export NVIDIA_BASE_URL=http://localhost:8000/v1
```

To use a different model, set `NVIDIA_CHAT_MODEL` (and `NVIDIA_EMBED_MODEL` / `NVIDIA_RERANK_MODEL` for the RAG
notebook) before starting the notebook. Pick IDs that are not deprecated in the
[model registry](../libs/ai-endpoints/docs/model_registry.md).

## Related packages

- [`langchain-nvidia-openshell`](../libs/openshell): `OpenShellSandbox`, a policy-enforced sandbox backend for Deep Agents.
- [`langchain-nvidia-switchyard`](../libs/switchyard): `SwitchyardRoutingMiddleware`, model routing middleware for agents.

## Archive

[archive/](archive) holds notebooks from deprecated NVIDIA blueprints, kept for reference only. They are not maintained or
tested: [structured_report_generation.ipynb](archive/structured_report_generation.ipynb) and
[structured_report_generation_elastic.ipynb](archive/structured_report_generation_elastic/structured_report_generation_elastic.ipynb).
