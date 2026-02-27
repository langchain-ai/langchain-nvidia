# 🦜️🔗 LangChain NVIDIA

## Packages

This repository contains packages with NVIDIA integrations with LangChain:
- [langchain-nvidia-ai-endpoints](https://pypi.org/project/langchain-nvidia-ai-endpoints/) integrates [NVIDIA AI Foundation Models and Endpoints](https://www.nvidia.com/en-us/ai-data-science/foundation-models/), including [Nemotron](https://www.nvidia.com/en-us/ai-data-science/foundation-models/nemotron/), NVIDIA's open model family built for agentic AI.
- [langchain-nvidia-trt](https://pypi.org/project/langchain-nvidia-trt/) implements integrations of NVIDIA [TensorRT](https://developer.nvidia.com/tensorrt) models.
- [langchain-nvidia-langgraph](./libs/langgraph/) provides NVIDIA-optimized LangGraph execution with parallel and speculative execution strategies.

## Testing

### Cookbooks

See the notebooks in the [cookbook](./cookbook) directory for examples of using `ChatNVIDIA` and `NVIDIAEmbeddings` with LangGraph for agentic RAG and tool-calling agents.

### Studio

See the [studio](./studio) directory to test the agentic RAG workflow in LangGraph Studio.

Simply load the `studio` directory in [LangGraph Studio](https://github.com/langchain-ai/langgraph-studio?tab=readme-ov-file#download) and click the "Run" button with an input question.

This will run agentic RAG where it first reflects on the question to decide whether to use web search or vectorstore retrieval. It also grades retrieved documents as well as generated answers.

![Screenshot 2024-12-04 at 11 19 54 AM](https://github.com/user-attachments/assets/736544ff-6597-4eb4-89d1-e1e5863baad4)
