# 🦜️🔗 LangChain NVIDIA

## Packages

This repository contains two packages with NVIDIA integrations with LangChain:
- [langchain-nvidia-ai-endpoints](https://pypi.org/project/langchain-nvidia-ai-endpoints/) integrates [NVIDIA AI Foundation Models and Endpoints](https://www.nvidia.com/en-us/ai-data-science/foundation-models/).
- [langchain-nvidia-trt](https://pypi.org/project/langchain-nvidia-trt/) implements integrations of NVIDIA [TensorRT](https://developer.nvidia.com/tensorrt) models.

## Testing

### Cookbooks

See the notebooks in the [cookbook](./cookbook) directory for examples of using `ChatNVIDIA` and `NVIDIAEmbeddings` with LangGraph for agentic RAG and tool-calling agents.

### Studio

See the [studio](./studio) directory to test the agentic RAG workflow in LangGraph Studio.

Simply load the `studio` directory in [LangGraph Studio](https://github.com/langchain-ai/langgraph-studio?tab=readme-ov-file#download) and click the "Run" button with an input question.
