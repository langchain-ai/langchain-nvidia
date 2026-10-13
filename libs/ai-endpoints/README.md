# LangChain NVIDIA AI Endpoints

LangChain chat, embedding, and reranking integrations for NVIDIA models, hosted on [build.nvidia.com](https://build.nvidia.com/) or self-hosted with [NVIDIA NIM](https://www.nvidia.com/en-us/ai-data-science/products/nim-microservices/).

```bash
pip install -U langchain-nvidia-ai-endpoints
```

Get a free API key at [build.nvidia.com](https://build.nvidia.com/) (profile icon, then **API Keys**), then:

```bash
export NVIDIA_API_KEY="nvapi-..."
```

```python
from langchain_nvidia_ai_endpoints import ChatNVIDIA
llm = ChatNVIDIA(model="nvidia/nemotron-3-super-120b-a12b")
print(llm.invoke("Write a haiku about GPUs.").content)
```

## Run it on your own GPU (NIM)

Start a NIM container for the model, then point the same code at it. A local NIM usually needs no API key.

```python
llm = ChatNVIDIA(base_url="http://localhost:8000/v1", model="nvidia/nemotron-3-super-120b-a12b")
print(llm.invoke("Write a haiku about GPUs.").content)
```

See the [self-hosting guide](https://github.com/langchain-ai/langchain-nvidia/blob/main/libs/ai-endpoints/docs/self_hosting.md) for container setup and the [advanced configuration](https://github.com/langchain-ai/langchain-nvidia/blob/main/libs/ai-endpoints/docs/advanced_configuration.md) page for base-URL rules and the Doctor preflight.

## Embeddings and reranking

```python
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, NVIDIARerank

embedder = NVIDIAEmbeddings(model="nvidia/llama-nemotron-embed-1b-v2")
vector = embedder.embed_query("What is NVIDIA NIM?")

ranker = NVIDIARerank(model="nvidia/llama-nemotron-rerank-1b-v2")
docs = [Document(page_content="NIM serves models as containers."), Document(page_content="Bananas are yellow.")]
print(ranker.compress_documents(query="What is NVIDIA NIM?", documents=docs)[0].page_content)
```

## Learn more

- [Advanced configuration](https://github.com/langchain-ai/langchain-nvidia/blob/main/libs/ai-endpoints/docs/advanced_configuration.md): base URLs, the Doctor preflight, multiple self-hosted NIMs, custom endpoints, and API key verification.
- [Self-hosting guide](https://github.com/langchain-ai/langchain-nvidia/blob/main/libs/ai-endpoints/docs/self_hosting.md): run models on your own GPU with NIM.
- [Adoption guide](https://github.com/langchain-ai/langchain-nvidia/blob/main/libs/ai-endpoints/docs/adoption_guide.md): hosted vs. self-hosted, streaming and tool calling, embeddings retrieval, and migrating from OpenAI-compatible clients.
- [Model registry](https://github.com/langchain-ai/langchain-nvidia/blob/main/libs/ai-endpoints/docs/model_registry.md): current and deprecated model IDs.
- [Cookbooks](https://github.com/langchain-ai/langchain-nvidia/blob/main/cookbook): LangGraph agentic RAG and tool-calling notebooks.
- [Telemetry & Privacy](https://github.com/langchain-ai/langchain-nvidia/blob/main/libs/ai-endpoints/README.md#telemetry--privacy): what is collected and how to opt out.

The rest of this page covers more features: streaming, batching, model discovery, multimodal input, completions, Dynamo, and retrieval.


## Use Stream, Batch, and Async

The models exposed by the NVIDIA API natively support streaming, and they expose a batch method to handle concurrent requests, as well as async methods for invoke, stream, and batch. 

The following examples demonstrate how to use batch and stream, and their async versions.


```python
from langchain_nvidia_ai_endpoints import ChatNVIDIA

llm = ChatNVIDIA(model="nvidia/nemotron-3-super-120b-a12b")

# Batch example
print(llm.batch(["What's 2*3?", "What's 2*6?"]))

# Batch example (async)
await llm.abatch(["What's 2*3?", "What's 2*6?"])

# Stream example
for chunk in llm.stream("How far can a seagull fly in one day?"):
    # Show the token separations
    print(chunk.content, end="|")

# Stream example (async)
async for chunk in llm.astream("How long does it take for monarch butterflies to migrate?"):
    print(chunk.content, end="|")
```


## Get a List of Supported Models

You can query `available_models` to get a list of the models that you can access with your API credentials. 
Use the following code.

```python
llm = ChatNVIDIA(model="nvidia/nemotron-3-super-120b-a12b")
[model.id for model in llm.available_models if model.model_type]
```

You should see output similar to the following.

```python
#[
# ...
# 'mistralai/mixtral-8x22b-instruct-v0.1',
# 'nvidia/llama-3.3-nemotron-super-49b-v1.5',
# 'nvidia/nemotron-3-super-120b-a12b',
# 'nvidia/nemotron-nano-12b-v2-vl',
# 'qwen/qwen2.5-coder-32b-instruct',
# ...
#]
```

> [!TIP]
> To find out more about a specific model, on the [Models page](https://build.nvidia.com/models), search for the name of the model (without the company and `/`), click the model, and then click **Model Card**.



## Work With Different Model Types

Some model types support unique prompting techniques and chat messages. 
Use this section to learn about a few examples.


### Nemotron

[Nemotron](https://www.nvidia.com/en-us/ai-data-science/foundation-models/nemotron/) is NVIDIA's open model family optimized for agentic AI. The models deliver strong reasoning and tool-calling performance with high throughput. `nvidia/nemotron-3-super-120b-a12b` is a recommended starting point: a powerful model with 12B active parameters and 120B parameters in total, with a 1M token context window, available on the NVIDIA API Catalog and as a self-hosted NIM.

```python
from langchain_nvidia_ai_endpoints import ChatNVIDIA

llm = ChatNVIDIA(model="nvidia/nemotron-3-super-120b-a12b")
result = llm.invoke("Plan a three-step agentic workflow for competitive research.")
print(result.content)
```


### General Chat

Models such as `nvidia/nemotron-3-super-120b-a12b` and `mistralai/mixtral-8x22b-instruct-v0.1`
are good all-around models that you can use for any LangChain chat messages.

The following example generates a simple chat response.

```python
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant named Fred."),
        ("user", "{input}")
    ]
)
chain = (
    prompt
    | ChatNVIDIA(model="nvidia/nemotron-3-super-120b-a12b")
    | StrOutputParser()
)

for txt in chain.stream({"input": "What's your name?"}):
    print(txt, end="")
```


### Code Generation

Code generation models, such as `qwen/qwen2.5-coder-32b-instruct` and `mistralai/codestral-22b-instruct-v0.1`, 
tend to perform better on code-generation and structured code tasks. 

The following example generates python code to solve a problem.

```python
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert coding AI. Respond only in valid python; no narration whatsoever."),
        ("user", "{input}")
    ]
)
chain = (
    prompt
    | ChatNVIDIA(model="qwen/qwen2.5-coder-32b-instruct", max_completion_tokens=419)
    | StrOutputParser()
)

for txt in chain.stream({"input": "How do I solve this fizz buzz problem?"}):
    print(txt, end="")
```


### Multimodal Support

NVIDIA also has models that support multimodal inputs, such as `nvidia/nemotron-nano-12b-v2-vl`. 
You can provide both images and text for the model to reason over. 
These models accept LangChain's standard image formats. 
Local file paths are no longer accepted directly. If you are upgrading code that
passed local paths, read trusted local files explicitly and pass them as
`data:image/...;base64,...` URIs.

The following example asks the model to describe an image.

```python
import requests
import base64

# Initialize the image content
image_url = "https://picsum.photos/seed/kitten/300/200"
image_content = requests.get(image_url).content

# Initialize the model
from langchain_nvidia_ai_endpoints import ChatNVIDIA
llm = ChatNVIDIA(model="nvidia/nemotron-nano-12b-v2-vl")

# Pass an image as a URL
from langchain_core.messages import HumanMessage

llm.invoke(
    [
        HumanMessage(content=[
            {"type": "text", "text": "Describe this image:"},
            {"type": "image_url", "image_url": {"url": image_url}},
        ])
    ])

# Pass an image as a base64 encoded string
b64_string = base64.b64encode(image_content).decode('utf-8')
llm.invoke(
    [
        HumanMessage(content=[
            {"type": "text", "text": "Describe this image:"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_string}"}},
        ])
    ])
```


The NVIDIA API uniquely accepts images as base64 images within HTML `img` tags. 
While this isn't interoperable with other LLMs, you can prompt the model accordingly 
as shown in the following example.

```python
base64_with_mime_type = f"data:image/png;base64,{b64_string}"
llm.invoke(
    f"What's in this image?\n<img src=\"{base64_with_mime_type}\" />"
)
```


### Completions

You can work with models that support the Completions API. 
These models accept a `prompt` instead of `messages`.

The following example gets a list of models that support the Completions API.

```python
from langchain_nvidia_ai_endpoints import NVIDIA

completions_llm = NVIDIA().bind(max_tokens=512)
[model.id for model in completions_llm.get_available_models()]

# [
#   ...
# ]
```

The following example uses the Completions API to generate a code example.

```python
prompt = "# Function that does quicksort written in Rust without comments:"
for chunk in completions_llm.stream(prompt):
    print(chunk, end="", flush=True)
```


### Dynamo KV Cache Optimization

`ChatNVIDIADynamo` is a drop-in replacement for `ChatNVIDIA` that injects NVIDIA Dynamo KV cache routing hints into requests. Use it with Dynamo-enabled deployments to improve inference scheduling.

```python
from langchain_nvidia_ai_endpoints import ChatNVIDIADynamo

llm = ChatNVIDIADynamo(model="nvidia/nemotron-3-super-120b-a12b", osl=1024, iat=100)
result = llm.invoke("Write a short poem.")
```


### Embeddings in Detail

The following example connects to an embeddings model.

```python
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

embedder = NVIDIAEmbeddings(model="nvidia/llama-nemotron-embed-1b-v2")
embedder.embed_query("What's the temperature today?")
embedder.embed_documents([
    "The temperature is 42 degrees.",
    "Class is dismissed at 9 PM."
])
```

### Ranking in Detail

The following example connects to a ranking model.

```python
from langchain_nvidia_ai_endpoints import NVIDIARerank
from langchain_core.documents import Document

query = "What is the GPU memory bandwidth of H100 SXM?"
passages = [
    "The Hopper GPU is paired with the Grace CPU using NVIDIA's ultra-fast chip-to-chip interconnect, delivering 900GB/s of bandwidth, 7X faster than PCIe Gen5. This innovative design will deliver up to 30X higher aggregate system memory bandwidth to the GPU compared to today's fastest servers and up to 10X higher performance for applications running terabytes of data.",
    "A100 provides up to 20X higher performance over the prior generation and can be partitioned into seven GPU instances to dynamically adjust to shifting demands. The A100 80GB debuts the world's fastest memory bandwidth at over 2 terabytes per second (TB/s) to run the largest models and datasets.",
    "Accelerated servers with H100 deliver the compute power—along with 3 terabytes per second (TB/s) of memory bandwidth per GPU and scalability with NVLink and NVSwitch™.",
]

client = NVIDIARerank(model="nvidia/llama-nemotron-rerank-1b-v2")

response = client.compress_documents(
  query=query,
  documents=[Document(page_content=passage) for passage in passages]
)

print(f"Most relevant: {response[0].page_content}\nLeast relevant: {response[-1].page_content}")
```

Reranked results are separate `Document` objects with `metadata["relevance_score"]`.
The input documents and earlier result scores remain unchanged across subsequent
sync or async calls, so callers can safely reuse the same source documents.

### Ranking with a Vision-Language Rerank Model

`NVIDIARerank` can also rerank documents that carry an image alongside their text using a ranking VLM such as
[nvidia/llama-nemotron-rerank-vl-1b-v2](https://build.nvidia.com/nvidia/llama-nemotron-rerank-vl-1b-v2).

The following example connects to a ranking VLM.

```python
from langchain_nvidia_ai_endpoints import NVIDIARerank
from langchain_core.documents import Document

query = "Show me a picture of a cat"
image = "data:image/png;base64,<base64-encoded-image>"
documents = [
    Document(
        page_content="The picture of a cat.",
        metadata={"image": image},
    ),
    Document(
        page_content="",
        metadata={"image": image},
    ),
    Document(page_content="The weather today is sunny."),
]

ranker = NVIDIARerank(model="nvidia/llama-nemotron-rerank-vl-1b-v2")
response = ranker.compress_documents(query=query, documents=documents)
```

### Retrieval

The following example connects to a running NVIDIA RAG Blueprint server to retrieve relevant documents via the `/v1/search` endpoint.

```python
from langchain_nvidia_ai_endpoints import NVIDIARetriever

retriever = NVIDIARetriever(base_url="http://localhost:8081", k=4)
docs = retriever.invoke("What is machine learning?")
```


## Self-host with NVIDIA NIM Microservices

For connecting chat, embedding, and reranking clients to several local NIM
microservices, and for deployments that expose a full inference URL, see
[Self-host with NVIDIA NIM Microservices](https://github.com/langchain-ai/langchain-nvidia/blob/main/libs/ai-endpoints/docs/advanced_configuration.md#self-host-with-nvidia-nim-microservices).

---

## Telemetry & Privacy

`langchain-nvidia-ai-endpoints` includes content-free, aggregate telemetry to
help NVIDIA understand adoption and reliability of supported NVIDIA NIM
integrations. Telemetry is enabled by default for supported NVIDIA-hosted NIM
endpoint usage and can be disabled at any time.

Telemetry is limited to aggregate operational metrics such as connector and
framework version, operation type, recognized model family, request and outcome
counts, token counts, latency buckets, and coarse error categories. These totals
are best-effort and are not billing-grade.

The telemetry is aggregate-only and does not include request or response content,
credentials, endpoint URLs, or persistent user, device, or session identifiers.
It is not used to track individual users, for billing, or for precise capacity
accounting.

Telemetry is aggregated in memory and emitted only for supported NVIDIA-hosted
NIM endpoints. Self-hosted endpoints do not emit this telemetry.

You may opt out of telemetry collection at any time. Opting out applies only to
telemetry collection by the `langchain-nvidia-ai-endpoints` connector itself.
To disable telemetry for the current shell, set
`NVIDIA_USAGE_TELEMETRY_ENABLED=false` before running your application:

```bash
export NVIDIA_USAGE_TELEMETRY_ENABLED=false
```

You can also disable telemetry for a single client instance:

```python
llm = ChatNVIDIA(
    model="nvidia/nemotron-3-super-120b-a12b",
    usage_telemetry_enabled=False,
)
```

If telemetry is disabled, the connector does not send telemetry requests and
telemetry has no effect on connector operation.

**Use of third-party endpoints, including NVIDIA Build:** `langchain-nvidia-ai-endpoints`
can be configured to use various inference endpoints, including
[build.nvidia.com](https://build.nvidia.com/) (NVIDIA Build). If you choose to
use NVIDIA Build or any other third-party endpoint, that endpoint's own terms of
service and privacy practices apply independently of this library. Any opt-out
you exercise within `langchain-nvidia-ai-endpoints` does not extend to data
collection by your chosen endpoint. NVIDIA Build is intended for evaluation and
testing purposes only and may not be used in production environments. Do not
submit confidential information or personal data when using NVIDIA Build.
