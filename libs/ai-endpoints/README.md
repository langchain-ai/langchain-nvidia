# NVIDIA NIMs

The `langchain-nvidia-ai-endpoints` package contains LangChain integrations for chat models and embeddings powered by [NVIDIA AI Foundation Models](https://www.nvidia.com/en-us/ai-data-science/foundation-models/), and hosted on [NVIDIA API Catalog.](https://build.nvidia.com/)

NVIDIA AI Foundation models are community and NVIDIA-built models and are NVIDIA-optimized to deliver the best performance on NVIDIA accelerated infrastructure.  Using the API, you can query live endpoints available on the NVIDIA API Catalog to get quick results from a DGX-hosted cloud compute environment. All models are source-accessible and can be deployed on your own compute cluster using NVIDIA NIM which is part of NVIDIA AI Enterprise.

Models can be exported from NVIDIA’s API catalog with NVIDIA NIM, which is included with the NVIDIA AI Enterprise license, and run them on-premises, giving Enterprises ownership of their customizations and full control of their IP and AI application. NIMs are packaged as container images on a per model/model family basis and are distributed as NGC container images through the NVIDIA NGC Catalog. At their core, NIMs are containers that provide interactive APIs for running inference on an AI Model. 

Below is an example on how to use some common functionality surrounding text-generative and embedding models.

## Installation

```python
%pip install -U --quiet langchain-nvidia-ai-endpoints
```

## Setup

**To get started:**
1. Create a free account with [NVIDIA](https://build.nvidia.com/), which hosts NVIDIA AI Foundation models.
2. Click on your model of choice.
3. Under Input select the Python tab, and click `Get API Key`. Then click `Generate Key`.
4. Copy and save the generated key as NVIDIA_API_KEY. From there, you should have access to the endpoints.

```python
import getpass
import os

if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    nvidia_api_key = getpass.getpass("Enter your NVIDIA API key: ")
    assert nvidia_api_key.startswith("nvapi-"), f"{nvidia_api_key[:5]}... is not a valid key"
    os.environ["NVIDIA_API_KEY"] = nvidia_api_key
```

## Working with NVIDIA API Catalog
```python
## Core LC Chat Interface
from langchain_nvidia_ai_endpoints import ChatNVIDIA

llm = ChatNVIDIA(model="meta/llama3-70b-instruct", max_tokens=419)
result = llm.invoke("Write a ballad about LangChain.")
print(result.content)
```

## Working with NVIDIA NIMs
When ready to deploy, you can self-host models with NVIDIA NIM—which is included with the NVIDIA AI Enterprise software license—and run them anywhere, giving you ownership of your customizations and full control of your intellectual property (IP) and AI applications.

[Learn more about NIMs](https://developer.nvidia.com/blog/nvidia-nim-offers-optimized-inference-microservices-for-deploying-ai-models-at-scale/)

```python
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank

# connect to an chat NIM running at localhost:8000, specifying a specific model
llm = ChatNVIDIA(base_url="http://localhost:8000/v1", model="meta-llama3-8b-instruct")

# connect to an embedding NIM running at localhost:8080
embedder = NVIDIAEmbeddings(base_url="http://localhost:8080/v1")

# connect to a reranking NIM running at localhost:2016
ranker = NVIDIARerank(base_url="http://localhost:2016/v1")
```

## Stream, Batch, and Async

These models natively support streaming, and as is the case with all LangChain LLMs they expose a batch method to handle concurrent requests, as well as async methods for invoke, stream, and batch. Below are a few examples.

```python
print(llm.batch(["What's 2*3?", "What's 2*6?"]))
# Or via the async API
# await llm.abatch(["What's 2*3?", "What's 2*6?"])
```

```python
for chunk in llm.stream("How far can a seagull fly in one day?"):
    # Show the token separations
    print(chunk.content, end="|")
```

```python
async for chunk in llm.astream("How long does it take for monarch butterflies to migrate?"):
    print(chunk.content, end="|")
```

## Supported models

Querying `available_models` will still give you all of the other models offered by your API credentials.

```python
[model.id for model in llm.available_models if model.model_type]

#[
# ...
# 'databricks/dbrx-instruct',
# 'google/codegemma-7b',
# 'google/gemma-2b',
# 'google/gemma-7b',
# 'google/recurrentgemma-2b',
# 'meta/codellama-70b',
# 'meta/llama2-70b',
# 'meta/llama3-70b-instruct',
# 'meta/llama3-8b-instruct',
# 'microsoft/phi-3-mini-128k-instruct',
# 'mistralai/mistral-7b-instruct-v0.2',
# 'mistralai/mistral-large',
# 'mistralai/mixtral-8x22b-instruct-v0.1',
# 'mistralai/mixtral-8x7b-instruct-v0.1',
# 'snowflake/arctic',
# ...
#]
```

## Model types

All of these models above are supported and can be accessed via `ChatNVIDIA`.

Some model types support unique prompting techniques and chat messages. We will review a few important ones below.

**To find out more about a specific model, please navigate to the NVIDIA NIM section of ai.nvidia.com [as linked here](https://docs.api.nvidia.com/nim/).**

### General Chat

Models such as `meta/llama3-8b-instruct` and `mistralai/mixtral-8x22b-instruct-v0.1` are good all-around models that you can use for with any LangChain chat messages. Example below.

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
    | ChatNVIDIA(model="meta/llama3-8b-instruct")
    | StrOutputParser()
)

for txt in chain.stream({"input": "What's your name?"}):
    print(txt, end="")
```

### Code Generation

These models accept the same arguments and input structure as regular chat models, but they tend to perform better on code-genreation and structured code tasks. An example of this is `meta/codellama-70b` and `google/codegemma-7b`.

```python
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert coding AI. Respond only in valid python; no narration whatsoever."),
        ("user", "{input}")
    ]
)
chain = (
    prompt
    | ChatNVIDIA(model="meta/codellama-70b", max_tokens=419)
    | StrOutputParser()
)

for txt in chain.stream({"input": "How do I solve this fizz buzz problem?"}):
    print(txt, end="")
```

## Multimodal

NVIDIA also supports multimodal inputs, meaning you can provide both images and text for the model to reason over.

An example model supporting multimodal inputs is `nvidia/neva-22b`.

These models accept LangChain's standard image formats. Below are examples.

```python
import requests

image_url = "https://picsum.photos/seed/kitten/300/200"
image_content = requests.get(image_url).content
```

Initialize the model like so:

```python
from langchain_nvidia_ai_endpoints import ChatNVIDIA

llm = ChatNVIDIA(model="nvidia/neva-22b")
```

#### Passing an image as a URL

```python
from langchain_core.messages import HumanMessage

llm.invoke(
    [
        HumanMessage(content=[
            {"type": "text", "text": "Describe this image:"},
            {"type": "image_url", "image_url": {"url": image_url}},
        ])
    ])
```

#### Passing an image as a base64 encoded string

```python
import base64
b64_string = base64.b64encode(image_content).decode('utf-8')
llm.invoke(
    [
        HumanMessage(content=[
            {"type": "text", "text": "Describe this image:"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_string}"}},
        ])
    ])
```

#### Directly within the string

The NVIDIA API uniquely accepts images as base64 images inlined within <img> HTML tags. While this isn't interoperable with other LLMs, you can directly prompt the model accordingly.

```python
base64_with_mime_type = f"data:image/png;base64,{b64_string}"
llm.invoke(
    f'What\'s in this image?\n<img src="{base64_with_mime_type}" />'
)
```

## Completions

You can also work with models that support the Completions API. These models accept a `prompt` instead of `messages`.

```python
completions_llm = NVIDIA().bind(max_tokens=512)
[model.id for model in completions_llm.get_available_models()]

# [
#   ...
#   'bigcode/starcoder2-7b',
#   'bigcode/starcoder2-15b',
#   ...
# ]
```

```python
prompt = "# Function that does quicksort written in Rust without comments:"
for chunk in completions_llm.stream(prompt):
    print(chunk, end="", flush=True)
```


## Embeddings

You can also connect to embeddings models through this package. Below is an example:

```python
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

embedder = NVIDIAEmbeddings(model="NV-Embed-QA")
embedder.embed_query("What's the temperature today?")
embedder.embed_documents([
    "The temperature is 42 degrees.",
    "Class is dismissed at 9 PM."
])
```
