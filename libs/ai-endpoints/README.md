# langchain-nvidia-ai-endpoints

The `langchain-nvidia-ai-endpoints` package contains LangChain integrations for chat models and embeddings powered by the [NVIDIA AI Foundation Model](https://www.nvidia.com/en-us/ai-data-science/foundation-models/) playground environment. 

> [NVIDIA AI Foundation Endpoints](https://www.nvidia.com/en-us/ai-data-science/foundation-models/) give users easy access to hosted endpoints for generative AI models like Llama-2, SteerLM, Mistral, etc. Using the API, you can query live endpoints available on the [NVIDIA API Catalog](https://build.nvidia.com/) to get quick results from a DGX-hosted cloud compute environment. All models are source-accessible and can be deployed on your own compute cluster.

Below is an example on how to use some common functionality surrounding text-generative and embedding models

## Installation

```python
%pip install -U --quiet langchain-nvidia-ai-endpoints
```

## Setup

**To get started:**
1. Create a free account with [NVIDIA](https://build.nvidia.com/), which hosts NVIDIA AI Foundation models
2. Click on your model of choice
3. Under Input select the Python tab, and click Get API Key. Then click Generate Key
4. Copy and save the generated key as NVIDIA_API_KEY. From there, you should have access to the endpoints.

```python
import getpass
import os

if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    nvidia_api_key = getpass.getpass("Enter your NVIDIA API key: ")
    assert nvidia_api_key.startswith("nvapi-"), f"{nvidia_api_key[:5]}... is not a valid key"
    os.environ["NVIDIA_API_KEY"] = nvidia_api_key
```

```python
## Core LC Chat Interface
from langchain_nvidia_ai_endpoints import ChatNVIDIA

llm = ChatNVIDIA(model="meta/llama3-70b-instruct", max_tokens=419)
result = llm.invoke("Write a ballad about LangChain.")
print(result.content)
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

## Steering LLMs

> [SteerLM-optimized models](https://developer.nvidia.com/blog/announcing-steerlm-a-simple-and-practical-technique-to-customize-llms-during-inference/) supports "dynamic steering" of model outputs at inference time.

This lets you "control" the complexity, verbosity, and creativity of the model via integer labels on a scale from 0 to 9. Under the hood, these are passed as a special type of assistant message to the model.

The "steer" models support this type of input, such as `steerlm_llama_70b`

```python
from langchain_nvidia_ai_endpoints import ChatNVIDIA

llm = ChatNVIDIA(model="steerlm_llama_70b")
# Try making it uncreative and not verbose
complex_result = llm.invoke(
    "What's a PB&J?",
    labels={"creativity": 0, "complexity": 3, "verbosity": 0}
)
print("Un-creative\n")
print(complex_result.content)

# Try making it very creative and verbose
print("\n\nCreative\n")
creative_result = llm.invoke(
    "What's a PB&J?",
    labels={"creativity": 9, "complexity": 3, "verbosity": 9}
)
print(creative_result.content)
```

#### Use within LCEL

The labels are passed as invocation params. You can `bind` these to the LLM using the `bind` method on the LLM to include it within a declarative, functional chain. Below is an example.

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
    | ChatNVIDIA(model="steerlm_llama_70b").bind(labels={"creativity": 9, "complexity": 0, "verbosity": 9})
    | StrOutputParser()
)

for txt in chain.stream({"input": "Why is a PB&J?"}):
    print(txt, end="")
```

## Multimodal

NVIDIA also supports multimodal inputs, meaning you can provide both images and text for the model to reason over.

An example model supporting multimodal inputs is `ai-neva-22b`.

These models accept LangChain's standard image formats. Below are examples.

```python
import requests

image_url = "https://picsum.photos/seed/kitten/300/200"
image_content = requests.get(image_url).content
```

Initialize the model like so:

```python
from langchain_nvidia_ai_endpoints import ChatNVIDIA

llm = ChatNVIDIA(model="ai-neva-22b")
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

## RAG: Context models

NVIDIA also has Q&A models that support a special "context" chat message containing retrieved context (such as documents within a RAG chain). This is useful to avoid prompt-injecting the model.

**Note:** Only "user" (human) and "context" chat messages are supported for these models, not system or AI messages useful in conversational flows.

The `_qa_` models like `nemotron_qa_8b` support this.

```python
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import ChatMessage
prompt = ChatPromptTemplate.from_messages(
    [
        ChatMessage(role="context", content="Parrots and Cats have signed the peace accord."),
        ("user", "{input}")
    ]
)
llm = ChatNVIDIA(model="nemotron_qa_8b")
chain = (
    prompt
    | llm
    | StrOutputParser()
)
chain.invoke({"input": "What was signed?"})
```

## Embeddings

You can also connect to embeddings models through this package. Below is an example:

```python
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

embedder = NVIDIAEmbeddings(model="ai-embed-qa-4")
embedder.embed_query("What's the temperature today?")
embedder.embed_documents([
    "The temperature is 42 degrees.",
    "Class is dismissed at 9 PM."
])
```
