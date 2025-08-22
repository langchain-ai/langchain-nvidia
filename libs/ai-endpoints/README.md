# NVIDIA NIM Microservices

The `langchain-nvidia-ai-endpoints` package contains LangChain integrations for chat models and embeddings powered by [NVIDIA AI Foundation Models](https://www.nvidia.com/en-us/ai-data-science/foundation-models/), and hosted on the [NVIDIA API Catalog](https://build.nvidia.com/).

NVIDIA AI Foundation models are community- and NVIDIA-built models that are optimized to deliver the best performance on NVIDIA-accelerated infrastructure. 
You can use the API to query live endpoints that are available on the NVIDIA API Catalog to get quick results from a DGX-hosted cloud compute environment. 
or you can download models from NVIDIA's API catalog with NVIDIA NIM, which is included with the NVIDIA AI Enterprise license. 
The ability to run models on-premises gives your enterprise ownership of your customizations and full control of your IP and AI application. 

NIM microservices are packaged as container images on a per model/model family basis 
and are distributed as NGC container images through the [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/). 
At their core, NIM microservices are containers that provide interactive APIs for running inference on an AI Model. 

Use this documentation to learn how to install the `langchain-nvidia-ai-endpoints` package 
and use it for some common functionality for text-generative and embedding models.


## Get Started

### Install langchain-nvidia-ai-endpoints

To install the `langchain-nvidia-ai-endpoints` package, use the following code.

```python
%pip install -U --quiet langchain-nvidia-ai-endpoints
```


### Get Access to the NVIDIA API Catalog

To get access to the NVIDIA API Catalog, do the following:

1. Create a free account on the [NVIDIA API Catalog](https://build.nvidia.com/) and log in.
2. Click your profile icon, and then click **API Keys**. The **API Keys** page appears.
3. Click **Generate API Key**. The **Generate API Key** window appears.
4. Click **Generate Key**.  You should see **API Key Granted**, and your key appears.
5. Copy and save the key as `NVIDIA_API_KEY`.
6. To verify your key, use the following code.

    ```python
    import getpass
    import os

    if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
        nvidia_api_key = getpass.getpass("Enter your NVIDIA API key: ")
        assert nvidia_api_key.startswith("nvapi-"), f"{nvidia_api_key[:5]}... is not a valid key"
        os.environ["NVIDIA_API_KEY"] = nvidia_api_key
    ```

You can now use your key to access endpoints on the NVIDIA API Catalog.


## Invoke the Core Chat Interface

Use the following code to invoke the core chat interface.

```python
## Core LC Chat Interface
from langchain_nvidia_ai_endpoints import ChatNVIDIA

llm = ChatNVIDIA(model="meta/llama3-70b-instruct", max_tokens=419)
result = llm.invoke("Write a ballad about LangChain.")
print(result.content)
```


## Use Stream, Batch, and Async

The models exposed by the NVIDIA API natively support streaming, and they expose a batch method to handle concurrent requests, as well as async methods for invoke, stream, and batch. 

The following examples demonstrate how to use batch and stream, and their async versions.


```python
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
[model.id for model in llm.available_models if model.model_type]
```

You should see output similar to the following.

```python
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

> [!TIP]
> To find out more about a specific model, on the [Models page](https://build.nvidia.com/models), search for the name of the model (without the company and `/`), click the model, and then click **Model Card**.



## Work With Different Model Types

Some model types support unique prompting techniques and chat messages. 
Use this section to learn about a few examples.


### General Chat

Models such as `meta/llama3-8b-instruct` and `mistralai/mixtral-8x22b-instruct-v0.1` 
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
    | ChatNVIDIA(model="meta/llama3-8b-instruct")
    | StrOutputParser()
)

for txt in chain.stream({"input": "What's your name?"}):
    print(txt, end="")
```


### Code Generation

Code generation models, such as `meta/codellama-70b` and `google/codegemma-7b`, 
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
    | ChatNVIDIA(model="meta/codellama-70b", max_tokens=419)
    | StrOutputParser()
)

for txt in chain.stream({"input": "How do I solve this fizz buzz problem?"}):
    print(txt, end="")
```


### Multimodal Support

NVIDIA also has models that support multimodal inputs, such as `nvidia/neva-22b`. 
You can provide both images and text for the model to reason over. 
These models accept LangChain's standard image formats. 

The following example asks the model to describe an image.

```python
import requests
import base64

# Initialize the image content
image_url = "https://picsum.photos/seed/kitten/300/200"
image_content = requests.get(image_url).content

# Initialize the model
from langchain_nvidia_ai_endpoints import ChatNVIDIA
llm = ChatNVIDIA(model="nvidia/neva-22b")

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
    f'What\'s in this image?\n<img src="{base64_with_mime_type}" />'
)
```


### Completions

You can work with models that support the Completions API. 
These models accept a `prompt` instead of `messages`.

The following example gets a list of models that support the Completions API.

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

The following example uses the Completions API to generate a code example.

```python
prompt = "# Function that does quicksort written in Rust without comments:"
for chunk in completions_llm.stream(prompt):
    print(chunk, end="", flush=True)
```


### Embeddings

The following example connects to an embeddings model.

```python
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

embedder = NVIDIAEmbeddings(model="NV-Embed-QA")
embedder.embed_query("What's the temperature today?")
embedder.embed_documents([
    "The temperature is 42 degrees.",
    "Class is dismissed at 9 PM."
])
```

### Ranking

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

client = NVIDIARerank(model="nvidia/llama-3.2-nv-rerankqa-1b-v1")

response = client.compress_documents(
  query=query,
  documents=[Document(page_content=passage) for passage in passages]
)

print(f"Most relevant: {response[0].page_content}\nLeast relevant: {response[-1].page_content}")
```


## Self-host with NVIDIA NIM Microservices

When you are ready to deploy your AI application, you can self-host models with NVIDIA NIM. 
For more information, refer to [NVIDIA NIM Microservices](https://www.nvidia.com/en-us/ai-data-science/products/nim-microservices/).

The following code connects to locally hosted NIM Microservices.

```python
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank

# Connect to an chat NIM running at localhost:8000, and specify a model
llm = ChatNVIDIA(base_url="http://localhost:8000/v1", model="meta-llama3-8b-instruct")

# Connect to an embedding NIM running at localhost:8080
embedder = NVIDIAEmbeddings(base_url="http://localhost:8080/v1")

# Connect to a reranking NIM running at localhost:2016
ranker = NVIDIARerank(base_url="http://localhost:2016/v1")
```
