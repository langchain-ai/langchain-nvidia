"""
**LangChain NVIDIA AI Endpoints**

This comprehensive module integrates NVIDIA's state-of-the-art NIM endpoints,
featuring advanced models for conversational AI and semantic embeddings,
into the LangChain framework. It provides robust classes for seamless interaction
with AI models, particularly tailored for enriching conversational experiences
and enhancing semantic understanding in various applications.

**Features:**

1. **`ChatNVIDIA`:** This class serves as the primary interface for interacting
   with chat models. Users can effortlessly utilize advanced models like 'Nemotron'
   to engage in rich, context-aware conversations, applicable across diverse
   domains from customer support to interactive storytelling.

2. **`NVIDIAEmbeddings`:** The class offers capabilities to generate sophisticated
   embeddings using AI models. These embeddings are instrumental for tasks like
   semantic analysis, text similarity assessments, and contextual understanding,
   significantly enhancing the depth of NLP applications.

3. **`NVIDIARerank`:** This class provides an interface for reranking search results
    using AI models. Users can leverage this functionality to enhance search
    relevance and improve user experience in information retrieval systems.

4. **`NVIDIA`:** This class enables users to interact with large language models
    through a completions, or prompting, interface. Users can generate text
    completions, summaries, and other language model outputs using this class.
    This class is particularly useful for code generation tasks.

**Installation:**

Install this module easily using pip:

```python
pip install langchain-nvidia-ai-endpoints
```

After setting up the environment, interact with NIM endpoints -

## Utilizing chat models:

```python
from langchain_nvidia import ChatNVIDIA

llm = ChatNVIDIA(model="nvidia/llama-3.1-nemotron-51b-instruct")
response = llm.invoke("Tell me about the LangChain integration.")
```

## Generating semantic embeddings:

Create embeddings useful in various NLP tasks:

```python
from langchain_nvidia import NVIDIAEmbeddings

embedder = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")
embedding = embedder.embed_query("Exploring AI capabilities.")
```

## Code completion using large language models:

```python
from langchain_nvidia import NVIDIA

llm = NVIDIA(model="meta/codellama-70b")
completion = llm.invoke("def hello_world():")
```
"""  # noqa: E501

from langchain_nvidia_ai_endpoints import *  # noqa: F403
from langchain_nvidia_ai_endpoints import __all__  # noqa: F401
