from langchain_nvidia_ai_endpoints import ChatNVIDIA, Model, register_model

model = Model(
    id="meta/llama-3.1-8b-instruct",
    endpoint="https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/e62a4350-2218-4cf5-9262-112432d239f8",
)
register_model(model)

client = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")
print(client.invoke("hello"))
