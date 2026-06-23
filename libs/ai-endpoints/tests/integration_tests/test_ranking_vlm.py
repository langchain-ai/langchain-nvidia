import base64

import pytest
from langchain_core.documents import Document

from langchain_nvidia_ai_endpoints import NVIDIARerank


def _image_data_uri(image_path: str) -> str:
    image_type = image_path.rsplit(".", 1)[1]
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/{image_type};base64,{encoded}"


@pytest.mark.parametrize(
    "img",
    [
        _image_data_uri("tests/data/nvidia-picasso.jpg"),
        _image_data_uri("tests/data/nvidia-picasso.png"),
        _image_data_uri("tests/data/nvidia-picasso.webp"),
        _image_data_uri("tests/data/nvidia-picasso.gif"),
    ],
    ids=["jpg", "png", "webp", "gif"],
)
@pytest.mark.parametrize(
    "func", ["compress", "acompress"], ids=["compress", "acompress"]
)
@pytest.mark.asyncio
async def test_vlm_reranker_image_type(
    rerank_vlm_model: str, mode: dict, func: str, img: str
) -> None:
    ranker = NVIDIARerank(model=rerank_vlm_model, **mode)
    documents = [
        Document(
            page_content="A colorful cat with a frog and Times Square.",
            metadata={"image": img},
        ),
        Document(page_content="The weather today is sunny."),
    ]
    query = "Show me a picture of a cat or a frog"
    if func == "compress":
        results = ranker.compress_documents(documents=documents, query=query)
    else:
        results = await ranker.acompress_documents(documents=documents, query=query)
    assert len(results) > 0
    for doc in results:
        assert "relevance_score" in doc.metadata
        assert isinstance(doc.metadata["relevance_score"], float)


@pytest.mark.parametrize(
    "img",
    [
        _image_data_uri("tests/data/nvidia-picasso.jpg"),
        _image_data_uri("tests/data/nvidia-picasso.png"),
        _image_data_uri("tests/data/nvidia-picasso.webp"),
        _image_data_uri("tests/data/nvidia-picasso.gif"),
    ],
    ids=["jpg", "png", "webp", "gif"],
)
@pytest.mark.parametrize(
    "func", ["compress", "acompress"], ids=["compress", "acompress"]
)
@pytest.mark.asyncio
async def test_vlm_reranker_image_only(
    rerank_vlm_model: str, mode: dict, func: str, img: str
) -> None:
    """VLM reranker should work with image-only passages (empty text)."""
    ranker = NVIDIARerank(model=rerank_vlm_model, **mode)
    documents = [
        Document(
            page_content="",
            metadata={"image": img},
        ),
        Document(page_content="The weather today is sunny."),
    ]
    query = "Show me a picture of a cat or a frog"
    if func == "compress":
        results = ranker.compress_documents(documents=documents, query=query)
    else:
        results = await ranker.acompress_documents(documents=documents, query=query)
    assert len(results) > 0
    for doc in results:
        assert "relevance_score" in doc.metadata
        assert isinstance(doc.metadata["relevance_score"], float)
