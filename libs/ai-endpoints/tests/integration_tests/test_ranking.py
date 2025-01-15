import os
from typing import List

import faker
import pytest
from langchain_core.documents import Document

from langchain_nvidia_ai_endpoints import NVIDIARerank  # type: ignore


class CharacterTextSplitter:
    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size

    def create_documents(self, text: str) -> List[Document]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size):
            chunk = " ".join(words[i : i + self.chunk_size])
            chunks.append(Document(page_content=chunk))
        return chunks


@pytest.fixture
def text() -> str:
    fake = faker.Faker()
    fake.seed_instance(os.environ.get("FAKER_SEED", 13131))
    return fake.paragraph(2016)


@pytest.fixture
def query() -> str:
    return "what are human rights?"


@pytest.fixture
def splitter() -> CharacterTextSplitter:
    return CharacterTextSplitter(chunk_size=300)


@pytest.fixture
def documents(text: str, splitter: CharacterTextSplitter) -> List[Document]:
    return splitter.create_documents(text)


def test_langchain_reranker_direct(
    query: str, documents: List[Document], rerank_model: str, mode: dict
) -> None:
    ranker = NVIDIARerank(model=rerank_model, **mode)
    result_docs = ranker.compress_documents(documents=documents, query=query)
    assert len(result_docs) > 0
    for doc in result_docs:
        assert "relevance_score" in doc.metadata
        assert doc.metadata["relevance_score"] is not None
        assert isinstance(doc.metadata["relevance_score"], float)


def test_langchain_reranker_direct_empty_docs(
    query: str, rerank_model: str, mode: dict
) -> None:
    ranker = NVIDIARerank(model=rerank_model, **mode)
    result_docs = ranker.compress_documents(documents=[], query=query)
    assert len(result_docs) == 0


def test_langchain_reranker_direct_top_n_negative(
    query: str, documents: List[Document], rerank_model: str, mode: dict
) -> None:
    orig = NVIDIARerank.model_config["validate_assignment"]
    NVIDIARerank.model_config["validate_assignment"] = False
    ranker = NVIDIARerank(model=rerank_model, **mode)
    ranker.top_n = -100
    NVIDIARerank.model_config["validate_assignment"] = orig
    result_docs = ranker.compress_documents(documents=documents, query=query)
    assert len(result_docs) == 0


def test_langchain_reranker_direct_top_n_zero(
    query: str, documents: List[Document], rerank_model: str, mode: dict
) -> None:
    ranker = NVIDIARerank(model=rerank_model, **mode)
    ranker.top_n = 0
    result_docs = ranker.compress_documents(documents=documents, query=query)
    assert len(result_docs) == 0


def test_langchain_reranker_direct_top_n_one(
    query: str, documents: List[Document], rerank_model: str, mode: dict
) -> None:
    ranker = NVIDIARerank(model=rerank_model, **mode)
    ranker.top_n = 1
    result_docs = ranker.compress_documents(documents=documents, query=query)
    assert len(result_docs) == 1


def test_langchain_reranker_direct_top_n_equal_len_docs(
    query: str, documents: List[Document], rerank_model: str, mode: dict
) -> None:
    ranker = NVIDIARerank(model=rerank_model, **mode)
    ranker.top_n = len(documents)
    result_docs = ranker.compress_documents(documents=documents, query=query)
    assert len(result_docs) == len(documents)


def test_langchain_reranker_direct_top_n_greater_len_docs(
    query: str, documents: List[Document], rerank_model: str, mode: dict
) -> None:
    ranker = NVIDIARerank(model=rerank_model, **mode)
    ranker.top_n = len(documents) * 2
    result_docs = ranker.compress_documents(documents=documents, query=query)
    assert len(result_docs) == len(documents)


@pytest.mark.parametrize("batch_size", [-10, 0])
def test_rerank_invalid_max_batch_size(
    rerank_model: str, mode: dict, batch_size: int
) -> None:
    ranker = NVIDIARerank(model=rerank_model, **mode)
    with pytest.raises(ValueError):
        ranker.max_batch_size = batch_size


def test_rerank_invalid_top_n(rerank_model: str, mode: dict) -> None:
    ranker = NVIDIARerank(model=rerank_model, **mode)
    with pytest.raises(ValueError):
        ranker.top_n = -10


@pytest.mark.parametrize(
    "batch_size, top_n",
    [
        (7, 7),  # batch_size == top_n
        (17, 7),  # batch_size > top_n
        (3, 13),  # batch_size < top_n
        (1, 1),  # batch_size == top_n, corner case 1
        (1, 10),  # batch_size < top_n, corner case 1
        (10, 1),  # batch_size > top_n, corner case 1
    ],
)
def test_rerank_batching(
    query: str,
    documents: List[Document],
    rerank_model: str,
    mode: dict,
    batch_size: int,
    top_n: int,
) -> None:
    assert len(documents) > batch_size, "test requires more documents"

    ranker = NVIDIARerank(model=rerank_model, **mode)
    ranker.top_n = top_n
    ranker.max_batch_size = batch_size
    result_docs = ranker.compress_documents(documents=documents, query=query)
    assert len(result_docs) == min(len(documents), top_n)
    for doc in result_docs:
        assert "relevance_score" in doc.metadata
        assert doc.metadata["relevance_score"] is not None
        assert isinstance(doc.metadata["relevance_score"], float)
    assert all(
        result_docs[i].metadata["relevance_score"]
        >= result_docs[i + 1].metadata["relevance_score"]
        for i in range(len(result_docs) - 1)
    ), "results are not sorted"

    #
    # there's a bug in the service that causes the results to be inconsistent
    # depending on the batch shapes. running this test with FAKER_SEED=13131
    # will demonstrate the issue.
    #
    # reference_ranker = NVIDIARerank(
    #     model=rerank_model, max_batch_size=len(documents), top_n=len(documents)
    # ).mode(**mode)
    # reference_docs = reference_ranker.compress_documents(
    #     documents=[doc.copy(deep=True) for doc in documents], query=query
    # )
    # for i in range(top_n):
    #     assert result_docs[i].page_content == reference_docs[i].page_content
    # assert all(
    #     result_docs[i].page_content == reference_docs[i].page_content
    #     for i in range(top_n)
    # ), "batched results do not match unbatched results"


@pytest.mark.parametrize("truncate", ["END"])
def test_truncate_positive(rerank_model: str, mode: dict, truncate: str) -> None:
    query = "What is acceleration?"
    documents = [
        Document(page_content="NVIDIA " * length)
        for length in [32, 1024, 64, 128, 2048, 256, 512]
    ]
    client = NVIDIARerank(
        model=rerank_model, top_n=len(documents), truncate=truncate, **mode
    )
    response = client.compress_documents(documents=documents, query=query)
    assert len(response) == len(documents)


@pytest.mark.parametrize("truncate", [None, "NONE"])
def test_truncate_negative(rerank_model: str, mode: dict, truncate: str) -> None:
    if rerank_model == "nv-rerank-qa-mistral-4b:1":
        pytest.skip("nv-rerank-qa-mistral-4b:1 truncates by default")
    query = "What is acceleration?"
    documents = [
        Document(page_content="NVIDIA " * length)
        for length in [32, 1024, 64, 128, 10240, 256, 512]
    ]
    truncate_param = {}
    if truncate:
        truncate_param = {"truncate": truncate}
    client = NVIDIARerank(model=rerank_model, **truncate_param, **mode)
    with pytest.raises(Exception) as e:
        client.compress_documents(documents=documents, query=query)
    assert "400" in str(e.value)
    assert "exceeds maximum allowed" in str(e.value)
