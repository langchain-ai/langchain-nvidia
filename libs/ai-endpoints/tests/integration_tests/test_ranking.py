import faker
import pytest
from langchain_core.documents import Document
from requests.exceptions import ConnectionError, MissingSchema

from langchain_nvidia_ai_endpoints import NVIDIARerank  # type: ignore


class CharacterTextSplitter:
    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size

    def create_documents(self, text: str) -> list[Document]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size):
            chunk = " ".join(words[i : i + self.chunk_size])
            chunks.append(Document(page_content=chunk))
        return chunks


@pytest.fixture
def text() -> str:
    return faker.Faker().paragraph(2016)


@pytest.fixture
def query() -> str:
    return "what are human rights?"


@pytest.fixture
def splitter() -> CharacterTextSplitter:
    return CharacterTextSplitter(chunk_size=300)


@pytest.fixture
def documents(text: str, splitter: CharacterTextSplitter) -> list[Document]:
    return splitter.create_documents(text)


def test_langchain_reranker_direct(
    query: str, documents: list[Document], rerank_model: str, mode: dict
) -> None:
    ranker = NVIDIARerank(model=rerank_model).mode(**mode)
    result_docs = ranker.compress_documents(documents=documents, query=query)
    assert len(result_docs) > 0
    for doc in result_docs:
        assert "relevance_score" in doc.metadata
        assert doc.metadata["relevance_score"] is not None
        assert isinstance(doc.metadata["relevance_score"], float)


def test_langchain_reranker_direct_empty_docs(
    query: str, rerank_model: str, mode: dict
) -> None:
    ranker = NVIDIARerank(model=rerank_model).mode(**mode)
    result_docs = ranker.compress_documents(documents=[], query=query)
    assert len(result_docs) == 0


def test_langchain_reranker_direct_top_n_negative(
    query: str, documents: list[Document], rerank_model: str, mode: dict
) -> None:
    ranker = NVIDIARerank(model=rerank_model).mode(**mode)
    ranker.top_n = -100
    result_docs = ranker.compress_documents(documents=documents, query=query)
    assert len(result_docs) == 0


def test_langchain_reranker_direct_top_n_zero(
    query: str, documents: list[Document], rerank_model: str, mode: dict
) -> None:
    ranker = NVIDIARerank(model=rerank_model).mode(**mode)
    ranker.top_n = 0
    result_docs = ranker.compress_documents(documents=documents, query=query)
    assert len(result_docs) == 0


def test_langchain_reranker_direct_top_n_one(
    query: str, documents: list[Document], rerank_model: str, mode: dict
) -> None:
    ranker = NVIDIARerank(model=rerank_model).mode(**mode)
    ranker.top_n = 1
    result_docs = ranker.compress_documents(documents=documents, query=query)
    assert len(result_docs) == 1


def test_langchain_reranker_direct_top_n_equal_len_docs(
    query: str, documents: list[Document], rerank_model: str, mode: dict
) -> None:
    ranker = NVIDIARerank(model=rerank_model).mode(**mode)
    ranker.top_n = len(documents)
    result_docs = ranker.compress_documents(documents=documents, query=query)
    assert len(result_docs) == len(documents)


def test_langchain_reranker_direct_top_n_greater_len_docs(
    query: str, documents: list[Document], rerank_model: str, mode: dict
) -> None:
    ranker = NVIDIARerank(model=rerank_model).mode(**mode)
    ranker.top_n = len(documents) * 2
    result_docs = ranker.compress_documents(documents=documents, query=query)
    assert len(result_docs) == len(documents)


def test_langchain_reranker_direct_endpoint_bogus(
    query: str, documents: list[Document]
) -> None:
    ranker = NVIDIARerank().mode(mode="nim", base_url="bogus")
    with pytest.raises(MissingSchema):
        ranker.compress_documents(documents=documents, query=query)


def test_langchain_reranker_direct_endpoint_unavailable(
    query: str, documents: list[Document]
) -> None:
    ranker = NVIDIARerank().mode(mode="nim", base_url="http://localhost:12321")
    with pytest.raises(ConnectionError):
        ranker.compress_documents(documents=documents, query=query)
