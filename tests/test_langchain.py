import faker
import pytest
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
from requests.exceptions import ConnectionError, MissingSchema

from ranking.langchain import Reranker  # type: ignore


@pytest.fixture
def text() -> str:
    return faker.Faker().paragraph(2016)


@pytest.fixture
def query() -> str:
    return "what are human rights?"


@pytest.fixture
def splitter() -> CharacterTextSplitter:
    return CharacterTextSplitter(chunk_size=300, chunk_overlap=50, separator=" ")


@pytest.fixture
def documents(text: str, splitter: CharacterTextSplitter) -> list[Document]:
    return splitter.create_documents([text])


@pytest.mark.requires_service
def test_langchain_reranker_direct(query: str, documents: list[Document]) -> None:
    ranker = Reranker()
    result_docs = ranker.compress_documents(documents=documents, query=query)
    assert len(result_docs) > 0
    for doc in result_docs:
        assert "relevance_score" in doc.metadata
        assert doc.metadata["relevance_score"] is not None
        assert type(doc.metadata["relevance_score"]) is float


def test_langchain_reranker_direct_empty_docs(query: str) -> None:
    ranker = Reranker()
    result_docs = ranker.compress_documents(documents=[], query=query)
    assert len(result_docs) == 0


def test_langchain_reranker_direct_top_n_negative(query: str, documents: list[Document]) -> None:
    ranker = Reranker()
    ranker.top_n = -100
    result_docs = ranker.compress_documents(documents=documents, query=query)
    assert len(result_docs) == 0


def test_langchain_reranker_direct_top_n_zero(query: str, documents: list[Document]) -> None:
    ranker = Reranker()
    ranker.top_n = 0
    result_docs = ranker.compress_documents(documents=documents, query=query)
    assert len(result_docs) == 0


@pytest.mark.requires_service
def test_langchain_reranker_direct_top_n_one(query: str, documents: list[Document]) -> None:
    ranker = Reranker()
    ranker.top_n = 1
    result_docs = ranker.compress_documents(documents=documents, query=query)
    assert len(result_docs) == 1


@pytest.mark.requires_service
def test_langchain_reranker_direct_top_n_equal_len_docs(
    query: str, documents: list[Document]
) -> None:
    ranker = Reranker()
    ranker.top_n = len(documents)
    result_docs = ranker.compress_documents(documents=documents, query=query)
    assert len(result_docs) == len(documents)


@pytest.mark.requires_service
def test_langchain_reranker_direct_top_n_greater_len_docs(
    query: str, documents: list[Document]
) -> None:
    ranker = Reranker()
    ranker.top_n = len(documents) * 2
    result_docs = ranker.compress_documents(documents=documents, query=query)
    assert len(result_docs) == len(documents)


def test_langchain_reranker_direct_endpoint_bogus(query: str, documents: list[Document]) -> None:
    ranker = Reranker()
    with pytest.raises(MissingSchema):
        ranker.endpoint = "bogus"
        ranker.compress_documents(documents=documents, query=query)


def test_langchain_reranker_direct_endpoint_unavailable(
    query: str, documents: list[Document]
) -> None:
    ranker = Reranker()
    with pytest.raises(ConnectionError):
        ranker.endpoint = "http://localhost:12321"
        ranker.compress_documents(documents=documents, query=query)
