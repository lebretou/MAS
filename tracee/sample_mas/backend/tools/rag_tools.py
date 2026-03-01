"""simple rag helper for analysis planning."""

import os
import threading
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

_vector_store: FAISS | None = None
_vector_store_lock = threading.Lock()


def _build_knowledge_docs() -> list[Document]:
    return [
        Document(
            page_content=(
                "for numeric distribution analysis, use histogram, kde, and boxplot. "
                "when comparing category distributions, use countplot or grouped bar chart."
            ),
            metadata={"topic": "visualization"},
        ),
        Document(
            page_content=(
                "for correlation analysis, focus on numeric columns only. "
                "pearson correlation is common for linear relationships. "
                "use heatmaps with annotations for readability."
            ),
            metadata={"topic": "statistics"},
        ),
        Document(
            page_content=(
                "for data quality checks, inspect null counts, duplicate rows, and outliers. "
                "validate column existence before transformations to avoid runtime failures."
            ),
            metadata={"topic": "data_cleaning"},
        ),
        Document(
            page_content=(
                "for model-oriented analysis, split features and target explicitly, "
                "report baseline metrics, and include train-test split assumptions."
            ),
            metadata={"topic": "modeling"},
        ),
        Document(
            page_content=(
                "for regression analysis, check linearity assumptions, residual patterns, "
                "and multicollinearity. report r2 and error metrics clearly."
            ),
            metadata={"topic": "regression"},
        ),
    ]


def _get_vector_store() -> FAISS:
    global _vector_store
    if _vector_store is None:
        with _vector_store_lock:
            if _vector_store is None:
                if not os.getenv("OPENAI_API_KEY"):
                    raise ValueError("OPENAI_API_KEY must be set for RAG retrieval")
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                _vector_store = FAISS.from_documents(_build_knowledge_docs(), embeddings)
    return _vector_store


def retrieve_analysis_context(query: str, k: int = 3) -> str:
    """retrieve concise context snippets relevant to the query."""
    docs = _get_vector_store().similarity_search(query, k=k)
    if not docs:
        return ""

    lines = []
    for idx, doc in enumerate(docs, start=1):
        topic = doc.metadata.get("topic", "general")
        lines.append(f"{idx}. [{topic}] {doc.page_content}")
    return "\n".join(lines)


@tool
def retrieve_analysis_context_tool(query: str, k: int = 3) -> str:
    """retrieve analysis guidance snippets for planning."""
    try:
        return retrieve_analysis_context(query=query, k=k)
    except Exception as exc:
        return f"RAG context unavailable: {exc}"
