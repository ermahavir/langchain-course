from typing import List, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    State of the graph.

    Attributes:
        question: The question to answer.
        generation: LLM generation.
        web_search: whether to perform a web search.
        documents: list of documents
    """

    question: str
    generation: str
    web_search: bool
    documents: List[Document]
