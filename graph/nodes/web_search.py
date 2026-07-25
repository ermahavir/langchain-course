from typing import Any, Dict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState

load_dotenv()

web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    print("--- WEB SEARCH NODE ---")
    question = state["question"]
    documents = state.get("documents") or []

    tavily_response = web_search_tool.invoke(question)
    tavily_results = tavily_response.get("results", [])

    joined_tavily_results = "\n\n".join(
        [tavily_result["content"] for tavily_result in tavily_results]
    )
    web_results = Document(page_content=joined_tavily_results)

    if documents is not None:
        documents = list(documents) + [web_results]
    else:
        documents = [web_results]

    return {"documents": documents, "question": question}


if __name__ == "__main__":
    print(web_search(state={"question": "agent memory", "documents": None}))
