import os
import re

from typing import Any, Dict, List

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
    base_url="https://inference-api.nvidia.com/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
    show_progress_bar=False,
    chunk_size=50, # 50 langchain documents and text objects that we're going to embed at a single request.
    retry_min_seconds=0, # No retries
)

vectorstore = PineconeVectorStore(index_name=os.getenv("INDEX"), embedding=embeddings)

model = init_chat_model(
    model=os.getenv("OPENAI_MODEL"), model_provider="openai",
    base_url="https://inference-api.nvidia.com/v1",
    temperature=0.0,
)

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about LangChain."""
    print(f"Retrieving context for query: {query}")
    retrieved_docs = vectorstore.as_retriever().invoke(query, k=5)

    # Serialize the documents.
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )

    # Return both serialized content and raw documents
    return serialized, retrieved_docs

def parse_answer_and_source(answer_payload: Any) -> Dict[str, str]:
    """Parse model output and extract clean answer + source URL."""
    # 1) Normalize payload -> text
    if isinstance(answer_payload, str):
        text = answer_payload.strip()
    elif isinstance(answer_payload, list):
        text_parts: List[str] = []
        for block in answer_payload:
            if isinstance(block, dict) and block.get("type") == "text":
                t = str(block.get("text", "")).strip()
                if t:
                    text_parts.append(t)
        text = "\n\n".join(text_parts).strip()
    elif isinstance(answer_payload, dict):
        text = str(answer_payload.get("text", "")).strip()
    else:
        text = str(answer_payload).strip()
    answer_text = text
    source = "N/A"
    source_label = "Source"
    # 2) Parse inline "Source:" / "Sources:" blocks.
    source_match = re.search(
        r"\bSources?\s*:\s*(.+)$",
        answer_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if source_match:
        source = source_match.group(1).strip()
        answer_text = answer_text[: source_match.start()].strip()

    # 3) Parse markdown source links anywhere in the answer
    md_match = re.search(r"\[Source[s]?\]\((https?://[^)\s]+)\)", answer_text, flags=re.IGNORECASE)
    if md_match:
        source = md_match.group(1)
        source_label = "Source (MD)"
        answer_text = re.sub(
            r"\s*\[Source[s]?\]\(https?://[^)]+\)",
            "",
            answer_text,
            flags=re.IGNORECASE,
        ).strip()
    # 4) Normalize source to first URL if source contains prose/markdown
    url_match = re.search(r"https?://[^\s)\]]+", source)
    if url_match:
        source = url_match.group(0).rstrip(".,;)]")
    # 5) Fallback: if still N/A, try extracting URL from full text
    if source == "N/A":
        fallback_url = re.search(r"https?://[^\s)\]]+", text)
        if fallback_url:
            source = fallback_url.group(0).rstrip(".,;)]")
    return {
        "answer": answer_text.strip(),
        "source": source,
        "source_label": source_label,
    }

def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.

    Args:
        query: The user's question.

    Returns:
        Dictionary containing:
            - answer: The generated answer.
            - context: List of retrieved documents.
    """
    # Create agent with retrieval tool.
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so. "
        "ALWAYS answer in the same language as the question. NEVER ever use a different language. "
        "Do NOT include inline Source/Sources text in the answer body."
    )

    agent = create_agent(
        model=model,
        tools=[retrieve_context],
        system_prompt=system_prompt,
    )

    messages = [
    {
        "role": "system",
        "content": "Reply strictly in the same language as the user's question. Do not mix languages.",
    },
    {"role": "user", "content": query}]
    response = agent.invoke({"messages": messages})

    # Parse final model answer (handles list/dict/string formats)
    answer_payload = response["messages"][-1].content if response.get("messages") else ""
    parsed_result = parse_answer_and_source(answer_payload)

    #extract context doc from ToolMessage artifact
    context_docs = []
    for msg in response["messages"]:
        if isinstance(msg, ToolMessage) and hasattr(msg, "artifact"):
            if isinstance(msg.artifact, list):
                context_docs.extend(msg.artifact)
            elif isinstance(msg.artifact, dict):
                context_docs.append(msg.artifact)

    return {
        "answer": parsed_result.get("answer", "").strip(),
        "context": context_docs,
    }

if __name__ == "__main__":
    query = "What is a deep agent in LangChain?"
    result = run_llm(query)
    print(result)
    #print_answer_and_source(result)
    
   

