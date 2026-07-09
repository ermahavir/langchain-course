import os
import re

from typing import Any, Dict

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
        "If you cannot find the answer in the retrieved documentation, say so."
    )

    agent = create_agent(
        model=model,
        tools=[retrieve_context],
        system_prompt=system_prompt,
    )

    messages = [
        {"role":"user", "content": query}, 
    ]

    response = agent.invoke({"messages": messages})
    answer = response["messages"][-1].content

    #extract context doc from ToolMessage artifact
    context_docs = []
    for msg in response["messages"]:
        if isinstance(msg, ToolMessage) and hasattr(msg, "artifact"):
            if isinstance(msg.artifact, list):
                context_docs.extend(msg.artifact)
            elif isinstance(msg.artifact, dict):
                context_docs.append(msg.artifacts)

    return {
        "answer": answer,
        "context": context_docs,
    }

def print_answer_and_source(result: dict) -> None:
    answer_blocks = result.get("answer", [])
    for block in answer_blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text", "").strip()
            answer_text = text
            source = "N/A"
            source_label = "Source"

            # Handle inline "Source: <...>" format.
            if "Source:" in text:
                answer_text, source = text.rsplit("Source:", 1)
                answer_text = answer_text.strip()
                source = source.strip()
            else:
                # Handle markdown-style source links, e.g. [Source](https://...).
                match = re.search(r"\[Source\]\((https?://[^)]+)\)", text)
                if match:
                    source = match.group(1)
                    source_label = "Source (MD)"
                    # Remove markdown source links wherever they appear.
                    answer_text = re.sub(
                        r"\s*\[Source\]\(https?://[^)]+\)",
                        "",
                        text,
                    ).strip()

            print("Answer:")
            print(answer_text)
            # Clean up stray markdown punctuation around extracted source.
            source = source.strip().strip("[]()")
            # Normalize source to the first URL if the model includes prose/markdown.
            url_match = re.search(r"https?://[^\s)\]]+", source)
            if url_match:
                source = url_match.group(0).rstrip(".,;")
            print(f"\n{source_label}:")
            print(source)
            return
    print("No text answer block found.")

if __name__ == "__main__":
    query = "What is a deep agent in LangChain?"
    result = run_llm(query)
    print(result)
    #print_answer_and_source(result)
    
   

