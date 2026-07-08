import os
from dotenv import load_dotenv

from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


load_dotenv()

print("Initializing Components...")

embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0,
)

vectorstore = PineconeVectorStore(index_name=os.getenv("INDEX"), embedding=embeddings)

#Retriever will provide the search capabilities
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Top 3 relevant chunks

print("Loading Prompt Template...")
prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context: {context}
    If you don't know the answer, just say that you don't know. Do not try to make up an answer.
    Answer in the same language as the question.
    Question: {question}
    Provide a detailed answer: """ 
)

def extract_text(response) -> str:
    """Extract human-readable text from LangChain model response."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                txt = block.get("text", "")
                if txt:
                    parts.append(txt)
        return "\n".join(parts).strip()
    return str(content).strip()

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def retrieval_chain_without_lcel(query):
    """Retrieval chain without LCEL"""
    docs = retriever.invoke(query)
    context = format_docs(docs)

    messages = prompt_template.format_messages(context=context, question=query)

    response = llm.invoke(messages)
    
    return extract_text(response)

def retrieval_chain_with_lcel():
    """Retrieval chain with LCEL"""
    retrieval_chain = (
        RunnablePassthrough.assign(
            context = itemgetter("question")| retriever | format_docs
        ) 
        | prompt_template 
        | llm 
        | StrOutputParser()
    )

    return retrieval_chain
    


if __name__ == "__main__":
    print("retrieving relevant chunks...")

    query = "What is vector database?"

    print("Retrieving without LCEL...")
    response_without_lcel = retrieval_chain_without_lcel(query)
    print("Response without LCEL:")
    print(response_without_lcel)

    print("Retrieving with LCEL...")
    chain_with_lcel = retrieval_chain_with_lcel()
    response_with_lcel = chain_with_lcel.invoke({"question": query})
    print("Response with LCEL:")
    print(response_with_lcel)