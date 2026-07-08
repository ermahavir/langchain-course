import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv(override=True)

def main():
    print("Ingesting data...")
    loader = TextLoader("/users/mprasad/Desktop/vectorblog.txt")
    documents = loader.load()

    print("splitting documents...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print(f"Created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(
        # Use OpenAI-compatible embeddings through NVIDIA Inference Hub.
        model=os.getenv("EMBEDDING_MODEL"),
        base_url="https://inference-api.nvidia.com/v1",
    )

    print("Ingesting documents into Pinecone...")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.getenv("INDEX"))
    
    print("Done!")

if __name__ == "__main__":
    main()
