from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader

import os

urls = [
#  "https://openai.com/blog/chatgpt",
#   "https://www.technologyreview.com/2023/02/07/1067928/what-is-generative-ai/",
#   "https://blog.research.google/2017/08/transformer-novel-neural-network.html",
#   "https://www.ibm.com/blog/what-is-generative-ai/",
#   "https://developer.nvidia.com/blog/improving-generative-ai-inference-efficiency/",
#   "https://hai.stanford.edu/news/how-large-language-models-will-transform-engineering",
#   "https://a16z.com/emerging-architectures-for-llm-applications/",
#   "https://huggingface.co/blog/rlhf",
#   "https://ai.meta.com/blog/large-language-model-llama-meta-ai/",
#   "https://www.anthropic.com/news/core-views-on-ai-safety",
  "https://lilianweng.github.io/posts/2023-06-23-agent/",
  "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
  "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
  "https://lilianweng.github.io/posts/2025-05-01-reasoning/",
  "https://lilianweng.github.io/posts/2024-04-12-video-diffusion/"
]

load_dotenv()

#Load URLs in Langchain Docs in loop
docs = [UnstructuredLoader(web_url=url, chunking_strategy="basic", max_characters=1000000).load() for url in urls]


# item is the value to put into the new list
# for sublist in docs — take each inner list from docs.
# for item in sublist — take each item from that inner list.
doc_list = [item for sublist in docs for item in sublist]

#Store chunks in Chroma
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=100);

doc_splits = text_splitter.split_documents(doc_list)

openai_embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
    base_url=os.getenv("OPENAI_API_BASE"),
    api_key=os.getenv("OPENAI_API_KEY"),
)
vectorstore = Chroma.from_documents(documents=doc_splits, collection_name="rag-chroma", 
    embedding=openai_embeddings, persist_directory="./.chroma_db")

retriever = Chroma(
    collection_name="rag-chroma",
    embedding_function=openai_embeddings,
    persist_directory="./.chroma_db"
)
