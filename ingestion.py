import asyncio
import os
import ssl
import certifi

from typing import Any, Dict, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma # Vector store for local storage
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from logger import log_info, log_success, log_error, log_warning, log_header, Colors
from dotenv import load_dotenv


load_dotenv(override=True)

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
    base_url="https://inference-api.nvidia.com/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
    show_progress_bar=True,
    chunk_size=50, # 50 langchain documents and text objects that we're going to embed at a single request.
    retry_min_seconds=0, # No retries
)

#vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
vectorstore = PineconeVectorStore(index_name=os.getenv("INDEX"), embedding=embeddings)

tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=100)
tavily_crawl = TavilyCrawl(max_depth=5, max_breadth=20, max_pages=100)

def chunk_urls(urls: List[str], chunk_size: int = 20) -> List[List[str]]:
    chunks = [] # List of lists of strings
    for i in range(0, len(urls), chunk_size):
        chunk = urls[i:i + chunk_size]
        chunks.append(chunk)

    # for (std::size_t i = 0; i < urls.size(); i += chunk_size) {
    #     std::size_t end = std::min(i + chunk_size, urls.size());
    #     std::vector<std::string> chunk(urls.begin() + i, urls.begin() + end);
    #     chunks.push_back(chunk);
    # }
    return chunks

async def extract_batch(urls: List[str], batch_number: int) -> Dict[str, Any]:
    try:
        log_info(f"Extracting batch number {batch_number} of {len(urls)} URLs", Colors.BLUE)
        docs = await tavily_extract.ainvoke(input={"urls": urls})
        log_success(
            f"Batch number {batch_number} completed with {len(docs.get('results', []))} documents"
        )
        return docs
    except Exception as e:
        log_error(f"Error extracting batch of {len(urls)} URLs: {e}", Colors.RED)
        return {"results": []}

async def async_extract(url_batches: List[List[str]]):
    log_header("Async Document Extraction")
    log_info(f"Extracting concurrent extraction of {len(url_batches)} batches of URLs", Colors.DARKCYAN)
    tasks = [extract_batch(batch, i) for i, batch in enumerate(url_batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_pages = []
    failed_batches = 0

    for result in results:
        if isinstance(result, Exception):
            failed_batches += 1
        else:
            for extracted_page in result.get("results", []):
                document = Document(
                    page_content=extracted_page.get("raw_content", ""),
                    metadata={"source": extracted_page.get("url")}
                )
                all_pages.append(document)

    log_success(f"Concurrent extraction completed with {len(all_pages)} documents, {failed_batches} failed")

    if failed_batches > 0:
        log_warning(f"Failed to extract {failed_batches} batches")

    return all_pages 

async def async_index_documents(documents: List[Document], batch_size: int = 50):
    log_header("Vector storage phase")

    log_info(f"Preparing to add {len(documents)} documents to vector store in batches of {batch_size}", Colors.DARKCYAN)

    batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]

    log_info(f"Found {len(batches)} batches to process", Colors.YELLOW)

    async def add_batch(batch: List[Document], batch_number: int):
        try:
            await vectorstore.aadd_documents(batch)
            log_success(f"Batch number {batch_number} added to vector store with {len(batch)} documents")
        except Exception as e:
            log_error(f"Error adding batch number {batch_number} to vector store: {e}", Colors.RED)
            return False
        return True

    # Process batches concurrently 
    # NOTE: vectorstore implements an async context manager (__aenter__ / __aexit__)
    # The internal async index/session (shared resource) is opened and closed once in these calls.
    #
    # async with vectorstore: keeps that session open for all concurrent batches,
    # then closes it once after they all finish.
    tasks = [add_batch(batch, i) for i, batch in enumerate(batches)]

    if hasattr(vectorstore, "__aenter__") and hasattr(vectorstore, "__aexit__"):
        async with vectorstore:
            results = await asyncio.gather(*tasks, return_exceptions=True)
    else:
        results = await asyncio.gather(*tasks, return_exceptions=True)


    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(f"Vector storage completed with {successful} successful batches")
    else:
        log_warning(f"Failed to add {len(batches) - successful} batches")

async def main():
    # log_header("Document Ingestion Pipeline")

    # log_info(" Tavily Crawler Starting to crawl documentation from https://python.langchain.com", Colors.PURPLE)

    # result = tavily_crawl.invoke({
    #     "url": "https://docs.langchain.com/oss/python/langchain/overview", #"https://python.langchain.com"
    #     "max_depth" : 1, # How far from base url we want to crawl
    #     "extract_depth" : "advanced", #Retrieve more data - tables, embedded content with higher success rate
    #     "instructions": "content on ai agent" # Which page to scrape - kind of filter
    # })

    # all_docs = [Document(page_content=result["raw_content"], metadata={"source": result["url"]}) for result in result["results"]]
    # log_success(f"Tavily Crawler Completed with {len(all_docs)} documents")

    log_header("Document Ingestion Pipeline")

    log_info("Tavil Map Starting to map documentation from https://python.langchain.com", Colors.PURPLE)

    site_map = tavily_map.invoke("https://python.langchain.com")

    chunks = chunk_urls(site_map["results"], 5)

    log_success(f"URL processing completed with {len(site_map['results'])} documents, URLs chunked into {len(chunks)} chunks")

    log_info("Starting concurrent document extraction", Colors.DARKCYAN)
    all_pages = await async_extract(chunks)
    log_success(f"Concurrent extraction completed with {len(all_pages)} documents") 

    log_info("Starting chunking of documents with 4000 chunk size and 200 overlap", Colors.YELLOW)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_pages)
    log_success(f"Created {len(splitted_docs)} chunks from {len(all_pages)} documents")

    await async_index_documents(splitted_docs, batch_size=10)

    log_header("Document Ingestion Pipeline Completed")
    print(f"URLs mapped: {len(site_map['results'])}")
    print(f"Documents extracted: {len(all_pages)}")
    print(f"Chunks created: {len(splitted_docs)}")

    
if __name__ == "__main__":
    asyncio.run(main())

