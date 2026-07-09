import os
from dotenv import load_dotenv

# from langchain_openai import OpenAIEmbeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from operator import itemgetter

load_dotenv(override=True)

def main():
    print("Hello from langchain-course!")
    print(os.getenv("OPENAI_API_KEY"))

if __name__ == "__main__":
    main()
