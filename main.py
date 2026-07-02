from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

# from tavily import TavilyClient

# tavily = TavilyClient()

# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches the web for information
#     Args:
#         query: The query to search the web for
#     Returns:
#         The search result
#     """
#     print(f"""Search the web for information {query}""")
#     return tavily.search(query)

llm = ChatOpenAI( model="openai/openai/gpt-5.2-codex",
        base_url="https://inference-api.nvidia.com/v1",
        temperature=0)
#tools = [search]
tools = [TavilySearch(max_results=5)]

agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": HumanMessage(content="Search for 3 job postings for an AI engineer using LangChain in the bay area on linkedin and list their details. Broaden search with other keywords if the job listings are less than 3 ")})
    print(result)

if __name__ == "__main__":
    main()
