from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
import os
load_dotenv()

@tool
def triple(number: int) -> int:
    """
    Args:
        number: The number to triple.
    Returns:
        The triple of the number.
    """
    return number * 3

tools = [TavilySearch(max_results=1), triple]

llm = ChatOpenAI(
    # OpenAI model exposed through Inference Hub (set to one you have access to)
    model=os.getenv("OPENAI_MODEL"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0,
    # Uncomment only if required by your org policy for sensitive data:
    # default_headers={"dataClassification": "sensitive"},
).bind_tools(tools)

