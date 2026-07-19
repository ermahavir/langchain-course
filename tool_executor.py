from dotenv import load_dotenv
from typing import List
import os
load_dotenv()

from langchain_tavily import TavilySearch
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode

from schemas import AnswerQuestion, ReviseAnswer

tavily_tool = TavilySearch(max_results=5)


def run_queries(search_queries: List[str], **kwargs):
    """Run generated query"""
    return tavily_tool.batch([{"query" : query} for query in search_queries])


execute_tool = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__, 
            args_schema=AnswerQuestion),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__, 
            args_schema=ReviseAnswer),
    ]
)
