from typing import Literal
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

import os
load_dotenv()

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource : Literal["vectorstore", "websearch"] = Field (..., 
    description="Given a user question chose to route it to a vectorstore or a websearch.")

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL"), base_url=os.getenv("OPENAI_BASE_URL"), temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)

system = """
You are an expert at routing user question to vectorstore or websearch.

The vectorstore contains documents related to agents, prompts engineering and adversial attacks.

Use the vectorstore for question on those topics. For everything else, use the websearch.

"""

route_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("user", "{question}")
])

question_router = route_prompt | structured_llm_router

