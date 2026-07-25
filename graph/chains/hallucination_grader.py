from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field
from dotenv import load_dotenv

import os

load_dotenv()

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL"), base_url=os.getenv("OPENAI_BASE_URL"), temperature=0)

class GradeHallucination(BaseModel):
    """
    Grade the answer to the question for hallucination.
    """
    binary_score: bool = Field(description="Whether the answer is grounded in the facts 'yes' or 'no'.")

structured_llm_grader = llm.with_structured_output(GradeHallucination)

system = """ 
    You are a grader assessing whether an LLM generation is grounded in / supported by a set of documents
    Give a binary score 'yes' or 'no' to indicate if the generation is grounded in the documents.
    """

hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Set of facts: \n\n{documents} \n\n LLM Generation: \n\n{generation}"),
])

hallucination_chain = hallucination_prompt | structured_llm_grader