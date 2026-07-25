from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv

load_dotenv()

class GradeAnswer(BaseModel):
    """
    Grade the answer to the question.
    """
    binary_score: bool = Field(description="Answer addresses the question'yes' or 'no'.")

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL"), base_url=os.getenv("OPENAI_BASE_URL"), temperature=0)
structured_llm_answer_grader = llm.with_structured_output(GradeAnswer)

system = """
You are a grader assessing whether an LLM generation addresses the question.
Give a binary score 'yes' or 'no' to indicate if the generation addresses the question.
'yes' means the generation addresses the question, 'no' means it does not.
"""

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Question: {question} \n\n Generation: {generation}"),
])

answer_grader_chain : RunnableSequence = answer_prompt | structured_llm_answer_grader