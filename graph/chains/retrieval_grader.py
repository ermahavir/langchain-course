# question and retrieved document
# determine 
import os
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL"), base_url=os.getenv("OPENAI_BASE_URL"), temperature=0)

class GradeDocuments(BaseModel):
    """
    Binary score of the retrieved documents.
    """
    binary_score: str = Field(description="Documents are relevant to the question: Yes or No")


structured_llm_grader = llm.with_structured_output(GradeDocuments)

system_prompt = """
You are a grader assessing the relevance of a retrieved document to a user question

If at lest 50% of the document contains keyword(s) or semantic meaning related to the question, 
grade it as relevant. Otherwise, grade it as not relevant.

Give a binary score: Yes or No to indicate if the document is relevant to the question.
"""

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Retrieved Document:\n\n {document} \n\n User Question:\n\n {question}"),
])

retrieval_grader = grade_prompt | structured_llm_grader 