from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

import os

load_dotenv()

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL"), base_url=os.getenv("OPENAI_BASE_URL"), temperature=0)

prompt = ChatPromptTemplate.from_messages([
    (
        "human", 
        "You are an assistant for question-answering tasks. Use the following pieces of retrieved context"
        "to answer the question. If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n"
        "Question: {question}\n"
        "Context: {context}\n"
        "Answer: "
    ),
])

# We want the answer to be a single string without any other text or formatting.
class Generation(BaseModel):
    """
    Answer to the question.
    """
    answer: str = Field(description="Complete Answer to the question without any other text or formatting.")

structured_llm_generation = llm.with_structured_output(Generation)

generation_chain = prompt | structured_llm_generation