from typing import List

from pydantic import BaseModel, Field

class Reflection(BaseModel):
    missing : str = Field(description="Crititque of what is missing?")
    superfluous : str = Field(description="Crititque of what is superfluous?")


class AnswerQuestion(BaseModel):
    answer : str = Field(description="~250 words answer to the user's question.")
    reflection : Reflection = Field(description="Reflection on the initial answer.")
    search_queries : List[str] = Field(description="1-3 Search queries to research information and improve the answer.")


class ReviseAnswer(AnswerQuestion):
    """Revise your previous answer using the new information.
    """
    references : List[str] = Field(description="Citations motivating your updated answer.")
    
