import datetime
from dotenv import load_dotenv
import os
load_dotenv()

from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser, PydanticToolsParser

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_openai import ChatOpenAI

from schemas import AnswerQuestion, ReviseAnswer
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
                Current time: {time}

                1. {first_instruction}
                2. Reflect and critique your answer. Be severe to maximize improvement.
                3. Recommend search queries to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL"), base_url=os.getenv("OPENAI_BASE_URL"), temperature=0)
parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])


first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer."
)

first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion" # FORCE to use AnswerQuestion tool
)

revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

revisor = actor_prompt_template.partial(first_instruction=revise_instructions) | llm.bind_tools(
    tools=[ReviseAnswer], tool_choice="ReviseAnswer"
)

if __name__ == "__main__":
    human_message = HumanMessage(content="Write about the AI powered SOC/Autonomous SOC problem domain"
    "List companies that do that and raised capital"
    )

    chain = (
        first_responder
        | parser_pydantic
    )

    response = chain.invoke({"messages": [human_message]})
    print(response)

