from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
import os

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
            "You are a viral tweeter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detailed recommentations, including requests for length, virality, style etc."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
            "You are a Tweeter techie influencer assistant tasked with writing excellent Tweeter posts."
            "Generate the best Tweeter posts possible for the user's request."
            "If the user provides critique, respond with a revised version of your previous attempts."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL"), base_url=os.getenv("OPENAI_BASE_URL"), temperature=0)

generate_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm

