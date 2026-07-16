# Annotated is going to help us add metadata to the type hints
from typing import Annotated, TypedDict

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import END, StateGraph
from langgraph.graph import add_messages

from chains import generate_chain, reflection_chain


class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def generation_node(state: MessageGraph) -> MessageGraph:
    response = generate_chain.invoke({"messages": state["messages"]})
    return {"messages": [response]}


def reflection_node(state: MessageGraph) -> MessageGraph:
    response = reflection_chain.invoke({"messages": state["messages"]})
    return { "messages": [HumanMessage(content=response.content)]} # Typecast AIMessage
