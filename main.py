# Annotated is going to help us add metadata to the type hints
from typing import Annotated, TypedDict

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import END, StateGraph
from langgraph.graph import add_messages

from chains import generate_chain, reflection_chain
from nodes import generation_node, reflection_node
from nodes import MessageGraph


REFLECT = "reflect"
GENERATE = "generate"
    

builder  = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.set_entry_point(GENERATE)
builder.add_node(REFLECT, reflection_node)

def should_continue(state: MessageGraph) -> str:
    if len(state["messages"]) > 6:
        return END
    return REFLECT

builder.add_conditional_edges(GENERATE, should_continue, {REFLECT: REFLECT, END: END})
builder.add_edge(REFLECT, GENERATE)


graph = builder.compile()
print(graph.get_graph().draw_mermaid_png(output_file_path="graph.png"))
graph.get_graph().print_ascii() #uv add grandalf 

def main():

    inputs = HumanMessage(content="""
        Make this tweet better: @LangchainAI Newly Tool calling is seriously underrated
        After a long wait, it is here - making implementation of agents across different models with function - calling
        super easy. Made a video covering their newest blog post.
    """)

    response =graph.invoke({"messages": [inputs]})
    

if __name__ == "__main__":
    main()
    