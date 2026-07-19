from dotenv import load_dotenv
from typing import Literal

from langchain_core.messages import ToolMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState


from chains import first_responder, revisor
from tool_executor import execute_tool

MAX_ITERATIONS = 2

def draft_node(state: MessagesState) -> MessagesState:
    response = first_responder.invoke(state["messages"])
    return {"messages": [response]}

def revise_node(state: MessagesState) -> MessagesState:
    response = revisor.invoke(state["messages"])
    return {"messages": [response]}

def event_loop(state: MessagesState) -> Literal["execute_tool", END]:
    count_total_visits = sum(isinstance(item, ToolMessage) for item in state["messages"])
    num_iterations = count_total_visits
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tool"


builder = StateGraph(MessagesState)
builder.add_node("draft", draft_node)
builder.add_node("execute_tool", execute_tool)
builder.add_node("revise", revise_node)

builder.add_edge(START, "draft")
builder.add_edge("draft", "execute_tool")
builder.add_edge("execute_tool", "revise")
builder.add_conditional_edges("revise", event_loop, ["execute_tool", END])
graph = builder.compile()

graph.get_graph().draw_mermaid_png(output_file_path="reflection_agent.png")


res = graph.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital.",
            }    
        ]
    }
)

last_message = res["messages"][-1]

if isinstance(last_message, AIMessage) and last_message.tool_calls:
    print(last_message.tool_calls[0]["args"]["answer"])

print(res)

# def main():
#     print("Hello from relexion agent!")

# if __name__ == "__main__":
#     main()
