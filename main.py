from dotenv import load_dotenv
import os

load_dotenv()


from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, START, END

from nodes import run_agent_reasoning, tool_node

AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1

def should_continue(state: MessagesState) -> str:
    """
    Check if the agent should continue reasoning.
    """
    if (state["messages"][LAST].tool_calls):
        return ACT
    return END

flow = StateGraph(MessagesState)
flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.set_entry_point(AGENT_REASON) # Tell langgraph that this is the starting point of the graph
flow.add_node(ACT, tool_node)

# From agent_reason to act (which is  the ToolNode) or end depedning on the 
# return value of should_continue
flow.add_conditional_edges(AGENT_REASON, should_continue, {
    END: END,
    ACT: ACT
})

# Always, go to the agent_reason node after the tool_node is executed
flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow.png")

def main():
    print("Hello ReAct LangGraph!")
    res = app.invoke({"messages": [HumanMessage(content="What is the temp in Paris in Celsius?List it and triple it")]})
    print(res["messages"][LAST].content)


if __name__ == "__main__":
    main()
