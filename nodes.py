from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from react import tools, llm

load_dotenv()

SYSTEM_MESSAGE = """
You are a helpful assistant that can use tools to answer questions. Always use the tools to answer the question.
"""

def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning workflow.
    """

    response = llm.invoke([{"role": "system", "content": SYSTEM_MESSAGE}, *state["messages"]])
    return {"messages": [response]}

# NOTE:
# MessagesState is defined as:
#   messages: Annotated[list[AnyMessage], add_messages]
#
# When any node returns {"messages": [new_msg, ...]}, LangGraph merges them via
# "add_messages" reducer: new messages are appended; if a message has the same id as an
# existing one, that entry is updated instead.
#
# ToolNode returns ToolMessages this way; run_agent_reasoning returns AIMessages
# the same way (see return above).
tool_node = ToolNode(tools)

