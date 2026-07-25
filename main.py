import asyncio
import os
from dotenv import load_dotenv

# ClientSession is framework which allows a python file to act as an MCP client
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv() 

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL"), base_url=os.getenv("OPENAI_BASE_URL"), temperature=1)


stdio_server_params = StdioServerParameters(
    command="python",
    args=["/Users/mprasad/Desktop/courses/langchain-course/servers/math_server.py"],
)

async def main():
    async with stdio_client(stdio_server_params) as (read, write):
        async with ClientSession(read_stream=read, write_stream=write) as session:
            await session.initialize()
            print("Initialized session")
            tools = await load_mcp_tools(session) #await session.list_tools()
            #print(tools)

            agent = create_agent(llm, tools)

            system_message = SystemMessage(content=
                "You are a helpful assistant that can answer questions with the help of given tools. "
                "Always use provided tools to answer the questions.")

            result = await agent.ainvoke({"messages": [system_message, HumanMessage(content="What is 50 + 5 * 3?")]})
            print(result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())