from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

def main():
    print(os.getenv("OPENAI_API_KEY"))
    print("Hello from langchain-course!")


if __name__ == "__main__":
    main()
