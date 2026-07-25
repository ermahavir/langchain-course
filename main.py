import asyncio
import os
from dotenv import load_dotenv

load_dotenv() 

async def main():
    print("Hello from langchain-course!")


if __name__ == "__main__":
    print(os.getenv("OPENAI_API_KEY"))
    asyncio.run(main())