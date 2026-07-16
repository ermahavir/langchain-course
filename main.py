import os
from dotenv import load_dotenv

load_dotenv()

def main():
    print(os.getenv("OPENAI_API_KEY"))
    print("Hello from langgraph relection agent!")


if __name__ == "__main__":
    main()
