from dotenv import load_dotenv
import os

load_dotenv()

def main():
    print("Hello from langchain-course!")

if __name__ == "__main__":
    print(os.getenv("PINECONE_API_KEY"))
    main()
