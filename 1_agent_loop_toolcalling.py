from sys import flags
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL="qwen3:1.7b"

@tool
def get_product_price(product_name: str) -> float:
    """Look up the price of a product in the catalog"""
    print(f"Looking up the price of {product_name}")
    prices = {
        "laptop": 1.00,
        "phone": 0.50,
        "keyboard": 2.00,
    }
    return prices.get(product_name, 0.0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to a product based on the discount tier
    Available discount tiers are: gold, silver, bronze
    """
    print(f"Applying discount to {price} based on {discount_tier}")
    discounts = {
        "gold": 25,
        "silver": 10,
        "bronze": 5,
    }

    discount_percentage = discounts.get(discount_tier, 0.0) / 100
    return round(price * (1 - discount_percentage), 2)

@traceable(name = "Langchain Agent Loop")
def agent_loop(question: str) -> str:
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)# or
# llm = init_chat_model("ollama:qwen3:1.7b", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("="* 60)

    messages = [
        SystemMessage(content=
            "You are a helpful shopping assistant"
            "You have access to a product catalog tool to find the price of a product"
            "You have access to a discount tool\n\n"
            "STRICT RULES: you must follow these rules strictly\n\n"
            "1. Never guess or assume any product price."
            "You MUST call get_product_price tool to get the price of the product\n"
            "2. Never apply a discount without first calling apply_discount tool"
            "Pass the exact price returned by get_product_price tool to apply_discount tool"
            "Do NOT pass any other price or information to apply_discount tool"
            "3. Never calculate discount yourself using math. Always use apply_discount tool"
            "4. If the user doesn't specify a discount tier, ask user which tier to use. "
            "Do not assume a discount tier."
            "5. Before calling any tool, write out your internal thought process inside <think></think> tags."),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n== Iteration {iteration} ==")
        aimessage = llm_with_tools.invoke(messages)
        tool_calls = aimessage.tool_calls
        if not tool_calls:
            print(f"Final answer: {aimessage.content}")
            return aimessage.content

        #process only the first tool call: 1 per iteration
        if tool_calls:
            tool_call = tool_calls[0]
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]
            print(f"Tool call: {tool_name} with args: {tool_args}")
            tool_to_use = tools_dict.get(tool_name)
            if tool_to_use is None:
                print(f"Tool {tool_name} not found")
                raise ValueError(f"Tool {tool_name} not found")
            
            observation = tool_to_use.invoke(tool_args)
            print(f"Tool result: {observation}")

            messages.append(aimessage)
            messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))

    print("Max iterations reached. Error")
    return None

if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools)")
    result = agent_loop("What is the price of a laptop after applying a gold discount?")
    print(result)

# model = init_chat_model(
#     model="openai/openai/gpt-5.2-codex",
#     temperature=0,
#     api_key=os.getenv("OPENAI_API_KEY"),
#     base_url="https://inference-api.nvidia.com/v1",
# )

