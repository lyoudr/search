
from langchain.agents import Tool

from app.agents.order_agent.order_tool import create_order_tool_with_db
from app.agents import create_agent

create_order_langchain_tool = Tool(
    name="CreateOrder",
    func=create_order_tool_with_db,
    description=(
        "Create a new order for a user."
        "You can provide input as natural language like 'Order for Alice: 1x Shampoo, 2x Conditioner'."
        "Example text: 'Order for Alice: 1x Shampoo, 2x Conditioner'."
    )
)

order_agent = create_agent(
    tools=[create_order_langchain_tool]
)