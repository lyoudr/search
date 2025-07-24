from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver

from app.agents.medical_agent.pinecone_tool import vector_search_tool

model = init_chat_model("openai:gpt-4")
memory = MemorySaver() # As mentioned earlier, this agent is stateless. This means it does not remember previous interactions. To give it memory we need to pass in a checkpointer.
tools = [vector_search_tool]

agent_executor = create_react_agent(
    model=model,
    tools=tools,
    checkpointer=memory
)

