import os
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferMemory

# Define LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

os.environ["OPENAI_API_KEY"] = ""

# Create memory (stores chat context)
memory = ConversationBufferMemory(memory_key="chat_history")

def create_agent(tools: list):
    """
    Create an agent with the specified tools.
    """
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent="zero-shot-react-description",
        memory=memory,
        verbose=True
    )