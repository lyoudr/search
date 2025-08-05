from langchain.agents import Tool

from app.agents.calendar_agent.calendar_tool import (
    check_availability_tool,
    book_calendar_tool
)
from app.agents import create_agent


check_availability_langchain_tool = Tool(
    name="CheckAvailability",
    func=check_availability_tool,
    description="Use for queries about available time slots. Input: 'YYYY-MM-DD'."
)

book_calendar_langchain_tool = Tool(
    name="BookCalendar",
    func=book_calendar_tool,
    description="Use for booking appointments. Input: 'YYYY-MM-DD HH:MM, Service name, Customer name'."
)


multi_tool_agent = create_agent(
    tools = [
        check_availability_langchain_tool,
        book_calendar_langchain_tool,
    ]
)