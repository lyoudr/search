from fastapi import APIRouter, Depends 

from app.schemas.req_res.medical import MedicalQuestionRequest
from app.agents.medical_agent.medical_agent import agent_executor

import uuid 

router = APIRouter(tags=["medical"], prefix="/medical")

@router.post(
    "/",
    summary="Ask medical question"
)
def ask_medical_question(
    payload: MedicalQuestionRequest
):
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    input_messages = {"role": "user", "content": payload.question}
    for step in agent_executor.stream(
        {"messages":[input_messages]}, config, stream_mode="values"
    ):
        step["messages"][-1].pretty_print()