from pydantic import BaseModel
from typing import Optional, List
from datetime import date, datetime


# UpcomingEvent schemas
class UpcomingEventBase(BaseModel):
    assignment_name: str
    classes: Optional[List[str]] = None
    done: bool = False
    due_date: Optional[date] = None
    priority: Optional[str] = None  # High, Medium, Low, Complete
    status: Optional[str] = None  # Not started, In progress, Done
    teachers: Optional[List[str]] = None


class UpcomingEventCreate(UpcomingEventBase):
    pass


class UpcomingEventUpdate(BaseModel):
    assignment_name: Optional[str] = None
    classes: Optional[List[str]] = None
    done: Optional[bool] = None
    due_date: Optional[date] = None
    priority: Optional[str] = None
    status: Optional[str] = None
    teachers: Optional[List[str]] = None


class UpcomingEventResponse(UpcomingEventBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

