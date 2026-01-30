from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from app.models import get_db
from app.schemas.req_res.upcoming_events import (
    UpcomingEventCreate,
    UpcomingEventUpdate,
    UpcomingEventResponse
)
from app.repositories import upcoming_event_repository

router = APIRouter(tags=["upcoming-events"], prefix="/upcoming-events")


@router.post(
    "/",
    response_model=UpcomingEventResponse,
    summary="Create a new upcoming event",
)
def create_upcoming_event(
    event: UpcomingEventCreate,
    db: Session = Depends(get_db)
):
    """Create a new upcoming event/assignment"""
    return upcoming_event_repository.create_upcoming_event(
        db=db,
        assignment_name=event.assignment_name,
        classes=event.classes,
        done=event.done,
        due_date=event.due_date,
        priority=event.priority,
        status=event.status,
        teachers=event.teachers
    )


@router.get(
    "/",
    response_model=List[UpcomingEventResponse],
    summary="Get all upcoming events",
)
def get_all_upcoming_events(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    done: Optional[bool] = None,
    status: Optional[str] = None,
    priority: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get all upcoming events with optional filters"""
    return upcoming_event_repository.get_all_upcoming_events(
        db=db,
        skip=skip,
        limit=limit,
        done=done,
        status=status,
        priority=priority
    )


@router.get(
    "/{event_id}",
    response_model=UpcomingEventResponse,
    summary="Get an upcoming event by ID",
)
def get_upcoming_event(
    event_id: int,
    db: Session = Depends(get_db)
):
    """Get a specific upcoming event by ID"""
    event = upcoming_event_repository.get_upcoming_event_by_id(db, event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Upcoming event not found")
    return event


@router.put(
    "/{event_id}",
    response_model=UpcomingEventResponse,
    summary="Update an upcoming event",
)
def update_upcoming_event(
    event_id: int,
    event_update: UpcomingEventUpdate,
    db: Session = Depends(get_db)
):
    """Update an existing upcoming event"""
    updated_event = upcoming_event_repository.update_upcoming_event(
        db=db,
        event_id=event_id,
        assignment_name=event_update.assignment_name,
        classes=event_update.classes,
        done=event_update.done,
        due_date=event_update.due_date,
        priority=event_update.priority,
        status=event_update.status,
        teachers=event_update.teachers
    )
    if not updated_event:
        raise HTTPException(status_code=404, detail="Upcoming event not found")
    return updated_event


@router.delete(
    "/{event_id}",
    summary="Delete an upcoming event",
)
def delete_upcoming_event(
    event_id: int,
    db: Session = Depends(get_db)
):
    """Delete an upcoming event"""
    success = upcoming_event_repository.delete_upcoming_event(db, event_id)
    if not success:
        raise HTTPException(status_code=404, detail="Upcoming event not found")
    return {"status": "success", "message": f"Upcoming event {event_id} deleted successfully"}

