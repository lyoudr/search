from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import date
from app.models.analyze import UpcomingEvent


def create_upcoming_event(
    db: Session,
    assignment_name: str,
    classes: Optional[List[str]] = None,
    done: bool = False,
    due_date: Optional[date] = None,
    priority: Optional[str] = None,
    status: Optional[str] = None,
    teachers: Optional[List[str]] = None
) -> UpcomingEvent:
    """Create a new upcoming event record"""
    upcoming_event = UpcomingEvent(
        assignment_name=assignment_name,
        classes=classes,
        done=done,
        due_date=due_date,
        priority=priority,
        status=status,
        teachers=teachers
    )
    db.add(upcoming_event)
    db.commit()
    db.refresh(upcoming_event)
    return upcoming_event


def get_upcoming_event_by_id(db: Session, event_id: int) -> Optional[UpcomingEvent]:
    """Get upcoming event by ID"""
    return db.query(UpcomingEvent).filter(UpcomingEvent.id == event_id).first()


def get_all_upcoming_events(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    done: Optional[bool] = None,
    status: Optional[str] = None,
    priority: Optional[str] = None
) -> List[UpcomingEvent]:
    """Get all upcoming events with optional filters"""
    query = db.query(UpcomingEvent)
    
    if done is not None:
        query = query.filter(UpcomingEvent.done == done)
    if status is not None:
        query = query.filter(UpcomingEvent.status == status)
    if priority is not None:
        query = query.filter(UpcomingEvent.priority == priority)
    
    return query.offset(skip).limit(limit).all()


def update_upcoming_event(
    db: Session,
    event_id: int,
    assignment_name: Optional[str] = None,
    classes: Optional[List[str]] = None,
    done: Optional[bool] = None,
    due_date: Optional[date] = None,
    priority: Optional[str] = None,
    status: Optional[str] = None,
    teachers: Optional[List[str]] = None
) -> Optional[UpcomingEvent]:
    """Update an upcoming event"""
    upcoming_event = db.query(UpcomingEvent).filter(UpcomingEvent.id == event_id).first()
    if not upcoming_event:
        return None
    
    if assignment_name is not None:
        upcoming_event.assignment_name = assignment_name
    if classes is not None:
        upcoming_event.classes = classes
    if done is not None:
        upcoming_event.done = done
    if due_date is not None:
        upcoming_event.due_date = due_date
    if priority is not None:
        upcoming_event.priority = priority
    if status is not None:
        upcoming_event.status = status
    if teachers is not None:
        upcoming_event.teachers = teachers
    
    db.commit()
    db.refresh(upcoming_event)
    return upcoming_event


def delete_upcoming_event(db: Session, event_id: int) -> bool:
    """Delete an upcoming event"""
    upcoming_event = db.query(UpcomingEvent).filter(UpcomingEvent.id == event_id).first()
    if not upcoming_event:
        return False
    
    db.delete(upcoming_event)
    db.commit()
    return True

