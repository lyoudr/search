from sqlalchemy import Column, Integer, String
from . import Base 

class User(Base):
    __tablename__ = 'user'

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(100), nullable=False, unique=True)
    name = Column(String(100), nullable=False)

    def __repr__(self):
        return f"User: {self.email}"