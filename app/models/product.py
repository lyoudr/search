from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Text, Enum
from . import Base 

class Product(Base):
    __tablename__ = 'product'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=False)
    price = Column(Float, nullable=False)
    stock = Column(Integer, default=0)

    def __repr__(self):
        return f"Product: {self.name}"