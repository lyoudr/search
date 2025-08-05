from sqlalchemy import Column, Integer, Float, ForeignKey, DateTime, Enum
from datetime import datetime
from . import Base 

class Order(Base):
    __tablename__ = 'order'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('user.id'), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(
        Enum("pending", "paid", "shipped", "completed", "cancelled", name="order_status_enum"),
        default="pending"
    )
    total_price = Column(Float, nullable=False, default=0.0, server_default="0.0")


class OrderItem(Base):
    __tablename__ = 'order_item'

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(Integer, ForeignKey('order.id'), nullable=False)
    product_id = Column(Integer, ForeignKey('product.id'), nullable=False)
    quantity = Column(Integer, nullable=False, default=1, server_default="1")