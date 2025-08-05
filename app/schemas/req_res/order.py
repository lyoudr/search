from pydantic import BaseModel 
from typing import List 

class OrderItemCreate(BaseModel):
    product_id: int 
    quantity: int 

class OrderCreate(BaseModel):
    user_id: int 
    items: List[OrderItemCreate]
