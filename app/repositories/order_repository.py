from sqlalchemy.orm import Session 

from app.models.order import Order, OrderItem
from app.models.product import Product 
from app.schemas.req_res.order import OrderCreate 

def create_order(db: Session, order_data: OrderCreate) -> Order:
    order = Order(user_id=order_data.user_id)
    db.add(order)
    db.flush()
    total_price = 0.0

    for item in order_data.items:
        # Fetch product to ge the price
        product = db.query(Product).filter(Product.id == item.product_id).first()
        if not product:
            raise ValueError(f"Product with ID {item.product_id} not found")
        
        subtotal = product.price * item.quantity
        total_price += subtotal 

        order_item = OrderItem(
            order_id = order.id,
            product_id = item.product_id,
            quantity = item.quantity
        )
        db.add(order_item)
    
    order.total_price = total_price

    db.commit()
    db.refresh(order)
    return order