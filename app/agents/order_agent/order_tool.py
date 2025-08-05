from app.repositories.order_repository import create_order
from app.schemas.req_res.order import OrderCreate, OrderItemCreate
from app.models.user import User 
from app.models import get_db

from sqlalchemy.orm import Session

db = next(get_db())

def create_order_tool_with_db(input: str) -> str:
    return create_order_tool(input, db)

# 🛒 3. Tool: CreateOrder — Store a New Order (Mock Example)
def create_order_tool(input: str, db: Session) -> str:
    """
    Accpets:
    - Text: "Order for Alice: 1x Shampoo, 2x Conditioner"
    """
    try:
        # Parse natural language: 'Order for Alice: 1x Shampoo, 2x Conditioner'
        import re
        match = re.search(r"Order for ([^:]+):(.+)", input)
        if not match:
            return "❌ Input format error: Please use 'Order for <name>: <items>' or JSON."
        customer_name = match.group(1).strip()
        items_str = match.group(2)
        # Find user by name
        user = db.query(User).filter(User.name == customer_name).first()
        if not user:
            return f"❌ User '{customer_name}' not found."
        # Parse items: e.g. '1x Shampoo, 2x Conditioner'
        items = []
        from app.models.product import Product
        for item_match in re.finditer(r"(\d+)\s*[xX]\s*([^,]+)", items_str):
            qty = int(item_match.group(1))
            product_name = item_match.group(2).strip()
            product = db.query(Product).filter(Product.name == product_name).first()
            if not product:
                # Optionally, handle missing product (skip or return error)
                continue
            print('product_id is ->', product.id)
            items.append(OrderItemCreate(product_id=product.id, quantity=qty))
        user_id = user.id
        # 🧾 Create Pydantic order schema
        order_data = OrderCreate(user_id=user_id, items=items)
        # 💾 Call DB function
        order = create_order(db=db, order_data=order_data)
        return f"✅ Order #{order.id} created for {customer_name}, total: ${order.total_price:.2f}"
    except Exception as e:
        return f"❌ Failed to create order: {str(e)}"