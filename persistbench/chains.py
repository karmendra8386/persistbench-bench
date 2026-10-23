"""The Week 1 task-chain suite: 12 self-built chains.

Each chain = a Session-1 origin task (a tool plan), one critical step where a
fault may be injected, the intended facts (what must be true in the DB for the
origin task to count as successful), and three dependent tasks for Sessions
2-4 whose correct resolution depends on Session 1's true outcome. The chain's
``primary_fact`` is the fact targeted by the correction event before Session 4.

Ground truth is read from the world DB via :func:`resolve_fact`, so it is
correct by construction.
"""
from __future__ import annotations


def _product(sku, name, price_cents, stock):
    return {"sku": sku, "name": name, "price_cents": price_cents, "stock": stock}


def _order(order_id, sku, qty, unit_price_cents, total_cents, address,
           status="placed", payment_status="unpaid", payment_id=None):
    return {
        "order_id": order_id,
        "customer_id": "CUST-1",
        "sku": sku,
        "qty": qty,
        "unit_price_cents": unit_price_cents,
        "total_cents": total_cents,
        "status": status,
        "payment_status": payment_status,
        "address": address,
        "payment_id": payment_id,
        "tracking": None,
        "cancellation_fee_cents": None,
    }


def _prepaid_payment(payment_id, order_id, amount_cents):
    return {"payment_id": payment_id, "order_id": order_id,
            "amount_cents": amount_cents, "status": "captured"}


def _report(session, fact):
    return {"session": session, "kind": "report", "fact": fact}


def _return_label(session):
    return {"session": session, "kind": "action", "tool": "create_return_label",
            "arg": "tracking", "fact": "tracking"}


def _decision(session, fact, threshold):
    return {"session": session, "kind": "decision", "fact": fact, "threshold": threshold}


CHAINS = [
    {
        "chain_id": "chain01_pay_ship_laptop",
        "title": "Pay for and ship a laptop order",
        "origin_instruction": "Charge the customer for order O1001 and ship it.",
        "setup": {
            "products": {"SKU-LAPTOP": _product("SKU-LAPTOP", "Laptop", 150000, 5)},
            "orders": {"O1001": _order("O1001", "SKU-LAPTOP", 1, 150000, 150000, "100 Pine St")},
            "payments": {},
        },
        "order": "O1001", "product": None,
        "plan": [("charge_payment", {"order_id": "O1001"}), ("ship_order", {"order_id": "O1001"})],
        "critical_step": 1,
        "fault": {"wrong_field": "tracking", "wrong_value": "TRK-999999",
                  "stale_field": "order_status", "stale_value": "paid",
                  "error_message": "Shipping service unavailable (carrier API timeout)"},
        "intended_facts": {"order_status": "shipped", "payment_status": "captured",
                           "charged_cents": 150000, "tracking": "TRK-000001"},
        "primary_fact": "tracking",
        "dependents": [_report(2, "tracking"), _return_label(3), _report(4, "tracking")],
    },
    {
        "chain_id": "chain02_charge_phone",
        "title": "Charge for a phone order (fulfilment later)",
        "origin_instruction": "Charge the customer for order O1002.",
        "setup": {
            "products": {"SKU-PHONE": _product("SKU-PHONE", "Phone", 80000, 10)},
            "orders": {"O1002": _order("O1002", "SKU-PHONE", 2, 80000, 160000, "5 Oak Ave")},
            "payments": {},
        },
        "order": "O1002", "product": None,
        "plan": [("charge_payment", {"order_id": "O1002"})],
        "critical_step": 0,
        "fault": {"wrong_field": "amount_cents", "wrong_value": 16000,
                  "stale_field": "payment_status", "stale_value": "unpaid",
                  "error_message": "Payment gateway timeout"},
        "intended_facts": {"payment_status": "captured", "charged_cents": 160000,
                           "order_status": "paid"},
        "primary_fact": "charged_cents",
        "dependents": [_report(2, "charged_cents"), _report(3, "payment_status"),
                       _report(4, "charged_cents")],
    },
    {
        "chain_id": "chain03_address_ship_book",
        "title": "Update the delivery address of a prepaid book order, then ship it",
        "origin_instruction": "Change order O1003's delivery address to 88 Birch Ave and ship it.",
        "setup": {
            "products": {"SKU-BOOK": _product("SKU-BOOK", "Book", 2500, 50)},
            "orders": {"O1003": _order("O1003", "SKU-BOOK", 3, 2500, 7500, "22 Elm St",
                                       status="paid", payment_status="captured",
                                       payment_id="PAY-9003")},
            "payments": {"PAY-9003": _prepaid_payment("PAY-9003", "O1003", 7500)},
        },
        "order": "O1003", "product": None,
        "plan": [("update_shipping_address", {"order_id": "O1003", "address": "88 Birch Ave"}),
                 ("ship_order", {"order_id": "O1003"})],
        "critical_step": 0,
        "fault": {"wrong_field": "address", "wrong_value": "88 Birch Avenue, Apt 4",
                  "stale_field": "address", "stale_value": "22 Elm St",
                  "error_message": "Address service unavailable"},
        "intended_facts": {"address": "88 Birch Ave", "order_status": "shipped",
                           "tracking": "TRK-000001"},
        "primary_fact": "address",
        "dependents": [_report(2, "address"), _report(3, "tracking"), _report(4, "address")],
    },
    {
        "chain_id": "chain04_refund_chair",
        "title": "Refund a prepaid chair order",
        "origin_instruction": "Refund order O1004 in full.",
        "setup": {
            "products": {"SKU-CHAIR": _product("SKU-CHAIR", "Chair", 12000, 8)},
            "orders": {"O1004": _order("O1004", "SKU-CHAIR", 1, 12000, 12000, "7 Walnut Way",
                                       status="paid", payment_status="captured",
                                       payment_id="PAY-9004")},
            "payments": {"PAY-9004": _prepaid_payment("PAY-9004", "O1004", 12000)},
        },
        "order": "O1004", "product": None,
        "plan": [("refund_order", {"order_id": "O1004"})],
        "critical_step": 0,
        "fault": {"wrong_field": "refund_amount_cents", "wrong_value": 1200,
                  "stale_field": "payment_status", "stale_value": "captured",
                  "error_message": "Refund service unavailable"},
        "intended_facts": {"payment_status": "refunded", "refund_cents": 12000,
                           "order_status": "refunded"},
        "primary_fact": "refund_cents",
        "dependents": [_report(2, "refund_cents"), _report(3, "payment_status"),
                       _report(4, "refund_cents")],
    },
    {
        "chain_id": "chain05_restock_lamp",
        "title": "Restock lamps from zero",
        "origin_instruction": "Restock SKU-LAMP with 40 units.",
        "setup": {
            "products": {"SKU-LAMP": _product("SKU-LAMP", "Lamp", 4500, 0)},
            "orders": {},
            "payments": {},
        },
        "order": None, "product": "SKU-LAMP",
        "plan": [("restock", {"sku": "SKU-LAMP", "qty": 40})],
        "critical_step": 0,
        "fault": {"wrong_field": "stock", "wrong_value": 4,
                  "stale_field": "stock", "stale_value": 0,
                  "error_message": "Inventory service unavailable"},
        "intended_facts": {"stock": 40},
        "primary_fact": "stock",
        "dependents": [_report(2, "stock"), _decision(3, "stock", 25), _report(4, "stock")],
    },
    {
        "chain_id": "chain06_pay_ship_monitors",
        "title": "Pay for and ship a two-monitor order",
        "origin_instruction": "Charge the customer for order O1006 and ship it.",
        "setup": {
            "products": {"SKU-MONITOR": _product("SKU-MONITOR", "Monitor", 22000, 6)},
            "orders": {"O1006": _order("O1006", "SKU-MONITOR", 2, 22000, 44000, "300 River Rd")},
            "payments": {},
        },
        "order": "O1006", "product": None,
        "plan": [("charge_payment", {"order_id": "O1006"}), ("ship_order", {"order_id": "O1006"})],
        "critical_step": 1,
        "fault": {"wrong_field": "tracking", "wrong_value": "TRK-999999",
                  "stale_field": "order_status", "stale_value": "paid",
                  "error_message": "Shipping service unavailable (carrier API timeout)"},
        "intended_facts": {"order_status": "shipped", "payment_status": "captured",
                           "charged_cents": 44000, "tracking": "TRK-000001"},
        "primary_fact": "tracking",
        "dependents": [_report(2, "tracking"), _return_label(3), _report(4, "tracking")],
    },
    {
        "chain_id": "chain07_charge_discount_keyboard",
        "title": "Charge a discounted keyboard order (total below list price)",
        "origin_instruction": "Charge the customer for order O1007 at its discounted total.",
        "setup": {
            "products": {"SKU-KEYBOARD": _product("SKU-KEYBOARD", "Keyboard", 7500, 15)},
            "orders": {"O1007": _order("O1007", "SKU-KEYBOARD", 1, 7500, 6750, "41 Hill St")},
            "payments": {},
        },
        "order": "O1007", "product": None,
        "plan": [("charge_payment", {"order_id": "O1007"})],
        "critical_step": 0,
        "fault": {"wrong_field": "amount_cents", "wrong_value": 7500,
                  "stale_field": "payment_status", "stale_value": "unpaid",
                  "error_message": "Payment gateway timeout"},
        "intended_facts": {"charged_cents": 6750, "payment_status": "captured",
                           "order_status": "paid"},
        "primary_fact": "charged_cents",
        "dependents": [_report(2, "charged_cents"), _report(3, "payment_status"),
                       _report(4, "charged_cents")],
    },
    {
        "chain_id": "chain08_cancel_desk",
        "title": "Cancel an unpaid desk order",
        "origin_instruction": "Cancel order O1008.",
        "setup": {
            "products": {"SKU-DESK": _product("SKU-DESK", "Desk", 30000, 3)},
            "orders": {"O1008": _order("O1008", "SKU-DESK", 1, 30000, 30000, "9 Lake Dr")},
            "payments": {},
        },
        "order": "O1008", "product": None,
        "plan": [("cancel_order", {"order_id": "O1008"})],
        "critical_step": 0,
        "fault": {"wrong_field": "cancellation_fee_cents", "wrong_value": 4500,
                  "stale_field": "order_status", "stale_value": "placed",
                  "error_message": "Order service unavailable"},
        "intended_facts": {"order_status": "cancelled", "cancellation_fee_cents": 0},
        "primary_fact": "cancellation_fee_cents",
        "dependents": [_report(2, "order_status"), _report(3, "cancellation_fee_cents"),
                       _report(4, "cancellation_fee_cents")],
    },
    {
        "chain_id": "chain09_update_ship_tablet",
        "title": "Update a prepaid tablet order's address, then ship it",
        "origin_instruction": "Change order O1009's delivery address to 9 Cedar Ct and ship it.",
        "setup": {
            "products": {"SKU-TABLET": _product("SKU-TABLET", "Tablet", 45000, 4)},
            "orders": {"O1009": _order("O1009", "SKU-TABLET", 1, 45000, 45000, "31 Maple Dr",
                                       status="paid", payment_status="captured",
                                       payment_id="PAY-9009")},
            "payments": {"PAY-9009": _prepaid_payment("PAY-9009", "O1009", 45000)},
        },
        "order": "O1009", "product": None,
        "plan": [("update_shipping_address", {"order_id": "O1009", "address": "9 Cedar Ct"}),
                 ("ship_order", {"order_id": "O1009"})],
        "critical_step": 1,
        "fault": {"wrong_field": "tracking", "wrong_value": "TRK-999999",
                  "stale_field": "order_status", "stale_value": "paid",
                  "error_message": "Shipping service unavailable (carrier API timeout)"},
        "intended_facts": {"address": "9 Cedar Ct", "order_status": "shipped",
                           "tracking": "TRK-000001"},
        "primary_fact": "tracking",
        "dependents": [_report(2, "address"), _return_label(3), _report(4, "tracking")],
    },
    {
        "chain_id": "chain10_refund_headphones",
        "title": "Refund a prepaid headphones order (damaged item)",
        "origin_instruction": "Refund order O1010 in full.",
        "setup": {
            "products": {"SKU-HEADPHONES": _product("SKU-HEADPHONES", "Headphones", 9900, 20)},
            "orders": {"O1010": _order("O1010", "SKU-HEADPHONES", 2, 9900, 19800,
                                       "77 Sunset Blvd", status="paid",
                                       payment_status="captured", payment_id="PAY-9010")},
            "payments": {"PAY-9010": _prepaid_payment("PAY-9010", "O1010", 19800)},
        },
        "order": "O1010", "product": None,
        "plan": [("refund_order", {"order_id": "O1010"})],
        "critical_step": 0,
        "fault": {"wrong_field": "refund_amount_cents", "wrong_value": 1980,
                  "stale_field": "payment_status", "stale_value": "captured",
                  "error_message": "Refund service unavailable"},
        "intended_facts": {"payment_status": "refunded", "refund_cents": 19800,
                           "order_status": "refunded"},
        "primary_fact": "refund_cents",
        "dependents": [_report(2, "refund_cents"), _report(3, "order_status"),
                       _report(4, "refund_cents")],
    },
    {
        "chain_id": "chain11_restock_sofa",
        "title": "Restock sofas from one unit",
        "origin_instruction": "Restock SKU-SOFA with 5 units.",
        "setup": {
            "products": {"SKU-SOFA": _product("SKU-SOFA", "Sofa", 120000, 1)},
            "orders": {},
            "payments": {},
        },
        "order": None, "product": "SKU-SOFA",
        "plan": [("restock", {"sku": "SKU-SOFA", "qty": 5})],
        "critical_step": 0,
        "fault": {"wrong_field": "stock", "wrong_value": 60,
                  "stale_field": "stock", "stale_value": 1,
                  "error_message": "Inventory service unavailable"},
        "intended_facts": {"stock": 6},
        "primary_fact": "stock",
        "dependents": [_report(2, "stock"), _decision(3, "stock", 4), _report(4, "stock")],
    },
    {
        "chain_id": "chain12_ship_mugs",
        "title": "Ship a prepaid bulk mug order",
        "origin_instruction": "Ship order O1012.",
        "setup": {
            "products": {"SKU-MUG": _product("SKU-MUG", "Mug", 1500, 100)},
            "orders": {"O1012": _order("O1012", "SKU-MUG", 10, 1500, 15000, "12 Garden Ln",
                                       status="paid", payment_status="captured",
                                       payment_id="PAY-9012")},
            "payments": {"PAY-9012": _prepaid_payment("PAY-9012", "O1012", 15000)},
        },
        "order": "O1012", "product": None,
        "plan": [("ship_order", {"order_id": "O1012"})],
        "critical_step": 0,
        "fault": {"wrong_field": "tracking", "wrong_value": "TRK-999999",
                  "stale_field": "order_status", "stale_value": "paid",
                  "error_message": "Shipping service unavailable (carrier API timeout)"},
        "intended_facts": {"order_status": "shipped", "tracking": "TRK-000001"},
        "primary_fact": "tracking",
        "dependents": [_report(2, "tracking"), _return_label(3), _report(4, "tracking")],
    },
]

CHAINS_BY_ID = {chain["chain_id"]: chain for chain in CHAINS}


def build_world(chain: dict):
    """Instantiate a fresh World from a chain's setup (deep-copied)."""
    import copy

    from .world import World

    state = {
        "products": copy.deepcopy(chain["setup"]["products"]),
        "orders": copy.deepcopy(chain["setup"]["orders"]),
        "payments": copy.deepcopy(chain["setup"]["payments"]),
        "shipments": {},
        "counters": {"payment": 0, "shipment": 0, "label": 0},
    }
    return World(state)


def resolve_fact(world, chain: dict, key: str):
    """Read one ground-truth fact from the world DB. Correct by construction."""
    state = world.state
    if key == "stock":
        return state["products"][chain["product"]]["stock"]
    order = state["orders"][chain["order"]]
    if key == "order_status":
        return order["status"]
    if key == "payment_status":
        return order["payment_status"]
    if key == "address":
        return order["address"]
    if key == "tracking":
        return order["tracking"]
    if key == "cancellation_fee_cents":
        return order["cancellation_fee_cents"]
    if key in ("charged_cents", "refund_cents"):
        payment_id = order["payment_id"]
        if payment_id is None or payment_id not in state["payments"]:
            return None
        return state["payments"][payment_id]["amount_cents"]
    raise KeyError(f"unknown fact key {key!r}")
