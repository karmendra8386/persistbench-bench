"""Deterministic retail task world with database ground truth.

All state lives in one dict-of-tables "database". Tools are the only way to
mutate it. The scorer reads the same database, so ground truth is correct by
construction. Standard library only.
"""
from __future__ import annotations

import copy


class World:
    def __init__(self, state: dict | None = None):
        self.state = state or {
            "products": {},
            "orders": {},
            "payments": {},
            "shipments": {},
            "counters": {"payment": 0, "shipment": 0, "label": 0},
        }

    def clone(self) -> "World":
        return World(copy.deepcopy(self.state))

    # ---- read tools -------------------------------------------------------
    def get_order(self, order_id: str) -> dict:
        order = self.state["orders"].get(order_id)
        if order is None:
            return {"error": f"unknown order {order_id}"}
        return copy.deepcopy(order)

    def get_product(self, sku: str) -> dict:
        product = self.state["products"].get(sku)
        if product is None:
            return {"error": f"unknown sku {sku}"}
        return copy.deepcopy(product)

    def get_shipment(self, tracking: str) -> dict:
        shipment = self.state["shipments"].get(tracking)
        if shipment is None:
            return {"error": f"unknown tracking {tracking}"}
        return copy.deepcopy(shipment)

    # ---- write tools ------------------------------------------------------
    def charge_payment(self, order_id: str) -> dict:
        order = self.state["orders"].get(order_id)
        if order is None:
            return {"error": f"unknown order {order_id}"}
        if order["payment_status"] != "unpaid" or order["status"] != "placed":
            return {"error": f"order {order_id} is not payable"}
        self.state["counters"]["payment"] += 1
        payment_id = f"PAY-{self.state['counters']['payment']:04d}"
        self.state["payments"][payment_id] = {
            "payment_id": payment_id,
            "order_id": order_id,
            "amount_cents": order["total_cents"],
            "status": "captured",
        }
        order["payment_id"] = payment_id
        order["payment_status"] = "captured"
        order["status"] = "paid"
        return {
            "payment_id": payment_id,
            "amount_cents": order["total_cents"],
            "payment_status": "captured",
            "order_status": "paid",
        }

    def ship_order(self, order_id: str) -> dict:
        order = self.state["orders"].get(order_id)
        if order is None:
            return {"error": f"unknown order {order_id}"}
        if order["status"] != "paid":
            return {"error": f"order {order_id} is not paid"}
        self.state["counters"]["shipment"] += 1
        tracking = f"TRK-{self.state['counters']['shipment']:06d}"
        order["tracking"] = tracking
        order["status"] = "shipped"
        self.state["shipments"][tracking] = {
            "tracking": tracking,
            "order_id": order_id,
            "address": order["address"],
            "status": "in_transit",
        }
        # A ship confirmation reports tracking + status; the delivery address
        # of record lives in the DB (shipments table), not in this response.
        return {"tracking": tracking, "order_status": "shipped"}

    def update_shipping_address(self, order_id: str, address: str) -> dict:
        order = self.state["orders"].get(order_id)
        if order is None:
            return {"error": f"unknown order {order_id}"}
        order["address"] = address
        return {"order_id": order_id, "address": address}

    def cancel_order(self, order_id: str) -> dict:
        order = self.state["orders"].get(order_id)
        if order is None:
            return {"error": f"unknown order {order_id}"}
        if order["status"] not in ("placed", "paid"):
            return {"error": f"order {order_id} cannot be cancelled"}
        if order["payment_status"] == "captured":
            return {"error": f"order {order_id} is paid; refund first"}
        order["status"] = "cancelled"
        order["cancellation_fee_cents"] = 0
        return {"order_status": "cancelled", "cancellation_fee_cents": 0}

    def refund_order(self, order_id: str) -> dict:
        order = self.state["orders"].get(order_id)
        if order is None:
            return {"error": f"unknown order {order_id}"}
        if order["payment_status"] != "captured":
            return {"error": f"order {order_id} has no captured payment"}
        payment = self.state["payments"][order["payment_id"]]
        payment["status"] = "refunded"
        order["payment_status"] = "refunded"
        order["status"] = "refunded"
        return {
            "refund_amount_cents": payment["amount_cents"],
            "payment_status": "refunded",
            "order_status": "refunded",
        }

    def restock(self, sku: str, qty: int) -> dict:
        product = self.state["products"].get(sku)
        if product is None:
            return {"error": f"unknown sku {sku}"}
        product["stock"] += qty
        return {"sku": sku, "stock": product["stock"]}

    def create_return_label(self, tracking: str) -> dict:
        shipment = self.state["shipments"].get(tracking)
        if shipment is None:
            return {"error": f"unknown tracking {tracking}"}
        self.state["counters"]["label"] += 1
        label_id = f"LBL-{self.state['counters']['label']:04d}"
        shipment["return_label"] = label_id
        return {"label_id": label_id, "status": "created", "tracking": tracking}
