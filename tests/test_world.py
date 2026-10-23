"""World tool semantics: the DB is the ground truth every score reads."""
from persistbench.chains import CHAINS_BY_ID, build_world


def _world(chain_id):
    return build_world(CHAINS_BY_ID[chain_id])


def test_charge_then_ship_flow():
    world = _world("chain01_pay_ship_laptop")
    charge = world.charge_payment("O1001")
    assert charge["payment_status"] == "captured"
    assert charge["amount_cents"] == 150000
    ship = world.ship_order("O1001")
    assert ship["tracking"] == "TRK-000001"
    order = world.get_order("O1001")
    assert order["status"] == "shipped"
    assert order["payment_status"] == "captured"
    shipment = world.get_shipment("TRK-000001")
    assert shipment["order_id"] == "O1001"


def test_ship_before_payment_is_an_error_and_changes_nothing():
    world = _world("chain01_pay_ship_laptop")
    result = world.ship_order("O1001")
    assert "error" in result
    assert world.get_order("O1001")["status"] == "placed"


def test_double_charge_is_an_error():
    world = _world("chain02_charge_phone")
    assert "error" not in world.charge_payment("O1002")
    assert "error" in world.charge_payment("O1002")


def test_refund_flow():
    world = _world("chain04_refund_chair")
    result = world.refund_order("O1004")
    assert result["refund_amount_cents"] == 12000
    assert world.get_order("O1004")["payment_status"] == "refunded"
    assert world.state["payments"]["PAY-9004"]["status"] == "refunded"


def test_cancel_unpaid_order_sets_zero_fee():
    world = _world("chain08_cancel_desk")
    result = world.cancel_order("O1008")
    assert result["order_status"] == "cancelled"
    assert world.get_order("O1008")["cancellation_fee_cents"] == 0


def test_restock_accumulates():
    world = _world("chain11_restock_sofa")
    assert world.restock("SKU-SOFA", 5)["stock"] == 6


def test_return_label_requires_real_tracking():
    world = _world("chain12_ship_mugs")
    assert "error" in world.create_return_label("TRK-999999")
    world.ship_order("O1012")
    assert world.create_return_label("TRK-000001")["status"] == "created"


def test_unknown_entities_return_errors():
    world = _world("chain01_pay_ship_laptop")
    assert "error" in world.get_order("NOPE")
    assert "error" in world.get_product("SKU-NOPE")
    assert "error" in world.charge_payment("NOPE")
