from shop.events import EventBus
from shop.inventory import Inventory, OutOfStock


def test_receive_and_reserve():
    inv = Inventory()
    inv.receive("AB-1", 10)
    inv.reserve("AB-1", 4)
    assert inv.available("AB-1") == 6
    assert inv.snapshot() == {"AB-1": (10, 4, 6)}


def test_out_of_stock():
    inv = Inventory()
    inv.receive("AB-1", 2)
    try:
        inv.reserve("AB-1", 3)
    except OutOfStock as error:
        assert error.available == 2
        return
    raise AssertionError("reserved more than available")


def test_commit_removes_units():
    inv = Inventory()
    inv.receive("AB-1", 10)
    inv.reserve("AB-1", 4)
    inv.commit("AB-1", 4)
    assert inv.snapshot()["AB-1"] == (6, 0, 6)


def test_low_stock_event_once():
    bus = EventBus()
    inv = Inventory(bus, threshold=5)
    inv.receive("AB-1", 8)
    inv.reserve("AB-1", 2)
    inv.reserve("AB-1", 2)
    inv.reserve("AB-1", 1)
    assert [e.data for e in bus.history("low_stock")] == [{"sku": "AB-1", "available": 4}]
    assert inv.low_stock() == ["AB-1"]


def test_adjust():
    inv = Inventory()
    inv.receive("AB-1", 10)
    assert inv.adjust("AB-1", 7) == -3
    assert inv.available("AB-1") == 7
