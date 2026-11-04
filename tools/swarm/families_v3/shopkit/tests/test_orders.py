from shop.orders import OrderError
from tests.helpers import make_service, place


def test_lines_merge():
    service = make_service()
    order = service.create("C-1")
    service.add_line(order.id, "BK-100", 1)
    service.add_line(order.id, "BK-100", 2)
    assert [(line.sku, line.qty) for line in order.lines] == [("BK-100", 3)]


def test_place_reserves_and_freezes_quote():
    service = make_service()
    order = place(service, "C-1", [("TL-330", 2)])
    assert order.status == "placed"
    assert service.inventory.snapshot()["TL-330"] == (30, 2, 28)
    assert order.quote["total"] == 21360
    service.catalog["TL-330"].price = service.catalog["TL-330"].price.times(2)
    assert order.quote["total"] == 21360


def test_ship():
    service = make_service()
    order = place(service, "C-2", [("FD-210", 3)])
    service.ship(order.id)
    assert order.status == "shipped"
    assert service.inventory.snapshot()["FD-210"] == (27, 0, 27)
    assert [e.kind for e in service.bus.history()] == ["order_created", "order_placed", "order_shipped"]


def test_status_rules():
    service = make_service()
    order = place(service, "C-1", [("BK-110", 1)])
    for action in (lambda: service.add_line(order.id, "BK-100", 1), lambda: service.place(order.id)):
        try:
            action()
        except OrderError:
            continue
        raise AssertionError("changed a placed order")


def test_unknown_coupon():
    service = make_service()
    order = service.create("C-1")
    try:
        service.set_coupon(order.id, "NOPE")
    except OrderError:
        return
    raise AssertionError("accepted an unknown coupon")
