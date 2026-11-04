from shop import config, pricing
from shop.money import Money
from tests.helpers import make_service


def quote(customer_id, lines, coupon=None):
    service = make_service()
    order = service.create(customer_id, coupon=coupon)
    for sku, qty in lines:
        service.add_line(order.id, sku, qty)
    return service.preview(order.id)


def test_standard_rate_and_shipping():
    q = quote("C-1", [("TL-340", 1)])
    assert q.goods == Money(3999)
    assert q.tax == Money(800)
    assert q.shipping == Money(795)
    assert q.total == Money(5594)


def test_reduced_rate_overrides_region():
    q = quote("C-4", [("BK-100", 1), ("FD-210", 1)])
    assert [line.tax_rate for line in q.lines] == ["0", "5"]
    assert q.tax == Money(43)


def test_tier_discount():
    q = quote("C-3", [("TL-330", 1)])
    assert q.discount == Money(890)
    assert q.net == Money(8010)
    assert q.shipping == Money(0)


def test_free_shipping_threshold():
    assert quote("C-1", [("GD-520", 1)]).shipping == Money(795)
    assert quote("C-1", [("GD-520", 2)]).shipping == Money(0)


def test_coupon_fixed_is_capped():
    q = quote("C-1", [("GD-530", 1)], coupon="FIVEOFF")
    assert q.net == Money(140)


def test_unknown_region():
    service = make_service()
    service.customers["C-1"].region = "XX"
    order = service.create("C-1")
    service.add_line(order.id, "TL-330", 1)
    try:
        service.preview(order.id)
    except pricing.PricingError:
        return
    raise AssertionError("priced an order for an unknown region")


def test_config_is_read_at_call_time():
    old = dict(config.TAX_RATES)
    try:
        config.TAX_RATES["UK"] = "10"
        assert quote("C-1", [("TL-330", 1)]).tax == Money(890)
    finally:
        config.TAX_RATES.clear()
        config.TAX_RATES.update(old)
