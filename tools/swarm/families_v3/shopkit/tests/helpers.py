"""Shared fixtures: a small deterministic shop."""

import datetime as dt

from shop.events import EventBus
from shop.inventory import Inventory
from shop.models import Customer, Product
from shop.money import Money
from shop.orders import OrderService

PRODUCTS = [
    ("BK-100", "Atlas of Rivers", "books", "24.99", 1200),
    ("FD-210", "Olive oil 1l", "food", "8.50", 1100),
    ("TL-330", "Cordless drill", "tools", "89.00", 2100),
    ("TY-410", "Wooden train set", "children", "34.95", 1800),
    ("GD-520", "Hose reel", "garden", "45.20", 4300),
    ("TL-340", "Socket set", "tools", "39.99", 2600),
    ("BK-110", "Field guide to moths", "books", "12.75", 450),
    ("GD-530", "Seed tray (pack of 10)", "garden", "6.40", 300),
]
CUSTOMERS = [
    ("C-1", "Ada Brook", "UK", "standard"),
    ("C-2", "Bram Visser", "NL", "silver"),
    ("C-3", "Clara Weiss", "DE", "gold"),
    ("C-4", "Dara Quinn", "IE", "standard"),
]


class Clock:
    def __init__(self, today=dt.date(2026, 9, 14)):
        self.today = today

    def __call__(self):
        return self.today


def make_service(stock=30, clock=None):
    bus = EventBus()
    inventory = Inventory(bus)
    catalog = {sku: Product(sku, name, cat, Money.parse(price), grams) for sku, name, cat, price, grams in PRODUCTS}
    customers = {cid: Customer(cid, name, region, tier) for cid, name, region, tier in CUSTOMERS}
    for sku in catalog:
        inventory.receive(sku, stock)
    return OrderService(catalog, customers, inventory, bus, clock or Clock())


def place(service, customer_id, lines, coupon=None):
    order = service.create(customer_id, coupon=coupon)
    for sku, qty in lines:
        service.add_line(order.id, sku, qty)
    service.place(order.id)
    return order


def sample_service():
    """Five orders over two months: three placed or shipped, one cancelled, one draft."""
    clock = Clock(dt.date(2026, 8, 30))
    service = make_service(clock=clock)
    place(service, "C-1", [("BK-100", 2), ("TL-330", 1)])
    clock.today = dt.date(2026, 9, 2)
    o2 = place(service, "C-2", [("FD-210", 12), ("GD-530", 3)], coupon="WELCOME10")
    service.ship(o2.id)
    o3 = place(service, "C-3", [("TY-410", 1), ("TL-340", 2), ("BK-110", 1)], coupon="FIVEOFF")
    o4 = place(service, "C-4", [("GD-520", 1)])
    service.cancel(o4.id)
    draft = service.create("C-1")
    service.add_line(draft.id, "BK-110", 1)
    return service, [o3]
