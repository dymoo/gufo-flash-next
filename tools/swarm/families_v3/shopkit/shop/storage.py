"""Persist the whole shop state as one JSON document.

Format 1 (the current format) stores money as decimal strings ("12.34") next
to a "currency" field on the record that owns them, dates as ISO strings, and
an order's frozen quote as minor units. Records are written in a stable order
so that dumps can be diffed.
"""

from __future__ import annotations

import datetime as dt
import json

from shop import config
from shop.events import EventBus
from shop.inventory import Inventory
from shop.models import Customer, Order, OrderLine, Product
from shop.money import Money
from shop.orders import OrderService


class StorageError(ValueError):
    pass


def _money(text: str, currency: str) -> Money:
    try:
        return Money.parse(text, currency)
    except ValueError as error:
        raise StorageError(f"bad amount {text!r}: {error}") from None


# ---------------------------------------------------------------- records --

def dump_product(p: Product) -> dict:
    return {"sku": p.sku, "name": p.name, "category": p.category, "price": p.price.amount(),
            "currency": p.price.currency, "weight_g": p.weight_g}


def load_product(d: dict) -> Product:
    return Product(d["sku"], d["name"], d["category"], _money(d["price"], d["currency"]), int(d["weight_g"]))


def dump_customer(c: Customer) -> dict:
    return {"id": c.id, "name": c.name, "region": c.region, "tier": c.tier}


def load_customer(d: dict) -> Customer:
    return Customer(d["id"], d["name"], d["region"], d.get("tier", "standard"))


def dump_order(o: Order) -> dict:
    currency = o.lines[0].unit_price.currency if o.lines else config.DEFAULT_CURRENCY
    return {"id": o.id, "customer": o.customer_id, "created": o.created.isoformat(), "status": o.status,
            "coupon": o.coupon, "currency": currency,
            "lines": [{"sku": line.sku, "qty": line.qty, "price": line.unit_price.amount()} for line in o.lines],
            "quote": o.quote}


def load_order(d: dict) -> Order:
    lines = [OrderLine(line["sku"], int(line["qty"]), _money(line["price"], d["currency"])) for line in d["lines"]]
    return Order(d["id"], d["customer"], dt.date.fromisoformat(d["created"]), lines, d["status"], d.get("coupon"),
                 d.get("quote"))


# ------------------------------------------------------------------ state --

def dump_state(service: OrderService) -> str:
    inv = service.inventory
    doc = {"format": config.STORAGE_FORMAT,
           "products": [dump_product(p) for _, p in sorted(service.catalog.items())],
           "customers": [dump_customer(c) for _, c in sorted(service.customers.items())],
           "stock": {sku: [inv.on_hand.get(sku, 0), inv.reserved.get(sku, 0)] for sku in inv.snapshot()},
           "orders": [dump_order(o) for _, o in sorted(service.orders.items())],
           "next_order": service._next}
    return json.dumps(doc, indent=1, sort_keys=True)


def load_state(text: str, bus: EventBus | None = None, clock=None) -> OrderService:
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as error:
        raise StorageError(f"not JSON: {error}") from None
    version = doc.get("format")
    if version != 1:
        raise StorageError(f"unsupported format {version!r}")
    bus = bus or EventBus()
    inventory = Inventory(bus)
    for sku, (on_hand, reserved) in doc.get("stock", {}).items():
        inventory.on_hand[sku] = int(on_hand)
        inventory.reserved[sku] = int(reserved)
    catalog = {p.sku: p for p in map(load_product, doc.get("products", []))}
    customers = {c.id: c for c in map(load_customer, doc.get("customers", []))}
    service = OrderService(catalog, customers, inventory, bus, clock)
    for order in map(load_order, doc.get("orders", [])):
        service.orders[order.id] = order
    service._next = int(doc.get("next_order", len(service.orders) + 1))
    return service
