"""Domain records. Plain dataclasses; services own the behaviour."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from shop.money import Money

TIERS = ("standard", "silver", "gold")
ORDER_STATUSES = ("draft", "placed", "shipped", "cancelled")


@dataclass
class Product:
    sku: str
    name: str
    category: str
    price: Money
    weight_g: int

    def __post_init__(self):
        if not self.sku or not self.sku.isupper():
            raise ValueError(f"bad sku {self.sku!r}")
        if self.weight_g < 0:
            raise ValueError("weight must be non-negative")
        if self.price.is_negative():
            raise ValueError("price must be non-negative")


@dataclass
class Customer:
    id: str
    name: str
    region: str
    tier: str = "standard"

    def __post_init__(self):
        if self.tier not in TIERS:
            raise ValueError(f"unknown tier {self.tier!r}")


@dataclass
class OrderLine:
    sku: str
    qty: int
    unit_price: Money

    def __post_init__(self):
        if self.qty <= 0:
            raise ValueError("quantity must be positive")

    def gross(self) -> Money:
        return self.unit_price.times(self.qty)


@dataclass
class Order:
    id: str
    customer_id: str
    created: dt.date
    lines: list[OrderLine] = field(default_factory=list)
    status: str = "draft"
    coupon: str | None = None
    quote: dict | None = None  # set when placed: pricing.Quote.as_dict()

    def line(self, sku: str) -> OrderLine | None:
        return next((line for line in self.lines if line.sku == sku), None)

    def units(self) -> int:
        return sum(line.qty for line in self.lines)

    def is_open(self) -> bool:
        return self.status in ("draft", "placed")
