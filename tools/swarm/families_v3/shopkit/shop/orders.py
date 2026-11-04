"""Order lifecycle: draft -> placed -> shipped, or placed -> cancelled.

The service owns the orders and coordinates inventory, pricing and events. A
draft can be edited freely; placing reserves stock for every line (all or
nothing), prices the order and freezes the quote on it.
"""

from __future__ import annotations

import datetime as dt

from shop import config, pricing
from shop.inventory import OutOfStock
from shop.models import Order, OrderLine


class OrderError(Exception):
    pass


class OrderService:
    def __init__(self, catalog: dict, customers: dict, inventory, bus, clock=None):
        self.catalog = catalog          # sku -> Product
        self.customers = customers      # id -> Customer
        self.inventory = inventory
        self.bus = bus
        self.clock = clock or dt.date.today
        self.orders: dict[str, Order] = {}
        self._next = 1

    # ------------------------------------------------------------ helpers --

    def get(self, order_id: str) -> Order:
        order = self.orders.get(order_id)
        if order is None:
            raise OrderError(f"no order {order_id!r}")
        return order

    def _expect(self, order: Order, *statuses: str) -> None:
        if order.status not in statuses:
            raise OrderError(f"order {order.id} is {order.status}, expected {' or '.join(statuses)}")

    def customer_of(self, order: Order):
        return self.customers[order.customer_id]

    # -------------------------------------------------------------- draft --

    def create(self, customer_id: str, coupon: str | None = None) -> Order:
        if customer_id not in self.customers:
            raise OrderError(f"no customer {customer_id!r}")
        order = Order(f"O-{self._next:05d}", customer_id, self.clock(), coupon=coupon)
        self._next += 1
        self.orders[order.id] = order
        self.bus.publish("order_created", order_id=order.id, customer_id=customer_id)
        return order

    def add_line(self, order_id: str, sku: str, qty: int) -> OrderLine:
        """Add units of a product at its current price; repeated skus merge into one line."""
        order = self.get(order_id)
        self._expect(order, "draft")
        product = self.catalog.get(sku)
        if product is None:
            raise OrderError(f"no product {sku!r}")
        line = order.line(sku)
        if line is None:
            line = OrderLine(sku, qty, product.price)
            order.lines.append(line)
        else:
            line.qty += qty
        return line

    def remove_line(self, order_id: str, sku: str) -> None:
        order = self.get(order_id)
        self._expect(order, "draft")
        if order.line(sku) is None:
            raise OrderError(f"order {order_id} has no line {sku}")
        order.lines = [line for line in order.lines if line.sku != sku]

    def set_coupon(self, order_id: str, code: str | None) -> None:
        order = self.get(order_id)
        self._expect(order, "draft")
        if code is not None and code not in config.COUPONS:
            raise OrderError(f"unknown coupon {code!r}")
        order.coupon = code

    def preview(self, order_id: str) -> pricing.Quote:
        order = self.get(order_id)
        return pricing.quote(order, self.catalog, self.customer_of(order))

    # ---------------------------------------------------------- lifecycle --

    def place(self, order_id: str) -> pricing.Quote:
        order = self.get(order_id)
        self._expect(order, "draft")
        quote = pricing.quote(order, self.catalog, self.customer_of(order))
        done = []
        try:
            for line in order.lines:
                self.inventory.reserve(line.sku, line.qty)
                done.append(line)
        except OutOfStock as error:
            for line in done:
                self.inventory.release(line.sku, line.qty)
            raise OrderError(str(error)) from error
        order.status = "placed"
        order.quote = quote.as_dict()
        self.bus.publish("order_placed", order_id=order.id, customer_id=order.customer_id,
                         total=quote.total.minor)
        return quote

    def cancel(self, order_id: str) -> None:
        order = self.get(order_id)
        self._expect(order, "draft", "placed")
        if order.status == "placed":
            for line in order.lines:
                self.inventory.release(line.sku, line.qty)
        order.status = "cancelled"
        self.bus.publish("order_cancelled", order_id=order.id, customer_id=order.customer_id)

    def ship(self, order_id: str) -> None:
        order = self.get(order_id)
        self._expect(order, "placed")
        for line in order.lines:
            self.inventory.commit(line.sku, line.qty)
        order.status = "shipped"
        self.bus.publish("order_shipped", order_id=order.id, customer_id=order.customer_id)

    # ------------------------------------------------------------ queries --

    def by_customer(self, customer_id: str) -> list[Order]:
        return [o for o in self.orders.values() if o.customer_id == customer_id]

    def by_status(self, status: str) -> list[Order]:
        return [o for o in self.orders.values() if o.status == status]

    def revenue_orders(self) -> list[Order]:
        """Orders that count as revenue: placed or shipped."""
        return [o for o in self.orders.values() if o.status in ("placed", "shipped")]
