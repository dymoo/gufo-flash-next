"""Stock levels with reservations.

on_hand counts units in the warehouse; reserved counts units promised to placed
orders that have not shipped. available = on_hand - reserved. Placing an order
reserves, cancelling releases, shipping commits (removes the units from both).
"""

from __future__ import annotations

from shop import config


class OutOfStock(Exception):
    def __init__(self, sku: str, wanted: int, available: int):
        super().__init__(f"{sku}: wanted {wanted}, available {available}")
        self.sku, self.wanted, self.available = sku, wanted, available


class Inventory:
    def __init__(self, bus=None, threshold: int | None = None):
        self.bus = bus
        self.threshold = config.LOW_STOCK_THRESHOLD if threshold is None else threshold
        self.on_hand: dict[str, int] = {}
        self.reserved: dict[str, int] = {}

    # ------------------------------------------------------------ queries --

    def available(self, sku: str) -> int:
        return self.on_hand.get(sku, 0) - self.reserved.get(sku, 0)

    def snapshot(self) -> dict[str, tuple[int, int, int]]:
        """sku -> (on_hand, reserved, available), for every sku ever stocked."""
        return {sku: (self.on_hand.get(sku, 0), self.reserved.get(sku, 0), self.available(sku))
                for sku in sorted(set(self.on_hand) | set(self.reserved))}

    def low_stock(self) -> list[str]:
        return [sku for sku in sorted(self.on_hand) if self.available(sku) < self.threshold]

    # ---------------------------------------------------------- movements --

    def receive(self, sku: str, qty: int) -> None:
        if qty <= 0:
            raise ValueError("quantity must be positive")
        self.on_hand[sku] = self.on_hand.get(sku, 0) + qty

    def reserve(self, sku: str, qty: int) -> None:
        if qty <= 0:
            raise ValueError("quantity must be positive")
        before = self.available(sku)
        if qty > before:
            raise OutOfStock(sku, qty, before)
        self.reserved[sku] = self.reserved.get(sku, 0) + qty
        self._check_low(sku, before)

    def release(self, sku: str, qty: int) -> None:
        """Return reserved units to availability (an order was cancelled)."""
        if qty <= 0:
            raise ValueError("quantity must be positive")
        if qty > self.reserved.get(sku, 0):
            raise ValueError(f"{sku}: releasing {qty} but only {self.reserved.get(sku, 0)} reserved")
        self.reserved[sku] -= qty

    def commit(self, sku: str, qty: int) -> None:
        """Ship reserved units: they leave the warehouse."""
        if qty <= 0:
            raise ValueError("quantity must be positive")
        if qty > self.reserved.get(sku, 0):
            raise ValueError(f"{sku}: committing {qty} but only {self.reserved.get(sku, 0)} reserved")
        self.reserved[sku] -= qty
        self.on_hand[sku] -= qty

    def adjust(self, sku: str, counted: int) -> int:
        """Stock-take: set on_hand to the counted units; returns the difference."""
        if counted < self.reserved.get(sku, 0):
            raise ValueError(f"{sku}: counted {counted} is below the {self.reserved.get(sku, 0)} reserved")
        before = self.available(sku)
        diff = counted - self.on_hand.get(sku, 0)
        self.on_hand[sku] = counted
        self._check_low(sku, before)
        return diff

    def _check_low(self, sku: str, before: int) -> None:
        """Publish low_stock when availability crosses below the threshold."""
        after = self.available(sku)
        if self.bus is not None and before >= self.threshold > after:
            self.bus.publish("low_stock", sku=sku, available=after)
