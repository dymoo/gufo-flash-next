"""Money in integer minor units (pence, cents).

Amounts never pass through floats. Every operation that can produce a fraction
of a minor unit rounds half up (away from zero for negative amounts), and
allocation distributes leftover minor units so the parts always sum exactly to
the whole.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

CURRENCIES = {"GBP": "£", "EUR": "€", "USD": "$"}
_AMOUNT = re.compile(r"^(-)?([£€$])?(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d{1,2}))?$")


class CurrencyMismatch(ValueError):
    """Arithmetic between two different currencies."""


@dataclass(frozen=True, order=True)
class Money:
    minor: int
    currency: str = "GBP"

    def __post_init__(self):
        if self.currency not in CURRENCIES:
            raise ValueError(f"unknown currency {self.currency!r}")
        if not isinstance(self.minor, int) or isinstance(self.minor, bool):
            raise TypeError("minor units must be an int")

    # ------------------------------------------------------------ creation --

    @classmethod
    def zero(cls, currency: str = "GBP") -> Money:
        return cls(0, currency)

    @classmethod
    def parse(cls, text: str, currency: str = "GBP") -> Money:
        """'12.34', '£1,234.5', '-3' -> Money. A symbol, if present, must match the currency."""
        m = _AMOUNT.match(text.strip())
        if not m:
            raise ValueError(f"not an amount: {text!r}")
        sign, symbol, whole, frac = m.groups()
        if symbol and symbol != CURRENCIES[currency]:
            raise CurrencyMismatch(f"{text!r} is not in {currency}")
        minor = int(whole.replace(",", "")) * 100 + int((frac or "0").ljust(2, "0"))
        return cls(-minor if sign else minor, currency)

    @classmethod
    def from_decimal(cls, value: Decimal | str, currency: str = "GBP") -> Money:
        return cls(_round(Decimal(value) * 100), currency)

    # ---------------------------------------------------------- arithmetic --

    def _same(self, other: Money) -> None:
        if not isinstance(other, Money):
            raise TypeError(f"expected Money, got {type(other).__name__}")
        if other.currency != self.currency:
            raise CurrencyMismatch(f"{self.currency} and {other.currency}")

    def __add__(self, other: Money) -> Money:
        self._same(other)
        return Money(self.minor + other.minor, self.currency)

    def __sub__(self, other: Money) -> Money:
        self._same(other)
        return Money(self.minor - other.minor, self.currency)

    def __neg__(self) -> Money:
        return Money(-self.minor, self.currency)

    def times(self, quantity: int) -> Money:
        """Exact multiplication by a whole quantity."""
        if not isinstance(quantity, int):
            raise TypeError("quantity must be an int")
        return Money(self.minor * quantity, self.currency)

    def scale(self, factor: Decimal | str) -> Money:
        """Multiply by a decimal factor, rounding half up to a whole minor unit."""
        return Money(_round(Decimal(self.minor) * Decimal(factor)), self.currency)

    def percent(self, rate: Decimal | str) -> Money:
        """`rate` percent of this amount, rounded half up ('20' -> 20%)."""
        return self.scale(Decimal(rate) / 100)

    def allocate(self, weights: list[int]) -> list[Money]:
        """Split into parts proportional to non-negative integer weights.

        Each part is first rounded down; the leftover minor units go one each to the
        parts with the largest remainders (earlier parts win ties), so the parts sum
        exactly to this amount. All-zero weights split evenly.
        """
        if not weights:
            raise ValueError("no weights")
        if any(w < 0 for w in weights):
            raise ValueError("weights must be non-negative")
        if not any(weights):
            weights = [1] * len(weights)
        total = sum(weights)
        sign = -1 if self.minor < 0 else 1
        amount = abs(self.minor)
        shares = [amount * w // total for w in weights]
        remainders = [amount * w % total for w in weights]
        leftover = amount - sum(shares)
        order = sorted(range(len(weights)), key=lambda i: (-remainders[i], i))
        for i in order[:leftover]:
            shares[i] += 1
        return [Money(sign * s, self.currency) for s in shares]

    # ------------------------------------------------------------ queries --

    def is_zero(self) -> bool:
        return self.minor == 0

    def is_negative(self) -> bool:
        return self.minor < 0

    def max(self, other: Money) -> Money:
        self._same(other)
        return self if self.minor >= other.minor else other

    def min(self, other: Money) -> Money:
        self._same(other)
        return self if self.minor <= other.minor else other

    # --------------------------------------------------------- formatting --

    def amount(self) -> str:
        """'1234.50' style, no symbol, no thousands separator."""
        sign = "-" if self.minor < 0 else ""
        whole, frac = divmod(abs(self.minor), 100)
        return f"{sign}{whole}.{frac:02d}"

    def format(self) -> str:
        """'£1,234.50' style; negative amounts get a leading minus."""
        sign = "-" if self.minor < 0 else ""
        whole, frac = divmod(abs(self.minor), 100)
        return f"{sign}{CURRENCIES[self.currency]}{whole:,}.{frac:02d}"

    def __str__(self) -> str:
        return self.format()


def _round(value: Decimal) -> int:
    return int(value.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def total(amounts, currency: str = "GBP") -> Money:
    """Sum of an iterable of Money; zero in `currency` when empty."""
    result = Money.zero(currency)
    for amount in amounts:
        result = result + amount
    return result
