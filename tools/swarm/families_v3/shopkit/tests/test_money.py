from decimal import Decimal

from shop.money import CurrencyMismatch, Money, total


def test_parse_and_format():
    assert Money.parse("12.34") == Money(1234)
    assert Money.parse("£1,234.5") == Money(123450)
    assert Money.parse("-3") == Money(-300)
    assert Money(123456).format() == "£1,234.56"
    assert Money(-5, "EUR").format() == "-€0.05"
    assert Money(700).amount() == "7.00"


def test_parse_rejects_garbage():
    for text in ("", "12.345", "1,23", "abc", "€5"):
        try:
            Money.parse(text)
        except ValueError:
            continue
        raise AssertionError(f"parsed {text!r}")


def test_arithmetic():
    assert Money(250) + Money(125) == Money(375)
    assert Money(250) - Money(300) == Money(-50)
    assert Money(199).times(3) == Money(597)
    assert -Money(5) == Money(-5)
    assert total([Money(1), Money(2), Money(3)]) == Money(6)
    assert total([], "EUR") == Money(0, "EUR")


def test_currency_mismatch():
    try:
        Money(1) + Money(1, "EUR")
    except CurrencyMismatch:
        return
    raise AssertionError("mixed currencies were added")


def test_percent():
    assert Money(10000).percent("20") == Money(2000)
    assert Money(8900).percent("19") == Money(1691)
    assert Money(1000).scale(Decimal("0.333")) == Money(333)


def test_allocate_sums():
    parts = Money(1000).allocate([1, 1, 1])
    assert sum(p.minor for p in parts) == 1000
    assert Money(600).allocate([1, 2]) == [Money(200), Money(400)]
    assert Money(0).allocate([3, 4]) == [Money(0), Money(0)]


def test_min_max():
    assert Money(5).max(Money(7)) == Money(7)
    assert Money(5).min(Money(7)) == Money(5)
