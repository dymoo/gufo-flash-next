from shop import reports
from shop.money import Money
from tests.helpers import sample_service


def test_sales_by_category():
    service, _ = sample_service()
    assert reports.sales_by_category(service) == {
        "books": Money(6095), "children": Money(3008), "food": Money(8459), "garden": Money(1642), "tools": Money(15785)}


def test_revenue_by_month():
    service, _ = sample_service()
    assert reports.revenue_by_month(service) == {"2026-08": Money(15678), "2026-09": Money(23317)}


def test_top_customers():
    service, _ = sample_service()
    assert reports.top_customers(service, 2) == [("Ada Brook", Money(15678)), ("Clara Weiss", Money(12448))]


def test_format_table():
    text = reports.format_table(["a", "bb"], [["x", "1"], ["yy", "22"]])
    assert text.splitlines() == ["a  | bb", "---+---", "x  |  1", "yy | 22"]
