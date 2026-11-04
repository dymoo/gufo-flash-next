"""Read-only reports over placed and shipped orders.

Reports read the quote frozen on each order when it was placed, never the
current catalog prices, so a later price change does not rewrite history.
"""

from __future__ import annotations

from collections import defaultdict

from shop.money import Money


def _currency(service) -> str:
    for order in service.revenue_orders():
        return order.quote["currency"]
    return "GBP"


def sales_by_category(service) -> dict[str, Money]:
    """Net sales (after all discounts, before tax) per product category."""
    currency = _currency(service)
    sums = defaultdict(int)
    for order in service.revenue_orders():
        for line in order.quote["lines"]:
            sums[service.catalog[line["sku"]].category] += line["net"]
    return {category: Money(sums[category], currency) for category in sorted(sums)}


def revenue_by_month(service) -> dict[str, Money]:
    """Order totals (including tax and shipping) per creation month, 'YYYY-MM'."""
    currency = _currency(service)
    sums = defaultdict(int)
    for order in service.revenue_orders():
        sums[order.created.strftime("%Y-%m")] += order.quote["total"]
    return {month: Money(sums[month], currency) for month in sorted(sums)}


def top_customers(service, n: int = 3) -> list[tuple[str, Money]]:
    """The n customers with the highest order totals; ties by name."""
    currency = _currency(service)
    sums = defaultdict(int)
    for order in service.revenue_orders():
        sums[order.customer_id] += order.quote["total"]
    ranked = sorted(sums.items(), key=lambda kv: (-kv[1], service.customers[kv[0]].name))
    return [(service.customers[cid].name, Money(minor, currency)) for cid, minor in ranked[:n]]


def tax_summary(service) -> dict[str, tuple[Money, Money]]:
    """rate -> (net, tax) over all order lines, rates sorted numerically."""
    currency = _currency(service)
    net, tax = defaultdict(int), defaultdict(int)
    for order in service.revenue_orders():
        for line in order.quote["lines"]:
            net[line["tax_rate"]] += line["net"]
            tax[line["tax_rate"]] += line["tax"]
    return {rate: (Money(net[rate], currency), Money(tax[rate], currency)) for rate in sorted(net, key=float)}


def stock_rows(inventory, catalog) -> list[list[str]]:
    rows = []
    for sku, (on_hand, reserved, available) in inventory.snapshot().items():
        name = catalog[sku].name if sku in catalog else "?"
        flag = "LOW" if available < inventory.threshold else ""
        rows.append([sku, name, str(on_hand), str(reserved), str(available), flag])
    return rows


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    """Left-aligned text columns; numbers right-aligned."""
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)] if rows else [len(h) for h in headers]

    def cell(value, width):
        text = str(value)
        numeric = text.replace(".", "", 1).replace(",", "").lstrip("-£€$").isdigit()
        return text.rjust(width) if numeric else text.ljust(width)

    lines = [" | ".join(h.ljust(w) for h, w in zip(headers, widths)), "-+-".join("-" * w for w in widths)]
    lines += [" | ".join(cell(v, w) for v, w in zip(row, widths)) for row in rows]
    return "\n".join(line.rstrip() for line in lines)
