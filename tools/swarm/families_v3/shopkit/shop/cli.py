"""Command-line front end. `run(service, argv)` returns the text a command prints.

Commands:
  stock <SKU>                 stock levels of one product
  stock                       the stock table
  order show <ID>             one order with its lines and totals
  customer <ID>               a customer's summary
  report sales|months|tax     sales by category, revenue by month, tax by rate
  report top [N]              the top N customers (default 3)
"""

from __future__ import annotations

from shop import reports
from shop.money import Money


class UsageError(Exception):
    pass


def cmd_stock(service, args: list[str]) -> str:
    inv = service.inventory
    if not args:
        return reports.format_table(["sku", "name", "on hand", "reserved", "available", ""],
                                    reports.stock_rows(inv, service.catalog))
    sku = args[0].upper()
    if sku not in inv.on_hand:
        raise UsageError(f"unknown sku {sku}")
    on_hand, reserved, available = inv.snapshot()[sku]
    return f"{sku}: on hand {on_hand}, reserved {reserved}, available {available}"


def cmd_order(service, args: list[str]) -> str:
    if len(args) != 2 or args[0] != "show":
        raise UsageError("usage: order show <ID>")
    order = service.get(args[1])
    customer = service.customers[order.customer_id]
    out = [f"Order {order.id} ({order.status}) for {customer.name}, created {order.created.isoformat()}"]
    for line in order.lines:
        name = service.catalog[line.sku].name
        out.append(f"  {line.qty} x {line.sku} {name} @ {line.unit_price.format()}")
    if order.quote:
        q, cur = order.quote, order.quote["currency"]
        for label in ("goods", "discount", "net", "tax", "shipping", "total"):
            out.append(f"  {label:<9}{Money(q[label], cur).format():>12}")
    return "\n".join(out)


def cmd_customer(service, args: list[str]) -> str:
    if len(args) != 1:
        raise UsageError("usage: customer <ID>")
    customer = service.customers.get(args[0])
    if customer is None:
        raise UsageError(f"unknown customer {args[0]}")
    orders = [o for o in service.by_customer(customer.id) if o.status in ("placed", "shipped")]
    spent = sum(o.quote["total"] for o in orders)
    currency = orders[0].quote["currency"] if orders else "GBP"
    return (f"{customer.name} ({customer.tier}, {customer.region}): {len(orders)} orders, "
            f"total {Money(spent, currency).format()}")


def cmd_report(service, args: list[str]) -> str:
    if not args:
        raise UsageError("usage: report sales|months|tax|top [N]")
    kind = args[0]
    if kind == "sales":
        rows = [[c, m.format()] for c, m in reports.sales_by_category(service).items()]
        return reports.format_table(["category", "net sales"], rows)
    if kind == "months":
        rows = [[month, m.format()] for month, m in reports.revenue_by_month(service).items()]
        return reports.format_table(["month", "revenue"], rows)
    if kind == "tax":
        rows = [[f"{rate}%", net.format(), tax.format()] for rate, (net, tax) in reports.tax_summary(service).items()]
        return reports.format_table(["rate", "net", "tax"], rows)
    if kind == "top":
        n = int(args[1]) if len(args) > 1 else 3
        rows = [[str(i), name, m.format()] for i, (name, m) in enumerate(reports.top_customers(service, n), 1)]
        return reports.format_table(["#", "customer", "total"], rows)
    raise UsageError(f"unknown report {kind!r}")


COMMANDS = {"stock": cmd_stock, "order": cmd_order, "customer": cmd_customer, "report": cmd_report}


def run(service, argv: list[str]) -> str:
    if not argv or argv[0] not in COMMANDS:
        return "error: commands are " + ", ".join(sorted(COMMANDS))
    try:
        return COMMANDS[argv[0]](service, argv[1:])
    except (UsageError, LookupError, ValueError) as error:
        return f"error: {error}"
