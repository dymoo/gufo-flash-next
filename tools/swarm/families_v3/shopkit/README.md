# shopkit

Order management for the web shop: catalog, customers, stock, pricing, orders,
reports and a small command-line front end. Python 3.10+, standard library only.

- `shop/money.py` - integer minor-unit money
- `shop/config.py` - business rules (tax, discounts, coupons, shipping)
- `shop/models.py` - records
- `shop/events.py` - event bus
- `shop/inventory.py` - stock with reservations
- `shop/pricing.py` - quotes
- `shop/orders.py` - order lifecycle
- `shop/storage.py` - JSON persistence
- `shop/reports.py` - reports over placed orders
- `shop/cli.py` - commands

Run the test suite with the `run_tests` tool.
