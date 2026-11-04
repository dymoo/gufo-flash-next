"""Checker-only support: drive two copies of the shop package through the same seeded
script and compare everything observable. `shop` is the agent's code, `refshop` the
reference solution, `baseshop` the original repository (for migrations)."""

import datetime as dt
import importlib
import json
import random

CATEGORIES = ["books", "food", "tools", "children", "garden", "toys", "office"]
REGIONS = ["UK", "IE", "DE", "NL", "US"]
TIERS = ["standard", "silver", "gold"]
COUPONS = [None, None, None, "WELCOME10", "FIVEOFF", "SPRING15"]


def mod(pkg, name):
    return importlib.import_module(f"{pkg}.{name}")


class Clock:
    def __init__(self):
        self.today = dt.date(2026, 7, 1)

    def __call__(self):
        return self.today


def skus(seed):
    rng = random.Random(f"skus:{seed}")
    return [f"{rng.choice('ABCDEFGH')}{rng.choice('KLMN')}-{100 + i}" for i in range(12)]


def build(pkg, seed, stock=40, **service_kwargs):
    rng = random.Random(f"build:{seed}")
    Money = mod(pkg, "money").Money
    models = mod(pkg, "models")
    bus = mod(pkg, "events").EventBus()
    inventory = mod(pkg, "inventory").Inventory(bus)
    catalog = {}
    for i, sku in enumerate(skus(seed)):
        catalog[sku] = models.Product(sku, f"Item {i}", rng.choice(CATEGORIES), Money(rng.randrange(99, 20000)),
                                      rng.randrange(50, 6000))
        inventory.receive(sku, stock)
    customers = {f"C-{i}": models.Customer(f"C-{i}", f"Name {i:02d}", rng.choice(REGIONS), rng.choice(TIERS))
                 for i in range(6)}
    clock = Clock()
    service = mod(pkg, "orders").OrderService(catalog, customers, inventory, bus, clock, **service_kwargs)
    return service, clock


def script(seed, n=30, partial=False):
    """Pure data: the same actions for every package under test. Each order is followed by
    exactly one step: stay a draft, place, cancel, ship, or ship part of a line."""
    rng = random.Random(f"script:{seed}")
    names = sorted(skus(seed))
    actions = []
    for _ in range(n):
        lines = [(rng.choice(names), rng.randrange(1, 14)) for _ in range(rng.randrange(1, 5))]
        actions.append(("order", f"C-{rng.randrange(6)}", lines, rng.choice(COUPONS), rng.randrange(0, 90)))
        follow = rng.random()
        if partial and follow < 0.3:
            sku, qty = lines[0]
            actions.append(("ship_line", sku, rng.randrange(1, qty + 2)))
            then = rng.random()
            if then < 0.3:
                actions.append(("cancel",))
            elif then < 0.6:
                actions.append(("ship",))
            elif then < 0.8:
                actions.append(("ship_line", sku, rng.randrange(1, qty + 1)))
        elif follow < 0.45:
            actions.append(("place",))
        elif follow < 0.65:
            actions.append(("cancel",))
        elif follow < 0.85:
            actions.append(("ship",))
        else:
            actions.append(("draft",))
    return actions


def run(pkg, seed, partial=False, make_kwargs=None):
    service, clock = build(pkg, seed, **(make_kwargs(pkg) if make_kwargs else {}))
    outcomes, current = [], None
    for action in script(seed, partial=partial):
        try:
            if action[0] == "order":
                _, customer, lines, coupon, day = action
                clock.today = dt.date(2026, 7, 1) + dt.timedelta(days=day)
                current = None
                current = service.create(customer, coupon=coupon)
                for sku, qty in lines:
                    service.add_line(current.id, sku, qty)
            elif action[0] != "draft":
                if current.status == "draft":
                    service.place(current.id)
                if action[0] == "cancel":
                    service.cancel(current.id)
                elif action[0] == "ship":
                    service.ship(current.id)
                elif action[0] == "ship_line":
                    service.ship_line(current.id, action[1], action[2])
            outcomes.append("ok")
        except Exception as error:  # noqa: BLE001 - the outcome is compared, not raised
            outcomes.append(type(error).__name__)
    return service, outcomes


def money(m):
    return (m.minor, m.currency)


def commands(service):
    argv = [["stock"], ["report", "sales"], ["report", "months"], ["report", "tax"], ["report", "top", "5"],
            ["report", "promotions"]]
    argv += [["stock", sku] for sku in sorted(service.catalog)]
    argv += [["order", "show", oid] for oid in sorted(service.orders)]
    argv += [["customer", cid] for cid in sorted(service.customers)]
    return argv


def snapshot(pkg, service):
    reports, cli, storage = mod(pkg, "reports"), mod(pkg, "cli"), mod(pkg, "storage")
    snap = {
        "orders": {oid: (o.status, o.customer_id, o.created.isoformat(), o.coupon, o.quote,
                         getattr(o, "points", None),
                         [(line.sku, line.qty, money(line.unit_price), getattr(line, "shipped", None))
                          for line in o.lines]) for oid, o in sorted(service.orders.items())},
        "customers": {cid: (c.name, c.region, c.tier, getattr(c, "points", None))
                      for cid, c in sorted(service.customers.items())},
        "stock": service.inventory.snapshot(),
        "events": [(e.kind, e.data) for e in service.bus.history()],
        "sales": {k: money(v) for k, v in reports.sales_by_category(service).items()},
        "months": {k: money(v) for k, v in reports.revenue_by_month(service).items()},
        "top": [(name, money(m)) for name, m in reports.top_customers(service, 10)],
        "tax": {k: (money(a), money(b)) for k, (a, b) in reports.tax_summary(service).items()},
        "cli": [(" ".join(a), cli.run(service, a)) for a in commands(service)],
    }
    text = storage.dump_state(service)
    snap["doc"] = json.loads(text)
    snap["reload"] = json.loads(storage.dump_state(storage.load_state(text)))
    return snap


def diff(a, b, path="state"):
    """The first difference between two snapshots, or None."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b), key=str):
            if key not in a or key not in b:
                return f"{path}[{key!r}] only in {'reference' if key in b else 'yours'}"
            found = diff(a[key], b[key], f"{path}[{key!r}]")
            if found:
                return found
        return None
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return f"{path}: {len(a)} items, expected {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            found = diff(x, y, f"{path}[{i}]")
            if found:
                return found
        return None
    return None if a == b else f"{path}: {a!r}, expected {b!r}"


def same_behaviour(seed, partial=False, make_kwargs=None):
    """make_kwargs(pkg) -> extra OrderService keyword arguments built from that package."""
    mine, my_outcomes = run("shop", seed, partial, make_kwargs)
    ref, ref_outcomes = run("refshop", seed, partial, make_kwargs)
    found = diff(my_outcomes, ref_outcomes, "outcomes") or diff(snapshot("shop", mine), snapshot("refshop", ref))
    assert found is None, f"seed {seed}: {found}"
