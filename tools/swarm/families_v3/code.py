"""Coding (hard tier): three-item tickets in a ten-module order-management app (`shopkit`).

The repository is the app in families_v3/shopkit plus seeded utility modules
(about 2K lines with tests). Every task has three items. Variants: a feature that
touches four or more modules (loyalty points, partial shipments, promotions), a
storage-format migration that must still read the old format, or a refactor of the
pricing API through every call site - each plus two bugs known only from support
tickets, whose symptoms show in one module and whose causes are in another; or
three such bugs, the first with a failing test. The checker
runs the original visible tests, scenario unit tests and a seeded differential
test that drives the agent's package and the reference solution through the same
random order script and compares everything observable: quotes, stock, events,
reports, CLI output and stored documents.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from families import Call, Step, Task, rng_for
from families.code import FILLER_KINDS, NOUNS, RUNNER, CodeTask, _filler

HERE = Path(__file__).resolve().parent
KINDS = ["feature", "bug", "migration", "refactor"]


def _tree(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): p.read_text() for p in sorted(root.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and not p.name.startswith(".")}


BASE = _tree(HERE / "shopkit")
SUPPORT = (HERE / "shopkit_hidden" / "zz_support.py").read_text()
RUNNER3 = RUNNER.replace("rootdir: utilkit", "rootdir: shopkit")


def run_suite3(files: dict[str, str], timeout: float = 60.0) -> tuple[bool, str]:
    """families.code.run_suite with the shopkit banner (same sandbox: rlimits, uid 65534 when root)."""
    with tempfile.TemporaryDirectory(prefix="swarm-code3-") as d:
        root = Path(d)
        for path, content in files.items():
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        (root / ".runner.py").write_text(RUNNER3)
        for p in [root, *root.rglob("*")]:
            os.chmod(p, 0o777 if p.is_dir() else 0o666)
        try:
            drop = {"user": 65534, "group": 65534, "extra_groups": []} if os.geteuid() == 0 else {}
            done = subprocess.run([sys.executable, "-I", "-B", ".runner.py"], cwd=root, capture_output=True,
                                  text=True, timeout=timeout, **drop)
        except subprocess.TimeoutExpired:
            return False, f"ERROR: the test run exceeded {timeout:.0f} s and was killed."
    out = done.stdout + (("\n" + done.stderr[-2000:]) if done.stderr.strip() else "")
    return done.returncode == 0, out


def apply(files: dict[str, str], patches) -> dict[str, str]:
    files = dict(files)
    for path, old, new in patches:
        assert files[path].count(old) == 1, (path, old[:60])
        files[path] = files[path].replace(old, new)
    return files


def renamed(files: dict[str, str], package: str) -> dict[str, str]:
    """The shop package's sources as another top-level package (imports rewritten)."""
    out = {}
    for path, text in files.items():
        if path.startswith("shop/"):
            out[package + path[4:]] = re.sub(r"\b(from|import) shop\b", rf"\1 {package}", text)
    return out


DONE = ("Change no other observable behaviour: the checker compares events, reports, CLI output and stored "
        "documents with a reference implementation. Do not modify anything under tests/. When you are done, reply "
        "with a one-line summary.")
ROOT_CAUSE = ("Find and fix the root cause in the package code; do not special-case the reports, the CLI or the "
              "tests' data.")

# --------------------------------------------------------------- scenarios --
# Each: kind, prompt, start patches (bugs), solution patches, visible test files (added to the
# repository; they fail before the work and pass after), hidden tests (checker only).

S = {}

S["loyalty"] = dict(
    kind="feature",
    prompt=(
        "Add loyalty points. (1) `Customer` gets an int field `points` (default 0) and `Order` an int field "
        "`points` (default 0). (2) When an order is placed, its customer earns 1 point per whole currency unit "
        "of the order's net amount (the quote's net: after all discounts and the coupon, before tax and "
        "shipping), rounded down; gold-tier customers earn double. Record the points on the order, add them to "
        "the customer, and include them in the existing 'order_placed' event as `points=<n>`. (3) Cancelling a "
        "placed order takes back the points it earned (a customer's points never drop below 0) and sets the "
        "order's points to 0; cancelling a draft changes no points. (4) Storage persists both as a \"points\" "
        "key in each customer record and each order record; records without the key load as 0. The storage "
        "format number stays 1. (5) The CLI `customer <ID>` line ends with `, <n> points`, for example "
        "`Ada Brook (standard, UK): 1 orders, total £106.80, 89 points`. "),
    solution=[
        ("shop/models.py", '    tier: str = "standard"\n', '    tier: str = "standard"\n    points: int = 0\n'),
        ("shop/models.py", "    quote: dict | None = None  # set when placed: pricing.Quote.as_dict()\n",
         "    quote: dict | None = None  # set when placed: pricing.Quote.as_dict()\n    points: int = 0\n"),
        ("shop/orders.py", '''        order.status = "placed"
        order.quote = quote.as_dict()
        self.bus.publish("order_placed", order_id=order.id, customer_id=order.customer_id,
                         total=quote.total.minor)''', '''        order.status = "placed"
        order.quote = quote.as_dict()
        customer = self.customer_of(order)
        order.points = max(quote.net.minor, 0) // 100 * (2 if customer.tier == "gold" else 1)
        customer.points += order.points
        self.bus.publish("order_placed", order_id=order.id, customer_id=order.customer_id,
                         total=quote.total.minor, points=order.points)'''),
        ("shop/orders.py", '''            for line in order.lines:
                self.inventory.release(line.sku, line.qty)
        order.status = "cancelled"''', '''            for line in order.lines:
                self.inventory.release(line.sku, line.qty)
            customer = self.customer_of(order)
            customer.points = max(0, customer.points - order.points)
            order.points = 0
        order.status = "cancelled"'''),
        ("shop/storage.py", '''    return {"id": c.id, "name": c.name, "region": c.region, "tier": c.tier}''',
         '''    return {"id": c.id, "name": c.name, "region": c.region, "tier": c.tier, "points": c.points}'''),
        ("shop/storage.py", '''    return Customer(d["id"], d["name"], d["region"], d.get("tier", "standard"))''',
         '''    return Customer(d["id"], d["name"], d["region"], d.get("tier", "standard"), int(d.get("points", 0)))'''),
        ("shop/storage.py", '''            "quote": o.quote}''', '''            "quote": o.quote, "points": o.points}'''),
        ("shop/storage.py", '''                 d.get("quote"))''', '''                 d.get("quote"), int(d.get("points", 0)))'''),
        ("shop/cli.py", '''            f"total {Money(spent, currency).format()}")''',
         '''            f"total {Money(spent, currency).format()}, {customer.points} points")'''),
    ],
    visible={"tests/test_points.py": '''from shop import cli
from tests.helpers import make_service, place


def test_points_earned_on_place():
    service = make_service()
    order = place(service, "C-1", [("TL-330", 1)])
    assert order.points == 89
    assert service.customers["C-1"].points == 89


def test_cli_shows_points():
    service = make_service()
    place(service, "C-1", [("TL-330", 1)])
    assert cli.run(service, ["customer", "C-1"]) == "Ada Brook (standard, UK): 1 orders, total £106.80, 89 points"
'''},
    hidden='''import json

from shop import cli, storage
from tests.helpers import make_service, place


def test_hidden_gold_double_and_floor():
    service = make_service()
    order = place(service, "C-3", [("TL-330", 1)])
    assert order.points == 160
    assert service.customers["C-3"].points == 160
    small = place(service, "C-1", [("GD-530", 1)], coupon="FIVEOFF")
    assert small.points == 1
    assert service.bus.last("order_placed").data["points"] == 1


def test_hidden_cancel_takes_back():
    service = make_service()
    first = place(service, "C-1", [("TL-330", 1)])
    place(service, "C-1", [("BK-110", 1)])
    service.cancel(first.id)
    assert service.customers["C-1"].points == 12
    assert first.points == 0
    draft = service.create("C-1")
    service.add_line(draft.id, "TL-330", 1)
    service.cancel(draft.id)
    assert service.customers["C-1"].points == 12


def test_hidden_points_never_negative():
    service = make_service()
    order = place(service, "C-2", [("TL-330", 1)])
    service.customers["C-2"].points = 10
    service.cancel(order.id)
    assert service.customers["C-2"].points == 0


def test_hidden_storage_points():
    service = make_service()
    place(service, "C-1", [("TL-330", 1)])
    doc = json.loads(storage.dump_state(service))
    assert [c["points"] for c in doc["customers"]] == [89, 0, 0, 0]
    assert doc["orders"][0]["points"] == 89
    for record in doc["customers"] + doc["orders"]:
        del record["points"]
    loaded = storage.load_state(json.dumps(doc))
    assert loaded.customers["C-1"].points == 0
    assert loaded.orders["O-00001"].points == 0
    assert doc["format"] == 1


def test_hidden_cli_customer():
    service = make_service()
    assert cli.run(service, ["customer", "C-4"]) == "Dara Quinn (standard, IE): 0 orders, total £0.00, 0 points"
''')

S["partial_ship"] = dict(
    kind="feature",
    prompt=(
        "Support partial shipments. (1) `OrderLine` gets an int field `shipped` (default 0): units shipped so "
        "far. (2) Add the order status \"partially_shipped\" to `models.ORDER_STATUSES`. (3) New "
        "`OrderService.ship_line(order_id, sku, qty)`: allowed for placed and partially_shipped orders; qty must "
        "be positive and at most the line's unshipped units, and the order must have that line (otherwise raise "
        "OrderError); it commits the units in inventory, adds them to the line's shipped count, sets the status "
        "to \"shipped\" when every line is fully shipped and to \"partially_shipped\" otherwise, and publishes "
        "\"line_shipped\" with order_id, sku and qty. (4) `ship()` is allowed for placed and partially_shipped "
        "orders, ships every line's remaining units, marks every line fully shipped and publishes only "
        "\"order_shipped\" as before. (5) A partially_shipped order cannot be cancelled (OrderError). (6) "
        "partially_shipped orders count as revenue wherever placed and shipped orders do: revenue_orders, every "
        "report and the CLI customer summary. (7) Storage persists a \"shipped\" key in each order line record; a "
        "line without it loads as fully shipped if its order's status is shipped, else as 0. (8) CLI `order "
        "show`: for a partially_shipped order, each line row ends with ` (shipped <shipped>/<qty>)`. "),
    solution=[
        ("shop/models.py", 'ORDER_STATUSES = ("draft", "placed", "shipped", "cancelled")',
         'ORDER_STATUSES = ("draft", "placed", "partially_shipped", "shipped", "cancelled")'),
        ("shop/models.py", '''    unit_price: Money

    def __post_init__(self):
        if self.qty <= 0:''', '''    unit_price: Money
    shipped: int = 0

    def __post_init__(self):
        if self.qty <= 0:'''),
        ("shop/orders.py", '''    def ship(self, order_id: str) -> None:
        order = self.get(order_id)
        self._expect(order, "placed")
        for line in order.lines:
            self.inventory.commit(line.sku, line.qty)
        order.status = "shipped"''', '''    def ship_line(self, order_id: str, sku: str, qty: int) -> None:
        order = self.get(order_id)
        self._expect(order, "placed", "partially_shipped")
        line = order.line(sku)
        if line is None:
            raise OrderError(f"order {order_id} has no line {sku}")
        if qty <= 0 or qty > line.qty - line.shipped:
            raise OrderError(f"cannot ship {qty} of {sku}: {line.qty - line.shipped} unshipped")
        self.inventory.commit(sku, qty)
        line.shipped += qty
        done = all(line.shipped == line.qty for line in order.lines)
        order.status = "shipped" if done else "partially_shipped"
        self.bus.publish("line_shipped", order_id=order.id, sku=sku, qty=qty)

    def ship(self, order_id: str) -> None:
        order = self.get(order_id)
        self._expect(order, "placed", "partially_shipped")
        for line in order.lines:
            if line.qty > line.shipped:
                self.inventory.commit(line.sku, line.qty - line.shipped)
            line.shipped = line.qty
        order.status = "shipped"'''),
        ("shop/orders.py", '''        return [o for o in self.orders.values() if o.status in ("placed", "shipped")]''',
         '''        return [o for o in self.orders.values() if o.status in ("placed", "partially_shipped", "shipped")]'''),
        ("shop/storage.py",
         '''            "lines": [{"sku": line.sku, "qty": line.qty, "price": line.unit_price.amount()} for line in o.lines],''',
         '''            "lines": [{"sku": line.sku, "qty": line.qty, "price": line.unit_price.amount(), "shipped": line.shipped}
                      for line in o.lines],'''),
        ("shop/storage.py",
         '''    lines = [OrderLine(line["sku"], int(line["qty"]), _money(line["price"], d["currency"])) for line in d["lines"]]''',
         '''    lines = [OrderLine(line["sku"], int(line["qty"]), _money(line["price"], d["currency"]),
                       int(line.get("shipped", line["qty"] if d["status"] == "shipped" else 0))) for line in d["lines"]]'''),
        ("shop/cli.py", '''        out.append(f"  {line.qty} x {line.sku} {name} @ {line.unit_price.format()}")''',
         '''        row = f"  {line.qty} x {line.sku} {name} @ {line.unit_price.format()}"
        if order.status == "partially_shipped":
            row += f" (shipped {line.shipped}/{line.qty})"
        out.append(row)'''),
        ("shop/cli.py", '''    orders = [o for o in service.by_customer(customer.id) if o.status in ("placed", "shipped")]''',
         '''    orders = [o for o in service.by_customer(customer.id) if o.status in ("placed", "partially_shipped", "shipped")]'''),
    ],
    partial=True,
    visible={"tests/test_partial.py": '''from tests.helpers import make_service, place


def test_ship_line_partially():
    service = make_service()
    order = place(service, "C-1", [("TL-330", 3), ("BK-100", 1)])
    service.ship_line(order.id, "TL-330", 2)
    assert order.status == "partially_shipped"
    assert service.inventory.snapshot()["TL-330"] == (28, 1, 27)


def test_partial_counts_as_revenue():
    service = make_service()
    order = place(service, "C-1", [("TL-330", 3), ("BK-100", 1)])
    service.ship_line(order.id, "BK-100", 1)
    assert [o.id for o in service.revenue_orders()] == [order.id]
'''},
    hidden='''import json

from shop import cli, reports, storage
from shop.orders import OrderError
from tests.helpers import make_service, place


def _raises(fn):
    try:
        fn()
    except OrderError:
        return True
    return False


def test_hidden_ship_line_rules():
    service = make_service()
    order = place(service, "C-1", [("TL-330", 3), ("BK-100", 1)])
    assert _raises(lambda: service.ship_line(order.id, "TL-330", 0))
    assert _raises(lambda: service.ship_line(order.id, "TL-330", 4))
    assert _raises(lambda: service.ship_line(order.id, "GD-520", 1))
    draft = service.create("C-2")
    service.add_line(draft.id, "BK-110", 1)
    assert _raises(lambda: service.ship_line(draft.id, "BK-110", 1))
    service.ship_line(order.id, "TL-330", 3)
    assert order.status == "partially_shipped"
    service.ship_line(order.id, "BK-100", 1)
    assert order.status == "shipped"
    assert [e.kind for e in service.bus.history()][-2:] == ["line_shipped", "line_shipped"]
    assert service.bus.last("line_shipped").data == {"order_id": order.id, "sku": "BK-100", "qty": 1}
    assert _raises(lambda: service.ship_line(order.id, "BK-100", 1))


def test_hidden_ship_after_partial():
    service = make_service()
    order = place(service, "C-1", [("TL-330", 3), ("BK-100", 2)])
    service.ship_line(order.id, "TL-330", 1)
    service.ship(order.id)
    assert order.status == "shipped"
    assert [(line.shipped, line.qty) for line in order.lines] == [(3, 3), (2, 2)]
    assert service.inventory.snapshot()["TL-330"] == (27, 0, 27)
    assert service.inventory.snapshot()["BK-100"] == (28, 0, 28)
    assert service.bus.last("order_shipped").data["order_id"] == order.id


def test_hidden_no_cancel_after_partial():
    service = make_service()
    order = place(service, "C-1", [("TL-330", 3)])
    service.ship_line(order.id, "TL-330", 1)
    assert _raises(lambda: service.cancel(order.id))
    assert order.status == "partially_shipped"


def test_hidden_reports_and_cli():
    service = make_service()
    order = place(service, "C-1", [("TL-330", 3), ("BK-100", 1)])
    service.ship_line(order.id, "TL-330", 1)
    assert reports.top_customers(service, 1)[0][0] == "Ada Brook"
    assert cli.run(service, ["customer", "C-1"]).startswith("Ada Brook (standard, UK): 1 orders, total ")
    rows = cli.run(service, ["order", "show", order.id]).splitlines()
    assert rows[1].endswith("(shipped 1/3)") and rows[2].endswith("(shipped 0/1)")
    service.ship(order.id)
    assert "(shipped " not in cli.run(service, ["order", "show", order.id])


def test_hidden_storage_shipped():
    service = make_service()
    a = place(service, "C-1", [("TL-330", 3)])
    service.ship_line(a.id, "TL-330", 2)
    b = place(service, "C-2", [("BK-100", 2)])
    service.ship(b.id)
    place(service, "C-3", [("BK-110", 2)])
    doc = json.loads(storage.dump_state(service))
    assert [line["shipped"] for o in doc["orders"] for line in o["lines"]] == [2, 2, 0]
    for o in doc["orders"]:
        for line in o["lines"]:
            del line["shipped"]
    loaded = storage.load_state(json.dumps(doc))
    assert [line.shipped for o in sorted(loaded.orders) for line in loaded.orders[o].lines] == [0, 2, 0]
''')

S["allocate"] = dict(
    kind="bug",
    prompt=("Finance reconciled last month: for orders that used a coupon, the tax report's net amount per tax rate and the category sales report are off by a penny or two against the order lines."),
    start=[("shop/money.py", "order = sorted(range(len(weights)), key=lambda i: (-remainders[i], i))",
            "order = sorted(range(len(weights)), key=lambda i: (remainders[i], i))")],
    visible={"tests/test_reports.py+": '''

def test_tax_summary():
    service, _ = sample_service()
    assert reports.tax_summary(service) == {
        "0": (Money(6095), Money(0)), "5": (Money(11467), Money(573)), "19": (Money(6885), Money(1308)),
        "20": (Money(8900), Money(1780)), "21": (Money(1642), Money(345))}
'''},
    hidden='''import random

from refshop.money import Money as Ref
from shop.money import Money


def test_hidden_allocate_examples():
    assert Money(1122).allocate([9399, 1824]) == [Money(940), Money(182)]
    assert Money(500).allocate([3145, 7198, 1147]) == [Money(137), Money(313), Money(50)]
    assert Money(1).allocate([1, 1]) == [Money(1), Money(0)]
    assert Money(-7).allocate([1, 1, 1]) == [Money(-3), Money(-2), Money(-2)]
    assert Money(10).allocate([0, 0]) == [Money(5), Money(5)]


def test_hidden_allocate_random():
    rng = random.Random(SEED)
    for _ in range(400):
        amount = rng.randrange(-5000, 5000)
        weights = [rng.randrange(0, 60) for _ in range(rng.randrange(1, 6))]
        assert [m.minor for m in Money(amount).allocate(weights)] == [m.minor for m in Ref(amount).allocate(weights)]
''')

S["release"] = dict(
    kind="bug",
    prompt=('Warehouse stock keeps drifting down: after customers cancel orders, the system shows fewer units on hand than are physically on the shelves.'),
    start=[("shop/inventory.py", '''            raise ValueError(f"{sku}: releasing {qty} but only {self.reserved.get(sku, 0)} reserved")
        self.reserved[sku] -= qty''', '''            raise ValueError(f"{sku}: releasing {qty} but only {self.reserved.get(sku, 0)} reserved")
        self.reserved[sku] -= qty
        self.on_hand[sku] -= qty''')],
    visible={"tests/test_cli.py+": '''

def test_stock_after_cancel():
    service, _ = sample_service()
    assert cli.run(service, ["stock", "GD-520"]) == "GD-520: on hand 30, reserved 0, available 30"
'''},
    hidden='''from shop.inventory import Inventory


def test_hidden_release():
    inv = Inventory()
    inv.receive("AB-1", 10)
    inv.reserve("AB-1", 4)
    inv.release("AB-1", 3)
    assert inv.snapshot()["AB-1"] == (10, 1, 9)
    try:
        inv.release("AB-1", 2)
    except ValueError:
        pass
    else:
        raise AssertionError("released more than reserved")
    inv.commit("AB-1", 1)
    assert inv.snapshot()["AB-1"] == (9, 0, 9)
''')

S["stock_swap"] = dict(
    kind="bug",
    prompt=('Every time the service restarts (it saves its state and loads it again), the stock screen shows reservation numbers that cannot be right.'),
    start=[("shop/storage.py", "    for sku, (on_hand, reserved) in doc.get(\"stock\", {}).items():",
            "    for sku, (reserved, on_hand) in doc.get(\"stock\", {}).items():")],
    visible={"tests/test_cli.py+": '''

def test_stock_after_reload():
    from shop import storage

    service, _ = sample_service()
    loaded = storage.load_state(storage.dump_state(service))
    assert cli.run(loaded, ["stock", "TL-340"]) == "TL-340: on hand 30, reserved 2, available 28"
'''},
    hidden='''import json

from shop import storage
from tests.helpers import sample_service


def test_hidden_stock_document_and_reload():
    service, _ = sample_service()
    doc = json.loads(storage.dump_state(service))
    assert doc["stock"]["BK-100"] == [30, 2]
    assert doc["stock"]["FD-210"] == [18, 0]
    loaded = storage.load_state(storage.dump_state(service))
    assert loaded.inventory.snapshot() == service.inventory.snapshot()
''')

S["rounding"] = dict(
    kind="bug",
    prompt=('A customer says their discount was a penny less than our published rules give; support has seen it on a few orders where the amount ends in half a penny.'),
    start=[("shop/money.py", "from decimal import ROUND_HALF_UP, Decimal", "from decimal import ROUND_HALF_EVEN, Decimal"),
           ("shop/money.py", "rounding=ROUND_HALF_UP", "rounding=ROUND_HALF_EVEN")],
    visible={"tests/test_cli.py+": '''

def test_order_show_half_penny_discount():
    from tests.helpers import make_service, place

    service = make_service()
    order = place(service, "C-2", [("FD-210", 1)])
    assert cli.run(service, ["order", "show", order.id]).splitlines()[3].split() == ["discount", "£0.43"]
'''},
    hidden='''from shop.money import Money


def test_hidden_half_up():
    assert Money(5).percent("10") == Money(1)
    assert Money(25).percent("10") == Money(3)
    assert Money(-25).percent("10") == Money(-3)
    assert Money(850).percent("5") == Money(43)
    assert Money(15).scale("0.5") == Money(8)
    assert Money.from_decimal("0.125") == Money(13)
    assert Money.from_decimal("-0.125") == Money(-13)
''')

S["storage_v2"] = dict(
    kind="migration",
    prompt=(
        "Migrate the storage format from 1 to 2. `dump_state` must write format 2 (set config.STORAGE_FORMAT = "
        "2); `load_state` must read both format 1 and format 2 documents and reject any other format with "
        "StorageError. Format 2 differs from format 1 only in these ways: (a) money is an object "
        "{\"minor\": <int minor units>, \"currency\": \"<code>\"} - a product's \"price\" and an order line's "
        "\"price\"; the separate \"currency\" keys of product and order records are removed; (b) an order's "
        "\"customer\" key is renamed \"customer_id\"; (c) \"stock\" becomes a list of {\"sku\", \"on_hand\", "
        "\"reserved\"} objects sorted by sku. Everything else (keys, the order's frozen quote, next_order, "
        "dates) stays as it is. The record functions (dump_product, load_product, dump_order, load_order, ...) "
        "use format 2; a malformed money object (not a dict with an int \"minor\" and a known currency) raises "
        "StorageError. Loading a format 1 document must give exactly the same state as before. "),
    solution=[
        ("shop/config.py", "STORAGE_FORMAT = 1", "STORAGE_FORMAT = 2"),
        ("shop/storage.py", '''Format 1 (the current format) stores money as decimal strings ("12.34") next
to a "currency" field on the record that owns them, dates as ISO strings, and
an order's frozen quote as minor units. Records are written in a stable order
so that dumps can be diffed.''', '''Format 2 (the current format) stores money as {"minor": int, "currency": code}
objects, dates as ISO strings, stock as a list of {"sku", "on_hand", "reserved"}
records and an order's frozen quote as minor units. Format 1 documents (money as
decimal strings next to a "currency" field on the owning record, the order's
customer under "customer", stock as sku -> [on_hand, reserved]) still load.
Records are written in a stable order so that dumps can be diffed.'''),
        ("shop/storage.py", '''# ---------------------------------------------------------------- records --

def dump_product(p: Product) -> dict:
    return {"sku": p.sku, "name": p.name, "category": p.category, "price": p.price.amount(),
            "currency": p.price.currency, "weight_g": p.weight_g}


def load_product(d: dict) -> Product:
    return Product(d["sku"], d["name"], d["category"], _money(d["price"], d["currency"]), int(d["weight_g"]))''',
         '''def dump_money(m: Money) -> dict:
    return {"minor": m.minor, "currency": m.currency}


def load_money(d) -> Money:
    if not isinstance(d, dict) or not isinstance(d.get("minor"), int) or isinstance(d.get("minor"), bool):
        raise StorageError(f"bad money {d!r}")
    try:
        return Money(d["minor"], d.get("currency"))
    except ValueError as error:
        raise StorageError(f"bad money {d!r}: {error}") from None


# ---------------------------------------------------------------- records --

def dump_product(p: Product) -> dict:
    return {"sku": p.sku, "name": p.name, "category": p.category, "price": dump_money(p.price),
            "weight_g": p.weight_g}


def load_product(d: dict) -> Product:
    return Product(d["sku"], d["name"], d["category"], load_money(d["price"]), int(d["weight_g"]))


def load_product_v1(d: dict) -> Product:
    return Product(d["sku"], d["name"], d["category"], _money(d["price"], d["currency"]), int(d["weight_g"]))'''),
        ("shop/storage.py", '''def dump_order(o: Order) -> dict:
    currency = o.lines[0].unit_price.currency if o.lines else config.DEFAULT_CURRENCY
    return {"id": o.id, "customer": o.customer_id, "created": o.created.isoformat(), "status": o.status,
            "coupon": o.coupon, "currency": currency,
            "lines": [{"sku": line.sku, "qty": line.qty, "price": line.unit_price.amount()} for line in o.lines],
            "quote": o.quote}


def load_order(d: dict) -> Order:''', '''def dump_order(o: Order) -> dict:
    return {"id": o.id, "customer_id": o.customer_id, "created": o.created.isoformat(), "status": o.status,
            "coupon": o.coupon,
            "lines": [{"sku": line.sku, "qty": line.qty, "price": dump_money(line.unit_price)} for line in o.lines],
            "quote": o.quote}


def load_order(d: dict) -> Order:
    lines = [OrderLine(line["sku"], int(line["qty"]), load_money(line["price"])) for line in d["lines"]]
    return Order(d["id"], d["customer_id"], dt.date.fromisoformat(d["created"]), lines, d["status"],
                 d.get("coupon"), d.get("quote"))


def load_order_v1(d: dict) -> Order:'''),
        ("shop/storage.py", '''           "stock": {sku: [inv.on_hand.get(sku, 0), inv.reserved.get(sku, 0)] for sku in inv.snapshot()},''',
         '''           "stock": [{"sku": sku, "on_hand": inv.on_hand.get(sku, 0), "reserved": inv.reserved.get(sku, 0)}
                     for sku in inv.snapshot()],'''),
        ("shop/storage.py", '''    if version != 1:
        raise StorageError(f"unsupported format {version!r}")
    bus = bus or EventBus()
    inventory = Inventory(bus)
    for sku, (on_hand, reserved) in doc.get("stock", {}).items():
        inventory.on_hand[sku] = int(on_hand)
        inventory.reserved[sku] = int(reserved)
    catalog = {p.sku: p for p in map(load_product, doc.get("products", []))}
    customers = {c.id: c for c in map(load_customer, doc.get("customers", []))}
    service = OrderService(catalog, customers, inventory, bus, clock)
    for order in map(load_order, doc.get("orders", [])):''', '''    if version not in (1, 2):
        raise StorageError(f"unsupported format {version!r}")
    bus = bus or EventBus()
    inventory = Inventory(bus)
    if version == 1:
        stock = [(sku, on_hand, reserved) for sku, (on_hand, reserved) in doc.get("stock", {}).items()]
    else:
        stock = [(s["sku"], s["on_hand"], s["reserved"]) for s in doc.get("stock", [])]
    for sku, on_hand, reserved in stock:
        inventory.on_hand[sku] = int(on_hand)
        inventory.reserved[sku] = int(reserved)
    load_p, load_o = (load_product_v1, load_order_v1) if version == 1 else (load_product, load_order)
    catalog = {p.sku: p for p in map(load_p, doc.get("products", []))}
    customers = {c.id: c for c in map(load_customer, doc.get("customers", []))}
    service = OrderService(catalog, customers, inventory, bus, clock)
    for order in map(load_o, doc.get("orders", [])):'''),
    ],
    visible={"tests/test_storage.py": '''import json

from shop import storage
from tests.helpers import sample_service

FORMAT_1 = """{"format": 1, "next_order": 2,
 "customers": [{"id": "C-1", "name": "Ada Brook", "region": "UK", "tier": "standard"}],
 "products": [{"sku": "BK-100", "name": "Atlas of Rivers", "category": "books", "price": "24.99",
               "currency": "GBP", "weight_g": 1200}],
 "stock": {"BK-100": [30, 2]},
 "orders": [{"id": "O-00001", "customer": "C-1", "created": "2026-08-30", "status": "placed", "coupon": null,
             "currency": "GBP", "lines": [{"sku": "BK-100", "qty": 2, "price": "24.99"}], "quote": null}]}"""


def test_round_trip_orders():
    service, _ = sample_service()
    loaded = storage.load_state(storage.dump_state(service))
    assert sorted(loaded.orders) == sorted(service.orders)
    for order_id, order in service.orders.items():
        assert storage.dump_order(loaded.orders[order_id]) == storage.dump_order(order)


def test_document_shape():
    service, _ = sample_service()
    doc = json.loads(storage.dump_state(service))
    assert doc["format"] == 2
    assert doc["products"][0] == {"sku": "BK-100", "name": "Atlas of Rivers", "category": "books",
                                  "price": {"minor": 2499, "currency": "GBP"}, "weight_g": 1200}
    assert doc["stock"][0] == {"sku": "BK-100", "on_hand": 30, "reserved": 2}
    assert doc["orders"][0]["customer_id"] == "C-1"
    assert doc["next_order"] == 6


def test_loads_format_1():
    service = storage.load_state(FORMAT_1)
    assert service.orders["O-00001"].customer_id == "C-1"
    assert service.catalog["BK-100"].price.minor == 2499
    assert service.inventory.snapshot() == {"BK-100": (30, 2, 28)}
    assert json.loads(storage.dump_state(service))["format"] == 2


def test_rejects_unknown_format():
    try:
        storage.load_state('{"format": 99}')
    except storage.StorageError:
        return
    raise AssertionError("loaded an unknown format")
'''},
    hidden='''import json

from tests.zz_support import build, diff, run, snapshot


def test_hidden_format_1_loads_identically():
    import baseshop.storage
    import refshop.storage
    import shop.storage

    for k in range(3):
        old, _ = run("baseshop", SEED * 10 + k)
        text = baseshop.storage.dump_state(old)
        found = diff(snapshot("shop", shop.storage.load_state(text)), snapshot("refshop", refshop.storage.load_state(text)))
        assert found is None, found


def test_hidden_rejects_bad_money_and_formats():
    import shop.storage

    service, _ = run("shop", SEED)
    doc = json.loads(shop.storage.dump_state(service))
    for bad in ("12.34", {"minor": "1234", "currency": "GBP"}, {"minor": 1234, "currency": "XXX"}, {"minor": True,
                                                                                                    "currency": "GBP"}):
        broken = json.loads(json.dumps(doc))
        broken["products"][0]["price"] = bad
        try:
            shop.storage.load_state(json.dumps(broken))
        except shop.storage.StorageError:
            continue
        raise AssertionError(f"loaded price {bad!r}")
    for version in (0, 3, None, "2"):
        try:
            shop.storage.load_state(json.dumps({"format": version}))
        except shop.storage.StorageError:
            continue
        raise AssertionError(f"loaded format {version!r}")
''')

S["pricebook"] = dict(
    kind="refactor",
    prompt=(
        "Refactor pricing so a service can price with its own rules instead of the global config. (1) Add "
        "`pricing.PriceBook`, a frozen dataclass with the fields tax_rates, reduced_rates, tier_discounts, "
        "bulk_quantity, bulk_percent, coupons, shipping_bands, heavy_shipping and free_shipping_over (the "
        "values of the matching config names), and a classmethod `PriceBook.from_config()` that copies the "
        "current config values. (2) Every pricing function takes the book as its last parameter and reads "
        "rules only from it: `tax_rate(category, region, book)`, `shipping_for(weight_g, net, book)`, "
        "`coupon_amount(code, discounted, book)` and `quote(order, catalog, customer, book)`; nothing in "
        "pricing reads config except from_config. (3) `OrderService` takes an optional keyword argument "
        "`book` (default: `PriceBook.from_config()` taken when the service is created), stores it as "
        "`service.book` and uses it for preview, place and set_coupon (which must reject codes that are not in "
        "the service's book). (4) `storage.load_state` takes an optional keyword argument `book` and passes it "
        "to the service. Update every call site. Behaviour with the default book must not change. "),
    solution=[
        ("shop/pricing.py", '''class PricingError(ValueError):
    pass
''', '''class PricingError(ValueError):
    pass


@dataclass(frozen=True)
class PriceBook:
    """The pricing rules one service prices with."""
    tax_rates: dict
    reduced_rates: dict
    tier_discounts: dict
    bulk_quantity: int
    bulk_percent: str
    coupons: dict
    shipping_bands: list
    heavy_shipping: int
    free_shipping_over: int

    @classmethod
    def from_config(cls) -> PriceBook:
        return cls(dict(config.TAX_RATES), dict(config.REDUCED_RATES), dict(config.TIER_DISCOUNTS),
                   config.BULK_QUANTITY, config.BULK_PERCENT, dict(config.COUPONS), list(config.SHIPPING_BANDS),
                   config.HEAVY_SHIPPING, config.FREE_SHIPPING_OVER)
'''),
        ("shop/pricing.py", '''def tax_rate(category: str, region: str) -> str:
    if region not in config.TAX_RATES:
        raise PricingError(f"no tax rate for region {region!r}")
    return config.REDUCED_RATES.get(category, config.TAX_RATES[region])


def shipping_for(weight_g: int, net: Money) -> Money:
    if net.minor >= config.FREE_SHIPPING_OVER:
        return Money.zero(net.currency)
    for max_grams, price in config.SHIPPING_BANDS:
        if weight_g <= max_grams:
            return Money(price, net.currency)
    return Money(config.HEAVY_SHIPPING, net.currency)


def coupon_amount(code: str | None, discounted: Money) -> Money:
    if not code:
        return Money.zero(discounted.currency)
    if code not in config.COUPONS:
        raise PricingError(f"unknown coupon {code!r}")
    kind, value = config.COUPONS[code]''', '''def tax_rate(category: str, region: str, book: PriceBook) -> str:
    if region not in book.tax_rates:
        raise PricingError(f"no tax rate for region {region!r}")
    return book.reduced_rates.get(category, book.tax_rates[region])


def shipping_for(weight_g: int, net: Money, book: PriceBook) -> Money:
    if net.minor >= book.free_shipping_over:
        return Money.zero(net.currency)
    for max_grams, price in book.shipping_bands:
        if weight_g <= max_grams:
            return Money(price, net.currency)
    return Money(book.heavy_shipping, net.currency)


def coupon_amount(code: str | None, discounted: Money, book: PriceBook) -> Money:
    if not code:
        return Money.zero(discounted.currency)
    if code not in book.coupons:
        raise PricingError(f"unknown coupon {code!r}")
    kind, value = book.coupons[code]'''),
        ("shop/pricing.py", '''def quote(order, catalog: dict, customer) -> Quote:
    """Price an order for a customer. `catalog` maps sku -> Product."""
    if not order.lines:
        raise PricingError("order has no lines")
    currency = order.lines[0].unit_price.currency
    tier = config.TIER_DISCOUNTS[customer.tier]''', '''def quote(order, catalog: dict, customer, book: PriceBook) -> Quote:
    """Price an order for a customer with a price book. `catalog` maps sku -> Product."""
    if not order.lines:
        raise PricingError("order has no lines")
    currency = order.lines[0].unit_price.currency
    tier = book.tier_discounts[customer.tier]'''),
        ("shop/pricing.py", '''        bulk = gross.percent(config.BULK_PERCENT) if line.qty >= config.BULK_QUANTITY else Money.zero(currency)''',
         '''        bulk = gross.percent(book.bulk_percent) if line.qty >= book.bulk_quantity else Money.zero(currency)'''),
        ("shop/pricing.py", '''    coupon = coupon_amount(order.coupon, discounted)''', '''    coupon = coupon_amount(order.coupon, discounted, book)'''),
        ("shop/pricing.py", '''        rate = tax_rate(product.category, customer.region)''',
         '''        rate = tax_rate(product.category, customer.region, book)'''),
        ("shop/pricing.py", '''    shipping = shipping_for(weight, net)''', '''    shipping = shipping_for(weight, net, book)'''),
        ("shop/orders.py", '''    def __init__(self, catalog: dict, customers: dict, inventory, bus, clock=None):''',
         '''    def __init__(self, catalog: dict, customers: dict, inventory, bus, clock=None, book=None):'''),
        ("shop/orders.py", '''        self.clock = clock or dt.date.today
''', '''        self.clock = clock or dt.date.today
        self.book = book if book is not None else pricing.PriceBook.from_config()
'''),
        ("shop/orders.py", '''        if code is not None and code not in config.COUPONS:''',
         '''        if code is not None and code not in self.book.coupons:'''),
        ("shop/orders.py", '''        return pricing.quote(order, self.catalog, self.customer_of(order))''',
         '''        return pricing.quote(order, self.catalog, self.customer_of(order), self.book)'''),
        ("shop/orders.py", '''        quote = pricing.quote(order, self.catalog, self.customer_of(order))''',
         '''        quote = pricing.quote(order, self.catalog, self.customer_of(order), self.book)'''),
        ("shop/orders.py", "from shop import config, pricing\n", "from shop import pricing\n"),
        ("shop/storage.py", '''def load_state(text: str, bus: EventBus | None = None, clock=None) -> OrderService:''',
         '''def load_state(text: str, bus: EventBus | None = None, clock=None, book=None) -> OrderService:'''),
        ("shop/storage.py", '''    service = OrderService(catalog, customers, inventory, bus, clock)''',
         '''    service = OrderService(catalog, customers, inventory, bus, clock, book=book)'''),
    ],
    visible={"tests/test_pricebook.py": '''import dataclasses

from shop.orders import OrderService
from shop.pricing import PriceBook
from tests.helpers import make_service, place


def test_service_uses_its_book():
    base = make_service()
    book = PriceBook.from_config()
    book = dataclasses.replace(book, tax_rates={**book.tax_rates, "UK": "10"})
    service = OrderService(base.catalog, base.customers, base.inventory, base.bus, base.clock, book=book)
    order = place(service, "C-1", [("TL-330", 1)])
    assert order.quote["tax"] == 890
'''},
    hidden='''import dataclasses
import random

from shop import config, pricing, storage
from shop.orders import OrderError, OrderService
from tests.helpers import make_service, place
from tests.zz_support import same_behaviour


def random_book(pkg, seed):
    import importlib

    rng = random.Random(f"book:{seed}")
    PriceBook = importlib.import_module(f"{pkg}.pricing").PriceBook
    return PriceBook(
        tax_rates={r: str(rng.randrange(0, 30)) for r in ["UK", "IE", "DE", "NL", "US"]},
        reduced_rates={c: str(rng.randrange(0, 10)) for c in rng.sample(["books", "food", "children", "toys"], 2)},
        tier_discounts={"standard": "0", "silver": str(rng.randrange(1, 10)), "gold": str(rng.randrange(5, 25))},
        bulk_quantity=rng.randrange(3, 15), bulk_percent=str(rng.randrange(1, 9)),
        coupons={"WELCOME10": ("percent", str(rng.randrange(5, 30))), "FIVEOFF": ("fixed", rng.randrange(100, 2000)),
                 "SPRING15": ("percent", "15")},
        shipping_bands=[(rng.randrange(500, 2000), rng.randrange(100, 500)),
                        (rng.randrange(4000, 9000), rng.randrange(600, 900))],
        heavy_shipping=rng.randrange(1000, 4000), free_shipping_over=rng.randrange(3000, 20000))


def test_hidden_random_books():
    for k in range(3):
        same_behaviour(SEED * 7 + k, make_kwargs=lambda pkg, k=k: {"book": random_book(pkg, SEED * 7 + k)})


def test_hidden_book_is_independent_of_config():
    service = make_service()
    old = dict(config.TAX_RATES)
    try:
        config.TAX_RATES["UK"] = "50"
        order = place(service, "C-1", [("TL-330", 1)])
        assert order.quote["tax"] == 1780
        book = dataclasses.replace(service.book, tax_rates={**service.book.tax_rates, "UK": "10"})
        assert pricing.tax_rate("tools", "UK", book) == "10"
    finally:
        config.TAX_RATES.clear()
        config.TAX_RATES.update(old)


def test_hidden_coupons_come_from_the_book():
    base = make_service()
    book = dataclasses.replace(base.book, coupons={"ONLYHERE": ("fixed", 100)})
    service = OrderService(base.catalog, base.customers, base.inventory, base.bus, base.clock, book=book)
    order = service.create("C-1")
    service.add_line(order.id, "TL-330", 1)
    service.set_coupon(order.id, "ONLYHERE")
    try:
        service.set_coupon(order.id, "WELCOME10")
    except OrderError:
        pass
    else:
        raise AssertionError("accepted a coupon that is not in the book")
    assert service.place(order.id).net.minor == 8800


def test_hidden_load_state_book():
    base = make_service()
    book = dataclasses.replace(base.book, tax_rates={**base.book.tax_rates, "UK": "10"})
    loaded = storage.load_state(storage.dump_state(base), book=book)
    assert loaded.book is book
    assert place(loaded, "C-1", [("TL-330", 1)]).quote["tax"] == 890
''')

PROMO_CONFIG = (
    "shop/config.py", '''# Version written by storage.dump_state.
STORAGE_FORMAT = 1''', '''# Version written by storage.dump_state.
STORAGE_FORMAT = 1

# Promotions from the trading team (see the pricing module). Dates are inclusive and refer to the order's
# created date. "multibuy": buy `buy` units of `sku` and get `free` more free; "category": `percent` off.
PROMOTIONS = [
    {"id": "TOOLS-15", "kind": "category", "category": "tools", "percent": "15", "start": "2026-10-01",
     "end": "2026-11-30"},
    {"id": "OIL-3FOR2", "kind": "multibuy", "sku": "FD-210", "buy": 2, "free": 1, "start": "2026-10-01",
     "end": "2026-12-31"},
    {"id": "BOOKS-10", "kind": "category", "category": "books", "percent": "10", "start": "2026-10-15",
     "end": "2026-12-31"},
    {"id": "TRAY-5FOR4", "kind": "multibuy", "sku": "GD-530", "buy": 4, "free": 1, "start": "2026-10-01",
     "end": "2026-10-31"},
    {"id": "GARDEN-20", "kind": "category", "category": "garden", "percent": "20", "start": "2026-10-10",
     "end": "2026-10-31"},
]''')

S["promotions"] = dict(
    kind="feature",
    prompt=(
        "Implement the trading team's promotions (config.PROMOTIONS; the data is already there). (1) A promotion "
        "applies to an order whose created date is between its start and end dates, inclusive. (2) multibuy: on a "
        "line of that sku, every `buy` + `free` units include `free` units at no charge: the discount is "
        "(qty // (buy + free)) x free x unit price. (3) category: `percent` off the gross of each line in that "
        "category, rounded half up like Money.percent. (4) A line gets at most one promotion: the applicable one with "
        "the largest discount on that line; if two give the same discount, the one listed first in "
        "config.PROMOTIONS; a zero discount is not a promotion. (5) The promotion comes off first: the bulk discount "
        "applies to the line's gross minus the promotion, and the tier discount to what is left after that. (6) "
        "Coupons cannot be combined with promotions: if any line of an order has a promotion, the order's coupon "
        "gives no discount (the code stays on the order). (7) `pricing.LineQuote` gets `promotion` (Money) and "
        "`promotion_id` (str or None); the line's `discount` includes the promotion; each line of "
        "`Quote.as_dict()` gains \"promotion\" (minor units) and \"promotion_id\". (8) New "
        "`reports.promotion_summary(service)`: promotion id -> total promotion discount (Money) over placed and "
        "shipped orders, from their frozen quotes, sorted by id, omitting promotions never used. (9) CLI `report "
        "promotions` prints `reports.format_table([\"promotion\", \"discount\"], rows)` with formatted amounts. "
        "(10) The 'order_placed' event gains `promotions=<sorted list of the distinct promotion ids applied>`."),
    start=[PROMO_CONFIG],
    solution=[
        PROMO_CONFIG,
        ("shop/pricing.py", '''    net: Money           # gross - discount - coupon
    tax_rate: str
    tax: Money
''', '''    net: Money           # gross - discount - coupon
    tax_rate: str
    tax: Money
    promotion: Money = Money(0)       # part of discount
    promotion_id: str | None = None
'''),
        ("shop/pricing.py", '''                "lines": [{"sku": q.sku, "qty": q.qty, "net": q.net.minor, "tax_rate": q.tax_rate,
                           "tax": q.tax.minor} for q in self.lines]}''',
         '''                "lines": [{"sku": q.sku, "qty": q.qty, "net": q.net.minor, "tax_rate": q.tax_rate,
                           "tax": q.tax.minor, "promotion": q.promotion.minor, "promotion_id": q.promotion_id}
                          for q in self.lines]}'''),
        ("shop/pricing.py", '''def quote(order, catalog: dict, customer) -> Quote:''', '''def active_promotions(day) -> list[dict]:
    return [p for p in config.PROMOTIONS if p["start"] <= day.isoformat() <= p["end"]]


def promotion_for(line, product, promotions: list[dict]) -> tuple[Money, str | None]:
    """The best applicable promotion on a line; ties go to the one listed first."""
    best, best_id = Money.zero(line.unit_price.currency), None
    for p in promotions:
        if p["kind"] == "multibuy" and p["sku"] == line.sku:
            discount = line.unit_price.times(line.qty // (p["buy"] + p["free"]) * p["free"])
        elif p["kind"] == "category" and p["category"] == product.category:
            discount = line.gross().percent(p["percent"])
        else:
            continue
        if discount.minor > best.minor:
            best, best_id = discount, p["id"]
    return best, best_id


def quote(order, catalog: dict, customer) -> Quote:'''),
        ("shop/pricing.py", '''    tier = config.TIER_DISCOUNTS[customer.tier]
    staged = []''', '''    tier = config.TIER_DISCOUNTS[customer.tier]
    promotions = active_promotions(order.created)
    staged = []'''),
        ("shop/pricing.py", '''        gross = line.gross()
        bulk = gross.percent(config.BULK_PERCENT) if line.qty >= config.BULK_QUANTITY else Money.zero(currency)
        tiered = (gross - bulk).percent(tier)
        staged.append((line, product, gross, bulk + tiered))
    discounted = total((g - d for _, _, g, d in staged), currency)
    coupon = coupon_amount(order.coupon, discounted)
    shares = coupon.allocate([(g - d).minor for _, _, g, d in staged])
    lines = []
    for (line, product, gross, discount), share in zip(staged, shares):
        net = gross - discount - share
        rate = tax_rate(product.category, customer.region)
        lines.append(LineQuote(line.sku, line.qty, gross, discount, share, net, rate, net.percent(rate)))''',
         '''        gross = line.gross()
        promo, promo_id = promotion_for(line, product, promotions)
        rest = gross - promo
        bulk = rest.percent(config.BULK_PERCENT) if line.qty >= config.BULK_QUANTITY else Money.zero(currency)
        tiered = (rest - bulk).percent(tier)
        staged.append((line, product, gross, promo + bulk + tiered, promo, promo_id))
    discounted = total((g - d for _, _, g, d, _, _ in staged), currency)
    promoted = any(pid for *_, pid in staged)
    coupon = Money.zero(currency) if promoted else coupon_amount(order.coupon, discounted)
    shares = coupon.allocate([(g - d).minor for _, _, g, d, _, _ in staged])
    lines = []
    for (line, product, gross, discount, promo, promo_id), share in zip(staged, shares):
        net = gross - discount - share
        rate = tax_rate(product.category, customer.region)
        lines.append(LineQuote(line.sku, line.qty, gross, discount, share, net, rate, net.percent(rate), promo,
                               promo_id))'''),
        ("shop/reports.py", '''def stock_rows(inventory, catalog) -> list[list[str]]:''', '''def promotion_summary(service) -> dict[str, Money]:
    """Promotion id -> total promotion discount over placed and shipped orders."""
    currency = _currency(service)
    sums = defaultdict(int)
    for order in service.revenue_orders():
        for line in order.quote["lines"]:
            if line.get("promotion_id"):
                sums[line["promotion_id"]] += line["promotion"]
    return {pid: Money(sums[pid], currency) for pid in sorted(sums)}


def stock_rows(inventory, catalog) -> list[list[str]]:'''),
        ("shop/cli.py", '''    if kind == "top":''', '''    if kind == "promotions":
        rows = [[pid, m.format()] for pid, m in reports.promotion_summary(service).items()]
        return reports.format_table(["promotion", "discount"], rows)
    if kind == "top":'''),
        ("shop/orders.py", '''        self.bus.publish("order_placed", order_id=order.id, customer_id=order.customer_id,
                         total=quote.total.minor)''', '''        self.bus.publish("order_placed", order_id=order.id, customer_id=order.customer_id,
                         total=quote.total.minor,
                         promotions=sorted({q.promotion_id for q in quote.lines if q.promotion_id}))'''),
    ],
    visible={"tests/test_promotions.py": '''import datetime as dt

from shop import reports
from shop.money import Money
from tests.helpers import Clock, make_service, place


def test_category_promotion():
    service = make_service(clock=Clock(dt.date(2026, 10, 5)))
    order = place(service, "C-1", [("TL-330", 1)])
    assert order.quote["lines"][0]["promotion_id"] == "TOOLS-15"
    assert order.quote["lines"][0]["promotion"] == 1335


def test_multibuy_and_summary():
    service = make_service(clock=Clock(dt.date(2026, 10, 5)))
    place(service, "C-1", [("FD-210", 7)])
    assert reports.promotion_summary(service) == {"OIL-3FOR2": Money(1700)}
'''},
    hidden='''import datetime as dt
import random

from shop import cli, config, reports
from shop.money import Money
from tests.helpers import Clock, make_service, place
from tests.zz_support import same_behaviour, skus


def promo_quote(day, customer, lines, coupon=None):
    service = make_service(clock=Clock(day))
    return service, place(service, customer, lines, coupon=coupon)


def test_hidden_date_window():
    _, before = promo_quote(dt.date(2026, 9, 30), "C-1", [("TL-330", 1)])
    _, first = promo_quote(dt.date(2026, 10, 1), "C-1", [("TL-330", 1)])
    _, last = promo_quote(dt.date(2026, 11, 30), "C-1", [("TL-330", 1)])
    _, after = promo_quote(dt.date(2026, 12, 1), "C-1", [("TL-330", 1)])
    assert [q.quote["lines"][0]["promotion_id"] for q in (before, first, last, after)] == [None, "TOOLS-15", "TOOLS-15", None]


def test_hidden_multibuy_units():
    _, order = promo_quote(dt.date(2026, 10, 5), "C-1", [("FD-210", 8)])
    assert order.quote["lines"][0]["promotion"] == 1700
    _, order = promo_quote(dt.date(2026, 10, 5), "C-1", [("FD-210", 2)])
    assert order.quote["lines"][0]["promotion_id"] is None


def test_hidden_best_promotion_wins():
    _, order = promo_quote(dt.date(2026, 10, 20), "C-1", [("GD-530", 5)])  # 5-for-4 640 vs 20% of 3200 = 640
    assert order.quote["lines"][0]["promotion_id"] == "TRAY-5FOR4"
    _, order = promo_quote(dt.date(2026, 10, 20), "C-1", [("GD-530", 6)])  # 640 vs 768
    assert order.quote["lines"][0]["promotion_id"] == "GARDEN-20"
    assert order.quote["lines"][0]["promotion"] == 768


def test_hidden_promotion_before_bulk_and_tier():
    _, order = promo_quote(dt.date(2026, 10, 5), "C-3", [("TL-340", 10)])
    line = order.quote["lines"][0]
    gross, promo = 39990, 5999
    bulk = round((gross - promo) * 0.03)
    tier = round((gross - promo - bulk) * 0.10)
    assert line["promotion"] == promo
    assert order.quote["discount"] == promo + bulk + tier


def test_hidden_coupon_not_combined():
    _, order = promo_quote(dt.date(2026, 10, 5), "C-1", [("TL-330", 1), ("BK-100", 1)], coupon="WELCOME10")
    assert order.quote["discount"] == 1335 and order.coupon == "WELCOME10"
    _, plain = promo_quote(dt.date(2026, 10, 5), "C-1", [("BK-100", 1)], coupon="WELCOME10")
    assert plain.quote["discount"] == 250


def test_hidden_report_cli_event():
    service = make_service(clock=Clock(dt.date(2026, 10, 20)))
    place(service, "C-1", [("TL-330", 1), ("BK-110", 2)])
    place(service, "C-2", [("FD-210", 3)])
    assert reports.promotion_summary(service) == {"BOOKS-10": Money(255), "OIL-3FOR2": Money(850),
                                                  "TOOLS-15": Money(1335)}
    assert cli.run(service, ["report", "promotions"]).splitlines()[2:] == [
        "BOOKS-10  |    £2.55", "OIL-3FOR2 |    £8.50", "TOOLS-15  |   £13.35"]
    assert service.bus.history("order_placed")[0].data["promotions"] == ["BOOKS-10", "TOOLS-15"]


def test_hidden_random_promotions():
    import refshop.config

    rng = random.Random(SEED)
    names = skus(SEED)
    promos = [{"id": f"P{i}", "kind": "multibuy", "sku": rng.choice(names), "buy": rng.randrange(1, 4),
               "free": rng.randrange(1, 3), "start": "2026-07-01", "end": "2026-09-30"} for i in range(3)]
    promos += [{"id": f"C{i}", "kind": "category", "category": c, "percent": str(rng.randrange(5, 30)),
                "start": "2026-08-01", "end": "2026-09-15"} for i, c in enumerate(["books", "tools", "food"])]
    saved = (config.PROMOTIONS, refshop.config.PROMOTIONS)
    try:
        config.PROMOTIONS = refshop.config.PROMOTIONS = promos
        same_behaviour(SEED + 11)
    finally:
        config.PROMOTIONS, refshop.config.PROMOTIONS = saved
''')

POOLS = {kind: sorted(name for name, s in S.items() if s["kind"] == kind) for kind in KINDS}
INCOMPATIBLE = {("storage_v2", "stock_swap")}  # the bug's hidden test pins storage format 1
# Every task is a three-item ticket: a feature, migration or refactor plus two bugs known only from support
# reports, or three bugs (the first with a failing visible test, the others known only from their reports).
VARIANTS = ["feature", "migration", "refactor", "bug"]
PAIRS = [(a, b) for i, a in enumerate(POOLS["bug"]) for b in POOLS["bug"][i + 1:]]
COMBOS = {kind: [(a, *pair) for a in POOLS[kind] for pair in PAIRS
                 if a not in pair and not any((a, b) in INCOMPATIBLE for b in pair)] for kind in VARIANTS}

GENERIC = '''from tests.zz_support import same_behaviour

SEED = {seed}


def test_hidden_same_behaviour_1():
    same_behaviour(SEED, partial={partial})


def test_hidden_same_behaviour_2():
    same_behaviour(SEED + 1, partial={partial})


def test_hidden_same_behaviour_3():
    same_behaviour(SEED + 2, partial={partial})
'''

UTIL = {"textutil": "Text helpers shared by the shop's import and export jobs.",
        "datautil": "Data helpers shared by the shop's batch jobs."}


class Code3Task(Task):
    family = "code3"
    max_turns = 50
    tools = CodeTask.tools
    latency = CodeTask.latency
    # the file tools are the v2 coding agent's
    _path = CodeTask._path
    _writable = CodeTask._writable
    tool_list_files = CodeTask.tool_list_files
    tool_read_file = CodeTask.tool_read_file
    tool_grep = CodeTask.tool_grep
    tool_edit_file = CodeTask.tool_edit_file
    tool_write_file = CodeTask.tool_write_file

    def __init__(self, index: int, seed: int, combo: tuple[str, ...] | None = None):
        super().__init__(index, seed)
        rng = rng_for(self.family, seed, index)
        self.tier = "hard"
        self.kind = VARIANTS[index % 4]
        self.scenarios = rng.choice(COMBOS[self.kind])
        if combo is not None:  # tests: every combination
            self.scenarios = combo
        self.scenario = "+".join(self.scenarios)
        first, *bugs = (S[name] for name in self.scenarios)
        base = dict(BASE)
        base.update(self._fillers(rng))
        start = apply(base, first.get("start", []) + [p for bug in bugs for p in bug["start"]])
        solution = apply(base, first.get("solution", []))
        for path, text in first["visible"].items():  # the bug's own symptom test is withheld
            if path.endswith("+"):
                path = path[:-1]
                start[path] = start[path] + text
                solution[path] = solution[path] + text
            else:
                start[path] = solution[path] = text
        self.files, self.reference = start, solution
        self.base = base
        self.check_seed = rng.randrange(1, 10**6)
        reports = " ".join(f"({i + 2}) Support ticket: {bug['prompt'].strip()}" for i, bug in enumerate(bugs))
        failing = " The test suite fails as well." if self.kind == "bug" else ""
        items = f"(1) {first['prompt'].strip()}{failing} {reports} For the support tickets, {ROOT_CAUSE[0].lower()}{ROOT_CAUSE[1:]}"
        self.prompt = f"{self.header()} Repository: shopkit (see README.md). Three items. {items} {DONE}"

    def _fillers(self, rng) -> dict[str, str]:
        """Two utility modules of correct helper functions with visible tests: realistic bulk."""
        pairs = [(k, n) for k in FILLER_KINDS for n in NOUNS]
        rng.shuffle(pairs)
        files, k = {}, 0
        for module, doc in UTIL.items():
            units = [_filler(kind, noun, rng) for kind, noun in pairs[k:k + 30]]
            k += 30
            src = f'"""{doc}"""\n\nimport re\n\n' + "\n".join(u[0] for u in units)
            files[f"shop/{module}.py"] = src
            tests = "\n\n".join(f"def test_{name}():\n" + "".join(f"    assert {a}\n" for a in asserts)
                                for _, asserts, name in units)
            files[f"tests/test_{module}.py"] = (f"from shop.{module} import (\n    "
                                                + ",\n    ".join(sorted(u[2] for u in units)) + ",\n)\n\n\n" + tests)
        return files

    def tool_run_tests(self) -> str:
        _, out = run_suite3(dict(self.files))
        return out[-12000:]

    def check(self, final: str) -> tuple[bool, str]:
        specs = [S[name] for name in self.scenarios]
        files = {p: c for p, c in self.files.items() if not p.startswith(("tests/", "refshop/", "baseshop/"))}
        files.update({p: c for p, c in self.reference.items() if p.startswith("tests/")})
        files.update(renamed(self.reference, "refshop"))
        files.update(renamed(self.base, "baseshop"))
        files["tests/zz_support.py"] = SUPPORT
        hidden = "\n\n".join(spec["hidden"] for spec in specs).replace("SEED", str(self.check_seed))
        partial = any(spec.get("partial") for spec in specs)
        files["tests/test_zz_hidden.py"] = GENERIC.format(seed=self.check_seed, partial=partial) + "\n\n" + hidden
        ok, out = run_suite3(files)
        failed = [line[7:].split(" - ")[0] for line in out.splitlines() if line.startswith("FAILED ")]
        self.requirements = self._requirements(out)
        return ok, "all tests pass" if ok else ("failing: " + ", ".join(failed[:6]) if failed else out[-300:])

    def _requirements(self, out: str) -> dict[str, bool]:
        """Per requirement: the original visible tests, each scenario test, and the differential test."""
        passed = {m[0]: m[1] == "PASSED" for m in re.findall(r"^(tests/\S+::\w+) (PASSED|FAILED)$", out, re.M)}
        broken = set(re.findall(r"^ERROR collecting (tests/\S+)$", out, re.M))
        ok = lambda path, name: path not in broken and passed.get(f"{path}::{name}", False)  # noqa: E731
        visible = [k for k in passed if not k.startswith("tests/test_zz_hidden.py")]
        result = {"visible_tests": not broken - {"tests/test_zz_hidden.py"} and all(passed[k] for k in visible)}
        hidden = "tests/test_zz_hidden.py"
        result["differential"] = all(ok(hidden, f"test_hidden_same_behaviour_{i}") for i in (1, 2, 3))
        for name in self.scenarios:
            for test in re.findall(r"^def (test_hidden_\w+)", S[name]["hidden"], re.M):
                result[f"{S[name]['kind']}.{name}.{test[len('test_hidden_'):]}"] = ok(hidden, test)
        return result

    def oracle(self) -> list[Step]:
        changed = sorted(p for p in self.reference if self.reference[p] != self.files.get(p))
        return [Step(calls=[Call("list_files", {}), Call("run_tests", {})]),
                Step(calls=[Call("read_file", {"path": p}) for p in changed]),
                Step(calls=[Call("write_file", {"path": p, "content": self.reference[p]}) for p in changed]),
                Step(calls=[Call("run_tests", {})]),
                Step(content=f"Done ({self.scenario}): changed {', '.join(changed)}; the suite passes.")]
