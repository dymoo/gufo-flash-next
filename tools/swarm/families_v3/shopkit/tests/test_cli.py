from shop import cli
from tests.helpers import sample_service


def test_stock_line():
    service, _ = sample_service()
    assert cli.run(service, ["stock", "fd-210"]) == "FD-210: on hand 18, reserved 0, available 18"


def test_order_show():
    service, _ = sample_service()
    lines = cli.run(service, ["order", "show", "O-00003"]).splitlines()
    assert lines[0] == "Order O-00003 (placed) for Clara Weiss, created 2026-09-02"
    assert lines[-1].split() == ["total", "£124.48"]


def test_report_top():
    service, _ = sample_service()
    assert cli.run(service, ["report", "top", "1"]).splitlines()[-1] == "1 | Ada Brook | £156.78"


def test_errors():
    service, _ = sample_service()
    assert cli.run(service, ["nope"]).startswith("error: commands are")
    assert cli.run(service, ["customer", "C-9"]) == "error: unknown customer C-9"
