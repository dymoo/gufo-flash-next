import json

from shop import storage
from tests.helpers import sample_service


def test_round_trip_orders():
    service, _ = sample_service()
    loaded = storage.load_state(storage.dump_state(service))
    assert sorted(loaded.orders) == sorted(service.orders)
    for order_id, order in service.orders.items():
        assert storage.dump_order(loaded.orders[order_id]) == storage.dump_order(order)


def test_document_shape():
    service, _ = sample_service()
    doc = json.loads(storage.dump_state(service))
    assert doc["format"] == 1
    assert doc["products"][0] == {"sku": "BK-100", "name": "Atlas of Rivers", "category": "books",
                                  "price": "24.99", "currency": "GBP", "weight_g": 1200}
    assert doc["next_order"] == 6


def test_rejects_unknown_format():
    try:
        storage.load_state('{"format": 99}')
    except storage.StorageError:
        return
    raise AssertionError("loaded an unknown format")
