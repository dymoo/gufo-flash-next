"""Business configuration.

These values are owned by the finance and operations teams. Code reads them
through this module only; tests may monkeypatch them.
"""

DEFAULT_CURRENCY = "GBP"

# Standard tax rate per customer region, in percent (strings, used as Decimal).
TAX_RATES = {"UK": "20", "IE": "23", "DE": "19", "NL": "21", "US": "0"}

# Categories with a reduced rate that overrides the region's standard rate.
REDUCED_RATES = {"books": "0", "food": "5", "children": "5"}

# Percent off goods by customer tier, applied after the bulk discount.
TIER_DISCOUNTS = {"standard": "0", "silver": "5", "gold": "10"}

# A line with at least BULK_QUANTITY units gets BULK_PERCENT off that line.
BULK_QUANTITY = 10
BULK_PERCENT = "3"

# Coupons: code -> (kind, value). "percent" takes value percent off the
# discounted goods; "fixed" takes value minor units off, capped at the goods.
COUPONS = {"WELCOME10": ("percent", "10"), "FIVEOFF": ("fixed", 500), "SPRING15": ("percent", "15")}

# Shipping by total order weight: (max grams, price in minor units). Heavier
# orders pay HEAVY_SHIPPING. Orders whose net goods reach FREE_SHIPPING_OVER
# ship free.
SHIPPING_BANDS = [(1000, 395), (5000, 795), (20000, 1495)]
HEAVY_SHIPPING = 2495
FREE_SHIPPING_OVER = 7500

# Inventory publishes "low_stock" when available units drop below this.
LOW_STOCK_THRESHOLD = 5

# Version written by storage.dump_state.
STORAGE_FORMAT = 1
