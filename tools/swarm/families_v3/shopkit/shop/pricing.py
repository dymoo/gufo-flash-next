"""Order pricing.

For each line: gross = unit price x qty; a bulk discount (config.BULK_PERCENT)
applies to lines with at least config.BULK_QUANTITY units; then the customer's
tier discount applies to what is left. A coupon applies to the sum of the
discounted lines and is allocated back to the lines in proportion to their net
amounts (Money.allocate), so tax can be computed per line. Tax is the line's
rate (a category's reduced rate, else the region's standard rate) applied to
the line's amount after the coupon, rounded per line. Shipping is by total
weight, free when the net goods reach config.FREE_SHIPPING_OVER, and untaxed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shop import config
from shop.money import Money, total


class PricingError(ValueError):
    pass


@dataclass
class LineQuote:
    sku: str
    qty: int
    gross: Money
    discount: Money      # bulk + tier
    coupon: Money        # this line's share of the coupon
    net: Money           # gross - discount - coupon
    tax_rate: str
    tax: Money


@dataclass
class Quote:
    lines: list[LineQuote] = field(default_factory=list)
    goods: Money = Money(0)       # sum of gross
    discount: Money = Money(0)    # sum of line discounts and coupon
    net: Money = Money(0)         # goods - discount
    tax: Money = Money(0)
    shipping: Money = Money(0)
    total: Money = Money(0)       # net + tax + shipping

    def as_dict(self) -> dict:
        """Plain values (minor units) for storage and reports."""
        return {"currency": self.total.currency,
                "goods": self.goods.minor, "discount": self.discount.minor, "net": self.net.minor,
                "tax": self.tax.minor, "shipping": self.shipping.minor, "total": self.total.minor,
                "lines": [{"sku": q.sku, "qty": q.qty, "net": q.net.minor, "tax_rate": q.tax_rate,
                           "tax": q.tax.minor} for q in self.lines]}


def tax_rate(category: str, region: str) -> str:
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
    kind, value = config.COUPONS[code]
    if kind == "percent":
        return discounted.percent(value)
    return Money(value, discounted.currency).min(discounted)


def quote(order, catalog: dict, customer) -> Quote:
    """Price an order for a customer. `catalog` maps sku -> Product."""
    if not order.lines:
        raise PricingError("order has no lines")
    currency = order.lines[0].unit_price.currency
    tier = config.TIER_DISCOUNTS[customer.tier]
    staged = []
    for line in order.lines:
        product = catalog.get(line.sku)
        if product is None:
            raise PricingError(f"unknown sku {line.sku!r}")
        gross = line.gross()
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
        lines.append(LineQuote(line.sku, line.qty, gross, discount, share, net, rate, net.percent(rate)))
    goods = total((q.gross for q in lines), currency)
    net = total((q.net for q in lines), currency)
    tax = total((q.tax for q in lines), currency)
    weight = sum(catalog[q.sku].weight_g * q.qty for q in lines)
    shipping = shipping_for(weight, net)
    return Quote(lines, goods, goods - net, net, tax, shipping, net + tax + shipping)
